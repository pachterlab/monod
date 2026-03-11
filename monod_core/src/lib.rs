/// monod_core — Rust implementation of CME model core math.
///
/// Exported PyO3 functions
/// -----------------------
/// eval_model_pss_2d(bio_model, p_log, limits, fixed_quad_t, quad_order) -> Vec<f64>
///     6 two-modality bio_models, seq_model="None".
///
/// eval_model_pss_protein_bursty(p_log, limits, fit_unspliced,
///                                protein_limit, min_fudge, max_fudge) -> Vec<f64>
///     ProteinBursty bio_model, seq_model="None".

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// ============================================================================
// Gauss-Legendre quadrature — matches numpy.polynomial.legendre.leggauss()
// ============================================================================

fn legendre_poly(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p_prev = 1.0_f64;
    let mut p_curr = x;
    for k in 1..n {
        let p_next =
            ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);
    (p_curr, dp)
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = (n + 1) / 2;
    let mut x = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];
    let pi = std::f64::consts::PI;

    for i in 0..m {
        let mut xi = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_poly(n, xi);
            let dx = p / dp;
            xi -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        x[i] = xi;
        x[n - 1 - i] = -xi;
        let (_, dp) = legendre_poly(n, xi);
        let wi = 2.0 / ((1.0 - xi * xi) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

// ============================================================================
// Caches: GL nodes, FFT planner, 2-D mesh
// ============================================================================

/// GL nodes/weights — keyed by order, computed once per process.
static GL_CACHE: OnceLock<Mutex<HashMap<usize, (Vec<f64>, Vec<f64>)>>> = OnceLock::new();

fn gauss_legendre_cached(n: usize) -> (Vec<f64>, Vec<f64>) {
    let cache = GL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(entry) = map.get(&n) {
        return entry.clone();
    }
    let result = gauss_legendre(n);
    map.insert(n, result.clone());
    result
}

/// Per-thread FftPlanner — plans are cached inside the planner between calls.
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// 2-D mesh cache — keyed by (l0, l1), rebuilt only when grid changes.
static MESH_CACHE_2D: OnceLock<Mutex<HashMap<(usize, usize), (Vec<Complex64>, Vec<Complex64>)>>> =
    OnceLock::new();

fn build_mesh_2d_cached(l0: usize, l1: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let cache = MESH_CACHE_2D.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(mesh) = map.get(&(l0, l1)) {
        return mesh.clone();
    }
    let mesh = build_mesh_2d(l0, l1);
    map.insert((l0, l1), mesh.clone());
    mesh
}

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn np_isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

// ============================================================================
// 2-D mesh builder
// ============================================================================

/// Returns (g0, g1) each of length n_grid = l0 * (l1/2+1), row-major.
fn build_mesh_2d(l0: usize, l1: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mx1 = l1 / 2 + 1;
    let pi = std::f64::consts::PI;

    let u0: Vec<Complex64> = (0..l0)
        .map(|k| {
            let a = -2.0 * pi * k as f64 / l0 as f64;
            Complex64::new(a.cos() - 1.0, a.sin())
        })
        .collect();
    let u1: Vec<Complex64> = (0..mx1)
        .map(|k| {
            let a = -2.0 * pi * k as f64 / l1 as f64;
            Complex64::new(a.cos() - 1.0, a.sin())
        })
        .collect();

    let mut g0 = Vec::with_capacity(l0 * mx1);
    let mut g1 = Vec::with_capacity(l0 * mx1);
    for i in 0..l0 {
        for j in 0..mx1 {
            g0.push(u0[i]);
            g1.push(u1[j]);
        }
    }
    (g0, g1)
}

/// Returns (g0, g1, g2) each of length n_grid = mx0*mx1*mx2, row-major.
/// mx = [mx0, mx1, mx2] (already reduced: mx2 = l2/2+1, possibly mx0=1).
fn build_mesh_3d(
    mx: &[usize; 3],
    limits: &[usize; 3],
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let pi = std::f64::consts::PI;
    let u: Vec<Vec<Complex64>> = (0..3)
        .map(|s| {
            (0..mx[s])
                .map(|k| {
                    let a = -2.0 * pi * k as f64 / limits[s] as f64;
                    Complex64::new(a.cos() - 1.0, a.sin())
                })
                .collect()
        })
        .collect();

    let n = mx[0] * mx[1] * mx[2];
    let mut g0 = Vec::with_capacity(n);
    let mut g1 = Vec::with_capacity(n);
    let mut g2 = Vec::with_capacity(n);
    for i in 0..mx[0] {
        for j in 0..mx[1] {
            for k in 0..mx[2] {
                g0.push(u[0][i]);
                g1.push(u[1][j]);
                g2.push(u[2][k]);
            }
        }
    }
    (g0, g1, g2)
}

// ============================================================================
// Per-model log-PGF — 2D analytical models
// ============================================================================

fn pgf_constitutive(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let (beta, gamma) = (p[0], p[1]);
    g0.iter().zip(g1).map(|(&a, &b)| a / beta + b / gamma).collect()
}

fn pgf_extrinsic(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let (alpha, beta, gamma) = (p[0], p[1], p[2]);
    let one = Complex64::new(1.0, 0.0);
    g0.iter()
        .zip(g1)
        .map(|(&a, &b)| (one - a / beta - b / gamma).ln() * (-alpha))
        .collect()
}

fn pgf_delay(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let (b, beta, tauinv) = (p[0], p[1], p[2]);
    let tau = 1.0 / tauinv;
    let exp_bt = (-beta * tau).exp();
    let one = Complex64::new(1.0, 0.0);
    g0.iter()
        .zip(g1)
        .map(|(&g0k, &g1k)| {
            let u = g1k + (g0k - g1k) * exp_bt;
            let term1 = (one - u * b).ln() * (-1.0 / beta);
            let ratio = (u * b - one) / (g0k * b - one);
            let term2 = ratio.ln() / (beta * (one - g1k * b));
            let term3 = g1k * b * tau / (one - g1k * b);
            term1 + term2 + term3
        })
        .collect()
}

fn pgf_delayed_splicing(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let (b, tauinv, gamma) = (p[0], p[1], p[2]);
    let tau = 1.0 / tauinv;
    let one = Complex64::new(1.0, 0.0);
    g0.iter()
        .zip(g1)
        .map(|(&a, &b_g)| {
            let term1 = a * b * tau / (one - a * b);
            let term2 = (one - b_g * b).ln() * (-1.0 / gamma);
            term1 + term2
        })
        .collect()
}

// ============================================================================
// Per-model log-PGF — quadrature models (parallelised over grid points)
// ============================================================================

fn pgf_bursty(
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    t: f64,
    quad_order: usize,
) -> Vec<Complex64> {
    let (b, beta, gamma) = (p[0], p[1], p[2]);
    let one = Complex64::new(1.0, 0.0);
    let (xi, wi) = gauss_legendre_cached(quad_order);
    let (t_half, t_mid) = (t / 2.0, t / 2.0);
    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    let (c1, c2): (Vec<Complex64>, Vec<Complex64>) = if !close {
        let c2: Vec<_> = g1.iter().map(|&v| v * f_factor).collect();
        let c1: Vec<_> = g0.iter().zip(&c2).map(|(&v, &c)| v - c).collect();
        (c1, c2)
    } else {
        (g0.to_vec(), g1.to_vec()) // placeholders; close-case uses g0[k]/g1[k] directly
    };

    // Precompute quadrature abscissae and exponentials — same for every grid point.
    let xs: Vec<f64> = xi.iter().map(|&xq| t_mid + t_half * xq).collect();
    let eb_vals: Vec<f64> = xs.iter().map(|&x| (-beta * x).exp()).collect();
    let eg_vals: Vec<f64> = xs.iter().map(|&x| (-gamma * x).exp()).collect();
    let ws: Vec<f64> = wi.iter().map(|&wq| wq * t_half).collect();

    (0..g0.len())
        .into_par_iter()
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            for q in 0..xs.len() {
                let u = if close {
                    (g0[k] * eb_vals[q] + g1[k] * (xs[q] * beta * eg_vals[q])) * b
                } else {
                    (c1[k] * eb_vals[q] + c2[k] * eg_vals[q]) * b
                };
                acc += (u / (one - u)) * ws[q];
            }
            acc
        })
        .collect()
}

fn pgf_cir(
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    t: f64,
    quad_order: usize,
) -> Vec<Complex64> {
    let (b, beta, gamma) = (p[0], p[1], p[2]);
    let one = Complex64::new(1.0, 0.0);
    let four = Complex64::new(4.0, 0.0);
    let (xi, wi) = gauss_legendre_cached(quad_order);
    let (t_half, t_mid) = (t / 2.0, t / 2.0);
    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    let (c1, c2): (Vec<Complex64>, Vec<Complex64>) = if !close {
        let c2: Vec<_> = g1.iter().map(|&v| v * f_factor).collect();
        let c1: Vec<_> = g0.iter().zip(&c2).map(|(&v, &c)| v - c).collect();
        (c1, c2)
    } else {
        (g0.to_vec(), vec![Complex64::new(0.0, 0.0); g0.len()])
    };

    // Precompute quadrature abscissae and exponentials — same for every grid point.
    let xs: Vec<f64> = xi.iter().map(|&xq| t_mid + t_half * xq).collect();
    let eb_vals: Vec<f64> = xs.iter().map(|&x| (-beta * x).exp()).collect();
    let eg_vals: Vec<f64> = xs.iter().map(|&x| (-gamma * x).exp()).collect();
    let ws: Vec<f64> = wi.iter().map(|&wq| wq * t_half).collect();

    let mut gf: Vec<Complex64> = (0..g0.len())
        .into_par_iter()
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            for q in 0..xs.len() {
                let u = if close {
                    (c1[k] * eb_vals[q] + g1[k] * (xs[q] * beta * eg_vals[q])) * b
                } else {
                    (c1[k] * eb_vals[q] + c2[k] * eg_vals[q]) * b
                };
                let integrand = one - (one - four * u).sqrt();
                acc += integrand * ws[q];
            }
            acc
        })
        .collect();
    gf.iter_mut().for_each(|v| *v /= 2.0);
    gf
}

// ============================================================================
// ProteinBursty log-PGF — f32 complex RK4 ODE, matches Python np.complex64
// ============================================================================

/// Evaluate the ODE RHS for the 3-species protein model (f64 complex scalars).
///
/// Python uses np.complex64 initial conditions but upcasts to complex128 during
/// RK4 (because dt is np.float64 and np.float64 × complex64 → complex128).
/// We match by casting initial conditions from f64 to f32 and back (to simulate
/// the np.array(g, dtype=np.complex64) truncation), then running the ODE in f64.
#[inline(always)]
fn protein_ode(
    u0: Complex64,
    u1: Complex64,
    u2: Complex64,
    beta: f64,
    gamma: f64,
    k_p: f64,
    gamma_p: f64,
) -> (Complex64, Complex64, Complex64) {
    let du0 = (u1 - u0) * beta;
    let du1 = u1 * (-gamma) + u2 * k_p * (u1 + Complex64::new(1.0, 0.0));
    let du2 = u2 * (-gamma_p);
    (du0, du1, du2)
}

#[inline(always)]
fn rk4_step(
    u0: Complex64,
    u1: Complex64,
    u2: Complex64,
    dt: f64,
    beta: f64,
    gamma: f64,
    k_p: f64,
    gamma_p: f64,
) -> (Complex64, Complex64, Complex64) {
    let (j1_0, j1_1, j1_2) = protein_ode(u0, u1, u2, beta, gamma, k_p, gamma_p);
    let h = dt / 2.0;
    let (j2_0, j2_1, j2_2) = protein_ode(
        u0 + j1_0 * h, u1 + j1_1 * h, u2 + j1_2 * h,
        beta, gamma, k_p, gamma_p,
    );
    let (j3_0, j3_1, j3_2) = protein_ode(
        u0 + j2_0 * h, u1 + j2_1 * h, u2 + j2_2 * h,
        beta, gamma, k_p, gamma_p,
    );
    let (j4_0, j4_1, j4_2) = protein_ode(
        u0 + j3_0 * dt, u1 + j3_1 * dt, u2 + j3_2 * dt,
        beta, gamma, k_p, gamma_p,
    );
    let s = dt / 6.0;
    (
        u0 + (j1_0 + j2_0 * 2.0 + j3_0 * 2.0 + j4_0) * s,
        u1 + (j1_1 + j2_1 * 2.0 + j3_1 * 2.0 + j4_1) * s,
        u2 + (j1_2 + j2_2 * 2.0 + j3_2 * 2.0 + j4_2) * s,
    )
}

/// Compute the ProteinBursty log-PGF over all grid points (parallelised).
/// Initial conditions are truncated to f32 precision (matching Python's
/// np.array(g, dtype=np.complex64)), then the ODE runs in f64 (matching
/// how numpy upcasts np.float64 parameters × complex64 → complex128).
fn protein_pgf(
    g0: &[Complex64],
    g1: &[Complex64],
    g2: &[Complex64],
    p: &[f64],
    min_fudge: f64,
    max_fudge: f64,
) -> Vec<Complex64> {
    let n_grid = g0.len();
    let (b, beta, gamma, k_p, gamma_p) = (p[0], p[1], p[2], p[3], p[4]);

    let dt = p.iter().map(|&v| 1.0 / v).fold(f64::INFINITY, f64::min) * min_fudge;
    let t_max = p.iter().map(|&v| 1.0 / v).fold(0.0_f64, f64::max) * max_fudge;
    let num_tsteps = (t_max / dt).ceil() as usize;
    let one = Complex64::new(1.0, 0.0);

    (0..n_grid)
        .into_par_iter()
        .map(|k| {
            // Truncate to f32 and back to match Python's dtype=np.complex64 cast
            let mut u0 = Complex64::new(g0[k].re as f32 as f64, g0[k].im as f32 as f64);
            let mut u1 = Complex64::new(g1[k].re as f32 as f64, g1[k].im as f32 as f64);
            let mut u2 = Complex64::new(g2[k].re as f32 as f64, g2[k].im as f32 as f64);

            let mut phi = u0 * b / (one - u0 * b) * (dt / 2.0);

            for _ in 0..num_tsteps {
                (u0, u1, u2) = rk4_step(u0, u1, u2, dt, beta, gamma, k_p, gamma_p);
                phi += u0 * b / (one - u0 * b) * dt;
            }
            while u0.norm() > 1e-3 {
                (u0, u1, u2) = rk4_step(u0, u1, u2, dt, beta, gamma, k_p, gamma_p);
                phi += u0 * b / (one - u0 * b) * dt;
            }
            (u0, u1, u2) = rk4_step(u0, u1, u2, dt, beta, gamma, k_p, gamma_p);
            phi += u0 * b / (one - u0 * b) * (dt / 2.0);
            phi
        })
        .collect()
}

// ============================================================================
// irfftn helpers — 1-D building blocks
// ============================================================================

/// Apply IFFT of length `n` in-place to `buf`, using the given planner.
fn ifft_inplace(buf: &mut Vec<FftComplex<f64>>, n: usize, planner: &mut FftPlanner<f64>) {
    let fft = planner.plan_fft_inverse(n);
    let mut scratch = vec![FftComplex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
    fft.process_with_scratch(buf, &mut scratch);
}

/// 1-D irfft: fill Hermitian buffer of length `n` from `mx = n/2+1` inputs,
/// apply IFFT, return real part.
fn irfft_row(
    input: &[FftComplex<f64>],
    n: usize,
    buf: &mut Vec<FftComplex<f64>>,
    planner: &mut FftPlanner<f64>,
) -> Vec<f64> {
    let mx = n / 2 + 1;
    for k in 0..mx {
        buf[k] = input[k];
    }
    for k in 1..mx {
        let nk = n - k;
        if nk >= mx {
            buf[nk] = FftComplex::new(input[k].re, -input[k].im);
        }
    }
    ifft_inplace(buf, n, planner);
    buf[..n].iter().map(|c| c.re).collect()
}

// ============================================================================
// 2-D irfftn
// ============================================================================

/// irfftn for shape [n0, mx1=n1/2+1] → [n0, n1].
/// Algorithm (verified vs scipy):
///   1. ifft along axis 0 for each column j = 0..mx1
///   2. irfft along axis 1 for each row i = 0..n0
fn irfftn_2d(input: &[Complex64], n0: usize, n1: usize) -> Vec<f64> {
    FFT_PLANNER.with(|planner_cell| {
        let mut planner = planner_cell.borrow_mut();
        let mx1 = n1 / 2 + 1;

        // Step 1: ifft along axis 0
        let fft_n0 = planner.plan_fft_inverse(n0);
        let mut scratch_n0 = vec![FftComplex::new(0.0, 0.0); fft_n0.get_inplace_scratch_len()];
        let mut mid = vec![FftComplex::new(0.0, 0.0); n0 * mx1];
        let mut col_buf = vec![FftComplex::new(0.0, 0.0); n0];

        for j in 0..mx1 {
            for i in 0..n0 {
                let c = input[i * mx1 + j];
                col_buf[i] = FftComplex::new(c.re, c.im);
            }
            fft_n0.process_with_scratch(&mut col_buf, &mut scratch_n0);
            for i in 0..n0 {
                mid[i * mx1 + j] = col_buf[i];
            }
        }

        // Step 2: irfft along axis 1
        let mut row_buf = vec![FftComplex::new(0.0, 0.0); n1];
        let mut row_in = vec![FftComplex::new(0.0, 0.0); mx1];
        let mut result = vec![0.0_f64; n0 * n1];
        for i in 0..n0 {
            for k in 0..mx1 {
                row_in[k] = mid[i * mx1 + k];
            }
            let real_row = irfft_row(&row_in, n1, &mut row_buf, &mut planner);
            result[i * n1..(i + 1) * n1].copy_from_slice(&real_row);
        }
        result
    })
}

// ============================================================================
// 3-D irfftn
// ============================================================================

/// irfftn for shape [mx0, n1, mx2_in] → [n0, n1, n2].
///
/// mx0 is the actual size of axis 0 in `input` (may be < n0 for zero-padding).
/// mx2_in is the actual rfft half-size of axis 2 in `input` (may be < n2/2+1
/// when protein coarse-graining is active).
///
/// Matches scipy.fft.irfftn(gf.reshape(mx), s=(n0,n1,n2)).
///
/// Algorithm (verified vs scipy):
///   1. ifft along axis 0 (zero-pad from mx0 to n0)
///   2. ifft along axis 1
///   3. irfft along axis 2 (zero-pad rfft coefficients from mx2_in to n2/2+1)
fn irfftn_3d(input: &[Complex64], mx0: usize, mx2_in: usize, n0: usize, n1: usize, n2: usize) -> Vec<f64> {
    FFT_PLANNER.with(|planner_cell| { irfftn_3d_inner(input, mx0, mx2_in, n0, n1, n2, &mut planner_cell.borrow_mut()) })
}

fn irfftn_3d_inner(input: &[Complex64], mx0: usize, mx2_in: usize, n0: usize, n1: usize, n2: usize, mut planner: &mut FftPlanner<f64>) -> Vec<f64> {
    let mx2_out = n2 / 2 + 1;

    // ---- Step 1: ifft along axis 0 (zero-pad mx0 → n0) ----
    let fft_n0 = planner.plan_fft_inverse(n0);
    let mut scratch_n0 = vec![FftComplex::new(0.0, 0.0); fft_n0.get_inplace_scratch_len()];
    let mut buf0 = vec![FftComplex::new(0.0, 0.0); n0 * n1 * mx2_in];
    let mut col0 = vec![FftComplex::new(0.0, 0.0); n0];

    for j1 in 0..n1 {
        for j2 in 0..mx2_in {
            for i in 0..mx0 {
                let c = input[i * n1 * mx2_in + j1 * mx2_in + j2];
                col0[i] = FftComplex::new(c.re, c.im);
            }
            for i in mx0..n0 {
                col0[i] = FftComplex::new(0.0, 0.0);
            }
            fft_n0.process_with_scratch(&mut col0, &mut scratch_n0);
            for i in 0..n0 {
                buf0[i * n1 * mx2_in + j1 * mx2_in + j2] = col0[i];
            }
        }
    }

    // ---- Step 2: ifft along axis 1 ----
    let fft_n1 = planner.plan_fft_inverse(n1);
    let mut scratch_n1 = vec![FftComplex::new(0.0, 0.0); fft_n1.get_inplace_scratch_len()];
    let mut buf1 = vec![FftComplex::new(0.0, 0.0); n0 * n1 * mx2_in];
    let mut col1 = vec![FftComplex::new(0.0, 0.0); n1];

    for i in 0..n0 {
        for j2 in 0..mx2_in {
            for j1 in 0..n1 {
                col1[j1] = buf0[i * n1 * mx2_in + j1 * mx2_in + j2];
            }
            fft_n1.process_with_scratch(&mut col1, &mut scratch_n1);
            for j1 in 0..n1 {
                buf1[i * n1 * mx2_in + j1 * mx2_in + j2] = col1[j1];
            }
        }
    }

    // ---- Step 3: irfft along axis 2 (zero-pad mx2_in → mx2_out) ----
    let mut row_buf = vec![FftComplex::new(0.0, 0.0); n2];
    let mut row_in = vec![FftComplex::new(0.0, 0.0); mx2_out];
    let mut result = vec![0.0_f64; n0 * n1 * n2];
    let copy_len = mx2_in.min(mx2_out);

    for i in 0..n0 {
        for j1 in 0..n1 {
            for j2 in 0..mx2_out { row_in[j2] = FftComplex::new(0.0, 0.0); }
            for j2 in 0..copy_len {
                row_in[j2] = buf1[i * n1 * mx2_in + j1 * mx2_in + j2];
            }
            let real_row = irfft_row(&row_in, n2, &mut row_buf, &mut planner);
            let start = i * n1 * n2 + j1 * n2;
            result[start..start + n2].copy_from_slice(&real_row);
        }
    }
    result
}

// ============================================================================
// eval_model_pss — 2-D models
// ============================================================================

#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, fixed_quad_t, quad_order, samp_log=None))]
fn eval_model_pss_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let (l0, l1) = (limits[0], limits[1]);
    let (base_g0, base_g1) = build_mesh_2d_cached(l0, l1);

    // Apply Poisson technical noise: g → exp(λ·g) − 1 per modality.
    let (g0, g1): (Vec<Complex64>, Vec<Complex64>) = if let Some(ref samp) = samp_log {
        let lam0 = 10.0_f64.powf(samp[0]);
        let lam1 = 10.0_f64.powf(samp[1]);
        let one = Complex64::new(1.0, 0.0);
        let g0t = base_g0.iter().map(|&z| (lam0 * z).exp() - one).collect();
        let g1t = base_g1.iter().map(|&z| (lam1 * z).exp() - one).collect();
        (g0t, g1t)
    } else {
        (base_g0, base_g1)
    };

    let gf_log: Vec<Complex64> = match bio_model {
        "Constitutive" => pgf_constitutive(&g0, &g1, &p),
        "Extrinsic" => pgf_extrinsic(&g0, &g1, &p),
        "Delay" => pgf_delay(&g0, &g1, &p),
        "DelayedSplicing" => pgf_delayed_splicing(&g0, &g1, &p),
        "Bursty" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_bursty(&g0, &g1, &p, t, quad_order)
        }
        "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_cir(&g0, &g1, &p, t, quad_order)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown bio_model for eval_model_pss_2d: {bio_model}"
            )))
        }
    };

    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();
    let pss_raw = irfftn_2d(&gf, l0, l1);
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    Ok(pss_raw.iter().map(|x| x.abs() / abs_sum).collect())
}

// ============================================================================
// eval_model_pss — ProteinBursty (3-D)
// ============================================================================

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eval_model_pss_protein_bursty(
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fit_unspliced: bool,
    protein_limit: f64,      // use f64::INFINITY for no coarse-graining
    min_fudge: f64,
    max_fudge: f64,
) -> PyResult<Vec<f64>> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let mut mx = [limits[0], limits[1], limits[2]];

    // Protein axis coarse-graining (mirrors Python logic)
    let scale = (mx[2] as f64 / protein_limit).floor() as usize + 1;
    mx[2] = (mx[2] + scale - 1) / scale;
    if !fit_unspliced {
        mx[0] = 1;
    }
    mx[2] = mx[2] / 2 + 1; // rfft half

    let lims = [limits[0], limits[1], limits[2]];
    let (g0, g1, g2) = build_mesh_3d(&mx, &lims);

    let gf_log = protein_pgf(&g0, &g1, &g2, &p, min_fudge, max_fudge);
    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();

    let pss_raw = irfftn_3d(&gf, mx[0], mx[2], lims[0], lims[1], lims[2]);
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    Ok(pss_raw.iter().map(|x| x.abs() / abs_sum).collect())
}

// ============================================================================
// protein_bursty_pgf — return gf array to Python for scipy irfftn
// ============================================================================

/// Compute the protein-bursty generating function on a coarse grid, returning
/// the complex gf values as (re, im, mx_shape) for Python to irfftn with scipy.
///
/// This gives scipy-exact PSS values (no floating-point drift vs the pure-Python
/// baseline) while still parallelising the expensive ODE integration in Rust.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn protein_bursty_pgf(
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let mut mx = [limits[0], limits[1], limits[2]];

    let scale = (mx[2] as f64 / protein_limit).floor() as usize + 1;
    mx[2] = (mx[2] + scale - 1) / scale;
    if !fit_unspliced {
        mx[0] = 1;
    }
    mx[2] = mx[2] / 2 + 1;

    let lims = [limits[0], limits[1], limits[2]];
    let (g0, g1, g2) = build_mesh_3d(&mx, &lims);

    let gf_log = protein_pgf(&g0, &g1, &g2, &p, min_fudge, max_fudge);
    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();

    let re: Vec<f64> = gf.iter().map(|z| z.re).collect();
    let im: Vec<f64> = gf.iter().map(|z| z.im).collect();
    let mx_list = vec![mx[0], mx[1], mx[2]];

    Ok((re, im, mx_list))
}

// ============================================================================
// PyO3 module
// ============================================================================

#[pymodule]
fn monod_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_model_pss_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_model_pss_protein_bursty, m)?)?;
    m.add_function(wrap_pyfunction!(protein_bursty_pgf, m)?)?;
    Ok(())
}
