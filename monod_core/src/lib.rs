/// monod_core — Rust implementation of CME model core math.
///
/// Exported PyO3 functions
/// -----------------------
/// eval_model_pss_2d(bio_model, p_log, limits, fixed_quad_t, quad_order, samp_log=None) -> Vec<f64>
///     6 two-modality bio_models, seq_model="None" or seq_model="Poisson".
///
/// eval_model_pss_protein_bursty(p_log, limits, fit_unspliced,
///                                protein_limit, min_fudge, max_fudge) -> Vec<f64>
///     ProteinBursty bio_model, seq_model="None".
///
/// Optimizations over prior version:
///   - Type unification: Complex64 (num_complex 0.4) == rustfft's internal Complex<f64>.
///     No field-by-field copies between FftComplex and Complex64.
///   - Thread-local scratch + row buffers: eliminates per-task heap allocation in
///     parallel FFT loops; buffers grow on demand and are reused across calls.
///   - Arc-wrapped cache entries: mesh and GL caches return Arc pointers, not Vec clones.
///   - Parallel exp and normalization: par_iter for gf.exp() and pss normalization.
///   - Eliminated mid/buf0/buf1 allocations in irfftn_{2,3}d: after column IFFTs,
///     row data is gathered directly with stride access, saving 1 (2D) or 2 (3D) allocations.
///   - Shared protein_bursty_core helper: eliminates duplicated logic between
///     eval_model_pss_protein_bursty and protein_bursty_pgf.

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ============================================================================
// Thread-local FFT planner, scratch buffer, and row buffer
// ============================================================================

// Per-thread FftPlanner — plans are cached inside the planner between calls.
// rustfft 6.x uses num_complex 0.4, same as our num-complex dependency,
// so Complex64 is the same type as rustfft's internal Complex<f64>.
thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    /// Reused scratch buffer for FFT scratch space — grown as needed, never shrunk.
    static SCRATCH_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Reused row buffer for irfft (Hermitian expansion + output) — grown as needed.
    static ROW_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
}

/// Get a cached FFT plan for inverse FFT of length `n`.
#[inline]
fn plan_ifft(n: usize) -> Arc<dyn rustfft::Fft<f64>> {
    FFT_PLANNER.with(|cell| cell.borrow_mut().plan_fft_inverse(n))
}

/// Run an in-place inverse FFT using the thread-local scratch buffer.
#[inline]
fn ifft_inplace(buf: &mut [Complex64], fft: &Arc<dyn rustfft::Fft<f64>>) {
    SCRATCH_BUF.with(|sc| {
        let mut scratch = sc.borrow_mut();
        let len = fft.get_inplace_scratch_len();
        if scratch.len() < len {
            scratch.resize(len, Complex64::new(0.0, 0.0));
        }
        fft.process_with_scratch(buf, &mut scratch[..len]);
    });
}

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
// Caches: GL nodes, 2-D mesh, Poisson-transformed 2-D mesh
// ============================================================================

/// GL nodes/weights — Arc-wrapped, keyed by order, computed once per process.
static GL_CACHE: OnceLock<Mutex<HashMap<usize, Arc<(Vec<f64>, Vec<f64>)>>>> = OnceLock::new();

fn gauss_legendre_cached(n: usize) -> Arc<(Vec<f64>, Vec<f64>)> {
    let cache = GL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(entry) = map.get(&n) {
        return Arc::clone(entry);
    }
    let result = Arc::new(gauss_legendre(n));
    map.insert(n, Arc::clone(&result));
    result
}

/// 2-D base mesh cache — Arc-wrapped, keyed by (l0, l1).
static MESH_CACHE_2D: OnceLock<Mutex<HashMap<(usize, usize), Arc<(Vec<Complex64>, Vec<Complex64>)>>>> =
    OnceLock::new();

fn build_mesh_2d_cached(l0: usize, l1: usize) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    let cache = MESH_CACHE_2D.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(mesh) = map.get(&(l0, l1)) {
        return Arc::clone(mesh);
    }
    let mesh = Arc::new(build_mesh_2d(l0, l1));
    map.insert((l0, l1), Arc::clone(&mesh));
    mesh
}

/// Poisson-transformed 2-D mesh cache — Arc-wrapped, keyed by (l0, l1, lam0_bits, lam1_bits).
static MESH_CACHE_POISSON: OnceLock<Mutex<HashMap<(usize, usize, u64, u64), Arc<(Vec<Complex64>, Vec<Complex64>)>>>> =
    OnceLock::new();

fn build_mesh_2d_poisson_cached(
    l0: usize,
    l1: usize,
    lam0: f64,
    lam1: f64,
) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    let key = (l0, l1, lam0.to_bits(), lam1.to_bits());
    let cache = MESH_CACHE_POISSON.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(mesh) = map.get(&key) {
        return Arc::clone(mesh);
    }
    let base = build_mesh_2d_cached(l0, l1);
    let one = Complex64::new(1.0, 0.0);
    let g0t: Vec<Complex64> = base.0.iter().map(|&z| (lam0 * z).exp() - one).collect();
    let g1t: Vec<Complex64> = base.1.iter().map(|&z| (lam1 * z).exp() - one).collect();
    let mesh = Arc::new((g0t, g1t));
    map.insert(key, Arc::clone(&mesh));
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
    let gl = gauss_legendre_cached(quad_order);
    let (t_half, t_mid) = (t / 2.0, t / 2.0);
    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    // Precompute quadrature abscissae and exponentials — same for every grid point.
    let xs: Vec<f64> = gl.0.iter().map(|&xq| t_mid + t_half * xq).collect();
    let eb_vals: Vec<f64> = xs.iter().map(|&x| (-beta * x).exp()).collect();
    let eg_vals: Vec<f64> = xs.iter().map(|&x| (-gamma * x).exp()).collect();
    let ws: Vec<f64> = gl.1.iter().map(|&wq| wq * t_half).collect();
    let nq = xs.len();

    (0..g0.len())
        .into_par_iter()
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            if close {
                for q in 0..nq {
                    let u = (g0[k] * eb_vals[q] + g1[k] * (xs[q] * beta * eg_vals[q])) * b;
                    acc += (u / (one - u)) * ws[q];
                }
            } else {
                let c2k = g1[k] * f_factor;
                let c1k = g0[k] - c2k;
                for q in 0..nq {
                    let u = (c1k * eb_vals[q] + c2k * eg_vals[q]) * b;
                    acc += (u / (one - u)) * ws[q];
                }
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
    let gl = gauss_legendre_cached(quad_order);
    let (t_half, t_mid) = (t / 2.0, t / 2.0);
    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    // Precompute quadrature abscissae and exponentials — same for every grid point.
    let xs: Vec<f64> = gl.0.iter().map(|&xq| t_mid + t_half * xq).collect();
    let eb_vals: Vec<f64> = xs.iter().map(|&x| (-beta * x).exp()).collect();
    let eg_vals: Vec<f64> = xs.iter().map(|&x| (-gamma * x).exp()).collect();
    let ws: Vec<f64> = gl.1.iter().map(|&wq| wq * t_half).collect();
    let nq = xs.len();

    let mut gf: Vec<Complex64> = (0..g0.len())
        .into_par_iter()
        .map(|k| {
            let mut acc = Complex64::new(0.0, 0.0);
            if close {
                for q in 0..nq {
                    let u = (g0[k] * eb_vals[q] + g1[k] * (xs[q] * beta * eg_vals[q])) * b;
                    let integrand = one - (one - four * u).sqrt();
                    acc += integrand * ws[q];
                }
            } else {
                let c2k = g1[k] * f_factor;
                let c1k = g0[k] - c2k;
                for q in 0..nq {
                    let u = (c1k * eb_vals[q] + c2k * eg_vals[q]) * b;
                    let integrand = one - (one - four * u).sqrt();
                    acc += integrand * ws[q];
                }
            }
            acc
        })
        .collect();
    gf.par_iter_mut().for_each(|v| *v /= 2.0);
    gf
}

// ============================================================================
// ProteinBursty log-PGF — f32 complex RK4 ODE, matches Python np.complex64
// ============================================================================

/// Evaluate the ODE RHS for the 3-species protein model.
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

/// Compute the ProteinBursty log-PGF over all grid points.
///
/// Uses a three-phase approach to match Python's global-max termination:
///   Phase 1 — fixed steps, fully parallel per point (no per-step barrier).
///   Phase 2 — variable steps with global-max termination (matches Python's
///             `while np.max(np.abs(u_tilde[0])) >= 1e-3`); parallel per step.
///   Phase 3 — final half-step, parallel per point.
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

    // Phase 1: fixed trajectory per point, fully parallel (no sync per step).
    let mut states: Vec<(Complex64, Complex64, Complex64, Complex64)> =
        (0..n_grid)
            .into_par_iter()
            .map(|k| {
                // Truncate to f32 and back to match Python's dtype=np.complex64 cast.
                let mut u0 = Complex64::new(g0[k].re as f32 as f64, g0[k].im as f32 as f64);
                let mut u1 = Complex64::new(g1[k].re as f32 as f64, g1[k].im as f32 as f64);
                let mut u2 = Complex64::new(g2[k].re as f32 as f64, g2[k].im as f32 as f64);
                let mut phi = u0 * b / (one - u0 * b) * (dt / 2.0);
                for _ in 0..num_tsteps {
                    let (nu0, nu1, nu2) = rk4_step(u0, u1, u2, dt, beta, gamma, k_p, gamma_p);
                    u0 = nu0; u1 = nu1; u2 = nu2;
                    phi += u0 * b / (one - u0 * b) * dt;
                }
                (u0, u1, u2, phi)
            })
            .collect();

    // Phase 2: global-max termination — matches Python's
    // `while np.max(np.abs(u_tilde[0])) >= 1e-3`.
    loop {
        let max_norm = states.par_iter().map(|s| s.0.norm()).reduce(|| 0.0_f64, f64::max);
        if max_norm < 1e-3 {
            break;
        }
        states.par_iter_mut().for_each(|s| {
            let (nu0, nu1, nu2) = rk4_step(s.0, s.1, s.2, dt, beta, gamma, k_p, gamma_p);
            s.0 = nu0; s.1 = nu1; s.2 = nu2;
            s.3 += nu0 * b / (one - nu0 * b) * dt;
        });
    }

    // Phase 3: final half-step, parallel.
    states.par_iter_mut().for_each(|s| {
        let (nu0, _, _) = rk4_step(s.0, s.1, s.2, dt, beta, gamma, k_p, gamma_p);
        s.3 += nu0 * b / (one - nu0 * b) * (dt / 2.0);
    });

    states.into_par_iter().map(|s| s.3).collect()
}

// ============================================================================
// 2-D irfftn — parallel column IFFTs + parallel row irffts, no mid buffer
// ============================================================================

/// irfftn for shape [n0, mx1=n1/2+1] → [n0, n1].
///
/// Algorithm (verified vs scipy):
///   1. Transpose input to column-major; parallel IFFT of length n0 per column j.
///   2. For each row i: gather row i from col_buf with stride n0, expand Hermitian
///      conjugate, irfft — no intermediate `mid` buffer allocation.
///
/// Each parallel task uses its own thread-local FftPlanner, scratch buffer, and row buffer.
fn irfftn_2d(input: &[Complex64], n0: usize, n1: usize) -> Vec<f64> {
    let mx1 = n1 / 2 + 1;
    let zero = Complex64::new(0.0, 0.0);

    // Step 1: Transpose to column-major, then parallel IFFT along axis 0.
    let mut col_buf = vec![zero; mx1 * n0];
    for i in 0..n0 {
        for j in 0..mx1 {
            col_buf[j * n0 + i] = input[i * mx1 + j];
        }
    }
    col_buf.par_chunks_mut(n0).for_each(|col| {
        let fft = plan_ifft(n0);
        ifft_inplace(col, &fft);
    });

    // Step 2: Parallel irfft along axis 1.
    // Gather each row i from col_buf with stride n0 — no mid allocation needed.
    let mut result = vec![0.0_f64; n0 * n1];
    result.par_chunks_mut(n1).enumerate().for_each(|(i, row_out)| {
        let fft = plan_ifft(n1);
        ROW_BUF.with(|rb| {
            let mut buf = rb.borrow_mut();
            if buf.len() < n1 {
                buf.resize(n1, zero);
            }
            // Zero, then gather positive frequencies from col_buf (stride n0).
            buf[..n1].fill(zero);
            for k in 0..mx1 {
                buf[k] = col_buf[k * n0 + i];
            }
            // Fill Hermitian conjugate for negative frequencies.
            for k in 1..mx1 {
                let nk = n1 - k;
                if nk >= mx1 {
                    buf[nk] = buf[k].conj();
                }
            }
            ifft_inplace(&mut buf[..n1], &fft);
            for (out, c) in row_out.iter_mut().zip(buf[..n1].iter()) {
                *out = c.re;
            }
        });
    });
    result
}

// ============================================================================
// 3-D irfftn — parallel all three axes, no buf0/buf1 intermediate allocations
// ============================================================================

/// irfftn for shape [mx0, n1, mx2_in] → [n0, n1, n2].
///
/// Algorithm (verified vs scipy):
///   1. Transpose to column-major; parallel IFFT along axis 0 (zero-pad mx0 → n0).
///   2. Gather columns (i, j2) directly from col0_buf (no buf0 allocation);
///      parallel IFFT along axis 1.
///   3. For each row (i, j1): gather with stride n1 from col1_buf (no buf1 allocation),
///      expand Hermitian conjugate (zero-pad mx2_in → mx2_out), irfft.
///
/// Each parallel task uses its own thread-local FftPlanner, scratch buffer, and row buffer.
fn irfftn_3d(input: &[Complex64], mx0: usize, mx2_in: usize, n0: usize, n1: usize, n2: usize) -> Vec<f64> {
    let mx2_out = n2 / 2 + 1;
    let zero = Complex64::new(0.0, 0.0);

    // Step 1: Transpose to column-major, parallel IFFT along axis 0 (zero-pad mx0 → n0).
    // col0_buf layout: [c * n0 + i] where c = j1 * mx2_in + j2.
    let n_cols_0 = n1 * mx2_in;
    let mut col0_buf = vec![zero; n_cols_0 * n0];
    for j1 in 0..n1 {
        for j2 in 0..mx2_in {
            let c = j1 * mx2_in + j2;
            for i in 0..mx0 {
                col0_buf[c * n0 + i] = input[i * n1 * mx2_in + j1 * mx2_in + j2];
            }
            // Indices mx0..n0 are already zero (zero-padding).
        }
    }
    col0_buf.par_chunks_mut(n0).for_each(|col| {
        let fft = plan_ifft(n0);
        ifft_inplace(col, &fft);
    });
    // After: col0_buf[(j1*mx2_in+j2)*n0 + i] = IFFT result at axis-0 position i.

    // Step 2: Gather columns (i, j2) of length n1 from col0_buf, parallel IFFT along axis 1.
    // col1_buf layout: [(i*mx2_in+j2)*n1 + j1] — eliminates buf0 intermediate allocation.
    let n_cols_1 = n0 * mx2_in;
    let mut col1_buf = vec![zero; n_cols_1 * n1];
    for i in 0..n0 {
        for j2 in 0..mx2_in {
            let c1 = i * mx2_in + j2;
            for j1 in 0..n1 {
                col1_buf[c1 * n1 + j1] = col0_buf[(j1 * mx2_in + j2) * n0 + i];
            }
        }
    }
    col1_buf.par_chunks_mut(n1).for_each(|col| {
        let fft = plan_ifft(n1);
        ifft_inplace(col, &fft);
    });
    // After: col1_buf[(i*mx2_in+j2)*n1 + j1] = IFFT result at axis-1 position j1.

    // Step 3: Parallel irfft along axis 2.
    // Gather row (i, j1) from col1_buf with stride n1 — eliminates buf1 intermediate allocation.
    let copy_len = mx2_in.min(mx2_out);
    let mut result = vec![0.0_f64; n0 * n1 * n2];
    result.par_chunks_mut(n2).enumerate().for_each(|(row_idx, row_out)| {
        let i = row_idx / n1;
        let j1 = row_idx % n1;
        let fft = plan_ifft(n2);
        ROW_BUF.with(|rb| {
            let mut buf = rb.borrow_mut();
            if buf.len() < n2 {
                buf.resize(n2, zero);
            }
            // Zero, then gather positive frequencies: buf[j2] = col1_buf[(i*mx2_in+j2)*n1 + j1].
            buf[..n2].fill(zero);
            for j2 in 0..copy_len {
                buf[j2] = col1_buf[(i * mx2_in + j2) * n1 + j1];
            }
            // Fill Hermitian conjugate for negative frequencies.
            for k in 1..copy_len {
                let nk = n2 - k;
                if nk >= mx2_out {
                    buf[nk] = buf[k].conj();
                }
            }
            ifft_inplace(&mut buf[..n2], &fft);
            for (out, c) in row_out.iter_mut().zip(buf[..n2].iter()) {
                *out = c.re;
            }
        });
    });
    result
}

// ============================================================================
// Shared protein_bursty_core helper
// ============================================================================

/// Shared logic for ProteinBursty: coarse-graining, mesh build, PGF evaluation, exp.
/// Returns (gf, mx, lims) where gf is the exponentiated generating function on the
/// coarse grid, mx is the reduced grid shape, and lims is the full output shape.
fn protein_bursty_core(
    p_log: &[f64],
    limits: &[usize],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
) -> (Vec<Complex64>, [usize; 3], [usize; 3]) {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let mut mx = [limits[0], limits[1], limits[2]];

    let scale = (mx[2] as f64 / protein_limit).floor() as usize + 1;
    mx[2] = (mx[2] + scale - 1) / scale;
    if !fit_unspliced {
        mx[0] = 1;
    }
    mx[2] = mx[2] / 2 + 1; // rfft half

    let lims = [limits[0], limits[1], limits[2]];
    let (g0, g1, g2) = build_mesh_3d(&mx, &lims);

    let gf_log = protein_pgf(&g0, &g1, &g2, &p, min_fudge, max_fudge);
    // Parallel exp over the grid.
    let gf: Vec<Complex64> = gf_log.par_iter().map(|z| z.exp()).collect();
    (gf, mx, lims)
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

    // Apply Poisson technical noise: g → exp(λ·g) − 1 per modality.
    let mesh = if let Some(ref samp) = samp_log {
        let lam0 = 10.0_f64.powf(samp[0]);
        let lam1 = 10.0_f64.powf(samp[1]);
        build_mesh_2d_poisson_cached(l0, l1, lam0, lam1)
    } else {
        build_mesh_2d_cached(l0, l1)
    };
    let (g0, g1) = (&mesh.0, &mesh.1);

    let gf_log: Vec<Complex64> = match bio_model {
        "Constitutive" => pgf_constitutive(g0, g1, &p),
        "Extrinsic" => pgf_extrinsic(g0, g1, &p),
        "Delay" => pgf_delay(g0, g1, &p),
        "DelayedSplicing" => pgf_delayed_splicing(g0, g1, &p),
        "Bursty" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_bursty(g0, g1, &p, t, quad_order)
        }
        "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_cir(g0, g1, &p, t, quad_order)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown bio_model for eval_model_pss_2d: {bio_model}"
            )))
        }
    };

    // Parallel exp, then irfftn, then parallel normalize.
    let gf: Vec<Complex64> = gf_log.par_iter().map(|z| z.exp()).collect();
    let pss_raw = irfftn_2d(&gf, l0, l1);
    let abs_sum: f64 = pss_raw.par_iter().map(|x| x.abs()).sum();
    Ok(pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect())
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
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
) -> PyResult<Vec<f64>> {
    let (gf, mx, lims) =
        protein_bursty_core(&p_log, &limits, fit_unspliced, protein_limit, min_fudge, max_fudge);
    let pss_raw = irfftn_3d(&gf, mx[0], mx[2], lims[0], lims[1], lims[2]);
    let abs_sum: f64 = pss_raw.par_iter().map(|x| x.abs()).sum();
    Ok(pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect())
}

// ============================================================================
// protein_bursty_pgf — return gf array to Python for scipy irfftn
// ============================================================================

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
    let (gf, mx, _) =
        protein_bursty_core(&p_log, &limits, fit_unspliced, protein_limit, min_fudge, max_fudge);
    let re: Vec<f64> = gf.iter().map(|z| z.re).collect();
    let im: Vec<f64> = gf.iter().map(|z| z.im).collect();
    Ok((re, im, vec![mx[0], mx[1], mx[2]]))
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
