/// monod_core — Rust implementation of CME model core math.
///
/// Exported PyO3 functions
/// -----------------------
/// eval_model_pss_2d(bio_model, p_log, limits, fixed_quad_t, quad_order,
///                  samp_log=None, amb_model="None", amb_log=None) -> Vec<f64>
///     6 two-modality bio_models, seq_model="None"/"Poisson", amb_model="None"/"Equal"/"Unequal".
///     p_log contains only bio params; amb_log = [log10_p] (Equal) or [log10_p0, log10_p1] (Unequal).
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
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use realfft::RealFftPlanner;
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
    /// Per-thread RealFftPlanner — used for the row irfft step in irfftn_2d.
    /// Plans a ComplexToReal (inverse) FFT, which is ~2× faster than a full
    /// complex IFFT of the same size because it exploits real-output symmetry.
    static REAL_FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
    /// Reused scratch buffer for FFT scratch space — grown as needed, never shrunk.
    static SCRATCH_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Reused half-spectrum buffer for irfftn_2d row step (length mx1 = n1/2+1).
    static HALF_BUF: RefCell<Vec<Complex64>> = RefCell::new(Vec::new());
    /// Reused full-length row buffer for irfftn_3d (Hermitian expansion + complex IFFT).
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

/// 3-D ambient mesh cache — Arc-wrapped, keyed by (l0, l1, l2).
/// Stores (g0, g1, g_amb) each of length l0 * l1 * (l2/2+1).
static MESH_CACHE_3D: OnceLock<Mutex<HashMap<(usize, usize, usize), Arc<(Vec<Complex64>, Vec<Complex64>, Vec<Complex64>)>>>> =
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

fn build_mesh_3d_cached(
    l0: usize,
    l1: usize,
    l2: usize,
) -> Arc<(Vec<Complex64>, Vec<Complex64>, Vec<Complex64>)> {
    let cache = MESH_CACHE_3D.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(mesh) = map.get(&(l0, l1, l2)) {
        return Arc::clone(mesh);
    }
    let mx2 = l2 / 2 + 1;
    let mesh = Arc::new(build_mesh_3d(&[l0, l1, mx2], &[l0, l1, l2]));
    map.insert((l0, l1, l2), Arc::clone(&mesh));
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

/// Grid-point count below which we skip rayon and use sequential iterators.
/// Rayon thread-pool spinup + PyO3 call overhead dominate for small grids;
/// switching to sequential avoids this for the common inference case where
/// per-gene grids have median ~100–500 rfft points (3×3 or 5×5 gridsize).
const PAR_THRESHOLD: usize = 2048;

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

    let compute = |k: usize| {
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
    };
    if g0.len() >= PAR_THRESHOLD {
        (0..g0.len()).into_par_iter().map(compute).collect()
    } else {
        (0..g0.len()).map(compute).collect()
    }
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

    let compute = |k: usize| {
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
    };
    let mut gf: Vec<Complex64> = if g0.len() >= PAR_THRESHOLD {
        (0..g0.len()).into_par_iter().map(compute).collect()
    } else {
        (0..g0.len()).map(compute).collect()
    };
    if g0.len() >= PAR_THRESHOLD {
        gf.par_iter_mut().for_each(|v| *v /= 2.0);
    } else {
        gf.iter_mut().for_each(|v| *v /= 2.0);
    }
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

    let phase1_fn = |k: usize| {
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
    };

    // Phase 1: fixed trajectory per point.
    let mut states: Vec<(Complex64, Complex64, Complex64, Complex64)> = if n_grid >= PAR_THRESHOLD {
        (0..n_grid).into_par_iter().map(phase1_fn).collect()
    } else {
        (0..n_grid).map(phase1_fn).collect()
    };

    // Phase 2: global-max termination — matches Python's
    // `while np.max(np.abs(u_tilde[0])) >= 1e-3`.
    let phase2_step = |s: &mut (Complex64, Complex64, Complex64, Complex64)| {
        let (nu0, nu1, nu2) = rk4_step(s.0, s.1, s.2, dt, beta, gamma, k_p, gamma_p);
        s.0 = nu0; s.1 = nu1; s.2 = nu2;
        s.3 += nu0 * b / (one - nu0 * b) * dt;
    };
    loop {
        let max_norm = if n_grid >= PAR_THRESHOLD {
            states.par_iter().map(|s| s.0.norm()).reduce(|| 0.0_f64, f64::max)
        } else {
            states.iter().map(|s| s.0.norm()).fold(0.0_f64, f64::max)
        };
        if max_norm < 1e-3 {
            break;
        }
        if n_grid >= PAR_THRESHOLD {
            states.par_iter_mut().for_each(phase2_step);
        } else {
            states.iter_mut().for_each(phase2_step);
        }
    }

    // Phase 3: final half-step.
    let phase3_step = |s: &mut (Complex64, Complex64, Complex64, Complex64)| {
        let (nu0, _, _) = rk4_step(s.0, s.1, s.2, dt, beta, gamma, k_p, gamma_p);
        s.3 += nu0 * b / (one - nu0 * b) * (dt / 2.0);
    };
    if n_grid >= PAR_THRESHOLD {
        states.par_iter_mut().for_each(phase3_step);
    } else {
        states.iter_mut().for_each(phase3_step);
    }

    if n_grid >= PAR_THRESHOLD {
        states.into_par_iter().map(|s| s.3).collect()
    } else {
        states.into_iter().map(|s| s.3).collect()
    }
}

// ============================================================================
// 2-D irfftn — parallel column IFFTs + parallel row irffts, no mid buffer
// ============================================================================

/// irfftn for shape [n0, mx1=n1/2+1] → [n0, n1].
///
/// Algorithm (verified vs scipy):
///   1. Transpose input to column-major; parallel IFFT of length n0 per column j.
///   2. For each row i: gather the half-spectrum (length mx1) from col_buf with
///      stride n0, then run a real-output IFFT via the `realfft` crate.
///      This avoids filling in the Hermitian conjugate and uses a size-n1/2
///      complex FFT internally, making the row step ~2× faster than a full
///      complex IFFT of length n1.
///
/// Each parallel task uses its own thread-local planners and scratch buffers.
fn irfftn_2d(input: &[Complex64], n0: usize, n1: usize) -> Vec<f64> {
    let mx1 = n1 / 2 + 1;
    let zero = Complex64::new(0.0, 0.0);

    // Step 1: Transpose to column-major, then IFFT along axis 0.
    let mut col_buf = vec![zero; mx1 * n0];
    for i in 0..n0 {
        for j in 0..mx1 {
            col_buf[j * n0 + i] = input[i * mx1 + j];
        }
    }
    if input.len() >= PAR_THRESHOLD {
        col_buf.par_chunks_mut(n0).for_each(|col| {
            let fft = plan_ifft(n0);
            ifft_inplace(col, &fft);
        });
    } else {
        col_buf.chunks_mut(n0).for_each(|col| {
            let fft = plan_ifft(n0);
            ifft_inplace(col, &fft);
        });
    }

    // Step 2: irfft along axis 1 using realfft (ComplexToReal plan).
    // Gather each row i (length mx1) from col_buf with stride n0, then run
    // the real-output IFFT directly on the half-spectrum — no Hermitian fill.
    let mut result = vec![0.0_f64; n0 * n1];
    let irfft_row = |(i, row_out): (usize, &mut [f64])| {
        let irfft = REAL_FFT_PLANNER.with(|cell| cell.borrow_mut().plan_fft_inverse(n1));
        let scratch_len = irfft.get_scratch_len();
        HALF_BUF.with(|hb| {
            SCRATCH_BUF.with(|sc| {
                let mut half = hb.borrow_mut();
                let mut scratch = sc.borrow_mut();
                if half.len() < mx1 {
                    half.resize(mx1, zero);
                }
                if scratch.len() < scratch_len {
                    scratch.resize(scratch_len, zero);
                }
                // Gather positive-frequency slice from col_buf (stride n0).
                for k in 0..mx1 {
                    half[k] = col_buf[k * n0 + i];
                }
                // realfft requires the DC (k=0) bin to be real-valued.
                // For even n1 only, the Nyquist bin (k=mx1-1) must also be real.
                // Column IFFTs leave small floating-point residuals; zero them.
                half[0].im = 0.0;
                if n1 % 2 == 0 {
                    half[mx1 - 1].im = 0.0;
                }
                // Real-output IFFT: half[..mx1] → row_out[..n1].
                // Input is modified in-place (scratch); output is f64.
                irfft.process_with_scratch(
                    &mut half[..mx1],
                    row_out,
                    &mut scratch[..scratch_len],
                ).expect("irfft row failed");
            });
        });
    };
    if n0 * n1 >= PAR_THRESHOLD {
        result.par_chunks_mut(n1).enumerate().for_each(irfft_row);
    } else {
        result.chunks_mut(n1).enumerate().for_each(irfft_row);
    }
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
    if col0_buf.len() >= PAR_THRESHOLD {
        col0_buf.par_chunks_mut(n0).for_each(|col| {
            let fft = plan_ifft(n0);
            ifft_inplace(col, &fft);
        });
    } else {
        col0_buf.chunks_mut(n0).for_each(|col| {
            let fft = plan_ifft(n0);
            ifft_inplace(col, &fft);
        });
    }
    // After: col0_buf[(j1*mx2_in+j2)*n0 + i] = IFFT result at axis-0 position i.

    // Step 2: Gather columns (i, j2) of length n1 from col0_buf, IFFT along axis 1.
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
    if col1_buf.len() >= PAR_THRESHOLD {
        col1_buf.par_chunks_mut(n1).for_each(|col| {
            let fft = plan_ifft(n1);
            ifft_inplace(col, &fft);
        });
    } else {
        col1_buf.chunks_mut(n1).for_each(|col| {
            let fft = plan_ifft(n1);
            ifft_inplace(col, &fft);
        });
    }
    // After: col1_buf[(i*mx2_in+j2)*n1 + j1] = IFFT result at axis-1 position j1.

    // Step 3: irfft along axis 2.
    // Gather row (i, j1) from col1_buf with stride n1 — eliminates buf1 intermediate allocation.
    let copy_len = mx2_in.min(mx2_out);
    let mut result = vec![0.0_f64; n0 * n1 * n2];
    let irfft_row3 = |(row_idx, row_out): (usize, &mut [f64])| {
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
    };
    if n0 * n1 * n2 >= PAR_THRESHOLD {
        result.par_chunks_mut(n2).enumerate().for_each(irfft_row3);
    } else {
        result.chunks_mut(n2).enumerate().for_each(irfft_row3);
    }
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
    let n = gf_log.len();
    let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
        gf_log.par_iter().map(|z| z.exp()).collect()
    } else {
        gf_log.iter().map(|z| z.exp()).collect()
    };
    (gf, mx, lims)
}

// ============================================================================
// eval_model_pss — 2-D models
// ============================================================================

/// Evaluate log-PGF for any supported 2-D bio_model given g0/g1 mesh slices.
fn eval_pgf_2d(
    bio_model: &str,
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
) -> PyResult<Vec<Complex64>> {
    Ok(match bio_model {
        "Constitutive" => pgf_constitutive(g0, g1, p),
        "Extrinsic"    => pgf_extrinsic(g0, g1, p),
        "Delay"        => pgf_delay(g0, g1, p),
        "DelayedSplicing" => pgf_delayed_splicing(g0, g1, p),
        "Bursty" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_bursty(g0, g1, p, t, quad_order)
        }
        "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_cir(g0, g1, p, t, quad_order)
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for eval_model_pss_2d: {bio_model}"
        ))),
    })
}

#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, fixed_quad_t, quad_order, samp_log=None, amb_model="None", amb_log=None))]
fn eval_model_pss_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<Vec<f64>>,
    amb_model: &str,
    amb_log: Option<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();

    if amb_model == "None" {
        // --- existing 2-D path ---
        let (l0, l1) = (limits[0], limits[1]);
        let mesh = if let Some(ref samp) = samp_log {
            let lam0 = 10.0_f64.powf(samp[0]);
            let lam1 = 10.0_f64.powf(samp[1]);
            build_mesh_2d_poisson_cached(l0, l1, lam0, lam1)
        } else {
            build_mesh_2d_cached(l0, l1)
        };
        let (g0, g1) = (&mesh.0, &mesh.1);
        let gf_log = eval_pgf_2d(bio_model, g0, g1, &p, fixed_quad_t, quad_order)?;
        let n = gf_log.len();
        let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
            gf_log.par_iter().map(|z| z.exp()).collect()
        } else {
            gf_log.iter().map(|z| z.exp()).collect()
        };
        let pss_raw = irfftn_2d(&gf, l0, l1);
        let pn = pss_raw.len();
        let abs_sum: f64 = if pn >= PAR_THRESHOLD {
            pss_raw.par_iter().map(|x| x.abs()).sum()
        } else {
            pss_raw.iter().map(|x| x.abs()).sum()
        };
        return Ok(if pn >= PAR_THRESHOLD {
            pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect()
        } else {
            pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
        });
    }

    // --- 3-D ambient path ---
    // limits = [l0, l1, l_amb].  p_log contains only bio params; amb_log has
    // [log10_p_amb] (Equal) or [log10_p0, log10_p1] (Unequal).
    let (l0, l1, l2) = (limits[0], limits[1], limits[2]);
    let mx2 = l2 / 2 + 1;

    let amb = amb_log.as_deref().unwrap_or(&[]);
    let (p_amb0, p_amb1): (f64, f64) = match amb_model {
        "Equal"   => { let v = 10.0_f64.powf(amb[0]); (v, v) }
        "Unequal" => (10.0_f64.powf(amb[0]), 10.0_f64.powf(amb[1])),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown amb_model: {amb_model}")
        )),
    };

    // Build (or retrieve cached) 3-D base mesh.
    let mesh3 = build_mesh_3d_cached(l0, l1, l2);
    let (g0_base, g1_base, g_amb) = (&mesh3.0, &mesh3.1, &mesh3.2);

    // Mix: g0_eff = (1-p0)*g0 + p0*g_amb,  g1_eff = (1-p1)*g1 + p1*g_amb.
    let one  = Complex64::new(1.0, 0.0);
    let q0   = Complex64::new(1.0 - p_amb0, 0.0);
    let pa0  = Complex64::new(p_amb0, 0.0);
    let q1   = Complex64::new(1.0 - p_amb1, 0.0);
    let pa1  = Complex64::new(p_amb1, 0.0);

    let mut g0_eff: Vec<Complex64> = g0_base.iter().zip(g_amb.iter())
        .map(|(&g0, &ga)| g0 * q0 + ga * pa0)
        .collect();
    let mut g1_eff: Vec<Complex64> = g1_base.iter().zip(g_amb.iter())
        .map(|(&g1, &ga)| g1 * q1 + ga * pa1)
        .collect();

    // Apply Poisson seq_model if requested (applied to mixed variables).
    if let Some(ref samp) = samp_log {
        let lam0 = 10.0_f64.powf(samp[0]);
        let lam1 = 10.0_f64.powf(samp[1]);
        g0_eff.iter_mut().for_each(|z| *z = (*z * lam0).exp() - one);
        g1_eff.iter_mut().for_each(|z| *z = (*z * lam1).exp() - one);
    }

    // Evaluate log-PGF, exp, 3-D irfftn, normalize.
    let gf_log = eval_pgf_2d(bio_model, &g0_eff, &g1_eff, &p, fixed_quad_t, quad_order)?;
    let n = gf_log.len();
    let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
        gf_log.par_iter().map(|z| z.exp()).collect()
    } else {
        gf_log.iter().map(|z| z.exp()).collect()
    };
    // irfftn_3d(input, mx0, mx2_in, n0, n1, n2): mx0=l0 (no axis-0 coarsening)
    let pss_raw = irfftn_3d(&gf, l0, mx2, l0, l1, l2);
    let pn = pss_raw.len();
    let abs_sum: f64 = if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs()).sum()
    } else {
        pss_raw.iter().map(|x| x.abs()).sum()
    };
    Ok(if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect()
    } else {
        pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
    })
}

// ============================================================================
// eval_model_pss_2d_batch — gene-parallel batch evaluation
// ============================================================================

/// Evaluate the 2-D PSS for a single (params, limits) pair without PyO3 types.
///
/// Caller must validate `bio_model` before calling (panics on unknown model).
/// Used as the per-gene unit in `eval_model_pss_2d_batch`.
fn eval_model_pss_2d_seq(
    bio_model: &str,
    p_log: &[f64],
    limits: &[usize],
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<&[f64]>,
) -> Vec<f64> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let (l0, l1) = (limits[0], limits[1]);

    let mesh = if let Some(samp) = samp_log {
        let lam0 = 10.0_f64.powf(samp[0]);
        let lam1 = 10.0_f64.powf(samp[1]);
        build_mesh_2d_poisson_cached(l0, l1, lam0, lam1)
    } else {
        build_mesh_2d_cached(l0, l1)
    };
    let (g0, g1) = (&mesh.0, &mesh.1);

    let gf_log = match bio_model {
        "Constitutive" => pgf_constitutive(g0, g1, &p),
        "Extrinsic"    => pgf_extrinsic(g0, g1, &p),
        "Delay"        => pgf_delay(g0, g1, &p),
        "DelayedSplicing" => pgf_delayed_splicing(g0, g1, &p),
        "Bursty" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_bursty(g0, g1, &p, t, quad_order)
        }
        "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_cir(g0, g1, &p, t, quad_order)
        }
        other => panic!("Unknown bio_model in eval_model_pss_2d_seq: {other}"),
    };

    let n = gf_log.len();
    let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
        gf_log.par_iter().map(|z| z.exp()).collect()
    } else {
        gf_log.iter().map(|z| z.exp()).collect()
    };
    let pss_raw = irfftn_2d(&gf, l0, l1);
    let pn = pss_raw.len();
    let abs_sum: f64 = if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs()).sum()
    } else {
        pss_raw.iter().map(|x| x.abs()).sum()
    };
    if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect()
    } else {
        pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
    }
}

/// Evaluate the 2-D PSS for N (params, limits) pairs in parallel using rayon.
///
/// The GIL is released for the entire computation via `py.allow_threads()`,
/// so rayon workers run freely on all available cores.  Each entry is
/// processed by `eval_model_pss_2d_seq`, which is sequential for small grids
/// (n_grid < PAR_THRESHOLD); gene-level parallelism across the batch fills
/// all cores instead.
///
/// Parameters
/// ----------
/// bio_model    : one of the six supported 2-D bio_models
/// params_list  : log10 biological parameters per call
/// limits_list  : grid dimensions per call
/// fixed_quad_t : quadrature time-scale multiplier
/// quad_order   : number of Gauss-Legendre quadrature points
/// samp_list    : optional list of Poisson sampling params per call
///                (pass None for seq_model="None"; list entries may be None)
#[pyfunction]
#[pyo3(signature = (bio_model, params_list, limits_list, fixed_quad_t, quad_order, samp_list=None, num_threads=None))]
fn eval_model_pss_2d_batch(
    py: Python<'_>,
    bio_model: String,
    params_list: Vec<Vec<f64>>,
    limits_list: Vec<Vec<usize>>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    num_threads: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    // Validate bio_model once, before releasing the GIL.
    match bio_model.as_str() {
        "Constitutive" | "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => {}
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for eval_model_pss_2d_batch: {other}"
        ))),
    }
    let n = params_list.len();
    let results = py.allow_threads(|| {
        let run = || {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let samp = samp_list
                        .as_ref()
                        .and_then(|sl| sl[i].as_deref());
                    eval_model_pss_2d_seq(
                        &bio_model,
                        &params_list[i],
                        &limits_list[i],
                        fixed_quad_t,
                        quad_order,
                        samp,
                    )
                })
                .collect::<Vec<Vec<f64>>>()
        };
        match num_threads {
            Some(nt) => rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            None => run(),
        }
    });
    Ok(results)
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
    let pn = pss_raw.len();
    let abs_sum: f64 = if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs()).sum()
    } else {
        pss_raw.iter().map(|x| x.abs()).sum()
    };
    Ok(if pn >= PAR_THRESHOLD {
        pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect()
    } else {
        pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
    })
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
// Parallel unique-histogram extraction
// ============================================================================

// Dense counting table threshold: if (max_l0+1)*(max_l1+1)*… ≤ this, use
// the counting approach instead of O(n log n) sort.
// 16384 = 2^14 keeps the table (128 KB) inside L2 cache; for 2 layers this
// covers max_count ≈ 128 per layer, which handles the vast majority of real
// RNA-seq genes.
const DENSE_THRESHOLD: usize = 1 << 14; // 16384

// Per-thread scratch buffers reused across genes — avoids one allocation per
// gene for the sort-based fallback path.
thread_local! {
    static FLAT_BUF: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    static ORDER_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    static DENSE_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}

/// Compute unique-microstate histograms for every gene (column) in parallel.
///
/// Parameters
/// ----------
/// layers : list of 2-D int64 numpy arrays, each shaped (n_cells, n_genes).
///          Elements are treated as non-negative integer counts.
///
/// Returns
/// -------
/// (coords, freqs) where
///   coords[g] — list of unique microstates; each microstate is a list of i64
///               with one entry per layer, length = n_layers
///   freqs[g]  — corresponding normalised frequencies (count / n_cells)
#[pyfunction]
fn make_histograms_unique(
    py: Python<'_>,
    layers: Vec<PyReadonlyArray2<'_, i64>>,
) -> PyResult<(Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>)> {
    if layers.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n_cells = layers[0].shape()[0];
    let n_genes = layers[0].shape()[1];
    let n_layers = layers.len();

    // Borrow flat row-major slices from numpy arrays under the GIL.
    let raw: Vec<&[i64]> = layers
        .iter()
        .map(|arr| {
            arr.as_slice().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "layer array must be C-contiguous: {e}"
                ))
            })
        })
        .collect::<PyResult<_>>()?;

    // Release the GIL during the parallel computation.
    let results: Vec<(Vec<Vec<i64>>, Vec<f64>)> = py.allow_threads(|| {
        (0..n_genes)
            .into_par_iter()
            .map(|gene| {
                // --- Column copy (one stride-n_genes pass per layer) ---
                // Merging this into the flat-buffer build below would require
                // two stride reads per layer; keeping it separate allows each
                // subsequent step to work purely sequentially.
                let cols: Vec<Vec<i64>> = raw
                    .iter()
                    .map(|layer| {
                        (0..n_cells)
                            .map(|cell| layer[cell * n_genes + gene])
                            .collect()
                    })
                    .collect();

                // --- Max per layer for this gene ---
                let max_per_layer: Vec<usize> = cols
                    .iter()
                    .map(|col| col.iter().copied().max().unwrap_or(0).max(0) as usize)
                    .collect();

                let total_states: usize =
                    max_per_layer.iter().map(|&m| m + 1).product();

                if total_states <= DENSE_THRESHOLD {
                    // ---- O(n_cells + total_states) counting path ----
                    //
                    // Build strides for row-major multi-dim indexing so that
                    // iterating flat_idx = 0..total_states yields microstates
                    // in lexicographic order (matching np.unique's output).
                    let mut strides = vec![1usize; n_layers];
                    for l in (0..n_layers - 1).rev() {
                        strides[l] = strides[l + 1] * (max_per_layer[l + 1] + 1);
                    }

                    // Reuse thread-local dense table; extend if needed, then zero.
                    DENSE_BUF.with(|db| {
                        let mut table = db.borrow_mut();
                        if table.len() < total_states {
                            table.resize(total_states, 0);
                        }
                        let table = &mut table[..total_states];
                        table.fill(0);

                        // Count each cell's microstate in O(n_cells).
                        for cell in 0..n_cells {
                            let idx: usize = cols
                                .iter()
                                .zip(strides.iter())
                                .map(|(col, &s)| col[cell] as usize * s)
                                .sum();
                            table[idx] += 1;
                        }

                        // Collect non-zero entries — already in lexicographic order.
                        let mut unique: Vec<Vec<i64>> = Vec::new();
                        let mut freqs: Vec<f64> = Vec::new();
                        for flat_idx in 0..total_states {
                            if table[flat_idx] == 0 {
                                continue;
                            }
                            let mut microstate = vec![0i64; n_layers];
                            let mut rem = flat_idx;
                            for l in 0..n_layers {
                                microstate[l] = (rem / strides[l]) as i64;
                                rem %= strides[l];
                            }
                            unique.push(microstate);
                            freqs.push(table[flat_idx] as f64 / n_cells as f64);
                        }
                        (unique, freqs)
                    })
                } else {
                    // ---- O(n_cells log n_cells) sort-based fallback ----
                    //
                    // Reuse thread-local flat (n_cells × n_layers) and order buffers.
                    FLAT_BUF.with(|fb| {
                        ORDER_BUF.with(|ob| {
                            let mut flat = fb.borrow_mut();
                            let mut order = ob.borrow_mut();

                            // Resize buffers only if they need to grow.
                            let flat_len = n_cells * n_layers;
                            if flat.len() < flat_len {
                                flat.resize(flat_len, 0);
                            }
                            if order.len() < n_cells {
                                order.resize(n_cells, 0);
                            }
                            let flat = &mut flat[..flat_len];
                            let order = &mut order[..n_cells];

                            // Build interleaved microstate rows from contiguous cols.
                            for cell in 0..n_cells {
                                for (l, col) in cols.iter().enumerate() {
                                    flat[cell * n_layers + l] = col[cell];
                                }
                            }

                            // Sort row indices (one Vec<usize>, no per-cell allocs).
                            for (i, v) in order.iter_mut().enumerate() {
                                *v = i;
                            }
                            order.sort_unstable_by(|&a, &b| {
                                flat[a * n_layers..(a + 1) * n_layers]
                                    .cmp(&flat[b * n_layers..(b + 1) * n_layers])
                            });

                            // Run-length encode.
                            let mut unique: Vec<Vec<i64>> = Vec::new();
                            let mut counts: Vec<usize> = Vec::new();
                            let mut prev_start = usize::MAX;
                            for &idx in order.iter() {
                                let row_start = idx * n_layers;
                                if prev_start != usize::MAX
                                    && flat[prev_start..prev_start + n_layers]
                                        == flat[row_start..row_start + n_layers]
                                {
                                    *counts.last_mut().unwrap() += 1;
                                } else {
                                    unique.push(
                                        flat[row_start..row_start + n_layers].to_vec(),
                                    );
                                    counts.push(1);
                                    prev_start = row_start;
                                }
                            }

                            let freqs: Vec<f64> = counts
                                .iter()
                                .map(|&c| c as f64 / n_cells as f64)
                                .collect();
                            (unique, freqs)
                        })
                    })
                }
            })
            .collect()
    });

    let mut all_coords = Vec::with_capacity(n_genes);
    let mut all_freqs = Vec::with_capacity(n_genes);
    for (c, f) in results {
        all_coords.push(c);
        all_freqs.push(f);
    }
    Ok((all_coords, all_freqs))
}

// ============================================================================
// Custom reaction-network ODE integration — general n-species, parallel RK4
// ============================================================================

/// Compact encoding of one elementary reaction.
/// `kind`: 0 = geometric-burst production, 1 = deterministic production, 2 = first-order.
/// `rate_idx`: index of the rate constant in the `params` slice.
/// `extra1`: kind 0 → burst-param idx; kind 2 → reactant species idx; else -1.
/// `extra2`: kind 0 → burst species idx; else -1.
/// `prod_start`/`prod_end`: range in flat `prod_sp`/`prod_st` arrays.
#[derive(Clone, Copy)]
struct RxnInfo {
    kind:       u8,
    rate_idx:   u32,
    extra1:     i32,
    extra2:     i32,
    prod_start: u32,
    prod_end:   u32,
}

/// Build an N-dimensional complex mesh for irfftn.
/// Returns (g, mx_sizes, n_grid) where g[s] is the flat grid for species s.
/// Last axis has rfft half-length (l/2+1); all others full.
fn build_mesh_nd(limits: &[usize]) -> (Vec<Vec<Complex64>>, Vec<usize>, usize) {
    let n = limits.len();
    let two_pi = 2.0 * std::f64::consts::PI;

    let mx: Vec<usize> = limits.iter().enumerate().map(|(i, &l)| {
        if i == n - 1 { l / 2 + 1 } else { l }
    }).collect();

    let n_grid: usize = mx.iter().product();

    let freq_vecs: Vec<Vec<Complex64>> = limits.iter().enumerate().map(|(ax, &l)| {
        (0..mx[ax]).map(|k| {
            let theta = -two_pi * k as f64 / l as f64;
            Complex64::new(theta.cos() - 1.0, theta.sin())
        }).collect()
    }).collect();

    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * mx[i + 1];
    }

    let mut g = vec![vec![Complex64::new(0.0, 0.0); n_grid]; n];
    for pt in 0..n_grid {
        for s in 0..n {
            let idx = (pt / strides[s]) % mx[s];
            g[s][pt] = freq_vecs[s][idx];
        }
    }

    (g, mx, n_grid)
}

/// Compute the custom-network ODE RHS for one grid point.
#[inline(always)]
fn custom_rhs_point(
    u:       &[Complex64],
    params:  &[f64],
    rxns:    &[RxnInfo],
    prod_sp: &[u32],
    prod_st: &[u32],
) -> (Vec<Complex64>, Complex64) {
    let n   = u.len();
    let one = Complex64::new(1.0, 0.0);
    let mut du   = vec![Complex64::new(0.0, 0.0); n];
    let mut dphi = Complex64::new(0.0, 0.0);

    for rxn in rxns {
        let k  = Complex64::new(params[rxn.rate_idx as usize], 0.0);
        let ps = rxn.prod_start as usize;
        let pe = rxn.prod_end   as usize;

        match rxn.kind {
            0 => {
                // Geometric-burst production: dφ += k·b·u[sp]/(1−b·u[sp])
                let b  = Complex64::new(params[rxn.extra1 as usize], 0.0);
                let ui = u[rxn.extra2 as usize];
                dphi += k * b * ui / (one - b * ui);
            }
            1 => {
                // Deterministic production: dφ += k·(z_prod − 1)
                let mut z = one;
                for i in ps..pe {
                    z *= (one + u[prod_sp[i] as usize]).powu(prod_st[i]);
                }
                dphi += k * (z - one);
            }
            _ => {
                // First-order: du[ri] += k·(z_prod − (1 + u[ri]))
                let ri    = rxn.extra1 as usize;
                let mut z = one;
                for i in ps..pe {
                    z *= (one + u[prod_sp[i] as usize]).powu(prod_st[i]);
                }
                du[ri] += k * (z - (one + u[ri]));
            }
        }
    }
    (du, dphi)
}

/// One RK4 step for one grid point (in-place).
/// φ is accumulated via RK4 weights, matching the Python `_rk4_step` convention.
#[inline(always)]
fn custom_rk4_point(
    u:       &mut Vec<Complex64>,
    phi:     &mut Complex64,
    params:  &[f64],
    rxns:    &[RxnInfo],
    prod_sp: &[u32],
    prod_st: &[u32],
    dt:      f64,
) {
    let n = u.len();
    let h = dt / 2.0;
    let s = dt / 6.0;

    let (k1u, k1p) = custom_rhs_point(u, params, rxns, prod_sp, prod_st);

    let u2: Vec<_> = (0..n).map(|i| u[i] + k1u[i] * h).collect();
    let (k2u, k2p) = custom_rhs_point(&u2, params, rxns, prod_sp, prod_st);

    let u3: Vec<_> = (0..n).map(|i| u[i] + k2u[i] * h).collect();
    let (k3u, k3p) = custom_rhs_point(&u3, params, rxns, prod_sp, prod_st);

    let u4: Vec<_> = (0..n).map(|i| u[i] + k3u[i] * dt).collect();
    let (k4u, k4p) = custom_rhs_point(&u4, params, rxns, prod_sp, prod_st);

    for i in 0..n {
        u[i] += (k1u[i] + k2u[i] * 2.0 + k3u[i] * 2.0 + k4u[i]) * s;
    }
    *phi += (k1p + k2p * 2.0 + k3p * 2.0 + k4p) * s;
}

/// Integrate the custom network over all grid points in parallel (rayon).
/// Returns exp(φ) at each grid point.
fn custom_pgf_parallel(
    g:               &[Vec<Complex64>],
    params:          &[f64],
    rxns:            &[RxnInfo],
    prod_sp:         &[u32],
    prod_st:         &[u32],
    n_species:       usize,
    n_grid:          usize,
    dt:              f64,
    n_steps:         usize,
    max_while_steps: usize,
) -> Vec<Complex64> {
    // Phase 1: fixed steps, fully parallel.
    let mut states: Vec<(Vec<Complex64>, Complex64)> = (0..n_grid)
        .into_par_iter()
        .map(|k| {
            // Truncate to f32 precision to match Python's complex64 cast.
            let mut u: Vec<Complex64> = (0..n_species)
                .map(|s| Complex64::new(
                    g[s][k].re as f32 as f64,
                    g[s][k].im as f32 as f64,
                ))
                .collect();
            // Leading trapezoidal half-step.
            let (_, dp0) = custom_rhs_point(&u, params, rxns, prod_sp, prod_st);
            let mut phi = dp0 * (dt / 2.0);
            for _ in 0..n_steps {
                custom_rk4_point(&mut u, &mut phi, params, rxns, prod_sp, prod_st, dt);
            }
            (u, phi)
        })
        .collect();

    // Phase 2: global-max termination — matches Python `while np.max(|u[0]|) > 1e-3`.
    let mut while_steps = 0usize;
    loop {
        let max_norm = states.par_iter().map(|(u, _)| u[0].norm()).reduce(|| 0.0_f64, f64::max);
        if max_norm < 1e-3 || while_steps >= max_while_steps {
            break;
        }
        states.par_iter_mut().for_each(|(u, phi)| {
            custom_rk4_point(u, phi, params, rxns, prod_sp, prod_st, dt);
        });
        while_steps += 1;
    }

    // Phase 3: trailing trapezoidal half-step, then exp(φ).
    states.into_par_iter().map(|(u, mut phi)| {
        let (_, dpf) = custom_rhs_point(&u, params, rxns, prod_sp, prod_st);
        phi += dpf * (dt / 2.0);
        phi.exp()
    }).collect()
}

/// Evaluate the custom reaction-network log-PGF over all grid points in
/// parallel (rayon), returning exp(φ) as split real/imaginary Vec<f64>.
///
/// Reaction topology is encoded as parallel arrays (one entry per reaction):
/// * `rxn_kinds`     — 0=geometric burst, 1=deterministic prod, 2=first-order
/// * `rxn_rate_idxs` — index into `params` for each rate constant
/// * `rxn_extra1`    — kind 0: burst-param idx; kind 2: reactant species idx; else -1
/// * `rxn_extra2`    — kind 0: burst species idx; else -1
/// * `prod_sp`/`prod_st`/`prod_off` — CSR products (species idx, stoich, offsets)
/// * `params`        — **linear-scale** values (including the normalised rate if any)
/// * `limits`        — grid size per species (length = n_species)
/// * `dt`, `n_steps`, `max_while_steps` — integration control
///
/// Returns `(re_exphi, im_exphi, shape)` in row-major flat order.
#[pyfunction]
fn eval_custom_network_pgf(
    n_species:       usize,
    limits:          Vec<usize>,
    rxn_kinds:       Vec<u8>,
    rxn_rate_idxs:   Vec<u32>,
    rxn_extra1:      Vec<i32>,
    rxn_extra2:      Vec<i32>,
    prod_sp:         Vec<u32>,
    prod_st:         Vec<u32>,
    prod_off:        Vec<u32>,
    params:          Vec<f64>,
    dt:              f64,
    n_steps:         usize,
    max_while_steps: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let n_rxns = rxn_kinds.len();
    let rxns: Vec<RxnInfo> = (0..n_rxns)
        .map(|i| RxnInfo {
            kind:       rxn_kinds[i],
            rate_idx:   rxn_rate_idxs[i],
            extra1:     rxn_extra1[i],
            extra2:     rxn_extra2[i],
            prod_start: prod_off[i],
            prod_end:   prod_off[i + 1],
        })
        .collect();

    let (g, mx, n_grid) = build_mesh_nd(&limits);

    let exp_phi = custom_pgf_parallel(
        &g, &params, &rxns, &prod_sp, &prod_st,
        n_species, n_grid, dt, n_steps, max_while_steps,
    );

    let re: Vec<f64> = exp_phi.iter().map(|z| z.re).collect();
    let im: Vec<f64> = exp_phi.iter().map(|z| z.im).collect();
    Ok((re, im, mx))
}

// ============================================================================
// KLD evaluation helpers (sparse "unique" histogram)
// ============================================================================

/// Compute KLD(data ‖ model) from a sparse histogram.
///
/// pss     : flat PSS array of length l0*l1 (row-major: index = u*l1 + s)
/// l1      : number of spliced bins (second dimension)
/// u_idx   : unspliced indices of observed microstates
/// s_idx   : spliced indices of observed microstates
/// f       : fractional frequencies (counts / n_cells), same length as u_idx
/// eps     : minimum probability floor
///
/// Returns sum_i f[i] * ln(f[i] / pss[u[i],s[i]]).
#[inline]
fn compute_kld_sparse(
    pss: &[f64],
    l1: usize,
    u_idx: &[u64],
    s_idx: &[u64],
    f: &[f64],
    eps: f64,
) -> f64 {
    u_idx
        .iter()
        .zip(s_idx.iter())
        .zip(f.iter())
        .map(|((&u, &s), &fi)| {
            let pval = pss[u as usize * l1 + s as usize].max(eps);
            fi * (fi / pval).ln()
        })
        .sum()
}

/// Evaluate the KLD between a sparse observed histogram and a 2-D CME model.
///
/// Parameters (Python-visible)
/// ---------------------------
/// bio_model    : model name (same as eval_model_pss_2d)
/// p_log        : log10 biological parameters
/// limits       : [l0, l1] grid dimensions
/// u_idx        : unspliced bin indices for each unique observed microstate
/// s_idx        : spliced bin indices for each unique observed microstate
/// f            : fractional frequency per microstate (counts / n_cells)
/// fixed_quad_t : quadrature time-scale multiplier
/// quad_order   : number of Gauss-Legendre quadrature points
/// samp_log     : optional log10 Poisson sampling parameters [lam0, lam1]
/// eps          : minimum probability floor (default 1e-15)
///
/// Returns the scalar KLD value.
#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, u_idx, s_idx, f, fixed_quad_t, quad_order, samp_log=None, eps=1e-15))]
#[allow(clippy::too_many_arguments)]
fn eval_kld_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    u_idx: Vec<u64>,
    s_idx: Vec<u64>,
    f: Vec<f64>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<Vec<f64>>,
    eps: f64,
) -> PyResult<f64> {
    let pss = eval_model_pss_2d_seq(
        bio_model,
        &p_log,
        &limits,
        fixed_quad_t,
        quad_order,
        samp_log.as_deref(),
    );
    let l1 = limits[1];
    Ok(compute_kld_sparse(&pss, l1, &u_idx, &s_idx, &f, eps))
}

/// Evaluate the KLD and its forward finite-difference gradient for a 2-D CME model.
///
/// All (n_params + 1) PSS evaluations (base point + one per parameter) are run
/// in parallel via rayon with the GIL released, making gradient computation
/// effectively free compared to calling eval_kld_2d n_params+1 times.
///
/// Parameters
/// ----------
/// Same as eval_kld_2d, plus:
/// fd_eps : finite-difference step size in log10 space (default 1e-6)
///
/// Returns (kld: float, grad: list[float]) where grad[i] = d KLD / d log10(θ_i).
#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, u_idx, s_idx, f, fixed_quad_t, quad_order, fd_eps=1e-6, samp_log=None, eps=1e-15))]
#[allow(clippy::too_many_arguments)]
fn eval_kld_grad_2d(
    py: Python<'_>,
    bio_model: String,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    u_idx: Vec<u64>,
    s_idx: Vec<u64>,
    f: Vec<f64>,
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    samp_log: Option<Vec<f64>>,
    eps: f64,
) -> PyResult<(f64, Vec<f64>)> {
    // Validate model before releasing GIL.
    match bio_model.as_str() {
        "Constitutive" | "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => {}
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for eval_kld_grad_2d: {other}"
        ))),
    }
    let n_params = p_log.len();
    let l1 = limits[1];
    let samp_ref: Option<Vec<f64>> = samp_log;

    // Evaluate base point + n_params perturbations in parallel (all with GIL released).
    let all_klds: Vec<f64> = py.allow_threads(|| {
        (0..=n_params)
            .into_par_iter()
            .map(|i| {
                let p_eval: Vec<f64> = if i == 0 {
                    p_log.clone()
                } else {
                    let mut p_eps = p_log.clone();
                    p_eps[i - 1] += fd_eps;
                    p_eps
                };
                let pss = eval_model_pss_2d_seq(
                    &bio_model,
                    &p_eval,
                    &limits,
                    fixed_quad_t,
                    quad_order,
                    samp_ref.as_deref(),
                );
                compute_kld_sparse(&pss, l1, &u_idx, &s_idx, &f, eps)
            })
            .collect()
    });

    let kld0 = all_klds[0];
    let grad: Vec<f64> = (1..=n_params).map(|i| (all_klds[i] - kld0) / fd_eps).collect();
    Ok((kld0, grad))
}

// ============================================================================
// L-BFGS-B optimizer (box-constrained, no Python callbacks)
// ============================================================================

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&ai, &bi)| ai * bi).sum()
}

/// Evaluate KLD + finite-difference gradient at x, sequentially.
/// Called from within a rayon task — pure Rust, no GIL, no nested rayon.
fn eval_kld_and_grad_seq(
    bio_model: &str,
    x: &[f64],
    limits: &[usize],
    u_idx: &[u64],
    s_idx: &[u64],
    f_data: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    samp_log: Option<&[f64]>,
    eps: f64,
) -> (f64, Vec<f64>) {
    let l1 = limits[1];
    let n = x.len();
    let pss0 = eval_model_pss_2d_seq(bio_model, x, limits, fixed_quad_t, quad_order, samp_log);
    let kld0 = compute_kld_sparse(&pss0, l1, u_idx, s_idx, f_data, eps);
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            let pss = eval_model_pss_2d_seq(bio_model, &x_eps, limits, fixed_quad_t, quad_order, samp_log);
            let kld = compute_kld_sparse(&pss, l1, u_idx, s_idx, f_data, eps);
            (kld - kld0) / fd_eps
        })
        .collect();
    (kld0, grad)
}

/// Minimize KLD(x) s.t. lb ≤ x ≤ ub using L-BFGS-B.
///
/// Uses Armijo sufficient-decrease backtracking line search with box projection.
/// L-BFGS memory vectors are stored in circular buffers of size `m_mem`.
/// Convergence: projected-gradient inf-norm < gtol OR relative f-change < ftol.
///
/// Called from within rayon tasks — pure Rust, no GIL, no nested rayon.
fn lbfgsb_minimize(
    bio_model: &str,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    limits: &[usize],
    u_idx: &[u64],
    s_idx: &[u64],
    f_data: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    samp_log: Option<&[f64]>,
    eps: f64,
    m_mem: usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();

    // Project initial point onto box constraints.
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();
    let (mut fx, mut gx) = eval_kld_and_grad_seq(
        bio_model, &x, limits, u_idx, s_idx, f_data,
        fixed_quad_t, quad_order, fd_eps, samp_log, eps,
    );

    // L-BFGS memory: s[k] = x_{k+1} - x_k,  y[k] = g_{k+1} - g_k
    let mut s_buf: Vec<Vec<f64>> = Vec::with_capacity(m_mem);
    let mut y_buf: Vec<Vec<f64>> = Vec::with_capacity(m_mem);
    let mut rho_buf: Vec<f64> = Vec::with_capacity(m_mem);

    for _iter in 0..maxiter {
        // --- Projected-gradient convergence check ---------------------------
        let pg_inf: f64 = (0..n)
            .map(|i| {
                if (x[i] - lb[i]).abs() < 1e-15 {
                    gx[i].min(0.0).abs()
                } else if (x[i] - ub[i]).abs() < 1e-15 {
                    gx[i].max(0.0).abs()
                } else {
                    gx[i].abs()
                }
            })
            .fold(0.0_f64, f64::max);
        if pg_inf < gtol {
            break;
        }

        // --- L-BFGS two-loop recursion: compute H_k * g_k -------------------
        let mem = s_buf.len();
        let mut q = gx.clone();
        let mut alphas = vec![0.0_f64; mem];

        // First loop (most recent pair first).
        for i in (0..mem).rev() {
            let a = rho_buf[i] * dot(&s_buf[i], &q);
            alphas[i] = a;
            for j in 0..n {
                q[j] -= a * y_buf[i][j];
            }
        }

        // Initial Hessian scaling: γ = s^T y / y^T y (most recent pair).
        let gamma = if mem > 0 {
            let sy = dot(&s_buf[mem - 1], &y_buf[mem - 1]);
            let yy = dot(&y_buf[mem - 1], &y_buf[mem - 1]);
            if yy > 1e-30 { sy / yy } else { 1.0 }
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

        // Second loop (oldest pair first).
        for i in 0..mem {
            let beta = rho_buf[i] * dot(&y_buf[i], &r);
            for j in 0..n {
                r[j] += s_buf[i][j] * (alphas[i] - beta);
            }
        }

        // --- Search direction: d = -H*g, zeroing components at active bounds -
        let mut d: Vec<f64> = (0..n)
            .map(|i| {
                let di = -r[i];
                if (x[i] - lb[i]).abs() < 1e-15 && di < 0.0 { 0.0 }
                else if (x[i] - ub[i]).abs() < 1e-15 && di > 0.0 { 0.0 }
                else { di }
            })
            .collect();

        // Reset memory and fall back to projected steepest descent if d ≈ 0.
        if dot(&d, &d) < 1e-30 {
            s_buf.clear();
            y_buf.clear();
            rho_buf.clear();
            d = (0..n)
                .map(|i| {
                    let di = -gx[i];
                    if (x[i] - lb[i]).abs() < 1e-15 && di < 0.0 { 0.0 }
                    else if (x[i] - ub[i]).abs() < 1e-15 && di > 0.0 { 0.0 }
                    else { di }
                })
                .collect();
            if dot(&d, &d) < 1e-30 {
                break;
            }
        }

        // --- Backtracking line search (standard Armijo sufficient decrease) ----
        // Condition: f(clip(x + α·d)) ≤ f(x) + c1 · α · ∇f(x)ᵀd
        //
        // We use α · g^T · d (the UNCONSTRAINED directional derivative scaled by α),
        // NOT g^T · (x_new - x) (the actual clipped step).  When α·d clips to a
        // boundary, g^T · (x_new - x) ≪ α · g^T · d, making the latter much more
        // stringent.  Using the unconstrained form prevents accepting large steps
        // that jump to boundary local minima — matching scipy's Fortran L-BFGS-B.
        const C1: f64 = 1e-4;
        let phi_prime_0: f64 = dot(&gx, &d); // g^T · d  (must be < 0 for descent)
        let mut alpha = 1.0_f64;
        let mut x_new: Vec<f64>;
        let mut fx_new: f64;
        let mut gx_new: Vec<f64>;

        loop {
            x_new = (0..n).map(|i| (x[i] + alpha * d[i]).max(lb[i]).min(ub[i])).collect();
            let r = eval_kld_and_grad_seq(
                bio_model, &x_new, limits, u_idx, s_idx, f_data,
                fixed_quad_t, quad_order, fd_eps, samp_log, eps,
            );
            fx_new = r.0;
            gx_new = r.1;
            // Standard Armijo: f_new ≤ f + c1 · α · phi'(0)
            // phi_prime_0 < 0, so the threshold decreases proportionally with α.
            if fx_new <= fx + C1 * alpha * phi_prime_0 || alpha < 1e-12 {
                break;
            }
            alpha *= 0.5;
        }

        // --- Convergence by relative function improvement -------------------
        let f_improve = (fx - fx_new).abs() / fx.abs().max(1.0);

        // --- Update L-BFGS memory -------------------------------------------
        let sk: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let yk: Vec<f64> = (0..n).map(|i| gx_new[i] - gx[i]).collect();
        let sy = dot(&sk, &yk);

        x = x_new;
        fx = fx_new;
        gx = gx_new;

        // Curvature condition: only add pair when s·y > ε·‖y‖² (skip near-flat).
        let yy = dot(&yk, &yk);
        if sy > 1e-10 * yy.max(1e-60) {
            if s_buf.len() == m_mem {
                s_buf.remove(0);
                y_buf.remove(0);
                rho_buf.remove(0);
            }
            s_buf.push(sk);
            y_buf.push(yk);
            rho_buf.push(1.0 / sy);
        }

        if f_improve < ftol {
            break;
        }
    }

    (x, fx)
}

/// Optimize a single gene with L-BFGS-B using multiple restarts.
///
/// x0_list : num_restarts × n_params initial points (log10 scale)
/// lb / ub : parameter bounds (log10 scale)
/// Returns (x_opt, kld_min).
#[pyfunction]
#[pyo3(signature = (bio_model, x0_list, lb, ub, limits, u_idx, s_idx, f,
                    fixed_quad_t, quad_order, fd_eps=1e-6, maxiter=1000,
                    ftol=1e-10, gtol=1e-6, samp_log=None, eps=1e-15, m_lbfgs=10))]
#[allow(clippy::too_many_arguments)]
fn optimize_gene_2d(
    py: Python<'_>,
    bio_model: String,
    x0_list: Vec<Vec<f64>>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    limits: Vec<usize>,
    u_idx: Vec<u64>,
    s_idx: Vec<u64>,
    f: Vec<f64>,
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    samp_log: Option<Vec<f64>>,
    eps: f64,
    m_lbfgs: usize,
) -> PyResult<(Vec<f64>, f64)> {
    match bio_model.as_str() {
        "Constitutive" | "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => {}
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for optimize_gene_2d: {other}"
        ))),
    }
    const ERR_THRESH: f64 = 0.99;
    let result = py.allow_threads(|| {
        let mut best_x: Vec<f64> = (0..lb.len())
            .map(|i| x0_list[0][i].max(lb[i]).min(ub[i]))
            .collect();
        let mut best_kld = f64::INFINITY;
        for x0 in &x0_list {
            let (x_opt, kld) = lbfgsb_minimize(
                &bio_model, x0, &lb, &ub, &limits,
                &u_idx, &s_idx, &f,
                fixed_quad_t, quad_order, fd_eps,
                maxiter, ftol, gtol,
                samp_log.as_deref(), eps, m_lbfgs,
            );
            if kld < best_kld * ERR_THRESH {
                best_x = x_opt;
                best_kld = kld;
            }
        }
        (best_x, best_kld)
    });
    Ok(result)
}

/// Optimize all genes in parallel using L-BFGS-B + rayon.
///
/// x0_list : n_genes × num_restarts × n_params initial points
/// Returns (param_estimates, klds) each of length n_genes.
#[pyfunction]
#[pyo3(signature = (bio_model, x0_list, lb, ub, limits_list, u_idx_list, s_idx_list, f_list,
                    fixed_quad_t, quad_order, fd_eps=1e-6, maxiter=1000,
                    ftol=1e-10, gtol=1e-6, samp_list=None, eps=1e-15,
                    m_lbfgs=10, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn optimize_genes_2d(
    py: Python<'_>,
    bio_model: String,
    x0_list: Vec<Vec<Vec<f64>>>,   // n_genes × num_restarts × n_params
    lb: Vec<f64>,
    ub: Vec<f64>,
    limits_list: Vec<Vec<usize>>,
    u_idx_list: Vec<Vec<u64>>,
    s_idx_list: Vec<Vec<u64>>,
    f_list: Vec<Vec<f64>>,
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    eps: f64,
    m_lbfgs: usize,
    num_threads: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    match bio_model.as_str() {
        "Constitutive" | "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => {}
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for optimize_genes_2d: {other}"
        ))),
    }
    let n_genes = x0_list.len();
    const ERR_THRESH: f64 = 0.99;

    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        let run = || {
            (0..n_genes)
                .into_par_iter()
                .map(|gi| {
                    let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                    let mut best_x: Vec<f64> = (0..lb.len())
                        .map(|i| x0_list[gi][0][i].max(lb[i]).min(ub[i]))
                        .collect();
                    let mut best_kld = f64::INFINITY;
                    for x0 in &x0_list[gi] {
                        let (x_opt, kld) = lbfgsb_minimize(
                            &bio_model, x0, &lb, &ub, &limits_list[gi],
                            &u_idx_list[gi], &s_idx_list[gi], &f_list[gi],
                            fixed_quad_t, quad_order, fd_eps,
                            maxiter, ftol, gtol,
                            samp, eps, m_lbfgs,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        };
        match num_threads {
            Some(nt) => rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            None => run(),
        }
    });

    let params: Vec<Vec<f64>> = results.iter().map(|(x, _)| x.clone()).collect();
    let klds: Vec<f64> = results.iter().map(|(_, k)| *k).collect();
    Ok((params, klds))
}

// ============================================================================
// PyO3 module
// ============================================================================

#[pymodule]
fn monod_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_model_pss_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_model_pss_2d_batch, m)?)?;
    m.add_function(wrap_pyfunction!(eval_model_pss_protein_bursty, m)?)?;
    m.add_function(wrap_pyfunction!(protein_bursty_pgf, m)?)?;
    m.add_function(wrap_pyfunction!(eval_kld_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_kld_grad_2d, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_gene_2d, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_genes_2d, m)?)?;
    m.add_function(wrap_pyfunction!(make_histograms_unique, m)?)?;
    m.add_function(wrap_pyfunction!(eval_custom_network_pgf, m)?)?;
    Ok(())
}
