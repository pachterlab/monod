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

use lbfgsb_rs_pure::{IterationControl, LBFGSB};
use num_complex::Complex64;
#[cfg(feature = "ruanndata")]
use ruanndata::{read_h5ad, write_h5ad, ArrayData, ArrayValue, MatrixData, RuAnnData, SeriesData, UnsValue};
use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use ndarray::Array3;
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

/// Bernoulli-transformed 2-D mesh cache — keyed by (l0, l1, s0_bits, s1_bits).
/// The Bernoulli transform is a simple scalar multiply: g_i *= s_i.
/// s0 and s1 are the raw sampling parameters (log10 capture rates as stored in
/// regressor_optimum — used directly without 10^ conversion, matching Python).
static MESH_CACHE_BERNOULLI: OnceLock<Mutex<HashMap<(usize, usize, u64, u64), Arc<(Vec<Complex64>, Vec<Complex64>)>>>> =
    OnceLock::new();

fn build_mesh_2d_bernoulli_cached(
    l0: usize,
    l1: usize,
    s0: f64,
    s1: f64,
) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    let key = (l0, l1, s0.to_bits(), s1.to_bits());
    let cache = MESH_CACHE_BERNOULLI.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(mesh) = map.get(&key) {
        return Arc::clone(mesh);
    }
    let base = build_mesh_2d_cached(l0, l1);
    let cs0 = Complex64::new(s0, 0.0);
    let cs1 = Complex64::new(s1, 0.0);
    let g0t: Vec<Complex64> = base.0.iter().map(|&z| z * cs0).collect();
    let g1t: Vec<Complex64> = base.1.iter().map(|&z| z * cs1).collect();
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
// Bursty/CIR analytic gradient — single quadrature pass for phi + dphi
// ============================================================================

/// Compute log-PGF (phi) and its gradient w.r.t. log10 parameters in one
/// Gauss-Legendre quadrature pass.
///
/// Returns [phi, dphi_db, dphi_dbeta, dphi_dgamma] as four Vec<Complex64>.
/// For Bursty: integrand = U/(1-U), d(integrand)/dU = 1/(1-U)^2
/// For CIR:    integrand = (1-sqrt(1-4U))/2, d(integrand)/dU = 1/sqrt(1-4U)
/// d phi / d log10(param) = param * ln10 * ∫ d(integrand)/dU * dU/d(param) dt
fn pgf_bursty_cir_with_dphi(
    bio_model: &str,
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    t: f64,
    quad_order: usize,
) -> [Vec<Complex64>; 4] {
    let (b, beta, gamma) = (p[0], p[1], p[2]);
    let ln10 = f64::ln(10.0);
    let one = Complex64::new(1.0, 0.0);
    let four = Complex64::new(4.0, 0.0);
    let is_cir = bio_model == "CIR";
    let gl = gauss_legendre_cached(quad_order);
    let t_half = t / 2.0;
    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };
    // Precompute per-quadrature-point values (shared across grid points)
    let xs: Vec<f64>  = gl.0.iter().map(|&xi| t_half + t_half * xi).collect();
    let eb: Vec<f64>  = xs.iter().map(|&x| (-beta  * x).exp()).collect();
    let eg: Vec<f64>  = xs.iter().map(|&x| (-gamma * x).exp()).collect();
    let ws: Vec<f64>  = gl.1.iter().map(|&wi| wi * t_half).collect();
    let nq = xs.len();
    // Scale factors for dphi_dbeta and dphi_dgamma (these have no k-dependence)
    let bg2 = if !close { gamma / (beta - gamma).powi(2) } else { 0.0 };
    let bb2 = if !close { beta  / (beta - gamma).powi(2) } else { 0.0 };

    let compute = |k: usize| {
        let mut phi_k        = Complex64::new(0.0, 0.0);
        let mut d_phi_db_k   = Complex64::new(0.0, 0.0);
        let mut d_phi_dbeta_k  = Complex64::new(0.0, 0.0);
        let mut d_phi_dgamma_k = Complex64::new(0.0, 0.0);

        if close {
            // U = b * (g0 * exp(-beta*t) + g1 * beta * t * exp(-gamma*t))
            for q in 0..nq {
                let u = (g0[k] * eb[q] + g1[k] * (xs[q] * beta * eg[q])) * b;
                let (integrand, d_intg_du) = if is_cir {
                    let sq = (one - four * u).sqrt();
                    ((one - sq) * 0.5, one / sq * 0.5)
                } else {
                    let inv = one / (one - u);
                    (u * inv, inv * inv)
                };
                phi_k += integrand * ws[q];
                // d phi / d log10(b): param*dU/d(param) = b*(U/b) = U
                d_phi_db_k += d_intg_du * u * ws[q];
                // dU/dbeta = b*(-g0*t*eb + g1*t*eg)
                let du_dbeta = (Complex64::new(-xs[q], 0.0) * g0[k] * eb[q]
                              + Complex64::new( xs[q], 0.0) * g1[k] * eg[q]) * b;
                d_phi_dbeta_k += d_intg_du * du_dbeta * ws[q];
                // dU/dgamma = -b*g1*beta*t^2*eg
                let du_dgamma = g1[k] * Complex64::new(-beta * xs[q] * xs[q] * eg[q] * b, 0.0);
                d_phi_dgamma_k += d_intg_du * du_dgamma * ws[q];
            }
        } else {
            let c2k = g1[k] * f_factor;
            let c1k = g0[k] - c2k;
            for q in 0..nq {
                let u = (c1k * eb[q] + c2k * eg[q]) * b;
                let (integrand, d_intg_du) = if is_cir {
                    let sq = (one - four * u).sqrt();
                    ((one - sq) * 0.5, one / sq * 0.5)
                } else {
                    let inv = one / (one - u);
                    (u * inv, inv * inv)
                };
                phi_k += integrand * ws[q];
                d_phi_db_k += d_intg_du * u * ws[q];
                // dU/dbeta = b * [g1*gamma/(beta-gamma)^2 * (eb-eg) - c1*t*eb]
                let du_dbeta = (g1[k] * bg2 * (eb[q] - eg[q]) - c1k * xs[q] * eb[q]) * b;
                d_phi_dbeta_k += d_intg_du * du_dbeta * ws[q];
                // dU/dgamma = b * [g1*beta/(beta-gamma)^2 * (eg-eb) - c2*t*eg]
                let du_dgamma = (g1[k] * bb2 * (eg[q] - eb[q]) - c2k * xs[q] * eg[q]) * b;
                d_phi_dgamma_k += d_intg_du * du_dgamma * ws[q];
            }
        }
        // Apply param * LN10 factors
        (
            phi_k,
            d_phi_db_k   * Complex64::new(ln10,        0.0),  // b*LN10 * integral(d_intg/dU * U/b) = LN10 * integral(...)
            d_phi_dbeta_k  * Complex64::new(beta  * ln10, 0.0),
            d_phi_dgamma_k * Complex64::new(gamma * ln10, 0.0),
        )
    };

    let n = g0.len();
    let tuples: Vec<(Complex64, Complex64, Complex64, Complex64)> = if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(compute).collect()
    } else {
        (0..n).map(compute).collect()
    };
    let mut phi        = Vec::with_capacity(n);
    let mut dphi_db    = Vec::with_capacity(n);
    let mut dphi_dbeta = Vec::with_capacity(n);
    let mut dphi_dgamma = Vec::with_capacity(n);
    for (a, b_, c, d) in tuples {
        phi.push(a); dphi_db.push(b_); dphi_dbeta.push(c); dphi_dgamma.push(d);
    }
    [phi, dphi_db, dphi_dbeta, dphi_dgamma]
}

/// Evaluate KLD + analytic gradient for Bursty or CIR.
/// Called from within rayon tasks — pure Rust, no GIL, no nested rayon.
fn eval_kld_and_grad_analytic_seq(
    bio_model: &str,
    x: &[f64],
    limits: &[usize],
    u_idx: &[u64],
    s_idx: &[u64],
    f_data: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<&[f64]>,
    eps: f64,
    seq_model: &str,
) -> (f64, Vec<f64>) {
    let p: Vec<f64> = x.iter().map(|&v| 10.0_f64.powf(v)).collect();
    let (l0, l1) = (limits[0], limits[1]);
    let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);

    // Build mesh with the appropriate sequencing model transform.
    let mesh = match (samp_log, seq_model) {
        (Some(samp), "Poisson") => {
            build_mesh_2d_poisson_cached(l0, l1, 10f64.powf(samp[0]), 10f64.powf(samp[1]))
        }
        (Some(samp), "Bernoulli") => {
            build_mesh_2d_bernoulli_cached(l0, l1, samp[0], samp[1])
        }
        _ => build_mesh_2d_cached(l0, l1),
    };
    let (g0, g1) = (&mesh.0, &mesh.1);

    // Single quadrature pass: phi and all three dphi.
    let [phi, dphi_db, dphi_dbeta, dphi_dgamma] =
        pgf_bursty_cir_with_dphi(bio_model, g0, g1, &p, t, quad_order);

    // G = exp(phi), PSS_unnorm = irfft2d(G), norm = sum(|pss_unnorm|)
    let n_freq = phi.len();
    let g_exp: Vec<Complex64> = phi.iter().map(|z| z.exp()).collect();
    let pss_unnorm = irfftn_2d(&g_exp, l0, l1);
    let norm: f64 = pss_unnorm.iter().map(|v| v.abs()).sum();
    if norm == 0.0 {
        // Degenerate — fall back to large KLD, zero grad.
        return (f64::MAX / 2.0, vec![0.0; x.len()]);
    }
    // Normalized PSS for KLD (take abs, same as eval_model_pss_2d_seq).
    let pss: Vec<f64> = pss_unnorm.iter().map(|v| v.abs() / norm).collect();
    let kld = compute_kld_sparse(&pss, l1, u_idx, s_idx, f_data, eps);

    // Gradient: for each parameter i, compute dR_i = irfft2d(G * dphi_i).real
    let dphi_list: [&Vec<Complex64>; 3] = [&dphi_db, &dphi_dbeta, &dphi_dgamma];
    let n_data = u_idx.len();
    let grad: Vec<f64> = dphi_list.iter().map(|dphi_i| {
        // dG_i = G .* dphi_i
        let dg_i: Vec<Complex64> = g_exp.iter().zip(dphi_i.iter()).map(|(g, d)| g * d).collect();
        let dr_i = irfftn_2d(&dg_i, l0, l1);
        let dn_i: f64 = dr_i.iter().sum();
        // Gradient accumulation over data points.
        let sum_term: f64 = (0..n_data).map(|j| {
            let u = u_idx[j] as usize;
            let s = s_idx[j] as usize;
            let dr_j = dr_i[u * l1 + s];
            let pss_unnorm_j = pss_unnorm[u * l1 + s].max(eps * norm);
            f_data[j] * dr_j / pss_unnorm_j
        }).sum();
        -sum_term + dn_i / norm
    }).collect();

    (kld, grad)
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
    // Cap num_tsteps so extreme params (tiny dt, huge t_max) can't cause multi-million
    // step loops during optimizer exploration.  50_000 covers all realistic rate ranges
    // (typical params give O(100–10_000) steps) while keeping gradients finite.
    let num_tsteps = ((t_max / dt).ceil() as usize).min(50_000);
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
    // max_while caps iterations so extreme parameters don't loop forever.
    let max_while = 10 * num_tsteps + 10_000;
    let phase2_step = |s: &mut (Complex64, Complex64, Complex64, Complex64)| {
        let (nu0, nu1, nu2) = rk4_step(s.0, s.1, s.2, dt, beta, gamma, k_p, gamma_p);
        s.0 = nu0; s.1 = nu1; s.2 = nu2;
        s.3 += nu0 * b / (one - nu0 * b) * dt;
    };
    let mut while_steps = 0usize;
    loop {
        let max_norm = if n_grid >= PAR_THRESHOLD {
            states.par_iter().map(|s| s.0.norm()).reduce(|| 0.0_f64, f64::max)
        } else {
            states.iter().map(|s| s.0.norm()).fold(0.0_f64, f64::max)
        };
        if max_norm < 1e-3 || while_steps >= max_while {
            break;
        }
        if n_grid >= PAR_THRESHOLD {
            states.par_iter_mut().for_each(phase2_step);
        } else {
            states.iter_mut().for_each(phase2_step);
        }
        while_steps += 1;
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
#[pyo3(signature = (bio_model, p_log, limits, fixed_quad_t, quad_order, samp_log=None, amb_model="None", amb_log=None, seq_model="Poisson"))]
fn eval_model_pss_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<Vec<f64>>,
    amb_model: &str,
    amb_log: Option<Vec<f64>>,
    seq_model: &str,
) -> PyResult<Vec<f64>> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();

    if amb_model == "None" {
        // --- 2-D path (no ambient model) ---
        let (l0, l1) = (limits[0], limits[1]);
        let mesh = match (samp_log.as_deref(), seq_model) {
            (Some(samp), "Poisson") => {
                build_mesh_2d_poisson_cached(l0, l1, 10f64.powf(samp[0]), 10f64.powf(samp[1]))
            }
            (Some(samp), "Bernoulli") => {
                build_mesh_2d_bernoulli_cached(l0, l1, samp[0], samp[1])
            }
            _ => build_mesh_2d_cached(l0, l1),
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
    seq_model: &str,
) -> Vec<f64> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let (l0, l1) = (limits[0], limits[1]);

    let mesh = match (samp_log, seq_model) {
        (Some(samp), "Poisson") => {
            build_mesh_2d_poisson_cached(l0, l1, 10f64.powf(samp[0]), 10f64.powf(samp[1]))
        }
        (Some(samp), "Bernoulli") => {
            build_mesh_2d_bernoulli_cached(l0, l1, samp[0], samp[1])
        }
        _ => build_mesh_2d_cached(l0, l1),
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
/// samp_list    : optional list of sampling params per call
///                (pass None for seq_model="None"; list entries may be None)
/// seq_model    : sequencing model ("None", "Poisson", or "Bernoulli"); default "Poisson"
#[pyfunction]
#[pyo3(signature = (bio_model, params_list, limits_list, fixed_quad_t, quad_order, samp_list=None, num_threads=None, seq_model="Poisson"))]
fn eval_model_pss_2d_batch(
    py: Python<'_>,
    bio_model: String,
    params_list: Vec<Vec<f64>>,
    limits_list: Vec<Vec<usize>>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    num_threads: Option<usize>,
    seq_model: &str,
) -> PyResult<Vec<Vec<f64>>> {
    // Validate bio_model once, before releasing the GIL.
    validate_bio_model_2d(&bio_model, "eval_model_pss_2d_batch")?;
    let seq_model = seq_model.to_owned();
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
                        &seq_model,
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

/// Fraction by which a new restart's KLD must beat the current best to be accepted.
const ERR_THRESH: f64 = 0.99;

/// Validate that a bio_model string names a supported 2-D model, returning a
/// `PyValueError` with a descriptive message if not.
fn validate_bio_model_2d(bio_model: &str, caller: &str) -> pyo3::PyResult<()> {
    match bio_model {
        "Constitutive" | "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => Ok(()),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown bio_model for {caller}: {other}"
        ))),
    }
}

// Per-thread scratch buffers reused across genes — avoids one allocation per
// gene for the sort-based fallback path.
thread_local! {
    static FLAT_BUF: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    static ORDER_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
    static DENSE_BUF: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}

/// Compute per-gene expression moments (mean, variance, covariance) in parallel.
///
/// Parameters
/// ----------
/// layers     : ordered list of 2-D int64 numpy arrays, each (n_cells, n_genes).
/// layer_names: modality names matching the layer order (e.g. ["unspliced","spliced"]).
///
/// Returns
/// -------
/// List of length n_genes; each element is a dict with keys:
///   "MOM_{name}_mean"          — per-layer population mean  (ddof=0)
///   "MOM_{name}_var"           — per-layer population variance (ddof=0, matches numpy .var())
///   "MOM_cov_{name_i}_{name_j}"— pairwise sample covariance (ddof=1, matches numpy.cov()[0,1])
#[pyfunction]
fn compute_moments(
    py: Python<'_>,
    layers: Vec<PyReadonlyArray2<'_, i64>>,
    layer_names: Vec<String>,
) -> PyResult<Vec<std::collections::HashMap<String, f64>>> {
    if layers.is_empty() {
        return Ok(Vec::new());
    }
    if layers.len() != layer_names.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "layers and layer_names must have the same length",
        ));
    }
    let n_layers = layers.len();
    let n_cells = layers[0].shape()[0];
    let n_genes = layers[0].shape()[1];

    // Transpose each (n_cells, n_genes) row-major numpy array to column-major [g * n_cells + c].
    let flat_layers: Vec<Vec<f64>> = layers
        .iter()
        .map(|arr| {
            let slice = arr.as_slice().expect("layer array must be C-contiguous");
            let mut col_major = vec![0.0f64; n_cells * n_genes];
            for c in 0..n_cells {
                for g in 0..n_genes {
                    col_major[g * n_cells + c] = slice[c * n_genes + g] as f64;
                }
            }
            col_major
        })
        .collect();

    let result =
        py.allow_threads(|| compute_moments_inner(&flat_layers, &layer_names, n_cells, n_genes));

    Ok(result)
}

// ============================================================================
// SearchData — pure-Rust Stage 3 container
// ============================================================================

/// Helper: compute moments from column-major f64 layer data.
/// `layers_flat[l]` is column-major: index `g * n_cells + c`.
fn compute_moments_inner(
    layers_flat: &[Vec<f64>],
    layer_names: &[String],
    n_cells: usize,
    n_genes: usize,
) -> Vec<HashMap<String, f64>> {
    let n_layers = layers_flat.len();
    (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let mut dict = HashMap::new();
            let mut means = Vec::with_capacity(n_layers);
            for (l, flat) in layers_flat.iter().enumerate() {
                let col = &flat[g * n_cells .. (g + 1) * n_cells];
                let mean = col.iter().sum::<f64>() / n_cells as f64;
                let var  = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
                    / n_cells as f64;
                dict.insert(format!("MOM_{}_mean", layer_names[l]), mean);
                dict.insert(format!("MOM_{}_var",  layer_names[l]), var);
                means.push(mean);
            }
            for i in 0..n_layers {
                for j in (i + 1)..n_layers {
                    let col_i = &layers_flat[i][g * n_cells .. (g + 1) * n_cells];
                    let col_j = &layers_flat[j][g * n_cells .. (g + 1) * n_cells];
                    let cov = col_i.iter().zip(col_j.iter())
                        .map(|(xi, xj)| (xi - means[i]) * (xj - means[j]))
                        .sum::<f64>()
                        / (n_cells as f64 - 1.0);
                    dict.insert(
                        format!("MOM_cov_{}_{}", layer_names[i], layer_names[j]),
                        cov,
                    );
                }
            }
            dict
        })
        .collect()
}

// ============================================================================
// Method-of-Moments parameter initialisation
// ============================================================================

/// Compute Method-of-Moments (MoM) log10 parameter estimates for one gene.
///
/// Mirrors `cme_toolbox.CMEModel.get_MoM`.  Returns log10 estimates clipped to
/// [lb_log, ub_log].  Falls back to the midpoint of that range when the formula
/// produces non-finite values.
///
/// * `samp_lin` — sampling parameters already in **linear** space (10^samp).
///   Pass `None` when `seq_model == "None"`.
fn mom_x0_inner(
    bio_model:  &str,
    seq_model:  &str,
    amb_model:  &str,
    moments:    &HashMap<String, f64>,
    lb_log:     &[f64],
    ub_log:     &[f64],
    samp_lin:   Option<&[f64]>,
) -> Vec<f64> {
    let get = |key: &str| moments.get(key).copied().unwrap_or(1.0);

    let lb: Vec<f64> = lb_log.iter().map(|&x| 10f64.powf(x)).collect();
    let ub: Vec<f64> = ub_log.iter().map(|&x| 10f64.powf(x)).collect();

    let u_mean = get("MOM_unspliced_mean");
    let u_var  = get("MOM_unspliced_var");
    let s_mean = get("MOM_spliced_mean");

    // Helper: map non-finite (div-by-zero, NaN) → f64::INFINITY so the final
    // clamp brings it to ub[j], matching Python's numpy behaviour where
    // 1.0/0.0 == inf and np.clip(inf, lb, ub) == ub.
    let finite_or_inf = |v: f64| if v.is_nan() { f64::INFINITY } else { v };

    let mut x0: Vec<f64> = match bio_model {
        "Bursty" | "CIR" => {
            let mut b = if u_mean > 0.0 { u_var / u_mean - 1.0 } else { 1.0 };
            if !b.is_finite() { b = 1.0; }
            if let Some(samp) = samp_lin {
                match seq_model {
                    "Bernoulli" => { b /= samp[0]; }
                    "Poisson"   => { b = b / samp[0] - 1.0; }
                    _ => {}
                }
            }
            b = b.clamp(lb[0], ub[0]);
            let beta  = finite_or_inf(b / u_mean);  // inf → clamp → ub
            let gamma = finite_or_inf(b / s_mean);
            vec![b, beta, gamma]
        }
        "ProteinBursty" => {
            let p_mean = get("MOM_protein_mean");
            let up_cov = get("MOM_cov_unspliced_protein");
            let mut b = if u_mean > 0.0 { u_var / u_mean - 1.0 } else { 1.0 };
            if !b.is_finite() { b = 1.0; }
            if let Some(samp) = samp_lin {
                match seq_model {
                    "Bernoulli" => { b /= samp[0]; }
                    "Poisson"   => { b = b / samp[0] - 1.0; }
                    _ => {}
                }
            }
            b = b.clamp(lb[0], ub[0]);
            let beta  = finite_or_inf(b / u_mean);
            let gamma = finite_or_inf(b / s_mean);
            let r = finite_or_inf(p_mean * gamma / b);
            let denom = b * b * r - up_cov * (beta + gamma);
            let gamma_p = if denom.abs() > 1e-30 {
                finite_or_inf(up_cov * (beta + gamma) * beta / denom)
            } else {
                f64::INFINITY
            };
            let gamma_p = gamma_p.clamp(lb[4], ub[4]);
            let k_p     = (r * gamma_p).clamp(lb[3], ub[3]);
            vec![b, beta, gamma, k_p, gamma_p]
        }
        "Delay" => {
            let raw_b = if u_mean > 0.0 { u_var / u_mean - 1.0 } else { 1.0 };
            let mut b = if raw_b.is_finite() { raw_b } else { 1.0 };
            if let Some(samp) = samp_lin {
                match seq_model {
                    "Bernoulli" => { b /= samp[0]; }
                    "Poisson"   => { b = b / samp[0] - 1.0; }
                    _ => {}
                }
            }
            b = b.clamp(lb[0], ub[0]);
            let beta   = finite_or_inf(b / u_mean);
            let tauinv = finite_or_inf(b / s_mean);
            vec![b, beta, tauinv]
        }
        "DelayedSplicing" => {
            let raw_b = if u_mean > 0.0 { (u_var / u_mean - 1.0) / 2.0 } else { 1.0 };
            let b = if raw_b.is_finite() { raw_b } else { 1.0 };
            let b = b.clamp(lb[0], ub[0]);
            let tauinv = finite_or_inf(b / u_mean);
            let gamma  = finite_or_inf(b / s_mean);
            vec![b, tauinv, gamma]
        }
        "Constitutive" => {
            let beta  = finite_or_inf(1.0 / u_mean);
            let gamma = finite_or_inf(1.0 / s_mean);
            vec![beta, gamma]
        }
        "Extrinsic" => {
            let alpha = {
                let raw = if seq_model == "Poisson" {
                    let samp0 = samp_lin.map(|s| s[0]).unwrap_or(1.0);
                    u_mean * u_mean / (u_var - u_mean * (1.0 + samp0))
                } else {
                    u_mean * u_mean / (u_var - u_mean)
                };
                finite_or_inf(raw)
            };
            let beta  = finite_or_inf(alpha / u_mean);
            let gamma = finite_or_inf(alpha / s_mean);
            vec![alpha, beta, gamma]
        }
        _ => {
            // Unknown model: return midpoint
            return lb_log.iter().zip(ub_log.iter())
                .map(|(&l, &u)| (l + u) / 2.0)
                .collect();
        }
    };

    // Seq-model scaling of rate parameters (mirrors Python x0[1:] *= samp)
    if let Some(samp) = samp_lin {
        if matches!(seq_model, "Bernoulli" | "Poisson") {
            match bio_model {
                "Constitutive" => {
                    for (xi, si) in x0.iter_mut().zip(samp.iter()) { *xi *= si; }
                }
                "ProteinBursty" => {
                    // x0[[1,2]] *= samp[:2]; x0[-1] *= samp[2]/samp[1]
                    if samp.len() > 0 { x0[1] *= samp[0]; }
                    if samp.len() > 1 { x0[2] *= samp[1]; }
                    if samp.len() > 2 && samp[1] > 0.0 {
                        let last = x0.len() - 1;
                        x0[last] *= samp[2] / samp[1];
                    }
                }
                _ => {
                    // x0[1:] *= samp
                    for (xi, si) in x0[1..].iter_mut().zip(samp.iter()) { *xi *= si; }
                }
            }
        }
    }

    // Ambiguity model appends
    match amb_model {
        "Equal"   => { x0.push(0.1); }
        "Unequal" => { x0.push(0.1); x0.push(0.1); }
        _ => {}
    }

    // Clip all parameters to [lb, ub] in linear space
    let n_params = x0.len().min(lb.len());
    for j in 0..n_params { x0[j] = x0[j].clamp(lb[j], ub[j]); }

    // Convert to log10; fallback to midpoint on any non-finite
    let x0_log: Vec<f64> = x0.iter().map(|&v| v.log10()).collect();
    if x0_log.iter().any(|v| !v.is_finite()) {
        return lb_log.iter().zip(ub_log.iter()).map(|(&l, &u)| (l + u) / 2.0).collect();
    }
    x0_log
}

/// Pure-Rust container for model-ready data (Stage 3).
///
/// Holds histogram data, grid limits, moments, raw layers, and metadata as
/// Rust-native types.  All attributes are accessible from Python via properties
/// that return the same types as the legacy Python `SearchData` class, so
/// existing inference and visualisation code works without changes.
///
/// Construction
/// ------------
/// ```python
/// sd = monod_core.SearchData(
///     layers,          # list of (n_cells, n_genes) int64 C-contiguous arrays
///     layer_names,     # list of modality name strings
///     limits,          # (n_layers, n_genes) int64 numpy array  (= M)
///     coords,          # list[list[list[int]]]  from make_state_dist
///     freqs,           # list[list[float]]      from make_state_dist
///     gene_names,      # list of gene-name strings
///     n_cells,         # int
///     hist_type,       # "unique" | "grid" | "none"
///     gene_log_lengths=None,  # list[float] or None
///     k=None,          # int or None
///     epochs=None,     # int or None
/// )
/// ```
// ── Pure-Rust inference container ─────────────────────────────────────────────
//
// No PyO3 attributes here. Python bindings live in `PySearchData` below.
// Construction from Python-side arrays goes through `searchdata_from_arrays`.

pub struct SearchData {
    pub coords:           Vec<Vec<Vec<i64>>>,    // [gene][microstate][layer] — kept for 3-D optimizers
    pub coords_by_layer:  Vec<Vec<Vec<u64>>>,    // [layer][gene][microstate] — pre-split for 2-D optimizer
    pub freqs:            Vec<Vec<f64>>,         // [gene][microstate]
    pub limits:           Vec<usize>,            // flat [gene * n_layers + layer]
    pub moments:          Vec<HashMap<String, f64>>,
    pub layers_data:      Vec<Vec<i64>>,         // [layer][g * n_cells + c] column-major
    pub n_layers:         usize,
    pub n_cells:          usize,
    pub n_genes:          usize,
    pub gene_names:       Vec<String>,
    pub hist_type:        String,
    pub layer_names:      Vec<String>,
    pub gene_log_lengths:         Option<Vec<f64>>,
    pub gene_log_lengths_spliced: Option<Vec<f64>>,
    pub k:                        Option<usize>,
    pub epochs:                   Option<usize>,
}

/// Build the `[layer][gene][microstate]` u64 view from `[gene][microstate][layer]` i64 coords.
fn build_coords_by_layer(coords: &[Vec<Vec<i64>>], n_genes: usize, n_layers: usize) -> Vec<Vec<Vec<u64>>> {
    let mut cbl: Vec<Vec<Vec<u64>>> = vec![vec![vec![]; n_genes]; n_layers];
    for (g, gene_coords) in coords.iter().enumerate() {
        for ms in gene_coords.iter() {
            for (l, &count) in ms.iter().enumerate().take(n_layers) {
                cbl[l][g].push(count as u64);
            }
        }
    }
    cbl
}

impl SearchData {
    pub fn new(
        coords:           Vec<Vec<Vec<i64>>>,
        freqs:            Vec<Vec<f64>>,
        limits:           Vec<usize>,
        moments:          Vec<HashMap<String, f64>>,
        layers_data:      Vec<Vec<i64>>,
        n_layers:         usize,
        n_cells:          usize,
        n_genes:          usize,
        gene_names:       Vec<String>,
        hist_type:        String,
        layer_names:      Vec<String>,
        gene_log_lengths:         Option<Vec<f64>>,
        gene_log_lengths_spliced: Option<Vec<f64>>,
        k:                        Option<usize>,
        epochs:                   Option<usize>,
    ) -> Self {
        let coords_by_layer = build_coords_by_layer(&coords, n_genes, n_layers);
        Self {
            coords, coords_by_layer, freqs, limits, moments, layers_data,
            n_layers, n_cells, n_genes, gene_names, hist_type,
            layer_names, gene_log_lengths, gene_log_lengths_spliced, k, epochs,
        }
    }

    /// Clone the fields consumed by 3-D / custom-network `optimize_genes_*_sd` functions.
    /// Returns coords in `[gene][microstate][layer]`, freqs, flat limits, n_genes, n_layers.
    pub fn extract_for_parallel(
        &self,
    ) -> (Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>, Vec<usize>, usize, usize) {
        (self.coords.clone(), self.freqs.clone(), self.limits.clone(), self.n_genes, self.n_layers)
    }

    pub fn mom_x0_all_inner(
        &self,
        bio_model: &str,
        seq_model: &str,
        amb_model: &str,
        lb:        &[f64],
        ub:        &[f64],
        samp_list: Option<&[Option<Vec<f64>>]>,
    ) -> Vec<Vec<f64>> {
        (0..self.n_genes).map(|gi| {
            let samp_lin_vec: Option<Vec<f64>> = samp_list
                .and_then(|sl| sl.get(gi))
                .and_then(|s| s.as_ref())
                .map(|s| s.iter().map(|&v| 10f64.powf(v)).collect());
            mom_x0_inner(
                bio_model, seq_model, amb_model,
                &self.moments[gi], lb, ub,
                samp_lin_vec.as_deref(),
            )
        }).collect()
    }
}

// ── Thin Python wrapper ────────────────────────────────────────────────────────

#[pyclass(name = "SearchData")]
pub struct PySearchData {
    pub inner: SearchData,
    #[cfg(feature = "ruanndata")]
    pub adata: Option<RuAnnData>,
}

impl std::ops::Deref for PySearchData {
    type Target = SearchData;
    fn deref(&self) -> &SearchData { &self.inner }
}

#[pymethods]
impl PySearchData {
    // ── Simple scalar getters ──────────────────────────────────────────────

    #[getter] fn n_genes(&self)      -> usize        { self.inner.n_genes }
    #[getter] fn n_cells(&self)      -> usize        { self.inner.n_cells }
    #[getter] fn hist_type(&self)    -> &str         { &self.inner.hist_type }
    #[getter] fn layer_names(&self)  -> Vec<String>  { self.inner.layer_names.clone() }
    #[getter] fn gene_names(&self)   -> Vec<String>  { self.inner.gene_names.clone() }
    #[getter] fn k(&self)            -> Option<usize>{ self.inner.k }
    #[getter] fn epochs(&self)       -> Option<usize>{ self.inner.epochs }

    #[getter]
    fn gene_log_lengths<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.gene_log_lengths
            .as_ref()
            .map(|v| PyArray1::from_vec_bound(py, v.clone()))
    }

    #[getter]
    fn gene_log_lengths_spliced<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.gene_log_lengths_spliced
            .as_ref()
            .map(|v| PyArray1::from_vec_bound(py, v.clone()))
    }

    /// Returns M as a (n_layers, n_genes) int64 numpy array.
    #[getter(M)]
    fn get_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let nl = self.inner.n_layers;
        let rows: Vec<Vec<i64>> = (0..nl)
            .map(|l| (0..self.inner.n_genes).map(|g| self.inner.limits[g * nl + l] as i64).collect())
            .collect();
        PyArray2::from_vec2_bound(py, &rows)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Returns the raw count layers as a (n_layers, n_cells, n_genes) int64 numpy array.
    #[getter]
    fn layers<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray<i64, ndarray::Ix3>> {
        let nc = self.inner.n_cells;
        let arr = Array3::from_shape_fn(
            (self.inner.n_layers, nc, self.inner.n_genes),
            |(l, c, g)| self.inner.layers_data[l][g * nc + c],
        );
        arr.into_pyarray_bound(py)
    }

    /// Returns hist as a Python list of (coords_array, freqs_array) tuples.
    #[getter]
    fn hist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        use pyo3::types::{PyList, PyTuple};
        let items: Vec<Bound<'py, PyTuple>> = self.inner.coords
            .iter()
            .zip(self.inner.freqs.iter())
            .map(|(c, f)| -> PyResult<Bound<'py, PyTuple>> {
                let coords_arr = PyArray2::from_vec2_bound(py, c)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
                let freqs_arr = PyArray1::from_vec_bound(py, f.clone());
                Ok(PyTuple::new_bound(py, [coords_arr.into_any(), freqs_arr.into_any()]))
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new_bound(py, &items))
    }

    /// Returns moments as a Python list of dicts (one per gene).
    #[getter]
    fn moments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        use pyo3::types::{PyDict, PyList};
        let dicts: Vec<Bound<'py, PyDict>> = self.inner.moments
            .iter()
            .map(|m| -> PyResult<Bound<'py, PyDict>> {
                let d = PyDict::new_bound(py);
                for (k, v) in m { d.set_item(k, v)?; }
                Ok(d)
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new_bound(py, &dicts))
    }

    // ── Pickle / disk ──────────────────────────────────────────────────────

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDict;
        let d = PyDict::new_bound(py);
        d.set_item("coords",           &self.inner.coords)?;
        d.set_item("freqs",            &self.inner.freqs)?;
        d.set_item("limits",           &self.inner.limits)?;
        d.set_item("moments_keys", self.inner.moments.iter()
            .map(|m| m.keys().cloned().collect::<Vec<_>>()).collect::<Vec<_>>())?;
        d.set_item("moments_vals", self.inner.moments.iter()
            .map(|m| m.values().copied().collect::<Vec<_>>()).collect::<Vec<_>>())?;
        d.set_item("layers_data",      &self.inner.layers_data)?;
        d.set_item("n_layers",         self.inner.n_layers)?;
        d.set_item("n_cells",          self.inner.n_cells)?;
        d.set_item("n_genes",          self.inner.n_genes)?;
        d.set_item("gene_names",       &self.inner.gene_names)?;
        d.set_item("hist_type",        &self.inner.hist_type)?;
        d.set_item("layer_names",      &self.inner.layer_names)?;
        d.set_item("gene_log_lengths",         &self.inner.gene_log_lengths)?;
        d.set_item("gene_log_lengths_spliced", &self.inner.gene_log_lengths_spliced)?;
        d.set_item("k",                        self.inner.k)?;
        // coords_by_layer is derived from coords and not serialized.
        d.set_item("epochs",           self.inner.epochs)?;
        Ok(d)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        use pyo3::types::PyDict;
        let d = state.downcast::<PyDict>()?;
        let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            d.get_item(key)?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("missing key: {key}"))
            })
        };
        self.inner.coords  = get("coords")?.extract()?;
        self.inner.freqs   = get("freqs")?.extract()?;
        self.inner.limits  = get("limits")?.extract()?;
        let keys: Vec<Vec<String>> = get("moments_keys")?.extract()?;
        let vals: Vec<Vec<f64>>    = get("moments_vals")?.extract()?;
        self.inner.moments = keys.into_iter().zip(vals)
            .map(|(ks, vs)| ks.into_iter().zip(vs).collect())
            .collect();
        self.inner.layers_data      = get("layers_data")?.extract()?;
        self.inner.n_layers         = get("n_layers")?.extract()?;
        self.inner.n_cells          = get("n_cells")?.extract()?;
        self.inner.n_genes          = get("n_genes")?.extract()?;
        self.inner.gene_names       = get("gene_names")?.extract()?;
        self.inner.hist_type        = get("hist_type")?.extract()?;
        self.inner.layer_names      = get("layer_names")?.extract()?;
        self.inner.gene_log_lengths         = get("gene_log_lengths")?.extract()?;
        self.inner.gene_log_lengths_spliced = get("gene_log_lengths_spliced")?.extract()?;
        self.inner.k                        = get("k")?.extract()?;
        self.inner.epochs                   = get("epochs")?.extract()?;
        // Recompute derived field.
        self.inner.coords_by_layer = build_coords_by_layer(
            &self.inner.coords, self.inner.n_genes, self.inner.n_layers,
        );
        Ok(())
    }

    fn store_on_disk(&self, py: Python<'_>, inference_string: &str) -> PyResult<String> {
        let full_path = format!("{inference_string}/search_data.res");
        let state = self.__getstate__(py)?;
        let pickle = py.import_bound("pickle")?;
        let builtins = py.import_bound("builtins")?;
        let f = builtins.getattr("open")?.call1((&full_path, "wb"))?;
        pickle.call_method1("dump", (state, &f))?;
        f.call_method0("close")?;
        Ok(full_path)
    }

    #[pyo3(signature = (bio_model, seq_model, amb_model, lb, ub, samp_list=None))]
    fn mom_x0_all(
        &self,
        bio_model: String,
        seq_model: String,
        amb_model: String,
        lb:        Vec<f64>,
        ub:        Vec<f64>,
        samp_list: Option<Vec<Option<Vec<f64>>>>,
    ) -> Vec<Vec<f64>> {
        self.inner.mom_x0_all_inner(
            &bio_model, &seq_model, &amb_model,
            &lb, &ub,
            samp_list.as_deref(),
        )
    }
}

// ── Python construction helper ─────────────────────────────────────────────────
//
// Replaces `SearchData(layers, ...)` construction from Python.
// numpy → Rust conversion happens here; the pure SearchData struct never sees PyO3 types.

#[pyfunction]
#[pyo3(signature = (layers, layer_names, limits, coords, freqs, gene_names,
                    n_cells, hist_type,
                    gene_log_lengths=None, gene_log_lengths_spliced=None,
                    k=None, epochs=None))]
#[allow(clippy::too_many_arguments)]
fn searchdata_from_arrays(
    py: Python<'_>,
    layers: Vec<PyReadonlyArray2<'_, i64>>,
    layer_names: Vec<String>,
    limits: PyReadonlyArray2<'_, i64>,
    coords: Vec<Vec<Vec<i64>>>,
    freqs: Vec<Vec<f64>>,
    gene_names: Vec<String>,
    n_cells: usize,
    hist_type: String,
    gene_log_lengths: Option<Vec<f64>>,
    gene_log_lengths_spliced: Option<Vec<f64>>,
    k: Option<usize>,
    epochs: Option<usize>,
) -> PyResult<PySearchData> {
    let n_layers = layers.len();
    let n_genes = if n_layers > 0 { layers[0].shape()[1] } else { 0 };

    // Transpose each (n_cells, n_genes) row-major layer to column-major [g * n_cells + c].
    let layers_data: Vec<Vec<i64>> = layers
        .iter()
        .map(|arr| {
            let slice = arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err(
                "layer arrays must be C-contiguous",
            ))?;
            let mut col_major = vec![0i64; n_cells * n_genes];
            for c in 0..n_cells {
                for g in 0..n_genes {
                    col_major[g * n_cells + c] = slice[c * n_genes + g];
                }
            }
            Ok(col_major)
        })
        .collect::<PyResult<_>>()?;

    let limits_n_cols = limits.shape()[1];
    let limits_slice = limits.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("limits array must be C-contiguous")
    })?;
    // Flat limits: index = g * n_layers + l.
    let limits_flat: Vec<usize> = (0..n_genes)
        .flat_map(|g| (0..n_layers).map(move |l| limits_slice[l * limits_n_cols + g] as usize))
        .collect();

    // Cast to f64 in column-major order for moments computation.
    let layers_flat: Vec<Vec<f64>> = layers_data.iter()
        .map(|v| v.iter().map(|&x| x as f64).collect())
        .collect();
    let moments = py.allow_threads(|| {
        compute_moments_inner(&layers_flat, &layer_names, n_cells, n_genes)
    });

    Ok(PySearchData {
        inner: SearchData::new(
            coords, freqs, limits_flat, moments, layers_data,
            n_layers, n_cells, n_genes, gene_names, hist_type,
            layer_names, gene_log_lengths, gene_log_lengths_spliced, k, epochs,
        ),
        #[cfg(feature = "ruanndata")]
        adata: None,
    })
}

/// Run `run` on a rayon thread pool of the requested size, falling back to the
/// global pool if `num_threads` is `None` or pool creation fails.
fn run_with_pool<T, F>(num_threads: Option<usize>, run: F) -> Vec<T>
where
    T: Send,
    F: Fn() -> Vec<T> + Send + Sync,
{
    match num_threads {
        Some(nt) => rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .map(|pool| pool.install(&run))
            .unwrap_or_else(|_| run()),
        None => run(),
    }
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
fn make_state_dist(
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
// ruanndata h5ad I/O helpers  (feature-gated: only built when "ruanndata" is enabled)
// ============================================================================

/// Convert any MatrixData variant to a flat row-major (n_cells × n_genes) Vec<i64>.
/// Returns (n_cells, n_genes, flat).
#[cfg(feature = "ruanndata")]
fn matrix_to_dense_i64(matrix: &MatrixData) -> Result<(usize, usize, Vec<i64>), String> {
    /// Cast one element of an ArrayValue to i64.
    fn av_get(av: &ArrayValue, idx: usize) -> i64 {
        match av {
            ArrayValue::Int64(v)  => v[idx],
            ArrayValue::Int32(v)  => v[idx] as i64,
            ArrayValue::UInt64(v) => v[idx] as i64,
            ArrayValue::UInt32(v) => v[idx] as i64,
            ArrayValue::Float64(v) => v[idx].round() as i64,
            ArrayValue::Float32(v) => v[idx].round() as i64,
            _ => 0,
        }
    }
    match matrix {
        MatrixData::Dense { array } => {
            let shape = &array.shape;
            if shape.len() != 2 {
                return Err(format!("expected 2-D dense array, got shape {:?}", shape));
            }
            let (nr, nc) = (shape[0], shape[1]);
            let flat: Vec<i64> = (0..nr * nc).map(|i| av_get(&array.values, i)).collect();
            Ok((nr, nc, flat))
        }
        MatrixData::Csr { n_rows, n_cols, data, indices, indptr } => {
            let mut flat = vec![0i64; n_rows * n_cols];
            for row in 0..*n_rows {
                for k in indptr[row]..indptr[row + 1] {
                    let col = indices[k];
                    flat[row * n_cols + col] = av_get(&data.values, k);
                }
            }
            Ok((*n_rows, *n_cols, flat))
        }
        MatrixData::Csc { n_rows, n_cols, data, indices, indptr } => {
            let mut flat = vec![0i64; n_rows * n_cols];
            for col in 0..*n_cols {
                for k in indptr[col]..indptr[col + 1] {
                    let row = indices[k];
                    flat[row * n_cols + col] = av_get(&data.values, k);
                }
            }
            Ok((*n_rows, *n_cols, flat))
        }
    }
}


// ── Private h5ad loading helper ──────────────────────────────────────────────

/// All data extracted from an h5ad file, ready to build histograms or SearchData.
///
/// `adata_sub` is the h5ad sliced to selected genes with state distributions and
/// limits stored in `uns`:
///   - `uns["state_dist_coords"]`: `List` of per-gene Int64 arrays, shape `[n_states, n_layers]`
///   - `uns["state_dist_freqs"]`:  `List` of per-gene Float64 arrays, length `n_states`
///   - `uns["limits"]`:            Int64 array, shape `[n_genes, n_layers]` (max + padding)
///   Both state-dist keys are registered in `uns_var_keys` for correct `slice_var` behaviour.
///
/// `layers_sel` holds the dense subsetted counts column-major: `[layer][g * n_cells + c]`.
#[cfg(feature = "ruanndata")]
struct H5adInner {
    adata_sub:  RuAnnData,
    layers_sel: Vec<Vec<i64>>,
}

/// Core h5ad ingestion: read file, filter/select genes, compute histograms,
/// and build a subsetted layer matrix for downstream moments.
///
/// Must be called outside the GIL (inside `py.allow_threads`).
#[cfg(feature = "ruanndata")]
#[allow(clippy::too_many_arguments)]
fn load_h5ad_inner(
    filepath:   &str,
    layer_names: &[String],
    gene_names:  Option<&[String]>,
    min_means:   &[f64],
    max_maxes:   &[f64],
    min_maxes:   &[f64],
    padding:     usize,
) -> Result<H5adInner, String> {
    let n_layers = layer_names.len();

    // ── Read h5ad ─────────────────────────────────────────────────────────
    let adata = read_h5ad(filepath)
        .map_err(|e| format!("read_h5ad failed: {e}"))?;
    let all_gene_names: &[String] = &adata.var.index;
    let n_total_genes = all_gene_names.len();

    // ── Densify requested layers (n_cells × n_total_genes, row-major) ─────
    let mut layers_flat: Vec<Vec<i64>> = Vec::with_capacity(n_layers);
    let mut n_cells = 0usize;
    for lname in layer_names {
        let mat = adata.layers.get(lname)
            .ok_or_else(|| format!("layer '{}' not found", lname))?;
        let (nr, nc, flat) = matrix_to_dense_i64(mat)?;
        if nc != n_total_genes {
            return Err(format!(
                "layer '{}': {} columns but var has {} genes", lname, nc, n_total_genes
            ));
        }
        if layers_flat.is_empty() { n_cells = nr; }
        else if nr != n_cells {
            return Err(format!("layers have inconsistent cell counts ({} vs {})", n_cells, nr));
        }
        layers_flat.push(flat);
    }

    // ── Build (gene_name, gene_idx) list in requested order ───────────────
    let ordered: Vec<(String, usize)> = if let Some(requested) = gene_names {
        let name_to_idx: HashMap<&str, usize> = all_gene_names
            .iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
        requested.iter().map(|g| {
            let idx = name_to_idx.get(g.as_str())
                .copied()
                .ok_or_else(|| format!("gene '{}' not found in var", g))?;
            Ok((g.clone(), idx))
        }).collect::<Result<Vec<_>, String>>()?
    } else {
        let mut passing = Vec::new();
        for gene in 0..n_total_genes {
            let mut ok = true;
            for (l, flat) in layers_flat.iter().enumerate() {
                let (sum, max) = (0..n_cells).fold((0i64, 0i64), |(s, m), c| {
                    let v = flat[c * n_total_genes + gene];
                    (s + v, m.max(v))
                });
                let mean = sum as f64 / n_cells as f64;
                let max  = max as f64;
                if mean < min_means[l] || max > max_maxes[l] || max < min_maxes[l] {
                    ok = false; break;
                }
            }
            if ok { passing.push((all_gene_names[gene].clone(), gene)); }
        }
        passing
    };

    // Save column indices before the parallel iterator consumes `ordered`.
    let gene_indices: Vec<usize> = ordered.iter().map(|(_, idx)| *idx).collect();

    // ── Parallel histogram computation ────────────────────────────────────
    let results: Vec<(String, Vec<Vec<i64>>, Vec<f64>, Vec<usize>)> = ordered
        .into_par_iter()
        .map(|(gname, gene_idx)| {
            let cols: Vec<Vec<i64>> = layers_flat.iter()
                .map(|flat| (0..n_cells).map(|c| flat[c * n_total_genes + gene_idx]).collect())
                .collect();

            let max_per_layer: Vec<usize> = cols.iter()
                .map(|col| col.iter().copied().max().unwrap_or(0).max(0) as usize)
                .collect();
            let limits: Vec<usize> = max_per_layer.iter().map(|&m| m + padding).collect();
            let total_states: usize = max_per_layer.iter().map(|&m| m + 1).product();

            let (coords, freqs) = if total_states <= DENSE_THRESHOLD {
                let mut strides = vec![1usize; n_layers];
                for l in (0..n_layers - 1).rev() {
                    strides[l] = strides[l + 1] * (max_per_layer[l + 1] + 1);
                }
                DENSE_BUF.with(|db| {
                    let mut table = db.borrow_mut();
                    if table.len() < total_states { table.resize(total_states, 0); }
                    let table = &mut table[..total_states];
                    table.fill(0);
                    for cell in 0..n_cells {
                        let idx: usize = cols.iter().zip(strides.iter())
                            .map(|(col, &s)| col[cell] as usize * s).sum();
                        table[idx] += 1;
                    }
                    let mut unique = Vec::new();
                    let mut freqs  = Vec::new();
                    for flat_idx in 0..total_states {
                        if table[flat_idx] == 0 { continue; }
                        let mut ms = vec![0i64; n_layers];
                        let mut rem = flat_idx;
                        for l in 0..n_layers {
                            ms[l] = (rem / strides[l]) as i64;
                            rem %= strides[l];
                        }
                        unique.push(ms);
                        freqs.push(table[flat_idx] as f64 / n_cells as f64);
                    }
                    (unique, freqs)
                })
            } else {
                FLAT_BUF.with(|fb| { ORDER_BUF.with(|ob| {
                    let mut flat_b = fb.borrow_mut();
                    let mut order  = ob.borrow_mut();
                    let flat_len = n_cells * n_layers;
                    if flat_b.len() < flat_len { flat_b.resize(flat_len, 0); }
                    if order.len()  < n_cells  { order.resize(n_cells, 0); }
                    let flat_b = &mut flat_b[..flat_len];
                    let order  = &mut order[..n_cells];
                    for cell in 0..n_cells {
                        for (l, col) in cols.iter().enumerate() {
                            flat_b[cell * n_layers + l] = col[cell];
                        }
                    }
                    for (i, v) in order.iter_mut().enumerate() { *v = i; }
                    order.sort_unstable_by(|&a, &b| {
                        flat_b[a * n_layers..(a+1)*n_layers]
                            .cmp(&flat_b[b * n_layers..(b+1)*n_layers])
                    });
                    let mut unique: Vec<Vec<i64>> = Vec::new();
                    let mut counts: Vec<usize> = Vec::new();
                    let mut prev_start = usize::MAX;
                    for &idx in order.iter() {
                        let rs = idx * n_layers;
                        if prev_start != usize::MAX
                            && flat_b[prev_start..prev_start+n_layers]
                                == flat_b[rs..rs+n_layers]
                        {
                            *counts.last_mut().unwrap() += 1;
                        } else {
                            unique.push(flat_b[rs..rs+n_layers].to_vec());
                            counts.push(1);
                            prev_start = rs;
                        }
                    }
                    let freqs = counts.iter().map(|&c| c as f64 / n_cells as f64).collect();
                    (unique, freqs)
                })})
            };

            (gname, coords, freqs, limits)
        })
        .collect();

    let n_sel = results.len();
    let mut coords_out  = Vec::with_capacity(n_sel);
    let mut freqs_out   = Vec::with_capacity(n_sel);
    let mut limits_out  = Vec::with_capacity(n_sel);
    for (_name, coords, freqs, limits) in results {
        coords_out.push(coords);
        freqs_out.push(freqs);
        limits_out.push(limits);
    }

    // ── Build subsetted layers for moments: [layer][g * n_cells + c] column-major ──
    let mut layers_sel: Vec<Vec<i64>> = vec![vec![0i64; n_cells * n_sel]; n_layers];
    for (sel_g, &orig_g) in gene_indices.iter().enumerate() {
        for (l, full_flat) in layers_flat.iter().enumerate() {
            for c in 0..n_cells {
                layers_sel[l][sel_g * n_cells + c] = full_flat[c * n_total_genes + orig_g];
            }
        }
    }

    // ── Slice adata to selected genes and store state dists + limits in uns ─
    let mut adata_sub = adata
        .slice_var(&gene_indices)
        .map_err(|e| format!("slice_var failed: {e}"))?;

    let coords_list: Vec<UnsValue> = coords_out.iter().map(|gene_coords| {
        let n_states = gene_coords.len();
        let flat: Vec<i64> = gene_coords.iter().flat_map(|ms| ms.iter().copied()).collect();
        UnsValue::Array(ArrayData {
            shape: vec![n_states, n_layers],
            values: ArrayValue::Int64(flat),
        })
    }).collect();

    let freqs_list: Vec<UnsValue> = freqs_out.iter().map(|gene_freqs| {
        UnsValue::Array(ArrayData {
            shape: vec![gene_freqs.len()],
            values: ArrayValue::Float64(gene_freqs.clone()),
        })
    }).collect();

    let limits_flat: Vec<i64> = limits_out.iter()
        .flat_map(|gene_limits| gene_limits.iter().map(|&x| x as i64))
        .collect();

    adata_sub.uns.insert("state_dist_coords".into(), UnsValue::List(coords_list));
    adata_sub.uns.insert("state_dist_freqs".into(),  UnsValue::List(freqs_list));
    adata_sub.uns.insert("limits".into(), UnsValue::Array(ArrayData {
        shape: vec![n_sel, n_layers],
        values: ArrayValue::Int64(limits_flat),
    }));
    Ok(H5adInner { adata_sub, layers_sel })
}

// ── H5adInner accessors ───────────────────────────────────────────────────────

/// Extract (gene_names, coords, freqs, limits) from an `H5adInner` for the
/// `load_histograms_h5ad` return type.
#[cfg(feature = "ruanndata")]
fn h5ad_inner_to_histograms(
    inner: H5adInner,
    n_layers: usize,
) -> Result<(Vec<String>, Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>, Vec<Vec<usize>>), String> {
    let n_genes = inner.adata_sub.n_vars();
    let gene_names = inner.adata_sub.var.index.clone();
    let (coords, freqs) = extract_state_dists(&inner.adata_sub, n_genes, n_layers)?;
    let limits_flat = extract_limits(&inner.adata_sub, n_genes, n_layers)?;
    // Re-nest for the Python-facing return type.
    let limits_nested: Vec<Vec<usize>> = limits_flat
        .chunks(n_layers)
        .map(|c| c.to_vec())
        .collect();
    Ok((gene_names, coords, freqs, limits_nested))
}

/// Read the monod state-distribution for gene `gene_idx` from a `RuAnnData`
/// enriched by `load_h5ad_inner`.
///
/// Returns `(flat_coords, n_layers, freqs)` where `flat_coords` is shape
/// `[n_states × n_layers]` and `freqs` is length `n_states`.
#[cfg(feature = "ruanndata")]
fn get_state_dist(adata: &RuAnnData, gene_idx: usize) -> Option<(&[i64], usize, &[f64])> {
    let coords_entry = match adata.uns.get("state_dist_coords")? {
        UnsValue::List(v) => v.get(gene_idx)?,
        _ => return None,
    };
    let freqs_entry = match adata.uns.get("state_dist_freqs")? {
        UnsValue::List(v) => v.get(gene_idx)?,
        _ => return None,
    };

    let (flat_coords, n_layers) = match coords_entry {
        UnsValue::Array(arr) if arr.shape.len() == 2 => {
            if let ArrayValue::Int64(v) = &arr.values {
                (v.as_slice(), arr.shape[1])
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let freqs = match freqs_entry {
        UnsValue::Array(arr) if arr.shape.len() == 1 => {
            if let ArrayValue::Float64(v) = &arr.values {
                v.as_slice()
            } else {
                return None;
            }
        }
        _ => return None,
    };

    Some((flat_coords, n_layers, freqs))
}

/// Unpack `uns["state_dist_coords"]` and `uns["state_dist_freqs"]` into the
/// `Vec<Vec<Vec<i64>>>` / `Vec<Vec<f64>>` shapes expected by `SearchData`.
#[cfg(feature = "ruanndata")]
fn extract_state_dists(
    adata: &RuAnnData,
    n_genes: usize,
    n_layers: usize,
) -> Result<(Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>), String> {
    let mut coords = Vec::with_capacity(n_genes);
    let mut freqs  = Vec::with_capacity(n_genes);
    for g in 0..n_genes {
        let (flat_coords, n_mod, gene_freqs) = get_state_dist(adata, g)
            .ok_or_else(|| format!("state_dist missing for gene {g}"))?;
        if n_mod != n_layers {
            return Err(format!(
                "gene {g}: state_dist has {n_mod} modalities but expected {n_layers}"
            ));
        }
        let coords_g: Vec<Vec<i64>> = flat_coords.chunks(n_mod).map(|c| c.to_vec()).collect();
        coords.push(coords_g);
        freqs.push(gene_freqs.to_vec());
    }
    Ok((coords, freqs))
}

/// Unpack `uns["limits"]` (shape `[n_genes, n_layers]`) into per-gene `Vec<Vec<usize>>`.
#[cfg(feature = "ruanndata")]
fn extract_limits(
    adata: &RuAnnData,
    n_genes: usize,
    n_layers: usize,
) -> Result<Vec<usize>, String> {
    match adata.uns.get("limits") {
        Some(UnsValue::Array(arr)) => {
            if let ArrayValue::Int64(v) = &arr.values {
                // Stored as [n_genes, n_layers] row-major; flatten to [g * n_layers + l].
                Ok((0..n_genes)
                    .flat_map(|g| (0..n_layers).map(move |l| v[g * n_layers + l] as usize))
                    .collect())
            } else {
                Err("uns['limits'] values are not Int64".into())
            }
        }
        _ => Err("uns['limits'] not found or has unexpected type".into()),
    }
}

// ── Public pyfunction wrappers ────────────────────────────────────────────────

/// Read an h5ad file, apply optional expression filter, compute unique histograms.
///
/// Parameters
/// ----------
/// filepath      : path to the .h5ad file.
/// layer_names   : ordered list of layer keys (e.g. ["unspliced","spliced"]).
/// gene_names    : if Some, use exactly these genes (must be present in var.index).
///                 if None, apply expression filter and return all passing genes.
/// min_means     : per-layer minimum mean expression (default 0.01 each).
/// max_maxes     : per-layer maximum allowed peak count (default 350 each).
/// min_maxes     : per-layer minimum required peak count (default 4 each).
/// padding       : added to per-gene per-layer maximum to form grid limit (default 10).
///
/// Returns
/// -------
/// (gene_names, coords, freqs, limits) where
///   gene_names  : Vec<String> of selected gene names (in requested / filter order)
///   coords[g]   : unique microstates (each a Vec<i64> of length n_layers)
///   freqs[g]    : normalised frequencies corresponding to coords[g]
///   limits[g]   : Vec<usize> per-layer grid bound (max_val + padding)
#[cfg(feature = "ruanndata")]
#[pyfunction]
#[pyo3(signature = (filepath, layer_names, gene_names=None,
                    min_means=None, max_maxes=None, min_maxes=None, padding=10))]
#[allow(clippy::too_many_arguments)]
fn load_histograms_h5ad(
    py: Python<'_>,
    filepath: String,
    layer_names: Vec<String>,
    gene_names: Option<Vec<String>>,
    min_means: Option<Vec<f64>>,
    max_maxes: Option<Vec<f64>>,
    min_maxes: Option<Vec<f64>>,
    padding: usize,
) -> PyResult<(Vec<String>, Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>, Vec<Vec<usize>>)> {
    let n_layers = layer_names.len();
    if n_layers == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("layer_names must not be empty"));
    }
    let min_means  = min_means .unwrap_or_else(|| vec![0.01;  n_layers]);
    let max_maxes  = max_maxes .unwrap_or_else(|| vec![350.0; n_layers]);
    let min_maxes  = min_maxes .unwrap_or_else(|| vec![4.0;   n_layers]);
    if min_means.len() != n_layers || max_maxes.len() != n_layers || min_maxes.len() != n_layers {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_means, max_maxes, min_maxes must each have length == len(layer_names)",
        ));
    }

    let result = py.allow_threads(|| {
        let inner = load_h5ad_inner(
            &filepath,
            &layer_names,
            gene_names.as_deref(),
            &min_means,
            &max_maxes,
            &min_maxes,
            padding,
        )?;
        h5ad_inner_to_histograms(inner, n_layers)
    });

    result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Read an h5ad file and return a fully constructed `SearchData` object.
///
/// This is the zero-Python-roundtrip path: the file is read, genes filtered,
/// unique histograms and moments computed, all inside a single GIL-free block.
///
/// Parameters are identical to `load_histograms_h5ad`, with the addition of:
/// hist_type     : histogram type string stored on the SearchData (default "unique").
///
/// Returns
/// -------
/// A `SearchData` object ready for direct use with `optimize_genes_*_sd`.
#[cfg(feature = "ruanndata")]
#[pyfunction]
#[pyo3(signature = (filepath, layer_names, gene_names=None,
                    min_means=None, max_maxes=None, min_maxes=None, padding=10,
                    hist_type="unique", log_lengths_col=None,
                    spliced_log_lengths_col=None))]
#[allow(clippy::too_many_arguments)]
fn searchdata_from_h5ad(
    py: Python<'_>,
    filepath: String,
    layer_names: Vec<String>,
    gene_names: Option<Vec<String>>,
    min_means: Option<Vec<f64>>,
    max_maxes: Option<Vec<f64>>,
    min_maxes: Option<Vec<f64>>,
    padding: usize,
    hist_type: &str,
    log_lengths_col: Option<String>,
    spliced_log_lengths_col: Option<String>,
) -> PyResult<PySearchData> {
    let n_layers = layer_names.len();
    if n_layers == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("layer_names must not be empty"));
    }
    let min_means  = min_means .unwrap_or_else(|| vec![0.01;  n_layers]);
    let max_maxes  = max_maxes .unwrap_or_else(|| vec![350.0; n_layers]);
    let min_maxes  = min_maxes .unwrap_or_else(|| vec![4.0;   n_layers]);
    if min_means.len() != n_layers || max_maxes.len() != n_layers || min_maxes.len() != n_layers {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_means, max_maxes, min_maxes must each have length == len(layer_names)",
        ));
    }
    let hist_type = hist_type.to_string();

    let result = py.allow_threads(|| -> Result<(SearchData, RuAnnData), String> {
        let inner = load_h5ad_inner(
            &filepath,
            &layer_names,
            gene_names.as_deref(),
            &min_means,
            &max_maxes,
            &min_maxes,
            padding,
        )?;

        let n_genes = inner.adata_sub.n_vars();
        let n_cells = inner.adata_sub.n_obs();

        let layers_f64: Vec<Vec<f64>> = inner.layers_sel.iter()
            .map(|v| v.iter().map(|&x| x as f64).collect())
            .collect();
        let moments = compute_moments_inner(&layers_f64, &layer_names, n_cells, n_genes);

        let (coords, freqs) = extract_state_dists(&inner.adata_sub, n_genes, n_layers)?;
        let limits          = extract_limits(&inner.adata_sub, n_genes, n_layers)?;
        let gene_names_out  = inner.adata_sub.var.index.clone();

        let read_float64_col = |col: &str| -> Option<Vec<f64>> {
            inner.adata_sub.var.columns.get(col).and_then(|series| {
                if let SeriesData::Float64 { values } = series {
                    Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
                } else {
                    None
                }
            })
        };
        let gene_log_lengths         = log_lengths_col.as_deref().and_then(read_float64_col);
        let gene_log_lengths_spliced = spliced_log_lengths_col.as_deref().and_then(read_float64_col);

        let sd = SearchData::new(
            coords, freqs, limits, moments, inner.layers_sel,
            n_layers, n_cells, n_genes, gene_names_out, hist_type,
            layer_names.to_vec(), gene_log_lengths, gene_log_lengths_spliced, None, None,
        );
        Ok((sd, inner.adata_sub))
    });

    result
        .map(|(sd, adata_sub)| PySearchData { inner: sd, adata: Some(adata_sub) })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
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
/// samp_log     : optional log10 sampling parameters [s0, s1]
/// eps          : minimum probability floor (default 1e-15)
/// seq_model    : sequencing model ("None", "Poisson", or "Bernoulli"); default "Poisson"
///
/// Returns the scalar KLD value.
#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, u_idx, s_idx, f, fixed_quad_t, quad_order, samp_log=None, eps=1e-15, seq_model="Poisson"))]
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
    seq_model: &str,
) -> PyResult<f64> {
    let pss = eval_model_pss_2d_seq(
        bio_model,
        &p_log,
        &limits,
        fixed_quad_t,
        quad_order,
        samp_log.as_deref(),
        seq_model,
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
#[pyo3(signature = (bio_model, p_log, limits, u_idx, s_idx, f, fixed_quad_t, quad_order, fd_eps=1e-6, samp_log=None, eps=1e-15, seq_model="Poisson"))]
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
    seq_model: &str,
) -> PyResult<(f64, Vec<f64>)> {
    // Validate model before releasing GIL.
    validate_bio_model_2d(&bio_model, "eval_kld_grad_2d")?;
    let seq_model = seq_model.to_owned();
    let samp_ref: Option<Vec<f64>> = samp_log;

    // Bursty and CIR: use the semi-analytical gradient (faster and more accurate).
    if bio_model == "Bursty" || bio_model == "CIR" {
        let (kld0, grad) = py.allow_threads(|| {
            eval_kld_and_grad_analytic_seq(
                &bio_model, &p_log, &limits, &u_idx, &s_idx, &f,
                fixed_quad_t, quad_order, samp_ref.as_deref(), eps, &seq_model,
            )
        });
        return Ok((kld0, grad));
    }

    let n_params = p_log.len();
    let l1 = limits[1];
    // Other models: parallel forward FD.
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
                    &seq_model,
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
    seq_model: &str,
) -> (f64, Vec<f64>) {
    // Bursty and CIR: semi-analytical gradient (same path as the scipy callback).
    if bio_model == "Bursty" || bio_model == "CIR" {
        return eval_kld_and_grad_analytic_seq(
            bio_model, x, limits, u_idx, s_idx, f_data,
            fixed_quad_t, quad_order, samp_log, eps, seq_model,
        );
    }
    // All other models: forward finite differences.
    let l1 = limits[1];
    let n = x.len();
    let pss0 = eval_model_pss_2d_seq(bio_model, x, limits, fixed_quad_t, quad_order, samp_log, seq_model);
    let kld0 = compute_kld_sparse(&pss0, l1, u_idx, s_idx, f_data, eps);
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            let pss = eval_model_pss_2d_seq(bio_model, &x_eps, limits, fixed_quad_t, quad_order, samp_log, seq_model);
            let kld = compute_kld_sparse(&pss, l1, u_idx, s_idx, f_data, eps);
            (kld - kld0) / fd_eps
        })
        .collect();
    (kld0, grad)
}

/// Minimize KLD(x) s.t. lb ≤ x ≤ ub using L-BFGS-B.
///
/// Delegates to `lbfgsb_rs_pure::LBFGSB` which uses a Moré-Thuente safeguarded
/// line search (same as scipy's Fortran L-BFGS-B), Cauchy point + subspace
/// minimization, and compact column-major memory storage.
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
    seq_model: &str,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();

    let mut solver = LBFGSB::new(m_mem)
        .with_max_iter(maxiter)
        .with_pgtol(gtol)
        .with_ftol(ftol);

    let mut f_and_grad = |xx: &[f64]| -> (f64, Vec<f64>) {
        eval_kld_and_grad_seq(
            bio_model, xx, limits, u_idx, s_idx, f_data,
            fixed_quad_t, quad_order, fd_eps, samp_log, eps, seq_model,
        )
    };

    // ftol is now handled inside the patched solver (scipy-compatible normalization).
    // The callback is a no-op; convergence is driven by pgtol and ftol in the solver.
    let mut callback = |_info: &lbfgsb_rs_pure::IterationInfo, _x: &[f64]| {
        IterationControl::Continue
    };

    match solver.minimize_with_callback(&mut x, lb, ub, &mut f_and_grad, &mut callback) {
        Ok(sol) => (sol.x, sol.f),
        Err(_) => {
            // Line-search or numerical failure: return the projected x0 with its f value.
            let (f0, _) = f_and_grad(&x);
            (x, f0)
        }
    }
}

/// Optimize a single gene with L-BFGS-B using multiple restarts.
///
/// x0_list : num_restarts × n_params initial points (log10 scale)
/// lb / ub : parameter bounds (log10 scale)
/// Returns (x_opt, kld_min).
#[pyfunction]
#[pyo3(signature = (bio_model, x0_list, lb, ub, limits, u_idx, s_idx, f,
                    fixed_quad_t, quad_order, fd_eps=1e-6, maxiter=1000,
                    ftol=1e-10, gtol=1e-6, samp_log=None, eps=1e-15, m_lbfgs=10,
                    seq_model="Poisson"))]
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
    seq_model: &str,
) -> PyResult<(Vec<f64>, f64)> {
    validate_bio_model_2d(&bio_model, "optimize_gene_2d")?;
    let seq_model = seq_model.to_owned();

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
                samp_log.as_deref(), eps, m_lbfgs, &seq_model,
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
                    m_lbfgs=10, num_threads=None, seq_model="Poisson"))]
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
    seq_model: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    validate_bio_model_2d(&bio_model, "optimize_genes_2d")?;
    let seq_model = seq_model.to_owned();
    let n_genes = x0_list.len();

    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
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
                            samp, eps, m_lbfgs, &seq_model,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        })
    });

    let (params, klds): (Vec<Vec<f64>>, Vec<f64>) = results.into_iter().unzip();
    Ok((params, klds))
}

/// Optimize all genes in parallel using L-BFGS-B, reading histogram data directly
/// from a Rust `SearchData` — no Python marshal loop required.
///
/// Equivalent to `optimize_genes_2d` but the per-gene `u_idx`, `s_idx`, `f`, and
/// `limits` are extracted from `sd.coords`, `sd.freqs`, and `sd.limits` in Rust,
/// bypassing the Python loop that converts numpy arrays to lists.
///
/// Only supports 2-modality models (coords columns 0 and 1 are the two layers).
#[pyfunction]
#[pyo3(signature = (sd, bio_model, x0_list, lb, ub, fixed_quad_t, quad_order,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    base_samp=None, use_lengths_unspliced=false, use_lengths_spliced=false,
                    eps=1e-15, m_lbfgs=10, num_threads=None, seq_model="Poisson"))]
#[allow(clippy::too_many_arguments)]
fn optimize_genes_2d_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    bio_model: String,
    x0_list: Vec<Vec<Vec<f64>>>,   // n_genes × num_restarts × n_params
    lb: Vec<f64>,
    ub: Vec<f64>,
    fixed_quad_t: f64,
    quad_order: usize,
    fd_eps: f64,
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    base_samp: Option<Vec<f64>>,   // grid-point sampling params (log10); length offset added per-gene if use_lengths_*
    use_lengths_unspliced: bool,    // if true, add gene_log_lengths[gi] to base_samp[0] per gene
    use_lengths_spliced: bool,      // if true, add gene_log_lengths[gi] to base_samp[1] per gene
    eps: f64,
    m_lbfgs: usize,
    num_threads: Option<usize>,
    seq_model: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    validate_bio_model_2d(&bio_model, "optimize_genes_2d_sd")?;
    let seq_model = seq_model.to_owned();

    // Extract all needed data from the Rust struct while holding the GIL borrow.
    // We collect into owned Vecs so they are Send and can cross the allow_threads boundary.
    let (u_idx_list, s_idx_list, f_list, limits_flat, n_lim, n_genes, gene_log_lengths, gene_log_lengths_spliced) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optimize_genes_2d_sd requires at least 2 layers (2D models only)",
            ));
        }
        let n    = sd_ref.n_genes;
        let n_lm = sd_ref.n_layers;
        // Direct clones from pre-split coords_by_layer — no transformation needed.
        let u = sd_ref.coords_by_layer[0].clone();
        let s = sd_ref.coords_by_layer[1].clone();
        let f = sd_ref.freqs.clone();
        let lim  = sd_ref.limits.clone();
        let gll  = sd_ref.gene_log_lengths.clone();
        let gll_s = sd_ref.gene_log_lengths_spliced.clone();
        (u, s, f, lim, n_lm, n, gll, gll_s)
    }; // sd_ref dropped — GIL borrow released before allow_threads

    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes)
                .into_par_iter()
                .map(|gi| {
                    // Compute per-gene samp: add gene_log_lengths[gi] to the
                    // appropriate channels. Spliced channel uses gene_log_lengths_spliced
                    // when present, falling back to gene_log_lengths.
                    let samp_gene: Option<Vec<f64>> = base_samp.as_ref().map(|bs| {
                        let mut s = bs.clone();
                        if use_lengths_unspliced && s.len() > 0 {
                            if let Some(ref gll) = gene_log_lengths { s[0] += gll[gi]; }
                        }
                        if use_lengths_spliced && s.len() > 1 {
                            let spl_lengths = gene_log_lengths_spliced.as_ref()
                                .or(gene_log_lengths.as_ref());
                            if let Some(ref gll) = spl_lengths { s[1] += gll[gi]; }
                        }
                        s
                    });
                    let mut best_x: Vec<f64> = (0..lb.len())
                        .map(|i| x0_list[gi][0][i].max(lb[i]).min(ub[i]))
                        .collect();
                    let mut best_kld = f64::INFINITY;
                    for x0 in &x0_list[gi] {
                        let (x_opt, kld) = lbfgsb_minimize(
                            &bio_model, x0, &lb, &ub,
                            &limits_flat[gi * n_lim .. (gi + 1) * n_lim],
                            &u_idx_list[gi], &s_idx_list[gi], &f_list[gi],
                            fixed_quad_t, quad_order, fd_eps,
                            maxiter, ftol, gtol,
                            samp_gene.as_deref(), eps, m_lbfgs, &seq_model,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        })
    });

    let (params, klds): (Vec<Vec<f64>>, Vec<f64>) = results.into_iter().unzip();
    Ok((params, klds))
}

// ============================================================================
// N-dimensional sparse KLD helper and row-major strides
// ============================================================================

/// KLD(data ‖ model) for a sparse N-D histogram.
///
/// pss     : flat PSS, shape [l0, l1, …, l_{N-1}] in row-major order
/// strides : row-major strides[i] = product(limits[i+1..])
/// coords  : per-microstate count vectors, coords[ms][i] = count in layer i
/// f       : fractional frequencies matching coords
/// eps     : probability floor
#[inline]
fn compute_kld_sparse_nd(
    pss:     &[f64],
    strides: &[usize],
    coords:  &[Vec<i64>],
    f:       &[f64],
    eps:     f64,
) -> f64 {
    coords
        .iter()
        .zip(f.iter())
        .map(|(ms, &fi)| {
            let idx: usize = ms
                .iter()
                .zip(strides.iter())
                .map(|(&c, &s)| c as usize * s)
                .sum();
            let pval = pss[idx].max(eps);
            fi * (fi / pval).ln()
        })
        .sum()
}

/// Build row-major strides for a shape given by `limits`.
/// strides[i] = product(limits[i+1..])
fn row_major_strides(limits: &[usize]) -> Vec<usize> {
    let n = limits.len();
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * limits[i + 1];
    }
    strides
}

// ============================================================================
// ProteinBursty L-BFGS-B optimizer (3-D, no GIL round-trips)
// ============================================================================

/// Evaluate ProteinBursty PSS in-process (no PyO3 overhead).
fn eval_model_pss_protein_bursty_seq(
    p_log:         &[f64],
    limits:        &[usize],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge:     f64,
    max_fudge:     f64,
) -> Vec<f64> {
    let (gf, mx, lims) =
        protein_bursty_core(p_log, limits, fit_unspliced, protein_limit, min_fudge, max_fudge);
    let pss_raw = irfftn_3d(&gf, mx[0], mx[2], lims[0], lims[1], lims[2]);
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
}

/// KLD + forward finite-difference gradient for ProteinBursty (sequential).
/// Called from within rayon tasks — no nested parallelism.
fn eval_kld_and_grad_seq_protein_bursty(
    x:             &[f64],
    limits:        &[usize],
    strides:       &[usize],
    coords:        &[Vec<i64>],
    f_data:        &[f64],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge:     f64,
    max_fudge:     f64,
    fd_eps:        f64,
    eps:           f64,
) -> (f64, Vec<f64>) {
    let n = x.len();
    let pss0 = eval_model_pss_protein_bursty_seq(
        x, limits, fit_unspliced, protein_limit, min_fudge, max_fudge,
    );
    let kld0 = compute_kld_sparse_nd(&pss0, strides, coords, f_data, eps);
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            let pss = eval_model_pss_protein_bursty_seq(
                &x_eps, limits, fit_unspliced, protein_limit, min_fudge, max_fudge,
            );
            let kld = compute_kld_sparse_nd(&pss, strides, coords, f_data, eps);
            (kld - kld0) / fd_eps
        })
        .collect();
    (kld0, grad)
}

/// L-BFGS-B box-constrained minimization for ProteinBursty.
fn lbfgsb_minimize_protein_bursty(
    x0:            &[f64],
    lb:            &[f64],
    ub:            &[f64],
    limits:        &[usize],
    strides:       &[usize],
    coords:        &[Vec<i64>],
    f_data:        &[f64],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge:     f64,
    max_fudge:     f64,
    fd_eps:        f64,
    maxiter:       usize,
    ftol:          f64,
    gtol:          f64,
    eps:           f64,
    m_mem:         usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();
    let mut solver = LBFGSB::new(m_mem).with_max_iter(maxiter).with_pgtol(gtol).with_ftol(ftol);
    let mut f_and_grad = |xx: &[f64]| -> (f64, Vec<f64>) {
        eval_kld_and_grad_seq_protein_bursty(
            xx, limits, strides, coords, f_data,
            fit_unspliced, protein_limit, min_fudge, max_fudge, fd_eps, eps,
        )
    };
    let mut prev_f = f64::INFINITY;
    let mut callback = |info: &lbfgsb_rs_pure::IterationInfo, _x: &[f64]| {
        let improve = (prev_f - info.f).abs() / prev_f.abs().max(1.0);
        prev_f = info.f;
        if improve < ftol {
            IterationControl::StopConverged
        } else {
            IterationControl::Continue
        }
    };
    match solver.minimize_with_callback(&mut x, lb, ub, &mut f_and_grad, &mut callback) {
        Ok(sol) => (sol.x, sol.f),
        Err(_) => {
            let (f0, _) = f_and_grad(&x);
            (x, f0)
        }
    }
}

/// Optimize all genes with the ProteinBursty model in parallel, reading histogram
/// data directly from a Rust `SearchData` — no Python marshal loop.
#[pyfunction]
#[pyo3(signature = (sd, x0_list, lb, ub, fit_unspliced, protein_limit, min_fudge, max_fudge,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    eps=1e-15, m_lbfgs=10, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn optimize_genes_protein_bursty_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    x0_list: Vec<Vec<Vec<f64>>>,   // n_genes × num_restarts × n_params
    lb: Vec<f64>,
    ub: Vec<f64>,
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
    fd_eps: f64,
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    eps: f64,
    m_lbfgs: usize,
    num_threads: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (coords_list, f_list, limits_flat, n_genes, n_lim) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optimize_genes_protein_bursty_sd requires at least 3 layers",
            ));
        }
        sd_ref.extract_for_parallel()
    };


    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes)
                .into_par_iter()
                .map(|gi| {
                    let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                    let strides = row_major_strides(lims);
                    let mut best_x: Vec<f64> = (0..lb.len())
                        .map(|i| x0_list[gi][0][i].max(lb[i]).min(ub[i]))
                        .collect();
                    let mut best_kld = f64::INFINITY;
                    for x0 in &x0_list[gi] {
                        let (x_opt, kld) = lbfgsb_minimize_protein_bursty(
                            x0, &lb, &ub, lims, &strides,
                            &coords_list[gi], &f_list[gi],
                            fit_unspliced, protein_limit, min_fudge, max_fudge,
                            fd_eps, maxiter, ftol, gtol, eps, m_lbfgs,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        })
    });

    let (params, klds): (Vec<Vec<f64>>, Vec<f64>) = results.into_iter().unzip();
    Ok((params, klds))
}

// ============================================================================
// 2-D amb_model L-BFGS-B optimizer (Equal / Unequal ambient noise)
// ============================================================================

/// Evaluate 2-D PSS with ambient-noise model (no PyO3 overhead).
///
/// p_log layout: [bio_params..., amb_params] where:
///   amb_model="Equal"   → 1 ambient param appended (log10 p_amb)
///   amb_model="Unequal" → 2 ambient params appended (log10 p_amb0, log10 p_amb1)
/// limits: [l0, l1, l_amb]
fn eval_model_pss_2d_amb_seq(
    bio_model:    &str,
    p_log:        &[f64],
    limits:       &[usize],
    fixed_quad_t: f64,
    quad_order:   usize,
    samp_log:     Option<&[f64]>,
    amb_model:    &str,   // "Equal" or "Unequal"
) -> Vec<f64> {
    let n_amb: usize = if amb_model == "Equal" { 1 } else { 2 };
    let n_bio = p_log.len() - n_amb;
    let p: Vec<f64> = p_log[..n_bio].iter().map(|&v| 10.0_f64.powf(v)).collect();
    let amb = &p_log[n_bio..];

    let (l0, l1, l2) = (limits[0], limits[1], limits[2]);
    let mx2 = l2 / 2 + 1;

    let (p_amb0, p_amb1): (f64, f64) = if amb_model == "Equal" {
        let v = 10.0_f64.powf(amb[0]);
        (v, v)
    } else {
        (10.0_f64.powf(amb[0]), 10.0_f64.powf(amb[1]))
    };

    let mesh3 = build_mesh_3d_cached(l0, l1, l2);
    let (g0_base, g1_base, g_amb_base) = (&mesh3.0, &mesh3.1, &mesh3.2);

    let one  = Complex64::new(1.0, 0.0);
    let q0   = Complex64::new(1.0 - p_amb0, 0.0);
    let pa0  = Complex64::new(p_amb0, 0.0);
    let q1   = Complex64::new(1.0 - p_amb1, 0.0);
    let pa1  = Complex64::new(p_amb1, 0.0);

    let mut g0_eff: Vec<Complex64> = g0_base
        .iter()
        .zip(g_amb_base.iter())
        .map(|(&g0, &ga)| g0 * q0 + ga * pa0)
        .collect();
    let mut g1_eff: Vec<Complex64> = g1_base
        .iter()
        .zip(g_amb_base.iter())
        .map(|(&g1, &ga)| g1 * q1 + ga * pa1)
        .collect();

    if let Some(samp) = samp_log {
        let lam0 = 10.0_f64.powf(samp[0]);
        let lam1 = 10.0_f64.powf(samp[1]);
        g0_eff.iter_mut().for_each(|z| *z = (*z * lam0).exp() - one);
        g1_eff.iter_mut().for_each(|z| *z = (*z * lam1).exp() - one);
    }

    let gf_log = match bio_model {
        "Constitutive"    => pgf_constitutive(&g0_eff, &g1_eff, &p),
        "Extrinsic"       => pgf_extrinsic(&g0_eff, &g1_eff, &p),
        "Delay"           => pgf_delay(&g0_eff, &g1_eff, &p),
        "DelayedSplicing" => pgf_delayed_splicing(&g0_eff, &g1_eff, &p),
        "Bursty" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_bursty(&g0_eff, &g1_eff, &p, t, quad_order)
        }
        "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            pgf_cir(&g0_eff, &g1_eff, &p, t, quad_order)
        }
        other => panic!("Unknown bio_model in eval_model_pss_2d_amb_seq: {other}"),
    };

    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();
    let pss_raw = irfftn_3d(&gf, l0, mx2, l0, l1, l2);
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
}

/// KLD + forward finite-difference gradient for 2-D amb_model (sequential).
fn eval_kld_and_grad_seq_amb(
    bio_model:    &str,
    x:            &[f64],
    limits:       &[usize],
    strides:      &[usize],
    coords:       &[Vec<i64>],
    f_data:       &[f64],
    fixed_quad_t: f64,
    quad_order:   usize,
    samp_log:     Option<&[f64]>,
    amb_model:    &str,
    fd_eps:       f64,
    eps:          f64,
) -> (f64, Vec<f64>) {
    let n = x.len();
    let pss0 = eval_model_pss_2d_amb_seq(
        bio_model, x, limits, fixed_quad_t, quad_order, samp_log, amb_model,
    );
    let kld0 = compute_kld_sparse_nd(&pss0, strides, coords, f_data, eps);
    let grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            let pss = eval_model_pss_2d_amb_seq(
                bio_model, &x_eps, limits, fixed_quad_t, quad_order, samp_log, amb_model,
            );
            let kld = compute_kld_sparse_nd(&pss, strides, coords, f_data, eps);
            (kld - kld0) / fd_eps
        })
        .collect();
    (kld0, grad)
}

/// L-BFGS-B box-constrained minimization for 2-D amb_model.
fn lbfgsb_minimize_amb(
    bio_model:    &str,
    amb_model:    &str,
    x0:           &[f64],
    lb:           &[f64],
    ub:           &[f64],
    limits:       &[usize],
    strides:      &[usize],
    coords:       &[Vec<i64>],
    f_data:       &[f64],
    fixed_quad_t: f64,
    quad_order:   usize,
    samp_log:     Option<&[f64]>,
    fd_eps:       f64,
    maxiter:      usize,
    ftol:         f64,
    gtol:         f64,
    eps:          f64,
    m_mem:        usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();
    let mut solver = LBFGSB::new(m_mem).with_max_iter(maxiter).with_pgtol(gtol).with_ftol(ftol);
    let mut f_and_grad = |xx: &[f64]| -> (f64, Vec<f64>) {
        eval_kld_and_grad_seq_amb(
            bio_model, xx, limits, strides, coords, f_data,
            fixed_quad_t, quad_order, samp_log, amb_model, fd_eps, eps,
        )
    };
    let mut prev_f = f64::INFINITY;
    let mut callback = |info: &lbfgsb_rs_pure::IterationInfo, _x: &[f64]| {
        let improve = (prev_f - info.f).abs() / prev_f.abs().max(1.0);
        prev_f = info.f;
        if improve < ftol {
            IterationControl::StopConverged
        } else {
            IterationControl::Continue
        }
    };
    match solver.minimize_with_callback(&mut x, lb, ub, &mut f_and_grad, &mut callback) {
        Ok(sol) => (sol.x, sol.f),
        Err(_) => {
            let (f0, _) = f_and_grad(&x);
            (x, f0)
        }
    }
}

/// Optimize all genes with a 2-D bio_model + ambient noise, reading histogram data
/// directly from a Rust `SearchData` — no Python marshal loop.
///
/// p_log layout: [bio_params..., amb_params] where n_amb = 1 (Equal) or 2 (Unequal).
/// `limits` for each gene must be [l0, l1, l_amb] (3 entries).
#[pyfunction]
#[pyo3(signature = (sd, bio_model, amb_model, x0_list, lb, ub, fixed_quad_t, quad_order,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    samp_list=None, eps=1e-15, m_lbfgs=10, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn optimize_genes_2d_amb_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    bio_model: String,
    amb_model: String,
    x0_list: Vec<Vec<Vec<f64>>>,   // n_genes × num_restarts × n_params
    lb: Vec<f64>,
    ub: Vec<f64>,
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
    validate_bio_model_2d(&bio_model, "optimize_genes_2d_amb_sd")?;
    match amb_model.as_str() {
        "Equal" | "Unequal" => {}
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "optimize_genes_2d_amb_sd requires amb_model=Equal or Unequal, got: {other}"
        ))),
    }

    let (coords_list, f_list, limits_flat, n_genes, n_lim) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optimize_genes_2d_amb_sd requires at least 3 layers (u, s, ambient)",
            ));
        }
        sd_ref.extract_for_parallel()
    };


    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes)
                .into_par_iter()
                .map(|gi| {
                    let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                    let strides = row_major_strides(lims);
                    let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                    let mut best_x: Vec<f64> = (0..lb.len())
                        .map(|i| x0_list[gi][0][i].max(lb[i]).min(ub[i]))
                        .collect();
                    let mut best_kld = f64::INFINITY;
                    for x0 in &x0_list[gi] {
                        let (x_opt, kld) = lbfgsb_minimize_amb(
                            &bio_model, &amb_model, x0, &lb, &ub,
                            lims, &strides, &coords_list[gi], &f_list[gi],
                            fixed_quad_t, quad_order, samp,
                            fd_eps, maxiter, ftol, gtol, eps, m_lbfgs,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        })
    });

    let (params, klds): (Vec<Vec<f64>>, Vec<f64>) = results.into_iter().unzip();
    Ok((params, klds))
}

// ============================================================================
// Custom reaction-network L-BFGS-B optimizer
// ============================================================================

/// Build `Vec<RxnInfo>` from flat topology arrays.
fn build_rxns(
    rxn_kinds:     &[u8],
    rxn_rate_idxs: &[u32],
    rxn_extra1:    &[i32],
    rxn_extra2:    &[i32],
    prod_off:      &[u32],
) -> Vec<RxnInfo> {
    (0..rxn_kinds.len())
        .map(|i| RxnInfo {
            kind:       rxn_kinds[i],
            rate_idx:   rxn_rate_idxs[i],
            extra1:     rxn_extra1[i],
            extra2:     rxn_extra2[i],
            prod_start: prod_off[i],
            prod_end:   prod_off[i + 1],
        })
        .collect()
}

/// Evaluate the custom-network PSS in-process (no PyO3).
///
/// Supports 2-D and 3-D species counts; panics for any other N.
fn eval_custom_network_pss_seq(
    n_species:        usize,
    limits:           &[usize],
    rxns:             &[RxnInfo],
    prod_sp:          &[u32],
    prod_st:          &[u32],
    params_linear:    &[f64],
    dt:               f64,
    n_steps:          usize,
    max_while_steps:  usize,
) -> Vec<f64> {
    let (g, mx, n_grid) = build_mesh_nd(limits);
    let exp_phi = custom_pgf_parallel(
        &g, params_linear, rxns, prod_sp, prod_st,
        n_species, n_grid, dt, n_steps, max_while_steps,
    );

    let pss_raw: Vec<f64> = match n_species {
        2 => irfftn_2d(&exp_phi, limits[0], limits[1]),
        3 => irfftn_3d(&exp_phi, mx[0], mx[2], limits[0], limits[1], limits[2]),
        _ => panic!("eval_custom_network_pss_seq: only 2- and 3-species networks supported"),
    };

    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    pss_raw.iter().map(|x| x.abs() / abs_sum).collect()
}

/// Derive `params_linear`, `dt`, `n_steps`, and `max_while` from the log10 optimizer
/// state `x`.  `x` covers all rate parameters; `norm_rate = 1.0` is appended if present.
fn custom_network_integration_params(
    x:             &[f64],
    has_norm_rate: bool,
    min_fudge:     f64,
    max_fudge:     f64,
) -> (Vec<f64>, f64, usize, usize) {
    let p_lin: Vec<f64> = x.iter().map(|&v| 10.0_f64.powf(v)).collect();
    let dt_f  = p_lin.iter().map(|&v| 1.0 / v).fold(f64::INFINITY, f64::min) * min_fudge;
    let t_max = p_lin.iter().map(|&v| 1.0 / v).fold(0.0_f64, f64::max) * max_fudge;
    let n_steps   = (t_max / dt_f).ceil() as usize;
    let max_while = 10 * n_steps + 10_000;
    let mut params_linear = p_lin;
    if has_norm_rate {
        params_linear.push(1.0);
    }
    (params_linear, dt_f, n_steps, max_while)
}

/// KLD + forward finite-difference gradient for a custom network (sequential).
fn eval_kld_and_grad_seq_custom(
    x:             &[f64],
    n_species:     usize,
    limits:        &[usize],
    strides:       &[usize],
    rxns:          &[RxnInfo],
    prod_sp:       &[u32],
    prod_st:       &[u32],
    coords:        &[Vec<i64>],
    f_data:        &[f64],
    has_norm_rate: bool,
    min_fudge:     f64,
    max_fudge:     f64,
    fd_eps:        f64,
    eps:           f64,
) -> (f64, Vec<f64>) {
    let n = x.len();
    let (params, dt, n_steps, max_while) =
        custom_network_integration_params(x, has_norm_rate, min_fudge, max_fudge);
    let pss0 = eval_custom_network_pss_seq(
        n_species, limits, rxns, prod_sp, prod_st, &params, dt, n_steps, max_while,
    );
    let kld0 = compute_kld_sparse_nd(&pss0, strides, coords, f_data, eps);

    let grad: Vec<f64> = (0..n)
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            let (params_eps, dt_eps, ns_eps, mw_eps) =
                custom_network_integration_params(&x_eps, has_norm_rate, min_fudge, max_fudge);
            let pss = eval_custom_network_pss_seq(
                n_species, limits, rxns, prod_sp, prod_st,
                &params_eps, dt_eps, ns_eps, mw_eps,
            );
            let kld = compute_kld_sparse_nd(&pss, strides, coords, f_data, eps);
            (kld - kld0) / fd_eps
        })
        .collect();
    (kld0, grad)
}

/// L-BFGS-B box-constrained minimization for a custom reaction network.
fn lbfgsb_minimize_custom(
    x0:            &[f64],
    lb:            &[f64],
    ub:            &[f64],
    n_species:     usize,
    limits:        &[usize],
    strides:       &[usize],
    rxns:          &[RxnInfo],
    prod_sp:       &[u32],
    prod_st:       &[u32],
    coords:        &[Vec<i64>],
    f_data:        &[f64],
    has_norm_rate: bool,
    min_fudge:     f64,
    max_fudge:     f64,
    fd_eps:        f64,
    maxiter:       usize,
    ftol:          f64,
    gtol:          f64,
    eps:           f64,
    m_mem:         usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut x: Vec<f64> = (0..n).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();
    let mut solver = LBFGSB::new(m_mem).with_max_iter(maxiter).with_pgtol(gtol).with_ftol(ftol);
    let mut f_and_grad = |xx: &[f64]| -> (f64, Vec<f64>) {
        eval_kld_and_grad_seq_custom(
            xx, n_species, limits, strides, rxns, prod_sp, prod_st,
            coords, f_data, has_norm_rate, min_fudge, max_fudge, fd_eps, eps,
        )
    };
    let mut prev_f = f64::INFINITY;
    let mut callback = |info: &lbfgsb_rs_pure::IterationInfo, _x: &[f64]| {
        let improve = (prev_f - info.f).abs() / prev_f.abs().max(1.0);
        prev_f = info.f;
        if improve < ftol {
            IterationControl::StopConverged
        } else {
            IterationControl::Continue
        }
    };
    match solver.minimize_with_callback(&mut x, lb, ub, &mut f_and_grad, &mut callback) {
        Ok(sol) => (sol.x, sol.f),
        Err(_) => {
            let (f0, _) = f_and_grad(&x);
            (x, f0)
        }
    }
}

/// Optimize all genes with a custom reaction network in parallel, reading histogram
/// data directly from a Rust `SearchData` — no Python marshal loop.
///
/// Only 2- and 3-species networks are supported.
/// Reaction topology is passed as flat arrays (same encoding as `eval_custom_network_pgf`).
#[pyfunction]
#[pyo3(signature = (sd, x0_list, lb, ub, n_species,
                    rxn_kinds, rxn_rate_idxs, rxn_extra1, rxn_extra2,
                    prod_sp, prod_st, prod_off,
                    has_norm_rate, min_fudge, max_fudge,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    eps=1e-15, m_lbfgs=10, num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn optimize_genes_custom_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    x0_list: Vec<Vec<Vec<f64>>>,   // n_genes × num_restarts × n_params
    lb: Vec<f64>,
    ub: Vec<f64>,
    n_species: usize,
    rxn_kinds:     Vec<u8>,
    rxn_rate_idxs: Vec<u32>,
    rxn_extra1:    Vec<i32>,
    rxn_extra2:    Vec<i32>,
    prod_sp:       Vec<u32>,
    prod_st:       Vec<u32>,
    prod_off:      Vec<u32>,
    has_norm_rate: bool,
    min_fudge:     f64,
    max_fudge:     f64,
    fd_eps:        f64,
    maxiter:       usize,
    ftol:          f64,
    gtol:          f64,
    eps:           f64,
    m_lbfgs:       usize,
    num_threads:   Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_species != 2 && n_species != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "optimize_genes_custom_sd only supports 2- and 3-species custom networks",
        ));
    }

    let (coords_list, f_list, limits_flat, n_genes, n_lim) = sd.borrow().extract_for_parallel();

    let rxns = build_rxns(&rxn_kinds, &rxn_rate_idxs, &rxn_extra1, &rxn_extra2, &prod_off);


    let results: Vec<(Vec<f64>, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes)
                .into_par_iter()
                .map(|gi| {
                    let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                    let strides = row_major_strides(lims);
                    let mut best_x: Vec<f64> = (0..lb.len())
                        .map(|i| x0_list[gi][0][i].max(lb[i]).min(ub[i]))
                        .collect();
                    let mut best_kld = f64::INFINITY;
                    for x0 in &x0_list[gi] {
                        let (x_opt, kld) = lbfgsb_minimize_custom(
                            x0, &lb, &ub, n_species, lims, &strides,
                            &rxns, &prod_sp, &prod_st,
                            &coords_list[gi], &f_list[gi],
                            has_norm_rate, min_fudge, max_fudge,
                            fd_eps, maxiter, ftol, gtol, eps, m_lbfgs,
                        );
                        if kld < best_kld * ERR_THRESH {
                            best_x = x_opt;
                            best_kld = kld;
                        }
                    }
                    (best_x, best_kld)
                })
                .collect()
        })
    });

    let (params, klds): (Vec<Vec<f64>>, Vec<f64>) = results.into_iter().unzip();
    Ok((params, klds))
}

// ============================================================================
// E-step for MEK-Means EM algorithm
// ============================================================================

/// Compute the E-step for the MEK-Means EM algorithm in Rust with rayon parallelism.
///
/// For each cluster k, evaluates eval_model_pss_2d_seq for every gene (in parallel),
/// looks up per-cell log-probabilities by indexing PSS[u_obs[g][c], s_obs[g][c]],
/// and accumulates logL[c][k] = sum_genes log P(obs | theta_k).
/// Adds log(weights[k]) and applies softmax to return Q, lower_bound, q_func.
///
/// Parameters
/// ----------
/// bio_model     : one of the six supported 2D models
/// params_per_k  : n_k × n_genes × n_params (log10 param vectors)
/// limits_list   : n_genes × 2 grid bounds [m_u, m_s]
/// u_obs         : n_genes × n_cells unspliced count observations
/// s_obs         : n_genes × n_cells spliced count observations
/// weights       : n_k mixture weights (must sum to ~1)
/// fixed_quad_t  : quadrature time-scale multiplier
/// quad_order    : Gauss-Legendre quadrature order
/// samp_list     : optional n_genes list of sampling params; None for seq_model="None"
/// eps           : probability floor before log (default 1e-15)
/// num_threads   : optional rayon thread-pool size
/// seq_model     : sequencing model ("None", "Poisson", or "Bernoulli"); default "Poisson"
///
/// Returns (Q, lower_bound, q_func)
///   Q            : n_cells × n_k posterior matrix
///   lower_bound  : mean over cells of logsumexp(logL, axis=1)
///   q_func       : sum of Q * logL (EM Q-function value)
#[pyfunction]
#[pyo3(signature = (bio_model, params_per_k, limits_list, u_obs, s_obs,
                    weights, fixed_quad_t, quad_order,
                    samp_list=None, eps=1e-15, num_threads=None, seq_model="Poisson"))]
#[allow(clippy::too_many_arguments)]
fn e_step_2d(
    py: Python<'_>,
    bio_model: String,
    params_per_k: Vec<Vec<Vec<f64>>>,
    limits_list: Vec<Vec<usize>>,
    u_obs: Vec<Vec<u64>>,
    s_obs: Vec<Vec<u64>>,
    weights: Vec<f64>,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    eps: f64,
    num_threads: Option<usize>,
    seq_model: &str,
) -> PyResult<(Vec<Vec<f64>>, f64, f64)> {
    validate_bio_model_2d(&bio_model, "e_step_2d")?;
    let seq_model = seq_model.to_owned();
    let n_k = params_per_k.len();
    let n_genes = limits_list.len();
    let n_cells = if n_genes > 0 && !u_obs.is_empty() { u_obs[0].len() } else { 0 };

    let result = py.allow_threads(|| {
        let run = || -> (Vec<Vec<f64>>, f64, f64) {
            // logL[c][k] accumulated across genes
            let mut logL = vec![vec![0.0_f64; n_k]; n_cells];

            for k in 0..n_k {
                // Parallel over genes: each produces a per-cell log-prob vector
                let gene_log_probs: Vec<Vec<f64>> = (0..n_genes)
                    .into_par_iter()
                    .map(|g| {
                        let samp = samp_list.as_ref().and_then(|sl| sl[g].as_deref());
                        let pss = eval_model_pss_2d_seq(
                            &bio_model,
                            &params_per_k[k][g],
                            &limits_list[g],
                            fixed_quad_t,
                            quad_order,
                            samp,
                            &seq_model,
                        );
                        let m_s = limits_list[g][1];
                        u_obs[g].iter().zip(s_obs[g].iter()).map(|(&u, &s)| {
                            let idx = u as usize * m_s + s as usize;
                            let p = if idx < pss.len() { pss[idx] } else { 0.0 };
                            p.max(eps).ln()
                        }).collect::<Vec<f64>>()
                    })
                    .collect();

                // Sum over genes into logL[c][k] and add log(weight)
                let log_w = weights[k].max(f64::MIN_POSITIVE).ln();
                for c in 0..n_cells {
                    let gene_sum: f64 = gene_log_probs.iter().map(|lp| lp[c]).sum();
                    logL[c][k] = gene_sum + log_w;
                }
            }

            // Softmax row-wise → Q; accumulate logsumexp for lower_bound
            let mut q_mat = vec![vec![0.0_f64; n_k]; n_cells];
            let mut total_lse = 0.0_f64;
            for c in 0..n_cells {
                let max_val = logL[c].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = logL[c].iter().map(|&v| (v - max_val).exp()).sum();
                let lse = max_val + sum_exp.ln();
                total_lse += lse;
                for k in 0..n_k {
                    q_mat[c][k] = (logL[c][k] - max_val).exp() / sum_exp;
                }
            }
            let lower_bound = if n_cells > 0 { total_lse / n_cells as f64 } else { 0.0 };

            // q_func = sum(Q * logL)
            let q_func: f64 = q_mat.iter().zip(logL.iter()).map(|(q_row, l_row)| {
                q_row.iter().zip(l_row.iter()).map(|(&q, &l)| q * l).sum::<f64>()
            }).sum();

            (q_mat, lower_bound, q_func)
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

    Ok(result)
}

/// E-step accumulation from precomputed PSS grids (PSS caching support).
///
/// `grids_per_k[k][g]` is the flat PSS array for cluster k, gene g.
/// `limits_list[g] = [m_u, m_s]` — `m_s` is used to index into the flat grid.
/// Returns (Q, lower_bound, q_func) identical to e_step_2d but skips PSS computation.
#[pyfunction]
#[pyo3(signature = (grids_per_k, limits_list, u_obs, s_obs, weights, eps=1e-15, num_threads=None))]
fn e_step_2d_from_grids(
    py: Python<'_>,
    grids_per_k: Vec<Vec<Vec<f64>>>,   // n_k × n_genes × grid_flat
    limits_list: Vec<Vec<usize>>,
    u_obs: Vec<Vec<u64>>,
    s_obs: Vec<Vec<u64>>,
    weights: Vec<f64>,
    eps: f64,
    num_threads: Option<usize>,
) -> PyResult<(Vec<Vec<f64>>, f64, f64)> {
    let n_k = grids_per_k.len();
    let n_genes = limits_list.len();
    let n_cells = if n_genes > 0 && !u_obs.is_empty() { u_obs[0].len() } else { 0 };

    let result = py.allow_threads(|| {
        let run = || -> (Vec<Vec<f64>>, f64, f64) {
            let mut logL = vec![vec![0.0_f64; n_k]; n_cells];

            for k in 0..n_k {
                let gene_log_probs: Vec<Vec<f64>> = (0..n_genes)
                    .into_par_iter()
                    .map(|g| {
                        let pss = &grids_per_k[k][g];
                        let m_s = limits_list[g][1];
                        u_obs[g].iter().zip(s_obs[g].iter()).map(|(&u, &s)| {
                            let idx = u as usize * m_s + s as usize;
                            let p = if idx < pss.len() { pss[idx] } else { 0.0 };
                            p.max(eps).ln()
                        }).collect::<Vec<f64>>()
                    })
                    .collect();

                let log_w = weights[k].max(f64::MIN_POSITIVE).ln();
                for c in 0..n_cells {
                    let gene_sum: f64 = gene_log_probs.iter().map(|lp| lp[c]).sum();
                    logL[c][k] = gene_sum + log_w;
                }
            }

            // Softmax row-wise → Q; logsumexp → lower_bound; Q·logL → q_func
            let mut q_mat = vec![vec![0.0_f64; n_k]; n_cells];
            let mut total_lse = 0.0_f64;
            for c in 0..n_cells {
                let max_val = logL[c].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = logL[c].iter().map(|&v| (v - max_val).exp()).sum();
                let lse = max_val + sum_exp.ln();
                total_lse += lse;
                for k in 0..n_k {
                    q_mat[c][k] = (logL[c][k] - max_val).exp() / sum_exp;
                }
            }
            let lower_bound = if n_cells > 0 { total_lse / n_cells as f64 } else { 0.0 };
            let q_func: f64 = q_mat.iter().zip(logL.iter()).map(|(q_row, l_row)| {
                q_row.iter().zip(l_row.iter()).map(|(&q, &l)| q * l).sum::<f64>()
            }).sum();

            (q_mat, lower_bound, q_func)
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

    Ok(result)
}

// ============================================================================
// Phase 3: write inference results back into a RuAnnData and save as h5ad
// ============================================================================

/// Annotate the `RuAnnData` carried by `search_data` with fitted parameters and
/// KLD values, then write the result to `output_path` as an h5ad file.
///
/// Parameters
/// ----------
/// search_data  : SearchData built via `searchdata_from_h5ad` (must carry adata)
/// param_names  : column names for each parameter (e.g. ["b", "beta", "gamma"])
/// params       : 2-D array, shape [n_genes, n_params], log10 parameter values
/// klds         : 1-D array, shape [n_genes], per-gene KLD at optimum
/// output_path  : destination h5ad path
#[cfg(feature = "ruanndata")]
#[pyfunction]
fn annotate_inference_results(
    py: Python<'_>,
    search_data: &Bound<'_, PySearchData>,
    param_names: Vec<String>,
    params: PyReadonlyArray2<'_, f64>,
    klds: PyReadonlyArray1<'_, f64>,
    output_path: String,
) -> PyResult<()> {
    let sd = search_data.borrow();
    let mut adata = sd.adata
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
            "search_data has no attached RuAnnData; build it with searchdata_from_h5ad",
        ))?
        .clone();

    let n_genes = adata.n_vars();
    let params_arr = params.as_array();
    let klds_arr   = klds.as_array();

    if params_arr.nrows() != n_genes || klds_arr.len() != n_genes {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params/klds length ({}/{}) does not match n_genes ({})",
            params_arr.nrows(), klds_arr.len(), n_genes,
        )));
    }
    if params_arr.ncols() != param_names.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params has {} columns but {} param_names were given",
            params_arr.ncols(), param_names.len(),
        )));
    }

    // Add one var column per parameter.
    for (col_idx, name) in param_names.iter().enumerate() {
        let values: Vec<Option<f64>> = (0..n_genes)
            .map(|g| Some(params_arr[[g, col_idx]]))
            .collect();
        adata.var.columns.insert(
            name.clone(),
            SeriesData::Float64 { values },
        );
    }

    // Add kld column.
    let kld_values: Vec<Option<f64>> = (0..n_genes).map(|g| Some(klds_arr[g])).collect();
    adata.var.columns.insert("kld".to_string(), SeriesData::Float64 { values: kld_values });

    // Strip internal monod scaffolding that uses encoding types Python anndata cannot read.
    for key in &["state_dist_coords", "state_dist_freqs", "limits"] {
        adata.uns.remove(*key);
    }

    py.allow_threads(|| {
        write_h5ad(&output_path, &adata).map_err(|e| e.to_string())
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
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
    m.add_function(wrap_pyfunction!(optimize_genes_2d_sd, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_genes_protein_bursty_sd, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_genes_2d_amb_sd, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_genes_custom_sd, m)?)?;
    m.add_function(wrap_pyfunction!(e_step_2d, m)?)?;
    m.add_function(wrap_pyfunction!(e_step_2d_from_grids, m)?)?;
    m.add_function(wrap_pyfunction!(compute_moments, m)?)?;
    m.add_class::<PySearchData>()?;
    m.add_function(wrap_pyfunction!(searchdata_from_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(make_state_dist, m)?)?;
    #[cfg(feature = "ruanndata")]
    m.add_function(wrap_pyfunction!(load_histograms_h5ad, m)?)?;
    #[cfg(feature = "ruanndata")]
    m.add_function(wrap_pyfunction!(searchdata_from_h5ad, m)?)?;
    #[cfg(feature = "ruanndata")]
    m.add_function(wrap_pyfunction!(annotate_inference_results, m)?)?;
    m.add_function(wrap_pyfunction!(eval_custom_network_pgf, m)?)?;
    Ok(())
}
