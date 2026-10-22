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
use indexmap::IndexMap;
use ruanndata::{read_h5ad, write_h5ad, ArrayData, ArrayValue, DataFrame, MatrixData, RuAnnData, SeriesData, UnsValue};
use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use ndarray::Array3;
use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::hash::Hash;

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
// ----------------------------------------------------------------------------
// Bounded mesh caches.
//
// Meshes are keyed by grid shape (and, for the Poisson/Bernoulli transforms, by
// the sampling parameters).  With per-gene limits and per-gene length offsets
// every (gene, grid point) pair is a distinct key, so an unbounded map grows
// without limit over a run.  `BoundedCache` tracks the total number of stored
// complex entries and clears itself when the next insert would exceed the cap
// (default 8M entries = 128 MB per cache).  Arc handles already handed out stay
// valid, so a clear never invalidates an in-flight computation; the hot
// working set (one transformed mesh per optimising gene) is rebuilt cheaply.
// ----------------------------------------------------------------------------

static MESH_CACHE_CAP_ENTRIES: AtomicUsize = AtomicUsize::new(8 << 20);

struct BoundedCache<K, V> {
    map: HashMap<K, Arc<V>>,
    total_entries: usize,
}

impl<K: Eq + Hash + Clone, V> BoundedCache<K, V> {
    fn new() -> Self {
        BoundedCache { map: HashMap::new(), total_entries: 0 }
    }
    fn get(&self, key: &K) -> Option<Arc<V>> {
        self.map.get(key).map(Arc::clone)
    }
    fn insert(&mut self, key: K, value: V, n_entries: usize) -> Arc<V> {
        if let Some(existing) = self.map.get(&key) {
            return Arc::clone(existing); // another thread built it meanwhile
        }
        let cap = MESH_CACHE_CAP_ENTRIES.load(Ordering::Relaxed);
        if !self.map.is_empty() && self.total_entries + n_entries > cap {
            self.map.clear();
            self.total_entries = 0;
        }
        let a = Arc::new(value);
        self.map.insert(key, Arc::clone(&a));
        self.total_entries += n_entries;
        a
    }
    fn clear(&mut self) {
        self.map.clear();
        self.total_entries = 0;
    }
}

/// Look up `key`; on a miss, build OUTSIDE the lock (so concurrent first-time
/// builds do not serialise) and insert.
fn cached<K: Eq + Hash + Clone, V>(
    cell: &OnceLock<Mutex<BoundedCache<K, V>>>,
    key: K,
    build: impl FnOnce() -> (V, usize),
) -> Arc<V> {
    let cache = cell.get_or_init(|| Mutex::new(BoundedCache::new()));
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return v;
    }
    let (v, n) = build();
    cache.lock().unwrap().insert(key, v, n)
}

fn cache_stats<K: Eq + Hash + Clone, V>(cell: &OnceLock<Mutex<BoundedCache<K, V>>>) -> (usize, usize) {
    match cell.get() {
        Some(m) => { let g = m.lock().unwrap(); (g.map.len(), g.total_entries) }
        None => (0, 0),
    }
}

fn cache_clear<K: Eq + Hash + Clone, V>(cell: &OnceLock<Mutex<BoundedCache<K, V>>>) {
    if let Some(m) = cell.get() { m.lock().unwrap().clear(); }
}

static MESH_CACHE_2D: OnceLock<Mutex<BoundedCache<(usize, usize), (Vec<Complex64>, Vec<Complex64>)>>> =
    OnceLock::new();

/// 3-D ambient mesh cache — Arc-wrapped, keyed by (l0, l1, l2).
/// Stores (g0, g1, g_amb) each of length l0 * l1 * (l2/2+1).
static MESH_CACHE_3D: OnceLock<Mutex<BoundedCache<(usize, usize, usize), (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>)>>> =
    OnceLock::new();

fn build_mesh_2d_cached(l0: usize, l1: usize) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    cached(&MESH_CACHE_2D, (l0, l1), || {
        let m = build_mesh_2d(l0, l1);
        let n = m.0.len() + m.1.len();
        (m, n)
    })
}

fn build_mesh_3d_cached(
    l0: usize,
    l1: usize,
    l2: usize,
) -> Arc<(Vec<Complex64>, Vec<Complex64>, Vec<Complex64>)> {
    cached(&MESH_CACHE_3D, (l0, l1, l2), || {
        let mx2 = l2 / 2 + 1;
        let m = build_mesh_3d(&[l0, l1, mx2], &[l0, l1, l2]);
        let n = m.0.len() + m.1.len() + m.2.len();
        (m, n)
    })
}

/// Poisson-transformed 2-D mesh cache — Arc-wrapped, keyed by (l0, l1, lam0_bits, lam1_bits).
static MESH_CACHE_POISSON: OnceLock<Mutex<BoundedCache<(usize, usize, u64, u64), (Vec<Complex64>, Vec<Complex64>)>>> =
    OnceLock::new();

fn build_mesh_2d_poisson_cached(
    l0: usize,
    l1: usize,
    lam0: f64,
    lam1: f64,
) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    let key = (l0, l1, lam0.to_bits(), lam1.to_bits());
    cached(&MESH_CACHE_POISSON, key, || {
        let base = build_mesh_2d_cached(l0, l1);
        let one = Complex64::new(1.0, 0.0);
        let g0t: Vec<Complex64> = base.0.iter().map(|&z| (lam0 * z).exp() - one).collect();
        let g1t: Vec<Complex64> = base.1.iter().map(|&z| (lam1 * z).exp() - one).collect();
        let n = g0t.len() + g1t.len();
        ((g0t, g1t), n)
    })
}

/// Bernoulli-transformed 2-D mesh cache — keyed by (l0, l1, s0_bits, s1_bits).
/// The Bernoulli transform is a scalar multiply by the capture probability:
/// g_i *= 10^s_i.  s0 and s1 are log10 capture probabilities (the same
/// convention as the Poisson model and as the sampling bounds / MoM init).
static MESH_CACHE_BERNOULLI: OnceLock<Mutex<BoundedCache<(usize, usize, u64, u64), (Vec<Complex64>, Vec<Complex64>)>>> =
    OnceLock::new();

fn build_mesh_2d_bernoulli_cached(
    l0: usize,
    l1: usize,
    s0: f64,
    s1: f64,
) -> Arc<(Vec<Complex64>, Vec<Complex64>)> {
    let key = (l0, l1, s0.to_bits(), s1.to_bits());
    cached(&MESH_CACHE_BERNOULLI, key, || {
        let base = build_mesh_2d_cached(l0, l1);
        let cs0 = Complex64::new(10f64.powf(s0), 0.0);
        let cs1 = Complex64::new(10f64.powf(s1), 0.0);
        let g0t: Vec<Complex64> = base.0.iter().map(|&z| z * cs0).collect();
        let g1t: Vec<Complex64> = base.1.iter().map(|&z| z * cs1).collect();
        let n = g0t.len() + g1t.len();
        ((g0t, g1t), n)
    })
}

/// Clear every mesh cache (2-D base, 3-D base, Poisson, Bernoulli).
#[pyfunction]
fn mesh_cache_clear() {
    cache_clear(&MESH_CACHE_2D);
    cache_clear(&MESH_CACHE_3D);
    cache_clear(&MESH_CACHE_POISSON);
    cache_clear(&MESH_CACHE_BERNOULLI);
}

/// Per-cache (number of meshes, total complex entries) plus the active cap.
#[pyfunction]
fn mesh_cache_stats() -> HashMap<String, (usize, usize)> {
    let mut m = HashMap::new();
    m.insert("mesh_2d".to_string(), cache_stats(&MESH_CACHE_2D));
    m.insert("mesh_3d".to_string(), cache_stats(&MESH_CACHE_3D));
    m.insert("poisson".to_string(), cache_stats(&MESH_CACHE_POISSON));
    m.insert("bernoulli".to_string(), cache_stats(&MESH_CACHE_BERNOULLI));
    m.insert("cap_entries".to_string(), (MESH_CACHE_CAP_ENTRIES.load(Ordering::Relaxed), 0));
    m
}

/// Set the per-cache cap in complex entries (16 bytes each). Default 8M (128 MB).
#[pyfunction]
fn mesh_cache_set_cap(entries: usize) {
    MESH_CACHE_CAP_ENTRIES.store(entries.max(1), Ordering::Relaxed);
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
/// Mesh of PGF arguments g = exp(-2*pi*i*k/limit) - 1 on a grid of shape `mx`
/// (the last axis is the rfft half), one flat row-major vector per axis.
fn build_mesh_axes(mx: &[usize], limits: &[usize]) -> Vec<Vec<Complex64>> {
    let n = mx.len();
    let two_pi = 2.0 * std::f64::consts::PI;
    let freq_vecs: Vec<Vec<Complex64>> = (0..n)
        .map(|ax| {
            (0..mx[ax])
                .map(|k| {
                    let theta = -two_pi * k as f64 / limits[ax] as f64;
                    Complex64::new(theta.cos() - 1.0, theta.sin())
                })
                .collect()
        })
        .collect();
    let n_grid: usize = mx.iter().product();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * mx[i + 1];
    }
    let mut g = vec![vec![Complex64::new(0.0, 0.0); n_grid]; n];
    for pt in 0..n_grid {
        for ax in 0..n {
            g[ax][pt] = freq_vecs[ax][(pt / strides[ax]) % mx[ax]];
        }
    }
    g
}

fn build_mesh_2d(l0: usize, l1: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut g = build_mesh_axes(&[l0, l1 / 2 + 1], &[l0, l1]);
    let g1 = g.pop().unwrap();
    (g.pop().unwrap(), g1)
}

/// Returns (g0, g1, g2) each of length mx0*mx1*mx2, row-major.
fn build_mesh_3d(
    mx: &[usize; 3],
    limits: &[usize; 3],
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let mut g = build_mesh_axes(mx, limits);
    let g2 = g.pop().unwrap();
    let g1 = g.pop().unwrap();
    (g.pop().unwrap(), g1, g2)
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
    let _n_freq = phi.len();
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
// ProteinBursty log-PGF — exponential integrator with error-controlled steps
// ============================================================================
//
// Characteristics of the bursty transcription + translation model, written in
// the PGF variables v = u0 + 1 (unspliced), w = u1 + 1 (spliced):
//
//     v' = beta (w - v)
//     w' = a(t) w + gamma,      a(t) = -gamma + k_p g2 exp(-gamma_p t)
//     phi = int_0^inf  b u0 / (1 - b u0) dt,   u0 = v - 1
//
// The protein characteristic u2 = g2 exp(-gamma_p t) is analytic, so the system
// is linear with one time-dependent coefficient whose integral
// A(t) = int_0^t a = -gamma t + (k_p g2 / gamma_p)(1 - exp(-gamma_p t)) is known
// in closed form.  Each step propagates the homogeneous part of w exactly and
// treats the forcing and the v-coupling with the step-average coefficient, which
// is exact for constant coefficients: the stiff rates never limit the step.  The
// integrand is accumulated with Simpson's rule on two half-steps, and the step
// size is chosen by comparing one full step with the two half-steps, taking the
// maximum error over all grid points so the whole grid shares one schedule.
// The resulting schedule (sequence of accepted steps) is deterministic in the
// parameters and the grid; gradient stencils reuse the base point's schedule.
//
// Replaces the fixed-step RK4 integrator, whose step count grew linearly with
// the spread between the fastest and slowest rate (up to 10^5 steps per point
// at the default bounds); this scheme needs 80-300 steps in every regime and is
// 10-100x more accurate on the generating function.

/// Error tolerance per step on the generating-function scale (|exp(phi)| <= 1).
const PROTEIN_ODE_TOL: f64 = 1e-6;
/// First step as a fraction of the fastest time scale.
const PROTEIN_ODE_C0: f64 = 0.05;
/// Hard cap on accepted steps (adaptive mode); the per-step error target is then
/// no longer guaranteed but the integration still covers the full horizon
/// as far as the cap allows.
const PROTEIN_ODE_MAX_STEPS: usize = 20_000;

#[inline(always)]
fn rphi1(z: f64) -> f64 {
    if z.abs() > 1e-4 { z.exp_m1() / z } else { 1.0 + z * 0.5 + z * z * (1.0 / 6.0) }
}

#[inline(always)]
fn protein_integrand(v: Complex64, b: f64) -> Complex64 {
    let u = v - Complex64::new(1.0, 0.0);
    u * b / (Complex64::new(1.0, 0.0) - u * b)
}

/// Per-step scalars shared by every grid point (the only point-dependent input
/// of a step is g2).
#[derive(Clone, Copy)]
struct ProteinStepConsts {
    h: f64,
    /// -gamma h (real part of the integral of a over the step)
    dar: f64,
    /// (k_p / gamma_p) e^{-gamma_p t} (1 - e^{-gamma_p h}); times g2 gives the rest
    dac: f64,
    /// e^{-beta h}
    eb: f64,
    /// int_0^h e^{-beta (h - s)} ds
    i0: f64,
}

#[inline(always)]
fn protein_step_consts(t: f64, h: f64, beta: f64, gamma: f64, k_p: f64, gamma_p: f64) -> ProteinStepConsts {
    ProteinStepConsts {
        h,
        dar: -gamma * h,
        dac: (k_p / gamma_p) * (-gamma_p * t).exp() * (-(-gamma_p * h).exp_m1()),
        eb: (-beta * h).exp(),
        i0: h * rphi1(-beta * h),
    }
}

/// One exponential step with precomputed per-step scalars.
#[inline(always)]
fn protein_exp_step(
    c: &ProteinStepConsts,
    v: Complex64,
    w: Complex64,
    g2: Complex64,
    beta: f64,
    gamma: f64,
) -> (Complex64, Complex64) {
    let one = Complex64::new(1.0, 0.0);
    // Exact integral of a(t) over the step.
    let da = Complex64::new(c.dar, 0.0) + g2 * c.dac;
    let abar = da / c.h;
    let e = da.exp();
    let p1 = if da.norm() > 1e-4 { (e - one) / da } else { one + da * 0.5 + da * da * (1.0 / 6.0) };
    let w_new = e * w + p1 * (gamma * c.h);
    let s = abar + beta;
    let z = s * c.h;
    let i1 = if z.norm() > 1e-3 {
        (e - c.eb) / s
    } else {
        (one + z * 0.5 + z * z * (1.0 / 6.0)) * (c.h * c.eb)
    };
    let goa = abar.inv() * gamma;
    let v_new = v * c.eb + ((w + goa) * i1 - goa * c.i0) * beta;
    (v_new, w_new)
}

#[derive(Clone, Copy)]
struct ProteinState {
    v: Complex64,
    w: Complex64,
    phi: Complex64,
}

/// Trial step result for one grid point: new state, Simpson increment, and the
/// weighted local error estimate (adaptive mode only).
#[inline(always)]
fn protein_trial(
    s: &ProteinState,
    c_half1: &ProteinStepConsts,
    c_half2: &ProteinStepConsts,
    c_full: &ProteinStepConsts,
    g2: Complex64,
    b: f64,
    beta: f64,
    gamma: f64,
    estimate: bool,
) -> (Complex64, Complex64, Complex64, f64) {
    let h = c_full.h;
    let f0 = protein_integrand(s.v, b);
    let (vm, wm) = protein_exp_step(c_half1, s.v, s.w, g2, beta, gamma);
    let (v2, w2) = protein_exp_step(c_half2, vm, wm, g2, beta, gamma);
    let fm = protein_integrand(vm, b);
    let f2 = protein_integrand(v2, b);
    let simpson = (f0 + fm * 4.0 + f2) * (h / 6.0);
    let err = if estimate {
        let trap = (f0 + fm * 2.0 + f2) * (h / 4.0);
        let (v1, _w1) = protein_exp_step(c_full, s.v, s.w, g2, beta, gamma);
        let wgt = s.phi.re.exp();
        wgt * ((simpson - trap).norm() + b * h * (v2 - v1).norm() / 3.0)
    } else {
        0.0
    };
    (v2, w2, simpson, err)
}

/// Compute the ProteinBursty log-PGF over all grid points.
///
/// `fixed = Some(steps)` replays a previously accepted step sequence (used for
/// finite-difference stencils); `None` selects the steps adaptively.  Returns
/// the log-PGF and the step sequence actually used.
fn protein_pgf_sched(
    g0: &[Complex64],
    g1: &[Complex64],
    g2: &[Complex64],
    p: &[f64],
    _min_fudge: f64,
    _max_fudge: f64,
    fixed: Option<&[f64]>,
) -> (Vec<Complex64>, Vec<f64>) {
    let n_grid = g0.len();
    let (b, beta, gamma, k_p, gamma_p) = (p[0], p[1], p[2], p[3], p[4]);
    let one = Complex64::new(1.0, 0.0);

    let lam_fast = beta.max(gamma + 2.0 * k_p).max(gamma_p);
    let lam_slow = beta.min(gamma).min(gamma_p);
    let t_end = (1.0 / PROTEIN_ODE_TOL).ln() / lam_slow
        + (2.0 * k_p / gamma).ln().max(0.0) / gamma_p;

    let mut states: Vec<ProteinState> = (0..n_grid)
        .map(|k| ProteinState { v: g0[k] + one, w: g1[k] + one, phi: Complex64::new(0.0, 0.0) })
        .collect();
    let mut trial: Vec<(Complex64, Complex64, Complex64, f64)> =
        vec![(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), 0.0); n_grid];

    let adaptive = fixed.is_none();
    let mut steps: Vec<f64> = Vec::with_capacity(fixed.map_or(256, |f| f.len()));
    let mut t = 0.0_f64;
    let mut h = PROTEIN_ODE_C0 / lam_fast;
    let h_min = 1e-12 / lam_fast;
    let par = n_grid >= PAR_THRESHOLD;

    loop {
        match fixed {
            Some(f) => {
                if steps.len() >= f.len() { break; }
                h = f[steps.len()];
            }
            None => {
                if t >= t_end || steps.len() >= PROTEIN_ODE_MAX_STEPS { break; }
                h = h.min(t_end - t);
            }
        }
        let estimate = adaptive;
        let c_half1 = protein_step_consts(t, 0.5 * h, beta, gamma, k_p, gamma_p);
        let c_half2 = protein_step_consts(t + 0.5 * h, 0.5 * h, beta, gamma, k_p, gamma_p);
        let c_full = protein_step_consts(t, h, beta, gamma, k_p, gamma_p);
        if par {
            states.par_iter().zip(trial.par_iter_mut()).enumerate().for_each(|(k, (s, out))| {
                *out = protein_trial(s, &c_half1, &c_half2, &c_full, g2[k], b, beta, gamma, estimate);
            });
        } else {
            for k in 0..n_grid {
                trial[k] = protein_trial(&states[k], &c_half1, &c_half2, &c_full, g2[k], b, beta, gamma, estimate);
            }
        }
        let mut fac = 1.0;
        if adaptive {
            let err = if par {
                trial.par_iter().map(|x| x.3).reduce(|| 0.0_f64, f64::max)
            } else {
                trial.iter().map(|x| x.3).fold(0.0_f64, f64::max)
            };
            if err > PROTEIN_ODE_TOL && h > h_min {
                h *= (0.9 * (PROTEIN_ODE_TOL / err).powf(1.0 / 3.0)).max(0.2);
                continue;
            }
            fac = (0.9 * (PROTEIN_ODE_TOL / err.max(1e-300)).powf(1.0 / 3.0)).clamp(0.2, 2.0);
        }
        let accept = |(s, tr): (&mut ProteinState, &(Complex64, Complex64, Complex64, f64))| {
            s.v = tr.0; s.w = tr.1; s.phi += tr.2;
        };
        if par {
            states.par_iter_mut().zip(trial.par_iter()).for_each(accept);
        } else {
            states.iter_mut().zip(trial.iter()).for_each(accept);
        }
        t += h;
        steps.push(h);
        h *= fac;
    }

    let gf_log: Vec<Complex64> = if par {
        states.into_par_iter().map(|s| s.phi).collect()
    } else {
        states.into_iter().map(|s| s.phi).collect()
    };
    (gf_log, steps)
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
    let (gf, mx, lims, _) =
        protein_bursty_core_sched(p_log, limits, fit_unspliced, protein_limit, min_fudge, max_fudge, None);
    (gf, mx, lims)
}

/// `protein_bursty_core` with an optional fixed integration schedule; also returns
/// the schedule actually used, so gradient stencils can reuse it.
fn protein_bursty_core_sched(
    p_log: &[f64],
    limits: &[usize],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
    fixed: Option<&[f64]>,
) -> (Vec<Complex64>, [usize; 3], [usize; 3], Vec<f64>) {
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

    let (gf_log, sched) = protein_pgf_sched(&g0, &g1, &g2, &p, min_fudge, max_fudge, fixed);
    let n = gf_log.len();
    let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
        gf_log.par_iter().map(|z| z.exp()).collect()
    } else {
        gf_log.iter().map(|z| z.exp()).collect()
    };
    (gf, mx, lims, sched)
}

// ============================================================================
// eval_model_pss — 2-D models
// ============================================================================

/// Evaluate log-PGF for any supported 2-D bio_model given g0/g1 mesh slices.
/// Log-PGF of any two-species model on a (g0, g1) mesh; `p` is linear-scale.
/// Panics on an unknown model: the Python-facing entry points validate first.
fn pgf_2d(
    bio_model: &str,
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
) -> Vec<Complex64> {
    match bio_model {
        "Constitutive" => pgf_constitutive(g0, g1, p),
        "Extrinsic"    => pgf_extrinsic(g0, g1, p),
        "Delay"        => pgf_delay(g0, g1, p),
        "DelayedSplicing" => pgf_delayed_splicing(g0, g1, p),
        "Bursty" | "CIR" => {
            let t = fixed_quad_t * (1.0 / p[1] + 1.0 / p[2] + 1.0);
            if bio_model == "Bursty" { pgf_bursty(g0, g1, p, t, quad_order) }
            else { pgf_cir(g0, g1, p, t, quad_order) }
        }
        other => panic!("Unknown bio_model {other:?}"),
    }
}

/// `pgf_2d` with a Python error for unknown models.
fn eval_pgf_2d(
    bio_model: &str,
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    fixed_quad_t: f64,
    quad_order: usize,
) -> PyResult<Vec<Complex64>> {
    validate_bio_model_2d(bio_model, "eval_model_pss_2d")?;
    Ok(pgf_2d(bio_model, g0, g1, p, fixed_quad_t, quad_order))
}

/// |x| / sum|x| over a raw inverse-FFT result.
fn normalize_abs(pss_raw: Vec<f64>) -> Vec<f64> {
    let par = pss_raw.len() >= PAR_THRESHOLD;
    let abs_sum: f64 = if par { pss_raw.par_iter().map(|x| x.abs()).sum() }
                       else { pss_raw.iter().map(|x| x.abs()).sum() };
    if par { pss_raw.par_iter().map(|x| x.abs() / abs_sum).collect() }
    else { pss_raw.iter().map(|x| x.abs() / abs_sum).collect() }
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
    validate_bio_model_2d(bio_model, "eval_model_pss_2d")?;
    let n_amb = match amb_model {
        "None" => 0usize,
        "Equal" => 1,
        "Unequal" => 2,
        other => return verr(format!("eval_model_pss_2d: unknown amb_model {other:?} (use None, Equal, Unequal)")),
    };
    validate_limits(&limits, if n_amb == 0 { 2 } else { 3 }, "eval_model_pss_2d")?;
    if let Some(n) = n_bio_params(bio_model) {
        if p_log.len() < n {
            return verr(format!("eval_model_pss_2d: {bio_model} takes {n} parameters, got {}", p_log.len()));
        }
    }
    if let Some(sl) = samp_log.as_ref() {
        if sl.len() < 2 {
            return verr(format!("eval_model_pss_2d: samp_log needs 2 entries, got {}", sl.len()));
        }
    }
    if n_amb > 0 && amb_log.as_ref().map_or(true, |a| a.len() < n_amb) {
        return verr(format!("eval_model_pss_2d: amb_model={amb_model} needs {n_amb} amb_log entries"));
    }
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
    Ok(normalize_abs(pss_raw))
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

    let gf_log = pgf_2d(bio_model, g0, g1, &p, fixed_quad_t, quad_order);

    let n = gf_log.len();
    let gf: Vec<Complex64> = if n >= PAR_THRESHOLD {
        gf_log.par_iter().map(|z| z.exp()).collect()
    } else {
        gf_log.iter().map(|z| z.exp()).collect()
    };
    normalize_abs(irfftn_2d(&gf, l0, l1))
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
    py: Python<'_>,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
) -> PyResult<Vec<f64>> {
    validate_limits(&limits, 3, "eval_model_pss_protein_bursty")?;
    if p_log.len() != 5 {
        return verr(format!("eval_model_pss_protein_bursty: expected 5 parameters, got {}", p_log.len()));
    }
    if protein_limit.is_nan() || protein_limit <= 0.0 {
        // +inf is allowed: it disables coarse-graining of the protein axis.
        return verr(format!("eval_model_pss_protein_bursty: protein_limit must be > 0, got {protein_limit}"));
    }
    if !(min_fudge > 0.0) || !(max_fudge > 0.0) {
        return verr(format!("eval_model_pss_protein_bursty: min_fudge/max_fudge must be > 0"));
    }
    // Released GIL: lets Python threads run several genes' integrations at once
    // (the Nelder-Mead pre-search in inference.py relies on this).
    let pss_raw = py.allow_threads(|| {
        let (gf, mx, lims) =
            protein_bursty_core(&p_log, &limits, fit_unspliced, protein_limit, min_fudge, max_fudge);
        irfftn_3d(&gf, mx[0], mx[2], lims[0], lims[1], lims[2])
    });
    Ok(normalize_abs(pss_raw))
}

/// `eval_model_pss_protein_bursty` that also returns the integration schedule
/// used and replays `fixed_schedule` when given, so `CMEModel.pinned_schedule`
/// pins the Rust path exactly as it pins the Python one.
#[pyfunction]
#[pyo3(signature = (p_log, limits, fit_unspliced, protein_limit, min_fudge, max_fudge, fixed_schedule=None))]
#[allow(clippy::too_many_arguments)]
fn eval_model_pss_protein_bursty_sched(
    py: Python<'_>,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
    fixed_schedule: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    validate_limits(&limits, 3, "eval_model_pss_protein_bursty_sched")?;
    if p_log.len() != 5 {
        return verr(format!("eval_model_pss_protein_bursty_sched: expected 5 parameters, got {}", p_log.len()));
    }
    if protein_limit.is_nan() || protein_limit <= 0.0 {
        return verr(format!("eval_model_pss_protein_bursty_sched: protein_limit must be > 0, got {protein_limit}"));
    }
    if !(min_fudge > 0.0) || !(max_fudge > 0.0) {
        return verr(format!("eval_model_pss_protein_bursty_sched: min_fudge/max_fudge must be > 0"));
    }
    Ok(py.allow_threads(|| {
        eval_model_pss_protein_bursty_seq_sched(
            &p_log, &limits, fit_unspliced, protein_limit, min_fudge, max_fudge, fixed_schedule.as_deref(),
        )
    }))
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

// ----------------------------------------------------------------------------
// Input validation helpers — turn would-be panics (index out of bounds, clamp
// with min > max, chunks of size 0) into Python ValueErrors with a message.
// ----------------------------------------------------------------------------

/// Number of gene-specific (biological) parameters per built-in model.
fn n_bio_params(bio_model: &str) -> Option<usize> {
    match bio_model {
        "Constitutive" => Some(2),
        "Bursty" | "CIR" | "Extrinsic" | "Delay" | "DelayedSplicing" => Some(3),
        "ProteinBursty" => Some(5),
        _ => None,
    }
}

fn verr<T>(msg: String) -> pyo3::PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg))
}

/// lb/ub must have equal length (== n_params when known), be finite, and lb <= ub.
fn validate_bounds(lb: &[f64], ub: &[f64], n_params: Option<usize>, caller: &str) -> pyo3::PyResult<()> {
    if lb.len() != ub.len() {
        return verr(format!("{caller}: lb and ub must have the same length (got {} and {})", lb.len(), ub.len()));
    }
    if let Some(n) = n_params {
        if lb.len() < n {
            return verr(format!("{caller}: expected at least {n} parameter bounds, got {}", lb.len()));
        }
    }
    for i in 0..lb.len() {
        if !lb[i].is_finite() || !ub[i].is_finite() {
            return verr(format!("{caller}: bounds must be finite (index {i}: lb={}, ub={})", lb[i], ub[i]));
        }
        if lb[i] > ub[i] {
            return verr(format!("{caller}: lb[{i}]={} exceeds ub[{i}]={}", lb[i], ub[i]));
        }
    }
    Ok(())
}

/// limits must have exactly `n_expected` entries, all >= 1.
fn validate_limits(limits: &[usize], n_expected: usize, caller: &str) -> pyo3::PyResult<()> {
    if limits.len() != n_expected {
        return verr(format!("{caller}: expected {n_expected} grid limits, got {}", limits.len()));
    }
    if let Some(i) = limits.iter().position(|&l| l == 0) {
        return verr(format!("{caller}: grid limit at index {i} is 0; every limit must be >= 1"));
    }
    Ok(())
}

/// Every observed count must lie strictly inside the PMF grid.
fn validate_coords_2d(u_idx: &[u64], s_idx: &[u64], l0: usize, l1: usize, caller: &str) -> pyo3::PyResult<()> {
    if u_idx.len() != s_idx.len() {
        return verr(format!("{caller}: u_idx and s_idx must have the same length"));
    }
    if let Some(&m) = u_idx.iter().max() {
        if m as usize >= l0 {
            return verr(format!("{caller}: observed unspliced count {m} >= grid limit {l0}; increase padding"));
        }
    }
    if let Some(&m) = s_idx.iter().max() {
        if m as usize >= l1 {
            return verr(format!("{caller}: observed spliced count {m} >= grid limit {l1}; increase padding"));
        }
    }
    Ok(())
}

/// x0_list must be n_genes x (>=1 restarts) x n_params.
fn validate_x0_list(x0_list: &[Vec<Vec<f64>>], n_genes: usize, n_params: usize, caller: &str) -> pyo3::PyResult<()> {
    if x0_list.len() != n_genes {
        return verr(format!("{caller}: x0_list has {} genes, search data has {n_genes}", x0_list.len()));
    }
    for (g, restarts) in x0_list.iter().enumerate() {
        if restarts.is_empty() {
            return verr(format!("{caller}: gene {g} has no restart initial points"));
        }
        if let Some(bad) = restarts.iter().find(|x| x.len() != n_params) {
            return verr(format!("{caller}: gene {g} initial point has {} entries, expected {n_params}", bad.len()));
        }
    }
    Ok(())
}

/// Optimizer termination codes surfaced to Python via `return_status=True`.
const OPT_STATUS_CONVERGED: u8 = 0;
const OPT_STATUS_MAXITER: u8 = 1;
const OPT_STATUS_LINESEARCH: u8 = 2;
const OPT_STATUS_ERROR: u8 = 3;

/// Per-iteration L-BFGS-B trace on stderr when `MONOD_LBFGSB_VERBOSE` is set
/// (debugging aid; columns: n, iter, nfev, nskip, col, ||proj g||, f).
fn lbfgsb_verbose() -> bool {
    std::env::var_os("MONOD_LBFGSB_VERBOSE").is_some()
}

fn solver_status_code(status: &lbfgsb_rs_pure::Status) -> u8 {
    match status {
        lbfgsb_rs_pure::Status::Converged => OPT_STATUS_CONVERGED,
        lbfgsb_rs_pure::Status::MaxIter => OPT_STATUS_MAXITER,
        lbfgsb_rs_pure::Status::LineSearchFailure => OPT_STATUS_LINESEARCH,
        #[allow(unreachable_patterns)]
        _ => OPT_STATUS_ERROR,
    }
}

fn split_results(results: Vec<(Vec<f64>, f64, u8)>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<u8>) {
    let n = results.len();
    let (mut p, mut k, mut st) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for (x, kld, s) in results {
        p.push(x); k.push(kld); st.push(s);
    }
    (p, k, st)
}

fn pack_results(py: Python<'_>, params: Vec<Vec<f64>>, klds: Vec<f64>, status: Vec<u8>, return_status: bool) -> PyObject {
    if return_status {
        (params, klds, status).into_py(py)
    } else {
        (params, klds).into_py(py)
    }
}

/// L-BFGS-B settings shared by every optimizer entry point.
#[derive(Clone, Copy)]
struct LbfgsbSettings {
    maxiter: usize,
    ftol: f64,
    gtol: f64,
    m_mem: usize,
}

/// Minimize `obj` (value and gradient at log10 parameters) within [lb, ub]
/// from `x0`.  Returns (x, f, status code).  Pure Rust: called inside rayon
/// tasks with the GIL released.
fn lbfgsb_minimize_objective<O>(
    obj: &O,
    x0: &[f64],
    lb: &[f64],
    ub: &[f64],
    settings: LbfgsbSettings,
) -> (Vec<f64>, f64, u8)
where
    O: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let mut x: Vec<f64> = (0..x0.len()).map(|i| x0[i].max(lb[i]).min(ub[i])).collect();
    let mut solver = LBFGSB::new(settings.m_mem)
        .with_max_iter(settings.maxiter)
        .with_pgtol(settings.gtol)
        .with_ftol(settings.ftol)
        .with_verbose(lbfgsb_verbose());
    let mut f_and_grad = |xx: &[f64]| obj(xx);
    // ftol and pgtol stopping live inside the solver; the callback only observes.
    let mut callback = |_info: &lbfgsb_rs_pure::IterationInfo, _x: &[f64]| IterationControl::Continue;
    match solver.minimize_with_callback(&mut x, lb, ub, &mut f_and_grad, &mut callback) {
        Ok(sol) => (sol.x, sol.f, solver_status_code(&sol.status)),
        Err(_) => {
            // Bounds are validated up front, so this is a numerical failure:
            // return the projected x0, flagged as an error.
            let (f0, _) = obj(&x);
            (x, f0, OPT_STATUS_ERROR)
        }
    }
}

/// Minimize from every start in `x0_list` and keep the best.  A later start
/// replaces the incumbent only if it improves the KLD by more than 1 %
/// (`ERR_THRESH`), matching the scipy path in inference.py.
fn best_of_restarts<O>(
    obj: &O,
    x0_list: &[Vec<f64>],
    lb: &[f64],
    ub: &[f64],
    settings: LbfgsbSettings,
) -> (Vec<f64>, f64, u8)
where
    O: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let mut best_x: Vec<f64> = (0..lb.len()).map(|i| x0_list[0][i].max(lb[i]).min(ub[i])).collect();
    let mut best_kld = f64::INFINITY;
    let mut best_status: u8 = OPT_STATUS_ERROR;
    for x0 in x0_list {
        let (x_opt, kld, st) = lbfgsb_minimize_objective(obj, x0, lb, ub, settings);
        if kld < best_kld * ERR_THRESH {
            best_x = x_opt;
            best_kld = kld;
            best_status = st;
        }
    }
    (best_x, best_kld, best_status)
}

/// Optimize every gene in parallel.  `objective(gi)` builds gene `gi`'s
/// objective (KLD and gradient as a function of the log10 parameters); the
/// restart loop, solver settings and status handling are shared by all model
/// families.  Call inside `py.allow_threads` and `run_with_pool`.
fn optimize_genes_parallel<F, O>(
    x0_list: &[Vec<Vec<f64>>],
    lb: &[f64],
    ub: &[f64],
    settings: LbfgsbSettings,
    objective: F,
) -> Vec<(Vec<f64>, f64, u8)>
where
    F: Fn(usize) -> O + Sync,
    O: Fn(&[f64]) -> (f64, Vec<f64>),
{
    (0..x0_list.len())
        .into_par_iter()
        .map(|gi| {
            let obj = objective(gi);
            best_of_restarts(&obj, &x0_list[gi], lb, ub, settings)
        })
        .collect()
}

/// Forward finite-difference gradient of `kld_at` around `x`, given the value
/// `kld0` at `x`.
fn fd_gradient<K>(x: &[f64], fd_eps: f64, kld0: f64, kld_at: K) -> Vec<f64>
where
    K: Fn(&[f64]) -> f64,
{
    (0..x.len())
        .map(|i| {
            let mut x_eps = x.to_vec();
            x_eps[i] += fd_eps;
            (kld_at(&x_eps) - kld0) / fd_eps
        })
        .collect()
}

/// Central finite-difference gradient of `kld_at` around `x`.  Used where the
/// objective carries a small noise floor (far-tail histogram bins near the
/// FFT's absolute precision): a wider, symmetric stencil keeps the noise out
/// of the gradient at the cost of 2n evaluations.
fn fd_gradient_central<K>(x: &[f64], fd_eps: f64, kld_at: K) -> Vec<f64>
where
    K: Fn(&[f64]) -> f64,
{
    (0..x.len())
        .map(|i| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += fd_eps;
            xm[i] -= fd_eps;
            (kld_at(&xp) - kld_at(&xm)) / (2.0 * fd_eps)
        })
        .collect()
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
    let _n_layers = layers.len();
    let n_cells = layers[0].shape()[0];
    let n_genes = layers[0].shape()[1];
    for (l, arr) in layers.iter().enumerate() {
        if !arr.is_c_contiguous() {
            return verr(format!("compute_moments: layer {l} must be C-contiguous (use np.ascontiguousarray)"));
        }
        if arr.shape() != [n_cells, n_genes] {
            return verr(format!("compute_moments: layer {l} has shape {:?}, expected ({n_cells}, {n_genes})", arr.shape()));
        }
    }

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
    pub adata:            RuAnnData,             // owns raw counts, var metadata, annotation
    pub coords:           Vec<Vec<Vec<i64>>>,    // [gene][microstate][layer] — kept for 3-D optimizers
    pub coords_by_layer:  Vec<Vec<Vec<u64>>>,    // [layer][gene][microstate] — pre-split for 2-D optimizer
    pub freqs:            Vec<Vec<f64>>,         // [gene][microstate]
    pub limits:           Vec<usize>,            // flat [gene * n_layers + layer]
    pub moments:          Vec<HashMap<String, f64>>, // derived cache for MoM initialisation
    pub n_layers:         usize,
    pub n_cells:          usize,
    pub n_genes:          usize,
    pub gene_names:       Vec<String>,
    pub hist_type:        String,
    pub layer_names:      Vec<String>,
    pub k:                Option<usize>,
    pub epochs:           Option<usize>,
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
        adata:            RuAnnData,
        coords:           Vec<Vec<Vec<i64>>>,
        freqs:            Vec<Vec<f64>>,
        limits:           Vec<usize>,
        moments:          Vec<HashMap<String, f64>>,
        n_layers:         usize,
        n_cells:          usize,
        n_genes:          usize,
        gene_names:       Vec<String>,
        hist_type:        String,
        layer_names:      Vec<String>,
        k:                Option<usize>,
        epochs:           Option<usize>,
    ) -> Self {
        let coords_by_layer = build_coords_by_layer(&coords, n_genes, n_layers);
        Self {
            adata, coords, coords_by_layer, freqs, limits, moments,
            n_layers, n_cells, n_genes, gene_names, hist_type,
            layer_names, k, epochs,
        }
    }

    /// Read gene_log_lengths from adata.var (returns None if column absent).
    pub fn extract_gene_log_lengths(&self) -> Option<Vec<f64>> {
        self.adata.var.columns.get("gene_log_lengths").and_then(|s| {
            if let SeriesData::Float64 { values } = s {
                Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            } else { None }
        })
    }

    /// Read gene_log_lengths_spliced from adata.var (returns None if column absent).
    pub fn extract_gene_log_lengths_spliced(&self) -> Option<Vec<f64>> {
        self.adata.var.columns.get("gene_log_lengths_spliced").and_then(|s| {
            if let SeriesData::Float64 { values } = s {
                Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            } else { None }
        })
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

/// Construct a minimal `RuAnnData` from raw count arrays.
///
/// `layers_row_major[l]` must be row-major flat (n_cells × n_genes): `[c * n_genes + g]`.
/// `gene_log_lengths` / `gene_log_lengths_spliced` are stored in `var.columns` when provided.
fn make_minimal_adata(
    gene_names:               &[String],
    layer_names:              &[String],
    layers_row_major:         &[Vec<i64>],
    n_cells:                  usize,
    n_genes:                  usize,
    gene_log_lengths:         Option<Vec<f64>>,
    gene_log_lengths_spliced: Option<Vec<f64>>,
) -> RuAnnData {
    let obs_names: Vec<String> = (0..n_cells).map(|i| i.to_string()).collect();

    let mut var_columns: IndexMap<String, SeriesData> = IndexMap::new();
    if let Some(gll) = gene_log_lengths {
        var_columns.insert(
            "gene_log_lengths".to_string(),
            SeriesData::Float64 { values: gll.into_iter().map(Some).collect() },
        );
    }
    if let Some(glls) = gene_log_lengths_spliced {
        var_columns.insert(
            "gene_log_lengths_spliced".to_string(),
            SeriesData::Float64 { values: glls.into_iter().map(Some).collect() },
        );
    }

    let obs = DataFrame {
        index_name: "obs_names".to_string(),
        index:      obs_names,
        columns:    IndexMap::new(),
    };
    let var = DataFrame {
        index_name: "var_names".to_string(),
        index:      gene_names.to_vec(),
        columns:    var_columns,
    };

    let x_flat = if layers_row_major.is_empty() {
        vec![0i64; n_cells * n_genes]
    } else {
        layers_row_major[0].clone()
    };
    let x = MatrixData::Dense {
        array: ArrayData { shape: vec![n_cells, n_genes], values: ArrayValue::Int64(x_flat) },
    };

    let mut layers: IndexMap<String, MatrixData> = IndexMap::new();
    for (l, lname) in layer_names.iter().enumerate() {
        layers.insert(lname.clone(), MatrixData::Dense {
            array: ArrayData {
                shape: vec![n_cells, n_genes],
                values: ArrayValue::Int64(layers_row_major[l].clone()),
            },
        });
    }

    RuAnnData {
        format_version: "0.1.0".to_string(),
        x,
        obs,
        var,
        raw:  None,
        layers,
        obsm: IndexMap::new(),
        varm: IndexMap::new(),
        obsp: IndexMap::new(),
        varp: IndexMap::new(),
        uns:  IndexMap::new(),
    }
}

// ── Thin Python wrapper ────────────────────────────────────────────────────────

#[pyclass(name = "SearchData")]
pub struct PySearchData {
    pub inner: SearchData,
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
        self.inner.extract_gene_log_lengths()
            .map(|v| PyArray1::from_vec_bound(py, v))
    }

    #[getter]
    fn gene_log_lengths_spliced<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.extract_gene_log_lengths_spliced()
            .map(|v| PyArray1::from_vec_bound(py, v))
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
    fn layers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray<i64, ndarray::Ix3>>> {
        let nc = self.inner.n_cells;
        let ng = self.inner.n_genes;
        let nl = self.inner.n_layers;
        let mut dense: Vec<Vec<i64>> = Vec::with_capacity(nl);
        for lname in &self.inner.layer_names {
            let mat = self.inner.adata.layers.get(lname)
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                    format!("layer '{}' not found in adata", lname)
                ))?;
            let (_, _, flat) = matrix_to_dense_i64(mat)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            // flat is row-major [c * ng + g]
            dense.push(flat);
        }
        let arr = Array3::from_shape_fn((nl, nc, ng), |(l, c, g)| dense[l][c * ng + g]);
        Ok(arr.into_pyarray_bound(py))
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
        // Serialize layers as row-major flat vecs from adata.layers.
        let layers_data: Vec<Vec<i64>> = self.inner.layer_names.iter()
            .map(|lname| {
                let mat = self.inner.adata.layers.get(lname)
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                        format!("layer '{}' not found in adata", lname)
                    ))?;
                let (_, _, flat) = matrix_to_dense_i64(mat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
                Ok(flat)
            })
            .collect::<PyResult<_>>()?;
        d.set_item("layers_data",      layers_data)?;
        d.set_item("n_layers",         self.inner.n_layers)?;
        d.set_item("n_cells",          self.inner.n_cells)?;
        d.set_item("n_genes",          self.inner.n_genes)?;
        d.set_item("gene_names",       &self.inner.gene_names)?;
        d.set_item("hist_type",        &self.inner.hist_type)?;
        d.set_item("layer_names",      &self.inner.layer_names)?;
        d.set_item("gene_log_lengths",         self.inner.extract_gene_log_lengths())?;
        d.set_item("gene_log_lengths_spliced", self.inner.extract_gene_log_lengths_spliced())?;
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
        // layers_data stored as row-major [c * n_genes + g] per layer.
        let layers_data: Vec<Vec<i64>>  = get("layers_data")?.extract()?;
        self.inner.n_layers             = get("n_layers")?.extract()?;
        self.inner.n_cells              = get("n_cells")?.extract()?;
        self.inner.n_genes              = get("n_genes")?.extract()?;
        self.inner.gene_names           = get("gene_names")?.extract()?;
        self.inner.hist_type            = get("hist_type")?.extract()?;
        self.inner.layer_names          = get("layer_names")?.extract()?;
        let gene_log_lengths: Option<Vec<f64>>         = get("gene_log_lengths")?.extract()?;
        let gene_log_lengths_spliced: Option<Vec<f64>> = get("gene_log_lengths_spliced")?.extract()?;
        self.inner.k                    = get("k")?.extract()?;
        self.inner.epochs               = get("epochs")?.extract()?;
        // Reconstruct adata from serialised components.
        self.inner.adata = make_minimal_adata(
            &self.inner.gene_names,
            &self.inner.layer_names,
            &layers_data,
            self.inner.n_cells,
            self.inner.n_genes,
            gene_log_lengths,
            gene_log_lengths_spliced,
        );
        // Recompute derived field.
        self.inner.coords_by_layer = build_coords_by_layer(
            &self.inner.coords, self.inner.n_genes, self.inner.n_layers,
        );
        Ok(())
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
    ) -> PyResult<Vec<Vec<f64>>> {
        if n_bio_params(&bio_model).is_none() {
            return verr(format!(
                "mom_x0_all: unknown bio_model {bio_model:?}; method-of-moments initialisation is \
                 available for Constitutive, Bursty, CIR, Extrinsic, Delay, DelayedSplicing, ProteinBursty"
            ));
        }
        validate_bounds(&lb, &ub, None, "mom_x0_all")?;
        if let Some(n) = n_bio_params(&bio_model) {
            if lb.len() < n {
                return verr(format!("mom_x0_all: {bio_model} needs at least {n} bounds, got {}", lb.len()));
            }
        }
        Ok(self.inner.mom_x0_all_inner(
            &bio_model, &seq_model, &amb_model,
            &lb, &ub,
            samp_list.as_deref(),
        ))
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
    if n_layers == 0 {
        return verr("searchdata_from_arrays: at least one layer is required".to_string());
    }
    for (l, arr) in layers.iter().enumerate() {
        let sh = arr.shape();
        if !arr.is_c_contiguous() {
            return verr(format!("searchdata_from_arrays: layer {l} must be C-contiguous (use np.ascontiguousarray)"));
        }
        if sh[0] != n_cells || sh[1] != n_genes {
            return verr(format!("searchdata_from_arrays: layer {l} has shape {sh:?}, expected ({n_cells}, {n_genes})"));
        }
    }
    {
        let lsh = limits.shape();
        if lsh[0] < n_layers || lsh[1] < n_genes {
            return verr(format!("searchdata_from_arrays: limits has shape {lsh:?}, need at least ({n_layers}, {n_genes})"));
        }
    }
    if coords.len() != n_genes || freqs.len() != n_genes || gene_names.len() != n_genes {
        return verr(format!("searchdata_from_arrays: coords ({}), freqs ({}) and gene_names ({}) must all have n_genes={n_genes} entries",
            coords.len(), freqs.len(), gene_names.len()));
    }
    {
        let lsl = limits.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("limits array must be C-contiguous"))?;
        let ncol = limits.shape()[1];
        for g in 0..n_genes {
            if coords[g].len() != freqs[g].len() {
                return verr(format!("searchdata_from_arrays: gene {g} has {} microstates but {} frequencies", coords[g].len(), freqs[g].len()));
            }
            for ms in &coords[g] {
                if ms.len() != n_layers {
                    return verr(format!("searchdata_from_arrays: gene {g} microstate has {} entries, expected n_layers={n_layers}", ms.len()));
                }
                for l in 0..n_layers {
                    let lim = lsl[l * ncol + g];
                    if ms[l] < 0 || ms[l] >= lim {
                        return verr(format!("searchdata_from_arrays: gene {g} layer {l}: observed count {} outside grid [0, {lim}); increase padding", ms[l]));
                    }
                }
            }
        }
    }

    // Borrow row-major slices from numpy (for adata storage) and build column-major
    // copies (for moments computation).
    let mut layers_row_major: Vec<Vec<i64>> = Vec::with_capacity(n_layers);
    let mut layers_col_major: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
    for arr in &layers {
        let slice = arr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err(
            "layer arrays must be C-contiguous",
        ))?;
        layers_row_major.push(slice.to_vec());
        let mut col_major = vec![0f64; n_cells * n_genes];
        for c in 0..n_cells {
            for g in 0..n_genes {
                col_major[g * n_cells + c] = slice[c * n_genes + g] as f64;
            }
        }
        layers_col_major.push(col_major);
    }

    let limits_n_cols = limits.shape()[1];
    let limits_slice = limits.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("limits array must be C-contiguous")
    })?;
    let limits_flat: Vec<usize> = (0..n_genes)
        .flat_map(|g| (0..n_layers).map(move |l| limits_slice[l * limits_n_cols + g] as usize))
        .collect();

    let moments = py.allow_threads(|| {
        compute_moments_inner(&layers_col_major, &layer_names, n_cells, n_genes)
    });

    let adata = make_minimal_adata(
        &gene_names, &layer_names, &layers_row_major,
        n_cells, n_genes, gene_log_lengths, gene_log_lengths_spliced,
    );

    Ok(PySearchData {
        inner: SearchData::new(
            adata, coords, freqs, limits_flat, moments,
            n_layers, n_cells, n_genes, gene_names, hist_type,
            layer_names, k, epochs,
        ),
    })
}

// ============================================================================
// Cell partitioning: subset SearchData by cluster assignment
// ============================================================================

/// Partition a `SearchData` into per-cluster subsets based on hard cell assignments.
///
/// For each cluster `k` in `0..n_clusters`, filters the raw layer counts to
/// cells where `assigns[c] == k`, recomputes M (max + `padding`), histograms,
/// and moments, and returns a new `SearchData`.  Clusters with no assigned
/// cells yield `None`.
///
/// Parameters
/// ----------
/// sd         : source Rust SearchData (must have ≥ 2 layers)
/// assigns    : per-cell cluster index, length == sd.n_cells
/// n_clusters : total number of clusters k (length of returned Vec)
/// padding    : added to per-gene per-layer max to set M (default 10)
///
/// Returns
/// -------
/// List of length `n_clusters`; each entry is a `SearchData` or `None`.
#[pyfunction]
#[pyo3(signature = (sd, assigns, n_clusters, padding=10))]
fn partition_searchdata_2d(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    assigns: Vec<usize>,
    n_clusters: usize,
    padding: usize,
) -> PyResult<Vec<Option<PySearchData>>> {
    let sd_ref = sd.borrow();
    let nc    = sd_ref.inner.n_cells;
    let ng    = sd_ref.inner.n_genes;
    let nl    = sd_ref.inner.n_layers;
    if assigns.len() != nc {
        return verr(format!("partition_searchdata_2d: assigns has {} entries, search data has {nc} cells", assigns.len()));
    }
    if n_clusters == 0 {
        return verr("partition_searchdata_2d: n_clusters must be >= 1".to_string());
    }
    if let Some(&bad) = assigns.iter().find(|&&a| a >= n_clusters) {
        return verr(format!("partition_searchdata_2d: cluster label {bad} >= n_clusters={n_clusters}"));
    }

    // Extract raw row-major layers from adata under the GIL.
    let mut raw_layers: Vec<Vec<i64>> = Vec::with_capacity(nl);
    for lname in &sd_ref.inner.layer_names {
        let mat = sd_ref.inner.adata.layers.get(lname)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("layer '{}' not found in adata", lname)))?;
        let (_, _, flat) = matrix_to_dense_i64(mat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        raw_layers.push(flat);
    }

    let gene_log_lengths = sd_ref.inner.extract_gene_log_lengths();
    let gene_log_lengths_spliced = sd_ref.inner.adata.var.columns
        .get("gene_log_lengths_spliced")
        .and_then(|s| if let ruanndata::SeriesData::Float64 { values } = s {
            Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect::<Vec<f64>>())
        } else { None });
    let layer_names  = sd_ref.inner.layer_names.clone();
    let gene_names   = sd_ref.inner.gene_names.clone();
    let k_opt        = sd_ref.inner.k;
    let epochs_opt   = sd_ref.inner.epochs;
    drop(sd_ref);

    // Build per-cluster SearchData in parallel (over clusters) — pure Rust.
    let cluster_data: Vec<Option<SearchData>> = py.allow_threads(|| {
        (0..n_clusters).into_par_iter().map(|k| {
            // Cell indices for this cluster.
            let cell_idx: Vec<usize> = assigns.iter().enumerate()
                .filter_map(|(c, &a)| if a == k { Some(c) } else { None })
                .collect();
            if cell_idx.is_empty() { return None; }
            let new_nc = cell_idx.len();

            // Build row-major filtered layers and column-major for moments.
            let mut rm_layers: Vec<Vec<i64>>  = Vec::with_capacity(nl);
            let mut cm_layers: Vec<Vec<f64>>  = Vec::with_capacity(nl);
            for layer in &raw_layers {
                let mut rm = Vec::with_capacity(new_nc * ng);
                let mut cm = vec![0f64; ng * new_nc];
                for (new_c, &old_c) in cell_idx.iter().enumerate() {
                    for g in 0..ng {
                        let v = layer[old_c * ng + g];
                        rm.push(v);
                        cm[g * new_nc + new_c] = v as f64;
                    }
                }
                rm_layers.push(rm);
                cm_layers.push(cm);
            }

            // Limits: max per (gene, layer) + padding.
            let mut limits_flat = vec![0usize; ng * nl];
            for (l, layer) in rm_layers.iter().enumerate() {
                for g in 0..ng {
                    let max_val = (0..new_nc)
                        .map(|c| layer[c * ng + g])
                        .max()
                        .unwrap_or(0)
                        .max(0) as usize;
                    limits_flat[g * nl + l] = max_val.saturating_add(padding);
                }
            }

            // Per-gene histograms via shared helper (rayon within each cluster).
            let hist_pairs: Vec<(Vec<Vec<i64>>, Vec<f64>)> = (0..ng).map(|g| {
                let cols: Vec<Vec<i64>> = rm_layers.iter()
                    .map(|layer| (0..new_nc).map(|c| layer[c * ng + g]).collect())
                    .collect();
                build_gene_histogram(&cols, new_nc)
            }).collect();
            let (coords, freqs): (Vec<Vec<Vec<i64>>>, Vec<Vec<f64>>) =
                hist_pairs.into_iter().unzip();

            // Moments.
            let moments = compute_moments_inner(&cm_layers, &layer_names, new_nc, ng);

            // Build minimal adata.
            let adata = make_minimal_adata(
                &gene_names, &layer_names, &rm_layers,
                new_nc, ng,
                gene_log_lengths.clone(),
                gene_log_lengths_spliced.clone(),
            );

            Some(SearchData::new(
                adata, coords, freqs, limits_flat, moments,
                nl, new_nc, ng,
                gene_names.clone(), "unique".to_string(),
                layer_names.clone(), k_opt, epochs_opt,
            ))
        }).collect()
    });

    Ok(cluster_data.into_iter()
        .map(|opt| opt.map(|inner| PySearchData { inner }))
        .collect())
}

// ============================================================================
// KMeans initialisation (optional feature "kmeans")
// ============================================================================

/// KMeans++ initialisation followed by Lloyd iterations using matrixmultiply-accelerated
/// distance computation.
///
/// Distance matrix is computed as:
///   D[i,k] = ||X[i]||² + ||C[k]||² - 2 · (X @ C^T)[i,k]
/// where the matrix multiply uses `matrixmultiply::dgemm` (SIMD, cache-blocked).
///
/// KMeans++ seeding updates min-distances with rayon so each new centroid's
/// contribution is computed across cells in parallel.
///
/// Parameters
/// ----------
/// data     – (n_samples × n_features) C-contiguous f64 array
/// k        – number of clusters
/// max_iter – maximum Lloyd iterations
/// tol      – stop when max centroid shift < tol
/// seed     – RNG seed for reproducibility
#[cfg(feature = "kmeans")]
fn kmeans_fit(
    data: &ndarray::Array2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> ndarray::Array1<usize> {
    use rand_xoshiro::rand_core::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256Plus;

    let n = data.nrows();
    let d = data.ncols();
    assert!(k > 0 && n >= k, "k must be > 0 and <= n_samples");

    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    // ── KMeans++ initialisation ─────────────────────────────────────────────
    // centroids_flat: row-major (k × d), built incrementally.
    let mut centroids_flat = vec![0f64; k * d];

    // First centroid: pick uniformly at random.
    let first = (rng.next_u64() as usize) % n;
    for (dst, &src) in centroids_flat[..d].iter_mut().zip(data.row(first).iter()) {
        *dst = src;
    }

    // min_dists[i] = squared distance from cell i to its nearest centroid so far.
    let mut min_dists = vec![f64::INFINITY; n];

    for ki in 1..k {
        // Own the just-added centroid so the rayon closure can capture it without
        // holding a borrow on centroids_flat while we later write to it.
        let c_prev: Vec<f64> = centroids_flat[(ki - 1) * d..ki * d].to_vec();

        // Rayon: update min_dists with the distance to c_prev.
        min_dists.par_iter_mut().enumerate().for_each(|(i, d_min)| {
            let diff_sq: f64 = data.row(i).iter().zip(c_prev.iter())
                .map(|(a, b)| (a - b) * (a - b)).sum();
            if diff_sq < *d_min { *d_min = diff_sq; }
        });

        // Sample next centroid proportional to min_dists (D² weighting).
        let total: f64 = min_dists.iter().sum();
        let mut target = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
        let mut chosen = n - 1;
        for (i, &dist) in min_dists.iter().enumerate() {
            target -= dist;
            if target <= 0.0 { chosen = i; break; }
        }
        for (dst, &src) in centroids_flat[ki * d..(ki + 1) * d].iter_mut()
            .zip(data.row(chosen).iter())
        {
            *dst = src;
        }
    }

    // ── Lloyd iterations ────────────────────────────────────────────────────
    // Precompute squared row norms of data (constant across iterations).
    let x_norms: Vec<f64> = (0..n)
        .map(|i| data.row(i).iter().map(|v| v * v).sum())
        .collect();

    let mut assignments = vec![0usize; n];
    // xct[i * k + j] = dot(X[i], C[j])  — reused every iteration.
    let mut xct = vec![0f64; n * k];

    for _ in 0..max_iter {
        // Centroid squared norms.
        let c_norms: Vec<f64> = (0..k)
            .map(|j| centroids_flat[j * d..(j + 1) * d].iter().map(|v| v * v).sum())
            .collect();

        // X @ C^T via dgemm: A=(n×d) row-major, B=C^T via (k×d) with transposed strides.
        // SAFETY: all three buffers are valid, non-overlapping, and sized correctly.
        unsafe {
            matrixmultiply::dgemm(
                n, d, k,
                1.0,
                data.as_ptr(),           d as isize, 1,   // A: (n×d) row-major
                centroids_flat.as_ptr(), 1, d as isize,   // B = C^T: row-stride=1, col-stride=d
                0.0,
                xct.as_mut_ptr(),        k as isize, 1,   // result: (n×k) row-major
            );
        }

        // Assign each cell to its nearest centroid.
        let new_assignments: Vec<usize> = (0..n).into_par_iter().map(|i| {
            let xi_sq = x_norms[i];
            let base  = i * k;
            let mut best_j = 0;
            let mut best_d = f64::INFINITY;
            for j in 0..k {
                let dist = xi_sq + c_norms[j] - 2.0 * xct[base + j];
                if dist < best_d { best_d = dist; best_j = j; }
            }
            best_j
        }).collect();

        let changed: usize = new_assignments.iter().zip(assignments.iter())
            .filter(|(a, b)| a != b).count();
        assignments = new_assignments;
        if changed == 0 { break; }

        // Update centroids and check convergence.
        let mut new_centroids = vec![0f64; k * d];
        let mut counts = vec![0usize; k];
        for (i, &cl) in assignments.iter().enumerate() {
            let dst = &mut new_centroids[cl * d..(cl + 1) * d];
            for (s, &v) in dst.iter_mut().zip(data.row(i).iter()) { *s += v; }
            counts[cl] += 1;
        }

        let mut max_shift = 0f64;
        for j in 0..k {
            let cnt = counts[j].max(1) as f64;
            let mut shift_sq = 0f64;
            for g in 0..d {
                let new_val = new_centroids[j * d + g] / cnt;
                let delta   = new_val - centroids_flat[j * d + g];
                shift_sq += delta * delta;
                centroids_flat[j * d + g] = new_val;
            }
            max_shift = max_shift.max(shift_sq.sqrt());
        }

        if max_shift < tol { break; }
    }

    ndarray::Array1::from(assignments)
}

/// Initialise the EM posterior Q using KMeans on log1p-normalised (U+S) expression.
///
/// Replicates the Python `_initialize_Q` preprocessing:
///   1. Sum layers 0 and 1 → S_total  (n_cells × n_genes)
///   2. Normalize each cell to 10k counts
///   3. log1p transform; NaN → 0
///   4. KMeans++ + Lloyd via matrixmultiply-accelerated distance matrix
///   5. Biased Q: uniform(0,1) random matrix, then Q[cell, cluster] = 0.9
///
/// Parameters
/// ----------
/// layers : (n_layers, n_cells, n_genes) int64 numpy array
/// k      : number of clusters
/// seed   : RNG seed for reproducibility (default 0)
///
/// Returns
/// -------
/// (Q, labels) — Q is (n_cells, k) float64; labels is (n_cells,) int64
#[cfg(feature = "kmeans")]
#[pyfunction]
#[pyo3(signature = (layers, k, seed=0))]
fn initialize_q_kmeans<'py>(
    py: Python<'py>,
    layers: numpy::PyReadonlyArray3<'py, i64>,
    k: usize,
    seed: u64,
) -> PyResult<(
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<i64>>,
)> {
    use ndarray::{Array1, Array2};
    use rand_xoshiro::rand_core::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256Plus;

    let arr = layers.as_array();
    let shape = arr.shape();
    if shape.len() != 3 || shape[0] < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "layers must be (n_layers, n_cells, n_genes) with at least 2 layers",
        ));
    }
    let n_cells = shape[1];
    let n_genes = shape[2];

    // Step 1: S_total = layers[0] + layers[1]  (n_cells × n_genes, f64)
    let mut s_total = Array2::<f64>::zeros((n_cells, n_genes));
    for g in 0..n_genes {
        for c in 0..n_cells {
            s_total[[c, g]] = arr[[0, c, g]] as f64 + arr[[1, c, g]] as f64;
        }
    }

    // Step 2: per-cell normalize to 10k counts (NaN-safe: if tots==0 leave row as-is)
    for c in 0..n_cells {
        let tot: f64 = s_total.row(c).sum();
        if tot > 0.0 {
            let scale = 1e4 / tot;
            for g in 0..n_genes { s_total[[c, g]] *= scale; }
        }
    }

    // Step 3: log1p; NaN → 0 (row with zero total is already 0)
    s_total.mapv_inplace(|v| if v.is_nan() { 0.0 } else { v.ln_1p() });

    // Step 4: KMeans++ + Lloyd (custom, matrixmultiply-accelerated, GIL-free)
    let labels_arr: Array1<usize> = py.allow_threads(|| {
        kmeans_fit(&s_total, k, 300, 1e-4, seed)
    });

    // Step 5: build biased Q.
    let mut rng2 = Xoshiro256Plus::seed_from_u64(seed.wrapping_add(1));
    let mut q_flat = vec![0f64; n_cells * k];
    for v in q_flat.iter_mut() {
        *v = (rng2.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    }
    for (c, &lab) in labels_arr.iter().enumerate() {
        q_flat[c * k + lab] = 0.9;
    }
    for c in 0..n_cells {
        let row_sum: f64 = q_flat[c * k..(c + 1) * k].iter().sum();
        if row_sum > 0.0 {
            for v in q_flat[c * k..(c + 1) * k].iter_mut() { *v /= row_sum; }
        }
    }

    let q_arr = Array2::from_shape_vec((n_cells, k), q_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    let lab_arr = labels_arr.mapv(|v| v as i64);

    Ok((
        q_arr.into_pyarray_bound(py),
        lab_arr.into_pyarray_bound(py),
    ))
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

/// Build a unique-microstate histogram for a single gene.
///
/// `cols[l]` holds the counts for layer `l` across all cells: `cols[l][c]`.
/// Returns `(microstates, frequencies)` where each microstate is a `Vec<i64>`
/// with one entry per layer, and frequencies are `count / n_cells`.
///
/// Uses the same dense-counting / sort-based fallback logic as `make_state_dist`.
fn build_gene_histogram(cols: &[Vec<i64>], n_cells: usize) -> (Vec<Vec<i64>>, Vec<f64>) {
    let n_layers = cols.len();

    let max_per_layer: Vec<usize> = cols
        .iter()
        .map(|col| col.iter().copied().max().unwrap_or(0).max(0) as usize)
        .collect();

    let total_states: usize = max_per_layer.iter().map(|&m| m + 1).product();

    if total_states <= DENSE_THRESHOLD {
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
                    .map(|(col, &s)| col[cell] as usize * s)
                    .sum();
                table[idx] += 1;
            }
            let mut unique: Vec<Vec<i64>> = Vec::new();
            let mut freqs: Vec<f64> = Vec::new();
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
            let mut flat = fb.borrow_mut();
            let mut order = ob.borrow_mut();
            let flat_len = n_cells * n_layers;
            if flat.len() < flat_len { flat.resize(flat_len, 0); }
            if order.len() < n_cells { order.resize(n_cells, 0); }
            let flat = &mut flat[..flat_len];
            let order = &mut order[..n_cells];
            for cell in 0..n_cells {
                for (l, col) in cols.iter().enumerate() {
                    flat[cell * n_layers + l] = col[cell];
                }
            }
            for (i, v) in order.iter_mut().enumerate() { *v = i; }
            order.sort_unstable_by(|&a, &b| {
                flat[a * n_layers..(a + 1) * n_layers]
                    .cmp(&flat[b * n_layers..(b + 1) * n_layers])
            });
            let mut unique: Vec<Vec<i64>> = Vec::new();
            let mut counts: Vec<usize> = Vec::new();
            let mut prev_start = usize::MAX;
            for &idx in order.iter() {
                let row_start = idx * n_layers;
                if prev_start != usize::MAX
                    && flat[prev_start..prev_start + n_layers] == flat[row_start..row_start + n_layers]
                {
                    *counts.last_mut().unwrap() += 1;
                } else {
                    unique.push(flat[row_start..row_start + n_layers].to_vec());
                    counts.push(1);
                    prev_start = row_start;
                }
            }
            let freqs: Vec<f64> = counts.iter().map(|&c| c as f64 / n_cells as f64).collect();
            (unique, freqs)
        })})
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
    for (l, arr) in layers.iter().enumerate() {
        if !arr.is_c_contiguous() {
            return verr(format!("make_state_dist: layer {l} must be C-contiguous (use np.ascontiguousarray)"));
        }
        if arr.shape() != [n_cells, n_genes] {
            return verr(format!("make_state_dist: layer {l} has shape {:?}, expected ({n_cells}, {n_genes})", arr.shape()));
        }
        if let Some(v) = raw[l].iter().find(|&&v| v < 0) {
            return verr(format!("make_state_dist: layer {l} contains a negative count ({v}); layers must be non-negative integer counts"));
        }
    }

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
        MatrixData::ShiftedClrCsr { .. } => Err(
            "layer is stored as a shifted-CLR matrix, not integer counts; \
             monod requires raw count layers".to_string(),
        ),
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
/// `adata_sub.layers` holds the subsetted count matrices (from `slice_var`).
struct H5adInner {
    adata_sub: RuAnnData,
}

/// Core h5ad ingestion: read file, filter/select genes, compute histograms.
///
/// Must be called outside the GIL (inside `py.allow_threads`).
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
    Ok(H5adInner { adata_sub })
}

// ── H5adInner accessors ───────────────────────────────────────────────────────

/// Extract (gene_names, coords, freqs, limits) from an `H5adInner` for the
/// `load_histograms_h5ad` return type.
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
    if padding == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "padding must be >= 1 so that the largest observed count lies inside the PMF grid",
        ));
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
    if padding == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "padding must be >= 1 so that the largest observed count lies inside the PMF grid",
        ));
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

    let result = py.allow_threads(|| -> Result<SearchData, String> {
        let mut inner = load_h5ad_inner(
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

        // Compute moments from adata_sub.layers (row-major) — transpose to column-major.
        let layers_col_major: Vec<Vec<f64>> = layer_names.iter()
            .map(|lname| {
                let mat = inner.adata_sub.layers.get(lname)
                    .ok_or_else(|| format!("layer '{}' missing after slice_var", lname))?;
                let (nr, nc, flat_row) = matrix_to_dense_i64(mat)?;
                let mut col = vec![0f64; nr * nc];
                for c in 0..nr { for g in 0..nc { col[g * nr + c] = flat_row[c * nc + g] as f64; } }
                Ok(col)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let moments = compute_moments_inner(&layers_col_major, &layer_names, n_cells, n_genes);

        let (coords, freqs) = extract_state_dists(&inner.adata_sub, n_genes, n_layers)?;
        let limits          = extract_limits(&inner.adata_sub, n_genes, n_layers)?;
        let gene_names_out  = inner.adata_sub.var.index.clone();

        // Read optional length columns into owned Vecs, then store under canonical names.
        let gene_log_lengths: Option<Vec<f64>> = log_lengths_col.as_deref()
            .and_then(|col| inner.adata_sub.var.columns.get(col))
            .and_then(|s| if let SeriesData::Float64 { values } = s {
                Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            } else { None });
        let gene_log_lengths_spliced: Option<Vec<f64>> = spliced_log_lengths_col.as_deref()
            .and_then(|col| inner.adata_sub.var.columns.get(col))
            .and_then(|s| if let SeriesData::Float64 { values } = s {
                Some(values.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            } else { None });
        if let Some(ref vals) = gene_log_lengths {
            inner.adata_sub.var.columns.insert(
                "gene_log_lengths".to_string(),
                SeriesData::Float64 { values: vals.iter().map(|&v| Some(v)).collect() },
            );
        }
        if let Some(ref vals) = gene_log_lengths_spliced {
            inner.adata_sub.var.columns.insert(
                "gene_log_lengths_spliced".to_string(),
                SeriesData::Float64 { values: vals.iter().map(|&v| Some(v)).collect() },
            );
        }

        Ok(SearchData::new(
            inner.adata_sub, coords, freqs, limits, moments,
            n_layers, n_cells, n_genes, gene_names_out, hist_type,
            layer_names.to_vec(), None, None,
        ))
    });

    result
        .map(|sd| PySearchData { inner: sd })
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
    /// 0 = geometric-burst production, 1 = deterministic production,
    /// 2 = first-order, 3 = delayed first-order (`=>`)
    kind:       u8,
    rate_idx:   u32,
    /// kind 0: burst-param index; kinds 2/3: reactant species index; else -1
    extra1:     i32,
    /// kind 0: burst species index; kind 3: index of the inverse-delay
    /// parameter in `params`; else -1
    extra2:     i32,
    prod_start: u32,
    prod_end:   u32,
    /// kind 3: position of this reaction's delay among the network's distinct
    /// delays (order of first appearance); 0 otherwise
    tau_slot:   u32,
}

/// Build the reaction table from the flat arrays passed in from Python.
fn build_rxn_infos(
    kinds: &[u8],
    rate_idxs: &[u32],
    extra1: &[i32],
    extra2: &[i32],
    prod_off: &[u32],
) -> Vec<RxnInfo> {
    let mut tau_params: Vec<i32> = Vec::new();
    (0..kinds.len())
        .map(|i| {
            let tau_slot = if kinds[i] == 3 {
                match tau_params.iter().position(|&t| t == extra2[i]) {
                    Some(s) => s,
                    None => { tau_params.push(extra2[i]); tau_params.len() - 1 }
                }
            } else {
                0
            };
            RxnInfo {
                kind:       kinds[i],
                rate_idx:   rate_idxs[i],
                extra1:     extra1[i],
                extra2:     extra2[i],
                prod_start: prod_off[i],
                prod_end:   prod_off[i + 1],
                tau_slot:   tau_slot as u32,
            }
        })
        .collect()
}

/// Build an N-dimensional complex mesh for irfftn.
/// Returns (g, mx_sizes, n_grid) where g[s] is the flat grid for species s.
/// Last axis has rfft half-length (l/2+1); all others full.
fn build_mesh_nd(limits: &[usize]) -> (Vec<Vec<Complex64>>, Vec<usize>, usize) {
    let n = limits.len();
    let mx: Vec<usize> = limits.iter().enumerate()
        .map(|(i, &l)| if i == n - 1 { l / 2 + 1 } else { l }).collect();
    let n_grid: usize = mx.iter().product();
    (build_mesh_axes(&mx, limits), mx, n_grid)
}

// ============================================================================
// Custom reaction networks: exponential integrator
// ============================================================================
//
// The characteristic equations of a first-order network are
//
//     du/dt = A u + N(u, u_delayed),      dphi/dt = S(u),
//
// where A is the real, constant first-order rate matrix, N collects the terms
// a linear system does not have (several products from one reaction, as in
// S -> S + P, and the product terms of delayed `=>` reactions, which read the
// state tau ago) and S is the production source.  Every step propagates the
// linear part exactly through exp(A h) and treats N with the second-order
// exponential Runge-Kutta scheme (ETD2RK):
//
//     a       = E u + P1 N(u, t)
//     u_{n+1} = a + P2 (N(a, t+h) - N(u, t)),
//     E = exp(A h),  P1 = h phi_1(A h),  P2 = h phi_2(A h),
//
// so the stiff rates never limit the step.  phi is accumulated with Simpson's
// rule on two half-steps.  The step size is controlled by comparing one full
// step with the two half-steps (whose difference also feeds a Richardson
// extrapolation of u), taking the maximum error over all grid points so the
// whole grid shares one schedule; the accepted steps are returned and can be
// replayed (finite-difference stencils use the base point's schedule).
// Delayed states come from the accepted history by cubic Hermite
// interpolation (every knot stores its derivative); a delay shorter than the
// step is read off the step's own predictor.  Mirrors `ReactionNetwork.eval_pgf`.

const CUSTOM_ODE_TOL: f64 = 1e-6;        // per-step error target on the generating-function scale
const CUSTOM_ODE_C0: f64 = 0.05;         // first step as a fraction of the fastest time scale
const CUSTOM_ODE_U_END: f64 = 1e-8;      // stop once exp(Re phi)·|u| is below this everywhere (for a delay's length, if any)
const CUSTOM_ODE_MAX_STEPS: usize = 20_000;
const CUSTOM_ODE_H_MAX: f64 = 1e4;       // step cap in units of the slowest time scale
const CUSTOM_MAX_SPECIES: usize = 8;
const CUSTOM_FD_EPS: f64 = 1e-4;         // central finite-difference step for custom-network gradients (log10 units)

/// Small dense real matrix, row-major.
#[derive(Clone)]
struct SmallMat {
    n: usize,
    a: Vec<f64>,
}

impl SmallMat {
    fn zeros(n: usize) -> Self { SmallMat { n, a: vec![0.0; n * n] } }
    fn eye(n: usize) -> Self {
        let mut m = SmallMat::zeros(n);
        for i in 0..n { m.a[i * n + i] = 1.0; }
        m
    }
    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> f64 { self.a[i * self.n + j] }
    #[inline(always)]
    fn set(&mut self, i: usize, j: usize, v: f64) { self.a[i * self.n + j] = v; }
    fn mul(&self, o: &SmallMat) -> SmallMat {
        let n = self.n;
        let mut r = SmallMat::zeros(n);
        for i in 0..n {
            for k in 0..n {
                let aik = self.a[i * n + k];
                for j in 0..n {
                    r.a[i * n + j] += aik * o.a[k * n + j];
                }
            }
        }
        r
    }
    fn norm1(&self) -> f64 {
        let n = self.n;
        (0..n)
            .map(|j| (0..n).map(|i| self.a[i * n + j].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }
    /// exp(self): scaling and squaring around a degree-18 Taylor polynomial
    /// (the scaled norm is at most 1/2, so the truncation error is below 1e-22).
    fn expm(&self) -> SmallMat {
        let n = self.n;
        let nrm = self.norm1();
        let s = if nrm > 0.5 { (nrm / 0.5).log2().ceil() as i32 } else { 0 };
        let scale = 0.5_f64.powi(s);
        let mut x = self.clone();
        for v in x.a.iter_mut() { *v *= scale; }
        let mut r = SmallMat::eye(n);
        for k in (1..=18).rev() {
            let t = x.mul(&r);
            r = SmallMat::eye(n);
            let inv = 1.0 / (k as f64);
            for (ri, ti) in r.a.iter_mut().zip(t.a.iter()) { *ri += ti * inv; }
        }
        for _ in 0..s { r = r.mul(&r); }
        r
    }
    /// out = self · v for a complex vector.
    #[inline(always)]
    fn apply(&self, v: &[Complex64], out: &mut [Complex64]) {
        let n = self.n;
        for i in 0..n {
            let mut s = Complex64::new(0.0, 0.0);
            for j in 0..n { s += v[j] * self.a[i * n + j]; }
            out[i] = s;
        }
    }
}

/// Step operators for one step size: E = exp(A h), P1 = h phi_1(A h), P2 = h phi_2(A h),
/// read off exp([[A h, I, 0], [0, 0, I], [0, 0, 0]]).
struct CustomStepOps { e: SmallMat, p1: SmallMat, p2: SmallMat }

fn custom_step_ops(a: &SmallMat, h: f64) -> CustomStepOps {
    let n = a.n;
    let mut m = SmallMat::zeros(3 * n);
    for i in 0..n {
        for j in 0..n { m.set(i, j, a.get(i, j) * h); }
        m.set(i, n + i, 1.0);
        m.set(n + i, 2 * n + i, 1.0);
    }
    let x = m.expm();
    let mut e = SmallMat::zeros(n);
    let mut p1 = SmallMat::zeros(n);
    let mut p2 = SmallMat::zeros(n);
    for i in 0..n {
        for j in 0..n {
            e.set(i, j, x.get(i, j));
            p1.set(i, j, h * x.get(i, n + j));
            p2.set(i, j, h * x.get(i, 2 * n + j));
        }
    }
    CustomStepOps { e, p1, p2 }
}

/// A network at fixed (linear-scale) parameters, split into its linear part
/// and the remainder the exponential integrator treats explicitly.
struct CustomSystem<'a> {
    n:        usize,
    a:        SmallMat,
    /// per reaction: contributes to N (multi-product first-order or delayed)
    nonlin:   Vec<bool>,
    /// N vanishes identically: u is propagated by exp(A h) alone
    linear:   bool,
    rxns:     &'a [RxnInfo],
    prod_sp:  &'a [u32],
    prod_st:  &'a [u32],
    params:   &'a [f64],
    /// delay lengths by tau slot
    taus:     Vec<f64>,
    lam_fast: f64,
    lam_slow: f64,
}

fn custom_system<'a>(
    n: usize, rxns: &'a [RxnInfo], prod_sp: &'a [u32], prod_st: &'a [u32], params: &'a [f64],
) -> CustomSystem<'a> {
    let mut a = SmallMat::zeros(n);
    let mut nonlin = vec![false; rxns.len()];
    let mut taus: Vec<f64> = Vec::new();
    let mut rates: Vec<f64> = Vec::new();
    for (ri, r) in rxns.iter().enumerate() {
        let k = params[r.rate_idx as usize];
        let (ps, pe) = (r.prod_start as usize, r.prod_end as usize);
        match r.kind {
            2 => {
                let i = r.extra1 as usize;
                a.a[i * n + i] -= k;
                for q in ps..pe {
                    let j = prod_sp[q] as usize;
                    a.a[i * n + j] += k * prod_st[q] as f64;
                }
                nonlin[ri] = !(pe == ps || (pe == ps + 1 && prod_st[ps] == 1));
                rates.push(k);
            }
            3 => {
                let i = r.extra1 as usize;
                a.a[i * n + i] -= k;
                nonlin[ri] = true;
                let slot = r.tau_slot as usize;
                if taus.len() <= slot { taus.resize(slot + 1, 0.0); }
                taus[slot] = 1.0 / params[r.extra2 as usize];
                rates.push(k);
            }
            _ => {}
        }
    }
    if rates.is_empty() {
        rates = rxns.iter().map(|r| params[r.rate_idx as usize]).collect();
    }
    let rates: Vec<f64> = rates.into_iter().filter(|&k| k > 0.0).collect();
    let diag: Vec<f64> = (0..n).map(|i| -a.a[i * n + i]).filter(|&d| d > 0.0).collect();
    let lam_fast = rates.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    let lam_slow = if diag.is_empty() {
        rates.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        diag.iter().cloned().fold(f64::INFINITY, f64::min)
    };
    let lam_slow = if lam_slow.is_finite() { lam_slow } else { lam_fast };
    let linear = !nonlin.iter().any(|&b| b);
    CustomSystem { n, a, nonlin, linear, rxns, prod_sp, prod_st, params, taus, lam_fast, lam_slow }
}

const CUSTOM_MAX_DELAYS: usize = 4;
type CVec = [Complex64; CUSTOM_MAX_SPECIES];
type DelSet = [CVec; CUSTOM_MAX_DELAYS];

#[inline(always)]
fn czero_vec() -> CVec { [Complex64::new(0.0, 0.0); CUSTOM_MAX_SPECIES] }
#[inline(always)]
fn dzero() -> DelSet { [czero_vec(); CUSTOM_MAX_DELAYS] }

/// Remainder N(u): multi-product first-order terms k (Z(u) − 1 − Σ s_j u_j) and
/// delayed product terms k (Z(u_del) − 1).  `udel[slot]` holds the delayed
/// state for each distinct delay.
#[inline(always)]
fn custom_nonlin(sys: &CustomSystem, u: &CVec, udel: &DelSet, out: &mut CVec) {
    let one = Complex64::new(1.0, 0.0);
    for v in out.iter_mut().take(sys.n) { *v = Complex64::new(0.0, 0.0); }
    for (ri, r) in sys.rxns.iter().enumerate() {
        if !sys.nonlin[ri] { continue; }
        let k = sys.params[r.rate_idx as usize];
        let i = r.extra1 as usize;
        let (ps, pe) = (r.prod_start as usize, r.prod_end as usize);
        if r.kind == 2 {
            let mut z = one;
            let mut lin = Complex64::new(0.0, 0.0);
            for q in ps..pe {
                let uj = u[sys.prod_sp[q] as usize];
                z *= (one + uj).powu(sys.prod_st[q]);
                lin += uj * (sys.prod_st[q] as f64);
            }
            out[i] += (z - one - lin) * k;
        } else {
            let ud = &udel[r.tau_slot as usize];
            let mut z = one;
            for q in ps..pe {
                z *= (one + ud[sys.prod_sp[q] as usize]).powu(sys.prod_st[q]);
            }
            out[i] += (z - one) * k;
        }
    }
}

/// Source S(u) = dphi/dt from the production reactions.
#[inline(always)]
fn custom_source(sys: &CustomSystem, u: &CVec) -> Complex64 {
    let one = Complex64::new(1.0, 0.0);
    let mut s = Complex64::new(0.0, 0.0);
    for r in sys.rxns.iter() {
        let k = sys.params[r.rate_idx as usize];
        match r.kind {
            0 => {
                let b = sys.params[r.extra1 as usize];
                let ui = u[r.extra2 as usize];
                s += ui * (k * b) / (one - ui * b);
            }
            1 => {
                let mut z = one;
                for q in r.prod_start as usize..r.prod_end as usize {
                    z *= (one + u[sys.prod_sp[q] as usize]).powu(sys.prod_st[q]);
                }
                s += (z - one) * k;
            }
            _ => {}
        }
    }
    s
}

/// Full right-hand side A u + N(u, u_del) at one point (history derivatives).
#[inline(always)]
fn custom_rhs(sys: &CustomSystem, u: &CVec, udel: &DelSet) -> CVec {
    let n = sys.n;
    let mut out = czero_vec();
    sys.a.apply(&u[..n], &mut out[..n]);
    if !sys.linear {
        let mut tmp = czero_vec();
        custom_nonlin(sys, u, udel, &mut tmp);
        for i in 0..n { out[i] += tmp[i]; }
    }
    out
}

#[inline(always)]
fn cvec_lerp(a: &CVec, b: &CVec, w: f64, n: usize) -> CVec {
    let mut out = czero_vec();
    for i in 0..n { out[i] = a[i] + (b[i] - a[i]) * w; }
    out
}

/// One ETD2RK step of size `h` from `u` at time `t`.  `del_start[slot]` is the
/// delayed state at the start of the step; `del_end[slot]` the one at its end
/// where `end_known[slot]` (the history reaches that far, delay ≥ h), otherwise
/// it is read off the line through (t, u) and (t + h, predictor).
#[inline(always)]
fn custom_etd_step(
    sys: &CustomSystem, u: &CVec, h: f64, ops: &CustomStepOps,
    del_start: &DelSet, del_end: &DelSet, end_known: &[bool; CUSTOM_MAX_DELAYS],
) -> CVec {
    let n = sys.n;
    let mut out = czero_vec();
    if sys.linear {
        ops.e.apply(&u[..n], &mut out[..n]);
        return out;
    }
    let mut n0 = czero_vec();
    custom_nonlin(sys, u, del_start, &mut n0);
    let mut a = czero_vec();
    let mut tmp = czero_vec();
    ops.e.apply(&u[..n], &mut a[..n]);
    ops.p1.apply(&n0[..n], &mut tmp[..n]);
    for i in 0..n { a[i] += tmp[i]; }
    let mut del_a = *del_end;
    for s in 0..sys.taus.len() {
        if !end_known[s] { del_a[s] = cvec_lerp(u, &a, (h - sys.taus[s]) / h, n); }
    }
    let mut n1 = czero_vec();
    custom_nonlin(sys, &a, &del_a, &mut n1);
    for i in 0..n { n1[i] -= n0[i]; }
    ops.p2.apply(&n1[..n], &mut tmp[..n]);
    for i in 0..n { out[i] = a[i] + tmp[i]; }
    out
}

/// Delayed states of one grid point for the three stage times of a step:
/// t − τ (always in the history), t + h/2 − τ and t + h − τ (in the history
/// only when the delay reaches back that far).
struct PointDel {
    d0: DelSet,
    dm: DelSet,
    d1: DelSet,
    has_m: [bool; CUSTOM_MAX_DELAYS],
    has_1: [bool; CUSTOM_MAX_DELAYS],
}

struct CustomTrial { um: CVec, u2: CVec, dphi: Complex64, err: f64 }

/// Two half-steps (Simpson for phi) and, when `estimate`, one full step for the
/// error estimate.
#[inline(always)]
fn custom_trial(
    sys: &CustomSystem, u: &CVec, phi: Complex64, h: f64,
    ops_h: &CustomStepOps, ops_h2: &CustomStepOps, pd: &PointDel, estimate: bool,
) -> CustomTrial {
    let n = sys.n;
    let n_del = sys.taus.len();
    let hh = 0.5 * h;
    // first half-step
    let um = custom_etd_step(sys, u, hh, ops_h2, &pd.d0, &pd.dm, &pd.has_m);
    // second half-step: starts at t + h/2 with history knots (t, u), (t + h/2, um)
    let mut dm2 = pd.dm;
    for s in 0..n_del {
        if !pd.has_m[s] { dm2[s] = cvec_lerp(u, &um, (hh - sys.taus[s]) / hh, n); }
    }
    let mut d1b = pd.d1;
    let mut k1b = pd.has_1;
    for s in 0..n_del {
        if !pd.has_1[s] && sys.taus[s] >= hh {
            d1b[s] = cvec_lerp(u, &um, (h - sys.taus[s]) / hh, n);
            k1b[s] = true;
        }
    }
    let mut u2 = custom_etd_step(sys, &um, hh, ops_h2, &dm2, &d1b, &k1b);
    let s0 = custom_source(sys, u);
    let sm = custom_source(sys, &um);
    let s2 = custom_source(sys, &u2);
    let dphi = (s0 + sm * 4.0 + s2) * (h / 6.0);
    let mut du = 0.0_f64;
    if !sys.linear {
        // one full step: its difference from the two half-steps estimates the
        // error, and Richardson extrapolation removes the leading error term
        let u1 = custom_etd_step(sys, u, h, ops_h, &pd.d0, &pd.d1, &pd.has_1);
        for i in 0..n {
            du = du.max((u2[i] - u1[i]).norm());
            u2[i] = u2[i] + (u2[i] - u1[i]) / 3.0;
        }
    }
    let mut err = 0.0;
    if estimate {
        let trap = (s0 + sm * 2.0 + s2) * (h / 4.0);
        err = phi.re.exp() * ((dphi - trap).norm() + du);
    }
    CustomTrial { um, u2, dphi, err }
}

/// Accepted history of a delayed network: per knot the time, the state of
/// every grid point (point-major n_grid × n) and, once known, its derivative.
/// `h_ctrl` is the step size the error controller chose for the step that
/// produced the knot (before any clipping to a delay breakpoint); `amp` is
/// max over the grid of exp(Re phi)·|u| there, the weight the error control uses.
struct HistKnot { t: f64, u: Vec<Complex64>, f: Option<Vec<Complex64>>, h_ctrl: f64, amp: f64 }

struct CustomHistory { n: usize, n_grid: usize, knots: std::collections::VecDeque<HistKnot> }

impl CustomHistory {
    /// State at time `s` for every grid point: zero before the observation
    /// time; cubic Hermite interpolation between accepted knots when both
    /// derivatives are known, linear otherwise.  The state jumps from zero to
    /// the initial condition at the observation time: a stage at the end of
    /// a step reads the left limit (`left`), one at the start the right limit,
    /// so a step that lands on a delay sees no forcing and the next one sees
    /// it from its first stage.
    fn at(&self, s: f64, left: bool) -> Vec<Complex64> {
        let len = self.n * self.n_grid;
        // a landing on the onset is exact only up to rounding
        let t0 = self.knots[0].t;
        let tol = 1e-10 * self.knots.back().map_or(1.0, |k| k.t.max(1.0));
        let s = if (s - t0).abs() <= tol { t0 } else { s };
        if s < t0 || (left && s <= t0) {
            return vec![Complex64::new(0.0, 0.0); len];
        }
        let m = self.knots.len();
        let mut lo = 0usize;
        let mut hi = m;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.knots[mid].t > s { hi = mid; } else { lo = mid + 1; }
        }
        if lo >= m { return self.knots[m - 1].u.clone(); }
        let a = &self.knots[lo - 1];
        let b = &self.knots[lo];
        let d = b.t - a.t;
        let th = (s - a.t) / d;
        match (&a.f, &b.f) {
            (Some(fa), Some(fb)) => {
                let h00 = (2.0 * th - 3.0) * th * th + 1.0;
                let h10 = ((th - 2.0) * th + 1.0) * th;
                let h01 = (3.0 - 2.0 * th) * th * th;
                let h11 = (th - 1.0) * th * th;
                (0..len).map(|i| a.u[i] * h00 + fa[i] * (h10 * d) + b.u[i] * h01 + fb[i] * (h11 * d)).collect()
            }
            _ => a.u.iter().zip(b.u.iter()).map(|(x, y)| x + (y - x) * th).collect(),
        }
    }
    fn push(&mut self, t: f64, u: Vec<Complex64>, h_ctrl: f64, amp: f64) {
        self.knots.push_back(HistKnot { t, u, f: None, h_ctrl, amp });
    }
    /// Largest step from `s0` that does not out-run the resolution of the
    /// history it reads: within the window [s0, s0 + h] the step is capped at
    /// twice the step the controller chose there, wherever the history is
    /// large enough to matter (amp ≥ tolerance).  Without this a step that
    /// has grown long during a quiet stretch reads a delayed pulse narrower
    /// than itself only at its stage times, both the full step and the
    /// half-steps miss it, and the pulse is skipped.
    fn max_step(&self, s0: f64, h: f64) -> f64 {
        let m = self.knots.len();
        let mut k = self.knots.partition_point(|kn| kn.t <= s0).max(1);
        let mut h = h;
        while k < m {
            if self.knots[k - 1].t >= s0 + h { break; }
            if self.knots[k].amp >= CUSTOM_ODE_TOL { h = h.min(2.0 * self.knots[k].h_ctrl); }
            k += 1;
        }
        h
    }
    fn prune(&mut self, t_oldest_needed: f64) {
        while self.knots.len() >= 2 && self.knots[1].t <= t_oldest_needed {
            self.knots.pop_front();
        }
    }
}

/// Integrate the characteristics of a custom network over all grid points.
/// Returns (phi per grid point, accepted step sizes).
/// Map a recorded schedule onto new delay breakpoints.  The time axis is
/// stretched piecewise-linearly so that every knot keeps its position relative
/// to the delays (the knot that landed on τ lands on τ'), which makes a
/// replayed result a smooth function of the delays; without it a stencil point
/// would see the delayed forcing switch on inside a step.  Identity when there
/// are no delays, when they are unchanged, or when their order would change.
fn warp_schedule(steps: &[f64], base: &[f64], new: &[f64]) -> Vec<f64> {
    if base.is_empty() || base.len() != new.len() || base == new { return steps.to_vec(); }
    let mut order: Vec<usize> = (0..base.len()).collect();
    order.sort_by(|&a, &b| base[a].partial_cmp(&base[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut bx = vec![0.0_f64];
    let mut by = vec![0.0_f64];
    for &i in &order {
        let (b, nb) = (base[i], new[i]);
        if b <= *bx.last().unwrap() {
            if nb != *by.last().unwrap() { return steps.to_vec(); }
            continue;
        }
        if nb <= *by.last().unwrap() { return steps.to_vec(); }
        bx.push(b);
        by.push(nb);
    }
    let last = bx.len() - 1;
    let f = |t: f64| -> f64 {
        let k = bx.partition_point(|&b| b <= t);
        if k > last { by[last] + (t - bx[last]) }
        else { by[k - 1] + (t - bx[k - 1]) * ((by[k] - by[k - 1]) / (bx[k] - bx[k - 1])) }
    };
    let mut t = 0.0_f64;
    let mut out = Vec::with_capacity(steps.len());
    for &h in steps {
        let t1 = t + h;
        out.push(f(t1) - f(t));
        t = t1;
    }
    out
}

/// Delay lengths in slot order at the given parameters.
fn custom_breakpoints(rxns: &[RxnInfo], params: &[f64]) -> Vec<f64> {
    let mut taus: Vec<f64> = Vec::new();
    for r in rxns {
        if r.kind == 3 {
            let slot = r.tau_slot as usize;
            if taus.len() <= slot { taus.resize(slot + 1, 0.0); }
            taus[slot] = 1.0 / params[r.extra2 as usize];
        }
    }
    taus
}

/// Integrate the characteristics of a custom network over all grid points.
/// `fixed` = (step sizes, delay breakpoints they were recorded with) replays a
/// schedule instead of adapting.  Returns (phi per grid point, accepted steps).
fn custom_pgf_sched(
    g: &[Vec<Complex64>],
    sys: &CustomSystem,
    n_grid: usize,
    fixed: Option<(&[f64], &[f64])>,
) -> (Vec<Complex64>, Vec<f64>) {
    let n = sys.n;
    assert!(n <= CUSTOM_MAX_SPECIES, "custom networks support at most 8 species");
    let n_del = sys.taus.len();
    assert!(n_del <= CUSTOM_MAX_DELAYS, "custom networks support at most 4 distinct delays");
    let tau_max = sys.taus.iter().cloned().fold(0.0_f64, f64::max);
    // nonlinear and delayed networks do enough work per point to parallelise small grids
    let par = n_grid >= if sys.linear { PAR_THRESHOLD } else { 256 };
    let adaptive = fixed.is_none();

    let mut u: Vec<CVec> = (0..n_grid).map(|k| {
        let mut v = czero_vec();
        for s in 0..n { v[s] = g[s][k]; }
        v
    }).collect();
    let mut phi: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_grid];
    let mut trial: Vec<CustomTrial> = (0..n_grid)
        .map(|_| CustomTrial { um: czero_vec(), u2: czero_vec(), dphi: Complex64::new(0.0, 0.0), err: 0.0 })
        .collect();
    let flat = |u: &[CVec]| -> Vec<Complex64> {
        let mut out = Vec::with_capacity(n_grid * n);
        for v in u { out.extend_from_slice(&v[..n]); }
        out
    };
    // delayed states of point k, gathered from per-slot global arrays
    let gather = |k: usize, global: &[Vec<Complex64>]| -> DelSet {
        let mut out = dzero();
        for (s, v) in global.iter().enumerate() {
            out[s][..n].copy_from_slice(&v[k * n..(k + 1) * n]);
        }
        out
    };
    // derivative of every grid point for a knot at time `tk` with states `st`
    let knot_deriv = |hs: &CustomHistory, tk: f64, st: &[CVec]| -> Vec<Complex64> {
        let del: Vec<Vec<Complex64>> = sys.taus.iter().map(|&tau| hs.at(tk - tau, false)).collect();
        let f: Vec<CVec> = if par {
            st.par_iter().enumerate().map(|(k, uk)| custom_rhs(sys, uk, &gather(k, &del))).collect()
        } else {
            st.iter().enumerate().map(|(k, uk)| custom_rhs(sys, uk, &gather(k, &del))).collect()
        };
        flat(&f)
    };
    let h0 = CUSTOM_ODE_C0 / sys.lam_fast;
    let mut hist = if n_del > 0 {
        let mut hs = CustomHistory { n, n_grid, knots: std::collections::VecDeque::new() };
        hs.push(0.0, flat(&u), h0, 1.0);
        let f0 = knot_deriv(&hs, 0.0, &u);
        hs.knots[0].f = Some(f0);
        Some(hs)
    } else { None };

    let mut steps: Vec<f64> = Vec::with_capacity(fixed.map_or(256, |f| f.0.len()));
    let mut t = 0.0_f64;
    let mut h = h0;
    // the controller's proposal; `h` is what is actually taken after the
    // history cap and the clipping to a delay breakpoint
    let mut h_prop = h0;
    let mut clipped = false;
    let h_min = 1e-12 / sys.lam_fast;
    let h_max = CUSTOM_ODE_H_MAX / sys.lam_slow;
    // A delayed network is finished only once the state has been negligible
    // for a full delay: the history still feeds the system until then.
    let mut quiet_since: Option<f64> = None;
    // A replayed schedule is warped onto the current delays so every knot keeps
    // its position relative to them (see `warp_schedule`).
    let warped: Option<Vec<f64>> = fixed.map(|(st, bp)| warp_schedule(st, bp, &sys.taus));
    let fixed: Option<&[f64]> = warped.as_deref();

    loop {
        match fixed {
            Some(f) => {
                if steps.len() >= f.len() { break; }
                h = f[steps.len()];
            }
            None => {
                if steps.len() >= CUSTOM_ODE_MAX_STEPS { break; }
                h = h_prop.min(h_max);
                if let Some(hs) = &hist {
                    for &tau in &sys.taus { h = hs.max_step(t - tau, h); }
                }
                // land exactly on the time a delayed product first switches on
                clipped = false;
                for &tau in &sys.taus {
                    if t < tau && t + h > tau && tau - t > 1e-6 * h { h = tau - t; clipped = true; }
                }
            }
        }
        let t_mid = t + 0.5 * h;
        let t_end = t + h;
        let ops_h2 = custom_step_ops(&sys.a, 0.5 * h);
        let ops_h = if sys.linear { None } else { Some(custom_step_ops(&sys.a, h)) };

        // delayed states from the accepted history, per slot and stage time
        let (g0, gm, g1, has_m, has_1) = match &hist {
            Some(hs) => {
                let mut g0 = Vec::new(); let mut gm = Vec::new(); let mut g1 = Vec::new();
                let mut has_m = [false; CUSTOM_MAX_DELAYS];
                let mut has_1 = [false; CUSTOM_MAX_DELAYS];
                for (s, &tau) in sys.taus.iter().enumerate() {
                    g0.push(hs.at(t - tau, false));
                    if tau >= 0.5 * h { gm.push(hs.at(t_mid - tau, false)); has_m[s] = true; } else { gm.push(Vec::new()); }
                    if tau >= h { g1.push(hs.at(t_end - tau, true)); has_1[s] = true; } else { g1.push(Vec::new()); }
                }
                (g0, gm, g1, has_m, has_1)
            }
            None => (Vec::new(), Vec::new(), Vec::new(), [false; CUSTOM_MAX_DELAYS], [false; CUSTOM_MAX_DELAYS]),
        };
        let pick = |k: usize| -> PointDel {
            let mut pd = PointDel { d0: dzero(), dm: dzero(), d1: dzero(), has_m, has_1 };
            for s in 0..n_del {
                pd.d0[s][..n].copy_from_slice(&g0[s][k * n..(k + 1) * n]);
                if has_m[s] { pd.dm[s][..n].copy_from_slice(&gm[s][k * n..(k + 1) * n]); }
                if has_1[s] { pd.d1[s][..n].copy_from_slice(&g1[s][k * n..(k + 1) * n]); }
            }
            pd
        };
        let run = |k: usize, uk: &CVec, pk: Complex64| -> CustomTrial {
            let pd = pick(k);
            custom_trial(sys, uk, pk, h, ops_h.as_ref().unwrap_or(&ops_h2), &ops_h2, &pd, adaptive)
        };
        if par {
            trial.par_iter_mut().enumerate().for_each(|(k, out)| { *out = run(k, &u[k], phi[k]); });
        } else {
            for k in 0..n_grid { trial[k] = run(k, &u[k], phi[k]); }
        }

        let mut fac = 1.0;
        if adaptive {
            let err = if par {
                trial.par_iter().map(|x| x.err).reduce(|| 0.0_f64, f64::max)
            } else {
                trial.iter().map(|x| x.err).fold(0.0_f64, f64::max)
            };
            if err > CUSTOM_ODE_TOL && h > h_min {
                h_prop = h * (0.9 * (CUSTOM_ODE_TOL / err).powf(1.0 / 3.0)).max(0.2);
                continue;
            }
            fac = (0.9 * (CUSTOM_ODE_TOL / err.max(1e-300)).powf(1.0 / 3.0)).clamp(0.2, 2.0);
        }
        // accept
        for k in 0..n_grid {
            u[k] = trial[k].u2;
            phi[k] += trial[k].dphi;
        }
        t = t_end;
        steps.push(h);
        let remaining = if !adaptive { 0.0 } else if par {
            u.par_iter().zip(phi.par_iter()).map(|(uk, pk)| {
                let mut m = 0.0_f64;
                for i in 0..n { m = m.max(uk[i].norm()); }
                pk.re.exp() * m
            }).reduce(|| 0.0_f64, f64::max)
        } else {
            u.iter().zip(phi.iter()).map(|(uk, pk)| {
                let mut m = 0.0_f64;
                for i in 0..n { m = m.max(uk[i].norm()); }
                pk.re.exp() * m
            }).fold(0.0_f64, f64::max)
        };
        if let Some(hs) = hist.as_mut() {
            let um: Vec<CVec> = trial.iter().map(|x| x.um).collect();
            let h_ctrl = if clipped { h_prop } else { h };
            hs.push(t_mid, flat(&um), h_ctrl, remaining);
            hs.push(t_end, flat(&u), h_ctrl, remaining);
            let m = hs.knots.len();
            let fm = knot_deriv(hs, t_mid, &um);
            hs.knots[m - 2].f = Some(fm);
            let fe = knot_deriv(hs, t_end, &u);
            hs.knots[m - 1].f = Some(fe);
            hs.prune(t - tau_max);
        }
        if adaptive {
            if remaining < CUSTOM_ODE_U_END {
                if n_del == 0 { break; }
                let since = *quiet_since.get_or_insert(t);
                if t - since >= tau_max { break; }
            } else {
                quiet_since = None;
            }
            // after landing on a breakpoint resume from the unclipped proposal
            h_prop = (if clipped { h_prop } else { h }) * fac;
        }
    }
    (phi, steps)
}

/// Integrate the custom network over all grid points (rayon) and return
/// (exp(phi) per grid point, accepted step sizes).
fn custom_pgf_parallel(
    g:         &[Vec<Complex64>],
    params:    &[f64],
    rxns:      &[RxnInfo],
    prod_sp:   &[u32],
    prod_st:   &[u32],
    n_species: usize,
    n_grid:    usize,
    fixed:     Option<(&[f64], &[f64])>,
) -> (Vec<Complex64>, Vec<f64>) {
    let sys = custom_system(n_species, rxns, prod_sp, prod_st, params);
    let (phi, steps) = custom_pgf_sched(g, &sys, n_grid, fixed);
    let exp_phi: Vec<Complex64> = if n_grid >= PAR_THRESHOLD {
        phi.into_par_iter().map(|p| p.exp()).collect()
    } else {
        phi.into_iter().map(|p| p.exp()).collect()
    };
    (exp_phi, steps)
}

/// Evaluate the custom reaction-network log-PGF over all grid points in
/// parallel (rayon), returning exp(φ) as split real/imaginary Vec<f64>.
///
/// Reaction topology is encoded as parallel arrays (one entry per reaction):
/// * `rxn_kinds`     — 0=geometric burst, 1=deterministic prod, 2=first-order,
///                     3=delayed first-order (`=>`)
/// * `rxn_rate_idxs` — index into `params` for each rate constant
/// * `rxn_extra1`    — kind 0: burst-param idx; kinds 2/3: reactant species idx; else -1
/// * `rxn_extra2`    — kind 0: burst species idx; kind 3: inverse-delay param idx; else -1
/// * `prod_sp`/`prod_st`/`prod_off` — CSR products (species idx, stoich, offsets)
/// * `params`        — **linear-scale** values (including the normalised rate if any)
/// * `limits`        — grid size per species (length = n_species)
/// * `fixed_schedule` — step sizes to replay instead of adapting (finite-difference
///                     stencils pass the base point's schedule), with
///                     `fixed_breakpoints` the delay lengths it was recorded at
///
/// Returns `(re_exphi, im_exphi, shape, schedule)`: exp(phi) in row-major flat
/// order and the accepted step sizes.
#[pyfunction]
#[pyo3(signature = (n_species, limits, rxn_kinds, rxn_rate_idxs, rxn_extra1, rxn_extra2, prod_sp, prod_st, prod_off, params, fixed_schedule=None, fixed_breakpoints=None))]
fn eval_custom_network_pgf(
    py:              Python<'_>,
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
    fixed_schedule:  Option<Vec<f64>>,
    fixed_breakpoints: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>)> {
    if n_species > CUSTOM_MAX_SPECIES {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "custom networks support at most 8 species",
        ));
    }
    let rxns = build_rxn_infos(&rxn_kinds, &rxn_rate_idxs, &rxn_extra1, &rxn_extra2, &prod_off);
    if rxns.iter().filter(|r| r.kind == 3).map(|r| r.tau_slot as usize + 1).max().unwrap_or(0) > CUSTOM_MAX_DELAYS {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "custom networks support at most 4 distinct delays",
        ));
    }
    let (exp_phi, mx, steps) = py.allow_threads(|| {
        let (g, mx, n_grid) = build_mesh_nd(&limits);
        let bp: &[f64] = fixed_breakpoints.as_deref().unwrap_or(&[]);
        let (exp_phi, steps) = custom_pgf_parallel(
            &g, &params, &rxns, &prod_sp, &prod_st, n_species, n_grid,
            fixed_schedule.as_deref().map(|s| (s, bp)),
        );
        (exp_phi, mx, steps)
    });
    let re: Vec<f64> = exp_phi.iter().map(|z| z.re).collect();
    let im: Vec<f64> = exp_phi.iter().map(|z| z.im).collect();
    Ok((re, im, mx, steps))
}

// ============================================================================
// Chromatin (Markov-modulated transcription) kernel
// ============================================================================
//
// Mirrors `monod.chromatin.eval_joint_pss`.  A locus of binary sites is a
// Markov chain with generator H over N configurations; in configuration c it
// transcribes RNA species j at rate A[c, j], and the RNA follows the linear
// scheme with characteristic generator C.  The vector generating function
// G_c(u) obeys, along the characteristics u(tau) = exp(C tau) u0 (tau = time
// to go),
//
//     dG/dsigma = H^T G + diag(A u(tau)) G,      sigma = T - tau,
//
// integrated forward from the horizon T, starting from the chain's stationary
// distribution.  Each step is a Strang splitting of two exact sub-flows: the
// diagonal transcription term, whose integral over the step is analytic
// (exp(C tau_a) · delta phi_1(C delta)), and exp(H^T h), shared by the whole
// grid.  The step is chosen by comparing a full step with two half steps
// (maximum error over the grid, so the grid shares one schedule) and the
// difference feeds a Richardson extrapolation.  The full step's exp(H^T h)
// is the square of the half step's, and when the step doubles the previous
// full-step exponential is the new half-step one, so a smooth stretch costs
// one matrix product per step and no matrix exponential.
//
// Structure in the inputs is used when present (the Python side detects it):
//   - one transcribed species: the exponent of every configuration is its
//     rate times one shared number, so exponentials are taken once per
//     distinct nonzero rate (one per grid point for a switch);
//   - rates that factorise over sites, A = S W with S the 0/1 site matrix
//     (the additive rule, or one transcript per site): n exponentials and
//     one complex product per configuration instead of 2^n exponentials;
//   - independent sites (H a Kronecker sum): exp(H^T h) is a product of 2 x 2
//     factors applied bit by bit, 2 n N operations per point instead of N^2.
//
// The grid is stored in blocks of CHROMATIN_BLOCK points, each block
// state-major with separate real and imaginary planes (index c * B + i), so
// the dense apply and the diagonal exponentials run over contiguous points;
// blocks are independent given the step operators and run in parallel.

const CHROMATIN_MAX_STEPS: usize = 20_000;
const CHROMATIN_BLOCK: usize = 64;          // grid points per block
const CHROMATIN_PAR_MIN: usize = 4096;      // n_states * n_grid above which blocks run in parallel

/// How the transcription rates depend on the configuration.
enum ChromatinDiag {
    /// exp(K u) with K = A M: one exponential per configuration
    General,
    /// one transcribed species `col`; `group[c]` indexes the distinct rate of configuration c
    Shared { col: usize, group: Vec<usize>, distinct: Vec<f64> },
    /// A = S W, W (n_sites × m)
    Factored { n_sites: usize, w: Vec<f64> },
}

/// H^T, dense or as independent 2 × 2 site generators (transposed).
enum ChromatinChain { Dense(SmallMat), Kron(Vec<SmallMat>) }

/// exp(H^T h) in the matching form.
#[derive(Clone)]
enum ChainExp { Dense(SmallMat), Kron(Vec<SmallMat>) }

impl ChainExp {
    fn square(&self) -> ChainExp {
        match self {
            ChainExp::Dense(e) => ChainExp::Dense(e.mul(e)),
            ChainExp::Kron(es) => ChainExp::Kron(es.iter().map(|e| e.mul(e)).collect()),
        }
    }
}

impl ChromatinChain {
    fn expm(&self, h: f64) -> ChainExp {
        let scaled = |m: &SmallMat| {
            let mut x = m.clone();
            for v in x.a.iter_mut() { *v *= h; }
            x.expm()
        };
        match self {
            ChromatinChain::Dense(ht) => ChainExp::Dense(scaled(ht)),
            ChromatinChain::Kron(hts) => ChainExp::Kron(hts.iter().map(scaled).collect()),
        }
    }
}

struct ChromatinSys<'a> {
    n:     usize,                 // configurations
    m:     usize,                 // RNA species
    chain: ChromatinChain,
    a:     &'a [f64],             // (n, m) row-major transcription rates
    c:     SmallMat,              // (m, m) characteristic generator
    diag:  ChromatinDiag,
    /// u[j][b * B + i]: PGF argument of species j at point i of block b (zero-padded)
    ur: Vec<Vec<f64>>,
    ui: Vec<Vec<f64>>,
}

/// The diagonal sub-flow's operator for one half of a step, in the form the
/// mode needs: K = A M (n × m), the row M[col, :] (m), or W M (n_sites × m).
enum DiagOps { General(Vec<f64>), Shared(Vec<f64>), Factored(Vec<f64>) }

/// Step operators for a Strang step from time-to-go `tau` to `tau - h`:
/// d1 over [tau - h/2, tau], E = exp(H^T h), d2 over [tau - h, tau - h/2].
struct ChromatinOps { d1: DiagOps, e: ChainExp, d2: DiagOps }

/// ∫_{tau_a}^{tau_a+delta} exp(C s) ds = exp(C tau_a) · delta phi_1(C delta),
/// phi_1 read off exp([[C delta, I], [0, 0]]).
fn chromatin_char_integral(c: &SmallMat, tau_a: f64, delta: f64) -> SmallMat {
    let m = c.n;
    let mut aug = SmallMat::zeros(2 * m);
    for i in 0..m {
        for j in 0..m { aug.set(i, j, c.get(i, j) * delta); }
        aug.set(i, m + i, 1.0);
    }
    let x = aug.expm();
    let mut phi = SmallMat::zeros(m);
    for i in 0..m {
        for j in 0..m { phi.set(i, j, delta * x.get(i, m + j)); }
    }
    let mut ct = c.clone();
    for v in ct.a.iter_mut() { *v *= tau_a; }
    ct.expm().mul(&phi)
}

/// X (rows × m) · M (m × m) as a flat (rows × m) array.
fn chromatin_mat_mul(x: &[f64], rows: usize, m: usize, mm: &SmallMat) -> Vec<f64> {
    let mut k = vec![0.0; rows * m];
    for r in 0..rows {
        for j in 0..m {
            let mut s = 0.0;
            for l in 0..m { s += x[r * m + l] * mm.get(l, j); }
            k[r * m + j] = s;
        }
    }
    k
}

fn chromatin_diag_ops(sys: &ChromatinSys, mm: &SmallMat) -> DiagOps {
    let m = sys.m;
    match &sys.diag {
        ChromatinDiag::General => DiagOps::General(chromatin_mat_mul(sys.a, sys.n, m, mm)),
        ChromatinDiag::Shared { col, .. } => DiagOps::Shared((0..m).map(|l| mm.get(*col, l)).collect()),
        ChromatinDiag::Factored { n_sites, w } => DiagOps::Factored(chromatin_mat_mul(w, *n_sites, m, mm)),
    }
}

fn chromatin_ops(sys: &ChromatinSys, tau: f64, h: f64, e: ChainExp) -> ChromatinOps {
    let hh = 0.5 * h;
    let d1 = chromatin_diag_ops(sys, &chromatin_char_integral(&sys.c, tau - hh, hh));
    let d2 = chromatin_diag_ops(sys, &chromatin_char_integral(&sys.c, tau - h, hh));
    ChromatinOps { d1, e, d2 }
}

/// (exp(x))·(gr, gi) in place for x = xr + i xi.
#[inline(always)]
fn cmul_exp(xr: f64, xi: f64, gr: &mut f64, gi: &mut f64) {
    let s = xr.exp();
    let (sn, cs) = xi.sin_cos();
    let (er, ei) = (s * cs, s * sn);
    let (x, y) = (*gr, *gi);
    *gr = x * er - y * ei;
    *gi = x * ei + y * er;
}

/// (ar + i ai) · (br + i bi)
#[inline(always)]
fn cmul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) { (ar * br - ai * bi, ar * bi + ai * br) }

/// g *= exp(d · u) on one block; `sr`, `si`, `tr`, `ti` are scratch of n × B.
#[inline]
fn chromatin_diag(sys: &ChromatinSys, d: &DiagOps, b0: usize, gr: &mut [f64], gi: &mut [f64],
                  sr: &mut [f64], si: &mut [f64], tr: &mut [f64], ti: &mut [f64]) {
    let (n, m, bw) = (sys.n, sys.m, CHROMATIN_BLOCK);
    // y[r] = Σ_l coef[r, l] u_l at every point of the block, written to (sr, si) row r
    let dots = |coef: &[f64], rows: usize, sr: &mut [f64], si: &mut [f64]| {
        for r in 0..rows {
            for i in 0..bw {
                let (mut yr, mut yi) = (0.0, 0.0);
                for l in 0..m {
                    let k = coef[r * m + l];
                    yr += k * sys.ur[l][b0 + i];
                    yi += k * sys.ui[l][b0 + i];
                }
                sr[r * bw + i] = yr;
                si[r * bw + i] = yi;
            }
        }
    };
    match (d, &sys.diag) {
        (DiagOps::General(k), _) => {
            dots(k, n, sr, si);
            for c in 0..n {
                for i in 0..bw {
                    let idx = c * bw + i;
                    cmul_exp(sr[idx], si[idx], &mut gr[idx], &mut gi[idx]);
                }
            }
        }
        (DiagOps::Shared(row), ChromatinDiag::Shared { group, distinct, .. }) => {
            dots(row, 1, sr, si);                                   // z at every point in row 0
            // exp(a_d z) for every distinct nonzero rate, into rows of (tr, ti)
            for (dd, &a) in distinct.iter().enumerate() {
                if a == 0.0 { continue; }
                for i in 0..bw {
                    let (zr, zi) = (a * sr[i], a * si[i]);
                    let s = zr.exp();
                    let (sn, cs) = zi.sin_cos();
                    tr[dd * bw + i] = s * cs;
                    ti[dd * bw + i] = s * sn;
                }
            }
            for c in 0..n {
                let dd = group[c];
                if distinct[dd] == 0.0 { continue; }
                for i in 0..bw {
                    let (x, y) = cmul(gr[c * bw + i], gi[c * bw + i], tr[dd * bw + i], ti[dd * bw + i]);
                    gr[c * bw + i] = x;
                    gi[c * bw + i] = y;
                }
            }
        }
        (DiagOps::Factored(wm), ChromatinDiag::Factored { n_sites, .. }) => {
            let ns = *n_sites;
            dots(wm, ns, sr, si);                                   // y_j at every point, rows 0..ns
            for j in 0..ns {
                for i in 0..bw {
                    let idx = j * bw + i;
                    let s = sr[idx].exp();
                    let (sn, cs) = si[idx].sin_cos();
                    sr[idx] = s * cs;                               // e_j = exp(y_j) in place
                    si[idx] = s * sn;
                }
            }
            // E_c = Π_{j in c} e_j by one product per configuration (parent = c without its lowest bit)
            for i in 0..bw { tr[i] = 1.0; ti[i] = 0.0; }
            for c in 1..n {
                let j = c.trailing_zeros() as usize;
                let parent = c & (c - 1);
                for i in 0..bw {
                    let (x, y) = cmul(tr[parent * bw + i], ti[parent * bw + i], sr[j * bw + i], si[j * bw + i]);
                    tr[c * bw + i] = x;
                    ti[c * bw + i] = y;
                }
                for i in 0..bw {
                    let (x, y) = cmul(gr[c * bw + i], gi[c * bw + i], tr[c * bw + i], ti[c * bw + i]);
                    gr[c * bw + i] = x;
                    gi[c * bw + i] = y;
                }
            }
        }
        _ => unreachable!("diagonal operator does not match the system's mode"),
    }
}

/// tmp = E · g on one block (g state-major planes).
#[inline]
fn chromatin_apply(n: usize, e: &ChainExp, gr: &[f64], gi: &[f64], tr: &mut [f64], ti: &mut [f64]) {
    let bw = CHROMATIN_BLOCK;
    match e {
        ChainExp::Dense(e) => {
            for v in tr.iter_mut() { *v = 0.0; }
            for v in ti.iter_mut() { *v = 0.0; }
            for r in 0..n {
                let (trr, tir) = (&mut tr[r * bw..(r + 1) * bw], &mut ti[r * bw..(r + 1) * bw]);
                for k in 0..n {
                    let erk = e.get(r, k);
                    if erk == 0.0 { continue; }
                    let (grk, gik) = (&gr[k * bw..(k + 1) * bw], &gi[k * bw..(k + 1) * bw]);
                    for i in 0..bw {
                        trr[i] += erk * grk[i];
                        tir[i] += erk * gik[i];
                    }
                }
            }
        }
        ChainExp::Kron(es) => {
            tr.copy_from_slice(gr);
            ti.copy_from_slice(gi);
            for (j, ej) in es.iter().enumerate() {
                let bit = 1usize << j;
                let (e00, e01, e10, e11) = (ej.get(0, 0), ej.get(0, 1), ej.get(1, 0), ej.get(1, 1));
                for c in 0..n {
                    if c & bit != 0 { continue; }
                    let c1 = c | bit;
                    for plane in [&mut *tr, &mut *ti] {
                        let (lo, hi) = plane.split_at_mut(c1 * bw);
                        let r0 = &mut lo[c * bw..(c + 1) * bw];
                        let r1 = &mut hi[..bw];
                        for i in 0..bw {
                            let (x0, x1) = (r0[i], r1[i]);
                            r0[i] = e00 * x0 + e01 * x1;
                            r1[i] = e10 * x0 + e11 * x1;
                        }
                    }
                }
            }
        }
    }
}

/// One Strang step on a block, in place (`sr`, `si`, `tr`, `ti` scratch).
#[inline]
fn chromatin_step_block(sys: &ChromatinSys, ops: &ChromatinOps, b0: usize,
                        gr: &mut [f64], gi: &mut [f64],
                        sr: &mut [f64], si: &mut [f64], tr: &mut [f64], ti: &mut [f64]) {
    chromatin_diag(sys, &ops.d1, b0, gr, gi, sr, si, tr, ti);
    chromatin_apply(sys.n, &ops.e, gr, gi, tr, ti);
    gr.copy_from_slice(tr);
    gi.copy_from_slice(ti);
    chromatin_diag(sys, &ops.d2, b0, gr, gi, sr, si, tr, ti);
}

/// Trial step on block `b`: writes the Richardson-extrapolated result to
/// `out` and returns the block's maximum difference between the full step
/// and the two half steps.
fn chromatin_trial_block(
    sys: &ChromatinSys, full: &ChromatinOps, half1: &ChromatinOps, half2: &ChromatinOps,
    b: usize, gr: &[f64], gi: &[f64], outr: &mut [f64], outi: &mut [f64],
) -> f64 {
    let len = gr.len();
    let b0 = b * CHROMATIN_BLOCK;
    let mut g1r = gr.to_vec(); let mut g1i = gi.to_vec();
    let mut g2r = gr.to_vec(); let mut g2i = gi.to_vec();
    let mut sr = vec![0.0; len]; let mut si = vec![0.0; len];
    let mut tr = vec![0.0; len]; let mut ti = vec![0.0; len];
    chromatin_step_block(sys, full, b0, &mut g1r, &mut g1i, &mut sr, &mut si, &mut tr, &mut ti);
    chromatin_step_block(sys, half1, b0, &mut g2r, &mut g2i, &mut sr, &mut si, &mut tr, &mut ti);
    chromatin_step_block(sys, half2, b0, &mut g2r, &mut g2i, &mut sr, &mut si, &mut tr, &mut ti);
    let mut err = 0.0_f64;
    for i in 0..len {
        let (dr, di) = (g2r[i] - g1r[i], g2i[i] - g1i[i]);
        err = err.max((dr * dr + di * di).sqrt());
        outr[i] = g2r[i] + dr / 3.0;
        outi[i] = g2i[i] + di / 3.0;
    }
    err
}

fn chromatin_pgf(
    sys: &ChromatinSys, w: &[f64], n_blocks: usize, t_horizon: f64, h_first: f64, h_min: f64, tol: f64,
    fixed: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, bw) = (sys.n, CHROMATIN_BLOCK);
    let blen = n * bw;
    let mut gr: Vec<f64> = (0..n_blocks).flat_map(|_| (0..n).flat_map(|c| std::iter::repeat(w[c]).take(bw))).collect();
    let mut gi = vec![0.0; n_blocks * blen];
    let mut outr = vec![0.0; n_blocks * blen];
    let mut outi = vec![0.0; n_blocks * blen];
    let par = n * n_blocks * bw >= CHROMATIN_PAR_MIN;
    let adaptive = fixed.is_none();
    let mut steps: Vec<f64> = Vec::new();
    let mut tau = t_horizon;
    let mut h = h_first;
    let mut last_e: Option<(f64, ChainExp)> = None;      // the last accepted step's (h, exp(H^T h))
    loop {
        if adaptive {
            if tau <= 0.0 || steps.len() >= CHROMATIN_MAX_STEPS { break; }
            h = h.min(tau);
        } else {
            let f = fixed.unwrap();
            if steps.len() >= f.len() { break; }
            h = f[steps.len()];
        }
        let hh = 0.5 * h;
        let e_half = match &last_e {
            Some((hl, el)) if *hl == hh => el.clone(),
            _ => sys.chain.expm(hh),
        };
        let full = chromatin_ops(sys, tau, h, e_half.square());
        let half1 = chromatin_ops(sys, tau, hh, e_half.clone());
        let half2 = chromatin_ops(sys, tau - hh, hh, e_half);
        let err = if par {
            outr.par_chunks_mut(blen).zip(outi.par_chunks_mut(blen))
                .zip(gr.par_chunks(blen).zip(gi.par_chunks(blen)))
                .enumerate()
                .map(|(b, ((or_, oi), (sr, si)))| chromatin_trial_block(sys, &full, &half1, &half2, b, sr, si, or_, oi))
                .reduce(|| 0.0_f64, f64::max)
        } else {
            let mut e = 0.0_f64;
            for b in 0..n_blocks {
                let r = b * blen..(b + 1) * blen;
                e = e.max(chromatin_trial_block(sys, &full, &half1, &half2, b, &gr[r.clone()], &gi[r.clone()],
                                                &mut outr[r.clone()], &mut outi[r]));
            }
            e
        };
        let mut fac = 1.0;
        if adaptive {
            if err > tol && h > h_min {
                h *= (0.9 * (tol / err).powf(1.0 / 3.0)).max(0.2);
                continue;
            }
            fac = (0.9 * (tol / err.max(1e-300)).powf(1.0 / 3.0)).max(0.2).min(2.0);
        }
        std::mem::swap(&mut gr, &mut outr);
        std::mem::swap(&mut gi, &mut outi);
        last_e = Some((h, full.e));
        tau -= h;
        steps.push(h);
        if adaptive { h *= fac; }
    }
    (gr, gi, steps)
}

/// Vector generating function of the chromatin model on the half-spectrum
/// mesh of `limits`.  `h` (n_states²), `a` (n_states × n_species) and `c`
/// (n_species²) are row-major; `w` is the chain's stationary distribution;
/// `t_horizon`, `h_first` and `h_min` are the horizon, first step and
/// smallest step (the Python side derives them from the rates).
/// `site_weights` (n_sites × n_species, with n_states = 2^n_sites) states
/// that A = S W for the 0/1 site matrix S; `independent` (k_on_j, k_off_j
/// per site) that H is the Kronecker sum of independent sites.  Returns
/// `(re, im, mesh_shape, steps)` with G point-major: index `p * n_states + c`.
#[pyfunction]
#[pyo3(signature = (n_states, n_species, h, a, c, limits, w, t_horizon, h_first, h_min, tol, fixed_schedule=None, site_weights=None, independent=None, u_re=None, u_im=None))]
fn eval_chromatin_pgf(
    py:             Python<'_>,
    n_states:       usize,
    n_species:      usize,
    h:              Vec<f64>,
    a:              Vec<f64>,
    c:              Vec<f64>,
    limits:         Vec<usize>,
    w:              Vec<f64>,
    t_horizon:      f64,
    h_first:        f64,
    h_min:          f64,
    tol:            f64,
    fixed_schedule: Option<Vec<f64>>,
    site_weights:   Option<Vec<f64>>,
    independent:    Option<Vec<f64>>,
    u_re:           Option<Vec<f64>>,
    u_im:           Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>)> {
    let (n, m) = (n_states, n_species);
    if let (Some(ur), Some(ui)) = (&u_re, &u_im) {
        if ur.len() != ui.len() || ur.len() % m != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("eval_chromatin_pgf: u_re/u_im must be n_species × n_grid"));
        }
    } else if u_re.is_some() != u_im.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err("eval_chromatin_pgf: give both u_re and u_im or neither"));
    }
    if h.len() != n * n || a.len() != n * m || c.len() != m * m || w.len() != n || limits.len() != m {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval_chromatin_pgf: inconsistent shapes of h, a, c, w or limits",
        ));
    }
    let n_sites = n.trailing_zeros() as usize;
    let pow2 = n.is_power_of_two();
    if let Some(sw) = &site_weights {
        if !pow2 || sw.len() != n_sites * m {
            return Err(pyo3::exceptions::PyValueError::new_err("eval_chromatin_pgf: site_weights must be (log2 n_states) × n_species"));
        }
    }
    if let Some(ind) = &independent {
        if !pow2 || ind.len() != 2 * n_sites {
            return Err(pyo3::exceptions::PyValueError::new_err("eval_chromatin_pgf: independent must hold (k_on, k_off) per site"));
        }
    }
    let (re, im, mx, steps) = py.allow_threads(|| {
        // the PGF arguments: the half-spectrum mesh of `limits`, or the caller's
        // (already transformed by a sequencing-noise model)
        let (u, mx, n_grid) = match (&u_re, &u_im) {
            (Some(ur), Some(ui)) => {
                let n_grid = ur.len() / m;
                let u: Vec<Vec<Complex64>> = (0..m)
                    .map(|j| (0..n_grid).map(|p| Complex64::new(ur[j * n_grid + p], ui[j * n_grid + p])).collect())
                    .collect();
                (u, vec![n_grid], n_grid)
            }
            _ => build_mesh_nd(&limits),
        };
        let bw = CHROMATIN_BLOCK;
        let n_blocks = (n_grid + bw - 1) / bw;
        let ur: Vec<Vec<f64>> = u.iter().map(|uj| (0..n_blocks * bw).map(|p| if p < n_grid { uj[p].re } else { 0.0 }).collect()).collect();
        let ui: Vec<Vec<f64>> = u.iter().map(|uj| (0..n_blocks * bw).map(|p| if p < n_grid { uj[p].im } else { 0.0 }).collect()).collect();
        let chain = match &independent {
            Some(ind) => ChromatinChain::Kron((0..n_sites).map(|j| {
                let (kon, koff) = (ind[2 * j], ind[2 * j + 1]);
                SmallMat { n: 2, a: vec![-kon, koff, kon, -koff] }      // transposed 2 × 2 generator
            }).collect()),
            None => {
                let mut ht = SmallMat::zeros(n);
                for i in 0..n {
                    for j in 0..n { ht.set(i, j, h[j * n + i]); }
                }
                ChromatinChain::Dense(ht)
            }
        };
        let diag = if let Some(sw) = site_weights {
            ChromatinDiag::Factored { n_sites, w: sw }
        } else {
            let nonzero: Vec<usize> = (0..m).filter(|&j| (0..n).any(|cc| a[cc * m + j] != 0.0)).collect();
            if nonzero.len() == 1 {
                let col = nonzero[0];
                let mut distinct: Vec<f64> = Vec::new();
                let mut group = vec![0usize; n];
                for cc in 0..n {
                    let r = a[cc * m + col];
                    group[cc] = match distinct.iter().position(|&x| x == r) {
                        Some(k) => k,
                        None => { distinct.push(r); distinct.len() - 1 }
                    };
                }
                ChromatinDiag::Shared { col, group, distinct }
            } else {
                ChromatinDiag::General
            }
        };
        let cm = SmallMat { n: m, a: c.clone() };
        let sys = ChromatinSys { n, m, chain, a: &a, c: cm, diag, ur, ui };
        let (gr, gi, steps) = chromatin_pgf(&sys, &w, n_blocks, t_horizon, h_first, h_min, tol, fixed_schedule.as_deref());
        // back to point-major (p * n + c)
        let mut re = vec![0.0; n_grid * n];
        let mut im = vec![0.0; n_grid * n];
        for p in 0..n_grid {
            let (b, i) = (p / bw, p % bw);
            for cc in 0..n {
                re[p * n + cc] = gr[b * n * bw + cc * bw + i];
                im[p * n + cc] = gi[b * n * bw + cc * bw + i];
            }
        }
        (re, im, mx, steps)
    });
    Ok((re, im, mx, steps))
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
    validate_bio_model_2d(bio_model, "eval_kld_2d/eval_logl_2d")?;
    validate_limits(&limits, 2, "eval_kld_2d/eval_logl_2d")?;
    validate_coords_2d(&u_idx, &s_idx, limits[0], limits[1], "eval_kld_2d/eval_logl_2d")?;
    if let Some(n) = n_bio_params(bio_model) {
        if p_log.len() < n {
            return verr(format!("eval_kld_2d/eval_logl_2d: {bio_model} takes {n} parameters, got {}", p_log.len()));
        }
    }
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

// ============================================================================
// Chi-squared goodness-of-fit testing
// ============================================================================

/// Log-gamma via Lanczos approximation (7-term, good to ~15 significant figures).
fn lgamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let z = z - 1.0;
    let x: f64 = C[0] + C[1..].iter().enumerate()
        .map(|(i, &c)| c / (z + i as f64 + 1.0))
        .sum::<f64>();
    let t = z + G + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Regularized lower incomplete gamma P(a, x) via series expansion.
fn gamma_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..300 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 3e-10 { break; }
    }
    sum * (-x + a * x.ln() - lgamma(a)).exp()
}

/// Regularized upper incomplete gamma Q(a, x) via Lentz continued fraction.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=300i32 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = (an * d + b).max(1e-300).min(1e300);
        c = (b + an / c).max(1e-300).min(1e300);
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-10 { break; }
    }
    (-x + a * x.ln() - lgamma(a)).exp() * h
}

/// Chi-squared survival function P(X > stat | df degrees of freedom).
fn chi2_sf(stat: f64, df: usize) -> f64 {
    if stat <= 0.0 || df == 0 { return 1.0; }
    let a = df as f64 / 2.0;
    let x = stat / 2.0;
    if x < a + 1.0 { 1.0 - gamma_series(a, x) } else { gamma_cf(a, x) }
}

/// Group (obs, exp) count pairs into merged bins until both accumulators >= thr.
/// Leftover counts at the end are folded into the last completed bin.
fn group_bins(obs: &[f64], exp: &[f64], thr: f64) -> (Vec<f64>, Vec<f64>) {
    let mut bin_obs: Vec<f64> = Vec::new();
    let mut bin_exp: Vec<f64> = Vec::new();
    let mut run_obs = 0.0f64;
    let mut run_exp = 0.0f64;
    for (&o, &e) in obs.iter().zip(exp.iter()) {
        run_obs += o;
        run_exp += e;
        if run_obs.min(run_exp) >= thr {
            bin_obs.push(run_obs);
            bin_exp.push(run_exp);
            run_obs = 0.0;
            run_exp = 0.0;
        }
    }
    if bin_obs.is_empty() {
        bin_obs.push(run_obs);
        bin_exp.push(run_exp);
    } else {
        *bin_obs.last_mut().unwrap() += run_obs;
        *bin_exp.last_mut().unwrap() += run_exp;
    }
    (bin_obs, bin_exp)
}

/// Pearson chi-squared statistic: Σ (O−E)²/E (skips bins with E=0).
fn chi2_statistic(obs: &[f64], exp: &[f64]) -> f64 {
    obs.iter().zip(exp.iter())
        .map(|(&o, &e)| if e > 0.0 { (o - e).powi(2) / e } else { 0.0 })
        .sum()
}

/// Hellinger distance between observed and expected count arrays.
///
/// H = (1/√2) * Σ (√(e_i/n) − √(o_i/n))²
fn hellinger_distance(obs: &[f64], exp: &[f64], n_cells: f64) -> f64 {
    let sum: f64 = obs.iter().zip(exp.iter())
        .map(|(&o, &e)| {
            let d = (e / n_cells).sqrt() - (o / n_cells).sqrt();
            d * d
        })
        .sum();
    sum / 2.0_f64.sqrt()
}

/// Compute chi-squared goodness-of-fit statistics for all genes in parallel,
/// reading histogram data directly from a Rust `SearchData`.
///
/// For each gene:
///   1. Evaluates PSS at the given biological + sampling parameters.
///   2. Computes Hellinger distance between observed and expected distributions.
///   3. Groups bins until both observed and expected counts reach `grouping_thr`.
///   4. Computes Pearson chi-squared statistic and survival-function p-value.
///
/// Parameters
/// ----------
/// sd            : Rust SearchData (2-layer only)
/// bio_model     : model name
/// phys_optimum  : n_genes × n_params log10 biological parameters
/// samp_list     : n_genes optional sampling parameters; pass None for seq_model="None"
/// fixed_quad_t  : quadrature time horizon
/// quad_order    : number of Gauss-Legendre points
/// n_params_ddof : degrees of freedom consumed by parameter fitting (= number of bio params)
/// grouping_thr  : minimum bin count before finalising a bin (default 5)
/// eps           : PSS probability floor (default 1e-15)
/// seq_model     : "None", "Poisson", or "Bernoulli"
/// num_threads   : rayon thread-pool size (None = all available)
///
/// Returns (csq, pval, hellinger) — three Vec<f64> of length n_genes.
#[pyfunction]
#[pyo3(signature = (sd, bio_model, phys_optimum, samp_list=None,
                    fixed_quad_t=200.0, quad_order=40,
                    n_params_ddof=3, grouping_thr=5.0, eps=1e-15,
                    seq_model="Poisson", num_threads=None))]
#[allow(clippy::too_many_arguments)]
fn chisquare_testing_2d_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    bio_model: String,
    phys_optimum: Vec<Vec<f64>>,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    fixed_quad_t: f64,
    quad_order: usize,
    n_params_ddof: usize,
    grouping_thr: f64,
    eps: f64,
    seq_model: &str,
    num_threads: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    validate_bio_model_2d(&bio_model, "chisquare_testing_2d_sd")?;
    let seq_model = seq_model.to_owned();

    let (u_list, s_list, f_list, limits_flat, n_lim, n_genes, n_cells) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chisquare_testing_2d_sd requires at least 2 layers",
            ));
        }
        let n_lm = sd_ref.n_layers;
        let n    = sd_ref.n_genes;
        let u    = sd_ref.coords_by_layer[0].clone();
        let s    = sd_ref.coords_by_layer[1].clone();
        let f    = sd_ref.freqs.clone();
        let lim  = sd_ref.limits.clone();
        let nc   = sd_ref.n_cells;
        (u, s, f, lim, n_lm, n, nc)
    };
    let nc_f = n_cells as f64;

    let results: Vec<(f64, f64, f64)> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes).into_par_iter().map(|gi| {
                let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                let limits: Vec<usize> = (0..n_lim)
                    .map(|l| limits_flat[gi * n_lim + l])
                    .collect();
                let l1 = limits[1];

                let pss = eval_model_pss_2d_seq(
                    &bio_model, &phys_optimum[gi], &limits,
                    fixed_quad_t, quad_order, samp, &seq_model,
                );

                let u   = &u_list[gi];
                let s   = &s_list[gi];
                let f   = &f_list[gi];
                let n_obs = u.len();

                // Build (obs_counts, exp_counts) with a remainder bin appended.
                let mut obs: Vec<f64> = Vec::with_capacity(n_obs + 1);
                let mut exp: Vec<f64> = Vec::with_capacity(n_obs + 1);
                let mut exp_sum = 0.0f64;
                for k in 0..n_obs {
                    let o_k = f[k] * nc_f;
                    let e_k = pss[u[k] as usize * l1 + s[k] as usize].max(eps) * nc_f;
                    obs.push(o_k);
                    exp.push(e_k);
                    exp_sum += e_k;
                }
                obs.push(0.0);
                exp.push((nc_f - exp_sum).max(0.0));

                // Hellinger distance (before normalisation, matching Python).
                let h = hellinger_distance(&obs, &exp, nc_f);

                // Normalise expected if sums differ (numerical safety).
                let obs_sum: f64 = obs.iter().sum();
                let exp_sum_total: f64 = exp.iter().sum();
                if (obs_sum - exp_sum_total).abs() > 1e-9 * obs_sum.abs() && exp_sum_total > 0.0 {
                    let scale = obs_sum / exp_sum_total;
                    for e in &mut exp { *e *= scale; }
                }

                // Bin grouping + chi-squared.
                let (bin_obs, bin_exp) = group_bins(&obs, &exp, grouping_thr);
                let stat = chi2_statistic(&bin_obs, &bin_exp);
                // Match scipy.stats.chisquare: no degrees of freedom left -> NaN p-value
                // (which the caller treats as "not rejected"), rather than clamping to 1.
                let dof_i = bin_obs.len() as i64 - 1 - n_params_ddof as i64;
                let pval = if dof_i <= 0 { f64::NAN } else { chi2_sf(stat, dof_i as usize) };

                (stat, pval, h)
            }).collect()
        })
    });

    let csq: Vec<f64>      = results.iter().map(|r| r.0).collect();
    let pval: Vec<f64>     = results.iter().map(|r| r.1).collect();
    let hellinger: Vec<f64> = results.iter().map(|r| r.2).collect();
    Ok((csq, pval, hellinger))
}

// ============================================================================
// Log-likelihood evaluation (sparse "unique" histogram)
// ============================================================================

/// Compute log-likelihood of data under model from a sparse histogram.
///
/// pss     : flat PSS array of length l0*l1 (row-major: index = u*l1 + s)
/// l1      : number of spliced bins (second dimension)
/// u_idx   : unspliced indices of observed microstates
/// s_idx   : spliced indices of observed microstates
/// f       : fractional frequencies (counts / n_cells), same length as u_idx
/// n_cells : number of cells (scales the result)
/// eps     : minimum probability floor
///
/// Returns n_cells * sum_i f[i] * ln(pss[u[i],s[i]].max(eps)).
#[inline]
fn compute_logl_sparse(
    pss: &[f64],
    l1: usize,
    u_idx: &[u64],
    s_idx: &[u64],
    f: &[f64],
    n_cells: f64,
    eps: f64,
) -> f64 {
    let s: f64 = u_idx
        .iter()
        .zip(s_idx.iter())
        .zip(f.iter())
        .map(|((&u, &s), &fi)| {
            let pval = pss[u as usize * l1 + s as usize].max(eps);
            fi * pval.ln()
        })
        .sum();
    n_cells * s
}

/// Evaluate the log-likelihood of data under a 2-D CME model.
///
/// Parameters (Python-visible)
/// ---------------------------
/// bio_model    : model name (same as eval_model_pss_2d)
/// p_log        : log10 biological parameters
/// limits       : [l0, l1] grid dimensions
/// u_idx        : unspliced bin indices for each unique observed microstate
/// s_idx        : spliced bin indices for each unique observed microstate
/// f            : fractional frequency per microstate (counts / n_cells)
/// n_cells      : number of cells
/// fixed_quad_t : quadrature time-scale multiplier
/// quad_order   : number of Gauss-Legendre quadrature points
/// samp_log     : optional log10 sampling parameters [s0, s1]
/// eps          : minimum probability floor (default 1e-15)
/// seq_model    : sequencing model ("None", "Poisson", or "Bernoulli"); default "Poisson"
///
/// Returns the scalar log-likelihood value.
#[pyfunction]
#[pyo3(signature = (bio_model, p_log, limits, u_idx, s_idx, f, n_cells, fixed_quad_t, quad_order, samp_log=None, eps=1e-15, seq_model="Poisson"))]
#[allow(clippy::too_many_arguments)]
fn eval_logl_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    u_idx: Vec<u64>,
    s_idx: Vec<u64>,
    f: Vec<f64>,
    n_cells: f64,
    fixed_quad_t: f64,
    quad_order: usize,
    samp_log: Option<Vec<f64>>,
    eps: f64,
    seq_model: &str,
) -> PyResult<f64> {
    validate_bio_model_2d(bio_model, "eval_kld_2d/eval_logl_2d")?;
    validate_limits(&limits, 2, "eval_kld_2d/eval_logl_2d")?;
    validate_coords_2d(&u_idx, &s_idx, limits[0], limits[1], "eval_kld_2d/eval_logl_2d")?;
    if let Some(n) = n_bio_params(bio_model) {
        if p_log.len() < n {
            return verr(format!("eval_kld_2d/eval_logl_2d: {bio_model} takes {n} parameters, got {}", p_log.len()));
        }
    }
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
    Ok(compute_logl_sparse(&pss, l1, &u_idx, &s_idx, &f, n_cells, eps))
}

/// Evaluate log-likelihood for all genes in a SearchData in parallel.
///
/// Returns a Vec<f64> of length n_genes: logL[g] = n_cells * sum_i f[i] * ln(pss[u[i],s[i]].max(eps))
#[pyfunction]
#[pyo3(signature = (sd, bio_model, phys_optimum, n_cells, samp_list=None,
                    fixed_quad_t=200.0, quad_order=40,
                    eps=1e-20, seq_model="Poisson", num_threads=None))]
fn eval_logl_2d_sd(
    py: Python<'_>,
    sd: &Bound<'_, PySearchData>,
    bio_model: String,
    phys_optimum: Vec<Vec<f64>>,
    n_cells: f64,
    samp_list: Option<Vec<Option<Vec<f64>>>>,
    fixed_quad_t: f64,
    quad_order: usize,
    eps: f64,
    seq_model: &str,
    num_threads: Option<usize>,
) -> PyResult<Vec<f64>> {
    validate_bio_model_2d(&bio_model, "eval_logl_2d_sd")?;
    let seq_model = seq_model.to_owned();

    let (u_list, s_list, f_list, limits_flat, n_lim, n_genes) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eval_logl_2d_sd requires at least 2 layers",
            ));
        }
        let n_lm = sd_ref.n_layers;
        let n    = sd_ref.n_genes;
        let u    = sd_ref.coords_by_layer[0].clone();
        let s    = sd_ref.coords_by_layer[1].clone();
        let f    = sd_ref.freqs.clone();
        let lim  = sd_ref.limits.clone();
        (u, s, f, lim, n_lm, n)
    };

    let logl: Vec<f64> = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            (0..n_genes).into_par_iter().map(|gi| {
                let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                let limits: Vec<usize> = (0..n_lim)
                    .map(|l| limits_flat[gi * n_lim + l])
                    .collect();
                let l1 = limits[1];

                let pss = eval_model_pss_2d_seq(
                    &bio_model, &phys_optimum[gi], &limits,
                    fixed_quad_t, quad_order, samp, &seq_model,
                );

                compute_logl_sparse(&pss, l1, &u_list[gi], &s_list[gi], &f_list[gi], n_cells, eps)
            }).collect()
        })
    });

    Ok(logl)
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
/// KLD and forward-FD gradient (w.r.t. log10 parameters) for the ProteinBursty
/// model.  All stencil points are integrated with the base point's schedule so the
/// gradient is free of step-count discontinuities.  `coords` is (n_points, 3)
/// histogram support, `f` the matching frequencies.
#[pyfunction]
#[pyo3(signature = (p_log, limits, coords, f, fit_unspliced, protein_limit, min_fudge, max_fudge, fd_eps=1e-6, eps=1e-15, fixed_schedule=None))]
#[allow(clippy::too_many_arguments)]
fn eval_kld_grad_protein_bursty(
    py: Python<'_>,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    coords: Vec<Vec<i64>>,
    f: Vec<f64>,
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge: f64,
    max_fudge: f64,
    fd_eps: f64,
    eps: f64,
    fixed_schedule: Option<Vec<f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    use pyo3::exceptions::PyValueError;
    if p_log.len() != 5 {
        return Err(PyValueError::new_err("eval_kld_grad_protein_bursty: expected 5 log10 parameters"));
    }
    if limits.len() != 3 {
        return Err(PyValueError::new_err("eval_kld_grad_protein_bursty: expected 3 limits"));
    }
    if coords.len() != f.len() {
        return Err(PyValueError::new_err("eval_kld_grad_protein_bursty: coords and f length mismatch"));
    }
    for (i, c) in coords.iter().enumerate() {
        if c.len() != 3 || c.iter().zip(limits.iter()).any(|(&v, &l)| v < 0 || v as usize >= l) {
            return Err(PyValueError::new_err(format!(
                "eval_kld_grad_protein_bursty: coords[{i}] outside grid {limits:?}"
            )));
        }
    }
    let strides = row_major_strides(&limits);
    Ok(py.allow_threads(|| {
        eval_kld_and_grad_seq_protein_bursty(
            &p_log, &limits, &strides, &coords, &f,
            fit_unspliced, protein_limit, min_fudge, max_fudge, fd_eps, eps, fixed_schedule.as_deref(),
        )
    }))
}

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
    let kld_at = |xx: &[f64]| {
        let pss = eval_model_pss_2d_seq(bio_model, xx, limits, fixed_quad_t, quad_order, samp_log, seq_model);
        compute_kld_sparse(&pss, l1, u_idx, s_idx, f_data, eps)
    };
    let kld0 = kld_at(x);
    (kld0, fd_gradient(x, fd_eps, kld0, kld_at))
}

/// Minimize KLD(x) s.t. lb ≤ x ≤ ub using L-BFGS-B.
///
/// Delegates to `lbfgsb_rs_pure::LBFGSB` which uses a Moré-Thuente safeguarded
/// line search (same as scipy's Fortran L-BFGS-B), Cauchy point + subspace
/// minimization, and compact column-major memory storage.
///

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
    validate_bounds(&lb, &ub, n_bio_params(&bio_model), "optimize_gene_2d")?;
    validate_limits(&limits, 2, "optimize_gene_2d")?;
    validate_coords_2d(&u_idx, &s_idx, limits[0], limits[1], "optimize_gene_2d")?;
    validate_x0_list(std::slice::from_ref(&x0_list), 1, lb.len(), "optimize_gene_2d")?;
    let seq_model = seq_model.to_owned();

    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let (best_x, best_kld, _status) = py.allow_threads(|| {
        let obj = |x: &[f64]| eval_kld_and_grad_seq(
            &bio_model, x, &limits, &u_idx, &s_idx, &f,
            fixed_quad_t, quad_order, fd_eps, samp_log.as_deref(), eps, &seq_model,
        );
        best_of_restarts(&obj, &x0_list, &lb, &ub, settings)
    });
    Ok((best_x, best_kld))
}

/// Optimize all genes in parallel using L-BFGS-B + rayon.
///
/// x0_list : n_genes × num_restarts × n_params initial points
/// Returns (param_estimates, klds) each of length n_genes.
#[pyfunction]
#[pyo3(signature = (bio_model, x0_list, lb, ub, limits_list, u_idx_list, s_idx_list, f_list,
                    fixed_quad_t, quad_order, fd_eps=1e-6, maxiter=1000,
                    ftol=1e-10, gtol=1e-6, samp_list=None, eps=1e-15,
                    m_lbfgs=10, num_threads=None, seq_model="Poisson", return_status=false))]
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
    return_status: bool,
) -> PyResult<PyObject> {
    validate_bio_model_2d(&bio_model, "optimize_genes_2d")?;
    validate_bounds(&lb, &ub, n_bio_params(&bio_model), "optimize_genes_2d")?;
    let seq_model = seq_model.to_owned();
    let n_genes = x0_list.len();
    if limits_list.len() != n_genes || u_idx_list.len() != n_genes || s_idx_list.len() != n_genes || f_list.len() != n_genes {
        return verr(format!("optimize_genes_2d: x0_list, limits_list, u_idx_list, s_idx_list, f_list must all have n_genes={n_genes} entries"));
    }
    validate_x0_list(&x0_list, n_genes, lb.len(), "optimize_genes_2d")?;
    for gi in 0..n_genes {
        validate_limits(&limits_list[gi], 2, "optimize_genes_2d")?;
        validate_coords_2d(&u_idx_list[gi], &s_idx_list[gi], limits_list[gi][0], limits_list[gi][1], "optimize_genes_2d")?;
    }
    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let results = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            optimize_genes_parallel(&x0_list, &lb, &ub, settings, |gi| {
                let (bm, sm) = (bio_model.as_str(), seq_model.as_str());
                let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                let (lims, u, s_, f) = (&limits_list[gi], &u_idx_list[gi], &s_idx_list[gi], &f_list[gi]);
                move |x: &[f64]| eval_kld_and_grad_seq(
                    bm, x, lims, u, s_, f, fixed_quad_t, quad_order, fd_eps, samp, eps, sm,
                )
            })
        })
    });

    let (params, klds, status) = split_results(results);
    Ok(pack_results(py, params, klds, status, return_status))
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
                    eps=1e-15, m_lbfgs=10, num_threads=None, seq_model="Poisson", return_status=false))]
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
    return_status: bool,
) -> PyResult<PyObject> {
    validate_bio_model_2d(&bio_model, "optimize_genes_2d_sd")?;
    validate_bounds(&lb, &ub, n_bio_params(&bio_model), "optimize_genes_2d_sd")?;
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
        let lim   = sd_ref.limits.clone();
        let gll   = sd_ref.extract_gene_log_lengths();
        let gll_s = sd_ref.extract_gene_log_lengths_spliced();
        (u, s, f, lim, n_lm, n, gll, gll_s)
    }; // sd_ref dropped — GIL borrow released before allow_threads
    validate_x0_list(&x0_list, n_genes, lb.len(), "optimize_genes_2d_sd")?;
    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let results = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            optimize_genes_parallel(&x0_list, &lb, &ub, settings, |gi| {
                // Per-gene samp: add gene_log_lengths[gi] to the requested
                // channels.  The spliced channel uses gene_log_lengths_spliced
                // when present, falling back to gene_log_lengths.
                let samp_gene: Option<Vec<f64>> = base_samp.as_ref().map(|bs| {
                    let mut s = bs.clone();
                    if use_lengths_unspliced && !s.is_empty() {
                        if let Some(ref gll) = gene_log_lengths { s[0] += gll[gi]; }
                    }
                    if use_lengths_spliced && s.len() > 1 {
                        let spl = gene_log_lengths_spliced.as_ref().or(gene_log_lengths.as_ref());
                        if let Some(gll) = spl { s[1] += gll[gi]; }
                    }
                    s
                });
                let (bm, sm) = (bio_model.as_str(), seq_model.as_str());
                let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                let (u, s_, f) = (&u_idx_list[gi], &s_idx_list[gi], &f_list[gi]);
                move |x: &[f64]| eval_kld_and_grad_seq(
                    bm, x, lims, u, s_, f, fixed_quad_t, quad_order, fd_eps,
                    samp_gene.as_deref(), eps, sm,
                )
            })
        })
    });

    let (params, klds, status) = split_results(results);
    Ok(pack_results(py, params, klds, status, return_status))
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


/// Sequential ProteinBursty PSS with an optional fixed integration schedule.
fn eval_model_pss_protein_bursty_seq_sched(
    p_log:         &[f64],
    limits:        &[usize],
    fit_unspliced: bool,
    protein_limit: f64,
    min_fudge:     f64,
    max_fudge:     f64,
    fixed:         Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let (gf, mx, lims, sched) =
        protein_bursty_core_sched(p_log, limits, fit_unspliced, protein_limit, min_fudge, max_fudge, fixed);
    let pss_raw = irfftn_3d(&gf, mx[0], mx[2], lims[0], lims[1], lims[2]);
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    (pss_raw.iter().map(|x| x.abs() / abs_sum).collect(), sched)
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
    fixed:         Option<&[f64]>,
) -> (f64, Vec<f64>) {
    // Base point with the adaptive schedule (or an outer pin's); every perturbed
    // point replays that exact step sequence so the difference quotient sees
    // only the smooth dependence on the parameters, not jumps in the
    // discretisation.
    let (pss0, sched) = eval_model_pss_protein_bursty_seq_sched(
        x, limits, fit_unspliced, protein_limit, min_fudge, max_fudge, fixed,
    );
    let kld0 = compute_kld_sparse_nd(&pss0, strides, coords, f_data, eps);
    let grad = fd_gradient(x, fd_eps, kld0, |xx| {
        let (pss, _) = eval_model_pss_protein_bursty_seq_sched(
            xx, limits, fit_unspliced, protein_limit, min_fudge, max_fudge, Some(&sched),
        );
        compute_kld_sparse_nd(&pss, strides, coords, f_data, eps)
    });
    (kld0, grad)
}


/// Optimize all genes with the ProteinBursty model in parallel, reading histogram
/// data directly from a Rust `SearchData` — no Python marshal loop.
#[pyfunction]
#[pyo3(signature = (sd, x0_list, lb, ub, fit_unspliced, protein_limit, min_fudge, max_fudge,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    eps=1e-15, m_lbfgs=10, num_threads=None, return_status=false))]
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
    return_status: bool,
) -> PyResult<PyObject> {
    let (coords_list, f_list, limits_flat, n_genes, n_lim) = {
        let sd_ref = sd.borrow();
        if sd_ref.n_layers < 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "optimize_genes_protein_bursty_sd requires at least 3 layers",
            ));
        }
        sd_ref.extract_for_parallel()
    };
    validate_bounds(&lb, &ub, Some(5), "optimize_genes_protein_bursty_sd")?;
    validate_x0_list(&x0_list, n_genes, lb.len(), "optimize_genes_protein_bursty_sd")?;
    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let results = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            optimize_genes_parallel(&x0_list, &lb, &ub, settings, |gi| {
                let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                let strides = row_major_strides(lims);
                let (coords, f) = (&coords_list[gi], &f_list[gi]);
                move |x: &[f64]| eval_kld_and_grad_seq_protein_bursty(
                    x, lims, &strides, coords, f,
                    fit_unspliced, protein_limit, min_fudge, max_fudge, fd_eps, eps, None,
                )
            })
        })
    });

    let (params, klds, status) = split_results(results);
    Ok(pack_results(py, params, klds, status, return_status))
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

    let gf_log = pgf_2d(bio_model, &g0_eff, &g1_eff, &p, fixed_quad_t, quad_order);

    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();
    normalize_abs(irfftn_3d(&gf, l0, mx2, l0, l1, l2))
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
    let kld_at = |xx: &[f64]| {
        let pss = eval_model_pss_2d_amb_seq(bio_model, xx, limits, fixed_quad_t, quad_order, samp_log, amb_model);
        compute_kld_sparse_nd(&pss, strides, coords, f_data, eps)
    };
    let kld0 = kld_at(x);
    (kld0, fd_gradient(x, fd_eps, kld0, kld_at))
}


/// Optimize all genes with a 2-D bio_model + ambient noise, reading histogram data
/// directly from a Rust `SearchData` — no Python marshal loop.
///
/// p_log layout: [bio_params..., amb_params] where n_amb = 1 (Equal) or 2 (Unequal).
/// `limits` for each gene must be [l0, l1, l_amb] (3 entries).
#[pyfunction]
#[pyo3(signature = (sd, bio_model, amb_model, x0_list, lb, ub, fixed_quad_t, quad_order,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    samp_list=None, eps=1e-15, m_lbfgs=10, num_threads=None, return_status=false))]
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
    return_status: bool,
) -> PyResult<PyObject> {
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
    validate_bounds(&lb, &ub, None, "optimize_genes_2d_amb_sd")?;
    validate_x0_list(&x0_list, n_genes, lb.len(), "optimize_genes_2d_amb_sd")?;
    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let results = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            optimize_genes_parallel(&x0_list, &lb, &ub, settings, |gi| {
                let (bm, am) = (bio_model.as_str(), amb_model.as_str());
                let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                let strides = row_major_strides(lims);
                let samp = samp_list.as_ref().and_then(|sl| sl[gi].as_deref());
                let (coords, f) = (&coords_list[gi], &f_list[gi]);
                move |x: &[f64]| eval_kld_and_grad_seq_amb(
                    bm, x, lims, &strides, coords, f,
                    fixed_quad_t, quad_order, samp, am, fd_eps, eps,
                )
            })
        })
    });

    let (params, klds, status) = split_results(results);
    Ok(pack_results(py, params, klds, status, return_status))
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
    build_rxn_infos(rxn_kinds, rxn_rate_idxs, rxn_extra1, rxn_extra2, prod_off)
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
    fixed:            Option<(&[f64], &[f64])>,
) -> (Vec<f64>, Vec<f64>) {
    let (g, mx, n_grid) = build_mesh_nd(limits);
    let (exp_phi, steps) = custom_pgf_parallel(
        &g, params_linear, rxns, prod_sp, prod_st, n_species, n_grid, fixed,
    );

    let pss_raw: Vec<f64> = match n_species {
        2 => irfftn_2d(&exp_phi, limits[0], limits[1]),
        3 => irfftn_3d(&exp_phi, mx[0], mx[2], limits[0], limits[1], limits[2]),
        _ => panic!("eval_custom_network_pss_seq: only 2- and 3-species networks supported"),
    };

    (normalize_abs(pss_raw), steps)
}

/// Linear-scale parameters from the log10 optimizer state `x`; `norm_rate = 1.0`
/// is appended if present.
fn custom_network_params(x: &[f64], has_norm_rate: bool) -> Vec<f64> {
    let mut params_linear: Vec<f64> = x.iter().map(|&v| 10.0_f64.powf(v)).collect();
    if has_norm_rate {
        params_linear.push(1.0);
    }
    params_linear
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
    fd_eps:        f64,
    eps:           f64,
) -> (f64, Vec<f64>) {
    let params = custom_network_params(x, has_norm_rate);
    let (pss0, steps) = eval_custom_network_pss_seq(
        n_species, limits, rxns, prod_sp, prod_st, &params, None,
    );
    let breakpoints = custom_breakpoints(rxns, &params);
    let kld0 = compute_kld_sparse_nd(&pss0, strides, coords, f_data, eps);
    // Perturbed points replay the base point's step schedule so the finite
    // difference is not polluted by step-size jumps; the stencil is central
    // and wide because far-tail bins put a ~1e-9 noise floor on the KLD.
    let grad = fd_gradient_central(x, fd_eps.max(CUSTOM_FD_EPS), |xx| {
        let params_eps = custom_network_params(xx, has_norm_rate);
        let (pss, _) = eval_custom_network_pss_seq(
            n_species, limits, rxns, prod_sp, prod_st, &params_eps, Some((&steps, &breakpoints)),
        );
        compute_kld_sparse_nd(&pss, strides, coords, f_data, eps)
    });
    (kld0, grad)
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
                    has_norm_rate,
                    fd_eps=1e-6, maxiter=1000, ftol=1e-10, gtol=1e-6,
                    eps=1e-15, m_lbfgs=10, num_threads=None, return_status=false))]
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
    fd_eps:        f64,
    maxiter:       usize,
    ftol:          f64,
    gtol:          f64,
    eps:           f64,
    m_lbfgs:       usize,
    num_threads:   Option<usize>,
    return_status: bool,
) -> PyResult<PyObject> {
    if n_species != 2 && n_species != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "optimize_genes_custom_sd only supports 2- and 3-species custom networks",
        ));
    }

    let (coords_list, f_list, limits_flat, n_genes, n_lim) = sd.borrow().extract_for_parallel();
    validate_bounds(&lb, &ub, None, "optimize_genes_custom_sd")?;
    validate_x0_list(&x0_list, n_genes, lb.len(), "optimize_genes_custom_sd")?;

    let rxns = build_rxns(&rxn_kinds, &rxn_rate_idxs, &rxn_extra1, &rxn_extra2, &prod_off);
    let settings = LbfgsbSettings { maxiter, ftol, gtol, m_mem: m_lbfgs };
    let results = py.allow_threads(|| {
        run_with_pool(num_threads, || {
            optimize_genes_parallel(&x0_list, &lb, &ub, settings, |gi| {
                let lims = &limits_flat[gi * n_lim .. (gi + 1) * n_lim];
                let strides = row_major_strides(lims);
                let (rx, ps, pst) = (&rxns[..], &prod_sp[..], &prod_st[..]);
                let (coords, f) = (&coords_list[gi], &f_list[gi]);
                move |x: &[f64]| eval_kld_and_grad_seq_custom(
                    x, n_species, lims, &strides, rx, ps, pst, coords, f,
                    has_norm_rate, fd_eps, eps,
                )
            })
        })
    });

    let (params, klds, status) = split_results(results);
    Ok(pack_results(py, params, klds, status, return_status))
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
    if n_k == 0 {
        return verr("e_step_2d: params_per_k is empty".to_string());
    }
    if weights.len() != n_k {
        return verr(format!("e_step_2d: weights has {} entries, params_per_k has {n_k} clusters", weights.len()));
    }
    if u_obs.len() != n_genes || s_obs.len() != n_genes {
        return verr(format!("e_step_2d: u_obs ({}) and s_obs ({}) must have n_genes={n_genes} entries", u_obs.len(), s_obs.len()));
    }
    if let Some(sl) = samp_list.as_ref() {
        if sl.len() != n_genes {
            return verr(format!("e_step_2d: samp_list has {} entries, expected n_genes={n_genes}", sl.len()));
        }
    }
    let n_bio = n_bio_params(&bio_model).unwrap_or(0);
    for (k, pk) in params_per_k.iter().enumerate() {
        if pk.len() != n_genes {
            return verr(format!("e_step_2d: params_per_k[{k}] has {} genes, expected {n_genes}", pk.len()));
        }
        if let Some(bad) = pk.iter().position(|p| p.len() < n_bio) {
            return verr(format!("e_step_2d: params_per_k[{k}][{bad}] has {} entries, {bio_model} needs {n_bio}", pk[bad].len()));
        }
    }
    for g in 0..n_genes {
        validate_limits(&limits_list[g], 2, "e_step_2d")?;
        if u_obs[g].len() != n_cells || s_obs[g].len() != n_cells {
            return verr(format!("e_step_2d: gene {g} has {}/{} observations, expected n_cells={n_cells}", u_obs[g].len(), s_obs[g].len()));
        }
        validate_coords_2d(&u_obs[g], &s_obs[g], limits_list[g][0], limits_list[g][1], "e_step_2d")?;
    }

    let result = py.allow_threads(|| {
        let run = || -> (Vec<Vec<f64>>, f64, f64) {
            // log_l[c][k] accumulated across genes
            let mut log_l = vec![vec![0.0_f64; n_k]; n_cells];

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

                // Sum over genes into log_l[c][k] and add log(weight)
                let log_w = weights[k].max(f64::MIN_POSITIVE).ln();
                for c in 0..n_cells {
                    let gene_sum: f64 = gene_log_probs.iter().map(|lp| lp[c]).sum();
                    log_l[c][k] = gene_sum + log_w;
                }
            }

            // Softmax row-wise → Q; accumulate logsumexp for lower_bound
            let mut q_mat = vec![vec![0.0_f64; n_k]; n_cells];
            let mut total_lse = 0.0_f64;
            for c in 0..n_cells {
                let max_val = log_l[c].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_l[c].iter().map(|&v| (v - max_val).exp()).sum();
                let lse = max_val + sum_exp.ln();
                total_lse += lse;
                for k in 0..n_k {
                    q_mat[c][k] = (log_l[c][k] - max_val).exp() / sum_exp;
                }
            }
            let lower_bound = if n_cells > 0 { total_lse / n_cells as f64 } else { 0.0 };

            // q_func = sum(Q * log_l)
            let q_func: f64 = q_mat.iter().zip(log_l.iter()).map(|(q_row, l_row)| {
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
            let mut log_l = vec![vec![0.0_f64; n_k]; n_cells];

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
                    log_l[c][k] = gene_sum + log_w;
                }
            }

            // Softmax row-wise → Q; logsumexp → lower_bound; Q·log_l → q_func
            let mut q_mat = vec![vec![0.0_f64; n_k]; n_cells];
            let mut total_lse = 0.0_f64;
            for c in 0..n_cells {
                let max_val = log_l[c].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = log_l[c].iter().map(|&v| (v - max_val).exp()).sum();
                let lse = max_val + sum_exp.ln();
                total_lse += lse;
                for k in 0..n_k {
                    q_mat[c][k] = (log_l[c][k] - max_val).exp() / sum_exp;
                }
            }
            let lower_bound = if n_cells > 0 { total_lse / n_cells as f64 } else { 0.0 };
            let q_func: f64 = q_mat.iter().zip(log_l.iter()).map(|(q_row, l_row)| {
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
    let mut adata = sd.inner.adata.clone();

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
        adata.uns.shift_remove(*key);
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
    m.add_function(wrap_pyfunction!(eval_model_pss_protein_bursty_sched, m)?)?;
    m.add_function(wrap_pyfunction!(eval_kld_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_logl_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_logl_2d_sd, m)?)?;
    m.add_function(wrap_pyfunction!(chisquare_testing_2d_sd, m)?)?;
    m.add_function(wrap_pyfunction!(eval_kld_grad_2d, m)?)?;
    m.add_function(wrap_pyfunction!(eval_kld_grad_protein_bursty, m)?)?;
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
    m.add_function(wrap_pyfunction!(load_histograms_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(searchdata_from_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_inference_results, m)?)?;
    m.add_function(wrap_pyfunction!(eval_custom_network_pgf, m)?)?;
    m.add_function(wrap_pyfunction!(eval_chromatin_pgf, m)?)?;
    m.add_function(wrap_pyfunction!(partition_searchdata_2d, m)?)?;
    #[cfg(feature = "kmeans")]
    m.add_function(wrap_pyfunction!(initialize_q_kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_cache_clear, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(mesh_cache_set_cap, m)?)?;
    Ok(())
}
