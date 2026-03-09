/// monod_core — Rust implementation of CME model core math.
///
/// Exports a single PyO3 function:
///   eval_model_pss_2d(bio_model, p_log, limits, fixed_quad_t, quad_order) -> Vec<f64>
///
/// Handles the six 2-modality bio_models with seq_model="None":
///   Constitutive, Bursty, CIR, Extrinsic, Delay, DelayedSplicing
///
/// The return value is a flat (row-major) array of shape [limits[0], limits[1]],
/// abs-normalised to sum to 1.  The caller in Python reshapes and calls squeeze().

use num_complex::Complex64;
use pyo3::prelude::*;
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};

// ============================================================================
// Gauss-Legendre quadrature
// Matches numpy.polynomial.legendre.leggauss() to machine precision.
// ============================================================================

/// Evaluate Legendre polynomial P_n(x) and its derivative P'_n(x)
/// via the three-term recurrence.
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
    // P'_n(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);
    (p_curr, dp)
}

/// Compute Gauss-Legendre nodes and weights for integration over [-1, 1].
/// Uses symmetric Newton-Raphson initialised from Chebyshev nodes, matching
/// numpy's leggauss algorithm.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let m = (n + 1) / 2; // number of positive (or zero) roots
    let mut x = vec![0.0_f64; n];
    let mut w = vec![0.0_f64; n];
    let pi = std::f64::consts::PI;

    for i in 0..m {
        // Initial estimate (same as numpy: cos(pi * (i+0.75) / (n+0.5)))
        let mut xi = (pi * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();

        // Newton-Raphson until absolute delta < 1e-15
        for _ in 0..100 {
            let (p, dp) = legendre_poly(n, xi);
            let dx = p / dp;
            xi -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        // Store root and its reflection
        x[i] = xi;
        x[n - 1 - i] = -xi;

        // Weight: 2 / ((1 - x^2) * P'_n(x)^2)
        let (_, dp) = legendre_poly(n, xi);
        let wi = 2.0 / ((1.0 - xi * xi) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }

    (x, w)
}

// ============================================================================
// Helper: numpy.isclose with default tolerances (rtol=1e-5, atol=1e-8)
// ============================================================================

#[inline]
fn np_isclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

// ============================================================================
// Mesh building
// ============================================================================

/// Build the complex PGF argument arrays for a 2D model.
///
/// For limits = [L0, L1]:
///   mx1 = L1/2 + 1
///   u0[k] = exp(-2πi·k/L0) - 1  for k = 0..L0
///   u1[k] = exp(-2πi·k/L1) - 1  for k = 0..mx1
///
/// Returns (g0, g1) each of length n_grid = L0 * mx1, in C (row-major) order.
fn build_mesh_2d(l0: usize, l1: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mx1 = l1 / 2 + 1;
    let n_grid = l0 * mx1;
    let pi = std::f64::consts::PI;

    let u0: Vec<Complex64> = (0..l0)
        .map(|k| {
            let angle = -2.0 * pi * k as f64 / l0 as f64;
            Complex64::new(angle.cos() - 1.0, angle.sin())
        })
        .collect();

    let u1: Vec<Complex64> = (0..mx1)
        .map(|k| {
            let angle = -2.0 * pi * k as f64 / l1 as f64;
            Complex64::new(angle.cos() - 1.0, angle.sin())
        })
        .collect();

    // Meshgrid with indexing="ij": g0 varies along axis-0, g1 along axis-1
    let mut g0 = Vec::with_capacity(n_grid);
    let mut g1 = Vec::with_capacity(n_grid);
    for i in 0..l0 {
        for j in 0..mx1 {
            g0.push(u0[i]);
            g1.push(u1[j]);
        }
    }

    (g0, g1)
}

// ============================================================================
// Per-model log-PGF computations
// ============================================================================

/// Constitutive: gf[k] = g0[k]/beta + g1[k]/gamma
fn pgf_constitutive(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let beta = p[0];
    let gamma = p[1];
    g0.iter()
        .zip(g1.iter())
        .map(|(&g0k, &g1k)| g0k / beta + g1k / gamma)
        .collect()
}

/// Extrinsic: gf[k] = -alpha * ln(1 - g0[k]/beta - g1[k]/gamma)
fn pgf_extrinsic(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let alpha = p[0];
    let beta = p[1];
    let gamma = p[2];
    let one = Complex64::new(1.0, 0.0);
    g0.iter()
        .zip(g1.iter())
        .map(|(&g0k, &g1k)| {
            let inner = one - g0k / beta - g1k / gamma;
            inner.ln() * (-alpha)
        })
        .collect()
}

/// Delay: bursty with delayed degradation.
fn pgf_delay(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let b = p[0];
    let beta = p[1];
    let tauinv = p[2];
    let tau = 1.0 / tauinv;
    let exp_bt = (-beta * tau).exp();
    let one = Complex64::new(1.0, 0.0);

    g0.iter()
        .zip(g1.iter())
        .map(|(&g0k, &g1k)| {
            let u = g1k + (g0k - g1k) * exp_bt;
            let term1 = (one - u * b).ln() * (-1.0 / beta);
            // Compute log as a single log of the ratio (not log(a)-log(b)) to
            // stay on the principal branch, matching np.log((b*U-1)/(b*g0-1)).
            let ratio = (u * b - one) / (g0k * b - one);
            let term2 = ratio.ln() / (beta * (one - g1k * b));
            let term3 = g1k * b * tau / (one - g1k * b);
            term1 + term2 + term3
        })
        .collect()
}

/// DelayedSplicing: gf[k] = tau*b*g0/(1-b*g0) - (1/gamma)*ln(1-b*g1)
fn pgf_delayed_splicing(g0: &[Complex64], g1: &[Complex64], p: &[f64]) -> Vec<Complex64> {
    let b = p[0];
    let tauinv = p[1];
    let gamma = p[2];
    let tau = 1.0 / tauinv;
    let one = Complex64::new(1.0, 0.0);

    g0.iter()
        .zip(g1.iter())
        .map(|(&g0k, &g1k)| {
            let term1 = g0k * b * tau / (one - g0k * b);
            let term2 = (one - g1k * b).ln() * (-1.0 / gamma);
            term1 + term2
        })
        .collect()
}

/// Bursty: PGF via Gauss-Legendre quadrature over [0, T].
///
/// Integrand (Singh & Bokes 2012):
///   f(x) = U / (1 - U)
///   U = b * (exp(-beta*x)*c1 + exp(-gamma*x)*c2)
///
/// When beta ≈ gamma (np.isclose): c1 = g0, c2 = x*beta*g1  (x-dependent)
/// Otherwise:                       c1 = g0 - f*g1, c2 = f*g1  where f = beta/(beta-gamma)
fn pgf_bursty(
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    t: f64,
    quad_order: usize,
) -> Vec<Complex64> {
    let b = p[0];
    let beta = p[1];
    let gamma = p[2];
    let n_grid = g0.len();
    let one = Complex64::new(1.0, 0.0);

    let (xi, wi) = gauss_legendre(quad_order);
    let t_half = t / 2.0;
    let t_mid = t / 2.0;

    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    // Precompute x-independent c1, c2 when beta != gamma
    let c1: Vec<Complex64>;
    let c2: Vec<Complex64>;
    if !close {
        c2 = g1.iter().map(|&g1k| g1k * f_factor).collect();
        c1 = g0.iter().zip(c2.iter()).map(|(&g0k, &c2k)| g0k - c2k).collect();
    } else {
        c1 = vec![Complex64::new(0.0, 0.0); n_grid]; // unused
        c2 = vec![Complex64::new(0.0, 0.0); n_grid]; // unused
    }

    let mut gf = vec![Complex64::new(0.0, 0.0); n_grid];

    for (xq, wq) in xi.iter().zip(wi.iter()) {
        let x_real = t_mid + t_half * xq; // map from [-1,1] to [0,T]
        let eb = (-beta * x_real).exp();
        let eg = (-gamma * x_real).exp();

        for k in 0..n_grid {
            let u = if close {
                // c1 = g0[k], c2 = x_real * beta * g1[k]
                (g0[k] * eb + g1[k] * (x_real * beta * eg)) * b
            } else {
                (c1[k] * eb + c2[k] * eg) * b
            };
            let integrand = u / (one - u);
            gf[k] = gf[k] + integrand * (wq * t_half);
        }
    }

    gf
}

/// CIR: inverse-Gaussian-driven CME (Gorin et al. 2021), via GL quadrature.
///
/// Integrand:
///   f(x) = 1 - sqrt(1 - 4*U)   (then divide the full result by 2)
///   U = b * (exp(-beta*x)*c1 + exp(-gamma*x)*c2)
fn pgf_cir(
    g0: &[Complex64],
    g1: &[Complex64],
    p: &[f64],
    t: f64,
    quad_order: usize,
) -> Vec<Complex64> {
    let b = p[0];
    let beta = p[1];
    let gamma = p[2];
    let n_grid = g0.len();
    let one = Complex64::new(1.0, 0.0);
    let four = Complex64::new(4.0, 0.0);

    let (xi, wi) = gauss_legendre(quad_order);
    let t_half = t / 2.0;
    let t_mid = t / 2.0;

    let close = np_isclose(beta, gamma);
    let f_factor = if !close { beta / (beta - gamma) } else { 0.0 };

    let c1: Vec<Complex64>;
    let c2: Vec<Complex64>;
    if !close {
        c2 = g1.iter().map(|&g1k| g1k * f_factor).collect();
        c1 = g0.iter().zip(c2.iter()).map(|(&g0k, &c2k)| g0k - c2k).collect();
    } else {
        c1 = g0.to_vec(); // c1 = g0 in isclose case
        c2 = vec![Complex64::new(0.0, 0.0); n_grid]; // unused
    }

    let mut gf = vec![Complex64::new(0.0, 0.0); n_grid];

    for (xq, wq) in xi.iter().zip(wi.iter()) {
        let x_real = t_mid + t_half * xq;
        let eb = (-beta * x_real).exp();
        let eg = (-gamma * x_real).exp();

        for k in 0..n_grid {
            let u = if close {
                // c1 = g0[k], c2 = x_real * beta * g1[k]
                (c1[k] * eb + g1[k] * (x_real * beta * eg)) * b
            } else {
                (c1[k] * eb + c2[k] * eg) * b
            };
            let under_sqrt = one - four * u;
            let integrand = one - under_sqrt.sqrt();
            gf[k] = gf[k] + integrand * (wq * t_half);
        }
    }

    // CIR: divide by 2
    gf.iter_mut().for_each(|v| *v /= 2.0);
    gf
}

// ============================================================================
// 2-D inverse real FFT
// ============================================================================

/// Compute 2D irfftn matching scipy.fft.irfftn(a, s=[n0, n1]).
///
/// Input:  flat complex array of length n0 * (n1/2+1), row-major (C order).
/// Output: flat real array of length n0 * n1, row-major.
///
/// The result is unnormalized by n0*n1 relative to the mathematical IDFT,
/// but since eval_model_pss abs-normalises, this constant factor cancels.
///
/// Algorithm (verified to match scipy/numpy):
///   1. For each k1-column (j=0..mx1): IFFT of length n0  → complex mid array [n0, mx1].
///   2. For each i-row: build Hermitian buffer from mid[i,:] and apply IFFT of length n1
///      → take real part  → result [n0, n1].
fn irfftn_2d(input: &[Complex64], n0: usize, n1: usize) -> Vec<f64> {
    let mx1 = n1 / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();

    // --- Step 1: ifft along axis 0 (first axis) ---
    // For each column j = 0..mx1, gather the n0 values and apply IFFT of length n0.
    let fft_n0 = planner.plan_fft_inverse(n0);
    let scratch_len_n0 = fft_n0.get_inplace_scratch_len();
    let mut scratch_n0 = vec![FftComplex::new(0.0_f64, 0.0_f64); scratch_len_n0];

    // Intermediate: complex array [n0, mx1]
    let mut mid = vec![FftComplex::new(0.0_f64, 0.0_f64); n0 * mx1];
    let mut col_buf = vec![FftComplex::new(0.0_f64, 0.0_f64); n0];

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

    // --- Step 2: irfft along axis 1 (last axis) ---
    // For each row i, build the Hermitian-symmetric full complex buffer from mid[i,:]
    // and apply IFFT of length n1, then take real part.
    let fft_n1 = planner.plan_fft_inverse(n1);
    let scratch_len_n1 = fft_n1.get_inplace_scratch_len();
    let mut scratch_n1 = vec![FftComplex::new(0.0_f64, 0.0_f64); scratch_len_n1];

    let mut result = vec![0.0_f64; n0 * n1];
    let mut row_buf = vec![FftComplex::new(0.0_f64, 0.0_f64); n1];

    for i in 0..n0 {
        // Positive frequencies from mid
        for k in 0..mx1 {
            row_buf[k] = mid[i * mx1 + k];
        }
        // Conjugate (negative) frequencies: buf[n1-k] = conj(mid[i, k]) for k=1..mx1-1
        for k in 1..mx1 {
            let nk = n1 - k;
            if nk >= mx1 {
                let c = mid[i * mx1 + k];
                row_buf[nk] = FftComplex::new(c.re, -c.im);
            }
        }
        fft_n1.process_with_scratch(&mut row_buf, &mut scratch_n1);
        for j in 0..n1 {
            result[i * n1 + j] = row_buf[j].re;
        }
    }

    result
}

// ============================================================================
// eval_model_pss for 2D models (seq_model="None", amb_model="None")
// ============================================================================

/// Evaluate the steady-state PMF for a 2D CME bio-model.
///
/// Parameters
/// ----------
/// bio_model:    "Bursty", "Constitutive", "Extrinsic", "Delay",
///               "DelayedSplicing", or "CIR"
/// p_log:        log10 biological parameters (same order as CMEModel)
/// limits:       [L0, L1] grid sizes (same as CMEModel.eval_model_pss limits)
/// fixed_quad_t: fixed_quad_T attribute of CMEModel
/// quad_order:   quad_order attribute of CMEModel
///
/// Returns
/// -------
/// Flat (row-major) array of length L0*L1, abs-normalised to sum to 1.
/// The caller should reshape to (L0, L1) and call squeeze().
#[pyfunction]
fn eval_model_pss_2d(
    bio_model: &str,
    p_log: Vec<f64>,
    limits: Vec<usize>,
    fixed_quad_t: f64,
    quad_order: usize,
) -> PyResult<Vec<f64>> {
    let p: Vec<f64> = p_log.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let l0 = limits[0];
    let l1 = limits[1];

    let (g0, g1) = build_mesh_2d(l0, l1);

    // Compute log-PGF
    let gf_log: Vec<Complex64> = match bio_model {
        "Constitutive" => pgf_constitutive(&g0, &g1, &p),
        "Extrinsic" => pgf_extrinsic(&g0, &g1, &p),
        "Delay" => pgf_delay(&g0, &g1, &p),
        "DelayedSplicing" => pgf_delayed_splicing(&g0, &g1, &p),
        "Bursty" => {
            let beta = p[1];
            let gamma = p[2];
            let t = fixed_quad_t * (1.0 / beta + 1.0 / gamma + 1.0);
            pgf_bursty(&g0, &g1, &p, t, quad_order)
        }
        "CIR" => {
            let beta = p[1];
            let gamma = p[2];
            let t = fixed_quad_t * (1.0 / beta + 1.0 / gamma + 1.0);
            pgf_cir(&g0, &g1, &p, t, quad_order)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown bio_model for eval_model_pss_2d: {}",
                bio_model
            )))
        }
    };

    // exp(log-PGF)  [shape: n_grid = l0 * mx1]
    let gf: Vec<Complex64> = gf_log.iter().map(|z| z.exp()).collect();

    // irfftn(gf.reshape([l0, mx1]), s=[l0, l1])
    let pss_raw = irfftn_2d(&gf, l0, l1);

    // abs-normalise
    let abs_sum: f64 = pss_raw.iter().map(|x| x.abs()).sum();
    let pss: Vec<f64> = pss_raw.iter().map(|x| x.abs() / abs_sum).collect();

    Ok(pss)
}

// ============================================================================
// PyO3 module
// ============================================================================

#[pymodule]
fn monod_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_model_pss_2d, m)?)?;
    Ok(())
}
