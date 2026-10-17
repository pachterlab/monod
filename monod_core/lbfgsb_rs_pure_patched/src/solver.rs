#![allow(clippy::many_single_char_names)]
//! High-level LBFGS-B solver implementation.
//!
//! `LBFGSB` follows the reference algorithm (Byrd, Lu, Nocedal & Zhu 1995)
//! with the search direction computed by *dense* small-n linear algebra
//! rather than the compact-representation machinery:
//!   - H (inverse Hessian approximation) from the two-loop recursion applied
//!     to each unit vector; B = H⁻¹ by Cholesky
//!   - generalized Cauchy point along the projected steepest-descent path
//!   - quadratic model minimised over the free variables (reduced Newton
//!     step), truncated to the box
//!   - scipy-compatible s'y skip threshold, line search and ftol stopping
//!
//! This is exact for the quadratic model and costs O(n³) per iteration, which
//! is negligible for the problems this crate serves (a handful of parameters,
//! objectives that cost milliseconds to seconds).  The compact-form helpers in
//! `subalgorithms` are still maintained each iteration (`matupd`, `formt_ref`)
//! but no longer drive the direction.
//!
//! Set the `verbose` flag for a per-iteration trace on stderr.

use crate::blas;
use crate::linesearch;
use crate::subalgorithms;
use std::f64;

/// Information passed to iteration callbacks.
#[derive(Debug, Clone)]
pub struct IterationInfo {
    pub iteration: usize,
    pub f: f64,
    pub proj_grad_norm: f64,
    pub n_func_evals: usize,
    pub n_segments: usize,
    pub n_skipped: usize,
    pub n_active: usize,
}

/// Control action returned by iteration callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationControl {
    Continue,
    StopConverged,
    StopCustom,
}

/// Compact L-BFGS-B solver with full Cauchy + subspace minimization.
pub struct LBFGSB {
    /// maximum number of corrections to keep (m)
    m: usize,
    max_iter: usize,
    /// projected gradient tolerance (infinity norm of projected gradient)
    pgtol: f64,
    /// function improvement tolerance — mirrors scipy's `ftol`.
    /// Convergence when (f_prev - f) / max(|f_prev|, |f|, 1) <= ftol.
    ftol: f64,
    verbose: bool,

    // ── Classic L-BFGS history (s, y, rho) for two-loop recursion ──────────
    s: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    rho: Vec<f64>,

    // ── Compact column-major matrices (reference pipeline) ──────────────────
    /// WS: n×m column-major storage for S vectors
    ws: Vec<f64>,
    /// WY: n×m column-major storage for Y vectors
    wy: Vec<f64>,
    /// SY: m×m column-major  S'Y
    sy: Vec<f64>,
    /// SS: m×m column-major  S'S
    ss: Vec<f64>,
    /// WT: m×m Cholesky factor, updated by formt_ref after each matupd.
    ///     Required by bmv (inside cauchy) for the W*M*W' products.
    wt: Vec<f64>,

    // ── Cauchy workspace (all n-sized) ──────────────────────────────────────
    xcp: Vec<f64>,       // Cauchy point
    d_ws: Vec<f64>,      // direction scratch used inside cauchy
    t_ws: Vec<f64>,      // breakpoint times
    iorder_ws: Vec<i32>, // breakpoint ordering

    // ── Cauchy workspace (2m-sized) ─────────────────────────────────────────
    /// c = W'(xcp - x), built by cauchy; used for the bmv RHS correction
    c_ws: Vec<f64>,
    p_ws: Vec<f64>,   // scratch inside cauchy
    wbp_ws: Vec<f64>, // scratch inside cauchy
    v_ws: Vec<f64>,   // scratch inside cauchy

    // ── Algorithm state ─────────────────────────────────────────────────────
    head: usize,
    itail: usize,
    iupdat: usize,
    col: usize,
    theta: f64,
}

impl LBFGSB {
    pub fn new(m: usize) -> Self {
        let m = if m == 0 { 1 } else { m };
        LBFGSB {
            m,
            max_iter: 1000,
            pgtol: 1e-6,
            ftol: 1e-10,
            verbose: false,
            s: Vec::new(),
            y: Vec::new(),
            rho: Vec::new(),
            ws: Vec::new(),
            wy: Vec::new(),
            sy: Vec::new(),
            ss: Vec::new(),
            wt: Vec::new(),
            xcp: Vec::new(),
            d_ws: Vec::new(),
            t_ws: Vec::new(),
            iorder_ws: Vec::new(),
            c_ws: Vec::new(),
            p_ws: Vec::new(),
            wbp_ws: Vec::new(),
            v_ws: Vec::new(),
            head: 0,
            itail: 0,
            iupdat: 0,
            col: 0,
            theta: 1.0,
        }
    }

    pub fn with_max_iter(mut self, it: usize) -> Self {
        self.max_iter = it;
        self
    }

    pub fn with_pgtol(mut self, tol: f64) -> Self {
        self.pgtol = tol;
        self
    }

    /// Set function improvement tolerance (scipy-compatible ftol).
    /// Convergence when `(f_prev - f) / max(|f_prev|, |f|, 1) <= ftol`.
    /// Set to 0.0 to disable.
    pub fn with_ftol(mut self, tol: f64) -> Self {
        self.ftol = tol;
        self
    }

    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    // ── Public API ───────────────────────────────────────────────────────────

    pub fn minimize<F>(
        &mut self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        f_and_grad: &mut F,
    ) -> Result<crate::Solution, &'static str>
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let mut noop = |_info: &IterationInfo, _x: &[f64]| IterationControl::Continue;
        self.minimize_with_callback(x, lower, upper, f_and_grad, &mut noop)
    }

    pub fn minimize_with_callback<F, C>(
        &mut self,
        x: &mut [f64],
        lower: &[f64],
        upper: &[f64],
        f_and_grad: &mut F,
        callback: &mut C,
    ) -> Result<crate::Solution, &'static str>
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
        C: FnMut(&IterationInfo, &[f64]) -> IterationControl,
    {
        let n = x.len();
        if lower.len() != n || upper.len() != n {
            return Err("length mismatch between x and bounds");
        }

        // Project initial x and validate bounds.
        for i in 0..n {
            if lower[i] > upper[i] {
                return Err("lower > upper for some index");
            }
            x[i] = x[i].max(lower[i]).min(upper[i]);
        }

        // nbd flags: 0=unbounded, 1=lower only, 2=both, 3=upper only
        let mut nbd = vec![0i32; n];
        for i in 0..n {
            let lo_inf = lower[i].is_infinite() && lower[i].is_sign_negative();
            let hi_inf = upper[i].is_infinite() && upper[i].is_sign_positive();
            nbd[i] = if lo_inf && hi_inf { 0 }
                     else if !lo_inf && hi_inf { 1 }
                     else if lo_inf && !hi_inf { 3 }
                     else { 2 };
        }

        // iwhere tracks bound status of each variable (updated by cauchy each iter).
        let mut iwhere = vec![0_i32; n];
        subalgorithms::cmprlb(x, lower, upper, &nbd, &mut iwhere)
            .map_err(|_| "cmprlb failed")?;

        // Evaluation counter (shared with line-search wrapper).
        let eval_count = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let ec2 = eval_count.clone();
        let mut func = |xx: &[f64]| -> (f64, Vec<f64>) {
            ec2.set(ec2.get() + 1);
            f_and_grad(xx)
        };

        let (mut f, mut grad) = func(x);
        if grad.len() != n {
            return Err("gradient length mismatch");
        }

        // Lazy workspace allocation — keyed on n (problem dimension).
        if self.ws.len() != n * self.m {
            self.ws       = vec![0.0; n * self.m];
            self.wy       = vec![0.0; n * self.m];
            self.sy       = vec![0.0; self.m * self.m];
            self.ss       = vec![0.0; self.m * self.m];
            self.wt       = vec![0.0; self.m * self.m];
            self.xcp      = vec![0.0; n];
            self.d_ws     = vec![0.0; n];
            self.t_ws     = vec![0.0; n];
            self.iorder_ws= vec![0i32; n];
            self.c_ws     = vec![0.0; 2 * self.m];
            self.p_ws     = vec![0.0; 2 * self.m];
            self.wbp_ws   = vec![0.0; 2 * self.m];
            self.v_ws     = vec![0.0; 2 * self.m];
            self.head  = 0;
            self.itail = 0;
            self.iupdat= 0;
            self.col   = 0;
            self.theta = 1.0;
        }

        let mut pg_norm =
            subalgorithms::projgr(x, lower, upper, &nbd, &grad)
                .map_err(|_| "projgr failed")?;

        if self.verbose {
            eprintln!("iter=0 f={:.6e} ||proj_grad||_inf={:.3e}", f, pg_norm);
        }

        if pg_norm <= self.pgtol {
            return Ok(crate::Solution { x: x.to_vec(), f, iterations: 0,
                                        status: crate::Status::Converged });
        }

        let mut nskip: usize = 0;

        for iter in 1..=self.max_iter {
            // ── Direction computation ──────────────────────────────────────────
            let d = self.compute_direction(
                n, x, lower, upper, &nbd, &grad, &mut iwhere, pg_norm,
            );

            // ── Descent check; fallback to projected gradient if needed ────────
            let mut d = d;
            let d_dot_grad = blas::ddot(&d, &grad);
            if d_dot_grad >= 0.0 {
                if self.verbose {
                    eprintln!(
                        "iter {}: d not descent (d·g={:.3e}), using -proj_grad",
                        iter, d_dot_grad
                    );
                }
                for i in 0..n {
                    d[i] = if (x[i] <= lower[i] && grad[i] >= 0.0)
                              || (x[i] >= upper[i] && grad[i] <= 0.0) { 0.0 }
                           else { -grad[i] };
                }
            }

            // ── Line search ───────────────────────────────────────────────────
            let f_prev = f;
            // The line search overwrites `f` and `grad` in place on success, so
            // keep the pre-step gradient: the curvature pair needs
            // y = g_new - g_old.  (Computing it against the already-updated
            // `grad` gives y = 0, every pair is skipped, and the solver
            // degenerates to projected steepest descent.)
            let g_old = grad.clone();
            match linesearch::lnsrlb_scipy_search(x, &d, &mut f, &mut grad, lower, upper, &mut func) {
                Ok((x_new, f_new, g_new, stp_accepted, gdold)) => {
                    let s_vec = sub_vecs(&x_new, x);
                    let y_vec = sub_vecs(&g_new, &g_old);
                    let sty = blas::ddot(&s_vec, &y_vec);
                    let yy  = blas::ddot(&y_vec, &y_vec);
                    // Match scipy Fortran: skip if s'y <= eps * |g0'd| * stp
                    // i.e., sty <= eps * (-gdold * stp_accepted)
                    let ddum = (-gdold * stp_accepted).max(0.0);
                    if sty > f64::EPSILON * ddum {
                        self.push_correction(&s_vec, &y_vec);
                        self.iupdat += 1;
                        let _ = subalgorithms::matupd(
                            n, self.m,
                            &mut self.ws, &mut self.wy, &mut self.sy, &mut self.ss,
                            &s_vec, &y_vec,
                            &mut self.itail, self.iupdat,
                            &mut self.col, &mut self.head, &mut self.theta,
                            yy, sty, 1.0, blas::ddot(&s_vec, &s_vec),
                        );
                        // Update WT (Cholesky of theta*S'S + L*D^{-1}*L') so that
                        // cauchy's bmv calls are correct next iteration.
                        let _ = subalgorithms::formt_ref(
                            self.m, &mut self.wt, &self.sy, &self.ss,
                            self.col, self.theta,
                        );
                    } else {
                        nskip = nskip.saturating_add(1);
                    }
                    x.copy_from_slice(&x_new);
                    f = f_new;
                    grad = g_new;
                }
                Err(_) => {
                    // Match Fortran mainlb: if memory is non-empty, reset and restart.
                    // If memory is empty (col==0), the step itself failed — return error.
                    if self.col > 0 {
                        self.col = 0;
                        self.head = 0;
                        self.iupdat = 0;
                        self.theta = 1.0;
                        self.s.clear();
                        self.y.clear();
                        self.rho.clear();
                        for v in self.ws.iter_mut() { *v = 0.0; }
                        for v in self.wy.iter_mut() { *v = 0.0; }
                        for v in self.sy.iter_mut() { *v = 0.0; }
                        for v in self.ss.iter_mut() { *v = 0.0; }
                        for v in self.wt.iter_mut() { *v = 0.0; }
                        continue;
                    } else {
                        return Ok(crate::Solution { x: x.to_vec(), f,
                                                    iterations: iter - 1,
                                                    status: crate::Status::LineSearchFailure });
                    }
                }
            }

            // ── ftol check ────────────────────────────────────────────────────
            // Only declare convergence when f actually decreased (f < f_prev)
            // but the improvement was negligible.  Never fire when f got worse
            // (that would terminate at a bad point after a WARN-restart).
            if self.ftol > 0.0 && f < f_prev {
                let denom = f_prev.abs().max(f.abs()).max(1.0);
                if (f_prev - f) / denom <= self.ftol {
                    return Ok(crate::Solution { x: x.to_vec(), f, iterations: iter,
                                                status: crate::Status::Converged });
                }
            }

            // ── Convergence check ─────────────────────────────────────────────
            pg_norm = subalgorithms::projgr(x, lower, upper, &nbd, &grad)
                .map_err(|_| "projgr failed")?;

            if self.verbose {
                eprintln!("{:5} {:5} {:5} {:5} {:5} {:12.5e} {:12.5e}",
                          n, iter, eval_count.get(), nskip, self.col, pg_norm, f);
            }

            let info = IterationInfo {
                iteration: iter,
                f,
                proj_grad_norm: pg_norm,
                n_func_evals: eval_count.get(),
                n_segments: 0,
                n_skipped: nskip,
                n_active: 0,
            };

            match callback(&info, x) {
                IterationControl::Continue => {
                    if pg_norm <= self.pgtol {
                        return Ok(crate::Solution { x: x.to_vec(), f, iterations: iter,
                                                    status: crate::Status::Converged });
                    }
                }
                IterationControl::StopConverged => {
                    return Ok(crate::Solution { x: x.to_vec(), f, iterations: iter,
                                                status: crate::Status::Converged });
                }
                IterationControl::StopCustom => {
                    return Ok(crate::Solution { x: x.to_vec(), f, iterations: iter,
                                                status: crate::Status::MaxIter });
                }
            }
        }

        Ok(crate::Solution { x: x.to_vec(), f, iterations: self.max_iter,
                             status: crate::Status::MaxIter })
    }

    // ── Direction computation ────────────────────────────────────────────────

    /// Compute a descent direction using the full L-BFGS-B algorithm:
    ///   1. Cauchy point → identifies which variables hit bounds.
    ///   2. Free-variable set from `iwhere`.
    ///   3. Reduced gradient at xcp (with W_Z * M * c correction via bmv).
    ///   4. L-BFGS two-loop restricted to free variables → Newton correction.
    ///   5. x_bar = project(xcp + correction); d = x_bar - x.
    ///
    /// Falls back to projected L-BFGS (no Cauchy) if:
    ///   - col == 0 (no curvature yet), or
    ///   - cauchy computation fails.
    /// Search direction d = x̄ - x, where x̄ is the L-BFGS-B subspace minimiser
    /// (generalized Cauchy point, then the quadratic model minimised over the
    /// free variables and truncated to the box).
    ///
    /// The model Hessian B is formed *densely* from the stored (s, y) pairs:
    /// H = two-loop inverse, B = H⁻¹.  monod's problems have a handful of
    /// parameters, so the O(n³) dense algebra is negligible next to one
    /// objective evaluation, and it sidesteps the compact-representation
    /// bookkeeping entirely.  Falls back to the projected negative gradient
    /// when B is not positive definite or there is no memory yet.
    fn compute_direction(
        &mut self,
        n: usize,
        x: &[f64],
        lower: &[f64],
        upper: &[f64],
        _nbd: &[i32],
        grad: &[f64],
        _iwhere: &mut [i32],
        _pg_norm: f64,
    ) -> Vec<f64> {
        if self.s.is_empty() {
            return projected_neg_grad(x, lower, upper, grad, n);
        }
        let h = self.dense_inverse_hessian(n);
        let b = match dense::cholesky_inverse(&h, n) {
            Some(b) => b,
            None => return projected_neg_grad(x, lower, upper, grad, n),
        };

        // ── Generalized Cauchy point ──────────────────────────────────────
        let (xcp, free) = dense::generalized_cauchy_point(x, lower, upper, grad, &b, n);
        let nfree = free.len();
        if nfree == 0 {
            return sub_vecs(&xcp, x);
        }

        // ── Subspace minimisation over the free variables ─────────────────
        // r = g + B (xcp - x);  solve B_FF d̄ = -r_F.
        let dz = sub_vecs(&xcp, x);
        let r: Vec<f64> = (0..n)
            .map(|i| grad[i] + (0..n).map(|k| b[i * n + k] * dz[k]).sum::<f64>())
            .collect();
        let mut bff = vec![0.0; nfree * nfree];
        for (a, &i) in free.iter().enumerate() {
            for (c, &k) in free.iter().enumerate() {
                bff[a * nfree + c] = b[i * n + k];
            }
        }
        let rhs: Vec<f64> = free.iter().map(|&i| -r[i]).collect();
        let dbar = match dense::cholesky_solve(&bff, &rhs, nfree) {
            Some(d) => d,
            None => return projected_neg_grad(x, lower, upper, grad, n),
        };

        // Truncate the subspace step to the feasible box (L-BFGS-B 2.1 style).
        let mut alpha = 1.0_f64;
        for (a, &i) in free.iter().enumerate() {
            let d = dbar[a];
            if d > 0.0 && upper[i].is_finite() {
                alpha = alpha.min((upper[i] - xcp[i]) / d);
            } else if d < 0.0 && lower[i].is_finite() {
                alpha = alpha.min((lower[i] - xcp[i]) / d);
            }
        }
        let alpha = alpha.max(0.0);
        let mut xbar = xcp.clone();
        for (a, &i) in free.iter().enumerate() {
            xbar[i] = (xcp[i] + alpha * dbar[a]).max(lower[i]).min(upper[i]);
        }
        sub_vecs(&xbar, x)
    }

    /// Dense n×n inverse Hessian approximation from the two-loop recursion
    /// applied to each unit vector (row-major).
    fn dense_inverse_hessian(&self, n: usize) -> Vec<f64> {
        let col = self.s.len();
        let gamma = {
            let last = col - 1;
            let sy = blas::ddot(&self.s[last], &self.y[last]);
            let yy = blas::ddot(&self.y[last], &self.y[last]);
            if yy > 0.0 { (sy / yy).max(1e-20) } else { 1.0 }
        };
        let mut h = vec![0.0; n * n];
        let mut alpha = vec![0.0f64; col];
        for k in 0..n {
            let mut q = vec![0.0; n];
            q[k] = 1.0;
            for i in (0..col).rev() {
                alpha[i] = self.rho[i] * blas::ddot(&self.s[i], &q);
                for (qi, &yi) in q.iter_mut().zip(self.y[i].iter()) { *qi -= alpha[i] * yi; }
            }
            for v in q.iter_mut() { *v *= gamma; }
            for i in 0..col {
                let beta = self.rho[i] * blas::ddot(&self.y[i], &q);
                for (qi, &si) in q.iter_mut().zip(self.s[i].iter()) { *qi += si * (alpha[i] - beta); }
            }
            for i in 0..n { h[i * n + k] = q[i]; }
        }
        // symmetrise against round-off
        for i in 0..n { for k in (i + 1)..n {
            let v = 0.5 * (h[i * n + k] + h[k * n + i]);
            h[i * n + k] = v; h[k * n + i] = v;
        } }
        h
    }

    fn push_correction(&mut self, s_vec: &[f64], y_vec: &[f64]) {
        let sty = blas::ddot(s_vec, y_vec);
        if sty == 0.0 { return; }
        let rho_val = 1.0 / sty;
        if self.s.len() == self.m {
            self.s.remove(0);
            self.y.remove(0);
            self.rho.remove(0);
        }
        self.s.push(s_vec.to_vec());
        self.y.push(y_vec.to_vec());
        self.rho.push(rho_val);
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Projected negative gradient: -g with zeros for components at active bounds.
fn projected_neg_grad(x: &[f64], lower: &[f64], upper: &[f64], grad: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| {
        if (x[i] <= lower[i] && grad[i] >= 0.0) || (x[i] >= upper[i] && grad[i] <= 0.0) {
            0.0
        } else {
            -grad[i]
        }
    }).collect()
}

/// Element-wise a - b.
fn sub_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
}

/// Small dense linear algebra for the subspace step (row-major matrices).
mod dense {
    /// Cholesky factor L (lower, row-major) of SPD `a`; None if not SPD.
    fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j { sum -= l[i * n + k] * l[j * n + k]; }
                if i == j {
                    if !(sum > 0.0) || !sum.is_finite() { return None; }
                    l[i * n + i] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        Some(l)
    }

    fn solve_with_factor(l: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
        let mut z = vec![0.0; n];
        for i in 0..n {
            let mut sum = rhs[i];
            for k in 0..i { sum -= l[i * n + k] * z[k]; }
            z[i] = sum / l[i * n + i];
        }
        let mut xs = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = z[i];
            for k in (i + 1)..n { sum -= l[k * n + i] * xs[k]; }
            xs[i] = sum / l[i * n + i];
        }
        xs
    }

    /// Solve a x = rhs for SPD `a`.
    pub fn cholesky_solve(a: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
        let l = cholesky(a, n)?;
        Some(solve_with_factor(&l, rhs, n))
    }

    /// Inverse of SPD `a`.
    pub fn cholesky_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
        let l = cholesky(a, n)?;
        let mut inv = vec![0.0; n * n];
        for k in 0..n {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            let col = solve_with_factor(&l, &e, n);
            for i in 0..n { inv[i * n + k] = col[i]; }
        }
        for i in 0..n { for k in (i + 1)..n {
            let v = 0.5 * (inv[i * n + k] + inv[k * n + i]);
            inv[i * n + k] = v; inv[k * n + i] = v;
        } }
        Some(inv)
    }

    /// Generalized Cauchy point of the quadratic model
    /// m(x + z) = g'z + ½ z'Bz along the projected steepest-descent path
    /// (Byrd, Lu, Nocedal & Zhu 1995, Algorithm CP), with a dense B.
    /// Returns (xcp, indices of variables free at xcp).
    pub fn generalized_cauchy_point(
        x: &[f64], lower: &[f64], upper: &[f64], g: &[f64], b: &[f64], n: usize,
    ) -> (Vec<f64>, Vec<usize>) {
        // Breakpoints along d = -g.
        let mut t = vec![f64::INFINITY; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            if g[i] < 0.0 && upper[i].is_finite() {
                t[i] = (x[i] - upper[i]) / g[i];
            } else if g[i] > 0.0 && lower[i].is_finite() {
                t[i] = (x[i] - lower[i]) / g[i];
            }
            if t[i] > 0.0 && g[i] != 0.0 { d[i] = -g[i]; }
        }
        let mut order: Vec<usize> = (0..n).filter(|&i| t[i] > 0.0 && t[i].is_finite()).collect();
        order.sort_by(|&a, &c| t[a].partial_cmp(&t[c]).unwrap());

        let mut z = vec![0.0; n];      // displacement so far
        let mut fixed = vec![false; n];
        for i in 0..n { if d[i] == 0.0 { fixed[i] = true; } }
        let mut t_old = 0.0_f64;
        let mut idx = 0usize;
        loop {
            // model derivatives along the current segment
            let bz: Vec<f64> = (0..n).map(|i| (0..n).map(|k| b[i * n + k] * z[k]).sum()).collect();
            let bd: Vec<f64> = (0..n).map(|i| (0..n).map(|k| b[i * n + k] * d[k]).sum()).collect();
            let fp: f64 = (0..n).map(|i| d[i] * (g[i] + bz[i])).sum();
            let fpp: f64 = (0..n).map(|i| d[i] * bd[i]).sum();
            let seg_end = if idx < order.len() { t[order[idx]] } else { f64::INFINITY };
            let dt_seg = seg_end - t_old;
            // minimiser within the segment
            let dt_min = if fpp > 0.0 { -fp / fpp } else if fp < 0.0 { f64::INFINITY } else { 0.0 };
            if fp >= 0.0 || dt_min < dt_seg {
                let dt = dt_min.max(0.0);
                for i in 0..n { z[i] += dt * d[i]; }
                break;
            }
            if idx >= order.len() { break; } // unbounded segment, fp<0, fpp<=0: cannot happen for SPD B
            // advance to the breakpoint and fix that variable
            for i in 0..n { z[i] += dt_seg * d[i]; }
            let bidx = order[idx];
            z[bidx] = if g[bidx] < 0.0 { upper[bidx] - x[bidx] } else { lower[bidx] - x[bidx] };
            d[bidx] = 0.0;
            fixed[bidx] = true;
            t_old = seg_end;
            idx += 1;
            // all breakpoints with the same t
            while idx < order.len() && t[order[idx]] <= t_old {
                let bj = order[idx];
                z[bj] = if g[bj] < 0.0 { upper[bj] - x[bj] } else { lower[bj] - x[bj] };
                d[bj] = 0.0; fixed[bj] = true; idx += 1;
            }
        }
        let xcp: Vec<f64> = (0..n).map(|i| (x[i] + z[i]).max(lower[i]).min(upper[i])).collect();
        let free: Vec<usize> = (0..n)
            .filter(|&i| !fixed[i] && xcp[i] > lower[i] && xcp[i] < upper[i])
            .collect();
        (xcp, free)
    }
}
