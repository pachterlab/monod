#![allow(clippy::many_single_char_names)]
//! High-level LBFGS-B solver implementation using the helper modules.
//!
//! This module provides `LBFGSB`, a Rust implementation of the full reference
//! L-BFGS-B algorithm with:
//!   - Generalized Cauchy point computation (`cauchy`)
//!   - Free-variable subspace minimization via restricted two-loop L-BFGS
//!   - Compact matrix maintenance (`matupd`, `formt_ref`) for `bmv`/Cauchy
//!   - Scipy-compatible sty threshold and ftol stopping
//!
//! Per-iteration algorithm:
//!   1. Compute Cauchy point xcp (determines which variables hit their bounds)
//!   2. Identify free variables at xcp
//!   3. Build reduced gradient at xcp (includes W_Z * M * c correction via bmv)
//!   4. Apply L-BFGS two-loop restricted to free variables → Newton correction
//!   5. x_bar = xcp + correction (projected to box)
//!   6. d = x_bar - x; line search along d
//!   If any step fails, falls back to projected L-BFGS (no Cauchy).

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
            match linesearch::lnsrlb_search(x, &d, &mut f, &mut grad, lower, upper, &mut func) {
                Ok((x_new, f_new, g_new)) => {
                    let s_vec = sub_vecs(&x_new, x);
                    let y_vec = sub_vecs(&g_new, &grad);
                    let sty = blas::ddot(&s_vec, &y_vec);
                    let yy  = blas::ddot(&y_vec, &y_vec);
                    // Relative sty threshold — matches reference Fortran `ys > epsmch * yy`.
                    if sty > f64::EPSILON * yy {
                        self.push_correction(s_vec.clone(), y_vec.clone());
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
                Err(status) => {
                    return Ok(crate::Solution { x: x.to_vec(), f,
                                                iterations: iter - 1, status });
                }
            }

            // ── ftol check ────────────────────────────────────────────────────
            if self.ftol > 0.0 {
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
    fn compute_direction(
        &mut self,
        n: usize,
        x: &[f64],
        lower: &[f64],
        upper: &[f64],
        nbd: &[i32],
        grad: &[f64],
        iwhere: &mut [i32],
        pg_norm: f64,
    ) -> Vec<f64> {
        if self.col == 0 {
            // No curvature yet: projected steepest descent.
            return projected_neg_grad(x, lower, upper, grad, n);
        }

        // ── Step 1: Cauchy point ─────────────────────────────────────────────
        let mut nseg = 0i32;
        let mut info_cau = 0i32;
        let cauchy_ok = subalgorithms::cauchy(
            n, x, lower, upper, nbd, grad,
            &mut self.iorder_ws, iwhere,
            &mut self.t_ws, &mut self.d_ws,
            &mut self.xcp,
            self.m, &self.wy, &self.ws, &self.sy, &self.wt,
            self.theta, self.col, self.head,
            &mut self.p_ws, &mut self.c_ws, &mut self.wbp_ws, &mut self.v_ws,
            &mut nseg, -1, pg_norm, &mut info_cau, f64::EPSILON,
        ).is_ok() && info_cau == 0;

        if !cauchy_ok {
            // Cauchy failed: fall back to projected L-BFGS direction.
            return self.projected_lbfgs_direction(x, lower, upper, grad, n);
        }

        // ── Step 2: Free-variable set ─────────────────────────────────────────
        let (index, nfree) = subalgorithms::freev_ref(iwhere);
        if nfree == 0 {
            // All variables at their Cauchy-point bounds: step = xcp - x.
            return sub_vecs(&self.xcp[..n], x);
        }
        let ind = &index[0..nfree];

        // ── Step 3: Reduced gradient at xcp ──────────────────────────────────
        // r_free[jj] = g[k] + theta*(xcp[k] - x[k])  (gradient of quadratic model at xcp)
        // Then subtract the W_Z * M * c correction (curvature adjustment for the xcp step).
        let col = self.col;
        let col2 = 2 * col;
        let mut r_free = vec![0.0f64; nfree];
        for (jj, &k) in ind.iter().enumerate() {
            r_free[jj] = grad[k] + self.theta * (self.xcp[k] - x[k]);
        }

        // W_Z * M * c, where c = W'(xcp-x) was built by cauchy into c_ws.
        // M * c is computed via bmv using the Cholesky factor in wt.
        let mut mv = vec![0.0f64; col2];
        if subalgorithms::bmv(
            self.m, &self.sy, &self.wt, col,
            &self.c_ws[0..col2], &mut mv,
        ).is_ok() {
            // Subtract W_Z * mv from r_free.
            // W = [WY, WS] so W[k, j] = wy[cidx*n+k] (j<col) or ws[cidx*n+k] (j>=col).
            let mut pointr = self.head;
            for j in 0..col {
                let cidx = pointr % self.m;
                for (jj, &k) in ind.iter().enumerate() {
                    r_free[jj] -= self.wy[cidx * n + k] * mv[j]
                                + self.ws[cidx * n + k] * mv[col + j];
                }
                pointr += 1;
            }
        }
        // If bmv fails (wt not yet factored, degenerate curvature), we proceed
        // without the correction — still a valid (if imprecise) descent direction.

        // ── Step 4: L-BFGS two-loop restricted to free variables ─────────────
        // Computes d_free = -B_ZZ^{-1} r_free, where B_ZZ is the L-BFGS Hessian
        // restricted to the free-variable subspace Z.
        //
        // Implementation: standard two-loop with inner products summed only over
        // the free indices, using the full-space s/y history.
        let mut q: Vec<f64> = r_free.iter().map(|v| -*v).collect();
        let mut alpha = vec![0.0f64; col];

        // First pass (newest → oldest)
        for i in (0..col).rev() {
            let dot: f64 = ind.iter().enumerate()
                .map(|(jj, &k)| self.s[i][k] * q[jj])
                .sum();
            alpha[i] = self.rho[i] * dot;
            for (jj, &k) in ind.iter().enumerate() {
                q[jj] -= alpha[i] * self.y[i][k];
            }
        }

        // H₀ scaling: use full-space s/y to be consistent with theta.
        if col > 0 {
            let last = col - 1;
            let sy_val = blas::ddot(&self.s[last], &self.y[last]);
            let yy_val = blas::ddot(&self.y[last], &self.y[last]);
            let gamma  = if yy_val > 0.0 { sy_val / yy_val } else { 1.0 };
            let g = gamma.max(1e-20);
            for q_item in q.iter_mut() { *q_item *= g; }
        }

        // Second pass (oldest → newest)
        for i in 0..col {
            let dot: f64 = ind.iter().enumerate()
                .map(|(jj, &k)| self.y[i][k] * q[jj])
                .sum();
            let beta = self.rho[i] * dot;
            for (jj, &k) in ind.iter().enumerate() {
                q[jj] += self.s[i][k] * (alpha[i] - beta);
            }
        }
        // q is now the Newton step d_free = B_ZZ^{-1} (-r_free) = -B_ZZ^{-1} r_free.

        // ── Step 5: Build x_bar = xcp + correction, project to box ───────────
        let mut x_bar = self.xcp[..n].to_vec();
        for (jj, &k) in ind.iter().enumerate() {
            let xi = (x_bar[k] + q[jj]).max(lower[k]).min(upper[k]);
            x_bar[k] = xi;
        }

        // d = x_bar - x
        sub_vecs(&x_bar, x)
    }

    /// Fallback projected L-BFGS direction (no Cauchy point).
    /// Used when cauchy() fails or col==0.
    fn projected_lbfgs_direction(
        &self,
        x: &[f64],
        lower: &[f64],
        upper: &[f64],
        grad: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let col = self.s.len();
        let mut q: Vec<f64> = grad.iter().map(|v| -*v).collect();
        let mut alpha = vec![0.0f64; col];

        for i in (0..col).rev() {
            alpha[i] = self.rho[i] * blas::ddot(&self.s[i], &q);
            for (qi, &yi) in q.iter_mut().zip(self.y[i].iter()) {
                *qi -= alpha[i] * yi;
            }
        }
        if col > 0 {
            let last = col - 1;
            let sy = blas::ddot(&self.s[last], &self.y[last]);
            let yy = blas::ddot(&self.y[last], &self.y[last]);
            let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
            for v in q.iter_mut() { *v *= gamma.max(1e-20); }
        }
        for (i, ((&rho_i, &alpha_i), (y_i, s_i))) in self.rho.iter()
            .zip(alpha.iter())
            .zip(self.y.iter().zip(self.s.iter()))
            .take(col)
            .enumerate()
        {
            let _ = i;
            let beta = rho_i * blas::ddot(y_i, &q);
            for (qi, &si) in q.iter_mut().zip(s_i.iter()) {
                *qi += si * (alpha_i - beta);
            }
        }
        // Active-set projection: zero components pushing into active bounds.
        for i in 0..n {
            if (x[i] <= lower[i] && q[i] < 0.0) || (x[i] >= upper[i] && q[i] > 0.0) {
                q[i] = 0.0;
            }
        }
        q
    }

    /// Push a new correction pair (s, y) into the L-BFGS memory.
    fn push_correction(&mut self, s_vec: Vec<f64>, y_vec: Vec<f64>) {
        let sty = blas::ddot(&s_vec, &y_vec);
        if sty == 0.0 { return; }
        let rho_val = 1.0 / sty;
        if self.s.len() == self.m {
            self.s.remove(0);
            self.y.remove(0);
            self.rho.remove(0);
        }
        self.s.push(s_vec);
        self.y.push(y_vec);
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
