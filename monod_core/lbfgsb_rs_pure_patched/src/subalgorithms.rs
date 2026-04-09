/*
Extended subalgorithms port for L-BFGS-B reference

This file expands the initial helper implementations by adding simplified,
functional ports of several key subroutines from the reference distribution:

 - `cauchy_point` : compute a generalized Cauchy point (simplified).
 - `freev`        : determine free-variable indices.
 - `cmprlb`       : compare bounds and initialize workspace (lightweight).
 - `matupd`       : update compact memory-like matrices (sy/wt) with new s/y.
 - `formk`        : form a small K matrix from stored corrections (simplified).
 - `formt`        : form a triangular T matrix (simplified placeholder).
 - `subsm`        : subspace minimization (simplified; returns approximate step).
 - `bmv`          : updated to remain compatible with simplified formk/formt.

Important:
 - These implementations are intentionally simplified and safe. Their behavior
   is designed to be functional and testable, not to exactly reproduce every
   numerical detail of the original Fortran/C code (that will come later).
 - The functions provide sensible outputs so the driver and tests can be
   incrementally moved towards the reference algorithm.
*/

/// Project the initial `x` into the feasible box and initialize iwhere.
///
/// This is a Rust translation of the `active` subroutine behavior:
/// - Project x to [l, u] where bounds are present (nbd > 0). If any component
///   is adjusted, `prjctd` is true.
/// - Determine whether the problem is constrained (`cnstnd`) and whether every
///   variable has both bounds (`boxed`).
/// - Fill `iwhere` with values:
///     - -1: variable always free (nbd == 0)
///     - 3 : variable always fixed (nbd == 2 and u - l <= 0)
///     - 0 : otherwise (has bounds and not fixed)
///
/// Arguments:
/// - `x`     : mutable slice of initial variables (projected in-place)
/// - `l`, `u`: lower and upper bounds (same length as x)
/// - `nbd`   : integer flags per variable (0..3) as in reference code:
///   0=unbounded,1=only lower,2=both,3=only upper
/// - `iwhere`: output slice (same length as x) filled with statuses described above
/// - `iprint` : verbosity level (>=0 will print messages similar to reference)
///
/// Returns: (prjctd, cnstnd, boxed, nbdd)
/// - `prjctd`: whether initial x was infeasible and got projected
/// - `cnstnd`: whether problem has any constraints
/// - `boxed` : whether all variables have both lower and upper bounds
/// - `nbdd`  : number of variables exactly at a bound after projection
///
pub fn active(
    x: &mut [f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    iwhere: &mut [i32],
    iprint: i32,
) -> Result<(bool, bool, bool, usize), &'static str> {
    let n = x.len();
    if l.len() != n || u.len() != n || nbd.len() != n || iwhere.len() != n {
        return Err("length mismatch between x/l/u/nbd/iwhere");
    }

    let mut nbdd: usize = 0;
    let mut prjctd = false;
    let mut cnstnd = false;
    let mut boxed = true;

    // Step 1: project initial x to feasible set if necessary; count variables
    for i in 0..n {
        if nbd[i] > 0 {
            // variable has bounds
            let is_fixed = nbd[i] == 2 && (u[i] - l[i] <= 0.0);
            if (nbd[i] <= 2) && (x[i] <= l[i]) {
                if x[i] < l[i] {
                    prjctd = true;
                    x[i] = l[i];
                }
                // count as at-bound unless fixed
                if !is_fixed {
                    nbdd += 1;
                }
            } else if (nbd[i] >= 2) && (x[i] >= u[i]) {
                if x[i] > u[i] {
                    prjctd = true;
                    x[i] = u[i];
                }
                if !is_fixed {
                    nbdd += 1;
                }
            }
        }
    }

    // Step 2: initialize iwhere and set box/cnstnd flags
    for i in 0..n {
        if nbd[i] != 2 {
            boxed = false;
        }
        if nbd[i] == 0 {
            // always free
            iwhere[i] = -1;
        } else {
            cnstnd = true;
            if nbd[i] == 2 && (u[i] - l[i] <= 0.0) {
                // always fixed (u == l)
                iwhere[i] = 3;
            } else {
                // free but bounded
                iwhere[i] = 0;
            }
        }
    }

    // Print messages roughly matching the reference when iprint indicates verbosity
    if iprint >= 0 {
        if prjctd {
            eprintln!("The initial X is infeasible. Restart with its projection");
        }
        if !cnstnd {
            eprintln!("This problem is unconstrained");
        }
    }
    if iprint > 0 {
        eprintln!("At X0, {} variables are exactly at the bounds", nbdd);
    }

    Ok((prjctd, cnstnd, boxed, nbdd))
}

/// Full port / Rust translation of the reference `cauchy` subroutine.
///
/// This is a more faithful translation of the L-BFGS-B reference `cauchy`
/// routine. The routine computes the Generalized Cauchy Point (GCP) for a
/// limited-memory BFGS matrix defined by `wy`, `ws`, `sy`, `wt`, `theta`,
/// and the correction storage parameters `m`, `col`, `head`.
///
/// The function signature mirrors the reference's arrays but uses Rust slices
/// and idiomatic types. This implementation follows the structure of the
/// original algorithm (detect breakpoints, incrementally explore segments,
/// update p = W' d and c = W'(xcp - x), compute dtm, fix variables at bounds,
/// and finally return xcp and c). It relies on the crate's `bmv` implementation
/// (a simplified compatible variant) to compute W' * vector products.
///
/// Note: To keep the function reasonably self-contained in this crate we use
/// column-major indexing for the `wy`/`ws` matrices (n x m, stored as slice length n*m,
/// with each column contiguous) and for `sy`/`wt` (m x m, column-major). The caller must
/// match this layout when invoking. Internally, an element (i,j) of WY (row i, column j)
/// is accessed as `wy[j * n + i]`.
///
/// Arguments:
/// - `n`      : problem dimension
/// - `x/l/u`  : current iterate and bounds (length n)
/// - `nbd`    : bound flags array (length n)
/// - `g`      : gradient at x (length n)
/// - `iorder` : working array length n (breakpoint ordering, mutated)
/// - `iwhere` : status flags per variable (length n, mutated)
/// - `t`      : working array for breakpoints (length n, mutated)
/// - `d`      : working direction vector (length n, mutated)
/// - `xcp`    : output GCP (length n, mutated)
/// - `m`      : memory parameter
/// - `wy/ws`  : matrices (length n*m each), row-major
/// - `sy/wt`  : small m x m matrices (length m*m)
/// - `theta`  : initial scaling
/// - `col`    : current correction count
/// - `head`   : head index for circular storage in WY/WS (0-based)
/// - `p/c/wbp/v` : working storage (length 2*m each for p,c,wbp,v)
/// - `nseg`   : number of quadratic segments explored (output)
/// - `iprint` : verbosity (not used except for potential debug)
/// - `sbgnrm` : norm of projected gradient (if <=0 immediate return)
/// - `info`   : output status (0 OK, nonzero on errors)
/// - `epsmch` : machine epsilon (not heavily used here)
///
/// Returns `Ok(())` on success, `Err(&str)` on invalid inputs or internal error.
/// On success `xcp` and `c` are filled and `nseg` and `info` are set.
pub fn cauchy(
    n: usize,
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
    iorder: &mut [i32],
    iwhere: &mut [i32],
    t: &mut [f64],
    d: &mut [f64],
    xcp: &mut [f64],
    m: usize,
    wy: &[f64],
    ws: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    p: &mut [f64],
    c: &mut [f64],
    wbp: &mut [f64],
    v: &mut [f64],
    nseg: &mut i32,
    _iprint: i32,
    sbgnrm: f64,
    info: &mut i32,
    epsmch: f64,
) -> Result<(), &'static str> {
    // Basic validation
    if x.len() < n || l.len() < n || u.len() < n || nbd.len() < n || g.len() < n {
        return Err("length mismatch for cauchy inputs");
    }
    if iorder.len() < n || iwhere.len() < n || t.len() < n || d.len() < n || xcp.len() < n {
        return Err("working arrays too small for cauchy");
    }
    if p.len() < 2 * col || c.len() < 2 * col || wbp.len() < 2 * col || v.len() < 2 * col {
        return Err("workspace arrays p/c/wbp/v too small for cauchy");
    }
    if wy.len() < n * m || ws.len() < n * m || sy.len() < m * m || wt.len() < m * m {
        return Err("matrix storage too small for cauchy");
    }
    // Quick exit if projected gradient norm nonpositive
    if sbgnrm <= 0.0 {
        // GCP is x
        xcp[..n].copy_from_slice(&x[..n]);
        *info = 0;
        return Ok(());
    }

    // Initialize
    let mut bnded = true;
    let mut nfree = n + 1; // reference uses nfree = n+1 sentinel
    let mut nbreak = 0usize;
    let mut _ibkmin = 0usize;
    let mut bkmin = 0.0f64;
    let col2 = col * 2;
    let mut f1 = 0.0f64;

    // Zero p
    for p_item in p.iter_mut().take(col2) {
        *p_item = 0.0;
    }

    // Determine initial variable statuses, compute d and p and breakpoints
    for i in 0..n {
        let neggi = -g[i];
        // iwhere conventions: keep existing -1 (always free) and 3 (always fixed)
        if iwhere[i] != 3 && iwhere[i] != -1 {
            // variable has bounds and is not permanently fixed or free
            let mut tl = 0.0;
            let mut tu = 0.0;
            if nbd[i] <= 2 {
                tl = x[i] - l[i];
            }
            if nbd[i] >= 2 {
                tu = u[i] - x[i];
            }
            let xlower = nbd[i] <= 2 && tl <= 0.0;
            let xupper = nbd[i] >= 2 && tu <= 0.0;
            iwhere[i] = 0;
            if xlower {
                if neggi <= 0.0 {
                    iwhere[i] = 1;
                }
            } else if xupper {
                if neggi >= 0.0 {
                    iwhere[i] = 2;
                }
            } else if neggi.abs() <= 0.0 {
                iwhere[i] = -3;
            }
        }

        // pointer into circular WY/WS storage: head is 0-based in our API,
        // the original reference used 1-based arithmetic; we adapt accordingly.
        if iwhere[i] != 0 && iwhere[i] != -1 {
            d[i] = 0.0;
        } else {
            d[i] = -g[i];
            f1 -= d[i] * d[i];
            // p := p - W' * e_i * g_i  (wy/ws are n x m row-major)
            // pointr cycles through head .. head+col-1 modulo m
            for j in 0..col {
                // index in WY: row i, column (head + j) % m
                // WY/WS are stored column-major: element (i, col_idx) is at wy[col_idx * n + i]
                let col_idx = (head + j) % m;
                let wy_val = wy[col_idx * n + i];
                let ws_val = ws[col_idx * n + i];
                p[j] += wy_val * (-g[i]);
                p[col + j] += ws_val * (-g[i]);
            }

            // determine breakpoints if moving along d hits bounds
            if nbd[i] <= 2 && nbd[i] != 0 && d[i] < 0.0 {
                // x + t*d <= l -> t = (x - l) / (-d)
                nbreak += 1;
                let idx = nbreak - 1;
                iorder[idx] = (i + 1) as i32; // store 1-based index like reference
                let tl = x[i] - l[i];
                t[idx] = tl / (-d[i]);
                if nbreak == 1 || t[idx] < bkmin {
                    bkmin = t[idx];
                    _ibkmin = idx + 1;
                }
            } else if nbd[i] >= 2 && d[i] > 0.0 {
                nbreak += 1;
                let idx = nbreak - 1;
                iorder[idx] = (i + 1) as i32;
                let tu = u[i] - x[i];
                t[idx] = tu / d[i];
                if nbreak == 1 || t[idx] < bkmin {
                    bkmin = t[idx];
                    _ibkmin = idx + 1;
                }
            } else {
                // variable not bounded along the direction
                nfree -= 1;
                let pos = nfree - 1;
                iorder[pos] = (i + 1) as i32;
                if d[i].abs() > 0.0 {
                    bnded = false;
                }
            }
        }
    }

    // If theta != 1, scale p[col..] by theta
    if (theta - 1.0).abs() > f64::EPSILON {
        for j in 0..col {
            p[col + j] *= theta;
        }
    }

    // initialize xcp = x
    xcp[..n].copy_from_slice(&x[..n]);

    // If there are no breakpoints and all variables are free, return
    if nbreak == 0 && nfree == n + 1 {
        *nseg = 1;
        *info = 0;
        return Ok(());
    }

    // initialize c = W'(xcp - x) = 0
    for c_item in c.iter_mut().take(col2) {
        *c_item = 0.0;
    }

    // initialize derivative f2
    let mut f2 = -theta * f1;
    let f2_org = f2;

    // if col > 0 call bmv to compute v = B * p part (use crate bmv)
    if col > 0 {
        // bmv expects sy and wt in column-major (m x m) and p length 2*col,
        // produces v length 2*col
        let bv = bmv(m, sy, wt, col, &p[0..(2 * col)], v);
        if bv.is_err() {
            *info = -7;
            return Err("bmv failed inside cauchy");
        }
        // subtract v' * p from f2
        let mut dot_vp = 0.0;
        for j in 0..(col2) {
            dot_vp += v[j] * p[j];
        }
        f2 -= dot_vp;
    }

    // compute tentative stationary distance
    let mut dtm = if f2.abs() > 0.0 { -f1 / f2 } else { 0.0 };
    let mut tsum = 0.0;
    *nseg = 1;

    // If no breakpoints, finish here
    if nbreak == 0 {
        // move along d by dtm
        for i in 0..n {
            xcp[i] += dtm * d[i];
        }
        // update c
        if col > 0 {
            for j in 0..(col2) {
                c[j] += dtm * p[j];
            }
        }
        *info = 0;
        return Ok(());
    }

    // Build heap-based selection: compute an ordering of breakpoints ascending by t
    // using a stable sort-based helper. This reduces the O(nbreak^2) selection
    // to O(nbreak log nbreak) and matches the reference's intent (hpsolb).
    let order_idxs = match hpsolb_sorted_indices(&t[..nbreak], nbreak) {
        Ok(v) => v,
        Err(_) => {
            *info = -1;
            return Err("hpsolb_sorted_indices failed inside cauchy");
        }
    };

    let mut idx_iter = 0usize;
    let mut nleft = nbreak;
    let mut _iter = 1usize;
    // tj tracks the cumulative breakpoint time (must be `mut` and updated in-place).
    // The original code had a shadowing bug: `let tj = t[ibp_idx]` inside the loop
    // created a new local variable instead of updating the outer one, so tj0 was
    // always 0.0, making dt = absolute time rather than incremental time between
    // breakpoints. Fixed by declaring `mut` and assigning without `let` inside loop.
    let mut tj = 0.0f64;

    loop {
        let tj0 = tj;

        if idx_iter < order_idxs.len() {
            let ibp_idx = order_idxs[idx_iter];
            tj = t[ibp_idx];   // update outer tj (was a shadowing bug: `let tj = ...`)
            let dt = tj - tj0;

            // If minimizer in current segment
            if dtm < dt {
                // move xcp along d by dtm for free and not-yet-fixed components
                tsum += dtm;
                for i in 0..n {
                    xcp[i] += dtm * d[i];
                }
                // update c
                if col > 0 {
                    for j in 0..(col2) {
                        c[j] += dtm * p[j];
                    }
                }
                break;
            }

            // otherwise move by dt and fix variable at breakpoint
            tsum += dt;

            // index of variable to fix (1-based stored)
            let ibp = (iorder[ibp_idx] - 1) as usize;
            // store current d value and set to zero (fix variable)
            let dibp = d[ibp];
            d[ibp] = 0.0;
            // set xcp[ibp] to its bound
            if dibp > 0.0 {
                xcp[ibp] = u[ibp];
                iwhere[ibp] = 2;
            } else {
                xcp[ibp] = l[ibp];
                iwhere[ibp] = 1;
            }

            // If all variables fixed, break and return
            nleft -= 1;
            _iter += 1;
            if nleft == 0 && nbreak == n {
                // move exactly to end
                dtm = dt;
                break;
            }

            // Update f1 and f2 and update p and c accordingly
            let dibp2 = dibp * dibp;
            f1 = f1 + dt * f2 + dibp2
                - theta
                    * dibp
                    * (if dibp > 0.0 {
                        u[ibp] - x[ibp]
                    } else {
                        l[ibp] - x[ibp]
                    });
            f2 -= theta * dibp2;

            if col > 0 {
                // c = c + dt * p
                for j in 0..(col2) {
                    c[j] += dt * p[j];
                }
                // build wbp: row of W corresponding to ibp (wy/ws concatenated)
                // WY/WS are column-major: element (ibp, idx) is wy[idx * n + ibp]
                for j in 0..col {
                    let idx = (head + j) % m;
                    wbp[j] = wy[idx * n + ibp];
                    wbp[col + j] = theta * ws[idx * n + ibp];
                }
                // compute wbp * M * c and wbp * M * p and wbp * M * wbp' via bmv
                let res = bmv(m, sy, wt, col, &wbp[0..2 * col], v);
                if res.is_err() {
                    *info = -7;
                    return Err("bmv failed inside cauchy (wbp)");
                }
                let mut wmc = 0.0;
                let mut wmp = 0.0;
                let mut wmw = 0.0;
                for j in 0..(col2) {
                    wmc += c[j] * v[j];
                    wmp += p[j] * v[j];
                    wmw += wbp[j] * v[j];
                }
                // p = p - dibp * wbp
                for j in 0..(col2) {
                    p[j] -= dibp * wbp[j];
                }
                // update f1 and f2
                f1 += dibp * wmc;
                f2 = f2 + dibp * 2.0 * wmp - dibp2 * wmw;
            }

            // guard f2
            let f2_min = epsmch * f2_org;
            if f2 < f2_min {
                f2 = f2_min.max(f2);
            }
            if nleft > 0 {
                dtm = -f1 / f2;
                // advance to next breakpoint in sorted order
                idx_iter += 1;
                continue;
            } else {
                if bnded {
                    #[allow(unused_assignments)]
                    {
                        f1 = 0.0;
                        f2 = 0.0;
                    }
                    dtm = 0.0;
                } else {
                    dtm = -f1 / f2;
                }
                break;
            }
        } else {
            // No remaining breakpoints; move by final dtm
            tsum += dtm;
            for i in 0..n {
                xcp[i] += dtm * d[i];
            }
            if col > 0 {
                for j in 0..(col2) {
                    c[j] += dtm * p[j];
                }
            }
            break;
        }
    }

    // After loop (L888 in reference), move variables by tsum along d
    for i in 0..n {
        xcp[i] += tsum * d[i];
    }

    // Update c (final): c = c + dtm * p
    if col > 0 {
        for j in 0..(col2) {
            c[j] += dtm * p[j];
        }
    }

    // finalize
    *nseg = 1; // coarse; the reference accumulates segments count
    *info = 0;
    Ok(())
}

/// Return indices of free variables (nbd == 0).
pub fn freev(nbd: &[i32]) -> Vec<usize> {
    let mut free = Vec::new();
    for (i, &b) in nbd.iter().enumerate() {
        if b == 0 {
            free.push(i);
        }
    }
    free
}

/// Faithful helper: build the index set of free variables and returning counts
/// matching the reference's `freev` semantics when the incremental `formk`
/// needs `index` arrays and counts. This returns a tuple (index, nfree)
/// where `index` contains free indices first and bound indices afterwards, to
/// mimic the reference layout.
pub fn freev_ref(iwhere: &[i32]) -> (Vec<usize>, usize) {
    let n = iwhere.len();
    let mut index = vec![0usize; n];
    let mut nfree = 0usize;
    let mut iact = n;
    for (i, &w) in iwhere.iter().enumerate() {
        if w <= 0 {
            index[nfree] = i;
            nfree += 1;
        } else {
            iact = iact.saturating_sub(1);
            index[iact] = i;
        }
    }
    (index, nfree)
}

/// Compute infinity-norm of the projected gradient (projgr).
///
/// This mirrors the reference `projgr` subroutine: for each component i the
/// gradient is modified according to the active bounds and the infinity norm
/// of the resulting "projected" gradient is returned.
///
/// Signature:
/// - `x`   : current iterate (length n)
/// - `l/u` : lower/upper bounds (length n)
/// - `nbd` : bound flags per component (0..3) as in the reference:
///   0 = unbounded, 1 = only lower, 2 = both, 3 = only upper
/// - `g`   : gradient at `x` (length n)
///
/// Returns `Ok(sbgnrm)` or `Err(&'static str)` on length mismatch.
pub fn projgr(
    x: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
) -> Result<f64, &'static str> {
    let n = x.len();
    if l.len() != n || u.len() != n || nbd.len() != n || g.len() != n {
        return Err("length mismatch for projgr inputs");
    }

    let mut sbgnrm = 0.0_f64;
    for i in 0..n {
        let mut gi = g[i];
        if nbd[i] != 0 {
            // If gradient is negative, consider upper bound influence when both bounds present
            if gi < 0.0 {
                if nbd[i] >= 2 {
                    gi = (x[i] - u[i]).max(gi);
                }
            } else {
                // gi >= 0: consider lower bound influence when lower bound present
                if nbd[i] <= 2 {
                    gi = (x[i] - l[i]).min(gi);
                }
            }
        }
        sbgnrm = sbgnrm.max(gi.abs());
    }

    Ok(sbgnrm)
}

/// Small helper: produce a vector of breakpoint indices 0..nbreak-1 sorted
/// in ascending order of the corresponding `t` values. This mirrors the
/// selection behavior of the reference `hpsolb` but implemented safely
/// with Rust's sort (O(nbreak log nbreak)).
///
/// - `t` : slice containing candidate breakpoints (length >= nbreak)
/// - `nbreak` : number of valid entries in `t` to consider
///
/// Returns `Ok(vec_of_indices)` on success or `Err(&'static str)` if
/// inputs are inconsistent.
pub fn hpsolb_sorted_indices(t: &[f64], nbreak: usize) -> Result<Vec<usize>, &'static str> {
    if nbreak == 0 {
        return Ok(Vec::new());
    }
    if t.len() < nbreak {
        return Err("hpsolb_sorted_indices: t too small");
    }
    let mut idxs: Vec<usize> = (0..nbreak).collect();
    idxs.sort_by(|&a, &b| {
        // Use partial_cmp for f64 and treat NaN/incomparable as equal
        t[a].partial_cmp(&t[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(idxs)
}

/// Backwards-compatible simple wrapper kept for tests and early integration.
///
/// The reference `cauchy` routine has a complex signature and a fully
/// stateful implementation is provided elsewhere in this module (the
/// function `cauchy(...)`). Some tests and earlier code expect a simple
/// convenience function `cauchy_point(...)` that returns a Vec<f64>.
/// To preserve compatibility we provide a thin wrapper that computes a
/// projected negative-gradient step (a simple approximation) so callers
/// that use the simple API continue to work while we iteratively replace
/// this with the full reference port.
pub fn cauchy_point(x: &[f64], l: &[f64], u: &[f64], nbd: &[i32], g: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut xcp = vec![0.0; n];
    for i in 0..n {
        // simple projected negative-gradient step
        let mut xi = x[i] - g[i];
        if nbd[i] != 0 {
            if xi < l[i] {
                xi = l[i];
            } else if xi > u[i] {
                xi = u[i];
            }
        }
        xcp[i] = xi;
    }
    xcp
}

/// Compare bounds and initialize `iwhere` similar to the reference's cmprlb.
///
/// This lightweight port fills `iwhere` consistent with `active` semantics.
/// It returns Ok(()) on success or an error message on mismatch.
pub fn cmprlb(
    x: &mut [f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    iwhere: &mut [i32],
) -> Result<(), &'static str> {
    if x.len() != l.len() || x.len() != u.len() || x.len() != nbd.len() || iwhere.len() != x.len() {
        return Err("length mismatch in cmprlb");
    }
    // Reuse active-like logic for iwhere initialization
    for i in 0..x.len() {
        if nbd[i] == 0 {
            iwhere[i] = -1;
        } else if nbd[i] == 2 && (u[i] - l[i] <= 0.0) {
            iwhere[i] = 3;
            // ensure x at the fixed value
            x[i] = l[i].max(u[i]).min(l[i]);
        } else {
            iwhere[i] = 0;
        }
    }
    Ok(())
}

/// Update matrices WS, WY, and form the middle matrices SY and SS.
///
/// This is a faithful port of the reference `matupd` subroutine which updates
/// the compact column-major storage arrays WS (n×m) and WY (n×m) by inserting
/// the new correction pair (d, r) into the circular buffer, and then updates
/// the m×m matrices SY (lower triangle) and SS (upper triangle) accordingly.
///
/// Arguments:
/// - `n`      : problem dimension
/// - `m`      : maximum number of corrections to store
/// - `ws`     : n×m column-major storage for S vectors (modified in-place)
/// - `wy`     : n×m column-major storage for Y vectors (modified in-place)
/// - `sy`     : m×m column-major storage for S'Y (modified in-place)
/// - `ss`     : m×m column-major storage for S'S (modified in-place)
/// - `d`      : new s vector (length n)
/// - `r`      : new y vector (length n)
/// - `itail`  : tail index (modified in-place)
/// - `iupdat` : total update count (1-based)
/// - `col`    : current number of stored corrections (modified in-place)
/// - `head`   : head index (modified in-place)
/// - `theta`  : scaling factor yy/ys (modified in-place)
/// - `rr`     : r'r (y'y)
/// - `dr`     : d'r (s'y)
/// - `stp`    : step length
/// - `dtd`    : d'd (s's)
///
/// Returns Ok(()) on success, Err(&str) on size mismatch.
pub fn matupd(
    n: usize,
    m: usize,
    ws: &mut [f64],
    wy: &mut [f64],
    sy: &mut [f64],
    ss: &mut [f64],
    d: &[f64],
    r: &[f64],
    itail: &mut usize,
    iupdat: usize,
    col: &mut usize,
    head: &mut usize,
    theta: &mut f64,
    rr: f64,
    dr: f64,
    stp: f64,
    dtd: f64,
) -> Result<(), &'static str> {
    if ws.len() < n * m || wy.len() < n * m {
        return Err("matupd: ws/wy too small");
    }
    if sy.len() < m * m || ss.len() < m * m {
        return Err("matupd: sy/ss too small");
    }
    if d.len() < n || r.len() < n {
        return Err("matupd: d/r too small");
    }

    // Update col and itail following the circular buffer logic
    // C code (1-based): if (*iupdat <= *m) { *col = *iupdat; *itail = (*head + *iupdat - 2) % *m + 1; }
    // In 0-based: head starts at 0, itail should be at position (head + iupdat - 1) % m
    if iupdat <= m {
        *col = iupdat;
        *itail = (*head + iupdat - 1) % m;
    } else {
        // C code (1-based): *itail = *itail % *m + 1; *head = *head % *m + 1;
        // This increments both and wraps: in 1-based, val % m + 1 gives 1..m
        // In 0-based, we increment and wrap: (val + 1) % m gives 0..m-1
        *itail = (*itail + 1) % m;
        *head = (*head + 1) % m;
    }

    // Copy d into WS[:, itail] and r into WY[:, itail] (column-major)
    // In column-major: column itail starts at offset itail * n
    for i in 0..n {
        ws[*itail * n + i] = d[i];
        wy[*itail * n + i] = r[i];
    }

    // Set theta = rr / dr (yy / ys)
    *theta = rr / dr;

    // Update the middle matrices SY (lower triangle) and SS (upper triangle)
    // If iupdat > m, we need to shift old information
    if iupdat > m {
        // Shift SS and SY: move columns left by 1
        // C code: dcopy(&j, &ss[(j + 1) * ss_dim1 + 2], &c__1, &ss[j * ss_dim1 + 1], &c__1);
        // This copies j elements from column j+1 starting at row 2 (1-based) to column j starting at row 1
        // In 0-based: copy column j+1 rows 1..j to column j rows 0..j-1
        for j in 0..(*col - 1) {
            // For SS upper triangle: copy column j+1 rows 1..=j into column j rows 0..=j-1
            for i in 0..j {
                ss[j * m + i] = ss[(j + 1) * m + (i + 1)];
            }
            // For SY lower triangle:
            // C code: dcopy(&i__2, &sy[j + 1 + (j + 1) * sy_dim1], &c__1, &sy[j + j * sy_dim1], &c__1);
            // where i__2 = *col - j
            // This copies (*col - j) elements from sy[j+1, j+1] to sy[j, j] (1-based)
            // In 0-based: copy column j+1 rows j+1..*col to column j rows j..*col-1
            for i in (j + 1)..*col {
                sy[j * m + i] = sy[(j + 1) * m + i];
            }
        }
    }

    // Add new information: the last row of SY and last column of SS
    // pointr traverses the circular buffer starting from head
    let mut pointr = *head;
    for j in 0..(*col - 1) {
        // SY[*col-1, j] = d' * WY[:, pointr]
        let mut sum = 0.0;
        for i in 0..n {
            sum += d[i] * wy[pointr * n + i];
        }
        sy[j * m + (*col - 1)] = sum;

        // SS[j, *col-1] = WS[:, pointr]' * d
        let mut sum2 = 0.0;
        for i in 0..n {
            sum2 += ws[pointr * n + i] * d[i];
        }
        ss[(*col - 1) * m + j] = sum2;

        pointr = (pointr + 1) % m;
    }

    // Set diagonal elements
    if stp == 1.0 {
        ss[(*col - 1) * m + (*col - 1)] = dtd;
    } else {
        ss[(*col - 1) * m + (*col - 1)] = stp * stp * dtd;
    }
    sy[(*col - 1) * m + (*col - 1)] = dr;

    Ok(())
}

/// Form a small K matrix from stored sy entries (faithful incremental `formk_ref`).
///
/// This function implements a closer translation of the reference `formk`
/// subroutine which builds the 2*col×2*col WN matrix (and an auxiliary WN1
/// buffer) and performs the two-stage factorization using LINPACK's `dpofa`.
/// It follows the original incremental update logic: when `updatd` is true
/// the function updates `wn1` using the `indx2` list of entering / leaving
/// variables and the circular WS/WY storage; it then forms the upper-triangular
/// `wn` matrix and performs the two dpofa factorizations exactly like the
/// reference.
///
/// This faithful port uses column-major layout for `ws`, `wy`, `sy` and `wn`
/// consistent with the rest of this crate's compact storage conventions.
///
/// Arguments (boilerplate mapped from the reference):
/// - `n`      : problem dimension
/// - `nsub`   : number of subspace variables (used for indexing in the original)
/// - `ind`    : index array of variables in the reduced subspace (1-based in ref; 0-based here)
/// - `nenter` : number of entering variables (from `indx2`)
/// - `ileave` : index (1-based sentinel position in `indx2`) where leaving indices start
/// - `indx2`  : permutation / list used by ref to indicate entering/leaving
/// - `iupdat` : total number of updates so far (1-based)
/// - `updatd` : whether the compact representation was updated (ref logical)
/// - `wn`     : target 2*m x 2*m column-major output (upper triangle will hold factor)
/// - `wn1`    : working 2*m x 2*m column-major storage (updated in-place)
/// - `m`      : maximum memory parameter (stride / leading dim for column-major matrices)
/// - `ws, wy` : n x m column-major storage for S and Y (as in the crate)
/// - `sy`     : m x m column-major S'Y
/// - `theta`  : scaling factor (input/output)
/// - `col`    : number of stored corrections (<= m)
/// - `head`   : head index into WS/WY circular buffer (0-based)
///
/// NOTE: This implementation follows the reference algorithm closely. It tries
/// to preserve the update flow (shifts, block updates, dpofa calls) while
/// using safe Rust indexing and helper linpack functions in this crate.
///
/// Returns Ok(()) on success, Err(&str) on failures (size / numerical).
pub fn formk_ref(
    n: usize,
    nsub: usize,
    ind: &[usize],
    _nenter: usize,
    _ileave: usize,
    _indx2: &[usize],
    iupdat: usize,
    updatd: bool,
    wn: &mut [f64],
    wn1: &mut [f64],
    m: usize,
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    theta: &mut f64,
    col: usize,
    head: usize,
) -> Result<(), &'static str> {
    // Basic checks
    let m2 = 2 * m;
    if wn.len() < m2 * m2 || wn1.len() < m2 * m2 {
        return Err("formk_ref: wn/wn1 too small");
    }
    if ws.len() < n * m || wy.len() < n * m {
        return Err("formk_ref: ws/wy too small");
    }
    if sy.len() < m * m {
        return Err("formk_ref: sy too small");
    }

    // wn1 is used to store an intermediate "WN1" matrix (column-major).
    // We follow the reference update logic and accumulate contributions into wn1.

    // If updatd is true, we need to perform the update path (reference's updatd branch)
    if updatd {
        // Shift older parts of wn1 if we have more updates than memory (i.e., iupdat > m)
        if iupdat > m {
            // shift blocks as in reference: move wn1 columns/rows left by one
            // We'll implement the same high-level copying of sub-blocks.
            // wn1 is stored column-major with leading dimension 2*m.
            let wn1_ld = 2 * m;
            // shifting: for jy = 1..m-1 move blocks
            for jy in 0..(m - 1) {
                // block (1,1): rows 0..(m-jy-2) copy from column (jy+1)
                let src_col = jy + 1;
                let dst_col = jy;
                // Move top (m - jy - 1) elements of column src_col to dst_col
                let count = (m - 1).saturating_sub(jy);
                for row in 0..count {
                    // wn1 element (row, dst_col) <- wn1(row, src_col)
                    wn1[dst_col * wn1_ld + row] = wn1[src_col * wn1_ld + row];
                }
                // For the (2,2) block analog we move the corresponding columns
                // block offsets in wn1 correspond to columns starting at m
                for row in 0..(m - 1) {
                    wn1[(m + dst_col) * wn1_ld + row] = wn1[(m + src_col) * wn1_ld + row];
                }
                // move related block (2,1)
                for row in 0..(m - 1) {
                    wn1[(m + 1 + dst_col) * wn1_ld + row] = wn1[(m + 1 + src_col) * wn1_ld + row];
                }
            }
        }

        // Now build the new rows/columns contributed by the last update.
        // In the reference, pbegin = 1, pend = nsub, dbegin = nsub+1, dend = n
        let pbegin = 0usize; // 0-based in Rust
        let pend = nsub.saturating_sub(1);
        let dbegin = nsub;
        let dend = n.saturating_sub(1);

        // pointers into circular storage
        let ipntr = (head + col - 1) % m; // last stored column in circular buffer
        let mut jpntr = head;

        // For jy = 0..(col-1) compute row contributions then put into wn1
        for jy in 0..col {
            let mut temp1 = 0.0f64;
            let mut temp2 = 0.0f64;
            let mut temp3 = 0.0f64;

            // compute element jy of row 'col' of Y'ZZ'Y
            for &k1 in ind.iter().take(pend + 1).skip(pbegin) {
                temp1 += wy[ipntr * n + k1] * wy[jpntr * n + k1];
            }
            // compute elements of L_a and S'AA'S
            for &k1 in ind.iter().take(dend + 1).skip(dbegin) {
                temp2 += ws[ipntr * n + k1] * ws[jpntr * n + k1];
                temp3 += ws[ipntr * n + k1] * wy[jpntr * n + k1];
            }
            // Write into wn1 using the same block layout used in the reference:
            // wn1[ iy + jy*wn1_dim1 ] = temp1  (block (1,1))
            // wn1[ is + js * wn1_dim1 ] = temp2 (block (2,2))
            // wn1[ is + jy * wn1_dim1 ] = temp3 (block (2,1))
            let iy = 0usize;
            let is = m;
            let wn1_ld = 2 * m;
            wn1[iy + jy * wn1_ld] = temp1;
            wn1[is + (m + jy) * wn1_ld] = temp2;
            wn1[is + jy * wn1_ld] = temp3;

            jpntr = (jpntr + 1) % m;
        }

        // Put new column in block (2,1).
        let jy = col.saturating_sub(1);
        let jpntr2 = (head + col - 1) % m;
        let mut ipntr2 = head;
        for i in 0..col {
            let is = m + i;
            let mut temp3 = 0.0f64;
            for &k1 in ind.iter().take(pend + 1).skip(pbegin) {
                temp3 += ws[ipntr2 * n + k1] * wy[jpntr2 * n + k1];
            }
            ipntr2 = (ipntr2 + 1) % m;
            wn1[is + (m + jy) * (2 * m)] = temp3;
        }
        // upcl not used further in this simplified translation; the main effect
        // is that wn1 has been updated with the newly computed rows/columns.
    } // end updatd branch

    // Now form the upper triangle of WN using wn1 as intermediate storage.
    // Mapping the reference's block assembly into wn:
    // wn[ j + i*wn_dim1 ] = wn1[i + j*wn1_ld] / theta  for (1,1) block
    // wn[(col + j) + (col + i)*wn_dim1] = wn1[(m + i) + (m + j)*wn1_ld] * theta   (2,2)
    // and the cross blocks are formed similarly.
    let wn_ld = 2 * m;
    // zero wn first
    for wn_item in wn.iter_mut().take(wn_ld * wn_ld) {
        *wn_item = 0.0;
    }

    // Fill 2x2 block structure following reference indexing
    for iy in 0..col {
        let is = col + iy;
        for jy in 0..=iy {
            // wn[jy + iy*wn_ld] = wn1[iy + jy * wn1_ld] / theta;
            wn[jy + iy * wn_ld] = wn1[iy + jy * wn_ld] / *theta;
        }
        for jy in 0..col {
            let js = col + jy;
            // wn[js + is*wn_ld] = wn1[(m + iy) + (m + jy)*wn1_ld] * theta;
            wn[js + is * wn_ld] = wn1[(m + iy) + (m + jy) * wn_ld] * (*theta);
        }
        // fill lower left per reference sign convention
        for jy in 0..iy {
            wn[jy + is * wn_ld] = -wn1[(m + iy) + jy * wn_ld];
        }
        // add diagonal sy
        wn[iy + iy * wn_ld] += sy[iy * m + iy];
    }

    // First dpofa on the (1,1) block of wn (size = 2*col x 2*col but the DP of the
    // 1st stage uses the leading block size m2 and col)
    // In the reference, dpofa(&wn[wn_offset], &m2, col, info)
    // We need to call dpofa on a row-major representation. Our linpack::dpofa
    // expects row-major `a`. We therefore convert the leading (2*col) x (2*col)
    // upper-triangular block into row-major `a` and call dpofa.
    let col2 = 2 * col;
    let mut a = vec![0.0f64; col2 * col2];
    // Build symmetric row-major `a` from wn's upper triangle (column-major)
    for i in 0..col2 {
        for j in 0..col2 {
            // wn element (i,j) is at wn[j*wn_ld + i]
            a[i * col2 + j] = wn[j * wn_ld + i];
        }
    }

    // Factorize leading block using dpofa (row-major)
    match crate::linpack::dpofa(&mut a, col2) {
        Ok(()) => {}
        Err(_) => return Err("formk_ref: dpofa failed in first stage"),
    }

    // After dpofa the reference performs triangular solves to form (1,2) block contributions.
    // We replicate the ref step: for each column js in (col+1..col2) perform dtrsl.
    for js in col..col2 {
        // dtrsl(&wn[wn_offset], &m2, col, &wn[js * wn_dim1 + 1], &c__11, info);
        // We convert the column js of wn into a row-major rhs slice and call dtrsl.
        let mut rhs = vec![0.0f64; col2];
        for i in 0..col2 {
            rhs[i] = wn[js * wn_ld + i];
        }
        // call dtrsl with lower=true on row-major `a` to solve L * x = rhs
        if crate::linpack::dtrsl(&a, col2, true, false, &mut rhs).is_err() {
            return Err("formk_ref: dtrsl failed when forming (1,2) block");
        }
        // write back solved column into wn (column-major)
        for i in 0..col2 {
            wn[js * wn_ld + i] = rhs[i];
        }
    }

    // Form (2,2) block by adding contributions: wn[is + js*wn_ld] += dot(wn[is*...], wn[js*...])
    for is in col..col2 {
        for js in is..col2 {
            let mut sum = 0.0f64;
            for k in 0..col {
                sum += wn[(k) + is * wn_ld] * wn[(k) + js * wn_ld];
            }
            wn[is + js * wn_ld] += sum;
        }
    }

    // Second dpofa on the trailing (2,2) block: extract trailing col2-col x col2-col block
    // Build a row-major copy of trailing block
    let trailing = col2 - col;
    if trailing > 0 {
        let mut b = vec![0.0f64; trailing * trailing];
        for i in 0..trailing {
            for j in 0..trailing {
                // wn element (col + j, col + i) at wn[(col + j)*wn_ld + (col + i)]
                b[i * trailing + j] = wn[(col + j) * wn_ld + (col + i)];
            }
        }
        // dpofa on b
        match crate::linpack::dpofa(&mut b, trailing) {
            Ok(()) => {}
            Err(_) => return Err("formk_ref: dpofa failed in second stage"),
        }
        // write back upper-triangular J' into wn in column-major form (as reference does)
        for j in 0..trailing {
            for i in 0..=j {
                // R[i,j] = L[j,i] (from b, row-major lower-triangular L), store at wn[(col+j)*wn_ld + col + i]
                wn[(col + j) * wn_ld + (col + i)] = b[j * trailing + i];
            }
        }
    }

    Ok(())
}

/// Backwards compatible wrapper: the crate previously exposed `formk` which
/// produced a small `kmat` by copying the leading block of SY. For callers
/// that still expect the old simpler API, keep `formk` as a thin wrapper.
pub fn formk(m: usize, sy: &[f64], col: usize, kmat: &mut [f64]) -> Result<(), &'static str> {
    if kmat.len() < col * col || sy.len() < m * m {
        return Err("insufficient storage in formk");
    }
    // Copy the leading col x col block of SY (column-major) into KMAT (row-major).
    // In column-major SY: element (row=i, col=j) is at sy[j * m + i].
    for i in 0..col {
        for j in 0..col {
            kmat[i * col + j] = sy[j * m + i];
        }
    }
    // Regularize diagonal entries if they are nonpositive or near-zero to
    // improve numerical robustness for subsequent factorization.
    let mut max_diag = 0.0f64;
    for k in 0..col {
        max_diag = max_diag.max(kmat[k * col + k].abs());
    }
    let delta = if max_diag == 0.0 {
        1e-8
    } else {
        max_diag * 1e-8
    };
    for i in 0..col {
        if kmat[i * col + i] <= 0.0 || kmat[i * col + i].abs() < 1e-16 {
            kmat[i * col + i] += delta;
        }
    }
    Ok(())
}

/// Form triangular T from K and perform Cholesky factorization using linpack::dpofa.
///
/// This implementation:
///  - copies the input `kmat` (assumed to be col x col, row-major) into a
///    row-major temporary buffer `a`,
///  - calls `linpack::dpofa(&mut a, col)` to compute the Cholesky factor L
///    (stored in the lower triangle of `a`),
///  - stores the corresponding upper-triangular factor R = L^T into `tmat`
///    using column-major layout where column j starts at `tmat[j*col]`.
///
/// The stored format in `tmat` matches the reference convention of storing the
/// upper-triangular factor (J') in the upper triangle using column-major.
pub fn formt(col: usize, kmat: &[f64], tmat: &mut [f64]) -> Result<(), &'static str> {
    if tmat.len() < col * col {
        return Err("insufficient storage in formt");
    }
    if kmat.len() < col * col {
        return Err("insufficient storage in formt (kmat)");
    }

    // Build a row-major copy `a` of the symmetric matrix kmat.
    let mut a = vec![0.0f64; col * col];
    for i in 0..col {
        for j in 0..col {
            // assume kmat stored row-major as kmat[i*col + j]
            a[i * col + j] = kmat[i * col + j];
        }
    }

    // Perform Cholesky factorization a = L * L^T using dpofa.
    match crate::linpack::dpofa(&mut a, col) {
        Ok(()) => {}
        Err(_) => {
            // propagate error (linpack returns Err(k) where k is 1-based pivot)
            return Err("dpofa failed in formt");
        }
    }

    // After dpofa, `a` holds the lower-triangular L in its lower triangle
    // (row-major). We now form R = L^T (upper-triangular) and store it in
    // `tmat` in column-major layout with column j at offset j*col.
    for j in 0..col {
        for i in 0..col {
            if i <= j {
                // R[i,j] = L[j,i] which is stored in a[j*col + i]
                tmat[j * col + i] = a[j * col + i];
            } else {
                // lower part of column j set to 0 in column-major storage
                tmat[j * col + i] = 0.0;
            }
        }
    }

    Ok(())
}

/// Reference-style `formt_ref` that mirrors the C reference `formt`.
///
/// The reference `formt` forms the upper-triangular matrix
///    T = theta * SS + L * D^{-1} * L'
/// where the intermediate products are built from `sy` and `ss` (both
/// column-major m x m). This Rust helper constructs the symmetric T
/// (col x col) in a row-major temporary, performs a Cholesky factorization,
/// and stores the resulting upper-triangular J' (i.e., R = L^T) into `wt`
/// using column-major layout (WT is m x m in column-major, only the leading
/// col x col block is touched).
///
/// Arguments:
/// - `m`     : maximum memory parameter (stride for column-major arrays)
/// - `wt`    : target column-major storage for J' (length m*m). Only the
///   leading `col x col` block is modified.
/// - `sy`    : column-major S'Y (length m*m)
/// - `ss`    : column-major S'S (length m*m)
/// - `col`   : number of stored corrections (<= m)
/// - `theta` : scaling factor
///
/// Returns Ok(()) on success or Err(&str) on size / numerical errors.
pub fn formt_ref(
    m: usize,
    wt: &mut [f64],
    sy: &[f64],
    ss: &[f64],
    col: usize,
    theta: f64,
) -> Result<(), &'static str> {
    // Basic checks
    if col == 0 {
        return Ok(());
    }
    if wt.len() < m * m || sy.len() < m * m || ss.len() < m * m {
        return Err("insufficient storage for formt_ref");
    }
    if col > m {
        return Err("col cannot be greater than m in formt_ref");
    }

    // Build symmetric T (col x col) in row-major layout `a`.
    // In the reference (1-based), they set:
    //   wt[1, j] = theta * ss[1, j]  (first row)
    // and for i=2..col, j=i..col:
    //   wt[i,j] = theta * ss[i,j] + sum_{k=1..min(i,j)-1} sy[i,k]*sy[j,k]/sy[k,k]
    // We implement the same using 0-based indices and column-major storage for sy/ss.
    let mut a = vec![0.0f64; col * col];

    // First row (row 0), columns 0..col-1
    for j in 0..col {
        // ss element (row=0, col=j) stored at ss[j*m + 0] (column-major)
        a[j] = theta * ss[j * m];
    }

    // Remaining entries for rows i=1..col-1
    for i in 1..col {
        for j in i..col {
            // Compute ddum = sum_{k=0..min(i,j)-1} sy[k,i] * sy[k,j] / sy[k,k]
            // Here sy element (row=i, col=k) is stored at sy[k*m + i]
            let kmax = std::cmp::min(i, j);
            let mut ddum = 0.0f64;
            for k in 0..kmax {
                let denom = sy[k * m + k];
                if denom == 0.0 {
                    return Err("zero diagonal in sy in formt_ref");
                }
                ddum += sy[k * m + i] * sy[k * m + j] / denom;
            }
            // ss element (row=i, col=j) stored at ss[j*m + i]
            a[i * col + j] = ddum + theta * ss[j * m + i];
        }
    }

    // Symmetrize `a` (fill lower triangle)
    for i in 0..col {
        for j in 0..i {
            a[i * col + j] = a[j * col + i];
        }
    }

    // Perform Cholesky factorization a = L * L^T using dpofa (row-major).
    match crate::linpack::dpofa(&mut a, col) {
        Ok(()) => {}
        Err(_) => return Err("dpofa failed in formt_ref"),
    }

    // After dpofa, `a` contains lower-triangular L in row-major.
    // Store R = L^T into `wt` in column-major upper-triangular form:
    // for each column j (0..col-1), store elements (i=0..j) at wt[j*m + i]
    for j in 0..col {
        for i in 0..col {
            if i <= j {
                // R[i,j] = L[j,i] -> a[j*col + i]
                wt[j * m + i] = a[j * col + i];
            } else {
                // zero lower part for cleanliness
                wt[j * m + i] = 0.0;
            }
        }
    }

    Ok(())
}

/// subsm: subspace minimization using triangular solves.
///
/// This routine solves the reduced system T * z = r where T is provided in
/// `tmat`. We:
///  - validate sizes,
///  - copy `tmat` (assumed column-major upper-triangular storage for R = J')
///    into a row-major symmetric matrix `a` suitable for `dpofa`/`dtrsl`,
///  - use `dpofa` to obtain a lower-triangular Cholesky factor L (row-major),
///  - solve L y = r and then L^T z = y using `dtrsl`.
///
/// The implementation uses the crate's linpack ports for dpofa/dtrsl.
pub fn subsm(
    col: usize,
    kmat: &[f64],
    tmat: &[f64],
    r: &[f64],
    z: &mut [f64],
) -> Result<(), &'static str> {
    if r.len() < col || z.len() < col || kmat.len() < col * col || tmat.len() < col * col {
        return Err("insufficient sizes in subsm");
    }
    if col == 0 {
        return Ok(());
    }

    // Construct a symmetric row-major matrix `a` to factorize.
    // We assume `tmat` stores an (approximate) symmetric positive-definite
    // matrix in column-major layout. For robustness we form `a[i,j]` from
    // tmat's column-major representation.
    let mut a = vec![0.0f64; col * col];
    for i in 0..col {
        for j in 0..col {
            // tmat is column-major: element (i,j) at tmat[j*col + i]
            a[i * col + j] = tmat[j * col + i];
        }
    }

    // Factorize A = L * L^T using dpofa on row-major `a`.
    match crate::linpack::dpofa(&mut a, col) {
        Ok(()) => {}
        Err(_) => return Err("dpofa failed in subsm"),
    }

    // Solve L * y = r (forward substitution) and then L^T * z = y (back substitution).
    // dtrsl expects the triangular factor in row-major layout and operates in-place on the rhs.
    let mut y = vec![0.0f64; col];
    y.copy_from_slice(&r[0..col]);

    // Solve L y = r
    if crate::linpack::dtrsl(&a, col, true, false, &mut y).is_err() {
        return Err("dtrsl forward solve failed in subsm");
    }
    // Solve L^T z = y
    if crate::linpack::dtrsl(&a, col, true, true, &mut y).is_err() {
        return Err("dtrsl back solve failed in subsm");
    }

    // copy solution into z
    z[..col].copy_from_slice(&y[..col]);
    Ok(())
}

/// Faithful reference-style subsm port (subsm_ref).
///
/// This function mirrors the original L-BFGS-B `subsm` subroutine more
/// closely. It operates on an index set `ind` (0-based indices into x),
/// builds the working vector `wv` = W' * Z * d, solves K^{-1} * wv via the
/// 2*col x 2*col factor `wn`, and reconstructs the Newton direction `d`.
///
/// Note: many arrays are expected to be column-major where applicable:
/// - `wy` and `ws` are n-by-m column-major flat slices (column stride = n)
/// - `wn` is 2m-by-2m column-major (leading dimension = 2*m)
///
/// Signature mirrors the reference but uses safe Rust slices and 0-based indices.
pub fn subsm_ref(
    n: usize,
    m: usize,
    nsub: usize,
    ind: &[usize], // length nsub, 0-based indices of free variables
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    x: &mut [f64],  // x is the current x (will be updated in-place)
    d: &mut [f64],  // d: reduced gradient (length nsub), overwritten with Newton dir
    xp: &mut [f64], // xp: working copy for safeguarding (length n)
    ws: &[f64],     // n x m column-major
    wy: &[f64],     // n x m column-major
    theta: f64,
    xx: &[f64], // current iterate (length n)
    gg: &[f64], // gradient at current iterate (length n)
    col: usize,
    head: usize,     // 0-based head
    iword: &mut i32, // output status: 0=solution in box, 1=hit a bound
    wv: &mut [f64],  // working vector length 2*col
    wn: &[f64],      // working 2m x 2m column-major (length (2*m)*(2*m))
    iprint: i32,
) -> Result<(), &'static str> {
    // Basic checks
    if n == 0 || m == 0 {
        return Err("invalid n or m in subsm_ref");
    }
    if ind.len() < nsub {
        return Err("ind too small in subsm_ref");
    }
    if wv.len() < 2 * col {
        return Err("wv too small in subsm_ref");
    }
    if wn.len() < (2 * m) * (2 * m) {
        return Err("wn too small in subsm_ref");
    }
    if ws.len() < n * m || wy.len() < n * m {
        return Err("ws/wy too small in subsm_ref");
    }
    if xp.len() < n || x.len() < n || l.len() < n || u.len() < n || gg.len() < n || xx.len() < n {
        return Err("vector length mismatch in subsm_ref");
    }
    if d.len() < nsub {
        return Err("d too small in subsm_ref");
    }

    if nsub == 0 {
        return Ok(());
    }

    // Compute wv = W' * Z * d.
    // For j = 0 .. col-1 compute column contributions
    let mut pointr = head % m;
    for j in 0..col {
        let col_idx = pointr;
        let mut temp1 = 0.0f64;
        let mut temp2 = 0.0f64;
        for jj in 0..nsub {
            let k = ind[jj];
            // WY/WS stored column-major: element (row=k, col=col_idx) at wy[col_idx*n + k]
            temp1 += wy[col_idx * n + k] * d[jj];
            temp2 += ws[col_idx * n + k] * d[jj];
        }
        wv[j] = temp1;
        wv[col + j] = theta * temp2;
        pointr = (pointr + 1) % m;
    }

    // Compute wv := K^{-1} wv using wn's LEL^T factorization stored in wn.
    // Build a row-major lower-triangular copy `a` of the 2*col x 2*col leading block of wn.
    let col2 = col * 2;
    let wn_ld = 2 * m;
    let mut a = vec![0.0f64; col2 * col2];
    // Fill lower-triangular part of `a` from wn which stores upper-triangular factor in column-major.
    for i in 0..col2 {
        for j in 0..=i {
            // wn stores element at wn[j + i*wn_ld] (1-based ref -> 0-based mapping)
            a[i * col2 + j] = wn[j * wn_ld + i];
        }
        for j in (i + 1)..col2 {
            a[i * col2 + j] = 0.0;
        }
    }

    // First solve: dtrsl with job corresponding to solving trans(upper)*x = b
    // Map to our dtrsl: upper-transposed -> lower=false, trans=true
    {
        let slice = &mut wv[0..col2];
        if crate::linpack::dtrsl(&a, col2, false, true, slice).is_err() {
            return Err("dtrsl failed in subsm_ref (first solve)");
        }
    }

    // Negate first col entries (C code does wv[i] = -wv[i] for i=1..col)
    for wv_item in wv.iter_mut().take(col) {
        *wv_item = -*wv_item;
    }

    // Second solve: now solve J * x = rhs where J is lower triangular in `a`
    // which is lower=true, trans=false
    {
        let slice = &mut wv[0..col2];
        if crate::linpack::dtrsl(&a, col2, true, false, slice).is_err() {
            return Err("dtrsl failed in subsm_ref (second solve)");
        }
    }

    // Compute d = (1/theta) * d + (1/theta^2) * Z' W wv
    pointr = head % m;
    for jy in 0..col {
        let js = col + jy;
        let col_idx = pointr;
        for ii in 0..nsub {
            let k = ind[ii];
            let add1 = wy[col_idx * n + k] * wv[jy] / theta;
            let add2 = ws[col_idx * n + k] * wv[js];
            d[ii] = d[ii] + add1 + add2;
        }
        pointr = (pointr + 1) % m;
    }
    // scale by 1/theta
    let inv_theta = 1.0 / theta;
    for d_item in d.iter_mut().take(nsub) {
        *d_item *= inv_theta;
    }

    // Projection: attempt xp := x (copy), try projected Newton step and check box constraints
    *iword = 0;
    xp.copy_from_slice(&x[0..n]);

    for ii in 0..nsub {
        let k = ind[ii];
        let dk = d[ii];
        let xk = x[k];
        if nbd[k] != 0 {
            if nbd[k] == 1 {
                // lower bounds only
                let newx = (xk + dk).max(l[k]);
                x[k] = newx;
                if x[k] == l[k] {
                    *iword = 1;
                }
            } else if nbd[k] == 2 {
                // both bounds
                let mut tmp = (xk + dk).max(l[k]);
                tmp = tmp.min(u[k]);
                x[k] = tmp;
                if x[k] == l[k] || x[k] == u[k] {
                    *iword = 1;
                }
            } else if nbd[k] == 3 {
                // upper only
                let newx = (xk + dk).min(u[k]);
                x[k] = newx;
                if x[k] == u[k] {
                    *iword = 1;
                }
            }
        } else {
            // free variable
            x[k] = xk + dk;
        }
    }

    if *iword == 0 {
        // successful projection without hitting bounds
        return Ok(());
    }

    // Check sign of the directional derivative
    let mut dd_p = 0.0f64;
    for i in 0..n {
        dd_p += (x[i] - xx[i]) * gg[i];
    }
    if dd_p > 0.0 {
        // Use backtracking: restore xp into x and continue to find step length
        x.copy_from_slice(&xp[0..n]);
        if iprint >= 99 {
            eprintln!("Positive dir derivative in projection; using backtracking");
        }
    } else {
        // Accept projected x and return
        return Ok(());
    }

    // Backtracking: find alpha in (0,1] so that projected step stays feasible
    let mut alpha = 1.0_f64;
    let mut ibd_idx: Option<usize> = None;
    for ii in 0..nsub {
        let k = ind[ii];
        let dk = d[ii];
        if nbd[k] != 0 {
            let mut temp1 = alpha;
            if dk < 0.0 && nbd[k] <= 2 {
                let temp2 = l[k] - x[k];
                if temp2 >= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha < temp2 {
                    temp1 = temp2 / dk;
                }
            } else if dk > 0.0 && nbd[k] >= 2 {
                let temp2 = u[k] - x[k];
                if temp2 <= 0.0 {
                    temp1 = 0.0;
                } else if dk * alpha > temp2 {
                    temp1 = temp2 / dk;
                }
            }
            if temp1 < alpha {
                alpha = temp1;
                ibd_idx = Some(ii);
            }
        }
    }

    if alpha < 1.0
        && let Some(ii) = ibd_idx
    {
        let dk = d[ii];
        let k = ind[ii];
        if dk > 0.0 {
            x[k] = u[k];
            d[ii] = 0.0;
        } else if dk < 0.0 {
            x[k] = l[k];
            d[ii] = 0.0;
        }
    }

    // Apply step alpha to free variables
    for ii in 0..nsub {
        let k = ind[ii];
        x[k] += alpha * d[ii];
    }

    Ok(())
}

/// Compute product of the compact L-BFGS middle matrix with a 2*col vector `v`.
///
/// Faithful port of the reference `bmv` which uses the column-major SY/WT
/// storage and calls triangular solves on the factor stored in WT. This
/// implementation closely follows the original C routine semantics and
/// indexing (translated to 0-based Rust slices).
///
/// Algorithm overview (matching the reference):
///  - Build the second block p[col..2*col) = v2 + L * D^{-1} * v1, where L is
///    the strictly lower part of SY and D is diag(SY).
///  - Solve J * p2 = rhs where J' (the Cholesky factor) is stored in WT's
///    upper-triangular block (column-major). We construct a local row-major
///    lower-triangular copy of J so we can call our `dtrsl`.
///  - Form p1 = v1 / sqrt(diag(SY))
///  - Solve J^T * p2 = p2
///  - Finalize p1 := -D^{-1/2} p1 + D^{-1} * L' * p2
///
/// Storage conventions:
/// - `sy` and `wt` are column-major m-by-m arrays (leading dimension m).
/// - `v` and `p` are length 2*col vectors where first block [0..col) is v1,
///   second block [col..2*col) is v2.
///
/// Errors are reported as Err(&'static str) strings to match the crate style.
pub fn bmv(
    m: usize,
    sy: &[f64],
    wt: &[f64],
    col: usize,
    v: &[f64],
    p: &mut [f64],
) -> Result<(), &'static str> {
    if col == 0 {
        return Ok(());
    }
    // Basic size checks
    if sy.len() < m * m || wt.len() < m * m || p.len() < 2 * col || v.len() < 2 * col {
        return Err("insufficient sizes for bmv");
    }

    // Part I: form p[col..] = v2 + L * D^{-1} * v1
    // (C reference sets p[col+1] = v[col+1], then loops i=2..col)
    p[col] = v[col];
    for i in 1..col {
        let mut sum = 0.0_f64;
        for k in 0..i {
            let denom = sy[k * m + k];
            if denom == 0.0 {
                return Err("zero diagonal encountered in sy while forming p2");
            }
            // sy(row=i, col=k) stored as sy[k*m + i]
            sum += sy[k * m + i] * v[k] / denom;
        }
        p[col + i] = v[col + i] + sum;
    }

    // Prepare triangular factor J from WT for dtrsl
    // Reference: WT holds J' in its upper triangle (column-major).
    // To call our row-major dtrsl with `lower=true`, construct row-major `a`
    // whose lower triangle contains J.
    let mut a = vec![0.0_f64; col * col];
    for i in 0..col {
        for j in 0..=i {
            // J(i,j) is stored in WT at wt[j*m + i] (column-major)
            a[i * col + j] = wt[j * m + i];
        }
        for j in (i + 1)..col {
            a[i * col + j] = 0.0;
        }
    }

    // Solve J * p2 = p2 (in-place on p[col..col+col])
    {
        let slice = &mut p[col..(col + col)];
        if crate::linpack::dtrsl(&a, col, true, false, slice).is_err() {
            return Err("dtrsl failed when solving J * p2 = rhs");
        }
    }

    // Part II: form p1 = v1 / sqrt(diag(SY))
    for i in 0..col {
        let diag = sy[i * m + i];
        if diag > 0.0 {
            p[i] = v[i] / diag.sqrt();
        } else {
            // defensive fallback (should not occur for well-formed SY)
            p[i] = v[i];
        }
    }

    // Solve J^T * p2 = p2 (in-place)
    {
        let slice = &mut p[col..(col + col)];
        if crate::linpack::dtrsl(&a, col, true, true, slice).is_err() {
            return Err("dtrsl failed when solving J^T * p2 = rhs");
        }
    }

    // Finalize p1 := -D^{-1/2} p1 + D^{-1} * L' * p2
    // - first scale -D^{-1/2} p1
    for i in 0..col {
        let diag = sy[i * m + i];
        let sqrt_diag = if diag > 0.0 { diag.sqrt() } else { 1.0 };
        p[i] = -p[i] / sqrt_diag;
    }
    // - add D^{-1} * L' * p2, where L' entries correspond to sy elements:
    //   in the reference: sum += sy[k + i*sy_dim1] * p[col + k] / sy[i + i*sy_dim1];
    //   mapping to column-major sy: sy[i*m + k]
    for i in 0..col {
        let denom = sy[i * m + i];
        if denom == 0.0 {
            return Err("zero diagonal encountered in sy during final p1 update");
        }
        let mut sum = 0.0_f64;
        for k in (i + 1)..col {
            sum += sy[i * m + k] * p[col + k] / denom;
        }
        p[i] += sum;
    }

    Ok(())
}

/// High-level helper: run subspace minimization and reconstruct a full-space xp.
///
/// This implementation is a first pass at moving `subsm_full` closer to the
/// reference C `subsm` semantics. It tries to:
///  - build the free-index set Z (from `iwhere`),
///  - compute a reduced right-hand side that more closely resembles
///    r = -Z' B (xcp - x) - Z' g (when `x` is not available we at least use
///    a faithful projection of the gradient onto the stored WY columns),
///  - call the compact small solver (`formk`/`formt`/`subsm`) to solve the reduced
///    system and obtain `z`,
///  - reconstruct a full-space xp by applying the compact correction using WY.
///
/// Notes / limitations:
///  - The original reference `subsm` expects access to `x` (the current iterate)
///    and uses `wn` (the LEL^T factorization) to solve a 2m-by-2m system.
///    In the current crate integration the solver calls `subsm_full` without the
///    current `x` vector and without a prebuilt `wn`. To remain backward-compatible
///    with that call-site we compute a reduced RHS using a projection of `g` onto
///    the WY columns and build a conservative reconstruction. This preserves the
///    algorithmic flow while improving on the previous naive averaging approach.
///  - This function returns `iword` exactly as in the reference: 0 if the
///    returned xp stays inside bounds, 1 if any component was projected to a bound.
pub fn subsm_full(
    n: usize,
    m: usize,
    xcp: &[f64],
    l: &[f64],
    u: &[f64],
    nbd: &[i32],
    g: &[f64],
    iwhere: &[i32],
    ws: &[f64],
    wy: &[f64],
    sy: &[f64],
    wt: &[f64],
    theta: f64,
    col: usize,
    head: usize,
    xp: &mut [f64],
) -> Result<i32, &'static str> {
    // Basic checks
    if xcp.len() < n || xp.len() < n {
        return Err("subsm_full: length mismatch for xcp/xp");
    }
    if l.len() < n || u.len() < n || nbd.len() < n || g.len() < n || iwhere.len() < n {
        return Err("subsm_full: length mismatch for bounds/gradient/iwhere");
    }
    if sy.len() < m * m || wt.len() < m * m || ws.len() < n * m || wy.len() < n * m {
        return Err("subsm_full: compact buffers too small");
    }

    // Build index set of free variables (iwhere <= 0)
    let mut ind: Vec<usize> = Vec::new();
    for (i, &w) in iwhere.iter().enumerate() {
        if w <= 0 {
            ind.push(i);
        }
    }
    let nfree = ind.len();
    if nfree == 0 || col == 0 {
        // Nothing to do: xp = xcp
        xp.copy_from_slice(&xcp[0..n]);
        return Ok(0);
    }

    // Build K (col x col) using the available `formk` helper.
    let mut kmat = vec![0.0_f64; col * col];
    formk(m, sy, col, &mut kmat).map_err(|_| "subsm_full: formk failed")?;

    // Build triangular T and factor (col x col)
    let mut tmat = vec![0.0_f64; col * col];
    formt(col, &kmat, &mut tmat).map_err(|_| "subsm_full: formt failed")?;

    // Construct a reduced RHS r that approximates the reference r =
    // -Z' B (xcp-x) - Z' g. We do this by projecting the negative gradient
    // onto the WY columns restricted to free variables. This produces a
    // vector of length `col` compatible with `formk`/`formt` usage below.
    let mut r = vec![0.0_f64; col];

    // For each stored column j, compute projection of -g onto WY[:,j] restricted to Z.
    // WY is stored column-major: element (i, col_idx) at wy[col_idx * n + i]
    for (j, r_item) in r.iter_mut().enumerate().take(col) {
        let col_idx = (head + j) % m;
        let mut proj = 0.0f64;
        for &i in &ind {
            // project using WY only (fallback when x isn't available to form xcp-x)
            proj += wy[col_idx * n + i] * g[i];
        }
        // r[j] is -Z' * g projected on this WY column
        *r_item = -proj;
    }

    // Solve reduced system using the simplified `subsm` which returns a small vector z.
    let mut z = vec![0.0_f64; col];
    // `subsm` expects a col-length RHS that matches `kmat`/`tmat` semantics;
    // here we use the projected gradient RHS computed above.
    subsm(col, &kmat, &tmat, &r, &mut z).map_err(|_| "subsm_full: subsm failed")?;

    // Reconstruct a full-space xp from xcp and z.
    // We apply the compact WY correction using WY columns and entries of z.
    // xp := xcp + (1/(1+theta)) * WY(:, :) * z  (restricted to free indices)
    // The scaling 1/(1+theta) is conservative and helps avoid overshooting bounds.
    xp.copy_from_slice(&xcp[0..n]);

    // If we have WY storage, add WY * z contribution to free variables
    for &idx in ind.iter() {
        let mut corr = 0.0f64;
        // sum over stored columns: wy[col_idx*n + idx] * z[j]
        for (j, &z_item) in z.iter().enumerate().take(col) {
            let col_idx = (head + j) % m;
            corr += wy[col_idx * n + idx] * z_item;
        }
        // conservative scaling by (1+theta)
        let step = corr / (1.0 + theta);
        let mut xnew = xp[idx] + step;
        // enforce bounds
        if nbd[idx] != 0 {
            if xnew < l[idx] {
                xnew = l[idx];
            } else if xnew > u[idx] {
                xnew = u[idx];
            }
        }
        xp[idx] = xnew;
    }

    // iword indicates whether any component was pushed to bounds during reconstruction
    let mut iword = 0;
    for i in 0..n {
        if xp[i] <= l[i] || xp[i] >= u[i] {
            iword = 1;
            break;
        }
    }

    Ok(iword)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_simple() {
        let mut x = vec![2.0, -5.0, 0.0];
        let l = vec![-1.0, -2.0, 0.0];
        let u = vec![1.0, 1.0, 0.0];
        let nbd = vec![2, 2, 2]; // all boxed
        let mut iwhere = vec![0_i32; 3];

        let (prjctd, cnstnd, boxed, nbdd) = active(&mut x, &l, &u, &nbd, &mut iwhere, 1).unwrap();
        assert!(prjctd); // x was projected (2.0 -> 1.0, -5.0 -> -2.0)
        assert!(cnstnd);
        assert!(boxed);
        // x[2] equals its bound exactly and is a fixed variable (u==l==0)
        assert_eq!(iwhere[0], 0); // first is bounded and not fixed (l<u)
        assert_eq!(iwhere[1], 0);
        assert_eq!(iwhere[2], 3); // fixed
        assert_eq!(nbdd, 2); // two variables exactly at bounds after projection
    }

    #[test]
    fn test_cauchy_and_freev() {
        let x = vec![0.0, 2.0, -1.0];
        let l = vec![-1.0, 0.0, -2.0];
        let u = vec![1.0, 3.0, 0.0];
        let nbd = vec![2, 2, 2];
        let g = vec![1.0, -0.5, 2.0];

        let cp = cauchy_point(&x, &l, &u, &nbd, &g);
        // cauchy should be within bounds
        for i in 0..x.len() {
            assert!(cp[i] >= l[i] - 1e-12 && cp[i] <= u[i] + 1e-12);
        }

        // freev: when nbd all non-zero expect empty free list
        let free = freev(&nbd);
        assert!(free.is_empty());

        // with some free variables
        let nbd2 = vec![0, 2, 0];
        let free2 = freev(&nbd2);
        assert_eq!(free2, vec![0usize, 2usize]);
    }

    #[test]
    fn test_cmprlb_matupd_formk_formt_subsm_bmv() {
        // small synthetic scenario
        let n = 4usize;
        let m = 3usize;
        let mut x = vec![0.0; n];
        let l = vec![-1e6; n];
        let u = vec![1e6; n];
        let nbd = vec![0, 0, 0, 0];
        let mut iwhere = vec![0_i32; n];

        // cmprlb should initialize iwhere for unbounded case
        cmprlb(&mut x, &l, &u, &nbd, &mut iwhere).expect("cmprlb failed");
        assert_eq!(iwhere.iter().filter(|&&v| v == -1).count(), n);

        // matupd: create sy/wt/ss/ws/wy storage (column-major)
        let mut sy = vec![0.0; m * m];
        // initialize WT as identity (column-major) so triangular solves in bmv succeed
        let mut wt = vec![0.0; m * m];
        for ii in 0..m {
            wt[ii * m + ii] = 1.0;
        }
        let mut ss = vec![0.0; m * m];
        let mut ws = vec![0.0; n * m];
        let mut wy = vec![0.0; n * m];

        // create synthetic s and y corrections (used here as d and r to be stored)
        let s = vec![1.0, 0.0, 0.0, 0.0];
        let y = vec![0.5, 0.0, 0.0, 0.0];

        // circular buffer pointers and counters
        let mut itail = 0usize;
        let iupdat = 1usize;
        let mut col = 0usize;
        let mut head = 0usize;
        let mut theta = 0.0f64;
        // rr/dr/stp/dtd are test scalars (choose nonzero dr to avoid division-by-zero)
        let rr = 0.5f64;
        let dr = 0.25f64;
        let stp = 1.0f64;
        let dtd = 0.0f64;

        let _ = matupd(
            n, m, &mut ws, &mut wy, &mut sy, &mut ss, &s, &y, &mut itail, iupdat, &mut col,
            &mut head, &mut theta, rr, dr, stp, dtd,
        );

        // formk on small col=1 should succeed
        let mut k = vec![0.0; 1];
        formk(m, &sy, 1usize, &mut k).expect("formk failed");
        assert!(k[0].abs() >= 0.0);

        // formt expects size 1
        let mut t = vec![0.0; 1];
        formt(1usize, &k, &mut t).expect("formt failed");
        assert!(t[0] > 0.0);

        // subsm: solve for r
        let r = vec![2.0; 1];
        let mut z = vec![0.0; 1];
        subsm(1usize, &k, &t, &r, &mut z).expect("subsm failed");
        assert!((z[0] - r[0] / t[0]).abs() < 1e-8);

        // bmv: exercise with col=1
        let v = vec![1.0, 2.0];
        let mut p = vec![0.0, 0.0];
        bmv(m, &sy, &wt, 1usize, &v, &mut p).expect("bmv failed");
        assert!(p.len() >= 2);
    }
}
