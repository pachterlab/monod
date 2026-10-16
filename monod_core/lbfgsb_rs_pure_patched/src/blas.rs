//! Minimal, safe, and idiomatic level-1 BLAS routines used by this crate.
//!
//! This module provides simple Rust implementations of a small subset of the
//! BLAS level-1 functions used by the L-BFGS-B algorithm port:
//! - `ddot`: dot product
//! - `daxpy`: y := y + a*x
//! - `dcopy`: y := x
//! - `dscal`: x := a*x
//!
//! These implementations are intentionally simple and portable (no `unsafe`,
//! no platform-specific intrinsics). They use the element-wise operations on
//! Rust slices and silently operate on the length equal to the minimum of the
//! input slice lengths. This mirrors the defensive use in the rest of this
//! repository where slices are often sized to the problem dimension.
//!
//! Note: These are *not* optimized for high-performance numerical computing.
//! For production workloads you should link a tuned BLAS implementation (BLIS,
//! OpenBLAS, Intel MKL, ...). These helpers are sufficient for correctness and
//! unit testing of the algorithmic components.

/// Compute the dot product of `x` and `y`.
///
/// The result is sum_i x[i] * y[i] for i in 0..min(x.len(), y.len()).
///
/// Safe: will not read out-of-bounds; works with empty slices.
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

/// Compute `y := y + a * x` in-place.
///
/// Operates for i in 0..min(x.len(), y.len()). If `a == 0.0`, the call is a no-op.
pub fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    if a == 0.0 {
        return;
    }
    for i in 0..n {
        y[i] += a * x[i];
    }
}

/// Copy vector `x` into `y` (y := x).
///
/// Copies up to min(x.len(), y.len()) elements.
pub fn dcopy(x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    y[..n].copy_from_slice(&x[..n]);
}

/// Scale vector `x` by scalar `a` in-place (x := a * x).
///
/// If `a == 1.0` this is a no-op; if `a == 0.0`, the result is a zero vector.
pub fn dscal(a: f64, x: &mut [f64]) {
    if a == 1.0 {
        return;
    }
    for xi in x.iter_mut() {
        *xi *= a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddot_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, -1.0, 2.0];
        let res = ddot(&a, &b);
        // 1*4 + 2*(-1) + 3*2 = 4 -2 + 6 = 8
        assert_eq!(res, 8.0);
    }

    #[test]
    fn test_daxpy_and_dcopy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [0.5, -1.0, 2.0, 0.0];
        daxpy(2.0, &x, &mut y);
        // y = y + 2*x = [0.5+2, -1+4, 2+6, 0+8] = [2.5, 3.0, 8.0, 8.0]
        assert_eq!(y, [2.5, 3.0, 8.0, 8.0]);

        let mut z = [0.0; 4];
        dcopy(&y, &mut z);
        assert_eq!(z, y);
    }

    #[test]
    fn test_dscal_basic() {
        let mut v = [1.0, -2.0, 3.5];
        dscal(2.0, &mut v);
        assert_eq!(v, [2.0, -4.0, 7.0]);
        dscal(0.0, &mut v);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_shorter_lengths() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0, 0.0];
        dcopy(&x, &mut y);
        assert_eq!(y, [1.0, 2.0]);

        let res = ddot(&x, &y);
        assert_eq!(res, 1.0 * 1.0 + 2.0 * 2.0);
    }
}
