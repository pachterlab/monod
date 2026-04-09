/*
linpack.rs

Safe Rust ports of selected LINPACK routines used by L-BFGS-B:
- dpofa: Cholesky (LL^T) factorization for symmetric positive-definite matrices.
- dtrsl: solve triangular systems using the factor produced by dpofa.

The implementations assume matrices are stored in row-major order in a flat
slice `a` of length `n * n`. After a successful `dpofa`, the lower-triangular
factor L is stored in the lower triangle of `a` (including the diagonal).
Upper-triangle elements are set to zero for cleanliness.

These routines are intentionally straightforward and safe (no `unsafe` code).
They are suitable for correctness and unit-testing purposes. For high-performance
use, link an optimized BLAS/LAPACK library or use a specialized crate.
*/

/// Compute Cholesky factorization of a symmetric positive definite matrix A.
///
/// The input `a` is an n-by-n matrix stored in row-major order (length n*n).
/// On success, the lower-triangular Cholesky factor L is stored in the lower
/// triangle of `a` (including diagonal). Upper-triangle entries are set to 0.0.
/// Returns Ok(()) on success. If a non-positive pivot is encountered, returns
/// Err(k) where k is the (1-based) index of the leading minor that is not
/// positive definite (matching the linpack convention).
pub fn dpofa(a: &mut [f64], n: usize) -> Result<(), usize> {
    if a.len() < n * n {
        return Err(1); // insufficient length; treat as failure at first pivot
    }

    // Cholesky (column by column / row-major loop over j then i >= j)
    for j in 0..n {
        for i in j..n {
            // compute sum = a[i,j] - sum_{k=0..j-1} a[i,k] * a[j,k]
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }

            if i == j {
                // diagonal element
                if sum <= 0.0 {
                    // leading minor not positive definite
                    return Err(j + 1); // 1-based
                }
                a[j * n + j] = sum.sqrt();
            } else {
                // sub-diagonal element: L[i,j] = sum / L[j,j]
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    // Zero upper triangle (for cleanliness)
    for i in 0..n {
        for j in (i + 1)..n {
            a[i * n + j] = 0.0;
        }
    }

    Ok(())
}

/// Solve triangular systems using the factor produced by `dpofa`.
///
/// - `a` holds the triangular factor in row-major layout. We assume `a` stores
///   the lower-triangular factor L in its lower triangle (as produced by dpofa).
/// - If `lower` is true, `a` is treated as lower-triangular; otherwise as
///   upper-triangular.
/// - If `trans` is false, solve A * x = b; if true, solve A^T * x = b.
/// - The vector `b` is overwritten with the solution x.
///   Returns Ok(()) on success, or Err(()) if a zero diagonal is encountered.
pub fn dtrsl(a: &[f64], n: usize, lower: bool, trans: bool, b: &mut [f64]) -> Result<(), String> {
    if a.len() < n * n || b.len() < n {
        return Err("Invalid input dimensions".to_string());
    }

    if lower {
        if !trans {
            // Solve L x = b (forward substitution)
            for i in 0..n {
                let mut sum = b[i];
                for j in 0..i {
                    sum -= a[i * n + j] * b[j];
                }
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err("Zero diagonal encountered".to_string());
                }
                b[i] = sum / diag;
            }
            Ok(())
        } else {
            // Solve L^T x = b (back substitution)
            for i in (0..n).rev() {
                let mut sum = b[i];
                for j in (i + 1)..n {
                    // note: L^T has (i,j) element equal to a[j,i]
                    sum -= a[j * n + i] * b[j];
                }
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err("Zero diagonal encountered".to_string());
                }
                b[i] = sum / diag;
            }
            Ok(())
        }
    } else {
        // Upper-triangular case
        if !trans {
            // Solve U x = b (forward substitution if U is stored in row-major upper)
            for i in 0..n {
                let mut sum = b[i];
                for j in 0..i {
                    sum -= a[i * n + j] * b[j];
                }
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err("Zero diagonal encountered".to_string());
                }
                b[i] = sum / diag;
            }
            Ok(())
        } else {
            // Solve U^T x = b (back substitution)
            for i in (0..n).rev() {
                let mut sum = b[i];
                for j in (i + 1)..n {
                    sum -= a[j * n + i] * b[j];
                }
                let diag = a[i * n + i];
                if diag == 0.0 {
                    return Err("Zero diagonal encountered".to_string());
                }
                b[i] = sum / diag;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to compare matrices with tolerance
    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_dpofa_and_dtrsl_small() {
        // Example SPD matrix (3x3):
        // A = [ 4 12 -16
        //       12 37 -43
        //      -16 -43 98 ]
        // Known Cholesky L:
        // L = [ 2  0  0
        //       6  1  0
        //      -8  5  3 ]
        let mut a = vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        let n = 3usize;
        dpofa(&mut a, n).expect("dpofa should succeed");
        // expected lower-triangular stored in row-major layout:
        let expected_l = vec![2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0];
        assert!(approx_eq(&a, &expected_l, 1e-12));

        // Test triangular solve: given L and some x_true, compute b = L * x_true,
        // then solve L * x = b and verify we recover x_true.
        let x_true = [1.5, -2.0, 0.75];
        let mut b = vec![0.0; n];
        // compute b = L * x_true
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += a[i * n + j] * x_true[j];
            }
            b[i] = sum;
        }
        // solve L x = b using dtrsl (lower, not transposed)
        let mut b_copy = b.clone();
        dtrsl(&a, n, true, false, &mut b_copy).expect("dtrsl forward should succeed");
        for i in 0..n {
            assert!((b_copy[i] - x_true[i]).abs() < 1e-12);
        }

        // Solve L^T y = b2 where b2 = L^T * y_true and recover y_true
        let y_true = [0.25, -1.0, 2.0];
        let mut b2 = vec![0.0; n];
        // compute b2 = L^T * y_true
        for i in 0..n {
            let mut sum = 0.0;
            for j in i..n {
                // L^T[i,j] = L[j,i]
                sum += a[j * n + i] * y_true[j];
            }
            b2[i] = sum;
        }
        let mut b2_copy = b2.clone();
        dtrsl(&a, n, true, true, &mut b2_copy).expect("dtrsl transpose should succeed");
        for i in 0..n {
            assert!((b2_copy[i] - y_true[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dpofa_nonspd() {
        // non-SPD matrix (zero diagonal)
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        let res = dpofa(&mut a, 2);
        assert!(res.is_err());
    }
}
