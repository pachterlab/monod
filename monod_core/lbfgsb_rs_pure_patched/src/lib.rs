//! A compact, self-contained reimplementation of an L-BFGS-B style solver in Rust.
//!
//! This crate focuses on a safe, idiomatic Rust port of the algorithmic core of
//! the classical L-BFGS-B package. It aims for algorithmic parity and
//! reproducibility rather than maximal performance — for production use link a
//! tuned BLAS/LAPACK provider.
//!
//! The main public API is the `LBFGSB` type (in `solver`), together with the
//! `Status` and `Solution` types returned by the solver.
//!
//! Example
//! ```/dev/null/example.rs#L1-22
//! use lbfgsb_rs::{LBFGSB, Status};
//!
//! let mut x = vec![0.0];
//! let lower = vec![-1.0];
//! let upper = vec![1.0];
//!
//! let mut solver = LBFGSB::new(5);
//! let sol = solver
//!     .minimize(&mut x, &lower, &upper, |x| {
//!         let dx = x[0] - 0.5;
//!         (dx * dx, vec![2.0 * dx])
//!     })
//!     .unwrap();
//!
//! assert_eq!(sol.status, Status::Converged);
//! assert!((sol.x[0] - 0.5).abs() < 1e-6);
//! ```
//!
//! Modules
//! - `blas`         : lightweight level-1 BLAS helpers used throughout the crate.
//! - `linesearch`   : line-search helpers (dcstep / lnsrlb / convenience methods).
//! - `linpack`      : small linpack routines (Cholesky / triangular solve).
//! - `subalgorithms`: helper routines (active-set initialization, bmv partial port).
//! - `solver`       : high-level LBFGS-B driver (uses the above modules).
//!
//! The crate re-exports `LBFGSB` from `solver` and provides the `Status` and
//! `Solution` types here so `solver` can reference them as `crate::Status` and
//! `crate::Solution` (matching the original design).

#![allow(clippy::too_many_arguments)]

use std::f64;

mod blas;
mod linesearch;
pub mod linpack;
pub mod solver;
pub mod subalgorithms;

pub use crate::blas::*;
pub use crate::linesearch::*;
pub use crate::solver::{IterationControl, IterationInfo, LBFGSB};

/// Maximum number of backtracking line search iterations used by convenience routines.
pub const MAX_BACKTRACK: usize = 50;

/// Minimum curvature s'y to accept BFGS update.
pub const CURVATURE_EPS: f64 = 1e-12;

/// Enum describing solver termination.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Status {
    Converged,
    MaxIter,
    LineSearchFailure,
    NumericalFailure,
}

/// Solution returned by the solver.
#[derive(Debug, Clone)]
pub struct Solution {
    pub x: Vec<f64>,
    pub f: f64,
    pub iterations: usize,
    pub status: Status,
}
