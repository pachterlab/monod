//! Smooth test problems with known solutions.  Besides asserting the answer,
//! they bound the iteration count: a working quasi-Newton update solves these
//! in tens of iterations, projected steepest descent needs hundreds.
use lbfgsb_rs_pure::{IterationControl, LBFGSB};

fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    let mut f = 0.0;
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        f += 100.0 * a * a + b * b;
        g[i] += -400.0 * x[i] * a - 2.0 * b;
        g[i + 1] += 200.0 * a;
    }
    (f, g)
}

/// Ill-conditioned quadratic: f = sum_i c_i (x_i - 1)^2 with c_i spanning 1..1e4.
fn quadratic(x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    let c: Vec<f64> = (0..n).map(|i| 10f64.powf(4.0 * i as f64 / (n - 1) as f64)).collect();
    let f = x.iter().zip(&c).map(|(xi, ci)| ci * (xi - 1.0) * (xi - 1.0)).sum();
    let g = x.iter().zip(&c).map(|(xi, ci)| 2.0 * ci * (xi - 1.0)).collect();
    (f, g)
}

fn run<F: FnMut(&[f64]) -> (f64, Vec<f64>)>(
    mut f: F, x0: &[f64], lb: &[f64], ub: &[f64],
) -> (Vec<f64>, f64, usize, usize) {
    let mut x = x0.to_vec();
    let mut nfev = 0usize;
    let mut fg = |x: &[f64]| { nfev += 1; f(x) };
    let mut cb = |_: &lbfgsb_rs_pure::IterationInfo, _: &[f64]| IterationControl::Continue;
    let sol = LBFGSB::new(10).with_max_iter(2000).with_pgtol(1e-8).with_ftol(0.0)
        .with_verbose(std::env::var_os("LBFGSB_TEST_VERBOSE").is_some())
        .minimize_with_callback(&mut x, lb, ub, &mut fg, &mut cb)
        .expect("solver error");
    let its = sol.iterations;
    (sol.x, sol.f, its, nfev)
}

#[test]
fn rosenbrock_5d_unconstrained() {
    let n = 5;
    let x0: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let (x, f, its, nfev) = run(rosenbrock, &x0, &vec![-10.0; n], &vec![10.0; n]);
    eprintln!("rosenbrock 5d: f={f:.3e} its={its} nfev={nfev} x={x:?}");
    assert!(f < 1e-10, "f = {f}");
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-4));
    assert!(its < 100, "took {its} iterations (scipy: 44)");
}

#[test]
fn rosenbrock_5d_bounded_active() {
    // Upper bound 0.8 on every coordinate: the optimum sits on the bound.
    let n = 5;
    let x0 = vec![0.0; n];
    let (x, f, its, nfev) = run(rosenbrock, &x0, &vec![-2.0; n], &vec![0.8; n]);
    eprintln!("rosenbrock 5d bounded: f={f:.6} its={its} nfev={nfev} x={x:?}");
    // scipy L-BFGS-B reference from the same start: f = 2.2345 (approx), x[0..] on bounds
    let (_, f_ref, _, _) = run(rosenbrock, &x, &vec![-2.0; n], &vec![0.8; n]); // restart: must not improve
    assert!((f - f_ref).abs() < 1e-8, "restart improved {f} -> {f_ref}");
    assert!((f - 1.148224).abs() < 1e-5, "f = {f}, scipy: 1.148224");
    assert!(its < 60, "took {its} iterations (scipy: 20)");
}

#[test]
fn quadratic_10d_illconditioned() {
    let n = 10;
    let x0 = vec![0.0; n];
    let (x, f, its, nfev) = run(quadratic, &x0, &vec![-5.0; n], &vec![5.0; n]);
    eprintln!("quadratic 10d: f={f:.3e} its={its} nfev={nfev}");
    assert!(f < 1e-8, "f = {f}");
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-4));
    // scipy's L-BFGS-B needs 176 iterations here at gtol=1e-8 (condition number 1e4)
    assert!(its < 250, "took {its} iterations (scipy: 176)");
}

#[test]
fn quadratic_10d_bounded() {
    let n = 10;
    let x0 = vec![0.0; n];
    let lb = vec![-5.0; n];
    let ub: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.5 } else { 5.0 }).collect();
    let (x, f, its, nfev) = run(quadratic, &x0, &lb, &ub);
    eprintln!("quadratic 10d bounded: f={f:.6} its={its} nfev={nfev}");
    for i in 0..n {
        let target = if i % 3 == 0 { 0.5 } else { 1.0 };
        assert!((x[i] - target).abs() < 1e-4, "x[{i}] = {} expected {target}", x[i]);
    }
    assert!(its < 60, "took {its} iterations");
}
