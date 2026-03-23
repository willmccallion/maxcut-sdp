//! Tests for SDP solver correctness on known instances.
//!
//! Verifies Cholesky, eigendecomposition, and full SDP solves against
//! analytically known solutions.

use maxcut_sdp::graph::Graph;
use maxcut_sdp::sdp::linalg::DenseMatrix;
use maxcut_sdp::sdp::solver::{Solver, SolverConfig};
use maxcut_sdp::sdp::SdpProblem;

/// Cholesky of a known 2×2 SPD matrix.
#[test]
fn cholesky_2x2() {
    // A = [[4, 2], [2, 3]]
    // L = [[2, 0], [1, √2]]
    let a = DenseMatrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
    assert!(a.is_ok());
    let a = a.unwrap_or_else(|_| panic!("from_vec failed"));

    let l = a.cholesky();
    assert!(l.is_ok());
    let l = l.unwrap_or_else(|_| panic!("cholesky failed"));

    assert!((l.get(0, 0) - 2.0).abs() < 1e-10);
    assert!((l.get(1, 0) - 1.0).abs() < 1e-10);
    assert!((l.get(0, 1)).abs() < 1e-10);
    assert!((l.get(1, 1) - f64::sqrt(2.0)).abs() < 1e-10);
}

/// Cholesky should fail on a non-PD matrix.
#[test]
fn cholesky_non_pd() {
    // A = [[1, 2], [2, 1]] has eigenvalues -1 and 3
    let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]);
    assert!(a.is_ok());
    let a = a.unwrap_or_else(|_| panic!("from_vec failed"));
    assert!(a.cholesky().is_err());
}

/// Eigenvalues of a known 2×2 symmetric matrix.
#[test]
fn eigenvalues_2x2() {
    // A = [[2, 1], [1, 2]] has eigenvalues 1 and 3
    let a = DenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
    assert!(a.is_ok());
    let a = a.unwrap_or_else(|_| panic!("from_vec failed"));

    let eigs = a.eigenvalues_symmetric();
    assert!(eigs.is_ok());
    let mut eigs = eigs.unwrap_or_else(|_| panic!("eigenvalues failed"));
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or_else(|| panic!("NaN eigenvalue")));

    assert!((eigs[0] - 1.0).abs() < 1e-10);
    assert!((eigs[1] - 3.0).abs() < 1e-10);
}

/// Identity matrix properties.
#[test]
fn identity_properties() {
    let id = DenseMatrix::identity(3);

    assert!((id.trace() - 3.0).abs() < 1e-10);
    assert!(id.is_positive_definite());

    let det = id.log_det();
    assert!(det.is_ok());
    assert!((det.unwrap_or_else(|_| panic!("log_det failed"))).abs() < 1e-10);
}

/// Solve the 2×2 SDP from MATH.md §4:
/// min x_11 + x_22  s.t.  x_12 = 1, X ≽ 0
/// Optimal value = 2, X* = [[1,1],[1,1]]
#[test]
fn sdp_2x2_worked_example() {
    let c = DenseMatrix::identity(2);

    let a1 = DenseMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    assert!(a1.is_ok());

    let mut problem = SdpProblem::new(c);
    problem.add_constraint(
        a1.unwrap_or_else(|_| panic!("from_vec failed")),
        2.0, // tr(A1 X) = 2*x_12 = 2, so x_12 = 1
    );

    let solver = Solver::new(SolverConfig::default());
    let solution = solver.solve(&problem);
    let solution = solution.unwrap_or_else(|e| panic!("solver failed: {e}"));
    assert!((solution.primal_obj - 2.0).abs() < 1e-4);
    assert!(solution.duality_gap < 1e-6);
}

/// Max-Cut SDP on the triangle: SDP_OPT ≥ OPT = 2.
#[test]
fn maxcut_sdp_triangle() {
    let mut g = Graph::new(3);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(0, 2, 1.0);

    let problem = SdpProblem::maxcut_relaxation(&g);
    assert!(problem.is_ok());
    let problem = problem.unwrap_or_else(|_| panic!("maxcut_relaxation failed"));

    let solver = Solver::new(SolverConfig::default());
    let solution = solver.solve(&problem);
    assert!(solution.is_ok());

    let solution = solution.unwrap_or_else(|_| panic!("solver failed"));

    // SDP optimal for Max-Cut is a maximization: the negated primal objective
    // should give us SDP_OPT ≥ 2.0
    let sdp_opt = -solution.primal_obj;
    assert!(sdp_opt >= 2.0 - 1e-4);
}

/// Verify primal feasibility of the Max-Cut SDP solution.
#[test]
fn maxcut_sdp_feasibility() {
    let mut g = Graph::new(4);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    let _ = g.add_edge(3, 0, 1.0);

    let problem = SdpProblem::maxcut_relaxation(&g);
    assert!(problem.is_ok());
    let problem = problem.unwrap_or_else(|_| panic!("maxcut_relaxation failed"));

    let solver = Solver::new(SolverConfig::default());
    let solution = solver.solve(&problem);
    assert!(solution.is_ok());
    let solution = solution.unwrap_or_else(|_| panic!("solver failed"));

    // Check diagonal constraints: X_ii ≈ 1
    for i in 0..4 {
        assert!((solution.x.get(i, i) - 1.0).abs() < 1e-4);
    }

    // Check PSD: all eigenvalues ≥ 0
    let eigs = solution.x.eigenvalues_symmetric();
    assert!(eigs.is_ok());
    let eigs = eigs.unwrap_or_else(|_| panic!("eigenvalues failed"));
    for &e in &eigs {
        assert!(e > -1e-6);
    }
}
