//! The interior point method against SDPs with analytically known solutions.

mod common;

use common::{cycle, triangle};
use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::sdp::linalg::DenseMatrix;
use maxcut_sdp::sdp::solver::{Solver, SolverConfig};
use maxcut_sdp::sdp::{ConstraintMatrix, SdpProblem, SdpSolution};

fn solve(problem: &SdpProblem) -> SdpSolution {
    Solver::new(SolverConfig::default())
        .solve(problem)
        .unwrap_or_else(|e| panic!("solver failed: {e}"))
}

fn maxcut_relaxation(graph: &Graph) -> SdpProblem {
    SdpProblem::maxcut_relaxation(graph).unwrap_or_else(|e| panic!("relaxation failed: {e}"))
}

/// Solve the 2×2 SDP from MATH.md §4:
/// min `x_11` + `x_22`  s.t.  `x_12` = 1, X ≽ 0
/// Optimal value = 2, X* = [[1,1],[1,1]]
#[test]
fn sdp_2x2_worked_example() {
    let a1 = DenseMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0])
        .unwrap_or_else(|e| panic!("from_vec failed: {e}"));
    let mut problem = SdpProblem::new(DenseMatrix::identity(2));
    // tr(A1 X) = 2*x_12 = 2, so x_12 = 1
    problem
        .add_constraint(ConstraintMatrix::Dense(a1), 2.0)
        .unwrap_or_else(|e| panic!("add_constraint failed: {e}"));

    let solution = solve(&problem);

    assert!((solution.primal_obj - 2.0).abs() < 1e-4);
    assert!(solution.duality_gap < 1e-6);
}

/// The triangle's SDP value is `9/4`: three unit vectors at 120°.
#[test]
fn ipm_reaches_the_sdp_value_of_the_triangle() {
    let solution = solve(&maxcut_relaxation(&triangle(1.0)));

    let sdp_value = -solution.primal_obj;
    assert!((sdp_value - 2.25).abs() < 1e-6, "SDP value {sdp_value}");
}

#[test]
fn the_maxcut_solution_is_feasible() {
    let solution = solve(&maxcut_relaxation(&cycle(4)));

    for i in 0..4 {
        assert!((solution.x.get(i, i) - 1.0).abs() < 1e-4);
    }
    let eigenvalues = solution
        .x
        .eigenvalues_symmetric()
        .unwrap_or_else(|e| panic!("eigenvalues failed: {e}"));
    assert!(eigenvalues.iter().all(|&e| e > -1e-6), "{eigenvalues:?}");
}

/// The Max-Cut SDP value of the odd cycle `C_n` is `(n/2)(1 + cos(π/n))`.
#[test]
fn ipm_reaches_the_sdp_value_of_a_61_cycle() {
    let n = 61;
    let expected = (n as f64 / 2.0) * (1.0 + (std::f64::consts::PI / n as f64).cos());

    let solution = solve(&maxcut_relaxation(&cycle(n)));

    let bound = -solution.dual_obj;
    assert!(
        (bound - expected).abs() / expected < 1e-6,
        "bound {bound}, expected {expected}"
    );
}

#[test]
fn stopping_before_convergence_is_an_error() {
    let problem = maxcut_relaxation(&cycle(5));
    let solver = Solver::new(SolverConfig {
        max_iterations: 1,
        ..SolverConfig::default()
    });

    let result = solver.solve(&problem);

    assert!(matches!(result, Err(GwError::SolverFailed(_))));
}
