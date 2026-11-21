//! Integration tests for the Burer-Monteiro Max-Cut solver.

mod common;

use common::{complete, cycle};
use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::rounding::Rounding;
use maxcut_sdp::maxcut::{GwBmConfig, gw_maxcut_bm};
use maxcut_sdp::sdp::bm::{BmSolution, BmSolver, BmSolverConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::num::NonZeroUsize;

/// Full GW pipeline with BM on C5: cut >= `α_GW` · OPT.
#[test]
fn bm_five_cycle_gw_guarantee() {
    let g = cycle(5);
    let config = GwBmConfig {
        num_trials: NonZeroUsize::new(100).unwrap_or_else(|| panic!("100 trials")),
        ils_rounds: 5,
        rounding: Rounding::Hyperplane,
        solver_config: BmSolverConfig {
            rank: Some(4),
            ..BmSolverConfig::default()
        },
        num_restarts: NonZeroUsize::MIN,
        seed: Some(42),
        time_limit: None,
    };

    let result =
        gw_maxcut_bm(&g, &config, |_| {}).unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));

    // GW guarantee: cut >= α_GW · OPT ≈ 0.878 · 4 ≈ 3.51.
    let alpha_gw: f64 = 0.878;
    assert!(
        result.cut_value >= alpha_gw.mul_add(4.0, -0.01),
        "cut = {}, expected >= {}",
        result.cut_value,
        alpha_gw.mul_add(4.0, -0.01)
    );
}

/// BM on K4: cut >= `α_GW` · 4.
#[test]
fn bm_k4_gw_guarantee() {
    let g = complete(4);
    let config = GwBmConfig {
        num_trials: NonZeroUsize::new(100).unwrap_or_else(|| panic!("100 trials")),
        ils_rounds: 5,
        rounding: Rounding::Hyperplane,
        solver_config: BmSolverConfig {
            rank: Some(3),
            ..BmSolverConfig::default()
        },
        num_restarts: NonZeroUsize::MIN,
        seed: Some(7),
        time_limit: None,
    };

    let result =
        gw_maxcut_bm(&g, &config, |_| {}).unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));

    let alpha_gw: f64 = 0.878;
    assert!(
        result.cut_value >= alpha_gw.mul_add(4.0, -0.01),
        "cut = {}, expected >= {}",
        result.cut_value,
        alpha_gw.mul_add(4.0, -0.01)
    );
}

#[test]
fn bm_refuses_the_empty_graph() {
    let result = gw_maxcut_bm(&Graph::new(0), &GwBmConfig::default(), |_| {});

    assert!(matches!(result, Err(GwError::InvalidGraph(_))));
}

fn bm_with_rank(graph: &Graph, rank: usize) -> Result<BmSolution, GwError> {
    let config = BmSolverConfig {
        rank: Some(rank),
        ..BmSolverConfig::default()
    };
    BmSolver::new(config).solve_maxcut(graph, &mut StdRng::seed_from_u64(1), |_| {})
}

#[test]
fn bm_factors_at_the_rank_asked_for() {
    let solution = bm_with_rank(&complete(4), 4).unwrap_or_else(|e| panic!("{e}"));

    assert_eq!(solution.vectors.rows(), 4);
}

#[test]
fn bm_refuses_a_rank_outside_two_to_n() {
    for rank in [1, 5] {
        let result = bm_with_rank(&complete(4), rank);

        assert!(
            matches!(result, Err(GwError::InvalidConfig(_))),
            "rank {rank}: {:?}",
            result.map(|solution| solution.vectors.rows())
        );
    }
}
