//! Integration tests for the Burer-Monteiro Max-Cut solver.

mod common;

use common::{complete, cycle};
use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::rounding::Rounding;
use maxcut_sdp::maxcut::{GwBmConfig, gw_maxcut_bm};
use maxcut_sdp::sdp::bm::BmSolverConfig;
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
        seed: Some(42),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config).unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));

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
        seed: Some(7),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config).unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));

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
    let result = gw_maxcut_bm(&Graph::new(0), &GwBmConfig::default());

    assert!(matches!(result, Err(GwError::InvalidGraph(_))));
}
