//! Integration tests for the Burer-Monteiro Max-Cut solver.

mod common;

use common::{complete, cycle};
use maxcut_sdp::maxcut::rounding::Rounding;
use maxcut_sdp::maxcut::{GwBmConfig, GwConfig, gw_maxcut, gw_maxcut_bm};
use maxcut_sdp::sdp::solver_bm::BmSolverConfig;
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

/// BM SDP bound should be close to IPM SDP bound on small graphs.
#[test]
fn bm_bound_close_to_ipm() {
    let g = cycle(5);

    let ipm_config = GwConfig {
        num_trials: NonZeroUsize::new(10).unwrap_or_else(|| panic!("10 trials")),
        seed: Some(42),
        ..GwConfig::default()
    };
    let ipm_result = gw_maxcut(&g, &ipm_config).unwrap_or_else(|_| panic!("ipm failed"));

    let bm_config = GwBmConfig {
        num_trials: NonZeroUsize::new(10).unwrap_or_else(|| panic!("10 trials")),
        ils_rounds: 5,
        rounding: Rounding::Hyperplane,
        solver_config: BmSolverConfig {
            rank: Some(4),
            ..BmSolverConfig::default()
        },
        seed: Some(42),
        time_limit: None,
    };
    let bm_result = gw_maxcut_bm(&g, &bm_config).unwrap_or_else(|_| panic!("bm failed"));

    // SDP bounds should be within 1%.
    let (ipm_bound, bm_bound) = (ipm_result.sdp_bound.value(), bm_result.sdp_bound.value());
    let rel_diff = (ipm_bound - bm_bound).abs() / ipm_bound;
    assert!(
        rel_diff < 0.01,
        "IPM bound = {ipm_bound}, BM bound = {bm_bound}, relative diff = {rel_diff}"
    );
}

/// Rank override should work (use rank = n which is the maximum).
#[test]
fn bm_rank_override() {
    let g = complete(4);
    let config = GwBmConfig {
        num_trials: NonZeroUsize::new(50).unwrap_or_else(|| panic!("50 trials")),
        ils_rounds: 5,
        rounding: Rounding::Hyperplane,
        solver_config: BmSolverConfig {
            rank: Some(4), // full rank = n
            ..BmSolverConfig::default()
        },
        seed: Some(99),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config).unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));

    // Should still produce a valid cut.
    assert!(result.cut_value >= 3.0, "cut = {}", result.cut_value);
}
