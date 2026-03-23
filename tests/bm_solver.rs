//! Integration tests for the Burer-Monteiro Max-Cut solver.

use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::{GwBmConfig, GwConfig, gw_maxcut, gw_maxcut_bm};
use maxcut_sdp::sdp::solver_bm::BmSolverConfig;

/// Helper: build the 5-cycle C_5 with unit weights (OPT = 4).
fn five_cycle() -> Graph {
    let mut g = Graph::new(5);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    let _ = g.add_edge(3, 4, 1.0);
    let _ = g.add_edge(4, 0, 1.0);
    g
}

/// Helper: build K_4 with unit weights (OPT = 4).
fn k4() -> Graph {
    let mut g = Graph::new(4);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(0, 2, 1.0);
    let _ = g.add_edge(0, 3, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(1, 3, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    g
}

/// Full GW pipeline with BM on C5: cut >= α_GW · OPT.
#[test]
fn bm_five_cycle_gw_guarantee() {
    let g = five_cycle();
    let config = GwBmConfig {
        num_trials: 100,
        ils_rounds: 5,
        solver_config: BmSolverConfig {
            rank: Some(4),
            ..BmSolverConfig::default()
        },
        seed: Some(42),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("gw_maxcut_bm failed"));

    // GW guarantee: cut >= α_GW · OPT ≈ 0.878 · 4 ≈ 3.51.
    let alpha_gw = 0.878;
    assert!(
        result.cut_value >= alpha_gw * 4.0 - 0.01,
        "cut = {}, expected >= {}",
        result.cut_value,
        alpha_gw * 4.0 - 0.01
    );
}

/// BM on K4: cut >= α_GW · 4.
#[test]
fn bm_k4_gw_guarantee() {
    let g = k4();
    let config = GwBmConfig {
        num_trials: 100,
        ils_rounds: 5,
        solver_config: BmSolverConfig {
            rank: Some(3),
            ..BmSolverConfig::default()
        },
        seed: Some(7),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("gw_maxcut_bm failed"));

    let alpha_gw = 0.878;
    assert!(
        result.cut_value >= alpha_gw * 4.0 - 0.01,
        "cut = {}, expected >= {}",
        result.cut_value,
        alpha_gw * 4.0 - 0.01
    );
}

/// BM SDP bound should be close to IPM SDP bound on small graphs.
#[test]
fn bm_bound_close_to_ipm() {
    let g = five_cycle();

    let ipm_config = GwConfig {
        num_trials: 10,
        seed: Some(42),
        ..GwConfig::default()
    };
    let ipm_result = gw_maxcut(&g, &ipm_config).unwrap_or_else(|_| panic!("ipm failed"));

    let bm_config = GwBmConfig {
        num_trials: 10,
        ils_rounds: 5,
        solver_config: BmSolverConfig {
            rank: Some(4),
            ..BmSolverConfig::default()
        },
        seed: Some(42),
        time_limit: None,
    };
    let bm_result = gw_maxcut_bm(&g, &bm_config).unwrap_or_else(|_| panic!("bm failed"));

    // SDP bounds should be within 1%.
    let rel_diff = (ipm_result.sdp_bound - bm_result.sdp_bound).abs() / ipm_result.sdp_bound;
    assert!(
        rel_diff < 0.01,
        "IPM bound = {}, BM bound = {}, relative diff = {rel_diff}",
        ipm_result.sdp_bound,
        bm_result.sdp_bound
    );
}

/// Rank override should work (use rank = n which is the maximum).
#[test]
fn bm_rank_override() {
    let g = k4();
    let config = GwBmConfig {
        num_trials: 50,
        ils_rounds: 5,
        solver_config: BmSolverConfig {
            rank: Some(4), // full rank = n
            ..BmSolverConfig::default()
        },
        seed: Some(99),
        time_limit: None,
    };

    let result = gw_maxcut_bm(&g, &config);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("gw_maxcut_bm failed"));

    // Should still produce a valid cut.
    assert!(result.cut_value >= 3.0, "cut = {}", result.cut_value);
}
