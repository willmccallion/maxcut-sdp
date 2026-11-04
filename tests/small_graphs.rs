//! Integration tests for Max-Cut on small graphs with known optimal values.
//!
//! These tests verify both the exact brute-force solver and the GW approximation
//! against hand-computed max-cut values.

use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::exact::exact_maxcut;
use maxcut_sdp::maxcut::{GwConfig, gw_maxcut};
use std::num::NonZeroUsize;

/// Triangle (`K_3`) with unit weights: OPT = 2 (cut any single vertex off).
#[test]
fn triangle_exact() {
    let mut g = Graph::new(3);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(0, 2, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));
    assert!((result.cut_value - 2.0).abs() < 1e-10);
}

/// 4-cycle (`C_4`) with unit weights: OPT = 4 (bipartite, all edges cut).
#[test]
fn four_cycle_exact() {
    let mut g = Graph::new(4);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    let _ = g.add_edge(3, 0, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));
    assert!((result.cut_value - 4.0).abs() < 1e-10);
}

/// 5-cycle (`C_5`) with unit weights: OPT = 4 (odd cycle, one edge uncut).
#[test]
fn five_cycle_exact() {
    let mut g = Graph::new(5);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    let _ = g.add_edge(3, 4, 1.0);
    let _ = g.add_edge(4, 0, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));
    assert!((result.cut_value - 4.0).abs() < 1e-10);
}

/// Complete graph `K_4` with unit weights: OPT = 4.
#[test]
fn k4_exact() {
    let mut g = Graph::new(4);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(0, 2, 1.0);
    let _ = g.add_edge(0, 3, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(1, 3, 1.0);
    let _ = g.add_edge(2, 3, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));
    assert!((result.cut_value - 4.0).abs() < 1e-10);
}

/// Single edge: OPT = 1.
#[test]
fn single_edge_exact() {
    let mut g = Graph::new(2);
    let _ = g.add_edge(0, 1, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));
    assert!((result.cut_value - 1.0).abs() < 1e-10);
}

/// GW on the 5-cycle should achieve at least `α_GW` · OPT ≈ 0.878 · 4 ≈ 3.51.
#[test]
fn five_cycle_gw() {
    let mut g = Graph::new(5);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(2, 3, 1.0);
    let _ = g.add_edge(3, 4, 1.0);
    let _ = g.add_edge(4, 0, 1.0);

    let config = GwConfig {
        num_trials: NonZeroUsize::new(100).unwrap_or_else(|| panic!("100 trials")),
        seed: Some(42),
        ..GwConfig::default()
    };

    let result = gw_maxcut(&g, &config);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("gw_maxcut failed"));

    // GW guarantee: cut >= α_GW * OPT
    let alpha_gw: f64 = 0.878;
    assert!(result.cut_value >= alpha_gw.mul_add(4.0, -0.01));
}

/// Verify that `Graph::cut_value` agrees with the partition from exact solver.
#[test]
fn cut_value_consistency() {
    let mut g = Graph::new(3);
    let _ = g.add_edge(0, 1, 1.0);
    let _ = g.add_edge(1, 2, 1.0);
    let _ = g.add_edge(0, 2, 1.0);

    let result = exact_maxcut(&g);
    assert!(result.is_ok());
    let result = result.unwrap_or_else(|_| panic!("exact_maxcut failed"));

    let recomputed = g.cut_value(&result.partition);
    assert!((result.cut_value - recomputed).abs() < 1e-10);
}
