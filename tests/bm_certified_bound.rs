//! The bound the Burer-Monteiro pipeline reports holds whether or not BM
//! converged, and is labelled uncertified where it is not checked.

mod common;

use std::num::NonZeroUsize;

use common::random_graph_with_edges;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::{
    GwBmConfig, GwConfig, MAX_CERTIFIED_VERTICES, SdpBound, gw_maxcut, gw_maxcut_bm,
};
use maxcut_sdp::sdp::bm::BmSolverConfig;

/// The SDP's value, from the IPM's certified bound.
fn sdp_value(graph: &Graph) -> f64 {
    let ipm = gw_maxcut(
        graph,
        &GwConfig {
            seed: Some(1),
            ..GwConfig::default()
        },
    );
    match ipm.unwrap_or_else(|e| panic!("ipm failed: {e}")).sdp_bound {
        SdpBound::Certified(bound) => bound,
        SdpBound::Relaxation(value) => panic!("the IPM left its value {value} uncertified"),
    }
}

/// A BM run with one rounding trial and BM stopped after `max_iterations`.
fn quick_bm_config(max_iterations: usize) -> GwBmConfig {
    GwBmConfig {
        num_trials: NonZeroUsize::MIN,
        ils_rounds: 0,
        solver_config: BmSolverConfig {
            max_iterations,
            ..BmSolverConfig::default()
        },
        seed: Some(1),
        ..GwBmConfig::default()
    }
}

#[test]
fn an_unconverged_bm_still_reports_a_bound_at_least_the_sdp_value() {
    let graph = random_graph_with_edges(60, 240, 9);

    let bm = gw_maxcut_bm(&graph, &quick_bm_config(2), |_| {});

    let bm = bm.unwrap_or_else(|e| panic!("bm failed: {e}"));
    let sdp = sdp_value(&graph);
    assert!(
        matches!(bm.sdp_bound, SdpBound::Certified(bound) if bound >= sdp - 1e-6),
        "BM bound {:?} not a certified bound on the SDP value {sdp}",
        bm.sdp_bound
    );
}

#[test]
fn a_converged_bm_certifies_a_bound_close_to_the_sdp_value() {
    let graph = random_graph_with_edges(60, 240, 9);

    let bm = gw_maxcut_bm(&graph, &quick_bm_config(10_000), |_| {});

    let bm = bm.unwrap_or_else(|e| panic!("bm failed: {e}"));
    let sdp = sdp_value(&graph);
    let SdpBound::Certified(bound) = bm.sdp_bound else {
        panic!("BM left a 60-vertex bound uncertified");
    };
    assert!(
        bound >= sdp - 1e-6 && bound <= sdp * (1.0 + 1e-4),
        "BM bound {bound} vs SDP value {sdp}"
    );
}

#[test]
fn past_the_certified_size_bm_reports_its_relaxation_value_and_no_ratio() {
    let n = MAX_CERTIFIED_VERTICES + 1;
    let graph = random_graph_with_edges(n, 2 * n, 9);

    let bm = gw_maxcut_bm(&graph, &quick_bm_config(10), |_| {});

    let bm = bm.unwrap_or_else(|e| panic!("bm failed: {e}"));
    assert!(
        matches!(bm.sdp_bound, SdpBound::Relaxation(value) if value > 0.0),
        "{:?}",
        bm.sdp_bound
    );
    assert_eq!(bm.approximation_ratio, None);
}
