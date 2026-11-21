//! The approximation ratio where the SDP bound is zero and no ratio exists.

mod common;

use common::{cycle, triangle};
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::{Backend, BmConfig, GwConfig, gw_maxcut};
use maxcut_sdp::sdp::ipm::IpmConfig;

/// The ratio each solver reports for `graph`: `[IPM, BM]`.
fn ratios(graph: &Graph) -> [Option<f64>; 2] {
    let config = |backend| GwConfig {
        seed: Some(1),
        backend,
        ..GwConfig::default()
    };
    let ipm = gw_maxcut(graph, &config(Backend::Ipm(IpmConfig::default())), |_| {});
    let bm = gw_maxcut(graph, &config(Backend::Bm(BmConfig::default())), |_| {});
    let ipm = ipm.unwrap_or_else(|e| panic!("ipm failed: {e}"));
    let bm = bm.unwrap_or_else(|e| panic!("bm failed: {e}"));
    [ipm.approximation_ratio, bm.approximation_ratio]
}

#[test]
fn neither_solver_reports_a_ratio_against_a_zero_bound() {
    assert_eq!(ratios(&triangle(-1.0)), [None, None], "[IPM, BM]");
}

#[test]
fn both_solvers_report_a_ratio_against_a_positive_bound() {
    let [ipm, bm] = ratios(&cycle(5));

    assert!(
        ipm.is_some_and(|r| r > 0.87) && bm.is_some_and(|r| r > 0.87),
        "[{ipm:?}, {bm:?}]"
    );
}
