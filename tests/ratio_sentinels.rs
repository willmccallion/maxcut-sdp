//! The approximation ratio where the SDP bound is zero and no ratio exists.

use std::process::Command;

use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::{GwBmConfig, GwConfig, gw_maxcut, gw_maxcut_bm};

fn graph_with(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
    let mut g = Graph::new(n);
    for &(u, v, w) in edges {
        g.add_edge(u, v, w)
            .unwrap_or_else(|e| panic!("valid edge rejected: {e}"));
    }
    g
}

fn negative_triangle() -> Graph {
    graph_with(3, &[(0, 1, -1.0), (1, 2, -1.0), (0, 2, -1.0)])
}

/// The ratio each solver reports for `graph`: `[IPM, BM]`.
fn ratios(graph: &Graph) -> [Option<f64>; 2] {
    let ipm = gw_maxcut(
        graph,
        &GwConfig {
            seed: Some(1),
            ..GwConfig::default()
        },
    );
    let bm = gw_maxcut_bm(
        graph,
        &GwBmConfig {
            seed: Some(1),
            ..GwBmConfig::default()
        },
    );
    let ipm = ipm.unwrap_or_else(|e| panic!("ipm failed: {e}"));
    let bm = bm.unwrap_or_else(|e| panic!("bm failed: {e}"));
    [ipm.approximation_ratio, bm.approximation_ratio]
}

#[test]
fn neither_solver_reports_a_ratio_against_a_zero_bound() {
    assert_eq!(ratios(&negative_triangle()), [None, None], "[IPM, BM]");
}

#[test]
fn both_solvers_report_a_ratio_against_a_positive_bound() {
    let cycle = graph_with(
        5,
        &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 0, 1.0),
        ],
    );

    let [ipm, bm] = ratios(&cycle);

    assert!(
        ipm.is_some_and(|r| r > 0.87) && bm.is_some_and(|r| r > 0.87),
        "[{ipm:?}, {bm:?}]"
    );
}

#[test]
fn the_cli_exact_comparison_never_prints_nan() {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("negative_triangle.txt");
    std::fs::write(&path, "3 3\n1 2 -1\n2 3 -1\n1 3 -1\n")
        .unwrap_or_else(|e| panic!("write graph: {e}"));

    let output = Command::new(env!("CARGO_BIN_EXE_maxcut-sdp"))
        .arg(&path)
        .args(["--exact", "--seed", "1"])
        .output()
        .unwrap_or_else(|e| panic!("run the CLI: {e}"));

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && !stdout.contains("NaN"),
        "stdout:\n{stdout}"
    );
}
