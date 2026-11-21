//! The graph sizes the brute-force exact solver accepts.

mod common;

use common::{cycle, graph_with};
use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::exact::{exact_maxcut, exact_maxcut_bounded};

#[test]
fn exact_rejects_more_vertices_than_a_mask_holds() {
    let edges: Vec<_> = (0..63).map(|v| (v, v + 1, 1.0)).collect();
    let path = graph_with(64, &edges);

    let result = exact_maxcut_bounded(&path, 64);

    assert!(
        matches!(result, Err(GwError::InvalidGraph(_))),
        "{:?}",
        result.map(|cut| cut.cut_value)
    );
}

#[test]
fn exact_rejects_graphs_past_its_default_limit() {
    let result = exact_maxcut(&cycle(25));

    assert!(
        matches!(result, Err(GwError::InvalidGraph(_))),
        "{:?}",
        result.map(|cut| cut.cut_value)
    );
}

#[test]
fn the_empty_graph_has_an_empty_cut_of_zero() {
    let cut = exact_maxcut(&Graph::new(0)).unwrap_or_else(|e| panic!("{e}"));

    assert_eq!(cut.partition, Vec::<bool>::new());
    assert_eq!(cut.cut_value, 0.0);
}
