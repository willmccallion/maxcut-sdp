//! The edges a graph refuses.

use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;

fn add_edge_error(u: usize, v: usize, weight: f64) -> GwError {
    let mut graph = Graph::new(3);
    match graph.add_edge(u, v, weight) {
        Err(e) => e,
        Ok(()) => panic!("accepted the edge ({u}, {v}, {weight})"),
    }
}

#[test]
fn a_graph_refuses_a_non_finite_weight() {
    assert!(matches!(
        add_edge_error(0, 1, f64::NAN),
        GwError::InvalidGraph(_)
    ));
}

#[test]
fn a_graph_refuses_a_vertex_out_of_range() {
    assert!(matches!(
        add_edge_error(0, 3, 1.0),
        GwError::InvalidGraph(_)
    ));
}

#[test]
fn a_graph_refuses_a_self_loop() {
    assert!(matches!(
        add_edge_error(1, 1, 1.0),
        GwError::InvalidGraph(_)
    ));
}
