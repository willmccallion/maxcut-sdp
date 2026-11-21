//! The exact brute-force solver against hand-computed maximum cuts.

mod common;

use common::{complete, cycle, graph_with, triangle};
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::exact::exact_maxcut;

#[test]
fn exact_finds_the_known_optimum_of_small_graphs() {
    let cases: [(&str, Graph, f64); 5] = [
        ("single edge", graph_with(2, &[(0, 1, 1.0)]), 1.0),
        ("triangle: one vertex cut off", triangle(1.0), 2.0),
        ("4-cycle: bipartite, every edge cut", cycle(4), 4.0),
        ("5-cycle: one edge uncut", cycle(5), 4.0),
        ("K4: two against two", complete(4), 4.0),
    ];

    for (name, graph, optimum) in cases {
        let cut = exact_maxcut(&graph).unwrap_or_else(|e| panic!("{name}: {e}"));

        assert!(
            (cut.cut_value - optimum).abs() < 1e-10,
            "{name}: cut {}, optimum {optimum}",
            cut.cut_value
        );
        assert_eq!(graph.cut_value(&cut.partition), cut.cut_value, "{name}");
    }
}
