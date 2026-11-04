//! Parser behavior on malformed weights and indented DIMACS input.

use maxcut_sdp::parser::{parse_auto, parse_biqmac, parse_dimacs};

#[test]
fn biqmac_rejects_non_numeric_weight() {
    let input = "2 1\n1 2 abc\n";

    let parsed = parse_biqmac(input);

    assert!(
        parsed.is_err(),
        "accepted weight `abc`; edges = {:?}",
        parsed.map(|g| g.edges().to_vec())
    );
}

#[test]
fn biqmac_rejects_comma_decimal_weight() {
    let input = "2 1\n1 2 2,5\n";

    let parsed = parse_biqmac(input);

    assert!(
        parsed.is_err(),
        "accepted weight `2,5`; edges = {:?}",
        parsed.map(|g| g.edges().to_vec())
    );
}

#[test]
fn dimacs_rejects_non_numeric_weight() {
    let input = "p edge 2 1\ne 1 2 abc\n";

    let parsed = parse_dimacs(input);

    assert!(
        parsed.is_err(),
        "accepted weight `abc`; edges = {:?}",
        parsed.map(|g| g.edges().to_vec())
    );
}

#[test]
fn auto_detects_indented_dimacs() {
    let input = "  p edge 3 2\n  e 1 2\n  e 2 3\n";

    let parsed = parse_auto(input);

    match parsed {
        Ok(graph) => {
            assert_eq!(graph.num_vertices(), 3);
            assert_eq!(graph.num_edges(), 2);
        }
        Err(e) => panic!("indented DIMACS failed to parse: {e}"),
    }
}

#[test]
fn biqmac_rejects_non_finite_weight() {
    let input = "3 2\n1 2 NaN\n2 3 inf\n";

    let parsed = parse_biqmac(input);

    assert!(
        parsed.is_err(),
        "accepted non-finite weights; edges = {:?}",
        parsed.map(|g| g.edges().to_vec())
    );
}

#[test]
fn a_missing_weight_is_one() {
    let parsed = parse_biqmac("2 1\n1 2\n");

    let edges = parsed.map(|g| g.edges().to_vec());
    assert_eq!(edges.ok(), Some(vec![(0, 1, 1.0)]));
}

#[test]
fn a_graph_refuses_a_non_finite_weight() {
    let mut graph = maxcut_sdp::graph::Graph::new(2);

    let added = graph.add_edge(0, 1, f64::NAN);

    assert!(added.is_err());
}
