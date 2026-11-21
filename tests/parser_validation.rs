//! Parser behavior on malformed input, and the lines its errors name.

use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
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
    let mut graph = Graph::new(2);

    let added = graph.add_edge(0, 1, f64::NAN);

    assert!(added.is_err());
}

/// The message of a parse that must fail.
fn parse_error(parsed: Result<Graph, GwError>) -> String {
    match parsed {
        Err(GwError::ParseError(message)) => message,
        Err(other) => panic!("expected a parse error, got {other}"),
        Ok(graph) => panic!("accepted malformed input; edges = {:?}", graph.edges()),
    }
}

#[test]
fn biqmac_error_names_the_file_line_after_blank_lines() {
    let input = "3 2\n\n1 2 1\n\n2 3 x\n";

    let message = parse_error(parse_biqmac(input));

    assert!(message.starts_with("line 5:"), "{message}");
}

#[test]
fn biqmac_rejects_a_zero_vertex_index() {
    let message = parse_error(parse_biqmac("2 1\n0 1 1\n"));

    assert!(message.starts_with("line 2:"), "{message}");
}

#[test]
fn biqmac_rejects_a_vertex_past_the_vertex_count() {
    let message = parse_error(parse_biqmac("2 1\n1 3 1\n"));

    assert!(message.contains("vertex 3"), "{message}");
}

#[test]
fn biqmac_rejects_a_header_without_an_edge_count() {
    let message = parse_error(parse_biqmac("2\n1 2 1\n"));

    assert!(message.contains("edge count"), "{message}");
}

#[test]
fn biqmac_rejects_empty_input() {
    let message = parse_error(parse_biqmac("\n  \n"));

    assert!(message.contains("empty"), "{message}");
}

#[test]
fn biqmac_rejects_a_self_loop_naming_its_line() {
    let message = parse_error(parse_biqmac("2 2\n1 2 1\n2 2 1\n"));

    assert!(message.starts_with("line 3:"), "{message}");
}

#[test]
fn dimacs_rejects_an_edge_before_the_problem_line() {
    let message = parse_error(parse_dimacs("e 1 2\np edge 2 1\n"));

    assert!(message.starts_with("line 1:"), "{message}");
}

#[test]
fn dimacs_requires_a_problem_line() {
    let message = parse_error(parse_dimacs("c only a comment\n"));

    assert!(message.contains("no problem line"), "{message}");
}

#[test]
fn dimacs_error_names_the_file_line() {
    let input = "c a comment\np edge 2 1\ne 1 2 abc\n";

    let message = parse_error(parse_dimacs(input));

    assert!(message.starts_with("line 3:"), "{message}");
}

#[test]
fn dimacs_skips_comment_lines() {
    let input = "c header comment\np edge 3 2\nc between edges\ne 1 2\ne 2 3 2.5\n";

    let edges = parse_dimacs(input).map(|g| g.edges().to_vec());

    assert_eq!(edges.ok(), Some(vec![(0, 1, 1.0), (1, 2, 2.5)]));
}
