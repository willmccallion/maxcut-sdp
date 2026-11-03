//! Parsers for graph instance file formats.
//!
//! Supports two standard formats:
//!
//! - **Biq Mac Library**: First line is `n m`, followed by `m` lines of `i j w_ij`.
//!   Vertex indices are 1-based.
//! - **DIMACS**: Problem line `p edge n m`, edge lines `e i j [w]`. Vertex indices
//!   are 1-based. Weights default to 1 if omitted.
//!
//! Use [`parse_auto`] to auto-detect the format from the input content.

use crate::GwError;
use crate::graph::Graph;

/// Parses a graph in Biq Mac Library format.
///
/// Format:
/// ```text
/// n m
/// i1 j1 w1
/// i2 j2 w2
/// ...
/// ```
///
/// Vertex indices are 1-based and converted to 0-based internally.
///
/// # Errors
///
/// Returns [`GwError::ParseError`] if the input is malformed.
pub fn parse_biqmac(input: &str) -> Result<Graph, GwError> {
    let mut lines = input.lines().filter(|l| !l.trim().is_empty());

    let header = lines
        .next()
        .ok_or_else(|| GwError::ParseError("empty input".to_string()))?;
    let mut parts = header.split_whitespace();

    let n: usize = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| GwError::ParseError("missing vertex count".to_string()))?;
    let m: usize = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| GwError::ParseError("missing edge count".to_string()))?;

    let mut graph = Graph::new(n);

    for (line_num, line) in lines.enumerate() {
        let mut parts = line.split_whitespace();
        let i: usize = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
            GwError::ParseError(format!("bad vertex on edge line {}", line_num + 2))
        })?;
        let j: usize = parts.next().and_then(|s| s.parse().ok()).ok_or_else(|| {
            GwError::ParseError(format!("bad vertex on edge line {}", line_num + 2))
        })?;
        let w: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);

        // Convert 1-based to 0-based.
        if i == 0 || j == 0 || i > n || j > n {
            return Err(GwError::ParseError(format!(
                "vertex index out of range on line {}: i={i}, j={j}",
                line_num + 2
            )));
        }
        graph.add_edge(i - 1, j - 1, w)?;
    }

    let _ = m; // m is informational; we accept files with fewer/more edges.
    Ok(graph)
}

/// Parses a graph in DIMACS format.
///
/// Format:
/// ```text
/// c comment lines
/// p edge n m
/// e i j
/// e i j w
/// ...
/// ```
///
/// Vertex indices are 1-based. Edge weights default to 1 if omitted.
///
/// # Errors
///
/// Returns [`GwError::ParseError`] if the input is malformed.
pub fn parse_dimacs(input: &str) -> Result<Graph, GwError> {
    let mut n = 0_usize;
    let mut graph: Option<Graph> = None;

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('c') {
            continue;
        }

        if trimmed.starts_with("p ") {
            let mut parts = trimmed.split_whitespace();
            let _ = parts.next(); // "p"
            let _ = parts.next(); // "edge"
            n = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| GwError::ParseError("bad problem line".to_string()))?;
            // m is informational
            graph = Some(Graph::new(n));
            continue;
        }

        if trimmed.starts_with("e ") {
            let g = graph
                .as_mut()
                .ok_or_else(|| GwError::ParseError("edge before problem line".to_string()))?;
            let mut parts = trimmed.split_whitespace();
            let _ = parts.next(); // "e"
            let i: usize = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| GwError::ParseError("bad edge line".to_string()))?;
            let j: usize = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| GwError::ParseError("bad edge line".to_string()))?;
            let w: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);

            if i == 0 || j == 0 || i > n || j > n {
                return Err(GwError::ParseError(format!(
                    "vertex index out of range: i={i}, j={j}"
                )));
            }
            g.add_edge(i - 1, j - 1, w)?;
        }
    }

    graph.ok_or_else(|| GwError::ParseError("no problem line found".to_string()))
}

/// Auto-detects the input format and parses accordingly.
///
/// Heuristic: if any line starts with `p ` or `e `, it's DIMACS; otherwise Biq Mac.
///
/// # Errors
///
/// Returns [`GwError::ParseError`] if parsing fails.
pub fn parse_auto(input: &str) -> Result<Graph, GwError> {
    let is_dimacs = input
        .lines()
        .any(|l| l.starts_with("p ") || l.starts_with("e "));
    if is_dimacs {
        parse_dimacs(input)
    } else {
        parse_biqmac(input)
    }
}
