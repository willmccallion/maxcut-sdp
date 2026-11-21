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

use std::fmt;

use crate::GwError;
use crate::graph::Graph;

/// Problem-line formats a DIMACS graph may declare: `edge`, and `col` from
/// the graph-colouring instances.
const DIMACS_FORMATS: [&str; 2] = ["edge", "col"];

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
/// Vertex indices are 1-based and converted to 0-based internally. The edge
/// count `m` must be present but is not checked against the edge lines.
///
/// # Errors
///
/// Returns [`GwError::ParseError`], naming the line, if the input is malformed.
pub fn parse_biqmac(input: &str) -> Result<Graph, GwError> {
    let mut lines = numbered_lines(input).filter(|(_, line)| !line.trim().is_empty());
    let (header_number, header) = lines
        .next()
        .ok_or_else(|| GwError::ParseError("empty input".to_string()))?;
    let n = parse_counts(header.split_whitespace()).map_err(|e| e.at(header_number))?;

    let mut graph = Graph::new(n);
    for (number, line) in lines {
        add_edge(&mut graph, line.split_whitespace()).map_err(|e| e.at(number))?;
    }
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
/// Vertex indices are 1-based. Edge weights default to 1 if omitted. The
/// problem line's format is `edge` or `col`, and it must come once, before
/// any edge.
///
/// # Errors
///
/// Returns [`GwError::ParseError`], naming the line, if the input is malformed.
pub fn parse_dimacs(input: &str) -> Result<Graph, GwError> {
    let mut graph: Option<Graph> = None;
    for (number, line) in numbered_lines(input) {
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("p") => {
                if graph.is_some() {
                    return Err(LineError::SecondProblemLine.at(number));
                }
                graph = Some(parse_problem_line(tokens).map_err(|e| e.at(number))?);
            }
            Some("e") => {
                let graph = graph
                    .as_mut()
                    .ok_or_else(|| LineError::EdgeBeforeProblemLine.at(number))?;
                add_edge(graph, tokens).map_err(|e| e.at(number))?;
            }
            _ => {}
        }
    }
    graph.ok_or_else(|| GwError::ParseError("no problem line found".to_string()))
}

/// Auto-detects the input format and parses accordingly.
///
/// Heuristic: if any line starts with `p ` or `e ` after its indentation,
/// it's DIMACS; otherwise Biq Mac.
///
/// # Errors
///
/// Returns [`GwError::ParseError`] if parsing fails.
pub fn parse_auto(input: &str) -> Result<Graph, GwError> {
    let is_dimacs = input
        .lines()
        .map(str::trim_start)
        .any(|l| l.starts_with("p ") || l.starts_with("e "));
    if is_dimacs {
        parse_dimacs(input)
    } else {
        parse_biqmac(input)
    }
}

/// The input's lines with their 1-based numbers in the file.
fn numbered_lines(input: &str) -> impl Iterator<Item = (usize, &str)> {
    input
        .lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
}

/// A Biq Mac header `n m`: the vertex count `n`, once `m` parses too.
fn parse_counts<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<usize, LineError> {
    let vertices = parse_count(tokens.next(), "vertex count")?;
    let _ = parse_count(tokens.next(), "edge count")?;
    Ok(vertices)
}

/// The empty graph a DIMACS problem line `p FORMAT n m` declares, given the
/// tokens after `p`.
fn parse_problem_line<'a>(mut tokens: impl Iterator<Item = &'a str>) -> Result<Graph, LineError> {
    let format = tokens.next().ok_or(LineError::Missing("format"))?;
    if !DIMACS_FORMATS.contains(&format) {
        return Err(LineError::UnknownFormat(format.to_string()));
    }
    Ok(Graph::new(parse_counts(tokens)?))
}

/// Adds the edge `i j [w]` that `tokens` give, with 1-based vertices.
fn add_edge<'a>(
    graph: &mut Graph,
    mut tokens: impl Iterator<Item = &'a str>,
) -> Result<(), LineError> {
    let n = graph.num_vertices();
    let u = parse_vertex(tokens.next(), n)?;
    let v = parse_vertex(tokens.next(), n)?;
    let weight = parse_weight(tokens.next())?;
    graph.add_edge(u, v, weight).map_err(LineError::Graph)
}

/// A count field, required to be a non-negative integer.
fn parse_count(token: Option<&str>, field: &'static str) -> Result<usize, LineError> {
    let token = token.ok_or(LineError::Missing(field))?;
    token.parse().map_err(|_| LineError::NotANumber {
        field,
        token: token.to_string(),
    })
}

/// The 0-based index of the 1-based vertex `token` names in an `n`-vertex graph.
fn parse_vertex(token: Option<&str>, n: usize) -> Result<usize, LineError> {
    let vertex = parse_count(token, "vertex")?;
    if vertex == 0 || vertex > n {
        return Err(LineError::VertexOutOfRange { vertex, n });
    }
    Ok(vertex - 1)
}

/// An edge's weight: 1 when the line gives none, otherwise the finite
/// number it gives.
fn parse_weight(token: Option<&str>) -> Result<f64, LineError> {
    let Some(token) = token else {
        return Ok(1.0);
    };
    token
        .parse::<f64>()
        .ok()
        .filter(|w| w.is_finite())
        .ok_or_else(|| LineError::NotANumber {
            field: "weight",
            token: token.to_string(),
        })
}

/// What is wrong with one line of an input file.
enum LineError {
    Missing(&'static str),
    NotANumber { field: &'static str, token: String },
    VertexOutOfRange { vertex: usize, n: usize },
    UnknownFormat(String),
    SecondProblemLine,
    EdgeBeforeProblemLine,
    Graph(GwError),
}

impl LineError {
    /// This error as found on line `number` of the file.
    fn at(self, number: usize) -> GwError {
        GwError::ParseError(format!("line {number}: {self}"))
    }
}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(field) => write!(f, "missing {field}"),
            Self::NotANumber { field, token } => write!(f, "bad {field} `{token}`"),
            Self::VertexOutOfRange { vertex, n } => {
                write!(f, "vertex {vertex} is outside 1..={n}")
            }
            Self::UnknownFormat(format) => write!(
                f,
                "problem line format `{format}` is not one of {DIMACS_FORMATS:?}"
            ),
            Self::SecondProblemLine => write!(f, "a second problem line"),
            Self::EdgeBeforeProblemLine => write!(f, "edge before the problem line"),
            Self::Graph(error) => write!(f, "{error}"),
        }
    }
}
