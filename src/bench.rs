//! Benchmarking utilities for measuring solver performance.
//!
//! Provides graph generators for reproducible benchmarks and a result type
//! for tracking timing and approximation quality across instances.

use crate::graph::Graph;
use rand::Rng;

/// Summary of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Name of the graph instance.
    pub graph_name: String,
    /// Number of vertices.
    pub n: usize,
    /// Number of edges.
    pub m: usize,
    /// Best cut value found by GW.
    pub cut_value: f64,
    /// SDP relaxation bound.
    pub sdp_bound: f64,
    /// Approximation ratio `cut_value / sdp_bound`.
    pub ratio: f64,
    /// Time spent solving the SDP.
    pub solve_time: std::time::Duration,
    /// Total time including rounding.
    pub total_time: std::time::Duration,
}

/// Generates an Erdos-Renyi random graph `G(n, p)` with random weights in `[0, 1]`.
///
/// Each possible edge is included independently with probability `edge_probability`.
/// Edge weights are drawn uniformly from `[0, 1]`.
pub fn generate_random_graph(n: usize, edge_probability: f64, rng: &mut impl Rng) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let sample: f64 = rng.random();
            if sample < edge_probability {
                let weight: f64 = rng.random();
                // add_edge only fails on invalid indices or self-loops, neither possible here.
                let _ = g.add_edge(i, j, weight);
            }
        }
    }
    g
}

/// Generates the complete graph `K_n` with unit weights.
pub fn generate_complete_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let _ = g.add_edge(i, j, 1.0);
        }
    }
    g
}

/// Generates the cycle graph `C_n` with unit weights.
pub fn generate_cycle(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        let j = (i + 1) % n;
        if i != j {
            let _ = g.add_edge(i, j, 1.0);
        }
    }
    g
}
