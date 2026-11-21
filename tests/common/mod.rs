//! Graphs the tests and benchmarks build.
#![allow(dead_code, reason = "each test binary uses its own subset")]

use std::ops::RangeInclusive;

use maxcut_sdp::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// An `n`-vertex graph with exactly `edges`, each `(u, v, weight)`.
pub fn graph_with(n: usize, edges: &[(usize, usize, f64)]) -> Graph {
    let mut graph = Graph::new(n);
    for &(u, v, weight) in edges {
        graph
            .add_edge(u, v, weight)
            .unwrap_or_else(|e| panic!("valid edge ({u}, {v}) rejected: {e}"));
    }
    graph
}

/// The cycle `C_n` with unit weights.
pub fn cycle(n: usize) -> Graph {
    assert!(n >= 3, "a simple cycle needs at least 3 vertices, got {n}");
    let edges: Vec<_> = (0..n).map(|i| (i, (i + 1) % n, 1.0)).collect();
    graph_with(n, &edges)
}

/// The complete graph `K_n` with unit weights.
pub fn complete(n: usize) -> Graph {
    let edges: Vec<_> = (0..n)
        .flat_map(|u| (u + 1..n).map(move |v| (u, v, 1.0)))
        .collect();
    graph_with(n, &edges)
}

/// The triangle `K_3` with every weight `weight`.
pub fn triangle(weight: f64) -> Graph {
    graph_with(3, &[(0, 1, weight), (1, 2, weight), (0, 2, weight)])
}

/// Erdős–Rényi `G(n, p)`: each pair joined with probability `p` by an edge
/// of integer weight drawn from `weights`.
pub fn erdos_renyi(n: usize, p: f64, weights: RangeInclusive<u32>, seed: u64) -> Graph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    for u in 0..n {
        for v in u + 1..n {
            if rng.random::<f64>() < p {
                edges.push((u, v, f64::from(rng.random_range(weights.clone()))));
            }
        }
    }
    graph_with(n, &edges)
}

/// About five neighbors per vertex, with weights 1 to 10.
pub fn sparse_random_graph(n: usize, seed: u64) -> Graph {
    erdos_renyi(n, 5.0 / n as f64, 1..=10, seed)
}

/// `edges` random edges between distinct vertices, with weights 1 to 5;
/// repeated pairs count as separate edges.
pub fn random_graph_with_edges(n: usize, edges: usize, seed: u64) -> Graph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chosen = Vec::with_capacity(edges);
    while chosen.len() < edges {
        let (u, v) = (rng.random_range(0..n), rng.random_range(0..n));
        if u != v {
            chosen.push((u, v, f64::from(rng.random_range(1..=5_u32))));
        }
    }
    graph_with(n, &chosen)
}
