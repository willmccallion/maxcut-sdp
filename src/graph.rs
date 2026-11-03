//! Weighted undirected graph representation.
//!
//! The [`Graph`] type stores a graph as both an edge list and adjacency lists,
//! supporting efficient iteration over edges and neighbors. The key method
//! [`Graph::laplacian`] constructs the graph Laplacian matrix `L = D - W`
//! needed for the Max-Cut SDP formulation.
//!
//! See [`maxcut/MATH.md`](../maxcut/MATH.md) §3 for the Laplacian definition.

use crate::GwError;
use crate::sdp::linalg::DenseMatrix;

/// A weighted undirected graph.
///
/// Vertices are indexed `0..n`. Edges are stored both as a flat list and in
/// per-vertex adjacency lists for efficient traversal.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices.
    n: usize,
    /// Edge list: `(u, v, weight)` with `u < v`.
    edges: Vec<(usize, usize, f64)>,
    /// Adjacency lists: `adjacency[v]` contains `(neighbor, weight)` pairs.
    adjacency: Vec<Vec<(usize, f64)>>,
}

impl Graph {
    /// Creates a new graph with `n` vertices and no edges.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            adjacency: vec![Vec::new(); n],
        }
    }

    /// Adds an undirected edge between vertices `u` and `v` with the given weight.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidGraph`] if `u` or `v` is out of range,
    /// or if `u == v` (self-loops are not allowed).
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) -> Result<(), GwError> {
        if u >= self.n || v >= self.n {
            return Err(GwError::InvalidGraph(format!(
                "vertex index out of range: u={u}, v={v}, n={}",
                self.n
            )));
        }
        if u == v {
            return Err(GwError::InvalidGraph(format!(
                "self-loops not allowed: u=v={u}"
            )));
        }
        let (lo, hi) = if u < v { (u, v) } else { (v, u) };
        self.edges.push((lo, hi, weight));
        self.adjacency[u].push((v, weight));
        self.adjacency[v].push((u, weight));
        Ok(())
    }

    /// Returns the number of vertices.
    pub const fn num_vertices(&self) -> usize {
        self.n
    }

    /// Returns the number of edges.
    pub const fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns the neighbors of vertex `v` as `(neighbor, weight)` pairs.
    ///
    /// # Panics
    ///
    /// Panics if `v >= n`.
    pub fn neighbors(&self, v: usize) -> &[(usize, f64)] {
        &self.adjacency[v]
    }

    /// Returns the full edge list as `(u, v, weight)` triples.
    pub fn edges(&self) -> &[(usize, usize, f64)] {
        &self.edges
    }

    /// Constructs the graph Laplacian matrix `L = D - W`.
    ///
    /// - `W` is the weighted adjacency matrix: `W_ij = w_ij` if `(i,j) ∈ E`
    /// - `D` is the degree matrix: `D_ii = Σ_j w_ij`
    /// - `L_ij = -w_ij` for `i ≠ j`, `L_ii = Σ_{k≠i} w_ik`
    ///
    /// The Laplacian is symmetric positive semidefinite with `L·1 = 0`.
    ///
    /// See [`maxcut/MATH.md`](../maxcut/MATH.md) §3 for details.
    pub fn laplacian(&self) -> DenseMatrix {
        let n = self.n;
        let mut l = DenseMatrix::zeros(n, n);

        // Single pass over edges: set off-diagonal entries and accumulate degree.
        for &(u, v, w) in &self.edges {
            l.set(u, v, l.get(u, v) - w);
            l.set(v, u, l.get(v, u) - w);
            l.set(u, u, l.get(u, u) + w);
            l.set(v, v, l.get(v, v) + w);
        }

        l
    }

    /// Returns the total weight of all edges.
    pub fn total_weight(&self) -> f64 {
        let mut w = 0.0_f64;
        for &(_, _, weight) in &self.edges {
            w += weight;
        }
        w
    }

    /// Computes the cut value for a given vertex partition.
    ///
    /// The partition is encoded as a boolean slice: vertices with `partition[i] = true`
    /// are in set `S`, and those with `false` are in `S̄`. The cut value is the total
    /// weight of edges crossing the partition.
    ///
    /// # Panics
    ///
    /// Panics if `partition.len() != n`.
    pub fn cut_value(&self, partition: &[bool]) -> f64 {
        assert_eq!(partition.len(), self.n);
        let mut cut = 0.0_f64;
        for &(u, v, w) in &self.edges {
            if partition[u] != partition[v] {
                cut += w;
            }
        }
        cut
    }
}
