//! Brute-force exact Max-Cut solver for small graphs.
//!
//! Enumerates all `2^n` possible vertex partitions and returns the one with the
//! maximum cut value. This is only feasible for small graphs, but provides
//! ground-truth values for testing the GW approximation.
//!
//! The inner loop is optimized to avoid allocation: the partition is decoded
//! from the integer bitmask directly.

use super::Cut;
use crate::GwError;
use crate::graph::Graph;

/// Most vertices a partition bitmask can hold with the shift `1 << n` in a `u64`.
const MAX_MASK_VERTICES: usize = 63;

/// The vertex limit of [`exact_maxcut`], where enumeration takes seconds.
pub const DEFAULT_MAX_VERTICES: usize = 24;

/// Computes the exact maximum cut by brute-force enumeration.
///
/// Iterates over all `2^n` partitions of the vertex set. Returns the partition
/// achieving the maximum cut value, along with the cut value itself.
///
/// # Errors
///
/// Returns [`GwError::InvalidGraph`] if the graph has more than
/// [`DEFAULT_MAX_VERTICES`] vertices.
pub fn exact_maxcut(graph: &Graph) -> Result<Cut, GwError> {
    exact_maxcut_bounded(graph, DEFAULT_MAX_VERTICES)
}

/// [`exact_maxcut`] with a caller-specified vertex limit.
///
/// Bit `i` of each mask puts vertex `i` in `S`. Only masks with vertex 0 in
/// `S` are tried, since a partition and its complement cut alike, and the
/// first mask of the largest cut wins.
///
/// # Errors
///
/// Returns [`GwError::InvalidGraph`] if `graph.num_vertices() > max_vertices`,
/// or past the 63 vertices a bitmask holds, whatever `max_vertices` is.
pub fn exact_maxcut_bounded(graph: &Graph, max_vertices: usize) -> Result<Cut, GwError> {
    let n = graph.num_vertices();
    let max_vertices = max_vertices.min(MAX_MASK_VERTICES);
    if n > max_vertices {
        return Err(GwError::InvalidGraph(format!(
            "exact solver limited to {max_vertices} vertices, got {n}"
        )));
    }
    if n == 0 {
        return Ok(Cut {
            partition: Vec::new(),
            cut_value: 0.0,
        });
    }

    let edges = graph.edges();
    let mut best_mask = 0_u64;
    let mut best_cut = 0.0_f64;
    for mask in (1..1_u64 << n).step_by(2) {
        let cut = mask_cut_value(edges, mask);
        if cut > best_cut {
            best_cut = cut;
            best_mask = mask;
        }
    }

    Ok(Cut {
        partition: partition_from_mask(best_mask, n),
        cut_value: best_cut,
    })
}

/// The weight of the edges whose ends' bits differ in `mask`.
fn mask_cut_value(edges: &[(usize, usize, f64)], mask: u64) -> f64 {
    let mut cut = 0.0_f64;
    for &(u, v, w) in edges {
        if (mask >> u) & 1 != (mask >> v) & 1 {
            cut += w;
        }
    }
    cut
}

/// The partition of `n` vertices whose bits `mask` sets in `S`.
fn partition_from_mask(mask: u64, n: usize) -> Vec<bool> {
    (0..n).map(|i| (mask >> i) & 1 == 1).collect()
}
