//! Brute-force exact Max-Cut solver for small graphs.
//!
//! Enumerates all `2^n` possible vertex partitions and returns the one with the
//! maximum cut value. This is only feasible for small graphs (`n ≤ 24` or so)
//! but provides ground-truth values for testing the GW approximation.
//!
//! The inner loop is optimized to avoid allocation: the partition is decoded
//! from the integer bitmask directly.

use super::MaxCutResult;
use crate::GwError;
use crate::graph::Graph;

/// Most vertices a partition bitmask can hold with the shift `1 << n` in a `u64`.
const MAX_MASK_VERTICES: usize = 63;

/// Computes the exact maximum cut by brute-force enumeration.
///
/// Iterates over all `2^n` partitions of the vertex set. Returns the partition
/// achieving the maximum cut value, along with the cut value itself.
///
/// # Errors
///
/// Returns [`GwError::InvalidGraph`] if the graph has more than 24 vertices.
pub fn exact_maxcut(graph: &Graph) -> Result<MaxCutResult, GwError> {
    exact_maxcut_bounded(graph, 24)
}

/// Computes the exact maximum cut, with a caller-specified vertex limit.
///
/// Same as [`exact_maxcut`] but allows the caller to set the maximum number
/// of vertices (rather than the hardcoded limit of 24).
///
/// The algorithm iterates over bitmasks `0..2^n`. For each bitmask, bit `i`
/// indicates whether vertex `i` is in set `S`. The cut value is computed
/// directly from the edge list using bitwise XOR to check crossing edges —
/// no allocation per iteration.
///
/// # Errors
///
/// Returns [`GwError::InvalidGraph`] if `graph.num_vertices() > max_vertices`,
/// or past the 63 vertices a bitmask holds, whatever `max_vertices` is.
pub fn exact_maxcut_bounded(graph: &Graph, max_vertices: usize) -> Result<MaxCutResult, GwError> {
    let n = graph.num_vertices();
    let max_vertices = max_vertices.min(MAX_MASK_VERTICES);
    if n > max_vertices {
        return Err(GwError::InvalidGraph(format!(
            "exact solver limited to {max_vertices} vertices, got {n}"
        )));
    }
    if n == 0 {
        return Ok(MaxCutResult {
            partition: Vec::new(),
            cut_value: 0.0,
            sdp_bound: 0.0,
            approximation_ratio: 1.0,
            num_trials: 0,
        });
    }

    let edges = graph.edges();
    let total_masks = 1_u64 << n;

    let mut best_mask = 0_u64;
    let mut best_cut = 0.0_f64;

    // By symmetry, we only need to check masks where vertex 0 is in S
    // (bit 0 = 1). The complementary partition gives the same cut value.
    // This halves the search space.
    let start = 1_u64;
    let end = total_masks;

    for mask in start..end {
        // Only consider masks with bit 0 set (symmetry breaking).
        if mask & 1 == 0 {
            continue;
        }

        let mut cut = 0.0_f64;
        for &(u, v, w) in edges {
            // Edge crosses the cut iff bits u and v differ.
            let u_side = (mask >> u) & 1;
            let v_side = (mask >> v) & 1;
            if u_side != v_side {
                cut += w;
            }
        }

        if cut > best_cut {
            best_cut = cut;
            best_mask = mask;
        }
    }

    // Decode the best mask into a partition vector.
    let mut partition = Vec::with_capacity(n);
    for i in 0..n {
        partition.push((best_mask >> i) & 1 == 1);
    }

    Ok(MaxCutResult {
        partition,
        cut_value: best_cut,
        sdp_bound: 0.0, // Not computed by exact solver.
        approximation_ratio: 1.0,
        num_trials: 0,
    })
}
