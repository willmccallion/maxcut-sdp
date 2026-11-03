//! Random hyperplane rounding for the GW algorithm.
//!
//! After solving the Max-Cut SDP relaxation, the optimal matrix `X*` encodes
//! a set of unit vectors `v_1, ..., v_n` (via `X* = V^T V`). This module
//! extracts those vectors and performs random hyperplane rounding:
//!
//! 1. Generate a random Gaussian vector `r ~ N(0, I)`
//! 2. Assign vertex `i` to `S` if `⟨v_i, r⟩ ≥ 0`, else to `S̄`
//!
//! The probability that vertices `i` and `j` are separated is
//! `arccos(⟨v_i, v_j⟩) / π`, which guarantees an expected cut value of
//! at least `α_GW · SDP_OPT`.
//!
//! See [`MATH.md`](../MATH.md) §4–§5 for the rounding analysis.

use super::MaxCutResult;
use crate::GwError;
use crate::graph::Graph;
use crate::sdp::linalg::DenseMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Extracts unit vectors from a PSD matrix via Cholesky factorization.
///
/// Given PSD matrix `X`, computes `L` such that `X = L L^T`, then returns
/// `L^T`. Column `i` of `L^T` (= row `i` of `L`) is the vector for vertex `i`.
/// The Gram property `(L^T)^T (L^T) = L L^T = X` ensures `⟨v_i, v_j⟩ = X_ij`.
///
/// At SDP optimality, `X*` may be PSD (not PD), so we add a small
/// regularization `X + εI` to ensure Cholesky succeeds.
///
/// # Errors
///
/// Returns an error if even the regularized matrix fails Cholesky.
pub fn extract_vectors(x: &DenseMatrix) -> Result<DenseMatrix, GwError> {
    let n = x.rows();

    // Try Cholesky directly first.
    if let Ok(l) = x.cholesky() {
        return Ok(l.transpose());
    }

    // Regularize: X + ε I.
    let eps = 1e-10;
    let mut x_reg = x.clone();
    for i in 0..n {
        x_reg.set(i, i, x_reg.get(i, i) + eps);
    }
    let l = x_reg.cholesky()?;
    Ok(l.transpose())
}

/// Performs a single random hyperplane rounding.
///
/// Given the `n × n` matrix `V` whose column `i` is the unit vector for
/// vertex `i`, generates a random Gaussian vector `r` and returns the
/// partition `sign(V^T r)` (i.e., `sign(⟨v_i, r⟩)`) as a boolean vector.
///
/// See [`MATH.md`](../MATH.md) §4.
pub fn random_hyperplane_round(vectors: &DenseMatrix, rng: &mut impl Rng) -> Vec<bool> {
    let n = vectors.cols(); // number of vertices
    let dim = vectors.rows(); // dimension of each vector

    // Generate random Gaussian vector r ∈ R^dim via Box-Muller transform.
    let r: Vec<f64> = (0..dim)
        .map(|_| {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect();

    // Compute sign(V^T r): for each vertex i, dot product of column i with r.
    let vdata = vectors.data();
    let mut partition = Vec::with_capacity(n);
    for i in 0..n {
        let col_start = i * dim;
        let col = &vdata[col_start..col_start + dim];
        let mut dot = 0.0_f64;
        for k in 0..dim {
            dot = col[k].mul_add(r[k], dot);
        }
        partition.push(dot >= 0.0);
    }
    partition
}

/// One-flip local search: greedy post-processing of a partition.
///
/// For each vertex, if flipping it to the other side of the partition increases
/// the cut value, flip it. Repeat until no single flip improves the cut.
/// This typically improves cut values by 1-3% over raw hyperplane rounding.
pub fn one_flip_local_search(partition: &mut [bool], graph: &Graph) -> f64 {
    let n = graph.num_vertices();
    let mut improved = true;
    while improved {
        improved = false;
        for v in 0..n {
            // Compute gain from flipping vertex v.
            // For each neighbor u: if same side, flipping v adds the edge to the cut (+w);
            // if different side, flipping v removes it (-w).
            let mut gain = 0.0_f64;
            for &(u, w) in graph.neighbors(v) {
                if partition[u] == partition[v] {
                    gain += w;
                } else {
                    gain -= w;
                }
            }
            if gain > 0.0 {
                partition[v] = !partition[v];
                improved = true;
            }
        }
    }
    graph.cut_value(partition)
}

/// Iterated local search: escapes 1-flip local optima via random perturbation.
///
/// Starting from a partition (not necessarily a local optimum), performs
/// `rounds` iterations of: run one-flip local search to a local optimum,
/// perturb by flipping random vertices, repeat. Uses variable perturbation
/// strength — small perturbations (n/100) most of the time for refinement,
/// with periodic large perturbations (n/10) to escape deep basins.
pub fn iterated_local_search(
    partition: &mut [bool],
    graph: &Graph,
    rounds: usize,
    rng: &mut impl Rng,
) -> f64 {
    let n = graph.num_vertices();
    let k_small = 1_usize.max(n / 100);
    let k_medium = 1_usize.max(n / 10);
    let k_large = 1_usize.max(n / 3);

    let mut best_cut = one_flip_local_search(partition, graph);
    let mut best_partition = partition.to_vec();

    for round in 0..rounds {
        // Tiered perturbation: small most of the time, medium every 5th,
        // large every 20th to break out of deep basins.
        let k = if round % 20 == 19 {
            k_large
        } else if round % 5 == 4 {
            k_medium
        } else {
            k_small
        };

        let mut candidate = best_partition.clone();
        for _ in 0..k {
            let v: usize = rng.random_range(0..n);
            candidate[v] = !candidate[v];
        }

        let cut = one_flip_local_search(&mut candidate, graph);

        if cut > best_cut {
            best_cut = cut;
            best_partition = candidate;
        }
    }

    partition.copy_from_slice(&best_partition);
    best_cut
}

/// Runs multiple rounding trials from pre-extracted vectors.
///
/// This is the core rounding loop used by both the IPM and BM paths.
/// The `vectors` matrix has dimension d × n where column i is vertex i's
/// unit vector (d = n for IPM/Cholesky, d = r for Burer-Monteiro).
///
/// Each trial performs hyperplane rounding, then iterated local search, which
/// alternates one-flip local search with random perturbations.
///
/// With `k = 100` trials, the probability of missing a near-optimal cut is
/// negligibly small. See [`MATH.md`](../MATH.md) §7.
pub fn multi_round_from_vectors(
    vectors: &DenseMatrix,
    graph: &Graph,
    trials: usize,
    ils_rounds: usize,
    sdp_bound: f64,
    rng: &mut impl Rng,
) -> MaxCutResult {
    // Generate deterministic per-trial seeds from the base RNG, then
    // run trials in parallel with independent thread-local RNGs.
    let seeds: Vec<u64> = (0..trials).map(|_| rng.random::<u64>()).collect();

    let (best_partition, best_cut) = seeds
        .par_iter()
        .map(|&seed| {
            let mut trial_rng = StdRng::seed_from_u64(seed);
            let mut partition = random_hyperplane_round(vectors, &mut trial_rng);
            let cut = iterated_local_search(&mut partition, graph, ils_rounds, &mut trial_rng);
            (partition, cut)
        })
        .reduce(
            || (Vec::new(), f64::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        );

    let ratio = if sdp_bound > 1e-15 {
        best_cut / sdp_bound
    } else {
        1.0
    };

    MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: trials,
    }
}

/// Like `multi_round_from_vectors`, but also returns the top-K partitions
/// for use as diverse starting points in refinement.
pub fn multi_round_with_pool(
    vectors: &DenseMatrix,
    graph: &Graph,
    trials: usize,
    ils_rounds: usize,
    sdp_bound: f64,
    pool_size: usize,
    rng: &mut impl Rng,
) -> (MaxCutResult, Vec<(Vec<bool>, f64)>) {
    let seeds: Vec<u64> = (0..trials).map(|_| rng.random::<u64>()).collect();

    let mut all_results: Vec<(Vec<bool>, f64)> = seeds
        .par_iter()
        .map(|&seed| {
            let mut trial_rng = StdRng::seed_from_u64(seed);
            let mut partition = random_hyperplane_round(vectors, &mut trial_rng);
            let cut = iterated_local_search(&mut partition, graph, ils_rounds, &mut trial_rng);
            (partition, cut)
        })
        .collect();

    // Sort descending by cut value, keep top pool_size.
    all_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let best_cut = all_results.first().map_or(f64::NEG_INFINITY, |r| r.1);
    let best_partition = all_results.first().map_or_else(Vec::new, |r| r.0.clone());
    all_results.truncate(pool_size);

    let ratio = if sdp_bound > 1e-15 {
        best_cut / sdp_bound
    } else {
        1.0
    };

    let result = MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: trials,
    };

    (result, all_results)
}

/// Runs multiple rounding trials and returns the best cut found.
///
/// Extracts unit vectors from the PSD matrix `x` via Cholesky, then delegates
/// to [`multi_round_from_vectors`].
///
/// # Errors
///
/// Returns an error if vector extraction fails.
pub fn multi_round(
    x: &DenseMatrix,
    graph: &Graph,
    trials: usize,
    ils_rounds: usize,
    sdp_bound: f64,
    rng: &mut impl Rng,
) -> Result<MaxCutResult, GwError> {
    let vectors = extract_vectors(x)?;
    Ok(multi_round_from_vectors(
        &vectors, graph, trials, ils_rounds, sdp_bound, rng,
    ))
}
