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
//! Trials can instead round with Zwick's outward rotations; see [`Rounding`].
//!
//! See [`MATH.md`](../MATH.md) §4–§5 for the rounding analysis.

use super::MaxCutResult;
use crate::GwError;
use crate::graph::Graph;
use crate::sdp::linalg::DenseMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::num::NonZeroUsize;

/// An outward rotation angle `θ` in radians, within `[0, π/2]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationAngle(f64);

impl RotationAngle {
    /// Returns `None` unless `radians` is a finite angle in `[0, π/2]`.
    pub fn new(radians: f64) -> Option<Self> {
        (0.0..=FRAC_PI_2)
            .contains(&radians)
            .then_some(Self(radians))
    }

    /// The angle in radians.
    pub const fn radians(self) -> f64 {
        self.0
    }
}

/// How a rounding trial turns the SDP vectors into a partition.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Rounding {
    /// Goemans-Williamson random hyperplane rounding.
    #[default]
    Hyperplane,
    /// Zwick's outward rotation by the given angle, then a random hyperplane.
    OutwardRotation(RotationAngle),
}

impl Rounding {
    fn partition(self, vectors: &DenseMatrix, rng: &mut impl Rng) -> Vec<bool> {
        match self {
            Self::Hyperplane => random_hyperplane_round(vectors, rng),
            Self::OutwardRotation(angle) => outward_rotation_round(vectors, angle, rng),
        }
    }
}

/// What every rounding trial does.
#[derive(Debug, Clone, Copy)]
pub struct TrialPlan {
    /// Number of independent rounding trials.
    pub trials: NonZeroUsize,
    /// Iterated local search rounds applied to each trial's partition.
    pub ils_rounds: usize,
    /// How each trial rounds the SDP vectors.
    pub rounding: Rounding,
}

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
    let r = gaussian_vector(vectors.rows(), rng);
    projections(vectors, &r)
        .into_iter()
        .map(|dot| dot >= 0.0)
        .collect()
}

/// Performs a single outward rotation rounding (Zwick 1999).
///
/// Rotates each vector `v_i` by `θ` towards its own new orthogonal direction
/// `e_i`, giving `v_i' = cos θ · v_i + sin θ · e_i`, and cuts the rotated
/// vectors with a random hyperplane `(r, g)`: vertex `i` goes to `S` when
/// `cos θ · ⟨v_i, r⟩ + sin θ · g_i ≥ 0`. For `i ≠ j`, `⟨v_i', v_j'⟩ =
/// cos² θ · X_ij`, so the pair is separated with probability
/// `arccos(cos² θ · X_ij) / π`. `θ = 0` is hyperplane rounding and
/// `θ = π/2` a uniformly random cut.
///
/// See [`MATH.md`](../MATH.md) §4.
pub fn outward_rotation_round(
    vectors: &DenseMatrix,
    angle: RotationAngle,
    rng: &mut impl Rng,
) -> Vec<bool> {
    let (sin, cos) = angle.radians().sin_cos();
    let r = gaussian_vector(vectors.rows(), rng);
    let g = gaussian_vector(vectors.cols(), rng);
    projections(vectors, &r)
        .into_iter()
        .zip(g)
        .map(|(dot, g_i)| cos.mul_add(dot, sin * g_i) >= 0.0)
        .collect()
}

/// Independent standard normal samples, via the Box-Muller transform.
fn gaussian_vector(len: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..len)
        .map(|_| {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
        })
        .collect()
}

/// `⟨v_i, r⟩` for every column `v_i` of `vectors`.
fn projections(vectors: &DenseMatrix, r: &[f64]) -> Vec<f64> {
    let dim = vectors.rows();
    let vdata = vectors.data();
    let mut dots = Vec::with_capacity(vectors.cols());
    for i in 0..vectors.cols() {
        let col = &vdata[i * dim..(i + 1) * dim];
        let mut dot = 0.0_f64;
        for k in 0..dim {
            dot = col[k].mul_add(r[k], dot);
        }
        dots.push(dot);
    }
    dots
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
/// Each trial rounds the vectors as `plan.rounding` says, then runs iterated
/// local search, which alternates one-flip local search with random
/// perturbations.
///
/// With `k = 100` trials, the probability of missing a near-optimal cut is
/// negligibly small. See [`MATH.md`](../MATH.md) §7.
pub fn multi_round_from_vectors(
    vectors: &DenseMatrix,
    graph: &Graph,
    plan: TrialPlan,
    sdp_bound: f64,
    rng: &mut impl Rng,
) -> MaxCutResult {
    // Generate deterministic per-trial seeds from the base RNG, then
    // run trials in parallel with independent thread-local RNGs.
    let seeds: Vec<u64> = (0..plan.trials.get())
        .map(|_| rng.random::<u64>())
        .collect();

    // Trials are indexed so that equal cuts go to the earliest, whatever
    // order rayon combines them in.
    let (_, best_partition, best_cut) = seeds
        .par_iter()
        .enumerate()
        .map(|(trial, &seed)| {
            let mut trial_rng = StdRng::seed_from_u64(seed);
            let mut partition = plan.rounding.partition(vectors, &mut trial_rng);
            let cut = iterated_local_search(&mut partition, graph, plan.ils_rounds, &mut trial_rng);
            (trial, partition, cut)
        })
        .reduce(
            || (usize::MAX, Vec::new(), f64::NEG_INFINITY),
            |a, b| match b.2.total_cmp(&a.2) {
                Ordering::Greater => b,
                Ordering::Equal if b.0 < a.0 => b,
                _ => a,
            },
        );

    let ratio = super::cut_ratio(best_cut, sdp_bound, graph.total_abs_weight());

    MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: plan.trials.get(),
    }
}

/// Like `multi_round_from_vectors`, but also returns the top-K partitions
/// for use as diverse starting points in refinement.
pub fn multi_round_with_pool(
    vectors: &DenseMatrix,
    graph: &Graph,
    plan: TrialPlan,
    sdp_bound: f64,
    pool_size: usize,
    rng: &mut impl Rng,
) -> (MaxCutResult, Vec<(Vec<bool>, f64)>) {
    let seeds: Vec<u64> = (0..plan.trials.get())
        .map(|_| rng.random::<u64>())
        .collect();

    let mut all_results: Vec<(Vec<bool>, f64)> = seeds
        .par_iter()
        .map(|&seed| {
            let mut trial_rng = StdRng::seed_from_u64(seed);
            let mut partition = plan.rounding.partition(vectors, &mut trial_rng);
            let cut = iterated_local_search(&mut partition, graph, plan.ils_rounds, &mut trial_rng);
            (partition, cut)
        })
        .collect();

    // Sort descending by cut value, equal cuts in trial order, keep top pool_size.
    all_results.sort_by(|a, b| b.1.total_cmp(&a.1));
    let best_cut = all_results.first().map_or(f64::NEG_INFINITY, |r| r.1);
    let best_partition = all_results.first().map_or_else(Vec::new, |r| r.0.clone());
    all_results.truncate(pool_size);

    let ratio = super::cut_ratio(best_cut, sdp_bound, graph.total_abs_weight());

    let result = MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: plan.trials.get(),
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
    plan: TrialPlan,
    sdp_bound: f64,
    rng: &mut impl Rng,
) -> Result<MaxCutResult, GwError> {
    let vectors = extract_vectors(x)?;
    Ok(multi_round_from_vectors(
        &vectors, graph, plan, sdp_bound, rng,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(dim: usize, n: usize, seed: u64) -> DenseMatrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let data = (0..dim * n).map(|_| rng.random::<f64>() - 0.5).collect();
        let Ok(vectors) = DenseMatrix::from_vec(dim, n, data) else {
            panic!("dim * n values fill a dim x n matrix");
        };
        vectors
    }

    fn angle(radians: f64) -> RotationAngle {
        let Some(angle) = RotationAngle::new(radians) else {
            panic!("{radians} is a valid rotation angle");
        };
        angle
    }

    #[test]
    fn rotation_angle_accepts_only_zero_to_half_pi() {
        let accepted = [0.0, 0.5, FRAC_PI_2].map(|r| RotationAngle::new(r).is_some());
        let rejected = [-0.01, FRAC_PI_2 + 0.01, f64::NAN, f64::INFINITY]
            .map(|r| RotationAngle::new(r).is_none());

        assert_eq!(accepted, [true; 3]);
        assert_eq!(rejected, [true; 4]);
    }

    #[test]
    fn zero_rotation_is_hyperplane_rounding() {
        let vectors = random_vectors(5, 200, 1);

        let hyperplane = random_hyperplane_round(&vectors, &mut StdRng::seed_from_u64(7));
        let rotated = outward_rotation_round(&vectors, angle(0.0), &mut StdRng::seed_from_u64(7));

        assert_eq!(hyperplane, rotated);
    }

    #[test]
    fn right_angle_rotation_ignores_the_vectors() {
        let first = random_vectors(5, 200, 1);
        let second = random_vectors(5, 200, 2);

        let from_first =
            outward_rotation_round(&first, angle(FRAC_PI_2), &mut StdRng::seed_from_u64(7));
        let from_second =
            outward_rotation_round(&second, angle(FRAC_PI_2), &mut StdRng::seed_from_u64(7));

        assert_eq!(from_first, from_second);
    }

    #[test]
    fn identical_vectors_separate_with_probability_arccos_cos_squared_over_pi() {
        let Ok(vectors) = DenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 1.0, 0.0]) else {
            panic!("four values fill a 2 x 2 matrix");
        };
        let theta = std::f64::consts::FRAC_PI_4;
        let expected = theta.cos().powi(2).acos() / std::f64::consts::PI;
        let mut rng = StdRng::seed_from_u64(3);
        let samples = 20_000;

        let separated = (0..samples)
            .filter(|_| {
                let partition = outward_rotation_round(&vectors, angle(theta), &mut rng);
                partition[0] != partition[1]
            })
            .count();

        let frequency = separated as f64 / f64::from(samples);
        assert!(
            (frequency - expected).abs() < 0.015,
            "separated {frequency}, expected {expected}"
        );
    }
}
