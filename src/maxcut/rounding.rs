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
//! See [`MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/maxcut/MATH.md) §4–§5 for the rounding analysis.

use super::Cut;
use crate::GwError;
use crate::graph::Graph;
use crate::random::box_muller;
use crate::sdp::linalg::{DenseMatrix, col, dot};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_2;
use std::num::NonZeroUsize;
use std::time::Instant;

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
    pub num_trials: NonZeroUsize,
    /// Iterated local search rounds applied to each trial's partition.
    pub ils_rounds: usize,
    /// How each trial rounds the SDP vectors.
    pub rounding: Rounding,
}

impl Default for TrialPlan {
    fn default() -> Self {
        Self {
            num_trials: super::DEFAULT_TRIALS,
            ils_rounds: super::DEFAULT_ILS_ROUNDS,
            rounding: Rounding::default(),
        }
    }
}

/// The `ε` of `X + εI`, which lets Cholesky factor an `X` that is only
/// positive semidefinite.
const CHOLESKY_REGULARIZATION: f64 = 1e-10;

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
pub(crate) fn extract_vectors(x: &DenseMatrix) -> Result<DenseMatrix, GwError> {
    // Try Cholesky directly first.
    if let Ok(l) = x.cholesky() {
        return Ok(l.transpose());
    }

    let mut x_reg = x.clone();
    x_reg.add_to_diagonal(CHOLESKY_REGULARIZATION);
    let l = x_reg.cholesky()?;
    Ok(l.transpose())
}

/// Performs a single random hyperplane rounding.
///
/// Given the `n × n` matrix `V` whose column `i` is the unit vector for
/// vertex `i`, generates a random Gaussian vector `r` and returns the
/// partition `sign(V^T r)` (i.e., `sign(⟨v_i, r⟩)`) as a boolean vector.
///
/// See [`MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/maxcut/MATH.md) §4.
pub(crate) fn random_hyperplane_round(vectors: &DenseMatrix, rng: &mut impl Rng) -> Vec<bool> {
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
/// See [`MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/maxcut/MATH.md) §4.
pub(crate) fn outward_rotation_round(
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
        .map(|_| box_muller(rng.random(), rng.random()).0)
        .collect()
}

/// `⟨v_i, r⟩` for every column `v_i` of `vectors`.
fn projections(vectors: &DenseMatrix, r: &[f64]) -> Vec<f64> {
    (0..vectors.cols())
        .map(|i| dot(col(vectors.data(), i, vectors.rows()), r))
        .collect()
}

/// One-flip local search: greedy post-processing of a partition.
///
/// For each vertex, if flipping it to the other side of the partition increases
/// the cut value, flip it. Repeat until no single flip improves the cut.
/// This typically improves cut values by 1-3% over raw hyperplane rounding.
pub(crate) fn one_flip_local_search(partition: &mut [bool], graph: &Graph) -> f64 {
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

/// Rounds in one cycle of iterated local search's perturbation sizes: the
/// largest comes once per cycle.
pub(crate) const PERTURBATION_CYCLE_ROUNDS: usize = 20;

/// Every this many rounds a medium perturbation replaces a small one.
const MEDIUM_PERTURBATION_PERIOD: usize = 5;

/// A small, medium or large perturbation flips `n` over this many vertices.
const SMALL_PERTURBATION_DIVISOR: usize = 100;
const MEDIUM_PERTURBATION_DIVISOR: usize = 10;
const LARGE_PERTURBATION_DIVISOR: usize = 3;

/// The vertices round `round` of iterated local search flips in an
/// `n`-vertex graph: `n/3` at the end of each cycle, `n/10` every fifth
/// round otherwise, `n/100` the rest, and at least one.
fn perturbation_size(round: usize, n: usize) -> usize {
    let divisor = if round % PERTURBATION_CYCLE_ROUNDS == PERTURBATION_CYCLE_ROUNDS - 1 {
        LARGE_PERTURBATION_DIVISOR
    } else if round % MEDIUM_PERTURBATION_PERIOD == MEDIUM_PERTURBATION_PERIOD - 1 {
        MEDIUM_PERTURBATION_DIVISOR
    } else {
        SMALL_PERTURBATION_DIVISOR
    };
    1.max(n / divisor)
}

/// Flips `count` vertices of `partition` drawn at random, with repeats.
fn perturb(partition: &mut [bool], count: usize, rng: &mut impl Rng) {
    for _ in 0..count {
        let v = rng.random_range(0..partition.len());
        partition[v] = !partition[v];
    }
}

/// Iterated local search: escapes 1-flip local optima via random perturbation.
///
/// Starting from a partition (not necessarily a local optimum), performs
/// `rounds` iterations of: run one-flip local search to a local optimum,
/// perturb by flipping random vertices, repeat. Uses variable perturbation
/// strength to escape deeper basins: n/100 vertices most rounds, n/10 every
/// fifth round, and n/3 once per [`PERTURBATION_CYCLE_ROUNDS`].
pub(crate) fn iterated_local_search(
    partition: &mut [bool],
    graph: &Graph,
    rounds: usize,
    rng: &mut impl Rng,
) -> f64 {
    iterated_local_search_until(partition, graph, rounds, None, rng).0
}

/// [`iterated_local_search`] that also stops starting rounds once
/// `deadline` has passed. Returns the cut and the rounds it ran.
pub(crate) fn iterated_local_search_until(
    partition: &mut [bool],
    graph: &Graph,
    rounds: usize,
    deadline: Option<Instant>,
    rng: &mut impl Rng,
) -> (f64, usize) {
    let n = graph.num_vertices();
    let mut best_cut = one_flip_local_search(partition, graph);
    let mut best_partition = partition.to_vec();

    let mut rounds_run = 0;
    for round in 0..rounds {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
        rounds_run += 1;
        let mut candidate = best_partition.clone();
        perturb(&mut candidate, perturbation_size(round, n), rng);
        let cut = one_flip_local_search(&mut candidate, graph);

        if cut > best_cut {
            best_cut = cut;
            best_partition = candidate;
        }
    }

    partition.copy_from_slice(&best_partition);
    (best_cut, rounds_run)
}

/// The cuts of every rounding trial, in trial order: never empty, as
/// `plan.num_trials` is not zero.
///
/// The `vectors` matrix is `d × n`, column `i` the unit vector of vertex
/// `i` (`d = n` from the IPM's Cholesky factor, `d = r` from BM). Each
/// trial rounds them as `plan.rounding` says and improves the partition by
/// iterated local search. Its seed is drawn from `rng` up front, so trials
/// run in parallel yet a seeded run is reproducible.
///
/// See [`MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/maxcut/MATH.md) §7.
pub(crate) fn round_trials(
    vectors: &DenseMatrix,
    graph: &Graph,
    plan: TrialPlan,
    rng: &mut impl Rng,
) -> Vec<Cut> {
    let seeds: Vec<u64> = (0..plan.num_trials.get())
        .map(|_| rng.random::<u64>())
        .collect();
    seeds
        .par_iter()
        .map(|&seed| {
            let mut trial_rng = StdRng::seed_from_u64(seed);
            let mut partition = plan.rounding.partition(vectors, &mut trial_rng);
            let cut_value =
                iterated_local_search(&mut partition, graph, plan.ils_rounds, &mut trial_rng);
            Cut {
                partition,
                cut_value,
            }
        })
        .collect()
}

/// The `pool_size` best of `cuts`, best first, equal cuts in their order in
/// `cuts`.
pub(crate) fn best_cuts(mut cuts: Vec<Cut>, pool_size: usize) -> Vec<Cut> {
    cuts.sort_by(|a, b| b.cut_value.total_cmp(&a.cut_value));
    cuts.truncate(pool_size);
    cuts
}

/// Rounds the PSD matrix `x`'s Cholesky vectors and returns the best cut
/// of [`round_trials`], the earliest among equals.
///
/// # Errors
///
/// Returns an error if vector extraction fails.
pub(crate) fn multi_round(
    x: &DenseMatrix,
    graph: &Graph,
    plan: TrialPlan,
    rng: &mut impl Rng,
) -> Result<Cut, GwError> {
    let vectors = extract_vectors(x)?;
    let mut cuts = round_trials(&vectors, graph, plan, rng);
    let best = first_best(&cuts);
    Ok(cuts.swap_remove(best))
}

/// The index of the first of the best cuts in the non-empty `cuts`.
fn first_best(cuts: &[Cut]) -> usize {
    (1..cuts.len()).fold(0, |best, i| {
        if cuts[i].cut_value.total_cmp(&cuts[best].cut_value).is_gt() {
            i
        } else {
            best
        }
    })
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

    /// About `degree` neighbours per vertex, with weights 1 to 10.
    fn random_graph(n: usize, degree: f64, seed: u64) -> Graph {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut graph = Graph::new(n);
        for u in 0..n {
            for v in u + 1..n {
                if rng.random::<f64>() < degree / n as f64 {
                    let weight = f64::from(rng.random_range(1..=10_u32));
                    graph
                        .add_edge(u, v, weight)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
            }
        }
        graph
    }

    fn random_partition(n: usize, seed: u64) -> Vec<bool> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random()).collect()
    }

    fn plan(trials: usize, ils_rounds: usize) -> TrialPlan {
        TrialPlan {
            num_trials: NonZeroUsize::new(trials).unwrap_or_else(|| panic!("{trials} trials")),
            ils_rounds,
            rounding: Rounding::Hyperplane,
        }
    }

    /// The cut gained by moving `v` to the other side.
    fn flip_gain(graph: &Graph, partition: &[bool], v: usize) -> f64 {
        graph
            .neighbors(v)
            .iter()
            .map(|&(u, w)| if partition[u] == partition[v] { w } else { -w })
            .sum()
    }

    #[test]
    fn one_flip_search_ends_where_no_flip_helps_and_reports_that_cut() {
        let graph = random_graph(200, 6.0, 1);
        let mut partition = random_partition(200, 2);

        let cut = one_flip_local_search(&mut partition, &graph);

        assert_eq!(cut, graph.cut_value(&partition));
        let improving: Vec<usize> = (0..200)
            .filter(|&v| flip_gain(&graph, &partition, v) > 0.0)
            .collect();
        assert_eq!(improving, Vec::<usize>::new());
    }

    #[test]
    fn iterated_search_is_never_worse_than_its_first_local_optimum() {
        let graph = random_graph(200, 6.0, 3);
        let start = random_partition(200, 4);
        let first_optimum = one_flip_local_search(&mut start.clone(), &graph);

        let mut partition = start;
        let cut = iterated_local_search(&mut partition, &graph, 30, &mut StdRng::seed_from_u64(5));

        assert!(cut >= first_optimum, "{cut} < {first_optimum}");
        assert_eq!(cut, graph.cut_value(&partition));
    }

    #[test]
    fn the_pool_holds_the_best_cuts_best_first() {
        let graph = random_graph(60, 4.0, 6);
        let vectors = random_vectors(8, 60, 7);
        let cuts = round_trials(&vectors, &graph, plan(20, 2), &mut StdRng::seed_from_u64(8));
        let best_value = cuts
            .iter()
            .map(|cut| cut.cut_value)
            .fold(f64::MIN, f64::max);

        let pool = best_cuts(cuts, 5);

        assert_eq!(pool.len(), 5);
        assert!(
            pool.windows(2)
                .all(|pair| pair[0].cut_value >= pair[1].cut_value)
        );
        assert_eq!(pool[0].cut_value, best_value);
    }

    #[test]
    fn the_pool_and_the_best_cut_agree_on_the_earliest_best_trial() {
        let graph = random_graph(60, 4.0, 9);
        let vectors = random_vectors(8, 60, 10);
        let cuts = round_trials(
            &vectors,
            &graph,
            plan(20, 0),
            &mut StdRng::seed_from_u64(11),
        );

        let best = &cuts[first_best(&cuts)];
        let pool = best_cuts(cuts.clone(), 5);

        assert_eq!(best.partition, pool[0].partition);
        assert!(
            cuts.iter()
                .take(first_best(&cuts))
                .all(|cut| cut.cut_value < best.cut_value)
        );
    }

    #[test]
    fn perturbations_grow_every_fifth_round_and_peak_once_per_cycle() {
        let sizes: Vec<usize> = (0..PERTURBATION_CYCLE_ROUNDS)
            .map(|round| perturbation_size(round, 3000))
            .collect();

        let mut expected = vec![30; PERTURBATION_CYCLE_ROUNDS];
        for medium in [4, 9, 14] {
            expected[medium] = 300;
        }
        expected[19] = 1000;
        assert_eq!(sizes, expected);
        assert_eq!(perturbation_size(PERTURBATION_CYCLE_ROUNDS, 3000), 30);
    }

    #[test]
    fn a_small_graph_still_perturbs_one_vertex() {
        assert_eq!(perturbation_size(0, 5), 1);
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
