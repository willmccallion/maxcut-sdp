//! Goemans-Williamson Max-Cut approximation algorithm.
//!
//! Orchestrates the full GW pipeline:
//! 1. Formulate the Max-Cut SDP relaxation from the graph Laplacian
//! 2. Solve the SDP by interior point method or Burer-Monteiro factorization
//! 3. Take unit vectors from the solution (a Cholesky factor of the IPM's `X`,
//!    or BM's factor itself)
//! 4. Run multiple random roundings, each improved by iterated local search
//! 5. Return the best partition found
//!
//! On graphs with non-negative weights, rounding achieves an expected
//! approximation ratio of `α_GW ≈ 0.878`, which is optimal under the Unique
//! Games Conjecture.
//!
//! See [`MATH.md`](./MATH.md) for the full mathematical derivation.

pub mod exact;
pub mod rounding;

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use crate::GwError;
use crate::graph::Graph;
use crate::maxcut::rounding::{Rounding, TrialPlan};
use crate::sdp::bm::{BmSolver, BmSolverConfig};
use crate::sdp::ipm::{IpmConfig, IpmSolver};
use crate::sdp::linalg::{DenseMatrix, col, dot};
use crate::sdp::{SdpProblem, maxcut_cost};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// A partition of the vertices and the weight of the edges it cuts.
#[derive(Debug, Clone)]
pub struct Cut {
    /// `partition[i] = true` means vertex `i` is in set `S`.
    pub partition: Vec<bool>,
    /// The total weight of edges crossing the partition.
    pub cut_value: f64,
}

/// The result of running the Goemans-Williamson algorithm.
#[derive(Debug, Clone)]
pub struct MaxCutResult {
    /// The vertex partition: `partition[i] = true` means vertex `i` is in set `S`.
    pub partition: Vec<bool>,
    /// The cut value (total weight of edges crossing the partition).
    pub cut_value: f64,
    /// What the SDP relaxation established about the optimal cut.
    pub sdp_bound: SdpBound,
    /// The approximation ratio `cut_value / sdp_bound`, or `None` when the
    /// bound is uncertified or zero and no ratio exists.
    pub approximation_ratio: Option<f64>,
    /// Number of rounding trials performed.
    pub num_trials: usize,
}

/// What the SDP stage established about the maximum cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdpBound {
    /// An upper bound on every cut, certified from a dual solution.
    Certified(f64),
    /// The relaxation's objective where the solver stopped. It lies at or
    /// below the SDP's value, so it bounds no cut; BM reports this for
    /// graphs past [`MAX_CERTIFIED_VERTICES`].
    Relaxation(f64),
}

impl SdpBound {
    /// The bound or relaxation value.
    pub const fn value(self) -> f64 {
        match self {
            Self::Certified(value) | Self::Relaxation(value) => value,
        }
    }

    /// `cut / bound` for a graph whose total absolute weight is `scale`, or
    /// `None` when the bound is uncertified or zero.
    fn ratio(self, cut: f64, scale: f64) -> Option<f64> {
        match self {
            Self::Certified(bound) => cut_ratio(cut, bound, scale),
            Self::Relaxation(_) => None,
        }
    }
}

/// Largest graph whose BM bound is certified. The certificate's dense
/// eigenvalue takes about 3 s and two 72 MB matrices at this size, and grows
/// as `n³` past it.
pub const MAX_CERTIFIED_VERTICES: usize = 3000;

/// Bounds within this fraction of a graph's total absolute weight of zero
/// are numerical noise, not a bound a ratio can be taken against.
const ZERO_BOUND_TOLERANCE: f64 = 1e-6;

/// `cut / reference` for a graph whose total absolute weight is `scale`, or
/// `None` when `reference` is zero up to noise, where no ratio is defined.
pub fn cut_ratio(cut: f64, reference: f64, scale: f64) -> Option<f64> {
    (reference > ZERO_BOUND_TOLERANCE * scale).then(|| cut / reference)
}

/// An upper bound on the Max-Cut SDP from any dual `y`, feasible or not.
///
/// With `S = C − Diag(y)`, every feasible `X` has `tr(C X) = Σ yᵢ + tr(S X)`,
/// and `tr(S X) ≥ n·λ_min(S)` because `tr X = n`. So the SDP value
/// `−min tr(C X)` is at most `−Σ yᵢ + n·max(0, −λ_min(S))`, which is the
/// dual objective when `y` is feasible and corrects it by what `y` lacks
/// when it is not. Exact up to the floating-point accuracy of `λ_min`.
fn certified_bound(cost: DenseMatrix, y: &[f64]) -> Result<f64, GwError> {
    let mut s = cost;
    s.sub_diagonal(y);
    let lambda_min = s.min_eigenvalue()?;
    let dual_objective: f64 = y.iter().sum();
    Ok((y.len() as f64).mul_add((-lambda_min).max(0.0), -dual_objective))
}

/// The dual `y` at which `X = VᵀV` meets the Max-Cut SDP's optimality
/// condition `(C − Diag(y)) X = 0`: `yᵢ = (C X)ᵢᵢ`, for `C = −L/4` and the
/// unit columns `vᵢ` of `vectors`, so `yᵢ = Σⱼ (wᵢⱼ/4)(⟨vᵢ, vⱼ⟩ − 1)`.
fn dual_from_vectors(graph: &Graph, vectors: &DenseMatrix) -> Vec<f64> {
    let rank = vectors.rows();
    let mut y = vec![0.0; graph.num_vertices()];
    for &(u, v, w) in graph.edges() {
        let inner = dot(col(vectors.data(), u, rank), col(vectors.data(), v, rank));
        let term = 0.25 * w * (inner - 1.0);
        y[u] += term;
        y[v] += term;
    }
    y
}

/// The bound BM's best `vectors`, of relaxation value `relaxation`, give:
/// certified from their dual up to [`MAX_CERTIFIED_VERTICES`], the
/// relaxation value past it.
fn bm_bound(graph: &Graph, relaxation: f64, vectors: &DenseMatrix) -> Result<SdpBound, GwError> {
    if graph.num_vertices() > MAX_CERTIFIED_VERTICES {
        return Ok(SdpBound::Relaxation(relaxation));
    }
    let y = dual_from_vectors(graph, vectors);
    Ok(SdpBound::Certified(certified_bound(
        maxcut_cost(graph),
        &y,
    )?))
}

/// Whether a step expected to take `expected`, started at `now`, finishes by
/// `deadline`; always when there is none.
fn fits_before(deadline: Option<Instant>, now: Instant, expected: Duration) -> bool {
    deadline.is_none_or(|deadline| now + expected <= deadline)
}

/// Rounding trials the configurations run by default.
pub const DEFAULT_TRIALS: NonZeroUsize = match NonZeroUsize::new(100) {
    Some(trials) => trials,
    None => NonZeroUsize::MIN,
};

/// BM restarts the configurations run by default.
pub const DEFAULT_RESTARTS: NonZeroUsize = match NonZeroUsize::new(3) {
    Some(restarts) => restarts,
    None => NonZeroUsize::MIN,
};

/// Configuration for the GW algorithm.
#[derive(Debug, Clone)]
pub struct GwConfig {
    /// Number of independent rounding trials.
    pub num_trials: NonZeroUsize,
    /// Number of iterated local search rounds per rounding trial.
    pub ils_rounds: usize,
    /// How each trial rounds the SDP vectors.
    pub rounding: Rounding,
    /// Configuration for the underlying SDP solver.
    pub solver_config: IpmConfig,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            num_trials: DEFAULT_TRIALS,
            ils_rounds: 50,
            rounding: Rounding::default(),
            solver_config: IpmConfig::default(),
            seed: None,
        }
    }
}

/// Configuration for the GW algorithm using the Burer-Monteiro solver.
#[derive(Debug, Clone)]
pub struct GwBmConfig {
    /// Number of independent rounding trials per BM restart.
    pub num_trials: NonZeroUsize,
    /// Number of iterated local search rounds per rounding trial.
    pub ils_rounds: usize,
    /// How each trial rounds the SDP vectors.
    pub rounding: Rounding,
    /// Configuration for the BM solver.
    pub solver_config: BmSolverConfig,
    /// BM solves from independent random starts, each rounded separately.
    pub num_restarts: NonZeroUsize,
    /// Optional RNG seed for reproducibility (used for both solver and rounding).
    pub seed: Option<u64>,
    /// Optional time budget. When set, a restart after the first starts only
    /// if, taking as long as the one before, it would finish within the
    /// budget, and whatever time is left refines the best cuts with iterated
    /// local search. The first restart always completes, so a budget shorter
    /// than one solve is exceeded by it.
    pub time_limit: Option<Duration>,
}

impl Default for GwBmConfig {
    fn default() -> Self {
        Self {
            num_trials: DEFAULT_TRIALS,
            ils_rounds: 50,
            rounding: Rounding::default(),
            solver_config: BmSolverConfig::default(),
            num_restarts: DEFAULT_RESTARTS,
            seed: None,
            time_limit: None,
        }
    }
}

/// Runs the full Goemans-Williamson Max-Cut algorithm.
///
/// Given a weighted undirected graph, this function:
/// 1. Builds the SDP relaxation (cost matrix `C = -(1/4)L`, diagonal constraints)
/// 2. Solves the SDP via primal-dual interior point method
/// 3. Rounds the solution `k` times using random hyperplanes
/// 4. Returns the best partition found
///
/// See [`MATH.md`](./MATH.md) §10 for the implementation checklist.
///
/// # Errors
///
/// Returns an error if the SDP solver fails or the graph is empty.
pub fn gw_maxcut(graph: &Graph, config: &GwConfig) -> Result<MaxCutResult, GwError> {
    // 1. Formulate SDP.
    let problem = SdpProblem::maxcut_relaxation(graph)?;

    // 2. Solve SDP.
    let solver = IpmSolver::new(config.solver_config.clone());
    let solution = solver.solve(&problem)?;

    // The Max-Cut SDP maximizes (1/4) tr(LX) = -tr(CX). The converged dual
    // is feasible only to within the tolerance, so its objective is certified.
    let sdp_bound = SdpBound::Certified(certified_bound(problem.into_cost(), &solution.y)?);

    // 3. Round with random hyperplanes.
    let mut rng = config
        .seed
        .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

    let plan = TrialPlan {
        trials: config.num_trials,
        ils_rounds: config.ils_rounds,
        rounding: config.rounding,
    };
    let best = rounding::multi_round(&solution.x, graph, plan, &mut rng)?;
    let ratio = sdp_bound.ratio(best.cut_value, graph.total_abs_weight());

    Ok(MaxCutResult {
        partition: best.partition,
        cut_value: best.cut_value,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: config.num_trials.get(),
    })
}

/// Runs the GW Max-Cut algorithm using the Burer-Monteiro low-rank solver.
///
/// This is the recommended path for large or sparse graphs (n > 100).
/// Per-iteration cost is O(|E|·r) instead of O(n³) for the dense IPM.
///
/// Each BM restart produces a different SDP solution with different vector
/// geometry. Rounding is performed from each restart's vectors independently,
/// and the best cut across all restarts is returned.
///
/// # Errors
///
/// Returns an error if the graph is empty or the solver fails.
pub fn gw_maxcut_bm(graph: &Graph, config: &GwBmConfig) -> Result<MaxCutResult, GwError> {
    let mut rng = config
        .seed
        .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

    // Run each restart independently: solve BM, then round from those vectors.
    // Different restarts produce different vector geometries, giving the rounding
    // more diverse starting points.
    let solver = BmSolver::new(config.solver_config.clone());

    let pool_size = 8;
    let start = Instant::now();
    let mut best_cut = f64::NEG_INFINITY;
    let mut best_partition: Vec<bool> = Vec::new();
    let mut best_relaxation: Option<(f64, DenseMatrix)> = None;
    let mut total_trials = 0_usize;

    // Collect top partitions from all restarts for the refinement pool.
    let mut candidate_pool: Vec<Cut> = Vec::new();

    let deadline = config.time_limit.map(|limit| start + limit);
    let mut last_restart = Duration::ZERO;
    let num_restarts = config.num_restarts.get();
    for restart in 0..num_restarts {
        let restart_start = Instant::now();
        if restart > 0 && !fits_before(deadline, restart_start, last_restart) {
            eprintln!("Time limit: stopping after {restart}/{num_restarts} restarts");
            break;
        }
        let ctx = if best_cut > f64::NEG_INFINITY {
            Some(best_cut)
        } else {
            None
        };
        let solution = solver.solve_maxcut_with_context(graph, &mut rng, ctx)?;
        let relaxation = -solution.objective;

        let plan = TrialPlan {
            trials: config.num_trials,
            ils_rounds: config.ils_rounds,
            rounding: config.rounding,
        };
        let (result, top_partitions) =
            rounding::multi_round_with_pool(&solution.vectors, graph, plan, pool_size, &mut rng);
        total_trials += config.num_trials.get();
        candidate_pool.extend(top_partitions);

        if result.cut_value > best_cut {
            best_cut = result.cut_value;
            best_partition.clone_from(&result.partition);
        }

        eprintln!(
            "Restart {}/{num_restarts}: relaxation={relaxation:.4}, cut={:.0}, best={best_cut:.0} ({:.1?})",
            restart + 1,
            result.cut_value,
            start.elapsed()
        );
        if best_relaxation
            .as_ref()
            .is_none_or(|(best, _)| relaxation > *best)
        {
            best_relaxation = Some((relaxation, solution.vectors));
        }
        last_restart = restart_start.elapsed();
    }

    let Some((relaxation, vectors)) = best_relaxation else {
        return Err(GwError::SolverFailed(
            "BM solver produced no solution".into(),
        ));
    };
    let sdp_bound = bm_bound(graph, relaxation, &vectors)?;
    if let SdpBound::Certified(bound) = sdp_bound {
        eprintln!("Certified bound: {bound:.4} ({:.1?})", start.elapsed());
    }

    // Refinement phase: if a time budget is set and time remains, maintain a
    // population of diverse partitions and refine them in parallel.
    if let Some(deadline) = deadline {
        // Build pool from the best partitions across all restarts.
        candidate_pool.sort_by(|a, b| b.cut_value.total_cmp(&a.cut_value));
        candidate_pool.truncate(pool_size);

        // Give each pool member its own RNG for parallel execution.
        let mut pool: Vec<(Vec<bool>, f64, StdRng)> = candidate_pool
            .into_iter()
            .map(|cut| {
                let seed: u64 = rng.random();
                (cut.partition, cut.cut_value, StdRng::seed_from_u64(seed))
            })
            .collect();

        let mut rounds_done = 0_usize;
        let mut last_report = start.elapsed();
        let report_interval = Duration::from_secs(10);
        while Instant::now() < deadline {
            // Run ILS on all pool members in parallel, each stopping at the deadline.
            let rounds: usize = pool
                .par_iter_mut()
                .map(|(partition, cut, slot_rng)| {
                    let (new_cut, rounds) = rounding::iterated_local_search_until(
                        partition,
                        graph,
                        rounding::PERTURBATION_CYCLE_ROUNDS,
                        Some(deadline),
                        slot_rng,
                    );
                    *cut = new_cut;
                    rounds
                })
                .sum();
            rounds_done += rounds;

            // Check for improvements across all slots.
            for (slot, (partition, cut, _)) in pool.iter().enumerate() {
                if *cut > best_cut {
                    best_cut = *cut;
                    best_partition.clone_from(partition);
                    eprintln!(
                        "  Refine: new best={best_cut:.0} (slot {slot}, {:.1?})",
                        start.elapsed()
                    );
                }
            }

            // Periodic progress report.
            let now = start.elapsed();
            if now.saturating_sub(last_report) >= report_interval {
                eprintln!("  Refine: best={best_cut:.0}, {rounds_done} rounds ({now:.1?})");
                last_report = now;
            }
        }

        if rounds_done > 0 {
            eprintln!(
                "Refinement done: {rounds_done} ILS rounds, best={best_cut:.0} ({:.1?})",
                start.elapsed()
            );
        }
    }

    let ratio = sdp_bound.ratio(best_cut, graph.total_abs_weight());

    Ok(MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound,
        approximation_ratio: ratio,
        num_trials: total_trials,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECOND: Duration = Duration::from_secs(1);

    #[test]
    fn a_step_fits_when_it_ends_by_the_deadline() {
        let now = Instant::now();

        assert!(fits_before(Some(now + 2 * SECOND), now, SECOND));
    }

    #[test]
    fn a_step_does_not_fit_when_it_would_end_after_the_deadline() {
        let now = Instant::now();

        assert!(!fits_before(Some(now + SECOND), now, 2 * SECOND));
    }

    fn five_cycle() -> Graph {
        let mut g = Graph::new(5);
        for (u, v) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)] {
            g.add_edge(u, v, 1.0).unwrap_or_else(|e| panic!("{e}"));
        }
        g
    }

    #[test]
    fn the_certificate_of_a_zero_dual_is_the_eigenvalue_bound() {
        let graph = five_cycle();
        let cost = maxcut_cost(&graph);
        let lambda_max = -cost.min_eigenvalue().unwrap_or_else(|e| panic!("{e}"));

        let bound = certified_bound(cost, &[0.0; 5]);

        let bound = bound.unwrap_or_else(|e| panic!("{e}"));
        assert!(
            (-5.0_f64).mul_add(lambda_max, bound).abs() < 1e-9,
            "{bound} vs {}",
            5.0 * lambda_max
        );
    }

    #[test]
    fn the_ipm_bound_is_certified_and_at_least_the_optimum() {
        let graph = five_cycle();
        let problem = SdpProblem::maxcut_relaxation(&graph).unwrap_or_else(|e| panic!("{e}"));
        let solution = IpmSolver::new(IpmConfig::default())
            .solve(&problem)
            .unwrap_or_else(|e| panic!("{e}"));

        let bound =
            certified_bound(problem.into_cost(), &solution.y).unwrap_or_else(|e| panic!("{e}"));

        assert!(
            bound >= 4.0,
            "the 5-cycle's maximum cut is 4; bound {bound}"
        );
        assert!(
            (bound + solution.dual_obj).abs() < 1e-6,
            "{bound} vs dual {}",
            -solution.dual_obj
        );
    }

    #[test]
    fn every_step_fits_without_a_deadline() {
        assert!(fits_before(None, Instant::now(), Duration::MAX / 2));
    }
}
