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

mod bm_pipeline;
pub mod exact;
pub mod rounding;

use std::num::NonZeroUsize;
use std::time::Duration;

use crate::GwError;
use crate::graph::Graph;
use crate::maxcut::bm_pipeline::{BmPipeline, Budget};
use crate::maxcut::rounding::{Rounding, TrialPlan};
use crate::sdp::bm::{BmProgress, BmSolverConfig};
use crate::sdp::ipm::{IpmConfig, IpmSolver};
use crate::sdp::linalg::{DenseMatrix, col, dot};
use crate::sdp::{SdpProblem, maxcut_cost};
use rand::SeedableRng;
use rand::rngs::StdRng;

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
    let problem = SdpProblem::maxcut_relaxation(graph)?;
    let solution = IpmSolver::new(config.solver_config.clone()).solve(&problem)?;
    // The converged dual is feasible only to within the tolerance, so its
    // objective is certified rather than taken as it stands.
    let sdp_bound = SdpBound::Certified(certified_bound(problem.into_cost(), &solution.y)?);

    let mut rng = seeded_rng(config.seed);
    let plan = TrialPlan {
        trials: config.num_trials,
        ils_rounds: config.ils_rounds,
        rounding: config.rounding,
    };
    let best = rounding::multi_round(&solution.x, graph, plan, &mut rng)?;
    Ok(max_cut_result(graph, best, sdp_bound, plan.trials.get()))
}

/// Runs the GW Max-Cut algorithm using the Burer-Monteiro low-rank solver,
/// reporting progress to `on_progress`.
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
pub fn gw_maxcut_bm(
    graph: &Graph,
    config: &GwBmConfig,
    on_progress: impl FnMut(Progress),
) -> Result<MaxCutResult, GwError> {
    let mut rng = seeded_rng(config.seed);
    let plan = TrialPlan {
        trials: config.num_trials,
        ils_rounds: config.ils_rounds,
        rounding: config.rounding,
    };
    let budget = Budget::start(config.time_limit);
    let mut pipeline = BmPipeline::new(
        graph,
        config.solver_config.clone(),
        plan,
        budget,
        on_progress,
    );

    let restarts = pipeline.run_restarts(config.num_restarts.get(), &mut rng)?;
    let sdp_bound = bm_bound(graph, restarts.relaxation, &restarts.vectors)?;
    if let SdpBound::Certified(bound) = sdp_bound {
        (pipeline.on_progress)(Progress::Certified {
            bound,
            elapsed: pipeline.budget.elapsed(),
        });
    }
    let best = match pipeline.budget.deadline() {
        Some(deadline) => pipeline.refine_until(deadline, restarts.pool, restarts.best, &mut rng),
        None => restarts.best,
    };
    Ok(max_cut_result(graph, best, sdp_bound, restarts.num_trials))
}

/// An RNG seeded with `seed`, or from the OS without one.
fn seeded_rng(seed: Option<u64>) -> StdRng {
    seed.map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64)
}

/// The result of `num_trials` trials whose best cut is `best`.
fn max_cut_result(
    graph: &Graph,
    best: Cut,
    sdp_bound: SdpBound,
    num_trials: usize,
) -> MaxCutResult {
    MaxCutResult {
        approximation_ratio: sdp_bound.ratio(best.cut_value, graph.total_abs_weight()),
        partition: best.partition,
        cut_value: best.cut_value,
        sdp_bound,
        num_trials,
    }
}

/// Progress a GW run reports as it goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Progress {
    /// A BM solve's progress.
    Bm(BmProgress),
    /// BM restart `restart` of `restarts` (from 1) finished rounding.
    Restart {
        /// Which restart, from 1.
        restart: usize,
        /// Restarts asked for.
        restarts: usize,
        /// The relaxation value this restart reached.
        relaxation: f64,
        /// This restart's best cut.
        cut: f64,
        /// The best cut of all restarts so far.
        best: f64,
        /// Time since the run started.
        elapsed: Duration,
    },
    /// The time limit left no room for another restart after `completed`.
    TimeLimitStop {
        /// Restarts run.
        completed: usize,
        /// Restarts asked for.
        restarts: usize,
    },
    /// The BM bound was certified.
    Certified {
        /// The certified upper bound.
        bound: f64,
        /// Time since the run started.
        elapsed: Duration,
    },
    /// Refinement found a better cut, in pool slot `slot`.
    RefineImproved {
        /// The new best cut.
        best: f64,
        /// The slot that found it.
        slot: usize,
        /// Time since the run started.
        elapsed: Duration,
    },
    /// A periodic refinement status.
    RefineStatus {
        /// The best cut so far.
        best: f64,
        /// Iterated local search rounds run so far.
        rounds: usize,
        /// Time since the run started.
        elapsed: Duration,
    },
    /// Refinement ended at the time limit.
    RefineDone {
        /// Iterated local search rounds run.
        rounds: usize,
        /// The best cut.
        best: f64,
        /// Time since the run started.
        elapsed: Duration,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
