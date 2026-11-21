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
use crate::maxcut::rounding::TrialPlan;
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
    /// The best cut found.
    pub cut: Cut,
    /// What the SDP relaxation established about the optimal cut.
    pub sdp_bound: SdpBound,
    /// The approximation ratio `cut.cut_value / sdp_bound`, or `None` when the
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

/// Iterated local search rounds per rounding trial by default.
pub const DEFAULT_ILS_ROUNDS: usize = 50;

/// Largest graph [`Backend::Auto`] solves with the IPM, whose `O(n³)`
/// iterations take seconds here and grow impractical soon after.
pub const AUTO_IPM_MAX_VERTICES: usize = 100;

/// Configuration for the GW algorithm.
#[derive(Debug, Clone, Default)]
pub struct GwConfig {
    /// The rounding trials, per BM restart with BM.
    pub plan: TrialPlan,
    /// RNG seed for reproducibility, used by BM's starts and the rounding.
    pub seed: Option<u64>,
    /// The SDP solver and its configuration.
    pub backend: Backend,
}

/// The SDP solver a run uses.
#[derive(Debug, Clone, Default)]
pub enum Backend {
    /// The IPM, with its default configuration, up to
    /// [`AUTO_IPM_MAX_VERTICES`], and BM, with its, past it.
    #[default]
    Auto,
    /// The primal-dual interior point method: dense, `O(n³)` per iteration.
    Ipm(IpmConfig),
    /// Burer-Monteiro: `O(|E|·r)` per iteration, for large sparse graphs.
    Bm(BmConfig),
}

/// The kind of SDP solver, whatever its configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// The interior point method.
    Ipm,
    /// Burer-Monteiro.
    Bm,
}

impl BackendKind {
    /// The solver [`Backend::Auto`] picks for a graph of `vertices` vertices.
    pub const fn auto_for(vertices: usize) -> Self {
        if vertices <= AUTO_IPM_MAX_VERTICES {
            Self::Ipm
        } else {
            Self::Bm
        }
    }
}

impl Backend {
    /// This backend's configuration for a graph of `vertices` vertices,
    /// choosing a solver if it is [`Backend::Auto`].
    fn resolve(&self, vertices: usize) -> Solver {
        match self {
            Self::Auto => match BackendKind::auto_for(vertices) {
                BackendKind::Ipm => Solver::Ipm(IpmConfig::default()),
                BackendKind::Bm => Solver::Bm(BmConfig::default()),
            },
            Self::Ipm(config) => Solver::Ipm(config.clone()),
            Self::Bm(config) => Solver::Bm(config.clone()),
        }
    }
}

/// A backend with its solver chosen.
enum Solver {
    Ipm(IpmConfig),
    Bm(BmConfig),
}

/// Configuration for the Burer-Monteiro backend.
#[derive(Debug, Clone)]
pub struct BmConfig {
    /// Configuration for each BM solve.
    pub solver: BmSolverConfig,
    /// BM solves from independent random starts, each rounded separately.
    pub num_restarts: NonZeroUsize,
    /// Optional time budget. When set, a restart after the first starts only
    /// if, taking as long as the one before, it would finish within the
    /// budget, and whatever time is left refines the best cuts with iterated
    /// local search. The first restart always completes, so a budget shorter
    /// than one solve is exceeded by it.
    pub time_limit: Option<Duration>,
}

impl Default for BmConfig {
    fn default() -> Self {
        Self {
            solver: BmSolverConfig::default(),
            num_restarts: DEFAULT_RESTARTS,
            time_limit: None,
        }
    }
}

/// Runs the Goemans-Williamson Max-Cut algorithm, reporting progress to
/// `on_progress`.
///
/// Solves the SDP relaxation with the configured backend, rounds its
/// vectors in `config.plan`'s trials, each improved by iterated local
/// search, and returns the best cut with the bound the SDP certifies.
/// BM rounds after each of its restarts and, with a time limit, refines the
/// best cuts until it runs out. See [`MATH.md`](./MATH.md) §10.
///
/// # Errors
///
/// Returns an error if the graph is empty or the solver fails.
pub fn gw_maxcut(
    graph: &Graph,
    config: &GwConfig,
    on_progress: impl FnMut(Progress),
) -> Result<MaxCutResult, GwError> {
    let mut rng = seeded_rng(config.seed);
    match config.backend.resolve(graph.num_vertices()) {
        Solver::Ipm(ipm) => ipm_maxcut(graph, &ipm, config.plan, &mut rng),
        Solver::Bm(bm) => bm_maxcut(graph, &bm, config.plan, &mut rng, on_progress),
    }
}

/// GW with the IPM, whose converged dual is feasible only to within the
/// tolerance, so its objective is certified rather than taken as it stands.
fn ipm_maxcut(
    graph: &Graph,
    config: &IpmConfig,
    plan: TrialPlan,
    rng: &mut StdRng,
) -> Result<MaxCutResult, GwError> {
    let problem = SdpProblem::maxcut_relaxation(graph)?;
    let solution = IpmSolver::new(config.clone()).solve(&problem)?;
    let sdp_bound = SdpBound::Certified(certified_bound(problem.into_cost(), &solution.y)?);
    let best = rounding::multi_round(&solution.x, graph, plan, rng)?;
    Ok(max_cut_result(
        graph,
        best,
        sdp_bound,
        plan.num_trials.get(),
    ))
}

/// GW with BM: restarts, the bound of the best relaxation, then refinement
/// if there is a time limit.
fn bm_maxcut(
    graph: &Graph,
    config: &BmConfig,
    plan: TrialPlan,
    rng: &mut StdRng,
    on_progress: impl FnMut(Progress),
) -> Result<MaxCutResult, GwError> {
    let budget = Budget::start(config.time_limit);
    let mut pipeline = BmPipeline::new(graph, config.solver.clone(), plan, budget, on_progress);

    let restarts = pipeline.run_restarts(config.num_restarts.get(), rng)?;
    let sdp_bound = bm_bound(graph, restarts.relaxation, &restarts.vectors)?;
    if let SdpBound::Certified(bound) = sdp_bound {
        (pipeline.on_progress)(Progress::Certified {
            bound,
            elapsed: pipeline.budget.elapsed(),
        });
    }
    let best = match pipeline.budget.deadline() {
        Some(deadline) => pipeline.refine_until(deadline, restarts.pool, restarts.best, rng),
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
        cut: best,
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
    fn auto_runs_the_ipm_up_to_its_limit() {
        assert_eq!(
            BackendKind::auto_for(AUTO_IPM_MAX_VERTICES),
            BackendKind::Ipm
        );
    }

    #[test]
    fn auto_runs_bm_past_the_ipm_limit() {
        assert_eq!(
            BackendKind::auto_for(AUTO_IPM_MAX_VERTICES + 1),
            BackendKind::Bm
        );
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
