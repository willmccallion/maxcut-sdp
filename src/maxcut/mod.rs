//! Goemans-Williamson Max-Cut approximation algorithm.
//!
//! Orchestrates the full GW pipeline:
//! 1. Formulate the Max-Cut SDP relaxation from the graph Laplacian
//! 2. Solve the SDP via interior point method
//! 3. Extract vectors from the SDP solution (Cholesky factorization)
//! 4. Run multiple random hyperplane roundings
//! 5. Return the best partition found
//!
//! The algorithm achieves an approximation ratio of `α_GW ≈ 0.878`, which is
//! optimal under the Unique Games Conjecture.
//!
//! See [`MATH.md`](./MATH.md) for the full mathematical derivation.

pub mod exact;
pub mod rounding;

use std::time::{Duration, Instant};

use crate::GwError;
use crate::graph::Graph;
use crate::sdp::SdpProblem;
use crate::sdp::solver::{Solver, SolverConfig};
use crate::sdp::solver_bm::{BmSolver, BmSolverConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// The result of running the Goemans-Williamson algorithm.
#[derive(Debug, Clone)]
pub struct MaxCutResult {
    /// The vertex partition: `partition[i] = true` means vertex `i` is in set `S`.
    pub partition: Vec<bool>,
    /// The cut value (total weight of edges crossing the partition).
    pub cut_value: f64,
    /// The SDP relaxation upper bound on the optimal cut.
    pub sdp_bound: f64,
    /// The approximation ratio: `cut_value / sdp_bound`.
    pub approximation_ratio: f64,
    /// Number of rounding trials performed.
    pub num_trials: usize,
}

/// Configuration for the GW algorithm.
#[derive(Debug, Clone)]
pub struct GwConfig {
    /// Number of independent rounding trials.
    pub num_trials: usize,
    /// Number of iterated local search rounds per rounding trial.
    pub ils_rounds: usize,
    /// Configuration for the underlying SDP solver.
    pub solver_config: SolverConfig,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GwConfig {
    fn default() -> Self {
        Self {
            num_trials: 100,
            ils_rounds: 50,
            solver_config: SolverConfig::default(),
            seed: None,
        }
    }
}

/// Configuration for the GW algorithm using the Burer-Monteiro solver.
#[derive(Debug, Clone)]
pub struct GwBmConfig {
    /// Number of independent rounding trials per BM restart.
    pub num_trials: usize,
    /// Number of iterated local search rounds per rounding trial.
    pub ils_rounds: usize,
    /// Configuration for the BM solver.
    pub solver_config: BmSolverConfig,
    /// Optional RNG seed for reproducibility (used for both solver and rounding).
    pub seed: Option<u64>,
    /// Optional time budget. When set, the solver runs as many restart+round
    /// cycles as fit within this duration, ignoring `num_restarts`.
    pub time_limit: Option<Duration>,
}

impl Default for GwBmConfig {
    fn default() -> Self {
        Self {
            num_trials: 100,
            ils_rounds: 50,
            solver_config: BmSolverConfig::default(),
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
/// See [`MATH.md`](./MATH.md) §9 for the implementation checklist.
///
/// # Errors
///
/// Returns an error if the SDP solver fails or the graph is empty.
pub fn gw_maxcut(graph: &Graph, config: &GwConfig) -> Result<MaxCutResult, GwError> {
    // 1. Formulate SDP.
    let problem = SdpProblem::maxcut_relaxation(graph)?;

    // 2. Solve SDP.
    let solver = Solver::new(config.solver_config.clone());
    let solution = solver.solve(&problem)?;

    // SDP bound: the Max-Cut objective is max (1/4) tr(LX) = -tr(CX).
    let sdp_bound = -solution.primal_obj;

    // 3. Round with random hyperplanes.
    let mut rng = config
        .seed
        .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);

    rounding::multi_round(
        &solution.x,
        graph,
        config.num_trials,
        config.ils_rounds,
        sdp_bound,
        &mut rng,
    )
}

/// Runs the GW Max-Cut algorithm using the Burer-Monteiro low-rank solver.
///
/// This is the recommended path for large or sparse graphs (n > 100).
/// Per-iteration cost is O(|E|·r) instead of O(n⁴) for the dense IPM.
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
    let single_restart_config = BmSolverConfig {
        num_restarts: 1,
        ..config.solver_config.clone()
    };
    let solver = BmSolver::new(single_restart_config);

    let pool_size = 8;
    let start = Instant::now();
    let mut best_cut = f64::NEG_INFINITY;
    let mut best_partition: Vec<bool> = Vec::new();
    let mut best_sdp_bound = f64::NEG_INFINITY;
    let mut total_trials = 0_usize;

    // Collect top partitions from all restarts for the refinement pool.
    let mut candidate_pool: Vec<(Vec<bool>, f64)> = Vec::new();

    let num_restarts = config.solver_config.num_restarts.max(1);
    for restart in 0..num_restarts {
        let ctx = if best_cut > f64::NEG_INFINITY {
            Some(best_cut)
        } else {
            None
        };
        let solution = solver.solve_maxcut_with_context(graph, &mut rng, ctx)?;
        let sdp_bound = -solution.objective;
        if sdp_bound > best_sdp_bound {
            best_sdp_bound = sdp_bound;
        }

        let (result, top_partitions) = rounding::multi_round_with_pool(
            &solution.vectors,
            graph,
            config.num_trials,
            config.ils_rounds,
            sdp_bound,
            pool_size,
            &mut rng,
        );
        total_trials += config.num_trials;
        candidate_pool.extend(top_partitions);

        if result.cut_value > best_cut {
            best_cut = result.cut_value;
            best_partition.clone_from(&result.partition);
        }

        eprintln!(
            "Restart {}/{num_restarts}: SDP={sdp_bound:.4}, cut={:.0}, best={best_cut:.0} ({:.1?})",
            restart + 1,
            result.cut_value,
            start.elapsed()
        );
    }

    // Refinement phase: if a time budget is set and time remains, maintain a
    // population of diverse partitions and refine them in parallel.
    if let Some(limit) = config.time_limit {
        // Build pool from the best partitions across all restarts.
        candidate_pool
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidate_pool.truncate(pool_size);

        // Give each pool member its own RNG for parallel execution.
        let mut pool: Vec<(Vec<bool>, f64, StdRng)> = candidate_pool
            .into_iter()
            .map(|(p, c)| {
                let seed: u64 = rng.random();
                (p, c, StdRng::seed_from_u64(seed))
            })
            .collect();

        let mut rounds_done = 0_usize;
        let mut last_report = start.elapsed();
        let report_interval = Duration::from_secs(10);
        let batch = 50;
        while start.elapsed() < limit {
            // Run ILS on all pool members in parallel.
            pool.par_iter_mut().for_each(|(partition, cut, slot_rng)| {
                let new_cut = rounding::iterated_local_search(partition, graph, batch, slot_rng);
                *cut = new_cut;
            });
            rounds_done += batch * pool.len();

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

    if best_partition.is_empty() {
        return Err(GwError::SolverFailed(
            "BM solver produced no solution".into(),
        ));
    }

    let ratio = if best_sdp_bound > 1e-15 {
        best_cut / best_sdp_bound
    } else {
        1.0
    };

    Ok(MaxCutResult {
        partition: best_partition,
        cut_value: best_cut,
        sdp_bound: best_sdp_bound,
        approximation_ratio: ratio,
        num_trials: total_trials,
    })
}
