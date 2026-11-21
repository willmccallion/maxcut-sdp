//! Reproducibility of seeded Burer-Monteiro runs.

mod common;

use common::sparse_random_graph;
use maxcut_sdp::maxcut::{GwBmConfig, SdpBound, gw_maxcut_bm};
use maxcut_sdp::sdp::bm::{BmSolver, BmSolverConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::num::NonZeroUsize;
use std::sync::OnceLock;

const RUNS: usize = 6;

#[test]
fn same_seed_gives_bitwise_identical_bm_relaxation() {
    let graph = sparse_random_graph(3000, 42);
    let solver = BmSolver::new(BmSolverConfig::default());

    let runs: Vec<(usize, u64, Vec<u64>)> = (0..RUNS)
        .map(|_| {
            let mut rng = StdRng::seed_from_u64(5);
            let Ok(solution) = solver.solve_maxcut(&graph, &mut rng) else {
                panic!("BM solve failed");
            };
            let vector_bits = solution
                .vectors
                .data()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            (
                solution.iterations,
                solution.objective.to_bits(),
                vector_bits,
            )
        })
        .collect();

    let summary: Vec<(usize, f64, f64)> = runs
        .iter()
        .map(|(iters, obj, bits)| {
            let max_vector_diff = bits
                .iter()
                .zip(&runs[0].2)
                .map(|(a, b)| (f64::from_bits(*a) - f64::from_bits(*b)).abs())
                .fold(0.0, f64::max);
            (*iters, f64::from_bits(*obj), max_vector_diff)
        })
        .collect();
    assert!(
        runs.iter().all(|run| run == &runs[0]),
        "(iterations, objective, max |V - V_run0|) per run: {summary:?}"
    );
}

/// Seeded pipeline runs on one graph, with one rounding trial and no ILS so
/// the cut follows the vectors closely; computed once for every test here.
fn seeded_bm_runs() -> &'static [(Vec<bool>, f64, SdpBound)] {
    static RUNS_ONCE: OnceLock<Vec<(Vec<bool>, f64, SdpBound)>> = OnceLock::new();
    RUNS_ONCE.get_or_init(|| {
        let graph = sparse_random_graph(3000, 42);
        let config = GwBmConfig {
            num_trials: NonZeroUsize::MIN,
            ils_rounds: 0,
            seed: Some(9),
            ..GwBmConfig::default()
        };
        (0..RUNS)
            .map(|_| {
                let r = gw_maxcut_bm(&graph, &config)
                    .unwrap_or_else(|e| panic!("gw_maxcut_bm failed: {e}"));
                (r.partition, r.cut_value, r.sdp_bound)
            })
            .collect()
    })
}

#[test]
fn same_seed_gives_identical_bm_cut_and_partition() {
    let runs = seeded_bm_runs();

    let summary: Vec<(f64, bool)> = runs
        .iter()
        .map(|(p, cut, _)| (*cut, p == &runs[0].0))
        .collect();
    assert!(
        runs.iter()
            .all(|run| run.0 == runs[0].0 && run.1.to_bits() == runs[0].1.to_bits()),
        "(cut, partition equals run 0) per run: {summary:?}"
    );
}

#[test]
fn same_seed_gives_bitwise_identical_bm_bound() {
    let runs = seeded_bm_runs();

    let bounds: Vec<f64> = runs.iter().map(|run| run.2.value()).collect();
    assert!(
        bounds.iter().all(|b| b.to_bits() == bounds[0].to_bits()),
        "sdp_bound per run: {bounds:?}"
    );
}
