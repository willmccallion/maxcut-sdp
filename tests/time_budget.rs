//! The `time_limit` budget of the Burer-Monteiro pipeline.

mod common;

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use common::sparse_random_graph;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::rounding::TrialPlan;
use maxcut_sdp::maxcut::{Backend, BmConfig, GwConfig, gw_maxcut};

fn elapsed_with_budget(graph: &Graph, num_restarts: usize, budget: Duration) -> Duration {
    let config = GwConfig {
        seed: Some(1),
        backend: Backend::Bm(BmConfig {
            num_restarts: NonZeroUsize::new(num_restarts).unwrap_or_else(|| panic!("no restarts")),
            time_limit: Some(budget),
            ..BmConfig::default()
        }),
        ..GwConfig::default()
    };

    let start = Instant::now();
    let result = gw_maxcut(graph, &config, |_| {});
    let elapsed = start.elapsed();

    if let Err(e) = result {
        panic!("solver failed: {e}");
    }
    elapsed
}

/// `--solver bm --time 1 --restarts 10` on a 4000-vertex graph. The margin is
/// deliberately generous (10 s over a 1 s budget) because the test machine may
/// be heavily loaded.
#[test]
fn time_limit_bounds_total_runtime() {
    let graph = sparse_random_graph(4000, 42);
    let budget = Duration::from_secs(1);
    let margin = Duration::from_secs(10);

    let elapsed = elapsed_with_budget(&graph, 10, budget);

    assert!(
        elapsed < budget + margin,
        "took {elapsed:.1?} with a {budget:?} budget"
    );
}

/// Many cheap restarts: each fits easily, so only a deadline check between
/// restarts can keep the total near the budget. Margin is generous (10 s).
#[test]
fn time_limit_is_checked_between_restarts() {
    let graph = sparse_random_graph(200, 42);
    let budget = Duration::from_secs(1);
    let margin = Duration::from_secs(10);

    let elapsed = elapsed_with_budget(&graph, 200, budget);

    assert!(
        elapsed < budget + margin,
        "took {elapsed:.1?} with a {budget:?} budget and 200 restarts"
    );
}

#[test]
fn a_time_limit_never_runs_more_restarts_than_asked() {
    let graph = sparse_random_graph(30, 7);
    let trials_per_restart = NonZeroUsize::new(10).unwrap_or_else(|| panic!("10 trials"));
    let config = GwConfig {
        plan: TrialPlan {
            num_trials: trials_per_restart,
            ils_rounds: 5,
            ..TrialPlan::default()
        },
        seed: Some(1),
        backend: Backend::Bm(BmConfig {
            num_restarts: NonZeroUsize::new(2).unwrap_or_else(|| panic!("2 restarts")),
            time_limit: Some(Duration::from_secs(2)),
            ..BmConfig::default()
        }),
    };

    let result =
        gw_maxcut(&graph, &config, |_| {}).unwrap_or_else(|e| panic!("solver failed: {e}"));

    assert_eq!(result.num_trials, 2 * trials_per_restart.get());
}

/// Refinement only replaces the best cut with a better one, so a timed run
/// keeps the untimed run's cut or beats it, and reports its partition's cut.
#[test]
fn refinement_never_loses_the_rounded_cut() {
    let graph = sparse_random_graph(300, 7);
    let config = |time_limit| GwConfig {
        seed: Some(3),
        backend: Backend::Bm(BmConfig {
            time_limit,
            ..BmConfig::default()
        }),
        ..GwConfig::default()
    };

    let untimed = gw_maxcut(&graph, &config(None), |_| {});
    let timed = gw_maxcut(&graph, &config(Some(Duration::from_secs(1))), |_| {});

    let untimed = untimed.unwrap_or_else(|e| panic!("untimed run failed: {e}"));
    let timed = timed.unwrap_or_else(|e| panic!("timed run failed: {e}"));
    assert!(
        timed.cut.cut_value >= untimed.cut.cut_value,
        "{} < {}",
        timed.cut.cut_value,
        untimed.cut.cut_value
    );
    assert_eq!(timed.cut.cut_value, graph.cut_value(&timed.cut.partition));
}
