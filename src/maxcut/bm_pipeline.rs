//! The Burer-Monteiro pipeline: restarts that each solve and round, then,
//! under a time limit, refinement of the best cuts until it runs out.

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::rounding::{self, PERTURBATION_CYCLE_ROUNDS, TrialPlan};
use super::{Cut, Progress};
use crate::GwError;
use crate::graph::Graph;
use crate::sdp::bm::{BmSolver, BmSolverConfig};
use crate::sdp::linalg::DenseMatrix;

/// Cuts each restart keeps as starting points for refinement, and the
/// number of refinement slots.
const POOL_SIZE: usize = 8;

/// Time between refinement status reports.
const REFINE_REPORT_INTERVAL: Duration = Duration::from_secs(10);

/// The wall clock a run measures progress and its time limit against.
pub struct Budget {
    start: Instant,
    deadline: Option<Instant>,
}

impl Budget {
    pub fn start(time_limit: Option<Duration>) -> Self {
        let start = Instant::now();
        Self {
            start,
            deadline: time_limit.map(|limit| start + limit),
        }
    }

    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// What the restarts found.
pub struct Restarts {
    /// The best cut of any restart, the earliest among equals.
    pub best: Cut,
    /// The highest relaxation value any restart reached.
    pub relaxation: f64,
    /// The vectors that reached it.
    pub vectors: DenseMatrix,
    /// Each restart's best cuts, as starting points for refinement.
    pub pool: Vec<Cut>,
    /// Rounding trials run across the restarts.
    pub num_trials: usize,
}

/// One restart's solve and rounding.
struct Restart {
    /// Its best cuts, best first.
    pool: Vec<Cut>,
    relaxation: f64,
    vectors: DenseMatrix,
}

/// A BM run's fixed inputs and where it reports progress.
pub struct BmPipeline<'a, P> {
    pub graph: &'a Graph,
    pub solver: BmSolver,
    pub plan: TrialPlan,
    pub budget: Budget,
    pub on_progress: P,
}

impl<'a, P: FnMut(Progress)> BmPipeline<'a, P> {
    pub const fn new(
        graph: &'a Graph,
        solver_config: BmSolverConfig,
        plan: TrialPlan,
        budget: Budget,
        on_progress: P,
    ) -> Self {
        Self {
            graph,
            solver: BmSolver::new(solver_config),
            plan,
            budget,
            on_progress,
        }
    }

    /// Runs up to `restarts` restarts: the first always, each later one
    /// only if, taking as long as the one before, it ends by the deadline.
    pub fn run_restarts(&mut self, restarts: usize, rng: &mut StdRng) -> Result<Restarts, GwError> {
        let first_start = Instant::now();
        let first = self.run_restart(rng)?;
        let mut outcome = Restarts {
            best: first.pool[0].clone(),
            relaxation: first.relaxation,
            vectors: first.vectors,
            pool: first.pool,
            num_trials: self.plan.trials.get(),
        };
        self.report_restart(
            1,
            restarts,
            outcome.relaxation,
            outcome.best.cut_value,
            &outcome,
        );
        let mut last_restart = first_start.elapsed();

        for restart in 1..restarts {
            let restart_start = Instant::now();
            if !fits_before(self.budget.deadline, restart_start, last_restart) {
                (self.on_progress)(Progress::TimeLimitStop {
                    completed: restart,
                    restarts,
                });
                break;
            }
            let next = self.run_restart(rng)?;
            let (relaxation, cut) = (next.relaxation, next.pool[0].cut_value);
            outcome.absorb(next, self.plan.trials.get());
            self.report_restart(restart + 1, restarts, relaxation, cut, &outcome);
            last_restart = restart_start.elapsed();
        }
        Ok(outcome)
    }

    fn run_restart(&mut self, rng: &mut StdRng) -> Result<Restart, GwError> {
        let on_progress = &mut self.on_progress;
        let solution = self.solver.solve_maxcut(self.graph, rng, |progress| {
            on_progress(Progress::Bm(progress));
        })?;
        let cuts = rounding::round_trials(&solution.vectors, self.graph, self.plan, rng);
        Ok(Restart {
            pool: rounding::best_cuts(cuts, POOL_SIZE),
            relaxation: -solution.objective,
            vectors: solution.vectors,
        })
    }

    fn report_restart(
        &mut self,
        restart: usize,
        restarts: usize,
        relaxation: f64,
        cut: f64,
        outcome: &Restarts,
    ) {
        (self.on_progress)(Progress::Restart {
            restart,
            restarts,
            relaxation,
            cut,
            best: outcome.best.cut_value,
            elapsed: self.budget.elapsed(),
        });
    }

    /// Refines the best cuts of `pool` with iterated local search in
    /// parallel until `deadline`, and returns `best` or a better cut found.
    pub fn refine_until(
        &mut self,
        deadline: Instant,
        pool: Vec<Cut>,
        mut best: Cut,
        rng: &mut StdRng,
    ) -> Cut {
        let mut slots: Vec<RefinementSlot> = rounding::best_cuts(pool, POOL_SIZE)
            .into_iter()
            .map(|cut| RefinementSlot {
                cut,
                rng: StdRng::seed_from_u64(rng.random()),
            })
            .collect();
        let mut rounds = 0_usize;
        let mut last_report = self.budget.elapsed();
        while Instant::now() < deadline {
            rounds += slots
                .par_iter_mut()
                .map(|slot| slot.refine_until(self.graph, deadline))
                .sum::<usize>();
            for (slot, refined) in slots.iter().enumerate() {
                if refined.cut.cut_value > best.cut_value {
                    best = refined.cut.clone();
                    (self.on_progress)(Progress::RefineImproved {
                        best: best.cut_value,
                        slot,
                        elapsed: self.budget.elapsed(),
                    });
                }
            }
            let now = self.budget.elapsed();
            if now.saturating_sub(last_report) >= REFINE_REPORT_INTERVAL {
                (self.on_progress)(Progress::RefineStatus {
                    best: best.cut_value,
                    rounds,
                    elapsed: now,
                });
                last_report = now;
            }
        }
        if rounds > 0 {
            (self.on_progress)(Progress::RefineDone {
                rounds,
                best: best.cut_value,
                elapsed: self.budget.elapsed(),
            });
        }
        best
    }
}

impl Restarts {
    /// Adds a later restart of `trials` rounding trials.
    fn absorb(&mut self, restart: Restart, trials: usize) {
        self.num_trials += trials;
        if restart.pool[0].cut_value > self.best.cut_value {
            self.best = restart.pool[0].clone();
        }
        if restart.relaxation > self.relaxation {
            self.relaxation = restart.relaxation;
            self.vectors = restart.vectors;
        }
        self.pool.extend(restart.pool);
    }
}

/// One cut being refined, with its own random stream.
struct RefinementSlot {
    cut: Cut,
    rng: StdRng,
}

impl RefinementSlot {
    /// Runs iterated local search on the cut until `deadline`, keeping the
    /// best cut, and returns the rounds run.
    fn refine_until(&mut self, graph: &Graph, deadline: Instant) -> usize {
        let (cut_value, rounds) = rounding::iterated_local_search_until(
            &mut self.cut.partition,
            graph,
            PERTURBATION_CYCLE_ROUNDS,
            Some(deadline),
            &mut self.rng,
        );
        self.cut.cut_value = cut_value;
        rounds
    }
}

/// Whether a step expected to take `expected`, started at `now`, finishes by
/// `deadline`; always when there is none.
fn fits_before(deadline: Option<Instant>, now: Instant, expected: Duration) -> bool {
    deadline.is_none_or(|deadline| now + expected <= deadline)
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

    #[test]
    fn every_step_fits_without_a_deadline() {
        assert!(fits_before(None, Instant::now(), Duration::MAX / 2));
    }
}
