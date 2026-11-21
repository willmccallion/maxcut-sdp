//! CLI entry point for maxcut-sdp.
//!
//! Parses a graph from an input file, runs the Goemans-Williamson algorithm,
//! and reports the cut value, SDP bound, and approximation ratio.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use maxcut_sdp::GwError;
use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::exact::exact_maxcut;
use maxcut_sdp::maxcut::rounding::{RotationAngle, Rounding, TrialPlan};
use maxcut_sdp::maxcut::{
    Backend, BackendKind, BmConfig, Cut, DEFAULT_ILS_ROUNDS, DEFAULT_RESTARTS, DEFAULT_TRIALS,
    GwConfig, MAX_CERTIFIED_VERTICES, MaxCutResult, Progress, SdpBound, cut_ratio, gw_maxcut,
};
use maxcut_sdp::parser::{parse_auto, parse_biqmac, parse_dimacs};
use maxcut_sdp::sdp::bm::{BmProgress, BmSolverConfig};
use maxcut_sdp::sdp::ipm::{DEFAULT_EPSILON, IpmConfig};

/// Goemans-Williamson Max-Cut approximation via semidefinite programming.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Path to the input graph file (Biq Mac or DIMACS format).
    input: PathBuf,

    /// Number of random hyperplane rounding trials.
    #[arg(long, default_value_t = DEFAULT_TRIALS)]
    trials: NonZeroUsize,

    /// IPM tolerance on the relative duality gap and infeasibilities (IPM only).
    #[arg(long, default_value_t = DEFAULT_EPSILON)]
    epsilon: f64,

    /// Also compute the exact max-cut (brute force, small graphs only).
    #[arg(long)]
    exact: bool,

    /// RNG seed for reproducible rounding.
    #[arg(long)]
    seed: Option<u64>,

    /// Force input format instead of auto-detecting.
    #[arg(long, value_enum)]
    format: Option<InputFormat>,

    /// SDP solver backend.
    #[arg(long, value_enum, default_value_t = SolverChoice::Auto)]
    solver: SolverChoice,

    /// BM factorization rank, within 2..=n (default: ⌈√(2n)⌉).
    #[arg(long)]
    rank: Option<usize>,

    /// Number of BM random restarts (each rounded independently).
    #[arg(long, default_value_t = DEFAULT_RESTARTS)]
    restarts: NonZeroUsize,

    /// Number of iterated local search rounds per rounding trial.
    #[arg(long, default_value_t = DEFAULT_ILS_ROUNDS)]
    ils_rounds: usize,

    /// Time budget in seconds (BM only). Restarts after the first start only
    /// if they fit, and the time left refines the best cuts; the first
    /// restart always completes.
    #[arg(long)]
    time: Option<u64>,

    /// Round with Zwick's outward rotation by this angle, in radians within
    /// [0, π/2], instead of plain hyperplane rounding.
    #[arg(long, value_parser = parse_rotation)]
    rotation: Option<RotationAngle>,
}

fn parse_rotation(arg: &str) -> Result<RotationAngle, String> {
    let radians: f64 = arg.parse().map_err(|e| format!("{e}"))?;
    RotationAngle::new(radians).ok_or_else(|| format!("{radians} is not within [0, π/2]"))
}

/// Supported input file formats.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum InputFormat {
    /// Biq Mac Library format.
    Biqmac,
    /// DIMACS graph format.
    Dimacs,
}

/// SDP solver backend, as requested.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SolverChoice {
    /// The IPM up to 100 vertices, BM past them.
    Auto,
    /// Primal-dual interior point method (dense, exact).
    Ipm,
    /// Burer-Monteiro low-rank factorization (sparse, scalable).
    Bm,
}

impl SolverChoice {
    /// The solver this choice runs on a graph of `vertices` vertices.
    const fn kind(self, vertices: usize) -> BackendKind {
        match self {
            Self::Auto => BackendKind::auto_for(vertices),
            Self::Ipm => BackendKind::Ipm,
            Self::Bm => BackendKind::Bm,
        }
    }
}

/// The name `--solver` takes for a backend.
const fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Ipm => "ipm",
        BackendKind::Bm => "bm",
    }
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// The most a reported cut may differ from the cut its partition gives.
const VERIFICATION_TOLERANCE: f64 = 1e-6;

/// Why a run failed.
#[derive(Debug)]
enum CliError {
    /// The input file could not be read.
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// Parsing or solving failed.
    Solver(GwError),
}

impl From<GwError> for CliError {
    fn from(error: GwError) -> Self {
        Self::Solver(error)
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::Solver(error) => write!(f, "{error}"),
        }
    }
}

fn run(cli: &Cli) -> Result<(), CliError> {
    let graph = read_graph(cli)?;
    println!(
        "Graph: {} vertices, {} edges, total weight {:.3}",
        graph.num_vertices(),
        graph.num_edges(),
        graph.total_weight()
    );
    let kind = cli.solver.kind(graph.num_vertices());
    println!("Solver:         {}", backend_name(kind));

    let start = Instant::now();
    let max_cut = solve(cli, kind, &graph)?;
    let elapsed = start.elapsed();
    verify_cut(&graph, &max_cut.cut)?;
    print_result(&graph, &max_cut, elapsed);
    if cli.exact {
        print_exact_comparison(&graph, &max_cut.cut);
    }
    Ok(())
}

fn read_graph(cli: &Cli) -> Result<Graph, CliError> {
    let input = std::fs::read_to_string(&cli.input).map_err(|source| CliError::Read {
        path: cli.input.clone(),
        source,
    })?;
    let graph = match cli.format {
        Some(InputFormat::Biqmac) => parse_biqmac(&input),
        Some(InputFormat::Dimacs) => parse_dimacs(&input),
        None => parse_auto(&input),
    };
    Ok(graph?)
}

/// Runs GW on `graph` with the `kind` of solver and the options `cli` gives.
fn solve(cli: &Cli, kind: BackendKind, graph: &Graph) -> Result<MaxCutResult, GwError> {
    let config = GwConfig {
        plan: TrialPlan {
            num_trials: cli.trials,
            ils_rounds: cli.ils_rounds,
            rounding: cli
                .rotation
                .map_or(Rounding::Hyperplane, Rounding::OutwardRotation),
        },
        seed: cli.seed,
        backend: backend(cli, kind),
    };
    let mut best_cut = None;
    gw_maxcut(graph, &config, |progress| {
        report_progress(progress, &mut best_cut);
    })
}

fn backend(cli: &Cli, kind: BackendKind) -> Backend {
    match kind {
        BackendKind::Ipm => Backend::Ipm(IpmConfig {
            epsilon: cli.epsilon,
            ..IpmConfig::default()
        }),
        BackendKind::Bm => Backend::Bm(BmConfig {
            solver: BmSolverConfig {
                rank: cli.rank,
                ..BmSolverConfig::default()
            },
            num_restarts: cli.restarts,
            time_limit: cli.time.map(Duration::from_secs),
        }),
    }
}

/// Checks the reported cut against the cut its partition gives.
fn verify_cut(graph: &Graph, cut: &Cut) -> Result<(), GwError> {
    let verified_cut = graph.cut_value(&cut.partition);
    if (verified_cut - cut.cut_value).abs() > VERIFICATION_TOLERANCE {
        return Err(GwError::SolverFailed(format!(
            "VERIFICATION FAILED: reported cut {:.6} but recomputed cut {verified_cut:.6}",
            cut.cut_value
        )));
    }
    Ok(())
}

fn print_result(graph: &Graph, max_cut: &MaxCutResult, elapsed: Duration) {
    println!("\n--- Goemans-Williamson Result ---");
    println!("{}", describe_bound(max_cut.sdp_bound));
    println!("GW cut value:   {:.6} (verified)", max_cut.cut.cut_value);
    println!("Approx ratio:   {}", describe_ratio(max_cut));
    println!("Rounding trials: {}", max_cut.num_trials);
    println!("Total time:     {elapsed:.3?}");

    let s_count = max_cut.cut.partition.iter().filter(|&&b| b).count();
    println!(
        "Partition:      |S| = {}, |S̄| = {}",
        s_count,
        graph.num_vertices() - s_count
    );
}

/// Prints the exact maximum cut beside the GW `cut`, or why it was skipped.
fn print_exact_comparison(graph: &Graph, cut: &Cut) {
    let start = Instant::now();
    match exact_maxcut(graph) {
        Ok(optimum) => {
            let elapsed = start.elapsed();
            println!("\n--- Exact Max-Cut ---");
            println!("Optimal cut:    {:.6}", optimum.cut_value);
            let gw_over_opt = cut_ratio(cut.cut_value, optimum.cut_value, graph.total_abs_weight());
            println!("GW / OPT:       {}", format_ratio(gw_over_opt));
            println!("Time:           {elapsed:.3?}");
        }
        Err(e) => eprintln!("Exact solver skipped: {e}"),
    }
}

/// Prints a run's progress to stderr. `best_cut` is the best cut of the
/// restarts so far, which the BM iteration lines show.
fn report_progress(progress: Progress, best_cut: &mut Option<f64>) {
    match progress {
        Progress::Bm(BmProgress::Iteration {
            iteration,
            relaxation,
            gradient_norm,
        }) => match best_cut {
            Some(best) => eprintln!(
                "  iter {iteration}: obj={relaxation:.4}, grad={gradient_norm:.2e}, best={best:.0}"
            ),
            None => eprintln!("  iter {iteration}: obj={relaxation:.4}, grad={gradient_norm:.2e}"),
        },
        Progress::Bm(BmProgress::Finished {
            iterations,
            relaxation,
            gradient_norm,
        }) => eprintln!("BM: iters={iterations}, obj={relaxation:.4}, grad={gradient_norm:.2e}"),
        Progress::Restart {
            restart,
            restarts,
            relaxation,
            cut,
            best,
            elapsed,
        } => {
            *best_cut = Some(best);
            eprintln!(
                "Restart {restart}/{restarts}: relaxation={relaxation:.4}, cut={cut:.0}, best={best:.0} ({elapsed:.1?})"
            );
        }
        Progress::TimeLimitStop {
            completed,
            restarts,
        } => eprintln!("Time limit: stopping after {completed}/{restarts} restarts"),
        Progress::Certified { bound, elapsed } => {
            eprintln!("Certified bound: {bound:.4} ({elapsed:.1?})");
        }
        Progress::RefineImproved {
            best,
            slot,
            elapsed,
        } => eprintln!("  Refine: new best={best:.0} (slot {slot}, {elapsed:.1?})"),
        Progress::RefineStatus {
            best,
            rounds,
            elapsed,
        } => eprintln!("  Refine: best={best:.0}, {rounds} rounds ({elapsed:.1?})"),
        Progress::RefineDone {
            rounds,
            best,
            elapsed,
        } => eprintln!("Refinement done: {rounds} ILS rounds, best={best:.0} ({elapsed:.1?})"),
    }
}

/// The SDP bound's report line, saying whether it is certified.
fn describe_bound(bound: SdpBound) -> String {
    match bound {
        SdpBound::Certified(bound) => format!("SDP bound:      {bound:.6} (certified)"),
        SdpBound::Relaxation(value) => format!(
            "SDP relaxation: {value:.6} (not a bound; BM certifies up to {MAX_CERTIFIED_VERTICES} vertices)"
        ),
    }
}

/// The approximation ratio, or why there is none.
fn describe_ratio(max_cut: &MaxCutResult) -> String {
    match max_cut.sdp_bound {
        SdpBound::Certified(_) => format_ratio(max_cut.approximation_ratio),
        SdpBound::Relaxation(_) => "none without a certified bound".to_string(),
    }
}

/// A ratio to six places, or why there is none.
fn format_ratio(ratio: Option<f64>) -> String {
    ratio.map_or_else(
        || "undefined (the bound is 0)".to_string(),
        |r| format!("{r:.6}"),
    )
}
