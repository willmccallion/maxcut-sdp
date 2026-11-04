//! CLI entry point for maxcut-sdp.
//!
//! Parses a graph from an input file, runs the Goemans-Williamson algorithm,
//! and reports the cut value, SDP bound, and approximation ratio.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use maxcut_sdp::GwError;
use maxcut_sdp::maxcut::exact::exact_maxcut;
use maxcut_sdp::maxcut::rounding::{RotationAngle, Rounding};
use maxcut_sdp::maxcut::{GwBmConfig, GwConfig, cut_ratio, gw_maxcut, gw_maxcut_bm};
use maxcut_sdp::parser::{parse_auto, parse_biqmac, parse_dimacs};
use maxcut_sdp::sdp::solver::SolverConfig;
use maxcut_sdp::sdp::solver_bm::BmSolverConfig;

/// Goemans-Williamson Max-Cut approximation via semidefinite programming.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Path to the input graph file (Biq Mac or DIMACS format).
    input: PathBuf,

    /// Number of random hyperplane rounding trials.
    #[arg(long, default_value_t = maxcut_sdp::maxcut::DEFAULT_TRIALS)]
    trials: NonZeroUsize,

    /// SDP solver convergence tolerance.
    #[arg(long, default_value_t = 1e-8)]
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
    #[arg(long, value_enum, default_value_t = SolverChoice::Ipm)]
    solver: SolverChoice,

    /// BM factorization rank override (default: ⌈√(2n)⌉).
    #[arg(long)]
    rank: Option<usize>,

    /// Number of BM random restarts (each rounded independently).
    #[arg(long, default_value_t = 3)]
    restarts: usize,

    /// Number of iterated local search rounds per rounding trial.
    #[arg(long, default_value_t = 50)]
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

/// SDP solver backend.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum SolverChoice {
    /// Primal-dual interior point method (dense, exact).
    Ipm,
    /// Burer-Monteiro low-rank factorization (sparse, scalable).
    Bm,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), GwError> {
    let input = std::fs::read_to_string(&cli.input)
        .map_err(|e| GwError::ParseError(format!("failed to read {}: {e}", cli.input.display())))?;

    let graph = match cli.format {
        Some(InputFormat::Biqmac) => parse_biqmac(&input)?,
        Some(InputFormat::Dimacs) => parse_dimacs(&input)?,
        None => parse_auto(&input)?,
    };

    println!(
        "Graph: {} vertices, {} edges, total weight {:.3}",
        graph.num_vertices(),
        graph.num_edges(),
        graph.total_weight()
    );

    // Run GW algorithm with selected solver.
    let rounding = cli
        .rotation
        .map_or(Rounding::Hyperplane, Rounding::OutwardRotation);
    let start = Instant::now();
    let result = match cli.solver {
        SolverChoice::Ipm => {
            let config = GwConfig {
                num_trials: cli.trials,
                ils_rounds: cli.ils_rounds,
                rounding,
                solver_config: SolverConfig {
                    epsilon: cli.epsilon,
                    ..SolverConfig::default()
                },
                seed: cli.seed,
            };
            gw_maxcut(&graph, &config)?
        }
        SolverChoice::Bm => {
            let config = GwBmConfig {
                num_trials: cli.trials,
                ils_rounds: cli.ils_rounds,
                rounding,
                solver_config: BmSolverConfig {
                    rank: cli.rank,
                    num_restarts: cli.restarts,
                    ..BmSolverConfig::default()
                },
                seed: cli.seed,
                time_limit: cli.time.map(Duration::from_secs),
            };
            gw_maxcut_bm(&graph, &config)?
        }
    };
    let elapsed = start.elapsed();

    // Verify: independently recompute the cut value from the partition.
    let verified_cut = graph.cut_value(&result.partition);
    if (verified_cut - result.cut_value).abs() > 1e-6 {
        return Err(GwError::SolverFailed(format!(
            "VERIFICATION FAILED: reported cut {:.6} but recomputed cut {verified_cut:.6}",
            result.cut_value
        )));
    }

    println!("\n--- Goemans-Williamson Result ---");
    println!("SDP bound:      {:.6}", result.sdp_bound);
    println!("GW cut value:   {:.6} (verified)", result.cut_value);
    println!(
        "Approx ratio:   {}",
        format_ratio(result.approximation_ratio)
    );
    println!("Rounding trials: {}", result.num_trials);
    println!("Total time:     {elapsed:.3?}");

    let s_count = result.partition.iter().filter(|&&b| b).count();
    println!(
        "Partition:      |S| = {}, |S̄| = {}",
        s_count,
        graph.num_vertices() - s_count
    );

    // Optionally run exact solver.
    if cli.exact {
        let exact_start = Instant::now();
        match exact_maxcut(&graph) {
            Ok(exact_result) => {
                let exact_elapsed = exact_start.elapsed();
                println!("\n--- Exact Max-Cut ---");
                println!("Optimal cut:    {:.6}", exact_result.cut_value);
                let gw_over_opt = cut_ratio(
                    result.cut_value,
                    exact_result.cut_value,
                    graph.total_abs_weight(),
                );
                println!("GW / OPT:       {}", format_ratio(gw_over_opt));
                println!("Time:           {exact_elapsed:.3?}");
            }
            Err(e) => {
                eprintln!("Exact solver skipped: {e}");
            }
        }
    }

    Ok(())
}

/// A ratio to six places, or why there is none.
fn format_ratio(ratio: Option<f64>) -> String {
    ratio.map_or_else(
        || "undefined (the bound is 0)".to_string(),
        |r| format!("{r:.6}"),
    )
}
