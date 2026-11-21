//! Peak memory of the dense interior point method as `n` grows.
//!
//! Each size is measured in a fresh child process (this test binary re-run
//! with a probe filter), since `VmHWM` is a per-process high-water mark.

mod common;

use std::process::Command;

use common::erdos_renyi;
use maxcut_sdp::sdp::SdpProblem;
use maxcut_sdp::sdp::ipm::{IpmConfig, IpmSolver};

const PROBE_N: &str = "MAXCUT_IPM_MEMORY_PROBE_N";

fn peak_rss_kib() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        panic!("cannot read /proc/self/status");
    };
    let Some(line) = status.lines().find(|l| l.starts_with("VmHWM:")) else {
        panic!("no VmHWM line");
    };
    let Some(Ok(kib)) = line.split_whitespace().nth(1).map(str::parse) else {
        panic!("malformed VmHWM line: {line}");
    };
    kib
}

/// IPM iterations each probe runs.
const PROBE_ITERATIONS: usize = 3;

/// Run only as a child of `ipm_peak_memory_grows_at_most_quadratically`.
#[test]
#[ignore = "memory probe; run by ipm_peak_memory_grows_at_most_quadratically"]
fn ipm_memory_probe() {
    let Ok(n) = std::env::var(PROBE_N).map(|s| s.parse::<usize>()) else {
        panic!("{PROBE_N} not set");
    };
    let Ok(n) = n else {
        panic!("{PROBE_N} is not a number");
    };
    if n > 0 {
        let graph = erdos_renyi(n, 0.1, 1..=1, 11);
        let Ok(problem) = SdpProblem::maxcut_relaxation(&graph) else {
            panic!("relaxation failed");
        };
        // Memory peaks within the first iterations, so a few suffice; the
        // unconverged result is not needed.
        let config = IpmConfig {
            max_iterations: PROBE_ITERATIONS,
            ..IpmConfig::default()
        };
        let _ = IpmSolver::new(config).solve(&problem);
    }
    println!("PEAK_RSS_KIB={}", peak_rss_kib());
}

fn measure_peak_rss_kib(n: usize) -> u64 {
    let Ok(exe) = std::env::current_exe() else {
        panic!("no current exe");
    };
    let Ok(output) = Command::new(exe)
        .args([
            "--exact",
            "ipm_memory_probe",
            "--include-ignored",
            "--nocapture",
        ])
        .env(PROBE_N, n.to_string())
        .output()
    else {
        panic!("cannot spawn probe");
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(Ok(kib)) = stdout
        .lines()
        .find_map(|l| l.strip_prefix("PEAK_RSS_KIB="))
        .map(str::parse)
    else {
        panic!("probe at n = {n} printed no peak: {stdout}");
    };
    kib
}

#[test]
fn ipm_peak_memory_grows_at_most_quadratically() {
    let baseline = measure_peak_rss_kib(0);
    let sizes = [200_usize, 300, 400];
    assert_eq!(sizes[2], 2 * sizes[0]);

    let peaks: Vec<u64> = sizes.iter().map(|&n| measure_peak_rss_kib(n)).collect();

    let growth: Vec<f64> = peaks
        .iter()
        .map(|&p| p.saturating_sub(baseline) as f64)
        .collect();
    let exponent = (growth[2] / growth[0]).log2();
    println!(
        "baseline {baseline} KiB; peaks {peaks:?} KiB at n = {sizes:?}; exponent {exponent:.2}"
    );
    assert!(
        exponent < 2.5,
        "peak RSS above baseline grows like n^{exponent:.2}: peaks {peaks:?} KiB at n = {sizes:?}, baseline {baseline} KiB"
    );
}
