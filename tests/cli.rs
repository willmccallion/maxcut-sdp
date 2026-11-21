//! The command-line program's behaviour on its options and inputs.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Writes `contents` to a file named `name` and returns its path.
fn graph_file(name: &str, contents: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, contents).unwrap_or_else(|e| panic!("write {name}: {e}"));
    path
}

/// A cycle on `n` vertices in Biq Mac format, in a file of `test`'s own,
/// since the tests run in parallel.
fn cycle_file(n: usize, test: &str) -> PathBuf {
    let mut text = format!("{n} {n}\n");
    for i in 1..=n {
        writeln!(text, "{i} {} 1", i % n + 1).unwrap_or_else(|e| panic!("{e}"));
    }
    graph_file(&format!("{test}_cycle_{n}.txt"), &text)
}

fn run(path: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_maxcut-sdp"))
        .arg(path)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("run the CLI: {e}"))
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn the_default_solves_a_large_graph_with_bm() {
    let output = run(
        &cycle_file(101, "the_default_solves_a_large_graph_with_bm"),
        &["--seed", "1", "--trials", "1"],
    );

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(
        stdout(&output).contains("Solver:         bm\n"),
        "{}",
        stdout(&output)
    );
}

#[test]
fn the_exact_comparison_never_prints_nan() {
    let path = graph_file("negative_triangle.txt", "3 3\n1 2 -1\n2 3 -1\n1 3 -1\n");

    let output = run(&path, &["--exact", "--seed", "1"]);

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(!stdout(&output).contains("NaN"), "{}", stdout(&output));
}

#[test]
fn zero_trials_are_refused_without_crashing() {
    let output = run(
        &cycle_file(5, "zero_trials_are_refused_without_crashing"),
        &["--trials", "0", "--seed", "1"],
    );

    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(stderr(&output).contains("--trials") && !stderr(&output).contains("panicked"));
}

#[test]
fn zero_restarts_are_refused() {
    let output = run(
        &cycle_file(5, "zero_restarts_are_refused"),
        &["--solver", "bm", "--restarts", "0"],
    );

    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--restarts"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_rank_past_the_vertex_count_is_refused() {
    let output = run(
        &cycle_file(5, "a_rank_past_the_vertex_count_is_refused"),
        &["--solver", "bm", "--rank", "6"],
    );

    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    assert!(stderr(&output).contains("rank 6"), "{}", stderr(&output));
}

#[test]
fn a_bm_flag_is_refused_when_the_ipm_runs() {
    let output = run(
        &cycle_file(5, "a_bm_flag_is_refused_when_the_ipm_runs"),
        &["--time", "5"],
    );

    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--time applies only"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_rotation_past_a_right_angle_is_refused() {
    let output = run(
        &cycle_file(5, "a_rotation_past_a_right_angle_is_refused"),
        &["--rotation", "2.0"],
    );

    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("--rotation"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn format_overrides_the_detected_one() {
    let path = cycle_file(5, "format_overrides_the_detected_one");

    let as_biqmac = run(&path, &["--format", "biqmac", "--seed", "1"]);
    let as_dimacs = run(&path, &["--format", "dimacs", "--seed", "1"]);

    assert!(as_biqmac.status.success(), "{}", stderr(&as_biqmac));
    assert_eq!(as_dimacs.status.code(), Some(1));
    assert!(
        stderr(&as_dimacs).contains("no problem line"),
        "{}",
        stderr(&as_dimacs)
    );
}

#[test]
fn an_unreadable_input_fails_naming_the_file() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no_such_graph.txt");

    let output = run(&path, &[]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("no_such_graph.txt"),
        "{}",
        stderr(&output)
    );
}
