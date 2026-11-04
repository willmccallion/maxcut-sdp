//! The CLI's default solver follows the graph's size.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A cycle on `n` vertices in Biq Mac format, written to a file named for it.
fn cycle_file(n: usize) -> PathBuf {
    let mut text = format!("{n} {n}\n");
    for i in 1..=n {
        let next = i % n + 1;
        writeln!(text, "{i} {next} 1").unwrap_or_else(|e| panic!("{e}"));
    }
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cycle_{n}.txt"));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write graph: {e}"));
    path
}

/// The `Solver:` line the CLI prints for an `n`-vertex cycle with no `--solver`.
fn default_solver_line(n: usize) -> String {
    let path = cycle_file(n);
    let output = Command::new(env!("CARGO_BIN_EXE_maxcut-sdp"))
        .args(["--seed", "1", "--trials", "1"])
        .arg(&path)
        .output()
        .unwrap_or_else(|e| panic!("run the CLI: {e}"));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find(|line| line.starts_with("Solver:"))
        .unwrap_or_else(|| panic!("no Solver: line"))
        .to_string()
}

#[test]
fn the_default_solves_a_small_graph_with_the_ipm() {
    assert_eq!(default_solver_line(100), "Solver:         ipm");
}

#[test]
fn the_default_solves_a_large_graph_with_bm() {
    assert_eq!(default_solver_line(101), "Solver:         bm");
}
