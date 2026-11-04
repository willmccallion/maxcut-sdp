//! A request for zero rounding trials, which cannot produce a cut.

use std::process::Command;

#[test]
fn the_cli_refuses_zero_trials_without_crashing() {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("zero_trials_c5.txt");
    std::fs::write(&path, "5 5\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 1 1\n")
        .unwrap_or_else(|e| panic!("write graph: {e}"));

    let output = Command::new(env!("CARGO_BIN_EXE_maxcut-sdp"))
        .arg(&path)
        .args(["--trials", "0", "--seed", "1"])
        .output()
        .unwrap_or_else(|e| panic!("run the CLI: {e}"));

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("--trials") && !stderr.contains("panicked"),
        "stderr: {stderr}"
    );
}
