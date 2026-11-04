//! Exact solver on graphs too large for a 64-bit mask.

use maxcut_sdp::graph::Graph;
use maxcut_sdp::maxcut::exact::exact_maxcut_bounded;

#[test]
fn exact_on_64_vertex_path_errors_or_returns_optimum() {
    let n = 64;
    let mut path = Graph::new(n);
    for v in 0..n - 1 {
        if let Err(e) = path.add_edge(v, v + 1, 1.0) {
            panic!("valid edge rejected: {e}");
        }
    }

    let result = exact_maxcut_bounded(&path, 64);

    if let Ok(r) = result {
        assert!(
            (r.cut_value - 63.0).abs() < 1e-9,
            "returned cut {} for a 64-vertex path whose max cut is 63",
            r.cut_value
        );
    }
}
