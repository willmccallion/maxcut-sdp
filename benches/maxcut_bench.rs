//! Criterion benchmarks for the maxcut-sdp pipeline.
#![allow(
    missing_docs,
    reason = "criterion_group! generates an undocumented pub fn"
)]

#[path = "../tests/common/mod.rs"]
mod common;

use common::{complete, cycle, sparse_random_graph};
use criterion::{Criterion, criterion_group, criterion_main};
use maxcut_sdp::maxcut::{Backend, BmConfig, GwConfig, gw_maxcut};
use maxcut_sdp::sdp::SdpProblem;
use maxcut_sdp::sdp::bm::{BmSolver, BmSolverConfig};
use maxcut_sdp::sdp::ipm::{IpmConfig, IpmSolver};
use maxcut_sdp::sdp::linalg::DenseMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Samples for the benchmarks whose single run takes a large fraction of a second.
const SLOW_SAMPLES: usize = 10;

/// `L(K_n) + I`, which is symmetric positive definite.
fn spd_matrix(n: usize) -> DenseMatrix {
    let mut a = complete(n).laplacian();
    for i in 0..n {
        a.set(i, i, a.get(i, i) + 1.0);
    }
    a
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    for n in [5, 10, 20, 50] {
        let a = spd_matrix(n);
        let _ = group.bench_function(format!("{n}x{n}"), |b| b.iter(|| a.cholesky()));
    }
    group.finish();
}

fn bench_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian");
    for n in [5, 10, 20, 50] {
        let g = complete(n);
        let _ = group.bench_function(format!("K{n}"), |b| b.iter(|| g.laplacian()));
    }
    group.finish();
}

fn bench_eigenvalues(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigenvalues");
    for n in [5, 10, 20] {
        let l = cycle(n).laplacian();
        let _ = group.bench_function(format!("C{n}_laplacian"), |b| {
            b.iter(|| l.eigenvalues_symmetric());
        });
    }
    group.finish();
}

fn bench_min_eigenvalue(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_eigenvalue");
    let _ = group.sample_size(SLOW_SAMPLES);
    for n in [200, 500] {
        let l = sparse_random_graph(n, 1).laplacian();
        let _ = group.bench_function(format!("sparse_{n}_laplacian"), |b| {
            b.iter(|| l.min_eigenvalue());
        });
    }
    group.finish();
}

fn bench_ipm(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipm_solve");
    let _ = group.sample_size(SLOW_SAMPLES);
    let solver = IpmSolver::new(IpmConfig::default());
    for n in [21, 51] {
        let problem = SdpProblem::maxcut_relaxation(&cycle(n))
            .unwrap_or_else(|e| panic!("relaxation failed: {e}"));
        let _ = group.bench_function(format!("C{n}"), |b| b.iter(|| solver.solve(&problem)));
    }
    group.finish();
}

fn bench_bm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm");
    let _ = group.sample_size(SLOW_SAMPLES);
    let graph = sparse_random_graph(1000, 1);
    let solver = BmSolver::new(BmSolverConfig::default());
    let _ = group.bench_function("solve_sparse_1000", |b| {
        b.iter(|| solver.solve_maxcut(&graph, &mut StdRng::seed_from_u64(1), |_| {}));
    });
    let config = GwConfig {
        seed: Some(1),
        backend: Backend::Bm(BmConfig::default()),
        ..GwConfig::default()
    };
    let _ = group.bench_function("solve_and_round_sparse_1000", |b| {
        b.iter(|| gw_maxcut(&graph, &config, |_| {}));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cholesky,
    bench_laplacian,
    bench_eigenvalues,
    bench_min_eigenvalue,
    bench_ipm,
    bench_bm
);
criterion_main!(benches);
