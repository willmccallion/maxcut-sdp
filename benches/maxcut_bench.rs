//! Criterion benchmarks for the maxcut-sdp pipeline.

use criterion::{Criterion, criterion_group, criterion_main};
use maxcut_sdp::bench::{generate_complete_graph, generate_cycle};
use maxcut_sdp::sdp::linalg::DenseMatrix;

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    for &n in &[5, 10, 20, 50] {
        let id = format!("{n}x{n}");

        // Build a known SPD matrix: A = I + ones (diagonally dominant)
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i + j * n] = 1.0;
            }
            data[i + i * n] += n as f64;
        }
        let m = DenseMatrix::from_vec(n, n, data).unwrap_or_else(|_| panic!("from_vec failed"));

        group.bench_function(&id, |b| {
            b.iter(|| {
                let _ = m.cholesky();
            });
        });
    }
    drop(group);
}

fn bench_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("laplacian");
    for &n in &[5, 10, 20, 50] {
        let g = generate_complete_graph(n);
        group.bench_function(&format!("K{n}"), |b| {
            b.iter(|| {
                let _ = g.laplacian();
            });
        });
    }
    drop(group);
}

fn bench_eigenvalues(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigenvalues");
    for &n in &[5, 10, 20] {
        let g = generate_cycle(n);
        let l = g.laplacian();
        group.bench_function(&format!("C{n}_laplacian"), |b| {
            b.iter(|| {
                let _ = l.eigenvalues_symmetric();
            });
        });
    }
    drop(group);
}

criterion_group!(benches, bench_cholesky, bench_laplacian, bench_eigenvalues);
criterion_main!(benches);
