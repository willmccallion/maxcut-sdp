//! The smallest eigenvalue against the full Jacobi spectrum and known cases.

use maxcut_sdp::GwError;
use maxcut_sdp::sdp::linalg::DenseMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_symmetric(n: usize, seed: u64) -> DenseMatrix {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = DenseMatrix::zeros(n, n);
    for j in 0..n {
        for i in j..n {
            let entry = rng.random_range(-1.0..1.0);
            a.set(i, j, entry);
            a.set(j, i, entry);
        }
    }
    a
}

fn min_eigenvalue(a: &DenseMatrix) -> f64 {
    a.min_eigenvalue().unwrap_or_else(|e| panic!("{e}"))
}

fn jacobi_min_eigenvalue(a: &DenseMatrix) -> f64 {
    let eigenvalues = a.eigenvalues_symmetric().unwrap_or_else(|e| panic!("{e}"));
    eigenvalues.into_iter().fold(f64::INFINITY, f64::min)
}

fn from_columns(n: usize, entries: Vec<f64>) -> DenseMatrix {
    DenseMatrix::from_vec(n, n, entries).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn agrees_with_jacobi_on_random_symmetric_matrices() {
    for (n, seed) in [(2, 1), (3, 2), (7, 3), (40, 4), (120, 5)] {
        let a = random_symmetric(n, seed);

        let (fast, jacobi) = (min_eigenvalue(&a), jacobi_min_eigenvalue(&a));

        assert!((fast - jacobi).abs() < 1e-11, "n={n}: {fast} vs {jacobi}");
    }
}

#[test]
fn finds_the_smallest_entry_of_a_diagonal_matrix_wherever_it_is() {
    let a = from_columns(3, vec![4.0, 0.0, 0.0, 0.0, -2.5, 0.0, 0.0, 0.0, 1.0]);

    assert_eq!(min_eigenvalue(&a), -2.5);
}

#[test]
fn handles_a_column_that_needs_no_reflection() {
    // Two decoupled blocks: [[2, 1], [1, 2]] (eigenvalues 1, 3) and [[0.5]].
    let a = from_columns(3, vec![2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.5]);

    assert!((min_eigenvalue(&a) - 0.5).abs() < 1e-15);
}

#[test]
fn finds_a_repeated_smallest_eigenvalue() {
    let a = DenseMatrix::identity(6).scale(-3.0);

    assert_eq!(min_eigenvalue(&a), -3.0);
}

#[test]
fn is_the_entry_of_a_one_by_one_matrix() {
    let a = from_columns(1, vec![7.25]);

    assert_eq!(min_eigenvalue(&a), 7.25);
}

#[test]
fn is_accurate_at_any_scale() {
    let a = random_symmetric(30, 6);
    let expected = jacobi_min_eigenvalue(&a);

    for scale in [1e-305, 1e-200, 1e200, 1e305] {
        let scaled = min_eigenvalue(&a.scale(scale)) / scale;

        assert!(
            (scaled - expected).abs() < 1e-11,
            "scale {scale:e}: {scaled}"
        );
    }
}

#[test]
fn is_zero_for_the_zero_matrix() {
    let a = DenseMatrix::zeros(4, 4);

    assert_eq!(min_eigenvalue(&a), 0.0);
}

#[test]
fn rejects_an_empty_matrix() {
    let a = DenseMatrix::zeros(0, 0);

    assert!(matches!(a.min_eigenvalue(), Err(GwError::InvalidMatrix(_))));
}

#[test]
fn rejects_a_non_square_matrix() {
    let a = DenseMatrix::zeros(2, 3);

    assert!(matches!(a.min_eigenvalue(), Err(GwError::InvalidMatrix(_))));
}

#[test]
fn rejects_a_non_finite_entry() {
    let mut a = DenseMatrix::identity(3);
    a.set(1, 1, f64::NAN);

    assert!(matches!(a.min_eigenvalue(), Err(GwError::InvalidMatrix(_))));
}
