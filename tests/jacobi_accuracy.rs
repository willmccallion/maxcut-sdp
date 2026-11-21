//! Accuracy of the Jacobi eigendecomposition at small scales.
//!
//! The failure after `MAX_JACOBI_SWEEPS` is not tested: no finite symmetric
//! matrix is known to need that many sweeps, and NaNs are refused up front.

mod common;

use common::erdos_renyi;
use maxcut_sdp::GwError;
use maxcut_sdp::sdp::linalg::DenseMatrix;

fn frobenius(m: &DenseMatrix) -> f64 {
    m.data().iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn random_laplacian(n: usize, seed: u64) -> DenseMatrix {
    erdos_renyi(n, 0.1, 1..=10, seed).laplacian()
}

/// `‖A Q − Q Λ‖_F / ‖A‖_F`.
fn relative_residual(a: &DenseMatrix) -> f64 {
    let Ok((eigenvalues, q)) = a.eigendecomposition_symmetric() else {
        panic!("eigendecomposition failed");
    };
    let Ok(aq) = a.mul(&q) else {
        panic!("A * Q failed");
    };
    let mut q_lambda = q;
    for (j, &lambda) in eigenvalues.iter().enumerate() {
        for i in 0..a.rows() {
            q_lambda.set(i, j, q_lambda.get(i, j) * lambda);
        }
    }
    let Ok(diff) = aq.sub(&q_lambda) else {
        panic!("A Q - Q Λ failed");
    };
    frobenius(&diff) / frobenius(a)
}

#[test]
fn eigendecomposition_of_unit_scale_laplacian_is_accurate() {
    let a = random_laplacian(80, 3);

    let residual = relative_residual(&a);

    assert!(residual < 1e-10, "relative residual {residual:e}");
}

#[test]
fn eigendecomposition_accuracy_is_independent_of_matrix_scale() {
    let a = random_laplacian(80, 3).scale(1e-12);

    let residual = relative_residual(&a);

    assert!(residual < 1e-10, "relative residual {residual:e}");
}

#[test]
fn tiny_off_diagonal_matrix_has_plus_minus_eigenvalues() {
    let Ok(a) = DenseMatrix::from_vec(2, 2, vec![0.0, 1e-15, 1e-15, 0.0]) else {
        panic!("four values fill a 2 x 2 matrix");
    };

    let Ok(eigenvalues) = a.eigenvalues_symmetric() else {
        panic!("eigenvalues failed");
    };

    assert!(
        (eigenvalues[0] + 1e-15).abs() < 1e-20 && (eigenvalues[1] - 1e-15).abs() < 1e-20,
        "eigenvalues {eigenvalues:?}, expected [-1e-15, 1e-15]"
    );
}

#[test]
fn eigendecomposition_rejects_a_non_finite_entry() {
    let Ok(a) = DenseMatrix::from_vec(2, 2, vec![1.0, f64::NAN, f64::NAN, 2.0]) else {
        panic!("four values fill a 2 x 2 matrix");
    };

    let result = a.eigendecomposition_symmetric();

    assert!(
        matches!(result, Err(GwError::InvalidMatrix(_))),
        "{:?}",
        result.map(|(values, _)| values)
    );
}
