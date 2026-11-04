//! Accuracy of the Jacobi eigendecomposition at small scales.

use maxcut_sdp::graph::Graph;
use maxcut_sdp::sdp::linalg::DenseMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn frobenius(m: &DenseMatrix) -> f64 {
    m.data().iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn random_laplacian(n: usize, seed: u64) -> DenseMatrix {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g = Graph::new(n);
    for u in 0..n {
        for v in u + 1..n {
            if rng.random::<f64>() < 0.1 {
                let w = f64::from(rng.random_range(1..=10_u32));
                if let Err(e) = g.add_edge(u, v, w) {
                    panic!("valid edge rejected: {e}");
                }
            }
        }
    }
    g.laplacian()
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
fn eigendecomposition_reports_failure_instead_of_returning_unconverged_values() {
    let Ok(a) = DenseMatrix::from_vec(2, 2, vec![1.0, f64::NAN, f64::NAN, 2.0]) else {
        panic!("four values fill a 2 x 2 matrix");
    };

    let result = a.eigendecomposition_symmetric();

    assert!(
        result.is_err(),
        "returned Ok after exhausting the sweep cap: {:?}",
        result.map(|(values, _)| values)
    );
}
