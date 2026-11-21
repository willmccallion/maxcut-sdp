//! The errors `DenseMatrix` reports for operands that do not fit.

use maxcut_sdp::GwError;
use maxcut_sdp::sdp::linalg::DenseMatrix;

fn zeros_2x3() -> DenseMatrix {
    DenseMatrix::zeros(2, 3)
}

fn zeros_3x2() -> DenseMatrix {
    DenseMatrix::zeros(3, 2)
}

const fn is_shape_mismatch_2x3_vs_3x2(error: &GwError) -> bool {
    matches!(
        error,
        GwError::ShapeMismatch {
            expected: (2, 3),
            got: (3, 2)
        }
    )
}

#[test]
fn from_vec_rejects_data_that_does_not_fill_the_shape() {
    let result = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0]);

    assert!(matches!(result, Err(GwError::InvalidMatrix(_))));
}

#[test]
fn mul_rejects_a_mismatched_inner_dimension() {
    let result = zeros_2x3().mul(&zeros_2x3());

    assert!(matches!(
        result,
        Err(GwError::DimensionMismatch {
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn element_wise_operations_report_both_shapes() {
    let mut a = zeros_2x3();
    let b = zeros_3x2();

    let errors = [
        a.inner_product(&b).err(),
        a.sub(&b).err(),
        a.add_assign(&b).err(),
        a.sub_assign(&b).err(),
    ];

    for error in errors {
        assert!(
            error.as_ref().is_some_and(is_shape_mismatch_2x3_vs_3x2),
            "{error:?}"
        );
    }
}

#[test]
fn factorizations_reject_a_non_square_matrix() {
    let a = zeros_2x3();

    assert!(matches!(a.cholesky(), Err(GwError::InvalidMatrix(_))));
    assert!(matches!(
        a.eigendecomposition_symmetric(),
        Err(GwError::InvalidMatrix(_))
    ));
}
