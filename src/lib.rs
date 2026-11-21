//! # maxcut-sdp
//!
//! A from-scratch semidefinite programming (SDP) solver with Goemans-Williamson
//! Max-Cut approximation.
//!
//! This crate implements the full pipeline:
//! 1. Parse a weighted graph from Biq Mac or DIMACS format
//! 2. Formulate the Max-Cut SDP relaxation (Laplacian-based)
//! 3. Solve via a primal-dual interior point method with Nesterov-Todd
//!    scaling, or Burer-Monteiro low-rank factorization for large graphs
//! 4. Round the SDP solution with random hyperplanes or outward rotations,
//!    then improve each cut by local search
//! 5. Return the best partition across multiple trials
//!
//! All linear algebra is implemented from scratch — no external BLAS, LAPACK,
//! or optimization libraries.

pub mod graph;
pub mod maxcut;
pub mod parser;
mod random;
pub mod sdp;

use std::fmt;

// Re-exports of key public types.
pub use graph::Graph;
pub use maxcut::MaxCutResult;
pub use sdp::bm::{BmSolution, BmSolver, BmSolverConfig};
pub use sdp::{SdpProblem, SdpSolution};

/// Errors that can occur throughout the maxcut-sdp pipeline.
#[derive(Debug)]
pub enum GwError {
    /// The graph is invalid (e.g., self-loop, out-of-range vertex index).
    InvalidGraph(String),

    /// A matrix operation failed (e.g., non-square input to Cholesky).
    InvalidMatrix(String),

    /// The SDP solver failed to converge or encountered a numerical issue.
    SolverFailed(String),

    /// A step of the interior point method failed on the error `cause`.
    SolverStep {
        /// The step that failed.
        stage: &'static str,
        /// What went wrong in it.
        cause: Box<Self>,
    },

    /// Failed to parse an input file.
    ParseError(String),

    /// A matrix expected to be positive definite is not.
    NotPositiveDefinite,

    /// A matrix's inner dimension, or a vector's length, does not match
    /// the operation's other operand.
    DimensionMismatch {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension encountered.
        got: usize,
    },

    /// Two matrices an element-wise operation combines differ in shape.
    ShapeMismatch {
        /// The `(rows, cols)` the operation needed.
        expected: (usize, usize),
        /// The `(rows, cols)` it was given.
        got: (usize, usize),
    },
}

impl fmt::Display for GwError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(msg) => write!(f, "invalid graph: {msg}"),
            Self::InvalidMatrix(msg) => write!(f, "invalid matrix: {msg}"),
            Self::SolverFailed(msg) => write!(f, "solver failed: {msg}"),
            Self::SolverStep { stage, cause } => write!(f, "solver failed at {stage}: {cause}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::NotPositiveDefinite => write!(f, "matrix is not positive definite"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::ShapeMismatch {
                expected: (expected_rows, expected_cols),
                got: (rows, cols),
            } => write!(
                f,
                "shape mismatch: expected {expected_rows}x{expected_cols}, got {rows}x{cols}"
            ),
        }
    }
}

impl std::error::Error for GwError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SolverStep { cause, .. } => Some(cause.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn a_failed_solver_step_names_the_step_and_keeps_its_cause() {
        let error = GwError::SolverStep {
            stage: "Schur Cholesky",
            cause: Box::new(GwError::NotPositiveDefinite),
        };

        assert_eq!(
            error.to_string(),
            "solver failed at Schur Cholesky: matrix is not positive definite"
        );
        assert!(matches!(
            error.source().and_then(|cause| cause.downcast_ref()),
            Some(GwError::NotPositiveDefinite)
        ));
    }
}
