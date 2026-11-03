//! SDP problem formulation and data types.
//!
//! Defines the standard-form semidefinite program:
//!
//! ```text
//! min  tr(C X)
//! s.t. tr(A_i X) = b_i,  i = 1, ..., m
//!      X ≽ 0
//! ```
//!
//! The key type [`SdpProblem`] packages the cost matrix `C` and the affine
//! constraints `(A_i, b_i)`. The [`SdpProblem::maxcut_relaxation`] constructor
//! builds the Max-Cut SDP directly from a [`Graph`].
//!
//! See [`MATH.md`](./MATH.md) §1–§2 for definitions and duality theory.

pub mod linalg;
pub mod solver;
pub mod solver_bm;

use crate::GwError;
use crate::graph::Graph;
use linalg::DenseMatrix;

/// A single affine constraint `tr(A_i X) = b_i` in the SDP.
#[derive(Debug, Clone)]
pub struct SdpConstraint {
    /// The constraint matrix `A_i`.
    pub a: DenseMatrix,
    /// The right-hand side scalar `b_i`.
    pub b: f64,
}

/// A semidefinite program in standard form.
///
/// Minimize `tr(C X)` subject to `tr(A_i X) = b_i` for all constraints,
/// and `X ≽ 0` (positive semidefinite).
#[derive(Debug, Clone)]
pub struct SdpProblem {
    /// Cost matrix `C` (symmetric, `n × n`).
    pub c: DenseMatrix,
    /// Affine equality constraints.
    pub constraints: Vec<SdpConstraint>,
    /// Matrix dimension `n`.
    pub n: usize,
}

/// The solution returned by the SDP solver.
#[derive(Debug, Clone)]
pub struct SdpSolution {
    /// Primal matrix `X` (symmetric PSD, `n × n`).
    pub x: DenseMatrix,
    /// Dual variables `y ∈ R^m`.
    pub y: Vec<f64>,
    /// Dual slack matrix `S = C - Σ y_i A_i` (symmetric PSD, `n × n`).
    pub s: DenseMatrix,
    /// Primal objective value `tr(C X)`.
    pub primal_obj: f64,
    /// Dual objective value `b^T y`.
    pub dual_obj: f64,
    /// Duality gap `⟨X, S⟩ / n`.
    pub duality_gap: f64,
    /// Number of IPM iterations taken.
    pub iterations: usize,
}

impl SdpProblem {
    /// Creates a new SDP with cost matrix `c` and no constraints.
    ///
    /// The matrix dimension `n` is inferred from `c`.
    pub const fn new(c: DenseMatrix) -> Self {
        let n = c.rows();
        Self {
            c,
            constraints: Vec::new(),
            n,
        }
    }

    /// Adds an affine constraint `tr(A X) = b`.
    pub fn add_constraint(&mut self, a: DenseMatrix, b: f64) {
        self.constraints.push(SdpConstraint { a, b });
    }

    /// Returns the number of constraints `m`.
    pub const fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Returns the matrix dimension `n`.
    pub const fn dimension(&self) -> usize {
        self.n
    }

    /// Constructs the Max-Cut SDP relaxation from a graph.
    ///
    /// The SDP is:
    /// ```text
    /// min  tr(C X)        where C = -(1/4) L  (L = graph Laplacian)
    /// s.t. X_ii = 1       for all i = 1, ..., n
    ///      X ≽ 0
    /// ```
    ///
    /// Each diagonal constraint uses `A_i = e_i e_i^T` (the matrix with a single
    /// `1` at position `(i, i)`), with `b_i = 1`.
    ///
    /// See [`maxcut/MATH.md`](../maxcut/MATH.md) §3 for the derivation.
    ///
    /// # Errors
    ///
    /// Returns an error if graph Laplacian construction fails.
    pub fn maxcut_relaxation(graph: &Graph) -> Result<Self, GwError> {
        let n = graph.num_vertices();
        if n == 0 {
            return Err(GwError::InvalidGraph("empty graph".to_string()));
        }

        // C = -(1/4) L (minimizing tr(CX) = maximizing (1/4) tr(LX))
        let c = graph.laplacian().scale(-0.25);

        let mut problem = Self::new(c);

        // Diagonal constraints: tr(E_ii X) = X_ii = 1
        for i in 0..n {
            let mut a = DenseMatrix::zeros(n, n);
            a.set(i, i, 1.0);
            problem.add_constraint(a, 1.0);
        }

        Ok(problem)
    }
}
