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
//! See [`MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/sdp/MATH.md) §1–§2 for definitions and duality theory.

pub mod bm;
pub mod ipm;
pub mod linalg;

use crate::GwError;
use crate::graph::Graph;
use linalg::DenseMatrix;

/// A single affine constraint `tr(A_i X) = b_i` in the SDP.
#[derive(Debug, Clone)]
pub struct SdpConstraint {
    a: ConstraintMatrix,
    b: f64,
}

impl SdpConstraint {
    /// The constraint matrix `A_i`.
    pub const fn matrix(&self) -> &ConstraintMatrix {
        &self.a
    }

    /// The right-hand side scalar `b_i`.
    pub const fn rhs(&self) -> f64 {
        self.b
    }
}

/// The matrix `A_i` of a constraint, kept in the form that is cheap for it.
#[derive(Debug, Clone)]
pub enum ConstraintMatrix {
    /// `e_k e_kᵀ`, which fixes the diagonal entry `X_kk`. Held as `k`, so
    /// Max-Cut's `n` constraints take `O(n)` memory instead of `O(n³)`.
    DiagonalUnit(usize),
    /// A general symmetric `n × n` matrix.
    Dense(DenseMatrix),
}

impl ConstraintMatrix {
    /// `⟨A, Y⟩ = tr(A Y)`.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if a dense `A` and `Y` differ in shape.
    pub fn inner_product(&self, y: &DenseMatrix) -> Result<f64, GwError> {
        match self {
            Self::DiagonalUnit(k) => Ok(y.get(*k, *k)),
            Self::Dense(a) => a.inner_product(y),
        }
    }

    /// Adds `coeff · A` to `target`.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if a dense `A` and `target` differ in shape.
    pub fn add_scaled_to(&self, target: &mut DenseMatrix, coeff: f64) -> Result<(), GwError> {
        match self {
            Self::DiagonalUnit(k) => {
                target.set(*k, *k, target.get(*k, *k) + coeff);
                Ok(())
            }
            Self::Dense(a) => target.add_assign(&a.scale(coeff)),
        }
    }

    /// Whether this is a matrix of an `n × n` problem.
    const fn fits(&self, n: usize) -> bool {
        match self {
            Self::DiagonalUnit(k) => *k < n,
            Self::Dense(a) => a.rows() == n && a.cols() == n,
        }
    }
}

/// The Max-Cut SDP's cost matrix `C = −L/4` for the Laplacian `L` of
/// `graph`: minimizing `tr(C X)` maximizes `tr(L X) / 4`.
pub(crate) fn maxcut_cost(graph: &Graph) -> DenseMatrix {
    graph.laplacian().scale(-0.25)
}

/// A semidefinite program in standard form.
///
/// Minimize `tr(C X)` subject to `tr(A_i X) = b_i` for all constraints,
/// and `X ≽ 0` (positive semidefinite).
#[derive(Debug, Clone)]
pub struct SdpProblem {
    c: DenseMatrix,
    constraints: Vec<SdpConstraint>,
}

/// The solution returned by the SDP solver.
#[derive(Debug, Clone)]
pub struct SdpSolution {
    /// Primal matrix `X` (symmetric PSD, `n × n`).
    pub x: DenseMatrix,
    /// Dual variables `y ∈ R^m`.
    pub y: Vec<f64>,
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
    /// Creates a new SDP with the symmetric cost matrix `c` and no constraints.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidMatrix`] if `c` is not square.
    pub fn new(c: DenseMatrix) -> Result<Self, GwError> {
        c.ensure_square("an SDP cost matrix")?;
        Ok(Self {
            c,
            constraints: Vec::new(),
        })
    }

    /// The cost matrix `C`.
    pub const fn cost(&self) -> &DenseMatrix {
        &self.c
    }

    /// The cost matrix `C`, taken out of the problem.
    pub fn into_cost(self) -> DenseMatrix {
        self.c
    }

    /// The affine equality constraints.
    pub fn constraints(&self) -> &[SdpConstraint] {
        &self.constraints
    }

    /// Adds an affine constraint `tr(A X) = b`.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidMatrix`] if `a` is not a matrix of this
    /// problem's dimension.
    pub fn add_constraint(&mut self, a: ConstraintMatrix, b: f64) -> Result<(), GwError> {
        if !a.fits(self.dimension()) {
            return Err(GwError::InvalidMatrix(format!(
                "constraint matrix does not fit an {n} x {n} problem",
                n = self.dimension()
            )));
        }
        self.constraints.push(SdpConstraint { a, b });
        Ok(())
    }

    /// The matrix dimension `n`.
    pub const fn dimension(&self) -> usize {
        self.c.rows()
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
    /// `1` at position `(i, i)`), with `b_i = 1`, held as
    /// [`ConstraintMatrix::DiagonalUnit`].
    ///
    /// See [`maxcut/MATH.md`](https://github.com/willmccallion/maxcut-sdp/blob/main/src/maxcut/MATH.md) §3 for the derivation.
    ///
    /// # Errors
    ///
    /// Returns an error if graph Laplacian construction fails.
    pub fn maxcut_relaxation(graph: &Graph) -> Result<Self, GwError> {
        let n = graph.num_vertices();
        if n == 0 {
            return Err(GwError::InvalidGraph("empty graph".to_string()));
        }

        let mut problem = Self::new(maxcut_cost(graph))?;

        // Diagonal constraints: tr(E_ii X) = X_ii = 1
        for i in 0..n {
            problem.add_constraint(ConstraintMatrix::DiagonalUnit(i), 1.0)?;
        }

        Ok(problem)
    }
}
