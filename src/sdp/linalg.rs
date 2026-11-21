//! Dense linear algebra for semidefinite programming.
//!
//! Provides [`DenseMatrix`], a column-major dense matrix type, along with all
//! the linear algebra primitives needed by the interior point solver:
//!
//! - Cholesky factorization (`LL^T`) and triangular solves
//! - Eigendecomposition of symmetric matrices (for NT scaling and step sizes)
//! - Matrix square root and regularized inverse (for Nesterov-Todd scaling)
//! - The smallest eigenvalue alone (for step sizes and certified bounds)
//! - Standard operations: multiply, subtract, scale, inner product
//!
//! All operations are tuned for cache efficiency on column-major layout.
//! Matrix multiply uses a blocked (tiled) algorithm to keep working sets in L1.
//! Eigendecomposition uses the Jacobi method with threshold sweeps; the
//! smallest eigenvalue alone, by Householder tridiagonalization and bisection.
//!
//! See [`MATH.md`](./MATH.md) §6 for the full list of operations and where
//! each is used in the solver.

use crate::GwError;
use rayon::prelude::*;

/// Block size for tiled matrix multiply. 64 × 64 f64 blocks = 32 KiB,
/// fits comfortably in a 48 KiB L1d cache with room for two source tiles.
const BLOCK: usize = 64;

/// Jacobi sweeps after which an eigendecomposition that has not converged
/// is reported as failed.
const MAX_JACOBI_SWEEPS: usize = 100;

/// Off-diagonal Frobenius norm, relative to the whole matrix's, at which
/// Jacobi sweeps have converged; relative so any scale is solved alike.
const JACOBI_REL_TOL: f64 = 1e-14;

/// Opening Jacobi sweeps that skip entries small against the off-diagonal
/// norm; later sweeps rotate every nonzero entry.
const THRESHOLD_SWEEPS: usize = 3;

/// An opening sweep skips entries below this fraction of the off-diagonal
/// norm divided by `n`.
const THRESHOLD_FACTOR: f64 = 0.2;

/// Diagonal entries closer than this rotate by `π/4`, as if equal.
const EQUAL_DIAGONAL_GAP: f64 = 1e-300;

/// A dense matrix stored in **column-major** order.
///
/// Element `(i, j)` is stored at index `i + j * rows` in the backing `Vec<f64>`.
/// This matches Fortran/LAPACK convention and is cache-friendly for column access.
#[derive(Clone, PartialEq)]
pub struct DenseMatrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl std::fmt::Debug for DenseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish_non_exhaustive()
    }
}

/// Column-major index. Inlined everywhere so the compiler folds it into LEA.
#[inline(always)]
const fn idx(i: usize, j: usize, rows: usize) -> usize {
    i + j * rows
}

/// Returns a contiguous slice for column `j` of a column-major matrix.
#[inline(always)]
pub(crate) fn col(data: &[f64], j: usize, rows: usize) -> &[f64] {
    let start = j * rows;
    &data[start..start + rows]
}

/// Returns a mutable contiguous slice for column `j` of a column-major matrix.
#[inline(always)]
pub(crate) fn col_mut(data: &mut [f64], j: usize, rows: usize) -> &mut [f64] {
    let start = j * rows;
    &mut data[start..start + rows]
}

/// Dot product of two equal-length slices, 4-way unrolled for FMA vectorization.
#[inline(always)]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut s3 = 0.0_f64;
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        s0 = a[base].mul_add(b[base], s0);
        s1 = a[base + 1].mul_add(b[base + 1], s1);
        s2 = a[base + 2].mul_add(b[base + 2], s2);
        s3 = a[base + 3].mul_add(b[base + 3], s3);
    }
    for i in (chunks * 4)..n {
        s0 = a[i].mul_add(b[i], s0);
    }
    (s0 + s1) + (s2 + s3)
}

/// `y[i] += alpha * x[i]` for all i (DAXPY kernel).
#[inline(always)]
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        y[i] = alpha.mul_add(x[i], y[i]);
    }
}

impl DenseMatrix {
    /// Creates a new zero matrix of size `rows × cols`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[idx(i, i, n)] = 1.0;
        }
        m
    }

    /// Creates a matrix from a flat `Vec<f64>` in column-major order.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidMatrix`] if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, GwError> {
        if data.len() != rows * cols {
            return Err(GwError::InvalidMatrix(format!(
                "expected {} elements for {rows}x{cols} matrix, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { data, rows, cols })
    }

    /// Returns the element at position `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[idx(i, j, self.rows)]
    }

    /// Sets the element at position `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= rows` or `j >= cols`.
    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[idx(i, j, self.rows)] = val;
    }

    /// Returns the number of rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the raw backing data slice.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable reference to the raw backing data.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns the transpose of this matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let (r, c) = (self.rows, self.cols);
        let mut out = Self::zeros(c, r);
        // Iterate output columns = input rows. Writing contiguous output
        // columns is cache-friendly for the output; reading along input
        // rows strides by `r` but each output column is a full pass.
        for j in 0..r {
            let out_col = col_mut(&mut out.data, j, c);
            for (i, val) in out_col.iter_mut().enumerate() {
                *val = self.data[idx(j, i, r)];
            }
        }
        out
    }

    /// Computes the matrix product `self * other` using blocked (tiled) multiply.
    ///
    /// The blocking strategy keeps `BLOCK × BLOCK` sub-tiles in L1 cache.
    /// Inner loop is column-oriented (axpy-based) for column-major layout.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::DimensionMismatch`] if `self.cols != other.rows`.
    pub fn mul(&self, other: &Self) -> Result<Self, GwError> {
        let (m, k, n) = (self.rows, self.cols, other.cols);
        if k != other.rows {
            return Err(GwError::DimensionMismatch {
                expected: k,
                got: other.rows,
            });
        }
        let mut c = Self::zeros(m, n);

        // Blocked multiply: C += A[:,kb..kb+BLOCK] * B[kb..kb+BLOCK, :]
        // The jb loop iterates over output column blocks,
        // the kb loop over the shared dimension,
        // the ib loop over output row blocks.
        // Inner kernel: rank-1-style column update via axpy.
        let a = &self.data;
        let b = &other.data;
        let cd = &mut c.data;

        for jb in (0..n).step_by(BLOCK) {
            let j_end = (jb + BLOCK).min(n);
            for kb in (0..k).step_by(BLOCK) {
                let k_end = (kb + BLOCK).min(k);
                for j in jb..j_end {
                    let c_col = col_mut(cd, j, m);
                    for p in kb..k_end {
                        let b_pj = b[idx(p, j, k)];
                        let a_col = col(a, p, m);
                        axpy(b_pj, a_col, c_col);
                    }
                }
            }
        }
        Ok(c)
    }

    /// The congruence `self · middle · self`, multiplied left to right.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::DimensionMismatch`] if the shapes do not chain.
    pub(crate) fn congruence(&self, middle: &Self) -> Result<Self, GwError> {
        self.mul(middle)?.mul(self)
    }

    /// Makes a square matrix exactly symmetric: `A ← (A + Aᵀ) / 2`.
    pub(crate) fn symmetrize(&mut self) {
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = f64::midpoint(self.data[idx(i, j, n)], self.data[idx(j, i, n)]);
                self.data[idx(i, j, n)] = avg;
                self.data[idx(j, i, n)] = avg;
            }
        }
    }

    /// Computes `self * other^T` efficiently for symmetric products.
    ///
    /// Avoids forming the explicit transpose. Both matrices accessed column-wise.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::DimensionMismatch`] if `self.cols != other.cols`.
    pub fn mul_transpose(&self, other: &Self) -> Result<Self, GwError> {
        let (m, k) = (self.rows, self.cols);
        let n = other.rows;
        if k != other.cols {
            return Err(GwError::DimensionMismatch {
                expected: k,
                got: other.cols,
            });
        }
        let mut c = Self::zeros(m, n);
        // C[i,j] = row_i(self) · row_j(other) = Σ_p self[i,p]*other[j,p]
        // Rewrite as column-oriented: for each column p, outer-product update
        // C += self[:,p] * other[:,p]^T  — but that's O(m·n) per p.
        // Instead use the blocked approach on the k-dimension.
        let a = &self.data;
        let bt = &other.data;
        let cd = &mut c.data;

        for kb in (0..k).step_by(BLOCK) {
            let k_end = (kb + BLOCK).min(k);
            for j in 0..n {
                let c_col = col_mut(cd, j, m);
                for p in kb..k_end {
                    let b_jp = bt[idx(j, p, n)];
                    let a_col = col(a, p, m);
                    axpy(b_jp, a_col, c_col);
                }
            }
        }
        Ok(c)
    }

    /// Computes the matrix difference `self - other`.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if the shapes differ.
    pub fn sub(&self, other: &Self) -> Result<Self, GwError> {
        let mut difference = self.clone();
        difference.sub_assign(other)?;
        Ok(difference)
    }

    /// Adds `other` to `self` in-place.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if the shapes differ.
    pub fn add_assign(&mut self, other: &Self) -> Result<(), GwError> {
        self.zip_assign(other, |a, b| a + b)
    }

    /// Subtracts `other` from `self` in-place.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if the shapes differ.
    pub fn sub_assign(&mut self, other: &Self) -> Result<(), GwError> {
        self.zip_assign(other, |a, b| a - b)
    }

    /// Replaces each element `a` with `combine(a, b)` for `other`'s `b`.
    fn zip_assign(
        &mut self,
        other: &Self,
        combine: impl Fn(f64, f64) -> f64,
    ) -> Result<(), GwError> {
        self.ensure_same_shape(other)?;
        for (a, &b) in self.data.iter_mut().zip(&other.data) {
            *a = combine(*a, b);
        }
        Ok(())
    }

    /// Returns a new matrix with every element multiplied by `scalar`.
    #[must_use]
    pub fn scale(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Self {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Multiplies every element by `scalar` in-place.
    pub fn scale_assign(&mut self, scalar: f64) {
        for x in &mut self.data {
            *x *= scalar;
        }
    }

    /// Computes the Frobenius inner product `⟨self, other⟩ = tr(self^T · other)`.
    ///
    /// Equal to the dot product of the flattened data vectors — computed as a
    /// single contiguous pass for maximum vectorization. Used extensively in the
    /// Schur complement computation (see [`MATH.md`](./MATH.md) §3.4).
    ///
    /// # Errors
    ///
    /// Returns [`GwError::ShapeMismatch`] if the shapes differ.
    pub fn inner_product(&self, other: &Self) -> Result<f64, GwError> {
        self.ensure_same_shape(other)?;
        Ok(dot(&self.data, &other.data))
    }

    /// Computes the Cholesky factorization `self = L L^T`.
    ///
    /// Returns the lower-triangular factor `L`. The input must be symmetric
    /// positive definite. Only the lower triangle of `self` is read.
    ///
    /// Uses the column-oriented (left-looking) algorithm: for each column `j`,
    /// update it using all previous columns, then scale. This accesses memory
    /// in contiguous column strides — optimal for column-major layout.
    ///
    /// See [`MATH.md`](./MATH.md) §3.3 (NT scaling) and §3.4 (Schur complement solve).
    ///
    /// # Errors
    ///
    /// Returns [`GwError::NotPositiveDefinite`] if the matrix is not positive definite,
    /// or [`GwError::InvalidMatrix`] if the matrix is not square.
    pub fn cholesky(&self) -> Result<Self, GwError> {
        self.ensure_square("Cholesky")?;
        let n = self.rows;

        // Copy lower triangle into output.
        let mut l = Self::zeros(n, n);
        for j in 0..n {
            for i in j..n {
                l.data[idx(i, j, n)] = self.data[idx(i, j, n)];
            }
        }

        let d = &mut l.data;

        for j in 0..n {
            // Subtract contributions from previous columns:
            // L[j:n, j] -= L[j:n, 0:j] * L[j, 0:j]^T
            for k in 0..j {
                let l_jk = d[idx(j, k, n)];
                // Subtract l_jk * L[j:n, k] from L[j:n, j]
                for i in j..n {
                    d[idx(i, j, n)] = l_jk.mul_add(-d[idx(i, k, n)], d[idx(i, j, n)]);
                }
            }

            let diag = d[idx(j, j, n)];
            if diag <= 0.0 {
                return Err(GwError::NotPositiveDefinite);
            }
            let inv_sqrt = 1.0 / diag.sqrt();

            // Scale column j.
            for i in j..n {
                d[idx(i, j, n)] *= inv_sqrt;
            }
        }

        Ok(l)
    }

    /// Solves `A x = b` given the Cholesky factor `L` of `A` (where `A = L L^T`).
    ///
    /// Performs forward substitution (`L y = b`) then back substitution (`L^T x = y`).
    ///
    /// # Errors
    ///
    /// Returns [`GwError::DimensionMismatch`] if dimensions are incompatible.
    pub fn cholesky_solve(l: &Self, b: &[f64]) -> Result<Vec<f64>, GwError> {
        let n = l.rows;
        if n != l.cols || b.len() != n {
            return Err(GwError::DimensionMismatch {
                expected: n,
                got: b.len(),
            });
        }

        let d = &l.data;
        let mut x = b.to_vec();

        // Forward substitution: L y = b
        for j in 0..n {
            x[j] /= d[idx(j, j, n)];
            let xj = x[j];
            for i in (j + 1)..n {
                x[i] = d[idx(i, j, n)].mul_add(-xj, x[i]);
            }
        }

        // Back substitution: L^T x = y
        for j in (0..n).rev() {
            for i in (j + 1)..n {
                x[j] = d[idx(i, j, n)].mul_add(-x[i], x[j]);
            }
            x[j] /= d[idx(j, j, n)];
        }

        Ok(x)
    }

    /// Computes the eigendecomposition `self = Q Λ Q^T` of a symmetric matrix.
    ///
    /// Uses the classical Jacobi eigenvalue algorithm with cyclic-by-row sweeps
    /// and a threshold to skip near-zero rotations. Convergence is typically
    /// quadratic once the threshold kicks in.
    ///
    /// Returns `(eigenvalues, Q)` where `Q` is orthogonal and the eigenvalues
    /// correspond to the diagonal of `Λ`, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidMatrix`] if the matrix is not square or has
    /// a non-finite entry, and [`GwError::SolverFailed`] if the sweeps do not
    /// converge.
    pub fn eigendecomposition_symmetric(&self) -> Result<(Vec<f64>, Self), GwError> {
        self.ensure_square("eigendecomposition")?;
        let n = self.rows;
        if n == 0 {
            return Ok((Vec::new(), Self::zeros(0, 0)));
        }
        if n == 1 {
            return Ok((vec![self.data[0]], Self::identity(1)));
        }

        self.ensure_finite("eigendecomposition")?;

        // Rotations preserve the Frobenius norm, so this is the scale throughout.
        let tolerance = JACOBI_REL_TOL * self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut jacobi = Jacobi::new(self);
        jacobi.diagonalize(tolerance)?;
        Ok(jacobi.sorted_eigenpairs())
    }

    /// Computes all eigenvalues of a symmetric matrix.
    ///
    /// Delegates to [`eigendecomposition_symmetric`](Self::eigendecomposition_symmetric)
    /// and discards the eigenvectors.
    ///
    /// # Errors
    ///
    /// As [`eigendecomposition_symmetric`](Self::eigendecomposition_symmetric).
    pub fn eigenvalues_symmetric(&self) -> Result<Vec<f64>, GwError> {
        let (vals, _) = self.eigendecomposition_symmetric()?;
        Ok(vals)
    }

    /// Computes the minimum eigenvalue of a symmetric matrix.
    ///
    /// Reduces the matrix to tridiagonal form and bisects on its inertia,
    /// which takes `O(n³)` flops like Jacobi but with a far smaller constant.
    /// Used for step size selection (see [`MATH.md`](./MATH.md) §3.5) and to
    /// certify SDP bounds.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidMatrix`] if the matrix is empty, not square,
    /// or has a non-finite entry.
    pub fn min_eigenvalue(&self) -> Result<f64, GwError> {
        self.ensure_square("minimum eigenvalue")?;
        if self.rows == 0 {
            return Err(GwError::InvalidMatrix("empty matrix".to_string()));
        }
        self.ensure_finite("minimum eigenvalue")?;
        let largest = self.data.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if largest == 0.0 {
            return Ok(0.0);
        }
        // Normalizing by a power of two is exact and keeps squares from
        // underflowing or overflowing during the reduction.
        let exponent = largest.log2().floor() as i32;
        let normalized = self
            .data
            .iter()
            .map(|&x| scale_by_power_of_two(x, -exponent))
            .collect();
        let tridiagonal = Tridiagonal::reduce(normalized, self.rows);
        Ok(scale_by_power_of_two(
            tridiagonal.min_eigenvalue(),
            exponent,
        ))
    }

    /// Applies a scalar function `f` to the eigenvalues: `Q diag(f(λ_i)) Q^T`.
    ///
    /// This is the core primitive for `sqrt_spd` and `inverse_regularized`.
    ///
    /// # Errors
    ///
    /// Returns an error if eigendecomposition fails.
    pub fn apply_eigen_fn(&self, f: impl Fn(f64) -> Result<f64, GwError>) -> Result<Self, GwError> {
        let (eigenvalues, q) = self.eigendecomposition_symmetric()?;
        let n = q.rows;

        // Build Q · diag(f(λ)) — scale each column of Q by f(λ_j).
        let mut q_scaled = q.clone();
        for (j, &eigval) in eigenvalues.iter().enumerate() {
            let fval = f(eigval)?;
            let c = col_mut(&mut q_scaled.data, j, n);
            for x in c.iter_mut() {
                *x *= fval;
            }
        }

        // Result = Q_scaled · Q^T = Q diag(f(λ)) Q^T
        q_scaled.mul_transpose(&q)
    }

    /// Computes the symmetric positive definite square root `self^{1/2}`.
    ///
    /// Via eigendecomposition: if `self = Q Λ Q^T`, then `self^{1/2} = Q Λ^{1/2} Q^T`.
    ///
    /// Used in Nesterov-Todd scaling computation (see [`MATH.md`](./MATH.md) §3.3).
    ///
    /// # Errors
    ///
    /// Returns [`GwError::NotPositiveDefinite`] if any eigenvalue is negative.
    pub fn sqrt_spd(&self) -> Result<Self, GwError> {
        self.apply_eigen_fn(|lam| {
            if lam < -1e-12 {
                return Err(GwError::NotPositiveDefinite);
            }
            Ok(lam.max(0.0).sqrt())
        })
    }

    /// The inverse square root `self^{-1/2}` of a symmetric matrix, with
    /// eigenvalues below `floor` raised to it first.
    ///
    /// # Errors
    ///
    /// As [`eigendecomposition_symmetric`](Self::eigendecomposition_symmetric).
    pub(crate) fn inv_sqrt_clamped(&self, floor: f64) -> Result<Self, GwError> {
        self.apply_eigen_fn(|lam| Ok(1.0 / lam.max(floor).sqrt()))
    }

    /// Computes the inverse of a symmetric PSD matrix via eigendecomposition
    /// with eigenvalue clamping.
    ///
    /// Eigenvalues below `floor` are clamped to `floor` before inversion.
    /// This prevents blow-up when the matrix is near-singular — common for
    /// the dual slack `S` during interior point iterations.
    ///
    /// # Errors
    ///
    /// Returns an error if eigendecomposition fails.
    pub fn inverse_regularized(&self, floor: f64) -> Result<Self, GwError> {
        self.apply_eigen_fn(|lam| {
            let clamped = lam.max(floor);
            Ok(1.0 / clamped)
        })
    }

    fn ensure_same_shape(&self, other: &Self) -> Result<(), GwError> {
        if (self.rows, self.cols) == (other.rows, other.cols) {
            return Ok(());
        }
        Err(GwError::ShapeMismatch {
            expected: (self.rows, self.cols),
            got: (other.rows, other.cols),
        })
    }

    fn ensure_square(&self, operation: &str) -> Result<(), GwError> {
        if self.rows == self.cols {
            return Ok(());
        }
        Err(GwError::InvalidMatrix(format!(
            "{operation} requires a square matrix, got {}x{}",
            self.rows, self.cols
        )))
    }

    fn ensure_finite(&self, operation: &str) -> Result<(), GwError> {
        if self.data.iter().all(|x| x.is_finite()) {
            return Ok(());
        }
        Err(GwError::InvalidMatrix(format!(
            "{operation} of a matrix with a non-finite entry"
        )))
    }
}

/// A symmetric matrix being diagonalized by Jacobi rotations, with the
/// rotations accumulated into `q`.
struct Jacobi {
    a: Vec<f64>,
    q: DenseMatrix,
    n: usize,
}

impl Jacobi {
    fn new(matrix: &DenseMatrix) -> Self {
        Self {
            a: matrix.data.clone(),
            q: DenseMatrix::identity(matrix.rows),
            n: matrix.rows,
        }
    }

    /// Sweeps until the off-diagonal norm is at most `tolerance`.
    fn diagonalize(&mut self, tolerance: f64) -> Result<(), GwError> {
        for sweep in 0..MAX_JACOBI_SWEEPS {
            let off_norm = self.off_diagonal_norm();
            if off_norm <= tolerance {
                return Ok(());
            }
            self.sweep(rotation_threshold(sweep, off_norm, self.n));
        }
        if self.off_diagonal_norm() <= tolerance {
            return Ok(());
        }
        Err(GwError::SolverFailed(format!(
            "Jacobi eigendecomposition did not converge in {MAX_JACOBI_SWEEPS} sweeps"
        )))
    }

    fn off_diagonal_norm(&self) -> f64 {
        let n = self.n;
        let mut norm_sq = 0.0_f64;
        for j in 0..n {
            for i in (j + 1)..n {
                let v = self.a[idx(i, j, n)];
                norm_sq = (2.0 * v).mul_add(v, norm_sq);
            }
        }
        norm_sq.sqrt()
    }

    /// One cyclic-by-row sweep, rotating away every entry above `threshold`.
    fn sweep(&mut self, threshold: f64) {
        let n = self.n;
        for j in 0..n {
            for i in (j + 1)..n {
                let a_ij = self.a[idx(i, j, n)];
                if a_ij.abs() > threshold {
                    let rotation = JacobiRotation::annihilating(
                        self.a[idx(i, i, n)],
                        self.a[idx(j, j, n)],
                        a_ij,
                    );
                    self.rotate(i, j, &rotation, a_ij);
                }
            }
        }
    }

    /// Applies `rotation`, which zeroes `a_ij`, to `a` and accumulates it into `q`.
    fn rotate(&mut self, i: usize, j: usize, rotation: &JacobiRotation, a_ij: f64) {
        let n = self.n;
        let a = &mut self.a;
        a[idx(i, i, n)] = rotation.t.mul_add(-a_ij, a[idx(i, i, n)]);
        a[idx(j, j, n)] = rotation.t.mul_add(a_ij, a[idx(j, j, n)]);
        a[idx(i, j, n)] = 0.0;
        a[idx(j, i, n)] = 0.0;
        for k in (0..n).filter(|&k| k != i && k != j) {
            let (new_ki, new_kj) = rotation.apply(a[idx(k, i, n)], a[idx(k, j, n)]);
            a[idx(k, i, n)] = new_ki;
            a[idx(i, k, n)] = new_ki;
            a[idx(k, j, n)] = new_kj;
            a[idx(j, k, n)] = new_kj;
        }
        let q = &mut self.q.data;
        for k in 0..n {
            let (new_ki, new_kj) = rotation.apply(q[idx(k, i, n)], q[idx(k, j, n)]);
            q[idx(k, i, n)] = new_ki;
            q[idx(k, j, n)] = new_kj;
        }
    }

    /// The diagonal's eigenvalues in ascending order, with `q`'s columns in
    /// the same order.
    fn sorted_eigenpairs(self) -> (Vec<f64>, DenseMatrix) {
        let n = self.n;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| self.a[idx(i, i, n)].total_cmp(&self.a[idx(j, j, n)]));

        let mut eigenvalues = Vec::with_capacity(n);
        let mut q_sorted = DenseMatrix::zeros(n, n);
        for (new_col, &old_col) in order.iter().enumerate() {
            eigenvalues.push(self.a[idx(old_col, old_col, n)]);
            col_mut(&mut q_sorted.data, new_col, n).copy_from_slice(col(&self.q.data, old_col, n));
        }
        (eigenvalues, q_sorted)
    }
}

/// The threshold below which `sweep` leaves an entry unrotated: a fraction
/// of the off-diagonal norm in the opening sweeps, zero after them.
fn rotation_threshold(sweep: usize, off_norm: f64, n: usize) -> f64 {
    if sweep < THRESHOLD_SWEEPS {
        THRESHOLD_FACTOR * off_norm / (n as f64 * n as f64).sqrt()
    } else {
        0.0
    }
}

/// A Givens rotation by `θ`, held as `t = tan θ`, `sin θ` and
/// `tau = sin θ / (1 + cos θ)` for the reduced-multiplication update.
struct JacobiRotation {
    t: f64,
    sin: f64,
    tau: f64,
}

impl JacobiRotation {
    /// The rotation that zeroes `a_ij` between diagonal entries `a_ii` and `a_jj`.
    fn annihilating(a_ii: f64, a_jj: f64, a_ij: f64) -> Self {
        let diff = a_jj - a_ii;
        let t = if diff.abs() < EQUAL_DIAGONAL_GAP {
            if a_ij >= 0.0 { 1.0 } else { -1.0 }
        } else {
            let tau = diff / (2.0 * a_ij);
            // The smaller root of t² + 2τt − 1 = 0, which avoids cancellation.
            if tau >= 0.0 {
                1.0 / (tau + tau.mul_add(tau, 1.0).sqrt())
            } else {
                -1.0 / (-tau + tau.mul_add(tau, 1.0).sqrt())
            }
        };
        let cos = 1.0 / t.mul_add(t, 1.0).sqrt();
        let sin = t * cos;
        Self {
            t,
            sin,
            tau: sin / (1.0 + cos),
        }
    }

    /// The rotated pair `(x_i, x_j)` of entries in columns `i` and `j`.
    fn apply(&self, x_i: f64, x_j: f64) -> (f64, f64) {
        (
            self.sin.mul_add(-self.tau.mul_add(x_i, x_j), x_i),
            self.sin.mul_add(self.tau.mul_add(-x_j, x_i), x_j),
        )
    }
}

/// A symmetric tridiagonal matrix, held as its diagonal and the squares of
/// the entries beside it, which are all its eigenvalues depend on.
struct Tridiagonal {
    diagonal: Vec<f64>,
    /// `squared_offdiagonal[i]` is `T[i + 1][i]²`.
    squared_offdiagonal: Vec<f64>,
    /// The smallest magnitude a pivot of `T − xI` is allowed, so a zero pivot
    /// counts as negative instead of dividing by zero.
    min_pivot: f64,
}

impl Tridiagonal {
    /// Reduces the symmetric `n × n` column-major matrix `a` to a
    /// tridiagonal one with the same eigenvalues, applying a Householder
    /// reflection `H A H` per column.
    fn reduce(mut a: Vec<f64>, n: usize) -> Self {
        let mut reflector = vec![0.0; n];
        let mut update = vec![0.0; n];
        let mut offdiagonal = Vec::with_capacity(n.saturating_sub(1));
        for k in 0..n.saturating_sub(1) {
            let below = &col(&a, k, n)[k + 1..];
            let v = &mut reflector[k + 1..];
            match Reflection::mapping(below, v) {
                Some((alpha, reflection)) => {
                    reflection.apply_to_trailing_block(
                        &mut a[(k + 1) * n..],
                        n,
                        &mut update[k + 1..],
                    );
                    offdiagonal.push(alpha);
                }
                None => offdiagonal.push(below[0]),
            }
        }
        let diagonal = (0..n).map(|i| a[idx(i, i, n)]).collect();
        let squared_offdiagonal: Vec<f64> = offdiagonal.iter().map(|e| e * e).collect();
        let largest = squared_offdiagonal.iter().copied().fold(1.0, f64::max);
        Self {
            diagonal,
            squared_offdiagonal,
            min_pivot: f64::MIN_POSITIVE * largest,
        }
    }

    /// How many eigenvalues lie below `x`: by Sylvester's law of inertia, the
    /// number of negative pivots in the `LDLᵀ` factorization of `T − xI`.
    fn count_below(&self, x: f64) -> usize {
        let mut negatives = 0;
        let mut pivot = 1.0;
        for (i, &d) in self.diagonal.iter().enumerate() {
            let coupling = if i == 0 {
                0.0
            } else {
                self.squared_offdiagonal[i - 1] / pivot
            };
            pivot = d - x - coupling;
            if pivot.abs() < self.min_pivot {
                pivot = -self.min_pivot;
            }
            if pivot < 0.0 {
                negatives += 1;
            }
        }
        negatives
    }

    /// The smallest eigenvalue, bisected to the last bit from a Gershgorin
    /// lower bound and the smallest diagonal entry, which bound it below and
    /// above.
    fn min_eigenvalue(&self) -> f64 {
        let n = self.diagonal.len();
        let radius = |i: usize| {
            let left = if i == 0 {
                0.0
            } else {
                self.squared_offdiagonal[i - 1].sqrt()
            };
            let right = if i + 1 == n {
                0.0
            } else {
                self.squared_offdiagonal[i].sqrt()
            };
            left + right
        };
        let mut lower = (0..n)
            .map(|i| self.diagonal[i] - radius(i))
            .fold(f64::INFINITY, f64::min);
        let mut upper = self.diagonal.iter().copied().fold(f64::INFINITY, f64::min);
        loop {
            let mid = f64::midpoint(lower, upper);
            if mid <= lower || mid >= upper {
                return lower;
            }
            if self.count_below(mid) > 0 {
                upper = mid;
            } else {
                lower = mid;
            }
        }
    }
}

/// `x · 2^exponent`, exact unless the result leaves `f64`'s range, for any
/// exponent a subnormal's normalization needs.
fn scale_by_power_of_two(x: f64, exponent: i32) -> f64 {
    let half = exponent / 2;
    x * 2.0_f64.powi(half) * 2.0_f64.powi(exponent - half)
}

/// A Householder reflection `H = I − beta·v vᵀ` acting on the last
/// `v.len()` coordinates.
struct Reflection<'a> {
    v: &'a [f64],
    beta: f64,
}

impl<'a> Reflection<'a> {
    /// The reflection, its vector written into `v`, that maps `x` onto
    /// `alpha·e₁`, with that `alpha`; `None` when `x` already is a multiple
    /// of `e₁`.
    fn mapping(x: &[f64], v: &'a mut [f64]) -> Option<(f64, Self)> {
        let tail_squared: f64 = x[1..].iter().map(|xi| xi * xi).sum();
        if tail_squared == 0.0 {
            return None;
        }
        let norm = x[0].mul_add(x[0], tail_squared).sqrt();
        let alpha = -norm.copysign(x[0]);
        v.copy_from_slice(x);
        v[0] -= alpha;
        let beta = 2.0 / v[0].mul_add(v[0], tail_squared);
        Some((alpha, Self { v, beta }))
    }

    /// Replaces the trailing block of the `n`-row columns in `block` by
    /// `H A H`: with `p = beta·A v` and `w = p − (beta·pᵀv / 2)·v`, that is
    /// `A − v wᵀ − w vᵀ`. `w` is scratch space of `v`'s length.
    fn apply_to_trailing_block(&self, block: &mut [f64], n: usize, w: &mut [f64]) {
        let first = n - self.v.len();
        w.par_iter_mut()
            .zip(block.par_chunks(n))
            .for_each(|(wj, column)| *wj = self.beta * dot(&column[first..], self.v));
        let correction = 0.5 * self.beta * dot(w, self.v);
        axpy(-correction, self.v, w);
        let w = &*w;
        block
            .par_chunks_mut(n)
            .zip(self.v.par_iter().zip(w.par_iter()))
            .for_each(|(column, (&vj, &wj))| {
                let column = &mut column[first..];
                axpy(-wj, self.v, column);
                axpy(-vj, w, column);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat_approx_eq(a: &DenseMatrix, b: &DenseMatrix, tol: f64) -> bool {
        a.rows == b.rows
            && a.cols == b.cols
            && a.data
                .iter()
                .zip(&b.data)
                .all(|(&x, &y)| approx_eq(x, y, tol))
    }

    #[test]
    fn transpose_roundtrip() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let mt = m.transpose();
        assert_eq!(mt.rows(), 3);
        assert_eq!(mt.cols(), 2);
        let mtt = mt.transpose();
        assert!(mat_approx_eq(&m, &mtt, 1e-15));
    }

    #[test]
    fn mul_identity() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let id = DenseMatrix::identity(2);
        let result = a.mul(&id).unwrap_or_else(|_| panic!("mul failed"));
        assert!(mat_approx_eq(&a, &result, 1e-15));
    }

    #[test]
    fn mul_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // col-major: A = [1,3,2,4], B = [5,7,6,8]
        // A*B = [[19,22],[43,50]] col-major = [19,43,22,50]
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let b = DenseMatrix::from_vec(2, 2, vec![5.0, 7.0, 6.0, 8.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let c = a.mul(&b).unwrap_or_else(|_| panic!("mul failed"));
        assert!(approx_eq(c.get(0, 0), 19.0, 1e-12));
        assert!(approx_eq(c.get(1, 0), 43.0, 1e-12));
        assert!(approx_eq(c.get(0, 1), 22.0, 1e-12));
        assert!(approx_eq(c.get(1, 1), 50.0, 1e-12));
    }

    #[test]
    fn inner_product_self() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        // ||A||_F^2 = 1 + 4 + 9 + 16 = 30
        let ip = a
            .inner_product(&a)
            .unwrap_or_else(|_| panic!("inner_product failed"));
        assert!(approx_eq(ip, 30.0, 1e-12));
    }

    #[test]
    fn cholesky_2x2() {
        // A = [[4,2],[2,3]]
        let a = DenseMatrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let l = a.cholesky().unwrap_or_else(|_| panic!("cholesky failed"));
        // Verify L * L^T = A
        let llt = l
            .mul_transpose(&l)
            .unwrap_or_else(|_| panic!("mul_transpose failed"));
        assert!(mat_approx_eq(&a, &llt, 1e-12));
    }

    #[test]
    fn cholesky_solve_2x2() {
        let a = DenseMatrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let l = a.cholesky().unwrap_or_else(|_| panic!("cholesky failed"));
        let b = vec![1.0, 2.0];
        let x = DenseMatrix::cholesky_solve(&l, &b).unwrap_or_else(|_| panic!("solve failed"));
        // A x = b => x = A^{-1} b = [[-1/8], [3/4]]
        assert!(approx_eq(x[0], -0.125, 1e-12));
        assert!(approx_eq(x[1], 0.75, 1e-12));
    }

    #[test]
    fn cholesky_non_pd() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        assert!(a.cholesky().is_err());
    }

    #[test]
    fn eigenvalues_2x2() {
        // A = [[2,1],[1,2]], eigenvalues = 1, 3
        let a = DenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let eigs = a
            .eigenvalues_symmetric()
            .unwrap_or_else(|_| panic!("eig failed"));
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn eigendecomposition_reconstruct() {
        // A = [[5,2],[2,3]]
        let a = DenseMatrix::from_vec(2, 2, vec![5.0, 2.0, 2.0, 3.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let (eigs, q) = a
            .eigendecomposition_symmetric()
            .unwrap_or_else(|_| panic!("eigendecomp failed"));
        // Reconstruct: A = Q diag(λ) Q^T
        let mut diag = DenseMatrix::zeros(2, 2);
        diag.set(0, 0, eigs[0]);
        diag.set(1, 1, eigs[1]);
        let qd = q.mul(&diag).unwrap_or_else(|_| panic!("mul failed"));
        let reconstructed = qd
            .mul_transpose(&q)
            .unwrap_or_else(|_| panic!("mul_transpose failed"));
        assert!(mat_approx_eq(&a, &reconstructed, 1e-10));
    }

    #[test]
    fn eigenvalues_3x3_symmetric() {
        // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Tridiagonal matrix from 1D Laplacian. Eigenvalues: 2-√2, 2, 2+√2
        let a = DenseMatrix::from_vec(3, 3, vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let eigs = a
            .eigenvalues_symmetric()
            .unwrap_or_else(|_| panic!("eig failed"));
        let sqrt2 = std::f64::consts::SQRT_2;
        assert!(approx_eq(eigs[0], 2.0 - sqrt2, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 2.0 + sqrt2, 1e-10));
    }

    #[test]
    fn sqrt_spd_identity() {
        let id = DenseMatrix::identity(3);
        let sq = id.sqrt_spd().unwrap_or_else(|_| panic!("sqrt_spd failed"));
        assert!(mat_approx_eq(&id, &sq, 1e-12));
    }

    #[test]
    fn sqrt_spd_roundtrip() {
        let a = DenseMatrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let sq = a.sqrt_spd().unwrap_or_else(|_| panic!("sqrt_spd failed"));
        let sq_sq = sq.mul(&sq).unwrap_or_else(|_| panic!("mul failed"));
        assert!(mat_approx_eq(&a, &sq_sq, 1e-10));
    }

    #[test]
    fn mul_transpose_equals_mul_explicit() {
        let a = DenseMatrix::from_vec(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let b = DenseMatrix::from_vec(2, 3, vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let via_transpose = a
            .mul(&b.transpose())
            .unwrap_or_else(|_| panic!("mul failed"));
        let direct = a
            .mul_transpose(&b)
            .unwrap_or_else(|_| panic!("mul_transpose failed"));
        assert!(mat_approx_eq(&via_transpose, &direct, 1e-12));
    }

    #[test]
    fn larger_cholesky() {
        // 5x5 diagonally dominant SPD
        let n = 5;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                data[idx(i, j, n)] = 1.0;
            }
            data[idx(i, i, n)] += n as f64;
        }
        let a = DenseMatrix::from_vec(n, n, data).unwrap_or_else(|_| panic!("from_vec failed"));
        let l = a.cholesky().unwrap_or_else(|_| panic!("cholesky failed"));
        let llt = l
            .mul_transpose(&l)
            .unwrap_or_else(|_| panic!("mul_transpose failed"));
        assert!(mat_approx_eq(&a, &llt, 1e-10));
    }

    #[test]
    fn sub_subtracts_elementwise() {
        let a = DenseMatrix::from_vec(2, 2, vec![6.0, 8.0, 10.0, 12.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let b = DenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let expected = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let diff = a.sub(&b).unwrap_or_else(|_| panic!("sub failed"));
        assert!(mat_approx_eq(&diff, &expected, 1e-15));
    }

    #[test]
    fn scale_correctness() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        let scaled = a.scale(3.0);
        let expected = DenseMatrix::from_vec(2, 2, vec![3.0, 6.0, 9.0, 12.0])
            .unwrap_or_else(|_| panic!("from_vec failed"));
        assert!(mat_approx_eq(&scaled, &expected, 1e-15));
    }
}
