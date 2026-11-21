//! Primal-dual interior point method for semidefinite programming.
//!
//! Implements the algorithm described in [`MATH.md`](./MATH.md) §3:
//!
//! 1. Initialize `X = I`, `S = C + αI`, `y = 0`
//! 2. At each iteration:
//!    - Compute the Nesterov-Todd scaling matrix `W`
//!    - Form and solve the Schur complement system for `Δy`
//!    - Recover search directions `ΔX`, `ΔS`
//!    - Line search to maintain positive definiteness
//!    - Update barrier parameter `μ`
//! 3. Terminate when the relative duality gap and the relative primal and
//!    dual infeasibilities are all < ε, or fail after `max_iterations`
//!
//! The per-iteration cost is `O(m² n² + m³)` where `n` is the matrix dimension
//! and `m` is the number of constraints. See [`MATH.md`](./MATH.md) §5.

use super::linalg::DenseMatrix;
use super::{ConstraintMatrix, SdpProblem, SdpSolution};
use crate::GwError;

/// Centering parameter `σ`: each step targets `μ_next = σ μ`. See [`MATH.md`](./MATH.md) §3.6.
const CENTERING: f64 = 0.2;

/// Configuration for the interior point solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of IPM iterations before giving up.
    pub max_iterations: usize,
    /// Convergence tolerance for the duality gap.
    pub epsilon: f64,
    /// Safety factor for step size (`γ` in the line search, typically 0.95–0.99).
    pub step_safety: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            epsilon: 1e-8,
            step_safety: 0.99,
        }
    }
}

/// The primal-dual interior point SDP solver.
#[derive(Debug)]
pub struct Solver {
    config: SolverConfig,
}

impl Solver {
    /// Creates a new solver with the given configuration.
    pub const fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solves the given SDP and returns the optimal solution.
    ///
    /// This is the main entry point. It runs the primal-dual interior point
    /// method with Nesterov-Todd scaling until the relative duality gap and
    /// the relative primal and dual infeasibilities all drop below
    /// `config.epsilon`.
    ///
    /// See [`MATH.md`](./MATH.md) §3.8 for the complete algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::SolverFailed`] on a numerical failure (e.g., Cholesky
    /// failure during iteration) or when `config.max_iterations` pass without
    /// converging.
    pub fn solve(&self, problem: &SdpProblem) -> Result<SdpSolution, GwError> {
        let n = problem.n;
        let m = problem.constraints.len();
        let (mut x, mut y, mut s) = self.initialize(problem)?;

        let mut iterations = 0;

        while !Self::has_converged(problem, &x, &y, &s, self.config.epsilon)? {
            if iterations == self.config.max_iterations {
                return Err(GwError::SolverFailed(format!(
                    "interior point method did not converge in {iterations} iterations"
                )));
            }
            iterations += 1;

            let mu_target = CENTERING * Self::complementarity(&x, &s, n)?;

            // Compute NT scaling W.
            let w = Self::compute_nt_scaling(&x, &s)?;

            // Compute search direction.
            let (dx, dy, ds) = Self::compute_search_direction(problem, &x, &y, &s, mu_target, &w)?;

            // Line search for step size.
            let alpha = Self::compute_step_size(&x, &dx, &s, &ds, self.config.step_safety)?;

            // Update iterates.
            let scaled_dx = dx.scale(alpha);
            let scaled_ds = ds.scale(alpha);
            x.add_assign(&scaled_dx)
                .map_err(|e| GwError::SolverFailed(format!("X update: {e}")))?;
            s.add_assign(&scaled_ds)
                .map_err(|e| GwError::SolverFailed(format!("S update: {e}")))?;
            for i in 0..m {
                y[i] = alpha.mul_add(dy[i], y[i]);
            }
        }

        let primal_obj = problem
            .c
            .inner_product(&x)
            .map_err(|e| GwError::SolverFailed(format!("primal obj: {e}")))?;
        let dual_obj: f64 = problem
            .constraints
            .iter()
            .zip(&y)
            .map(|(c, &yi)| c.b * yi)
            .sum();
        let duality_gap = Self::complementarity(&x, &s, n)?;

        Ok(SdpSolution {
            x,
            y,
            s,
            primal_obj,
            dual_obj,
            duality_gap,
            iterations,
        })
    }

    /// Computes the starting point `(X, y, S)`.
    ///
    /// `X = I` satisfies the primal constraints and `S = C + αI` is positive
    /// definite, but with `y = 0` the shift leaves the dual infeasible; the
    /// search direction drives that dual residual to zero.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    #[allow(clippy::unused_self)]
    fn initialize(
        &self,
        problem: &SdpProblem,
    ) -> Result<(DenseMatrix, Vec<f64>, DenseMatrix), GwError> {
        let n = problem.n;
        let m = problem.constraints.len();

        let x = DenseMatrix::identity(n);
        let y = vec![0.0; m];

        // S = C - Σ y_i A_i = C (since y = 0).
        // If C is not PD, shift: S = C + α I to ensure S ≻ 0.
        let mut s = problem.c.clone();
        let min_eig = s
            .min_eigenvalue()
            .map_err(|e| GwError::SolverFailed(format!("init eigenvalue: {e}")))?;
        if min_eig <= 0.0 {
            let shift = -min_eig + 1.0;
            let shift_mat = DenseMatrix::identity(n).scale(shift);
            s.add_assign(&shift_mat)
                .map_err(|e| GwError::SolverFailed(format!("init shift: {e}")))?;
        }

        Ok((x, y, s))
    }

    /// Computes the Nesterov-Todd scaling matrix `W` satisfying `W S W = X`.
    ///
    /// Algorithm (see [`MATH.md`](./MATH.md) §3.3):
    /// 1. Compute `X^{1/2}` via eigendecomposition
    /// 2. Form `P = X^{1/2} S X^{1/2}`
    /// 3. Compute `P^{-1/2}`
    /// 4. `W = X^{1/2} P^{-1/2} X^{1/2}`
    ///
    /// Verification: `W S W = X^{1/2} P^{-1/2} X^{1/2} S X^{1/2} P^{-1/2} X^{1/2}`
    ///             = `X^{1/2} P^{-1/2} P P^{-1/2} X^{1/2}` = `X^{1/2} I X^{1/2}` = `X`.
    ///
    /// # Errors
    ///
    /// Returns an error if `X` or `S` is not positive definite.
    fn compute_nt_scaling(x: &DenseMatrix, s: &DenseMatrix) -> Result<DenseMatrix, GwError> {
        let x_half = x
            .sqrt_spd()
            .map_err(|e| GwError::SolverFailed(format!("NT X^1/2: {e}")))?;

        // P = X^{1/2} S X^{1/2} — should be SPD, but floating-point
        // errors in the eigendecomposition can introduce tiny negative
        // eigenvalues. We symmetrize and clamp.
        let mut p = x_half
            .congruence(s)
            .map_err(|e| GwError::SolverFailed(format!("NT P: {e}")))?;
        p.symmetrize();

        // P^{-1/2} with tolerance-aware clamping of eigenvalues.
        let p_inv_half = p
            .apply_eigen_fn(|lam| {
                let clamped = lam.max(1e-14);
                Ok(1.0 / clamped.sqrt())
            })
            .map_err(|e| GwError::SolverFailed(format!("NT P^-1/2: {e}")))?;

        // W = X^{1/2} P^{-1/2} X^{1/2}
        let mut w = x_half
            .congruence(&p_inv_half)
            .map_err(|e| GwError::SolverFailed(format!("NT W: {e}")))?;
        w.symmetrize();

        Ok(w)
    }

    /// Forms the Schur complement matrix `M` and right-hand side `h`.
    ///
    /// From the NT-symmetrized KKT system:
    ///   `ΔX = σμ S⁻¹ − X − W ΔS W`
    ///   `ΔS = R_d − Σ Δyⱼ Aⱼ`
    ///   `tr(Aᵢ ΔX) = rₚᵢ`
    ///
    /// Substituting gives: `M Δy = h` where
    ///   `M_ij = tr(Aᵢ W Aⱼ W)`
    ///   `hᵢ = rₚᵢ − tr(Aᵢ (σμ S⁻¹ − X)) + tr(Aᵢ W R_d W)`
    ///
    /// # Errors
    ///
    /// Returns an error if matrix operations fail.
    fn form_schur_complement(
        problem: &SdpProblem,
        w: &DenseMatrix,
        x: &DenseMatrix,
        y: &[f64],
        s: &DenseMatrix,
        mu: f64,
    ) -> Result<(DenseMatrix, Vec<f64>), GwError> {
        let m = problem.constraints.len();

        let scaled: Vec<ScaledConstraint<'_>> = problem
            .constraints
            .iter()
            .map(|ci| ScaledConstraint::new(&ci.a, w))
            .collect::<Result<_, _>>()?;

        // M_ij = tr(A_i W A_j W)
        let mut schur = DenseMatrix::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let val = scaled[i].schur_entry(&scaled[j], w)?;
                schur.set(i, j, val);
                schur.set(j, i, val);
            }
        }

        // RHS: h_i = r_{p,i} - tr(A_i (σμ S^{-1} - X)) + tr(A_i W R_d W)
        let s_inv = s
            .inverse_regularized(1e-14)
            .map_err(|e| GwError::SolverFailed(format!("Schur S^-1: {e}")))?;
        let centering = s_inv
            .scale(mu)
            .sub(x)
            .map_err(|e| GwError::SolverFailed(format!("Schur centering: {e}")))?;

        let rd = Self::dual_residual(problem, y, s)?;

        let w_rd_w = w
            .congruence(&rd)
            .map_err(|e| GwError::SolverFailed(format!("Schur W*Rd*W: {e}")))?;

        let mut h = vec![0.0; m];
        for (i, ci) in problem.constraints.iter().enumerate() {
            let rp = ci.b
                - ci.a
                    .inner_product(x)
                    .map_err(|e| GwError::SolverFailed(format!("Schur rp: {e}")))?;
            let center_term =
                ci.a.inner_product(&centering)
                    .map_err(|e| GwError::SolverFailed(format!("Schur center: {e}")))?;
            let rd_term =
                ci.a.inner_product(&w_rd_w)
                    .map_err(|e| GwError::SolverFailed(format!("Schur rd_term: {e}")))?;
            h[i] = rp - center_term + rd_term;
        }

        Ok((schur, h))
    }

    /// Computes the full search direction `(ΔX, Δy, ΔS)`.
    ///
    /// 1. Form Schur complement system `M Δy = h`
    /// 2. Solve via Cholesky
    /// 3. Recover `ΔS = R_d − Σ Δyᵢ Aᵢ`
    /// 4. Recover `ΔX = σμ S⁻¹ − X − W ΔS W` (with symmetrization)
    ///
    /// See [`MATH.md`](./MATH.md) §3.4.
    ///
    /// # Errors
    ///
    /// Returns an error if the Schur system is singular or ill-conditioned.
    fn compute_search_direction(
        problem: &SdpProblem,
        x: &DenseMatrix,
        y: &[f64],
        s: &DenseMatrix,
        mu: f64,
        w: &DenseMatrix,
    ) -> Result<(DenseMatrix, Vec<f64>, DenseMatrix), GwError> {
        let m = problem.constraints.len();

        // Form and solve Schur complement system.
        let (schur, h) = Self::form_schur_complement(problem, w, x, y, s, mu)?;

        // Add small regularization for numerical stability.
        let mut schur_reg = schur;
        let reg = 1e-12;
        for i in 0..m {
            schur_reg.set(i, i, schur_reg.get(i, i) + reg);
        }

        let l_schur = schur_reg
            .cholesky()
            .map_err(|e| GwError::SolverFailed(format!("Schur Cholesky: {e}")))?;
        let dy = DenseMatrix::cholesky_solve(&l_schur, &h)
            .map_err(|e| GwError::SolverFailed(format!("Schur solve: {e}")))?;

        // ΔS = R_d − Σ Δy_i A_i
        let mut ds = Self::dual_residual(problem, y, s)?;
        for (ci, &dyi) in problem.constraints.iter().zip(&dy) {
            ci.a.add_scaled_to(&mut ds, -dyi)
                .map_err(|e| GwError::SolverFailed(format!("ds update: {e}")))?;
        }

        // ΔX = σμ S⁻¹ − X − W ΔS W  (note the MINUS sign on W ΔS W)
        let s_inv = s
            .inverse_regularized(1e-14)
            .map_err(|e| GwError::SolverFailed(format!("dx S^-1: {e}")))?;

        let mut dx = s_inv.scale(mu);
        dx.sub_assign(x)
            .map_err(|e| GwError::SolverFailed(format!("dx sub X: {e}")))?;

        let w_ds_w = w
            .congruence(&ds)
            .map_err(|e| GwError::SolverFailed(format!("dx W*dS*W: {e}")))?;
        dx.sub_assign(&w_ds_w)
            .map_err(|e| GwError::SolverFailed(format!("dx sub WdSW: {e}")))?;

        // Symmetrize ΔX to correct floating-point asymmetry.
        dx.symmetrize();

        Ok((dx, dy, ds))
    }

    /// Computes the maximum step size `α` that keeps `X + α ΔX ≻ 0` and `S + α ΔS ≻ 0`.
    ///
    /// Uses minimum eigenvalue of the scaled direction matrices.
    /// Applies the safety factor `γ`. See [`MATH.md`](./MATH.md) §3.5.
    ///
    /// # Errors
    ///
    /// Returns an error if eigenvalue computation fails.
    fn compute_step_size(
        x: &DenseMatrix,
        dx: &DenseMatrix,
        s: &DenseMatrix,
        ds: &DenseMatrix,
        safety: f64,
    ) -> Result<f64, GwError> {
        let alpha_x = max_feasible_step(x, dx)?;
        let alpha_s = max_feasible_step(s, ds)?;
        let alpha = safety * alpha_x.min(alpha_s);
        // Clamp to [0, 1].
        Ok(alpha.clamp(0.0, 1.0))
    }

    /// Computes the complementarity measure `μ = ⟨X, S⟩ / n`.
    ///
    /// See [`MATH.md`](./MATH.md) §3.6.
    fn complementarity(x: &DenseMatrix, s: &DenseMatrix, n: usize) -> Result<f64, GwError> {
        let xs = x
            .inner_product(s)
            .map_err(|e| GwError::SolverFailed(format!("complementarity: {e}")))?;
        Ok(xs / n as f64)
    }

    /// Whether the relative duality gap and the relative primal and dual
    /// infeasibilities are all below `epsilon`.
    ///
    /// See [`MATH.md`](./MATH.md) §3.7.
    fn has_converged(
        problem: &SdpProblem,
        x: &DenseMatrix,
        y: &[f64],
        s: &DenseMatrix,
        epsilon: f64,
    ) -> Result<bool, GwError> {
        let solver_failed = |what: &str, e: GwError| GwError::SolverFailed(format!("{what}: {e}"));
        let primal_obj = problem
            .c
            .inner_product(x)
            .map_err(|e| solver_failed("convergence primal obj", e))?;
        let xs = x
            .inner_product(s)
            .map_err(|e| solver_failed("convergence gap", e))?;
        let gap = xs / (1.0 + primal_obj.abs());

        let mut primal_residual_sq = 0.0_f64;
        let mut b_sq = 0.0_f64;
        for ci in &problem.constraints {
            let ax =
                ci.a.inner_product(x)
                    .map_err(|e| solver_failed("convergence A(X)", e))?;
            primal_residual_sq = (ci.b - ax).mul_add(ci.b - ax, primal_residual_sq);
            b_sq = ci.b.mul_add(ci.b, b_sq);
        }
        let primal_infeasibility = primal_residual_sq.sqrt() / (1.0 + b_sq.sqrt());

        let rd = Self::dual_residual(problem, y, s)?;
        let rd_norm = rd
            .inner_product(&rd)
            .map_err(|e| solver_failed("convergence |Rd|", e))?
            .sqrt();
        let c_norm = problem
            .c
            .inner_product(&problem.c)
            .map_err(|e| solver_failed("convergence |C|", e))?
            .sqrt();
        let dual_infeasibility = rd_norm / (1.0 + c_norm);

        Ok(gap < epsilon && primal_infeasibility < epsilon && dual_infeasibility < epsilon)
    }

    /// Computes the dual residual `R_d = C − Σ yᵢ Aᵢ − S`.
    fn dual_residual(
        problem: &SdpProblem,
        y: &[f64],
        s: &DenseMatrix,
    ) -> Result<DenseMatrix, GwError> {
        let mut rd = problem.c.clone();
        for (ci, &yi) in problem.constraints.iter().zip(y) {
            ci.a.add_scaled_to(&mut rd, -yi)
                .map_err(|e| GwError::SolverFailed(format!("Rd accum: {e}")))?;
        }
        rd.sub_assign(s)
            .map_err(|e| GwError::SolverFailed(format!("Rd sub S: {e}")))?;
        Ok(rd)
    }
}

/// A constraint as the Schur complement needs it under NT scaling `W`.
enum ScaledConstraint<'a> {
    /// `e_k e_kᵀ`, whose `W A W = w_k w_kᵀ` is never formed.
    Diagonal(usize),
    /// A dense `A` with its `W A W`.
    Dense {
        a: &'a DenseMatrix,
        waw: DenseMatrix,
    },
}

impl<'a> ScaledConstraint<'a> {
    fn new(a: &'a ConstraintMatrix, w: &DenseMatrix) -> Result<Self, GwError> {
        match a {
            ConstraintMatrix::DiagonalUnit(k) => Ok(Self::Diagonal(*k)),
            ConstraintMatrix::Dense(a) => {
                let waw = w
                    .congruence(a)
                    .map_err(|e| GwError::SolverFailed(format!("Schur W*A*W: {e}")))?;
                Ok(Self::Dense { a, waw })
            }
        }
    }

    /// `tr(A_self W A_other W)`: `W_pq²` for two diagonal constraints, and
    /// `(W A W)_kk` for a dense one against `e_k e_kᵀ`.
    fn schur_entry(&self, other: &Self, w: &DenseMatrix) -> Result<f64, GwError> {
        match (self, other) {
            (Self::Diagonal(p), Self::Diagonal(q)) => Ok(w.get(*p, *q).powi(2)),
            (Self::Dense { waw, .. }, Self::Diagonal(k))
            | (Self::Diagonal(k), Self::Dense { waw, .. }) => Ok(waw.get(*k, *k)),
            (Self::Dense { a, .. }, Self::Dense { waw, .. }) => a
                .inner_product(waw)
                .map_err(|e| GwError::SolverFailed(format!("Schur ip: {e}"))),
        }
    }
}

/// Computes the maximum `α > 0` such that `A + α ΔA ≻ 0`.
///
/// If all eigenvalues of `A^{-1/2} ΔA A^{-1/2}` are ≥ 0, then `α` can be
/// arbitrarily large (returns a large sentinel). Otherwise `α_max = -1/λ_min`
/// where `λ_min < 0` is the most negative eigenvalue.
///
/// Uses eigendecomposition of A (with clamping) so it works even when A
/// is only PSD rather than strictly PD.
fn max_feasible_step(a: &DenseMatrix, da: &DenseMatrix) -> Result<f64, GwError> {
    // A^{-1/2} via eigendecomposition with clamped eigenvalues.
    let a_inv_half = a
        .apply_eigen_fn(|lam| {
            let clamped = lam.max(1e-14);
            Ok(1.0 / clamped.sqrt())
        })
        .map_err(|e| GwError::SolverFailed(format!("step A^-1/2: {e}")))?;

    let mut scaled = a_inv_half
        .congruence(da)
        .map_err(|e| GwError::SolverFailed(format!("step A^-1/2 dA A^-1/2: {e}")))?;
    scaled.symmetrize();

    let min_eig = scaled
        .min_eigenvalue()
        .map_err(|e| GwError::SolverFailed(format!("step eigenvalue: {e}")))?;

    if min_eig >= 0.0 {
        Ok(1e10) // DA is PSD relative to A, step is essentially unbounded.
    } else {
        Ok(-1.0 / min_eig)
    }
}
