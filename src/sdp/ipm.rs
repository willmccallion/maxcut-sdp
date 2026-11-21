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

/// Eigenvalues below this are raised to it before `S⁻¹`, `P^{-1/2}` or
/// `A^{-1/2}` is formed, so a nearly singular iterate cannot blow them up.
/// See [`MATH.md`](./MATH.md) §6.
const EIGENVALUE_FLOOR: f64 = 1e-14;

/// Added to the Schur complement's diagonal so its Cholesky factorization
/// survives rounding error near the optimum.
const SCHUR_REGULARIZATION: f64 = 1e-12;

/// Centering parameter `σ`: each step targets `μ_next = σ μ`. See [`MATH.md`](./MATH.md) §3.6.
const CENTERING: f64 = 0.2;

/// The IPM's default tolerance on its relative gap and infeasibilities.
pub const DEFAULT_EPSILON: f64 = 1e-8;

/// Configuration for the interior point solver.
#[derive(Debug, Clone)]
pub struct IpmConfig {
    /// Maximum number of IPM iterations before giving up.
    pub max_iterations: usize,
    /// Tolerance on the relative duality gap and on the relative primal and
    /// dual infeasibilities.
    pub epsilon: f64,
    /// Safety factor for step size (`γ` in the line search, typically 0.95–0.99).
    pub step_safety: f64,
}

impl Default for IpmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            epsilon: DEFAULT_EPSILON,
            step_safety: 0.99,
        }
    }
}

/// The primal-dual interior point SDP solver.
#[derive(Debug)]
pub struct IpmSolver {
    config: IpmConfig,
}

impl IpmSolver {
    /// Creates a new solver with the given configuration.
    pub const fn new(config: IpmConfig) -> Self {
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
    /// Returns [`GwError::SolverStep`], naming the step and keeping its cause,
    /// on a numerical failure such as a Schur complement that is not positive
    /// definite, and [`GwError::SolverFailed`] when `config.max_iterations`
    /// pass without converging.
    pub fn solve(&self, problem: &SdpProblem) -> Result<SdpSolution, GwError> {
        let mut iterate = Iterate::initial(problem)?;
        for iterations in 0.. {
            let residuals = Residuals::of(problem, &iterate)?;
            if residuals.have_converged(problem, &iterate.x, self.config.epsilon)? {
                return Ok(iterate.into_solution(problem, &residuals, iterations));
            }
            if iterations == self.config.max_iterations {
                break;
            }
            let terms = IterationTerms::new(problem, &iterate, residuals)?;
            let direction = search_direction(problem, &iterate, &terms)?;
            let alpha = step_size(&iterate, &direction, self.config.step_safety)?;
            iterate.advance(&direction, alpha)?;
        }
        Err(GwError::SolverFailed(format!(
            "interior point method did not converge in {} iterations",
            self.config.max_iterations
        )))
    }
}

/// The primal-dual point `(X, y, S)` the method moves.
struct Iterate {
    x: DenseMatrix,
    y: Vec<f64>,
    s: DenseMatrix,
}

/// A step `(ΔX, Δy, ΔS)` from an iterate.
struct Direction {
    dx: DenseMatrix,
    dy: Vec<f64>,
    ds: DenseMatrix,
}

impl Iterate {
    /// The starting point: `X = I` satisfies the primal constraints, and
    /// `S = C + αI` is positive definite, though with `y = 0` the shift leaves
    /// the dual infeasible until the steps drive its residual to zero.
    fn initial(problem: &SdpProblem) -> Result<Self, GwError> {
        let mut s = problem.cost().clone();
        let min_eig = s.min_eigenvalue().map_err(at("init eigenvalue"))?;
        if min_eig <= 0.0 {
            s.add_to_diagonal(-min_eig + 1.0);
        }
        Ok(Self {
            x: DenseMatrix::identity(problem.dimension()),
            y: vec![0.0; problem.constraints().len()],
            s,
        })
    }

    /// Moves `alpha` of the way along `direction`.
    fn advance(&mut self, direction: &Direction, alpha: f64) -> Result<(), GwError> {
        self.x
            .add_assign(&direction.dx.scale(alpha))
            .map_err(at("X update"))?;
        self.s
            .add_assign(&direction.ds.scale(alpha))
            .map_err(at("S update"))?;
        for (yi, &dyi) in self.y.iter_mut().zip(&direction.dy) {
            *yi = alpha.mul_add(dyi, *yi);
        }
        Ok(())
    }

    fn into_solution(
        self,
        problem: &SdpProblem,
        residuals: &Residuals,
        iterations: usize,
    ) -> SdpSolution {
        let dual_obj = problem
            .constraints()
            .iter()
            .zip(&self.y)
            .map(|(c, &yi)| c.rhs() * yi)
            .sum();
        SdpSolution {
            x: self.x,
            y: self.y,
            primal_obj: residuals.primal_obj,
            dual_obj,
            duality_gap: residuals.complementarity / problem.dimension() as f64,
            iterations,
        }
    }
}

/// What an iterate leaves to do: the quantities the convergence test and the
/// next step share.
struct Residuals {
    /// `tr(C X)`.
    primal_obj: f64,
    /// `⟨X, S⟩`.
    complementarity: f64,
    /// The dual residual `R_d = C − Σ yᵢ Aᵢ − S`.
    rd: DenseMatrix,
}

impl Residuals {
    fn of(problem: &SdpProblem, iterate: &Iterate) -> Result<Self, GwError> {
        Ok(Self {
            primal_obj: problem
                .cost()
                .inner_product(&iterate.x)
                .map_err(at("primal objective"))?,
            complementarity: iterate
                .x
                .inner_product(&iterate.s)
                .map_err(at("complementarity"))?,
            rd: dual_residual(problem, &iterate.y, &iterate.s)?,
        })
    }

    /// Whether the relative duality gap and the relative primal and dual
    /// infeasibilities are all below `epsilon`. See [`MATH.md`](./MATH.md) §3.7.
    fn have_converged(
        &self,
        problem: &SdpProblem,
        x: &DenseMatrix,
        epsilon: f64,
    ) -> Result<bool, GwError> {
        Ok(self.relative_gap() < epsilon
            && primal_infeasibility(problem, x)? < epsilon
            && self.dual_infeasibility(problem)? < epsilon)
    }

    fn relative_gap(&self) -> f64 {
        self.complementarity / (1.0 + self.primal_obj.abs())
    }

    /// `‖R_d‖ / (1 + ‖C‖)`, in Frobenius norms.
    fn dual_infeasibility(&self, problem: &SdpProblem) -> Result<f64, GwError> {
        let rd_norm = self.rd.inner_product(&self.rd).map_err(at("|Rd|"))?.sqrt();
        let c_norm = problem
            .cost()
            .inner_product(problem.cost())
            .map_err(at("|C|"))?
            .sqrt();
        Ok(rd_norm / (1.0 + c_norm))
    }
}

/// `‖b − A(X)‖ / (1 + ‖b‖)`.
fn primal_infeasibility(problem: &SdpProblem, x: &DenseMatrix) -> Result<f64, GwError> {
    let mut residual_sq = 0.0_f64;
    let mut b_sq = 0.0_f64;
    for ci in problem.constraints() {
        let ax = ci.matrix().inner_product(x).map_err(at("A(X)"))?;
        residual_sq = (ci.rhs() - ax).mul_add(ci.rhs() - ax, residual_sq);
        b_sq = ci.rhs().mul_add(ci.rhs(), b_sq);
    }
    Ok(residual_sq.sqrt() / (1.0 + b_sq.sqrt()))
}

/// Computes the dual residual `R_d = C − Σ yᵢ Aᵢ − S`.
fn dual_residual(problem: &SdpProblem, y: &[f64], s: &DenseMatrix) -> Result<DenseMatrix, GwError> {
    let mut rd = problem.cost().clone();
    for (ci, &yi) in problem.constraints().iter().zip(y) {
        ci.matrix()
            .add_scaled_to(&mut rd, -yi)
            .map_err(at("Rd accum"))?;
    }
    rd.sub_assign(s).map_err(at("Rd sub S"))?;
    Ok(rd)
}

/// What one iteration computes once and its steps share.
struct IterationTerms {
    /// The target `σμ`, for `μ = ⟨X, S⟩ / n`. See [`MATH.md`](./MATH.md) §3.6.
    mu: f64,
    /// The Nesterov-Todd scaling `W`.
    w: DenseMatrix,
    /// `S⁻¹`, its eigenvalues floored.
    s_inv: DenseMatrix,
    /// The dual residual `R_d`.
    rd: DenseMatrix,
}

impl IterationTerms {
    fn new(problem: &SdpProblem, iterate: &Iterate, residuals: Residuals) -> Result<Self, GwError> {
        Ok(Self {
            mu: CENTERING * (residuals.complementarity / problem.dimension() as f64),
            w: nt_scaling(&iterate.x, &iterate.s)?,
            s_inv: iterate
                .s
                .inverse_regularized(EIGENVALUE_FLOOR)
                .map_err(at("S^-1"))?,
            rd: residuals.rd,
        })
    }
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
fn nt_scaling(x: &DenseMatrix, s: &DenseMatrix) -> Result<DenseMatrix, GwError> {
    let x_half = x.sqrt_spd().map_err(at("NT X^1/2"))?;

    // Rounding error in X^{1/2} leaves P slightly asymmetric.
    let mut p = x_half.congruence(s).map_err(at("NT P"))?;
    p.symmetrize();
    let p_inv_half = p
        .inv_sqrt_clamped(EIGENVALUE_FLOOR)
        .map_err(at("NT P^-1/2"))?;

    let mut w = x_half.congruence(&p_inv_half).map_err(at("NT W"))?;
    w.symmetrize();
    Ok(w)
}

/// Computes the search direction `(ΔX, Δy, ΔS)`.
///
/// 1. Solve the Schur complement system `M Δy = h` by Cholesky
/// 2. Recover `ΔS = R_d − Σ Δyᵢ Aᵢ`
/// 3. Recover `ΔX = σμ S⁻¹ − X − W ΔS W`, symmetrized
///
/// See [`MATH.md`](./MATH.md) §3.4.
fn search_direction(
    problem: &SdpProblem,
    iterate: &Iterate,
    terms: &IterationTerms,
) -> Result<Direction, GwError> {
    let mut schur = schur_matrix(problem, &terms.w)?;
    schur.add_to_diagonal(SCHUR_REGULARIZATION);
    let l_schur = schur.cholesky().map_err(at("Schur Cholesky"))?;
    let h = schur_rhs(problem, &iterate.x, terms)?;
    let dy = DenseMatrix::cholesky_solve(&l_schur, &h).map_err(at("Schur solve"))?;

    let mut ds = terms.rd.clone();
    for (ci, &dyi) in problem.constraints().iter().zip(&dy) {
        ci.matrix().add_scaled_to(&mut ds, -dyi).map_err(at("dS"))?;
    }

    let mut dx = terms.s_inv.scale(terms.mu);
    dx.sub_assign(&iterate.x).map_err(at("dX sub X"))?;
    let w_ds_w = terms.w.congruence(&ds).map_err(at("dX W*dS*W"))?;
    dx.sub_assign(&w_ds_w).map_err(at("dX sub WdSW"))?;
    dx.symmetrize();

    Ok(Direction { dx, dy, ds })
}

/// The Schur complement `M_ij = tr(Aᵢ W Aⱼ W)` of the NT-scaled KKT system
/// `ΔX = σμ S⁻¹ − X − W ΔS W`, `ΔS = R_d − Σ Δyⱼ Aⱼ`, `tr(Aᵢ ΔX) = rₚᵢ`.
fn schur_matrix(problem: &SdpProblem, w: &DenseMatrix) -> Result<DenseMatrix, GwError> {
    let scaled: Vec<ScaledConstraint<'_>> = problem
        .constraints()
        .iter()
        .map(|ci| ScaledConstraint::new(ci.matrix(), w))
        .collect::<Result<_, _>>()?;
    let m = scaled.len();
    let mut schur = DenseMatrix::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            let entry = scaled[i].schur_entry(&scaled[j], w)?;
            schur.set(i, j, entry);
            schur.set(j, i, entry);
        }
    }
    Ok(schur)
}

/// The Schur system's right-hand side
/// `hᵢ = rₚᵢ − tr(Aᵢ (σμ S⁻¹ − X)) + tr(Aᵢ W R_d W)`.
fn schur_rhs(
    problem: &SdpProblem,
    x: &DenseMatrix,
    terms: &IterationTerms,
) -> Result<Vec<f64>, GwError> {
    let centering = terms
        .s_inv
        .scale(terms.mu)
        .sub(x)
        .map_err(at("Schur centering"))?;
    let w_rd_w = terms.w.congruence(&terms.rd).map_err(at("Schur W*Rd*W"))?;
    problem
        .constraints()
        .iter()
        .map(|ci| {
            let a = ci.matrix();
            let rp = ci.rhs() - a.inner_product(x).map_err(at("Schur rp"))?;
            let center_term = a.inner_product(&centering).map_err(at("Schur center"))?;
            let rd_term = a.inner_product(&w_rd_w).map_err(at("Schur rd_term"))?;
            Ok(rp - center_term + rd_term)
        })
        .collect()
}

/// The step `α ∈ [0, 1]` that keeps `X + α ΔX ≻ 0` and `S + α ΔS ≻ 0`,
/// shortened by the safety factor `γ`. See [`MATH.md`](./MATH.md) §3.5.
fn step_size(iterate: &Iterate, direction: &Direction, safety: f64) -> Result<f64, GwError> {
    let alpha_x = max_feasible_step(&iterate.x, &direction.dx)?;
    let alpha_s = max_feasible_step(&iterate.s, &direction.ds)?;
    Ok((safety * alpha_x.min(alpha_s)).clamp(0.0, 1.0))
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
                let waw = w.congruence(a).map_err(at("Schur W*A*W"))?;
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
            (Self::Dense { a, .. }, Self::Dense { waw, .. }) => {
                a.inner_product(waw).map_err(at("Schur ip"))
            }
        }
    }
}

/// The largest `α` with `A + α ΔA ≻ 0`: `−1/λ_min` for the smallest
/// eigenvalue of `A^{-1/2} ΔA A^{-1/2}`, or infinity when that is not
/// negative. `A`'s eigenvalues are floored, so a merely semidefinite `A`
/// still gives a step.
fn max_feasible_step(a: &DenseMatrix, da: &DenseMatrix) -> Result<f64, GwError> {
    let a_inv_half = a
        .inv_sqrt_clamped(EIGENVALUE_FLOOR)
        .map_err(at("step A^-1/2"))?;
    let mut scaled = a_inv_half
        .congruence(da)
        .map_err(at("step A^-1/2 dA A^-1/2"))?;
    scaled.symmetrize();

    let min_eig = scaled.min_eigenvalue().map_err(at("step eigenvalue"))?;
    Ok(if min_eig >= 0.0 {
        f64::INFINITY
    } else {
        -1.0 / min_eig
    })
}

/// Wraps an error that the IPM step `stage` ran into.
fn at(stage: &'static str) -> impl FnOnce(GwError) -> GwError {
    move |cause| GwError::SolverStep {
        stage,
        cause: Box::new(cause),
    }
}
