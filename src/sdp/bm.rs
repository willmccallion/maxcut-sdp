//! Burer-Monteiro low-rank Max-Cut solver.
//!
//! Replaces the dense interior point method, `O(n³)` per iteration, with nonlinear
//! conjugate gradient (Polak-Ribière+) on a low-rank factorization `X = VVᵀ`. Each column of V (stored
//! in column-major r×n layout) is constrained to the unit sphere S^{r−1}, making the
//! feasible set a product of spheres.
//!
//! Per-iteration cost is O(|E|·r) by iterating edges directly — the Laplacian is
//! never formed. The default rank r = ⌈√(2n)⌉ has r(r+1)/2 > n, so the SDP has an
//! optimum of that rank (Barvinok-Pataki), and for almost every cost matrix every
//! second-order critical point is optimal (Boumal et al. 2016). The solver stops at an
//! approximate first-order point, though, so its objective is no upper bound; the
//! pipeline certifies one ([`MATH.md`](./MATH.md) §7.10).
//!
//! Hot-path optimizations:
//! - Objective uses dot-product identity: ‖vi−vj‖² = 2(1−⟨vi,vj⟩), reusing the
//!   4-way unrolled `dot()` kernel from `linalg.rs` for full vectorization.
//! - Gradient uses `split_at_mut` to prove non-aliasing for simultaneous column updates.
//! - Gradient, tangent projection, and norm are fused into a single O(n·r) pass.
//! - Edge column offsets are pre-computed and sorted for cache locality.
//!
//! See [`MATH.md`](./MATH.md) §7 for the full derivation.

use crate::GwError;
use crate::graph::Graph;
use crate::random::box_muller;
use crate::sdp::linalg::{DenseMatrix, dot};
use rand::Rng;
use rayon::prelude::*;

/// Configuration for the Burer-Monteiro solver.
#[derive(Debug, Clone)]
pub struct BmSolverConfig {
    /// Rank of the factorization, within `2..=n`. `None` means `⌈√(2n)⌉`,
    /// clamped to that range.
    pub rank: Option<usize>,
    /// Maximum gradient descent iterations per restart.
    pub max_iterations: usize,
    /// Stop when ‖Riemannian gradient‖_F < this value.
    pub gradient_tolerance: f64,
    /// Stop when relative objective improvement over a window drops below this.
    pub objective_tolerance: f64,
    /// Number of consecutive stalled objective windows before early stopping.
    pub max_stall_windows: usize,
    /// Initial Armijo step size.
    pub initial_step_size: f64,
    /// Armijo sufficient-decrease parameter c₁.
    pub armijo_c1: f64,
    /// Backtracking contraction factor β ∈ (0, 1).
    pub backtracking_beta: f64,
}

impl Default for BmSolverConfig {
    fn default() -> Self {
        Self {
            rank: None,
            max_iterations: 10_000,
            gradient_tolerance: 1e-8,
            objective_tolerance: 1e-6,
            max_stall_windows: 3,
            initial_step_size: 1.0,
            armijo_c1: 1e-4,
            backtracking_beta: 0.5,
        }
    }
}

/// The solution returned by the BM solver.
#[derive(Debug, Clone)]
pub struct BmSolution {
    /// Unit vectors for each vertex, stored as an r × n matrix
    /// (column i = vertex i's unit vector in R^r).
    pub vectors: DenseMatrix,
    /// The BM objective value (negative for Max-Cut).
    pub objective: f64,
    /// Gradient descent iterations taken.
    pub iterations: usize,
}

/// Burer-Monteiro solver for Max-Cut.
#[derive(Debug)]
pub struct BmSolver {
    config: BmSolverConfig,
}

impl BmSolver {
    /// Creates a new BM solver with the given configuration.
    pub const fn new(config: BmSolverConfig) -> Self {
        Self { config }
    }

    /// Solves the Max-Cut SDP relaxation via Burer-Monteiro with nonlinear CG,
    /// from one random start drawn from `rng`, reporting progress to
    /// `on_progress` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`GwError::InvalidGraph`] if the graph is empty, and
    /// [`GwError::InvalidConfig`] if the configured rank is outside `2..=n`.
    pub fn solve_maxcut(
        &self,
        graph: &Graph,
        rng: &mut impl Rng,
        mut on_progress: impl FnMut(BmProgress),
    ) -> Result<BmSolution, GwError> {
        let n = graph.num_vertices();
        if n == 0 {
            return Err(GwError::InvalidGraph("empty graph".into()));
        }
        let r = self.factor_rank(n)?;
        let edges = column_edges(graph, r);
        let start = random_unit_vectors(n, r, rng);
        Ok(self.descend(&edges, start, &mut on_progress))
    }

    /// The configured rank, or by default `⌈√(2n)⌉` clamped to `[2, n]`.
    fn factor_rank(&self, n: usize) -> Result<usize, GwError> {
        let Some(rank) = self.config.rank else {
            return Ok((((2.0 * n as f64).sqrt()).ceil() as usize).max(2).min(n));
        };
        if !(2..=n).contains(&rank) {
            return Err(GwError::InvalidConfig(format!(
                "BM rank {rank} is outside 2..={n} for a {n}-vertex graph"
            )));
        }
        Ok(rank)
    }

    /// Runs conjugate gradient from `start` until the gradient vanishes, the
    /// objective stalls, the line search fails or the iterations run out.
    fn descend(
        &self,
        edges: &[ColumnEdge],
        start: DenseMatrix,
        on_progress: &mut impl FnMut(BmProgress),
    ) -> BmSolution {
        let n = start.cols();
        let mut descent = Descent::new(&self.config, edges, start);
        let mut stall = StallDetector::new(n, descent.objective, &self.config);
        let mut gradient_norm = 0.0;
        let mut iterations = 0;
        for iteration in 0..self.config.max_iterations {
            gradient_norm = descent.compute_gradient();
            iterations = iteration + 1;
            if gradient_norm < self.config.gradient_tolerance
                || stall.has_stalled(iteration, descent.objective)
            {
                break;
            }
            if iteration > 0 && iteration % PROGRESS_INTERVAL == 0 {
                on_progress(BmProgress::Iteration {
                    iteration,
                    relaxation: -descent.objective,
                    gradient_norm,
                });
            }
            let dir_deriv = descent.set_search_direction(iteration, gradient_norm);
            if !descent.armijo_step(dir_deriv) {
                break;
            }
        }
        on_progress(BmProgress::Finished {
            iterations,
            relaxation: -descent.objective,
            gradient_norm,
        });
        BmSolution {
            vectors: descent.v,
            objective: descent.objective,
            iterations,
        }
    }
}

/// Progress a BM solve reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BmProgress {
    /// Every [`PROGRESS_INTERVAL`] iterations after the first.
    Iteration {
        /// Iterations so far.
        iteration: usize,
        /// The relaxation value `−f` reached.
        relaxation: f64,
        /// The Riemannian gradient's Frobenius norm.
        gradient_norm: f64,
    },
    /// When the solve stops.
    Finished {
        /// Iterations taken.
        iterations: usize,
        /// The relaxation value `−f` reached.
        relaxation: f64,
        /// The last Riemannian gradient's Frobenius norm.
        gradient_norm: f64,
    },
}

/// Iterations between [`BmProgress::Iteration`] reports.
pub const PROGRESS_INTERVAL: usize = 500;

/// Fewest iterations between objective-stall checks; large graphs check every
/// `n / 10`.
const MIN_STALL_CHECK_INTERVAL: usize = 100;

/// The Armijo search gives up once the step shrinks below this.
const MIN_STEP: f64 = 1e-20;

/// A previous gradient with a squared norm below this restarts CG as
/// steepest descent instead of dividing by it.
const MIN_PREVIOUS_GRADIENT_NORM_SQ: f64 = 1e-30;

/// The state of one conjugate-gradient descent and its work buffers.
struct Descent<'a> {
    config: &'a BmSolverConfig,
    edges: &'a [ColumnEdge],
    v: DenseMatrix,
    v_trial: DenseMatrix,
    riem_grad: DenseMatrix,
    riem_grad_prev: DenseMatrix,
    search_dir: DenseMatrix,
    /// The objective `f` at `v`.
    objective: f64,
    /// The last accepted Armijo step.
    alpha: f64,
    prev_grad_norm_sq: f64,
}

impl<'a> Descent<'a> {
    fn new(config: &'a BmSolverConfig, edges: &'a [ColumnEdge], v: DenseMatrix) -> Self {
        let (r, n) = (v.rows(), v.cols());
        Self {
            config,
            edges,
            objective: compute_objective_sparse(edges, v.data(), r),
            v,
            v_trial: DenseMatrix::zeros(r, n),
            riem_grad: DenseMatrix::zeros(r, n),
            riem_grad_prev: DenseMatrix::zeros(r, n),
            search_dir: DenseMatrix::zeros(r, n),
            alpha: config.initial_step_size,
            prev_grad_norm_sq: 0.0,
        }
    }

    /// Computes the Riemannian gradient at `v` and returns its norm.
    fn compute_gradient(&mut self) -> f64 {
        compute_riemannian_gradient(self.edges, &self.v, &mut self.riem_grad)
    }

    /// Sets the Polak-Ribière+ direction, or steepest descent where that is
    /// not a descent direction, and returns its directional derivative.
    fn set_search_direction(&mut self, iteration: usize, gradient_norm: f64) -> f64 {
        let beta = if iteration == 0 {
            0.0
        } else {
            compute_pr_beta(
                &self.riem_grad,
                &self.riem_grad_prev,
                self.prev_grad_norm_sq,
                self.v.cols(),
                iteration,
            )
        };
        if beta == 0.0 {
            set_steepest_descent(&mut self.search_dir, &self.riem_grad);
        } else {
            // Transport the previous direction to the tangent space at v.
            project_to_tangent_inplace(&self.v, &mut self.search_dir);
            let gdata = self.riem_grad.data();
            for (d, &g) in self.search_dir.data_mut().iter_mut().zip(gdata) {
                *d = beta.mul_add(*d, -g);
            }
        }
        let mut dir_deriv = dot(self.riem_grad.data(), self.search_dir.data());
        if dir_deriv >= 0.0 {
            set_steepest_descent(&mut self.search_dir, &self.riem_grad);
            dir_deriv = dot(self.riem_grad.data(), self.search_dir.data());
        }
        self.riem_grad_prev
            .data_mut()
            .copy_from_slice(self.riem_grad.data());
        self.prev_grad_norm_sq = gradient_norm * gradient_norm;
        dir_deriv
    }

    /// Backtracks along the search direction until the Armijo condition
    /// holds and moves there; `false` when the step shrinks away first.
    fn armijo_step(&mut self, dir_deriv: f64) -> bool {
        let config = self.config;
        let r = self.v.rows();
        self.alpha = (self.alpha / config.backtracking_beta).min(config.initial_step_size);
        let sufficient_decrease = config.armijo_c1 * dir_deriv;
        loop {
            step_and_retract(&self.v, &self.search_dir, self.alpha, &mut self.v_trial);
            let f_trial = compute_objective_sparse(self.edges, self.v_trial.data(), r);
            if f_trial <= self.alpha.mul_add(sufficient_decrease, self.objective) {
                std::mem::swap(&mut self.v, &mut self.v_trial);
                self.objective = f_trial;
                return true;
            }
            self.alpha *= config.backtracking_beta;
            if self.alpha < MIN_STEP {
                return false;
            }
        }
    }
}

/// Stops a descent once the objective's relative improvement over a window
/// of iterations has stayed below tolerance for `max_stall_windows` windows.
struct StallDetector {
    interval: usize,
    checkpoint: f64,
    stalls: usize,
    tolerance: f64,
    max_stalls: usize,
}

impl StallDetector {
    fn new(n: usize, objective: f64, config: &BmSolverConfig) -> Self {
        Self {
            interval: MIN_STALL_CHECK_INTERVAL.max(n / 10),
            checkpoint: objective,
            stalls: 0,
            tolerance: config.objective_tolerance,
            max_stalls: config.max_stall_windows,
        }
    }

    /// Whether `objective` at `iteration` ends the descent.
    fn has_stalled(&mut self, iteration: usize, objective: f64) -> bool {
        if iteration == 0 || !iteration.is_multiple_of(self.interval) {
            return false;
        }
        let improvement = (self.checkpoint - objective) / objective.abs().max(1.0);
        self.checkpoint = objective;
        if improvement < self.tolerance {
            self.stalls += 1;
            self.stalls >= self.max_stalls
        } else {
            self.stalls = 0;
            false
        }
    }
}

/// `d ← −g`.
fn set_steepest_descent(direction: &mut DenseMatrix, gradient: &DenseMatrix) {
    for (d, &g) in direction.data_mut().iter_mut().zip(gradient.data()) {
        *d = -g;
    }
}

/// Projects `h` onto the tangent space of the unit sphere at `v`:
/// `h ← h − ⟨v, h⟩ v`.
fn project_to_tangent(v: &[f64], h: &mut [f64]) {
    let d = dot(v, h);
    for (h_k, &v_k) in h.iter_mut().zip(v) {
        *h_k = d.mul_add(-v_k, *h_k);
    }
}

/// An edge `(i, j)` addressed by where its endpoints' columns start in a
/// column-major `r × n` factor: `i_offset = i·r`, `j_offset = j·r`, with
/// `i < j`.
#[derive(Debug, Clone, Copy)]
struct ColumnEdge {
    i_offset: usize,
    j_offset: usize,
    weight: f64,
}

/// The graph's edges as column offsets into a rank-`r` factor, sorted by
/// `i_offset` for cache locality.
fn column_edges(graph: &Graph, r: usize) -> Vec<ColumnEdge> {
    let mut edges: Vec<ColumnEdge> = graph
        .edges()
        .iter()
        .map(|&(i, j, weight)| ColumnEdge {
            i_offset: i * r,
            j_offset: j * r,
            weight,
        })
        .collect();
    edges.sort_unstable_by_key(|edge| edge.i_offset);
    edges
}

/// Computes the BM objective via the dot-product identity — O(|E|·r).
///
/// For unit vectors, `‖vi − vj‖² = 2(1 − ⟨vi, vj⟩)`, so
/// `f = Σ_{(i,j)} −(1/4) w ‖vi−vj‖² = Σ −(1/2) w (1 − ⟨vi,vj⟩)`.
///
/// Pre-slicing to `vi`/`vj` eliminates bounds checks. The `dot()` kernel
/// uses 4-way unrolling + FMA for full vectorization.
fn compute_objective_sparse(edges: &[ColumnEdge], vdata: &[f64], r: usize) -> f64 {
    let partial_sums: Vec<f64> = edges
        .par_chunks(SUM_CHUNK)
        .map(|chunk| {
            chunk
                .iter()
                .map(|edge| {
                    let vi = &vdata[edge.i_offset..edge.i_offset + r];
                    let vj = &vdata[edge.j_offset..edge.j_offset + r];
                    let half_w = 0.5 * edge.weight;
                    half_w.mul_add(dot(vi, vj), -half_w)
                })
                .sum::<f64>()
        })
        .collect();
    partial_sums.iter().sum()
}

/// Terms per partial sum of a parallel reduction. The chunks are fixed and
/// their sums added in order, so the result's rounding, and with it a seeded
/// run, does not depend on how rayon schedules the work.
const SUM_CHUNK: usize = 1024;

/// Fused Riemannian gradient computation: Euclidean gradient + tangent projection + norm.
///
/// Returns `‖riem_grad‖_F`. Eliminates the separate `euc_grad` buffer and saves
/// two full O(n·r) memory passes compared to separate gradient/projection/norm calls.
fn compute_riemannian_gradient(
    edges: &[ColumnEdge],
    v: &DenseMatrix,
    riem_grad: &mut DenseMatrix,
) -> f64 {
    let r = v.rows();
    let vdata = v.data();
    let gdata = riem_grad.data_mut();

    // 1. Zero buffer.
    for x in gdata.iter_mut() {
        *x = 0.0;
    }

    // 2. Edge-iteration gradient accumulation (sequential — edges share vertices).
    for &ColumnEdge {
        i_offset: ci,
        j_offset: cj,
        weight,
    } in edges
    {
        let half_w = 0.5 * weight;
        let vi = &vdata[ci..ci + r];
        let vj = &vdata[cj..cj + r];

        let (left, right) = gdata.split_at_mut(cj);
        let gi = &mut left[ci..ci + r];
        let gj = &mut right[..r];

        for k in 0..r {
            let diff = vi[k] - vj[k];
            let hw = half_w * diff;
            gi[k] -= hw;
            gj[k] += hw;
        }
    }

    // 3. Per-column tangent projection + norm² (parallel — columns are independent).
    let partial_sums: Vec<f64> = gdata
        .par_chunks_mut(r * SUM_CHUNK)
        .zip(vdata.par_chunks(r * SUM_CHUNK))
        .map(|(g_cols, v_cols)| {
            let mut norm_sq = 0.0_f64;
            for (g_col, v_col) in g_cols.chunks_mut(r).zip(v_cols.chunks(r)) {
                project_to_tangent(v_col, g_col);
                norm_sq += dot(g_col, g_col);
            }
            norm_sq
        })
        .collect();
    partial_sums.iter().sum::<f64>().sqrt()
}

/// Projects each column of `h` onto the tangent space at the corresponding column of `v`.
///
/// For each vertex i: `h_i ← h_i − ⟨h_i, v_i⟩ v_i`.
fn project_to_tangent_inplace(v: &DenseMatrix, h: &mut DenseMatrix) {
    let r = v.rows();
    let vdata = v.data();
    let hdata = h.data_mut();

    hdata
        .par_chunks_mut(r)
        .zip(vdata.par_chunks(r))
        .for_each(|(h_col, v_col)| project_to_tangent(v_col, h_col));
}

/// Computes the Polak-Ribière+ beta with periodic restarts.
///
/// `β = max(0, ⟨g_new, g_new − T(g_old)⟩ / ‖g_old‖²)`, and `β = 0` every
/// `n` iterations; the clamp at 0 restarts as steepest descent otherwise.
fn compute_pr_beta(
    grad_new: &DenseMatrix,
    grad_old: &DenseMatrix,
    prev_grad_norm_sq: f64,
    n: usize,
    iter: usize,
) -> f64 {
    // Periodic restart.
    if iter.is_multiple_of(n) {
        return 0.0;
    }

    if prev_grad_norm_sq < MIN_PREVIOUS_GRADIENT_NORM_SQ {
        return 0.0;
    }

    // Since g_new is tangent at v (⟨gn_i, v_i⟩ = 0), parallel transport of
    // g_old doesn't affect inner products with g_new:
    //   ⟨gn, T(go)⟩ = ⟨gn, go − ⟨go,v⟩v⟩ = ⟨gn, go⟩
    // So the numerator simplifies to ‖gn‖² − ⟨gn, go⟩.
    let gn = grad_new.data();
    let go = grad_old.data();

    let new_norm_sq = dot(gn, gn);
    let gn_go = dot(gn, go);

    // PR+ formula: β = max(0, (‖gn‖² − ⟨gn, go⟩) / ‖go‖²)
    // The max(0, ...) clamp (PR+) provides automatic restarts when CG
    // direction quality degrades, without needing an explicit Powell check.
    ((new_norm_sq - gn_go) / prev_grad_norm_sq).max(0.0)
}

/// Takes a step along `dir` and retracts onto the product of spheres.
///
/// For each vertex i: `v_trial[i] = normalize(v[i] + α · dir[i])`.
fn step_and_retract(v: &DenseMatrix, dir: &DenseMatrix, alpha: f64, v_trial: &mut DenseMatrix) {
    let r = v.rows();
    let vdata = v.data();
    let ddata = dir.data();
    let tdata = v_trial.data_mut();

    tdata
        .par_chunks_mut(r)
        .zip(vdata.par_chunks(r))
        .zip(ddata.par_chunks(r))
        .for_each(|((t_col, v_col), d_col)| {
            let mut norm_sq = 0.0_f64;
            for k in 0..t_col.len() {
                let val = alpha.mul_add(d_col[k], v_col[k]);
                t_col[k] = val;
                norm_sq = val.mul_add(val, norm_sq);
            }
            let inv_norm = 1.0 / norm_sq.sqrt();
            for val in t_col.iter_mut() {
                *val *= inv_norm;
            }
        });
}

/// Generates an r × n matrix of random unit vectors (one per column) via Box-Muller.
fn random_unit_vectors(n: usize, r: usize, rng: &mut impl Rng) -> DenseMatrix {
    let mut v = DenseMatrix::zeros(r, n);
    let data = v.data_mut();
    for pair in data.chunks_mut(2) {
        let (first, second) = box_muller(rng.random(), rng.random());
        pair[0] = first;
        if let Some(last) = pair.get_mut(1) {
            *last = second;
        }
    }

    // Normalize each column to unit length.
    for c in data.chunks_mut(r) {
        let norm_sq = dot(c, c);
        let inv_norm = 1.0 / norm_sq.sqrt();
        for val in c.iter_mut() {
            *val *= inv_norm;
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    fn graph_with(n: usize, edges: &[(usize, usize)]) -> Graph {
        let mut g = Graph::new(n);
        for &(u, v) in edges {
            g.add_edge(u, v, 1.0).unwrap_or_else(|e| panic!("{e}"));
        }
        g
    }

    fn triangle() -> Graph {
        graph_with(3, &[(0, 1), (1, 2), (0, 2)])
    }

    fn single_edge() -> Graph {
        graph_with(2, &[(0, 1)])
    }

    #[test]
    fn random_unit_vectors_have_unit_norm() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(123);
        let v = random_unit_vectors(50, 7, &mut rng);

        for i in 0..50 {
            let start = i * 7;
            let c = &v.data()[start..start + 7];
            let norm_sq: f64 = c.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-14,
                "column {i} norm² = {norm_sq}"
            );
        }
    }

    #[test]
    fn sparse_gradient_matches_dense_on_k3() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(7);
        let g = triangle();
        let r = 2;
        let n = 3;
        let v = random_unit_vectors(n, r, &mut rng);
        let edges = column_edges(&g, r);

        // Compute Riemannian gradient (which includes tangent projection).
        let mut riem_grad = DenseMatrix::zeros(r, n);
        let _ = compute_riemannian_gradient(&edges, &v, &mut riem_grad);

        // Dense gradient: −(1/2) L V.
        let l = g.laplacian();
        let vl = v.mul(&l).unwrap_or_else(|_| panic!("mul failed"));
        let euc_grad_dense = vl.scale(-0.5);

        // Project dense Euclidean gradient to tangent space manually.
        let mut riem_dense = DenseMatrix::zeros(r, n);
        let vdata = v.data();
        let edata = euc_grad_dense.data();
        let rdata = riem_dense.data_mut();

        rdata.copy_from_slice(edata);
        for (r_col, v_col) in rdata.chunks_mut(r).zip(vdata.chunks(r)) {
            project_to_tangent(v_col, r_col);
        }

        for idx in 0..(r * n) {
            let diff = (riem_grad.data()[idx] - riem_dense.data()[idx]).abs();
            assert!(diff < 1e-12, "mismatch at index {idx}: {diff}");
        }
    }

    #[test]
    fn sparse_objective_matches_dense_on_k3() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(99);
        let g = triangle();
        let r = 2;
        let n = 3;
        let v = random_unit_vectors(n, r, &mut rng);
        let edges = column_edges(&g, r);

        let obj_sparse = compute_objective_sparse(&edges, v.data(), r);

        // Dense: f = −(1/4) tr(V_math^T L V_math) = −(1/4) tr(V L V^T)
        let l = g.laplacian();
        let vt = v.transpose(); // n×r
        let lv = l.mul(&vt).unwrap_or_else(|_| panic!("mul failed")); // n×r
        let trace_val = vt
            .inner_product(&lv)
            .unwrap_or_else(|_| panic!("inner_product failed"));
        let obj_dense = -0.25 * trace_val;

        assert!(
            (obj_sparse - obj_dense).abs() < 1e-12,
            "sparse={obj_sparse}, dense={obj_dense}"
        );
    }

    #[test]
    fn riemannian_gradient_is_tangent() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let g = triangle();
        let r = 3;
        let n = 3;
        let v = random_unit_vectors(n, r, &mut rng);
        let edges = column_edges(&g, r);

        let mut riem_grad = DenseMatrix::zeros(r, n);
        let _ = compute_riemannian_gradient(&edges, &v, &mut riem_grad);

        // Each riem_grad column must be orthogonal to the corresponding v column.
        for i in 0..n {
            let start = i * r;
            let vi = &v.data()[start..start + r];
            let ri = &riem_grad.data()[start..start + r];
            let d: f64 = vi.iter().zip(ri).map(|(a, b)| a * b).sum();
            assert!(d.abs() < 1e-14, "vertex {i}: dot = {d}");
        }
    }

    #[test]
    fn retraction_produces_unit_norm() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(55);
        let r = 4;
        let n = 10;
        let v = random_unit_vectors(n, r, &mut rng);
        let grad = random_unit_vectors(n, r, &mut rng); // arbitrary direction
        let mut v_trial = DenseMatrix::zeros(r, n);

        step_and_retract(&v, &grad, 0.3, &mut v_trial);

        for i in 0..n {
            let start = i * r;
            let c = &v_trial.data()[start..start + r];
            let norm_sq: f64 = c.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-14,
                "column {i} norm² = {norm_sq}"
            );
        }
    }

    #[test]
    fn single_edge_converges_to_antipodal() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(0);
        let g = single_edge();

        let solver = BmSolver::new(BmSolverConfig {
            rank: Some(2),
            ..BmSolverConfig::default()
        });
        let sol = solver
            .solve_maxcut(&g, &mut rng, |_| {})
            .unwrap_or_else(|_| panic!("solve failed"));

        // Optimal: ‖v0 − v1‖² = 4 (antipodal), objective = −(1/4)·1·4 = −1.
        assert!(
            (sol.objective - (-1.0)).abs() < 1e-6,
            "objective = {}",
            sol.objective
        );
    }

    #[test]
    fn triangle_converges() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(1);
        let g = triangle();

        let solver = BmSolver::new(BmSolverConfig {
            rank: Some(3),
            ..BmSolverConfig::default()
        });
        let sol = solver
            .solve_maxcut(&g, &mut rng, |_| {})
            .unwrap_or_else(|_| panic!("solve failed"));

        // SDP bound for K3 = −objective ≈ 2.25 (from GW relaxation).
        let sdp_bound = -sol.objective;
        assert!(sdp_bound > 2.2, "sdp_bound = {sdp_bound}, expected > 2.2");
        assert!(sdp_bound < 2.3, "sdp_bound = {sdp_bound}, expected < 2.3");
    }
}
