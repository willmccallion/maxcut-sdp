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
use crate::sdp::linalg::{DenseMatrix, dot};
use rand::Rng;
use rayon::prelude::*;

/// Configuration for the Burer-Monteiro solver.
#[derive(Debug, Clone)]
pub struct BmSolverConfig {
    /// Rank of the factorization. `None` = ⌈√(2n)⌉, clamped to [2, n].
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
    /// Number of random restarts (best solution kept).
    pub num_restarts: usize,
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
            num_restarts: 1,
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
    /// Final Riemannian gradient Frobenius norm.
    pub gradient_norm: f64,
    /// Total gradient descent iterations taken (across the best restart).
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

    /// Solves the Max-Cut SDP relaxation via Burer-Monteiro with nonlinear CG.
    ///
    /// Returns a [`BmSolution`] whose `vectors` field can be passed directly to
    /// [`multi_round_from_vectors`](crate::maxcut::rounding::multi_round_from_vectors).
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty.
    pub fn solve_maxcut(&self, graph: &Graph, rng: &mut impl Rng) -> Result<BmSolution, GwError> {
        self.solve_maxcut_with_context(graph, rng, None)
    }

    /// Like `solve_maxcut`, but prints `best_known_cut` alongside iteration progress.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty.
    pub fn solve_maxcut_with_context(
        &self,
        graph: &Graph,
        rng: &mut impl Rng,
        best_known_cut: Option<f64>,
    ) -> Result<BmSolution, GwError> {
        let n = graph.num_vertices();
        if n == 0 {
            return Err(GwError::InvalidGraph("empty graph".into()));
        }

        let r = self
            .config
            .rank
            .unwrap_or_else(|| ((2.0 * n as f64).sqrt()).ceil() as usize);
        let r = r.max(2).min(n);

        let edges = column_edges(graph, r);

        let mut best_obj = f64::INFINITY;
        let mut best_grad_norm = 0.0;
        let mut best_iters = 0;
        let mut best_v: Option<DenseMatrix> = None;

        // Pre-allocate work buffers (reused across restarts).
        let mut riem_grad = DenseMatrix::zeros(r, n);
        let mut search_dir = DenseMatrix::zeros(r, n);
        let mut riem_grad_prev = DenseMatrix::zeros(r, n);
        let mut v_trial = DenseMatrix::zeros(r, n);

        for _ in 0..self.config.num_restarts {
            let mut v = random_unit_vectors(n, r, rng);
            let mut f_current = compute_objective_sparse(&edges, v.data(), r);
            let mut alpha = self.config.initial_step_size;
            let mut final_grad_norm = 0.0;
            let mut iters = 0;
            let mut prev_grad_norm_sq = 0.0_f64;
            let mut f_checkpoint = f_current;
            let obj_check_interval = 100_usize.max(n / 10);
            let mut stall_count = 0_usize;

            for iter in 0..self.config.max_iterations {
                // Phase 5: Fused gradient + tangent projection + norm.
                let grad_norm = compute_riemannian_gradient(&edges, &v, &mut riem_grad);
                final_grad_norm = grad_norm;
                iters = iter + 1;

                if grad_norm < self.config.gradient_tolerance {
                    break;
                }

                // Objective-based stopping: if relative improvement over the
                // last window is negligible, further iterations won't help rounding.
                if iter > 0 && iter % obj_check_interval == 0 {
                    let denom = f_current.abs().max(1.0);
                    let rel_improve = (f_checkpoint - f_current) / denom;
                    f_checkpoint = f_current;
                    if rel_improve < self.config.objective_tolerance {
                        stall_count += 1;
                        if stall_count >= self.config.max_stall_windows {
                            break;
                        }
                    } else {
                        stall_count = 0;
                    }
                }

                if iter > 0 && iter % 500 == 0 {
                    if let Some(bk) = best_known_cut {
                        eprintln!(
                            "  iter {iter}: obj={:.4}, grad={grad_norm:.2e}, best={bk:.0}",
                            -f_current
                        );
                    } else {
                        eprintln!("  iter {iter}: obj={:.4}, grad={grad_norm:.2e}", -f_current);
                    }
                }

                let grad_norm_sq = grad_norm * grad_norm;

                // Phase 6: Nonlinear CG (Polak-Ribière with restarts).
                let beta = if iter == 0 {
                    0.0
                } else {
                    compute_pr_beta(&riem_grad, &riem_grad_prev, prev_grad_norm_sq, n, iter)
                };

                if beta == 0.0 {
                    // Steepest descent: d = -g
                    let ddata = search_dir.data_mut();
                    let gdata = riem_grad.data();
                    for k in 0..ddata.len() {
                        ddata[k] = -gdata[k];
                    }
                } else {
                    // d = -g + beta * T(d_prev)
                    // First transport d_prev to tangent space at current v,
                    // then combine.
                    project_to_tangent_inplace(&v, &mut search_dir);
                    let ddata = search_dir.data_mut();
                    let gdata = riem_grad.data();
                    for k in 0..ddata.len() {
                        ddata[k] = beta.mul_add(ddata[k], -gdata[k]);
                    }
                }

                // Check descent: if ⟨g, d⟩ ≥ 0, reset to steepest descent.
                let dir_deriv = dot(riem_grad.data(), search_dir.data());
                if dir_deriv >= 0.0 {
                    let ddata = search_dir.data_mut();
                    let gdata = riem_grad.data();
                    for k in 0..ddata.len() {
                        ddata[k] = -gdata[k];
                    }
                }

                // Save current gradient for next iteration's CG update.
                riem_grad_prev.data_mut().copy_from_slice(riem_grad.data());
                prev_grad_norm_sq = grad_norm_sq;

                // Armijo backtracking along search_dir.
                let dir_deriv_for_armijo = dot(riem_grad.data(), search_dir.data());
                alpha = (alpha / self.config.backtracking_beta).min(self.config.initial_step_size);
                let sufficient_decrease = self.config.armijo_c1 * dir_deriv_for_armijo;

                let mut accepted = false;
                loop {
                    step_and_retract(&v, &search_dir, alpha, &mut v_trial);
                    let f_trial = compute_objective_sparse(&edges, v_trial.data(), r);

                    if f_trial <= alpha.mul_add(sufficient_decrease, f_current) {
                        std::mem::swap(&mut v, &mut v_trial);
                        f_current = f_trial;
                        accepted = true;
                        break;
                    }

                    alpha *= self.config.backtracking_beta;
                    if alpha < 1e-20 {
                        break;
                    }
                }

                if !accepted {
                    break;
                }
            }

            eprintln!(
                "BM: iters={iters}, obj={:.4}, grad={final_grad_norm:.2e}",
                -f_current
            );

            if f_current < best_obj {
                best_obj = f_current;
                best_grad_norm = final_grad_norm;
                best_iters = iters;
                best_v = Some(v);
            }
        }

        best_v.map_or_else(
            || {
                Err(GwError::SolverFailed(
                    "BM solver produced no solution".into(),
                ))
            },
            |vectors| {
                Ok(BmSolution {
                    vectors,
                    objective: best_obj,
                    gradient_norm: best_grad_norm,
                    iterations: best_iters,
                })
            },
        )
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
                let d = dot(v_col, g_col);
                for k in 0..g_col.len() {
                    g_col[k] = d.mul_add(-v_col[k], g_col[k]);
                }
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
        .for_each(|(h_col, v_col)| {
            let d = dot(v_col, h_col);
            for k in 0..h_col.len() {
                h_col[k] = d.mul_add(-v_col[k], h_col[k]);
            }
        });
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

    if prev_grad_norm_sq < 1e-30 {
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
    let total = r * n;

    // Box-Muller: generate pairs of Gaussian samples.
    let pairs = total / 2;
    for p in 0..pairs {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        data[2 * p] = mag * angle.cos();
        data[2 * p + 1] = mag * angle.sin();
    }
    if total % 2 == 1 {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        data[total - 1] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
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

        for i in 0..n {
            let start = i * r;
            let vi = &vdata[start..start + r];
            let ei = &edata[start..start + r];
            let d = dot(vi, ei);
            for k in 0..r {
                rdata[start + k] = d.mul_add(-vi[k], ei[k]);
            }
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
            .solve_maxcut(&g, &mut rng)
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
            .solve_maxcut(&g, &mut rng)
            .unwrap_or_else(|_| panic!("solve failed"));

        // SDP bound for K3 = −objective ≈ 2.25 (from GW relaxation).
        let sdp_bound = -sol.objective;
        assert!(sdp_bound > 2.2, "sdp_bound = {sdp_bound}, expected > 2.2");
        assert!(sdp_bound < 2.3, "sdp_bound = {sdp_bound}, expected < 2.3");
    }
}
