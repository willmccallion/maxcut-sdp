# Semidefinite Programming & Interior Point Method

Mathematical reference for the `sdp` module. This document derives the primal-dual interior
point method step by step, from the standard form SDP through Newton direction computation
and convergence analysis. It serves as the implementation guide for `solver.rs` and `linalg.rs`.

---

## 1. Standard Form SDP

A **semidefinite program** (SDP) optimizes a linear objective over the intersection of an
affine subspace with the cone of positive semidefinite matrices.

**Primal problem:**

$$
\begin{aligned}
\min_{X \in \mathbb{S}^n} \quad & \langle C, X \rangle \\
\text{subject to} \quad & \langle A_i, X \rangle = b_i, \quad i = 1, \ldots, m \\
& X \succeq 0
\end{aligned}
$$

where:
- $X, C, A_i \in \mathbb{S}^n$ are $n \times n$ real symmetric matrices
- $\langle A, B \rangle = \text{tr}(A^\top B) = \sum_{j,k} A_{jk} B_{jk}$ is the matrix inner product
- $X \succeq 0$ means $X$ is positive semidefinite (all eigenvalues $\geq 0$)
- $b \in \mathbb{R}^m$ is the constraint right-hand side

The feasible set is the intersection of an affine subspace (the $m$ linear constraints) with
the positive semidefinite cone $\mathbb{S}^n_+$.

---

## 2. Dual SDP and Duality Theory

**Dual problem:**

$$
\begin{aligned}
\max_{y \in \mathbb{R}^m} \quad & b^\top y \\
\text{subject to} \quad & \sum_{i=1}^{m} y_i A_i + S = C \\
& S \succeq 0
\end{aligned}
$$

The variable $S \in \mathbb{S}^n$ is the **dual slack matrix**. The dual constraint can be
written as $S = C - \sum_{i=1}^{m} y_i A_i \succeq 0$.

### Weak Duality

For any primal feasible $X$ and dual feasible $(y, S)$:

$$
\langle C, X \rangle - b^\top y = \langle C - \sum_i y_i A_i, X \rangle = \langle S, X \rangle \geq 0
$$

The last inequality holds because both $S \succeq 0$ and $X \succeq 0$, and the inner product
of two PSD matrices is nonnegative. The quantity $\langle S, X \rangle$ is the **duality gap**.

### Strong Duality (Slater's Condition)

Strong duality ($\langle S, X \rangle = 0$ at optimality) holds if **Slater's condition** is
satisfied: there exists a **strictly feasible** primal point $\hat{X} \succ 0$ (positive
definite, not just semidefinite) satisfying all constraints, or a strictly feasible dual
point $(\hat{y}, \hat{S})$ with $\hat{S} \succ 0$.

For the Max-Cut SDP relaxation, Slater's condition is easily verified: $X = I$ is strictly
feasible for the primal (it's positive definite and satisfies $X_{ii} = 1$).

### Complementary Slackness

At optimality under strong duality: $XS = 0$ (equivalently $\langle X, S \rangle = 0$).
This is the matrix generalization of the complementary slackness condition from linear programming.

---

## 3. Primal-Dual Interior Point Method

The interior point method (IPM) follows the **central path** — a smooth curve of strictly
feasible points parameterized by a barrier parameter $\mu > 0$ — and drives $\mu \to 0$ to
approach the optimal solution.

### 3.1 Central Path

The central path consists of points $(X(\mu), y(\mu), S(\mu))$ satisfying:

$$
\begin{aligned}
\langle A_i, X \rangle &= b_i, \quad i = 1, \ldots, m & \text{(primal feasibility)} \\
\sum_{i=1}^{m} y_i A_i + S &= C & \text{(dual feasibility)} \\
XS &= \mu I & \text{(centrality condition)} \\
X &\succ 0, \quad S \succ 0
\end{aligned}
$$

As $\mu \to 0$, the centrality condition $XS = \mu I \to 0$ approaches complementary
slackness, and the central path converges to the primal-dual optimal solution.

**Barrier interpretation:** The central path point $X(\mu)$ minimizes the log-barrier problem:

$$
\min_X \quad \langle C, X \rangle - \mu \ln \det(X) \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i
$$

The gradient of $-\mu \ln \det(X)$ is $-\mu X^{-1}$, so the KKT conditions of this barrier
problem yield the central path equations.

### 3.2 KKT System and Newton Direction

At the current iterate $(X, y, S)$ with $X \succ 0$, $S \succ 0$, we linearize the central
path conditions to compute a Newton step $(\Delta X, \Delta y, \Delta S)$.

The residuals are:

$$
\begin{aligned}
r_p^{(i)} &= b_i - \langle A_i, X \rangle & \text{(primal residual)} \\
R_d &= C - \sum_i y_i A_i - S & \text{(dual residual)} \\
R_c &= \mu I - XS & \text{(centrality residual)}
\end{aligned}
$$

The linearized Newton system is:

$$
\begin{aligned}
\langle A_i, \Delta X \rangle &= r_p^{(i)}, \quad i = 1, \ldots, m \\
\sum_i \Delta y_i \cdot A_i + \Delta S &= R_d \\
S \, \Delta X + X \, \Delta S &= R_c
\end{aligned}
$$

**The symmetry problem:** The equation $S \, \Delta X + X \, \Delta S = R_c$ does not
guarantee that $\Delta X$ and $\Delta S$ are symmetric, even though $X$ and $S$ are symmetric.
The product of two symmetric matrices is symmetric only if they commute. This is a fundamental
difficulty specific to SDPs (not present in linear programming).

### 3.3 Nesterov-Todd (NT) Scaling

The **Nesterov-Todd direction** resolves the symmetry problem by introducing a scaling matrix
$W$ that symmetrizes the system. The NT scaling point $W$ is the unique symmetric positive
definite matrix satisfying:

$$
W S W = X
$$

**Computing $W$:** Given $X, S \succ 0$:

1. Compute the Cholesky factorization $X = L_X L_X^\top$
2. Form $M = L_X^\top S L_X$ (this is SPD)
3. Compute the eigendecomposition $M = Q \Lambda Q^\top$
4. Then $W = L_X Q \Lambda^{-1/2} Q^\top L_X^\top$

Equivalently, $W = X^{1/2} (X^{1/2} S X^{1/2})^{-1/2} X^{1/2}$ where $X^{1/2}$ is the
unique symmetric positive definite square root.

**Symmetrized Newton system with NT scaling:**

Define the scaled variable $\bar{\Delta X} = W^{-1} \Delta X \, W^{-1}$. The centrality
equation becomes:

$$
\Delta S + W \, \bar{\Delta X} \, W = \mu (W S W)^{-1} - S + R_d^{(\text{sym})}
$$

This system preserves symmetry of the search directions. In practice, the NT direction
is computed by:

1. **Eliminate $\Delta S$:** From the dual equation, $\Delta S = R_d - \sum_i \Delta y_i A_i$
2. **Substitute into the centrality equation** to express $\Delta X$ in terms of $\Delta y$
3. **Substitute into the primal equations** to get the **Schur complement system** for $\Delta y$

### 3.4 Schur Complement System

After elimination, $\Delta y$ satisfies the $m \times m$ linear system:

$$
M \, \Delta y = h
$$

where the **Schur complement matrix** $M \in \mathbb{R}^{m \times m}$ has entries:

$$
M_{ij} = \langle A_i, W A_j W \rangle = \text{tr}(A_i W A_j W)
$$

and the right-hand side $h \in \mathbb{R}^m$ has entries:

$$
h_i = b_i - \langle A_i, X \rangle - \langle A_i, \mu S^{-1} - X \rangle + \langle A_i, W R_d W \rangle
$$

The matrix $M$ is symmetric positive definite (since $W \succ 0$ and the constraint matrices
$A_i$ are linearly independent), so we solve via **Cholesky factorization**: $M = L_M L_M^\top$,
then forward/back substitution.

Once $\Delta y$ is known:
- $\Delta S = R_d - \sum_i \Delta y_i A_i$
- $\Delta X = \mu S^{-1} - X - W \Delta S \, W$ (symmetrized)

### 3.5 Step Size Selection

We need the updated iterates to remain strictly feasible:

$$
X + \alpha \, \Delta X \succ 0 \quad \text{and} \quad S + \alpha \, \Delta S \succ 0
$$

The maximum feasible step size for $X$ is:

$$
\alpha_X^{\max} = \begin{cases}
\infty & \text{if } \Delta X \succeq 0 \\
\frac{-1}{\lambda_{\min}(X^{-1/2} \Delta X \, X^{-1/2})} & \text{otherwise}
\end{cases}
$$

where $\lambda_{\min}(\cdot)$ denotes the minimum eigenvalue. Similarly for $\alpha_S^{\max}$
using $S$ and $\Delta S$. The actual step size uses a safety factor:

$$
\alpha = \gamma \cdot \min(\alpha_X^{\max}, \alpha_S^{\max})
$$

where $\gamma \in (0, 1)$ (typically $\gamma = 0.99$ or $0.95$) ensures we stay strictly
in the interior.

**Implementation:** Rather than computing $X^{-1/2}$, we can compute the eigenvalues of
$L_X^{-1} \Delta X \, L_X^{-\top}$ (where $X = L_X L_X^\top$) and find the most negative one.

### 3.6 Barrier Parameter Update

The barrier parameter $\mu$ controls how quickly we approach the boundary of the PSD cone.
The current complementarity measure is:

$$
\mu_{\text{current}} = \frac{\langle X, S \rangle}{n}
$$

The next barrier parameter is:

$$
\mu_{\text{next}} = \sigma \cdot \mu_{\text{current}}
$$

where $\sigma \in (0, 1)$ is the **centering parameter**. This solver uses the long-step choice

$$
\sigma = 0.2
$$

The short-step schedule $\sigma = 1 - 0.3/\sqrt{n}$ achieves the $O(\sqrt{n} \log(1/\varepsilon))$
iteration bound, but it cuts $\mu$ by only about 4% per iteration at $n = 60$, so reaching
$\varepsilon = 10^{-8}$ takes several hundred iterations. With $\sigma = 0.2$ each full step cuts
$\mu$ fivefold; the worst-case bound becomes $O(n \log(1/\varepsilon))$, but on the Biq Mac
instances ($n \leq 100$) the solver converges in 16–24 iterations, and its bounds agree with an
independent solver to about $10^{-8}$.

### 3.7 Convergence Criterion

The algorithm terminates when the following conditions are met:

$$
\begin{aligned}
\text{Duality gap:} \quad & \frac{\langle X, S \rangle}{1 + |\langle C, X \rangle|} < \varepsilon \\
\text{Primal infeasibility:} \quad & \frac{\|b - \mathcal{A}(X)\|_2}{1 + \|b\|_2} < \varepsilon \\
\text{Dual infeasibility:} \quad & \frac{\|C - \mathcal{A}^{*}(y) - S\|_F}{1 + \|C\|_F} < \varepsilon
\end{aligned}
$$

where $\mathcal{A}(X) = (\langle A_1, X \rangle, \ldots, \langle A_m, X \rangle)^\top$ and
$\mathcal{A}^{*}(y) = \sum_i y_i A_i$. Typical tolerance: $\varepsilon = 10^{-8}$.

### 3.8 Complete Algorithm

```
Input: SDP data (C, {A_i}, b), tolerance ε, max iterations K
Output: Approximate optimal (X*, y*, S*)

1. Initialize: X = I, S = I, y = 0 (or use heuristic)
2. For k = 1, 2, ..., K:
   a. Compute μ = ⟨X, S⟩ / n
   b. If convergence criteria met: return (X, y, S)
   c. Set μ_target = σ · μ  where σ = 0.2
   d. Compute NT scaling matrix W from X, S
   e. Form Schur complement matrix M and RHS h
   f. Solve M Δy = h via Cholesky
   g. Recover ΔX, ΔS from Δy
   h. Compute step size α with safety factor γ
   i. Update: X ← X + α ΔX,  y ← y + α Δy,  S ← S + α ΔS
3. Fail: max iterations reached without converging
```

---

## 4. Worked Example: 2×2 SDP

Consider the SDP:

$$
\begin{aligned}
\min \quad & x_{11} + x_{22} \\
\text{s.t.} \quad & x_{12} = 1 \\
& X = \begin{pmatrix} x_{11} & x_{12} \\ x_{12} & x_{22} \end{pmatrix} \succeq 0
\end{aligned}
$$

**Data:**

$$
C = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I, \quad
A_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad
b_1 = 2
$$

Note: $\langle A_1, X \rangle = 2 x_{12} = 2$, so $x_{12} = 1$.

**Optimal solution:** $X \succeq 0$ with $x_{12} = 1$ requires $x_{11} x_{22} \geq x_{12}^2 = 1$.
Minimizing $x_{11} + x_{22}$ subject to $x_{11} x_{22} \geq 1$: by AM-GM,
$x_{11} + x_{22} \geq 2\sqrt{x_{11} x_{22}} \geq 2$, with equality at $x_{11} = x_{22} = 1$.

$$
X^{*} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}, \quad \text{optimal value} = 2
$$

Note that $X^{*}$ is PSD (eigenvalues $0$ and $2$) but not PD — it lies on the boundary.

**Dual:** $\max \, 2y_1$ subject to $C - y_1 A_1 = I - y_1 \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \succeq 0$.
This requires $1 - y_1^2 \geq 0$ (from the determinant condition), so $|y_1| \leq 1$.
Maximizing $2y_1$: $y_1^{*} = 1$, dual optimal $= 2$. Zero duality gap confirms strong duality.

**IPM iteration (sketch):**

Starting from $X = 2I$, $S = 2I$, $y = 0$, $\mu = 4$:

- **Iteration 1:** $\mu = 4$, set $\mu_{\text{target}} \approx 3.15$.
  Compute NT scaling $W$, Schur complement (scalar: $m = 1$), solve for $\Delta y$.
  After step: $x_{12}$ moves toward $1$, gap decreases.

- **Iterations 2–10:** $\mu$ decreases geometrically. $X$ converges to $X^{*}$, $S$ converges
  to $S^{*} = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$, gap $\to 0$.

Typically converges in $\sim 10\text{–}15$ iterations to $\varepsilon = 10^{-8}$.

---

## 5. Computational Complexity

### Per-Iteration Cost

| Operation | Cost |
|-----------|------|
| NT scaling $W$ (eigendecomposition of $n \times n$ matrix) | $O(n^3)$ |
| Schur complement matrix $M$ ($m^2$ inner products of $n \times n$ matrices) | $O(m^2 n^2)$ |
| Cholesky factorization of $M$ ($m \times m$) | $O(m^3)$ |
| Search direction recovery ($\Delta X$, $\Delta S$) | $O(m n^2)$ |
| Step size (eigenvalues of $n \times n$ matrix) | $O(n^3)$ |
| **Total per iteration** | $O(m^2 n^2 + m^3)$ |

### Iteration Count

With long-step centering (§3.6) the primal-dual IPM needs $O(n \log(1/\varepsilon))$ iterations in
the worst case, independent of $m$ (the number of constraints); in practice it takes 16–24 on the
Biq Mac instances.

### Max-Cut Specific

For Max-Cut on an $n$-vertex graph:
- Matrix dimension: $n \times n$
- Number of constraints: $m = n$ (one diagonal constraint per vertex)
- Per iteration: $O(n^2 \cdot n^2 + n^3) = O(n^4)$
- Total: $O(n^4)$ times the iteration count

For a from-scratch solver without exploiting sparsity, practical limit is roughly $n \leq 100$–$200$.

---

## 6. Linear Algebra Operations Reference

The following operations from `linalg.rs` are needed by the solver. Each references the
section of this document where it appears.

| Operation | Used in | Notes |
|-----------|---------|-------|
| Cholesky factorization $A = LL^\top$ | §3.3 (NT scaling), §3.4 (Schur solve) | Fails iff $A$ is not positive definite |
| Forward/back substitution $Lx = b$, $L^\top x = b$ | §3.4 (Schur solve) | $O(n^2)$ per solve |
| Matrix multiply $AB$ | §3.3 (forming $WAW$), §3.4 | $O(n^3)$ naive, can optimize later |
| Trace inner product $\langle A, B \rangle = \text{tr}(A^\top B)$ | §3.4 (Schur entries) | $O(n^2)$, just sum element products |
| Eigendecomposition $A = Q \Lambda Q^\top$ | §3.3 (NT scaling), §3.5 (step size) | Symmetric matrices only; Jacobi or QR |
| Minimum eigenvalue $\lambda_{\min}(A)$ | §3.5 (step size), bound certificates | Householder tridiagonalization, then Sturm-sequence bisection |
| Log-determinant $\ln \det(A) = 2 \sum_i \ln L_{ii}$ | §3.1 (barrier) | Via Cholesky diagonal |
| Matrix inverse $A^{-1}$ | §3.4 (computing $S^{-1}$) | Via Cholesky: solve $L L^\top X = I$ |
| Symmetric matrix square root $A^{1/2}$ | §3.3 (NT scaling) | Via eigendecomposition: $Q \Lambda^{1/2} Q^\top$ |
| Symmetric matrix inverse square root $A^{-1/2}$ | §3.5 (step size) | Via eigendecomposition: $Q \Lambda^{-1/2} Q^\top$ |

---

## 7. Burer-Monteiro Low-Rank Factorization

The dense IPM (§3) is $O(n^4)$ per iteration. For large or sparse graphs, the **Burer-Monteiro
(BM) method** replaces the $n \times n$ PSD matrix $X$ with a low-rank factorization $X = VV^\top$
where $V \in \mathbb{R}^{n \times r}$ and $r \ll n$, then optimizes over $V$ directly.

### 7.1 Reformulated Problem

The Max-Cut SDP relaxation (see `maxcut/MATH.md` §3) is:

$$
\max \; \frac{1}{4} \langle L, X \rangle \quad \text{s.t.} \quad X_{ii} = 1,\; X \succeq 0
$$

Substituting $X = VV^\top$ and enforcing $X_{ii} = \|v_i\|^2 = 1$ gives:

$$
\min_{V \in \mathbb{R}^{n \times r}} \; f(V) = -\frac{1}{4} \sum_{(i,j) \in E} w_{ij} \|v_i - v_j\|^2
\quad \text{s.t.} \quad \|v_i\| = 1 \;\;\forall\, i
$$

Minimizing the negative quantity maximizes the cut. The PSD constraint is automatically
satisfied since $VV^\top$ is always positive semidefinite.

### 7.2 Rank Selection (Barvinok-Pataki Bound)

The Barvinok-Pataki bound states that the SDP has an optimal solution $X^{*}$ with
$\text{rank}(X^{*}) = r^{*}$ satisfying $\binom{r^{*}+1}{2} \leq m$ (the number of
constraints). For Max-Cut, $m = n$ (one diagonal constraint per vertex), so
$r^{*} \leq \lceil\sqrt{2n}\rceil$.

Using $r \geq r^{*}$ guarantees that the BM formulation can represent the SDP optimum.

### 7.3 No Spurious Local Minima

**Theorem (Boumal, Voroninski, Bandeira 2016):** When $r \geq \sqrt{2n}$, every second-order
critical point of $f(V)$ on the product of spheres is a global optimum of the SDP relaxation.

This means that gradient descent (with sufficient rank) converges to the SDP optimum
regardless of initialization — there are no spurious local minima to get trapped in.

### 7.4 Product-of-Spheres Manifold

The constraint $\|v_i\| = 1$ for all $i$ defines the **product of spheres** manifold:

$$
\mathcal{M} = \underbrace{S^{r-1} \times \cdots \times S^{r-1}}_{n \text{ copies}}
$$

The **tangent space** at $V$ consists of matrices $\dot{V}$ with $\langle v_i, \dot{v}_i \rangle = 0$
for all $i$ (each tangent vector is orthogonal to the corresponding point on its sphere).

### 7.5 Riemannian Gradient

The Euclidean gradient of the BM objective is:

$$
\nabla f(V) = -\frac{1}{2} L V
$$

where $L$ is the graph Laplacian. The **Riemannian gradient** is the projection of
$\nabla f$ onto the tangent space:

$$
\text{grad} f(V)_i = \nabla f(V)_i - \langle \nabla f(V)_i, v_i \rangle \, v_i
$$

This subtracts the normal component so the gradient lies in the tangent space.

### 7.6 Sparse Gradient Computation

Rather than forming the $n \times n$ Laplacian and computing $-\frac{1}{2}LV$ in $O(n^2 r)$,
we compute the gradient by iterating over edges in $O(|E| \cdot r)$:

$$
\nabla f(V)_i = -\frac{1}{2} \sum_{j:\,(i,j)\in E} w_{ij}(v_i - v_j)
$$

For each edge $(i, j, w)$, we accumulate:
- $\text{grad}[i] \mathrel{-}= \frac{w}{2}(v_i - v_j)$
- $\text{grad}[j] \mathrel{+}= \frac{w}{2}(v_i - v_j)$

This is the key to scalability: per-iteration cost scales with graph sparsity, not $n^2$.

### 7.7 Retraction

After taking a step $V' = V - \alpha \cdot \text{grad} f(V)$ in the tangent space,
the result no longer lies on $\mathcal{M}$. The **retraction** maps it back:

$$
\text{Retr}(V')_i = \frac{v'_i}{\|v'_i\|}
$$

This is the simplest retraction on the sphere (normalization), with $O(n \cdot r)$ cost.

### 7.8 Armijo Backtracking on Manifolds

Step size selection uses standard Armijo backtracking adapted for manifolds:

1. Start with step size $\alpha$ (try growing from previous: $\alpha \leftarrow \min(\alpha/\beta, \alpha_0)$)
2. Compute trial point $V' = \text{Retr}(V - \alpha \cdot g)$ where $g = \text{grad} f(V)$
3. Accept if $f(V') \leq f(V) - c_1 \alpha \|g\|_F^2$ (sufficient decrease)
4. Otherwise shrink: $\alpha \leftarrow \beta \alpha$ and repeat

Parameters: $c_1 = 10^{-4}$, $\beta = 0.5$, $\alpha_0 = 1.0$.

### 7.9 Complexity Comparison

| | Dense IPM (§3) | Burer-Monteiro (§7) |
|---|---|---|
| **Per iteration** | $O(n^4)$ | $O(\|E\| \cdot r)$ |
| **Iterations** | $O(n \log(1/\varepsilon))$ worst case, ~20 in practice | $O(1/\varepsilon)$ (gradient descent) |
| **Memory** | $O(n^2)$ | $O(n \cdot r)$ |
| **Practical limit** | $n \approx 100$–$200$ | $n \approx 10{,}000+$ (sparse) |
| **Guarantee** | Exact SDP solution | Global SDP optimum (when $r \geq \sqrt{2n}$) |

For a sparse graph with $|E| = O(n)$ and $r = O(\sqrt{n})$, the per-iteration cost is
$O(n^{3/2})$ — a factor of $n^{5/2}$ improvement over the dense IPM.
