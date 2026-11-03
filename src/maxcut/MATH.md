# Goemans-Williamson Max-Cut Algorithm

Mathematical reference for the `maxcut` module. This document derives the Goemans-Williamson
(GW) approximation algorithm for the Maximum Cut problem, proves the $\alpha_{\text{GW}} \approx 0.878$
approximation guarantee, and provides worked examples. It serves as the implementation guide
for `mod.rs`, `rounding.rs`, and `exact.rs`.

---

## 1. The Maximum Cut Problem

Given an undirected graph $G = (V, E)$ with nonnegative edge weights $w_{ij} \geq 0$, the
**Maximum Cut** (Max-Cut) problem asks for a partition of the vertices into two sets $S$ and
$\bar{S} = V \setminus S$ that maximizes the total weight of edges crossing the partition:

$$
\text{MaxCut}(G) = \max_{S \subseteq V} \sum_{\substack{(i,j) \in E \\ i \in S, \, j \in \bar{S}}} w_{ij}
$$

Max-Cut is NP-hard (Karp, 1972). No polynomial-time algorithm can solve it exactly unless
P = NP.

### Integer Programming Formulation

Assign each vertex a label $x_i \in \{-1, +1\}$. Vertices $i$ and $j$ are on opposite sides
of the cut if and only if $x_i \neq x_j$, which happens when $x_i x_j = -1$. The indicator
for this event is $\frac{1 - x_i x_j}{2}$, giving:

$$
\text{MaxCut}(G) = \max_{x \in \{-1,+1\}^n} \frac{1}{2} \sum_{(i,j) \in E} w_{ij} (1 - x_i x_j)
$$

This is an integer quadratic program — NP-hard to solve exactly.

---

## 2. Vector Relaxation

The key insight of Goemans and Williamson is to **relax** the scalar constraint
$x_i \in \{-1, +1\}$ to a vector constraint $\mathbf{v}_i \in \mathbb{R}^n$ with
$\|\mathbf{v}_i\| = 1$ (unit vectors on the sphere $S^{n-1}$).

$$
\max_{\mathbf{v}_1, \ldots, \mathbf{v}_n \in S^{n-1}} \frac{1}{2} \sum_{(i,j) \in E} w_{ij} (1 - \langle \mathbf{v}_i, \mathbf{v}_j \rangle)
$$

This is a relaxation because $\{-1, +1\} \subset S^{n-1}$ (scalars $\pm 1$ are unit vectors
in $\mathbb{R}^1 \subset \mathbb{R}^n$). So the vector program's optimal value is at least as
large as the integer program's.

---

## 3. Equivalence to SDP Relaxation

Define the **Gram matrix** $X \in \mathbb{S}^n$ by $X_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$.
The matrix $X$ has the following properties:

1. $X \succeq 0$ (positive semidefinite) — because $X = V^\top V$ where column $i$ of $V$ is $\mathbf{v}_i$
2. $X_{ii} = \|\mathbf{v}_i\|^2 = 1$ for all $i$ (unit vector constraint)

Conversely, any $X \succeq 0$ with $X_{ii} = 1$ can be factored as $X = V^\top V$ to recover
unit vectors. This establishes a bijection between feasible vector programs and feasible SDPs.

### The Laplacian Matrix

The **graph Laplacian** $L \in \mathbb{S}^n$ is defined as:

$$
L = D - W
$$

where:
- $W$ is the **weighted adjacency matrix**: $W_{ij} = w_{ij}$ if $(i,j) \in E$, else $0$
- $D$ is the **degree matrix**: $D_{ii} = \sum_{j} w_{ij}$ (diagonal)

Explicitly:

$$
L_{ij} = \begin{cases}
\sum_{k \neq i} w_{ik} & \text{if } i = j \\
-w_{ij} & \text{if } i \neq j
\end{cases}
$$

The Laplacian is always positive semidefinite ($L \succeq 0$), with $L \mathbf{1} = 0$.

### SDP Formulation

The vector program objective rewrites as:

$$
\frac{1}{2} \sum_{(i,j) \in E} w_{ij}(1 - X_{ij})
= \frac{1}{2} \sum_{(i,j) \in E} w_{ij} - \frac{1}{2} \sum_{(i,j) \in E} w_{ij} X_{ij}
$$

Using the identity $\langle L, X \rangle = \sum_{(i,j) \in E} w_{ij}(X_{ii} + X_{jj} - 2X_{ij})$
and $X_{ii} = 1$:

$$
\langle L, X \rangle = \sum_{(i,j) \in E} w_{ij}(2 - 2X_{ij}) = 2 \sum_{(i,j) \in E} w_{ij}(1 - X_{ij})
$$

Therefore the objective equals $\frac{1}{4} \langle L, X \rangle$. The **Max-Cut SDP relaxation** is:

$$
\begin{aligned}
\text{SDP}_{\text{OPT}} = \max_{X \in \mathbb{S}^n} \quad & \frac{1}{4} \langle L, X \rangle \\
\text{subject to} \quad & X_{ii} = 1, \quad i = 1, \ldots, n \\
& X \succeq 0
\end{aligned}
$$

To put this in standard minimization form for the SDP solver: set $C = -\frac{1}{4} L$, and
use constraint matrices $A_i = e_i e_i^\top$ (the matrix with $1$ at position $(i,i)$ and $0$
elsewhere), with $b_i = 1$.

---

## 4. Random Hyperplane Rounding

Given the SDP optimal solution $X^{*}$, the GW algorithm **rounds** the continuous solution to
a discrete partition.

### Rounding Procedure

1. Compute vectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ from $X^{*}$ via Cholesky:
   $X^{*} = V^\top V$, and $\mathbf{v}_i$ is column $i$ of $V$.
2. Generate a random vector $\mathbf{r} \sim \mathcal{N}(0, I_n)$ (standard Gaussian in $\mathbb{R}^n$).
3. Assign vertices to the partition:

$$
\text{vertex } i \in \begin{cases} S & \text{if } \langle \mathbf{v}_i, \mathbf{r} \rangle \geq 0 \\ \bar{S} & \text{if } \langle \mathbf{v}_i, \mathbf{r} \rangle < 0 \end{cases}
$$

The random hyperplane with normal $\mathbf{r}$ separates $\mathbb{R}^n$ into two half-spaces.
Vertices whose vectors fall on opposite sides of this hyperplane are placed in different sets.

### Separation Probability

**Claim:** For any two unit vectors $\mathbf{u}, \mathbf{v} \in S^{n-1}$:

$$
\Pr_{\mathbf{r} \sim \mathcal{N}(0, I)}\left[\text{sign}(\langle \mathbf{u}, \mathbf{r} \rangle) \neq \text{sign}(\langle \mathbf{v}, \mathbf{r} \rangle)\right] = \frac{\arccos(\langle \mathbf{u}, \mathbf{v} \rangle)}{\pi}
$$

**Proof:** The projections $\langle \mathbf{u}, \mathbf{r} \rangle$ and $\langle \mathbf{v}, \mathbf{r} \rangle$
are jointly Gaussian with zero mean, unit variance, and correlation
$\langle \mathbf{u}, \mathbf{v} \rangle = \cos \theta$. The probability that two standard
Gaussians with correlation $\cos \theta$ have different signs equals the fraction of the plane
where they disagree, which is $\theta / \pi$. $\square$

### Expected Cut Value

$$
\mathbb{E}[\text{cut}] = \sum_{(i,j) \in E} w_{ij} \cdot \frac{\arccos(X^{*}_{ij})}{\pi}
$$

---

## 5. The Approximation Guarantee

### The Key Inequality

**Theorem (Goemans-Williamson, 1995):** For all $z \in [-1, 1]$:

$$
\frac{\arccos(z)}{\pi} \geq \alpha_{\text{GW}} \cdot \frac{1 - z}{2}
$$

where $\alpha_{\text{GW}} = \min_{\theta \in (0, \pi]} f(\theta)$ and

$$
f(\theta) = \frac{2}{\pi} \cdot \frac{\theta}{1 - \cos \theta}
$$

### Proof of the Inequality

Substitute $z = \cos \theta$ where $\theta = \arccos(z) \in [0, \pi]$. The inequality becomes:

$$
\frac{\theta}{\pi} \geq \alpha_{\text{GW}} \cdot \frac{1 - \cos \theta}{2}
$$

Rearranging:

$$
\frac{2\theta}{\pi(1 - \cos \theta)} = f(\theta) \geq \alpha_{\text{GW}}
$$

So we need $\alpha_{\text{GW}} = \min_{\theta \in (0, \pi]} f(\theta)$.

**Behavior at the endpoints:**
- As $\theta \to 0^+$: Using $1 - \cos \theta \approx \theta^2/2$, we get $f(\theta) \approx \frac{4}{\pi \theta} \to \infty$
- At $\theta = \pi$: $f(\pi) = \frac{2\pi}{\pi \cdot 2} = 1$

So $f$ decreases from $+\infty$, reaches a minimum, then returns to $1$. The minimum is in the interior.

**Finding the critical point:** Set $f'(\theta) = 0$.

$$
f(\theta) = \frac{2}{\pi} \cdot \frac{\theta}{1 - \cos \theta}
$$

$$
f'(\theta) = \frac{2}{\pi} \cdot \frac{(1 - \cos \theta) - \theta \sin \theta}{(1 - \cos \theta)^2}
$$

Setting the numerator to zero:

$$
1 - \cos \theta = \theta \sin \theta
$$

This is a transcendental equation. Using the identity $1 - \cos \theta = 2 \sin^2(\theta/2)$
and $\theta \sin \theta = 2\theta \sin(\theta/2) \cos(\theta/2)$:

$$
2 \sin^2(\theta/2) = 2\theta \sin(\theta/2) \cos(\theta/2)
$$

$$
\sin(\theta/2) = \theta \cos(\theta/2)
$$

$$
\tan(\theta/2) = \theta
$$

The equation $\tan(\theta/2) = \theta$ has a unique solution in $(0, \pi)$:

$$
\theta^{*} \approx 2.331122 \text{ radians} \approx 133.6°
$$

**Computing $\alpha_{\text{GW}}$:**

At $\theta^{*}$, since $\tan(\theta^{*}/2) = \theta^{*}$, we can compute
$\cos \theta^{*} = (1 - \theta^{*2})/(1 + \theta^{*2}) \approx -0.6892$. Then:

$$
\alpha_{\text{GW}} = f(\theta^{*}) = \frac{2}{\pi} \cdot \frac{\theta^{*}}{1 - \cos \theta^{*}} \approx \frac{2}{\pi} \cdot \frac{2.331122}{1.6892} \approx \frac{2}{\pi} \cdot 1.3800 \approx 0.87856\ldots
$$

### Main Theorem

**Theorem:** The GW algorithm achieves an expected cut value satisfying:

$$
\mathbb{E}[\text{cut}] \geq \alpha_{\text{GW}} \cdot \text{SDP}_{\text{OPT}} \geq \alpha_{\text{GW}} \cdot \text{OPT}
$$

**Proof:**

$$
\mathbb{E}[\text{cut}] = \sum_{(i,j) \in E} w_{ij} \cdot \frac{\arccos(X^{*}_{ij})}{\pi}
\geq \sum_{(i,j) \in E} w_{ij} \cdot \alpha_{\text{GW}} \cdot \frac{1 - X^{*}_{ij}}{2}
= \alpha_{\text{GW}} \cdot \frac{1}{4} \langle L, X^{*} \rangle
= \alpha_{\text{GW}} \cdot \text{SDP}_{\text{OPT}}
$$

The first inequality applies the key inequality to each edge. Since
$\text{SDP}_{\text{OPT}} \geq \text{OPT}$ (the SDP is a relaxation of the integer program):

$$
\mathbb{E}[\text{cut}] \geq \alpha_{\text{GW}} \cdot \text{OPT} \approx 0.878 \cdot \text{OPT} \qquad \square
$$

---

## 6. Unique Games Conjecture and Optimality

The **Unique Games Conjecture** (Khot, 2002) is a major open problem in computational
complexity. It asserts that for every $\varepsilon > 0$, there exists a finite alphabet size
$k$ such that it is NP-hard to distinguish between instances of the Unique Label Cover problem
that are $(1-\varepsilon)$-satisfiable versus at most $\varepsilon$-satisfiable.

**Theorem (Khot, Kindler, Mossel, O'Donnell, 2007):** Assuming the Unique Games Conjecture,
for every $\varepsilon > 0$, it is NP-hard to approximate Max-Cut to within a factor of
$\alpha_{\text{GW}} + \varepsilon$.

This means that — assuming UGC — the Goemans-Williamson algorithm is **optimal**: no
polynomial-time algorithm can achieve a better approximation ratio than $\alpha_{\text{GW}}$.
The GW algorithm is the best we can do in polynomial time (up to the truth of UGC).

---

## 7. Multiple Rounding Trials

A single hyperplane rounding is a random experiment. Running $k$ independent rounding trials
and taking the best cut improves the probability of finding a good solution.

### Analysis

Let $c^{*} = \alpha_{\text{GW}} \cdot \text{SDP}_{\text{OPT}}$. By Markov's inequality applied
to the complementary random variable, each trial independently has at least a constant
probability $p > 0$ of producing a cut of value $\geq c^{*}$. After $k$ independent trials:

$$
\Pr[\text{best cut} < c^{*}] \leq (1 - p)^k
$$

For the failure probability to be at most $\delta$:

$$
k \geq \frac{\ln(1/\delta)}{\ln(1/(1-p))} \approx \frac{\ln(1/\delta)}{p}
$$

### Practical Recommendation

With $k = 100$ trials, the probability of failing to find a cut at least as good as the
expected value is negligibly small. In practice, multiple trials almost always find a cut
very close to (or exceeding) $\text{SDP}_{\text{OPT}}$ on small to medium instances.

The rounding step is computationally cheap ($O(n)$ per trial) compared to the SDP solve
($O(n^{4.5})$), so running many trials adds negligible overhead.

---

## 8. One-Flip Local Search

After hyperplane rounding produces a partition, a simple **greedy local search** can improve the
cut value: move single vertices to the other side while doing so increases the cut.

### Algorithm

Given a partition $(S, \bar{S})$:

1. For each vertex $v \in V$, compute the **flip gain**:

$$
\Delta(v) = \sum_{u \in N(v)} w_{vu} \cdot \begin{cases} +1 & \text{if } u \text{ and } v \text{ are on the same side} \\ -1 & \text{if } u \text{ and } v \text{ are on opposite sides} \end{cases}
$$

   Equivalently, $\Delta(v) = w_{\text{same}}(v) - w_{\text{cut}}(v)$, where $w_{\text{same}}(v)$
   is the total weight of edges from $v$ to vertices on its side, and $w_{\text{cut}}(v)$ is the
   total weight of edges from $v$ to the other side.

2. If $\Delta(v) > 0$, flip $v$ (move it to the other side). The cut value increases by exactly
   $\Delta(v)$.

3. Repeat until no single vertex flip improves the cut (a **local optimum**).

### Analysis

**Convergence:** Each flip strictly increases the cut value. Since the cut value is bounded above
by the total edge weight $W = \sum_{(i,j) \in E} w_{ij}$, and each flip increases it by at
least $\min_{(i,j) \in E} w_{ij} > 0$ (for positive-weight graphs), the algorithm terminates
in at most $W / w_{\min}$ steps. For unit-weight graphs, this is at most $|E|$ flips.

**Cost per sweep:** Each full sweep over all vertices takes $O(|E|)$ time (each edge is examined
twice — once from each endpoint). In practice, convergence happens in a small number of sweeps
(typically 2–5 for sparse Gset instances).

**Improvement:** On standard benchmark instances (Gset), one-flip local search typically improves
the cut value by 1–3% over raw hyperplane rounding. This closes the gap between the theoretical
$\alpha_{\text{GW}} \approx 0.878$ guarantee and the best-known cut values. The improvement is
largest when the hyperplane rounding produces a partition that is "almost good" but has a few
misplaced vertices.

**Composability with multiple trials:** Since local search is cheap ($O(|E|)$ per convergence)
relative to the SDP solve, it is applied independently to each of the $k$ rounding trials. The
best locally-optimal partition across all trials is returned.

### Theoretical Guarantee

The local search alone (without the SDP) provides a $\frac{1}{2}$-approximation: start from any
partition and flip until stable. At a local optimum, every vertex $v$ satisfies
$w_{\text{cut}}(v) \geq w_{\text{same}}(v)$, so:

$$
\text{cut} = \frac{1}{2} \sum_{v \in V} w_{\text{cut}}(v) \geq \frac{1}{2} \sum_{v \in V} \frac{w_{\text{cut}}(v) + w_{\text{same}}(v)}{2} = \frac{1}{2} \sum_{v \in V} \frac{\deg_w(v)}{2} = \frac{W}{2} \geq \frac{\text{OPT}}{2}
$$

Combined with GW rounding, the local search can only improve the cut, so the $\alpha_{\text{GW}}$
guarantee is preserved: the result is at least $\max(\alpha_{\text{GW}} \cdot \text{SDP}_{\text{OPT}},\; \text{local optimum})$.

---

## 9. Worked Example: The 5-Cycle

Consider the 5-cycle $C_5$ with vertices $\{0, 1, 2, 3, 4\}$ and unit-weight edges
$\{(0,1), (1,2), (2,3), (3,4), (4,0)\}$.

### Exact Max-Cut

The maximum cut partitions vertices into $S$ and $\bar{S}$ to maximize crossing edges. For an
odd cycle $C_n$, the maximum cut has $n - 1$ edges (we cannot 2-color all edges of an odd cycle).

For $C_5$: $\text{OPT} = 4$. For example, $S = \{0, 2\}$, $\bar{S} = \{1, 3, 4\}$ cuts edges
$(0,1), (1,2), (2,3), (4,0)$ — four out of five edges.

### SDP Optimal Solution

The optimal SDP solution for $C_5$ places the unit vectors $\mathbf{v}_0, \ldots, \mathbf{v}_4$
as vertices of a regular "star" configuration in $\mathbb{R}^5$, with equal angles between
adjacent vectors:

$$
\langle \mathbf{v}_i, \mathbf{v}_{i+1} \rangle = \cos\left(\frac{4\pi}{5}\right) = \cos(144°) \approx -0.8090
$$

for each edge $(i, i+1 \mod 5)$.

**SDP optimal value:**

Using $\langle L, X \rangle = \sum_{(i,j) \in E} w_{ij}(X_{ii} + X_{jj} - 2X_{ij}) = 5(2 - 2\cos 144°)$:

$$
\text{SDP}_{\text{OPT}} = \frac{1}{4} \langle L, X^{*} \rangle = \frac{5}{2}(1 - \cos 144°) = \frac{5}{2}(1 + 0.8090) \approx 4.523
$$

### GW Guarantee

$$
\mathbb{E}[\text{cut}] \geq \alpha_{\text{GW}} \cdot \text{SDP}_{\text{OPT}} \approx 0.878 \times 4.523 \approx 3.97
$$

Since the expected GW cut is $\approx 3.97$ and the maximum cut is $4$, the algorithm achieves
nearly optimal performance on this instance.

We can verify directly:

$$
\mathbb{E}[\text{cut}] = 5 \times \frac{\arccos(\cos 144°)}{\pi} = 5 \times \frac{4\pi/5}{\pi} = 5 \times \frac{4}{5} = 4.0
$$

Remarkably, for the 5-cycle the expected GW cut is **exactly** $4 = \text{OPT}$. This happens
because $\arccos(\cos 144°)/\pi = 4/5$ exactly, and $5 \times 4/5 = 4$.

### Approximation Ratio on $C_5$

$$
\text{ratio} = \frac{\text{OPT}}{\text{SDP}_{\text{OPT}}} = \frac{4}{4.523} \approx 0.884
$$

This is slightly above $\alpha_{\text{GW}} \approx 0.878$, confirming that the SDP relaxation
is not too loose on this instance.

---

## 10. Implementation Checklist

| Step | Module | Function |
|------|--------|----------|
| 1. Build graph Laplacian $L$ | `graph.rs` | `Graph::laplacian()` |
| 2. Formulate SDP: $C = -\frac{1}{4}L$, constraints $X_{ii} = 1$ | `sdp/mod.rs` | `SdpProblem::maxcut_relaxation()` |
| 3. Solve SDP via IPM | `sdp/solver.rs` | `Solver::solve()` |
| 4. Extract vectors via Cholesky: $X^{*} = V^\top V$ | `maxcut/rounding.rs` | `extract_vectors()` |
| 5. Random hyperplane rounding ($k$ trials) | `maxcut/rounding.rs` | `multi_round()` |
| 6. One-flip local search (per trial) | `maxcut/rounding.rs` | `one_flip_local_search()` |
| 7. Return best partition and approximation ratio | `maxcut/mod.rs` | `gw_maxcut()` |
