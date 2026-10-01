# How it works

The representation combines a geometric **block partition** with algebraic **factor compression** and componentwise **precision allocation**. Those are separate decisions: a block can be geometrically inadmissible yet still be stored as quantized factors.

## From geometry to blocks

The row and column indices use the same stable median-split cluster tree. At each split, the coordinate with the largest spread is selected. Clusters stop growing below `leaf_size` indices. Bounding boxes determine geometric admissibility from separation and cluster diameter; for panel methods, supply support bounds that cover the full panels, not just their collocation points.

The accepted blocks form a disjoint partition of the matrix. `qhmat` distinguishes:

| Block kind | Geometry | Storage in the proposed representation |
| --- | --- | --- |
| Diagonal | Same row and column index cluster | Dense FP64 |
| Admissible non-diagonal | Sufficient box separation | Quantized SVD factors |
| Inadmissible non-diagonal | Near field at terminal clusters | Quantized SVD factors |

This differs from a conventional policy that keeps all near-field blocks dense. A geometric admissibility flag is **not** a promise of low rank, nor does inadmissibility prohibit factorization. At tight tolerances the near-field rank can equal the smaller block dimension; the format may then cost more than dense storage.

## One block, two approximations

For a non-diagonal block $A_k\in\mathbb{R}^{m_k\times n_k}$, construction first chooses a retained SVD:

$$
A_k \approx T_k=\sum_{j=1}^{r_k}\sigma_{k,j}u_{k,j}v_{k,j}^{\mathsf T},
\qquad \eta_k=\|A_k-T_k\|_{\max}.
$$

The factors are balanced as $P_k=U_k\Sigma_k^{1/2}$ and $Q_k=V_k\Sigma_k^{1/2}$. Each retained column gets a selectable signed-integer width $b_{k,j}$ and its own pair of scales. The stored block reconstructs from those quantized columns; the singular values are used for allocation and bounding, not stored as a separate dense matrix.

`rank_tolerance` limits the relative Frobenius tail, while `truncation_fraction` limits the portion of the block max-norm budget spent on truncation. The implementation checks the measured max-norm residual before allocating bits. `max_rank` can cap the retained rank; if the cap makes the truncation allowance impossible, construction fails.

## What the allocator controls

For $q_b=2^{b-1}-1$, define

$$
c(b)=\frac{1}{q_b}+\frac{1}{4q_b^2},\qquad
Q_k=\sum_{j=1}^{r_k}\sigma_{k,j}c(b_{k,j}).
$$

The allocator chooses widths from the supplied discrete menu to fit its budget $B_k^{\mathrm q}$. The package's `allocate_bits` returns the chosen widths, the surrogate value, feasibility, and **factor payload only**: $(m_k+n_k)\sum_j b_{k,j}$ bits. Scales and precision labels are counted later by the complete matrix storage model.

For the high-level constructor, $B_k=\varepsilon\|A_k\|_{\max}$, and the quantization allowance is what remains after the measured truncation residual and `arithmetic_reserve` are subtracted. The reserve is bookkeeping for potential arithmetic effects; by itself it does **not** certify downstream FP32/CUDA computation.

!!! info "Feasibility is not optimal storage"
    A feasible bit assignment meets the surrogate constraint. It does not imply minimum total H-matrix storage once metadata, block ranks, and C++ allocation overhead are considered. Nor does it mean that the bit assignment minimizes measured error.

## Why the global matrix bound is simple

Each matrix entry belongs to exactly one accepted block. In exact arithmetic, the maximum over all entries is therefore the maximum over block errors:

$$
\|A-\widetilde A\|_{\max}
\le \max_{k\in\mathcal L_{\mathrm{nd}}}(\eta_k+Q_k).
$$

`h.reconstruction_bound` exposes this **representation** bound. Floating-point reconstruction or application adds arithmetic effects not covered by that property. In particular, if $y=Ax$ and $\widetilde y=\widetilde A x$, the representation contribution obeys

$$
|(y-\widetilde y)_i|
\le \sum_j |A_{ij}-\widetilde A_{ij}|\,|x_j|
\le \|A-\widetilde A\|_{\max}\,\|x\|_1,
$$

before accounting for floating-point multiplication. Thus a small matrix max-norm error is not automatically the same numerical value as the vector-output error. Discretization or quadrature error in how $A$ was assembled is a separate question again.

## When to expect a benefit

The method is most promising when many non-diagonal blocks have ranks well below their dimensions and the precision menu can assign narrow widths to weaker singular components. It is less attractive when near-field ranks are high, the partition creates many tiny blocks, or C++ metadata dominates their payload. The source matrix and SVD workspace also exist during construction, so this package is currently a **post-assembly compressor**, not a way to avoid assembling $A$ in the first place.

For performance, benchmark the exact operation of interest with the relevant right-hand-side count. The current CUDA path uploads data per call and no general speedup over optimized dense GEMM or a floating-factor H-matrix is claimed.
