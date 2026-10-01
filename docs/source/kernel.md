# Kernel matrices

Kernel matrices appear in interpolation, covariance models, particle methods, and discretized nonlocal operators. A point kernel evaluates an interaction $K(x_i,x_j)$ for each pair of sites; a dense matrix is easy to form but requires $O(n^2)$ storage.

## Gaussian example

For points $x_i\in\mathbb R^2$ on a smooth curve, this example uses

$$
A_{ij}=\exp\!\left(-\frac{\|x_i-x_j\|_2^2}{2\ell^2}\right),
\qquad \ell=0.25,
\qquad Y=AX.
$$

Far-separated point groups tend to have smooth interactions and are natural candidates for factorization. This example is not a theorem that every chosen block is low rank: ranks and storage depend on geometry, partition size, and tolerance.

```python title="docs/examples/kernel.py"
--8<-- "docs/examples/kernel.py"
```

Run from the repository root:

```bash
python docs/examples/kernel.py
```

The script prints both $\|A-\widetilde A\|_{\max}$ and the package's representation bound. The application comparison $\|AX-\widetilde A X\|_{\max}$ includes the chosen right-hand sides and floating-point computation, so it should be interpreted separately.

!!! info "What to look for"
    Check `h.storage_report().resident_compression`, not only `logical_compression`. At $n=48$, fixed per-block metadata can outweigh the integer payload even when the theoretical format is compact. Larger $n$ does not automatically fix this; the partition and retained ranks matter.

## Sweep a tolerance

For an application study, keep the matrix and geometry fixed while varying tolerance. This separates compression changes from changes to the source problem:

```python
for tol in (0.1, 0.2, 0.3):
    try:
        candidate = qhmat.HMatrix.from_dense(
            matrix, points, tolerance=tol, leaf_size=8
        )
    except qhmat.InfeasibleError:
        print(tol, "infeasible")
        continue
    report = candidate.storage_report()
    error = np.max(np.abs(matrix - candidate.to_dense()))
    print(tol, error, candidate.reconstruction_bound,
          report.logical_compression, report.resident_compression)
```

Do not equate a lower logical bit count with a faster `@` operation. Timings depend on block traversal, rank, right-hand-side count, and backend implementation.
