# Quickstart

This walkthrough keeps the dense reference matrix available so the representation error can be checked. The reference is useful for validation but still occupies memory until the caller releases it.

## 1. Construct a smooth kernel matrix

Choose points $x_i=i/63$ and the regularized reciprocal kernel

$$
A_{ij} = \frac{1}{|x_i-x_j|+0.4}.
$$

```python
import numpy as np
import qhmat

points = np.linspace(0.0, 1.0, 64)[:, None]
matrix = 1.0 / (np.abs(points - points.T) + 0.4)

h = qhmat.HMatrix.from_dense(
    matrix,
    points,
    tolerance=0.25,
    leaf_size=8,
    verbose=1,
)
```

`tolerance` is **block relative**: each non-diagonal block $A_k$ gets a max-norm budget $B_k=0.25\|A_k\|_{\max}$. It is not a single global relative error imposed on the complete matrix. `leaf_size` controls terminal cluster size, not a maximum rank.

## 2. Apply it and check the representation

```python
x = np.ones((64, 3))
y = h @ x

measured = np.max(np.abs(matrix - h.to_dense()))
print("measured max-norm reconstruction difference:", measured)
print("exact-arithmetic representation bound:", h.reconstruction_bound)
print("max application difference:", np.max(np.abs(y - matrix @ x)))
```

`h.to_dense()` constructs a new dense FP64 array and is intended for inspection or testing, not memory-efficient application. The last printed quantity is an **application difference**; do not compare it directly against `reconstruction_bound`, which bounds individual matrix entries in exact arithmetic.

## 3. Compare storage honestly

```python
report = h.storage_report()
print("input array bytes:", qhmat.memory_bytes(matrix))
print("theoretical bits:", report.logical_bits)
print("C++ resident estimate:", report.resident_bytes)
print("resident compression vs FP64:", report.resident_compression)
```

For small matrices, metadata and vector capacities can make the C++ resident estimate much larger than the theoretical packed size. A compression ratio below one means the representation is **larger** than the FP64 dense baseline under that accounting. See [Storage and error guide](storage.md) for the four different memory measures.

## 4. Save and reload

```python
h.save("operator.qhmat")
loaded = qhmat.HMatrix.load("operator.qhmat")
print(np.max(np.abs(loaded @ x - y)))
```

`save` refuses to replace an existing file. The checksum detects accidental corruption; it is not an authenticity or security guarantee. A loaded matrix has no record of the original NumPy array's byte count.

!!! tip "If construction is infeasible"
    Catch `qhmat.InfeasibleError`, then examine the requested tolerance, available bit widths, and `max_rank`. The strict representation never hides an infeasible non-diagonal block behind a dense fallback. A tighter tolerance usually requires more retained rank or higher precision, and may eliminate a storage advantage.
