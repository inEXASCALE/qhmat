# Python API

The public package is `qhmat`. Start with `HMatrix.from_dense`; use the lower-level objects only when you need explicit allocation or partition control. The Python class and function descriptions below are extracted from their source docstrings, with short usage examples immediately after each group.

```python
import numpy as np
import qhmat

points = np.arange(8, dtype=float)[:, None]
matrix = 1.0 / (1.0 + np.abs(points - points.T))
```

## Construct and apply: `HMatrix`

::: qhmat.matrix.HMatrix
    options:
      members:
        - from_dense
        - from_partition
        - matmul
        - to_dense
        - storage_report
        - save
        - load
        - reconstruction_bound
        - logical_bits
        - allocated_bytes
        - resident_bytes

### Construct from geometry

`HMatrix.from_dense(matrix, points, *, tolerance, ...)` forms a median-split cluster tree and a disjoint block partition. `points` must have one row per source-matrix index. Supply `support_lower` and `support_upper` for extended BEM panels.

```python
h = qhmat.HMatrix.from_dense(matrix, points, tolerance=0.25,
                             leaf_size=2, verbose=1)
print(h.shape, len(h.blocks))
```

`verbose=1` prints sizes only on success. `max_rank` limits retained rank; `bit_widths` controls the allowed integer menu, not a requested uniform width. Construction raises `InfeasibleError` if any non-diagonal block cannot meet its budget.

### Reuse a partition

`HMatrix.from_partition` accepts blocks in **permuted index coordinates**. The row and column permutation must be the same; the rectangles must cover the matrix exactly once.

```python
part = qhmat.spatial_partition(points, points, points, 2, 1.0)
h2 = qhmat.HMatrix.from_partition(matrix, part.permutation, part.blocks,
                                  tolerance=0.25)
```

### Multiply or reconstruct

`matmul(x, device="cpu")` accepts a vector of shape `(n,)` or a multi-right-hand-side array of shape `(n, p)`. `@` calls the CPU path. `to_dense()` allocates a complete FP64 matrix for inspection.

```python
x = np.ones((8, 2))
print((h @ x).shape)
print(np.max(np.abs(matrix - h.to_dense())))
```

CUDA is selected only by `h.matmul(x, device="cuda")` in a CUDA-enabled build. Neither the CPU nor CUDA call automatically returns an error bound for the application.

### Save and load

`save(path)` writes a versioned checksummed file and refuses to overwrite. `load(path)` validates it before returning a new `HMatrix`.

```python
h.save("operator.qhmat")
loaded = qhmat.HMatrix.load("operator.qhmat")
print(loaded.shape)
```

The checksum is for accidental corruption detection, not malicious-file authentication.

## Memory accounting

::: qhmat.matrix.memory_bytes

Quick example: the same function accepts either a NumPy array or a hierarchical matrix. The meanings differ deliberately: NumPy returns `nbytes`; an H-matrix returns an estimated owned C++ size.

```python
print(qhmat.memory_bytes(matrix), qhmat.memory_bytes(h))
```

::: qhmat.matrix.StorageReport
    options:
      members:
        - logical_bytes
        - logical_compression
        - resident_compression

Quick example: `storage_report()` compares logical storage and estimated C++ resident bytes with a dense FP64 payload. `source_array_bytes` is the input array's visible payload, if known; it becomes `None` after `load`.

```python
report = h.storage_report()
print(report.dense_fp64_bytes, report.logical_bits, report.resident_bytes)
print(report.logical_compression, report.resident_compression)
```

`h.logical_bits` counts factor code bits, two FP64 scales per retained component, precision labels, and FP64 diagonal entries. `h.allocated_bytes` counts native vector payloads. `h.resident_bytes` adds C++ objects and vector capacity. None includes all construction workspace or process RSS. See [Storage and error guide](storage.md).

## Feasibility and allocation

::: qhmat.matrix.InfeasibleError

Quick example:

```python
try:
    qhmat.HMatrix.from_dense(matrix, points, tolerance=1e-12,
                             leaf_size=2, bit_widths=(4, 8))
except qhmat.InfeasibleError as exc:
    print("cannot meet the strict budget:", exc)
```

### `coefficient(bits)`

Returns $c(b)=1/q_b+1/(4q_b^2)$ for $q_b=2^{b-1}-1$. It decreases as bit width increases.

```python
print(qhmat.coefficient(8), qhmat.coefficient(16))
```

### `allocate_bits(singular_values, bit_widths, budget, rows, cols)`

Choose a width per retained component under $\sum_j\sigma_jc(b_j)\le B^{\mathrm q}$. Singular values must be nonnegative and sorted descending. `payload_bits` counts $(m+n)\sum_j b_j$; it excludes scales and labels.

```python
allocation = qhmat.allocate_bits(np.array([2.0, 1.0]), [4, 8, 16],
                                 budget=0.1, rows=8, cols=8)
print(allocation.feasible, allocation.bits, allocation.bound)
print(allocation.payload_bits)
```

The returned `Allocation` has `bits`, `bound`, `payload_bits`, and `feasible`. A feasible assignment is **not** a claim of globally minimum H-matrix memory, because block ranks and metadata are outside this local payload objective.

## Partitions and block metadata

### `spatial_partition(points, support_lower, support_upper, leaf_size, admissibility)`

All three coordinate arrays have shape `(n, d)`. The support bounds must contain their corresponding point coordinates. For point kernels, the bounds may equal the points; for BEM, use complete panel bounding boxes.

```python
part = qhmat.spatial_partition(points, points, points, 2, 1.0)
print(part.permutation, len(part.blocks))
```

`Partition.permutation` maps clustered positions to source indices; `Partition.blocks` is a disjoint rectangular cover. Each `Rectangle(row0, row1, col0, col1, admissible=False)` uses half-open clustered-coordinate ranges.

```python
rect = qhmat.Rectangle(0, 2, 2, 4, True)
print(rect.row0, rect.row1, rect.admissible)
```

`BlockInfo` is available from `h.blocks`. It reports rectangle, dense flag, retained rank, measured truncation residual, quantization surrogate, and per-block storage counters. `rectangle.admissible` describes **geometry**; `block.dense` describes **storage**.

```python
for block in h.blocks[:3]:
    print(block.rectangle.admissible, block.dense,
          block.rank, block.quantization_bound)
```

## Low-level `NativeHMatrix`

`NativeHMatrix(order, permutation)` is the C++ builder used by the high-level API. Add a disjoint complete partition, call `finalize()`, then multiply, reconstruct, inspect, or save. Only diagonal blocks may be added with `add_dense`; `add_low_rank` takes SVD vectors `u`, `v`, singular values, a width per component, menu size, and a measured truncation error. Balanced packing occurs internally.

```python
native = qhmat.NativeHMatrix(2, [0, 1])
native.add_dense(qhmat.Rectangle(0, 1, 0, 1), np.array([[2.0]]))
native.add_dense(qhmat.Rectangle(1, 2, 1, 2), np.array([[3.0]]))
for row, col in ((0, 1), (1, 0)):
    native.add_low_rank(qhmat.Rectangle(row, row + 1, col, col + 1),
                        np.ones((1, 1)), np.array([1.0]),
                        np.ones((1, 1)), [8], 1, 0.0)
native.finalize()
print(native.matmul(np.ones(2)))
```

Its read-only properties include `order`, `blocks`, `logical_bits`, `allocated_bytes`, `resident_bytes`, and `reconstruction_bound`. `reconstruct()`, `save(path)`, and `NativeHMatrix.load(path)` mirror the high-level operations. Most users should prefer `HMatrix.from_dense` or `from_partition` because they enforce the rank and budget workflow.

## Build information

`qhmat.cuda_built` is a boolean from the compiled extension; it does **not** test GPU availability. `qhmat.__version__` is the Python package version.

```python
print(qhmat.__version__, qhmat.cuda_built)
```
