# Storage and error guide

The phrase “memory saved” can mean several different things. `qhmat` exposes enough counters to avoid comparing unlike quantities.

## Which size are you measuring?

| Quantity | Access | Answers | Excludes |
| --- | --- | --- | --- |
| Original NumPy payload | `qhmat.memory_bytes(A)` or `A.nbytes` | How many bytes are visible in this array? | Python object/backing allocation beyond a view |
| Dense FP64 baseline | `h.storage_report().dense_fp64_bytes` | What would an $n\times n$ FP64 array's payload cost? | Python/allocator overhead |
| Logical compressed storage | `h.logical_bits` | How many theoretical bits for codes, scales, labels, and FP64 diagonal values? | Tree headers, word padding, workspace |
| Native vector payload | `h.allocated_bytes` | How many bytes are used by native vector elements and offsets? | Unused vector capacity and C++ objects |
| C++ resident estimate | `h.resident_bytes` or `qhmat.memory_bytes(h)` | What do owned objects and vector capacities approximately occupy? | Allocator bookkeeping, Python wrapper, CUDA copies, source $A$, SVD workspace |

```python
report = h.storage_report()
print(report.source_array_bytes, report.dense_fp64_bytes)
print(report.logical_bits, report.logical_bytes)
print(report.allocated_bytes, report.resident_bytes)
print(report.logical_compression, report.resident_compression)
```

`logical_compression` divides the FP64 baseline by the logical bits converted to bytes. `resident_compression` divides it by the C++ resident estimate. They can differ substantially for small blocks. A ratio below $1$ means expansion, not compression. Neither ratio includes the source NumPy array if you keep it alive.

!!! warning "Resident estimate is not RSS"
    The estimate uses C++ `sizeof` and vector **capacity**, but cannot know an allocator's hidden metadata, memory fragmentation, other Python objects, or temporary peak usage. Measuring the process RSS before/after construction is also noisy because both the source matrix and SVD temporaries may still be live.

## Error quantities that should not be conflated

Suppose $A$ is a discretized kernel or integral operator. The following differences answer different questions:

1. **Discretization error:** difference between the continuum operator and assembled $A$. Not measured by `qhmat`.
2. **SVD truncation error:** $\eta_k=\|A_k-T_k\|_{\max}$ for each non-diagonal block.
3. **Factor quantization surrogate:** $Q_k=\sum_j\sigma_{k,j}c(b_{k,j})$.
4. **Representation error:** $\|A-\widetilde A\|_{\max}$; bounded in exact arithmetic by the maximum of $\eta_k+Q_k$ over blocks.
5. **Application error:** difference in $AX$ versus `h @ X`; depends on $X$ and execution arithmetic.

To test the representation, compare a small reference matrix to `h.to_dense()` and inspect `h.reconstruction_bound`. To test an application, compare `A @ X` against `h @ X` with the same right-hand sides, but do not call the representation bound an application bound.

## A useful diagnostic loop

```python
for block in h.blocks:
    if block.dense:
        continue
    print(
        block.rectangle.admissible,
        block.rank,
        block.truncation_error,
        block.quantization_bound,
        block.logical_bits,
    )
```

If stored bytes exceed the dense baseline, first inspect the **non-diagonal inadmissible** blocks and their ranks, then the number of small blocks and precision labels. Increasing tolerance may reduce rank and/or assigned bit widths, but those effects should be reported separately in a study. A tight tolerance can make construction infeasible rather than trigger a hidden dense fallback.
