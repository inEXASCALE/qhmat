# H-matrix application benchmark

The revised policy uses `configs/application_offdiagonal_a100.json`. It exports
quantized factors for all non-diagonal leaves, including inadmissible blocks,
and retains only diagonal working blocks densely. Infeasible configurations
are rejected, never exported with non-diagonal dense substitutions. Rebuild
the binary for its 24/32-bit software-decoder support; computations remain
FP32. The commands and dense-fallback protocol below refer to the separately
retained `application_a100.json` archive unless the revised config is supplied.

This benchmark measures **Y = A X**, not reconstruction of low-rank blocks and
not an H-matrix/H-matrix product. Every low-rank leaf applies `U (V.T X)` to all
RHS columns together. The existing `qmat_cuda` reconstruction microbenchmark
and published `results_a100` remain unchanged. No application speedup is claimed
until the new CUDA measurements have been run on the target node.

## Three stages on the A100 node

Run from the repository root, using the Python environment with
`experiments/requirements.txt` installed. Check `nvidia-smi` on the compute node,
not the login/front node; retain the MIG configuration and competing-load context.

```bash
# Stage 1a: needed only when the source NPZ/leaf files are absent or outdated.
python experiments/prepare_data.py --config experiments/configs/a100.json

# Stage 1b: fixed partitions/ranks, packed factors, RHS and FP64 references.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/prepare_application.py \
  --config experiments/configs/application_a100.json

# Build with the compatible CUDA/host compiler pair already used on your node.
cmake -S experiments/cuda -B experiments/cuda/build-apply \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build experiments/cuda/build-apply -j
ctest --test-dir experiments/cuda/build-apply --output-on-failure

# Stage 2: CUDA measurements and validation, no plotting.
python experiments/run_application.py \
  --config experiments/configs/application_a100.json \
  --binary experiments/cuda/build-apply/qmat_apply

# Stage 3: plot the saved measurements, without rerunning CUDA.
python experiments/plot_application.py --config experiments/configs/application_a100.json
```

If CUDA configuration fails with the previous `rsqrt` header mismatch, use the
compatible toolkit/host/sysroot environment from the successful native run.
This benchmark does not fix incompatible system headers; do not add
`--allow-unsupported-compiler` as a substitute for a compatible toolchain.

The default suite has 10 existing BEM/point-kernel matrices, orders 1024--2048
(1536 for surface BEM), tolerances 0.005/0.01/0.05, and RHS counts 1/8/32/128.
It records 3,600 method/trial rows: 10 matrices x 3 tolerances x 4 RHS counts x
6 methods x 5 timing trials. Construction and FP64 reference formation can take
substantial CPU time; SVDs are cached across tolerances. `preparation_and_reference_ms`
includes reference/validation work and must NOT be presented as pure construction
time or used to calculate a construction break-even point.

New paths are `data_application_a100`, `results_application_a100`, and
`figures_application_a100`, all under `experiments/`. Inputs are versioned binary
files plus a manifest. Input/source/binary hashes, timing order and device versions
are retained. Successful or failed runs are not silently overwritten: use a new
configuration with a new results directory for another run. Input paths in the
application manifest are relative to the manifest directory.

## Smoke and correctness checks first

```bash
python experiments/prepare_application.py --smoke
experiments/cuda/build-apply/qmat_apply_host \
  experiments/data_application_a100/smoke/smoke_tol0.05.bin
experiments/cuda/build-apply/qmat_apply \
  experiments/data_application_a100/smoke/smoke_tol0.05.bin \
  8 2 5 2 17 /tmp/qmat-apply-smoke.csv
compute-sanitizer --tool memcheck experiments/cuda/build-apply/qmat_apply \
  experiments/data_application_a100/smoke/smoke_tol0.05.bin \
  1 1 2 1 17 /tmp/qmat-apply-memcheck.csv
python -m unittest discover -s experiments/tests -v
# On the GPU node, enable fixture execution through all six CUDA paths:
QMAT_APPLY_BINARY="$PWD/experiments/cuda/build-apply/qmat_apply" \
  python -m unittest discover -s experiments/tests -p test_application.py -v
```

Alternatively `run_application.py --smoke` and `plot_application.py --smoke`
run the complete pipeline on the small synthetic fixture. Those files live in
separate `smoke/` subdirectories and are not publication results. The CPU-only
build (`-DQMAT_ENABLE_CUDA=OFF`) includes `qmat_apply_host` for input/reference checks.

## Compared implementations

| Method | Low-rank operation | Per-application decoding |
|---|---|---|
| `dense_fp32` | Dense cuBLAS SGEMM on original-order A | None |
| `h_fp32_batched` | Two shape-bucketed cuBLAS batched GEMMs | None |
| `adaptive_unpacked_batched` | Same two batched GEMMs | Full factor decode included |
| `adaptive_predecoded_batched` | Same two batched GEMMs | Excluded: diagnostic only |
| `h_fp32_fused` | Two tiled SIMT kernels | None: matched fused control |
| `adaptive_packed_fused` | Same tiled kernels with packed-weight loads | Tile-local decode included |

All paths use FP32 arithmetic with TF32 disabled. The fused kernels do **not**
claim INT4/INT8/INT12/INT16 Tensor Core execution. They decode only the current
factor tile into shared memory, preserve per-column scales and bit assignments,
and do not allocate a full decoded factor image as a working requirement.
Unlike a channel-by-channel implementation, different stored bit widths can be
loaded in one tile because the arithmetic is the same FP32 type. This is a
weight-compressed application experiment, not a native integer GEMM experiment.

Buckets use exact `(m,n,rank)` shapes, including rank-zero dense blocks; no rank
padding or changed truncation is hidden. Every H method uses the same buckets,
same permutation gather, same dense-block batched SGEMM, and same deterministic
row reduction. The row reduction includes all overlapping output contributions,
uses one writer per output entry, and scatters back to original matrix ordering.
No atomics, dense reconstructed blocks, or CPU per-RHS GEMV loop are used.

## Fairness and timing scope

- Allocation and SVD decisions use the existing reconstruction allocator and
  fallback checks. They are frozen before GPU timing; this is not an allocator
  redesigned for RHS-specific application error or latency.
- Unquantized and adaptive H variants have identical retained ranks and actual
  dense/fallback partition. The ordinary H reference uses FP32 balanced factors.
  Dense/fallback **working** blocks are FP32 in both variants. This differs from
  the paper's FP64 dense-leaf *storage-only* controls, and their conversion error
  is included in the reference comparisons here.
- Each timing trial randomizes all six method runs. CUDA events time complete
  resident-device applications after warmup, including required gather, decode,
  GEMMs, intermediate writes, output reduction and inverse permutation.
- SVD, bit allocation, packing, host/device transfers, pointer-table preparation,
  cuBLAS initialization, and FP64 checking are outside application timing.
  The predecoded variant is explicitly diagnostic and stores floating factors.
- Speedups use the dense and batched-H medians from the **same trial**. Five
  randomized trials in one process are not five independent machine sessions.
  Plot bands are observed min/max of paired trial speedups, not confidence intervals.
- `factor_bytes`, `plan_bytes`, and `workspace_bytes` are logical per-method device
  requirements, not measured peak allocation. They exclude input X, cuBLAS-private
  workspace, allocator granularity and co-resident comparison variants. The runner
  keeps all variants resident to alternate them; `plan_prepare_ms` is the combined
  setup/transfer/decode time for that comparison plan, not a per-method setup cost.
- Matrices and tolerances are never pooled into one speedup curve. Report latency,
  error and both speedup denominators, including slow and all-dense/fallback cases.

## Numerical checks and theoretical scope

The input records original-order X (FP32), original FP64 A X, the FP64 product
of rounded dense FP32 A with X, and FP64 applications of ordinary and decoded
quantized FP32 factors. Thus errors include source truncation/quantization and
dense conversion, rather than comparing only against a saved factor product.

The GPU checks every entry against its representation-specific FP64 application.
For a low-rank leaf with dimensions m x n and rank r, an absolute arithmetic
envelope uses

```
[(1 + gamma_n(u)) (1 + gamma_r(u)) - 1] |U| (|V|^T |X|).
```

Dense leaves use `gamma_n(u) |D| |X|`. Contributions are propagated through the
fixed row summation with its own gamma factor. A conservative FP64 reference
allowance is included. These standard FP32 bounds assume finite normal-range
arithmetic, round-to-nearest, no overflow/underflow, and TF32 disabled.
They are **new application checks**, not a claim that the manuscript's channel
reconstruction arithmetic theorem directly certifies two-stage application.

Preparation also checks the representation error propagation
`|(A_leaf - Ahat_leaf) X_J| <= delta_leaf * sum(abs(X_J), axis=0)`, with delta
computed against the actual rounded factors/dense blocks. These are measured
FP64 residual-based diagnostics, not interval-arithmetic enclosures. The allocator
still targets the manuscript's block-relative reconstruction budget, so neither
equal tolerance nor identical rank implies equal output accuracy across methods.

For the paper, report a storage/speed tradeoff only after the complete A100 run.
Speedup over dense alone is evidence for the low-rank structure, not automatically
for adaptive bits. `adaptive_packed_fused` versus `h_fp32_fused` isolates weight
compression under the same tiled kernel; versus `h_fp32_batched` tests whether it
also beats the library-based H baseline. Retain the reconstruction diagnostics in
the supplement. Native integer channels, quantized RHS/intermediates, per-method
construction amortization and H-matrix/H-matrix multiplication remain separate
future extensions, not implemented or claimed by this benchmark.
