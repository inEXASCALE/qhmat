# Quantized H-Matrix Experiments

The experiment harness is split into three reproducible stages.

1. Prepare data files.
2. Run numerical and GPU experiments.
3. Plot figures from saved CSV files.

The current storage policy keeps **only diagonal leaf blocks dense**. Every
non-diagonal leaf, including geometrically inadmissible near-field blocks, is
represented by quantized factors. The geometric partition is unchanged.

## Revised diagonal-only experiment (run this)

```bash
python experiments/prepare_data.py --config experiments/configs/offdiagonal_a100.json
python experiments/run_experiments.py --config experiments/configs/offdiagonal_a100.json --skip-gpu
python experiments/plot_offdiagonal.py --config experiments/configs/offdiagonal_a100.json
```

These stages use `data_offdiagonal_a100`, `results_offdiagonal_a100`, and
`figures_offdiagonal_a100`, respectively. They do not overwrite the archived
`*_a100` outputs. The source matrices and geometric partitions are the same,
but new preparation is needed to record the revised configuration. No CUDA
build is required for this CPU storage/certificate study. To smoke-test the
pipeline, substitute `offdiagonal_smoke.json` in all three commands.
Dense source matrices and reference arrays remain available as benchmark
inputs; they are not part of the measured compressed H-matrix representation.

- Total relative tolerances span `1e-6` to `1e-1`. These are not the SVD tolerance.
- `rank_tol=0.1` is an upper relative Frobenius tail target. Rank selection also
  enforces a tail target of `truncation_budget_fraction * block_budget` (25%).
  The actual max-norm residual, not this upper allowance, is subtracted before
  bit allocation. `max_rank=0` permits ranks up to the smaller block dimension.
- The hierarchical integer menu is `[4,8,12,16,24,32]`, scales are FP64, and
  reconstruction is FP64 with a 2% arithmetic reserve. Local allocator studies
  retain their original four-width menu. These wider stored integers do not
  imply native GPU arithmetic support.
- Failed non-diagonal candidates are **never replaced by dense blocks**.
  `feasible=0` configurations have blank global error, storage, and compression
  fields. Leaf rows record `stored=0` and `failure_reason`; inspect these before
  changing rank limits, bit menus, or arithmetic settings. There is no claim
  that every non-diagonal block has small rank or beats dense storage.
- Summary CSV columns distinguish `diagonal`, `admissible`, and
  `offdiagonal_inadmissible` storage (including scales/maps). Same-partition
  floating controls retain the same ranks and diagonal FP64 blocks. Plotting
  emits per-case figures plus `rank_diagnostics.csv`, with infeasible points
  missing rather than zero. Font and legend settings use the shared `plot_style`.
- Prepared leaf JSON's historical `low_rank` flag denotes **geometric
  admissibility**, not the revised storage decision. Result leaf CSV's
  `low_rank`, `stored`, `admissible`, and `block_kind` fields make that decision
  explicit. A diagonal block means identical row/column index clusters after
  the common permutation; a partly diagonal-crossing rectangular block is
  rejected instead of misclassified.

The manuscript's hierarchical tables and figures now use the uploaded
diagonal-only results at commit `4c4e48f`: 1,320 feasible configurations,
zero certificate violations, and no off-diagonal dense fallback. The old
`results_a100` hierarchy is retained only as a historical archive. Local
allocator and native validation experiments retain their separate archives.
Rank selection, integer menu, and arithmetic also differ from that archive;
do not attribute all changes in compression solely to near-field factorization.
The existing `a100.json`/`default.json` explicitly select `admissible_only` for
reproducing that archive; do not use them for the new storage policy.

Optional CUDA application validation uses the same strict decisions:

```bash
python experiments/prepare_application.py --config experiments/configs/application_offdiagonal_a100.json
python experiments/run_application.py --config experiments/configs/application_offdiagonal_a100.json --binary experiments/cuda/build-apply/qmat_apply
python experiments/plot_application.py --config experiments/configs/application_offdiagonal_a100.json
```

Rebuild `qmat_apply` from this revision first using your already working CUDA
toolchain. Its software decoder now supports 24/32-bit storage safely;
arithmetic stays FP32. Export refuses an infeasible non-diagonal representation.
These application directories are also separate from the archived run.

The remaining instructions document the original, archived experiment suite.

## Setup

```bash
python3 -m venv experiments/.venv
source experiments/.venv/bin/activate
pip install -r experiments/requirements.txt
```

GPU performance experiments require a CUDA-enabled PyTorch installation. The
CPU certificate experiments only require NumPy. Plotting requires Matplotlib.

## Stage 1: Data

```bash
python experiments/prepare_data.py --config experiments/configs/default.json
```

This writes synthetic low-rank test blocks and standard H-matrix leaf-partition
test matrices under `experiments/data/`.

It also generates 2D/3D Euclidean `1/r` and `log(r)` point-kernel matrices and
constant-panel collocation BEM matrices on polygonal curves and box surfaces.
See [MATRIX_SUITES.md](MATRIX_SUITES.md) for precise matrix definitions,
singular self-term treatment, experiment sizes, and comparison protocol.

## Stage 2: Results

```bash
python experiments/run_experiments.py --config experiments/configs/default.json
```

This writes CSV files under `experiments/results/`:

- `local_certificate.csv` for local certificate and allocator comparisons
- `hmatrix_certificate.csv` for global H-matrix certificate, storage, fallback counts, and budget-policy summaries
- `hmatrix_leaves.csv` for leafwise budgets, distances, ranks, bit assignments, and error components
- `gpu_performance.csv`, when CUDA/PyTorch is available
- `arithmetic.csv` for measured FP32/FP64 reconstruction errors and arithmetic bounds

The GPU stage can be skipped with:

```bash
python experiments/run_experiments.py --config experiments/configs/default.json --skip-gpu
```

## Stage 3: Figures

```bash
python experiments/plot_results.py --config experiments/configs/default.json
```

All font sizes for axes, ticks, legends, titles, tick mark geometry, grid opacity,
and saved figure formats are controlled by the `plot_style` block in the JSON
config.

The plotting stage creates figures for local/global certificate validation,
allocator storage, reserve-greedy optimality gap against dynamic programming,
H-matrix budget-policy ablations, storage with metadata, error decomposition,
precision distribution, precision versus block separation, and GPU timing when
available.

## A100 Run

For the final run on A100, use:

```bash
python experiments/prepare_data.py --config experiments/configs/a100.json
python experiments/run_experiments.py --config experiments/configs/a100.json
python experiments/plot_results.py --config experiments/configs/a100.json
```

The A100 configuration uses larger low-rank and H-matrix cases, ranks
`32, 64, 128, 256` for GPU timing, and `100` timed repetitions with CUDA events.

The `shared_scale_int8_gemm` result is a hardware-oriented reference for integer
GEMM. The theorem-level certificate corresponds to the column-wise scaling model,
which is measured separately by `columnwise_dequant_gemm`.

## Native C++/CUDA on A100

The native runner requires CMake >= 3.24, a C++17 compiler and CUDA Toolkit >= 11
with cuBLAS. It does not require PyTorch. Build for A100 (SM80), then run both
host and device self-tests before collecting results:

```bash
cmake -S experiments/cuda -B experiments/cuda/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build experiments/cuda/build -j
ctest --test-dir experiments/cuda/build --output-on-failure
python experiments/prepare_data.py --config experiments/configs/a100.json
python experiments/run_experiments.py --config experiments/configs/a100.json --skip-gpu
python experiments/run_native.py --config experiments/configs/a100.json
python experiments/plot_results.py --config experiments/configs/a100.json
```

Stage 1 creates `QMATGPU1` binary files: eight-byte magic, three little-endian
uint32 dimensions (m,n,rank), rank uint32 bit widths, then column-major FP32 P
and Q. Regenerate inputs on the target machine because manifests use absolute
paths. `gpu.native_shapes`, `native_modes`, and `native_budget_fraction` can
override the default shape/mode sweep and the sum-of-singular-values allocation
budget. The default modes are uniform4/8/12/16, mixed, and adaptive.

Stage 2 writes `native_performance.csv`, per-case CSVs in `native_raw/`, and
`native_run.json` with config and input/binary hashes. Validation failure exits
nonzero and stops aggregation. Hardware and library versions are in each row.
Full-matrix FP64 references are outside timing. The GPU self-test checks odd
dimensions, zero columns, all bit widths, mixed channels, and aligned INT8;
small cases are also checked against a C++ long-double reference.

The strict FP32 baseline uses `CUBLAS_COMPUTE_32F_PEDANTIC`; TF32 and FP16
references use explicit cuBLAS compute modes, as described in the
[NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html).
`packed_columnwise_channels` times unpack + channel GEMMs + accumulation.
`columnwise_channels` times already dequantized GEMMs + accumulation.
`unpack_only` is a decoding measurement, not GEMM throughput. Shared-scale INT8
is a separately quantized comparison and runs for uniform8 on aligned sizes
without INT32 overflow. TF32/FP16 rows measure error without claiming the strict
FP32 certificate. Packed 4/12/16-bit paths decode to FP32; no native integer
Tensor Core throughput at those widths is claimed.

Buffer allocation, transfers, CPU packing, reference computation, and tree
construction are excluded from CUDA-event timings. CPU packing time is reported
separately; FP16 timing excludes input conversion. Storage fields distinguish
physical payload, metadata, and additional execution workspace. Compare latency
alongside relative error; the modes do not have matched accuracy by construction.
The benchmark requires sufficient memory for full-output FP64 validation and
measures local reconstruction, not an entire H-matrix application.

