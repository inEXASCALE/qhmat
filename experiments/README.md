# Quantized H-Matrix Experiments

The experiment harness is split into three reproducible stages.

1. Prepare data files.
2. Run numerical and GPU experiments.
3. Plot figures from saved CSV files.

For the separate per-tensor/per-channel/adaptive storage comparison on
selected prepared matrices, see [QUANTIZATION_GRANULARITY.md](QUANTIZATION_GRANULARITY.md).

The revised diagonal-only-dense experiment keeps only diagonal blocks dense
in FP64. Every non-diagonal block, including geometrically inadmissible
near-field blocks, is represented by quantized factors; these blocks need not
have low numerical rank. The geometric partition is unchanged. The archived
admissible-only experiments below use a different storage policy.

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

Host validation can also be built without a CUDA toolkit:

```bash
cmake -S experiments/cuda -B experiments/cuda/build-host -DQMAT_ENABLE_CUDA=OFF
cmake --build experiments/cuda/build-host -j
ctest --test-dir experiments/cuda/build-host --output-on-failure
python -m unittest discover -s experiments/tests -v
```

See [AUDIT.md](AUDIT.md) for the claim-to-code map and remaining extensions.

## Replot existing results

Stage 3 is independent of data generation and benchmarking. On the node, with
the experiment Python environment active, run only:

```bash
python experiments/plot_results.py --config experiments/configs/a100.json
```

The plotter reads existing CSVs without modifying measurements. Legends are
deduplicated across panels and centered below the entire figure, underneath
tick labels, offset text, and axis labels. Their measured width controls the
number of columns; extra legend rows expand the saved canvas rather than
reducing the data area. Both PNG and PDF use the same layout.

All settings live in `plot_style` in the selected JSON config:
`axis_label_size`, `tick_label_size`, `legend_font_size`, `title_font_size`,
`legend_max_columns` (4), `legend_gap_points` (14),
`legend_side_padding_points` (12), `legend_column_spacing` (1.4), and
`legend_label_spacing` (0.6). Gap and side padding are in points; column and
label spacing are in legend-font units. Increase the gap for more whitespace.

The uploaded A100 results omit `hmatrix_leaves.csv`. Without it, the plotter
warns and preserves existing precision-distribution and precision-vs-distance
figures, instead of overwriting them with empty charts. Run stage 3 on the node
where that CSV remains available to restyle those figures too. Undefined
certificate/actual ratios for zero-error cases and empty distance bins are
excluded from averages or left missing, never plotted as zero measurements.

See [RESULTS_REVIEW.md](RESULTS_REVIEW.md) for the uploaded-run assessment,
timing-scope caveats, and recommended next measurements.

## Complete H-matrix application performance

The new three-stage `prepare_application.py`, `run_application.py`, and
`plot_application.py` workflow compares dense FP32, ordinary H-matrix, and
adaptive-bit **Y = A X** on identical frozen leaf partitions and ranks. It uses
multi-RHS batched GEMMs and matched fused packed/FP32 kernels; it does not
reconstruct dense leaf matrices inside the timed application. See
[APPLICATION_BENCHMARK.md](APPLICATION_BENCHMARK.md) for node commands, smoke
tests, timing scopes, error checks, and interpretation. Application results use
separate directories and do not replace the existing reconstruction results.

## Selected manuscript figures

The focused A100 figure set can be regenerated independently:

```bash
python experiments/plot_paper.py --config experiments/configs/a100.json
```

This reads the committed local, hierarchical, and native CSVs and writes
`paper_allocator_comparison`, `paper_hmatrix_storage`, and
`paper_native_scale_rank` in the configured figure formats. It does not rerun
benchmarks, overwrite the full diagnostic figures, or require leaf records.
It uses the same `plot_style` font and external-bottom-legend controls as
`plot_results.py`. This selected view requires the A100 case/rank sweep;
missing or duplicate measurements fail instead of silently changing the sample.

Allocator payload is normalized by uniform16 *factor* payload, with infeasible
assignments left missing. The native figure uses matching mode-run FP32
denominators and a common logarithmic scale for small and large outputs; ranks
remain explicit rather than being pooled into an error--latency frontier.
The native scale/rank plot now belongs to `ex_supplement.tex`; the main paper
retains three figure groups and two tables, including same-partition storage
controls and the fixed-workload GPU comparison.

Regenerate the focused supplementary figures from the same saved run with:

```bash
python experiments/plot_supplement.py --config experiments/configs/a100.json
```

To regenerate only `supp_allocator_n512.pdf` and `.png`, without touching
other figures or rerunning any experiment:

```bash
python experiments/plot_supplement.py --config experiments/configs/a100.json --only allocator
```

Tolerance panels use the full configured range with 5% logarithmic padding
on both sides so boundary markers remain visible even when only the largest
tolerances are feasible. Tick values and missing/infeasible data are unchanged.

This writes the `supp_*` local-certificate, allocator, precision-map, 1D-kernel,
and real-leaf GPU figures, and refreshes `paper_native_scale_rank`. The real-leaf
plots require the same 68 inputs in all six modes. Boxplot spreads represent
different workloads, not timing confidence intervals. All styles use the same
`plot_style` configuration. No benchmark or measurement file is changed.

The manuscript numbers use the frozen archive at commit `d6164dc`, which adds
same-partition controls and updates the native timings. Earlier assessment
documents refer to their explicitly named historical commits. Test the current
tables, figure inputs, and cross-document references with:

```bash
python -m unittest discover -s experiments/tests -v
```

Compile with the repository's `latexmkrc`, which automatically builds the peer
document's labels. See [BUILDING.md](../BUILDING.md) for local and Overleaf
instructions and the clean/incremental cross-reference test.

## Same-partition floating-factor storage controls

The restored A100 archive and new CPU runs record `stored_dense_bits`, `retained_factor_entries`,
`same_partition_fp32_bits`, and `same_partition_fp64_bits` in
`hmatrix_certificate.csv`. For the realized compressed leaves, let
`S = sum((m_k+n_k)*r_k)`; let `D` be the FP64 cost of dense and fallback leaves.
The controls are `D+32*S` and `D+64*S`. Singular values are absorbed into the
floating factors, so these controls have no quantization scales or precision
maps. Tree topology, allocation overhead, and execution workspace are excluded
from both sides. These are storage-only controls, not new FP32 accuracy tests
or independently optimized H-matrix baselines.

On the node retaining the **same run's** complete `hmatrix_leaves.csv`, recover
the controls without rerunning data generation, CPU experiments, or CUDA:

```bash
python experiments/summarize_storage.py --config experiments/configs/a100.json
```

The command validates leaf totals, payload/metadata costs, and fallback counts
against every archived summary row before writing the separate
`hmatrix_storage_baselines.csv`. It leaves existing results untouched and
rejects missing, duplicated, or mismatched records. The current archive already
contains the controls in `hmatrix_certificate.csv`, so this recovery step is
optional and is not needed to reproduce the manuscript's main table. The full
leaf CSV remains untracked. Do not infer controls from unweighted mean bit
widths or substitute leaf records from a different run.

## Independent precision-menu and reserve study

This CPU-only study supplements the theory and emulation results. It does not
read or overwrite `data_a100`, `results_a100`, `figures_a100`, or the application
benchmark directories. Run the three stages separately from the repository root:

```bash
python experiments/prepare_allocator_study.py --config experiments/configs/allocator_study.json
python experiments/run_allocator_study.py --config experiments/configs/allocator_study.json
python experiments/plot_allocator_study.py --config experiments/configs/allocator_study.json
```

The default configuration generates 50 low-rank factor triples (200 by 180,
rank 25; five spectra and ten seeds), then evaluates nine tolerances and four
nested precision menus. Greedy, ordered DP, and uniform allocation use the same
matrix and singular-value certificate budget. Reported payloads exclude scales
and metadata; additional bit widths are emulated, not native GPU formats.

Stage 1 writes ignored NPZ inputs and a hashed manifest under
`experiments/data_allocator_study`. Stage 2 verifies those inputs and writes
`experiments/results_allocator_study`: menu measurements, paired transitions,
reserve traces, exact-rational scalar-boundary comparisons, reciprocal-scaling
controls, summary statistics, and provenance. The provenance records source and
input hashes, software versions, the checkout revision, and result CSV hashes;
source hashes also identify edits not yet committed at run time. The reciprocal
scaling control fixes the bit map and compares independent column scales with
shared channel scales. Floating-point threshold cases are explicitly flagged,
not treated as interval-certified decisions.

Stage 3 reads only the archived results and writes four PDF/PNG figures under
`experiments/figures_allocator_study`. The checked-in results suffice to rerun
this stage without regenerating data or running the experiments. Edit the shared
`plot_style` keys in the configuration to change axis-label, tick-label, legend,
and title font sizes. Legends remain outside the axes below the x labels.
Changing only plotting style does not invalidate the experimental hashes.

The primary paired-menu figure is included in `main.tex`; spectrum-resolved
statistics, reserve/boundary diagnostics, and the scaling control are in
`ex_supplement.tex`. Tests cover numerical certificates, nested-menu DP
monotonicity, input/result integrity, the three-stage smoke pipeline, archived
manuscript numbers, and legend placement:

```bash
python -m unittest discover -s experiments/tests -v
```
