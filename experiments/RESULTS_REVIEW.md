# Uploaded A100 run: results review

## Diagonal-only manuscript update

The main paper now uses `results_offdiagonal_a100` and
`figures_offdiagonal_a100` uploaded at `4c4e48f`. All 1,320 hierarchical
configurations (44 matrices, 10 tolerances, 3 policies) are feasible, with no
off-diagonal dense fallback or measured certificate violations. Recorded
source hashes match the experiment implementation. This is an analysis of
the uploaded outputs, not a new large-scale run on the editing machine.

At block-relative tolerance 0.01, the eight largest-size cases in the main
table give dense-FP64 compression 8.890--39.141, same-partition FP32 ratios
1.666--2.757, and same-partition FP64 ratios 2.721--5.361. These ratios use
the same diagonal blocks and ranks; they do not imply speedup or superiority
over optimized equal-accuracy H-matrices. Metadata is 3.69--17.92% of storage.
Circle BEM and the 3D uniform log-distance kernel figures expose contrasting
storage composition; the supplement contains cube BEM, tolerance/rank tables,
and a fixed-matrix budget-policy comparison.

Rank diagnostics include full-rank non-diagonal inadmissible blocks for 3D
BEM and uniform point kernels at 1e-6. No universal near-field low-rank claim
is warranted. The uploaded archive contains summary CSVs and rank diagnostics
but not `hmatrix_leaves.csv`; individual leaves were not independently
reconstructed for this manuscript update. No experiment code, input, result,
or figure is changed. Verify the table cells and aggregate claims with:

```bash
python -m unittest discover -s experiments/tests -p test_offdiagonal_evidence.py -v
```

## Historical run

Historical admissible-only baseline: none of the storage numbers below
evaluates the revised diagonal-only dense policy. Its new runs use separate
`results_offdiagonal_a100` and `figures_offdiagonal_a100` directories.

Reviewed input: commit `d04310a` (`update`) on `main`. The observations below
come from `results_a100/`, not from new measurements on the plotting machine.
The raw CSVs and provenance JSON are unchanged by this review.

## Completeness and numerical checks

- `local_certificate.csv`: 252 rows (6 cases x 6 tolerances x 7 allocators).
  There are 76 feasible rows, all satisfying measured error <= singular-value
  certificate with the validation tolerance. Reserve greedy, DP, and uniform16
  each have 24 feasible / 12 infeasible rows; uniform12 has 4 / 32.
  Naive greedy, uniform4, and uniform8 have no feasible rows in this sweep.
  Infeasibility is not a certificate violation, but must remain visible in plots.
- `hmatrix_certificate.csv`: 792 rows (44 cases x 6 tolerances x 3 policies),
  with no measured global error exceeding its certificate plus `1e-12`.
- `arithmetic.csv`: 48 rows, with no measured rounding error exceeding the
  representation-plus-GEMM certificate plus `1e-14`.
- `native_performance.csv`: 2,923 rows from 79 inputs x 6 modes. The 474 raw
  CSVs contain exactly the same multiset of measurement fields as the aggregate.
  All 1,501 `certificate_pass` rows satisfy their separate quantization,
  arithmetic, and total bounds including each row's `reference_slack`.
  The 948 TF32/FP16 `error_measured` rows are not certificate validations;
  474 `decode_pass` rows test decoding, not reconstruction accuracy.
  All recorded timing triples satisfy 0 < min <= median <= max.
- `gpu_performance.csv` is empty. The native CUDA data are the performance
  evidence; there is no separate uploaded PyTorch performance sweep.
- `hmatrix_leaves.csv` is absent. Leaf bit histograms and distance-bin statistics
  cannot be independently checked or regenerated from this checkout. The plotter
  preserves their existing images and warns. Replot on the node with that CSV.

These are numerical checks against FP64 references, not interval-arithmetic
proofs. Uploaded inputs and the CUDA binary are represented by hashes in
`native_run.json`; this review does not independently rerun that binary.

## Allocation and storage

For the 24 matched feasible case/tolerance pairs, reserve-greedy payload / DP
payload has median **1.0910**, minimum **1.0000**, and maximum **1.5034**.
The maximum occurs at `lowrank_n256_r64_algebraic`, tolerance `0.1`.
The results support feasibility and a useful heuristic, not an unqualified
claim of near-optimal allocation. Report the distribution or worst case as
well as the mean-by-tolerance figure.

Default `block_relative` policy, dense FP64 bits / stored factor bits including
the recorded scales and map metadata:

| Family | Rows | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 1D kernels | 36 | 1.324 | 5.319 | 10.289 |
| BEM 2D | 48 | 1.289 | 3.176 | 13.402 |
| BEM 3D | 36 | 1.000 | 1.127 | 2.613 |
| Euclidean point kernels | 144 | 1.000 | 1.940 | 3.602 |

These medians span sizes, geometries, and tolerances; they are descriptive,
not matched-size comparisons. The maxima occur at tolerance `0.1`, respectively
for matern1d N=1024, circle BEM N=2048, cube BEM N=1536, and uniform 2D log kernel
N=2048. Present tolerance curves rather than only the maxima. Storage here is
not total application memory or peak GPU execution memory.

There are 72 zero-error, zero-certificate rows from four fully dense cases:
cube/cuboid BEM N=96 and uniform 3D inverse/log kernels N=512, over all policies
and tolerances. Their certificate/actual ratio is undefined, not zero.
Exclude these rows from ratio averages; keep their compression ratio of one.

For nonzero-error rows under `block_relative`, the family median certificate /
actual ratios are 18.48 (1D), 19.66 (BEM 2D), 15.08 (BEM 3D), and 16.24 (point
kernels). The bounds are valid but conservative. `global_relative` eliminates
reported allocation fallbacks in this sweep and improves storage, while its
point-kernel certificate/actual ratio reaches 181.03. Its budget semantics are
different: compare policies with measured global error as well as storage.
The global certificates are absolute errors; do not compare them directly
with the dimensionless `tol` column without the relevant matrix norm.

## Native GPU performance

The device string is **NVIDIA A100 80GB PCIe MIG 3g.40gb**, not an exclusive
full A100. Report this exact partition alongside runtime `13020`, driver API
version `13030`, cuBLAS `130300`, 20 warmups, and 100 timed repetitions.

No `packed_columnwise_channels` row beats its same-run strict-FP32 baseline:
all 474 speedups are below one (maximum **0.9783x**). Uniform predecoded paths
have median speedup around one, as expected for their FP32 GEMM realization.
Mixed/adaptive paths pay for multiple GEMMs and accumulation.

For the 68 real BEM/kernel leaf inputs (block orders 24 through 512):

| Packed mode | Median speedup vs strict FP32 |
| --- | ---: |
| Uniform4 | 0.7071x |
| Uniform8 | 0.7143x |
| Uniform12 | 0.6886x |
| Uniform16 | 0.6830x |
| Mixed | 0.2857x |
| Adaptive | 0.3704x |

The real-leaf shared-scale INT8 median is **0.8333x**. These are per-input
ratios, not total H-matrix application speedups. Real leaves with the same
rank remain separate observations, not a connected synthetic rank sweep.
Native reconstruction errors are relative to the saved FP32 factors' FP64
product, not the original BEM/kernel block; they do not include `source_eta`
or establish full-operator or PDE-solution accuracy.

For the synthetic 4096 x 4096 output, the following speedups use uniform8
baseline runs for the hardware references and the adaptive run for packed:

| Rank | TF32 | FP16 inputs | Shared-scale INT8 | Packed adaptive |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.792x | 2.024x | 0.639x | 0.147x |
| 64 | 2.872x | 3.039x | 1.033x | 0.244x |
| 128 | 4.301x | 5.233x | 1.648x | 0.398x |
| 256 | 5.190x | 9.748x | 2.212x | 0.556x |

At rank 256, relative max errors are approximately `4.594e-4` (TF32),
`4.588e-4` (FP16), `9.228e-3` (shared INT8), and `4.138e-4` (packed adaptive).
These methods do not have matched errors or identical quantization models.
TF32/FP16 and shared-scale INT8 speedups must not be attributed to the
paper's columnwise mixed-bit algorithm.

Timings exclude allocation, transfers, CPU packing, reference computation,
and H-tree construction; FP16 also excludes conversion. Packed timing includes
decode + channel GEMMs + accumulation, not a full end-to-end application.
At synthetic rank 256, packed adaptive stores 3,257,344 payload-plus-metadata
bytes versus 8,388,608 FP32 factor bytes, but also reports **75,497,472 bytes**
of execution workspace. A storage reduction is not a peak-memory reduction.

## Limitations and next measurements

1. Keep the paper's central claim on error certification and storage reduction.
   Present the current CUDA implementation as a validated reference execution
   path with measured overhead, not as an accelerated H-matrix solver.
2. Repeat independent benchmark runs and save every timing sample. One
   `unpack_only` row (4096 x 4096, rank 64, uniform12) has min/median/max of
   0.01434 / 0.01536 / 2.65728 ms. FP32 medians across the six mode runs of the
   same input differ by up to 1.8x. This is evidence of timing variability,
   not enough to identify its cause. Do not treat min/max as confidence bounds.
   Record MIG configuration, clocks, concurrent workload, and execution order;
   interleave matched baselines and report independent-run uncertainty.
3. Add equal-error or equal-certified-budget comparisons. The local uniform4/8
   infeasibility and the shared-INT8 accuracy gap make bits-only comparisons
   insufficient. Show Pareto frontiers without combining incompatible scopes.
4. Increase BEM 3D and selected point-kernel sizes with memory checks. Current
   BEM 3D sizes are small and include entirely dense partitions. Report
   admissible/dense fractions, rank distributions, and fallback reasons to
   explain the weaker compression rather than generalizing from these sizes.
5. Restore the leaf CSV (a compressed archival copy is sufficient for sharing)
   before finalizing precision-distance claims. Distinguish leaf counts from
   rank-weighted and byte-weighted precision distributions.
6. As a separate optimization study, evaluate batched small-block execution,
   fused decode/accumulate, and reuse of decoded factors; then measure an
   H-matrix matvec/matmul or solver workflow with preprocessing amortization.
   These are proposed experiments, not demonstrated improvements.

## Plotting changes

Every regenerable figure uses an external bottom-center shared legend, with
renderer-measured clearance below x labels, tick labels, and offset text.
Local certificate and allocator panels use a compact multi-row layout.
Matched measured/bound curves share colors; native modes use consistent
colors with distinct packed/predecoded markers. PNG/PDF exports grow to fit
the legend without shrinking the data axes. No measured CSV values are edited.
