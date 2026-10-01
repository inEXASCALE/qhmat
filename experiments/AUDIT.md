# Paper-to-code audit

The revised diagonal-only policy is tested in `tests/test_offdiagonal.py`:
non-diagonal inadmissible blocks are factorized, diagonal blocks alone remain
dense, and failed candidates produce no complete-representation statistics.
The end-to-end configuration is `configs/offdiagonal_a100.json`; see the README
for the isolated three-stage rerun. CUDA application decoding additionally
supports 24/32-bit stored codes, without claiming native arithmetic support.

The table below documents the **archived admissible-only protocol**, not the
revised policy. This table
distinguishes saved measurements from proposed extensions. No A100 timings are
committed or claimed without executing the native benchmark on that device.

| Paper claim | Implementation / output | Coverage |
| --- | --- | --- |
| Algorithm 1, reserve-safe allocation | `quantization.py:reserve_greedy_allocation` | Same tail-first channels and highest-bit reserve; directly recomputed final certificate |
| Local singular-value and geometry bounds | `local_certificate.csv` | Both bounds, actual error, feasibility, bit map and spectrum |
| Uniform, naive, DP comparisons | `local_certificate.csv` | All configured uniform widths; DP verified against enumeration at small rank |
| Precision by singular-value index | `allocation_pattern_*.pdf` | One plot per saved local case, tolerance curves |
| Unrepaired-greedy failures | `allocator_failures.pdf` | Feasibility failures retained, including uniform infeasibility |
| Global maximum over nonoverlapping leaves | `hmatrix_certificate.csv` | Actual reconstruction, max leaf certificate, dense fallbacks for all three failure paths |
| Budget-policy ablation | `hmatrix_leaves.csv` | Block-relative, global-relative, hybrid |
| Storage and geometry diagnostics | H-matrix CSVs and plots | Payload, scale/map estimate, precision distribution, separation diagnostic |
| FP32/FP64 arithmetic validation | `arithmetic.csv` | Actual dtype changed, sequential accumulation, factor-conversion term included |
| Native performance | `cuda/benchmark.cu`, `native_performance.csv` | Packed column scales, channel reconstruction, FP32/TF32/FP16 and shared-scale INT8 |
| Reproducibility | `manifest.json`, `native_run.json` | Saved inputs/config, input/binary hashes, CUDA/driver/cuBLAS versions in CSV |
| Shared plot style | `plot_results.py` | All new plots use the existing font/tick/legend configuration |

## Corrections made during this audit

- Failed allocation or arithmetic-reserve checks previously left compressed
  values in the reconstructed matrix while excluding their certificates. All
  failed candidates now become exact dense leaves with dense storage counted.
- The H-matrix runner used FP64 reconstruction even when its bound assumed FP32.
  The reconstruction dtype now follows the configured unit roundoff.
- Python's purported column-wise GPU path used a single scale for the whole
  factor. It now computes one scale per column; the shared-scale proxy is distinct.
  Both Python and native GPU runners now consume stage-one saved factors.
- The optional PyTorch import happened outside the missing-dependency handler.
  Missing PyTorch is now skipped, but real CUDA execution errors propagate.
- Local plots omitted the computed geometry bound and the per-index map. Both
  are now retained and visualized; infeasible methods have a separate figure.
- Native results distinguish packed storage from FP32 workspaces and preconverted
  baselines. Decoding-only time is not a matrix-product speedup.
- Storage comparisons use per-case panels to avoid comparing means over different
  feasible subsets. The tolerance sweep also includes 0.05 and 0.1 so that flat
  spectra at larger ranks have feasible high-bit cases despite the conservative
  singular-value-only bound.
- The abstract no longer asserts experimentally demonstrated speedups before
  target-GPU measurements exist.

## Interpretation and remaining extensions

The matrix suite now includes synthetic low-rank blocks, three 1D kernel families,
2D/3D Euclidean inverse-distance/log-distance point kernels, and 2D/3D analytic
constant-panel BEM collocation matrices. Spatial partitions use Euclidean boxes
and full BEM supports; representative far-field factors feed the native runner.
The per-family experiment plan and definitions are in MATRIX_SUITES.md.
It is not a downloaded application-matrix collection. A real application dataset,
multi-seed uncertainty estimates, pairwise summation,
batched irregular leaves, and full H-matrix matvec/matmul/solve timing would
strengthen a broader application-performance claim but are not implemented here.
The current paper protocol explicitly limits its performance claims to local
channel reconstruction. FP64 residuals/references are numerical checks, not
interval-certified residual bounds. Native low-bit storage does not establish
native 4-/12-/16-bit integer arithmetic throughput.
