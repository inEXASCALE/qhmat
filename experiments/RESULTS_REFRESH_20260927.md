# Manuscript Results Refresh, 2026-09-27

The current results were compared with `rerun_backup/20260927_073811`.
The numerical and storage conclusions of the manuscript are reproduced.
The figures included by `main.tex` and `ex_supplement.tex` are regenerated
from the current result directories, preserving their existing filenames.

## Result Differences

| Study | Latest result | Difference from the previous run |
| --- | --- | --- |
| Local rank-64 allocation | 252 records, 76 feasible method/case records | CSV identical; all allocation and local-error claims unchanged |
| FP32/FP64 arithmetic | 48 checks | CSV identical; all checks pass |
| Diagonal-only hierarchy | 1,320 feasible configurations, zero violations | Storage, ranks, and allocation summaries unchanged; error/bound values differ at floating-point scale |
| Precision-menu study | 5,400 records, 3,780 feasible combinations | All five result CSVs and the summary JSON identical |
| Native reconstruction | 2,923 records, including 1,501 strict-bound reconstruction checks | Timing fields change; numerical errors and validation outcomes identical |
| Fixed-partition application | 3,600 method/trial records over 120 configurations | Timing fields change; input hashes, numerical errors, and validation outcomes identical |

The maximum absolute change in hierarchical measured error is
`1.1368683772161603e-13`. The maximum relative changes are
`2.1113005876927105e-7` for measured error and `1.849565922640295e-8`
for the error bound. Relative differences are largest for very small errors.
All displayed hierarchical table entries retain their previous rounded values.
The rank diagnostics CSV is identical to the previous run.

For block-relative budgets, maximum dense-FP64 compression remains
`44.262287015879224`, and maximum same-partition FP64 storage gain remains
`5.69530004767715`. The abstract's `44.3` and `5.70` now explicitly refer
to block-relative budgets. Global-relative budgets allow greater compression
and larger errors; they are not an equal-accuracy comparison.

## Manuscript Corrections

- The CPU run metadata report Python `3.12.11` and NumPy `2.5.3`, replacing
  the previous NumPy `2.4.6` description.
- Native and application records identify the device as
  `NVIDIA A100 80GB PCIe MIG 3g.40gb`. The manuscript retains only
  `NVIDIA A100`, since these experiments concern numerical validation,
  not performance claims. The full device details remain in the raw
  metadata. CUDA runtime, driver API, and cuBLAS identifiers remain
  `13020`, `13030`, and `130300`.
- The application run uses `application_a100.json` and the admissible-only
  source policy. Non-diagonal inadmissible blocks remain dense. Three of
  30 matrix/tolerance representations have admissible-block dense fallback:
  ellipse BEM at N=1024, tolerance 0.005 (two blocks), and N=2048 at
  tolerances 0.005 and 0.01 (six and two blocks).
- This application run uses only 4/8/12/16-bit factors. Decoder support
  for 24/32-bit storage is not evidence of a GPU test of the new
  diagonal-only hierarchy.
- The largest application arithmetic-error/allowance and
  representation-error/bound ratios are `0.034695604077734046` and
  `0.1381437832807757`; the supplement reports upward-rounded `0.0347`
  and `0.1382` as numerical validation margins.
- Figure legends and the arithmetic-validation axis use "bound" instead
  of "certificate", consistent with the revised manuscript terminology.
  Panel-title sizes and legends below the axes are preserved.
- The spectrum-resolved menu plot and reciprocal-scaling control had
  clipped long vertical labels. The former now uses the compact payload
  ratio notation, and the latter wraps its error label onto two lines.

## Validation

The full experiment test suite passes (69 tests, two NVIDIA-only tests
skipped locally). All 13 included PDF figures are rendered for visual
inspection. The active cross-document references, figure paths, and
rounded table entries pass the manuscript checks. Raw experiment CSVs
and JSON metadata are unchanged by this refresh. The commented alternative
allocation formula contains a repeated label, but it is inactive; the
reference checker now ignores comments before checking for duplicates.
No LaTeX installation is available locally, so the full manuscript PDF
and final page count are not rebuilt here.

No timing or speedup claim is added to the manuscript. Updated timings
remain in the experimental CSVs and diagnostic figures.
