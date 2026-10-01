# Quantization granularity comparison

This isolated experiment compares three representations of the same balanced
SVD factors. It does not replace any existing result or figure directory.

- **Per-tensor, fixed bits:** one symmetric scale for the entire left factor
  and one for the entire right factor; the lowest feasible bit width in the
  configured menu is selected separately for each block. Its quantization
  bound is $r a d c(b)$, where $a$ and $d$ are the factor max norms.
- **Per-channel, fixed bits:** one scale per factor column on each side, but a
  common bit width for all columns of a block. The bound is
  $c(b)\sum_j\sigma_j$.
- **Adaptive, per-channel:** the paper's reserve-based singular-value-aware
  bit assignment, with one scale per column on each side and bound
  $\sum_j\sigma_j c(b_j)$.

All methods use the same prepared matrix, geometric partition, retained rank,
block-relative total budget, 2% arithmetic reserve, and integer menu. The
reported feasibility concerns **truncation plus factor quantization**; the
measured matrix product is evaluated in FP64. No arithmetic certificate or
hardware-speed claim is inferred from this comparison. Failed configurations
have blank storage and error fields, not a dense off-diagonal fallback.

The storage total includes the same FP64 diagonal blocks, integer factor
payload, FP64 scales, and precision identifiers. A fixed-width block stores
one menu identifier; an adaptive block stores one identifier per retained
component. Tree topology, temporary workspace, and the cost of finding the
assignment are excluded. The same-partition FP64 factor control retains the
same ranks and diagonal blocks. The best feasible fixed width is a deliberately
strong baseline: the greedy method need not use less storage in every case.

The configuration selects circle BEM, 2D $1/r$, and 3D $\log r$ matrices at
$N=512$ and $N=2048$, all already present in the diagonal-only experiment.
On a machine where `experiments/data_offdiagonal_a100/manifest.json` already
exists, no preparation or CUDA build is needed:

```bash
python experiments/run_quantization_granularity.py \
  --config experiments/configs/quantization_granularity.json
python experiments/plot_quantization_granularity.py \
  --config experiments/configs/quantization_granularity.json
python -m unittest discover -s experiments/tests -p 'test_quantization_granularity.py' -v
```

If the prepared inputs are absent, first run
`python experiments/prepare_data.py --config experiments/configs/offdiagonal_a100.json`.
Results are written only to `experiments/results_quantization_granularity/`;
plots go only to `experiments/figures_quantization_granularity/`. The plotting
script reads the saved CSV, checks its provenance, and uses the shared
`plot_style` font sizes and external legend layout. Missing infeasible points
are not connected. Upload both output directories before incorporating any
quantitative claims into the supplementary material.
