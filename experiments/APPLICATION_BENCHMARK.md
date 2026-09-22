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

