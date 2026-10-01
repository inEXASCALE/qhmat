# qhmat

[![CI](https://github.com/inEXASCALE/qhmat/actions/workflows/ci.yml/badge.svg)](https://github.com/inEXASCALE/qhmat/actions/workflows/ci.yml)
[![PyPI: pending](https://img.shields.io/badge/PyPI-pending-566573)](https://pypi.org/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-2E6B5D)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-Read%20the%20Docs-08766A)](docs/index.rst)

Adaptive integer-quantized hierarchical matrices with a C++ core, Python bindings, and an optional CUDA backend.

See the [documentation](docs/index.rst) for installation, a quickstart,
kernel/BEM/integral-operator examples, and the full API reference. Build it
locally with `python -m pip install -e '.[docs]'` followed by `make html`;
open `docs/_build/html/index.html`.

In the proposed diagonal-only-dense representation, `qhmat` stores diagonal blocks densely in FP64 and every non-diagonal block, whether admissible or inadmissible, as quantized SVD factors. This factorized format does not imply that every near-field block has low numerical rank or uses less storage than a dense block. Each retained singular component receives its own signed-integer bit width and two balanced-factor scales. The reserve-greedy allocator chooses widths subject to the singular-value error surrogate

$$
  \sum_j \sigma_j\left(\frac{1}{2^{b_j-1}-1}
  +\frac{1}{4(2^{b_j-1}-1)^2}\right) \leq B^{\mathrm q}.
$$

The package uses a strict construction policy: an infeasible off-diagonal block raises an error; it is never silently stored as dense FP64. The archived admissible-only comparison and GPU application experiments in `experiments/` use a different storage policy.

## Install

Python 3.10+, a C++17 compiler, and CMake are required for a source build. NumPy is installed automatically.

```bash
pip install .
```

Once a release is available on PyPI, installation will be `pip install qhmat`. The default build is CPU-only and does not need the CUDA toolkit. For an editable development install:

```bash
pip install -e '.[test]'
python -m pytest -q
```

To compile the optional CUDA backend on an NVIDIA machine, set a suitable compute capability (for an A100, `80`):

```bash
pip install . -Ccmake.define.QHMAT_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80
```

CUDA requires a compatible toolkit, host compiler, and driver. The CPU wheel remains usable without them. `cuda_built` reports whether the installed extension contains the CUDA path.

## Quick start

```python
import numpy as np
import qhmat

points = np.linspace(0.0, 1.0, 64)[:, None]
distance = np.abs(points - points.T)
matrix = 1.0 / (distance + 0.4)

h = qhmat.HMatrix.from_dense(
    matrix, points, tolerance=0.25,
    leaf_size=8, bit_widths=(4, 8, 12, 16, 24, 32),
)
x = np.ones((64, 4))
y = h @ x

print(h.shape, h.logical_bits, h.reconstruction_bound)
print(np.max(np.abs(matrix - h.to_dense())))
print(qhmat.memory_bytes(matrix), qhmat.memory_bytes(h))
print(h.storage_report())
```

Pass `verbose=1` to `HMatrix.from_dense` or `HMatrix.from_partition` to print
the input array payload, dense FP64 baseline, logical storage, and estimated
C++ resident size once construction succeeds.

`from_dense` uses a stable median-split cluster tree and Euclidean bounding-box admissibility. Optional `support_lower` and `support_upper` arrays make the boxes enclose full BEM panel supports. `from_partition` accepts a user-supplied disjoint block partition. NumPy's LAPACK-backed SVD runs during construction; bit allocation, packing, storage accounting, reconstruction, and multiplication run in C++.

The package exposes `allocate_bits`, `coefficient`, `spatial_partition`, `Rectangle`, and `NativeHMatrix` for lower-level integration. `HMatrix.matmul(x, device="cuda")` selects the optional CUDA path; the `@` operator uses CPU.

## Storage and error semantics

- `logical_bits` counts packed integer payload, two FP64 scales per component, direct precision labels, and FP64 diagonal entries. It excludes tree topology, headers, and workspace.
- `allocated_bytes` counts native vector payloads, including word padding and in-memory offsets; it excludes C++ object and allocator overhead. It is not an on-disk format size.
- `memory_bytes(array)` returns NumPy's `array.nbytes` (for a view, only its visible elements). `memory_bytes(h)` and `h.resident_bytes` estimate owned C++ memory using object sizes and vector capacities; they exclude allocator overhead, the Python wrapper, CUDA allocations, and temporary construction workspace. Neither is process RSS.
- `h.storage_report()` compares the dense FP64 baseline with logical storage and the C++ resident estimate. `source_array_bytes` records the original NumPy input payload when built from an array; it is unavailable after `load`. Compression ratios use the FP64 baseline, which may differ from the input array size if its dtype is not FP64. The source array stays alive if the caller retains it.
- `save` writes a versioned, checksummed binary representation. `load` validates the checksum, dimensions, and block partition. The checksum detects accidental corruption, not malicious tampering. Saving never overwrites an existing file; remove or rename it explicitly first.

- `reconstruction_bound` is the maximum over blocks of measured SVD truncation error plus the singular-value quantization surrogate. It bounds the stored representation in exact arithmetic, not the floating-point error of `matmul` or `to_dense`.
- The construction reserves 2% of each block budget by default for arithmetic, but this reserve alone is **not** a certified bound for every downstream operation. Application-specific arithmetic analysis is required.
- The CUDA path reads the packed codes directly, but currently uploads representation and right-hand sides per call. It is a correctness backend, not a claim of GPU speedup; GPU validation is required before relying on it in production.

To reuse a constructed matrix in another process, call `h.save("matrix.qhmat")` once and later `qhmat.HMatrix.load("matrix.qhmat")`.

## Development and release

The compiled package is built with scikit-build-core and pybind11. The `experiments/` directory contains the paper's independent research scripts and is excluded from package distributions. CI builds and tests the CPU package; CUDA builds must also be checked on an NVIDIA runner.


The project is licensed under [MPL-2.0](LICENSE). Source and issue tracking are at [inEXASCALE/qhmat](https://github.com/inEXASCALE/qhmat).
