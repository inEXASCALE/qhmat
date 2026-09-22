# BEM and Euclidean kernel experiments

All matrices are generated locally and deterministically; no external mesh or
matrix download is required. The saved FP64 matrix is the compression reference.

The revised `offdiagonal_a100.json` uses these same matrices and geometric
partitions, but quantizes every non-diagonal leaf, including inadmissible ones.
Only diagonal leaves stay dense. Rank selection and failure handling are
specified in the README; the dense-fallback discussion below describes the
archived `a100.json` policy only.

## Matrix definitions and sizes

| Family | Geometry | Matrix entry | Smoke configuration | A100 configuration |
| --- | --- | --- | --- | --- |
| Point kernel | Uniform / two-cluster clouds in 2D and 3D | `1/r`, off diagonal | N=128 | N=512,1024,2048 |
| Point kernel | Same points as corresponding inverse kernel | `log(r)`, off diagonal | N=128 | N=512,1024,2048 |
| 2D Laplace BEM | Circle radius 0.5; ellipse semiaxes 0.75,0.25 | `-integral_panel log(r) ds / (2*pi)` | N=128 | N=256,512,1024,2048 |
| 3D Laplace BEM | Cube side 1; cuboid sides 1.5,1,0.5 | `integral_panel 1/r dS / (4*pi)` | N=384 | N=96,384,1536 |

Here `r = ||x-y||_2` is always the Euclidean distance, using natural logarithms
and dimensionless coordinates. The point kernels use zero diagonals to omit
self interactions and leave every off-diagonal value unregularized. Coincident
distinct points are rejected. This is not a claim that either kernel is finite
at r=0, nor that these matrices are positive definite. In particular `log(r)`
in 3D is a requested kernel benchmark, not the 3D Laplace Green's function.

The BEM discretization uses piecewise constant source density and collocation at
panel midpoints. Circle and ellipse boundaries are approximated by straight
segments. Closed cube/cuboid surfaces use rectangular elements. Source-panel
integrals, including the singular self entries, are evaluated analytically.
No epsilon is inserted in the distance and no symmetry is imposed. This is a
single-layer collocation matrix, not a Galerkin matrix or a complete PDE solve.
For context, the single-layer boundary operator and the distinction between
collocation and weak-form assembly are described by the
[Bempp operator documentation](https://bempp.com/handbook/api/boundary_operators.html);
the 2D constant-element collocation convention is also described in
[Constant Boundary Elements on graphics hardware](https://www.scielo.br/j/jbsmse/a/fM3cjFzddcqgqZHdJGMVLkG/?lang=en).

## Experimental sequence

1. Validate assembly before compression: analytic self terms, independent
   high-order Gauss-Legendre quadrature away from singularities, rectangular
   far-field area/r behavior, and refinement toward the constant-density trace
   on a circle. These checks are in `tests/test_geometric_matrices.py`.
2. For every matrix, use the existing six tolerances and three budget policies;
   record actual max-norm error, certificate, storage including scale/map
   metadata, bit distribution, and dense-fallback fraction.
3. Compare N-scaling within each kernel/geometry. Compare uniform and clustered
   clouds to isolate geometry. Inspect precision against actual Euclidean box
   separation/diameter, not differences of reordered indices.
4. Export up to two largest distinct admissible leaves per new matrix to the
   native C++/CUDA benchmark. Repeat all precision modes on the saved factors.
   Report source block/residual alongside GPU error and timing; reconstruction
   timing excludes BEM assembly, SVD, and H-tree construction. A coarse matrix
   with no admissible leaves produces no native leaf input.

The spatial cluster tree uses a stable median split along the largest coordinate
spread. The saved permutation maps reordered DOFs to original points/panels.
BEM admissibility uses full panel-support boxes; it is intentionally conservative
on coarse meshes. Those meshes remain useful dense controls, with compression
ratio 1 if all leaves are dense. SVD factorizations are reused across tolerances
and policies for the same matrix.

The largest planned N=2048 matrix occupies 32 MiB in FP64; N=1536 occupies
18 MiB. Temporary arrays, factorization caches, output CSVs, and native GPU
validation require additional memory. Data are assembled one matrix at a time
with row chunks of 128. All 44 H-matrix cases in the A100 config include the
six original 1D cases and 38 new geometric cases. This is a controlled first
scaling study; larger meshes should follow measured runtime and memory usage.

## Run and inspect

```bash
python -m unittest discover -s experiments/tests -v
python experiments/prepare_data.py --config experiments/configs/a100.json
python experiments/run_experiments.py --config experiments/configs/a100.json --skip-gpu
python experiments/run_native.py --config experiments/configs/a100.json
python experiments/plot_results.py --config experiments/configs/a100.json
```

Build and device-test the native binary first using the instructions in README.
`geometric_suites` controls matrix dimensions, point distributions, panel counts,
clustering leaf size, and native leaves per case. The new matrices join the
existing H-matrix manifest and CSVs; geometry, mesh, and permutation are in NPZ
files, while assembly/diagonal conventions are in the manifest. The BEM
discretization error is not part of the quantization certificate.

New figures are organized in `figures_a100/suites/<family_geometry_kernel>/`.
Each group includes the original diagnostics plus `size_scaling`, with stored
compression ratio and fallback fraction versus N. Native plots are separated
by originating matrix and block shape. Every figure uses the shared plot style.
