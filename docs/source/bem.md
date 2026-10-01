# Boundary-element matrices

A boundary-element method (BEM) turns a boundary integral into interactions among panels. Geometry-aware clustering is especially useful here, but the kernel's singularity and the panel supports must be handled correctly before compression.

## Planar Laplace single layer

Let $\Gamma$ be a polygonal approximation of the unit circle, with straight panels $\Gamma_j$ and midpoint collocation sites $x_i$. The continuous single-layer kernel and the collocation matrix are

$$
G(x,y)=-\frac{1}{2\pi}\log\|x-y\|_2,
\qquad A_{ij}=\int_{\Gamma_j}G(x_i,y)\,ds_y.
$$

For $i\ne j$, the small example uses midpoint quadrature,

$$
A_{ij}\approx -\frac{|\Gamma_j|}{2\pi}\log\|x_i-x_j\|_2.
$$

For the panel containing its own collocation point, it uses the exact straight-panel integral. If $\ell_i=|\Gamma_i|$,

$$
A_{ii}=-\frac{\ell_i}{2\pi}\left(\log\frac{\ell_i}{2}-1\right).
$$

This avoids replacing the diagonal by an arbitrary finite number. It does **not** make the off-diagonal midpoint rule accurate for touching or nearly touching panels; a production BEM solver needs suitable near-singular quadrature.

```python title="docs/examples/bem.py"
--8<-- "docs/examples/bem.py"
```

Run with `python docs/examples/bem.py`.

## Why support boxes matter

For a panel method, the collocation midpoint alone does not describe the physical region occupied by a panel. The call passes

```python
support_lower=np.minimum(start, end),
support_upper=np.maximum(start, end),
```

so each cluster box encloses its entire straight panel. If support boxes are omitted, geometric admissibility may be judged from point sites alone. The storage format still factorizes every non-diagonal block, but the resulting partition and rank profile can differ.

## Separate three error sources

1. **Boundary discretization:** replacing a smooth circle by a polygon and the continuous density by panel values.
2. **Quadrature:** replacing nonself panel integrals by midpoint samples.
3. **Compression:** approximating the already assembled finite matrix by quantized factors.

`reconstruction_bound` addresses only item 3 in exact arithmetic. It cannot certify a continuum BEM solution. The self-panel integral matters to item 2; it is stored in an FP64 diagonal block, not quantized.

!!! tip "Interpreting an apparent lack of memory savings"
    Small BEM examples may have excellent theoretical compression but a C++ resident estimate above dense FP64 because many block objects and offsets remain. Inspect `h.blocks` by admissibility and rank before concluding that the bit allocator is the limiting factor.
