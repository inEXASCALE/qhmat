# Integral operators

An integral operator becomes a finite matrix only after quadrature. `qhmat` compresses **that finite matrix**, not the continuous operator directly. This distinction is central when interpreting error bounds.

## A Nyström discretization

Consider the one-dimensional operator

$$
(Tf)(x)=\int_0^1 e^{-|x-y|}f(y)\,dy.
$$

With midpoint nodes $x_i=(i+\tfrac12)/n$ and uniform weights $w_j=1/n$, the Nyström matrix is

$$
A_{ij}=e^{-|x_i-x_j|}w_j,
\qquad (Tf)(x_i)\approx (Af)_i.
$$

The quadrature weights belong **inside** the matrix being compressed. Otherwise the stored object represents a different discrete operator and any error comparison is misleading.

```python title="docs/examples/integral.py"
--8<-- "docs/examples/integral.py"
```

Run with `python docs/examples/integral.py`.

## What the diagnostics mean

The script prints the difference between the dense Nyström matrix and its hierarchical reconstruction, as well as the difference between dense and hierarchical application to sampled $f(x)=\sin(\pi x)$. Neither is the quadrature error relative to the exact continuum operator.

For a rigorous end-to-end estimate, a separate analysis must combine the quadrature/discretization error with the representation and application-arithmetic terms. `h.reconstruction_bound` supplies only the representation part for the assembled $A$.

## Multiple right-hand sides

Once built, the same operator can act on several sampled functions:

```python
rhs = np.column_stack((
    np.sin(np.pi * nodes),
    np.cos(2 * np.pi * nodes),
    np.ones_like(nodes),
))
y = h @ rhs
print(y.shape)
```

Repeated application can amortize construction cost, but the current package does not claim to beat optimized dense GEMM. Measure the workload you actually care about, including the number of right-hand sides and whether the representation stays resident.
