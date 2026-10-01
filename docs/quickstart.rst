Quickstart
==========

Construct a matrix
------------------

This example samples the smooth kernel

.. math::

   A_{ij}=\frac{1}{|x_i-x_j|+0.4},\qquad x_i=\frac{i}{63}.

.. code-block:: python

   import numpy as np
   import qhmat

   points = np.linspace(0.0, 1.0, 64)[:, None]
   matrix = 1.0 / (np.abs(points - points.T) + 0.4)
   h = qhmat.HMatrix.from_dense(matrix, points, tolerance=0.25,
                                leaf_size=8, verbose=1)

The relative tolerance gives each non-diagonal block :math:`A_k` a
max-norm budget :math:`\varepsilon\|A_k\|_{\max}`. Diagonal blocks stay
dense in FP64; other blocks are factorized even when geometrically
inadmissible. Tight settings may raise :class:`qhmat.InfeasibleError`.

Apply and inspect
-----------------

.. code-block:: python

   x = np.ones((64, 3))
   y = h @ x                       # CPU, three right-hand sides
   reconstructed = h.to_dense()    # Validation only: allocates a dense array
   print(np.max(np.abs(matrix - reconstructed)))
   print(h.reconstruction_bound)
   print(h.storage_report())
   print(qhmat.memory_bytes(matrix), qhmat.memory_bytes(h))

``reconstruction_bound`` controls the stored representation in exact
arithmetic. The displayed application difference ``y - matrix @ x`` is a
different quantity; it need not lie below that same bound.

``logical_bits`` counts the paper-style packed payload, scales, labels and
FP64 diagonal entries. ``resident_bytes`` estimates the in-memory C++ object
and vector capacities. Neither includes the input array or temporary SVD
workspace. A retained input array still occupies memory after construction.

Persist and reload
------------------

.. code-block:: python

   h.save("operator.qhmat")
   restored = qhmat.HMatrix.load("operator.qhmat")
   print(np.max(np.abs(restored @ x - y)))

``save`` refuses to overwrite an existing path. The file format is versioned
and checksummed, but the checksum is for accidental corruption detection, not
security against malicious files.
