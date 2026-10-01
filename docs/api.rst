API reference
=============

This page documents the public ``qhmat`` namespace. Examples below use this
small matrix unless a section defines its own inputs:

.. code-block:: python

   import numpy as np
   import qhmat

   points = np.arange(8, dtype=float)[:, None]
   matrix = 1.0 / (1.0 + np.abs(points - points.T))
   h = qhmat.HMatrix.from_dense(matrix, points, tolerance=0.25,
                                leaf_size=2)

High-level matrix
-----------------

.. autoclass:: qhmat.HMatrix

``HMatrix`` owns a native representation; it does not retain the source
NumPy array. Its partition is strict: dense blocks are diagonal only.

.. automethod:: qhmat.HMatrix.from_dense

Quick example: ``qhmat.HMatrix.from_dense(matrix, points, tolerance=0.25)``
constructs from data and point coordinates. Use ``verbose=1`` to print sizes.

.. automethod:: qhmat.HMatrix.from_partition

Quick example: ``partition = qhmat.spatial_partition(points, points, points,
2, 1.0)`` followed by
``qhmat.HMatrix.from_partition(matrix, partition.permutation,
partition.blocks, tolerance=0.25)`` reuses an existing partition.

.. automethod:: qhmat.HMatrix.matmul

Quick example: ``h.matmul(np.ones((8, 2)))`` applies two right-hand sides.
``h @ np.ones(8)`` is the equivalent CPU operator for a single vector.
``device="cuda"`` requires a CUDA-enabled installation.

.. automethod:: qhmat.HMatrix.to_dense

Quick example: ``np.max(np.abs(matrix - h.to_dense()))`` checks the measured
matrix reconstruction difference. This allocates a new dense FP64 array.

.. automethod:: qhmat.HMatrix.storage_report

Quick example: ``h.storage_report().resident_compression`` compares the
estimated C++ resident size with a dense FP64 representation.

.. automethod:: qhmat.HMatrix.save

Quick example: ``h.save("operator.qhmat")`` writes a new file. An existing
file is never overwritten.

.. automethod:: qhmat.HMatrix.load

Quick example: ``loaded = qhmat.HMatrix.load("operator.qhmat")`` validates and
loads it. The input array's ``source_array_bytes`` is not saved.

Matrix properties
~~~~~~~~~~~~~~~~~

``h.shape`` is ``(n, n)``; ``h.blocks`` is a list of :class:`qhmat.BlockInfo`.
``h.logical_bits`` is the packed theoretical representation (including scales,
labels, and dense diagonal), whereas ``h.allocated_bytes`` counts native
vector payloads and ``h.resident_bytes`` also counts C++ object/vector
capacities. ``h.reconstruction_bound`` is a max-norm representation bound in
exact arithmetic, not an application-error or GPU-rounding bound.

Quick example:

.. code-block:: python

   print(h.shape, len(h.blocks), h.reconstruction_bound)
   print(h.logical_bits, h.allocated_bytes, h.resident_bytes)

Storage and errors
------------------

.. autofunction:: qhmat.memory_bytes

Quick example: ``qhmat.memory_bytes(matrix)`` returns NumPy payload bytes;
``qhmat.memory_bytes(h)`` returns an estimated owned C++ size. The latter is
not process RSS and excludes temporary construction workspace.

.. autoclass:: qhmat.StorageReport

``source_array_bytes`` records the original input payload, if known;
``dense_fp64_bytes`` is :math:`8n^2`. ``logical_bits`` and ``logical_bytes``
describe the theoretical storage, ``allocated_bytes`` the native vector
payload, and ``resident_bytes`` the owned C++ size estimate. The two
compression properties divide the FP64 baseline by the corresponding size.

Quick example:

.. code-block:: python

   report = h.storage_report()
   print(report.source_array_bytes, report.dense_fp64_bytes)
   print(report.logical_bytes, report.resident_bytes)
   print(report.logical_compression, report.resident_compression)

.. autoclass:: qhmat.InfeasibleError

Quick example: catch a strict-construction failure without silently replacing
the offending off-diagonal block with dense storage:

.. code-block:: python

   try:
       qhmat.HMatrix.from_dense(matrix, points, tolerance=1e-12,
                                leaf_size=2, bit_widths=(4, 8))
   except qhmat.InfeasibleError as exc:
       print("budget infeasible:", exc)

Allocation and geometry
-----------------------

.. autofunction:: qhmat.coefficient

Returns :math:`c(b)=q_b^{-1}+(4q_b^2)^{-1}` with
:math:`q_b=2^{b-1}-1`. Quick example: ``qhmat.coefficient(8)`` evaluates the
surrogate coefficient for an 8-bit signed integer.

.. autofunction:: qhmat.allocate_bits

For descending singular values :math:`\sigma_j`, a block with :math:`m` rows
and :math:`n` columns receives widths :math:`b_j` under
:math:`\sum_j\sigma_j c(b_j)\le B^{\mathrm q}`. ``payload_bits`` counts
:math:`(m+n)\sum_j b_j` and excludes scales and precision-map metadata.

Quick example:

.. code-block:: python

   alloc = qhmat.allocate_bits(np.array([2.0, 1.0]), [4, 8, 16],
                               budget=0.1, rows=8, cols=8)
   print(alloc.feasible, alloc.bits, alloc.bound, alloc.payload_bits)

.. autoclass:: qhmat.Allocation

An allocation has ``bits``, ``bound``, ``payload_bits``, and ``feasible``.
Quick example: ``qhmat.allocate_bits([2.0, 1.0], [8, 16], 0.1, 8, 8).bits``.

.. autofunction:: qhmat.spatial_partition

Pass arrays of shape ``(n, d)`` for points and their coordinatewise support
bounds. The partition uses a median-split cluster tree; ``leaf_size`` limits
terminal cluster size and ``admissibility`` controls box separation.

Quick example:

.. code-block:: python

   part = qhmat.spatial_partition(points, points, points,
                                  leaf_size=2, admissibility=1.0)
   print(part.permutation, len(part.blocks))

.. autoclass:: qhmat.Partition

``permutation`` maps clustered positions to source indices; ``blocks`` covers
all source matrix entries once in those clustered coordinates. Quick example:
``part = qhmat.spatial_partition(points, points, points, 2, 1.0)`` and
``print(part.permutation)``.

.. autoclass:: qhmat.Rectangle

``Rectangle(row0, row1, col0, col1, admissible=False)`` uses half-open index
ranges in clustered coordinates. ``admissible`` describes geometry, not
whether the stored block is quantized. Quick example:
``qhmat.Rectangle(0, 2, 2, 4, True)``.

.. autoclass:: qhmat.BlockInfo

Each stored block reports its ``rectangle``, ``dense`` flag, retained ``rank``,
``truncation_error``, ``quantization_bound``, ``logical_bits``, and
``allocated_bytes``. Quick example: ``[(b.dense, b.rank) for b in h.blocks]``.

Low-level native interface
--------------------------

.. autoclass:: qhmat.NativeHMatrix

Most users should prefer :meth:`qhmat.HMatrix.from_dense` or
:meth:`qhmat.HMatrix.from_partition`. The native builder expects a complete,
non-overlapping partition; only its diagonal blocks can be added as dense.
``add_low_rank(rect, u, singular_values, v, bits, menu_size,
truncation_error)`` expects unbalanced SVD factors and performs balanced
packing internally. ``finalize()`` validates coverage before application,
reconstruction, or serialization.

Quick example:

.. code-block:: python

   native = qhmat.NativeHMatrix(2, [0, 1])
   native.add_dense(qhmat.Rectangle(0, 1, 0, 1), np.array([[2.0]]))
   native.add_dense(qhmat.Rectangle(1, 2, 1, 2), np.array([[3.0]]))
   for row, col in ((0, 1), (1, 0)):
       native.add_low_rank(qhmat.Rectangle(row, row + 1, col, col + 1),
                           np.ones((1, 1)), np.array([1.0]),
                           np.ones((1, 1)), [8], 1, 0.0)
   native.finalize()
   print(native.matmul(np.ones(2)))

``native.reconstruct()`` returns the dense FP64 matrix; ``native.save(path)``
and ``qhmat.NativeHMatrix.load(path)`` handle persistence. Properties include
``order``, ``blocks``, ``logical_bits``, ``allocated_bytes``,
``resident_bytes``, and ``reconstruction_bound``. ``memory_bytes(native)``
returns the same resident estimate.

Build information
-----------------

``qhmat.cuda_built`` is a boolean set by the compiled extension; it does not
probe whether a GPU is present. Quick example: ``print(qhmat.cuda_built)``.
``qhmat.__version__`` is the Python package version; quick example:
``print(qhmat.__version__)``.
