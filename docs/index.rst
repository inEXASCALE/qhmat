qhmat
=====

Adaptive integer-quantized hierarchical matrices
------------------------------------------------

``qhmat`` builds a hierarchical representation of a dense matrix: diagonal
blocks remain FP64, while admissible and non-diagonal inadmissible blocks are
stored as quantized SVD factors. A block that cannot meet its requested budget
causes construction to fail rather than silently changing storage policy.

The high-level Python API uses NumPy for SVD and a C++17 core for allocation,
packing, storage accounting, reconstruction, and multiplication. A CUDA
backend is optional and is intended for correctness validation, not a claimed
GPU speedup.

.. note::

   The reported reconstruction bound concerns the stored matrix in exact
   arithmetic. It is not an application-error bound for arbitrary right-hand
   sides and does not include all floating-point effects.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   install
   quickstart
   applications
   api
   license
   contributing
