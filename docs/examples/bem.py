"""Midpoint-panel discretization of the planar Laplace single-layer kernel."""

import numpy as np
import qhmat

n = 48
theta = 2 * np.pi * np.arange(n + 1) / n
vertices = np.column_stack((np.cos(theta), np.sin(theta)))
start, end = vertices[:-1], vertices[1:]
centers = (start + end) / 2
lengths = np.linalg.norm(end - start, axis=1)

# Off-diagonal midpoint quadrature for -(1 / (2*pi)) * log(|x-y|).
distance = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
np.fill_diagonal(distance, 1.0)
matrix = -np.log(distance) * lengths[None, :] / (2 * np.pi)

# Exact self integral for a straight panel of length ell, collocated at its midpoint.
np.fill_diagonal(matrix, -lengths * (np.log(lengths / 2) - 1) / (2 * np.pi))

h = qhmat.HMatrix.from_dense(
    matrix, centers, tolerance=0.25, leaf_size=6,
    support_lower=np.minimum(start, end),
    support_upper=np.maximum(start, end),
)
density = np.cos(theta[:-1])
potential = h @ density

print("max application difference:", np.max(np.abs(potential - matrix @ density)))
print("max reconstruction difference:", np.max(np.abs(matrix - h.to_dense())))
print("representation bound:", h.reconstruction_bound)
print("storage:", h.storage_report())
