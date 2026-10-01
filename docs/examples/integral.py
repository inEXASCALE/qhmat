"""Nyström midpoint discretization of an exponential integral operator."""

import numpy as np
import qhmat

n = 64
nodes = (np.arange(n) + 0.5) / n
weights = np.full(n, 1.0 / n)
points = nodes[:, None]
matrix = np.exp(-np.abs(nodes[:, None] - nodes[None, :])) * weights[None, :]

h = qhmat.HMatrix.from_dense(matrix, points, tolerance=0.2, leaf_size=8)
samples = np.sin(np.pi * nodes)
approximation = h @ samples

print("max application difference:", np.max(np.abs(approximation - matrix @ samples)))
print("max reconstruction difference:", np.max(np.abs(matrix - h.to_dense())))
print("representation bound:", h.reconstruction_bound)
print("storage:", h.storage_report())
