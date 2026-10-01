"""Gaussian point-kernel matrix with multiple right-hand sides."""

import numpy as np
import qhmat

n = 48
t = np.linspace(0.0, 1.0, n)
points = np.column_stack((t, 0.2 * np.sin(2 * np.pi * t)))
distance_squared = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
length_scale = 0.25
matrix = np.exp(-distance_squared / (2 * length_scale**2))

h = qhmat.HMatrix.from_dense(matrix, points, tolerance=0.2, leaf_size=8)
x = np.column_stack((np.ones(n), np.cos(2 * np.pi * t)))
y = h @ x

print("max application difference:", np.max(np.abs(y - matrix @ x)))
print("max reconstruction difference:", np.max(np.abs(matrix - h.to_dense())))
print("representation bound:", h.reconstruction_bound)
print("storage:", h.storage_report())
