"""Euclidean point kernels and constant-panel collocation single-layer matrices."""
from __future__ import annotations

import numpy as np

from .spatial import spatial_partition


def point_cloud(n, dimension, distribution, seed):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(-0.5, 0.5, size=(n, dimension))
    if distribution == "clustered":
        centers = np.array([[-.3] * dimension, [.3] * dimension])
        return centers[np.arange(n) % 2] + .07 * rng.standard_normal((n, dimension))
    raise ValueError(f"unknown point distribution: {distribution}")


def euclidean_kernel(points, kernel, row_chunk=128):
    n = len(points)
    matrix = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, row_chunk):
        stop = min(n, start + row_chunk)
        r = np.linalg.norm(points[start:stop, None, :] - points[None, :, :], axis=2)
        r[np.arange(stop - start), np.arange(start, stop)] = 1
        if np.any(r <= 0):
            raise ValueError("coincident distinct kernel points")
        if kernel == "inverse_distance":
            block = 1 / r
        elif kernel == "log_distance":
            block = np.log(r)
        else:
            raise ValueError(f"unknown Euclidean kernel: {kernel}")
        block[np.arange(stop - start), np.arange(start, stop)] = 0
        matrix[start:stop] = block
    return matrix


def polygon_panels(n, geometry):
    if n < 4:
        raise ValueError("at least four boundary panels required")
    axes = {"circle": (.5, .5), "ellipse": (.75, .25)}[geometry]
    angles = 2 * np.pi * np.arange(n) / n
    vertices = np.column_stack((axes[0] * np.cos(angles), axes[1] * np.sin(angles)))
    return np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)


def log_segment_integrals(targets, panels):
    """Integral of log(|x-y|) over each straight segment, including on-panel limits."""
    start, end = panels[:, 0], panels[:, 1]
    length = np.linalg.norm(end - start, axis=1)
    if np.any(length <= 0):
        raise ValueError("zero-length panel")
    tangent = (end - start) / length[:, None]
    delta = start[None] - targets[:, None]
    t0 = np.sum(delta * tangent[None], axis=2)
    # A 2D cross product avoids subtracting nearly equal squared distances.
    h = np.abs(delta[..., 0] * tangent[None, :, 1] - delta[..., 1] * tangent[None, :, 0])

    def primitive(t):
        radius = np.hypot(t, h)
        term = np.zeros_like(radius)
        np.multiply(t, np.log(np.maximum(radius, np.finfo(float).tiny)), out=term, where=t != 0)
        return term - t + h * np.arctan2(t, h)

    return primitive(t0 + length[None]) - primitive(t0)


def box_panels(subdivisions, geometry):
    """Axis-aligned rectangular elements on a closed box; N=6*subdivisions**2."""
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    extent = np.array({"cube": (1., 1., 1.), "cuboid": (1.5, 1., .5)}[geometry])
    centers, widths, normals = [], [], []
    for normal in range(3):
        axes = [axis for axis in range(3) if axis != normal]
        grids = [(np.arange(subdivisions) + .5) * extent[k] / subdivisions - extent[k] / 2 for k in axes]
        uu, vv = np.meshgrid(*grids, indexing="ij")
        for side in (-1, 1):
            c = np.zeros((subdivisions**2, 3))
            c[:, normal] = side * extent[normal] / 2
            c[:, axes[0]], c[:, axes[1]] = uu.ravel(), vv.ravel()
            half = np.zeros_like(c)
            half[:, axes] = extent[axes] / (2 * subdivisions)
            centers.append(c)
            widths.append(half)
            normals.extend([normal] * len(c))
    return np.vstack(centers), np.vstack(widths), np.asarray(normals)


def inverse_rectangle_integrals(targets, centers, halfwidths, normals):
    """Analytic integral of 1/r over axis-aligned rectangles, including self terms."""
    values = np.empty((len(targets), len(centers)))
    for normal in range(3):
        indices = np.flatnonzero(normals == normal)
        if len(indices) == 0:
            continue
        axes = [axis for axis in range(3) if axis != normal]
        delta = centers[None, indices] - targets[:, None]
        z = np.abs(delta[..., normal])

        def primitive(u, v):
            radius = np.sqrt(u*u + v*v + z*z)
            # Rationalized log arguments preserve accuracy for negative coordinates.
            def log_plus(a, other):
                direct = radius + a
                with np.errstate(divide="ignore", invalid="ignore"):
                    rationalized = (other*other + z*z) / (radius - a)
                arg = np.where(a < 0, rationalized, direct)
                return np.log(np.maximum(arg, np.finfo(float).tiny))
            return u * log_plus(v, u) + v * log_plus(u, v) - z * np.arctan2(u*v, z*radius)

        result = np.zeros((len(targets), len(indices)))
        for su in (-1, 1):
            for sv in (-1, 1):
                u = delta[..., axes[0]] + su * halfwidths[None, indices, axes[0]]
                v = delta[..., axes[1]] + sv * halfwidths[None, indices, axes[1]]
                result += su * sv * primitive(u, v)
        values[:, indices] = result
    return values


def geometric_cases(cfg, h_cfg, seed):
    """Yield one assembled matrix at a time to bound peak dense-matrix memory."""
    leaf_size = cfg.get("leaf_size", h_cfg["leaf_size"])
    eta = h_cfg["admissibility_eta"]
    kernels = cfg.get("point_kernels", {})
    for dimension in kernels.get("dimensions", []):
        for distribution in kernels["distributions"]:
            for n in kernels["sizes"]:
                points = point_cloud(n, dimension, distribution, seed)
                permutation, leaves = spatial_partition(points, leaf_size, eta)
                points = points[permutation]
                for kernel in kernels["kernels"]:
                    yield dict(case_id=f"kernel_{kernel}_{dimension}d_{distribution}_n{n}",
                        family="point_kernel", geometry=f"{dimension}d_{distribution}", kernel=kernel,
                        matrix=euclidean_kernel(points, kernel), points=points,
                        permutation=permutation, leaves=leaves,
                        metadata=dict(dimension=dimension, distribution=distribution, seed=seed,
                            distance="euclidean_l2", diagonal="zero_omitted_self_interaction",
                            off_diagonal_regularization="none"))
    bem2d = cfg.get("bem2d", {})
    for geometry in bem2d.get("geometries", []):
        for n in bem2d["sizes"]:
            panels = polygon_panels(n, geometry)
            points = panels.mean(axis=1)
            permutation, leaves = spatial_partition(points, leaf_size, eta, panels.min(axis=1), panels.max(axis=1))
            panels, points = panels[permutation], points[permutation]
            matrix = np.empty((n, n))
            for start in range(0, n, 128):
                matrix[start:start+128] = -log_segment_integrals(points[start:start+128], panels) / (2*np.pi)
            lengths = np.linalg.norm(panels[:, 1] - panels[:, 0], axis=1)
            np.fill_diagonal(matrix, -lengths * (np.log(lengths / 2) - 1) / (2*np.pi))
            yield dict(case_id=f"bem2d_log_{geometry}_n{n}", family="bem2d", geometry=geometry,
                kernel="minus_log_over_2pi", matrix=matrix, points=points, permutation=permutation,
                leaves=leaves, panels=panels, weights=lengths,
                metadata=dict(dimension=2, discretization="constant_panel_midpoint_collocation",
                    assembly="analytic_segment_integrals", diagonal="analytic_self_integral",
                    boundary="polygonal_approximation", distance="euclidean_l2"))
    bem3d = cfg.get("bem3d", {})
    for geometry in bem3d.get("geometries", []):
        for subdivision in bem3d["subdivisions"]:
            points, widths, normals = box_panels(subdivision, geometry)
            permutation, leaves = spatial_partition(points, bem3d.get("leaf_size", leaf_size), eta, points-widths, points+widths)
            points, widths, normals = points[permutation], widths[permutation], normals[permutation]
            n = len(points)
            matrix = np.empty((n, n))
            for start in range(0, n, 128):
                matrix[start:start+128] = inverse_rectangle_integrals(points[start:start+128], points, widths, normals) / (4*np.pi)
            yield dict(case_id=f"bem3d_inverse_{geometry}_n{n}", family="bem3d", geometry=geometry,
                kernel="inverse_over_4pi", matrix=matrix, points=points, permutation=permutation,
                halfwidths=widths, normals=normals, leaves=leaves,
                weights=np.prod(np.where(widths > 0, 2*widths, 1), axis=1),
                metadata=dict(dimension=3, discretization="constant_panel_midpoint_collocation",
                    assembly="analytic_rectangle_integrals", diagonal="analytic_self_integral",
                    subdivisions=subdivision, boundary="closed_box", distance="euclidean_l2"))
