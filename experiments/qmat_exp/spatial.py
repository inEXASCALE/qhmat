from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .matrices import Leaf


@dataclass
class Cluster:
    start: int
    end: int
    lower: np.ndarray
    upper: np.ndarray
    children: tuple


def spatial_partition(points, leaf_size, eta, support_lower=None, support_upper=None):
    """Median cluster tree; BEM boxes enclose full panel supports, not just centers."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0 or leaf_size < 1 or eta <= 0:
        raise ValueError("invalid geometry or clustering parameters")
    lower = points if support_lower is None else np.asarray(support_lower)
    upper = points if support_upper is None else np.asarray(support_upper)
    if lower.shape != points.shape or upper.shape != points.shape or np.any(lower > upper):
        raise ValueError("invalid support boxes")
    order = []

    def cluster(indices):
        start = len(order)
        lo, hi = lower[indices].min(axis=0), upper[indices].max(axis=0)
        if len(indices) <= leaf_size:
            order.extend(indices.tolist())
            children = ()
        else:
            axis = np.argmax(np.ptp(points[indices], axis=0))
            indices = indices[np.argsort(points[indices, axis], kind="stable")]
            mid = len(indices) // 2
            children = (cluster(indices[:mid]), cluster(indices[mid:]))
        return Cluster(start, len(order), lo, hi, children)

    root = cluster(np.arange(len(points)))
    leaves = []

    def partition(a, b):
        gap = np.maximum(0, np.maximum(a.lower - b.upper, b.lower - a.upper))
        separation = float(np.linalg.norm(gap))
        diameter = float(max(np.linalg.norm(a.upper - a.lower), np.linalg.norm(b.upper - b.lower)))
        admissible = separation > 0 and separation >= eta * diameter
        if admissible or (not a.children and not b.children):
            leaves.append(Leaf(a.start, a.end, b.start, b.end, admissible,
                               separation, diameter, "euclidean_bbox"))
            return
        for aa in a.children or (a,):
            for bb in b.children or (b,):
                partition(aa, bb)

    partition(root, root)
    return np.asarray(order), leaves
