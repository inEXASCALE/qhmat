from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Leaf:
    row0: int
    row1: int
    col0: int
    col1: int
    low_rank: bool  # Geometric admissibility flag in the prepared partition.
    separation: float = 0.0
    diameter: float = 0.0
    distance_metric: str = "index"

    @property
    def diagonal(self) -> bool:
        return (self.row0, self.row1) == (self.col0, self.col1)

    @property
    def off_diagonal(self) -> bool:
        return self.row1 <= self.col0 or self.col1 <= self.row0

    @property
    def block_kind(self) -> str:
        if self.diagonal:
            return "diagonal"
        return "admissible" if self.low_rank else "offdiagonal_inadmissible"

    @property
    def shape(self) -> tuple[int, int]:
        return self.row1 - self.row0, self.col1 - self.col0

    def to_dict(self) -> dict:
        return asdict(self)


def random_orthonormal(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    return q[:, :cols]


def singular_values(kind: str, rank: int, cfg: dict) -> np.ndarray:
    j = np.arange(rank, dtype=np.float64)
    if kind == "exponential":
        return cfg.get("exponential_decay", 0.82) ** j
    if kind == "algebraic":
        power = cfg.get("algebraic_power", 1.4)
        return (j + 1.0) ** (-power)
    if kind == "flat":
        return np.ones(rank, dtype=np.float64)
    raise ValueError(f"unknown spectrum kind: {kind}")


def synthetic_low_rank_case(
    case_id: str,
    size: int,
    rank: int,
    spectrum: str,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    u = random_orthonormal(size, rank, rng)
    v = random_orthonormal(size, rank, rng)
    sigma = singular_values(spectrum, rank, cfg)
    matrix = (u * sigma) @ v.T
    return {
        "case_id": case_id,
        "size": size,
        "rank": rank,
        "spectrum": spectrum,
        "u": u,
        "v": v,
        "sigma": sigma,
        "matrix": matrix,
    }


def kernel_matrix(points: np.ndarray, kernel: str) -> np.ndarray:
    x = points.reshape(-1, 1)
    r = np.abs(x - x.T)
    h = 1.0 / max(1, points.size - 1)
    if kernel == "laplace1d":
        return 1.0 / (r + 2.0 * h)
    if kernel == "gaussian1d":
        ell = 0.18
        return np.exp(-((r / ell) ** 2))
    if kernel == "matern1d":
        ell = 0.16
        z = np.sqrt(3.0) * r / ell
        return (1.0 + z) * np.exp(-z)
    raise ValueError(f"unknown kernel: {kernel}")


def _interval_diameter(points: np.ndarray, start: int, end: int) -> float:
    if end - start <= 1:
        return 0.0
    return float(points[end - 1] - points[start])


def _interval_separation(
    points: np.ndarray, row0: int, row1: int, col0: int, col1: int
) -> float:
    if row1 <= col0:
        return float(points[col0] - points[row1 - 1])
    if col1 <= row0:
        return float(points[row0] - points[col1 - 1])
    return 0.0


def _split_interval(start: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
    mid = (start + end) // 2
    return (start, mid), (mid, end)


def build_hmatrix_leaves(
    points: np.ndarray,
    leaf_size: int,
    admissibility_eta: float,
) -> list[Leaf]:
    leaves: list[Leaf] = []
    n = points.size

    def rec(row0: int, row1: int, col0: int, col1: int) -> None:
        m = row1 - row0
        ncols = col1 - col0
        diam_r = _interval_diameter(points, row0, row1)
        diam_c = _interval_diameter(points, col0, col1)
        sep = _interval_separation(points, row0, row1, col0, col1)
        admissible = sep >= admissibility_eta * max(diam_r, diam_c) and sep > 0.0
        if admissible:
            leaves.append(Leaf(row0, row1, col0, col1, True, sep, max(diam_r, diam_c), "euclidean_bbox"))
            return
        if m <= leaf_size and ncols <= leaf_size:
            leaves.append(Leaf(row0, row1, col0, col1, False))
            return

        row_parts = [(row0, row1)]
        col_parts = [(col0, col1)]
        if m > leaf_size:
            row_parts = list(_split_interval(row0, row1))
        if ncols > leaf_size:
            col_parts = list(_split_interval(col0, col1))
        for rr0, rr1 in row_parts:
            for cc0, cc1 in col_parts:
                rec(rr0, rr1, cc0, cc1)

    rec(0, n, 0, n)
    return leaves


def hmatrix_case(
    case_id: str,
    n: int,
    kernel: str,
    leaf_size: int,
    admissibility_eta: float,
) -> dict:
    points = np.linspace(0.0, 1.0, n)
    matrix = kernel_matrix(points, kernel)
    leaves = build_hmatrix_leaves(points, leaf_size, admissibility_eta)
    return {
        "case_id": case_id,
        "n": n,
        "kernel": kernel,
        "points": points,
        "matrix": matrix,
        "leaves": [leaf.to_dict() for leaf in leaves],
    }


def load_leaves(items: list[dict]) -> list[Leaf]:
    return [Leaf(**item) for item in items]


def validate_partition(n, leaves):
    # Row intervals avoid an O(N^2) coverage bitmap on large inputs.
    for row in range(n):
        intervals = sorted((l.col0, l.col1) for l in leaves if l.row0 <= row < l.row1)
        end = 0
        for start, stop in intervals:
            if start != end or not start < stop <= n:
                raise ValueError("leaf partition has a gap or overlap")
            end = stop
        if end != n:
            raise ValueError("leaf partition does not cover the matrix")
    if any(not 0 <= l.row0 < l.row1 <= n for l in leaves):
        raise ValueError("invalid leaf rows")
