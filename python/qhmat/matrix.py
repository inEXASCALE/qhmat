"""High-level construction around the native quantized matrix representation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os

import numpy as np

from ._core import NativeHMatrix, Rectangle, allocate, spatial_partition


class InfeasibleError(ValueError):
    """The requested strict diagonal-only representation exceeds its budget."""


@dataclass(frozen=True)
class StorageReport:
    """Sizes of a stored representation, excluding construction workspace."""

    dense_fp64_bytes: int
    logical_bits: int
    allocated_bytes: int
    resident_bytes: int
    source_array_bytes: int | None = None

    @property
    def logical_bytes(self) -> int:
        """Logical storage rounded up to whole bytes."""
        return (self.logical_bits + 7) // 8

    @property
    def logical_compression(self) -> float:
        return self.dense_fp64_bytes / (self.logical_bits / 8)

    @property
    def resident_compression(self) -> float:
        return self.dense_fp64_bytes / self.resident_bytes


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    raise AssertionError("unreachable")


def _print_storage_report(report: StorageReport) -> None:
    if report.source_array_bytes is not None:
        print(f"Input array: {_format_bytes(report.source_array_bytes)}")
    print(f"Dense FP64 baseline: {_format_bytes(report.dense_fp64_bytes)}")
    print(f"H-matrix logical storage: {report.logical_bits} bits "
          f"({report.logical_compression:.2f}x compression vs FP64)")
    print(f"H-matrix C++ resident estimate: {_format_bytes(report.resident_bytes)} "
          f"({report.resident_compression:.2f}x compression vs FP64)")


class HMatrix:
    """A diagonal-dense, off-diagonal quantized hierarchical matrix.

    Construction uses NumPy's LAPACK-backed SVD. Packing, allocation, storage
    accounting, reconstruction, and multiplication are implemented in C++.
    The reported bound concerns the saved matrix in exact arithmetic; it does
    not include floating-point error in ``matmul`` or ``to_dense``.
    """

    def __init__(self, native: NativeHMatrix, source_array_bytes: int | None = None):
        self._native = native
        self._source_array_bytes = source_array_bytes

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        points: np.ndarray,
        *,
        tolerance: float,
        leaf_size: int = 32,
        admissibility: float = 1.0,
        bit_widths: Sequence[int] = (4, 8, 12, 16, 24, 32),
        rank_tolerance: float = 0.1,
        max_rank: int | None = None,
        truncation_fraction: float = 0.25,
        arithmetic_reserve: float = 0.02,
        support_lower: np.ndarray | None = None,
        support_upper: np.ndarray | None = None,
        verbose: bool = False,
    ) -> HMatrix:
        """Partition a dense matrix and quantize every non-diagonal block.

        ``points[i]`` describes row and column ``i``; support boxes may be
        supplied for BEM panels. ``tolerance`` is relative to each block's
        max norm. No non-diagonal dense fallback is permitted. ``verbose=1``
        prints a storage summary after successful construction.
        """
        a = np.asarray(matrix, dtype=np.float64)
        p = np.asarray(points, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError("matrix must be nonempty and square")
        if p.ndim != 2 or p.shape[0] != a.shape[0] or p.shape[1] == 0:
            raise ValueError("points must have shape (matrix order, dimension)")
        if not np.isfinite(a).all() or not np.isfinite(p).all():
            raise ValueError("matrix and points must be finite")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not isinstance(leaf_size, int) or leaf_size < 1:
            raise ValueError("leaf_size must be a positive integer")
        if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance <= 1:
            raise ValueError("rank_tolerance must lie in (0, 1]")
        if max_rank is not None and (not isinstance(max_rank, int) or max_rank < 1):
            raise ValueError("max_rank must be a positive integer or None")
        if not np.isfinite(truncation_fraction) or not np.isfinite(arithmetic_reserve):
            raise ValueError("budget fractions must be finite")
        if not 0 < truncation_fraction < 1 - arithmetic_reserve or not 0 <= arithmetic_reserve < 1:
            raise ValueError("invalid truncation/arithmetic budget fractions")
        widths = tuple(bit_widths)
        if not widths or any(type(b) is not int or not 2 <= b <= 32 for b in widths) or len(set(widths)) != len(widths):
            raise ValueError("bit_widths must contain distinct integers in [2, 32]")

        lo = p if support_lower is None else np.asarray(support_lower, dtype=np.float64)
        hi = p if support_upper is None else np.asarray(support_upper, dtype=np.float64)
        partition = spatial_partition(p, lo, hi, leaf_size, admissibility)
        result = cls.from_partition(
            a, partition.permutation, partition.blocks, tolerance=tolerance,
            bit_widths=widths, rank_tolerance=rank_tolerance, max_rank=max_rank,
            truncation_fraction=truncation_fraction,
            arithmetic_reserve=arithmetic_reserve,
        )
        result._source_array_bytes = int(np.asarray(matrix).nbytes)
        if verbose:
            _print_storage_report(result.storage_report())
        return result

    @classmethod
    def from_partition(
        cls,
        matrix: np.ndarray,
        permutation: Sequence[int],
        blocks: Sequence[Rectangle],
        *,
        tolerance: float,
        bit_widths: Sequence[int] = (4, 8, 12, 16, 24, 32),
        rank_tolerance: float = 0.1,
        max_rank: int | None = None,
        truncation_fraction: float = 0.25,
        arithmetic_reserve: float = 0.02,
        verbose: bool = False,
    ) -> HMatrix:
        """Build from a disjoint partition; ``verbose=1`` prints storage."""
        a = np.asarray(matrix, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
            raise ValueError("matrix must be nonempty and square")
        if not np.isfinite(a).all():
            raise ValueError("matrix must be finite")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance <= 1:
            raise ValueError("rank_tolerance must lie in (0, 1]")
        if max_rank is not None and (not isinstance(max_rank, int) or max_rank < 1):
            raise ValueError("max_rank must be a positive integer or None")
        if not np.isfinite(truncation_fraction) or not np.isfinite(arithmetic_reserve):
            raise ValueError("budget fractions must be finite")
        if not 0 < truncation_fraction < 1 - arithmetic_reserve or not 0 <= arithmetic_reserve < 1:
            raise ValueError("invalid truncation/arithmetic budget fractions")
        widths = tuple(bit_widths)
        if not widths or any(type(b) is not int or not 2 <= b <= 32 for b in widths) or len(set(widths)) != len(widths):
            raise ValueError("bit_widths must contain distinct integers in [2, 32]")
        indices = tuple(permutation)
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) for i in indices):
            raise ValueError("permutation must contain integer indices")
        perm = tuple(int(i) for i in indices)
        if len(perm) != a.shape[0] or set(perm) != set(range(a.shape[0])):
            raise ValueError("permutation must contain each index exactly once")
        native = NativeHMatrix(a.shape[0], perm)
        for rect in blocks:
            if not isinstance(rect, Rectangle):
                raise TypeError("blocks must contain Rectangle objects")
            block = np.ascontiguousarray(
                a[np.ix_(perm[rect.row0:rect.row1], perm[rect.col0:rect.col1])]
            )
            if block.size == 0:
                raise ValueError("empty block in partition")
            if rect.row0 == rect.col0 and rect.row1 == rect.col1:
                native.add_dense(rect, block)
                continue
            if not (rect.row1 <= rect.col0 or rect.col1 <= rect.row0):
                raise ValueError("non-diagonal block intersects the diagonal")
            budget = tolerance * float(np.max(np.abs(block)))
            if budget == 0.0:
                native.add_low_rank(
                    rect, np.zeros((block.shape[0], 0)), np.empty(0),
                    np.zeros((block.shape[1], 0)), [], len(widths), 0.0,
                )
                continue
            u, sigma, vt = np.linalg.svd(block, full_matrices=False)
            tail = np.sqrt(np.r_[np.cumsum((sigma[::-1] ** 2))[::-1], 0.0])
            target = min(rank_tolerance * tail[0], truncation_fraction * budget)
            cap = sigma.size if max_rank is None else min(max_rank, sigma.size)
            rank = next((r for r in range(1, cap + 1) if tail[r] <= target), None)
            if rank is None:
                raise InfeasibleError(
                    f"block ({rect.row0}:{rect.row1}, {rect.col0}:{rect.col1}) "
                    "cannot satisfy the truncation allowance"
                )
            rank = min(rank, int(np.count_nonzero(sigma > 0)))
            sigma = sigma[:rank]
            u = u[:, :rank]
            v = vt[:rank, :].T
            residual = float(np.max(np.abs(block - (u * sigma) @ v.T)))
            q_budget = budget * (1 - arithmetic_reserve) - residual
            if residual > truncation_fraction * budget or q_budget < 0:
                raise InfeasibleError("measured truncation error exceeds the block allowance")
            allocation = allocate(sigma, widths, q_budget, *block.shape)
            if not allocation.feasible:
                raise InfeasibleError(
                    f"block ({rect.row0}:{rect.row1}, {rect.col0}:{rect.col1}) "
                    "is infeasible even at the highest bit width"
                )
            native.add_low_rank(rect, u, sigma, v, allocation.bits, len(widths), residual)
        native.finalize()
        result = cls(native, source_array_bytes=int(np.asarray(matrix).nbytes))
        if verbose:
            _print_storage_report(result.storage_report())
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return self._native.order, self._native.order

    @property
    def blocks(self):
        return self._native.blocks

    @property
    def logical_bits(self) -> int:
        """Packed payload, two FP64 scales, and direct precision labels."""
        return self._native.logical_bits

    @property
    def allocated_bytes(self) -> int:
        """Native vector payload bytes, excluding container and tree overhead."""
        return self._native.allocated_bytes

    @property
    def resident_bytes(self) -> int:
        """Estimated owned C++ bytes, including vector capacities and objects.

        Excludes allocator overhead, the Python wrapper, construction workspace,
        source arrays, and any CUDA allocations. It is not process RSS.
        """
        return self._native.resident_bytes

    def storage_report(self) -> StorageReport:
        """Compare the stored representation with a dense FP64 matrix."""
        order = self._native.order
        return StorageReport(
            dense_fp64_bytes=order * order * np.dtype(np.float64).itemsize,
            logical_bits=self.logical_bits,
            allocated_bytes=self.allocated_bytes,
            resident_bytes=self.resident_bytes,
            source_array_bytes=self._source_array_bytes,
        )

    @property
    def reconstruction_bound(self) -> float:
        """Max-norm bound for the stored matrix in exact arithmetic."""
        return self._native.reconstruction_bound

    def matmul(self, x: np.ndarray, *, device: str = "cpu") -> np.ndarray:
        """Apply the quantized matrix to one or more right-hand sides."""
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        return self._native.matmul(np.asarray(x, dtype=np.float64), device == "cuda")

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matmul(x)

    def to_dense(self) -> np.ndarray:
        """Reconstruct a dense FP64 matrix for inspection or validation."""
        return self._native.reconstruct()

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a versioned packed representation; existing files are not replaced."""
        self._native.save(os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> HMatrix:
        """Load and validate a saved packed representation."""
        return cls(NativeHMatrix.load(os.fspath(path)))


def memory_bytes(value: np.ndarray | HMatrix | NativeHMatrix) -> int:
    """Return array payload bytes or estimated owned C++ H-matrix bytes.

    Array views count their visible elements, not the full backing allocation.
    H-matrix estimates exclude Python, allocator, and temporary workspace costs.
    """
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, HMatrix):
        return value.resident_bytes
    if isinstance(value, NativeHMatrix):
        return value.resident_bytes
    raise TypeError("memory_bytes expects a NumPy array or qhmat H-matrix")
