"""Adaptive integer-quantized hierarchical matrices."""

from ._core import Allocation, BlockInfo, Partition, Rectangle, coefficient
from ._core import allocate as allocate_bits
from ._core import spatial_partition
from ._core import NativeHMatrix
from ._core import cuda_built
from .matrix import HMatrix, InfeasibleError, StorageReport, memory_bytes

__version__ = "0.1.0"

__all__ = [
    "Allocation",
    "BlockInfo",
    "HMatrix",
    "InfeasibleError",
    "NativeHMatrix",
    "Partition",
    "Rectangle",
    "StorageReport",
    "allocate_bits",
    "coefficient",
    "cuda_built",
    "memory_bytes",
    "spatial_partition",
]
