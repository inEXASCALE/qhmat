"""Versioned, little-endian input shared by NumPy and the standalone CUDA runner."""
from __future__ import annotations

import struct
import numpy as np
from pathlib import Path

from .matrices import random_orthonormal, singular_values
from .quantization import reserve_greedy_allocation


def read_native_data(path):
    payload = Path(path).read_bytes()
    if len(payload) < 20:
        raise ValueError("truncated native header")
    magic, m, n, rank = struct.unpack("<8sIII", payload[:20])
    if magic != b"QMATGPU1" or min(m, n, rank) == 0:
        raise ValueError("invalid native header")
    if len(payload) != 20 + 4 * rank * (1 + m + n):
        raise ValueError("invalid native data length")
    bits = np.frombuffer(payload, dtype="<u4", count=rank, offset=20).copy()
    p = np.frombuffer(payload, dtype="<f4", count=m * rank, offset=20 + 4 * rank).reshape((m, rank), order="F").copy()
    q = np.frombuffer(payload, dtype="<f4", count=n * rank, offset=20 + 4 * rank * (1 + m)).reshape((n, rank), order="F").copy()
    if not np.isin(bits, [4, 8, 12, 16]).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid native values")
    return bits, p, q


def write_native_data(path, bits, p, q):
    m, rank = p.shape
    n = q.shape[0]
    if q.shape[1] != rank or len(bits) != rank:
        raise ValueError("incompatible native factors")
    with Path(path).open("wb") as f:
        f.write(struct.pack("<8sIII", b"QMATGPU1", m, n, rank))
        f.write(np.asarray(bits, dtype="<u4").tobytes())
        f.write(np.asarray(p, dtype="<f4").tobytes(order="F"))
        f.write(np.asarray(q, dtype="<f4").tobytes(order="F"))


def export_geometric_native(case, cfg, data_dir):
    from .hmatrix import truncated_svd

    if not cfg.get("gpu", {}).get("enabled", False):
        return []
    count = cfg.get("geometric_suites", {}).get("native_leaves_per_case", 2)
    leaves = sorted(enumerate(case["leaves"]), key=lambda pair: -np.prod(pair[1].shape))
    entries, seen = [], set()
    for leaf_id, leaf in leaves:
        if len(entries) >= count:
            break
        candidate = leaf.off_diagonal if cfg["hmatrix"].get("storage_policy") == "diagonal_only" else leaf.low_rank
        if not candidate or min(leaf.shape) < 4:
            continue
        key = tuple(sorted(((leaf.row0, leaf.row1), (leaf.col0, leaf.col1))))
        if key in seen:
            continue
        seen.add(key)
        block = case["matrix"][leaf.row0:leaf.row1, leaf.col0:leaf.col1]
        u, sigma, v, _, eta = truncated_svd(block, cfg["hmatrix"]["rank_tol"], cfg["hmatrix"]["max_rank"])
        budget = cfg["gpu"].get("native_budget_fraction", .002) * float(sigma.sum())
        alloc = reserve_greedy_allocation(sigma, cfg["bit_set"], budget, *block.shape)
        if not alloc.feasible:
            raise ValueError("geometric native allocation infeasible")
        case_id = f"{case['case_id']}_leaf{leaf_id}"
        path = data_dir / f"{case_id}.bin"
        write_native_data(path, alloc.bits, u * np.sqrt(sigma), v * np.sqrt(sigma))
        entries.append(dict(case_id=case_id, path=str(path), m=block.shape[0], n=block.shape[1],
            rank=len(sigma), family=case["family"], source_case=case["case_id"], leaf_id=leaf_id,
            eta=eta, allocation_budget=budget, allocation_cert=alloc.cert,
            bits=alloc.bits.tolist(), budget_normalization="sum_sigma"))
    return entries


def prepare_native_data(cfg, data_dir):
    gpu = cfg.get("gpu", {})
    if not gpu.get("enabled", False):
        return []
    rng = np.random.default_rng(cfg["seed"])
    entries = []
    shapes = gpu.get("native_shapes", [[128, 128], [gpu["m"], gpu["n"]]])
    for m, n in shapes:
        for rank in gpu["ranks"]:
            if rank > min(m, n):
                continue
            sigma = singular_values("algebraic", rank, cfg["low_rank"])
            p = random_orthonormal(m, rank, rng) * np.sqrt(sigma)
            q = random_orthonormal(n, rank, rng) * np.sqrt(sigma)
            budget = gpu.get("native_budget_fraction", 0.002) * float(sigma.sum())
            alloc = reserve_greedy_allocation(sigma, cfg["bit_set"], budget, m, n)
            if not alloc.feasible:
                raise ValueError("native input allocation infeasible; increase native_budget_fraction")
            case_id = f"gpu_m{m}_n{n}_r{rank}_seed{cfg['seed']}"
            path = data_dir / f"{case_id}.bin"
            write_native_data(path, alloc.bits, p, q)
            entries.append(dict(case_id=case_id, path=str(path), m=m, n=n, rank=rank,
                                seed=cfg["seed"], allocation_budget=budget,
                                allocation_cert=alloc.cert, bits=alloc.bits.tolist(),
                                budget_normalization="sum_sigma"))
    return entries
