"""Matched storage comparison for factor-quantization granularity."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .hmatrix import block_budget, budgeted_svd
from .matrices import Leaf, validate_partition
from .quantization import c_value, quantize_vector, reserve_greedy_allocation


METHODS = ("per_tensor", "per_channel", "adaptive")


@dataclass(frozen=True)
class Candidate:
    method: str
    feasible: bool
    bits: np.ndarray
    bound: float
    payload_bits: int
    scale_bits: int
    map_bits: int

    @property
    def total_bits(self) -> int:
        return self.payload_bits + self.scale_bits + self.map_bits


def choose_candidate(u, sigma, v, budget, bit_set, method, scale_width=64):
    """Choose the lowest feasible uniform width or the reserve-greedy assignment."""
    if method not in METHODS:
        raise ValueError(f"unknown granularity: {method}")
    if not bit_set or sorted(set(bit_set)) != sorted(bit_set):
        raise ValueError("bit widths must be distinct")
    m, r = u.shape
    n = v.shape[0]
    if v.shape[1] != r or sigma.shape != (r,) or np.any(sigma < 0):
        raise ValueError("invalid balanced-factor dimensions or spectrum")
    map_width = math.ceil(math.log2(len(bit_set)))
    if r == 0:
        return Candidate(method, budget >= 0, np.zeros(0, dtype=np.int64), 0.0, 0, 0, 0)

    if method == "adaptive":
        alloc = reserve_greedy_allocation(sigma, bit_set, budget, m, n)
        return Candidate(method, alloc.feasible, alloc.bits, alloc.cert,
                         alloc.payload_bits, 2 * r * scale_width, r * map_width)

    weight = float(np.sum(sigma))
    if method == "per_tensor":
        roots = np.sqrt(sigma)
        left_max = float(np.max(np.abs(u * roots)))
        right_max = float(np.max(np.abs(v * roots)))
        weight = r * left_max * right_max
    width = next((b for b in sorted(bit_set) if weight * c_value(b) <= budget), None)
    feasible = width is not None
    width = max(bit_set) if width is None else width
    bits = np.full(r, width, dtype=np.int64)
    return Candidate(method, feasible, bits, weight * c_value(width),
                     (m + n) * r * width,
                     (2 if method == "per_tensor" else 2 * r) * scale_width,
                     map_width)


def decoded_product(u, sigma, v, candidate):
    if len(sigma) == 0:
        return np.zeros((u.shape[0], v.shape[0]), dtype=np.float64)
    roots = np.sqrt(sigma)
    left, right = u * roots, v * roots
    if candidate.method == "per_tensor":
        _, _, left = quantize_vector(left, int(candidate.bits[0]))
        _, _, right = quantize_vector(right, int(candidate.bits[0]))
    else:
        left = np.column_stack([quantize_vector(left[:, j], int(candidate.bits[j]))[2]
                                for j in range(len(sigma))])
        right = np.column_stack([quantize_vector(right[:, j], int(candidate.bits[j]))[2]
                                 for j in range(len(sigma))])
    return left @ right.T


def compare_hmatrix_case(case_id, family, matrix, leaves: list[Leaf], cfg):
    """Hold the partition and tolerance-dependent SVD ranks fixed across methods."""
    hcfg = cfg["hmatrix"]
    if hcfg["storage_policy"] != "diagonal_only":
        raise ValueError("this comparison requires the diagonal-only dense policy")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("expected a finite square matrix")
    validate_partition(len(matrix), leaves)
    if any(not (leaf.diagonal or leaf.off_diagonal) for leaf in leaves):
        raise ValueError("diagonal-crossing block")
    bit_set = hcfg["bit_set"]
    scale_width = cfg["granularity"]["scale_bits"]
    policy = cfg["granularity"]["budget_policy"]
    cache = {}
    full_norm = float(np.max(np.abs(matrix)))
    rows = []
    for tol in cfg["tolerances"]:
        dense_bits = 0
        factor_entries = 0
        offdiag_blocks = 0
        max_rank = 0
        totals = {method: dict(feasible=True, failed_blocks=0, payload_bits=0,
                               scale_bits=0, map_bits=0, global_actual_error=0.0,
                               global_representation_bound=0.0, max_budget_fraction=0.0,
                               weighted_bits=0, retained_components=0,
                               bit_counts=Counter())
                  for method in METHODS}
        for leaf_id, leaf in enumerate(leaves):
            block = matrix[leaf.row0:leaf.row1, leaf.col0:leaf.col1]
            if leaf.diagonal:
                dense_bits += block.size * 64
                continue
            offdiag_blocks += 1
            budget = block_budget(block, full_norm, tol, policy, hcfg["hybrid_gamma"])
            u, sigma, v, _, eta, tail_met = budgeted_svd(
                block, hcfg["rank_tol"], hcfg["max_rank"], budget,
                hcfg["truncation_budget_fraction"], cache, leaf_id)
            rank = len(sigma)
            max_rank = max(max_rank, rank)
            factor_entries += (block.shape[0] + block.shape[1]) * rank
            q_budget = budget * (1 - hcfg["gemm_reserve_fraction"]) - eta
            rank_feasible = tail_met and eta <= hcfg["truncation_budget_fraction"] * budget
            for method in METHODS:
                total = totals[method]
                if not rank_feasible or q_budget < 0:
                    total["feasible"] = False
                    total["failed_blocks"] += 1
                    continue
                candidate = choose_candidate(u, sigma, v, q_budget, bit_set, method, scale_width)
                if not candidate.feasible:
                    total["feasible"] = False
                    total["failed_blocks"] += 1
                    continue
                decoded = decoded_product(u, sigma, v, candidate)
                measured = float(np.max(np.abs(block - decoded)))
                bound = eta + candidate.bound
                slack = 1e-12 * max(1.0, float(np.max(np.abs(block))))
                if measured > bound + slack:
                    raise AssertionError(f"factor quantization bound failed: {case_id}, {method}")
                total["payload_bits"] += candidate.payload_bits
                total["scale_bits"] += candidate.scale_bits
                total["map_bits"] += candidate.map_bits
                total["global_actual_error"] = max(total["global_actual_error"], measured)
                total["global_representation_bound"] = max(total["global_representation_bound"], bound)
                fraction = bound / budget if budget > 0 else 0.0
                total["max_budget_fraction"] = max(total["max_budget_fraction"], fraction)
                total["weighted_bits"] += int(np.sum(candidate.bits))
                total["retained_components"] += rank
                total["bit_counts"].update(int(bit) for bit in candidate.bits)

        fp64_factors = dense_bits + 64 * factor_entries
        for method in METHODS:
            total = totals[method]
            feasible = total["feasible"]
            stored = dense_bits + total["payload_bits"] + total["scale_bits"] + total["map_bits"]
            rows.append(dict(case_id=case_id, family=family, n=len(matrix), tol=tol,
                             budget_policy=policy, method=method, feasible=int(feasible),
                             failed_blocks=total["failed_blocks"], offdiag_blocks=offdiag_blocks,
                             max_rank=max_rank, diagonal_bits=dense_bits,
                             payload_bits=total["payload_bits"] if feasible else "",
                             scale_bits=total["scale_bits"] if feasible else "",
                             map_bits=total["map_bits"] if feasible else "",
                             total_bits=stored if feasible else "",
                             same_partition_fp64_bits=fp64_factors,
                             dense64_bits=matrix.size * 64,
                             storage_fraction_fp64_factors=stored / fp64_factors if feasible else "",
                             compression_vs_dense64=matrix.size * 64 / stored if feasible else "",
                             average_factor_bits=(total["weighted_bits"] / total["retained_components"]
                                                  if feasible and total["retained_components"] else ""),
                             bit_counts=dict(sorted(total["bit_counts"].items())) if feasible else "",
                             global_actual_error=total["global_actual_error"] if feasible else "",
                             global_representation_bound=total["global_representation_bound"] if feasible else "",
                             max_budget_fraction=total["max_budget_fraction"] if feasible else ""))
    return rows
