from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .matrices import Leaf, validate_partition
from .storage import same_partition_storage
from .quantization import (
    all_allocation_methods,
    cert_from_bits,
    channel_bit_counts,
    channel_gemm_certificate,
    geometry_cert,
    reconstruct_quantized,
    reserve_greedy_allocation,
    realized_gemm_certificate,
)


@dataclass
class LeafResult:
    case_id: str
    tol: float
    budget_policy: str
    leaf_id: int
    row0: int
    row1: int
    col0: int
    col1: int
    low_rank: bool
    rank: int
    eta: float
    budget: float
    block_norm: float
    avg_bits: float
    separation_index: int
    diameter_index: int
    q_cert: float
    r_cert: float
    leaf_cert: float
    actual_error: float
    payload_bits: int
    scale_bits: int
    map_bits: int
    bit_counts: dict[str, int]
    feasible: bool
    fallback_reason: str = ""
    separation: float = 0.0
    diameter: float = 0.0
    distance_metric: str = "index"


def truncated_svd(block: np.ndarray, rank_tol: float, max_rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    u, sigma, vt = np.linalg.svd(block, full_matrices=False)
    max_rank = sigma.size if max_rank == 0 else max_rank
    if sigma.size == 0:
        return u[:, :0], sigma[:0], vt[:0, :].T, np.zeros_like(block), 0.0
    total_energy = float(np.sum(sigma * sigma))
    tail = np.concatenate([np.cumsum((sigma[::-1] ** 2))[::-1], [0.0]])
    rank = min(max_rank, sigma.size)
    for r in range(1, min(max_rank, sigma.size) + 1):
        rel_tail = np.sqrt(tail[r] / total_energy) if total_energy > 0.0 else 0.0
        if rel_tail <= rank_tol:
            rank = r
            break
    u_r = u[:, :rank]
    s_r = sigma[:rank]
    v_r = vt[:rank, :].T
    approx = (u_r * s_r) @ v_r.T
    eta = float(np.max(np.abs(block - approx)))
    return u_r, s_r, v_r, approx, eta


def block_budget(block: np.ndarray, full_norm: float, tol: float, policy: str, hybrid_gamma: float) -> float:
    if policy == "global_relative":
        return tol * full_norm
    if policy == "block_relative":
        return tol * float(np.max(np.abs(block)))
    if policy == "hybrid":
        return tol * max(float(np.max(np.abs(block))), hybrid_gamma * full_norm)
    raise ValueError(f"unknown budget policy: {policy}")


def leaf_index_geometry(leaf: Leaf) -> tuple[int, int]:
    if leaf.row1 <= leaf.col0:
        separation = leaf.col0 - leaf.row1
    elif leaf.col1 <= leaf.row0:
        separation = leaf.row0 - leaf.col1
    else:
        separation = 0
    diameter = max(leaf.row1 - leaf.row0, leaf.col1 - leaf.col0)
    return separation, diameter


def budgeted_svd(block, rank_tol, max_rank, budget, truncation_fraction, cache, leaf_id):
    """Select a rank using a Frobenius tail sufficient for the max-norm budget.

    Cache the full SVD, not a tolerance-dependent truncation. A rank cap is a
    resource limit, not permission to switch an off-diagonal block to dense.
    """
    key = ("full_svd", leaf_id)
    if key not in cache:
        u, sigma, vt = np.linalg.svd(block, full_matrices=False)
        cache[key] = u, sigma, vt.T
    u, sigma, v = cache[key]
    tail = np.sqrt(np.r_[np.cumsum((sigma[::-1] ** 2))[::-1], 0.0])
    target = min(rank_tol * tail[0], truncation_fraction * budget)
    cap = len(sigma) if max_rank == 0 else min(max_rank, len(sigma))
    rank = next((r for r in range(1, cap + 1) if tail[r] <= target), cap)
    ur, sr, vr = u[:, :rank], sigma[:rank], v[:, :rank]
    approx = (ur * sr) @ vr.T
    eta = float(np.max(np.abs(block - approx)))
    return ur, sr, vr, approx, eta, bool(tail[rank] <= target)


def run_hmatrix_case(
    case_id: str,
    matrix: np.ndarray,
    leaves: list[Leaf],
    tolerances: list[float],
    bit_set: list[int],
    rank_tol: float,
    max_rank: int,
    budget_policy: str,
    hybrid_gamma: float,
    gemm_reserve_fraction: float,
    unit_roundoff: float,
    scale_bits: int = 64,
    factorizations: dict | None = None,
    storage_policy: str = "diagonal_only",
    truncation_budget_fraction: float = 0.25,
    representations: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    if storage_policy not in ("diagonal_only", "admissible_only"):
        raise ValueError("unknown storage_policy")
    strict = storage_policy == "diagonal_only"
    if strict:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("diagonal_only requires a square, commonly indexed matrix")
        if not np.isfinite(matrix).all() or not 0 < rank_tol <= 1 or max_rank < 0:
            raise ValueError("invalid matrix/rank parameters")
        if not 0 < truncation_budget_fraction < 1 - gemm_reserve_fraction:
            raise ValueError("invalid truncation/arithmetic budget fractions")
        if not 0 <= gemm_reserve_fraction < 1 or any(not np.isfinite(t) or t <= 0 for t in tolerances):
            raise ValueError("invalid tolerance or arithmetic reserve")
        if not bit_set or len(set(bit_set)) != len(bit_set) or any(type(b) is not int or not 2 <= b <= 32 for b in bit_set):
            raise ValueError("integer precision menu must contain distinct widths in [2,32]")
        if any(not (leaf.diagonal or leaf.off_diagonal) for leaf in leaves):
            raise ValueError("leaf intersects the diagonal without being a diagonal block")
        validate_partition(len(matrix), leaves)
    full_norm = float(np.max(np.abs(matrix)))
    dtypes = {float(np.finfo(t).eps / 2): t for t in (np.float32, np.float64)}
    if unit_roundoff not in dtypes:
        raise ValueError("accumulator_unit_roundoff must describe float32 or float64")
    accumulation_dtype = dtypes[unit_roundoff]
    summary_rows: list[dict] = []
    leaf_rows: list[dict] = []
    factorizations = {} if factorizations is None else factorizations

    for tol in tolerances:
        approx_full = np.zeros_like(matrix)
        leaf_results: list[LeafResult] = []
        for leaf_id, leaf in enumerate(leaves):
            fallback_reason = ""
            block = matrix[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1]
            block_norm = float(np.max(np.abs(block))) if block.size else 0.0
            separation_index, diameter_index = leaf_index_geometry(leaf)
            compress = leaf.off_diagonal if strict else leaf.low_rank
            if not compress:
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = block
                leaf_results.append(
                    LeafResult(
                        case_id,
                        tol,
                        budget_policy,
                        leaf_id,
                        leaf.row0,
                        leaf.row1,
                        leaf.col0,
                        leaf.col1,
                        False,
                        0,
                        0.0,
                        0.0,
                        block_norm,
                        64.0,
                        separation_index,
                        diameter_index,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        block.size * 64,
                        0,
                        0,
                        {},
                        True,
                        separation=leaf.separation,
                        diameter=leaf.diameter,
                        distance_metric=leaf.distance_metric,
                    )
                )
                continue

            b_tot = block_budget(block, full_norm, tol, budget_policy, hybrid_gamma)
            if strict:
                u, sigma, v, low_rank_block, eta, tail_target_met = budgeted_svd(
                    block, rank_tol, max_rank, b_tot, truncation_budget_fraction,
                    factorizations, leaf_id)
            else:
                if leaf_id not in factorizations:
                    factorizations[leaf_id] = truncated_svd(block, rank_tol, max_rank)
                u, sigma, v, low_rank_block, eta = factorizations[leaf_id]
            b_gemm = gemm_reserve_fraction * b_tot
            b_q = b_tot - eta - b_gemm
            if b_q < 0.0 or (strict and (not tail_target_met or not np.isfinite(eta)
                                       or eta > truncation_budget_fraction * b_tot)):
                fallback_reason = "truncation_budget"
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = block
                bits = np.full(sigma.shape, max(bit_set), dtype=np.int64)
                q_cert = 0.0
                r_cert = 0.0
                payload_bits = block.size * 64
                scale_payload = 0
                map_payload = 0
                bit_counts = {}
                actual = 0.0
                feasible = False
            elif sigma.size == 0:
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = low_rank_block
                bits = np.zeros(0, dtype=np.int64)
                q_cert = 0.0
                r_cert = 0.0
                payload_bits = 0
                scale_payload = 0
                map_payload = 0
                bit_counts = {}
                actual = float(np.max(np.abs(block - low_rank_block)))
                feasible = True
            else:
                alloc = reserve_greedy_allocation(sigma, bit_set, b_q, block.shape[0], block.shape[1])
                bits = alloc.bits
                exact_hat, tilde = reconstruct_quantized(u, sigma, v, bits, accumulation_dtype)
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = tilde
                q_cert = alloc.cert
                r_cert = realized_gemm_certificate(sigma, bits, bit_set, unit_roundoff)
                payload_bits = alloc.payload_bits
                scale_payload = int(2 * sigma.size * scale_bits)
                map_payload = int(sigma.size * math.ceil(math.log2(len(bit_set)))) if sigma.size else 0
                bit_counts = channel_bit_counts(bits, bit_set)
                actual = float(np.max(np.abs(block - tilde)))
                feasible = alloc.feasible and r_cert <= b_gemm + 1e-30
                if not feasible:
                    fallback_reason = "quantization_budget" if not alloc.feasible else "arithmetic_reserve"

            if not feasible and not strict:
                # A failed compressed candidate must not remain in the returned matrix.
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = block
                eta = q_cert = r_cert = actual = 0.0
                payload_bits, scale_payload, map_payload = block.size * 64, 0, 0
                bits, bit_counts = np.zeros(0, dtype=np.int64), {}
            elif not feasible:
                # An infeasible candidate is not a stored representation. Never
                # report its dense replacement as a successful strict-policy run.
                approx_full[leaf.row0 : leaf.row1, leaf.col0 : leaf.col1] = np.nan
                q_cert = r_cert = actual = float("nan")
                payload_bits = scale_payload = map_payload = 0
                bits, bit_counts = np.zeros(0, dtype=np.int64), {}
            elif representations is not None:
                representations[(tol, leaf_id)] = (u, sigma, v, eta, bits.copy())

            leaf_results.append(
                LeafResult(
                    case_id=case_id,
                    tol=tol,
                    budget_policy=budget_policy,
                    leaf_id=leaf_id,
                    row0=leaf.row0,
                    row1=leaf.row1,
                    col0=leaf.col0,
                    col1=leaf.col1,
                    low_rank=True,
                    rank=int(sigma.size),
                    eta=eta,
                    budget=b_tot,
                    block_norm=block_norm,
                    avg_bits=float(np.mean(bits)) if bits.size else 0.0,
                    separation_index=separation_index,
                    diameter_index=diameter_index,
                    q_cert=q_cert,
                    r_cert=r_cert,
                    leaf_cert=eta + q_cert + r_cert,
                    actual_error=actual,
                    payload_bits=payload_bits,
                    scale_bits=scale_payload,
                    map_bits=map_payload,
                    bit_counts=bit_counts,
                    feasible=feasible,
                    fallback_reason=fallback_reason,
                    separation=leaf.separation,
                    diameter=leaf.diameter,
                    distance_metric=leaf.distance_metric,
                )
            )

        global_actual = float(np.max(np.abs(matrix - approx_full)))
        low_rank_results = [item for item in leaf_results if item.low_rank]
        dense_results = [item for item in leaf_results if not item.low_rank]
        fallback_dense_results = [item for item in low_rank_results if not item.feasible]
        certified_results = [item for item in low_rank_results if item.feasible]
        global_cert = max((item.leaf_cert for item in certified_results), default=0.0)
        dominant_leaf = max(certified_results, key=lambda item: item.leaf_cert, default=None)
        mean_low_rank_avg_bits = float(np.mean([item.avg_bits for item in certified_results])) if certified_results else 0.0
        payload_bits = sum(item.payload_bits for item in leaf_results)
        scale_bits_total = sum(item.scale_bits for item in leaf_results)
        map_bits_total = sum(item.map_bits for item in leaf_results)
        baseline_dense_bits = matrix.size * 64
        summary_rows.append(
            {
                "case_id": case_id,
                "tol": tol,
                "budget_policy": budget_policy,
                "global_actual_error": global_actual,
                "global_cert": global_cert,
                "cert_over_actual": global_cert / global_actual if global_actual > 0.0 else "",
                "payload_bits": payload_bits,
                "scale_bits": scale_bits_total,
                "map_bits": map_bits_total,
                "total_bits_with_metadata": payload_bits + scale_bits_total + map_bits_total,
                "dense64_bits": baseline_dense_bits,
                "payload_compression": baseline_dense_bits / payload_bits if payload_bits else "",
                "total_compression": baseline_dense_bits / (payload_bits + scale_bits_total + map_bits_total) if payload_bits + scale_bits_total + map_bits_total else "",
                "low_rank_leaves": len(low_rank_results),
                "dense_leaves": len(dense_results),
                "feasible_low_rank_leaves": sum(1 for item in low_rank_results if item.feasible),
                "fallback_dense_leaves": len(fallback_dense_results),
                "mean_low_rank_avg_bits": mean_low_rank_avg_bits,
                "max_leaf_actual": max((item.actual_error for item in low_rank_results), default=0.0),
                "max_eta": max((item.eta for item in certified_results), default=0.0),
                "max_q_cert": max((item.q_cert for item in certified_results), default=0.0),
                "max_r_cert": max((item.r_cert for item in certified_results), default=0.0),
                "dominant_leaf_id": dominant_leaf.leaf_id if dominant_leaf is not None else "",
                "dominant_eta": dominant_leaf.eta if dominant_leaf is not None else 0.0,
                "dominant_q_cert": dominant_leaf.q_cert if dominant_leaf is not None else 0.0,
                "dominant_r_cert": dominant_leaf.r_cert if dominant_leaf is not None else 0.0,
                **(same_partition_storage(vars(item) for item in leaf_results)
                   if not (strict and fallback_dense_results) else {}),
            }
        )
        result = summary_rows[-1]
        result.update(storage_policy=storage_policy,
                      feasible=int(not (strict and fallback_dense_results)),
                      failed_offdiagonal_leaves=len(fallback_dense_results) if strict else 0)
        if strict:
            result["fallback_dense_leaves"] = 0
            if fallback_dense_results:
                for name in ("global_actual_error", "global_cert", "cert_over_actual",
                             "payload_bits", "scale_bits", "map_bits", "total_bits_with_metadata",
                             "payload_compression", "total_compression", "stored_dense_bits",
                             "retained_factor_entries", "same_partition_fp32_bits", "same_partition_fp64_bits",
                             "max_leaf_actual", "max_eta", "max_q_cert", "max_r_cert",
                             "dominant_leaf_id", "dominant_eta", "dominant_q_cert", "dominant_r_cert"):
                    result[name] = ""
        for kind in ("diagonal", "admissible", "offdiagonal_inadmissible"):
            selected = [item for item, leaf in zip(leaf_results, leaves) if leaf.block_kind == kind]
            result[f"{kind}_leaves"] = len(selected)
            result[f"{kind}_bits"] = (sum(item.payload_bits + item.scale_bits + item.map_bits for item in selected)
                                       if result["feasible"] else "")
        for item in leaf_results:
            leaf_rows.append(
                {
                    "case_id": item.case_id,
                    "tol": item.tol,
                    "budget_policy": item.budget_policy,
                    "leaf_id": item.leaf_id,
                    "row0": item.row0,
                    "row1": item.row1,
                    "col0": item.col0,
                    "col1": item.col1,
                    "low_rank": int(item.low_rank),
                    "rank": item.rank,
                    "eta": item.eta,
                    "budget": item.budget,
                    "block_norm": item.block_norm,
                    "avg_bits": item.avg_bits,
                    "separation_index": item.separation_index,
                    "diameter_index": item.diameter_index,
                    "q_cert": item.q_cert,
                    "r_cert": item.r_cert,
                    "leaf_cert": item.leaf_cert,
                    "actual_error": item.actual_error,
                    "payload_bits": item.payload_bits,
                    "scale_bits": item.scale_bits,
                    "map_bits": item.map_bits,
                    "bit_counts": item.bit_counts,
                    "feasible": int(item.feasible),
                    "fallback_reason": "" if strict else item.fallback_reason,
                    "failure_reason": item.fallback_reason if strict else "",
                    "separation": item.separation,
                    "diameter": item.diameter,
                    "distance_metric": item.distance_metric,
                    "storage_policy": storage_policy,
                    "block_kind": leaves[item.leaf_id].block_kind,
                    "admissible": int(leaves[item.leaf_id].low_rank),
                    "stored": int(not strict or item.feasible),
                    "bits_by_index": representations[(tol, item.leaf_id)][4].tolist()
                    if representations is not None and (tol, item.leaf_id) in representations else [],
                }
            )
            if strict and not item.feasible:
                for field in ("q_cert", "r_cert", "leaf_cert", "actual_error",
                              "payload_bits", "scale_bits", "map_bits", "avg_bits"):
                    leaf_rows[-1][field] = ""
    return summary_rows, leaf_rows


def run_local_case(
    case_id: str,
    u: np.ndarray,
    sigma: np.ndarray,
    v: np.ndarray,
    matrix: np.ndarray,
    tolerances: list[float],
    bit_set: list[int],
) -> list[dict]:
    rows: list[dict] = []
    m, n = matrix.shape
    norm = float(np.max(np.abs(matrix)))
    dense64_bits = matrix.size * 64
    for tol in tolerances:
        budget = tol * norm
        for alloc in all_allocation_methods(sigma, bit_set, budget, m, n, include_dp=True):
            exact_hat, tilde = reconstruct_quantized(u, sigma, v, alloc.bits, np.float64)
            actual_q = float(np.max(np.abs(matrix - exact_hat)))
            geom = geometry_cert(sigma, u, v, alloc.bits)
            rows.append(
                {
                    "case_id": case_id,
                    "tol": tol,
                    "method": alloc.method,
                    "feasible": int(alloc.feasible),
                    "actual_error": actual_q,
                    "cert_sv": alloc.cert,
                    "cert_geom": geom,
                    "budget": budget,
                    "payload_bits": alloc.payload_bits,
                    "dense64_bits": dense64_bits,
                    "payload_compression": dense64_bits / alloc.payload_bits if alloc.payload_bits else "",
                    "avg_bits": alloc.avg_bits,
                    "bit_counts": channel_bit_counts(alloc.bits, bit_set),
                    "bits_by_index": alloc.bits.tolist(),
                    "singular_values": sigma.tolist(),
                }
            )
    return rows
