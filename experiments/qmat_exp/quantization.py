from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


def q_value(bits: int) -> int:
    if bits < 2:
        raise ValueError("signed symmetric quantization requires bits >= 2")
    return 2 ** (bits - 1) - 1


def c_value(bits: int) -> float:
    q = q_value(bits)
    return 1.0 / q + 1.0 / (4.0 * q * q)


def rho_value(bits: int) -> float:
    return 1.0 + 1.0 / (2.0 * q_value(bits))


def gamma(length: int, unit_roundoff: float) -> float:
    if length <= 0:
        return 0.0
    gu = length * unit_roundoff
    if gu >= 1.0:
        return float("inf")
    return gu / (1.0 - gu)


def quantize_vector(vec: np.ndarray, bits: int) -> tuple[np.ndarray, float, np.ndarray]:
    max_abs = float(np.max(np.abs(vec)))
    if max_abs == 0.0:
        return np.zeros_like(vec, dtype=np.int64), 0.0, np.zeros_like(vec)
    scale = max_abs / q_value(bits)
    ints = np.rint(vec / scale).astype(np.int64)
    ints = np.clip(ints, -q_value(bits), q_value(bits))
    return ints, scale, ints.astype(np.float64) * scale


@dataclass
class Allocation:
    method: str
    feasible: bool
    bits: np.ndarray
    cert: float
    payload_bits: int

    @property
    def avg_bits(self) -> float:
        return float(np.mean(self.bits)) if self.bits.size else 0.0


def cert_from_bits(sigma: np.ndarray, bits: np.ndarray) -> float:
    return float(sum(float(s) * c_value(int(b)) for s, b in zip(sigma, bits)))


def geometry_cert(sigma: np.ndarray, u: np.ndarray, v: np.ndarray, bits: np.ndarray) -> float:
    alpha = np.max(np.abs(u), axis=0)
    beta = np.max(np.abs(v), axis=0)
    return float(sum(s * a * b * c_value(int(bb)) for s, a, b, bb in zip(sigma, alpha, beta, bits)))


def uniform_allocation(
    sigma: np.ndarray,
    bit: int,
    budget: float,
    m: int,
    n: int,
) -> Allocation:
    bits = np.full(sigma.shape, bit, dtype=np.int64)
    cert = cert_from_bits(sigma, bits)
    return Allocation(
        method=f"uniform{bit}",
        feasible=cert <= budget,
        bits=bits,
        cert=cert,
        payload_bits=int((m + n) * int(np.sum(bits))),
    )


def reserve_greedy_allocation(
    sigma: np.ndarray,
    bit_set: list[int],
    budget: float,
    m: int,
    n: int,
    *,
    trace: list[dict] | None = None,
) -> Allocation:
    bits_sorted = sorted(bit_set)
    coeffs = [c_value(b) for b in bits_sorted]
    c_high = coeffs[-1]
    prefix = np.concatenate([[0.0], np.cumsum(sigma)])
    rank = sigma.size
    def record(stage, t, remaining):
        if trace is not None:
            reserve = c_high * prefix[t]
            trace.append(dict(stage=stage, assigned=rank - t,
                              remaining_budget=float(remaining),
                              minimum_future_error=float(reserve),
                              slack=float(remaining - reserve)))

    record("initial", rank, budget)
    if c_high * prefix[rank] > budget:
        return Allocation("reserve_greedy", False, np.full(rank, bits_sorted[-1]), c_high * prefix[rank], (m + n) * rank * bits_sorted[-1])

    out = np.zeros(rank, dtype=np.int64)
    remaining = float(budget)
    t = rank
    for bit, coeff in zip(bits_sorted[:-1], coeffs[:-1]):
        channel_cost = 0.0
        while t >= 1:
            candidate = channel_cost + coeff * sigma[t - 1] + c_high * prefix[t - 1]
            if candidate <= remaining:
                out[t - 1] = bit
                channel_cost += coeff * sigma[t - 1]
                t -= 1
                record("accept", t, remaining - channel_cost)
            else:
                break
        remaining -= channel_cost
    if t >= 1:
        out[:t] = bits_sorted[-1]
        remaining -= c_high * prefix[t]
        record("highest", 0, remaining)
    cert = cert_from_bits(sigma, out)
    return Allocation("reserve_greedy", cert <= budget, out, cert, int((m + n) * int(np.sum(out))))


def naive_low_to_high_allocation(
    sigma: np.ndarray,
    bit_set: list[int],
    budget: float,
    m: int,
    n: int,
    *,
    trace: list[dict] | None = None,
) -> Allocation:
    bits_sorted = sorted(bit_set)
    rank = sigma.size
    def record(stage, t, remaining):
        if trace is not None:
            reserve = c_value(bits_sorted[-1]) * float(np.sum(sigma[:t]))
            trace.append(dict(stage=stage, assigned=rank - t,
                              remaining_budget=float(remaining),
                              minimum_future_error=reserve,
                              slack=float(remaining - reserve)))

    record("initial", rank, budget)
    out = np.zeros(rank, dtype=np.int64)
    spent = 0.0
    t = rank
    for bit in bits_sorted[:-1]:
        coeff = c_value(bit)
        while t >= 1 and spent + coeff * sigma[t - 1] <= budget:
            out[t - 1] = bit
            spent += coeff * sigma[t - 1]
            t -= 1
            record("accept", t, budget - spent)
    if t >= 1:
        out[:t] = bits_sorted[-1]
    cert = cert_from_bits(sigma, out)
    if t >= 1:
        record("highest", 0, budget - cert)
    return Allocation("naive_greedy", cert <= budget, out, cert, int((m + n) * int(np.sum(out))))


def dp_optimal_allocation(
    sigma: np.ndarray,
    bit_set: list[int],
    budget: float,
    m: int,
    n: int,
) -> Allocation:
    bits_sorted = sorted(bit_set)
    rank = sigma.size
    min_sum = rank * bits_sorted[0]
    max_sum = rank * bits_sorted[-1]
    inf = float("inf")
    dp = np.full((rank + 1, max_sum + 1), inf)
    choice = np.full((rank + 1, max_sum + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0
    for i in range(rank):
        for bit in bits_sorted:
            add = sigma[i] * c_value(bit)
            prev = dp[i, : max_sum + 1 - bit]
            cand = prev + add
            target = np.arange(bit, max_sum + 1)
            improve = cand < dp[i + 1, target]
            dp[i + 1, target[improve]] = cand[improve]
            choice[i + 1, target[improve]] = bit

    best_sum = None
    for total_bits in range(min_sum, max_sum + 1):
        if dp[rank, total_bits] <= budget:
            best_sum = total_bits
            break
    if best_sum is None:
        bits = np.full(rank, bits_sorted[-1], dtype=np.int64)
        return Allocation("dp_optimal", False, bits, cert_from_bits(sigma, bits), int((m + n) * int(np.sum(bits))))

    bits = np.zeros(rank, dtype=np.int64)
    total = best_sum
    for i in range(rank, 0, -1):
        bit = int(choice[i, total])
        bits[i - 1] = bit
        total -= bit
    cert = cert_from_bits(sigma, bits)
    return Allocation("dp_optimal", True, bits, cert, int((m + n) * int(np.sum(bits))))


def reconstruct_quantized(
    u: np.ndarray,
    sigma: np.ndarray,
    v: np.ndarray,
    bits: np.ndarray,
    accumulation_dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    rank = sigma.size
    exact_terms: list[np.ndarray] = []
    channels: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for j in range(rank):
        root = float(np.sqrt(sigma[j]))
        p = root * u[:, j]
        q = root * v[:, j]
        _, _, phat = quantize_vector(p, int(bits[j]))
        _, _, qhat = quantize_vector(q, int(bits[j]))
        exact_terms.append(np.outer(phat, qhat))
        left, right = channels.setdefault(int(bits[j]), ([], []))
        left.append(phat)
        right.append(qhat)

    exact_hat = np.sum(exact_terms, axis=0, dtype=np.float64) if exact_terms else np.zeros((u.shape[0], v.shape[0]))
    tilde = np.zeros_like(exact_hat, dtype=accumulation_dtype)
    for bit in sorted(channels):
        left, right = channels[bit]
        pmat = np.column_stack(left).astype(accumulation_dtype, copy=False)
        qmat = np.column_stack(right).astype(accumulation_dtype, copy=False)
        tilde = (tilde + pmat @ qmat.T).astype(accumulation_dtype, copy=False)
    return exact_hat, tilde.astype(np.float64)


def channel_bit_counts(bits: np.ndarray, bit_set: list[int]) -> dict[str, int]:
    return {str(bit): int(np.sum(bits == bit)) for bit in bit_set}


def channel_gemm_certificate(
    sigma: np.ndarray,
    bits: np.ndarray,
    bit_set: list[int],
    unit_roundoff: float,
) -> float:
    total = 0.0
    nonempty = [bit for bit in bit_set if np.any(bits == bit)]
    s_count = len(nonempty)
    for bit in nonempty:
        mask = bits == bit
        g = int(np.sum(mask))
        coeff = gamma(g, unit_roundoff) * rho_value(bit) ** 2
        coeff += gamma(s_count - 1, unit_roundoff) * (1.0 + gamma(g, unit_roundoff)) * rho_value(bit) ** 2
        total += coeff * float(np.sum(sigma[mask]))
    return float(total)


def realized_gemm_certificate(sigma, bits, bit_set, unit_roundoff):
    """Include one rounding of each dequantized factor into the GEMM dtype."""
    weight = sum(float(s) * rho_value(int(b)) ** 2 for s, b in zip(sigma, bits))
    representation = (2 * unit_roundoff + unit_roundoff ** 2) * weight
    core = channel_gemm_certificate(sigma, bits, bit_set, unit_roundoff)
    return representation + (1 + unit_roundoff) ** 2 * core


def arithmetic_diagnostics(case_id, u, sigma, v, tolerances, bit_set):
    rows = []
    norm = float(np.max(np.abs((u * sigma) @ v.T)))
    for tol in tolerances:
        alloc = reserve_greedy_allocation(sigma, bit_set, tol * norm, u.shape[0], v.shape[0])
        if not alloc.feasible:
            continue
        for dtype in (np.float32, np.float64):
            unit = float(np.finfo(dtype).eps / 2)
            reference, computed = reconstruct_quantized(u, sigma, v, alloc.bits, dtype)
            core = channel_gemm_certificate(sigma, alloc.bits, bit_set, unit)
            total = realized_gemm_certificate(sigma, alloc.bits, bit_set, unit)
            rows.append(dict(case_id=case_id, tol=tol, dtype=np.dtype(dtype).name,
                             actual_rounding=float(np.max(np.abs(reference - computed))),
                             gemm_cert=core, representation_and_gemm_cert=total,
                             channels=int(np.unique(alloc.bits).size),
                             q_cert=alloc.cert))
    return rows


def all_allocation_methods(
    sigma: np.ndarray,
    bit_set: list[int],
    budget: float,
    m: int,
    n: int,
    include_dp: bool = True,
) -> list[Allocation]:
    methods = [
        reserve_greedy_allocation(sigma, bit_set, budget, m, n),
        naive_low_to_high_allocation(sigma, bit_set, budget, m, n),
    ]
    methods.extend(uniform_allocation(sigma, bit, budget, m, n) for bit in bit_set)
    if include_dp:
        methods.append(dp_optimal_allocation(sigma, bit_set, budget, m, n))
    return methods


def brute_force_allocations_for_debug(rank: int, bit_set: list[int]):
    yield from product(bit_set, repeat=rank)
