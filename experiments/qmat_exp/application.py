"""Frozen H-matrix application inputs; construction is never in GPU timing."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .hmatrix import run_hmatrix_case
from .matrices import validate_partition
from .quantization import quantize_vector, gamma


MAGIC = b"QMATAPP1"


def pack_columns(values, bits):
    if values.ndim != 2 or len(bits) != values.shape[1] or not np.isfinite(values).all():
        raise ValueError("invalid factor shape or values")
    if any(b not in (4, 8, 12, 16, 24, 32) for b in bits):
        raise ValueError("application packing supports only 4/8/12/16/24/32 bits")
    words, offsets, scales = [], [], []
    decoded = np.empty(values.shape, dtype=np.float32, order="F")
    for j, b in enumerate(bits):
        b = int(b)
        integers, scale, exact = quantize_vector(values[:, j], b)
        offsets.append(len(words))
        scales.append(scale)
        column = [0] * ((len(integers) * b + 31) // 32)
        for i, value in enumerate(integers):
            raw = int(value) & ((1 << b) - 1)
            word, shift = divmod(i * b, 32)
            column[word] |= (raw << shift) & 0xffffffff
            if shift + b > 32:
                column[word + 1] |= raw >> (32 - shift)
        words.extend(column)
        decoded[:, j] = exact
    return dict(words=np.asarray(words, dtype="<u4"), offsets=np.asarray(offsets, dtype="<u8"),
                scales=np.asarray(scales, dtype="<f8"), decoded=decoded)


def build_application(matrix, leaves, permutation, x, cfg, tol, factorizations=None):
    """Use exactly the existing reconstruction allocator/fallback decisions.

    Dense/fallback working blocks are FP32 in both H implementations. Their
    conversion error is included in the source-reference checks, unlike the
    manuscript's FP64 dense *storage* controls.
    """
    n = len(matrix)
    if not 1 <= n <= 16384 or x.ndim != 2 or not 1 <= x.shape[1] <= 4096:
        raise ValueError("unsupported application dimensions")
    validate_partition(n, leaves)
    permutation = np.asarray(permutation, dtype=np.uint32)
    if sorted(permutation.tolist()) != list(range(n)):
        raise ValueError("invalid permutation")
    if matrix.shape != (n, n) or x.ndim != 2 or x.shape[0] != n:
        raise ValueError("incompatible application shapes")
    if not np.isfinite(matrix).all() or not np.isfinite(x).all():
        raise ValueError("nonfinite input")
    h = cfg["hmatrix"]
    cache = {} if factorizations is None else factorizations
    representations = {}
    summary, decisions = run_hmatrix_case("application", matrix, leaves, [tol], h.get("bit_set", cfg["bit_set"]),
        h["rank_tol"], h["max_rank"], h["budget_policy"], h["hybrid_gamma"],
        h["gemm_reserve_fraction"], h["accumulator_unit_roundoff"], factorizations=cache,
        storage_policy=h.get("storage_policy", "diagonal_only"),
        truncation_budget_fraction=h.get("truncation_budget_fraction", .25),
        representations=representations)
    if not summary[0]["feasible"]:
        reasons = sorted({r["failure_reason"] for r in decisions if not r["feasible"]})
        raise ValueError(f"No diagonal-only representation at tolerance {tol}: {reasons}")
    inverse = np.argsort(permutation)
    source = matrix[np.ix_(inverse, inverse)]
    dense = source.astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(dense).all() or not np.isfinite(x).all():
        raise ValueError("FP32 input conversion overflow")
    xp = x[permutation].astype(np.float64)
    yp = [np.zeros(x.shape, dtype=np.float64) for _ in range(2)]
    weights = [np.zeros(x.shape, dtype=np.float64) for _ in range(2)]
    rounding = [np.zeros(x.shape, dtype=np.float64) for _ in range(2)]
    model = [np.zeros(x.shape, dtype=np.float64) for _ in range(2)]
    counts = np.zeros(n, dtype=int)
    records = []
    u32, u64 = 2.0**-24, 2.0**-53
    for leaf_id, (leaf, decision) in enumerate(zip(leaves, decisions)):
        a, b, c, d = leaf.row0, leaf.row1, leaf.col0, leaf.col1
        block = matrix[a:b, c:d]
        counts[a:b] += 1
        rec = dict(row0=a, row1=b, col0=c, col1=d, rank=0)
        if decision["low_rank"] and decision["feasible"] and decision["rank"] > 0:
            u, sigma, v, eta, bits = representations[(tol, leaf_id)]
            p, q = u * np.sqrt(sigma), v * np.sqrt(sigma)
            pp, pq = pack_columns(p, bits), pack_columns(q, bits)
            rec.update(rank=len(sigma), bits=bits.astype("<u4"),
                       p=p.astype(np.float32), q=q.astype(np.float32), pp=pp, pq=pq)
            pairs = [(rec["p"], rec["q"]), (pp["decoded"], pq["decoded"])]
            for j, (left, right) in enumerate(pairs):
                left, right = left.astype(np.float64), right.astype(np.float64)
                yp[j][a:b] += left @ (right.T @ xp[c:d])
                weight = np.abs(left) @ (np.abs(right).T @ np.abs(xp[c:d]))
                weights[j][a:b] += weight
                coefficient = (1 + gamma(d-c, u32)) * (1 + gamma(len(sigma), u32)) - 1
                rounding[j][a:b] += coefficient * weight
                residual = float(np.max(np.abs(block - left @ right.T)))
                model[j][a:b] += residual * np.sum(np.abs(xp[c:d]), axis=0)
        else:
            rec["dense"] = block.astype(np.float32)
            working = rec["dense"].astype(np.float64)
            for j in range(2):
                yp[j][a:b] += working @ xp[c:d]
                weight = np.abs(working) @ np.abs(xp[c:d])
                weights[j][a:b] += weight
                rounding[j][a:b] += gamma(d-c, u32) * weight
                model[j][a:b] += np.max(np.abs(block-working)) * np.sum(np.abs(xp[c:d]), axis=0)
        records.append(rec)
    dense_weight = np.abs(dense.astype(np.float64)) @ np.abs(x.astype(np.float64))
    ref_factor = 16 * gamma(2*n + 2*h["max_rank"] + int(max(counts)), u64)
    allowances = [(gamma(n, u32) + ref_factor) * np.max(dense_weight, axis=0)]
    model_bounds = [np.max(np.abs(source-dense), axis=1).max() * np.sum(np.abs(x), axis=0)]
    for j in range(2):
        g = np.asarray([gamma(int(k), u32) for k in counts])[:, None]
        bound = rounding[j] + g * (weights[j] + rounding[j]) + ref_factor * weights[j]
        allowances.append(np.max(bound, axis=0))
        model_bounds.append(np.max(model[j] + ref_factor*weights[j], axis=0))
    refs = [source @ x.astype(np.float64), dense.astype(np.float64) @ x.astype(np.float64),
            yp[0][inverse], yp[1][inverse]]
    for ref, allowance in zip(refs[1:], model_bounds):
        if np.any(np.max(np.abs(ref-refs[0]), axis=0) > allowance + ref_factor*np.max(dense_weight, axis=0)):
            raise ValueError("application residual propagation check failed")
    return dict(n=n, rhs=x.shape[1], permutation=permutation, dense=dense, x=x,
                refs=refs, allowances=np.asarray(allowances), model_bounds=np.asarray(model_bounds),
                leaves=records, fallback_count=sum(bool(r["fallback_reason"]) for r in decisions))


def write_application(path, data):
    def array(f, values, dtype):
        f.write(np.asarray(values, dtype=dtype).tobytes(order="F"))
    with Path(path).open("wb") as f:
        f.write(struct.pack("<8sIII", MAGIC, data["n"], data["rhs"], len(data["leaves"])))
        array(f, data["permutation"], "<u4")
        for key in ("dense", "x"):
            array(f, data[key], "<f4")
        for ref in data["refs"]:
            array(f, ref, "<f8")
        for key in ("allowances", "model_bounds"):
            # One contiguous RHS vector per method.
            array(f, data[key].T, "<f8")
        for leaf in data["leaves"]:
            array(f, [leaf[k] for k in ("row0", "row1", "col0", "col1", "rank")], "<u4")
            if not leaf["rank"]:
                array(f, leaf["dense"], "<f4")
                continue
            array(f, leaf["bits"], "<u4")
            for key in ("p", "q"):
                array(f, leaf[key], "<f4")
            for key in ("pp", "pq"):
                packed = leaf[key]
                array(f, [len(packed["words"])], "<u8")
                for field, dtype in (("offsets", "<u8"), ("scales", "<f8"), ("words", "<u4")):
                    array(f, packed[field], dtype)
