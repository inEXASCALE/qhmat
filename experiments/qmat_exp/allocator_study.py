"""Paired precision-menu ablations, reserve diagnostics, and scaling controls."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from .quantization import (c_value, dp_optimal_allocation,
                           geometry_cert, naive_low_to_high_allocation,
                           quantize_vector, reconstruct_quantized,
                           reserve_greedy_allocation, uniform_allocation)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_spec(cfg):
    return {k: v for k, v in cfg.items()
            if k not in {"data_dir", "results_dir", "figures_dir", "plot_style"}}


def spec_hash(cfg):
    return hashlib.sha256(json.dumps(experiment_spec(cfg), sort_keys=True).encode()).hexdigest()


def validate_config(cfg):
    if not 0 < cfg["rank"] <= min(cfg["m"], cfg["n"]):
        raise ValueError("Rank must fit both factor dimensions")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Seeds must be distinct and nonempty")
    menus = list(cfg["menus"].values())
    if len(menus) < 2:
        raise ValueError("At least two precision menus are required")
    for bits in menus:
        if bits != sorted(set(bits)) or not bits or min(bits) < 2 or max(bits) > 32:
            raise ValueError("Menus must contain distinct increasing integer widths in [2,32]")
        if any(type(b) is not int for b in bits):
            raise ValueError("Bit widths must be integers")
    if any(not set(a) < set(b) for a, b in zip(menus, menus[1:])):
        raise ValueError("Menus must be strictly nested")
    if any(t <= 0 or not np.isfinite(t) for t in cfg["tolerances"]):
        raise ValueError("Tolerances must be positive and finite")
    if len(set(cfg["tolerances"])) != len(cfg["tolerances"]):
        raise ValueError("Duplicate tolerances")
    if not cfg["tolerances"] or not cfg["spectra"]:
        raise ValueError("Spectra and tolerances must be nonempty")
    if len({str(cfg[k]) for k in ("data_dir", "results_dir", "figures_dir")}) != 3:
        raise ValueError("Data, results and figures must use separate directories")


def synthetic_case(cfg, spectrum, seed):
    rng = np.random.default_rng(seed)
    r = cfg["rank"]
    u, _ = np.linalg.qr(rng.standard_normal((cfg["m"], r)))
    v, _ = np.linalg.qr(rng.standard_normal((cfg["n"], r)))
    spec = cfg["spectra"][spectrum]
    if spec["kind"] == "exponential":
        sigma = float(spec["parameter"]) ** np.arange(r)
    elif spec["kind"] == "algebraic":
        sigma = np.arange(1, r + 1, dtype=float) ** -float(spec["parameter"])
    elif spec["kind"] == "flat":
        sigma = np.ones(r)
    else:
        raise ValueError(f"Unknown spectrum {spec}")
    if np.any(sigma[:-1] < sigma[1:]) or np.any(sigma < 0):
        raise ValueError("Spectrum must be nonnegative and nonincreasing")
    return u, sigma, v


def menu_case(cfg, item, u, sigma, v):
    matrix = (u * sigma) @ v.T
    norm = float(np.max(np.abs(matrix)))
    rows = []
    for tol in cfg["tolerances"]:
        budget = tol * norm
        for menu, bits in cfg["menus"].items():
            greedy = reserve_greedy_allocation(sigma, bits, budget, len(u), len(v))
            dp = dp_optimal_allocation(sigma, bits, budget, len(u), len(v))
            uniform = next((a for b in bits if
                            (a := uniform_allocation(sigma, b, budget, len(u), len(v))).feasible),
                           uniform_allocation(sigma, bits[-1], budget, len(u), len(v)))
            for method, alloc in (("greedy", greedy), ("dp", dp), ("uniform", uniform)):
                row = dict(case_id=item["case_id"], spectrum=item["spectrum"], seed=item["seed"],
                           m=len(u), n=len(v), rank=len(sigma), tol=tol, menu=menu, method=method,
                           feasible=int(alloc.feasible), budget=budget, matrix_max=norm,
                           cert_sv=alloc.cert, payload_bits=alloc.payload_bits if alloc.feasible else "",
                           factor16_bits=16 * (len(u) + len(v)) * len(sigma),
                           bits_by_index=alloc.bits.tolist() if alloc.feasible else [],
                           actual_error="", cert_geom="", numerical_allowance="", validation_ok="")
                if alloc.feasible:
                    reference, _ = reconstruct_quantized(u, sigma, v, alloc.bits)
                    error = float(np.max(np.abs(reference - matrix)))
                    geom = geometry_cert(sigma, u, v, alloc.bits)
                    allowance = 512 * np.finfo(float).eps * float(np.sum(sigma))
                    ok = error <= geom + allowance and geom <= alloc.cert + allowance
                    ok = ok and alloc.cert <= budget + allowance
                    if not ok:
                        raise AssertionError(f"Numerical certificate violation: {row}")
                    row.update(actual_error=error, cert_geom=geom,
                               numerical_allowance=allowance, validation_ok=1)
                rows.append(row)
    return rows


def quantiles(values):
    if not values:
        return dict(count=0, min=None, median=None, q90=None, max=None)
    a = np.array(values, dtype=float)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                q90=float(np.quantile(a, .9)), max=float(a.max()))


def summarize(rows, cfg):
    key = lambda r: (r["case_id"], float(r["tol"]), r["menu"], r["method"])
    indexed = {key(r): r for r in rows}
    expected = len(cfg["seeds"]) * len(cfg["spectra"]) * len(cfg["tolerances"]) * len(cfg["menus"]) * 3
    if len(indexed) != len(rows) or len(rows) != expected:
        raise ValueError("Incomplete or duplicate menu results")
    groups = []
    for spectrum in ["all", *cfg["spectra"]]:
        for menu in cfg["menus"]:
            selected = [r for r in rows if r["menu"] == menu and r["method"] == "greedy"
                        and (spectrum == "all" or r["spectrum"] == spectrum)]
            matched = [r for r in selected if int(r["feasible"])]
            ratios = []
            for r in selected:
                d = indexed[(r["case_id"], float(r["tol"]), menu, "dp")]
                f = indexed[(r["case_id"], float(r["tol"]), menu, "uniform")]
                if int(r["feasible"]) != int(d["feasible"]) or int(r["feasible"]) != int(f["feasible"]):
                    raise AssertionError("Feasibility disagreement")
                if int(r["feasible"]):
                    ratio = int(r["payload_bits"]) / int(d["payload_bits"])
                    if ratio < 1:
                        raise AssertionError("Greedy payload below DP optimum")
                    ratios.append(ratio)
            groups.append(dict(spectrum=spectrum, menu=menu, cases=len(selected), feasible=len(matched),
                               greedy_over_dp=quantiles(ratios)))
    pairs = []
    menus = list(cfg["menus"])
    for a, b in zip(menus, menus[1:]):
        for r in rows:
            if r["menu"] != a or r["method"] != "greedy":
                continue
            base = (r["case_id"], float(r["tol"]))
            g2 = indexed[(*base, b, "greedy")]
            d1, d2 = (indexed[(*base, menu, "dp")] for menu in (a, b))
            both = bool(int(r["feasible"]) and int(g2["feasible"]))
            if int(r["feasible"]) and not int(g2["feasible"]):
                raise AssertionError("Nested menus lost feasibility")
            if both and int(d2["payload_bits"]) > int(d1["payload_bits"]):
                raise AssertionError("Nested menus increased optimal payload")
            pairs.append(dict(case_id=base[0], spectrum=r["spectrum"], seed=int(r["seed"]), tol=base[1],
                              before=a, after=b, paired_feasible=int(both),
                              newly_feasible=int(not int(r["feasible"]) and int(g2["feasible"])),
                              greedy_ratio=int(g2["payload_bits"]) / int(r["payload_bits"]) if both else "",
                              dp_ratio=int(d2["payload_bits"]) / int(d1["payload_bits"]) if both else ""))
    transitions = []
    for a, b in zip(menus, menus[1:]):
        matched = [r for r in pairs if r["before"] == a and r["paired_feasible"]]
        transitions.append(dict(before=a, after=b, paired=len(matched),
                                newly_feasible=sum(r["newly_feasible"] for r in pairs if r["before"] == a),
                                greedy_worse=sum(r["greedy_ratio"] > 1 for r in matched),
                                greedy_ratio=quantiles([r["greedy_ratio"] for r in matched]),
                                dp_ratio=quantiles([r["dp_ratio"] for r in matched])))
    return dict(groups=groups, transitions=transitions), pairs


def reserve_diagnostics(cfg):
    bits = [4, 8, 16]
    sigma = np.ones(8)
    budget = 6 * c_value(4) + .5 * c_value(16)
    trace_rows = []
    for method, allocate in (("greedy", reserve_greedy_allocation),
                             ("unrepaired", naive_low_to_high_allocation)):
        trace = []
        alloc = allocate(sigma, bits, budget, 8, 8, trace=trace)
        for step, state in enumerate(trace):
            trace_rows.append(dict(method=method, step=step, **state, budget=budget,
                                   minimum_initial_error=8 * c_value(16), final_feasible=int(alloc.feasible)))
    boundary = []
    q = 2 ** 15 - 1
    coeff_exact = Fraction(1, q) + Fraction(1, 4 * q * q)
    for exponent in cfg["boundary_scale_exponents"]:
        scale = float(2.0 ** exponent)
        sigma = np.full(2, scale)
        exact_min = 2 * Fraction.from_float(scale) * coeff_exact
        for ratio in cfg["boundary_ratios"]:
            budget = float(exact_min * Fraction(str(ratio)))
            exact_margin = Fraction.from_float(budget) - exact_min
            band = 64 * np.finfo(float).eps * float(exact_min)
            near = abs(float(exact_margin)) <= band
            oracle = Fraction.from_float(budget) >= exact_min
            for method, allocate in (("greedy", reserve_greedy_allocation), ("dp", dp_optimal_allocation)):
                result = allocate(sigma, bits, budget, 2, 2)
                if not near and result.feasible != oracle:
                    raise AssertionError("Feasibility mismatch outside roundoff boundary band")
                boundary.append(dict(scale_exponent=exponent, budget_ratio=ratio, budget=budget,
                                     exact_margin=float(exact_margin), boundary_band=band,
                                     near_boundary=int(near), rational_oracle_feasible=int(oracle),
                                     method=method, feasible=int(result.feasible), cert=result.cert))
    return trace_rows, boundary


def scaling_diagnostics(cfg, item, u, sigma, v):
    bits = np.resize(np.array([4, 8, 12, 16]), len(sigma))
    p, q = u * np.sqrt(sigma), v * np.sqrt(sigma)
    matrix = (u * sigma) @ v.T

    def reconstruct(left, right, shared):
        if not shared:
            lp = np.column_stack([quantize_vector(left[:, j], int(b))[2] for j, b in enumerate(bits)])
            rp = np.column_stack([quantize_vector(right[:, j], int(b))[2] for j, b in enumerate(bits)])
            return lp @ rp.T
        out = np.zeros((len(u), len(v)))
        for b in np.unique(bits):
            mask = bits == b
            l, r = left[:, mask], right[:, mask]
            lh = quantize_vector(l.ravel(), int(b))[2].reshape(l.shape)
            rh = quantize_vector(r.ravel(), int(b))[2].reshape(r.shape)
            out += lh @ rh.T
        return out

    rows = []
    for mode in ("columnwise", "shared_channel"):
        original = reconstruct(p, q, mode == "shared_channel")
        for exponent in cfg["scaling_exponents"]:
            d = np.exp2(np.linspace(-float(exponent), float(exponent), len(sigma)))
            result = reconstruct(p * d, q / d, mode == "shared_channel")
            norm = float(np.max(np.abs(matrix)))
            difference = float(np.max(np.abs(result - original))) / norm
            if mode == "columnwise" and difference > 1e-12:
                raise AssertionError("Columnwise reciprocal scaling control failed")
            rows.append(dict(case_id=item["case_id"], spectrum=item["spectrum"], seed=item["seed"],
                             mode=mode, exponent=exponent, relative_change=difference,
                             relative_error=float(np.max(np.abs(result - matrix))) / norm,
                             original_relative_error=float(np.max(np.abs(original - matrix))) / norm))
    return rows
