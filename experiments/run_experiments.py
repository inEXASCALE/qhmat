from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from qmat_exp.hmatrix import run_hmatrix_case, run_local_case
from qmat_exp.io import ensure_dir, load_config, read_json, resolve_path, write_csv, write_json
from qmat_exp.matrices import load_leaves
from qmat_exp.quantization import arithmetic_diagnostics


LOCAL_FIELDS = [
    "case_id",
    "tol",
    "method",
    "feasible",
    "actual_error",
    "cert_sv",
    "cert_geom",
    "budget",
    "payload_bits",
    "dense64_bits",
    "payload_compression",
    "avg_bits",
    "bit_counts",
    "bits_by_index",
    "singular_values",
]

HMATRIX_FIELDS = [
    "storage_policy", "feasible", "failed_offdiagonal_leaves",
    "diagonal_leaves", "admissible_leaves", "offdiagonal_inadmissible_leaves",
    "diagonal_bits", "admissible_bits", "offdiagonal_inadmissible_bits",
    "family", "geometry", "n", "kernel",
    "case_id",
    "tol",
    "budget_policy",
    "global_actual_error",
    "global_cert",
    "cert_over_actual",
    "payload_bits",
    "scale_bits",
    "map_bits",
    "total_bits_with_metadata",
    "dense64_bits",
    "payload_compression",
    "total_compression",
    "low_rank_leaves",
    "dense_leaves",
    "feasible_low_rank_leaves",
    "fallback_dense_leaves",
    "mean_low_rank_avg_bits",
    "max_leaf_actual",
    "max_eta",
    "max_q_cert",
    "max_r_cert",
    "dominant_leaf_id",
    "dominant_eta",
    "dominant_q_cert",
    "dominant_r_cert",
    "stored_dense_bits",
    "retained_factor_entries",
    "same_partition_fp32_bits",
    "same_partition_fp64_bits",
]

LEAF_FIELDS = [
    "storage_policy", "block_kind", "admissible", "stored", "bits_by_index", "failure_reason",
    "family", "geometry", "n", "kernel",
    "case_id",
    "tol",
    "budget_policy",
    "leaf_id",
    "row0",
    "row1",
    "col0",
    "col1",
    "low_rank",
    "rank",
    "eta",
    "budget",
    "block_norm",
    "avg_bits",
    "separation_index",
    "diameter_index",
    "q_cert",
    "r_cert",
    "leaf_cert",
    "actual_error",
    "payload_bits",
    "scale_bits",
    "map_bits",
    "bit_counts",
    "feasible",
    "fallback_reason",
    "separation", "diameter", "distance_metric",
]

GPU_FIELDS = [
    "device",
    "m",
    "n",
    "rank",
    "bits",
    "kernel",
    "median_ms",
    "min_ms",
    "max_ms",
    "speedup_vs_fp32",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qmat paper experiments.")
    parser.add_argument("--config", default="experiments/configs/offdiagonal_smoke.json")
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def run_gpu_if_available(cfg: dict) -> list[dict]:
    try:
        from qmat_exp.gpu_perf import run_gpu_performance
        return run_gpu_performance(cfg)
    except ImportError as exc:
        return [{"device": f"skipped: {exc}", "m": "", "n": "", "rank": "", "bits": "", "kernel": "", "median_ms": "", "min_ms": "", "max_ms": "", "speedup_vs_fp32": ""}]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg["results_dir"])
    manifest_path = resolve_path(cfg["data_dir"]) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run prepare_data.py first; missing {manifest_path}")
    manifest = read_json(manifest_path)
    if "config" in manifest:
        for key in ("seed", "low_rank", "hmatrix", "geometric_suites"):
            prepared, requested = manifest["config"].get(key), cfg.get(key)
            if key == "hmatrix":
                prepared, requested = dict(prepared), dict(requested)
                prepared.setdefault("storage_policy", "admissible_only")
                requested.setdefault("storage_policy", "admissible_only")
            if prepared != requested:
                raise ValueError(f"Prepared {key} config differs; rerun prepare_data.py")

    local_rows: list[dict] = []
    arithmetic_rows: list[dict] = []
    t0 = time.time()
    for item in manifest["low_rank"]:
        data = np.load(resolve_path(item["path"]))
        local_rows.extend(
            run_local_case(
                item["case_id"],
                data["u"],
                data["sigma"],
                data["v"],
                data["matrix"],
                cfg["tolerances"],
                cfg["bit_set"],
            )
        )
        arithmetic_rows.extend(arithmetic_diagnostics(item["case_id"], data["u"],
                              data["sigma"], data["v"], cfg["tolerances"], cfg["bit_set"]))
    write_csv(results_dir / "local_certificate.csv", local_rows, LOCAL_FIELDS)
    write_csv(results_dir / "arithmetic.csv", arithmetic_rows,
              ["case_id", "tol", "dtype", "actual_rounding", "gemm_cert",
               "representation_and_gemm_cert", "channels", "q_cert"])

    h_summary_rows: list[dict] = []
    h_leaf_rows: list[dict] = []
    h_cfg = cfg["hmatrix"]
    budget_policies = h_cfg.get("budget_policies", [h_cfg["budget_policy"]])
    for item in manifest["hmatrix"]:
        data = np.load(resolve_path(item["path"]))
        leaves = load_leaves(read_json(resolve_path(item["leaves_path"])))
        factorizations = {}
        for budget_policy in budget_policies:
            summary, leaves_detail = run_hmatrix_case(
                item["case_id"],
                data["matrix"],
                leaves,
                cfg["tolerances"],
                h_cfg.get("bit_set", cfg["bit_set"]),
                h_cfg["rank_tol"],
                h_cfg["max_rank"],
                budget_policy,
                h_cfg["hybrid_gamma"],
                h_cfg["gemm_reserve_fraction"],
                h_cfg["accumulator_unit_roundoff"],
                factorizations=factorizations,
                storage_policy=h_cfg.get("storage_policy", "diagonal_only"),
                truncation_budget_fraction=h_cfg.get("truncation_budget_fraction", .25),
                representations={},
            )
            for row in summary + leaves_detail:
                row.update(family=item.get("family", "kernel1d"), geometry=item.get("geometry", item["kernel"]), n=item["n"], kernel=item["kernel"])
            h_summary_rows.extend(summary)
            h_leaf_rows.extend(leaves_detail)
        print(f"completed {item['case_id']}", flush=True)
    write_csv(results_dir / "hmatrix_certificate.csv", h_summary_rows, HMATRIX_FIELDS)
    write_csv(results_dir / "hmatrix_leaves.csv", h_leaf_rows, LEAF_FIELDS)

    # FP64 references are numerical oracles, so allow explicitly bounded reference-scale slack.
    for row in local_rows:
        slack = 64 * np.finfo(np.float64).eps * len(row["singular_values"]) * sum(row["singular_values"])
        if row["actual_error"] > min(row["cert_geom"], row["cert_sv"]) + slack:
            raise RuntimeError(f"Local certificate violation: {row['case_id']}")
    for row in arithmetic_rows:
        if row["actual_rounding"] > row["representation_and_gemm_cert"] + 1e-14:
            raise RuntimeError(f"Arithmetic certificate violation: {row['case_id']}")
    for row in h_summary_rows:
        if row["feasible"] and row["global_actual_error"] > row["global_cert"] + 1e-12:
            raise RuntimeError(f"H-matrix certificate violation: {row['case_id']}")
    write_json(results_dir / "numerical_validation.json", dict(config=cfg,
               python=platform.python_version(), numpy=np.__version__,
               source_sha256={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), *sorted((Path(__file__).parent / "qmat_exp").glob("*.py"))]},
               local_checks=len(local_rows), arithmetic_checks=len(arithmetic_rows),
               hmatrix_checks=sum(r["feasible"] for r in h_summary_rows),
               infeasible_hmatrix_configurations=sum(not r["feasible"] for r in h_summary_rows), violations=0,
               arithmetic_absolute_slack=1e-14, hmatrix_absolute_slack=1e-12,
               reference="FP64 numerical reference; not interval arithmetic"))

    gpu_rows: list[dict] = []
    if cfg.get("gpu", {}).get("enabled", False) and not args.skip_gpu:
        gpu_rows = run_gpu_if_available(cfg)
    write_csv(results_dir / "gpu_performance.csv", gpu_rows, GPU_FIELDS)

    print(
        json.dumps(
            {
                "results_dir": str(results_dir),
                "local_rows": len(local_rows),
                "arithmetic_rows": len(arithmetic_rows),
                "hmatrix_rows": len(h_summary_rows),
                "feasible_hmatrix_configurations": sum(r["feasible"] for r in h_summary_rows),
                "infeasible_hmatrix_configurations": sum(not r["feasible"] for r in h_summary_rows),
                "leaf_rows": len(h_leaf_rows),
                "gpu_rows": len(gpu_rows),
                "elapsed_s": round(time.time() - t0, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
