"""Stage 2: paired, randomized-order CUDA application measurements."""
import argparse
import platform
import subprocess

import numpy as np

from prepare_application import sha256, validate_config
from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path, write_csv, write_json


METHODS = ("dense_fp32", "h_fp32_batched", "adaptive_unpacked_batched",
           "adaptive_predecoded_batched", "h_fp32_fused", "adaptive_packed_fused")


def validate_rows(rows, cfg, n, rhs):
    expected = {(m, trial) for m in METHODS for trial in range(cfg["trials"])}
    keys = [(r["method"], int(r["trial"])) for r in rows]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("incomplete or duplicated application measurements")
    for r in rows:
        if int(r["n"]) != n or int(r["rhs"]) != rhs or r["validation"] != "arithmetic_pass":
            raise ValueError("application metadata or numerical validation failed")
        for field in ("median_ms", "min_ms", "max_ms", "speedup_vs_dense", "speedup_vs_h_fp32"):
            if not np.isfinite(float(r[field])) or float(r[field]) <= 0:
                raise ValueError(f"invalid timing: {field}")
        if not float(r["min_ms"]) <= float(r["median_ms"]) <= float(r["max_ms"]):
            raise ValueError("inconsistent timing statistics")
        for field in ("relative_max_error", "max_abs_error", "arithmetic_error", "arithmetic_allowance",
                      "representation_error", "representation_bound"):
            if not np.isfinite(float(r[field])) or float(r[field]) < 0:
                raise ValueError(f"invalid validation value: {field}")
        if float(r["arithmetic_error"]) > float(r["arithmetic_allowance"]):
            raise ValueError("arithmetic error exceeds allowance")
    for trial in range(cfg["trials"]):
        group = {r["method"]: r for r in rows if int(r["trial"]) == trial}
        if sorted(int(r["order_index"]) for r in group.values()) != list(range(len(METHODS))):
            raise ValueError("invalid randomized method order")
        for r in group.values():
            for baseline, field in (("dense_fp32", "speedup_vs_dense"), ("h_fp32_batched", "speedup_vs_h_fp32")):
                ratio = float(group[baseline]["median_ms"])/float(r["median_ms"])
                if not np.isclose(ratio, float(r[field]), rtol=1e-12, atol=0):
                    raise ValueError("unpaired speedup denominator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/application_a100.json")
    parser.add_argument("--binary", default="experiments/cuda/build/qmat_apply")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    data_dir = resolve_path(cfg["data_dir"]) / ("smoke" if args.smoke else "")
    manifest = read_json(data_dir / "manifest.json")
    if manifest["config"] != cfg or manifest["smoke"] != args.smoke:
        raise ValueError("application manifest mismatch; prepare inputs again")
    binary = resolve_path(args.binary)
    binary_hash = sha256(binary)
    results = ensure_dir(resolve_path(cfg["results_dir"]) / ("smoke" if args.smoke else ""))
    # Refuse replacement rather than mixing results or deleting a previous successful run.
    provenance = results / "application_run.json"
    if provenance.exists() or (results / "application_performance.csv").exists():
        raise FileExistsError(f"Choose a new results_dir; prior run exists: {provenance}")
    raw_dir = ensure_dir(results / "raw")
    write_json(provenance, dict(status="running", config=cfg, smoke=args.smoke,
        binary_sha256=binary_hash, platform=platform.platform(), manifest_sha256=sha256(data_dir / "manifest.json")))
    rows = []
    try:
        for item in manifest["inputs"]:
            path = data_dir / item["file"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"input hash mismatch: {path}")
            for rhs in cfg["rhs_counts"]:
                raw = raw_dir / f"{path.stem}_rhs{rhs}.csv"
                subprocess.run([str(binary), str(path), str(rhs), str(cfg["warmup"]), str(cfg["repeats"]),
                                str(cfg["trials"]), str(cfg["seed"]), str(raw)], check=True)
                measured = read_csv(raw)
                validate_rows(measured, cfg, item["n"], rhs)
                rows.extend(dict(case_id=item["case_id"], tol=item["tol"], input_sha256=item["sha256"], **r) for r in measured)
                print(f"{path.stem}, rhs={rhs}: validated {len(measured)} measurements", flush=True)
        if sha256(binary) != binary_hash:
            raise ValueError("CUDA binary changed during the run")
        if not rows:
            raise ValueError("empty application suite")
        write_csv(results / "application_performance.csv", rows, list(rows[0]))
        run = read_json(provenance)
        write_json(provenance, dict(run, status="complete", inputs=manifest["inputs"]))
    except Exception as error:
        run = read_json(provenance)
        write_json(provenance, dict(run, status="failed", error=str(error)))
        raise


if __name__ == "__main__":
    main()
