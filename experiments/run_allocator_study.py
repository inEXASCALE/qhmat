"""Stage 2: measure allocation quality and numerical controls (no GPU timing)."""
import argparse
import platform
import subprocess

import numpy as np

from qmat_exp.allocator_study import (menu_case, reserve_diagnostics, scaling_diagnostics,
                                     sha256, spec_hash, summarize, validate_config)
from qmat_exp.io import ensure_dir, load_config, read_json, repo_root, resolve_path, write_csv, write_json


def run(cfg):
    validate_config(cfg)
    data = resolve_path(cfg["data_dir"])
    manifest = read_json(data / "manifest.json")
    if manifest["spec_hash"] != spec_hash(cfg):
        raise ValueError("Prepared configuration differs; rerun prepare_allocator_study.py")
    expected = {f"{spectrum}_seed{seed}" for spectrum in cfg["spectra"] for seed in cfg["seeds"]}
    if len(manifest["cases"]) != len(expected) or {r["case_id"] for r in manifest["cases"]} != expected:
        raise ValueError("Incomplete or duplicate prepared cases")
    # Validate every input before producing results.
    for item in manifest["cases"]:
        if item["case_id"] != f"{item['spectrum']}_seed{item['seed']}":
            raise ValueError("Prepared metadata does not match case identifier")
        if sha256(data / item["filename"]) != item["sha256"]:
            raise ValueError(f"Prepared input hash mismatch: {item['case_id']}")
    rows, scaling = [], []
    for item in manifest["cases"]:
        with np.load(data / item["filename"], allow_pickle=False) as arrays:
            u, sigma, v = arrays["u"], arrays["sigma"], arrays["v"]
        rank = cfg["rank"]
        if u.shape != (cfg["m"], rank) or v.shape != (cfg["n"], rank) or sigma.shape != (rank,):
            raise ValueError("Prepared factor dimensions do not match configuration")
        if not all(np.all(np.isfinite(a)) for a in (u, sigma, v)) or np.any(sigma < 0) or np.any(np.diff(sigma) > 0):
            raise ValueError("Invalid prepared spectrum or factor values")
        if max(np.linalg.norm(a.T @ a - np.eye(rank), ord=np.inf) for a in (u, v)) > 1e-10:
            raise ValueError("Singular-value certificate requires orthonormal factors")
        rows.extend(menu_case(cfg, item, u, sigma, v))
        scaling.extend(scaling_diagnostics(cfg, item, u, sigma, v))
        print(f"Measured {item['case_id']}", flush=True)
    summary, pairs = summarize(rows, cfg)
    trace, boundary = reserve_diagnostics(cfg)
    summary["scaling"] = {mode: dict(max_relative_change=max(r["relative_change"] for r in scaling if r["mode"] == mode),
                                     max_relative_error=max(r["relative_error"] for r in scaling if r["mode"] == mode))
                          for mode in ("columnwise", "shared_channel")}
    summary["boundary"] = dict(rows=len(boundary), near_boundary=sum(r["near_boundary"] for r in boundary),
                                off_boundary_mismatches=sum(r["feasible"] != r["rational_oracle_feasible"]
                                                           for r in boundary if not r["near_boundary"]))
    directory = ensure_dir(cfg["results_dir"])
    tables = dict(menu_results=rows, paired_transitions=pairs, reserve_trace=trace,
                  feasibility_boundary=boundary, scaling=scaling)
    for name, values in tables.items():
        write_csv(directory / f"{name}.csv", values, list(values[0]))
    write_json(directory / "summary.json", summary)
    source_paths = ["experiments/prepare_allocator_study.py", "experiments/run_allocator_study.py",
                    "experiments/qmat_exp/allocator_study.py", "experiments/qmat_exp/quantization.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True)
    write_json(directory / "provenance.json", dict(
        spec_hash=spec_hash(cfg), config=cfg, manifest=manifest,
        manifest_sha256=sha256(data / "manifest.json"), python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), git_revision=revision.stdout.strip(),
        source_sha256={p: sha256(repo_root() / p) for p in source_paths},
        result_sha256={f"{name}.csv": sha256(directory / f"{name}.csv") for name in tables}))
    print(f"Results and numerical checks written to {directory}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/allocator_study.json")
    run(load_config(parser.parse_args().config))
