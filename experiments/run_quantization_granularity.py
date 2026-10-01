"""Stage 2: compare matched factor-quantization granularities on existing matrices."""
from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess

import numpy as np

from qmat_exp.granularity import compare_hmatrix_case
from qmat_exp.io import ensure_dir, load_config, read_json, repo_root, resolve_path, write_csv, write_json
from qmat_exp.matrices import load_leaves


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cfg):
    comparison = cfg["granularity"]
    cases = comparison["cases"]
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)) or not ids or any(case["panel"] not in ("small", "large") for case in cases):
        raise ValueError("comparison cases must be unique and assigned to a size panel")
    if cfg["hmatrix"]["storage_policy"] != "diagonal_only" or comparison["budget_policy"] not in cfg["hmatrix"]["budget_policies"]:
        raise ValueError("comparison must use a configured diagonal-only budget policy")
    if comparison["scale_bits"] <= 0 or not cfg["tolerances"]:
        raise ValueError("invalid scale width or tolerance grid")
    data_dir = resolve_path(cfg["data_dir"])
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}; prepare the offdiagonal data first")
    manifest = read_json(manifest_path)
    for key in ("seed", "hmatrix", "geometric_suites"):
        if manifest["config"].get(key) != cfg.get(key):
            raise ValueError(f"prepared {key} differs from this comparison configuration")
    available = {item["case_id"]: item for item in manifest["hmatrix"]}
    if any(case_id not in available for case_id in ids):
        raise ValueError("selected case missing from prepared matrix manifest")
    inputs = {}
    for case_id in ids:
        item = available[case_id]
        inputs[case_id] = {key: sha256(resolve_path(item[key])) for key in ("path", "leaves_path")}

    directory = ensure_dir(cfg["results_dir"])
    raw = ensure_dir(directory / "raw")
    rows = []
    for case_id in ids:
        print(f"running {case_id}", flush=True)
        item = available[case_id]
        with np.load(resolve_path(item["path"]), allow_pickle=False) as arrays:
            matrix = arrays["matrix"]
        leaves = load_leaves(read_json(item["leaves_path"]))
        case_rows = compare_hmatrix_case(case_id, item["family"], matrix, leaves, cfg)
        write_csv(raw / f"{case_id}.csv", case_rows, list(case_rows[0]))
        rows.extend(case_rows)
        print(f"completed {case_id}", flush=True)
    expected = len(ids) * len(cfg["tolerances"]) * 3
    if len(rows) != expected:
        raise AssertionError("incomplete granularity comparison")
    result_path = directory / "granularity_results.csv"
    write_csv(result_path, rows, list(rows[0]))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(),
                              capture_output=True, text=True)
    sources = ["experiments/run_quantization_granularity.py",
               "experiments/qmat_exp/granularity.py",
               "experiments/qmat_exp/hmatrix.py",
               "experiments/qmat_exp/quantization.py"]
    write_json(directory / "granularity_run.json", dict(
        config=cfg, manifest_sha256=sha256(manifest_path), input_sha256=inputs,
        source_sha256={name: sha256(repo_root() / name) for name in sources},
        result_sha256=sha256(result_path), rows=len(rows),
        feasible_rows=sum(int(row["feasible"]) for row in rows),
        python=platform.python_version(), numpy=np.__version__,
        git_revision=revision.stdout.strip(),
        scope="FP64 factor-product emulation; quantization and truncation bounds only; no kernel timing"))
    print(f"Granularity comparison written to {result_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/quantization_granularity.json")
    run(load_config(parser.parse_args().config))
