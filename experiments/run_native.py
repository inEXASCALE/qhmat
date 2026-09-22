from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description="Stage 2: standalone C++/CUDA performance and validation")
    parser.add_argument("--config", default="experiments/configs/a100.json")
    parser.add_argument("--binary", default="experiments/cuda/build/qmat_cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    binary = resolve_path(args.binary)
    if not binary.is_file():
        raise FileNotFoundError(f"Build the CUDA target first: {binary}")
    manifest = read_json(resolve_path(cfg["data_dir"]) / "manifest.json")
    if not manifest.get("native_gpu"):
        raise ValueError("No native inputs. Rerun prepare_data.py with GPU enabled.")
    for key in ("seed", "low_rank", "gpu", "bit_set", "geometric_suites", "hmatrix"):
        if manifest.get("config", {}).get(key) != cfg.get(key):
            raise ValueError(f"Prepared {key} config differs; rerun prepare_data.py")
    results = ensure_dir(cfg["results_dir"])
    raw_dir = ensure_dir(results / "native_raw")
    gpu = cfg["gpu"]
    modes = gpu.get("native_modes", ["uniform4", "uniform8", "uniform12", "uniform16", "mixed", "adaptive"])
    rows, inputs = [], []
    output_path = results / "native_performance.csv"
    # Retain raw per-case output on failure, but never present a stale aggregate as this run.
    output_path.unlink(missing_ok=True)
    for item in manifest["native_gpu"]:
        data_path = resolve_path(item["path"])
        inputs.append(dict(case_id=item["case_id"], sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()))
        for mode in modes:
            raw = raw_dir / f"{item['case_id']}_{mode}.csv"
            command = [str(binary), str(data_path), mode, str(gpu["warmup"]), str(gpu["repeats"]), str(raw)]
            subprocess.run(command, check=True)
            case_rows = read_csv(raw)
            if not case_rows:
                raise RuntimeError(f"No native results for {item['case_id']} / {mode}")
            rows.extend(dict(case_id=item["case_id"], family=item.get("family", "synthetic"),
                source_case=item.get("source_case", "synthetic"), source_leaf=item.get("leaf_id", ""),
                source_eta=item.get("eta", ""), **row) for row in case_rows)
            print(f"{item['case_id']} {mode}: {len(case_rows)} measurements", flush=True)
    write_csv(output_path, rows, list(rows[0]))
    write_json(results / "native_run.json", dict(config=cfg, platform=platform.platform(),
               binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), inputs=inputs))
    print(json.dumps(dict(output=str(output_path), rows=len(rows)), indent=2))


if __name__ == "__main__":
    main()
