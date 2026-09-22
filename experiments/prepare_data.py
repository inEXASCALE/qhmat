from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qmat_exp.io import ensure_dir, load_config, read_json, write_json
from qmat_exp.matrices import hmatrix_case, load_leaves, synthetic_low_rank_case
from qmat_exp.native_data import prepare_native_data
from qmat_exp.native_data import export_geometric_native
from qmat_exp.geometric_matrices import geometric_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare synthetic data for qmat paper experiments.")
    parser.add_argument("--config", default="experiments/configs/offdiagonal_smoke.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])
    data_dir = ensure_dir(cfg["data_dir"])

    manifest: dict = {"low_rank": [], "hmatrix": [],
                      "partition_schema": "geometric_admissibility_v1"}

    lr_cfg = cfg["low_rank"]
    for size in lr_cfg["sizes"]:
        for spectrum in lr_cfg["spectra"]:
            case_id = f"lowrank_n{size}_r{lr_cfg['rank']}_{spectrum}"
            case = synthetic_low_rank_case(case_id, size, lr_cfg["rank"], spectrum, lr_cfg, rng)
            path = data_dir / f"{case_id}.npz"
            np.savez_compressed(
                path,
                u=case["u"],
                v=case["v"],
                sigma=case["sigma"],
                matrix=case["matrix"],
            )
            manifest["low_rank"].append(
                {
                    "case_id": case_id,
                    "path": str(path),
                    "size": size,
                    "rank": lr_cfg["rank"],
                    "spectrum": spectrum,
                }
            )

    h_cfg = cfg["hmatrix"]
    h_sizes = h_cfg.get("sizes", [h_cfg.get("n")])
    for size in h_sizes:
        if size is None:
            raise ValueError("hmatrix config must define either 'sizes' or 'n'")
        for kernel in h_cfg["kernels"]:
            case_id = f"hmatrix_n{size}_leaf{h_cfg['leaf_size']}_{kernel}"
            case = hmatrix_case(
                case_id,
                int(size),
                kernel,
                h_cfg["leaf_size"],
                h_cfg["admissibility_eta"],
            )
            path = data_dir / f"{case_id}.npz"
            leaves_path = data_dir / f"{case_id}_leaves.json"
            np.savez_compressed(path, points=case["points"], matrix=case["matrix"])
            write_json(leaves_path, case["leaves"])
            manifest["hmatrix"].append(
                {
                    "case_id": case_id,
                    "path": str(path),
                    "leaves_path": str(leaves_path),
                    "n": int(size),
                    "kernel": kernel,
                    "leaf_count": len(case["leaves"]),
                    "low_rank_leaf_count": sum(1 for item in case["leaves"] if item["low_rank"]),
                    "dense_leaf_count": sum(1 for item in case["leaves"] if not item["low_rank"]),
                }
            )

    manifest["native_gpu"] = prepare_native_data(cfg, data_dir)
    for case in geometric_cases(cfg.get("geometric_suites", {}), h_cfg, cfg["seed"]):
        path = data_dir / f"{case['case_id']}.npz"
        leaves_path = data_dir / f"{case['case_id']}_leaves.json"
        arrays = {key: value for key, value in case.items() if isinstance(value, np.ndarray)}
        if not np.isfinite(case["matrix"]).all():
            raise ValueError(f"Nonfinite assembled matrix: {case['case_id']}")
        np.savez_compressed(path, **arrays)
        write_json(leaves_path, [leaf.to_dict() for leaf in case["leaves"]])
        manifest["hmatrix"].append(dict(case_id=case["case_id"], path=str(path),
            leaves_path=str(leaves_path), n=len(case["matrix"]), kernel=case["kernel"],
            family=case["family"], geometry=case["geometry"], metadata=case["metadata"],
            leaf_count=len(case["leaves"]),
            low_rank_leaf_count=sum(leaf.low_rank for leaf in case["leaves"]),
            dense_leaf_count=sum(not leaf.low_rank for leaf in case["leaves"]),
            dense_bytes=case["matrix"].nbytes,
            symmetry_max_error=float(np.max(np.abs(case["matrix"] - case["matrix"].T)))))
        manifest["native_gpu"].extend(export_geometric_native(case, cfg, data_dir))
        print(f"prepared {case['case_id']}", flush=True)
    manifest["config"] = cfg
    for item in manifest["hmatrix"]:
        leaves = load_leaves(read_json(item["leaves_path"]))
        item["admissible_leaf_count"] = sum(l.low_rank for l in leaves)
        item["diagonal_leaf_count"] = sum(l.diagonal for l in leaves)
        item["offdiagonal_inadmissible_leaf_count"] = sum(l.off_diagonal and not l.low_rank for l in leaves)
        item["quantization_candidate_count"] = sum(
            l.off_diagonal if h_cfg.get("storage_policy", "diagonal_only") == "diagonal_only" else l.low_rank
            for l in leaves)
    write_json(data_dir / "manifest.json", manifest)
    print(json.dumps({"data_dir": str(data_dir), "low_rank_cases": len(manifest["low_rank"]), "hmatrix_cases": len(manifest["hmatrix"]), "native_gpu_cases": len(manifest["native_gpu"])}, indent=2))


if __name__ == "__main__":
    main()
