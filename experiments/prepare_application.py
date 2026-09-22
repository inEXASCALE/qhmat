"""Stage 1: export complete, fixed-partition H-matrix applications."""
import argparse
import hashlib
import time

import numpy as np

from qmat_exp.application import build_application, write_application
from qmat_exp.io import ensure_dir, load_config, read_json, resolve_path, write_json
from qmat_exp.matrices import Leaf, hmatrix_case


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(cfg):
    for key in ("rhs_counts", "tolerances", "case_ids"):
        if not cfg[key] or len(cfg[key]) != len(set(cfg[key])):
            raise ValueError(f"empty or duplicated {key}")
    if any(type(p) is not int or not 1 <= p <= 4096 for p in cfg["rhs_counts"]):
        raise ValueError("invalid RHS counts")
    if any(not np.isfinite(t) or t <= 0 for t in cfg["tolerances"]):
        raise ValueError("invalid tolerances")
    if cfg["warmup"] < 0 or min(cfg["repeats"], cfg["trials"]) < 1:
        raise ValueError("invalid timing repetitions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/application_a100.json")
    parser.add_argument("--smoke", action="store_true", help="Small generated fixture, not publication data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    source = load_config(cfg["source_config"])
    directory = ensure_dir(cfg["data_dir"])
    if args.smoke:
        directory = ensure_dir(directory / "smoke")
        fixture = hmatrix_case("smoke", 65, "gaussian1d", 8, 1.0)
        items = [dict(case_id="smoke", fixture=fixture)]
        tolerances = [0.05]
    else:
        manifest = read_json(resolve_path(source["data_dir"]) / "manifest.json")
        for key in ("seed", "hmatrix", "geometric_suites"):
            prepared, requested = manifest["config"].get(key), source.get(key)
            if key == "hmatrix":
                prepared, requested = dict(prepared), dict(requested)
                prepared.setdefault("storage_policy", "admissible_only")
                requested.setdefault("storage_policy", "admissible_only")
            if prepared != requested:
                raise ValueError(f"source manifest/config mismatch: {key}")
        by_id = {item["case_id"]: item for item in manifest["hmatrix"]}
        missing = set(cfg["case_ids"]) - set(by_id)
        if missing:
            raise ValueError(f"Run prepare_data.py first; missing matrices: {sorted(missing)}")
        items = [by_id[key] for key in cfg["case_ids"]]
        tolerances = cfg["tolerances"]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Choose a new data_dir or explicitly remove the previous manifest: {manifest_path}")
    entries = []
    for item in items:
        if args.smoke:
            matrix = item["fixture"]["matrix"]
            leaves = [Leaf(**v) for v in item["fixture"]["leaves"]]
            permutation = np.arange(len(matrix))
            sources = {}
        else:
            with np.load(resolve_path(item["path"]), allow_pickle=False) as arrays:
                matrix = arrays["matrix"]
                permutation = arrays["permutation"] if "permutation" in arrays else np.arange(len(matrix))
            leaves = [Leaf(**v) for v in read_json(item["leaves_path"])]
            sources = {key: sha256(resolve_path(item[key])) for key in ("path", "leaves_path")}
        rng = np.random.default_rng(cfg["seed"])
        x = rng.standard_normal((len(matrix), max(cfg["rhs_counts"]))).astype(np.float32)
        cache = {}
        # First construction includes SVD; later tolerances reuse it, explicitly recorded.
        for index, tol in enumerate(tolerances):
            start = time.perf_counter()
            data = build_application(matrix, leaves, permutation, x, source, tol, cache)
            construction_ms = (time.perf_counter()-start)*1000
            name = f"{item['case_id']}_tol{tol:g}.bin"
            path = directory / name
            write_application(path, data)
            entries.append(dict(case_id=item["case_id"], tol=tol, file=name, sha256=sha256(path),
                source_hashes=sources, n=len(matrix), max_rhs=data["rhs"], fallback_count=data["fallback_count"],
                compressed_leaves=sum(l["rank"] > 0 for l in data["leaves"]),
                preparation_and_reference_ms=construction_ms, reused_svd=bool(index)))
            print(f"prepared {name}", flush=True)
    write_json(manifest_path, dict(config=cfg, source_config=source, smoke=args.smoke, inputs=entries))
    print(f"Application manifest: {manifest_path}")


if __name__ == "__main__":
    main()
