"""Recover same-partition storage controls from a run's saved leaf records."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from qmat_exp.io import load_config, read_csv, resolve_path, write_csv
from qmat_exp.storage import same_partition_storage


KEYS = ("case_id", "tol", "budget_policy")
COUNTERS = ("payload_bits", "scale_bits", "map_bits", "dense64_bits",
            "low_rank_leaves", "dense_leaves", "feasible_low_rank_leaves", "fallback_dense_leaves")
BASELINES = ("stored_dense_bits", "retained_factor_entries",
             "same_partition_fp32_bits", "same_partition_fp64_bits")


def summarize(leaves, summaries):
    totals = defaultdict(lambda: defaultdict(int))
    seen = defaultdict(set)
    for leaf in leaves:
        key = tuple(leaf[k] for k in KEYS)
        if leaf["leaf_id"] in seen[key]:
            raise ValueError(f"Duplicate leaf in {key}: {leaf['leaf_id']}")
        seen[key].add(leaf["leaf_id"])
        total = totals[key]
        for field, value in same_partition_storage([leaf]).items():
            total[field] += value
        for field in ("payload_bits", "scale_bits", "map_bits"):
            total[field] += int(leaf[field])
        m = int(leaf["row1"]) - int(leaf["row0"])
        n = int(leaf["col1"]) - int(leaf["col0"])
        total["dense64_bits"] += 64 * m * n
        low_rank, feasible = int(leaf["low_rank"]), int(leaf["feasible"])
        total["low_rank_leaves"] += low_rank
        total["dense_leaves"] += 1 - low_rank
        total["feasible_low_rank_leaves"] += low_rank * feasible
        total["fallback_dense_leaves"] += low_rank * (1 - feasible)

    output = []
    summary_keys = set()
    for summary in summaries:
        key = tuple(summary[k] for k in KEYS)
        if key in summary_keys:
            raise ValueError(f"Duplicate summary: {key}")
        summary_keys.add(key)
        if key not in totals:
            raise ValueError(f"Missing leaf records for {key}")
        total = totals[key]
        for field in COUNTERS:
            if total[field] != int(summary[field]):
                raise ValueError(f"Leaf/summary mismatch for {key}: {field}")
        stored = sum(total[k] for k in ("payload_bits", "scale_bits", "map_bits"))
        if stored != int(summary["total_bits_with_metadata"]):
            raise ValueError(f"Stored-total mismatch for {key}")
        output.append({**dict(zip(KEYS, key)), "total_bits_with_metadata": stored,
                       **{k: total[k] for k in BASELINES}})
    if set(totals) != summary_keys:
        raise ValueError("Leaf records contain cases not present in the summary")
    if not output:
        raise ValueError("No matching records")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/a100.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    directory = resolve_path(cfg["results_dir"])
    source = directory / "hmatrix_leaves.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; run on the node holding this run's leaf records")
    with source.open(newline="") as stream:
        rows = summarize(csv.DictReader(stream), read_csv(directory / "hmatrix_certificate.csv"))
    output = directory / "hmatrix_storage_baselines.csv"
    write_csv(output, rows, list(KEYS) + ["total_bits_with_metadata"] + list(BASELINES))
    print(f"Verified and wrote {len(rows)} same-partition storage controls to {output}")


if __name__ == "__main__":
    main()
