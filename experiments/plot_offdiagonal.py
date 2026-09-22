"""Stage 3: strict diagonal-only storage, certificate, and rank diagnostics."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from plot_paper import tolerance_axis, unique_rows
from plot_results import line_style, load_matplotlib, savefig
from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path, write_csv


def validate_results(cfg, rows):
    if cfg["hmatrix"]["storage_policy"] != "diagonal_only" or not rows:
        raise ValueError("expected nonempty diagonal-only results")
    unique_rows(rows, ("case_id", "budget_policy", "tol"))
    for row in rows:
        if row["storage_policy"] != "diagonal_only" or int(row["fallback_dense_leaves"]):
            raise ValueError("mixed policies or forbidden off-diagonal dense fallback")
        if row["feasible"] == "1":
            total = int(row["total_bits_with_metadata"])
            if total != sum(int(row[f"{k}_bits"]) for k in ("diagonal", "admissible", "offdiagonal_inadmissible")):
                raise ValueError("inconsistent storage decomposition")
            if int(row["stored_dense_bits"]) != int(row["diagonal_bits"]):
                raise ValueError("non-diagonal dense storage detected")
        elif row["feasible"] != "0" or row["total_compression"] or row["global_actual_error"]:
            raise ValueError("infeasible configuration must not report storage/error results")


def plot_case(cfg, plt, directory, rows, case, policy):
    items = sorted((r for r in rows if r["case_id"] == case and r["budget_policy"] == policy), key=lambda r: float(r["tol"]))
    if [float(r["tol"]) for r in items] != sorted(cfg["tolerances"]):
        raise ValueError(f"incomplete tolerance sweep: {case}, {policy}")
    xs = [float(r["tol"]) for r in items]
    def values(fn):
        return [fn(r) if r["feasible"] == "1" else float("nan") for r in items]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
    for index, field, label in ((0, "global_actual_error", "Measured error"), (1, "global_cert", "Certificate")):
        axes[0].loglog(xs, values(lambda r: float(r[field]) or float("nan")), label=label, **line_style(cfg, index))
    axes[0].set_title("Reconstruction")
    axes[0].set_ylabel("max-norm error")
    for index, label, numerator in ((2, "Dense FP64", "dense64_bits"),
                                    (3, "Same-partition FP32", "same_partition_fp32_bits"),
                                    (4, "Same-partition FP64", "same_partition_fp64_bits")):
        axes[1].semilogx(xs, values(lambda r: int(r[numerator])/int(r["total_bits_with_metadata"])), label=label, **line_style(cfg, index))
    axes[1].axhline(1, color="0.6", linewidth=.8)
    axes[1].set_title("Storage ratios")
    axes[1].set_ylabel("baseline / stored bits")
    for index, kind, label in ((5, "diagonal", "Diagonal dense"),
                               (6, "admissible", "Admissible factors"),
                               (7, "offdiagonal_inadmissible", "Inadmissible off-diagonal factors")):
        axes[2].semilogx(xs, values(lambda r: 100*int(r[f"{kind}_bits"])/int(r["total_bits_with_metadata"])), label=label, **line_style(cfg, index))
    axes[2].set_title("Storage composition")
    axes[2].set_ylabel("stored bits (%)")
    axes[2].set_ylim(-2, 102)
    for ax in axes:
        tolerance_axis(ax, cfg)
    savefig(fig, directory, f"{case}_{policy}", cfg["plot_style"]["formats"], cfg)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/offdiagonal_a100.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = resolve_path(cfg["results_dir"])
    if read_json(results / "numerical_validation.json")["config"] != cfg:
        raise ValueError("results/config mismatch; do not mix runs")
    rows = read_csv(results / "hmatrix_certificate.csv")
    validate_results(cfg, rows)
    directory, plt = ensure_dir(cfg["figures_dir"]), load_matplotlib(cfg)
    for case, policy in sorted({(r["case_id"], r["budget_policy"]) for r in rows}):
        fig = plot_case(cfg, plt, directory, rows, case, policy)
        plt.close(fig)
    leaves = read_csv(results / "hmatrix_leaves.csv")
    ranks, groups = [], defaultdict(list)
    for row in leaves:
        groups[(row["case_id"], row["budget_policy"], row["tol"], row["block_kind"])].append(row)
    for (case, policy, tol, kind), group in sorted(groups.items()):
        successful = [r for r in group if r["stored"] == "1" and r["low_rank"] == "1"]
        fractions = [int(r["rank"])/min(int(r["row1"])-int(r["row0"]), int(r["col1"])-int(r["col0"])) for r in successful]
        ranks.append(dict(case_id=case, budget_policy=policy, tol=tol, block_kind=kind,
                          leaves=len(group), failed=sum(r["stored"] == "0" for r in group),
                          median_rank_fraction=float(np.median(fractions)) if fractions else "",
                          max_rank_fraction=max(fractions, default="")))
    write_csv(directory / "rank_diagnostics.csv", ranks, list(ranks[0]))
    print(f"Figures and rank diagnostics: {directory}")


if __name__ == "__main__":
    main()
