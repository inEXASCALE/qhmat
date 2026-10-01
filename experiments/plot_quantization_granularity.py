"""Stage 3: matched storage and error-bound figures for scaling granularity."""
from __future__ import annotations

import argparse
import math

from plot_paper import panel_title_size
from plot_results import apply_axis_style, line_style, load_matplotlib, savefig
from qmat_exp.granularity import METHODS
from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path
from run_quantization_granularity import sha256


LABELS = {"per_tensor": "Per-tensor, fixed bits",
          "per_channel": "Per-channel, fixed bits",
          "adaptive": "Adaptive, per-channel"}
PLOTS = (("storage_fraction_fp64_factors", "Stored bits / FP64 factor bits", "storage"),
         ("max_budget_fraction", "Max. block bound / budget", "bound"))


def validate_rows(cfg, rows):
    expected = {(case["case_id"], float(tol), method)
                for case in cfg["granularity"]["cases"]
                for tol in cfg["tolerances"] for method in METHODS}
    seen = set()
    for row in rows:
        key = (row["case_id"], float(row["tol"]), row["method"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected granularity result")
        seen.add(key)
        if row["budget_policy"] != cfg["granularity"]["budget_policy"]:
            raise ValueError("mixed budget policies")
        if row["feasible"] == "1":
            total = sum(int(row[name]) for name in ("diagonal_bits", "payload_bits", "scale_bits", "map_bits"))
            reference = int(row["same_partition_fp64_bits"])
            fraction = float(row["storage_fraction_fp64_factors"])
            actual = float(row["global_actual_error"])
            bound = float(row["global_representation_bound"])
            usage = float(row["max_budget_fraction"])
            if (total != int(row["total_bits"]) or reference <= 0
                    or not math.isfinite(fraction)
                    or not math.isclose(fraction, total / reference, rel_tol=1e-10)
                    or actual > bound + 1e-12 * max(1.0, bound)
                    or not 0 <= usage <= 1 + 1e-10):
                raise ValueError("invalid storage accounting")
        elif row["feasible"] != "0" or row["total_bits"] or row["global_actual_error"]:
            raise ValueError("infeasible rows must not report partial storage or error")
    if seen != expected:
        raise ValueError("incomplete granularity comparison")


def make_figure(cfg, plt, rows, panel, field, ylabel):
    cases = [case for case in cfg["granularity"]["cases"] if case["panel"] == panel]
    fig, axes = plt.subplots(1, len(cases), figsize=(7.5, 3.0), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    ticks = [1e-6, 1e-4, 1e-2, 1e-1]
    for ax, case in zip(axes, cases):
        case_rows = [r for r in rows if r["case_id"] == case["case_id"]]
        n = int(case_rows[0]["n"])
        for index, method in enumerate(METHODS):
            selected = sorted((r for r in case_rows if r["method"] == method),
                              key=lambda r: float(r["tol"]))
            xs = [float(row["tol"]) for row in selected]
            ys = [float(row[field]) if row["feasible"] == "1" else float("nan")
                  for row in selected]
            ax.semilogx(xs, ys, label=LABELS[method], **line_style(cfg, index))
        ax.set_title(f"{case['label']}\n$N={n}$", fontsize=panel_title_size(cfg))
        ax.set_xlabel("Relative tolerance")
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{-1}$"])
        ax.set_xlim(min(cfg["tolerances"]) / 1.5, max(cfg["tolerances"]) * 1.5)
        apply_axis_style(ax, cfg)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)
    if field == "max_budget_fraction":
        axes[0].axhline(1, color="0.55", linewidth=0.8)
        axes[0].set_ylim(0, 1.05)
    return fig


def plot(cfg):
    directory = resolve_path(cfg["results_dir"])
    result_path = directory / "granularity_results.csv"
    provenance = read_json(directory / "granularity_run.json")
    if provenance["config"] != cfg or provenance["result_sha256"] != sha256(result_path):
        raise ValueError("results/config mismatch or modified result file")
    rows = read_csv(result_path)
    validate_rows(cfg, rows)
    plt = load_matplotlib(cfg)
    figures = ensure_dir(cfg["figures_dir"])
    for panel in ("small", "large"):
        for field, ylabel, name in PLOTS:
            fig = make_figure(cfg, plt, rows, panel, field, ylabel)
            savefig(fig, figures, f"supp_granularity_{name}_{panel}",
                    cfg["plot_style"]["formats"], cfg)
            plt.close(fig)
    print(f"Granularity figures written to {figures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/quantization_granularity.json")
    plot(load_config(parser.parse_args().config))
