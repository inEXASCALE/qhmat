"""Stage 3: matched-workload application plots; never pool sizes or tolerances."""
import argparse
from collections import defaultdict

import numpy as np

from plot_results import apply_axis_style, line_style, load_matplotlib, savefig
from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path
from run_application import METHODS, validate_rows


LABELS = ("Dense FP32", "H FP32, batched", "Adaptive, unpack + batched",
          "Adaptive, predecoded", "H FP32, fused", "Adaptive, packed fused")


def plot_case(cfg, plt, rows, directory, name):
    rhs = sorted(cfg["rhs_counts"])
    if len(rows) != len(rhs)*len(METHODS)*cfg["trials"]:
        raise ValueError("unexpected rows in application figure")
    for p in rhs:
        selected = [r for r in rows if int(r["rhs"]) == p]
        if not selected:
            raise ValueError(f"missing RHS count {p}")
        validate_rows(selected, cfg, int(selected[0]["n"]), p)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    fields = ("speedup_vs_dense", "speedup_vs_h_fp32", "relative_max_error")
    for index, (method, label) in enumerate(zip(METHODS, LABELS)):
        for ax, field in zip(axes, fields):
            values = [[float(r[field]) for r in rows if r["method"] == method and int(r["rhs"]) == p] for p in rhs]
            med = [np.median(v) for v in values]
            ax.plot(rhs, med, label=label, **line_style(cfg, index))
            if field != "relative_max_error":
                ax.fill_between(rhs, [min(v) for v in values], [max(v) for v in values], alpha=.12,
                                color=ax.lines[-1].get_color())
    for ax, label in zip(axes, ("speedup vs dense FP32", "speedup vs H FP32 batched", "relative max-norm error")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(rhs, [str(p) for p in rhs])
        ax.set_xlabel("right-hand sides")
        ax.set_ylabel(label)
        if ax is axes[-1]:
            errors = [float(r["relative_max_error"]) for r in rows]
            if min(errors) > 0:
                ax.set_yscale("log")
            elif max(errors) > 0:
                # Retain exact zero without inventing a positive error floor.
                ax.set_yscale("symlog", linthresh=min(e for e in errors if e > 0)/10)
                ax.set_ylim(0, max(errors)*1.5)
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([0])
        else:
            ax.set_yscale("log")
            ax.axhline(1, color="black", linestyle=":", linewidth=1)
        apply_axis_style(ax, cfg)
    savefig(fig, directory, name, cfg["plot_style"]["formats"], cfg)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/application_a100.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = load_config(cfg["source_config"])
    cfg = dict(cfg, plot_style=source["plot_style"])
    results = resolve_path(cfg["results_dir"]) / ("smoke" if args.smoke else "")
    run = read_json(results / "application_run.json")
    if run["status"] != "complete" or run["smoke"] != args.smoke:
        raise ValueError("only completed, matching runs may be plotted")
    if run["config"] != {k: v for k, v in cfg.items() if k != "plot_style"}:
        raise ValueError("plot config differs from measurement config")
    rows = read_csv(results / "application_performance.csv")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["case_id"], float(r["tol"]))].append(r)
    expected = {(r["case_id"], float(r["tol"])) for r in run["inputs"]}
    if set(grouped) != expected:
        raise ValueError("missing or unexpected cases in application results")
    directory = ensure_dir(resolve_path(cfg["figures_dir"]) / ("smoke" if args.smoke else ""))
    plt = load_matplotlib(cfg)
    for (case, tol), values in grouped.items():
        plt.close(plot_case(cfg, plt, values, directory, f"apply_{case}_tol{tol:g}"))
    print(f"Application figures written to {directory}")


if __name__ == "__main__":
    main()
