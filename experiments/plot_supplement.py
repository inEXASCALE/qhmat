"""Focused supplemental diagnostics from the archived A100 measurements."""
from __future__ import annotations

import argparse
import json

from plot_paper import plot_allocator, plot_native, tolerance_axis, unique_rows
from plot_results import LINE_PALETTE, apply_axis_style, line_style, load_matplotlib, savefig
from qmat_exp.io import ensure_dir, load_config, read_csv, resolve_path


MODES = ("uniform4", "uniform8", "uniform12", "uniform16", "mixed", "adaptive")
MODE_LABELS = ("U4", "U8", "U12", "U16", "Mixed", "Adaptive")


def local_cases(cfg, n):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv")
    result = []
    for spectrum in ("exponential", "algebraic", "flat"):
        selected = [r for r in rows if r["case_id"] == f"lowrank_n{n}_r64_{spectrum}"
                    and r["method"] == "reserve_greedy"]
        unique_rows(selected, ("tol",))
        if sorted(float(r["tol"]) for r in selected) != sorted(cfg["tolerances"]):
            raise ValueError(f"Incomplete local sweep: {n}, {spectrum}")
        result.append((spectrum, sorted([r for r in selected if r["feasible"] == "1"],
                                       key=lambda r: float(r["tol"]))))
    return result


def plot_local_bounds(cfg, plt, directory, n):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, (spectrum, items) in zip(axes, local_cases(cfg, n)):
        for index, (field, label) in enumerate((("actual_error", "Measured error"),
                ("cert_sv", "SV certificate"), ("cert_geom", "Geometry certificate"))):
            ax.loglog([float(r["tol"]) for r in items],
                      [float(r[field]) / float(r["budget"]) for r in items],
                      label=label, **line_style(cfg, index))
        ax.set_title(spectrum.capitalize())
        tolerance_axis(ax, cfg)
    axes[0].set_ylabel("error or bound / budget")
    savefig(fig, directory, f"supp_local_bounds_n{n}", cfg["plot_style"]["formats"], cfg)
    return fig


def plot_precision_maps(cfg, plt, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (spectrum, items) in zip(axes, local_cases(cfg, 256)[:2]):
        for row in items:
            index = cfg["tolerances"].index(float(row["tol"]))
            bits = json.loads(row["bits_by_index"])
            ax.step(range(1, len(bits) + 1), bits, where="mid",
                    label=f"tol={row['tol']}", **line_style(cfg, index))
        ax.set_title(spectrum.capitalize())
        ax.set_xlabel("singular-value index")
        ax.set_xticks([1, 16, 32, 48, 64])
        ax.set_yticks(cfg["bit_set"])
        ax.set_ylim(3, 17)
        apply_axis_style(ax, cfg)
    axes[0].set_ylabel("assigned bits")
    savefig(fig, directory, "supp_precision_maps", cfg["plot_style"]["formats"], cfg)
    return fig


def plot_kernel_storage(cfg, plt, directory):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "hmatrix_certificate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, n in zip(axes, (512, 1024)):
        for index, (kernel, label) in enumerate((("laplace1d", "Regularized inverse distance"),
                ("gaussian1d", "Gaussian"), ("matern1d", "Matern 3/2"))):
            items = [r for r in rows if r["family"] == "kernel1d" and int(r["n"]) == n
                     and r["kernel"] == kernel and r["budget_policy"] == "block_relative"]
            unique_rows(items, ("tol",))
            items.sort(key=lambda r: float(r["tol"]))
            if [float(r["tol"]) for r in items] != sorted(cfg["tolerances"]):
                raise ValueError(f"Incomplete kernel sweep: {kernel}, {n}")
            ax.semilogx([float(r["tol"]) for r in items],
                        [float(r["total_compression"]) for r in items],
                        label=label, **line_style(cfg, index))
        ax.set_title(f"N={n}")
        tolerance_axis(ax, cfg)
    axes[0].set_ylabel("dense FP64 / stored bits")
    savefig(fig, directory, "supp_kernel_storage", cfg["plot_style"]["formats"], cfg)
    return fig


def real_leaf_groups(rows):
    groups = []
    reference_ids = None
    for mode in MODES:
        selected = [r for r in rows if r["source_case"] != "synthetic"
                    and r["mode"] == mode and r["kernel"] == "packed_columnwise_channels"]
        indexed = unique_rows(selected, ("case_id",))
        ids = set(indexed)
        if len(ids) != 68 or (reference_ids is not None and ids != reference_ids):
            raise ValueError(f"Expected the same 68 real leaves in every mode: {mode}")
        reference_ids = ids
        groups.append([indexed[key] for key in sorted(ids)])
    return groups


def plot_real_leaves(cfg, plt, directory):
    groups = real_leaf_groups(read_csv(resolve_path(cfg["results_dir"]) / "native_performance.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, field in zip(axes, ("speedup_vs_fp32", "relative_max_error")):
        boxes = ax.boxplot([[float(r[field]) for r in group] for group in groups],
                           whis=(0, 100), patch_artist=True, widths=0.6, showfliers=False)
        for index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(LINE_PALETTE[index])
            box.set_alpha(0.65)
        for median in boxes["medians"]:
            median.set_color("black")
        ax.set_xticks(range(1, 7), MODE_LABELS, rotation=30, ha="right")
        ax.set_xlabel("precision map")
        apply_axis_style(ax, cfg)
    axes[0].set_ylabel("speedup vs strict FP32")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("relative max-norm error")
    axes[1].set_yscale("log")
    savefig(fig, directory, "supp_native_real_leaves", cfg["plot_style"]["formats"], cfg)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/a100.json")
    parser.add_argument("--only", choices=("allocator",),
                        help="Regenerate only the n=512 allocator PDF/PNG")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.only and cfg.get("hmatrix", {}).get("storage_policy") == "diagonal_only":
        raise ValueError("Use plot_offdiagonal.py for strict-policy hierarchical results")
    plt = load_matplotlib(cfg)
    directory = ensure_dir(cfg["figures_dir"])
    if args.only == "allocator":
        plt.close(plot_allocator(cfg, plt, directory, n=512))
        print(f"Allocator figure written to {directory}")
        return
    for n in (256, 512):
        plt.close(plot_local_bounds(cfg, plt, directory, n))
    plt.close(plot_allocator(cfg, plt, directory, n=512))
    for plot in (plot_precision_maps, plot_kernel_storage, plot_real_leaves, plot_native):
        plt.close(plot(cfg, plt, directory))
    print(f"Supplement figures written to {directory}")


if __name__ == "__main__":
    main()
