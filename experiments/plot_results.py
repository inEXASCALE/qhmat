from __future__ import annotations

import argparse
import ast
import json
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

from qmat_exp.io import ensure_dir, load_config, read_csv, resolve_path


LINE_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6F4C9B",
    "#17A398", "#B38B00", "#4C78A8", "#E45756", "#72B7B2",
)
LINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
MARKER_FILLS = (True, False, True, False, True, False, True, False, True, False)
BAR_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")
BAR_HATCHES = ("", "//", "\\\\", "xx", "..", "++")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot qmat paper experiment figures.")
    parser.add_argument("--config", default="experiments/configs/default.json")
    return parser.parse_args()


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def parse_counts(value: str) -> dict[str, int]:
    if not value:
        return {}
    try:
        return {str(k): int(v) for k, v in json.loads(value).items()}
    except json.JSONDecodeError:
        return {str(k): int(v) for k, v in ast.literal_eval(value).items()}


def default_budget_policy(cfg: dict) -> str:
    return cfg.get("hmatrix", {}).get("budget_policy", "block_relative")


def load_matplotlib(cfg: dict):
    cache_dir = ensure_dir("experiments/.mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = cfg["plot_style"]
    plt.rcParams.update(
        {
            "axes.labelsize": style["axis_label_size"],
            "xtick.labelsize": style["tick_label_size"],
            "ytick.labelsize": style["tick_label_size"],
            "legend.fontsize": style["legend_font_size"],
            "axes.titlesize": style["title_font_size"],
            "lines.linewidth": style["line_width"],
            "lines.markersize": style["marker_size"],
            "figure.dpi": style["figure_dpi"],
            "savefig.dpi": style["figure_dpi"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.labelpad": 8,
            "axes.titlepad": 12,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )
    return plt


def apply_axis_style(ax, cfg: dict, tolerance_ticks: bool = False) -> None:
    style = cfg["plot_style"]
    ax.tick_params(
        axis="both",
        which="major",
        length=style.get("major_tick_length", 4.0),
        width=style.get("major_tick_width", 0.8),
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=style.get("minor_tick_length", 2.0),
        width=style.get("minor_tick_width", 0.6),
    )
    if tolerance_ticks:
        ticks = [float(x) for x in cfg["tolerances"]]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.grid(True, which="major", alpha=style.get("grid_alpha", 0.25))
    ax.grid(True, which="minor", alpha=style.get("grid_alpha", 0.25) / 3)


def line_style(cfg: dict, index: int, *, color=None, marker=None, linestyle=None, filled=None) -> dict:
    style = cfg["plot_style"]
    color = color or LINE_PALETTE[index % len(LINE_PALETTE)]
    marker = marker or LINE_MARKERS[index % len(LINE_MARKERS)]
    filled = MARKER_FILLS[index % len(MARKER_FILLS)] if filled is None else filled
    base_size = style.get("marker_size", 6.0)
    return {
        "color": color,
        "linestyle": linestyle if linestyle is not None else LINESTYLES[index % len(LINESTYLES)],
        "marker": marker,
        "markersize": base_size * (1.0 + 0.08 * (index % 3)),
        "markerfacecolor": color if filled else "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.9 if filled else 1.35,
        "linewidth": style.get("line_width", 2.0),
    }


def style_bar_axis(ax, cfg: dict) -> None:
    apply_axis_style(ax, cfg)
    style = cfg["plot_style"]
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", which="major", alpha=style.get("grid_alpha", 0.25))
    ax.grid(False, axis="y", which="minor")
    ax.margins(y=0.08)


def bar_kwargs(index: int, *, stacked: bool = False) -> dict:
    return {
        "color": BAR_COLORS[index % len(BAR_COLORS)],
        "edgecolor": "#334155" if stacked else "#ffffff",
        "linewidth": 0.5 if stacked else 0.9,
        "alpha": 0.9,
        "hatch": BAR_HATCHES[index % len(BAR_HATCHES)],
    }


def bottom_legend(fig, cfg: dict):
    """Place a shared legend below all axis decorations, measured in pixels."""
    entries = {}
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            entries.setdefault(label, handle)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for legend in list(fig.legends):
        legend.remove()
    fig.tight_layout(pad=1.2)
    fig.canvas.draw()
    if not entries:
        return None
    style = cfg["plot_style"]
    renderer = fig.canvas.get_renderer()
    gap = style.get("legend_gap_points", 14) * fig.dpi / 72
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    anchor = (0.5, (bottom - gap) / fig.bbox.height)
    width = fig.bbox.width - 2 * style.get("legend_side_padding_points", 12) * fig.dpi / 72
    columns = min(len(entries), max(1, int(style.get("legend_max_columns", 4))))
    # Save with bbox_inches="tight": legend rows grow the canvas, not shrink axes.
    while True:
        legend = fig.legend(
            list(entries.values()), list(entries), loc="upper center",
            bbox_to_anchor=anchor, bbox_transform=fig.transFigure,
            ncol=columns, borderaxespad=0, borderpad=0,
            fontsize=style["legend_font_size"], frameon=False,
            columnspacing=style.get("legend_column_spacing", 1.4),
            labelspacing=style.get("legend_label_spacing", 0.6),
            handlelength=2.4,
        )
        fig.canvas.draw()
        if legend.get_window_extent(fig.canvas.get_renderer()).width <= width or columns == 1:
            return legend
        legend.remove()
        columns -= 1


def savefig(fig, fig_dir: Path, name: str, formats: list[str], cfg: dict) -> None:
    bottom_legend(fig, cfg)
    for fmt in formats:
        fig.savefig(fig_dir / f"{name}.{fmt}", bbox_inches="tight")


def case_label(case_id: str) -> str:
    low_rank = re.fullmatch(r"lowrank_n(\d+)_r(\d+)_(.+)", case_id)
    if low_rank:
        n, rank, spectrum = low_rank.groups()
        return f"{spectrum.capitalize()}, N={n}, r={rank}"
    old_kernel = re.fullmatch(r"hmatrix_n(\d+)_leaf\d+_(.+)", case_id)
    if old_kernel:
        n, kernel = old_kernel.groups()
        return f"{kernel.replace('1d', '').capitalize()}, N={n}"
    size = re.search(r"_n(\d+)$", case_id)
    return f"N={size[1]}" if size else case_id.replace("_", " ")


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def result_rows(cfg, filename):
    cache = cfg.setdefault("_result_cache", {})
    if filename not in cache:
        cache[filename] = read_csv(resolve_path(cfg["results_dir"]) / filename)
    rows = cache[filename]
    ids = cfg.get("_case_ids")
    return rows if ids is None else [row for row in rows if row["case_id"] in ids]


def plot_local_certificate(cfg: dict, plt, fig_dir: Path) -> None:
    rows = [r for r in read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv") if r["method"] == "reserve_greedy" and r["feasible"] == "1"]
    grouped = group_rows(rows, ("case_id",))
    if not grouped:
        return
    columns = min(3, len(grouped))
    fig, axes = plt.subplots((len(grouped) + columns - 1) // columns, columns,
                             figsize=(5 * columns, 3.8 * ((len(grouped) + columns - 1) // columns)), squeeze=False)
    for ax, (case_id, items) in zip(axes.flat, grouped.items()):
        items = sorted(items, key=lambda x: as_float(x, "tol"))
        xs = [as_float(x, "tol") for x in items]
        actual = [as_float(x, "actual_error") for x in items]
        cert = [as_float(x, "cert_sv") for x in items]
        geom = [as_float(x, "cert_geom") for x in items]
        ax.loglog(xs, actual, label="Measured error", **line_style(cfg, 0, marker="o", linestyle="-", filled=True))
        ax.loglog(xs, cert, label="Singular-value bound", **line_style(cfg, 1, marker="s", linestyle="--", filled=False))
        ax.loglog(xs, geom, label="Geometry bound", **line_style(cfg, 2, marker="^", linestyle="-.", filled=True))
        ax.set_xlabel("relative tolerance")
        ax.set_ylabel(r"$\|T-\widehat T\|_{\max}$")
        ax.set_title(case_label(case_id[0]))
        apply_axis_style(ax, cfg, tolerance_ticks=True)
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    savefig(fig, fig_dir, "local_certificate_validation", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_allocator_payload(cfg: dict, plt, fig_dir: Path) -> None:
    rows = [r for r in read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv") if r["feasible"] == "1"]
    methods = ["reserve_greedy", "dp_optimal", "naive_greedy"] + [f"uniform{b}" for b in cfg["bit_set"]]
    # Per-case panels avoid averaging different feasible subsets across methods.
    grouped = group_rows(rows, ("case_id",))
    if not grouped:
        return
    columns = min(3, len(grouped))
    nrows = (len(grouped) + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 3.8 * nrows), squeeze=False)
    for ax, ((case_id,), items) in zip(axes.flat, grouped.items()):
        for method_index, method in enumerate(methods):
            by_tol = {as_float(r, "tol"): r for r in items if r["method"] == method}
            xs = [float(t) for t in cfg["tolerances"]]
            ys = [as_float(by_tol[x], "payload_compression") if x in by_tol else float("nan") for x in xs]
            ax.semilogx(xs, ys, label=method, **line_style(cfg, method_index))
        ax.set_xlabel("relative tolerance")
        ax.set_ylabel("dense FP64 / payload bits")
        ax.set_title(case_label(case_id))
        apply_axis_style(ax, cfg, tolerance_ticks=True)
        ax.legend()
    for ax in list(axes.flat)[len(grouped):]:
        ax.set_visible(False)
    savefig(fig, fig_dir, "allocator_storage_comparison", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_allocator_optimality_gap(cfg: dict, plt, fig_dir: Path) -> None:
    rows = [r for r in read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv") if r["feasible"] == "1"]
    grouped = group_rows(rows, ("case_id", "tol"))
    by_tol: dict[float, list[float]] = defaultdict(list)
    for (_, tol), items in grouped.items():
        by_method = {item["method"]: item for item in items}
        if "reserve_greedy" not in by_method or "dp_optimal" not in by_method:
            continue
        dp = as_float(by_method["dp_optimal"], "payload_bits")
        greedy = as_float(by_method["reserve_greedy"], "payload_bits")
        if dp > 0:
            by_tol[float(tol)].append(greedy / dp)
    xs = sorted(by_tol)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx(xs, [mean(by_tol[x]) for x in xs], **line_style(cfg, 0, marker="D", linestyle="-", filled=True))
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("reserve greedy / DP payload")
    ax.set_title("Allocation optimality gap")
    apply_axis_style(ax, cfg, tolerance_ticks=True)
    savefig(fig, fig_dir, "allocator_optimality_gap", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_hmatrix_certificate(cfg: dict, plt, fig_dir: Path) -> None:
    policy = default_budget_policy(cfg)
    rows = [r for r in result_rows(cfg, "hmatrix_certificate.csv") if r["budget_policy"] == policy]
    grouped = group_rows(rows, ("case_id",))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for series_index, (case_id, items) in enumerate(grouped.items()):
        items = sorted(items, key=lambda x: as_float(x, "tol"))
        xs = [as_float(x, "tol") for x in items]
        actual = [as_float(x, "global_actual_error") or float("nan") for x in items]
        cert = [as_float(x, "global_cert") or float("nan") for x in items]
        color = LINE_PALETTE[series_index % len(LINE_PALETTE)]
        measured = line_style(cfg, series_index, color=color, marker=LINE_MARKERS[series_index % len(LINE_MARKERS)],
                              linestyle=LINESTYLES[series_index % len(LINESTYLES)], filled=True)
        bound = line_style(cfg, series_index, color=color, marker=LINE_MARKERS[(series_index + 1) % len(LINE_MARKERS)],
                           linestyle="--", filled=False)
        ax.loglog(xs, actual, label=f"{case_label(case_id[0])}: measured", **measured)
        ax.loglog(xs, cert, label=f"{case_label(case_id[0])}: bound", **bound)
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel(r"$\|A-\widetilde A\|_{\max}$")
    ax.set_title(f"Global H-matrix error bound ({policy})")
    apply_axis_style(ax, cfg, tolerance_ticks=True)
    ax.legend()
    savefig(fig, fig_dir, "hmatrix_global_certificate", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_budget_policy_comparison(cfg: dict, plt, fig_dir: Path) -> None:
    rows = result_rows(cfg, "hmatrix_certificate.csv")
    grouped: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(as_float(row, "tol"), row["budget_policy"])].append(row)
    policies = sorted({row["budget_policy"] for row in rows})
    if not policies:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for policy_index, policy in enumerate(policies):
        xs = sorted({tol for tol, p in grouped if p == policy})
        compression = [mean([as_float(r, "total_compression") for r in grouped[(x, policy)]]) for x in xs]
        ratios = [mean([as_float(r, "cert_over_actual") for r in grouped[(x, policy)]
                        if as_float(r, "global_actual_error") > 0]) for x in xs]
        axes[0].semilogx(xs, compression, label=policy, **line_style(cfg, policy_index))
        axes[1].semilogx(xs, ratios, label=policy, **line_style(cfg, policy_index))
    axes[0].set_xlabel("relative tolerance")
    axes[0].set_ylabel("dense FP64 / stored bits")
    axes[0].set_title("Budget policy storage")
    axes[1].set_xlabel("relative tolerance")
    axes[1].set_ylabel("bound / actual error")
    axes[1].set_title("Budget policy conservatism")
    for ax in axes:
        apply_axis_style(ax, cfg, tolerance_ticks=True)
        ax.legend()
    savefig(fig, fig_dir, "hmatrix_budget_policy_comparison", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_storage(cfg: dict, plt, fig_dir: Path) -> None:
    policy = default_budget_policy(cfg)
    rows = [r for r in result_rows(cfg, "hmatrix_certificate.csv") if r["budget_policy"] == policy]
    grouped = group_rows(rows, ("case_id",))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for series_index, (case_id, items) in enumerate(grouped.items()):
        items = sorted(items, key=lambda x: as_float(x, "tol"))
        xs = [as_float(x, "tol") for x in items]
        payload = [as_float(x, "payload_compression") for x in items]
        total = [as_float(x, "total_compression") for x in items]
        color = LINE_PALETTE[series_index % len(LINE_PALETTE)]
        payload_style = line_style(cfg, series_index, color=color, marker=LINE_MARKERS[series_index % len(LINE_MARKERS)],
                                   linestyle=LINESTYLES[series_index % len(LINESTYLES)], filled=True)
        total_style = line_style(cfg, series_index, color=color, marker=LINE_MARKERS[(series_index + 1) % len(LINE_MARKERS)],
                                 linestyle="--", filled=False)
        ax.semilogx(xs, payload, label=f"{case_label(case_id[0])}: payload", **payload_style)
        ax.semilogx(xs, total, label=f"{case_label(case_id[0])}: + metadata", **total_style)
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("dense FP64 / stored bits")
    ax.set_title(f"H-matrix storage savings ({policy})")
    apply_axis_style(ax, cfg, tolerance_ticks=True)
    ax.legend()
    savefig(fig, fig_dir, "hmatrix_storage", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_error_components(cfg: dict, plt, fig_dir: Path) -> None:
    policy = default_budget_policy(cfg)
    rows = [r for r in result_rows(cfg, "hmatrix_certificate.csv") if r["budget_policy"] == policy]
    by_tol: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_tol[as_float(row, "tol")].append(row)
    xs = sorted(by_tol)
    if not xs:
        return
    eta = [mean([as_float(r, "dominant_eta") for r in by_tol[x]]) for x in xs]
    qcert = [mean([as_float(r, "dominant_q_cert") for r in by_tol[x]]) for x in xs]
    rcert = [mean([as_float(r, "dominant_r_cert") for r in by_tol[x]]) for x in xs]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.stackplot(xs, eta, qcert, rcert, labels=[r"truncation $\eta$", r"quantization $Q$", r"GEMM $R$"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("dominant-block bound components")
    ax.set_title(f"Error decomposition ({policy})")
    apply_axis_style(ax, cfg, tolerance_ticks=True)
    ax.legend(loc="upper left")
    savefig(fig, fig_dir, "hmatrix_error_components", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_bit_histogram(cfg: dict, plt, fig_dir: Path) -> None:
    policy = default_budget_policy(cfg)
    rows = [r for r in result_rows(cfg, "hmatrix_leaves.csv") if r["low_rank"] == "1" and r.get("budget_policy") == policy]
    if not rows:
        return
    bit_set = [str(b) for b in cfg["bit_set"]]
    by_tol: dict[float, dict[str, int]] = defaultdict(lambda: {b: 0 for b in bit_set})
    for row in rows:
        counts = parse_counts(row["bit_counts"])
        tol = as_float(row, "tol")
        for bit in bit_set:
            by_tol[tol][bit] += int(counts.get(bit, 0))
    xs = sorted(by_tol)
    bottoms = [0] * len(xs)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = [f"{x:g}" for x in xs]
    for bit_index, bit in enumerate(bit_set):
        ys = [by_tol[x][bit] for x in xs]
        ax.bar(labels, ys, bottom=bottoms, label=f"{bit}-bit", width=0.72, **bar_kwargs(bit_index, stacked=True))
        bottoms = [b + y for b, y in zip(bottoms, ys)]
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("rank components")
    ax.set_title(f"Precision distribution ({policy})")
    style_bar_axis(ax, cfg)
    ax.legend()
    savefig(fig, fig_dir, "hmatrix_bit_distribution", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def distance_bin(row: dict) -> str:
    geometric = row.get("distance_metric") == "euclidean_bbox"
    sep = as_float(row, "separation" if geometric else "separation_index")
    diam = max(as_float(row, "diameter" if geometric else "diameter_index"), 1e-30)
    ratio = sep / diam
    if ratio < 1.0:
        return "<1"
    if ratio < 2.0:
        return "1-2"
    if ratio < 4.0:
        return "2-4"
    return ">=4"


def plot_precision_vs_distance(cfg: dict, plt, fig_dir: Path) -> None:
    policy = default_budget_policy(cfg)
    tol = min(float(x) for x in cfg["tolerances"])
    rows = [r for r in result_rows(cfg, "hmatrix_leaves.csv") if r["low_rank"] == "1" and r["feasible"] == "1" and r.get("budget_policy") == policy and as_float(r, "tol") == tol]
    bins = ["<1", "1-2", "2-4", ">=4"]
    by_bin: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_bin[distance_bin(row)].append(as_float(row, "avg_bits"))
    if not any(by_bin.values()):
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(bins, [mean(by_bin[b]) for b in bins], width=0.68,
                  color=[BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(bins))],
                  edgecolor="#ffffff", linewidth=0.9, alpha=0.9)
    for i, bar in enumerate(bars):
        bar.set_hatch(BAR_HATCHES[i % len(BAR_HATCHES)])
    ax.set_xlabel("Euclidean box separation / diameter")
    ax.set_ylabel("average assigned bits")
    ax.set_title(f"Precision vs. block separation ({tol:g}, {policy})")
    style_bar_axis(ax, cfg)
    savefig(fig, fig_dir, "hmatrix_precision_vs_distance", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_gpu(cfg: dict, plt, fig_dir: Path) -> None:
    rows = [r for r in read_csv(resolve_path(cfg["results_dir"]) / "gpu_performance.csv") if r.get("median_ms")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    grouped = group_rows(rows, ("kernel", "bits"))
    for series_index, (key, items) in enumerate(grouped.items()):
        kernel, bits = key
        items = sorted(items, key=lambda x: as_float(x, "rank"))
        xs = [as_float(x, "rank") for x in items]
        ys = [as_float(x, "speedup_vs_fp32") for x in items]
        ax.plot(xs, ys, label=f"{kernel}, {bits}-bit", **line_style(cfg, series_index))
    ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("rank")
    ax.set_ylabel("speedup vs FP32")
    ax.set_title("GPU arithmetic realization")
    apply_axis_style(ax, cfg)
    ax.legend()
    savefig(fig, fig_dir, "gpu_performance", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def plot_missing_diagnostics(cfg, plt, fig_dir):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv")
    for (case_id,), items in group_rows(rows, ("case_id",)).items():
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        tol_index = 0
        for row in items:
            if row["method"] != "reserve_greedy" or row["feasible"] != "1":
                continue
            bits = json.loads(row["bits_by_index"])
            ax.step(range(1, len(bits) + 1), bits, where="mid", label=f"tol={row['tol']}",
                    **line_style(cfg, tol_index))
            tol_index += 1
        ax.set_xlabel("singular-value index (descending)")
        ax.set_ylabel("assigned bits")
        ax.set_yticks(cfg["bit_set"])
        apply_axis_style(ax, cfg)
        if ax.lines:
            ax.legend()
        ax.set_title(case_label(case_id))
        savefig(fig, fig_dir, f"allocation_pattern_{case_id}", cfg["plot_style"]["formats"], cfg)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method_index, ((method,), items) in enumerate(group_rows(rows, ("method",)).items()):
        by_tol = group_rows(items, ("tol",))
        xs = sorted(float(k[0]) for k in by_tol)
        ys = [mean([1 - int(r["feasible"]) for r in items if float(r["tol"]) == x]) for x in xs]
        ax.semilogx(xs, ys, label=method, **line_style(cfg, method_index))
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("infeasible fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    apply_axis_style(ax, cfg, tolerance_ticks=True)
    savefig(fig, fig_dir, "allocator_failures", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)

    rows = read_csv(resolve_path(cfg["results_dir"]) / "arithmetic.csv")
    if rows:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for dtype_index, ((dtype,), items) in enumerate(group_rows(rows, ("dtype",)).items()):
            xs = [as_float(r, "representation_and_gemm_cert") for r in items]
            ys = [as_float(r, "actual_rounding") for r in items]
            ax.loglog(xs, ys, label=dtype, **line_style(cfg, dtype_index, linestyle="None"))
        low = min(as_float(r, "representation_and_gemm_cert") for r in rows)
        high = max(as_float(r, "representation_and_gemm_cert") for r in rows)
        ax.loglog([low, high], [low, high], "k--", label="bound equality")
        ax.set_xlabel("representation + GEMM error bound")
        ax.set_ylabel("measured arithmetic error")
        ax.legend()
        apply_axis_style(ax, cfg)
        savefig(fig, fig_dir, "arithmetic_validation", cfg["plot_style"]["formats"], cfg)
        plt.close(fig)


def plot_native(cfg, plt, fig_dir):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "native_performance.csv")
    for row in rows:
        row.setdefault("source_case", "synthetic")
    for (source, m, n), cases in group_rows(rows, ("source_case", "m", "n")).items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        references = ["fp32_cublas", "tf32_cublas", "fp16_cublas", "shared_scale_int8"]
        modes = ["uniform4", "uniform8", "uniform12", "uniform16", "mixed", "adaptive"]
        def series_order(entry):
            kernel, mode = entry[0]
            if kernel in references:
                return (0, references.index(kernel), 0)
            return (1, modes.index(mode) if mode in modes else len(modes),
                    0 if kernel == "columnwise_channels" else 1)
        series = sorted(group_rows(cases, ("kernel", "mode")).items(), key=series_order)
        for series_index, ((kernel, mode), items) in enumerate(series):
            if kernel == "unpack_only":
                continue
            if kernel in ("fp32_cublas", "tf32_cublas", "fp16_cublas") and mode != "uniform8":
                continue
            items = sorted(items, key=lambda r: as_float(r, "rank"))
            names = {"fp32_cublas": "FP32", "tf32_cublas": "TF32", "fp16_cublas": "FP16 inputs",
                     "columnwise_channels": "Predecoded", "packed_columnwise_channels": "Packed",
                     "shared_scale_int8": "Shared-scale INT8"}
            mode_label = mode.replace("uniform", "") + "-bit" if mode.startswith("uniform") else mode
            label = names.get(kernel, kernel)
            if kernel in ("columnwise_channels", "packed_columnwise_channels"):
                label += f": {mode_label}"
            color = {"fp32_cublas": "#333333", "tf32_cublas": "#6f4c9b", "fp16_cublas": "#8c564b",
                     "shared_scale_int8": "#17a398"}.get(kernel)
            if color is None:
                color = {"uniform4": "#0072b2", "uniform8": "#d55e00", "uniform12": "#009e73",
                         "uniform16": "#cc79a7", "mixed": "#b38b00", "adaptive": "#56b4e9"}.get(mode, "#666666")
            marker = {"packed_columnwise_channels": "s", "columnwise_channels": "o"}.get(
                kernel, LINE_MARKERS[series_index % len(LINE_MARKERS)]
            )
            native_line_style = "--" if kernel == "columnwise_channels" else "-"
            # Distinct real leaves at the same rank are observations, not a rank sweep.
            if source != "synthetic":
                native_line_style = "None"
            speed_style = line_style(cfg, series_index, color=color, marker=marker,
                                     linestyle=native_line_style,
                                     filled=kernel != "packed_columnwise_channels")
            latency_style = line_style(cfg, series_index, color=color, marker=marker,
                                       linestyle="None",
                                       filled=kernel != "packed_columnwise_channels")
            axes[0].plot([as_float(r, "rank") for r in items],
                         [as_float(r, "speedup_vs_fp32") for r in items],
                         label=label, **speed_style)
            axes[1].loglog([max(as_float(r, "relative_max_error"), 1e-30) for r in items],
                           [as_float(r, "median_ms") for r in items],
                           label=label, **latency_style)
        axes[0].axhline(1, color="black", linewidth=1)
        axes[0].set_xlabel("rank")
        axes[0].set_xticks(sorted({int(r["rank"]) for r in cases}))
        axes[0].set_ylabel("speedup vs strict FP32")
        axes[1].set_xlabel("relative max-norm error")
        axes[1].set_ylabel("latency (ms)")
        for ax in axes:
            apply_axis_style(ax, cfg)
        axes[0].set_title(f"{case_label(source)} | block {m} x {n}")
        axes[1].set_title("Measured error and execution latency")
        savefig(fig, fig_dir, f"native_{source}_m{m}_n{n}", cfg["plot_style"]["formats"], cfg)
        plt.close(fig)


def plot_suite_scaling(cfg, plt, fig_dir):
    rows = [r for r in result_rows(cfg, "hmatrix_certificate.csv")
            if r["budget_policy"] == default_budget_policy(cfg)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for tol_index, ((tol,), items) in enumerate(group_rows(rows, ("tol",)).items()):
        items = sorted(items, key=lambda r: as_float(r, "n"))
        xs = [as_float(r, "n") for r in items]
        axes[0].semilogx(xs, [as_float(r, "total_compression") for r in items],
                         label=f"tol={tol}", **line_style(cfg, tol_index))
        axes[1].semilogx(xs, [as_float(r, "fallback_dense_leaves") / max(1, as_float(r, "low_rank_leaves")) for r in items],
                         label=f"tol={tol}", **line_style(cfg, tol_index))
    axes[0].set_ylabel("dense FP64 / stored bits")
    axes[1].set_ylabel("admissible-leaf fallback fraction")
    for ax in axes:
        ax.set_xlabel("matrix order N")
        ax.legend()
        apply_axis_style(ax, cfg)
    fig.tight_layout()
    savefig(fig, fig_dir, "size_scaling", cfg["plot_style"]["formats"], cfg)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.get("hmatrix", {}).get("storage_policy") == "diagonal_only":
        raise ValueError("Use plot_offdiagonal.py for strict-policy results, including infeasible configurations")
    fig_dir = ensure_dir(cfg["figures_dir"])
    plt = load_matplotlib(cfg)
    if not (resolve_path(cfg["results_dir"]) / "hmatrix_leaves.csv").exists():
        warnings.warn("hmatrix_leaves.csv is missing: precision-distribution and precision-vs-distance "
                      "figures will not be regenerated; existing files are preserved.", stacklevel=1)
    plot_local_certificate(cfg, plt, fig_dir)
    plot_allocator_payload(cfg, plt, fig_dir)
    plot_allocator_optimality_gap(cfg, plt, fig_dir)
    hrows = result_rows(cfg, "hmatrix_certificate.csv")
    groups = defaultdict(set)
    for row in hrows:
        family = row.get("family", "kernel1d")
        key = "kernel1d" if family == "kernel1d" else f"{family}_{row['geometry']}_{row['kernel']}"
        groups[key].add(row["case_id"])
    for key, ids in groups.items():
        subset = dict(cfg, _case_ids=ids)
        directory = fig_dir if key == "kernel1d" else ensure_dir(fig_dir / "suites" / key)
        for plot in (plot_hmatrix_certificate, plot_budget_policy_comparison, plot_storage,
                     plot_error_components, plot_bit_histogram, plot_precision_vs_distance):
            plot(subset, plt, directory)
        if key != "kernel1d":
            plot_suite_scaling(subset, plt, directory)
    plot_gpu(cfg, plt, fig_dir)
    plot_missing_diagnostics(cfg, plt, fig_dir)
    plot_native(cfg, plt, fig_dir)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
