"""Selected manuscript figures, regenerated only from recorded measurements."""
from __future__ import annotations

import argparse

from plot_results import apply_axis_style, line_style, load_matplotlib, savefig
from qmat_exp.io import ensure_dir, load_config, read_csv, resolve_path


NATIVE_SERIES = (
    ("fp32_cublas", "uniform8", "Strict FP32"),
    ("tf32_cublas", "uniform8", "TF32"),
    ("fp16_cublas", "uniform8", "FP16 inputs"),
    ("shared_scale_int8", "uniform8", "Shared-scale INT8"),
    ("columnwise_channels", "adaptive", "Adaptive, predecoded"),
    ("packed_columnwise_channels", "adaptive", "Adaptive, packed"),
)


def unique_rows(rows, keys):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in indexed:
            raise ValueError(f"Duplicate measurement: {key}")
        indexed[key] = row
    return indexed


def allocator_series(rows, case_id, method, tolerances):
    indexed = unique_rows([r for r in rows if r["case_id"] == case_id], ("method", "tol"))
    values = []
    for tol in tolerances:
        row = indexed[(method, str(tol))]
        baseline = indexed[("uniform16", str(tol))]
        values.append(float(row["payload_bits"]) / float(baseline["payload_bits"])
                      if row["feasible"] == "1" else float("nan"))
    return values


def tolerance_axis(ax, cfg):
    apply_axis_style(ax, cfg)
    ticks = [cfg["tolerances"][i] for i in (0, (len(cfg["tolerances"]) - 1) // 2, -1)]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    # Pad the complete log range, not just the subset with feasible points.
    lo, hi = min(cfg["tolerances"]), max(cfg["tolerances"])
    padding = (hi / lo) ** .05 if hi > lo else 1.1
    ax.set_xlim(lo / padding, hi * padding)
    ax.set_xlabel("relative tolerance")


def plot_allocator(cfg, plt, directory, n=256):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "local_certificate.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    methods = (("reserve_greedy", "Reserve greedy"), ("dp_optimal", "DP optimum"),
               ("uniform16", "Uniform 16-bit"), ("uniform12", "Uniform 12-bit"))
    for ax, spectrum in zip(axes, ("exponential", "algebraic", "flat")):
        for index, (method, label) in enumerate(methods):
            values = allocator_series(rows, f"lowrank_n{n}_r64_{spectrum}", method, cfg["tolerances"])
            ax.semilogx(cfg["tolerances"], values, label=label, **line_style(cfg, index))
        ax.set_title(spectrum.capitalize())
        ax.set_ylim(0.4, 1.05)
        ax.set_yticks([0.5, 0.75, 1.0])
        tolerance_axis(ax, cfg)
    axes[0].set_ylabel(r"payload $M/M_{16}$")
    name = "paper_allocator_comparison" if n == 256 else f"supp_allocator_n{n}"
    savefig(fig, directory, name, cfg["plot_style"]["formats"], cfg)
    return fig


def plot_hmatrix(cfg, plt, directory):
    if cfg.get("hmatrix", {}).get("storage_policy") == "diagonal_only":
        raise ValueError("Use plot_offdiagonal.py; dense-fallback plots describe only the archived policy")
    rows = [r for r in read_csv(resolve_path(cfg["results_dir"]) / "hmatrix_certificate.csv")
            if r["family"] == "bem2d" and r["geometry"] == "circle"
            and r["budget_policy"] == "block_relative"]
    unique_rows(rows, ("n", "tol"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for index, n in enumerate((256, 512, 1024, 2048)):
        items = sorted([r for r in rows if int(r["n"]) == n], key=lambda r: float(r["tol"]))
        if len(items) != len(cfg["tolerances"]):
            raise ValueError(f"Incomplete circle BEM sweep for N={n}")
        xs = [float(r["tol"]) for r in items]
        axes[0].semilogx(xs, [float(r["total_compression"]) for r in items],
                        label=f"N={n}", **line_style(cfg, index))
        axes[1].semilogx(xs, [int(r["fallback_dense_leaves"]) / int(r["low_rank_leaves"])
                             for r in items], label=f"N={n}", **line_style(cfg, index))
    axes[0].set_title("Stored representation")
    axes[0].set_ylabel("dense FP64 / stored bits")
    axes[0].set_ylim(0, 15)
    axes[1].set_title("Dense fallback")
    axes[1].set_ylabel("admissible-leaf fraction")
    axes[1].set_ylim(-0.01, 0.22)
    for ax in axes:
        tolerance_axis(ax, cfg)
    savefig(fig, directory, "paper_hmatrix_storage", cfg["plot_style"]["formats"], cfg)
    return fig


def native_series(rows, n, kernel, mode):
    selected = [r for r in rows if r["source_case"] == "synthetic"
                and int(r["m"]) == n and int(r["n"]) == n
                and r["kernel"] == kernel and r["mode"] == mode]
    unique_rows(selected, ("rank",))
    expected = [r for r in (32, 64, 128, 256) if r <= n]
    selected.sort(key=lambda r: int(r["rank"]))
    if [int(r["rank"]) for r in selected] != expected:
        raise ValueError(f"Incomplete native sweep: {n}, {kernel}, {mode}")
    return selected


def plot_native(cfg, plt, directory):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "native_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, n in zip(axes, (128, 4096)):
        for index, (kernel, mode, label) in enumerate(NATIVE_SERIES):
            items = native_series(rows, n, kernel, mode)
            style = line_style(cfg, index)
            if kernel == "fp32_cublas":
                style.update(color="#333333", markeredgecolor="#333333",
                             markerfacecolor="#333333", linestyle=":")
            if mode == "adaptive":
                style.update(color="#0072B2", markeredgecolor="#0072B2",
                             markerfacecolor="white" if kernel.startswith("packed") else "#0072B2")
            ax.semilogy([int(r["rank"]) for r in items],
                        [float(r["speedup_vs_fp32"]) for r in items], label=label, **style)
        ax.set_title(f"{n} x {n}")
        ax.set_xlabel("rank")
        ax.set_xticks([r for r in (32, 64, 128, 256) if r <= n])
        ax.set_ylim(0.1, 12)
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
        apply_axis_style(ax, cfg)
    axes[0].set_ylabel("speedup vs strict FP32")
    savefig(fig, directory, "paper_native_scale_rank", cfg["plot_style"]["formats"], cfg)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/a100.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    plt = load_matplotlib(cfg)
    directory = ensure_dir(cfg["figures_dir"])
    for plot in (plot_allocator, plot_hmatrix, plot_native):
        plt.close(plot(cfg, plt, directory))
    print(f"Manuscript figures written to {directory}")


if __name__ == "__main__":
    main()
