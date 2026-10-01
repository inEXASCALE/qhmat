"""Stage 3: figures from archived allocator-study results only."""
import argparse

import numpy as np

from plot_results import LINE_PALETTE, apply_axis_style, line_style, load_matplotlib, savefig
from qmat_exp.allocator_study import sha256, spec_hash, summarize
from qmat_exp.io import ensure_dir, load_config, read_csv, read_json, resolve_path


LABELS = {"exp055": r"Exponential, $a=0.55$", "exp075": r"Exponential, $a=0.75$",
          "exp090": r"Exponential, $a=0.90$", "algebraic": "Algebraic", "flat": "Flat"}


def load_results(cfg):
    directory = resolve_path(cfg["results_dir"])
    provenance = read_json(directory / "provenance.json")
    if provenance["spec_hash"] != spec_hash(cfg):
        raise ValueError("Results do not match the experimental configuration")
    for name, digest in provenance["result_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Result hash mismatch: {name}")
    rows = read_csv(directory / "menu_results.csv")
    summary, pairs = summarize(rows, cfg)
    return rows, summary, pairs


def menu_effects(cfg, plt, directory, summary, pairs):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    menus = list(cfg["menus"])
    for i, (a, b) in enumerate(zip(menus, menus[1:])):
        selected = [r for r in pairs if r["before"] == a and r["paired_feasible"]]
        for j, (field, label) in enumerate((("greedy_ratio", "Greedy"), ("dp_ratio", "DP"))):
            values = [r[field] for r in selected]
            if not values:
                raise ValueError(f"No paired feasible cases for {a}->{b}")
            box = ax.boxplot([values], positions=[i + (j - .5) * .32], widths=.25,
                             patch_artist=True, whis=(0, 100), showfliers=False, manage_ticks=False)
            box["boxes"][0].set_facecolor(LINE_PALETTE[j])
            for artist in box["whiskers"] + box["caps"]:
                artist.set_color(LINE_PALETTE[j])
            if i == 0:
                box["boxes"][0].set_label(label)
            box["medians"][0].set_color("black")
    ax.axhline(1, color="0.3", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(menus) - 1), [rf"$B_{i}\to B_{i+1}$" for i in range(len(menus)-1)])
    ax.set_ylabel("payload after / before")
    ax.set_xlabel("precision-menu extension")
    ax.set_title("(a) Paired feasible cases")
    ax = axes[1]
    groups = [g for g in summary["groups"] if g["spectrum"] == "all"]
    counts = [g["feasible"] for g in groups]
    ax.bar(range(len(menus)), counts, color=LINE_PALETTE[2], width=.6)
    for i, count in enumerate(counts):
        ax.text(i, count + .02 * groups[i]["cases"], str(count), ha="center",
                fontsize=cfg["plot_style"]["tick_label_size"])
    ax.set_xticks(range(len(menus)), [rf"$B_{i}$" for i in range(len(menus))])
    ax.set_ylim(0, 1.15 * max(g["cases"] for g in groups))
    ax.set_ylabel("feasible cases")
    ax.set_xlabel("available precision menu")
    ax.set_title("(b) Feasibility")
    for ax in axes:
        apply_axis_style(ax, cfg)
    savefig(fig, directory, "paper_menu_effects", cfg["plot_style"]["formats"], cfg)
    return fig


def spectrum_gaps(cfg, plt, directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8))
    menus = list(cfg["menus"])
    for ax, stat in zip(axes, ("median", "max")):
        for i, spectrum in enumerate(cfg["spectra"]):
            values = [next(g for g in summary["groups"] if g["spectrum"] == spectrum and g["menu"] == menu)
                      ["greedy_over_dp"][stat] for menu in menus]
            ax.plot(range(len(menus)), [np.nan if v is None else v for v in values],
                    label=LABELS.get(spectrum, spectrum), **line_style(cfg, i))
        ax.axhline(1, color="0.3", linestyle=":", linewidth=1)
        ax.set_xticks(range(len(menus)), [rf"$B_{i}$" for i in range(len(menus))])
        ax.set_xlabel("available precision menu")
        ax.set_ylabel(stat + r" $M_{\mathrm{G}}/M_*$")
        apply_axis_style(ax, cfg)
    savefig(fig, directory, "supp_menu_spectra", cfg["plot_style"]["formats"], cfg)
    return fig


def reserve_figure(cfg, plt, directory):
    source = resolve_path(cfg["results_dir"])
    trace = read_csv(source / "reserve_trace.csv")
    boundary = read_csv(source / "feasibility_boundary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9))
    for i, (method, label) in enumerate((("greedy", "Reserve greedy"), ("unrepaired", "Unrepaired"))):
        rows = [r for r in trace if r["method"] == method]
        axes[0].plot([int(r["assigned"]) for r in rows],
                     [float(r["slack"]) / float(r["minimum_initial_error"]) for r in rows],
                     label=label, **line_style(cfg, i))
    axes[0].axhline(0, color="0.3", linestyle=":", linewidth=1)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylim(bottom=-.5)
    axes[0].set_xlabel("assigned components")
    axes[0].set_ylabel(r"$(R-C-c_L S_t)/B_{\min}$")
    axes[0].set_title("(a) Remaining reserve slack")
    for i, (method, label) in enumerate((("greedy", "Reserve greedy"), ("dp", "DP"))):
        rows = [r for r in boundary if r["method"] == method and int(r["scale_exponent"]) == 0]
        axes[1].plot([float(r["budget_ratio"]) - 1 for r in rows],
                     [np.nan if int(r["near_boundary"]) else int(r["feasible"]) for r in rows],
                     label=label, **line_style(cfg, 0 if i == 0 else 2))
    axes[1].axvline(0, color="0.5", linestyle=":", linewidth=1, label="Rounded threshold")
    axes[1].set_xscale("symlog", linthresh=1e-9)
    axes[1].margins(x=.06)
    axes[1].set_xticks([-.02, -1e-8, 0, 1e-8, .02], ["-0.02", r"$-10^{-8}$", "0", r"$10^{-8}$", "0.02"])
    axes[1].set_yticks([0, 1], ["Infeasible", "Feasible"])
    axes[1].set_ylim(-.15, 1.15)
    axes[1].set_xlabel(r"$B/B_{\min}-1$")
    axes[1].set_title("(b) Highest-bit boundary")
    for ax in axes:
        apply_axis_style(ax, cfg)
    savefig(fig, directory, "supp_reserve_boundary", cfg["plot_style"]["formats"], cfg)
    return fig


def scaling_figure(cfg, plt, directory):
    rows = read_csv(resolve_path(cfg["results_dir"]) / "scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8))
    for ax, field in zip(axes, ("relative_change", "relative_error")):
        for i, (mode, label) in enumerate((("columnwise", "Separate column scales"),
                                           ("shared_channel", "Shared channel scales"))):
            values = [[float(r[field]) for r in rows if r["mode"] == mode and int(r["exponent"]) == e]
                      for e in cfg["scaling_exponents"]]
            x = cfg["scaling_exponents"]
            ax.plot(x, [np.median(v) for v in values], label=label, **line_style(cfg, i))
            ax.fill_between(x, [min(v) for v in values], [max(v) for v in values], color=LINE_PALETTE[i], alpha=.15)
        ax.set_xlabel(r"scaling parameter $e$")
        ax.set_xticks(cfg["scaling_exponents"])
        all_values = [float(r[field]) for r in rows]
        if field == "relative_change":
            ax.set_yscale("symlog", linthresh=1e-16)
            ax.set_ylim(0, max(all_values) * 1.5)
            ax.set_yticks([0, 1e-15, 1e-10, 1e-5, 1])
        elif min(all_values) > 0:
            ax.set_yscale("log")
            ax.set_ylim(min(all_values) * .8, max(all_values) * 1.5)
        else:
            ax.set_yscale("symlog", linthresh=1e-16)
            ax.set_ylim(0, max(max(all_values) * 1.5, 1e-16))
        apply_axis_style(ax, cfg)
    axes[0].set_ylabel("relative change in\nquantized product")
    axes[1].set_ylabel("relative reconstruction\nerror")
    savefig(fig, directory, "supp_scaling_control", cfg["plot_style"]["formats"], cfg)
    return fig


def plot(cfg):
    _, summary, pairs = load_results(cfg)
    directory = ensure_dir(cfg["figures_dir"])
    plt = load_matplotlib(cfg)
    for fig in (menu_effects(cfg, plt, directory, summary, pairs),
                spectrum_gaps(cfg, plt, directory, summary),
                reserve_figure(cfg, plt, directory), scaling_figure(cfg, plt, directory)):
        plt.close(fig)
    print(f"Figures written to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/allocator_study.json")
    plot(load_config(parser.parse_args().config))
