import copy
import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_paper import allocator_series, native_series, plot_allocator, plot_hmatrix, plot_native
from plot_results import load_matplotlib
import plot_supplement
from qmat_exp.hmatrix import run_hmatrix_case
from qmat_exp.io import load_config, read_csv
from qmat_exp.matrices import Leaf
from qmat_exp.storage import same_partition_storage
from summarize_storage import summarize


ROOT = Path(__file__).resolve().parents[2]


class StorageControlTests(unittest.TestCase):
    def test_mixed_partition_retains_fp64_dense_and_fallback(self):
        leaves = [dict(row0=0, row1=2, col0=0, col1=3, low_rank=1, feasible=1, rank=1),
                  dict(row0=0, row1=2, col0=3, col1=6, low_rank=0, feasible=1, rank=0),
                  dict(row0=2, row1=4, col0=0, col1=6, low_rank=1, feasible=0, rank=2)]
        expected = dict(stored_dense_bits=18 * 64, retained_factor_entries=5,
                        same_partition_fp32_bits=18 * 64 + 5 * 32,
                        same_partition_fp64_bits=18 * 64 + 5 * 64)
        self.assertEqual(same_partition_storage(leaves), expected)
        self.assertEqual(same_partition_storage([{k: str(v) for k, v in r.items()}
                                                 for r in leaves]), expected)
        self.assertEqual(same_partition_storage([])["same_partition_fp64_bits"], 0)

    def sample_run(self):
        return run_hmatrix_case("test", np.ones((4, 4)), [Leaf(0, 4, 0, 4, True)],
                                [.1, 1e-12], [4, 8, 12, 16], 1e-12, 4,
                                "block_relative", .001, .02, 2**-24,
                                storage_policy="admissible_only")

    def test_export_matches_leaf_recovery_including_fallback(self):
        summaries, leaves = self.sample_run()
        recovered = summarize(leaves, summaries)
        self.assertEqual(len(recovered), 2)
        for expected, row in zip(summaries, recovered):
            for field in ("stored_dense_bits", "retained_factor_entries",
                          "same_partition_fp32_bits", "same_partition_fp64_bits"):
                self.assertEqual(expected[field], row[field])
        self.assertEqual(recovered[0]["retained_factor_entries"], 8)
        self.assertEqual(recovered[1]["same_partition_fp32_bits"], 16 * 64)

    def test_reject_mismatched_missing_and_duplicate_records(self):
        summaries, leaves = self.sample_run()
        altered = copy.deepcopy(summaries)
        altered[0]["payload_bits"] += 1
        for records, aggregate in ((leaves, altered), (leaves[:-1], summaries),
                                   (leaves + [leaves[0]], summaries), (leaves, summaries[:-1])):
            with self.subTest(records=len(records), summaries=len(aggregate)):
                with self.assertRaises(ValueError):
                    summarize(records, aggregate)


class PaperFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = read_csv(ROOT / "experiments/results_a100/local_certificate.csv")
        cls.native = read_csv(ROOT / "experiments/results_a100/native_performance.csv")

    def test_infeasible_points_are_missing_not_zero(self):
        values = allocator_series(self.local, "lowrank_n256_r64_algebraic", "uniform12", [.002, .1])
        self.assertTrue(math.isnan(values[0]))
        self.assertEqual(values[1], .75)

    def test_reported_greedy_counterexamples(self):
        for n, expected in ((256, 1.1354166666666667), (512, 1.1458333333333333)):
            case = f"lowrank_n{n}_r64_algebraic"
            greedy = allocator_series(self.local, case, "reserve_greedy", [.05, .1])
            uniform = allocator_series(self.local, case, "uniform12", [.1])[0]
            self.assertGreater(greedy[1], greedy[0])
            self.assertAlmostEqual(greedy[1] / uniform, expected)

    def test_native_selection_and_missing_ranks(self):
        items = native_series(self.native, 128, "packed_columnwise_channels", "adaptive")
        self.assertEqual([int(r["rank"]) for r in items], [32, 64, 128])
        references = native_series(self.native, 128, "fp32_cublas", "adaptive")
        for row, reference in zip(items, references):
            self.assertAlmostEqual(float(row["speedup_vs_fp32"]),
                                   float(reference["median_ms"]) / float(row["median_ms"]), places=6)
        with self.assertRaises(ValueError):
            native_series(items[:-1], 128, "packed_columnwise_channels", "adaptive")
        with self.assertRaises(ValueError):
            native_series(items + [items[0]], 128, "packed_columnwise_channels", "adaptive")

    def test_allocator_endpoints_fit_inside_common_tolerance_limits(self):
        cfg = load_config(ROOT / "experiments/configs/a100.json")
        cfg["plot_style"]["formats"] = ["png"]
        plt = load_matplotlib(cfg)
        with tempfile.TemporaryDirectory() as directory:
            for n in (256, 512):
                fig = plot_allocator(cfg, plt, Path(directory), n=n)
                limits = fig.axes[0].get_xlim()
                self.assertLess(limits[0], min(cfg["tolerances"]))
                self.assertGreater(limits[1], max(cfg["tolerances"]))
                for ax, spectrum in zip(fig.axes, ("exponential", "algebraic", "flat")):
                    self.assertEqual(ax.get_xlim(), limits)
                    np.testing.assert_allclose(ax.get_xticks(), [.002, .01, .1])
                    for line, method in zip(ax.lines, ("reserve_greedy", "dp_optimal", "uniform16", "uniform12")):
                        expected = allocator_series(self.local, f"lowrank_n{n}_r64_{spectrum}", method, cfg["tolerances"])
                        np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
                        radius = (line.get_markersize() + line.get_markeredgewidth()) * fig.dpi / 144
                        for x, y in zip(line.get_xdata(), line.get_ydata()):
                            if math.isfinite(y):
                                px = ax.transData.transform((x, y))[0]
                                self.assertGreaterEqual(px - ax.bbox.x0, radius)
                                self.assertGreaterEqual(ax.bbox.x1 - px, radius)
                plt.close(fig)

    def test_allocator_only_does_not_generate_other_figures(self):
        cfg = load_config(ROOT / "experiments/configs/a100.json")
        with tempfile.TemporaryDirectory() as directory:
            cfg["figures_dir"] = directory
            with mock.patch.object(sys, "argv", ["plot_supplement.py", "--only", "allocator"]), \
                    mock.patch.object(plot_supplement, "load_config", return_value=cfg), \
                    mock.patch.object(plot_supplement, "plot_local_bounds", side_effect=AssertionError("Unrelated plot")), \
                    contextlib.redirect_stdout(io.StringIO()):
                plot_supplement.main()
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"supp_allocator_n512.pdf", "supp_allocator_n512.png"})

    def test_all_paper_legends_clear_axis_decorations(self):
        cfg = load_config(ROOT / "experiments/configs/a100.json")
        plt = load_matplotlib(cfg)
        cfg["plot_style"]["formats"] = ["pdf"]
        with tempfile.TemporaryDirectory() as directory:
            for plot in (plot_allocator, plot_hmatrix, plot_native):
                fig = plot(cfg, plt, Path(directory))
                renderer = fig.canvas.get_renderer()
                legend = fig.legends[0].get_window_extent(renderer)
                for ax in fig.axes:
                    self.assertLess(legend.y1, ax.get_tightbbox(renderer).y0)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
