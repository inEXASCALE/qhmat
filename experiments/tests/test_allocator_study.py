import contextlib
import copy
import io
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_allocator_study import prepare
from run_allocator_study import run
from plot_allocator_study import (load_results, menu_effects, reserve_figure,
                                   scaling_figure, spectrum_gaps)
from plot_results import load_matplotlib
from qmat_exp.allocator_study import (reserve_diagnostics, scaling_diagnostics,
                                     synthetic_case, validate_config)
from qmat_exp.io import load_config, read_csv, read_json
from qmat_exp.quantization import naive_low_to_high_allocation, reserve_greedy_allocation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/allocator_study.json"


class AllocatorStudyTests(unittest.TestCase):
    def test_trace_does_not_change_allocations_and_records_effective_reserve(self):
        rng = np.random.default_rng(3)
        for allocate in (reserve_greedy_allocation, naive_low_to_high_allocation):
            for _ in range(20):
                sigma = np.sort(rng.random(9))[::-1]
                trace = []
                args = (sigma, [4, 8, 16], .1, 9, 10)
                ordinary, traced = allocate(*args), allocate(*args, trace=trace)
                self.assertEqual(ordinary.feasible, traced.feasible)
                self.assertEqual(ordinary.cert, traced.cert)
                np.testing.assert_array_equal(ordinary.bits, traced.bits)
                self.assertEqual(trace[0]["assigned"], 0)
                self.assertEqual(trace[-1]["assigned"], 9)
                for state in trace:
                    self.assertAlmostEqual(state["slack"], state["remaining_budget"] - state["minimum_future_error"])
                    if allocate is reserve_greedy_allocation:
                        self.assertGreaterEqual(state["slack"], -1e-15)

    def test_failure_mechanism_and_exact_rational_boundary(self):
        cfg = load_config(CONFIG)
        trace, boundary = reserve_diagnostics(cfg)
        self.assertTrue(all(r["slack"] >= 0 for r in trace if r["method"] == "greedy"))
        self.assertTrue(any(r["slack"] < 0 for r in trace if r["method"] == "unrepaired"))
        self.assertEqual(len(boundary), 42)
        self.assertEqual(sum(r["near_boundary"] for r in boundary), 6)
        for row in boundary:
            if not row["near_boundary"]:
                self.assertEqual(row["feasible"], row["rational_oracle_feasible"])
            if row["budget_ratio"] < 1:
                self.assertFalse(row["feasible"])
        for ratio in cfg["boundary_ratios"]:
            outcomes = {r["feasible"] for r in boundary if r["budget_ratio"] == ratio}
            self.assertEqual(len(outcomes), 1)

    def test_scaling_invariance_and_shared_scale_countercontrol(self):
        cfg = load_config(CONFIG)
        u, sigma, v = synthetic_case(cfg, "exp075", 7)
        rows = scaling_diagnostics(cfg, dict(case_id="test", spectrum="exp075", seed=7), u, sigma, v)
        self.assertLess(max(r["relative_change"] for r in rows if r["mode"] == "columnwise"), 1e-12)
        self.assertGreater(max(r["relative_change"] for r in rows if r["mode"] == "shared_channel"), .1)

    def test_pipeline_is_independent_and_rejects_stale_inputs_results(self):
        cfg = load_config(CONFIG)
        cfg.update(m=12, n=10, rank=5, seeds=[7], spectra={"exp075": cfg["spectra"]["exp075"]},
                   tolerances=[.01, .1])
        with tempfile.TemporaryDirectory() as temp:
            for key in ("data_dir", "results_dir", "figures_dir"):
                cfg[key] = str(Path(temp) / key)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(cfg)
                run(cfg)
            rows, _, _ = load_results(cfg)
            self.assertEqual(len(rows), 24)
            altered = copy.deepcopy(cfg)
            altered["tolerances"] = [.1]
            with self.assertRaises(ValueError):
                run(altered)
            input_path = Path(cfg["data_dir"]) / manifest["cases"][0]["filename"]
            with input_path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                run(cfg)
            result = Path(cfg["results_dir"]) / "menu_results.csv"
            with result.open("a") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                load_results(cfg)

    def test_nested_menus_required(self):
        cfg = load_config(CONFIG)
        cfg["menus"]["B1"] = [4, 6, 8]
        with self.assertRaises(ValueError):
            validate_config(cfg)


class ArchivedAllocatorEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.rows, cls.summary, cls.pairs = load_results(cls.cfg)

    def test_counts_and_paired_claims(self):
        self.assertEqual(len(self.rows), 5400)
        self.assertEqual(sum(int(r["feasible"]) for r in self.rows), 3780)
        for r in self.rows:
            if int(r["feasible"]):
                self.assertEqual(r["validation_ok"], "1")
                quantities = [float(r[k]) for k in
                              ("actual_error", "cert_geom", "cert_sv", "budget", "numerical_allowance")]
                self.assertTrue(all(math.isfinite(x) and x >= 0 for x in quantities))
                error, geom, sv, budget, allowance = quantities
                self.assertLessEqual(error, geom + allowance)
                self.assertLessEqual(geom, sv + allowance)
                self.assertLessEqual(sv, budget + allowance)
        transitions = self.summary["transitions"]
        self.assertEqual([t["paired"] for t in transitions], [180, 180, 450])
        self.assertEqual([t["greedy_worse"] for t in transitions], [153, 177, 80])
        self.assertEqual([t["newly_feasible"] for t in transitions], [0, 270, 0])
        for t in transitions:
            self.assertLessEqual(t["dp_ratio"]["max"], 1)
        archived = read_json(ROOT / self.cfg["results_dir"] / "summary.json")
        self.assertEqual(self.summary["groups"], archived["groups"])
        self.assertEqual(self.summary["transitions"], archived["transitions"])

    def test_archived_diagnostics_match_summary(self):
        directory = ROOT / self.cfg["results_dir"]
        archived = read_json(directory / "summary.json")
        scaling = read_csv(directory / "scaling.csv")
        self.assertEqual(len(scaling), 500)
        for mode in ("columnwise", "shared_channel"):
            rows = [r for r in scaling if r["mode"] == mode]
            for field in ("relative_change", "relative_error"):
                self.assertEqual(archived["scaling"][mode]["max_" + field],
                                 max(float(r[field]) for r in rows))
        boundary = read_csv(directory / "feasibility_boundary.csv")
        self.assertEqual(archived["boundary"], dict(
            rows=len(boundary), near_boundary=sum(int(r["near_boundary"]) for r in boundary),
            off_boundary_mismatches=sum(r["feasible"] != r["rational_oracle_feasible"]
                                       for r in boundary if not int(r["near_boundary"]))))

    def test_manuscript_claims_match_archive(self):
        main = " ".join((ROOT / "main.tex").read_text().split())
        supplement = " ".join((ROOT / "ex_supplement.tex").read_text().split())
        for phrase in ("153 of 180", "24.65", "270", "177"):
            self.assertIn(phrase, main)
        for phrase in ("2.587", "36", "six"):
            self.assertIn(phrase, supplement)
        directory = ROOT / self.cfg["results_dir"]
        summary = read_json(directory / "summary.json")
        measured = summary["scaling"]["columnwise"]["max_relative_change"]
        match = re.search(r"is below \$([\d.]+)\\times10\^\{(-?\d+)\}\$", supplement)
        self.assertIsNotNone(match)
        printed = float(match[1]) * 10 ** int(match[2])
        self.assertLess(measured, printed)
        self.assertLess(printed - measured, 10 ** (int(match[2]) - 2))
        provenance = read_json(directory / "provenance.json")
        self.assertIn(f"Python {provenance['python']}", supplement)
        self.assertIn(f"NumPy {provenance['numpy']}", supplement)

    def test_figures_have_external_legends_and_valid_limits(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["plot_style"]["formats"] = ["png"]
        plt = load_matplotlib(cfg)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            figures = [menu_effects(cfg, plt, directory, self.summary, self.pairs),
                       spectrum_gaps(cfg, plt, directory, self.summary),
                       reserve_figure(cfg, plt, directory), scaling_figure(cfg, plt, directory)]
            for fig in figures:
                renderer = fig.canvas.get_renderer()
                legend = fig.legends[0].get_window_extent(renderer)
                for ax in fig.axes:
                    self.assertLess(legend.y1, ax.get_tightbbox(renderer).y0)
                    self.assertTrue(np.all(np.isfinite(ax.get_ylim())))
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
