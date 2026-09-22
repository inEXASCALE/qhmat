"""Diagonal-only dense storage, strict failure semantics, and native export."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmat_exp.application import build_application, pack_columns, write_application
from qmat_exp.hmatrix import run_hmatrix_case
from qmat_exp.io import load_config
from qmat_exp.matrices import Leaf, build_hmatrix_leaves, kernel_matrix
from qmat_exp.quantization import reconstruct_quantized
from plot_offdiagonal import plot_case, validate_results
from plot_results import load_matplotlib

ROOT = Path(__file__).resolve().parents[2]


def fixture(zero=False):
    n = 16
    rng = np.random.default_rng(18)
    a = rng.normal(size=(n, n))
    a[:8, 8:] = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 8))
    a[8:, :8] = a[:8, 8:].T
    if zero:
        a[:] = 0
    # Both off-diagonal leaves deliberately fail geometric admissibility.
    return a, [Leaf(i, i+8, j, j+8, False) for i in (0, 8) for j in (0, 8)]


def run(a, leaves, **overrides):
    args = dict(case_id="strict", matrix=a, leaves=leaves, tolerances=[.01],
                bit_set=[4, 8, 12, 16, 24, 32], rank_tol=.1, max_rank=0,
                budget_policy="block_relative", hybrid_gamma=.001,
                gemm_reserve_fraction=.02, unit_roundoff=2**-53)
    args.update(overrides)
    return run_hmatrix_case(**args)


class OffdiagonalTests(unittest.TestCase):
    def test_inadmissible_offdiagonal_is_quantized_not_dense(self):
        a, leaves = fixture()
        reps = {}
        summary, rows = run(a, leaves, representations=reps)
        s = summary[0]
        self.assertEqual((s["feasible"], s["dense_leaves"], s["fallback_dense_leaves"]), (1, 2, 0))
        self.assertEqual(s["stored_dense_bits"], 2*8*8*64)
        self.assertEqual(s["total_bits_with_metadata"], sum(s[f"{kind}_bits"] for kind in
                         ("diagonal", "admissible", "offdiagonal_inadmissible")))
        reconstructed = a.copy()
        for row, leaf in zip(rows, leaves):
            self.assertEqual(bool(row["low_rank"]), leaf.off_diagonal)
            self.assertEqual(row["admissible"], 0)
            if leaf.off_diagonal:
                u, sigma, v, eta, bits = reps[(.01, row["leaf_id"])]
                self.assertEqual(len(bits), row["rank"])
                self.assertLessEqual(eta, .25*row["budget"])
                self.assertLessEqual(row["q_cert"] + eta + .02*row["budget"], row["budget"] + 1e-15)
                self.assertLessEqual(row["r_cert"], .02*row["budget"])
                _, block = reconstruct_quantized(u, sigma, v, bits)
                reconstructed[leaf.row0:leaf.row1, leaf.col0:leaf.col1] = block
        self.assertAlmostEqual(np.max(np.abs(a-reconstructed)), s["global_actual_error"])
        self.assertLessEqual(s["global_actual_error"], s["global_cert"] + 1e-12)

    def test_failures_never_become_dense_or_successful_zero_error(self):
        a, leaves = fixture()
        for options, reason in ((dict(max_rank=1), "truncation_budget"),
                                (dict(bit_set=[4], tolerances=[1e-8]), "quantization_budget"),
                                (dict(gemm_reserve_fraction=0), "arithmetic_reserve")):
            with self.subTest(reason=reason):
                summary, rows = run(a, leaves, **options)
                s = summary[0]
                self.assertEqual(s["feasible"], 0)
                self.assertEqual(s["fallback_dense_leaves"], 0)
                self.assertEqual(s["failed_offdiagonal_leaves"], 2)
                for key in ("total_compression", "global_actual_error", "global_cert", "same_partition_fp64_bits"):
                    self.assertEqual(s[key], "")
                for row in rows:
                    if row["low_rank"]:
                        self.assertEqual((row["stored"], row["failure_reason"], row["payload_bits"]), (0, reason, ""))

    def test_diagonal_is_dense_even_if_marked_admissible(self):
        a = np.eye(4)
        summary, rows = run(a, [Leaf(0, 4, 0, 4, True)])
        self.assertEqual(summary[0]["stored_dense_bits"], 16*64)
        self.assertEqual(rows[0]["low_rank"], 0)

    def test_rank_cap_must_also_meet_relative_tail_target(self):
        a, leaves = fixture()
        a[:8, 8:] = a[8:, :8] = np.diag([1., .01, 0, 0, 0, 0, 0, 0])
        summary, rows = run(a, leaves, max_rank=1, rank_tol=1e-4, tolerances=[.1])
        self.assertEqual(summary[0]["feasible"], 0)
        for row in rows:
            if row["low_rank"]:
                self.assertLess(row["eta"], .25 * row["budget"])
                self.assertEqual(row["failure_reason"], "truncation_budget")

    def test_reject_overlapping_diagonal_or_incomplete_partition(self):
        with self.assertRaisesRegex(ValueError, "intersects the diagonal"):
            run(np.eye(4), [Leaf(0, 3, 1, 4, False)])
        with self.assertRaisesRegex(ValueError, "cover"):
            run(np.eye(4), [Leaf(0, 2, 0, 2, False)])

    def test_full_rank_offdiagonal_is_not_assumed_compressible(self):
        a, leaves = fixture()
        a[:8, 8:] = a[8:, :8] = np.eye(8)
        summary, rows = run(a, leaves, tolerances=[1e-8], bit_set=[32])
        self.assertEqual(summary[0]["feasible"], 1)
        for row in rows:
            if row["low_rank"]:
                self.assertEqual(row["rank"], 8)
                self.assertGreater(row["payload_bits"]+row["scale_bits"]+row["map_bits"], 8*8*64)

    def test_svd_cache_is_independent_of_tolerance_and_policy(self):
        points = np.linspace(0, 1, 33)
        a, leaves = kernel_matrix(points, "laplace1d"), build_hmatrix_leaves(points, 8, 1.)
        cache = {}
        for tolerance in (.1, 1e-6):
            for policy in ("block_relative", "global_relative"):
                cached = run(a, leaves, tolerances=[tolerance], budget_policy=policy, factorizations=cache)
                fresh = run(a, leaves, tolerances=[tolerance], budget_policy=policy)
                self.assertEqual(cached, fresh)
                self.assertEqual(cached[0][0]["feasible"], 1)
        self.assertEqual(len(cache), sum(l.off_diagonal for l in leaves))

    def test_application_preserves_strict_decisions_and_32_bit_decode(self):
        a, leaves = fixture()
        cfg = load_config("experiments/configs/offdiagonal_a100.json")
        cfg["hmatrix"]["bit_set"] = [24, 32]
        x = np.ones((len(a), 2), dtype=np.float32)
        data = build_application(a, leaves, np.arange(len(a)), x, cfg, 1e-8)
        self.assertEqual(data["fallback_count"], 0)
        self.assertTrue(all(bool(l["rank"]) == leaf.off_diagonal for l, leaf in zip(data["leaves"], leaves)))
        self.assertIn(32, {int(b) for l in data["leaves"] if l["rank"] for b in l["bits"]})
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if compiler and shutil.which(compiler[0]):
            with tempfile.TemporaryDirectory() as directory:
                binary, path = Path(directory)/"host", Path(directory)/"input.bin"
                subprocess.run(compiler+["-std=c++17", "-O2", str(ROOT/"experiments/cuda/application_host_test.cpp"), "-o", str(binary)], check=True, capture_output=True)
                write_application(path, data)
                subprocess.run([str(binary), str(path)], check=True, capture_output=True)
                cfg["hmatrix"]["bit_set"] = [24]
                data24 = build_application(a, leaves, np.arange(len(a)), x, cfg, 1e-5)
                write_application(path, data24)
                subprocess.run([str(binary), str(path)], check=True, capture_output=True)
        cfg["hmatrix"]["bit_set"] = [4]
        with self.assertRaisesRegex(ValueError, "No diagonal-only representation"):
            build_application(a, leaves, np.arange(len(a)), x, cfg, 1e-8)

    def test_zero_offdiagonal_stays_factorized(self):
        a, leaves = fixture(zero=True)
        cfg = load_config("experiments/configs/offdiagonal_a100.json")
        data = build_application(a, leaves, np.arange(len(a)), np.ones((len(a), 1)), cfg, .01)
        for leaf, output in zip(leaves, data["leaves"]):
            self.assertEqual("dense" in output, leaf.diagonal)
        for reference in data["refs"]:
            np.testing.assert_array_equal(reference, 0)

    def test_config_isolation_and_no_changes_to_local_precision_menu(self):
        cfg = load_config("experiments/configs/offdiagonal_a100.json")
        old = load_config("experiments/configs/a100.json")
        self.assertEqual(cfg["bit_set"], old["bit_set"])
        self.assertEqual(cfg["hmatrix"]["storage_policy"], "diagonal_only")
        for key in ("data_dir", "results_dir", "figures_dir"):
            self.assertNotEqual(cfg[key], old[key])

    def test_plot_gaps_and_external_legend(self):
        a, leaves = fixture()
        summary, _ = run(a, leaves, tolerances=[1e-8, .1], bit_set=[4, 8, 12, 16])
        rows = [{key: str(value) for key, value in row.items()} for row in summary]
        cfg = load_config("experiments/configs/offdiagonal_smoke.json")
        cfg["tolerances"] = [1e-8, .1]
        cfg["plot_style"]["formats"] = ["png"]
        validate_results(cfg, rows)
        with self.assertRaises(ValueError):
            validate_results(cfg, rows + rows)
        plt = load_matplotlib(cfg)
        with tempfile.TemporaryDirectory() as directory:
            fig = plot_case(cfg, plt, Path(directory), rows, "strict", "block_relative")
            self.assertTrue(np.isnan(fig.axes[0].lines[0].get_ydata()[0]))
            self.assertTrue(np.isfinite(fig.axes[0].lines[0].get_ydata()[1]))
            renderer = fig.canvas.get_renderer()
            for legend in fig.legends:
                for ax in fig.axes:
                    self.assertLess(legend.get_window_extent(renderer).y1, ax.get_tightbbox(renderer).y0)
            plt.close(fig)

    @unittest.skipUnless(os.environ.get("QMAT_APPLY_BINARY"), "requires the rebuilt CUDA binary on an NVIDIA node")
    def test_strict_cuda_export_when_available(self):
        a, leaves = fixture()
        cfg = load_config("experiments/configs/offdiagonal_a100.json")
        cfg["hmatrix"]["bit_set"] = [24, 32]
        data = build_application(a, leaves, np.arange(len(a)), np.ones((len(a), 2)), cfg, 1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path, csv = Path(directory)/"input.bin", Path(directory)/"result.csv"
            write_application(path, data)
            subprocess.run([os.environ["QMAT_APPLY_BINARY"], str(path), "2", "1", "2", "2", "3", str(csv)], check=True)


if __name__ == "__main__":
    unittest.main()
