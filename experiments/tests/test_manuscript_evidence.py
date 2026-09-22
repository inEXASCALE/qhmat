"""Check manuscript tables and supplemental selections against the saved run."""
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_paper import native_series
from plot_results import load_matplotlib
from plot_supplement import (MODES, local_cases, plot_kernel_storage, plot_local_bounds,
                             plot_precision_maps, plot_real_leaves, real_leaf_groups)
from qmat_exp.io import load_config, read_csv


ROOT = Path(__file__).resolve().parents[2]
H_CASES = (
    ("bem2d_log_circle_n2048", "2D BEM, circle"),
    ("bem2d_log_ellipse_n2048", "2D BEM, ellipse"),
    ("bem3d_inverse_cube_n1536", "3D BEM, cube"),
    ("bem3d_inverse_cuboid_n1536", "3D BEM, cuboid"),
    ("kernel_inverse_distance_2d_uniform_n2048", "2D uniform, $1/r$"),
    ("kernel_log_distance_2d_uniform_n2048", r"2D uniform, $\log r$"),
    ("kernel_inverse_distance_3d_uniform_n2048", "3D uniform, $1/r$"),
    ("kernel_log_distance_3d_uniform_n2048", r"3D uniform, $\log r$"),
)


def compact(text):
    return re.sub(r"\s+", "", text)


class ManuscriptEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = (ROOT / "main.tex").read_text()
        cls.supp = (ROOT / "ex_supplement.tex").read_text()
        cls.h = read_csv(ROOT / "experiments/results_offdiagonal_a100/hmatrix_certificate.csv")
        cls.archived_h = read_csv(ROOT / "experiments/results_a100/hmatrix_certificate.csv")
        cls.native = read_csv(ROOT / "experiments/results_a100/native_performance.csv")
        cls.cfg = load_config(ROOT / "experiments/configs/a100.json")

    def test_all_same_partition_control_identities(self):
        self.assertEqual(len(self.h), 1320)
        self.assertEqual(len(self.archived_h), 792)
        for row in self.h + self.archived_h:
            d, s = int(row["stored_dense_bits"]), int(row["retained_factor_entries"])
            self.assertLessEqual(d, int(row["total_bits_with_metadata"]))
            for bits in (32, 64):
                self.assertEqual(int(row[f"same_partition_fp{bits}_bits"]), d + bits * s)

    def test_main_and_supplement_hierarchical_tables(self):
        for case, label in H_CASES:
            rows = {float(r["tol"]): r for r in self.h
                    if r["case_id"] == case and r["budget_policy"] == "block_relative"}
            r = rows[.01]
            q = int(r["total_bits_with_metadata"])
            values = [float(r["total_compression"]), int(r["same_partition_fp32_bits"]) / q,
                      int(r["same_partition_fp64_bits"]) / q]
            main_row = f"{label} & {r['n']} & " + " & ".join(f"{x:.3f}" for x in values)
            main_row += f" & {float(r['cert_over_actual']):.2f}"
            self.assertIn(compact(main_row), compact(self.main))
            supp_row = f"{label} & {r['n']} & " + " & ".join(
                f"{float(rows[t]['total_compression']):.3f}" for t in (1e-6, .002, .01, .1))
            self.assertIn(compact(supp_row), compact(self.supp))

    def test_native_validation_claims_and_scope(self):
        kernels = {"fp32_cublas", "columnwise_channels", "packed_columnwise_channels",
                   "shared_scale_int8"}
        certified = [r for r in self.native if r["kernel"] in kernels]
        self.assertEqual(len(certified), 1501)
        self.assertTrue(all(r["validation"] == "certificate_pass" for r in certified))
        for source in (self.main, self.supp):
            self.assertIn("1,501 certified reconstruction", source)
            self.assertNotIn(r"\label{tab:exp-gpu}", source)
            self.assertNotIn("Speedup", source)
        self.assertIn("not native mixed-bit integer arithmetic", self.main)
        self.assertIn("not validations of the columnwise mixed-bit", self.supp)

    def test_incremental_storage_ranges(self):
        cases = {case for case, _ in H_CASES}
        rows = [r for r in self.h if r["case_id"] in cases
                and r["budget_policy"] == "block_relative" and float(r["tol"]) == .01]
        self.assertEqual(len(rows), 8)
        for bits in (32, 64):
            gains = [int(r[f"same_partition_fp{bits}_bits"]) /
                     int(r["total_bits_with_metadata"]) for r in rows]
            self.assertIn(f"{min(gains):.3f}$--${max(gains):.3f}", self.main)
        archived = [r for r in self.archived_h if r["case_id"] in cases
                    and r["budget_policy"] == "block_relative" and float(r["tol"]) == .01]
        shares = [100 * int(r["stored_dense_bits"]) /
                  int(r["total_bits_with_metadata"]) for r in archived]
        self.assertIn(f"{min(shares):.1f}--{max(shares):.1f}\\%", self.supp)

    def test_fixed_matrix_policy_table(self):
        rows = [r for r in self.h
                if r["case_id"] == "kernel_inverse_distance_2d_clustered_n2048"
                and float(r["tol"]) == .01]
        self.assertEqual(len(rows), 3)
        for r in rows:
            label = {"block_relative": "Block-relative", "hybrid": "Hybrid",
                     "global_relative": "Global-relative"}[r["budget_policy"]]
            error = float(r["global_actual_error"])
            cert = float(r["global_cert"])
            values = f"{error:.4f} & {cert:.4f}"
            expected = (f"{label} & {float(r['total_compression']):.3f} & "
                        f"{values}")
            self.assertIn(compact(expected), compact(self.supp))

    def test_full_synthetic_validation_coverage(self):
        for n in (128, 512, 4096):
            for mode in ("uniform4", "uniform8", "uniform12", "uniform16", "mixed", "adaptive"):
                rows = native_series(self.native, n, "packed_columnwise_channels", mode)
                self.assertEqual(len(rows), 3 if n == 128 else 4)
                self.assertTrue(all(r["validation"] == "certificate_pass" for r in rows))
                self.assertTrue(all(math.isfinite(float(r["relative_max_error"])) for r in rows))
        self.assertIn("11 synthetic and 68 admissible-block", self.supp)
        self.assertIn("each tested in six modes", self.supp)

    def test_document_split_and_bidirectional_references(self):
        self.assertEqual(len(re.findall(r"\\begin\{figure\}", self.main)), 5)
        self.assertEqual(len(re.findall(r"\\begin\{table\}", self.main)), 1)
        self.assertIn(r"\myexternaldocument{main}", self.supp)
        labels = re.findall(r"\\label\{([^}]+)\}", self.main + self.supp)
        self.assertEqual(len(labels), len(set(labels)))
        # Ignore commented source before checking the active references.
        text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", self.main + self.supp, flags=re.S)
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
        for target in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text):
            self.assertIn(target, labels)
        for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_supplement_uses_matched_samples(self):
        for n in (256, 512):
            groups = local_cases(self.cfg, n)
            self.assertEqual(len(groups), 3)
        groups = real_leaf_groups(self.native)
        self.assertEqual([len(g) for g in groups], [68] * len(MODES))
        with self.assertRaises(ValueError):
            real_leaf_groups([r for r in self.native if r["mode"] != "uniform4"])

    def test_supplement_legend_clearance_and_finite_data(self):
        cfg = dict(self.cfg, plot_style=dict(self.cfg["plot_style"], formats=["pdf"]))
        plt = load_matplotlib(cfg)
        with tempfile.TemporaryDirectory() as directory:
            plots = [lambda c, p, d: plot_local_bounds(c, p, d, 256),
                     lambda c, p, d: plot_local_bounds(c, p, d, 512),
                     plot_precision_maps, plot_kernel_storage, plot_real_leaves]
            for plot in plots:
                fig = plot(cfg, plt, Path(directory))
                renderer = fig.canvas.get_renderer()
                for legend in fig.legends:
                    for ax in fig.axes:
                        self.assertLess(legend.get_window_extent(renderer).y1,
                                        ax.get_tightbbox(renderer).y0)
                for ax in fig.axes:
                    for line in ax.lines:
                        self.assertTrue(all(math.isfinite(float(x)) for x in line.get_ydata()))
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
