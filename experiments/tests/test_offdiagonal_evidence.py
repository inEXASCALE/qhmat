"""Check published diagonal-only summaries, manuscript tables, and figure inputs."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
sys.path.insert(0, str(EXP))
from qmat_exp.io import load_config

CASES = [
    ("2D BEM, circle", "bem2d_log_circle_n2048"),
    ("2D BEM, ellipse", "bem2d_log_ellipse_n2048"),
    ("3D BEM, cube", "bem3d_inverse_cube_n1536"),
    ("3D BEM, cuboid", "bem3d_inverse_cuboid_n1536"),
    ("2D uniform, $1/r$", "kernel_inverse_distance_2d_uniform_n2048"),
    ("2D uniform, $\\log r$", "kernel_log_distance_2d_uniform_n2048"),
    ("3D uniform, $1/r$", "kernel_inverse_distance_3d_uniform_n2048"),
    ("3D uniform, $\\log r$", "kernel_log_distance_3d_uniform_n2048"),
]


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def table_rows(source, label):
    # Scope every check to its labelled table, not matching numbers elsewhere.
    table = source.split("\\label{" + label + "}", 1)[1].split("\\end{table}", 1)[0]
    return [[cell.strip() for cell in line.rstrip().removesuffix("\\\\").split("&")]
            for line in table.splitlines() if " & " in line and line.rstrip().endswith("\\\\")]


class OffdiagonalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = csv_rows(EXP / "results_offdiagonal_a100/hmatrix_certificate.csv")
        cls.by_key = {(r["case_id"], r["budget_policy"], float(r["tol"])): r for r in cls.rows}
        cls.ranks = csv_rows(EXP / "figures_offdiagonal_a100/rank_diagnostics.csv")
        cls.rank_keys = {(r["case_id"], r["budget_policy"], float(r["tol"]), r["block_kind"]): r
                         for r in cls.ranks}
        cls.validation = json.loads((EXP / "results_offdiagonal_a100/numerical_validation.json").read_text())
        cls.main = (ROOT / "main.tex").read_text()
        cls.supp = (ROOT / "ex_supplement.tex").read_text()

    def test_complete_grid_provenance_and_feasibility(self):
        cfg = load_config(EXP / "configs/offdiagonal_a100.json")
        self.assertEqual(cfg, self.validation["config"])
        self.assertEqual(len(self.rows), 1320)
        cases = {r["case_id"] for r in self.rows}
        self.assertEqual(len(cases), 44)
        self.assertEqual(set(self.by_key), set(itertools.product(cases, cfg["hmatrix"]["budget_policies"], cfg["tolerances"])))
        self.assertEqual(self.validation["violations"], 0)
        self.assertEqual(self.validation["infeasible_hmatrix_configurations"], 0)
        for name, digest in self.validation["source_sha256"].items():
            self.assertEqual(hashlib.sha256((EXP / name).read_bytes()).hexdigest(), digest, name)
        for r in self.rows:
            self.assertEqual(r["storage_policy"], "diagonal_only")
            self.assertEqual(int(r["feasible"]), 1)
            self.assertEqual(int(r["fallback_dense_leaves"]), 0)
            self.assertEqual(int(r["failed_offdiagonal_leaves"]), 0)
            self.assertGreater(float(r["global_actual_error"]), 0)
            self.assertLessEqual(float(r["global_actual_error"]), float(r["global_cert"]) + 1e-12)
            self.assertEqual(int(r["total_bits_with_metadata"]), sum(int(r[k+"_bits"]) for k in
                             ("diagonal", "admissible", "offdiagonal_inadmissible")))
            d, s = int(r["diagonal_bits"]), int(r["retained_factor_entries"])
            self.assertEqual(d, int(r["stored_dense_bits"]))
            for bits in (32, 64):
                self.assertEqual(d + bits*s, int(r[f"same_partition_fp{bits}_bits"]))

    def test_main_storage_table(self):
        actual = table_rows(self.main, "tab:exp-hierarchical")
        for label, case in CASES:
            r = self.by_key[case, "block_relative", .01]
            m = int(r["total_bits_with_metadata"])
            expected = [label, r["n"], f'{float(r["total_compression"]):.3f}',
                        f'{int(r["same_partition_fp32_bits"])/m:.3f}',
                        f'{int(r["same_partition_fp64_bits"])/m:.3f}',
                        f'{float(r["cert_over_actual"]):.2f}']
            self.assertIn(expected, actual)

    def test_supplement_tolerance_and_rank_tables(self):
        tolerances = table_rows(self.supp, "tab:supp-h-tolerances")
        ranks = table_rows(self.supp, "tab:supp-nearfield-ranks")
        self.assertEqual(len(self.ranks), len(self.rank_keys))
        for label, case in CASES:
            expected = [label, self.by_key[case, "block_relative", .01]["n"]]
            expected += [f'{float(self.by_key[case, "block_relative", t]["total_compression"]):.3f}'
                         for t in (1e-6, .002, .01, .1)]
            self.assertIn(expected, tolerances)
            expected = [label]
            for t in (1e-6, .01):
                r = self.rank_keys[case, "block_relative", t, "offdiagonal_inadmissible"]
                expected += [f'{float(r[k]):.3f}' for k in ("median_rank_fraction", "max_rank_fraction")]
            self.assertIn(expected, ranks)
        for r in self.ranks:
            summary = self.by_key[r["case_id"], r["budget_policy"], float(r["tol"])]
            self.assertEqual(int(r["leaves"]), int(summary[r["block_kind"] + "_leaves"]))
            self.assertEqual(int(r["failed"]), 0)

    def test_policy_table_and_dominant_certificate_claims(self):
        actual = table_rows(self.supp, "tab:supp-policies")
        for name, policy in (("Block-relative", "block_relative"), ("Hybrid", "hybrid"),
                             ("Global-relative", "global_relative")):
            r = self.by_key["kernel_inverse_distance_2d_clustered_n2048", policy, .01]
            self.assertIn([name, f'{float(r["total_compression"]):.3f}',
                           f'{float(r["global_actual_error"]):.4f}', f'{float(r["global_cert"]):.4f}'], actual)
        block = [r for r in self.rows if r["budget_policy"] == "block_relative"]
        ratios = [float(r["cert_over_actual"]) for r in block]
        self.assertEqual((f"{min(ratios):.2f}", f"{max(ratios):.2f}"), ("4.39", "36.75"))
        self.assertGreaterEqual(min(float(r["dominant_q_cert"])/float(r["global_cert"]) for r in block), .86)
        for case, policy in {(r["case_id"], r["budget_policy"]) for r in self.rows}:
            items = sorted((r for r in self.rows if (r["case_id"], r["budget_policy"]) == (case, policy)),
                           key=lambda r: float(r["tol"]))
            self.assertTrue(all(float(a["total_compression"]) <= float(b["total_compression"])
                                for a, b in zip(items, items[1:])))

    def test_figure_inputs_and_stale_protocol_text(self):
        for source in (self.main, self.supp):
            self.assertNotIn("revised runs are pending", source)
            for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
                self.assertTrue((ROOT / path).is_file(), path)
        for case in ("bem2d_log_circle_n2048", "kernel_log_distance_3d_uniform_n2048"):
            self.assertIn(f"figures_offdiagonal_a100/{case}_block_relative.pdf", self.main)
        self.assertNotIn("paper_hmatrix_storage.pdf", self.main)


if __name__ == "__main__":
    unittest.main()
