"""Matched quantization-granularity baselines and isolated plotting pipeline."""
from pathlib import Path
import statistics
import tempfile
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmat_exp.granularity import choose_candidate, compare_hmatrix_case, decoded_product
from qmat_exp.io import load_config, read_csv, write_json
from qmat_exp.matrices import Leaf
from plot_quantization_granularity import plot, validate_rows
from run_quantization_granularity import run


ROOT = Path(__file__).resolve().parents[2]


def tiny_case():
    matrix = np.eye(4)
    matrix[:2, 2:] = np.array([[0.2, 0.1], [0.0, 0.0]])
    matrix[2:, :2] = matrix[:2, 2:].T
    leaves = [Leaf(i, i + 2, j, j + 2, False) for i in (0, 2) for j in (0, 2)]
    return matrix, leaves


class GranularityTests(unittest.TestCase):
    def test_archived_comparison_matches_supplement(self):
        cfg = load_config(ROOT / "experiments/configs/quantization_granularity.json")
        rows = read_csv(ROOT / cfg["results_dir"] / "granularity_results.csv")
        validate_rows(cfg, rows)
        self.assertEqual(len(rows), 180)
        self.assertTrue(all(row["feasible"] == "1" for row in rows))
        self.assertTrue(all(float(row["global_actual_error"]) <=
                            float(row["global_representation_bound"]) + 1e-12
                            for row in rows))
        by_key = {(row["case_id"], row["tol"], row["method"]): row for row in rows}
        pairs = {(row["case_id"], row["tol"]) for row in rows}
        self.assertEqual(len(pairs), 60)
        channel_ratios, tensor_ratios = [], []
        for case_id, tol in pairs:
            adaptive = int(by_key[case_id, tol, "adaptive"]["total_bits"])
            channel_ratios.append(int(by_key[case_id, tol, "per_channel"]["total_bits"]) / adaptive)
            tensor_ratios.append(int(by_key[case_id, tol, "per_tensor"]["total_bits"]) / adaptive)
        self.assertEqual(sum(ratio > 1 for ratio in channel_ratios), 60)
        self.assertEqual(sum(ratio > 1 for ratio in tensor_ratios), 45)
        supplement = (ROOT / "ex_supplement.tex").read_text()
        for value in (statistics.median(channel_ratios), min(channel_ratios),
                      max(channel_ratios), statistics.median(tensor_ratios)):
            self.assertIn(f"{value:.3f}", supplement)
        self.assertIn("45 of 60 pairs", supplement)
        self.assertIn("other 15 pairs", supplement)
        self.assertIn("supp_granularity_storage_large.pdf", supplement)
        self.assertTrue((ROOT / cfg["figures_dir"] / "supp_granularity_storage_large.pdf").is_file())

    def test_shared_scale_bound_and_metadata_accounting(self):
        u = v = np.eye(3)[:, :2]
        sigma = np.array([4.0, 0.01])
        menu = [4, 8, 12, 16, 24, 32]
        tensor = choose_candidate(u, sigma, v, 0.7, menu, "per_tensor")
        channel = choose_candidate(u, sigma, v, 0.7, menu, "per_channel")
        adaptive = choose_candidate(u, sigma, v, 0.7, menu, "adaptive")
        self.assertTrue(all(c.feasible for c in (tensor, channel, adaptive)))
        self.assertEqual(int(channel.bits[0]), 4)
        self.assertGreater(int(tensor.bits[0]), int(channel.bits[0]))
        self.assertEqual((tensor.scale_bits, channel.scale_bits, adaptive.scale_bits),
                         (128, 256, 256))
        self.assertEqual((tensor.map_bits, channel.map_bits, adaptive.map_bits), (3, 3, 6))
        exact = (u * sigma) @ v.T
        for candidate in (tensor, channel, adaptive):
            actual = float(np.max(np.abs(exact - decoded_product(u, sigma, v, candidate))))
            self.assertLessEqual(actual, candidate.bound + 1e-14)

    def test_matched_partition_rank_and_budget(self):
        cfg = load_config(ROOT / "experiments/configs/quantization_granularity.json")
        cfg["tolerances"] = [0.01, 0.1]
        matrix, leaves = tiny_case()
        rows = compare_hmatrix_case("tiny", "synthetic", matrix, leaves, cfg)
        self.assertEqual(len(rows), 6)
        for tol in cfg["tolerances"]:
            matched = [row for row in rows if row["tol"] == tol]
            self.assertEqual({row["method"] for row in matched},
                             {"per_tensor", "per_channel", "adaptive"})
            self.assertEqual({row["max_rank"] for row in matched}, {1})
            self.assertEqual({row["same_partition_fp64_bits"] for row in matched}, {1024})
            for row in matched:
                self.assertEqual(row["feasible"], 1)
                self.assertEqual(row["total_bits"], sum(row[name] for name in
                                 ("diagonal_bits", "payload_bits", "scale_bits", "map_bits")))
                self.assertEqual(sum(row["bit_counts"].values()), 2)
                self.assertLessEqual(row["global_actual_error"],
                                     row["global_representation_bound"] + 1e-12)

        cfg["hmatrix"]["bit_set"] = [4]
        infeasible = compare_hmatrix_case("tiny", "synthetic", matrix, leaves, cfg)
        self.assertTrue(all(row["feasible"] == 0 and row["total_bits"] == ""
                            and row["global_actual_error"] == "" for row in infeasible))

    def test_zero_block_needs_no_error_budget(self):
        cfg = load_config(ROOT / "experiments/configs/quantization_granularity.json")
        cfg["tolerances"] = [0.01]
        _, leaves = tiny_case()
        rows = compare_hmatrix_case("zero", "synthetic", np.zeros((4, 4)), leaves, cfg)
        self.assertTrue(all(row["feasible"] == 1 and row["max_budget_fraction"] == 0
                            for row in rows))

    def test_isolated_runner_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cfg = load_config(ROOT / "experiments/configs/quantization_granularity.json")
            cfg["data_dir"] = str(base / "data")
            cfg["results_dir"] = str(base / "results")
            cfg["figures_dir"] = str(base / "figures")
            cfg["tolerances"] = [0.01, 0.1]
            cfg["granularity"]["cases"] = [
                dict(case_id="tiny_small", label="Tiny", panel="small"),
                dict(case_id="tiny_large", label="Tiny", panel="large")]
            data = base / "data"
            data.mkdir()
            matrix, leaves = tiny_case()
            manifest_cases = []
            for case in cfg["granularity"]["cases"]:
                case_id = case["case_id"]
                path, leaves_path = data / f"{case_id}.npz", data / f"{case_id}_leaves.json"
                np.savez_compressed(path, matrix=matrix)
                write_json(leaves_path, [leaf.to_dict() for leaf in leaves])
                manifest_cases.append(dict(case_id=case_id, path=str(path),
                                           leaves_path=str(leaves_path), family="synthetic"))
            write_json(data / "manifest.json", dict(
                config={key: cfg[key] for key in ("seed", "hmatrix", "geometric_suites")},
                hmatrix=manifest_cases))
            rows = run(cfg)
            self.assertEqual(len(rows), 12)
            plot(cfg)
            self.assertEqual(len(list((base / "figures").glob("*.pdf"))), 4)
            with self.assertRaises(ValueError):
                validate_rows(cfg, [dict(row) for row in []])
            with (base / "results" / "granularity_results.csv").open("a") as output:
                output.write("\n")
            with self.assertRaises(ValueError):
                plot(cfg)


if __name__ == "__main__":
    unittest.main()
