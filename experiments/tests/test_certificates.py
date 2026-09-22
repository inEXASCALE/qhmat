import itertools
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmat_exp.hmatrix import run_hmatrix_case
from qmat_exp.matrices import Leaf, build_hmatrix_leaves, synthetic_low_rank_case
from qmat_exp.native_data import prepare_native_data, read_native_data
from qmat_exp.quantization import (arithmetic_diagnostics, c_value, cert_from_bits,
                                   dp_optimal_allocation, naive_low_to_high_allocation,
                                   reserve_greedy_allocation)


class CertificateTests(unittest.TestCase):
    def test_dp_matches_exhaustive_search_and_reserve_is_feasible(self):
        rng = np.random.default_rng(19)
        bit_set = [4, 8, 12]
        for _ in range(10):
            sigma = np.sort(rng.random(5))[::-1]
            budget = rng.uniform(c_value(12), c_value(4)) * sigma.sum()
            candidates = [b for b in itertools.product(bit_set, repeat=5)
                          if cert_from_bits(sigma, np.array(b)) <= budget]
            dp = dp_optimal_allocation(sigma, bit_set, budget, 8, 9)
            greedy = reserve_greedy_allocation(sigma, bit_set, budget, 8, 9)
            self.assertTrue(dp.feasible)
            self.assertEqual(dp.payload_bits, 17 * min(map(sum, candidates)))
            self.assertTrue(greedy.feasible)
            self.assertLessEqual(greedy.cert, budget)
            self.assertGreaterEqual(greedy.payload_bits, dp.payload_bits)
            self.assertTrue(np.all(np.diff(greedy.bits) <= 0))

    def test_naive_counterexample(self):
        sigma = np.ones(2)
        budget = c_value(4)
        naive = naive_low_to_high_allocation(sigma, [4, 8, 16], budget, 2, 2)
        repaired = reserve_greedy_allocation(sigma, [4, 8, 16], budget, 2, 2)
        self.assertFalse(naive.feasible)
        self.assertTrue(repaired.feasible)

    def test_all_dense_fallback_paths_restore_matrix_and_count_storage(self):
        cases = [(np.eye(4), 1, 1e-4, [4, 8, 16], .02, "truncation_budget"),
                 (np.ones((4, 4)), 4, 1e-6, [4], .02, "quantization_budget"),
                 (np.ones((4, 4)), 4, 1., [16], 0., "arithmetic_reserve")]
        for matrix, rank, tol, bits, reserve, reason in cases:
            with self.subTest(reason=reason):
                summary, leaves = run_hmatrix_case("fallback", matrix, [Leaf(0, 4, 0, 4, True)],
                    [tol], bits, 1e-12, rank, "block_relative", .001, reserve, 2**-24,
                    storage_policy="admissible_only")
                self.assertEqual(summary[0]["global_actual_error"], 0)
                self.assertEqual(summary[0]["global_cert"], 0)
                self.assertEqual(summary[0]["fallback_dense_leaves"], 1)
                self.assertEqual(summary[0]["total_bits_with_metadata"], matrix.size * 64)
                self.assertEqual(leaves[0]["fallback_reason"], reason)

    def test_partition_covers_every_entry_once(self):
        for n in [17, 32, 65]:
            cover = np.zeros((n, n), dtype=int)
            for leaf in build_hmatrix_leaves(np.linspace(0, 1, n), 8, 1.):
                cover[leaf.row0:leaf.row1, leaf.col0:leaf.col1] += 1
            np.testing.assert_array_equal(cover, np.ones_like(cover))

    def test_actual_accumulator_precision_and_bound(self):
        case = synthetic_low_rank_case("test", 24, 8, "algebraic", {}, np.random.default_rng(3))
        rows = arithmetic_diagnostics("test", case["u"], case["sigma"], case["v"], [.01], [4, 8, 12, 16])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertLessEqual(row["actual_rounding"], row["representation_and_gemm_cert"] + 1e-15)
        self.assertGreater(rows[0]["actual_rounding"], 100 * rows[1]["actual_rounding"])

    def test_native_binary_layout(self):
        cfg = dict(seed=7, bit_set=[4, 8, 12, 16], low_rank={},
                   gpu=dict(enabled=True, m=17, n=19, ranks=[4], native_shapes=[[17, 19]]))
        with tempfile.TemporaryDirectory() as tmp:
            entries = prepare_native_data(cfg, Path(tmp))
            payload = Path(entries[0]["path"]).read_bytes()
            self.assertEqual(struct.unpack("<8sIII", payload[:20]), (b"QMATGPU1", 17, 19, 4))
            self.assertEqual(len(payload), 20 + 4 * 4 + (17 + 19) * 4 * 4)
            bits = np.frombuffer(payload, dtype="<u4", count=4, offset=20)
            np.testing.assert_array_equal(bits, entries[0]["bits"])
            loaded_bits, p, q = read_native_data(entries[0]["path"])
            np.testing.assert_array_equal(loaded_bits, bits)
            self.assertEqual(p.shape, (17, 4))
            self.assertEqual(q.shape, (19, 4))
            np.testing.assert_array_equal(p.ravel(order="F"),
                np.frombuffer(payload, dtype="<f4", count=17 * 4, offset=36))


if __name__ == "__main__":
    unittest.main()
