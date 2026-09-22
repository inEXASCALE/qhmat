import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmat_exp.geometric_matrices import (euclidean_kernel, polygon_panels,
    log_segment_integrals, box_panels, inverse_rectangle_integrals, geometric_cases)
from qmat_exp.spatial import spatial_partition
from qmat_exp.hmatrix import run_hmatrix_case
from qmat_exp.native_data import export_geometric_native, read_native_data


class GeometricMatrixTests(unittest.TestCase):
    def test_exact_point_kernel_and_omitted_self(self):
        points = np.array([[0., 0., 0.], [3., 4., 0.], [0., 0., 2.]])
        inverse, log = euclidean_kernel(points, "inverse_distance"), euclidean_kernel(points, "log_distance")
        self.assertEqual(inverse[0, 1], .2)
        self.assertEqual(log[0, 1], np.log(5))
        np.testing.assert_array_equal(np.diag(log), 0)
        np.testing.assert_array_equal(inverse, inverse.T)
        with self.assertRaises(ValueError):
            euclidean_kernel(np.zeros((2, 3)), "inverse_distance")

    def test_log_panel_integrals_against_quadrature_and_self_formula(self):
        panels = polygon_panels(16, "ellipse")
        targets = np.array([[1., .2], [-.2, .8]])
        nodes, weights = leggauss(64)
        midpoint = panels.mean(axis=1)
        half = (panels[:, 1] - panels[:, 0]) / 2
        quadrature_points = midpoint[:, None] + nodes[None, :, None] * half[:, None]
        sampled = np.log(np.linalg.norm(targets[:, None, None] - quadrature_points[None], axis=3))
        expected = np.einsum("ijk,k->ij", sampled, weights) * np.linalg.norm(half, axis=1)
        np.testing.assert_allclose(log_segment_integrals(targets, panels), expected, atol=2e-14)
        lengths = 2 * np.linalg.norm(half, axis=1)
        np.testing.assert_allclose(np.diag(log_segment_integrals(midpoint, panels)),
                                   lengths * (np.log(lengths / 2) - 1), atol=2e-14)

    def test_rectangle_integral_against_quadrature_self_and_far_limit(self):
        centers, widths, normals = box_panels(2, "cuboid")
        targets = np.array([[1., 1., 1.], [-1., .8, .7]])
        calculated = inverse_rectangle_integrals(targets, centers, widths, normals)
        nodes, weights = leggauss(40)
        for j in range(len(centers)):
            axes = [a for a in range(3) if a != normals[j]]
            uv = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
            samples = np.tile(centers[j], (len(uv), 1))
            samples[:, axes] += uv * widths[j, axes]
            expected = ((1 / np.linalg.norm(targets[:, None] - samples[None], axis=2)) @
                        np.outer(weights, weights).ravel()) * np.prod(widths[j, axes])
            np.testing.assert_allclose(calculated[:, j], expected, rtol=2e-12, atol=2e-14)
        center = np.zeros((1, 3))
        halfwidth = np.array([[1., 1., 0.]])
        diagonal = inverse_rectangle_integrals(center, center, halfwidth, np.array([2]))[0, 0]
        self.assertAlmostEqual(diagonal, 8 * np.arcsinh(1), places=13)
        far = inverse_rectangle_integrals(np.array([[0., 0., 100.]]), center, halfwidth, np.array([2]))[0, 0]
        self.assertLess(abs(far / .04 - 1), 4e-5)

    def test_spatial_partition_and_panel_support_admissibility(self):
        points, widths, _ = box_panels(4, "cube")
        order, leaves = spatial_partition(points, 4, 1., points-widths, points+widths)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(points)))
        cover = np.zeros((len(points), len(points)), dtype=int)
        self.assertTrue(any(leaf.low_rank for leaf in leaves))
        for leaf in leaves:
            cover[leaf.row0:leaf.row1, leaf.col0:leaf.col1] += 1
            if leaf.low_rank:
                self.assertGreater(leaf.separation, 0)
                self.assertGreaterEqual(leaf.separation, leaf.diameter)
                self.assertEqual(leaf.distance_metric, "euclidean_bbox")
        np.testing.assert_array_equal(cover, 1)

    def test_bem_compression_certificate_and_constant_density_circle_limit(self):
        h_cfg = dict(leaf_size=8, admissibility_eta=1.)
        cfg = dict(bem2d=dict(geometries=["circle"], sizes=[32, 64]),
                   bem3d=dict(geometries=["cube"], subdivisions=[2]))
        circle_errors = []
        for case in geometric_cases(cfg, h_cfg, 7):
            self.assertTrue(np.isfinite(case["matrix"]).all())
            if case["family"] == "bem2d":
                # Uniform density on a circle of radius R has trace -R log(R).
                circle_errors.append(np.max(np.abs(case["matrix"].sum(axis=1) + .5*np.log(.5))))
            summary, _ = run_hmatrix_case(case["case_id"], case["matrix"], case["leaves"],
                [.02], [4, 8, 12, 16], 1e-6, 24, "block_relative", .001, .02, 2**-24,
                storage_policy="admissible_only")
            self.assertLessEqual(summary[0]["global_actual_error"], summary[0]["global_cert"] + 1e-12)
        self.assertLess(circle_errors[1], circle_errors[0] / 2)

    def test_exported_native_factors_retain_source_leaf(self):
        h_cfg = dict(leaf_size=8, admissibility_eta=1., rank_tol=1e-6, max_rank=24)
        suites = dict(bem2d=dict(geometries=["ellipse"], sizes=[64]), native_leaves_per_case=2)
        case = next(geometric_cases(suites, h_cfg, 7))
        cfg = dict(hmatrix=h_cfg, geometric_suites=suites, bit_set=[4, 8, 12, 16], gpu=dict(enabled=True))
        with tempfile.TemporaryDirectory() as directory:
            entries = export_geometric_native(case, cfg, Path(directory))
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertEqual(entry["source_case"], case["case_id"])
                _, p, q = read_native_data(entry["path"])
                leaf = case["leaves"][entry["leaf_id"]]
                block = case["matrix"][leaf.row0:leaf.row1, leaf.col0:leaf.col1]
                error = np.max(np.abs(block - p.astype(float) @ q.astype(float).T))
                self.assertLessEqual(error, entry["eta"] + 1e-6 * np.max(np.abs(block)))


if __name__ == "__main__":
    unittest.main()
