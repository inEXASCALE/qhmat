"""Application export/reference checks, including optional real CUDA execution."""
import copy
import os
from pathlib import Path
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_application import plot_case
from plot_results import load_matplotlib
from qmat_exp.application import build_application, pack_columns, validate_partition, write_application
from qmat_exp.io import load_config, read_csv
from qmat_exp.matrices import Leaf
from run_application import METHODS, validate_rows

ROOT = Path(__file__).resolve().parents[2]


def fixture(tol=.1, zero=False, bit_set=None):
    rng = np.random.default_rng(11)
    n = 13
    matrix = rng.normal(size=(n, n))
    leaves = [Leaf(0, 5, 0, 7, False), Leaf(0, 5, 7, n, True),
              Leaf(5, n, 0, 7, True), Leaf(5, n, 7, n, False)]
    for leaf in leaves:
        if leaf.low_rank:
            m, k = leaf.shape
            matrix[leaf.row0:leaf.row1, leaf.col0:leaf.col1] = rng.normal(size=(m, 2)) @ rng.normal(size=(2, k))
    if zero:
        matrix[:] = 0
    cfg = load_config("experiments/configs/a100.json")
    if bit_set is not None:
        cfg["bit_set"] = bit_set
    permutation = rng.permutation(n)
    x = rng.normal(size=(n, 32)).astype(np.float32)
    return build_application(matrix, leaves, permutation, x, cfg, tol)


def decode(packed, rows, bits):
    result = np.empty((rows, len(bits)), dtype=np.float32)
    for j, bits_j in enumerate(bits):
        b = int(bits_j)
        for i in range(rows):
            word, shift = divmod(i*b, 32)
            word += int(packed["offsets"][j])
            raw = int(packed["words"][word]) >> shift
            if shift+b > 32:
                raw |= int(packed["words"][word+1]) << (32-shift)
            raw &= (1 << b)-1
            value = raw-(1 << b) if raw & (1 << (b-1)) else raw
            result[i, j] = value*packed["scales"][j]
    return result


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.host = cls.directory / "host"
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if compiler and shutil.which(compiler[0]):
            result = subprocess.run(compiler + ["-std=c++17", "-O2", str(ROOT / "experiments/cuda/application_host_test.cpp"),
                            "-o", str(cls.host)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pack_every_width_cross_word_and_zero_column(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(37, 5))
        values[:, 2] = 0
        bits = [4, 8, 12, 16, 12]
        packed = pack_columns(values, bits)
        np.testing.assert_array_equal(decode(packed, 37, bits), packed["decoded"])
        with self.assertRaises(ValueError):
            pack_columns(values, [4, 8, 12, 16, 20])
        with self.assertRaises(ValueError):
            pack_columns(values, [4])

    def test_two_stage_equals_reconstructed_reference(self):
        data = fixture()
        self.assertEqual(sum(l["rank"] > 0 for l in data["leaves"]), 2)
        self.assertEqual(data["fallback_count"], 0)
        perm = data["permutation"]
        for quantized in (False, True):
            matrix = np.zeros((data["n"], data["n"]))
            for l in data["leaves"]:
                if l["rank"]:
                    p, q = (l["pp"]["decoded"], l["pq"]["decoded"]) if quantized else (l["p"], l["q"])
                    block = p.astype(float) @ q.astype(float).T
                else:
                    block = l["dense"]
                matrix[l["row0"]:l["row1"], l["col0"]:l["col1"]] = block
            expected = np.empty(data["x"].shape)
            expected[perm] = matrix @ data["x"][perm].astype(float)
            np.testing.assert_allclose(expected, data["refs"][3 if quantized else 2], rtol=1e-12, atol=1e-12)

    def test_fallback_and_zero(self):
        data = fixture(tol=1e-12)
        self.assertEqual(data["fallback_count"], 2)
        self.assertTrue(all(l["rank"] == 0 for l in data["leaves"]))
        np.testing.assert_allclose(data["refs"][1], data["refs"][2], atol=1e-12)
        zero = fixture(zero=True)
        for ref in zero["refs"]:
            np.testing.assert_array_equal(ref, 0)

    def test_fp32_application_arithmetic_envelope(self):
        data = fixture()
        perm = data["permutation"]
        xp = data["x"][perm]
        for quantized in (False, True):
            y = np.zeros_like(xp)
            for l in data["leaves"]:
                xj = xp[l["col0"]:l["col1"]]
                if l["rank"]:
                    p, q = (l["pp"]["decoded"], l["pq"]["decoded"]) if quantized else (l["p"], l["q"])
                    block_y = p @ (q.T @ xj)
                else:
                    block_y = l["dense"] @ xj
                y[l["row0"]:l["row1"]] += block_y
            original_order = np.empty_like(y)
            original_order[perm] = y
            method = 2 if quantized else 1
            error = np.max(np.abs(original_order-data["refs"][method+1]), axis=0)
            self.assertTrue(np.all(error <= data["allowances"][method]))

    def test_partition_rejects_gaps_and_overlap(self):
        with self.assertRaises(ValueError):
            validate_partition(3, [Leaf(0, 3, 0, 2, False)])
        with self.assertRaises(ValueError):
            validate_partition(3, [Leaf(0, 3, 0, 3, False), Leaf(0, 1, 0, 1, False)])

    def test_host_binary_roundtrip_and_malformed_input(self):
        if not self.host.exists():
            self.skipTest("C++ compiler unavailable")
        path = self.directory / "roundtrip.bin"
        for data in (fixture(), fixture(tol=1e-12), fixture(zero=True), fixture(bit_set=[12]), fixture(bit_set=[16])):
            write_application(path, data)
            subprocess.run([str(self.host), str(path)], check=True, capture_output=True)
        write_application(path, fixture())
        payload = path.read_bytes()
        malformed = [payload[:-1], payload+b"x", b"INVALID!"+payload[8:],
                     payload[:20]+struct.pack("<I", 1000)+payload[24:]]
        for i, contents in enumerate(malformed):
            bad = self.directory / f"bad{i}.bin"
            bad.write_bytes(contents)
            result = subprocess.run([str(self.host), str(bad)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_cuda_when_available(self):
        binary = os.environ.get("QMAT_APPLY_BINARY")
        if not binary:
            self.skipTest("set QMAT_APPLY_BINARY on an NVIDIA GPU node")
        path, csv = self.directory / "cuda.bin", self.directory / "cuda.csv"
        for data in (fixture(), fixture(tol=1e-12), fixture(zero=True), fixture(bit_set=[12]), fixture(bit_set=[16])):
            write_application(path, data)
            for rhs in (1, 8, 32):
                subprocess.run([binary, str(path), str(rhs), "1", "2", "2", "3", str(csv)], check=True)
                validate_rows(read_csv(csv), dict(trials=2), data["n"], rhs)


def plot_fixture():
    cfg = load_config("experiments/configs/application_a100.json")
    cfg = dict(cfg, rhs_counts=[1, 8, 32], trials=2,
               plot_style=dict(load_config(cfg["source_config"])["plot_style"], formats=["png"]))
    rows = []
    for p in cfg["rhs_counts"]:
        for trial in range(2):
            for index, method in enumerate(METHODS):
                ms = float(index+1)*(1+.01*trial)
                rows.append(dict(method=method, trial=str(trial), order_index=str(index), n="13", rhs=str(p),
                    median_ms=ms, min_ms=ms*.99, max_ms=ms*1.01,
                    speedup_vs_dense=1/(index+1), speedup_vs_h_fp32=2/(index+1),
                    max_abs_error=0, relative_max_error=0, arithmetic_error=0, arithmetic_allowance=1e-6,
                    representation_error=0, representation_bound=1e-6, validation="arithmetic_pass"))
    return cfg, rows


class ApplicationResultTests(unittest.TestCase):
    def test_reject_unpaired_or_incomplete_results(self):
        cfg, rows = plot_fixture()
        rows = [r for r in rows if r["rhs"] == "1"]
        validate_rows(rows, cfg, 13, 1)
        with self.assertRaises(ValueError):
            validate_rows(rows[:-1], cfg, 13, 1)
        bad = copy.deepcopy(rows)
        bad[0]["speedup_vs_dense"] = 2
        with self.assertRaises(ValueError):
            validate_rows(bad, cfg, 13, 1)
        bad = copy.deepcopy(rows)
        bad[0]["relative_max_error"] = float("nan")
        with self.assertRaises(ValueError):
            validate_rows(bad, cfg, 13, 1)

    def test_plot_zero_error_and_external_legend(self):
        cfg, rows = plot_fixture()
        plt = load_matplotlib(cfg)
        with tempfile.TemporaryDirectory() as d:
            fig = plot_case(cfg, plt, rows, Path(d), "TEST_FIXTURE_NOT_MEASUREMENTS")
            renderer = fig.canvas.get_renderer()
            for legend in fig.legends:
                for ax in fig.axes:
                    self.assertLess(legend.get_window_extent(renderer).y1, ax.get_tightbbox(renderer).y0)
            self.assertTrue(all(np.isfinite(line.get_ydata()).all() for ax in fig.axes for line in ax.lines))
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
