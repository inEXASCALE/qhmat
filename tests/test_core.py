import numpy as np
import pytest

import qhmat


def test_allocator_feasibility_and_storage():
    sigma = np.array([4.0, 1.0, 0.2])
    allocation = qhmat.allocate_bits(sigma, [16, 4, 8], 0.08, 20, 16)
    assert allocation.feasible
    assert allocation.bits == sorted(allocation.bits, reverse=True)
    assert allocation.bound <= 0.08
    assert allocation.payload_bits == 36 * sum(allocation.bits)
    assert not qhmat.allocate_bits(sigma, [4, 8], 1e-8, 20, 16).feasible


@pytest.mark.parametrize("bits", [2, 4, 8, 12, 16, 24, 32])
def test_coefficient_decreases(bits):
    assert qhmat.coefficient(bits) > 0
    if bits > 2:
        assert qhmat.coefficient(bits) < qhmat.coefficient(2)


def test_allocator_rejects_invalid_input():
    with pytest.raises(ValueError):
        qhmat.allocate_bits([1.0, 2.0], [4, 8], 0.1, 3, 3)
    with pytest.raises(ValueError):
        qhmat.allocate_bits([2.0, 1.0], [4, 4], 0.1, 3, 3)
    with pytest.raises(ValueError):
        qhmat.allocate_bits([2.0], [1, 8], 0.1, 3, 3)


def test_spatial_partition_covers_entries():
    points = np.arange(8, dtype=float)[:, None]
    partition = qhmat.spatial_partition(points, points, points, 2, 1.0)
    assert sorted(partition.permutation) == list(range(8))
    coverage = np.zeros((8, 8), dtype=int)
    for block in partition.blocks:
        coverage[block.row0:block.row1, block.col0:block.col1] += 1
    np.testing.assert_array_equal(coverage, np.ones((8, 8), dtype=int))
    assert any(block.admissible for block in partition.blocks)


def test_native_block_apply_and_metadata():
    matrix = qhmat.NativeHMatrix(2, [0, 1])
    matrix.add_dense(qhmat.Rectangle(0, 1, 0, 1), np.array([[2.0]]))
    matrix.add_dense(qhmat.Rectangle(1, 2, 1, 2), np.array([[3.0]]))
    for row, col in [(0, 1), (1, 0)]:
        matrix.add_low_rank(
            qhmat.Rectangle(row, row + 1, col, col + 1),
            np.ones((1, 1)), np.array([1.0]), np.ones((1, 1)),
            [8], 3, 0.0,
        )
    matrix.finalize()
    expected = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(matrix.reconstruct(), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrix.matmul(np.array([1.0, 2.0])), expected @ [1.0, 2.0])
    np.testing.assert_allclose(matrix.matmul(np.eye(2)), expected)
    assert matrix.logical_bits == 2 * 64 + 2 * (2 * 8 + 128 + 2)
    assert matrix.reconstruction_bound >= 0


def test_native_rejects_partition_gap():
    matrix = qhmat.NativeHMatrix(2, [0, 1])
    matrix.add_dense(qhmat.Rectangle(0, 1, 0, 1), np.ones((1, 1)))
    with pytest.raises(ValueError, match="partition"):
        matrix.finalize()


def test_high_level_kernel_matrix():
    points = np.linspace(0, 1, 12)[:, None]
    distance = np.abs(points - points.T)
    source = 1.0 / (distance + 0.4)
    matrix = qhmat.HMatrix.from_dense(
        source, points, tolerance=0.25, leaf_size=3,
        bit_widths=(4, 8, 16, 32),
    )
    assert matrix.shape == source.shape
    assert any(not block.dense for block in matrix.blocks)
    error = np.max(np.abs(source - matrix.to_dense()))
    assert error <= matrix.reconstruction_bound + 1e-12
    np.testing.assert_allclose(matrix @ np.eye(12), matrix.to_dense(), atol=1e-12)
    assert matrix.logical_bits < source.size * 64


def test_memory_report_and_verbose(capsys, tmp_path):
    source = np.diag([2.0, 3.0]).astype(np.float32)
    matrix = qhmat.HMatrix.from_dense(
        source, np.arange(2)[:, None], tolerance=0.1, leaf_size=1, verbose=1
    )
    output = capsys.readouterr().out
    assert output.count("Input array:") == 1
    assert "Dense FP64 baseline:" in output
    assert "H-matrix C++ resident estimate:" in output
    report = matrix.storage_report()
    assert report.source_array_bytes == source.nbytes == qhmat.memory_bytes(source)
    assert qhmat.memory_bytes(source[:1]) == source[:1].nbytes
    assert report.dense_fp64_bytes == source.size * 8
    assert report.logical_bits == matrix.logical_bits
    assert report.logical_bytes == (matrix.logical_bits + 7) // 8
    assert report.logical_compression == report.dense_fp64_bytes / (report.logical_bits / 8)
    assert report.resident_bytes == qhmat.memory_bytes(matrix)
    assert report.resident_bytes == qhmat.memory_bytes(matrix._native)
    assert report.resident_bytes > matrix.allocated_bytes
    assert report.resident_compression == report.dense_fp64_bytes / report.resident_bytes

    path = tmp_path / "memory.qhmat"
    matrix.save(path)
    loaded = qhmat.HMatrix.load(path).storage_report()
    assert loaded.source_array_bytes is None
    assert loaded.dense_fp64_bytes == report.dense_fp64_bytes
    assert loaded.logical_bits == report.logical_bits
    assert loaded.resident_bytes >= loaded.allocated_bytes
    with pytest.raises(TypeError, match="NumPy array"):
        qhmat.memory_bytes([1.0, 2.0])


def test_partition_verbose_and_infeasible_output(capsys):
    source = np.diag([2.0, 3.0])
    rectangles = [
        qhmat.Rectangle(i, i + 1, j, j + 1)
        for i in range(2) for j in range(2)
    ]
    qhmat.HMatrix.from_partition(
        source, [0, 1], rectangles, tolerance=0.1, verbose=1
    )
    assert capsys.readouterr().out.count("Input array:") == 1
    qhmat.HMatrix.from_partition(source, [0, 1], rectangles, tolerance=0.1)
    assert capsys.readouterr().out == ""
    with pytest.raises(qhmat.InfeasibleError):
        qhmat.HMatrix.from_dense(
            np.ones((4, 4)), np.arange(4)[:, None], tolerance=1e-12,
            leaf_size=1, bit_widths=(4, 8), verbose=1,
        )
    assert capsys.readouterr().out == ""


def test_infeasible_budget_has_no_dense_offdiagonal_fallback():
    points = np.arange(4, dtype=float)[:, None]
    source = np.ones((4, 4))
    with pytest.raises(qhmat.InfeasibleError):
        qhmat.HMatrix.from_dense(
            source, points, tolerance=1e-12, leaf_size=1, bit_widths=(4, 8)
        )


def test_nonfinite_rejected():
    source = np.eye(3)
    source[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        qhmat.HMatrix.from_dense(source, np.arange(3)[:, None], tolerance=0.1)


def test_numpy_integer_permutation_and_zero_offdiagonal(tmp_path):
    source = np.diag([2.0, 3.0])
    rectangles = [
        qhmat.Rectangle(0, 1, 0, 1), qhmat.Rectangle(0, 1, 1, 2),
        qhmat.Rectangle(1, 2, 0, 1), qhmat.Rectangle(1, 2, 1, 2),
    ]
    matrix = qhmat.HMatrix.from_partition(
        source, np.array([1, 0], dtype=np.int64), rectangles, tolerance=0.1
    )
    np.testing.assert_array_equal(matrix.to_dense(), source)
    assert sum(not block.dense for block in matrix.blocks) == 2
    path = tmp_path / "zero-rank.qhmat"
    matrix.save(path)
    np.testing.assert_array_equal(qhmat.HMatrix.load(path).to_dense(), source)


@pytest.mark.parametrize("bit", [2, 4, 8, 12, 16, 24, 32])
def test_random_orthogonal_factor_quantization(bit):
    rng = np.random.default_rng(200 + bit)
    u, _ = np.linalg.qr(rng.normal(size=(9, 3)))
    v, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    sigma = np.array([3.0, 1.0, 0.2])
    a = (u * sigma) @ v.T
    native = qhmat.NativeHMatrix(17, list(range(17)))
    native.add_dense(qhmat.Rectangle(0, 9, 0, 9), np.eye(9))
    native.add_dense(qhmat.Rectangle(9, 17, 9, 17), np.eye(8))
    rect = qhmat.Rectangle(0, 9, 9, 17)
    native.add_low_rank(rect, u, sigma, v, [bit] * 3, 1, 0.0)
    native.add_low_rank(
        qhmat.Rectangle(9, 17, 0, 9), v, sigma, u, [bit] * 3, 1, 0.0
    )
    native.finalize()
    expected = np.eye(17)
    expected[:9, 9:] = a
    expected[9:, :9] = a.T
    reconstructed = native.reconstruct()
    assert np.max(np.abs(reconstructed - expected)) <= native.reconstruction_bound + 1e-12
    x = rng.normal(size=(17, 4))
    np.testing.assert_allclose(native.matmul(x), reconstructed @ x, atol=1e-12)


def test_cuda_request_without_cuda_build():
    if qhmat.cuda_built:
        pytest.skip("CUDA-enabled wheel requires an NVIDIA runner")
    matrix = qhmat.NativeHMatrix(1, [0])
    matrix.add_dense(qhmat.Rectangle(0, 1, 0, 1), np.array([[2.0]]))
    matrix.finalize()
    with pytest.raises(RuntimeError, match="without CUDA"):
        matrix.matmul(np.array([1.0]), True)


def test_packed_file_round_trip_and_corruption(tmp_path):
    points = np.linspace(0, 1, 8)[:, None]
    source = 1.0 / (0.4 + np.abs(points - points.T))
    matrix = qhmat.HMatrix.from_dense(
        source, points, tolerance=0.25, leaf_size=2,
        bit_widths=(4, 8, 16, 32),
    )
    path = tmp_path / "matrix.qhmat"
    matrix.save(path)
    loaded = qhmat.HMatrix.load(path)
    np.testing.assert_array_equal(loaded.to_dense(), matrix.to_dense())
    np.testing.assert_array_equal(loaded @ np.eye(8), matrix @ np.eye(8))
    assert loaded.logical_bits == matrix.logical_bits
    assert loaded.reconstruction_bound == matrix.reconstruction_bound
    with pytest.raises(ValueError, match="existing"):
        matrix.save(path)

    bad = tmp_path / "bad.qhmat"
    data = bytearray(path.read_bytes())
    data[-9] ^= 1
    bad.write_bytes(data)
    with pytest.raises(RuntimeError, match="checksum"):
        qhmat.HMatrix.load(bad)

    truncated = tmp_path / "truncated.qhmat"
    truncated.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(RuntimeError, match="truncated|trailing"):
        qhmat.HMatrix.load(truncated)
