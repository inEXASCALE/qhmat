import numpy as np
import pytest

import qhmat


@pytest.mark.skipif(not qhmat.cuda_built, reason="qhmat was built without CUDA")
def test_cuda_matches_cpu_for_packed_blocks():
    points = np.linspace(0.0, 1.0, 10)[:, None]
    source = 1.0 / (0.4 + np.abs(points - points.T))
    matrix = qhmat.HMatrix.from_dense(
        source, points, tolerance=0.25, leaf_size=2,
        bit_widths=(4, 8, 16, 32),
    )
    rng = np.random.default_rng(11)
    rhs = rng.normal(size=(10, 3))
    try:
        gpu = matrix.matmul(rhs, device="cuda")
    except RuntimeError as exc:
        if "no CUDA device" in str(exc) or "cudaGetDeviceCount" in str(exc):
            pytest.skip(str(exc))
        raise
    np.testing.assert_allclose(gpu, matrix.matmul(rhs), rtol=1e-12, atol=1e-12)
