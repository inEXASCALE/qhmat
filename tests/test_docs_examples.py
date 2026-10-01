"""Keep published application examples executable with the public API."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("name", ["kernel", "bem", "integral"])
def test_documentation_example(name):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "docs" / "examples" / f"{name}.py")],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "max application difference:" in result.stdout
    assert "representation bound:" in result.stdout
