# Installation

## From a checkout

`qhmat` currently has no published PyPI release. A source build needs Python 3.10+, a C++17 compiler, and CMake. The default installation is CPU-only; NumPy, scikit-build-core, and pybind11 are handled by pip.

=== "macOS / Linux"

    ```bash
    git clone https://github.com/inEXASCALE/qhmat.git
    cd qhmat
    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -e '.[test,docs]'
    python -c 'import qhmat; print(qhmat.__version__, qhmat.cuda_built)'
    ```

=== "Windows PowerShell"

    ```powershell
    git clone https://github.com/inEXASCALE/qhmat.git
    cd qhmat
    py -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e ".[test,docs]"
    python -c "import qhmat; print(qhmat.__version__, qhmat.cuda_built)"
    ```

!!! note "Use one Python environment"
    Run `python -m pip`, `python -m pytest`, and `python -m mkdocs` with the **same interpreter**. A missing `qhmat._core` indicates that the native extension was not installed in the environment running the command. `python -c 'import qhmat._core; print(qhmat._core.__file__)'` checks the compiled extension directly.

For a non-editable install use `python -m pip install .`. The default CPU build does not require CUDA.

## Optional CUDA build

Enable the optional backend on a host with a compatible NVIDIA toolkit, driver, GPU, and supported host compiler. For an A100, compute capability 8.0 is selected by:

```bash
python -m pip install . \
  -Ccmake.define.QHMAT_ENABLE_CUDA=ON \
  -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80
python -c 'import qhmat; print(qhmat.cuda_built)'
```

`cuda_built` reports that the extension contains CUDA support, **not** that a usable GPU is present. The CPU `@` operator stays on CPU; request CUDA explicitly with `h.matmul(x, device="cuda")`. The current CUDA path is a correctness backend and uploads data on each call, so it is not evidence of a speedup.

## Build the documentation

```bash
python -m pip install -e '.[docs]'
make html
```

The static site is written to `docs/_build/html/index.html`. `make -C docs html` is equivalent. For a live local preview, run `python -m mkdocs serve`; the build uses `--strict` so broken internal links and other MkDocs warnings fail CI.

## Quick verification

```bash
python -m pytest -q
python docs/examples/kernel.py
python docs/examples/bem.py
python docs/examples/integral.py
```

The numerical examples use small matrices so they run on ordinary laptops. GPU validation must be performed separately on an NVIDIA host.
