# Contributing

Contributions that improve numerical correctness, portability, documentation, tests, or performance measurement are welcome. For a substantial behavioral change, start with an [issue](https://github.com/inEXASCALE/qhmat/issues) explaining the use case and the proposed API. Small fixes can go directly to a focused pull request.

## Development workflow

1. Fork and clone the repository; make a focused branch.
2. Install the package and development tools with `python -m pip install -e '.[test,docs]'`.
3. Add focused tests. For numerical claims, specify the matrix, error metric, tolerance, reference computation, and arithmetic mode.
4. Run `python -m pytest -q` and `make html` locally. The latter uses strict MkDocs warnings.
5. Open a pull request describing the behavior change, tests, and known limitations.

The public Python API lives in `python/qhmat`; C++ and pybind11 code live in `src/qhmat`. The paper's `experiments/` workflows are independent research scripts, not part of the installed package. Avoid changing their result formats as an incidental side effect of package work.

## Documentation contributions

The site uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and Markdown files in `docs/source`. `mkdocs.yml` defines navigation and the automated Python API extraction. Runnable examples live in `docs/examples` and are exercised by `tests/test_docs_examples.py`. Add an example script when a tutorial needs more than a short snippet; do not let displayed code drift away from the public API.

```bash
python -m pip install -e '.[test,docs]'
make html
python -m pytest -q
```

The generated output is `docs/_build/html` and is not committed. Read the Docs uses the same `mkdocs.yml` through `.readthedocs.yaml`.

## Review principles

- State whether a storage number is **logical payload**, native vector payload, C++ resident estimate, serialized file size, or process RSS.
- Keep representation, application, and discretization errors separate.
- Test CPU behavior without requiring a CUDA toolkit. CUDA changes also need validation on a compatible NVIDIA host.
- Do not imply that a correctness-only CUDA path is faster than an optimized dense or floating-factor baseline without matched timing evidence.

The project is licensed under [MPL-2.0](license.md). Source and issue tracking are at [inEXASCALE/qhmat](https://github.com/inEXASCALE/qhmat).
