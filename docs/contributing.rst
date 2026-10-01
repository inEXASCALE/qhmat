Collaboration and contributing
==============================

Contributions that improve correctness, documentation, portability, and
reproducibility are welcome. Before a substantial change, open an issue in
the `qhmat issue tracker <https://github.com/inEXASCALE/qhmat/issues>`_ with
the use case and expected behavior. Small fixes can go directly to a pull
request.

Development workflow
--------------------

1. Fork and clone the repository, then create a focused branch.
2. Install the package and documentation tools with
   ``python -m pip install -e '.[test,docs]'``.
3. Add focused tests for changed behavior. For numerical results, state the
   matrix, tolerance, reference computation, and error metric explicitly.
4. Run ``python -m pytest -q`` and ``make html``. The documentation build is
   strict about Sphinx warnings and unresolved references.
5. Open a pull request describing the change, tests, and any limitations.

Code and experimental boundaries
--------------------------------

The public Python API is in ``python/qhmat``; the C++ core and pybind11
bindings are in ``src/qhmat``. Keep the paper's independent ``experiments/``
workflows separate from package behavior unless the change explicitly needs
both. Document whether a size is logical payload, C++ resident estimate,
serialized file length, or process RSS. Do not infer GPU speedup from a
correctness-only CUDA check.

CUDA changes should be tested on a compatible NVIDIA host as well as through
the CPU test suite. The default CPU installation must continue to work
without the CUDA toolkit. Examples and documentation should execute with the
released public API rather than private helpers.

Project links
-------------

- `Source repository <https://github.com/inEXASCALE/qhmat>`_
- `Issues and feature requests <https://github.com/inEXASCALE/qhmat/issues>`_
- :doc:`license`
