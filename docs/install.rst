Installation
============

Requirements
------------

``qhmat`` needs Python 3.10 or newer, NumPy, a C++17 compiler, and CMake to
build from source. The package uses scikit-build-core and pybind11; pip installs
the build dependencies automatically. The default installation is CPU-only.
The PyPI release is pending, so install from a local checkout for now.

.. code-block:: console

   git clone https://github.com/inEXASCALE/qhmat.git
   cd qhmat
   python -m pip install .
   python -c "import qhmat; print(qhmat.__version__)"

Editable development installation
---------------------------------

.. code-block:: console

   python -m pip install -e '.[test,docs]'
   python -m pytest -q

CUDA backend (optional)
-----------------------

CUDA needs a compatible toolkit, host compiler, NVIDIA driver, and GPU. For
example, on an A100 (compute capability 8.0):

.. code-block:: console

   python -m pip install . -Ccmake.define.QHMAT_ENABLE_CUDA=ON \
     -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80
   python -c "import qhmat; print(qhmat.cuda_built)"

The CPU ``@`` operator does not switch to CUDA automatically. Select it
explicitly with ``h.matmul(x, device="cuda")``. The current CUDA path uploads
the representation and right-hand side on each call; it should not be treated
as a performance claim.

Build this documentation
------------------------

With the editable development installation above, run from the repository
root:

.. code-block:: console

   make html

Open ``docs/_build/html/index.html`` in a browser. ``make -C docs html`` is
equivalent. The build treats Sphinx warnings and broken references as errors.
