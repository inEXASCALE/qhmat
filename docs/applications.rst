Applications
============

These small CPU examples use only NumPy and ``qhmat``. Each reports the
max-norm matrix reconstruction difference, the representation bound, and the
application difference against a dense multiplication. The examples use
moderate tolerances to keep construction feasible on ordinary laptops;
production tolerances and ranks should be chosen for the target problem.

Gaussian point kernel
---------------------

For points :math:`x_i\in\mathbb{R}^2`, define

.. math::

   A_{ij} = \exp\left(-\frac{\|x_i-x_j\|_2^2}{2\ell^2}\right),
   \qquad Y=AX.

The example uses points on a smooth curve, :math:`\ell=0.25`, and two
right-hand sides. Bounding-box separation determines admissibility; it does
not change the rule that every non-diagonal block is factorized.

.. literalinclude:: examples/kernel.py
   :language: python
   :caption: docs/examples/kernel.py

Run it with ``python docs/examples/kernel.py``.

Planar BEM single-layer matrix
------------------------------

Let :math:`\Gamma` be a polygonal approximation of the unit circle with
straight panels :math:`\Gamma_j`. The 2D Laplace single-layer kernel is

.. math::

   G(x,y)=-\frac{1}{2\pi}\log\|x-y\|_2,
   \qquad A_{ij}=\int_{\Gamma_j}G(x_i,y)\,ds_y,

where :math:`x_i` is the midpoint of panel :math:`i`. For :math:`i\ne j`,
the example uses midpoint quadrature,

.. math::

   A_{ij}\approx -\frac{|\Gamma_j|}{2\pi}\log\|x_i-x_j\|_2.

For the straight self panel of length :math:`\ell_i`, the singular integral
is evaluated analytically:

.. math::

   A_{ii}=-\frac{\ell_i}{2\pi}
   \left(\log\frac{\ell_i}{2}-1\right).

The panel endpoints define ``support_lower`` and ``support_upper`` so the
cluster boxes enclose entire panels. This is a small collocation example,
not an assertion that midpoint quadrature is sufficient for near-singular
off-diagonal interactions in a production BEM solver.

.. literalinclude:: examples/bem.py
   :language: python
   :caption: docs/examples/bem.py

Run it with ``python docs/examples/bem.py``.

Integral operator on an interval
--------------------------------

Consider

.. math::

   (Tf)(x)=\int_0^1 e^{-|x-y|}f(y)\,dy.

With midpoint nodes :math:`x_i=(i+\tfrac12)/n` and uniform quadrature
weights :math:`w_j=1/n`, the Nyström matrix is

.. math::

   A_{ij}=e^{-|x_i-x_j|}w_j,\qquad (Tf)(x_i)\approx (Af)_i.

The quadrature weights are included in the matrix before compression. The
reported ``reconstruction_bound`` concerns this discretized matrix; it does
not bound the continuum quadrature error :math:`Tf-Af`.

.. literalinclude:: examples/integral.py
   :language: python
   :caption: docs/examples/integral.py

Run it with ``python docs/examples/integral.py``.
