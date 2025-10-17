.. _hist:

:mod:`~aptapy.hist` --- Histograms
==================================

The module provides an abstract base class for n-dimensional histograms along
with concrete implementations for 1D and 2D histograms:

* :class:`~aptapy.hist.AbstractHistogram`: Abstract base class for histograms;
* :class:`~aptapy.hist.Histogram1d`: 1D histogram implementation;
* :class:`~aptapy.hist.Histogram2d`: 2D histogram implementation.

Histograms are constructed with the bin edges (and, optionally, labels to be
used at the plotting stage) and are filled using the
:meth:`~aptapy.hist.AbstractHistogram.fill` method. The basic semantics is as
follows:

.. code-block:: python

    import numpy as np
    from aptapy.hist import Histogram1d, Histogram2d

    rng = np.random.default_rng()
    edges = np.linspace(-5., 5., 100)

    hist = Histogram1d(edges, "x")
    hist.fill(rng.normal(size=1000))
    hist.plot()

    hist = Histogram2d(edges, edges, 'x', 'y')
    hist.fill(rng.normal(size=1000), rng.normal(size=1000))
    hist.plot()

Histograms support weighted filling and basic arithmetic operations (addition
and subtraction) between histograms with identical binning.

.. warning::

   Multiplication by a scalar is not yet supported.

.. seealso::

   Have a look at the :ref:`sphx_glr_auto_examples_simple_hist1d.py`,
   :ref:`sphx_glr_auto_examples_simple_hist2d.py` and
   :ref:`sphx_glr_auto_examples_weighted_hist1d.py` examples.


Module documentation
--------------------

.. automodule:: aptapy.hist
