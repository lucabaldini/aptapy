.. aptapy documentation master file, created by
   sphinx-quickstart on Sun Aug 24 11:47:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

aptapy documentation
====================

.. image:: _static/logo.png
   :alt: Project logo
   :width: 200px
   :align: left

This is a small, pure-Python library providing statistical tools for online monitoring
and analysis of experimental data, with a focus on histogramming, time series, and
fitting. It is designed to be lightweight and easy to use, making it suitable for
integration into existing data processing pipelines.

More specifically, the functionality provided by the package include:

* one dimensional fitting models, providing a slick, object-oriented alternative
  to `scipy.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_,
  supporting arbitrarily complex fits with bound and/or frozen parameters;
  (this is somewhat similar in spirit to the beautiful, and much more mature,
  `lmfit <https://lmfit.github.io/lmfit-py/>`_ package, which is definitely
  worth looking at);
* interactive n-dimensional histograms, supporting weights, error propagation,
  basic arithmetics, and fitting interface;
* strip charts for online monitoring of series data, with support for interactive cursors
  and zooming, and for time series with POSIX timestamps.

The :doc:`example gallery <auto_examples/index>` is probably the best place to start.
Have fun!


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install
   auto_examples/index
   develop
   team
   release_notes

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   plotting
   hist
   modeling
   models
   strip
