.. _plotting:

:mod:`~aptapy.plotting` --- Plotting tools
==========================================

This module provides all the plotting facilities that the other modules in the package
make use of.

At the very basic level, the module provides a complete matplotlib setup tailored
for interactive use in a GUI environment. This is encapsulated in the
:meth:`~aptapy.plotting.apply_stylesheet()` function, which is automatically called
with the default stylesheet when importing the module.

.. note::

   In order to have a consistent plotting experience you are advised to always
   import ``pyplot`` from this module, rather than directly from ``matplotlib``,
   i.e., use:

   .. code-block:: python

       from aptapy.plotting import plt

   rather than:

   .. code-block:: python

       from matplotlib import pyplot as plt

This will ensure that the configuration block is properly executed.

The :meth:`~aptapy.plotting.setup_axes()` and :meth:`~aptapy.plotting.setup_gca()` functions
provide a handy shorcut to set up axes via keyword arguments, encompassing the
most common customizations (titles, labels, grids, legends, etc.).


Interactive cursors
-------------------

The module provides a zoomable, interactive cursor object, implemented in the
:class:`~aptapy.plotting.VerticalCursor` class. When activated, a cursor displays
the numerical values of the x and y coordinates of the plottable 1-dimensional
objects, and allows to zoom in and out interactively on the matplotlib canvas,
more specifically:

* left click and drag: select a rectangle for zooming in, with the zoom being
  applier on release;
* right click: restore the initial view.

The cursor follows the mouse position when no button is clicked.

.. seealso::

   Cursors interact seamlessly with :class:`~aptapy.strip.StripChart` objects,
   as illustrated in the :ref:`sphx_glr_auto_examples_interactive_cursor.py`
   example.

.. warning::

   At this time the cursor code is not optimized for efficiency---keep this in mind
   of the experience is not super fluid. There is undoubtedly room for improvement,
   e.g., using blitting (see `issue #11 <https://github.com/lucabaldini/aptapy/issues/11>`_).
   but we would like to let the API settle before we venture into that.


The plottable hierarchy
-----------------------

The module defines an abstract base class, :class:`~aptapy.plotting.AbstractPlottable`,
which is the base class for all objects that can be plotted on a matplotlib
canvas. The class defines the basic interface that all plottable objects must
implement, as well as some common functionality, and is inherited by all the histogram
and strip chart classes defined in the package, as well as by all the fitting models.


Styling matplotlib
------------------

All native matplotlib styling facilities aside, aptapy provides a few custom stylesheets
that can be used to style plots consistently across the package. The stylesheets are
stored in the ``aptapy/styles`` folder, and can be applied in two different
ways:

* globally, via the :meth:`~aptapy.plotting.apply_stylesheet()` function;
* temporarily, via the :meth:`~aptapy.plotting.stylesheet_context()` context manager.

.. seealso::

   The :ref:`sphx_glr_auto_examples_dark_theme.py` and
   :ref:`sphx_glr_auto_examples_xkcd_theme.py`
   examples illustrate some of the custom aptapy stylesheets.

.. warning::

   If you are using a matplotlib version newer than 3.7.1, you can use the
   dotted package-style syntax in conjunction with all the matplotlib styling
   facilities, and refer to the aptapy stylesheets directly as, e.g.,
   ``aptapy.styles.aptapy-dark``. The custom functions provided in this module
   allow you to refer to the aptapy stylesheets by name (e.g., ``aptapy-dark``)
   and should support older matplotlib versions as well.


Module documentation
--------------------

.. automodule:: aptapy.plotting

