.. _plotting:

:mod:`~aptapy.plotting` --- Plotting tools
==========================================

This module provides all the plotting facilities that the other modules in the package
make use of.

At the very basic level, the module provides a complete matplotlib setup tailored
for interactive use in a GUI environment. This is encapsulated in the
:meth:`~aptapy.plotting.configure()` function, which is automatically called when
importing the module.

.. note::

   In order to have a consistent plotting experience you are advised to always
   import ``pyplot`` from this module, rather than directly from ``matplotlib``,
   i.e., use:

   .. code-block:: python

       from aptapy.plotting import plt

    rather than:

    ..code-block:: python

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


Module documentation
--------------------

.. automodule:: aptapy.plotting

