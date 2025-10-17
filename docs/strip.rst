.. _strip:

:mod:`~aptapy.strip` --- Strip charts
=====================================

This module provides a  :class:`~aptapy.strip.StripChart` class representing a
sliding strip chart, that is, a scatter plot where the number of points is limited
to a maximum, so that the thing acts essentially as a sliding window, typically in time.
This is mainly meant to represent the time history of a signal over a reasonable
span---a long-term acquisition might go on for weeks, and it would not make sense
to try and plot on the screen millions of points, but the last segment of the
acquisition is the most important part when we want to monitor what is happening.

Internally the class uses two distinct :class:`collections.deque` objects to store
the data points, and the public interface is modeled after that of deques: you add
a single point with :meth:`~aptapy.strip.StripChart.append()`, add multiple points
with :meth:`~aptapy.strip.StripChart.extend()`, and clear things up with
:meth:`~aptapy.strip.StripChart.clear()`.

.. code-block:: python

    from aptapy.strip import StripChart

    chart = StripChart(max_length=1000, label='Signal')

    # add a single point
    chart.append(0., 0.)

    # add multiple points
    chart.extend([1., 2., 3.], [4., 5., 6.])

    # plot the current contents of the strip chart
    chart.plot()

.. seealso::

   Have a look at the :ref:`sphx_glr_auto_examples_simple_strip_chart.py` and
   :ref:`sphx_glr_auto_examples_interactive_cursor.py` examples.


Strip charts and wall time
--------------------------

A fairly common use case for strip charts is to plot the time history of a signal
against wall time. In this case, the x-axis represents time in seconds since the
epoch (i.e., POSIX timestamps), e.g., from a call to ``time.time()``. In order to
facilitate this use case, the module provides the :class:`~aptapy.strip.EpochStripChart`
class.

.. note::

   ``time.time()`` always returns a floating-point number, representing seconds since the
   Unix epoch (January 1, 1970, 00:00:00 UTC).
   Note the UTC part: ``time.time()`` is not affected by the local time zone or daylight
   saving time (not even mentioning leap seconds), and we do nothing clever internally
   to this module to even try and keep track of any of that.

Internally, the class converts the seconds-since-epoch values to numpy ``datetime64``
objects at plotting time. We support ``s``, ``ms``, ``us``, and ``ns`` resolutions, with
the default being ``ms`` (milliseconds), see the :class:`~aptapy.strip.EpochStripChart`
documentation.

.. note::

   ``datetime64`` is the numpy native timestamp data type. Unlike Python's ``datetime.datetime``,
   which stores a full object per entry, ``datetime64`` is a compact numeric type that
   stores timestamps as integer counts since the Unix epoch (1970-01-01) with a fixed
   time unit. That time unit is specified in brackets, e.g., ``datetime64[s]``
   (second precision), ``datetime64[ms]`` (millisecond precision) and
   ``datetime64[ns]`` (nanosecond precision).

   In order to convert from POSIX timestamps (i.e., floating-point seconds since epoch)
   to ``datetime64``, numpy simply multiplies the floating-point number by the
   appropriate factor (1 for seconds, 1000 for milliseconds and so on and so forth) and
   then casts to integer. This is all done internally to the
   :class:`~aptapy.strip.EpochStripChart` class. Good for you.

The class does its best to format the x-axis labels in a human-friendly way,
so that the appearance is sensible regardless of the time span represented in the strip
chart. Open an issue if you do run into an edge case!

.. seealso::

   Have a look at the :ref:`sphx_glr_auto_examples_epoch_strip_chart.py` example.




Module documentation
--------------------

.. automodule:: aptapy.strip
