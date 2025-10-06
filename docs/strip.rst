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
a single point with :meth:`~aptapy.strip.StripChart.append()`, and multiple points
with :meth:`~aptapy.strip.StripChart.extend()`.

.. code-block:: python

    from aptapy.strip import StripChart

    chart = StripChart(max_length=1000, label='Signal')

    # add a single point
    chart.append(0., 0.)

    # add multiple points
    chart.extend([1., 2., 3.], [4., 5., 6.])

    # plot the current contents of the strip chart
    chart.plot()

Strip chart objects can operate in two fundamentally different modes: if the
``datetime`` flag is ``False``, the x-axis is treated as a simple numeric value
(e.g., time in seconds since the start of the acquisition, or a simple numeric
index), while if it is ``True``, the x-axis is treated as a series of
POSIX timestamps that are converted to datetime objects for plotting purposes.


Module documentation
--------------------

.. automodule:: aptapy.strip
