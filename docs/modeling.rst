.. _modeling:

:mod:`~aptapy.modeling` --- Fitting models
==========================================

The modeling module provides tools for fitting models to data, including parameter
estimation and uncertainty quantification.

Parameters
----------

The first central concept in the modeling module is that of a fit parameter,
represented by the :class:`~aptapy.modeling.FitParameter` class. A fit parameter
is a named mutable object that holds a value, an optional uncertainty, and optional
bounds, along with a flag that indicate whether they should be varied or not in a fit.

:class:`~aptapy.modeling.FitParameter` objects provide all the facilities for
pretty-printing their value and uncertainty. The following example shows the basic
semantics of the class:

>>> from aptapy.modeling import FitParameter
>>> param = FitParameter(1.0, "amplitude", error=0.1)
>>> print(param)
Amplitude: 1.0 ± 0.1


Fit status
----------

:class:`~aptapy.modeling.FitStatus` is a small bookkeeping class that holds all the
information about the status of a fit, such as the chisquare, the number of degrees of
freedom and the fit range.

.. warning::

   At this point the implementation of the class is fairly minimal, and it is very
   likely that we will be adding stuff along the way.


Simple models
-------------

Chances are you will not have to interact with :class:`~aptapy.modeling.FitParameter`
and :class:`~aptapy.modeling.FitStatus` objects a lot, but they are central to defining
and using simple fit models, and heavily used internally.

The easiest way to see how you would go about defining an actual fit model is to
look at the source code for a simple one.

.. literalinclude:: ../src/aptapy/modeling.py
   :language: python
   :pyobject: Line
   :linenos:

All we really have to do is to subclass :class:`~aptapy.modeling.AbstractFitModel`,
listing all the fit parameters as class attributes (assigning them sensible default
values), and implement the :meth:`~aptapy.modeling.AbstractFitModel.evaluate` method,
which takes as first argument the independent variable and then the values of all the
fit parameters.
In this particular case we are sayng that the ``Line`` model has two fit parameters,
``intercept`` and ``slope``, and, well, the model itself evaluates as a straight line
as we would expect.

When we create an instance of a fitting model

>>> model = Line()

a few things happen under the hood:

* the class instance gets its own `copy` of each fit parameter, so that we can
  change their values and settings without affecting the class definition, nor other
  class instances;
* the class instance registers the fit parameters as attributes of the instance,
  so that we can access them as, e.g., ``model.intercept``, ``model.slope``.

That it's pretty much it. The next thing that you proabably want to do is to fit
the model to a series of data points, which you do in pretty much the same fashion
as you would do with ``scipy.optimize.curve_fit`` using the
:meth:`~aptapy.modeling.AbstractFitModel.fit` method. This will return a
:meth:`~aptapy.modeling.FitStatus` object containing information about the fit.




Composite models
----------------



Module documentation
--------------------

.. automodule:: aptapy.modeling
