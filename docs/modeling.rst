.. _modeling:

:mod:`~aptapy.modeling` --- Fitting models
==========================================

The modeling module provides tools for fitting models to data, including parameter
estimation and uncertainty quantification.

Readily available simple models include

* :class:`~aptapy.modeling.Constant`: a constant value;
* :class:`~aptapy.modeling.Line`: a straight line;
* :class:`~aptapy.modeling.PowerLaw`: a power law;
* :class:`~aptapy.modeling.Gaussian`: a Gaussian function.

More complex models can be built by summing simple ones, e.g.,

>>> from aptapy.modeling import Line, Gaussian
>>>
>>> model = Line() + Gaussian()

The main fitting engine supports bounded fits and/or fits with fixed parameters.

.. seealso::

   Have a look at the :ref:`sphx_glr_auto_examples_simple_fit.py`,
   :ref:`sphx_glr_auto_examples_composite_fit.py` and
   :ref:`sphx_glr_auto_examples_constrained_fit.py` examples.



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
>>>
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

.. note::

   It goes without saying that the order of the fit parameters in the argument
   list of the :meth:`~aptapy.modeling.AbstractFitModel.evaluate` method must
   match the order in which they are defined as class attributes.

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


Fitting primer
~~~~~~~~~~~~~~

Assuming that you have a set of data points ``xdata``, ``ydata``, the latter with
associated uncertainties ``yerrors``, the simplest fit goes like

>>> from aptapy.modeling import Line
>>>
>>> model = Line()
>>> status = model.fit(xdata, ydata, sigma=yerrors)

You can fit within a subrange of the input data by specifying the ``min`` and/or
the ``max`` keyword arguments:

>>> from aptapy.modeling import Line
>>>
>>> model = Line()
>>> status = model.fit(xdata, ydata, sigma=yerrors, xmin=0., xmax=10.)

You can set bounds on the fit parameters, e.g., force the slope to be positive
by doing

>>> from aptapy.modeling import Line
>>>
>>> model = Line()
>>> model.slope.minimum = 0.
>>> status = model.fit(xdata, ydata, sigma=yerrors)

and you can freeze any of the parameters to a fixed value during the fit

>>> from aptapy.modeling import Line
>>>
>>> model = Line()
>>> model.intercept.freeze(0.)
>>> status = model.fit(xdata, ydata, sigma=yerrors)

Or, really, any linear combination of the above. The fit status is able to
pretty-print itself, and the fitted model can be plotted by just doing

>>> model.plot()
>>> plt.legend()

(The legend bit is put there on purpose, as by default the fitting model will
add a nice entry in the legend with all the relevant information.)

Fitting models interact nicely with one-dimensional histograms from the
:mod:`~aptapy.hist`, module so you can do

>>> import numpy as np
>>> from aptapy.hist import Histogram1D
>>> from aptapy.modeling import Line
>>>
>>> hist = Histogram1D(np.linspace(0., 1., 100))
>>> hist.fill(np.random.rand(1000))
>>> model = Line()
>>> status = model.fit_histogram(hist)


Composite models
----------------

The modeling module also provides a way to build composite models by summing
simple ones. This is achieved by means of the :class:`~aptapy.modeling.FitModelSum`,
which is design to hold a list of components and interoperate with the rest
of the world in exactly the same fashion as simple models.

Chances are you will never have to instantiate a :class:`~aptapy.modeling.FitModelSum`
object directly, as the ``+`` operator will do the trick in most of the cases, e.g.,

>>> from aptapy.modeling import Line, Gaussian
>>> model = Line() + Gaussian()
>>> status = model.fit(xdata, ydata, sigma=yerrors)



Module documentation
--------------------

.. automodule:: aptapy.modeling
