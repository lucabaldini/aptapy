.. _models:

:mod:`~aptapy.models` --- Fitting models
========================================

Readily available fit models include

* :class:`~aptapy.models.Constant`
* :class:`~aptapy.models.Line`
* :class:`~aptapy.models.Quadratic`
* :class:`~aptapy.models.PowerLaw`
* :class:`~aptapy.models.Exponential`
* :class:`~aptapy.models.ExponentialComplement`
* :class:`~aptapy.models.StretchedExponential`
* :class:`~aptapy.models.StretchedExponentialComplement`
* :class:`~aptapy.models.Gaussian`
* :class:`~aptapy.models.GaussianCDF`
* :class:`~aptapy.models.GaussianCDFComplement`

Polynomial models
-----------------



Location-scale models
---------------------

A number of models included in this module belong to the `location-scale` family,
i.e., they can be expressed in terms of a location parameter and a non-negative
scale parameter and, ultimately, are characterized by a universal shape function
:math:`g(z)` of the standardized variable

.. math::
    z = \frac{x - m}{s},

where :math:`m` is the location parameter and :math:`s` is the scale parameter.
The gaussian probability density function is the prototypical example of
location-scale model (with the mean as location and the standard deviation
as scale), but many other models belong to this family---both peak-like and
sigmoid-like.

From the point of view of the practical implementation, all location-scale
models in :mod:`aptapy.models` inherit from the base class
:class:`~aptapy.modeling.AbstractLocationScaleFitModel`, which provides all the
necessary common functionality. All of them features at least three parameters:

* `amplitude`: a multiplicative factor, whose precise meaning depends on the context;
* `location`: the location parameter;
* `scale`: the scale parameter.

Concrete classes inheriting from :class:`~aptapy.modeling.AbstractLocationScaleFitModel`
must implement the :meth:`~aptapy.modeling.AbstractLocationScaleFitModel.shape`,
method, providing the universal shape function :math:`g(z)`.


Peak-like models
~~~~~~~~~~~~~~~~

Location-scale peak-like models all inherit from the abstract base class
:class:`~aptapy.modeling.AbstractPeakFitModel`, and at the very minimum they must
provide a concrete implementation of the shape function.
The basic contract, here, is that the latter is normalized to unit area, and the
amplitude parameter of the general model

.. math::
    f(x; A, m, s, ...) = \frac{A}{s} g\left(\frac{x - m}{s}; ...\right)

represents the area under the peak. The abstract base class implements the
evaluation method, which delegates the actual computation to the shape
function to be implemented in derived classes:

.. literalinclude:: ../src/aptapy/modeling.py
   :language: python
   :pyobject: AbstractPeakFitModel.evaluate

where the dots indicate any additional model parameters.


Sigmoid models
~~~~~~~~~~~~~~

Location-scale sigmoid-like models all inherit from the abstract base class
:class:`~aptapy.modeling.AbstractSigmoidFitModel`, and, just like in the previous
case, they must provide a concrete implementation of the shape function.
The latter is expected to be a monotonically increasing function, ranging
from 0 to 1 as its argument goes from -infinity to +infinity, and the meaning
of the amplitude parameter in this case is that of the total change in the function
value across the transition region, so that the general model reads

.. math::
    f(x; A, m, s, ...) = A g\left(\frac{x - m}{s}; ...\right)

(note that there is no division by the scale parameter in this case). Conversely,
the implementation of the evaluation method reads:

.. literalinclude:: ../src/aptapy/modeling.py
   :language: python
   :pyobject: AbstractSigmoidFitModel.evaluate


Module documentation
--------------------

.. automodule:: aptapy.models
