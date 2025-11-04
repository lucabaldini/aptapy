.. _models:

:mod:`~aptapy.models` --- Fitting models
========================================

This page documents the various fitting models readily available in the package.


Polynomials
-----------

:class:`~aptapy.models.Constant`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`~aptapy.models.Line`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`~aptapy.models.Quadratic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Exponentials and power-laws
---------------------------

:class:`~aptapy.models.PowerLaw`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`~aptapy.models.Exponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`~aptapy.models.ExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~aptapy.models.StretchedExponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


:class:`~aptapy.models.StretchedExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Peak-like models
----------------

Peak-like models are location-scale models defined in terms of a standardized
shape function :math:`g(z)`

.. math::
    f(x; A, m, s, ...) = \frac{A}{s} g\left(\frac{x - m}{s}; ...\right)

where :math:`A` is the amplitude (area under the peak), :math:`m` is the location
(parameter specifying the peak position), and :math:`s` is the scale (parameter
specifying the peak width).

.. seealso:: :ref:`modeling`


:class:`~aptapy.models.Gaussian`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gaussian (normal) distribution. This is the prototypical example of
location-scale peak-like model, where the location parameter is the mean and
the scale parameter is the standard deviation. The shape function is given by

.. math::
    g(z) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} z^2}

and the corresponding Python implementation is

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Gaussian.shape

The model has no additional parameters beyond the standard location-scale ones.


:class:`~aptapy.models.Lorentzian`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lorentzian model. This is the same as the Cauchy and Breit-Wigner distributions,
modulo minor differences in the parametrization. The shape function is given by

.. math::
    g(z) = \frac{1}{\pi (1 + z^2)}

and the corresponding Python implementation is

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Lorentzian.shape

The distribution is symmetric with respect to the location parameter and is
noted for not having finite moments of any order beyond the mean (that is, the
tails are fairly prominent).

The model has no additional parameters beyond the standard location-scale ones.


:class:`~aptapy.models.LogNormal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a location-scale model in logarithmic space, i.e., the logarithm of the
variable is distributed according to a normal (Gaussian) distribution. The shape
function is given by

.. math::
    g(z) = \frac{1}{z \sqrt{2 \pi}} e^{-\frac{1}{2} (\ln z)^2}, \quad z > 0

and the corresponding Python implementation is

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: LogNormal.shape

The support of the distribution is :math:`z > 0` in the reduced variable,
i.e., :math:`x > m` in the original variable. It is asymmetric, with a prominent
right tail.

The model has no additional parameters beyond the standard location-scale ones.


:class:`~aptapy.models.Moyal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Moyal distribution is often used in high-energy physics to describe energy
loss spectra as a poor-man approximation to the Landau distribution. The shape
function is given by

.. math::
    g(z) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{1}{2} (z + e^{-z})}

and the corresponding Python implementation is

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Moyal.shape

The model has no additional parameters beyond the standard location-scale ones.


Sigmoid models
--------------

Sigmoid models are location-scale models defined in terms of a standardized
shape function :math:`g(z)` that is a monotonically increasing function,
ranging from 0 to 1 as its argument goes from -infinity to +infinity.

.. note::

   In this case the amplitude parameter does not represent an area (as in peak-like
   models), but rather the total increase of the function from its lower asymptote
   to its upper asymptote. When the amplitude is negative we switch to the
   complement.


:class:`~aptapy.models.Erf`
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. math::
    g(z) = \frac{1}{2} \left(1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right)

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Erf.shape


:class:`~aptapy.models.Logistic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    g(z) = \frac{1}{1 + e^{-z}}

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Logistic.shape


:class:`~aptapy.models.Arctangent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    g(z) = \frac{1}{2} + \frac{1}{\pi} \arctan(z)

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Arctangent.shape


:class:`~aptapy.models.HyperbolicTangent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    g(z) = \frac{1}{2} \left(1 + \tanh(z)\right)

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: HyperbolicTangent.shape


Module documentation
--------------------

.. automodule:: aptapy.models
