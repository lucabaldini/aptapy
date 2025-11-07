.. _models:

:mod:`~aptapy.models` --- Fitting models
========================================

This page documents the various fitting models readily available in the package.


Polynomials
-----------

:class:`~aptapy.models.Constant`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = c
    \quad \text{with} \quad
    c \rightarrow \texttt{value}


:class:`~aptapy.models.Line`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = mx + q
    \quad \text{with} \quad
    \begin{cases}
    m \rightarrow \texttt{slope} \\
    q \rightarrow \texttt{intercept}
    \end{cases}


:class:`~aptapy.models.Quadratic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Exponentials and power-laws
---------------------------

:class:`~aptapy.models.PowerLaw`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = N x^\Gamma
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    \Gamma \rightarrow \texttt{index}
    \end{cases}


:class:`~aptapy.models.Exponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = N \exp \left\{-\frac{(x - x_0)}{X}\right\}
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
    \end{cases}



:class:`~aptapy.models.ExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = N \left [ 1- \exp\left\{-\frac{(x - x_0)}{X}\right\} \right ]
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
    \end{cases}

:class:`~aptapy.models.StretchedExponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = N \exp \left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\}
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    \gamma \rightarrow \texttt{stretch}\\
    x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
    \end{cases}


:class:`~aptapy.models.StretchedExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    f(x) = N \left [ 1- \exp\left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\} \right ]
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    \gamma \rightarrow \texttt{stretch}\\
    x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
    \end{cases}



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


:class:`~aptapy.models.Alpha`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from `SciPy.stats.alpha <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html>`_.

This is a peak-like model defined in terms of a standardized shape function

.. math::

    f(x, a) = \frac{1}{x^2 \Phi(a) \sqrt{2\pi}} \; \exp(-\frac{1}{2} (a-1/x)^2)

where :math:`\Phi` is the normal CDF, :math:`x > 0`, and :math:`a > 0`.

The distribution is asymmetric, with a a long tail on the right. The mean and the
standard deviation of the distribution are always infinite. As the shape parameter
``a`` increases, the distribution tends to be more and more peaked in proximity of the
origin.


:class:`~aptapy.models.Anglit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from `SciPy.stats.anglit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anglit.html>`_.

This is a peak-like model defined in terms of a standardized shape function

.. math::

    f(x) = \sin(2x + \pi/2) = \cos(2x)

for :math:`-\pi/4 \le x \le \pi/4`.


:class:`~aptapy.models.Argus`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from `SciPy.stats.argus <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.argus.html>`_.

This is a peak-like model defined in terms of a standardized shape function

.. math::

    f(x, \chi) = \frac{\chi^3}{\sqrt{2\pi} \Psi(\chi)} x \sqrt{1-x^2}
                    \exp(-\chi^2 (1 - x^2)/2)

for :math:`0 < x < 1` and :math:`\chi > 0`, where

.. math::

    \Psi(\chi) = \Phi(\chi) - \chi \phi(\chi) - 1/2

with :math:`\Phi` and :math:`\phi` being the CDF and PDF of a standard
normal distribution, respectively.

The distribution is asymmetric, with a a longer tail on the left. As the shape parameter
:math:`\chi` increases, the distribution tends to be more and more peaked in proximity
of 1.


:class:`~aptapy.models.Gaussian`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



:class:`~aptapy.models.Lorentzian`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



:class:`~aptapy.models.LogNormal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



:class:`~aptapy.models.Moyal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
