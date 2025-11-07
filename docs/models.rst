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

Wrapped from :scipy_rv_wrap:`alpha`;
support: :math:`z > 0`;
shape parameter(s): :math:`a > 0`.

(Note the mean and the standard deviation of the distribution are always infinite.)

.. image:: /_static/plots/alpha_shape.png


:class:`~aptapy.models.Anglit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`anglit`;
support: :math:`-\pi/4 \le z \le \pi/4`.

.. image:: /_static/plots/anglit_shape.png


:class:`~aptapy.models.Argus`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`argus`;
support: :math:`0 < z < 1`;
shape parameter(s): :math:`\chi > 0`.

.. image:: /_static/plots/argus_shape.png


:class:`~aptapy.models.Bradford`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`bradford`;
support: :math:`0 < z < 1`;
shape parameter(s): :math:`c > 0`.

.. image:: /_static/plots/bradford_shape.png


:class:`~aptapy.models.Burr`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`burr`;
support: :math:`z > 0`;
shape parameter(s): :math:`c, d > 0`.

.. image:: /_static/plots/burr_shape.png


:class:`~aptapy.models.Burr12`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`burr12`;
support: :math:`z > 0`;
shape parameter(s): :math:`c, d > 0`.

.. image:: /_static/plots/burr12_shape.png


:class:`~aptapy.models.Cauchy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`cauchy`;
support: :math:`z > 0`.

.. image:: /_static/plots/cauchy_shape.png


:class:`~aptapy.models.Chi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`chi`;
support: :math:`z > 0`;
shape parameter(s): :math:`\text{df} > 0`.

.. image:: /_static/plots/chi_shape.png


:class:`~aptapy.models.Chisquare`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`chi2`;
support: :math:`z > 0`
shape parameter(s): :math:`\text{df} > 0`.

.. image:: /_static/plots/chisquare_shape.png


:class:`~aptapy.models.Cosine`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`cosine`;
support: :math:`-\pi \le z \le \pi`.

.. image:: /_static/plots/cosine_shape.png


:class:`~aptapy.models.CrystalBall`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`crystalball`;
support: :math:`-\infty < z < \infty`;
shape parameter(s): :math:`m > 1`, :math:`\beta > 0`.

.. image:: /_static/plots/crystalball_shape.png




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
