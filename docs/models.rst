.. _models:

:mod:`~aptapy.models` --- Fitting models
========================================

This page documents the various fitting models readily available in the package.


Polynomials
-----------



:class:`~aptapy.models.Constant`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant model. (Note this is equivalent to a polynomial of degree zero, but
we keep it separate for clarity.)

.. math::

    f(x;~c) = c
    \quad \text{with} \quad
    c \rightarrow \texttt{value}

Fitting with a constant model is equivalent to computing the weighted average of
the values of the dependent variable.


:class:`~aptapy.models.Line`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A straight-line model. (Note this is equivalent to a polynomial of degree one, but
we keep it separate for clarity.)

.. math::

    f(x;~m, q) = mx + q
    \quad \text{with} \quad
    \begin{cases}
    m \rightarrow \texttt{slope} \\
    q \rightarrow \texttt{intercept}
    \end{cases}


:class:`~aptapy.models.Polynomial`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple polynomial model of arbitrary degree.

.. math::

    f(x;~c_0, c_1, \ldots, c_n) = \sum_{k=0}^{n} c_k x^k
    \quad \text{with} \quad
    c_k \rightarrow \texttt{c0, c1, ..., cn}

The degree of the polynomial is set at initialization time via the ``degree`` argument
in the constructor, e.g.,

>>> quadratic = Polynomial(2)
>>> cubic = Polynomial(degree=3)


:class:`~aptapy.models.Quadratic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple alias for a polynomial of degree two.


:class:`~aptapy.models.Cubic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple alias for a polynomial of degree three.


Exponentials and power-laws
---------------------------

:class:`~aptapy.models.PowerLaw`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A power-law model with the general form:

.. math::

    f(x;~N, \Gamma) = N \left(\frac{x}{x_0}\right)^\Gamma
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    \Gamma \rightarrow \texttt{index}\\
    x_0 \rightarrow \texttt{pivot}~\text{(set at creation time)}
    \end{cases}

.. note::

   The overloaded ``plot()`` method automatically switches to a log-log scale.


:class:`~aptapy.models.Exponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple exponential model:

.. math::

    f(x;~N, X) = N \exp \left\{-\frac{(x - x_0)}{X}\right\}
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    x_0 \rightarrow \texttt{location}~\text{(set at creation time)}
    \end{cases}

.. note::

   Note that the ``location`` parameter is not a fit parameter, but rather a fixed
   offset that is set when the model instance is created, e.g.,

   >>> exponential = Exponential(location=2.0)

   The basic idea behind this is to avoid the degeneracy between the location and the
   prefactor, and this is one of the main reasons this model is not implemented wrapping the
   scipy exponential distribution.


:class:`~aptapy.models.ExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exponential complement, describing an exponential rise:

.. math::

    f(x;~N, X) = N \left [ 1- \exp\left\{-\frac{(x - x_0)}{X}\right\} \right ]
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    x_0 \rightarrow \texttt{location}~\text{(set at creation time)}
    \end{cases}

.. note::

   See notes on :class:`~aptapy.models.Exponential` regarding the ``location`` parameter.


:class:`~aptapy.models.StretchedExponential`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A stretched exponential model:

.. math::

    f(x;~N, X, \gamma) = N \exp \left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\}
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    \gamma \rightarrow \texttt{stretch}\\
    x_0 \rightarrow \texttt{location}~\text{(set at creation time)}
    \end{cases}

.. note::

   See notes on :class:`~aptapy.models.Exponential` regarding the ``location`` parameter.


:class:`~aptapy.models.StretchedExponentialComplement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complement of the stretched exponential model:

.. math::

    f(x;~N, X, \gamma) = N \left [ 1- \exp\left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\} \right ]
    \quad \text{with} \quad
    \begin{cases}
    N \rightarrow \texttt{prefactor}\\
    X \rightarrow \texttt{scale}\\
    \gamma \rightarrow \texttt{stretch}\\
    x_0 \rightarrow \texttt{location}~\text{(set at creation time)}
    \end{cases}

.. note::

   See notes on :class:`~aptapy.models.Exponential` regarding the ``location`` parameter.



Models related to the gaussian
------------------------------

:class:`~aptapy.models.Gaussian`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is actually not wrapped from :scipy_rv_wrap:`norm`, but rather a direct
implementation of the gaussian (normal) distribution, mainly to override the
parameter names.

Among other things, this class provides a method to perform an iterative fit
around the peak, which is not available in the generic :class:`~aptapy.models.Normal`
class.

.. image:: /_static/plots/gaussian_shape.png

.. seealso:: The :class:`~aptapy.models.Normal` is an alternative, equivalent
   implementation wrapping :scipy_rv_wrap:`norm` as all the other location-scale
   models. Any additional features specific to the gaussian distribution will
   be implemented in :class:`~aptapy.models.Gaussian`, so users are encouraged
   to use this class over :class:`~aptapy.models.Normal`, except for testing
   purposes.


:class:`~aptapy.models.Fe55Forest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gaussian line forest for the Kα and Kβ emission features of :math:`^{55}\mathrm{Fe}`
decay, using intensity-weighted mean energies from the X-ray database
(<https://xraydb.seescience.org/>).


:class:`~aptapy.models.Probit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a custom implementation of the inverse of the cumulative distribution function
(also known as the percent-point function) of a normal distribution with generic
location and scale.

.. math::
    f(x;~\mu,~\sigma) = \mu + \sigma \Phi^{-1}(x)
    \quad \text{with} \quad
    \begin{cases}
    \mu \rightarrow \texttt{offset}\\
    \sigma \rightarrow \texttt{sigma}
    \end{cases}

where :math:`\Phi^{-1}(x)` is the actual Probit function (the percent-point function of
the standard normal distribution).

.. note::

   For completeness, this is internally implemented using the :func:`scipy.special.ndtri`
   function, rather than using :meth:`scipy.stats.norm.ppf`, but the two are
   fully equivalent.

The support of this model is the interval :math:`0 < x < 1`, since the Probit
function diverges at the boundaries. Therefore, when plotting this model, the
default plotting range is automatically set to a slightly smaller interval in order
to avoid issues at the boundaries.

Note that, since the mean of the underlying normal distribution translates into
a vertical shift of the Probit function, the former is called ``offset`` in this
context---not ``mu``. For completeness, the ``sigma`` parameter controls the
scale of the vertical excursion of the model. Also note that we do not provide a
``prefactor`` parameter since it would be completely degenerate with the other two,
and would be basically impossible to fit in a sensible way.


Sigmoid models
--------------

Sigmoid models are location-scale models defined in terms of a standardized
shape function :math:`g(z)`

.. math::
    f(x; A, m, s, \ldots) = A g\left(\frac{x - m}{s}; \ldots \right)

where :math:`A` is the amplitude (total height of the sigmoid), :math:`m` is the
location (typically the point where the value of the function if 50% of the
amplitude) and :math:`s` is the scale parameter, representing the width of the
transtion.

.. note::

   In this case the amplitude parameter does not represent an area (as in peak-like
   models), but rather the total increase of the function from its lower asymptote
   to its upper asymptote.

   Note when the scale parameter is negative, we switch to the complement of the
   sigmoid function, i.e., a monotonically decreasing function from 1 to 0
   (in standard form).

:math:`g(z)` is generally a monotonically increasing function,
ranging from 0 to 1 as its argument goes from -infinity to +infinity, as illustrated
below for some of the models available in the package.

.. image:: /_static/plots/sigmoid_shapes.png


:class:`~aptapy.models.ErfSigmoid`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cumulative function of a gaussian distribution:

.. math::
    g(z) = \frac{1}{2} \left(1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right)

.. warning::

   The naming might be slightly unfortunate here, as, strictly speaking, this is not
   the error function defined, e.g., in :mod:`scipy.special`, but hopefully it is clear
   enough in the context of model fitting.


:class:`~aptapy.models.LogisticSigmoid`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A logistic sigmoid defined by the standard shape function:

.. math::
    g(z) = \frac{1}{1 + e^{-z}}


:class:`~aptapy.models.Arctangent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An arctangent sigmoid defined by the standard shape function:

.. math::
    g(z) = \frac{1}{2} + \frac{1}{\pi} \arctan(z)

.. literalinclude:: ../src/aptapy/models.py
   :language: python
   :pyobject: Arctangent.shape


:class:`~aptapy.models.HyperbolicTangent`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An hyperbolic tangent sigmoid defined by the standard shape function:

.. math::
    g(z) = \frac{1}{2} \left(1 + \tanh(z)\right)



Continuous random variables
---------------------------

Most of the models defined in this package are wrappers around continuous random
variables defined in :mod:`scipy.stats`, which provide a large variety of
standardized distributions. Most (but not all) of these distributions can be
interpreted as peak-like models. All of them are location-scale models
defined in terms of a standardized shape function :math:`g(z)`

.. math::
    f(x; A, m, s, \ldots) = \frac{A}{s} g\left(\frac{x - m}{s}; \ldots \right)

where :math:`A` is the amplitude (area under the peak), :math:`m` is the location
(parameter specifying the peak position), and :math:`s` is the scale (parameter
specifying the peak width). The trailing ``\ldots`` indicates any additional shape
parameters that might be required by the specific distribution.

.. seealso:: :ref:`modeling`

The rest of this section lists all the available distributions, with a brief
description of their support and shape parameters. For more details on each
distribution, please refer to the corresponding documentation in :mod:`scipy.stats`.


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


:class:`~aptapy.models.Arcsine`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`arcsine`;
support: :math:`0\le z \le 1`.

.. image:: /_static/plots/arcsine_shape.png


:class:`~aptapy.models.Argus`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`argus`;
support: :math:`0 < z < 1`;
shape parameter(s): :math:`\chi > 0`.

.. image:: /_static/plots/argus_shape.png


:class:`~aptapy.models.Beta`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`beta`;
support: :math:`0 < z < 1`;
shape parameter(s): :math:`a> 0`, :math:`b > 0`.

.. image:: /_static/plots/beta_shape.png


:class:`~aptapy.models.BetaPrime`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`betaprime`;
support: :math:`0 < z < 1`;
shape parameter(s): :math:`a> 0`, :math:`b > 0`.

.. image:: /_static/plots/betaprime_shape.png



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


:class:`~aptapy.models.Gibrat`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`gibrat`;
support: :math:`z > 0`.

.. image:: /_static/plots/gibrat_shape.png


:class:`~aptapy.models.GumbelL`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`gumbel_l`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/gumbell_shape.png


:class:`~aptapy.models.GumbelR`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`gumbel_r`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/gumbelr_shape.png


:class:`~aptapy.models.HalfCauchy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`halfcauchy`;
support: :math:`z > 0`.

.. image:: /_static/plots/halfcauchy_shape.png


:class:`~aptapy.models.HalfLogistic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`halflogistic`;
support: :math:`z > 0`.

.. image:: /_static/plots/halflogistic_shape.png


:class:`~aptapy.models.HalfNorm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`halfnorm`;
support: :math:`z > 0`.

.. image:: /_static/plots/halfnorm_shape.png


:class:`~aptapy.models.HyperSecant`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`hypsecant`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/hypersecant_shape.png


:class:`~aptapy.models.Landau`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`landau`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/landau_shape.png


:class:`~aptapy.models.Laplace`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`laplace`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/laplace_shape.png


:class:`~aptapy.models.Levy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`levy`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/levy_shape.png



:class:`~aptapy.models.LevyL`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`levy_l`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/levyl_shape.png


:class:`~aptapy.models.Logistic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`logistic`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/logistic_shape.png


:class:`~aptapy.models.LogNormal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`lognorm`;
support: :math:`z > 0`;
shape parameter(s): :math:`s > 0`.

.. image:: /_static/plots/lognormal_shape.png


:class:`~aptapy.models.Maxwell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`maxwell`;
support: :math:`0 < z < \infty`.

.. image:: /_static/plots/maxwell_shape.png


:class:`~aptapy.models.Moyal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`moyal`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/moyal_shape.png


:class:`~aptapy.models.Nakagami`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`nakagami`;
support: :math:`0 < z < \infty`.

.. image:: /_static/plots/nakagami_shape.png


:class:`~aptapy.models.Rayleigh`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`rayleigh`;
support: :math:`0 < z < \infty`.

.. image:: /_static/plots/rayleigh_shape.png


:class:`~aptapy.models.Semicircular`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`semicircular`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/semicircular_shape.png


:class:`~aptapy.models.Student`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`t`;
support: :math:`-\infty < z < \infty`.

.. image:: /_static/plots/student_shape.png


:class:`~aptapy.models.Wald`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrapped from :scipy_rv_wrap:`wald`;
support: :math:`0 < z < \infty`.

.. image:: /_static/plots/wald_shape.png



Module documentation
--------------------

.. automodule:: aptapy.models
