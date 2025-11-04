# Copyright (C) 2025 Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Built in models.
"""

from numbers import Number
from typing import Tuple

import matplotlib
import numpy as np
import scipy.special
import scipy.stats
import uncertainties

from .modeling import AbstractFitModel, AbstractPeakFitModel, AbstractSigmoidFitModel,\
    FitParameter, wrap_rv_continuous
from .plotting import plt
from .typing_ import ArrayLike

__all__ = [
    "Constant",
    "Line",
    "Quadratic",
    "PowerLaw",
    "Exponential",
    "ExponentialComplement",
    "StretchedExponential",
    "StretchedExponentialComplement",
    "Gaussian",
    "Lorentzian",
    "LogNormal",
    "Moyal",
    "Erf",
    "Logistic",
    "Arctangent",
    "HyperbolicTangent",
]


class Constant(AbstractFitModel):

    r"""Constant model.

    .. math::

        f(x) = c
        \quad \text{with} \quad
        c \rightarrow \texttt{value}
    """

    value = FitParameter(1.)

    @staticmethod
    def evaluate(x: ArrayLike, value: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        if isinstance(x, Number):
            return value
        return np.full(x.shape, value)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is simply using the weighted average of the y data, using the inverse
        of the squares of the errors as weights.

        .. note::

           This should provide the exact result in most cases, but, in the spirit of
           providing a common interface across all models, we are not overloading the
           fit() method. (Everything will continue working as expected, e.g., when
           one uses bounds on parameters.)
        """
        if isinstance(sigma, Number):
            sigma = np.full(ydata.shape, sigma)
        self.value.init(np.average(ydata, weights=1. / sigma**2.))

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        return self.value.value * (x2 - x1)


class Line(AbstractFitModel):

    r"""Linear model.

    .. math::

        f(x) = mx + q
        \quad \text{with} \quad
        \begin{cases}
        m \rightarrow \texttt{slope} \\
        q \rightarrow \texttt{intercept}
        \end{cases}
    """

    slope = FitParameter(1.)
    intercept = FitParameter(0.)

    @staticmethod
    def evaluate(x: ArrayLike, slope: float, intercept: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return slope * x + intercept

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is simply using a weighted linear regression.

        .. note::

           This should provide the exact result in most cases, but, in the spirit of
           providing a common interface across all models, we are not overloading the
           fit() method. (Everything will continue working as expected, e.g., when
           one uses bounds on parameters.)
        """
        # pylint: disable=invalid-name
        if isinstance(sigma, Number):
            sigma = np.full(ydata.shape, sigma)
        weights = 1. / sigma**2.
        S0x = weights.sum()
        S1x = (weights * xdata).sum()
        S2x = (weights * xdata**2.).sum()
        S0xy = (weights * ydata).sum()
        S1xy = (weights * xdata * ydata).sum()
        D = S0x * S2x - S1x**2.
        if D != 0.:
            self.slope.init((S0x * S1xy - S1x * S0xy) / D)
            self.intercept.init((S2x * S0xy - S1x * S1xy) / D)

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        slope, intercept = self.parameter_values()
        return 0.5 * slope * (x2**2 - x1**2) + intercept * (x2 - x1)


class Polynomial(AbstractFitModel):

    """Generic polynomial model.

    Note that this is a convenience class to be used when one needs polynomials
    of arbitrary degree. For common low-order polynomials, consider using the
    dedicated classes (e.g., Line, Quadratic, etc.), which provide better
    initial parameter estimation.

    Arguments
    ---------
    degree : int
        The degree of the polynomial.

    label : str, optional
        The model label.

    xlabel : str, optional
        The label for the x axis.

    ylabel : str, optional
        The label for the y axis.
    """

    def __init__(self, degree: int, label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        """Constructor.
        """
        super().__init__(label, xlabel, ylabel)
        self.degree = degree
        for i in range(degree + 1):
            setattr(self, f"c{i}", FitParameter(0.))

    @staticmethod
    def evaluate(x: ArrayLike, *coefficients: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        result = np.zeros_like(x)
        degree = len(coefficients) - 1
        for i, c in enumerate(coefficients):
            result += c * x**(degree - i)
        return result


class Quadratic(AbstractFitModel):

    r"""Quadratic model.

    .. math::

        f(x) = ax^2 + bx + c
        \quad \text{with} \quad
        \begin{cases}
        a \rightarrow \texttt{a}\\
        b \rightarrow \texttt{b}\\
        c \rightarrow \texttt{c}
        \end{cases}
    """

    a = FitParameter(1.)
    b = FitParameter(1.)
    c = FitParameter(0.)

    @staticmethod
    def evaluate(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return a * x**2 + b * x + c

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        a, b, c = self.parameter_values()
        return a * (x2**3 - x1**3) / 3. + b * (x2**2 - x1**2) / 2. + c * (x2 - x1)


class PowerLaw(AbstractFitModel):

    r"""Power-law model.

    .. math::

        f(x) = N x^\Gamma
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        \Gamma \rightarrow \texttt{index}
        \end{cases}
    """

    prefactor = FitParameter(1.)
    index = FitParameter(-2.)

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, index: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return prefactor * x**index

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is using a weighted linear regression in log-log space. Note this is
        not an exact solution in the original space, for which a numerical optimization
        using non-linear least squares would be needed.
        """
        # pylint: disable=invalid-name
        mask = np.logical_and(xdata > 0., ydata > 0.)
        xdata = xdata[mask]
        ydata = ydata[mask]
        if isinstance(sigma, np.ndarray):
            sigma = sigma[mask]
        X = np.log(xdata)
        Y = np.log(ydata)
        # Propagate the errors to log space.
        weights = ydata**2. / sigma**2.
        S = weights.sum()
        X0 = (weights * X).sum() / S
        Y0 = (weights * Y).sum() / S
        Sxx = (weights * (X - X0)**2.).sum()
        Sxy = (weights * (X - X0) * (Y - Y0)).sum()
        if Sxx != 0.:
            self.index.init(Sxy / Sxx)
            self.prefactor.init(np.exp(Y0 - self.index.value * X0))

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        prefactor, index = self.parameter_values()
        if index == -1.:
            return prefactor * np.log(x2 / x1)
        return prefactor / (index + 1.) * (x2**(index + 1.) - x1**(index + 1.))

    def default_plotting_range(self) -> Tuple[float, float]:
        """Overloaded method.

        We might be smarter here, but for now we just return a fixed range that is
        not bogus when the index is negative, which should cover the most common
        use cases.
        """
        return (0.1, 10.)

    def plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False, **kwargs) -> None:
        """Overloaded method.

        In addition to the base class implementation, this also sets log scales
        on both axes.
        """
        super().plot(axes, fit_output=fit_output, **kwargs)
        plt.xscale("log")
        plt.yscale("log")


class Exponential(AbstractFitModel):

    r"""Exponential model.

    .. math::

        f(x) = N \exp \left\{-\frac{(x - x_0)}{X}\right\}
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        X \rightarrow \texttt{scale}\\
        x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
        \end{cases}

    Note this is an example of a model with a state, i.e., one where ``evaluate()``
    is not a static method, as we have an ``origin`` attribute that needs to be
    taken into account. This is done in the spirit of facilitating fits where
    the exponential decay starts at a non-zero x value.

    (One might argue that ``origin`` should be a fit parameter as well, but that
    would be degenerate with the ``scale`` parameter, and it would have to be
    fixed in most cases anyway, so a simple attribute seems more appropriate here.)

    Arguments
    ---------
    origin : float, optional
        The origin of the exponential decay (default 0.).

    label : str, optional
        The model label.

    xlabel : str, optional
        The label for the x axis.

    ylabel : str, optional
        The label for the y axis.
    """

    prefactor = FitParameter(1.)
    scale = FitParameter(1.)

    def __init__(self, origin: float = 0., label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        """Constructor.
        """
        super().__init__(label, xlabel, ylabel)
        self.origin = origin

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        x = x - self.origin
        return prefactor * np.exp(-x / scale)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is using a weighted linear regression in lin-log space. Note this is
        not an exact solution in the original space, for which a numerical optimization
        using non-linear least squares would be needed.
        """
        # pylint: disable=invalid-name
        # Filter out non-positive ydata values, as we shall take the logarithm.
        mask = ydata > 0.
        xdata = xdata[mask]
        ydata = ydata[mask]
        if isinstance(sigma, np.ndarray):
            sigma = sigma[mask]
        X = xdata - self.origin
        Y = np.log(ydata)
        # Propagate the errors to log space.
        weights = ydata**2. / sigma**2.
        S = weights.sum()
        X0 = (weights * X).sum() / S
        Y0 = (weights * Y).sum() / S
        Sxx = (weights * (X - X0)**2.).sum()
        Sxy = (weights * (X - X0) * (Y - Y0)).sum()
        if Sxx != 0.:
            b = -Sxy / Sxx
            self.prefactor.init(np.exp(Y0 + b * X0))
            if not np.isclose(b, 0.):
                self.scale.init(1. / b)

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        prefactor, scale = self.parameter_values()
        x1 = x1 - self.origin
        x2 = x2 - self.origin
        return prefactor * scale * (np.exp(-x1 / scale) - np.exp(-x2 / scale))

    def default_plotting_range(self, scale_factor: int = 5) -> Tuple[float, float]:
        """Overloaded method.
        """
        return (self.origin, self.origin + scale_factor * self.scale.value)


class ExponentialComplement(Exponential):

    r"""Exponential complement model.

    .. math::

        f(x) = N \left [ 1- \exp\left\{-\frac{(x - x_0)}{X}\right\} \right ]
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        X \rightarrow \texttt{scale}\\
        x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
        \end{cases}
    """

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return prefactor - Exponential.evaluate(self, x, prefactor, scale)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        Note we just pretend that the maximum of the y values is a reasonable estimate
        of the prefactor, and go back to the plain exponential case via the
        transformation ydata -> prefactor - ydata.
        """
        Exponential.init_parameters(self, xdata, ydata.max() - ydata, sigma)


class StretchedExponential(Exponential):

    r"""Stretched exponential model.

    .. math::

        f(x) = N \exp \left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\}
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        X \rightarrow \texttt{scale}\\
        \gamma \rightarrow \texttt{stretch}\\
        x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
        \end{cases}
    """

    stretch = FitParameter(1., minimum=0.)

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float, stretch: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        x = x - self.origin
        return prefactor * np.exp(-(x / scale)**stretch)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.):
        """Overloaded method.

        Note this a little bit flaky, in that we pretend that the data are well
        approximated by a plain exponential, and do not even try at estimating the
        stretch factor. When the latter is significantly different from 1 this will
        not be very accurate, but hopefully good enough to get the fit started.
        """
        Exponential.init_parameters(self, xdata, ydata, sigma)
        self.stretch.init(1.)


class StretchedExponentialComplement(StretchedExponential):

    r"""Stretched exponential complement model.

    .. math::

        f(x) = N \left [ 1- \exp\left\{-\left[\frac{(x - x_0)}{X}\right]^\gamma\right\} \right ]
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        X \rightarrow \texttt{scale}\\
        \gamma \rightarrow \texttt{stretch}\\
        x_0 \rightarrow \texttt{origin}~\text{(not a parameter)}
        \end{cases}
    """

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float, stretch: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return prefactor - StretchedExponential.evaluate(self, x, prefactor, scale, stretch)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        See the comment in the corresponding docstrings of the ExponentialComplement
        class.
        """
        StretchedExponential.init_parameters(self, xdata, ydata.max() - ydata, sigma)


@wrap_rv_continuous(scipy.stats.norm)
class Gaussian(AbstractPeakFitModel):

    """Gaussian model.
    """

    def fwhm(self) -> float:
        return 2. * np.sqrt(2. * np.log(2.)) * self.scale.value


@wrap_rv_continuous(scipy.stats.cauchy)
class Lorentzian(AbstractPeakFitModel):

    """Lorentzian model.
    """

    def fwhm(self) -> float:
        return 2. * self.scale.value


class LogNormal(AbstractPeakFitModel):

    """Log-normal model.
    """

    @staticmethod
    def shape(z):
        """Overloaded method.

        Note the shape function is only defined for positive z values, which
        requires a little bit of extra work for making sure that we are returning
        0 for z <= 0 without causing any RuntimeWarning due to zero-division
        errors and/or invalid inputs to the logarithm calculation.
        """
        z = np.asarray(z, dtype=float)
        val = np.zeros_like(z, dtype=float)
        mask = z > 0
        z = z[mask]
        val[mask] = 1. / (z * np.sqrt(2. * np.pi)) * np.exp(-0.5 * np.log(z)**2.)
        return val

    def evaluate(self, x: ArrayLike, amplitude: float, location: float,
                 scale: float) -> ArrayLike:
        """Overloaded method.
        """
        return super().evaluate(x, amplitude, location, scale) / scale

    def fwhm(self) -> float:
        return 2. * np.sinh(np.sqrt(2. * np.log(2.))) * np.exp(self.location.value - self.scale.value**2.)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.):
        """Overloaded method.

        Since the localtion for the Log-normal distribution is the leftmost edge
        of the distribution support, calling the base class implementation would
        necessarily oversertimate the location parameter by an amount of the
        order of the scale parameter. We thus adjust the initial location estimate
        accordingly.
        """
        super().init_parameters(xdata, ydata, sigma)
        self.location.init(self.location.value - self.scale.value)

    def default_plotting_range(self) -> Tuple[float, float]:
        return super().default_plotting_range((0., 7.5))


@wrap_rv_continuous(scipy.stats.moyal)
class Moyal(AbstractPeakFitModel):

    """Moyal model.
    """

    def fwhm(self) -> float:
        """Overloaded method.

        The underlying equation is trancendental, so we need to resort to a
        numerical solution.
        """
        return 3.5632 * self.scale.value

    def default_plotting_range(self) -> Tuple[float, float]:
        return super().default_plotting_range((5., 10.))


class Erf(AbstractSigmoidFitModel):

    """Error function model.
    """

    @staticmethod
    def shape(z):
        return 0.5 * (1. + scipy.special.erf(z / np.sqrt(2.)))


class Logistic(AbstractSigmoidFitModel):

    """Logistic function model.
    """

    @staticmethod
    def shape(z):
        return 1. / (1. + np.exp(-z))


class Arctangent(AbstractSigmoidFitModel):

    """Arctangent function model.
    """

    @staticmethod
    def shape(z):
        return 0.5 + np.arctan(z) / np.pi


class HyperbolicTangent(AbstractSigmoidFitModel):

    """Hyperbolic tangent function model.
    """

    @staticmethod
    def shape(z):
        return 0.5 * (1. + np.tanh(z))
