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
import uncertainties

from .modeling import AbstractFitModel, AbstractPeakFitModel, AbstractSigmoidFitModel, FitParameter
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
    "GaussianCDF",
    "GaussianCDFComplement",
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


class AbstractGaussian(AbstractFitModel):

    """Common base class for Gaussian-like models.

    This provides a couple of convenience methods that are useful for all the
    models derived from a gaussian (e.g., the gaussian itself, the error function,
    and its inverse). Note that, for the right method to be picked up,
    subclasses should derive from this class *before* deriving from
    AbstractFitModel, so that the method resolution order (MRO) works as expected.

    Note the evaluate() method is not implemented here, which means that the class
    cannot be instantiated directly.
    """

    prefactor = FitParameter(1.)
    mean = FitParameter(0.)
    sigma = FitParameter(1., minimum=0.)

    # A few useful constants.
    _SQRT2 = np.sqrt(2.)
    _NORM_CONSTANT = 1. / np.sqrt(2. * np.pi)
    _SIGMA_TO_FWHM = 2. * np.sqrt(2. * np.log(2.))

    def default_plotting_range(self, num_sigma: int = 5) -> Tuple[float, float]:
        """Convenience function to return a default plotting range for all the
        models derived from a gaussian (e.g., the gaussian itself, the error
        function, and its inverse).

        Arguments
        ---------
        num_sigma : int, optional
            The number of sigmas to use for the plotting range (default 5).

        Returns
        -------
        Tuple[float, float]
            The default plotting range for the model.
        """
        # pylint: disable=no-member
        mean, half_width = self.mean.value, num_sigma * self.sigma.value
        return (mean - half_width, mean + half_width)

    def fwhm(self) -> uncertainties.ufloat:
        """Return the full-width at half-maximum (FWHM) of the gaussian.

        Returns
        -------
        fwhm : uncertainties.ufloat
            The FWHM of the gaussian.
        """
        # pylint: disable=no-member
        return self.sigma.ufloat() * self._SIGMA_TO_FWHM


class Gaussian(AbstractGaussian):

    r"""Gaussian model.

    .. math::

        f(x) = \frac{N}{\sigma \sqrt{2 \pi}}
        \exp\left\{-\frac{(x - \mu)^2}{2 \sigma^2}\right\}
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        \mu \rightarrow \texttt{mean}\\
        \sigma \rightarrow \texttt{sigma}
        \end{cases}
    """

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, mean: float, sigma: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        z = (x - mean) / sigma
        return prefactor * AbstractGaussian._NORM_CONSTANT / sigma * np.exp(-0.5 * z**2.)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.
        """
        delta = np.diff(xdata)
        delta = np.append(delta, delta[-1])
        prefactor = (delta * ydata).sum()
        mean = np.average(xdata, weights=ydata)
        variance = np.average((xdata - mean)**2., weights=ydata)
        self.prefactor.init(prefactor)
        self.mean.init(mean)
        self.sigma.init(np.sqrt(variance))

    def integral(self, x1: float, x2: float) -> float:
        """Overloaded method with the analytical integral.
        """
        prefactor, mean, sigma = self.parameter_values()
        zmin = (x1 - mean) / (sigma * self._SQRT2)
        zmax = (x2 - mean) / (sigma * self._SQRT2)
        return prefactor * 0.5 * (scipy.special.erf(zmax) - scipy.special.erf(zmin))


class GaussianCDF(AbstractGaussian):

    r"""Gaussian cumulative distribution function (CDF) model.

    .. math::

        f(x) = \frac{N}{2} \left [ 1 + \text{erf}
        \left\{ \frac{(x - \mu)}{\sigma \sqrt{2}} \right\} \right ]
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        \mu \rightarrow \texttt{mean}\\
        \sigma \rightarrow \texttt{sigma}
        \end{cases}
    """

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, mean: float, sigma: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        z = (x - mean) / sigma
        return prefactor * 0.5 * (1. + scipy.special.erf(z / AbstractGaussian._SQRT2))


class GaussianCDFComplement(AbstractGaussian):

    r"""Complement of the gaussian cumulative distribution function (CDF) model.

    .. math::

        f(x) = \frac{N}{2} \left [ 1 - \text{erf}
        \left\{ \frac{(x - \mu)}{\sigma \sqrt{2}} \right\} \right ]
        \quad \text{with} \quad
        \begin{cases}
        N \rightarrow \texttt{prefactor}\\
        \mu \rightarrow \texttt{mean}\\
        \sigma \rightarrow \texttt{sigma}
        \end{cases}
    """

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, mean: float, sigma: float) -> ArrayLike:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return prefactor - GaussianCDF.evaluate(x, prefactor, mean, sigma)


class Gaussian2(AbstractPeakFitModel):

    """Alternative Gaussian model.
    """

    @staticmethod
    def shape(z):
        """Overloaded method.
        """
        return 1. / np.sqrt(2. * np.pi) * np.exp(-0.5 * z**2.)


class Lorentzian(AbstractPeakFitModel):

    """Lorentzian model.
    """

    @staticmethod
    def shape(z):
        """Overloaded method.
        """
        return 1. / np.pi / (1.0 + z**2)
