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
    "Polynomial",
    "Quadratic",
    "Cubic",
    "PowerLaw",
    "Exponential",
    "ExponentialComplement",
    "StretchedExponential",
    "StretchedExponentialComplement",
    "Erf",
    "Logistic",
    "Arctangent",
    "HyperbolicTangent",
]


class Constant(AbstractFitModel):

    """Constant model.
    """

    value = FitParameter(1.)

    @staticmethod
    def evaluate(x: ArrayLike, value: float) -> ArrayLike:
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

    def primitive(self, x: ArrayLike) -> ArrayLike:
        return self.value.value * x


class Line(AbstractFitModel):

    """Linear model.
    """

    slope = FitParameter(1.)
    intercept = FitParameter(0.)

    @staticmethod
    def evaluate(x: ArrayLike, slope: float, intercept: float) -> ArrayLike:
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

    def primitive(self, x: ArrayLike) -> ArrayLike:
        slope, intercept = self.parameter_values()
        return 0.5 * slope * x**2 + intercept * x


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
        super().__init__(label, xlabel, ylabel)
        self.degree = degree
        for i in range(degree + 1):
            setattr(self, f"c{i}", FitParameter(0.))

    @staticmethod
    def evaluate(x: ArrayLike, *coefficients: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        result = np.zeros_like(x)
        degree = len(coefficients) - 1
        for i, c in enumerate(coefficients):
            result += c * x**(degree - i)
        return result

    def primitive(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Analytical primitive not implemented for generic Polynomial.")


class Quadratic(Polynomial):

    """Quadratic model.

    This is just a convenience subclass of the generic Polynomial model with
    degree fixed to 2.
    """

    def __init__(self, label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        super().__init__(degree=2, label=label, xlabel=xlabel, ylabel=ylabel)


class Cubic(Polynomial):

    """Cubic model.

    This is just a convenience subclass of the generic Polynomial model with
    degree fixed to 3.
    """

    def __init__(self, label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        super().__init__(degree=3, label=label, xlabel=xlabel, ylabel=ylabel)


class PowerLaw(AbstractFitModel):

    """Power-law model.
    """

    prefactor = FitParameter(1.)
    index = FitParameter(-2.)

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, index: float) -> ArrayLike:
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

    def primitive(self, x: ArrayLike) -> ArrayLike:
        prefactor, index = self.parameter_values()
        if index == -1.:
            return prefactor * np.log(x)
        return prefactor / (index + 1.) * (x**(index + 1.))

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

    """Exponential model.

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
        super().__init__(label, xlabel, ylabel)
        self.origin = origin

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float) -> ArrayLike:
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

    def primitive(self, x: ArrayLike) -> ArrayLike:
        prefactor, scale = self.parameter_values()
        return prefactor * scale * (np.exp(-(x - self.origin) / scale))

    def default_plotting_range(self, scale_factor: int = 5) -> Tuple[float, float]:
        return (self.origin, self.origin + scale_factor * self.scale.value)


class ExponentialComplement(Exponential):

    """Exponential complement model.
    """

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float) -> ArrayLike:
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

    """Stretched exponential model.
    """

    stretch = FitParameter(1., minimum=0.)

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float, stretch: float) -> ArrayLike:
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

    """Stretched exponential complement model.
    """

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float, stretch: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return prefactor - StretchedExponential.evaluate(self, x, prefactor, scale, stretch)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        See the comment in the corresponding docstrings of the ExponentialComplement class.
        """
        StretchedExponential.init_parameters(self, xdata, ydata.max() - ydata, sigma)


@wrap_rv_continuous(scipy.stats.alpha, plotting_range=(0., 5.))
class Alpha(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.anglit, plotting_range=(-np.pi / 4., np.pi / 4.))
class Anglit(AbstractPeakFitModel):

    pass


# @wrap_rv_continuous(scipy.stats.arcsine)
# class Arcsine(AbstractPeakFitModel):

#     pass


@wrap_rv_continuous(scipy.stats.argus, plotting_range=(0., 1.))
class Argus(AbstractPeakFitModel):

    pass

@wrap_rv_continuous(scipy.stats.beta, plotting_range=(0., 1.))
class Beta(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.betaprime, plotting_range=(0., 6.))
class BetaPrime(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.bradford, plotting_range=(0., 1.))
class Bradford(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.burr, plotting_range=(0., 5.))
class Burr(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.burr12, plotting_range=(0., 5.))
class Burr12(AbstractPeakFitModel):

    pass



@wrap_rv_continuous(scipy.stats.chi, plotting_range=(0., 5.))
class Chi(AbstractPeakFitModel):

    """Check if df needs to be integer.
    """

    pass


@wrap_rv_continuous(scipy.stats.chi2, plotting_range=(0., 5.))
class Chisquare(AbstractPeakFitModel):

    """Check if df needs to be integer.
    """

    pass


@wrap_rv_continuous(scipy.stats.cosine)
class Cosine(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.crystalball)
class CrystalBall(AbstractPeakFitModel):

    """Note the shape parameter m needs to be > 1.
    """

    pass


@wrap_rv_continuous(scipy.stats.dgamma)
class DoubleGamma(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.fisk, plotting_range=(0., 5.))
class Fisk(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.norm, location_alias="mu", scale_alias="sigma")
class Gaussian(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.genlogistic)
class GeneralizedLogistic(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.gennorm)
class GeneralizedNormal(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.landau, plotting_range=(-3., 10.))
class Landau(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.cauchy)
class Lorentzian(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.lognorm, plotting_range=(0., 7.5))
class LogNormal(AbstractPeakFitModel):

    pass


@wrap_rv_continuous(scipy.stats.moyal, plotting_range=(-4., 10.))
class Moyal(AbstractPeakFitModel):

    pass




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
