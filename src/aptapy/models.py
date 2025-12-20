# Copyright 2023--2025 the aptapy team
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
from typing import Tuple, Union

import matplotlib
import numpy as np
import scipy.integrate
import scipy.special
import scipy.stats

from .hist import Histogram1d
from .modeling import (
    SIGMA_TO_FWHM,
    AbstractCRVFitModel,
    AbstractFitModel,
    AbstractSigmoidFitModel,
    FitParameter,
    FitStatus,
    GaussianForestBase,
    PhonyCRVFitModel,
    line_forest,
    wrap_rv_continuous,
)
from .plotting import last_line_color, plt
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
    "Gaussian",
    "Fe55Forest",
    "Probit",
    "ErfSigmoid",
    "LogisticSigmoid",
    "Arctangent",
    "HyperbolicTangent",
    "Alpha",
    "Anglit",
    "Arcsine",
    "Argus",
    "Beta",
    "BetaPrime",
    "Bradford",
    "Burr",
    "Burr12",
    "Cauchy",
    "Chi",
    "Chisquare",
    "Cosine",
    "CrystalBall",
    "Gibrat",
    "GumbelL",
    "GumbelR",
    "HalfCauchy",
    "HalfLogistic",
    "HalfNorm",
    "HyperSecant",
    "Landau",
    "Laplace",
    "Levy",
    "LevyL",
    "Logistic",
    "LogNormal",
    "Lorentzian",
    "Maxwell",
    "Moyal",
    "Nakagami",
    "Normal",
    "Rayleigh",
    "Semicircular",
    "Student",
    "Wald",
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
        # Initialize the fit parameters for the class. Note that
        # 1. we have to do this *before* we call the constructor of the base class,
        #    as all the class attributes need to be in place when that is called;
        # 2. we have to set the attributes on the class itself, not on the instance,
        #    otherwise they would not be recognized as fit parameters.
        self.degree = degree
        for i in range(degree + 1):
            setattr(self.__class__, f"c{degree - i}", FitParameter(0.))
        super().__init__(label, xlabel, ylabel)

    @staticmethod
    def evaluate(x: ArrayLike, *coefficients: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        result = np.zeros_like(x)
        degree = len(coefficients) - 1
        for i, c in enumerate(coefficients):
            result += c * x**(degree - i)
        return result

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is using a weighted linear regression.

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
        # Build the Vandermonde matrix.
        V = np.vander(xdata, N=self.degree + 1, increasing=False)
        W = np.diag(weights)
        VTW = V.T @ W
        A = VTW @ V
        b = VTW @ ydata
        coeffs = np.linalg.solve(A, b)
        for i, param in enumerate(self):
            param.init(coeffs[i])

    @staticmethod
    def primitive(x: ArrayLike, *coefficients: float) -> ArrayLike:
        raise NotImplementedError("Analytical primitive not implemented for generic Polynomial.")


class Quadratic(Polynomial):

    """Quadratic model.

    This is just a convenience subclass of the generic Polynomial model with
    degree fixed to 2.
    """

    def __init__(self, label: str = None, xlabel: str = None, ylabel: str = None) -> None:
        super().__init__(degree=2, label=label, xlabel=xlabel, ylabel=ylabel)


class Cubic(Polynomial):

    """Cubic model.

    This is just a convenience subclass of the generic Polynomial model with
    degree fixed to 3.
    """

    def __init__(self, label: str = None, xlabel: str = None, ylabel: str = None) -> None:
        super().__init__(degree=3, label=label, xlabel=xlabel, ylabel=ylabel)


class PowerLaw(AbstractFitModel):

    """Power-law model.

    Arguments
    ---------
    pivot : float, optional
        The pivot point of the power-law (default 1.).

    label : str, optional
        The model label.

    xlabel : str, optional
        The label for the x axis.

    ylabel : str, optional
        The label for the y axis.
    """

    prefactor = FitParameter(1.)
    index = FitParameter(-2.)

    def __init__(self, pivot: float = 1., label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        super().__init__(label, xlabel, ylabel)
        self.pivot = pivot

    def evaluate(self, x: ArrayLike, prefactor: float, index: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return prefactor * (x / self.pivot)**index

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
            self.prefactor.init(np.exp(Y0 - self.index.value * X0) * self.pivot**self.index.value)

    def primitive(self, x: ArrayLike) -> ArrayLike:
        """Overloaded method.
        """
        prefactor, index = self.parameter_values()
        if index == -1.:
            return prefactor * self.pivot * np.log(x)
        return prefactor * self.pivot / (index + 1.) * ((x / self.pivot)**(index + 1.))

    def default_plotting_range(self) -> Tuple[float, float]:
        """Overloaded method.

        We might be smarter here, but for now we just return a fixed range that is
        not bogus when the index is negative, which should cover the most common
        use cases.
        """
        # pylint: disable=arguments-differ
        return (0.1 * self.pivot, 10. * self.pivot)

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
    is not a static method, as we have a ``location`` attribute that needs to be
    taken into account. This is done in the spirit of facilitating fits where
    the exponential decay starts at a non-zero x value.

    (One might argue that ``location`` should be a fit parameter as well, but that
    would be degenerate with the ``scale`` parameter, and it would have to be
    fixed in most cases anyway, so a simple attribute seems more appropriate here.)

    Arguments
    ---------
    location : float, optional
        The location of the exponential decay (default 0.).

    label : str, optional
        The model label.

    xlabel : str, optional
        The label for the x axis.

    ylabel : str, optional
        The label for the y axis.
    """

    prefactor = FitParameter(1.)
    scale = FitParameter(1., minimum=0.)

    def __init__(self, location: float = 0., label: str = None, xlabel: str = None,
                 ylabel: str = None) -> None:
        super().__init__(label, xlabel, ylabel)
        self.location = location

    def evaluate(self, x: ArrayLike, prefactor: float, scale: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        x = x - self.location
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
        X = xdata - self.location
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
        return -prefactor * scale * (np.exp(-(x - self.location) / scale))

    def default_plotting_range(self, scale_factor: int = 5) -> Tuple[float, float]:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return (self.location, self.location + scale_factor * self.scale.value)


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
        x = x - self.location
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

    def default_plotting_range(self, scale_factor: int = 5) -> Tuple[float, float]:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return (
            self.location,
            self.location + scale_factor * self.scale.value / self.stretch.value**1.5
        )


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


class Gaussian(AbstractFitModel):

    """This is a re-implementation from scratch of the normal distribution, which is
    also available as a wrapper around scipy.stats.norm as Normal.

    The main reason for this is to be able to call the underlying parameters with
    more familiar names (mu, sigma, rather than loc, scale), as well as
    providing additional convenience methods such the iterative fitting around
    the peak.
    """

    amplitude = FitParameter(1.)
    mu = FitParameter(0.)
    sigma = FitParameter(1., minimum=0.)

    @staticmethod
    def evaluate(x, amplitude, mu, sigma, *args):
        # pylint: disable=arguments-differ
        return amplitude * scipy.stats.norm.pdf(x, *args, loc=mu, scale=sigma)

    @staticmethod
    def primitive(x, amplitude, mu, sigma, *args):
        return amplitude * scipy.stats.norm.cdf(x, *args, loc=mu, scale=sigma)

    def median(self):
        return self.mu.value

    def mean(self):
        return self.mu.value

    def std(self):
        return self.sigma.value

    def fwhm(self):
        """Calculate the FWHM.
        """
        return SIGMA_TO_FWHM * self.sigma.ufloat()

    def rvs(self, size: int = 1, random_state=None):
        """Generate random variates from the underlying distribution at the current
        parameter values.

        Arguments
        ---------
        size : int, optional
            The number of random variates to generate (default 1).

        random_state : int or np.random.Generator, optional
            The random seed or generator to use (default None).
        """
        return scipy.stats.norm.rvs(loc=self.mu.value, scale=self.sigma.value,
                                    size=size, random_state=random_state)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.
        """
        self.amplitude.init(scipy.integrate.trapezoid(ydata, xdata))
        self.mu.init(np.average(xdata, weights=ydata))
        self.sigma.init(np.sqrt(np.average((xdata - self.mu.value)**2, weights=ydata)))

    def fit_iterative(self, xdata: Union[ArrayLike, Histogram1d], ydata: ArrayLike = None, *,
            p0: ArrayLike = None, sigma: ArrayLike = None, num_sigma_left: float = 2.,
            num_sigma_right: float = 2., num_iterations: int = 2, **kwargs) -> "FitStatus":
        """Fit the core of Gaussian data within a given number of sigma around the peak.

        This function performs a first round of fit to the data (either a histogram or
        scatter plot data) and then repeats the fit iteratively, limiting the fit range
        to a specified interval defined in terms of deviations (in sigma) around the peak.

        Arguments
        ----------
        xdata : array_like or Histogram1d
            The data (scatter plot x values) or histogram to fit.

        ydata : array_like, optional
            The y data to fit (if xdata is not a Histogram1d).

        p0 : array_like, optional
            The initial values for the fit parameters.

        sigma : array_like, optional
            The uncertainties on the y data.

        num_sigma_left : float
            The number of sigma on the left of the peak to be used to define the
            fitting range.

        num_sigma_right : float
            The number of sigma on the right of the peak to be used to define the
            fitting range.

        num_iterations : int
            The number of iterations of the fit.

        kwargs : dict, optional
            Additional keyword arguments passed to `fit()`.

        Returns
        -------
        FitStatus
            The results of the fit.
        """
        fit_status = self.fit(xdata, ydata, p0=p0, sigma=sigma, **kwargs)
        for i in range(num_iterations):
            kwargs.update(xmin=self.mean() - num_sigma_left * self.std(),
                          xmax=self.mean() + num_sigma_right * self.std())
            try:
                fit_status = self.fit(xdata, ydata, p0=self.free_parameter_values(),
                                      sigma=sigma, **kwargs)
            except RuntimeError as exception:
                raise RuntimeError(f"Exception after {i + 1} iteration(s)") from exception
        return fit_status

    def default_plotting_range(self) -> Tuple[float, float]:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        return (self.mu.value - 5. * self.sigma.value, self.mu.value + 5. * self.sigma.value)

    def plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False,
             plot_mean: bool = True, **kwargs) -> None:
        """Plot the model.

        Note this is reimplemented from scratch to allow overplotting the mean of the
        distribution.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on (default: current axes).

        fit_output : bool, optional
            Whether to include the fit output in the legend (default: False).

        plot_mean : bool, optional
            Whether to overplot the mean of the distribution (default: True).

        kwargs : dict, optional
            Additional keyword arguments passed to `plt.plot()`.
        """
        super().plot(axes, fit_output=fit_output, **kwargs)
        if plot_mean:
            if axes is None:
                axes = plt.gca()
            color = last_line_color()
            x0 = self.mean()
            y0 = self(x0)
            axes.plot(x0, y0, "o", ms=5., color=matplotlib.rcParams["figure.facecolor"])
            axes.plot(x0, y0, "o", ms=1.5, color=color)


@line_forest(5.896, 6.492)
class Fe55Forest(GaussianForestBase):
    """Model representing the Kα and Kβ emission lines produced in the decay
    of 55Fe. The energy values are computed as the intensity-weighted mean of
    all possible emission lines contributing to each feature.

    The energy data are retrieved from the X-ray database at https://xraydb.seescience.org/
    """

    # https://xraydb.xrayabsorption.org/element/Mn
    # This is the sum of the intensities of Kb1, Kb3 and Kb5 lines.
    TABULATED_KB_INTENSITY = 0.12445

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.
        """
        # pylint: disable=no-member
        mu0 = xdata[np.argmax(ydata)]
        self.amplitude.init(scipy.integrate.trapezoid(ydata, xdata))
        self.intensity1.init(self.TABULATED_KB_INTENSITY)
        self.energy_scale.init(self.energies[0] / mu0)
        self.sigma.init(np.sqrt(np.average((xdata - mu0)**2, weights=ydata)))


class Probit(AbstractFitModel):

    """Custom implementation of the probit model, i.e., the percent-point function
    of a gaussian distribution.
    """

    offset = FitParameter(0.)
    sigma = FitParameter(1., minimum=0.)

    @staticmethod
    def evaluate(x: ArrayLike, offset: float, sigma: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return offset + sigma * scipy.special.ndtri(x)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.
        """
        # pylint: disable=no-member
        xmin, xmax = np.min(xdata), np.max(xdata)
        delta = np.max(ydata) - np.min(ydata)
        self.offset.init(np.mean(ydata))
        self.sigma.init(delta / (scipy.stats.norm.ppf(xmax) - scipy.stats.norm.ppf(xmin)))

    @staticmethod
    def default_plotting_range() -> Tuple[float, float]:
        """Overloaded method.

        Since the probit function diverges at 0 and 1, we limit the plotting range
        to a reasonable interval.
        """
        return (0.001, 0.999)


class ErfSigmoid(AbstractSigmoidFitModel):

    """Error function model.
    """

    @staticmethod
    def shape(z):
        # pylint: disable=arguments-differ
        return 0.5 * (1. + scipy.special.erf(z / np.sqrt(2.)))


class LogisticSigmoid(AbstractSigmoidFitModel):

    """Logistic function model.
    """

    @staticmethod
    def shape(z):
        # pylint: disable=arguments-differ
        return 1. / (1. + np.exp(-z))


class Arctangent(AbstractSigmoidFitModel):

    """Arctangent function model.
    """

    @staticmethod
    def shape(z):
        # pylint: disable=arguments-differ
        return 0.5 + np.arctan(z) / np.pi


class HyperbolicTangent(AbstractSigmoidFitModel):

    """Hyperbolic tangent function model.
    """

    @staticmethod
    def shape(z):
        # pylint: disable=arguments-differ
        return 0.5 * (1. + np.tanh(z))


@wrap_rv_continuous(scipy.stats.alpha)
class Alpha(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.anglit)
class Anglit(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.arcsine)
class Arcsine(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.argus)
class Argus(AbstractCRVFitModel):

    pass

@wrap_rv_continuous(scipy.stats.beta)
class Beta(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.betaprime)
class BetaPrime(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.bradford)
class Bradford(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.burr)
class Burr(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.burr12)
class Burr12(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.cauchy)
class Cauchy(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.chi)
class Chi(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.chi2)
class Chisquare(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.cosine)
class Cosine(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.crystalball, m=FitParameter(2., minimum=1.))
class CrystalBall(AbstractCRVFitModel):

    pass


# The Gibrat distribution is only available starting from scipy 1.12.0
try:
    @wrap_rv_continuous(scipy.stats.gibrat)
    class Gibrat(AbstractCRVFitModel):

        pass

except AttributeError:
    class Gibrat(PhonyCRVFitModel):

        def __init__(self, *args, **kwargs) -> None:
            # pylint: disable=unused-argument
            super().__init__("1.12.0")


@wrap_rv_continuous(scipy.stats.gumbel_l)
class GumbelL(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.gumbel_r)
class GumbelR(AbstractCRVFitModel):

    pass



@wrap_rv_continuous(scipy.stats.halfcauchy)
class HalfCauchy(AbstractCRVFitModel):

    pass



@wrap_rv_continuous(scipy.stats.halflogistic)
class HalfLogistic(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.halfnorm)
class HalfNorm(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.hypsecant)
class HyperSecant(AbstractCRVFitModel):

    pass


# The Landau distribution is only available starting from scipy 1.15.1
try:
    @wrap_rv_continuous(scipy.stats.landau)
    class Landau(AbstractCRVFitModel):

        def default_plotting_range(self) -> Tuple[float, float]:
            """Overloaded method.

            The Landau distribution is peculiar in that it has no definite mean or variance,
            and its support is unbounded. It is also asymmetric, with a long right tail.
            Therefore, we resort to a custom function for the plotting range.
            """
            # pylint: disable=arguments-differ
            location, scale = self.location.value, self.scale.value
            return (location - 2.5 * scale, location + 12.5 * scale)

except AttributeError:
    class Landau(PhonyCRVFitModel):

        def __init__(self, *args, **kwargs) -> None:
            # pylint: disable=unused-argument
            super().__init__("1.15.1")


@wrap_rv_continuous(scipy.stats.laplace)
class Laplace(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.levy)
class Levy(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.levy_l)
class LevyL(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.logistic)
class Logistic(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.lognorm)
class LogNormal(AbstractCRVFitModel):

    pass


class Lorentzian(Cauchy):

    """Alias for the Cauchy distribution.
    """


@wrap_rv_continuous(scipy.stats.maxwell)
class Maxwell(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.moyal)
class Moyal(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.nakagami)
class Nakagami(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.norm)
class Normal(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.rayleigh)
class Rayleigh(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.semicircular)
class Semicircular(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.t)
class Student(AbstractCRVFitModel):

    pass


@wrap_rv_continuous(scipy.stats.wald)
class Wald(AbstractCRVFitModel):

    pass
