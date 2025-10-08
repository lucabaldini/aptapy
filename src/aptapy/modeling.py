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

"""Modeling facilities.
"""

import enum
import functools
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from numbers import Number
from typing import Callable, Iterator, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from scipy.optimize import curve_fit
from scipy.stats import chi2

from .hist import Histogram1d
from .typing_ import ArrayLike


class Format(str, enum.Enum):

    """Small enum class to control string formatting.

    This is leveraging the custom formatting of the uncertainties package, where
    a trailing `P` means "pretty print" and a trailing `L` means "LaTeX".
    """

    PRETTY = "P"
    LATEX = "L"


@dataclass
class FitParameter:

    """Small class describing a fit parameter.
    """

    value: float
    _name: str = None
    error: float = None
    _frozen: bool = False
    minimum: float = -np.inf
    maximum: float = np.inf

    @property
    def name(self) -> str:
        """Return the parameter name.

        We are wrapping this into a property because, arguably, the parameter name is
        the only thing we never, ever want to change after the fact.

        Returns
        -------
        name : str
            The parameter name.
        """
        return self._name

    @property
    def frozen(self) -> bool:
        """Return True if the parameter is frozen.

        We are wrapping this into a property because we interact with this member
        via the freeze() and thaw() methods.

        Returns
        -------
        frozen : bool
            True if the parameter is frozen.
        """
        return self._frozen

    def is_bound(self) -> bool:
        """Return True if the parameter is bounded.

        Returns
        -------
        bounded : bool
            True if the parameter is bounded.
        """
        return not np.isinf(self.minimum) or not np.isinf(self.maximum)

    def copy(self, name: str) -> "FitParameter":
        """Create a copy of the parameter object with a new name.

        This is necessary because we define the fit parameters of the actual model as
        class variables holding the default value, and each instance gets their own
        copy of the parameter, where the name is automatically inferred.

        Note that, in addition to the name being passed as an argument, we only carry
        over the value and bounds of the original fit parameter: the new object is
        created with error = None and _frozen = False.

        Arguments
        ---------
        name : str
            The name for the new :class:`FitParameter` object.

        Returns
        -------
        parameter : FitParameter
            The new :class:`FitParameter` object.
        """
        return self.__class__(self.value, name, minimum=self.minimum, maximum=self.maximum)

    def set(self, value: float, error: float = None) -> None:
        """Set the parameter value and error.

        Arguments
        ---------
        value : float
            The new value for the parameter.

        error : float, optional
            The new error for the parameter (default None).
        """
        self.value = value
        self.error = error

    def freeze(self, value: float) -> None:
        """Freeze the fit parameter to a given value.

        Note that the error is set to None.

        Arguments
        ---------
        value : float
            The new value for the parameter.
        """
        self.set(value)
        self._frozen = True

    def thaw(self) -> None:
        """Un-freeze the fit parameter.
        """
        self._frozen = False

    def ufloat(self) -> uncertainties.ufloat:
        """Return the parameter value and error as a ufloat object.

        Returns
        -------
        ufloat : uncertainties.ufloat
            The parameter value and error as a ufloat object.
        """
        return uncertainties.ufloat(self.value, self.error)

    def pull(self, expected: float) -> float:
        """Calculate the pull of the parameter with respect to a given expected value.

        Arguments
        ---------
        expected : float
            The expected value for the parameter.

        Returns
        -------
        pull : float
            The pull of the parameter with respect to the expected value, defined as
            (value - expected) / error.

        Raises
        ------
        RuntimeError
            If the parameter has no error associated to it.
        """
        if self.error is None or self.error <= 0.:
            raise RuntimeError(f"Cannot calculate pull for parameter {self.name} "
                               "with no error")
        return (self.value - expected) / self.error

    def compatible_with(self, expected: float, num_sigma: float = 3.) -> bool:
        """Check if the parameter is compatible with an expected value within
        n_sigma.

        Arguments
        ---------
        expected : float
            The expected value for the parameter.

        num_sigma : float, optional
            The number of sigmas to use for the compatibility check (default 3).

        Returns
        -------
        compatible : bool
            True if the parameter is compatible with the expected value within
            num_sigma.
        """
        return abs(self.pull(expected)) <= num_sigma

    def __format__(self, spec: str) -> str:
        """String formatting.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        # Keep in mind Python passes an empty string explicitly when you call
        # f"{parameter}", so we can't really assign a default value to spec.
        if self.error is not None:
            param = format(self.ufloat(), spec)
            if spec.endswith(Format.LATEX):
                param = f"${param}$"
        else:
            # Note in this case we are not passing the format spec to format(), as
            # the only thing we can do in absence of an error is to use the
            # Python default formatting.
            param = format(self.value, "g")
        text = f"{self._name.title()}: {param}"
        info = []
        if self._frozen:
            info.append("frozen")
        if not np.isinf(self.minimum):
            info.append(f"min={self.minimum}")
        if not np.isinf(self.maximum):
            info.append(f"max={self.maximum}")
        if info:
            text = f"{text} ({', '.join(info)})"
        return text

    def __str__(self) -> str:
        """String formatting.

        This is meant to provide a more human-readable version of the parameter formatting
        than the default ``__repr__`` implementation from the dataclass decorator, and it
        is what is used in the actual printout of the fit parameters from a fit.

        Returns
        -------
        text : str
            The formatted string.
        """
        return format(self, Format.PRETTY)


@dataclass
class FitStatus:

    """Small dataclass to hold the fit status.
    """

    chisquare: float = None
    dof: int = None
    pvalue: float = None
    fit_range: Tuple[float, float] = None

    def reset(self) -> None:
        """Reset the fit status.
        """
        self.chisquare = None
        self.dof = None
        self.pvalue = None
        self.fit_range = None

    def update(self, chisquare: float, dof: int = None) -> None:
        """Update the fit status, i.e., set the chisquare and calculate the
        corresponding p-value.

        Arguments
        ---------
        chisquare : float
            The chisquare of the fit.

        dof : int, optional
            The number of degrees of freedom of the fit.
        """
        self.chisquare = chisquare
        if dof is not None:
            self.dof = dof
        self.pvalue = chi2.sf(self.chisquare, self.dof)
        # chi2.sf() returns the survival function, i.e., 1 - cdf. If the survival
        # function is > 0.5, we flip it around, so that we always report the smallest
        # tail, and the pvalue is the probability of obtaining a chisquare value more
        # `extreme` of the one we got.
        if self.pvalue > 0.5:
            self.pvalue = 1. - self.pvalue

    def __format__(self, spec: str) -> str:
        """String formatting.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        if self.chisquare is None:
            return "N/A"
        if spec.endswith(Format.LATEX):
            return f"$\\chi^2$ = {self.chisquare:.2f} / {self.dof} dof"
        if spec.endswith(Format.PRETTY):
            return f"χ² = {self.chisquare:.2f} / {self.dof} dof"
        return f"chisquare = {self.chisquare:.2f} / {self.dof} dof"

    def __str__(self) -> str:
        """String formatting.

        Returns
        -------
        text : str
            The formatted string.
        """
        return format(self, Format.PRETTY)


class AbstractFitModelBase(ABC):

    """Abstract base class for all the fit classes.

    This is a acting a base class for both simple fit models and for composite models
    (e.g., sums of simple ones).
    """

    def __init__(self) -> None:
        """Constructor.
        """
        self.status = FitStatus()

    @abstractmethod
    def __len__(self) -> int:
        """Delegated to concrete classes: this should return the `total` number of
        fit parameters (not only the free ones) in the model.

        .. note::

           I still have mixed feelings about this method, as it is not clear whether
           we are returning the number of parameters, or the number of free parameters,
           but I think it is fine, as long as we document it. Also note that, while
           the number of parameters is fixed once and for all for simple models,
           it can change at runtime for composite models.

        Returns
        -------
        n : int
            The total number of fit parameters in the model.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[FitParameter]:
        """Delegated to concrete classes: this should return an iterator over `all`
        the fit parameters in the model.

        Returns
        -------
        iterator : Iterator[FitParameter]
            An iterator over all the fit parameters in the model.
        """

    @staticmethod
    @abstractmethod
    def evaluate(x: ArrayLike, *parameter_values: Sequence[float]) -> ArrayLike:
        """Evaluate the model at a given set of parameter values.

        Arguments
        ---------
        x : array_like
            The value(s) of the independent variable.

        parameter_values : sequence of float
            The value of the model parameters.

        Returns
        -------
        y : array_like
            The value(s) of the model at the given value(s) of the independent variable
            for a given set of parameter values.
        """

    def name(self) -> str:
        """Return the model name, e.g., for legends.

        Note this can be reimplemented in concrete subclasses, but it should provide
        a sensible default value in most circumstances.

        Returns
        -------
        name : str
            The model name.
        """
        return self.__class__.__name__

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the model at the current value of the parameters.

        Arguments
        ---------
        x : array_like
            The value(s) of the independent variable.

        Returns
        -------
        y : array_like
            The value(s) of the model at the given value(s) of the independent variable
            for the current set of parameter values.
        """
        return self.evaluate(x, *self.parameter_values())

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike) -> None:
        """Optional hook to change the current parameter values of the model, prior
        to a fit, based on the input data.

        Arguments
        ---------
        xdata : array_like
            The input values of the independent variable.

        ydata : array_like
            The input values of the dependent variable.

        sigma : array_like
            The input uncertainties on the dependent variable.
        """
        # pylint: disable=unused-argument
        return

    def parameter_values(self) -> Tuple[float]:
        """Return the current parameter values.

        Note this only relies on the __iter__() method, so it works both for simple
        and composite models.

        Returns
        -------
        values : tuple of float
            The current parameter values.
        """
        return tuple(parameter.value for parameter in self)

    def free_parameters(self) -> Tuple[FitParameter]:
        """Return the list of free parameters.

        Note this only relies on the __iter__() method, so it works both for simple
        and composite models.

        Returns
        -------
        parameters : tuple of FitParameter
            The list of free parameters.
        """
        return tuple(parameter for parameter in self if not parameter.frozen)

    def free_parameter_values(self) -> Tuple[float]:
        """Return the current parameter values.

        Returns
        -------
        values : tuple of float
            The current parameter values.
        """
        return tuple(parameter.value for parameter in self.free_parameters())

    def bounds(self) -> Tuple[ArrayLike, ArrayLike]:
        """Return the bounds on the fit parameters in a form that can be use by the
        fitting method.

        Returns
        -------
        bounds : 2-tuple of array_like
            The lower and upper bounds on the (free) fit parameters.
        """
        free_parameters = self.free_parameters()
        return (tuple(parameter.minimum for parameter in free_parameters),
                tuple(parameter.maximum for parameter in free_parameters))

    def update_parameters(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        """Update the model parameters based on the output of the ``curve_fit()`` call.

        Arguments
        ---------
        popt : array_like
            The optimal values for the fit parameters.

        pcov : array_like
            The covariance matrix for the fit parameters.
        """
        for parameter, value, error in zip(self.free_parameters(), popt, np.sqrt(pcov.diagonal())):
            parameter.value = value
            parameter.error = error

    def calculate_chisquare(self, xdata: np.ndarray, ydata: np.ndarray, sigma) -> float:
        """Calculate the chisquare of the fit to some input data with the current
        model parameters.

        Arguments
        ---------
        xdata : array_like
            The input values of the independent variable.

        ydata : array_like
            The input values of the dependent variable.

        sigma : array_like
            The input uncertainties on the dependent variable.

        Returns
        -------
        chisquare : float
            The chisquare of the fit.
        """
        return float((((ydata - self(xdata)) / sigma)**2.).sum())

    @staticmethod
    def freeze(model_function, **constraints) -> Callable:
        """Freeze a subset of the model parameters.

        Arguments
        ---------
        model_function : callable
            The model function to freeze parameters for.

        constraints : dict
            The parameters to freeze, as keyword arguments.

        Returns
        -------
        wrapper : callable
            A wrapper around the model function with the given parameters frozen.
        """
        if not constraints:
            return model_function

        # Cache a couple of constant to save on line length later.
        positional_only = inspect.Parameter.POSITIONAL_ONLY
        positional_or_keyword = inspect.Parameter.POSITIONAL_OR_KEYWORD

        # scipy.optimize.curve_fit assumes the first argument of the model function
        # is the independent variable...
        x, *parameters = inspect.signature(model_function).parameters.values()
        # ... while all the others, internally, are passed positionally only
        # (i.e., never as keywords), so here we cache all the names of the
        # positional parameters.
        parameter_names = [parameter.name for parameter in parameters if
                           parameter.kind in (positional_only, positional_or_keyword)]

        # Make sure the constraints are valid, and we are not trying to freeze one
        # or more non-existing parameter(s). This is actually clever, as it uses the fact
        # that set(dict) returns the set of the keys, and after subtracting the two sets
        # you end up with all the names of the unknown parameters, which is handy to
        # print out an error message.
        unknown_parameter_names = set(constraints) - set(parameter_names)
        if unknown_parameter_names:
            raise ValueError(f"Cannot freeze unknown parameters {unknown_parameter_names}")

        # Now we need to build the signature for the new function, starting from  a
        # clean copy of the parameter for the independent variable...
        parameters = [x.replace(default=inspect.Parameter.empty, kind=positional_or_keyword)]
        # ... and following up with all the free parameters.
        free_parameter_names = [name for name in parameter_names if name not in constraints]
        num_free_parameters = len(free_parameter_names)
        for name in free_parameter_names:
            parameters.append(inspect.Parameter(name, kind=positional_or_keyword))
        signature = inspect.Signature(parameters)

        # And we have everything to prepare the glorious wrapper!
        @functools.wraps(model_function)
        def wrapper(x, *args):
            if len(args) != num_free_parameters:
                raise TypeError(f"Frozen wrapper got {len(args)} parameters instead of " \
                                f"{num_free_parameters} ({free_parameter_names})")
            parameter_dict = {**dict(zip(free_parameter_names, args)), **constraints}
            return model_function(x, *[parameter_dict[name] for name in parameter_names])

        wrapper.__signature__ = signature
        return wrapper

    def fit(self, xdata: ArrayLike, ydata: ArrayLike, p0: ArrayLike = None,
            sigma: ArrayLike = 1., absolute_sigma: bool = False, xmin: float = -np.inf,
            xmax: float = np.inf, **kwargs) -> FitStatus:
        """Fit a series of points.

        Arguments
        ---------
        xdata : array_like
            The input values of the independent variable.

        ydata : array_like
            The input values of the dependent variable.

        p0 : array_like, optional
            The initial values for the fit parameters.

        sigma : array_like
            The input uncertainties on the dependent variable.

        absolute_sigma : bool, optional (default False)
            See the `curve_fit()` documentation for details.

        xmin : float, optional (default -inf)
            The minimum value of the independent variable to fit.

        xmax : float, optional (default inf)
            The maximum value of the independent variable to fit.

        Returns
        -------
        status : FitStatus
            The status of the fit.
        """
        # Reset the fit status.
        self.status.reset()

        # Prepare the data. We want to make sure all the relevant things are numpy
        # arrays so that we can vectorize operations downstream, taking advantage of
        # the broadcast facilities.
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        if isinstance(sigma, Number):
            sigma = np.full(ydata.shape, sigma)
        sigma = np.asarray(sigma)
        # If we are fitting over a subrange, filter the input data.
        mask = np.logical_and(xdata >= xmin, xdata <= xmax)
        # Also, filter out any points with non-positive uncertainties.
        mask = np.logical_and(mask, sigma > 0.)
        # (And, since we are at it, make sure we have enough degrees of freedom.)
        self.status.dof = int(mask.sum() - len(self.free_parameters()))
        if self.status.dof < 0:
            raise RuntimeError(f"{self.name()} has no degrees of freedom")
        xdata = xdata[mask]
        ydata = ydata[mask]
        sigma = sigma[mask]

        # Cache the fit range for later use.
        self.status.fit_range = (xdata.min(), xdata.max())

        # If we are not passing default starting points for the model parameters,
        # try and do something sensible.
        if p0 is None:
            self.init_parameters(xdata, ydata, sigma)
            p0 = self.free_parameter_values()

        # Do the actual fit.
        constraints = {parameter.name: parameter.value for parameter in self \
                       if parameter.frozen}
        model = self.freeze(self.evaluate, **constraints)
        args = model, xdata, ydata, p0, sigma, absolute_sigma, True, self.bounds()
        popt, pcov = curve_fit(*args, **kwargs)
        self.update_parameters(popt, pcov)
        self.status.update(self.calculate_chisquare(xdata, ydata, sigma))
        return self.status

    def fit_histogram(self, histogram: Histogram1d, p0: ArrayLike = None, **kwargs) -> None:
        """Convenience function for fitting a 1-dimensional histogram.

        Arguments
        ---------
        histogram : Histogram1d
            The histogram to fit.

        p0 : array_like, optional
            The initial values for the fit parameters.

        kwargs : dict, optional
            Additional keyword arguments passed to `fit()`.
        """
        args = histogram.bin_centers(), histogram.content, p0, histogram.errors
        return self.fit(*args, **kwargs)

    def default_plotting_range(self) -> Tuple[float, float]:
        """Return the default plotting range for the model.

        This can be reimplemented in concrete models, and can be parameter-dependent
        (e.g., for a gaussian we might want to plot within 5 sigma from the mean by
        default).

        Returns
        -------
        Tuple[float, float]
            The default plotting range for the model.
        """
        return (0., 1.)

    def _plotting_range(self, xmin: float = None, xmax: float = None,
                        fit_padding: float = 0.) -> Tuple[float, float]:
        """Convenience function trying to come up with the most sensible plot range
        for the model.

        Arguments
        ---------
        xmin : float, optional
            The minimum value of the independent variable to plot.

        xmax : float, optional
            The maximum value of the independent variable to plot.

        fit_padding : float, optional
            The amount of padding to add to the fit range.

        Returns
        -------
        Tuple[float, float]
            The plotting range for the model.
        """
        # If we have fitted the model to some data, we take the fit range and pad it
        # a little bit.
        if self.status.fit_range is not None:
            _xmin, _xmax = self.status.fit_range
            fit_padding *= (_xmax - _xmin)
            _xmin -= fit_padding
            _xmax += fit_padding
        # Otherwise we fall back to the default plotting range for the model.
        else:
            _xmin, _xmax = self.default_plotting_range()
        # And are free to override either end!
        if xmin is not None:
            _xmin = xmin
        if xmax is not None:
            _xmax = xmax
        return (_xmin, _xmax)

    def plot(self, xmin: float = None, xmax: float = None, num_points: int = 200) -> np.ndarray:
        """Plot the model.

        Arguments
        ---------
        xmin : float, optional
            The minimum value of the independent variable to plot.

        xmax : float, optional
            The maximum value of the independent variable to plot.

        num_points : int, optional
            The number of points to use for the plot.

        Returns
        -------
        x : np.ndarray
            The x values used for the plot, that can be used downstream to add
            artists on the plot itself (e.g., composite models can use the same
            grid to draw the components).
        """
        x = np.linspace(*self._plotting_range(xmin, xmax), num_points)
        y = self(x)
        plt.plot(x, y, label=format(self, Format.LATEX))
        return x

    def __format__(self, spec: str) -> str:
        """String formatting.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        text = f"{self.name()}\n"
        if self.status is not None:
            text = f"{text}{format(self.status, spec)}\n"
        for parameter in self:
            text = f"{text}{format(parameter, spec)}\n"
        return text.strip("\n")

    def __str__(self):
        """String formatting.

        Returns
        -------
        text : str
            The formatted string.
        """
        return format(self, Format.PRETTY)


class AbstractFitModel(AbstractFitModelBase):

    """Abstract base class for a fit model.
    """

    def __init__(self) -> None:
        """Constructor.

        Here we loop over the FitParameter objects defined at the class level, and
        create copies that are attached to the instance, so that the latter has its
        own state.
        """
        super().__init__()
        self._parameters = []
        for name, value in self.__class__.__dict__.items():
            if isinstance(value, FitParameter):
                parameter = value.copy(name)
                # Note we also set one instance attribute for each parameter so
                # that we can use the notation model.parameter
                setattr(self, name, parameter)
                self._parameters.append(parameter)

    def __len__(self) -> int:
        """Return the `total` number of fit parameters in the model.
        """
        return len(self._parameters)

    def __iter__(self) -> Iterator[FitParameter]:
        """Iterate over `all` the model parameters.
        """
        return iter(self._parameters)

    def __add__(self, other):
        """Model sum.
        """
        if not isinstance(other, AbstractFitModel):
            raise TypeError(f"{other} is not a fit model")
        return FitModelSum(self, other)


class FitModelSum(AbstractFitModelBase):

    """Composite model representing the sum of an arbitrary number of simple models.

    Arguments
    ---------
    components : sequence of AbstractFitModel
        The components of the composite model.
    """

    def __init__(self, *components: AbstractFitModel) -> None:
        """Constructor.
        """
        super().__init__()
        self._components = components

    def name(self) -> str:
        """Return the model name.
        """
        return " + ".join(component.name() for component in self._components)

    def __len__(self) -> int:
        """Return the sum of `all` the fit parameters in the underlying models.
        """
        return sum(len(component) for component in self._components)

    def __iter__(self) -> Iterator[FitParameter]:
        """Iterate over `all` the parameters of the underlying components.
        """
        return chain(*self._components)

    def evaluate(self, x: ArrayLike, *parameter_values) -> ArrayLike:
        """Overloaded method for evaluating the model.

        Note this is not a static method, as we need to access the list of components
        to sum over.
        """
        # pylint: disable=arguments-differ
        cursor = 0
        value = np.zeros(x.shape)
        for component in self._components:
            value += component.evaluate(x, *parameter_values[cursor:cursor + len(component)])
            cursor += len(component)
        return value

    def plot(self, xmin: float = None, xmax: float = None, num_points: int = 200) -> None:
        """Overloaded method for plotting the model.
        """
        x = super().plot(xmin, xmax, num_points)
        color = plt.gca().lines[-1].get_color()
        for component in self._components:
            y = component(x)
            plt.plot(x, y, label=None, ls="--", color=color)

    def __format__(self, spec: str) -> str:
        """String formatting.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        text = f"{self.name()}\n"
        if self.status is not None:
            text = f"{text}{format(self.status, spec)}\n"
        for component in self._components:
            text = f"{text}[{component.name()}]\n"
            for parameter in component:
                text = f"{text}{format(parameter, spec)}\n"
        return text.strip("\n")

    def __add__(self, other: AbstractFitModel) -> "FitModelSum":
        """Implementation of the model sum (i.e., using the `+` operator).

        Note that, in the spirit of keeping the interfaces as simple as possible,
        we are not implementing in-place addition (i.e., `+=`), and we only
        allow ``AbstractFitModel`` objects (not ``FitModelSum``) on the right
        hand side, which is all is needed to support the sum of an arbitrary
        number of models.
        """
        return self.__class__(*self._components, other)


class Constant(AbstractFitModel):

    """Constant model.
    """

    value = FitParameter(1.)

    @staticmethod
    def evaluate(x: ArrayLike, value: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return np.full(x.shape, value)


class Line(AbstractFitModel):

    """Linear model.
    """

    slope = FitParameter(1.)
    intercept = FitParameter(0.)

    @staticmethod
    def evaluate(x: ArrayLike, slope: float, intercept: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return slope * x + intercept


class PowerLaw(AbstractFitModel):

    """Power-law model.
    """

    prefactor = FitParameter(1.)
    index = FitParameter(-1.)

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, index: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return prefactor * x**index


class Gaussian(AbstractFitModel):

    """Gaussian model.
    """

    prefactor = FitParameter(1.)
    mean = FitParameter(0.)
    sigma = FitParameter(1., minimum=0.)

    _NORM_CONSTANT = 1. / np.sqrt(2. * np.pi)
    _SIGMA_TO_FWHM = 2. * np.sqrt(2. * np.log(2.))

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, mean: float, sigma: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        z = (x - mean) / sigma
        return prefactor * Gaussian._NORM_CONSTANT / sigma * np.exp(-0.5 * z**2.)

    def default_plotting_range(self, num_sigma: int = 5) -> Tuple[float, float]:
        mean, half_width = self.mean.value, num_sigma * self.sigma.value
        return (mean - half_width, mean + half_width)

    def fwhm(self) -> uncertainties.ufloat:
        """Return the full-width at half-maximum (FWHM) of the gaussian.

        Returns
        -------
        fwhm : uncertainties.ufloat
            The FWHM of the gaussian.
        """
        return self.sigma.ufloat() * self._SIGMA_TO_FWHM
