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
from itertools import chain
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Iterator, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from scipy.optimize import curve_fit

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
        """
        return self._name

    @property
    def frozen(self) -> bool:
        """Return True if the parameter is frozen.

        We are wrapping this into a property because we interact with this member
        via the freeze() and thaw() methods.
        """
        return self._frozen

    def is_bound(self) -> bool:
        """Return True if the parameter is bounded.
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
            The name for the new FitParameter object.
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
        """
        return uncertainties.ufloat(self.value, self.error)

    def __format__(self, spec: str) -> str:
        """String formatting.
        """
        # Keep in mind Python passes an empty string explicitly when you call
        # f"{parameter}", so we can't really assign a default value to spec.
        if self.error is not None:
            param = format(self.ufloat(), spec)
            if spec.endswith(Format.LATEX):
                param = f"${param}$"
        else:
            spec = spec.rstrip(Format.PRETTY).rstrip(Format.LATEX)
            param = format(self.value, spec)
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
        """
        return format(self, Format.PRETTY)


@dataclass
class FitStatus:

    """Small dataclass to hold the fit status.
    """

    chisquare: float = None
    dof: int = None
    # pvalue: float = None
    fit_range: Tuple[float, float] = None

    def reset(self) -> None:
        """Reset the fit status.
        """
        self.chisquare = None
        self.dof = None
        self.fit_range = None

    def __format__(self, spec: str) -> str:
        """String formatting.
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
        """

    @abstractmethod
    def __iter__(self) -> Iterator[FitParameter]:
        """Delegated to concrete classes: this should return an iterator over `all`
        the fit parameters in the model.
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
        """

    def name(self) -> str:
        """Return the model name, e.g., for legends.

        Note this can be reimplemented in concrete subclasses, but it should provide
        a sensible default value in most circumstances.
        """
        return self.__class__.__name__

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the model at the current value of the parameters.
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
        """
        return tuple(parameter.value for parameter in self)

    def free_parameters(self) -> Tuple[FitParameter]:
        """Return the list of free parameters.

        Note this only relies on the __iter__() method, so it works both for simple
        and composite models.
        """
        return tuple(parameter for parameter in self if not parameter.frozen)

    def free_parameter_values(self) -> Tuple[float]:
        """Return the current parameter values.
        """
        return tuple(parameter.value for parameter in self.free_parameters())

    def bounds(self) -> Tuple[ArrayLike, ArrayLike]:
        """Return the bounds on the fit parameters in a form that can be use by the
        fitting method.
        """
        free_parameters = self.free_parameters()
        return (tuple(parameter.minimum for parameter in free_parameters),
                tuple(parameter.maximum for parameter in free_parameters))

    def update_parameters(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        """Update the model parameters based on the output of the ``curve_fit()`` call.

        Note this only relies on the __iter__() method, so it works both for simple
        and composite models.
        """
        for parameter, value, error in zip(self.free_parameters(), popt, np.sqrt(pcov.diagonal())):
            parameter.value = value
            parameter.error = error

    def prepare_fit(self, xdata: ArrayLike, ydata: ArrayLike, p0: ArrayLike = None,
                    sigma: ArrayLike = 1., xmin: float = -np.inf, xmax: float = np.inf) -> None:
        """Setup the model and the input data for a fit.
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
        self.status.dof = int(mask.sum() - len(self))
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
        return xdata, ydata, p0, sigma

    def calculate_chisqure(self, xdata: np.ndarray, ydata: np.ndarray, sigma) -> float:
        """Calculate the chisquare of the fit to some input data with the current
        model parameters.
        """
        return float((((ydata - self(xdata)) / sigma)**2.).sum())

    def fit_histogram(self, histogram: "Histogram1d", p0: ArrayLike = None, **kwargs) -> None:
        """Convenience function for fitting a 1-dimensional histogram.

        Arguments
        ---------
        histogram : Histogram1d
            The histogram to fit.

        p0 : array_like, optional
            The initial values for the fit parameters.

        **kwargs : dict, optional
            Additional keyword arguments passed to `fit()`.
        """
        args = histogram.bin_centers(), histogram.content, p0, histogram.errors
        return self.fit(*args, **kwargs)

    def default_plotting_range(self) -> Tuple[float, float]:
        """Return the default plotting range for the model.

        This can be reimplemnted in concrete models, and can be parameter-dependent
        (e.g., for a gaussian we might want to plot within 5 sigma from the mean by
        dafault).
        """
        return (0., 1.)

    def _plotting_range(self, xmin: float = None, xmax: float = None,
                        fit_padding: float = 0.) -> Tuple[float, float]:
        """Convenience function trying to come up with the most sensible plot range
        for the model.
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

    def plot(self, xmin: float = None, xmax: float = None, num_points: int = 200) -> None:
        """Plot the model.
        """
        x = np.linspace(*self._plotting_range(xmin, xmax), num_points)
        y = self(x)
        plt.plot(x, y, label=format(self, Format.LATEX))


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

    @staticmethod
    def freeze(model_function, **constraints):
        """Freeze a subset of the model parameters.
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
            xmax: float = np.inf, **kwargs) -> None:
        """Fit a series of points.
        """
        # Prepare the data and the initial parameter values.
        xdata, ydata, p0, sigma = self.prepare_fit(xdata, ydata, p0, sigma, xmin, xmax)

        # Do the actual fit.
        constraints = {parameter.name: parameter.value for parameter in self \
                       if parameter.frozen}
        model = self.freeze(self.evaluate, **constraints)
        args = model, xdata, ydata, p0, sigma, absolute_sigma, True, self.bounds()
        popt, pcov = curve_fit(*args, **kwargs)
        self.update_parameters(popt, pcov)
        self.status.chisquare = self.calculate_chisqure(xdata, ydata, sigma)
        return self.status

    def __format__(self, spec: str) -> str:
        """String formatting.
        """
        text = f"{self.name()} ({format(self.status, spec)})\n"
        for parameter in self._parameters:
            text = f"{text}{format(parameter, spec)}\n"
        return text.strip("\n")

    def __add__(self, other):
        """Model sum.
        """
        if not isinstance(other, AbstractFitModel):
            raise TypeError(f"{other} is not a fit model")
        return FitModelSum(self, other)

    def __str__(self):
        """String formatting.
        """
        return format(self, Format.PRETTY)


class FitModelSum(AbstractFitModelBase):

    def __init__(self, *components) -> None:
        """
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
        """
        """
        cursor = 0
        value = np.zeros(x.shape)
        for component in self._components:
            value += component.evaluate(x, *parameter_values[cursor:cursor + len(component)])
            cursor += len(component)
        return value

    def fit(self, xdata: ArrayLike, ydata: ArrayLike, p0: ArrayLike = None,
            sigma: ArrayLike = 1., absolute_sigma: bool = False, xmin: float = -np.inf,
            xmax: float = np.inf, **kwargs) -> None:
        """Fit a series of points.
        """
        # Prepare the data and the initial parameter values.
        xdata, ydata, p0, sigma = self.prepare_fit(xdata, ydata, p0, sigma, xmin, xmax)

        # Do the actual fit.
        #constraints = {parameter.name: parameter.value for parameter in self \
        #               if parameter.frozen}
        #model = self.freeze(self.evaluate, **constraints)
        model = self.evaluate
        args = model, xdata, ydata, p0, sigma, absolute_sigma, True, self.bounds()
        popt, pcov = curve_fit(*args, **kwargs)
        self.update_parameters(popt, pcov)
        self.status.chisquare = self.calculate_chisqure(xdata, ydata, sigma)
        return self.status

    def plot(self, xmin: float = None, xmax: float = None, num_points: int = 200) -> None:
        """Plot the model.
        """
        x = np.linspace(*self._plotting_range(xmin, xmax), num_points)
        y = self(x)
        plt.plot(x, y)


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
    sigma = FitParameter(1.)

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, mean: float, sigma: float) -> ArrayLike:
        # pylint: disable=arguments-differ
        return prefactor * np.exp(-0.5 * ((x - mean) / sigma) ** 2.)

    def default_plotting_range(self, num_sigma: int = 5) -> Tuple[float, float]:
        mean, half_width = self.mean.value, num_sigma * self.sigma.value
        return (mean - half_width, mean + half_width)
