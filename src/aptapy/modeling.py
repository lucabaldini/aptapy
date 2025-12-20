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

"""Modeling core facilities.
"""

import enum
import functools
import inspect
from abc import abstractmethod
from dataclasses import dataclass, fields
from itertools import chain
from numbers import Number
from typing import Callable, Dict, Iterator, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
import uncertainties

from .hist import Histogram1d
from .plotting import AbstractPlottable, last_line_color
from .typing_ import ArrayLike

SIGMA_TO_FWHM = 2. * np.sqrt(2. * np.log(2.))


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
        if self._frozen:
            raise RuntimeError(f"Cannot set value for frozen parameter {self.name}")
        if value < self.minimum or value > self.maximum:
            raise ValueError(f"Cannot set value {value} for parameter {self.name}, "
                             f"out of bounds [{self.minimum}, {self.maximum}]")
        self.value = float(value)
        self.error = float(error) if error is not None else None

    def init(self, value: float) -> None:
        """Initialize the fit parameter to a given value, unless it is frozen, or
        the value is out of bounds.

        .. warning::

           Note this silently does nothing if the parameter is frozen, or if the value
           is out of bounds, so its behavior is inconsistent with that of set(), which
           raises an exception in both cases. This is intentional, and this method should
           only be used to initialize the parameter prior to a fit.

        Arguments
        ---------
        value : float
            The new value for the parameter.

        """
        if self._frozen:
            return
        if value < self.minimum or value > self.maximum:
            return
        self.set(value)

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
        # Setting the error to a very small number if it is None to avoid unexpected
        # behaviors of uncertainties.ufloat().
        error = self.error if self.error is not None else 1e-15
        return uncertainties.ufloat(self.value, error)

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
        text = f"{self._name}: {param}"
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

    popt: np.ndarray = None
    pcov: np.ndarray = None
    chisquare: float = None
    dof: int = None
    pvalue: float = None
    correlated_pars: np.ndarray = None

    def reset(self) -> None:
        """Reset the fit status.
        """
        for field in fields(self):
            setattr(self, field.name, None)

    def valid(self) -> bool:
        """Return True if the fit status is valid, i.e., all the fields are set.

        Returns
        -------
        valid : bool
            True if the fit status is valid.
        """
        return all(getattr(self, field.name) is not None for field in fields(self))

    def update(self, popt: np.ndarray, pcov: np.ndarray, chisquare: float, dof: int) -> None:
        """Update the fit status, i.e., set the chisquare and calculate the
        corresponding p-value.

        Arguments
        ---------
        popt : array_like
            The optimal values for the fit parameters.

        pcov : array_like
            The covariance matrix for the fit parameters.

        chisquare : float
            The chisquare of the fit.

        dof : int
            The number of degrees of freedom of the fit.
        """
        self.popt = popt
        self.pcov = pcov
        self.chisquare = chisquare
        self.dof = dof
        self.pvalue = scipy.stats.chi2.sf(self.chisquare, self.dof)
        self.correlated_pars = uncertainties.correlated_values(popt, pcov)
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
            return f"$\\chi^2$: {self.chisquare:.2f} / {self.dof} dof"
        if spec.endswith(Format.PRETTY):
            return f"χ²: {self.chisquare:.2f} / {self.dof} dof"
        return f"chisquare: {self.chisquare:.2f} / {self.dof} dof"

    def __str__(self) -> str:
        """String formatting.

        Returns
        -------
        text : str
            The formatted string.
        """
        return format(self, Format.PRETTY)


class AbstractFitModelBase(AbstractPlottable):

    """Abstract base class for all the fit classes.

    This is a acting a base class for both simple fit models and for composite models
    (e.g., sums of simple ones).

    Arguments
    ---------
    label : str, optional
        The label for the model. If this is None, the model name is used as default,
        which makes sense because the name is how we would label a fit model in
        most circumstances.

    xlabel : str, optional
        The label for the x-axis.

    ylabel : str, optional
        The label for the y-axis.
    """
    # pylint: disable=too-many-public-methods
    def __init__(self, label: str = None, xlabel: str = None, ylabel: str = None) -> None:
        """Constructor.
        """
        super().__init__(label, xlabel, ylabel)
        if self.label is None:
            self.label = self.name()
        self.status = FitStatus()
        # Plotting range overriding the default coded in default_plotting_range().
        # This is set when fitting, and can be overridden programmatically by the user
        # via set_plotting_range() at any time.
        self._plotting_range = None
        # Number of points to use when plotting the model.
        self.num_plotting_points = 200

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
    def evaluate(x: ArrayLike, *parameter_values: float) -> ArrayLike:
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

    def _wrap_evaluate(self) -> Callable:
        """Helper function to build a wrapper around the evaluate() method with
        the (correct) explicit signature, including all the parameter names.

        This is used, e.g., by FitModelSum and GaussianForestBase to wrap the evaluate()
        method, which is expressed in terms of a generic signature, before the
        method itself is passed to the freeze() method.
        """
        # Build the correct signature for the evaluate() method. Note that
        # the method bound to the class has signature (self, x, *parameter_values)
        # and we do want a wrapper with signature (x, param1, param2, ...).
        parameters = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY)]
        parameter_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        parameters.extend(inspect.Parameter(par.name, parameter_kind) for par in self)
        signature = inspect.Signature(parameters)

        # Create a loose wrapper around evaluate().
        @functools.wraps(self.evaluate)
        def wrapper(x, *args):
            return self.evaluate(x, *args)

        # Set the correct signature on the wrapper and return it.
        wrapper.__signature__ = signature
        return wrapper

    def jacobian(self, x: ArrayLike, *parameter_values: float, eps: float = 1.e-8) -> np.ndarray:
        """Numerically calculate the Jacobian matrix of partial derivatives of the model
        with respect to the parameters.

        This is used, e.g., to plot confidence bands around the best-fit model.

        Arguments
        ----------
        x : array_like
            The value(s) of the independent variable.

        parameter_values : sequence of float
            The value of the model parameters. If no parameters are passed, the current
            values are used by default. Alternatively, all the model parameters must
            be passed, otherwise a ValueError is raised.

        eps : float, optional
            The step size to use for the numerical differentiation.

        Returns
        -------
        J : ndarray
            The Jacobian matrix of partial derivatives. The shape of the array is (m, n),
            where m is the number of data points where the Jacobian is calculated, and
            n the number of parameters.
        """
        # pylint: disable=invalid-name
        if len(parameter_values) == 0:
            parameter_values = self.parameter_values()
        if len(parameter_values) != len(self):
            raise ValueError(f"{len(self)} parameters expected in the Jacobian calculation")

        p = np.array(parameter_values, dtype=float)
        x = np.atleast_1d(x)
        m = len(x)
        n = len(p)
        J = np.zeros((m, n))
        for i in range(n):
            dp = np.zeros_like(p)
            dp[i] = eps
            J[:, i] = (self.evaluate(x, *(p + dp)) - self.evaluate(x, *(p - dp))) / (2. * eps)
        return J

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

    def set_parameters(self, *parameter_values: float) -> None:
        """Set the model parameters to the given values.

        Arguments
        ---------
        parameter_values : sequence of float
            The new values for the model parameters.
        """
        for parameter, value in zip(self, parameter_values):
            parameter.set(value)

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
            parameter.set(value, error)

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

        # Cache a couple of constants to save on line length later.
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

    def fit(self, xdata: Union[ArrayLike, Histogram1d], ydata: ArrayLike = None, *,
            p0: ArrayLike = None, sigma: ArrayLike = None, absolute_sigma: bool = False,
            xmin: float = -np.inf, xmax: float = np.inf, **kwargs) -> FitStatus:
        """Fit a series of points.

        Arguments
        ---------
        xdata : array_like or one-dimensional histogram
            The input values of the independent variable or a 1-dimensional histogram.

        ydata : array_like, optional
            The input values of the dependent variable.

        p0 : array_like, optional
            The initial values for the fit parameters.

        sigma : array_like, optional
            The input uncertainties on the dependent variable.

        absolute_sigma : bool, optional (default False)
            See the `curve_fit()` documentation for details.

        xmin : float, optional (default -inf)
            The minimum value of the independent variable to fit. Note that if
            xmin < xmax the (xmax, xmin) interval is excluded from the fit.

        xmax : float, optional (default inf)
            The maximum value of the independent variable to fit. Note that if
            xmin < xmax the (xmax, xmin) interval is excluded from the fit.

        kwargs : dict, optional
            Additional keyword arguments passed to `curve_fit()`.

        Returns
        -------
        status : FitStatus
            The status of the fit.
        """
        # Dispatch the input arguments if we are fitting a histogram.
        if isinstance(xdata, Histogram1d):
            if ydata is not None:
                raise ValueError("ydata must be None when xdata is a Histogram1d")
            if sigma is not None:
                raise ValueError("sigma must be None when xdata is a Histogram1d")
            histogram = xdata
            xdata = histogram.bin_centers()
            ydata = histogram.content
            sigma = histogram.errors
        else:
            if ydata is None:
                raise ValueError("ydata must be provided when xdata is array-like")

        # Reset the fit status.
        self.status.reset()

        # Prepare the data. We want to make sure all the relevant things are numpy
        # arrays so that we can vectorize operations downstream, taking advantage of
        # the broadcast facilities.
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        if sigma is None:
            sigma = np.ones_like(ydata)
        elif isinstance(sigma, Number):
            sigma = np.full(ydata.shape, sigma)
        sigma = np.asarray(sigma)
        # If we are fitting over a subrange, filter the input data. We have three cases:
        # if the limits are equal we raise...
        if xmin == xmax:
            raise ValueError("xmin and xmax cannot be equal")
        # ... if xmin < xmax (which is the usual case) we do a standard in-range
        # selection...
        if xmin < xmax:
            mask = np.logical_and(xdata >= xmin, xdata <= xmax)
        # ... while if xmin > xmax we take this as signaling that we want to
        # exclude the (xmax, xmin) interval, and do an out-of-range selection.
        else:
            mask = np.logical_or(xdata >= xmin, xdata <= xmax)
        # Also, filter out any points with non-positive uncertainties.
        mask = np.logical_and(mask, sigma > 0.)
        # (And, since we are at it, make sure we have enough degrees of freedom.)
        dof = int(mask.sum() - len(self.free_parameters()))
        if dof < 0:
            raise RuntimeError(f"{self.name()} has no degrees of freedom")
        xdata = xdata[mask]
        ydata = ydata[mask]
        sigma = sigma[mask]

        # Cache the fit range for later use.
        self._plotting_range = (xdata.min(), xdata.max())

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
        popt, pcov = scipy.optimize.curve_fit(*args, **kwargs)
        self.update_parameters(popt, pcov)
        self.status.update(popt, pcov, self.calculate_chisquare(xdata, ydata, sigma), dof)
        return self.status

    @staticmethod
    def default_plotting_range() -> Tuple[float, float]:
        """Return the default plotting range for the model.

        This can be reimplemented in concrete models, and can be parameter-dependent
        (e.g., for a gaussian we might want to plot within 5 sigma from the mean by default).
        And if you think for a moment to move this to a ``DEFAULT_PLOTTING_RANGE`` class variable,
        keep in mind that having it as a method allows for parameter-dependent default ranges.

        Returns
        -------
        Tuple[float, float]
            The default plotting range for the model.
        """
        return (0., 1.)

    def set_plotting_range(self, xmin: float, xmax: float) -> None:
        """Set a custom plotting range for the model.

        Arguments
        ---------
        xmin : float
            The minimum x value for plotting.

        xmax : float
            The maximum x value for plotting.
        """
        self._plotting_range = (xmin, xmax)

    def plotting_range(self) -> Tuple[float, float]:
        """Return the current plotting range for the model.

        If a custom plotting range has been set via `set_plotting_range()`, or as
        a part of a fit, that is returned, otherwise the default plotting range for
        the model is used.

        Returns
        -------
        Tuple[float, float]
            The plotting range for the model.
        """
        if self._plotting_range is not None:
            return self._plotting_range
        return self.default_plotting_range()

    def _plotting_grid(self) -> np.ndarray:
        """Return the grid of x values to use for plotting the model.

        Returns
        -------
        x : np.ndarray
            The x values used for plotting the model.
        """
        return np.linspace(*self.plotting_range(), self.num_plotting_points)

    def _render(self, axes: matplotlib.axes.Axes = None, **kwargs) -> None:
        """Render the model on the given axes.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on (default: current axes).

        kwargs : dict, optional
            Additional keyword arguments passed to `axes.plot()`.
        """
        x = self._plotting_grid()
        axes.plot(x, self(x), **kwargs)

    def plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False,
             **kwargs) -> matplotlib.axes.Axes:
        """Plot the model.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on (default: current axes).

        kwargs : dict, optional
            Additional keyword arguments passed to `plt.plot()`.
        """
        kwargs.setdefault("label", self.label)
        if fit_output:
            kwargs["label"] = f"{kwargs['label']}\n{self._format_fit_output(Format.LATEX)}"
        return super().plot(axes, **kwargs)

    def confidence_band(self, x: ArrayLike, num_sigma: float = 1.) -> np.ndarray:
        """Return the vertical width of the n-sigma confidence band at the given x values.

        Note this assumes that the model has been fitted to data and is equipped with a
        valid FitStatus. A RuntimeError is raised if that is not the case.

        Arguments
        ---------
        x : array_like
            The x values where the confidence delta is calculated.

        num_sigma : float
            The number of sigmas for the band (default 1).

        Returns
        -------
        delta : np.ndarray
            The vertical width of the n-sigma confidence band at the given x values.
        """
        # pylint: disable=invalid-name
        if not self.status.valid():
            raise RuntimeError("Invalid fit status, cannot calculate confidence band")
        J = self.jacobian(x, *self.status.popt)
        return num_sigma * np.sqrt(np.einsum("ij,jk,ik->i", J, self.status.pcov, J))

    def plot_confidence_band(self, axes: matplotlib.axes.Axes = None, num_sigma: float = 1.,
                             **kwargs) -> matplotlib.axes.Axes:
        """Plot the n-sigma confidence band around the best-fit model.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on (default: current axes).

        num_sigma : float, optional
            The number of sigmas for the confidence band (default: 1).

        kwargs : dict, optional
            Additional keyword arguments passed to `axes.fill_between()`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the confidence band plotted.
        """
        if axes is None:
            axes = plt.gca()
        kwargs.setdefault("color", last_line_color(axes))
        kwargs.setdefault("alpha", 0.25)
        kwargs.setdefault("label", f"{num_sigma}σ confidence band")
        x = self._plotting_grid()
        y = self(x)
        delta = self.confidence_band(x, num_sigma)
        axes.fill_between(x, y - delta, y + delta, **kwargs)
        return axes

    def random_fit_dataset(self, sigma: ArrayLike, num_points: int = 25,
                           seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random sample from the model, adding gaussian noise.

        Arguments
        ---------
        sigma : array_like
            The standard deviation of the gaussian noise to add to the model.

        num_points : int, optional
            The number of points to generate (default 25).

        seed : int, optional
            The random seed to use (default None).

        Returns
        -------
        xdata : np.ndarray
            The x values of the random sample.

        ydata : np.ndarray
            The y values of the random sample.
        """
        xdata = np.linspace(*self.plotting_range(), num_points)
        if isinstance(sigma, Number):
            sigma = np.full(xdata.shape, sigma)
        if len(sigma) != len(xdata):
            raise ValueError("Length of sigma does not match number of points")
        ydata = self(xdata) + np.random.default_rng(seed).normal(0., sigma)
        return xdata, ydata

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
        raise NotImplementedError(f"{self.__class__.__name__} does not implement rvs()")

    def random_histogram(self, edges: np.ndarray, size: int, random_state=None) -> Histogram1d:
        """Generate a histogram filled with random variates from the underlying
        distribution at the current parameter values.

        Arguments
        ---------
        edges : np.ndarray
            The bin edges of the histogram.

        size : int, optional
            The number of random variates to generate (default 100000).

        random_state : int or np.random.Generator, optional
            The random seed or generator to use (default None).

        Returns
        -------
        Histogram1d
            A histogram filled with random variates from the distribution.
        """
        return Histogram1d(edges).fill(self.rvs(size, random_state=random_state))

    def _format_fit_output(self, spec: str) -> str:
        """String formatting for fit output.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        text = ""
        if self.status.valid():
            text = f"{text}{format(self.status, spec)}\n"
        for parameter in self:
            text = f"{text}{format(parameter, spec)}\n"
        return text.strip("\n")

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
        return f"{self.name()}\n{self._format_fit_output(spec)}"

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

    def __init__(self, label: str = None, xlabel: str = None, ylabel: str = None) -> None:
        """Constructor.

        Here we loop over the FitParameter objects defined at the class level, and
        create copies that are attached to the instance, so that the latter has its
        own state.
        """
        super().__init__(label, xlabel, ylabel)
        self._parameters = []
        # Note we cannot loop over self.__dict__.items() here, as that would
        # only return the members defined in the actual class, and not the
        # inherited ones.
        for name, value in self.__class__._parameter_dict().items():
            parameter = value.copy(name)
            # Note we also set one instance attribute for each parameter so
            # that we can use the notation model.parameter
            setattr(self, name, parameter)
            self._parameters.append(parameter)

    @classmethod
    def _parameter_dict(cls) -> Dict[str, FitParameter]:
        """Return a dictionary of all the FitParameter objects defined in the class
        and its base classes.

        This is a subtle one, as what we really want, here, is all members of a class
        (including inherited ones) that are of a specific type (FitParameter), in the
        order they were defined. All of these thing are instrumental to make the
        fit model work, so we need to be careful.

        Also note the we are looping over the MRO in reverse order, so that we
        preserve the order of definition of the parameters, even when they are
        inherited from base classes. If a parameter is re-defined in a derived class,
        the derived class definition takes precedence, as we are using a dictionary
        to collect the parameters.

        Arguments
        ---------
        cls : type
            The class to inspect.

        Returns
        -------
        param_dict : dict
            A dictionary mapping parameter names to their FitParameter objects.
        """
        param_dict = {}
        for base in reversed(cls.__mro__):
            param_dict.update({name: value for name, value in base.__dict__.items() if
                               isinstance(value, FitParameter)})
        return param_dict

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

    def quadrature(self, x1: float, x2: float) -> float:
        """Calculate the integral of the model between x1 and x2 using
        numerical integration.

        Arguments
        ---------
        x1 : float
            The minimum value of the independent variable to integrate over.

        x2 : float
            The maximum value of the independent variable to integrate over.

        Returns
        -------
        integral : float
            The integral of the model between x1 and x2.
        """
        value, _ = scipy.integrate.quad(self, x1, x2)
        return value

    def integral(self, x1: float, x2: float) -> float:
        """Default implementation of the integral of the model between x1 and x2.
        Subclasses can (and are encouraged to) overload this method with an
        analytical implementation, when available.

        Arguments
        ---------
        x1 : float
            The minimum value of the independent variable to integrate over.

        x2 : float
            The maximum value of the independent variable to integrate over.

        Returns
        -------
        integral : float
            The integral of the model between x1 and x2.
        """
        return self.quadrature(x1, x2)


class AbstractSigmoidFitModel(AbstractFitModel):

    """Abstract base class for fit models representing sigmoids.
    """

    amplitude = FitParameter(1.)
    location = FitParameter(0.)
    scale = FitParameter(1.)

    @staticmethod
    @abstractmethod
    def shape(z: ArrayLike, *parameter_values: float) -> ArrayLike:
        """Abstract method for the normalized shape of the sigmoid model. Subclasses
        must implement this method.

        Arguments
        ---------
        z : array_like
            The normalized independent variable.

        parameter_values : float
            Additional shape parameters for the sigmoid.

        Returns
        -------
        array_like
            The value of the sigmoid shape function at z.
        """

    def evaluate(self, x: ArrayLike, amplitude: float, location: float,
                 scale: float, *parameter_values: float) -> ArrayLike:
        """Overloaded method for evaluating the model.

        Note if the scale is negative, we take the complement of the sigmoid function.
        """
        # pylint: disable=arguments-differ
        z = (x - location) / abs(scale)
        val = amplitude * self.shape(z, *parameter_values)
        return val if scale >= 0. else amplitude - val

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.):
        """Overloaded method.
        """
        # Simple initialization based on the data statistics.
        delta = np.diff(ydata)
        self.amplitude.set(ydata.max() - ydata.min())
        self.location.set(xdata[np.argmax(abs(delta))])
        scale = np.std(xdata) / 4.
        if delta.mean() < 0.:
            scale = -scale
        self.scale.set(scale)

    def default_plotting_range(self) -> Tuple[float, float]:
        """Overloaded method.

        By default the plotting range is set to be an interval centered on the
        location parameter, and extending for a number of scale units on each side.
        """
        # pylint: disable=arguments-differ
        left, right = 5., 5.
        location = self.location.value
        scale = self.scale.value
        return (location - left * scale, location + right * scale)


class AbstractCRVFitModel(AbstractFitModel):

    """Abstract base class for fit models based on continuous random variables.

    (Typically we will use this, in conjunction with the `wrap_rv_continuous`
    decorator, to wrap continuous random variables from scipy.stats).

    The general rule for the signature of scipy distributions is that they accept
    all the shape parameters first, and then loc and scale.
    This decorator creates a fit model class with the appropriate methods to
    Read dist.shapes (and numargs) to know the positional shape args.
    Assume loc and scale keywords are always supported.

    """

    amplitude = FitParameter(1.)
    location = FitParameter(0.)
    scale = FitParameter(1., minimum=0)
    _rv = None

    @classmethod
    def evaluate(cls, x, amplitude, location, scale, *args):
        """Overloaded method for evaluating the model.

        This takes the pdf of the underlying distribution and scales it by the amplitude.
        """
        # pylint: disable=arguments-differ
        return amplitude * cls._rv.pdf(x, *args, loc=location, scale=scale)

    @classmethod
    def primitive(cls, x, amplitude, location, scale, *args):
        """Overloaded method for evaluating the primitive of the model.

        Note this is not just a primitive, it is the actual cumulative distribution
        function (cdf) scaled by the amplitude. We keep the ``primitive()`` name for
        because in general not all the fit models are normalizable, and still we want
        to keep a common interface.
        """
        return amplitude * cls._rv.cdf(x, *args, loc=location, scale=scale)

    def support(self):
        """Return the support of the underlying distribution at the current
        parameter values.
        """
        _, location, scale, *args = self.parameter_values()
        return tuple(float(value) for value in self._rv.support(*args, loc=location, scale=scale))

    def ppf(self, p: ArrayLike):
        """Return the percent point function (inverse of cdf) of the underlying
        distribution for a given quantile at the current parameter values.

        Arguments
        ---------
        p : array_like
            The quantile(s) to evaluate the ppf at.
        """
        _, location, scale, *args = self.parameter_values()
        return self._rv.ppf(p, *args, loc=location, scale=scale)

    def median(self):
        """Return the median of the underlying distribution at the current parameter
        values.
        """
        _, location, scale, *args = self.parameter_values()
        return float(self._rv.median(*args, loc=location, scale=scale))

    def mean(self):
        """Return the mean of the underlying distribution at the current parameter
        values.
        """
        _, location, scale, *args = self.parameter_values()
        return float(self._rv.mean(*args, loc=location, scale=scale))

    def std(self):
        """Return the standard deviation of the underlying distribution at the current parameter
        values.
        """
        _, location, scale, *args = self.parameter_values()
        return float(self._rv.std(*args, loc=location, scale=scale))

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
        _, location, scale, *args = self.parameter_values()
        return self._rv.rvs(*args, loc=location, scale=scale, size=size, random_state=random_state)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        """Overloaded method.

        This is tailored on unimodal distributions, where we start from the
        basic statistics (average, standard deviation and area) of the input sample
        and try to match the amplitude, location and scale of the distribution
        to be fitted. No attempt is made at setting the shape parameters (if any).
        """
        # Calculate the average, standard deviation, and integral of the input data.
        location = np.average(xdata, weights=ydata)
        scale = np.sqrt(np.average((xdata - location)**2, weights=ydata))
        amplitude = scipy.integrate.trapezoid(ydata, xdata)
        # If the underlying distribution has a finite standard deviation
        # we can rescale the scale parameter accordingly. Note that this is
        # independent of the current location and scale, and only depends on the
        # shape of the distribution.
        std = self.std()
        if not np.isinf(std) and not np.isnan(std):
            scale = scale * self.scale.value / std
        # If the underlying distribution has a finite mean we can shift
        # the location parameter accordingly. Note this depends on the
        # current value of the scale parameter, which is why we do this after
        # rescaling it.
        mean = self.mean()
        if not np.isinf(mean) and not np.isnan(mean):
            delta = (mean - self.location.value) * scale / self.scale.value
            location -= delta
        # And we are good to go!
        self.location.init(location)
        self.scale.init(scale)
        self.amplitude.init(amplitude)

    def default_plotting_range(self) -> Tuple[float, float]:
        """Overloaded method.

        Note we have access to all the goodies of a scipy.stats.rv_continuous object
        here (e.g., the support of the function, and the mean and standard deviation
        when they are finite), so we can be fairly clever in setting up a generic method
        that works out of the box in many cases.
        """
        # pylint: disable=arguments-differ
        # If the distribution has finite support, use it.
        minimum, maximum = self.support()
        if np.isfinite(minimum) and np.isfinite(maximum):
            return (minimum, maximum)
        # Otherwise use the underlying ppf. First thing first, we need some measure of
        # the center and width of the distribution. The median should be always defined,
        # as it only relies on the ppf, i.e., the pdf being normalizable. For the width
        # of the distribution we try the standard deviation first, but if that is not
        # defined we fall back to using the scale parameter.
        center = self.median()
        width = self.std()
        if not np.isfinite(width):
            width = self.scale.value
        # Now use the ppf to get a reasonable plotting range.
        alpha = 0.005
        padding = 1.5 * width
        minimum = max(minimum, center - 5. * width)
        maximum = min(maximum, center + 5. * width)
        left = max(self.ppf(alpha) - padding, minimum)
        right = min(self.ppf(1. - alpha) + padding, maximum)
        return (left, right)

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
        if axes is None:
            axes = plt.gca()
        kwargs.setdefault("label", self.label)
        if fit_output:
            kwargs["label"] = f"{kwargs['label']}\n{self._format_fit_output(Format.LATEX)}"
        x = self._plotting_grid()
        axes.plot(x, self(x), **kwargs)
        x0 = self.mean()
        # If we are not plotting the mean, or if the mean is not finite, just plot the model
        # and return.
        if not plot_mean or not np.isfinite(x0):
            return
        # Otherwise plot the model and overplot the mean with a dot.
        color = last_line_color(axes)
        y0 = self(x0)
        axes.plot(x0, y0, "o", ms=5., color=matplotlib.rcParams["figure.facecolor"])
        axes.plot(x0, y0, "o", ms=1.5, color=color)


class PhonyCRVFitModel:

    """Phony class to provide a mechanism not to break everything when a particular
    scipy.stats distribution is not available in a given scipy version.
    """

    def __init__(self, scipy_version: str) -> None:
        """Constructor.
        """
        msg = f"The {self.__class__.__name__} distribution is only available in " \
              f"scipy >= {scipy_version}."
        raise NotImplementedError(msg)


def wrap_rv_continuous(rv, **shape_parameters) -> type:

    """Decorator to wrap a scipy.stats.rv_continuous object into a fit model.

    This is fairly minimal, and basically accounts to adding all the necessary shape
    parameters to the underlying fit model class. Note the name of the parameters is
    inferred from the rv.shapes attribute, and each shape parameter is set to 1. by
    default (with a minimum of 0.) unless this is overridden via the shape_parameters
    argument.

    Arguments
    ---------
    rv : scipy.stats.rv_continuous
        The scipy.stats.rv_continuous object to wrap.

    shape_parameters : dict, optional
        Additional shape parameters to be setup with non-default FitParameter
        objects (e.g., to set different minimum/maximum values).
    """

    def _wrapper(cls: type):
        """Wrapper function---cache the underlying continuos random variable, and
        add all the necessary shape parameters to the class.
        """
        # pylint: disable=protected-access
        cls._rv = rv
        if rv.numargs > 0:
            for name in rv.shapes.split(", "):
                parameter = shape_parameters.get(name, FitParameter(1., minimum=0.))
                setattr(cls, name, parameter)
        return cls

    return _wrapper


def line_forest(*energies: float) -> Callable[[type], type]:

    """Decorator to build a line forest fit model.

    A line forest is a collection of spectral lines at known energies, each
    with an independent amplitude, all sharing a common energy scale and
    with a line width (sigma) that scales as the square root of the line
    energy.

    This decorator is simply adding a class attribute to store the line
    energies, and creating all the necessary FitParameter objects.

    While the decorator is agnostic as to what is the actual line shape,
    the GaussianForestBase class is a good example of how to use this
    decorator to build a line forest fit model.

    Arguments
    ---------
    energies : float
        The energies of the lines comprised in the forest. (These are typically
        provided in physical units, e.g., keV, whereas the energy scale
        parameters determines the conversion between the energy and whatever
        units the fit model is actually evaluated in. e.g., ADC counts).
    """
    def _wrapper(cls: type):
        # pylint: disable=protected-access
        cls.energies = energies
        cls.amplitude = FitParameter(1., minimum=0.)
        for i in range(1, len(energies)):
            setattr(cls, f'intensity{i}', FitParameter(0.5/(len(energies) - 1), minimum=0.,
                                                       maximum=1.))
        cls.energy_scale = FitParameter(1., minimum=0.)
        cls.sigma = FitParameter(1., minimum=0.)

        return cls

    return _wrapper


class GaussianForestBase(AbstractFitModel):

    """Abstract base model representing a forest of Gaussian spectral lines
    at fixed energies.

    Concrete models needs to be decorated with the `@line_forest` decorator,
    specifying the energies of the lines included in the forest.

    Each peak corresponds to a known energy, and the model allows for
    fitting the amplitudes, a global energy scale, and a common width
    (sigma) that scales as the square root of the line energy, as it is
    common to observe in particle detectors.
    """

    def evaluate(self, x: ArrayLike, *parameter_values) -> ArrayLike:
        # pylint: disable=no-member
        # pylint: disable=arguments-differ
        amplitude, *intensities, energy_scale, sigma = parameter_values
        intensities = [1. - sum(intensities)] + list(intensities)
        y = sum(
            amplitude * intensity * scipy.stats.norm.pdf(
                x,
                loc=energy / energy_scale,
                scale=sigma / np.sqrt(energy / self.energies[0]))
                for intensity, energy in zip(intensities, self.energies)
        )
        return y

    def freeze(self, model_function, **constraints) -> Callable:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        if not constraints:
            return model_function
        return AbstractFitModel.freeze(self._wrap_evaluate(), **constraints)

    def _intensities(self):
        """Return the current values of the line intensities for the forest,
        properly normalized to one.
        """
        # pylint: disable=no-member
        intensities = [getattr(self, f"intensity{i}").value for i in range(1, len(self.energies))]
        intensities = [1. - sum(intensities)] + intensities
        return intensities

    def fwhm(self):
        """Calculate the FWHM of the main line of the forest.
        """
        # pylint: disable=no-member
        return SIGMA_TO_FWHM * self.sigma.ufloat()

    def rvs(self, size: int = 1, random_state=None):
        # pylint: disable=no-member
        """Generate random variates from the underlying distribution at the current
        parameter values.

        Arguments
        ---------
        size : int, optional
            The number of random variates to generate (default 1).

        random_state : int or np.random.Generator, optional
            The random seed or generator to use (default None).
        """
        rng = np.random.default_rng(random_state)
        vals = rng.choice(self.energies, size=size, p=self._intensities())
        loc = vals / self.energy_scale.value
        scale = self.sigma.value / np.sqrt(vals / self.energies[0])
        return scipy.stats.norm.rvs(loc=loc, scale=scale, random_state=rng)

    def init_parameters(self, xdata: ArrayLike, ydata: ArrayLike, sigma: ArrayLike = 1.) -> None:
        # pylint: disable=no-member
        """Overloaded method.
        """
        mu0 = xdata[np.argmax(ydata)]
        self.amplitude.init(scipy.integrate.trapezoid(ydata, xdata))
        self.energy_scale.init(self.energies[0] / mu0)
        self.sigma.init(np.sqrt(np.average((xdata - mu0)**2, weights=ydata)))

    def fit_iterative(self, xdata: Union[ArrayLike, Histogram1d], ydata: ArrayLike = None, *,
            p0: ArrayLike = None, sigma: ArrayLike = None, num_sigma_left: float = 2.,
            num_sigma_right: float = 2., num_iterations: int = 2, **kwargs) -> "FitStatus":
        """Fit iteratively line forest spectrum data within a given number of sigma around the
        peaks.

        This function performs a first round of fit to the data (either a histogram or
        scatter plot data) and then repeats the fit iteratively, limiting the fit range
        to a specified interval defined in terms of deviations (in sigma) around the peaks.

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
            The number of sigma on the left of the first peak to be used to define the
            fitting range.

        num_sigma_right : float
            The number of sigma on the right of the last peak to be used to define the
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
        # pylint: disable=no-member
        fit_status = self.fit(xdata, ydata, p0=p0, sigma=sigma, **kwargs)
        for i in range(num_iterations):
            _xmin = self.energies[0] / self.energy_scale.value - num_sigma_left * self.sigma.value
            _xmax = self.energies[-1] / self.energy_scale.value + num_sigma_right * \
                  (self.sigma.value / np.sqrt(self.energies[-1] / self.energies[0]))
            kwargs.update(xmin=_xmin, xmax=_xmax)
            try:
                fit_status = self.fit(xdata, ydata, p0=self.free_parameter_values(),
                                      sigma=sigma, **kwargs)
            except RuntimeError as exception:
                raise RuntimeError(f"Exception after {i + 1} iteration(s)") from exception
        return fit_status

    def default_plotting_range(self) -> Tuple[float, float]:
        # pylint: disable=no-member, arguments-differ
        """Overloaded method.
        """
        emin = min(self.energies) / self.energy_scale.value
        emax = max(self.energies) / self.energy_scale.value
        return (emin - 5 * (self.sigma.value / np.sqrt(min(self.energies) / self.energies[0])),
                emax + 5 * (self.sigma.value / np.sqrt(max(self.energies) / self.energies[0])))

    def plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False,
             plot_components: bool = True, **kwargs) -> matplotlib.axes.Axes:
        # pylint: disable=no-member
        """
        Overloaded method for plotting the model.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes on which to plot the model. If None, uses the current axes.

        fit_output : bool, optional
            If True, displays the fit output on the legend. Default is False.

        plot_components : bool, optional
            If True, plots the individual components of the model as dashed lines.
            Default is True.

        kwargs
            Additional keyword arguments passed to the parent class.

        Returns
        -------
        None
        """
        axes = super().plot(axes, fit_output=fit_output, **kwargs)
        x = self._plotting_grid()
        if plot_components:
            for intensity, energy in zip(self._intensities(), self.energies):
                amplitude = self.amplitude.value * intensity
                loc = energy / self.energy_scale.value
                scale = self.sigma.value / np.sqrt(energy / self.energies[0])
                y = amplitude * scipy.stats.norm.pdf(x, loc=loc, scale=scale)
                axes.plot(x, y, label=None, ls="--")


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
        # Note we set the _components attribute before calling the superclass
        # constructor, as there is at least a bit (setting the default label
        # to the model name) where we need the components to be defined.
        self._components = components
        super().__init__()

    def name(self) -> str:
        """Return the model name.
        """
        return " + ".join(component.name() for component in self._components)

    def __getitem__(self, index: int) -> AbstractFitModel:
        """Return the component at the given index.

        Arguments
        ---------
        index : int
            The index of the component to return.

        Returns
        -------
        component : AbstractFitModel
            The component at the given index.
        """
        return self._components[index]

    def __len__(self) -> int:
        """Return the sum of `all` the fit parameters in the underlying models.
        """
        return sum(len(component) for component in self._components)

    def __iter__(self) -> Iterator[FitParameter]:
        """Iterate over `all` the parameters of the underlying components.
        """
        return chain(*self._components)

    def freeze(self, model_function, **constraints) -> Callable:
        """Overloaded method.

        This is a tricky one, for two distinct reasons: (i) for a FitModelSum object
        evaluate() is not a static method, as it needs to access the list of components
        to sum over; (ii) since components can be added at runtime, the original
        signature of the function is generic, so we need to build a new signature that
        reflects the actual parameters of the model when we actually want to use it in a
        fit. In order to make this work, when freezing parameters we build a wrapper
        around evaluate() with the correct signature, and pass it downstream to the
        static freeze() method of the parent class AbstractFitModel.
        """
        # pylint: disable=arguments-differ
        if not constraints:
            return model_function
        return AbstractFitModel.freeze(self._wrap_evaluate(), **constraints)

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

    def integral(self, x1: float, x2: float) -> float:
        """Calculate the integral of the model between x1 and x2.

        This is implemented as the sum of the integrals of the components.

        Arguments
        ---------
        x1 : float
            The minimum value of the independent variable to integrate over.

        x2 : float
            The maximum value of the independent variable to integrate over.

        Returns
        -------
        integral : float
            The integral of the model between x1 and x2.
        """
        return sum(component.integral(x1, x2) for component in self._components)

    def plot(self, axes: matplotlib.axes.Axes = None, fit_output: bool = False,
             plot_components: bool = True, **kwargs) -> matplotlib.axes.Axes:
        """
        Overloaded method for plotting the model.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes on which to plot the model. If None, uses the current axes.

        fit_output : bool, optional
            If True, displays the fit output on the legend. Default is False.

        plot_components : bool, optional
            If True, plots the individual components of the model as dashed lines.
            Default is True.

        kwargs
            Additional keyword arguments passed to the parent class.

        Returns
        -------
        None
        """
        axes = super().plot(axes, fit_output=fit_output, **kwargs)
        # color = last_line_color(axes)
        x = self._plotting_grid()
        if plot_components:
            for component in self._components:
                axes.plot(x, component(x), label=None, ls="--")#, color=color)


    def _format_fit_output(self, spec: str) -> str:
        """String formatting for fit output.

        Arguments
        ---------
        spec : str
            The format specification.

        Returns
        -------
        text : str
            The formatted string.
        """
        text = ""
        if self.status is not None:
            text = f"{text}{format(self.status, spec)}\n"
        for component in self._components:
            text = f"{text}[{component.name()}]\n"
            for parameter in component:
                text = f"{text}{format(parameter, spec)}\n"
        return text.strip("\n")

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
        return f"{self.name()}\n{self._format_fit_output(spec)}"

    def __add__(self, other: AbstractFitModel) -> "FitModelSum":
        """Implementation of the model sum (i.e., using the `+` operator).

        Note that, in the spirit of keeping the interfaces as simple as possible,
        we are not implementing in-place addition (i.e., `+=`), and we only
        allow ``AbstractFitModel`` objects (not ``FitModelSum``) on the right
        hand side, which is all is needed to support the sum of an arbitrary
        number of models.
        """
        return self.__class__(*self._components, other)
