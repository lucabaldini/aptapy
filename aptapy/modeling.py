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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import inspect
from numbers import Number
from typing import Tuple, Iterator

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties

from aptapy.typing import ArrayLike


@dataclass
class FitParameter:

    """Small class describing a fit parameter.
    """

    _name: str
    value: float
    error: float = None
    frozen: bool = False
    minimum: float = -np.inf
    maximum: float = np.inf

    def __post_init__(self) -> None:
        """Post-initialization.

        Here we basically cache the value that is given to the parameter at creation time
        so that we can later reset it to that value.
        """
        self._default_value = self.value

    def reset(self) -> None:
        """Reset the parameter.
        """
        self.error = None

    def is_bound(self) -> bool:
        """Return True if the parameter is bounded.
        """
        return not np.isinf(self.minimum) or not np.isinf(self.maximum)

    def ufloat(self) -> uncertainties.ufloat:
        """Return the parameter value and error as a ufloat object.
        """
        return uncertainties.ufloat(self.value, self.error)

    def __str__(self) -> str:
        """String formatting.

        This is meant to provide a more human-readable version of the parameter formatting
        than the default ``__repr__`` implementation from the dataclass decorator, and it
        is what is used in the actual printout of the fit parameters from a fit.
        """
        text = f'{self._name} ='
        if self.error is None:
            text = f'{text} {self.value}'
        else:
            text = f'{text} {self.ufloat()}'
        if self.frozen:
            text = f'{text} (frozen)'
        if self.is_bound():
            text = f'{text} [{self.minimum}--{self.maximum}]'
        return text


@dataclass
class FitStatus:

    """Small dataclass to hold the fit status.
    """

    chisquare: float
    dof: int
    # pvalue: float

    def __str__(self) -> str:
        """String formatting.
        """
        return f'chisquare = {self.chisquare:.2f} / {self.dof} dof'


class AbstractFitModel(ABC):

    """Abstract base class for a fit model.
    """

    def __init__(self):
        """Constructor.
        """
        self._parameters = []
        for name, annotation in self.__annotations__.items():
            if annotation is FitParameter:
                parameter = FitParameter(name, getattr(self, name, 1.))
                setattr(self, name, parameter)
                self._parameters.append(parameter)
        self.status = None
        self._fit_range = None

    def reset(self) -> None:
        """Reset all the fit parameters.
        """
        self.status = None
        self._fit_range = None
        for parameter in self:
            parameter.reset()

    def name(self) -> str:
        """Return the model name.
        """
        return self.__class__.__name__

    def __len__(self) -> int:
        """Overloaded method.
        """
        return len(self._parameters)

    def __iter__(self) -> Iterator[FitParameter]:
        """Iteration protocol.
        """
        return iter(self._parameters)

    def parameter_values(self) -> Tuple[float]:
        """Return the current parameter values.
        """
        return tuple(parameter.value for parameter in self)

    @staticmethod
    @abstractmethod
    def evaluate(x: ArrayLike, *parameter_values: Tuple[float]) -> ArrayLike:
        """Evaluate the model at a given value (or set of values) of the independent variable,
        for a given set of model parameters.

        Arguments
        ---------
        x : array_like
            The value(s) of the independent variable.

        *params : tuple of float
            The value of the model parameters.
        """

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the model at the current value of the parameters.
        """
        return self.evaluate(x, *self.parameter_values())

    def set_parameters(self, *values: float) -> None:
        """
        """
        for parameter, value in zip(self, values):
            parameter.value = value

    def init_parameters(self, x: ArrayLike, y: ArrayLike, sigma: ArrayLike) -> None:
        """
        """

    def _update_parameters(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        """Update the model parameters based on the output of the ``curve_fit()`` call.
        """
        for parameter, value, error in zip(self, popt, np.sqrt(pcov.diagonal())):
            parameter.value = value
            parameter.error = error

    def bounds(self) -> Tuple[ArrayLike, ArrayLike]:
        """Return the bounds on the fit parameters in a form that can be use by the
        fitting method.
        """
        return (tuple(parameter.minimum for parameter in self),
                tuple(parameter.maximum for parameter in self))

    def calculate_chisqure(self, xdata: np.ndarray, ydata: np.ndarray, sigma) -> float:
        """Calculate the chisquare of the fit to some input data with the current
        model parameters.
        """
        return float((((ydata - self(xdata)) / sigma)**2.).sum())

    def fit(self, xdata: ArrayLike, ydata: ArrayLike, p0: ArrayLike = None,
            sigma: ArrayLike = 1., absolute_sigma: bool = False, xmin: float = -np.inf,
            xmax: float = np.inf, **kwargs) -> None:
        """Fit a series of points.
        """
        # Reset the fit status and the fit parameters.
        self.reset()

        # Prepare the data. We want to make sure all the relevant things are numpy
        # arrays so that we can vectorize operations downstream, taking advantage of
        # the broadcast facilities.
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)
        # If we are fitting over a subrange, filter the input data.
        mask = np.logical_and(xdata >= xmin, xdata <= xmax)
        # (And, since we are at it, make sure we have enough degrees of freedom.)
        dof = int(mask.sum() - len(self))
        if dof < 0:
            raise RuntimeError('The model has no degrees of freedom')
        xdata = xdata[mask]
        ydata = ydata[mask]
        if not isinstance(sigma, Number):
            sigma = np.asarray(sigma)[mask]
        # Cache the fit range for later use.
        self._fit_range = (xdata.min(), xdata.max())

        # If we are not passing default starting points for the model parameters,
        # try and do something sensible.
        if p0 is None:
            self.init_parameters(xdata, ydata, sigma)
            p0 = self.parameter_values()

        # _kwargs = {parameter._name: parameter.value for parameter in self if parameter.frozen}
        # if _kwargs:
        #     func = self.freeze(self.evaluate, **_kwargs)
        #     p0 = p0[1:]
        #     print(_kwargs, func, p0)
        # else:
        #     func = self.evaluate

        # Do the actual fit.
        popt, pcov = curve_fit(self.evaluate, xdata, ydata, p0, sigma, absolute_sigma,
                               True, self.bounds(), **kwargs)
        self._update_parameters(popt, pcov)
        chisquare = self.calculate_chisqure(xdata, ydata, sigma)
        self.status = FitStatus(chisquare, dof)
        return self.status

    def default_plotting_range(self) -> Tuple[float, float]:
        """Return the default plotting range for the model.

        This can be reimplemnted in concrete models, and can be parameter-dependent
        (e.g., for a gaussian we might want to plot within 5 sigma from the mean by
        dafeault).
        """
        return (0., 1.)

    def _plotting_range(self, xmin: float = None, xmax: float = None,
                        fit_padding: float = 0.) -> Tuple[float, float]:
        """Convenience function trying to come up with the most sensible plot range
        for the model.
        """
        # If we have fitted the model to some data, we take the fit range and pad it
        # a little bit.
        if self._fit_range is not None:
            _xmin, _xmax = self._fit_range
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
        x = np.linspace(*self._plotting_range(), num_points)
        y = self(x)
        plt.plot(x, y)

    def __str__(self):
        """String formatting.
        """
        text = f'{self.__class__.__name__} ({self.status})\n'
        for parameter in self._parameters:
            text = f'{text}{parameter}\n'
        return text


class Constant(AbstractFitModel):

    """Constant model.
    """

    value: FitParameter = 1.

    @staticmethod
    def evaluate(x: ArrayLike, value: float) -> ArrayLike:
        return np.full(value, x.shape)


class Line(AbstractFitModel):

    """Linear model.
    """

    slope: FitParameter = 1.
    intercept: FitParameter = 0.

    @staticmethod
    def evaluate(x: ArrayLike, slope: float, intercept: float) -> ArrayLike:
        return slope * x + intercept


class PowerLaw(AbstractFitModel):

    """Power-law model.
    """

    prefactor: FitParameter = 1.
    index: FitParameter = -1.

    @staticmethod
    def evaluate(x: ArrayLike, prefactor: float, index: float) -> ArrayLike:
        return prefactor * x**index


class Gaussian(AbstractFitModel):

    """Gaussian model.
    """

    prefactor: FitParameter = 1.
    mean: FitParameter = 0.
    sigma: FitParameter = 1.

    @staticmethod
    def evaluate(x: ArrayLike, normalization: float, mean: float, sigma: float) -> ArrayLike:
        return normalization * np.exp(-0.5 * ((x - mean) / sigma) ** 2.)

    def default_plotting_range(self, num_sigma: int = 5) -> Tuple[float, float]:
        mean, half_width = self.mean.value, num_sigma * self.sigma.value
        return (mean - half_width, mean + half_width)
