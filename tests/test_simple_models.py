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

"""Unit tests for the modeling module.
"""

import inspect
from typing import Callable, Sequence

import numpy as np
import pytest

from aptapy import models
from aptapy.plotting import plt, setup_gca


def _test_model_base(model_class: type, *parameter_values: float, sigma: float = 0.1,
                     num_sigma: float = 5.):
    """Basic tests for the Model base class.
    """
    plt.figure(model_class.__name__)
    # Create the model and set the basic parameters.
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]")
    model.set_parameters(*parameter_values)
    print(model)

    # Generate a random dataset.
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    #color = last_line_color()

    # Reset the model to a generic state before initializing the parameters.
    default_parameters = tuple(1. for _ in model)
    model.set_parameters(*default_parameters)
    model.init_parameters(xdata, ydata, sigma)
    model.plot(label="Initial guess", ls="--", color="gray")
    initial_values = model.parameter_values()
    print(f"Initial values: {initial_values}")
    model.fit(xdata, ydata, sigma=sigma)
    model.plot(fit_output=True)
    for param, guess, ground_truth in zip(model, initial_values, parameter_values):
        # Note that we can programmatically relax the test on the initial guess by
        # increasing num_sigma, since for the majority of models the init_parameters()
        # is not meant to provide initial guess statistically compatible with the ground truth
        # truth. The final best-fit parameters, on the other hand, should be within a
        # reasonable number of sigma from the truth.
        assert param.compatible_with(guess, num_sigma)
        assert param.compatible_with(ground_truth, 5.)

    #xmin, xmax = model.default_plotting_range()
    #setup_gca(xmin=xmin, xmax=xmax)
    plt.legend()


def test_constant():
    _test_model_base(models.Constant, 5.)


def test_line():
    _test_model_base(models.Line, 2., 5.)


#def test_quadratic():
#    _test_model_base(models.Quadratic, 1., 2., 16.)


# def test_power_law():
#     """Test the PowerLaw model---note we do this for two different indices.
#     """
#     for index in (-2., -1.):
#         plt.figure(f"{inspect.currentframe().f_code.co_name}_index{abs(index)}")
#         prefactor = 10.
#         if index == -1.:
#             def integral(xmin, xmax, prefactor=prefactor):
#                 return prefactor * np.log(xmax / xmin)
#         else:
#             def integral(xmin, xmax, prefactor=prefactor, index=index):
#                 return (prefactor / (index + 1.) * (xmax**(index + 1.) - xmin**(index + 1.)))
#         _test_model_base(models.PowerLaw, (prefactor, index), integral)


# def test_exponential():
#     """Test the Exponential model.
#     """
#     plt.figure(f"{inspect.currentframe().f_code.co_name}")
#     prefactor, scale = 10., 2.
#     def integral(xmin, xmax):
#         return prefactor * scale * (np.exp(-xmin / scale) - np.exp(-xmax / scale))
#     _test_model_base(models.Exponential, (prefactor, scale), integral)


# def test_exponential_complement():
#     """Test the ExponentialComplement model.
#     """
#     plt.figure(f"{inspect.currentframe().f_code.co_name}")
#     prefactor, scale = 10., 2.
#     _test_model_base(models.ExponentialComplement, (prefactor, scale,), None)


# def test_stretched_exponential():
#     """Test the StretchedExponential model.
#     """
#     plt.figure(f"{inspect.currentframe().f_code.co_name}")
#     prefactor, scale, gamma = 10., 2., 0.5
#     # The initialization of the parameters is pretty flaky in this case...
#     _test_model_base(models.StretchedExponential, (prefactor, scale, gamma), None, num_sigma=50.)


# def test_stretched_exponential_complement():
#     """Test the StretchedExponentialComplement model.
#     """
#     plt.figure(f"{inspect.currentframe().f_code.co_name}")
#     prefactor, scale, gamma = 10., 2., 0.5
#     # The initialization of the parameters is pretty flaky in this case...
#     _test_model_base(models.StretchedExponentialComplement, (prefactor, scale, gamma),
#                      None, num_sigma=50.)
