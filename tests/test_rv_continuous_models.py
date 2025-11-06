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

"""Unit tests for all the fit models wrapping rv_continuous scipy objects.
"""


import inspect
from typing import Callable, Sequence

import numpy as np

from aptapy.models import (
    Alpha,
    Anglit,
    Argus,
    Beta,
    Gaussian,
    LogNormal,
    Lorentzian,
    Moyal,
)
from aptapy.hist import Histogram1d
from aptapy.plotting import plt, setup_gca


def _test_model_base(model_class: type, parameter_values: Sequence[float] = (1., 10., 2.),
                     threshold: float = 0.001):
    """Basic tests for a given fit model.

    This creates a model of the given class, sets the given ground truth parameter values,
    generates a random sample from the model, fills a histogram with it, runs the parameter
    initialization on the histogram data, fits the model to the histogram data, and checks
    that the p-value of the fit is above the given threshold.

    Arguments
    ----------
    model_class: type
        The model class to be tested.

    parameter_values: Sequence[float]
        The ground truth parameter values to be set in the model for the generation
        of the random sample.

    threshold: float
        The p-value threshold for the fit to be considered acceptable.
    """
    # Create the model and set the basic parameters.
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]")
    model.set_parameters(*parameter_values)
    xmin, xmax = model.plotting_range()

    # Generate a random sample and fill a histogram.
    sample = model.random_sample(100000, 313)
    histogram = Histogram1d(np.linspace(xmin, xmax, 100)).fill(sample)
    histogram.plot(label="Random sample")

    # Run the parameter initialization on the histogram data and plot the
    # initial guess for the model.
    model.init_parameters(histogram.bin_centers(), histogram.content, histogram.errors)
    model.plot(label="Initial guess", ls="--", color="gray")

    # Fit the model to the histogram data, check the chisquare and plot the result.
    # Note that, since we are fitting a random sample, we cannot test the
    # amplitude against the ground truth. We *could* check all the other parameter
    # values, but since this is a quick test, we just make sure that the p-value
    # is acceptable.
    status = model.fit_histogram(histogram)
    assert status.pvalue > threshold
    model.plot(fit_output=True)
    setup_gca(xmin=xmin, xmax=xmax)
    plt.axvline(model.mean(), color="gray", ls="--")
    plt.axvline(model.mean() + model.std(), color="gray", ls="--")
    plt.legend()


def test_alpha():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Alpha)


def test_anglit():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Anglit)


def test_argus():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Argus)


#def test_beta():
#    plt.figure(f"{inspect.currentframe().f_code.co_name}")
#    _test_model_base(Beta, (1., 10., 2., 2.31, 0.627))


def test_gaussian():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Gaussian)


def test_log_normal():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(LogNormal)


def test_lorentzian():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Lorentzian)


def test_moyal():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Moyal)
