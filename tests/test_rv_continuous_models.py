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
import sys

import numpy as np

from aptapy.models import (
    Alpha,
    Anglit,
    Argus,
    Beta,
    BetaPrime,
    Bradford,
    Burr,
    Burr12,
    Chi,
    Chisquare,
    Cosine,
    CrystalBall,
    DoubleGamma,
    Fisk,
    Gaussian,
    GeneralizedLogistic,
    GeneralizedNormal,
    Landau,
    LogNormal,
    Lorentzian,
    Moyal,
)
from aptapy.hist import Histogram1d
from aptapy.plotting import plt, setup_gca

_EPSILON = sys.float_info.epsilon


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
    print(model)

    # Generate a random sample and fill a histogram.
    sample = model.random_sample(100000, 313)
    histogram = Histogram1d(np.linspace(xmin, xmax, 100)).fill(sample)
    histogram.plot(label="Random sample")

    # Run the parameter initialization on the histogram data and plot the
    # initial guess for the model. Note we reset the parameters to their default
    # values before running the initialization, so that we are effectively testing
    # what happens in real life.
    model.set_parameters(1., 0., 1.)
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


def _test_model_shape(model_class: type, shape_parameters: Sequence[float] = (_EPSILON, 1., 2., 5.)):
    """Test the shape of a given fit model.

    This creates a model of the given class, and plots its shape for different
    values of its shape parameters.

    Arguments
    ----------
    model_class: type
        The model class to be tested.
    """
    model = model_class()
    x = np.linspace(*model.plotting_range(), 250)
    for shape in shape_parameters:
        model.set_parameters(1., 0., 1., shape)
        print(f"Shape = {shape} -> mean: {model.mean():.3f}, std: {model.std():.3f}")
        plt.plot(x, model(x), label=f"shape = {shape}")
    setup_gca(xlabel="x", ylabel="pdf(x)")
    plt.legend()


def test_alpha():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Alpha)
    plt.figure(f"{inspect.currentframe().f_code.co_name} shape")
    _test_model_shape(Alpha)


def test_anglit():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Anglit)


def test_argus():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Argus)
    plt.figure(f"{inspect.currentframe().f_code.co_name} shape")
    _test_model_shape(Argus)


def test_beta():
   plt.figure(f"{inspect.currentframe().f_code.co_name}")
   _test_model_base(Beta, (1., 10., 2., 2.31, 0.627))


def test_beta_prime():
   plt.figure(f"{inspect.currentframe().f_code.co_name}")
   _test_model_base(BetaPrime, (1., 10., 2., 2.31, 0.627))


def test_bradford():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Bradford)


def test_burr():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Burr)


def test_chi():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Chi)


def test_chisquare():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Chisquare)


def test_cosine():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Cosine)


def test_crystal_ball():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(CrystalBall, (10., 10., 2., 1., 2.))


def test_double_gamma():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(DoubleGamma)


# def test_fisk():
#     plt.figure(f"{inspect.currentframe().f_code.co_name}")
#     _test_model_base(Fisk)


def test_gaussian():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Gaussian)


def test_generalized_logistic():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(GeneralizedLogistic)


def test_generalized_normal():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(GeneralizedNormal)


def test_landau():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Landau)


def test_log_normal():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(LogNormal)


def test_lorentzian():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Lorentzian)


def test_moyal():
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    _test_model_base(Moyal)


if __name__ == "__main__":
    test_alpha()
    test_argus()
    plt.show()
