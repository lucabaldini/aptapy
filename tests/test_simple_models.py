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

"""Unit tests for the modeling module.
"""

import numpy as np
import scipy.stats

from aptapy import models
from aptapy.plotting import plt


def _test_model_base(model_class: type, *parameter_values: float, sigma: float = 0.1,
                     num_sigma: float = 5., figure_name: str = None, **kwargs) -> None:
    """Basic tests for the Model base class.

    Arguments
    ----------
    model_class: type
        The model class to be tested.

    parameter_values: float
        The ground-truth parameter values to be used for generating the random dataset.

    sigma: float
        The standard deviation of the noise to be added to the generated dataset.

    num_sigma: float
        The number of standard deviations within which the initial parameter guesses
        should be compatible with the ground-truth values.

    kwargs: dict
        Additional keyword arguments to be passed to the model constructor.
    """
    if figure_name is None:
        figure_name = model_class.__name__
    plt.figure(figure_name)
    # Create the model and set the basic parameters.
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]", **kwargs)
    # Saving the default parameters.
    default_parameters = model.parameter_values()
    # Setting the test parameters.
    model.set_parameters(*parameter_values)
    print(model)

    # Generate a random dataset.
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    #color = last_line_color()

    # Reset the model to a generic state before initializing the parameters.
    model.set_parameters(*default_parameters)
    model.init_parameters(xdata, ydata, sigma)
    model.plot(label="Initial guess", ls="--", color="gray")
    initial_values = model.parameter_values()
    print(f"Initial values: {initial_values}")
    model.fit(xdata, ydata, sigma=sigma)
    model.plot(fit_output=True)
    plt.legend()
    for param, guess, ground_truth in zip(model, initial_values, parameter_values):
        # Note that we can programmatically relax the test on the initial guess by
        # increasing num_sigma, since for the majority of models the init_parameters()
        # is not meant to provide initial guess statistically compatible with the ground truth
        # truth. The final best-fit parameters, on the other hand, should be within a
        # reasonable number of sigma from the truth.
        assert param.compatible_with(guess, num_sigma)
        assert param.compatible_with(ground_truth, 5.)


def test_constant():
    _test_model_base(models.Constant, 5.)


def test_line():
    _test_model_base(models.Line, 2., 5.)


def test_quadratic():
    _test_model_base(models.Quadratic, 1., 2., 16.)


def test_cubic():
    _test_model_base(models.Cubic, 1., 2., 3., 4.)


def test_polynomial():
    _test_model_base(models.Polynomial, 1., -2., 3., -4., 5., degree=4)


def test_power_law():
    _test_model_base(models.PowerLaw, 10., -2.)
    _test_model_base(models.PowerLaw, 10., -1.)
    _test_model_base(models.PowerLaw, 10., -3., figure_name="PowerLaw pivot", pivot=10.)


def test_exponential():
    _test_model_base(models.Exponential, 5., 2., location=10.)


def test_exponential_complement():
    _test_model_base(models.ExponentialComplement, 5., 2., location=10.)


def test_stretched_exponential():
    _test_model_base(models.StretchedExponential, 5., 2., 0.5, num_sigma=50.)


def test_stretched_exponential_complement():
    _test_model_base(models.StretchedExponentialComplement, 5., 2., 0.5, num_sigma=50.)


def test_line_forest():
    _test_model_base(models.Fe55Forest, 10., 0.2, 1., 0.2, sigma=0.5, num_sigma=500.)


def test_probit():
    """Custom unit test for the Probit model.
    """
    # Cache the test parameters.
    offset = 0.5
    sigma = 0.12
    sigma_y = 0.01

    model = models.Probit()
    x = np.linspace(0., 1., 100)
    y = model.evaluate(x, offset, sigma)
    # Make sure that we got the ppf of the gaussian right.
    assert np.allclose(y, scipy.stats.norm.ppf(x, loc=offset, scale=sigma))

    # Standard test.
    _test_model_base(models.Probit, offset, sigma, sigma=sigma_y)

    # Note this is not using the generic test function since we want to
    # freeze some parameters during the fit.
    plt.figure("Probit typical")
    model.set_parameters(offset, sigma)
    xdata, ydata = model.random_fit_dataset(sigma_y, seed=313)
    plt.errorbar(xdata, ydata, sigma_y, fmt="o", label="Random data")
    model.offset.freeze(offset)
    model.fit(xdata, ydata, sigma=sigma_y)
    assert model.sigma.compatible_with(sigma, 5.)
    model.plot(fit_output=True)
    plt.legend()
