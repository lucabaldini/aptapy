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

"""Unit tests for all the sigmoids models.
"""

from aptapy import models
from aptapy.plotting import last_line_color, plt


def _test_base(model_class: type, location: float = 10., amplitude: float = 5.,
               num_sigma: float = 15.) -> None:
    """Basic tests for a given fit model.

    This cretates a model of the given class, sets the given ground truth parameter values,
    generates a random dataset from the model, runs the parameter initialization on the
    dataset, fits the model to the dataset, and checks that the fitted parameters are
    compatible with the ground truth values.

    Arguments
    ----------
    model_class: type
        The model class to be tested.

    location: float
        The ground truth location parameter value to be set in the model for the generation
        of the random dataset.

    scale: float
        The ground truth scale parameter value to be set in the model for the generation
        of the random dataset.

    num_sigma: float
        The number of sigma within which the initial parameter guesses should be
        compatible with the ground truth values.
    """
    # pylint: disable=R0801
    plt.figure(model_class.__name__)
    # Create the model and set the basic parameters.
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]")
    for scale in (2., -2.):
        # Setup the model.
        sigma = 0.025 * abs(amplitude)
        parameter_values = (amplitude, location, scale)
        model.set_parameters(*parameter_values)
        print(model)

        # Generate a random dataset.
        xdata, ydata = model.random_fit_dataset(sigma, seed=313)
        plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
        color = last_line_color()

        # Reset the model to a generic state before initializing the parameters.
        model.set_parameters(1., 0., 1.)
        model.init_parameters(xdata, ydata, sigma)
        model.plot(label="Initial guess", ls="--", color="gray")
        initial_values = model.parameter_values()
        print(f"Initial values: {initial_values}")
        model.fit(xdata, ydata, sigma=sigma)
        model.plot(color=color)
        for param, guess, ground_truth in zip(model, initial_values, parameter_values):
            # Note that we can programmatically relax the test on the initial guess by
            # increasing num_sigma, since for the majority of models the init_parameters()
            # is not meant to provide initial guess statistically compatible with the ground truth
            # truth. The final best-fit parameters, on the other hand, should be within a
            # reasonable number of sigma from the truth.
            assert param.compatible_with(guess, num_sigma)
            assert param.compatible_with(ground_truth, 5.)


def test_erf_sigmoid():
    _test_base(models.ErfSigmoid)


def test_logistic_sigmoid():
    _test_base(models.LogisticSigmoid)


def test_arctan():
    _test_base(models.Arctangent)


def test_hyperbolic_tangent():
    _test_base(models.HyperbolicTangent)
