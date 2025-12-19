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

"""Unit tests for all the fit models wrapping rv_continuous scipy objects.
"""

import contextlib

import numpy as np

from aptapy import models
from aptapy.plotting import plt, setup_gca


def _test_base(model_class: type, *shape_parameters, location: float = 10.,
               scale: float = 2., threshold: float = 0.001):
    """Basic tests for a given fit model.

    This creates a model of the given class, sets the given ground truth parameter values,
    generates a random sample from the model, fills a histogram with it, runs the parameter
    initialization on the histogram data, fits the model to the histogram data, and checks
    that the p-value of the fit is above the given threshold.

    Arguments
    ----------
    model_class: type
        The model class to be tested.

    shape_parameters: Sequence[float]
        The ground truth shape parameter values to be set in the model for the generation
        of the random sample.

    location: float
        The ground truth location parameter value to be set in the model for the generation
        of the random sample.

    scale: float
        The ground truth scale parameter value to be set in the model for the generation
        of the random sample.

    threshold: float
        The p-value threshold for the fit to be considered acceptable.
    """
    plt.figure(model_class.__name__)
    # Create the model and set the basic parameters.
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]")
    parameter_values = (1., location, scale, *shape_parameters)
    model.set_parameters(*parameter_values)
    xmin, xmax = model.plotting_range()
    print(model)

    # Generate a random histogram.
    edges = np.linspace(xmin, xmax, 101)
    histogram = model.random_histogram(edges, size=100000, random_state=313)
    histogram.plot(label="Random sample")

    # Run the parameter initialization on the histogram data and plot the
    # initial guess for the model. Note we reset the parameters to their default
    # values before running the initialization, so that we are effectively testing
    # what happens in real life.
    model.set_parameters(1., 0., 1.)
    model.init_parameters(histogram.bin_centers(), histogram.content, histogram.errors)
    model.plot(plot_mean=False, label="Initial guess", ls="--", color="gray")

    # Fit the model to the histogram data, check the chisquare and plot the result.
    # Note that, since we are fitting a random sample, we cannot test the
    # amplitude against the ground truth. We *could* check all the other parameter
    # values, but since this is a quick test, we just make sure that the p-value
    # is acceptable.
    status = model.fit(histogram)
    assert status.pvalue > threshold
    model.plot(fit_output=True)
    setup_gca(xmin=xmin, xmax=xmax)
    # plt.axvline(model.mean(), color="gray", ls="--")
    # plt.axvline(model.mean() + model.std(), color="gray", ls="--")
    plt.legend()


def test_alpha():
    _test_base(models.Alpha)


def test_anglit():
    _test_base(models.Anglit)


def test_arcsine():
    _test_base(models.Arcsine)


def test_argus():
    _test_base(models.Argus)


def test_beta():
    _test_base(models.Beta, 2.31, 0.627)


def test_beta_prime():
    _test_base(models.BetaPrime, 2., 4.)


def test_bradford():
    _test_base(models.Bradford, 1.)


def test_burr():
    _test_base(models.Burr, 4., 4.)


def test_burr12():
    _test_base(models.Burr12, 2., 2.)


def test_cauchy():
    _test_base(models.Cauchy)


def test_chi():
    _test_base(models.Chi, 8., location=0., scale=1.)


def test_chisquare():
    _test_base(models.Chisquare, 8., location=0., scale=1.)


def test_cosine():
    _test_base(models.Cosine)


def test_crystal_ball():
    _test_base(models.CrystalBall, 1., 2.)


def test_gaussian():
    _test_base(models.Gaussian)


def test_gibrat():
    """Gibrat distribution is only available in scipy >= 1.12.0
    """
    with contextlib.suppress(NotImplementedError):
        _test_base(models.Gibrat)


def test_gumbel_l():
    _test_base(models.GumbelL)


def test_gumbel_r():
    _test_base(models.GumbelR)


def test_half_cauchy():
    _test_base(models.HalfCauchy)


def test_half_logistic():
    _test_base(models.HalfLogistic)


def test_half_norm():
    _test_base(models.HalfNorm)


def test_hyper_secant():
    _test_base(models.HyperSecant)


def test_landau():
    """The Landau distribution is only available in scipy >= 1.15.1
    """
    with contextlib.suppress(NotImplementedError):
        _test_base(models.Landau)


def test_laplace():
    _test_base(models.Laplace)


def test_levy():
    _test_base(models.Levy)


def test_levy_l():
    _test_base(models.LevyL)


def test_logistic():
    _test_base(models.Logistic)


def test_log_normal():
    _test_base(models.LogNormal)


def test_lorentzian():
    _test_base(models.Lorentzian)


def test_maxwell():
    _test_base(models.Maxwell)


def test_moyal():
    _test_base(models.Moyal)


def test_nakagami():
    _test_base(models.Nakagami, 2.)


def test_normal():
    _test_base(models.Normal)


def test_rayleigh():
    _test_base(models.Rayleigh)


def test_semicircular():
    _test_base(models.Semicircular)


def test_student():
    _test_base(models.Student, 4.)


def test_wald():
    _test_base(models.Wald)
