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

import inspect

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import FitParameter, FitStatus
from aptapy.models import Constant, Exponential, Fe55Forest, Gaussian, Line
from aptapy.plotting import plt

_RNG = np.random.default_rng(313)

TEST_HISTOGRAM = Histogram1d(np.linspace(-5., 5., 100), label="Random data")
TEST_HISTOGRAM.fill(_RNG.normal(size=100000))
NUM_SIGMA = 4.


def test_fit_parameter():
    """Test the FitParameter class and the various interfaces.
    """
    parameter = FitParameter(1., 'normalization')
    assert not parameter.is_bound()
    assert not parameter.frozen
    print(parameter)
    parameter.set(3., 0.1)
    assert parameter.value == 3.
    assert parameter.error == 0.1
    print(parameter)
    parameter.set(4.)
    assert parameter.value == 4.
    assert parameter.error is None
    print(parameter)
    parameter = FitParameter(1., 'normalization', 0.1)
    assert not parameter.frozen
    assert not parameter.is_bound()
    assert parameter.ufloat().n == 1.
    assert parameter.ufloat().s == 0.1
    print(parameter)
    parameter = FitParameter(1., 'normalization', _frozen=True)
    assert not parameter.is_bound()
    assert parameter.frozen
    print(parameter)
    parameter.thaw()
    assert not parameter.frozen
    print(parameter)
    parameter = FitParameter(1., 'normalization', minimum=0.)
    assert parameter.is_bound()
    assert not parameter.frozen
    print(parameter)
    parameter.freeze(3.)
    assert parameter.value == 3.
    assert parameter.error is None
    assert parameter.frozen
    assert parameter.ufloat().n == 3.
    assert np.isclose(parameter.ufloat().s, 0.)
    print(parameter)


def test_fit_status():
    """Test the FitStatus class.
    """
    status = FitStatus()
    assert not status.valid()
    chisquare = 10.5
    dof = 8
    status.update(np.array([1., 2.]), np.array([[0.1, 0.], [0., 0.2]]), chisquare, dof)
    assert status.valid()
    assert status.chisquare == chisquare
    assert status.dof == dof
    par_sum = status.correlated_pars[1] + status.correlated_pars[0]
    assert par_sum.n == 3.
    assert np.isclose(par_sum.s, np.sqrt(0.1 + 0.2))
    status.reset()
    status.update(np.array([1., 2.]), np.array([[0.1, 0.05], [0.05, 0.2]]), chisquare, dof)
    par_sum = status.correlated_pars[1] + status.correlated_pars[0]
    assert np.isclose(par_sum.s, np.sqrt(0.1 + 0.2 + 2 * 0.05))
    status.reset()
    assert not status.valid()


def test_model_parameters():
    """We want to make sure that every model get its own set of parameters that can
    be varied independently.
    """
    g1 = Gaussian()
    g2 = Gaussian()
    p1 = g1.amplitude
    p2 = g2.amplitude
    print(p1, id(p1))
    print(p2, id(p2))
    assert p1 == p2
    assert id(p1) != id(p2)


def test_gaussian_fit():
    """Simple Gaussian fit.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    TEST_HISTOGRAM.plot()
    model.fit(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mu.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_subrange():
    """Gaussian fit in a subrange.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    TEST_HISTOGRAM.plot()
    model.fit(TEST_HISTOGRAM, xmin=-2., xmax=2.)
    model.plot(fit_output=True)
    assert model.mu.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_bound():
    """Test a bounded fit.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    model.mu.minimum = 0.05
    model.mu.value = 0.1
    TEST_HISTOGRAM.plot()
    model.fit(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mu.value >= model.mu.minimum
    plt.legend()


def test_gaussian_fit_frozen():
    """Gaussian fit with frozen parameters.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    # Calculate the normalization from the histogram.
    model.amplitude.freeze(TEST_HISTOGRAM.area())
    TEST_HISTOGRAM.plot()
    model.fit(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mu.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_frozen_and_bound():
    """And yet more complex: Gaussian fit with frozen and bound parameters.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    print(model)
    model.sigma.freeze(1.1)
    model.mu.minimum = 0.05
    model.mu.value = 0.1
    TEST_HISTOGRAM.plot()
    model.fit(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mu.value >= model.mu.minimum
    assert model.sigma.value == 1.1
    plt.legend()


def test_gaussian_fit_iterative_hist():
    """Test an iterative fit for a Gaussian histogram.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    print(model)
    TEST_HISTOGRAM.plot()
    model.fit_iterative(TEST_HISTOGRAM, num_iterations=3, num_sigma_left=3., num_sigma_right=3.,
                        xmin=-5., xmax=5.)
    model.plot(fit_output=True)
    plt.legend()
    assert model.mu.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001


def test_gaussian_fit_iterative_scatter():
    """Test an iterative fit for a Gaussian scatter plot.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian(xlabel="x [a.u.]", ylabel="y [a.u.]")
    print(model)
    sigma = 0.01
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    model.fit_iterative(xdata, ydata, sigma=sigma, num_iterations=3)
    model.plot(fit_output=True)
    plt.legend()
    assert model.mu.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001


def test_sum_gauss_line():
    """Test the sum of of two models.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    hist = TEST_HISTOGRAM.copy()
    u = _RNG.random(100000)
    x = 5. - 10. * np.sqrt(1 - u)
    hist.fill(x)
    model = Gaussian() + Line()
    hist.plot()
    model.fit(hist)
    model.plot(fit_output=True)
    plt.legend()


def test_multiple_sum():
    """Test the sum of multiple models.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian() + Line() + Constant()
    model.set_plotting_range(-5., 5.)
    model.plot()
    plt.legend()


def test_sum_frozen():
    """Test fitting the sum of two models with a frozen parameter.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    error = 0.05
    x = np.linspace(0., 8., 50)
    y = np.exp(-x) + 1. + _RNG.normal(scale=error, size=x.shape)
    plt.errorbar(x, y, error, label="Data", fmt="o")

    model = Exponential() + Constant()
    model[1].value.freeze(1.)
    model.fit(x, y, sigma=error)
    model.plot(fit_output=True, plot_components=False)
    plt.legend()


def test_shifted_exponential():
    """Test the shifted exponential model.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    error = 0.05
    x0 = 10.
    x = np.linspace(x0, 8. + x0, 50)
    y = np.exp(-(x - x0)) + _RNG.normal(scale=error, size=x.shape)
    plt.errorbar(x, y, error, label="Data", fmt="o")

    model = Exponential(x0)
    model.fit(x, y, sigma=error)
    model.plot(fit_output=True)
    plt.legend()


def test_shifted_exponential_frozen():
    """Test the shifted exponential model with a frozen parameter.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    error = 0.05
    x0 = 10.
    x = np.linspace(x0, 8. + x0, 50)
    y = np.exp(-(x - x0)) + _RNG.normal(scale=error, size=x.shape)
    plt.errorbar(x, y, error, label="Data", fmt="o")

    model = Exponential(x0)
    model.scale.freeze(1.)
    model.fit(x, y, sigma=error)
    model.plot(fit_output=True)
    plt.legend()


def test_confidence_band_constant():
    """Test the confidence band plotting.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Constant()
    model.set_parameters(10.)
    sigma = 0.1
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    model.fit(xdata, ydata, sigma=sigma)
    model.plot(fit_output=True)
    delta = model.confidence_band(xdata)
    assert np.allclose(delta, model.value.error)
    model.plot_confidence_band()
    plt.legend()


def test_confidence_band_linear():
    """Test the confidence band plotting.

    Note this is particularly simple, as the width of the 1-sigma band is equal to the
    uncertainty on the fitted parameter.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Line()
    model.set_parameters(2., 10.)
    sigma = 0.1
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    model.fit(xdata, ydata, sigma=sigma)
    model.plot(fit_output=True)
    model.plot_confidence_band(num_sigma=2.)
    plt.legend()


def test_gaussian_forest_fit_iterative_scatter():
    """Test an iterative fit for a GaussianForestBase child model class scatter plot.
    """
    # pylint: disable=no-member
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Fe55Forest(xlabel="x [a.u.]", ylabel="y [a.u.]")
    model.set_parameters(10, 0.15, 1., 0.3)
    print(model)
    sigma = 0.01
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    model.fit_iterative(xdata, ydata, sigma=sigma, num_sigma_left=1.5, num_sigma_right=1.5,
                        num_iterations=3)
    model.plot(fit_output=True)
    plt.legend()
    assert model.intensity1.compatible_with(0.15, NUM_SIGMA)
    assert model.sigma.compatible_with(0.3, NUM_SIGMA)
    assert model.status.pvalue > 0.001


def test_random_histogram():
    """Test the generation of a random histogram.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    edges = np.linspace(model.mu.value - 3 * model.sigma.value,
                         model.mu.value + 3 * model.sigma.value, 101)
    hist = model.random_histogram(edges=edges, size=100000, random_state=_RNG)
    hist.plot(label="Random data")
    plt.legend()
    assert np.array_equal(hist.bin_edges(), edges)
