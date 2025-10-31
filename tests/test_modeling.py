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

from aptapy.hist import Histogram1d
from aptapy.modeling import (
    Constant,
    Erf,
    ErfInverse,
    Exponential,
    ExponentialInverse,
    FitParameter,
    Gaussian,
    Line,
    PowerLaw,
    Quadratic,
)
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
    print(parameter)


def test_model_parameters():
    """We want to make sure that every model get its own set of parameters that can
    be varied independently.
    """
    g1 = Gaussian()
    g2 = Gaussian()
    p1 = g1.prefactor
    p2 = g2.prefactor
    print(p1, id(p1))
    print(p2, id(p2))
    assert p1 == p2
    assert id(p1) != id(p2)


def _test_model_base(model_class: type, parameter_values: Sequence[float],
                     integral: Callable = None, sigma: float = 0.1):
    """Basic tests for the Model base class.
    """
    model = model_class(xlabel="x [a.u.]", ylabel="y [a.u.]")
    model.set_parameters(*parameter_values)
    xmin, xmax = model.plotting_range()
    # Integral.
    if integral is not None:
        target = integral(xmin, xmax)
        assert model.quadrature(xmin, xmax) == pytest.approx(target)
        assert model.integral(xmin, xmax) == pytest.approx(target)
    # Parameter initialization and fitting. Note that if the parameter initialization
    # is not implemented for the model, this will be a no-op, and the fit starts
    # from the ground truth---no need to tweak the test function to handle this case.
    xdata, ydata = model.random_sample(sigma)
    model.init_parameters(xdata, ydata, sigma)
    initial_values = model.parameter_values()
    model.fit(xdata, ydata, sigma=sigma)
    for param, guess, ground_truth in zip(model, initial_values, parameter_values):
        assert param.compatible_with(guess)
        assert param.compatible_with(ground_truth)
    # Plotting.
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Random data')
    model.plot(fit_output=True)
    plt.legend()


def test_constant():
    """Test the Constant model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    value = 5.
    integral = lambda xmin, xmax: value * (xmax - xmin)
    _test_model_base(Constant, (value, ), integral)


def test_line():
    """Test the Line model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    slope, intercept = 2., 5.
    integral = lambda xmin, xmax: 0.5 * slope * (xmax**2 - xmin**2) + intercept * (xmax - xmin)
    _test_model_base(Line, (slope, intercept), integral)


def test_quadratic():
    """Test the Quadratic model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    a, b, c = 1., 2., 16.
    integral = lambda xmin, xmax: (a * (xmax**3 - xmin**3) / 3. +
                                   b * (xmax**2 - xmin**2) / 2. +
                                   c * (xmax - xmin))
    _test_model_base(Quadratic, (a, b, c), integral)


def test_power_law():
    """Test the PowerLaw model---note we do this for two different indices.
    """
    for index in (-2., -1.):
        plt.figure(f"{inspect.currentframe().f_code.co_name}_index{abs(index)}")
        prefactor = 10.
        if index == -1.:
            integral = lambda xmin, xmax: prefactor * np.log(xmax / xmin)
        else:
            integral = lambda xmin, xmax: (prefactor / (index + 1.) *
                                           (xmax**(index + 1.) - xmin**(index + 1.)))
        _test_model_base(PowerLaw, (prefactor, index), integral)


def test_exponential():
    """Test the Exponential model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale = 10., 2.
    integral = lambda xmin, xmax: prefactor * scale * (np.exp(-xmin / scale) - np.exp(-xmax / scale))
    _test_model_base(Exponential, (prefactor, scale), integral)


def test_exponential_inverse():
    """Test the ExponentialInverse model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale = 10, 2.
    _test_model_base(ExponentialInverse, (prefactor, scale,), None)


def test_gaussian():
    """Test the Gaussian model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, mean, sigma = 10., 0., 1.
    integral = lambda xmin, xmax: prefactor
    _test_model_base(Gaussian, (prefactor, mean, sigma), integral)


def test_gaussian_fit():
    """Simple Gaussian fit.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mean.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_subrange():
    """Gaussian fit in a subrange.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM, xmin=-2., xmax=2.)
    model.plot(fit_output=True)
    assert model.mean.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_bound():
    """Test a bounded fit.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    model.mean.minimum = 0.05
    model.mean.value = 0.1
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mean.value >= model.mean.minimum
    plt.legend()


def test_gaussian_fit_frozen():
    """Gaussian fit with frozen parameters.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    # Calculate the normalization from the histogram.
    model.prefactor.freeze(TEST_HISTOGRAM.area())
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mean.compatible_with(0., NUM_SIGMA)
    assert model.sigma.compatible_with(1., NUM_SIGMA)
    assert model.status.pvalue > 0.001
    plt.legend()


def test_gaussian_fit_frozen_and_bound():
    """And yet more complex: Gaussian fit with frozen and bound parameters.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    model.sigma.freeze(1.1)
    model.mean.minimum = 0.05
    model.mean.value = 0.1
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM)
    model.plot(fit_output=True)
    assert model.mean.value >= model.mean.minimum
    assert model.sigma.value == 1.1
    plt.legend()


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
    model.fit_histogram(hist)
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
    """Test the shifted exponential model.
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


if __name__ == "__main__":
    test_constant()
    test_line()
    test_quadratic()
    test_power_law()
    test_exponential()
    test_exponential_inverse()
    test_gaussian()
    plt.show()