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

import numpy as np
import pytest

from aptapy.hist import Histogram1d
from aptapy.modeling import (
    Constant,
    Erf,
    ErfInverse,
    Exponential,
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


def test_plot():
    """Test the plot method of the models.
    """
    for model in (Constant(), Line(), Quadratic(), PowerLaw(), Exponential(),
                  Gaussian(), Erf(), ErfInverse()):
        plt.figure(f"{inspect.currentframe().f_code.co_name}_{model.__class__.__name__}")
        model.plot()
        plt.legend()


def test_integral():
    """Test the integral method of the models.
    """
    # pylint: disable=too-many-statements
    # Constant.
    xmin = 0.
    xmax = 1.
    value = 1.
    target = value * (xmax - xmin)
    model = Constant()
    model.value.freeze(1.)
    assert model.quadrature(xmin, xmax) == pytest.approx(target)
    assert model.integral(xmin, xmax) == pytest.approx(target)
    # Line.
    slope = 1.
    intercept = 1.
    target = 0.5 * slope * (xmax**2 - xmin**2) + intercept * (xmax - xmin)
    model = Line()
    model.slope.freeze(slope)
    model.intercept.freeze(intercept)
    assert model.quadrature(xmin, xmax) == pytest.approx(target)
    assert model.integral(xmin, xmax) == pytest.approx(target)
    # Quadratic.
    a = 1.
    b = 1.
    c = 1.
    target = a * (xmax**3 - xmin**3) / 3. + b * (xmax**2 - xmin**2) / 2. + c * (xmax - xmin)
    model = Quadratic()
    model.a.freeze(a)
    model.b.freeze(b)
    model.c.freeze(c)
    assert model.quadrature(xmin, xmax) == pytest.approx(target)
    assert model.integral(xmin, xmax) == pytest.approx(target)
    # PowerLaw.
    xmin = 1.
    xmax = 10.
    prefactor = 1.
    for index in (-2., -1.):
        if index == -1.:
            target = prefactor * np.log(xmax / xmin)
        else:
            target = prefactor / (index + 1.) * (xmax**(index + 1.) - xmin**(index + 1.))
        model = PowerLaw()
        model.prefactor.freeze(prefactor)
        model.index.freeze(index)
        assert model.quadrature(xmin, xmax) == pytest.approx(target)
        assert model.integral(xmin, xmax) == pytest.approx(target)
    # Exponential.
    xmin = 0.
    xmax = 10.
    prefactor = 1.
    scale = 1.
    target = prefactor * scale * (np.exp(-xmin / scale) - np.exp(-xmax / scale))
    model = Exponential()
    model.prefactor.freeze(prefactor)
    model.scale.freeze(scale)
    assert model.quadrature(xmin, xmax) == pytest.approx(target)
    assert model.integral(xmin, xmax) == pytest.approx(target)
    # Gaussian.
    xmin = -5.
    xmax = 5.
    prefactor = 1.
    mean = 0.
    sigma = 1.
    target = 1.
    model = Gaussian()
    model.prefactor.freeze(prefactor)
    model.mean.freeze(mean)
    model.sigma.freeze(sigma)
    assert model.quadrature(xmin, xmax) == pytest.approx(target)
    assert model.integral(xmin, xmax) == pytest.approx(target)


def test_gaussian_fit():
    """Simple Gaussian fit.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian()
    TEST_HISTOGRAM.plot()
    model.fit_histogram(TEST_HISTOGRAM)
    model.plot()
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
    model.plot()
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
    model.plot()
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
    model.plot()
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
    model.plot()
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
    model.plot()
    plt.legend()


def test_multiple_sum():
    """Test the sum of multiple models.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    model = Gaussian() + Line() + Constant()
    model.plot(-5., 5.)
    plt.legend()


if __name__ == '__main__':
    test_plot()
    test_gaussian_fit()
    test_gaussian_fit_subrange()
    test_gaussian_fit_bound()
    test_gaussian_fit_frozen()
    test_gaussian_fit_frozen_and_bound()
    test_sum_gauss_line()
    test_multiple_sum()
    plt.show()
