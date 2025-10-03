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

import numpy as np

from aptapy.modeling import FitParameter, Gaussian
from aptapy.plotting import plt


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


def _test_data_set(model, xmin, xmax, num_points=25, relative_error=0.05, min_error=0.01):
    """
    """
    rng = np.random.default_rng(seed=313)
    xdata = np.linspace(xmin, xmax, num_points)
    ydata = model(xdata)
    sigma = ydata * relative_error + min_error
    ydata += rng.normal(0., sigma)
    return xdata, ydata, sigma


def test_gaussian_fit():
    """Test the Gaussian model.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    plt.figure('Gaussian fit')
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Data')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()
    plt.legend()


def test_gaussian_fit_subrange():
    """Test fit in a subrange.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    plt.figure('Gaussian fit in subrange')
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Data')
    model.fit(xdata, ydata, sigma=sigma, xmin=-2., xmax=2.)
    print(model)
    model.plot()
    plt.legend()


def test_gaussian_fit_bound():
    """Test a bounded fit.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    model.mean.minimum = 0.1
    model.mean.value = 0.2
    plt.figure('Gaussian fit bound')
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Data')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()
    plt.legend()


def test_gaussian_fit_frozen():
    """Fit with a frozen parameter.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    model.prefactor.freeze()
    plt.figure('Gaussian fit frozen')
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Data')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()
    plt.legend()


def test_gaussian_fit_frozen_and_bound():
    """And yet more complex: frozen and bound.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    model.sigma.freeze(1.1)
    model.sigma.minimum = 0.
    plt.figure('Gaussian fit frozen and bound')
    plt.errorbar(xdata, ydata, sigma, fmt='o', label='Data')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()
    plt.legend()


if __name__ == '__main__':
    test_gaussian_fit()
    test_gaussian_fit_subrange()
    test_gaussian_fit_bound()
    test_gaussian_fit_frozen()
    test_gaussian_fit_frozen_and_bound()
    plt.show()
