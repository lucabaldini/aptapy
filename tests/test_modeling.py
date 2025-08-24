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

import matplotlib.pyplot as plt
import numpy as np

from aptapy.modeling import FitParameter, Gaussian


def test_fit_parameter():
    """Test the FitParameter class.
    """
    parameter = FitParameter('normalization', 1.)
    assert not parameter.is_bound()
    print(parameter)
    parameter = FitParameter('normalization', 1., 0.1)
    assert not parameter.is_bound()
    print(parameter)
    parameter = FitParameter('normalization', 1., frozen=True)
    assert not parameter.is_bound()
    print(parameter)
    parameter = FitParameter('normalization', 1., minimum=0.)
    assert parameter.is_bound()
    print(parameter)


def test_model_parameters():
    """We want to make sure that every model get its own set of parameters that can
    be varied independently.
    """
    g1 = Gaussian()
    g2 = Gaussian()
    assert g1.prefactor == g2.prefactor
    assert id(g1.prefactor) != id(g2.prefactor)


def _test_data_set(model, xmin, xmax, num_points=25, relative_error=0.05):
    """
    """
    xdata = np.linspace(xmin, xmax, num_points)
    ydata = model(xdata)
    sigma = ydata * relative_error
    ydata += np.random.normal(0., sigma)
    return xdata, ydata, sigma


def test_gaussian_fit(relative_error=0.05):
    """Test the Gaussian model.
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    plt.figure('Gaussian fit')
    plt.errorbar(xdata, ydata, sigma, fmt='o')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()


def test_gaussian_fit_subrange():
    """
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    plt.figure('Gaussian fit in subrange')
    plt.errorbar(xdata, ydata, sigma, fmt='o')
    model.fit(xdata, ydata, sigma=sigma, xmin=-2., xmax=2.)
    print(model)
    model.plot()


def test_gaussian_fit_bounded():
    """
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    model.mean.minimum = 0.1
    model.mean.value = 0.2
    plt.figure('Gaussian fit bounded')
    plt.errorbar(xdata, ydata, sigma, fmt='o')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()


def test_gaussian_fit_frozen():
    """
    """
    model = Gaussian()
    xdata, ydata, sigma = _test_data_set(model, -4., 4.)
    model.prefactor.frozen = True
    plt.figure('Gaussian fit frozen')
    plt.errorbar(xdata, ydata, sigma, fmt='o')
    model.fit(xdata, ydata, sigma=sigma)
    print(model)
    model.plot()



if __name__ == '__main__':
    test_gaussian_fit()
    test_gaussian_fit_subrange()
    test_gaussian_fit_bounded()
    test_gaussian_fit_frozen()
    plt.show()