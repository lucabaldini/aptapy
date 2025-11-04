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

from aptapy.models import (
    Arctangent,
    Constant,
    Erf,
    Exponential,
    ExponentialComplement,
    Gaussian,
    Line,
    Logistic,
    LogNormal,
    Lorentzian,
    Moyal,
    PowerLaw,
    Quadratic,
    StretchedExponential,
    StretchedExponentialComplement,
)
from aptapy.plotting import plt, setup_gca


def _test_model_base(model_class: type, parameter_values: Sequence[float],
                     integral: Callable[[float, float], float] = None,
                     sigma: float = 0.1, num_sigma: float = 5.):
    """Basic tests for the Model base class.
    """
    print("Testing model:", model_class.__name__)
    print("Ground truth:", parameter_values)
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
    xdata, ydata = model.random_fit_dataset(sigma, seed=313)
    plt.errorbar(xdata, ydata, sigma, fmt="o", label="Random data")
    model.init_parameters(xdata, ydata, sigma)
    model.plot(label="Initial guess", ls="--", color="gray")
    initial_values = model.parameter_values()
    print(f"Initial values: {initial_values}")
    model.fit(xdata, ydata, sigma=sigma)
    print("Fitted values:", model.parameter_values())
    for param, guess, ground_truth in zip(model, initial_values, parameter_values):
        # Note that we can programmatically relax the test on the initial guess by
        # increasing num_sigma, since for the majority of models the init_parameters()
        # is not meant to provide initial guess statistically compatible with the ground truth
        # truth. The final best-fit parameters, on the other hand, should be within a
        # reasonable number of sigma from the truth.
        assert param.compatible_with(guess, num_sigma)
        assert param.compatible_with(ground_truth, 5.)
    model.plot(fit_output=True)
    xmin, xmax = model.default_plotting_range()
    setup_gca(xmin=xmin, xmax=xmax)
    plt.legend()


def test_constant():
    """Test the Constant model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    value = 5.
    def integral(xmin, xmax):
        return value * (xmax - xmin)
    _test_model_base(Constant, (value, ), integral)


def test_line():
    """Test the Line model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    slope, intercept = 2., 5.
    def integral(xmin, xmax):
        return 0.5 * slope * (xmax**2 - xmin**2) + intercept * (xmax - xmin)
    _test_model_base(Line, (slope, intercept), integral)


def test_quadratic():
    """Test the Quadratic model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    a, b, c = 1., 2., 16.
    def integral(xmin, xmax):
        return (a * (xmax**3 - xmin**3) / 3. + b * (xmax**2 - xmin**2) / 2. + c * (xmax - xmin))
    _test_model_base(Quadratic, (a, b, c), integral)


def test_power_law():
    """Test the PowerLaw model---note we do this for two different indices.
    """
    for index in (-2., -1.):
        plt.figure(f"{inspect.currentframe().f_code.co_name}_index{abs(index)}")
        prefactor = 10.
        if index == -1.:
            def integral(xmin, xmax, prefactor=prefactor):
                return prefactor * np.log(xmax / xmin)
        else:
            def integral(xmin, xmax, prefactor=prefactor, index=index):
                return (prefactor / (index + 1.) * (xmax**(index + 1.) - xmin**(index + 1.)))
        _test_model_base(PowerLaw, (prefactor, index), integral)


def test_exponential():
    """Test the Exponential model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale = 10., 2.
    def integral(xmin, xmax):
        return prefactor * scale * (np.exp(-xmin / scale) - np.exp(-xmax / scale))
    _test_model_base(Exponential, (prefactor, scale), integral)


def test_exponential_complement():
    """Test the ExponentialComplement model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale = 10., 2.
    _test_model_base(ExponentialComplement, (prefactor, scale,), None)


def test_stretched_exponential():
    """Test the StretchedExponential model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale, gamma = 10., 2., 0.5
    # The initialization of the parameters is pretty flaky in this case...
    _test_model_base(StretchedExponential, (prefactor, scale, gamma), None, num_sigma=50.)


def test_stretched_exponential_complement():
    """Test the StretchedExponentialComplement model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, scale, gamma = 10., 2., 0.5
    # The initialization of the parameters is pretty flaky in this case...
    _test_model_base(StretchedExponentialComplement, (prefactor, scale, gamma), None, num_sigma=50.)


def test_gaussian_cdf():
    """Test the GaussianCDF model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, mean, sigma = 5., 0., 1.
    _test_model_base(GaussianCDF, (prefactor, mean, sigma), None)


def test_gaussian_cdf_complement():
    """Test the GaussianCDFComplement model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    prefactor, mean, sigma = 5., 0., 1.
    _test_model_base(GaussianCDFComplement, (prefactor, mean, sigma), None)


def test_gaussian():
    """Test the Gaussian model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Gaussian, (amplitude, location, scale), None, sigma=0.05, num_sigma=100.)


def test_lorentzian():
    """Test the Lorentzian model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Lorentzian, (amplitude, location, scale), None, sigma=0.05, num_sigma=100.)


def test_log_normal():
    """Test the LogNormal model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    model = LogNormal()
    model.set_parameters(amplitude, location, scale)
    print(model)

    import scipy.stats

    x = np.linspace(location, location + 5 * scale, 100)
    plt.plot(x, model(x))
    plt.plot(x, scipy.stats.lognorm.pdf(x, 1., location, scale) * amplitude)

    #_test_model_base(LogNormal, (amplitude, location, scale), None, sigma=0.1, num_sigma=100.)


def test_moyal():
    """Test the Moyal model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Moyal, (amplitude, location, scale), None, sigma=0.05, num_sigma=100.)


def test_erf():
    """Test the Erf model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Erf, (amplitude, location, scale), None, sigma=0.25)


def test_erf_complement():
    """Test the ErfComplement model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(ErfComplement, (amplitude, location, scale), None, sigma=0.25)


def test_logistic():
    """Test the Logistic model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Logistic, (amplitude, location, scale), None, sigma=0.25)


def test_arctangent():
    """Test the Arctangent model.
    """
    plt.figure(f"{inspect.currentframe().f_code.co_name}")
    amplitude, location, scale = 10., 10., 2.
    _test_model_base(Arctangent, (amplitude, location, scale), None, sigma=0.25)


if __name__ == "__main__":
    #test_gaussian()
    #test_lorentzian()
    test_log_normal()
    #test_moyal()
    #test_erf()
    #test_erf_complement()
    #test_logistic()
    #test_arctangent()
    plt.show()