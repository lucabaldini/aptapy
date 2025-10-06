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

"""Unit tests for the hist module.
"""

import inspect

import numpy as np
import pytest

from aptapy.hist import Histogram1d, Histogram2d
from aptapy.plotting import plt

_RNG = np.random.default_rng()


def test_init1d():
    """Test all the initialization cross checks.
    """
    edges = np.array([[1., 2.], [3., 4]])
    with pytest.raises(ValueError, match="not a 1-dimensional array"):
        _ = Histogram1d(edges)
    edges = np.array([1.])
    with pytest.raises(ValueError, match="less than 2 entries"):
        _ = Histogram1d(edges)
    edges = np.array([2., 1.])
    with pytest.raises(ValueError, match="not strictly increasing"):
        _ = Histogram1d(edges)


def test_binning1d():
    """Test the binning-related methods.
    """
    edges = np.linspace(0., 1., 11)
    hist = Histogram1d(edges)
    assert np.allclose(hist.content, 0.)
    assert np.allclose(hist.errors, 0.)
    assert np.allclose(hist.bin_centers(), np.linspace(0.05, 0.95, 10))
    assert np.allclose(hist.bin_widths(), 0.1)


def test_filling1d():
    """Simple filling test with a 1-bin, 1-dimensional histogram.
    """
    hist = Histogram1d(np.linspace(0., 1., 2))
    # Fill with a numpy array.
    hist.fill(np.full(100, 0.5))
    assert hist.content == 100.
    # Fill with a number.
    hist.fill(0.5)
    assert hist.content == 101.


def test_compat1d():
    """Test the histogram compatibility.
    """
    # pylint: disable=protected-access
    hist = Histogram1d(np.array([0., 1., 2]))
    hist._check_compat(hist.copy())
    with pytest.raises(TypeError, match="not a histogram"):
        hist._check_compat(None)
    with pytest.raises(ValueError, match="dimensionality/shape mismatch"):
        hist._check_compat(Histogram1d(np.array([0., 1., 2., 3.])))
    with pytest.raises(ValueError, match="bin edges differ"):
        hist._check_compat(Histogram1d(np.array([0., 1.1, 2.])))


def test_arithmetics1d():
    """Test the basic arithmetics.
    """
    # pylint: disable=protected-access
    sample1 = _RNG.uniform(size=10000)
    sample2 = _RNG.uniform(size=10000)
    edges = np.linspace(0., 1., 100)
    hist1 = Histogram1d(edges).fill(sample1)
    hist2 = Histogram1d(edges).fill(sample2)
    hist3 = Histogram1d(edges).fill(sample1).fill(sample2)
    hist_sum = hist1 + hist2
    hist_sub = hist1 - hist1
    assert np.allclose(hist_sum._sumw, hist3._sumw)
    assert np.allclose(hist_sum._sumw2, hist3._sumw2)
    assert np.allclose(hist_sub._sumw, 0.)


def test_plotting1d(size: int = 100000):
    """Test plotting.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    # Create the first histogram. This has no label attached, so we will have to
    # provide one at plotting time, if we want to have a corresponding legend entry.
    hist1 = Histogram1d(np.linspace(-5., 5., 100), xlabel='x')
    hist1.fill(_RNG.normal(size=size))
    hist1.plot(label='Standard histogram')
    # Create a second histogram, this time with a label---this should have a
    # proper entry in the legend automatically.
    hist2 = Histogram1d(np.linspace(-5., 5., 100), label='Offset histogram')
    hist2.fill(_RNG.normal(size=size, loc=1.))
    hist2.plot()
    # And this one should end up with no legend entry, as it has no label
    hist3 = Histogram1d(np.linspace(-5., 5., 100))
    hist3.fill(_RNG.normal(size=size // 2, loc=-1.))
    hist3.plot()
    plt.legend()


def test_plotting2d(size: int = 100000):
    """Test plotting.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    edges = np.linspace(-5., 5., 100)
    hist = Histogram2d(edges, edges, 'x', 'y')
    # Note we are adding different offsets to x and y so that we can see
    # the effect on the plot.
    hist.fill(_RNG.normal(size=size) + 1., _RNG.normal(size=size) - 1.)
    hist.plot()
    plt.gca().set_aspect('equal')


if __name__ == '__main__':
    test_plotting1d()
    test_plotting2d()
    plt.show()
