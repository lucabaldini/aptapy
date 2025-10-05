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

import numpy as np
import pytest

from aptapy.hist import Histogram1d
from aptapy.plotting import plt


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


def test_plotting1d():
    """Test plotting.
    """
    plt.figure("test")
    edges = np.linspace(-5., 5., 100)
    h = Histogram1d(edges, 'x')
    h.fill(np.random.normal(size=100000))
    h.plot()


if __name__ == '__main__':
    test_plotting1d()
    plt.show()
