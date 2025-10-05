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


def test_init():
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


def test_filling1d():
    """
    """
    hist = Histogram1d(np.linspace(0., 1., 100))
    hist.fill(np.random.random(size=100000))


def test_plotting1d():
    """
    """
    plt.figure("test")
    edges = np.linspace(-5., 5., 100)
    h = Histogram1d(edges, 'x')
    h.fill(np.random.normal(size=100000))
    h.plot()


if __name__ == '__main__':
    test_plotting1d()
    plt.show()