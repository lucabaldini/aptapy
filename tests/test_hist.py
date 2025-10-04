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

from aptapy.hist import Histogram1d
from aptapy.plotting import plt


def test():
    """
    """
    plt.figure("test")
    edges = np.linspace(-5., 5., 100)
    h = Histogram1d(edges, 'x')
    h.fill(np.random.normal(size=100000))
    h.plot()


if __name__ == '__main__':
    test()
    plt.show()