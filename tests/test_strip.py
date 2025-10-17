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

"""Unit tests for the strip module.
"""

import inspect
import time

import numpy as np

from aptapy.plotting import plt
from aptapy.strip import EpochStripChart, StripChart

_RNG = np.random.default_rng(313)


def test_strip_chart_seconds():
    """Test a strip chart with seconds on the x axis.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    chart = StripChart(label='Strip chart', xlabel='Time [s]')
    t = np.linspace(0., 10., 100)
    y = np.sin(t)
    chart.extend(t, y)
    chart.plot()
    plt.legend()


def test_strip_chart_datetime(num_points: int = 100):
    """Test a strip chart with datetime on the x axis.
    """
    t0 = time.time()
    y = _RNG.random(num_points)
    for duration in (10, 100, 1000, 10000, 100000):
        plt.figure(f"{inspect.currentframe().f_code.co_name}_{duration}")
        chart = EpochStripChart(label="Random data")
        t = t0 + np.linspace(0., duration, num_points)
        chart.extend(t, y)
        chart.plot()


if __name__ == '__main__':
    test_strip_chart_seconds()
    test_strip_chart_datetime()
    plt.show()
