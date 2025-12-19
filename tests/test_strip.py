# Copyright 2023--2025 the aptapy team
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
import pytest

from aptapy.plotting import plt
from aptapy.strip import EpochStripChart, StripChart

_RNG = np.random.default_rng(313)


def test_put():
    """Test the basic interaction with StripChart objects.
    """
    chart = StripChart(max_length=100)
    chart.put(1, 2)
    assert len(chart) == 1
    chart.put(2., 3.)
    assert len(chart) == 2
    chart.put(3, 4.)
    chart.put(4., 3)
    chart.put(np.float32(5.), np.int32(6))
    chart.put(np.array(5.), np.array(6))
    assert len(chart) == 6
    chart.put([6, 7, 8], [9, 10, 11])
    assert len(chart) == 9
    chart.put((6, 7, 8), (9, 10, 11))
    assert len(chart) == 12
    chart.put(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert len(chart) == 15
    with pytest.raises(ValueError):
        chart.put(3., (1, 2))
    assert len(chart) == 15
    with pytest.raises(ValueError):
        chart.put((1, 2), 3)
    assert len(chart) == 15
    with pytest.raises(ValueError):
        chart.put((1, 2, 3), (1, 2))
    assert len(chart) == 15


def test_strip_chart_seconds():
    """Test a strip chart with seconds on the x axis.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    chart = StripChart(label='Strip chart', xlabel='Time [s]')
    t = np.linspace(0., 10., 100)
    y = np.sin(t)
    chart.put(t, y)
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
        chart.put(t, y)
        chart.plot()
