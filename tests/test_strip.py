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

import numpy as np

from aptapy.plotting import plt
from aptapy.strip import StripChart


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


def test_strip_chart_datetime():
    """Test a strip chart with datetime on the x axis.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    chart = StripChart(datetime=True, xlabel='UTC time')
    t = np.linspace(1_600_000_000., 1_600_000_900., 100)
    y = np.sin(np.linspace(0., 10., 100))
    chart.extend(t, y)
    chart.plot()


if __name__ == '__main__':
    test_strip_chart_seconds()
    test_strip_chart_datetime()
    plt.show()
