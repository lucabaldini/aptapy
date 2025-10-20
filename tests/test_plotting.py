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

"""Unit tests for the plotting module.
"""

import inspect

import numpy as np

from aptapy.plotting import ConstrainedTextMarker, VerticalCursor, plt, setup_gca
from aptapy.strip import StripChart


def test_marker():
    """Test the ConstrainedTextMarker.
    """
    # pylint: disable=protected-access
    plt.figure(inspect.currentframe().f_code.co_name)
    marker = ConstrainedTextMarker(np.sin)
    x, y = marker._marker.get_data()
    # Make sure the marker position is None, and the marker is not visible.
    assert x[0] is None
    assert y[0] is None
    assert not marker._marker.get_visible()
    assert not marker._text.get_visible()
    # Move the marker, and make sure it is in the right place.
    pos = 2.
    marker.move(pos)
    x, y = marker._marker.get_data()
    assert x[0] == pos
    assert y[0] == np.sin(pos)


def test_cursor():
    """Test the VerticalCursor class.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    x = np.linspace(0., 2. * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    cursor = VerticalCursor()
    plt.plot(x, y1)
    cursor.add_marker(np.sin)
    plt.plot(x, y2)
    cursor.add_marker(np.cos)
    setup_gca(xmin=0., xmax=2. * np.pi, ymin=-1.25, ymax=1.25)
    cursor.activate()
    return cursor


def test_strip_cursor():
    """Test a vertical cursor with a strip chart.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    x = np.linspace(0., 2. * np.pi, 100)
    chart1 = StripChart().put(x, np.sin(x))
    chart2 = StripChart().put(x, np.cos(x))
    cursor = VerticalCursor()
    chart1.plot()
    cursor.add_marker(chart1.spline())
    chart2.plot()
    cursor.add_marker(chart2.spline())
    setup_gca(xmin=0., xmax=2. * np.pi, ymin=-1.25, ymax=1.25)
    cursor.activate()
    return cursor


if __name__ == '__main__':
    # Note we have to keep a reference to the cursor not to lose it.
    cursor1 = test_cursor()
    cursor2 = test_strip_cursor()
    plt.show()
