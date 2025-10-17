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

"""Strip charts.
"""

import collections
from numbers import Number
from typing import Sequence

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .plotting import plt, setup_axes


class StripChart:

    """Class describing a sliding strip chart, that is, a scatter plot where the
    number of points is limited to a maximum, so that the thing acts essentially
    as a sliding window, typically in time.

    Arguments
    ---------
    max_length : int, optional
        the maximum number of points to keep in the strip chart. If None (the default),
        the number of points is unlimited.

    label : str, optional
        a text label for the data series (default is None).

    xlabel : str, optional
        the label for the x axis.

    ylabel : str, optional
        the label for the y axis.

    datetime : bool, optional
        if True, the x values are treated as POSIX timestamps and converted to
        datetime objects for plotting purposes (default is False).
    """

    def __init__(self, max_length: int = None, label: str = "", xlabel: str = None,
                 ylabel: str = None, datetime: bool = False) -> None:
        """Constructor.
        """
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._datetime = datetime
        self.x = collections.deque(maxlen=max_length)
        self.y = collections.deque(maxlen=max_length)

    def clear(self) -> None:
        """Reset the strip chart.
        """
        self.x.clear()
        self.y.clear()

    def append(self, x: float, y: float) -> "StripChart":
        """Append a single data point to the strip chart.

        Note this returns the strip chart itself in order to allow for
        chaining operations.

        Arguments
        ---------
        x : float
            The x value to append to the strip chart.

        y : float
            The y value to append to the strip chart.

        Returns
        -------
        StripChart
            The strip chart itself
        """
        if not isinstance(x, Number):
            raise TypeError("x must be a number")
        if not isinstance(y, Number):
            raise TypeError("y must be a number")
        self.x.append(x)
        self.y.append(y)
        return self

    def extend(self, x: Sequence[float], y: Sequence[float]) -> "StripChart":
        """Append multiple data points to the strip chart.

        Note this returns the strip chart itself in order to allow for
        chaining operations.

        Arguments
        ---------
        x : sequence[float]
            The x values to append to the strip chart.

        y : sequence[float]
            The y values to append to the strip chart.

        Returns
        -------
        StripChart
            The strip chart itself
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        self.x.extend(x)
        self.y.extend(y)
        return self

    def spline(self, k: int = 1) -> InterpolatedUnivariateSpline:
        """Return an interpolating spline through all the underlying
        data points.

        This is useful, e.g., when adding a vertical cursor to the strip chart.

        Arguments
        ---------
        k : int
            The order of the spline (default 1).

        Returns
        -------
        InterpolatedUnivariateSpline
            The interpolating spline.
        """
        return InterpolatedUnivariateSpline(self.x, self.y, k=k)

    def plot(self, axes=None, **kwargs) -> None:
        """Plot the strip chart.
        """
        kwargs.setdefault("label", self.label)
        if axes is None:
            axes = plt.gca()
        x = np.array(self.x).astype("datetime64[s]") if self._datetime else self.x
        axes.plot(x, self.y, **kwargs)
        setup_axes(axes, xlabel=self.xlabel, ylabel=self.ylabel, grids=True)
