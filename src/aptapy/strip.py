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
from typing import Sequence

import numpy as np

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

    def __init__(self, max_length: int = None, label: str = '', xlabel: str = None,
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

    def append(self, x: float, y: float) -> None:
        """Append a data point to the strip chart.
        """
        self.x.append(x)
        self.y.append(y)

    def extend(self, x: Sequence[float], y: Sequence[float]) -> None:
        """Append multiple data points to the strip chart.
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        self.x.extend(x)
        self.y.extend(y)

    def plot(self, axes=None, **kwargs) -> None:
        """Plot the strip chart.
        """
        kwargs.setdefault("label", self.label)
        if axes is None:
            axes = plt.gca()
        x = np.array(self.x).astype('datetime64[s]') if self._datetime else self.x
        axes.plot(x, self.y, **kwargs)
        setup_axes(axes, xlabel=self.xlabel, ylabel=self.ylabel, grids=True)
