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

import numpy as np
from matplotlib import dates as mdates
from scipy.interpolate import InterpolatedUnivariateSpline

from .plotting import plt, setup_axes
from .typing_ import ArrayLike


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
    """

    def __init__(self, max_length: int = None, label: str = "", xlabel: str = None,
                 ylabel: str = None) -> None:
        """Constructor.
        """
        self.label = label
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x = collections.deque(maxlen=max_length)
        self.y = collections.deque(maxlen=max_length)

    def __len__(self) -> int:
        """Return the number of points in the strip chart.
        """
        return len(self.x)

    def set_max_length(self, max_length: int) -> None:
        """Set the maximum length of the strip chart.

        Arguments
        ---------
        max_length : int
            the new maximum number of points to keep in the strip chart.

        Note this creates two new deque objects under the hood but labels are
        preserved. There is no attempt to preserve existing data points.
        """
        self.x = collections.deque(maxlen=max_length)
        self.y = collections.deque(maxlen=max_length)

    def clear(self) -> None:
        """Reset the strip chart.
        """
        self.x.clear()
        self.y.clear()

    @staticmethod
    def _is_numerical_scalar(value) -> bool:
        """Return whether the given value is a numerical scalar.

        This is more tricky than one would expect as, while np.int32(1) and alike
        are instances of Number, 0-dim numpy arrays, e.g., numpy.array(1), are not.
        """
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and \
            np.ndim(value) == 0:
            return True
        return isinstance(value, Number)

    def put(self, x: ArrayLike, y: ArrayLike) -> "StripChart":
        """Append data points to the strip chart.

        This is supposed to correctly interoperate with all the data types we
        will be typically deal with: numbers, numpy scalars, and iterables,
        including numpy arrays.

        Note this returns the strip chart itself in order to allow for
        chaining operations.

        Arguments
        ---------
        x : array-like
            The x value(s) to append to the strip chart.

        y : array-like
            The y value(s) to append to the strip chart.

        Returns
        -------
        StripChart
            The strip chart itself
        """
        # Are x and y both scalars in the largest possible sense?
        if self._is_numerical_scalar(x) and self._is_numerical_scalar(y):
            self.x.append(x)
            self.y.append(y)
            return self
        # At this point, since we are not dealing with two scalars, we
        # are expecting two sequences of the same length, so we shall just
        # assume that len() works on both x and y and raise an exception otherwise.
        try:
            if len(x) != len(y):
                raise ValueError("x and y must have the same length")
        except TypeError as error:
            raise ValueError("x and y must be both scalars or both sequences") from error
        # And if we made it all the way here, we are all set, and can extend the deques.
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
        axes.plot(self.x, self.y, **kwargs)
        setup_axes(axes, xlabel=self.xlabel, ylabel=self.ylabel, grids=True)


class EpochStripChart(StripChart):

    """Class describing a sliding strip chart with epoch time on the x axis.

    Operationally, this assumes that the values on the x axis are seconds since the
    Unix epoch (January 1st, 1970), e.g., from a time.time() call. These are then
    converted into NumPy datetime64 values (with the desired resolution) at plot time.

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

    resolution : str, optional
        the resolution for the x axis. Supported values are "s" (seconds),
        "ms" (milliseconds), "us" (microseconds), and "ns" (nanoseconds). Default is "ms".
    """

    _RESOLUTION_MULTIPLIER_DICT = {
        "s": 1,
        "ms": 1_000,
        "us": 1_000_000,
        "ns": 1_000_000_000
        }

    def __init__(self, max_length: int = None, label: str = "", xlabel: str = "Date and Time (UTC)",
                 ylabel: str = None, resolution: str = "ms") -> None:
        """Constructor.
        """
        if resolution not in self._RESOLUTION_MULTIPLIER_DICT:
            raise ValueError(f"Unsupported resolution '{resolution}'")
        super().__init__(max_length, label, xlabel, ylabel)
        # AutoDateLocator automatically chooses tick spacing (seconds,
        # minutes, hours, days, etc.) depending on your data range.
        self.locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        # ConciseDateFormatter (introduced in Matplotlib 3.1) produces
        # compact, readable labels
        self.formatter = mdates.ConciseDateFormatter(self.locator)
        # Cache the numpy datetime64 type...
        self._type = f"datetime64[{resolution}]"
        # ...and the associated multiplier to convert from seconds since epoch.
        self._multiplier = self._RESOLUTION_MULTIPLIER_DICT[resolution]

    def plot(self, axes=None, **kwargs) -> None:
        """Plot the strip chart.

        This is more tricky than one would expect, as NumPy's datetime64 type stores
        timestamps as integer counts of a specific unit (like seconds, milliseconds,
        or nanoseconds) from the epoch. Assuming that we are using seconds since the
        epoch as input, we need to convert those into the appropriate integer counts.
        This boils down to using a multiplier depending on the desired resolution.
        """
        kwargs.setdefault("label", self.label)
        if axes is None:
            axes = plt.gca()
        # Convert seconds since epoch into appropriate datetime64 type.
        # Now, this might be an overkill, but the series of numpy conversions is meant
        # to turn the float seconds into the floating-point representation of the
        # nearest integer, which is then cast into an actual integer, and finally into
        # the desired datetime64 type.
        x = np.rint(self._multiplier * np.asarray(self.x)).astype('int64').astype(self._type)
        axes.plot(x, self.y, **kwargs)
        # Set up datetime x axis.
        axes.xaxis.set_major_locator(self.locator)
        axes.xaxis.set_major_formatter(self.formatter)
        setup_axes(axes, xlabel=self.xlabel, ylabel=self.ylabel, grids=True)
