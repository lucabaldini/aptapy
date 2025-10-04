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

"""Histogram facilities.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from .plotting import matplotlib, plt


class AbstractHistogram(ABC):

    """Abstract base class for an n-dimensional histogram.

    Arguments
    ---------
    edges : n-dimensional sequence of arrays
        the bin edges on the different axes.

    labels : n-dimensional tuple of strings
        the text labels for the different axes.
    """

    PLOT_OPTIONS = {}

    def __init__(self, edges: Sequence[np.ndarray], labels: Sequence[str] = None) -> None:
        """Constructor.
        """
        # Edges are fixed once and forever, so we create a copy. Also, no matter
        # which kind of sequence we are passing, we turn the thing into a tuple.
        self._edges = tuple(np.asarray(item, dtype=float).copy() for item in edges)
        self._num_axes = len(self._edges)

        # And a few basic checks on the input arguments.
        for item in self._edges:
            if item.ndim != 1:
                raise ValueError(f"Bin edges {item} are not a 1-dimensional array.")
            if item.size < 2:
                raise ValueError(f"Bin edges {item} have less than 2 entries.")
            #if any(np.any(np.diff(item) <= 0)):
            #    raise ValueError(f"Bin edges {item} not strictly increasing.")
        if labels is not None and len(labels) > self._num_axes + 1:
            raise ValueError(f"Too many labels {labels} for {self._num_axes} axes.")

        # Go ahead and create all the necessary data structures.
        self._shape = tuple(item.size - 1 for item in self._edges)
        self._sumw = self._zeros()
        self._sumw2 = self._zeros()
        # Prepare the axis labels: set them all to None...
        self._labels = [None] * self._num_axes
        # ... and overwrite with the input arguments, if any.
        if labels is not None:
            self._labels[:len(labels)] = labels

    def _zeros(self, dtype: type = float) -> np.ndarray:
        """Return an array of zeros of the proper shape for the underlying
        histograms quantities.
        """
        return np.zeros(shape=self._shape, dtype=dtype)

    @property
    def bin_contents(self) -> np.ndarray:
        """Return the bin contents.
        """
        return self._sumw

    @property
    def bin_errors(self) -> np.ndarray:
        """Return the bin errors.
        """
        return np.sqrt(self._sumw2)

    def bin_edges(self, axis: int = 0) -> np.array:
        """Return a view on the binning for specific axis.
        """
        return self._edges[axis].view()

    def bin_centers(self, axis: int = 0) -> np.array:
        """Return the bin centers for a specific axis.
        """
        return 0.5 * (self._edges[axis][1:] + self._edges[axis][:-1])

    def bin_widths(self, axis: int = 0) -> np.array:
        """Return the bin widths for a specific axis.
        """
        return np.diff(self._edges[axis])

    def set_axis_label(self, axis: int, label: str) -> None:
        """Set the label for a given axis.
        """
        self.labels[axis] = label

    def fill(self, *values: np.ndarray, weights: np.ndarray = None) -> "AbstractHistogram":
        """Fill the histogram from unbinned data.

        Note this method is returning the histogram instance, so that the function
        call can be chained.
        """
        values = np.vstack(values).T
        sumw, _ = np.histogramdd(values, bins=self._edges, weights=weights)
        if weights is None:
            sumw2 = sumw
        else:
            sumw2, _ = np.histogramdd(values, bins=self._edges, weights=weights**2.)
        self._sumw += sumw
        self._sumw2 += sumw2
        return self

    # @staticmethod
    # def bisect(binning: np.array, values: np.array, side: str = 'left') -> np.array:
    #     """Return the indices corresponding to a given array of values for a
    #     given binning.
    #     """
    #     return np.searchsorted(binning, values, side) - 1

    # def find_bin(self, *coords):
    #     """Find the bin corresponding to a given set of "physical" coordinates
    #     on the histogram axes.

    #     This returns a tuple of integer indices that can be used to address
    #     the histogram content.
    #     """
    #     return tuple(self.bisect(binning, value) for binning, value in zip(self.binning, coords))

    # def find_bin_value(self, *coords):
    #     """Find the histogram content corresponding to a given set of "physical"
    #     coordinates on the histogram axes.
    #     """
    #     return self.content[self.find_bin(*coords)]


    # def empty_copy(self):
    #     """Create an empty copy of a histogram.
    #     """
    #     return self.__class__(*self.binning, *self.labels)

    # def copy(self):
    #     """Create a full copy of a histogram.
    #     """
    #     hist = self.empty_copy()
    #     hist.set_content(self.content.copy(), self.entries.copy())
    #     return hist

    # def __add__(self, other):
    #     """Histogram addition.
    #     """
    #     hist = self.empty_copy()
    #     hist.set_content(self.content + other.content, self.entries + other.entries,
    #                      np.sqrt(self._sumw2 + other._sumw2))
    #     return hist

    # def __sub__(self, other):
    #     """Histogram subtraction.
    #     """
    #     hist = self.empty_copy()
    #     hist.set_content(self.content - other.content, self.entries + other.entries,
    #                      np.sqrt(self._sumw2 + other._sumw2))
    #     return hist

    # def __mul__(self, value):
    #     """Histogram multiplication by a scalar.
    #     """
    #     hist = self.empty_copy()
    #     hist.set_content(self.content * value, self.entries, self.errors() * value)
    #     return hist

    # def __rmul__(self, value):
    #     """Histogram multiplication by a scalar.
    #     """
    #     return self.__mul__(value)

    @abstractmethod
    def _do_plot(self, axes, **kwargs) -> None:
        pass

    def plot(self, axes=None, **kwargs) -> None:
        """Plot the histogram.
        """
        if axes is None:
            axes = plt.gca()
        for key, value in self.PLOT_OPTIONS.items():
            kwargs.setdefault(key, value)
        self._do_plot(axes, **kwargs)
        #setup_axes(axes, xlabel=self.labels[0], ylabel=self.labels[1])


class Histogram1d(AbstractHistogram):

    """A one-dimensional histogram.
    """

    PLOT_OPTIONS = dict(lw=1.25, alpha=0.4, histtype="stepfilled")

    def __init__(self, xedges: np.array, xlabel: str = "", ylabel: str = "Entries/bin") -> None:
        """Constructor.
        """
        super().__init__((xedges, ), [xlabel, ylabel])

    def _do_plot(self, axes, **kwargs) -> None:
        """Overloaded make_plot() method.
        """
        axes.hist(self.bin_centers(0), self._edges[0], weights=self.bin_contents, **kwargs)


# class Histogram2d(HistogramBase):

#     """A two-dimensional histogram.
#     """

#     PLOT_OPTIONS = dict(cmap=plt.get_cmap('hot'))
#     # pylint: disable=invalid-name

#     def __init__(self, xbinning, ybinning, xlabel='', ylabel='', zlabel='Entries/bin'):
#         """Constructor.
#         """
#         # pylint: disable=too-many-arguments
#         HistogramBase.__init__(self, (xbinning, ybinning), [xlabel, ylabel, zlabel])

#     def _plot(self, axes, logz=False, **kwargs):
#         """Overloaded make_plot() method.
#         """
#         # pylint: disable=arguments-differ
#         x, y = (v.flatten() for v in np.meshgrid(self.bin_centers(0), self.bin_centers(1)))
#         bins = self.binning
#         w = self.content.T.flatten()
#         if logz:
#             # Hack for a deprecated functionality in matplotlib 3.3.0
#             # Parameters norm and vmin/vmax should not be used simultaneously
#             # If logz is requested, we intercent the bounds when created the norm
#             # and refrain from passing vmin/vmax downstream.
#             vmin = kwargs.pop('vmin', None)
#             vmax = kwargs.pop('vmax', None)
#             kwargs.setdefault('norm', matplotlib.colors.LogNorm(vmin, vmax))
#         axes.hist2d(x, y, bins, weights=w, **kwargs)
#         color_bar = axes.colorbar()
#         if self.labels[2] is not None:
#             color_bar.set_label(self.labels[2])

#     def slice(self, bin_index: int, axis: int = 0):
#         """Return a slice of the two-dimensional histogram along the given axis.
#         """
#         hist = Histogram1d(self.binning[axis], self.labels[axis])
#         hist.set_content(self.content[:, bin_index], self.entries[:, bin_index])
#         return hist

#     def slices(self, axis: int = 0):
#         """Return all the slices along a given axis.
#         """
#         return tuple(self.slice(bin_index, axis) for bin_index in range(self._shape[axis]))

#     def hslice(self, bin_index: int):
#         """Return the horizontal slice for a given bin.
#         """
#         return self.slice(bin_index, 0)

#     def hslices(self):
#         """Return a list of all the horizontal slices.
#         """
#         return self.slices(0)

#     def hbisect(self, y: float):
#         """Return the horizontal slice corresponding to a given y value.
#         """
#         return self.hslice(self.bisect(self.binning[1], y))

#     def vslice(self, bin_index):
#         """Return the vertical slice for a given bin.
#         """
#         return self.slice(bin_index, 1)

#     def vslices(self):
#         """Return a list of all the vertical slices.
#         """
#         return self.slices(1)

#     def vbisect(self, x):
#         """Return the vertical slice corresponding to a given y value.
#         """
#         return self.vslice(self.bisect(self.binning[0], x))
