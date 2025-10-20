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
from typing import List, Sequence, Tuple

import numpy as np

from .plotting import matplotlib, plt, setup_axes
from .typing_ import ArrayLike

__all__ = [
    "Histogram1d",
    "Histogram2d",
]

class AbstractHistogram(ABC):

    """Abstract base class for an n-dimensional histogram.

    Arguments
    ---------
    edges : n-dimensional sequence of arrays
        the bin edges on the different axes.

    axis_labels : sequence of strings
        the text labels for the different histogram axes.
    """

    DEFAULT_PLOT_OPTIONS = {}

    def __init__(self, edges: Sequence[np.ndarray], label: str, axis_labels: List[str]) -> None:
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
            if np.any(np.diff(item) <= 0):
                raise ValueError(f"Bin edges {item} not strictly increasing.")
        if axis_labels is not None and len(axis_labels) > self._num_axes + 1:
            raise ValueError(f"Too many axis labels {axis_labels} for {self._num_axes} axes.")

        # Go ahead and create all the necessary data structures.
        self._shape = tuple(item.size - 1 for item in self._edges)
        self._sumw = np.zeros(self._shape, dtype=float)
        self._sumw2 = np.zeros(self._shape, dtype=float)
        self.label = label
        self.axis_labels = axis_labels

    @property
    def content(self) -> np.ndarray:
        """Return the bin contents.
        """
        return self._sumw

    @property
    def errors(self) -> np.ndarray:
        """Return the bin errors.
        """
        return np.sqrt(self._sumw2)

    def bin_edges(self, axis: int = 0) -> np.ndarray:
        """Return a view on the binning for specific axis.
        """
        return self._edges[axis].view()

    def bin_centers(self, axis: int = 0) -> np.ndarray:
        """Return the bin centers for a specific axis.
        """
        return 0.5 * (self._edges[axis][1:] + self._edges[axis][:-1])

    def bin_widths(self, axis: int = 0) -> np.ndarray:
        """Return the bin widths for a specific axis.
        """
        return np.diff(self._edges[axis])

    def binned_statistics(self, axis: int = 0) -> Tuple[float, float]:
        """Return the mean and standard deviation along a specific axis, based
        on the binned data.

        Note this returns nan for for both mean and stddev if the histogram is
        empty (i.e., the sum of weights along the specified axis is zero).

        .. note::

           This is a crude estimate of the underlying statistics that might be
           useful for monitoring purposes, but should not be relied upon for
           quantitative analysis.

           This is not the same as computing the mean and standard deviation of
           the unbinned data that filled the histogram, as some information is
           lost in the binning process.

           In addition, note that we are not applying any bias correction to
           the standard deviation, as we are assuming that the histogram is
           filled with a sufficiently large number of entries. (In most circumstances
           the effect should be smaller than that of the binning itself.)

        Arguments
        ---------
        axis : int
            the axis along which to compute the statistics.

        Returns
        -------
        mean : float
            the mean value along the specified axis.
        stddev : float
            the standard deviation along the specified axis.
        """
        values = self.bin_centers(axis)
        weights = self.content.sum(axis=tuple(i for i in range(self.content.ndim) if i != axis))
        # Check the sum of weights---if zero, return NaN for both mean and stddev.
        # See https://github.com/lucabaldini/aptapy/issues/15
        if weights.sum() == 0.:
            return float('nan'), float('nan')
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean)**2, weights=weights)
        return float(mean), float(np.sqrt(variance))

    def fill(self, *values: ArrayLike, weights: ArrayLike = None) -> "AbstractHistogram":
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

    def copy(self) -> "AbstractHistogram":
        """Create a full copy of a histogram.
        """
        # pylint: disable=protected-access
        # Note we really need the * in the constructor, here, as the abstract
        # base class is never instantiated, and the arguments are unpacked in the
        # constructors of all the derived classes.
        histogram = self.__class__(*self._edges, self.label, *self.axis_labels)
        histogram._sumw = self._sumw.copy()
        histogram._sumw2 = self._sumw2.copy()
        return histogram

    def _check_compat(self, other: "AbstractHistogram") -> None:
        """Check whether two histogram objects are compatible with each other,
        meaning, e.g., that they can be summed or subtracted.
        """
        # pylint: disable=protected-access
        if not isinstance(other, AbstractHistogram):
            raise TypeError(f"{other} is not a histogram.")
        if self._num_axes != other._num_axes or self._shape != other._shape:
            raise ValueError("Histogram dimensionality/shape mismatch.")
        for edges in zip(self._edges, other._edges):
            if not np.allclose(*edges):
                raise ValueError("Histogram bin edges differ.")

    def __iadd__(self, other: "AbstractHistogram") -> "AbstractHistogram":
        """Histogram addition (in place).
        """
        self._check_compat(other)
        self._sumw += other._sumw
        self._sumw2 += other._sumw2
        return self

    def __add__(self, other: "AbstractHistogram") -> "AbstractHistogram":
        """Histogram addition.
        """
        histogram = self.copy()
        histogram += other
        return histogram

    def __isub__(self, other: "AbstractHistogram") -> "AbstractHistogram":
        """Histogram subtraction (in place).
        """
        self._check_compat(other)
        self._sumw -= other._sumw
        self._sumw2 += other._sumw2
        return self

    def __sub__(self, other: "AbstractHistogram") -> "AbstractHistogram":
        """Histogram subtraction.
        """
        histogram = self.copy()
        histogram -= other
        return histogram

    @abstractmethod
    def _do_plot(self, axes, **kwargs) -> None:
        pass

    def plot(self, axes=None, **kwargs) -> None:
        """Plot the histogram.
        """
        if axes is None:
            axes = plt.gca()
        for key, value in self.DEFAULT_PLOT_OPTIONS.items():
            kwargs.setdefault(key, value)
        self._do_plot(axes, **kwargs)

    def __repr__(self) -> str:
        """String representation of the histogram.
        """
        return f"{self.__class__.__name__}({self._num_axes} axes, shape={self._shape})"


class Histogram1d(AbstractHistogram):

    """One-dimensional histogram.

    Arguments
    ---------
    edges : 1-dimensional array
        the bin edges.

    label : str
        overall label for the histogram (if defined, this will be used in the
        legend by default).

    xlabel : str
        the text label for the x axis.

    ylabel : str
        the text label for the y axis (default: "Entries/bin").
    """

    DEFAULT_PLOT_OPTIONS = dict(linewidth=1.25, alpha=0.4, histtype="stepfilled")

    def __init__(self, xedges: np.ndarray, label: str = None, xlabel: str = None,
                 ylabel: str = "Entries/bin") -> None:
        """Constructor.
        """
        super().__init__((xedges, ), label, [xlabel, ylabel])

    def area(self) -> float:
        """Return the total area under the histogram.

        This is potentially useful when fitting a model to the histogram, e.g.,
        to freeze the prefactor of a gaussian to the histogram normalization.

        Returns
        -------
        area : float
            The total area under the histogram.
        """
        return (self.content * self.bin_widths()).sum()

    def _do_plot(self, axes: matplotlib.axes._axes.Axes, **kwargs) -> None:
        """Overloaded make_plot() method.
        """
        # If we are not explicitly providing a label at plotting time, use
        # the one attached to the histogram, if any.
        kwargs.setdefault("label", f"{self}")
        axes.hist(self.bin_centers(0), self._edges[0], weights=self.content, **kwargs)
        setup_axes(axes, xlabel=self.axis_labels[0], ylabel=self.axis_labels[1])

    def __str__(self) -> str:
        """String formatting.
        """
        mean, rms = self.binned_statistics()
        text = self.label or self.__class__.__name__
        text = f"{text}\nMean: {mean:g}\nRMS: {rms:g}"
        return text


class Histogram2d(AbstractHistogram):

    """Two-dimensional histogram.

    Arguments
    ---------
    xedges : 1-dimensional array
        the bin edges on the x axis.

    yedges : 1-dimensional array
        the bin edges on the y axis.

    label : str
        overall label for the histogram

    xlabel : str
        the text label for the x axis.

    ylabel : str
        the text label for the y axis.

    zlabel : str
        the text label for the z axis (default: "Entries/bin").
    """

    DEFAULT_PLOT_OPTIONS = dict(cmap=plt.get_cmap("hot"))

    def __init__(self, xedges, yedges, label: str = None, xlabel: str = None,
                 ylabel: str = None, zlabel: str = "Entries/bin") -> None:
        """Constructor.
        """
        super().__init__((xedges, yedges), label, [xlabel, ylabel, zlabel])

    def _do_plot(self, axes: matplotlib.axes._axes.Axes, logz: bool = False, **kwargs) -> None:
        """Overloaded make_plot() method.
        """
        # pylint: disable=arguments-differ
        if logz:
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            kwargs.setdefault("norm", matplotlib.colors.LogNorm(vmin, vmax))
        mappable = axes.pcolormesh(*self._edges, self.content.T, **kwargs)
        plt.colorbar(mappable, ax=axes, label=self.axis_labels[2])
        setup_axes(axes, xlabel=self.axis_labels[0], ylabel=self.axis_labels[1])
