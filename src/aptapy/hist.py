# Copyright 2023--2026 the aptapy team
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

from abc import abstractmethod
from typing import Callable, List, Sequence, Tuple, Union

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from .plotting import AbstractPlottable, plt
from .typing_ import ArrayLike, PathLike

__all__ = [
    "Histogram1d",
    "Histogram2d",
    "Histogram3d",
]

class AbstractHistogram(AbstractPlottable):

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
        # Note we assume at least two labels (x and y) for plotting purposes---they
        # both can be None, but subclasses must provide them.
        if len(axis_labels) < 2:
            raise ValueError("At least two axis labels must be provided.")
        self.axis_labels = axis_labels
        super().__init__(label, *self.axis_labels[:2])
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

    def set_content(self, content: ArrayLike, errors: ArrayLike = None) -> "AbstractHistogram":
        """Fill the histogram from binned data

        Arguments
        ----------
        content : ArrayLike
            The content of the bins

        errors : ArrayLike, optional
            The errors of the bins; if None, assume Poisson statistics (default).

        Returns
        -------
        AbstractHistogram
            The histogram instance.
        """
        if content.shape != self._shape:
            raise ValueError("Shape of content does not match number of bins")
        self._sumw = content
        if errors is None:
            self._sumw2 = content
        else:
            if errors.shape != self._shape:
                raise ValueError("Shape of errors does not match number of bins")
            self._sumw2 = errors**2
        return self

    def copy(self, label: str = None) -> "AbstractHistogram":
        """Create a full copy of a histogram.

        Arguments
        ---------
        label : str
            the label for the copied histogram. If None (default), the label of the
            original histogram is used.
        """
        # pylint: disable=protected-access
        # Note we really need the * in the constructor, here, as the abstract
        # base class is never instantiated, and the arguments are unpacked in the
        # constructors of all the derived classes.
        histogram = self.__class__(*self._edges, label or self.label, *self.axis_labels)
        histogram._sumw = self._sumw.copy()
        histogram._sumw2 = self._sumw2.copy()
        return histogram

    def _axis_modulo(self, axis: int) -> int:
        """Normalize the axis index modulo the number of axes, so that we can seamlessly
        use negative indices as well, following the normal Python convention.

        Arguments
        ---------
        axis : int
            the axis index to normalize.

        Returns
        -------
        int
            the normalized axis index.
        """
        return axis % self._num_axes

    def slice1d(self, *bin_indices: int, axis: int = -1) -> "Histogram1d":
        """Extract a 1D histogram along a given axis for the specified bin.

        Arguments
        ---------
        bin_indices : int
            the bin indices.

        axis : int
            the axis along which to extract the column (default: -1, i.e., the last axis).

        Returns
        -------
        hist : Histogram1d
            the one-dimensional histogram along the specified axis for the given bin indices.
        """
        # Check the input arguments: we need exactly num_axes - 1 bin indices,
        # and assume that the axis, if a non-default value is given, is given as
        # a keyword argument.
        if len(bin_indices) != self._num_axes - 1:
            raise ValueError(f"Exactly {self._num_axes - 1} bin indices are required.")
        axis = self._axis_modulo(axis)
        # Generate the list of axes other than the specified one to check the bin indices.
        axes = list(range(self._num_axes))
        axes.remove(axis)
        for _ax, _bin in zip(axes, bin_indices):
            if not 0 <= _bin < self._shape[_ax]:
                raise IndexError(f"Bin index out of range for axis {_ax}.")
        # We are good to go: we build the new, one-dimensional histogram...
        edges = self._edges[axis]
        if self.label is None:
            label = f"Slice at bins {bin_indices}"
        else:
            label = f"{self.label} slice at bins {bin_indices}"
        hist = Histogram1d(edges, label=label, xlabel=self.axis_labels[axis])
        # ... and set the actual content.
        indices = list(bin_indices)
        indices.insert(axis, slice(None))
        indices = tuple(indices)
        hist.set_content(self.content[indices], self.errors[indices])
        return hist

    @abstractmethod
    def _projection_hist_class(self):
        """Return the class of the histogram resulting from a projection.

        Note that, in order to be able to be instantiated, all subclasses of
        AbstractHistogram must implement this method.
        """

    def _empty_projection_histogram(self, axis: int) -> "AbstractHistogram":
        """Return an empty histogram for the projection along the specified axis.

        Arguments
        ---------
        axis : int
            the axis along which to project.

        Returns
        -------
        AbstractHistogram
            The empty histogram for projection along the specified axis.
        """
        axis = self._axis_modulo(axis)
        edges = [self._edges[ax] for ax in range(self._num_axes) if ax != axis]
        hist_type = self._projection_hist_class()
        labels = [self.axis_labels[ax] for ax in range(self._num_axes) if ax != axis]
        kwargs = dict(xlabel=labels[0])
        if self._num_axes > 2:
            kwargs["ylabel"] = labels[1]
        histogram = hist_type(*edges, **kwargs)
        return histogram

    def _expand_bin_centers(self, axis: int) -> np.ndarray:
        """Expand the dimensions of the bin centers along the specified axis in
        order to make them broadcastable with the histogram content.

        If we have a 3-dimensional histogram with binning [1., 2., 3.] along the
        z axis and we expand the latter along axis=2, we get an array with shape
        (1, 1, 3) containing the bin centers themselves.

        Arguments
        ---------
        axis : int
            the axis along which to expand the bin centers.

        Returns
        -------
        np.ndarray
            the expanded bin centers, broadcastable with the histogram content.
        """
        axis = self._axis_modulo(axis)
        bin_centers = self.bin_centers(axis)
        axes = [ax for ax in range(self._num_axes) if ax != axis]
        return np.expand_dims(bin_centers, axis=axes)

    def _project_base(self, axis: int, values: np.ndarray) -> "AbstractHistogram":
        """Base method for projecting a given quantity along a specified axis.

        Arguments
        ---------
        axis : int
            the axis along which to project.

        values : np.ndarray
            the values to set in the projected histogram.

        Returns
        -------
        AbstractHistogram
            the histogram containing the projected values.
        """
        axis = self._axis_modulo(axis)
        histogram = self._empty_projection_histogram(axis)
        histogram.set_content(values)
        return histogram

    def project_mean(self, axis: int = -1) -> "AbstractHistogram":
        """Project the (binned) mean along a specific axis over the remaining axes.

        Arguments
        ---------
        axis : int
            the axis along which to project (default: -1, i.e., the last axis).

        Returns
        -------
        AbstractHistogram
            the histogram containing the mean values along the specified axis.
        """
        axis = self._axis_modulo(axis)
        bin_centers = self._expand_bin_centers(axis)
        norm = np.sum(self.content, axis=axis)
        # Ignore the division warnings---the nan values will be changed to 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.sum(self.content * bin_centers, axis=axis) / norm
        mean = np.nan_to_num(mean, nan=0.)
        return self._project_base(axis, mean)

    def project_statistics(self, axis: int = -1) -> "AbstractHistogram":
        """Project the (binned) statistics along a specific axis over the remaining axes.

        Note we do the mean and rms in the same function as the latter needs the
        former anyway. If you only need the mean, consider using project_mean() instead.

        .. warning::

           This is numerically instable, as we are accumulating squares of bin
           centers. We should probably think carefully about a better way to do this.

        Arguments
        ---------
        axis : int
            the axis along which to project (default: -1, i.e., the last axis).

        Returns
        -------
        mean_hist : AbstractHistogram
            the histogram containing the mean values along the specified axis.

        rms_hist : AbstractHistogram
            the histogram containing the RMS values along the specified axis.
        """
        axis = self._axis_modulo(axis)
        bin_centers = self._expand_bin_centers(axis)
        norm = np.sum(self.content, axis=axis)
        # Ignore the division warnings, the nan values will be changed to 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = np.sum(self.content * bin_centers, axis=axis) / norm
            sum2 = np.sum(self.content * bin_centers**2, axis=axis) / norm
        mean = np.nan_to_num(mean, nan=0.)
        sum2 = np.nan_to_num(sum2, nan=0.)
        rms = np.sqrt(sum2 - mean**2)
        return self._project_base(axis, mean), self._project_base(axis, rms)

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
        # It is not immediately obvious what the right label for the new histogram
        # should be in all possible cases, so we just drop it.
        histogram.label = None
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
        # It is not immediately obvious what the right label for the new histogram
        # should be in all possible cases, so we just drop it.
        histogram.label = None
        histogram -= other
        return histogram

    def plot(self, axes: matplotlib.axes.Axes = None, **kwargs) -> None:
        """Overloaded plot() method.

        Before the actual plotting, this method sets some default plotting options
        specific to the histogram type. Subclasses can override the
        DEFAULT_PLOT_OPTIONS class attribute to provide their own defaults.
        """
        for key, value in self.DEFAULT_PLOT_OPTIONS.items():
            kwargs.setdefault(key, value)
        AbstractPlottable.plot(self, axes, **kwargs)

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

    # Note neither alpha nor histtype can be configured via style sheets,
    # so we have to set the defaults here.
    DEFAULT_PLOT_OPTIONS = dict(alpha=0.4, histtype="stepfilled")

    def __init__(self, xedges: np.ndarray, label: str = None, xlabel: str = None,
                 ylabel: str = "Entries/bin") -> None:
        """Constructor.
        """
        super().__init__((xedges, ), label, [xlabel, ylabel])

    def _projection_hist_class(self):
        """Overloaded method.
        """
        raise NotImplementedError("1D histograms cannot be projected.")

    @classmethod
    def from_amptek_file(cls, file_path: PathLike) -> "Histogram1d":
        """Return a Histogram1d filled with ADC counts from a file acquired with
        the Amptek MCA8000A Multichannel Analyzer, see
        https://www.amptek.com/internal-products/mca8000a-multichannel-analyzer-software-downloads

        Arguments
        ----------
        file_path : PathLike
            The path of the file to read.

        Returns
        -------
        Histogram1d
            A Histogram1d object with bins corresponding to ADC channels and filled
            with the counts from the file.
        """
        with open(file_path, encoding="UTF-8") as input_file:
            lines = input_file.readlines()
        start = lines.index("<<DATA>>\n")+1
        stop = lines.index("<<END>>\n")

        adc_counts = np.array(lines[start:stop], dtype=float)
        num_channels = len(adc_counts)
        xedges = np.arange(-0.5, num_channels + 0.5)
        hist = cls(xedges=xedges, xlabel="ADC Channel")
        return hist.set_content(adc_counts)

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

    def fwhm(self) -> float:
        """Return the full width at half maximum (FWHM) of the histogram.

        If the distribution is not well behaved (e.g., the maximum is at the edge or it is flat),
        a ValueError is raised.

        Returns
        -------
        fwhm : float
            The full width at half maximum of the histogram.
        """
        x = self.bin_centers()
        y = self.content
        indices = np.arange(len(y))
        imax = np.argmax(y)

        # Calculate the value of the half maximum---note this will be subject to
        # statistical fluctuations, and we make no attempt at smoothing here.
        half_max = 0.5 * y[imax]
        # Create slices for the left and right parts of the histogram (where left
        # and right really means left and right of the bin with the maximum value)...
        left_slice = slice(None, imax)
        right_slice = slice(imax, None)
        # ... and cache the corresponding values.
        left_values = y[left_slice]
        right_values = y[right_slice]

        # Create proper masks identifying the values that are below half maximum
        # and greater than zero on both sides of the histogram maximum.
        left_mask = np.logical_and(left_values < half_max, left_values > 0.)
        right_mask = np.logical_and(right_values < half_max, right_values > 0.)
        # If the maximum of the distribution is at the edge of the histogram,
        # or if the distribution does not cross half maximum on both sides, we
        # just cannot compute the FWHM, and we give up.
        if not np.any(left_mask) or not np.any(right_mask):
            raise ValueError("FWHM cannot be computed for the histogram")

        # At this point the first crude estimates for the indices of the crossing
        # points are just the last and first indices where the masks are true on
        # the left and right slices, respectively. (Note the indices are sorted by
        # construction, so we can just use -1 and 0 to get the right values, and
        # we do not need to use min() and max()).
        il = indices[left_slice][left_mask][-1]
        ir = indices[right_slice][right_mask][0]

        # Refine using interpolation.
        xl = x[il] + (half_max - y[il]) / (y[il + 1] - y[il]) * (x[il + 1] - x[il])
        xr = x[ir] - (half_max - y[ir]) / (y[ir - 1] - y[ir]) * (x[ir] - x[ir - 1])
        return xr - xl

    def _normalized_cumsum(self) -> np.ndarray:
        """Return the normalized cumulative sum of the histogram contents.

        Returns
        -------
        cumsum : np.ndarray
            the normalized cumulative sum of the histogram contents.
        """
        # We add another bin at the beginning to match the edges array dimension.
        cumsum = np.insert(np.cumsum(self.content), 0, 0.)
        cumsum /= cumsum[-1]
        return cumsum

    def cdf(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the cumulative distribution function (CDF) of the histogram
        at the specified values.

        Internally we are using a PCHIP interpolation to avoid oscillations and
        preserve monotonicity. Note we are deliberately not making the creation
        of the interpolator a cached property (which is only supported in Python
        3.8+, by the way) as the histogram content might change after the each call,
        and we want to always have the up-to-date CDF.

        Arguments
        ---------
        x : ArrayLike
            the values where to evaluate the cdf.

        Returns
        -------
        cdf : ArrayLike
            the cumulative distribution function (CDF) of the histogram evaluated at x.
        """
        # Create the interpolator on the fly.
        cdf_interpolator = PchipInterpolator(self.bin_edges(), self._normalized_cumsum())
        # Evaluate the interpolator on the input grid and ensure that we return
        # 0 and 1 for values outside the histogram range.
        return np.clip(cdf_interpolator(x), 0.0, 1.0)

    def ppf(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the percent point function (PPF) of the histogram at the specified
        values.

        The PPF is calculated by interpolating the inverse of the cumulative sums of the
        histogram contents.

        Note that the PPF is only defined in the [0, 1] domain. For values outside
        this range, NaN is returned. The same notes about the CDF interpolation
        apply here, namely: internally we are using a PCHIP interpolation to avoid
        oscillations and preserve monotonicity. We are deliberately not making the
        creation of the interpolator a cached property (which is only supported in Python
        3.8+, by the way) as the histogram content might change after the each call,
        and we want to always have the up-to-date PPF.

        Arguments
        ---------
        x : ArrayLike
            the values where to evaluate the ppf.

        Returns
        -------
        ppf : ArrayLike
            the percent point function (PPF) of the histogram evaluated at x.
        """
        # Create the interpolator on the fly.
        _x, _idx = np.unique(self._normalized_cumsum(), return_index=True)
        _y = self.bin_edges()[_idx]
        ppf_interpolator = PchipInterpolator(_x, _y)
        # Ensure that we return NaN for values outside the [0, 1] domain.
        results = np.where((x >= 0) & (x <= 1), ppf_interpolator(x), np.nan)
        if np.isscalar(x):
            return results.item()
        return results

    def _coverage_interval(self, x: ArrayLike, coverage: float) -> ArrayLike:
        """Calculate the coverage interval width, given the lower edge of the interval.

        Arguments
        ---------
        x : ArrayLike
            the lower edges where to calculate the interval widths.
        coverage : float
            the coverage of the interval

        Returns
        -------
        delta : ArrayLike
            the widths of the interval
        """
        if coverage < 0. or coverage > 1.:
            raise ValueError("Coverage must be between 0 and 1.")
        return self.ppf(coverage + self.cdf(x)) - x

    def minimum_coverage_interval(self, coverage: float) -> Tuple[float, float]:
        """Calculate the minimum coverage interval of the histogram for a given coverage.

        Arguments
        ---------
        coverage : float
            the coverage of the interval

        Returns
        -------
        xmin, xmax : Tuple[float, float]
            the left and right edges of the minimum coverage interval.
        """
        # If the coverage is 1., return the full range of the histogram with non-zero
        # content.
        if coverage == 1.:
            edges = self.bin_edges()
            cumsum = self._normalized_cumsum()
            xmin = edges[cumsum > 0][0]
            xmax = edges[cumsum == 1][0]
        else:
            xa = self.bin_edges()[self._normalized_cumsum() > 0.][0]
            xb = self.ppf(1. - coverage)
            res = minimize_scalar(self._coverage_interval, args=(coverage,), bounds=(xa, xb),
                                  method="bounded")
            xmin = res.x
            xmax = xmin + res.fun
        return xmin, xmax

    def __isub__(self, other: Union["Histogram1d", Callable]) -> "Histogram1d":
        """Overloaded in-place subtraction operator.

        Here we allow subtracting either another histogram (as in the base class)
        or a callable object (e.g., a fitting model) that is evaluated at the bin
        centers.
        """
        if isinstance(other, Histogram1d):
            return super().__isub__(other)
        if callable(other):
            # Assume other is a model with no uncertainties. We evaluate the model
            # at the bin centers and subtract the result from the histogram content.
            # The bin errors stay unchanged.
            self._sumw -= other(self.bin_centers())
            return self
        raise NotImplementedError(f"Cannot subtract {type(other)} from Histogram1d")

    def plot(self, axes: matplotlib.axes.Axes = None, statistics: bool = False,
             errors: bool = False, **kwargs) -> None:
        """Overloaded plot() method.

        This method adds an option to include basic statistics (mean and RMS) in the
        legend entry. Note that, apart from this addition, the method behaves as the
        base class dictates.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            the axes where to plot the histogram (default: current axes).

        statistics : bool, optional
            whether to include basic statistics (mean and RMS) in the legend entry
            (default: False).

        errors : bool, optional
            whether to overplot the error bars (default: False).

        kwargs : keyword arguments
            additional keyword arguments passed to the plotting backend.
        """
        kwargs.setdefault("label", self.label)
        label = kwargs["label"]
        if label is not None and statistics:
            mean, rms = self.binned_statistics()
            kwargs["label"] = f"{label}\nMean: {mean:g}\nRMS: {rms:g}"
        super().plot(axes, **kwargs)
        if errors:
            # Need to recover the color and alpha used in the histogram
            # plotting to make the error bars match.
            color = kwargs.get("color", plt.rcParams["patch.edgecolor"])
            alpha = kwargs.get("alpha", self.DEFAULT_PLOT_OPTIONS["alpha"])
            self._render_errors(axes, fmt=',', color=color, alpha=alpha)

    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Overloaded method.
        """
        axes.hist(self.bin_centers(0), self._edges[0], weights=self.content, **kwargs)

    def _render_errors(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Small convenience function to overplot the error bars on the histogram.
        """
        axes.errorbar(self.bin_centers(0), self._sumw, self.errors, **kwargs)

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

    def _projection_hist_class(self):
        """Overloaded method.
        """
        return Histogram1d

    def _render(self, axes: matplotlib.axes.Axes, logz: bool = False, **kwargs) -> None:
        """Overloaded method.
        """
        # pylint: disable=arguments-differ
        if logz:
            vmin = kwargs.pop("vmin", None)
            vmax = kwargs.pop("vmax", None)
            kwargs.setdefault("norm", matplotlib.colors.LogNorm(vmin, vmax))
        mappable = axes.pcolormesh(*self._edges, self.content.T, **kwargs)
        plt.colorbar(mappable, ax=axes, label=self.axis_labels[2])


class Histogram3d(AbstractHistogram):

    """Three-dimensional histogram.

    Arguments
    ---------
    xedges : 1-dimensional array
        the bin edges on the x axis.

    yedges : 1-dimensional array
        the bin edges on the y axis.

    zedges : 1-dimensional array
        the bin edges on the z axis.

    label : str
        overall label for the histogram

    xlabel : str
        the text label for the x axis.

    ylabel : str
        the text label for the y axis.

    zlabel : str
        the text label for the z axis (default: None).
    """

    def __init__(self, xedges, yedges, zedges, label: str = None, xlabel: str = None,
                 ylabel: str = None, zlabel: str = None) -> None:
        """Constructor.
        """
        super().__init__((xedges, yedges, zedges), label, [xlabel, ylabel, zlabel])

    def _projection_hist_class(self):
        """Overloaded method.
        """
        return Histogram2d

    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Overloaded method.

        Note that 3D histograms cannot be directly plotted, so we just raise
        a NotImplementedError.
        """
        raise NotImplementedError("3D histograms cannot be plotted yet.")
