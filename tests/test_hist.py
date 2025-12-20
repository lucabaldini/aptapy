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

"""Unit tests for the hist module.
"""

import inspect

import numpy as np
import pytest

from aptapy.hist import Histogram1d, Histogram2d, Histogram3d
from aptapy.models import Gaussian
from aptapy.plotting import plt

_RNG = np.random.default_rng(313)


def test_init1d():
    """Test all the initialization cross checks.
    """
    edges = np.array([[1., 2.], [3., 4]])
    with pytest.raises(ValueError, match="not a 1-dimensional array"):
        _ = Histogram1d(edges)
    edges = np.array([1.])
    with pytest.raises(ValueError, match="less than 2 entries"):
        _ = Histogram1d(edges)
    edges = np.array([2., 1.])
    with pytest.raises(ValueError, match="not strictly increasing"):
        _ = Histogram1d(edges)


def test_binning1d():
    """Test the binning-related methods.
    """
    edges = np.linspace(0., 1., 11)
    hist = Histogram1d(edges)
    assert np.allclose(hist.content, 0.)
    assert np.allclose(hist.errors, 0.)
    assert np.allclose(hist.bin_centers(), np.linspace(0.05, 0.95, 10))
    assert np.allclose(hist.bin_widths(), 0.1)


def test_empty1d():
    """Test the empty histogram state.

    See issue https://github.com/lucabaldini/aptapy/issues/15
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    hist = Histogram1d(np.linspace(0., 1., 11), label="Empty histogram")
    hist.plot(statistics=True)
    plt.legend()


def test_filling1d():
    """Simple filling test with a 1-bin, 1-dimensional histogram.
    """
    hist = Histogram1d(np.linspace(0., 1., 2))
    # Fill with a numpy array.
    hist.fill(np.full(100, 0.5))
    assert hist.content == 100.
    # Fill with a number.
    hist.fill(0.5)
    assert hist.content == 101.


def test_setting_content1d():
    """Test setting the content of a 2-bin, 1-dimensional histogram.
    """
    hist = Histogram1d(np.linspace(0., 2., 3))
    content = np.array([10, 20])
    hist.set_content(content)
    assert np.array_equal(hist.content, content)
    assert np.array_equal(hist.errors, np.sqrt(content))

    errors = np.array([1, 1])
    hist.set_content(content, errors)
    assert np.array_equal(hist.content, content)
    assert np.array_equal(hist.errors, errors)


def test_compat1d():
    """Test the histogram compatibility.
    """
    # pylint: disable=protected-access
    hist = Histogram1d(np.array([0., 1., 2]))
    hist._check_compat(hist.copy())
    with pytest.raises(TypeError, match="not a histogram"):
        hist._check_compat(None)
    with pytest.raises(ValueError, match="dimensionality/shape mismatch"):
        hist._check_compat(Histogram1d(np.array([0., 1., 2., 3.])))
    with pytest.raises(ValueError, match="bin edges differ"):
        hist._check_compat(Histogram1d(np.array([0., 1.1, 2.])))


def test_arithmetics1d():
    """Test the basic arithmetics.
    """
    # pylint: disable=protected-access
    sample1 = _RNG.uniform(size=10000)
    sample2 = _RNG.uniform(size=10000)
    edges = np.linspace(0., 1., 100)
    hist1 = Histogram1d(edges).fill(sample1)
    hist2 = Histogram1d(edges).fill(sample2)
    hist3 = Histogram1d(edges).fill(sample1).fill(sample2)
    hist_sum = hist1 + hist2
    hist_sub = hist1 - hist1
    assert np.allclose(hist_sum._sumw, hist3._sumw)
    assert np.allclose(hist_sum._sumw2, hist3._sumw2)
    assert np.allclose(hist_sub._sumw, 0.)


def test_plotting1d(size: int = 100000):
    """Test plotting.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    # Create the first histogram. This has no label attached, and we do
    # provide one at plotting time.
    mean = 0.
    sigma = 1.
    hist1 = Histogram1d(np.linspace(-5., 5., 100), xlabel="x")
    hist1.fill(_RNG.normal(size=size, loc=mean, scale=sigma))
    hist1.plot(label="Standard histogram")
    m, s = hist1.binned_statistics()
    # Rough checks on the binned statistics---we want the mean to be within 10
    # sigma/sqrt(N) and the stddev to be within 2% of the true value.
    # (Note the binning has an effect on the actual values, so we cannot
    # expect perfect agreement.)
    assert abs((m - mean) / sigma * np.sqrt(size)) < 10.
    assert abs(s / sigma - 1.) < 0.02

    # Create a second histogram, this time with a label---this should have a
    # proper entry in the legend automatically.
    mean = 1.
    sigma = 1.5
    hist2 = Histogram1d(np.linspace(-5., 5., 100), label="Offset histogram")
    hist2.fill(_RNG.normal(size=size, loc=mean, scale=sigma))
    hist2.plot(statistics=True)
    m, s = hist2.binned_statistics()
    assert abs((m - mean) / sigma * np.sqrt(size)) < 10.
    assert abs(s / sigma - 1.) < 0.02

    # And this one should end up with no legend entry, as we are explicitly
    # providing label=None at plotting time.
    mean = -1.
    sigma = 0.5
    hist3 = Histogram1d(np.linspace(-5., 5., 100))
    hist3.fill(_RNG.normal(size=size, loc=mean, scale=sigma))
    hist3.plot(label=None)
    m, s = hist3.binned_statistics()
    assert abs((m - mean) / sigma * np.sqrt(size)) < 10.
    assert abs(s / sigma - 1.) < 0.02
    plt.legend()


def test_plotting2d(size: int = 100000, x0: float = 1., y0: float = -1.):
    """Test plotting.
    """
    plt.figure(inspect.currentframe().f_code.co_name)
    edges = np.linspace(-5., 5., 100)
    hist = Histogram2d(edges, edges, xlabel="x", ylabel="y")
    # Note we are adding different offsets to x and y so that we can see
    # the effect on the plot.
    hist.fill(_RNG.normal(size=size, loc=x0), _RNG.normal(size=size, loc=y0))
    hist.plot()
    mx, sx = hist.binned_statistics(0)
    my, sy = hist.binned_statistics(1)
    assert abs((mx - x0) * np.sqrt(size)) < 10.
    assert abs((my - y0) * np.sqrt(size)) < 10.
    assert abs(sx - 1.) < 0.02
    assert abs(sy - 1.) < 0.02
    plt.gca().set_aspect("equal")


def test_from_amptek_file(datadir):
    """Test building histogram from amptek file
    """
    plt.figure(inspect.currentframe().f_code.co_name)

    file_path = datadir / "amptek_test.mca"
    histogram = Histogram1d.from_amptek_file(file_path)
    histogram.plot()

    model = Gaussian()
    model.fit(histogram, xmin=20, xmax=35)
    model.plot(fit_output=True)
    dof = model.status.dof
    plt.legend()

    mean, std = histogram.binned_statistics()
    assert mean != 0
    assert std != 0
    assert model.status.chisquare - dof <= 5 * np.sqrt(2 * dof)


def test_slice1d():
    """Test extracting a 1D slice from a 2-dimensional histogram.
    """
    # Test parameters.
    sample_size = 10000
    bin_index = 5

    # Create and fill a 2-dimensional histogram.
    xedges = np.linspace(0., 1., 11)
    yedges = np.linspace(0., 1., 21)
    hist = Histogram2d(xedges, yedges, xlabel="x", ylabel="y")
    x = _RNG.uniform(size=sample_size)
    y = _RNG.uniform(0., 0.9, size=sample_size)
    hist.fill(x, y)

    # This is extracting a vertical slice at bin index 5.
    plt.figure(f"{inspect.currentframe().f_code.co_name}_vertical")
    vslice = hist.slice1d(bin_index)
    vslice.plot()
    assert vslice.content.shape == (len(yedges) - 1,)
    assert np.array_equal(vslice.content, hist.content[bin_index, :])
    assert np.array_equal(vslice.errors, hist.errors[bin_index, :])
    # Since we are at it, selecting axis=1 should give the same result as the default (-1).
    assert np.array_equal(vslice.content, hist.slice1d(bin_index, axis=1).content)

    plt.figure(f"{inspect.currentframe().f_code.co_name}_horizontal")
    hslice = hist.slice1d(bin_index, axis=0)
    hslice.plot()
    assert hslice.content.shape == (len(xedges) - 1,)
    assert np.array_equal(hslice.content, hist.content[:, bin_index])
    assert np.array_equal(hslice.errors, hist.errors[:, bin_index])

    # And if we pass two indices, we should fail miserably.
    with pytest.raises(ValueError, match="bin indices are required"):
        hist.slice1d(0, 1)


def test_project2d():
    """Test projecting a 2-dimensional histogram along the y axis.
    """
    # Test parameters
    sample_size = 10000
    num_xbins = 50
    num_ybins = 30
    bin_index = 1

    # Create the histogram and fill it with uniform random numbers.
    xedges = np.linspace(0., 1., num_xbins + 1)
    yedges = np.linspace(0., 1., num_ybins + 1)
    hist2d = Histogram2d(xedges, yedges, xlabel="x", ylabel="y")
    x, y = _RNG.uniform(0., 0.9, size=(2, sample_size))
    hist2d.fill(x, y)

    # Plot the original histogram and a test slice.
    plt.figure(f"{inspect.currentframe().f_code.co_name}_2d")
    hist2d.plot()
    plt.figure(f"{inspect.currentframe().f_code.co_name}_slice")
    vslice = hist2d.slice1d(bin_index)
    vslice.plot(statistics=True)
    plt.legend()

    # Do the actual projections.
    hist_meany, hist_rmsy = hist2d.project_statistics()
    assert hist_meany.content.shape == (num_xbins,)
    assert hist_rmsy.content.shape == (num_xbins,)
    # And, since we have a slice already, we can compare the binned statistics
    # for the one-dimensional slice with the projected ones in the proper bin.
    bin_mean, bin_rms = vslice.binned_statistics()
    assert hist_meany.content[bin_index] == pytest.approx(bin_mean)
    assert hist_rmsy.content[bin_index] == pytest.approx(bin_rms)
    plt.figure(f"{inspect.currentframe().f_code.co_name}_proj_vertical")
    hist_meany.plot(label="Mean y")
    hist_rmsy.plot(label="RMS y")
    plt.legend()

    hist_meanx, hist_rmsx = hist2d.project_statistics(axis=0)
    assert hist_meanx.content.shape == (num_ybins,)
    assert hist_rmsx.content.shape == (num_ybins,)
    plt.figure(f"{inspect.currentframe().f_code.co_name}_proj_horizontal")
    hist_meanx.plot(label="Mean x")
    hist_rmsx.plot(label="RMS x")
    plt.legend()


def test_hist3d():
    """Test basic functionalities of 3D histograms.
    """
    # Test parameters.
    sample_size = 10000
    num_xbins = 10
    num_ybins = 10
    dynamic_range = 10.
    bin_indices = (5, 5)

    # Create and fill a 3-dimensional histogram.
    xedges = np.arange(-0.5, 0.5 + num_xbins)
    yedges = np.arange(-0.5, 0.5 + num_ybins)
    zedges = np.linspace(-5., 5. + dynamic_range, 101)
    hist3d = Histogram3d(xedges, yedges, zedges, xlabel="x", ylabel="y", zlabel="z")
    for x in hist3d.bin_centers(0):
        for y in hist3d.bin_centers(1):
            mean = (x + y) / (num_xbins + num_ybins) * dynamic_range
            sample = _RNG.normal(loc=mean, scale=1., size=sample_size)
            hist3d.fill(np.full_like(sample, x), np.full_like(sample, y), sample)

    # Look at a sample, one-dimensional slice.
    plt.figure(f"{inspect.currentframe().f_code.co_name}_slice")
    slice_hist = hist3d.slice1d(*bin_indices)
    slice_hist.plot(statistics=True)
    plt.legend()

    # Test projecting the basic statistics over the x-y plane.
    hist_mean, hist_rms = hist3d.project_statistics()
    plt.figure(f"{inspect.currentframe().f_code.co_name}_mean")
    hist_mean.plot(label="Mean z")
    plt.figure(f"{inspect.currentframe().f_code.co_name}_rms")
    hist_rms.plot(label="RMS z")
    mean_slice, rms_slice = slice_hist.binned_statistics()
    assert hist_mean.content[bin_indices] == pytest.approx(mean_slice)
    assert hist_rms.content[bin_indices] == pytest.approx(rms_slice)
