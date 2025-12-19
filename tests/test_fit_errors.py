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

"""Unit tests for the fit error posteriors.
"""

import numpy as np
import pytest

from aptapy import models
from aptapy.hist import Histogram1d
from aptapy.modeling import AbstractFitModelBase
from aptapy.plotting import plt


def _test_pulls(model: AbstractFitModelBase, ground_truth: dict, sample_size: int = 10000,
                num_realizations: int = 1000, debug: bool = False) -> None:
    """Generate many independent random datasets, fit them, and plot the pull
    distributions of the best-fit parameters.
    """
    # Initialize the pulls.
    pulls = {key: [] for key in ground_truth}

    # Loop over the random realizations.
    for _ in range(num_realizations):
        # Reset the parameters to the ground-truth values.
        for name, value in ground_truth.items():
            getattr(model, name).set(value)
        # Generate a random dataset and fit it.
        _hist = model.random_histogram(sample_size)
        try:
            model.fit(_hist)
            if debug:
                _hist.plot(label="Random data")
                model.plot(fit_output=True)
                plt.legend()
                plt.show()
        except RuntimeError:
            pass
        # Compute the pulls and store them.
        for name, value in ground_truth.items():
            pulls[name].append(getattr(model, name).pull(value))

    # Plot the pull distributions.
    for name, values in pulls.items():
        plt.figure(f"{model.name()}_{name}_pulls")
        hist = Histogram1d(np.linspace(-5., 5., 50)).fill(values)
        hist.plot()
        gauss = models.Gaussian()
        gauss.fit(hist)
        gauss.plot(fit_output=True)
        plt.legend()


@pytest.mark.skip("This is specifically for debugging purposes")
def test_gaussian_pulls() -> None:
    """Test the pulls for the Gaussian model.
    """
    model = models.Gaussian()
    ground_truth = dict(mu=10., sigma=2.)
    _test_pulls(model, ground_truth)


@pytest.mark.skip("This is specifically for debugging purposes")
def test_fe55_pulls() -> None:
    """Test the pulls for the Fe-55 model.
    """
    # pylint: disable=no-member
    model = models.Fe55Forest()
    model.intensity1.freeze(0.2)
    ground_truth = dict(energy_scale=1., sigma=0.5)
    _test_pulls(model, ground_truth, debug=False)
