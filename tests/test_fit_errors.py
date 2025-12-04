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

"""Unit tests for the fit error posteriors.
"""

import numpy as np

from aptapy import models
from aptapy.hist import Histogram1d
from aptapy.plotting import plt


def _test_pulls(model_class: type, ground_truth: dict, sample_size: int = 10000,
                num_realizations: int = 1000) -> None:
    """Generate many independent random datasets, fit them, and plot the pull
    distributions of the best-fit parameters.
    """
    # Create the model and initialize the pulls.
    model = model_class()
    pulls = {key: [] for key in ground_truth}

    # Loop over the random realizations.
    for i in range(num_realizations):
        # Reset the parameters to the ground-truth values.
        for name, value in ground_truth.items():
            model.__getattribute__(name).set(value)
        # Generate a random dataset and fit it.
        _hist = model.random_histogram(sample_size)
        try:
            model.fit(_hist)
        except RuntimeError:
            pass
        # Compute the pulls and store them.
        for name, value in ground_truth.items():
            pulls[name].append(model.__getattribute__(name).pull(value))

    # Plot the pull distributions.
    for name, values in pulls.items():
        plt.figure(f"{model.name()}_{name}_pulls")
        hist = Histogram1d(np.linspace(-5., 5., 51)).fill(values)
        hist.plot()
        gauss = models.Gaussian()
        gauss.fit(hist)
        gauss.plot(fit_output=True)
        plt.legend()


def test_gaussian_pulls() -> None:
    """Test the pulls for the Gaussian model.
    """
    ground_truth = dict(mu=10., sigma=2.)
    _test_pulls(models.Gaussian, ground_truth)