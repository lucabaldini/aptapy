"""
Histogram arithmetics
=====================

Elaborate example creating a signal and background histogram, fitting a background model,
and subtracting it from the total histogram.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.models import Exponential, Gaussian
from aptapy.plotting import residual_axes, setup_gca

# Custom command to create two vertically aligned axes with shared x-axis.
fig, ax1, ax2 = residual_axes()

# Create signal and background histograms, and fill them with random data.
binning = np.linspace(0., 10., 100)
signal = Histogram1d(binning, label="Signal", xlabel="x [a. u.]")
signal.fill(np.random.default_rng().normal(loc=6., scale=0.5, size=10000))
background = Histogram1d(binning, label="Background")
background.fill(np.random.default_rng().exponential(scale=3., size=100000))

# Sum the two histograms to create the total histogram, and fit an exponential
total = signal + background
background_model = Exponential()
# Note the use of xmin > xmax to fit outside the signal region: this will actually
# ignore all data between 4 and 8.
background_model.fit(total, xmin=8, xmax=4.)
print(background_model)

# Plot the histograms and the fit results on the upper axes.
total.plot(ax1, label="Total")
background_model.plot(ax1, fit_output=True)
signal.plot(ax1, alpha=0.2, ls='dotted')
background.plot(ax1, alpha=0.2, ls='dotted')
ax1.legend()

# Subtract the background model from the total histogram, and plot the result,
sub_signal = total - background_model
# Note the errors argument, which cause the error bars to be overplotted to
# the histogram.
sub_signal.plot(ax2, errors=True, label="Subtracted signal")

# Fit the subtracted histogram with a Gaussian.
model = Gaussian()
model.fit(sub_signal)
print(model)
model.plot(ax2, fit_output=True)
ax2.legend()

# Setup the x-axis limits.
setup_gca(xmin=0., xmax=10.)
