"""
Exponential fit
===============

Fit some real data with exponential and stretched exponential models.
"""

# %%

import pathlib

import numpy as np

from aptapy.models import Constant, Exponential, StretchedExponential
from aptapy.plotting import setup_axes, residual_axes


# These are real data recorded with a thermistor as it cools down in open air.
file_path = pathlib.Path.cwd() / "data" / "exponential_data.txt"
t, T = np.loadtxt(file_path, unpack=True)

# Setup two different models: a simple exponential + constant (Newton law)
# and a stretched exponential + constant.
simple_model = Constant() + Exponential()
simple_model.fit(t, T, sigma=0.1)

stretched_model = Constant() + StretchedExponential()
stretched_model.fit(t, T, sigma=0.1)

# The fit and compare the two---the latter is clearly better.
_, ax1, ax2 = residual_axes(height_ratio=0.3)
ax1.errorbar(t, T, fmt="o", label="Data", alpha=0.75, ms=5., color="gray")
simple_model.plot(ax1, fit_output=True, plot_components=False, zorder=10)
stretched_model.plot(ax1, fit_output=True, plot_components=False, zorder=10)
setup_axes(ax1, ylabel="Temperature [°C]", legend=True)

ax2.errorbar(t, T - simple_model(t), fmt="o", ms=5.)
ax2.errorbar(t, T - stretched_model(t), fmt="o", ms=5.)
setup_axes(ax2, xlabel="Time [s]", ylabel="Residuals [°C]", ymin=-1.25, ymax=1.25)
