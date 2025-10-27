"""
Simple 1-D histogram
====================

Simple one-dimensional histogram filled with random numbers from a normal
distribution.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.plotting import plt

# Create and fill the histogram...
hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
hist.fill(np.random.default_rng().normal(size=100000))
# ...and plot it including the basic binned statistics in the legend.
hist.plot(statistics=True)

plt.legend()