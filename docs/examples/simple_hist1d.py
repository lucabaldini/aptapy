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

hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
hist.fill(np.random.default_rng().normal(size=100000))
hist.plot()

plt.legend()