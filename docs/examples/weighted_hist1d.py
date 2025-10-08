"""
Weighted 1-D histogram
======================

Weighted one-dimensional histogram to emulate a triangular distribution.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.plotting import plt

hist = Histogram1d(np.linspace(0., 1., 100), label="Random data", xlabel="u")
# Histogram sample: extract n random numbers uniformly distributed in [0, 1)
# and assign weights w = 1 - u (that is, w = 1 for u = 0 and 0 for u = 1).
# The result should be equivalent to sampling a triangular distribution.
n = 100000
sample = np.random.default_rng().random(size=n)
weights = 1. - sample
hist.fill(sample, weights=weights)
hist.plot()

plt.legend()