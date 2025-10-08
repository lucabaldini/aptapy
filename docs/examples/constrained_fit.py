"""
Constrained fit example
=======================

Gaussian fit to histogram data where the prefactor is frozen based on the
histogram normalization.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import Gaussian
from aptapy.plotting import plt

hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
hist.fill(np.random.default_rng().normal(size=100000))
hist.plot()

model = Gaussian()
# Fix the prefactor to the histogram area---note this only works because the
# Gaussian model is normalized to 1 over the full range when the prefactor is 1.
model.prefactor.freeze(hist.area())
model.fit_histogram(hist)
print(model)
model.plot()

plt.legend()