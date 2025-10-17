"""
Constrained fit
===============

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
# Fix the prefactor. This is a generally useful technique, as you should
# never fit the normalization of a histogram.
model.prefactor.freeze(hist.area() / model.integral(-5., 5.))
model.fit_histogram(hist)
print(model)
model.plot()

plt.legend()