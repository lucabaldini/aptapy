"""
Simple fit example
==================

Simple gaussian fit to histogram data.
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
model.fit_histogram(hist)
print(model)
model.plot()

plt.legend()