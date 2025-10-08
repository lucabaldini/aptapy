"""
Simple fit example
==================

Short narrative paragraph shown above the thumbnail.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import Gaussian
from aptapy.plotting import plt

hist = Histogram1d(np.linspace(-5., 5., 100), label="Test data", xlabel="x")
rng = np.random.default_rng(313)
hist.fill(rng.normal(size=100000))
model = Gaussian()
hist.plot()
model.fit_histogram(hist)
print(model)
model.plot()
plt.legend()