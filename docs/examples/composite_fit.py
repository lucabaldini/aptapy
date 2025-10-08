"""
Composite fit example
=====================

Short narrative paragraph shown above the thumbnail.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import Gaussian, Line
from aptapy.plotting import plt

hist = Histogram1d(np.linspace(-5., 5., 100), label="Test data")
rng = np.random.default_rng(313)
hist.fill(rng.normal(size=100000))
hist.fill(5. - 10. * np.sqrt(1 - rng.random(100000)))
model = Gaussian() + Line()
hist.plot()
model.fit_histogram(hist)
print(model)
model.plot()
plt.legend()