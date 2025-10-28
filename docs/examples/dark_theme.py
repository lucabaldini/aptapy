"""
Dark theme
==========

Simple gaussian fit to histogram data with a dark theme.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import Gaussian
from aptapy.plotting import apply_stylesheet, plt

apply_stylesheet("aptapy.styles.aptapy-dark")

hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
hist.fill(np.random.default_rng().normal(size=100000))
hist.plot(statistics=True)

model = Gaussian()
model.fit_histogram(hist)
print(model)
# Plot the model, including the fit output in the legend.
model.plot(fit_output=True)

plt.legend()
