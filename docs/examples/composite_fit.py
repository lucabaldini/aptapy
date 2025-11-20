"""
Composite fit
=============

Composite fit with a Gaussian plus a straight line.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.models import Gaussian, Line
from aptapy.plotting import plt

hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
# Fill with the sum of a gaussian...
hist.fill(np.random.default_rng().normal(size=100000))
# ... and a triangular distribution (this is done via the inverse transform method).
hist.fill(5. - 10. * np.sqrt(1 - np.random.default_rng().random(100000)))
hist.plot()

model = Gaussian() + Line()
model.fit(hist)
print(model)
model.plot(fit_output=True)

plt.legend()