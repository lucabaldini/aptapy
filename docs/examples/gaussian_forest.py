"""
Gaussian forest
===============

Illustration of the GaussianForestBase class and the line_forest decorator.
"""

# %%

import numpy as np

from aptapy.modeling import line_forest
from aptapy.models import GaussianForestBase
from aptapy.plotting import plt

@line_forest(5., 8.)
class ExampleForest(GaussianForestBase):
    """Example of a GaussianForestBase child class with lines centered at 1. and 3. [a.u.]
    """

# Instantiate the class and initialize the parameters
model = ExampleForest()
model.intensity1.init(0.3)
model.sigma.init(0.75)

# Generate a random histogram with the given parameters
hist = model.random_histogram(np.linspace(0., 10., 100), size=100000)
hist.label = "Random data"
hist.xlabel = "x"
hist.plot()

model.fit(hist)
model.plot(fit_output=True)
plt.legend()
