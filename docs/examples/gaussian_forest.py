"""
Gaussian forest
===============

Illustration of the GaussianForestBase class and the line_forest decorator.
"""

# %%

from aptapy.modeling import line_forest
from aptapy.models import GaussianForestBase
from aptapy.plotting import plt

@line_forest(1., 3.)
class ExampleForest(GaussianForestBase):
    """Example of a GaussianForestBase child class with lines centered at 1. and 3. [a.u.]
    """

# Instantiate the class and initialize the parameters
model = ExampleForest()
model.intensity1.init(0.3)
model.sigma.init(0.75)

# Generate a random histogram with the given parameters
hist = model.random_histogram(size=100000, num_bins=100)
hist.label = "Random data"
hist.xlabel = "x"
hist.plot()

model.fit(hist)
model.plot(fit_output=True)
plt.legend()
