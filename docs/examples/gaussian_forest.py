"""
Gaussian forest
===============

Definition of a GaussianForest child class and creation and fit of a histogram from the
given model.
"""

# %%

from aptapy.modeling import line_forest
from aptapy.models import GaussianForest
from aptapy.plotting import plt

@line_forest(1., 3.)
class ExampleForest(GaussianForest):
    """Example of a GaussianForest child class with lines centeres in 1. and 2. [a.u.]
    """

# Instantiate the class and initialize the parameters
model = ExampleForest()
model.amplitude1.init(0.1)
# Generate a random histogram with the given parameters
hist = model.random_histogram(size=100000, num_bins=100)
hist.plot()

model.fit(hist)
model.plot(fit_output=True)

plt.legend()
