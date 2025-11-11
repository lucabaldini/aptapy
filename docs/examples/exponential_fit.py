"""
Exponential fit
===============


"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.models import Gaussian
from aptapy.plotting import plt


file_path = "data/exponential_data.txt"
data = np.loadtxt(file_path)

print(data)

hist = Histogram1d(np.linspace(-5., 5., 100), label="Random data", xlabel="z")
hist.fill(np.random.default_rng().normal(size=100000))
hist.plot(statistics=True)

model = Gaussian()
model.fit_histogram(hist)
print(model)
# Plot the model, including the fit output in the legend.
model.plot(fit_output=True)

plt.legend()
