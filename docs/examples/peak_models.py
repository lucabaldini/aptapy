"""
Peak models
===========

Illustration of the available peak models with the standard location and scale
parameters
"""

# %%

from aptapy.plotting import plt
from aptapy.models import Gaussian, LogNormal, Lorentzian, Moyal

Gaussian().plot()
Lorentzian().plot()
LogNormal().plot()
Moyal().plot()

plt.legend()
