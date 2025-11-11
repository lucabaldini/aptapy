"""
Peak models
===========

Illustration of some of the available peak models with the standard location and
scale parameters
"""

# %%

from aptapy.plotting import plt
from aptapy.models import Gaussian, Landau, LogNormal, Lorentzian, Moyal

Gaussian().plot()
Landau().plot()
Lorentzian().plot()
LogNormal().plot()
Moyal().plot()

plt.legend()
