"""
Sigmoid models
==============

Illustration of the available sigmoid models with the standard location and scale
parameters
"""

# %%

from aptapy.plotting import plt
from aptapy.models import Erf, Logistic, Arctangent, HyperbolicTangent

Erf().plot()
Logistic().plot()
Arctangent().plot()
HyperbolicTangent().plot()

plt.legend()
