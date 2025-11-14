"""
Sigmoid models
==============

Illustration of the available sigmoid models with the standard location and scale
parameters
"""

# %%

from aptapy.plotting import plt
from aptapy.models import ErfSigmoid, LogisticSigmoid, Arctangent, HyperbolicTangent

ErfSigmoid().plot()
LogisticSigmoid().plot()
Arctangent().plot()
HyperbolicTangent().plot()

plt.legend()
