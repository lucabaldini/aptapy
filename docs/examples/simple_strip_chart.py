"""
Strip chart
===========

Simple strip chart with Unix time.
"""

# %%

import time

import numpy as np

from aptapy.plotting import plt
from aptapy.strip import StripChart

t0 = time.time()
t = np.linspace(t0, t0 + 3600., 100)
y = np.random.normal(size=t.shape)

chart = StripChart(label="Random data", datetime=True)
chart.extend(t, y)
chart.plot()

plt.legend()