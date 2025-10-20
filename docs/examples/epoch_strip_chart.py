"""
Epoch strip chart
==================

Strip chart where the values on the x-axis are seconds since the Unix epoch.
"""

# %%


import numpy as np

from aptapy.plotting import plt
from aptapy.strip import EpochStripChart

t0 = 1760717637
t = np.linspace(t0, t0 + 3600., 100)
y = np.random.normal(size=t.shape)

chart = EpochStripChart(label="Random data").put(t, y)
chart.plot()

plt.legend()