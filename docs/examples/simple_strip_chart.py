"""
Simple strip chart
==================

Simple strip chart
"""

# %%

import numpy as np

from aptapy.plotting import plt
from aptapy.strip import StripChart

x = np.linspace(0., 10., 100)
y = np.random.normal(size=x.shape)

chart = StripChart(label="Random data").put(x, y)
chart.plot()

plt.legend()
