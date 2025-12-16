"""
3-D Histogram
=====================


"""

# %%

# We could change this example to show collapsing axis with a 2d histogram
import numpy as np

from aptapy.hist import Histogram3d
from aptapy.plotting import plt

# 
x, y = np.random.default_rng().random((2, 100000))
z = 1 - x / 2 - y / 2

# Create and fill the 3D histogram
edges = np.linspace(0., 1., 100)
hist3d = Histogram3d(edges, edges, edges, label="Random data", xlabel="x", ylabel="y", zlabel="z")
hist3d.fill(x, y, z)

hist_mean, hist_rms = hist3d.collapse_axis(2)
plt.figure("3D Histogram - Mean")
hist_mean.plot()

plt.figure()
slice = hist3d.extract_column(10,  10)
slice.plot(label="x=0 slice")
plt.gca().set_aspect("equal")
plt.show()