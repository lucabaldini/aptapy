"""
Histogram arithmetics
=====================

Explain.
"""

# %%

import numpy as np

from aptapy.hist import Histogram1d
from aptapy.modeling import Exponential, Gaussian
from aptapy.plotting import apply_stylesheet, plt, setup_gca

apply_stylesheet("aptapy-xkcd")

fig = plt.figure()
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1.25], hspace=0.05))
fig.align_ylabels((ax1, ax2))

binning = np.linspace(0., 10., 100)
signal = Histogram1d(binning, label="Signal", xlabel="x [a. u.]")
signal.fill(np.random.default_rng().normal(loc=6., scale=0.5, size=10000))
background = Histogram1d(binning, label="Background")
background.fill(np.random.default_rng().exponential(scale=3., size=100000))
total = signal + background
model = Exponential()
model.fit_histogram(total, xmin=8, xmax=4.)

total.plot(ax1, label="Total")
model.plot(ax1, fit_output=True)
signal.plot(ax1, alpha=0.15, ls='dotted')
background.plot(ax1, alpha=0.15, ls='dotted')
ax1.legend()

sub_signal = total - model
sub_signal.plot(ax2,label="Subtracted signal")

model = Gaussian()
model.fit_histogram(sub_signal)
model.plot(ax2, fit_output=True)
ax2.legend()

setup_gca(xmin=0., xmax=10.)

plt.show()