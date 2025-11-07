# Copyright (C) 2025 Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Script to make plots for all the scipy.stats.rv_continuous objects wrapped in
the models module.
"""

import sys

import numpy as np

from aptapy import models
from aptapy.plotting import plt, setup_gca, last_line_color


_EPSILON = sys.float_info.epsilon
DEFAULT_SHAPE_PARAMETERS = (_EPSILON, 1., 2., 4.)


def plot_mean_marker(model):
    """Plot a marker at the position of the mean of the given model.
    """
    x = model.mean()
    y = model(x)
    color = last_line_color()
    plt.plot(x, y, "o", ms=5., color="white")
    plt.plot(x, y, "o", ms=1., color=color)


def plot_rv_shape(model_class, shape_parameters=None, **kwargs):
    """
    """
    kwargs.setdefault("xlabel", "z")
    kwargs.setdefault("ylabel", "g(z)")

    model = model_class()
    if shape_parameters is None and len(model) > 3:
        shape_parameters = DEFAULT_SHAPE_PARAMETERS

    z = np.linspace(*model.plotting_range(), 250)

    plt.figure(model_class.__name__)
    legend_title = f"{model_class.__name__} fit model"
    setup_gca(**kwargs)

    # Case 1: the distribution has no shape parameters.
    if shape_parameters is None:
        plt.plot(z, model(z), label=" ")
        plot_mean_marker(model)
        plt.legend(title=legend_title)
        return

    # Case 2: the distribution has shape parameters.
    param_names = tuple(parameter.name for parameter in model)[3:]
    if len(param_names) == 1:
        param_names = param_names[0]
    for shape in shape_parameters:
        try:
            model.set_parameters(1., 0., 1., *shape)
        except TypeError:
            model.set_parameters(1., 0., 1., shape)
        if isinstance(shape, (float, int)):
            if shape == _EPSILON:
                label = f"{param_names} = 0+"
            else:
                label = f"{param_names} = {shape}"
        else:
            label = ", ".join(f"{name} = {value}" for name, value in zip(param_names, shape))
        plt.plot(z, model(z), label=label)
        plot_mean_marker(model)
    param_names = ", ".join(param_names)
    plt.ylabel(f"g(z; {param_names})")
    plt.legend(title=legend_title)


def create_figures():
    """Create all the figures for the rv_continuous models.
    """
    plot_rv_shape(models.Alpha)
    plot_rv_shape(models.Anglit)
    # plot_rv_shape(models.Arcsine)
    plot_rv_shape(models.Argus)
    plot_rv_shape(models.Beta, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))
    plot_rv_shape(models.BetaPrime, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))


if __name__ == "__main__":
    create_figures()
    plt.show()