# Copyright 2023--2025 the aptapy team
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

import pathlib
import sys

from aptapy import models
from aptapy.plotting import plt, setup_gca


_EPSILON = sys.float_info.epsilon
DEFAULT_SHAPE_PARAMETERS = (_EPSILON, 1., 2., 4.)
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "_static" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_rv_shape(model_class, shape_parameters=None, location=0., scale=1., **kwargs):
    """
    """
    kwargs.setdefault("xlabel", "z")
    kwargs.setdefault("ylabel", "g(z)")

    model = model_class()

    if shape_parameters is None and len(model) > 3:
        shape_parameters = DEFAULT_SHAPE_PARAMETERS

    plt.figure(model_class.__name__)
    legend_title = f"{model_class.__name__} fit model"
    setup_gca(**kwargs)

    file_name = f"{model_class.__name__.lower()}_shape.png"
    file_path = OUTPUT_DIR / file_name

    # Case 1: the distribution has no shape parameters.
    if shape_parameters is None:
        model.set_parameters(1., location, scale)
        model.plot(label="Model")
        plt.legend(title=legend_title)
        print(f"Saving figure to {file_path}...")
        plt.savefig(file_path, dpi=150)
        plt.close()
        return

    # Case 2: the distribution has shape parameters.
    param_names = tuple(parameter.name for parameter in model)[3:]
    if len(param_names) == 1:
        param_names = param_names[0]
    for shape in shape_parameters:
        try:
            model.set_parameters(1., location, scale, *shape)
        except TypeError:
            model.set_parameters(1., location, scale, shape)
        if isinstance(shape, (float, int)):
            if shape == _EPSILON:
                label = f"{param_names} = 0+"
            else:
                label = f"{param_names} = {shape}"
        else:
            label = ", ".join(f"{name} = {value}" for name, value in zip(param_names, shape))

        model.plot(label=label)
    param_names = ", ".join(param_names)
    plt.ylabel(f"g(z; {param_names})")
    plt.legend(title=legend_title)
    print(f"Saving figure to {file_path}...")
    plt.savefig(file_path, dpi=150)
    plt.close()


def create_figures():
    """Create all the figures for the rv_continuous models.
    """
    print("Creating rv_continuous model shape figures...")
    plot_rv_shape(models.Alpha)
    plot_rv_shape(models.Anglit)
    plot_rv_shape(models.Arcsine)
    plot_rv_shape(models.Argus)
    plot_rv_shape(models.Beta, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))
    plot_rv_shape(models.BetaPrime, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))
    plot_rv_shape(models.Bradford)
    plot_rv_shape(models.Burr, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))
    plot_rv_shape(models.Burr12, ((1., 1.), (1., 4.), (4., 1.), (2., 4.), (4., 2.), (4., 4.)))
    plot_rv_shape(models.Cauchy)
    plot_rv_shape(models.Chi, (1., 2., 4., 10.))
    plot_rv_shape(models.Chisquare, (1., 2., 4., 8.))
    plot_rv_shape(models.Cosine)
    plot_rv_shape(models.CrystalBall, ((1., 2.), (2., 2.), (3., 3.)))
    plot_rv_shape(models.Gaussian)
    plot_rv_shape(models.Gibrat)
    plot_rv_shape(models.GumbelL)
    plot_rv_shape(models.GumbelR)
    plot_rv_shape(models.HalfCauchy)
    plot_rv_shape(models.HalfLogistic)
    plot_rv_shape(models.HalfNorm)
    plot_rv_shape(models.HyperSecant)
    plot_rv_shape(models.Landau)
    plot_rv_shape(models.Laplace)
    plot_rv_shape(models.Levy)
    plot_rv_shape(models.LevyL)
    plot_rv_shape(models.Logistic)
    plot_rv_shape(models.LogNormal, (0.5, 1., 2.))
    plot_rv_shape(models.Maxwell)
    plot_rv_shape(models.Moyal)
    plot_rv_shape(models.Nakagami, (1., 2., 4.))
    plot_rv_shape(models.Rayleigh)
    plot_rv_shape(models.Semicircular)
    plot_rv_shape(models.Student, (1., 2., 4., 10.))
    plot_rv_shape(models.Wald)
    plt.close("all")
