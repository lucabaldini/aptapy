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

"""Script to make plots for all the sigmoids in the models module.
"""

import pathlib

from aptapy import models
from aptapy.plotting import plt, setup_gca


OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "_static" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_figures():
    """Create all the figures for the rv_continuous models.
    """
    print("Creating sigmoid model shape figures...")
    plt.figure("sigmoid_shapes")
    for model_class in (models.ErfSigmoid,
                        models.LogisticSigmoid,
                        models.Arctangent,
                        models.HyperbolicTangent):
        model = model_class()
        model.set_plotting_range(-5., 5.)
        model.plot()
    setup_gca(xlabel="z", ylabel="g(z)")
    plt.legend()
    file_name = f"sigmoid_shapes.png"
    file_path = OUTPUT_DIR / file_name
    print(f"Saving figure to {file_path}...")
    plt.savefig(file_path, dpi=150)
    plt.close("all")
