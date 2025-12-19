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

import pathlib

import matplotlib.pyplot as plt
import pytest


def pytest_addoption(parser):
    """Add command line options for pytest to keep the figures interactive.
    """
    parser.addoption("--interactive", action="store_true", default=False,
        help="Keep matplotlib figures open after tests for inspection.")


@pytest.fixture(autouse=True)
def manage_figures(request):
    """Automatically close all matplotlib figures after each test,
    unless explicitly told not to.
    """
    yield
    if not request.config.getoption("--interactive"):
        plt.close("all")


def pytest_sessionfinish(session):
    """At the end of the test session, show any remaining figures if requested.
    """
    if session.config.getoption("--interactive"):
        # Show any open figures *once* at the end of the session
        plt.show()


@pytest.fixture
def datadir():
    return pathlib.Path(__file__).parent / "data"
