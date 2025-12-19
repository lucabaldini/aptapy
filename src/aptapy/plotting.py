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

"""Plotting facilities.
"""

import pathlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, Generator, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.backend_bases import FigureCanvasBase

__all__ = [
    "AbstractPlottable",
    "VerticalCursor",
    "setup_axes",
    "setup_gca",
    "last_line_color",
    "apply_stylesheet",
    "stylesheet_context",
    "reset",
]


# Define paths to the styles and fonts directories.
_APTAPY_SRC = pathlib.Path(__file__).parent.resolve()
_APTAPY_STYLES = _APTAPY_SRC / "styles"
_APTAPY_FONTS = _APTAPY_SRC / "fonts"

# Register the Humor Sans font in the matplotlib font manager, most notably for
# the xkcd style. Note we ship the font file within the aptapy package as a
# replacement for the default xkcd font, that is not freely distributable---see
# the non-commercial clause in the license at https://github.com/ipython/xkcd-font.
# Note that we have to cast the Path object to str, as addfont() does not
# accept Path objects in older versions of matplotlib (surely not in 3.5.3).
matplotlib.font_manager.fontManager.addfont(str(_APTAPY_FONTS / "Humor-Sans.ttf"))


def _normalize_stylesheet_designator(designator: str) -> str:
    """Normalize a stylesheet designator.

    This would not be needed in recent versions of matplotlib, where a dotted
    package-style_name syntax is supported directly by ``plt.style.use()`` and
    ``plt.style.context()``, but unfortunately this is only available in
    matplotlib 3.7.1 and later. (If you are using a newer matplotlib version,
    by all means you can use all the useful facilities referring to the
    stylesheet provided by the package as ``aptapy.styles.style_name``.) This
    function is purely to support older matplotlib versions, and allow to
    refer to the stylesheets in the ``styles`` folder by name.

    For the same reasons, note that the ``pathlib.Path`` object is cast to a string
    before being returned, as older matplotlib versions do not accept Path
    objects.

    The basic rules are:

    * if the designator starts with ``aptapy``, it is assumed to refer to one
      of the stylesheets shipped with aptapy, and the corresponding path
      is returned;
    * otherwise, the designator is casted to string and returned, assuming it is
      either a valid matplotlib style name, or a path to a custom stylesheet.

    Arguments
    ---------
    designator : str
        The designator of the style to get the path for.
    """
    if designator.startswith("aptapy"):
        file_path = _APTAPY_STYLES / f"{designator}.mplstyle"
        if not file_path.is_file():
            raise ValueError(f"Style '{designator}' not found in aptapy styles.")
        # Note we have to cast to str for compatibility with older matplotlib versions.
        return str(file_path)
    return str(designator)


def apply_stylesheet(style: str = "aptapy") -> None:
    """Apply a given matplotlib stylesheet permanently.

    See https://matplotlib.org/stable/users/explain/customizing.html for more
    information about the basic matplotlib customization.

    Note `plt.style.use <https://matplotlib.org/stable/api/style_api.html#matplotlib.style.use>`_
    accepts dotted names of the form ``package.style_name`` (in that case, ``package``
    should be an importable Python package name; style files in subpackages are
    allowed too.)

    If you want to temporarily apply a given style, consider using the
    ``plt.style.context()`` context manager instead (the rules for the stylesheets
    are exactly the same).

    Arguments
    ---------
    style : str
        The style to use for the plot.
    """
    # If we are using the xkcd style, we also need to enter the native xkcd
    # matplotlib context.
    if style == "aptapy-xkcd":
        plt.xkcd()
    plt.style.use(_normalize_stylesheet_designator(style))


@contextmanager
def stylesheet_context(style: str = "aptapy") -> Generator[None, None, None]:
    """Context manager to temporarily apply a given matplotlib stylesheet.

    Arguments
    ---------
    style : str
        The style to use for the plot.
    """
    # If we are using the xkcd style, we also need to enter the native xkcd
    # matplotlib context.
    if style == "aptapy-xkcd":
        with plt.xkcd(), plt.style.context(_normalize_stylesheet_designator(style)):
            yield
    else:
        with plt.style.context(_normalize_stylesheet_designator(style)):
            yield


def reset(gallery_conf: Dict = None, fname: str = None) -> None:
    """Reset the matplotlib configuration to the default one.

    This is the hook called by sphinx-gallery before running each example
    script, to avoid that configuration changes in one example affect the
    subsequent ones. The callback is configured in ``docs/conf.py``, via the
    ``reset_module`` key of the ``sphinx_gallery_conf``.
    """
    # pylint: disable=unused-argument
    apply_stylesheet()


# Note that we immediately apply the default stylesheet when importing
# this module, so that any plotting operation done afterwards respects
# the aptapy style.
reset()


@dataclass
class AbstractPlottable(ABC):

    """Abstract base class for plottable objects.

    This is a small convenience class that defines a common interface for plottable
    objects, and it is meant to guarantee a consistent interface across different
    plottable objects, such as fitting models, histograms and strip charts.

    This is largely based on the matplotlib plotting interface. A plottable object has
    three basic attributes:

    * a ``label``, that is used to label the data series in the plot legend;
    * an ``xlabel``, that is used to label the x axis of the plot;
    * a ``ylabel``, that is used to label the y axis of the plot.

    The main public interface is the ``plot()`` method, that takes care of plotting
    the object on the given axes (defaulting to the current axes), taking care of
    setting up the labels as needed. What the ``plot()`` method does internally is
    delegated to the ``_render()`` slot, that must be implemented by derived classes.
    """

    label: str = None
    xlabel: str = None
    ylabel: str = None

    def plot(self, axes: matplotlib.axes.Axes = None, **kwargs) -> matplotlib.axes.Axes:
        """Plot the object on the given axes (or on the current axes if none
        is passed as an argument).

        The intended behavior for underlying text labels is that:

        * if the ``label`` attribute is set on the plottable object, this is used
          to create an entry in the legend;
        * if a ``label`` keyword argument is passed to this method, this overrides
          the object attribute;
        * if no ``label`` attribute is set on the object, and no ``label`` keyword
          argument is passed, no entry is created in the legend (i.e., we recover
          the native matplotlib behavior);
        * if the ``xlabel`` and/or ``ylabel`` attributes are set on the object,
          these are used to label the corresponding axes.

        Derived classes can add behavior on top of this (e.g., enrich the label based
        on the current state of the object), but for consistency they should respect
        the intended behavior described above.

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes are used.

        kwargs : keyword arguments
            Additional keyword arguments passed to the _render() method.
            Note that the specifics depends on how _render() is implemented, and
            which type of matplotlib object the plottable is representing.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the object has been plotted on.
        """
        # Set the default value for the label keyword argument, if not already set.
        # Note that if self.label is None, matplotlib will do nothing, as expected.
        kwargs.setdefault("label", self.label)
        if axes is None:
            axes = plt.gca()
        self._render(axes, **kwargs)
        if self.xlabel is not None:
            axes.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            axes.set_ylabel(self.ylabel)
        return axes

    @abstractmethod
    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Render the object on the given axes.

        Arguments
        ---------
        axes : matplotlib.axes.Axes
            The axes to plot on.

        kwargs : keyword arguments
            Additional keyword arguments.
        """


class ConstrainedTextMarker:

    """Small class describing a marker constrained to move along a given path.

    This is essentially the datum of a matplotlib marker and a text label that
    is bound to move on a given trajectory (given as a series of discrete x-y
    coordinates), with the label representing the y value of the curve at a
    given position.

    Arguments
    ---------
    trajectory : Callable[[float], float]
        A callable representing the trajectory of the marker. It must accept a
        single float argument (the x coordinate) and return a single float value
        (the y coordinate).

    axes : matplotlib.axes.Axes, optional
        The axes to draw the marker and associated text on. If None, the current
        axes are used.

    **kwargs : keyword arguments
        Additional keyword arguments passed to the Line2D constructor.
    """

    TEXT_SIZE = "x-small"

    def __init__(self, trajectory: Callable[[float], float], axes: matplotlib.axes.Axes = None,
                 **kwargs) -> None:
        """Constructor.
        """
        if axes is None:
            axes = plt.gca()
        self._trajectory = trajectory
        # Setup the marker...
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("color", "black")
        self._marker = matplotlib.lines.Line2D([None], [None], **kwargs)
        axes.add_line(self._marker)
        # ...and the text label.
        text_kwargs = dict(size=self.TEXT_SIZE, color=kwargs["color"], ha="left", va="bottom")
        self._text = axes.text(None, None, "", **text_kwargs)
        self.set_visible(False)

    def set_visible(self, visible: bool = True) -> None:
        """Set the visibility of the marker and associated text label.

        Arguments
        ---------
        visible : bool
            Flag indicating whether the marker and text label should be visible or not.
        """
        self._marker.set_visible(visible)
        self._text.set_visible(visible)

    def move(self, x: float) -> None:
        """Move the marker to a given x position, with the corresponding y position
        being calculated from the underlying trajectory.

        If the trajectory is not defined at the given x position, this can be signaled
        by raising a ValueError exception from within the trajectory callable. In this
        case, the marker and associated text will be hidden.

        Arguments
        ---------
        x : float
            The x position to move the marker to.
        """
        try:
            y = self._trajectory(x)
        except ValueError:
            self._marker.set_data([None], [None])
            self._text.set_text("")
            return
        self._marker.set_data([x], [y])
        self._text.set_position((x, y))
        self._text.set_text(f"  y = {y:g}")


class MouseButton(IntEnum):

    """Small enum class representing the mouse buttons.

    Interestingly enough, matplotlib does not seem to ship these constants, so
    we have to start all over.
    """

    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    SCROLL_UP = 4
    SCROLL_DOWN = 5


class VerticalCursor:

    """Class representing a zoomable vertical cursor attached to a matplotlib
    Axes object.

    Arguments
    ---------
    axes : matplotlib.axes.Axes, optional
        The axes to draw the cursor on. If None, the current axes are used.

    kwargs : keyword arguments
        Additional keyword arguments passed to axvline().
    """

    TEXT_SIZE = ConstrainedTextMarker.TEXT_SIZE

    def __init__(self, axes: matplotlib.axes.Axes = None, **kwargs) -> None:
        """Constructor.
        """
        self._axes = axes or plt.gca()
        # Setup the vertical line...
        kwargs.setdefault("color", "black")
        kwargs.setdefault("lw", 0.8)
        kwargs.setdefault("ls", "--")
        self._line = self._axes.axvline(**kwargs)
        # ... and the text label.
        text_kwargs = dict(size=self.TEXT_SIZE, color=kwargs["color"], ha="center", va="bottom",
                           transform=self._axes.get_xaxis_transform())
        self._text = self._axes.text(None, None, "", **text_kwargs)
        # Empty placeholders for all the other elements.
        self._markers = []
        self._last_press_position = None
        self._initial_limits = None
        self._zoom_rectangle = patches.Rectangle((0, 0), 0, 0, **kwargs)
        self._axes.add_patch(self._zoom_rectangle)
        self.set_visible(False)

    @property
    def canvas(self) -> FigureCanvasBase:
        """Return the underlying matplotlib canvas.
        """
        return self._axes.figure.canvas

    def redraw_canvas(self) -> None:
        """Trigger a re-draw of the underlying canvas.

        This is factored into separate function, as which function, e.g.,
        draw() or draw_idle(), has important performance implications, and
        this approach allow for a transparent, class-wide switch between one
        hook and the other.
        """
        self.canvas.draw_idle()

    def add_marker(self, trajectory: Callable[[float], float], **kwargs) -> None:
        """Add a new marker to the cursor.

        Note the default color is taken from the last Line2D object that has
        been drawn on the canvas, which makes convenient, e.g., to add a marker
        right after you have plotted a strip chart.

        Arguments
        ---------
        trajectory : Callable[[float], float]
            A callable representing the trajectory of the data set.

        kwargs : keyword arguments
            Additional keyword arguments passed to the ConstrainedTextMarker constructor.
        """
        kwargs.setdefault("color", last_line_color(self._axes))
        self._markers.append(ConstrainedTextMarker(trajectory, self._axes, **kwargs))

    def set_visible(self, visible: bool) -> bool:
        """Set the visibility of the cursor elements.

        Arguments
        ---------
        visible : bool
            Flag indicating whether the cursor elements should be visible or not.

        Returns
        -------
        bool
            True if a redraw is needed, False otherwise.
        """
        need_redraw = self._line.get_visible() != visible
        self._line.set_visible(visible)
        self._text.set_visible(visible)
        for marker in self._markers:
            marker.set_visible(visible)
        return need_redraw

    def move(self, x: float) -> None:
        """Move the cursor to a given x position.

        Arguments
        ---------
        x : float
            The x position to move the cursor to.
        """
        self._line.set_xdata([x])
        self._text.set_position((x, 1.01))
        self._text.set_text(f"x = {x:g}")
        for marker in self._markers:
            marker.move(x)

    def _rectangle_coords(self, event: matplotlib.backend_bases.MouseEvent) -> Tuple:
        """Return the (x0, y0, x1, y1) coordinates of the rectangle defined
        by the ``_last_press_position`` and the current event position.

        The tuple is guaranteed to be in the right order, i.e., x1 >= x0 and
        y1 >= y0, which simplifies the operations downstream.

        Arguments
        ---------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event we want to respond to.
        """
        x0, y0 = self._last_press_position
        x1, y1 = event.xdata, event.ydata
        # Make sure the numbers are in the right order.
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, y0, x1, y1

    def on_button_press(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Function processing the mouse button press events.

        Arguments
        ---------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event we want to respond to.
        """
        # If we are outside the axes, we just don't care.
        if not event.inaxes:
            return
        # If we press the left mouse button we want to cache the initial
        # position of the mouse event, and make the zoom rectangle visible,
        # anchoring it to the position itself.
        # Note we really have to zero the dimensions of ``zoom_rectangle``
        # as we don't now how the last zoom operation left it.
        if event.button == MouseButton.LEFT:
            self._last_press_position = event.xdata, event.ydata
            self._zoom_rectangle.set_visible(True)
            self._zoom_rectangle.set_xy(self._last_press_position)
            self._zoom_rectangle.set_width(0)
            self._zoom_rectangle.set_height(0)
        # If we press the right mouse button, we want to restore the initial
        # axes limits.
        elif event.button == MouseButton.RIGHT:
            xlim, ylim = self._initial_limits
            self._axes.set_xlim(xlim)
            self._axes.set_ylim(ylim)
            self.redraw_canvas()

    def on_button_release(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Function processing the mouse button release events.

        Arguments
        ---------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event we want to respond to.
        """
        if event.button == MouseButton.LEFT:
            # If the event is inside the axes, we want to cache the corners of the
            # rectangle defined by the initial press position and the current event.
            if event.inaxes:
                x0, y0, x1, y1 = self._rectangle_coords(event)
                # And if the rectangle is not degenerate, we want to set the new
                # axes limits accordingly.
                if (x0, y0) != (x1, y1):
                    self._axes.set_xlim(x0, x1)
                    self._axes.set_ylim(y0, y1)
            # In any case we want to hide the zoom rectangle...
            self._zoom_rectangle.set_visible(False)
            # ...set the last press position to None, as this is important for
            # ``on_motion_notify`` events to determine whether we are trying to zoom...
            self._last_press_position = None
            # ...and finally redraw the canvas.
            self.redraw_canvas()

    def on_motion_notify(self, event: matplotlib.backend_bases.MouseEvent) -> None:
        """Function processing the mouse events.

        Arguments
        ---------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event we want to respond to.
        """
        if not event.inaxes:
            if self.set_visible(False):
                self.redraw_canvas()
        else:
            self.move(event.xdata)
            self.set_visible(True)
            if self._last_press_position is not None:
                x0, y0, x1, y1 = self._rectangle_coords(event)
                self._zoom_rectangle.set_xy((x0, y0))
                self._zoom_rectangle.set_width(x1 - x0)
                self._zoom_rectangle.set_height(y1 - y0)
            self.redraw_canvas()

    def activate(self) -> None:
        """Enable the cursor by connecting the mouse motion event to the
        on_mouse_move() method.
        """
        # Cache the canvas limits in order to be able to restore the home
        # configuration after a zoom.
        self._initial_limits = (self._axes.get_xlim(), self._axes.get_ylim())
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion_notify)

    def deactivate(self) -> None:
        """Disable the cursor by disconnecting the mouse motion event.
        """
        self.canvas.mpl_disconnect(self.on_button_press)
        self.canvas.mpl_disconnect(self.on_button_release)
        self.canvas.mpl_disconnect(self.on_motion_notify)


def setup_axes(axes, **kwargs):
    """Setup a generic axes object.
    """
    if kwargs.get("logx"):
        axes.set_xscale("log")
    if kwargs.get("logy"):
        axes.set_yscale("log")
    xticks = kwargs.get("xticks")
    if xticks is not None:
        axes.set_xticks(xticks)
    yticks = kwargs.get("yticks")
    if yticks is not None:
        axes.set_yticks(yticks)
    xlabel = kwargs.get("xlabel")
    if xlabel is not None:
        axes.set_xlabel(xlabel)
    ylabel = kwargs.get("ylabel")
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    xmin, xmax, ymin, ymax = [kwargs.get(key) for key in ("xmin", "xmax", "ymin", "ymax")]
    # Set axis limits individually to avoid passing None to axes.axis()
    if xmin is not None or xmax is not None:
        axes.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        axes.set_ylim(bottom=ymin, top=ymax)
    if kwargs.get("grids"):
        axes.grid(True, which="both")
    if kwargs.get("legend"):
        axes.legend()


def setup_gca(**kwargs):
    """Setup the axes for the current plot.
    """
    setup_axes(plt.gca(), **kwargs)


def last_line_color(axes: matplotlib.axes.Axes = None, default: str = "black") -> str:
    """Return the color used to draw the last line

    Arguments
    ---------
    axes : matplotlib.axes.Axes
        The axes to get the last line color from.

    default : str
        The default color to return if no lines are found.
    """
    if axes is None:
        axes = plt.gca()
    try:
        return axes.get_lines()[-1].get_color()
    except IndexError:
        return default


def subplot_vstack(num_rows: int = 2, sharex: bool = True, height_ratios: List = None,
                   gridspec_kw: Dict = None, **kwargs) -> List:
    """Create a vertical stack of axes in a new figure.

    This is intended to be a small wrapper around the ``subplots()`` methods for
    the case of a single column of vertically stacked axes. The signature of the
    function only exposes the relevant parameters for this specific case.

    Arguments
    ---------
    num_rows : int
        The number of rows (i.e., axes) to create.

    sharex : bool
        Flag indicating whether the x axis should be shared across all axes. Defaults to True.

    height_ratios : List
        A list of height ratios for the different axes. If None, all axes will have the
        same height.

    gridspec_kw : Dict
        Additional keyword arguments passed to the ``GridSpec`` constructor.

    kwargs : keyword arguments
        Additional keyword arguments passed to the ``figure()`` constructor.
    """
    # Create the figure. Note the heuristic in the default figsize: we take the
    # default figure width and scale the default height with the square root of
    # the number of rows, i.e., this 1.41 times the default height for 2 rows,
    # 2 time for 4 rows, and so on.
    width, height = plt.rcParams["figure.figsize"]
    kwargs.setdefault("figsize", (width, height * np.sqrt(num_rows)))
    fig = plt.figure(**kwargs)

    # Create the subplots. If no height ratios are given, all the subplots
    # will have the same height.
    height_ratios = height_ratios or [1.] * num_rows
    if gridspec_kw is None:
        gridspec_kw = {}
    gridspec_kw.setdefault("hspace", 0.05)
    # Create the subplots.
    axes_list = fig.subplots(num_rows, 1, sharex=sharex, gridspec_kw=gridspec_kw,
                             height_ratios=height_ratios)
    # Align all the labels on the y axis for all the subplots.
    fig.align_ylabels(axes_list)
    # Hide the x axis labels for all but the last (bottom-most) axes.
    for ax in axes_list[:-1]:
        ax.xaxis.label.set_visible(False)

    # Here it would have been nice to be able to do ``fig, *axes_list``, but
    # unfortunately this functionality (iterable unpacking in return statements)
    # was only added in Python 3.8, and we are still committed to support
    # Python 3.7.
    return [fig] + list(axes_list)


def residual_axes(sharex: bool = True, height_ratio: float = 0.5,
                  gridspec_kw: Dict = None, **kwargs) -> List:
    """Create a vertical stack of two subplots suitable for a residual plot.

    Arguments
    ---------
    sharex : bool
        Flag indicating whether the x axis should be shared across both axes. Defaults to True.

    height_ratio : float
        The height ratio between the residuals axes and the main axes. Note this is
        a float, and it not the same thing as the height_ratios argument of
        subplot_vstack().

    gridspec_kw : Dict
        Additional keyword arguments passed to the ``GridSpec`` constructor.

    kwargs : keyword arguments
        Additional keyword arguments passed to the ``figure()`` constructor.
    """
    height_ratios = [1., height_ratio]
    return subplot_vstack(2, sharex, height_ratios, gridspec_kw, **kwargs)
