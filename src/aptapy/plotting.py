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

"""Plotting facilities.
"""

from enum import IntEnum
from typing import Any, Callable, Tuple

import matplotlib
import matplotlib.pyplot as plt  # noqa: F401 pylint: disable=unused-import
from cycler import cycler
from loguru import logger
from matplotlib import patches
from matplotlib.backend_bases import FigureCanvasBase

__all__ = [
    "VerticalCursor",
    "setup_axes",
    "setup_gca",
    "last_line_color",
    "configure",
]


DEFAULT_FIGURE_WIDTH = 8.
DEFAULT_FIGURE_HEIGHT = 6.
DEFAULT_FIGURE_SIZE = (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT)
DEFAULT_COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


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
        text_kwargs = dict(size=self.TEXT_SIZE, color=kwargs["color"], ha="left", va="center")
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
        being calculated from the underlying spline.

        Arguments
        ---------
        x : float
            The x position to move the marker to.
        """
        y = self._trajectory(x)
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
        kwargs.setdefault("color", last_line_color())
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
            x0, y0, x1, y1 = self._rectangle_coords(event)
            # Set the last press position to None, as this is important for
            # ``motion_notify`` events to determine whether we are trying to
            # zoom or not. Note it is important to do this immediately, as
            # if we are just clicking without moving the mouse we would be
            # implicitly defining a null rectangle that we cannot zoom upon,
            # and in this case the function returns at the next line.
            self._last_press_position = None
            # If the rectangle is null, return.
            if (x0, y0) == (x1, y1):
                return
            self._axes.set_xlim(x0, x1)
            self._axes.set_ylim(y0, y1)
            self._zoom_rectangle.set_visible(False)
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


def last_line_color(default: str = "black") -> str:
    """Return the color used to draw the last line
    """
    try:
        return plt.gca().get_lines()[-1].get_color()
    except IndexError:
        return default


def _set(key: str, value: Any):
    """Set the value for a single matplotlib parameter.

    The actual command is encapsulated into a try except block because this
    is intended to work across different matplotlib versions. If a setting
    cannot be applied for whatever reason, this will happily move on.
    """
    try:
        matplotlib.rcParams[key] = value
    except KeyError:
        logger.warning(f"Unknown matplotlib rc param {key}, skipping...")
    except ValueError as exception:
        logger.warning(f"{exception}, skipping...")


def configure(*args) -> None:
    """See https://matplotlib.org/stable/users/explain/customizing.html for more
    information.

    .. note::

       Note that this function can be used as a hook by Sphinx Gallery to
       configure the plotting environment for each example, so that the matplotlib
       configuration is consistent across all examples and is not reset each time.
       This is the reason why the function signature includes unused arguments.
    """
    # pylint:disable=too-many-statements, unused-argument

    # Backends
    _set("interactive", False)
    _set("timezone", "UTC")

    # Lines
    # See https://matplotlib.org/stable/api/artist_api.html#module-matplotlib.lines
    _set("lines.linewidth", 1.5)  # line width in points
    _set("lines.linestyle", "-")  # solid line
    _set("lines.color", "C0")  # has no affect on plot(); see axes.prop_cycle
    _set("lines.marker", "None")  # the default marker
    _set("lines.markerfacecolor", "auto")  # the default marker face color
    _set("lines.markeredgecolor", "auto")  # the default marker edge color
    _set("lines.markeredgewidth", 1.0)  # the line width around the marker symbol
    _set("lines.markersize", 6)  # marker size, in points
    _set("lines.dash_joinstyle", "round")  # {miter, round, bevel}
    _set("lines.dash_capstyle", "butt")  # {butt, round, projecting}
    _set("lines.solid_joinstyle", "round")  # {miter, round, bevel}
    _set("lines.solid_capstyle", "projecting")  # {butt, round, projecting}
    _set("lines.antialiased", True)  # render lines in antialiased (no jaggies)
    # The three standard dash patterns. These are scaled by the linewidth.
    _set("lines.dashed_pattern", (3.7, 1.6))
    _set("lines.dashdot_pattern", (6.4, 1.6, 1, 1.6))
    _set("lines.dotted_pattern", (1, 1.65))
    _set("lines.scale_dashes", True)
    _set("markers.fillstyle", "full")  # {full, left, right, bottom, top, none}
    _set("pcolor.shading", "auto")
    # Whether to snap the mesh to pixel boundaries. This is provided solely to allow
    # old test images to remain unchanged. Set to False to obtain the previous behavior.
    _set("pcolormesh.snap", True)

    # Patches are graphical objects that fill 2D space, like polygons or circles.
    # See https://matplotlib.org/stable/api/artist_api.html#module-matplotlib.patches
    _set("patch.linewidth", 1.0)  # edge width in points.
    _set("patch.facecolor", "C0")
    # By default, Patches and Collections do not draw edges. This value is only used
    # if facecolor is "none" (an Artist without facecolor and edgecolor would be
    # invisible) or if patch.force_edgecolor is True.
    _set("patch.edgecolor", "black")
    # By default, Patches and Collections do not draw edges. Set this to True to draw
    # edges with patch.edgedcolor as the default edgecolor. This is mainly relevant
    # for styles.
    _set("patch.force_edgecolor", True)
    _set("patch.antialiased", True)  # render patches in antialiased (no jaggies)

    # Hatches
    _set("hatch.color", "black")
    _set("hatch.linewidth", 1.0)

    # Boxplot---we don"t really use these much, but you never know...
    _set("boxplot.notch", False)
    _set("boxplot.vertical", True)
    _set("boxplot.whiskers", 1.5)
    _set("boxplot.bootstrap", None)
    _set("boxplot.patchartist", False)
    _set("boxplot.showmeans", False)
    _set("boxplot.showcaps", True)
    _set("boxplot.showbox", True)
    _set("boxplot.showfliers", True)
    _set("boxplot.meanline", False)
    _set("boxplot.flierprops.color", "black")
    _set("boxplot.flierprops.marker", "o")
    _set("boxplot.flierprops.markerfacecolor", "none")
    _set("boxplot.flierprops.markeredgecolor", "black")
    _set("boxplot.flierprops.markeredgewidth", 1.0)
    _set("boxplot.flierprops.markersize", 6)
    _set("boxplot.flierprops.linestyle", "none")
    _set("boxplot.flierprops.linewidth", 1.0)
    _set("boxplot.boxprops.color", "black")
    _set("boxplot.boxprops.linewidth", 1.0)
    _set("boxplot.boxprops.linestyle", "-")
    _set("boxplot.whiskerprops.color", "black")
    _set("boxplot.whiskerprops.linewidth", 1.0)
    _set("boxplot.whiskerprops.linestyle", "-")
    _set("boxplot.capprops.color", "black")
    _set("boxplot.capprops.linewidth", 1.0)
    _set("boxplot.capprops.linestyle", "-")
    _set("boxplot.medianprops.color", "C1")
    _set("boxplot.medianprops.linewidth", 1.0)
    _set("boxplot.medianprops.linestyle", "-")
    _set("boxplot.meanprops.color", "C2")
    _set("boxplot.meanprops.marker", "^")
    _set("boxplot.meanprops.markerfacecolor", "C2")
    _set("boxplot.meanprops.markeredgecolor", "C2")
    _set("boxplot.meanprops.markersize", 6)
    _set("boxplot.meanprops.linestyle", "--")
    _set("boxplot.meanprops.linewidth", 1.0)

    # The font properties used by `text.Text`.
    # See https://matplotlib.org/stable/api/font_manager_api.html
    # Note that for font.serif, font.sans-serif, and font.monospace, the first element
    # of the list (a DejaVu font) will always be used because DejaVu is shipped with
    # Matplotlib and is thus guaranteed to be available.
    _set("font.family", "sans-serif")
    _set("font.style", "normal")  # {normal (or roman), italic,  oblique}
    _set("font.variant", "normal")  # {normal, small-caps}
    # The font.weight property has effectively 13 values: normal, bold, bolder, lighter,
    # 100, 200, 300, ..., 900. Normal is the same as 400, and bold is 700. bolder and
    # lighter are relative values with respect to the current weight.
    _set("font.weight", "normal")
    _set("font.stretch", "normal")  # Currently not implemented.
    # The font.size property is the default font size for text, given in points. 10 pt
    # is the standard value. Special text sizes can be defined relative to font.size,
    # using the following values: xx-small, x-small, small, medium, large, x-large,
    # xx-large, larger, or smaller
    _set("font.size", 14.0)

    ## The text properties used by `text.Text`.
    ## See https://matplotlib.org/stable/api/artist_api.html#module-matplotlib.text
    _set("text.color", "black")
    # FreeType hinting flag {default, no_autohint, force_autohint, no_hinting}
    _set("text.hinting", "force_autohint")
    # Specifies the amount of softness for hinting in the horizontal direction.
    # A value of 1 will hint to full pixels. A value of 2 will hint to half pixels etc.
    _set("text.hinting_factor", 8)
    # Specifies the scaling factor for kerning values. This is provided solely to
    # allow old test images to remain unchanged. Set to 6 to obtain previous behavior.
    # Values  other than 0 or 6 have no defined meaning.
    _set("text.kerning_factor", 0)
    _set("text.antialiased", True)  # This only affects raster outputs.
    # Use mathtext if there is an even number of unescaped dollar signs.
    _set("text.parse_math", True)

    # LaTeX
    # See https://matplotlib.org/stable/users/explain/text/usetex.html
    _set("text.usetex", False)  # use latex for all text handling.
    # Font set can be {dejavusans, dejavuserif, cm, stixsans, custom}, where
    # "custom" is defined by the mathtext.bf, .cal, .it, ..., settings which map
    # a TeX font name to a fontconfig font pattern.
    _set("mathtext.fontset", "dejavusans")
    _set("mathtext.fallback", "cm")
    _set("mathtext.default", "it")

    # Axes
    # See https://matplotlib.org/stable/api/axes_api.html#module-matplotlib.axes
    _set("axes.facecolor", "white")  # axes background color
    _set("axes.edgecolor", "black")  # axes edge color
    _set("axes.linewidth", 1.2)  # edge line width
    _set("axes.grid", True)  # display grid or not
    _set("axes.grid.axis", "both")  # which axis the grid should apply to
    _set("axes.grid.which", "major")  # grid lines at {major, minor, both} ticks
    _set("axes.titlelocation", "center")  # alignment of the title: {left, right, center}
    _set("axes.titlesize", "large")  # font size of the axes title
    _set("axes.titleweight", "normal")  # font weight of title
    # Color of the axes title, auto falls back tO text.color as default value
    _set("axes.titlecolor", "auto")
    _set("axes.titley", None)  # position title (axes relative units).  None implies auto
    _set("axes.titlepad", 6.0)  # pad between axes and title in points
    _set("axes.labelsize", "medium")  # font size of the x and y labels
    _set("axes.labelpad", 4.0)  # space between label and axis
    _set("axes.labelweight", "normal")  # weight of the x and y labels
    _set("axes.labelcolor", "black")
    # Draw axis gridlines and ticks:
    # - below patches (True)
    # - above patches but below lines ("line")
    # - above all (False)
    _set("axes.axisbelow", "line")
    # Use scientific notation if log10 of the axis range is smaller than the
    # first or larger than the second
    _set("axes.formatter.limits", (-5, 6))
    _set("axes.formatter.use_locale", False)
     # When True, use mathtext for scientific notation.
    _set("axes.formatter.use_mathtext", False)
    # Minimum exponent to format in scientific notation
    _set("axes.formatter.min_exponent", 0)
    # If True, the tick label formatter will default to labeling ticks relative
    # to an offset when the data range is small compared to the minimum absolute
    # value of the data.
    _set("axes.formatter.useoffset", True)
    # When useoffset is True, the offset will be used when it can remove
    # at least this number of significant digits from tick labels.
    _set("axes.formatter.offset_threshold", 4)
    _set("axes.spines.left", True)  # display axis spines
    _set("axes.spines.bottom", True)
    _set("axes.spines.top", True)
    _set("axes.spines.right", True)
    # use Unicode for the minus symbol rather than hyphen.  See
    # https://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes
    _set("axes.unicode_minus", True)
    _set("axes.prop_cycle", cycler("color", DEFAULT_COLOR_CYCLE))
    _set("axes.xmargin", .05)  # x margin.  See `axes.Axes.margins`
    _set("axes.ymargin", .05)  # y margin.  See `axes.Axes.margins`
    _set("axes.zmargin", .05)  # z margin.  See `axes.Axes.margins`
    # If "data", use axes.xmargin and axes.ymargin as is.
    # If "round_numbers", after application of margins, axis limits are further expanded
    # to the nearest "round" number.
    _set("axes.autolimit_mode", "data")
    _set("polaraxes.grid", True)   # display grid on polar axes
    _set("axes3d.grid", True)   # display grid on 3D axes
    # Automatically add margin when manually setting 3D axis limits
    _set("axes3d.automargin", False)
    _set("axes3d.xaxis.panecolor", (0.95, 0.95, 0.95, 0.5))  # background pane on 3D axes
    _set("axes3d.yaxis.panecolor", (0.90, 0.90, 0.90, 0.5))  # background pane on 3D axes
    _set("axes3d.zaxis.panecolor", (0.925, 0.925, 0.925, 0.5))  # background pane on 3D axes
    _set("axes3d.mouserotationstyle", "arcball")  # {azel, trackball, sphere, arcball}
    _set("axes3d.trackballsize", 0.667)  # trackball diameter, in units of the Axes bbox
    # trackball border width, in units of the Axes bbox (only for "sphere" and "arcball" style)
    _set("axes3d.trackballborder", 0.2)

    # Axis
    _set("xaxis.labellocation", "center")  # {left, right, center}
    _set("yaxis.labellocation", "center")  # {bottom, top, center}

    # Dates
    # These control the default format strings used in AutoDateFormatter.
    _set("date.autoformatter.year", "%Y")
    _set("date.autoformatter.month", "%Y-%m")
    _set("date.autoformatter.day", "%Y-%m-%d")
    _set("date.autoformatter.hour", "%m-%d %H")
    _set("date.autoformatter.minute", "%d %H:%M")
    _set("date.autoformatter.second", "%H:%M:%S")
    _set("date.autoformatter.microsecond", "%M:%S.%f")
    # The reference date for Matplotlib"s internal date representation
    # See https://matplotlib.org/stable/gallery/ticks/date_precision_and_epochs.html
    _set("date.epoch", "1970-01-01T00:00:00")
    _set("date.converter", "auto")  # {auto, concise}
    # For auto converter whether to use interval_multiples:
    _set("date.interval_multiples", True)

    # Ticks
    # See https://matplotlib.org/stable/api/axis_api.html#matplotlib.axis.Tick
    _set("xtick.top", False)  # draw ticks on the top side
    _set("xtick.bottom", True)  # draw ticks on the bottom side
    _set("xtick.labeltop", False)  # draw label on the top
    _set("xtick.labelbottom", True)  # draw label on the bottom
    _set("xtick.major.size", 3.5)  # major tick size in points
    _set("xtick.minor.size", 2)  # minor tick size in points
    _set("xtick.major.width", 0.8)  # major tick width in points
    _set("xtick.minor.width", 0.6)  # minor tick width in points
    _set("xtick.major.pad", 3.5)  # distance to major tick label in points
    _set("xtick.minor.pad", 3.4)  # distance to the minor tick label in points
    _set("xtick.color", "black")  # color of the ticks
    # Color of the tick labels or inherit from xtick.color
    _set("xtick.labelcolor", "inherit")
    _set("xtick.labelsize", "medium")  # font size of the tick labels
    _set("xtick.direction", "out")  # direction: {in, out, inout}
    _set("xtick.minor.visible", False)  # visibility of minor ticks on x-axis
    _set("xtick.major.top", True)  # draw x axis top major ticks
    _set("xtick.major.bottom", True)  # draw x axis bottom major ticks
    _set("xtick.minor.top", True)  # draw x axis top minor ticks
    _set("xtick.minor.bottom", True)  # draw x axis bottom minor ticks
    _set("xtick.minor.ndivs", "auto")  # number of minor ticks between the major ticks on x-axis
    _set("xtick.alignment", "center")  # alignment of xticks
    _set("ytick.left", True)  # draw ticks on the left side
    _set("ytick.right", False)  # draw ticks on the right side
    _set("ytick.labelleft", True)  # draw tick labels on the left side
    _set("ytick.labelright", False)  # draw tick labels on the right side
    _set("ytick.major.size", 3.5)  # major tick size in points
    _set("ytick.minor.size", 2)  # minor tick size in points
    _set("ytick.major.width", 0.8)  # major tick width in points
    _set("ytick.minor.width", 0.6)  # minor tick width in points
    _set("ytick.major.pad", 3.5)  # distance to major tick label in points
    _set("ytick.minor.pad", 3.4)  # distance to the minor tick label in points
    _set("ytick.color", "black")  # color of the ticks
    _set("ytick.labelcolor", "inherit")  # color of the tick labels or inherit from ytick.color
    _set("ytick.labelsize", "medium")  # font size of the tick labels
    _set("ytick.direction", "out")  # direction: {in, out, inout}
    _set("ytick.minor.visible", False)  # visibility of minor ticks on y-axis
    _set("ytick.major.left", True)  # draw y axis left major ticks
    _set("ytick.major.right", True)  # draw y axis right major ticks
    _set("ytick.minor.left", True)  # draw y axis left minor ticks
    _set("ytick.minor.right", True)  # draw y axis right minor ticks
    _set("ytick.minor.ndivs", "auto")  # number of minor ticks between the major ticks on y-axis
    _set("ytick.alignment", "center_baseline")  # alignment of yticks

    # Grids
    _set("grid.color", "#c0c0c0")  # grid color
    _set("grid.linestyle", "--")  # line style
    _set("grid.linewidth", 0.8)  # in points
    _set("grid.alpha", 0.8)  # transparency, between 0.0 and 1.0


    # Legends
    _set("legend.loc", "best")
    _set("legend.frameon", True)  # if True, draw the legend on a background patch
    _set("legend.framealpha", 0.75)  # legend patch transparency
    _set("legend.facecolor", "inherit")  # inherit from axes.facecolor; or color spec
    _set("legend.edgecolor", "#a0a0a0")  # background patch boundary color
    # If True, use a rounded box for the legend background, else a rectangle
    _set("legend.fancybox", True)
    _set("legend.shadow", False)  # if True, give background a shadow effect
    _set("legend.numpoints", 1)  # the number of marker points in the legend line
    _set("legend.scatterpoints", 1)  # number of scatter points
    _set("legend.markerscale", 1.0)  # the relative size of legend markers vs. original
    _set("legend.fontsize", "small")
    _set("legend.labelcolor", None)
    _set("legend.title_fontsize", None)  # None sets to the same as the default axes.
    # Dimensions as fraction of font size:
    _set("legend.borderpad", 0.4)  # border whitespace
    _set("legend.labelspacing", 0.5)  # the vertical space between the legend entries
    _set("legend.handlelength", 2.0)  # the length of the legend lines
    _set("legend.handleheight", 0.7)  # the height of the legend handle
    _set("legend.handletextpad", 0.8)  # the space between the legend line and legend text
    _set("legend.borderaxespad", 0.5)  # the border between the axes and legend edge
    _set("legend.columnspacing", 2.0)  # column separation

    # Figures
    ## See https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure
    _set("figure.titlesize", "large")  # size of the figure title
    _set("figure.titleweight", "normal")  # weight of the figure title
    _set("figure.labelsize", "large")  # size of the figure label
    _set("figure.labelweight", "normal")  # weight of the figure label
    _set("figure.figsize", DEFAULT_FIGURE_SIZE)  # figure size in inches
    _set("figure.dpi", 100)  # figure dots per inch
    _set("figure.facecolor", "white")  # figure face color
    _set("figure.edgecolor", "white")  # figure edge color
    _set("figure.frameon", True)  # enable figure frame
    _set("figure.max_open_warning", 20)
    _set("figure.raise_window", True)  # Raise the GUI window to front when show() is called
    # The figure subplot parameters.
    # All dimensions are a fraction of the figure width and height.
    _set("figure.subplot.left", 0.125)  # the left side of the subplots of the figure
    _set("figure.subplot.right", 0.97)  # the right side of the subplots of the figure
    _set("figure.subplot.bottom", 0.11)  # the bottom of the subplots of the figure
    _set("figure.subplot.top", 0.96)  # the top of the subplots of the figure
    # Amount of width reserved for space between subplots, expressed as a fraction
    # of the average axis width.
    _set("figure.subplot.wspace", 0.2)
    # Amount of height reserved for space between subplots, expressed as a fraction
    # of the average axis height
    _set("figure.subplot.hspace", 0.2)
    # When True, automatically adjust subplot parameters to make the plot fit the figure
    # using `tight_layout`
    _set("figure.autolayout", False)
     # When True, automatically make plot elements fit on the figure.
     # (Not compatible with `autolayout`, above).
    _set("figure.constrained_layout.use", False)
    # Padding (in inches) around axes; defaults to 3/72 inches, i.e. 3 points.
    _set("figure.constrained_layout.h_pad", 0.04167)
    _set("figure.constrained_layout.w_pad", 0.04167)
    # Spacing between subplots, relative to the subplot sizes.  Much smaller than for
    # tight_layout (figure.subplot.hspace, figure.subplot.wspace) as constrained_layout
    # already takes surrounding texts (titles, labels, # ticklabels) into account.
    _set("figure.constrained_layout.hspace", 0.02)
    _set("figure.constrained_layout.wspace", 0.02)

    # Images
    _set("image.aspect", "equal")   # {equal, auto} or a number
    _set("image.interpolation", "auto")  # see help(imshow) for options
    _set("image.interpolation_stage", "auto")  # see help(imshow) for options
    _set("image.cmap", "viridis")  # A colormap name (plasma, magma, etc.)
    _set("image.lut", 256)  # the size of the colormap lookup table
    _set("image.origin", "upper")  # {lower, upper}
    _set("image.resample", True)
    # When True, all the images on a set of axes are combined into a single composite
    # image before saving a figure as a vector graphics file, such as a PDF.
    _set("image.composite_image", True)

    # Various plots.
    _set("contour.negative_linestyle", "dashed")  # string or on-off ink sequence
    _set("contour.corner_mask", True)  # {True, False}
    _set("contour.linewidth", None)
    _set("contour.algorithm", "mpl2014")  # {mpl2005, mpl2014, serial, threaded}
    _set("errorbar.capsize", 0)  # length of end cap on error bars in pixels
    _set("hist.bins", 10)  # The default number of histogram bins or "auto".
    _set("scatter.marker", "o")  # The default marker type for scatter plots.
    _set("scatter.edgecolors", "face")  # The default edge colors for scatter plots.

    # Paths
    # When True, simplify paths by removing "invisible" points to reduce file size
    # and increase rendering speed
    _set("path.simplify", True)
    # The threshold of similarity below which vertices will be removed in
    # the simplification process.
    _set("path.simplify_threshold", 0.111111111111)
    # When True, rectilinear axis-aligned paths will be snapped to the nearest pixel
    # when certain criteria are met. When False, paths will never be snapped.
    _set("path.snap", True)
    # May be None, or a tuple of the form: path.sketch: (scale, length, randomness)
    # - scale is the amplitude of the wiggle perpendicular to the line (in pixels).
    # - length is the length of the wiggle along the line (in pixels).
    # - randomness is the factor by which the length is randomly scaled.
    _set("path.sketch", None)

    # Saving figures...
    # The default savefig parameters can be different from the display parameters
    _set("savefig.dpi", 300)  # figure dots per inch or "figure"
    _set("savefig.facecolor", "auto")  # figure face color when saving
    _set("savefig.edgecolor", "auto")  # figure edge color when saving
    _set("savefig.format", "png")  # {png, ps, pdf, svg}
    _set("savefig.bbox", "standard")  # {tight, standard}
    _set("savefig.pad_inches", 0.1)  # padding to be used, when bbox is set to "tight"
    # Default directory in savefig dialog, gets updated after interactive saves,
    # unless set to the empty string (i.e. the current directory); use "." to start
    # at the current directory but update after interactive saves
    _set("savefig.directory", "")
    # Whether figures are saved with a transparent background by default
    _set("savefig.transparent", False)
    # Orientation of saved figure, for PostScript output only
    _set("savefig.orientation", "portrait")
    _set("macosx.window_mode", "system")
    _set("tk.window_focus", False)  # Maintain shell focus for TkAgg
    # Integer from 0 to 9, 0 disables compression (good for debugging)
    _set("pdf.compression", 6)
    _set("pdf.fonttype", 3)  # Output Type 3 (Type3) or Type 42 (TrueType)
    _set("pdf.use14corefonts", False)
    _set("pdf.inheritcolor", False)
    _set("svg.image_inline", True)  # Write raster image data directly into the SVG file
    # How to handle SVG fonts:
    # - path: embed characters as paths -- supported by most SVG renderers
    # - None: assume fonts are installed on the machine where the SVG will be viewed.
    _set("svg.fonttype", "path")
    _set("svg.hashsalt", None)  # If not None, use this string as hash salt instead of uuid4
    # If not None, use this string as the value for the `id` attribute in the top <svg> tag
    _set("svg.id", None)
    # See https://matplotlib.org/stable/tutorials/text/pgf.html for more information.
    _set("pgf.rcfonts", True)
    _set("pgf.texsystem", "xelatex")
    _set("docstring.hardcopy", False)  # set this when you want to generate hardcopy docstring

    # Animations
    # How to display the animation as HTML in the IPython notebook:
    # - "html5" uses HTML5 video tag
    # - "jshtml" creates a JavaScript animation
    _set("animation.html", "none")
    _set("animation.writer", "ffmpeg")  # MovieWriter "backend" to use
    _set("animation.codec", "h264")  # Codec to use for writing movie
    # Controls size/quality trade-off for movie.
    # -1 implies let utility auto-determine
    _set("animation.bitrate", -1)
    _set("animation.frame_format", "png")  # Controls frame format used by temp files
    # Path to ffmpeg binary. Unqualified paths are resolved by subprocess.Popen.
    _set("animation.ffmpeg_path", "ffmpeg")
    # Path to ImageMagick"s convert binary. Unqualified paths are resolved by
    # subprocess.Popen, except that on Windows, we look up an install of
    # ImageMagick in the registry (as convert is also the name of a system tool).
    _set("animation.convert_path", "convert")
    # Limit, in MB, of size of base64 encoded animation in HTML (i.e. IPython notebook)
    _set("animation.embed_limit", 20.0)


configure()
