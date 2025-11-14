.. _release_notes:

Release notes
=============


Version 0.14.0 (2025-11-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~


* Major refactoring of the modeling infrastructure, introducing a more structured and
  extensible framework for fitting models.
* New abstract base classes (AbstractSigmoidFitModel, AbstractCRVFitModel) to support
  different model families
* Added 40+ new model classes wrapping scipy.stats continuous random variables
* Refactored existing models to use a consistent primitive() method instead of integral()
* Reorganized tests into category-specific files (simple, sigmoid, scipy models)
* Updated parameter naming.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/49
  - https://github.com/lucabaldini/aptapy/issues/46
  - https://github.com/lucabaldini/aptapy/issues/43


Version 0.13.0 (2025-11-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Major refactoring of the modeling module (now split into the `models` and `modeling`
  modules, the former containing all the core functionality and latter the actual
  models) to improve code organization and maintainability.
* Erf and ErfInverse models have been renamed.
* New models from the exponential family, including the StretchedExponential and
  its complement, and all exponential models now supporting a generic origin.
* Added facility to draw random samples from fit models, in order to facilitate
  model validation and testing.
* Base class AbstractPlottable now correctly propagated to all fit models.
* Detailed mathematical formulae added to the docstrings of all model classes.
* Fix for issue #33 (cannot freeze parameters for composite models).
* Components within composite models now accessible via indexing (e.g.,
  ``composite_model[0]`` returns the first component).
* Added a unit test for composite model parameter freezing.
* New ``plot_components`` argument in the ``plot()`` method of fit models to
  control individual components of composite models.
* Arguments names for the ``integral()`` methods of fit models updated from
  ``xmin``, ``xmax`` to ``x1``, ``x2`` for consistency.
* Enhanced unit test coverage for the modeling module.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/42
  - https://github.com/lucabaldini/aptapy/pull/38
  - https://github.com/lucabaldini/aptapy/pull/37
  - https://github.com/lucabaldini/aptapy/issues/41
  - https://github.com/lucabaldini/aptapy/issues/39
  - https://github.com/lucabaldini/aptapy/issues/36
  - https://github.com/lucabaldini/aptapy/issues/35
  - https://github.com/lucabaldini/aptapy/issues/34
  - https://github.com/lucabaldini/aptapy/issues/33


Version 0.12.0 (2025-10-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added support for subtracting callable models from ``Histogram1d`` objects
  (this is useful, e.g., to create a residual histogram wrt a fit model).
* Introduced ``subplot_vstack()`` and ``residual_axes()`` functions for
  creating multi-panel plots.
* Enhanced the ``fit()`` method to support excluding intervals when xmin > xmax.
* Improved histogram copying functionality to allow optional relabeling.
* Updated documentation and example gallery.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/32
  - https://github.com/lucabaldini/aptapy/issues/30


Version 0.11.0 (2025-10-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New mechanism for matplotlib style management in the plotting module.
* Replace the ``configure()`` function with ``apply_stylesheet()`` and ``stylesheet_context()``
  context manager for better control over style application.
* Shipping a custom dark theme and bundling the Humor Sans font for xkcd-style
  stylesheet.
* Dependencies on ``cycler`` and ``loguru`` packages removed.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/31
  - https://github.com/lucabaldini/aptapy/issues/27


Version 0.10.2 (2025-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Default vertical alignment for the text in the ``ConstrainedTextMarker`` class changed
  from "center" to "bottom" to avoid overlapping with the underlying trajectory for
  nearly horizontal data sets.
* Fixed a bug in the interactive cursor logic triggered by mouse events outside the axes
  area.
* Fixed random seed in tests/test_hist.py to ensure consistent results across runs.
* Added a --interactive option to pytest to allow keeping matplotlib figures open
  after test execution for interactive inspection.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/29


Version 0.10.1 (2025-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Small refactoring in modeling.py.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/28


Version 0.10.0 (2025-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New ``AbstractPlottable`` base class with standard ``plot()`` and abstract ``_render()``
  methods.
* Refactored ``AbstractFitModelBase``, ``AbstractHistogram``, and ``StripChart``, as
  well as all fit models, to inherit from ``AbstractPlottable``.
* Simplified plotting range management for fit models.
* Example gallery revamped.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/26


Version 0.9.3 (2025-10-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added dependencies on sphinxcontrib-programoutput and nox.
* Added new sections in the documentation for the installation and development
  workflows.
* Refactored nox tasks for better build cleanup functionality
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/25


Version 0.9.2 (2025-10-22)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added error handling in ConstrainedTextMarker.move() to gracefully hide markers
  when trajectory calculations fail (e.g., when extrapolating outside data range).
* Enhanced StripChart.spline() to support configurable extrapolation behavior
  via the ext parameter.
* Refactored last_line_color() to accept an optional axes parameter, improving
  reusability and eliminating redundant plt.gca() calls.
* Updated unit tests.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/24


Version 0.9.1 (2025-10-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed package logo not appearing on PyPI by using absolute URL in README.md.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/22
  - https://github.com/lucabaldini/aptapy/issues/21


Version 0.8.0 (2025-10-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Public interface for the StripChart class improved: append() and extend() merged
  into put(), that should handle both single values and iterables.
* Added __len__() method to support len() on StripChart objects.
* Comprehensive test coverage for various input types and error conditions.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/20
  - https://github.com/lucabaldini/aptapy/issues/19


Version 0.7.1 (2025-10-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix for issue #15 (traceback when plotting empty histograms).
* set_max_length() method added to strip charts to allow changing the max length
  of the underlying deques.
* Avoid catching bare exception in __init__.py.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/18
  - https://github.com/lucabaldini/aptapy/pull/17
  - https://github.com/lucabaldini/aptapy/issues/16
  - https://github.com/lucabaldini/aptapy/issues/15


Version 0.7.0 (2025-10-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Strip chart formatting on the x-axis improved, and full refactoring of the
  StripChart class, with the addition of the EpochStripChart subclass.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/14
  - https://github.com/lucabaldini/aptapy/issues/13


Version 0.6.0 (2025-10-17)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Addition of VerticalCursor and ConstrainedTextMarker classes for interactive
  plotting.
* Enhancement of StripChart with method chaining and spline interpolation \
  capabilities.
* Comprehensive test coverage for the new cursor functionality.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/12


Version 0.5.0 (2025-10-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added init_parameters method to most model classes.
* Updated import structure to use scipy.special module directly instead of importing erf.
* Added comprehensive test coverage for the new parameter initialization functionality.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/10
  - https://github.com/lucabaldini/aptapy/issues/9


Version 0.4.0 (2025-10-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added 2-dimensional histogram example.
* Adds several new model classes (Quadratic, PowerLaw, Exponential, Erf, ErfInverse).
* Implements analytical integration methods for models where possible, with a fallback
  to numerical integration in the base class.
* Updates the FitStatus class with a completion check method.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/7


Version 0.3.2 (2025-10-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Adding binned_statistics method in AbstractHistogram base class to calculate
  statistics from histogram bins
* Adds extensive test coverage in both 1D and 2D histogram test functions with
  statistical validation
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/6


Version 0.3.1 (2025-10-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Minor changes.


Version 0.3.0 (2025-10-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New strip-chart facilities added.
* Introduction of model summation capability through operator overloading
* Refactored class hierarchy with new abstract base classes
* Enhanced parameter compatibility checking methods
* Improved histogram integration for fitting
* Adds sphinx-gallery integration with 5 example scripts demonstrating histogram
  and fitting functionality
* Improves statistical analysis by adding p-value calculations and fixing degrees
  of freedom calculations
* Updates test assertions to include p-value validation
* Pull requests merged  and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/3
  - https://github.com/lucabaldini/aptapy/pull/4
  - https://github.com/lucabaldini/aptapy/pull/5


Version 0.2.0 (2025-10-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* New histogram facilities added.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/2


Version 0.1.1 (2025-10-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initial release on PyPI.
