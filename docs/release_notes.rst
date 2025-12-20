.. _release_notes:

Release notes
=============


Version 0.18.0 (2025-12-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Changed parametrization of the PowerLaw model to include an arbitrary pivot that
  can be set at creation time.
* Probit fitting model added.
* Added the Histogram3d class for three-dimensional histograms.
* New slice1d(), project_mean(), and project_statistics() methods added in the histogram base class.
* Minor warning fixes in various places (docs and unit tests).
* Updated copyright notice.
* Added team page in the documentation.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/86
  - https://github.com/lucabaldini/aptapy/pull/85
  - https://github.com/lucabaldini/aptapy/pull/82
  - https://github.com/lucabaldini/aptapy/pull/80
  - https://github.com/lucabaldini/aptapy/issues/84
  - https://github.com/lucabaldini/aptapy/issues/83
  - https://github.com/lucabaldini/aptapy/issues/81
  - https://github.com/lucabaldini/aptapy/issues/79


Version 0.17.2 (2025-12-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fix for issue #74 (error when calling ufloat with no errors).
* Correlated fit parameters added to the FitStatus object.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/76
  - https://github.com/lucabaldini/aptapy/issues/74


Version 0.17.1 (2025-12-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Moved ``random_histogram()`` to the ``AbstractFitModelBase`` class
* Signature updated to require histogram edges instead of the number of bins.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/75
  - https://github.com/lucabaldini/aptapy/issues/67


Version 0.17.0 (2025-12-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Refactored GaussianForestBase parameter structure from individual amplitudes
  to a shared amplitude with intensity ratios.
* Moved GaussianForestBase from models.py to modeling.py.
* Significant refactoring to allow for freezing parameters in line forests.
* Added the ``fit_iterative()`` method to the ``GaussianForestBase`` class.
* Added the ``fwhm()`` method to the ``Gaussian`` model class.
* Modified the ``fwhm()`` method of the ``GaussianForestBase`` to return the real FWHM and not
  the resolution of the line.
* Added unit tests.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/72
  - https://github.com/lucabaldini/aptapy/issues/71
  - https://github.com/lucabaldini/aptapy/issues/70
  - https://github.com/lucabaldini/aptapy/issues/68
  - https://github.com/lucabaldini/aptapy/issues/66
  - https://github.com/lucabaldini/aptapy/issues/65


Version 0.16.0 (2025-12-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added the ``GaussianForestBase`` and ``Fe55Forest`` model classes.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/62
  - https://github.com/lucabaldini/aptapy/issues/47


Version 0.15.1 (2025-11-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~


* Fix for a bug in the gaussian iterative fitting in a range.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/issues/60


Version 0.15.0 (2025-11-21)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Naming order for polynomial coefficients changed to match standard mathematical
  notation (e.g., c2 for quadratic term, c1 for linear term, c0 for constant term).
* Keeping track of the covariance matrix in the FitResult object.
* Added numerical Jacobian calculation in the base class for fit models.
* Added capability to plot confidence bands around fit models.
* Added ``fit_iterative()`` method to the ``Gaussian`` model class to perform iterative fitting.
* Added the ``set_content()`` method to ``AbstractHistogram`` to fill a histogram with binned data
* Slight change to the main fitting interface, and fit_histogram() function removed.
* Added the ``from_amptek_file()`` method to create a ``Histogram1d`` from an MCA8000A multichannel
  analyzer output file.
* Pull requests merged and issues closed:

  - https://github.com/lucabaldini/aptapy/pull/58
  - https://github.com/lucabaldini/aptapy/pull/56
  - https://github.com/lucabaldini/aptapy/pull/54
  - https://github.com/lucabaldini/aptapy/pull/53
  - https://github.com/lucabaldini/aptapy/pull/52
  - https://github.com/lucabaldini/aptapy/pull/51
  - https://github.com/lucabaldini/aptapy/issues/50
  - https://github.com/lucabaldini/aptapy/issues/40


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
