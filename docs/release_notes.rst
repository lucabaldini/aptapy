.. _release_notes:

Release notes
=============


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
