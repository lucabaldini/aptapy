.. _install:

Installation
============

If you simply want to install the latest released version of aptapy and start
using it, things are fairly simple:

.. code-block:: console

   pip install aptapy

if this is your first time installing aptapy, or

.. code-block:: console

   pip install --upgrade aptapy

if you already have an older version installed and want to upgrade to the latest
release. If you come across any problem, by all means feel free to open an
issue on the `GitHub issue tracker <https://github.com/lucabaldini/aptapy/issues>`_.

(It goes without saying, we assume you got the standard lecture on
Python `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_.)


Pre-requisites
--------------

If you are installing aptapy from PyPI using pip, you do not particularly care about
pre-requisites, as pip will take care of installing any required dependency
automatically. In any event, here is the relevant portion of the ``pyproject.toml``
file listing the required dependencies:

.. literalinclude:: ../pyproject.toml
   :start-after: # start-deps
   :end-before: # end-deps

.. note::

   We do test aptapy on Python 3.7 and later in our continuous integration.
   We cannot exclude that the thing might work on earlier versions of Python,
   but if your Python predates 3.7, you should definitely consider upgrading---aptapy
   is probably not your biggest problem in this case.


Editable installation
---------------------

If you plan on contributing to the development of aptapy, or just want to
explore the codebase, you may want to install it in "editable" mode. To do so,
first clone the repository from GitHub:

.. code-block:: console

   git clone git@github.com:lucabaldini/aptapy.git

and then do a pip editable install from within the repository directory:

.. code-block:: console

   cd aptapy
   pip install -e .[dev,docs]

(Invoking ``pip`` with the ``-e`` command-line switch will place a special link
in the proper folder pointing back to your local version of the source files---instead
of copying the source tree---so that you will always see the last version of the
code as you modify it, e.g., in the local copy of your git repository.)