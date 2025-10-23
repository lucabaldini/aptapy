.. _develop:

Development
===========

All the basic development tasks are automated via `nox <https://nox.thea.codes/en/stable/>`_,
and the most effective way to grasp the development workflow is to inspect the `noxfile.py`,
which should be pretty much self-explanatory:

.. literalinclude:: ../noxfile.py
   :language: python
   :start-at: import


Keep in mind that, in order to run nox sessions, you need to have ``nox`` installed in
your Python environment (e.g., via `pip install nox`), along with all the optional
dependencies in the ``pyproject.toml``:

.. literalinclude:: ../pyproject.toml
   :start-at: [project.optional-dependencies]
   :end-before: # end-deps


Creating a release
------------------

The package includes a simple release script, located in the ``tools/`` directory,
which automates the version bump, changelog update, git tagging, and publishing
to PyPI. To use it, simply run:

.. program-output:: python ../tools/release.py --help
