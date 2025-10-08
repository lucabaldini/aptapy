# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

from aptapy import __version__, __name__ as __package_name__


# Get package metadata.
_metadata = importlib.metadata.metadata(__package_name__)


# --- Project information ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = __package_name__
author = _metadata["Author-email"]
copyright = f"2025-%Y, {author}"
version = __version__
release = version


# --- General configuration ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True
}
todo_include_todos = True


sphinx_gallery_conf = {
    "examples_dirs": ["examples"],      # source example scripts (relative to conf.py)
    "gallery_dirs": ["auto_examples"],  # generated output (reST + images)
    "filename_pattern": r".*",          # build all files in examples/
    # Optional niceties:
    "download_all_examples": False,
    #"remove_config_comments": True,
    # "backreferences_dir": "gen_modules/backreferences",
    # "doc_module": ("yourpkg",),       # populate backrefs for your package API
    # "thumbnail_size": (320, 240),
    "reset_modules": ("matplotlib", "aptapy.plotting.configure"),
}

# Options for syntax highlighting.
pygments_style = "default"
pygments_dark_style = "default"

# Options for internationalization.
language = "en"

# Options for markup.
rst_prolog = f"""
"""

# Options for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for templating.
templates_path = ["_templates"]


# --- Options for HTML output ---
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_theme_options = {
    "awesome_external_links": True,
}
html_logo = "_static/logo_small.png"
html_favicon = "_static/favicon.ico"
html_permalinks_icon = "<span>#</span>"
html_static_path = ["_static"]