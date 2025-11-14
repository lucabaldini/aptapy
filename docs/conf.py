# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import pathlib
import sys

from aptapy import __version__, __name__ as __package_name__

_SCRIPTS_DIR = pathlib.Path(__file__).parent / "_scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

def setup(app):
    import scipy_rv_continuous
    import sigmoids
    scipy_rv_continuous.create_figures()
    sigmoids.create_figures()
    return {"version": "1.0", "parallel_read_safe": True}


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
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
    "sphinx_gallery.gen_gallery",
]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True
}
todo_include_todos = True


extlinks = {
    "scipy_rv_wrap": ("https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.%s.html",
                      "scipy.stats.%s"),
}

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
    "reset_modules": ("matplotlib", "aptapy.plotting.reset"),
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
html_css_files = ['aptapy.css']