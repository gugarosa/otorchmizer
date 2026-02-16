# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import otorchmizer

# -- Project information -----------------------------------------------------

project = "otorchmizer"
copyright = "2026, Gustavo de Rosa"
author = "Gustavo de Rosa"

version = otorchmizer.__version__
release = otorchmizer.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = None

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "otorchmizer_doc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (
        master_doc,
        "otorchmizer.tex",
        "Otorchmizer Documentation",
        "Gustavo de Rosa",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "otorchmizer", "Otorchmizer Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "otorchmizer",
        "Otorchmizer Documentation",
        author,
        "otorchmizer",
        "A PyTorch-based nature-inspired meta-heuristic optimization framework.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------

autodoc_default_options = {"exclude-members": "__weakref__"}
autodoc_member_order = "bysource"
