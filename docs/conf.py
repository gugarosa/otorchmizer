# Copyright (c) 2021-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from importlib.metadata import version as package_version

project = "otorchmizer"
copyright = "2021-2026, Gustavo de Rosa"
author = "Gustavo de Rosa"

release = package_version("otorchmizer")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
napoleon_numpy_docstring = False
autoclass_content = "both"
autodoc_typehints = "none"

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = None

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True,
}

htmlhelp_basename = "otorchmizer_doc"

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

man_pages = [(master_doc, "otorchmizer", "Otorchmizer Documentation", [author], 1)]

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

epub_title = project
epub_exclude_files = ["search.html"]

autodoc_default_options = {"exclude-members": "__weakref__"}
autodoc_member_order = "bysource"
