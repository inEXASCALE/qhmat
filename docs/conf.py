"""Sphinx configuration for qhmat's user and API documentation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import qhmat

project = "qhmat"
author = "Xinye Chen and contributors"
release = qhmat.__version__
copyright = "Xinye Chen and contributors"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.napoleon"]
root_doc = "index"
exclude_patterns = ["_build"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
nitpick_ignore_regex = [
    ("py:class", r"numpy\..*"),
    ("py:class", r"collections\.abc\.Sequence"),
    ("py:class", r"os\.PathLike"),
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_title = f"qhmat {release} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
