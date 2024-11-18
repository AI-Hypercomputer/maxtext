# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaxText"
copyright = "2024, MaxText developers"
author = "MaxText developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb", ".md"]

exclude_patterns = [
    "advanced_docs/*",
    "concepts/checkpointing.md",
    "terminologies.md",
    "guides/inference.md",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = []
# html_logo = "_static/flax.png"

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = [
    "dollarmath",
    "linkify",
    "colon_fence",
]
