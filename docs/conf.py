# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder for MaxText.

This file configures the Sphinx build process for the MaxText project
documentation. It sets project information, specifies extensions, defines HTML
output options, and configures MyST parser settings.

For more information on Sphinx configuration, see the official documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaxText"
# pylint: disable=redefined-builtin
copyright = "2023–2026 Google LLC"
author = "MaxText developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/editable_commands.js"]
html_logo = "_static/maxtext.png"

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = [
    "dollarmath",
    "linkify",
    "colon_fence",
]
myst_linkify_fuzzy_links = False

# Theme-specific options
# https://sphinx-book-theme.readthedocs.io/en/stable/reference.html
html_theme_options = {
    "analytics": {
        "google_analytics_id": "G-RC6NRK4GNN",
    },
    "show_navbar_depth": 1,
    "show_toc_level": 1,
    "repository_url": "https://github.com/AI-Hypercomputer/maxtext",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "navigation_with_keys": True,
    "home_page_in_toc": True,
}

# Remove specific documents from ToC
exclude_patterns = [
    "run_maxtext/run_maxtext_via_multihost_job.md",
    "run_maxtext/run_maxtext_via_multihost_runner.md",
    "reference/core_concepts/llm_calculator.ipynb",
]
