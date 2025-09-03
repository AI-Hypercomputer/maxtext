# Copyright 2023–2025 Google LLC
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


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaxText"
copyright = "2025, MaxText developers"
author = "MaxText developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb", ".md"]

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
myst_linkify_fuzzy_links = False

# Remove specific documents from ToC
exclude_patterns = [
    "guides/run_src/MaxText_via_multihost_job.md",
    "guides/run_src/MaxText_via_multihost_runner.md",
    "guides/llm_calculator.ipynb",
]
