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

# -- Path setup --------------------------------------------------------------
import os.path
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MaxText"
copyright = "2025, Google LLC"
author = "Google LLC"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
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

# -- Options for autodoc ----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# Remove specific documents from ToC
exclude_patterns = [
    os.path.join("guides", "run_maxtext_via_multihost_job.md"),
    os.path.join("guides", "run_maxtext_via_multihost_runner.md"),
    os.path.join("guides", "llm_calculator.ipynb"),
]


# -- Autogenerate API documentation ------------------------------------------
# This function automatically runs sphinx-apidoc to generate API documentation
# from the docstrings in the source code.
def run_apidoc(_):
    # Use subprocess to run sphinx-apidoc. This is safer than calling main()
    # directly within the Sphinx process, especially on macOS, as it avoids
    # potential multiprocessing/forking issues like the "mutex lock failed" error.
    import subprocess
    import sys

    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = '1'

    pkg_path = os.path.join(REPO_ROOT, "src", "MaxText")
    # The path where the generated RST files will be stored
    output_path = os.path.join(REPO_ROOT, "docs", "reference", "api_generated")

    # Command to run sphinx-apidoc
    # Note: We use `sys.executable -m sphinx.ext.apidoc` to ensure we're using
    # the apidoc from the same Python environment as Sphinx.
    command = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "--module-first",
        "--force",
        "--separate",
        "--output-dir",
        output_path,
        pkg_path,
        # Paths to exclude
        os.path.join(REPO_ROOT, "tests"),
        os.path.join(REPO_ROOT, "src", "MaxText", "experimental"),
    ]

    # Run the command and check for errors
    try:
        print("Running sphinx-apidoc...")
        subprocess.check_call(command, env={**os.environ, **{'OBJC_DISABLE_INITIALIZE_FORK_SAFETY': '1'}})
    except subprocess.CalledProcessError as e:
        print(f"sphinx-apidoc failed with error: {e}", file=sys.stderr)
        sys.exit(1)


# Connect the apidoc generation to the Sphinx build process
def setup(app):
  run_apidoc(None)
  print('running:', app)
  #app.connect("builder-inited", run_apidoc)
