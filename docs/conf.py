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

import os
import os.path
import sys
import logging
from sphinx.util import logging as sphinx_logging

MAXTEXT_REPO_ROOT = os.environ.get("MAXTEXT_REPO_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(MAXTEXT_REPO_ROOT, "src")))

project = "MaxText"
# pylint: disable=redefined-builtin
copyright = "2023–2026, Google LLC"
author = "MaxText developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    # This needs to be before autodoc^
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb", ".md"]

# Suppress specific warnings
suppress_warnings = [
    "autodoc.import_object"
]

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

# -- Options for autodoc ----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = [
    "cloud_tpu_diagnostics",
    "google_cloud_mldiagnostics",
    "jetstream",
    "librosa",
    "ml_goodput_measurement",
    "pathwaysutils",
    "safetensors",
    "tensorflow_datasets",
    "torch",
    "tpu_inference",
    "transformer_engine",
    "vllm",
]
autosummary_generate = True

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
    os.path.join("guides", "run_maxtext", "run_maxtext_via_multihost_job.md"),
    os.path.join("guides", "run_maxtext", "run_maxtext_via_multihost_runner.md"),
    os.path.join("explanations", "llm_calculator.ipynb"),
    os.path.join("run_maxtext", "run_maxtext_via_multihost_job.md"),
    os.path.join("run_maxtext", "run_maxtext_via_multihost_runner.md"),
    os.path.join("reference", "core_concepts", "llm_calculator.ipynb"),
    os.path.join("reference", "api_generated", "modules.rst"),
    os.path.join("reference", "api_generated", "install_maxtext_extra_deps.rst"),
    os.path.join("reference", "api_generated", "install_maxtext_extra_deps.install_github_deps.rst"),
]


# -- Autogenerate API documentation ------------------------------------------
def run_apidoc(_):
  """Runs sphinx-apidoc to generate API documentation.

  This function is connected to the Sphinx build process and is triggered to
  automatically generate the reStructuredText (RST) files for the API
  documentation from the docstrings in the MaxText source code.

  Args:
    _: The Sphinx application object. Not used.
  """
  # directly within the Sphinx process, especially on macOS, as it avoids
  # potential multiprocessing/forking issues like the "mutex lock failed" error.
  # pylint: disable=import-outside-toplevel
  import subprocess

  os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "1"

  assert os.path.isfile(os.path.join(MAXTEXT_REPO_ROOT, "pyproject.toml"))

  # The path where the generated RST files will be stored
  output_path = os.path.join(MAXTEXT_REPO_ROOT, "docs", "reference", "api_generated")

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
      os.path.join(MAXTEXT_REPO_ROOT, "src"),
      # Paths to exclude
      os.path.join(MAXTEXT_REPO_ROOT, "tests"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "experimental"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "inference_mlperf"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "scratch_code"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "utils", "ckpt_conversion"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "rl"),
      os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText", "multimodal_utils.py"),
  ]

  # Run the command and check for errors
  try:
    print("Running sphinx-apidoc...")
    subprocess.check_call(
      command, env={**os.environ, **{"OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "1"}}
    )
  except subprocess.CalledProcessError as e:
    print(f"sphinx-apidoc failed with error: {e}", file=sys.stderr)
    sys.exit(1)


class FilterSphinxWarnings(logging.Filter):
  """Filter autosummary 'duplicate object description' warnings.

  These warnings are unnecessary as they do not cause missing documentation
  or rendering issues, so it is safe to filter them out.
  """

  def __init__(self, app):
    self.app = app
    super().__init__()

  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    filter_out = ("duplicate object description",)
    return not msg.strip().startswith(filter_out)


def setup(app):
  """Set up the Sphinx application with custom behavior."""

  # Connect the apidoc generation to the Sphinx build process
  run_apidoc(None)
  print("running:", app)

  # Set up custom logging filters
  logger = logging.getLogger("sphinx")
  warning_handler, *_ = [
    h for h in logger.handlers if isinstance(h, sphinx_logging.WarningStreamHandler)
  ]
  warning_handler.filters.insert(0, FilterSphinxWarnings(app))
