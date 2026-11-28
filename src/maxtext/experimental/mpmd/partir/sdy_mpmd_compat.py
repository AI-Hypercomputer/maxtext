# Copyright 2026 Google LLC
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

"""Imports Shardy MPMD (OSS PartIR:MPMD) on top of the installed jaxlib wheel.

Shardy MPMD's Python frontend (``shardy.integrations.python.jax.mpmd``) is an optional dependency: it is neither on
PyPI nor in the jax or jaxlib wheels, and it must come from the Shardy commit the installed jaxlib was built from
(``sdy_mpmd_versions.SHARDY_COMMITS``, whose ``install_command`` prints the pip line). Only the code that builds,
lowers or traces an MPMD program imports this module, so the rest of the package runs without it.

Two Bazel-internal module names that the extension and the frontend import do not exist in the jaxlib wheel:

  * ``jaxlib._sdy_mpmd`` imports ``jax.jaxlib._jax``: the import fails with "No module named 'jax.jaxlib'", and aliasing
    only the package makes Python load a second copy of ``_jax.so``, which aborts;
  * the frontend's ``jit.py`` does ``from mlir import ir``; the wheel ships MLIR's Python bindings as ``jaxlib.mlir``.

Each name is pointed at the module jax has already loaded, before the frontend is imported.
"""

import sys

import jax  # pylint: disable=unused-import
import jaxlib
from jaxlib import _jax
from jaxlib import mlir as jaxlib_mlir
from jaxlib.mlir import ir as jaxlib_mlir_ir

from maxtext.utils import max_logging

from maxtext.experimental.mpmd.partir import sdy_mpmd_versions

# The aliases must exist before `jaxlib._sdy_mpmd` or the frontend is imported. `setdefault` keeps a top-level
# `mlir` module that is already imported; an installed `mlir` package that nothing has imported yet is shadowed.
sys.modules.setdefault("jax.jaxlib", jaxlib)
sys.modules.setdefault("jax.jaxlib._jax", _jax)
sys.modules.setdefault("mlir", jaxlib_mlir)
sys.modules.setdefault("mlir.ir", jaxlib_mlir_ir)

try:
  # pylint: disable=wrong-import-position
  from shardy.integrations.python.jax import mpmd
  from shardy.integrations.python.jax.mpmd import pipeline as shardy_pipeline
except ModuleNotFoundError as error:
  if not (error.name or "").startswith("shardy"):  # one of the frontend's own dependencies is missing
    raise
  raise ModuleNotFoundError(
      f"Shardy MPMD's Python frontend is not installed: {sdy_mpmd_versions.install_command(jaxlib.__version__)}.",
      name=error.name,
  ) from error

_MISMATCH = sdy_mpmd_versions.version_mismatch(jaxlib.__version__, sdy_mpmd_versions.installed_commit())
if _MISMATCH:
  max_logging.warning(_MISMATCH)

__all__ = ["mpmd", "shardy_pipeline"]
