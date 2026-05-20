# Copyright 2025 Google LLC
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

"""Pytest configuration helpers for decoupled test selection.

Automatically apply the `decoupled` marker (when DECOUPLE_GCLOUD=TRUE) to
tests that remain collected. Tests that are explicitly skipped because they
require external integrations or specific hardware (for example `tpu_only`)
are not marked.
"""

import pytest
import jax
import importlib.util

# --- Monkeypatch for absl.testing.parameterized ---
# Context: Decorating a test method with @parameterized.named_parameters returns a custom
# iterable container (_ParameterizedTestIter) instead of a standard function object.
# Problem: When pytest markers are applied above @parameterized in the decorator stack:
#
#   @pytest.mark.cpu_only
#   @parameterized.named_parameters(...)
#   def test_foo(self, ...):
#
# pytest attaches the marker attributes exclusively to the outer iterable container object.
# During class initialization, the test metaclass unwraps the base function to generate
# individual test methods, omitting the outer container entirely. Consequently, marker
# attributes attached to the outer container are dropped and lost before pytest collection.
# Solution: Intercept _ParameterizedTestIter.__iter__ to dynamically propagate any discovered
# pytestmark attributes from the outer container object down to all generated test methods.
from absl.testing import parameterized

try:
  # pylint: disable=protected-access
  _orig_iter = parameterized._ParameterizedTestIter.__iter__

  def _custom_iter(self):
    """Custom iterator propagating outer pytestmark attributes to generated test methods."""
    outer_marks = getattr(self, "pytestmark", None)
    if outer_marks is None:
      yield from _orig_iter(self)
    else:
      if not isinstance(outer_marks, list):
        outer_marks = [outer_marks]

      for func in _orig_iter(self):
        existing_marks = getattr(func, "pytestmark", [])
        if not isinstance(existing_marks, list):
          existing_marks = [existing_marks]
        func.pytestmark = existing_marks + outer_marks
        yield func

  parameterized._ParameterizedTestIter.__iter__ = _custom_iter
  # pylint: enable=protected-access
except AttributeError:
  pass

import os

if os.getenv("JAX_PLATFORMS") == "proxy":
  # Import maxtext early to register the pathways proxy backend before JAX is queried.
  import maxtext  # pylint: disable=unused-import

try:
  _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_TPU = False

try:
  _HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_GPU = False

from maxtext.common.gcloud_stub import is_decoupled

# Configure JAX to use unsafe_rbg PRNG implementation to match main scripts.
if is_decoupled():
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")


GCP_MARKERS = {"external_serving", "external_training"}


def _has_tpu_backend_support() -> bool:
  """Whether JAX has TPU backend support installed (PJRT TPU plugin).

  This is intentionally *not* the same as having TPU hardware available.
  """
  try:
    if importlib.util.find_spec("jaxlib") is None:
      return False
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False

  # Heuristic: TPU backend support is provided via the `libtpu` package.
  try:
    return importlib.util.find_spec("libtpu") is not None
  except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
    return False


def pytest_collection_modifyitems(config, items):
  """Customize pytest collection behavior.

  - Skip hardware-specific tests when hardware is missing.
  - Deselect tests marked as external_serving/training in decoupled mode.
  - Mark remaining tests with the `decoupled` marker when running decoupled.
  """
  decoupled = is_decoupled()
  remaining = []
  deselected = []

  skip_no_tpu = None
  skip_no_gpu = None
  skip_no_tpu_backend = None
  if not _HAS_TPU:
    skip_no_tpu = pytest.mark.skip(reason="Skipped: requires TPU hardware, none detected")

  if not _HAS_GPU:
    skip_no_gpu = pytest.mark.skip(reason="Skipped: requires GPU hardware, none detected")

  if not _has_tpu_backend_support():
    skip_no_tpu_backend = pytest.mark.skip(
        reason=(
            "Skipped: requires a TPU-enabled JAX install (TPU PJRT plugin). "
            "Install a TPU-enabled jax/jaxlib build to run this test."
        )
    )

  for item in items:
    # Iterate thru the markers of every test.
    cur_test_markers = {m.name for m in item.iter_markers()}

    # Hardware skip retains skip semantics.
    if skip_no_tpu and "tpu_only" in cur_test_markers:
      item.add_marker(skip_no_tpu)
      remaining.append(item)
      continue

    if skip_no_gpu and "gpu_only" in cur_test_markers:
      item.add_marker(skip_no_gpu)
      remaining.append(item)
      continue

    if skip_no_tpu_backend and "tpu_backend" in cur_test_markers:
      item.add_marker(skip_no_tpu_backend)
      remaining.append(item)
      continue

    if decoupled and (cur_test_markers & GCP_MARKERS):
      # Deselect tests marked as external_serving/training entirely.
      deselected.append(item)
      continue

    remaining.append(item)

  # Update items in-place to only keep remaining tests.
  items[:] = remaining
  if deselected:
    config.hook.pytest_deselected(items=deselected)

  # Add decoupled marker to all remaining tests when running decoupled.
  if decoupled:
    for item in remaining:
      item.add_marker(pytest.mark.decoupled)


def pytest_configure(config):
  for m in [
      "gpu_only: tests that require GPU hardware",
      "tpu_only: tests that require TPU hardware",
      "tpu_backend: tests that require a TPU-enabled JAX install (TPU PJRT plugin), but not TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCP deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)
