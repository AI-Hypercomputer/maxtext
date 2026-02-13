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
from maxtext.common.gcloud_stub import is_decoupled
import jax

# Configure JAX to use unsafe_rbg PRNG implementation to match main scripts.
if is_decoupled():
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

try:
  _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_TPU = False

try:
  _HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:  # pragma: no cover  pylint: disable=broad-exception-caught
  _HAS_GPU = False


GCP_MARKERS = {"external_serving", "external_training"}


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
  if not _HAS_TPU:
    skip_no_tpu = pytest.mark.skip(reason="Skipped: requires TPU hardware, none detected")

  if not _HAS_GPU:
    skip_no_gpu = pytest.mark.skip(reason="Skipped: requires GPU hardware, none detected")

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
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCP deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)
