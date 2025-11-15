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

# Automatically apply the 'decoupled' marker (when DECOUPLE_GCLOUD=TRUE) ONLY to
# tests that don't get skipped in decoupled mode. If a test is skipped because it
# requires TPU hardware (`tpu_only`) or any external integration
# (`external_serving`, `external_training`, `diagnostics`), we do NOT add the
# decoupled marker. This lets us select the decoupled tests using `-m decoupled` 
# easily.

import pytest
from MaxText.gcloud_stub import is_decoupled
import jax

# Configure JAX to use unsafe_rbg PRNG implementation to match main scripts
if is_decoupled():
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

try:
  _HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
except Exception:  # pragma: no cover
  _HAS_TPU = False

try:
  _HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
except Exception:  # pragma: no cover
  _HAS_GPU = False


GCE_MARKERS = {"external_serving", "external_training"}

def pytest_collection_modifyitems(config, items):
  """Customize collection:
  - If no GPU, still skip gpu_only tests (explicit skip marker).
  - If no TPU, still skip tpu_only tests (explicit skip marker).
  - If decoupled, deselect tests marked external_serving/external_training, don't skip them,
    so that they don't appear as skipped, but they simply aren't part of the session.
  - Apply decoupled marker to remaining collected tests.
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
    cur_test_markers = {m.name for m in item.iter_markers()} # Iterate thru the markers of every test
    # Hardware skip retains skip semantics
    if skip_no_tpu and "tpu_only" in cur_test_markers:
      item.add_marker(skip_no_tpu)
      remaining.append(item)
      continue

    if skip_no_gpu and "gpu_only" in cur_test_markers:
      item.add_marker(skip_no_gpu)
      remaining.append(item)
      continue

    if decoupled and (cur_test_markers & GCE_MARKERS):
      # Deselect external tests entirely
      deselected.append(item)
      continue
    remaining.append(item)

  # Update items in-place to only keep remaining tests
  items[:] = remaining
  if deselected:
    config.hook.pytest_deselected(items=deselected)

  # Add decoupled marker to all remaining tests if decoupled
  if decoupled:
    for item in remaining:
      item.add_marker(pytest.mark.decoupled)

def pytest_configure(config):
  for m in [
      "gpu_only: tests that require GPU hardware",
      "tpu_only: tests that require TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCE deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)
