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


GCE_MARKERS = {"external_serving", "external_training"}

def pytest_collection_modifyitems(config, items):
  """Customize collection:
  - If no TPU, still skip tpu_only tests (explicit skip marker).
  - If decoupled, deselect tests marked external_serving/external_training, don't skip them,
    so that they don't appear as skipped, but they simply aren't part of the session.
  - Apply decoupled marker to remaining collected tests.
  """
  decoupled = is_decoupled()
  remaining = []
  deselected = []

  skip_no_tpu = None
  if not _HAS_TPU:
    skip_no_tpu = pytest.mark.skip(reason="Skipped: requires TPU hardware, none detected")

  for item in items:
    cur_test_markers = {m.name for m in item.iter_markers()} # Iterate thru the markers of every test
    # Hardware skip retains skip semantics
    if skip_no_tpu and "tpu_only" in cur_test_markers:
      item.add_marker(skip_no_tpu)
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
      "tpu_only: tests that require TPU hardware",
      "external_serving: JetStream / serving / decode server components",
      "external_training: goodput integrations",
      "decoupled: marked on tests that are not skipped due to GCE deps, when DECOUPLE_GCLOUD=TRUE",
  ]:
    config.addinivalue_line("markers", m)
