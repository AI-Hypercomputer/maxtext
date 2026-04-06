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

"""Tests for model_creation_utils."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp

from orbax import checkpoint as ocp

# Import the private helpers under test.
from maxtext.utils.model_creation_utils import _fix_restore_args_for_shape_mismatch


# ---------------------------------------------------------------------------
# Minimal stub for ArrayMetadata (avoids a real Orbax checkpoint on disk).
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _FakeArrayMetadata:
  shape: tuple
  dtype: object = jnp.float32
  sharding: object = None


def _is_fake_meta(x):
  return isinstance(x, _FakeArrayMetadata)


# Monkey-patch the module-level helper so our fake metadata is recognised.
import maxtext.utils.model_creation_utils as _mcu

_orig_is_orbax = _mcu._is_orbax_array_metadata  # pylint: disable=protected-access
_mcu._is_orbax_array_metadata = _is_fake_meta  # pylint: disable=protected-access


def _make_restore_arg(global_shape):
  """Return an ArrayRestoreArgs with a trivial NamedSharding."""
  mesh = jax.sharding.Mesh(jax.local_devices()[:1], ("x",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return ocp.ArrayRestoreArgs(
      global_shape=global_shape,
      shape=global_shape,
      sharding=sharding,
      dtype=jnp.float32,
  )


class FixRestoreArgsRankGuardTest(unittest.TestCase):
  """_fix_restore_args_for_shape_mismatch must not touch args when stored rank != model rank."""

  def setUp(self):
    self.mesh = jax.sharding.Mesh(jax.local_devices()[:1], ("x",))

  def _run_fix(self, stored_shape, model_shape):
    restore_args = {"kernel": _make_restore_arg(model_shape)}
    metadata_tree = {"kernel": _FakeArrayMetadata(shape=stored_shape)}
    return _fix_restore_args_for_shape_mismatch(restore_args, metadata_tree, self.mesh)

  def test_scanned_ckpt_unscanned_model_not_modified(self):
    """Rank mismatch (scanned ckpt rank 4 vs unscanned model rank 3): arg must be unchanged."""
    # Simulates: scanned checkpoint key kernel (94, 4096, 4, 128) vs vLLM model (4096, 64, 128).
    stored_shape = (94, 4096, 4, 128)
    model_shape = (4096, 64, 128)
    fixed = self._run_fix(stored_shape, model_shape)
    arg = fixed["kernel"]
    # The restore arg should be unchanged — global_shape still points to model_shape.
    self.assertEqual(arg.global_shape, model_shape)

  def test_same_rank_shape_mismatch_is_modified(self):
    """Same rank, shape mismatch (KV padding): arg should be switched to replicated."""
    # Simulates: unscanned checkpoint (4096, 4, 128) vs padded model (4096, 64, 128).
    stored_shape = (4096, 4, 128)
    model_shape = (4096, 64, 128)
    fixed = self._run_fix(stored_shape, model_shape)
    arg = fixed["kernel"]
    # global_shape must be cleared (set to None) so Orbax loads the stored shape as-is.
    self.assertIsNone(arg.global_shape)

  def test_same_shape_no_modification(self):
    """Identical shapes: arg must be unchanged."""
    shape = (4096, 4, 128)
    fixed = self._run_fix(shape, shape)
    arg = fixed["kernel"]
    self.assertEqual(arg.global_shape, shape)

  def test_scanned_both_same_rank_shape_mismatch_is_modified(self):
    """Scanned ckpt + scanned model + KV padding (rank 4 both): arg must be modified."""
    stored_shape = (94, 4096, 4, 128)
    model_shape = (94, 4096, 64, 128)
    fixed = self._run_fix(stored_shape, model_shape)
    arg = fixed["kernel"]
    self.assertIsNone(arg.global_shape)


if __name__ == "__main__":
  unittest.main()
