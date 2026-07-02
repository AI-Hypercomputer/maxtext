# Copyright 2025-2026 Google LLC
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

"""Tests for restoring NNX emergency checkpoints.

NNX saves in the on-disk checkpoint layout: Linen params/opt_state/step, plus an nnx_aux
subtree for the rngs and batch stats Linen has no place for. The emergency manager, unlike the
regular one, bakes in the abstract it is built with and restores against that, ignoring the
restore-time item. So that abstract has to be in the on-disk layout already, or Orbax compares
it against the checkpoint and raises.

checkpointing_nnx_roundtrip_test covers the same ground for the regular manager. The emergency
manager writes one state tree, and nnx_aux is part of it, so the same partition has to survive
there too. These tests check that it does.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax


class _Model(nnx.Module):
  """Linear + batch-norm + dropout: exercises weights, batch stats, and rng together."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.bn = nnx.BatchNorm(3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    x = self.bn(self.linear(x), use_running_average=deterministic)
    return self.dropout(x, deterministic=deterministic)


class _CacheModel(nnx.Module):
  """Linear + dropout + a cache, to check ephemeral variables are not checkpointed."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    self.cache = nnx.Cache(jnp.zeros((3,)))

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


_TX = optax.adamw(1e-3)

# Rows must differ: batch-norm centres identical rows to zero, which zeroes the gradients and leaves
# the adam moments at zero, so a dropped-moment restore would go unnoticed.
_TRAIN_X = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)


def _config():
  """Minimal config with the fields save/restore reads for an NNX emergency run."""
  return SimpleNamespace(
      enable_diloco=False,
      enable_checkpointing=True,
      enable_continuous_checkpointing=False,
      enable_emergency_checkpoint=True,
      enable_multi_tier_checkpointing=False,
      enable_autocheckpoint=False,
      checkpoint_period=1,
      local_checkpoint_period=1,
      async_checkpointing=False,
      dataset_type="tfds",
      lora=None,
      checkpoint_storage_target_data_file_size_bytes=checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
      elastic_enabled=False,
  )


def _abstract_state(model_cls):
  """An abstract (ShapeDtypeStruct) nnx.State for `model_cls`.

  Mirrors what get_abstract_state hands the emergency manager: SDS leaves with a replicated
  sharding so Orbax can build restore args in single- or multi-host CI.
  """
  mesh = jax.sharding.Mesh(jax.devices(), ("x",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def make():
    model = model_cls(nnx.Rngs(9))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))

  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding) if hasattr(x, "shape") else x,
      nnx.eval_shape(make),
  )


def _unrestored(x):
  """Whether `x` is a leaf the checkpoint didn't carry.

  Used as `is_leaf` when comparing the round trip. `jax.tree_map` treats None as an empty
  subtree, so without this a dropped leaf is skipped and reads as a match; and an
  unmaterialized ShapeDtypeStruct reaches `jnp.asarray` and dies without naming the path.
  """
  return x is None or isinstance(x, jax.ShapeDtypeStruct)


def _leaf_equal(a, b):
  """Value equality for one leaf of the round trip, handling PRNG keys (compared by key_data).

  A leaf the checkpoint didn't carry -- None, or an unmaterialized ShapeDtypeStruct -- is
  unequal, so it is reported with its path rather than skipped.
  """
  if _unrestored(a) or _unrestored(b):
    return a is None and b is None
  a, b = jnp.asarray(a), jnp.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
    return bool(jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b)))
  return bool(jnp.allclose(a, b))


class TestEmergencyManagerAbstractLayout(unittest.TestCase):
  """The manager is built with a checkpoint-layout abstract, not the NNX one.

  Orbax is mocked, so this runs in normal CI without an emergency-checkpoint setup.
  """

  def _abstract_handed_to_manager(self, abstract_state):
    """Calls the constructor with Orbax mocked, and returns the abstract it was handed."""
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    with (
        mock.patch.object(checkpointing.emergency_checkpointing, "CheckpointManager") as manager_cls,
        mock.patch.object(
            checkpointing.emergency_checkpointing.gcs_utils, "mkdir_and_check_permissions", side_effect=epath.Path
        ),
        tempfile.TemporaryDirectory() as d,
    ):
      checkpointing.create_orbax_emergency_checkpoint_manager(
          os.path.join(d, "local"),
          os.path.join(d, "persist"),
          mesh,
          abstract_state,
          local_save_interval_steps=1,
          persistent_save_interval_steps=1,
      )
    return manager_cls.call_args.kwargs["abstract_state"]

  def test_nnx_abstract_is_converted_to_checkpoint_layout(self):
    """An NNX abstract is reshaped into the same on-disk layout the save path writes."""
    passed = self._abstract_handed_to_manager(_abstract_state(_Model))
    self.assertNotIsInstance(passed, nnx.State)
    # The on-disk keys, not the NNX model/optimizer roots that caused the mismatch: the optimizer
    # maps to opt_state/step, and nnx_aux carries the dropout rng and batch stats.
    self.assertCountEqual(["params", "opt_state", "step", "nnx_aux"], passed.keys())

  def test_non_nnx_abstract_is_passed_through_unchanged(self):
    """A Linen state is already in the on-disk layout, so it is handed over as-is."""
    linen_like = SimpleNamespace(params={"a": 1}, opt_state=(), step=0)
    self.assertIs(self._abstract_handed_to_manager(linen_like), linen_like)


class TestEmergencySaveRestoreRoundTrip(unittest.TestCase):
  """Real save->restore cycles through create_orbax_emergency_checkpoint_manager."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  # --- helpers ---------------------------------------------------------------

  def _manager(self, model_cls, name):
    """Returns an emergency manager built from `model_cls`'s abstract, under its own subdirectory."""
    try:
      return checkpointing.create_orbax_emergency_checkpoint_manager(
          os.path.join(self._dir, name, "local"),
          os.path.join(self._dir, name, "persist"),
          jax.sharding.Mesh(jax.devices(), ("x",)),
          _abstract_state(model_cls),
          local_save_interval_steps=1,
          persistent_save_interval_steps=1,
      )
    except Exception as e:  # pylint: disable=broad-except
      raise unittest.SkipTest(f"emergency manager unavailable in this environment: {e}")

  def _trained(self, model):
    """One training step: advances step + dropout rng + batch stats + optimizer moments."""
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    grads = nnx.grad(lambda m: jnp.mean(m(_TRAIN_X, deterministic=False) ** 2))(state.model)
    state.apply_gradients(grads)
    return state

  def _save(self, manager, state, step=1):
    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), _config(), data_iterator=None, step=step)
    manager.wait_until_finished()

  def _restore(self, manager, model_cls, seed=123):
    """The trainer's restore: fill the leaves the checkpoint didn't carry from a fresh init."""
    full, _ = checkpointing.load_state_if_possible(
        manager,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_state(model_cls),
        dataset_type="tfds",
        maxtext_config=_config(),
    )
    # The emergency branch hands back the nnx.State itself, not under an "items" key.
    model = model_cls(nnx.Rngs(seed))
    init = nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))
    merged = jax.tree.map(
        lambda ckpt, i: i if isinstance(ckpt, jax.ShapeDtypeStruct) else ckpt,
        full.to_pure_dict(),  # checkpoint values, with placeholders where it carried nothing
        init.to_pure_dict(),
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )
    nnx.replace_by_pure_dict(init, merged)  # missing leaves keep their init value
    return init.to_pure_dict()

  # --- full-state fidelity ---------------------------------------------------

  def test_full_state_round_trip_is_exact(self):
    """Every leaf -- weights, optimizer, dropout rng, batch stats -- round-trips unchanged."""
    manager = self._manager(_Model, "exact")
    state = self._trained(_Model(nnx.Rngs(0)))
    saved = nnx.state(state).to_pure_dict()
    self._save(manager, state)
    restored = self._restore(manager, _Model)

    self.assertEqual(jax.tree_util.tree_structure(saved), jax.tree_util.tree_structure(restored))
    matches = jax.tree_util.tree_map(_leaf_equal, saved, restored, is_leaf=_unrestored)
    mismatched = [jax.tree_util.keystr(p) for p, ok in jax.tree_util.tree_leaves_with_path(matches) if not ok]
    self.assertEqual(mismatched, [], f"leaves differ after round trip: {mismatched}")
    # Nothing was skipped: every leaf of the saved state was actually compared.
    self.assertEqual(len(jax.tree_util.tree_leaves(saved, is_leaf=_unrestored)), len(jax.tree_util.tree_leaves(matches)))

  def test_nnx_aux_restored_not_reset_to_init(self):
    """The dropout rng, batch stats and optimizer step come from the checkpoint, not a fresh init.

    A reset would leave the rng count at 0 and the batch-norm mean at its init value, and both
    keep the tree structure intact. So this checks the values, not just the shape.
    """
    manager = self._manager(_Model, "aux")
    state = self._trained(_Model(nnx.Rngs(0)))
    saved = nnx.state(state).to_pure_dict()
    saved_count = int(saved["model"]["dropout"]["rngs"]["count"])
    self.assertGreater(saved_count, 0, "fixture: training did not advance the dropout rng")
    self._save(manager, state)
    restored = self._restore(manager, _Model)

    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), saved_count)
    self.assertTrue(jnp.allclose(restored["model"]["bn"]["mean"], saved["model"]["bn"]["mean"]))
    self.assertEqual(int(restored["optimizer"]["step"]), 1)

  # --- exclusion of ephemeral variables --------------------------------------

  def test_caches_excluded_from_checkpoint(self):
    """A cache is ephemeral, so it is absent from the abstract the manager bakes in.

    On restore it keeps its init value instead of the saved one, while the dropout rng in the
    same model still comes back from nnx_aux.
    """
    manager = self._manager(_CacheModel, "cache")
    model = _CacheModel(nnx.Rngs(0))
    model.cache.value = jnp.ones((3,))  # the value that would come back if caches were checkpointed
    state = self._trained(model)
    self._save(manager, state)
    restored = self._restore(manager, _CacheModel)  # a fresh init cache is zeros

    self.assertTrue(jnp.allclose(restored["model"]["cache"], jnp.zeros((3,))))
    saved_count = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), saved_count)


if __name__ == "__main__":
  unittest.main()
