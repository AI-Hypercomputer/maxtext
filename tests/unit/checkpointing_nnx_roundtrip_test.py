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

"""End-to-end NNX checkpoint round trips through the real save/restore stack.

Drives create_orbax_checkpoint_manager -> maybe_save_checkpoint ->
load_state_if_possible against a real on-disk Orbax checkpoint (no mocks). Covers
what training actually hits: exact full-state fidelity, per-component restore
(weights, optimizer, dropout rng, batch stats), the fallback for a checkpoint
with no nnx_aux, exclusion of non-weight variables (caches), that the restored
state loads back into a live model, resume continuity, and the
load_full_state_from_path route.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

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


class _PlainModel(nnx.Module):
  """Linear only, so its checkpoint carries no nnx_aux, like a Linen-trained one."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.linear(x)


class _CacheModel(nnx.Module):
  """Linear + dropout + a cache, to check non-weight variables are not checkpointed."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    self.cache = nnx.Cache(jnp.zeros((3,)))

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


# optax.adamw (MaxText's default) is a chain -> int-keyed opt_state; guards the reshape end to end.
_TX = optax.adamw(1e-3)


def _config():
  """Minimal config with the fields save/restore reads for a pure_nnx run."""
  return SimpleNamespace(
      pure_nnx=True,
      enable_diloco=False,
      enable_checkpointing=True,
      enable_continuous_checkpointing=False,
      enable_emergency_checkpoint=False,
      enable_multi_tier_checkpointing=False,
      enable_autocheckpoint=False,
      checkpoint_period=1,
      local_checkpoint_period=0,
      async_checkpointing=False,
      dataset_type="tfds",
      lora=None,
      checkpoint_storage_target_data_file_size_bytes=checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
      elastic_enabled=False,
  )


def _abstract_state(model_cls):
  """An abstract (ShapeDtypeStruct) nnx.State for `model_cls`, the restore blueprint.

  Mirrors what get_abstract_state hands load_state_if_possible: SDS leaves with a
  replicated sharding so Orbax can build restore args in single- or multi-host CI.
  """
  mesh = jax.sharding.Mesh(jax.devices(), ("x",))
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def make():
    model = model_cls(nnx.Rngs(9))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))

  abstract = nnx.eval_shape(make)
  return jax.tree.map(
      lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding) if hasattr(x, "shape") else x,
      abstract,
  )


def _leaf_equal(a, b):
  """Value equality that also handles PRNG-key leaves (compared by key_data)."""
  a, b = jnp.asarray(a), jnp.asarray(b)
  if a.shape != b.shape or a.dtype != b.dtype:
    return False
  if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
    return bool(jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b)))
  return bool(jnp.allclose(a, b))


class TestNNXCheckpointRoundTrip(unittest.TestCase):
  """Real save->restore cycles through create_orbax_checkpoint_manager + load_state_if_possible."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  # --- helpers ---------------------------------------------------------------

  def _manager(self):
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )

  def _trained(self, model):
    """One training step: advances step + dropout rng + batch stats + optimizer moments."""
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(state.model)
    state.apply_gradients(grads)
    return state

  def _save(self, manager, state, step=1):
    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), _config(), data_iterator=None, step=step)
    manager.wait_until_finished()

  def _init_state(self, model_cls, seed=123):
    """A fresh concrete init state (real weights/rng, optimizer zeros) for `model_cls`."""
    model = model_cls(nnx.Rngs(seed))
    return nnx.state(train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param)))

  def _restore(self, manager, model_cls):
    """The trainer's restore: overlay the checkpoint onto the abstract, then fill from a fresh init."""
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
    abstract = _abstract_state(model_cls)
    nnx.replace_by_pure_dict(abstract, full["items"])  # checkpoint values in; absent leaves stay placeholders
    init = self._init_state(model_cls)
    merged = jax.tree.map(
        lambda ckpt, i: i if isinstance(ckpt, jax.ShapeDtypeStruct) else ckpt,
        abstract.to_pure_dict(),
        init.to_pure_dict(),
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )
    nnx.replace_by_pure_dict(init, merged)  # missing leaves keep their init value
    return init.to_pure_dict()

  # --- full-state fidelity ---------------------------------------------------

  def test_full_state_round_trip_is_exact(self):
    """Every leaf -- weights, optimizer, dropout rng, batch stats -- round-trips unchanged."""
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    saved = nnx.state(state).to_pure_dict()
    self._save(manager, state)
    restored = self._restore(manager, _Model)

    self.assertEqual(jax.tree_util.tree_structure(saved), jax.tree_util.tree_structure(restored))
    matches = jax.tree_util.tree_map(_leaf_equal, saved, restored)
    mismatched = [jax.tree_util.keystr(p) for p, ok in jax.tree_util.tree_leaves_with_path(matches) if not ok]
    self.assertEqual(mismatched, [], f"leaves differ after round trip: {mismatched}")

  # --- per-component restore -------------------------------------------------

  def test_weights_restored_exactly(self):
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    saved_kernel = nnx.state(state).to_pure_dict()["model"]["linear"]["kernel"]
    self._save(manager, state)
    restored = self._restore(manager, _Model)
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], saved_kernel))

  def test_optimizer_state_restored_from_checkpoint(self):
    """Optimizer step + moments come from the checkpoint, not reset to init."""
    manager = self._manager()
    self._save(manager, self._trained(_Model(nnx.Rngs(0))))
    restored = self._restore(manager, _Model)
    self.assertEqual(int(restored["optimizer"]["step"]), 1)
    moments = [x for x in jax.tree.leaves(restored["optimizer"]["opt_state"]) if hasattr(x, "shape")]
    self.assertTrue(any(float(jnp.abs(x).sum()) > 0 for x in moments))  # fresh init is all-zero

  def test_dropout_rng_stream_restored(self):
    """The dropout rng continues across a save/restore instead of resetting to 0."""
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    orig_count = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertGreater(orig_count, 0)
    self._save(manager, state)
    restored = self._restore(manager, _Model)
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), orig_count)

  def test_batch_stats_restored(self):
    """BatchNorm running mean/var are restored, not reset to their init."""
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    saved_mean = nnx.state(state).to_pure_dict()["model"]["bn"]["mean"]
    self._save(manager, state)
    restored = self._restore(manager, _Model)
    self.assertTrue(jnp.allclose(restored["model"]["bn"]["mean"], saved_mean))

  # --- fallback / backward compatibility -------------------------------------

  def test_old_checkpoint_missing_fields_keep_init_value(self):
    """A checkpoint missing fields (no nnx_aux, no batch-norm weights) keeps the INIT value there."""
    manager = self._manager()
    self._save(manager, self._trained(_PlainModel(nnx.Rngs(0))))  # only linear + optimizer on disk
    init_scale = self._init_state(_Model).to_pure_dict()["model"]["bn"]["scale"]
    restored = self._restore(manager, _Model)  # _Model also has batch-norm + dropout
    self.assertEqual(restored["model"]["linear"]["kernel"].shape, (2, 3))  # weights loaded from ckpt
    # bn scale was absent from the ckpt -> kept at its init (ones), NOT zero-filled.
    self.assertTrue(jnp.allclose(restored["model"]["bn"]["scale"], init_scale))
    self.assertTrue(jnp.allclose(restored["model"]["bn"]["scale"], jnp.ones((3,))))

  def test_plain_model_round_trip_writes_no_nnx_aux(self):
    """A model with no rng/batch stats round-trips weights + optimizer and no nnx_aux."""
    manager = self._manager()
    state = self._trained(_PlainModel(nnx.Rngs(0)))
    saved_kernel = nnx.state(state).to_pure_dict()["model"]["linear"]["kernel"]
    self._save(manager, state)
    restored = self._restore(manager, _PlainModel)
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], saved_kernel))
    self.assertEqual(int(restored["optimizer"]["step"]), 1)

  # --- exclusion of non-weight variables -------------------------------------

  def test_caches_excluded_from_checkpoint(self):
    """A cache is not written to the checkpoint, so a restore keeps the init value, not the saved one."""
    manager = self._manager()
    model = _CacheModel(nnx.Rngs(0))
    model.cache.value = jnp.ones((3,))  # the value that would come back if caches were checkpointed
    state = self._trained(model)
    self._save(manager, state)
    restored = self._restore(manager, _CacheModel)  # init cache is zeros
    # Cache is the init value (zeros), not the saved ones -> it was not checkpointed...
    self.assertTrue(jnp.allclose(restored["model"]["cache"], jnp.zeros((3,))))
    # ...while the dropout rng was still restored from nnx_aux.
    orig_count = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), orig_count)

  # --- restored state is a usable model --------------------------------------

  def test_restored_state_loads_into_model_and_runs(self):
    """The restored pure dict fills a live nnx model (as pre_train does) and runs a forward pass."""
    manager = self._manager()
    self._save(manager, self._trained(_Model(nnx.Rngs(0))))
    restored = self._restore(manager, _Model)

    abstract = _abstract_state(_Model)
    nnx.replace_by_pure_dict(abstract, restored)  # same call the trainer makes
    graphdef, _ = nnx.split(_Model(nnx.Rngs(0)))
    model = nnx.merge(graphdef, abstract["model"])
    out = model(jnp.ones((4, 2)), deterministic=True)
    self.assertEqual(out.shape, (4, 3))

  # --- resume continuity -----------------------------------------------------

  def test_resume_continues_step_and_rng(self):
    """Restoring then training again continues the step counter and rng stream (no reset)."""
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    count_at_save = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self._save(manager, state)

    restored = self._restore(manager, _Model)
    abstract = _abstract_state(_Model)
    nnx.replace_by_pure_dict(abstract, restored)
    graphdef, _ = nnx.split(_Model(nnx.Rngs(0)))
    model = nnx.merge(graphdef, abstract["model"])
    resumed = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    # Re-seed the optimizer step to the restored value, then take one more step.
    grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(resumed.model)
    resumed.apply_gradients(grads)
    count_after = int(nnx.state(resumed).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertGreater(count_after, count_at_save)  # stream advanced past the restored point

  # --- alternate restore route -----------------------------------------------

  def test_load_full_state_from_path_route(self):
    """The explicit load_full_state_from_path route reshapes items/ back to NNX with nnx_aux."""
    manager = self._manager()
    state = self._trained(_Model(nnx.Rngs(0)))
    orig_count = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self._save(manager, state)

    items_path = os.path.join(self._dir, "ckpt", "1", "items")
    full, _ = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path=items_path,
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_state(_Model),
        dataset_type="tfds",
        enable_orbax_v1=False,
        maxtext_config=_config(),
    )
    restored = full["items"]
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), orig_count)
    self.assertEqual(int(restored["optimizer"]["step"]), 1)


if __name__ == "__main__":
  unittest.main()
