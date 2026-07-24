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

"""Integration test for the real setup_initial_state NNX restore (init + overlay).

Drives maxtext_utils.setup_initial_state end to end on CPU: it builds a concrete
init state and overlays a real on-disk checkpoint onto it, so anything the
checkpoint carried comes from disk and anything it didn't keeps its init value.
Only get_abstract_state is patched (to swap the full model-creation machinery for
a tiny model); the restore/init/overlay path under test runs for real. This is the
unit-level guard for the train->save->resume flow verified on TPU.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.utils import maxtext_utils
import optax
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

_TX = optax.scale_by_adam()  # flat opt_state {count, mu, nu}, as an un-chained optimizer (e.g. adam_pax) produces

# Rows must differ: batch-norm centres identical rows to zero, which zeroes the gradients and leaves
# the adam moments at zero, so a dropped-moment restore would go unnoticed.
_TRAIN_X = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)


class _Model(nnx.Module):
  """Linear + batch-norm + dropout: weights, batch stats, and an rng stream."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.bn = nnx.BatchNorm(3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.bn(self.linear(x), use_running_average=deterministic), deterministic=deterministic)


class _PlainModel(nnx.Module):
  """Linear only -> checkpoint has no batch stats / rng (like a Linen-trained one)."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.linear(x)


class _RngModel(nnx.Module):
  """Same weights as _PlainModel plus a dropout rng -> the extra state is non-Param."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 3, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


def _init_fn(model_cls, seed):
  def fn():
    model = model_cls(nnx.Rngs(seed))
    return train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))

  return fn


def _config():
  """A config with every field setup_initial_state + load_state_if_possible + save read for pure_nnx."""
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
      logical_axis_rules=[],
      load_parameters_path="",
      load_full_state_path="",
      checkpoint_storage_concurrent_gb=8,
      enable_single_replica_ckpt_restoring=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      enable_orbax_v1=False,
      checkpoint_conversion_fn=None,
      source_checkpoint_layout="orbax",
      expansion_factor_real_data=-1,
      scan_layers=False,
  )


class TestSetupInitialStateNNX(unittest.TestCase):
  """Drive the real setup_initial_state restore path with a tiny model on CPU."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
    self._mesh = jax.sharding.Mesh(jax.devices(), ("x",))

  def _manager(self):
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )

  def _sharded_abstract(self, init_fn):
    repl = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
    abstract = nnx.eval_shape(lambda: nnx.state(init_fn()))
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=repl) if hasattr(x, "shape") else x, abstract
    )

  def _setup(self, manager, init_fn, config):
    """Call the real setup_initial_state, patching only get_abstract_state for the tiny model."""
    abstract = self._sharded_abstract(init_fn)
    repl = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
    shardings = jax.tree.map(lambda _: repl, abstract)
    with mock.patch.object(maxtext_utils, "get_abstract_state", return_value=(abstract, None, shardings)):
      state, *_ = maxtext_utils.setup_initial_state(None, config, self._mesh, manager, init_fn, True)
    return state.to_pure_dict()

  def _train_and_save(self, manager, model_cls, config, seed=0):
    """One training step (advances step/rng/batch-stats/optimizer), then save; return the saved pure dict."""
    ts = _init_fn(model_cls, seed)()
    grads = nnx.grad(lambda m: jnp.mean(m(_TRAIN_X, deterministic=False) ** 2))(ts.model)
    ts.apply_gradients(grads)
    saved = nnx.state(ts).to_pure_dict()
    checkpointing.maybe_save_checkpoint(manager, nnx.state(ts), config, data_iterator=None, step=1)
    manager.wait_until_finished()
    return saved

  def test_restores_full_state_via_overlay(self):
    """setup_initial_state overlays the checkpoint onto init: weights + optimizer + rng + batch stats come from disk."""
    config = _config()
    manager = self._manager()
    saved = self._train_and_save(manager, _Model, config, seed=0)
    restored = self._setup(manager, _init_fn(_Model, 999), config)  # a DIFFERENT init seed than the ckpt
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], saved["model"]["linear"]["kernel"]))
    self.assertEqual(int(restored["optimizer"]["step"]), int(saved["optimizer"]["step"]))
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), int(saved["model"]["dropout"]["rngs"]["count"]))
    self.assertTrue(jnp.allclose(restored["model"]["bn"]["mean"], saved["model"]["bn"]["mean"]))
    # mu/nu specifically: a fresh init leaves them all-zero, and the `count` scalar alone must not
    # be able to satisfy this. The first assert also fails loudly if the fixture stops training.
    for moment in ("mu", "nu"):
      saved_leaves = jax.tree.leaves(saved["optimizer"]["opt_state"][moment])
      restored_leaves = jax.tree.leaves(restored["optimizer"]["opt_state"][moment])
      self.assertTrue(any(float(jnp.abs(x).sum()) > 0 for x in saved_leaves), f"fixture: saved {moment} is zero")
      for want, got in zip(saved_leaves, restored_leaves):
        self.assertTrue(jnp.allclose(want, got))

  def test_no_checkpoint_returns_fresh_init(self):
    """With nothing on disk, setup_initial_state returns a plain init state."""
    config = _config()
    manager = self._manager()  # nothing saved
    init_kernel = nnx.state(_init_fn(_Model, 7)()).to_pure_dict()["model"]["linear"]["kernel"]
    state = self._setup(manager, _init_fn(_Model, 7), config)
    self.assertTrue(jnp.allclose(state["model"]["linear"]["kernel"], init_kernel))

  def test_old_checkpoint_without_nnx_aux_keeps_init_value(self):
    """A checkpoint carrying every weight but no nnx_aux keeps the rng at its init, not key(0)."""
    config = _config()
    manager = self._manager()
    saved = self._train_and_save(manager, _PlainModel, config, seed=0)  # linear + optimizer, no nnx_aux
    init = nnx.state(_init_fn(_RngModel, 999)()).to_pure_dict()
    restored = self._setup(manager, _init_fn(_RngModel, 999), config)
    # The rng was absent from the ckpt -> kept at its init.
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), int(init["model"]["dropout"]["rngs"]["count"]))
    # The linear weights really came from the ckpt, not from init -- compare values, not shapes.
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], saved["model"]["linear"]["kernel"]))
    self.assertFalse(jnp.allclose(restored["model"]["linear"]["kernel"], init["model"]["linear"]["kernel"]))

  def test_missing_weight_raises_naming_the_path(self):
    """A weight the checkpoint doesn't carry is an error, not a silent untrained init value."""
    config = _config()
    manager = self._manager()
    self._train_and_save(manager, _PlainModel, config, seed=0)  # no batch-norm weights on disk
    with self.assertRaises(ValueError) as ctx:
      self._setup(manager, _init_fn(_Model, 999), config)  # _Model's bn scale/bias are nnx.Param
    self.assertIn("bn/scale", str(ctx.exception))

  def test_emergency_manager_overlay_is_not_unwrapped(self):
    """An emergency manager hands back the overlay directly, not under an `items` key."""
    config = _config()
    trained = _init_fn(_Model, 0)()
    grads = nnx.grad(lambda m: jnp.mean(m(_TRAIN_X, deterministic=False) ** 2))(trained.model)
    trained.apply_gradients(grads)
    overlay = nnx.state(trained)  # what an emergency restore returns: the NNX state, unwrapped
    manager = mock.Mock(spec=emergency_checkpoint_manager.CheckpointManager)

    with mock.patch.object(checkpointing, "load_state_if_possible", return_value=(overlay, None)):
      restored = self._setup(manager, _init_fn(_Model, 999), config)  # a DIFFERENT init seed

    # Indexing `restored["items"]` on an nnx.State would raise, so reaching here proves the
    # emergency side ran; the values confirm the overlay, not the init, won.
    saved = overlay.to_pure_dict()
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], saved["model"]["linear"]["kernel"]))
    self.assertEqual(int(restored["optimizer"]["step"]), int(saved["optimizer"]["step"]))

  def test_params_only_load_overlays_weights_and_keeps_init(self):
    """A params-only load (load_parameters_path) writes the weights and leaves everything else at init."""
    config = _config()
    _, weights, _ = nnx.split(_init_fn(_Model, 0)().model, nnx.Param, ...)
    init = nnx.state(_init_fn(_Model, 999)()).to_pure_dict()

    with mock.patch.object(checkpointing, "load_state_if_possible", return_value=(None, weights)):
      restored = self._setup(self._manager(), _init_fn(_Model, 999), config)

    # Weights come from the params load...
    want = weights.to_pure_dict()
    self.assertTrue(jnp.allclose(restored["model"]["linear"]["kernel"], want["linear"]["kernel"]))
    self.assertFalse(jnp.allclose(restored["model"]["linear"]["kernel"], init["model"]["linear"]["kernel"]))
    # ...while the optimizer and rng stream stay at their fresh init.
    self.assertEqual(int(restored["optimizer"]["step"]), int(init["optimizer"]["step"]))
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), int(init["model"]["dropout"]["rngs"]["count"]))


if __name__ == "__main__":
  unittest.main()
