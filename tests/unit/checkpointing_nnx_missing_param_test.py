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

"""End-to-end coverage of the missing-weight check on NNX restore.

Drives the real create_orbax_checkpoint_manager -> maybe_save_checkpoint ->
load_state_if_possible stack (no mocks). `partial_restore` returns a weight the checkpoint
lacks as an unmaterialized ShapeDtypeStruct; restore raises naming it rather than training on
an untrained init value. The check filters by nnx.Param, so rngs/dropout absence never trips it.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from flax import nnx
import jax
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
import optax


class _Model(nnx.Module):
  """Linear + dropout, so the state carries rngs that split out into nnx_aux."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


class _ModelExtraLayer(nnx.Module):
  """`_Model` plus a layer absent from a `_Model` checkpoint, i.e. a weight with no array on disk."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    self.extra = nnx.Linear(1, 1, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.extra(self.dropout(self.linear(x), deterministic=deterministic))


class _WiderModel(nnx.Module):
  """`_Model` with a wider linear -- every weight is present but at a different shape."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 4, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


_TX = optax.adam(1e-3)


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
  """An abstract (ShapeDtypeStruct) nnx.State for `model_cls`, the restore blueprint."""
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


class TestMissingParamRoundTrip(unittest.TestCase):
  """Real save->restore covering the missing-weight policy end to end."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  def _save_model(self):
    """Saves a one-step `_Model` checkpoint and returns its manager."""
    manager = checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )
    model = _Model(nnx.Rngs(0))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), _config(), data_iterator=None, step=1)
    manager.wait_until_finished()
    return manager

  def _load_overlay(self, manager, model_cls):
    """Returns the restore overlay load_state_if_possible produces (leaves not on disk kept as placeholders)."""
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
    return full["items"].to_pure_dict()

  def test_missing_weight_raises_naming_the_path(self):
    """A full-state resume against a model with an extra layer raises, naming that layer."""
    manager = self._save_model()
    with self.assertRaises(ValueError) as ctx:
      self._load_overlay(manager, _ModelExtraLayer)
    msg = str(ctx.exception)
    self.assertIn("extra", msg)  # the exact offending parameter path
    self.assertIn("missing", msg)

  def test_params_only_load_missing_weight_raises(self):
    """The params-only route (load_parameters_path, e.g. SFT) checks too, instead of failing later.

    Without the check the weight comes back as a ShapeDtypeStruct, reaches the model, and fails
    deep in the first step with a TypeError that never names it.
    """
    self._save_model()  # a `_Model` checkpoint: linear only, no `extra`
    items = os.path.join(self._dir, "ckpt", "1", "items")
    # The real caller hands over an abstract params state, as load_state_if_possible does.
    params_abstract = nnx.eval_shape(lambda: nnx.split(_ModelExtraLayer(nnx.Rngs(0)), nnx.Param, ...)[1])
    with self.assertRaises(ValueError) as ctx:
      checkpointing.load_params_from_path(items, params_abstract, 8)
    self.assertIn("extra", str(ctx.exception))
    self.assertIn("missing", str(ctx.exception))

  def test_params_only_load_shape_mismatch_raises(self):
    """Orbax restores a stored array at its own shape, so a wider model must be caught here.

    Without the check the model silently receives the checkpoint's shape and fails later on a
    broadcast, or trains on the wrong weights if the shapes happen to broadcast.
    """
    self._save_model()  # `_Model` has linear (2, 1)
    items = os.path.join(self._dir, "ckpt", "1", "items")
    params_abstract = nnx.eval_shape(lambda: nnx.split(_WiderModel(nnx.Rngs(0)), nnx.Param, ...)[1])
    with self.assertRaises(ValueError) as ctx:
      checkpointing.load_params_from_path(items, params_abstract, 8)
    msg = str(ctx.exception)
    self.assertIn("linear/kernel", msg)
    self.assertIn("(2, 1)", msg)  # what the checkpoint had
    self.assertIn("(2, 4)", msg)  # what the model expects

  def test_matching_config_restores_clean(self):
    """Negative control: an exact-arch restore doesn't raise and the overlay carries the weight."""
    manager = self._save_model()
    overlay = self._load_overlay(manager, _Model)  # must not raise the missing-weight error
    self.assertEqual(overlay["model"]["linear"]["kernel"].shape, (2, 1))
    self.assertNotIsInstance(overlay["model"]["linear"]["kernel"], jax.ShapeDtypeStruct)


if __name__ == "__main__":
  unittest.main()
