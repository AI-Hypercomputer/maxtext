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
load_state_if_possible against a real on-disk Orbax checkpoint (no mocks), which
is what exercises the `items/nnx_aux` subtree and the RNG/dropout persistence the
way training actually hits them, including the fallback for a checkpoint that has
no nnx_aux (a Linen-trained or pre-persistence checkpoint).
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
  """Linear + dropout, so the state carries rngs/dropout that go under nnx_aux on save."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x, deterministic=False):
    return self.dropout(self.linear(x), deterministic=deterministic)


class _PlainModel(nnx.Module):
  """Linear only (no dropout), so its checkpoint carries no nnx_aux, like a Linen-trained one."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


_TX = optax.adam(1e-3)


def _config():
  """Minimal config with the fields save/restore reads for a pure_nnx run."""
  return SimpleNamespace(
      pure_nnx=True,
      enable_diloco=False,
      enable_checkpointing=True,
      enable_continuous_checkpointing=False,
      enable_emergency_checkpoint=False,
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


class TestNNXCheckpointRoundTrip(unittest.TestCase):
  """Real save->restore cycles covering rng persistence and fallback."""

  def setUp(self):
    self._dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)

  def _save_step(self):
    """Trains one step (advances step + dropout rng) and saves; returns (manager, orig_count)."""
    manager = checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )
    model = _Model(nnx.Rngs(0))
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2)), deterministic=False) ** 2))(state.model)
    state.apply_gradients(grads)
    orig_count = int(nnx.state(state).to_pure_dict()["model"]["dropout"]["rngs"]["count"])
    self.assertGreater(orig_count, 0)  # dropout actually advanced the stream
    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), _config(), data_iterator=None, step=1)
    manager.wait_until_finished()
    return manager, orig_count

  def _restore(self, manager, model_cls):
    """Restores the saved checkpoint into an abstract state for `model_cls`."""
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
    return full["items"]

  def test_rng_dropout_state_survives_round_trip(self):
    """The dropout rng stream continues across a save/restore instead of resetting to 0."""
    manager, orig_count = self._save_step()
    restored = self._restore(manager, _Model)
    # Recovered from items/nnx_aux; a lost stream would default back to 0.
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), orig_count)
    self.assertEqual(int(restored["optimizer"]["step"]), 1)

  def test_old_checkpoint_without_nnx_aux_still_loads(self):
    """A checkpoint with no nnx_aux (Linen-trained / pre-persistence) loads; rng gets the base default."""
    manager = checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._dir, "ckpt"),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )
    model = _PlainModel(nnx.Rngs(0))  # no dropout -> to_checkpoint_dict writes no nnx_aux
    state = train_state_nnx.TrainStateNNX(model, nnx.Optimizer(model, _TX, wrt=nnx.Param))
    checkpointing.maybe_save_checkpoint(manager, nnx.state(state), _config(), data_iterator=None, step=1)
    manager.wait_until_finished()
    # Restore into an abstract that DOES have dropout: the absent rng falls back to the base default.
    restored = self._restore(manager, _Model)
    self.assertEqual(int(restored["model"]["dropout"]["rngs"]["count"]), 0)


if __name__ == "__main__":
  unittest.main()
