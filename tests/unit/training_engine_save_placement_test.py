# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that save_checkpoint settles uncompiled optimizer state onto the mesh.

Verifies that `MaxTextTrainingEngine.save_checkpoint` places all optimizer leaves
(particularly 0-D scalar leaves like `step` and `count`) onto the full device mesh
both before any compile (`self._state is None`) and when uncompiled state already
exists (`self._state is not None and not self._compiled`).
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

# Force 4 CPU devices before JAX initialization if not already configured.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.training_engine import checkpointing
from maxtext.training_engine.maxtext_engine import MaxTextTrainingEngine
from tests.utils.test_helpers import get_test_config_path
import numpy as np


def _create_engine(
    ckpt_dir: str, *, shard_optimizer_over_data: bool = False
) -> tuple[MaxTextTrainingEngine, jax.sharding.Mesh]:
  devices = jax.devices()
  extra_cfg_kwargs = {}
  if len(devices) >= 4:
    if shard_optimizer_over_data:
      mesh_devices = np.array(devices[:4]).reshape((4, 1))
      extra_cfg_kwargs = {"ici_data_parallelism": 4, "ici_fsdp_parallelism": 1, "shard_mode": "explicit"}
      mesh = jax.sharding.Mesh(
          mesh_devices,
          ("data", "fsdp"),
          axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
      )
    else:
      mesh_devices = np.array(devices[:4]).reshape((2, 2))
      mesh = jax.sharding.Mesh(mesh_devices, ("data", "fsdp"))
  else:
    mesh_devices = np.array(devices)
    if shard_optimizer_over_data:
      extra_cfg_kwargs = {"ici_data_parallelism": len(devices), "ici_fsdp_parallelism": 1, "shard_mode": "explicit"}
      mesh = jax.sharding.Mesh(mesh_devices, ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    else:
      mesh = jax.sharding.Mesh(mesh_devices, ("data",))

  cfg = pyconfig.initialize(
      [None, get_test_config_path("base.yml")],
      run_name="save_placement_test",
      model_name="default",
      override_model_config=True,
      base_emb_dim=16,
      base_mlp_dim=32,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      head_dim=8,
      base_num_decoder_layers=1,
      vocab_size=128,
      max_target_length=16,
      per_device_batch_size=1.0,
      enable_checkpointing=True,
      async_checkpointing=False,
      checkpoint_period=1,
      checkpoint_dir=ckpt_dir,
      convert_checkpoint_if_possible=False,
      load_parameters_path="",
      shard_optimizer_over_data=shard_optimizer_over_data,
      **extra_cfg_kwargs,
  )
  engine = MaxTextTrainingEngine(training_config=cfg, mesh=mesh)
  return engine, mesh


class TrainingEngineSavePlacementTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)

  def test_save_before_compile_places_optimizer_state_on_mesh(self):
    engine, mesh = _create_engine(self.ckpt_dir)
    self.assertIsNone(engine._state)
    self.assertFalse(engine._compiled)

    with mock.patch.object(
        engine._checkpoint_manager, "save_checkpoint", wraps=engine._checkpoint_manager.save_checkpoint
    ) as mock_save:
      engine.save_checkpoint(metadata={"step": 0})

    mock_save.assert_called_once()
    saved_state = mock_save.call_args.kwargs["checkpoint_state"]
    opt_from_saved = nnx.state(saved_state.optimizer, nnx.optimizer.OptState)
    opt_from_engine = nnx.state(engine._optimizer, nnx.optimizer.OptState)

    expected_device_set = set(mesh.devices.flat)
    checked_leaves = 0
    for tree in (opt_from_engine, opt_from_saved):
      for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "sharding") and hasattr(leaf.sharding, "device_set"):
          self.assertEqual(leaf.sharding.device_set, expected_device_set)
          checked_leaves += 1

    # Load-bearing check: adamw step, count, and moment trees mu/nu across parameters.
    self.assertGreaterEqual(checked_leaves, 10)

  def test_save_uncompiled_state_with_single_device_leaf_places_on_mesh(self):
    engine, mesh = _create_engine(self.ckpt_dir, shard_optimizer_over_data=True)
    # Instantiate train state so self._state is not None while remaining uncompiled.
    self.assertIsNotNone(engine.state)
    self.assertIsNotNone(engine._state)
    self.assertFalse(engine._compiled)

    # Simulate an uncompiled state where a scalar leaf carries a SingleDeviceSharding.
    single_device = list(mesh.devices.flat)[0]
    single_device_sharding = jax.sharding.SingleDeviceSharding(single_device)
    engine._optimizer.step.value = jax.device_put(jnp.array(42, dtype=jnp.int32), single_device_sharding)
    if len(mesh.devices.flat) > 1:
      self.assertEqual(engine._optimizer.step.value.sharding.device_set, {single_device})

    with mock.patch.object(
        engine._checkpoint_manager, "save_checkpoint", wraps=engine._checkpoint_manager.save_checkpoint
    ) as mock_save:
      engine.save_checkpoint(metadata={"step": 1})

    mock_save.assert_called_once()
    saved_state = mock_save.call_args.kwargs["checkpoint_state"]
    opt_from_saved = nnx.state(saved_state.optimizer, nnx.optimizer.OptState)
    opt_from_engine = nnx.state(engine._optimizer, nnx.optimizer.OptState)

    expected_device_set = set(mesh.devices.flat)
    checked_leaves = 0
    data_sharded_leaves = 0
    for tree in (opt_from_engine, opt_from_saved):
      for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "sharding") and hasattr(leaf.sharding, "device_set"):
          self.assertEqual(leaf.sharding.device_set, expected_device_set)
          checked_leaves += 1
          if leaf.ndim > 0 and hasattr(leaf.sharding, "spec"):
            if any(axis == "data" or (isinstance(axis, tuple) and "data" in axis) for axis in leaf.sharding.spec):
              data_sharded_leaves += 1

    self.assertGreaterEqual(checked_leaves, 10)
    if mesh.shape.get("data", 1) > 1:
      self.assertGreater(data_sharded_leaves, 0)
    self.assertEqual(int(engine._optimizer.step.value), 42)

  def test_colocated_python_async_save_stages_params_to_pinned_host(self):
    engine, mesh = _create_engine(self.ckpt_dir)
    # Inject dummy PRNG key into model state to verify PRNG key leaves are not staged.
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    engine.model._dummy_key = jax.device_put(nnx.Rngs(42).params(), replicated_sharding)

    saved_items = {}
    mock_inner_mgr = mock.MagicMock()

    def fake_save(step, args, custom_metadata=None, **kwargs):
      for k, v in args._items.items():
        saved_items[k] = v.item
      return True

    mock_inner_mgr.save.side_effect = fake_save
    mock_inner_mgr.should_save.return_value = True

    mgr = engine._checkpoint_manager
    mgr._checkpoint_manager = mock_inner_mgr
    mgr._async_checkpointing = True

    with mock.patch.object(checkpointing, "_REGISTERED_IMPL", checkpointing._COLOCATED_PYTHON):
      success = mgr.save_checkpoint(
          step=1,
          checkpoint_state=checkpointing.CheckpointState(model=engine.model),
      )

    self.assertTrue(success)
    self.assertIn("model_params", saved_items)
    staged_params = saved_items["model_params"]

    # Verify model_params arrays are staged to pinned_host while PRNG keys are not.
    staged_array_leaves = 0
    key_leaves = 0
    for leaf in jax.tree.leaves(staged_params):
      if isinstance(leaf, jax.Array):
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
          key_leaves += 1
          self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "device")
        else:
          staged_array_leaves += 1
          self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "pinned_host")

    self.assertGreaterEqual(staged_array_leaves, 5)
    self.assertGreaterEqual(key_leaves, 1)

    # Verify checkpoint_state.model's live leaves remain on device.
    live_device_leaves = 0
    for leaf in jax.tree.leaves(nnx.state(engine.model)):
      if isinstance(leaf, jax.Array):
        live_device_leaves += 1
        self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "device")
    self.assertGreaterEqual(live_device_leaves, staged_array_leaves)

  def test_colocated_python_sync_or_other_impl_does_not_stage_to_pinned_host(self):
    engine, mesh = _create_engine(self.ckpt_dir)
    saved_items = {}
    mock_inner_mgr = mock.MagicMock()

    def fake_save(step, args, custom_metadata=None, **kwargs):
      for k, v in args._items.items():
        saved_items[k] = v.item
      return True

    mock_inner_mgr.save.side_effect = fake_save
    mock_inner_mgr.should_save.return_value = True

    mgr = engine._checkpoint_manager
    mgr._checkpoint_manager = mock_inner_mgr

    # Case 1: colocated_python but async_checkpointing = False
    mgr._async_checkpointing = False
    with mock.patch.object(checkpointing, "_REGISTERED_IMPL", checkpointing._COLOCATED_PYTHON):
      mgr.save_checkpoint(
          step=1,
          checkpoint_state=checkpointing.CheckpointState(model=engine.model),
      )

    for leaf in jax.tree.leaves(saved_items["model_params"]):
      if isinstance(leaf, jax.Array):
        self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "device")

    # Case 2: async_checkpointing = True but _REGISTERED_IMPL != colocated_python
    saved_items.clear()
    mgr._async_checkpointing = True
    with mock.patch.object(checkpointing, "_REGISTERED_IMPL", checkpointing._PERSISTENCE):
      mgr.save_checkpoint(
          step=2,
          checkpoint_state=checkpointing.CheckpointState(model=engine.model),
      )

    for leaf in jax.tree.leaves(saved_items["model_params"]):
      if isinstance(leaf, jax.Array):
        self.assertEqual(getattr(leaf.sharding, "memory_kind", None), "device")


if __name__ == "__main__":
  unittest.main()
