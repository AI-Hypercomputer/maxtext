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


if __name__ == "__main__":
  unittest.main()
