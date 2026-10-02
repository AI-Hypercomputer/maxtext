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

"""Unit tests for pre-dispatch uniform device-set assertion in training_engine/checkpointing.py."""

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
from tests.utils.test_helpers import get_test_config_path
import numpy as np
import optax


def _make_mesh_and_shardings():
  devices = jax.devices()
  assert len(devices) >= 2, f"Expected >= 2 CPU devices, got {len(devices)}"
  mesh = jax.sharding.Mesh(np.array(devices), ("data",))
  full_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  single_sharding = jax.sharding.SingleDeviceSharding(devices[0])
  return mesh, full_sharding, single_sharding


class _TinyModel(nnx.Module):

  def __init__(self, full_sharding: jax.sharding.Sharding):
    self.w1 = nnx.Param(jax.device_put(jnp.ones((4, 4), dtype=jnp.float32), full_sharding))
    self.w2 = nnx.Param(jax.device_put(jnp.zeros((4,), dtype=jnp.float32), full_sharding))


class TrainingEngineDeviceSetAssertTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.ckpt_dir, ignore_errors=True)
    self._orig_impl = checkpointing._REGISTERED_IMPL
    self.addCleanup(self._restore_impl)

  def _restore_impl(self):
    checkpointing._REGISTERED_IMPL = self._orig_impl

  def test_assert_uniform_device_set_raises_with_offending_keypath(self):
    _, full_sharding, single_sharding = _make_mesh_and_shardings()
    mixed_tree = {
        "params": {
            "w1": jax.device_put(jnp.ones((4, 4)), full_sharding),
            "w2": jax.device_put(jnp.ones((4, 4)), full_sharding),
            "w3": jax.device_put(jnp.ones((4, 4)), full_sharding),
        },
        "opt_state": {
            "count": jax.device_put(jnp.array(0, dtype=jnp.int32), single_sharding),
        },
    }

    with self.assertRaises(checkpointing.PathwaysCheckpointingUnavailableError) as ctx:
      checkpointing._assert_uniform_device_set(mixed_tree, item="optimizer_state")

    msg = str(ctx.exception)
    self.assertIn("optimizer_state", msg)
    self.assertIn("opt_state", msg)
    self.assertIn("count", msg)

  def test_assert_uniform_device_set_passes_when_uniform(self):
    _, full_sharding, _ = _make_mesh_and_shardings()
    uniform_tree = {
        "w1": jax.device_put(jnp.ones((4, 4)), full_sharding),
        "count": jax.device_put(jnp.array(0, dtype=jnp.int32), full_sharding),
    }
    checkpointing._assert_uniform_device_set(uniform_tree, item="optimizer_state")

  def test_save_checkpoint_gates_assertion_on_colocated_python(self):
    _, full_sharding, single_sharding = _make_mesh_and_shardings()
    cfg = pyconfig.initialize(
        [None, get_test_config_path("base.yml")],
        run_name="device_set_assert_test",
        enable_checkpointing=True,
        async_checkpointing=False,
        checkpoint_period=1,
        checkpoint_dir=self.ckpt_dir,
    )
    mgr = checkpointing.CheckpointManager(self.ckpt_dir, cfg)
    model = _TinyModel(full_sharding)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    # Place all optimizer leaves onto full_sharding, except `step` on single_sharding.
    opt_state = nnx.state(optimizer, nnx.optimizer.OptState)
    placed_opt_state = jax.tree.map(lambda x: jax.device_put(x, full_sharding), opt_state)
    nnx.update(optimizer, placed_opt_state)
    optimizer.step.value = jax.device_put(jnp.array(1, dtype=jnp.int32), single_sharding)

    state = checkpointing.CheckpointState(model=model, optimizer=optimizer)

    # 1. Under _COLOCATED_PYTHON: save_checkpoint fails fast before dispatching to Orbax.
    checkpointing._REGISTERED_IMPL = checkpointing._COLOCATED_PYTHON
    with mock.patch.object(mgr._checkpoint_manager, "save") as mock_ocp_save:
      with self.assertRaises(checkpointing.PathwaysCheckpointingUnavailableError) as ctx:
        mgr.save_checkpoint(step=1, checkpoint_state=state)
      mock_ocp_save.assert_not_called()
    self.assertIn("optimizer_state", str(ctx.exception))
    self.assertIn("step", str(ctx.exception))

    # 2. Under _PERSISTENCE: pre-dispatch assertion is gated off.
    checkpointing._REGISTERED_IMPL = checkpointing._PERSISTENCE
    with mock.patch.object(mgr._checkpoint_manager, "save", return_value=True) as mock_ocp_save:
      saved = mgr.save_checkpoint(step=1, checkpoint_state=state)
      self.assertTrue(saved)
      mock_ocp_save.assert_called_once()


if __name__ == "__main__":
  unittest.main()
