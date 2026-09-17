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

"""Unit tests for how `pre_train.train.recover` restores the train state after a slice failure.

Pathways, mesh setup and compilation are mocked out. Persistent checkpoints are real Orbax checkpoints of a tiny NNX
TrainState, saved and restored through MaxText's checkpointing code.
"""

import threading
import types
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.trainers.pre_train import train as pre_train
import numpy as np
import optax


def _make_state(seed, step):
  model = nnx.Linear(2, 3, rngs=nnx.Rngs(seed))
  optimizer = nnx.Optimizer(model, optax.adamw(0.1), wrt=nnx.Param)
  optimizer.step.set_value(jnp.asarray(step, dtype=jnp.uint32))
  return train_state_nnx.TrainStateNNX(model, optimizer)


def _kernel(state):
  return np.asarray(nnx.to_pure_dict(nnx.state(state))["model"]["kernel"])


class _FakeSnapshotter:
  """Mirrors the pathways Snapshotter attributes that recover() uses."""

  loaded_steps = []

  def __init__(self, *, replica_axis_index=0):
    self.replica_axis_index = replica_axis_index
    self._lock = threading.Lock()
    self._latest_snapshot = None  # (pure state dict or the exception load() raises, step)

  @property
  def latest(self):
    if self._latest_snapshot is None:
      return None
    return types.SimpleNamespace(step=self._latest_snapshot[1])

  def load(self, abstract_state):
    del abstract_state
    snapshot, step = self._latest_snapshot
    if isinstance(snapshot, Exception):
      raise snapshot
    _FakeSnapshotter.loaded_steps.append(step)
    return snapshot


class RecoverTest(absltest.TestCase):
  """Tests for the state recover() restores from the in-memory snapshot or the persistent checkpoint."""

  def setUp(self):
    """Mocks slice discovery, train loop setup and recompilation around recover()'s restore logic."""
    super().setUp()
    _FakeSnapshotter.loaded_steps = []
    self.fresh_state = _make_state(seed=0, step=0)
    self.checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        self.create_tempdir().full_path, enable_checkpointing=True, use_async=False, save_interval_steps=1
    )
    mesh = jax.make_mesh((1,), ("data",))
    setup_results = (
        jax.random.PRNGKey(0),  # init_rng
        self.checkpoint_manager,
        None,  # state_mesh_shardings
        object(),  # NNX model graphdef stand-in
        mesh,
        None,  # learning_rate_schedule
        None,
        None,
        None,  # rampup_manager
        None,  # eval_data_iterator
        self.fresh_state,
    )
    self.enter_context(mock.patch.object(jax.config, "update"))
    self.wait_for_slices = self.enter_context(
        mock.patch.object(pre_train.elastic, "wait_for_slices", return_value=frozenset({0}))
    )
    for name in ("record_elastic_wait_end_and_reinit_start", "mutate_config_for_topology", "record_elastic_reinit_end"):
      self.enter_context(mock.patch.object(pre_train.elastic_utils, name))
    self.enter_context(mock.patch.object(pre_train.elastic_utils, "live_devices", return_value=jax.devices()[:1]))
    self.enter_context(mock.patch.object(pre_train, "Snapshotter", _FakeSnapshotter))
    self.enter_context(mock.patch.object(pre_train.train_utils, "setup_train_loop", return_value=setup_results))
    self.enter_context(
        mock.patch.object(pre_train.sharding, "maybe_update_params_sharding_with_opt", return_value=(None, None))
    )
    self.enter_context(mock.patch.object(pre_train.train_utils, "jit_train_and_eval_step", return_value=(None, None)))
    self.enter_context(mock.patch.object(pre_train, "recreate_dataloaders", return_value=(None, None, None)))

    # Real checkpoint restores, except for the errors a test queues up for the first calls.
    self.load_errors = []
    load_state_if_possible = checkpointing.load_state_if_possible

    def _load(*args, **kwargs):
      if self.load_errors:
        raise self.load_errors.pop(0)
      return load_state_if_possible(*args, **kwargs)

    self.load_checkpoint = self.enter_context(
        mock.patch.object(pre_train.checkpointing, "load_state_if_possible", side_effect=_load)
    )

  def _save_checkpoint(self, step, state):
    """Saves `state` as a persistent checkpoint the way the train loop does."""
    save_config = types.SimpleNamespace(
        pure_nnx=True,
        enable_diloco=False,
        enable_checkpointing=True,
        checkpoint_period=20,
        enable_continuous_checkpointing=False,
        enable_emergency_checkpoint=False,
        enable_multi_tier_checkpointing=False,
        enable_autocheckpoint=False,
        dataset_type=None,
    )
    checkpointing.save_checkpoint(self.checkpoint_manager, step, state, save_config, force=True)

  def _recover(self, snapshot, snapshot_step, load_errors=()):
    """Runs recover() with the given latest snapshot; returns (resume step, restored state)."""
    snapshotter = _FakeSnapshotter()
    if snapshot is not None:
      if isinstance(snapshot, train_state_nnx.TrainStateNNX):
        snapshot = {
            "model": nnx.to_pure_dict(nnx.state(snapshot.model)),
            "optimizer": nnx.to_pure_dict(nnx.state(snapshot.optimizer)),
        }
      snapshotter._latest_snapshot = (snapshot, snapshot_step)  # pylint: disable=protected-access
    self.load_errors.extend(load_errors)

    config = types.SimpleNamespace(
        elastic_min_slice_count=1,
        num_slices=1,
        elastic_timeout_seconds=1,
        load_parameters_path="",
        load_full_state_path="",
        checkpoint_storage_concurrent_gb=1,
        enable_single_replica_ckpt_restoring=False,
        dataset_type=None,
        checkpoint_storage_use_ocdbt=True,
        checkpoint_storage_use_zarr3=True,
        enable_orbax_v1=False,
        checkpoint_conversion_fn=None,
        source_checkpoint_layout="orbax",
        expansion_factor_real_data=-1,
        logical_axis_rules=(),
    )
    python_vars = {
        "recorder": None,
        "elastic_manager": types.SimpleNamespace(
            slice_to_devices={}, default_device=None, active_slice_indices=frozenset({0})
        ),
        "snapshot": snapshotter,
        "rampup_manager": None,
        "metric_logger_instance": None,
        "checkpoint_manager": self.checkpoint_manager,
    }
    jax_device_state = {}
    pre_train.recover(jax_device_state, python_vars, {"config": config})
    return python_vars["step"], jax_device_state["state"]

  def test_restores_snapshot(self):
    snapshot = _make_state(seed=1, step=1816)
    self._save_checkpoint(1800, _make_state(seed=2, step=1801))

    step, state = self._recover(snapshot, 1815)

    self.load_checkpoint.assert_not_called()
    self.assertEqual(_FakeSnapshotter.loaded_steps, [1815])
    self.assertEqual(step, 1815)
    np.testing.assert_array_equal(_kernel(state), _kernel(snapshot))

  def test_restores_checkpoint_when_snapshot_fails(self):
    checkpoint = _make_state(seed=2, step=1801)
    self._save_checkpoint(1800, checkpoint)

    step, state = self._recover(RuntimeError("No active replicas found."), 1815)

    self.load_checkpoint.assert_called_once()
    self.assertEqual(step, 1801)
    np.testing.assert_array_equal(_kernel(state), _kernel(checkpoint))


if __name__ == "__main__":
  absltest.main()
