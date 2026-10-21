# Copyright 2023–2026 Google LLC
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

"""Unit tests for the in-memory elastic snapshot core."""

import functools
import threading
from unittest import mock

from absl.testing import absltest
from grain import experimental as grain_experimental
import jax
import jax.numpy as jnp
from maxtext.common import checkpointing
from maxtext.common import snapshotting
from maxtext.common import train_state_nnx
from maxtext.input_pipeline import multihost_dataloading
from maxtext.utils import elastic_utils
import numpy as np
import pathwaysutils


class FakeConfig:
  """Fake configuration object."""

  def __init__(self):
    self.elastic_enabled = True
    self.elastic_backup_kind = "snapshot"
    self.elastic_snapshot_interval = 10
    self.mesh_axes = ["diloco", "data", "fsdp"]
    self.shard_optimizer_over_data = False


class FakeElasticIterator(grain_experimental.ElasticIterator):
  """An `ElasticIterator` with just the state API, without a dataset behind it."""

  def __init__(self, index=0):  # pylint: disable=super-init-not-called
    self.state = {"global_next_index": index}

  def get_state(self):
    return self.state

  def set_state(self, state):
    self.state = state


def fake_data_iterator(local_iterator):
  """A `MultiHostDataLoadIterator` wrapping `local_iterator`, without running its constructor."""
  data_iterator = multihost_dataloading.MultiHostDataLoadIterator.__new__(multihost_dataloading.MultiHostDataLoadIterator)
  data_iterator.local_iterator = local_iterator
  return data_iterator


def fake_checkpointer(latest_step=None, saving=False):
  checkpointer = mock.Mock(spec=checkpointing.ocp.training.Checkpointer)
  if latest_step is None:
    checkpointer.latest = None
  else:
    checkpointer.latest = mock.Mock(step=latest_step)
  checkpointer.is_saving_in_progress.return_value = saving
  return checkpointer


class FakeSnapshotter:
  """Stands in for the Orbax Snapshotter, which needs Pathways."""

  def __init__(self, *, replica_axis_index=0, latest_snapshot=None):
    self.replica_axis_index = replica_axis_index
    self._lock = threading.Lock()
    self._latest_snapshot = latest_snapshot
    self.save = mock.Mock()
    self.join = mock.Mock()
    self.load = mock.Mock()

  @property
  def latest(self):
    if self._latest_snapshot is None:
      return None
    return mock.Mock(step=self._latest_snapshot[1])


class SnapshottingTest(absltest.TestCase):
  """Unit tests for the in-memory elastic snapshot core."""

  def setUp(self):
    super().setUp()
    self.config = FakeConfig()
    self.enter_context(mock.patch.object(pathwaysutils, "is_pathways_backend_used", return_value=True))
    self.real_snapshotter_class = snapshotting.Snapshotter
    self.enter_context(mock.patch.object(snapshotting, "Snapshotter", FakeSnapshotter))
    self.fake_manager = mock.Mock()
    self.fake_manager.available_inactive_slices = frozenset()
    self.enter_context(mock.patch.object(elastic_utils, "elastic_manager", self.fake_manager))

  def test_create_snapshotter_is_none_outside_snapshot_mode(self):
    self.config.elastic_backup_kind = "checkpoint"
    self.assertIsNone(snapshotting.create_snapshotter(self.config))

  def test_create_snapshotter_is_none_without_elastic(self):
    self.config.elastic_enabled = False
    self.assertIsNone(snapshotting.create_snapshotter(self.config))

  def test_create_snapshotter_splits_along_data_axis(self):
    snapshotter = snapshotting.create_snapshotter(self.config)
    self.assertIsInstance(snapshotter, FakeSnapshotter)
    self.assertEqual(snapshotter.replica_axis_index, 1)

  def test_create_snapshotter_makes_a_new_one_each_time(self):
    """The module keeps no state; the caller owns the snapshotter."""
    self.assertIsNot(snapshotting.create_snapshotter(self.config), snapshotting.create_snapshotter(self.config))

  def test_reset_snapshotter_keeps_latest_snapshot_and_empties_the_old_one(self):
    old_snapshotter = FakeSnapshotter(replica_axis_index=1, latest_snapshot=("pinned", 5))

    new_snapshotter = snapshotting.reset_snapshotter(old_snapshotter)

    self.assertIsNot(new_snapshotter, old_snapshotter)
    self.assertEqual(new_snapshotter.replica_axis_index, 1)
    self.assertEqual(new_snapshotter._latest_snapshot, ("pinned", 5))  # pylint: disable=protected-access
    # The old snapshotter outlives the call (its worker thread owns it) and must not keep the snapshot alive.
    self.assertIsNone(old_snapshotter._latest_snapshot)  # pylint: disable=protected-access
    self.assertIsNone(old_snapshotter.latest)

  def test_reset_snapshotter_without_snapshot(self):
    new_snapshotter = snapshotting.reset_snapshotter(FakeSnapshotter(replica_axis_index=2))
    self.assertEqual(new_snapshotter.replica_axis_index, 2)
    self.assertIsNone(new_snapshotter.latest)

  def test_reset_snapshotter_against_real_orbax_snapshotter(self):
    """The swap reaches into Orbax's private fields; this guards against them being renamed."""
    self.enter_context(mock.patch.object(snapshotting, "Snapshotter", self.real_snapshotter_class))
    old_snapshotter = self.real_snapshotter_class(replica_axis_index=1)
    old_snapshotter._latest_snapshot = ("pinned", 5)  # pylint: disable=protected-access

    new_snapshotter = snapshotting.reset_snapshotter(old_snapshotter)

    self.assertIsInstance(new_snapshotter, self.real_snapshotter_class)
    self.assertEqual(new_snapshotter.latest.step, 5)
    self.assertIsNone(old_snapshotter.latest)

  def test_latest_snapshot_step(self):
    self.assertIsNone(snapshotting.latest_snapshot_step(None))
    self.assertIsNone(snapshotting.latest_snapshot_step(FakeSnapshotter()))
    self.assertEqual(snapshotting.latest_snapshot_step(FakeSnapshotter(latest_snapshot=("pinned", 30))), 30)

  def test_wait_for_snapshot(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.assertTrue(snapshotting.wait_for_snapshot(snapshotter, 30))
    self.assertFalse(snapshotting.wait_for_snapshot(snapshotter, 31))  # The save failed or was skipped.
    self.assertEqual(snapshotter.join.call_count, 2)

  def _patch_checkpoint_dict(self):
    def fake_to_checkpoint_dict(state):
      return {"params": state}

    self.enter_context(mock.patch.object(train_state_nnx, "to_checkpoint_dict", side_effect=fake_to_checkpoint_dict))

  def test_save_snapshot_saves_checkpoint_layout_and_iterator_state_under_step(self):
    self._patch_checkpoint_dict()
    snapshotter = FakeSnapshotter()
    index = (3 << 32) + 5
    data_iterator = fake_data_iterator(FakeElasticIterator(index))

    snapshotting.save_snapshot(snapshotter, "state", 7, data_iterator)

    snapshotter.save.assert_called_once()
    step, saved = snapshotter.save.call_args.args
    self.assertEqual(step, 7)
    self.assertEqual(saved["items"], {"params": "state"})
    self.assertEqual(saved["data_iterator"], {"global_next_index": index})

  def test_save_snapshot_without_elastic_iterator_carries_no_position(self):
    self._patch_checkpoint_dict()
    snapshotter = FakeSnapshotter()

    snapshotting.save_snapshot(snapshotter, "state", 7, data_iterator=None)

    self.assertIsNone(snapshotter.save.call_args.args[1]["data_iterator"])

  def test_save_snapshot_with_synthetic_data_iterator_carries_no_position(self):
    self._patch_checkpoint_dict()
    snapshotter = FakeSnapshotter()
    synthetic_iterator = mock.create_autospec(snapshotting.synthetic_data_processing.SyntheticDataIterator, instance=True)

    snapshotting.save_snapshot(snapshotter, "state", 7, data_iterator=synthetic_iterator)

    self.assertIsNone(snapshotter.save.call_args.args[1]["data_iterator"])

  def test_save_snapshot_finds_elastic_iterator_on_dataloader_attribute(self):
    """In OSS grain 0.2.18, `ElasticIterator` is on `dataloader` while `local_iterator` is `_ElasticMapDatasetIterator`."""
    self._patch_checkpoint_dict()
    snapshotter = FakeSnapshotter()
    data_iterator = fake_data_iterator(local_iterator=object())
    data_iterator.dataloader = FakeElasticIterator(42)

    snapshotting.save_snapshot(snapshotter, "state", 7, data_iterator)

    self.assertEqual(snapshotter.save.call_args.args[1]["data_iterator"], {"global_next_index": 42})

  def test_save_snapshot_rejects_non_elastic_iterator(self):
    self._patch_checkpoint_dict()

    with self.assertRaisesRegex(ValueError, "ElasticIterator"):
      snapshotting.save_snapshot(FakeSnapshotter(), "state", 7, fake_data_iterator(object()))

    with self.assertRaisesRegex(ValueError, "ElasticIterator"):
      snapshotting.save_snapshot(FakeSnapshotter(), "state", 7, [fake_data_iterator(FakeElasticIterator())])

  def test_should_save_snapshot_on_interval_unless_checkpoint_saving(self):
    self.assertFalse(snapshotting.should_save_snapshot(self.config, 11, None))
    self.assertTrue(snapshotting.should_save_snapshot(self.config, 20, None))
    self.assertTrue(snapshotting.should_save_snapshot(self.config, 20, fake_checkpointer(saving=False)))
    self.assertFalse(snapshotting.should_save_snapshot(self.config, 20, fake_checkpointer(saving=True)))

  def test_maybe_save_snapshot_noop_without_snapshotter(self):
    save_snapshot = self.enter_context(mock.patch.object(snapshotting, "save_snapshot"))
    snapshotting.maybe_save_snapshot(None, self.config, "state", 10, None, None)
    save_snapshot.assert_not_called()

  def test_maybe_save_snapshot_saves_when_due(self):
    save_snapshot = self.enter_context(mock.patch.object(snapshotting, "save_snapshot"))
    snapshotter = FakeSnapshotter()

    snapshotting.maybe_save_snapshot(snapshotter, self.config, "state", 11, None, "iterator")
    save_snapshot.assert_not_called()
    snapshotting.maybe_save_snapshot(snapshotter, self.config, "state", 20, None, "iterator")
    save_snapshot.assert_called_once_with(snapshotter, "state", 20, "iterator")
    snapshotter.join.assert_not_called()  # Periodic saves never block the training loop.

  def test_maybe_save_snapshot_snapshots_and_waits_before_scale_up(self):
    self.fake_manager.available_inactive_slices = frozenset({1})
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 0))
    checkpointer = fake_checkpointer()
    calls = []

    def complete_save(step, _):
      calls.append(("save", step))
      snapshotter._latest_snapshot = ("pinned", step)  # pylint: disable=protected-access

    snapshotter.save.side_effect = complete_save
    snapshotter.join.side_effect = functools.partial(calls.append, "join")
    self._patch_checkpoint_dict()

    # Step 7 is not on the interval; the scale-up still gets a snapshot, after the checkpoint and earlier saves.
    snapshotting.maybe_save_snapshot(snapshotter, self.config, "state", 7, checkpointer, None)

    checkpointer.wait.assert_called_once()
    # Waited for earlier saves, saved this step, then waited for it.
    self.assertEqual(calls, ["join", ("save", 7), "join"])

  def test_maybe_save_snapshot_scale_up_proceeds_when_snapshot_does_not_complete(self):
    """A snapshot that fails must not stop the interrupt; the loss is logged, not fatal."""
    self.fake_manager.available_inactive_slices = frozenset({1})
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 0))  # `save` is a Mock: nothing ever completes.
    self._patch_checkpoint_dict()
    log = self.enter_context(mock.patch.object(snapshotting.max_logging, "log"))

    snapshotting.maybe_save_snapshot(snapshotter, self.config, "state", 7, None, None)

    self.assertEqual(snapshotter.save.call_args.args[0], 7)
    self.assertTrue(any("did not complete before the scale-up" in str(c.args[0]) for c in log.call_args_list))

  def test_should_restore_snapshot(self):
    self.assertFalse(snapshotting.should_restore_snapshot(None, None))
    self.assertFalse(snapshotting.should_restore_snapshot(FakeSnapshotter(), fake_checkpointer(latest_step=5)))
    with_snapshot = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.assertTrue(snapshotting.should_restore_snapshot(with_snapshot, None))
    self.assertTrue(snapshotting.should_restore_snapshot(with_snapshot, fake_checkpointer()))
    self.assertTrue(snapshotting.should_restore_snapshot(with_snapshot, fake_checkpointer(latest_step=20)))
    self.assertTrue(snapshotting.should_restore_snapshot(with_snapshot, fake_checkpointer(latest_step=30)))
    self.assertFalse(snapshotting.should_restore_snapshot(with_snapshot, fake_checkpointer(latest_step=40)))

  def test_replicated_single_device_arrays_restore_to_original_shardings(self):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ("data",))
    target = {
        "params": jax.device_put(jnp.ones(4), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())),
        "step": jax.device_put(jnp.zeros((), dtype=jnp.int32), jax.sharding.SingleDeviceSharding(device)),
        "rng": jax.device_put(jax.random.key(0), jax.sharding.SingleDeviceSharding(device)),
    }

    # pylint: disable=protected-access
    replicated = snapshotting._replicate_single_device_sharded_arrays(target)
    restored = snapshotting._restore_original_shardings(replicated, target)
    # pylint: enable=protected-access

    self.assertIsInstance(replicated["step"].sharding, jax.sharding.NamedSharding)
    self.assertIsInstance(replicated["rng"].sharding, jax.sharding.NamedSharding)
    self.assertIsInstance(restored["step"].sharding, jax.sharding.SingleDeviceSharding)
    self.assertIsInstance(restored["rng"].sharding, jax.sharding.SingleDeviceSharding)
    self.assertTrue(jax.dtypes.issubdtype(restored["rng"].dtype, jax.dtypes.prng_key))
    np.testing.assert_array_equal(jax.random.key_data(restored["rng"]), jax.random.key_data(target["rng"]))

  def test_replicate_single_device_arrays_handles_abstract_leaves(self):
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ("data",))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    target = {
        "params": jax.ShapeDtypeStruct((4,), jnp.float32, sharding=replicated_sharding),
        "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=jax.sharding.SingleDeviceSharding(device)),
    }

    replicated = snapshotting._replicate_single_device_sharded_arrays(target)  # pylint: disable=protected-access

    self.assertIsInstance(replicated["step"], jax.ShapeDtypeStruct)
    self.assertIsInstance(replicated["step"].sharding, jax.sharding.NamedSharding)
    self.assertEqual(replicated["step"].dtype, jnp.int32)

  def test_replicate_single_device_arrays_noop_without_mesh(self):
    device = jax.devices()[0]
    target = {"step": jax.device_put(jnp.zeros(()), jax.sharding.SingleDeviceSharding(device))}
    replicated = snapshotting._replicate_single_device_sharded_arrays(target)  # pylint: disable=protected-access
    self.assertIs(replicated, target)

  def test_load_snapshot_noop_without_snapshot(self):
    self.assertIsNone(snapshotting.load_snapshot(None, "abstract", self.config))
    self.assertIsNone(snapshotting.load_snapshot(FakeSnapshotter(), "abstract", self.config))

  def test_load_snapshot_restores_state_and_iterator_state(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    saved_state = {"global_next_index": (3 << 32) + 5}
    load_snapshot = self.enter_context(
        mock.patch.object(snapshotting, "_load_snapshot", return_value=("restored", saved_state))
    )
    iterator = FakeElasticIterator(0)

    restored = snapshotting.load_snapshot(snapshotter, "abstract", self.config, fake_data_iterator(iterator))

    self.assertEqual(restored, "restored")
    self.assertEqual(iterator.state, saved_state)
    load_snapshot.assert_called_once_with(snapshotter, "abstract", self.config, {"global_next_index": 0})

  def test_load_snapshot_restores_without_iterator_for_synthetic_data(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    load_snapshot = self.enter_context(mock.patch.object(snapshotting, "_load_snapshot", return_value=("restored", None)))

    self.assertEqual(snapshotting.load_snapshot(snapshotter, "abstract", self.config, None), "restored")

    load_snapshot.assert_called_once_with(snapshotter, "abstract", self.config, None)

  def test_load_snapshot_restores_position_through_the_snapshotter(self):
    """End to end through `_load_snapshot` with a fake `Snapshotter.load` on CPU, on a one-device mesh."""
    device = jax.devices()[0]
    mesh = jax.sharding.Mesh(np.array([device]), ("data",))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    abstract_items = {"params": jax.ShapeDtypeStruct((2,), jnp.float32, sharding=replicated)}
    saved = {
        "items": {"params": jax.device_put(jnp.ones(2), replicated)},
        "data_iterator": {"global_next_index": 42},
    }
    snapshotter = FakeSnapshotter(latest_snapshot=(saved, 30))

    def fake_load(target):
      # The target must mirror the saved structure, iterator state included, for Orbax's tree map.
      self.assertEqual(jax.tree.structure(target), jax.tree.structure(saved))
      self.assertEqual(target["data_iterator"], {"global_next_index": 0})
      return saved

    def fake_restored_linen_to_nnx(items, abstract_state, config=None):
      del abstract_state, config
      return items

    snapshotter.load.side_effect = fake_load
    self.enter_context(mock.patch.object(train_state_nnx, "to_checkpoint_dict", return_value=abstract_items))
    self.enter_context(mock.patch.object(checkpointing, "restored_linen_to_nnx", side_effect=fake_restored_linen_to_nnx))
    iterator = FakeElasticIterator(0)

    restored = snapshotting.load_snapshot(snapshotter, "abstract", self.config, fake_data_iterator(iterator))

    np.testing.assert_array_equal(restored["params"], jnp.ones(2))
    self.assertEqual(iterator.state, {"global_next_index": 42})

  def test_load_snapshot_returns_none_when_no_replica_survived(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.enter_context(
        mock.patch.object(snapshotting, "_load_snapshot", side_effect=RuntimeError("No active replicas found."))
    )

    self.assertIsNone(snapshotting.load_snapshot(snapshotter, "abstract", self.config))

  def test_load_snapshot_reraises_slice_down(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.enter_context(
        mock.patch.object(snapshotting, "_load_snapshot", side_effect=jax.errors.JaxRuntimeError("DATA_LOSS"))
    )
    self.enter_context(mock.patch.object(elastic_utils, "is_slice_down_error", return_value=True))

    with self.assertRaises(jax.errors.JaxRuntimeError):
      snapshotting.load_snapshot(snapshotter, "abstract", self.config)

  def test_load_snapshot_falls_back_on_non_slice_down_jax_error(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.enter_context(
        mock.patch.object(snapshotting, "_load_snapshot", side_effect=jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED"))
    )
    self.enter_context(mock.patch.object(elastic_utils, "is_slice_down_error", return_value=False))

    self.assertIsNone(snapshotting.load_snapshot(snapshotter, "abstract", self.config))

  def test_load_state_if_possible_prefers_snapshot(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    manager = fake_checkpointer(latest_step=20)
    self.enter_context(mock.patch.object(snapshotting, "load_snapshot", return_value="snapshot_state"))

    restored, raw_params = checkpointing.load_state_if_possible(
        manager,
        None,
        "",
        "",
        32,
        "abstract",
        maxtext_config=self.config,
        snapshotter=snapshotter,
    )

    self.assertEqual(restored, {"items": "snapshot_state"})
    self.assertIsNone(raw_params)

  def test_load_state_if_possible_falls_back_when_snapshot_fails(self):
    snapshotter = FakeSnapshotter(latest_snapshot=("pinned", 30))
    self.enter_context(mock.patch.object(snapshotting, "load_snapshot", return_value=None))

    restored, raw_params = checkpointing.load_state_if_possible(
        None,
        None,
        "",
        "",
        32,
        "abstract",
        maxtext_config=self.config,
        snapshotter=snapshotter,
    )

    self.assertIsNone(restored)
    self.assertIsNone(raw_params)


if __name__ == "__main__":
  absltest.main()
