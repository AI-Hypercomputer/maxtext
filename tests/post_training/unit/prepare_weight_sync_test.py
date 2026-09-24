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

"""Unit tests for MaxTextTrainingEngine.prepare_weight_sync single synchronizer logic."""

# pylint: disable=protected-access

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

# Ensure tunix C-extension / protobuf initializes before transformers/orbax
try:
  import tunix.experimental.weight_sync.raiden_synchronizer  # pylint: disable=unused-import
except ImportError:
  pass

import jax
import jax.numpy as jnp
import pytest

from maxtext.configs import pyconfig
from maxtext.training_engine.maxtext_engine import MaxTextTrainingEngine
from tests.utils.test_helpers import get_test_config_path

pytestmark = [pytest.mark.post_training]


class PrepareWeightSyncTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Create engine instance without running heavy __init__
    self.engine = MaxTextTrainingEngine.__new__(MaxTextTrainingEngine)
    self.engine._raiden_sync = None
    self.engine._last_staged_step = None
    self.engine._staged_metadata = None
    self.engine._train_step = 0
    self.engine._profiler = None
    self.engine._throttler = mock.MagicMock()
    self.engine._checkpoint_manager = mock.MagicMock()
    self.engine._config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_prepare_weight_sync",
        scan_layers=False,
        num_decoder_layers=2,
        param_scan_axis=1,
        inhomogeneous_layer_cycle_interval=1,
        weight_sync_debug=False,
        enable_checkpointing=False,
    )
    self.engine._use_weight_converter = True
    self.engine._weight_converter = mock.MagicMock()
    # `__new__` skips `__init__`, so every attribute `prepare_weight_sync` reads has to be
    # set here. None is the default a real engine carries until the rollout worker calls
    # `set_target_state`, and it selects target-free conversion.
    self.engine._target_state = None
    self.engine._rollout_backend = "maxtext"
    self.engine._get_trainable_params_state = mock.MagicMock(return_value={"layer": jnp.zeros((4, 4))})

  def _make_dummy_metadata(self, num_vars=2):
    meta = mock.MagicMock()
    meta.variables = [f"var_{i}" for i in range(num_vars)]
    meta.mesh_axes = (1, 1)
    return meta

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_single_synchronizer_creation_and_binding(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.active = True
    mock_sync.release_buffers.return_value = 0
    mock_sync.work_unit_metadata_all.return_value = [self._make_dummy_metadata(num_vars=2)]
    mock_sync.checksums.return_value = {}
    mock_sync_cls.return_value = mock_sync

    converted = {"param_0": 0, "param_1": 1}
    self.engine._weight_converter.convert.return_value = converted

    metadata = self.engine.prepare_weight_sync()

    self.assertEqual(len(metadata), 1)
    self.assertIs(self.engine._raiden_sync, mock_sync)
    self.engine._checkpoint_manager.wait_until_finished.assert_not_called()
    mock_sync_cls.assert_called_once_with(
        job_name="trainer",
        worker_index=jax.process_index(),
        auto_h2d=False,
        parallelism=4,
    )

    self.engine._weight_converter.convert.assert_called_once()
    mock_sync.bind.assert_called_once_with(converted)
    mock_sync.d2h.assert_called_once()
    mock_sync.work_unit_metadata_all.assert_called_once()

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_rebind_reuses_single_sync_instance(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.active = True
    mock_sync.release_buffers.return_value = 1
    mock_sync.work_unit_metadata_all.return_value = [self._make_dummy_metadata(num_vars=2)]
    mock_sync.checksums.return_value = {}
    mock_sync_cls.return_value = mock_sync

    # Round 1
    self.engine._weight_converter.convert.return_value = {"p0": 0}
    self.engine.prepare_weight_sync()
    self.assertEqual(mock_sync_cls.call_count, 1)

    # Round 2 at step 1
    self.engine._train_step = 1
    self.engine._weight_converter.convert.return_value = {"p0": 0}
    self.engine.prepare_weight_sync()

    # Still only 1 synchronizer instance created, and pre-convert purge called on round 2
    self.assertEqual(mock_sync_cls.call_count, 1)
    self.assertEqual(mock_sync.bind.call_count, 2)
    mock_sync.release_buffers.assert_called_once()

  def test_release_weight_sync_orders_metrics_before_purge(self):
    call_order = []
    mock_sync = mock.MagicMock()
    mock_sync.metrics.side_effect = lambda: (call_order.append("metrics"), {"bytes": 1024})[1]
    mock_sync.release_buffers.side_effect = lambda: (call_order.append("release_buffers"), 3)[1]
    self.engine._raiden_sync = mock_sync
    self.engine._last_staged_step = 1
    self.engine._staged_metadata = [{"metadata": "dummy"}]

    res = self.engine.release_weight_sync()

    self.assertTrue(res)
    self.assertIsNone(self.engine._last_staged_step)
    self.assertIsNone(self.engine._staged_metadata)
    self.assertEqual(call_order, ["metrics", "release_buffers"])

  def test_drain_before_convert_and_single_tree_peak_liveness(self):
    import gc
    import weakref

    class _Leaf:
      pass

    events = []
    live_refs = []
    peak_live_trees_during_convert = []

    class _FakeSync:
      def __init__(self, **kwargs):
        self.arrays = []
        self.names = []
        self.active = False

      def bind(self, tree):
        self.arrays.clear()
        self.names.clear()
        for k, v in tree.items():
          self.names.append(k)
          self.arrays.append(v)

      def release_buffers(self) -> int:
        events.append("release_buffers")
        n = len(self.arrays)
        self.arrays.clear()
        self.names.clear()
        return n

      def metrics(self):
        events.append(f"metrics(n={len(self.arrays)})")
        return {"count": len(self.arrays)}

      def work_unit_metadata_all(self):
        return []

    self.engine._checkpoint_manager.wait_until_finished.side_effect = (
        lambda: events.append("wait_until_finished")
    )

    def _convert_side_effect(_):
      gc.collect()
      alive_before = sum(1 for r in live_refs if r() is not None)
      leaf = _Leaf()
      live_refs.append(weakref.ref(leaf))
      alive_now = sum(1 for r in live_refs if r() is not None)
      peak_live_trees_during_convert.append((alive_before, alive_now))
      events.append("convert")
      return {"w": leaf}

    self.engine._weight_converter.convert.side_effect = _convert_side_effect

    with mock.patch(
        "tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer",
        side_effect=_FakeSync,
    ):
      # Step 0: prepare -> abort/normal release_weight_sync
      self.engine._train_step = 0
      self.engine.prepare_weight_sync()
      self.assertEqual(len(self.engine._raiden_sync.arrays), 1)
      self.engine.release_weight_sync()
      self.assertEqual(len(self.engine._raiden_sync.arrays), 0)
      self.assertIsNone(live_refs[0]())

      # Step 1 (also tests re-entry even if release_weight_sync was skipped):
      # Even if a tree were still bound, pre-convert _purge_raiden_buffers() frees it BEFORE convert().
      self.engine._train_step = 1
      self.engine.prepare_weight_sync()
      self.engine._train_step = 2
      self.engine.prepare_weight_sync()

    # Every convert() saw 0 prior trees alive (peak live converted trees == 1, never 2)
    self.assertEqual(peak_live_trees_during_convert, [(0, 1), (0, 1), (0, 1)])
    # prepare_weight_sync must NOT block on _checkpoint_manager.wait_until_finished(),
    # allowing async checkpointing to overlap with weight sync.
    self.assertNotIn("wait_until_finished", events)
    self.engine._checkpoint_manager.wait_until_finished.assert_not_called()
    # metrics runs when arrays are still bound (n=1), before release_buffers
    self.assertIn("metrics(n=1)", events)

  def test_release_weight_sync_without_syncs(self):
    self.engine._raiden_sync = None
    self.engine._last_staged_step = 1
    self.engine._staged_metadata = [{"metadata": "dummy"}]

    res = self.engine.release_weight_sync()

    self.assertTrue(res)
    self.assertIsNone(self.engine._last_staged_step)
    self.assertIsNone(self.engine._staged_metadata)

  def test_close(self):
    mock_sync = mock.MagicMock()
    self.engine._raiden_sync = mock_sync
    self.engine._last_staged_step = 1
    self.engine._staged_metadata = [{"metadata": "dummy"}]
    self.engine.save_checkpoint = mock.MagicMock()
    self.engine._checkpoint_manager = mock.MagicMock()
    self.engine._throttler = mock.MagicMock()
    self.engine._metrics_recorder = mock.MagicMock()
    self.engine._metrics_logger = mock.MagicMock()

    self.engine.close()

    mock_sync.close.assert_called_once()
    self.assertIsNone(self.engine._raiden_sync)
    self.assertIsNone(self.engine._last_staged_step)
    self.assertIsNone(self.engine._staged_metadata)


if __name__ == "__main__":
  unittest.main()

