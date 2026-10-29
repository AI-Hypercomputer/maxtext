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

import gc
import os
import re
import weakref

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib
import unittest
from unittest import mock

# Ensure tunix C-extension / protobuf initializes before transformers/orbax
try:
  import tunix.dist_rl.weight_sync.raiden_synchronizer  # pylint: disable=unused-import
except ImportError:
  pass

import jax
import jax.numpy as jnp
import pytest

from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
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

  @mock.patch("tunix.dist_rl.weight_sync.raiden_synchronizer.RaidenSynchronizer")
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
    # Async checkpoint persistence overlaps with the weight sync instead of being waited for.
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

  @mock.patch("tunix.dist_rl.weight_sync.raiden_synchronizer.RaidenSynchronizer")
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

    # Still only 1 synchronizer instance created, and round 2 released round 1's tree before converting.
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
    """At most one converted tree is alive: each round's tree is released before the next is converted."""

    class _Leaf:
      pass

    events = []
    live_refs = []
    live_trees_at_convert = []

    class _FakeSync:
      """The parts of `RaidenSynchronizer` the engine uses, holding the bound tree as it does."""

      def __init__(self, **kwargs):
        del kwargs
        self.arrays = []
        self.names = []
        self.active = False

      def bind(self, tree):
        self.names = list(tree)
        self.arrays = list(tree.values())

      def release_buffers(self) -> int:
        events.append("release_buffers")
        released = len(self.arrays)
        self.arrays = []
        self.names = []
        return released

      def metrics(self):
        events.append(f"metrics(n={len(self.arrays)})")
        return {"count": len(self.arrays)}

      def work_unit_metadata_all(self):
        return []

    def convert(*args, **kwargs):
      del args, kwargs
      gc.collect()
      alive_before = sum(1 for ref in live_refs if ref() is not None)
      leaf = _Leaf()
      live_refs.append(weakref.ref(leaf))
      live_trees_at_convert.append((alive_before, sum(1 for ref in live_refs if ref() is not None)))
      events.append("convert")
      return {"w": leaf}

    self.engine._weight_converter.convert.side_effect = convert

    with mock.patch("tunix.dist_rl.weight_sync.raiden_synchronizer.RaidenSynchronizer", side_effect=_FakeSync):
      # A normal round: prepare, then release.
      self.engine.prepare_weight_sync()
      self.assertEqual(len(self.engine._raiden_sync.arrays), 1)
      self.engine.release_weight_sync()
      self.assertEqual(self.engine._raiden_sync.arrays, [])
      self.assertIsNone(live_refs[0]())

      # Two rounds without a release in between: the second still releases the first's tree before converting.
      self.engine._train_step = 1
      self.engine.prepare_weight_sync()
      self.engine._train_step = 2
      self.engine.prepare_weight_sync()

    self.assertEqual(live_trees_at_convert, [(0, 1), (0, 1), (0, 1)])
    self.engine._checkpoint_manager.wait_until_finished.assert_not_called()
    # The metrics are read while the tree is still bound, before it is released.
    self.assertEqual(events[:3], ["convert", "metrics(n=1)", "release_buffers"])

  @mock.patch("tunix.dist_rl.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_weight_sync_logs_sub_step_timings(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.active = True
    mock_sync.release_buffers.return_value = 2
    mock_sync.work_unit_metadata_all.return_value = [self._make_dummy_metadata(num_vars=3)]
    mock_sync_cls.return_value = mock_sync
    self.engine._weight_converter.convert.return_value = {"p0": 0, "p1": 1}
    self.engine._train_step = 7

    with self.assertLogs(level="INFO") as logs:
      self.engine.prepare_weight_sync()
      self.engine.prepare_weight_sync()  # Same step: reuses the staged weights.
      self.engine.release_weight_sync()
    timings = [record.getMessage() for record in logs.records if "TRAINER_WEIGHT_SYNC_TIMING" in record.getMessage()]

    self.assertEqual(len(timings), 3, "\n".join(timings))
    seconds = r"\d+\.\d{3}"
    self.assertRegex(
        timings[0],
        r"step=prepare_weight_sync train_step=7 transport=raiden"
        rf" throttler_wait_s={seconds} purge_buffers_s={seconds} convert_s={seconds} bind_s={seconds}"
        rf" d2h_s={seconds} checksum_s={seconds} metadata_s={seconds} total_s={seconds} work_units=1 variables=3$",
    )
    self.assertRegex(
        timings[1],
        rf"step=prepare_weight_sync train_step=7 transport=raiden reused=True total_s={seconds} work_units=1"
        r" variables=3$",
    )
    self.assertRegex(
        timings[2],
        rf"step=release_weight_sync train_step=7 purge_buffers_s={seconds} purged_leaves=2 total_s={seconds}$",
    )
    # The sub-steps are parts of the whole.
    parts = [float(v) for v in re.findall(r"_s=([\d.]+)", timings[0])]
    self.assertLessEqual(sum(parts[:-1]), parts[-1] + 1e-3 * len(parts))

  @mock.patch("tunix.dist_rl.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_weight_staging_is_annotated_in_profiles(self, mock_sync_cls):
    mock_sync = mock.MagicMock()
    mock_sync.work_unit_metadata_all.return_value = []
    mock_sync_cls.return_value = mock_sync
    self.engine._weight_converter.convert.return_value = {"p0": 0}
    entered = []

    @contextlib.contextmanager
    def trace_annotation(name, **kwargs):
      del kwargs
      entered.append(name)
      self.engine._throttler.wait_for_all.assert_not_called()
      yield
      mock_sync.work_unit_metadata_all.assert_called_once()

    with mock.patch.object(jax.profiler, "TraceAnnotation", side_effect=trace_annotation):
      self.engine.prepare_weight_sync()
      self.engine.prepare_weight_sync()  # Reused: no staging work to annotate.

    self.assertEqual(entered, ["prepare_weight_sync"])

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

  def test_rollout_weight_dtype_config_reaches_weight_converter(self):
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_rollout_weight_dtype",
        scan_layers=True,
        num_decoder_layers=2,
        param_scan_axis=1,
        inhomogeneous_layer_cycle_interval=1,
        weight_dtype="float32",
        rollout_weight_dtype="bfloat16",
        enable_checkpointing=False,
    )
    self.assertEqual(cfg.weight_dtype, jnp.float32)
    self.assertEqual(cfg.rollout_weight_dtype, jnp.bfloat16)

    with (
        mock.patch.object(MaxTextTrainingEngine, "_build_model", return_value=mock.MagicMock()),
        mock.patch.object(MaxTextTrainingEngine, "_build_optimizer", return_value=mock.MagicMock()),
        mock.patch.object(
            maxtext_engine.train_utils,
            "create_training_optimizer",
            return_value=(mock.MagicMock(), mock.MagicMock()),
        ),
        mock.patch.object(maxtext_engine.checkpointing, "CheckpointManager"),
        mock.patch.object(maxtext_engine.metrics_module, "MetricsLogger"),
    ):
      engine = MaxTextTrainingEngine(
          training_config=cfg,
          mesh=mock.MagicMock(),
          wrap_with_tunix_adapter=False,
      )
    self.assertIsNotNone(engine._weight_converter)
    self.assertEqual(engine._weight_converter._direct._resolve_target_dtype(), jnp.bfloat16)


if __name__ == "__main__":
  unittest.main()
