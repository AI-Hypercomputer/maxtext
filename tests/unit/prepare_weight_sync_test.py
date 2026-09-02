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

"""Unit tests for MaxTextTrainingEngine.prepare_weight_sync streaming logic."""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


import sys
import types as pytypes
import unittest
from unittest import mock

# Ensure tunix C-extension / protobuf initializes before transformers/orbax
try:
  import tunix.experimental.weight_sync.raiden_synchronizer  # pylint: disable=unused-import
except ImportError:
  pass

import jax
import jax.numpy as jnp
from maxtext.training_engine.maxtext_engine import MaxTextTrainingEngine, _RAIDEN_WORKER_INDEX_STRIDE


class PrepareWeightSyncTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Create engine instance without running heavy __init__
    self.engine = MaxTextTrainingEngine.__new__(MaxTextTrainingEngine)
    self.engine._raiden_syncs = None
    self.engine._last_staged_step = None
    self.engine._staged_metadata = None
    self.engine._train_step = 0
    self.engine._throttler = mock.MagicMock()
    self.engine._config = pytypes.SimpleNamespace(
        scan_layers=False,
        num_decoder_layers=2,
        param_scan_axis=1,
        weight_sync_debug=False,
    )
    self.engine._use_weight_converter = True
    self.engine._weight_converter = mock.MagicMock()
    self.engine._rollout_backend = "maxtext"
    self.engine._warned_raiden_sync_chunks = False
    self.engine._get_trainable_params_state = mock.MagicMock(return_value={"layer": jnp.zeros((4, 4))})

  def _make_dummy_metadata(self, num_vars=2):
    meta = mock.MagicMock()
    meta.variables = [f"var_{i}" for i in range(num_vars)]
    meta.mesh_axes = (1, 1)
    return meta

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_streaming_grows_syncs_and_accumulates_metadata(self, mock_sync_cls):
    created_syncs = []

    def make_sync(*args, **kwargs):
      s = mock.MagicMock()
      s.active = True
      s.worker_index = kwargs.get("worker_index")
      s.work_unit_metadata.return_value = self._make_dummy_metadata(num_vars=2)
      s.checksums.return_value = {}
      created_syncs.append(s)
      return s

    mock_sync_cls.side_effect = make_sync

    pieces = [{"piece_0": 0}, {"piece_1": 1}, {"piece_2": 2}]
    self.engine._weight_converter.convert_streaming.return_value = iter(pieces)

    metadata = self.engine.prepare_weight_sync()

    self.assertEqual(len(metadata), 3)
    self.assertEqual(len(self.engine._raiden_syncs), 3)
    self.assertEqual(len(created_syncs), 3)

    # Worker indices must be unique and properly strided
    expected_indices = [
        jax.process_index() * _RAIDEN_WORKER_INDEX_STRIDE + i + 1 for i in range(3)
    ]
    actual_indices = [s.worker_index for s in created_syncs]
    self.assertEqual(actual_indices, expected_indices)

    # Check that bind, d2h, metadata, release were called on each sync
    for s, p in zip(created_syncs, pieces):
      s.bind.assert_called_once_with(p)
      s.d2h.assert_called_once()
      s.work_unit_metadata.assert_called_once()
      s.release_host_arrays.assert_called_once()

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_rebind_reuses_sync_instances(self, mock_sync_cls):
    mock_syncs = []

    def make_sync(*args, **kwargs):
      s = mock.MagicMock()
      s.active = True
      s.worker_index = kwargs.get("worker_index")
      s.work_unit_metadata.return_value = self._make_dummy_metadata(num_vars=2)
      mock_syncs.append(s)
      return s

    mock_sync_cls.side_effect = make_sync

    # Round 1
    self.engine._weight_converter.convert_streaming.return_value = iter([{"p0": 0}, {"p1": 1}])
    self.engine.prepare_weight_sync()
    self.assertEqual(len(mock_syncs), 2)
    first_round_syncs = list(self.engine._raiden_syncs)

    # Round 2 at step 1
    self.engine._train_step = 1
    self.engine._weight_converter.convert_streaming.return_value = iter([{"p0": 0}, {"p1": 1}])
    self.engine.prepare_weight_sync()

    # No new instances created
    self.assertEqual(len(mock_syncs), 2)
    self.assertEqual(self.engine._raiden_syncs, first_round_syncs)

  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_piece_count_mismatch_between_rounds_raises(self, mock_sync_cls):
    mock_sync_cls.side_effect = lambda *a, **kw: mock.MagicMock(
        active=True, work_unit_metadata=mock.MagicMock(return_value=self._make_dummy_metadata())
    )

    # Round 1 has 2 pieces
    self.engine._weight_converter.convert_streaming.return_value = iter([{"p0": 0}, {"p1": 1}])
    self.engine.prepare_weight_sync()

    # Round 2 has 3 pieces
    self.engine._train_step = 1
    self.engine._weight_converter.convert_streaming.return_value = iter([{"p0": 0}, {"p1": 1}, {"p2": 2}])
    with self.assertRaisesRegex(RuntimeError, "weight-sync piece count changed from 2 to 3"):
      self.engine.prepare_weight_sync()

  @mock.patch.dict(os.environ, {"RAIDEN_WEIGHT_SYNC_CHUNKS": "4"})
  @mock.patch("tunix.experimental.weight_sync.raiden_synchronizer.RaidenSynchronizer")
  def test_deprecated_chunks_env_var_warning(self, mock_sync_cls):
    mock_sync_cls.side_effect = lambda *a, **kw: mock.MagicMock(
        active=True, work_unit_metadata=mock.MagicMock(return_value=self._make_dummy_metadata())
    )
    self.engine._weight_converter.convert_streaming.return_value = iter([{"p0": 0}])
    with mock.patch("absl.logging.warning") as mock_warn:
      self.engine.prepare_weight_sync()
      mock_warn.assert_called()
      self.assertTrue(any("RAIDEN_WEIGHT_SYNC_CHUNKS is deprecated" in str(call) for call in mock_warn.call_args_list))
    self.assertTrue(self.engine._warned_raiden_sync_chunks)


if __name__ == "__main__":
  unittest.main()
