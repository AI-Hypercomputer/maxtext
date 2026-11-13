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

"""Tests for shared base training and data loading hooks for post-training"""

from collections import defaultdict
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from absl.testing import absltest
import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.external_training, pytest.mark.post_training]

import jax
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.trainers.post_train.hooks import BaseDataHooks, BaseTrainingHooks, GoodputTrainingHooks
from maxtext.common.metric_logger import MetricLogger
from maxtext.utils import exceptions
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import train_utils

if "maxtext" in sys.modules and not hasattr(sys.modules["maxtext"], "trainers"):
  setattr(
      sys.modules["maxtext"],
      "trainers",
      sys.modules["maxtext.src.maxtext.trainers"],
  )


class DummyTrainingHooks(BaseTrainingHooks):
  """Dummy training hooks for testing base functionality."""

  def get_total_weights(self, batch) -> jax.Array:
    return np.sum(batch["targets_segmentation"] != 0)


class PreaveragedDummyTrainingHooks(DummyTrainingHooks):
  """Models an objective that passes a weighted eval mean."""

  def eval_loss_is_preaveraged(self) -> bool:
    return True


class BaseHooksTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.metrics_file = os.path.join(self.test_dir, "metrics.txt")
    self.config = pyconfig.initialize(
        ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
        per_device_batch_size=1,
        run_name="test",
        base_output_directory=self.test_dir,
        tensorboard_dir=self.test_dir,
        metrics_dir=self.test_dir,
        metrics_file=self.metrics_file,
        skip_jax_distributed_system=True,
    )
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)
    self.learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)

    self.training_hooks = DummyTrainingHooks(self.config, self.mesh, self.learning_rate_schedule, goodput_recorder=None)
    self.training_hooks.metric_logger = MetricLogger(self.config, self.learning_rate_schedule)
    self.training_hooks.metric_logger.metadata = defaultdict(float)

    expected_shape = [jax.device_count(), self.config.max_target_length]
    self.expected_batch = {
        "targets_segmentation": np.ones(expected_shape, dtype=int),
    }
    self.mock_train_ctx = MagicMock()
    self.mock_train_ctx.metrics_logger.get_metric.side_effect = KeyError("Metric not found")

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def _read_logged_metrics(self, num_expected=1):
    """Read and parse metrics logged by the MetricLogger."""
    metrics = []
    if os.path.exists(self.metrics_file):
      with open(self.metrics_file, "r", encoding="utf8") as f:
        for line in f:
          metrics.append(json.loads(line))
    self.assertEqual(len(metrics), num_expected)
    return metrics

  def test_training_hooks_for_train_step(self):
    self.training_hooks.metadata = {"first_train_step": 0}
    self.mock_train_ctx.data_hooks.train_batch = self.expected_batch
    self.mock_train_ctx.train_steps = 0
    self.training_hooks.on_train_step_start(self.mock_train_ctx)
    self.mock_train_ctx.train_steps = 1
    self.training_hooks.on_train_step_start(self.mock_train_ctx)
    self.training_hooks.on_train_step_end(self.mock_train_ctx, train_step=1, train_loss=5.0, step_time=0.004)

    metrics = self._read_logged_metrics(num_expected=1)[0]
    self.assertEqual(metrics["step"], 1)
    self.assertAlmostEqual(metrics["learning/loss"], 5.0)
    self.assertEqual(metrics["learning/total_weights"], (jax.device_count() * self.config.max_target_length))

  def test_training_hooks_for_eval_step(self):
    self.mock_train_ctx.data_hooks.eval_batch = self.expected_batch
    self.mock_train_ctx.train_steps = 0
    total_eval_steps = 2
    for _ in range(total_eval_steps):
      self.training_hooks.on_eval_step_start(self.mock_train_ctx)
    self.training_hooks.on_eval_step_end(self.mock_train_ctx, eval_loss=10.0)

    metrics = self._read_logged_metrics(num_expected=1)[0]
    self.assertEqual(metrics["step"], 0)
    self.assertAlmostEqual(metrics["eval/total_loss"], 10.0)
    self.assertAlmostEqual(metrics["eval/avg_loss"], 5.0)
    self.assertAlmostEqual(metrics["eval/avg_perplexity"], np.exp(5.0), places=2)
    self.assertEqual(metrics["eval/total_weights"], jax.device_count() * self.config.max_target_length * total_eval_steps)

  def test_preaveraged_eval_loss_is_not_divided_again(self):
    training_hooks = PreaveragedDummyTrainingHooks(
        self.config,
        self.mesh,
        self.learning_rate_schedule,
        goodput_recorder=None,
    )
    training_hooks.metric_logger = MetricLogger(self.config, self.learning_rate_schedule)
    training_hooks.metric_logger.metadata = defaultdict(float)
    self.mock_train_ctx.data_hooks.eval_batch = self.expected_batch
    self.mock_train_ctx.train_steps = 0
    for _ in range(2):
      training_hooks.on_eval_step_start(self.mock_train_ctx)

    training_hooks.on_eval_step_end(self.mock_train_ctx, eval_loss=3.0)

    metrics = self._read_logged_metrics(num_expected=1)[0]
    self.assertAlmostEqual(metrics["eval/total_loss"], 3.0)
    self.assertAlmostEqual(metrics["eval/avg_loss"], 3.0)
    self.assertAlmostEqual(metrics["eval/avg_perplexity"], np.exp(3.0), places=2)

  def test_on_train_end_asserts_if_on_train_start_not_called(self):
    with self.assertRaises(AssertionError):
      self.training_hooks.on_train_end(self.mock_train_ctx)

  def test_goodput_training_hooks(self):
    mock_recorder = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.enable_goodput_recording = True
    hooks = GoodputTrainingHooks(mock_cfg, mock_recorder)
    self.mock_train_ctx.train_steps = 3
    hooks.on_train_start(self.mock_train_ctx)
    hooks.on_train_step_start(self.mock_train_ctx)
    hooks.on_train_step_end(self.mock_train_ctx, train_step=4, train_loss=1.5)
    hooks.on_eval_step_start(self.mock_train_ctx)
    hooks.on_eval_step_end(self.mock_train_ctx, eval_loss=1.2)
    hooks.on_train_end(self.mock_train_ctx)
    mock_recorder.record_step_start_time.assert_called_once_with(3)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_load_next_train_batch_stop_iteration_returns_none(self, mock_create_data_iterator):
    mock_train_iter = MagicMock()
    mock_train_iter.__next__.side_effect = StopIteration()
    mock_create_data_iterator.return_value = (mock_train_iter, None)

    data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)
    batch = data_hooks.load_next_train_batch(self.mock_train_ctx)
    self.assertIsNone(batch)
    self.assertIsNone(data_hooks.train_batch)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_load_next_train_batch_propagates_unexpected_exception(self, mock_create_data_iterator):
    mock_train_iter = MagicMock()
    mock_train_iter.__next__.side_effect = IndexError("list index out of range")
    mock_create_data_iterator.return_value = (mock_train_iter, None)

    data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)
    with self.assertRaises(exceptions.StopTraining) as cm:
      data_hooks.load_next_train_batch(self.mock_train_ctx)
    self.assertIsInstance(cm.exception.__cause__, IndexError)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_load_next_eval_batch_stop_iteration_returns_none(self, mock_create_data_iterator):
    mock_eval_iter = MagicMock()
    mock_eval_iter.__next__.side_effect = StopIteration()
    mock_create_data_iterator.return_value = (MagicMock(), mock_eval_iter)

    data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)
    batch = data_hooks.load_next_eval_batch(self.mock_train_ctx)
    self.assertIsNone(batch)
    self.assertIsNone(data_hooks.eval_batch)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_load_next_eval_batch_propagates_unexpected_exception(self, mock_create_data_iterator):
    mock_eval_iter = MagicMock()
    mock_eval_iter.__next__.side_effect = RuntimeError("eval data stream failed")
    mock_create_data_iterator.return_value = (MagicMock(), mock_eval_iter)

    data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)
    with self.assertRaises(RuntimeError):
      data_hooks.load_next_eval_batch(self.mock_train_ctx)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_data_hooks_no_reorder_when_reordered_cp_size_is_one(
      self,
      mock_create_data_iterator,
  ):
    seq_len = 8
    raw_batch = {
        "inputs": np.broadcast_to(
            np.arange(seq_len, dtype=np.int32),
            (jax.device_count(), seq_len),
        ),
    }
    train_iter = iter([raw_batch])
    eval_iter = iter([raw_batch])
    mock_create_data_iterator.return_value = (train_iter, eval_iter)

    with patch.object(max_utils, "reordered_cp_size", return_value=1):
      data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)

    self.assertIs(data_hooks.train_data_iterator, train_iter)
    self.assertIs(data_hooks.eval_data_iterator, eval_iter)
    train_batch = data_hooks.load_next_train_batch(self.mock_train_ctx)
    eval_batch = data_hooks.load_next_eval_batch(self.mock_train_ctx)
    np.testing.assert_array_equal(
        np.asarray(train_batch["inputs"]),
        raw_batch["inputs"],
    )
    np.testing.assert_array_equal(
        np.asarray(eval_batch["inputs"]),
        raw_batch["inputs"],
    )

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_data_hooks_reorders_batches_when_reordered_cp_size_gt_one(
      self,
      mock_create_data_iterator,
  ):
    seq_len = 8
    sft_keys = (
        "inputs",
        "inputs_position",
        "inputs_segmentation",
        "targets",
        "targets_position",
        "targets_segmentation",
        "targets_loss_mask",
    )
    seq = np.broadcast_to(
        np.arange(seq_len, dtype=np.int32),
        (jax.device_count(), seq_len),
    )
    raw_batch = {key: seq.copy() for key in sft_keys}
    mock_create_data_iterator.return_value = (
        iter([raw_batch]),
        iter([raw_batch]),
    )

    with patch.object(max_utils, "reordered_cp_size", return_value=2):
      data_hooks = BaseDataHooks(self.config, self.mesh, goodput_recorder=None)

    # pylint: disable=protected-access
    self.assertIsInstance(
        data_hooks.train_data_iterator,
        train_utils._ReorderedDataIterator,
    )
    self.assertIsInstance(
        data_hooks.eval_data_iterator,
        train_utils._ReorderedDataIterator,
    )
    train_batch = data_hooks.load_next_train_batch(self.mock_train_ctx)
    eval_batch = data_hooks.load_next_eval_batch(self.mock_train_ctx)
    expected_reordered = np.broadcast_to(
        np.array([0, 1, 6, 7, 2, 3, 4, 5], dtype=np.int32),
        (jax.device_count(), seq_len),
    )
    for key in sft_keys:
      np.testing.assert_array_equal(
          np.asarray(train_batch[key]),
          expected_reordered,
      )
      np.testing.assert_array_equal(
          np.asarray(eval_batch[key]),
          expected_reordered,
      )


if __name__ == "__main__":
  absltest.main()
