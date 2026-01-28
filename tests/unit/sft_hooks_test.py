# Copyright 2023–2025 Google LLC
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

"""Tests for training and data loading hooks for SFT"""
import pytest

pytestmark = pytest.mark.tpu_only

import jax

import numpy as np
import os
import unittest
from unittest.mock import MagicMock, patch
from jax.sharding import Mesh

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from maxtext.trainers.post_train.sft import hooks
from maxtext.utils import maxtext_utils


class SFTHooksTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        ["", os.path.join(MAXTEXT_PKG_DIR, "configs", "sft.yml")],
        per_device_batch_size=1,
        run_name="test",
        base_output_directory="test",
        skip_jax_distributed_system=True,
    )
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)

    self.training_hooks = hooks.SFTTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    self.training_hooks.metric_logger = MagicMock()

    expected_shape = [jax.device_count(), self.config.max_target_length]
    self.expected_batch = {
        "inputs": np.zeros(expected_shape, dtype=int),
        "targets_segmentation": np.ones(expected_shape, dtype=int),
    }
    self.mock_data_iterator = MagicMock()
    self.mock_data_iterator.__next__.return_value = self.expected_batch

    self.mock_train_ctx = MagicMock()

  @patch("maxtext.trainers.post_train.sft.hooks.create_data_iterator")
  def test_data_hooks_load_next_train_batch(self, mock_create_data_iterator):
    mock_create_data_iterator.return_value = self.mock_data_iterator, None
    data_hooks = hooks.SFTDataHooks(self.config, self.mesh, goodput_recorder=None)
    data_hooks.load_next_train_batch(train_ctx=None)

    self.assertIsNotNone(data_hooks.train_batch)
    self.assertEqual(data_hooks.train_batch["inputs"].shape, self.expected_batch["inputs"].shape)
    self.assertTrue((data_hooks.train_batch["inputs"] == self.expected_batch["inputs"]).all())

  @patch("maxtext.trainers.post_train.sft.hooks.create_data_iterator")
  def test_data_hooks_load_next_eval_batch(self, mock_create_data_iterator):
    mock_create_data_iterator.return_value = None, self.mock_data_iterator
    data_hooks = hooks.SFTDataHooks(self.config, self.mesh, goodput_recorder=None)
    data_hooks.load_next_eval_batch(train_ctx=None)

    self.assertIsNotNone(data_hooks.eval_batch)
    self.assertEqual(data_hooks.eval_batch["inputs"].shape, self.expected_batch["inputs"].shape)
    self.assertTrue((data_hooks.eval_batch["inputs"] == self.expected_batch["inputs"]).all())

  def test_training_hooks_for_train_step(self):
    self.training_hooks.metadata = {"first_train_step": 0}
    self.mock_train_ctx.data_hooks.train_batch = self.expected_batch
    self.mock_train_ctx.train_steps = 0
    self.training_hooks.on_train_step_start(self.mock_train_ctx)
    self.mock_train_ctx.train_steps = 1
    self.training_hooks.on_train_step_start(self.mock_train_ctx)
    self.training_hooks.on_train_step_end(self.mock_train_ctx, train_step=1, train_loss=5.0, step_time=0.004)

    expected_metrics = {
        "scalar": {
            "learning/loss": 5.0,
            "learning/total_weights": (jax.device_count() * self.config.max_target_length),
        }
    }
    self.training_hooks.metric_logger.record_train_metrics.assert_called()
    self.training_hooks.metric_logger.write_metrics.assert_called_with(expected_metrics, 1)
    self.assertEqual(len(self.training_hooks.train_metadata), 1)

  def test_training_hooks_for_eval_step(self):
    self.mock_train_ctx.data_hooks.eval_batch = self.expected_batch
    self.mock_train_ctx.train_steps = 0
    total_eval_steps = 2
    for _ in range(total_eval_steps):
      self.training_hooks.on_eval_step_start(self.mock_train_ctx)
    self.training_hooks.on_eval_step_end(self.mock_train_ctx, eval_loss=10.0)

    expected_metrics = {
        "scalar": {
            "eval/total_loss": 10.0,
            "eval/avg_loss": 5.0,
            "eval/total_weights": jax.device_count() * self.config.max_target_length * total_eval_steps,
        }
    }
    self.training_hooks.metric_logger.write_metrics.assert_called_with(expected_metrics, 0, is_training=False)
    self.assertEqual(len(self.training_hooks.eval_metadata), 0)

  def test_on_train_end_asserts_if_on_train_start_not_called(self):
    with self.assertRaises(AssertionError):
      self.training_hooks.on_train_end(self.mock_train_ctx)

  def test_on_train_step_end_asserts_if_on_train_step_start_not_called(self):
    with self.assertRaises(AssertionError):
      self.training_hooks.on_train_step_end(self.mock_train_ctx, train_step=1, train_loss=5.0, step_time=0.004)

  def test_on_eval_step_end_asserts_if_on_eval_step_start_not_called(self):
    with self.assertRaises(AssertionError):
      self.training_hooks.on_eval_step_end(self.mock_train_ctx, eval_loss=10.0)


if __name__ == "__main__":
  unittest.main()
