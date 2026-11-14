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

"""Tests for training and data loading hooks for SFT"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.external_training, pytest.mark.post_training]

import jax
from jax.sharding import Mesh
import numpy as np
import os
import shutil
import tempfile

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.trainers.post_train import hooks as post_train_hooks
from maxtext.trainers.post_train.sft import hooks as sft_hooks
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils


class SFTHooksTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.config = pyconfig.initialize(
        ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
        per_device_batch_size=1,
        run_name="test",
        base_output_directory=self.test_dir,
        tensorboard_dir=self.test_dir,
        skip_jax_distributed_system=True,
    )
    self.mesh = Mesh(maxtext_utils.create_device_mesh(self.config), self.config.mesh_axes)

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_sft_data_hooks_load_next_train_batch(self, mock_create_data_iterator):
    expected_batch = {"inputs": np.zeros([jax.device_count(), self.config.max_target_length], dtype=int)}
    mock_data_iterator = MagicMock()
    mock_data_iterator.__next__.return_value = expected_batch
    mock_create_data_iterator.return_value = mock_data_iterator, None

    data_hooks = sft_hooks.SFTDataHooks(self.config, self.mesh, goodput_recorder=None)
    data_hooks.load_next_train_batch(train_ctx=None)

    self.assertIsNotNone(data_hooks.train_batch)
    self.assertEqual(data_hooks.train_batch["inputs"].shape, expected_batch["inputs"].shape)

  @patch.object(post_train_hooks, "create_data_iterator")
  def test_sft_data_hooks_reorders_all_sft_and_diffusion_keys(self, mock_create_data_iterator):
    seq_len = 8
    if jax.device_count() >= 4:
      config = pyconfig.initialize(
          ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
          per_device_batch_size=1,
          ici_fsdp_parallelism=2,
          ici_context_parallelism=2,
          run_name="test",
          base_output_directory=self.test_dir,
          tensorboard_dir=self.test_dir,
          skip_jax_distributed_system=True,
      )
      mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "context"))
      batch_size = 4
      cp_ctx = nullcontext()
    else:
      config = self.config
      mesh = self.mesh
      batch_size = jax.device_count()
      cp_ctx = patch.object(max_utils, "reordered_cp_size", return_value=2)

    sft_keys = (
        "inputs",
        "inputs_position",
        "inputs_segmentation",
        "targets",
        "targets_position",
        "targets_segmentation",
        "completion_mask",
        "corruption_mask",
        "targets_loss_mask",
    )
    seq = np.broadcast_to(np.arange(seq_len, dtype=np.int32), (batch_size, seq_len))
    raw_batch = {key: seq.copy() for key in sft_keys}
    mock_create_data_iterator.return_value = (iter([raw_batch]), iter([raw_batch]))

    with cp_ctx:
      data_hooks = sft_hooks.SFTDataHooks(config, mesh, goodput_recorder=None)

    train_batch = data_hooks.load_next_train_batch(train_ctx=None)
    train_ctx = SimpleNamespace(training_hooks=SimpleNamespace(eval_metadata={"eval_step_count": 0}))
    eval_batch = data_hooks.load_next_eval_batch(train_ctx=train_ctx)
    expected_reordered = np.broadcast_to(
        np.array([0, 1, 6, 7, 2, 3, 4, 5], dtype=np.int32),
        (batch_size, seq_len),
    )
    for key in sft_keys:
      np.testing.assert_array_equal(np.asarray(train_batch[key]), expected_reordered)
      np.testing.assert_array_equal(np.asarray(eval_batch[key]), expected_reordered)

  def test_sft_training_hooks_get_total_weights(self):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)
    training_hooks = sft_hooks.SFTTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    batch = {"targets_segmentation": np.array([[1, 1, 0], [1, 0, 0]])}
    total_weights = training_hooks.get_total_weights(batch)
    self.assertEqual(total_weights, 3)

  def test_sft_training_hooks_prefers_explicit_loss_mask(self):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)
    training_hooks = sft_hooks.SFTTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    batch = {
        "targets_segmentation": np.ones((2, 4), dtype=np.int32),
        "targets_loss_mask": np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.int32),
    }

    total_weights = training_hooks.get_total_weights(batch)

    self.assertEqual(total_weights, 3)

  def test_sft_training_hooks_falls_back_from_none_loss_mask(self):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)
    training_hooks = sft_hooks.SFTTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    batch = {
        "targets_segmentation": np.array([[1, 1, 0], [1, 0, 0]]),
        "targets_loss_mask": None,
    }

    total_weights = training_hooks.get_total_weights(batch)

    self.assertEqual(total_weights, 3)

  def test_preaveraged_eval_loss_is_objective_driven(self):
    training_hooks = object.__new__(sft_hooks.SFTTrainingHooks)

    training_hooks.config = SimpleNamespace()
    self.assertFalse(training_hooks.eval_loss_is_preaveraged())

    training_hooks.config = SimpleNamespace(training_objective="causal_lm")
    self.assertFalse(training_hooks.eval_loss_is_preaveraged())

    training_hooks.config = SimpleNamespace(training_objective="block_diffusion")
    self.assertTrue(training_hooks.eval_loss_is_preaveraged())


if __name__ == "__main__":
  unittest.main()
