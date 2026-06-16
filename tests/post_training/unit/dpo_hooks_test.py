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

"""Tests for training and data loading hooks for DPO"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.external_training, pytest.mark.post_training]

import jax
from jax.sharding import Mesh
import numpy as np
import os
import shutil
import tempfile

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.trainers.post_train.dpo import hooks as dpo_hooks
from maxtext.utils import maxtext_utils


class DPOHooksTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.config = pyconfig.initialize_pydantic(
        ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "dpo.yml")],
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
  def test_dpo_data_hooks_load_next_train_batch(self, mock_create_data_iterator):
    expected_batch = {"inputs": np.zeros([jax.device_count(), self.config.max_target_length], dtype=int)}
    mock_data_iterator = MagicMock()
    mock_data_iterator.__next__.return_value = expected_batch
    mock_create_data_iterator.return_value = mock_data_iterator, None

    data_hooks = dpo_hooks.DPODataHooks(self.config, self.mesh, goodput_recorder=None)
    data_hooks.load_next_train_batch(train_ctx=None)

    self.assertIsNotNone(data_hooks.train_batch)
    self.assertEqual(data_hooks.train_batch["inputs"].shape, expected_batch["inputs"].shape)

  def test_dpo_training_hooks_get_total_weights(self):
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(self.config)
    training_hooks = dpo_hooks.DPOTrainingHooks(self.config, self.mesh, learning_rate_schedule, goodput_recorder=None)
    batch = {"chosen_mask": np.array([[1, 1, 0], [1, 0, 0]]), "rejected_mask": np.array([[1, 0, 0], [1, 1, 0]])}
    total_weights = training_hooks.get_total_weights(batch)
    self.assertEqual(total_weights, 6)


if __name__ == "__main__":
  unittest.main()
