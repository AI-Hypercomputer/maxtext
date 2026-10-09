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

"""Integration tests for SFT training (train_sft.py)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("tunix")
pytestmark = [pytest.mark.post_training, pytest.mark.integration_test]

from maxtext.configs import pyconfig
from maxtext.trainers.post_train.sft import train_sft
from maxtext.utils import exceptions
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_CONFIGS_DIR


class TrainSFTIntegrationTest(unittest.TestCase):
  """Integration tests for train_sft."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp(prefix="train_sft_integration_test_")
    self.config = pyconfig.initialize(
        ["", os.path.join(MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml")],
        run_name="test_sft_integration",
        base_output_directory=self.test_dir,
        dataset_type="synthetic",
        model_name="default",
        override_model_config=True,
        base_emb_dim=8,
        base_num_query_heads=4,
        base_num_kv_heads=4,
        base_mlp_dim=32,
        base_num_decoder_layers=1,
        head_dim=32,
        max_target_length=32,
        vocab_size=64,
        per_device_batch_size=1.0,
        steps=2,
        tokenizer_path=os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2"),
        use_tunix_gradient_accumulation=False,
        attention="dot_product",
        skip_jax_distributed_system=True,
    )

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)
    super().tearDown()

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_train_sft_stops_gracefully_on_stop_iteration(self, mock_create_data_iterator):
    """Verifies that StopIteration ends SFT training gracefully without raising."""
    mock_train_iter = MagicMock()
    mock_train_iter.__next__.side_effect = StopIteration()
    mock_create_data_iterator.return_value = (mock_train_iter, None)

    trainer, _ = train_sft.train(self.config, goodput_recorder=None)
    self.assertEqual(trainer.train_steps, 0)

  @patch("maxtext.trainers.post_train.hooks.create_data_iterator")
  def test_train_sft_fails_on_data_loading_exception(self, mock_create_data_iterator):
    """Verifies that a data-loading exception in SFT training raises instead of exiting cleanly."""
    mock_train_iter = MagicMock()
    mock_train_iter.__next__.side_effect = IndexError("list index out of range")
    mock_create_data_iterator.return_value = (mock_train_iter, None)

    with self.assertRaises(exceptions.StopTraining) as cm:
      train_sft.train(self.config, goodput_recorder=None)
    self.assertIsInstance(cm.exception.__cause__, IndexError)


if __name__ == "__main__":
  unittest.main()
