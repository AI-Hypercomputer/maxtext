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

"""Unit tests for train_utils.py."""

import sys
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import optax

from maxtext.utils.train_utils import validate_train_config, create_training_optimizer
from maxtext.utils import maxtext_utils


@dataclass
class MockConfig:
  """Minimal mock config for validate_train_config tests."""

  run_name: str = "test_run"
  dataset_path: str = "gs://test-bucket/data"
  base_output_directory: str = "gs://test-bucket/output"
  steps: int = 100
  quantization: str = ""
  gradient_accumulation_steps: int = 1
  packing: bool = False
  dataset_type: str = "tfds"

  # Fields needed for create_training_optimizer
  opt_type: str = "adamw"
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  adam_eps: float = 1e-8
  adam_eps_root: float = 0.0
  adam_weight_decay: float = 0.1
  mu_dtype: str = ""
  learning_rate: float = 1e-4
  learning_rate_schedule_steps: int = 1000
  warmup_steps_fraction: float = 0.1
  cosine_learning_rate_final_fraction: float = 0.0
  steps: int = 100
  lr_schedule_type: str = "cosine"
  use_iota_embed: bool = False


class TestValidateTrainConfig(unittest.TestCase):
  """Tests for validate_train_config."""

  def test_valid_config_passes(self):
    """Verifies no exception raised for a valid config."""
    config = MockConfig()
    # Should not raise
    validate_train_config(config)

  def test_missing_run_name_raises(self):
    """Verifies AssertionError when run_name is empty."""
    config = MockConfig(run_name="")
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_zero_steps_raises(self):
    """Verifies AssertionError when steps is 0."""
    config = MockConfig(steps=0)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_negative_steps_raises(self):
    """Verifies AssertionError when steps is negative."""
    config = MockConfig(steps=-5)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_fp8_with_grad_accumulation_raises(self):
    """Verifies AssertionError for fp8 quantization + gradient_accumulation_steps > 1."""
    config = MockConfig(quantization="fp8", gradient_accumulation_steps=2)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_nanoo_fp8_with_grad_accumulation_raises(self):
    """Verifies AssertionError for nanoo_fp8 quantization + gradient_accumulation_steps > 1."""
    config = MockConfig(quantization="nanoo_fp8", gradient_accumulation_steps=4)
    with self.assertRaises(AssertionError):
      validate_train_config(config)

  def test_fp8_with_single_grad_accumulation_passes(self):
    """Verifies no error for fp8 with gradient_accumulation_steps=1."""
    config = MockConfig(quantization="fp8", gradient_accumulation_steps=1)
    validate_train_config(config)  # Should not raise

  def test_packing_with_synthetic_data_logs_warning(self):
    """Verifies no exception for packing + synthetic (just logs a warning)."""
    config = MockConfig(packing=True, dataset_type="synthetic")
    # Should not raise - just log a warning
    validate_train_config(config)

  def test_local_dataset_path_logs_warning(self):
    """Verifies no exception for local dataset_path (just logs a warning)."""
    config = MockConfig(dataset_path="/local/path/to/data")
    validate_train_config(config)  # Should not raise

  def test_local_output_directory_logs_warning(self):
    """Verifies no exception for local base_output_directory (just logs a warning)."""
    config = MockConfig(base_output_directory="/local/output")
    validate_train_config(config)  # Should not raise


class TestCreateTrainingOptimizer(unittest.TestCase):
  """Tests for create_training_optimizer."""

  def _make_config(self, opt_type="adamw", **kwargs):
    """Creates a mock config for optimizer tests."""
    cfg = MockConfig(opt_type=opt_type, **kwargs)
    return cfg

  def _mock_lr_schedule(self):
    """Returns a mock learning rate schedule that returns a fixed value."""
    return lambda step: 1e-4

  def test_adamw_optimizer_returns_schedule_and_tx(self):
    """Verifies create_training_optimizer returns a schedule and optax transform for adamw."""
    config = MagicMock()
    config.opt_type = "adamw"
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_eps = 1e-8
    config.adam_eps_root = 0.0
    config.adam_weight_decay = 0.01
    config.mu_dtype = None
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False

    schedule, tx = create_training_optimizer(config, model=None)

    self.assertIsNotNone(schedule)
    self.assertIsNotNone(tx)
    # Verify it's an optax GradientTransformation
    self.assertTrue(hasattr(tx, "init"))
    self.assertTrue(hasattr(tx, "update"))

  def test_adam_pax_optimizer_returns_tx(self):
    """Verifies create_training_optimizer works for adam_pax optimizer."""
    config = MagicMock()
    config.opt_type = "adam_pax"
    config.adam_b1 = 0.9
    config.adam_b2 = 0.999
    config.adam_eps = 1e-8
    config.adam_eps_root = 0.0
    config.adam_weight_decay = 0.01
    config.mu_dtype = None
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.1
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False

    _, tx = create_training_optimizer(config, model=None)

    self.assertIsNotNone(tx)
    self.assertTrue(hasattr(tx, "init"))
    self.assertTrue(hasattr(tx, "update"))

  def test_sgd_optimizer_returns_tx(self):
    """Verifies create_training_optimizer works for sgd optimizer."""
    config = MagicMock()
    config.opt_type = "sgd"
    config.learning_rate = 1e-4
    config.warmup_steps_fraction = 0.0
    config.cosine_learning_rate_final_fraction = 0.0
    config.steps = 100
    config.learning_rate_schedule_steps = 100
    config.lr_schedule_type = "cosine"
    config.use_iota_embed = False

    _, tx = create_training_optimizer(config, model=None)

    self.assertIsNotNone(tx)
    self.assertTrue(hasattr(tx, "init"))


if __name__ == "__main__":
  unittest.main()
