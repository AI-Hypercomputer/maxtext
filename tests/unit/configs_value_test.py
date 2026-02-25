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

"""Tests for the new pydantic-based configuration system."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pydantic

from MaxText.globals import MAXTEXT_REPO_ROOT
from maxtext.configs import pyconfig, types
from maxtext.configs.pyconfig import initialize_pydantic

# Path to the base.yml config. This assumes that `pytest` is run from the project root.
_BASE_CONFIG_PATH = os.path.join(MAXTEXT_REPO_ROOT, "src", "maxtext", "configs", "base.yml")


class ConfigTest(unittest.TestCase):
  """Tests for the new pydantic-based configuration system."""

  def test_basic_config_loading(self):
    """Tests that a basic config loads and we can access a value."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=1"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.run_name, "test")
    self.assertEqual(config.steps, 1)
    self.assertIsInstance(config, pyconfig.HyperParameters)

  def test_type_conversion(self):
    """Tests that CLI arguments are converted to the correct types."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "per_device_batch_size=3.5",
        "enable_checkpointing=false",
        "steps=50",
    ]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.per_device_batch_size, 3.5)
    self.assertIsInstance(config.per_device_batch_size, float)
    self.assertEqual(config.enable_checkpointing, False)
    self.assertIsInstance(config.enable_checkpointing, bool)
    self.assertEqual(config.steps, 50)
    self.assertIsInstance(config.steps, int)

  def test_model_override(self):
    """Tests that model-specific configs override base.yml."""
    argv = ["", _BASE_CONFIG_PATH, "model_name=llama2-7b", "run_name=test"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.base_emb_dim, 4096)  # From llama2-7b.yml
    self.assertEqual(config.base_num_decoder_layers, 32)  # From llama2-7b.yml
    self.assertEqual(config.decoder_block, types.DecoderBlockType.LLAMA2)  # from llama2-7b.yml
    self.assertEqual(config.steps, 150001)  # From base.yml, not overridden

  def test_derived_values(self):
    """Tests that derived values are calculated correctly."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "global_parameter_scale=4",
        "per_device_batch_size=8",
        "gradient_accumulation_steps=2",
    ]
    # Mock jax.devices() to be deterministic
    mock_devices = [MagicMock(slice_index=0) for _ in range(8)]
    with patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    # global_parameter_scale=4 -> emb_scale=1, num_head_scale=1, mlp_dim_scale=1, layer_scale=0
    # base_emb_dim=2048, base_num_query_heads=16, base_mlp_dim=7168
    self.assertEqual(config.emb_dim, 2048 * (2**1))
    self.assertEqual(config.num_query_heads, 16 * (2**1))
    self.assertEqual(config.mlp_dim, 7168 * (2**1))

    # global_batch_size_to_train_on = per_device_batch_size * num_devices * gradient_accumulation_steps
    # num_devices is mocked to 8
    self.assertEqual(config.global_batch_size_to_train_on, 8 * 8 * 2)

  def test_validation_error(self):
    """Tests that a validation error is raised for invalid config."""
    # A negative number for steps should trigger a ValidationError in the pydantic model.
    argv = ["", _BASE_CONFIG_PATH, "steps=-5"]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)

  @patch.dict(os.environ, {pyconfig.yaml_key_to_env_key("steps"): "123"})
  def test_env_override(self):
    """Tests that environment variables override YAML values."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.steps, 123)

  @patch.dict(os.environ, {pyconfig.yaml_key_to_env_key("steps"): "123"})
  def test_cli_overrides_env_is_disallowed(self):
    """Tests that CLI arguments overriding environment variables fails."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=456"]
    # The new config logic explicitly forbids overriding the same key
    # from both CLI and environment variables to prevent ambiguity.
    with self.assertRaises(ValueError):
      pyconfig.initialize(argv)

  def test_llama3_tokenizer_correction(self):
    """Tests that tokenizer_type is forced to 'tiktoken' for llama3."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "model_name=llama3-8b",
        "tokenizer_path=assets/tokenizer_llama3.tiktoken",
        "run_name=test",
    ]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.tokenizer_type, "tiktoken")

  def test_initialize_pydantic_bad_keys(self):
    """Test that `pydantic.ValidationError` is raised on keys not in MaxTextConfig"""
    with self.assertRaises(ValueError):
      initialize_pydantic(
          [
              "",
              _BASE_CONFIG_PATH,
              "tokenizer_path=assets/tokenizer_llama3.tiktoken",
              "NOT_A_VALID_KEY=test",
          ]
      )


if __name__ == "__main__":
  unittest.main()
