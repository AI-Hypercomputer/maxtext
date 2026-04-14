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

"""Tests for Qwix LoRA utils in lora_utils.py"""
import unittest
from unittest import mock
import jax
from maxtext.utils import lora_utils


class LoraUtilsTest(unittest.TestCase):
  """Tests for lora_utils.py (Qwix LoRA Utils)"""

  # pylint: disable=protected-access

  def test_get_lora_module_path(self):
    mock_config = mock.MagicMock()
    mock_config.lora_module_path = ""

    mock_config.model_name = "llama3.1-8b"
    path = lora_utils._get_lora_module_path(mock_config)
    self.assertEqual(path, "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))")

    mock_config.model_name = "unknown_model"
    path = lora_utils._get_lora_module_path(mock_config)
    # Fallback to default
    self.assertEqual(path, "decoder/layers/.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))")

    mock_config.lora_module_path = "custom/path"
    path = lora_utils._get_lora_module_path(mock_config)
    self.assertEqual(path, "custom/path")

  def test_build_lora_provider(self):
    mock_config = mock.MagicMock()
    mock_config.lora_module_path = "custom/path"
    mock_config.lora_rank = 8
    mock_config.lora_alpha = 16.0
    mock_config.lora_tile_size = None
    mock_config.lora_weight_qtype = None

    with mock.patch("qwix.LoraProvider") as mock_provider:
      lora_utils._build_lora_provider(mock_config)
      mock_provider.assert_called_once_with(module_path="custom/path", rank=8, alpha=16.0, dropout=0.0)

  def test_prepare_dummy_inputs(self):
    tokens, positions = lora_utils._prepare_dummy_inputs()
    self.assertEqual(tokens.shape, (1, 1))
    self.assertEqual(positions.shape, (1, 1))

  def test_verify_lora_parameters_enabled(self):
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock()

    with mock.patch("tunix.sft.utils.is_lora_enabled", return_value=True):
      # Should not raise
      lora_utils._verify_lora_parameters(mock_model, mock_config)

  def test_verify_lora_parameters_not_enabled_no_match(self):
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock()
    mock_config.lora_module_path = "non_existent"
    mock_config.model_name = "llama"

    with mock.patch("tunix.sft.utils.is_lora_enabled", return_value=False):
      mock_model.iter_modules.return_value = []
      with self.assertRaisesRegex(ValueError, "no LoRA parameters found"):
        lora_utils._verify_lora_parameters(mock_model, mock_config)

  def test_apply_lora_to_model_disabled(self):
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock()
    mock_config.enable_lora = False
    mock_config.lora_input_adapters_path = ""
    result = lora_utils.apply_lora_to_model(mock_model, None, mock_config)
    self.assertEqual(result, mock_model)

  def test_apply_lora_to_model_adapters_loaded(self):
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock()
    mock_config.lora_input_adapters_path = "some/path"
    result = lora_utils.apply_lora_to_model(mock_model, None, mock_config)
    self.assertEqual(result, mock_model)

  def test_apply_lora_to_model(self):
    mock_model = mock.MagicMock()
    mock_mesh = mock.MagicMock()
    mock_config = mock.MagicMock()
    mock_config.enable_lora = True
    mock_config.lora_input_adapters_path = ""
    mock_config.logical_axis_rules = []

    with mock.patch("maxtext.utils.lora_utils._build_lora_provider"):
      with mock.patch("qwix.apply_lora_to_model", return_value=mock_model) as mock_qwix_apply:
        with mock.patch("maxtext.utils.lora_utils._verify_lora_parameters"):
          with mock.patch("flax.nnx.split", return_value=(mock.MagicMock(), mock.MagicMock())):
            with mock.patch("flax.nnx.merge") as mock_merge:
              with mock.patch("tunix.rl.reshard.reshard_pytree") as mock_reshard:
                with mock.patch("jax.tree.map", return_value=mock.MagicMock()):
                  with mock.patch("flax.nnx.get_partition_spec"):
                    lora_utils.apply_lora_to_model(mock_model, mock_mesh, mock_config)
                    mock_qwix_apply.assert_called_once()
                    mock_reshard.assert_called_once()
                    mock_merge.assert_called_once()


if __name__ == "__main__":
  unittest.main()
