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
import sys
import unittest
from unittest import mock
import jax
import optax
import pytest
from flax import nnx

# Skip the entire test suite if dependencies are missing
pytestmark = [pytest.mark.post_training]

# Now safe to do top-level imports
from tunix.sft import peft_trainer
from maxtext.utils import lora_utils
from maxtext.utils import model_creation_utils
from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

# ---------------------------------------------------------------------------
# Shared minimal config overrides
# ---------------------------------------------------------------------------
_BASE_CONFIG = {
    "per_device_batch_size": 1.0,
    "run_name": "lora_utils_test",
    "enable_checkpointing": False,
    "base_num_decoder_layers": 1,
    "attention": "dot_product",
    "max_target_length": 8,
    "base_emb_dim": 128,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 256,
    "max_prefill_predict_length": 4,
    "model_name": "llama2-7b",
    "enable_nnx": True,
    "pure_nnx_decoder": True,
    "override_model_config": True,
    "weight_dtype": "bfloat16",
}


def _make_config(**overrides):
  """Return a MaxTextConfig object suitable for unit tests."""
  # Use initialize_pydantic to get nested models as objects (attribute access)
  return pyconfig.initialize_pydantic(
      [sys.argv[0], get_test_config_path()],
      **_BASE_CONFIG,
      **overrides,
  )


class LoraUtilsTest(unittest.TestCase):
  """Tests for lora_utils.py (Qwix LoRA Utils)"""

  # pylint: disable=protected-access

  def test_get_lora_module_path(self):
    """Test retrieving LoRA module path from config."""
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.lora = mock.MagicMock()
    mock_config.lora.lora_module_path = ""

    mock_config.model_name = "llama3.1-8b"
    path = lora_utils._get_lora_module_path(mock_config)
    self.assertEqual(
        path,
        "decoder/layers/(?:[0-9]+/)?.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))",
    )

    mock_config.model_name = "unknown_model"
    # Ensure lora.lora_module_path is still empty string to trigger fallback
    mock_config.lora.lora_module_path = ""
    path = lora_utils._get_lora_module_path(mock_config)
    # Fallback to default
    self.assertEqual(
        path,
        "decoder/layers/(?:[0-9]+/)?.*(self_attention/(query|key|value|out)|mlp/(wi_0|wi_1|wo))",
    )

    mock_config.lora.lora_module_path = "custom/path"
    path = lora_utils._get_lora_module_path(mock_config)
    self.assertEqual(path, "custom/path")

  def test_build_lora_provider(self):
    """Test building Qwix LoraProvider from config."""
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.model_name = "default"
    mock_config.lora = mock.MagicMock()
    mock_config.lora.lora_module_path = "custom/path"
    mock_config.lora.lora_rank = 8
    mock_config.lora.lora_alpha = 16.0

    with mock.patch("qwix.LoraProvider") as mock_provider:
      lora_utils._build_lora_provider(mock_config)
      mock_provider.assert_called_once_with(module_path="custom/path", rank=8, alpha=16.0, dropout=0.0)

  def test_prepare_dummy_inputs(self):
    """Test preparation of dummy inputs for LoRA verification."""
    tokens, positions = lora_utils._prepare_dummy_inputs()
    self.assertEqual(tokens.shape, (1, 1))
    self.assertEqual(positions.shape, (1, 1))

  def test_verify_lora_parameters_enabled(self):
    """Test verification of LoRA parameters when enabled."""
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)

    # Note: we use our local is_lora_enabled now
    with mock.patch("maxtext.utils.lora_utils.is_lora_enabled", return_value=True):
      # Should not raise
      lora_utils._verify_lora_parameters(mock_model, mock_config)

  def test_verify_lora_parameters_not_enabled_no_match(self):
    """Test verification fails when LoRA parameters are expected but not found."""
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.lora = mock.MagicMock()
    mock_config.model_name = "llama"
    mock_config.lora.lora_module_path = "non_existent"

    with mock.patch("maxtext.utils.lora_utils.is_lora_enabled", return_value=False):
      mock_model.iter_modules.return_value = []
      with self.assertRaisesRegex(ValueError, "no LoRA parameters found"):
        lora_utils._verify_lora_parameters(mock_model, mock_config)

  def test_apply_lora_to_model_disabled(self):
    """Test applying LoRA when it is disabled in config."""
    cfg = _make_config(lora={"enable_lora": False})
    model, _ = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)
    # Pydantic MaxTextConfig supports direct attribute access
    self.assertFalse(cfg.lora.enable_lora)
    result = lora_utils.apply_lora_to_model(model, None, cfg)
    self.assertEqual(result, model)
    self.assertFalse(lora_utils.is_lora_enabled(result))

  def test_apply_lora_to_model_adapters_loaded(self):
    """Test applying LoRA when adapters are already provided."""
    cfg = _make_config(**{"lora_input_adapters_path": "some/path"})
    model, _ = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)
    result = lora_utils.apply_lora_to_model(model, None, cfg)
    self.assertEqual(result, model)
    # is_lora_enabled checks for LoRAParam which Qwix adds.
    # If we skip Qwix, it should stay False.
    self.assertFalse(lora_utils.is_lora_enabled(result))

  def _run_apply_lora_test(self, scan_layers: bool):
    """Helper to run LoRA application test with/without scanned layers."""
    # Passing nested dict as 'lora' kwarg to _make_config
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_module_path": ".*mlp/wi_.*",
        },
        scan_layers=scan_layers,
    )

    # Create a real small model using standard creation utils
    model, _ = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)

    # Verify model is NOT lora enabled initially
    self.assertFalse(lora_utils.is_lora_enabled(model))

    # Apply LoRA
    lora_model = lora_utils.apply_lora_to_model(model, model.mesh, cfg)

    # Verify we can find LoRAParam in the state
    _, state = nnx.split(lora_model)
    lora_params = nnx.filter_state(state, nnx.LoRAParam)
    self.assertGreater(len(jax.tree_util.tree_leaves(lora_params)), 0)

    # Verify it IS now LoRA enabled
    self.assertTrue(lora_utils.is_lora_enabled(lora_model))

    # Test fit for PeftTrainer
    trainer_cfg = peft_trainer.TrainingConfig(eval_every_n_steps=10)
    optimizer = optax.adam(1e-4)

    # This instantiation will fail if wrt=nnx.LoRAParam cannot find any matching params
    trainer = peft_trainer.PeftTrainer(model=lora_model, optimizer=optimizer, training_config=trainer_cfg)

    # Verify optimizer is indeed targeting LoRAParams
    opt_state = nnx.state(trainer.optimizer)
    self.assertGreater(len(jax.tree_util.tree_leaves(opt_state)), 0)

  def test_apply_lora_to_model_scan_layers_false(self):
    """Test applying LoRA to model with scan_layers=False."""
    self._run_apply_lora_test(scan_layers=False)

  def test_apply_lora_to_model_scan_layers_true(self):
    """Test applying LoRA to model with scan_layers=True."""
    self._run_apply_lora_test(scan_layers=True)

  def test_restore_lora_from_path(self):
    """Test restoration of LoRA parameters from a path."""
    cfg = _make_config(
        lora={"enable_lora": True, "lora_restore_path": "some/path", "lora_rank": 4, "lora_alpha": 8.0},
        scan_layers=False,
    )
    model, _ = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)
    model = lora_utils.apply_lora_to_model(model, None, cfg)

    trainer = mock.MagicMock()
    trainer.model = model
    trainer.train_steps = 0

    restored_state = nnx.state(model, nnx.LoRAParam)

    with mock.patch("orbax.checkpoint.PyTreeCheckpointer.restore", return_value=restored_state) as mock_restore:
      with mock.patch("flax.nnx.update") as mock_update:
        lora_utils.restore_lora_from_path(trainer, cfg)
        mock_restore.assert_called_once()
        args, kwargs = mock_restore.call_args
        self.assertEqual(args[0], "some/path")
        # Handle cases where partial_restore is passed as kwarg or within args object
        if "partial_restore" in kwargs:
          self.assertTrue(kwargs["partial_restore"])
        elif "args" in kwargs and hasattr(kwargs["args"], "partial_restore"):
          self.assertTrue(kwargs["args"].partial_restore)
        mock_update.assert_called_once()


if __name__ == "__main__":
  unittest.main()
