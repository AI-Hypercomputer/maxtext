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
import re
import sys
import tempfile
import unittest
from unittest import mock

from etils import epath
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

# Skip the entire test suite if dependencies are missing
pytestmark = [pytest.mark.post_training]

# Now safe to do top-level imports
from maxtext.common import checkpointing
from tunix.sft import peft_trainer
from maxtext.utils import lora_utils
from maxtext.utils import model_creation_utils
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from jax.sharding import Mesh
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
  config_dict = _BASE_CONFIG.copy()
  config_dict.update(overrides)
  # Use initialize_pydantic to get nested models as objects (attribute access)
  return pyconfig.initialize_pydantic(
      [sys.argv[0], get_test_config_path()],
      **config_dict,
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

    mock_config.model_name = "gemma4-9b"
    path = lora_utils._get_lora_module_path(mock_config)
    self.assertEqual(
        path,
        "decoder/((scanned_blocks|layers_remainder)/)?layers.*/.*"
        "(self_attention/(query|key|value|out)|mlp/.*(wi_0|wi_1|wo|shared_experts/(wi_0|wi_1|wo)))",
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
    mock_config.lora.lora_weight_qtype = "int8"
    mock_config.lora.lora_tile_size = 32

    with mock.patch("qwix.LoraProvider") as mock_provider:
      lora_utils._build_lora_provider(mock_config)
      mock_provider.assert_called_once_with(
          module_path="custom/path",
          rank=8,
          alpha=16.0,
          dropout=0.0,
          weight_qtype="int8",
          tile_size=32,
      )

  def test_prepare_dummy_inputs(self):
    """Test preparation of dummy inputs for LoRA verification."""
    tokens, positions = lora_utils._prepare_dummy_inputs()
    self.assertEqual(tokens.shape, (1, 1))
    self.assertEqual(positions.shape, (1, 1))

  def test_verify_lora_parameters_success(self):
    """Test verification of LoRA parameters with matches and enabled LoRA."""
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.lora = mock.MagicMock()
    mock_config.lora.lora_module_path = ".*mlp/wi_0.*"

    mock_param = nnx.LoRAParam(0.0)
    mock_graph_entries = [
        (("decoder", "layers", 0, "mlp", "wi_0", "lora_a"), mock_param),
    ]

    with (
        mock.patch("maxtext.utils.lora_utils.nnx.iter_graph", return_value=mock_graph_entries),
        mock.patch("maxtext.utils.lora_utils.is_lora_enabled", return_value=True),
        mock.patch("maxtext.utils.max_logging.log") as mock_log,
    ):
      lora_utils._verify_lora_parameters(mock_model, mock_config)

      # Should log the successful match pattern summary
      log_calls = [call[0][0] for call in mock_log.call_args_list]
      self.assertTrue(any("successfully matched" in msg for msg in log_calls))
      self.assertTrue(any("Sample matched submodules" in msg for msg in log_calls))

  def test_verify_lora_parameters_not_enabled_no_match(self):
    """Test verification fails with ValueError when no modules match at all."""
    mock_model = mock.MagicMock()
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.lora = mock.MagicMock()
    mock_config.lora.lora_module_path = "non_existent"

    with (
        mock.patch("flax.nnx.iter_modules", return_value=[]),
        mock.patch("maxtext.utils.lora_utils.is_lora_enabled", return_value=False),
    ):
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

  def _run_apply_lora_test(
      self,
      scan_layers: bool,
      weight_qtype=None,
      tile_size=None,
      mock_multihost: bool = False,
  ):
    """Helper to run LoRA application test with/without scanned layers and optional QLoRA."""
    # Passing nested dict as 'lora' kwarg to _make_config
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8.0,
            "lora_module_path": ".*mlp/wi_.*",
            "lora_weight_qtype": weight_qtype,
            "lora_tile_size": tile_size,
        },
        scan_layers=scan_layers,
    )

    # Create a real small model using standard creation utils
    model, mesh = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)

    # Verify model is NOT lora enabled initially
    self.assertFalse(lora_utils.is_lora_enabled(model))

    if mock_multihost:
      devices_array = maxtext_utils.create_device_mesh(cfg)
      dummy_mesh = Mesh(devices_array, cfg.mesh_axes)

      # Just verify that apply_lora_to_model runs successfully with the dummy mesh
      lora_model = lora_utils.apply_lora_to_model(model, dummy_mesh, cfg)
    else:
      # Apply LoRA
      lora_model = lora_utils.apply_lora_to_model(model, mesh, cfg)

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
    """Test applying standard LoRA to model with scan_layers=False."""
    self._run_apply_lora_test(scan_layers=False)

  def test_apply_lora_to_model_scan_layers_true(self):
    """Test applying standard LoRA to model with scan_layers=True."""
    self._run_apply_lora_test(scan_layers=True)

  def test_apply_qlora_to_model_scan_layers_false(self):
    """Test applying QLoRA to model with scan_layers=False."""
    self._run_apply_lora_test(scan_layers=False, weight_qtype="int8", tile_size=32)

  def test_apply_qlora_to_model_scan_layers_true(self):
    """Test applying QLoRA to model with scan_layers=True."""
    self._run_apply_lora_test(scan_layers=True, weight_qtype="int8", tile_size=32)

  def test_apply_lora_multihost_mock(self):
    """Test applying LoRA with a dummy mesh to trigger the multi-host reshard callback."""
    self._run_apply_lora_test(scan_layers=False, mock_multihost=True)

  def test_restore_lora_from_path(self):
    """Test restoration of LoRA parameters from a path."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_restore_path": "some/path",
            "lora_rank": 4,
            "lora_alpha": 8.0,
        },
        scan_layers=False,
    )
    model, _ = model_creation_utils.from_pretrained(cfg, mesh=None, model_mode=model_creation_utils.MODEL_MODE_TRAIN)
    model = lora_utils.apply_lora_to_model(model, None, cfg)

    restored_state = nnx.state(model, nnx.LoRAParam)

    with mock.patch("orbax.checkpoint.PyTreeCheckpointer.restore", return_value=restored_state) as mock_restore:
      with mock.patch("flax.nnx.update") as mock_update:
        lora_utils.restore_lora_from_path(model, cfg)
        mock_restore.assert_called_once()
        args, kwargs = mock_restore.call_args
        self.assertEqual(args[0], "some/path")
        # Handle cases where partial_restore is passed as kwarg or within args object
        if "partial_restore" in kwargs:
          self.assertTrue(kwargs["partial_restore"])
        elif "args" in kwargs and hasattr(kwargs["args"], "partial_restore"):
          self.assertTrue(kwargs["args"].partial_restore)
        mock_update.assert_called_once()

  def test_sync_lora_metadata_default_syncs(self):
    """Test that default lora rank/alpha are successfully synced from checkpoint metadata."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_restore_path": "dummy/path",
            "lora_rank": 0,
            "lora_alpha": 0.0,
        }
    )
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 32, "lora_alpha": 64.0}}

    with mock.patch("orbax.checkpoint.StandardCheckpointer.metadata", return_value=mock_metadata):
      lora_utils.sync_lora_metadata(cfg)
      self.assertEqual(cfg.lora.lora_rank, 32)
      self.assertEqual(cfg.lora.lora_alpha, 64.0)

  def test_sync_lora_metadata_matching_passes(self):
    """Test that matching non-default parameters pass without errors."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_restore_path": "dummy/path",
            "lora_rank": 32,
            "lora_alpha": 64.0,
        }
    )
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 32, "lora_alpha": 64.0}}

    with mock.patch("orbax.checkpoint.StandardCheckpointer.metadata", return_value=mock_metadata):
      # Should not raise ValueError
      lora_utils.sync_lora_metadata(cfg)
      self.assertEqual(cfg.lora.lora_rank, 32)
      self.assertEqual(cfg.lora.lora_alpha, 64.0)

  def test_sync_lora_metadata_rank_mismatch_fails(self):
    """Test that configured rank mismatching checkpoint metadata rank raises ValueError."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_restore_path": "dummy/path",
            "lora_rank": 8,
            "lora_alpha": 64.0,
        }
    )
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 32, "lora_alpha": 64.0}}

    with mock.patch("orbax.checkpoint.StandardCheckpointer.metadata", return_value=mock_metadata):
      with self.assertRaisesRegex(ValueError, "Configured lora_rank .* does not match"):
        lora_utils.sync_lora_metadata(cfg)

  def test_sync_lora_metadata_alpha_mismatch_fails(self):
    """Test that configured alpha mismatching checkpoint metadata alpha raises ValueError."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_restore_path": "dummy/path",
            "lora_rank": 32,
            "lora_alpha": 16.0,
        }
    )
    mock_metadata = mock.MagicMock()
    mock_metadata.custom_metadata = {"lora": {"lora_rank": 32, "lora_alpha": 64.0}}

    with mock.patch("orbax.checkpoint.StandardCheckpointer.metadata", return_value=mock_metadata):
      with self.assertRaisesRegex(ValueError, "Configured lora_alpha .* does not match"):
        lora_utils.sync_lora_metadata(cfg)

  def test_save_checkpoint_passes_metadata(self):
    """Test that save_checkpoint correctly generates and passes custom lora metadata to CheckpointManager."""
    cfg = _make_config(
        lora={"enable_lora": True, "lora_rank": 8, "lora_alpha": 16.0},
        enable_checkpointing=True,
    )
    mock_manager = mock.MagicMock()
    mock_state = mock.MagicMock()

    with mock.patch("jax.block_until_ready"):
      checkpointing.save_checkpoint(mock_manager, step=10, state=mock_state, config=cfg)
      mock_manager.save.assert_called_once()
      _, kwargs = mock_manager.save.call_args
      self.assertIn("custom_metadata", kwargs)
      self.assertEqual(kwargs["custom_metadata"]["lora"], cfg.lora.model_dump())

  def test_save_and_restore_metadata_integration(self):
    """Integration test checking that Orbax CheckpointManager writes and reads custom LoRA metadata."""

    cfg_save = _make_config(
        lora={"enable_lora": True, "lora_rank": 8, "lora_alpha": 16.0},
        enable_checkpointing=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      manager = checkpointing.create_orbax_checkpoint_manager(
          tmpdir,
          enable_checkpointing=True,
          use_async=False,
          save_interval_steps=1,
          use_ocdbt=False,
          use_zarr3=False,
      )

      # Use save_checkpoint wrapper with a simple state
      dummy_state = {"weight": jnp.array([1.0, 2.0])}
      checkpointing.save_checkpoint(manager, step=0, state=dummy_state, config=cfg_save)
      manager.wait_until_finished()

      # Now verify that the saved checkpoint contains metadata on disk
      checkpoint_dir = epath.Path(tmpdir) / "0"
      self.assertTrue((checkpoint_dir / "_CHECKPOINT_METADATA").exists())

      # Restore using sync_lora_metadata on a config with default rank/alpha
      cfg_restore = _make_config(
          lora={
              "enable_lora": True,
              "lora_restore_path": str(checkpoint_dir),
              "lora_rank": 0,
              "lora_alpha": 0.0,
          }
      )
      lora_utils.sync_lora_metadata(cfg_restore)

      # Verify values were successfully synced back
      self.assertEqual(cfg_restore.lora.lora_rank, 8)
      self.assertEqual(cfg_restore.lora.lora_alpha, 16.0)

  def test_gemma4_lora_path_matching(self):
    """Test that the Gemma4 LoRA regex correctly matches all expected parameter paths."""
    mock_config = mock.MagicMock(spec=pyconfig.HyperParameters)
    mock_config.lora = mock.MagicMock()
    mock_config.lora.lora_module_path = ""
    mock_config.model_name = "gemma4-9b"

    path_regex = lora_utils._get_lora_module_path(mock_config)
    compiled = re.compile(path_regex)

    # Expected matching paths:
    matching_paths = [
        # Scan layers = True, Dense/MoE attention
        "decoder/scanned_blocks/layers/self_attention/query/kernel",
        "decoder/scanned_blocks/layers/self_attention/key/kernel",
        "decoder/scanned_blocks/layers/self_attention/value/kernel",
        "decoder/scanned_blocks/layers/self_attention/out/kernel",
        # Scan layers = True, Dense MLP
        "decoder/scanned_blocks/layers/mlp/wi_0/kernel",
        "decoder/scanned_blocks/layers/mlp/wi_1/kernel",
        "decoder/scanned_blocks/layers/mlp/wo/kernel",
        # Scan layers = True, MoE MLP
        "decoder/scanned_blocks/layers/mlp/shared_experts/wi_0/kernel",
        "decoder/scanned_blocks/layers/mlp/shared_experts/wi_1/kernel",
        "decoder/scanned_blocks/layers/mlp/shared_experts/wo/kernel",
        # Scan layers = False, Dense/MoE attention
        "decoder/layers_remainder/layers/0/self_attention/query/kernel",
        "decoder/layers_remainder/layers/0/self_attention/key/kernel",
        "decoder/layers_remainder/layers/0/self_attention/value/kernel",
        "decoder/layers_remainder/layers/0/self_attention/out/kernel",
        # Scan layers = False, Dense MLP
        "decoder/layers_remainder/layers/0/mlp/wi_0/kernel",
        "decoder/layers_remainder/layers/0/mlp/wi_1/kernel",
        "decoder/layers_remainder/layers/0/mlp/wo/kernel",
        # Scan layers = False, MoE MLP
        "decoder/layers_remainder/layers/0/mlp/shared_experts/wi_0/kernel",
        "decoder/layers_remainder/layers/0/mlp/shared_experts/wi_1/kernel",
        "decoder/layers_remainder/layers/0/mlp/shared_experts/wo/kernel",
    ]

    for path in matching_paths:
      self.assertTrue(compiled.search(path), f"Failed to match valid path: {path}")

    # Expected non-matching paths (e.g. layernorm, embedding):
    non_matching_paths = [
        "decoder/scanned_blocks/layers/pre_self_attention_norm/scale",
        "decoder/scanned_blocks/layers/post_self_attention_norm/scale",
        "decoder/layers_remainder/layers/0/pre_self_attention_norm/scale",
        "decoder/layers_remainder/layers/0/post_self_attention_norm/scale",
        "decoder/final_norm/scale",
        "token_embedder/embedding",
    ]

    for path in non_matching_paths:
      self.assertFalse(compiled.search(path), f"Incorrectly matched invalid path: {path}")

  def test_checkpoint_saving_lora_only(self):
    """Test that the native checkpointer only saves LoRA parameters when enable_lora is True."""
    # Create a mock config with lora enabled
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16.0,
        }
    )
    cfg.pure_nnx = True
    cfg.enable_diloco = False
    cfg.enable_checkpointing = True

    # Create a mock state
    mock_state = mock.MagicMock()
    mock_state.to_pure_dict.return_value = {
        "model": {
            "decoder": {
                "layers": {
                    "0": {
                        "self_attention": {
                            "query": {
                                "kernel": jnp.array([1.0]),  # Base weight
                                "lora_a": jnp.array([2.0]),  # LoRA weight
                            }
                        }
                    }
                }
            }
        },
        "optimizer": {
            "step": jnp.array(0, dtype=jnp.uint32),
            "opt_state": {},
        },
    }

    # When nnx.state(state.model, nnx.LoRAParam) is called, return only the lora variables
    mock_lora_state = mock.MagicMock()
    mock_lora_state.to_pure_dict.return_value = {
        "decoder": {
            "layers": {
                "0": {
                    "self_attention": {
                        "query": {
                            "lora_a": jnp.array([2.0]),
                        }
                    }
                }
            }
        }
    }

    with (
        mock.patch("flax.nnx.state") as mock_nnx_state,
        mock.patch("maxtext.common.checkpointing.save_checkpoint") as mock_save,
    ):
      mock_nnx_state.return_value = mock_lora_state

      # Call maybe_save_checkpoint
      mock_manager = mock.MagicMock()
      mock_manager.latest_step.return_value = -1
      mock_manager.reached_preemption.return_value = False

      checkpointing.maybe_save_checkpoint(
          checkpoint_manager=mock_manager,
          state=mock_state,
          config=cfg,
          data_iterator=None,
          step=0,
      )

      # Ensure save_checkpoint was called
      self.assertTrue(mock_save.called)

      # Extract the saved state passed to save_checkpoint
      saved_state = mock_save.call_args[0][2]

      # Verify that saved_state has the legacy Linen format ("params" and "opt_state")
      self.assertIn("params", saved_state)
      self.assertIn("step", saved_state)

      # Verify that base weights are GONE, but LoRA parameters are SAVED
      saved_params = saved_state["params"]["params"]
      # Inside saved_params, only lora_a exists, kernel is gone!
      query_params = saved_params["decoder"]["layers"]["0"]["self_attention"]["query"]
      self.assertIn("lora_a", query_params)
      self.assertNotIn("kernel", query_params)

  def test_checkpoint_restoration_preserves_base_weights(self):
    """Test that _populate_pure_dict_from_partial preserves base model weights under LoRA."""
    cfg = _make_config(
        lora={
            "enable_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16.0,
        }
    )

    # Live concrete state before restoration (containing pre-trained base weight [5.0])
    abstract_pure = {
        "model": {
            "decoder": {
                "layers": {
                    "0": {
                        "self_attention": {
                            "query": {
                                "kernel": jnp.array([5.0]),  # Loaded pre-trained base weight
                                "lora_a": jnp.array([1.0]),  # Initialized LoRA weight
                            }
                        }
                    }
                }
            }
        }
    }

    # Restored checkpoint state (only containing saved LoRA weights [9.0])
    partial_concrete = {
        "model": {
            "decoder": {
                "layers": {
                    "0": {
                        "self_attention": {
                            "query": {
                                "lora_a": jnp.array([9.0]),  # Saved LoRA weight
                            }
                        }
                    }
                }
            }
        }
    }

    # Call _populate_pure_dict_from_partial
    restored = checkpointing._populate_pure_dict_from_partial(
        abstract_pure=abstract_pure,
        partial_concrete=partial_concrete,
        config=cfg,
    )

    # Verify results
    query_restored = restored["model"]["decoder"]["layers"]["0"]["self_attention"]["query"]

    # 1. Base weights ("kernel") must be preserved exactly as [5.0] (not reset to zeros!)
    self.assertEqual(query_restored["kernel"][0], 5.0)

    # 2. LoRA weights ("lora_a") must be restored exactly as [9.0]
    self.assertEqual(query_restored["lora_a"][0], 9.0)


if __name__ == "__main__":
  unittest.main()
