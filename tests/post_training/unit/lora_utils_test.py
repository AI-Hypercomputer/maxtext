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
import numpy as np
import optax
import pytest
from flax import nnx
from qwix._src.core.qarray import QArray

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
    """Test that checkpoint restoration preserves base model weights under LoRA."""
    # Simulating Orbax restoration behavior: missing base weights are restored as ShapeDtypeStructs
    overlay = {
        "model": {
            "decoder": {
                "layers": {
                    "0": {
                        "self_attention": {
                            "query": {
                                "kernel": jax.ShapeDtypeStruct(
                                    (1,), jnp.float32
                                ),  # Missing base weight is returned as ShapeDtypeStruct
                                "lora_a": jnp.array([9.0]),  # Saved LoRA weight
                            }
                        }
                    }
                }
            }
        }
    }

    # Live concrete state before restoration (containing pre-trained base weight [5.0] and initialized lora_a [1.0])
    state = {
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

    # Perform the exact same jax.tree.map merge as in setup_initial_state
    merged = jax.tree.map(
        lambda ckpt, init: init if isinstance(ckpt, jax.ShapeDtypeStruct) else ckpt,
        overlay,
        state,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )

    query_restored = merged["model"]["decoder"]["layers"]["0"]["self_attention"]["query"]

    # 1. Base weights ("kernel") must be preserved exactly as [5.0]
    self.assertEqual(query_restored["kernel"][0], 5.0)

    # 2. LoRA weights ("lora_a") must be restored exactly as [9.0]
    self.assertEqual(query_restored["lora_a"][0], 9.0)


# ---------------------------------------------------------------------------
# NNX-specific LoRA Helpers & Tests
# ---------------------------------------------------------------------------

from maxtext.utils import sharding
from maxtext.utils.lora_utils import (
    apply_lora_on_base_params,
    apply_lora_on_base_params_nnx,
    get_lora_abstract_state_nnx,
    unapply_lora_from_base_params,
    unapply_lora_from_base_params_nnx,
)


def _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4, dtype=jnp.float32):
  """Tiny NNX-shaped abstract state for one attention block."""

  def _sds(shape):
    return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=None)

  return {
      "decoder": {
          "layers": {
              "self_attention": {
                  "query": {"kernel": _sds((emb, num_heads, head_dim))},
                  "key": {"kernel": _sds((emb, num_heads, head_dim))},
                  "value": {"kernel": _sds((emb, num_heads, head_dim))},
                  "out": {"kernel": _sds((emb, num_heads, head_dim))},
              },
              "mlp": {"wi": {"kernel": _sds((emb, 4 * emb))}},
          },
          "shared_embedding": {"embedding": _sds((100, emb))},
      },
  }


def _make_linen_attention_abstract(emb=8, num_heads=2, head_dim=4, dtype=jnp.float32):
  """Linen-shaped equivalent (with the `{"params": ...}` outer wrap)."""
  return {"params": _make_nnx_attention_abstract(emb, num_heads, head_dim, dtype)}


def _lora_config(rank=4, alpha=8.0, target_modules=("q_proj", "v_proj")):
  return {
      "r": rank,
      "lora_alpha": alpha,
      "target_modules": list(target_modules),
  }


class TestGetLoraAbstractStateNnx(unittest.TestCase):
  """`get_lora_abstract_state_nnx` shape, sharding, and error-path coverage."""

  def test_lora_shapes_for_query_and_value(self):
    abs_params = _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4)
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(rank=4))
    attn = state.params["decoder"]["layers"]["self_attention"]

    a = attn["query"]["lora_a.kernel"]
    b = attn["query"]["lora_b.kernel"]
    self.assertEqual(a.shape, (8, 4))
    self.assertEqual(b.shape, (4, 2, 4))
    self.assertEqual(a.dtype, jnp.float32)
    self.assertEqual(b.dtype, jnp.float32)

    a = attn["value"]["lora_a.kernel"]
    b = attn["value"]["lora_b.kernel"]
    self.assertEqual(a.shape, (8, 4))
    self.assertEqual(b.shape, (4, 2, 4))

  def test_non_target_modules_emit_none_leaves(self):
    abs_params = _make_nnx_attention_abstract()
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(target_modules=("q_proj",)))
    attn = state.params["decoder"]["layers"]["self_attention"]
    self.assertIn("lora_a.kernel", attn["query"])
    self.assertIsNone(attn["key"]["kernel"])
    self.assertIsNone(attn["value"]["kernel"])
    self.assertIsNone(attn["out"]["kernel"])
    self.assertIsNone(state.params["decoder"]["layers"]["mlp"]["wi"]["kernel"])
    self.assertIsNone(state.params["decoder"]["shared_embedding"]["embedding"])

  def test_o_proj_has_distinct_shape(self):
    abs_params = _make_nnx_attention_abstract(emb=8, num_heads=2, head_dim=4)
    state, _ = get_lora_abstract_state_nnx(abs_params, _lora_config(rank=3, target_modules=("o_proj",)))
    out = state.params["decoder"]["layers"]["self_attention"]["out"]
    a = out["lora_a.kernel"]
    b = out["lora_b.kernel"]
    # For a 3D base (emb, num_heads, head_dim): lora_a.shape ends with rank,
    # lora_b shape is (rank, head_dim).
    self.assertEqual(a.shape, (8, 2, 3))
    self.assertEqual(b.shape, (3, 4))

  def test_unsupported_leaf_type_raises(self):
    bad = {"decoder": {"layers": {"self_attention": {"query": {"kernel": jnp.zeros((4, 2, 2))}}}}}
    with self.assertRaises(ValueError):
      get_lora_abstract_state_nnx(bad, _lora_config())

  def test_unexpected_leaf_name_raises(self):
    bad = {"decoder": {"layers": {"self_attention": {"query": {"weight": jax.ShapeDtypeStruct((4, 2), jnp.float32)}}}}}
    with self.assertRaises(ValueError):
      get_lora_abstract_state_nnx(bad, _lora_config())


def _concrete_base(rng=None, emb=4, num_heads=2, head_dim=3):
  """Concrete arrays mirroring the abstract structure used above (NNX-shape)."""
  if rng is None:
    rng = jax.random.key(0)
  k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
  shape_attn = (emb, num_heads, head_dim)
  return {
      "decoder": {
          "layers": {
              "self_attention": {
                  "query": {"kernel": jax.random.normal(k1, shape_attn)},
                  "key": {"kernel": jax.random.normal(k2, shape_attn)},
                  "value": {"kernel": jax.random.normal(k3, shape_attn)},
                  "out": {"kernel": jax.random.normal(k4, shape_attn)},
              },
              "mlp": {"wi": {"kernel": jax.random.normal(k5, (emb, 4 * emb))}},
          },
          "shared_embedding": {"embedding": jax.random.normal(k6, (100, emb))},
      },
  }


def _build_lora_params(base, lora_config_dict, rng):
  """Build a concrete LoRA tree (random arrays) matching `base`."""
  abs_tree = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=None), base)
  lora_state, _ = get_lora_abstract_state_nnx(abs_tree, lora_config_dict)

  def _to_concrete(leaf, rng_key):
    if leaf is None:
      return None
    return jax.random.normal(rng_key, leaf.shape, leaf.dtype)

  leaves, tree = jax.tree_util.tree_flatten(lora_state.params, is_leaf=lambda x: x is None)
  rngs = jax.random.split(rng, max(1, len(leaves)))
  out_leaves = [_to_concrete(l, r) for l, r in zip(leaves, rngs)]
  return jax.tree_util.tree_unflatten(tree, out_leaves)


class TestApplyLoraNnx(unittest.TestCase):
  """`apply_lora_on_base_params_nnx` round-trip and Linen-vs-NNX parity."""

  def test_apply_then_unapply_is_identity(self):
    rng = jax.random.key(42)
    base_orig = _concrete_base(rng)
    base = jax.tree_util.tree_map(jnp.copy, base_orig)
    lora = _build_lora_params(
        base,
        _lora_config(rank=2, target_modules=("q_proj", "v_proj")),
        jax.random.key(7),
    )
    apply_lora_on_base_params_nnx(base, lora, lora_scale_factor=0.5)
    # The query and value kernels are targets and must have changed.
    self.assertFalse(
        jnp.allclose(
            base["decoder"]["layers"]["self_attention"]["query"]["kernel"],
            base_orig["decoder"]["layers"]["self_attention"]["query"]["kernel"],
        )
    )
    # The key and out kernels are non-targets and must be untouched.
    np.testing.assert_array_equal(
        np.asarray(base["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(base["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
    )
    unapply_lora_from_base_params_nnx(base, lora, lora_scale_factor=0.5)
    np.testing.assert_allclose(
        np.asarray(base["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(base["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        np.asarray(base_orig["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_numerical_parity_with_linen_apply(self):
    """The NNX and Linen apply paths produce identical results on the same inputs."""
    rng = jax.random.key(123)
    base_nnx = _concrete_base(rng)
    base_linen = {"params": jax.tree_util.tree_map(jnp.copy, base_nnx)}
    lora = _build_lora_params(
        base_nnx,
        _lora_config(rank=2, target_modules=("q_proj",)),
        jax.random.key(5),
    )
    apply_lora_on_base_params_nnx(base_nnx, lora, lora_scale_factor=0.7)
    apply_lora_on_base_params(base_linen, {"params": lora}, lora_scale_factor=0.7)
    np.testing.assert_allclose(
        np.asarray(base_nnx["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_linen["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-6,
    )

  def test_apply_with_unexpected_lora_key_raises(self):
    base = _concrete_base()
    bad = {"decoder": {"layers": {"self_attention": {"query": {"unexpected": jnp.zeros((4, 2))}}}}}
    with self.assertRaises(ValueError):
      apply_lora_on_base_params_nnx(base, bad)


class TestLinenLoraRegression(unittest.TestCase):
  """Smoke tests for the Linen apply / unapply helpers (no other unit test exercises them)."""

  def _linen_pair(self, rng=None):
    """Build a Linen-shape (with `{"params": ...}` outer wrapper) base + lora pair."""
    if rng is None:
      rng = jax.random.key(99)
    base_inner = _concrete_base(rng)
    base = {"params": jax.tree_util.tree_map(jnp.copy, base_inner)}
    lora_inner = _build_lora_params(
        base_inner,
        _lora_config(rank=2, target_modules=("q_proj", "v_proj")),
        jax.random.key(7),
    )
    lora = {"params": lora_inner}
    return base, lora

  def test_linen_apply_then_unapply_is_identity(self):
    base, lora = self._linen_pair()
    base_orig = jax.tree_util.tree_map(jnp.copy, base)
    apply_lora_on_base_params(base, lora, lora_scale_factor=0.5)
    unapply_lora_from_base_params(base, lora, lora_scale_factor=0.5)
    np.testing.assert_allclose(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["value"]["kernel"]),
        rtol=1e-5,
        atol=1e-6,
    )

  def test_linen_apply_only_modifies_target_modules(self):
    base, lora = self._linen_pair()
    base_orig = jax.tree_util.tree_map(jnp.copy, base)
    apply_lora_on_base_params(base, lora, lora_scale_factor=1.0)
    # query and value are targets and must change.
    self.assertFalse(
        jnp.allclose(
            base["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"],
            base_orig["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"],
        )
    )
    # key and out are non-target and must be untouched.
    np.testing.assert_array_equal(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["key"]["kernel"]),
    )
    np.testing.assert_array_equal(
        np.asarray(base["params"]["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
        np.asarray(base_orig["params"]["decoder"]["layers"]["self_attention"]["out"]["kernel"]),
    )


class TestShardingExtractionNnx(unittest.TestCase):
  """Test sharding parameter extraction under different LoRA configurations."""

  def test_sharding_extracts_only_lora_params(self):
    class ToyModel(nnx.Module):

      def __init__(self):
        self.p = nnx.Param(jnp.ones((2, 2)))
        self.lora_p = nnx.LoRAParam(jnp.zeros((2, 2)))

    class DummyConfig:

      class DummyLora:
        enable_lora = True

      lora = DummyLora()
      shard_optimizer_over_data = False
      pure_nnx = True

    model = ToyModel()
    _, state = nnx.split(model)

    class DummyStateMeshShardings:
      model = state

    prev_params, _ = sharding.maybe_update_params_sharding_with_opt(DummyConfig(), DummyStateMeshShardings())
    self.assertIn("lora_p", prev_params)
    self.assertNotIn("p", prev_params)


class TestRestoreLoraNnx(unittest.TestCase):
  """Unit tests for lora_utils.restore_lora_from_path under NNX."""

  def test_restore_lora_from_standalone_and_full_trainstate(self):
    # 1. Setup a toy model with LoRA parameters
    class ToyModel(nnx.Module):

      def __init__(self):
        self.p = nnx.Param(jnp.ones((2, 2)))
        self.lora_p = nnx.LoRAParam(jnp.zeros((2, 2)))

    class DummyLoraConfig:
      enable_lora = True
      lora_restore_path = "dummy/restore/path"
      lora_rank = 4
      lora_alpha = 8.0

    class DummyConfig:
      lora = DummyLoraConfig()
      scan_layers = False

    model = ToyModel()

    # 2. Mocking Orbax restore for standalone LoRA checkpoint
    standalone_restored_state = {"lora_p": {"value": jnp.ones((2, 2)) * 5.0}}

    with mock.patch(
        "orbax.checkpoint.PyTreeCheckpointer.restore",
        return_value=standalone_restored_state,
    ) as mock_restore:
      lora_utils.restore_lora_from_path(model, DummyConfig())
      mock_restore.assert_called_once()
      # Verify that lora_p value was updated to 5.0
      np.testing.assert_allclose(np.asarray(model.lora_p[...]), 5.0)

    # Reset lora_p value
    model.lora_p[...] = jnp.zeros((2, 2))

    # 3. Mocking Orbax restore for full TrainState checkpoint (dict with "model")
    full_trainstate_dict = {
        "model": {"lora_p": {"value": jnp.ones((2, 2)) * 10.0}},
        "optimizer": {},
    }
    with mock.patch(
        "orbax.checkpoint.PyTreeCheckpointer.restore",
        return_value=full_trainstate_dict,
    ) as mock_restore:
      lora_utils.restore_lora_from_path(model, DummyConfig())
      mock_restore.assert_called_once()
      # Verify that lora_p value was updated to 10.0
      np.testing.assert_allclose(np.asarray(model.lora_p[...]), 10.0)

    # Reset lora_p value
    model.lora_p[...] = jnp.zeros((2, 2))

    # 4. Mocking Orbax restore for full TrainState checkpoint (object with .model attribute)
    class DummyTrainState:

      def __init__(self):
        self.model = {"lora_p": {"value": jnp.ones((2, 2)) * 15.0}}

    dummy_state_obj = DummyTrainState()
    with mock.patch("orbax.checkpoint.PyTreeCheckpointer.restore", return_value=dummy_state_obj) as mock_restore:
      lora_utils.restore_lora_from_path(model, DummyConfig())
      mock_restore.assert_called_once()
      # Verify that lora_p value was updated to 15.0
      np.testing.assert_allclose(np.asarray(model.lora_p[...]), 15.0)

    # 5. Mocking Orbax restore for checkpoint missing LoRA parameters (raises ValueError)
    base_only_checkpoint = {
        "p": {"value": jnp.ones((2, 2)) * 20.0},
    }
    with mock.patch("orbax.checkpoint.PyTreeCheckpointer.restore", return_value=base_only_checkpoint) as mock_restore:
      with self.assertRaises(ValueError) as context:
        lora_utils.restore_lora_from_path(model, DummyConfig())
      self.assertIn("No LoRA/adapter parameters were successfully restored", str(context.exception))
      mock_restore.assert_called_once()


class TestLoraResumeAndConversion(unittest.TestCase):
  """Tests for setup_initial_state on LoRA resume path and upfront conversion."""

  def test_setup_initial_state_lora_resume(self):
    """Test setup_initial_state on LoRA resume path."""
    # Setup mock config
    config = mock.MagicMock()
    config.pure_nnx = True
    config.lora = mock.MagicMock()
    config.lora.enable_lora = True
    config.load_parameters_path = "mock/base_params_path"
    config.load_full_state_path = ""
    config.checkpoint_storage_concurrent_gb = 8
    config.checkpoint_storage_use_ocdbt = False
    config.checkpoint_storage_use_zarr3 = False

    # Mock State with model containing nnx.Param and nnx.LoRAParam
    class MockModel(nnx.Module):

      def __init__(self):
        self.kernel = nnx.Param(jax.numpy.ones((2, 2)))
        self.lora_a = nnx.LoRAParam(jax.numpy.zeros((2, 2)))

    class MockTrainState(nnx.Module):

      def __init__(self, model):
        self.model = model

    mock_model = MockModel()
    state_mesh_shardings = mock.MagicMock()

    # init_state_fn mock returning a proper nnx.Module container
    def init_state_fn():
      return MockTrainState(mock_model)

    # Simulating Orbax restoration behavior: missing base weights are restored as ShapeDtypeStructs
    restored_pure_dict = nnx.state(MockTrainState(mock_model), nnx.LoRAParam).to_pure_dict()
    restored_pure_dict["model"] = {
        "kernel": jax.ShapeDtypeStruct((2, 2), jax.numpy.float32),
        "lora_a": restored_pure_dict["model"]["lora_a"],
    }
    restored = {"items": nnx.State(restored_pure_dict)}

    # Mock checkpointing.load_params_from_path to return updated base params
    loaded_base = nnx.State({"kernel": nnx.Param(jax.numpy.ones((2, 2)) * 5.0)})

    with (
        mock.patch(
            "maxtext.utils.maxtext_utils.get_abstract_state",
            return_value=(mock.MagicMock(), mock.MagicMock(), state_mesh_shardings),
        ),
        mock.patch("maxtext.common.checkpointing.load_state_if_possible", return_value=(restored, None)),
        mock.patch("maxtext.common.checkpointing.load_params_from_path", return_value=loaded_base) as mock_load,
        mock.patch("jax.jit", side_effect=lambda f, *args, **kwargs: f),
    ):
      state, _, _, _, was_restored = maxtext_utils.setup_initial_state(None, config, None, None, init_state_fn, True)
      # The base model should be restored, meaning its kernel has updated value of 5.0
      mock_load.assert_called_once()
      self.assertTrue(was_restored)
      self.assertEqual(state.model.kernel.get_value()[0, 0], 5.0)

  def test_setup_initial_state_upfront_lora_conversion(self):
    """Test upfront conversion of quantized base weights on LoRA path."""
    config = mock.MagicMock()
    config.pure_nnx = True
    config.lora = mock.MagicMock()
    config.lora.enable_lora = True
    config.load_parameters_path = ""
    config.load_full_state_path = ""
    config.checkpoint_storage_concurrent_gb = 8
    config.checkpoint_storage_use_ocdbt = False
    config.checkpoint_storage_use_zarr3 = False

    # Mock variables representing quantized weights as nnx.State
    quantized_state = nnx.State({"qvalue": jax.numpy.ones((2, 2)), "scale": jax.numpy.ones((1, 2))})

    class MockModel(nnx.Module):

      def __init__(self):
        # Initial value of Param holds the State
        self.kernel = nnx.Param(quantized_state)

    class MockTrainState(nnx.Module):

      def __init__(self, model):
        self.model = model

    mock_model = MockModel()
    state_mesh_shardings = mock.MagicMock()

    # init_state_fn mock returning a proper nnx.Module container
    def init_state_fn():
      return MockTrainState(mock_model)

    with (
        mock.patch(
            "maxtext.utils.maxtext_utils.get_abstract_state",
            return_value=(mock.MagicMock(), mock.MagicMock(), state_mesh_shardings),
        ),
        mock.patch("maxtext.common.checkpointing.load_state_if_possible", return_value=(None, None)),
        mock.patch("jax.jit", side_effect=lambda f, *args, **kwargs: f),
    ):
      state, _, _, _, _ = maxtext_utils.setup_initial_state(None, config, None, None, init_state_fn, True)

    # Assert kernel value is converted from nnx.State to QArray
    self.assertTrue(isinstance(state.model.kernel.get_value(), QArray))


if __name__ == "__main__":
  unittest.main()
