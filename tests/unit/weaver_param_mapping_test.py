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

"""Unit and roundtrip tests for WeaverOmniTransformer checkpoint conversion and parameter mapping."""

import json
import os
import tempfile
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion import to_maxtext
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import utils as conversion_utils
from maxtext.models import weaver
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import numpy as np
import orbax.checkpoint as ocp
import pytest
from safetensors.numpy import save_file as save_safetensors


def build_mini_weaver_hf_config(
    *,
    num_hidden_layers: int = 1,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    head_dim: int = 16,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    in_channels: int = 12,
    patch_size: int = 2,
    time_embed_in_channels: int = 32,
) -> dict:
  """Builds a mini HuggingFace / Diffusers Weaver configuration dictionary."""
  return {
      "architectures": ["WeaverOmniTransformer2DModel"],
      "model_type": "weaver_omni",
      "num_hidden_layers": num_hidden_layers,
      "hidden_size": hidden_size,
      "num_attention_heads": num_attention_heads,
      "num_key_value_heads": num_key_value_heads,
      "head_dim": head_dim,
      "intermediate_size": intermediate_size,
      "vocab_size": vocab_size,
      "in_channels": in_channels,
      "patch_size": patch_size,
      "time_embed_in_channels": time_embed_in_channels,
      "timestep_scale": 0.001,
      "rms_norm_eps": 1e-6,
      "hidden_act": "silu",
      "attention_bias": False,
      "qk_norm_for_text": True,
      "qk_norm_for_diffusion": True,
      "use_und_k_norm_for_gen": False,
      "mrope_section": [4, 2, 2],
      "rope_theta": 5000000.0,
  }


def generate_synthetic_weaver_hf_weights(hf_cfg: dict, seed: int = 42) -> dict[str, np.ndarray]:
  """Generates synthetic HuggingFace safetensors weights with PyTorch (out_dim, in_dim) shapes."""
  rng = np.random.default_rng(seed)
  shape_map = hf_shape.WEAVER_HF_WEIGHTS_TO_SHAPE(hf_cfg)
  use_und_k_norm_for_gen = hf_cfg.get("use_und_k_norm_for_gen", False)

  weights = {}
  for hf_key, shape in shape_map.items():
    if hf_key == "lm_head.weight":
      continue
    if not use_und_k_norm_for_gen and "k_norm_und_for_gen" in hf_key:
      continue
    weights[hf_key] = rng.standard_normal(size=tuple(shape)).astype(np.float32)
  return weights


def slice_or_create_mini_weaver_safetensors(
    output_dir: str,
    num_layers: int = 1,
    source_dir: str | None = None,
    hf_cfg: dict | None = None,
) -> dict[str, np.ndarray]:
  """Slices a full Weaver safetensors checkpoint to `num_layers` or synthesizes one."""
  os.makedirs(output_dir, exist_ok=True)
  if hf_cfg is None:
    hf_cfg = build_mini_weaver_hf_config(num_hidden_layers=num_layers)
  else:
    hf_cfg = dict(hf_cfg)
    hf_cfg["num_hidden_layers"] = num_layers

  if source_dir is not None and os.path.isdir(source_dir):
    full_state = conversion_utils.load_hf_dict_from_safetensors(source_dir, token="", revision=None, framework="np")
    valid_prefixes = tuple(f"layers.{i}." for i in range(num_layers))
    sliced_weights = {
        k: np.asarray(v) for k, v in full_state.items() if not k.startswith("layers.") or k.startswith(valid_prefixes)
    }
  else:
    sliced_weights = generate_synthetic_weaver_hf_weights(hf_cfg)

  save_safetensors(sliced_weights, os.path.join(output_dir, "model.safetensors"))
  with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(hf_cfg, f, indent=2)
  return sliced_weights


@pytest.mark.cpu_only
class WeaverParamMappingTest(unittest.TestCase):
  """Tests for WeaverOmniTransformer parameter mappings, hooks, and Orbax conversion."""

  def test_unscanned_param_mapping_covers_all_weaver_weights(self):
    """Verifies unscanned mapping covers all understanding, generation, norm, and modality weights."""
    hf_cfg = build_mini_weaver_hf_config(num_hidden_layers=2)
    mt_cfg = weaver.WeaverConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        latent_channels=12,
        patch_size=2,
        time_embed_in_channels=32,
        mrope_section=(4, 2, 2),
        scan_layers=False,
    )
    mt_cfg.decoder_block = "weaver"

    mapping = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=False)

    # Top-level modality & normalization weights
    self.assertEqual(mapping["params-embed_tokens-embedding"], "embed_tokens.weight")
    self.assertEqual(mapping["params-norm-scale"], "norm.weight")
    self.assertEqual(mapping["params-norm_moe_gen-scale"], "norm_moe_gen.weight")
    self.assertEqual(mapping["params-proj_in-kernel"], "proj_in.weight")
    self.assertEqual(mapping["params-proj_in-bias"], "proj_in.bias")
    self.assertEqual(mapping["params-proj_out-kernel"], "proj_out.weight")
    self.assertEqual(mapping["params-proj_out-bias"], "proj_out.bias")
    self.assertEqual(mapping["params-time_embedder-mlp_0-kernel"], "time_embedder.linear_1.weight")
    self.assertEqual(mapping["params-time_embedder-mlp_0-bias"], "time_embedder.linear_1.bias")
    self.assertEqual(mapping["params-time_embedder-mlp_2-kernel"], "time_embedder.linear_2.weight")
    self.assertEqual(mapping["params-time_embedder-mlp_2-bias"], "time_embedder.linear_2.bias")

    # Per-layer generation & understanding weights
    for layer_idx in range(2):
      prefix = f"params-layers_{layer_idx}"
      hf_prefix = f"layers.{layer_idx}"
      # Generation attention projections
      self.assertEqual(mapping[f"{prefix}-self_attn-q_proj_gen-kernel"], f"{hf_prefix}.self_attn.add_q_proj.weight")
      self.assertEqual(mapping[f"{prefix}-self_attn-k_proj_gen-kernel"], f"{hf_prefix}.self_attn.add_k_proj.weight")
      self.assertEqual(mapping[f"{prefix}-self_attn-v_proj_gen-kernel"], f"{hf_prefix}.self_attn.add_v_proj.weight")
      self.assertEqual(mapping[f"{prefix}-self_attn-o_proj_gen-kernel"], f"{hf_prefix}.self_attn.to_add_out.weight")
      # Generation QK norms
      self.assertEqual(mapping[f"{prefix}-self_attn-q_norm_gen-scale"], f"{hf_prefix}.self_attn.norm_added_q.weight")
      self.assertEqual(mapping[f"{prefix}-self_attn-k_norm_gen-scale"], f"{hf_prefix}.self_attn.norm_added_k.weight")
      # Generation layer norms
      self.assertEqual(mapping[f"{prefix}-input_layernorm_moe_gen-scale"], f"{hf_prefix}.input_layernorm_moe_gen.weight")
      self.assertEqual(
          mapping[f"{prefix}-post_attention_layernorm_moe_gen-scale"],
          f"{hf_prefix}.post_attention_layernorm_moe_gen.weight",
      )
      # Generation MLP
      self.assertEqual(mapping[f"{prefix}-mlp_moe_gen-gate_proj-kernel"], f"{hf_prefix}.mlp_moe_gen.gate_proj.weight")
      self.assertEqual(mapping[f"{prefix}-mlp_moe_gen-up_proj-kernel"], f"{hf_prefix}.mlp_moe_gen.up_proj.weight")
      self.assertEqual(mapping[f"{prefix}-mlp_moe_gen-down_proj-kernel"], f"{hf_prefix}.mlp_moe_gen.down_proj.weight")

      # Verify exact match with WeaverOmniTransformer parameter tree
    abstract_model = nnx.eval_shape(lambda: weaver.WeaverOmniTransformer(mt_cfg, rngs=nnx.Rngs(0)))
    abstract_tree = {"params": nnx.state(abstract_model, nnx.Param).to_pure_dict()}
    flat_leaves, _ = jax.tree_util.tree_flatten_with_path(abstract_tree)
    model_keys = {"-".join(conversion_utils.param_key_parts_from_path(p)) for p, _ in flat_leaves}

    self.assertSetEqual(set(mapping.keys()), model_keys)

  def test_scanned_param_mapping_covers_all_weaver_weights(self):
    """Verifies scanned mapping (`scan_layers=True`) matches WeaverOmniTransformer scanned parameter tree."""
    hf_cfg = build_mini_weaver_hf_config(num_hidden_layers=2)
    mt_cfg = weaver.WeaverConfig(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        latent_channels=12,
        patch_size=2,
        time_embed_in_channels=32,
        mrope_section=(4, 2, 2),
        scan_layers=True,
    )
    mt_cfg.decoder_block = "weaver"

    mapping = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=True)
    self.assertEqual(
        mapping["params-scanned_layers-self_attn-q_proj_gen-kernel"],
        ["layers.0.self_attn.add_q_proj.weight", "layers.1.self_attn.add_q_proj.weight"],
    )
    self.assertEqual(
        mapping["params-scanned_layers-mlp_moe_gen-down_proj-kernel"],
        ["layers.0.mlp_moe_gen.down_proj.weight", "layers.1.mlp_moe_gen.down_proj.weight"],
    )

    abstract_model = nnx.eval_shape(lambda: weaver.WeaverOmniTransformer(mt_cfg, rngs=nnx.Rngs(0)))
    abstract_tree = {"params": nnx.state(abstract_model, nnx.Param).to_pure_dict()}
    flat_leaves, _ = jax.tree_util.tree_flatten_with_path(abstract_tree)
    model_keys = {"-".join(conversion_utils.param_key_parts_from_path(p)) for p, _ in flat_leaves}

    self.assertSetEqual(set(mapping.keys()), model_keys)

  def test_weight_transposition_hooks_and_roundtrip(self):
    """Verifies PyTorch (out_dim, in_dim) <-> Flax (in_dim, out_dim) transpositions and roundtrip."""
    for scan_layers in (False, True):
      with self.subTest(scan_layers=scan_layers):
        hf_cfg = build_mini_weaver_hf_config(num_hidden_layers=2)
        mt_cfg = weaver.WeaverConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
            latent_channels=12,
            patch_size=2,
            time_embed_in_channels=32,
            mrope_section=(4, 2, 2),
            scan_layers=scan_layers,
            param_scan_axis=1,
            weight_dtype="float32",
        )
        mt_cfg.decoder_block = "weaver"
        mt_cfg.base_num_decoder_layers = 2

        hf_weights = generate_synthetic_weaver_hf_weights(hf_cfg, seed=123)
        param_map = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, mt_cfg, scan_layers=scan_layers)
        hooks_to_mt = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN(
            hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=False
        )
        hooks_to_hf = param_mapping.WEAVER_MAXTEXT_TO_HF_PARAM_HOOK_FN(
            hf_cfg, mt_cfg, scan_layers=scan_layers, saving_to_hf=True
        )
        hf_shape_map = hf_shape.WEAVER_HF_WEIGHTS_TO_SHAPE(hf_cfg)

        abstract_model = nnx.eval_shape(lambda cfg=mt_cfg: weaver.WeaverOmniTransformer(cfg, rngs=nnx.Rngs(0)))
        abstract_tree = {"params": nnx.state(abstract_model, nnx.Param).to_pure_dict()}
        flat_leaves, _ = jax.tree_util.tree_flatten_with_path(abstract_tree)
        mt_shapes = {"-".join(conversion_utils.param_key_parts_from_path(p)): leaf.shape for p, leaf in flat_leaves}

        # Convert HF -> MaxText and verify shapes + transposition
        mt_weights = {}
        for mt_key, hf_source in param_map.items():
          target_shape = mt_shapes[mt_key]
          load_fn = to_maxtext._get_hf_loading_function(  # pylint: disable=protected-access
              hf_source,
              hf_weights.__getitem__,
              hooks_to_mt.get(mt_key),
              target_shape,
              mt_cfg,
              mt_key,
          )
          mt_arr = load_fn()
          self.assertEqual(mt_arr.shape, target_shape, msg=f"Shape mismatch for {mt_key}")
          self.assertFalse(np.isnan(mt_arr).any(), msg=f"NaN detected in {mt_key}")
          mt_weights[mt_key] = jnp.asarray(mt_arr)

        # Verify explicit transposition on non-square proj_in: HF (64, 48) -> Flax (48, 64)
        np.testing.assert_array_equal(np.asarray(mt_weights["params-proj_in-kernel"]), hf_weights["proj_in.weight"].T)
        np.testing.assert_array_equal(np.asarray(mt_weights["params-proj_out-kernel"]), hf_weights["proj_out.weight"].T)
        np.testing.assert_array_equal(
            np.asarray(mt_weights["params-time_embedder-mlp_0-kernel"]),
            hf_weights["time_embedder.linear_1.weight"].T,
        )

        # Roundtrip MaxText -> HF and verify exact numerical equality for every parameter
        recovered_hf_weights = {}
        for mt_key, mt_val in mt_weights.items():
          pairs = conversion_utils.process_maxtext_param(
              mt_key,
              mt_val,
              param_map,
              hooks_to_hf,
              hf_shape_map,
              mt_cfg,
          )
          for hf_k, hf_v in pairs:
            recovered_hf_weights[hf_k] = hf_v

        self.assertSetEqual(set(recovered_hf_weights.keys()), set(hf_weights.keys()))
        for hf_k, orig_arr in hf_weights.items():
          np.testing.assert_array_equal(recovered_hf_weights[hf_k], orig_arr, err_msg=f"Roundtrip mismatch for {hf_k}")

  def test_mini_1layer_safetensors_to_orbax_conversion(self):
    """Converts a mini 1-layer safetensors checkpoint to Orbax without unmapped warnings or NaNs."""
    hf_cfg = build_mini_weaver_hf_config(
        num_hidden_layers=1,
        in_channels=48,
        patch_size=2,
        time_embed_in_channels=256,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      hf_dir = os.path.join(tmpdir, "hf_mini_1layer")
      orbax_dir = os.path.join(tmpdir, "orbax_out")
      sliced_weights = slice_or_create_mini_weaver_safetensors(hf_dir, num_layers=1, hf_cfg=hf_cfg)

      mock_hf_cfg_obj = mock.Mock()
      mock_hf_cfg_obj.to_dict.return_value = hf_cfg

      args = [
          "",
          os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml"),
          "model_name=weaver-mini-diffuser",
          "override_model_config=True",
          f"base_output_directory={orbax_dir}",
          "run_name=mini_weaver_test",
          "hardware=cpu",
          "skip_jax_distributed_system=True",
          "scan_layers=False",
          "base_num_decoder_layers=1",
          "base_emb_dim=64",
          "base_num_query_heads=4",
          "base_num_kv_heads=2",
          "head_dim=16",
          "base_mlp_dim=128",
          "vocab_size=256",
          "mrope_section=[4,2,2]",
          "weight_dtype=float32",
          "dtype=float32",
      ]

      logged_messages = []
      orig_log = to_maxtext.max_logging.log

      def _capturing_log(msg, *a, **kw):
        logged_messages.append(str(msg))
        return orig_log(msg, *a, **kw)

      with (
          mock.patch.dict(to_maxtext.HF_MODEL_CONFIGS, {"weaver-mini-diffuser": mock_hf_cfg_obj}),
          mock.patch.object(to_maxtext.max_logging, "log", side_effect=_capturing_log),
          mock.patch.object(conversion_utils.max_logging, "log", side_effect=_capturing_log),
      ):
        to_maxtext.main(
            args=args,
            lazy_load_tensors=False,
            eager_load_method="safetensors",
            hf_model_path=hf_dir,
            save_dtype="float32",
            simulated_cpu_devices_count=1,
        )

      # Verify zero unmapped / extra parameter warnings
      warning_msgs = [m for m in logged_messages if "extra keys in param_map are skipped" in m]
      self.assertEqual(warning_msgs, [], msg=f"Unexpected unmapped parameter warnings: {warning_msgs}")

      # Restore Orbax checkpoint and verify zero NaNs and exact weight match
      ckpt_path = os.path.join(orbax_dir, "0", "items")
      self.assertTrue(os.path.exists(ckpt_path), msg=f"Expected Orbax checkpoint at {ckpt_path}")
      checkpointer = ocp.PyTreeCheckpointer()
      restored = checkpointer.restore(ckpt_path)
      extracted_mt = conversion_utils.detect_and_extract_checkpoint(restored)

      self.assertGreater(len(extracted_mt), 0)
      for k, arr in extracted_mt.items():
        self.assertFalse(np.isnan(np.asarray(arr)).any(), msg=f"NaN found in restored Orbax weight {k}")

      # Verify transposed weights in restored Orbax checkpoint match original safetensors
      np.testing.assert_allclose(
          np.asarray(extracted_mt["params-proj_in-kernel"]),
          sliced_weights["proj_in.weight"].T,
          rtol=1e-6,
          atol=1e-6,
      )
      np.testing.assert_allclose(
          np.asarray(extracted_mt["params-layers_0-self_attn-q_proj_gen-kernel"]),
          sliced_weights["layers.0.self_attn.add_q_proj.weight"].T,
          rtol=1e-6,
          atol=1e-6,
      )
      np.testing.assert_allclose(
          np.asarray(extracted_mt["params-layers_0-mlp_moe_gen-down_proj-kernel"]),
          sliced_weights["layers.0.mlp_moe_gen.down_proj.weight"].T,
          rtol=1e-6,
          atol=1e-6,
      )


if __name__ == "__main__":
  unittest.main()
