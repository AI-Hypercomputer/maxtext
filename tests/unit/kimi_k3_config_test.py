# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi-K3 model config (yml + schema) and HF -> MaxText checkpoint param mapping tests.

Everything except the safetensors-index comparison runs on CPU without the HF reference.
"""

import json
import os
import sys
import unittest

import jax.numpy as jnp
import numpy as np
import pydantic
from transformers import PretrainedConfig

from maxtext.checkpoint_conversion.utils.hf_model_configs import kimi_k3_config, kimi_k3_dict
from maxtext.checkpoint_conversion.utils.hf_shape import HF_SHAPE
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING, VLLM_HOOK_FNS
from maxtext.configs import pyconfig
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, requires_kimi_k3_reference
from tests.utils.test_helpers import get_test_config_path


def _hf_full_attn_layers_0idx() -> list[int]:
  return sorted(i - 1 for i in kimi_k3_dict["linear_attn_config"]["full_attn_layers"])


def _kimi_k3_config(**overrides):
  """Initializes the released kimi-k3 config (CPU, unscanned, unpacked) with `overrides`."""
  kwargs = {
      "run_name": "kimi_k3_config_test",
      "model_name": "kimi-k3",
      "override_model_config": True,
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "max_target_length": 8,
      "per_device_batch_size": 1,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


class KimiK3PyconfigTest(unittest.TestCase):
  """Loads kimi-k3.yml through pyconfig (schema with extra='forbid', types, validators)."""

  def test_yml_loads_and_matches_hf_config(self):
    cfg = _kimi_k3_config()
    self.assertFalse(cfg.scan_layers)
    self.assertEqual(cfg.decoder_block.value, "kimi_linear")
    self.assertEqual(cfg.base_emb_dim, kimi_k3_dict["hidden_size"])
    self.assertEqual(cfg.base_num_decoder_layers, kimi_k3_dict["num_hidden_layers"])
    self.assertEqual(cfg.base_num_query_heads, kimi_k3_dict["num_attention_heads"])
    self.assertEqual(cfg.base_num_kv_heads, kimi_k3_dict["num_key_value_heads"])
    self.assertEqual(cfg.vocab_size, kimi_k3_dict["vocab_size"])
    self.assertEqual(cfg.normalization_layer_epsilon, kimi_k3_dict["rms_norm_eps"])
    for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
      self.assertEqual(getattr(cfg, name), kimi_k3_dict[name], name)
    self.assertEqual(cfg.base_mlp_dim, kimi_k3_dict["intermediate_size"])
    self.assertEqual(cfg.base_moe_mlp_dim, kimi_k3_dict["moe_intermediate_size"])
    self.assertEqual(cfg.moe_intermediate_size, kimi_k3_dict["moe_intermediate_size"])
    self.assertEqual(cfg.shared_experts, kimi_k3_dict["num_shared_experts"])
    self.assertEqual(cfg.num_shared_experts, kimi_k3_dict["num_shared_experts"])
    self.assertEqual(
        cfg.shared_intermediate_size, kimi_k3_dict["moe_intermediate_size"] * kimi_k3_dict["num_shared_experts"]
    )
    self.assertEqual(cfg.routed_expert_hidden_size, kimi_k3_dict["routed_expert_hidden_size"])
    self.assertEqual(cfg.num_experts, kimi_k3_dict["num_experts"])
    self.assertEqual(cfg.num_experts_per_tok, kimi_k3_dict["num_experts_per_token"])
    self.assertEqual(cfg.first_num_dense_layers, kimi_k3_dict["first_k_dense_replace"])
    self.assertEqual(cfg.attn_res_block_size, kimi_k3_dict["attn_res_block_size"])
    self.assertEqual(cfg.short_conv_kernel_size, kimi_k3_dict["linear_attn_config"]["short_conv_kernel_size"])
    self.assertEqual(list(cfg.full_attn_layers), _hf_full_attn_layers_0idx())

  def test_mismatched_alias_pair_is_rejected(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "moe_intermediate_size"):
      _kimi_k3_config(base_moe_mlp_dim=64)
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "num_shared_experts"):
      _kimi_k3_config(shared_experts=1)
    cfg = _kimi_k3_config(base_moe_mlp_dim=64, moe_intermediate_size=64, shared_intermediate_size=128)
    self.assertEqual(cfg.moe_intermediate_size, 64)
    self.assertEqual(cfg.shared_intermediate_size, 128)

  def test_zero_alias_value_is_validated(self):
    """An explicit 0 is a value, not "unset": num_shared_experts=0 must still match shared_experts."""
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "num_shared_experts"):
      _kimi_k3_config(num_shared_experts=0, shared_intermediate_size=0)
    cfg = _kimi_k3_config(num_shared_experts=0, shared_experts=0, shared_intermediate_size=0)
    self.assertEqual(cfg.num_shared_experts, 0)

  def test_shared_intermediate_size_must_match_product(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "shared_intermediate_size"):
      _kimi_k3_config(base_moe_mlp_dim=64, moe_intermediate_size=64)
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "shared_intermediate_size"):
      _kimi_k3_config(shared_intermediate_size=3072)


class KimiK3ParamMappingTest(unittest.TestCase):
  """HF -> MaxText mapping layout against the in-tree HF config (no reference needed)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mapping = PARAM_MAPPING["kimi-k3"](kimi_k3_dict, None, scan_layers=False)
    cls.hooks = HOOK_FNS["kimi-k3"](kimi_k3_dict, None, scan_layers=False, saving_to_hf=False)

  def test_not_registered_for_vllm(self):
    self.assertNotIn("kimi-k3", VLLM_HOOK_FNS)

  def test_scan_layers_is_not_implemented(self):
    with self.assertRaises(NotImplementedError):
      PARAM_MAPPING["kimi-k3"](kimi_k3_dict, None, scan_layers=True)
    with self.assertRaises(NotImplementedError):
      HOOK_FNS["kimi-k3"](kimi_k3_dict, None, scan_layers=True, saving_to_hf=False)

  def test_layer_layout_must_be_given(self):
    with self.assertRaisesRegex(ValueError, "full_attn_layers"):
      PARAM_MAPPING["kimi-k3"]({"num_hidden_layers": 4, "num_experts": 2}, None, scan_layers=False)
    with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
      PARAM_MAPPING["kimi-k3"]({"full_attn_layers": [3], "num_experts": 2}, None, scan_layers=False)
    with self.assertRaisesRegex(ValueError, "out-of-range"):
      PARAM_MAPPING["kimi-k3"]({"num_hidden_layers": 4, "full_attn_layers": [4]}, None, scan_layers=False)

  def test_top_level_layout_is_0_indexed(self):
    mapping = PARAM_MAPPING["kimi-k3"](
        {"num_hidden_layers": 4, "num_experts": 2, "first_k_dense_replace": 1, "full_attn_layers": [3]},
        None,
        scan_layers=False,
    )
    self.assertIn("params-decoder-layers_3-self_attention-wq_a-kernel", mapping)
    self.assertIn("params-decoder-layers_2-self_attention-q_proj-kernel", mapping)

  def test_no_silent_production_defaults(self):
    """Fields that change the key layout must be given; nothing falls back to the 93-layer release."""
    with self.assertRaisesRegex(ValueError, "first_k_dense_replace"):
      PARAM_MAPPING["kimi-k3"]({"num_hidden_layers": 4, "num_experts": 2, "full_attn_layers": [3]}, None)
    with self.assertRaisesRegex(ValueError, "first_k_dense_replace"):
      HOOK_FNS["kimi-k3"]({"num_hidden_layers": 4, "full_attn_layers": [3]}, None)
    with self.assertRaisesRegex(ValueError, "num_experts"):
      PARAM_MAPPING["kimi-k3"]({"num_hidden_layers": 4, "first_k_dense_replace": 1, "full_attn_layers": [3]}, None)
    # An all-dense stack does not need `num_experts`.
    mapping = PARAM_MAPPING["kimi-k3"](
        {"num_hidden_layers": 2, "first_k_dense_replace": 2, "full_attn_layers": [1]}, None
    )
    self.assertIn("params-decoder-layers_1-mlp-wi_0-kernel", mapping)

  def test_global_keys(self):
    expected = {
        "params-token_embedder-embedding": "language_model.model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "language_model.model.norm.weight",
        "params-decoder-logits_dense-kernel": "language_model.lm_head.weight",
        "params-decoder-output_attn_res_norm-scale": "language_model.model.output_attn_res_norm.weight",
        "params-decoder-output_attn_res_proj-kernel": "language_model.model.output_attn_res_proj.weight",
    }
    for mt_key, hf_key in expected.items():
      self.assertEqual(self.mapping[mt_key], hf_key)

  def test_per_layer_keys(self):
    full_attn = set(_hf_full_attn_layers_0idx())
    num_experts = kimi_k3_dict["num_experts"]
    first_dense = kimi_k3_dict["first_k_dense_replace"]
    common = ("pre_self_attention_layer_norm-scale", "post_self_attention_layer_norm-scale")
    common += ("self_attention_res_norm-scale", "self_attention_res_proj-kernel", "mlp_res_norm-scale")
    common += ("mlp_res_proj-kernel",)
    mla = ("wq_a-kernel", "q_norm-scale", "wq_b-kernel", "wkv_a-kernel", "kv_norm-scale", "wkv_b-kernel")
    mla += ("g_proj-kernel", "out-kernel")
    kda = ("q_proj-kernel", "q_conv1d-kernel", "k_proj-kernel", "k_conv1d-kernel", "v_proj-kernel")
    kda += ("v_conv1d-kernel", "f_a_proj-kernel", "f_b_proj-kernel", "A_log", "dt_bias", "b_proj-kernel")
    kda += ("g_proj-kernel", "o_norm-scale", "out-kernel")
    moe = ("routed_experts-gate-kernel", "routed_experts-gate-bias", "routed_expert_down_proj-kernel")
    moe += ("routed_expert_norm-scale", "routed_expert_up_proj-kernel", "shared_expert-wi_0-kernel")
    moe += ("shared_expert-wi_1-kernel", "shared_expert-wo-kernel")

    for i in range(kimi_k3_dict["num_hidden_layers"]):
      prefix = f"params-decoder-layers_{i}"
      for name in common:
        self.assertIn(f"{prefix}-{name}", self.mapping)
      attn_present, attn_absent = (mla, kda) if i in full_attn else (kda, mla)
      for name in attn_present:
        self.assertIn(f"{prefix}-self_attention-{name}", self.mapping)
      for name in set(attn_absent) - set(attn_present):
        self.assertNotIn(f"{prefix}-self_attention-{name}", self.mapping)
      if i < first_dense:
        for name in ("wi_0-kernel", "wi_1-kernel", "wo-kernel"):
          self.assertIn(f"{prefix}-mlp-{name}", self.mapping)
        self.assertNotIn(f"{prefix}-mlp-routed_experts-wi_0", self.mapping)
      else:
        for name in moe:
          self.assertIn(f"{prefix}-mlp-{name}", self.mapping)
        for name in ("wi_0", "wi_1", "wo"):
          self.assertEqual(len(self.mapping[f"{prefix}-mlp-routed_experts-{name}"]), num_experts)
    # Every hook must refer to a mapped key.
    self.assertFalse(set(self.hooks) - set(self.mapping))

  @requires_kimi_k3_reference
  def test_mapping_covers_released_safetensors_index(self):
    """The mapping's HF keys are exactly the released checkpoint's (MXFP4 sidecars folded)."""
    index_file = os.path.join(KIMI_K3_REFERENCE_DIR, "model.safetensors.index.json")
    if not os.path.isfile(index_file):
      self.skipTest(f"{index_file} not found")
    with open(index_file, "r", encoding="utf-8") as f:
      weight_map = json.load(f)["weight_map"]

    index_keys = set()
    for k in weight_map:
      if not k.startswith("language_model.") or k.endswith(".weight_scale"):
        continue
      index_keys.add(k[: -len("_packed")] if k.endswith(".weight_packed") else k)

    mapped_keys = set()
    for hf_val in self.mapping.values():
      mapped_keys.update([hf_val] if isinstance(hf_val, str) else hf_val)

    missing, extra = index_keys - mapped_keys, mapped_keys - index_keys
    self.assertFalse(missing, f"index keys with no mapping: {sorted(missing)[:10]}")
    self.assertFalse(extra, f"mapped keys not in index: {sorted(extra)[:10]}")


class KimiK3ParamHookTest(unittest.TestCase):
  """HF -> MaxText hook shapes on random tensors (MaxText -> HF export is not covered here)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.hooks = HOOK_FNS["kimi-k3"](kimi_k3_dict, None, scan_layers=False, saving_to_hf=False)
    cls.mla = _hf_full_attn_layers_0idx()[0]

  def test_transpose(self):
    x = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(self.hooks["params-decoder-logits_dense-kernel"](x), x.T)

  def test_conv_permute(self):
    x = np.arange(24).reshape(6, 1, 4)  # HF [C, 1, K]
    y = self.hooks["params-decoder-layers_0-self_attention-q_conv1d-kernel"](x, None)
    self.assertEqual(y.shape, (4, 1, 6))  # MaxText [K, 1, C]
    np.testing.assert_array_equal(y[:, 0, :], x[:, 0, :].T)

  def test_stacked_expert_transpose(self):
    x = np.arange(24).reshape(2, 3, 4)
    y = self.hooks["params-decoder-layers_1-mlp-routed_experts-wi_0"](x, None)
    np.testing.assert_array_equal(y, np.transpose(x, (0, 2, 1)))

  def test_mla_reshape_hooks_use_target_shape(self):
    rng = np.random.default_rng(0)
    q_lora, heads, qk_head = 4, 2, 3
    wq_b = rng.standard_normal((heads * qk_head, q_lora))  # HF [out, in]
    y = self.hooks[f"params-decoder-layers_{self.mla}-self_attention-wq_b-kernel"](wq_b, (q_lora, heads, qk_head))
    np.testing.assert_array_equal(y, wq_b.T.reshape(q_lora, heads, qk_head))
    out = rng.standard_normal((5, heads * qk_head))  # HF o_proj [hidden, heads * v_head]
    y = self.hooks[f"params-decoder-layers_{self.mla}-self_attention-out-kernel"](out, (heads, qk_head, 5))
    np.testing.assert_array_equal(y, out.T.reshape(heads, qk_head, 5))

  def test_a_log_padding_is_sliced_only_if_zero(self):
    a_log_hook = self.hooks["params-decoder-layers_0-self_attention-A_log"]
    padded = np.concatenate([np.arange(1, 4, dtype=np.float32), np.zeros(2, np.float32)])
    np.testing.assert_array_equal(a_log_hook(padded, (3,)), padded[:3])
    with self.assertRaises(ValueError):
      a_log_hook(np.ones(5, np.float32), (3,))
    with self.assertRaises(ValueError):
      a_log_hook(np.ones(2, np.float32), (3,))

  def test_a_log_without_target_shape_slices_to_kda_num_heads(self):
    """Without a target shape, the release's [head_dim]-padded A_log is sliced to the KDA num_heads."""
    a_log_hook = self.hooks["params-decoder-layers_0-self_attention-A_log"]
    heads = kimi_k3_dict["linear_attn_config"]["num_heads"]
    head_dim = kimi_k3_dict["linear_attn_config"]["head_dim"]
    padded = np.concatenate([np.arange(1, heads + 1, dtype=np.float32), np.zeros(head_dim - heads, np.float32)])
    np.testing.assert_array_equal(a_log_hook(padded), padded[:heads])

  def test_a_log_is_zero_padded_back_on_export(self):
    hooks = HOOK_FNS["kimi-k3"](kimi_k3_dict, None, scan_layers=False, saving_to_hf=True)
    a_log_hook = hooks["params-decoder-layers_0-self_attention-A_log"]
    heads = kimi_k3_dict["linear_attn_config"]["num_heads"]
    a_log = np.arange(1, heads + 1, dtype=np.float32)
    hf_shape = HF_SHAPE["kimi-k3"](kimi_k3_dict)["language_model.model.layers.0.self_attn.A_log"]
    y = a_log_hook(a_log, hf_shape)
    self.assertEqual(list(y.shape), hf_shape)
    np.testing.assert_array_equal(y[:heads], a_log)
    np.testing.assert_array_equal(y[heads:], 0)
    y = a_log_hook(jnp.asarray(a_log), hf_shape)  # to_huggingface passes jax arrays
    self.assertEqual(list(y.shape), hf_shape)
    np.testing.assert_array_equal(np.asarray(y)[:heads], a_log)
    # Round trip back to MaxText.
    np.testing.assert_array_equal(
        self.hooks["params-decoder-layers_0-self_attention-A_log"](np.asarray(y), (heads,)), a_log
    )


def _released_text_config_nested() -> dict:
  """The released top-level `config.json` layout: LM fields nested under `text_config`."""
  return {
      "architectures": ["KimiK3ForConditionalGeneration"],
      "model_type": "kimi_k3",
      "text_config": dict(kimi_k3_dict),
      "vision_config": {"model_type": "moonvit3d"},
  }


class KimiK3HfConfigInputTest(unittest.TestCase):
  """Mapping/hooks/shapes accept a flat dict, a nested `text_config` dict, or a `PretrainedConfig`."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ref_mapping = PARAM_MAPPING["kimi-k3"](kimi_k3_dict, None, scan_layers=False)
    cls.ref_hooks = HOOK_FNS["kimi-k3"](kimi_k3_dict, None, scan_layers=False, saving_to_hf=False)

  def _assert_same_as_flat(self, config):
    self.assertEqual(PARAM_MAPPING["kimi-k3"](config, None, scan_layers=False), self.ref_mapping)
    hooks = HOOK_FNS["kimi-k3"](config, None, scan_layers=False, saving_to_hf=False)
    self.assertEqual(set(hooks), set(self.ref_hooks))
    for key, fn in hooks.items():
      self.assertEqual(fn.__name__, self.ref_hooks[key].__name__, key)

  def test_nested_text_config_dict(self):
    self._assert_same_as_flat(_released_text_config_nested())

  def test_pretrained_config_object(self):
    self._assert_same_as_flat(kimi_k3_config)
    self._assert_same_as_flat(PretrainedConfig.from_dict(_released_text_config_nested()))

  def test_hf_shape_keys_match_mapping(self):
    mapped = set()
    for hf_val in self.ref_mapping.values():
      mapped.update([hf_val] if isinstance(hf_val, str) else hf_val)
    shapes = HF_SHAPE["kimi-k3"](kimi_k3_dict)
    self.assertEqual(set(shapes), mapped)
    self.assertEqual(HF_SHAPE["kimi-k3"](_released_text_config_nested()), shapes)
    kda = "language_model.model.layers.0.self_attn"
    self.assertEqual(shapes[f"{kda}.A_log"], [128])
    self.assertEqual(shapes[f"{kda}.q_conv1d.weight"], [96 * 128, 1, 4])
    self.assertEqual(shapes["language_model.model.layers.3.self_attn.q_b_proj.weight"], [96 * 192, 1536])

  def test_hf_shape_helpers_match_param_mapping(self):
    """`hf_shape` keeps local copies of the param_mapping config helpers (it must stay maxtext-import-free)."""
    from maxtext.checkpoint_conversion.utils import hf_shape, param_mapping  # pylint: disable=import-outside-toplevel

    with open(hf_shape.__file__, encoding="utf-8") as f:
      src = f.read()
    self.assertNotRegex(src, r"(?m)^\s*(from|import)\s+maxtext")
    for config in (kimi_k3_dict, _released_text_config_nested(), kimi_k3_config):
      self.assertEqual(hf_shape._kimi_k3_hf_text_config(config), param_mapping._kimi_k3_text_config(config))  # pylint: disable=protected-access
      self.assertEqual(
          hf_shape._kimi_k3_hf_full_attn_layers(config),  # pylint: disable=protected-access
          param_mapping._resolve_kimi_k3_full_attn_layers(config),  # pylint: disable=protected-access
      )
    for bad in ({"num_hidden_layers": 4}, {"full_attn_layers": [3]}, {"num_hidden_layers": 4, "full_attn_layers": [4]}):
      with self.assertRaises(ValueError):
        hf_shape._kimi_k3_hf_full_attn_layers(bad)  # pylint: disable=protected-access
      with self.assertRaises(ValueError):
        param_mapping._resolve_kimi_k3_full_attn_layers(bad)  # pylint: disable=protected-access

  @requires_kimi_k3_reference
  def test_released_config_json(self):
    """The real released `config.json` (nested) yields the same mapping/hooks as the flat stand-in."""
    config_file = os.path.join(KIMI_K3_REFERENCE_DIR, "config.json")
    if not os.path.isfile(config_file):
      self.skipTest(f"{config_file} not found")
    with open(config_file, "r", encoding="utf-8") as f:
      released = json.load(f)
    self.assertIn("text_config", released)
    self._assert_same_as_flat(released)
    text_config = released["text_config"]
    for key, value in kimi_k3_dict.items():
      self.assertEqual(text_config.get(key), value, key)


if __name__ == "__main__":
  unittest.main()
