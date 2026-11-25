# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests gdn_mamba_block_size and vLLM hybrid model config registration."""

from types import SimpleNamespace
import unittest

from maxtext.configs import types as config_types
from maxtext.integration.vllm import hybrid_cache_utils


class GdnMambaBlockSizeFieldTest(unittest.TestCase):
  """`gdn_mamba_block_size` must be a declared config field."""

  def test_field_is_declared(self):
    # pyconfig._prepare_for_pydantic validates every YAML/CLI key against
    # model_fields, so the adapter's override raises
    # "'gdn_mamba_block_size' not in ..." unless the field is declared here,
    # and the server never starts.
    self.assertIn("gdn_mamba_block_size", config_types.Qwen3Next.model_fields)

  def test_default_means_no_prefix_caching(self):
    # Training, MaxEngine and vLLM non-align serving never set it.
    self.assertEqual(config_types.Qwen3Next.model_fields["gdn_mamba_block_size"].default, 0)


class InstallTpuHybridCoordinatorTest(unittest.TestCase):
  """Tests TPUHybridKVCacheCoordinator hook installation helper."""

  def test_maybe_install_tpu_hybrid_coordinator_installs_hooks_and_skips_when_unavailable(
      self,
  ):
    self.assertFalse(hybrid_cache_utils.maybe_install_tpu_hybrid_coordinator(None))

    vllm_cfg = SimpleNamespace(additional_config={"_mamba_num_blocks": 9})

    def missing_tpu_inference(_):
      raise ImportError("No module named 'tpu_inference'")

    self.assertFalse(hybrid_cache_utils.maybe_install_tpu_hybrid_coordinator(vllm_cfg, missing_tpu_inference))

    older_mod = SimpleNamespace()
    self.assertFalse(hybrid_cache_utils.maybe_install_tpu_hybrid_coordinator(vllm_cfg, lambda _: older_mod))

    installed: list[object] = []
    coord_mod = SimpleNamespace(maybe_install_hybrid_coordinator_hooks=installed.append)
    self.assertTrue(hybrid_cache_utils.maybe_install_tpu_hybrid_coordinator(vllm_cfg, lambda _: coord_mod))
    self.assertEqual(installed, [vllm_cfg])


class IsHybridHfConfigTest(unittest.TestCase):
  """`is_hybrid_hf_config` is model-agnostic."""

  def test_layer_types_with_recurrent_layers(self):
    # Qwen3-Next / Qwen3.5 style.
    self.assertTrue(
        hybrid_cache_utils.is_hybrid_hf_config(
            SimpleNamespace(model_type="qwen3_5_moe_text", layer_types=["linear_attention"] * 3 + ["full_attention"])
        )
    )
    # Unknown model type: detection does not depend on the model family.
    self.assertTrue(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=["mamba", "attention"])))

  def test_layers_block_type(self):
    # Jamba / NemotronH style.
    self.assertTrue(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layers_block_type=["mamba", "attention"])))
    self.assertTrue(
        hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layers_block_type=["mamba", "mlp", "attention", "moe"]))
    )
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layers_block_type=["attention", "attention"]))
    )
    # MLP/MoE entries are not recurrent layers.
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layers_block_type=["attention", "mlp", "moe"]))
    )

  def test_attn_type_list(self):
    # MiniMax style: 0 = linear attention, 1 = softmax attention.
    self.assertTrue(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(attn_type_list=[0, 0, 1])))
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(attn_type_list=[1, 1])))
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(attn_type_list=[0, 0])))

  def test_attention_only_models_are_not_hybrid(self):
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(None))
    # Llama: no per-layer metadata.
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(model_type="llama")))
    # Gemma: sliding + full attention.
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=["sliding_attention", "full_attention"]))
    )
    # granite-4.0-micro: hybrid config class, but attention-only layers.
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=["attention", "attention"])))
    # Sparse attention still uses a full KV cache (vLLM's _ATTENTION_LAYER_TYPES).
    for spelling in ("deepseek_sparse_attention", "qwen_sparse_attention", "indexed_attention"):
      self.assertFalse(
          hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=[spelling, "full_attention"])), spelling
      )
    # An attention spelling this module has never seen is not mistaken for a recurrent layer.
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=["future_attention", "full_attention"]))
    )

  def test_recurrent_only_models_are_not_hybrid(self):
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(layer_types=["mamba", "mamba"])))

  def test_full_attention_interval_without_layer_types(self):
    # Qwen3-Next family loaded as a generic PretrainedConfig (e.g. qwen4_exp on transformers < 5.8, or
    # qwen3.8-2.4t-a95b without Qwen3_5MoeTextConfig): only full_attention_interval is set.
    self.assertTrue(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(full_attention_interval=4)))
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(full_attention_interval=1)))
    # Materialized layer_types take precedence.
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(
            SimpleNamespace(full_attention_interval=4, layer_types=["full_attention", "full_attention"])
        )
    )

  def test_kimi_linear_attn_config(self):
    self.assertTrue(
        hybrid_cache_utils.is_hybrid_hf_config(
            SimpleNamespace(linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4]})
        )
    )
    self.assertFalse(
        hybrid_cache_utils.is_hybrid_hf_config(
            SimpleNamespace(linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2]})
        )
    )
    self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(SimpleNamespace(linear_attn_config=None)))


class GetHfTextConfigTest(unittest.TestCase):
  """`get_hf_text_config` unwraps multimodal configs."""

  def test_unwraps_hf_config_text_config(self):
    # Qwen3.5-VL style: the text model's layer_types live on hf_config.text_config.
    text = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    model_config = SimpleNamespace(hf_config=SimpleNamespace(text_config=text))
    self.assertIs(hybrid_cache_utils.get_hf_text_config(model_config), text)


class ValidateHybridDetectionTest(unittest.TestCase):
  """The HF config and the MaxText decoder block must agree on hybrid-ness."""

  _QWEN_HF = SimpleNamespace(model_type="qwen3_5_moe_text", layer_types=["linear_attention", "full_attention"])
  _LLAMA_HF = SimpleNamespace(model_type="llama")

  def test_consistent_configs_pass(self):
    hybrid_cache_utils.validate_hybrid_detection("qwen3_5", self._QWEN_HF)
    hybrid_cache_utils.validate_hybrid_detection(SimpleNamespace(value="qwen3_next"), self._QWEN_HF)
    hybrid_cache_utils.validate_hybrid_detection("llama2", self._LLAMA_HF)

  def test_hybrid_decoder_block_with_non_hybrid_hf_config_raises(self):
    # E.g. a checkpoint whose config.json the HF detection does not understand.
    with self.assertRaisesRegex(ValueError, "does not describe a hybrid model"):
      hybrid_cache_utils.validate_hybrid_detection("qwen3_5", SimpleNamespace(model_type="qwen4_exp"))

  def test_hybrid_hf_config_with_unsupported_decoder_block_raises(self):
    with self.assertRaisesRegex(ValueError, "no hybrid KV cache support"):
      hybrid_cache_utils.validate_hybrid_detection("llama2", self._QWEN_HF)


class HybridModelConfigRegistrationTest(unittest.TestCase):
  """Tests MaxTextForCausalLM registration in vLLM's MODELS_CONFIG_MAP."""

  def setUp(self):
    super().setUp()
    self.calls = []
    calls = self.calls

    class DummyVerifyBase:
      pass

    class DummyHybridVerifier:

      @staticmethod
      def verify_and_update_config(vllm_config):
        calls.append(vllm_config)

    self.verify_base = DummyVerifyBase
    self.models_map = {}
    self.assertTrue(
        hybrid_cache_utils.register_maxtext_hybrid_model_config(self.models_map, DummyVerifyBase, DummyHybridVerifier)
    )
    self.cfg_cls = self.models_map["MaxTextForCausalLM"]

  def _vllm_config(self, hf_text_config, is_hybrid=False):
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf_text_config, is_hybrid=is_hybrid))

  def test_registration_is_idempotent(self):
    self.assertTrue(issubclass(self.cfg_cls, self.verify_base))
    self.assertFalse(hybrid_cache_utils.register_maxtext_hybrid_model_config(self.models_map, object, object))
    self.assertIs(self.models_map["MaxTextForCausalLM"], self.cfg_cls)

  def test_applies_hybrid_hook_to_hybrid_models(self):
    for hf_text_config in (
        SimpleNamespace(layer_types=["linear_attention", "full_attention"]),
        SimpleNamespace(layers_block_type=["mamba", "attention"]),
        SimpleNamespace(attn_type_list=[0, 1]),
    ):
      vllm_config = self._vllm_config(hf_text_config)
      self.cfg_cls.verify_and_update_config(vllm_config)
      self.assertIs(self.calls[-1], vllm_config)
    self.assertEqual(len(self.calls), 3)

  def test_skips_non_hybrid_models(self):
    self.cfg_cls.verify_and_update_config(self._vllm_config(SimpleNamespace(model_type="llama")))
    self.cfg_cls.verify_and_update_config(
        self._vllm_config(SimpleNamespace(layer_types=["sliding_attention", "full_attention"]))
    )
    self.cfg_cls.verify_and_update_config(SimpleNamespace(model_config=None))
    self.assertEqual(self.calls, [])

  def test_skips_when_vllm_already_treats_model_as_hybrid(self):
    # vLLM runs HybridAttentionMambaModelConfig itself in that case.
    self.cfg_cls.verify_and_update_config(
        self._vllm_config(SimpleNamespace(layer_types=["linear_attention", "full_attention"]), is_hybrid=True)
    )
    self.assertEqual(self.calls, [])


if __name__ == "__main__":
  unittest.main()
