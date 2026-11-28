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

  def test_pure_recurrent_and_has_recurrent_layers_detection(self):
    for hf_cfg in (
        SimpleNamespace(layer_types=["linear_attention", "linear_attention"]),
        SimpleNamespace(layer_types=["mamba", "mamba"]),
        SimpleNamespace(layers_block_type=["mamba", "linear_attention"]),
        SimpleNamespace(attn_type_list=[0, 0]),
        SimpleNamespace(linear_attn_config={"kda_layers": [1, 2], "full_attn_layers": []}),
    ):
      self.assertFalse(hybrid_cache_utils.is_hybrid_hf_config(hf_cfg))
      self.assertTrue(hybrid_cache_utils.is_pure_recurrent_hf_config(hf_cfg))
      self.assertTrue(hybrid_cache_utils.has_recurrent_layers_hf_config(hf_cfg))

    hybrid_cfg = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    self.assertTrue(hybrid_cache_utils.is_hybrid_hf_config(hybrid_cfg))
    self.assertFalse(hybrid_cache_utils.is_pure_recurrent_hf_config(hybrid_cfg))
    self.assertTrue(hybrid_cache_utils.has_recurrent_layers_hf_config(hybrid_cfg))

    for attn_only in (
        None,
        SimpleNamespace(model_type="llama"),
        SimpleNamespace(layer_types=["full_attention", "sliding_attention"]),
        SimpleNamespace(attn_type_list=[1, 1]),
    ):
      self.assertFalse(hybrid_cache_utils.is_pure_recurrent_hf_config(attn_only))
      self.assertFalse(hybrid_cache_utils.has_recurrent_layers_hf_config(attn_only))

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
    hybrid_cache_utils.validate_hybrid_detection(
        "qwen3_5", SimpleNamespace(model_type="qwen3_5_text", layer_types=["linear_attention", "linear_attention"])
    )
    hybrid_cache_utils.validate_hybrid_detection("llama2", self._LLAMA_HF)

  def test_hybrid_decoder_block_with_non_hybrid_hf_config_raises(self):
    # E.g. a checkpoint whose config.json the HF detection does not understand.
    with self.assertRaisesRegex(ValueError, "does not describe a hybrid"):
      hybrid_cache_utils.validate_hybrid_detection("qwen3_5", SimpleNamespace(model_type="qwen4_exp"))

  def test_hybrid_hf_config_with_unsupported_decoder_block_raises(self):
    with self.assertRaisesRegex(ValueError, "no hybrid KV cache support"):
      hybrid_cache_utils.validate_hybrid_detection("llama2", self._QWEN_HF)
    with self.assertRaisesRegex(ValueError, "no hybrid KV cache support"):
      hybrid_cache_utils.validate_hybrid_detection(
          "llama2", SimpleNamespace(model_type="mamba", layer_types=["mamba", "mamba"])
      )


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

  def test_applies_hybrid_hook_to_pure_recurrent_models(self):
    for hf_text_config in (
        SimpleNamespace(layer_types=["linear_attention", "linear_attention"]),
        SimpleNamespace(layers_block_type=["mamba", "mamba"]),
        SimpleNamespace(attn_type_list=[0, 0]),
        SimpleNamespace(linear_attn_config={"kda_layers": [1, 2], "full_attn_layers": []}),
    ):
      vllm_config = self._vllm_config(hf_text_config)
      self.cfg_cls.verify_and_update_config(vllm_config)
      self.assertIs(self.calls[-1], vllm_config)
    self.assertEqual(len(self.calls), 4)

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


class ComputeHybridKvCacheGroupsTest(unittest.TestCase):
  """Tests KV/recurrent cache group sizing including pure-recurrent edge cases."""

  def test_balanced_and_imbalanced_hybrid_groups(self):
    # Balanced (16 / 12 < 1.5): group by larger count -> 1 group each.
    self.assertEqual(hybrid_cache_utils.compute_hybrid_kv_cache_groups(12, 16, imbalance_threshold=1.5), (16, 1, 1))
    # Imbalanced (24 / 8 >= 1.5): group by smaller count -> 1 attn, 3 mamba.
    self.assertEqual(hybrid_cache_utils.compute_hybrid_kv_cache_groups(8, 24, imbalance_threshold=1.5), (8, 1, 3))

  def test_pure_recurrent_and_pure_attention_do_not_divide_by_zero(self):
    # Pure recurrent or tiny configs where num_layers < interval (num_attn == 0).
    self.assertEqual(hybrid_cache_utils.compute_hybrid_kv_cache_groups(0, 24, imbalance_threshold=1.5), (24, 0, 1))
    self.assertEqual(hybrid_cache_utils.compute_hybrid_kv_cache_groups(0, 2, imbalance_threshold=1.5), (2, 0, 1))
    # Pure attention (num_mamba == 0).
    self.assertEqual(hybrid_cache_utils.compute_hybrid_kv_cache_groups(12, 0, imbalance_threshold=1.5), (12, 1, 0))

  def test_rejects_zero_total_or_negative_counts(self):
    with self.assertRaises(ValueError):
      hybrid_cache_utils.compute_hybrid_kv_cache_groups(0, 0)
    with self.assertRaises(ValueError):
      hybrid_cache_utils.compute_hybrid_kv_cache_groups(-1, 4)
    with self.assertRaises(ValueError):
      hybrid_cache_utils.compute_hybrid_kv_cache_groups(4, -1)


class PatchedGetKvCacheSpecTest(unittest.TestCase):
  """Tests patched get_kv_cache_spec behavior for pure-recurrent and hybrid layouts."""

  def _make_stub_manager(self, num_layers: int, interval: int, decoder_block: object = "qwen3_next"):
    """Creates a stub KVCacheManager and patched get_kv_cache_spec."""
    torch_stub = SimpleNamespace(
        bfloat16=SimpleNamespace(itemsize=2),
        float32=SimpleNamespace(itemsize=4),
        empty=lambda _shape, dtype: SimpleNamespace(element_size=lambda: dtype.itemsize),
    )
    cfg = SimpleNamespace(
        decoder_block=decoder_block,
        base_num_decoder_layers=num_layers,
        inhomogeneous_layer_cycle_interval=interval,
        gdn_num_value_heads=4,
        gdn_num_key_heads=2,
        gdn_key_head_dim=8,
        gdn_value_head_dim=8,
        gdn_conv_kernel_dim=4,
    )
    compact_calls: list[dict[str, int]] = []
    attn_page_calls: list[tuple[object, ...]] = []
    coord_calls: list[object] = []

    class FakeMambaSpec:

      def __init__(self, **kwargs):
        self.kwargs = kwargs

    class FakeKVCacheManager:
      """Minimal KVCacheManager stub for testing patched get_kv_cache_spec."""

      def __init__(self):
        self.runner = SimpleNamespace(
            model=SimpleNamespace(maxtext_config=cfg),
            cache_config=SimpleNamespace(block_size=16, mamba_cache_mode="none", mamba_page_size_padded=None),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(),
                get_total_num_kv_heads=lambda: 2,
                get_head_size=lambda: 64,
            ),
            mesh="mesh",
            kv_cache_dtype="bf16",
            vllm_config=SimpleNamespace(additional_config={}),
        )
        self._hybrid_uniform_page_size_bytes = 0

      def _maybe_set_compact_mamba_num_blocks_override(
          self,
          attn_page_size_bytes: int,
          unpadded_mamba_page_size_bytes: int,
          num_attn_layers: int,
          num_mamba_layers: int,
      ):
        """Records compact mamba sizing arguments and guards against zero division."""
        # Mirrors tpu-inference dividing by attn_bytes_per_block when unpinned.
        attn_bytes_per_block = num_attn_layers * attn_page_size_bytes
        _ = 1024 // attn_bytes_per_block
        compact_calls.append(
            {
                "attn_page_size_bytes": attn_page_size_bytes,
                "unpadded_mamba_page_size_bytes": unpadded_mamba_page_size_bytes,
                "num_attn_layers": num_attn_layers,
                "num_mamba_layers": num_mamba_layers,
            }
        )

      def _create_attention_spec(self, block_size, num_kv_heads, head_size, sliding_window=None):
        return ("attn_spec", block_size, num_kv_heads, head_size, sliding_window)

    def fake_import(name: str):
      """Stubs optional tpu_inference module imports."""
      if name == "tpu_inference.layers.common.sharding":
        return SimpleNamespace(ShardingAxisName=SimpleNamespace(KV_HEAD="kv", ATTN_HEAD="attn"))
      if name == "tpu_inference.utils":
        return SimpleNamespace(
            get_mesh_shape_product=lambda mesh, axis: 1,
            get_padded_num_heads=lambda heads, cnt: heads,
            get_padded_head_dim=lambda dim: dim,
        )
      if name == "tpu_inference.runner.kv_cache":

        def get_attn_page(*args):
          attn_page_calls.append(args)
          return 4096

        return SimpleNamespace(get_attention_page_size_bytes=get_attn_page)
      if name == "tpu_inference.core.hybrid_coordinator":
        return SimpleNamespace(maybe_install_hybrid_coordinator_hooks=coord_calls.append)
      raise ImportError(name)

    def orig_spec(manager):
      del manager
      return {f"layer.{i}": SimpleNamespace(block_size=16, sliding_window=None) for i in range(num_layers)}

    patched = hybrid_cache_utils.make_patched_get_kv_cache_spec(
        orig_spec,
        FakeMambaSpec,
        torch_stub,
        imbalance_threshold=1.5,
        import_module=fake_import,
    )
    return FakeKVCacheManager(), patched, FakeMambaSpec, compact_calls, attn_page_calls, coord_calls, torch_stub, cfg

  def test_zero_attn_layers_skips_dual_pool_hooks_and_avoids_divide_by_zero(self):
    manager, patched, fake_mamba_spec, compact_calls, attn_page_calls, coord_calls, torch_stub, cfg = (
        self._make_stub_manager(num_layers=2, interval=4)
    )
    _, _, expected_mamba_bytes = hybrid_cache_utils.build_qwen_gdn_cache_layout(cfg, torch_stub)

    spec = patched(manager)

    self.assertEqual(compact_calls, [])
    self.assertEqual(attn_page_calls, [])
    self.assertEqual(coord_calls, [])
    # pylint: disable=protected-access
    self.assertEqual(manager._hybrid_uniform_page_size_bytes, expected_mamba_bytes)
    # pylint: enable=protected-access
    self.assertEqual(manager.runner.cache_config.mamba_page_size_padded, expected_mamba_bytes)
    self.assertEqual(set(spec.keys()), {"layer.0", "layer.1"})
    for layer_spec in spec.values():
      self.assertIsInstance(layer_spec, fake_mamba_spec)
      self.assertEqual(layer_spec.kwargs["page_size_padded"], expected_mamba_bytes)

  def test_hybrid_layers_runs_dual_pool_hooks_and_assigns_mixed_specs(self):
    manager, patched, fake_mamba_spec, compact_calls, attn_page_calls, coord_calls, torch_stub, cfg = (
        self._make_stub_manager(num_layers=4, interval=4)
    )
    _, _, expected_mamba_bytes = hybrid_cache_utils.build_qwen_gdn_cache_layout(cfg, torch_stub)

    spec = patched(manager)

    self.assertEqual(len(attn_page_calls), 1)
    self.assertEqual(len(compact_calls), 1)
    self.assertEqual(compact_calls[0]["num_attn_layers"], 1)
    self.assertEqual(compact_calls[0]["num_mamba_layers"], 3)
    self.assertEqual(len(coord_calls), 1)
    expected_uniform = 4096 + 3 * expected_mamba_bytes
    # pylint: disable=protected-access
    self.assertEqual(manager._hybrid_uniform_page_size_bytes, expected_uniform)
    # pylint: enable=protected-access
    for i in range(3):
      self.assertIsInstance(spec[f"layer.{i}"], fake_mamba_spec)
      self.assertEqual(spec[f"layer.{i}"].kwargs["page_size_padded"], expected_uniform)
    self.assertEqual(spec["layer.3"], ("attn_spec", 16, 2, 64, None))

  def test_fallbacks_for_non_maxtext_runners_and_enum_decoder_block(self):
    manager, patched, fake_mamba_spec, compact_calls, _, _, _, _ = self._make_stub_manager(
        num_layers=2, interval=4, decoder_block=SimpleNamespace(value="qwen3_next")
    )
    enum_spec = patched(manager)
    self.assertEqual(compact_calls, [])
    self.assertIsInstance(enum_spec["layer.0"], fake_mamba_spec)
    self.assertIsInstance(enum_spec["layer.1"], fake_mamba_spec)

    # Runner without `model` or `maxtext_config` falls back to original spec.
    no_model_manager = SimpleNamespace(runner=SimpleNamespace())
    no_model_spec = patched(no_model_manager)
    self.assertEqual(no_model_spec["layer.0"].block_size, 16)

    no_cfg_manager = SimpleNamespace(runner=SimpleNamespace(model=SimpleNamespace()))
    no_cfg_spec = patched(no_cfg_manager)
    self.assertEqual(no_cfg_spec["layer.1"].block_size, 16)


if __name__ == "__main__":
  unittest.main()
