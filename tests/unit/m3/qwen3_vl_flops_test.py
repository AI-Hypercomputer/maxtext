# Copyright 2026 Google LLC
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

"""Unit tests for `maxtext.m3.models.qwen3_vl.qwen3_vl_flops`."""

import types

from absl.testing import absltest
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.models.qwen3 import qwen3_flops
from maxtext.m3.models.qwen3_vl import qwen3_vl_flops
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def _make_mock_vl_config(**overrides) -> types.SimpleNamespace:
  """Creates a mock Qwen3-VL 2B configuration for FLOPs testing."""
  cfg = types.SimpleNamespace(
      model_name="qwen3-vl-2b",
      use_m3_model=True,
      use_multimodal=True,
      freeze_vision_encoder_params=True,
      per_device_batch_size=2,
      max_target_length=128,
      emb_dim=128,
      mlp_dim=256,
      num_decoder_layers=2,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=32,
      mlp_activations=("silu", "linear"),
      vocab_size=512,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      logits_via_embedding=True,
      normalize_embedding_logits=False,
      rope_max_timescale=5_000_000.0,
      enable_dropout=False,
      scan_layers=False,
      dtype="float32",
      dtype_mm="float32",
      weight_dtype="float32",
      float32_logits=True,
      image_size_for_vit=64,
      hidden_size_for_vit=64,
      intermediate_size_for_vit=128,
      num_attention_heads_for_vit=4,
      num_hidden_layers_for_vit=3,
      num_channels_for_vit=3,
      patch_size_for_vit=16,
      temporal_patch_size_for_vit=2,
      spatial_merge_size_for_vit=2,
      out_hidden_size_for_vit=128,
      num_position_embeddings_for_vit=16,
      rope_theta_for_vit=10_000,
      deepstack_visual_indexes_for_vit=[0, 1],
      use_mrope=True,
      mrope_section=[8, 4, 4],
      gradient_accumulation_steps=1,
      use_dpo=False,
  )
  for k, v in overrides.items():
    setattr(cfg, k, v)
  return cfg


class Qwen3VlFlopsTest(absltest.TestCase):
  """Tests multimodal and frozen-vision FLOP accounting for Qwen3-VL."""

  def test_text_only_mode_matches_qwen3_tflops_exactly(self):
    cfg = _make_mock_vl_config(use_multimodal=False)
    vl_res = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(cfg, log=False)
    qwen3_res = qwen3_flops.calculate_qwen3_tflops_per_device(cfg, log=False)
    self.assertEqual(vl_res, qwen3_res)

  def test_vision_weight_fwd_flops_match_exact_instantiated_parameter_kernels(self):
    cfg = _make_mock_vl_config()
    abstract_model = nnx.eval_shape(lambda: create_qwen3_vl_model(cfg, mesh=None, rngs=nnx.Rngs(0)))
    vision_params = nnx.to_flat_state(nnx.state(abstract_model.vision_encoder, nnx.Param))

    batch = cfg.per_device_batch_size
    num_patches = (cfg.image_size_for_vit // cfg.patch_size_for_vit) ** 2
    merged_tokens = num_patches // (cfg.spatial_merge_size_for_vit**2)

    expected_weight_fwd_flops = 0
    for path, var in vision_params:
      shape = var.shape
      # Sum FLOPs over all projection/convolution `kernel` weights (excluding biases, LayerNorm scales, and pos_embed).
      if path[-1] != "kernel":
        continue
      tokens = merged_tokens if ("merger" in ".".join(str(p) for p in path)) else num_patches
      num_weights = 1
      for dim in shape:
        num_weights *= dim
      expected_weight_fwd_flops += 2 * batch * tokens * num_weights

    weight_fwd_flops, attn_fwd_flops = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg)
    self.assertEqual(weight_fwd_flops, expected_weight_fwd_flops)
    expected_attn_fwd_flops = 4 * batch * (num_patches**2) * cfg.hidden_size_for_vit * cfg.num_hidden_layers_for_vit
    self.assertEqual(attn_fwd_flops, expected_attn_fwd_flops)

  def test_frozen_vs_unfrozen_multiplier_and_negative_control(self):
    text_cfg = _make_mock_vl_config(use_multimodal=False)
    text_total, text_weight, text_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(text_cfg, log=False)

    frozen_cfg = _make_mock_vl_config(use_multimodal=True, freeze_vision_encoder_params=True)
    unfrozen_cfg = _make_mock_vl_config(use_multimodal=True, freeze_vision_encoder_params=False)

    frozen_total, frozen_weight, frozen_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(frozen_cfg, log=False)
    unfrozen_total, unfrozen_weight, unfrozen_attn = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(
        unfrozen_cfg, log=False
    )

    enc_w, merg_w, vis_a_fwd = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(frozen_cfg)

    # When frozen: encoder is 1x forward-only; mergers are trainable 3x fwd+bwd.
    # learnable_weight_tflops excludes frozen encoder and includes 3x merger weights.
    self.assertAlmostEqual(frozen_weight - text_weight, 3.0 * merg_w / 10**12, places=12)
    self.assertAlmostEqual(frozen_attn - text_attn, vis_a_fwd / 10**12, places=12)
    expected_frozen_total = text_total + (enc_w + 3.0 * merg_w + vis_a_fwd) / 10**12
    self.assertAlmostEqual(frozen_total, expected_frozen_total, places=12)

    # When unfrozen: both encoder and mergers are 3x fwd+bwd.
    self.assertAlmostEqual(unfrozen_weight - text_weight, 3.0 * (enc_w + merg_w) / 10**12, places=12)
    self.assertAlmostEqual(unfrozen_attn - text_attn, 3.0 * vis_a_fwd / 10**12, places=12)
    expected_unfrozen_total = text_total + 3.0 * (enc_w + merg_w + vis_a_fwd) / 10**12
    self.assertAlmostEqual(unfrozen_total, expected_unfrozen_total, places=12)

    # Negative control: difference between unfrozen and frozen is 2 * enc_w + 2 * vis_a_fwd.
    self.assertAlmostEqual(unfrozen_total - frozen_total, (2.0 * enc_w + 2.0 * vis_a_fwd) / 10**12, places=12)

    # Verify `freeze_vision_tower` alias override also works cleanly.
    alias_unfrozen = _make_mock_vl_config(
        use_multimodal=True, freeze_vision_encoder_params=True, freeze_vision_tower=False
    )
    self.assertAlmostEqual(
        qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(alias_unfrozen, log=False)[0],
        unfrozen_total,
        places=12,
    )

  def test_non_square_image_multi_image_dpo_and_validation(self):
    rect_cfg = _make_mock_vl_config(image_size_for_vit=[64, 96])
    w_rect, a_rect = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(rect_cfg)
    self.assertGreater(w_rect, 0)
    self.assertGreater(a_rect, 0)

    # Test max_num_images_per_example scaling vs single image.
    cfg_1img = _make_mock_vl_config(image_size_for_vit=64, max_num_images_per_example=1)
    cfg_3img = _make_mock_vl_config(image_size_for_vit=64, max_num_images_per_example=3)
    w_1, a_1 = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg_1img)
    w_3, a_3 = qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(cfg_3img)
    self.assertEqual(w_3, 3 * w_1)
    self.assertEqual(a_3, 3 * a_1)

    dpo_cfg = _make_mock_vl_config(use_dpo=True, freeze_vision_encoder_params=True)
    dpo_total, _, _ = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(dpo_cfg, log=False)
    base_total, _, _ = qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device(
        _make_mock_vl_config(use_dpo=False, freeze_vision_encoder_params=True), log=False
    )
    self.assertGreater(dpo_total, base_total)

    with self.assertRaisesRegex(ValueError, "divisible"):
      qwen3_vl_flops.calculate_qwen3_vl_vision_fwd_flops(_make_mock_vl_config(image_size_for_vit=50))

  def test_registry_and_maxtext_utils_dispatch_for_qwen3_vl_2b_config(self):
    self.assertIs(
        m3_models.MODEL_FLOPS_REGISTRY["qwen3-vl-2b"],
        qwen3_vl_flops.calculate_qwen3_vl_tflops_per_device,
    )
    output_dir = self.create_tempdir().full_path
    cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-vl-2b",
            "use_m3_model=true",
            "scan_layers=false",
            "per_device_batch_size=1",
            "max_target_length=512",
            "enable_checkpointing=false",
            f"base_output_directory={output_dir}",
            "run_name=qwen3_vl_flops_dispatch",
        ]
    )
    total_mm, weight_mm, attn_mm = maxtext_utils.calculate_tflops_training_per_device(cfg, log=False)
    total_txt, _, _ = qwen3_flops.calculate_qwen3_tflops_per_device(cfg, log=False)
    self.assertGreater(total_mm, total_txt)
    enc_w, _, _ = qwen3_vl_flops.calculate_qwen3_vl_vision_components_fwd_flops(cfg)
    # When frozen, total_tflops includes 1x frozen encoder weight TFLOPs while learnable weight_mm excludes it.
    self.assertAlmostEqual(total_mm, weight_mm + attn_mm + enc_w / 10**12, places=9)


if __name__ == "__main__":
  absltest.main()
