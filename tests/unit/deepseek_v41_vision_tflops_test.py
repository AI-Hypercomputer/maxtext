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

"""Unit tests for DeepSeek-V4.1 ViT + Aligner training TFLOPs calculation (U7)."""

import functools
import math
import sys
from typing import Any
import unittest

from flax import nnx
import jax
import numpy as np

from maxtext.configs import pyconfig
from maxtext.models import deepseek41_vision
from maxtext.utils import maxtext_utils


def _init_ds41_config(*extra_args: str, model_name: str = "deepseek4.1-flash") -> Any:
  """Initializes a DeepSeek-V4.1 MaxTextConfig with explicit CLI overrides."""
  return pyconfig.initialize(
      [
          sys.argv[0],
          "src/maxtext/configs/base.yml",
          f"model_name={model_name}",
          "run_name=ds41_vision_tflops_test",
          "enable_checkpointing=false",
          "skip_jax_distributed_system=true",
          "override_model_config=true",
          *extra_args,
      ]
  )


def _instantiate_abstract_vit(config: Any) -> deepseek41_vision.DeepSeek41VisionEncoder:
  """Helper for nnx.eval_shape abstract instantiation of DeepSeek41VisionEncoder."""
  return deepseek41_vision.DeepSeek41VisionEncoder(config=config, rngs=nnx.Rngs(params=0))


def _instantiate_abstract_projector(config: Any) -> deepseek41_vision.DeepSeek41VisionProjector:
  """Helper for nnx.eval_shape abstract instantiation of DeepSeek41VisionProjector."""
  return deepseek41_vision.DeepSeek41VisionProjector(config=config, rngs=nnx.Rngs(params=1))


class DeepSeek41VisionTflopsTest(unittest.TestCase):
  """Verifies exact closed-form FLOPs, spatial padding, scaling, no-double-counting, and parameter counts."""

  def test_release_39x39_closed_form_frozen_and_trainable_vit(self) -> None:
    """Verifies exact hand-computed FLOPs at 39x39 (P=1521, K=169) for frozen (1x) and trainable (3x) ViT."""
    p = 39 * 39
    k = 13 * 13
    self.assertEqual(p, 1521)
    self.assertEqual(k, 169)

    patch_embed_fwd = 2 * p * 588 * 1024
    vit_layer_weight_fwd = 2 * p * 1024 * 3072 + 2 * p * 1024 * 1024 + 2 * p * 1024 * 5632 + 2 * p * 2816 * 1024
    vit_layer_attn_fwd = 4 * (p**2) * 1024
    vit_weight_fwd = patch_embed_fwd + 32 * vit_layer_weight_fwd
    vit_attn_fwd = 32 * vit_layer_attn_fwd
    vit_total_fwd = vit_weight_fwd + vit_attn_fwd
    aligner_fwd = 2 * k * 9216 * 5120 + 2 * k * 5120 * 5120

    self.assertEqual(vit_weight_fwd, 1_252_220_755_968)
    self.assertEqual(vit_attn_fwd, 303_227_338_752)
    self.assertEqual(vit_total_fwd, 1_555_448_094_720)
    self.assertEqual(aligner_fwd, 24_809_308_160)

    # 1. Frozen ViT (1x ViT + 3x Aligner)
    cfg_frozen = _init_ds41_config(
        "use_multimodal=true",
        "freeze_vision_encoder_params=true",
        "per_device_batch_size=1.0",
        "max_num_images_per_example=-1",
    )
    expected_frozen_weight = (1 * vit_weight_fwd + 3 * aligner_fwd) / 1e12
    expected_frozen_attn = (1 * vit_attn_fwd) / 1e12
    expected_frozen_total = (1 * vit_total_fwd + 3 * aligner_fwd) / 1e12

    for fn in (maxtext_utils.calculate_deepseek41_vision_tflops, maxtext_utils.calculate_vision_encoder_tflops):
      tot, lw, attn = fn(cfg_frozen)
      np.testing.assert_allclose(tot, expected_frozen_total, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(lw, expected_frozen_weight, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(attn, expected_frozen_attn, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(tot, lw + attn, rtol=1e-12, atol=0.0)

    # 2. Trainable ViT (3x ViT + 3x Aligner)
    cfg_trainable = _init_ds41_config(
        "use_multimodal=true",
        "freeze_vision_encoder_params=false",
        "per_device_batch_size=1.0",
        "max_num_images_per_example=-1",
    )
    expected_train_weight = (3 * vit_weight_fwd + 3 * aligner_fwd) / 1e12
    expected_train_attn = (3 * vit_attn_fwd) / 1e12
    expected_train_total = (3 * vit_total_fwd + 3 * aligner_fwd) / 1e12

    for fn in (maxtext_utils.calculate_deepseek41_vision_tflops, maxtext_utils.calculate_vision_encoder_tflops):
      tot_tr, lw_tr, attn_tr = fn(cfg_trainable)
      np.testing.assert_allclose(tot_tr, expected_train_total, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(lw_tr, expected_train_weight, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(attn_tr, expected_train_attn, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(tot_tr, lw_tr + attn_tr, rtol=1e-12, atol=0.0)

    # Trainable minus frozen equals exactly 2x ViT forward FLOPs.
    np.testing.assert_allclose(
        expected_train_total - expected_frozen_total,
        (2 * vit_total_fwd) / 1e12,
        rtol=1e-12,
        atol=0.0,
    )

  def test_spatial_padding_grid_7x5_uses_padded_k6_tokens(self) -> None:
    """Verifies 7x5 grid (P=35) pads to Hp=9, Wp=6 -> K=6 Aligner tokens rather than unpadded 35//9=3."""
    h, w, r = 7, 5, 3
    p = h * w
    hp = math.ceil(h / r) * r
    wp = math.ceil(w / r) * r
    k_padded = (hp // r) * (wp // r)
    self.assertEqual((p, hp, wp, k_padded), (35, 9, 6, 6))

    vit_weight_fwd = 2 * p * 588 * 1024 + 32 * (
        2 * p * 1024 * 3072 + 2 * p * 1024 * 1024 + 2 * p * 1024 * 5632 + 2 * p * 2816 * 1024
    )
    vit_attn_fwd = 32 * (4 * (p**2) * 1024)
    aligner_per_token_fwd = 2 * 9216 * 5120 + 2 * 5120 * 5120
    aligner_padded_fwd = k_padded * aligner_per_token_fwd
    aligner_unpadded_trunc_fwd = (p // (r * r)) * aligner_per_token_fwd

    cfg_padded = _init_ds41_config(
        "use_multimodal=true",
        "freeze_vision_encoder_params=true",
        "per_device_batch_size=1.0",
        "ds41_vision_grid=[7,5]",
    )
    tot, lw, attn = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_padded)
    expected_lw = (vit_weight_fwd + 3 * aligner_padded_fwd) / 1e12
    expected_attn = vit_attn_fwd / 1e12
    expected_tot = expected_lw + expected_attn

    np.testing.assert_allclose(tot, expected_tot, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(lw, expected_lw, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(attn, expected_attn, rtol=1e-12, atol=0.0)

    # Confirm padded K=6 is strictly greater than unpadded K=3 by 3 tokens of Aligner training FLOPs.
    unpadded_lw = (vit_weight_fwd + 3 * aligner_unpadded_trunc_fwd) / 1e12
    np.testing.assert_allclose(
        lw - unpadded_lw,
        (3 * (6 - 3) * aligner_per_token_fwd) / 1e12,
        rtol=1e-12,
        atol=0.0,
    )

  def test_num_images_and_per_device_batch_size_scaling_and_validation(self) -> None:
    """Verifies max_num_images_per_example (-1, 1, 4), per_device_batch_size (0.125, 1.0, 2.0), and guards."""
    cfg_default_neg1 = _init_ds41_config(
        "use_multimodal=true",
        "per_device_batch_size=1.0",
        "max_num_images_per_example=-1",
    )
    cfg_one = _init_ds41_config(
        "use_multimodal=true",
        "per_device_batch_size=1.0",
        "max_num_images_per_example=1",
    )
    cfg_four = _init_ds41_config(
        "use_multimodal=true",
        "per_device_batch_size=1.0",
        "max_num_images_per_example=4",
    )
    cfg_pdb_eighth = _init_ds41_config(
        "use_multimodal=true",
        "per_device_batch_size=0.125",
        "max_num_images_per_example=1",
    )
    cfg_pdb_two_img_four = _init_ds41_config(
        "use_multimodal=true",
        "per_device_batch_size=2.0",
        "max_num_images_per_example=4",
    )

    res_neg1 = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_default_neg1)
    res_one = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_one)
    res_four = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_four)
    res_eighth = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_pdb_eighth)
    res_eight_x = maxtext_utils.calculate_deepseek41_vision_tflops(cfg_pdb_two_img_four)

    self.assertEqual(res_neg1, res_one)
    for v_one, v_four, v_eighth, v_eight_x in zip(res_one, res_four, res_eighth, res_eight_x, strict=True):
      self.assertGreater(v_one, 0.0)
      np.testing.assert_allclose(v_four, 4.0 * v_one, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(v_eighth, 0.125 * v_one, rtol=1e-12, atol=0.0)
      np.testing.assert_allclose(v_eight_x, 8.0 * v_one, rtol=1e-12, atol=0.0)

    # Positive baseline before negative validation assertions (Rule 7 liveness).
    self.assertAlmostEqual(res_one[0], 1.6298760192, places=12)

    class _InvalidConfigProxy:
      """Overrides a single attribute on a valid config to test precondition guards."""

      def __init__(self, base: Any, key: str, value: Any) -> None:
        self._base = base
        self._key = key
        self._value = value

      def __getattr__(self, name: str) -> Any:
        if name == self._key:
          return self._value
        return getattr(self._base, name)

    bad_grid_cfg = _InvalidConfigProxy(cfg_one, "ds41_vision_grid", [0, 39])
    with self.assertRaisesRegex(ValueError, "ds41_vision_grid"):
      maxtext_utils.calculate_deepseek41_vision_tflops(bad_grid_cfg)  # type: ignore[arg-type]

    bad_ratio_cfg = _InvalidConfigProxy(cfg_one, "ds41_vision_downsample_ratio", 0)
    with self.assertRaisesRegex(ValueError, "ds41_vision_downsample_ratio"):
      maxtext_utils.calculate_deepseek41_vision_tflops(bad_ratio_cfg)  # type: ignore[arg-type]

  def test_calculate_tflops_training_per_device_no_double_counting(self) -> None:
    """Verifies total_tflops(mm=True) - total_tflops(mm=False) == calculate_vision_encoder_tflops on full and mini."""
    for model_name in ("deepseek4.1-flash", "deepseek4.1-flash-mini"):
      for freeze_str in ("true", "false"):
        for grad_accum in (1, 2):
          cfg_txt = _init_ds41_config(
              "use_multimodal=false",
              f"freeze_vision_encoder_params={freeze_str}",
              "per_device_batch_size=1.0",
              "max_target_length=2048",
              f"gradient_accumulation_steps={grad_accum}",
              model_name=model_name,
          )
          cfg_mm = _init_ds41_config(
              "use_multimodal=true",
              f"freeze_vision_encoder_params={freeze_str}",
              "per_device_batch_size=1.0",
              "max_target_length=2048",
              f"gradient_accumulation_steps={grad_accum}",
              model_name=model_name,
          )
          tot_txt, lw_txt, attn_txt = maxtext_utils.calculate_tflops_training_per_device(cfg_txt, log=False)
          tot_mm, lw_mm, attn_mm = maxtext_utils.calculate_tflops_training_per_device(cfg_mm, log=False)
          vis_tot, vis_lw, vis_attn = maxtext_utils.calculate_vision_encoder_tflops(cfg_mm)

          self.assertGreater(tot_txt, 0.0)
          self.assertGreater(vis_tot, 0.0)
          np.testing.assert_allclose(tot_mm - tot_txt, grad_accum * vis_tot, rtol=1e-12, atol=1e-12)
          np.testing.assert_allclose(lw_mm - lw_txt, grad_accum * vis_lw, rtol=1e-12, atol=1e-12)
          np.testing.assert_allclose(attn_mm - attn_txt, grad_accum * vis_attn, rtol=1e-12, atol=1e-12)

          # Verify text-only TFLOPs are 100% unchanged from the base decoder formula.
          total_ffn_flops = maxtext_utils.calculate_routed_and_shared_ffn_tflops_per_device(cfg_txt)
          qkv_flops = (
              2
              * cfg_txt.per_device_batch_size
              * cfg_txt.max_target_length
              * cfg_txt.emb_dim
              * (cfg_txt.num_query_heads + 2 * cfg_txt.num_kv_heads)
              * cfg_txt.head_dim
          )
          proj_flops = (
              2
              * cfg_txt.per_device_batch_size
              * cfg_txt.max_target_length
              * cfg_txt.emb_dim
              * cfg_txt.num_query_heads
              * cfg_txt.head_dim
          )
          causal_attn_flops = (
              2
              * cfg_txt.per_device_batch_size
              * (cfg_txt.max_target_length**2)
              * cfg_txt.num_query_heads
              * cfg_txt.head_dim
          )
          emb_flops = 2 * cfg_txt.per_device_batch_size * cfg_txt.max_target_length * cfg_txt.emb_dim * cfg_txt.vocab_size
          expected_lw_txt = (
              (total_ffn_flops + (qkv_flops + proj_flops) * cfg_txt.num_decoder_layers + emb_flops)
              * 3
              / 1e12
              * grad_accum
          )
          expected_attn_txt = causal_attn_flops * cfg_txt.num_decoder_layers * 3 / 1e12 * grad_accum
          np.testing.assert_allclose(lw_txt, expected_lw_txt, rtol=1e-12, atol=0.0)
          np.testing.assert_allclose(attn_txt, expected_attn_txt, rtol=1e-12, atol=0.0)
          np.testing.assert_allclose(tot_txt, expected_lw_txt + expected_attn_txt, rtol=1e-12, atol=0.0)

  def test_module_parameter_counts_match_tflops_linear_shapes(self) -> None:
    """Cross-checks instantiated 32-layer ViT (411.8M) and Aligner (73.4M) shapes via nnx.eval_shape."""
    cfg = _init_ds41_config("use_multimodal=true", "per_device_batch_size=1.0")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    with mesh:
      abs_vit = nnx.eval_shape(functools.partial(_instantiate_abstract_vit, cfg))
      abs_proj = nnx.eval_shape(functools.partial(_instantiate_abstract_projector, cfg))

    vit_leaves = jax.tree_util.tree_leaves(nnx.state(abs_vit, nnx.Param))
    proj_leaves = jax.tree_util.tree_leaves(nnx.state(abs_proj, nnx.Param))

    vit_total_params = sum(int(np.prod(leaf.shape)) for leaf in vit_leaves)
    vit_2d_kernel_params = sum(int(np.prod(leaf.shape)) for leaf in vit_leaves if len(leaf.shape) == 2)
    proj_total_params = sum(int(np.prod(leaf.shape)) for leaf in proj_leaves)
    proj_2d_kernel_params = sum(int(np.prod(leaf.shape)) for leaf in proj_leaves if len(leaf.shape) == 2)
    proj_1d_params = sum(int(np.prod(leaf.shape)) for leaf in proj_leaves if len(leaf.shape) == 1)

    # Exact parameter counts at D_vit=1024, L_vit=32, I_vit=2816, D_llm=5120:
    # ViT: 411,643,904 (2D linear kernels) + 198,656 (biases + RMSNorms) = 411,842,560 (~411.8M).
    self.assertEqual(vit_2d_kernel_params, 411_643_904)
    self.assertEqual(vit_total_params, 411_842_560)
    self.assertAlmostEqual(vit_total_params / 1e6, 411.8, places=1)

    # Aligner: 73,400,320 (2D linear kernels w1+w2) + 10,240 (w1/w2 biases) + 15,360 (3 delimiters) = 73,425,920 (~73.4M).
    self.assertEqual(proj_2d_kernel_params, 73_400_320)
    self.assertEqual(proj_1d_params, 25_600)
    self.assertEqual(proj_total_params, 73_425_920)
    self.assertAlmostEqual(proj_total_params / 1e6, 73.4, places=1)

    # Every 2D linear kernel parameter in ViT runs once per patch (P=1521, 2 FLOPs/MAC),
    # and every 2D linear kernel parameter in Aligner runs once per aligned token (K=169, 2 FLOPs/MAC).
    p = 39 * 39
    k = 13 * 13
    _, lw_frozen, _ = maxtext_utils.calculate_deepseek41_vision_tflops(cfg)
    expected_lw_from_module_shapes = (1 * (2 * p * vit_2d_kernel_params) + 3 * (2 * k * proj_2d_kernel_params)) / 1e12
    np.testing.assert_allclose(lw_frozen, expected_lw_from_module_shapes, rtol=1e-12, atol=0.0)


if __name__ == "__main__":
  unittest.main()
