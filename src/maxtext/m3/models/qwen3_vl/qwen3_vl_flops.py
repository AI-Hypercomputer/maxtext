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
# allow-bespoke-m3-infra: multimodal Qwen3-VL ViT and DeepStack PatchMerger FLOPs registered in MODEL_FLOPS_REGISTRY

"""Per-device training TFLOPs calculation for multimodal Qwen3-VL m3 models.

Accounts for both the dense Qwen3 text decoder and `Qwen3VLVisionTower`. When
`freeze_vision_encoder_params=True`, `Qwen3VLVisionTower` stops gradients on
both `enc_out` and `deep_feats`, so `Qwen3VLVisionEncoder_0` (patch embedding,
`Qwen3VLVisionBlock` layers, and `len(deepstack_visual_indexes_for_vit)`
DeepStack `Qwen3VLVisionPatchMerger`s) executes forward-only (`1x`) while the
final `Qwen3VLVisionProjector_0` merger remains trainable (`3x`).
"""

from collections.abc import Sequence
from typing import Any

from maxtext.m3.models.qwen3 import qwen3_flops


def _resolve_image_hw(config: Any) -> tuple[int, int]:
  """Extracts `(height, width)` in pixels from `config.image_size_for_vit`."""
  image_size = config.image_size_for_vit
  if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
    if len(image_size) != 2:
      raise ValueError(f"Expected length-2 image_size_for_vit, got {image_size!r}.")
    return int(image_size[0]), int(image_size[1])
  if image_size is None:
    raise ValueError("image_size_for_vit must be set when use_multimodal=True.")
  size = int(image_size)
  return size, size


def calculate_qwen3_vl_vision_components_fwd_flops(config: Any) -> tuple[float, float, float]:
  """Calculates `(encoder_weight_flops, merger_weight_flops, vision_attn_flops)` per microbatch.

  `encoder_weight_flops` covers `Qwen3VLVisionEncoder_0` (including intermediate
  DeepStack mergers gated by `freeze_vision_encoder_params`); `merger_weight_flops`
  covers the final trainable `Qwen3VLVisionProjector_0` merger.
  """
  batch = config.per_device_batch_size
  raw_imgs = int(config.max_num_images_per_example)
  num_images = raw_imgs if raw_imgs > 0 else 1
  batch_images = batch * num_images

  img_h, img_w = _resolve_image_hw(config)
  patch_size = int(config.patch_size_for_vit)
  temporal_patch_size = int(config.temporal_patch_size_for_vit)
  spatial_merge_size = int(config.spatial_merge_size_for_vit)
  in_channels = int(config.num_channels_for_vit)
  hidden_size = int(config.hidden_size_for_vit)
  intermediate_size = int(config.intermediate_size_for_vit)
  num_heads = int(config.num_attention_heads_for_vit)
  head_dim = hidden_size // num_heads
  attn_dim = num_heads * head_dim
  num_layers = int(config.num_hidden_layers_for_vit)
  out_hidden_size = int(config.out_hidden_size_for_vit)
  deepstack_indexes = tuple(config.deepstack_visual_indexes_for_vit)

  stride_hw = patch_size * spatial_merge_size
  if img_h <= 0 or img_w <= 0 or img_h % stride_hw != 0 or img_w % stride_hw != 0:
    raise ValueError(
        f"image_size_for_vit=({img_h}, {img_w}) must be positive and divisible by "
        f"patch_size_for_vit * spatial_merge_size_for_vit = {stride_hw}."
    )
  if temporal_patch_size <= 0:
    raise ValueError(f"temporal_patch_size_for_vit={temporal_patch_size} must be positive.")

  num_patches = (img_h // patch_size) * (img_w // patch_size)

  # 1. Qwen3VLVisionPatchEmbed: non-overlapping 3D conv `(T_p, P, P, C) -> D_vit`.
  patch_volume = in_channels * temporal_patch_size * patch_size * patch_size
  patch_embed_flops = 2 * batch_images * num_patches * patch_volume * hidden_size

  # 2. Qwen3VLVisionBlock x `num_layers`: Q/K/V/Out projections, global attention,
  # and 2-layer GELU MLP (`D_vit -> M_vit -> D_vit`).
  qkv_out_flops_per_layer = 4 * (2 * batch_images * num_patches * hidden_size * attn_dim)
  mlp_flops_per_layer = 4 * batch_images * num_patches * hidden_size * intermediate_size
  blocks_weight_flops = (qkv_out_flops_per_layer + mlp_flops_per_layer) * num_layers
  vision_attn_flops = 4 * batch_images * num_patches * num_patches * attn_dim * num_layers

  # 3. Spatial PatchMergers: `len(deepstack_indexes)` intermediate DeepStack mergers inside
  # `Qwen3VLVisionEncoder_0` (frozen when `freeze_vision_encoder_params=True`) plus `1` final
  # `Qwen3VLVisionProjector_0` merger (`S_vit / merge^2` tokens).
  tokens_per_block = spatial_merge_size * spatial_merge_size
  merged_tokens = num_patches // tokens_per_block
  merged_dim = hidden_size * tokens_per_block
  single_merger_flops = 2 * batch_images * merged_tokens * (merged_dim * merged_dim + merged_dim * out_hidden_size)
  deepstack_merger_flops = len(deepstack_indexes) * single_merger_flops

  encoder_weight_flops = patch_embed_flops + blocks_weight_flops + deepstack_merger_flops
  merger_weight_flops = single_merger_flops

  return encoder_weight_flops, merger_weight_flops, vision_attn_flops


def calculate_qwen3_vl_vision_fwd_flops(config: Any) -> tuple[int, int]:  # public-api
  """Calculates exact `(weight_fwd_flops, attention_fwd_flops)` per microbatch for `Qwen3VLVisionTower`."""
  enc_w_flops, merger_w_flops, attn_flops = calculate_qwen3_vl_vision_components_fwd_flops(config)
  return enc_w_flops + merger_w_flops, attn_flops


def calculate_qwen3_vl_tflops_per_device(config: Any, log: bool = True) -> tuple[float, float, float]:
  """Calculates `(total_tflops, learnable_weight_tflops, attention_tflops)` per device."""
  if not config.use_multimodal:
    return qwen3_flops.calculate_qwen3_tflops_per_device(config, log=log)

  _, text_weight_tflops, text_attn_tflops = qwen3_flops.calculate_qwen3_tflops_per_device(config, log=False)
  if config.use_dpo:
    # `calculate_qwen3_tflops_per_device` folds DPO's 1x reference forward into
    # `total_tflops` (`reference_model_tflops = learnable_weight_tflops / 3`) and
    # `attention_tflops`. Keep the exact decomposition consistent here.
    text_ref_tflops = text_weight_tflops / 3.0
  else:
    text_ref_tflops = 0.0

  encoder_weight_flops, merger_weight_flops, vision_attn_fwd_flops = calculate_qwen3_vl_vision_components_fwd_flops(
      config
  )
  grad_accum = config.gradient_accumulation_steps

  if config.freeze_vision_encoder_params:
    frozen_encoder_tflops = encoder_weight_flops / 10**12 * grad_accum
    vision_attn_tflops = vision_attn_fwd_flops / 10**12 * grad_accum
    trainable_vision_weight_tflops = 3 * merger_weight_flops / 10**12 * grad_accum
  else:
    frozen_encoder_tflops = 0.0
    vision_attn_tflops = 3 * vision_attn_fwd_flops / 10**12 * grad_accum
    trainable_vision_weight_tflops = 3 * (encoder_weight_flops + merger_weight_flops) / 10**12 * grad_accum

  if config.use_dpo:
    vision_ref_weight_tflops = (encoder_weight_flops + merger_weight_flops) / 10**12 * grad_accum
    vision_attn_tflops += vision_attn_fwd_flops / 10**12 * grad_accum
  else:
    vision_ref_weight_tflops = 0.0

  learnable_weight_tflops = text_weight_tflops + trainable_vision_weight_tflops
  attention_tflops = text_attn_tflops + vision_attn_tflops
  total_tflops = (
      learnable_weight_tflops + frozen_encoder_tflops + attention_tflops + text_ref_tflops + vision_ref_weight_tflops
  )

  if log:
    print(
        "Per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops / total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops / total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
