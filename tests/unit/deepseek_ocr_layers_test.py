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

"""Tests for DeepSeek-OCR-2 layers and multimodal plumbing."""

import os
import sys
import unittest
from typing import Optional, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx

from maxtext.models import deepseek_ocr
from maxtext.models import deepseek
from maxtext.multimodal import utils as mm_utils
from maxtext.multimodal.processor_deepseek_ocr import (
    DEEPSEEK_OCR_CROP_TOKENS,
    DEEPSEEK_OCR_GLOBAL_TOKENS,
    DEEPSEEK_OCR_IMAGE_TOKEN_ID,
    DEEPSEEK_OCR_SEPARATOR_TOKENS,
    DeepseekOCR2PreprocessorOutput,
    add_extra_tokens_for_images_deepseek_ocr,
    get_image_offsets_deepseek_ocr,
)
from maxtext.configs import pyconfig
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from PIL import Image, ImageOps

import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as PTQwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as PTQwen2DecoderLayer

torch.set_grad_enabled(False)

# ==============================================================================
# Helper functions for HF Image Preprocessing Reference
# ==============================================================================


def _normalize_hwc(image):
  arr = np.array(image, dtype=np.float32) / 255.0
  return (arr - 0.5) / 0.5


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
  best_ratio_diff = float("inf")
  best_ratio = (1, 1)
  area = width * height
  for ratio in target_ratios:
    target_aspect_ratio = ratio[0] / ratio[1]
    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
    if ratio_diff < best_ratio_diff:
      best_ratio_diff = ratio_diff
      best_ratio = ratio
    elif ratio_diff == best_ratio_diff:
      if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
        best_ratio = ratio
  return best_ratio


def _hf_dynamic_preprocess(image, min_num=2, max_num=6, image_size=768):
  orig_width, orig_height = image.size
  aspect_ratio = orig_width / orig_height
  target_ratios = set(
      (i, j)
      for n in range(min_num, max_num + 1)
      for i in range(1, n + 1)
      for j in range(1, n + 1)
      if i * j <= max_num and i * j >= min_num
  )
  target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
  target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
  target_width = image_size * target_aspect_ratio[0]
  target_height = image_size * target_aspect_ratio[1]
  blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

  resized_img = image.resize((target_width, target_height))
  processed_images = []
  for i in range(blocks):
    box = (
        (i % (target_width // image_size)) * image_size,
        (i // (target_width // image_size)) * image_size,
        ((i % (target_width // image_size)) + 1) * image_size,
        ((i // (target_width // image_size)) + 1) * image_size,
    )
    processed_images.append(resized_img.crop(box))
  return processed_images, target_aspect_ratio


def _hf_reference_inputs(image, base_size=1024, image_size=768):
  if image.size[0] <= image_size and image.size[1] <= image_size:
    crops_raw = []
    crop_ratio = [1, 1]
  else:
    crops_raw, crop_ratio = _hf_dynamic_preprocess(image, min_num=2, max_num=6, image_size=image_size)

  global_view = ImageOps.pad(image, (base_size, base_size), color=(127, 127, 127))
  images_ori = _normalize_hwc(global_view)
  images_crop = np.stack([_normalize_hwc(crop) for crop in crops_raw], axis=0) if crops_raw else np.zeros((0, image_size, image_size, 3), dtype=np.float32)

  return {
      "images_ori": images_ori,
      "images_crop": images_crop,
      "crop_ratio": np.array(crop_ratio, dtype=np.int32),
  }

# ==============================================================================
# Helper functions for weight copying
# ==============================================================================


def to_jax(pt_tensor: torch.Tensor) -> jax.Array:
  return jnp.asarray(pt_tensor.detach().cpu().numpy())


def copy_linear_weights(torch_linear, jax_linear):
  jax_linear.kernel.value = jnp.array(torch_linear.weight.detach().cpu().numpy().T)
  if torch_linear.bias is not None and jax_linear.bias is not None:
    jax_linear.bias.value = jnp.array(torch_linear.bias.detach().cpu().numpy())


def copy_rmsnorm_weights(torch_norm, jax_norm):
  if hasattr(torch_norm, "weight") and hasattr(jax_norm, "scale"):
    jax_norm.scale.value = jnp.array(torch_norm.weight.detach().cpu().numpy())


def copy_qwen2_attention_weights(torch_attn, jax_attn):
  num_heads = jax_attn.num_query_heads
  num_kv_heads = jax_attn.num_kv_heads
  head_dim = jax_attn.head_dim
  hidden_size = num_heads * head_dim
  kv_dim = num_kv_heads * head_dim
  output_dim = hidden_size

  q_weight = torch_attn.q_proj.weight.detach().cpu().numpy()
  k_weight = torch_attn.k_proj.weight.detach().cpu().numpy()
  v_weight = torch_attn.v_proj.weight.detach().cpu().numpy()

  q_bias = torch_attn.q_proj.bias.detach().cpu().numpy() if torch_attn.q_proj.bias is not None else np.zeros(hidden_size)
  k_bias = torch_attn.k_proj.bias.detach().cpu().numpy() if torch_attn.k_proj.bias is not None else np.zeros(kv_dim)
  v_bias = torch_attn.v_proj.bias.detach().cpu().numpy() if torch_attn.v_proj.bias is not None else np.zeros(kv_dim)

  jax_attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim))
  if jax_attn.query.bias is not None:
    jax_attn.query.bias.value = jnp.array(q_bias.reshape(num_heads, head_dim))

  jax_attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, num_kv_heads, head_dim))
  if jax_attn.key.bias is not None:
    jax_attn.key.bias.value = jnp.array(k_bias.reshape(num_kv_heads, head_dim))

  jax_attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, num_kv_heads, head_dim))
  if jax_attn.value.bias is not None:
    jax_attn.value.bias.value = jnp.array(v_bias.reshape(num_kv_heads, head_dim))

  out_weight = torch_attn.o_proj.weight.detach().cpu().numpy()
  jax_attn.out.kernel.value = jnp.array(out_weight.T.reshape(num_heads, head_dim, output_dim))
  if torch_attn.o_proj.bias is not None and jax_attn.out.bias is not None:
    jax_attn.out.bias.value = jnp.array(torch_attn.o_proj.bias.detach().cpu().numpy())


def copy_llama_attention_weights(torch_attn, jax_attn):
  num_heads = jax_attn.num_query_heads
  num_kv_heads = jax_attn.num_kv_heads
  head_dim = jax_attn.head_dim
  hidden_size = num_heads * head_dim

  q_weight = torch_attn.q_proj.weight.detach().cpu().numpy()
  k_weight = torch_attn.k_proj.weight.detach().cpu().numpy()
  v_weight = torch_attn.v_proj.weight.detach().cpu().numpy()
  out_weight = torch_attn.o_proj.weight.detach().cpu().numpy()

  q_scale = head_dim**-0.5
  jax_attn.query.kernel.value = jnp.asarray((q_weight.T * q_scale).reshape(hidden_size, num_heads, head_dim))
  jax_attn.key.kernel.value = jnp.asarray(k_weight.T.reshape(hidden_size, num_kv_heads, head_dim))
  jax_attn.value.kernel.value = jnp.asarray(v_weight.T.reshape(hidden_size, num_kv_heads, head_dim))
  jax_attn.out.kernel.value = jnp.asarray(out_weight.T.reshape(num_heads, head_dim, hidden_size))

  if torch_attn.q_proj.bias is not None and jax_attn.query.bias is not None:
    jax_attn.query.bias.value = jnp.asarray(torch_attn.q_proj.bias.detach().cpu().numpy().reshape(num_heads, head_dim))
  if torch_attn.k_proj.bias is not None and jax_attn.key.bias is not None:
    jax_attn.key.bias.value = jnp.asarray(torch_attn.k_proj.bias.detach().cpu().numpy().reshape(num_kv_heads, head_dim))
  if torch_attn.v_proj.bias is not None and jax_attn.value.bias is not None:
    jax_attn.value.bias.value = jnp.asarray(torch_attn.v_proj.bias.detach().cpu().numpy().reshape(num_kv_heads, head_dim))
  if torch_attn.o_proj.bias is not None and jax_attn.out.bias is not None:
    jax_attn.out.bias.value = jnp.asarray(torch_attn.o_proj.bias.detach().cpu().numpy())


def copy_llama_decoder_layer_weights(torch_layer, jax_layer):
  copy_rmsnorm_weights(torch_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
  copy_rmsnorm_weights(torch_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)
  copy_llama_attention_weights(torch_layer.self_attn, jax_layer.self_attention)
  copy_linear_weights(torch_layer.mlp.gate_proj, jax_layer.mlp.wi_0)
  copy_linear_weights(torch_layer.mlp.up_proj, jax_layer.mlp.wi_1)
  copy_linear_weights(torch_layer.mlp.down_proj, jax_layer.mlp.wo)


def make_llama_decoder_layer(hidden_size, num_heads, num_kv_heads, intermediate_size, seq_len):
  config = LlamaConfig(
      hidden_size=hidden_size,
      intermediate_size=intermediate_size,
      num_hidden_layers=1,
      num_attention_heads=num_heads,
      num_key_value_heads=num_kv_heads,
      vocab_size=128,
      max_position_embeddings=seq_len,
      rms_norm_eps=1e-6,
      attention_dropout=0.0,
      hidden_act="silu",
      attention_bias=False,
  )
  config._attn_implementation = "eager"  # pylint: disable=protected-access
  return LlamaDecoderLayer(config, layer_idx=0).eval(), LlamaRotaryEmbedding(config)


# ==============================================================================
# PyTorch Reference Implementation (from deepencoderv2.py)
# ==============================================================================


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
  max_rel_dist = int(2 * max(q_size, k_size) - 1)
  if rel_pos.shape[0] != max_rel_dist:
    dtype = rel_pos.dtype
    rel_pos = rel_pos.to(torch.float32)
    rel_pos_resized = F.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
        size=max_rel_dist,
        mode="linear",
    ).to(dtype)
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
  else:
    rel_pos_resized = rel_pos

  q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
  k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
  relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

  return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
  q_h, q_w = q_size
  k_h, k_w = k_size
  Rh = get_rel_pos(q_h, k_h, rel_pos_h)
  Rw = get_rel_pos(q_w, k_w, rel_pos_w)

  B, _, dim = q.shape
  r_q = q.reshape(B, q_h, q_w, dim)
  rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
  rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
  rel_h = rel_h.unsqueeze(-1)
  rel_w = rel_w.unsqueeze(-2)
  rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
  rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

  return rel_h, rel_w


class PTPatchEmbed(nn.Module):

  def __init__(
      self,
      kernel_size: Tuple[int, int] = (16, 16),
      stride: Tuple[int, int] = (16, 16),
      in_chans: int = 3,
      embed_dim: int = 768,
  ) -> None:
    super().__init__()
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.proj(x)
    x = x.permute(0, 2, 3, 1)
    return x


class PTMLPBlock(nn.Module):

  def __init__(self, embedding_dim: int, mlp_dim: int) -> None:
    super().__init__()
    self.lin1 = nn.Linear(embedding_dim, mlp_dim)
    self.lin2 = nn.Linear(mlp_dim, embedding_dim)
    self.act = nn.GELU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.lin2(self.act(self.lin1(x)))


class PTAttention(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int = 8,
      qkv_bias: bool = True,
      use_rel_pos: bool = False,
      input_size: Optional[Tuple[int, int]] = None,
  ) -> None:
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim**-0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.proj = nn.Linear(dim, dim)

    self.use_rel_pos = use_rel_pos
    if self.use_rel_pos:
      assert input_size is not None
      self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
      self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

    rel_h, rel_w = None, None
    if self.use_rel_pos:
      rel_h, rel_w = add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

    q = q.view(B, self.num_heads, H * W, -1)
    k = k.view(B, self.num_heads, H * W, -1)
    v = v.view(B, self.num_heads, H * W, -1)

    if self.use_rel_pos:
      rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
      rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
      attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
      x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    else:
      x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    x = self.proj(x)
    return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
  B, H, W, C = x.shape
  pad_h = (window_size - H % window_size) % window_size
  pad_w = (window_size - W % window_size) % window_size
  if pad_h > 0 or pad_w > 0:
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
  Hp, Wp = H + pad_h, W + pad_w
  x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
  Hp, Wp = pad_hw
  H, W = hw
  B = windows.shape[0] // (Hp * Wp // window_size // window_size)
  x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
  if Hp > H or Wp > W:
    x = x[:, :H, :W, :].contiguous()
  return x


class PTBlock(nn.Module):

  def __init__(
      self,
      dim: int,
      num_heads: int,
      mlp_ratio: float = 4.0,
      qkv_bias: bool = True,
      window_size: int = 0,
      input_size: Optional[Tuple[int, int]] = None,
  ) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = PTAttention(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        use_rel_pos=True,
        input_size=input_size if window_size == 0 else (window_size, window_size),
    )
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = PTMLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
    self.window_size = window_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = x
    x = self.norm1(x)
    if self.window_size > 0:
      H, W = x.shape[1], x.shape[2]
      x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)
    if self.window_size > 0:
      x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    x = x + self.mlp(self.norm2(x))
    return x


class PTLayerNorm2d(nn.Module):

  def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(num_channels))
    self.bias = nn.Parameter(torch.zeros(num_channels))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.eps)
    x = self.weight[:, None, None] * x + self.bias[:, None, None]
    return x


class PTImageEncoderViT(nn.Module):

  def __init__(
      self,
      img_size: int = 1024,
      patch_size: int = 16,
      in_chans: int = 3,
      embed_dim: int = 768,
      depth: int = 12,
      num_heads: int = 12,
      mlp_ratio: float = 4.0,
      out_chans: int = 256,
      qkv_bias: bool = True,
      window_size: int = 14,
      global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
  ) -> None:
    super().__init__()
    self.patch_embed = PTPatchEmbed(
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
    self.blocks = nn.ModuleList()
    for i in range(depth):
      block = PTBlock(
          dim=embed_dim,
          num_heads=num_heads,
          mlp_ratio=mlp_ratio,
          qkv_bias=qkv_bias,
          window_size=window_size if i not in global_attn_indexes else 0,
          input_size=(img_size // patch_size, img_size // patch_size),
      )
      self.blocks.append(block)

    self.neck = nn.Sequential(
        nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
        PTLayerNorm2d(out_chans),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        PTLayerNorm2d(out_chans),
    )
    self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
    self.net_3 = nn.Conv2d(512, 896, kernel_size=3, stride=2, padding=1, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    tgt_size = x.shape[1]
    src_size = self.pos_embed.shape[1]
    if src_size != tgt_size:
      pos_embed_resized = self.pos_embed.permute(0, 3, 1, 2)
      pos_embed_resized = F.interpolate(
          pos_embed_resized, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False
      )
      pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1)
    else:
      pos_embed_resized = self.pos_embed
    x = x + pos_embed_resized
    for blk in self.blocks:
      x = blk(x)
    x = self.neck(x.permute(0, 3, 1, 2))
    x2 = self.net_2(x)
    x3 = self.net_3(x2)
    return x3


class PTCustomQwen2Decoder(nn.Module):

  def __init__(
      self,
      decoder_layer: int = 24,
      max_position_embeddings: int = 131072,
      hidden_dimension: int = 896,
      num_attention_heads: int = 14,
      num_key_value_heads: int = 2,
      intermediate_size: int = 4864,
      vocab_size: int = 151936,
      attn_implementation: str = "sdpa",
      rms_norm_eps: float = 1e-06,
      rope_theta: float = 1000000.0,
      attention_dropout: float = 0.0,
      hidden_act: str = "silu",
  ):
    super().__init__()
    Qwen2Model = getattr(transformers.models.qwen2.modeling_qwen2, "Qwen2Model")
    Qwen2Config = getattr(transformers, "Qwen2Config")

    config = Qwen2Config(
        hidden_size=hidden_dimension,
        num_hidden_layers=decoder_layer,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        attention_dropout=attention_dropout,
        hidden_act=hidden_act,
        _attn_implementation=attn_implementation,
    )

    self.model = self._create_custom_model(Qwen2Model, config)
    del self.model.embed_tokens

  def _create_custom_model(self, Qwen2Model, config):
    class CustomQwen2ModelInner(Qwen2Model):

      def forward(
          self,
          inputs_embeds=None,
          attention_mask=None,
          position_ids=None,
          past_key_values=None,
          token_type_ids=None,
          use_cache=None,
          output_attentions=None,
          output_hidden_states=None,
          return_dict=None,
          cache_position=None,
      ):
        self._current_token_type_ids = token_type_ids
        if token_type_ids is not None and not isinstance(attention_mask, dict):
          attention_mask = {
              "full_attention": self._update_causal_mask(
                  attention_mask,
                  inputs_embeds,
                  cache_position,
                  past_key_values,
                  output_attentions,
              )
          }
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

      def _update_causal_mask(
          self,
          attention_mask,
          input_tensor,
          cache_position,
          past_key_values,
          output_attentions,
      ):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]
        token_type_ids = self._current_token_type_ids

        causal_mask = self._create_custom_4d_mask(
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
            token_type_ids=token_type_ids,
        )

        if attention_mask is not None and attention_mask.dim() == 2:
          padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
          padding_mask = (1.0 - padding_mask) * min_dtype
          causal_mask = causal_mask + padding_mask

        return causal_mask

      def _create_custom_4d_mask(
          self,
          sequence_length,
          dtype,
          device,
          batch_size,
          token_type_ids,
      ):
        min_dtype = torch.finfo(dtype).min
        masks = []
        for b in range(batch_size):
          mask = torch.full((sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device)
          type_ids = token_type_ids[b]
          image_positions = (type_ids == 0).nonzero(as_tuple=True)[0]
          text_positions = (type_ids == 1).nonzero(as_tuple=True)[0]

          if len(image_positions) > 0:
            mask[image_positions[:, None], image_positions] = 0.0

          for i, text_pos in enumerate(text_positions):
            if len(image_positions) > 0:
              mask[text_pos, image_positions] = 0.0
            mask[text_pos, text_positions[: i + 1]] = 0.0

          masks.append(mask)

        mask = torch.stack(masks, dim=0).unsqueeze(1)
        return mask

    return CustomQwen2ModelInner(config)

  def forward(self, inputs_embeds, token_type_ids, attention_mask=None, **kwargs):
    return self.model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask, **kwargs)


class PTQwen2Decoder2Encoder(nn.Module):

  def __init__(
      self,
      decoder_layer: int,
      hidden_dimension: int,
      num_attention_heads: int,
      num_key_value_heads: int,
      intermediate_size: int,
  ):
    super().__init__()
    self.model = PTCustomQwen2Decoder(
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        attn_implementation="sdpa",
    )
    self.query_768 = nn.Embedding(144, hidden_dimension)
    self.query_1024 = nn.Embedding(256, hidden_dimension)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1, 2)
    bs, n_query, _ = x.shape
    if n_query == 144:
      param_img = self.query_768.weight
    elif n_query == 256:
      param_img = self.query_1024.weight

    batch_query_imgs = param_img.unsqueeze(0).expand(bs, -1, -1)
    x_combined = torch.cat([x, batch_query_imgs], dim=1)

    token_type_ids = torch.cat(
        [
            torch.zeros(bs, n_query, dtype=torch.long),
            torch.ones(bs, n_query, dtype=torch.long),
        ],
        dim=1,
    )

    y = self.model(x_combined, token_type_ids)[0]
    y = y[:, n_query:, :]
    return y


# ==============================================================================
# Test Cases
# ==============================================================================


class DeepseekOCRLayersTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 1
    self.decoder_seq_len = 5
    self.decoder_hidden_size = 32
    self.decoder_num_heads = 4
    self.decoder_num_kv_heads = 2
    self.decoder_head_dim = self.decoder_hidden_size // self.decoder_num_heads
    self.decoder_intermediate_size = 64

    self.config = pyconfig.initialize(
        [
            sys.argv[0],
            "src/maxtext/configs/base.yml",
            "run_name=test",
            "skip_jax_distributed_system=True",
            "attention=dot_product",
            "dtype=float32",
            "weight_dtype=float32",
            "attention_bias=True",
            "matmul_precision=highest",
            "rope_max_timescale=1000000.0",
            "normalization_layer_epsilon=1e-6",
            "base_emb_dim=1280",
            "base_num_query_heads=10",
            "base_num_kv_heads=10",
            "base_mlp_dim=6848",
            "base_num_decoder_layers=12",
            "vision_connector_emb_dim=896",
            "vision_connector_num_query_heads=14",
            "vision_connector_num_kv_heads=2",
            "vision_connector_mlp_dim=4864",
            "vision_connector_num_layers=24",
            "decoder_block=deepseek",
            "use_mla=false",
            "attention_type=global",
            "dropout_rate=0.0",
            "enable_dropout=false",
            "base_emb_dim=32",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "base_mlp_dim=64",
            "base_moe_mlp_dim=16",
            "base_num_decoder_layers=2",
            "first_num_dense_layers=1",
            "head_dim=8",
            "mlp_activations=['silu', 'linear']",
            "rope_interleave=false",
            "max_target_length=5",
            "max_prefill_predict_length=5",
            "per_device_batch_size=1",
            "global_batch_size_to_load=1",
            "global_batch_size_to_train_on=1",
            "num_experts=4",
            "num_experts_per_tok=2",
            "shared_experts=1",
            "routed_score_func=softmax",
            "routed_scaling_factor=1.0",
            "routed_bias=false",
            "sparse_matmul=false",
            "scan_layers=false",
            "fused_mlp=false",
        ]
    )
    self.mesh = Mesh(np.array(jax.devices()[:1]), ("data",))

  def _decoder_inputs(self):
    torch.manual_seed(123)
    x_pt = torch.randn(self.batch_size, self.decoder_seq_len, self.decoder_hidden_size)
    x_jax = to_jax(x_pt)
    positions = jnp.broadcast_to(jnp.arange(self.decoder_seq_len, dtype=jnp.int32), (self.batch_size, self.decoder_seq_len))
    segment_ids = jnp.ones((self.batch_size, self.decoder_seq_len), dtype=jnp.int32)
    return x_pt, x_jax, positions, segment_ids

  def _jax_dense_decoder_layer(self):
    return deepseek.DeepSeekDenseLayer(
        config=self.config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(0),
        quant=None,
        layer_idx=0,
    )

  def test_processor_expands_actual_visual_token_count(self):
    pixel_mask = np.zeros(
        (1, DEEPSEEK_OCR_GLOBAL_TOKENS + 6 * DEEPSEEK_OCR_CROP_TOKENS + DEEPSEEK_OCR_SEPARATOR_TOKENS),
        dtype=np.bool_,
    )
    pixel_mask[:, : 2 * DEEPSEEK_OCR_CROP_TOKENS] = True
    pixel_mask[:, 6 * DEEPSEEK_OCR_CROP_TOKENS :] = True
    processor_output = DeepseekOCR2PreprocessorOutput(
        pixel_values=np.zeros((1, 7, 1024, 1024, 3), dtype=np.float32),
        pixel_mask=pixel_mask,
        num_images=1,
    )

    tokens = np.array([11, DEEPSEEK_OCR_IMAGE_TOKEN_ID, 12], dtype=np.int32)
    expanded = add_extra_tokens_for_images_deepseek_ocr(tokens, processor_output)

    expected_visual_tokens = DEEPSEEK_OCR_GLOBAL_TOKENS + 2 * DEEPSEEK_OCR_CROP_TOKENS + DEEPSEEK_OCR_SEPARATOR_TOKENS
    self.assertEqual(int((expanded == DEEPSEEK_OCR_IMAGE_TOKEN_ID).sum()), expected_visual_tokens)
    self.assertEqual(expanded.shape[0], tokens.shape[0] + expected_visual_tokens - 1)
    self.assertEqual(get_image_offsets_deepseek_ocr(processor_output), expected_visual_tokens - 1)

  def test_merge_mm_embeddings_accepts_deepseek_token_level_mask(self):
    text_embeddings = jnp.zeros((1, 5, 2), dtype=jnp.float32)
    multimodal_embeddings = jnp.arange(10, dtype=jnp.float32).reshape(1, 1, 5, 2)
    text_mask = jnp.array([[False, True, True, True, False]])
    token_mask = jnp.array([[True, True, True, False, False]])

    merged = mm_utils.merge_mm_embeddings(text_embeddings, multimodal_embeddings, text_mask, token_mask)

    np.testing.assert_array_equal(np.asarray(merged[0, 1:4]), np.asarray(multimodal_embeddings[0, 0, :3]))
    np.testing.assert_array_equal(np.asarray(merged[0, 4]), np.zeros((2,), dtype=np.float32))

  def test_deepseek_ocr_text_config_stays_non_mla(self):
    self.assertEqual(self.config.decoder_block, DecoderBlockType.DEEPSEEK)
    self.assertFalse(self.config.use_mla)
    self.assertEqual(self.config.attention_type, "global")
    self.assertEqual(self.config.num_query_heads, self.decoder_num_heads)
    self.assertEqual(self.config.num_kv_heads, self.decoder_num_kv_heads)
    self.assertEqual(self.config.head_dim, self.decoder_head_dim)

  def test_dense_decoder_layer_forward_shape_and_finite(self):
    _, x_jax, positions, segment_ids = self._decoder_inputs()
    layer = self._jax_dense_decoder_layer()

    out, kv_cache = layer(
        x_jax,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNone(kv_cache)
    self.assertEqual(out.shape, (self.batch_size, self.decoder_seq_len, self.decoder_hidden_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_dense_decoder_layer_is_causal(self):
    _, x_jax, positions, segment_ids = self._decoder_inputs()
    layer = self._jax_dense_decoder_layer()

    out_a, _ = layer(
        x_jax,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    x_changed = x_jax.at[:, -1, :].add(25.0)
    out_b, _ = layer(
        x_changed,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    np.testing.assert_allclose(np.asarray(out_a[:, :-1, :]), np.asarray(out_b[:, :-1, :]), rtol=1e-5, atol=1e-5)
    self.assertGreater(float(jnp.max(jnp.abs(out_a[:, -1, :] - out_b[:, -1, :]))), 1e-3)

  def test_moe_decoder_layer_forward_shape_and_finite(self):
    _, x_jax, positions, segment_ids = self._decoder_inputs()
    layer = deepseek.DeepSeekMoELayer(
        config=self.config,
        model_mode=MODEL_MODE_TRAIN,
        mesh=self.mesh,
        rngs=nnx.Rngs(1),
        quant=None,
        layer_idx=1,
    )

    out, kv_cache = layer(
        x_jax,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertIsNone(kv_cache)
    self.assertEqual(out.shape, (self.batch_size, self.decoder_seq_len, self.decoder_hidden_size))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_one_layer_dense_forward_matches_hf_random_weights(self):
    torch.manual_seed(7)
    hf_layer, hf_rope = make_llama_decoder_layer(
        hidden_size=self.decoder_hidden_size,
        num_heads=self.decoder_num_heads,
        num_kv_heads=self.decoder_num_kv_heads,
        intermediate_size=self.decoder_intermediate_size,
        seq_len=self.decoder_seq_len,
    )
    jax_layer = self._jax_dense_decoder_layer()
    copy_llama_decoder_layer_weights(hf_layer, jax_layer)

    x_pt, x_jax, positions, segment_ids = self._decoder_inputs()
    position_ids_pt = torch.zeros((self.batch_size, self.decoder_seq_len), dtype=torch.long)
    positions = jnp.zeros_like(positions)
    causal_mask = torch.full(
        (self.batch_size, 1, self.decoder_seq_len, self.decoder_seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
      hf_out = hf_layer(
          x_pt,
          attention_mask=causal_mask,
          position_ids=position_ids_pt,
          position_embeddings=hf_rope(x_pt, position_ids_pt),
          use_cache=False,
      )

    jax_out, _ = jax_layer(
        x_jax,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    if isinstance(hf_out, tuple):
      hf_out = hf_out[0]
    np.testing.assert_allclose(np.asarray(to_jax(hf_out)), np.asarray(jax_out), rtol=1e-4, atol=1e-4)

  def test_sam_attention(self):
    dim = 768
    num_heads = 12
    input_size = (64, 64)
    batch_size = 2

    # PyTorch
    pt_layer = PTAttention(dim=dim, num_heads=num_heads, qkv_bias=True, use_rel_pos=True, input_size=input_size)
    pt_layer.eval()

    # JAX
    rngs = nnx.Rngs(0)
    jax_layer = deepseek_ocr.SAMAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        use_rel_pos=True,
        input_size=input_size,
        rngs=rngs,
    )

    # Copy weights
    jax_layer.qkv.kernel.value = to_jax(pt_layer.qkv.weight.T)
    jax_layer.qkv.bias.value = to_jax(pt_layer.qkv.bias)
    jax_layer.proj.kernel.value = to_jax(pt_layer.proj.weight.T)
    jax_layer.proj.bias.value = to_jax(pt_layer.proj.bias)
    jax_layer.rel_pos_h.value = to_jax(pt_layer.rel_pos_h)
    jax_layer.rel_pos_w.value = to_jax(pt_layer.rel_pos_w)

    # Input
    x_pt = torch.randn(batch_size, input_size[0], input_size[1], dim)
    x_jax = to_jax(x_pt)

    # Forward
    with torch.no_grad():
      out_pt = pt_layer(x_pt)
    out_jax = jax_layer(x_jax)

    # Compare
    np.testing.assert_allclose(to_jax(out_pt), out_jax, rtol=3e-4, atol=3e-4)

  def test_sam_block(self):
    dim = 768
    num_heads = 12
    input_size = (64, 64)
    batch_size = 2
    window_size = 14

    # PyTorch
    pt_layer = PTBlock(
        dim=dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, window_size=window_size, input_size=input_size
    )
    pt_layer.eval()

    # JAX
    rngs = nnx.Rngs(0)
    jax_layer = deepseek_ocr.SAMBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_rel_pos=True,
        window_size=window_size,
        input_size=input_size,
        rngs=rngs,
    )

    # Copy weights
    jax_layer.norm1.scale.value = to_jax(pt_layer.norm1.weight)
    jax_layer.norm1.bias.value = to_jax(pt_layer.norm1.bias)
    jax_layer.norm2.scale.value = to_jax(pt_layer.norm2.weight)
    jax_layer.norm2.bias.value = to_jax(pt_layer.norm2.bias)

    jax_layer.attn.qkv.kernel.value = to_jax(pt_layer.attn.qkv.weight.T)
    jax_layer.attn.qkv.bias.value = to_jax(pt_layer.attn.qkv.bias)
    jax_layer.attn.proj.kernel.value = to_jax(pt_layer.attn.proj.weight.T)
    jax_layer.attn.proj.bias.value = to_jax(pt_layer.attn.proj.bias)
    jax_layer.attn.rel_pos_h.value = to_jax(pt_layer.attn.rel_pos_h)
    jax_layer.attn.rel_pos_w.value = to_jax(pt_layer.attn.rel_pos_w)

    jax_layer.lin1.kernel.value = to_jax(pt_layer.mlp.lin1.weight.T)
    jax_layer.lin1.bias.value = to_jax(pt_layer.mlp.lin1.bias)
    jax_layer.lin2.kernel.value = to_jax(pt_layer.mlp.lin2.weight.T)
    jax_layer.lin2.bias.value = to_jax(pt_layer.mlp.lin2.bias)

    # Input
    x_pt = torch.randn(batch_size, input_size[0], input_size[1], dim)
    x_jax = to_jax(x_pt)

    # Forward
    with torch.no_grad():
      out_pt = pt_layer(x_pt)
    out_jax = jax_layer(x_jax)

    # Compare
    np.testing.assert_allclose(to_jax(out_pt), out_jax, rtol=4e-3, atol=4e-3)

  def test_sam_vit_b(self):
    batch_size = 2
    img_size = 1024

    # PyTorch
    pt_layer = PTImageEncoderViT(img_size=img_size)
    pt_layer.eval()

    # JAX
    rngs = nnx.Rngs(0)
    jax_layer = deepseek_ocr.SAMViTB(config=self.config, mesh=self.mesh, rngs=rngs)

    # Copy weights
    jax_layer.patch_embed.kernel.value = to_jax(pt_layer.patch_embed.proj.weight.permute(2, 3, 1, 0))
    jax_layer.patch_embed.bias.value = to_jax(pt_layer.patch_embed.proj.bias)
    jax_layer.pos_embed.value = to_jax(pt_layer.pos_embed)

    for i in range(12):
      pt_blk = pt_layer.blocks[i]
      jax_blk = getattr(jax_layer, f"block_{i}")

      jax_blk.norm1.scale.value = to_jax(pt_blk.norm1.weight)
      jax_blk.norm1.bias.value = to_jax(pt_blk.norm1.bias)
      jax_blk.norm2.scale.value = to_jax(pt_blk.norm2.weight)
      jax_blk.norm2.bias.value = to_jax(pt_blk.norm2.bias)

      jax_blk.attn.qkv.kernel.value = to_jax(pt_blk.attn.qkv.weight.T)
      jax_blk.attn.qkv.bias.value = to_jax(pt_blk.attn.qkv.bias)
      jax_blk.attn.proj.kernel.value = to_jax(pt_blk.attn.proj.weight.T)
      jax_blk.attn.proj.bias.value = to_jax(pt_blk.attn.proj.bias)
      jax_blk.attn.rel_pos_h.value = to_jax(pt_blk.attn.rel_pos_h)
      jax_blk.attn.rel_pos_w.value = to_jax(pt_blk.attn.rel_pos_w)

      jax_blk.lin1.kernel.value = to_jax(pt_blk.mlp.lin1.weight.T)
      jax_blk.lin1.bias.value = to_jax(pt_blk.mlp.lin1.bias)
      jax_blk.lin2.kernel.value = to_jax(pt_blk.mlp.lin2.weight.T)
      jax_blk.lin2.bias.value = to_jax(pt_blk.mlp.lin2.bias)

    jax_layer.neck_conv1.kernel.value = to_jax(pt_layer.neck[0].weight.permute(2, 3, 1, 0))
    jax_layer.neck_ln1.scale.value = to_jax(pt_layer.neck[1].weight)
    jax_layer.neck_ln1.bias.value = to_jax(pt_layer.neck[1].bias)
    jax_layer.neck_conv2.kernel.value = to_jax(pt_layer.neck[2].weight.permute(2, 3, 1, 0))
    jax_layer.neck_ln2.scale.value = to_jax(pt_layer.neck[3].weight)
    jax_layer.neck_ln2.bias.value = to_jax(pt_layer.neck[3].bias)

    jax_layer.net_2.kernel.value = to_jax(pt_layer.net_2.weight.permute(2, 3, 1, 0))
    jax_layer.net_3.kernel.value = to_jax(pt_layer.net_3.weight.permute(2, 3, 1, 0))

    # Input (PyTorch expects NCHW, JAX expects NHWC)
    x_pt = torch.randn(batch_size, 3, img_size, img_size)
    x_jax = to_jax(x_pt.permute(0, 2, 3, 1))

    # Forward
    with torch.no_grad():
      out_pt = pt_layer(x_pt)
    out_jax = jax_layer(x_jax)

    # Compare
    out_pt_nhwc = out_pt.permute(0, 2, 3, 1)
    np.testing.assert_allclose(to_jax(out_pt_nhwc), out_jax, rtol=1.2e-2, atol=1.2e-2)

  def test_qwen2_encoder_layer(self):
    # Config
    pt_config = PTQwen2Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        rms_norm_eps=1e-6,
    )

    # PyTorch
    pt_layer = PTQwen2DecoderLayer(pt_config, layer_idx=0)
    pt_layer.eval()

    # JAX
    from maxtext.configs.pyconfig import HyperParameters

    pydantic_config = self.config._pydantic_config
    update_dict = {
        "decoder_block": DecoderBlockType.QWEN2,
        "emb_dim": 896,
        "num_query_heads": 14,
        "num_kv_heads": 2,
        "mlp_dim": 4864,
        "head_dim": 896 // 14,
        "max_target_length": 288,
        "max_prefill_predict_length": 288,
    }
    if hasattr(pydantic_config, "model_copy"):
      new_pydantic_config = pydantic_config.model_copy(update=update_dict)
    else:
      new_pydantic_config = pydantic_config.copy(update=update_dict)
    connector_config = HyperParameters(new_pydantic_config)
    rngs = nnx.Rngs(0)
    jax_layer = deepseek_ocr.Qwen2EncoderLayer(
        config=connector_config,
        mesh=self.mesh,
        quant=None,
        rngs=rngs,
    )

    # Copy weights
    copy_rmsnorm_weights(pt_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
    copy_rmsnorm_weights(pt_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)
    copy_qwen2_attention_weights(pt_layer.self_attn, jax_layer.self_attention)
    copy_linear_weights(pt_layer.mlp.gate_proj, jax_layer.mlp.wi_0)
    copy_linear_weights(pt_layer.mlp.up_proj, jax_layer.mlp.wi_1)
    copy_linear_weights(pt_layer.mlp.down_proj, jax_layer.mlp.wo)

    # Input
    batch_size = 1
    seq_len = 288
    x_pt = torch.randn(batch_size, seq_len, 896)
    x_jax = to_jax(x_pt)

    # Mask & Positions
    token_type_ids = torch.cat(
        [
            torch.zeros(batch_size, 144, dtype=torch.long),
            torch.ones(batch_size, 144, dtype=torch.long),
        ],
        dim=1,
    )

    decoder = PTCustomQwen2Decoder(
        decoder_layer=1,
        hidden_dimension=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
    )
    decoder.model.layers[0].load_state_dict(pt_layer.state_dict())
    decoder.eval()

    with torch.no_grad():
      decoder.model._current_token_type_ids = token_type_ids
      position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
      mask_pt = decoder.model._update_causal_mask(None, x_pt, None, None, None)
      position_embeddings = decoder.model.rotary_emb(x_pt, position_ids)

      # Run PyTorch Layer directly
      out_pt = pt_layer(
          hidden_states=x_pt,
          attention_mask=mask_pt,
          position_ids=position_ids,
          position_embeddings=position_embeddings,
      )[0]

    # JAX Run
    mask_image = jnp.ones((batch_size, 144), dtype=jnp.bool_)
    mask_query = jnp.zeros((batch_size, 144), dtype=jnp.bool_)
    bidirectional_mask = jnp.concatenate([mask_image, mask_query], axis=1)
    decoder_positions = jnp.arange(seq_len)[None, :]

    out_jax = jax_layer(
        x_jax,
        bidirectional_mask=bidirectional_mask,
        decoder_positions=decoder_positions,
        deterministic=True,
    )

    if out_pt.ndim != out_jax.ndim:
      out_pt_compared = out_pt.unsqueeze(0)
    else:
      out_pt_compared = out_pt
    np.testing.assert_allclose(to_jax(out_pt_compared), out_jax, rtol=1.5e-3, atol=1.5e-3)

  def test_qwen2_decoder2encoder(self):
    batch_size = 1

    # PyTorch
    pt_model = PTQwen2Decoder2Encoder(
        decoder_layer=2,  # Use 2 layers for faster test
        hidden_dimension=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
    )
    pt_model.eval()

    # JAX
    from maxtext.configs.pyconfig import HyperParameters

    pydantic_config = self.config._pydantic_config
    update_dict = {
        "decoder_block": DecoderBlockType.QWEN2,
        "vision_connector_num_layers": 2,
        "max_target_length": 512,
        "max_prefill_predict_length": 512,
    }
    if hasattr(pydantic_config, "model_copy"):
      new_pydantic_config = pydantic_config.model_copy(update=update_dict)
    else:
      new_pydantic_config = pydantic_config.copy(update=update_dict)
    test_config = HyperParameters(new_pydantic_config)
    rngs = nnx.Rngs(0)
    jax_model = deepseek_ocr.Qwen2Decoder2Encoder(
        config=test_config,
        mesh=self.mesh,
        quant=None,
        rngs=rngs,
    )

    # Copy weights
    jax_model.query_768.embedding.value = to_jax(pt_model.query_768.weight)
    jax_model.query_1024.embedding.value = to_jax(pt_model.query_1024.weight)
    copy_rmsnorm_weights(pt_model.model.model.norm, jax_model.norm)

    for i in range(2):
      pt_layer = pt_model.model.model.layers[i]
      jax_layer = getattr(jax_model, f"layer_{i}")

      copy_rmsnorm_weights(pt_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
      copy_rmsnorm_weights(pt_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)
      copy_qwen2_attention_weights(pt_layer.self_attn, jax_layer.self_attention)
      copy_linear_weights(pt_layer.mlp.gate_proj, jax_layer.mlp.wi_0)
      copy_linear_weights(pt_layer.mlp.up_proj, jax_layer.mlp.wi_1)
      copy_linear_weights(pt_layer.mlp.down_proj, jax_layer.mlp.wo)

    for H, W in [(12, 12)]:
      x_pt = torch.randn(batch_size, H, W, 896)
      x_jax = to_jax(x_pt)

      with torch.no_grad():
        out_pt = pt_model(x_pt)
      out_jax = jax_model(x_jax)
      np.testing.assert_allclose(to_jax(out_pt), out_jax, rtol=1.5e-3, atol=1.5e-3)

  def test_image_preprocessing(self):
    # Create a dummy image with a specific size that triggers cropping
    # e.g., 1600 x 1200
    np.random.seed(42)
    random_array = np.random.randint(0, 256, (1200, 1600, 3), dtype=np.uint8)
    image = Image.fromarray(random_array)

    # HF Reference
    hf_res = _hf_reference_inputs(image)

    # MaxText
    from maxtext.multimodal.processor_deepseek_ocr import preprocess_mm_data_deepseek_ocr
    mt_res = preprocess_mm_data_deepseek_ocr(image)

    # Compare aspect ratios
    np.testing.assert_array_equal(hf_res["crop_ratio"], mt_res.aspect_ratios[0])

    # Compare global view
    mt_global = mt_res.pixel_values[0, 0]
    np.testing.assert_allclose(hf_res["images_ori"], mt_global, atol=1e-5)

    # Compare crops
    num_crops = hf_res["images_crop"].shape[0]
    for i in range(num_crops):
      mt_crop = mt_res.pixel_values[0, i + 1, :768, :768, :]
      np.testing.assert_allclose(hf_res["images_crop"][i], mt_crop, atol=1e-5)

  def test_deepseek_ocr_vision_encoder(self):
    # JAX Config override for fast test
    from maxtext.configs.pyconfig import HyperParameters
    pydantic_config = self.config._pydantic_config
    update_dict = {
        "decoder_block": DecoderBlockType.QWEN2,
        "vision_connector_num_layers": 2,
        "max_target_length": 512,
        "max_prefill_predict_length": 512,
    }
    if hasattr(pydantic_config, "model_copy"):
      new_pydantic_config = pydantic_config.model_copy(update=update_dict)
    else:
      new_pydantic_config = pydantic_config.copy(update=update_dict)
    test_config = HyperParameters(new_pydantic_config)

    rngs = nnx.Rngs(0)
    jax_model = deepseek_ocr.DeepseekOCR2VisionEncoder(config=test_config, mesh=self.mesh, rngs=rngs)

    # PT Models
    pt_sam = PTImageEncoderViT(img_size=1024)
    pt_qwen = PTQwen2Decoder2Encoder(
        decoder_layer=2,
        hidden_dimension=896,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
    )
    pt_sam.eval()
    pt_qwen.eval()

    # Copy weights
    # SAM
    jax_model.sam_model.patch_embed.kernel.value = to_jax(pt_sam.patch_embed.proj.weight.permute(2, 3, 1, 0))
    jax_model.sam_model.patch_embed.bias.value = to_jax(pt_sam.patch_embed.proj.bias)
    jax_model.sam_model.pos_embed.value = to_jax(pt_sam.pos_embed)
    for i in range(12):
      pt_blk = pt_sam.blocks[i]
      jax_blk = getattr(jax_model.sam_model, f"block_{i}")
      jax_blk.norm1.scale.value = to_jax(pt_blk.norm1.weight)
      jax_blk.norm1.bias.value = to_jax(pt_blk.norm1.bias)
      jax_blk.norm2.scale.value = to_jax(pt_blk.norm2.weight)
      jax_blk.norm2.bias.value = to_jax(pt_blk.norm2.bias)
      jax_blk.attn.qkv.kernel.value = to_jax(pt_blk.attn.qkv.weight.T)
      jax_blk.attn.qkv.bias.value = to_jax(pt_blk.attn.qkv.bias)
      jax_blk.attn.proj.kernel.value = to_jax(pt_blk.attn.proj.weight.T)
      jax_blk.attn.proj.bias.value = to_jax(pt_blk.attn.proj.bias)
      jax_blk.attn.rel_pos_h.value = to_jax(pt_blk.attn.rel_pos_h)
      jax_blk.attn.rel_pos_w.value = to_jax(pt_blk.attn.rel_pos_w)
      jax_blk.lin1.kernel.value = to_jax(pt_blk.mlp.lin1.weight.T)
      jax_blk.lin1.bias.value = to_jax(pt_blk.mlp.lin1.bias)
      jax_blk.lin2.kernel.value = to_jax(pt_blk.mlp.lin2.weight.T)
      jax_blk.lin2.bias.value = to_jax(pt_blk.mlp.lin2.bias)

    jax_model.sam_model.neck_conv1.kernel.value = to_jax(pt_sam.neck[0].weight.permute(2, 3, 1, 0))
    jax_model.sam_model.neck_ln1.scale.value = to_jax(pt_sam.neck[1].weight)
    jax_model.sam_model.neck_ln1.bias.value = to_jax(pt_sam.neck[1].bias)
    jax_model.sam_model.neck_conv2.kernel.value = to_jax(pt_sam.neck[2].weight.permute(2, 3, 1, 0))
    jax_model.sam_model.neck_ln2.scale.value = to_jax(pt_sam.neck[3].weight)
    jax_model.sam_model.neck_ln2.bias.value = to_jax(pt_sam.neck[3].bias)
    jax_model.sam_model.net_2.kernel.value = to_jax(pt_sam.net_2.weight.permute(2, 3, 1, 0))
    jax_model.sam_model.net_3.kernel.value = to_jax(pt_sam.net_3.weight.permute(2, 3, 1, 0))

    # Qwen2
    jax_model.qwen2_model.query_768.embedding.value = to_jax(pt_qwen.query_768.weight)
    jax_model.qwen2_model.query_1024.embedding.value = to_jax(pt_qwen.query_1024.weight)
    copy_rmsnorm_weights(pt_qwen.model.model.norm, jax_model.qwen2_model.norm)
    for i in range(2):
      pt_layer = pt_qwen.model.model.layers[i]
      jax_layer = getattr(jax_model.qwen2_model, f"layer_{i}")
      copy_rmsnorm_weights(pt_layer.input_layernorm, jax_layer.pre_self_attention_layer_norm)
      copy_rmsnorm_weights(pt_layer.post_attention_layernorm, jax_layer.post_self_attention_layer_norm)
      copy_qwen2_attention_weights(pt_layer.self_attn, jax_layer.self_attention)
      copy_linear_weights(pt_layer.mlp.gate_proj, jax_layer.mlp.wi_0)
      copy_linear_weights(pt_layer.mlp.up_proj, jax_layer.mlp.wi_1)
      copy_linear_weights(pt_layer.mlp.down_proj, jax_layer.mlp.wo)

    # Input: 1 image, 6 crops + 1 global (total 7 views)
    # JAX expects [B, N, 7, H, W, C]
    x_jax = jax.random.normal(jax.random.PRNGKey(42), (1, 1, 7, 1024, 1024, 3))
    x_pt_global = torch.from_numpy(np.array(x_jax[0, 0, 0].transpose(2, 0, 1))).unsqueeze(0)
    x_pt_crops = torch.from_numpy(np.array(x_jax[0, 0, 1:, :768, :768, :].transpose(0, 3, 1, 2)))

    # Run PT
    with torch.no_grad():
      sam_global = pt_sam(x_pt_global)
      qwen_global = pt_qwen(sam_global.permute(0, 2, 3, 1))

      sam_crops = pt_sam(x_pt_crops)
      qwen_crops = pt_qwen(sam_crops.permute(0, 2, 3, 1))

      pt_out = torch.cat([qwen_crops.reshape(864, 896), qwen_global.reshape(256, 896)], dim=0).unsqueeze(0).unsqueeze(0)

    # Run JAX
    jax_out = jax_model(x_jax, deterministic=True)

    # Compare (uses tolerance of 1e-2 to account for cumulative numerical error across SAM and Qwen2)
    np.testing.assert_allclose(to_jax(pt_out), jax_out, rtol=1e-2, atol=1e-2)

  def test_text_image_embedding_merge_realistic(self):
    text_embeddings = jnp.arange(1024 * 128, dtype=jnp.float32).reshape(1, 1024, 128)
    multimodal_embeddings = (jnp.arange(1121 * 128, dtype=jnp.float32) + 1000000.0).reshape(1, 1, 1121, 128)

    mask = np.zeros((1, 1024), dtype=np.bool_)
    mask[0, 100:933] = True
    mask = jnp.array(mask)

    pixel_mask = np.zeros((1, 1121), dtype=np.bool_)
    pixel_mask[0, :576] = True
    pixel_mask[0, 864:1121] = True
    pixel_mask = jnp.array(pixel_mask)

    merged = mm_utils.merge_mm_embeddings(
        text_embeddings=text_embeddings,
        multimodal_embeddings=multimodal_embeddings,
        mask=mask,
        token_masks=pixel_mask,
    )

    active_mm = multimodal_embeddings[0, 0][pixel_mask[0]]
    np.testing.assert_allclose(merged[0, 100:933], active_mm, atol=1e-5)
    np.testing.assert_allclose(merged[0, :100], text_embeddings[0, :100], atol=1e-5)
    np.testing.assert_allclose(merged[0, 933:], text_embeddings[0, 933:], atol=1e-5)


if __name__ == "__main__":
  unittest.main()
