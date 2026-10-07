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

"""Qwen3-VL Multimodal Model Family in m3 format.

Self-contained NNX implementation for multimodal forward passes:
- No imports from maxtext.layers.
- NNX parameters and built-ins preserve existing checkpoint paths.
- Pure RoPE functions: apply_vision_rope (2D) and apply_mrope (3D).
- Qwen3VLDecoder inherits directly from Qwen3Decoder.
- Reuses Qwen3MLP from qwen3.
- Parameter hierarchy matches legacy checkpoints (Qwen3VLVisionEncoder_0, Qwen3VLVisionProjector_0).
"""

from typing import Any, Optional, Sequence, Union
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from maxtext.common.common_types import Config, MultimodalInput
from maxtext.m3.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3Decoder,
)


# ---------------------------------------------------------------------------
# Core Pure Linear Layer
# ---------------------------------------------------------------------------


class DenseGeneral(nnx.Module):
  """Self-contained linear layer matching legacy checkpoint kernel/bias structure."""

  def __init__(
      self,
      in_features_shape: Union[int, tuple[int, ...]],
      out_features_shape: Union[int, tuple[int, ...]],
      use_bias: bool = True,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.in_features_shape = (in_features_shape,) if isinstance(in_features_shape, int) else tuple(in_features_shape)
    self.out_features_shape = (out_features_shape,) if isinstance(out_features_shape, int) else tuple(out_features_shape)
    self.use_bias = use_bias
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    kernel_shape = self.in_features_shape + self.out_features_shape
    bias_shape = self.out_features_shape

    if rngs is not None:
      fan_in = np.prod(self.in_features_shape)
      init_fn = nnx.initializers.normal(stddev=float(fan_in) ** -0.5)
      self.kernel = nnx.Param(init_fn(rngs.params(), kernel_shape, weight_dtype))
      if use_bias:
        self.bias = nnx.Param(jnp.zeros(bias_shape, dtype=weight_dtype))
      else:
        self.bias = None
    else:
      self.kernel = nnx.Param(jnp.zeros(kernel_shape, dtype=weight_dtype))
      if use_bias:
        self.bias = nnx.Param(jnp.zeros(bias_shape, dtype=weight_dtype))
      else:
        self.bias = None

  def __call__(self, x: jax.Array) -> jax.Array:
    num_in_axes = len(self.in_features_shape)
    x_in_axes = tuple(range(x.ndim - num_in_axes, x.ndim))
    kernel_in_axes = tuple(range(num_in_axes))
    out = jnp.tensordot(x.astype(self.dtype), self.kernel[...].astype(self.dtype), axes=(x_in_axes, kernel_in_axes))
    if self.bias is not None:
      out = out + self.bias[...].astype(self.dtype)
    return out.astype(self.dtype)


# ---------------------------------------------------------------------------
# Pure RoPE Mathematics: Vision 2D RoPE & Language 3D M-RoPE
# ---------------------------------------------------------------------------


def generate_block_coords(
    video_grid_thw: jax.Array, max_patches: int, merge_size: int = 2
) -> tuple[jax.Array, jax.Array]:
  """Generate row and col coordinates in block-based order for padded video."""
  V_T = video_grid_thw[:, 0:1]
  V_H = video_grid_thw[:, 1:2]
  V_W = video_grid_thw[:, 2:3]

  merged_w = V_W // merge_size
  V_len = V_T * V_H * V_W

  idx = jnp.arange(max_patches, dtype=jnp.int32)[None, :]
  is_valid = idx < V_len

  stride_lh = V_H * V_W
  safe_stride_lh = jnp.maximum(stride_lh, 1)
  s_idx = idx % safe_stride_lh

  intra_col = s_idx % merge_size
  intra_row = (s_idx // merge_size) % merge_size

  safe_merged_w = jnp.maximum(merged_w, 1)
  block_elements = merge_size * merge_size
  block_col = (s_idx // block_elements) % safe_merged_w
  block_row = s_idx // (safe_merged_w * block_elements)

  row = jnp.where(is_valid, block_row * merge_size + intra_row, 0)
  col = jnp.where(is_valid, block_col * merge_size + intra_col, 0)

  return row, col


def apply_vision_rope(
    inputs: jax.Array,
    num_frames: int,
    height: int,
    width: int,
    spatial_merge_size: int = 2,
    rope_theta: float = 10000.0,
    valid_grid: Optional[tuple[int, int, int] | jax.Array] = None,
    token_mask: Optional[jax.Array] = None,
) -> jax.Array:
  """Applies 2D rotary position embeddings to vision patch tokens."""
  is_3d = inputs.ndim == 3
  if is_3d:
    inputs = inputs[None, :, :, :]

  batch_size = inputs.shape[0]
  head_dim = inputs.shape[-1]
  max_patches = num_frames * height * width

  if valid_grid is not None:
    if isinstance(valid_grid, (tuple, list)):
      valid_grid_arr = jnp.array([valid_grid], dtype=jnp.int32)
    elif valid_grid.ndim == 1:
      valid_grid_arr = valid_grid[None, :]
    else:
      valid_grid_arr = valid_grid
  else:
    valid_grid_arr = jnp.tile(
        jnp.array([[num_frames, height, width]], dtype=jnp.int32),
        (batch_size, 1),
    )

  row, col = generate_block_coords(valid_grid_arr, max_patches, spatial_merge_size)

  max_hw = max(height, width)
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim // 2, 2, dtype=jnp.float32) / (head_dim // 2)))
  positions = jnp.arange(max_hw, dtype=jnp.float32)
  freq_table = jnp.outer(positions, inv_freq)

  row_freqs = freq_table[row]
  col_freqs = freq_table[col]
  embeddings = jnp.concatenate([row_freqs, col_freqs], axis=-1)
  embeddings = jnp.concatenate([embeddings, embeddings], axis=-1)

  cos_emb = jnp.cos(embeddings)
  sin_emb = jnp.sin(embeddings)

  if token_mask is not None:
    is_valid = token_mask[:, :, None]
    cos_emb = jnp.where(is_valid, cos_emb, 1.0)
    sin_emb = jnp.where(is_valid, sin_emb, 0.0)

  cos_emb = cos_emb[:, :, None, :].astype(inputs.dtype)
  sin_emb = sin_emb[:, :, None, :].astype(inputs.dtype)

  x1 = inputs[..., : head_dim // 2]
  x2 = inputs[..., head_dim // 2 :]
  rotated = jnp.concatenate([-x2, x1], axis=-1)

  out = inputs * cos_emb + rotated * sin_emb
  if is_3d:
    out = out[0]
  return out


def _apply_interleaved_mrope(freqs: jax.Array, mrope_section: tuple[int, int, int]) -> jax.Array:
  freqs_t = freqs[..., 0, :]
  for dim_idx, offset in enumerate([1, 2], start=1):
    section_size = mrope_section[dim_idx] * 3
    idx = slice(offset, section_size, 3)
    freqs_t = freqs_t.at[..., idx].set(freqs[..., dim_idx, idx])
  return freqs_t


def apply_mrope(
    x: jax.Array,
    positions: jax.Array,
    mrope_section: tuple[int, int, int] = (24, 20, 20),
    rope_theta: float = 5000000.0,
) -> jax.Array:
  """Applies 3D Multimodal Rotary Position Embeddings (M-RoPE) matching HuggingFace and MaxText."""

  b, _, _, d = x.shape
  half_dim = d // 2

  # The preprocessor supplies (batch, sequence, 3); text-only positions are (batch, sequence).
  if positions.ndim == 3 and positions.shape[-1] == 1:
    positions = jnp.squeeze(positions, axis=-1)
  if positions.ndim == 2:
    positions = jnp.broadcast_to(positions[..., None], positions.shape + (3,))
  if positions.shape != (b, x.shape[1], 3):
    raise ValueError(f"positions must have shape {(b, x.shape[1], 3)}, got {positions.shape}")

  fraction = 2.0 * jnp.arange(0, half_dim, 1, dtype=jnp.float32) / d
  timescale = 1.0 * (rope_theta / 1.0) ** fraction
  inv_freq = 1.0 / timescale

  freqs = positions[..., None] * inv_freq[None, None, None, :]
  freqs = _apply_interleaved_mrope(freqs, mrope_section)

  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb)[:, :, None, :].astype(x.dtype)
  sin = jnp.sin(emb)[:, :, None, :].astype(x.dtype)

  x1 = x[..., :half_dim]
  x2 = x[..., half_dim:]
  rotated = jnp.concatenate([-x2, x1], axis=-1)
  return (x * cos) + (rotated * sin)


# ---------------------------------------------------------------------------
# Self-contained Vision Tower Components
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchEmbed(nnx.Module):
  """3D convolution-based patch embedding for vision inputs."""

  def __init__(
      self,
      config: Config,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.patch_size = config.patch_size_for_vit
    self.temporal_patch_size = config.temporal_patch_size_for_vit
    self.in_channels = config.num_channels_for_vit
    self.embed_dim = config.hidden_size_for_vit

    kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)

    self.proj = nnx.Conv(
        in_features=self.in_channels,
        out_features=self.embed_dim,
        kernel_size=kernel_size,
        strides=kernel_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self, hidden_states: jax.Array, video_mask: Optional[jax.Array] = None
  ) -> tuple[jax.Array, Optional[jax.Array]]:
    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    hidden_states = self.proj(hidden_states)
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1] * hidden_states.shape[2] * hidden_states.shape[3]
    hidden_states = hidden_states.reshape(batch_size, seq_len, self.embed_dim)

    attention_mask = None
    if video_mask is not None:
      mask_patch_elements = self.temporal_patch_size * self.patch_size * self.patch_size
      attention_mask = video_mask.reshape(video_mask.shape[0], -1, mask_patch_elements).max(axis=-1).astype(jnp.int32)

    return hidden_states, attention_mask


class Qwen3VLVisionPosEmbedInterpolate(nnx.Module):
  """Self-contained bilinear interpolation of learned 2D positional embeddings."""

  def __init__(
      self,
      num_position_embeddings: int = 2304,
      hidden_size: int = 1024,
      spatial_merge_size: int = 2,
      dtype: Any = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.num_position_embeddings = num_position_embeddings
    self.hidden_size = hidden_size
    self.spatial_merge_size = spatial_merge_size
    self.dtype = dtype
    self.num_grid_per_side = int(num_position_embeddings**0.5)

    if rngs is not None:
      init_fn = nnx.initializers.normal(stddev=self.hidden_size**-0.5)
      self.pos_embed = nnx.Param(init_fn(rngs.params(), (self.num_position_embeddings, self.hidden_size), self.dtype))
    else:
      self.pos_embed = nnx.Param(jnp.zeros((self.num_position_embeddings, self.hidden_size), dtype=self.dtype))

  def __call__(
      self,
      num_frames: int,
      height: int,
      width: int,
      video_grid_thw: Optional[jax.Array] = None,
      attention_mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    if video_grid_thw is not None:
      if isinstance(video_grid_thw, (tuple, list)):
        video_grid_thw = jnp.array([video_grid_thw], dtype=jnp.int32)
      elif video_grid_thw.ndim == 1:
        video_grid_thw = video_grid_thw[None, :]
      batch_size = video_grid_thw.shape[0]
    elif attention_mask is not None:
      batch_size = attention_mask.shape[0]
    else:
      batch_size = 1

    max_patches = num_frames * height * width

    if video_grid_thw is None:
      video_grid_thw = jnp.tile(
          jnp.array([[num_frames, height, width]], dtype=jnp.int32),
          (batch_size, 1),
      )
    if attention_mask is None:
      attention_mask = jnp.ones((batch_size, max_patches), dtype=jnp.int32)

    row, col = generate_block_coords(video_grid_thw, max_patches, self.spatial_merge_size)
    V_H = video_grid_thw[:, 1:2]
    V_W = video_grid_thw[:, 2:3]
    row_norm = row / jnp.maximum(V_H - 1, 1)
    col_norm = col / jnp.maximum(V_W - 1, 1)

    N = self.num_grid_per_side
    table = self.pos_embed[...].reshape(N, N, self.hidden_size)

    y = row_norm * (N - 1)
    x = col_norm * (N - 1)

    y0 = jnp.floor(y).astype(jnp.int32)
    x0 = jnp.floor(x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, N - 1)
    x1 = jnp.minimum(x0 + 1, N - 1)

    dy = (y - y0)[:, :, None]
    dx = (x - x0)[:, :, None]

    embed_00 = table[y0, x0]
    embed_01 = table[y0, x1]
    embed_10 = table[y1, x0]
    embed_11 = table[y1, x1]

    interpolated = (
        (1.0 - dy) * (1.0 - dx) * embed_00 + (1.0 - dy) * dx * embed_01 + dy * (1.0 - dx) * embed_10 + dy * dx * embed_11
    )
    return interpolated * attention_mask[:, :, None]


class Qwen3VLVisionAttn(nnx.Module):
  """Self-contained Q/K/V/Out projection matching legacy checkpoint parameter paths."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.num_heads = config.num_attention_heads_for_vit
    self.head_dim = config.hidden_size_for_vit // self.num_heads
    self.hidden_size = config.hidden_size_for_vit

    self.query = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=(self.num_heads, self.head_dim),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.key = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=(self.num_heads, self.head_dim),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.value = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=(self.num_heads, self.head_dim),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.out = DenseGeneral(
        in_features_shape=(self.num_heads, self.head_dim),
        out_features_shape=self.hidden_size,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )


class Qwen3VLVisionAttention(nnx.Module):
  """Vision attention wrapper matching legacy checkpoint hierarchy."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.num_heads = config.num_attention_heads_for_vit
    self.head_dim = config.hidden_size_for_vit // self.num_heads
    self.hidden_size = config.hidden_size_for_vit
    self.attn = Qwen3VLVisionAttn(config, rngs=rngs)

  def __call__(
      self,
      hidden_states: jax.Array,
      num_frames: int,
      height: int,
      width: int,
      attention_mask: Optional[jax.Array] = None,
      valid_grid: Optional[tuple[int, int, int]] = None,
  ) -> jax.Array:
    q = self.attn.query(hidden_states)  # (b, s, num_heads, head_dim)
    k = self.attn.key(hidden_states)  # (b, s, num_heads, head_dim)
    v = self.attn.value(hidden_states)  # (b, s, num_heads, head_dim)

    q = apply_vision_rope(
        q,
        num_frames,
        height,
        width,
        self.config.spatial_merge_size_for_vit,
        self.config.rope_theta_for_vit,
        valid_grid=valid_grid,
        token_mask=attention_mask,
    )
    k = apply_vision_rope(
        k,
        num_frames,
        height,
        width,
        self.config.spatial_merge_size_for_vit,
        self.config.rope_theta_for_vit,
        valid_grid=valid_grid,
        token_mask=attention_mask,
    )

    q = q * (self.head_dim**-0.5)

    score_dtype = jnp.float32 if self.config.float32_logits else q.dtype
    scores = jnp.einsum("bshd,bthd->bhst", q.astype(score_dtype), k.astype(score_dtype))
    if attention_mask is not None:
      mask = attention_mask[:, None, None, :]
      scores = jnp.where(mask, scores, -1e10)

    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    attn_out = jnp.einsum("bhst,bthd->bshd", weights, v)
    return self.attn.out(attn_out)


class Qwen3VLVisionBlock(nnx.Module):
  """Vision transformer block with LayerNorm, Attention, and GELU MLP."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    hs = self.config.hidden_size_for_vit
    inter_dim = self.config.intermediate_size_for_vit

    self.ln1 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=hs, epsilon=config.normalization_layer_epsilon, rngs=rngs)
    self.attn = Qwen3VLVisionAttention(config=config, rngs=rngs)

    self.mlp = DenseGeneral(
        in_features_shape=hs,
        out_features_shape=inter_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.mlp_out = DenseGeneral(
        in_features_shape=inter_dim,
        out_features_shape=hs,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      x: jax.Array,
      num_frames: int,
      height: int,
      width: int,
      attention_mask: Optional[jax.Array] = None,
      valid_grid: Optional[tuple[int, int, int]] = None,
  ) -> jax.Array:
    x = x + self.attn(
        self.ln1(x),
        num_frames=num_frames,
        height=height,
        width=width,
        attention_mask=attention_mask,
        valid_grid=valid_grid,
    )
    y = self.ln2(x)
    y = self.mlp(y)
    y = jax.nn.gelu(y)
    y = self.mlp_out(y)
    return x + y


class Qwen3VLVisionPatchMerger(nnx.Module):
  """Vision patch merger that spatially merges patches using an MLP."""

  def __init__(
      self,
      config: Config,
      use_postshuffle_norm: bool = False,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.use_postshuffle_norm = use_postshuffle_norm
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    spatial_merge_size = config.spatial_merge_size_for_vit
    base_hidden_size = config.hidden_size_for_vit
    out_hidden_size = config.out_hidden_size_for_vit

    self.hidden_size = base_hidden_size * (spatial_merge_size**2)

    ln_features = self.hidden_size if use_postshuffle_norm else base_hidden_size
    self.ln_q = nnx.LayerNorm(
        num_features=ln_features,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        rngs=rngs,
    )

    self.mlp_0 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.hidden_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )
    self.mlp_2 = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=out_hidden_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden: jax.Array) -> jax.Array:
    spatial_merge_size = self.config.spatial_merge_size_for_vit
    base_hidden_size = self.config.hidden_size_for_vit
    tokens_per_block = spatial_merge_size**2

    batch_size = hidden.shape[0]
    seq_len = hidden.shape[1]
    num_blocks = seq_len // tokens_per_block

    hidden = hidden.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    if self.use_postshuffle_norm:
      hidden = self.ln_q(hidden)
    else:
      hidden_unmerged = hidden.reshape(batch_size, seq_len, base_hidden_size)
      hidden_unmerged = self.ln_q(hidden_unmerged)
      hidden = hidden_unmerged.reshape(batch_size, num_blocks, tokens_per_block * base_hidden_size)

    hidden = self.mlp_0(hidden)
    hidden = jax.nn.gelu(hidden)
    hidden = self.mlp_2(hidden)
    return hidden


class Qwen3VLVisionEncoder(nnx.Module):
  """Vision encoder with patch embedding, positional interpolation, and blocks."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.patch_embed = Qwen3VLVisionPatchEmbed(config=config, rngs=rngs)

    num_pos = config.num_position_embeddings_for_vit
    hs = config.hidden_size_for_vit
    self.spatial_merge_size = config.spatial_merge_size_for_vit

    self.pos_embed_interpolate = Qwen3VLVisionPosEmbedInterpolate(
        num_position_embeddings=num_pos,
        hidden_size=hs,
        spatial_merge_size=self.spatial_merge_size,
        rngs=rngs,
    )

    self.depth = config.num_hidden_layers_for_vit
    for i in range(self.depth):
      setattr(self, f"blocks_{i}", Qwen3VLVisionBlock(config=config, rngs=rngs))

    self.deep_idx = tuple(config.deepstack_visual_indexes_for_vit)
    for i, _ in enumerate(self.deep_idx):
      setattr(self, f"merger_{i}", Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True, rngs=rngs))

  def __call__(
      self,
      hidden_states: jax.Array,
      video_mask: Optional[jax.Array] = None,
      video_grid_thw: Optional[Any] = None,
  ):
    batch_size, _, num_frames, height, width = hidden_states.shape
    num_frames = num_frames // self.config.temporal_patch_size_for_vit
    height = height // self.config.patch_size_for_vit
    width = width // self.config.patch_size_for_vit
    attention_mask = None
    if video_mask is not None:
      mask_patch_elements = (
          self.config.temporal_patch_size_for_vit * self.config.patch_size_for_vit * self.config.patch_size_for_vit
      )
      attention_mask = video_mask.reshape(batch_size, -1, mask_patch_elements).max(axis=-1).astype(jnp.int32)
    hidden_states = hidden_states.reshape(
        -1,
        self.config.num_channels_for_vit,
        self.config.temporal_patch_size_for_vit,
        self.config.patch_size_for_vit,
        self.config.patch_size_for_vit,
    )

    x, _ = self.patch_embed(hidden_states)
    x = x.reshape(batch_size, -1, self.config.hidden_size_for_vit)
    pos = self.pos_embed_interpolate(
        num_frames,
        height,
        width,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    x = x + pos
    valid_grid = video_grid_thw

    h_traj = []
    for i in range(self.depth):
      blk = getattr(self, f"blocks_{i}")
      x = blk(
          x,
          num_frames=num_frames,
          height=height,
          width=width,
          attention_mask=attention_mask,
          valid_grid=valid_grid,
      )
      h_traj.append(x)

    deep_feats = []
    for i, idx in enumerate(self.deep_idx):
      merger = getattr(self, f"merger_{i}")
      deep_feats.append(merger(h_traj[idx]))

    return x, deep_feats


class Qwen3VLVisionProjector(nnx.Module):
  """Projection layer converting vision encoder output to model embedding space."""

  def __init__(self, config: Config, *, rngs: Optional[nnx.Rngs] = None):
    self.config = config
    self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False, rngs=rngs)

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    return self.merger(hidden_states)


class Qwen3VLVisionTower(nnx.Module):
  """Container module holding Qwen3VLVisionEncoder_0 and Qwen3VLVisionProjector_0 to match checkpoint."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.Qwen3VLVisionEncoder_0 = Qwen3VLVisionEncoder(config=config, rngs=rngs)
    self.Qwen3VLVisionProjector_0 = Qwen3VLVisionProjector(config=config, rngs=rngs)

  def __call__(
      self,
      input_images: Optional[jax.Array] = None,
      input_masks: Optional[jax.Array] = None,
      video_grid_thw: Optional[Any] = None,
  ):
    enc_out, deep_feats = self.Qwen3VLVisionEncoder_0(
        input_images,
        video_mask=input_masks,
        video_grid_thw=video_grid_thw,
    )
    proj_out = self.Qwen3VLVisionProjector_0(enc_out)
    return proj_out, deep_feats


# ---------------------------------------------------------------------------
# Qwen3-VL Text Attention & Decoder
# ---------------------------------------------------------------------------


class Qwen3VLAttention(nnx.Module):
  """Full-sequence Qwen3 text attention with multi-axis projections and 3D MRoPE."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim

    self.query = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_query_heads, self.head_dim),
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.key = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.value = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.out = DenseGeneral(
        in_features_shape=(self.num_query_heads, self.head_dim),
        out_features_shape=config.emb_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.query_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kw)
    self.key_norm = nnx.RMSNorm(num_features=self.head_dim, **norm_kw)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
  ) -> jax.Array:
    s = inputs.shape[1]
    query = self.query(inputs)
    key = self.key(inputs)
    value = self.value(inputs)

    query = self.query_norm(query)
    key = self.key_norm(key)

    if decoder_positions is not None:
      mrope_section = getattr(self.config, "mrope_section", (24, 20, 20))
      rope_theta = getattr(self.config, "rope_max_timescale", 5000000.0)
      query = apply_mrope(query, decoder_positions, mrope_section=mrope_section, rope_theta=rope_theta)
      key = apply_mrope(key, decoder_positions, mrope_section=mrope_section, rope_theta=rope_theta)

    query = query * (self.head_dim**-0.5)

    # Group query heads without materializing repeated keys and values.
    groups = self.num_query_heads // self.num_kv_heads
    query = query.reshape(inputs.shape[0], s, self.num_kv_heads, groups, self.head_dim)
    score_dtype = jnp.float32 if self.config.float32_logits else query.dtype
    scores = jnp.einsum("bskgd,btkd->bkgst", query.astype(score_dtype), key.astype(score_dtype))
    causal_mask = jnp.tril(jnp.ones((s, s), dtype=bool))[None, :, :]
    if decoder_segment_ids is not None:
      causal_mask = causal_mask & (decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :])
    scores = jnp.where(causal_mask[:, None, None, :, :], scores, -1e10)
    weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
    attn_out = jnp.einsum("bkgst,btkd->bskgd", weights, value)
    attn_out = attn_out.reshape(inputs.shape[0], s, self.num_query_heads, self.head_dim)
    return self.out(attn_out)


class Qwen3VLDecoderLayer(nnx.Module):
  """Qwen3-VL Decoder Layer with Qwen3VLAttention and Qwen3MLP."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    norm_kw = {
        "epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "param_dtype": config.weight_dtype,
        "rngs": rngs,
    }
    self.pre_self_attention_layer_norm = nnx.RMSNorm(num_features=config.emb_dim, **norm_kw)
    self.post_self_attention_layer_norm = nnx.RMSNorm(num_features=config.emb_dim, **norm_kw)
    self.self_attention = Qwen3VLAttention(config, rngs=rngs)
    self.mlp = Qwen3MLP(config, rngs=rngs)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      **kwargs,
  ) -> jax.Array:
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out = self.self_attention(
        normed_attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
    )
    x = (inputs + attn_out).astype(self.config.dtype)

    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.config.dtype)
    return output


def deepstack_process(hidden_states: jax.Array, mask: jax.Array, visual_embeds: jax.Array) -> jax.Array:
  """Adds deepstack visual features to hidden states at visual token positions."""
  visual_token_idx = jnp.cumsum(mask, axis=1) - 1
  batch_idx = jnp.arange(hidden_states.shape[0])[:, jnp.newaxis]
  visual_embeds_scattered = visual_embeds[batch_idx, visual_token_idx, :]
  mask_expanded = mask[:, :, jnp.newaxis]
  return hidden_states + jnp.where(mask_expanded, visual_embeds_scattered, 0.0)


class Qwen3VLDecoder(Qwen3Decoder):
  """Qwen3-VL decoder stack specializing the dense Qwen3 layers."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(config, mesh, rngs=rngs, layer_cls=Qwen3VLDecoderLayer)

  def __call__(
      self,
      shared_embedding: Any,
      decoder_input_tokens: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      deepstack_visual_embeds: Optional[Sequence[jax.Array]] = None,
      multimodal_input: Optional[MultimodalInput] = None,
      deterministic: bool = False,
  ) -> tuple[jax.Array, jax.Array]:
    y = shared_embedding(decoder_input_tokens)
    visual_mask = None
    if multimodal_input is not None and multimodal_input.image_embeddings is not None:
      image_pad_token_id = 151655
      visual_mask = decoder_input_tokens == image_pad_token_id
      if multimodal_input.image_masks is not None:
        valid_count = multimodal_input.image_masks.sum(axis=1, keepdims=True)
        visual_mask = visual_mask & (jnp.cumsum(visual_mask, axis=1) <= valid_count)
      y = merge_multimodal_embeddings(
          text_embeddings=y,
          multimodal_embeddings=multimodal_input.image_embeddings,
          mask=visual_mask,
          feature_mask=multimodal_input.image_masks,
      )

    for lyr in range(self.num_layers):
      layer = getattr(self, f"layers_{lyr}")
      y = layer(
          y,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
      )
      if deepstack_visual_embeds is not None and lyr < len(deepstack_visual_embeds):
        feat = deepstack_visual_embeds[lyr]
        if feat is not None and multimodal_input is not None and multimodal_input.image_masks is not None:
          feat = _compact_visual_features(feat, multimodal_input.image_masks)
        if visual_mask is not None and feat is not None:
          y = deepstack_process(y, visual_mask, feat)

    logits = self.apply_output_head(shared_embedding, y, deterministic=deterministic)
    return logits, y


def _compact_visual_features(features: jax.Array, feature_mask: jax.Array) -> jax.Array:
  """Move valid features to the front, preserving their preprocessor order."""
  indices = jnp.argsort(~feature_mask.astype(jnp.bool_), axis=1, stable=True)
  return jnp.take_along_axis(features, indices[..., None], axis=1)


def merge_multimodal_embeddings(
    text_embeddings: jax.Array,
    multimodal_embeddings: jax.Array,
    mask: jax.Array,
    feature_mask: Optional[jax.Array] = None,
) -> jax.Array:
  """Merges visual tokens into text embeddings at mask positions."""
  batch_size, _, d_model = text_embeddings.shape
  flat_mm = multimodal_embeddings.reshape(batch_size, -1, d_model).astype(text_embeddings.dtype)
  if feature_mask is not None:
    flat_mm = _compact_visual_features(flat_mm, feature_mask)
    mask = mask & (jnp.cumsum(mask, axis=1) <= feature_mask.sum(axis=1, keepdims=True))

  def _merge_single(t_emb, m_emb, m_mask):
    cumsum_mask = jnp.cumsum(m_mask != 0)
    mask_bool = (m_mask != 0) & (cumsum_mask <= m_emb.shape[0])
    mask_expanded = mask_bool[:, jnp.newaxis]
    mm_idx = jnp.clip(cumsum_mask - 1, 0, m_emb.shape[0] - 1)
    mm_aligned = m_emb[mm_idx, :]
    return jnp.where(mask_expanded, mm_aligned, t_emb)

  return jax.vmap(_merge_single, in_axes=(0, 0, 0))(text_embeddings, flat_mm, mask)


class Qwen3VLModel(nnx.Module):
  """Top-level self-contained Qwen3-VL multimodal model in m3 format."""

  def __init__(
      self,
      config: Config,
      mesh: Optional[Mesh] = None,
      *,
      rngs: nnx.Rngs,
      **kwargs,
  ):
    self.config = config

    # 1. Vision Tower matching checkpoint parameter hierarchy
    self.vision_encoder = (
        Qwen3VLVisionTower(config=config, rngs=rngs) if getattr(config, "use_multimodal", False) else None
    )

    # 2. Text LLM Token Embedder & Decoder
    self.token_embedder = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.decoder = Qwen3VLDecoder(
        config=config,
        rngs=rngs,
    )

  def logits_from_hidden_states_for_vocab_tiling(
      self, hidden_states: jax.Array, deterministic: bool, model_mode: Optional[str] = None, **kwargs
  ) -> jax.Array:
    return self.decoder.apply_output_head(
        shared_embedding=self.token_embedder,
        y=hidden_states,
        deterministic=deterministic,
    )

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
      encoder_images: Optional[jax.Array] = None,
      encoder_image_masks: Optional[jax.Array] = None,
      encoder_video_grid_thw: Optional[jax.Array] = None,
      enable_dropout: bool = False,
      **kwargs,
  ) -> jax.Array:
    deterministic = kwargs.get("deterministic", not enable_dropout)

    # Process vision inputs if provided
    image_embeddings = None
    deepstack_feats = None

    if getattr(self.config, "use_multimodal", False) and encoder_images is not None and self.vision_encoder is not None:
      image_embeddings, deepstack_feats = self.vision_encoder(
          input_images=encoder_images,
          input_masks=encoder_image_masks,
          video_grid_thw=encoder_video_grid_thw,
      )

    if decoder_positions is None:
      decoder_positions = jnp.broadcast_to(
          jnp.arange(decoder_input_tokens.shape[1], dtype=jnp.int32)[None, :],
          decoder_input_tokens.shape,
      )

    # Keep encoded modality features together at the decoder fusion boundary.
    multimodal_input = None
    if image_embeddings is not None:
      feature_mask = None
      if encoder_image_masks is not None:
        patch_elements = self.config.temporal_patch_size_for_vit * self.config.patch_size_for_vit**2
        patch_mask = encoder_image_masks.reshape(image_embeddings.shape[0], -1, patch_elements).any(axis=-1)
        feature_mask = patch_mask.reshape(image_embeddings.shape[0], -1, self.config.spatial_merge_size_for_vit**2).any(
            axis=-1
        )
      multimodal_input = MultimodalInput(
          image_embeddings=image_embeddings,
          image_masks=feature_mask,
      )

    # Decoder forward pass
    logits, _ = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        deepstack_visual_embeds=deepstack_feats,
        multimodal_input=multimodal_input,
    )
    return logits


def create_qwen3_vl_model(
    config: Config,
    mesh: Optional[Mesh] = None,
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Qwen3VLModel:
  """Factory function creating a Qwen3VLModel instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3VLModel(config, mesh, rngs=rngs, **kwargs)
