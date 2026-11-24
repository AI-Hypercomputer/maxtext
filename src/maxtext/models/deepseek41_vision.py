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

"""DeepSeek-V4.1 vision encoder (ViT with per-image 2D RoPE) and 3x3 aligner/span projector."""

import functools
import math

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common.common_types import Config, DType
from maxtext.layers import linears

_VISION_RMSNORM_EPS = 1e-6


class _VisionRMSNorm(nnx.Module):
  """Vision RMSNorm matching release epsilon (1e-6) and multiply order."""

  def __init__(
      self,
      dim: int,
      *,
      weight_dtype: DType,
      eps: float = _VISION_RMSNORM_EPS,
  ):
    self.eps = eps
    self.scale = nnx.Param(jnp.ones((dim,), dtype=weight_dtype), out_sharding=(None,))

  def __call__(self, x: jax.Array) -> jax.Array:
    orig_dtype = x.dtype
    work_dtype = jnp.promote_types(orig_dtype, jnp.float32)
    x_work = x.astype(work_dtype)
    normed = x_work * jax.lax.rsqrt(jnp.mean(jnp.square(x_work), axis=-1, keepdims=True) + self.eps)
    return (self.scale[...].astype(work_dtype) * normed).astype(orig_dtype)


def vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    dtype: DType = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
  """Computes 2D `[row, col]` rotary cos/sin tables of shape `(n_h * n_w, 1, dim)`."""
  if n_h <= 0 or n_w <= 0:
    raise ValueError(f"Vision RoPE grid dimensions must be positive, got {(n_h, n_w)}.")
  if dim <= 0 or dim % 2 != 0:
    raise ValueError(f"Vision RoPE dim must be positive and even, got {dim}.")
  work_dtype = jnp.promote_types(dtype, jnp.float32)
  inv_freq = jnp.asarray([1.0 / (float(theta) ** (i / float(dim))) for i in range(0, dim, 2)], dtype=work_dtype)
  hpos = jnp.broadcast_to(jnp.arange(n_h)[:, None], (n_h, n_w))
  wpos = jnp.broadcast_to(jnp.arange(n_w)[None, :], (n_h, n_w))
  freqs = jnp.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(work_dtype) * inv_freq
  freqs = freqs.reshape(n_h * n_w, dim)
  return jnp.cos(freqs)[:, None, :], jnp.sin(freqs)[:, None, :]


def vision_cos_sin_batched(
    image_grids: jax.Array,
    num_patches: int,
    dim: int,
    theta: float,
    dtype: DType = jnp.float32,
    image_masks: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes per-image 2D RoPE `(cos, sin)` `[M, P, 1, dim]` and validity mask `[M, P]`."""
  if num_patches <= 0:
    raise ValueError(f"num_patches must be positive, got {num_patches}.")
  if dim <= 0 or dim % 2 != 0:
    raise ValueError(f"Vision RoPE dim must be positive and even, got {dim}.")
  grids_flat = image_grids.reshape(-1, 2).astype(jnp.int32)
  n_h = grids_flat[:, 0]
  n_w = grids_flat[:, 1]
  slot_valid = (n_h > 0) & (n_w > 0)
  if image_masks is not None:
    slot_valid = slot_valid & (image_masks.reshape(-1) != 0)
  valid_counts = jnp.where(slot_valid, n_h * n_w, 0)
  patch_idx = jnp.arange(num_patches, dtype=jnp.int32)[None, :]
  patch_mask = patch_idx < valid_counts[:, None]
  safe_w = jnp.maximum(n_w, 1)[:, None]
  hpos = jnp.where(patch_mask, patch_idx // safe_w, 0)
  wpos = jnp.where(patch_mask, patch_idx % safe_w, 0)
  work_dtype = jnp.promote_types(dtype, jnp.float32)
  inv_freq = jnp.asarray([1.0 / (float(theta) ** (i / float(dim))) for i in range(0, dim, 2)], dtype=work_dtype)
  freqs = jnp.stack([hpos, wpos], axis=-1)[..., None].astype(work_dtype) * inv_freq
  freqs = freqs.reshape(grids_flat.shape[0], num_patches, dim)
  return jnp.cos(freqs)[:, :, None, :], jnp.sin(freqs)[:, :, None, :], patch_mask


def max_aligned_tokens_for_buffer(num_patches: int, downsample_ratio: int, max_n_token: int) -> int:
  """Returns static upper bound `K_align_max` on `ceil(H/r) * ceil(W/r)` for buffer `num_patches`."""
  if num_patches <= 0 or downsample_ratio <= 0 or max_n_token < 4:
    raise ValueError(
        f"Invalid aligner buffer parameters: num_patches={num_patches}, "
        f"downsample_ratio={downsample_ratio}, max_n_token={max_n_token}."
    )
  return min(max_n_token - 3, math.ceil(num_patches / downsample_ratio))


def max_span_tokens_for_buffer(num_patches: int, downsample_ratio: int, max_n_token: int) -> int:
  """Returns static upper bound `T_span_max` on `1 + n_llm_h * (n_llm_w + 1) + 1`."""
  if num_patches <= 0 or downsample_ratio <= 0 or max_n_token < 4:
    raise ValueError(
        f"Invalid span buffer parameters: num_patches={num_patches}, "
        f"downsample_ratio={downsample_ratio}, max_n_token={max_n_token}."
    )
  return min(max_n_token, 2 * math.ceil(num_patches / downsample_ratio) + 2)


def compute_image_span_mask(
    image_grids: jax.Array,
    *,
    span_len_max: int,
    downsample_ratio: int,
    image_masks: jax.Array | None = None,
) -> jax.Array:
  """Computes `int32[B, N, span_len_max]` token validity mask for dynamic per-image spans."""
  grids = image_grids.astype(jnp.int32)
  n_h = grids[..., 0]
  n_w = grids[..., 1]
  slot_valid = (n_h > 0) & (n_w > 0)
  if image_masks is not None:
    slot_valid = slot_valid & (image_masks.reshape(grids.shape[:-1]) != 0)
  r = int(downsample_ratio)
  n_llm_h = jnp.where(slot_valid, (n_h + r - 1) // r, 0)
  n_llm_w = jnp.where(slot_valid, (n_w + r - 1) // r, 0)
  span_len = jnp.where(slot_valid, 1 + n_llm_h * (n_llm_w + 1) + 1, 0)
  tok_idx = jnp.arange(span_len_max, dtype=jnp.int32)
  return (tok_idx < span_len[..., None]).astype(jnp.int32)


def apply_vision_rotary(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
  """Applies half-split 2D rotary embedding to `[..., seq, heads, head_dim]`."""
  orig_dtype = x.dtype
  work_dtype = jnp.promote_types(orig_dtype, jnp.float32)
  x1, x2 = jnp.split(x.astype(work_dtype), 2, axis=-1)
  cos_w = cos.astype(work_dtype)
  sin_w = sin.astype(work_dtype)
  rotated = jnp.concatenate([x1 * cos_w - x2 * sin_w, x2 * cos_w + x1 * sin_w], axis=-1)
  return rotated.astype(orig_dtype)


class DeepSeek41VisionAttention(nnx.Module):
  """Bidirectional multi-head self-attention with 2D RoPE for DeepSeek-V4.1 ViT."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.dim = config.hidden_size_for_vit
    self.n_heads = config.num_attention_heads_for_vit
    if self.dim % self.n_heads != 0:
      raise ValueError(
          f"hidden_size_for_vit ({self.dim}) must be divisible by num_attention_heads_for_vit ({self.n_heads})."
      )
    self.head_dim = self.dim // self.n_heads
    self.scale = self.head_dim**-0.5
    self.wqkv = linears.DenseGeneral(
        in_features_shape=self.dim,
        out_features_shape=3 * self.dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=self.dim,
        out_features_shape=self.dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("mlp", "embed"),
        rngs=rngs,
    )

  def __call__(
      self,
      x: jax.Array,
      cos: jax.Array,
      sin: jax.Array,
      patch_mask: jax.Array | None = None,
  ) -> jax.Array:
    qkv = self.wqkv(x).reshape(*x.shape[:-1], 3, self.n_heads, self.head_dim)
    q = apply_vision_rotary(qkv[..., 0, :, :], cos, sin)
    k = apply_vision_rotary(qkv[..., 1, :, :], cos, sin)
    v = qkv[..., 2, :, :]
    qk_dtype = jnp.promote_types(q.dtype, jnp.float32) if getattr(self.config, "float32_qk_product", False) else q.dtype
    precision = jax.lax.Precision(self.config.matmul_precision)
    logits = jnp.einsum(
        "...qhd,...khd->...hqk", q.astype(qk_dtype), k.astype(qk_dtype), precision=precision
    ) * jnp.asarray(self.scale, dtype=qk_dtype)
    softmax_dtype = jnp.promote_types(logits.dtype, jnp.float32)
    if patch_mask is not None:
      valid_q = patch_mask[:, None, :, None]
      valid_k = patch_mask[:, None, None, :]
      safe_logits = jnp.where(valid_q & valid_k, logits.astype(softmax_dtype), 0.0)
      masked_logits = jnp.where(valid_q & ~valid_k, jnp.asarray(-1e30, dtype=softmax_dtype), safe_logits)
      probs = jnp.where(valid_q & valid_k, jax.nn.softmax(masked_logits, axis=-1), 0.0).astype(v.dtype)
    else:
      probs = jax.nn.softmax(logits.astype(softmax_dtype), axis=-1).astype(v.dtype)
    out = jnp.einsum("...hqk,...khd->...qhd", probs, v, precision=precision).astype(q.dtype)
    if patch_mask is not None:
      out = jnp.where(patch_mask[:, :, None, None], out, 0.0)
    proj_out = self.wo(out.reshape(*x.shape[:-1], self.dim))
    if patch_mask is not None:
      proj_out = jnp.where(patch_mask[:, :, None], proj_out, 0.0)
    return proj_out


class DeepSeek41VisionMLP(nnx.Module):
  """SwiGLU MLP with fused 2D `[gate | up]` projection for DeepSeek-V4.1 ViT."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    dim = config.hidden_size_for_vit
    inter_dim = config.intermediate_size_for_vit
    self.w1 = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=2 * inter_dim,
        use_bias=False,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=rngs,
    )
    self.w2 = linears.DenseGeneral(
        in_features_shape=inter_dim,
        out_features_shape=dim,
        use_bias=False,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("mlp", "embed"),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    gate, up = jnp.split(self.w1(x), 2, axis=-1)
    return self.w2(jax.nn.silu(gate) * up)


class DeepSeek41VisionBlock(nnx.Module):
  """One pre-norm transformer block of the DeepSeek-V4.1 ViT."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    dim = config.hidden_size_for_vit
    self.norm1 = _VisionRMSNorm(dim, weight_dtype=config.weight_dtype)
    self.attn = DeepSeek41VisionAttention(config=config, rngs=rngs)
    self.norm2 = _VisionRMSNorm(dim, weight_dtype=config.weight_dtype)
    self.mlp = DeepSeek41VisionMLP(config=config, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      cos: jax.Array,
      sin: jax.Array,
      patch_mask: jax.Array | None = None,
  ) -> jax.Array:
    x = x + self.attn(self.norm1(x), cos, sin, patch_mask=patch_mask)
    mlp_out = self.mlp(self.norm2(x))
    if patch_mask is not None:
      mlp_out = jnp.where(patch_mask[:, :, None], mlp_out, 0.0)
    return x + mlp_out


def _run_vision_block_remat(
    block: DeepSeek41VisionBlock,
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    patch_mask: jax.Array | None = None,
) -> jax.Array:
  """Runs one ViT transformer block under activation checkpointing."""
  graphdef, params, other = nnx.split(block, nnx.Param, nnx.Not(nnx.Param))

  @functools.partial(jax.checkpoint, prevent_cse=True)
  def _forward(
      params_state: nnx.State,
      x_in: jax.Array,
      cos_in: jax.Array,
      sin_in: jax.Array,
      mask_in: jax.Array | None,
  ) -> jax.Array:
    return nnx.merge(graphdef, params_state, other, copy=True)(x_in, cos_in, sin_in, patch_mask=mask_in)

  return _forward(params, x, cos, sin, patch_mask)


class DeepSeek41VisionEncoder(nnx.Module):
  """DeepSeek-V4.1 vision tower encoding `[B, N, P, 3*p*p]` patches to `[B, N, P, C]`."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    dim = config.hidden_size_for_vit
    n_heads = config.num_attention_heads_for_vit
    if dim % n_heads != 0:
      raise ValueError(f"hidden_size_for_vit ({dim}) must be divisible by num_attention_heads_for_vit ({n_heads}).")
    head_dim = dim // n_heads
    if head_dim % 4 != 0:
      raise ValueError(f"ViT head_dim ({head_dim}) must be divisible by 4 for 2D RoPE.")
    self.rope_dim = head_dim // 2
    self.rope_theta = float(config.rope_theta_for_vit)
    self.patch_dim = config.num_channels_for_vit * (config.patch_size_for_vit**2)
    # patch_dim (3 * 14 * 14 = 588) is not divisible by FSDP >= 8; shard hidden_size_for_vit along 'embed'.
    self.patch_embed = linears.DenseGeneral(
        in_features_shape=self.patch_dim,
        out_features_shape=dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=(None, "embed"),
        rngs=rngs,
    )
    self.num_layers = config.num_hidden_layers_for_vit
    for i in range(self.num_layers):
      setattr(self, f"blocks_{i}", DeepSeek41VisionBlock(config=config, rngs=rngs))
    self.norm = _VisionRMSNorm(dim, weight_dtype=config.weight_dtype)

  def embed_patches(self, patches: jax.Array) -> jax.Array:
    """Casts flattened patches `[..., P, 3*p*p]` to `dtype_mm` and applies `patch_embed`."""
    patches_mm = patches.astype(self.config.dtype_mm)
    if not jnp.issubdtype(patches_mm.dtype, jnp.floating):
      raise TypeError(
          f"DeepSeek41VisionEncoder.embed_patches requires floating-point patches after cast, got {patches_mm.dtype}."
      )
    return self.patch_embed(patches_mm)

  def __call__(
      self,
      input_images: jax.Array,
      deterministic: bool = False,
      image_grids: jax.Array | None = None,
      image_masks: jax.Array | None = None,
  ) -> jax.Array:
    del deterministic
    if image_grids is None:
      n_vit_h, n_vit_w = self.config.ds41_vision_grid
      expected_patches = n_vit_h * n_vit_w
      if input_images.ndim != 4 or input_images.shape[2:] != (expected_patches, self.patch_dim):
        raise ValueError(
            f"Expected input_images shape [batch, num_images, {expected_patches}, {self.patch_dim}] "
            f"for ds41_vision_grid={self.config.ds41_vision_grid!r}, got {input_images.shape}."
        )
      batch_size, num_images = input_images.shape[:2]
      x = self.embed_patches(input_images.reshape(batch_size * num_images, expected_patches, self.patch_dim))
      cos, sin = vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta, dtype=x.dtype)
      patch_mask = None
      num_patches = expected_patches
    else:
      if input_images.ndim != 4 or input_images.shape[3] != self.patch_dim or input_images.shape[2] <= 0:
        raise ValueError(
            f"Expected input_images shape [batch, num_images, P_max, {self.patch_dim}], got {input_images.shape}."
        )
      batch_size, num_images, num_patches, _ = input_images.shape
      if image_grids.shape != (batch_size, num_images, 2):
        raise ValueError(f"Expected image_grids shape {(batch_size, num_images, 2)}, got {image_grids.shape}.")
      cos, sin, patch_mask = vision_cos_sin_batched(
          image_grids,
          num_patches,
          self.rope_dim,
          self.rope_theta,
          dtype=input_images.dtype,
          image_masks=image_masks,
      )
      flat_patches = input_images.reshape(batch_size * num_images, num_patches, self.patch_dim)
      flat_patches = jnp.where(patch_mask[:, :, None], flat_patches, 0.0)
      x = jnp.where(patch_mask[:, :, None], self.embed_patches(flat_patches), 0.0)
      if jnp.promote_types(x.dtype, jnp.float32) != cos.dtype:
        cos, sin, _ = vision_cos_sin_batched(
            image_grids,
            num_patches,
            self.rope_dim,
            self.rope_theta,
            dtype=x.dtype,
            image_masks=image_masks,
        )

    use_remat = (
        not self.config.freeze_vision_encoder_params
        and bool(self.config.remat_policy)
        and self.config.remat_policy != "none"
    )
    for i in range(self.num_layers):
      block = getattr(self, f"blocks_{i}")
      if use_remat:
        x = _run_vision_block_remat(block, x, cos, sin, patch_mask=patch_mask)
      else:
        x = block(x, cos, sin, patch_mask=patch_mask)
    x = self.norm(x)
    if patch_mask is not None:
      x = jnp.where(patch_mask[:, :, None], x, 0.0)
    return x.reshape(batch_size, num_images, num_patches, self.config.hidden_size_for_vit)


class DeepSeek41VisionProjector(nnx.Module):
  """3x3 spatial aligner and `[START] + (row + NEWLINE) * H + [END]` span assembler."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    r = config.ds41_vision_downsample_ratio
    vit_dim = config.hidden_size_for_vit
    emb_dim = config.emb_dim
    self.w1 = linears.DenseGeneral(
        in_features_shape=vit_dim * (r**2),
        out_features_shape=emb_dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=rngs,
    )
    self.w2 = linears.DenseGeneral(
        in_features_shape=emb_dim,
        out_features_shape=emb_dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("mlp", "embed"),
        rngs=rngs,
    )
    delim_init = jax.nn.initializers.normal(stddev=emb_dim**-0.5)
    self.image_start = nnx.Param(delim_init(rngs.params(), (emb_dim,), config.weight_dtype), out_sharding=(None,))
    self.image_end = nnx.Param(delim_init(rngs.params(), (emb_dim,), config.weight_dtype), out_sharding=(None,))
    self.image_newline = nnx.Param(delim_init(rngs.params(), (emb_dim,), config.weight_dtype), out_sharding=(None,))

  def align(self, vit_features: jax.Array, n_vit_h: int, n_vit_w: int) -> jax.Array:
    """Zero-pads `[..., P, C]` bottom/right to multiples of `r`, 3x3-unfolds, and runs MLP."""
    *leading, num_patches, vit_dim = vit_features.shape
    if num_patches != n_vit_h * n_vit_w or vit_dim != self.config.hidden_size_for_vit:
      raise ValueError(
          f"Expected vit_features trailing shape ({n_vit_h * n_vit_w}, {self.config.hidden_size_for_vit}), "
          f"got {(num_patches, vit_dim)}."
      )
    r = self.config.ds41_vision_downsample_ratio
    pad_h = (-n_vit_h) % r
    pad_w = (-n_vit_w) % r
    grid = vit_features.reshape(*leading, n_vit_h, n_vit_w, vit_dim)
    if pad_h > 0 or pad_w > 0:
      grid = jnp.pad(grid, [(0, 0)] * len(leading) + [(0, pad_h), (0, pad_w), (0, 0)])
    n_llm_h = (n_vit_h + pad_h) // r
    n_llm_w = (n_vit_w + pad_w) // r
    lead_axes = tuple(range(len(leading)))
    base = len(leading)
    unfolded = jnp.transpose(
        grid.reshape(*leading, n_llm_h, r, n_llm_w, r, vit_dim),
        lead_axes + (base, base + 2, base + 4, base + 1, base + 3),
    )
    unfolded = unfolded.reshape(*leading, n_llm_h * n_llm_w, vit_dim * r * r)
    hidden = self.w1(unfolded)
    work_dtype = jnp.promote_types(hidden.dtype, jnp.float32)
    hidden = jax.nn.gelu(hidden.astype(work_dtype), approximate=False).astype(hidden.dtype)
    return self.w2(hidden)

  def align_dynamic(
      self,
      vit_features: jax.Array,
      image_grids: jax.Array,
      image_masks: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Zero-pads per-image `(n_vit_h, n_vit_w)` grids to multiples of `r`, 3x3-unfolds, and runs MLP."""
    batch_size, num_images, num_patches, vit_dim = vit_features.shape
    if vit_dim != self.config.hidden_size_for_vit:
      raise ValueError(f"Expected vit_features last dim {self.config.hidden_size_for_vit}, got {vit_dim}.")
    if image_grids.shape != (batch_size, num_images, 2):
      raise ValueError(f"Expected image_grids shape {(batch_size, num_images, 2)}, got {image_grids.shape}.")
    r = int(self.config.ds41_vision_downsample_ratio)
    k_align_max = max_aligned_tokens_for_buffer(num_patches, r, int(self.config.ds41_vision_max_n_token))
    num_slots = batch_size * num_images
    grids_flat = image_grids.reshape(num_slots, 2).astype(jnp.int32)
    n_h = grids_flat[:, 0]
    n_w = grids_flat[:, 1]
    slot_valid = (n_h > 0) & (n_w > 0)
    if image_masks is not None:
      slot_valid = slot_valid & (image_masks.reshape(num_slots) != 0)
    n_llm_h = jnp.where(slot_valid, (n_h + r - 1) // r, 0)
    n_llm_w = jnp.where(slot_valid, (n_w + r - 1) // r, 0)
    num_aligned = n_llm_h * n_llm_w
    safe_llm_w = jnp.maximum(n_llm_w, 1)[:, None]
    k_idx = jnp.arange(k_align_max, dtype=jnp.int32)[None, :]
    k_valid = k_idx < num_aligned[:, None]
    ih = k_idx // safe_llm_w
    iw = k_idx % safe_llm_w
    dh = jnp.arange(r, dtype=jnp.int32)[None, None, :, None]
    dw = jnp.arange(r, dtype=jnp.int32)[None, None, None, :]
    patch_r = ih[:, :, None, None] * r + dh
    patch_c = iw[:, :, None, None] * r + dw
    subpatch_valid = (
        k_valid[:, :, None, None] & (patch_r < n_h[:, None, None, None]) & (patch_c < n_w[:, None, None, None])
    )
    flat_patch_idx = patch_r * n_w[:, None, None, None] + patch_c
    safe_patch_idx = jnp.where(subpatch_valid, jnp.clip(flat_patch_idx, 0, num_patches - 1), 0)
    feats_flat = vit_features.reshape(num_slots, num_patches, vit_dim)
    gathered = jnp.take_along_axis(
        feats_flat, safe_patch_idx.reshape(num_slots, k_align_max * r * r)[..., None], axis=1
    ).reshape(num_slots, k_align_max, r, r, vit_dim)
    gathered = jnp.where(subpatch_valid[..., None], gathered, 0.0)
    unfolded = jnp.transpose(gathered, (0, 1, 4, 2, 3)).reshape(batch_size, num_images, k_align_max, vit_dim * r * r)
    hidden = self.w1(unfolded)
    work_dtype = jnp.promote_types(hidden.dtype, jnp.float32)
    hidden = jax.nn.gelu(hidden.astype(work_dtype), approximate=False).astype(hidden.dtype)
    aligned = jnp.where(k_valid.reshape(batch_size, num_images, k_align_max, 1), self.w2(hidden), 0.0)
    return (
        aligned,
        n_llm_h.reshape(batch_size, num_images),
        n_llm_w.reshape(batch_size, num_images),
        slot_valid.reshape(batch_size, num_images),
    )

  def __call__(
      self,
      vit_features: jax.Array,
      image_grids: jax.Array | None = None,
      image_masks: jax.Array | None = None,
      *,
      return_span_mask: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    if vit_features.ndim != 4:
      raise ValueError(f"Expected vit_features of rank 4 [B, N, P, C], got shape {vit_features.shape}.")
    if image_grids is None:
      n_vit_h, n_vit_w = self.config.ds41_vision_grid
      aligned = self.align(vit_features, n_vit_h, n_vit_w)
      r = self.config.ds41_vision_downsample_ratio
      n_llm_h = (n_vit_h + (-n_vit_h) % r) // r
      n_llm_w = (n_vit_w + (-n_vit_w) % r) // r
      batch_size, num_images, _, emb_dim = aligned.shape
      rows = aligned.reshape(batch_size, num_images, n_llm_h, n_llm_w, emb_dim)
      out_dtype = aligned.dtype
      start_tok = jnp.broadcast_to(self.image_start[...].astype(out_dtype), (batch_size, num_images, 1, emb_dim))
      newline_tok = jnp.broadcast_to(
          self.image_newline[...].astype(out_dtype), (batch_size, num_images, n_llm_h, 1, emb_dim)
      )
      end_tok = jnp.broadcast_to(self.image_end[...].astype(out_dtype), (batch_size, num_images, 1, emb_dim))
      rows_with_newline = jnp.concatenate([rows, newline_tok], axis=3).reshape(
          batch_size, num_images, n_llm_h * (n_llm_w + 1), emb_dim
      )
      spans = jnp.concatenate([start_tok, rows_with_newline, end_tok], axis=2)
      if return_span_mask:
        if image_masks is None:
          span_mask = jnp.ones(spans.shape[:3], dtype=jnp.bool_)
        else:
          span_mask = jnp.broadcast_to((image_masks.reshape(batch_size, num_images) != 0)[..., None], spans.shape[:3])
        return spans, span_mask
      return spans

    aligned, n_llm_h, n_llm_w, slot_valid = self.align_dynamic(vit_features, image_grids, image_masks=image_masks)
    batch_size, num_images, k_align_max, emb_dim = aligned.shape
    t_span_max = max_span_tokens_for_buffer(
        vit_features.shape[2],
        int(self.config.ds41_vision_downsample_ratio),
        int(self.config.ds41_vision_max_n_token),
    )
    span_len = jnp.where(slot_valid, 1 + n_llm_h * (n_llm_w + 1) + 1, 0)
    tok_idx = jnp.arange(t_span_max, dtype=jnp.int32)[None, None, :]
    span_valid = tok_idx < span_len[:, :, None]
    is_start = span_valid & (tok_idx == 0)
    is_end = span_valid & (tok_idx == (span_len[:, :, None] - 1))
    is_body = span_valid & (tok_idx > 0) & (tok_idx < (span_len[:, :, None] - 1))
    body_idx = jnp.maximum(tok_idx - 1, 0)
    row_stride = n_llm_w[:, :, None] + 1
    body_row = body_idx // row_stride
    body_col = body_idx % row_stride
    is_newline = is_body & (body_col == n_llm_w[:, :, None])
    is_image_tok = is_body & (body_col < n_llm_w[:, :, None])
    align_idx = body_row * n_llm_w[:, :, None] + body_col
    safe_align_idx = jnp.where(is_image_tok, jnp.clip(align_idx, 0, k_align_max - 1), 0)
    img_tok = jnp.take_along_axis(aligned, safe_align_idx[..., None], axis=2)
    out_dtype = aligned.dtype
    start_vec = self.image_start[...].astype(out_dtype)
    newline_vec = self.image_newline[...].astype(out_dtype)
    end_vec = self.image_end[...].astype(out_dtype)
    spans = jnp.where(is_image_tok[..., None], img_tok, jnp.zeros_like(img_tok))
    spans = jnp.where(is_start[..., None], start_vec, spans)
    spans = jnp.where(is_newline[..., None], newline_vec, spans)
    spans = jnp.where(is_end[..., None], end_vec, spans)
    if return_span_mask:
      return spans, span_valid
    return spans
