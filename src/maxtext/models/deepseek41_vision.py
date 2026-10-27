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

"""DeepSeek-V4.1 vision encoder (ViT with 2D RoPE) and 3x3 aligner/span projector."""

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
  inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=work_dtype) / dim))
  hpos = jnp.broadcast_to(jnp.arange(n_h)[:, None], (n_h, n_w))
  wpos = jnp.broadcast_to(jnp.arange(n_w)[None, :], (n_h, n_w))
  freqs = jnp.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(work_dtype) * inv_freq
  freqs = freqs.reshape(n_h * n_w, dim)
  return jnp.cos(freqs)[:, None, :], jnp.sin(freqs)[:, None, :]


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
        kernel_axes=(None, None),
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=self.dim,
        out_features_shape=self.dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=(None, None),
        rngs=rngs,
    )

  def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    qkv = self.wqkv(x).reshape(*x.shape[:-1], 3, self.n_heads, self.head_dim)
    q = apply_vision_rotary(qkv[..., 0, :, :], cos, sin)
    k = apply_vision_rotary(qkv[..., 1, :, :], cos, sin)
    v = qkv[..., 2, :, :]
    work_dtype = jnp.promote_types(q.dtype, jnp.float32)
    precision = jax.lax.Precision(self.config.matmul_precision)
    logits = (
        jnp.einsum("...qhd,...khd->...hqk", q.astype(work_dtype), k.astype(work_dtype), precision=precision) * self.scale
    )
    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("...hqk,...khd->...qhd", probs, v.astype(work_dtype), precision=precision).astype(q.dtype)
    return self.wo(out.reshape(*x.shape[:-1], self.dim))


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
        kernel_axes=(None, None),
        rngs=rngs,
    )
    self.w2 = linears.DenseGeneral(
        in_features_shape=inter_dim,
        out_features_shape=dim,
        use_bias=False,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=(None, None),
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

  def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    x = x + self.attn(self.norm1(x), cos, sin)
    return x + self.mlp(self.norm2(x))


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
    self.patch_embed = linears.DenseGeneral(
        in_features_shape=self.patch_dim,
        out_features_shape=dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=(None, None),
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

  def __call__(self, input_images: jax.Array, deterministic: bool = False) -> jax.Array:
    del deterministic
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
    for i in range(self.num_layers):
      x = getattr(self, f"blocks_{i}")(x, cos, sin)
    x = self.norm(x)
    return x.reshape(batch_size, num_images, expected_patches, self.config.hidden_size_for_vit)


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
        kernel_axes=(None, None),
        rngs=rngs,
    )
    self.w2 = linears.DenseGeneral(
        in_features_shape=emb_dim,
        out_features_shape=emb_dim,
        use_bias=True,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=(None, None),
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

  def __call__(self, vit_features: jax.Array) -> jax.Array:
    if vit_features.ndim != 4:
      raise ValueError(f"Expected vit_features of rank 4 [B, N, P, C], got shape {vit_features.shape}.")
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
    return jnp.concatenate([start_tok, rows_with_newline, end_tok], axis=2)
