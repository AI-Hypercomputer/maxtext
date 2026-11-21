# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ming-Image diffusion transformer (a Z-Image / S3-DiT single-stream DiT) in pure Flax NNX.

Port of `diffusion/transformer.py::DiffusionTransformer` of Ming-Image-Ant. The reference takes per-sample
variable-length lists; this port takes static, padded batches plus keep masks and reproduces the reference exactly
for the valid tokens:

* Image tokens: the latent is patchified ("(h w) (ph pw c)") and embedded; 3D RoPE position ids are
  `(cap_padded_len + 1, h, w)`, where `cap_padded_len` is the sample's total caption length rounded up to a multiple
  of `SEQ_MULTI_OF` (32), so the image position depends on the caption length.
* Caption tokens: `[cap_embedder(cap_feats[:L1]), cap_feats_2[:L2]]` (`cap_feats_2` is already `dim` wide and is not
  embedded); position ids `(1 + i, 0, 0)` for the i-th valid caption token.
* The reference's alignment padding (to multiples of 32) is zero-filled and masked out of attention
  (`alignment_padding_mode="zero_masked"`), so it never affects valid tokens: here padding is simply masked out.
  Attention masks act on keys only (as in the reference); outputs of padded queries are discarded.
* Layer order: `noise_refiner` (image only, modulated) -> `context_refiner` (caption only, unmodulated) ->
  `layers` (joint [image, caption] sequence, modulated) -> `all_final_layer` on the image tokens.
"""

# NNX modules override `__call__` with their own signatures; constructors mirror the HF / Wan keyword sets.
# pylint: disable=arguments-differ,too-many-positional-arguments

import functools
import math
from typing import Any, Callable, Optional, Sequence

from flax import nnx
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from maxtext.common import common_types
from maxtext.m3.core.sharding import Sharding
from maxtext.m3.models.wan.transformer_wan import AttentionOp
from maxtext.m3.models.wan.utils_wan import BlockSizes, constrain, param_metadata

# Logical activation axes (looked up in the Wan `logical_axis_rules` table).
BATCH = common_types.BATCH  # activation_batch
LENGTH = common_types.LENGTH  # activation_length
KV_LENGTH = common_types.KV_LENGTH  # activation_kv_length
HEAD = common_types.HEAD  # activation_heads
D_KV = common_types.D_KV  # activation_kv
EMBED = common_types.EMBED  # activation_embed

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
FREQUENCY_EMBEDDING_SIZE = 256
T_EMBEDDER_MID_SIZE = 1024
ZERO_MASKED_PADDING = "zero_masked"
LEARNED_PADDING = "learned"

_LOW_PRECISION = (jnp.bfloat16, jnp.float16)


def _linear(
    rngs: nnx.Rngs,
    in_features: int,
    out_features: int,
    use_bias: bool,
    sharding: Sharding | None,
    name: str,
    axes: tuple,
    dtype: Any,
    weights_dtype: Any,
    precision: Any,
) -> nnx.Linear:
  """`nnx.Linear` with the init / sharding-metadata conventions of this file (`axes` = logical kernel axes)."""
  return nnx.Linear(
      in_features,
      out_features,
      use_bias=use_bias,
      rngs=rngs,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.initializers.lecun_normal(),
      kernel_metadata=param_metadata(sharding, f"{name}_kernel", axes),
      bias_init=nnx.initializers.zeros,
      bias_metadata=param_metadata(sharding, f"{name}_bias", (axes[1],)),
  )


class MingRMSNorm(nnx.Module):
  """diffusers `RMSNorm`: fp32 statistics; low-precision weights cast the normed input before scaling."""

  def __init__(self, dim: int, eps: float, sharding: Sharding | None, name: str, axes: tuple, weights_dtype: Any):
    self.eps = eps
    self.scale = nnx.Param(jnp.ones((dim,), weights_dtype), **param_metadata(sharding, f"{name}_scale", axes))

  def __call__(self, x: Array) -> Array:
    scale = self.scale[...]
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
    if scale.dtype in _LOW_PRECISION:
      return normed.astype(scale.dtype) * scale
    return (normed * scale).astype(x.dtype)


def timestep_embedding(t: Array, dim: int = FREQUENCY_EMBEDDING_SIZE, max_period: float = 10000.0) -> Array:
  """Sinusoidal timestep embedding [B] -> [B, dim] (cos first), in float32."""
  half = dim // 2
  freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
  args = t.astype(jnp.float32)[:, None] * freqs[None]
  emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  if dim % 2:
    emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
  return emb


class TimestepEmbedder(nnx.Module):
  """`t_embedder`: sinusoidal embedding -> Linear -> SiLU -> Linear (HF `mlp.0` / `mlp.2`)."""

  def __init__(self, rngs, out_size, mid_size, sharding, weights_dtype, precision):
    # dtype=None: computes in the param dtype, as the reference casts the fp32 embedding to the weight dtype.
    kw = {"sharding": sharding, "dtype": None, "weights_dtype": weights_dtype, "precision": precision}
    self.mlp_0 = _linear(rngs, FREQUENCY_EMBEDDING_SIZE, mid_size, True, name="t_mlp_0", axes=(None, None), **kw)
    self.mlp_2 = _linear(rngs, mid_size, out_size, True, name="t_mlp_2", axes=(None, None), **kw)

  def __call__(self, t: Array) -> Array:
    t_freq = timestep_embedding(t).astype(self.mlp_0.kernel.dtype)
    return self.mlp_2(jax.nn.silu(self.mlp_0(t_freq)))


@functools.lru_cache(maxsize=8)
def _rope_tables(theta: float, axes_dims: tuple, axes_lens: tuple) -> tuple:
  """Per-axis (cos, sin) tables [axis_len, axis_dim // 2], built like the reference (float64 angles -> float32)."""
  tables = []
  for d, e in zip(axes_dims, axes_lens):
    freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    angles = np.outer(np.arange(e, dtype=np.float64), freqs).astype(np.float32)
    tables.append((np.cos(angles), np.sin(angles)))
  return tuple(tables)


def rope_cos_sin(pos_ids: Array, theta: float, axes_dims: Sequence[int], axes_lens: Sequence[int]):
  """3D RoPE: int position ids [B, S, 3] -> (cos, sin) [B, S, head_dim // 2] float32."""
  tables = _rope_tables(float(theta), tuple(axes_dims), tuple(axes_lens))
  cos, sin = [], []
  for axis, (cos_t, sin_t) in enumerate(tables):
    ids = pos_ids[..., axis]
    cos.append(jnp.asarray(cos_t)[ids])
    sin.append(jnp.asarray(sin_t)[ids])
  return jnp.concatenate(cos, axis=-1), jnp.concatenate(sin, axis=-1)


def apply_rotary_emb(x: Array, cos: Array, sin: Array) -> Array:
  """Interleaved-pair (complex) RoPE in float32; x [B, S, H, D], cos/sin [B, S, D // 2]."""
  xf = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
  c, s = cos[:, :, None, :], sin[:, :, None, :]
  x0, x1 = xf[..., 0], xf[..., 1]
  out = jnp.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)
  return out.reshape(x.shape).astype(x.dtype)


class MingAttention(nnx.Module):
  """Single-stream self attention: q/k/v projections, per-head RMS qk-norm, 3D RoPE, output projection."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      n_heads: int,
      qk_norm: bool,
      sharding: Sharding | None,
      dtype: Any,
      weights_dtype: Any,
      precision: Any,
      attention: str,
      flash_block_sizes: BlockSizes | None,
      flash_min_seq_length: int,
      mask_padding_tokens: bool,
      attention_config: dict,
  ):
    self.heads = n_heads
    self.head_dim = dim // n_heads
    kw = {"sharding": sharding, "dtype": dtype, "weights_dtype": weights_dtype, "precision": precision}
    self.to_q = _linear(rngs, dim, dim, False, name="attn_q", axes=("embed", "heads"), **kw)
    self.to_k = _linear(rngs, dim, dim, False, name="attn_k", axes=("embed", "heads"), **kw)
    self.to_v = _linear(rngs, dim, dim, False, name="attn_v", axes=("embed", "heads"), **kw)
    self.to_out = _linear(rngs, dim, dim, False, name="attn_out", axes=("heads", "embed"), **kw)
    self.norm_q = nnx.data(None)
    self.norm_k = nnx.data(None)
    if qk_norm:  # diffusers `Attention(qk_norm="rms_norm", eps=1e-5)`: per head, eps fixed at 1e-5.
      self.norm_q = MingRMSNorm(self.head_dim, 1e-5, sharding, "attn_q_norm", (None,), weights_dtype)
      self.norm_k = MingRMSNorm(self.head_dim, 1e-5, sharding, "attn_k_norm", (None,), weights_dtype)
    self.attention_op = AttentionOp(
        sharding=sharding,
        attention_kernel=attention,
        scale=self.head_dim**-0.5,
        heads=n_heads,
        dim_head=self.head_dim,
        # Queries sequence-sharded (`activation_length`), keys / values kept whole on every shard
        # (`activation_kv_length`): unlike Wan's head-sharded self attention, this works for 30 heads on 8 chips.
        axis_names_q=(BATCH, HEAD, LENGTH, D_KV),
        axis_names_kv=(BATCH, HEAD, KV_LENGTH, D_KV),
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        mask_padding_tokens=mask_padding_tokens,
        use_base2_exp=attention_config.get("use_base2_exp", False),
        use_experimental_scheduler=attention_config.get("use_experimental_scheduler", False),
    )

  def __call__(self, x: Array, attn_mask: Optional[Array], freqs: tuple) -> Array:
    b, s, _ = x.shape
    q = self.to_q(x).reshape(b, s, self.heads, self.head_dim)
    k = self.to_k(x).reshape(b, s, self.heads, self.head_dim)
    v = self.to_v(x).reshape(b, s, self.heads, self.head_dim)
    if self.norm_q is not None:
      q = self.norm_q(q)
      k = self.norm_k(k)
    q = apply_rotary_emb(q, *freqs)
    k = apply_rotary_emb(k, *freqs)
    q, k, v = (jnp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))
    out = self.attention_op.apply_attention(q, k, v, attention_mask=attn_mask)  # [B, S, H * D]
    return self.to_out(out.astype(x.dtype))


class FeedForward(nnx.Module):
  """SwiGLU: w2(silu(w1 x) * w3 x)."""

  def __init__(self, rngs, dim, hidden_dim, sharding, dtype, weights_dtype, precision):
    kw = {"sharding": sharding, "dtype": dtype, "weights_dtype": weights_dtype, "precision": precision}
    self.w1 = _linear(rngs, dim, hidden_dim, False, name="ffn_w1", axes=("embed", "mlp"), **kw)
    self.w2 = _linear(rngs, hidden_dim, dim, False, name="ffn_w2", axes=("mlp", "embed"), **kw)
    self.w3 = _linear(rngs, dim, hidden_dim, False, name="ffn_w3", axes=("embed", "mlp"), **kw)

  def __call__(self, x: Array) -> Array:
    return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class MingTransformerBlock(nnx.Module):
  """Sandwich-norm DiT block; `modulation=True` adds tanh-gated adaLN scales from the timestep embedding."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      n_heads: int,
      norm_eps: float,
      qk_norm: bool,
      modulation: bool,
      sharding: Sharding | None,
      dtype: Any,
      weights_dtype: Any,
      precision: Any,
      attention_kwargs: dict,
  ):
    self.modulation = modulation
    self.sharding = sharding
    self.attention = MingAttention(
        rngs, dim, n_heads, qk_norm, sharding, dtype, weights_dtype, precision, **attention_kwargs
    )
    self.feed_forward = FeedForward(rngs, dim, int(dim / 3 * 8), sharding, dtype, weights_dtype, precision)
    for name in ("attention_norm1", "ffn_norm1", "attention_norm2", "ffn_norm2"):
      setattr(self, name, MingRMSNorm(dim, norm_eps, sharding, name, ("norm",), weights_dtype))
    self.adaLN_modulation = nnx.data(None)  # pylint: disable=invalid-name
    if modulation:
      self.adaLN_modulation = _linear(  # pylint: disable=invalid-name
          rngs,
          min(dim, ADALN_EMBED_DIM),
          4 * dim,
          True,
          sharding=sharding,
          name="adaln",
          axes=(None, "embed"),
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )

  def __call__(self, x: Array, attn_mask: Optional[Array], freqs: tuple, adaln_input: Optional[Array] = None) -> Array:
    if self.modulation:
      mod = self.adaLN_modulation(adaln_input)[:, None, :]
      scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(mod, 4, axis=-1)
      gate_msa, gate_mlp = jnp.tanh(gate_msa), jnp.tanh(gate_mlp)
      scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
      attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask, freqs)
      x = x + gate_msa * self.attention_norm2(attn_out)
      x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    else:
      attn_out = self.attention(self.attention_norm1(x), attn_mask, freqs)
      x = x + self.attention_norm2(attn_out)
      x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
    return constrain(self.sharding, x, "block_output", (BATCH, LENGTH, EMBED))


class FinalLayer(nnx.Module):
  """LayerNorm (no affine, eps 1e-6, fp32) * (1 + adaLN(silu(c))) -> Linear. HF `adaLN_modulation.1`, `linear`."""

  def __init__(self, rngs, hidden_size, out_channels, sharding, dtype, weights_dtype, precision):
    kw = {"sharding": sharding, "dtype": dtype, "weights_dtype": weights_dtype, "precision": precision}
    self.linear = _linear(rngs, hidden_size, out_channels, True, name="final_linear", axes=("embed", None), **kw)
    self.adaLN_modulation = _linear(  # pylint: disable=invalid-name
        rngs, min(hidden_size, ADALN_EMBED_DIM), hidden_size, True, name="final_adaln", axes=(None, "embed"), **kw
    )

  def __call__(self, x: Array, c: Array) -> Array:
    scale = 1.0 + self.adaLN_modulation(jax.nn.silu(c))
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    normed = ((xf - mean) * jax.lax.rsqrt(var + 1e-6)).astype(x.dtype)
    return self.linear(normed * scale[:, None, :])


def _round_up(x: Array, multiple: int) -> Array:
  return (x + multiple - 1) // multiple * multiple


class MingImageTransformer(nnx.Module):
  """Ming-Image-0.1 diffusion transformer.

  `__init__` takes the HF `transformer/config.json` keys plus the runtime kwargs the diffusion pipelines inject
  (`sharding`, `dtype`, `weights_dtype`, `precision`, `attention`, `flash_block_sizes`, `flash_min_seq_length`,
  `scan_layers`, ...). `scan_layers=True` stacks the main `layers` (param paths `layers-<name>` with a leading layer
  axis); the two-layer refiners are always per-layer lists. Only `alignment_padding_mode="zero_masked"` (the
  checkpoint's mode; the learned pad tokens of the layer-decompose checkpoints are not supported) and a single patch
  size are supported.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      all_patch_size: Sequence[int] = (2,),
      all_f_patch_size: Sequence[int] = (1,),
      in_channels: int = 16,
      dim: int = 3840,
      n_layers: int = 30,
      n_refiner_layers: int = 2,
      n_heads: int = 30,
      n_kv_heads: int = 30,
      norm_eps: float = 1e-5,
      qk_norm: bool = True,
      cap_feat_dim: int = 2560,
      rope_theta: float = 256.0,
      t_scale: float = 1000.0,
      axes_dims: Sequence[int] = (32, 48, 48),
      axes_lens: Sequence[int] = (20480, 512, 512),
      alignment_padding_mode: str = ZERO_MASKED_PADDING,
      multi_frame_output: bool = False,
      siglip_feat_dim: Optional[int] = None,
      # Runtime kwargs.
      sharding: Sharding | None = None,
      dtype: Any = jnp.float32,
      weights_dtype: Any = jnp.float32,
      precision: Any = None,
      attention: str = "dot_product",
      flash_block_sizes: BlockSizes | None = None,
      flash_min_seq_length: int = 4096,
      mask_padding_tokens: bool = True,
      scan_layers: bool = False,
      attention_config: Optional[dict] = None,
      dropout: float = 0.0,  # pylint: disable=unused-argument
      enable_jax_named_scopes: bool = False,  # pylint: disable=unused-argument
  ):
    if alignment_padding_mode != ZERO_MASKED_PADDING:
      raise NotImplementedError(f"Only alignment_padding_mode={ZERO_MASKED_PADDING!r} is supported.")
    if len(all_patch_size) != 1 or tuple(all_f_patch_size) != (1,):
      raise NotImplementedError("Only a single spatial patch size with f_patch_size=1 is supported.")
    if n_kv_heads != n_heads:
      raise NotImplementedError("The reference attention ignores n_kv_heads; it must equal n_heads.")
    if siglip_feat_dim is not None:
      raise NotImplementedError("siglip_feat_dim is not supported.")
    del multi_frame_output  # Single frame (F=1): the reference's frame slicing is a no-op.
    head_dim = dim // n_heads
    if head_dim != sum(axes_dims):
      raise ValueError(f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}.")

    self.patch_size = int(all_patch_size[0])
    self.in_channels = in_channels
    self.dim = dim
    self.n_layers = n_layers
    self.t_scale = t_scale
    self.rope_theta = rope_theta
    self.axes_dims = tuple(axes_dims)
    self.axes_lens = tuple(axes_lens)
    self.scan_layers = scan_layers
    self.dtype = dtype
    self.sharding = sharding

    kw = {"sharding": sharding, "dtype": dtype, "weights_dtype": weights_dtype, "precision": precision}
    patch_dim = self.patch_size * self.patch_size * in_channels
    self.all_x_embedder = _linear(rngs, patch_dim, dim, True, name="x_embedder", axes=(None, "embed"), **kw)
    self.all_final_layer = FinalLayer(rngs, dim, patch_dim, **kw)
    self.t_embedder = TimestepEmbedder(
        rngs, min(dim, ADALN_EMBED_DIM), T_EMBEDDER_MID_SIZE, sharding, weights_dtype, precision
    )
    self.cap_embedder = nnx.List(
        [
            MingRMSNorm(cap_feat_dim, norm_eps, sharding, "cap_norm", ("norm",), weights_dtype),
            _linear(rngs, cap_feat_dim, dim, True, name="cap_linear", axes=(None, "embed"), **kw),
        ]
    )

    block_kwargs = {
        "dim": dim,
        "n_heads": n_heads,
        "norm_eps": norm_eps,
        "qk_norm": qk_norm,
        "sharding": sharding,
        "dtype": dtype,
        "weights_dtype": weights_dtype,
        "precision": precision,
        "attention_kwargs": {
            "attention": attention,
            "flash_block_sizes": flash_block_sizes,
            "flash_min_seq_length": flash_min_seq_length,
            "mask_padding_tokens": mask_padding_tokens,
            "attention_config": dict(attention_config or {}),
        },
    }
    self.noise_refiner = nnx.List(
        [MingTransformerBlock(rngs, modulation=True, **block_kwargs) for _ in range(n_refiner_layers)]
    )
    self.context_refiner = nnx.List(
        [MingTransformerBlock(rngs, modulation=False, **block_kwargs) for _ in range(n_refiner_layers)]
    )

    if scan_layers:

      @nnx.split_rngs(splits=n_layers)
      @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
      def init_block(rngs):
        return MingTransformerBlock(rngs, modulation=True, **block_kwargs)

      self.layers = init_block(rngs)
    else:
      self.layers = nnx.List([MingTransformerBlock(rngs, modulation=True, **block_kwargs) for _ in range(n_layers)])

  def _rope(self, pos_ids: Array) -> tuple:
    return rope_cos_sin(pos_ids, self.rope_theta, self.axes_dims, self.axes_lens)

  def _embed_captions(self, cap_feats, cap_mask, cap_feats_2, cap2_mask):
    """-> (caption tokens [B, Lc (+ Lc2), dim], keep mask, total valid length [B], position-in-caption [B, L])."""
    parts, masks = [], []
    if cap_feats is not None:
      norm, linear = self.cap_embedder
      parts.append(linear(norm(cap_feats)))
      masks.append(cap_mask)
    if cap_feats_2 is not None:
      parts.append(cap_feats_2.astype(parts[0].dtype if parts else self.dtype))
      masks.append(cap2_mask)
    if not parts:
      raise ValueError("At least one of cap_feats / cap_feats_2 is required.")
    cap = jnp.concatenate(parts, axis=1)
    valid = jnp.concatenate([m.astype(jnp.bool_) for m in masks], axis=1)
    cap = jnp.where(valid[..., None], cap, jnp.zeros((), cap.dtype))  # zero_masked padding
    # The reference concatenates [cap_feats[:L1], cap_feats_2[:L2]]: the i-th valid token sits at position i.
    pos = jnp.cumsum(valid.astype(jnp.int32), axis=1) - 1
    return cap, valid, jnp.sum(valid.astype(jnp.int32), axis=1), jnp.maximum(pos, 0)

  def _run_layers(self, x, attn_mask, freqs, adaln_input):
    """Runs the joint-stream `layers` (a Python loop, or `nnx.scan` over the stacked layers)."""
    if not self.scan_layers:
      for layer in self.layers:
        x = layer(x, attn_mask, freqs, adaln_input)
      return x

    def body(carry, block):
      return block(carry, attn_mask, freqs, adaln_input), None

    x, _ = nnx.scan(body, length=self.n_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(x, self.layers)
    return x

  @jax.named_scope("MingImageTransformer")
  def __call__(
      self,
      latents: Array,
      timestep: Array,
      cap_feats: Optional[Array],
      cap_mask: Optional[Array],
      cap_feats_2: Optional[Array] = None,
      cap2_mask: Optional[Array] = None,
  ) -> Array:
    """Predicts the flow (the reference output, not negated).

    Args:
      latents: [B, H, W, C] channel-last latents (H, W divisible by the patch size).
      timestep: [B] float, already normalized as the reference pipeline passes it ((1000 - t) / 1000).
      cap_feats: [B, Lc, cap_feat_dim] text-encoder features (embedded by `cap_embedder`), or None.
      cap_mask: [B, Lc] bool keep mask of `cap_feats` (valid tokens first).
      cap_feats_2: [B, Lc2, dim] connector features (appended after `cap_feats`, not embedded), or None.
      cap2_mask: [B, Lc2] bool keep mask of `cap_feats_2` (valid tokens first).

    Returns:
      [B, H, W, C].
    """
    b, h, w, c = latents.shape
    p = self.patch_size
    ht, wt = h // p, w // p
    n_img = ht * wt
    if cap_mask is None and cap_feats is not None:
      cap_mask = jnp.ones(cap_feats.shape[:2], jnp.bool_)
    if cap2_mask is None and cap_feats_2 is not None:
      cap2_mask = jnp.ones(cap_feats_2.shape[:2], jnp.bool_)

    adaln = self.t_embedder(timestep * self.t_scale)

    # "(h ph) (w pw) c -> (h w) (ph pw c)"
    x = latents.reshape(b, ht, p, wt, p, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, n_img, p * p * c)
    x = self.all_x_embedder(x.astype(self.dtype))
    adaln_input = adaln.astype(x.dtype)

    cap, cap_valid, cap_len, cap_pos = self._embed_captions(cap_feats, cap_mask, cap_feats_2, cap2_mask)
    cap = cap.astype(x.dtype)
    lc = cap.shape[1]

    # Position ids: caption (1 + i, 0, 0); image (round_up(cap_len, 32) + 1, h, w).
    zeros_c = jnp.zeros((b, lc), jnp.int32)
    cap_ids = jnp.stack([1 + cap_pos, zeros_c, zeros_c], axis=-1)
    img_start = _round_up(cap_len, SEQ_MULTI_OF) + 1
    hh, ww = jnp.meshgrid(jnp.arange(ht, dtype=jnp.int32), jnp.arange(wt, dtype=jnp.int32), indexing="ij")
    img_ids = jnp.stack(
        [
            jnp.broadcast_to(img_start[:, None], (b, n_img)),
            jnp.broadcast_to(hh.reshape(1, n_img), (b, n_img)),
            jnp.broadcast_to(ww.reshape(1, n_img), (b, n_img)),
        ],
        axis=-1,
    )
    x_freqs = self._rope(img_ids)
    cap_freqs = self._rope(cap_ids)

    for layer in self.noise_refiner:
      x = layer(x, None, x_freqs, adaln_input)
    for layer in self.context_refiner:
      cap = layer(cap, cap_valid, cap_freqs)

    unified = jnp.concatenate([x, cap], axis=1)
    unified = constrain(self.sharding, unified, "unified_input", (BATCH, LENGTH, EMBED))
    unified_mask = jnp.concatenate([jnp.ones((b, n_img), jnp.bool_), cap_valid], axis=1)
    unified_freqs = tuple(jnp.concatenate([xf, cf], axis=1) for xf, cf in zip(x_freqs, cap_freqs))
    unified = self._run_layers(unified, unified_mask, unified_freqs, adaln_input)

    out = self.all_final_layer(unified[:, :n_img], adaln_input)
    # "(h w) (ph pw c) -> (h ph) (w pw) c"
    out = out.reshape(b, ht, wt, p, p, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
    return out


def transformer_param_mapping(hf_config: dict, scan_layers: bool = False) -> tuple[dict, dict]:
  """Returns `({maxtext_key: hf_key | [hf_key per layer]}, {maxtext_key: hook(weight, target_shape)})`.

  Same format as `WAN2_1_HF_TO_MAXTEXT._build`: MaxText keys are NNX param paths joined by "-"; with `scan_layers`,
  the main `layers` map to a list of per-layer HF keys stacked along axis 0.
  """
  mapping: dict[str, Any] = {}
  hooks: dict[str, Callable] = {}

  def transpose(x, target_shape):  # pylint: disable=unused-argument
    """PyTorch linear weight (out, in) -> Flax kernel (in, out)."""
    return x.T

  def add(maxtext_key, hf_key, hook=None):
    mapping[maxtext_key] = hf_key
    if hook is not None:
      hooks[maxtext_key] = hook

  patch = hf_config.get("all_patch_size", (2,))[0]
  f_patch = hf_config.get("all_f_patch_size", (1,))[0]
  pk = f"{patch}-{f_patch}"
  n_layers = hf_config.get("n_layers", 30)
  n_refiner = hf_config.get("n_refiner_layers", 2)
  qk_norm = hf_config.get("qk_norm", True)

  def block(modulation):
    entries = []
    for name in ("to_q", "to_k", "to_v"):
      entries.append((f"attention-{name}-kernel", f"attention.{name}.weight", transpose))
    entries.append(("attention-to_out-kernel", "attention.to_out.0.weight", transpose))
    if qk_norm:
      entries += [(f"attention-{n}-scale", f"attention.{n}.weight", None) for n in ("norm_q", "norm_k")]
    for name in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
      entries.append((f"{name}-scale", f"{name}.weight", None))
    for name in ("w1", "w2", "w3"):
      entries.append((f"feed_forward-{name}-kernel", f"feed_forward.{name}.weight", transpose))
    if modulation:
      entries += [
          ("adaLN_modulation-kernel", "adaLN_modulation.0.weight", transpose),
          ("adaLN_modulation-bias", "adaLN_modulation.0.bias", None),
      ]
    return entries

  for prefix, count, modulation, scanned in (
      ("noise_refiner", n_refiner, True, False),
      ("context_refiner", n_refiner, False, False),
      ("layers", n_layers, True, scan_layers),
  ):
    for mt, hf, hook in block(modulation):
      if scanned:
        add(f"{prefix}-{mt}", [f"{prefix}.{i}.{hf}" for i in range(count)], hook)
      else:
        for i in range(count):
          add(f"{prefix}-{i}-{mt}", f"{prefix}.{i}.{hf}", hook)

  for mt, hf in (
      ("all_x_embedder", f"all_x_embedder.{pk}"),
      ("all_final_layer-linear", f"all_final_layer.{pk}.linear"),
      ("all_final_layer-adaLN_modulation", f"all_final_layer.{pk}.adaLN_modulation.1"),
      ("t_embedder-mlp_0", "t_embedder.mlp.0"),
      ("t_embedder-mlp_2", "t_embedder.mlp.2"),
      ("cap_embedder-1", "cap_embedder.1"),
  ):
    add(f"{mt}-kernel", f"{hf}.weight", transpose)
    add(f"{mt}-bias", f"{hf}.bias")
  add("cap_embedder-0-scale", "cap_embedder.0.weight")
  return mapping, hooks
