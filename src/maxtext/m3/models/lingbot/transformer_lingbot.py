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

"""LingBot-World-Fast causal diffusion transformer (Flax NNX).

LingBot-World-Fast (`robbyant/lingbot-world-fast`) is a Wan 2.2 I2V-A14B DiT distilled into a few-step, block-causal,
autoregressive world model with camera control. Compared to the Wan 2.1 `WanModel` (`m3/models/wan/transformer_wan.py`)
it adds:

* Camera conditioning: per-pixel Plücker rays are patchified and projected (`patch_embedding_wancamctrl` +
  `c2ws_hidden_states_layer{1,2}`) once, and every block injects them as a FiLM after self-attention
  (`cam_injector_layer{1,2}`, `cam_scale_layer`, `cam_shift_layer`).
* Block-causal self-attention with a KV cache: the video is generated `chunk_size` latent frames at a time; a chunk
  attends to itself and to the (clean) keys / values of all previous chunks, with RoPE positions offset by the chunk's
  start frame.
* I2V conditioning through channel concatenation (`in_dim` = 16 noise + 4 mask + 16 image-latent channels); no CLIP
  image branch.

The building blocks (attention projections, feed-forward, time / text embedders, LayerNorms) are the Wan 2.1 ones, with
the same parameter names, so the checkpoint mapping reuses the Wan 2.1 one. Numerics follow the reference
(`wan/modules/model_fast.py`): bf16 matmuls, a float32 residual stream, float32 modulation / time embedding / head.

Constructor arguments are the fields of the original `config.json` (`dim`, `ffn_dim`, `num_heads`, ...), so the converter
can pass the checkpoint config straight through.
"""

from typing import Optional, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common import common_types
from maxtext.m3.core.sharding import Sharding
from maxtext.m3.models.wan.transformer_wan import (
    FP32LayerNorm,
    WanAttention,
    WanFeedForward,
    WanTimeTextImageEmbedding,
    _pad_axis,
)
from maxtext.m3.models.wan.utils_wan import (
    BlockSizes,
    SELF_ATTN_HEAD,
    constrain,
    param_metadata,
    tokamax_splash,
)

try:
  from tokamax._src.ops.splash_attention import splash_attention_mask as tokamax_splash_mask
except ImportError:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask

Array = common_types.Array
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD

# Number of Plücker channels per pixel (ray origin + direction) for the camera-controlled ("cam") checkpoint.
CAM_CONTROL_DIM = 6
# Latent-pixel stride of the VAE: the Plücker map is computed at full resolution and folded into channels.
VAE_SPATIAL_STRIDE = 8


# =============================================================================
# 1. RoPE (3D, with a start-frame offset for the causal chunks)
# =============================================================================


def rope_axis_dims(head_dim: int) -> Tuple[int, int, int]:
  """Rotary dims (t, h, w) of a head: the reference splits `head_dim` as d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)."""
  hw = 2 * (head_dim // 6)
  return head_dim - 2 * hw, hw, hw


def rope_angles(
    head_dim: int,
    start_frame: int,
    num_frames: int,
    height: int,
    width: int,
    theta: float = 10000.0,
) -> np.ndarray:
  """Rotation angles [num_frames * height * width, head_dim // 2] (float64) of `causal_rope_apply`.

  Computed on the host in float64 like the reference (`rope_params`); the frame axis starts at `start_frame`.
  """
  angles = []
  for dim, length, offset in zip(rope_axis_dims(head_dim), (num_frames, height, width), (start_frame, 0, 0)):
    inv_freq = 1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles.append(np.outer(np.arange(offset, offset + length, dtype=np.float64), inv_freq))
  f_ang, h_ang, w_ang = angles[0], angles[1], angles[2]
  grid = np.concatenate(
      [
          np.broadcast_to(f_ang[:, None, None, :], (num_frames, height, width, f_ang.shape[-1])),
          np.broadcast_to(h_ang[None, :, None, :], (num_frames, height, width, h_ang.shape[-1])),
          np.broadcast_to(w_ang[None, None, :, :], (num_frames, height, width, w_ang.shape[-1])),
      ],
      axis=-1,
  )
  return grid.reshape(num_frames * height * width, -1)


def rope_cos_sin(head_dim: int, start_frame: int, num_frames: int, height: int, width: int) -> Tuple[Array, Array]:
  """(cos, sin) [L, head_dim // 2] float32 tables for `apply_rope_interleaved`."""
  angles = rope_angles(head_dim, start_frame, num_frames, height, width)
  return jnp.asarray(np.cos(angles), jnp.float32), jnp.asarray(np.sin(angles), jnp.float32)


def apply_rope_interleaved(x: Array, cos: Array, sin: Array) -> Array:
  """Rotates interleaved pairs (x[2i], x[2i+1]) of x [B, L, H, D] by the [L, D // 2] angles, in float32."""
  dtype = x.dtype
  pairs = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
  x0, x1 = pairs[..., 0], pairs[..., 1]
  cos, sin = cos[None, :, None, :], sin[None, :, None, :]
  out = jnp.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
  return out.reshape(x.shape).astype(dtype)


# =============================================================================
# 2. Block-causal self-attention with a KV cache
# =============================================================================


def _masked_dot_product_attention(query: Array, key: Array, value: Array, kv_valid: Array | None, scale: float) -> Array:
  """XLA attention of q [B, Q, H, D] over k / v [B, K, H, D]; keys with kv_valid [K] == False are ignored."""
  scores = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32) * scale
  if kv_valid is not None:
    scores = jnp.where(kv_valid[None, None, None, :], scores, -jnp.inf)
  weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
  return jnp.einsum("bhqk,bkhd->bqhd", weights, value, preferred_element_type=jnp.float32).astype(query.dtype)


def _masked_flash_attention(
    query: Array,
    key: Array,
    value: Array,
    kv_valid: Array | None,
    scale: float,
    block_sizes: BlockSizes | None,
    heads_spec,
) -> Array:
  """Tokamax Splash attention of q [B, Q, H, D] over k / v [B, K, H, D], heads sharded by `heads_spec`.

  Unlike Wan's `_flash_attention` (which treats Q != K as short-text cross attention and spans the whole KV in one
  block), the KV cache is long: q and kv are blocked independently, and the not-yet-written cache tail (kv_valid False)
  and the block padding are excluded with segment ids.
  """
  q_len, kv_len, head_dim = query.shape[1], key.shape[1], query.shape[-1]
  block_q = min(block_sizes.block_q if block_sizes is not None else 512, -q_len % 128 + q_len)
  block_kv = min(block_sizes.block_kv if block_sizes is not None else 512, -kv_len % 128 + kv_len)
  block_kv_compute = min(block_sizes.block_kv_compute if block_sizes is not None else block_kv, block_kv)
  config = tokamax_splash.SplashConfig(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q,
      block_kv_dkv=block_kv,
      block_kv_dkv_compute=block_kv_compute,
  )
  # [B, S, H, D] -> [B, H, S, D]; scale q (Splash does not scale).
  q = jnp.transpose(query * jnp.asarray(scale, query.dtype), (0, 2, 1, 3))
  k = jnp.transpose(key, (0, 2, 1, 3))
  v = jnp.transpose(value, (0, 2, 1, 3))
  if kv_valid is None:
    kv_valid = jnp.ones((kv_len,), jnp.bool_)

  def local_attention(q, k, v, kv_valid):
    q = _pad_axis(_pad_axis(q, 2, block_q), 3, 128)
    k = _pad_axis(_pad_axis(k, 2, block_kv), 3, 128)
    v = _pad_axis(_pad_axis(v, 2, block_kv), 3, 128)
    kernel = tokamax_splash.make_splash_mha(
        mask=tokamax_splash_mask.FullMask(_shape=(q.shape[2], k.shape[2])),
        config=config,
        q_seq_shards=1,
        save_residuals=False,
    )
    q_ids = (jnp.arange(q.shape[2]) < q_len).astype(jnp.int32)
    kv_ids = _pad_axis(kv_valid.astype(jnp.int32), 0, block_kv)
    segment_ids = tokamax_splash.SegmentIds(q=q_ids, kv=kv_ids)
    out = jax.vmap(kernel, in_axes=(0, 0, 0, None))(q, k, v, segment_ids)
    return out[:, :, :q_len, :head_dim].astype(q.dtype)

  mesh = jax.sharding.get_abstract_mesh()
  if mesh.empty:
    out = local_attention(q, k, v, kv_valid)
  else:
    spec = jax.sharding.PartitionSpec(None, heads_spec, None, None)
    out = jax.shard_map(
        local_attention,
        mesh=mesh,
        in_specs=(spec, spec, spec, jax.sharding.PartitionSpec()),
        out_specs=spec,
        check_vma=False,
    )(q, k, v, kv_valid)
  return jnp.transpose(out, (0, 2, 1, 3))


class LingBotSelfAttention(WanAttention):
  """Wan self-attention (same projections / QK RMSNorm / parameter names) made block-causal with a KV cache.

  `__call__` projects the current chunk, applies RoPE at its absolute positions, writes its keys / values into the
  layer's cache at `current_start` and attends over the cache prefix `[0, current_start + chunk_len)`. It returns the
  chunk's (roped) keys / values so the caller can commit them to the cache.
  """

  def __init__(self, *args, attention_kernel: str = "dot_product", **kwargs):
    super().__init__(*args, attention_kernel=attention_kernel, is_self_attention=True, **kwargs)
    self.kernel = attention_kernel
    self.scale = self.dim_head**-0.5
    if self.sharding is None:
      self.heads_spec = None
    else:
      self.heads_spec = self.sharding.map_axis(SELF_ATTN_HEAD, "self_attn_heads", None)

  def __call__(  # pylint: disable=arguments-renamed,arguments-differ
      self,
      hidden_states: Array,
      rope_cos: Array,
      rope_sin: Array,
      cache_k: Optional[Array] = None,
      cache_v: Optional[Array] = None,
      current_start: Optional[Array] = None,
  ) -> Tuple[Array, Array, Array]:
    batch, length, _ = hidden_states.shape
    query = self.norm_q(self.query(hidden_states)).reshape(batch, length, self.heads, self.dim_head)
    key = self.norm_k(self.key(hidden_states)).reshape(batch, length, self.heads, self.dim_head)
    value = self.value(hidden_states).reshape(batch, length, self.heads, self.dim_head)
    query = apply_rope_interleaved(query, rope_cos, rope_sin)
    key = apply_rope_interleaved(key, rope_cos, rope_sin)

    if cache_k is None:
      full_key, full_value, kv_valid = key, value, None
    else:
      start = jnp.asarray(current_start, jnp.int32)
      zero = jnp.zeros((), jnp.int32)
      full_key = jax.lax.dynamic_update_slice(cache_k, key.astype(cache_k.dtype), (zero, start, zero, zero))
      full_value = jax.lax.dynamic_update_slice(cache_v, value.astype(cache_v.dtype), (zero, start, zero, zero))
      kv_valid = jnp.arange(cache_k.shape[1]) < start + length

    if self.kernel == "flash":
      out = _masked_flash_attention(
          query, full_key, full_value, kv_valid, self.scale, self.attention_op.flash_block_sizes, self.heads_spec
      )
    else:
      out = _masked_dot_product_attention(query, full_key, full_value, kv_valid, self.scale)
    out = self.proj_attn(out.reshape(batch, length, self.inner_dim))
    return out, key, value


# =============================================================================
# 3. Causal DiT block and model
# =============================================================================


def _linear(rngs, in_features, out_features, name, axes, dtype, weights_dtype, precision, sharding):
  return nnx.Linear(
      in_features,
      out_features,
      rngs=rngs,
      use_bias=True,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.initializers.xavier_uniform(),
      kernel_metadata=param_metadata(sharding, f"{name}_kernel", axes),
      bias_init=nnx.initializers.zeros,
      bias_metadata=param_metadata(sharding, f"{name}_bias", (axes[1],)),
  )


class LingBotBlock(nnx.Module):
  """Causal Wan block: modulated block-causal self-attention, camera FiLM, cross-attention and FFN.

  The residual stream is float32, as in the reference (`CausalWanAttentionBlock` adds the bf16 branch outputs under a
  float32 autocast). Parameter names follow `WanTransformerBlock`: `norm2` is the affine cross-attention norm and
  `norm3` the FFN norm (`norm3` / `norm2` in the original checkpoint).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      eps: float = 1e-6,
      sharding: Sharding | None = None,
      dtype: jnp.dtype = jnp.bfloat16,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      flash_block_sizes: BlockSizes = None,
      flash_min_seq_length: int = 0,
  ):
    self.sharding = sharding
    self.dtype = dtype
    attn_kwargs = {
        "rngs": rngs,
        "query_dim": dim,
        "heads": num_heads,
        "dim_head": dim // num_heads,
        "qk_norm": "rms_norm_across_heads",
        "eps": eps,
        "flash_block_sizes": flash_block_sizes,
        "sharding": sharding,
        "dtype": dtype,
        "weights_dtype": weights_dtype,
        "precision": precision,
        "mask_padding_tokens": True,
    }
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = LingBotSelfAttention(**attn_kwargs, attention_kernel=attention)
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)
    # Cross attention over the 512 text tokens (no padding mask, as the reference).
    self.attn2 = WanAttention(
        **attn_kwargs,
        attention_kernel=attention,
        flash_min_seq_length=flash_min_seq_length,
        is_self_attention=False,
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        sharding=sharding,
    )
    self.adaln_scale_shift_table = nnx.Param(jax.random.normal(rngs.params(), (1, 6, dim)) / dim**0.5)
    lin = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "sharding": sharding}
    self.cam_injector_layer1 = _linear(rngs, dim, dim, "cam_injector_1", ("embed", "mlp"), **lin)
    self.cam_injector_layer2 = _linear(rngs, dim, dim, "cam_injector_2", ("mlp", "embed"), **lin)
    self.cam_scale_layer = _linear(rngs, dim, dim, "cam_scale", ("embed", "mlp"), **lin)
    self.cam_shift_layer = _linear(rngs, dim, dim, "cam_shift", ("embed", "mlp"), **lin)

  def __call__(  # pylint: disable=arguments-differ
      self,
      hidden_states: Array,
      temb: Array,
      cam: Array,
      rope_cos: Array,
      rope_sin: Array,
      cross_kv: Tuple[Array, Array],
      cache_k: Optional[Array] = None,
      cache_v: Optional[Array] = None,
      current_start: Optional[Array] = None,
  ) -> Tuple[Array, Array, Array]:
    """hidden_states [B, L, D] float32, temb [B, 6, D] float32, cam [B, L, D] -> (hidden_states, chunk_k, chunk_v)."""
    x = hidden_states.astype(jnp.float32)
    x = constrain(self.sharding, x, "block_hidden", (BATCH, LENGTH, HEAD))
    mod = self.adaln_scale_shift_table[...].astype(jnp.float32) + temb.astype(jnp.float32)  # [B, 6, D]
    shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (mod[:, i : i + 1, :] for i in range(6))

    # 1. Block-causal self-attention.
    norm_x = self.norm1(x) * (1 + scale_msa) + shift_msa
    attn_out, chunk_k, chunk_v = self.attn1(
        norm_x.astype(self.dtype), rope_cos, rope_sin, cache_k, cache_v, current_start
    )
    x = x + attn_out.astype(jnp.float32) * gate_msa

    # 2. Camera injection (bf16 like the reference: (1 + scale) is rounded before scaling the fp32 stream).
    cam = cam.astype(self.dtype)
    cam_hidden = self.cam_injector_layer2(jax.nn.silu(self.cam_injector_layer1(cam))) + cam
    cam_scale = self.cam_scale_layer(cam_hidden)
    cam_shift = self.cam_shift_layer(cam_hidden)
    x = (1.0 + cam_scale).astype(jnp.float32) * x + cam_shift.astype(jnp.float32)

    # 3. Cross-attention over the cached text keys / values.
    cross_out = self.attn2(
        hidden_states=self.norm2(x).astype(self.dtype),
        encoder_hidden_states=cross_kv[0],
        cached_kv={"text": cross_kv},
    )
    x = x + cross_out.astype(jnp.float32)

    # 4. Feed-forward.
    norm_x = self.norm3(x) * (1 + c_scale) + c_shift
    ffn_out = self.ffn(norm_x.astype(self.dtype))
    x = x + ffn_out.astype(jnp.float32) * c_gate
    return x, chunk_k, chunk_v

  def compute_cross_kv(self, context: Array) -> Tuple[Array, Array]:
    return self.attn2.compute_kv(context)["text"]


class LingBotWorldModel(nnx.Module):
  """LingBot-World-Fast causal DiT (`WanModelFast` in the reference).

  Inputs are channel-first like the reference (`[B, C, F, H, W]` latents). One call denoises one chunk of `F` latent
  frames starting at latent frame `current_start // frame_seqlen`, attending to the KV cache of the previous chunks.
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type: str = "i2v",
      control_type: str = "cam",
      patch_size: Tuple[int, int, int] = (1, 2, 2),
      text_len: int = 512,
      in_dim: int = 36,
      dim: int = 5120,
      ffn_dim: int = 13824,
      freq_dim: int = 256,
      text_dim: int = 4096,
      out_dim: int = 16,
      num_heads: int = 40,
      num_layers: int = 40,
      eps: float = 1e-6,
      sharding: Sharding | None = None,
      dtype: jnp.dtype = jnp.bfloat16,
      weights_dtype: jnp.dtype = jnp.bfloat16,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      flash_block_sizes: BlockSizes = None,
      flash_min_seq_length: int = 0,
      scan_layers: bool = True,
  ):
    if control_type != "cam":
      raise NotImplementedError(f"control_type={control_type!r}: only the camera-pose ('cam') checkpoint is supported.")
    self.model_type = model_type
    self.patch_size = tuple(patch_size)
    self.text_len = text_len
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.sharding = sharding
    self.dtype = dtype
    fp32_precision = jax.lax.Precision.HIGHEST

    self.patch_embedding = nnx.Conv(
        in_dim,
        dim,
        rngs=rngs,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        kernel_metadata=param_metadata(sharding, "patch_embedding_kernel", (None, None, None, None, "conv_out")),
    )
    # Time embedding / projection in float32 (`HIGHEST`, the reference runs them under a float32 autocast), text
    # embedding as in Wan 2.1.
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=dim,
        time_freq_dim=freq_dim,
        time_proj_dim=dim * 6,
        text_embed_dim=text_dim,
        precision=fp32_precision,
        sharding=sharding,
    )
    lin = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "sharding": sharding}
    cam_in = CAM_CONTROL_DIM * VAE_SPATIAL_STRIDE * VAE_SPATIAL_STRIDE * int(np.prod(self.patch_size))
    self.patch_embedding_wancamctrl = _linear(rngs, cam_in, dim, "cam_patch_embedding", (None, "embed"), **lin)
    self.c2ws_hidden_states_layer1 = _linear(rngs, dim, dim, "cam_hidden_1", ("embed", "mlp"), **lin)
    self.c2ws_hidden_states_layer2 = _linear(rngs, dim, dim, "cam_hidden_2", ("mlp", "embed"), **lin)

    block_kwargs = {
        "dim": dim,
        "ffn_dim": ffn_dim,
        "num_heads": num_heads,
        "eps": eps,
        "sharding": sharding,
        "dtype": dtype,
        "weights_dtype": weights_dtype,
        "precision": precision,
        "attention": attention,
        "flash_block_sizes": flash_block_sizes,
        "flash_min_seq_length": flash_min_seq_length,
    }
    if scan_layers:

      @nnx.split_rngs(splits=num_layers)
      @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
      def init_block(rngs):
        return LingBotBlock(rngs=rngs, **block_kwargs)

      self.blocks = init_block(rngs)
    else:
      self.blocks = nnx.data([LingBotBlock(rngs=rngs, **block_kwargs) for _ in range(num_layers)])

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=out_dim * int(np.prod(self.patch_size)),
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        precision=fp32_precision,
        kernel_init=nnx.initializers.xavier_uniform(),
        kernel_metadata=param_metadata(sharding, "proj_out_kernel", ("embed", None)),
    )
    self.scale_shift_table = nnx.Param(jax.random.normal(rngs.params(), (1, 2, dim)) / dim**0.5)

  # --------------------------------------------------------------------------- helpers

  def kv_cache_shape(self, batch: int, kv_size: int) -> Tuple[int, ...]:
    """Self-attention cache shape `[num_layers, batch, kv_size, heads, head_dim]` (keys and values each)."""
    return (self.num_layers, batch, kv_size, self.num_heads, self.head_dim)

  def _layers(self):
    if self.scan_layers:
      return None
    return list(self.blocks)

  def compute_cross_kv(self, context: Array) -> Tuple[Array, Array]:
    """Text embedding + per-layer cross-attention K / V of the [B, text_len, text_dim] T5 states -> ([L, B, T, D], ...)."""
    context = self.condition_embedder.text_embedder(context).astype(self.dtype)
    if self.scan_layers:

      @nnx.vmap(in_axes=(0, None), out_axes=0, transform_metadata={nnx.PARTITION_NAME: None})
      def per_layer(block, ctx):
        return block.compute_cross_kv(ctx)

      return per_layer(self.blocks, context)
    kvs = [block.compute_cross_kv(context) for block in self.blocks]
    return jnp.stack([k for k, _ in kvs]), jnp.stack([v for _, v in kvs])

  def embed_camera(self, cam: Array) -> Array:
    """Plücker map [B, 6 * 64, F, H, W] (8x8 pixels folded into channels) -> camera tokens [B, F*H/2*W/2, D]."""
    batch, channels, frames, height, width = cam.shape
    p_t, p_h, p_w = self.patch_size
    # '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)'
    cam = cam.reshape(batch, channels, frames // p_t, p_t, height // p_h, p_h, width // p_w, p_w)
    cam = jnp.transpose(cam, (0, 2, 4, 6, 1, 3, 5, 7))
    cam = cam.reshape(batch, -1, channels * p_t * p_h * p_w).astype(self.dtype)
    cam = self.patch_embedding_wancamctrl(cam)
    return cam + self.c2ws_hidden_states_layer2(jax.nn.silu(self.c2ws_hidden_states_layer1(cam)))

  # --------------------------------------------------------------------------- forward

  @jax.named_scope("LingBotWorldModel")
  def __call__(  # pylint: disable=arguments-differ
      self,
      latents: Array,
      timestep: Array,
      condition: Array,
      cam: Array,
      cross_kv: Tuple[Array, Array],
      rope_cos: Array,
      rope_sin: Array,
      kv_cache: Optional[Tuple[Array, Array]] = None,
      current_start: Optional[Array] = None,
  ) -> Tuple[Array, Tuple[Array, Array]]:
    """Denoises one chunk.

    Args:
      latents: [B, out_dim, F, H, W] noisy latents of the chunk (float32).
      timestep: [B] diffusion timesteps (0..1000).
      condition: [B, in_dim - out_dim, F, H, W] I2V conditioning (first-frame mask + image latents) of the chunk.
      cam: [B, 6 * 64, F, H, W] Plücker map of the chunk.
      cross_kv: per-layer text K / V from `compute_cross_kv`.
      rope_cos / rope_sin: [F * H/2 * W/2, head_dim / 2] from `rope_cos_sin` at the chunk's start frame.
      kv_cache: (keys, values) `[num_layers, B, kv_size, H, head_dim]`, or None for a cache-free (single chunk) call.
      current_start: token offset of the chunk in the cache.

    Returns:
      (flow prediction [B, out_dim, F, H, W] float32, (chunk keys, chunk values) `[num_layers, B, L, H, head_dim]`).
    """
    batch, _, frames, height, width = latents.shape
    p_t, p_h, p_w = self.patch_size
    f, h, w = frames // p_t, height // p_h, width // p_w

    x = jnp.concatenate([latents.astype(jnp.float32), condition.astype(jnp.float32)], axis=1)
    x = jnp.transpose(x, (0, 2, 3, 4, 1)).astype(self.dtype)
    x = self.patch_embedding(x).reshape(batch, f * h * w, self.dim).astype(jnp.float32)

    # Time embedding (float32), shared by every token of the chunk: [B, D] and the 6 block modulations [B, 6, D].
    t_freq = self.condition_embedder.timesteps_proj(timestep.astype(jnp.float32))
    temb = self.condition_embedder.time_embedder(t_freq)
    temb_proj = self.condition_embedder.time_proj(jax.nn.silu(temb)).reshape(batch, 6, self.dim)

    cam_tokens = self.embed_camera(cam)

    if kv_cache is not None:
      cache_k, cache_v = kv_cache
    else:
      cache_k = cache_v = None

    if self.scan_layers:

      def body(carry, layer_inputs):
        block, cross_k, cross_v, layer_k, layer_v = layer_inputs
        out, chunk_k, chunk_v = block(
            carry, temb_proj, cam_tokens, rope_cos, rope_sin, (cross_k, cross_v), layer_k, layer_v, current_start
        )
        return out, (chunk_k, chunk_v)

      x, (new_k, new_v) = nnx.scan(body, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
          x, (self.blocks, cross_kv[0], cross_kv[1], cache_k, cache_v)
      )
    else:
      new_ks, new_vs = [], []
      for i, block in enumerate(self.blocks):
        layer_k = None if cache_k is None else cache_k[i]
        layer_v = None if cache_v is None else cache_v[i]
        x, chunk_k, chunk_v = block(
            x,
            temb_proj,
            cam_tokens,
            rope_cos,
            rope_sin,
            (cross_kv[0][i], cross_kv[1][i]),
            layer_k,
            layer_v,
            current_start,
        )
        new_ks.append(chunk_k)
        new_vs.append(chunk_v)
      new_k, new_v = jnp.stack(new_ks), jnp.stack(new_vs)

    # Head (float32).
    head_mod = self.scale_shift_table[...].astype(jnp.float32) + temb[:, None, :].astype(jnp.float32)
    shift, scale = head_mod[:, 0:1, :], head_mod[:, 1:2, :]
    x = self.proj_out(self.norm_out(x) * (1 + scale) + shift)

    # Unpatchify: [B, L, p_t * p_h * p_w * C] -> [B, C, F, H, W].
    x = x.reshape(batch, f, h, w, p_t, p_h, p_w, self.out_dim)
    x = jnp.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6)).reshape(batch, self.out_dim, frames, height, width)
    return x.astype(jnp.float32), (new_k, new_v)
