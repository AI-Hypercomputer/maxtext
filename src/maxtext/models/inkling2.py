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

"""Inkling 2 (looped MoE + Block AttnRes) decoder layers and building blocks."""

# pylint: disable=arguments-differ,too-many-positional-arguments

from __future__ import annotations

import math
from typing import Any, NamedTuple

from flax import nnx
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    Array,
    BATCH,
    BATCH_ATTN,
    Config,
    EMBED,
    HEAD,
    KV_HEAD,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import sharding
from maxtext.utils.sharding import create_sharding

# =============================================================================
# Core Math Helpers (Exact FP32 parity with HF `modeling_inkling2_loop.py`)
# =============================================================================


def _unweighted_rms_norm(x: Array, eps: float = 1e-6) -> Array:
  """Unweighted RMSNorm computed in float32, returning float32."""
  x32 = x.astype(jnp.float32)
  inv_rms = lax.rsqrt(jnp.mean(lax.square(x32), axis=-1, keepdims=True) + eps)
  return x32 * inv_rms


def _l2_clip(x: Array, threshold: float, eps: float = 1e-6) -> Array:
  """Clips per-vector L2 norm to `threshold` in float32, returning input dtype."""
  in_dtype = x.dtype
  x32 = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(lax.square(x32), axis=-1, keepdims=True))
  scale = jnp.minimum(1.0, threshold / (norm + eps))
  return (x32 * scale).astype(in_dtype)


def _log_scale(x: Array, positions: Array, n_floor: float, alpha: float) -> Array:
  """Position-dependent logarithmic scaling for full-attention layers.

  `tau(p) = 1 + alpha * log(max((p + 1) / n_floor, 1.0))`
  """
  in_dtype = x.dtype
  pos_f32 = (positions.astype(jnp.float32) + 1.0) / n_floor
  tau = 1.0 + alpha * jnp.log(jnp.maximum(pos_f32, 1.0))
  # x: [B, T, H, D], tau: [B, T] -> [B, T, 1, 1]
  return (x.astype(jnp.float32) * tau[..., None, None]).astype(in_dtype)


def _explicit_reshard(
    x: Array,
    logical_axes: tuple[str | None, ...],
    mesh: Mesh | None,
    shard_mode: ShardMode,
    debug_sharding: bool = False,
) -> Array:
  """Applies `maybe_shard_with_logical` when `mesh` is present."""
  if mesh is None:
    return x
  return sharding.maybe_shard_with_logical(
      x,
      logical_axes,
      mesh,
      shard_mode,
      debug_sharding=debug_sharding,
  )


def _dot_general_out_sharding(
    mesh: Mesh | None,
    shard_mode: ShardMode,
    logical_axes: tuple[str | None, ...],
):
  """Returns `NamedSharding` for `lax.dot_general` in explicit mode, else `None`."""
  if mesh is None or shard_mode != ShardMode.EXPLICIT:
    return None
  return create_sharding(mesh, logical_axes)


def _causal_shift_right(x: Array, lag: int, axis: int = 1) -> Array:
  """Shifts `x` right by `lag` steps along `axis`, zero-padding on the left."""
  if lag == 0:
    return x
  seq_len = x.shape[axis]
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (lag, 0)
  padded = jnp.pad(x, pad_width)
  return lax.slice_in_dim(padded, 0, seq_len, axis=axis)


def _extract_last_valid_conv_state(
    x_masked: Array,
    valid_mask: Array | None,
    state_len: int,
) -> Array:
  """Extracts the trailing `state_len` valid tokens of `x_masked` per batch row.

  Handles right-padded prefill sequences (`valid_mask = [1, 1, 1, 0, 0]`) as well
  as unpadded sequences (`valid_mask is None`) and prompts shorter than `state_len`.
  """
  if state_len == 0:
    return x_masked[:, :0, ...]
  pad_width = [(0, 0)] * x_masked.ndim
  pad_width[1] = (state_len, 0)
  padded = jnp.pad(x_masked, pad_width)
  if valid_mask is None:
    return padded[:, -state_len:, ...]
  # For right-padded sequence of n_valid tokens, valid tokens occupy [0..n_valid-1] in x_masked,
  # which are indices [state_len .. state_len + n_valid - 1] in padded; so the trailing
  # `state_len` valid tokens start at index `n_valid`.
  n_valid = jnp.sum(valid_mask.astype(jnp.int32), axis=1)  # [B]
  max_start = x_masked.shape[1]
  starts = jnp.clip(n_valid, 0, max_start)

  def _slice_row(row_padded, start_idx):
    return lax.dynamic_slice_in_dim(row_padded, start_idx, state_len, axis=0)

  return jax.vmap(_slice_row)(padded, starts)


# =============================================================================
# Block AttnRes Fixed Buffer & 48-Slot Autoregressive Decode Cache
# =============================================================================


class Inkling2AttnResBuffer(NamedTuple):
  """Fixed-shape AttnRes block buffer for `jax.lax.scan` carries.

  Attributes:
    blocks: Committed block states of shape `[B, T, max_blocks, H]` in `dtype`.
    count: Scalar `int32` count of committed blocks currently populated in `blocks`.
  """

  blocks: Array
  count: Array


def init_inkling2_attn_res_buffer(
    hidden_states: Array,
    max_blocks: int,
    mesh: Mesh | None = None,
    shard_mode: ShardMode = ShardMode.AUTO,
    debug_sharding: bool = False,
) -> Inkling2AttnResBuffer:
  """Allocates an empty fixed-shape `Inkling2AttnResBuffer`."""
  b, t, h = hidden_states.shape
  blocks = jnp.zeros((b, t, max_blocks, h), dtype=hidden_states.dtype)
  blocks = _explicit_reshard(
      blocks,
      (BATCH, "activation_norm_length", None, EMBED),
      mesh,
      shard_mode,
      debug_sharding=debug_sharding,
  )
  return Inkling2AttnResBuffer(blocks=blocks, count=jnp.asarray(0, dtype=jnp.int32))


def commit_block_state(
    block_states: Array | Inkling2AttnResBuffer,
    block_residual: Array,
) -> Array | Inkling2AttnResBuffer:
  """Commits `block_residual` (`[B, T, H]`) into dynamic or fixed `block_states`."""
  if isinstance(block_states, Inkling2AttnResBuffer):
    new_blocks = lax.dynamic_update_slice_in_dim(
        block_states.blocks,
        block_residual[:, :, None, :].astype(block_states.blocks.dtype),
        block_states.count,
        axis=2,
    )
    return Inkling2AttnResBuffer(blocks=new_blocks, count=block_states.count + jnp.asarray(1, dtype=jnp.int32))
  return jnp.concatenate([block_states, block_residual[:, :, None, :].astype(block_states.dtype)], axis=2)


class Inkling2LayerCacheSlice(NamedTuple):
  """Per-visit KV and 4x short-convolution cache tensors for one logical visit."""

  k: Array  # [B, max_len, Hkv, D]
  v: Array  # [B, max_len, Hkv, D]
  valid: Array  # [B, max_len] bool
  pos: Array  # [B, max_len] int32
  conv_k: Array  # [B, K - 1, Hkv, D] float32
  conv_v: Array  # [B, K - 1, Hkv, D] float32
  conv_post_attn: Array  # [B, K - 1, H] float32
  conv_post_mlp: Array  # [B, K - 1, H] float32


def total_logical_layers(config: Config) -> int:
  """Returns total logical layer visits (48 for Inkling 2 Mini)."""
  num_core = config.inkling2_core_end - config.inkling2_core_start
  return config.num_decoder_layers + (config.inkling2_num_passes - 1) * num_core


class Inkling2DecodeCache(nnx.Module):
  """48-slot per-visit KV + 4x short-convolution cache for PREFILL and AUTOREGRESSIVE modes.

  Stored strictly in `nnx.Cache` variables (never `nnx.Param`), and only instantiated
  when `model_mode != MODEL_MODE_TRAIN`.
  """

  def __init__(self, config: Config):
    num_visits = total_logical_layers(config)
    batch = max(
        1,
        int(
            getattr(config, "global_batch_size_to_eval_on", 0)
            or getattr(config, "global_batch_size_to_load", 0)
            or config.per_device_batch_size
        ),
    )
    max_len = int(config.max_target_length)
    hkv = int(config.num_kv_heads)
    d = int(config.head_dim)
    h = int(config.emb_dim)
    state_len = max(0, int(config.short_conv_kernel_size) - 1)
    dtype = config.dtype

    kv_axes = (None, "cache_batch", "cache_sequence", "cache_heads", "cache_kv")
    meta_axes = (None, "cache_batch", "cache_sequence")
    conv_kv_axes = (None, "cache_batch", None, "cache_heads", "cache_kv")
    conv_emb_axes = (None, "cache_batch", None, "activation_embed")

    self.cached_k = nnx.Cache(jnp.zeros((num_visits, batch, max_len, hkv, d), dtype=dtype), out_sharding=kv_axes)
    self.cached_v = nnx.Cache(jnp.zeros((num_visits, batch, max_len, hkv, d), dtype=dtype), out_sharding=kv_axes)
    self.cached_valid = nnx.Cache(jnp.zeros((num_visits, batch, max_len), dtype=jnp.bool_), out_sharding=meta_axes)
    self.cached_pos = nnx.Cache(jnp.zeros((num_visits, batch, max_len), dtype=jnp.int32), out_sharding=meta_axes)
    self.conv_k = nnx.Cache(
        jnp.zeros((num_visits, batch, state_len, hkv, d), dtype=jnp.float32), out_sharding=conv_kv_axes
    )
    self.conv_v = nnx.Cache(
        jnp.zeros((num_visits, batch, state_len, hkv, d), dtype=jnp.float32), out_sharding=conv_kv_axes
    )
    self.conv_post_attn = nnx.Cache(
        jnp.zeros((num_visits, batch, state_len, h), dtype=jnp.float32), out_sharding=conv_emb_axes
    )
    self.conv_post_mlp = nnx.Cache(
        jnp.zeros((num_visits, batch, state_len, h), dtype=jnp.float32), out_sharding=conv_emb_axes
    )

  def get_slot(self, visit_idx: int | Array, batch_size: int) -> Inkling2LayerCacheSlice:
    """Fetches a single visit's cache slice, slicing down to `batch_size` (raises if exceeding capacity)."""

    def _slice_b(arr):
      slot = arr[visit_idx]
      if batch_size > slot.shape[0]:
        raise ValueError(
            f"Inkling2DecodeCache batch capacity ({slot.shape[0]}) exceeded by batch_size={batch_size}. "
            "Increase per_device_batch_size / global_batch_size_to_eval_on to cover the decode batch."
        )
      return slot if slot.shape[0] == batch_size else slot[:batch_size]

    return Inkling2LayerCacheSlice(
        k=_slice_b(self.cached_k[...]),
        v=_slice_b(self.cached_v[...]),
        valid=_slice_b(self.cached_valid[...]),
        pos=_slice_b(self.cached_pos[...]),
        conv_k=_slice_b(self.conv_k[...]),
        conv_v=_slice_b(self.conv_v[...]),
        conv_post_attn=_slice_b(self.conv_post_attn[...]),
        conv_post_mlp=_slice_b(self.conv_post_mlp[...]),
    )

  def set_slot(self, visit_idx: int | Array, updated: Inkling2LayerCacheSlice) -> None:
    """Writes an updated visit slice back into `nnx.Cache` variables (raises if exceeding capacity)."""

    def _write(cache_var: nnx.Cache, val: Array) -> None:
      cur = cache_var[...]
      b = val.shape[0]
      if b > cur.shape[1]:
        raise ValueError(f"Inkling2DecodeCache batch capacity ({cur.shape[1]}) exceeded by update batch_size={b}.")
      cache_var[...] = cur.at[visit_idx, :b].set(val.astype(cur.dtype))

    _write(self.cached_k, updated.k)
    _write(self.cached_v, updated.v)
    _write(self.cached_valid, updated.valid)
    _write(self.cached_pos, updated.pos)
    _write(self.conv_k, updated.conv_k)
    _write(self.conv_v, updated.conv_v)
    _write(self.conv_post_attn, updated.conv_post_attn)
    _write(self.conv_post_mlp, updated.conv_post_mlp)


# =============================================================================
# Block AttnRes & Cross-Pass Message AttnRes / MessageGate
# =============================================================================


class Inkling2BlockAttnRes(nnx.Module):
  """Block Attention Residual pooling (normalizes both K and V, 1/sqrt(H) scaling, FP32)."""

  def __init__(
      self,
      emb_dim: int,
      num_queries: int = 1,
      eps: float = 1e-6,
      mesh: Mesh | None = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      debug_sharding: bool = False,
  ):
    self.emb_dim = emb_dim
    self.num_queries = num_queries
    self.eps = eps
    self.mesh = mesh
    self.shard_mode = shard_mode
    self.debug_sharding = debug_sharding

    if num_queries == 1:
      shape = (emb_dim,)
      axes = ("norm",)
    else:
      shape = (num_queries, emb_dim)
      axes = (None, "norm")
    self.query = nnx.Param(
        jnp.zeros(shape, dtype=jnp.float32),
        out_sharding=axes,
    )

  def __call__(
      self,
      block_states: Array | Inkling2AttnResBuffer,
      partial_block_states: Array,
  ) -> Array | tuple[Array, ...]:
    """Pools `[block_states, partial_block_states]` in FP32.

    Returns `[B, T, H]` (FP32) when `num_queries == 1`, or a tuple of `num_queries`
    `[B, T, H]` FP32 arrays when `num_queries > 1`.
    """
    if isinstance(block_states, Inkling2AttnResBuffer):
      blocks = block_states.blocks
      valid_count = block_states.count
      has_buffer_mask = True
    else:
      blocks = block_states
      valid_count = None
      has_buffer_mask = False

    stacked = jnp.concatenate(
        [blocks.astype(jnp.float32), partial_block_states[:, :, None, :].astype(jnp.float32)],
        axis=2,
    )  # [B, T, S, H]
    stacked_norm = _unweighted_rms_norm(stacked, eps=self.eps)

    q = self.query[...].astype(jnp.float32)
    if self.num_queries == 1:
      q = q[None, :]  # [1, H]

    # Contract over H -> [B, T, Q, S]
    # Note: in explicit mode, stacked_norm has last axis on `tensor` and q has last axis on `tensor`;
    # dot_general with explicit out_sharding cleanly reduces over `tensor`.
    logits_sharding = _dot_general_out_sharding(self.mesh, self.shard_mode, (BATCH, "activation_norm_length", None, None))
    logits = lax.dot_general(
        q,
        stacked_norm,
        dimension_numbers=(((1,), (3,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        out_sharding=(
            create_sharding(self.mesh, (None, BATCH, "activation_norm_length", None))
            if self.mesh is not None and self.shard_mode == ShardMode.EXPLICIT
            else None
        ),
    )  # [Q, B, T, S]
    logits = jnp.transpose(logits, (1, 2, 0, 3)) * (self.emb_dim**-0.5)  # [B, T, Q, S]
    if logits_sharding is not None:
      logits = _explicit_reshard(
          logits, (BATCH, "activation_norm_length", None, None), self.mesh, self.shard_mode, self.debug_sharding
      )

    if has_buffer_mask:
      max_blocks = blocks.shape[2]
      slot_idx = jnp.arange(max_blocks + 1, dtype=jnp.int32)
      # Committed slots [0 .. valid_count - 1] AND the final partial slot (index max_blocks) are valid.
      valid_slots = (slot_idx < valid_count) | (slot_idx == max_blocks)
      logits = jnp.where(valid_slots[None, None, None, :], logits, -jnp.inf)

    probs = jax.nn.softmax(logits, axis=-1)  # [B, T, Q, S] FP32

    # Weighted sum over S -> [B, T, Q, H] FP32
    out_sharding = _dot_general_out_sharding(self.mesh, self.shard_mode, (BATCH, "activation_norm_length", None, EMBED))
    out = lax.dot_general(
        probs,
        stacked_norm,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        precision=lax.Precision.HIGHEST,
        out_sharding=out_sharding,
    )  # [B, T, Q, H]

    if self.num_queries == 1:
      return out[:, :, 0, :]
    return tuple(out[:, :, i, :] for i in range(self.num_queries))


class Inkling2MessageAttnRes(nnx.Module):
  """Cross-pass Message Attention Residual pooling over previous pass's core blocks."""

  def __init__(
      self,
      num_core_layers: int,
      emb_dim: int,
      eps: float = 1e-6,
      mesh: Mesh | None = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      debug_sharding: bool = False,
  ):
    self.num_core_layers = num_core_layers
    self.emb_dim = emb_dim
    self.eps = eps
    self.mesh = mesh
    self.shard_mode = shard_mode
    self.debug_sharding = debug_sharding

    self.query = nnx.Param(
        jnp.zeros((num_core_layers, 4, emb_dim), dtype=jnp.float32),
        out_sharding=(None, None, "norm"),
    )

  def __call__(self, previous_block_states: Array, previous_hidden_states: Array) -> Array:
    """Computes `[B, T, num_core_layers, 4, H]` FP32 messages for all core layers at once."""
    stacked = jnp.concatenate(
        [previous_block_states.astype(jnp.float32), previous_hidden_states[:, :, None, :].astype(jnp.float32)],
        axis=2,
    )  # [B, T, S_prev, H]
    stacked_norm = _unweighted_rms_norm(stacked, eps=self.eps)

    q = self.query[...].astype(jnp.float32)  # [L_core, 4, H]
    q_flat = q.reshape(self.num_core_layers * 4, self.emb_dim)

    logits = lax.dot_general(
        q_flat,
        stacked_norm,
        dimension_numbers=(((1,), (3,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        out_sharding=(
            create_sharding(self.mesh, (None, BATCH, "activation_norm_length", None))
            if self.mesh is not None and self.shard_mode == ShardMode.EXPLICIT
            else None
        ),
    )  # [L_core*4, B, T, S_prev]
    logits = jnp.transpose(logits, (1, 2, 0, 3)) * (self.emb_dim**-0.5)  # [B, T, L_core*4, S_prev]
    probs = jax.nn.softmax(logits, axis=-1)

    out_sharding = _dot_general_out_sharding(self.mesh, self.shard_mode, (BATCH, "activation_norm_length", None, EMBED))
    out = lax.dot_general(
        probs,
        stacked_norm,
        dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
        precision=lax.Precision.HIGHEST,
        out_sharding=out_sharding,
    )  # [B, T, L_core*4, H]
    b, t, _, h = out.shape
    return out.reshape(b, t, self.num_core_layers, 4, h)


def _nd_zeros_init(key, shape, dtype, in_axis, out_axis):
  """Zero `NdInitializer` compatible with `linears.DenseGeneral`."""
  del key, in_axis, out_axis
  return jnp.zeros(shape, dtype=dtype)


class Inkling2MessageGate(nnx.Module):
  """Cross-pass low-rank message controller: `(1 - f) * dest + g * message` in FP32."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Quant | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.eps = config.normalization_layer_epsilon
    h = config.emb_dim
    rank = config.inkling2_message_gate_rank
    kernel_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")

    self.down_proj = linears.DenseGeneral(
        in_features_shape=2 * h,
        out_features_shape=rank,
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=(None, None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    # Kept strictly in FP32 matching HF `_keep_in_fp32_modules_strict`.
    # Zero-init output gates so multi-pass training (and 1-pass -> multi-pass warm-start) begins as an identity of 1-pass.
    self.forget_gate_proj = linears.DenseGeneral(
        in_features_shape=rank,
        out_features_shape=h,
        use_bias=False,
        kernel_init=_nd_zeros_init,
        kernel_axes=(None, "norm"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=None,
        rngs=rngs,
    )
    self.message_gate_proj = linears.DenseGeneral(
        in_features_shape=rank,
        out_features_shape=h,
        use_bias=False,
        kernel_init=_nd_zeros_init,
        kernel_axes=(None, "norm"),
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=None,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array, message: Array) -> Array:
    """Mixes `hidden_states` and `message` in FP32."""
    cfg = self.config
    dest32 = hidden_states.astype(jnp.float32)
    msg32 = message.astype(jnp.float32)

    norm_dest = _unweighted_rms_norm(dest32, eps=self.eps)
    norm_msg = _unweighted_rms_norm(msg32, eps=self.eps)
    if cfg.shard_mode == ShardMode.EXPLICIT:
      # Unshard embed dim before concatenating along axis=-1 so explicit sharding stays valid under TP > 1.
      norm_dest = _explicit_reshard(
          norm_dest, (BATCH, "activation_norm_length", None), self.mesh, cfg.shard_mode, cfg.debug_sharding
      )
      norm_msg = _explicit_reshard(
          norm_msg, (BATCH, "activation_norm_length", None), self.mesh, cfg.shard_mode, cfg.debug_sharding
      )
    u = jnp.concatenate([norm_dest, norm_msg], axis=-1).astype(cfg.dtype)

    down_out_sharding = (
        create_sharding(self.mesh, (BATCH, "activation_norm_length", None))
        if cfg.shard_mode == ShardMode.EXPLICIT
        else None
    )
    gate_out_sharding = (
        create_sharding(self.mesh, (BATCH, "activation_norm_length", EMBED))
        if cfg.shard_mode == ShardMode.EXPLICIT
        else None
    )
    z = self.down_proj(u, out_sharding=down_out_sharding).astype(jnp.float32)
    forget_gate = jnp.tanh(self.forget_gate_proj(z, out_sharding=gate_out_sharding))
    message_gate = jnp.tanh(self.message_gate_proj(z, out_sharding=gate_out_sharding))
    return (1.0 - forget_gate) * dest32 + message_gate * msg32


# =============================================================================
# Short Causal Convolutions (Static & Dynamic, Packed-Segment Exact + AR Cache)
# =============================================================================


class Inkling2ShortConvolution(nnx.Module):
  """Static depthwise causal short convolution with FP32 residual + `1/sqrt(K+1)` scale.

  Kernel layout: `(K, *feature_shape)` stored in FP32, matching PyTorch Conv1d tap order
  where `kernel[K - 1]` weights lag 0 (current token) and `kernel[0]` weights lag `K - 1`.
  """

  def __init__(
      self,
      feature_shape: tuple[int, ...],
      kernel_size: int,
      kernel_axes: tuple[str | None, ...],
  ):
    self.feature_shape = feature_shape
    self.kernel_size = kernel_size
    self.scale = 1.0 / math.sqrt(kernel_size + 1)
    self.kernel = nnx.Param(
        jnp.zeros((kernel_size,) + feature_shape, dtype=jnp.float32),
        out_sharding=(None,) + kernel_axes,
    )

  def __call__(
      self,
      x: Array,
      decoder_segment_ids: Array | None = None,
      conv_state: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[Array, Array | None]:
    """Applies causal short conv; returns `(out, next_conv_state)`."""
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    residual = x32
    k = self.kernel_size
    state_len = k - 1
    w = self.kernel[...].astype(jnp.float32)  # [K, *feature_shape]
    extra_dims = (1,) * len(self.feature_shape)

    valid_mask = None if decoder_segment_ids is None else (decoder_segment_ids != 0)
    x_masked = jnp.where(valid_mask.reshape(valid_mask.shape + extra_dims), x32, 0.0) if valid_mask is not None else x32

    next_conv_state = None
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and conv_state is not None:
      # seq_len == 1: prepend conv_state [B, K-1, *F] to x_masked [B, 1, *F] -> [B, K, *F]
      window = jnp.concatenate([conv_state.astype(jnp.float32), x_masked], axis=1)
      # window[:, 0] is lag K-1 (matches w[0]); window[:, K-1] is lag 0 (matches w[K-1]).
      y = jnp.sum(window * w[None, ...], axis=1, keepdims=True)
      next_conv_state = window[:, -state_len:, :] if state_len > 0 else conv_state
    else:
      if model_mode == MODEL_MODE_PREFILL and conv_state is not None:
        next_conv_state = _extract_last_valid_conv_state(x_masked, valid_mask, state_len)

      y = jnp.zeros_like(x32)
      for lag in range(k):
        x_lag = _causal_shift_right(x_masked, lag, axis=1)
        if decoder_segment_ids is not None:
          seg_lag = _causal_shift_right(decoder_segment_ids, lag, axis=1)
          tap_valid = (decoder_segment_ids != 0) & (seg_lag == decoder_segment_ids)
          x_lag = jnp.where(tap_valid.reshape(tap_valid.shape + extra_dims), x_lag, 0.0)
        # Lag 0 uses w[K - 1], lag K - 1 uses w[0].
        w_lag = w[k - 1 - lag]
        y = y + x_lag * w_lag[None, None, ...]

    out = ((y + residual) * self.scale).astype(in_dtype)
    return out, next_conv_state


class Inkling2DynamicShortConvolution(nnx.Module):
  """Dynamic per-KV-head causal short convolution for full-attention K and V.

  CRITICAL TAP ORDER: `static_w[0]` weights lag 0 (current token) and `static_w[K - 1]`
  weights lag `K - 1` (REVERSED relative to static `Inkling2ShortConvolution`), and the
  output scale is `1 / sqrt(2 * K + 1)` (= `1/3` for `K=4`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Quant | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    self.kernel_size = config.short_conv_kernel_size
    self.scale = 1.0 / math.sqrt(2 * self.kernel_size + 1)
    kernel_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")

    self.static_w = nnx.Param(
        jnp.zeros((self.kernel_size, self.num_kv_heads, self.head_dim), dtype=config.weight_dtype),
        out_sharding=(None, "kv_heads", "kv"),
    )
    self.weight_proj = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.kernel_size),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv_heads", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      dynamic_input: Array,
      decoder_segment_ids: Array | None = None,
      conv_state: Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[Array, Array | None]:
    """Applies dynamic causal short conv to `x` (`[B, T, Hkv, D]`)."""
    cfg = self.config
    in_dtype = x.dtype
    k = self.kernel_size
    state_len = k - 1

    wp_out_sharding = (
        create_sharding(self.mesh, (BATCH_ATTN, LENGTH, KV_HEAD, None)) if cfg.shard_mode == ShardMode.EXPLICIT else None
    )
    dyn_w = self.weight_proj(dynamic_input, out_sharding=wp_out_sharding).astype(jnp.float32)  # [B, T, Hkv, K]
    static_w = self.static_w[...].astype(jnp.float32)  # [K, Hkv, D]

    x32 = x.astype(jnp.float32)
    residual = x32
    valid_mask = None if decoder_segment_ids is None else (decoder_segment_ids != 0)
    x_masked = jnp.where(valid_mask[:, :, None, None], x32, 0.0) if valid_mask is not None else x32

    next_conv_state = None
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and conv_state is not None:
      # window[:, 0] is lag K-1; window[:, K-1] is lag 0.
      window = jnp.concatenate([conv_state.astype(jnp.float32), x_masked], axis=1)  # [B, K, Hkv, D]
      y = jnp.zeros_like(x32)
      for lag in range(k):
        # lag 0 sits at window[:, K - 1 - lag]
        x_lag = window[:, k - 1 - lag : k - lag, :, :]  # [B, 1, Hkv, D]
        w_lag = static_w[lag][None, None, :, :] + dyn_w[:, :, :, lag : lag + 1]  # [B, 1, Hkv, D]
        y = y + x_lag * w_lag
      next_conv_state = window[:, -state_len:, :, :] if state_len > 0 else conv_state
    else:
      if model_mode == MODEL_MODE_PREFILL and conv_state is not None:
        next_conv_state = _extract_last_valid_conv_state(x_masked, valid_mask, state_len)

      y = jnp.zeros_like(x32)
      for lag in range(k):
        x_lag = _causal_shift_right(x_masked, lag, axis=1)
        if decoder_segment_ids is not None:
          seg_lag = _causal_shift_right(decoder_segment_ids, lag, axis=1)
          tap_valid = (decoder_segment_ids != 0) & (seg_lag == decoder_segment_ids)
          x_lag = jnp.where(tap_valid[:, :, None, None], x_lag, 0.0)
        # Reversed order: static_w[0] and dyn_w[..., 0] weight lag 0.
        w_lag = static_w[lag][None, None, :, :] + dyn_w[:, :, :, lag : lag + 1]
        y = y + x_lag * w_lag

    out = ((y + residual) * self.scale).astype(in_dtype)
    return out, next_conv_state


# =============================================================================
# Inkling 2 Attention (NoPE + L2Clip + LogScale + RelPosBias + Per-Token Sinks)
# =============================================================================


class Inkling2Attention(nnx.Module):
  """Inkling 2 sliding-window or full-causal GQA attention with per-token sinks & B_rel."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      is_full_layer: bool,
      rngs: nnx.Rngs,
      quant: Quant | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.is_full_layer = is_full_layer
    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.num_kv_groups = self.num_query_heads // self.num_kv_heads
    self.head_dim = config.head_dim
    self.d_rel = config.inkling2_d_rel
    self.rel_extent = config.inkling2_rel_extent if is_full_layer else config.sliding_window_size
    self.sliding_window_size = None if is_full_layer else config.sliding_window_size
    self.scaling = self.head_dim**-0.5

    kernel_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")

    self.query = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_query_heads, self.head_dim),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "q_heads", "kv"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.key = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv_heads", "kv"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.value = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "kv_heads", "kv"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.rel_proj = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_query_heads, self.d_rel),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "q_heads", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.out = linears.DenseGeneral(
        in_features_shape=(self.num_query_heads, self.head_dim),
        out_features_shape=config.emb_dim,
        axis=(-2, -1),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("q_heads", "kv", "embed_attn"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.gate_attn = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=(self.num_query_heads, self.head_dim),
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "q_heads", "kv"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )
    self.sink_proj = linears.DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.num_query_heads,
        use_bias=False,
        kernel_init=kernel_init,
        kernel_axes=("embed_attn", "q_heads"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        quant=quant,
        rngs=rngs,
    )

    self.q_l2_clip_scale = nnx.Param(
        jnp.ones((self.num_query_heads, self.head_dim), dtype=config.weight_dtype),
        out_sharding=("q_heads", "kv"),
    )
    self.rel_l2_clip_scale = nnx.Param(
        jnp.ones((self.num_query_heads, self.d_rel), dtype=config.weight_dtype),
        out_sharding=("q_heads", None),
    )
    self.rel_logits_proj = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(rngs.params(), (self.d_rel, self.rel_extent), dtype=config.weight_dtype),
        out_sharding=(None, None),
    )

    if is_full_layer:
      self.k_sconv = Inkling2DynamicShortConvolution(config, mesh, rngs=rngs, quant=quant)
      self.v_sconv = Inkling2DynamicShortConvolution(config, mesh, rngs=rngs, quant=quant)
    else:
      self.k_sconv = Inkling2ShortConvolution(
          feature_shape=(self.num_kv_heads, self.head_dim),
          kernel_size=config.short_conv_kernel_size,
          kernel_axes=("kv_heads", "kv"),
      )
      self.v_sconv = Inkling2ShortConvolution(
          feature_shape=(self.num_kv_heads, self.head_dim),
          kernel_size=config.short_conv_kernel_size,
          kernel_axes=("kv_heads", "kv"),
      )

  def __call__(
      self,
      q_in: Array,
      k_in: Array,
      v_in: Array,
      decoder_segment_ids: Array | None,
      decoder_positions: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache_slice: Inkling2LayerCacheSlice | None = None,
  ) -> tuple[Array, tuple[Array, Array, Array, Array, Array | None, Array | None] | None]:
    """Runs Inkling 2 attention; returns `(out, updated_kv_conv_tuple_or_None)`."""
    cfg = self.config
    b, t_q, _ = q_in.shape
    if decoder_positions is None:
      if model_mode == MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            "Inkling2Attention in MODEL_MODE_AUTOREGRESSIVE requires explicit `decoder_positions` "
            "(omitting it would overwrite cache slot 0 every decode step)."
        )
      decoder_positions = jnp.broadcast_to(jnp.arange(t_q, dtype=jnp.int32)[None, :], (b, t_q))
    else:
      decoder_positions = decoder_positions.astype(jnp.int32)

    q_sharding = (
        create_sharding(self.mesh, (BATCH_ATTN, LENGTH, HEAD, None)) if cfg.shard_mode == ShardMode.EXPLICIT else None
    )
    kv_sharding = (
        create_sharding(self.mesh, (BATCH_ATTN, LENGTH, KV_HEAD, None)) if cfg.shard_mode == ShardMode.EXPLICIT else None
    )
    r_sharding = (
        create_sharding(self.mesh, (BATCH_ATTN, LENGTH, HEAD, None)) if cfg.shard_mode == ShardMode.EXPLICIT else None
    )
    sink_sharding = (
        create_sharding(self.mesh, (BATCH_ATTN, LENGTH, HEAD)) if cfg.shard_mode == ShardMode.EXPLICIT else None
    )

    # 1. Linear projections
    q = self.query(q_in, out_sharding=q_sharding)  # [B, T_q, Hq, D]
    k = self.key(k_in, out_sharding=kv_sharding)  # [B, T_q, Hkv, D]
    v = self.value(v_in, out_sharding=kv_sharding)  # [B, T_q, Hkv, D]
    r = self.rel_proj(q_in, out_sharding=r_sharding)  # [B, T_q, Hq, Dr]
    sinks = self.sink_proj(q_in, out_sharding=sink_sharding)  # [B, T_q, Hq]
    gate_attn = self.gate_attn(q_in, out_sharding=q_sharding)  # [B, T_q, Hq, D]

    q = checkpoint_name(q, "query_proj")
    k = checkpoint_name(k, "key_proj")
    v = checkpoint_name(v, "value_proj")

    # 2. K & V causal short convolutions
    conv_k_in = None if cache_slice is None else cache_slice.conv_k
    conv_v_in = None if cache_slice is None else cache_slice.conv_v
    if self.is_full_layer:
      k, next_conv_k = self.k_sconv(
          k, k_in, decoder_segment_ids=decoder_segment_ids, conv_state=conv_k_in, model_mode=model_mode
      )
      v, next_conv_v = self.v_sconv(
          v, v_in, decoder_segment_ids=decoder_segment_ids, conv_state=conv_v_in, model_mode=model_mode
      )
    else:
      k, next_conv_k = self.k_sconv(
          k, decoder_segment_ids=decoder_segment_ids, conv_state=conv_k_in, model_mode=model_mode
      )
      v, next_conv_v = self.v_sconv(
          v, decoder_segment_ids=decoder_segment_ids, conv_state=conv_v_in, model_mode=model_mode
      )

    # 3. L2Clip, V unweighted RMSNorm, and full-layer LogScale (matching HF rounding order)
    q_scaled = q.astype(jnp.float32) * self.q_l2_clip_scale[...].astype(jnp.float32)[None, None, :, :]
    q = _l2_clip(q_scaled, cfg.inkling2_qk_l2_clip_threshold).astype(cfg.dtype)
    if self.is_full_layer:
      q = _log_scale(q, decoder_positions, cfg.inkling2_log_scaling_n_floor, cfg.inkling2_log_scaling_alpha)

    k = _l2_clip(k, cfg.inkling2_qk_l2_clip_threshold).astype(cfg.dtype)
    v = _unweighted_rms_norm(v, eps=cfg.normalization_layer_epsilon).astype(cfg.dtype)

    r_scaled = r.astype(jnp.float32) * self.rel_l2_clip_scale[...].astype(jnp.float32)[None, None, :, :]
    r = _l2_clip(r_scaled, cfg.inkling2_rel_l2_clip_threshold).astype(cfg.dtype)
    if self.is_full_layer:
      r = _log_scale(r, decoder_positions, cfg.inkling2_log_scaling_n_floor, cfg.inkling2_log_scaling_alpha)

    # 4. KV cache update (PREFILL / AUTOREGRESSIVE) vs self-attention (TRAIN)
    updated_kv_conv = None
    if model_mode == MODEL_MODE_PREFILL and cache_slice is not None:
      max_len = cache_slice.k.shape[1]
      valid_q = jnp.ones((b, t_q), dtype=jnp.bool_) if decoder_segment_ids is None else (decoder_segment_ids != 0)
      k_cache = jnp.zeros((b, max_len, self.num_kv_heads, self.head_dim), dtype=cfg.dtype).at[:, :t_q].set(k)
      v_cache = jnp.zeros((b, max_len, self.num_kv_heads, self.head_dim), dtype=cfg.dtype).at[:, :t_q].set(v)
      valid_cache = jnp.zeros((b, max_len), dtype=jnp.bool_).at[:, :t_q].set(valid_q)
      pos_cache = jnp.zeros((b, max_len), dtype=jnp.int32).at[:, :t_q].set(decoder_positions)
      updated_kv_conv = (k_cache, v_cache, valid_cache, pos_cache, next_conv_k, next_conv_v)
      k_attn, v_attn = k, v
      kv_positions = decoder_positions
      kv_valid = valid_q
      is_ar_step = False
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and cache_slice is not None:
      # seq_len == 1: write into cache at position `decoder_positions[:, 0]`
      write_pos = decoder_positions[:, 0]  # [B]
      batch_idx = jnp.arange(b, dtype=jnp.int32)
      k_cache = cache_slice.k.at[batch_idx, write_pos].set(k[:, 0])
      v_cache = cache_slice.v.at[batch_idx, write_pos].set(v[:, 0])
      valid_cache = cache_slice.valid.at[batch_idx, write_pos].set(True)
      pos_cache = cache_slice.pos.at[batch_idx, write_pos].set(write_pos)
      updated_kv_conv = (k_cache, v_cache, valid_cache, pos_cache, next_conv_k, next_conv_v)
      k_attn, v_attn = k_cache, v_cache
      kv_positions = pos_cache
      kv_valid = valid_cache
      is_ar_step = True
    else:
      k_attn, v_attn = k, v
      kv_positions = decoder_positions
      kv_valid = None if decoder_segment_ids is None else (decoder_segment_ids != 0)
      is_ar_step = False

    t_kv = k_attn.shape[1]

    # 5. Build causal + sliding-window + segment/padding mask and relative position bias.
    delta = decoder_positions[:, :, None] - kv_positions[:, None, :]  # [B, T_q, T_kv]
    if is_ar_step:
      # In AR mode, attend to all valid cached entries with kv_pos <= q_pos
      attn_mask = kv_valid[:, None, :] & (delta >= 0)  # [B, T_q=1, T_kv]
    else:
      q_idx = jnp.arange(t_q, dtype=jnp.int32)[:, None]
      kv_idx = jnp.arange(t_kv, dtype=jnp.int32)[None, :]
      attn_mask = jnp.broadcast_to((q_idx >= kv_idx)[None, :, :], (b, t_q, t_kv))
      if decoder_segment_ids is not None:
        seg_q = decoder_segment_ids[:, :, None]
        seg_kv = decoder_segment_ids[:, None, :]
        attn_mask = attn_mask & (seg_q == seg_kv) & (seg_q != 0) & (seg_kv != 0)
    if self.sliding_window_size is not None:
      attn_mask = attn_mask & (delta >= 0) & (delta < self.sliding_window_size)

    # Relative position bias: rel_logits = R @ rel_logits_proj -> [B, Hq, T_q, rel_extent]
    rel_proj_w = self.rel_logits_proj[...].astype(cfg.dtype)
    rel_logits_sharding = _dot_general_out_sharding(self.mesh, cfg.shard_mode, (BATCH_ATTN, LENGTH, HEAD, None))
    rel_logits = lax.dot_general(
        r,
        rel_proj_w,
        dimension_numbers=(((3,), (0,)), ((), ())),
        precision=cfg.matmul_precision,
        out_sharding=rel_logits_sharding,
    )  # [B, T_q, Hq, Re]
    rel_logits = jnp.transpose(rel_logits, (0, 2, 1, 3))  # [B, Hq, T_q, Re]

    if not is_ar_step and t_q == t_kv and getattr(cfg, "inkling2_contiguous_segment_positions", True):
      # Static zero-gather/zero-scatter Toeplitz skew for unit-stride segments (`delta[i, j] == i - j`).
      # Reverses `rel_logits[..., :w]`, right-pads to width `T + 1`, flattens to `[B, Hq, T*(T+1)]`,
      # and slices `[w - 1 : w - 1 + T*T]` into `[B, Hq, T, T]`. Exact on every causal pair
      # (`0 <= i - j < w` reads `rel_logits[..., i, i - j]`; `i - j >= w` reads row `i - 1`'s
      # zero-padding region); strict upper-triangle entries (`i < j`) are masked to `-inf` below.
      w = min(t_q, self.rel_extent)
      rev = rel_logits[..., :w][..., ::-1]  # [B, Hq, T, W]
      padded = jnp.pad(rev, ((0, 0), (0, 0), (0, 0), (0, t_q + 1 - w)))  # [B, Hq, T, T + 1]
      flat = padded.reshape(b, self.num_query_heads, t_q * (t_q + 1))
      pos_bias = lax.slice_in_dim(flat, w - 1, w - 1 + t_q * t_q, axis=-1).reshape(b, self.num_query_heads, t_q, t_q)
    else:
      # General gather path for AR decode (`t_q == 1`) or non-unit-stride custom `decoder_positions`.
      valid_dist = (delta >= 0) & (delta < self.rel_extent)
      clamped_delta = jnp.clip(delta, 0, self.rel_extent - 1)
      gather_idx = jnp.broadcast_to(clamped_delta[:, None, :, :], (b, self.num_query_heads, t_q, t_kv))
      if cfg.shard_mode == ShardMode.EXPLICIT:
        gather_idx = _explicit_reshard(
            gather_idx, (BATCH_ATTN, HEAD, LENGTH, None), self.mesh, cfg.shard_mode, cfg.debug_sharding
        )
      pb = jnp.take_along_axis(rel_logits, gather_idx, axis=-1)
      vd = valid_dist[:, None, :, :]
      if cfg.shard_mode == ShardMode.EXPLICIT:
        vd = _explicit_reshard(vd, (BATCH_ATTN, None, LENGTH, None), self.mesh, cfg.shard_mode, cfg.debug_sharding)
      pos_bias = jnp.where(vd, pb, jnp.zeros((), pb.dtype))

    # 6. GQA dot-product attention scores + mask + per-token sink softmax
    # Reshape Q to [B, Hkv, G, T_q, D] so each shard keeps matching Hkv heads under TP > 1 with zero comms!
    q_g = jnp.transpose(q, (0, 2, 1, 3)).reshape(b, self.num_kv_heads, self.num_kv_groups, t_q, self.head_dim)
    k_g = jnp.transpose(k_attn, (0, 2, 1, 3))  # [B, Hkv, T_kv, D]
    v_g = jnp.transpose(v_attn, (0, 2, 1, 3))  # [B, Hkv, T_kv, D]

    qk_out_sharding = _dot_general_out_sharding(self.mesh, cfg.shard_mode, (BATCH_ATTN, KV_HEAD, None, LENGTH, None))
    attn_weights = lax.dot_general(
        q_g,
        k_g,
        dimension_numbers=(((4,), (3,)), ((0, 1), (0, 1))),
        precision=cfg.matmul_precision,
        out_sharding=qk_out_sharding,
    )  # [B, Hkv, G, T_q, T_kv]
    attn_weights = attn_weights.reshape(b, self.num_query_heads, t_q, t_kv) * jnp.asarray(
        self.scaling, attn_weights.dtype
    )
    attn_weights = attn_weights + pos_bias

    attn_weights_f32 = jnp.where(attn_mask[:, None, :, :], attn_weights.astype(jnp.float32), -jnp.inf)

    # Append per-token sink logit column [B, Hq, T_q, 1] (always finite -> zero NaNs even for padded queries!)
    sinks_f32 = jnp.transpose(sinks, (0, 2, 1))[..., None].astype(jnp.float32)  # [B, Hq, T_q, 1]
    combined = jnp.concatenate([attn_weights_f32, sinks_f32], axis=-1)
    probs = jax.nn.softmax(combined, axis=-1)[..., :-1].astype(cfg.dtype)  # [B, Hq, T_q, T_kv]

    probs_g = probs.reshape(b, self.num_kv_heads, self.num_kv_groups, t_q, t_kv)
    pv_out_sharding = _dot_general_out_sharding(self.mesh, cfg.shard_mode, (BATCH_ATTN, KV_HEAD, None, LENGTH, None))
    ctx = lax.dot_general(
        probs_g,
        v_g,
        dimension_numbers=(((4,), (2,)), ((0, 1), (0, 1))),
        precision=cfg.matmul_precision,
        out_sharding=pv_out_sharding,
    )  # [B, Hkv, G, T_q, D]
    ctx = jnp.transpose(ctx.reshape(b, self.num_query_heads, t_q, self.head_dim), (0, 2, 1, 3))  # [B, T_q, Hq, D]
    ctx = checkpoint_name(ctx, "context")

    # 7. Sigmoid output gate + output projection
    gated_ctx = (ctx.astype(jnp.float32) * jax.nn.sigmoid(gate_attn.astype(jnp.float32))).astype(cfg.dtype)
    out_sharding = (
        create_sharding(self.mesh, (BATCH, "activation_norm_length", EMBED))
        if cfg.shard_mode == ShardMode.EXPLICIT
        else None
    )
    out = self.out(gated_ctx, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, updated_kv_conv


# =============================================================================
# Inkling 2 Decoder Layer & Scannable Core Block
# =============================================================================


class Inkling2DecoderLayer(nnx.Module):
  """Single physical Inkling 2 decoder layer (zero dummy parameters)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      *,
      is_full_layer: bool,
      is_loop_layer: bool,
      has_attention_input: bool,
      is_block_boundary: bool,
      quant: Quant | None = None,
  ):
    self.config = config
    self.mesh = mesh
    self.is_full_layer = is_full_layer
    self.is_loop_layer = is_loop_layer
    self.has_attention_input = has_attention_input
    self.is_block_boundary = is_block_boundary

    # Block AttnRes
    if has_attention_input:
      self.attention_input = Inkling2BlockAttnRes(
          emb_dim=config.emb_dim,
          num_queries=3 if is_full_layer else 1,
          eps=config.normalization_layer_epsilon,
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
    else:
      self.attention_input = None

    self.mlp_input = Inkling2BlockAttnRes(
        emb_dim=config.emb_dim,
        num_queries=1,
        eps=config.normalization_layer_epsilon,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
    )

    # Pre-attention norms
    norm_kwargs = {
        "num_features": config.emb_dim,
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
        "kernel_axes": ("norm",),
        "epsilon": config.normalization_layer_epsilon,
        "shard_mode": config.shard_mode,
        "mesh": mesh,
        "rngs": rngs,
    }
    if is_full_layer:
      self.q_input_layernorm = RMSNorm(**norm_kwargs)
      self.k_input_layernorm = RMSNorm(**norm_kwargs)
      self.v_input_layernorm = RMSNorm(**norm_kwargs)
    else:
      self.input_layernorm = RMSNorm(**norm_kwargs)
    self.post_attention_layernorm = RMSNorm(**norm_kwargs)

    # Attention & post-attention short conv
    self.self_attention = Inkling2Attention(
        config=config,
        mesh=mesh,
        is_full_layer=is_full_layer,
        rngs=rngs,
        quant=quant,
    )
    self.post_attention_convolution = Inkling2ShortConvolution(
        feature_shape=(config.emb_dim,),
        kernel_size=config.short_conv_kernel_size,
        kernel_axes=("norm",),
    )

    # Sparse MoE (reuses MaxText RoutedMoE with SiTU + DeepSeek-V3 sigmoid routing + MoEBiasVar)
    kernel_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    self.mlp = moe.RoutedMoE(
        config=config,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        mesh=mesh,
        kernel_init=kernel_init,
        kernel_axes=("embed", None),
        intermediate_dim=config.moe_mlp_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        quant=quant,
        rngs=rngs,
    )
    self.post_mlp_convolution = Inkling2ShortConvolution(
        feature_shape=(config.emb_dim,),
        kernel_size=config.short_conv_kernel_size,
        kernel_axes=("norm",),
    )

    # Cross-pass message gate controllers (loop layers only -> zero dummy params on prelude/coda!)
    if is_loop_layer:
      self.q_message_gate = Inkling2MessageGate(config, mesh, rngs=rngs, quant=quant)
      self.k_message_gate = Inkling2MessageGate(config, mesh, rngs=rngs, quant=quant)
      self.v_message_gate = Inkling2MessageGate(config, mesh, rngs=rngs, quant=quant)
      self.mlp_message_gate = Inkling2MessageGate(config, mesh, rngs=rngs, quant=quant)

    self.out_sharding = (
        create_sharding(mesh, (BATCH, "activation_norm_length", EMBED))
        if config.shard_mode == ShardMode.EXPLICIT
        else None
    )

  def _norm(self, norm_layer: RMSNorm, x: Array) -> Array:
    """Applies `RMSNorm` with explicit out_sharding when enabled."""
    return norm_layer(x, out_sharding=self.out_sharding)

  def __call__(
      self,
      hidden_states: Array,
      block_states: Array | Inkling2AttnResBuffer,
      decoder_segment_ids: Array | None,
      decoder_positions: Array | None,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      layer_messages: Array | None = None,
      forced_routed_experts: Array | None = None,
      cache_slice: Inkling2LayerCacheSlice | None = None,
  ) -> tuple[Array, Array | Inkling2AttnResBuffer, Inkling2LayerCacheSlice | None, Array | None]:
    """Executes one logical visit of this decoder layer.

    Args:
      hidden_states: Current partial block state `[B, T, H]` in `config.dtype`.
      block_states: Committed block states (`[B, T, M, H]` or `Inkling2AttnResBuffer`).
      decoder_segment_ids: Optional `[B, T]` segment IDs (`0` = padding).
      decoder_positions: Optional `[B, T]` per-token positions.
      deterministic: Unused dropout flag (kept for standard decoder interface).
      model_mode: `MODEL_MODE_TRAIN`, `MODEL_MODE_PREFILL`, or `MODEL_MODE_AUTOREGRESSIVE`.
      layer_messages: Optional `[B, T, 4, H]` FP32 cross-pass messages (`None` on pass 0 / non-loop).
      forced_routed_experts: Optional `[B, T, top_k]` forced expert indices for router replay.
      cache_slice: Optional `Inkling2LayerCacheSlice` for PREFILL / AUTOREGRESSIVE.

    Returns:
      Tuple `(new_hidden_states, new_block_states, updated_cache_slice, selected_experts)`.
    """
    del deterministic
    cfg = self.config
    in_dtype = hidden_states.dtype
    hidden_states = _explicit_reshard(
        hidden_states, (BATCH, "activation_norm_length", EMBED), self.mesh, cfg.shard_mode, cfg.debug_sharding
    )
    hidden_states = checkpoint_name(hidden_states, "decoder_layer_input")
    block_residual = hidden_states

    # 1. Pre-attention Block AttnRes
    if self.attention_input is None:
      q_res = k_res = v_res = hidden_states
    elif self.is_full_layer:
      q_res, k_res, v_res = self.attention_input(block_states, hidden_states)
    else:
      q_res = k_res = v_res = self.attention_input(block_states, hidden_states)

    # 2. Pre-attention cross-pass message gates (passes >= 1 on loop layers)
    if self.is_loop_layer and layer_messages is not None:
      q_res = self.q_message_gate(q_res, layer_messages[:, :, 0, :])
      k_res = self.k_message_gate(k_res, layer_messages[:, :, 1, :])
      v_res = self.v_message_gate(v_res, layer_messages[:, :, 2, :])

    # 3. Pre-attention RMSNorms (output cast to in_dtype)
    if self.is_full_layer:
      q_in = self._norm(self.q_input_layernorm, q_res).astype(in_dtype)
      k_in = self._norm(self.k_input_layernorm, k_res).astype(in_dtype)
      v_in = self._norm(self.v_input_layernorm, v_res).astype(in_dtype)
    elif self.is_loop_layer and layer_messages is not None:
      # Distinct Q/K/V states after separate message gates, normalized through shared input_layernorm.
      q_in = self._norm(self.input_layernorm, q_res).astype(in_dtype)
      k_in = self._norm(self.input_layernorm, k_res).astype(in_dtype)
      v_in = self._norm(self.input_layernorm, v_res).astype(in_dtype)
    else:
      q_in = k_in = v_in = self._norm(self.input_layernorm, q_res).astype(in_dtype)

    # 4. Attention + post-attention short conv
    attn_out, updated_kv_conv = self.self_attention(
        q_in,
        k_in,
        v_in,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        model_mode=model_mode,
        cache_slice=cache_slice,
    )
    conv_pa_in = None if cache_slice is None else cache_slice.conv_post_attn
    attn_out, next_conv_pa = self.post_attention_convolution(
        attn_out,
        decoder_segment_ids=decoder_segment_ids,
        conv_state=conv_pa_in,
        model_mode=model_mode,
    )

    # 5. Block boundary commit & residual update (exact match with HF modeling_inkling2_loop.py)
    residual = attn_out.astype(jnp.float32)
    if self.is_block_boundary:
      block_states = commit_block_state(block_states, block_residual)
      # Partial state RESETS to `attn_out` (not `block_residual + attn_out`).
      partial = residual
    else:
      residual = block_residual.astype(jnp.float32) + residual
      partial = residual

    # 6. Pre-MLP Block AttnRes + optional MLP message gate + norm + RoutedMoE + post-MLP short conv
    mlp_res = self.mlp_input(block_states, partial)
    if self.is_loop_layer and layer_messages is not None:
      mlp_res = self.mlp_message_gate(mlp_res, layer_messages[:, :, 3, :])
    mlp_in = self._norm(self.post_attention_layernorm, mlp_res).astype(in_dtype)

    token_mask = moe.routing_token_mask(cfg, decoder_segment_ids, model_mode)
    mlp_out, lb_loss, bias_updates = self.mlp(
        mlp_in,
        out_sharding=self.out_sharding,
        forced_routed_experts=forced_routed_experts,
        token_mask=token_mask,
    )
    mlp_out = checkpoint_name(mlp_out, "mlpwo")
    selected_experts = None
    if cfg.return_routed_experts:
      gate_logits, pre_bias_logits = self.mlp.gate(mlp_in)
      _, selected_experts = self.mlp.get_topk(
          gate_logits,
          pre_bias_logits,
          forced_routed_experts=forced_routed_experts,
          token_mask=token_mask,
      )
    if lb_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", lb_loss)
    if bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", bias_updates)

    conv_pm_in = None if cache_slice is None else cache_slice.conv_post_mlp
    mlp_out, next_conv_pm = self.post_mlp_convolution(
        mlp_out,
        decoder_segment_ids=decoder_segment_ids,
        conv_state=conv_pm_in,
        model_mode=model_mode,
    )

    # 7. Final layer output cast back to `in_dtype` (matches HF `(residual + hidden_states).to(dtype)`)
    new_hidden_states = (residual + mlp_out.astype(jnp.float32)).astype(in_dtype)

    updated_cache = None
    if updated_kv_conv is not None:
      k_c, v_c, val_c, pos_c, ck_c, cv_c = updated_kv_conv
      updated_cache = Inkling2LayerCacheSlice(
          k=k_c,
          v=v_c,
          valid=val_c,
          pos=pos_c,
          conv_k=ck_c,
          conv_v=cv_c,
          conv_post_attn=next_conv_pa,
          conv_post_mlp=next_conv_pm,
      )
    return new_hidden_states, block_states, updated_cache, selected_experts


class Inkling2CoreScannableBlock(nnx.Module):
  """One `attn_res_block_size`-layer homogeneous core block (`B-1` sliding-loop + `1` full-loop).

  Scanned over `num_core_blocks = (core_end - core_start) // attn_res_block_size` when
  `scan_layers=True`, with zero dummy parameters and 1-level scan only.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: Quant | None,
      rngs: nnx.Rngs,
      *,
      remat_enabled: bool = False,
      remat_policy_fn: Any = None,
      prevent_cse: bool = True,
  ):
    del model_mode
    self.config = config
    self.mesh = mesh
    self.block_size = config.attn_res_block_size
    self.remat_enabled = remat_enabled
    self.remat_policy_fn = remat_policy_fn
    self.prevent_cse = prevent_cse

    for j in range(self.block_size):
      is_full = j == self.block_size - 1
      is_boundary = j == 0
      layer = Inkling2DecoderLayer(
          config=config,
          mesh=mesh,
          rngs=rngs,
          is_full_layer=is_full,
          is_loop_layer=True,
          has_attention_input=True,
          is_block_boundary=is_boundary,
          quant=quant,
      )
      setattr(self, f"sub_layer_{j}", layer)

  def _call_sub_layer(
      self,
      sub_idx: int,
      hidden_states: Array,
      block_states: Inkling2AttnResBuffer,
      decoder_segment_ids: Array | None,
      decoder_positions: Array | None,
      deterministic: bool,
      model_mode: str,
      layer_messages: Array | None,
      forced_routed_experts: Array | None,
      cache_slice: Inkling2LayerCacheSlice | None,
  ) -> tuple[Array, Inkling2AttnResBuffer, Inkling2LayerCacheSlice | None, Array | None]:
    """Calls `sub_layer_{sub_idx}`, wrapping in `jax.checkpoint` only when remat is enabled."""
    layer = getattr(self, f"sub_layer_{sub_idx}")
    if self.remat_enabled and model_mode == MODEL_MODE_TRAIN:
      graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
      carried_param_paths = {path for path, _ in nnx.to_flat_state(params)}

      def _pure_fn(p_in, r_in, h_in, b_in, msgs_in, fre_in):
        merged = nnx.merge(graphdef, p_in, r_in)
        h_out, b_out, c_out, exp_out = merged(
            h_in,
            b_in,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            layer_messages=msgs_in,
            forced_routed_experts=fre_in,
            cache_slice=cache_slice,
        )
        # Propagate newly materialized params (e.g. Qwix LoRA adapters) and sown intermediates
        # while keeping carried Param and MoEBiasVar read-only (avoids TraceContextError).
        _, updated_params, new_intermediates, _ = nnx.split(merged, nnx.Param, nnx.Intermediate, ...)
        new_params = nnx.from_flat_state(
            [(path, val) for path, val in nnx.to_flat_state(updated_params) if path not in carried_param_paths]
        )
        return (h_out, b_out, c_out, exp_out), (new_params, new_intermediates)

      ckpt_fn = jax.checkpoint(_pure_fn, policy=self.remat_policy_fn, prevent_cse=self.prevent_cse)
      (h_out, b_out, c_out, exp_out), (new_params, new_intermediates) = ckpt_fn(
          params, rest, hidden_states, block_states, layer_messages, forced_routed_experts
      )
      if list(new_params.flat_state()):
        nnx.update(layer, new_params)
      if list(new_intermediates.flat_state()):
        nnx.update(layer, new_intermediates)
      return h_out, b_out, c_out, exp_out

    return layer(
        hidden_states,
        block_states,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        layer_messages=layer_messages,
        forced_routed_experts=forced_routed_experts,
        cache_slice=cache_slice,
    )

  def __call__(
      self,
      carry: tuple[Array, Inkling2AttnResBuffer],
      decoder_segment_ids: Array | None,
      decoder_positions: Array | None,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      block_messages: Array | None = None,
      block_forced_routed_experts: Array | None = None,
      block_cache_slices: Inkling2LayerCacheSlice | None = None,
  ) -> tuple[tuple[Array, Inkling2AttnResBuffer], Inkling2LayerCacheSlice | None, Array | None]:
    """Runs all `block_size` sub-layers sequentially inside one scanned core block."""
    hidden_states, block_states = carry
    updated_caches = []
    block_experts = []
    for j in range(self.block_size):
      msg_j = None if block_messages is None else block_messages[:, :, j, :, :]
      fre_j = None if block_forced_routed_experts is None else block_forced_routed_experts[j]
      cache_j = None if block_cache_slices is None else jax.tree.map(lambda x, idx=j: x[idx], block_cache_slices)
      hidden_states, block_states, upd_j, exp_j = self._call_sub_layer(
          j,
          hidden_states,
          block_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          msg_j,
          fre_j,
          cache_j,
      )
      if upd_j is not None:
        updated_caches.append(upd_j)
      if exp_j is not None:
        block_experts.append(exp_j)
    stacked_cache = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *updated_caches) if updated_caches else None
    stacked_experts = jnp.stack(block_experts, axis=0) if block_experts else None
    return (hidden_states, block_states), stacked_cache, stacked_experts
