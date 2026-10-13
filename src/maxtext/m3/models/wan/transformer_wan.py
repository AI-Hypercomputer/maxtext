# Copyright 2023–2025 Google LLC
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

"""Wan Diffusion Transformer (DiT) architecture and blocks."""

import contextlib
from enum import Enum, auto
import functools
import math
from typing import Any, Dict, Optional, Tuple, Union

from flax import linen as nn
from flax import nnx
import jax
from jax import Array
from jax import checkpoint_policies as cp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

try:
  from tokamax.splash import splash_attention as tokamax_splash_base
  import custom_splash
except (ImportError, ModuleNotFoundError):
  tokamax_splash_base = None
  custom_splash = None

try:
  from maxdiffusion.kernels.splash_attention import splash_attention_mask as tokamax_splash_attention_mask
  from maxdiffusion.kernels.splash_attention import splash_attention_kernel as tokamax_splash_attention_kernel
  from maxdiffusion.kernels.splash_attention import ring_attention_kernel as tokamax_ring_attention_kernel
except (ImportError, ModuleNotFoundError):
  tokamax_splash_attention_mask = None
  tokamax_splash_attention_kernel = None
  tokamax_ring_attention_kernel = None

from maxtext.common import common_types
from maxtext.layers import quantizations
from maxtext.utils import max_logging
from .utils_wan import (
    BlockSizes,
    CONTEXT,
    CROSS_ATTN_HEAD,
    CROSS_ATTN_KV_LENGTH,
    CROSS_ATTN_Q_LENGTH,
    SELF_ATTN_HEAD,
    SELF_ATTN_KV_LENGTH,
    SELF_ATTN_Q_LENGTH,
    TpuType,
    get_activation,
    get_tpu_type,
    safe_getattr,
)

# Constants & type aliases
LOG2E = math.log2(math.e)
Array = common_types.Array
DType = common_types.DType
AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED
Quant = quantizations.AqtQuantization

INTERNAL_RING_AXIS = "ring"
INTERNAL_ULYSSES_AXIS = "ulysses"


# =============================================================================
# 1. Normalization Layers
# =============================================================================


class FP32LayerNorm(nnx.Module):
  """LayerNorm computed in float32 for numerical stability."""

  def __init__(self, rngs: nnx.Rngs, dim: int, eps: float, elementwise_affine: bool):
    self.layer_norm = nnx.LayerNorm(
        rngs=rngs,
        num_features=dim,
        epsilon=eps,
        use_bias=elementwise_affine,
        use_scale=elementwise_affine,
        param_dtype=jnp.float32,
        dtype=jnp.float32,
    )

  def __call__(self, inputs: jax.Array) -> jax.Array:
    origin_dtype = inputs.dtype
    return self.layer_norm(inputs.astype(dtype=jnp.float32)).astype(dtype=origin_dtype)


# =============================================================================
# 2. Attention Layers & Kernels
# =============================================================================


def _coerce_tokamax_block_sizes(block_sizes):
  """Return block sizes with use_fused_bwd_kernel=True, as Tokamax requires."""
  # Tokamax requires fused bwd; convert if needed.
  if getattr(block_sizes, "use_fused_bwd_kernel", False):
    return block_sizes

  # Fall back if some fields are missing.
  bq = block_sizes.block_q
  bkv = getattr(block_sizes, "block_kv", bq)
  bkv_compute = getattr(block_sizes, "block_kv_compute", bkv)
  bq_dkv = getattr(block_sizes, "block_q_dkv", bq)
  bkv_dkv = getattr(block_sizes, "block_kv_dkv", bkv)
  bkv_dkv_compute = getattr(block_sizes, "block_kv_dkv_compute", bkv_compute)
  return splash_attention_kernel.BlockSizes(
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkv_compute,
      block_q_dkv=bq_dkv,
      block_kv_dkv=bkv_dkv,
      block_kv_dkv_compute=bkv_dkv_compute,
      block_q_dq=None,
      block_kv_dq=None,
      use_fused_bwd_kernel=True,
  )


def _maybe_aqt_einsum(quant: Quant):
  return jnp.einsum if quant is None else quant.einsum()


def _check_attention_inputs(query: Array, key: Array, value: Array) -> None:
  """Check attention inputs."""

  assert key.ndim == value.ndim, "k, v must have same rank."
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
  assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
  assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
  assert query.shape[-1] == key.shape[-1], "q, k depths must match."


def _reshape_batch_dim_to_heads(tensor, heads):
  batch_size, seq_len, dim = tensor.shape
  head_size = heads
  tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  reshaped_tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _reshape_heads_to_batch_dim(tensor, heads):
  """Fold the heads dimension into the batch dimension."""
  if tensor.ndim == 3:
    batch_size, seq_len, dim = tensor.shape
    head_size = heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
  else:
    batch_size, head_size, seq_len, head_dim = tensor.shape
    reshaped_tensor = tensor.reshape(batch_size * head_size, seq_len, head_dim)
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _reshape_heads_to_head_dim(tensor):
  # takes a tensor of shape [b, h, s, d] and reshapes to [b, s, h * d]
  # This is used to transform the output of flash attention back into the format of other attention outputs
  b, h, _, d = tensor.shape
  tensor = jnp.transpose(tensor, axes=[0, 2, 1, 3])
  reshaped_tensor = jnp.reshape(tensor, (b, -1, h * d))
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _unflatten_heads(tensor, heads):
  # reshapes from [b, s, h * d] to [b, h, s, d] (input format to flash format)
  batch, seq, heads_and_dim_head = tensor.shape
  tensor = tensor.reshape(batch, seq, heads, heads_and_dim_head // heads)
  # Transpose to ('batch', 'heads', 'length', 'kv')
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _replace_mesh_axis(axis_spec, old_axis: str, new_axes: tuple[str, ...]):
  """Replace `old_axis` in a single PartitionSpec entry with `new_axes`."""
  if axis_spec == old_axis:
    return new_axes
  if isinstance(axis_spec, tuple):
    replacement = []
    for axis in axis_spec:
      if axis == old_axis:
        replacement.extend(new_axes)
      else:
        replacement.append(axis)
    return tuple(replacement)
  return axis_spec


def _replace_mesh_axis_names(axis_names, old_axis: str, new_axes: tuple[str, ...]):
  return jax.sharding.PartitionSpec(*(_replace_mesh_axis(axis_name, old_axis, new_axes) for axis_name in axis_names))


def _create_internal_ulysses_ring_mesh(
    mesh: Mesh,
    ring_shards: int,
    ulysses_shards: int,
    ring_axis: str = INTERNAL_RING_AXIS,
    ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
) -> Mesh:
  """Split the public context mesh axis into private ring and Ulysses axes."""
  mesh_axis_names = tuple(mesh.axis_names)
  context_axis_index = mesh_axis_names.index(CONTEXT)
  devices = mesh.devices
  new_shape = devices.shape[:context_axis_index] + (ring_shards, ulysses_shards) + devices.shape[context_axis_index + 1 :]
  new_axis_names = (
      mesh_axis_names[:context_axis_index] + (ring_axis, ulysses_axis) + mesh_axis_names[context_axis_index + 1 :]
  )
  return Mesh(devices.reshape(new_shape), new_axis_names)


def _reshape_data_for_flash(tensor, heads, num_context_shards=1):
  """Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.

  Pads seq_len to a multiple of flash_block_size, and ensures the resulting
  number of blocks is divisible by the number of shards.
  """
  if tensor.ndim != 4:
    tensor = _unflatten_heads(tensor, heads)

  org_seq_len = tensor.shape[2]

  # Pad sequence dimension so it is evenly divisible by the context mesh axis,
  # which shard_map requires.
  if num_context_shards <= 1:
    return tensor, org_seq_len
  rem = org_seq_len % num_context_shards
  if rem == 0:
    return tensor, org_seq_len
  pad_width = [(0, 0)] * tensor.ndim
  pad_width[2] = (0, num_context_shards - rem)
  return jnp.pad(tensor, pad_width), org_seq_len


def _pad_data_for_flash(tensor, heads, flash_block_size, num_shards: int = 1):
  """Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.

  Pads seq_len to a multiple of flash_block_size, and ensures the resulting
  number of blocks is divisible by the number of shards.
  """
  tensor, _ = _reshape_data_for_flash(tensor, heads)

  # Pad head_dim to 128 if less than that.
  kv_size = tensor.shape[-1]
  head_dim_pad = 0
  if kv_size < 128:
    head_dim_pad = 128 - kv_size

  # Pad seq_len with sharding constraints.
  seq_len = tensor.shape[2]

  # 1. First, pad seq_len to be a multiple of flash_block_size
  rem = seq_len % flash_block_size
  if rem != 0:
    seq_len_padded_pre = seq_len + (flash_block_size - rem)
  else:
    seq_len_padded_pre = seq_len

  # 2. Ensure num_blocks is divisible by num_shards
  num_blocks = seq_len_padded_pre // flash_block_size
  if num_blocks % num_shards != 0:
    num_blocks += num_shards - (num_blocks % num_shards)

  final_padded_len = num_blocks * flash_block_size
  seq_len_pad = final_padded_len - seq_len

  if kv_size < 128 or seq_len_pad != 0:
    npad = ((0, 0), (0, 0), (0, seq_len_pad), (0, head_dim_pad))
    tensor = jnp.pad(tensor, npad)

  return tensor, kv_size, seq_len


def _flash_sequence_length(tensor: Array) -> int:
  if tensor.ndim == 3:
    return tensor.shape[1]
  if tensor.ndim == 4:
    return tensor.shape[2]
  raise ValueError("Flash attention expects rank-3 or rank-4 inputs, got rank" f" {tensor.ndim}.")


def _select_flash_block_sizes(
    query: Array,
    key: Array,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype,
    attention_kernel: str,
    preserve_asymmetric_block_sizes: bool = False,
) -> BlockSizes:
  """Select Flash/Splash block sizes.

  Existing MaxDiffusion behavior is preserved by default. When
  preserve_asymmetric_block_sizes=True, explicitly configured block sizes are
  honored even when Q and KV have different sequence lengths; the existing
  padding path makes the tensors compatible with those block sizes.
  """
  query_seq_len = _flash_sequence_length(query)
  key_seq_len = _flash_sequence_length(key)

  q_max_block_size = 1024 if dtype == jnp.bfloat16 else 512

  if key_seq_len != query_seq_len:
    kv_max_block_size = ((key_seq_len + 127) // 128) * 128
  else:
    kv_max_block_size = q_max_block_size

  # Preserve the existing Tokamax conversion behavior.
  if flash_block_sizes is not None and not hasattr(flash_block_sizes, "use_fused_bwd_kernel"):
    flash_block_sizes = _coerce_tokamax_block_sizes(flash_block_sizes)

  # Existing self-attention behavior: configured values are returned unchanged.
  if flash_block_sizes is not None and key_seq_len == query_seq_len:
    if attention_kernel in ("tokamax_flash", "tokamax_ring"):
      return _coerce_tokamax_block_sizes(flash_block_sizes)
    return flash_block_sizes

  # NEW: opt-in behavior required by Klein KV-cache.
  #
  # Q and KV may have different sequence lengths, but _pad_data_for_flash()
  # pads each sequence independently to the configured block size. Therefore
  # block_q/block_kv do not need to divide the original sequence lengths.
  if preserve_asymmetric_block_sizes and flash_block_sizes is not None:
    if attention_kernel in ("tokamax_flash", "tokamax_ring"):
      return _coerce_tokamax_block_sizes(flash_block_sizes)
    return flash_block_sizes

  # Existing MaxDiffusion cross-attention behavior.
  block_size_q = flash_block_sizes.block_q if flash_block_sizes is not None else q_max_block_size

  use_tokamax = attention_kernel in ("tokamax_flash", "tokamax_ring")

  return splash_attention_kernel.BlockSizes(
      block_q=block_size_q,
      block_kv_compute=min(kv_max_block_size, key_seq_len),
      block_kv=min(kv_max_block_size, key_seq_len),
      block_q_dkv=block_size_q,
      block_kv_dkv=min(kv_max_block_size, key_seq_len),
      block_kv_dkv_compute=min(kv_max_block_size, query_seq_len),
      block_q_dq=None if use_tokamax else block_size_q,
      block_kv_dq=None if use_tokamax else min(kv_max_block_size, query_seq_len),
      use_fused_bwd_kernel=use_tokamax,
  )


def convert_to_tokamax_splash_config(
    block_sizes: BlockSizes,
    q_layout: Any = "HEAD_DIM_MINOR",
    k_layout: Any = "HEAD_DIM_MINOR",
    v_layout: Any = "HEAD_DIM_MINOR",
    residual_checkpoint_name: str | None = None,
    attn_logits_soft_cap: float | None = None,
    fuse_reciprocal: bool = True,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    max_logit_const: float | None = None,
    interpret: bool = False,
    dq_reduction_steps: int | None = None,
) -> Any:
  """Convert splash BlockSizes into a Tokamax SplashConfig."""
  assert block_sizes.use_fused_bwd_kernel, "Tokamax Splash attention only supports fused bwd kernel."
  return tokamax_splash_attention_kernel.SplashConfig(
      block_q=block_sizes.block_q,
      block_kv=block_sizes.block_kv,
      block_kv_compute=block_sizes.block_kv_compute,
      block_q_dkv=block_sizes.block_q_dkv,
      block_kv_dkv=block_sizes.block_kv_dkv,
      block_kv_dkv_compute=block_sizes.block_kv_dkv_compute,
      block_q_dq=None if block_sizes.use_fused_bwd_kernel else block_sizes.block_q_dq,
      block_kv_dq=None if block_sizes.use_fused_bwd_kernel else block_sizes.block_kv_dq,
      use_fused_bwd_kernel=block_sizes.use_fused_bwd_kernel,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      residual_checkpoint_name=residual_checkpoint_name,
      attn_logits_soft_cap=attn_logits_soft_cap,
      fuse_reciprocal=fuse_reciprocal,
      use_base2_exp=use_base2_exp,
      use_experimental_scheduler=use_experimental_scheduler,
      max_logit_const=max_logit_const,
      interpret=interpret,
      dq_reduction_steps=dq_reduction_steps,
  )


def _extract_custom_block_sizes(flash_block_sizes):
  """Pulls custom-kernel block sizes out of the (dict or BlockSizes-like) config.

  Mirrors the extraction used by the `ulysses_custom` path so the custom ring
  kernel honors the same `flash_block_sizes={...}` knobs.
  """
  bq = 4864
  bkv = 1024
  bkv_compute = 1024
  bkv_compute_in = 1024
  heads_per_tile = 1
  vmem_limit_bytes = None
  if flash_block_sizes is not None:
    if isinstance(flash_block_sizes, dict):
      get = flash_block_sizes.get
      bq = get("block_q", None) or bq
      bkv = get("block_kv", None) or bkv
      bkv_compute = get("block_kv_compute", None) or bkv_compute
      bkv_compute_in = get("block_kv_compute_in", None) or bkv_compute_in
      heads_per_tile = get("heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = get("vmem_limit_bytes", None) or vmem_limit_bytes
    else:
      bq = getattr(flash_block_sizes, "block_q", None) or bq
      bkv = getattr(flash_block_sizes, "block_kv", None) or bkv
      bkv_compute = getattr(flash_block_sizes, "block_kv_compute", None) or bkv_compute
      bkv_compute_in = getattr(flash_block_sizes, "block_kv_compute_in", None) or bkv_compute_in
      heads_per_tile = getattr(flash_block_sizes, "heads_per_tile", None) or heads_per_tile
      vmem_limit_bytes = getattr(flash_block_sizes, "vmem_limit_bytes", None) or vmem_limit_bytes
  # A BlockSizes object carries heads_per_tile=None when the config dict omitted
  # it; getattr then returns that None instead of the default, so coerce it back
  # to 1 (the custom-kernel default) to keep the `heads_per_tile > 1` guards safe.
  if heads_per_tile is None:
    heads_per_tile = 1
  return bq, bkv, bkv_compute, bkv_compute_in, heads_per_tile, vmem_limit_bytes


def _build_padding_segment_ids(
    query_seq_len: int,
    q_padded_len: int,
    key_seq_len: int,
    kv_padded_len: int,
    attention_mask: jax.Array | None,
    segment_ids_cls=splash_attention_kernel.SegmentIds,
):
  """Build splash segment ids that mask q/kv padding and the attention mask.

  Padding tokens get segment id 0, valid tokens 1. An optional attention_mask
  (batch, kv_len) is folded into the kv segment ids. Positions beyond an
  explicit mask are invalid, including sequence padding introduced before the
  local flash kernel. Shared by flash, Ulysses, and Ulysses+ring kernels.
  """
  q_indices = jax.lax.broadcasted_iota(jnp.int32, (q_padded_len,), 0)
  q_segment_ids = (q_indices < query_seq_len).astype(jnp.int32)

  kv_indices = jax.lax.broadcasted_iota(jnp.int32, (kv_padded_len,), 0)
  kv_segment_ids = (kv_indices < key_seq_len).astype(jnp.int32)

  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError("attention_mask must have shape [batch, kv_length], got" f" {attention_mask.shape}.")
    mask_len = min(key_seq_len, attention_mask.shape[1])
    kv_mask_for_batch = attention_mask[:, :mask_len].astype(jnp.int32)
    # An explicit mask is authoritative. This also masks sequence padding
    # introduced before shard_map to make the KV length context-divisible.
    if key_seq_len > mask_len:
      kv_mask_for_batch = jnp.concatenate(
          [
              kv_mask_for_batch,
              jnp.zeros((attention_mask.shape[0], key_seq_len - mask_len), jnp.int32),
          ],
          axis=1,
      )
    # Tokens past key_seq_len are padding.
    if kv_padded_len > key_seq_len:
      kv_mask_for_batch = jnp.concatenate(
          [
              kv_mask_for_batch,
              jnp.zeros(
                  (attention_mask.shape[0], kv_padded_len - key_seq_len),
                  jnp.int32,
              ),
          ],
          axis=1,
      )
    kv_segment_ids = (kv_segment_ids[None, :] * kv_mask_for_batch).astype(jnp.int32)
    q_segment_ids = jnp.broadcast_to(q_segment_ids[None, :], (attention_mask.shape[0], q_padded_len))

  return segment_ids_cls(q=q_segment_ids, kv=kv_segment_ids)


def _prepare_attention_mask_for_shard_map(
    attention_mask: jax.Array | None,
    batch_size: int,
    padded_kv_len: int,
) -> jax.Array | None:
  """Broadcasts and pads a canonical keep mask before entering shard_map."""
  if attention_mask is None:
    return None
  if attention_mask.ndim != 2:
    raise ValueError("attention_mask must have shape [batch, kv_length], got" f" {attention_mask.shape}.")
  if attention_mask.shape[0] == 1 and batch_size != 1:
    attention_mask = jnp.broadcast_to(attention_mask, (batch_size, attention_mask.shape[1]))
  elif attention_mask.shape[0] != batch_size:
    raise ValueError(
        "attention_mask batch dimension must be 1 or match the attention batch"
        f" ({batch_size}), got {attention_mask.shape[0]}."
    )

  attention_mask = attention_mask.astype(jnp.bool_)

  # `attention_mask.shape[1]` represents the true original sequence length of the KV states,
  # while `padded_kv_len` represents the padded sequence length required for XLA compilation/divisibility.
  # We pad the mask with `False` (masked out) to cover the padded dummy tokens.
  if attention_mask.shape[1] < padded_kv_len:
    attention_mask = jnp.pad(
        attention_mask,
        ((0, 0), (0, padded_kv_len - attention_mask.shape[1])),
        constant_values=False,
    )
  elif attention_mask.shape[1] > padded_kv_len:
    # If the user-provided mask exceeds the required length, we truncate it.
    attention_mask = attention_mask[:, :padded_kv_len]
  return attention_mask


def _mesh_axis_in_spec(axis_spec, mesh_axis: str) -> bool:
  """Returns whether a logical-to-mesh axis entry contains mesh_axis."""
  if isinstance(axis_spec, tuple):
    return mesh_axis in axis_spec
  return axis_spec == mesh_axis


def _ulysses_head_chunk_ranges(num_heads: int, ulysses_shards: int, num_chunks: int):
  """Build head-axis ranges for chunked Ulysses all-to-all.

  The Ulysses all-to-all splits each local chunk's head axis over
  `ulysses_shards`, so every returned range length is a multiple of
  `ulysses_shards`. When `num_chunks` does not evenly divide the number of
  Ulysses-sized head groups, earlier chunks get the floor-sized range and the
  final chunk carries the remainder.

  Returns:
    A list of `(start, end)` half-open ranges over the head axis. Concatenating
    tensors sliced with these ranges along the head axis restores the original
    head layout. For `num_chunks <= 1`, returns `[(0, num_heads)]`, which is the
    unchunked all-to-all path.
  """
  if num_chunks <= 1:
    return [(0, num_heads)]
  if num_heads % ulysses_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of heads to be divisible by the"
        f" Ulysses shard count, got heads={num_heads} and"
        f" ulysses_shards={ulysses_shards}."
    )

  head_groups = num_heads // ulysses_shards
  num_chunks = min(num_chunks, head_groups)
  regular_groups_per_chunk = max(1, head_groups // num_chunks)

  ranges = []
  start_group = 0
  for chunk_idx in range(num_chunks):
    end_group = head_groups if chunk_idx == num_chunks - 1 else min(start_group + regular_groups_per_chunk, head_groups)
    if start_group >= end_group:
      break
    ranges.append((start_group * ulysses_shards, end_group * ulysses_shards))
    start_group = end_group
  return ranges


def _run_chunked_ulysses_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    num_heads: int,
    ulysses_shards: int,
    ulysses_attention_chunks: int,
    attention_fn,
) -> jax.Array:
  """Runs Ulysses attention chunked or unchunked along the head axis.

  Splits the attention compute and communication into head-group chunks so XLA
  can overlap communication and compute.

  Args:
    query: The query tensor, [B, H, S, D].
    key: The key tensor, [B, H, S, D].
    value: The value tensor, [B, H, S, D].
    num_heads: The number of heads in query.
    ulysses_shards: The Ulysses/context shard count.
    ulysses_attention_chunks: Number of head-group chunks to split into.
    attention_fn: The local Ulysses attention function to call on each chunk,
      taking (query, key, value) and returning the attention output.

  Returns:
    The concatenated attention output tensor.
  """
  head_chunk_ranges = _ulysses_head_chunk_ranges(num_heads, ulysses_shards, ulysses_attention_chunks)
  if len(head_chunk_ranges) > 1:
    chunk_outputs = [
        attention_fn(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
        )
        for start, end in head_chunk_ranges
    ]
    return jnp.concatenate(chunk_outputs, axis=1)
  else:
    return attention_fn(query, key, value)


def _tpu_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    attention_kernel: str = "flash",
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    is_causal: bool = False,
    preserve_asymmetric_block_sizes: bool = False,
) -> jax.Array:
  """TPU Flash Attention"""

  num_context_shards = mesh.shape[CONTEXT] if CONTEXT in mesh.shape else 1
  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and attention_kernel == "tokamax_ring_custom":
    raise NotImplementedError("tokamax_ring_custom does not support attention_mask.")
  block_sizes = _select_flash_block_sizes(
      query,
      key,
      flash_block_sizes,
      dtype,
      attention_kernel,
      preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
  )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))

  def wrap_flash_attention(query, key, value, attention_mask):
    if attention_kernel == "tokamax_ring_custom":
      # Ring attention backed by the custom dense splash kernel. q stays local,
      # k/v rotate over the "context" axis (handled inside the ring kernel).
      (
          bq,
          bkv,
          bkv_compute,
          bkv_compute_in,
          heads_per_tile,
          vmem_limit_bytes,
      ) = _extract_custom_block_sizes(flash_block_sizes)
      if heads_per_tile > 1:
        raise NotImplementedError("tokamax_ring_custom currently supports heads_per_tile == 1 only.")
      query_local = query * LOG2E if use_base2_exp else query
      query_local, kv_size, query_seq_len = _pad_data_for_flash(query_local, heads, bq)
      key_local, _, key_seq_len = _pad_data_for_flash(key, heads, bkv)
      value_local, _, _ = _pad_data_for_flash(value, heads, bkv)

      bsizes = custom_splash._BlockSizes(  # pylint: disable=protected-access
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkv_compute,
          block_kv_compute_in=bkv_compute_in,
      )
      ring_kernel = tokamax_ring_attention_kernel.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          ring_axis="context",
      )
      vmapped_ring = jax.vmap(ring_kernel, in_axes=(0, 0, 0))
      attention_output = vmapped_ring(query_local, key_local, value_local)
      return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    uses_fused_kernel = block_sizes.use_fused_bwd_kernel
    block_q_sizes = (
        block_sizes.block_q,
        block_sizes.block_q_dkv,
    )
    block_kv_sizes = (
        block_sizes.block_kv,
        block_sizes.block_kv_dkv,
    )
    if uses_fused_kernel:
      block_q_sizes += (block_sizes.block_q_dkv,)
      block_kv_sizes += (block_sizes.block_kv_dkv,)
    else:
      block_q_sizes += (block_sizes.block_q_dq,)
      block_kv_sizes += (block_sizes.block_kv_dq,)

    block_q = max(*block_q_sizes)
    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)

    block_kv = max(*block_kv_sizes)
    key, _, key_seq_len = _pad_data_for_flash(key, heads, block_kv)
    value, _, _ = _pad_data_for_flash(value, heads, block_kv)

    if is_causal:
      mask = splash_attention_mask.CausalMask((query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])
    else:
      mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

    segment_ids_cls = (
        tokamax_splash_base.SegmentIds if attention_kernel == "tokamax_ring" else splash_attention_kernel.SegmentIds
    )
    segment_ids = _build_padding_segment_ids(
        query_seq_len,
        query.shape[2],
        key_seq_len,
        key.shape[2],
        attention_mask,
        segment_ids_cls,
    )

    # make_splash_mha is wrapped around shardmap and seq and head is already
    # sharded based on in_specs, therefore setting head_shards=1 and q_seq_shards=1.
    if attention_kernel == "tokamax_flash":
      if is_causal:
        mask = tokamax_splash_attention_mask.CausalMask(
            (query.shape[2], key.shape[2]),
        )
      else:
        mask = tokamax_splash_attention_mask.FullMask(
            _shape=(query.shape[2], key.shape[2]),
        )
      splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
          mask=mask,
          q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
          config=convert_to_tokamax_splash_config(
              block_sizes,
              residual_checkpoint_name=residual_checkpoint_name,
              use_base2_exp=use_base2_exp,
              use_experimental_scheduler=use_experimental_scheduler,
          ),
          save_residuals=False,
      )
    elif attention_kernel == "tokamax_ring":
      mask = tokamax_splash_attention_mask.FullMask(
          _shape=(query.shape[2], key.shape[2]),
      )
      splash_kernel = tokamax_ring_attention_kernel.make_ring_attention(
          mask=mask,
          is_mqa=False,
          config=convert_to_tokamax_splash_config(
              block_sizes,
              residual_checkpoint_name=residual_checkpoint_name,
              use_base2_exp=use_base2_exp,
              use_experimental_scheduler=use_experimental_scheduler,
          ),
          save_residuals=False,
          ring_axis=CONTEXT,
          # Padding-only IDs are identical on each shard. Explicit masks differ
          # by KV shard and must rotate together with K/V.
          rotate_segment_ids=attention_mask is not None,
      )
    else:
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,  # the sizes of the axis is sharding over heads
          q_seq_shards=1,  # the sizes of the axis is sharding over seq_len
          block_sizes=block_sizes,
          save_residuals="ring" in attention_kernel,
          residual_checkpoint_name=residual_checkpoint_name,
      )

    segment_ids_in_axes = 0 if attention_mask is not None else None
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))

    if not mask_padding_tokens and attention_mask is None:
      segment_ids = None
    if attention_kernel in ["flash", "tokamax_flash", "tokamax_ring"]:
      attention_output = vmapped_splash(query, key, value, segment_ids)
    else:
      if num_context_shards > 1:
        out, (lse,) = vmapped_splash(query, key, value, segment_ids)
        m = lse.astype(jnp.float32)
        l = jnp.exp(lse - m)
        o = out.astype(jnp.float32) * l[..., None]

        perm = [(j, (j + 1) % num_context_shards) for j in range(num_context_shards)]

        k1 = jax.lax.ppermute(key, axis_name=CONTEXT, perm=perm)
        v1 = jax.lax.ppermute(value, axis_name=CONTEXT, perm=perm)

        def ring_scan_body(carry, _):
          m, l, o, k_current, v_current = carry
          k_next = jax.lax.ppermute(k_current, axis_name=CONTEXT, perm=perm)
          v_next = jax.lax.ppermute(v_current, axis_name=CONTEXT, perm=perm)

          out_chunk, (lse_chunk,) = vmapped_splash(query, k_current, v_current, segment_ids)

          m_chunk = lse_chunk.astype(jnp.float32)
          m_old = m
          m = jnp.maximum(m_old, m_chunk)

          exp_m_diff = jnp.exp(m_old - m)
          exp_m_chunk_diff = jnp.exp(m_chunk - m)

          l = l * exp_m_diff + jnp.exp(lse_chunk - m)
          o = o * exp_m_diff[..., None]
          o += exp_m_chunk_diff[..., None] * out_chunk.astype(jnp.float32)

          # Return the updated state for the next iteration
          return (m, l, o, k_next, v_next), None

        initial_carry = (m, l, o, k1, v1)
        (_, l_final, o_final, _, _), _ = jax.lax.scan(ring_scan_body, initial_carry, None, length=num_context_shards - 1)

        attention_output = o_final / l_final[..., None]
      else:
        raise ValueError("ring attention requires context > 1")
    return attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

  data_dim = mesh.shape["data"] if "data" in mesh.shape else 1
  fsdp_dim = mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1
  devices_in_batch_sharding = data_dim * fsdp_dim
  # This warning might show up when doing model eval for example, when calculating model flops
  # and that is expected.
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in"
        f" data and fsdp axis, batch dimension: {query.shape[0]},"
        f" devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_flash_attention = jax.shard_map(
        lambda q, k, v: wrap_flash_attention(q, k, v, None),
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=q_axis_names,
        check_vma=False,
    )
    x = sharded_flash_attention(query, key, value)
  else:
    sharded_flash_attention = jax.shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names, mask_axis_names),
        out_specs=q_axis_names,
        check_vma=False,
    )
    x = sharded_flash_attention(query, key, value, attention_mask)
  # Trim back to original sequence length after context-axis padding.
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)

  return x


# ---------------------------------------------------------------------------
# Ulysses sequence-parallel attention
# ---------------------------------------------------------------------------


def _ulysses_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    use_custom_kernel: bool = False,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    use_fixed_m: bool = False,
    ulysses_attention_chunks: int = 1,
    preserve_asymmetric_block_sizes: bool = False,
) -> jax.Array:
  """Ulysses sequence-parallel attention.

  Tensors arrive sequence-sharded on the context axis.  Inside a shard_map the
  all-to-all collectives trade sequence shards for head shards, run local
  splash attention on the full sequence with a subset of heads, then
  all-to-all back.
  """
  axis_name = CONTEXT
  num_shards = mesh.shape[axis_name]

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_shards)
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  if attention_mask is not None and use_custom_kernel:
    raise NotImplementedError(
        "The custom dense splash kernel (use_custom_kernel) does not support"
        " attention_mask (it only handles padding via orig_seq_len); got a"
        " non-None attention_mask."
    )
  num_heads = query.shape[1]
  # Ulysses only redistributes existing heads across the context mesh; unlike
  # the earlier draft, we fail fast instead of padding synthetic heads.
  if num_heads % num_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of heads to be divisible by the"
        f" context shard count, got heads={num_heads} and"
        f" context_shards={num_shards}."
    )

  if not use_custom_kernel:
    block_sizes = _select_flash_block_sizes(
        query,
        key,
        flash_block_sizes,
        dtype,
        "flash",
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))
  mask_needs_ulysses_gather = _mesh_axis_in_spec(kv_axis_names[2], axis_name)

  def wrap_ulysses_attention(query, key, value, attention_mask):
    # Swap sharding: each device gives up a slice of heads and gathers
    # a slice of sequence, so the local kernel sees the full sequence.
    query = jax.lax.all_to_all(query, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    key = jax.lax.all_to_all(key, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    value = jax.lax.all_to_all(value, axis_name=axis_name, split_axis=1, concat_axis=2, tiled=True)
    if attention_mask is not None and mask_needs_ulysses_gather:
      attention_mask = jax.lax.all_gather(attention_mask, axis_name=axis_name, axis=1, tiled=True)

    if use_custom_kernel:
      if attention_mask is not None:
        raise NotImplementedError(
            "The custom dense splash kernel (use_custom_kernel) does not"
            " support attention_mask (it only handles padding via"
            " orig_seq_len); got a non-None attention_mask."
        )
      bq, bkv, bkv_compute, bkv_compute_in, heads_per_tile, vmem_limit_bytes = _extract_custom_block_sizes(
          flash_block_sizes
      )

      if use_base2_exp:
        query = query * LOG2E

      if use_fixed_m:
        # k-smoothing (output-invariant): subtracting the per-row key mean
        # forces every logit row to have mean 0, hence row-max >= 0 — the
        # precondition that keeps the fixed-m Cauchy-Schwarz bound flush-free.
        key = key - jnp.mean(key, axis=2, keepdims=True)

      query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, bq)
      key, _, key_seq_len = _pad_data_for_flash(key, heads, bkv)
      value, _, _ = _pad_data_for_flash(value, heads, bkv)

      mk_arr = None
      if use_fixed_m:
        # Per-(local-)head Cauchy-Schwarz inputs over the (batch, seq) slice;
        # padded rows have zero norm and never raise the max. mk[0] feeds the
        # in-kernel per-query bound, mk[1] flags heads within the no-flush gate.
        qf = query.astype(jnp.float32)
        kf = key.astype(jnp.float32)
        qn_max = jnp.sqrt((qf * qf).sum(-1)).max(axis=(0, 2))  # (local_heads,)
        mk_h = jnp.sqrt((kf * kf).sum(-1)).max(axis=(0, 2))  # (local_heads,)
        # pylint: disable-next=protected-access
        fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
        mk_arr = jnp.stack([mk_h, fixed_ok])  # (2, local_heads)

      bsizes = custom_splash._BlockSizes(  # pylint: disable=protected-access
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkv_compute,
          block_kv_compute_in=bkv_compute_in,
      )

      splash_kernel = custom_splash.make_splash_mha(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          heads_per_tile=heads_per_tile,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          use_fixed_m=use_fixed_m,
      )

      if use_fixed_m:
        vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))
        attention_output = vmapped_splash(query, key, value, mk_arr)
      else:
        vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0))
        attention_output = vmapped_splash(query, key, value)
      attention_output = jnp.swapaxes(attention_output, 2, 3)
      attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)
    else:
      # Run the same local splash kernel as standard TPU flash attention, but now
      # on full-sequence / fewer-heads tensors produced by the all-to-all above.
      # block_sizes is always set above when use_custom_kernel is False.
      uses_fused_kernel = block_sizes.use_fused_bwd_kernel  # pylint: disable=possibly-used-before-assignment
      block_q_sizes = (block_sizes.block_q, block_sizes.block_q_dkv)
      block_kv_sizes = (block_sizes.block_kv, block_sizes.block_kv_dkv)
      if uses_fused_kernel:
        block_q_sizes += (block_sizes.block_q_dkv,)
        block_kv_sizes += (block_sizes.block_kv_dkv,)
      else:
        block_q_sizes += (block_sizes.block_q_dq,)
        block_kv_sizes += (block_sizes.block_kv_dq,)

      block_q = max(*block_q_sizes)
      query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)
      block_kv = max(*block_kv_sizes)
      key, _, key_seq_len = _pad_data_for_flash(key, heads, block_kv)
      value, _, _ = _pad_data_for_flash(value, heads, block_kv)

      mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

      segment_ids = _build_padding_segment_ids(
          query_seq_len,
          query.shape[2],
          key_seq_len,
          key.shape[2],
          attention_mask,
      )
      if not mask_padding_tokens and attention_mask is None:
        segment_ids = None

      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=block_sizes,
          save_residuals=False,
          residual_checkpoint_name=residual_checkpoint_name,
      )
      segment_ids_in_axes = 0 if attention_mask is not None else None
      vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
      attention_output = vmapped_splash(query, key, value, segment_ids)
      attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
    attention_output = jax.lax.all_to_all(
        attention_output,
        axis_name=axis_name,
        split_axis=2,
        concat_axis=1,
        tiled=True,
    )
    return attention_output

  devices_in_batch_sharding = mesh.shape["data"] * (mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1)
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in"
        f" data and fsdp axis, batch dimension: {query.shape[0]},"
        f" devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  # Fold the (CFG) batch into the heads axis around the Ulysses exchange.
  # Each (batch, head) pair is an independent attention problem, so
  # [B, H, S, D] -> [1, B*H, S, D] is mathematically identity — but it makes
  # XLA compile the attention path as the batch=1 case. At batch=2 XLA
  # otherwise places the size-2 batch in the tile sublanes ({3,0,1,2:T(2,128)}
  # instead of T(8,128)) which quadruples the cost of every op touching the
  # a2a tensors inside the scanned layers (measured 7.0 -> expected ~3.5
  # s/step at 720p 81f cp8 CFG).
  batch = query.shape[0]
  # Only foldable when nothing else shards the batch: with data/fsdp > 1 the
  # shard_map still maps axis 0 onto those mesh axes, and a folded (size-1)
  # batch is not divisible by them. Those configs are already batch=1 per
  # device inside the shard_map, so they never hit the T(2,128) tiling problem
  # the fold exists to avoid.
  # Folding batch into heads destroys the one-mask-per-example association.
  # Keep the optimization for the common unmasked path only.
  fold_batch = (
      attention_mask is None and batch > 1 and devices_in_batch_sharding == 1 and (batch * num_heads) % num_shards == 0
  )
  if fold_batch:
    query = query.reshape(1, batch * num_heads, *query.shape[2:])
    key = key.reshape(1, batch * num_heads, *key.shape[2:])
    value = value.reshape(1, batch * num_heads, *value.shape[2:])
    effective_num_heads = batch * num_heads
  else:
    effective_num_heads = num_heads

  if attention_mask is None:
    sharded_ulysses_attention = jax.shard_map(
        lambda q, k, v: wrap_ulysses_attention(q, k, v, None),
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=q_axis_names,
        check_vma=False,
    )

    def run_ulysses_attention(q, k, v):
      return sharded_ulysses_attention(q, k, v)

  else:
    sharded_ulysses_attention = jax.shard_map(
        wrap_ulysses_attention,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names, mask_axis_names),
        out_specs=q_axis_names,
        check_vma=False,
    )

    def run_ulysses_attention(q, k, v):
      return sharded_ulysses_attention(q, k, v, attention_mask)

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      effective_num_heads,
      num_shards,
      ulysses_attention_chunks,
      run_ulysses_attention,
  )

  if fold_batch:
    x = x.reshape(batch, num_heads, *x.shape[2:])
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)

  return x


def _ulysses_ring_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    ulysses_axis: str = INTERNAL_ULYSSES_AXIS,
    ring_axis: str = INTERNAL_RING_AXIS,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    ulysses_shards: int = -1,
    ulysses_attention_chunks: int = 1,
    preserve_asymmetric_block_sizes: bool = False,
) -> jax.Array:
  """2D context-parallel attention using a private Ulysses x ring mesh.

  Public configs only shard sequence on the context axis.  Internally this
  reshapes that same device axis into hidden ring and Ulysses axes, runs the
  Ulysses all-to-all over the hidden Ulysses axis, and rotates K/V over the
  hidden ring axis.
  """

  context_axis = CONTEXT
  if context_axis not in mesh.shape:
    raise ValueError(f"Ulysses ring attention requires mesh axis {context_axis!r}, got mesh" f" axes {mesh.shape}.")

  num_context_shards = mesh.shape[context_axis]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("Ulysses ring attention requires ulysses_shards to be set from config" " or command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the requested Ulysses shard count to"
        " divide the context shard count, got"
        f" context_shards={num_context_shards} and"
        f" ulysses_shards={num_ulysses_shards}."
    )
  if heads % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the number of heads to be divisible"
        f" by the requested Ulysses shard count, got heads={heads} and"
        f" ulysses_shards={num_ulysses_shards}."
    )
  num_ring_shards = num_context_shards // num_ulysses_shards
  internal_mesh = _create_internal_ulysses_ring_mesh(
      mesh,
      ring_shards=num_ring_shards,
      ulysses_shards=num_ulysses_shards,
      ring_axis=ring_axis,
      ulysses_axis=ulysses_axis,
  )
  internal_sequence_axes = (ring_axis, ulysses_axis)
  num_sequence_shards = num_context_shards

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_sequence_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_sequence_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_sequence_shards)
  attention_mask = _prepare_attention_mask_for_shard_map(attention_mask, query.shape[0], key.shape[2])
  num_heads = query.shape[1]

  block_sizes = _select_flash_block_sizes(
      query,
      key,
      flash_block_sizes,
      dtype,
      "tokamax_ring",
      preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
  )

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  internal_q_axis_names = _replace_mesh_axis_names(q_axis_names, context_axis, internal_sequence_axes)
  internal_kv_axis_names = _replace_mesh_axis_names(kv_axis_names, context_axis, internal_sequence_axes)
  mask_axis_names = nn.logical_to_mesh_axes((axis_names_kv[0], axis_names_kv[2]))
  internal_mask_axis_names = _replace_mesh_axis_names(mask_axis_names, context_axis, internal_sequence_axes)
  mask_needs_ulysses_gather = _mesh_axis_in_spec(internal_mask_axis_names[1], ulysses_axis)

  def wrap_ulysses_ring_attention(query, key, value, attention_mask):
    # Swap sharding: each device gives up a slice of heads and gathers
    # a slice of sequence, so the local kernel sees the full sequence.
    query = jax.lax.all_to_all(query, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    key = jax.lax.all_to_all(key, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    value = jax.lax.all_to_all(value, axis_name=ulysses_axis, split_axis=1, concat_axis=2, tiled=True)
    if attention_mask is not None and mask_needs_ulysses_gather:
      attention_mask = jax.lax.all_gather(attention_mask, axis_name=ulysses_axis, axis=1, tiled=True)

    uses_fused_kernel = block_sizes.use_fused_bwd_kernel
    block_q_sizes = (block_sizes.block_q, block_sizes.block_q_dkv)
    block_kv_sizes = (block_sizes.block_kv, block_sizes.block_kv_dkv)
    if uses_fused_kernel:
      block_q_sizes += (block_sizes.block_q_dkv,)
      block_kv_sizes += (block_sizes.block_kv_dkv,)
    else:
      block_q_sizes += (block_sizes.block_q_dq,)
      block_kv_sizes += (block_sizes.block_kv_dq,)

    block_q = max(*block_q_sizes)
    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, block_q)
    block_kv = max(*block_kv_sizes)
    key, _, key_seq_len = _pad_data_for_flash(key, heads, block_kv)
    value, _, _ = _pad_data_for_flash(value, heads, block_kv)

    q_padded_len = query.shape[2]
    kv_padded_len = key.shape[2]
    total_kv_len = kv_padded_len * num_ring_shards

    # Mask q/kv padding via segment ids, same as the tokamax_ring kernel.
    # Padding-only IDs are identical per shard; explicit KV masks rotate.
    segment_ids = _build_padding_segment_ids(
        query_seq_len,
        q_padded_len,
        key_seq_len,
        kv_padded_len,
        attention_mask,
        tokamax_splash_base.SegmentIds,
    )

    if not mask_padding_tokens and attention_mask is None:
      segment_ids = None

    mask = tokamax_splash_attention_mask.FullMask(_shape=(q_padded_len, total_kv_len))

    splash_kernel = tokamax_ring_attention_kernel.make_ring_attention(
        mask=mask,
        is_mqa=False,
        config=convert_to_tokamax_splash_config(
            block_sizes,
            residual_checkpoint_name=residual_checkpoint_name,
            use_base2_exp=use_base2_exp,
            use_experimental_scheduler=use_experimental_scheduler,
        ),
        save_residuals=False,
        ring_axis=ring_axis,
        kv_seq_shards=num_ring_shards,
        rotate_segment_ids=attention_mask is not None,
    )
    segment_ids_in_axes = 0 if attention_mask is not None else None
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, segment_ids_in_axes))
    attention_output = vmapped_splash(query, key, value, segment_ids)
    attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    # Restore original layout: head-sharded/full-sequence -> sequence-sharded/full-heads.
    attention_output = jax.lax.all_to_all(
        attention_output,
        axis_name=ulysses_axis,
        split_axis=2,
        concat_axis=1,
        tiled=True,
    )
    return attention_output

  devices_in_batch_sharding = mesh.shape["data"] * (mesh.shape["fsdp"] if "fsdp" in mesh.shape else 1)
  if not (query.shape[0] / devices_in_batch_sharding).is_integer():
    max_logging.log(
        "Warning, batch dimension should be shardable among the devices in"
        f" data and fsdp axis, batch dimension: {query.shape[0]},"
        f" devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_ulysses_ring_attention = jax.shard_map(
        lambda q, k, v: wrap_ulysses_ring_attention(q, k, v, None),
        mesh=internal_mesh,
        in_specs=(
            internal_q_axis_names,
            internal_kv_axis_names,
            internal_kv_axis_names,
        ),
        out_specs=internal_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_ring_attention(q, k, v):
      return sharded_ulysses_ring_attention(q, k, v)

  else:
    sharded_ulysses_ring_attention = jax.shard_map(
        wrap_ulysses_ring_attention,
        mesh=internal_mesh,
        in_specs=(
            internal_q_axis_names,
            internal_kv_axis_names,
            internal_kv_axis_names,
            internal_mask_axis_names,
        ),
        out_specs=internal_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_ring_attention(q, k, v):
      return sharded_ulysses_ring_attention(q, k, v, attention_mask)

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      num_heads,
      num_ulysses_shards,
      ulysses_attention_chunks,
      run_ulysses_ring_attention,
  )
  x = jax.lax.with_sharding_constraint(x, q_axis_names)
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)

  return x


def _max_row_norm_per_head(x: jax.Array) -> jax.Array:
  """Largest row L2 norm per head of a `[B, H, S, D]` activation."""
  row_sq = jnp.square(x).sum(axis=-1, dtype=jnp.float32)
  # 1.01 keeps the result an upper bound despite bf16 mantissa loss.
  return jnp.sqrt(row_sq.max(axis=(0, 2))) * 1.01


def _ulysses_ring_custom_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    heads: int,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    dtype: jnp.dtype = jnp.float32,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: jax.Array = None,
    ulysses_shards: int = -1,
    use_base2_exp: bool = True,
    use_experimental_scheduler: bool = False,
    bidirectional: bool = False,
    use_fixed_m: bool = False,
    ulysses_attention_chunks: int = 1,
) -> jax.Array:
  """Hybrid Ulysses + Ring (USP) with the CUSTOM splash kernel on main's mesh.

  Uses origin/main's explicit internal `(ring, ulysses)` mesh
  (`_create_internal_ulysses_ring_mesh`, commit c104db51) instead of single-axis
  collective sub-groups: the public `context` axis is reshaped with the Ulysses
  axis innermost, so the Ulysses all-to-all stays INTRA-chip and the ring
  rotates
  ACROSS chips. The per-shard attention is our custom splash kernel
  (`make_custom_ring_attention`), not the tokamax_ring kernel main uses.

    1. all-to-all over the (intra-chip) Ulysses axis: trade sequence for heads;
    2. ring (full ppermute) over the (cross-chip) ring axis, online-softmax
    merge;
    3. all-to-all back to restore the sequence-sharded / full-heads layout.

  U = ulysses_shards (from config); R = context // U. U=context -> pure
  Ulysses, U=1 -> pure Ring (all on the same custom kernel).
  """
  if attention_mask is not None:
    raise NotImplementedError(
        "ulysses_ring_custom does not support attention_mask (the custom splash"
        " kernels only handle padding via orig_seq_len); got a non-None"
        " attention_mask."
    )
  axis_name = "context"
  num_context_shards = mesh.shape[axis_name]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("ulysses_ring_custom requires ulysses_shards to be set from config or" " command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        "ulysses_ring_custom requires ulysses_shards to divide the context"
        f" shard count, got context_shards={num_context_shards} and"
        f" ulysses_shards={num_ulysses_shards}."
    )
  num_ring_shards = num_context_shards // num_ulysses_shards

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
  num_heads = query.shape[1]
  if num_heads % num_ulysses_shards != 0:
    raise ValueError(f"Ulysses+Ring requires heads divisible by U={num_ulysses_shards}, got" f" heads={num_heads}.")

  (
      bq,
      bkv,
      bkv_compute,
      bkv_compute_in,
      heads_per_tile,
      vmem_limit_bytes,
  ) = _extract_custom_block_sizes(flash_block_sizes)
  if heads_per_tile > 1:
    raise NotImplementedError("ulysses_ring_custom currently supports heads_per_tile == 1 only.")

  internal_mesh = _create_internal_ulysses_ring_mesh(mesh, num_ring_shards, num_ulysses_shards)
  ring_axis = INTERNAL_RING_AXIS
  ulysses_axis = INTERNAL_ULYSSES_AXIS

  q_axis_names = nn.logical_to_mesh_axes(axis_names_q)
  kv_axis_names = nn.logical_to_mesh_axes(axis_names_kv)
  internal_q_axis_names = _replace_mesh_axis_names(q_axis_names, axis_name, (ring_axis, ulysses_axis))
  internal_kv_axis_names = _replace_mesh_axis_names(kv_axis_names, axis_name, (ring_axis, ulysses_axis))

  @functools.partial(
      jax.shard_map,
      mesh=internal_mesh,
      in_specs=(
          internal_q_axis_names,
          internal_kv_axis_names,
          internal_kv_axis_names,
      ),
      out_specs=internal_q_axis_names,
      check_vma=False,
  )
  def wrap_ulysses_ring_attention(query, key, value):
    fixed_m_norms = None
    if use_fixed_m and num_ring_shards > 1:
      # Fixed-m's Cauchy-Schwarz inputs, reduced on the PRE-a2a activation so
      # the reduction overlaps the all_to_all instead of stalling the first
      # ring step (taking them after the a2a measured +8% end to end).
      #
      # The barrier is load-bearing: the norms are a second consumer of these
      # activations, and without it XLA duplicates the producer chain into the
      # norm fusion instead of materializing once -- worth 1.46 ms/layer, the
      # difference between fixed-m breaking even and winning.
      #
      # Reducing them further upstream (on the flat [B, S, H*D] form, where
      # head_dim is contiguous) is exact and looks cheaper, but there the array
      # is still globally sharded, so the reduction becomes a per-layer
      # all-reduce over the context axis: measured WORSE (+54 ms per forward).
      query, key = jax.lax.optimization_barrier((query, key))
      qn_local = _max_row_norm_per_head(query)
      kn_local = _max_row_norm_per_head(key)
      if use_base2_exp:
        qn_local = qn_local * LOG2E
      # The accumulate-vs-LSE lax.cond predicate must be uniform along the RING
      # axis (every ppermute participant takes the same branch).
      qn_all = jax.lax.pmax(qn_local, (ring_axis, ulysses_axis))
      mk_all = jax.lax.pmax(kn_local, ulysses_axis)
      heads_per_dev = qn_all.shape[0] // num_ulysses_shards
      start_head = jax.lax.axis_index(ulysses_axis) * heads_per_dev
      fixed_m_norms = (
          jax.lax.dynamic_slice_in_dim(qn_all, start_head, heads_per_dev),
          jax.lax.dynamic_slice_in_dim(mk_all, start_head, heads_per_dev),
      )

    # (1) Ulysses all-to-all over the (intra-chip) ulysses axis: heads -> sequence,
    # so each device holds the full ring-chunk sequence with heads/U heads.
    a2a = functools.partial(jax.lax.all_to_all, axis_name=ulysses_axis, tiled=True)
    query = a2a(query, split_axis=1, concat_axis=2)
    key = a2a(key, split_axis=1, concat_axis=2)
    value = a2a(value, split_axis=1, concat_axis=2)

    if use_base2_exp:
      query = query * LOG2E

    if use_fixed_m and num_ring_shards == 1:
      # K-smoothing precondition for fixed-m (R=1 / pure-ulysses semantics,
      # same as _ulysses_attention). The R>1 ring path deliberately does NOT
      # smooth: no ring rank holds the full K to compute a mean, and a per-
      # shard mean would shift each hop's logits differently, breaking the
      # cross-shard merge; it gates on the un-smoothed halved bound instead.
      kbar = jnp.mean(key, axis=2, keepdims=True)
      key = key - kbar

    query, kv_size, query_seq_len = _pad_data_for_flash(query, heads, bq)
    key, _, key_seq_len = _pad_data_for_flash(key, heads, bkv)
    value, _, _ = _pad_data_for_flash(value, heads, bkv)

    mk_arr = None
    if use_fixed_m and num_ring_shards == 1:
      qf = query.astype(jnp.float32)
      kf = key.astype(jnp.float32)
      qn_max = jnp.sqrt((qf * qf).sum(-1)).max(axis=(0, 2))  # (local_heads,)
      mk_h = jnp.sqrt((kf * kf).sum(-1)).max(axis=(0, 2))  # (local_heads,) local
      # pylint: disable-next=protected-access
      fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
      mk_arr = jnp.stack([mk_h, fixed_ok])  # (2, local_heads)

    bsizes = custom_splash._BlockSizes(  # pylint: disable=protected-access
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkv_compute,
        block_kv_compute_in=bkv_compute_in,
    )
    if num_ring_shards == 1:
      # (2a) R=1: the ring is trivial (no rotation) -> use the lighter dedicated
      # splash kernel (fuse_reciprocal, no fp32 online-softmax residual windows).
      # Same math as the 1-step ring, and it fits BQ=8448 where the ring kernel
      # OOMs (its 3x residual windows). make_splash_mha returns [H, D, S].
      splash_kernel = custom_splash.make_splash_mha(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          heads_per_tile=heads_per_tile,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          use_fixed_m=use_fixed_m,
      )
      if use_fixed_m:
        attention_output = jnp.swapaxes(
            jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))(query, key, value, mk_arr),
            2,
            3,
        )
      else:
        attention_output = jnp.swapaxes(jax.vmap(splash_kernel, in_axes=(0, 0, 0))(query, key, value), 2, 3)
    else:
      # (2b) Ring (full ppermute over the cross-chip ring axis) with the custom kernel.
      # bidirectional=True -> wrap-free schedule (streams K/V both directions one hop
      # at a time), for a non-wrapping ring axis. Selected by attention=ulysses_ring_custom_bidir.
      ring_kernel = tokamax_ring_attention_kernel.make_custom_ring_attention(
          block_sizes=bsizes,
          orig_q_seq_len=query_seq_len,
          orig_kv_seq_len=key_seq_len,
          use_base2_exp=use_base2_exp,
          use_experimental_scheduler=use_experimental_scheduler,
          vmem_limit_bytes=vmem_limit_bytes,
          ring_axis=ring_axis,
          ring_size=num_ring_shards,
          bidirectional=bidirectional,
          use_fixed_m=use_fixed_m,
          fixed_m_norms=fixed_m_norms,
      )
      attention_output = jax.vmap(ring_kernel, in_axes=(0, 0, 0))(query, key, value)
    attention_output = attention_output[:, :, :query_seq_len, :kv_size].astype(query.dtype)

    # (3) Ulysses all-to-all back: sequence -> heads, restoring the layout.
    attention_output = a2a(attention_output, split_axis=2, concat_axis=1)
    return attention_output

  x = _run_chunked_ulysses_attention(
      query,
      key,
      value,
      num_heads,
      num_ulysses_shards,
      ulysses_attention_chunks,
      wrap_ulysses_ring_attention,
  )
  x = jax.lax.with_sharding_constraint(x, q_axis_names)
  x = x[:, :, :orig_q_seq_len, :]
  x = _reshape_heads_to_head_dim(x)
  return x


def _apply_attention_dot(
    query: Array,
    key: Array,
    value: Array,
    dtype: jnp.dtype,
    heads: int,
    dim_head: int,
    scale: float,
    split_head_dim: bool,
    float32_qk_product: bool,
    use_memory_efficient_attention: bool,
    attention_mask: Array = None,
):
  """Apply Attention."""
  if split_head_dim:
    b = key.shape[0]
    query_states = jnp.reshape(query, (b, -1, heads, dim_head))
    key_states = jnp.reshape(key, (b, -1, heads, dim_head))
    value_states = jnp.reshape(value, (b, -1, heads, dim_head))
  else:
    query_states = _reshape_heads_to_batch_dim(query, heads)
    key_states = _reshape_heads_to_batch_dim(key, heads)
    value_states = _reshape_heads_to_batch_dim(value, heads)

  if float32_qk_product:
    query_states = query_states.astype(jnp.float32)
    key_states = key_states.astype(jnp.float32)

  if use_memory_efficient_attention and attention_mask is None:
    query_states = query_states.transpose(1, 0, 2)
    key_states = key_states.transpose(1, 0, 2)
    value_states = value_states.transpose(1, 0, 2)

    # this if statement create a chunk size for each layer of the unet
    # the chunk size is equal to the query_length dimension of the deepest layer of the unet

    flatten_latent_dim = query_states.shape[-3]
    if flatten_latent_dim % 64 == 0:
      query_chunk_size = int(flatten_latent_dim / 64)
    elif flatten_latent_dim % 16 == 0:
      query_chunk_size = int(flatten_latent_dim / 16)
    elif flatten_latent_dim % 4 == 0:
      query_chunk_size = int(flatten_latent_dim / 4)
    else:
      query_chunk_size = int(flatten_latent_dim)

    hidden_states = jax_memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        query_chunk_size=query_chunk_size,
        key_chunk_size=4096 * 4,
    )

    hidden_states = hidden_states.transpose(1, 0, 2)
  else:
    if split_head_dim:
      attention_scores = jnp.einsum("b t n h, b f n h -> b n f t", key_states, query_states)
    else:
      attention_scores = jnp.einsum("b i d, b j d->b i j", query_states, key_states)

    attention_scores = attention_scores * scale
    if attention_mask is not None:
      attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)
    attention_probs = nn.softmax(attention_scores, axis=-1 if split_head_dim else 2)
    if attention_mask is not None:
      has_valid_key = jnp.any(attention_mask == 0, axis=-1, keepdims=True)
      attention_probs = jnp.where(has_valid_key, attention_probs, 0)

    attention_probs = attention_probs.astype(dtype)

    # attend to values
    if split_head_dim:
      hidden_states = jnp.einsum("b n f t, b t n h -> b f n h", attention_probs, value_states)
      b = hidden_states.shape[0]
      hidden_states = jnp.reshape(hidden_states, (b, -1, heads * dim_head))
    else:
      hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs, value_states)
      hidden_states = _reshape_batch_dim_to_heads(hidden_states, heads)

  return hidden_states


KERNEL_REGISTRY = {}


def register_kernel(name: str):
  def decorator(func):
    KERNEL_REGISTRY[name] = func
    return func

  return decorator


# Register existing kernels at module level with context dict
@register_kernel("dot_product")
def dot_product_kernel(q, k, v, context):
  return _apply_attention_dot(
      q,
      k,
      v,
      context["dtype"],
      context["heads"],
      context["dim_head"],
      context["scale"],
      context["split_head_dim"],
      context["float32_qk_product"],
      context["use_memory_efficient_attention"],
      context["attention_mask"],
  )


@register_kernel("ulysses_custom")
def ulysses_custom_kernel(q, k, v, context):
  return _ulysses_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_custom_kernel=True,
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
  )


@register_kernel("ulysses_ring_custom")
def ulysses_ring_custom_kernel(q, k, v, context):
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
  )


@register_kernel("ulysses_ring_custom_fixed_m")
def ulysses_ring_custom_fixed_m_kernel(q, k, v, context):
  """fixed-m variant of ulysses_ring_custom: the per-shard custom splash kernel

  uses the Cauchy-Schwarz fixed-m softmax bound (no in-kernel running-max
  rescale). max||k|| and the K-smoothing mean are taken LOCALLY per ring shard
  (no per-layer ring collective); the outer ring online-softmax merge still
  re-normalizes across shards, so per-shard bounds stay correct.
  """
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
      ulysses_attention_chunks=context.get("ulysses_attention_chunks", 1),
  )


@register_kernel("ulysses_ring_custom_bidir")
def ulysses_ring_custom_bidir_kernel(q, k, v, context):
  """Wrap-free (bidirectional) variant of ulysses_ring_custom: the ring streams

  K/V both directions one hop at a time, avoiding the diameter-length wrap hop
  on a non-wrapping ring axis. Same USP split as ulysses_ring_custom otherwise.
  """
  return _ulysses_ring_custom_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_shards=context["ulysses_shards"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      bidirectional=True,
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
  )


@register_kernel("ulysses_custom_fixed_m")
def ulysses_custom_fixed_m_kernel(q, k, v, context):
  return _ulysses_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_custom_kernel=True,
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      use_fixed_m=True,
  )


@register_kernel("ulysses")
def ulysses_kernel(q, k, v, context):
  return _ulysses_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("ulysses_ring")
def ulysses_ring_kernel(q, k, v, context):
  return _ulysses_ring_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      ulysses_shards=context["ulysses_shards"],
      ulysses_attention_chunks=context["ulysses_attention_chunks"],
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("flash")
def flash_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="flash",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("tokamax_flash")
def tokamax_flash_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_flash",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("tokamax_ring")
def tokamax_ring_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_ring",
      mask_padding_tokens=context["mask_padding_tokens"],
      residual_checkpoint_name=context["residual_checkpoint_name"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context["use_base2_exp"],
      use_experimental_scheduler=context["use_experimental_scheduler"],
      is_causal=context.get("is_causal", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


@register_kernel("tokamax_ring_custom")
def tokamax_ring_custom_kernel(q, k, v, context):
  return _tpu_flash_attention(
      q,
      k * context["scale"],
      v,
      context["heads"],
      context["mesh"],
      context["axis_names_q"],
      context["axis_names_kv"],
      context["flash_block_sizes"],
      context["dtype"],
      attention_kernel="tokamax_ring_custom",
      mask_padding_tokens=context["mask_padding_tokens"],
      attention_mask=context["attention_mask"],
      use_base2_exp=context.get("use_base2_exp", True),
      use_experimental_scheduler=context.get("use_experimental_scheduler", False),
      preserve_asymmetric_block_sizes=context.get("preserve_asymmetric_block_sizes", False),
  )


def _apply_attention(
    query: Array,
    key: Array,
    value: Array,
    heads: int,
    dim_head: int,
    split_head_dim: bool,
    float32_qk_product: bool,
    attention_kernel: str,
    flash_min_seq_length: int,
    use_memory_efficient_attention: bool,
    scale: float,
    dtype: jnp.dtype,
    mesh: Mesh,
    axis_names_q: AxisNames,
    axis_names_kv: AxisNames,
    flash_block_sizes: BlockSizes,
    mask_padding_tokens: bool = True,
    residual_checkpoint_name: str | None = None,
    attention_mask: Array = None,
    use_base2_exp: bool = False,
    use_experimental_scheduler: bool = False,
    ulysses_shards: int = -1,
    ulysses_attention_chunks: int = 1,
    is_causal: bool = False,
    preserve_asymmetric_block_sizes: bool = False,
):
  """Routes to different attention kernels using a module-level registry."""

  _check_attention_inputs(query, key, value)
  seq_len_idx = 1
  if query.ndim == 4:
    seq_len_idx = 2

  can_use_flash_attention = True
  if attention_kernel in [
      "flash",
      "tokamax_flash",
      "ulysses",
      "ulysses_custom",
      "ulysses_custom_fixed_m",
      "ulysses_ring",
  ]:
    can_use_flash_attention = (
        query.shape[seq_len_idx] >= flash_min_seq_length
        and key.shape[seq_len_idx] >= flash_min_seq_length
        and value.shape[seq_len_idx] >= flash_min_seq_length
    )

  effective_attention_kernel = attention_kernel
  if attention_kernel == "dot_product" or use_memory_efficient_attention or not can_use_flash_attention:
    effective_attention_kernel = "dot_product"

  # Masks enter the dispatcher as canonical [B, K] keep masks. Adapt them
  # only after fallback selection because a configured flash kernel may use
  # dot-product attention for short sequences.
  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError("attention_mask must have shape [batch, kv_length], got" f" {attention_mask.shape}.")
    attention_mask = attention_mask.astype(jnp.bool_)
    if effective_attention_kernel == "dot_product":
      attention_bias = jnp.where(
          attention_mask,
          jnp.asarray(0.0, dtype=dtype),
          jnp.asarray(-10000.0, dtype=dtype),
      )
      if split_head_dim:
        attention_mask = attention_bias[:, None, None, :]
      else:
        attention_mask = jnp.repeat(attention_bias, heads, axis=0)[:, None, :]

  context = {
      "heads": heads,
      "mesh": mesh,
      "axis_names_q": axis_names_q,
      "axis_names_kv": axis_names_kv,
      "flash_block_sizes": flash_block_sizes,
      "dtype": dtype,
      "mask_padding_tokens": mask_padding_tokens,
      "residual_checkpoint_name": residual_checkpoint_name,
      "attention_mask": attention_mask,
      "scale": scale,
      "use_base2_exp": use_base2_exp,
      "use_experimental_scheduler": use_experimental_scheduler,
      "ulysses_shards": ulysses_shards,
      "ulysses_attention_chunks": ulysses_attention_chunks,
      "dim_head": dim_head,
      "split_head_dim": split_head_dim,
      "float32_qk_product": float32_qk_product,
      "use_memory_efficient_attention": use_memory_efficient_attention,
      "is_causal": is_causal,
      "preserve_asymmetric_block_sizes": preserve_asymmetric_block_sizes,
  }

  # Module-level Registry lookup
  if effective_attention_kernel in KERNEL_REGISTRY:
    with jax.named_scope(f"kernel_{effective_attention_kernel}"):
      return KERNEL_REGISTRY[effective_attention_kernel](query, key, value, context)

  raise ValueError(f"Unexpected attention kernel {effective_attention_kernel=}.")


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
  """Multi-head dot product attention with a limited number of queries."""
  num_kv, num_heads, k_features = key.shape[-3:]
  v_features = value.shape[-1]
  key_chunk_size = min(key_chunk_size, num_kv)
  query = query / jnp.sqrt(k_features)

  @functools.partial(jax.checkpoint, prevent_cse=False)
  def summarize_chunk(query, key, value):
    attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

    max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    max_score = jax.lax.stop_gradient(max_score)
    exp_weights = jnp.exp(attn_weights - max_score)

    exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
    max_score = jnp.einsum("...qhk->...qh", max_score)

    return (exp_values, exp_weights.sum(axis=-1), max_score)

  def chunk_scanner(chunk_idx):
    # julienne key array
    key_chunk = jax.lax.dynamic_slice(
        operand=key,
        start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
        slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
    )

    # julienne value array
    value_chunk = jax.lax.dynamic_slice(
        operand=value,
        start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
        slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
    )

    return summarize_chunk(query, key_chunk, value_chunk)

  chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

  global_max = jnp.max(chunk_max, axis=0, keepdims=True)
  max_diffs = jnp.exp(chunk_max - global_max)

  chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
  chunk_weights *= max_diffs

  all_values = chunk_values.sum(axis=0)
  all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

  return all_values / all_weights


def jax_memory_efficient_attention(
    query,
    key,
    value,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
):
  r"""Flax Memory-efficient multi-head dot product attention.

  https://arxiv.org/abs/2112.05682v2
  https://github.com/AminRezaei0x443/memory-efficient-attention

  Args:
      query (`jnp.ndarray`): (batch..., query_length, head,
        query_key_depth_per_head)
      key (`jnp.ndarray`): (batch..., key_value_length, head,
        query_key_depth_per_head)
      value (`jnp.ndarray`): (batch..., key_value_length, head,
        value_depth_per_head) precision (`jax.lax.Precision`, *optional*,
        defaults to `jax.lax.Precision.HIGHEST`): numerical precision for
        computation
      query_chunk_size (`int`, *optional*, defaults to 1024): chunk size to
        divide query array value must divide query_length equally without
        remainder
      key_chunk_size (`int`, *optional*, defaults to 4096): chunk size to divide
        key and value array value must divide key_value_length equally without
        remainder

  Returns:
      (`jnp.ndarray`) with shape of (batch..., query_length, head,
      value_depth_per_head)
  """
  num_q, num_heads, q_features = query.shape[-3:]

  def chunk_scanner(chunk_idx, _):
    # julienne query array
    query_chunk = jax.lax.dynamic_slice(
        operand=query,
        start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
        slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
    )

    return (
        chunk_idx + query_chunk_size,  # unused ignore it
        _query_chunk_attention(
            query=query_chunk,
            key=key,
            value=value,
            precision=precision,
            key_chunk_size=key_chunk_size,
        ),
    )

  _, res = jax.lax.scan(
      f=chunk_scanner,
      init=0,
      xs=None,
      length=math.ceil(num_q / query_chunk_size),  # start counter  # stop counter
  )

  return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


def apply_rope(xq: Array, xk: Array, freqs_cis: Any) -> tuple[Array, Array]:
  """Apply rotary embeddings given (cos, sin) or packed rotation-matrix frequencies."""
  if isinstance(freqs_cis, (tuple, list)):
    cos, sin = freqs_cis
    if cos.ndim == 2:
      seq_len = cos.shape[0]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
      else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3 and cos.shape[0] == 1:
      seq_len = cos.shape[1]
      if xq.ndim == 4 and xq.shape[2] == seq_len:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
      else:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]

    def _rotate(x):
      x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
      x_real = x_reshaped[..., 0]
      x_imag = x_reshaped[..., 1]
      return jnp.stack([-x_imag, x_real], axis=-1).reshape(*x.shape)

    xq_out = xq * cos + _rotate(xq) * sin
    xk_out = xk * cos + _rotate(xk) * sin
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)

  xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)

  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

  return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)


# New Class for Wan I2V
class NNXSimpleFeedForward(nnx.Module):
  """Two-layer feed-forward (Linear -> activation -> Linear)."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      activation_fn: str = "gelu",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      sharding_specs: Optional[Any] = None,
  ):
    inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    net_0_kernel = safe_getattr(sharding_specs, "net_0_kernel", ("embed", "mlp"))
    net_0_bias = safe_getattr(sharding_specs, "net_0_bias", ("mlp",))
    net_2_kernel = safe_getattr(sharding_specs, "net_2_kernel", ("mlp", "embed"))
    net_2_bias = safe_getattr(sharding_specs, "net_2_bias", ("embed",))

    self.net_0 = nnx.Linear(
        dim,
        inner_dim,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_0_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_0_bias),
    )
    self.act = get_activation(activation_fn)
    self.net_2 = nnx.Linear(
        inner_dim,
        dim_out,
        rngs=rngs,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), net_2_kernel),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, net_2_bias),
    )

  def __call__(self, hidden_states: Array) -> Array:
    hidden_states = self.net_0(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.net_2(hidden_states)
    return hidden_states


class NNXAttentionOp(nnx.Module):
  """NNX attention op dispatching to the configured attention kernel."""

  def __init__(
      self,
      mesh: Mesh,
      attention_kernel: str,
      scale: float,
      heads: int,
      dim_head: int,
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = True,
      float32_qk_product: bool = True,
      axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV),
      axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV),
      # Uses splash attention on cross attention.
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      dtype: DType = jnp.float32,
      quant: Quant = None,
      mask_padding_tokens: bool = True,
      residual_checkpoint_name: str | None = None,
      use_base2_exp: bool = False,
      use_experimental_scheduler: bool = False,
      ulysses_shards: int = -1,
      ulysses_attention_chunks: int = 1,
  ):
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.ulysses_shards = ulysses_shards
    self.ulysses_attention_chunks = ulysses_attention_chunks

    self.mesh = mesh
    self.scale = scale
    self.heads = heads
    self.dim_head = dim_head
    self.attention_kernel = attention_kernel
    self.use_memory_efficient_attention = use_memory_efficient_attention
    self.split_head_dim = split_head_dim
    self.float32_qk_product = float32_qk_product
    self.axis_names_q = axis_names_q
    self.axis_names_kv = axis_names_kv
    self.flash_min_seq_length = flash_min_seq_length
    self.flash_block_sizes = flash_block_sizes
    self.dtype = dtype
    self.quant = quant
    self.mask_padding_tokens = mask_padding_tokens
    self.residual_checkpoint_name = residual_checkpoint_name

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      attention_mask: Array = None,
      preserve_asymmetric_block_sizes: bool = False,
  ):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        mask_padding_tokens=self.mask_padding_tokens,
        residual_checkpoint_name=self.residual_checkpoint_name,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp if hasattr(self, "use_base2_exp") else False,
        use_experimental_scheduler=self.use_experimental_scheduler
        if hasattr(self, "use_experimental_scheduler")
        else False,
        ulysses_shards=(self.ulysses_shards if hasattr(self, "ulysses_shards") else -1),
        ulysses_attention_chunks=(self.ulysses_attention_chunks if hasattr(self, "ulysses_attention_chunks") else 1),
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )


class AttentionOp(nn.Module):
  """Linen attention op dispatching to the configured attention kernel."""

  mesh: Mesh
  attention_kernel: str
  scale: float
  heads: int
  dim_head: int
  use_memory_efficient_attention: bool = False
  split_head_dim: bool = False
  float32_qk_product: bool = True
  axis_names_q: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV)
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  dtype: DType = jnp.float32
  quant: Quant = None
  use_base2_exp: bool = False
  use_experimental_scheduler: bool = False
  ulysses_shards: int = -1
  ulysses_attention_chunks: int = 1
  is_causal: bool = False

  def apply_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      attention_mask: Array = None,
      preserve_asymmetric_block_sizes: bool = False,
  ):
    return _apply_attention(
        query=query,
        key=key,
        value=value,
        heads=self.heads,
        dim_head=self.dim_head,
        split_head_dim=self.split_head_dim,
        float32_qk_product=self.float32_qk_product,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        scale=self.scale,
        dtype=self.dtype,
        mesh=self.mesh,
        axis_names_q=self.axis_names_q,
        axis_names_kv=self.axis_names_kv,
        flash_block_sizes=self.flash_block_sizes,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
        ulysses_shards=self.ulysses_shards,
        ulysses_attention_chunks=self.ulysses_attention_chunks,
        is_causal=self.is_causal,
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )


class FlaxWanAttention(nnx.Module):
  """Wan self/cross attention block (QKV projections, QK norm, RoPE, output projection)."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      query_dim: int,
      cross_attention_dim: Optional[int] = None,
      heads: int = 8,
      dim_head: int = 64,
      dropout: float = 0.0,
      eps: float = 1e-6,
      qk_norm: str = "rms_norm_across_heads",
      use_memory_efficient_attention: bool = False,
      split_head_dim: bool = False,
      attention_kernel: str = "flash",
      flash_min_seq_length: int = 0,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      key_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      value_axis_names: AxisNames = (BATCH, LENGTH, HEAD),
      out_axis_names: AxisNames = (BATCH, LENGTH, EMBED),
      precision: jax.lax.Precision = None,
      qkv_bias: bool = False,
      quant: Quant = None,
      is_self_attention: bool = True,
      mask_padding_tokens: bool = True,
      residual_checkpoint_name: str | None = None,
      enable_jax_named_scopes: bool = False,
      added_kv_proj_dim: Optional[int] = None,  # New for I2V
      image_seq_len: Optional[int] = None,  # New for I2V
      attention_config: Optional[dict] = None,
  ):
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        "ulysses_attention_chunks": 1,
        **(attention_config or {}),
    }

    if attention_kernel == "flash" and mesh is None:
      raise ValueError("The flash attention kernel requires a value for mesh, but mesh is" f" {self.mesh}")
    self.dim_head = dim_head
    self.heads = heads
    self.inner_dim = dim_head * heads
    scale = dim_head**-0.5
    self.qk_norm = qk_norm
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.out_axis_names = out_axis_names
    self.enable_jax_named_scopes = enable_jax_named_scopes

    cross_attention_remapped_to_flash = not is_self_attention and attention_kernel in (
        "tokamax_ring",
        "tokamax_ring_custom",
        "ulysses_ring",
        "ulysses_ring_custom",
        "ulysses_ring_custom_fixed_m",
        "ulysses_ring_custom_bidir",
        "ulysses_custom",
        "ulysses_custom_fixed_m",
    )
    cross_attention_uses_local_kv = not is_self_attention and (
        cross_attention_remapped_to_flash or attention_kernel in ("flash", "tokamax_flash")
    )
    if is_self_attention:
      axis_names_q = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (BATCH, SELF_ATTN_HEAD, SELF_ATTN_KV_LENGTH, D_KV)
    else:
      axis_names_q = (BATCH, CROSS_ATTN_HEAD, CROSS_ATTN_Q_LENGTH, D_KV)
      axis_names_kv = (
          BATCH,
          CROSS_ATTN_HEAD,
          None if cross_attention_uses_local_kv else CROSS_ATTN_KV_LENGTH,
          D_KV,
      )
    if cross_attention_remapped_to_flash:
      attention_kernel = "tokamax_flash"
    elif attention_kernel in ("tokamax_ring", "tokamax_ring_custom", "ulysses_ring") and not is_self_attention:
      attention_kernel = "tokamax_flash"  # do not use ring attention for cross attention
    elif (
        attention_kernel
        in (
            "ulysses_ring_custom",
            "ulysses_ring_custom_bidir",
            "ulysses_ring_custom_fixed_m",
        )
        and not is_self_attention
    ):
      attention_kernel = "ulysses_custom"  # plain ulysses (no ring) for cross attention
    self.added_kv_proj_dim = added_kv_proj_dim  # New for I2V
    self.image_seq_len = image_seq_len  # New for I2V
    tpu_type = get_tpu_type()
    self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128

    self.attention_op = NNXAttentionOp(
        mesh=mesh,
        attention_kernel=attention_kernel,
        scale=scale,
        heads=heads,
        dim_head=dim_head,
        use_memory_efficient_attention=use_memory_efficient_attention,
        split_head_dim=split_head_dim,
        float32_qk_product=False,
        axis_names_q=axis_names_q,
        axis_names_kv=axis_names_kv,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        dtype=dtype,
        quant=quant,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name=residual_checkpoint_name,
        use_base2_exp=attention_config["use_base2_exp"],
        use_experimental_scheduler=attention_config["use_experimental_scheduler"],
        ulysses_shards=attention_config["ulysses_shards"],
        ulysses_attention_chunks=attention_config["ulysses_attention_chunks"],
    )
    # None axes corresponds to the stacked weights across all blocks
    # because of the use of nnx.vmap and nnx.scan.
    # Dims are [num_blocks, embed, heads]
    kernel_axes = ("embed", "heads")
    qkv_init_kernel = nnx.with_partitioning(nnx.initializers.lecun_normal(), kernel_axes)

    self.query = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.key = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.value = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=qkv_init_kernel,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("heads",),
        ),
    )

    self.proj_attn = nnx.Linear(
        rngs=rngs,
        in_features=self.inner_dim,
        out_features=self.inner_dim,
        kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("heads", "embed")),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros,
            ("embed",),
        ),
    )

    self.drop_out = nnx.Dropout(dropout, deterministic=False)

    self.norm_q = nnx.data(None)
    self.norm_k = nnx.data(None)
    if qk_norm is not None:
      self.norm_q = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

      self.norm_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
          param_dtype=weights_dtype,
      )

    # New layers for I2V image conditioning
    self.add_k_proj = nnx.data(None)
    self.add_v_proj = nnx.data(None)
    self.norm_added_k = nnx.data(None)
    if self.added_kv_proj_dim is not None:
      self.add_k_proj = nnx.Linear(
          self.added_kv_proj_dim,
          self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          bias_init=nnx.with_partitioning(
              nnx.initializers.zeros,
              ("embed",),
          ),
      )
      self.add_v_proj = nnx.Linear(
          self.added_kv_proj_dim,
          self.inner_dim,
          rngs=rngs,
          dtype=dtype,
          param_dtype=weights_dtype,
          precision=precision,
          bias_init=nnx.with_partitioning(
              nnx.initializers.zeros,
              ("embed",),
          ),
      )
      self.norm_added_k = nnx.RMSNorm(
          num_features=self.inner_dim,
          rngs=rngs,
          epsilon=eps,
          dtype=dtype,
          param_dtype=weights_dtype,
          scale_init=nnx.with_partitioning(
              nnx.initializers.ones,
              ("norm",),
          ),
      )

  def _apply_rope(self, xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to query and key."""
    # 1. Extract cos and sin, keeping them in native bfloat16
    cos = jnp.real(freqs_cis).astype(xq.dtype)
    sin = jnp.imag(freqs_cis).astype(xq.dtype)

    # 2. Reshape the last dimension into pairs
    xq_reshaped = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_reshaped = xk.reshape(*xk.shape[:-1], -1, 2)

    # 3. Unbind the pairs
    xq_0, xq_1 = xq_reshaped[..., 0], xq_reshaped[..., 1]
    xk_0, xk_1 = xk_reshaped[..., 0], xk_reshaped[..., 1]

    # 4. Pure real arithmetic (XLA will fuse these instantly into FMA instructions)
    xq_out_0 = xq_0 * cos - xq_1 * sin
    xq_out_1 = xq_0 * sin + xq_1 * cos

    xk_out_0 = xk_0 * cos - xk_1 * sin
    xk_out_1 = xk_0 * sin + xk_1 * cos

    # 5. Stack and reshape back to original
    xq_out = jnp.stack([xq_out_0, xq_out_1], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([xk_out_0, xk_out_1], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array = None,
      rotary_emb: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
  ) -> jax.Array:
    hidden_states = nn.with_logical_constraint(hidden_states, (BATCH, LENGTH, HEAD))
    if encoder_hidden_states is not None:
      encoder_hidden_states = nn.with_logical_constraint(encoder_hidden_states, (BATCH, LENGTH, HEAD))
    dtype = hidden_states.dtype
    is_self_attention = encoder_hidden_states is None
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    is_i2v_cross_attention = self.added_kv_proj_dim is not None and not is_self_attention

    # For T2V self-attention and cross-attention, we skip passing the mask
    # to avoid overhead, as it should be all 1s for unpadded sequences.
    if not is_i2v_cross_attention:
      encoder_attention_mask = None

    if not is_i2v_cross_attention:
      with jax.named_scope("query_proj"):
        query_proj = self.query(hidden_states)

      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj = self.norm_q(query_proj)

      if not is_self_attention and cached_kv is not None and "text" in cached_kv:
        key_proj, value_proj = cached_kv["text"]
      else:
        with jax.named_scope("key_proj"):
          key_proj = self.key(encoder_hidden_states)
        with jax.named_scope("value_proj"):
          value_proj = self.value(encoder_hidden_states)

        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj = self.norm_k(key_proj)

      if rotary_emb is not None:
        with self.conditional_named_scope("attn_rope"):
          query_proj = _unflatten_heads(query_proj, self.heads)
          key_proj = _unflatten_heads(key_proj, self.heads)
          value_proj = _unflatten_heads(value_proj, self.heads)
          # output of _unflatten_heads Batch, heads, seq_len, head_dim
          query_proj, key_proj = self._apply_rope(query_proj, key_proj, rotary_emb)

      query_proj = checkpoint_name(query_proj, "query_proj")
      key_proj = checkpoint_name(key_proj, "key_proj")
      value_proj = checkpoint_name(value_proj, "value_proj")

      with jax.named_scope("apply_attention"):
        attn_output = self.attention_op.apply_attention(
            query_proj,
            key_proj,
            value_proj,
            attention_mask=encoder_attention_mask,
        )

    else:
      # NEW PATH for I2V CROSS-ATTENTION
      with self.conditional_named_scope("proj_query"):
        query_proj_raw = self.query(hidden_states)

      # Image embeddings are padded to multiples of 128 (v5p and below) or 256 (v6e and above) for TPU flash attention
      # Calculate the padded length to correctly split image and text embeddings
      if self.added_kv_proj_dim is not None:
        alignment = self.alignment
        if self.image_seq_len is not None:
          image_seq_len_actual = self.image_seq_len
        else:
          image_seq_len_actual = 257
        padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment  # 257 -> 384
        encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
        encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]

        # Use the passed encoder_attention_mask (created in embeddings_flax.py) if using Flash Attention
        # It contains the image mask: [1]*257 + [0]*127 for 257 real image tokens padded to 384
        if encoder_attention_mask is not None:
          encoder_attention_mask_img = encoder_attention_mask[:, :padded_img_len]
        else:
          # Fallback: no mask means treat all as valid (for dot product attention)
          encoder_attention_mask_img = None
      else:
        # If no image_seq_len is specified, treat all as text
        encoder_hidden_states_img = None
        encoder_hidden_states_text = encoder_hidden_states
        encoder_attention_mask_img = None

      if self.qk_norm:
        with self.conditional_named_scope("attn_q_norm"):
          query_proj_text = self.norm_q(query_proj_raw)
      else:
        query_proj_text = query_proj_raw

      # Text K/V
      if cached_kv is not None and "text" in cached_kv:
        key_proj_text, value_proj_text = cached_kv["text"]
      else:
        with self.conditional_named_scope("proj_key"):
          key_proj_text = self.key(encoder_hidden_states_text)
        if self.qk_norm:
          with self.conditional_named_scope("attn_k_norm"):
            key_proj_text = self.norm_k(key_proj_text)
        with self.conditional_named_scope("proj_value"):
          value_proj_text = self.value(encoder_hidden_states_text)

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        if cached_kv is not None and "image" in cached_kv:
          key_proj_img, value_proj_img = cached_kv["image"]
        else:
          with self.conditional_named_scope("add_proj_k"):
            key_proj_img = self.add_k_proj(encoder_hidden_states_img)
          with self.conditional_named_scope("norm_add_k"):
            key_proj_img = self.norm_added_k(key_proj_img)
          with self.conditional_named_scope("add_proj_v"):
            value_proj_img = self.add_v_proj(encoder_hidden_states_img)
        query_proj_img = query_proj_raw
        # Check norm_added_k too
        # Checkpointing
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")
        key_proj_img = checkpoint_name(key_proj_img, "key_proj_img")
        value_proj_img = checkpoint_name(value_proj_img, "value_proj_img")
        query_proj_img = checkpoint_name(query_proj_img, "query_proj_img")

        # Attention - tensors are (B, S, D)
        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output_text = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)
        with self.conditional_named_scope("cross_attn_img_apply"):
          # Pass encoder_attention_mask_img for image cross-attention to mask padded tokens
          attn_output_img = self.attention_op.apply_attention(
              query_proj_img,
              key_proj_img,
              value_proj_img,
              attention_mask=encoder_attention_mask_img,
          )

        attn_output = attn_output_text + attn_output_img
      else:
        # No image embeddings, only text cross-attention
        query_proj_text = checkpoint_name(query_proj_text, "query_proj")
        key_proj_text = checkpoint_name(key_proj_text, "key_proj_text")
        value_proj_text = checkpoint_name(value_proj_text, "value_proj_text")

        with self.conditional_named_scope("cross_attn_text_apply"):
          attn_output = self.attention_op.apply_attention(query_proj_text, key_proj_text, value_proj_text)

    attn_output = attn_output.astype(dtype=dtype)
    attn_output = checkpoint_name(attn_output, "attn_output")

    with jax.named_scope("proj_attn"):
      hidden_states = self.proj_attn(attn_output)
      if self.drop_out.rate > 0:
        hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    return hidden_states

  def compute_kv(
      self,
      encoder_hidden_states: jax.Array,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    """Project encoder states to cross-attention K/V (plus image K/V for I2V)."""
    is_i2v_cross_attention = self.added_kv_proj_dim is not None

    if not is_i2v_cross_attention:
      with jax.named_scope("key_proj"):
        key_proj = self.key(encoder_hidden_states)
      with jax.named_scope("value_proj"):
        value_proj = self.value(encoder_hidden_states)

      if self.qk_norm:
        with self.conditional_named_scope("attn_k_norm"):
          key_proj = self.norm_k(key_proj)

      return {"text": (key_proj, value_proj)}
    else:
      # Image embeddings are padded to multiples of 128 (v5p and below) or 256 (v6e and above) for TPU flash attention
      alignment = self.alignment
      if self.image_seq_len is not None:
        image_seq_len_actual = self.image_seq_len
      else:
        image_seq_len_actual = 257
      padded_img_len = ((image_seq_len_actual + alignment - 1) // alignment) * alignment

      if encoder_attention_mask is None:
        padded_img_len = image_seq_len_actual

      encoder_hidden_states_img = encoder_hidden_states[:, :padded_img_len, :]
      encoder_hidden_states_text = encoder_hidden_states[:, padded_img_len:, :]

      # Text K/V
      with self.conditional_named_scope("proj_key"):
        key_proj_text = self.key(encoder_hidden_states_text)
      if self.qk_norm:
        with self.conditional_named_scope("attn_k_norm"):
          key_proj_text = self.norm_k(key_proj_text)
      with self.conditional_named_scope("proj_value"):
        value_proj_text = self.value(encoder_hidden_states_text)

      # Image K/V (only if image embeddings are present)
      if encoder_hidden_states_img is not None:
        with self.conditional_named_scope("add_proj_k"):
          key_proj_img = self.add_k_proj(encoder_hidden_states_img)
        with self.conditional_named_scope("norm_add_k"):
          key_proj_img = self.norm_added_k(key_proj_img)
        with self.conditional_named_scope("add_proj_v"):
          value_proj_img = self.add_v_proj(encoder_hidden_states_img)

        return {
            "text": (key_proj_text, value_proj_text),
            "image": (key_proj_img, value_proj_img),
        }
      else:
        return {"text": (key_proj_text, value_proj_text)}


# =============================================================================
# 3. Embeddings
# =============================================================================


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
  """Returns the positional encoding (same as Tensor2Tensor).

  Args:
      timesteps: a 1-D or 2-D Tensor of indices. These may be fractional.
      embedding_dim: The number of output channels.
      min_timescale: The smallest time unit (should probably be 0.0).
      max_timescale: The largest time unit.

  Returns:
      a Tensor of timing signals [B, num_channels] or [B, N, num_channels]
  """
  assert timesteps.ndim <= 2, "Timesteps should be a 1d or 2d-array"
  assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
  num_timescales = float(embedding_dim // 2)
  log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
  inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  emb = jnp.expand_dims(timesteps, -1) * inv_timescales

  # scale embeddings
  scaled_time = scale * emb

  if flip_sin_to_cos:
    signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=-1)
  else:
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
  return signal


class NNXTimestepEmbedding(nnx.Module):
  r"""Time step Embedding Module. Learns embeddings for input time steps.

  Args:
      time_embed_dim (`int`, *optional*, defaults to `32`): Time step embedding
        dimension dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
        Parameters `dtype`
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      time_embed_dim: int = 32,
      act_fn: str = "silu",
      out_dim: int = None,
      post_act_fn: Optional[str] = None,
      cond_proj_dim: int = None,
      sample_proj_bias=True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      sharding_specs: Optional[Any] = None,
  ):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_channels,
        out_features=time_embed_dim,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_1_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_1_bias),
    )

    if cond_proj_dim is not None:
      # Never taken for Wan (cond_proj_dim is None); kept as ported from MaxDiffusion.
      self.cond_proj = nnx.Linear(  # pylint: disable=no-value-for-parameter
          rngs=rngs,
      )
    else:
      self.cond_proj = None

    self.act = get_activation(act_fn)

    if out_dim is not None:
      time_embed_dim_out = out_dim
    else:
      time_embed_dim_out = time_embed_dim

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=time_embed_dim,
        out_features=time_embed_dim_out,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_2_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_2_bias),
    )

    if post_act_fn is None:
      self.post_act = None
    else:
      self.post_act = get_activation(post_act_fn)

  def __call__(self, sample, condition=None):
    if condition is not None:
      sample = sample + self.cond_proj(condition)
    sample = self.linear_1(sample)

    if self.act is not None:
      sample = self.act(sample)
    sample = self.linear_2(sample)

    if self.post_act is not None:
      sample = self.post_act(sample)
    return sample


class FlaxTimestepEmbedding(nn.Module):
  r"""Time step Embedding Module. Learns embeddings for input time steps.

  Args:
      time_embed_dim (`int`, *optional*, defaults to `32`): Time step embedding
        dimension dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
        Parameters `dtype`
  """

  time_embed_dim: int = 32
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, temb):
    temb = nn.Dense(
        self.time_embed_dim,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="linear_1",
    )(temb)
    temb = nn.silu(temb)
    temb = nn.Dense(
        self.time_embed_dim,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        name="linear_2",
    )(temb)
    return temb


class NNXFlaxTimesteps(nnx.Module):
  """Sinusoidal timestep embeddings (NNX)."""

  def __init__(
      self,
      dim: int = 32,
      flip_sin_to_cos: bool = False,
      freq_shift: float = 1.0,
      scale: int = 1,
  ):
    self.dim = dim
    self.flip_sin_to_cos = flip_sin_to_cos
    self.freq_shift = freq_shift
    self.scale = scale

  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(
        timesteps,
        embedding_dim=self.dim,
        flip_sin_to_cos=self.flip_sin_to_cos,
        freq_shift=self.freq_shift,
    )


class FlaxTimesteps(nn.Module):
  r"""Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

  Args:
      dim (`int`, *optional*, defaults to `32`): Time step embedding dimension
  """

  dim: int = 32
  flip_sin_to_cos: bool = False
  freq_shift: float = 1.0
  scale: int = 1

  @nn.compact
  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(
        timesteps,
        embedding_dim=self.dim,
        flip_sin_to_cos=self.flip_sin_to_cos,
        freq_shift=self.freq_shift,
    )


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[jnp.array, int],
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=jnp.float32,
    use_real: bool = True,
):
  """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
  assert dim % 2 == 0

  if isinstance(pos, int):
    pos = jnp.arange(pos)

  theta = theta * ntk_factor
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor
  freqs = jnp.outer(pos, freqs)
  if use_real:
    # Flux
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)
  else:
    # Wan 2.1
    out = jnp.exp(1j * freqs)
  return out


class NNXWanImageEmbedding(nnx.Module):
  """Projects CLIP image features into the transformer width (I2V conditioning)."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_features: int,
      out_features: int,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: jax.lax.Precision,
      pos_embed_seq_len=None,
      alignment: Optional[int] = None,
      flash_min_seq_length: int = 4096,
  ):
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=in_features, elementwise_affine=True, eps=1e-6)
    self.ff = NNXSimpleFeedForward(
        rngs=rngs,
        dim=in_features,
        dim_out=out_features,
        mult=1,
        activation_fn="gelu",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=out_features, elementwise_affine=True, eps=1e-6)
    if alignment is None:
      tpu_type = get_tpu_type()
      self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    else:
      self.alignment = alignment
    self.flash_min_seq_length = flash_min_seq_length
    if pos_embed_seq_len is not None:
      self.pos_embed = nnx.Param(jnp.zeros((1, pos_embed_seq_len, in_features), dtype=dtype))
    else:
      self.pos_embed = nnx.data(None)

  def __call__(self, encoder_hidden_states_image: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden_states = encoder_hidden_states_image
    B, current_seq_len, _ = hidden_states.shape

    if self.pos_embed is not None:
      pe_len = self.pos_embed.value.shape[1]
      add_len = min(current_seq_len, pe_len)
      # Apply pos_embed to the original sequence length
      hidden_states = hidden_states.at[:, :add_len, :].add(self.pos_embed.value[:, :add_len, :])
      if current_seq_len > pe_len:
        max_logging.log(f"[WARN] Input seq_len {current_seq_len} > pos_embed len {pe_len}")

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.ff(hidden_states)
    hidden_states = self.norm2(hidden_states)
    # hidden_states shape: (B, current_seq_len, out_features)
    B, current_seq_len, D_out = hidden_states.shape
    use_flash_attn = current_seq_len >= self.flash_min_seq_length

    if use_flash_attn:
      # --- Dynamic Padding to nearest multiple of self.alignment ---
      num_blocks = (current_seq_len + self.alignment - 1) // self.alignment
      target_seq_len = num_blocks * self.alignment
    else:
      target_seq_len = current_seq_len

    # Create attention mask: 1 for real tokens, 0 for padded tokens
    attention_mask = jnp.ones((B, current_seq_len), dtype=jnp.int32)

    if current_seq_len < target_seq_len:
      padding_size = target_seq_len - current_seq_len
      padding = jnp.zeros((B, padding_size, D_out), dtype=hidden_states.dtype)
      hidden_states = jnp.concatenate([hidden_states, padding], axis=1)

      # Extend mask with zeros for padded positions
      padding_mask = jnp.zeros((B, padding_size), dtype=jnp.int32)
      attention_mask = jnp.concatenate([attention_mask, padding_mask], axis=1)
    if not use_flash_attn:
      attention_mask = None
    return hidden_states, attention_mask


class NNXPixArtAlphaTextProjection(nnx.Module):
  """Two-layer MLP projecting text encoder features to the transformer width."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_features: int,
      hidden_size: int,
      out_features: int = None,
      act_fn: str = "gelu_tanh",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      sharding_specs: Optional[Any] = None,
  ):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    if out_features is None:
      out_features = hidden_size

    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_features,
        out_features=hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_1_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_1_bias),
    )
    self.act_1 = get_activation(act_fn)

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=hidden_size,
        out_features=out_features,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_2_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_2_bias),
    )

  def __call__(self, caption):
    hidden_states = self.linear_1(caption)
    hidden_states = self.act_1(hidden_states)
    hidden_states = self.linear_2(hidden_states)
    return hidden_states


class PixArtAlphaTextProjection(nn.Module):
  """Projects caption embeddings.

  Also handles dropout for classifier-free guidance. Adapted from
  https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
  """

  hidden_size: int
  out_features: int = None
  act_fn: str = "gelu_tanh"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, caption):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_1",
    )(caption)
    if self.act_fn == "gelu_tanh":
      act_1 = nn.gelu
    elif self.act_fn == "silu":
      act_1 = nn.swish
    else:
      raise ValueError(f"Unknown activation function: {self.act_fn}")
    hidden_states = act_1(hidden_states)

    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_2",
    )(hidden_states)
    return hidden_states


# =============================================================================
# 4. Wan DiT Transformer Backbone
# =============================================================================

SKIP_GRADIENT_CHECKPOINT_KEY = "skip"


class GradientCheckpointType(Enum):
  """Gradient checkpointing (remat) policy selector."""

  NONE = auto()
  FULL = auto()
  MATMUL_WITHOUT_BATCH = auto()
  OFFLOAD_MATMUL_WITHOUT_BATCH = auto()
  CUSTOM = auto()
  HIDDEN_STATE_WITH_OFFLOAD = auto()
  FLUX_OPTIMIZED = auto()

  @classmethod
  def from_str(cls, s: Optional[str] = None) -> "GradientCheckpointType":
    if s is None:
      s = "none"
    return GradientCheckpointType[s.upper()]

  def to_jax_policy(
      self,
      names_which_can_be_saved: Optional[list] = None,
      names_which_can_be_offloaded: Optional[list] = None,
      block_type: Optional[str] = None,
  ):
    """Return the JAX checkpoint policy for this checkpoint type."""
    names_which_can_be_saved = [] if names_which_can_be_saved is None else names_which_can_be_saved
    names_which_can_be_offloaded = [] if names_which_can_be_offloaded is None else names_which_can_be_offloaded
    match self:
      case GradientCheckpointType.NONE:
        return SKIP_GRADIENT_CHECKPOINT_KEY
      case GradientCheckpointType.FULL:
        return None
      case GradientCheckpointType.OFFLOAD_MATMUL_WITHOUT_BATCH:
        return cp.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host")
      case GradientCheckpointType.CUSTOM:
        return cp.save_and_offload_only_these_names(
            names_which_can_be_saved=names_which_can_be_saved,
            names_which_can_be_offloaded=names_which_can_be_offloaded,
            offload_src="device",
            offload_dst="pinned_host",
        )
      case GradientCheckpointType.HIDDEN_STATE_WITH_OFFLOAD:
        return jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "hidden_states",
                "self_attn",
                "cross_attn",
            ],
            offload_src="device",
            offload_dst="pinned_host",
        )
      case GradientCheckpointType.MATMUL_WITHOUT_BATCH:
        return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      case GradientCheckpointType.FLUX_OPTIMIZED:
        if block_type == "double":
          return cp.save_any_names_but_these("img_qkv_proj", "txt_qkv_proj")
        elif block_type == "single":
          return cp.save_only_these_names("attn_output", "lin1_norm_hidden_states")
        else:
          return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

  def apply(
      self,
      module: nnx.Module,
      names_which_can_be_saved: Optional[list] = None,
      names_which_can_be_offloaded: Optional[list] = None,
      static_argnums=(),
      prevent_cse: bool = False,
  ) -> nnx.Module:
    """Wrap `module` with nnx.remat using this policy (or return it unchanged)."""
    names_which_can_be_saved = [] if names_which_can_be_saved is None else names_which_can_be_saved
    names_which_can_be_offloaded = [] if names_which_can_be_offloaded is None else names_which_can_be_offloaded
    policy = self.to_jax_policy(names_which_can_be_saved, names_which_can_be_offloaded)
    if policy == SKIP_GRADIENT_CHECKPOINT_KEY:
      return module
    return nnx.remat(
        module,
        prevent_cse=prevent_cse,
        policy=policy,
        static_argnums=static_argnums,
    )


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  """Return per-axis (t, h, w) complex rotary frequencies."""
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freq = get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32, use_real=False)
    freqs.append(freq)
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6

  dims = [t_size, hw_size, hw_size]

  # Calculate split indices as a static list of integers
  cumulative_sizes = np.cumsum(dims)
  split_indices = cumulative_sizes[:-1].tolist()
  freqs_split = jnp.split(freqs, split_indices, axis=1)
  return freqs_split


class WanRotaryPosEmbed(nnx.Module):
  """3D (frame, height, width) rotary positional embedding for Wan."""

  def __init__(
      self,
      attention_head_dim: int,
      patch_size: Tuple[int, int, int],
      max_seq_len: int,
      theta: float = 10000.0,
  ):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    _, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)

    freqs_f = jnp.expand_dims(jnp.expand_dims(freqs_split[0][:ppf], axis=1), axis=1)
    freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

    freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
    freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

    freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
    freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

    freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
    freqs_final = jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))
    return freqs_final


class WanTimeTextImageEmbedding(nnx.Module):
  """Timestep, text and optional image condition embeddings for Wan."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      time_freq_dim: int,
      time_proj_dim: int,
      text_embed_dim: int,
      image_embed_dim: Optional[int] = None,
      pos_embed_seq_len: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      flash_min_seq_length: int = 4096,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = NNXTimestepEmbedding(
        rngs=rngs,
        in_channels=time_freq_dim,
        time_embed_dim=dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=time_proj_dim,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
        rngs=rngs,
        in_features=text_embed_dim,
        hidden_size=dim,
        act_fn="gelu_tanh",
    )

    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = NNXWanImageEmbedding(
          rngs=rngs,
          in_features=image_embed_dim,
          out_features=dim,
          pos_embed_seq_len=pos_embed_seq_len,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          flash_min_seq_length=flash_min_seq_length,
      )

  def __call__(
      self,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      skip_embeddings: bool = False,
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))

    if not skip_embeddings:
      encoder_hidden_states = self.text_embedder(encoder_hidden_states)
      encoder_attention_mask = None
      if encoder_hidden_states_image is not None:
        (
            encoder_hidden_states_image,
            encoder_attention_mask,
        ) = self.image_embedder(encoder_hidden_states_image)
    else:
      encoder_attention_mask = None
      if (
          encoder_hidden_states_image is not None
          and encoder_hidden_states_image.shape[-1] != encoder_hidden_states.shape[-1]
      ):
        img_dim = encoder_hidden_states_image.shape[-1]
        text_dim = encoder_hidden_states.shape[-1]
        if img_dim < text_dim:
          pad_shape = (
              encoder_hidden_states_image.shape[0],
              encoder_hidden_states_image.shape[1],
              text_dim - img_dim,
          )
          encoder_hidden_states_image = jnp.concatenate(
              [
                  encoder_hidden_states_image,
                  jnp.zeros(pad_shape, dtype=encoder_hidden_states_image.dtype),
              ],
              axis=-1,
          )
        else:
          pad_shape = (
              encoder_hidden_states.shape[0],
              encoder_hidden_states.shape[1],
              img_dim - text_dim,
          )
          encoder_hidden_states = jnp.concatenate(
              [
                  encoder_hidden_states,
                  jnp.zeros(pad_shape, dtype=encoder_hidden_states.dtype),
              ],
              axis=-1,
          )

    return (
        temb,
        timestep_proj,
        encoder_hidden_states,
        encoder_hidden_states_image,
        encoder_attention_mask,
    )


class ApproximateGELU(nnx.Module):
  r"""The approximate form of the Gaussian Error Linear Unit (GELU).

  For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim_in: int,
      dim_out: int,
      bias: bool,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.proj = nnx.Linear(
        rngs=rngs,
        in_features=dim_in,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    with jax.named_scope("gelu"):
      x = self.proj(x)
    return nnx.gelu(x)


class WanFeedForward(nnx.Module):
  """Wan transformer feed-forward MLP with a configurable activation."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      dropout: float = 0.0,
      activation_fn: str = "geglu",
      final_dropout: bool = False,
      inner_dim: int = None,
      bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      enable_jax_named_scopes: bool = False,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.act_fn = nnx.data(None)
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(
          rngs=rngs,
          dim_in=dim,
          dim_out=inner_dim,
          bias=bias,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
      )
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")

    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "mlp",
                "embed",
            ),
        ),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
  ) -> jax.Array:
    hidden_states = self.act_fn(hidden_states)  # Output is (4, 75600, 13824)
    hidden_states = checkpoint_name(hidden_states, "ffn_activation")
    if self.drop_out.rate > 0:
      hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    with jax.named_scope("proj_out"):
      return self.proj_out(hidden_states)  # output is (4, 75600, 5120)


class WanTransformerBlock(nnx.Module):
  """Wan DiT block: modulated self-attention, cross-attention and feed-forward."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      added_kv_proj_dim: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      mask_padding_tokens: bool = True,
      enable_jax_named_scopes: bool = False,
      attention_config: Optional[dict] = None,
  ):
    self.enable_jax_named_scopes = enable_jax_named_scopes
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        "ulysses_attention_chunks": 1,
        **(attention_config or {}),
    }

    # 1. Self-attention
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=True,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="self_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        attention_config=attention_config,
    )

    # 1. Cross-attention
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        added_kv_proj_dim=added_kv_proj_dim,
        image_seq_len=image_seq_len,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="cross_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
        attention_config=attention_config,
    )
    assert cross_attn_norm is True
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)

    # 3. Feed-forward
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        dropout=dropout,
        enable_jax_named_scopes=enable_jax_named_scopes,
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      cached_kv: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
  ):
    with self.conditional_named_scope("transformer_block"):
      # Support both global [B, 6, dim] and per-token [B, seq_len, 6, dim] temb.
      # Per-token temb is used by TI2V where first-frame tokens have timestep=0.
      if temb.ndim == 4:  # Per-token: [B, seq_len, 6, dim]
        adaln = jnp.expand_dims(self.adaln_scale_shift_table, 0)  # [1, 1, 6, dim]
        combined = adaln + temb.astype(jnp.float32)  # [B, seq_len, 6, dim]
        parts = jnp.split(combined, 6, axis=2)
        shift_msa = parts[0].squeeze(2)
        scale_msa = parts[1].squeeze(2)
        gate_msa = parts[2].squeeze(2)
        c_shift_msa = parts[3].squeeze(2)
        c_scale_msa = parts[4].squeeze(2)
        c_gate_msa = parts[5].squeeze(2)
      else:  # Global: [B, 6, dim]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = jnp.split(
            (self.adaln_scale_shift_table + temb.astype(jnp.float32)),
            6,
            axis=1,
        )
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
      hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
      hidden_states = checkpoint_name(hidden_states, "hidden_states")
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_kv"))
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)

      # 1. Self-attention
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("self_attn_attn"):
          attn_output = self.attn1(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=norm_hidden_states,
              rotary_emb=rotary_emb,
              deterministic=deterministic,
              rngs=rngs,
          )
        with self.conditional_named_scope("self_attn_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)

      # 2. Cross-attention
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          attn_output = self.attn2(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=encoder_hidden_states,
              deterministic=deterministic,
              rngs=rngs,
              encoder_attention_mask=encoder_attention_mask,
              cached_kv=cached_kv,
          )
        with self.conditional_named_scope("cross_attn_residual"):
          hidden_states = hidden_states + attn_output

      # 3. Feed-forward
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(
              hidden_states.dtype
          )
      return hidden_states

  def compute_kv(
      self,
      encoder_hidden_states: jax.Array,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Dict[str, Tuple[jax.Array, jax.Array]]:
    return self.attn2.compute_kv(encoder_hidden_states, encoder_attention_mask)


class WanModel(nnx.Module):
  """Wan diffusion transformer backbone (patch embed, transformer blocks, output projection)."""

  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size: Tuple[int] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      eps: float = 1e-6,
      image_dim: Optional[int] = None,
      added_kv_proj_dim: Optional[int] = None,
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: Optional[list] = None,
      names_which_can_be_offloaded: Optional[list] = None,
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
      attention_config: Optional[dict] = None,
  ):
    names_which_can_be_saved = [] if names_which_can_be_saved is None else names_which_can_be_saved
    names_which_can_be_offloaded = [] if names_which_can_be_offloaded is None else names_which_can_be_offloaded
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.in_channels = in_channels
    self.patch_size = tuple(patch_size)
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes
    attention_config = {
        "use_base2_exp": False,
        "use_experimental_scheduler": False,
        "ulysses_shards": -1,
        "ulysses_attention_chunks": 1,
        **(attention_config or {}),
    }

    # 1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
        flash_min_seq_length=flash_min_seq_length,
    )

    # 3. Transformer blocks
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(
        in_axes=0,
        out_axes=0,
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )
    def init_block(rngs):
      return WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          dropout=dropout,
          mask_padding_tokens=mask_padding_tokens,
          enable_jax_named_scopes=enable_jax_named_scopes,
          added_kv_proj_dim=added_kv_proj_dim,
          image_seq_len=image_seq_len,
          attention_config=attention_config,
      )

    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved
    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(num_layers):
        block = WanTransformerBlock(
            rngs=rngs,
            dim=inner_dim,
            ffn_dim=ffn_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            image_seq_len=image_seq_len,
            flash_min_seq_length=flash_min_seq_length,
            flash_block_sizes=flash_block_sizes,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            attention=attention,
            dropout=dropout,
            mask_padding_tokens=mask_padding_tokens,
            enable_jax_named_scopes=enable_jax_named_scopes,
            attention_config=attention_config,
        )
        blocks.append(block)
      self.blocks = nnx.data(blocks)

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "embed")),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def compute_kv_cache(
      self,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      timestep: Optional[jax.Array] = None,
  ) -> Tuple[Dict[str, Tuple[jax.Array, jax.Array]], Optional[jax.Array]]:
    """Precompute per-layer cross-attention K/V for the text/image conditioning."""
    if timestep is None:
      batch_size = encoder_hidden_states.shape[0]
      timestep = jnp.zeros((batch_size,), dtype=jnp.int32)

    with self.conditional_named_scope("condition_embedder"):
      (
          _,
          _,
          encoder_hidden_states,
          encoder_hidden_states_image,
          encoder_attention_mask,
      ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)
      if encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)

    if self.scan_layers:

      @nnx.vmap(
          in_axes=(0, None, None),
          out_axes=0,
          transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
      )
      def _compute_kv(block, enc_states, enc_mask):
        return block.compute_kv(enc_states, enc_mask)

      kv_cache = _compute_kv(self.blocks, encoder_hidden_states, encoder_attention_mask)
    else:
      kv_cache_list = []
      for block in self.blocks:
        kv_cache_list.append(block.compute_kv(encoder_hidden_states, encoder_attention_mask))
      keys = kv_cache_list[0].keys()
      kv_cache = {}
      for k in keys:
        k_list = [d[k][0] for d in kv_cache_list]
        v_list = [d[k][1] for d in kv_cache_list]
        kv_cache[k] = (jnp.stack(k_list, axis=0), jnp.stack(v_list, axis=0))

    return kv_cache, encoder_attention_mask

  @jax.named_scope("WanModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      skip_blocks: Optional[jax.Array] = None,
      cached_residual: Optional[jax.Array] = None,
      return_residual: bool = False,
      kv_cache: Optional[Dict[str, Tuple[jax.Array, jax.Array]]] = None,
      rotary_emb: Optional[jax.Array] = None,
      encoder_attention_mask: Optional[jax.Array] = None,
  ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], Dict[str, jax.Array]]:
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      if rotary_emb is None:
        rotary_emb = self.rope(hidden_states)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    per_token_t = timestep.ndim == 2  # [B, seq_len] for TI2V
    with self.conditional_named_scope("condition_embedder"):
      if per_token_t:
        # Per-token timestep: process time and text embeddings separately.
        # This matches the official WAN 2.2 TI2V pipeline where first-frame
        # tokens receive timestep=0 (clean) and other tokens receive timestep=t.
        bt, sl = timestep.shape
        t_flat = timestep.reshape(-1)  # [B*seq_len]
        t_sinusoidal = self.condition_embedder.timesteps_proj(t_flat)  # [B*sl, freq_dim]
        t_sinusoidal = t_sinusoidal.reshape(bt, sl, -1)  # [B, sl, freq_dim]
        temb = self.condition_embedder.time_embedder(t_sinusoidal)  # [B, sl, dim]
        with jax.named_scope("time_proj"):
          timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))  # [B, sl, dim*6]
        timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)  # [B, sl, 6, dim]
        # Text processing
        if kv_cache is None:
          encoder_hidden_states_out = self.condition_embedder.text_embedder(encoder_hidden_states)
        else:
          encoder_hidden_states_out = encoder_hidden_states
        encoder_hidden_states_image_out = None
        encoder_attention_mask_out = None
      else:
        (
            temb,
            timestep_proj,
            encoder_hidden_states_out,
            encoder_hidden_states_image_out,
            encoder_attention_mask_out,
        ) = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            skip_embeddings=(kv_cache is not None),
        )
        timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_attention_mask is None:
      encoder_attention_mask = encoder_attention_mask_out

    if encoder_hidden_states_image_out is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image_out, encoder_hidden_states_out], axis=1)
      if kv_cache is None and encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (
                encoder_hidden_states.shape[0],
                encoder_hidden_states.shape[1] - encoder_hidden_states_image_out.shape[1],
            ),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)
      encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)
    else:
      encoder_hidden_states = encoder_hidden_states_out.astype(hidden_states.dtype)

    def _run_all_blocks(h):
      if self.scan_layers:

        def scan_fn(carry, block_input):
          hidden_states_carry, rngs_carry = carry
          if kv_cache is not None:
            block, layer_kv_cache = block_input
          else:
            block = block_input
            layer_kv_cache = None

          hidden_states = block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask,
              cached_kv=layer_kv_cache,
          )
          new_carry = (hidden_states, rngs_carry)
          return new_carry, None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn,
            self.names_which_can_be_saved,
            self.names_which_can_be_offloaded,
            prevent_cse=not self.scan_layers,
        )
        initial_carry = (h, rngs)

        if kv_cache is not None:
          scan_input = (self.blocks, kv_cache)
        else:
          scan_input = self.blocks

        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_layers,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, scan_input)

        h_out, _ = final_carry
      else:
        h_out = h
        for i, block in enumerate(self.blocks):
          layer_kv_cache = None
          if kv_cache is not None:
            # Lambda is applied immediately within this iteration.
            layer_kv_cache = jax.tree.map(lambda x: x[i], kv_cache)  # pylint: disable=cell-var-from-loop

          def layer_forward(hidden_states, l_kv):
            # Called within this iteration, so `block` is bound correctly.
            return block(  # pylint: disable=cell-var-from-loop
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                deterministic,
                rngs,
                encoder_attention_mask=encoder_attention_mask,
                cached_kv=l_kv,
            )

          rematted_layer_forward = self.gradient_checkpoint.apply(
              layer_forward,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          h_out = rematted_layer_forward(h_out, layer_kv_cache)
      return h_out

    hidden_states_before_blocks = hidden_states

    if skip_blocks:
      if cached_residual is None:
        raise ValueError("cached_residual must be provided when skip_blocks is True")
      hidden_states = hidden_states + cached_residual
    else:
      hidden_states = _run_all_blocks(hidden_states)

    residual_x = hidden_states - hidden_states_before_blocks

    if per_token_t:
      # temb: [B, seq_len, dim] — per-token modulation for final head
      combined_head = jnp.expand_dims(self.scale_shift_table, 0) + jnp.expand_dims(temb, 2)  # [B, sl, 2, dim]
      shift, scale = jnp.split(combined_head, 2, axis=2)
      shift = shift.squeeze(2)  # [B, sl, dim]
      scale = scale.squeeze(2)  # [B, sl, dim]
    else:
      shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if return_residual:
      return hidden_states, residual_x
    return hidden_states
