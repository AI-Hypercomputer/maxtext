# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import contextlib
import functools
import math
from typing import Optional, Callable, Tuple, Any, Dict
import flax.linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
try:
  from maxdiffusion.kernels.splash_attention import splash_attention_mask as tokamax_splash_attention_mask
  from maxdiffusion.kernels.splash_attention import splash_attention_kernel as tokamax_splash_attention_kernel
  from maxdiffusion.kernels.splash_attention import ring_attention_kernel as tokamax_ring_attention_kernel
  from maxdiffusion.kernels.splash_attention import base as tokamax_splash_base
  from ..kernels import custom_splash_attention as custom_splash
except ImportError:
  tokamax_splash_attention_mask = None
  tokamax_splash_attention_kernel = None
  tokamax_ring_attention_kernel = None
  tokamax_splash_base = None
  custom_splash = None
from einops import rearrange
from maxtext.common import common_types
from maxtext.utils import max_logging
from maxtext.utils.max_utils import get_tpu_type, TpuType, safe_getattr

from maxtext.layers import quantizations
from maxtext.models.wan.wan_utils import get_activation

LOG2E = math.log2(math.e)

Array = common_types.Array
Mesh = common_types.Mesh
DType = common_types.DType
BlockSizes = common_types.BlockSizes


AxisNames = common_types.AxisNames
CONTEXT = common_types.CONTEXT
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV
EMBED = common_types.EMBED
Quant = quantizations.AqtQuantization

SELF_ATTN_HEAD = common_types.SELF_ATTN_HEAD
SELF_ATTN_Q_LENGTH = common_types.SELF_ATTN_Q_LENGTH
SELF_ATTN_KV_LENGTH = common_types.SELF_ATTN_KV_LENGTH
CROSS_ATTN_HEAD = common_types.CROSS_ATTN_HEAD
CROSS_ATTN_Q_LENGTH = common_types.CROSS_ATTN_Q_LENGTH
CROSS_ATTN_KV_LENGTH = common_types.CROSS_ATTN_KV_LENGTH

INTERNAL_RING_AXIS = "ring"
INTERNAL_ULYSSES_AXIS = "ulysses"


def _coerce_tokamax_block_sizes(block_sizes):
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


def _reshape_data_from_cudnn_flash(tensor):
  # reshapes from [b, s, h, d] back to [b, s, h * d]
  return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


def _reshape_data_for_cudnn_flash(tensor, heads):
  # reshapes from [b, s, h * d] to [b, s, h, d] (input format to flash format)
  if len(tensor.shape) == 3:
    batch, seq, dim_head = tensor.shape
    tensor = tensor.reshape(batch, seq, heads, dim_head // heads)
  else:
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  return tensor


def _reshape_batch_dim_to_heads(tensor, heads):
  batch_size, seq_len, dim = tensor.shape
  head_size = heads
  tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
  tensor = jnp.transpose(tensor, (0, 2, 1, 3))
  reshaped_tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
  return nn.with_logical_constraint(reshaped_tensor, (BATCH, LENGTH, HEAD))


def _reshape_heads_to_batch_dim(tensor, heads):
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
  b, h, s, d = tensor.shape
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
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
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
  """
  Reshapes tensors for pallas flash attention adding padding to both seq_len and head_dim.
  Pads seq_len to a multiple of flash_block_size, and ensures the resulting number of
  blocks is divisible by the number of shards.
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
  raise ValueError(f"Flash attention expects rank-3 or rank-4 inputs, got rank {tensor.ndim}.")


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
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
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
              jnp.zeros((attention_mask.shape[0], kv_padded_len - key_seq_len), jnp.int32),
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
    raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
  if attention_mask.shape[0] == 1 and batch_size != 1:
    attention_mask = jnp.broadcast_to(attention_mask, (batch_size, attention_mask.shape[1]))
  elif attention_mask.shape[0] != batch_size:
    raise ValueError(
        f"attention_mask batch dimension must be 1 or match the attention batch ({batch_size}), "
        f"got {attention_mask.shape[0]}."
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
        "Ulysses attention requires the number of heads to be divisible by the Ulysses shard count, "
        f"got heads={num_heads} and ulysses_shards={ulysses_shards}."
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

      bsizes = custom_splash._BlockSizes(
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
          save_residuals=True if "ring" in attention_kernel else False,
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
        (m_final, l_final, o_final, _, _), _ = jax.lax.scan(
            ring_scan_body, initial_carry, None, length=num_context_shards - 1
        )

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
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_flash_attention = shard_map.shard_map(
        lambda q, k, v: wrap_flash_attention(q, k, v, None),
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names),
        out_specs=q_axis_names,
        check_rep=False,
    )
    x = sharded_flash_attention(query, key, value)
  else:
    sharded_flash_attention = shard_map.shard_map(
        wrap_flash_attention,
        mesh=mesh,
        in_specs=(q_axis_names, kv_axis_names, kv_axis_names, mask_axis_names),
        out_specs=q_axis_names,
        check_rep=False,
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
        "The custom dense splash kernel (use_custom_kernel) does not support attention_mask "
        "(it only handles padding via orig_seq_len); got a non-None attention_mask."
    )
  num_heads = query.shape[1]
  # Ulysses only redistributes existing heads across the context mesh; unlike
  # the earlier draft, we fail fast instead of padding synthetic heads.
  if num_heads % num_shards != 0:
    raise ValueError(
        "Ulysses attention requires the number of heads to be divisible by the context shard count, "
        f"got heads={num_heads} and context_shards={num_shards}."
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
            "The custom dense splash kernel (use_custom_kernel) does not support attention_mask "
            "(it only handles padding via orig_seq_len); got a non-None attention_mask."
        )
      bq, bkv, bkv_compute, bkv_compute_in, heads_per_tile, vmem_limit_bytes = _extract_custom_block_sizes(flash_block_sizes)

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
        fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
        mk_arr = jnp.stack([mk_h, fixed_ok])  # (2, local_heads)

      bsizes = custom_splash._BlockSizes(
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

      mask = splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

      segment_ids = _build_padding_segment_ids(query_seq_len, query.shape[2], key_seq_len, key.shape[2], attention_mask)
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
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
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
    raise ValueError(f"Ulysses ring attention requires mesh axis {context_axis!r}, got mesh axes {mesh.shape}.")

  num_context_shards = mesh.shape[context_axis]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("Ulysses ring attention requires ulysses_shards to be set from config or command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the requested Ulysses shard count to divide the context shard count, "
        f"got context_shards={num_context_shards} and ulysses_shards={num_ulysses_shards}."
    )
  if heads % num_ulysses_shards != 0:
    raise ValueError(
        "Ulysses ring attention requires the number of heads to be divisible by the requested Ulysses shard count, "
        f"got heads={heads} and ulysses_shards={num_ulysses_shards}."
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
        "Warning, batch dimension should be shardable among the devices in data and fsdp"
        f" axis, batch dimension: {query.shape[0]}, devices_in_batch_sharding: {devices_in_batch_sharding}"
    )
  if attention_mask is None:
    sharded_ulysses_ring_attention = jax.shard_map(
        lambda q, k, v: wrap_ulysses_ring_attention(q, k, v, None),
        mesh=internal_mesh,
        in_specs=(internal_q_axis_names, internal_kv_axis_names, internal_kv_axis_names),
        out_specs=internal_q_axis_names,
        check_vma=False,
    )

    def run_ulysses_ring_attention(q, k, v):
      return sharded_ulysses_ring_attention(q, k, v)

  else:
    sharded_ulysses_ring_attention = jax.shard_map(
        wrap_ulysses_ring_attention,
        mesh=internal_mesh,
        in_specs=(internal_q_axis_names, internal_kv_axis_names, internal_kv_axis_names, internal_mask_axis_names),
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
  axis innermost, so the Ulysses all-to-all stays INTRA-chip and the ring rotates
  ACROSS chips. The per-shard attention is our custom splash kernel
  (`make_custom_ring_attention`), not the tokamax_ring kernel main uses.

    1. all-to-all over the (intra-chip) Ulysses axis: trade sequence for heads;
    2. ring (full ppermute) over the (cross-chip) ring axis, online-softmax merge;
    3. all-to-all back to restore the sequence-sharded / full-heads layout.

  U = ulysses_shards (from config); R = context // U. U=context -> pure
  Ulysses, U=1 -> pure Ring (all on the same custom kernel).
  """
  if attention_mask is not None:
    raise NotImplementedError(
        "ulysses_ring_custom does not support attention_mask (the custom splash kernels only "
        "handle padding via orig_seq_len); got a non-None attention_mask."
    )
  axis_name = "context"
  num_context_shards = mesh.shape[axis_name]
  num_ulysses_shards = ulysses_shards
  if num_ulysses_shards <= 0:
    raise ValueError("ulysses_ring_custom requires ulysses_shards to be set from config or command line.")
  if num_context_shards % num_ulysses_shards != 0:
    raise ValueError(
        f"ulysses_ring_custom requires ulysses_shards to divide the context shard count, "
        f"got context_shards={num_context_shards} and ulysses_shards={num_ulysses_shards}."
    )
  num_ring_shards = num_context_shards // num_ulysses_shards

  query, orig_q_seq_len = _reshape_data_for_flash(query, heads, num_context_shards)
  key, _ = _reshape_data_for_flash(key, heads, num_context_shards)
  value, _ = _reshape_data_for_flash(value, heads, num_context_shards)
  num_heads = query.shape[1]
  if num_heads % num_ulysses_shards != 0:
    raise ValueError(f"Ulysses+Ring requires heads divisible by U={num_ulysses_shards}, got heads={num_heads}.")

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
      fixed_ok = (qn_max * mk_h <= custom_splash._FIXED_M_SAFE_BOUND).astype(jnp.float32)
      mk_arr = jnp.stack([mk_h, fixed_ok])  # (2, local_heads)

    bsizes = custom_splash._BlockSizes(
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


def _cudnn_flash_attention(query: Array, key: Array, value: Array, heads: int, mesh: Mesh, dpa_layer: Callable) -> Array:
  """CUDNN Flash Attention with Transformer Engine.
  1. Stable API, supports GQA
  2. Supports head_dim till 128; head_dim=256 support will be added soon
  """
  # These imports are only meant to work in a GPU build.
  # copied from tpu_flash_attention
  query = _reshape_data_for_cudnn_flash(query, heads)
  key = _reshape_data_for_cudnn_flash(key, heads)
  value = _reshape_data_for_cudnn_flash(value, heads)

  query = nn.with_logical_constraint(query, (BATCH, LENGTH, HEAD, D_KV))
  key = nn.with_logical_constraint(key, (BATCH, LENGTH, HEAD, D_KV))
  value = nn.with_logical_constraint(value, (BATCH, LENGTH, HEAD, D_KV))

  out = dpa_layer(query, key, value, mask=None)
  return _reshape_data_from_cudnn_flash(out)


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
  re-normalizes across shards, so per-shard bounds stay correct."""
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
  on a non-wrapping ring axis. Same USP split as ulysses_ring_custom otherwise."""
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


@register_kernel("cudnn_flash_te")
def cudnn_flash_te_kernel(q, k, v, context):
  return _cudnn_flash_attention(q, k, v, context["heads"], context["mesh"], context["dpa_layer"])


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
    dpa_layer: Callable,
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
      raise ValueError(f"attention_mask must have shape [batch, kv_length], got {attention_mask.shape}.")
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
      "dpa_layer": dpa_layer,
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
  r"""
  Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
  https://github.com/AminRezaei0x443/memory-efficient-attention

  Args:
      query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
      key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
      value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
      precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
          numerical precision for computation
      query_chunk_size (`int`, *optional*, defaults to 1024):
          chunk size to divide query array value must divide query_length equally without remainder
      key_chunk_size (`int`, *optional*, defaults to 4096):
          chunk size to divide key and value array value must divide key_value_length equally without remainder

  Returns:
      (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
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
    self.dpa_layer = None
    self.use_base2_exp = use_base2_exp
    self.use_experimental_scheduler = use_experimental_scheduler
    self.ulysses_shards = ulysses_shards
    self.ulysses_attention_chunks = ulysses_attention_chunks
    if attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      jax.config.update("jax_use_shardy_partitioner", False)

      dpa_layer = DotProductAttention(
          head_dim=dim_head,
          num_attention_heads=heads,
          num_gqa_groups=heads,
          attn_mask_type="no_mask",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
          attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
          # attention_dropout=self.dropout_rate,
          dropout_rng_name="aqt",
          dtype=dtype,
          qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
          scale_factor=scale,
          transpose_batch_sequence=False,
      )
      variables = {}
      self.dpa_layer = functools.partial(dpa_layer.apply, variables)

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
        dpa_layer=self.dpa_layer,
        mask_padding_tokens=self.mask_padding_tokens,
        residual_checkpoint_name=self.residual_checkpoint_name,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp if hasattr(self, "use_base2_exp") else False,
        use_experimental_scheduler=self.use_experimental_scheduler if hasattr(self, "use_experimental_scheduler") else False,
        ulysses_shards=(self.ulysses_shards if hasattr(self, "ulysses_shards") else -1),
        ulysses_attention_chunks=(self.ulysses_attention_chunks if hasattr(self, "ulysses_attention_chunks") else 1),
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )


class AttentionOp(nn.Module):
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

  def setup(self):
    self.dpa_layer = None
    if self.attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      jax.config.update("jax_use_shardy_partitioner", False)

      dpa_layer = DotProductAttention(
          head_dim=self.dim_head,
          num_attention_heads=self.heads,
          num_gqa_groups=self.heads,
          attn_mask_type="no_mask",  # 'no_mask', 'padding', 'causal', or 'padding_causal'
          attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
          # attention_dropout=self.dropout_rate,
          dropout_rng_name="aqt",
          dtype=self.dtype,
          # float32_logits=self.float32_logits,
          qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
          scale_factor=self.scale,
          transpose_batch_sequence=False,
      )
      variables = {}
      self.dpa_layer = functools.partial(dpa_layer.apply, variables)

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
        dpa_layer=self.dpa_layer,
        attention_mask=attention_mask,
        use_base2_exp=self.use_base2_exp,
        use_experimental_scheduler=self.use_experimental_scheduler,
        ulysses_shards=self.ulysses_shards,
        ulysses_attention_chunks=self.ulysses_attention_chunks,
        is_causal=self.is_causal,
        preserve_asymmetric_block_sizes=preserve_asymmetric_block_sizes,
    )


class FlaxWanAttention(nnx.Module):

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

    if attention_kernel in {"flash", "cudnn_flash_te"} and mesh is None:
      raise ValueError(f"The flash attention kernel requires a value for mesh, but mesh is {self.mesh}")
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
        cross_attention_remapped_to_flash or attention_kernel in ("flash", "tokamax_flash", "cudnn_flash_te")
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
        attention_kernel in ("ulysses_ring_custom", "ulysses_ring_custom_bidir", "ulysses_ring_custom_fixed_m")
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


