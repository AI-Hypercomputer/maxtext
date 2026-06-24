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
# ==============================================================================

"""Shared helpers for Tokamax ring-family Splash attention kernels."""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask_info as mask_info_lib

MaskInfo = mask_info_lib.MaskInfo


def dynamic_slice_mask_info(mask_info: MaskInfo, kv_shard_idx: jax.Array, ring_size: int) -> MaskInfo:
  """Slices MaskInfo for the current ring step."""

  def slice_if_exists(arr: jax.Array | None):
    if arr is None:
      return None

    shard_len = arr.shape[-1] // ring_size
    start_idx = kv_shard_idx * shard_len
    return lax.dynamic_slice_in_dim(arr, start_idx, shard_len, axis=-1)

  return MaskInfo(
      mask_next=slice_if_exists(mask_info.mask_next),
      active_rows=slice_if_exists(mask_info.active_rows),
      active_cols=slice_if_exists(mask_info.active_cols),
      num_active_blocks=slice_if_exists(mask_info.num_active_blocks),
      block_mask=slice_if_exists(mask_info.block_mask),
      partial_mask_blocks=mask_info.partial_mask_blocks,  # partial mask blocks are global
      q_sequence=mask_info.q_sequence,  # Q sequence stays stationary
  )


def offset_q_sequence_for_kv_shard(mask_info: MaskInfo, kv_shard_idx: jax.Array, kv_seq_len: int) -> MaskInfo:
  """Converts lazy mask Q ids to the current local KV coordinate frame."""
  if mask_info.q_sequence is None:
    return mask_info

  kv_shard_offset = jnp.asarray(kv_shard_idx, dtype=mask_info.q_sequence.dtype) * kv_seq_len
  return mask_info._replace(q_sequence=mask_info.q_sequence - kv_shard_offset)


def has_no_active_blocks(mask_info: MaskInfo) -> jax.Array | bool:
  """Returns whether the mask metadata has no active compute blocks."""
  if mask_info.num_active_blocks is None:
    return False
  has_no_scheduled_blocks = jnp.all(mask_info.num_active_blocks == 0)
  if mask_info.block_mask is None:
    return has_no_scheduled_blocks
  # Dynamic mask metadata is padded with -1 after num_active_blocks.
  block_ids = jnp.arange(mask_info.block_mask.shape[-1])
  active_blocks = block_ids < jnp.max(mask_info.num_active_blocks)
  block_mask = jnp.where(active_blocks, mask_info.block_mask, 0)
  has_no_compute_blocks = jnp.all(block_mask == 0)
  return jnp.logical_or(has_no_scheduled_blocks, has_no_compute_blocks)


def has_empty_attention_rows(logsumexp: jax.Array, max_logits: jax.Array, mask_value: float) -> jax.Array:
  mask_value = jnp.asarray(mask_value, dtype=logsumexp.dtype)
  return jnp.logical_and(logsumexp == mask_value, max_logits == mask_value)


def mask_sparsity(mask_info: MaskInfo) -> float:
  if mask_info.block_mask is None:
    return 1.0
  return float(np.mean(mask_info.block_mask > 0))


def has_axis(axis_name: str) -> bool:
  try:
    lax.axis_size(axis_name)
    return True
  except (NameError, ValueError):
    return False
