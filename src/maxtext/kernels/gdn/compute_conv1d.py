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

"""In-VMEM causal depthwise Conv1D computation."""

import jax
import jax.numpy as jnp

from . import config


def encode_segment_ids(
    segment_ids: jax.Array,
    init_seg: jax.Array | None = None,
) -> jax.Array:
  """Encodes segment_ids into signed active segments (+d for valid, -d for 0-padding)."""
  seg_i32 = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  if init_seg is not None:
    init_col = jnp.abs(init_seg.astype(jnp.int32))[..., None]
    seg_with_init = jnp.concatenate([init_col, seg_i32], axis=-1)
    active_with_init = jax.lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg_with_init, axis=-1)
    active = active_with_init[..., 1:]
  else:
    active = jax.lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg_i32, axis=-1)
  return jnp.where(seg_i32 > 0, seg_i32, -active).astype(jnp.float32)


def extract_segment_conv_state(
    conv_input: jax.Array,
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> jax.Array:
  """Extracts the trailing (kernel_size - 1) conv state masked to the final segment."""
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return conv_input[:, :0, :]
  tail_qkv = conv_input[:, -halo_len:, :]
  seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  if conv_halo_seg is not None:
    halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32), 0)
    full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
  else:
    full_seg = jnp.pad(seg_pos, ((0, 0), (halo_len, 0)))
  tail_seg = full_seg[:, -halo_len:]
  last_seg = full_seg[:, -1:]
  valid = (last_seg > 0) & (tail_seg == last_seg)
  return jnp.where(valid[..., None], tail_qkv, jnp.zeros_like(tail_qkv))


def causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, chunk, q, dim_size]
    conv_weight: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
    b_vreg: jax.Array | None = None,  # [seq, chunk, 1, aligned_num_v_heads]
) -> tuple[jax.Array, jax.Array]:
  """Perform causal Conv1D. Returns Conv1D output and convolution states."""

  assert lhs.ndim == 4

  out_list = []

  for c_idx in range(cfg.chunk_size):
    out = jnp.zeros((cfg.seq_tile_size, 1, cfg.dim_size), jnp.float32)

    end_idx = c_idx + cfg.prev_kernel_size
    start_idx = 1 + end_idx - cfg.kernel_size
    if cfg.has_seg_ids and b_vreg is not None:
      seg_curr = jnp.maximum(b_vreg[:, c_idx, 0, cfg.num_v_heads], 0.0)
    for k in range(cfg.kernel_size):
      lhs_curr = lhs[:, start_idx + k]
      if cfg.has_seg_ids and b_vreg is not None:
        tap_pos = start_idx + k
        if tap_pos < cfg.prev_kernel_size:
          seg_tap = jnp.maximum(b_vreg[:, tap_pos, 0, cfg.num_v_heads + 1], 0.0)
        else:
          seg_tap = jnp.maximum(b_vreg[:, tap_pos - cfg.prev_kernel_size, 0, cfg.num_v_heads], 0.0)
        same_doc = (seg_curr > 0.5) & (jnp.abs(seg_tap - seg_curr) < 0.5)
        lhs_curr = jnp.where(same_doc.reshape(-1, 1, 1), lhs_curr, 0.0)
      out += lhs_curr * conv_weight[k : k + 1]

    if conv_bias is not None:
      out += conv_bias.reshape(1, 1, -1)
    if cfg.has_seg_ids and b_vreg is not None:
      out = jnp.where((seg_curr > 0.5).reshape(-1, 1, 1), out, 0.0)

    out_list.append(out)

  # Last prev_kernel_size elements needs to be returned as conv_state. However,
  # real_sizes may be smaller than chunk_size. Therefore, slicing last
  # prev_kernel_size elements does not guarantee numeric correctness. Instead,
  # kernel iterate each rows and perform masking to fetch correct values.
  # NOTE: lhs[:, : prev_kernel_size] can be skipped since they were loaded from
  # previous conv states.
  new_conv_state = lhs[:, 1 : cfg.kernel_size]
  real_sizes = real_sizes.reshape(-1, 1, 1, 1)
  # NOTE: Even though for loop is invoked twice, since they are static loops,
  # compiler will perform loop fusion.
  for c_idx in range(2, cfg.chunk_size + 1):
    row_end = c_idx + cfg.prev_kernel_size
    new_conv_state = jnp.where(
        c_idx == real_sizes,
        lhs[:, c_idx:row_end],
        new_conv_state,
    )

  return jnp.stack(out_list, axis=1), new_conv_state
