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


def canonicalize_segment_ids(segment_ids: jax.Array) -> jax.Array:
  """Relabels segment IDs so every document gets a unique, increasing int32 ID starting at 1.

  A new document starts wherever a valid token's raw ID differs from the last
  valid token before it, so non-adjacent repeats of one raw ID ([1, 2, 1]) become
  distinct documents ([1, 2, 3]). 0-padding stays 0 and does not split a
  document: [1, 0, 1] keeps one document. The resulting IDs are bounded by the
  number of documents in the row, independent of the raw ID values.

  Args:
    segment_ids: [..., seq] raw segment IDs; values <= 0 are padding.

  Returns:
    [..., seq] int32 canonical segment IDs.
  """
  seg = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  active = jax.lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg, axis=-1)
  prev_active = jnp.concatenate([jnp.zeros_like(active[..., :1]), active[..., :-1]], axis=-1)
  valid = seg > 0
  new_doc = valid & (seg != prev_active)
  doc_id = jnp.cumsum(new_doc.astype(jnp.int32), axis=-1)
  return jnp.where(valid, doc_id, 0).astype(jnp.int32)


def initial_state_segment_metadata(
    segment_ids: jax.Array,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Returns (conv_halo_seg, init_seg) that attach caller conv/recurrent states to the first document.

  `segment_ids` must already be canonical (see `canonicalize_segment_ids`), so
  the first document is always 1. A row without valid tokens also uses 1, so an
  all-padding call passes the caller states through unchanged.

  Args:
    segment_ids: [batch, seq] canonical segment IDs.
    kernel_size: Conv1D kernel size.

  Returns:
    conv_halo_seg: [batch, kernel_size - 1] float32 segment IDs of the conv halo.
    init_seg: [batch] float32 segment ID of the initial recurrent state.
  """
  batch = segment_ids.shape[0]
  halo_len = max(kernel_size - 1, 0)
  return (
      jnp.ones((batch, halo_len), dtype=jnp.float32),
      jnp.ones((batch,), dtype=jnp.float32),
  )


def segment_conv_state_window(
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Locates the (kernel_size - 1) conv-state window ending at the last valid token.

  Positions index the virtual sequence [halo (kernel_size - 1), tokens (seq)].

  Args:
    segment_ids: [batch, seq] segment IDs (> 0 valid). Signed encodings are accepted.
    kernel_size: Conv1D kernel size.
    conv_halo_seg: Optional [batch, kernel_size - 1] segment IDs of the halo. None
      means the halo is not part of any document.

  Returns:
    window_pos: [batch, kernel_size - 1] int32 positions, clipped into range.
    keep: [batch, kernel_size - 1] bool, True where the window token belongs to
      the document of the last valid token.
    has_valid: [batch] bool, True if the halo or the tokens hold a valid token.
  """
  batch, seq_len = segment_ids.shape
  halo_len = kernel_size - 1
  seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  if conv_halo_seg is not None:
    halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32).reshape(batch, halo_len), 0)
  else:
    halo_pos = jnp.zeros((batch, halo_len), dtype=jnp.int32)
  full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
  total_len = halo_len + seq_len
  pos = jnp.arange(total_len, dtype=jnp.int32)
  end_idx = jnp.max(jnp.where(full_seg > 0, pos[None, :], -1), axis=1)
  has_valid = end_idx >= 0
  window = end_idx[:, None] - (halo_len - 1) + jnp.arange(halo_len, dtype=jnp.int32)[None, :]
  window_pos = jnp.clip(window, 0, total_len - 1)
  end_seg = jnp.take_along_axis(full_seg, jnp.maximum(end_idx, 0)[:, None], axis=1)
  window_seg = jnp.take_along_axis(full_seg, window_pos, axis=1)
  keep = has_valid[:, None] & (window >= 0) & (window_seg == end_seg)
  return window_pos, keep, has_valid


def extract_segment_conv_state_split(
    conv_prefix: jax.Array | None,
    x: jax.Array,
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> jax.Array:
  """Returns the (kernel_size - 1) conv state ending at the last valid token of [conv_prefix, x].

  Window tokens outside the document of the last valid token are zeroed.
  Trailing padding is treated as bucket padding and skipped. If neither the
  halo nor `x` holds a valid token, the result is zero.

  Args:
    conv_prefix: Optional [batch, kernel_size - 1, dim] halo values (None = zeros).
    x: [batch, seq, dim] tokens.
    segment_ids: [batch, seq] segment IDs of `x`.
    kernel_size: Conv1D kernel size.
    conv_halo_seg: Optional [batch, kernel_size - 1] segment IDs of the halo.

  Returns:
    [batch, kernel_size - 1, dim] conv state.
  """
  batch, seq_len, _ = x.shape
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return x[:, :0, :]
  window_pos, keep, _ = segment_conv_state_window(segment_ids, kernel_size, conv_halo_seg)
  from_prefix = window_pos < halo_len
  x_idx = jnp.clip(window_pos - halo_len, 0, seq_len - 1)
  x_vals = jnp.take_along_axis(x, x_idx[..., None], axis=1)
  if conv_prefix is not None:
    out_dtype = jnp.result_type(conv_prefix.dtype, x.dtype)
    p_idx = jnp.clip(window_pos, 0, halo_len - 1)
    p_vals = jnp.take_along_axis(conv_prefix.reshape(batch, halo_len, -1), p_idx[..., None], axis=1)
    vals = jnp.where(from_prefix[..., None], p_vals.astype(out_dtype), x_vals.astype(out_dtype))
  else:
    vals = jnp.where(from_prefix[..., None], jnp.zeros_like(x_vals), x_vals)
  return jnp.where(keep[..., None], vals, jnp.zeros_like(vals))


def extract_segment_conv_state_split_adjoint(
    d_state: jax.Array,
    dx: jax.Array,
    d_prefix: jax.Array | None,
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Adds the cotangent of `extract_segment_conv_state_split` into (dx, d_prefix).

  Uses a (kernel_size - 1)-row scatter-add, so no [batch, seq, dim] temporary is
  materialized.

  Args:
    d_state: [batch, kernel_size - 1, dim] cotangent of the extracted conv state.
    dx: [batch, seq, dim] accumulator for the cotangent of `x`.
    d_prefix: Optional [batch, kernel_size - 1, dim] accumulator for the halo.
    segment_ids: [batch, seq] segment IDs of `x`.
    kernel_size: Conv1D kernel size.
    conv_halo_seg: Optional [batch, kernel_size - 1] segment IDs of the halo.

  Returns:
    (dx, d_prefix) with the extraction cotangent added.
  """
  batch, seq_len, _ = dx.shape
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return dx, d_prefix
  window_pos, keep, _ = segment_conv_state_window(segment_ids, kernel_size, conv_halo_seg)
  upd = jnp.where(keep[..., None], d_state, jnp.zeros_like(d_state))
  from_prefix = window_pos < halo_len
  batch_idx = jnp.arange(batch, dtype=jnp.int32)[:, None]
  x_idx = jnp.where(from_prefix, seq_len, window_pos - halo_len)
  dx = dx.at[batch_idx, x_idx].add(upd.astype(dx.dtype), mode="drop")
  if d_prefix is not None:
    p_idx = jnp.where(from_prefix, window_pos, halo_len)
    d_prefix = d_prefix.at[batch_idx, p_idx].add(upd.astype(d_prefix.dtype), mode="drop")
  return dx, d_prefix


def extract_segment_conv_state(
    conv_input: jax.Array,
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> jax.Array:
  """Extracts the (kernel_size - 1) conv state ending at the last valid token of `conv_input`.

  Args:
    conv_input: [batch, (kernel_size - 1) + seq, dim] = [halo, tokens].
    segment_ids: [batch, seq] segment IDs of the tokens.
    kernel_size: Conv1D kernel size.
    conv_halo_seg: Optional [batch, kernel_size - 1] segment IDs of the halo.

  Returns:
    [batch, kernel_size - 1, dim] conv state.
  """
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return conv_input[:, :0, :]
  return extract_segment_conv_state_split(
      conv_input[:, :halo_len, :],
      conv_input[:, halo_len:, :],
      segment_ids,
      kernel_size,
      conv_halo_seg,
  )


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
