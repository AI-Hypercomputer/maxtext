# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-JAX GatedDeltaNet sequence-packing, segmented Conv1D, and chunk-inverse helpers."""

from __future__ import annotations

import functools
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import Array, DType, KV_BATCH, KV_HEAD, LENGTH, ShardMode
from maxtext.utils.sharding import (
    get_logical_axis_rules,
    logical_to_mesh_axes,
    remove_incompatible_mesh_axes_from_partition_spec,
)
import numpy as np


def block_doubling_unit_lower_inverse(s_mat: jax.Array, chunk_size: int) -> jax.Array:
  """Computes the exact inverse of (I + s_mat) for strictly lower-triangular s_mat with MXU-friendly matmuls.

  Bottom-up block doubling: starting from 1x1 blocks (inverse = I), each level merges adjacent b x b diagonal
  blocks whose inverses A11^-1 and A22^-1 are already known, using
  [[A11, 0], [A21, A22]]^-1 = [[A11^-1, 0], [-A22^-1 A21 A11^-1, A22^-1]].
  Every intermediate is a block of the true inverse, so unlike the Neumann product (I+N)(I+N^2)(I+N^4)... it
  does not suffer catastrophic cancellation in float32 when keys within a chunk are correlated and decay is weak
  (N^(2^j) can reach 1e7+ while the inverse stays O(1)). Costs 2 * ceil(log2(chunk_size)) matmuls per chunk and
  works for non-power-of-two chunk sizes.
  """
  prec = jax.lax.Precision.HIGHEST
  ar = jnp.arange(chunk_size)
  inv = jnp.broadcast_to(jnp.eye(chunk_size, dtype=jnp.float32), s_mat.shape)
  b = 1
  while b < chunk_size:
    # Lower-left b x b sub-block of each 2b x 2b diagonal block: the only part of s_mat not yet folded into inv.
    cross = ((ar[:, None] // (2 * b)) == (ar[None, :] // (2 * b))) & ((ar[:, None] // b) == (ar[None, :] // b) + 1)
    off = jnp.matmul(
        jnp.matmul(inv, jnp.where(cross, s_mat, 0.0), precision=prec),
        inv,
        precision=prec,
    )
    inv = inv - jnp.where(cross, off, 0.0)
    b *= 2
  return inv


def encode_segment_ids(
    segment_ids: jax.Array,
    init_seg: jax.Array | None = None,
) -> jax.Array:
  """Encodes canonical segment_ids into signed active segments (+d for valid, -d for 0-padding)."""
  seg_i32 = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  if init_seg is not None:
    init_col = jnp.abs(init_seg.astype(jnp.int32))[..., None]
    seg_with_init = jnp.concatenate([init_col, seg_i32], axis=-1)
    active_with_init = lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg_with_init, axis=-1)
    active = active_with_init[..., 1:]
  else:
    active = lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg_i32, axis=-1)
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
  active = lax.associative_scan(lambda a, b: jnp.where(b > 0, b, a), seg, axis=-1)
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


def extract_segment_conv_state(
    conv_input: jax.Array,
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: jax.Array | None = None,
) -> jax.Array:
  """Extracts the (kernel_size - 1) conv state ending at the last valid token of `conv_input`."""
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


def segment_next_conv_state(
    conv_state: Array,
    qkv: Array,
    decoder_segment_ids: Array,
    kernel_size: int,
    sequence_packing: bool = True,
) -> Array:
  """Returns the next Conv1D state of a prefill call: the window ending at the last valid token.

  The caller `conv_state` continues the first document. With
  `sequence_packing`, the window is masked to the document of the last valid
  token. Otherwise every non-padding token is one document. A call without
  valid tokens returns `conv_state` unchanged.
  """
  batch, seq_len, _ = qkv.shape
  seg_2d = jnp.broadcast_to(decoder_segment_ids, (batch, seq_len))
  if sequence_packing:
    seg_2d = canonicalize_segment_ids(seg_2d)
  else:
    seg_2d = (seg_2d != 0).astype(jnp.int32)
  conv_halo_seg, _ = initial_state_segment_metadata(seg_2d, kernel_size)
  out_dtype = jnp.result_type(conv_state.dtype, qkv.dtype)
  return extract_segment_conv_state_split(
      conv_state.astype(out_dtype),
      qkv.astype(out_dtype),
      seg_2d,
      kernel_size,
      conv_halo_seg,
  )


def _compute_segmented_conv1d_core(
    qkv: Array,
    conv_weight_3d: Array,
    conv_bias: None | Array,
    seg_2d: Array,
    kernel_size: int,
    conv_state: None | Array,
    conv_halo_seg: None | Array,
    out_dtype: DType,
    out_sharding: P | None = None,
) -> Array:
  """Core per-tap same-document depthwise Conv1D keeping conv_input in qkv.dtype."""
  batch, seq_len, channels = qkv.shape
  seg_pos = jnp.maximum(seg_2d.astype(jnp.int32), 0)
  valid_mask = (seg_pos > 0)[:, :, None]

  halo_len = kernel_size - 1
  if conv_state is not None:
    conv_input = jnp.concatenate([conv_state.astype(qkv.dtype), qkv], axis=1)
  else:
    conv_input = jnp.pad(qkv, ((0, 0), (halo_len, 0), (0, 0)))

  if conv_halo_seg is not None:
    halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32).reshape(batch, halo_len), 0)
    full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
  else:
    full_seg = jnp.pad(seg_pos, ((0, 0), (halo_len, 0)))

  zero_in = jnp.zeros((), dtype=conv_input.dtype)
  conv_input_masked = jnp.where((full_seg > 0)[:, :, None], conv_input, zero_in)

  conv_kwargs = {"out_sharding": out_sharding} if out_sharding is not None else {}
  base_out = lax.conv_general_dilated(
      lhs=conv_input_masked.astype(jnp.float32),
      rhs=conv_weight_3d.astype(jnp.float32),
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NHC", "HIO", "NHC"),
      feature_group_count=channels,
      precision=lax.Precision.HIGHEST,
      **conv_kwargs,
  )

  if halo_len > 0:
    has_cross_doc = jnp.zeros((batch, seq_len), dtype=jnp.bool_)
    for k in range(halo_len):
      has_cross_doc = has_cross_doc | (
          (seg_pos > 0) & (full_seg[:, k : k + seq_len] > 0) & (full_seg[:, k : k + seq_len] != seg_pos)
      )
    taps = jnp.stack(
        [
            jnp.where(
                (has_cross_doc & (full_seg[:, k : k + seq_len] == seg_pos))[:, :, None],
                conv_input_masked[:, k : k + seq_len, :],
                zero_in,
            )
            for k in range(kernel_size)
        ],
        axis=2,
    )
    boundary_out = jnp.einsum(
        "bskc,kc->bsc",
        taps,
        conv_weight_3d[:, 0, :],
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    conv_out = jnp.where(has_cross_doc[:, :, None], boundary_out, base_out)
  else:
    conv_out = base_out

  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  conv_out = jnp.where(valid_mask, conv_out, 0.0)
  return conv_out.astype(out_dtype)


def segmented_causal_depthwise_conv1d(
    qkv: Array,
    conv_weight: Array,
    conv_bias: None | Array = None,
    segment_ids: None | Array | DType = None,
    kernel_size: int | None = None,
    conv_state: None | Array = None,
    conv_halo_seg: None | Array = None,
    dtype: DType | None = None,
    *,
    mesh: jax.sharding.Mesh | None = None,
    cp_axes: tuple[str, ...] = (),
    shard_mode: ShardMode = ShardMode.AUTO,
    gdn_head_axis: str = KV_HEAD,
) -> Array:
  """Causal depthwise Conv1D with per-tap same-document masking and zero output on padding.

  Supports both the full keyword signature:
    `segmented_causal_depthwise_conv1d(qkv, conv_weight, conv_bias, segment_ids, kernel_size, ...)`
  and the 4-arg positional signature:
    `segmented_causal_depthwise_conv1d(x, kernel, segment_ids, dtype=...)`.
  """
  if (
      conv_bias is not None
      and conv_bias.ndim >= 2
      and (segment_ids is None or not isinstance(segment_ids, (jax.Array, np.ndarray)))
  ):
    if segment_ids is not None:
      dtype = segment_ids
    segment_ids = conv_bias
    conv_bias = None
  if segment_ids is None:
    raise ValueError("segment_ids must be provided to segmented_causal_depthwise_conv1d")
  if kernel_size is None:
    kernel_size = int(conv_weight.shape[0])
  out_dtype = dtype if dtype is not None else qkv.dtype

  batch, seq_len, _ = qkv.shape
  seg_2d = jnp.broadcast_to(segment_ids, (batch, seq_len))
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  active_cp_axes = tuple(ax for ax in cp_axes if mesh is not None and ax in mesh.axis_names and mesh.shape[ax] > 1)
  if active_cp_axes and mesh is not None and conv_state is None and conv_halo_seg is None:
    from maxtext.kernels.attention import gdn_cp  # pylint: disable=import-outside-toplevel

    logical_rules = get_logical_axis_rules()
    qkv_pspec = logical_to_mesh_axes((KV_BATCH, LENGTH, gdn_head_axis), mesh=mesh, rules=logical_rules)
    qkv_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        qkv_pspec, qkv.shape, mesh, dims=(0,), allow_remove_axes=True
    )
    seg_pspec = P(qkv_pspec[0], qkv_pspec[1])
    cw_pspec = P(None, None, qkv_pspec[2])
    cb_pspec = P(qkv_pspec[2]) if conv_bias is not None else P()
    cp_ax = active_cp_axes[0] if len(active_cp_axes) == 1 else active_cp_axes
    halo_len = kernel_size - 1

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(qkv_pspec, cw_pspec, cb_pspec, seg_pspec),
        out_specs=qkv_pspec,
        check_vma=False,
    )
    def _shard_mapped_conv(qkv_loc, cw_loc, cb_loc, seg_loc):
      s_enc_loc, halo_seg, _ = gdn_cp.gather_cp_segment_metadata(seg_loc, cp_ax, kernel_size, has_initial_state=False)
      seg_canon_loc = jnp.maximum(s_enc_loc, 0.0).astype(jnp.int32)
      n_cp = lax.axis_size(cp_ax)
      cp_idx = lax.axis_index(cp_ax)
      perm = [(i, i + 1) for i in range(n_cp - 1)]
      halo_qkv = (
          lax.ppermute(qkv_loc[:, -halo_len:, :], axis_name=cp_ax, perm=perm) if halo_len > 0 else qkv_loc[:, :0, :]
      )
      if halo_len > 0:
        halo_qkv = jnp.where(cp_idx == 0, jnp.zeros_like(halo_qkv), halo_qkv)
      return _compute_segmented_conv1d_core(
          qkv_loc,
          cw_loc,
          cb_loc,
          seg_canon_loc,
          kernel_size,
          halo_qkv if halo_len > 0 else None,
          halo_seg if halo_len > 0 else None,
          out_dtype,
      )

    return _shard_mapped_conv(qkv, conv_weight_3d, conv_bias, seg_2d)

  if shard_mode == ShardMode.EXPLICIT:
    qkv_sharding = getattr(jax.typeof(qkv), "sharding", None)
    qkv_spec = getattr(qkv_sharding, "spec", None)
    if qkv_spec is not None and len(qkv_spec) >= 3:
      # Canonicalize before resharding if unsharded along sequence, or reshard first so explicit sharding agrees.
      seg_2d = jax.sharding.reshard(seg_2d, P(qkv_spec[0], qkv_spec[1]))
      if conv_halo_seg is None:
        seg_2d = canonicalize_segment_ids(seg_2d)
        if conv_state is not None:
          conv_halo_seg, _ = initial_state_segment_metadata(seg_2d, kernel_size)
      conv_weight_3d = jax.sharding.reshard(conv_weight_3d, P(None, None, qkv_spec[2]))
      if conv_bias is not None:
        conv_bias = jax.sharding.reshard(conv_bias, P(qkv_spec[2]))
      if conv_halo_seg is not None:
        conv_halo_seg = jax.sharding.reshard(conv_halo_seg, P(qkv_spec[0], None))
      return _compute_segmented_conv1d_core(
          qkv,
          conv_weight_3d,
          conv_bias,
          seg_2d,
          kernel_size,
          conv_state,
          conv_halo_seg,
          out_dtype,
          out_sharding=P(qkv_spec[0], qkv_spec[1], qkv_spec[2]),
      )

  if conv_halo_seg is None:
    seg_2d = canonicalize_segment_ids(seg_2d)
    if conv_state is not None:
      conv_halo_seg, _ = initial_state_segment_metadata(seg_2d, kernel_size)

  return _compute_segmented_conv1d_core(
      qkv,
      conv_weight_3d,
      conv_bias,
      seg_2d,
      kernel_size,
      conv_state,
      conv_halo_seg,
      out_dtype,
  )


def prepare_jax_gdn_segment_masks(
    segment_ids: Array,
    q_c: Array,
    k_c: Array,
    v_c: Array,
    g_c: Array,
    beta_c: Array,
    *,
    batch_size: int,
    seq_len: int,
    num_chunks: int,
    chunk_size: int,
    cp_axis: None | str | tuple[str, ...] = None,
    init_seg: None | Array = None,
    has_initial_state: bool = False,
):
  """Prepares segment-aware masks and segment-local cumulative decay for Pure-JAX GDN.

  Without `init_seg`, raw `segment_ids` are canonicalized (globally under
  `cp_axis`), and `has_initial_state` makes the initial recurrent state continue
  the first document. With `init_seg`, `segment_ids` must already be canonical.
  """
  from maxtext.kernels.attention import gdn_cp  # pylint: disable=import-outside-toplevel

  seg_2d = jnp.broadcast_to(segment_ids, (batch_size, seq_len))
  if cp_axis is not None and init_seg is None:
    s_enc, _, init_seg = gdn_cp.gather_cp_segment_metadata(seg_2d, cp_axis, 4, has_initial_state=has_initial_state)
  else:
    if init_seg is None:
      seg_2d = canonicalize_segment_ids(seg_2d)
      if has_initial_state:
        init_seg = jnp.ones((batch_size,), dtype=jnp.float32)
    s_enc = encode_segment_ids(seg_2d, init_seg=init_seg)

  pad_len = num_chunks * chunk_size - seq_len
  if pad_len > 0:
    tail_active = -jnp.abs(s_enc[:, -1:])
    s_enc = jnp.concatenate([s_enc, jnp.broadcast_to(tail_active, (batch_size, pad_len))], axis=1)

  seg_c = s_enc.reshape(batch_size, num_chunks, 1, chunk_size)
  valid_c = seg_c > 0.5
  active_c = jnp.abs(seg_c)
  active_end = active_c[..., -1]
  if init_seg is not None:
    init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(batch_size, 1, 1))
  else:
    init_active = jnp.zeros((batch_size, 1, 1), dtype=jnp.float32)
  seg_prev = jnp.concatenate([init_active, active_end[:, :-1, :]], axis=1)[..., None]

  q_c = jnp.where(valid_c[..., None], q_c, 0.0)
  k_c = jnp.where(valid_c[..., None], k_c, 0.0)
  v_c = jnp.where(valid_c[..., None], v_c, 0.0)
  g_c = jnp.where(valid_c, g_c, 0.0)
  beta_c = jnp.where(valid_c, beta_c, 0.0)

  mask_tril_f32 = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  same_active = (jnp.abs(active_c[..., :, None] - active_c[..., None, :]) < 0.5) & (active_c[..., :, None] > 0.5)
  mask_cumsum = mask_tril_f32 * same_active.astype(jnp.float32)
  g_cumsum = jnp.einsum(
      "bnhij,bnhj->bnhi",
      mask_cumsum,
      g_c,
      precision=jax.lax.Precision.HIGHEST,
  )

  same_valid = (jnp.abs(seg_c[..., :, None] - seg_c[..., None, :]) < 0.5) & valid_c[..., :, None]
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1) & same_valid
  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0) & same_valid
  active_last = active_c[..., -1:]
  m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
  m_out = ((jnp.abs(seg_c - active_last) < 0.5) & (active_last > 0.5)).astype(jnp.float32)
  m_keep = ((jnp.abs(active_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)

  return (
      q_c,
      k_c,
      v_c,
      g_c,
      beta_c,
      g_cumsum,
      mask_strict,
      mask_causal,
      m_in,
      m_out,
      m_keep,
      valid_c,
  )
