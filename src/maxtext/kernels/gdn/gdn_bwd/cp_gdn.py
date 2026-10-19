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

r"""Sequence-sharded Context Parallelism (CP) primitives for GDN Pallas kernels.

Implements:
1. Causal Conv1D boundary halo exchange (`halo_exchange_for_conv`) and its
   reverse-mode adjoint (`halo_exchange_for_conv_bwd`).
2. Pre-batched local affine transition summary (`compose_local_from_t_inv`) from
   the forward kernel's cached `t_inv` ($T_c^{-1}$).
3. Zero-GEMM backward transition transpose ($\mathrm{d}M_{\mathrm{loc}} =
   M_{\mathrm{loc}}^\top$) and pre-batched local state gradient summary
   (`compose_bwd_local_from_t_inv`).
4. Reverse Hillis-Steele prefix scan (`incoming_grad_state`) over `cp_axis` in
   $\lceil \log_2(D) \rceil$ `lax.ppermute` steps with $\mathcal{O}(1)$
   memory per device.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from .. import compute_conv1d as local_compute_conv1d
from .. import compute_gdn as local_compute_gdn
from .runtime_utils import target_platform

# Precision of the f32 composition matmuls below: 6 bf16 passes on TPU. `gdn_cp_matmul_precision=high`
# passes Precision.HIGH (3 passes) to them through their `precision` argument instead.
DEFAULT_PRECISION = jax.lax.Precision.HIGHEST


def _compute_t_inv_batched(
    k_orig: jax.Array,  # [B, N_c, C, H_k, d_k] f32, already qk-normed if enabled
    beta_h: jax.Array,  # [N_c, B, H_v, C] f32, already masked if segment_ids
    cumsum_h: jax.Array,  # [N_c, B, H_v, C] f32
    seg_c: Optional[jax.Array],  # [N_c, B, 1, C] f32 or None
    valid_c: Optional[jax.Array],  # [N_c, B, 1, C] bool or None
    *,
    repeats: int,
    precision: jax.lax.Precision = DEFAULT_PRECISION,
) -> jax.Array:
  """Computes `t_inv_c: [N_c, B, H_v, C, C]` across all local chunks in one batched call."""
  num_chunks, batch, num_v_heads, chunk_size = beta_h.shape
  k_kq = jnp.transpose(k_orig, (1, 0, 3, 2, 4))  # [N_c, B, H_k, C, d_k]
  if valid_c is not None:
    k_kq = jnp.where(valid_c[..., None], k_kq, 0.0)
  kk_kq = jnp.matmul(k_kq, jnp.swapaxes(k_kq, -1, -2), precision=precision)  # [N_c, B, H_k, C, C]
  kk_h = jnp.repeat(kk_kq, repeats, axis=2)  # [N_c, B, H_v, C, C]

  mask_strict_base = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  if seg_c is not None and valid_c is not None:
    same_valid = (jnp.abs(seg_c[..., :, None] - seg_c[..., None, :]) < 0.5) & valid_c[..., :, None]
    mask_strict = mask_strict_base[None, None, None, :, :] * same_valid.astype(jnp.float32)
  else:
    mask_strict = mask_strict_base
  diff = cumsum_h[..., :, None] - cumsum_h[..., None, :]
  safe_diff = jnp.where(mask_strict > 0.5, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff) * mask_strict

  t_lower = kk_h * (beta_h[..., :, None] * g_mat_strict)
  eye = jnp.eye(chunk_size, dtype=jnp.float32)
  t_full = eye + t_lower
  t_flat = t_full.reshape(num_chunks * batch * num_v_heads, chunk_size, chunk_size)
  t_inv_flat = local_compute_gdn.invert_triangular_matrix(t_flat, precision=precision)
  return t_inv_flat.reshape(num_chunks, batch, num_v_heads, chunk_size, chunk_size)


def gather_cp_segment_metadata(
    segment_ids: Optional[jax.Array],
    cp_axis: str | tuple[str, ...],
    kernel_size: int,
    has_initial_state: bool = False,
) -> Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]:
  """Gathers global segment_ids across cp_axis and slices local (s_enc, conv_halo_seg, init_seg).

  Segment IDs are canonicalized on the gathered global sequence, so every rank
  sees the same document numbering. With `has_initial_state`, the caller
  conv/recurrent states continue the first document (canonical ID 1).
  """
  if segment_ids is None:
    return None, None, None
  batch_size, seq_len = segment_ids.shape
  halo_len = max(kernel_size - 1, 1)
  global_seg = lax.all_gather(segment_ids, axis_name=cp_axis, axis=1, tiled=True)
  global_seg = local_compute_conv1d.canonicalize_segment_ids(global_seg)
  if has_initial_state:
    init_global = jnp.ones((batch_size,), dtype=jnp.float32)
    pad_value = 1.0
  else:
    init_global = None
    pad_value = 0.0
  global_s_enc = local_compute_conv1d.encode_segment_ids(global_seg, init_seg=init_global)
  idx = lax.axis_index(cp_axis)
  start = idx * seq_len
  s_enc_local = lax.dynamic_slice_in_dim(global_s_enc, start, seq_len, axis=1)

  global_s_enc_pad = jnp.pad(global_s_enc, ((0, 0), (halo_len, 0)), constant_values=pad_value)
  if kernel_size - 1 > 0:
    conv_halo_seg = lax.dynamic_slice_in_dim(jnp.maximum(global_s_enc_pad, 0.0), start, kernel_size - 1, axis=1)
  else:
    conv_halo_seg = jnp.zeros((batch_size, 0), dtype=jnp.float32)
  init_seg = jnp.abs(lax.dynamic_slice_in_dim(global_s_enc_pad, start + halo_len - 1, 1, axis=1)[:, 0])
  return s_enc_local, conv_halo_seg, init_seg


def compose(
    left: Tuple[jax.Array, jax.Array],
    right: Tuple[jax.Array, jax.Array],
    precision: jax.lax.Precision = DEFAULT_PRECISION,
) -> Tuple[jax.Array, jax.Array]:
  """Composes two affine maps: (A_r, B_r) o (A_l, B_l) = (A_r @ A_l, A_r @ B_l + B_r)."""
  a_l, b_l = left
  a_r, b_r = right
  return (
      jnp.matmul(a_r, a_l, precision=precision),
      jnp.matmul(a_r, b_l, precision=precision) + b_r,
  )


def incoming_state(
    a_loc: jax.Array,
    b_loc: jax.Array,
    h_init: jax.Array,
    cp_axis: str | tuple[str, ...],
    precision: jax.lax.Precision = DEFAULT_PRECISION,
    *,
    fast_state_passing: bool = False,
    has_initial_state: bool = True,
    need_final_state: bool = True,
) -> Tuple[jax.Array, jax.Array]:
  """Hillis-Steele prefix scan across `cp_axis` returning (h_in, final_h) in O(log2 D) steps."""
  d_size = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)

  if fast_state_passing and d_size == 4 and not has_initial_state and a_loc.shape == b_loc.shape:
    p1 = lax.ppermute(jnp.where(idx == 2, a_loc, b_loc), cp_axis, [(0, 1), (1, 2), (2, 0)])
    p2 = lax.ppermute(jnp.where(idx == 0, b_loc, a_loc), cp_axis, [(0, 3), (1, 0)])
    lhs1 = jnp.where(idx == 0, p1, a_loc)
    rhs1 = jnp.where(idx == 0, p2, p1)
    m1_prod = jnp.matmul(lhs1, rhs1, precision=precision)
    m1_b = m1_prod + b_loc
    b_r2 = lax.ppermute(m1_b, cp_axis, [(1, 2), (2, 3)])
    a_r2 = lax.ppermute(m1_prod, cp_axis, [(0, 3)])
    h3_on_3 = jnp.matmul(a_r2, p2, precision=precision) + b_r2
    h_in = jnp.where(
        idx == 0,
        h_init,
        jnp.where(idx == 1, p1, jnp.where(idx == 2, b_r2, h3_on_3)),
    )
    if target_platform(a_loc) == "cpu":
      h_in = lax.optimization_barrier(h_in)
    if not need_final_state:
      return h_in, jnp.zeros_like(b_loc)
    last_b = jnp.matmul(a_loc, h3_on_3, precision=precision) + b_loc
    final_h = lax.psum(jnp.where(idx == 3, last_b, jnp.zeros_like(b_loc)), cp_axis)
    return h_in, final_h

  a_run, b_run = a_loc, b_loc
  step = 1
  while step < d_size:
    fwd = [(i, i + step) for i in range(d_size - step)]
    b_recv = lax.ppermute(b_run, cp_axis, fwd)
    live = idx >= step
    if fast_state_passing and (step * 2 >= d_size) and not has_initial_state and not need_final_state:
      b_cmp = jnp.matmul(a_run, b_recv, precision=precision) + b_run
      b_run = jnp.where(live, b_cmp, b_run)
    else:
      a_recv = lax.ppermute(a_run, cp_axis, fwd)
      a_cmp, b_cmp = compose((a_recv, b_recv), (a_run, b_run), precision)
      a_run = jnp.where(live, a_cmp, a_run)
      b_run = jnp.where(live, b_cmp, b_run)
    step *= 2

  shift1 = [(i, i + 1) for i in range(d_size - 1)]
  b_ex = lax.ppermute(b_run, cp_axis, shift1)
  if fast_state_passing and not has_initial_state:
    h_in = jnp.where(idx == 0, h_init, b_ex)
  else:
    a_ex = lax.ppermute(a_run, cp_axis, shift1)
    carried = jnp.matmul(a_ex, h_init, precision=precision) + b_ex
    h_in = jnp.where(idx == 0, h_init, carried)

  if fast_state_passing and not need_final_state:
    return h_in, jnp.zeros_like(b_loc)
  last = idx == (d_size - 1)
  b_tot = lax.psum(jnp.where(last, b_run, jnp.zeros_like(b_run)), cp_axis)
  if fast_state_passing and not has_initial_state:
    return h_in, b_tot
  a_tot = lax.psum(jnp.where(last, a_run, jnp.zeros_like(a_run)), cp_axis)
  final_h = jnp.matmul(a_tot, h_init, precision=precision) + b_tot
  return h_in, final_h


def halo_exchange_for_conv(
    qkv: jax.Array,
    init_conv_state: Optional[jax.Array],
    kernel_size: int,
    cp_axis: str | tuple[str, ...],
    segment_ids: Optional[jax.Array] = None,
) -> jax.Array:
  """Exchanges the trailing (kernel_size - 1) QKV tokens from rank r - 1 to rank r."""
  batch, seq_len, dim_size = qkv.shape
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return jnp.zeros((batch, 0, dim_size), dtype=qkv.dtype)

  d_size = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)

  if seq_len >= halo_len:
    tail_qkv = qkv[:, -halo_len:, :]
  else:
    tail_qkv = jnp.pad(qkv, ((0, 0), (halo_len - seq_len, 0), (0, 0)))

  shift1 = [(i, i + 1) for i in range(d_size - 1)]
  recv_halo = lax.ppermute(tail_qkv, cp_axis, shift1)

  if segment_ids is not None:
    seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
    if seq_len >= halo_len:
      tail_seg = seg_pos[:, -halo_len:]
    else:
      tail_seg = jnp.pad(seg_pos, ((0, 0), (halo_len - seq_len, 0)), constant_values=-1)
    recv_seg = lax.ppermute(tail_seg, cp_axis, shift1)
    head_seg = seg_pos[:, : min(seq_len, halo_len)]
    same_doc = jnp.any(
        (recv_seg[:, :, None] == head_seg[:, None, :]) & (recv_seg[:, :, None] > 0),
        axis=-1,
    )[..., None]
    recv_halo = jnp.where(same_doc, recv_halo, jnp.zeros_like(recv_halo))

  if init_conv_state is not None:
    rank0_halo = init_conv_state.astype(qkv.dtype)
  else:
    rank0_halo = jnp.zeros((batch, halo_len, dim_size), dtype=qkv.dtype)

  return jnp.where(idx == 0, rank0_halo, recv_halo)


def select_end_conv_state_rank(
    has_valid: Optional[jax.Array],
    cp_axis: str | tuple[str, ...],
) -> jax.Array:
  """Returns a per-row bool that is True on the rank owning the final Conv1D state.

  That is the last rank whose local tokens hold a valid token, or rank 0 if none
  does (rank 0's window then covers the caller conv state). A rank whose tokens
  are all padding never owns it: its halo is zeroed by `halo_exchange_for_conv`.
  Without segment metadata (`has_valid` None), it is the last rank.
  """
  d_size = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)
  if has_valid is None:
    return jnp.asarray(idx == (d_size - 1))
  owner = lax.pmax(jnp.where(has_valid, idx, 0).astype(jnp.int32), cp_axis)
  return owner == idx


def broadcast_end_conv_state(
    next_cs: jax.Array,
    cp_axis: str | tuple[str, ...],
    has_valid: Optional[jax.Array] = None,
) -> jax.Array:
  """Broadcasts the final Conv1D state from its owning CP rank to all ranks.

  Args:
    next_cs: [batch, kernel_size - 1, dim] local conv state of each rank.
    cp_axis: CP mesh axis name(s).
    has_valid: Optional [batch] bool, True where the rank's local tokens hold a
      valid token. None selects the last rank (D - 1).
  """
  is_owner = select_end_conv_state_rank(has_valid, cp_axis)
  if is_owner.ndim == 1:
    is_owner = is_owner[:, None, None]
  return lax.psum(jnp.where(is_owner, next_cs, jnp.zeros_like(next_cs)), cp_axis)


def halo_exchange_for_conv_bwd(
    dx: jax.Array,
    d_conv_state: jax.Array,
    d_conv_state_ext: Optional[jax.Array],
    kernel_size: int,
    cp_axis: str | tuple[str, ...],
    segment_ids: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
  """Reverse-mode adjoint of `halo_exchange_for_conv` using reverse `lax.ppermute`."""
  halo_len = kernel_size - 1
  if halo_len <= 0:
    return dx, jnp.zeros_like(d_conv_state)

  d_size = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)
  seq_len = dx.shape[1]

  d_cs_masked = d_conv_state
  if segment_ids is not None:
    seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
    if seq_len >= halo_len:
      tail_seg = seg_pos[:, -halo_len:]
    else:
      tail_seg = jnp.pad(seg_pos, ((0, 0), (halo_len - seq_len, 0)), constant_values=-1)
    shift1_fwd = [(i, i + 1) for i in range(d_size - 1)]
    recv_seg = lax.ppermute(tail_seg, cp_axis, shift1_fwd)
    head_seg = seg_pos[:, : min(seq_len, halo_len)]
    same_doc = jnp.any(
        (recv_seg[:, :, None] == head_seg[:, None, :]) & (recv_seg[:, :, None] > 0),
        axis=-1,
    )[..., None]
    d_cs_masked = jnp.where(
        idx == 0,
        d_conv_state,
        jnp.where(same_doc, d_conv_state, jnp.zeros_like(d_conv_state)),
    )

  shift1_bwd = [(i + 1, i) for i in range(d_size - 1)]
  recv_d_tail = lax.ppermute(d_cs_masked, cp_axis, shift1_bwd)

  if d_conv_state_ext is not None:
    ext_d_tail = d_conv_state_ext.astype(dx.dtype)
  else:
    ext_d_tail = jnp.zeros_like(d_cs_masked, dtype=dx.dtype)

  add_tail = jnp.where(idx == (d_size - 1), ext_d_tail, recv_d_tail.astype(dx.dtype))
  dx_updated = dx.at[:, -halo_len:, :].add(add_tail)

  d_init_cs = lax.psum(
      jnp.where(
          idx == 0,
          d_conv_state.astype(jnp.float32),
          jnp.zeros_like(d_conv_state, dtype=jnp.float32),
      ),
      cp_axis,
  ).astype(dx.dtype)
  return dx_updated, d_init_cs


def compose_local_from_t_inv(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    t_inv: Optional[jax.Array] = None,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    s_ext_pass1: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    precision: jax.lax.Precision = DEFAULT_PRECISION,
) -> Tuple[jax.Array, jax.Array]:
  """Folds local chunks into (M_local, S_ext_local) using pre-batched GEMMs outside lax.scan."""
  batch, seq_len, _ = qkv_conv.shape
  num_chunks = seq_len // chunk_size
  repeats = num_v_heads // num_k_heads
  q_size = num_k_heads * head_k_dim
  k_size = num_k_heads * head_k_dim

  if qkv_conv.shape[-1] == k_size and s_ext_pass1 is not None:
    k_slice = qkv_conv
  else:
    k_slice = qkv_conv[:, :, q_size : q_size + k_size]

  k_orig = k_slice.reshape(batch, num_chunks, chunk_size, num_k_heads, head_k_dim).astype(jnp.float32)
  if use_qk_norm_in_gdn:
    k_orig = k_orig * jax.lax.rsqrt(jnp.sum(k_orig**2, axis=-1, keepdims=True) + 1e-6)
  k_rep = jnp.repeat(k_orig, repeats, axis=3)
  k_h = jnp.transpose(k_rep, (1, 0, 3, 2, 4))  # [N_c, B, H_v, C, d_k]

  b_h = jnp.transpose(
      b.astype(jnp.float32).reshape(batch, num_chunks, chunk_size, num_v_heads),
      (1, 0, 3, 2),
  )
  a_h = jnp.transpose(
      a.astype(jnp.float32).reshape(batch, num_chunks, chunk_size, num_v_heads),
      (1, 0, 3, 2),
  )
  beta_h = jax.nn.sigmoid(b_h)

  a_log_f32 = a_log.astype(jnp.float32)[None, None, :, None]
  dt_bias_f32 = dt_bias.astype(jnp.float32)[None, None, :, None]
  log_g = -jnp.exp(a_log_f32) * jax.nn.softplus(a_h + dt_bias_f32)

  seg_c = None
  valid_c = None
  if segment_ids is not None:
    s_enc = local_compute_conv1d.encode_segment_ids(segment_ids.reshape(batch, seq_len), init_seg=init_seg)
    seg_c = jnp.transpose(s_enc.reshape(batch, num_chunks, chunk_size), (1, 0, 2))[:, :, None, :]
    valid_c = seg_c > 0.5
    active_c = jnp.abs(seg_c)
    active_end = active_c[..., -1]
    if init_seg is not None:
      init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(1, batch, 1))
    else:
      init_active = jnp.zeros((1, batch, 1), dtype=jnp.float32)
    seg_prev = jnp.concatenate([init_active, active_end[:-1]], axis=0)[..., None]

    k_h = jnp.where(valid_c[..., None], k_h, 0.0)
    beta_h = jnp.where(valid_c, beta_h, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)

    mask_tril = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
    same_active = (jnp.abs(active_c[..., :, None] - active_c[..., None, :]) < 0.5) & (active_c[..., :, None] > 0.5)
    mask_cumsum = mask_tril[None, None, None, :, :] * same_active.astype(jnp.float32)
    # The segment-masked cumsum feeds exp(); it stays at HIGHEST whatever `precision` is.
    cumsum_h = jnp.einsum("nbhij,nbhj->nbhi", mask_cumsum, log_g, precision=DEFAULT_PRECISION)

    active_last = active_c[..., -1:]
    m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_c - active_last) < 0.5) & (active_last > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)

    gating_forward = jnp.exp(cumsum_h)[..., None] * m_in[..., None]
    gating_last = jnp.exp(cumsum_h[..., -1])[..., None, None] * m_keep[..., None]
    bwd_diff = jnp.where(m_out > 0.5, cumsum_h[..., -1:] - cumsum_h, -1e4)
    gating_backward = jnp.exp(bwd_diff)[..., None] * m_out[..., None]
  else:
    cumsum_h = jnp.cumsum(log_g, axis=-1)
    gating_forward = jnp.exp(cumsum_h)[..., None]
    gating_last = jnp.exp(cumsum_h[..., -1])[..., None, None]
    gating_backward = jnp.exp(cumsum_h[..., -1:] - cumsum_h)[..., None]

  k_beta_g = k_h * beta_h[..., None] * gating_forward
  k_scaled_bwd = k_h * gating_backward
  if t_inv is None:
    t_inv_c = _compute_t_inv_batched(
        k_orig,
        beta_h,
        cumsum_h,
        seg_c,
        valid_c,
        repeats=repeats,
        precision=precision,
    )
  else:
    t_inv_c = jnp.transpose(t_inv.astype(jnp.float32), (1, 0, 2, 3, 4))  # [N_c, B, H_v, C, C]

  # Pre-batched GEMMs across all chunks outside any sequential loop
  w_all = jnp.matmul(t_inv_c, k_beta_g, precision=precision)  # [N_c, B, H_v, C, d_k]
  eye = jnp.eye(head_k_dim, dtype=jnp.float32)
  m_all = gating_last * eye - jnp.matmul(jnp.swapaxes(k_scaled_bwd, -1, -2), w_all, precision=precision)

  if s_ext_pass1 is not None:
    cur_m = m_all
    while cur_m.shape[0] > 1:
      n = cur_m.shape[0]
      if n % 2 == 1:
        rem = cur_m[-1:]
        cur_m = jnp.matmul(cur_m[1:-1:2], cur_m[0:-1:2], precision=precision)
        cur_m = jnp.concatenate([cur_m, rem], axis=0)
      else:
        cur_m = jnp.matmul(cur_m[1::2], cur_m[0::2], precision=precision)
    return cur_m[0], s_ext_pass1.astype(jnp.float32)

  v_orig = (
      qkv_conv[:, :, q_size + k_size :]
      .reshape(batch, num_chunks, chunk_size, num_v_heads, head_v_dim)
      .astype(jnp.float32)
  )
  v_h = jnp.transpose(v_orig, (1, 0, 3, 2, 4))
  if valid_c is not None:
    v_h = jnp.where(valid_c[..., None], v_h, 0.0)
  v_beta = v_h * beta_h[..., None]
  u_all = jnp.matmul(t_inv_c, v_beta, precision=precision)
  s_all = jnp.matmul(jnp.swapaxes(k_scaled_bwd, -1, -2), u_all, precision=precision)

  cur_m = m_all
  cur_s = s_all
  while cur_m.shape[0] > 1:
    n = cur_m.shape[0]
    if n % 2 == 1:
      rem_m, rem_s = cur_m[-1:], cur_s[-1:]
      even_m, odd_m = cur_m[0:-1:2], cur_m[1:-1:2]
      even_s, odd_s = cur_s[0:-1:2], cur_s[1:-1:2]
      next_s = jnp.matmul(odd_m, even_s, precision=precision) + odd_s
      next_m = jnp.matmul(odd_m, even_m, precision=precision)
      cur_m = jnp.concatenate([next_m, rem_m], axis=0)
      cur_s = jnp.concatenate([next_s, rem_s], axis=0)
    else:
      even_m, odd_m = cur_m[0::2], cur_m[1::2]
      even_s, odd_s = cur_s[0::2], cur_s[1::2]
      cur_s = jnp.matmul(odd_m, even_s, precision=precision) + odd_s
      cur_m = jnp.matmul(odd_m, even_m, precision=precision)
  return cur_m[0], cur_s[0]


def compose_bwd_local_from_t_inv(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    t_inv: Optional[jax.Array] = None,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    m_local_cached: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    precision: jax.lax.Precision = DEFAULT_PRECISION,
    head_major_glue: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Computes backward rank transition (dM_local, dS_ext_local) using pre-batched GEMMs.

  `head_major_glue` (on with gdn_bwd_natural_layout) keeps the math identical but changes where XLA may place the
  token-major -> head-major relayout of q / k / do: once per tensor, in bf16, on the un-repeated [N_c, B, H, C, d] view,
  pinned with `lax.optimization_barrier`, instead of on the GQA-repeated f32 tensors inside the gating fusions, which
  is where XLA's layout assignment puts it once the natural-layout backward kernel makes qkv_conv / do token-major.
  """
  batch, seq_len, _ = qkv_conv.shape
  num_chunks = seq_len // chunk_size
  repeats = num_v_heads // num_k_heads
  q_size = num_k_heads * head_k_dim
  k_size = num_k_heads * head_k_dim
  scale = 1.0 / jnp.sqrt(head_k_dim)

  if head_major_glue:

    def _head_major(x: jax.Array, heads: int, dim: int) -> jax.Array:
      """[B, S, heads * dim] -> [N_c, B, heads, C, dim], relayout pinned on this (un-repeated) tensor."""
      x = jnp.transpose(x.reshape(batch, num_chunks, chunk_size, heads, dim), (1, 0, 3, 2, 4))
      # Merging the three major dims leaves no bitcast-compatible token-major layout for this value, and the barrier
      # keeps the simplifier from cancelling the reshape pair or sinking the GQA broadcast through it.
      x = lax.optimization_barrier(x.reshape(num_chunks * batch * heads, chunk_size, dim))
      return x.reshape(num_chunks, batch, heads, chunk_size, dim)

    q_orig = _head_major(qkv_conv[:, :, :q_size], num_k_heads, head_k_dim).astype(jnp.float32)
    k_orig = _head_major(qkv_conv[:, :, q_size : q_size + k_size], num_k_heads, head_k_dim).astype(jnp.float32)
    if use_qk_norm_in_gdn:
      q_orig = q_orig * jax.lax.rsqrt(jnp.sum(q_orig**2, axis=-1, keepdims=True) + 1e-6) * scale
      k_orig = k_orig * jax.lax.rsqrt(jnp.sum(k_orig**2, axis=-1, keepdims=True) + 1e-6)
    else:
      q_orig = q_orig * scale
    q_h = jnp.repeat(q_orig, repeats, axis=2)  # [N_c, B, H_v, C, d_k]
    k_h = jnp.repeat(k_orig, repeats, axis=2)  # [N_c, B, H_v, C, d_k]
    do_h = _head_major(do.reshape(batch, seq_len, num_v_heads * head_v_dim), num_v_heads, head_v_dim).astype(
        jnp.float32
    )
  else:
    q_orig = qkv_conv[:, :, :q_size].reshape(batch, num_chunks, chunk_size, num_k_heads, head_k_dim).astype(jnp.float32)
    k_orig = (
        qkv_conv[:, :, q_size : q_size + k_size]
        .reshape(batch, num_chunks, chunk_size, num_k_heads, head_k_dim)
        .astype(jnp.float32)
    )
    if use_qk_norm_in_gdn:
      q_orig = q_orig * jax.lax.rsqrt(jnp.sum(q_orig**2, axis=-1, keepdims=True) + 1e-6) * scale
      k_orig = k_orig * jax.lax.rsqrt(jnp.sum(k_orig**2, axis=-1, keepdims=True) + 1e-6)
    else:
      q_orig = q_orig * scale

    q_rep = jnp.repeat(q_orig, repeats, axis=3)
    k_rep = jnp.repeat(k_orig, repeats, axis=3)
    q_h = jnp.transpose(q_rep, (1, 0, 3, 2, 4))  # [N_c, B, H_v, C, d_k]
    k_h = jnp.transpose(k_rep, (1, 0, 3, 2, 4))  # [N_c, B, H_v, C, d_k]
    do_h = jnp.transpose(
        do.astype(jnp.float32).reshape(batch, num_chunks, chunk_size, num_v_heads, head_v_dim),
        (1, 0, 3, 2, 4),
    )

  b_h = jnp.transpose(
      b.astype(jnp.float32).reshape(batch, num_chunks, chunk_size, num_v_heads),
      (1, 0, 3, 2),
  )
  a_h = jnp.transpose(
      a.astype(jnp.float32).reshape(batch, num_chunks, chunk_size, num_v_heads),
      (1, 0, 3, 2),
  )
  beta_h = jax.nn.sigmoid(b_h)

  a_log_f32 = a_log.astype(jnp.float32)[None, None, :, None]
  dt_bias_f32 = dt_bias.astype(jnp.float32)[None, None, :, None]
  log_g = -jnp.exp(a_log_f32) * jax.nn.softplus(a_h + dt_bias_f32)

  seg_c = None
  valid_c = None
  mask_causal_base = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)
  if segment_ids is not None:
    s_enc = local_compute_conv1d.encode_segment_ids(segment_ids.reshape(batch, seq_len), init_seg=init_seg)
    seg_c = jnp.transpose(s_enc.reshape(batch, num_chunks, chunk_size), (1, 0, 2))[:, :, None, :]
    valid_c = seg_c > 0.5
    active_c = jnp.abs(seg_c)
    active_end = active_c[..., -1]
    if init_seg is not None:
      init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(1, batch, 1))
    else:
      init_active = jnp.zeros((1, batch, 1), dtype=jnp.float32)
    seg_prev = jnp.concatenate([init_active, active_end[:-1]], axis=0)[..., None]

    q_h = jnp.where(valid_c[..., None], q_h, 0.0)
    k_h = jnp.where(valid_c[..., None], k_h, 0.0)
    do_h = jnp.where(valid_c[..., None], do_h, 0.0)
    beta_h = jnp.where(valid_c, beta_h, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)

    same_active = (jnp.abs(active_c[..., :, None] - active_c[..., None, :]) < 0.5) & (active_c[..., :, None] > 0.5)
    same_valid = (jnp.abs(seg_c[..., :, None] - seg_c[..., None, :]) < 0.5) & valid_c[..., :, None]
    mask_cumsum = mask_causal_base[None, None, None, :, :] * same_active.astype(jnp.float32)
    mask_causal = mask_causal_base[None, None, None, :, :] * same_valid.astype(jnp.float32)
    # The segment-masked cumsum feeds exp(); it stays at HIGHEST whatever `precision` is.
    cumsum_h = jnp.einsum("nbhij,nbhj->nbhi", mask_cumsum, log_g, precision=DEFAULT_PRECISION)

    active_last = active_c[..., -1:]
    m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_c - active_last) < 0.5) & (active_last > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)

    gating_forward = jnp.exp(cumsum_h)[..., None] * m_in[..., None]
    gating_last = jnp.exp(cumsum_h[..., -1])[..., None, None] * m_keep[..., None]
    bwd_diff = jnp.where(m_out > 0.5, cumsum_h[..., -1:] - cumsum_h, -1e4)
    gating_backward = jnp.exp(bwd_diff)[..., None] * m_out[..., None]
  else:
    cumsum_h = jnp.cumsum(log_g, axis=-1)
    gating_forward = jnp.exp(cumsum_h)[..., None]
    gating_last = jnp.exp(cumsum_h[..., -1])[..., None, None]
    gating_backward = jnp.exp(cumsum_h[..., -1:] - cumsum_h)[..., None]
    mask_causal = mask_causal_base

  q_g = q_h * gating_forward
  k_beta_g = k_h * beta_h[..., None] * gating_forward
  k_scaled_bwd = k_h * gating_backward
  if t_inv is None:
    t_inv_c = _compute_t_inv_batched(
        k_orig,
        beta_h,
        cumsum_h,
        seg_c,
        valid_c,
        repeats=repeats,
        precision=precision,
    )
  else:
    t_inv_c = jnp.transpose(t_inv.astype(jnp.float32), (1, 0, 2, 3, 4))

  diff = cumsum_h[..., :, None] - cumsum_h[..., None, :]
  safe_diff_causal = jnp.where(mask_causal > 0.5, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal

  # Pre-batched intra-chunk GEMMs across all chunks outside any sequential loop
  attn = jnp.matmul(q_h, jnp.swapaxes(k_h, -1, -2), precision=precision) * g_mat_causal
  dv_attn = jnp.matmul(jnp.swapaxes(attn, -1, -2), do_h, precision=precision)
  dv_beta_0 = jnp.matmul(jnp.swapaxes(t_inv_c, -1, -2), dv_attn, precision=precision)
  ds_loc_all = jnp.matmul(jnp.swapaxes(q_g, -1, -2), do_h, precision=precision) - jnp.matmul(
      jnp.swapaxes(k_beta_g, -1, -2), dv_beta_0, precision=precision
  )
  w_all = jnp.matmul(t_inv_c, k_beta_g, precision=precision)
  eye = jnp.eye(head_k_dim, dtype=jnp.float32)
  dm_all = gating_last * eye - jnp.matmul(jnp.swapaxes(w_all, -1, -2), k_scaled_bwd, precision=precision)

  cur_dm = dm_all
  cur_ds = ds_loc_all
  while cur_dm.shape[0] > 1:
    n = cur_dm.shape[0]
    if n % 2 == 1:
      rem_dm, rem_ds = cur_dm[:1], cur_ds[:1]
      even_dm, odd_dm = cur_dm[1::2], cur_dm[2::2]
      even_ds, odd_ds = cur_ds[1::2], cur_ds[2::2]
      next_ds = jnp.matmul(even_dm, odd_ds, precision=precision) + even_ds
      next_dm = jnp.matmul(even_dm, odd_dm, precision=precision)
      cur_dm = jnp.concatenate([rem_dm, next_dm], axis=0)
      cur_ds = jnp.concatenate([rem_ds, next_ds], axis=0)
    else:
      even_dm, odd_dm = cur_dm[0::2], cur_dm[1::2]
      even_ds, odd_ds = cur_ds[0::2], cur_ds[1::2]
      cur_ds = jnp.matmul(even_dm, odd_ds, precision=precision) + even_ds
      if m_local_cached is None or n > 2:
        cur_dm = jnp.matmul(even_dm, odd_dm, precision=precision)
      else:
        cur_dm = even_dm[:1]

  dm_local = jnp.swapaxes(m_local_cached.astype(jnp.float32), -1, -2) if m_local_cached is not None else cur_dm[0]
  return dm_local, cur_ds[0]


def incoming_grad_state(
    dm_loc: jax.Array,
    ds_ext_loc: jax.Array,
    dht_final: jax.Array,
    cp_axis: str | tuple[str, ...],
    precision: jax.lax.Precision = DEFAULT_PRECISION,
    *,
    fast_state_passing: bool = False,
    has_recurrent_state: bool = True,
    need_dh0_total: bool = True,
) -> Tuple[jax.Array, jax.Array]:
  """Reverse Hillis-Steele prefix scan across `cp_axis` from rank D-1 down to 0."""
  d_size = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)

  if fast_state_passing and d_size == 4 and not has_recurrent_state and dm_loc.shape == ds_ext_loc.shape:
    p1 = lax.ppermute(jnp.where(idx == 1, dm_loc, ds_ext_loc), cp_axis, [(3, 2), (2, 1), (1, 3)])
    p2 = lax.ppermute(jnp.where(idx == 3, ds_ext_loc, dm_loc), cp_axis, [(3, 0), (2, 3)])
    lhs1 = jnp.where(idx == 3, p1, dm_loc)
    rhs1 = jnp.where(idx == 3, p2, p1)
    m1_prod = jnp.matmul(lhs1, rhs1, precision=precision)
    m1_b = m1_prod + ds_ext_loc
    b_r2 = lax.ppermute(m1_b, cp_axis, [(2, 1), (1, 0)])
    a_r2 = lax.ppermute(m1_prod, cp_axis, [(3, 0)])
    dh0_on_0 = jnp.matmul(a_r2, p2, precision=precision) + b_r2
    dht_local = jnp.where(
        idx == 3,
        dht_final,
        jnp.where(idx == 2, p1, jnp.where(idx == 1, b_r2, dh0_on_0)),
    )
    if target_platform(dm_loc) == "cpu":
      dht_local = lax.optimization_barrier(dht_local)
    if not need_dh0_total:
      return dht_local, jnp.zeros_like(ds_ext_loc)
    first_b = jnp.matmul(dm_loc, dh0_on_0, precision=precision) + ds_ext_loc
    dh0_total = lax.psum(jnp.where(idx == 0, first_b, jnp.zeros_like(ds_ext_loc)), cp_axis)
    return dht_local, dh0_total

  a_run, b_run = dm_loc, ds_ext_loc
  step = 1
  while step < d_size:
    bwd = [(i + step, i) for i in range(d_size - step)]
    b_recv = lax.ppermute(b_run, cp_axis, bwd)
    live = idx < (d_size - step)
    if fast_state_passing and (step * 2 >= d_size) and not has_recurrent_state and not need_dh0_total:
      b_cmp = jnp.matmul(a_run, b_recv, precision=precision) + b_run
      b_run = jnp.where(live, b_cmp, b_run)
    else:
      a_recv = lax.ppermute(a_run, cp_axis, bwd)
      # Downstream rank (idx + step) acts first on dht_final;
      # current rank idx acts second.
      a_cmp, b_cmp = compose((a_recv, b_recv), (a_run, b_run), precision)
      a_run = jnp.where(live, a_cmp, a_run)
      b_run = jnp.where(live, b_cmp, b_run)
    step *= 2

  shift1_bwd = [(i + 1, i) for i in range(d_size - 1)]
  b_ex = lax.ppermute(b_run, cp_axis, shift1_bwd)
  if fast_state_passing and not has_recurrent_state:
    dht_local = jnp.where(idx == (d_size - 1), dht_final, b_ex)
  else:
    a_ex = lax.ppermute(a_run, cp_axis, shift1_bwd)
    carried = jnp.matmul(a_ex, dht_final, precision=precision) + b_ex
    dht_local = jnp.where(idx == (d_size - 1), dht_final, carried)

  if fast_state_passing and not need_dh0_total:
    return dht_local, jnp.zeros_like(ds_ext_loc)
  first = idx == 0
  b_tot = lax.psum(jnp.where(first, b_run, jnp.zeros_like(b_run)), cp_axis)
  if fast_state_passing and not has_recurrent_state:
    return dht_local, b_tot
  a_tot = lax.psum(jnp.where(first, a_run, jnp.zeros_like(a_run)), cp_axis)
  dh0_total = jnp.matmul(a_tot, dht_final, precision=precision) + b_tot
  return dht_local, dh0_total
