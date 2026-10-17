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

"""Core GDN forward computation with triangular inverse matrix caching."""

import dataclasses

import jax
import jax.numpy as jnp

from . import config

# ==============================================================================
# Triangular-inverse algorithm selection.
#
# `xor_dnc` is the default algorithm (pure-MXU divide-and-conquer with a
# half-row final merge). `baseline_b16` is the legacy 16-row block forward
# substitution algorithm. Both compute (I + L)^-1 for a batch of unit
# lower-triangular matrices `t: [batch, chunk, chunk]`.
# ==============================================================================

DEFAULT_TRI_INV_ALGO = "xor_dnc"
TRI_INV_ALGOS = ("xor_dnc", "baseline_b16")


def validate_tri_inv_algo(algo: str) -> str:
  if algo not in TRI_INV_ALGOS:
    raise ValueError(f"Unknown triangular-inverse algorithm {algo!r}; expected one of {TRI_INV_ALGOS}.")
  return algo


def default_head_group_size(
    chunk_size: int,
    num_kq_heads: int,
    num_v_heads: int,
    act_dtype: jnp.dtype | None = None,
) -> int | None:
  """Returns the default KQ-head group size for `chunk_size`, or None for all heads at once.

  At `chunk_size > 64`, processing all heads at once can exceed VMEM with `[num_v_heads,
  chunk_size, chunk_size]` intermediates. Bounding each group to at most 32 value heads
  for `bfloat16` activations (or 8 value heads for `float32` activations) at `chunk_size = 128`
  keeps peak VMEM comfortably within budget while preserving full MXU utilization.
  """
  if chunk_size <= 64 or num_kq_heads <= 0 or num_v_heads <= 0:
    return None
  v_per_kq = max(1, num_v_heads // num_kq_heads)
  base_v_heads = 8 if (act_dtype is not None and jnp.dtype(act_dtype) == jnp.float32) else 32
  max_v_heads = max(v_per_kq, (base_v_heads * 128) // chunk_size)
  max_kq = max(1, max_v_heads // v_per_kq)
  if num_kq_heads <= max_kq:
    return None
  for g in range(max_kq, 0, -1):
    if num_kq_heads % g == 0:
      return g
  return 1


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
  return x / norm


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
  match jnp.dtype(dtype).itemsize:
    case 4:
      return jnp.int32
    case 2:
      return jnp.int16
    case _:
      raise ValueError(f"Unsupported dtype: {dtype}")


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def _tri_inv_baseline_b16(
    t: jax.Array,
    block_size: int = 16,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
  """Compute inverse matrix of a given triangular matrix using 16-row block forward substitution."""

  # NOTE: if chunk_size=1, compiler will perform DCE.
  out_dtype = t.dtype
  chunk = t.shape[-1]
  block_size = min(block_size, chunk)
  num_blocks = chunk // block_size

  def local_forward_sub(t_mat: jax.Array, b_mat: jax.Array) -> jax.Array:
    x_list = []
    for i in range(block_size):
      b_i = b_mat[:, i, :]
      if i == 0:
        x_i = b_i
      else:
        stacked_x = jnp.stack(x_list, axis=1)
        all_prev_t = t_mat[:, i, :i]
        prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
        x_i = b_i - prev_sum
      x_list.append(x_i)
    return jnp.stack(x_list, axis=1)

  x_blocks = []
  iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
  iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
  identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = identity_mask[:, start:end, :]

    if i == 0:
      target_b = e_block
    else:
      interaction_t = t[:, start:end, :start]
      solved_x = jnp.concatenate(x_blocks, axis=1)
      prev_sum = jax.lax.dot(
          interaction_t,
          solved_x,
          dimension_numbers=(((2,), (1,)), ((0,), (0,))),
          precision=precision,
          preferred_element_type=jnp.float32,
      )
      target_b = e_block - prev_sum

    # NOTE: Utilize fp32 to minimize cost of sublane rolling.
    local_t = t[:, start:end, start:end].astype(jnp.float32)
    x_block = local_forward_sub(local_t, target_b)
    x_blocks.append(x_block.astype(out_dtype))

  return jnp.concatenate(x_blocks, axis=1)


def _bmm(lhs: jax.Array, rhs: jax.Array, precision: jax.lax.Precision | None) -> jax.Array:
  """Batched [n, a, b] @ [n, b, c] -> [n, a, c] with f32 accumulation."""
  return jax.lax.dot_general(
      lhs,
      rhs,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      precision=precision,
      preferred_element_type=jnp.float32,
  )


def _row_col_iotas(shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
  """Broadcastable [1, R, C] row and column int32 iotas for the last two dims of `shape`."""
  mask_shape = (1, shape[-2], shape[-1])
  return (
      jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1),
      jax.lax.broadcasted_iota(jnp.int32, mask_shape, 2),
  )


def _dnc_merge_levels(
    x: jax.Array,
    lower_part: jax.Array,
    first_block: int,
    precision: jax.lax.Precision | None,
    xor_rc: jax.Array | None = None,
) -> jax.Array:
  """Merges a block-diagonal inverse with `first_block`-sized blocks up to full size.

  At level b every 2b-window [[A, 0], [B, C]] gets its lower-left block
  -C^-1 B A^-1, i.e. X <- X - X ((M_b * L) X) at full width, where M_b selects the
  lower-left bxb block of each 2b-window ((r ^ c) in [b, 2b) and r > c).

  Args:
    x: [n, c, c] f32 block-diagonal inverse with `first_block`-sized blocks.
    lower_part: [n, c, c] f32 lower-triangular matrix (zeros above the diagonal)
      whose strictly lower part is L of T = I + L.
    first_block: Size of the diagonal blocks already inverted in `x`.
    precision: Matmul precision.
    xor_rc: Optional precomputed `bitwise_xor(iota_r, iota_c)` of `[1, c, c]`.

  Returns:
    [n, c, c] f32 inverse of T.
  """
  chunk = x.shape[-1]
  if xor_rc is None:
    iota_r, iota_c = _row_col_iotas(x.shape)
    xor_rc = jnp.bitwise_xor(iota_r, iota_c)
  b = first_block
  while b < chunk:
    if 2 * b == chunk and b >= 8:
      lower_bot = jnp.where(xor_rc[:, b:, :] >= b, lower_part[:, b:, :], 0.0)
      mid = _bmm(lower_bot, x, precision)
      mid_padded = jnp.concatenate([jnp.zeros_like(mid), mid], axis=1)
      bot = x[:, b:, :] - _bmm(x[:, b:, :], mid_padded, precision)
      x = jnp.concatenate([x[:, :b, :], bot], axis=1)
    else:
      level_mask = (xor_rc >= b) & (xor_rc < 2 * b)
      lower_b = jnp.where(level_mask, lower_part, 0.0)
      x = x - _bmm(x, _bmm(lower_b, x, precision), precision)
    b *= 2
  return x


def _tri_inv_xor_dnc(t: jax.Array, precision: jax.lax.Precision | None = None) -> jax.Array:
  """Pure-MXU divide and conquer (gdm/universal_ai/clup/delta_rule.py), 2(log2 C - 1) matmuls."""
  chunk = t.shape[-1]
  if chunk <= 1 or (chunk & (chunk - 1)) != 0:
    return _tri_inv_baseline_b16(t, precision=precision)
  out_dtype = t.dtype
  t32 = t.astype(jnp.float32)
  iota_r, iota_c = _row_col_iotas(t.shape)
  xor_rc = jnp.bitwise_xor(iota_r, iota_c)
  x = jnp.where(xor_rc == 0, 1.0, jnp.where(xor_rc == 1, -t32, 0.0))
  return _dnc_merge_levels(x, t32, 2, precision, xor_rc).astype(out_dtype)


def invert_triangular_matrix(
    t: jax.Array,
    block_size: int = 16,
    precision: jax.lax.Precision | None = None,
    *,
    algo: str = DEFAULT_TRI_INV_ALGO,
) -> jax.Array:
  """Computes (I + L)^-1 for unit lower-triangular `t: [batch, chunk, chunk]`.

  Args:
    t: Unit lower-triangular matrices (ones on the diagonal, zeros above it).
    block_size: Block size of the `baseline_b16` algorithm.
    precision: Matmul precision.
    algo: One of `TRI_INV_ALGOS`; defaults to `"xor_dnc"`.

  Returns:
    The inverses, in `t.dtype`.
  """
  validate_tri_inv_algo(algo)
  if algo == "xor_dnc":
    return _tri_inv_xor_dnc(t, precision)
  return _tri_inv_baseline_b16(t, block_size, precision)


def fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int) -> jax.Array:
  """Perform 1D transpose where results are broadcasted along src_dim."""
  assert x.shape[dst_dim] == 1
  # NOTE: Reshape performs better than squeeze / expand_dims.
  shape = list(x.shape)
  shape[dst_dim] = shape[src_dim]
  shape[src_dim] = 1
  return jnp.swapaxes(x, src_dim, dst_dim).reshape(shape)


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, chunk, aligned_num_v_heads]
    beta: jax.Array,  # [1, chunk, aligned_num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
    seg_c_col: jax.Array | None = None,  # [1, chunk, 1]
    seg_aux_col: jax.Array | None = None,  # [1, chunk, 1]
) -> tuple[jax.Array | None, jax.Array, jax.Array | None]:
  """Perform chunked GDN over input [num_heads, chunk, head_dim]; out is None in PER_SEQ_STATES mode.

  With `cfg.head_group_size` set, kq heads (and their value heads) are processed
  `head_group_size` at a time, which bounds the live [heads, chunk, chunk]
  intermediates at large chunk sizes.
  """
  group = cfg.head_group_size
  if group is None or group >= cfg.num_kq_heads:
    return _chunked_gdn_per_seq_heads(
        q_large, k_large, v_large, gating_log, beta, state_prev, cfg, seg_c_col=seg_c_col, seg_aux_col=seg_aux_col
    )
  if cfg.num_kq_heads % group != 0:
    raise ValueError(f"head_group_size={group} must divide num_kq_heads={cfg.num_kq_heads}.")
  group_v = group * cfg.v_per_kq_head
  group_cfg = dataclasses.replace(cfg, num_kq_heads=group, num_v_heads=group_v, head_group_size=None)
  # The decay cumsum is a chunk-long serial chain on one lane-dense vreg row that
  # every head shares: compute it once instead of once per head group. The lane ->
  # head transposes stay per group, so only that group's lane-padded
  # [heads, chunk, 1] columns are live at a time.
  valid_c, active_c, _ = _segment_columns(cfg, seg_c_col, seg_aux_col)
  if valid_c is not None:
    gating_log = jnp.where(valid_c, gating_log, 0.0)
    beta = jnp.where(valid_c, beta, 0.0)
  # [1, chunk, aligned_num_v_heads]
  g_cum_sum_all = _decay_cumsum(gating_log, cfg, active_c)
  outs, states, t_invs = [], [], []
  for g in range(cfg.num_kq_heads // group):
    kq_sl = slice(g * group, (g + 1) * group)
    v_sl = slice(g * group_v, (g + 1) * group_v)
    out, state, t_inv = _chunked_gdn_per_seq_heads(
        q_large[kq_sl],
        k_large[kq_sl],
        v_large[v_sl],
        None,
        beta[:, :, v_sl],
        state_prev[v_sl],
        group_cfg,
        seg_c_col=seg_c_col,
        seg_aux_col=seg_aux_col,
        decay_cumsum=g_cum_sum_all[:, :, v_sl],
    )
    if out is not None:
      outs.append(out)
    states.append(state)
    if cfg.emit_t_inv:
      t_invs.append(t_inv)
  out_cat = jnp.concatenate(outs, axis=0) if outs else None
  t_inv_cat = jnp.concatenate(t_invs, axis=0) if cfg.emit_t_inv else None
  return out_cat, jnp.concatenate(states, axis=0), t_inv_cat


def _segment_columns(
    cfg: config.GDNConfig,
    seg_c_col: jax.Array | None,
    seg_aux_col: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
  """(valid_c, active_c, seg_prev) [1, chunk, 1] columns for packed sequences, else Nones."""
  if cfg.has_seg_ids and seg_c_col is not None and seg_aux_col is not None:
    valid_c = seg_c_col > 0.5
    active_c = jnp.abs(seg_c_col)
    seg_prev = jnp.abs(seg_aux_col[:, cfg.prev_kernel_size : cfg.prev_kernel_size + 1, :])
    return valid_c, active_c, seg_prev
  return None, None, None


def _decay_cumsum(
    gating_log: jax.Array,  # [1, chunk, lanes], masked for packed sequences
    cfg: config.GDNConfig,
    active_c: jax.Array | None,
) -> jax.Array:
  """[1, chunk, lanes] f32 cumulative log-decay (restarting at document starts when packed)."""
  # Compute cumulative sum of decay in float32 to prevent bf16 cumsum drift.
  gating_log = gating_log.astype(jnp.float32)
  # [1, 1, num_v_heads]
  g_cum_sum_list = [gating_log[:, :1]]
  for row in range(1, cfg.chunk_size):
    if cfg.has_seg_ids and active_c is not None:
      same_as_prev = jnp.abs(active_c[:, row : row + 1] - active_c[:, row - 1 : row]) < 0.5
      prev_sum = jnp.where(same_as_prev, g_cum_sum_list[-1], 0.0)
      g_cum_sum_list.append(prev_sum + gating_log[:, row : row + 1])
    else:
      g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
  # [1, chunk, num_v_heads]
  return jnp.concat(g_cum_sum_list, axis=1)


def _chunked_gdn_per_seq_heads(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array | None,  # [1, chunk, >= num_v_heads]; None with decay_cumsum
    beta: jax.Array,  # [1, chunk, >= num_v_heads]; already masked with decay_cumsum
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
    seg_c_col: jax.Array | None = None,  # [1, chunk, 1]
    seg_aux_col: jax.Array | None = None,  # [1, chunk, 1]
    decay_cumsum: jax.Array | None = None,  # [1, chunk, >= num_v_heads]
) -> tuple[jax.Array | None, jax.Array, jax.Array]:
  """Perform chunked GDN over all heads of input [num_heads, chunk, head_dim]; out is None in PER_SEQ_STATES mode.

  `decay_cumsum` is the precomputed (`_decay_cumsum`) cumulative log-decay of these
  heads; it replaces `gating_log`, and `beta` must then already be masked.
  """
  # bf16 activations keep single-pass MXU matmuls even with an f32 compute dtype
  # (HIGHEST would triple the forward MXU cost); HIGHEST needs f32 activations and compute.
  dot_prec = (
      jax.lax.Precision.HIGHEST
      if jnp.dtype(cfg.dtypes.act_in) == jnp.float32 and jnp.dtype(cfg.dtypes.compute) == jnp.float32
      else jax.lax.Precision.DEFAULT
  )

  valid_c, active_c, seg_prev = _segment_columns(cfg, seg_c_col, seg_aux_col)
  if valid_c is not None:
    q_large = jnp.where(valid_c, q_large, 0.0)
    k_large = jnp.where(valid_c, k_large, 0.0)
    v_large = jnp.where(valid_c, v_large, 0.0)
    if decay_cumsum is None:
      gating_log = jnp.where(valid_c, gating_log, 0.0)
      beta = jnp.where(valid_c, beta, 0.0)

  # NOTE: Repeat along non lane/sublane dim is free.
  # [num_v_heads, chunk, kq_head_dim]
  q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
  k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

  # [1, chunk, num_v_heads]
  g_cum_sum_log = _decay_cumsum(gating_log, cfg, active_c) if decay_cumsum is None else decay_cumsum

  # [num_v_heads, chunk, 1]
  g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)
  g_cum_sum_log = g_cum_sum_log[: cfg.num_v_heads]
  beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
  beta_large = beta[: cfg.num_v_heads]

  # [num_v_heads, 1, chunk]
  g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
  # [num_v_heads, chunk, chunk]
  g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t

  if cfg.has_seg_ids and seg_c_col is not None and valid_c is not None and active_c is not None and seg_prev is not None:
    seg_c_t = fused_transpose_broadcast(seg_c_col, src_dim=1, dst_dim=2)
    same_doc_valid = (jnp.abs(seg_c_col - seg_c_t) < 0.5) & valid_c
    g_cum_sum_diff_log_safe = jnp.where(same_doc_valid, g_cum_sum_diff_log, -1e4)
    gating_map = jnp.where(same_doc_valid, jnp.exp(g_cum_sum_diff_log_safe), 0.0).astype(cfg.dtypes.compute)

    active_last = active_c[:, -1:, :]
    m_in = ((jnp.abs(active_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(g_cum_sum_log.dtype)
    m_out = ((jnp.abs(active_c - active_last) < 0.5) & (active_last > 0.5)).astype(g_cum_sum_log.dtype)
    m_keep = ((jnp.abs(active_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(g_cum_sum_log.dtype)

    bwd_diff_safe = jnp.where(m_out > 0.5, g_cum_sum_log[:, -1:] - g_cum_sum_log, -1e4)
    gating_backward = (jnp.exp(bwd_diff_safe) * m_out).astype(cfg.dtypes.compute)
    gating_forward = (jnp.exp(g_cum_sum_log) * m_in).astype(cfg.dtypes.compute)
    gating_last = (jnp.exp(g_cum_sum_log[:, -1:]) * m_keep).astype(cfg.dtypes.compute)
  else:
    gating_map = jnp.exp(g_cum_sum_diff_log).astype(cfg.dtypes.compute)
    # [num_v_heads, chunk, 1]
    gating_backward = jnp.exp(g_cum_sum_log[:, -1:] - g_cum_sum_log).astype(cfg.dtypes.compute)
    # [num_v_heads, chunk, 1]
    gating_forward = jnp.exp(g_cum_sum_log).astype(cfg.dtypes.compute)
    # [num_v_heads, 1, 1]
    gating_last = gating_forward[:, -1:]

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  mask_shape = (1, cfg.chunk_size, cfg.chunk_size)
  iota_r = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 1)
  iota_c = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 2)
  gating_map = jnp.where(iota_r >= iota_c, gating_map, 0)

  # Compute k @ k^T at num_kq_heads (4x fewer matmuls) before repeating to num_v_heads.
  k_k_t = jax.lax.dot(
      k_large,
      k_large,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)
  beta_k_k_t = jnp.repeat(k_k_t, cfg.v_per_kq_head, axis=0) * beta_large
  gating_beta_k_k_t = jnp.where(iota_r > iota_c, gating_map * beta_k_k_t, 0)
  t = jnp.where(iota_r == iota_c, 1, gating_beta_k_k_t)

  # [num_v_heads, chunk, chunk]
  t_inv = invert_triangular_matrix(t, precision=dot_prec, algo=cfg.tri_inv_algo)

  # [num_v_heads, chunk, v_head_dim]
  v_beta_large = v_large * beta_large
  # [num_v_heads, chunk, kv_head_dim]
  k_beta_gating = k_repeat * (beta_large * gating_forward)
  # NOTE: If v_head_dim < mxu size, concatenating them will help increase mxu
  # utilization. Also, if v_head_dim is multiple of lane size, concat / split
  # along lane dim is free - making this optimization strictly beneficial.
  # [num_v_heads, chunk, v_head_dim + kq_head_dim]
  merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
  merged_uw = jax.lax.dot(
      t_inv,
      merged_v_k,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)

  # [num_v_heads, chunk, v_head_dim]
  u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

  states_only = cfg.mode == config.GDNMode.PER_SEQ_STATES
  if states_only:
    # No output: only the w rows of the merged matmul below. Each row's dot product is the same,
    # so ws is bit-identical.
    # [num_v_heads, chunk, v_head_dim]
    ws = jax.lax.dot(
        w,
        state_prev,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=dot_prec,
        preferred_element_type=jnp.float32,
    )
  else:
    # [num_v_heads, chunk, kq_head_dim]
    q_large_gating = q_repeat * gating_forward
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # [num_v_heads, 2 * chunk, kq_head_dim]
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    # [num_v_heads, 2 * chunk, v_head_dim]
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=dot_prec,
        preferred_element_type=jnp.float32,
    )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
  ws = ws.astype(cfg.dtypes.compute)

  # [num_v_heads, chunk, v_head_dim]
  u_ws = u - ws

  # [num_v_heads, chunk, kq_head_dim]
  k_repeat_gating = k_repeat * gating_backward

  # [num_v_heads, kq_head_dim, v_head_dim]
  state_new = jax.lax.dot(
      k_repeat_gating,
      u_ws,
      dimension_numbers=(((1,), (1,)), ((0,), (0,))),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  )

  # [num_v_heads, kq_head_dim, v_head_dim]
  state_updated = state_prev * gating_last
  state = state_updated + state_new
  if states_only:
    return None, state, t_inv

  # [num_kq_heads, chunk, chunk]
  out_qk = jax.lax.dot(
      q_large,
      k_large,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(cfg.dtypes.compute)
  # NOTE: must perform repeat after matmul to reduce required compute.
  # [num_v_heads, chunk, chunk]
  out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
  out_qk *= gating_map

  # [num_v_heads, chunk, v_head_dim]
  out_new = jax.lax.dot(
      out_qk,
      u_ws,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  )
  out = out_updated + out_new
  if valid_c is not None:
    out = jnp.where(valid_c, out, 0.0)

  return out, state, t_inv


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array | None, jax.Array, jax.Array | None]:
  """Perform chunked GDN over input [seq, num_heads, chunk, head_dim]; out is None in PER_SEQ_STATES mode."""

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2)
  mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

  seg_c_per_seq = []
  seg_aux_per_seq = []
  if cfg.has_seg_ids:
    b_f32 = b_large.astype(jnp.float32)
    for idx in range(cfg.seq_tile_size):
      b_trans = fused_transpose_broadcast(b_f32[idx], src_dim=2, dst_dim=0)
      seg_c_per_seq.append(b_trans[cfg.num_v_heads : cfg.num_v_heads + 1])
      seg_aux_per_seq.append(b_trans[cfg.num_v_heads + 1 : cfg.num_v_heads + 2])

  # [seqs, num_kq_heads, chunk, kq_head_dim]
  q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
  k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
  # [seqs, num_v_heads, chunk, v_head_dim]
  v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

  b_large = b_large.astype(cfg.dtypes.compute)

  a_log_f32 = a_log.reshape(1, 1, 1, -1).astype(jnp.float32)
  dt_bias_f32 = dt_bias.reshape(1, 1, 1, -1).astype(jnp.float32)

  # NOTE: Any element-wise computations should occur before repeat.
  # [seqs, num_kq_heads, chunk, kq_head_dim]
  if cfg.use_qk_norm_in_gdn:
    q_large = l2_norm(q_large)
    k_large = l2_norm(k_large)
  q_scale = cfg.kq_head_dim**-0.5
  q_large *= q_scale

  # [seqs, 1, chunk, num_v_heads]
  beta = jax.nn.sigmoid(b_large)
  gating_log = -jnp.exp(a_log_f32) * jax.nn.softplus(a_large.astype(jnp.float32) + dt_bias_f32)

  beta = jnp.where(mask, beta, 0)
  # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
  # be masked to 1 since it signifies that strength of state from previous row
  # will be 1 (i.e., no decay) if current row is invalid.
  gating_log = jnp.where(mask, gating_log, 0.0)

  out_list = []
  state_list = []
  t_inv_list = []
  for idx in range(cfg.seq_tile_size):
    out, state, t_inv = chunked_gdn_per_seq(
        q_large[idx],
        k_large[idx],
        v_large[idx],
        gating_log[idx],
        beta[idx],
        state_prev[idx],
        cfg,
        seg_c_col=seg_c_per_seq[idx] if cfg.has_seg_ids else None,
        seg_aux_col=seg_aux_per_seq[idx] if cfg.has_seg_ids else None,
    )
    if out is not None:
      out_list.append(out.swapaxes(0, 1))
    state_list.append(state)
    if cfg.emit_t_inv:
      t_inv_list.append(t_inv)
  out = jnp.stack(out_list, axis=0) if out_list else None
  state = jnp.stack(state_list, axis=0)
  t_inv = jnp.stack(t_inv_list, axis=0) if cfg.emit_t_inv else None
  return out, state, t_inv


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim]."""

  out_list = []
  for c_idx in range(cfgs.chunk_size):
    # [num_v_heads, 1, kq_head_dim]
    q_curr = q_compact[:, c_idx]
    q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
    k_curr = k_compact[:, c_idx]
    k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

    # [num_v_heads, 1, v_head_dim]
    v_curr = v_compact[:, c_idx]

    # [num_v_heads, kq_head_dim, 1]
    k_curr_t = k_compact_t[:, c_idx]
    k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

    # [num_v_heads, 1, 1]
    beta_curr = beta[:, c_idx]
    gating_curr = gating_log[:, c_idx]

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_updated = state * gating_curr

    # [num_v_heads, 1, v_head_dim]
    v_updated = jax.lax.dot(
        k_curr,
        state_updated,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # [num_v_heads, 1, v_head_dim]
    v_diff = v_curr - v_updated
    v_new = beta_curr * v_diff

    # [num_v_heads, kq_head_dim, v_head_dim]
    # NOTE: Multiplication with k_curr_t needs to be deferred as much as
    # possible as it expands the dimension size by kq_head_dim.
    state_new = k_curr_t * v_new
    # [num_v_heads, kq_head_dim, v_head_dim]
    state = state_updated + state_new

    # [num_v_heads, 1, v_head_dim]
    out = jax.lax.dot(
        q_curr,
        state,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    out_list.append(out[:, 0, :])

  return jnp.stack(out_list, axis=0), state


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim]."""

  mask_dtype = get_mask_dtype(cfg.dtypes.compute)
  iota = jax.lax.broadcasted_iota(mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2)
  mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

  # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
  q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
  k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
  # [seqs, num_v_heads, chunk, 1, v_head_dim]
  v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

  b_compact = b_compact.astype(cfg.dtypes.compute)

  a_log_f32 = a_log.reshape(1, 1, 1, 1, -1).astype(jnp.float32)
  dt_bias_f32 = dt_bias.reshape(1, 1, 1, 1, -1).astype(jnp.float32)

  # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
  if cfg.use_qk_norm_in_gdn:
    q_compact = l2_norm(q_compact)
    k_compact = l2_norm(k_compact)
  q_scale = cfg.kq_head_dim**-0.5
  q_compact *= q_scale
  k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

  beta = jax.nn.sigmoid(b_compact)
  gating_log = -jnp.exp(a_log_f32) * jax.nn.softplus(a_compact.astype(jnp.float32) + dt_bias_f32)

  beta = jnp.where(mask, beta, 0)
  # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
  # be masked to 1 since it signifies that strength of state from previous row
  # will be 1 (i.e., no decay) if current row is invalid.
  gating_log = jnp.where(mask, gating_log, 0.0)
  gating_log = jnp.exp(gating_log).astype(cfg.dtypes.compute)

  beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
  beta = beta[:, : cfg.num_v_heads]
  gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
  gating_log = gating_log[:, : cfg.num_v_heads]

  out_list = []
  new_state_list = []

  for idx in range(cfg.seq_tile_size):
    out, state = recurrent_gdn_per_seq(
        q_compact[idx],
        k_compact[idx],
        k_compact_t[idx],
        v_compact[idx],
        gating_log[idx],
        beta[idx],
        state_prev[idx],
        cfg,
    )
    out_list.append(out)
    new_state_list.append(state)

  out = jnp.stack(out_list, axis=0)
  new_recurrent_state = jnp.stack(new_state_list, axis=0)
  t_inv = jnp.ones((cfg.seq_tile_size, cfg.num_v_heads, 1, 1), dtype=cfg.dtypes.compute)

  return out, new_recurrent_state, t_inv
