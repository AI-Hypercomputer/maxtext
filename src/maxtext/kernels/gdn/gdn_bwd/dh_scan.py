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

r"""Pallas scans that fold one sequence-sharded CP rank's GDN chunk transitions on Mosaic TPU.

Per chunk $c$ of the rank, with $w_c = T_c^{-1} k^{\beta g}_c$, the forward state map is
$h \mapsto M_c h + S_c$ with $M_c = g^{last}_c I - (k^{bwd}_c)^\top w_c$. `gdn_cp_m_scan` returns
$M_{N-1} \cdots M_0$, the `m_local` of `cp_gdn.compose_local_from_t_inv` (its `s_ext_pass1` branch): the same
terms as its balanced tree, multiplied in sequence, with every matmul at 3-pass bf16 (`Precision.HIGH`, as
`gdn_cp_matmul_precision=high` gives the tree) and the segment-masked gating cumsum at `HIGHEST`. It reads the head
groups of k / t_inv in place, with the block layout of `pallas_gdn_bwd_kernel`, and keeps the running product in
VMEM, so no [chunks, heads, d_k, d_k] f32 intermediate reaches HBM.
"""

import functools
from typing import Optional

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .. import compute_conv1d as local_compute_conv1d
from .pallas_mosaic_tpu_bwd import _gdn_matmul
from .runtime_utils import ensure_cpu_interpret_registered
from .runtime_utils import target_platform

# Value heads per grid step: all 64 heads of the Qwen3.5-397B layer in one group.
_M_SCAN_HEAD_TILE = 64

_precise_matmul = functools.partial(_gdn_matmul, compute_dtype=jnp.bfloat16, precise=True)


def _pick_head_tile(num_v_heads: int, repeats: int, target: int) -> int:
  """Largest divisor of num_v_heads up to target that holds whole GVA groups, as pallas_gdn_bwd_kernel picks it."""
  for candidate in range(min(num_v_heads, target), 0, -1):
    if num_v_heads % candidate == 0 and candidate % repeats == 0:
      return candidate
  return repeats if num_v_heads % repeats == 0 else num_v_heads


def _grouped_gate_inputs(
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: Optional[jax.Array],
    init_seg: Optional[jax.Array],
    *,
    num_chunks: int,
    chunk_size: int,
    num_groups: int,
    tile_v_heads: int,
):
  """Lays out b, a, a_log, dt_bias and the segment metadata per head group, as pallas_gdn_bwd_kernel does.

  Returns b / a as [B, G, C, chunk_size, padded], a_log / dt_bias as [B, G, 1, padded] and, per chunk, the active
  segment entering it (lane 0 of [B, G, C, 1, 128]). With segment IDs, their signed encoding rides in b's spare lane
  `tile_v_heads`, and b stays f32 then: bf16 rounds integers above 256, so distinct documents would share an ID.
  """
  batch_size = b.shape[0]
  seq_len = num_chunks * chunk_size
  has_seg_ids = segment_ids is not None
  padded = ((tile_v_heads + int(has_seg_ids) + 127) // 128) * 128
  lane_pad = ((0, 0),) * 4 + ((0, padded - tile_v_heads),)

  def per_group(x):
    x = x.reshape(batch_size, num_chunks, chunk_size, num_groups, tile_v_heads).transpose((0, 3, 1, 2, 4))
    return jnp.pad(x, lane_pad)

  def per_group_param(x):
    x = jnp.pad(x.reshape(num_groups, tile_v_heads), ((0, 0), (0, padded - tile_v_heads)))
    return jnp.broadcast_to(x[None, :, None, :], (batch_size, num_groups, 1, padded))

  b_5d = per_group(b.astype(jnp.float32) if has_seg_ids else b)
  a_5d = per_group(a)
  if has_seg_ids:
    s_enc = local_compute_conv1d.encode_segment_ids(segment_ids.reshape(batch_size, seq_len), init_seg=init_seg)
    s_enc_3d = s_enc.reshape(batch_size, num_chunks, chunk_size)
    b_5d = b_5d.at[:, :, :, :, tile_v_heads].set(s_enc_3d[:, None, :, :].astype(b_5d.dtype))
    if init_seg is not None:
      init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(batch_size, 1))
    else:
      init_active = jnp.zeros((batch_size, 1), dtype=jnp.float32)
    seg_prev = jnp.concatenate([init_active, jnp.abs(s_enc_3d[:, :-1, -1])], axis=1)
    seg_prev_hbm = jnp.pad(seg_prev[:, :, None, None], ((0, 0), (0, 0), (0, 0), (0, 127)))
  else:
    seg_prev_hbm = jnp.zeros((batch_size, num_chunks, 1, 128), dtype=jnp.float32)
  seg_prev_5d = jnp.broadcast_to(seg_prev_hbm[:, None], (batch_size, num_groups, num_chunks, 1, 128))
  return b_5d, a_5d, per_group_param(a_log), per_group_param(dt_bias), seg_prev_5d, padded


def _chunk_gates(b_ref, a_ref, a_log_ref, dt_bias_ref, seg_prev_ref, *, chunk_size, num_v_heads, has_seg_ids):
  """Segment masks and decay gates of one chunk and head group, as cp_gdn computes them.

  Returns (valid [C, 1] or None, beta_h [H, C], gating_forward [H, C, 1], gating_last [H, 1, 1],
  gating_backward [H, C, 1]).
  """
  b_val = b_ref[...][:, :num_v_heads].astype(jnp.float32)
  a_val = a_ref[...][:, :num_v_heads].astype(jnp.float32)
  a_log_val = a_log_ref[...][0, :num_v_heads].astype(jnp.float32)
  dt_bias_val = dt_bias_ref[...][0, :num_v_heads].astype(jnp.float32)
  beta = jax.nn.sigmoid(b_val)
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)
  mask_tril = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))

  valid_c = None
  mask_cumsum = mask_tril
  if has_seg_ids:
    seg_c = b_ref[...][:, num_v_heads : num_v_heads + 1].astype(jnp.float32)
    seg_prev = seg_prev_ref[...][0, 0].astype(jnp.float32)
    valid_c = seg_c > 0.5
    active_c = jnp.abs(seg_c)
    beta = jnp.where(valid_c, beta, 0.0)
    log_g = jnp.where(valid_c, log_g, 0.0)
    same_active = (jnp.abs(active_c - active_c.T) < 0.5) & (active_c > 0.5)
    mask_cumsum = mask_tril * same_active.astype(jnp.float32)

  # The cumsum feeds exp(); like cp_gdn it stays at HIGHEST (0/1 mask times f32: exact up to summation order).
  cumsum_h = jnp.transpose(jnp.dot(mask_cumsum, log_g, precision=jax.lax.Precision.HIGHEST), (1, 0))
  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  if has_seg_ids:
    m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
    m_out = ((jnp.abs(seg_c - active_c[-1:]) < 0.5) & (active_c[-1:] > 0.5)).astype(jnp.float32)
    m_keep = ((jnp.abs(active_c[-1:] - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
    gating_forward = gating_forward * m_in[None, :, :]
    gating_last = gating_last * m_keep[None, :, :]
    bwd_diff = jnp.where(m_out.T > 0.5, cumsum_h[:, -1:] - cumsum_h, -1e4)
    gating_backward = jnp.exp(bwd_diff)[:, :, None] * m_out[None, :, :]
  else:
    gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]
  beta_h = jnp.transpose(beta, (1, 0))
  return valid_c, beta_h, gating_forward, gating_last, gating_backward


def _unit_rows(x: jax.Array, use_qk_norm: bool) -> jax.Array:
  """The q / k l2 normalization of cp_gdn: x * rsqrt(sum(x**2) + 1e-6) over the head dim."""
  if not use_qk_norm:
    return x
  return x * jax.lax.rsqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-6)


def _m_scan_body(
    k_ref,
    b_ref,
    a_ref,
    t_inv_ref,
    a_log_ref,
    dt_bias_ref,
    seg_prev_ref,
    m_ref,
    acc_scr,
    *,
    chunk_size: int,
    num_kq_heads: int,
    num_v_heads: int,
    kq_head_dim: int,
    num_chunks: int,
    has_seg_ids: bool,
    use_qk_norm: bool,
):
  """Forward step per (batch, group, chunk): acc <- M_c acc = g_last acc - k_bwd^T (w (acc)), from acc = I."""
  c = pl.program_id(2)

  @pl.when(c == 0)
  def _init():
    rows = jax.lax.broadcasted_iota(jnp.int32, (kq_head_dim, kq_head_dim), 0)
    cols = jax.lax.broadcasted_iota(jnp.int32, (kq_head_dim, kq_head_dim), 1)
    acc_scr[...] = jnp.broadcast_to((rows == cols).astype(jnp.float32)[None], acc_scr.shape)

  valid_c, beta_h, gating_forward, gating_last, gating_backward = _chunk_gates(
      b_ref,
      a_ref,
      a_log_ref,
      dt_bias_ref,
      seg_prev_ref,
      chunk_size=chunk_size,
      num_v_heads=num_v_heads,
      has_seg_ids=has_seg_ids,
  )
  k_orig = k_ref[...].reshape((chunk_size, num_kq_heads, kq_head_dim)).astype(jnp.float32)
  if valid_c is not None:
    k_orig = jnp.where(valid_c[:, :, None], k_orig, 0.0)
  k_h = jnp.transpose(jnp.repeat(_unit_rows(k_orig, use_qk_norm), num_v_heads // num_kq_heads, axis=1), (1, 0, 2))
  k_beta_g = k_h * beta_h[:, :, None] * gating_forward
  k_bwd = k_h * gating_backward

  acc = acc_scr[...]
  w = _precise_matmul(t_inv_ref[...].astype(jnp.float32), k_beta_g)
  acc_new = acc * gating_last - _precise_matmul(jnp.swapaxes(k_bwd, -1, -2), _precise_matmul(w, acc))
  acc_scr[...] = acc_new

  @pl.when(c == num_chunks - 1)
  def _store():
    m_ref[0, ...] = acc_new


def _run_scan(body, args, in_specs, out_shape, out_spec, acc_shape, *, grid, interpret, name):
  """One pallas_call that pipelines `body` over grid=(B, G, C) with the running product in a VMEM scratch."""

  def outer(*refs):
    pltpu.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=[out_spec])(
        *refs[: len(args) + 1], scratches=tuple(refs[len(args) + 1 :])
    )

  hbm = pltpu.MemorySpace.HBM
  return pl.pallas_call(
      outer,
      grid=(),
      out_shape=out_shape,
      in_specs=[pl.BlockSpec(memory_space=hbm)] * len(args),
      out_specs=pl.BlockSpec(memory_space=hbm),
      scratch_shapes=[pltpu.VMEM(acc_shape, jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes),
          disable_bounds_checks=True,
      ),
      interpret=interpret,
      name=name,
  )(*args)


def _resolve_interpret(x: jax.Array, interpret: bool | pltpu.InterpretParams | None):
  if interpret is None:
    interpret = target_platform(x) == "cpu"
  if interpret:
    ensure_cpu_interpret_registered()
  return interpret


def gdn_cp_m_scan(
    k_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    t_inv: jax.Array,
    *,
    num_v_heads: int,
    kq_head_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    segment_ids: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    interpret: bool | pltpu.InterpretParams | None = None,
) -> jax.Array:
  """M_local of `cp_gdn.compose_local_from_t_inv` (s_ext_pass1 branch), [B, num_v_heads, d_k, d_k] f32.

  Args:
    k_conv: [B, S, num_kq_heads * d_k] conv output of the K channels, as passed to compose_local_from_t_inv.
    b, a: [B, S, num_v_heads] beta and decay pre-activations.
    a_log, dt_bias: [num_v_heads] decay parameters.
    t_inv: [B, S / chunk_size, num_v_heads, chunk_size, chunk_size] per-chunk T^{-1} of the forward kernel.
    num_v_heads, kq_head_dim, chunk_size, use_qk_norm_in_gdn: GDN layer geometry.
    segment_ids, init_seg: Rank-local segment metadata, as passed to compose_local_from_t_inv.
    interpret: Pallas interpret mode; None interprets on CPU.
  """
  interpret = _resolve_interpret(k_conv, interpret)
  batch_size, seq_len, k_size = k_conv.shape
  num_chunks = seq_len // chunk_size
  num_kq_heads = k_size // kq_head_dim
  repeats = num_v_heads // num_kq_heads
  tile_v_heads = _pick_head_tile(num_v_heads, repeats, _M_SCAN_HEAD_TILE)
  num_groups = num_v_heads // tile_v_heads
  tile_k_size = (tile_v_heads // repeats) * kq_head_dim
  has_seg_ids = segment_ids is not None

  b_5d, a_5d, a_log_4d, dt_bias_4d, seg_prev_5d, padded = _grouped_gate_inputs(
      b,
      a,
      a_log,
      dt_bias,
      segment_ids,
      init_seg,
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      num_groups=num_groups,
      tile_v_heads=tile_v_heads,
  )
  args = [
      k_conv.reshape(batch_size, num_chunks, chunk_size, k_size),
      b_5d,
      a_5d,
      t_inv.reshape(batch_size, num_chunks, num_groups, tile_v_heads, chunk_size, chunk_size).astype(jnp.float32),
      a_log_4d,
      dt_bias_4d,
      seg_prev_5d,
  ]
  in_specs = [
      # k [B, C, chunk_size, k_size], columns of group g
      pl.BlockSpec((None, None, chunk_size, tile_k_size), lambda b_, g, c: (b_, c, 0, g)),
      # b, a [B, G, C, chunk_size, padded]
      pl.BlockSpec((None, None, None, chunk_size, padded), lambda b_, g, c: (b_, g, c, 0, 0)),
      pl.BlockSpec((None, None, None, chunk_size, padded), lambda b_, g, c: (b_, g, c, 0, 0)),
      # t_inv [B, C, G, tile_v_heads, chunk_size, chunk_size]
      pl.BlockSpec((None, None, None, tile_v_heads, chunk_size, chunk_size), lambda b_, g, c: (b_, c, g, 0, 0, 0)),
      # a_log, dt_bias [B, G, 1, padded]
      pl.BlockSpec((None, None, 1, padded), lambda b_, g, c: (b_, g, 0, 0)),
      pl.BlockSpec((None, None, 1, padded), lambda b_, g, c: (b_, g, 0, 0)),
      # segment entering the chunk [B, G, C, 1, 128]
      pl.BlockSpec((None, None, None, 1, 128), lambda b_, g, c: (b_, g, c, 0, 0)),
  ]
  acc_shape = (tile_v_heads, kq_head_dim, kq_head_dim)
  body = functools.partial(
      _m_scan_body,
      chunk_size=chunk_size,
      num_kq_heads=tile_v_heads // repeats,
      num_v_heads=tile_v_heads,
      kq_head_dim=kq_head_dim,
      num_chunks=num_chunks,
      has_seg_ids=has_seg_ids,
      use_qk_norm=use_qk_norm_in_gdn,
  )
  m_6d = _run_scan(
      body,
      args,
      in_specs,
      jax.ShapeDtypeStruct((batch_size, num_groups, 1) + acc_shape, jnp.float32),
      pl.BlockSpec((None, None, 1) + acc_shape, lambda b_, g, c: (b_, g, 0, 0, 0, 0)),
      acc_shape,
      grid=(batch_size, num_groups, num_chunks),
      interpret=interpret,
      name="gdn_cp_m_scan",
  )
  return m_6d[:, :, 0].reshape(batch_size, num_v_heads, kq_head_dim, kq_head_dim)
