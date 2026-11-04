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
"""Context-parallel evaluation of the GatedDeltaNet inter-chunk recurrence.

The recurrence h_new = A @ h + B is affine in the state, so it composes
associatively and can be split across a sharded sequence:

    A_i = exp(g_last).I - k_g^T.w_i      B_i = k_g^T.u_i

Each device folds its local chunks into one (A, B) pair, the pairs compose
across devices with a prefix scan, and each device then replays its chunks from
the state that reaches it. Composition is associative but not commutative, so
device order is the sequence order. `context_parallel_load_balance` permutes
that order and `configs/types.py` rejects the combination.
"""

from __future__ import annotations

import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

_PREC = jax.lax.Precision.HIGHEST


def halo_exchange_for_conv(
    qkv: jax.Array,
    cp_axis: str | tuple[str, ...],
    halo_len: int,
) -> jax.Array:
  """Exchanges the trailing `halo_len` tokens from rank `i` to `i + 1` along `cp_axis`.

  Rank 0 receives zeros. Tagged with `checkpoint_name(..., "gdn_cp_conv_halo")`
  so `gdn_cp_state` under `remat_policy='custom'` can save or offload the halo
  instead of repeating the neighbor `ppermute` in the backward pass.
  """
  if halo_len <= 0:
    return qkv[:, :0, :]
  num_ranks = lax.axis_size(cp_axis)
  rank_idx = lax.axis_index(cp_axis)
  perm = [(i, i + 1) for i in range(num_ranks - 1)]
  halo = lax.ppermute(qkv[:, -halo_len:, :], axis_name=cp_axis, perm=perm)
  halo = jnp.where(rank_idx == 0, jnp.zeros_like(halo), halo)
  return checkpoint_name(halo, "gdn_cp_conv_halo")


def gather_cp_segment_metadata(
    segment_ids: jax.Array | None,
    cp_axis: str | tuple[str, ...],
    kernel_size: int,
    has_initial_state: bool = False,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
  """Gathers global segment_ids across cp_axis and slices local (s_enc, conv_halo_seg, init_seg).

  Segment IDs are canonicalized on the gathered global sequence, so every rank
  sees the same document numbering. With `has_initial_state`, the caller
  conv/recurrent states continue the first document (canonical ID 1).
  """
  if segment_ids is None:
    return None, None, None
  from maxtext.kernels.attention import gdn_packing  # pylint: disable=import-outside-toplevel

  batch_size, seq_len = segment_ids.shape
  halo_len = max(kernel_size - 1, 1)
  global_seg = lax.all_gather(segment_ids, axis_name=cp_axis, axis=1, tiled=True)
  global_seg = gdn_packing.canonicalize_segment_ids(global_seg)
  if has_initial_state:
    init_global = jnp.ones((batch_size,), dtype=jnp.float32)
    pad_value = 1.0
  else:
    init_global = None
    pad_value = 0.0
  global_s_enc = gdn_packing.encode_segment_ids(global_seg, init_seg=init_global)
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


def compose(left, right):
  """(A_r, B_r) . (A_l, B_l) = (A_r @ A_l, A_r @ B_l + B_r)."""
  A_l, B_l = left
  A_r, B_r = right
  return (
      jnp.matmul(A_r, A_l, precision=_PREC),
      jnp.matmul(A_r, B_l, precision=_PREC) + B_r,
  )


def compose_local(w, u, k, g, m_out=None, m_keep=None):
  """Fold this device's chunks into one affine map, in O(1) memory.

  lax.scan rather than associative_scan on purpose: associative_scan would
  materialize (A, B) and their running composition for every chunk, which is
  17 GB per device at a million tokens and defeats the point. The parallelism
  that matters here is across devices, not within one.
  """
  k_dim = k.shape[-1]
  eye = jnp.eye(k_dim, dtype=jnp.float32)

  # jax.checkpoint is required here. lax.scan keeps whatever
  # the body computes as a backward residual, so A_i and B_i get stacked over
  # every chunk even though the forward pass only ever needs one at a time. At
  # sequence 262,144 with ctx=4 that was 103 GB of f32[1024,4,16,128,128] in the
  # buffer dump. A_i and B_i are cheap to rebuild from w, u, k and g, which are
  # already live, so recompute them in the backward pass instead of storing them.
  @jax.checkpoint
  def body(carry, x):
    if m_out is not None and m_keep is not None:
      w_c, u_c, k_c, g_c, m_out_c, m_keep_c = x
      g_last = g_c[..., -1]
      decay = (jnp.exp(g_last) * m_keep_c[..., 0])[..., None, None]
      bwd_diff = jnp.where(m_out_c > 0.5, g_last[..., None] - g_c, -1e30)
      k_g = k_c.astype(jnp.float32) * (jnp.exp(bwd_diff) * m_out_c)[..., None]
    else:
      w_c, u_c, k_c, g_c = x
      g_last = g_c[..., -1]
      decay = jnp.exp(g_last)[..., None, None]
      k_g = k_c.astype(jnp.float32) * jnp.exp(g_last[..., None] - g_c)[..., None]
    k_g_T = k_g.swapaxes(-1, -2)
    A_i = decay * eye - jnp.matmul(k_g_T, w_c.astype(jnp.float32), precision=_PREC)
    B_i = jnp.matmul(k_g_T, u_c.astype(jnp.float32), precision=_PREC)
    return compose(carry, (A_i, B_i)), None

  lead = w.shape[1:-2]
  init = (
      jnp.broadcast_to(eye, lead + (k_dim, k_dim)).astype(jnp.float32),
      jnp.zeros(lead + (k_dim, u.shape[-1]), jnp.float32),
  )
  scan_inputs = (w, u, k, g, m_out, m_keep) if (m_out is not None and m_keep is not None) else (w, u, k, g)
  (A_loc, B_loc), _ = lax.scan(body, init, scan_inputs)
  return checkpoint_name(A_loc, "gdn_cp_m_local"), checkpoint_name(B_loc, "gdn_cp_m_local")


def incoming_state(A_loc, B_loc, h_init, cp_axis):
  """State entering this device, plus the final state after all devices.

  log2(D) exchanges of one small matrix pair are the only cross-device traffic
  in the scheme. Must be called inside a shard_map over `cp_axis`.
  """
  # A prefix scan over the device axis, not an all-gather. Gathering the D pairs
  # materializes [D, B, H, K, K] and [D, B, H, K, V] on every device, so the
  # composition costs O(D) per device per layer. At ctx=256 on Qwen3.5-27B that
  # is roughly 38 GB across the 48 GatedDeltaNet layers, and it is what makes
  # per-device memory grow with the context axis instead of shrinking.
  #
  # Hillis-Steele instead: log2(D) ppermute exchanges, one (A, B) pair live at a
  # time, O(1) in D, and log2(D) backward residuals rather than D.
  D = lax.axis_size(cp_axis)
  idx = lax.axis_index(cp_axis)

  # Inclusive prefix: each device ends holding the composition of ranks 0..idx.
  # compose(left, right) applies left first, so the pair arriving from the
  # earlier rank is the left operand.
  a_run, b_run = A_loc, B_loc
  step = 1
  while step < D:
    fwd = [(i, i + step) for i in range(D - step)]
    a_recv = lax.ppermute(a_run, cp_axis, fwd)
    b_recv = lax.ppermute(b_run, cp_axis, fwd)
    a_cmp, b_cmp = compose((a_recv, b_recv), (a_run, b_run))
    live = idx >= step
    a_run = jnp.where(live, a_cmp, a_run)
    b_run = jnp.where(live, b_cmp, b_run)
    step *= 2

  # A device starts from the exclusive prefix, which is the inclusive prefix of
  # the device before it. One more shift by a single rank.
  shift1 = [(i, i + 1) for i in range(D - 1)]
  a_ex = lax.ppermute(a_run, cp_axis, shift1)
  b_ex = lax.ppermute(b_run, cp_axis, shift1)
  carried = jnp.matmul(a_ex, h_init, precision=_PREC) + b_ex
  h_in = checkpoint_name(jnp.where(idx == 0, h_init, carried), "gdn_cp_s_in")

  # The final state is the last device's inclusive prefix, broadcast with a
  # masked psum rather than a gather so this stays O(1) in D too.
  last = idx == (D - 1)
  a_tot = lax.psum(jnp.where(last, a_run, jnp.zeros_like(a_run)), cp_axis)
  b_tot = lax.psum(jnp.where(last, b_run, jnp.zeros_like(b_run)), cp_axis)
  final_h = jnp.matmul(a_tot, h_init, precision=_PREC) + b_tot
  return h_in, final_h
