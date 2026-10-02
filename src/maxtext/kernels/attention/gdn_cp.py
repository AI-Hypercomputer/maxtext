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
import jax.numpy as jnp
from jax import lax

_PREC = jax.lax.Precision.HIGHEST


def compose(left, right):
  """(A_r, B_r) . (A_l, B_l) = (A_r @ A_l, A_r @ B_l + B_r)."""
  A_l, B_l = left
  A_r, B_r = right
  return (
      jnp.matmul(A_r, A_l, precision=_PREC),
      jnp.matmul(A_r, B_l, precision=_PREC) + B_r,
  )


def compose_local(w, u, k, g):
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
  (A_loc, B_loc), _ = lax.scan(body, init, (w, u, k, g))
  return A_loc, B_loc


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
  h_in = jnp.where(idx == 0, h_init, carried)

  # The final state is the last device's inclusive prefix, broadcast with a
  # masked psum rather than a gather so this stays O(1) in D too.
  last = idx == (D - 1)
  a_tot = lax.psum(jnp.where(last, a_run, jnp.zeros_like(a_run)), cp_axis)
  b_tot = lax.psum(jnp.where(last, b_run, jnp.zeros_like(b_run)), cp_axis)
  final_h = jnp.matmul(a_tot, h_init, precision=_PREC) + b_tot
  return h_in, final_h
