"""Context-parallel evaluation of the GatedDeltaNet inter-chunk recurrence.

The recurrence h_new = A @ h + B is affine in the state, so it composes
associatively and can be split across a sharded sequence. See
apply_gdn_context_parallel.py for the derivation and the measurements.
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
  materialise (A, B) and their running composition for every chunk, which is
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

  Gathering D pairs of small matrices is the only cross-device traffic in the
  scheme. Must be called inside a shard_map over `cp_axis`.
  """
  A_all = lax.all_gather(A_loc, cp_axis, axis=0, tiled=False)
  B_all = lax.all_gather(B_loc, cp_axis, axis=0, tiled=False)
  A_cum, B_cum = lax.associative_scan(compose, (A_all, B_all), axis=0)

  idx = lax.axis_index(cp_axis)
  prev = jnp.maximum(idx - 1, 0)
  carried = jnp.matmul(A_cum[prev], h_init, precision=_PREC) + B_cum[prev]
  h_in = jnp.where(idx == 0, h_init, carried)
  final_h = jnp.matmul(A_cum[-1], h_init, precision=_PREC) + B_cum[-1]
  return h_in, final_h
