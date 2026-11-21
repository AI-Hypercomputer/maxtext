# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replica weight move for online hot-expert replication on the ragged all-to-all MoE path.

Each expert-parallel (EP) device owns L = E / EP "home" experts and R extra
replica slots. Given a `ReplicaPlan` produced by
`moe_replica_planner.compute_plan_iter`, `extend_with_replicas` pads the local
expert weights `[L, ...]` to `[L + R, ...]` and populates rows `[L, L + R)` in
place via a single `ragged_all_to_all` straight from the home weight array.
Its custom VJP returns replica gradients `[L, L + R)` back to their home
devices and accumulates them into the home expert gradients.

Plan invariants required from `moe_replica_planner` (verified in
`moe_replica_planner_test.test_plan_satisfies_weight_move_invariants`):
  1. At most one replica row per `(home, receiver)` device pair.
  2. Distinct valid home devices (`rep_home[r] >= 0`) across each receiver `r`'s `R` slots.
  3. Valid send entries for home `h` compacted at the front of `send_slots[h, :k]`
     (`k = sum(send_peer_index[h] >= 0) <= max_send`), ordered by ascending receiver ID,
     with `0 <= send_slots[h, :k] < L`.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import qwix.pallas as qpl

MAX_REPLICA_SEND = 8


def _direct_move_params(plan, me, num_ep):
  """`ragged_all_to_all` params sending replica rows directly from `w_local`."""
  spi = plan.send_peer_index[me]  # [EP]
  send_sizes = (spi >= 0).astype(jnp.int32)
  in_off = jnp.where(spi >= 0, plan.send_slots[me][jnp.maximum(spi, 0)], 0).astype(jnp.int32)
  match = plan.rep_home == me  # [EP_r, R]
  out_slot = jnp.argmax(match, axis=1).astype(jnp.int32)
  output_offsets = jnp.where(send_sizes > 0, out_slot, 0).astype(jnp.int32)
  recv = (
      jnp.zeros((num_ep,), jnp.int32)
      .at[jnp.where(plan.rep_home[me] >= 0, plan.rep_home[me], 0)]
      .max((plan.rep_home[me] >= 0).astype(jnp.int32))
  )
  return in_off, send_sizes, output_offsets, recv


def _bwd_offsets(plan, me, num_ep):
  """`ragged_all_to_all` input/output offsets for returning replica grads to home devices."""
  r_slots = plan.num_replicas
  rep_home_me = plan.rep_home[me]
  bwd_in_off = (
      jnp.zeros((num_ep,), jnp.int32)
      .at[jnp.where(rep_home_me >= 0, rep_home_me, 0)]
      .max(jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0))
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  return bwd_in_off, bwd_out_off


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _extend_direct(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  in_off, send_sz, out_off, recv_sz = idx[:4]
  local = w.shape[0]
  wc = w.astype(dtype)
  buf = jnp.pad(wc, ((0, r_slots),) + ((0, 0),) * (w.ndim - 1))
  return jax.lax.ragged_all_to_all(wc, buf, in_off, send_sz, out_off + local, recv_sz, axis_name=axis_name)


def _extend_direct_fwd(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  return _extend_direct(axis_name, r_slots, max_send, dtype, wdtype, w, idx), (idx, w.shape)


def _extend_direct_bwd(axis_name, r_slots, max_send, dtype, wdtype, res, g):
  """Backward pass for `_extend_direct`: returns replica grads to home slots."""
  idx, shape = res
  _, send_sz, _, _, send_slots, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  local = shape[0]
  back = jnp.zeros((max_send,) + tuple(shape[1:]), g.dtype)
  back = jax.lax.ragged_all_to_all(
      g, back, bwd_in_off + local, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name
  )
  valid = (jnp.arange(max_send) < jnp.sum(send_sz)).astype(wdtype)
  back = back.astype(wdtype) * valid.reshape((max_send,) + (1,) * (len(shape) - 1))
  rows = jnp.arange(local, dtype=jnp.int32).reshape((local,) + (1,) * (len(shape) - 1))
  g_home = g[:local].astype(wdtype)
  # Costs O(max_send) full-size [L, ...] elementwise passes, fused into one kernel by XLA
  # (capped at MAX_REPLICA_SEND=8); dynamic `.at[send_slots].add(back)` prevents XLA fusion into
  # `g[:local]` and materializes an extra HBM copy.
  for c in range(max_send):
    g_home = g_home + jnp.where(rows == send_slots[c], back[c][None], jnp.zeros((), wdtype))
  return g_home, tuple(None for _ in idx)


_extend_direct.defvjp(_extend_direct_fwd, _extend_direct_bwd)


def _prepare_move(w_local, plan, axis_name, num_ep):
  """Validates `plan` against `w_local` and returns `(r_slots, max_send, idx)` or None when replication is off."""
  r_slots = int(plan.num_replicas)
  if r_slots < 0 or num_ep < 1:
    raise ValueError(f"Expected num_replicas >= 0 and num_ep >= 1, got num_replicas={r_slots}, num_ep={num_ep}.")
  if r_slots == 0 or num_ep == 1:
    return None
  if plan.send_slots.ndim != 2:
    raise ValueError(f"Expected 2D send_slots [EP, max_send], got shape {plan.send_slots.shape}.")
  max_send = int(plan.send_slots.shape[1])
  if max_send < 1 or max_send > MAX_REPLICA_SEND:
    raise ValueError(f"Expected 1 <= max_send <= {MAX_REPLICA_SEND}, got max_send={max_send}.")
  if not isinstance(plan.send_slots, jax.core.Tracer):
    slots_np = np.asarray(plan.send_slots)
    if np.any((slots_np < 0) | (slots_np >= w_local.shape[0])):
      raise ValueError(f"send_slots entries must be in [0, {w_local.shape[0]}), got {slots_np}.")
  me = jax.lax.axis_index(axis_name)
  in_off, send_sz, out_off, recv_sz = _direct_move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (in_off, send_sz, out_off, recv_sz, plan.send_slots[me], bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return r_slots, max_send, idx


def extend_with_replicas(w_local, plan, axis_name, num_ep, dtype):
  """Extends `w_local` `[L, ...]` to `[L + R, ...]` in `dtype`, summing replica grads back into home slots.

  Relies on the `moe_replica_planner.ReplicaPlan` invariants: at most one replica row per
  `(home, receiver)` pair, distinct valid home IDs across each receiver's `R` slots, and valid
  send entries compacted at the front of `send_slots[:, :max_send]` (`1 <= max_send <= 8`).
  Returns `w_local.astype(dtype)` directly when `plan.num_replicas == 0` or `num_ep == 1`.
  """
  move = _prepare_move(w_local, plan, axis_name, num_ep)
  if move is None:
    return w_local.astype(dtype)
  r_slots, max_send, idx = move
  return _extend_direct(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)


class _QuantMoveSpec(NamedTuple):
  """Static specification for home-side quantization and replica movement."""

  axis_name: str | tuple[str, ...]
  r_slots: int
  max_send: int
  dtype: jnp.dtype
  wdtype: jnp.dtype
  qtype: jnp.dtype
  calibration_method: str
  channelwise_axes: tuple[int, ...]


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _extend_quantized(spec, w, idx):
  """Quantizes `w` on its home shard and moves FP8 codes (and dynamic scales) to replica slots."""
  q = qpl.quantize(
      w.astype(spec.dtype),
      spec.qtype,
      channelwise_axes=list(spec.channelwise_axes),
      calibration_method=spec.calibration_method,
  )
  if q.zero_point is not None:
    raise NotImplementedError("extend_quantized_with_replicas does not support asymmetric zero_point quantization.")
  if idx is None:
    return q
  codes = _extend_direct(spec.axis_name, spec.r_slots, spec.max_send, q.qvalue.dtype, q.qvalue.dtype, q.qvalue, idx)
  # Static fixed-range calibration yields a constant (1, 1, 1) scale identical on all shards, so keep it
  # unmoved (matching R=0); dynamic calibration depends on the home shard's weights, so move one scale row
  # per replica alongside the FP8 codes.
  if spec.calibration_method.lower().startswith("fixed") and q.scale.size == 1:
    scale = q.scale
  else:
    scale = jnp.broadcast_to(q.scale, (w.shape[0], *q.scale.shape[1:]))
    scale = _extend_direct(spec.axis_name, spec.r_slots, spec.max_send, scale.dtype, scale.dtype, scale, idx)
  return qpl.QArray(qvalue=codes, scale=scale, zero_point=None, qtype=q.qtype)


def _extend_quantized_fwd(spec, w, idx):
  return _extend_quantized(spec, w, idx), (idx, w.shape)


def _extend_quantized_bwd(spec, res, g):
  """Straight-through: gmm returns d(dequantized weight) as g.qvalue; route replica grads home."""
  idx, _ = res
  if idx is None:
    return g.qvalue.astype(spec.wdtype), None
  return _extend_direct_bwd(spec.axis_name, spec.r_slots, spec.max_send, spec.dtype, spec.wdtype, res, g.qvalue)


_extend_quantized.defvjp(_extend_quantized_fwd, _extend_quantized_bwd)


def extend_quantized_with_replicas(
    w_local, plan, axis_name, num_ep, dtype, qtype, calibration_method="absmax", channelwise_axes=(2,)
):
  """Quantized variant of `extend_with_replicas` for Qwix `gmm` rules (e.g. `quantization=fp8_full`).

  Quantizes `w_local` `[L, K, N]` on its home device using the same rule as an unreplicated `gmm`
  (scale reduced over the `L` home experts and `K`, so home rows keep exact `R=0` codes and scales
  under both dynamic `absmax` and static `fixed` calibration). Moves the quantized codes (and, for
  non-static calibration, per-row home scales) via `ragged_all_to_all`, returning a `qpl.QArray` with
  codes `[L + R, K, N]` and scale `[L + R, 1, N]` / `[L + R, 1, 1]` (or `[1, 1, 1]` for static fixed
  calibration). The weight gradient is straight-through into `w_local` with replica grads summed home.
  """
  if w_local.ndim != 3:
    raise ValueError(f"Expected a 3D [L, K, N] expert weight, got shape {w_local.shape}.")
  channelwise_axes = tuple(channelwise_axes)
  if not set(channelwise_axes).issubset({2}):
    raise ValueError(f"Expected channelwise_axes subset of (2,), got {channelwise_axes}.")
  move = _prepare_move(w_local, plan, axis_name, num_ep)
  r_slots, max_send, idx = (0, 0, None) if move is None else move
  spec = _QuantMoveSpec(
      axis_name=axis_name,
      r_slots=r_slots,
      max_send=max_send,
      dtype=jnp.dtype(dtype),
      wdtype=jnp.dtype(w_local.dtype),
      qtype=jnp.dtype(qtype),
      calibration_method=calibration_method,
      channelwise_axes=channelwise_axes,
  )
  return _extend_quantized(spec, w_local, idx)
