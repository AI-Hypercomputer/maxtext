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
replica slots. Given a `ReplicaPlan` (`rep_home`, `send_slots`,
`send_peer_index`, `num_replicas`), `extend_with_replicas` pads the local
expert weights `[L, ...]` to `[L + R, ...]` and populates rows `[L, L + R)` in
place via a single `ragged_all_to_all` straight from the home weight array.
Its custom VJP returns replica gradients `[L, L + R)` back to their home
devices and accumulates them into the home expert gradients.
"""

import functools

import jax
import jax.numpy as jnp


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
  for c in range(max_send):
    g_home = g_home + jnp.where(rows == send_slots[c], back[c][None], jnp.zeros((), wdtype))
  return g_home, tuple(None for _ in idx)


_extend_direct.defvjp(_extend_direct_fwd, _extend_direct_bwd)


def extend_with_replicas(w_local, plan, axis_name, num_ep, dtype):
  """Extends `w_local` `[L, ...]` to `[L + R, ...]` in `dtype`, summing replica grads back into home slots."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _direct_move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (in_off, send_sz, out_off, recv_sz, plan.send_slots[me], bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _extend_direct(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)
