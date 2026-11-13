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

"""Replica weight moves for online hot-expert replication on the ragged all-to-all MoE path.

Each expert-parallel (EP) device owns L = E / EP "home" experts and R extra
replica slots. A replica plan (see moe_replica_planner.ReplicaPlan) says which
home expert each replica slot hosts. The ops here only read the plan's
attributes (rep_home, send_slots, send_peer_index, num_replicas).

Replica weights are moved with one ragged all-to-all per weight array; the
custom VJP sends replica gradients back and adds them into the home experts.
"""

import functools

import jax
import jax.numpy as jnp


def _move_params(plan, me, num_ep):
  """ragged_all_to_all params for moving replica rows (fwd)."""
  # As sender: to each peer p, 1 row from my send buffer at send_peer_index[me, p].
  spi = plan.send_peer_index[me]  # [EP]
  send_sizes = (spi >= 0).astype(jnp.int32)
  input_offsets = jnp.maximum(spi, 0).astype(jnp.int32)
  # Where my row lands at receiver p: the replica slot s with rep_home[p, s] == me.
  match = plan.rep_home == me  # [EP_r, R]
  out_slot = jnp.argmax(match, axis=1).astype(jnp.int32)
  output_offsets = jnp.where(send_sizes > 0, out_slot, 0).astype(jnp.int32)
  # As receiver: from each peer p, 1 row if p is the home of one of my replicas.
  recv = (
      jnp.zeros((num_ep,), jnp.int32)
      .at[jnp.where(plan.rep_home[me] >= 0, plan.rep_home[me], 0)]
      .max((plan.rep_home[me] >= 0).astype(jnp.int32))
  )
  return input_offsets, send_sizes, output_offsets, recv


def _bwd_offsets(plan, me, num_ep):
  """ragged_all_to_all input/output offsets for returning replica grads to home devices (bwd)."""
  r_slots = plan.num_replicas
  rep_home_me = plan.rep_home[me]
  bwd_in_off = (
      jnp.zeros((num_ep,), jnp.int32)
      .at[jnp.where(rep_home_me >= 0, rep_home_me, 0)]
      .max(jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0))
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  return bwd_in_off, bwd_out_off


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _move(axis_name, r_slots, max_send, w, idx):
  send_slots, in_off, send_sz, out_off, recv_sz = idx[:5]
  send_buf = w[send_slots]
  out = jnp.zeros((r_slots,) + w.shape[1:], w.dtype)
  return jax.lax.ragged_all_to_all(send_buf, out, in_off, send_sz, out_off, recv_sz, axis_name=axis_name)


def _move_fwd(axis_name, r_slots, max_send, w, idx):
  return _move(axis_name, r_slots, max_send, w, idx), (idx, w.shape)


def _move_bwd(axis_name, r_slots, max_send, res, g):
  idx, shape = res
  send_slots, _, send_sz, _, _, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  back = jnp.zeros((max_send,) + tuple(shape[1:]), g.dtype)
  back = jax.lax.ragged_all_to_all(g, back, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name)
  valid = (jnp.arange(max_send) < jnp.sum(send_sz)).astype(g.dtype)
  back = back * valid.reshape((max_send,) + (1,) * (len(shape) - 1))
  dw = jnp.zeros(shape, g.dtype).at[send_slots].add(back)
  return dw, tuple(None for _ in idx)


_move.defvjp(_move_fwd, _move_bwd)


def move_replicas(w_local, plan, axis_name, num_ep):
  """Returns replica rows [R, ...] gathered from their home devices. Differentiable in w_local."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (plan.send_slots[me], in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _move(axis_name, r_slots, max_send, w_local, idx)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _extend(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  send_slots, in_off, send_sz, out_off, recv_sz = idx[:5]
  wc = w.astype(dtype)
  out = jnp.zeros((r_slots,) + w.shape[1:], dtype)
  rep = jax.lax.ragged_all_to_all(wc[send_slots], out, in_off, send_sz, out_off, recv_sz, axis_name=axis_name)
  return jnp.concatenate([wc, rep], axis=0)


def _extend_fwd(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  return _extend(axis_name, r_slots, max_send, dtype, wdtype, w, idx), (idx, w.shape)


def _extend_bwd(axis_name, r_slots, max_send, dtype, wdtype, res, g):
  """Backward pass for `_extend`: returns replica grads to home slots."""
  idx, shape = res
  send_slots, _, send_sz, _, _, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  local = shape[0]
  g_home = g[:local].astype(wdtype)
  back = jnp.zeros((max_send,) + tuple(shape[1:]), g.dtype)
  back = jax.lax.ragged_all_to_all(
      g[local:], back, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name
  )
  valid = (jnp.arange(max_send) < jnp.sum(send_sz)).astype(wdtype)
  back = back.astype(wdtype) * valid.reshape((max_send,) + (1,) * (len(shape) - 1))
  return g_home.at[send_slots].add(back), tuple(None for _ in idx)


_extend.defvjp(_extend_fwd, _extend_bwd)


def extend_with_replicas(w_local, plan, axis_name, num_ep, dtype):
  """concat([w_local, replicas]) in `dtype`; one fused pass, replica grads summed into home experts."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (plan.send_slots[me], in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _extend(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _with_reps(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  send_slots, in_off, send_sz, out_off, recv_sz = idx[:5]
  out = jnp.zeros((r_slots,) + w.shape[1:], dtype)
  rep = jax.lax.ragged_all_to_all(
      w[send_slots].astype(dtype), out, in_off, send_sz, out_off, recv_sz, axis_name=axis_name
  )
  return w, rep


def _with_reps_fwd(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  return _with_reps(axis_name, r_slots, max_send, dtype, wdtype, w, idx), (idx,)


def _with_reps_bwd(axis_name, r_slots, max_send, dtype, wdtype, res, gs):
  (idx,) = res
  g_home, g_rep = gs
  send_slots, _, send_sz, _, _, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  back = jnp.zeros((max_send,) + tuple(g_rep.shape[1:]), g_rep.dtype)
  back = jax.lax.ragged_all_to_all(g_rep, back, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name)
  valid = (jnp.arange(max_send) < jnp.sum(send_sz)).astype(wdtype)
  back = back.astype(wdtype) * valid.reshape((max_send,) + (1,) * (g_rep.ndim - 1))
  # Unused entries in `send_slots` are padded with 0, so `send_slots` can contain duplicate 0s.
  return g_home.at[send_slots].add(back.astype(g_home.dtype)), tuple(None for _ in idx)


_with_reps.defvjp(_with_reps_fwd, _with_reps_bwd)


def home_and_replicas(w_local, plan, axis_name, num_ep, dtype):
  """Returns (w_local, replicas[R] in dtype). Replica grads are scatter-added into the home grad."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (plan.send_slots[me], in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _with_reps(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)


def _direct_move_params(plan, me, num_ep):
  """Like _move_params, but input offsets index the weight array itself (no gathered send buffer)."""
  _, send_sz, out_off, recv_sz = _move_params(plan, me, num_ep)
  spi = plan.send_peer_index[me]
  in_off = jnp.where(spi >= 0, plan.send_slots[me][jnp.maximum(spi, 0)], 0).astype(jnp.int32)
  return in_off, send_sz, out_off, recv_sz


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _extend_direct(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  in_off, send_sz, out_off, recv_sz = idx[:4]
  local = w.shape[0]
  wc = w.astype(dtype)
  buf = jnp.pad(wc, ((0, r_slots),) + ((0, 0),) * (w.ndim - 1))
  # Replica rows are sent straight from the weight array and land in place in rows [local, local + R).
  return jax.lax.ragged_all_to_all(wc, buf, in_off, send_sz, out_off + local, recv_sz, axis_name=axis_name)


def _extend_direct_fwd(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  return _extend_direct(axis_name, r_slots, max_send, dtype, wdtype, w, idx), (idx, w.shape)


def _extend_direct_bwd(axis_name, r_slots, max_send, dtype, wdtype, res, g):
  """Backward pass for `_extend_direct`: returns replica grads to home slots."""
  idx, shape = res
  _, send_sz, _, _, send_slots, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  local = shape[0]
  back = jnp.zeros((max_send,) + tuple(shape[1:]), g.dtype)
  # Replica grads are sent straight from rows [local, local + R) of the extended grad.
  back = jax.lax.ragged_all_to_all(
      g, back, bwd_in_off + local, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name
  )
  valid = (jnp.arange(max_send) < jnp.sum(send_sz)).astype(wdtype)
  back = back.astype(wdtype) * valid.reshape((max_send,) + (1,) * (len(shape) - 1))
  # g_home[send_slots[c]] += back[c], written as one elementwise pass (fuses with the slice of g).
  rows = jnp.arange(local, dtype=jnp.int32).reshape((local,) + (1,) * (len(shape) - 1))
  g_home = g[:local].astype(wdtype)
  for c in range(max_send):
    g_home = g_home + jnp.where(rows == send_slots[c], back[c][None], jnp.zeros((), wdtype))
  return g_home, tuple(None for _ in idx)


_extend_direct.defvjp(_extend_direct_fwd, _extend_direct_bwd)


def extend_with_replicas_direct(w_local, plan, axis_name, num_ep, dtype):
  """Same result and gradient as extend_with_replicas, without the gathered send buffer and the concat."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _direct_move_params(plan, me, num_ep)
  bwd_in_off, bwd_out_off = _bwd_offsets(plan, me, num_ep)
  idx = (in_off, send_sz, out_off, recv_sz, plan.send_slots[me], bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _extend_direct(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)
