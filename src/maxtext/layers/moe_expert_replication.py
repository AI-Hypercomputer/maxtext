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

"""Online hot-expert replication for the ragged all-to-all MoE path.

Each expert-parallel (EP) device owns L = E / EP "home" experts and R extra
replica slots, so the physical slot space has EP * (L + R) slots. Each step,
after routing, a plan is computed identically on every device from the
all-gathered per-source expert counts:

  * the hottest experts get one replica each, on the currently lightest device,
    with at most one replica per (home, receiver) pair and at most C replicas
    sent from any home device;
  * whole source shards' tokens for a replicated expert are rerouted to the
    replica (greedy, preferring sources on the receiver's board), so tokens are
    never split inside a source shard and no extra sort is needed.

Tokens are remapped logical expert -> physical slot before the dispatch sort.
Replica weights are moved with one ragged all-to-all per weight array; the
custom VJP sends replica gradients back and adds them into the home experts.

Every (token, k) maps to exactly one physical slot, so it is computed exactly
once with identical weights.
"""

import functools

import jax
import jax.numpy as jnp


class ReplicaPlan:
  """Plain container (not a pytree) for one layer's replication plan."""

  def __init__(self, route_table, rep_home, rep_slot, send_slots, send_peer_index, num_slots, num_replicas):
    self.route_table = route_table  # [EP, E] int32: (source shard, logical expert) -> physical slot
    self.rep_home = rep_home  # [EP, R] int32: home device of each replica slot, -1 if empty
    self.rep_slot = rep_slot  # [EP, R] int32: home-local slot of each replica, 0 if empty
    self.send_slots = send_slots  # [EP, C] int32: local slots each device sends (ordered by receiver)
    self.send_peer_index = send_peer_index  # [EP, EP] int32: index into sender's send buffer for receiver, -1 if none
    self.num_slots = num_slots  # L + R (python int)
    self.num_replicas = num_replicas  # R (python int)


def identity_route_table(num_ep, num_experts, num_replicas):
  local = num_experts // num_ep
  e = jnp.arange(num_experts, dtype=jnp.int32)
  phys = (e // local) * (local + num_replicas) + e % local
  return jnp.broadcast_to(phys[None, :], (num_ep, num_experts))


def compute_plan(counts_src, num_ep, num_replicas, max_send, board_size, enabled=True):
  """Greedy online replication plan, computed identically on every device.

  Args:
    counts_src: [EP, E] tokens each source shard routes to each logical expert.
    num_ep: EP size.
    num_replicas: R replica slots per device.
    max_send: C, max replicas any home device sends.
    board_size: devices per board (prefer same-board sources for a replica).
    enabled: if False, returns the identity plan (no replicas used).
  """
  ep, num_experts = counts_src.shape
  local = num_experts // num_ep
  r_slots = num_replicas
  counts_src = counts_src.astype(jnp.float32)
  c = jnp.sum(counts_src, axis=0)  # [E]
  dev_load = jnp.sum(c.reshape(num_ep, local), axis=1)  # [EP]
  num_cand = num_ep * r_slots
  cand_load, cand = jax.lax.top_k(c, num_cand)
  cand = cand.astype(jnp.int32)
  src_ids = jnp.arange(num_ep, dtype=jnp.int32)

  def body(i, state):
    dev_load, free, send_cnt, pair_used, rep_e, rep_mask = state
    e = cand[i]
    h = e // local
    blocked = (free <= 0) | (src_ids == h) | pair_used[h]
    r = jnp.argmin(jnp.where(blocked, jnp.inf, dev_load)).astype(jnp.int32)
    ok = (~blocked[r]) & (send_cnt[h] < max_send) & enabled
    target = (dev_load[h] - dev_load[r]) / 2.0
    # Source preference for receiver r: same board first, then by ring distance; home last.
    same_board = (src_ids // board_size) == (r // board_size)
    key = jnp.where(src_ids == h, 3 * num_ep, jnp.where(same_board, 0, num_ep) + (src_ids - r) % num_ep)
    order = jnp.argsort(key)
    contrib = counts_src[order, e]
    cum = jnp.cumsum(contrib)
    take_sorted = (cum <= target) & (jnp.arange(num_ep) < num_ep - 1)
    moved = jnp.sum(jnp.where(take_sorted, contrib, 0.0))
    ok = ok & (moved > 0)
    take = jnp.zeros((num_ep,), bool).at[order].set(take_sorted) & ok
    slot = (r_slots - free[r]).astype(jnp.int32)
    dev_load = dev_load.at[h].add(jnp.where(ok, -moved, 0.0)).at[r].add(jnp.where(ok, moved, 0.0))
    free = free.at[r].add(jnp.where(ok, -1, 0))
    send_cnt = send_cnt.at[h].add(jnp.where(ok, 1, 0))
    pair_used = pair_used.at[h, r].set(pair_used[h, r] | ok)
    rep_e = rep_e.at[r, jnp.minimum(slot, r_slots - 1)].set(jnp.where(ok, e, rep_e[r, jnp.minimum(slot, r_slots - 1)]))
    rep_mask = rep_mask.at[r, jnp.minimum(slot, r_slots - 1)].set(
        jnp.where(ok, take, rep_mask[r, jnp.minimum(slot, r_slots - 1)])
    )
    return dev_load, free, send_cnt, pair_used, rep_e, rep_mask

  init = (
      dev_load,
      jnp.full((num_ep,), r_slots, jnp.int32),
      jnp.zeros((num_ep,), jnp.int32),
      jnp.zeros((num_ep, num_ep), bool),
      jnp.full((num_ep, r_slots), -1, jnp.int32),
      jnp.zeros((num_ep, r_slots, num_ep), bool),
  )
  _, _, _, _, rep_e, rep_mask = jax.lax.fori_loop(0, num_cand, body, init)

  route = identity_route_table(num_ep, num_experts, r_slots)  # [EP(src), E]
  rr, ss = jnp.meshgrid(jnp.arange(num_ep), jnp.arange(r_slots), indexing="ij")
  phys_rep = (rr * (local + r_slots) + local + ss).astype(jnp.int32)  # [EP, R]
  valid = rep_e >= 0
  e_safe = jnp.where(valid, rep_e, 0)
  # For each (receiver, slot) and source: if rerouted, route_table[src, e] = phys_rep.
  for_src = rep_mask & valid[:, :, None]  # [EP_r, R, EP_src]
  src_idx = jnp.broadcast_to(jnp.arange(num_ep)[None, None, :], for_src.shape)
  e_idx = jnp.broadcast_to(e_safe[:, :, None], for_src.shape)
  vals = jnp.broadcast_to(phys_rep[:, :, None], for_src.shape)
  cur = route[src_idx, e_idx]
  route = route.at[src_idx, e_idx].set(jnp.where(for_src, vals, cur))

  rep_home = jnp.where(valid, e_safe // local, -1).astype(jnp.int32)
  rep_slot = jnp.where(valid, e_safe % local, 0).astype(jnp.int32)
  # Send side: for each home h, the replicas it serves, ordered by receiver.
  # send_peer_index[h, r] = position of r's row in h's send buffer.
  sends = jnp.zeros((num_ep, num_ep), bool)  # [h, r]
  slot_of = jnp.zeros((num_ep, num_ep), jnp.int32)  # [h, r] home-local slot sent to r
  hh = jnp.where(valid, rep_home, 0)
  sends = sends.at[hh, rr].max(valid)
  slot_of = slot_of.at[hh, rr].max(jnp.where(valid, rep_slot, 0))
  pos = jnp.cumsum(sends.astype(jnp.int32), axis=1) - 1
  send_peer_index = jnp.where(sends, pos, -1).astype(jnp.int32)
  send_slots = jnp.zeros((num_ep, max_send), jnp.int32)
  hidx = jnp.broadcast_to(jnp.arange(num_ep)[:, None], (num_ep, num_ep))
  send_slots = send_slots.at[hidx, jnp.clip(pos, 0, max_send - 1)].max(jnp.where(sends, slot_of, 0))
  return ReplicaPlan(route, rep_home, rep_slot, send_slots, send_peer_index, local + r_slots, r_slots)


def _move_params(plan, me, num_ep):
  """ragged_all_to_all params for moving replica rows (fwd)."""
  r_slots = plan.num_replicas
  # As sender: to each peer p, 1 row from my send buffer at send_peer_index[me, p].
  spi = plan.send_peer_index[me]  # [EP]
  send_sizes = (spi >= 0).astype(jnp.int32)
  input_offsets = jnp.maximum(spi, 0).astype(jnp.int32)
  # Where my row lands at receiver p: the replica slot s with rep_home[p, s] == me.
  match = plan.rep_home == me  # [EP_r, R]
  out_slot = jnp.argmax(match, axis=1).astype(jnp.int32)
  output_offsets = jnp.where(send_sizes > 0, out_slot, 0).astype(jnp.int32)
  # As receiver: from each peer p, 1 row if p is the home of one of my replicas.
  recv = jnp.zeros((num_ep,), jnp.int32).at[jnp.where(plan.rep_home[me] >= 0, plan.rep_home[me], 0)].max(
      (plan.rep_home[me] >= 0).astype(jnp.int32)
  )
  del r_slots
  return input_offsets, send_sizes, output_offsets, recv


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
  my_send_slots = plan.send_slots[me]
  rep_home_me = plan.rep_home[me]
  bwd_in_off = jnp.zeros((num_ep,), jnp.int32).at[jnp.where(rep_home_me >= 0, rep_home_me, 0)].max(
      jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0)
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  idx = (my_send_slots, in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
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
  idx, shape = res
  send_slots, _, send_sz, _, _, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz = idx
  local = shape[0]
  g_home = g[:local].astype(wdtype)
  back = jnp.zeros((max_send,) + tuple(shape[1:]), g.dtype)
  back = jax.lax.ragged_all_to_all(g[local:], back, bwd_in_off, bwd_send_sz, bwd_out_off, bwd_recv_sz, axis_name=axis_name)
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
  rep_home_me = plan.rep_home[me]
  bwd_in_off = jnp.zeros((num_ep,), jnp.int32).at[jnp.where(rep_home_me >= 0, rep_home_me, 0)].max(
      jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0)
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  idx = (plan.send_slots[me], in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _extend(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _with_reps(axis_name, r_slots, max_send, dtype, wdtype, w, idx):
  send_slots, in_off, send_sz, out_off, recv_sz = idx[:5]
  out = jnp.zeros((r_slots,) + w.shape[1:], dtype)
  rep = jax.lax.ragged_all_to_all(w[send_slots].astype(dtype), out, in_off, send_sz, out_off, recv_sz, axis_name=axis_name)
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
  return g_home.at[send_slots].add(back.astype(g_home.dtype), unique_indices=True), tuple(None for _ in idx)


_with_reps.defvjp(_with_reps_fwd, _with_reps_bwd)


def home_and_replicas(w_local, plan, axis_name, num_ep, dtype):
  """Returns (w_local, replicas[R] in dtype). Replica grads are scatter-added into the home grad in place."""
  me = jax.lax.axis_index(axis_name)
  max_send = plan.send_slots.shape[1]
  r_slots = plan.num_replicas
  in_off, send_sz, out_off, recv_sz = _move_params(plan, me, num_ep)
  rep_home_me = plan.rep_home[me]
  bwd_in_off = jnp.zeros((num_ep,), jnp.int32).at[jnp.where(rep_home_me >= 0, rep_home_me, 0)].max(
      jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0)
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  idx = (plan.send_slots[me], in_off, send_sz, out_off, recv_sz, bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _with_reps(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)


def compute_placement(c, num_ep, num_replicas, max_send):
  """Lagged placement: which logical expert each replica slot hosts, from (previous-step) global counts.

  Independent of the current step's routing, so the weight move can be issued at layer start.
  Returns rep_e [EP, R] (-1 = empty) and order [EP*R, 2] (receiver, slot) in assignment order.
  """
  c = c.astype(jnp.float32)
  num_experts = c.shape[0]
  local = num_experts // num_ep
  r_slots = num_replicas
  dev_load = jnp.sum(c.reshape(num_ep, local), axis=1)
  num_cand = num_ep * r_slots
  _, cand = jax.lax.top_k(c, num_cand)
  cand = cand.astype(jnp.int32)
  src_ids = jnp.arange(num_ep, dtype=jnp.int32)

  def body(i, state):
    dev_load, free, send_cnt, pair_used, rep_e, order, n = state
    e = cand[i]
    h = e // local
    blocked = (free <= 0) | (src_ids == h) | pair_used[h]
    r = jnp.argmin(jnp.where(blocked, jnp.inf, dev_load)).astype(jnp.int32)
    est = jnp.minimum((dev_load[h] - dev_load[r]) / 2.0, c[e] * (num_ep - 1) / num_ep)
    ok = (~blocked[r]) & (send_cnt[h] < max_send) & (est > 0)
    slot = jnp.minimum(r_slots - free[r], r_slots - 1).astype(jnp.int32)
    dev_load = dev_load.at[h].add(jnp.where(ok, -est, 0.0)).at[r].add(jnp.where(ok, est, 0.0))
    free = free.at[r].add(jnp.where(ok, -1, 0))
    send_cnt = send_cnt.at[h].add(jnp.where(ok, 1, 0))
    pair_used = pair_used.at[h, r].set(pair_used[h, r] | ok)
    rep_e = rep_e.at[r, slot].set(jnp.where(ok, e, rep_e[r, slot]))
    order = order.at[n].set(jnp.where(ok, jnp.stack([r, slot]), order[n]))
    n = n + ok.astype(jnp.int32)
    return dev_load, free, send_cnt, pair_used, rep_e, order, n

  init = (
      dev_load,
      jnp.full((num_ep,), r_slots, jnp.int32),
      jnp.zeros((num_ep,), jnp.int32),
      jnp.zeros((num_ep, num_ep), bool),
      jnp.full((num_ep, r_slots), -1, jnp.int32),
      jnp.zeros((num_cand, 2), jnp.int32),
      jnp.int32(0),
  )
  _, _, _, _, rep_e, order, n = jax.lax.fori_loop(0, num_cand, body, init)
  return rep_e, order, n


def plan_from_placement(counts_src, rep_e, order, n, num_ep, num_replicas, max_send, board_size):
  """Routes the current step's tokens given a fixed placement (rep_e). Weight-move tables depend only on rep_e."""
  ep, num_experts = counts_src.shape
  local = num_experts // num_ep
  r_slots = num_replicas
  counts_src = counts_src.astype(jnp.float32)
  c = jnp.sum(counts_src, axis=0)
  dev_load = jnp.sum(c.reshape(num_ep, local), axis=1)
  src_ids = jnp.arange(num_ep, dtype=jnp.int32)
  num_cand = order.shape[0]

  def body(i, state):
    dev_load, rep_mask = state
    r, slot = order[i, 0], order[i, 1]
    active = i < n
    e = jnp.maximum(rep_e[r, slot], 0)
    h = e // local
    target = (dev_load[h] - dev_load[r]) / 2.0
    same_board = (src_ids // board_size) == (r // board_size)
    key = jnp.where(src_ids == h, 3 * num_ep, jnp.where(same_board, 0, num_ep) + (src_ids - r) % num_ep)
    srt = jnp.argsort(key)
    contrib = counts_src[srt, e]
    cum = jnp.cumsum(contrib)
    take_sorted = (cum <= target) & (jnp.arange(num_ep) < num_ep - 1) & active
    moved = jnp.sum(jnp.where(take_sorted, contrib, 0.0))
    take = jnp.zeros((num_ep,), bool).at[srt].set(take_sorted)
    dev_load = dev_load.at[h].add(-moved).at[r].add(moved)
    rep_mask = rep_mask.at[r, slot].set(jnp.where(active, take, rep_mask[r, slot]))
    return dev_load, rep_mask

  _, rep_mask = jax.lax.fori_loop(0, num_cand, body, (dev_load, jnp.zeros((num_ep, r_slots, num_ep), bool)))
  return _finalize_plan(rep_e, rep_mask, num_ep, num_experts, r_slots, max_send)


def _finalize_plan(rep_e, rep_mask, num_ep, num_experts, r_slots, max_send):
  local = num_experts // num_ep
  route = identity_route_table(num_ep, num_experts, r_slots)
  rr, ss = jnp.meshgrid(jnp.arange(num_ep), jnp.arange(r_slots), indexing="ij")
  phys_rep = (rr * (local + r_slots) + local + ss).astype(jnp.int32)
  valid = rep_e >= 0
  e_safe = jnp.where(valid, rep_e, 0)
  for_src = rep_mask & valid[:, :, None]
  src_idx = jnp.broadcast_to(jnp.arange(num_ep)[None, None, :], for_src.shape)
  e_idx = jnp.broadcast_to(e_safe[:, :, None], for_src.shape)
  vals = jnp.broadcast_to(phys_rep[:, :, None], for_src.shape)
  cur = route[src_idx, e_idx]
  route = route.at[src_idx, e_idx].set(jnp.where(for_src, vals, cur))
  rep_home = jnp.where(valid, e_safe // local, -1).astype(jnp.int32)
  rep_slot = jnp.where(valid, e_safe % local, 0).astype(jnp.int32)
  sends = jnp.zeros((num_ep, num_ep), bool)
  slot_of = jnp.zeros((num_ep, num_ep), jnp.int32)
  hh = jnp.where(valid, rep_home, 0)
  sends = sends.at[hh, rr].max(valid)
  slot_of = slot_of.at[hh, rr].max(jnp.where(valid, rep_slot, 0))
  pos = jnp.cumsum(sends.astype(jnp.int32), axis=1) - 1
  send_peer_index = jnp.where(sends, pos, -1).astype(jnp.int32)
  send_slots = jnp.zeros((num_ep, max_send), jnp.int32)
  hidx = jnp.broadcast_to(jnp.arange(num_ep)[:, None], (num_ep, num_ep))
  send_slots = send_slots.at[hidx, jnp.clip(pos, 0, max_send - 1)].max(jnp.where(sends, slot_of, 0))
  return ReplicaPlan(route, rep_home, rep_slot, send_slots, send_peer_index, local + r_slots, r_slots)


def compute_plan_iter(counts_src, num_ep, num_replicas, max_send, board_size, enabled=True, rounds=None, shift=True):
  """Loop-free replication plan: a few unrolled rounds of hottest-to-coldest device matching.

  Each round re-ranks devices by their current (post-replication) load and pairs the i-th hottest with
  the ((i + j) mod H)-th coldest (H = EP // 2, j = round). The home sends its hottest not-yet-replicated
  local expert; whole source shards (same board first, never the home's own shard) move up to half the
  load gap. A pair is skipped if it was already used, the receiver has no free slot, or the home already
  sent max_send replicas. Every op is vectorized over the H pairs of a round.
  Same outputs and invariants as compute_plan.
  """
  ep, num_experts = counts_src.shape
  local = num_experts // num_ep
  r_slots = num_replicas
  half = num_ep // 2
  rounds = r_slots if rounds is None else rounds
  counts_src = counts_src.astype(jnp.float32)
  c = jnp.sum(counts_src, axis=0)
  c_loc = c.reshape(num_ep, local)
  dev_load = jnp.sum(c_loc, axis=1)
  src_ids = jnp.arange(num_ep, dtype=jnp.int32)
  pair_ids = jnp.arange(half, dtype=jnp.int32)
  rep_e = jnp.full((num_ep, r_slots), -1, jnp.int32)
  rep_mask = jnp.zeros((num_ep, r_slots, num_ep), bool)
  free = jnp.full((num_ep,), r_slots, jnp.int32)
  send_cnt = jnp.zeros((num_ep,), jnp.int32)
  pair_used = jnp.zeros((num_ep, num_ep), bool)
  sent = jnp.zeros((num_ep, local), bool)
  for j in range(rounds):
    by_load = jnp.argsort(-dev_load).astype(jnp.int32)
    h = by_load[:half]
    cold = by_load[::-1][:half]
    r = cold[(pair_ids + (j if shift else 0)) % half]
    loc = jnp.argmax(jnp.where(sent[h], -jnp.inf, c_loc[h]), axis=1).astype(jnp.int32)  # [H]
    e = h * local + loc
    target = (dev_load[h] - dev_load[r]) / 2.0
    is_home = src_ids[None, :] == h[:, None]
    same_board = (src_ids[None, :] // board_size) == (r[:, None] // board_size)
    key = jnp.where(is_home, 3 * num_ep, jnp.where(same_board, 0, num_ep) + (src_ids[None, :] - r[:, None]) % num_ep)
    contrib = counts_src[:, e].T  # [H, EP(src)]
    before = key[:, None, :] <= key[:, :, None]
    cum = jnp.sum(jnp.where(before, contrib[:, None, :], 0.0), axis=2)
    take = (cum <= target[:, None]) & ~is_home
    moved = jnp.sum(jnp.where(take, contrib, 0.0), axis=1)
    ok = (moved > 0) & enabled & (free[r] > 0) & (send_cnt[h] < max_send) & ~pair_used[h, r] & ~sent[h, loc]
    take = take & ok[:, None]
    moved = jnp.where(ok, moved, 0.0)
    slot = jnp.clip(r_slots - free[r], 0, r_slots - 1)
    dev_load = dev_load.at[h].add(-moved).at[r].add(moved)
    rep_e = rep_e.at[r, slot].set(jnp.where(ok, e, rep_e[r, slot]))
    rep_mask = rep_mask.at[r, slot].set(jnp.where(ok[:, None], take, rep_mask[r, slot]))
    free = free.at[r].add(-ok.astype(jnp.int32))
    send_cnt = send_cnt.at[h].add(ok.astype(jnp.int32))
    pair_used = pair_used.at[h, r].set(pair_used[h, r] | ok)
    sent = sent.at[h, loc].set(sent[h, loc] | ok)
  return _finalize_plan(rep_e, rep_mask, num_ep, num_experts, r_slots, max_send)


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
  rep_home_me = plan.rep_home[me]
  bwd_in_off = jnp.zeros((num_ep,), jnp.int32).at[jnp.where(rep_home_me >= 0, rep_home_me, 0)].max(
      jnp.where(rep_home_me >= 0, jnp.arange(r_slots, dtype=jnp.int32), 0)
  )
  bwd_out_off = jnp.where(plan.send_peer_index[:, me] >= 0, plan.send_peer_index[:, me], 0).astype(jnp.int32)
  idx = (in_off, send_sz, out_off, recv_sz, plan.send_slots[me], bwd_in_off, recv_sz, bwd_out_off, send_sz)
  return _extend_direct(axis_name, r_slots, max_send, jnp.dtype(dtype), jnp.dtype(w_local.dtype), w_local, idx)
