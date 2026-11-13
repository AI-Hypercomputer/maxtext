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

"""Online hot-expert replication planner for the ragged all-to-all MoE path.

Each expert-parallel (EP) device owns L = E / EP "home" experts and R extra
replica slots, so the physical slot space has EP * (L + R) slots. Each step,
after routing, a plan is computed identically on every device from the
all-gathered per-source expert counts using unrolled rounds of vectorized
hottest-to-coldest device matching:

  * the hottest local experts on overloaded devices get at most one replica
    each on light devices, with at most one replica per (home, receiver) pair
    and at most `max_send` replicas sent from any home device;
  * whole source shards' tokens for a replicated expert are rerouted to the
    replica (preferring sources on the receiver's board), so tokens are never
    split inside a source shard and no extra sort is needed.

Each logical expert gets at most one replica per step (`sent[h, loc]`), so a
single expert hotter than roughly 2x a device's mean load cannot be fully
balanced across more than two devices (home + one replica).

Tokens are remapped logical expert -> physical slot before the dispatch sort.
Every (token, k) maps to exactly one physical slot, so it is computed exactly
once with identical weights. Replica weights are moved by `moe_replica_weights`.
"""

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class ReplicaPlan:
  """PyTree container for one layer's online hot-expert replication plan.

  Array fields are traced PyTree children; `num_slots` and `num_replicas` are static Python ints.

  Invariants guaranteed by `compute_plan_iter` and relied on by
  `moe_replica_weights.extend_with_replicas`:
    1. At most one replica per `(home, receiver)` device pair.
    2. Distinct valid home devices (`rep_home[r] >= 0`) across each receiver `r`'s `R` slots.
    3. Valid send entries for home `h` are compacted at the front of `send_slots[h, :k]`
       (`k = sum(send_peer_index[h] >= 0) <= max_send`), ordered by ascending receiver ID so
       `send_peer_index[h, valid_receivers] == [0, ..., k - 1]` and `0 <= send_slots[h, :k] < L`.
  """

  route_table: jax.Array  # [EP, E] int32: (source shard, logical expert) -> physical slot
  rep_home: jax.Array  # [EP, R] int32: home device of each replica slot, -1 if empty
  rep_slot: jax.Array  # [EP, R] int32: home-local slot of each replica, 0 if empty
  send_slots: jax.Array  # [EP, C] int32: local slots each device sends (ordered by receiver)
  send_peer_index: jax.Array  # [EP, EP] int32: index into sender's send buffer for receiver, -1 if none
  num_slots: int = struct.field(pytree_node=False)  # L + R (static python int)
  num_replicas: int = struct.field(pytree_node=False)  # R (static python int)


def identity_route_table(num_ep, num_experts, num_replicas):
  """Returns `[EP, E]` route table mapping every `(src, expert)` to its home slot."""
  local = num_experts // num_ep
  e = jnp.arange(num_experts, dtype=jnp.int32)
  phys = (e // local) * (local + num_replicas) + e % local
  return jnp.broadcast_to(phys[None, :], (num_ep, num_experts))


def _finalize_plan(rep_e, rep_mask, num_ep, num_experts, r_slots, max_send):
  """Builds the physical route table and per-device send metadata from `(rep_e, rep_mask)`."""
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
  # Use `.max` rather than `.set` so empty replica slots (`valid=False`, `e_safe=0`) cannot clobber expert 0.
  reroute_mask = jnp.zeros((num_ep, num_experts), bool).at[src_idx, e_idx].max(for_src)
  reroute_phys = jnp.zeros((num_ep, num_experts), jnp.int32).at[src_idx, e_idx].max(jnp.where(for_src, vals, 0))
  route = jnp.where(reroute_mask, reroute_phys, route)
  rep_home = jnp.where(valid, e_safe // local, -1).astype(jnp.int32)
  rep_slot = jnp.where(valid, e_safe % local, 0).astype(jnp.int32)
  # Send side: for each home h, the replicas it serves, ordered by receiver.
  # send_peer_index[h, r] = position of r's row in h's send buffer.
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
  """Loop-free replication plan: unrolled rounds of hottest-to-coldest device matching.

  Each round re-ranks devices by their current (post-replication) load and pairs the i-th hottest with
  the ((i + j) mod H)-th coldest (H = EP // 2, j = round). The home sends its hottest not-yet-replicated
  local expert; whole source shards (same board first, never the home's own shard) move up to half the
  load gap. A pair is skipped if it was already used, the receiver has no free slot, or the home already
  sent `max_send` replicas. Every op is vectorized over the H pairs of a round.
  """
  _, num_experts = counts_src.shape
  if num_ep < 2 or num_experts % num_ep != 0:
    raise ValueError(f"Expected num_ep >= 2 dividing num_experts ({num_experts}), got num_ep={num_ep}.")
  if num_replicas < 1 or max_send < 1 or board_size < 1:
    raise ValueError(
        f"Expected num_replicas >= 1, max_send >= 1, board_size >= 1; got {num_replicas=}, {max_send=}, {board_size=}."
    )
  local = num_experts // num_ep
  r_slots = num_replicas
  half = num_ep // 2
  rounds = 2 * r_slots if rounds is None else rounds
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
    loc = jnp.argmax(jnp.where(sent[h], -jnp.inf, c_loc[h]), axis=1).astype(jnp.int32)
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
    # Conservative starvation guard (safe heuristic): if every positive non-home source shard
    # of (h, loc) exceeds half the current global max-to-min device load gap (or (h, loc) has
    # zero non-home tokens), no receiver can accept even a single shard in this or any later
    # round (since max(dev_load) is non-increasing and min(dev_load) is non-decreasing across
    # rounds). Mark such experts skipped so h can replicate cooler splittable experts, without
    # wrongly skipping experts that only failed due to a warm partner, full receiver, or key-prefix order.
    max_possible_gap = (jnp.max(dev_load) - jnp.min(dev_load)) / 2.0
    unmovable = enabled & jnp.all((contrib <= 0) | is_home | (contrib > max_possible_gap), axis=1)
    pair_used = pair_used.at[h, r].set(pair_used[h, r] | ok)
    sent = sent.at[h, loc].set(sent[h, loc] | ok | unmovable)
  return _finalize_plan(rep_e, rep_mask, num_ep, num_experts, r_slots, max_send)
