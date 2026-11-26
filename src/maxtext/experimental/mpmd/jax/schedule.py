# Copyright 2026 Google LLC
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

"""Pipeline schedules as data: a flat, static list of (chunk, kind, microbatch).

The driver (`ds_ep_pipeline`) never asks "which schedule am I running?".  It just
replays an `order` list.  Every published schedule is therefore just a different
(n_chunks, placement, order) triple:

    GPipe / 1F1B            C = S,     place[c] = c
    interleaved 1F1B        C = S * V, place[c] = c % S
    V-shape (ZBV/DualPipe)  C = 2 * S, place[c] = c if c < S else 2*S-1-c

`greedy` is one list scheduler covering all of them.  Two knobs decide which
textbook schedule falls out:

  * `cap[c]` -- how many microbatches chunk c may hold in flight.  `cap = M`
    everywhere gives GPipe (fill the pipe, then drain); `cap[c] = C - c` gives
    1F1B (the cap alone forces the F/B alternation -- you do NOT also need to
    prefer backward, and doing so actually *hurts*, see `eager_bwd`).
  * `f_key` / `b_key` -- which of a device's own chunks to advance when several
    are ready.  Only matters when a device owns >1 chunk, i.e. V > 1.

For interleaved and V-shaped placement there is no single rule that is optimal
at every (S, V, M), so `search` enumerates the small heuristic space and scores
it with `simulate`.  Empirically that reaches the analytic optimum
`(M + (S-1)/V) * (t_F + t_B)` at every configuration tested.  The snapshot
rounds do not reproduce Megatron-LM's own interleaved 1F1B (63 against 57 at
S=4, V=2, M=8 in unit costs); `megatron_interleaved` writes it per stage, for
`merge`.

Costs are in units of "one chunk forward"; a backward is 2.  To compare across
different V, divide the makespan by V (each chunk is 1/V of the model).

Kinds and side-cars.  A backward may also be split into 'I' (the input
gradient, including the recompute under remat) and 'W' (the weight gradient),
together one 'B'.  Two optional side-cars ride along with an order without
changing its shape:

  * `modes`: {(chunk, microbatch): mode} -- 'r' (remat, the default: the
    forward keeps only its inputs and the backward recomputes) or a stash
    policy name (the forward keeps a residual set and the backward replays it).
  * `fuse`: {j} -- order[j] (an F) and order[j+1] (a B or I of another
    microbatch on the same stage) run as ONE program.

`validate` is the contract between a generator and the driver; `merge` turns
per-stage action lists (how most published schedules are written down) into
one global order.  Measured per-program costs go through `cost_table`.

Zero bubble.  `zb_schedule` builds ZB-H1 / ZB-H2 (Qi et al.) on one chunk a
stage, with the middle chunks' backwards split into I and W: the fastest of
the handcrafted streams (`zb_h1`, `zb_h2`), the list scheduler (`zb_search`
over `zb_greedy`) and 1F1B with each B as I then W, under a cap on pending W
buffers.  `zb_costs` prices I and W where a cost table has none.  With
`split_last` the fused last chunk splits too (`splits_last`): its F runs the
forward, the head, the loss and the input gradient (`cost_table`'s AI), and a
W of its own per microbatch its layers' weight gradients, later.
"""

from __future__ import annotations

import functools
import inspect

# Backward passes cost ~2x a forward for a typical transformer stage; split,
# that is ~1 for the input gradient and ~1 for the weight gradient.
COST = {"F": 1.0, "B": 2.0, "I": 1.0, "W": 1.0}
KINDS = ("F", "B", "I", "W")


# ---------------------------------------------------------------------- legality
class ScheduleError(ValueError):
  """An order the driver must not run.  Raised before anything is dispatched:
  once a step starts, a bad order fails halfway through, after donated
  buffers are already gone."""


def _fmt(a):
  c, k, m = a
  return f"{c}{k}{m}"


def deps(c, k, m, C):
  """What action (c, k, m) of a C-chunk model waits for.

  `(c', 'BI', m)` stands for whichever of B or I ran for that chunk and
  microbatch -- the op that produces its input gradient.  The last chunk's
  backward is fused into its forward, so the chunk before it waits on a no-op
  B that is done when the last chunk's F is -- and so does a split last
  chunk's W (`splits_last`), whose F ran the input gradient.
  """
  if k == "F":
    return [(c - 1, "F", m)] if c else []
  if k in ("B", "I"):
    return [(c, "F", m)] + ([(c + 1, "BI", m)] if c < C - 1 else [])
  if k == "W":
    return [(c, "BI", m)]
  raise ScheduleError(f"unknown kind {k!r} in {(c, k, m)}")


def validate(order, place, n_micro, *, modes=None, fuse=frozenset(), fuse_last=True, fifo=True):
  """Raise `ScheduleError` unless the driver can replay `order` as is.

  Every rule here is one the driver relies on without checking:

    * every (c, m) has exactly one F; every non-last chunk one B, or one I
      and then one W -- and chunk 0 only B, since it has no input gradient;
    * F(c, m) after F(c-1, m); B or I(c, m) after F(c, m) and after B or
      I(c+1, m); W after its I;
    * `fuse_last`: the last chunk is one fused forward+backward program, so it
      has no I, and its B is a no-op that sits right after its F (after the
      pair, if that F heads one) in its stage's own sequence.  Ops of other
      stages may come between: a timed order lists everything that starts at
      the same instant together.  It has a W for every microbatch or for
      none: split (`splits_last`), its F runs the forward, the head, the loss
      and the input gradient, and its W, after its F (and the no-op B), the
      weight gradients it defers;
    * `fifo`: per chunk, the forwards and every op that writes the gradient
      accumulator (B, W, or the fused last chunk's F -- and a split last
      chunk's W's, which write the leaves its F does not) run in microbatch
      order.  That is what makes two schedules bitwise-interchangeable: the
      fp32 accumulation happens in the same order whatever else moves;
    * each j in `fuse` pairs an F with the next op, a B or I on the same stage
      and of a different microbatch (the same microbatch would depend on
      itself);
    * `modes` name only non-last chunks: the fused last chunk keeps nothing.
  """
  C, M = len(place), n_micro
  pos, spos, n_on = {}, [], [0] * (max(place) + 1)  # spos[j]: index in its stage's sequence
  for j, a in enumerate(order):
    a = tuple(a) if isinstance(a, (tuple, list)) else (a,)
    if len(a) != 3 or a[1] not in KINDS:
      raise ScheduleError(f"order[{j}] = {a!r} is not (chunk, kind in {KINDS}, microbatch)")
    c, k, m = a
    if not (0 <= c < C and 0 <= m < M):
      raise ScheduleError(f"order[{j}] = {_fmt(a)} is outside {C} chunks x {M} microbatches")
    if a in pos:
      raise ScheduleError(f"{_fmt(a)} appears twice (order[{pos[a]}] and order[{j}])")
    pos[a] = j
    spos.append(n_on[place[c]])
    n_on[place[c]] += 1
  fuse = frozenset(fuse)
  for j in sorted(fuse):
    if not 0 <= j < len(order) - 1:
      raise ScheduleError(f"fused pair at {j} is outside the order")
    (cf, kf, mf), (cb, kb, mb) = order[j], order[j + 1]
    if kf != "F" or kb not in ("B", "I"):
      raise ScheduleError(
          f"fused pair at {j} is {_fmt(order[j])}&{_fmt(order[j + 1])}: " "it must be an F and then a B or I"
      )
    if cb == C - 1 and fuse_last:
      raise ScheduleError(f"fused pair at {j}: {_fmt(order[j + 1])} is the fused last chunk's " "no-op backward")
    if place[cf] != place[cb]:
      raise ScheduleError(
          f"fused pair at {j}: {_fmt(order[j])} is on stage {place[cf]} and " f"{_fmt(order[j + 1])} on stage {place[cb]}"
      )
    if mf == mb:
      raise ScheduleError(
          f"fused pair at {j}: {_fmt(order[j])}&{_fmt(order[j + 1])} share " f"microbatch {mf}, so one waits on the other"
      )
  # the fused last chunk split: a W of its own (for every microbatch, checked below)
  last_w = fuse_last and any((C - 1, "W", m) in pos for m in range(M))
  for c in range(C):
    for m in range(M):
      has = {k: (c, k, m) in pos for k in KINDS}
      if not has["F"]:
        raise ScheduleError(f"{c}F{m} is missing")
      if c == C - 1 and fuse_last:
        if has["I"]:
          raise ScheduleError(f"chunk {c} is the fused last chunk: it has no I (its F runs the input gradient)")
        if last_w and not has["W"]:
          raise ScheduleError(f"{c}W{m} is missing: a split last chunk has a W for every microbatch")
        if not has["B"]:
          raise ScheduleError(f"{c}B{m} (the fused last chunk's no-op backward) is missing")
        jf, jb = pos[(c, "F", m)], pos[(c, "B", m)]
        if spos[jb] != spos[jf] + (2 if jf in fuse else 1):
          raise ScheduleError(
              f"{c}B{m} (order[{jb}]) is not right after its fused forward " f"(order[{jf}]) on stage {place[c]}"
          )
        continue
      if c == 0 and (has["I"] or has["W"]):
        raise ScheduleError(f"chunk 0 has no input gradient to split off: 0B{m} only, no I/W")
      if has["B"] and (has["I"] or has["W"]):
        raise ScheduleError(f"({c}, {m}) has both a B and an I/W")
      if not has["B"] and not (has["I"] and has["W"]):
        what = {
            (False, False): f"{c}B{m}",
            (True, False): f"{c}W{m} (after {c}I{m})",
            (False, True): f"{c}I{m} (before {c}W{m})",
        }[(has["I"], has["W"])]
        raise ScheduleError(f"{what} is missing")
  for a, j in pos.items():
    for dc, dk, dm in deps(*a, C):
      if dk == "BI":
        jd = pos.get((dc, "B", dm), pos.get((dc, "I", dm)))
        name = f"{dc}B{dm}/{dc}I{dm}"
      else:
        jd = pos.get((dc, dk, dm))
        name = _fmt((dc, dk, dm))
      if jd is None or jd >= j:
        raise ScheduleError(f"{_fmt(a)} at order[{j}] runs before {name}, which it needs")
  if fifo:
    for c in range(C):
      fs = sorted(range(M), key=lambda m: pos[(c, "F", m)])
      if fs != list(range(M)):
        raise ScheduleError(f"chunk {c} runs its forwards in microbatch order {fs}, not FIFO")
      if c == C - 1 and fuse_last:
        # its gradient writer is the forward, just checked -- and, split, its W's
        ws = sorted(range(M), key=lambda m, c=c: pos[(c, "W", m)]) if last_w else list(range(M))
        if ws != list(range(M)):
          raise ScheduleError(
              f"chunk {c} accumulates its deferred weight gradients in microbatch order {ws}, not FIFO: the fp32 "
              "sum would not be bitwise-reproducible"
          )
        continue
      wr = sorted(range(M), key=lambda m: pos.get((c, "W", m), pos.get((c, "B", m))))
      if wr != list(range(M)):
        raise ScheduleError(
            f"chunk {c} accumulates its weight gradients in microbatch order "
            f"{wr}, not FIFO: the fp32 sum would not be bitwise-reproducible"
        )
  for key, md in (modes or {}).items():
    c, m = key
    if not (0 <= c < C and 0 <= m < M):
      raise ScheduleError(f"mode for {key} is outside {C} chunks x {M} microbatches")
    if c == C - 1:
      raise ScheduleError(f"mode {md!r} on ({c}, {m}): the last chunk keeps no residuals")
    if not isinstance(md, str) or not md:
      raise ScheduleError(f"mode for ({c}, {m}) must be 'r' or a stash policy name, got {md!r}")


# --------------------------------------------------------------------- placement
def placement(n_stages, chunks_per_stage=1, shape="circular"):
  """Which device owns each chunk.  Chunks are always in model order."""
  C = n_stages * chunks_per_stage
  if shape == "circular":
    return [c % n_stages for c in range(C)]
  if shape == "v":
    if chunks_per_stage != 2:
      raise ValueError("v-shape needs chunks_per_stage=2")
    return [c if c < n_stages else 2 * n_stages - 1 - c for c in range(C)]
  raise ValueError(f"unknown shape {shape!r}")


# ---------------------------------------------------------------- list scheduler
def greedy(n_chunks, n_micro, place, cap=None, eager_bwd=False, f_key=None, b_key=None):
  """Emit a legal, deadlock-free (chunk, kind, microbatch) order.

  Devices advance in *snapshot rounds*: every device picks against frozen state,
  then all picks commit together.  Without the snapshot an action issued by
  device d becomes visible to device d+1 in the same round, which walks one
  microbatch through the whole pipe at once and serializes the streams.

  `eager_bwd` prefers B over F when both are ready.  Leave it False: with a 1F1B
  cap it over-fires backward on the early devices and starves everything
  downstream (measured 45% worse makespan at S=4, M=8).
  """
  C, M = n_chunks, n_micro
  cap = [C - c for c in range(C)] if cap is None else list(cap)
  if len(cap) != C:
    raise ValueError(f"cap must have {C} entries, got {len(cap)}")
  ndev = max(place) + 1
  own = [[c for c in range(C) if place[c] == d] for d in range(ndev)]
  fdone, bdone = set(), set()
  nf, nb, inflight = [0] * C, [0] * C, [0] * C
  f_key = f_key or (lambda c, m: (m, c))
  b_key = b_key or (lambda c, m: (m, -c))
  order, total, rounds = [], 2 * C * M, 0

  while len(order) < total:
    picks = []
    for d in range(ndev):

      def ready_b(d=d):
        cs = [c for c in own[d] if nb[c] < M and (c, nb[c]) in fdone and (c == C - 1 or (c + 1, nb[c]) in bdone)]
        if not cs:
          return None
        c = min(cs, key=lambda c: b_key(c, nb[c]))
        return ("B", c, nb[c])

      def ready_f(d=d):
        cs = [c for c in own[d] if nf[c] < M and inflight[c] < cap[c] and (c == 0 or (c - 1, nf[c]) in fdone)]
        if not cs:
          return None
        c = min(cs, key=lambda c: f_key(c, nf[c]))
        return ("F", c, nf[c])

      first, second = (ready_b, ready_f) if eager_bwd else (ready_f, ready_b)
      p = first() or second()
      if p:
        picks.append(p)
    if not picks:
      raise RuntimeError(f"deadlock after {rounds} rounds: nf={nf} nb={nb} " f"inflight={inflight} cap={cap}")
    for k, c, m in picks:
      order.append((c, k, m))
      if k == "F":
        fdone.add((c, m))
        nf[c] += 1
        inflight[c] += 1
        if c == C - 1:  # last chunk fuses F+B, so its F implies its B
          bdone.add((c, m))
          nb[c] += 1
          inflight[c] -= 1
          order.append((c, "B", m))
      else:
        bdone.add((c, m))
        nb[c] += 1
        inflight[c] -= 1
    rounds += 1
  return order


def gpipe(n_chunks, n_micro, place):
  """All forwards, then all backwards.  Holds M residual sets per chunk."""
  return greedy(n_chunks, n_micro, place, cap=[n_micro] * n_chunks)


def one_f_one_b(n_chunks, n_micro, place):
  """PipeDream-Flush.  Same makespan as GPipe, C-c residual sets on chunk c."""
  return greedy(n_chunks, n_micro, place)


def breadth_first(n_chunks, n_micro, place):
  """MaxText's circular-pipeline order: every device runs one of its chunks'
  forwards for all microbatches before it starts its next chunk's, then the
  backwards chunk by chunk in reverse (all forwards, then all backwards).
  Under the circular placement at M >= S it reaches the interleaved optimum,
  a bubble of (S-1)/(V M + S-1) at unit costs, and it visits each chunk at
  most twice a step -- once forward, once backward -- which is what a
  pipeline holding only the chunk it runs (`ds_ep_pipeline --fsdp-scope
  repeat`) needs.  `search` finds the same order at S=4, V=2 and 4, M=8; this
  pins it.  It is ds_ep_pipeline's default: a microbatch's stage crossings
  overlap the other microbatches' forwards (on 2 x v5p-128 with the relay
  boundary over DCN, M=8, a step took 5521 ms in this order and 8799 ms in
  gpipe's).  On V placement its bubble is longer than `search`'s.  Within a chunk the backwards run microbatch 0 first;
  MaxText's transposed scan runs them M-1 first."""
  return greedy(n_chunks, n_micro, place, cap=[n_micro] * n_chunks, f_key=lambda c, m: (c, m), b_key=lambda c, m: (-c, m))


# ------------------------------------------------------------- heuristic search
def _keys(blk):
  """Priority over a device's ready chunks, keyed on (chunk, next microbatch).

  `blk` groups microbatches into runs of `blk` on one chunk before rotating to
  the next -- Megatron's interleaved warmup rule, which wants blk = n_stages.
  """
  b = max(1, blk)
  return {
      "mb_lo_chunk": lambda c, m: (m // b, c, m),
      "mb_hi_chunk": lambda c, m: (m // b, -c, m),
      "chunk_lo_mb": lambda c, m: (c, m),
      "chunk_hi_mb": lambda c, m: (-c, m),
  }


def _caps(n_chunks, n_micro, place):
  """Candidate in-flight limits, cheapest memory first."""
  C, M = n_chunks, n_micro
  S = max(place) + 1
  V = C // S
  # Megatron-interleaved: device d warms up on (S-d-1)*2 + (V-1)*S microbatches,
  # spread over the V chunks it owns (the deeper chunk needs fewer).
  meg = [max(1, min(M, (S - place[c] - 1) * 2 + (V - 1 - c // S) * S + 1)) for c in range(C)]
  return {"1f1b": [C - c for c in range(C)], "megatron": meg, "none": [M] * C}


def _enumerate(n_chunks, n_micro, place, caps=None):
  """Yield (makespan, peak_residual_sets, order, tag) for the heuristic space."""
  C, M = n_chunks, n_micro
  S = max(place) + 1
  cands = _caps(C, M, place) if caps is None else caps
  for cap_name, cap in cands.items():
    for blk in sorted({1, S}):
      for eager in (False, True):
        keys = _keys(blk)
        for fn, fk in keys.items():
          for bn, bk in keys.items():
            try:
              o = greedy(C, M, place, cap=cap, eager_bwd=eager, f_key=fk, b_key=bk)
            except RuntimeError:
              continue  # this combination deadlocks; skip it
            yield (simulate(o, place)[0], max(peak_residuals(o, C)), o, (cap_name, blk, eager, fn, bn))


def search(n_chunks, n_micro, place, max_residual_sets=None, caps=None):
  """Fastest schedule in the heuristic space, subject to a memory budget.

  Returns (order, tag, makespan).  `max_residual_sets` caps how many in-flight
  microbatches any single chunk may hold; leave None for pure speed.  Ties break
  toward less memory.

  There is no single textbook rule that is optimal for every (S, V, M) once a
  device owns more than one chunk, so we score candidates with `simulate` rather
  than hard-coding one.
  """
  if max_residual_sets is not None:
    if max_residual_sets < 1:
      raise ValueError("max_residual_sets must be >= 1 to make progress")
    caps = _caps(n_chunks, n_micro, place) if caps is None else caps
    caps = {k: [min(c, max_residual_sets) for c in v] for k, v in caps.items()}
  best = None
  for mk, peak, o, tag in _enumerate(n_chunks, n_micro, place, caps):
    if max_residual_sets is not None and peak > max_residual_sets:
      continue
    if best is None or (mk, peak) < (best[0], best[1]):
      best = (mk, peak, o, tag)
  if best is None:
    raise RuntimeError("no schedule fits the residual budget " f"{max_residual_sets}; try raising it")
  return best[2], best[3], best[0]


def pareto(n_chunks, n_micro, place, caps=None):
  """The (makespan, peak_residual_sets) frontier, fastest first.

  Interleaving buys speed with memory, and how much of each you want depends on
  the model -- so hand back the whole tradeoff instead of picking for the user.
  """
  pts = {}
  for mk, peak, o, tag in _enumerate(n_chunks, n_micro, place, caps):
    if (mk, peak) not in pts:
      pts[(mk, peak)] = (o, tag)
  keys = sorted(pts)
  front = [k for k in keys if not any(q[0] <= k[0] and q[1] <= k[1] and q != k for q in keys)]
  return [(mk, peak) + pts[(mk, peak)] for mk, peak in front]


# ------------------------------------------------------------------- cost model
def _by_kind(cost):
  """A dict that prices by kind alone, like `COST` (not a `cost_table`)."""
  return isinstance(cost, dict) and set(cost) <= set(KINDS)


def _cost_fn(cost, C):
  """Any cost spec -> cost(c, kind, mode), for C chunks.

  None means `COST`.  A dict of kinds prices by kind alone; any other dict is
  a program-cost table such as `V6E_COSTS` and goes through `cost_table`.  A
  callable gets the mode if it has a `mode` parameter, takes *args, or
  requires three arguments; anything else is the older `cost(c, kind)` (maybe
  with defaulted extras, as closures over a table often have) and is wrapped
  so it never sees the mode.
  """
  cost = cost or COST
  if _by_kind(cost):
    return lambda c, k, mode="r", table=cost: table[k]
  if isinstance(cost, dict):
    return cost_table(cost, C)
  try:
    ps = inspect.signature(cost).parameters
  except (TypeError, ValueError):  # no introspectable signature: trust it
    return cost
  if "mode" in ps:
    return lambda c, k, mode="r": cost(c, k, mode=mode)
  pos = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
  if (
      any(p.kind == p.VAR_POSITIONAL for p in ps.values())
      or sum(p.kind in pos and p.default is p.empty for p in ps.values()) >= 3
  ):
    return cost
  return lambda c, k, mode="r": cost(c, k)


def _programs(order, fuse):
  """The ops of each program in turn: one op, or the two of a fused pair."""
  j = 0
  while j < len(order):
    n = 2 if j in fuse else 1
    yield order[j : j + n]
    j += n


def simulate(order, place, cost=None, *, modes=None, fuse=frozenset(), pair_gain=0.0):
  """Replay `order` against a per-device timeline.  -> (makespan, busy, util).

  Assumes transfers are free, which on a v6e sub-mesh boundary they nearly are
  (D2D at 64-184 GB/s depending on payload size, and overlapped with compute by
  JAX's async dispatch).

  `cost` is a {kind: cost} dict (default `COST`), a program-cost table (see
  `cost_table`) or `cost(c, kind, mode)`, and `modes` gives each (c, m) its
  mode.  A fused pair (j in `fuse`) is one busy interval of both costs minus
  `pair_gain`: it starts once both halves are ready, and both results land at
  its end.  A split last chunk (`splits_last`) has its no-op B priced as the
  input gradient its F runs, its I: F + I + W, not F + B + W.
  """
  # `search` scores ~200 orders with this, so the common case -- one op
  # priced by kind -- stays a straight-line loop.
  table = (cost or COST) if _by_kind(cost or COST) else None
  cost, modes = _cost_fn(cost, len(place)), modes or {}
  ndev = max(place) + 1
  free, busy, done = [0.0] * ndev, [0.0] * ndev, {}
  last = len(place) - 1 if splits_last(order, len(place)) else None

  def dep(c, k, m):
    if k == "F":
      return done.get((c - 1, "F", m), 0.0) if c else 0.0
    if k == "W":  # after its I -- a split last chunk's after its (no-op) B
      return done.get((c, "I", m), done.get((c, "B", m), 0.0))
    return max(done.get((c, "F", m), 0.0), done.get((c + 1, "B", m), done.get((c + 1, "I", m), 0.0)))

  j, n = 0, len(order)
  while j < n:
    if j in fuse:
      ops = order[j : j + 2]
      d = place[ops[0][0]]
      dur = sum(cost(c, "I" if c == last and k == "B" else k, modes.get((c, m), "r")) for c, k, m in ops) - pair_gain
      t = max(free[d], *(dep(*a) for a in ops)) + dur
      for a in ops:
        done[tuple(a)] = t
      j += 2
    else:
      c, k, m = order[j]
      d = place[c]
      p = "I" if c == last and k == "B" else k
      dur = table[p] if table is not None else cost(c, p, modes.get((c, m), "r"))
      t = max(free[d], dep(c, k, m)) + dur
      done[(c, k, m)] = t
      j += 1
    free[d] = t
    busy[d] += dur
  mk = max(free)
  return mk, busy, [b / mk for b in busy]


def peak_residuals(order, n_chunks):
  """Max simultaneously-live residual sets per chunk.  Multiply by the stage's
  per-microbatch residual bytes to get HBM.  The last chunk stores nothing --
  the driver fuses its forward and backward.  A set lives from its F to its
  B or I (the I consumes the residuals; W keeps a buffer, see `peak_wbufs`)."""
  live, peak = [0] * n_chunks, [0] * n_chunks
  for c, k, _ in order:
    if c == n_chunks - 1:
      continue
    if k == "F":
      live[c] += 1
      peak[c] = max(peak[c], live[c])
    elif k in ("B", "I"):
      live[c] -= 1
  return peak


def _set_size(set_bytes):
  """`set_bytes` -> size(c, mode).  A callable is used as is; a mapping gives
  bytes per mode, each a number or a per-chunk sequence / dict.  Left out,
  'r' costs 0 (so {'lite': ...} measures stash alone); any other mode it does
  not name raises KeyError, as `cost_table` does, rather than count a live
  stash set as free."""
  if callable(set_bytes):
    return set_bytes

  def size(c, mode):
    if mode not in set_bytes:
      if mode == "r":
        return 0
      raise KeyError(f"no residual-set bytes for mode {mode!r} (have {sorted(set_bytes)})")
    b = set_bytes[mode]
    return b[c] if isinstance(b, (list, tuple, dict)) else b

  return size


def peak_bytes(order, place, modes, set_bytes, temp_bytes=None):
  """Peak bytes of live residual sets per stage.  -> [bytes per stage].

  Like `peak_residuals`, walked in order (a set lives from its F to its B or
  I), but summed over the chunks each stage owns and priced by mode:
  `set_bytes(c, mode)`, or a {mode: bytes or per-chunk bytes} mapping (which
  must name every stash mode used; 'r' defaults to 0).  With modes None every
  set is 'r'.  A fused pair counts its F's new set and its B's old one
  together, as the program holds both.  `temp_bytes`, in the same forms:
  what a B / I of that mode holds in temporaries over a remat backward's,
  added while it runs, its own set still live -- the stash note's bound on
  how much the step's peak grows, instead of the live sets alone.
  """
  C, size, modes = len(place), _set_size(set_bytes), modes or {}
  temp = (lambda c, mode: 0) if temp_bytes is None else _set_size(temp_bytes)
  live, peak = [0] * (max(place) + 1), [0] * (max(place) + 1)
  for c, k, m in order:
    if c == C - 1:
      continue
    s, mode = place[c], modes.get((c, m), "r")
    if k == "F":
      live[s] += size(c, mode)
      peak[s] = max(peak[s], live[s])
    elif k in ("B", "I"):
      peak[s] = max(peak[s], live[s] + temp(c, mode))
      live[s] -= size(c, mode)
  return peak


def splits_last(order, n_chunks):
  """Does `order` split the fused last chunk, chunk n_chunks - 1 -- has it W's of its own (and no I: its F runs
  the input gradient)?  Its F then runs the forward, the head, the loss and the input gradient, priced
  cost(C-1, 'F') + cost(C-1, 'I') (`cost_table`'s AI), and leaves a W buffer for its W, which runs the weight
  gradients it defers."""
  ks = {k for c, k, _ in order if c == n_chunks - 1}
  return "W" in ks and "I" not in ks


def peak_wbufs(order, place):
  """Max pending weight-gradient buffers per stage: an I leaves one (what W
  needs: layer inputs and output cotangents) and its W consumes it -- as does
  a split last chunk's F (`splits_last`)."""
  live, peak = [0] * (max(place) + 1), [0] * (max(place) + 1)
  C = len(place)
  last = splits_last(order, C)
  for c, k, _ in order:
    s = place[c]
    if k == "I" or (last and c == C - 1 and k == "F"):
      live[s] += 1
      peak[s] = max(peak[s], live[s])
    elif k == "W":
      live[s] -= 1
  return peak


def peak_inflight(order, place):
  """Max microbatches in flight per stage: each from its F to the op that ends its backward, its W (or its B where the
  backward is whole) -- a residual set until its I, then a pending W buffer.  The paper's memory count, which
  `zb_greedy`'s `inflight_cap` bounds.  The fused last chunk holds no residual set: nothing here (a split one's W
  buffers, from its F to its W, are `peak_wbufs`', which `wbuf_cap` bounds)."""
  C = len(place)
  live, peak = [0] * (max(place) + 1), [0] * (max(place) + 1)
  for c, k, _ in order:
    if c == C - 1:
      continue
    s = place[c]
    if k == "F":
      live[s] += 1
      peak[s] = max(peak[s], live[s])
    elif k in ("B", "W"):
      live[s] -= 1
  return peak


def bubble(order, place, cost=None, *, modes=None, fuse=frozenset(), pair_gain=0.0):
  """Fraction of the makespan the busiest device spends idle."""
  mk, busy, _ = simulate(order, place, cost, modes=modes, fuse=fuse, pair_gain=pair_gain)
  return (mk - max(busy)) / mk


def describe(order, place, n_chunks=None):
  """Per-device action streams, for eyeballing that a schedule is what you think."""
  C = n_chunks or max(c for c, _, _ in order) + 1
  lines = []
  for d in range(max(place) + 1):
    acts = [f"{k}{m}" if C == max(place) + 1 else f"c{c}{k}{m}" for c, k, m in order if place[c] == d]
    lines.append(f"  dev{d}: " + " ".join(acts))
  mk, busy, util = simulate(order, place)
  lines.append(
      f"  makespan={mk:.1f} bubble={bubble(order, place) * 100:.1f}% " f"peak_residual_sets={peak_residuals(order, C)}"
  )
  return "\n".join(lines)


# ------------------------------------------------ timed list scheduling
def timed_greedy(n_chunks, n_micro, place, cost, xfer=0.0, prefer="F", f_key=None, b_key=None, cap=None, fuse_last=False):
  """Event-driven list scheduler with per-(chunk, kind) costs.

  `greedy` advances in unit-time snapshot rounds, which is exact only when
  every action costs the same.  Here each device, whenever it goes idle,
  starts the best action that is READY at that instant (dependencies done
  plus `xfer` for a cross-device hop); if none is ready it waits for the
  earliest one.  `cost(c, kind)`, or a cost dict as for `timed_simulate`,
  gives action durations; `prefer` picks F or B when both are ready;
  `fuse_last` makes the last chunk's F imply its B (the driver then runs one
  fused program).

  Returns (order, makespan).  The order is sorted by start time, which is a
  valid topological order for the single-threaded driver.
  """
  C, M = n_chunks, n_micro
  ndev = max(place) + 1
  if isinstance(cost, dict):
    cost = _cost_fn(cost, C)
  cap = [M] * C if cap is None else list(cap)
  f_key = f_key or (lambda c, m: (c, m))
  b_key = b_key or (lambda c, m: (-c, m))
  done = {}
  nf, nb, inflight = [0] * C, [0] * C, [0] * C
  free = [0.0] * ndev
  started = []

  def ready_time(c, k, m):
    if k == "F":
      if c == 0:
        return 0.0
      t = done.get((c - 1, "F", m))
      return None if t is None else t + (xfer if place[c - 1] != place[c] else 0.0)
    tf = done.get((c, "F", m))
    if tf is None:
      return None
    if c == C - 1:
      return tf
    tb = done.get((c + 1, "B", m))
    if tb is None:
      return None
    return max(tf, tb + (xfer if place[c + 1] != place[c] else 0.0))

  total = 2 * C * M
  while len(started) < total:
    # the device that frees up first picks next
    best = None
    for d in range(ndev):
      cands = []
      for c in range(C):
        if place[c] != d:
          continue
        if nf[c] < M and inflight[c] < cap[c]:
          r = ready_time(c, "F", nf[c])
          if r is not None:
            cands.append((max(r, free[d]), "F", c, nf[c]))
        if nb[c] < M and not (fuse_last and c == C - 1):
          r = ready_time(c, "B", nb[c])
          if r is not None:
            cands.append((max(r, free[d]), "B", c, nb[c]))
      if not cands:
        continue
      t0 = min(x[0] for x in cands)
      now = [x for x in cands if x[0] <= t0 + 1e-9]

      def rank(x):
        _, k, c, m = x
        pk = 0 if k == prefer else 1
        return (pk, f_key(c, m) if k == "F" else b_key(c, m))

      pick = min(now, key=rank)
      if best is None or pick[0] < best[0] - 1e-9 or (abs(pick[0] - best[0]) <= 1e-9 and d < best[4]):
        best = (*pick, d)
    if best is None:
      raise RuntimeError(f"timed_greedy deadlock: nf={nf} nb={nb} inflight={inflight}")
    t, k, c, m, d = best
    dur = cost(c, k) + (cost(c, "B") if (fuse_last and c == C - 1 and k == "F") else 0.0)
    free[d] = t + dur
    done[(c, k, m)] = t + dur
    started.append((t, c, k, m))
    if k == "F":
      nf[c] += 1
      inflight[c] += 1
      if fuse_last and c == C - 1:
        done[(c, "B", m)] = t + dur
        nb[c] += 1
        inflight[c] -= 1
        started.append((t + 1e-9, c, "B", m))
    else:
      nb[c] += 1
      inflight[c] -= 1
  started.sort(key=lambda x: x[0])
  return [(c, k, m) for _, c, k, m in started], max(free)


def timed_search(n_chunks, n_micro, place, cost, xfer=0.0, fuse_last=False, caps=None):
  """Best `timed_greedy` order over a small policy space.  -> (order, tag, makespan)."""
  C = n_chunks
  if isinstance(cost, dict):
    cost = _cost_fn(cost, C)
  S = max(place) + 1
  keys = {**_keys(1), **{f"{k}/blk{S}": v for k, v in _keys(S).items()}}
  cands = caps or {"none": None, "1f1b": [C - c for c in range(C)], **{f"cap{v}": [v] * C for v in (2, 3, 4)}}
  best = None
  for cn, cap in cands.items():
    for prefer in ("F", "B"):
      for fn, fk in keys.items():
        for bn, bk in keys.items():
          try:
            o, mk = timed_greedy(C, n_micro, place, cost, xfer, prefer, fk, bk, cap, fuse_last)
          except RuntimeError:
            continue
          if best is None or mk < best[2] - 1e-9:
            best = (o, (cn, prefer, fn, bn), mk)
  return best


# ------------------------------------------------- stash list scheduling
# How the stash scheduler sees memory.  A stash-mode forward leaves a
# residual set (`ds_ep.ChunkPrograms.build_stash`: `bytes` per device and
# microbatch) that stays live on its stage until its backward consumes it
# (donated; it is live while that backward runs, as an argument).  The budget
# is the bytes of simultaneously-live stash sets per stage -- `peak_bytes`
# with {policy: bytes}, which leaves out the layer inputs every mode keeps --
# on top of what a remat step holds.  That is not all the step's peak grows
# by: the stash programs' temporaries are LARGER than the remat ones', not
# smaller (G1, `memory_analysis` at --preset v6e).  A stash backward holds
# +0.38 GiB (attn, attn_min), +0.42 / +0.67 (lite, chunks 2 / 0) and
# -0.08 / +0.17 (lite_y) over the remat backward's 6.34 GiB -- STASH_BWD_TEMP,
# as shares of the set -- so the peak can grow by the live sets plus that:
# 4.06 + 0.67 GiB for three lite sets at config A, over a 4.5 GiB budget
# (`peak_bytes(..., temp_bytes=)`, `Pipeline.peak_stash_gib(temps=True)`,
# which the summary prints next to the budget).  The stash forwards' extra
# temporaries (+0.8 to 1.7 GiB over 1.75) stay below a backward's.
# How much headroom a chip really has (31.25 GiB on a v6e, less params,
# optimizer state, gathered weights, gradient accumulators, the largest
# program's temporaries and fragmentation) is a measurement, not this model:
# the default budget must be set from a `--mem-report` sweep on the TPU (G2),
# reading each run's peak against that bound, not against the budget alone.


def _per(x, n, what):
  """A number (every index), a sequence of n, or {index: value} (0 elsewhere) -> list of n."""
  if isinstance(x, dict):
    return [x.get(i, 0) for i in range(n)]
  if isinstance(x, (list, tuple)):
    if len(x) != n:
      raise ValueError(f"{what} has {len(x)} entries, want {n}")
    return list(x)
  return [x] * n


def stash_greedy(
    n_chunks,
    n_micro,
    place,
    cost,
    xfer=0.0,
    stash_bytes=0,
    budget_bytes=0,
    *,
    policy="s",
    caps=None,
    prefer="F",
    f_key=None,
    b_key=None,
    prio=None,
):
  """`timed_greedy`'s event engine, choosing each forward's mode as it goes.

  -> (order, modes, makespan, peak): `modes` is {(c, m): policy} for the
  stash-mode forwards (every other (c, m) is remat, 'r'), `peak` the peak
  live stash bytes per stage.

  Whenever a device goes idle it starts the best action READY at that
  instant, as `timed_greedy` does, ranked by `prefer` and `f_key` / `b_key`
  -- or by a priority vector `prio[(chunk, kind)]`, lower first, ties to the
  lower microbatch.  Picking a forward of a non-last chunk c sets its mode:
  stash (`policy`) if c holds fewer than `caps[c]` live stash sets (None: no
  cap) and its stage's live stash bytes plus `stash_bytes[c]` stay within
  `budget_bytes` (per stage; see the memory note above), else remat.  Its
  backward then costs `cost(c, 'B', policy)` and frees the set.  `cost` is
  as for `timed_simulate`; `stash_bytes` a number or one per chunk.

  The last chunk is one fused program (the driver's only mode), its no-op B
  right after its F.  Every chunk runs its forwards and backwards in
  microbatch order (FIFO), so the fp32 gradient accumulation happens in the
  same order as under any other schedule: order and modes change no
  gradient bit.  Deterministic.
  """
  C, M = n_chunks, n_micro
  S = max(place) + 1
  cost = _cost_fn(cost, C)
  sb = _per(stash_bytes, C, "stash_bytes")
  bud = _per(budget_bytes, S, "budget_bytes")
  cap = [M] * C if caps is None else [M if x is None else x for x in _per(caps, C, "caps")]
  f_key = f_key or (lambda c, m: (c, m))
  b_key = b_key or (lambda c, m: (-c, m))
  if prio is not None:
    rank = lambda k, c, m: (prio[(c, k)], m)
  else:
    rank = lambda k, c, m: (k != prefer, f_key(c, m) if k == "F" else b_key(c, m))
  own = [[c for c in range(C) if place[c] == d] for d in range(S)]
  hop_f = [0.0] + [xfer if place[c - 1] != place[c] else 0.0 for c in range(1, C)]  # into F(c)
  hop_b = [xfer if place[c + 1] != place[c] else 0.0 for c in range(C - 1)] + [0.0]  # into B(c)
  done_f = [[None] * M for _ in range(C)]
  done_b = [[None] * M for _ in range(C)]
  nf, nb, slive = [0] * C, [0] * C, [0] * C
  free, live, peak = [0.0] * S, [0] * S, [0] * S
  modes, picks = {}, []
  for _ in range(2 * C * M - M):  # every op but the last chunk's no-op backwards
    best = None
    for d in range(S):
      cands = []
      for c in own[d]:
        if nf[c] < M:
          m = nf[c]
          r = 0.0 if c == 0 else done_f[c - 1][m]
          if r is not None:
            cands.append((max(r + hop_f[c], free[d]), "F", c, m))
        if c != C - 1 and nb[c] < M:
          m = nb[c]
          tf, tb = done_f[c][m], done_b[c + 1][m]
          if tf is not None and tb is not None:
            cands.append((max(tf, tb + hop_b[c], free[d]), "B", c, m))
      if not cands:
        continue
      t0 = min(x[0] for x in cands)
      pick = min((x for x in cands if x[0] <= t0 + 1e-9), key=lambda x: rank(*x[1:]))
      if best is None or pick[0] < best[0] - 1e-9:
        best = pick + (d,)
    if best is None:
      raise RuntimeError(f"stash_greedy deadlock: nf={nf} nb={nb}")
    t, k, c, m, d = best
    s = place[c]
    if k == "F":
      mode = "r"
      if c != C - 1 and slive[c] < cap[c] and live[s] + sb[c] <= bud[s]:
        mode = policy
        modes[(c, m)] = mode
        slive[c] += 1
        live[s] += sb[c]
        peak[s] = max(peak[s], live[s])
      dur = cost(c, "F", mode) + (cost(c, "B") if c == C - 1 else 0.0)
      done_f[c][m] = t + dur
      if c == C - 1:
        done_b[c][m] = t + dur
      nf[c] += 1
    else:
      mode = modes.get((c, m), "r")
      if mode != "r":
        slive[c] -= 1
        live[s] -= sb[c]
      dur = cost(c, "B", mode)
      done_b[c][m] = t + dur
      nb[c] += 1
    free[d] = t + dur
    picks.append((t, c, k, m))
  # Picks are made in start order (a device only ever starts at or after the
  # instant of the previous pick), so the list is already a topological order;
  # the stable sort only settles ties the 1e-9 slack let through.
  order = []
  for _, c, k, m in sorted(picks, key=lambda p: p[0]):
    order.append((c, k, m))
    if c == C - 1:
      order.append((c, "B", m))
  return order, modes, max(free), peak


def _stash_caps(C, M, place, sb, bud, limit=32):
  """Per-chunk live stash caps that fill each stage's budget: every maximal
  allocation (one more set of any chunk would not fit), as the product over
  stages; the most even `limit` of them if there are more."""
  S = max(place) + 1
  per_stage = []
  for s in range(S):
    chs = [c for c in range(C - 1) if place[c] == s and sb[c] > 0]
    allocs = []

    def rec(i, left, cur, chs=chs, allocs=allocs):
      if i == len(chs):
        if all(sb[c] > left or n == M for c, n in zip(chs, cur)):
          allocs.append(tuple(cur))
        return
      for n in range(min(M, int(left // sb[chs[i]])) + 1):
        rec(i + 1, left - n * sb[chs[i]], cur + [n])

    rec(0, bud[s], [])
    per_stage.append((chs, allocs))
  out = [[M if sb[c] <= 0 and c != C - 1 else 0 for c in range(C)]]
  for chs, allocs in per_stage:
    nxt = []
    for cap in out:
      for a in allocs:
        cap = list(cap)
        for c, n in zip(chs, a):
          cap[c] = n
        nxt.append(cap)
    out = nxt
  if len(out) > limit:
    spread = lambda cap: max(cap[: C - 1]) - min(cap[: C - 1])
    out = sorted(out, key=spread)[:limit]
  return out


def stash_search(
    n_chunks, n_micro, place, cost, xfer=0.0, stash_bytes=0, budget_bytes=0, *, policy="s", caps=None, trials=128, seed=0
):
  """Best `stash_greedy` order and modes under a stash budget.
  -> (order, modes, tag, makespan)

  Tries per-chunk caps -- `caps` if given, else every allocation that fills
  each stage's `budget_bytes` with whole sets (one more set of any chunk
  would not fit) plus none at all (the budget alone, first come first
  served) -- against `timed_search`'s rank families and `trials` random
  priority vectors over (chunk, kind) drawn from `seed`.  The fastest wins;
  ties go to the least peak stash, then the first found.  Deterministic.

  Which forwards to stash is not only a matter of memory: a set lives from
  its forward to its backward, so stashing chunk 0 -- whose backward comes
  last -- holds budget for a whole round trip that later chunks could reuse
  several times over.  The caps are how the search tries that trade.
  """
  import random

  C, M = n_chunks, n_micro
  S = max(place) + 1
  cost = _cost_fn(cost, C)
  sb = _per(stash_bytes, C, "stash_bytes")
  bud = _per(budget_bytes, S, "budget_bytes")
  cands = [list(caps)] if caps is not None else _stash_caps(C, M, place, sb, bud) + [None]
  keys = {**_keys(1), **{f"{k}/blk{S}": v for k, v in _keys(S).items()}}
  ranks = [
      ((prefer, fn, bn), dict(prefer=prefer, f_key=fk, b_key=bk))
      for prefer in ("F", "B")
      for fn, fk in keys.items()
      for bn, bk in keys.items()
  ]
  rng = random.Random(seed)
  for t in range(trials):
    ranks.append((("prio", t), dict(prio={(c, k): rng.random() for c in range(C) for k in "FB"})))
  best = None
  for cap in cands:
    for rtag, kw in ranks:
      o, md, mk, pk = stash_greedy(C, M, place, cost, xfer, sb, bud, policy=policy, caps=cap, **kw)
      if best is None or mk < best[3] - 1e-9 or (mk <= best[3] + 1e-9 and sum(pk) < best[4]):
        best = (o, md, ("budget" if cap is None else tuple(cap), rtag), mk, sum(pk))
  return best[:4]


# The stash programs' costs as shares of a non-first chunk's remat forward F,
# for a cost table that has none of its own: what G1 measured on v6e (chunk
# 2, `--tiles v6e`, progbench deltas against the remat programs; F = 22.77
# ms), carried to other shapes and chips in proportion -- an estimate there.
#   STASH_SAVES      what the stash backward no longer recomputes: Bs = B -
#                    share x F.  (attn skips the attention projections and
#                    the splash forward, lite also the gathered tokens and the
#                    experts' up-projections, lite_y gmm_wo too; 'all' keeps
#                    the vjp's own intermediates and loses some of that to
#                    the extra memory traffic.)
#   STASH_FWD_EXTRA  what the stash forward costs over the remat one: Fs = F
#                    + share x F (writing the stash out, and the fusions it
#                    breaks).
STASH_SAVES = dict(attn=0.272, attn_min=0.244, lite=0.421, lite_y=0.534, all=0.440)
STASH_FWD_EXTRA = dict(attn=0.082, attn_min=0.087, lite=0.120, lite_y=0.215, all=0.200)
# What a stash backward holds in temporaries over the remat backward it
# replaces, as a share of its stash set's bytes (G1, `memory_analysis` temps
# of bwda_s less bwda's 6.34 GiB, the worse of chunks 0 and 2 -- attn was
# measured on chunk 2 only): attn +0.38 GiB on a 0.713 set, attn_min +0.38 on
# 0.659, lite +0.42 / +0.67 on 1.354, lite_y -0.08 / +0.17 on 2.229, all
# -0.83 / -0.86 (0: never counted below the remat backward).  See the stash
# note above; the scheduler's budget does not include it.
STASH_BWD_TEMP = dict(attn=0.533, attn_min=0.577, lite=0.495, lite_y=0.076, all=0.0)


def stash_estimates(k, policy, n_chunks):
  """{'Bs{c}:{policy}': ms, 'Fs{c}:{policy}': ms} for every non-last chunk c:
  ESTIMATES of the stash backward and forward from the remat costs of the
  table `k` (as `cost_table` reads it) and the shares above of the forward
  F of a non-first chunk (chunk 0's embedding is not recomputed in either
  mode, nor stashed).  For a table without stash costs of its own (the
  driver fills them in: TPU7x, the padded tiles' v6e table (V6E_COSTS), an
  `xprof costs` table)."""
  cost = cost_table(k, n_chunks)
  f = float(k["F"]) if "F" in k else cost(min(1, n_chunks - 2), "F")
  out = {}
  for c in range(n_chunks - 1):
    out[f"Bs{c}:{policy}"] = round(cost(c, "B") - STASH_SAVES[policy] * f, 2)
    out[f"Fs{c}:{policy}"] = round(cost(c, "F") + STASH_FWD_EXTRA[policy] * f, 2)
  return out


def timed_simulate(order, place, cost, xfer=0.0, fuse_last=False, *, modes=None, fuse=frozenset(), pair_gain=0.0):
  """Replay a fixed order with real costs.  -> (makespan, per-device busy).

  `cost(c, kind, mode)`, the older `cost(c, kind)`, or a program-cost table
  such as `V6E_COSTS`; `modes` gives each (c, m) its mode; a fused pair (j in
  `fuse`) is one program costing both halves minus `pair_gain`, whose results
  both land at its end.  `fuse_last`: the last chunk's F costs its F + B, or,
  split (`splits_last`: W's of its own), its F + I.
  """
  ndev = max(place) + 1
  C = max(c for c, _, _ in order) + 1
  cost, modes = _cost_fn(cost, C), modes or {}
  free, busy, done = [0.0] * ndev, [0.0] * ndev, {}
  rest = "I" if splits_last(order, C) else "B"  # what the fused last chunk's F runs besides its forward

  def dep_dur(c, k, m, d):
    mode = modes.get((c, m), "r")
    if k == "F":
      dep = 0.0 if c == 0 else done[(c - 1, "F", m)] + (xfer if place[c - 1] != d else 0.0)
      return dep, cost(c, "F", mode) + (cost(c, rest, mode) if fuse_last and c == C - 1 else 0.0)
    if k == "W":
      return done[(c, "BI", m)], cost(c, "W", mode)
    dep = done[(c, "F", m)]
    if c < C - 1:
      dep = max(dep, done[(c + 1, "BI", m)] + (xfer if place[c + 1] != d else 0.0))
    return dep, cost(c, k, mode)

  for ops in _programs(order, fuse):
    c, k, m = ops[0]
    d = place[c]
    if len(ops) == 1 and fuse_last and c == C - 1 and k == "B":
      done[(c, k, m)] = done[(c, "BI", m)] = done[(c, "F", m)]
      continue
    if len(ops) == 1:
      dep, dur = dep_dur(c, k, m, d)
    else:
      dd = [dep_dur(*a, d) for a in ops]
      dep, dur = max(x for x, _ in dd), sum(y for _, y in dd) - pair_gain
    t = max(free[d], dep) + dur
    for c, k, m in ops:
      done[(c, k, m)] = t
      if k in ("B", "I"):
        done[(c, "BI", m)] = t
    free[d] = t
    busy[d] += dur
  return max(free), busy


# ------------------------------------------------------ per-stage streams
def merge(streams, place, cost, xfer=0.0, pair_gain=0.0, *, modes=None, n_micro=None):
  """Per-stage action lists -> one global order.  -> (order, fuse, makespan)

  Most published schedules (1F1B, ZB-V, DualPipeV) are written down as what
  each stage runs, in its own order; the driver replays one global list.
  `streams[s]` is stage s's list of items, each an op `(c, kind, m)` or a
  fused pair `((cF, 'F', i), (cB, 'B' or 'I', j))` run as one program.

  Event-driven, like `timed_greedy` with every choice already made: start
  whichever stage's next item is ready earliest (its dependencies done, plus
  `xfer` for a cross-stage hop; ties to the lower stage).  `cost` is as for
  `timed_simulate`.  A pair lasts both costs minus `pair_gain`, and both
  results land at its end.  The last chunk is fused -- its F runs
  `cost(C-1, 'F') + cost(C-1, 'B')`, or split (W's of its own in the streams,
  `splits_last`) `cost(C-1, 'F') + cost(C-1, 'I')`, as
  `timed_simulate(fuse_last=True)` has it -- and its no-op B may be left out of
  the streams: it is re-inserted right after the item holding its F.

  The order is sorted by start time (ties to the lower stage), a valid
  topological order, and `fuse` holds the index of each pair's F.  A stage
  whose next item can never become ready raises `ScheduleError` -- which is
  how an illegal per-stage order is caught -- and so does a merged order
  that `validate` rejects (FIFO is not required here).  Pass `n_micro`, as
  every generator should: without it the count is read off the streams
  themselves, so a microbatch missing from every stage goes unnoticed.
  """
  C, S = len(place), max(place) + 1
  if len(streams) != S:
    raise ScheduleError(f"{len(streams)} streams for {S} stages")
  cost, modes = _cost_fn(cost, C), modes or {}
  items = []
  for s, st in enumerate(streams):
    row = []
    for it in st:
      ops = (tuple(it),) if isinstance(it[0], int) else tuple(tuple(a) for a in it)
      if len(ops) == 1 and ops[0][0] == C - 1 and ops[0][1] == "B":
        continue  # the fused last chunk's no-op backward; re-inserted below
      if len(ops) == 2:
        ops = tuple(sorted(ops, key=lambda a: a[1] != "F"))
      if len(ops) > 2 or (len(ops) == 2 and (ops[0][1] != "F" or ops[1][1] not in ("B", "I"))):
        raise ScheduleError(f'stage {s}: {"&".join(map(_fmt, ops))} is not an op or an F&B pair')
      for a in ops:
        if place[a[0]] != s:
          raise ScheduleError(f"{_fmt(a)} is in stage {s}'s stream but chunk {a[0]} is on " f"stage {place[a[0]]}")
      row.append(ops)
    items.append(row)

  done = {}

  def ready(c, k, m):
    hop = lambda c2: xfer if place[c2] != place[c] else 0.0
    if k == "F":
      if c == 0:
        return 0.0
      t = done.get((c - 1, "F", m))
      return None if t is None else t + hop(c - 1)
    if k == "W":
      return done.get((c, "BI", m))
    if k not in ("B", "I"):
      raise ScheduleError(f"unknown kind {k!r} in {(c, k, m)}")
    tf = done.get((c, "F", m))
    if tf is None or c == C - 1:
      return tf
    tb = done.get((c + 1, "BI", m))
    return None if tb is None else max(tf, tb + hop(c + 1))

  rest = "I" if splits_last([a for row in items for ops in row for a in ops], C) else "B"

  def dur(c, k, m):
    mode = modes.get((c, m), "r")
    if c == C - 1 and k == "F":
      return cost(c, "F", mode) + cost(c, rest, mode)
    return cost(c, k, mode)

  ptr, free = [0] * S, [0.0] * S
  started = []  # (start, stage, ops), in the order they start
  n_items = sum(len(r) for r in items)
  while len(started) < n_items:
    best = None
    for s in range(S):
      if ptr[s] == len(items[s]):
        continue
      ops = items[s][ptr[s]]
      ts = [ready(*a) for a in ops]
      if any(t is None for t in ts):
        continue
      t0 = max([free[s]] + ts)
      if best is None or t0 < best[0] - 1e-12:
        best = (t0, s, ops)
    if best is None:
      raise ScheduleError(
          "deadlock: no stage's next item can ever run -- "
          + ", ".join(
              f"s{s}: " + ("&".join(map(_fmt, items[s][ptr[s]])) if ptr[s] < len(items[s]) else "done") for s in range(S)
          )
      )
    t0, s, ops = best
    t1 = t0 + (dur(*ops[0]) if len(ops) == 1 else sum(dur(*a) for a in ops) - pair_gain)
    for c, k, m in ops:
      done[(c, k, m)] = t1
      if k in ("B", "I") or (k == "F" and c == C - 1):
        done[(c, "BI", m)] = t1
    started.append((t0, s, ops))
    free[s] = t1
    ptr[s] += 1
  order, fuse = [], set()
  for _, _, ops in started:
    if len(ops) == 2:
      fuse.add(len(order))
    order.extend(ops)
    order.extend((C - 1, "B", m) for c, k, m in ops if c == C - 1 and k == "F")
  if n_micro is None and order:
    n_micro = 1 + max(m for _, _, m in order)
  if n_micro is not None:
    validate(order, place, n_micro, modes=modes, fuse=fuse, fifo=False)
  return order, frozenset(fuse), max(free)


# ---------------------------------------------------------- measured costs
# Per-program medians in ms (`xprof costs` reads them off a capture): F0 / B0
# chunk 0 (which also embeds), F / B every other chunk, A the fused last chunk
# (forward, head and backward in one program, no recompute), xfer one stage
# boundary hop.  Only the ratios matter to the schedulers.
#   TPU7x:     the reference shape `--schedule timed` was first tuned on.
#   v6e:       PP2 x FSDP1 x EP2, 16 experts, gbs 16, M 8, V 2 -- one v6e-4 capture
#              (1613 ms/step) with the padded megablox tiles of MaxText's
#              production flags (one (512, D, 512) x3 for both weights: see
#              ds_ep.TILES).  Its update programs take another 15.9 / 19.2 ms
#              per stage.
#   v6e tiles: the same run with `--tiles v6e` (1319 ms/step), which unpads the
#              three gmm kernels with a 7168 contraction tile: ~3.2 ms off every
#              forward, ~15.6 off every backward and A; the updates are as above.
#              The driver also prices the default `--tiles clamped` with it
#              (unmeasured, but unpadded too: ds_ep_pipeline.default_costs).
#              Its stash programs (`--schedule stash`) are d-keys, G1's
#              progbench deltas against the remat programs: see
#              V6E_TILES_STASH_DELTAS at the end of the file.
# pair_gain: what an F&B pair program saves over its two halves run back to
# back (`merge` / `timed_simulate(pair_gain=)`; the driver's default for
# `--pair-gain`).  An F2&B0 pair under the padded tiles: 98.3 ms back to back,
# 96.4 as one program.  Not re-measured with the v6e tiles, nor for the other
# pair types (F1&B2, F2&B1, A3&B0 under V): the v6e-tiles table carries the
# same 1.9.
TPU7X_COSTS = dict(F0=18.5, F=16.5, B0=51.3, B=48.6, A=58.3, xfer=1.0)
V6E_COSTS = dict(F0=27.98, F=25.70, B0=71.65, B=68.70, A=86.98, xfer=1.0, pair_gain=1.9)
V6E_TILES_COSTS = dict(F0=24.75, F=22.44, B0=55.84, B=53.10, A=71.46, xfer=1.0, pair_gain=1.9)  # + stash d-keys, below


def cost_table(k, C, policy=None):
  """A cost table (ms per program) -> `cost(c, kind, mode='r')` for C chunks.

  Each cost is looked up most specific key first:

    forward             F{c}, F               (F0 is chunk 0's own key, so the
    remat backward      B{c}, B                old F0/F/B0/B tables mean what
    input gradient      I{c}, I                they always did)
    weight gradient     W{c}, W
    stash backward      Bs{c}:{p}, dBs{c}:{p}, Bs:{p}, dBs:{p}, Bs{c}, Bs   (policy p)
    stash forward       Fs{c}:{p}, dFs{c}:{p}, Fs:{p}, dFs:{p}, Fs{c}, Fs,
                        then the remat forward
    fused last chunk    A for its F; its B is then 0.  Split (W's of its own,
                        `splits_last`): its W is W{c}, else WA, and its F runs
                        AI (default A - W) -- F + I, so its I is AI - A

  A d-key is what the stash program costs over the remat one, so it follows
  the table's own F{c} / B{c}: a table that overrides those (`--costs`)
  reprices the stash programs with them, where an absolute stash key would go
  stale.  A mode is 'r' (remat), a stash policy name, or 's' for `policy`.  A
  missing key raises KeyError rather than guess; F and B of every chunk are
  looked up here, so a table short of one fails now, not inside a search.
  Keys it does not read (xfer, pair_gain, the `xprof costs` extras) are ignored.
  """

  def get(names, base=()):
    for n in names:
      if n in k:  # a d-key adds to the first of `base` the table has
        return float(k[n]) + (get(base) if n.startswith("d") else 0.0)
    raise KeyError(f"cost table has none of {names}")

  def cost(c, kind, mode="r"):
    if c == C - 1:
      if kind == "F":
        return get(["A"])
      if kind == "W":  # a split last chunk's deferred weight gradients
        return get([f"W{c}", "WA"])
      if kind == "I":  # ... and what its F, AI, takes off A
        return (get(["AI"]) if "AI" in k else get(["A"]) - get([f"W{c}", "WA"])) - get(["A"])
      return 0.0
    p = policy if mode == "s" else mode
    remat = [f"{kind}{c}", kind]
    if kind in ("F", "B") and mode != "r":
      s = f"{kind}s"
      names = [f"{s}{c}:{p}", f"d{s}{c}:{p}", f"{s}:{p}", f"d{s}:{p}"] if p else []
      names += [f"{s}{c}", s]
      if kind == "B":
        return get(names, remat)
      try:
        return get(names, remat)
      except KeyError:
        pass  # a stash forward costs what the remat one does unless measured
    if kind not in KINDS:
      raise ScheduleError(f"unknown kind {kind!r}")
    return get(remat)

  for c in range(C):
    cost(c, "F")
    cost(c, "B")
  return cost


# ------------------------------------------ interleaved 1F1B, F&B pairs, stash modes
def megatron_interleaved(n_stages, chunks_per_stage, n_micro, group=None, pair_stages=()):
  """Megatron-LM's interleaved 1F1B as per-stage streams, for `merge`.

  Circular placement, `placement(S, V)`: rank r owns model chunks j = 0..V-1,
  which are global chunks j*S + r.  Rank r runs the virtual microbatches
  k = 0..M*V-1 (`forward_backward_pipelining_with_interleaving`), in groups of
  G microbatches a chunk (`group`, Megatron's microbatch_group_size_per_vp_stage,
  default S):

    warmup     F(k)            k < W = min((S-r-1)*2 + (V-1)*G, M*V)
    steady     F(k+W), B(k)    for the remaining M*V - W
    cooldown   B(k)            for the last W

  where virtual microbatch k is model chunk (k % (G*V)) // G going forward
  (V-1 minus that going backward) and microbatch (k // (G*V)) * G + k % G.
  A group of S is just enough for a microbatch's forward to come round the S
  stages to rank 0's next chunk when nothing costs time between the stages: with
  G > S each chunk switch has G - S programs of slack against the boundary
  copies' latency, at the price of (V-1)(G-S) more warmup forwards held (their
  residuals), and of about G - S more boundary inputs waiting on the end ranks
  while they finish a group -- on rank 0 the activations for its next chunk, on
  rank S-1 the cotangents for its next chunk going backward -- which
  --offload-residuals, moving only the residuals to the host, leaves in HBM.
  The fused last chunk's B is left out; `merge` puts the no-op back.  Needs
  V >= 2 and G >= S, as Megatron does, and M a multiple of G: at G > S that is
  stricter than Megatron, which also runs a last group of S..G-1 microbatches.

  `pair_stages` (None: every rank; by default none) are the ranks whose
  steady state runs each F(k+W), B(k) as ONE program, an F&B pair: the
  forward of one microbatch beside the backward of another, so that XLA may
  overlap one's expert-parallel all-gather / reduce-scatter with the other's
  compute, as Megatron's combined 1F1B (overlap_moe_expert_parallel_comm)
  overlaps its all-to-alls.  Where B(k) is the fused last chunk's no-op its F
  runs alone, and the warmup and cooldown stay single ops, so every rank's
  sequence is the unpaired one: the merged order is as FIFO.  On rank S-1
  F(k+W) is the fused last chunk's exactly where B(k) is its no-op, so that
  program never pairs.  (Megatron warms up one forward more there so that
  F(k+W) and B(k) are never one microbatch's; they are only where B(k) is
  the last chunk's, which the fused program holds.)  A pair's F output
  leaves only when its B is done too, while the backward chain keeps each
  rank a step behind the next: a boundary copy's latency is no longer
  hidden, and in unit costs each steady-state step grows by about the
  copy's (S=8, V=2, M=128, G=8 at xfer 0.16: 833 against 794 unpaired),
  which the pair's overlap has to win back (`pair_gain`).
  """
  S, V, M = n_stages, chunks_per_stage, n_micro
  G = S if group is None else int(group)
  if V < 2:
    raise ValueError("interleaved 1F1B needs chunks_per_stage >= 2 (V=1 is plain 1F1B)")
  if G < S:
    raise ValueError(f"interleaved 1F1B needs a microbatch group of at least n_stages, got {G} and {S}")
  if M % G:
    what = "n_stages" if G == S else "its microbatch group"
    raise ValueError(f"interleaved 1F1B needs n_micro a multiple of {what}, got {M} and {G}")
  C, total = S * V, M * V

  def op(r, k, kind):
    j = (k % (G * V)) // G
    if kind == "B":
      j = V - 1 - j
    return (j * S + r, kind, (k // (G * V)) * G + k % G)

  paired = set(range(S)) if pair_stages is None else set(pair_stages)
  noop = lambda a: a[0] == C - 1 and a[1] == "B"
  streams = []
  for r in range(S):
    w = min((S - r - 1) * 2 + (V - 1) * G, total)
    st = [op(r, k, "F") for k in range(w)]
    for k in range(total - w):
      f, b = op(r, k + w, "F"), op(r, k, "B")
      st += [f] if noop(b) else [(f, b)] if r in paired else [f, b]
    st += [b for b in (op(r, k, "B") for k in range(total - w, total)) if not noop(b)]
    streams.append(st)
  return streams


def stash_modes(order, place, stash_bytes, budget_bytes, caps=None, *, policy=None):
  """Stash modes for a FIXED order under a per-stage byte budget.  -> {(c, m): policy}

  The post-pass for schedules that do not pick modes as they go (Megatron's
  interleaved 1F1B, its pairs): a (c, m) of a non-last chunk may stash -- keep its residual set
  from its F to its B or I and skip the recompute -- where it fits:

    * the live sets of every stage, priced as `peak_bytes` prices them with
      `stash_bytes`, stay within `budget_bytes` (a number, or one per stage),
      so `peak_bytes(order, place, modes, stash_bytes) <= budget`;
    * chunk c holds at most `caps[c]` stash sets at a time (`caps` a
      per-chunk sequence or {chunk: n}; None or a chunk left out: no cap).

  Every stashed set saves one recompute, so the aim is as many as fit.  The
  sets go in the order their stage frees them -- by the position of their B
  or I among the stage's ops, earliest first -- and each is taken if it fits
  over its whole life (else it stays remat).  This is the interval-scheduling
  greedy: of the sets still to place, the one freed first leaves the most
  room for the rest, so where every set of a stage adds the same bytes and
  no cap applies it stashes as many sets as any assignment can.  With
  per-chunk sizes or caps it is a heuristic.  On a FIFO order a chunk's B's
  run in microbatch order, so its sets are handed out in that order too.
  Deterministic.

  `stash_bytes` is as for `peak_bytes`: a {mode: bytes or per-chunk bytes}
  mapping (leave 'r' out to budget the stash sets alone, or name it to budget
  every live set) or `size(c, mode)`.  `policy` is the stash policy handed
  out; by default the one mode besides 'r' that the mapping names.  Raises
  ValueError if the remat sets alone exceed the budget.
  """
  C, S = len(place), max(place) + 1
  size = _set_size(stash_bytes)
  if policy is None:
    names = [p for p in stash_bytes if p != "r"] if isinstance(stash_bytes, dict) else []
    if len(names) != 1:
      raise ValueError(f"pass policy=: cannot tell the stash policy from {stash_bytes!r}")
    policy = names[0]
  if policy == "r":
    raise ValueError("'r' is remat, not a stash policy")
  budget = list(budget_bytes) if isinstance(budget_bytes, (list, tuple)) else [budget_bytes] * S
  if len(budget) != S:
    raise ValueError(f"budget_bytes must have {S} entries, got {len(budget)}")
  cap = dict(enumerate(caps)) if isinstance(caps, (list, tuple)) else dict(caps or {})
  cap = {c: n for c, n in cap.items() if n is not None}
  seq = [[] for _ in range(S)]  # each stage's own sequence: (order index, op)
  for j, a in enumerate(order):
    seq[place[a[0]]].append((j, tuple(a)))
  live = [[0] * len(q) for q in seq]  # bytes live at each position of a stage
  held = [[0] * len(seq[place[c]]) for c in range(C)]  # chunk c's stash sets there
  sets, at = [], {}
  for s, q in enumerate(seq):
    for i, (j, (c, k, m)) in enumerate(q):
      if c == C - 1:
        continue  # the fused last chunk keeps nothing
      if k == "F":
        at[(c, m)] = i
      elif k in ("B", "I"):
        if (c, m) not in at:
          raise ScheduleError(f"{_fmt((c, k, m))} at order[{j}] runs before {c}F{m}")
        f = at.pop((c, m))
        sets.append((s, i, c, m, f))  # it lives over positions [f, i)
        for t in range(f, i):
          live[s][t] += size(c, "r")
  if at:
    c, m = min(at)
    raise ScheduleError(f"{c}F{m} has no B or I")
  for s in range(S):
    if live[s] and max(live[s]) > budget[s]:
      raise ValueError(f"stage {s}: the remat sets alone need {max(live[s])}, over the budget " f"{budget[s]}")
  modes = {}
  for _, i, c, m, f in sorted(sets):  # stage by stage, the earliest freed first
    s, extra = place[c], size(c, policy) - size(c, "r")
    if max(live[s][f:i]) + extra > budget[s] or max(held[c][f:i]) >= cap.get(c, float("inf")):
      continue
    for t in range(f, i):
      live[s][t] += extra
      held[c][t] += 1
    modes[(c, m)] = policy
  return modes


# Stash programs with `--tiles v6e`, as `cost_table` d-keys: what the stash
# variant of a chunk's program costs over its remat one, so they follow any
# `--costs` that overrides F / B.  G1's progbench medians, `fwd_s` / `bwda_s`
# against `fwd` / `bwda` of the same chunk in the same session: chunk 2 as
# dFs:{p} / dBs:{p}, which every chunk but 0 takes, and chunk 0 as dFs0:{p} /
# dBs0:{p} (it also embeds).  attn was timed on chunk 2 alone (22.76 -> 24.62
# ms forward, 53.47 -> 47.27 backward), so chunk 0's is an ESTIMATE: chunk
# 2's deltas scaled as attn_min's are between the two chunks (x 1.75 / 1.97
# forward, x 4.14 / 5.56 backward).  `all` saves less than `lite_y`:
# measured, not a typo.  Absolute Fs / Bs keys a table names win.
# {policy: ((forward ms, backward ms) chunk 2, ... chunk 0)}:
V6E_TILES_STASH_DELTAS = {
    "attn": ((1.86, -6.20), (1.65, -4.62)),
    "attn_min": ((1.97, -5.56), (1.75, -4.14)),
    "lite": ((2.73, -9.58), (2.18, -9.16)),
    "lite_y": ((4.90, -12.15), (4.05, -11.00)),
    "all": ((4.56, -10.01), (4.20, -4.74)),
}
V6E_TILES_COSTS.update(
    {
        f"d{kind}s{c}:{p}": d
        for p, per in V6E_TILES_STASH_DELTAS.items()
        for c, deltas in zip(("", "0"), per)
        if deltas
        for kind, d in zip("FB", deltas)
    }
)


# ------------------------------------------------------- zero-bubble schedules
# Zero Bubble Pipeline Parallelism (Qi et al., ICLR 2024) splits a chunk's backward into I, the input gradient (with
# the recompute under remat), and W, the weight gradient, and moves the W's into the bubbles: nothing waits on a W
# but its own chunk's next gradient write.  ZB-H1 keeps 1F1B's residual sets (stage s holds at most S - s), ZB-H2
# about twice them (2 (S - s) - 1), whose deeper warm-up fills the start of the pipe.  The generators below are
# V = 1 schedules (one chunk a stage, place[c] = c) under the driver's rules: chunk 0 has no input gradient to send,
# so it runs a whole B (all W to the schedule, which may move it like one), and the last chunk is one fused forward
# + backward program, its F carrying its B.  The middle chunks split -- and with `split_last` the last one too, its F
# the forward, head, loss and input gradient (AI), a W of its own its layers' weight gradients -- so at S = 2 without
# it there is nothing to split and ZB-H1 is 1F1B.  On top of the residual sets a stage holds its pending W buffers
# (I done, W not), each what W reads from I -- at DeepSeek-V3 sizes GiBs a layer, tens of residual sets -- so that,
# more than the residual sets, is the memory that binds: `zb_schedule`'s `wbuf_cap`.  Every chunk's I's and W's run
# in microbatch order (FIFO), so its gradient sum is any other schedule's, bit for bit.

# W's share of a split chunk's backward, for a cost table without I / W keys (`zb_costs`): the weight-gradient
# matmuls are 0.29-0.33 of a remat backward's XLA flops on DeepSeek-V3 chunks (I re-runs the forward and computes
# dx).  Measured times differ -- I's attention core and collectives have no W counterpart -- so give I / W keys where
# they were measured.
ZB_W_SHARE = 0.3
# A split last chunk's W share of its fused program A (`zb_costs` with `last`): its layers' weight-gradient matmuls,
# about what a middle chunk's W is -- 0.3 of its B, on TPU7X_COSTS 14.6 of B's 48.6 ms, a quarter of A's 58.3 -- as the
# head, the decoder norm and the loss stay in its F (AI).  0.19-0.32 of A's XLA flops at TINY widths on CPU (bf16 with
# and without remat_fused_last, fp32 on XLA's ops).
ZB_W_SHARE_LAST = 0.25


def zb_costs(table, n_chunks, w_share=None, *, last=False, w_share_last=None):
  """`table` (`cost_table` keys) with an I and a W cost for every chunk that can split -- the middle ones, 1 .. C-2
  -- where it has none, so that I + W = B: W = w_share x B (default ZB_W_SHARE) and I = B - W, from B{c}, else B; a
  measured I alone gives W = B - I, a measured W alone I = B - W.  `last`: the fused last chunk split too
  (`splits_last`): its W, WA = w_share_last x A (default ZB_W_SHARE_LAST), where the table has neither W{C-1} nor WA
  -- or A - AI, a measured AI (its F, `cost_table`) alone.  The F, B and A costs, and so every order priced without I
  / W, stay as they are.  Refuses (ValueError) a negative I, W or AI, and a measured I or W alone above its chunk's B
  -- or the last chunk's W or AI alone above A: the other would come out negative (give both, as measured)."""
  share = ZB_W_SHARE if w_share is None else float(w_share)
  share_last = ZB_W_SHARE_LAST if w_share_last is None else float(w_share_last)
  for name, x in (("w_share", share), ("w_share_last", share_last)):
    if not 0.0 <= x <= 1.0:
      raise ValueError(f"{name} must be within [0, 1], got {x}")
  out = dict(table)
  remat = cost_table(table, n_chunks)  # also checks that the table prices every chunk's F and B

  def get(kind, c):
    return float(out[f"{kind}{c}"] if f"{kind}{c}" in out else out[kind])

  def has(kind, c):
    return f"{kind}{c}" in out or kind in out

  bad = []
  for c in range(1, n_chunks - 1):
    b = remat(c, "B")
    bad += [f"{kind}{c} = {get(kind, c)} < 0" for kind in "IW" if has(kind, c) and get(kind, c) < 0]
    bad += [
        f"{kind}{c} = {get(kind, c)} > B{c} = {b}, which leaves {other}{c} = B - {kind} negative"
        for kind, other in (("I", "W"), ("W", "I"))
        if has(kind, c) and not has(other, c) and get(kind, c) > b
    ]
  L = n_chunks - 1
  w_key = next((k for k in (f"W{L}", "WA") if k in out), None)  # the split last chunk's W, if measured
  if last:
    a = remat(L, "F")
    given = {name: float(out[key]) for name, key in (("W", w_key), ("AI", "AI")) if key in out}
    bad += [f"the last chunk's {name} = {x} < 0" for name, x in given.items() if x < 0]
    if len(given) == 1 and max(given.values()) > a:
      ((name, x),) = given.items()
      other = "AI" if name == "W" else "W"
      bad.append(f"the last chunk's {name} = {x} > A = {a}, which leaves its {other} = A - {name} negative")
  if bad:
    raise ValueError(f"zb_costs: {'; '.join(bad)} (give a chunk's I and W both, as measured, or neither)")
  for c in range(1, n_chunks - 1):
    b = remat(c, "B")
    has_c = {kind: has(kind, c) for kind in "IW"}
    if not has_c["W"]:
      out[f"W{c}"] = round(b - get("I", c) if has_c["I"] else share * b, 4)
    if not has_c["I"]:
      out[f"I{c}"] = round(b - get("W", c), 4)
  if last and w_key is None:
    a = remat(L, "F")
    out["WA"] = round(a - float(out["AI"]) if "AI" in out else share_last * a, 4)
  return out


def _zb_check(n_stages, n_micro):
  if n_stages < 2 or n_micro < 1:
    raise ValueError(f"a zero-bubble schedule needs >= 2 stages and >= 1 microbatch, got {n_stages} and {n_micro}")


def _zb_op(c, k, m, C, split_last=False):
  """One action of a zero-bubble stream, mapped onto the chunks that do not split: chunk 0's I is its whole B (and
  its W none), the fused last chunk's I and W are part of its F (none) -- its I only, where it splits too
  (`split_last`: a W of its own)."""
  if c == C - 1:
    return [(c, k, m)] if k == "F" or (k == "W" and split_last) else []
  if c == 0:
    return [] if k == "W" else [(c, "B" if k == "I" else k, m)]
  return [(c, k, m)]


def zb_h1(n_stages, n_micro, w_lag=None, *, split_last=False):
  """ZB-H1 as per-stage streams (place[c] = c), for `merge`: 1F1B's warm-up of S - 1 - r forwards on stage r and its
  steady state, each backward an I, and stage r's W's `w_lag[r]` microbatches behind its I's (default r) -- the
  stages near the front run each W soon after its I, those further down keep their I's back to back to feed the
  stages upstream and run the W's they owe in the drain, while the stages upstream are still busy.  Stage r holds at
  most S - r residual sets (F to I), as 1F1B does, and min(w_lag[r] + 1, M) pending W buffers (`peak_wbufs`) --
  `split_last`: the last stage too, its F's running the input gradient (`splits_last`) and its W's lagging them.
  Cost-agnostic; in the paper's setting (T_F = T_I = T_W, every chunk split) its bubble is a third of 1F1B's."""
  _zb_check(n_stages, n_micro)
  S, M = n_stages, n_micro
  lag = list(range(S)) if w_lag is None else _per(w_lag, S, "w_lag")
  op = functools.partial(_zb_op, C=S, split_last=split_last)
  streams = []
  for r in range(S):
    warm = min(S - 1 - r, M)
    st, w_next = [], 0
    for m in range(warm):
      st += op(r, "F", m)
    for k in range(M):
      if warm + k < M:
        st += op(r, "F", warm + k)
      st += op(r, "I", k)
      if k >= lag[r]:
        st += op(r, "W", w_next)
        w_next += 1
    while w_next < M:
      st += op(r, "W", w_next)
      w_next += 1
    streams.append(st)
  return streams


def zb_h2(n_stages, n_micro, w_lag=None, *, split_last=False):
  """ZB-H2 as per-stage streams (place[c] = c), for `merge`: stage r warms up with 2 (S - r) - 1 forwards -- enough to
  fill the bubble before its first I arrives at unit costs -- then runs I, F, W, its W's `w_lag[r]` microbatches
  behind its I's (default 2 r), and the deferred W's at the tail, where they fill the drain: 1F1B's trapezoid
  becomes a parallelogram.  Stage r holds at most 2 (S - r) - 1 residual sets and min(w_lag[r] + 1, M) pending W
  buffers -- `split_last`: the last stage min(w_lag + 2, M), its warm-up forward's W buffer besides (each F runs its
  I).  Cost-agnostic; in the paper's setting (T_F = T_I = T_W, every chunk split) its bubble is the warm-up ramp
  alone, (S - 1) T_F."""
  _zb_check(n_stages, n_micro)
  S, M = n_stages, n_micro
  lag = [2 * r for r in range(S)] if w_lag is None else _per(w_lag, S, "w_lag")
  op = functools.partial(_zb_op, C=S, split_last=split_last)
  streams = []
  for r in range(S):
    warm = min(2 * (S - r) - 1, M)
    st, f_next, w_next = [], 0, 0
    for _ in range(warm):
      st += op(r, "F", f_next)
      f_next += 1
    for k in range(M):
      st += op(r, "I", k)
      if f_next < M:
        st += op(r, "F", f_next)
        f_next += 1
      if k >= lag[r]:
        st += op(r, "W", w_next)
        w_next += 1
    while w_next < M:
      st += op(r, "W", w_next)
      w_next += 1
    streams.append(st)
  return streams


def zb_caps(n_stages, mem):
  """Per-stage memory limits in microbatches of ZB-H1 (mem 'h1': 1F1B's, S - s) and ZB-H2 ('h2': 2 (S - s) - 1).
  `zb_search` caps by them either the residual sets (F to I or B) or the microbatches in flight (F to W or B,
  `peak_inflight`), which bounds the residual sets too."""
  S = n_stages
  if mem == "h1":
    return [S - s for s in range(S)]
  if mem == "h2":
    return [2 * (S - s) - 1 for s in range(S)]
  raise ValueError(f"mem must be 'h1' or 'h2', got {mem!r}")


def zb_greedy(
    n_chunks,
    n_micro,
    place,
    cost,
    xfer=0.0,
    *,
    split=None,
    inflight_cap=None,
    res_cap=None,
    wbuf_cap=None,
    prio=("BI", "F", "W"),
    f_key=None,
    b_key=None,
    w_slack=0.0,
    slack_kinds=("I", "B"),
    w_first=False,
    fuse_last=True,
    split_last=False,
):
  """Event-driven list scheduler with a split backward -> (per-stage streams for `merge`, makespan).

  `timed_greedy`'s engine: whenever a stage goes idle it starts the best op READY at that instant (its dependencies
  done, plus `xfer` for a cross-stage hop), else waits for the earliest.  `split`: the chunks whose backward runs as
  I then W (default the middle ones; a fused last chunk never splits).  Per-stage caps, each a number or one per
  stage (None: no cap): `inflight_cap`, microbatches between their F and their W or B (`peak_inflight` -- a residual
  set, then a pending W buffer: the paper's memory count); `res_cap`, residual sets (F to I or B); `wbuf_cap`,
  pending W buffers (an I waits for a W).  `prio` ranks the kinds ready at the same instant ('BI': an I or an unsplit
  B, which unblocks the stage upstream); `f_key` / `b_key` rank a stage's chunks within a kind.  `w_first`: when no
  forward is ready but the stage has forwards left (its caps or its inputs hold them back), a ready W goes first --
  it frees the in-flight slot a forward may wait on.  `w_slack`: rather than start a W, the stage takes an op of
  `slack_kinds` that becomes ready within w_slack x cost(W) -- it would wait behind the W.

  Each chunk's F's, I's (or B's) and W's run in microbatch order, a W after its I; the fused last chunk's F costs its
  F + B (its no-op B is `merge`'s to re-insert) -- `split_last`, the fused last chunk split too: its F costs its F +
  I, leaves a W buffer (under `wbuf_cap`, not the in-flight or residual caps: it holds no residual set) and its W's
  run later, each after its F.  `cost` is as for `timed_simulate`.  Deterministic.  Raises RuntimeError if the caps
  deadlock (a split chunk needs `wbuf_cap` >= 1)."""
  C, M = n_chunks, n_micro
  S = max(place) + 1
  cost = _cost_fn(cost, C)
  L = C - 1 if fuse_last else -1  # the fused last chunk, if any
  split = set(range(1, C - 1)) if split is None else set(split) - {L}
  last_w = split_last and L >= 0  # the fused last chunk's W's
  inf_cap, r_cap, w_cap = (
      None if x is None else _per(x, S, what)
      for x, what in ((inflight_cap, "inflight_cap"), (res_cap, "res_cap"), (wbuf_cap, "wbuf_cap"))
  )
  own = [[c for c in range(C) if place[c] == d] for d in range(S)]
  f_key = f_key or (lambda c, m: (m, c))
  b_key = b_key or (lambda c, m: (m, -c))
  rank_of = {k: i for i, k in enumerate(prio)}
  done = {}
  nf, nb, nw = [0] * C, [0] * C, [0] * C
  free = [0.0] * S
  inflight, wbuf, res = [0] * S, [0] * S, [0] * S
  streams = [[] for _ in range(S)]

  def hop(a, b):
    return xfer if place[a] != place[b] else 0.0

  def ready_f(c, m):
    if c == 0:
      return 0.0
    t = done.get((c - 1, "F", m))
    return None if t is None else t + hop(c - 1, c)

  def ready_b(c, m):
    tf = done.get((c, "F", m))
    if tf is None or c == C - 1:
      return tf
    tb = done.get((c + 1, "BI", m))
    return None if tb is None else max(tf, tb + hop(c + 1, c))

  def rank(x):
    _, k, c, m = x
    return (rank_of["BI" if k in ("B", "I") else k], f_key(c, m) if k == "F" else b_key(c, m))

  def candidates(d):
    """(start, kind, chunk, microbatch) of every op stage d may start next, under its caps."""
    cands = []
    cap_ok = (inf_cap is None or inflight[d] < inf_cap[d]) and (r_cap is None or res[d] < r_cap[d])
    w_ok = w_cap is None or wbuf[d] < w_cap[d]
    for c in own[d]:
      ok = (w_ok or not last_w) if c == L else cap_ok
      r = ready_f(c, nf[c]) if nf[c] < M and ok else None
      if r is not None:
        cands.append((max(r, free[d]), "F", c, nf[c]))
      k = "I" if c in split else "B"
      if c != L and nb[c] < M and (k == "B" or w_ok):
        r = ready_b(c, nb[c])
        if r is not None:
          cands.append((max(r, free[d]), k, c, nb[c]))
      if c in split and nw[c] < nb[c]:
        cands.append((max(done[(c, "I", nw[c])], free[d]), "W", c, nw[c]))
      if c == L and last_w and nw[c] < nf[c]:
        cands.append((max(done[(c, "F", nw[c])], free[d]), "W", c, nw[c]))
    return cands

  total = sum((2 * M if last_w else M) if c == L else (3 * M if c in split else 2 * M) for c in range(C))
  for _ in range(total):
    best, best_t = None, float("inf")
    for d in range(S):
      cands = candidates(d)
      if not cands:
        continue
      t0 = min(x[0] for x in cands)
      now = [x for x in cands if x[0] <= t0 + 1e-9]
      pick = min(now, key=rank)
      if w_first and pick[1] != "W" and all(x[1] != "F" for x in now) and any(nf[c] < M for c in own[d]):
        pick = min([x for x in now if x[1] == "W"] or [pick], key=rank)
      if pick[1] == "W" and w_slack > 0:
        soon = [x for x in cands if x[1] in slack_kinds and x[0] < t0 + w_slack * cost(pick[2], "W") - 1e-9]
        if soon:
          pick = min(soon, key=lambda x: x[0])
      if pick[0] < best_t - 1e-9:
        best, best_t = (*pick, d), pick[0]
    if best is None:
      raise RuntimeError(f"zb_greedy deadlock: nf={nf} nb={nb} nw={nw} inflight={inflight} wbuf={wbuf}")
    t, k, c, m, d = best
    t1 = t + (cost(c, "F") + cost(c, "I" if last_w else "B") if (c == L and k == "F") else cost(c, k))
    free[d] = t1
    done[(c, k, m)] = t1
    streams[d].append((c, k, m))
    if k == "F":
      nf[c] += 1
      if c == L:  # its backward ran too (split: its input gradient, its W buffer left for its W)
        done[(c, "B", m)] = done[(c, "BI", m)] = t1
        wbuf[d] += last_w
      else:
        inflight[d] += 1
        res[d] += 1
    elif k == "W":
      nw[c] += 1
      wbuf[d] -= 1
      inflight[d] -= c != L
    else:
      nb[c] += 1
      res[d] -= 1
      done[(c, "BI", m)] = t1
      if k == "I":
        wbuf[d] += 1
      else:
        inflight[d] -= 1
  return streams, max(free)


# `zb_search`'s rules: which op a stage takes when several are ready (the kinds' rank, `w_first`) and when a W may
# start (`w_slack`).  The first is the plain greedy.
ZB_POLICIES = (
    ("greedy", {}),
    ("w_slack 0.5", {"w_slack": 0.5}),
    ("w_slack 1", {"w_slack": 1.0}),
    ("w_slack 1 +F", {"w_slack": 1.0, "slack_kinds": ("I", "B", "F")}),
    ("w_slack 0.5 +F", {"w_slack": 0.5, "slack_kinds": ("I", "B", "F")}),
    ("W before F", {"prio": ("BI", "W", "F")}),
    ("F first", {"prio": ("F", "BI", "W")}),
    ("w_first", {"w_first": True}),
    ("F first, w_first", {"prio": ("F", "BI", "W"), "w_first": True}),
    ("W before F, w_first", {"prio": ("BI", "W", "F"), "w_first": True}),
)
# ... each run under both memory forms: 'inflight' caps the microbatches in flight a stage (F to W or B: a pending W
# buffer counts, the paper's notion) by `zb_caps`; 'res' caps the residual sets by `zb_caps` and the microbatches in
# flight by S (ZB-H1) or 2 S - 1 (ZB-H2) on every stage.
ZB_MEMORY_FORMS = ("inflight", "res")


def _zb_mem_caps(n_stages, mem, form):
  S = n_stages
  if form == "inflight":
    return {"inflight_cap": zb_caps(S, mem)}
  if form == "res":
    return {"res_cap": zb_caps(S, mem), "inflight_cap": S if mem == "h1" else 2 * S - 1}
  raise ValueError(f"memory form must be one of {ZB_MEMORY_FORMS}, got {form!r}")


def _zb_fastest(cands):
  """(makespan, W buffers, item) in tie order -> (item, makespan) of the fastest; makespans within a relative 1e-9
  tie, and a tie goes to fewer W buffers, then to the earlier.  (None, inf) if there are none."""
  best, best_mk, best_wb = None, float("inf"), float("inf")
  for mk, wb, item in cands:
    tol = 1e-9 * max(1.0, abs(mk))
    if mk < best_mk - tol or (abs(mk - best_mk) <= tol and wb < best_wb):
      best, best_mk, best_wb = item, mk, wb
  return best, best_mk


def zb_search(
    n_stages,
    n_micro,
    cost,
    xfer=0.0,
    *,
    mem="h1",
    wbuf_cap=None,
    policies=ZB_POLICIES,
    forms=ZB_MEMORY_FORMS,
    split_last=False,
):
  """The fastest `zb_greedy` on S stages of one chunk each (place[c] = c), the middle chunks split (`split_last`: the
  fused last chunk too), over `policies` each under every memory form of `forms` at ZB-H1's (mem 'h1') or ZB-H2's
  ('h2') `zb_caps`, with at most `wbuf_cap` pending W buffers a stage (None: no cap) -> (streams, (mem, form,
  policy), makespan).  Ties go to fewer W buffers, then to the earlier form and policy."""
  _zb_check(n_stages, n_micro)
  S, place = n_stages, list(range(n_stages))

  def runs():
    for form in forms:
      caps = _zb_mem_caps(S, mem, form)
      for name, pol in policies:
        try:
          streams, mk = zb_greedy(S, n_micro, place, cost, xfer, wbuf_cap=wbuf_cap, split_last=split_last, **caps, **pol)
        except RuntimeError:
          continue  # these caps deadlock under this policy
        yield mk, max(peak_wbufs([a for st in streams for a in st], place)), (streams, (mem, form, name))

  best, mk = _zb_fastest(runs())
  if best is None:
    raise RuntimeError(f"zb_search: every policy deadlocks under mem={mem!r}, wbuf_cap={wbuf_cap!r}")
  return best[0], best[1], mk


def zb_schedule(n_stages, n_micro, cost, xfer=0.0, *, mem="h1", wbuf_cap=1, split_last=False):
  """ZB-H1 (mem 'h1') or ZB-H2 ('h2') on S stages of one chunk each (place[c] = c) for the driver -> (order, tag,
  makespan).  The fastest per `merge`, with `cost` (as for `timed_simulate`, I and W priced: `zb_costs`), of

    * 1F1B with each split B run as I and then W at once: 1F1B's order, timeline and memory, so the result is never
      slower than 1F1B where I + W <= B -- and at S = 2, where no chunk splits, it is 1F1B's order itself;
    * the handcrafted streams `zb_h1` (for ZB-H2 also `zb_h2`), their W lag cut to `wbuf_cap` - 1;
    * `zb_search` under ZB-H1's memory (for ZB-H2 also under its own): every rule of ZB_POLICIES under each of the
      ZB_MEMORY_FORMS.

  `split_last`: the fused last chunk splits too, in every candidate (`splits_last`: its F runs the forward, the head,
  the loss and the input gradient, AI, and its W's its layers' weight gradients) -- in 1F1B's, each W right after its
  F (and the no-op B): 1F1B's timeline with the input gradient upstream sooner, so never slower than 1F1B where AI +
  W <= A (and, at three stages or more, each middle chunk's I + W <= B), and at S = 2 faster, the last chunk the one
  that splits.

  Ties go to fewer pending W buffers, then to the earlier candidate above.  ZB-H2 tries every ZB-H1 candidate, so it
  is never the slower.  Memory: stage s holds at most S - s residual sets (ZB-H1, 1F1B's) or 2 (S - s) - 1 (ZB-H2),
  and at most `wbuf_cap` pending W buffers (None: no cap -- the streams then hold up to min(r + 1, M) (ZB-H1) or
  min(2 r + 1, M) (ZB-H2) on stage r, the searches up to S or 2 S - 1; a candidate over the cap is dropped).  A
  looser cap is not always faster: list scheduling under different caps can land either side (by 0.15% in 2 of 216
  simulated cases).  The order passes `validate`, FIFO included: every chunk's forwards and gradient writes run in
  microbatch order, so its fp32 accumulation is any other schedule's.  Deterministic."""
  _zb_check(n_stages, n_micro)
  if mem not in ("h1", "h2"):
    raise ValueError(f"mem must be 'h1' or 'h2', got {mem!r}")
  if wbuf_cap is not None and (isinstance(wbuf_cap, bool) or int(wbuf_cap) != wbuf_cap or wbuf_cap < 1):
    raise ValueError(f"wbuf_cap must be None (no cap) or an int >= 1 -- an I leaves a W buffer -- got {wbuf_cap!r}")
  S, M = n_stages, n_micro
  place = list(range(S))
  cost = _cost_fn(cost, S)
  mems = ("h1", "h2") if mem == "h2" else ("h1",)
  cands = []  # (order, tag), in the order ties go to
  base = []
  for c, k, m in one_f_one_b(S, M, place):
    if k == "B" and 0 < c < S - 1:
      base += [(c, "I", m), (c, "W", m)]
    else:
      base += [(c, k, m)] + ([(c, "W", m)] if split_last and k == "B" and c == S - 1 else [])
  cands.append((base, ("1f1b", "I+W")))
  for mm in mems:
    lag = None
    if wbuf_cap is not None:
      # a lag of d holds min(d + 1, M) W buffers -- ZB-H2's split last stage min(d + 2, M), its warm-up forward's too
      extra = [int(mm == "h2" and split_last and r == S - 1) for r in range(S)]
      lag = [max(0, min(r if mm == "h1" else 2 * r, int(wbuf_cap) - 1 - extra[r])) for r in range(S)]
    streams = (zb_h1 if mm == "h1" else zb_h2)(S, M, w_lag=lag, split_last=split_last)
    cands.append((merge(streams, place, cost, xfer, n_micro=M)[0], (f"zb_{mm}", "streams")))
  for mm in mems:
    streams, tag, _ = zb_search(S, M, cost, xfer, mem=mm, wbuf_cap=wbuf_cap, split_last=split_last)
    cands.append((merge(streams, place, cost, xfer, n_micro=M)[0], ("search",) + tag))
  (order, tag), mk = _zb_fastest(
      (timed_simulate(o, place, cost, xfer, fuse_last=True)[0], wb, (o, tag))
      for o, tag in cands
      for wb in (max(peak_wbufs(o, place)),)
      if wbuf_cap is None or wb <= wbuf_cap
  )
  validate(order, place, M)
  return order, tag, mk
