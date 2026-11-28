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

"""Tests for maxtext.experimental.mpmd.jax.schedule.  Pure Python -- no JAX, no accelerator."""

import itertools
import unittest

import pytest

from maxtext.experimental.mpmd.jax import schedule as sc

GRID = [(s, v, m) for s in (1, 2, 3, 4, 8) for v in (1, 2, 3, 4) for m in (1, 2, 3, 5, 8, 16)]


def assert_legal(t, order, n_chunks, n_micro):
  """Every action once, and never before what it depends on."""
  fdone, bdone, seen = set(), set(), set()
  for c, kind, m in order:
    t.assertNotIn((c, kind, m), seen, f"duplicate {(c, kind, m)}")
    seen.add((c, kind, m))
    if kind == "F":
      if c:
        t.assertIn((c - 1, m), fdone, f"F{(c, m)} before its upstream forward")
      fdone.add((c, m))
    else:
      t.assertIn((c, m), fdone, f"B{(c, m)} before its own forward")
      if c + 1 < n_chunks:
        t.assertIn((c + 1, m), bdone, f"B{(c, m)} before its downstream backward")
      bdone.add((c, m))
  t.assertEqual(len(seen), 2 * n_chunks * n_micro)


class PlacementTest(unittest.TestCase):

  def test_shapes(self):
    self.assertEqual(sc.placement(4, 1), [0, 1, 2, 3])
    self.assertEqual(sc.placement(4, 2), [0, 1, 2, 3, 0, 1, 2, 3])
    self.assertEqual(sc.placement(4, 2, "v"), [0, 1, 2, 3, 3, 2, 1, 0])
    with self.assertRaises(ValueError):
      sc.placement(4, 3, "v")
    with self.assertRaises(ValueError):
      sc.placement(4, 1, "nope")


class LegalityTest(unittest.TestCase):

  def test_all_generators_over_grid(self):
    n = 0
    for s, v, m in GRID:
      shapes = ["circular"] + (["v"] if v == 2 else [])
      for shape in shapes:
        place, c = sc.placement(s, v, shape), s * v
        for name, order in (
            ("gpipe", sc.gpipe(c, m, place)),
            ("1f1b", sc.one_f_one_b(c, m, place)),
            ("search", sc.search(c, m, place)[0]),
            ("breadth", sc.breadth_first(c, m, place)),
            ("search/mem1", sc.search(c, m, place, max_residual_sets=1)[0]),
        ):
          with self.subTest(s=s, v=v, shape=shape, m=m, sched=name):
            assert_legal(self, order, c, m)
          n += 1
    self.assertGreater(n, 500)

  def test_eager_bwd_is_also_legal(self):
    place = sc.placement(4, 2)
    assert_legal(self, sc.greedy(8, 8, place, eager_bwd=True), 8, 8)

  def test_bad_cap_length(self):
    with self.assertRaises(ValueError):
      sc.greedy(4, 4, sc.placement(4), cap=[1, 2])


class MemoryTest(unittest.TestCase):

  def test_1f1b_peak_is_depth_not_microbatches(self):
    place = sc.placement(4)
    for m in (8, 16, 64):
      peak = sc.peak_residuals(sc.one_f_one_b(4, m, place), 4)
      self.assertEqual(peak, [4, 3, 2, 0], f"M={m} should not change 1F1B memory")

  def test_gpipe_peak_grows_with_microbatches(self):
    place = sc.placement(4)
    self.assertEqual(sc.peak_residuals(sc.gpipe(4, 16, place), 4), [16, 16, 16, 0])

  def test_last_chunk_stores_nothing(self):
    """Its forward and backward are fused, so no residual is ever held."""
    for s, v, m in ((4, 1, 8), (4, 2, 8), (8, 1, 16)):
      place, c = sc.placement(s, v), s * v
      for order in (sc.gpipe(c, m, place), sc.one_f_one_b(c, m, place)):
        self.assertEqual(sc.peak_residuals(order, c)[-1], 0)

  def test_budget_is_respected(self):
    place = sc.placement(4, 2)
    for budget in (1, 2, 4, 8):
      order, _, _ = sc.search(8, 16, place, max_residual_sets=budget)
      self.assertLessEqual(max(sc.peak_residuals(order, 8)), budget)

  def test_budget_below_one_rejected(self):
    with self.assertRaises(ValueError):
      sc.search(4, 4, sc.placement(4), max_residual_sets=0)

  def test_residuals_are_released_by_i_not_w(self):
    self.assertEqual(sc.peak_residuals(SPLIT, 3), sc.peak_residuals(OK, 3))
    # Were W also a release, the set count would dip below what is live.
    o = [(1, "F", 0), (1, "I", 0), (1, "W", 0), (1, "F", 1), (1, "F", 2)]
    self.assertEqual(sc.peak_residuals(o, 3), [0, 2, 0])

  def test_peak_wbufs(self):
    self.assertEqual(sc.peak_wbufs(SPLIT, [0, 1, 2]), [0, 2, 0])
    self.assertEqual(sc.peak_wbufs(OK, [0, 1, 2]), [0, 0, 0])

  def test_peak_bytes_by_stage_and_mode(self):
    place, modes = [0, 1, 2], {(1, 0): "lite"}
    self.assertEqual(sc.peak_bytes(OK, place, modes, {"r": 1, "lite": 10}), [2, 11, 0])
    self.assertEqual(sc.peak_bytes(OK, place, modes, {"lite": 10}), [0, 10, 0])  # stash alone
    self.assertEqual(sc.peak_bytes(OK, place, None, {"r": [1, 3, 99]}), [2, 6, 0])
    self.assertEqual(sc.peak_bytes(SPLIT, place, modes, lambda c, md: 10 if md != "r" else 0), [0, 10, 0])
    with self.assertRaisesRegex(KeyError, "'lite'"):
      sc.peak_bytes(OK, place, modes, {"attn": 10})  # an unsized stash set is not free
    # A stage sums its chunks: at S=2, V=2, M=2 GPipe stage 0 runs 0F0 0F1 2F0
    # 2F1 before any backward, so it holds chunk 0's two sets and chunk 2's.
    place = sc.placement(2, 2)
    o = sc.gpipe(4, 2, place)
    self.assertEqual(sc.peak_residuals(o, 4), [2, 2, 2, 0])
    self.assertEqual(sc.peak_bytes(o, place, None, {"r": 1}), [4, 2])


class OptimalityTest(unittest.TestCase):
  """`search` should reach (M + (S-1)/V) * (t_F + t_B), the analytic optimum."""

  def bound(self, s, v, m):
    return (m + (s - 1) / v) * (sc.COST["F"] + sc.COST["B"])

  def test_search_hits_the_analytic_bound(self):
    for s, v, m in ((4, 1, 8), (4, 1, 16), (4, 2, 8), (4, 2, 16), (4, 4, 16), (8, 1, 16), (8, 2, 32), (16, 2, 32)):
      with self.subTest(s=s, v=v, m=m):
        place, c = sc.placement(s, v), s * v
        _, _, mk = sc.search(c, m, place)
        self.assertAlmostEqual(mk / v, self.bound(s, v, m), places=6)

  def test_1f1b_matches_gpipe_makespan_without_interleaving(self):
    """1F1B's whole point: same speed as GPipe, a fraction of the memory."""
    for s, m in ((2, 4), (4, 8), (4, 16), (8, 16)):
      place = sc.placement(s)
      g = sc.simulate(sc.gpipe(s, m, place), place)[0]
      f = sc.simulate(sc.one_f_one_b(s, m, place), place)[0]
      self.assertEqual(f, g, f"S={s} M={m}")
      self.assertLess(
          max(sc.peak_residuals(sc.one_f_one_b(s, m, place), s)), max(sc.peak_residuals(sc.gpipe(s, m, place), s))
      )

  def test_interleaving_shrinks_the_bubble(self):
    for s, m in ((4, 16), (8, 32)):
      b1 = sc.bubble(sc.search(s, m, sc.placement(s, 1))[0], sc.placement(s, 1))
      b2 = sc.bubble(sc.search(s * 2, m, sc.placement(s, 2))[0], sc.placement(s, 2))
      self.assertLess(b2, b1, f"S={s} M={m}: V=2 bubble {b2} !< V=1 {b1}")

  def test_breadth_first(self):
    """`breadth_first` is legal, reaches the analytic optimum under the
    circular placement at M >= S (a bubble of (S-1)/(V M + S-1) at unit
    costs), is the order `search` picks at S=4, V=2 and 4, M=8, and runs, on
    every device, all forwards before any backward and each chunk in at most
    two visits (one forward run, one backward run; the turnaround chunk's
    two runs meet in one) -- what --fsdp-scope repeat needs."""
    for s, v, m in ((2, 2, 4), (4, 1, 8), (4, 2, 8), (4, 4, 8), (4, 2, 16), (4, 4, 16), (8, 2, 16)):
      with self.subTest(s=s, v=v, m=m):
        place, c = sc.placement(s, v), s * v
        o = sc.breadth_first(c, m, place)
        assert_legal(self, o, c, m)
        mk = sc.simulate(o, place)[0]
        self.assertAlmostEqual(mk / v, self.bound(s, v, m), places=6)
        self.assertAlmostEqual(sc.bubble(o, place), (s - 1) / (v * m + s - 1), places=6)
        for d in range(s):
          seq = [(k, ch) for ch, k, _ in o if place[ch] == d and not (ch == c - 1 and k == "B")]
          kinds = [k for k, _ in seq]
          self.assertEqual(kinds, sorted(kinds, key="FB".index), f"device {d}: a backward before a forward")
          runs = [ch for i, (_, ch) in enumerate(seq) if i == 0 or seq[i - 1][1] != ch]
          self.assertLessEqual(max(runs.count(ch) for ch in set(runs)), 2, f"device {d}: {runs}")
    for v in (2, 4):
      place = sc.placement(4, v)
      self.assertEqual(sc.breadth_first(4 * v, 8, place), sc.search(4 * v, 8, place)[0], f"V={v}")

  def test_single_stage_has_no_bubble(self):
    self.assertEqual(sc.bubble(sc.gpipe(1, 4, [0]), [0]), 0.0)

  def test_pareto_is_a_frontier(self):
    front = sc.pareto(8, 16, sc.placement(4, 2))
    self.assertGreater(len(front), 1)
    mks = [f[0] for f in front]
    peaks = [f[1] for f in front]
    self.assertEqual(mks, sorted(mks))
    self.assertEqual(peaks, sorted(peaks, reverse=True))  # faster costs more memory


class SimulateTest(unittest.TestCase):

  def test_serial_chain(self):
    """One microbatch through S chunks is pure critical path: S*(F+B)."""
    place = sc.placement(4)
    mk, busy, _ = sc.simulate(sc.gpipe(4, 1, place), place)
    self.assertEqual(mk, 4 * (sc.COST["F"] + sc.COST["B"]))
    self.assertEqual(busy, [3.0] * 4)

  def test_busy_is_the_same_for_every_schedule(self):
    place = sc.placement(4, 2)
    work = [
        sc.simulate(o, place)[1] for o in (sc.gpipe(8, 8, place), sc.one_f_one_b(8, 8, place), sc.search(8, 8, place)[0])
    ]
    for b in work[1:]:
      self.assertEqual(b, work[0])

  def test_split_backward_takes_w_off_the_critical_path(self):
    """I then W costs what B does, but the upstream chunk waits only for I."""
    place = [0, 1, 2]
    full = [(0, "F", 0), (1, "F", 0), (2, "F", 0), (2, "B", 0), (1, "B", 0), (0, "B", 0)]
    split = full[:4] + [(1, "I", 0), (0, "B", 0), (1, "W", 0)]
    mk, busy, _ = sc.simulate(full, place)
    self.assertEqual(mk, 9.0)
    mk, busy2, _ = sc.simulate(split, place)
    self.assertEqual(mk, 8.0)  # B(0) starts after I(1), in parallel with W(1)
    self.assertEqual(busy2, busy)

  def test_fused_pair_is_one_interval(self):
    place = [0, 1]
    mk, busy, _ = sc.simulate(PAIRED, place, fuse={3}, pair_gain=0.5)
    self.assertEqual(mk, 11.5)
    self.assertEqual(busy, [5.5, 6.0])
    # Unpaired, 0F1 need not wait for 0B0 to be ready: a pair has to hide more
    # than it delays to pay.
    self.assertEqual(sc.simulate(PAIRED, place)[0], 9.0)
    self.assertEqual(
        sc.timed_simulate(PAIRED, place, sc.COST, fuse_last=True, fuse={3}, pair_gain=0.5), (11.5, [5.5, 6.0])
    )

  def test_modes_pick_the_cost(self):
    place = [0, 1, 2]
    cost = lambda c, k, mode="r": 1.0 if k == "F" else (0.5 if mode == "lite" else 2.0)
    o = [(0, "F", 0), (1, "F", 0), (2, "F", 0), (2, "B", 0), (1, "B", 0), (0, "B", 0)]
    self.assertEqual(sc.simulate(o, place, cost)[0], 9.0)
    self.assertEqual(sc.simulate(o, place, cost, modes={(1, 0): "lite"})[0], 7.5)
    self.assertEqual(sc.timed_simulate(o, place, cost, modes={(1, 0): "lite"})[0], 7.5)


# A legal order at S = C = 3, M = 2: the last chunk fused (its B a no-op right
# after its F), chunk 1's backward whole (OK) or split into I and W (SPLIT).
OK = [
    (0, "F", 0),
    (1, "F", 0),
    (2, "F", 0),
    (2, "B", 0),
    (0, "F", 1),
    (1, "F", 1),
    (2, "F", 1),
    (2, "B", 1),
    (1, "B", 0),
    (0, "B", 0),
    (1, "B", 1),
    (0, "B", 1),
]
SPLIT = [
    (0, "F", 0),
    (1, "F", 0),
    (2, "F", 0),
    (2, "B", 0),
    (0, "F", 1),
    (1, "F", 1),
    (2, "F", 1),
    (2, "B", 1),
    (1, "I", 0),
    (0, "B", 0),
    (1, "I", 1),
    (1, "W", 0),
    (0, "B", 1),
    (1, "W", 1),
]
# S = C = 2, M = 2, with F(0,1) and B(0,0) one program (fuse = {3}).
PAIRED = [(0, "F", 0), (1, "F", 0), (1, "B", 0), (0, "F", 1), (0, "B", 0), (1, "F", 1), (1, "B", 1), (0, "B", 1)]


class ValidateTest(unittest.TestCase):

  def test_accepts_legal_orders(self):
    sc.validate(OK, [0, 1, 2], 2)
    sc.validate(SPLIT, [0, 1, 2], 2)
    sc.validate(PAIRED, [0, 1], 2, fuse={3})
    sc.validate([list(a) for a in OK], [0, 1, 2], 2)  # lists, as from JSON
    sc.validate(OK, [0, 1, 2], 2, modes={(0, 0): "lite", (1, 1): "r"})

  def test_accepts_every_order_the_generators_emit(self):
    """The driver validates whatever it is about to run, so this is the
    contract: everything the existing schedules produce passes, FIFO included."""
    for s, v, m in ((2, 1, 4), (2, 2, 8), (4, 2, 8), (3, 2, 5), (4, 4, 16)):
      for shape in ["circular"] + (["v"] if v == 2 else []):
        place, c = sc.placement(s, v, shape), s * v
        orders = [
            sc.gpipe(c, m, place),
            sc.one_f_one_b(c, m, place),
            sc.search(c, m, place)[0],
            sc.search(c, m, place, max_residual_sets=1)[0],
        ]
        if v <= 2:
          cost = sc.cost_table(sc.TPU7X_COSTS, c)
          orders.append(sc.timed_search(c, m, place, cost, 1.0, fuse_last=True)[0])
        for o in orders:
          with self.subTest(s=s, v=v, m=m, shape=shape):
            sc.validate(o, place, m)

  def test_a_timed_order_may_interleave_other_stages_before_the_noop(self):
    # timed_greedy lists ops that start at the same instant together, so the
    # fused last chunk's no-op B can follow another stage's op; on its own
    # stage it is still right after its F.  (S=4, V=2, M=8 on TPU7x costs.)
    place = sc.placement(4, 2)
    o, _, _ = sc.timed_search(8, 8, place, sc.cost_table(sc.TPU7X_COSTS, 8), 1.0, fuse_last=True)
    gaps = [o.index((7, "B", m)) - o.index((7, "F", m)) for m in range(8)]
    self.assertGreater(max(gaps), 1)
    sc.validate(o, place, 8)
    moved = list(o)
    moved.remove((7, "B", 0))
    moved.insert(o.index((7, "F", 1)), (7, "B", 0))  # past the next F on its own stage
    with self.assertRaisesRegex(sc.ScheduleError, "not right after its fused forward"):
      sc.validate(moved, place, 8)

  def test_validate_rejects_bad_orders(self):
    place = [0, 1, 2]

    def bad(order, pattern, **kw):
      with self.assertRaisesRegex(sc.ScheduleError, pattern):
        sc.validate(order, kw.pop("place", place), 2, **kw)

    bad(OK + [(0, "F", 0)], "appears twice")  # duplicate F
    bad([a for a in OK if a != (0, "B", 1)], "0B1 is missing")  # missing B
    swap = list(OK)
    swap[8], swap[9] = swap[9], swap[8]  # 0B0 before 1B0
    bad(swap, "0B0 at order\\[8\\] runs before 1B0/1I0")
    nonfifo = OK[:8] + [(1, "B", 1), (1, "B", 0), (0, "B", 0), (0, "B", 1)]
    bad(nonfifo, "chunk 1 accumulates .* order \\[1, 0\\], not FIFO")  # non-FIFO B
    sc.validate(nonfifo, place, 2, fifo=False)  # legal otherwise
    w_first = [a for a in SPLIT if a != (1, "W", 0)]
    w_first.insert(w_first.index((1, "I", 0)), (1, "W", 0))  # W before I
    bad(w_first, "1W0 at order\\[8\\] runs before 1B0/1I0")
    bad([a for a in SPLIT if a != (1, "W", 1)], "1W1 .* is missing")  # I, no W
    bad([(0, "F", 0), (0, "I", 0)] + OK[1:9] + [(0, "W", 0)] + OK[10:], "chunk 0 has no input gradient")
    late = [a for a in OK if a != (2, "B", 0)]
    late.insert(late.index((2, "F", 1)) + 1, (2, "B", 0))  # no-op B drifted
    bad(late, "2B0 .* not right after its fused forward")
    bad(OK[:3] + [(0, "X", 0)] + OK[4:], "is not \\(chunk, kind")
    bad(OK + [(0, "F", 2)], "outside 3 chunks x 2 microbatches")
    bad(OK, "the last chunk keeps no residuals", modes={(2, 0): "lite"})
    # pairs: cross-stage, same microbatch, the no-op backward, not F-then-B
    cross = [(0, "F", 0), (1, "F", 0), (1, "B", 0), (0, "F", 1), (1, "F", 1), (0, "B", 0), (1, "B", 1), (0, "B", 1)]
    sc.validate(cross, [0, 1], 2)
    bad(cross, "1F1 is on stage 1 and 0B0 on stage 0", place=[0, 1], fuse={4})
    bad(PAIRED, "share microbatch 0", place=[0, 1], fuse={1}, fuse_last=False)
    bad(PAIRED, "no-op backward", place=[0, 1], fuse={1})
    bad(PAIRED, "must be an F and then a B or I", place=[0, 1], fuse={2})
    bad(PAIRED, "outside the order", place=[0, 1], fuse={7})

  def test_schedule_error_is_a_value_error(self):
    self.assertTrue(issubclass(sc.ScheduleError, ValueError))

  def test_deps(self):
    self.assertEqual(sc.deps(0, "F", 3, 4), [])
    self.assertEqual(sc.deps(2, "F", 3, 4), [(1, "F", 3)])
    self.assertEqual(sc.deps(1, "I", 3, 4), [(1, "F", 3), (2, "BI", 3)])
    self.assertEqual(sc.deps(3, "B", 3, 4), [(3, "F", 3)])
    self.assertEqual(sc.deps(1, "W", 3, 4), [(1, "BI", 3)])
    with self.assertRaises(sc.ScheduleError):
      sc.deps(1, "X", 0, 4)


class MergeTest(unittest.TestCase):

  def test_merge_detects_deadlock(self):
    # Stage 0 wants 0B0 before 2F0, but 0B0 needs 1B0, which needs 2B0.
    place = sc.placement(2, 2)
    streams = [[(0, "F", 0), (0, "B", 0), (2, "F", 0), (2, "B", 0)], [(1, "F", 0), (3, "F", 0), (1, "B", 0)]]
    with self.assertRaisesRegex(sc.ScheduleError, "deadlock.*s0: 0B0"):
      sc.merge(streams, place, sc.COST)

  def test_merge_rejects_an_op_on_the_wrong_stage(self):
    with self.assertRaisesRegex(sc.ScheduleError, "chunk 1 is on stage 1"):
      sc.merge([[(0, "F", 0), (1, "F", 0)], []], [0, 1], sc.COST)

  def test_merge_reproduces_timed_greedy(self):
    """Each stage's projection of a timed order, merged back, finishes at the
    same instant: merge is the same event engine with the choices made."""
    for shape in ("circular", "v"):
      for costs in (sc.V6E_COSTS, sc.TPU7X_COSTS):
        place = sc.placement(2, 2, shape)
        cost = sc.cost_table(costs, 4)
        o, _, mk = sc.timed_search(4, 8, place, cost, 1.0, fuse_last=True)
        streams = [[a for a in o if place[a[0]] == s] for s in range(2)]
        order, fuse, mk2 = sc.merge(streams, place, cost, 1.0, n_micro=8)
        with self.subTest(shape=shape, costs=costs):
          self.assertAlmostEqual(mk2, mk, places=9)
          self.assertEqual(fuse, frozenset())
          self.assertEqual([[a for a in order if place[a[0]] == s] for s in range(2)], streams)
          self.assertAlmostEqual(sc.timed_simulate(order, place, cost, 1.0, fuse_last=True)[0], mk, places=9)
          sc.validate(order, place, 8)

  def test_merge_pairs_and_the_noop(self):
    # The last chunk's no-op B is left out of the streams and comes back right
    # after its F; the pair (0F1, 0B0) is one program, starting when 0B0 can.
    streams = [[(0, "F", 0), ((0, "B", 0), (0, "F", 1)), (0, "B", 1)], [(1, "F", 0), (1, "F", 1)]]
    order, fuse, mk = sc.merge(streams, [0, 1], sc.COST, pair_gain=0.5, n_micro=2)
    self.assertEqual(order, PAIRED)
    self.assertEqual(fuse, frozenset({3}))
    self.assertEqual(mk, 11.5)
    sc.validate(order, [0, 1], 2, fuse=fuse)

  def test_merge_checks_the_microbatch_count(self):
    # Microbatch 7 left out of every stage: the streams agree with each other,
    # so only n_micro tells that the order is short.
    place = sc.placement(2, 2)
    cost = sc.cost_table(sc.V6E_COSTS, 4)
    o, _, _ = sc.timed_search(4, 8, place, cost, 1.0, fuse_last=True)
    full = [[a for a in o if place[a[0]] == s] for s in range(2)]
    short = [[a for a in st if a[2] != 7] for st in full]
    self.assertEqual(len(sc.merge(short, place, cost, 1.0)[0]), 7 * 8)
    with self.assertRaisesRegex(sc.ScheduleError, "0F7 is missing"):
      sc.merge(short, place, cost, 1.0, n_micro=8)
    self.assertEqual(sc.merge(full, place, cost, 1.0, n_micro=8), sc.merge(full, place, cost, 1.0))
    with self.assertRaisesRegex(sc.ScheduleError, "the last chunk keeps no residuals"):
      sc.merge(full, place, cost, 1.0, modes={(3, 0): "lite"}, n_micro=8)


class StashScheduleTest(unittest.TestCase):
  """stash_greedy / stash_search at config A (S=2, V=2, M=8) on the v6e-tile
  costs, whose stash programs (Bs, Fs) are G1's measured progbench deltas."""

  G = 2**30
  SET = {"attn": 0.713, "lite": 1.354}  # GiB per device per set at v6e dims (build_stash)

  def run_search(self, place, pol, budget_gib, **kw):
    C = len(place)
    sets = [int(self.SET[pol] * self.G)] * (C - 1) + [0]
    out = sc.stash_search(C, 8, place, sc.V6E_TILES_COSTS, 1.0, sets, int(budget_gib * self.G), policy=pol, **kw)
    return out, sets

  def test_stash_search_budget_and_fifo(self):
    """Every order is legal, FIFO per chunk (validate), within the budget on
    every stage (peak_bytes, the driver's check) and under the caps; modes
    name only non-last chunks; deterministic; no budget is no stash; and on
    these costs a 4.5 GiB lite budget beats the remat timed search."""
    for shape in ("circular", "v"):
      place = sc.placement(2, 2, shape)
      _, _, timed = sc.timed_search(4, 8, place, sc.V6E_TILES_COSTS, 1.0, fuse_last=True)
      for pol in ("attn", "lite"):
        for budget in (0, 1.5, 3.0, 4.5, 6.0):
          (order, modes, tag, mk), sets = self.run_search(place, pol, budget)
          with self.subTest(shape=shape, pol=pol, budget=budget):
            sc.validate(order, place, 8, modes=modes)  # FIFO included
            peak = sc.peak_bytes(order, place, modes, {pol: sets})
            self.assertTrue(all(b <= budget * self.G for b in peak), peak)
            self.assertTrue(all(md == pol and c != 3 for (c, _), md in modes.items()))
            self.assertEqual(bool(modes), budget >= self.SET[pol])
            if tag[0] != "budget":  # the caps it ran under
              live, top = [0] * 4, [0] * 4
              for c, k, m in order:
                if (c, m) in modes:
                  live[c] += 1 if k == "F" else -1
                  top[c] = max(top[c], live[c])
              self.assertTrue(all(t <= cap for t, cap in zip(top, tag[0])), (top, tag))
            mk2 = sc.timed_simulate(order, place, sc.V6E_TILES_COSTS, 1.0, fuse_last=True, modes=modes)[0]
            self.assertAlmostEqual(mk2, mk, places=6)  # the engine and the replay agree
            if budget == 0:
              self.assertLessEqual(mk, timed + 1e-6)  # random priorities only add
      (o1, m1, t1, mk1), _ = self.run_search(place, "lite", 4.5)
      (o2, m2, t2, mk2), _ = self.run_search(place, "lite", 4.5)
      self.assertEqual((o1, m1, t1, mk1), (o2, m2, t2, mk2))
      self.assertLess(mk1, timed - 50)
    # explicit caps: chunk 0 never stashes, chunk 2 at most once at a time
    place = sc.placement(2, 2)
    (order, modes, tag, _), _ = self.run_search(place, "lite", 6.0, caps=[0, 3, 1, 0])
    self.assertEqual(tag[0], (0, 3, 1, 0))
    self.assertFalse(any(c == 0 for c, _ in modes))
    self.assertEqual(sc.peak_residuals([a for a in order if (a[0], a[2]) in modes], 4)[2], 1)

  def test_stash_greedy_modes_follow_the_budget(self):
    # One set fits per stage: each stage's first forward stashes; its second
    # starts while that set is still live (before its backward), so it remats.
    place = [0, 1, 2]
    cost = lambda c, k, mode="r": 1.0 if k == "F" else (1.0 if mode == "p" else 2.0)
    order, modes, mk, peak = sc.stash_greedy(3, 2, place, cost, 0.0, [5, 5, 0], 5, policy="p")
    sc.validate(order, place, 2, modes=modes)
    self.assertLess(order.index((0, "F", 1)), order.index((0, "B", 0)))
    self.assertEqual(modes, {(0, 0): "p", (1, 0): "p"})
    self.assertEqual(peak, [5, 5, 0])
    # two fit: both microbatches stash
    _, modes, mk2, _ = sc.stash_greedy(3, 2, place, cost, 0.0, [5, 5, 0], 10, policy="p")
    self.assertEqual(len(modes), 4)
    self.assertLess(mk2, mk)
    order, modes, _, peak = sc.stash_greedy(3, 2, place, cost, 0.0, [5, 5, 0], 4, policy="p")
    self.assertEqual((modes, peak), ({}, [0, 0, 0]))
    order, modes, _, peak = sc.stash_greedy(3, 4, [0, 1, 0], cost, 0.0, 5, 10, policy="p")
    self.assertLessEqual(max(sc.peak_bytes(order, [0, 1, 0], modes, {"p": 5})), 10)

  def test_stash_peak_with_temporaries(self):
    """peak_bytes' `temp_bytes`: a stash backward's extra temporaries count
    while it runs, its own set still live -- G1 measured them larger than the
    remat backward's, so the step's peak grows by more than the budget holds:
    at config A a 4.5 GiB lite budget keeps three sets live (4.06 GiB) and
    the peak grows by up to 4.73."""
    place = [0, 1, 2]
    cost = lambda c, k, mode="r": 1.0 if k == "F" else (1.0 if mode == "p" else 2.0)
    for budget, want in ((10, [12, 12, 0]), (5, [7, 7, 0])):
      order, modes, _, peak = sc.stash_greedy(3, 2, place, cost, 0.0, [5, 5, 0], budget, policy="p")
      self.assertEqual(sc.peak_bytes(order, place, modes, {"p": [5, 5, 0]}), peak)
      self.assertEqual(sc.peak_bytes(order, place, modes, {"p": [5, 5, 0]}, {"p": 2}), want)
      # remat backwards add nothing
      self.assertEqual(sc.peak_bytes(order, place, {}, {"p": 5}, {"p": 2}), [0, 0, 0])
    # the shares are G1's: bwda_s temps less bwda's 6.34 GiB, per set
    for pol, gib, extra in (
        ("attn", 0.713, 0.38),
        ("attn_min", 0.659, 0.38),
        ("lite", 1.354, 0.67),
        ("lite_y", 2.229, 0.17),
    ):
      self.assertAlmostEqual(sc.STASH_BWD_TEMP[pol] * gib, extra, delta=0.005, msg=pol)
    self.assertEqual(sc.STASH_BWD_TEMP["all"], 0.0)
    for shape in ("circular", "v"):
      place = sc.placement(2, 2, shape)
      (order, modes, _, _), sets = self.run_search(place, "lite", 4.5)
      temps = [sc.STASH_BWD_TEMP["lite"] * b for b in sets]
      live = sc.peak_bytes(order, place, modes, {"lite": sets})
      grow = sc.peak_bytes(order, place, modes, {"lite": sets}, {"lite": temps})
      self.assertEqual([round(b / self.G, 2) for b in live], [4.06, 4.06], shape)
      self.assertEqual([round(b / self.G, 2) for b in grow], [4.73, 4.73], shape)
      self.assertTrue(all(g > 4.5 * self.G for g in grow))

  def test_stash_costs(self):
    """V6E_TILES_COSTS' measured stash programs (G1): every backward cheaper
    than remat's, every forward dearer, and lite_y < lite < attn in the
    backward.  A table without stash costs gets stash_estimates: B less the
    recompute a policy skips, F plus writing its stash, as shares of F."""
    cost = sc.cost_table(sc.V6E_TILES_COSTS, 4)
    for pol in ("attn", "attn_min", "lite", "lite_y"):
      for c in range(3):
        self.assertLess(cost(c, "B", pol), cost(c, "B"))
        self.assertGreater(cost(c, "F", pol), cost(c, "F"))
    self.assertLess(cost(1, "B", "lite_y"), cost(1, "B", "lite"))
    self.assertLess(cost(1, "B", "lite"), cost(1, "B", "attn"))
    self.assertEqual(cost(2, "B", "lite"), 43.52)  # 53.10 - (53.81 - 44.23)
    est = sc.stash_estimates(sc.V6E_COSTS, "lite", 4)
    self.assertEqual(sorted(est), ["Bs0:lite", "Bs1:lite", "Bs2:lite", "Fs0:lite", "Fs1:lite", "Fs2:lite"])
    s, e = sc.STASH_SAVES["lite"], sc.STASH_FWD_EXTRA["lite"]
    self.assertAlmostEqual(est["Bs1:lite"], 68.70 - s * 25.70, places=2)
    self.assertAlmostEqual(est["Bs0:lite"], 71.65 - s * 25.70, places=2)
    self.assertAlmostEqual(est["Fs0:lite"], 27.98 + e * 25.70, places=2)
    # the shares are the v6e measurements': on the table they came from, the
    # estimate lands within 0.2 ms of the measured non-first chunk (the shares
    # are of progbench's F, 22.77 ms; the table's F is the capture's, 22.44)
    est = sc.stash_estimates({k: v for k, v in sc.V6E_TILES_COSTS.items() if ":" not in k}, "lite", 4)
    self.assertAlmostEqual(est["Bs1:lite"], cost(1, "B", "lite"), delta=0.2)
    self.assertAlmostEqual(est["Fs1:lite"], cost(1, "F", "lite"), delta=0.2)


class CostTableTest(unittest.TestCase):

  def test_per_chunk_keys_and_fallbacks(self):
    k = {
        "F0": 3,
        "F": 2,
        "B0": 6,
        "B": 5,
        "A": 9,
        "F2": 2.5,
        "B1": 4.5,
        "xfer": 1.0,
        "Bs:lite": 3.5,
        "Bs0:lite": 4.0,
        "W1": 1.5,
        "dF_ag": 0.3,
    }
    cost = sc.cost_table(k, 4)
    self.assertEqual([cost(c, "F") for c in range(4)], [3, 2, 2.5, 9])
    self.assertEqual([cost(c, "B") for c in range(4)], [6, 4.5, 5, 0])
    self.assertEqual(cost(1, "B", "lite"), 3.5)
    self.assertEqual(cost(0, "B", "lite"), 4.0)
    self.assertEqual(cost(1, "F", "lite"), 2)  # a stash forward defaults to the remat one
    self.assertEqual(cost(1, "W"), 1.5)
    self.assertEqual(sc.cost_table(k, 4, policy="lite")(2, "B", "s"), 3.5)
    with self.assertRaises(KeyError):
      cost(1, "I")  # no I cost: say so, do not guess
    with self.assertRaises(KeyError):
      cost(1, "B", "attn")  # no cost for that policy
    with self.assertRaises(KeyError):
      sc.cost_table({"F": 1, "B": 2}, 4)  # no A: fails when built
    # d-keys: over the chunk's own remat cost; per chunk before any chunk, an
    # absolute key before a d-key, and both before the policy-less keys
    d = dict(k, **{"dBs:attn": -1.0, "dBs2:attn": -2.0, "Bs1:attn": 2.0, "dFs:attn": 0.5, "dBs:lite": -9.0, "Bs1": 1.0})
    cost = sc.cost_table(d, 4)
    self.assertEqual([cost(c, "B", "attn") for c in range(3)], [5.0, 2.0, 3.0])
    self.assertEqual([cost(c, "F", "attn") for c in range(3)], [3.5, 2.5, 3.0])
    self.assertEqual([cost(c, "B", "lite") for c in range(3)], [4.0, 3.5, 3.5])
    self.assertEqual(sc.cost_table(dict(k, **{"dBs:attn": -1.0}), 4)(2, "B", "attn"), 4.0)

  def test_matches_the_old_closure(self):
    """What `--schedule timed` used to price with, so its orders are unchanged."""
    for k in (sc.TPU7X_COSTS, sc.V6E_COSTS):

      def old(c, kind, k=k):
        if c == 3:
          return k["A"] if kind == "F" else 0.0
        if kind == "F":
          return k["F0"] if c == 0 else k["F"]
        return k["B0"] if c == 0 else k["B"]

      new = sc.cost_table(k, 4)
      for c in range(4):
        for kind in "FB":
          self.assertEqual(new(c, kind), old(c, kind))
      place = sc.placement(2, 2)
      o = sc.search(4, 8, place)[0]
      # an old two-argument cost is still accepted, and never sees the mode;
      # the table itself goes through cost_table
      want = sc.timed_simulate(o, place, new, 1.0, fuse_last=True)
      self.assertEqual(sc.timed_simulate(o, place, old, 1.0, fuse_last=True), want)
      self.assertEqual(sc.timed_simulate(o, place, k, 1.0, fuse_last=True), want)

  def test_v6e_calibration(self):
    """The circular search order measured on a v6e-4, with the measured v6e costs, reproduces its measured step.

    Loop 1589.3 ms + the slower stage's update 19.2 ms = 1608.5 ms, against
    1613 ms measured on the v6e-4 (0.3% low: the transfers and the host-variant
    extras are not in the costs)."""
    place = sc.placement(2, 2)
    o, _, _ = sc.search(4, 8, place)
    mk, busy = sc.timed_simulate(o, place, sc.V6E_COSTS, 1.0, fuse_last=True)
    step = mk + 19.2
    self.assertAlmostEqual(step, 1608.5, delta=1.0)
    self.assertLess(abs(step - 1613) / 1613, 0.03)
    self.assertGreater(busy[0] / mk, 0.97)  # stage 0 sets the step

  def test_timed_v6e(self):
    """`--schedule timed` with the v6e costs: -31 ms against the search order measured there."""
    place = sc.placement(2, 2)
    cost = sc.cost_table(sc.V6E_COSTS, 4)
    o, _, mk = sc.timed_search(4, 8, place, cost, 1.0, fuse_last=True)
    self.assertLessEqual(mk, 1558.5)
    base = sc.timed_simulate(sc.search(4, 8, place)[0], place, cost, 1.0, fuse_last=True)[0]
    self.assertLess(mk, base - 30)
    # the table itself does too, as it does for timed_simulate and merge
    self.assertEqual(sc.timed_search(4, 8, place, sc.V6E_COSTS, 1.0, fuse_last=True)[::2], (o, mk))
    self.assertEqual(
        sc.timed_greedy(4, 8, place, sc.V6E_COSTS, 1.0, fuse_last=True),
        sc.timed_greedy(4, 8, place, cost, 1.0, fuse_last=True),
    )

  def test_v6e_tiles_calibration(self):
    """The same run with `--tiles v6e`: loop 1293.9 ms + update 19.1 ms =
    1313.1 ms against 1319 measured (0.45% low, as above); `timed` finds
    1263.6."""
    place = sc.placement(2, 2)
    o, _, _ = sc.search(4, 8, place)
    mk, busy = sc.timed_simulate(o, place, sc.V6E_TILES_COSTS, 1.0, fuse_last=True)
    step = mk + 19.1
    self.assertAlmostEqual(step, 1313.1, delta=1.0)
    self.assertLess(abs(step - 1319) / 1319, 0.01)
    self.assertGreater(busy[0] / mk, 0.96)
    _, _, mk_t = sc.timed_search(4, 8, place, sc.V6E_TILES_COSTS, 1.0, fuse_last=True)
    self.assertLessEqual(mk_t, 1264.5)


def n_pairs(streams):
  return sum(isinstance(it[0], tuple) for st in streams for it in st)


# 1F1B's stage-0 lifetimes are uneven: 0F0 0F1 0B0 0F2 0B1 0B2 (sets of 2, 3, 2 ops).
LIVES = [
    (0, "F", 0),
    (1, "F", 0),
    (1, "B", 0),
    (0, "F", 1),
    (0, "B", 0),
    (1, "F", 1),
    (1, "B", 1),
    (0, "F", 2),
    (0, "B", 1),
    (1, "F", 2),
    (1, "B", 2),
    (0, "B", 2),
]


class GeneratorsTest(unittest.TestCase):
  """Megatron's interleaved 1F1B, its F&B pairs, and the stash post-pass."""

  def test_megatron_interleaved(self):
    # Megatron-LM's interleaved 1F1B: FIFO-valid, warmup min((S-r-1)*2 +
    # (V-1)*S, M*V) forwards on rank r, and in unit costs the analytic optimum
    # (M + (S-1)/V) (t_F + t_B) V = 3 (MV + S - 1) that `search` also reaches.
    # At M = S the first ranks warm up on every forward (the min clamp) and go
    # straight to the cooldown.
    for s, v, m in (
        (2, 2, 2),
        (2, 2, 4),
        (2, 2, 8),
        (3, 2, 3),
        (3, 2, 6),
        (4, 2, 4),
        (4, 2, 8),
        (4, 2, 16),
        (2, 4, 8),
        (4, 4, 8),
        (2, 5, 4),
    ):
      with self.subTest(s=s, v=v, m=m):
        place = sc.placement(s, v)
        streams = sc.megatron_interleaved(s, v, m)
        order, fuse, mk = sc.merge(streams, place, sc.COST, n_micro=m)
        sc.validate(order, place, m)  # fifo included
        self.assertFalse(fuse)
        self.assertEqual(mk, 3 * (m * v + s - 1))
        self.assertEqual(mk, sc.search(s * v, m, place)[2])
        for r, st in enumerate(streams):
          w = min((s - r - 1) * 2 + (v - 1) * s, m * v)
          self.assertEqual({k for _, k, _ in st[:w]}, {"F"})
          # the steady state opens with an F, an all-warmup rank's cooldown with a B
          self.assertEqual(st[w][1], "F" if w < m * v else "B")
    self.assertEqual(
        [st[:9] for st in sc.megatron_interleaved(4, 2, 4)[:2]],
        [
            [
                (0, "F", 0),
                (0, "F", 1),
                (0, "F", 2),
                (0, "F", 3),
                (4, "F", 0),
                (4, "F", 1),
                (4, "F", 2),
                (4, "F", 3),
                (4, "B", 0),
            ],
            [
                (1, "F", 0),
                (1, "F", 1),
                (1, "F", 2),
                (1, "F", 3),
                (5, "F", 0),
                (5, "F", 1),
                (5, "F", 2),
                (5, "F", 3),
                (5, "B", 0),
            ],
        ],
    )
    # every rank's stream at S=4, V=2, M=8, as Megatron-LM's schedule table runs it (the
    # last chunk's B left out, as merge wants it)
    self.assertEqual(
        [" ".join(map(sc._fmt, st)) for st in sc.megatron_interleaved(4, 2, 8)],
        [
            "0F0 0F1 0F2 0F3 4F0 4F1 4F2 4F3 0F4 0F5 0F6 4B0 0F7 4B1 4F4 4B2 4F5 4B3 4F6 0B0 4F7 0B1 "
            "0B2 0B3 4B4 4B5 4B6 4B7 0B4 0B5 0B6 0B7",
            "1F0 1F1 1F2 1F3 5F0 5F1 5F2 5F3 1F4 5B0 1F5 5B1 1F6 5B2 1F7 5B3 5F4 1B0 5F5 1B1 5F6 1B2 "
            "5F7 1B3 5B4 5B5 5B6 5B7 1B4 1B5 1B6 1B7",
            "2F0 2F1 2F2 2F3 6F0 6F1 6F2 6B0 6F3 6B1 2F4 6B2 2F5 6B3 2F6 2B0 2F7 2B1 6F4 2B2 6F5 2B3 "
            "6F6 6B4 6F7 6B5 6B6 6B7 2B4 2B5 2B6 2B7",
            "3F0 3F1 3F2 3F3 7F0 7F1 7F2 7F3 3F4 3B0 3F5 3B1 3F6 3B2 3F7 3B3 7F4 7F5 7F6 7F7 3B4 3B5 " "3B6 3B7",
        ],
    )
    with self.assertRaisesRegex(ValueError, "chunks_per_stage >= 2"):
      sc.megatron_interleaved(4, 1, 8)
    with self.assertRaisesRegex(ValueError, "multiple of n_stages"):
      sc.megatron_interleaved(4, 2, 6)

  def test_megatron_interleaved_groups(self):
    # Megatron-LM's microbatch_group_size_per_vp_stage G: G = S is the default; any G >= S that divides M is FIFO-valid
    # with warmup (S-r-1)*2 + (V-1)*G and the same zero-latency makespan, and against a boundary copy's latency the
    # slack of G > S shortens it (what stalled the GB200 runs at every chunk switch).
    for s, v, m in ((4, 2, 16), (8, 2, 32), (4, 3, 24)):
      place = sc.placement(s, v)
      self.assertEqual(sc.megatron_interleaved(s, v, m, group=s), sc.megatron_interleaved(s, v, m))
      mks = {}
      for g in (s, 2 * s):
        with self.subTest(s=s, v=v, m=m, g=g):
          streams = sc.megatron_interleaved(s, v, m, group=g)
          for xfer in (0.0, 1.0):
            order, fuse, mk = sc.merge(streams, place, sc.COST, xfer, n_micro=m)
            sc.validate(order, place, m)
            mks[g, xfer] = mk
          for r, st in enumerate(streams):
            w = min((s - r - 1) * 2 + (v - 1) * g, m * v)
            self.assertEqual({k for _, k, _ in st[:w]}, {"F"})
            self.assertEqual(st[w][1], "F" if w < m * v else "B")
          # chunk j's microbatches go in groups of g: rank 0 runs F of chunk 0 on microbatches 0..g-1 first
          self.assertEqual([a for a in streams[0][:g]], [(0, "F", i) for i in range(g)])
      self.assertEqual(mks[s, 0.0], mks[2 * s, 0.0])
      self.assertLess(mks[2 * s, 1.0], mks[s, 1.0])
    with self.assertRaisesRegex(ValueError, "at least n_stages"):
      sc.megatron_interleaved(4, 2, 16, group=2)
    # Megatron runs this one (a last group of 4 >= S microbatches); this order does not, and says which multiple
    with self.assertRaisesRegex(ValueError, "multiple of its microbatch group, got 12 and 8"):
      sc.megatron_interleaved(4, 2, 12, group=8)

  def test_megatron_interleaved_pairs(self):
    # pair_stages: every steady-state F(k+W), B(k) of the ranks named is one item, an F and then a B of another
    # microbatch, but where B(k) is the fused last chunk's no-op (only rank S-1 has those: the steps with
    # k % (G*V) < G), and the last chunk's program never pairs.  Each rank's op sequence is the unpaired one, the
    # merged order FIFO-valid, and in unit costs with free boundaries the makespan Megatron's, 3 (MV + S - 1); none
    # (the default) is the unpaired streams themselves.
    flat = lambda sts: [[a for it in st for a in (it if isinstance(it[0], tuple) else (it,))] for st in sts]
    for s, v, m, g in (
        (1, 2, 2, 1),
        (2, 2, 4, 2),
        (2, 2, 8, 2),
        (2, 2, 8, 4),
        (3, 2, 6, 3),
        (4, 2, 8, 4),
        (4, 2, 16, 8),
        (2, 4, 8, 2),
        (4, 3, 24, 4),
        (8, 2, 32, 8),
    ):
      place, C, total = sc.placement(s, v), s * v, m * v
      ref = sc.megatron_interleaved(s, v, m, group=g)
      self.assertEqual(sc.megatron_interleaved(s, v, m, group=g, pair_stages=()), ref)
      for ps in (None, {s - 1}, {0}):
        with self.subTest(s=s, v=v, m=m, g=g, pair_stages=ps):
          streams = sc.megatron_interleaved(s, v, m, group=g, pair_stages=ps)
          self.assertEqual(flat(streams), ref)
          order, fuse, mk = sc.merge(streams, place, sc.COST, n_micro=m)
          sc.validate(order, place, m, fuse=fuse)  # fifo included
          self.assertEqual(mk, 3 * (m * v + s - 1))
          self.assertEqual(len(fuse), n_pairs(streams))
          on = set(range(s)) if ps is None else ps
          for r, st in enumerate(streams):
            w = min((s - r - 1) * 2 + (v - 1) * g, total)
            want = total - w if r < s - 1 else sum(k % (g * v) >= g for k in range(total - w))
            pairs = [it for it in st if isinstance(it[0], tuple)]
            self.assertEqual(len(pairs), want if r in on else 0, r)
            for f, b in pairs:
              self.assertEqual((f[1], b[1], place[f[0]], place[b[0]]), ("F", "B", r, r))
              self.assertNotEqual(f[2], b[2])
              self.assertNotIn(C - 1, (f[0], b[0]))
    fmt = lambda it: "&".join(map(sc._fmt, it)) if isinstance(it[0], tuple) else sc._fmt(it)
    self.assertEqual(
        [" ".join(map(fmt, st)) for st in sc.megatron_interleaved(4, 2, 8, pair_stages=None)],
        [
            "0F0 0F1 0F2 0F3 4F0 4F1 4F2 4F3 0F4 0F5 0F6&4B0 0F7&4B1 4F4&4B2 4F5&4B3 4F6&0B0 4F7&0B1 0B2 0B3 4B4 4B5 "
            "4B6 4B7 0B4 0B5 0B6 0B7",
            "1F0 1F1 1F2 1F3 5F0 5F1 5F2 5F3 1F4&5B0 1F5&5B1 1F6&5B2 1F7&5B3 5F4&1B0 5F5&1B1 5F6&1B2 5F7&1B3 5B4 5B5 "
            "5B6 5B7 1B4 1B5 1B6 1B7",
            "2F0 2F1 2F2 2F3 6F0 6F1 6F2&6B0 6F3&6B1 2F4&6B2 2F5&6B3 2F6&2B0 2F7&2B1 6F4&2B2 6F5&2B3 6F6&6B4 6F7&6B5 "
            "6B6 6B7 2B4 2B5 2B6 2B7",
            "3F0 3F1 3F2 3F3 7F0 7F1 7F2 7F3 3F4&3B0 3F5&3B1 3F6&3B2 3F7&3B3 7F4 7F5 7F6 7F7 3B4 3B5 3B6 3B7",
        ],
    )
    # The benchmark's shape (S=8, V=2, M=128, G=8): [234, ..., 246] pairs on ranks 0-6 and 120 on rank 7; a pair's
    # F output leaves when its B ends, so a boundary copy's latency (0.16 forwards) shows in every steady-state step
    S, V, M, G = 8, 2, 128, 8
    place = sc.placement(S, V)
    streams = sc.megatron_interleaved(S, V, M, group=G, pair_stages=None)
    self.assertEqual([n_pairs([st]) for st in streams], [234, 236, 238, 240, 242, 244, 246, 120])
    unpaired = sc.megatron_interleaved(S, V, M, group=G)
    mk = lambda st, xfer: sc.merge(st, place, sc.COST, xfer, n_micro=M)[2]
    self.assertEqual((mk(unpaired, 0.0), mk(streams, 0.0)), (789.0, 789.0))
    self.assertAlmostEqual(mk(unpaired, 0.16), 793.8, delta=0.01)
    self.assertAlmostEqual(mk(streams, 0.16), 833.16, delta=0.01)

  def test_stash_modes_earliest_freed_first(self):
    place = [0, 1]
    sc.validate(LIVES, place, 3)
    self.assertEqual(sc.stash_modes(LIVES, place, {"lite": 10}, 9), {})
    # one set's worth: 0F0 (freed first), then 0F2 once 0F0 is gone; 0F1
    # overlaps both
    self.assertEqual(sc.stash_modes(LIVES, place, {"lite": 10}, 10), {(0, 0): "lite", (0, 2): "lite"})
    self.assertEqual(len(sc.stash_modes(LIVES, place, {"lite": 10}, 20)), 3)
    self.assertEqual(len(sc.stash_modes(LIVES, place, {"lite": 10}, [20, 0])), 3)
    self.assertEqual(sc.stash_modes(LIVES, place, {"lite": 10}, 20, caps=[1]), {(0, 0): "lite", (0, 2): "lite"})
    self.assertEqual(len(sc.stash_modes(LIVES, place, {"lite": 10}, 20, caps={1: 0})), 3)
    self.assertEqual(sc.stash_modes(LIVES, place, {"lite": 10}, 20, caps=[0]), {})
    # remat sets priced too: they take the budget first (live 1, 2, 1, 2, 1)
    self.assertEqual(sc.stash_modes(LIVES, place, {"r": 1, "lite": 10}, 11), {(0, 0): "lite", (0, 2): "lite"})
    self.assertEqual(sc.peak_bytes(LIVES, place, {(0, 0): "lite", (0, 2): "lite"}, {"r": 1, "lite": 10}), [11, 0])
    with self.assertRaisesRegex(ValueError, "remat sets alone need 2"):
      sc.stash_modes(LIVES, place, {"r": 1, "lite": 10}, 1)
    with self.assertRaisesRegex(ValueError, "pass policy="):
      sc.stash_modes(LIVES, place, {"lite": 10, "attn": 5}, 10)
    self.assertEqual(
        sc.stash_modes(LIVES, place, {"lite": 10, "attn": 5}, 10, policy="attn"),
        {(0, 0): "attn", (0, 1): "attn", (0, 2): "attn"},
    )
    self.assertEqual(len(sc.stash_modes(LIVES, place, lambda c, md: 0 if md == "r" else 10, 10, policy="lite")), 2)
    with self.assertRaisesRegex(sc.ScheduleError, "0F1 has no B or I"):
      sc.stash_modes([a for a in LIVES if a != (0, "B", 1)], place, {"lite": 10}, 10)
    # an I frees the set as a B does (SPLIT is OK with chunk 1's B's as I / W)
    self.assertEqual(sc.stash_modes(SPLIT, [0, 1, 2], {"lite": 1}, 1), sc.stash_modes(OK, [0, 1, 2], {"lite": 1}, 1))

  def test_stash_modes_as_many_as_fit(self):
    """With one size per stage and no caps, as many sets as any assignment
    stashes: against every subset of each stage's sets (`peak_bytes` judging
    the budget), on the driver's orders at M = 4, 5."""
    tiles, circ, vee = sc.V6E_TILES_COSTS, sc.placement(2, 2), sc.placement(2, 2, "v")

    def most(order, place, sizes, budget):
      n = 0
      for s in range(2):
        sets = [(c, m) for c, k, m in order if k == "F" and c < 3 and place[c] == s]
        n += max(
            len(sub)
            for r in range(len(sets) + 1)
            for sub in itertools.combinations(sets, r)
            if sc.peak_bytes(order, place, dict.fromkeys(sub, "lite"), sizes)[s] <= budget
        )
      return n

    for m in (4, 5):
      orders = {
          "search": (sc.search(4, m, circ)[0], circ),
          "timed": (sc.timed_search(4, m, circ, tiles, 1.0, fuse_last=True)[0], circ),
          "search_v": (sc.search(4, m, vee)[0], vee),
      }
      if m % 2 == 0:  # interleaved 1F1B, unpaired and paired: M a multiple of S
        for name, ps in (("interleaved", ()), ("interleaved_pairs", None)):
          streams = sc.megatron_interleaved(2, 2, m, pair_stages=ps)
          orders[name] = (sc.merge(streams, circ, tiles, 1.0, 2.0, n_micro=m)[0], circ)
      for name, (order, place) in orders.items():
        for sizes in ({"lite": 10}, {"r": 1, "lite": 11}):
          lo = max(sc.peak_bytes(order, place, None, sizes))
          for budget in range(lo, lo + 20 * m + 1, 5):
            with self.subTest(m=m, order=name, sizes=sizes, budget=budget):
              self.assertEqual(len(sc.stash_modes(order, place, sizes, budget)), most(order, place, sizes, budget))

  def test_stash_modes_budget(self):
    """`peak_bytes` stays within the budget, and more budget never gives fewer
    stash sets, on the orders the driver runs (MiB, v6e `lite` sets)."""
    tiles = sc.V6E_TILES_COSTS
    circ, vee, circ4 = sc.placement(2, 2), sc.placement(2, 2, "v"), sc.placement(4, 2)

    def merged(streams, place, cost, g=0.0):
      order, fuse, _ = sc.merge(streams, place, cost, 1.0, g, n_micro=8)
      return order, place, fuse

    orders = {
        "search": (sc.search(4, 8, circ)[0], circ, frozenset()),
        "timed": (sc.timed_search(4, 8, circ, tiles, 1.0, fuse_last=True)[0], circ, frozenset()),
        "search_v": (sc.search(4, 8, vee)[0], vee, frozenset()),
        "interleaved": merged(sc.megatron_interleaved(2, 2, 8), circ, tiles),
        "interleaved_pairs": merged(sc.megatron_interleaved(2, 2, 8, pair_stages=None), circ, tiles, 2.0),
        "search_s4": (sc.search(8, 8, circ4)[0], circ4, frozenset()),
    }
    for name, (order, place, fuse) in orders.items():
      n_sets = sum(1 for c, k, _ in order if k == "F" and c < len(place) - 1)
      for sizes, caps in (
          ({"lite": 1387}, None),
          ({"r": 56, "lite": 1443}, None),
          ({"lite": [1387, 1300, 1450, 0]}, None),
          ({"lite": 1387}, [1, 2, 3]),
      ):
        if len(place) != 4 and (caps or isinstance(sizes["lite"], list)):
          continue
        prev = -1
        for budget in range(56 * 16, 24 * 1024, 128):
          with self.subTest(order=name, sizes=sizes, caps=caps, budget=budget):
            modes = sc.stash_modes(order, place, sizes, budget, caps)
            self.assertTrue(all(p <= budget for p in sc.peak_bytes(order, place, modes, sizes)))
            self.assertGreaterEqual(len(modes), prev)
            prev = len(modes)
            if caps:
              for c, cap in enumerate(caps):
                one = lambda cc, md, c=c: int(cc == c and md != "r")
                self.assertLessEqual(max(sc.peak_bytes(order, place, modes, one)), cap)
        sc.validate(order, place, 8, modes=modes, fuse=fuse)
        if not caps:
          self.assertEqual(prev, n_sets, f"{name}: everything stashes given room")

  def test_v6e_tiles_stash_costs(self):
    # every policy prices chunk 0 and the rest: a slower forward (it writes the
    # residuals out) and a faster backward; the remat programs keep their
    # calibrated costs
    cost = sc.cost_table(sc.V6E_TILES_COSTS, 4)
    for p in ("attn", "attn_min", "lite", "lite_y", "all"):
      for c in range(3):
        with self.subTest(policy=p, chunk=c):
          self.assertGreater(cost(c, "F", p), cost(c, "F"))
          self.assertLess(cost(c, "B", p), cost(c, "B"))
          self.assertLess(cost(c, "F", p) + cost(c, "B", p), cost(c, "F") + cost(c, "B"))
    self.assertAlmostEqual(cost(1, "F", "attn"), 24.30, places=9)  # G1: 22.76 -> 24.62
    self.assertAlmostEqual(cost(1, "B", "attn"), 46.90, places=9)  #     53.47 -> 47.27
    self.assertAlmostEqual(cost(0, "B", "attn"), 55.84 - 4.62, places=9)  # chunk 2's, scaled
    self.assertAlmostEqual(cost(2, "B", "lite"), 53.10 - 9.58, places=9)  # 53.81 -> 44.23
    self.assertAlmostEqual(cost(0, "B", "lite"), 55.84 - 9.16, places=9)  # 56.32 -> 47.16
    self.assertEqual([cost(c, k) for c in range(4) for k in "FB"], [24.75, 55.84, 22.44, 53.10, 22.44, 53.10, 71.46, 0.0])
    # the pipeline's `--costs` merge (`dict(default_costs(...))`, then update):
    # new F / B reprice the stash programs too; an absolute stash key wins
    # where it is at least as specific as the d-key
    k = dict(sc.V6E_TILES_COSTS)
    k.update({"F": 30.0, "B": 70.0, "B0": 72.0, "Bs:lite": 50.0})
    cost = sc.cost_table(k, 4)
    self.assertAlmostEqual(cost(2, "F", "attn"), 31.86, places=9)
    self.assertAlmostEqual(cost(2, "B", "attn"), 63.80, places=9)
    self.assertAlmostEqual(cost(0, "B", "attn"), 67.38, places=9)
    self.assertEqual(cost(1, "B", "lite"), 50.0)
    self.assertAlmostEqual(cost(0, "B", "lite"), 72.0 - 9.16, places=9)


class TimedTest(unittest.TestCase):
  """The event-driven scheduler behind ds_ep_pipeline's --schedule timed."""

  @staticmethod
  def costs(n_chunks, F0=18.5, F=16.5, B0=51.3, B=48.6, A=58.3):
    """ds_ep_pipeline's cost model: chunk 0 carries the embedding, the last
    chunk is one fused forward + backward.  The defaults copy
    sc.TPU7X_COSTS (= ds_ep_pipeline.TIMED_COSTS, the timed schedule's
    costs off a v6e) by hand."""

    def cost(c, kind):
      if c == n_chunks - 1:
        return A if kind == "F" else 0.0
      if kind == "F":
        return F0 if c == 0 else F
      return B0 if c == 0 else B

    return cost

  def test_greedy_orders_are_legal_and_replay_to_their_makespan(self):
    n = 0
    for s, v, m in ((2, 1, 1), (2, 2, 1), (2, 2, 4), (2, 2, 8), (4, 1, 8), (4, 2, 5), (8, 1, 3)):
      place, c = sc.placement(s, v), s * v
      for fuse_last, cost, xfer in ((True, self.costs(c), 1.0), (False, lambda c, k: sc.COST[k], 0.0)):
        for prefer in "FB":
          for cap in (None, [c - i for i in range(c)]):
            with self.subTest(s=s, v=v, m=m, fuse_last=fuse_last, prefer=prefer, cap=cap):
              order, mk = sc.timed_greedy(c, m, place, cost, xfer, prefer=prefer, cap=cap, fuse_last=fuse_last)
              assert_legal(self, order, c, m)
              sim, busy = sc.timed_simulate(order, place, cost, xfer, fuse_last=fuse_last)
              self.assertAlmostEqual(sim, mk, places=6)
              # per-device work does not depend on the order, and bounds the makespan
              work = [m * sum(cost(i, "F") + cost(i, "B") for i in range(c) if place[i] == d) for d in range(s)]
              self.assertEqual(len(busy), s)
              for b, w in zip(busy, work):
                self.assertAlmostEqual(b, w, places=6)
              self.assertGreaterEqual(mk + 1e-9, max(work))
            n += 1
    self.assertGreater(n, 50)

  def test_search_is_legal_and_no_worse_than_its_policies(self):
    for s, v, m in ((2, 2, 1), (2, 2, 8), (4, 2, 8), (2, 4, 6)):
      place, c = sc.placement(s, v), s * v
      cost = self.costs(c)
      with self.subTest(s=s, v=v, m=m):
        order, tag, mk = sc.timed_search(c, m, place, cost, 1.0, fuse_last=True)
        assert_legal(self, order, c, m)
        self.assertEqual(len(tag), 4)
        self.assertAlmostEqual(sc.timed_simulate(order, place, cost, 1.0, fuse_last=True)[0], mk, places=6)
        for prefer in "FB":
          _, g = sc.timed_greedy(c, m, place, cost, 1.0, prefer=prefer, fuse_last=True)
          self.assertLessEqual(mk, g + 1e-9)


def one_f_one_b_iw(S, M, split_last=False):
  """1F1B's order on S stages with each middle chunk's B run as I and then W at once (`split_last`: and the last
  chunk's W right after its F and no-op B)."""
  out = []
  for c, k, m in sc.one_f_one_b(S, M, list(range(S))):
    if k == "B" and 0 < c < S - 1:
      out += [(c, "I", m), (c, "W", m)]
    else:
      out += [(c, k, m)] + ([(c, "W", m)] if split_last and k == "B" and c == S - 1 else [])
  return out


class ZeroBubbleSchedules(unittest.TestCase):
  """ZB-H1 / ZB-H2 on one chunk a stage (`zb_schedule`): orders the driver can replay -- legal and FIFO
  (`validate`), every middle chunk split, within their memory -- never slower than 1F1B, ZB-H2 never slower than
  ZB-H1; and the handcrafted streams, the list scheduler and the I / W cost split they are built from -- with the
  fused last chunk split too (split_last) as well.  On every PR the orders under the pipeline's default costs, the
  last chunk split too and not (test_orders_tpu7x_split), the rest scheduled_only."""

  UNIT = {"F0": 1, "F": 1, "B0": 2, "B": 2, "I": 1, "W": 1, "A": 3, "xfer": 0.0}  # the paper's T_F = T_I = T_W
  # XLA flops of DeepSeek-V3 chunks under remat (I re-runs the forward)
  REMAT = {"F0": 1, "F": 1, "B0": 2.99, "B": 2.99, "I": 2.06, "W": 0.88, "A": 3.27, "xfer": 0.0}

  def table(self, name, S, last=False):
    """The cost table `name` at S stages (`last`: the split last chunk's W priced too -- unit costs' 1, a third of
    A, as the paper's every chunk F, I and W alike; the others' sc.zb_costs)."""
    if name in ("unit", "remat"):
      k = self.UNIT if name == "unit" else self.REMAT
      return dict(k, WA=1.0) if last and name == "unit" else (sc.zb_costs(k, S, last=True) if last else k)
    return sc.zb_costs({"tpu7x": sc.TPU7X_COSTS, "v6e": sc.V6E_TILES_COSTS}[name], S, last=last)

  def check_orders(self, name, split_last=False):
    """S in {2, 3, 4, 8}, M in {1, 2, S, 2S, 8, 16}, wbuf_cap None / 1 / 2: validate (FIFO), chunk 0 whole, every
    middle chunk I + W on every microbatch, the fused last chunk F + its no-op B (`split_last`: and its W); replayed
    by `timed_simulate` to the same makespan; residual sets within ZB-H1's S - s / ZB-H2's 2 (S - s) - 1; W buffers
    within the cap (the last stage's too); never slower than 1F1B (I + W = B here, or less, and the split last
    chunk's AI + W = A); ZB-H2 never slower than ZB-H1."""
    for S in (2, 3, 4, 8):
      k = self.table(name, S, last=split_last)
      cost, xfer, place = sc.cost_table(k, S), k["xfer"], list(range(S))
      kinds = {(0, "F"), (0, "B"), (S - 1, "F"), (S - 1, "B")} | {(c, x) for c in range(1, S - 1) for x in "FIW"}
      kinds |= {(S - 1, "W")} if split_last else set()
      for M in sorted({1, 2, S, 2 * S, 8, 16}):
        f1b = sc.timed_simulate(sc.one_f_one_b(S, M, place), place, cost, xfer, fuse_last=True)[0]
        for cap in (None, 1, 2):
          mk = {}
          for mem in ("h1", "h2"):
            with self.subTest(costs=name, S=S, M=M, cap=cap, mem=mem):
              order, _, mk[mem] = sc.zb_schedule(S, M, cost, xfer, mem=mem, wbuf_cap=cap, split_last=split_last)
              sc.validate(order, place, M)
              self.assertEqual(sc.splits_last(order, S), split_last)
              self.assertEqual(len(order), len(set(order)))
              self.assertEqual({(c, x) for c, x, _ in order}, kinds)
              n = sum(M * (3 if 0 < c < S - 1 else 2) for c in range(S)) + M * split_last
              self.assertEqual(len(order), n)
              self.assertAlmostEqual(sc.timed_simulate(order, place, cost, xfer, fuse_last=True)[0], mk[mem], places=6)
              res, bound = sc.peak_residuals(order, S), sc.zb_caps(S, mem)
              self.assertTrue(all(res[s] <= bound[s] for s in range(S - 1)), (res, bound))
              if cap is not None:
                self.assertLessEqual(max(sc.peak_wbufs(order, place)), cap)
              self.assertLessEqual(mk[mem], f1b + 1e-6)
          self.assertLessEqual(mk["h2"], mk["h1"] + 1e-6, (name, S, M, cap))

  @pytest.mark.scheduled_only
  def test_orders_unit_costs(self):
    self.check_orders("unit")

  @pytest.mark.scheduled_only
  def test_orders_remat_costs(self):
    self.check_orders("remat")

  def test_orders_tpu7x_split(self):
    """check_orders on TPU7X_COSTS, the last chunk split too (split_last) and not."""
    for split_last in (False, True):
      self.check_orders("tpu7x", split_last=split_last)

  @pytest.mark.scheduled_only
  def test_orders_v6e_split(self):
    self.check_orders("v6e")

  @pytest.mark.scheduled_only
  def test_orders_split_last(self):
    """check_orders with the fused last chunk split too (split_last), on every other table."""
    for name in ("unit", "remat", "v6e"):
      self.check_orders(name, split_last=True)

  @pytest.mark.scheduled_only
  def test_split_last_two_stages(self):
    """At S = 2 the split last chunk is the one chunk that splits: ZB-H1 and ZB-H2 are no longer 1F1B's order, and
    faster -- at M = 8 on TPU7X_COSTS (its W 14.575, a quarter of A) 585.625 (ZB-H1: 1F1B with each last W right
    after its F) and 567.125 (ZB-H2) against 1F1B's 609.2."""
    place = [0, 1]
    for name in ("unit", "remat", "tpu7x", "v6e"):
      k = self.table(name, 2, last=True)
      cost = sc.cost_table(k, 2)
      for M in (1, 2, 3, 4, 8, 16):
        f1b = sc.timed_simulate(sc.one_f_one_b(2, M, place), place, cost, k["xfer"], fuse_last=True)[0]
        for cap in (None, 1, 2):
          for mem in ("h1", "h2"):
            with self.subTest(costs=name, M=M, cap=cap, mem=mem):
              order, tag, mk = sc.zb_schedule(2, M, cost, k["xfer"], mem=mem, wbuf_cap=cap, split_last=True)
              self.assertNotEqual(order, sc.one_f_one_b(2, M, place))
              self.assertEqual({x for c, x, _ in order if c == 1}, {"F", "B", "W"})
              self.assertLess(mk, f1b - 1e-6)
              if mem == "h1" and cap == 1:
                self.assertEqual((order, tag), (one_f_one_b_iw(2, M, split_last=True), ("1f1b", "I+W")))
    cost = sc.cost_table(self.table("tpu7x", 2, last=True), 2)
    self.assertAlmostEqual(sc.zb_schedule(2, 8, cost, 1.0, mem="h1", split_last=True)[2], 585.625, places=6)
    self.assertAlmostEqual(sc.zb_schedule(2, 8, cost, 1.0, mem="h2", split_last=True)[2], 567.125, places=6)

  @pytest.mark.scheduled_only
  def test_split_last_costs(self):
    """The split last chunk's costs: cost_table's W{C-1}, else WA, and its F + I = AI (A - W unless measured);
    zb_costs(last=True) prices WA = ZB_W_SHARE_LAST x A where neither is measured, A - AI from a measured AI, and
    refuses a negative one or one alone above A; timed_simulate and merge price its F as AI and its W after it,
    simulate its no-op B as the I its F runs; peak_wbufs counts its F's buffers.  Orders without its W's price as
    before."""
    k = sc.zb_costs(sc.TPU7X_COSTS, 3, last=True)
    self.assertEqual(sorted(set(k) - set(sc.TPU7X_COSTS)), ["I1", "W1", "WA"])
    self.assertEqual((sc.ZB_W_SHARE_LAST, k["WA"]), (0.25, round(0.25 * 58.3, 4)))
    cost = sc.cost_table(k, 3)
    self.assertEqual((cost(2, "F"), cost(2, "B"), cost(2, "W")), (58.3, 0.0, k["WA"]))
    self.assertAlmostEqual(cost(2, "F") + cost(2, "I"), 58.3 - k["WA"], places=9)  # AI = A - W
    self.assertEqual(sc.cost_table(dict(k, AI=40.0), 3)(2, "I"), 40.0 - 58.3)  # a measured AI
    self.assertEqual(sc.cost_table(dict(k, W2=9.0), 3)(2, "W"), 9.0)  # W{C-1} before WA
    with self.assertRaisesRegex(KeyError, "none of"):
      sc.cost_table(sc.TPU7X_COSTS, 3)(2, "W")
    self.assertEqual(sc.zb_costs(dict(sc.TPU7X_COSTS, AI=50.0), 3, last=True)["WA"], round(58.3 - 50.0, 4))
    self.assertNotIn("WA", sc.zb_costs(dict(sc.TPU7X_COSTS, W2=9.0), 3, last=True))
    self.assertEqual(sc.zb_costs(sc.TPU7X_COSTS, 2, last=True, w_share_last=0.5)["WA"], 29.15)
    self.assertNotIn("WA", sc.zb_costs(sc.TPU7X_COSTS, 3))  # only where the last chunk splits
    for bad, msg in (
        ({"WA": -1.0}, r"the last chunk's W = -1.0 < 0"),
        ({"AI": 60.0}, r"the last chunk's AI = 60.0 > A = 58.3, which leaves its W = A - AI negative"),
        ({"W2": 59.0}, r"the last chunk's W = 59.0 > A = 58.3, which leaves its AI = A - W negative"),
    ):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, msg):
        sc.zb_costs(dict(sc.TPU7X_COSTS, **bad), 3, last=True)
    with self.assertRaisesRegex(ValueError, "w_share_last must be within"):
      sc.zb_costs(sc.TPU7X_COSTS, 3, last=True, w_share_last=2.0)
    # the split F costs AI, its W after it: two stages, 1F1B with each last W at once
    place, M = [0, 1], 3
    k2 = sc.zb_costs(sc.TPU7X_COSTS, 2, last=True)
    cost = sc.cost_table(k2, 2)
    order = one_f_one_b_iw(2, M, split_last=True)
    self.assertTrue(sc.splits_last(order, 2) and not sc.splits_last(sc.one_f_one_b(2, M, place), 2))
    sc.validate(order, place, M)
    mk, busy = sc.timed_simulate(order, place, cost, 1.0, fuse_last=True)
    self.assertAlmostEqual(busy[1], M * 58.3, places=9)  # AI + W = A a microbatch
    base, busy0 = sc.timed_simulate(sc.one_f_one_b(2, M, place), place, cost, 1.0, fuse_last=True)
    self.assertAlmostEqual(busy0[1], M * 58.3, places=9)
    self.assertLess(mk, base)
    streams = [[a for a in order if a[0] == s and a[:2] != (1, "B")] for s in place]
    self.assertAlmostEqual(sc.merge(streams, place, cost, 1.0, n_micro=M)[2], mk, places=9)
    self.assertEqual(sc.peak_wbufs(order, place), [0, 1])
    deferred = sc.one_f_one_b(2, M, place) + [(1, "W", m) for m in range(M)]
    self.assertEqual(sc.peak_wbufs(deferred, place), [0, M])
    unit = sc.simulate(order, place)  # F + I + W = 1 + 1 + 1 on the last stage, as F + B
    self.assertEqual(unit[1][1], sc.simulate(sc.one_f_one_b(2, M, place), place)[1][1])

  @pytest.mark.scheduled_only
  def test_split_last_streams_and_search(self):
    """zb_h1 / zb_h2 with split_last: the last stage's stream its F's and W's, its W's `w_lag` behind -- min(lag + 1,
    M) W buffers (ZB-H2: min(lag + 2, M), its warm-up forward's too); zb_greedy keeps the last stage's W buffers
    under wbuf_cap (its F waits for a W), the in-flight and residual caps untouched by them; zb_schedule drops a
    candidate over the cap."""
    for S in (2, 3, 4):
      place = list(range(S))
      cost = sc.cost_table(self.table("tpu7x", S, last=True), S)
      for M in (1, 2, 5, 8):
        for h1 in (True, False):
          fn = sc.zb_h1 if h1 else sc.zb_h2
          for w_lag in (None, 0, 1):
            with self.subTest(S=S, M=M, fn=fn.__name__, w_lag=w_lag):
              streams = fn(S, M, w_lag=w_lag, split_last=True)
              self.assertEqual({(c, x) for c, x, _ in streams[S - 1]}, {(S - 1, "F"), (S - 1, "W")})
              order, _, _ = sc.merge(streams, place, cost, 1.0, n_micro=M)
              sc.validate(order, place, M)
              lag = (S - 1 if h1 else 2 * (S - 1)) if w_lag is None else w_lag
              self.assertEqual(sc.peak_wbufs(order, place)[S - 1], min(lag + (1 if h1 else 2), M))
              self.assertEqual(
                  sc.peak_residuals(order, S),
                  sc.peak_residuals(sc.merge(fn(S, M, w_lag=w_lag), place, cost, 1.0, n_micro=M)[0], S),
              )
      for cap in (1, 2):
        for name, pol in sc.ZB_POLICIES:
          with self.subTest(S=S, cap=cap, rule=name):
            streams, mk = sc.zb_greedy(
                S, 6, place, cost, 1.0, wbuf_cap=cap, split_last=True, inflight_cap=sc.zb_caps(S, "h1"), **pol
            )
            order, _, mk2 = sc.merge(streams, place, cost, 1.0, n_micro=6)
            sc.validate(order, place, 6)
            self.assertAlmostEqual(mk2, mk, places=6)
            self.assertLessEqual(max(sc.peak_wbufs(order, place)), cap)
            live = sc.peak_inflight(order, place)
            self.assertTrue(all(live[s] <= sc.zb_caps(S, "h1")[s] for s in range(S - 1)), live)
    with self.assertRaisesRegex(RuntimeError, "deadlock"):
      sc.zb_greedy(2, 4, [0, 1], sc.cost_table(self.table("tpu7x", 2, last=True), 2), wbuf_cap=0, split_last=True)
    # ZB-H2's stream holds two W buffers on the split last stage at lag 0: over a cap of 1, zb_schedule drops it
    cost = sc.cost_table(self.table("tpu7x", 2, last=True), 2)
    for M in (2, 8):
      order, tag, _ = sc.zb_schedule(2, M, cost, 1.0, mem="h2", wbuf_cap=1, split_last=True)
      self.assertNotEqual(tag, ("zb_h2", "streams"))
      self.assertLessEqual(max(sc.peak_wbufs(order, [0, 1])), 1)

  @pytest.mark.scheduled_only
  def test_two_stages_zbh1_is_1f1b(self):
    """At S = 2 chunk 0 runs a whole B and the last chunk is fused: nothing splits, and ZB-H1 is 1F1B's order
    itself.  ZB-H2's deeper warm-up still gains: 581.7 against 1F1B's 609.2 at M = 8 on TPU7X_COSTS."""
    place = [0, 1]
    for name in ("unit", "remat", "tpu7x", "v6e"):
      k = self.table(name, 2)
      cost = sc.cost_table(k, 2)
      for M in (1, 2, 3, 4, 8, 16):
        for cap in (None, 1, 2):
          with self.subTest(costs=name, M=M, cap=cap):
            order, tag, _ = sc.zb_schedule(2, M, cost, k["xfer"], mem="h1", wbuf_cap=cap)
            self.assertEqual(order, sc.one_f_one_b(2, M, place))
            self.assertEqual(tag, ("1f1b", "I+W"))
            order, _, _ = sc.zb_schedule(2, M, cost, k["xfer"], mem="h2", wbuf_cap=cap)
            self.assertFalse({x for _, x, _ in order} & {"I", "W"})
    cost = sc.cost_table(self.table("tpu7x", 2), 2)
    self.assertAlmostEqual(sc.zb_schedule(2, 8, cost, 1.0, mem="h2")[2], 581.7, places=6)
    self.assertAlmostEqual(sc.timed_simulate(sc.one_f_one_b(2, 8, place), place, cost, 1.0, fuse_last=True)[0], 609.2)

  @pytest.mark.scheduled_only
  def test_unit_costs_save_a_w_per_split_stage(self):
    """T_F = T_I = T_W with the last chunk fused: 1F1B's drain sends each microbatch's input gradient down through
    S - 2 split chunks, which ZB runs as I's (the W's come later): (S - 2) x T_W off 1F1B's makespan, at any M."""
    for S in (3, 4, 5, 8):
      cost, place = sc.cost_table(self.UNIT, S), list(range(S))
      for M in (1, 2, S, 8, 16):
        f1b = sc.timed_simulate(sc.one_f_one_b(S, M, place), place, cost, 0.0, fuse_last=True)[0]
        for mem in ("h1", "h2"):
          for cap in (None, 1):
            with self.subTest(S=S, M=M, mem=mem, cap=cap):
              self.assertEqual(f1b - sc.zb_schedule(S, M, cost, 0.0, mem=mem, wbuf_cap=cap)[2], S - 2)

  @pytest.mark.scheduled_only
  def test_paper_bubble(self):
    """The paper's setting -- every chunk split (chunk 0 and an unfused last chunk too), T_F = T_I = T_W, 1F1B's
    residual sets, the W's deferred without bound -- `zb_greedy` reaches the lower bound 3M + S - 1 (M >= S)."""
    cost = {"F": 1.0, "B": 2.0, "I": 1.0, "W": 1.0}
    for S, M in ((2, 8), (3, 6), (4, 8), (4, 16), (8, 8), (8, 16), (8, 32)):
      with self.subTest(S=S, M=M):
        _, mk = sc.zb_greedy(
            S, M, list(range(S)), cost, split=set(range(S)), res_cap=sc.zb_caps(S, "h1"), fuse_last=False
        )
        self.assertEqual(mk, 3 * M + S - 1)

  @pytest.mark.scheduled_only
  def test_handcrafted_streams(self):
    """zb_h1: 1F1B's warm-up (S - 1 - r forwards on stage r) and residual sets, W lag r by default; zb_h2: a warm-up of
    2 (S - r) - 1 forwards and as many residual sets, W lag 2 r.  A lag of d holds min(d + 1, M) W buffers on a split
    stage; at lag 0 ZB-H1's streams are 1F1B's with each B as I then W.  Every stream merges into a valid FIFO order."""
    for S in (2, 3, 4, 8):
      place = list(range(S))
      cost = sc.cost_table(self.table("tpu7x", S), S)
      for M in (1, 2, 3, 5, 8, 16):
        for h1 in (True, False):
          fn = sc.zb_h1 if h1 else sc.zb_h2
          for w_lag in (None, 0, 1):
            with self.subTest(S=S, M=M, fn=fn.__name__, w_lag=w_lag):
              streams = fn(S, M, w_lag=w_lag)
              order, fuse, _ = sc.merge(streams, place, cost, 1.0, n_micro=M)
              sc.validate(order, place, M, fuse=fuse)
              self.assertEqual(fuse, frozenset())
              res, wb = sc.peak_residuals(order, S), sc.peak_wbufs(order, place)
              for r in range(S - 1):
                warm = S - 1 - r if h1 else 2 * (S - r) - 1
                lag = (r if h1 else 2 * r) if w_lag is None else w_lag
                self.assertEqual({x for _, x, _ in streams[r][: min(warm, M)]}, {"F"})
                self.assertEqual(res[r], min(S - r if h1 else warm, M))
                if r:
                  self.assertEqual(wb[r], min(lag + 1, M))
              if h1:
                self.assertEqual(res[: S - 1], sc.peak_residuals(sc.one_f_one_b(S, M, place), S)[: S - 1])
              if h1 and w_lag == 0:
                iw = one_f_one_b_iw(S, M)
                self.assertEqual(streams, [[a for a in iw if a[0] == r and a[:2] != (S - 1, "B")] for r in range(S)])

  @pytest.mark.scheduled_only
  def test_search_respects_its_caps(self):
    """zb_greedy under each memory form of zb_search: 'inflight' -- at most zb_caps microbatches from F to W or B on
    a stage (`peak_inflight`) -- and 'res' -- zb_caps residual sets, S (ZB-H1) or 2S - 1 (ZB-H2) in flight -- with
    the W-buffer cap on top; zb_search's best keeps whichever its tag names."""
    for S, M in ((3, 5), (4, 8), (8, 16)):
      place = list(range(S))
      cost = sc.cost_table(self.table("tpu7x", S), S)
      for mem in ("h1", "h2"):
        caps, total = sc.zb_caps(S, mem), S if mem == "h1" else 2 * S - 1
        for cap in (None, 1, 2):
          for name, pol in sc.ZB_POLICIES:
            for form, kw in (("inflight", {"inflight_cap": caps}), ("res", {"res_cap": caps, "inflight_cap": total})):
              with self.subTest(S=S, M=M, mem=mem, cap=cap, rule=name, form=form):
                streams, mk = sc.zb_greedy(S, M, place, cost, 1.0, wbuf_cap=cap, **kw, **pol)
                order, _, mk2 = sc.merge(streams, place, cost, 1.0, n_micro=M)
                sc.validate(order, place, M)
                self.assertAlmostEqual(mk2, mk, places=6)  # merge replays the engine's timeline
                live = sc.peak_inflight(order, place)
                bound = caps if form == "inflight" else [total] * S
                self.assertTrue(all(live[s] <= bound[s] for s in range(S - 1)), (live, bound))
                res = sc.peak_residuals(order, S)
                self.assertTrue(all(res[s] <= caps[s] for s in range(S - 1)), (res, caps))
                if cap is not None:
                  self.assertLessEqual(max(sc.peak_wbufs(order, place)), cap)
          streams, tag, mk = sc.zb_search(S, M, cost, 1.0, mem=mem, wbuf_cap=cap)
          self.assertEqual(tag[0], mem)
          self.assertIn(tag[1], sc.ZB_MEMORY_FORMS)
          self.assertIn(tag[2], dict(sc.ZB_POLICIES))
    # a split chunk with no room for a W buffer can never run its I
    with self.assertRaisesRegex(RuntimeError, "deadlock"):
      sc.zb_greedy(4, 4, [0, 1, 2, 3], sc.cost_table(self.table("tpu7x", 4), 4), wbuf_cap=0)
    with self.assertRaisesRegex(RuntimeError, "every policy deadlocks"):
      sc.zb_search(4, 4, sc.cost_table(self.table("tpu7x", 4), 4), mem="h1", wbuf_cap=0)

  @pytest.mark.scheduled_only
  def test_deterministic(self):
    """Every process of a multi-controller run builds the order on its own: the same inputs, the same order."""
    cost = sc.cost_table(self.table("tpu7x", 8), 8)
    for mem in ("h1", "h2"):
      for cap in (None, 1):
        self.assertEqual(
            sc.zb_schedule(8, 16, cost, 1.0, mem=mem, wbuf_cap=cap),
            sc.zb_schedule(8, 16, cost, 1.0, mem=mem, wbuf_cap=cap),
        )

  @pytest.mark.scheduled_only
  def test_simulated_makespans(self):
    """Pinned makespans (rounded to 0.01) of the candidates zb_schedule picks from, and zb_schedule never above them.
    TPU7X_COSTS with B split I 32.724 / W 15.876 by flops: the handcrafted streams, uncapped; at wbuf_cap 1 the faster
    of the streams (W lag 0) and six rules of the search in the 'res' form.  V6E_TILES_COSTS split I 37.70 / W 15.40:
    the faster of 1F1B with I then W and six rules of the search in the 'inflight' form, under ZB-H1's caps (ZB-H2:
    and its own), uncapped."""
    tpu7x = dict(sc.TPU7X_COSTS, I=32.724, W=15.876)
    streams = {  # (S, M): (ZB-H1, ZB-H2)
        (4, 8): (674.5, 640.65),
        (4, 16): (1232.9, 1168.55),
        (4, 32): (2349.7, 2285.35),
        (8, 8): (845.54, 845.54),
        (8, 16): (1368.34, 1311.94),
        (8, 32): (2480.29, 2346.34),
    }
    capped = {
        (4, 8): (683.45, 660.65),
        (4, 16): (1232.9, 1181.45),
        (4, 32): (2349.7, 2285.35),
        (8, 8): (879.34, 865.54),
        (8, 16): (1409.14, 1386.34),
        (8, 32): (2480.29, 2427.94),
    }
    v6e = {"F0": 24.75, "F": 22.44, "B0": 55.84, "B": 53.10, "I": 37.70, "W": 15.40, "A": 71.46, "xfer": 1.0}
    searched = {
        (4, 8): (805.03, 778.55),
        (4, 16): (1443.15, 1371.51),
        (4, 32): (2732.59, 2658.34),
        (8, 8): (1053.59, 1027.11),
        (8, 16): (1657.91, 1620.07),
        (8, 32): (2907.35, 2828.71),
    }
    rules = dict(sc.ZB_POLICIES)
    res_rules = [
        (n, rules[n]) for n in ("greedy", "w_first", "F first", "F first, w_first", "W before F", "W before F, w_first")
    ]
    inflight_rules = [
        (n, rules[n]) for n in ("greedy", "w_slack 0.5", "w_slack 1", "w_slack 1 +F", "w_slack 0.5 +F", "W before F")
    ]
    for (S, M), want in streams.items():
      place, cost = list(range(S)), sc.cost_table(tpu7x, S)
      for i, (mem, fn) in enumerate((("h1", sc.zb_h1), ("h2", sc.zb_h2))):
        with self.subTest(costs="tpu7x", S=S, M=M, mem=mem):
          self.assertAlmostEqual(sc.merge(fn(S, M), place, cost, 1.0, n_micro=M)[2], want[i], places=2)
          self.assertLessEqual(sc.zb_schedule(S, M, cost, 1.0, mem=mem, wbuf_cap=None)[2], want[i] + 0.005)
          got = min(
              sc.merge(fn(S, M, w_lag=0), place, cost, 1.0, n_micro=M)[2],
              sc.zb_search(S, M, cost, 1.0, mem=mem, wbuf_cap=1, policies=res_rules, forms=("res",))[2],
          )
          self.assertAlmostEqual(got, capped[S, M][i], places=2)
          self.assertLessEqual(sc.zb_schedule(S, M, cost, 1.0, mem=mem, wbuf_cap=1)[2], capped[S, M][i] + 0.005)
    for (S, M), want in searched.items():
      place, cost = list(range(S)), sc.cost_table(v6e, S)
      base = sc.timed_simulate(one_f_one_b_iw(S, M), place, cost, 1.0, fuse_last=True)[0]
      for i, mem in enumerate(("h1", "h2")):
        with self.subTest(costs="v6e", S=S, M=M, mem=mem):
          got = min(
              [base]
              + [
                  sc.zb_search(S, M, cost, 1.0, mem=mm, policies=inflight_rules, forms=("inflight",))[2]
                  for mm in ("h1", "h2")[: i + 1]
              ]
          )
          self.assertAlmostEqual(got, want[i], places=2)
          self.assertLessEqual(sc.zb_schedule(S, M, cost, 1.0, mem=mem, wbuf_cap=None)[2], want[i] + 0.005)

  @pytest.mark.scheduled_only
  def test_zb_costs(self):
    """I and W for the middle chunks only, I + W = B: W = ZB_W_SHARE x B (B{c} first), a measured I or W keeping its
    value; F, B and A unchanged, so every order priced by them (timed, the stash search) is unchanged too.  A
    measured I or W alone above its B, which would leave the other negative, and a negative one are refused."""
    k = sc.zb_costs(sc.TPU7X_COSTS, 4)
    self.assertEqual(sorted(set(k) - set(sc.TPU7X_COSTS)), ["I1", "I2", "W1", "W2"])
    self.assertEqual(
        (k["W1"], k["I1"]), (round(sc.ZB_W_SHARE * 48.6, 4), round(48.6 - round(sc.ZB_W_SHARE * 48.6, 4), 4))
    )
    self.assertEqual(sc.ZB_W_SHARE, 0.3)
    cost = sc.cost_table(k, 4)
    for c in (1, 2):
      self.assertAlmostEqual(cost(c, "I") + cost(c, "W"), cost(c, "B"), places=9)
    self.assertEqual(sc.zb_costs(sc.TPU7X_COSTS, 2), dict(sc.TPU7X_COSTS))  # nothing splits at two chunks
    t = dict(sc.TPU7X_COSTS, B2=40.0, I3=30.0, W=10.0, B3=45.0)
    k = sc.zb_costs(t, 5)
    self.assertEqual((k["I1"], k["I2"], k["I3"]), (38.6, 30.0, 30.0))  # B - W, B2 - W, the measured I3 kept
    self.assertNotIn("W1", k)  # the table's own W prices every chunk
    k = sc.zb_costs(dict(sc.TPU7X_COSTS, I=40.0), 4, w_share=0.5)
    self.assertEqual((k["W1"], k["W2"], "I1" in k), (8.6, 8.6, False))
    self.assertEqual(sc.zb_costs(sc.TPU7X_COSTS, 3, w_share=0.5)["W1"], 24.3)
    with self.assertRaisesRegex(ValueError, "within"):
      sc.zb_costs(sc.TPU7X_COSTS, 4, w_share=1.5)
    for bad, msg in (
        ({"I2": 50.0}, r"I2 = 50.0 > B2 = 48.6, which leaves W2 = B - I negative"),
        ({"W": 60.0}, r"W1 = 60.0 > B1 = 48.6, which leaves I1 = B - W negative; W2 = 60.0 > B2"),
        ({"I1": -1.0, "W1": 1.0}, r"I1 = -1.0 < 0 \(give a chunk's I and W both"),
    ):
      with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, msg):
        sc.zb_costs(dict(sc.TPU7X_COSTS, **bad), 4)
    self.assertEqual(sc.zb_costs(dict(sc.TPU7X_COSTS, I1=50.0, W1=10.0), 4)["I1"], 50.0)  # both measured: as they are
    with self.assertRaises(KeyError):
      sc.zb_costs({"F": 1, "B": 2}, 4)  # no A: cost_table refuses it
    for table in (sc.TPU7X_COSTS, sc.V6E_COSTS, sc.V6E_TILES_COSTS):
      for S, V in ((2, 1), (3, 1), (4, 1), (8, 1), (2, 2), (4, 2)):
        C, place = S * V, sc.placement(S, V)
        with self.subTest(S=S, V=V):
          split = sc.zb_costs(table, C)
          for c in range(C):
            for x in ("F", "B"):
              self.assertEqual(sc.cost_table(split, C)(c, x), sc.cost_table(table, C)(c, x))
          if C <= 4:  # timed_search's policy space grows fast with the chunks
            self.assertEqual(
                sc.timed_search(C, 8, place, sc.cost_table(split, C), 1.0, fuse_last=True),
                sc.timed_search(C, 8, place, sc.cost_table(table, C), 1.0, fuse_last=True),
            )

  @pytest.mark.scheduled_only
  def test_arguments(self):
    cost = sc.cost_table(self.table("tpu7x", 4), 4)
    with self.assertRaisesRegex(ValueError, "mem must be"):
      sc.zb_schedule(4, 8, cost, mem="h3")
    for cap in (0, -1, 1.5, True):
      with self.assertRaisesRegex(ValueError, "wbuf_cap must be None"):
        sc.zb_schedule(4, 8, cost, wbuf_cap=cap)
    with self.assertRaisesRegex(ValueError, ">= 2 stages"):
      sc.zb_schedule(1, 8, sc.cost_table(sc.TPU7X_COSTS, 1))
    with self.assertRaisesRegex(ValueError, ">= 1 microbatch"):
      sc.zb_h2(4, 0)
    with self.assertRaisesRegex(ValueError, "mem must be"):
      sc.zb_caps(4, "h0")
    with self.assertRaisesRegex(ValueError, "w_lag has 3 entries"):
      sc.zb_h1(4, 8, w_lag=[0, 0, 0])
    self.assertEqual(sc.zb_caps(4, "h1"), [4, 3, 2, 1])
    self.assertEqual(sc.zb_caps(4, "h2"), [7, 5, 3, 1])


if __name__ == "__main__":
  unittest.main()
