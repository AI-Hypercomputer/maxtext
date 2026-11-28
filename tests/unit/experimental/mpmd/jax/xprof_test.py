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

"""Tests for maxtext.experimental.mpmd.jax.xprof.  Pure Python -- no JAX, no accelerator, and no
xplane file: the protobuf parsing is one thin layer, so what is tested here is
everything above it, against synthetic timelines whose answers are known by
construction.  The parse itself is in `xprof_pb_test.py`, which is a separate
file because it writes real xplane bytes with protobuf, and this one needs no
protobuf at all.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from maxtext.experimental.mpmd.jax import pathways
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax import xprof


def ev(start_us, dur_us, name="op", **stats):
  """An Event in microseconds, since picoseconds are unreadable in a test."""
  return xprof.Event(int(start_us * xprof.PS), int(dur_us * xprof.PS), name, stats)


def fake(mods_by_dev, host=(), ops_by_dev=None):
  """A Trace with `modules` preloaded and no file behind it.

  `host` is the driver thread's events, or a {line name: events} dict when the
  test cares which of several host lines gets read.
  """
  tr = xprof.Trace.__new__(xprof.Trace)
  tr.path = "<synthetic>"
  tr.devices = {d: {"XLA Modules": ms} for d, ms in mods_by_dev.items()}
  for d, os_ in (ops_by_dev or {}).items():
    tr.devices[d]["XLA Ops"] = os_
  tr.peaks = {d: (946.7e12, 1637.993152512e9) for d in mods_by_dev}
  tr.n_steps_line = {d: len(ms) for d, ms in mods_by_dev.items()}
  tr.host = dict(host) if isinstance(host, dict) else {"python3": list(host)}
  return tr


def run(cmd, tr, **kw):
  """A subcommand's stdout, with the arg defaults `main` would have parsed."""
  args = argparse.Namespace(device=None, module="jit_bwda_c0", step=-1, width=110, flops=0, bytes=0, min_us=20.0)
  vars(args).update(kw)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    cmd(tr, args)
  return buf.getvalue()


def main_out(tr, *argv):
  """`main`'s stdout for the command line `argv`, with `tr` standing in for the trace it would have read."""
  buf = io.StringIO()
  with mock.patch.multiple(xprof, find_xplane=lambda path: path, Trace=lambda path: tr), contextlib.redirect_stdout(buf):
    xprof.main(list(argv))
  return buf.getvalue()


class EventTest(unittest.TestCase):

  def test_base_strips_program_id_and_ordinal(self):
    self.assertEqual(ev(0, 1, "jit__bwda(1731438548)").base, "jit__bwda")
    self.assertEqual(ev(0, 1, "jit_fwd_raw(8006710057128423225)").base, "jit_fwd_raw")
    self.assertEqual(ev(0, 1, "%fusion.3 = f32[8,4]{1,0} fusion(...)").base, "fusion")
    self.assertEqual(ev(0, 1, "bitcast_dynamic-update-slice_fusion").base, "bitcast_dynamic-update-slice_fusion")

  def test_program_id_is_the_number_in_parentheses(self):
    self.assertEqual(ev(0, 1, "jit__bwda(17314385484412105626)").program_id, "17314385484412105626")
    # Two programs under one base name differ only here.
    a, b = ev(0, 1, "jit__bwda(111)"), ev(0, 1, "jit__bwda(222)")
    self.assertEqual(a.base, b.base)
    self.assertNotEqual(a.program_id, b.program_id)

  def test_program_id_absent(self):
    self.assertEqual(ev(0, 1, "copy").program_id, "")
    self.assertEqual(ev(0, 1, "fusion(unterminated").program_id, "")

  def test_end(self):
    self.assertEqual(ev(10, 5).end, int(15 * xprof.PS))


class StepsTest(unittest.TestCase):
  """`steps` is load-bearing: summary, timeline and gaps all slice by it."""

  def test_counts_repeats_of_the_module_sequence(self):
    mods = {d: [ev(d, 100, "f"), ev(1000 + d, 100, "b")] for d in range(4)}
    self.assertEqual(fake(mods).n_steps(), 1)  # 'f','b' does not repeat
    mods = {d: [ev(d, 10, "f"), ev(20 + d, 10, "b"), ev(1000 + d, 10, "f"), ev(1020 + d, 10, "b")] for d in range(4)}
    tr = fake(mods)
    self.assertEqual(tr.n_steps(), 2)
    self.assertEqual(len(tr.steps()), 2)

  def test_a_wide_gap_inside_a_step_does_not_split_it(self):
    # The mbs-256 gpipe failure: the fwd->bwd turnaround leaves all four
    # devices idle for longer than some real step edges, so any threshold
    # under it reports 4 steps for a 2-step capture.
    def step(t):
      return {d: [ev(t + d, 10, "f"), ev(t + 500 + d, 10, "b")] for d in range(4)}

    a, b = step(0), step(2000)
    mods = {d: a[d] + b[d] for d in range(4)}
    tr = fake(mods)
    # Three union gaps, all wide: two turnarounds and one real edge. Every
    # fixed threshold under the turnaround reports 4 steps.
    self.assertEqual(len(tr.union_gaps()), 3)
    self.assertGreater(min(tr.union_gaps()), int(400 * xprof.PS))
    self.assertEqual(tr.n_steps(), 2)
    self.assertEqual(len(tr.steps()), 2)

  def test_the_widest_union_gap_need_not_be_the_boundary(self):
    # The mbs-4096 1f1b failure: a hole in the drain is wider than the edge.
    mods = {
        d: [ev(d, 10, "f"), ev(900 + d, 10, "b"), ev(1000 + d, 10, "f"), ev(1010 + d, 10, "b")]  # 890us hole
        for d in range(4)
    }
    tr = fake(mods)
    steps = tr.steps()
    self.assertEqual(len(steps), 2)
    self.assertGreater(max(tr.union_gaps()), int(800 * xprof.PS))
    # Boundary is between module 1 and 2, not at the 890us hole.
    self.assertLess(steps[0][1], int(915 * xprof.PS))
    self.assertGreaterEqual(steps[1][0], int(1000 * xprof.PS))

  def test_the_step_window_spans_fill_and_drain(self):
    # Device 0 starts first and device 3 ends last; the window covers both, so
    # a stage that goes quiet early is measured as idle, not as finished.
    mods = {0: [ev(0, 10, "f")], 1: [ev(5, 10, "f")], 2: [ev(10, 10, "f")], 3: [ev(15, 10, "f")]}
    ((lo, hi),) = fake(mods).steps()
    self.assertEqual(lo, 0)
    self.assertEqual(hi, int(25 * xprof.PS))

  def test_uneven_module_counts_still_agree_on_a_step_count(self):
    # The last stage fuses fwd+bwd, so it runs fewer modules: 4/4/4/2 here,
    # 64/64/64/48 on a real capture.
    mods = {
        0: [ev(0, 10, "f"), ev(20, 10, "b"), ev(100, 10, "f"), ev(120, 10, "b")],
        1: [ev(5, 10, "a"), ev(105, 10, "a")],
    }
    self.assertEqual(fake(mods).n_steps(), 2)

  def test_a_device_out_of_phase_falls_back_to_one_step(self):
    # If the sequences cannot agree on any k, say one step rather than guess.
    mods = {0: [ev(0, 10, "f"), ev(20, 10, "f")], 1: [ev(0, 10, "f"), ev(20, 10, "b"), ev(40, 10, "f")]}
    self.assertEqual(fake(mods).n_steps(), 1)
    self.assertEqual(len(fake(mods).steps()), 1)

  def test_union_gaps_is_sorted_and_merges_overlaps(self):
    mods = {d: [ev(d, 10), ev(100 + d, 10), ev(1000 + d, 10)] for d in range(2)}
    g = fake(mods).union_gaps()
    self.assertEqual(g, sorted(g, reverse=True))
    self.assertEqual(len(g), 2)  # 4 devices' worth of spans, 2 holes

  def test_union_gaps_ignores_a_span_contained_in_another(self):
    # MPMD devices overlap: a short module on one device sits wholly inside a
    # long one on another. Letting the running end regress onto the short one
    # invents a gap that is not there -- and on the mbs-256 gpipe capture that
    # turns a true largest in-step gap of 130us into a reported 1,097us.
    mods = {0: [ev(0, 100, "f"), ev(200, 10, "f")], 1: [ev(10, 10, "g"), ev(210, 10, "g")]}
    self.assertEqual(fake(mods).union_gaps(), [int(100 * xprof.PS)])

  def test_no_modules(self):
    self.assertEqual(fake({0: []}).steps(), [])
    self.assertEqual(fake({0: []}).union_gaps(), [])
    self.assertEqual(fake({0: []}).n_steps(), 0)

  def test_single_device_capture(self):
    mods = {0: [ev(0, 100, "f"), ev(1000, 100, "f")]}
    self.assertEqual(len(fake(mods).steps()), 2)


class DeviceListTest(unittest.TestCase):

  def test_none_means_all(self):
    tr = fake({0: [], 3: [], 1: []})
    self.assertEqual(tr.device_list(None), [0, 1, 3])

  def test_bad_index_is_a_message_not_a_keyerror(self):
    tr = fake({0: [], 1: []})
    for bad in (9, -1):
      with self.assertRaises(SystemExit) as cm:
        tr.device_list(bad)
      self.assertIn("TPU:0, TPU:1", str(cm.exception))


class DriverBusyTest(unittest.TestCase):
  """The thread's whole life is six outermost frames: `jax.profiler.trace`'s
  `__enter__`, then per step one frame that issues and one that waits, then
  `__exit__`.  Only the frames that issued something count."""

  HOST = [
      ev(-40, 38, "$contextlib.py:132 __enter__"),  # jax.profiler.trace
      ev(-39, 36, "$profiler.py:151 start_trace"),
      ev(0, 20, "$ds_ep_pipeline.py:280 <lambda>"),
      ev(2, 1, "PjitFunction(step)"),
      ev(2.2, 0.5, "PjRtCApiLoadedExecutable::Execute"),  # nested leaf
      ev(20, 80, "$api.py:2751 block_until_ready"),
      ev(100, 20, "$ds_ep_pipeline.py:280 <lambda>"),
      ev(101, 1, "PjitFunction(step)"),
      ev(120, 80, "$api.py:2751 block_until_ready"),
      ev(200, 30, "$contextlib.py:141 __exit__"),
  ]

  def test_only_the_issuing_frames_count(self):
    tr = fake({0: [ev(0, 1)]}, host=self.HOST)
    self.assertEqual(tr.driver_busy(0, int(200 * xprof.PS)), int(40 * xprof.PS))

  def test_clamps_a_straddling_frame_to_the_window(self):
    tr = fake({0: [ev(0, 1)]}, host=self.HOST)
    # [10, 110) holds the back half of one issuing frame and the front half of
    # the next: 10us each, so 20 -- frames are clipped, not dropped.
    self.assertEqual(tr.driver_busy(int(10 * xprof.PS), int(110 * xprof.PS)), int(20 * xprof.PS))

  def test_the_profilers_own_startup_is_not_issuing(self):
    # The window comes from the device planes and this line from the host
    # plane.  Where the clocks are far apart the window opens before the first
    # dispatch, over the tail of `start_trace`.  Measuring the window minus
    # `block_until_ready` charges that as work -- 50us here, and 4.2ms on the
    # real mbs-256 gpipe capture, which is a 38% overstatement of its step 0.
    tr = fake({0: [ev(0, 1)]}, host=self.HOST)
    self.assertEqual(tr.driver_busy(int(-30 * xprof.PS), int(100 * xprof.PS)), int(20 * xprof.PS))

  def test_a_nested_execute_is_not_counted_twice(self):
    # The leaf Execute sits inside the issuing frame; summing leaves instead
    # would both double-count it and miss the transfer sends.
    tr = fake({0: [ev(0, 1)]}, host=self.HOST)
    self.assertEqual(tr.driver_busy(0, int(20 * xprof.PS)), int(20 * xprof.PS))

  def test_the_dispatching_line_is_found_whatever_it_is_called(self):
    # The host plane carries one line per thread and the key is the binary's
    # name, so `python3` is not a thing to hard-code: read the wrong line and
    # the driver block silently vanishes from `summary` rather than erroring.
    tr = fake(
        {0: [ev(0, 1)]}, host={"python3.12": [ev(0, 500, "$threading.py:1 _bootstrap")], "python3.12 (2)": self.HOST}
    )
    self.assertEqual(tr.driver_busy(0, int(200 * xprof.PS)), int(40 * xprof.PS))

  def test_no_host_plane(self):
    tr = fake({0: [ev(0, 1)]})
    tr.host = {}
    self.assertEqual(tr.driver_busy(0, 10**15), 0)


class AllIdleTest(unittest.TestCase):
  """`all_idle` is the measure that answers "is any chip working at all".

  Every other idle number in this file is per device and is dominated by the
  bubble, which is the schedule doing its job.  These holes are the part that
  buys nothing, so they need to be exact: a false one invents a stall, and a
  missed one hides the only idle worth chasing.
  """

  def us(self, holes):
    return [(a / xprof.PS, b / xprof.PS) for a, b, _, _ in holes]

  def test_a_hole_needs_every_device_quiet_at_once(self):
    tr = fake({0: [ev(0, 10), ev(25, 10)], 1: [ev(15, 10)]})
    self.assertEqual(self.us(tr.all_idle(0, int(35 * xprof.PS))), [(10.0, 15.0)])

  def test_work_on_another_device_is_not_a_hole(self):
    # TPU:0 idle from 10 to 15 while TPU:1 runs is the bubble, not waste.
    tr = fake({0: [ev(0, 10)], 1: [ev(5, 10)]})
    self.assertEqual(tr.all_idle(0, int(15 * xprof.PS)), [])

  def test_a_span_contained_in_another_does_not_open_a_hole(self):
    # The cursor has to advance monotonically. If the short inner span reset it
    # backwards, the outer span's own tail would be reported as nobody working.
    tr = fake({0: [ev(0, 50)], 1: [ev(10, 5)]})
    self.assertEqual(tr.all_idle(0, int(50 * xprof.PS)), [])

  def test_the_handoff_is_named_on_both_sides(self):
    tr = fake({0: [ev(0, 10, "jit__bwda(1)")], 1: [ev(20, 10, "jit__bwd(2)")]})
    ((_, _, before, after),) = tr.all_idle(0, int(30 * xprof.PS))
    self.assertEqual((before[0], before[1].base), (0, "jit__bwda"))
    self.assertEqual([(d, m.base) for d, m in after], [(1, "jit__bwd")])

  def test_lead_in_and_tail_are_holes_with_only_one_neighbour(self):
    tr = fake({0: [ev(10, 10)]})
    holes = tr.all_idle(0, int(30 * xprof.PS))
    self.assertEqual(self.us(holes), [(0.0, 10.0), (20.0, 30.0)])
    self.assertIsNone(holes[0][2])
    self.assertEqual(holes[1][3], [])

  def test_modules_outside_the_window_are_clipped_not_dropped(self):
    tr = fake({0: [ev(0, 20), ev(30, 20)]})
    self.assertEqual(self.us(tr.all_idle(int(10 * xprof.PS), int(40 * xprof.PS))), [(20.0, 30.0)])


class FramingTest(unittest.TestCase):

  HOST = [
      ev(50, 50, "$contextlib.py:132 __enter__"),
      ev(100, 10, "$ds_ep_pipeline.py:280 <lambda>"),
      ev(101, 5, "PjitFunction(step)"),
      ev(200, 500, "$contextlib.py:141 __exit__"),
  ]

  def test_profiler_start_up_and_teardown_are_split_from_the_work(self):
    tr = fake({0: [ev(100, 100)]}, host=self.HOST)
    up, lo, hi, down, frames = tr.framing()
    self.assertEqual((up / xprof.PS, down / xprof.PS), (50.0, 500.0))
    self.assertEqual((lo / xprof.PS, hi / xprof.PS), (100.0, 200.0))
    # The PjitFunction is inside the lambda, so it is not an outermost frame.
    self.assertEqual([f.name.split()[-1] for f in frames], ["__enter__", "<lambda>", "__exit__"])

  def test_the_dispatching_line_is_framed_whatever_it_is_called(self):
    # A host plane carries one line per thread and `python3` is only the
    # binary's name.  Framing the wrong thread reports some worker's lifetime
    # as the profiler's start-up.
    tr = fake(
        {0: [ev(100, 100)]},
        host={"python3.12": [ev(0, 900, "$contextlib.py:132 __enter__")], "python3.12 (2)": self.HOST},
    )
    up, _, _, down, _ = tr.framing()
    self.assertEqual((up / xprof.PS, down / xprof.PS), (50.0, 500.0))

  def test_a_capture_with_no_host_plane_answers_rather_than_raising(self):
    tr = fake({0: [ev(0, 10)]})
    tr.host = {}
    self.assertEqual(tr.framing(), (0, None, None, 0, []))


class MergedTest(unittest.TestCase):
  """Op intervals overlap; their union is the one occupancy measure that means
  the same thing on an MPMD and an SPMD capture."""

  def test_overlaps_are_counted_once(self):
    self.assertEqual(xprof.merged([ev(0, 100, "while"), ev(5, 40), ev(50, 40)]), int(100 * xprof.PS))

  def test_a_hole_is_not_covered(self):
    self.assertEqual(xprof.merged([ev(0, 10), ev(30, 10)]), int(20 * xprof.PS))

  def test_abutting_intervals_merge(self):
    # [0,10) abuts [10,20), which partially overlaps [15,25): union is 25us.
    # The abutting boundary on its own is unobservable in a scalar union --
    # fusing the pair and keeping it split give the same total -- so the third
    # interval is what makes this assertion able to fail. Partial overlap is
    # the case `test_overlaps_are_counted_once` misses; it only nests.
    self.assertEqual(xprof.merged([ev(0, 10), ev(10, 10), ev(15, 10)]), int(25 * xprof.PS))

  def test_empty(self):
    self.assertEqual(xprof.merged([]), 0)


class ChildrenTest(unittest.TestCase):
  """The containment stack: a `while` must be charged once, not twice."""

  def test_nested_ops_are_not_double_counted(self):
    ops = [ev(0, 100, "while"), ev(5, 40, "fusion"), ev(50, 40, "fusion"), ev(100, 5, "copy")]
    kids = xprof.children(ops, 0, int(110 * xprof.PS))
    self.assertEqual([k.base for k in kids], ["while", "copy"])
    self.assertEqual(sum(k.dur for k in kids), int(105 * xprof.PS))

  def test_siblings_are_all_kept(self):
    ops = [ev(0, 30, "a"), ev(30, 30, "b"), ev(60, 30, "c")]
    kids = xprof.children(ops, 0, int(90 * xprof.PS))
    self.assertEqual([k.base for k in kids], ["a", "b", "c"])

  def test_an_op_may_span_the_whole_module(self):
    # No module in any captured trace has an op exactly spanning it, so there
    # is no module-self entry to exclude -- and excluding one would zero the
    # coverage of a single-op module.
    ops = [ev(0, 100, "jit__bwda")]
    kids = xprof.children(ops, 0, int(100 * xprof.PS))
    self.assertEqual([k.base for k in kids], ["jit__bwda"])

  def test_zero_duration_ops_are_dropped(self):
    # Past the cursor, so the duration guard is the only clause that can reject
    # it: behind an earlier op the greedy `e.start >= cur` rejects it anyway and
    # the test passes with the guard deleted.
    ops = [ev(0, 50, "a"), ev(60, 0, "custom-call")]  # sorted by (start, -dur)
    self.assertEqual([k.base for k in xprof.children(ops, 0, int(70 * xprof.PS))], ["a"])

  def test_ops_outside_the_window_are_ignored(self):
    ops = [ev(0, 10, "before"), ev(20, 10, "inside"), ev(100, 10, "after")]
    kids = xprof.children(ops, int(15 * xprof.PS), int(40 * xprof.PS))
    self.assertEqual([k.base for k in kids], ["inside"])

  def test_partial_overlap_keeps_the_first_only(self):
    # Not nesting and not disjoint; the greedy pass must not count both.
    ops = [ev(0, 60, "a"), ev(50, 60, "b")]
    kids = xprof.children(ops, 0, int(110 * xprof.PS))
    self.assertEqual([k.base for k in kids], ["a"])

  def test_coverage_never_exceeds_the_module(self):
    ops = [ev(0, 100, "while"), ev(0, 100, "dup"), ev(5, 40, "f"), ev(50, 40, "g"), ev(10, 10, "h")]
    kids = xprof.children(ops, 0, int(100 * xprof.PS))
    self.assertLessEqual(sum(k.dur for k in kids), int(100 * xprof.PS))


class UncontendedTest(unittest.TestCase):

  def test_empty(self):
    self.assertEqual(xprof.uncontended([]), (0, 0.0, 0.0, 0.0))

  def test_median_survives_an_outlier_the_mean_does_not(self):
    n, mean, med, p10 = xprof.uncontended([10.0] * 9 + [1000.0])
    self.assertEqual(n, 10)
    self.assertEqual(med, 10.0)
    self.assertGreater(mean, 100.0)
    self.assertEqual(p10, 10.0)

  def test_p10_is_the_nearest_rank_percentile(self):
    for n in range(1, 50):
      _, _, _, p10 = xprof.uncontended([float(i) for i in range(n)])
      self.assertEqual(p10, float(-(-n // 10) - 1), f"n={n}")

  def test_p10_is_not_the_minimum_for_the_sample_sizes_we_have(self):
    # 14 and 28 instances per module is what these captures give; truncating
    # instead of rounding up made p10 the minimum for every n <= 19.
    for n in (14, 28):
      _, _, _, p10 = xprof.uncontended([float(i) for i in range(n)])
      self.assertGreater(p10, 0.0, f"n={n}")

  def test_single_sample(self):
    self.assertEqual(xprof.uncontended([7.0]), (1, 7.0, 7.0, 7.0))

  def test_the_input_need_not_arrive_sorted(self):
    # p10 indexes into the sample and module durations arrive in trace order,
    # so dropping the sort would pick an arbitrary instance -- on the real
    # 1f1b TPU:2 backward it picks 2628us, above the median, when the answer
    # is 1923us.  Every other input in this class is already ascending.
    self.assertEqual(xprof.uncontended([5.0, 4.0, 3.0, 2.0, 1.0]), (5, 3.0, 3.0, 1.0))


class ClassifierTest(unittest.TestCase):
  """Every module name that actually occurs must classify, not fall through."""

  REAL_NAMES = ["jit_fwd_raw(8006710057128423225)", "jit__bwda(123)", "jit__bwd(456)", "jit__fba(789)", "jit__fb(101)"]

  def test_char_covers_every_real_module_name(self):
    for nm in self.REAL_NAMES:
      self.assertNotEqual(xprof._char(nm), "o", nm)

  def test_bwda_is_not_read_as_bwd(self):
    # '_bwd' is a prefix of '_bwda'; order in CHARS decides, so pin it.
    self.assertEqual(xprof._char("jit__bwda(1)"), "B")
    self.assertEqual(xprof._char("jit__bwd(1)"), "b")
    self.assertEqual(xprof._kind("jit__bwda(1)"), "bwda")
    self.assertEqual(xprof._kind("jit__bwd(1)"), "bwd")

  def test_fba_is_not_read_as_fb(self):
    self.assertEqual(xprof._kind("jit__fba(1)"), "fba")
    self.assertEqual(xprof._kind("jit__fb(1)"), "fb")

  def test_unknown_falls_through_visibly(self):
    self.assertEqual(xprof._char("jit_something_new(1)"), "o")

  def test_split_backward_is_not_read_as_bwd(self):
    # the zero-bubble schedules' halves: '_bwd' is a prefix of '_bwdi' and '_bwdw', '_bwdw' of '_bwdwa'
    for name, ch, kind in (
        ("jit_bwdi_c1(1)", "I", "bwdi"),
        ("jit_bwdw_c1(1)", "W", "bwdw"),
        ("jit_bwdwa_c1(1)", "W", "bwdwa"),
        ("jit_bwdwa_c2_rsself(1)", "W", "bwdwa"),
        ("jit_bwda_c1(1)", "B", "bwda"),
        # the split last chunk's AI (--zb-split-last): '_fb' is a prefix of '_fbi', '_fba' of '_fbai'
        ("jit_fbi_c3(1)", "A", "fbi"),
        ("jit_fbai_c3(1)", "A", "fbai"),
        ("jit_fba_c3(1)", "A", "fba"),
    ):
      self.assertEqual((xprof._char(name), xprof._kind(name)), (ch, kind), name)


class ProgramCostsTest(unittest.TestCase):
  """`program_costs` (`xprof costs`): per-chunk program medians as a --costs table, the keys sched.cost_table reads."""

  @staticmethod
  def ms(start, dur, name):
    return ev(start * 1000, dur * 1000, name)

  def test_zero_bubble_capture(self):
    """A four-stage zbh1 step: chunk 0's backwards whole, chunks 1 and 2 split -- I<c> the median `bwdi_c<c>`, W<c>
    the median plain `bwdwa_c<c>` (the first microbatch's `bwdw_c<c>` and the reduce-scatter host's `_rsself` left
    out of it; the host's extra goes into dW_rs), B<c>, which a split chunk never runs, their sum -- and the fused
    last chunk's A."""
    m = self.ms
    mods = {
        0: [
            m(0, 10, "jit_fwd_c0(1)"),
            m(10, 30, "jit_bwd_c0(2)"),
            m(40, 32, "jit_bwda_c0(3)"),
            m(80, 32, "jit_bwda_c0(3)"),
        ],
        1: [
            m(0, 11, "jit_fwd_c1(4)"),
            m(11, 20, "jit_bwdi_c1(5)"),
            m(31, 8, "jit_bwdw_c1(6)"),
            m(40, 22, "jit_bwdi_c1(5)"),
            m(62, 9, "jit_bwdwa_c1(7)"),
            m(71, 21, "jit_bwdi_c1(5)"),
            m(92, 12, "jit_bwdwa_c1_rsself(8)"),
        ],
        2: [m(0, 12, "jit_fwd_c2(9)"), m(12, 24, "jit_bwdi_c2(10)"), m(36, 10, "jit_bwdw_c2(11)")],
        3: [m(0, 50, "jit_fb_c3(12)"), m(50, 52, "jit_fba_c3(13)")],
    }
    got = xprof.program_costs(fake(mods), sorted(mods))
    want = {"F0": 10.0, "F1": 11.0, "F2": 12.0, "B0": 32.0, "I1": 21.0, "W1": 9.0, "B1": 30.0, "I2": 24.0}
    want.update({"W2": 10.0, "B2": 34.0, "A": 52.0, "F": 11.5, "B": 32.0, "I": 22.5, "W": 9.5, "dW_rs": 3.0})
    self.assertEqual(got, want)

  def test_split_last_capture(self):
    """zbh1 at two stages with the last chunk split (--zb-split-last): AI the median plain `fbai_c<last>`, its W --
    W<last> and WA -- the median plain `bwdwa_c<last>`, and A, which it never runs, their sum; the last chunk's W out
    of the middle chunks' W median."""
    m = self.ms
    mods = {
        0: [m(0, 10, "jit_fwd_c0(1)"), m(10, 30, "jit_bwd_c0(2)"), m(40, 32, "jit_bwda_c0(3)")],
        1: [
            m(0, 40, "jit_fbi_c1(4)"),
            m(40, 10, "jit_bwdw_c1(5)"),
            m(50, 42, "jit_fbai_c1(6)"),
            m(92, 12, "jit_bwdwa_c1(7)"),
            m(104, 44, "jit_fbai_c1(6)"),
            m(148, 14, "jit_bwdwa_c1(7)"),
        ],
    }
    got = xprof.program_costs(fake(mods), sorted(mods))
    want = {"F0": 10.0, "B0": 32.0, "AI": 43.0, "W1": 13.0, "WA": 13.0, "A": 56.0}
    self.assertEqual(got, want)


class FindXplaneTest(unittest.TestCase):

  def test_picks_the_newest_of_several_captures(self):
    # `jax.profiler.trace` writes a fresh `plugins/profile/<ts>/` sibling on
    # every run, so re-profiling into one directory leaves several behind.
    # Silently taking the oldest makes a second run look like a no-op.
    with tempfile.TemporaryDirectory() as root:
      paths = []
      for i, ts in enumerate((100, 300, 200)):
        d = os.path.join(root, "plugins", "profile", f"run{i}")
        os.makedirs(d)
        p = os.path.join(d, "host.xplane.pb")
        with open(p, "wb") as f:
          f.write(b"")
        os.utime(p, (ts, ts))
        paths.append(p)
      self.assertEqual(xprof.find_xplane(root), paths[1])

  def test_a_bracketed_directory_is_a_path_not_a_pattern(self):
    # A capture directory's name may carry brackets: DIR/mpmd-<sched>-<label>, e.g. a label hybrid[noguard].
    with tempfile.TemporaryDirectory() as root:
      d = os.path.join(root, "mpmd-1f1b-hybrid[noguard]", "plugins", "profile", "run0")
      os.makedirs(d)
      p = os.path.join(d, "host.xplane.pb")
      with open(p, "wb") as f:
        f.write(b"")
      self.assertEqual(xprof.find_xplane(os.path.join(root, "mpmd-1f1b-hybrid[noguard]")), p)

  def test_a_pathways_session_takes_the_file_with_device_planes(self):
    """A Pathways capture (ds_ep_pipeline --profile gs://... under JAX_PLATFORMS=proxy, copied here) is one session
    directory of several files: the client's own (<host>.xplane.pb, client.<host>.xplane.pb), the worker host's with
    the device planes (server-<host>.0.0.xplane.pb) and a small one of the resource manager's -- whichever is
    newest, the worker's is read; an older capture's files stay out of it."""
    with tempfile.TemporaryDirectory() as root:
      old = os.path.join(root, "plugins", "profile", "2026_11_25_09_00_00")
      new = os.path.join(root, "plugins", "profile", "2026_11_25_10_00_00")
      for d in (old, new):
        os.makedirs(d)
      files = {
          (old, "server-vm.0.0.xplane.pb"): (b"..../device:TPU:0....", 100),
          (new, "server-vm.0.0.xplane.pb"): (b"..../device:TPU:0..../device:TPU:7....", 200),
          (new, "vm.xplane.pb"): (b"..../host:CPU....", 300),
          (new, "client.vm.xplane.pb"): (b"..../host:CPU....", 301),
          (new, "server-vm.xplane.pb"): (b"", 302),
      }
      for (d, name), (data, ts) in files.items():
        p = os.path.join(d, name)
        with open(p, "wb") as f:
          f.write(data)
        os.utime(p, (ts, ts))
      err = io.StringIO()
      with contextlib.redirect_stderr(err):
        self.assertEqual(xprof.find_xplane(root), os.path.join(new, "server-vm.0.0.xplane.pb"))
      self.assertIn("using the newest session's file with device planes", err.getvalue())
      self.assertTrue(xprof.has_device_planes(os.path.join(new, "server-vm.0.0.xplane.pb")))
      self.assertFalse(xprof.has_device_planes(os.path.join(new, "vm.xplane.pb")))
      self.assertFalse(xprof.has_device_planes(os.path.join(new, "server-vm.xplane.pb")))
      # a capture of the client alone (no device plane anywhere): the newest file, as ever
      for name in ("server-vm.0.0.xplane.pb",):
        os.remove(os.path.join(new, name))
      with contextlib.redirect_stderr(io.StringIO()):
        self.assertEqual(xprof.find_xplane(new), os.path.join(new, "server-vm.xplane.pb"))

  def test_sessions_go_by_their_names_not_mtimes(self):
    """A logdir copied from gs:// (gcloud storage cp; ds_ep_pipeline's Pathways --profile goes there) gives each file
    the time it was downloaded, not the time it was captured: where every capture directory is named as a session
    (YYYY_MM_DD_HH_MM_SS, a local time: what XProf sorts by), the session named last is the newest, whatever the
    mtimes.  Here the older session finished downloading last."""
    with tempfile.TemporaryDirectory() as root:
      old = os.path.join(root, "plugins", "profile", "2026_11_25_09_00_00")
      new = os.path.join(root, "plugins", "profile", "2026_11_25_10_00_00")
      files = {
          (old, "server-vm.0.0.xplane.pb"): (b"..../device:TPU:0....", 400),
          (old, "vm.xplane.pb"): (b"..../host:CPU....", 401),
          (new, "server-vm.0.0.xplane.pb"): (b"..../device:TPU:0....", 200),
          (new, "vm.xplane.pb"): (b"..../host:CPU....", 300),
      }
      for (d, name), (data, ts) in files.items():
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        with open(p, "wb") as f:
          f.write(data)
        os.utime(p, (ts, ts))
      err = io.StringIO()
      with contextlib.redirect_stderr(err):
        self.assertEqual(xprof.find_xplane(root), os.path.join(new, "server-vm.0.0.xplane.pb"))
      self.assertIn("using the newest session's file with device planes (sessions by name", err.getvalue())
      for name in ("server-vm.0.0.xplane.pb", "vm.xplane.pb"):  # one file a session
        os.remove(os.path.join(old if name == "vm.xplane.pb" else new, name))
      with contextlib.redirect_stderr(io.StringIO()):
        self.assertEqual(xprof.find_xplane(root), os.path.join(new, "vm.xplane.pb"))

  def test_sessions_named_off_utc_sort_in_time(self):
    """Every session is named in the local time of the host that names it (SESSION_NAME): a one-process one by jax,
    and ds_ep_pipeline's Pathways and multi-controller ones by pathways.session_id (pathways_test holds
    shared_profile_session to it).  So on a host off UTC the session named last is still the newest, either way round:
    in Los Angeles a --distributed capture at 22:00 UTC, then a one-process one at 22:30; in Tokyo the other way round.
    (The multi-controller session was named in UTC: in Los Angeles 2026_11_26_22_00_00, which sorted after the later
    one-process session's 2026_11_26_14_30_00, and the older capture was read.)"""
    t0 = 1795730400  # 2026-11-26 22:00:00 UTC
    distributed = ("s0-p0-vm.xplane.pb", pathways.session_id)  # a traced process's file, in stage order
    one_process = ("vm.xplane.pb", lambda t: time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(t)))  # jax's name
    saved = os.environ.get("TZ")
    try:
      for zone, runs in (("America/Los_Angeles", (distributed, one_process)), ("Asia/Tokyo", (one_process, distributed))):
        os.environ["TZ"] = zone
        time.tzset()
        if time.localtime(t0)[:6] == time.gmtime(t0)[:6]:
          self.skipTest(f"no time zone database: TZ={zone} gives UTC")
        with self.subTest(zone=zone), tempfile.TemporaryDirectory() as root:
          paths = []
          for i, (name, session) in enumerate(runs):
            t = t0 + 1800 * i
            d = os.path.join(root, "plugins", "profile", session(t))
            os.makedirs(d)
            p = os.path.join(d, name)
            with open(p, "wb") as f:
              f.write(b"..../device:TPU:0....")
            os.utime(p, (t, t))
            paths.append(p)
          with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(xprof.find_xplane(root), paths[1])
    finally:
      if saved is None:
        os.environ.pop("TZ", None)
      else:
        os.environ["TZ"] = saved
      time.tzset()

  def test_a_directory_with_no_capture_is_a_message(self):
    with tempfile.TemporaryDirectory() as root:
      with self.assertRaises(SystemExit):
        xprof.find_xplane(root)

  def test_a_file_is_taken_as_given(self):
    with tempfile.NamedTemporaryFile() as f:
      self.assertEqual(xprof.find_xplane(f.name), f.name)


class CommandTest(unittest.TestCase):
  """The subcommands themselves.  Everything above was a helper; these are
  what a reader actually runs, and they are where the arithmetic that a
  write-up quotes from them lives.

  The fixture is a two-step pipeline in miniature: TPU:0 runs the whole step,
  TPU:3 starts 40us late and finishes 40us early, so it has a real lead-in and
  a real drain and its two occupancy conventions must disagree.
  """

  def tr(self):
    mods = {
        0: [
            ev(0, 40, "jit_fwd_raw(1)"),
            ev(60, 40, "jit__bwda(1)"),
            ev(200, 40, "jit_fwd_raw(1)"),
            ev(260, 40, "jit__bwda(1)"),
        ],
        3: [ev(40, 20, "jit__fba(4)"), ev(240, 20, "jit__fba(4)")],
    }
    ops = {
        0: [
            ev(60, 40, "while.1"),
            ev(62, 10, "fusion.2"),
            ev(72, 20, "fusion.3"),
            ev(260, 40, "while.1"),
            ev(262, 10, "fusion.2"),
            ev(272, 20, "fusion.3"),
        ],
        3: [ev(40, 20, "fusion.9"), ev(240, 20, "fusion.9")],
    }
    return fake(mods, ops_by_dev=ops)

  def test_summary_separates_the_two_occupancy_conventions(self):
    out = run(xprof.cmd_summary, self.tr())
    self.assertIn("2 step(s)", out)
    # TPU:3 is busy 20 of its own 20us window but 20 of the step's 100us.
    self.assertIn("100.0%", out)
    self.assertIn("20.0%", out)
    # The two whole-capture means a write-up quotes. Busy 160us of a 300us
    # common clock on TPU:0 and 40us of it on TPU:3 -> 33.3%; measured on
    # their own windows (300us, 220us) -> 35.8%. Computing the first on the
    # per-device span collapses them into one number.
    self.assertIn("common clock        33.3% busy, 66.7% idle", out)
    self.assertIn("per-device window   35.8% busy, 64.2% idle", out)
    # Op occupancy is the UNION: TPU:0's three ops nest into 40us of wall per
    # step, not the 70us their durations sum to.
    self.assertIn("  TPU:0      0.08ms busy    30.8%  (occ/own 33.3% of its own " "0.24 ms)", out)

  def test_summary_survives_a_device_that_ran_nothing(self):
    tr = self.tr()
    tr.devices[1] = {"XLA Modules": [], "XLA Ops": []}
    out = run(xprof.cmd_summary, tr)  # must not raise
    # and the mean is over the two devices that contributed, not over 3 planes
    self.assertIn("common clock        33.3% busy, 66.7% idle", out)

  def test_summary_does_not_call_the_boundary_a_decoy(self):
    # The boundary IS a union gap. Leaving it in the pool makes the step edge
    # report as a decoy inside a step -- on the mbs-256 gpipe capture, as a
    # 485us hole that a threshold "would have split at and been wrong".
    tr = self.tr()
    tr.devices[3]["XLA Modules"] = [ev(42, 18, "jit__fba(4)"), ev(242, 18, "jit__fba(4)")]
    out = run(xprof.cmd_summary, tr)
    self.assertIn("the other 2 union gaps are all narrower", out)
    self.assertNotIn("INSIDE a step", out)

  def test_summary_flags_an_in_step_gap_as_wide_as_the_boundary(self):
    # The other half of the same claim: a 490us turnaround inside step 0 is
    # wider than the 90us boundary, which is exactly why no threshold works.
    mods = {0: [ev(0, 10, "f"), ev(500, 10, "b"), ev(600, 10, "f"), ev(610, 10, "b")]}
    out = run(xprof.cmd_summary, fake(mods))
    self.assertIn("1 gap(s) INSIDE a step", out)
    self.assertIn("490us", out)

  def test_summary_warns_that_a_1_step_capture_may_be_truncated(self):
    out = run(xprof.cmd_summary, fake({0: [ev(0, 10, "f"), ev(500, 10, "b")]}))
    self.assertIn("only 1 step", out)
    self.assertIn("490.0us", out)

  def test_gaps_charges_the_lead_in_and_the_drain(self):
    out = run(xprof.cmd_gaps, self.tr())
    self.assertIn("lead-in", out)
    self.assertIn("drain", out)

  def test_gaps_totals_reconcile_with_the_step_windows(self):
    # The bug this replaced: counting only the gaps *between* a device's own
    # modules measures idle in that device's own window, so the stage with
    # the worst drain scores as the one with the least idle.
    out = run(xprof.cmd_gaps, self.tr())
    tr = self.tr()
    for d in (0, 3):
      idle = sum((hi - lo) - sum(m.dur for m in tr.modules(d) if lo <= m.start and m.end <= hi) for lo, hi in tr.steps())
      row = [l for l in out.splitlines() if l.startswith(f"TPU:{d}")]
      self.assertTrue(row, f"no TPU:{d} row")
      self.assertIn(f"{idle / xprof.PS_MS:.2f}", row[0])

  def test_the_pipeline_reading_is_dropped_on_an_spmd_capture(self):
    # This tool gets pointed at the SPMD baseline too, and there "step lead-in
    # (waiting for the fill)" and "that spread IS the bubble" are not wrong so
    # much as meaningless -- one program replicated has no fill and no stage 0
    # to wait for. Names cannot make the distinction, because MaxText's TPU:0
    # additionally runs `jit__multi_slice`; a shared program id can.
    spmd = fake({d: [ev(0, 100, "jit_train_step(7)"), ev(200, 100, "jit_train_step(7)")] for d in range(2)})
    self.assertFalse(spmd.is_mpmd())
    self.assertTrue(self.tr().is_mpmd())
    self.assertNotIn("IS the bubble", run(xprof.cmd_summary, spmd))
    self.assertIn("IS the bubble", run(xprof.cmd_summary, self.tr()))
    self.assertNotIn("waiting for the fill", run(xprof.cmd_gaps, spmd))
    self.assertIn("waiting for the fill", run(xprof.cmd_gaps, self.tr()))

  def test_gaps_omits_zero_length_buckets(self):
    # TPU:0 has no lead-in and no drain; printing them as 0.00ms rows would
    # bury the one bucket that carries its idle.
    self.assertNotIn("0.00ms", run(xprof.cmd_gaps, self.tr()))

  # Two devices, two steps, one 5 us hole per step where neither is running,
  # and 55 us between the steps.  Every `idle` number below is checkable by
  # hand off these six numbers, which is the point of building it this way.
  IDLE = dict(
      mods_by_dev={
          0: [ev(0, 10, "jit_fwd_raw"), ev(25, 10, "jit__bwda"), ev(100, 10, "jit_fwd_raw"), ev(125, 10, "jit__bwda")],
          1: [ev(15, 10, "jit_fwd_raw"), ev(35, 10, "jit__bwda"), ev(115, 10, "jit_fwd_raw"), ev(135, 10, "jit__bwda")],
      }
  )

  def test_idle_counts_every_hole_but_lists_only_the_wide_ones(self):
    out = run(xprof.cmd_idle, fake(**self.IDLE), min_us=20)
    self.assertIn("nobody working for 0.005 ms (11.11%) in 1 hole(s)", out)
    self.assertNotIn("-> TPU:1", out)  # the 5 us hole is under the threshold
    wide = run(xprof.cmd_idle, fake(**self.IDLE), min_us=1)
    self.assertIn("TPU:0 fwd", wide)
    self.assertIn("-> TPU:1 fwd", wide)

  def test_idle_reports_the_step_boundary_apart_from_the_steps(self):
    # 5 us of hole in each of two steps, and 55 us between them: lumping the
    # boundary in with the steps would triple the in-step figure and hide that
    # the boundary is a host round trip rather than anything the pipeline did.
    out = run(xprof.cmd_idle, fake(**self.IDLE))
    self.assertIn("between steps: 0.055 ms over 1 boundary(ies), 55.0 us each", out)
    self.assertIn("0.065 ms = 44.83%", out)
    self.assertIn("0.010 ms inside steps, 0.055 ms between them", out)

  def test_idle_charges_the_profiler_rather_than_the_pipeline(self):
    tr = fake(
        {0: [ev(100, 100, "jit_fwd_raw")]},
        host=[ev(50, 50, "$contextlib.py:132 __enter__"), ev(200, 500, "$contextlib.py:141 __exit__")],
    )
    out = run(xprof.cmd_idle, tr)
    self.assertIn("0.65 ms of file, of which 0.10 ms (15.4%)", out)
    self.assertIn("profiler start-up", out)
    self.assertIn("profiler teardown, serializing the capture", out)
    self.assertIn("0.55 ms of that is profiler overhead", out)

  def test_idle_does_not_present_the_bubble_as_waste(self):
    # The two devices never idle simultaneously, so the honest answer is zero
    # -- even though each of them is idle for half the step.
    tr = fake(
        {0: [ev(0, 10, "jit_fwd_raw"), ev(20, 10, "jit__bwda")], 1: [ev(10, 10, "jit_fwd_raw"), ev(30, 10, "jit__bwda")]}
    )
    out = run(xprof.cmd_idle, tr)
    self.assertIn("nobody working for 0.000 ms (0.00%) in 0 hole(s)", out)
    self.assertIn("the only idle worth calling waste", out)

  def test_ops_lists_the_body_of_the_while_not_the_while_again(self):
    # `children` keeps whatever the window contains, and for this call the
    # window *is* the `while`, so it matches itself first and shadows its own
    # body.  The listing was a verbatim duplicate of the top-level row.
    out = run(xprof.cmd_ops, self.tr(), device=0, module="jit__bwda")
    after = out.split("inside the `while`")[1].splitlines()[1:]
    # Count and duration too, not just the name: both body children share the
    # base `fusion`, so a listing truncated to its first child still reads
    # ['fusion'] -- and that is the whole content of the repair.
    rows = [l.split()[:3] for l in after if l.startswith("     ")]
    self.assertEqual(rows, [["fusion", "2.0", "30.0"]])

  def test_ops_body_is_not_shadowed_by_a_twin_at_the_while_bounds(self):
    # Why the body is filtered by span and not by identity. No capture has two
    # ops at identical bounds, so only a test can hold this: filtering `o is
    # not k` leaves the twin, which spans the window and shadows the body.
    tr = self.tr()
    tr.devices[0]["XLA Ops"] = sorted(
        tr.devices[0]["XLA Ops"] + [ev(60, 40, "copy.7"), ev(260, 40, "copy.7")], key=lambda e: (e.start, -e.dur)
    )
    out = run(xprof.cmd_ops, tr, device=0, module="jit__bwda")
    after = out.split("inside the `while`")[1].splitlines()[1:]
    rows = [l.split()[:3] for l in after if l.startswith("     ")]
    self.assertEqual(rows, [["fusion", "2.0", "30.0"]])

  def test_ops_arithmetic_is_per_module_instance(self):
    # `tot` is the denominator of the mean, of the coverage line and of every
    # share -- and the shares are what a write-up quotes. Bounding coverage at
    # <= 100% is one-sided: halving `tot` shows up, doubling it does not.
    out = run(xprof.cmd_ops, self.tr(), device=0, module="jit__bwda")
    self.assertIn("mean module 40.0 us", out)
    self.assertIn("top-level ops cover 100.00%", out)
    row = [l.split() for l in out.splitlines() if l.startswith("  while")][0]
    self.assertEqual(row, ["while", "1.0", "40.0", "100.0%"])

  def test_ops_suppresses_only_the_sub_tenth_microsecond_rows(self):
    tr = fake(
        {0: [ev(0, 40, "jit__bwda(1)")]},
        ops_by_dev={0: [ev(0, 39.9, "big.1"), ev(39.9, 0.04, "tiny.2"), ev(39.94, 0.06, "small.3")]},
    )
    out = run(xprof.cmd_ops, tr, device=0, module="jit__bwda")
    self.assertIn("big", out)
    self.assertIn("small", out)
    self.assertNotIn("tiny", out)

  def test_ops_prefers_an_exact_name_over_a_longer_one_containing_it(self):
    # `-m jit__bwd` as a substring also catches `jit__bwda`, and on the search
    # capture pools 14 instances at 1151.8us with 98 at 2015.5us into a blended
    # 1907.6us that is neither. `-m bwd`, in both names, is refused with both
    # listed; a bare program id, in one name only, still falls back to it.
    tr = self.tr()
    tr.devices[0]["XLA Modules"].append(ev(400, 100, "jit__bwd(9)"))
    tr.devices[0]["XLA Ops"].append(ev(400, 100, "while.1"))
    self.assertIn("jit__bwd  n=1  mean module 100.0 us", run(xprof.cmd_ops, tr, device=0, module="jit__bwd"))
    self.assertIn("jit__bwda  n=2  mean module 40.0 us", run(xprof.cmd_ops, tr, device=0, module="jit__bwda"))
    with self.assertRaises(SystemExit) as cm:
      run(xprof.cmd_ops, tr, device=0, module="bwd")
    self.assertEqual([l.split() for l in str(cm.exception).splitlines()[1:]], [["1x", "jit__bwd"], ["2x", "jit__bwda"]])
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      self.assertIn("9  n=1  mean module 100.0 us", run(xprof.cmd_ops, tr, device=0, module="9"))
    self.assertIn("reading jit__bwd,", err.getvalue())

  def test_ops_on_a_module_that_is_not_there_is_a_message(self):
    with self.assertRaises(SystemExit):
      run(xprof.cmd_ops, self.tr(), module="jit_nonesuch")

  def _rows(self, out):
    """The rendered bars, keyed by device -- the legend guarantees the rest."""
    return {l.split()[0]: l.split("|")[1] for l in out.splitlines() if l.startswith("TPU:")}

  def test_timeline_renders_the_last_module_flush_against_the_edge(self):
    # TPU:0's last module ends exactly at `hi`, which is what makes the right
    # edge an off-by-one: the column index reaches `width` itself.
    out = run(xprof.cmd_timeline, self.tr(), step=0, width=10)
    self.assertEqual(self._rows(out), {"TPU:0": "FFFFF BBBB", "TPU:3": "    AAA   "})

  def test_timeline_defaults_to_the_last_step(self):
    # `--step -1` is the argparse default and the invocation the module
    # docstring shows, and nothing else takes that branch.
    out = run(xprof.cmd_timeline, self.tr(), width=10)
    self.assertIn("step 1 of 2", out)
    self.assertEqual(self._rows(out), {"TPU:0": "FFFFF BBBB", "TPU:3": "    AAA   "})

  def test_timeline_ignores_the_neighbouring_steps_modules(self):
    # The previous step's last module ends exactly where this window opens.
    # Unclamped it maps to columns -1..0, and `max(0, ...)` then pulls the
    # phantom into column 0 -- three of the four rows of the maxtext capture.
    mods = {
        0: [
            ev(0, 10, "jit_fwd_raw(1)"),
            ev(20, 75, "jit__bwda(1)"),
            ev(100, 10, "jit_fwd_raw(1)"),
            ev(120, 80, "jit__bwda(1)"),
        ],
        1: [ev(95, 5, "jit__fba(2)"), ev(195, 5, "jit__fba(2)")],
    }
    out = run(xprof.cmd_timeline, fake(mods), step=1, width=10)
    self.assertEqual(self._rows(out)["TPU:1"], "         A")

  def test_timeline_rejects_a_zero_width(self):
    with self.assertRaises(SystemExit):
      run(xprof.cmd_timeline, self.tr(), width=0)

  def test_timeline_rejects_a_step_that_is_not_there(self):
    with self.assertRaises(SystemExit):
      run(xprof.cmd_timeline, self.tr(), step=5)

  def test_modules_pins_the_row_it_is_quoted_for(self):
    # A write-up quotes n / total / mean / median / p10 off a row of this
    # table, and the mean-vs-median gap is the whole point of the sentence it
    # quotes them in. Its own fixture, because the class fixture's uniform
    # 40us durations cannot separate the three statistics and its trace order
    # already equals sort order.
    tr = fake(
        {
            0: [
                ev(0, 5, "jit_fwd_raw(1)"),
                ev(10, 500, "jit__bwda(2)"),
                ev(600, 10, "jit__bwda(1)"),
                ev(620, 20, "jit__bwda(1)"),
                ev(650, 30, "jit__bwda(1)"),
                ev(690, 40, "jit__bwda(1)"),
                ev(740, 1000, "jit__bwda(1)"),
            ]
        }
    )
    out = run(xprof.cmd_modules, tr, device=0)
    rows = [l.split() for l in out.splitlines() if l.strip().startswith("jit")]
    # Sorted by total, and the program id survives: two chunks, two rows.
    self.assertEqual([r[0] for r in rows], ["jit__bwda(1)", "jit__bwda(2)", "jit_fwd_raw(1)"])
    self.assertEqual(rows[0][1:], ["5", "1.10ms", "220.0us", "30.0us", "10.0us"])

  def test_modules_reports_per_device(self):
    out = run(xprof.cmd_modules, self.tr(), device=0)
    self.assertIn("jit__bwda", out)

  def test_roofline_scores_the_module_against_the_peaks(self):
    # Two 40us jit__bwda on TPU:0, so the mean is 40us by construction, and
    # 1e9 FLOP over 1e6 B is AI 1000 -- above the 578 ridge, so the floor is
    # the compute term, max(1e9/946.7e12, 1e6/1638e9) = 1.06us.
    out = run(xprof.cmd_roofline, self.tr(), device=0, module="jit__bwda", flops=1e9, bytes=1e6)
    self.assertIn("n=2", out)
    self.assertIn("946.7 TF/s, 1637.99 GB/s, ridge 578 FLOP/byte", out)
    self.assertIn("measured: 40.0 us", out)
    self.assertIn("25.0 TF/s", out)
    self.assertIn("= 2.6% of peak", out)
    self.assertIn("25.0 GB/s", out)
    self.assertIn("= 1.5% of peak", out)
    self.assertIn("intensity: 1000.0 FLOP/byte against a 578 ridge " "-> compute-bound", out)
    self.assertIn("roofline floor 1.1 us; measured is 37.87x that " "(2.6% of attainable)", out)

  def test_roofline_below_the_ridge_is_bandwidth_bound(self):
    # The headline verdict on such a capture, and the branch that carries it.
    out = run(xprof.cmd_roofline, self.tr(), device=0, module="jit__bwda", flops=1e9, bytes=1e9)
    self.assertIn("intensity: 1.0 FLOP/byte", out)
    self.assertIn("-> BANDWIDTH-bound", out)
    self.assertIn("AI below the ridge", out)

  def test_roofline_needs_the_model_the_capture_does_not_carry(self):
    # One count alone still scores its own half; only *neither* stops.
    self.assertIn("25.0 TF/s", run(xprof.cmd_roofline, self.tr(), device=0, module="jit__bwda", flops=1e9))
    out = run(xprof.cmd_roofline, self.tr(), device=0, module="jit__bwda")
    self.assertIn("No --flops/--bytes given", out)
    # The colon matters: the no-counts message says "intensity" itself.
    self.assertNotIn("intensity:", out)

  def test_roofline_needs_both_peaks(self):
    tr = self.tr()
    tr.peaks[0] = (None, 1637.993152512e9)
    with self.assertRaises(SystemExit):
      run(xprof.cmd_roofline, tr, device=0, flops=1e9, bytes=1e6)


class DefaultModuleTest(unittest.TestCase):
  """`-m`'s default, as `main` parses it, against programs named as the pipeline names them.

  `ds_ep` names a program `jit_<kind>_c<chunk>` and appends what else it carries (`ChunkPrograms.program`), so the
  default, chunk 0's plain accumulating backward, has variants whose names extend its own: the backward that also
  reduce-scatters another chunk (`_rs2p0`) and the chunk's last, which reduce-scatters its own gradient (`_rsself`).
  Every program below has a duration of its own, so a default that pooled any other with it shows in n and the mean.
  A plan need not have the plain program at all: the default then reads the one program whose name extends it, and
  where several do, exits with them listed rather than pool them.
  """

  def tr(self):
    # One step at S=2, V=2, M=5 under --rs-self all, captured on both stages (--profile-chips covering them): stage 0
    # (TPU:0) holds chunks 0 and 2, stage 1 (TPU:2) chunk 1 and the fused last chunk 3.
    t = iter(range(0, 10**6, 1000))

    def mods(*rows):
      return [ev(next(t), dur, name) for n, dur, name in rows for _ in range(n)]

    return fake(
        {
            0: mods(
                (1, 10, "jit_fwd_c0_agself(1)"),
                (4, 10, "jit_fwd_c0(2)"),
                (5, 11, "jit_fwd_c2(3)"),
                (1, 30, "jit_bwd_c2(4)"),
                (4, 38, "jit_bwda_c2(5)"),
                (1, 32, "jit_bwd_c0(6)"),
                (2, 40, "jit_bwda_c0(7)"),
                (1, 45, "jit_bwda_c0_rs2p0(8)"),
                (1, 100, "jit_bwda_c0_rsself(9)"),
                (1, 20, "jit_update_s0_ag0(10)"),
            ),
            2: mods(
                (5, 12, "jit_fwd_c1(11)"),
                (1, 80, "jit_fb_c3(12)"),
                (4, 85, "jit_fba_c3(13)"),
                (1, 31, "jit_bwd_c1(14)"),
                (2, 39, "jit_bwda_c1(15)"),
                (1, 46, "jit_bwda_c1_rs3p0(16)"),
                (1, 101, "jit_bwda_c1_rsself(17)"),
                (1, 21, "jit_update_s1(18)"),
            ),
        }
    )

  def m4(self):
    # Two steps at S=2, V=2, M=4 under the default flags (breadth, --fsdp-scope repeat), captured on both stages, with
    # the programs that plan's precompile lists: --rs-split 4 spreads chunk 2's reduce-scatter over chunk 0's four
    # backwards, so none of them is plain, and chunk 0's own reduce-scatter runs standalone (reduce_s0_0).
    t = iter(range(0, 10**6, 1000))
    s0 = (
        (10, "jit_fwd_c0_agself(1)"),
        (11, "jit_fwd_c0_ag2p1(2)"),
        (11, "jit_fwd_c0_ag2p0(3)"),
        (11, "jit_fwd_c0_ag2p3(4)"),
        *[(12, "jit_fwd_c2(5)")] * 4,
        (30, "jit_bwd_c2_ag0p4(6)"),
        (38, "jit_bwda_c2_ag0p0(7)"),
        (38, "jit_bwda_c2_ag0p6(8)"),
        (39, "jit_bwda_c2_ag0p1(9)"),
        (32, "jit_bwd_c0_rs2p4(10)"),
        (40, "jit_bwda_c0_rs2p5(11)"),
        (45, "jit_bwda_c0_rs2p6(12)"),
        (100, "jit_bwda_c0_rs2p0(13)"),
        (5, "jit_reduce_s0_0(14)"),
        (20, "jit_update_s0_ag0(15)"),
    )
    s1 = (
        (12, "jit_fwd_c1_ag3p5(21)"),
        (12, "jit_fwd_c1_ag3p1(22)"),
        (12, "jit_fwd_c1_ag3p0(23)"),
        (13, "jit_fwd_c1_ag3p2(24)"),
        (80, "jit_fb_c3_ag1p4(25)"),
        (85, "jit_fba_c3_ag1p5(26)"),
        (85, "jit_fba_c3_ag1p6(27)"),
        (86, "jit_fba_c3_ag1p0(28)"),
        (31, "jit_bwd_c1_rs3p5(29)"),
        (39, "jit_bwda_c1_rs3p1(30)"),
        (41, "jit_bwda_c1_rs3p0(31)"),
        (101, "jit_bwda_c1_rs3p2(32)"),
        (5, "jit_reduce_s1_1(33)"),
        (21, "jit_update_s1_ag1(34)"),
    )
    return fake({d: [ev(next(t), dur, name) for dur, name in rows * 2] for d, rows in ((0, s0), (2, s1))})

  def test_the_tests_run_with_the_defaults_main_parses(self):
    # `run` stands in for `main`'s argparse in every other test: a default that drifted between the two would test
    # a command line nobody types.
    got = []
    with mock.patch.object(xprof, "cmd_summary", lambda tr, args: got.append(vars(args))):
      main_out(fake({0: []}), "summary", "<synthetic>")
    run(lambda tr, args: got.append(vars(args)), None)
    parsed, helper = got
    self.assertEqual({k: v for k, v in parsed.items() if k not in ("cmd", "path")}, helper)

  def test_ops_and_roofline_default_to_chunk_0s_plain_accumulating_backward(self):
    # Pooled over both traced stages, as both pool without -d: only TPU:0 runs chunk 0, and of its five backwards
    # the default takes the two plain accumulating ones -- not the first (bwd), nor the two carrying a reduce-scatter.
    self.assertIn("2 devices pooled  jit_bwda_c0  n=2  mean module 40.0 us", main_out(self.tr(), "ops", "<synthetic>"))
    out = main_out(self.tr(), "roofline", "<synthetic>", "--flops", "1e9", "--bytes", "1e6")
    self.assertIn("2 devices pooled  jit_bwda_c0  n=2", out)
    self.assertIn("measured: 40.0 us per instance", out)

  def test_without_the_plain_program_the_default_takes_the_chunks_last_backward(self):
    # M=2 under --rs-self all: chunk 0's first backward does not accumulate and its second, the last, carries the
    # chunk's own reduce-scatter.  No program is named jit_bwda_c0 exactly, so the substring fallback finds that
    # one rather than exiting, and says which program that is.
    tr = fake(
        {
            0: [
                ev(0, 10, "jit_fwd_c0_agself(1)"),
                ev(20, 10, "jit_fwd_c0(2)"),
                ev(40, 32, "jit_bwd_c0(3)"),
                ev(80, 100, "jit_bwda_c0_rsself(4)"),
            ]
        }
    )
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      self.assertIn("TPU:0  jit_bwda_c0  n=1  mean module 100.0 us", main_out(tr, "ops", "<synthetic>"))
    self.assertIn("is named 'jit_bwda_c0': reading jit_bwda_c0_rsself,", err.getvalue())

  def test_the_default_lists_the_programs_it_would_pool_rather_than_pool_them(self):
    # No program is named jit_bwda_c0 and three names extend it: pooled, they read as one jit_bwda_c0 of
    # (40 + 45 + 100) / 3 = 61.7 us, which none of them takes.  Both commands that select by -m refuse, listing what
    # matched with its instances (two steps' worth); the first microbatch's backward, jit_bwd_c0_rs2p4, does not match.
    for cmd, *flags in (("ops",), ("roofline", "--flops", "1e9", "--bytes", "1e6")):
      with self.subTest(cmd=cmd):
        with self.assertRaises(SystemExit) as cm:
          main_out(self.m4(), cmd, "<synthetic>", *flags)
        head, *rows = str(cm.exception).splitlines()
        self.assertIn("no program on device(s) [0, 2] is named 'jit_bwda_c0', and 3 have it in their names", head)
        self.assertEqual(
            [r.split() for r in rows],
            [["2x", "jit_bwda_c0_rs2p0"], ["2x", "jit_bwda_c0_rs2p5"], ["2x", "jit_bwda_c0_rs2p6"]],
        )

  def test_a_module_given_with_m_is_held_to_the_same_rule(self):
    # `rs2p` is in four names, the first microbatch's backward's too: refused the same way.  A name read off
    # `modules` selects its program exactly, and a fragment of one name only still reads that program.
    tr = self.m4()
    with self.assertRaises(SystemExit) as cm:
      main_out(tr, "ops", "<synthetic>", "-m", "rs2p")
    self.assertEqual(
        [r.split()[1] for r in str(cm.exception).splitlines()[1:]],
        ["jit_bwd_c0_rs2p4", "jit_bwda_c0_rs2p0", "jit_bwda_c0_rs2p5", "jit_bwda_c0_rs2p6"],
    )
    out = main_out(tr, "ops", "<synthetic>", "-m", "jit_bwda_c0_rs2p5")
    self.assertIn("2 devices pooled  jit_bwda_c0_rs2p5  n=2  mean module 40.0 us", out)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      self.assertIn(
          "2 devices pooled  rs2p5  n=2  mean module 40.0 us", main_out(tr, "ops", "<synthetic>", "-m", "rs2p5")
      )
    self.assertIn("reading jit_bwda_c0_rs2p5,", err.getvalue())


class ScopesTest(unittest.TestCase):
  """`scopes`: FSDP traffic found by its named scope, kernels by their name."""

  def tr(self):
    fwd = "jit(fwd_c0)/shard_map/"
    bwd = "jit(bwd_c0)/shard_map/"
    mods = [ev(0, 20000, "jit_fwd_c0_agself(7)"), ev(30000, 20000, "jit_bwd_c0(8)")]
    ops = [
        # forward: one gather (start, assembly, done) and the three kernels
        ev(0, 100, "%collective-permute-start.1 = ...", tf_op=fwd + "fsdp_all_gather/ppermute:"),
        ev(1000, 2000, "%fusion.3 = ...", tf_op=fwd + "fsdp_all_gather/dynamic_update_slice:"),
        ev(4000, 3000, "%collective-permute-done.1 = ...", tf_op=fwd + "fsdp_all_gather/ppermute:"),
        ev(8000, 1000, "%gmm_wi_0.1 = ...", tf_op=fwd + "megablox/gmm_wi_0/pallas_call:"),
        ev(9000, 1000, "%gmm_wi_1.1 = ..."),
        ev(10000, 1000, "%gmm_wo.1 = ..."),
        ev(12000, 500, "%fusion.9 = ...", tf_op=fwd + "jvp(ep_all_gather)/ppermute:"),
        # backward: the reduce-scatter sits inside a `while` body
        ev(30000, 5000, "%while.2 = ...", tf_op=bwd + "while:"),
        ev(31000, 400, "%collective-permute-done.7 = ...", tf_op=bwd + "fsdp_reduce_scatter/ppermute:"),
        ev(36000, 1000, "%gmm_wo_dx.1 = ..."),
        ev(37000, 1000, "%gmm_wo_dw.1 = ..."),
        # the transpose of the EP reduce-scatter, an async call on SparseCore;
        # its update carries no tf_op
        ev(
            40000,
            300,
            "%all-gather.2.cloned.1.call-start = ...",
            tf_op=bwd + "transpose(jvp(ep_reduce_scatter))/all_gather:",
        ),
        ev(40500, 200, "%all-gather.2.cloned.1.call-update = ..."),
        ev(
            41000,
            3000,
            "%all-gather.2.cloned.1.call-done = ...",
            tf_op=bwd + "transpose(jvp(ep_reduce_scatter))/all_gather:",
        ),
    ]
    return fake({0: mods}, ops_by_dev={0: ops})

  @staticmethod
  def row(lines, i):
    return lines[i].split()[:7]

  def test_collectives_are_found_by_scope_and_kernels_by_name(self):
    lines = run(xprof.cmd_scopes, self.tr()).splitlines()
    fwd = lines.index("  jit_fwd_c0_agself  x1")
    # wait = the permute's start + done; the rest is the assembly
    self.assertEqual(self.row(lines, fwd + 1), ["fsdp_all_gather", "3.0", "ops", "5.10", "TC", "ms", "3.10"])
    # a scope under `jvp` is still the forward collective
    self.assertEqual(self.row(lines, fwd + 2), ["ep_all_gather", "1.0", "ops", "0.50", "TC", "ms", "0.00"])
    self.assertIn("kernels per instance: gmm_wi_0 1  gmm_wi_1 1  gmm_wo 1", lines[fwd + 3])
    bwd = lines.index("  jit_bwd_c0  x1")
    # found through the `while`, which is not itself charged
    self.assertEqual(self.row(lines, bwd + 1), ["fsdp_reduce_scatter", "1.0", "ops", "0.40", "TC", "ms", "0.40"])
    # the transposed collective has its own row, the update counted with it
    self.assertEqual(self.row(lines, bwd + 2), ["transpose(ep_reduce_scatter)", "3.0", "ops", "3.50", "TC", "ms", "3.50"])
    self.assertIn("kernels per instance: gmm_wo_dw 1  gmm_wo_dx 1", lines[bwd + 3])
    self.assertEqual(
        lines[-2],
        "  step total TC ms: fsdp_all_gather 5.10, fsdp_reduce_scatter "
        "0.40, ep_all_gather 0.50, transpose(ep_reduce_scatter) 3.50",
    )

  def test_a_step_is_the_devices_block_not_the_union_window(self):
    # TPU:1's step-0 program runs past TPU:0's start of step 1, so the union
    # window of step 0 overlaps TPU:0's first program of step 1.
    tr = fake(
        {
            0: [
                ev(0, 100, "jit_fwd_c0(1)"),
                ev(200, 100, "jit_bwd_c0(2)"),
                ev(1000, 100, "jit_fwd_c0(1)"),
                ev(1200, 100, "jit_bwd_c0(2)"),
            ],
            1: [ev(100, 1000, "jit_fwd_c1(3)"), ev(1150, 1000, "jit_fwd_c1(3)")],
        },
        ops_by_dev={0: [], 1: []},
    )
    out = run(xprof.cmd_scopes, tr, step=0)
    self.assertIn("  jit_fwd_c0  x1", out)
    self.assertNotIn("x2", out)

  def test_leaves_look_through_the_while(self):
    ops = self.tr().ops(0)
    self.assertEqual(
        [o.base for o in xprof.leaves(ops, 30000 * xprof.PS, 38000 * xprof.PS)],
        ["collective-permute-done", "gmm_wo_dx", "gmm_wo_dw"],
    )

  def test_scope_of(self):
    self.assertIsNone(xprof.scope_of(ev(0, 1, "fusion.1", tf_op="jit(f)/add:")))
    self.assertIsNone(xprof.scope_of(ev(0, 1, "fusion.1")))
    self.assertEqual(xprof.scope_of(ev(0, 1, "x", tf_op="jit(f)/jvp(fsdp_all_gather)/p:")), "fsdp_all_gather")
    self.assertEqual(
        xprof.scope_of(ev(0, 1, "x", tf_op="a/transpose(jvp(ep_all_gather))/p:")), "transpose(ep_all_gather)"
    )


class CostsTest(unittest.TestCase):
  """`costs`: the module table as the `--costs` JSON the timed scheduler reads."""

  def tr(self):
    # Stage 0 (TPU:0) owns chunks 0 and 2, stage 1 (TPU:2) chunks 1 and 3, the
    # fused last chunk -- a v6e-4 capture's layout, in round numbers.
    t = iter(range(0, 10**7, 100_000))

    def mods(*rows):
      return [ev(next(t), dur, name) for dur, name in rows]

    return fake(
        {
            0: mods(
                (28000, "jit_fwd_c0(1)"),
                (28000, "jit_fwd_c0(1)"),
                (28300, "jit_fwd_c0_ag2p0(5)"),
                (29000, "jit_fwd_c0_agself(8)"),
                (25700, "jit_fwd_c2(2)"),
                (25700, "jit_fwd_c2(2)"),
                (66000, "jit_bwd_c2(4)"),
                (68700, "jit_bwda_c2(7)"),
                (68700, "jit_bwda_c2(7)"),
                (71600, "jit_bwda_c0(3)"),
                (71700, "jit_bwda_c0(3)"),
                (72650, "jit_bwda_c0_rs2p0(6)"),
                (15900, "jit_update_s0_ag0(9)"),
            ),
            # chunk 1 only ever ran with a reduce-scatter folded in
            2: mods(
                (25600, "jit_fwd_c1(11)"),
                (25800, "jit_fwd_c1(11)"),
                (69700, "jit_bwda_c1_rs3p0(12)"),
                (84000, "jit_fb_c3(13)"),
                (87000, "jit_fba_c3(14)"),
                (87000, "jit_fba_c3(14)"),
                (19200, "jit_update_s1_ag1.3(15)"),
            ),
        }
    )

  def test_costs_from_modules(self):
    got = json.loads(run(xprof.cmd_costs, self.tr()))
    self.assertEqual(
        got,
        {
            "F0": 28.0,
            "F1": 25.7,
            "F2": 25.7,  # plain forwards
            "B0": 71.65,
            "B1": 68.7,
            "B2": 68.7,  # accumulating backwards (not bwd)
            "A": 87.0,  # fba, not the first-microbatch fb
            "F": 25.7,
            "B": 68.7,  # what any other chunk would get
            "dF_ag": 0.65,  # median of +0.3 and +1.0
            "dB_rs": 1.0,  # 72.65 - 71.65
            "update": 19.2,
        },
    )
    # B1: its only sample carried a reduce-scatter, so the delta comes off it
    self.assertEqual(got["B1"], 69.7 - got["dB_rs"])
    # and it is a table the scheduler takes as is
    from maxtext.experimental.mpmd.jax import schedule

    cost = schedule.cost_table(got, 4)
    self.assertEqual([cost(c, "F") for c in range(4)], [28.0, 25.7, 25.7, 87.0])
    self.assertEqual([cost(c, "B") for c in range(4)], [71.65, 68.7, 68.7, 0.0])

  def test_costs_of_one_device(self):
    got = json.loads(run(xprof.cmd_costs, self.tr(), device=2))
    self.assertEqual(sorted(got), ["A", "B", "B1", "F", "F1", "update"])
    self.assertEqual(got["B1"], 69.7)  # no plain backward to take a delta from

  def test_costs_under_repeat_scope(self):
    """--fsdp-scope repeat's program names: a backward carrying a gather (B + G) and a gather with a
    reduce-scatter (B + R + G) get keys of their own, not dF_ag's or dB_rs'; a fused last chunk whose every
    program carries a gather (M <= pf_split) is priced from those, less the median gather delta of the other
    kinds (+0.3 and +1.0 in forwards, +1.2 in a backward: 1.0)."""
    t = iter(range(0, 10**7, 100_000))

    def mods(*rows):
      return [ev(next(t), dur, name) for dur, name in rows]

    tr = fake(
        {
            0: mods(
                (28000, "jit_fwd_c0(1)"),
                (28000, "jit_fwd_c0(1)"),
                (28300, "jit_fwd_c0_ag2p0(5)"),
                (29000, "jit_fwd_c0_agself(8)"),
                (25700, "jit_fwd_c2(2)"),
                (68700, "jit_bwda_c2(7)"),
                (68700, "jit_bwda_c2(7)"),
                (69900, "jit_bwda_c2_ag0p0(9)"),
                (70900, "jit_bwda_c2_rs4p0_ag0p1(10)"),
                (71600, "jit_bwda_c0(3)"),
            ),
            2: mods(
                (25600, "jit_fwd_c1(11)"),
                (84500, "jit_fb_c3_ag1p0(13)"),
                (87400, "jit_fba_c3_ag1p1(14)"),
                (87600, "jit_fba_c3_ag1p2(15)"),
            ),
        }
    )
    got = json.loads(run(xprof.cmd_costs, tr))
    self.assertEqual(
        got,
        {
            "F0": 28.0,
            "F1": 25.6,
            "F2": 25.7,
            "B0": 71.6,
            "B2": 68.7,
            "A": 86.5,
            "F": 25.65,
            "B": 68.7,
            "dF_ag": 0.65,
            "dB_ag": 1.2,
            "dB_ag+rs": 2.2,
        },
    )

  def test_costs_name_what_a_partial_capture_lacks(self):
    """Stage 0 alone (TPU:0, one chip: the default where one process holds every stage) has no fused last
    chunk: the table still prints, and stderr says A would come from default_costs.  Every stage: no note."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      got = json.loads(run(xprof.cmd_costs, self.tr(), device=0))
    self.assertEqual(sorted(got), ["B", "B0", "B2", "F", "F0", "F2", "dB_rs", "dF_ag", "update"])
    self.assertIn("# no A cost in", err.getvalue())
    self.assertIn("--profile-chips", err.getvalue())
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      run(xprof.cmd_costs, self.tr())
    self.assertEqual(err.getvalue(), "")

  def test_costs_needs_ds_ep_programs(self):
    tr = fake({0: [ev(0, 40, "jit_fwd_raw(1)"), ev(60, 40, "jit__bwda(1)")]})
    with self.assertRaises(SystemExit):
      run(xprof.cmd_costs, tr)


class DeviceLinesTest(unittest.TestCase):

  def test_every_line_a_subcommand_reads_is_kept(self):
    # Dropping a line here is a silent 0.00%, not a crash: `counters_0` alone
    # is 43,641 zero-duration events per device and parsing it is wasted, but
    # the three that are kept are load-bearing.
    for ln in ("XLA Modules", "XLA Ops", "Steps"):
      self.assertIn(ln, xprof.Trace.DEVICE_LINES)


class ModuleNameTest(unittest.TestCase):
  """This module shares its name with the `xprof` package, which `import tokamax` imports (tokamax 0.0.17)."""

  def test_a_module_run_by_file_path_leaves_xprof_to_the_xprof_package(self):
    """`python3 src/maxtext/experimental/mpmd/jax/ds_ep_pipeline.py` puts the package's directory first on sys.path,
    where xprof.py
    would shadow the package; the package takes that entry off as it loads."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    os.makedirs(f"{d}/xprof")
    open(f"{d}/xprof/__init__.py", "w").close()  # a stand-in for the xprof package
    here = os.path.dirname(os.path.realpath(xprof.__file__))
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"  # what a run by file path does
        "import maxtext.experimental.mpmd.jax, xprof\n"
        "print(xprof.__file__)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([d, testing.child_pythonpath(__name__)]))
    r = subprocess.run([sys.executable, "-c", code, here], cwd=d, env=env, capture_output=True, text=True, timeout=120)
    self.assertEqual(r.returncode, 0, r.stderr)
    self.assertEqual(os.path.realpath(r.stdout.strip()), os.path.realpath(f"{d}/xprof/__init__.py"))


if __name__ == "__main__":
  unittest.main()
