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

"""ds_ep_aot without a TPU: the HBM report and fit verdict of an ahead-of-time
build over CPU devices (the TINY model), the multi-controller checks it runs
before compiling, and its topology table (and, where libtpu is installed, the
devices of 8 x tpu7x-4x4x4 as its compile-only topology gives them); and the
launchers (ds_ep_run.sh, ds_ep_aot.sh, ds_ep_multihost.sh, each run with a
stand-in for python3): their libtpu flag sets (libtpu_flags/) and the flag
they add, the module each runs, from the directory the package imports from
(found from the script's real place), and its environment, and
ds_ep_multihost.sh's GPU flags, host ids (from a stand-in metadata server
too), log and run records, and the directory of its own they live in until
they are copied.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python3 -m pytest -q tests/unit/experimental/mpmd/jax/ds_ep_aot_test.py
"""

from __future__ import annotations

import collections
import contextlib
import glob
import http.server
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_aot
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import mem_probe
from maxtext.experimental.mpmd.jax import schedule as sched
from tests.unit.experimental.mpmd.jax import multihost_test
from maxtext.experimental.mpmd.jax import testing

ROOT = testing.import_root()
HAVE_LIBTPU = importlib.util.find_spec("libtpu") is not None
G = 2**30
needs_bash = unittest.skipUnless(shutil.which("bash") and shutil.which("timeout"), "needs bash and timeout")


@pytest.mark.scheduled_only
class Report(unittest.TestCase):
  """TINY at PP=2 x FSDP=2 x EP=2, V=2 -- stage 0 holds chunks 0 and 2, stage
  1 chunk 1 and the fused last -- in GPipe order (every forward of a stage
  before its backwards), compiled ahead of time with ppermute boundaries."""

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 8:
      raise unittest.SkipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    a = dsp.make_parser().parse_args(
        [
            "--stages",
            "2",
            "--vv",
            "2",
            "--fsdp",
            "2",
            "--ep",
            "2",
            "--layers",
            "4",
            "--mb",
            "4",
            "--gbs",
            "16",
            "--schedule",
            "gpipe",
            "--no-overlap",
            "--transport",
            "ppermute",
            *[x for x in multihost_test.TINY if x != "--no-precompile"],
        ]
    )
    meshes = dsp.stage_meshes(a.stages, a.fsdp, a.ep, devices=jax.devices()[:8], layout=a.layout)
    cls.p = dsp.build(a, meshes=meshes, aot=True)
    cls.p.precompile(threads=8)

  def test_rows(self):
    """One row per stage; the peak estimate is `hbm_walk`'s -- what the step
    holds when the stage's heaviest program runs, that program's
    temporaries and new outputs included -- plus the stage's generated
    code.  Without the fusion plan the loop gathers every chunk first, so
    the walk holds them all at its peak."""
    rows = ds_ep_aot.report(self.p, 16 * G)
    self.assertEqual([r["stage"] for r in rows], [0, 1])
    progs = ds_ep_aot.stage_programs(self.p)
    for r in rows:
      self.assertTrue(r["fits"])
      self.assertEqual(r["uncompiled"], [])  # the walk priced every program from its compile
      self.assertAlmostEqual(r["headroom_gib"], ds_ep_aot.HEADROOM * 16)
      self.assertGreater(r["max_temp_gib"], 0)
      self.assertIsNotNone(r["max_temp_program"])
      code = sum(m.generated_code_size_in_bytes for _, m in progs[r["stage"]] if m is not None)  # 0 on CPU
      self.assertAlmostEqual(r["code_gib"], code / G)
      live = (
          r["persistent_gib"]
          + r["live_weights_gib"]
          + r["live_gacc_gib"]
          + r["live_grads_gib"]
          + r["live_residuals_gib"]
          + r["live_stash_gib"]
          + r["live_temp_gib"]
      )
      self.assertAlmostEqual(r["est_peak_gib"], live + r["code_gib"])
      self.assertEqual(r["live_inputs_gib"], 0)  # the boundary inputs only when asked (mem_probe asks)
      self.assertAlmostEqual(r["persistent_gib"], r["storage_gib"] + r["adam_gib"])
      self.assertGreater(r["est_total_gib"], r["est_peak_gib"] - r["code_gib"])
      if r["peak_at"][0].isdigit():  # an op of the order ('0B1'), not the step's start, a reduce or the update
        self.assertAlmostEqual(r["live_weights_gib"], r["compute_gib"])

  def test_residual_bytes_by_chunk(self):
    """At one layer per chunk a set of chunk 0 is its tokens and layer 0's
    input, one of any other chunk its input activation: GPipe keeps all M
    sets of every chunk of the stage at once."""
    p = self.p
    n = p.scfg.fsdp * p.scfg.ep
    act = p.mbs * p.seq * p.scfg.model.emb_dim * jnp.dtype(p.scfg.model.dtype).itemsize // n
    tok = p.mbs * p.seq * 4 // n
    rows = ds_ep_aot.report(p, 16 * G)
    self.assertEqual([r["residual_sets"] for r in rows], [2 * p.M, p.M])
    self.assertEqual([round(r["residuals_gib"] * G) for r in rows], [p.M * (tok + act + act), p.M * act])

  def test_boundary_copies_count_on_both_stages(self):
    progs = ds_ep_aot.stage_programs(self.p)
    for name in ("xfer_s0_to_s1", "xfer_s1_to_s0"):
      for s in (0, 1):
        self.assertIn(name, [n for n, _ in progs[s]])

  def test_only_compiled_stages(self):
    """--stages-to-compile: no row for a stage whose programs were not compiled."""
    local = self.p.local
    try:
      self.p.local = frozenset({1})
      self.assertEqual([r["stage"] for r in ds_ep_aot.report(self.p, 16 * G)], [1])
    finally:
      self.p.local = local

  def test_exit_unless_fits(self):
    """A stage fits when its estimated peak leaves `headroom` of HBM free:
    a peak at 95% of HBM fits with none, not with the default 10%."""
    self.assertIsNone(ds_ep_aot.exit_unless_fits(ds_ep_aot.report(self.p, 16 * G), 16 * G))
    rows = ds_ep_aot.report(self.p, 2**20)
    self.assertEqual([r["fits"] for r in rows], [False, False])
    with self.assertRaises(SystemExit) as cm:
      ds_ep_aot.exit_unless_fits(rows, 2**20)
    self.assertIn("stages [0, 1] do not fit", str(cm.exception.code))  # a message: exit status 1
    hbm = int(max(r["est_peak_gib"] for r in ds_ep_aot.report(self.p, 16 * G)) * G / 0.95)
    self.assertEqual([r["fits"] for r in ds_ep_aot.report(self.p, hbm, headroom=0)], [True, True])
    tight = ds_ep_aot.report(self.p, hbm)
    self.assertFalse(all(r["fits"] for r in tight))
    with self.assertRaisesRegex(SystemExit, "GiB of headroom"):
      ds_ep_aot.exit_unless_fits(tight, hbm)
    with self.assertRaisesRegex(ValueError, "share of HBM"):
      ds_ep_aot.report(self.p, hbm, headroom=1)

  def test_one_process_owns_both_stages(self):
    self.assertEqual(ds_ep_aot.multihost_problems(self.p.meshes, "native"), ["stages 0 and 1 share processes [0]"])


class HbmWalk(unittest.TestCase):
  """`hbm_walk` from the plan alone -- nothing compiled, so no program
  temporaries -- on TINY at PP=2 x EP=2 with three chunks a stage in the
  breadth-first order (`--schedule breadth`)."""

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 4:
      raise unittest.SkipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    args = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
        "--stages 2 --vv 3 --fsdp 1 --ep 2 --layers 6 --mb 4 --gbs 8 --schedule breadth"
    ).split()
    ap = dsp.make_parser()
    meshes = dsp.stage_meshes(2, 1, 2, devices=jax.devices()[:4])
    cls.p = {f: dsp.build(ap.parse_args(args + ["--fsdp-scope", f]), meshes=meshes, aot=True) for f in ("step", "repeat")}

  def test_repeat_holds_two_chunks(self):
    """Under 'repeat' a stage never holds more than two chunks' weights; under
    'step' it holds all three by its last forward, and its peak is higher
    by at least a chunk's weights."""
    for s in range(2):
      p, q = self.p["step"], self.p["repeat"]
      chunk = [ds_ep_aot._tree_bytes(q._avals(s, "compute", c)) for c in q.stage_chunks[s]]
      walk = {f: ds_ep_aot.hbm_walk(self.p[f], s, trace=True) for f in self.p}
      most = {f: max(parts["weights"] for _, parts in w["trace"]) for f, w in walk.items()}
      self.assertLessEqual(most["repeat"], sum(sorted(chunk)[-2:]), s)
      self.assertEqual(most["step"], sum(chunk), s)
      self.assertGreaterEqual(walk["step"]["peak"] - walk["repeat"]["peak"], min(chunk), s)
      # every program finds the weights of the chunk it runs (gathered in a forward, a backward or the fused
      # last chunk); the self-prefetching 0F0 brings its routed experts itself, as outputs
      for f in self.p:
        r = self.p[f]
        for at, parts in walk[f]["trace"]:
          if not at[0].isdigit():
            continue
          c = int(re.match(r"\d+", at).group())
          need = ds_ep_aot._tree_bytes(r._avals(s, "compute", r.chunks[c]))
          if r.self_pf_at is not None and at == r.op_name(r.self_pf_at):
            need -= sum(
                ds_ep_aot._shard_bytes(x)
                for i, x in enumerate(jax.tree.leaves(r._avals(s, "compute", r.chunks[c])))
                if i in r.routed_leaves
            )
          self.assertGreaterEqual(parts["weights"], need, (f, s, at))
      # by the loop's end 'repeat' has released every chunk (`release_at`), 'step' none; the loop drops them
      # then, so the update closes the step with no compute weights but the next step's gather (a temporary: 0)
      self.assertEqual((walk["repeat"]["loop_end_weights"], walk["step"]["loop_end_weights"]), (0, sum(chunk)), s)
      self.assertEqual(walk["repeat"]["trace"][-1][0], f"update_s{s}")
      self.assertEqual(walk["repeat"]["trace"][-1][1]["weights"], 0)
      # nothing compiled: every program priced without temporaries, and said so
      self.assertIn(f"update_s{s}", walk["repeat"]["uncompiled"])
      self.assertEqual(len(walk["repeat"]["uncompiled"]), sum(1 for at, _ in walk["repeat"]["trace"][1:]))

  def test_accumulators_come_and_go(self):
    """A chunk's accumulator is live from its first gradient write until its
    reduce-scatter takes it: none at the step's start, every chunk's
    reduced gradients at the update."""
    p = self.p["repeat"]
    for s in range(2):
      tr = ds_ep_aot.hbm_walk(p, s, trace=True)["trace"]
      self.assertEqual(tr[0][1]["gacc"], 0)
      self.assertEqual(tr[-1][1]["gacc"], 0)
      grads = sum(ds_ep_aot._tree_bytes(p._avals(s, "grads", c)) for c in p.stage_chunks[s])
      self.assertEqual(tr[-1][1]["grads"], grads)
      self.assertGreater(max(parts["gacc"] for _, parts in tr), 0)


class HbmWalkZeroBubble(unittest.TestCase):
  """`hbm_walk` under zbh1 -- four stages of one device, chunks 1 and 2's backwards split into I and W -- from the
  plan alone (nothing compiled): a split (chunk, microbatch)'s residual set lives from its forward to its I, its W
  buffer (Pipeline.wbuf_bytes: the crossing set and the arguments W reads) from its I to its W, and a W, the chunk's
  gradient writer, brings its accumulator.  And ahead of time the I and W programs compile as any other: the report
  prices each from its compile, and its W buffers (test_precompiled, scheduled_only).  With the last chunk split too
  (--zb-split-last on), its W buffer lives from its F (AI) to its W (test_split_last, scheduled_only)."""

  ARGS = [x for x in multihost_test.TINY if x != "--no-precompile"] + (
      "--stages 4 --vv 1 --fsdp 1 --ep 1 --layers 4 --mb 4 --gbs 8 --schedule zbh1"
  ).split()

  @classmethod
  def build(cls, extra=()):
    meshes = dsp.stage_meshes(4, 1, 1, devices=jax.devices()[:4])
    return dsp.build(dsp.make_parser().parse_args(cls.ARGS + list(extra)), meshes=meshes, aot=True)

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 4:
      raise unittest.SkipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    cls.p = cls.build()

  def test_wbufs_come_and_go(self):
    p = self.p
    self.assertEqual(p.split_chunks, [1, 2])
    peaks = sched.peak_wbufs(p.order, p.place)
    for s in range(p.S):
      c = p.stage_chunks[s][0].index
      tr = ds_ep_aot.hbm_walk(p, s, trace=True)["trace"]
      wb = {at: parts["wbuf"] for at, parts in tr}
      self.assertEqual(max(wb.values()), peaks[s] * p.wbuf_bytes.get(c, 0), s)
      self.assertEqual(p.peak_wbuf_gib()[s] * G, max(wb.values()), s)
      self.assertEqual(wb[f"update_s{s}"], 0, s)
      if c not in p.split_chunks:
        continue
      # a W runs with its own buffer live (and an I's comes out of it): the program names in order
      for m in range(p.M):
        self.assertGreaterEqual(wb[f"{c}W{m}"], p.wbuf_bytes[c], (s, m))
      res = {at: parts["residuals"] for at, parts in tr}
      self.assertEqual(res[f"{c}W{p.M - 1}"], 0, s)  # every set gone at its I, before the last W
      gacc = {at: parts["gacc"] for at, parts in tr}
      self.assertEqual((gacc[f"{c}I0"], gacc[f"{c}W0"]), (0, 0))  # the first W starts the accumulator
      self.assertGreater(gacc[f"{c}W1"], 0)
    self.assertEqual([r for r, _ in ds_ep_aot.live_residuals(p)], sched.peak_residuals(p.order, p.C)[:3] + [0])

  @pytest.mark.scheduled_only
  def test_precompiled(self):
    """Compiled ahead of time: the I and W programs (bwdi, bwdw, bwdwa, bwdwa_rsself) of both split chunks, priced
    by the report from their compiles, which keeps the W buffers apart (wbuf_gib, the peak; live_wbuf_gib at the
    stage's peak)."""
    p = self.build()
    p.precompile(threads=8)
    names = {n for n, _ in p.compiled}
    for c in (1, 2):
      self.assertLessEqual({f"bwdi_c{c}", f"bwdw_c{c}", f"bwdwa_c{c}", f"bwdwa_c{c}_rsself"}, names)
    rows = ds_ep_aot.report(p, 16 * G)
    for r in rows:
      self.assertEqual(r["uncompiled"], [])
      self.assertAlmostEqual(r["wbuf_gib"], p.peak_wbuf_gib()[r["stage"]])
      self.assertLessEqual(r["live_wbuf_gib"], r["wbuf_gib"])
    self.assertEqual([r["wbuf_gib"] > 0 for r in rows], [False, True, True, False])

  @pytest.mark.scheduled_only
  def test_split_last(self):
    """--zb-split-last on: the last chunk split too, its W buffer from each F (AI) to its W -- every stage but the
    first holds W buffers -- and its accumulator in its two parts, the leaves AI keeps from its first F and its
    layers' from its first W, which writes them (ds_ep.deferred_leaves); compiled ahead of time, the AI and the last
    W programs priced from their compiles."""
    p = self.build(["--zb-split-last", "on"])
    self.assertEqual((p.split_chunks, p.split_last), ([1, 2, 3], True))
    tr = ds_ep_aot.hbm_walk(p, 3, trace=True)["trace"]
    wb = {at: parts["wbuf"] for at, parts in tr}
    self.assertEqual(max(wb.values()), sched.peak_wbufs(p.order, p.place)[3] * p.wbuf_bytes[3])
    self.assertEqual(p.peak_wbuf_gib()[3] * G, max(wb.values()))
    for m in range(p.M):
      self.assertGreaterEqual(wb[f"3W{m}"], p.wbuf_bytes[3], m)  # a W runs with its own buffer live
    self.assertEqual(wb["update_s3"], 0)
    gacc = {at: parts["gacc"] for at, parts in tr}
    leaves = [ds_ep_aot._shard_bytes(x) for x in jax.tree.leaves(p._avals(3, "gacc", p.chunks[3]))]
    kept = sum(b for b, d in zip(leaves, ds_ep.deferred_leaves(p.chunks[3]), strict=True) if not d)
    self.assertLess(0, kept)
    self.assertLess(kept, sum(leaves))
    after_w0 = [at for at, _ in tr][[at for at, _ in tr].index("3W0") + 1]
    self.assertEqual((gacc["3F0"], gacc["3W0"], gacc[after_w0]), (0, kept, sum(leaves)))
    p.precompile(threads=8)
    self.assertLessEqual({"fbi_c3", "fbai_c3", "bwdw_c3", "bwdwa_c3"}, {n for n, _ in p.compiled})
    rows = ds_ep_aot.report(p, 16 * G)
    self.assertEqual([r["uncompiled"] for r in rows], [[]] * 4)
    self.assertEqual([r["wbuf_gib"] > 0 for r in rows], [False, True, True, True])


class HbmWalkOffload(unittest.TestCase):
  """`hbm_walk` under --offload-residuals on, from the plan alone (nothing compiled): a (chunk, microbatch)'s residual
  set lives in pinned host memory from its forward to its backward, and on the device only while that backward runs
  (Pipeline._restore brings it back for it) -- so the walk prices, at each program, the running backward's set and no
  other: at a backward, or an F&B pair (an F and another microbatch's B in one program), the B's set; at anything
  else none.  Everything else it prices is offload off's.  TINY at PP=2 x EP=2 in two orders: breadth-first over
  three chunks a stage, and interleaved 1F1B's, every stage's F&B pairs.  Pricing every residual at zero --
  the running backward's too -- would hide the set a backward reads exactly where a stage's peak tends to be."""

  ORDERS = {
      "breadth": "--stages 2 --vv 3 --fsdp 1 --ep 2 --layers 6 --mb 4 --gbs 8 --schedule breadth",
      "pairs": "--stages 2 --vv 2 --fsdp 1 --ep 2 --layers 4 --mb 4 --gbs 8 --schedule interleaved --pair-stages all",
  }

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 4:
      raise unittest.SkipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    base = [x for x in multihost_test.TINY if x != "--no-precompile"]
    ap = dsp.make_parser()
    meshes = dsp.stage_meshes(2, 1, 2, devices=jax.devices()[:4])
    cls.p = {
        (k, off): dsp.build(ap.parse_args(base + flags.split() + ["--offload-residuals", off]), meshes=meshes, aot=True)
        for k, flags in cls.ORDERS.items()
        for off in ("on", "off")
    }

  @staticmethod
  def programs(p, s):
    """Stage s's programs in order, as the step loop runs them: each its op, or an F&B pair's two (order[j] and
    order[j + 1] for j in p.fuse); the fused last chunk's no-op backward is none."""
    ops = p.ops()
    return [
        [op] + ([ops[op.j + 1]] if op.j in p.fuse else [])
        for op in ops
        if op.s == s and op.desc is not None and op.j - 1 not in p.fuse
    ]

  def test_only_the_running_backward(self):
    for k in self.ORDERS:
      on, off = self.p[(k, "on")], self.p[(k, "off")]
      self.assertEqual((on.offload_residuals, off.offload_residuals), (True, False))
      self.assertEqual(on.order, off.order)
      self.assertEqual(bool(on.fuse), k == "pairs", k)
      for s in range(2):
        size = ds_ep_aot.residual_set_bytes(on, s)
        walk = {q: ds_ep_aot.hbm_walk(self.p[(k, q)], s, trace=True) for q in ("on", "off")}
        tr = {q: [(at, parts) for at, parts in w["trace"] if at[0].isdigit()] for q, w in walk.items()}
        progs = self.programs(on, s)
        self.assertEqual([at for at, _ in tr["on"]], [on.op_name(pr[0].j) for pr in progs], (k, s))
        self.assertEqual([at for at, _ in tr["off"]], [at for at, _ in tr["on"]], (k, s))
        backwards = 0
        for (at, parts), (_, parts_off), pr in zip(tr["on"], tr["off"], progs):
          running = sum(size[o.c] for o in pr if o.kind == "B")
          self.assertEqual(parts["residuals"], running, (k, s, at))
          self.assertLessEqual(running, parts_off["residuals"], (k, s, at))  # off holds that set too, and more
          self.assertEqual(
              {x: v for x, v in parts.items() if x != "residuals"},
              {x: v for x, v in parts_off.items() if x != "residuals"},
              (k, s, at),
          )
          backwards += running > 0
        self.assertEqual(backwards, sum(1 for pr in progs if any(o.kind == "B" for o in pr)), (k, s))
        self.assertGreater(backwards, 0, (k, s))
        if k == "pairs":
          self.assertTrue(any(len(pr) == 2 and pr[1].kind == "B" for pr in progs), s)
        for at, parts in walk["on"]["trace"]:  # the step's start, its gather, the reduces, the update
          if not at[0].isdigit():
            self.assertEqual(parts["residuals"], 0, (k, s, at))
        # without offload the order keeps several sets live at once (all M of a chunk, breadth-first), which
        # offload prices at most the one a backward runs on
        most_off = max(parts["residuals"] for _, parts in walk["off"]["trace"])
        self.assertGreater(most_off, max(size.values()), (k, s))
        self.assertLessEqual(max(parts["residuals"] for _, parts in walk["on"]["trace"]), max(size.values()))
        self.assertLessEqual(walk["on"]["peak"], walk["off"]["peak"], (k, s))


class HbmWalkInputs(unittest.TestCase):
  """`hbm_walk(inputs=True)`: the boundary inputs waiting on a stage as the step loop files and takes them -- an
  activation from the forward that sends it until the forward that takes it, a cotangent from the backward (or fused
  last chunk) that sends it until the backward that takes it -- from the plan alone (nothing compiled).  TINY at PP=4
  x EP=2, V=2, M=16 in Megatron's interleaved order with --offload-residuals on, in groups of S = 4 (the default) and
  of 16: in groups of G > S stage 0 collects about G - S more activations for its second chunk while it finishes its
  group of chunk 0, and stage 3 about G - S more cotangents for chunk 3 while it finishes its group of the last chunk.
  The residuals' offload moves none of them, and the walk without `inputs` prices them at nothing."""

  ARGS = "--stages 4 --vv 2 --fsdp 1 --ep 2 --layers 8 --mb 16 --gbs 32 --schedule interleaved --offload-residuals on"
  # {group: the most inputs each stage holds at once, in activations}: what one step of the pipeline held in the step
  # loop on CPU, Pipeline._file and _Box.pop counted (HbmWalkInputsDriver checks the walk against the loop itself)
  HELD = {4: [4, 3, 4, 3], 16: [14, 3, 4, 13]}
  # Interleaved 1F1B's F&B pairs at PP=2 x EP=2, and V placement's turn (chunks 1 and 2 on stage 1) under search,
  # and the most each stage held in its step loop, counted alike
  PAIRS = (
      "--stages 2 --vv 2 --fsdp 1 --ep 2 --layers 4 --mb 8 --gbs 16 --schedule interleaved --pair-stages all "
      "--offload-residuals on"
  )
  PAIRS_HELD = [2, 4]
  TURN = (
      "--stages 2 --vv 2 --fsdp 1 --ep 2 --layers 4 --mb 8 --gbs 16 --placement v --schedule search "
      "--offload-residuals on"
  )
  TURN_HELD = [1, 4]

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 8:
      raise unittest.SkipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    tiny = [x for x in multihost_test.TINY if x != "--no-precompile"]
    ap = dsp.make_parser()
    meshes = dsp.stage_meshes(4, 1, 2, devices=jax.devices()[:8])
    cls.p = {
        g: dsp.build(ap.parse_args(tiny + cls.ARGS.split() + ["--interleave-group", str(g)]), meshes=meshes, aot=True)
        for g in cls.HELD
    }
    cls.pairs, cls.turn = (
        dsp.build(
            ap.parse_args(tiny + flags.split()), meshes=dsp.stage_meshes(2, 1, 2, devices=jax.devices()[:4]), aot=True
        )
        for flags in (cls.PAIRS, cls.TURN)
    )
    cls.act = ds_ep_aot._shard_bytes(cls.p[4]._avals(0, "act"))
    cls.walk = {
        (g, q): [ds_ep_aot.hbm_walk(p, s, trace=True, inputs=q) for s in range(4)]
        for g, p in cls.p.items()
        for q in (True, False)
    }

  def test_inputs_held(self):
    """The most each stage holds at once is the step loop's (HELD); every input is taken by the loop's end; without
    `inputs` the walk prices none, and everything else as with them."""
    for g, p in self.p.items():
      self.assertTrue(p.offload_residuals)
      self.assertEqual(
          [max(parts["inputs"] for _, parts in w["trace"]) for w in self.walk[g, True]],
          [n * self.act for n in self.HELD[g]],
          g,
      )
      for s in range(4):
        on, off = self.walk[g, True][s]["trace"], self.walk[g, False][s]["trace"]
        self.assertEqual([at for at, _ in on], [at for at, _ in off], (g, s))
        for (at, parts), (_, parts_off) in zip(on, off):
          self.assertEqual(parts_off["inputs"], 0, (g, s, at))
          self.assertEqual(
              {k: v for k, v in parts.items() if k != "inputs"},
              {k: v for k, v in parts_off.items() if k != "inputs"},
              (g, s, at),
          )
          if not at[0].isdigit():  # the step's start, the reduce, the update
            self.assertEqual(parts["inputs"], 0, (g, s, at))

  def test_pairs_and_the_turn(self):
    """An F&B pair takes both its inputs before it runs and files both its outputs after; at the V's turn stage 1
    files chunk 2's input for itself (no copy: the same buffer waits).  The most each stage holds at once is again
    the step loop's, and none is left at the step's end."""
    self.assertTrue(self.pairs.fuse)
    self.assertEqual(self.turn.place[1], self.turn.place[2])
    for p, held in ((self.pairs, self.PAIRS_HELD), (self.turn, self.TURN_HELD)):
      act = ds_ep_aot._shard_bytes(p._avals(0, "act"))
      for s in range(2):
        tr = ds_ep_aot.hbm_walk(p, s, trace=True, inputs=True)["trace"]
        self.assertEqual(max(parts["inputs"] for _, parts in tr), held[s] * act, (p.schedule_name, s))
        self.assertEqual({parts["inputs"] for at, parts in tr if not at[0].isdigit()}, {0}, (p.schedule_name, s))

  def test_groups_raise_the_last_stage_peak(self):
    """The last stage's cotangents wait through its backward phase, where its peak is: 13 activations of them at it
    in groups of 16, 3 in groups of 4.  So its estimated peak goes up with the group, where the walk without
    `inputs` has it no higher."""
    at_peak = {g: self.walk[g, True][3]["parts"]["inputs"] for g in self.p}
    self.assertEqual(at_peak, {4: 3 * self.act, 16: 13 * self.act})
    on = {g: self.walk[g, True][3]["peak"] for g in self.p}
    off = {g: self.walk[g, False][3]["peak"] for g in self.p}
    self.assertGreater(on[16], on[4])
    self.assertLessEqual(off[16], off[4])

  def test_report_and_mem_probe(self):
    """report(inputs=True) prices the inputs at the peak (`live_inputs_gib`; 0 without) into the estimate.  A device
    with room for the last stage's estimate without them but not with them fits by the one and not the other, and
    mem_probe's line says by how much."""
    p = self.p[16]
    off, on = ds_ep_aot.report(p, 16 * G, 0), ds_ep_aot.report(p, 16 * G, 0, inputs=True)
    self.assertEqual([r["live_inputs_gib"] for r in off], [0] * 4)
    for r, w in zip(on, self.walk[16, True]):
      self.assertAlmostEqual(r["live_inputs_gib"], w["parts"]["inputs"] / G)
      live = (
          r["persistent_gib"]
          + r["live_weights_gib"]
          + r["live_gacc_gib"]
          + r["live_grads_gib"]
          + r["live_residuals_gib"]
          + r["live_stash_gib"]
          + r["live_inputs_gib"]
          + r["live_temp_gib"]
      )
      self.assertAlmostEqual(r["est_peak_gib"], live + r["code_gib"])
    self.assertGreaterEqual((on[3]["est_peak_gib"] - off[3]["est_peak_gib"]) * G, 10 * self.act)
    hbm = int((off[3]["est_peak_gib"] + on[3]["est_peak_gib"]) / 2 * G)
    self.assertTrue(ds_ep_aot.report(p, hbm, 0)[3]["fits"])
    r = ds_ep_aot.report(p, hbm, 0, inputs=True)[3]
    self.assertFalse(r["fits"])
    self.assertRegex(mem_probe.stage_line(r), r"^stage 3: est\. peak .* \(DOES NOT FIT, [\d.]+ GiB over\) at 3")


@pytest.mark.scheduled_only
class HbmWalkInputsDriver(unittest.TestCase):
  """The walk's inputs against the step loop's own: one step of HbmWalkInputs' pipeline in groups of 16, run on CPU
  with Pipeline._file and _Box.pop counted -- the most each stage held at once is the walk's most (HELD)."""

  def test_step_loop_holds_what_the_walk_prices(self):
    if len(jax.devices()) < 8:
      self.skipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    a = dsp.make_parser().parse_args([*multihost_test.TINY, *HbmWalkInputs.ARGS.split(), "--interleave-group", "16"])
    p = dsp.build(a, meshes=dsp.stage_meshes(4, 1, 2, devices=jax.devices()[:8]))
    held, most = [0] * p.S, [0] * p.S
    real_file, real_pop = dsp.Pipeline._file, dsp._Box.pop

    def file(pipe, box, key, src, dst, x):
      real_file(pipe, box, key, src, dst, x)
      held[dst] += 1
      most[dst] = max(most[dst], held[dst])

    def pop(box, key):  # (chunk, microbatch): taken by an op of the chunk, on its stage
      held[p.place[key[0]]] -= 1
      return real_pop(box, key)

    params, opts, wc0 = p.init(jax.random.key(0))
    batch, lrs = p.synthetic_batch(), p.lr_arrays(a.lr)
    dsp.Pipeline._file, dsp._Box.pop = file, pop
    try:
      jax.block_until_ready(p.step(params, opts, wc0, batch, lrs))
    finally:
      dsp.Pipeline._file, dsp._Box.pop = real_file, real_pop
    self.assertEqual(held, [0] * p.S)
    act = ds_ep_aot._shard_bytes(p._avals(0, "act"))
    walk = [
        max(parts["inputs"] for _, parts in ds_ep_aot.hbm_walk(p, s, trace=True, inputs=True)["trace"])
        for s in range(p.S)
    ]
    self.assertEqual([n * act for n in most], walk)
    self.assertEqual(most, HbmWalkInputs.HELD[16])


class HbmWalkSaved(unittest.TestCase):
  """`hbm_walk` under --save-residuals on, from the plan alone (nothing compiled): a chunk's forward returns its layers'
  VJP residuals, which the step loop holds with the forward's input until the backward, in place of the recompute
  programs' layer inputs.  The walk prices a set at what the forward program returns (`residual_set_bytes`), so the
  residuals it holds come out larger than recompute's, and the boundary inputs as with recompute.  TINY at PP=4 x EP=2
  with per-layer trees (the saved path reads no stacked experts), in HbmWalkInputs' interleaved order in groups of 16,
  the residuals on the device."""

  ARGS = (
      "--stages 4 --vv 2 --fsdp 1 --ep 2 --layers 8 --mb 16 --gbs 32 --schedule interleaved --interleave-group 16 "
      "--layer-tree per_layer"
  )

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 8:
      raise unittest.SkipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    tiny = [x for x in multihost_test.TINY if x != "--no-precompile"]
    ap = dsp.make_parser()
    meshes = dsp.stage_meshes(4, 1, 2, devices=jax.devices()[:8])
    cls.p = {
        q: dsp.build(ap.parse_args(tiny + cls.ARGS.split() + ["--save-residuals", q]), meshes=meshes, aot=True)
        for q in ("on", "off")
    }

  def test_sets_priced_as_the_forward_returns_them(self):
    """A saved set is the forward's input and the residuals its program returns past its output activation (abstract
    evaluation of the program the step loop calls), more than recompute's input and n_res layer inputs."""
    on, off = self.p["on"], self.p["off"]
    self.assertEqual((on.scfg.save_residuals, off.scfg.save_residuals), (True, False))
    A = on._avals
    act, tok = ds_ep_aot._shard_bytes(A(0, "act")), ds_ep_aot._shard_bytes(A(0, "tok"))
    for s in range(4):
      size, size_off = ds_ep_aot.residual_set_bytes(on, s), ds_ep_aot.residual_set_bytes(off, s)
      for ch in on.stage_chunks[s]:
        if ch.last:
          continue
        x = tok if ch.first else act
        _, res = jax.eval_shape(
            on.progs[ch.index].fwd, A(s, "compute", ch), A(s, "tok" if ch.first else "act"), A(s, "scalar_i")
        )
        self.assertEqual(size[ch.index], x + ds_ep_aot._tree_bytes(res), (s, ch.index))
        self.assertEqual(size_off[ch.index], x + off.progs[ch.index].n_res * act, (s, ch.index))
        self.assertGreater(size[ch.index], size_off[ch.index], (s, ch.index))

  def test_walk_holds_them(self):
    """The most the walk holds at once is `live_residuals`' peak of those sets, higher saved than recomputed, and so is
    the stage's peak; the boundary inputs are priced program by program as with recompute (the same order)."""
    on, off = self.p["on"], self.p["off"]
    self.assertEqual(on.order, off.order)
    for s in range(4):
      walk = {q: ds_ep_aot.hbm_walk(p, s, trace=True, inputs=True) for q, p in self.p.items()}
      most = {q: max(parts["residuals"] for _, parts in w["trace"]) for q, w in walk.items()}
      self.assertEqual(most, {q: ds_ep_aot.live_residuals(p)[s][1] for q, p in self.p.items()}, s)
      self.assertGreater(most["on"], most["off"], s)
      self.assertGreater(walk["on"]["peak"], walk["off"]["peak"], s)
      self.assertEqual(
          [(at, parts["inputs"]) for at, parts in walk["on"]["trace"]],
          [(at, parts["inputs"]) for at, parts in walk["off"]["trace"]],
          s,
      )


@pytest.mark.scheduled_only
class HbmWalkSavedDriver(unittest.TestCase):
  """The walk's saved residual sets against the step loop's: one step at PP=2 x EP=2 under --save-residuals on, run
  on CPU with the sets the loop files (Pipeline._put_F) and takes (_take_B) weighed -- a set's bytes on a device are
  its `residual_set_bytes`, the most a stage holds at once is the walk's most, and none is left at the step's end."""

  def test_step_loop_holds_what_the_walk_prices(self):
    if len(jax.devices()) < 4:
      self.skipTest("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    a = dsp.make_parser().parse_args(
        [
            *multihost_test.TINY,
            *(
                "--stages 2 --vv 2 --fsdp 1 --ep 2 --layers 4 --mb 4 --gbs 8 --schedule interleaved "
                "--layer-tree per_layer --save-residuals on"
            ).split(),
        ]
    )
    p = dsp.build(a, meshes=dsp.stage_meshes(2, 1, 2, devices=jax.devices()[:4]))
    self.assertTrue(p.scfg.save_residuals)
    size = {c: ds_ep_aot.residual_set_bytes(p, p.place[c])[c] for c in range(p.C - 1)}
    held, most, sets = [0] * p.S, [0] * p.S, collections.defaultdict(set)
    real_put, real_take = dsp.Pipeline._put_F, dsp.Pipeline._take_B

    def weigh(x_r):  # a device's bytes of a held (input, residuals) set
      return sum(v.addressable_shards[0].data.nbytes for v in jax.tree.leaves(x_r))

    def put(pipe, st, op, args, out):
      real_put(pipe, st, op, args, out)
      n = weigh(st.res[(op.c, op.m)])
      sets[op.c].add(n)
      held[op.s] += n
      most[op.s] = max(most[op.s], held[op.s])

    def take(pipe, st, op):
      held[op.s] -= weigh(st.res[(op.c, op.m)])
      return real_take(pipe, st, op)

    params, opts, wc0 = p.init(jax.random.key(0))
    batch, lrs = p.synthetic_batch(), p.lr_arrays(a.lr)
    dsp.Pipeline._put_F, dsp.Pipeline._take_B = put, take
    try:
      jax.block_until_ready(p.step(params, opts, wc0, batch, lrs))
    finally:
      dsp.Pipeline._put_F, dsp.Pipeline._take_B = real_put, real_take
    self.assertEqual(held, [0] * p.S)
    self.assertEqual(dict(sets), {c: {n} for c, n in size.items()})
    walk = [max(parts["residuals"] for _, parts in ds_ep_aot.hbm_walk(p, s, trace=True)["trace"]) for s in range(p.S)]
    self.assertEqual(most, walk)
    self.assertGreater(min(walk), max(size.values()))  # sets of several microbatches at once on every stage


class MemProbeLine(unittest.TestCase):
  """mem_probe's verdict on a stage: the estimated peak against the device's memory (less any --headroom asked for),
  fits or not, and the margin -- what the peak leaves free, or how far over it is; and what its main asks of the
  report: the boundary inputs priced, at the --headroom given (none by default: ds_ep_aot's 10% is TPU-calibrated)."""

  ROW = dict(
      stage=3,
      est_peak_gib=162.7,
      hbm_gib=174.1,
      headroom_gib=0.0,
      fits=True,
      peak_at="3B12",
      persistent_gib=40.0,
      live_weights_gib=20.0,
      live_gacc_gib=30.0,
      live_grads_gib=10.0,
      live_residuals_gib=2.4,
      live_inputs_gib=1.5,
      live_temp_gib=58.8,
      code_gib=0.0,
  )

  def test_line(self):
    line = mem_probe.stage_line(self.ROW)
    self.assertEqual(
        line,
        "stage 3: est. peak 162.7 of 174.1 GiB (fits, 11.4 GiB free) at 3B12: persistent 40.0, "
        "weights 20.0, gacc 30.0, grads 10.0, residuals 2.4, inputs 1.5, temp 58.8, code 0.0",
    )
    self.assertIn("(DOES NOT FIT, 3.2 GiB over)", mem_probe.stage_line(dict(self.ROW, est_peak_gib=177.3, fits=False)))
    self.assertIn(
        "of 174.1 GiB less 17.4 headroom (DOES NOT FIT, 11.4 GiB free)",
        mem_probe.stage_line(dict(self.ROW, headroom_gib=17.4, fits=False)),
    )

  def test_headroom_flag(self):
    with self.assertRaisesRegex(SystemExit, "share of the device memory"):
      mem_probe.main(["--headroom", "1"])

  def test_main_prices_the_inputs(self):
    """main compiles the programs, then asks report for every stage's estimate with the boundary inputs, at the
    --headroom given (0 without the flag), and prints each stage's line -- on HbmWalkInputs' pipeline in groups of
    16, with report watched and nothing compiled (the walk prices the plan alone: no temporaries)."""
    if len(jax.devices()) < 8:
      self.skipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    calls, real = [], ds_ep_aot.report

    def report(p, hbm, headroom=ds_ep_aot.HEADROOM, inputs=False):
      calls.append(("report", headroom, inputs))
      return real(p, hbm, headroom, inputs=inputs)

    def precompile(p, *args, **kwargs):
      calls.append("precompile")
      p.compiled, p.compiled_stage = [], []
      return 0, 0.0

    argv = (
        [x for x in multihost_test.TINY if x != "--no-precompile"]
        + HbmWalkInputs.ARGS.split()
        + ["--interleave-group", "16"]
    )
    for flags, headroom in (([], 0.0), (["--headroom", "0.05"], 0.05)):
      with self.subTest(flags=flags):
        calls.clear()
        with (
            mock.patch.object(ds_ep_aot, "report", report),
            mock.patch.object(dsp.Pipeline, "precompile", precompile),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
          rows = mem_probe.main(argv + flags)
        self.assertEqual(calls, ["precompile", ("report", headroom, True)])
        self.assertEqual([r["stage"] for r in rows], [0, 1, 2, 3])
        self.assertGreater(rows[3]["live_inputs_gib"], 0)
        lines = out.getvalue().splitlines()
        for r in rows:
          self.assertIn(mem_probe.stage_line(r), lines)  # untagged: one controller (multi-controller: [p<process>])


@pytest.mark.scheduled_only
class ReportOverlap(unittest.TestCase):
  """The report on the fusion plan's programs: TINY at PP=2 x FSDP=2 x EP=2,
  V=2, in the default order and scope (breadth, weights per visit), every
  program compiled ahead of time."""

  @classmethod
  def setUpClass(cls):
    if len(jax.devices()) < 8:
      raise unittest.SkipTest("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    a = dsp.make_parser().parse_args(
        [
            "--stages",
            "2",
            "--vv",
            "2",
            "--fsdp",
            "2",
            "--ep",
            "2",
            "--layers",
            "4",
            "--mb",
            "4",
            "--gbs",
            "16",
            *[x for x in multihost_test.TINY if x != "--no-precompile"],
        ]
    )
    meshes = dsp.stage_meshes(a.stages, a.fsdp, a.ep, devices=jax.devices()[:8], layout=a.layout)
    cls.p = dsp.build(a, meshes=meshes, aot=True)
    cls.p.precompile(threads=8)

  def test_walk_prices_the_compiled_programs(self):
    """Every program the walk prices -- the ops' host variants (F + G, B + G,
    the fused last chunk + G, B + R), the final reduces, the update -- is
    one `precompile` compiled, and their temporaries are in the peak: priced
    at nothing, as a program missing from the compiled set would be, the
    estimate drops."""
    p = self.p
    self.assertEqual((p.schedule, p.fsdp_scope), ("breadth", "repeat"))
    self.assertEqual({"A" if p.order[j][0] == p.C - 1 else p.order[j][1] for j in p.pf}, {"F", "B", "A"})
    rows = ds_ep_aot.report(p, 16 * G)
    self.assertEqual([r["uncompiled"] for r in rows], [[], []])
    walks = [ds_ep_aot.hbm_walk(p, s, trace=True) for s in range(p.S)]
    self.assertTrue(all(any(parts["temp"] for _, parts in w["trace"]) for w in walks))
    real = ds_ep_aot._program_memory
    try:
      ds_ep_aot._program_memory = lambda exe: (0, 0)
      blind = ds_ep_aot.report(p, 16 * G)
    finally:
      ds_ep_aot._program_memory = real
    for r, b in zip(rows, blind):
      self.assertLess(b["est_peak_gib"], r["est_peak_gib"], r["stage"])


# libtpu's compile-only 8 x tpu7x-4x4x4 topology and one slice of it -> a JSON line: per slice its devices' ids,
# processes and (chip, core) places; the device kinds, devices a process, and the one slice's devices.
_TPU7X_4X4X4_DEVICES = r"""if True:
    import collections, json
    from maxtext.experimental.mpmd.jax import ds_ep_aot, ds_ep_pipeline as dsp
    devs = ds_ep_aot.topology_devices('tpu7x-4x4x4', 8)
    one = ds_ep_aot.topology_devices('tpu7x-4x4x4')
    by = collections.defaultdict(list)
    for d in devs:
        by[dsp.slice_index(d)].append(d)
    out = {'n': len(devs), 'kinds': sorted({d.device_kind for d in devs}), 'slices': sorted(by),
           'ids': [sorted(d.id for d in by[k]) for k in sorted(by)],
           'procs': [sorted({d.process_index for d in by[k]}) for k in sorted(by)],
           'places': [sorted({(*d.coords, d.core_on_chip) for d in by[k]}) for k in sorted(by)],
           'per_process': sorted(set(collections.Counter(d.process_index for d in devs).values())),
           'one': [len(one), len({d.process_index for d in one}), sorted({str(dsp.slice_index(d)) for d in one})]}
    print('RESULT ' + json.dumps(out))
"""


def _compile_only(t, code, timeout=300):
  """`python -c code` on the CPU backend, as slices_layout_test._run runs
  it: libtpu loads for a compile-only topology, which opens no TPU, so next
  to whatever holds this VM's (ALLOW_MULTIPLE_LIBTPU_LOAD) -> the JSON of
  its RESULT line, failing TestCase `t` with its output where it printed
  none."""
  env = dict(
      os.environ, JAX_PLATFORMS="cpu", ALLOW_MULTIPLE_LIBTPU_LOAD="1", PYTHONPATH=testing.child_pythonpath(__name__)
  )
  env.pop("MPMD_FAKE_SLICES", None)
  r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
  line = next((ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")), None)
  t.assertIsNotNone(line, r.stdout[-3000:] + r.stderr[-3000:])
  return json.loads(line[len("RESULT ") :])


class Topologies(unittest.TestCase):

  def test_tpu7x(self):
    """The TPU7x slice the DeepSeek-V3 study targets: 8x8x8 chips of two
    TensorCores, 94.74 GiB a device as the compile-only topology compiles."""
    topo, chip, hosts, wrap, hbm = ds_ep_aot.TOPOLOGIES["tpu7x-8x8x8"]
    self.assertEqual((topo, chip, hosts, wrap), ("tpu7x:8x8x8", "default", (2, 2, 1), (True, True, True)))
    self.assertAlmostEqual(hbm / G, 94.74, places=2)

  def test_tpu7x_4x4x4(self):
    """The slice the Qwen3.8 study targets, eight of them a stage each
    (MaxText's tpu7x-128): 4x4x4 chips of two TensorCores, 2x2x1 chips a
    host, wrapped, 94.74 GiB a device as the compile-only topology compiles."""
    topo, chip, hosts, wrap, hbm = ds_ep_aot.TOPOLOGIES["tpu7x-4x4x4"]
    self.assertEqual((topo, chip, hosts, wrap), ("tpu7x:4x4x4", "default", (2, 2, 1), (True, True, True)))
    self.assertAlmostEqual(hbm / G, 94.74, places=2)

  @unittest.skipUnless(HAVE_LIBTPU, "needs libtpu (compile-only TPU topologies)")
  def test_tpu7x_4x4x4_devices(self):
    """What the 8-slice AOT compiles for (`topology_devices`, in a
    subprocess on the CPU backend: no TPU is used): 1024 TPU7x devices, slice
    k's 128 on processes 16k..16k+15 (8 devices a host), with ids (k + 1) *
    100000 + i, i < 128, each a core of the 64 chips of a 4x4x4 box -- every
    slice the same chip coordinates.  One slice: 128 devices on 16 processes,
    with no slice index."""
    got = _compile_only(self, _TPU7X_4X4X4_DEVICES)
    self.assertEqual((got["n"], got["kinds"], got["slices"]), (1024, ["TPU7x"], list(range(8))))
    places = [[x, y, z, core] for x, y, z in sorted(np.ndindex(4, 4, 4)) for core in (0, 1)]
    for k in range(8):
      self.assertEqual(got["ids"][k], [(k + 1) * 100000 + i for i in range(128)], k)
      self.assertEqual(got["procs"][k], list(range(16 * k, 16 * k + 16)), k)
      self.assertEqual(got["places"][k], places, k)
    self.assertEqual(got["per_process"], [8])
    self.assertEqual(got["one"], [128, 16, ["None"]])


class _Dev:
  """A device as the multi-controller checks see one: its process."""

  def __init__(self, process_index):
    self.process_index = process_index


def _meshes(*stages):
  """Stand-ins for stage meshes (the checks read `.devices` only): each stage
  a list of its devices' processes, in mesh order."""
  return [
      types.SimpleNamespace(devices=np.array([_Dev(q) for q in procs], dtype=object).reshape(-1, 1)) for procs in stages
  ]


class MultihostProblems(unittest.TestCase):

  def test_hosts_of_their_own(self):
    meshes = _meshes([0, 0, 1, 1], [2, 2, 3, 3])
    self.assertEqual(ds_ep_aot.multihost_problems(meshes, "native"), [])
    self.assertEqual(ds_ep_aot.multihost_problems(meshes, "xfer"), [])

  def test_shared_processes(self):
    meshes = _meshes([0, 0, 1, 1], [1, 1, 2, 2], [3, 3, 4, 4])
    for transport in ("native", "ppermute", "xfer"):
      self.assertEqual(ds_ep_aot.multihost_problems(meshes, transport), ["stages 0 and 1 share processes [1]"])

  def test_xfer_one_peer_host(self):
    """Shard k of a stage goes to device k of the next: here host 0's two
    shards would go to hosts 2 and 3, which XferChannel cannot pair."""
    meshes = _meshes([0, 0, 1, 1], [2, 3, 2, 3])
    self.assertEqual(ds_ep_aot.multihost_problems(meshes, "native"), [])
    bad = ds_ep_aot.multihost_problems(meshes, "xfer")
    self.assertEqual(len(bad), 8)  # every process, both ways
    self.assertIn("s0->s1: process 0 would send to processes [2, 3]", bad[0])
    self.assertTrue(any(b.startswith("s1->s0: process 0 would receive from processes [2, 3]") for b in bad))


class StageExtent(unittest.TestCase):

  def test_z_range_on_deep_slices_only(self):
    """v5p-16 (2x2x2 chips, a host per z plane) as two stages, each its own plane; a slice one chip deep (z always 0,
    as on v6e) has no z."""
    dev = lambda p, x, y, z: types.SimpleNamespace(process_index=p, coords=(x, y, z))
    mesh = lambda ds: types.SimpleNamespace(devices=np.array(ds, dtype=object))
    v5p = [dev(z, x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    self.assertEqual(
        [ds_ep_aot.stage_extent(mesh(v5p[4 * s : 4 * s + 4]), v5p) for s in (0, 1)],
        ["hosts 0..0 (1), chips x 0..1 y 0..1 z 0..0", "hosts 1..1 (1), chips x 0..1 y 0..1 z 1..1"],
    )
    v6e = [dev(y // 2 * 2 + x // 2, x, y, 0) for y in range(4) for x in range(4)]
    self.assertEqual(ds_ep_aot.stage_extent(mesh(v6e[:8]), v6e), "hosts 0..1 (2), chips x 0..3 y 0..1")


# ------------------------------------------------------------------- launchers
# The launchers, their flag sets and gpu.xla_flags: files of the package, in its directory (the tests are in a
# directory of their own).
PKG_DIR = testing.package_dir()
LAUNCHERS = ("ds_ep_run.sh", "ds_ep_aot.sh", "ds_ep_multihost.sh")
FLAG_SETS = ("tpu7x", "v5p", "v5p_fsdp_extra")
MPMD = "--xla_max_concurrent_async_collective_permutes=64"  # added to any set; ds_ep_aot warns without it
PIPELINE = "maxtext.experimental.mpmd.jax.ds_ep_pipeline"
# The interpreter arguments each launcher puts before the caller's; ds_ep_multihost.sh's --json takes the record's path.
PREFIX = {
    "ds_ep_run.sh": ["-m", PIPELINE],
    "ds_ep_aot.sh": ["-m", "maxtext.experimental.mpmd.jax.ds_ep_aot"],
    "ds_ep_multihost.sh": ["-u", "-m", PIPELINE, "--distributed", "--json"],
}
# The default set, tpu7x: the 30 flags main's launchers read from the SPMD pipeline baseline's DEFAULT_LIBTPU_INIT_ARGS
# line, in its order (their default LIBTPU_INIT_ARGS is these, the MPMD flag and an empty EXTRA_LIBTPU_INIT_ARGS).
TPU7X = [
    "--xla_tpu_dvfs_p_state=7",
    "--xla_tpu_scoped_vmem_limit_kib=65536",
    "--xla_tpu_num_sparse_cores_for_gather_offloading=1",
    "--xla_tpu_bf16_emission_mode=NATIVE_EMISSION",
    "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true",
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
    "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
    "--xla_tpu_use_tc_device_shape_on_sc=True",
    "--xla_sc_disable_megacore_partitioning=True",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false",
    "--xla_enable_async_all_gather=true",
    "--xla_tpu_prefer_async_allgather_to_allreduce=true",
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
    "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true",
    "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true",
    "--xla_tpu_enable_concurrent_sparse_core_offloading=true",
    "--xla_tpu_enable_offloading_gather_to_sparsecore=true",
    "--xla_tpu_sparse_core_all_gather_latency_multiplier=1",
    "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3",
    "--xla_tpu_enable_sparse_core_collective_aggregator=true",
    "--xla_tpu_enable_latency_hiding_layer_scheduler=true",
    "--xla_tpu_scheduler_percent_shared_memory_limit=150",
    "--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true",
    "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
    "--xla_tpu_pcie_bandwidth_multiplier=0.03",
    "--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true",
    "--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false",
    "--xla_tpu_enable_3d_reduce_scatter_decomposer=false",
    "--xla_tpu_enable_ici_rs_pipelining=true",
]
# A stand-in for google.cloud.storage under ds_ep_multihost.sh's copy of its files to a gs:// RUN_DIR (the script that
# `python3 -` runs): the object gs://<bucket>/<name> is the file $FAKE_GCS/<bucket>/<name>, and each upload prints
# `UPLOAD gs://<bucket>/<name>`, the name as given ('//' and all, which the file's path would not show).
FAKE_GCS = """
import os
import shutil
import sys
import types


class Blob:

  def __init__(self, bucket, name):
    self.bucket, self.name = bucket, name

  def upload_from_filename(self, filename):
    print(f'UPLOAD gs://{self.bucket}/{self.name}')
    path = os.path.join(os.environ['FAKE_GCS'], self.bucket, *self.name.split('/'))   # a leading '/' stays inside
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.copy(filename, path)


class Bucket:

  def __init__(self, name):
    self.name = name

  def blob(self, name):
    return Blob(self.name, name)


class Client:

  def bucket(self, name):
    return Bucket(name)


storage = types.ModuleType('google.cloud.storage')
storage.Client = Client
cloud = types.ModuleType('google.cloud')
cloud.storage = storage
google = types.ModuleType('google')
google.cloud = cloud
sys.modules.update({'google': google, 'google.cloud': cloud, 'google.cloud.storage': storage})
sys.argv = sys.argv[1:]      # ['-', destination, files...], as `python3 -` has them
exec(compile(sys.stdin.read(), '<stdin>', 'exec'), {'__name__': '__main__'})
"""


def flag_set(name):
  """libtpu_flags/<name>.flags as the launchers read it: its flags, in order ('#' starts a comment)."""
  with open(os.path.join(PKG_DIR, "libtpu_flags", f"{name}.flags")) as f:
    return [x for line in f for x in line.split("#", 1)[0].split()]


def composed(flags, extra=""):
  """LIBTPU_INIT_ARGS as the launchers build it, byte for byte as main's did: the set's flags (or
  LIBTPU_INIT_ARGS_OVERRIDE's text, a str) joined by single spaces, the MPMD flag, then EXTRA_LIBTPU_INIT_ARGS as
  given -- so the default ends in a space."""
  return f'{flags if isinstance(flags, str) else " ".join(flags)} {MPMD} {extra}'


def _nowhere(d):
  """A path under `d` whose directory does not exist: a symlink to it is a file no one can write, root included."""
  return os.path.join(d, "no-such-dir", "file")


def _fake_python(d):
  """A stand-in for python3.  `-` (the scripts' inline helpers: ds_ep_multihost.sh's copy of its files to RUN_DIR,
  the MAXTEXT_DIR check) runs for real, with FAKE_GCS set against the stand-in for google.cloud.storage above.
  Anything else is the run: it prints its arguments, one a line, its working directory and environment, sleeps
  FAKE_SLEEP seconds if that is set, makes the file FAKE_UNWRITABLE names in the --json record's directory one no
  one can write (a symlink to `_nowhere(d)`) if that is set, and with FAKE_RC 0 (the default) writes the --json
  record, and next to it those of the processes in FAKE_LEADERS (stage leaders')."""
  shim = os.path.join(d, "fake_gcs.py")
  with open(shim, "w") as f:
    f.write(FAKE_GCS)
  py, shim, nowhere = shlex.quote(sys.executable), shlex.quote(shim), shlex.quote(_nowhere(d))
  path = os.path.join(d, "fakepy")
  with open(path, "w") as f:
    f.write(
        f"""#!/bin/bash
if [[ "$1" == "-" ]]; then
  if [[ -n "${{FAKE_GCS:-}}" ]]; then exec {py} {shim} "$@"; fi
  exec {py} "$@"
fi
printf 'FAKE ARG %s\\n' "$@"
echo "FAKE CWD=$PWD"
for v in PYTHONPATH LIBTPU_INIT_ARGS XLA_FLAGS PYTHONHASHSEED PYTHONUNBUFFERED JAX_CACHE_DIR; do
  if [[ -n "${{!v+set}}" ]]; then echo "FAKE $v=${{!v}}"; else echo "FAKE $v=<unset>"; fi
done
if [[ -n "${{FAKE_SLEEP:-}}" ]]; then sleep "$FAKE_SLEEP"; fi
json=""
prev=""
for a in "$@"; do if [[ "$prev" == --json ]]; then json=$a; fi; prev=$a; done
if [[ -n "${{FAKE_UNWRITABLE:-}}" && -n "$json" ]]; then ln -sfn {nowhere} "${{json%/*}}/$FAKE_UNWRITABLE"; fi
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --json && "${{FAKE_RC:-0}}" == 0 ]]; then
    echo '{{"step_ms": 1.0}}' > "$2"
    for p in ${{FAKE_LEADERS:-}}; do echo '{{"step_ms": 1.0}}' > "${{2%.json}}.p$p.json"; done
  fi
  shift
done
exit "${{FAKE_RC:-0}}"
"""
    )
  os.chmod(path, 0o755)
  return path


def _seen(r):
  """What the stand-in run saw, from its output: ARGS, its arguments; CWD; and the environment it prints, by name."""
  seen = {"ARGS": []}
  for line in r.stdout.splitlines():
    if line.startswith("FAKE ARG "):
      seen["ARGS"].append(line[len("FAKE ARG ") :])
    elif line.startswith("FAKE "):
      k, _, v = line[len("FAKE ") :].partition("=")
      seen[k] = v
  return seen


class Metadata(http.server.BaseHTTPRequestHandler):
  """A stand-in for a GCE VM's metadata server, for ds_ep_multihost.sh's host ids: the instance attribute <key> at
  /computeMetadata/v1/instance/attributes/<key> is the server's attrs[key] (404 if it has none; 403 without the
  Metadata-Flavor header, as the real one answers); every key asked goes to the server's `asked`."""

  def do_GET(self):
    prefix = "/computeMetadata/v1/instance/attributes/"
    key = self.path[len(prefix) :] if self.path.startswith(prefix) else self.path
    self.server.asked.append(key)
    if self.headers.get("Metadata-Flavor") != "Google":
      code, body = 403, "Missing Metadata-Flavor:Google header."
    else:
      code, body = (200, self.server.attrs[key]) if key in self.server.attrs else (404, "Not Found")
    data = body.encode()
    self.send_response(code)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, *args):  # no request log on stderr
    pass


@needs_bash
class Launchers(unittest.TestCase):
  """The launchers of this checkout, run with the stand-in python and an environment without the variables they read
  (but those a test sets), ds_ep_multihost.sh with a RUN_DIR, STEP_TAG and TMPDIR of the test's own, and kept off the
  metadata server (TPU_SKIP_MDS_QUERY) but where a test points it at a stand-in."""

  # What a launcher reads, or sets for the run: none may come from the environment the tests run in.
  READ = (
      "MPMD_FLAG_SET",
      "LIBTPU_INIT_ARGS",
      "LIBTPU_INIT_ARGS_OVERRIDE",
      "EXTRA_LIBTPU_INIT_ARGS",
      "JAX_CACHE_DIR",
      "PYTHONHASHSEED",
      "PYTHONUNBUFFERED",
      "PYTHONPATH",
      "RUN_DIR",
      "STEP_TAG",
      "MODULE",
      "TIMEOUT",
      "JOB_COMPLETION_INDEX",
      "TMPDIR",
      "GCE_METADATA_IP",
      "TPU_SKIP_MDS_QUERY",
      "MAXTEXT_DIR",
      "WID",
      "PLATFORM",
      "JAX_PLATFORMS",
      "XLA_FLAGS",
      "XLA_FLAGS_OVERRIDE",
      "EXTRA_XLA_FLAGS",
      "CDPATH",
  )

  def setUp(self):
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
    self.py = _fake_python(self.tmp)
    # ds_ep_multihost.sh's directory for its files until it copies them to RUN_DIR: $TMPDIR/mpmd.XXXXXXXX, under the
    # scratch directory; and the files a test leaves in /tmp under the names main's launcher wrote there.
    self.tag = f"launcher-test-{uuid.uuid4().hex[:8]}"
    self.addCleanup(self._remove_stale)
    self.run_dir = os.path.join(self.tmp, "runs")

  def _remove_stale(self):
    for f in glob.glob(f"/tmp/mpmd_{self.tag}*"):
      if os.path.isdir(f) and not os.path.islink(f):
        shutil.rmtree(f)
      else:
        os.remove(f)

  def _work_dirs(self):
    """The directories ds_ep_multihost.sh made under the test's TMPDIR and left there."""
    return glob.glob(os.path.join(self.tmp, "mpmd.*"))

  def _only_logs_left(self, n=1):
    """`n` directories of ds_ep_multihost.sh's under the test's TMPDIR, each with the host's log alone: once the copy
    succeeds the run records go and the log stays (a watcher that reads these logs finds it there).  Removed here, so
    a later check starts afresh."""
    dirs = self._work_dirs()
    self.assertEqual(sorted(os.listdir(d) for d in dirs), [[f"mpmd_{self.tag}_host0.log"]] * n)
    for d in dirs:
      shutil.rmtree(d)

  def _record(self, r):
    """The run record's path the stand-in run of ds_ep_multihost.sh was given (--json): a file named for the tag, in a
    directory made for the invocation under TMPDIR."""
    args = _seen(r)["ARGS"]
    path = args[args.index("--json") + 1]
    self.assertRegex(path, rf"^{re.escape(self.tmp)}/mpmd\.[A-Za-z0-9]{{8}}/mpmd_{re.escape(self.tag)}\.json$")
    return path

  def _run(self, script, *args, cwd=None, **env):
    """`bash script args` -- a launcher of this checkout by name, or a path (a relative one from `cwd`) -- from `cwd`
    (default: a scratch directory), with `env` on top of the tests' environment; a value None unsets the variable."""
    keep = {
        k: v
        for k, v in os.environ.items()
        if k not in self.READ and not k.startswith(("TPU_WORKER", "MEGASCALE_", "MAXTEXT_MPMD_", "FAKE_"))
    }
    keep.update(PYTHON=self.py, RUN_DIR=self.run_dir, STEP_TAG=self.tag, TMPDIR=self.tmp, TPU_SKIP_MDS_QUERY="1")
    for k, v in env.items():
      if v is None:
        keep.pop(k, None)
      else:
        keep[k] = v
    path = script if os.sep in script else os.path.join(PKG_DIR, script)
    return subprocess.run(
        ["bash", path, *args], env=keep, cwd=cwd or self.tmp, capture_output=True, text=True, timeout=120
    )

  def _log(self, run_dir, host):
    """ds_ep_multihost.sh's log of `host`, copied to `run_dir`, as lines."""
    with open(os.path.join(run_dir, self.tag, "logs", f"mpmd_{self.tag}_host{host}.log")) as f:
      return f.read().splitlines()

  def test_flag_set_files(self):
    """Each set is one flag a line and whole-line comments, so the `grep -v '^#'` its header shows reads what the
    launchers read; no flag twice.  tpu7x: the 30 libtpu flags both sides of the TPU7x measurements ran, in the order
    main's launchers gave them, the scoped VMEM limit the megablox tiles need among them, and not
    --xla_tpu_enable_ici_ag_pipelining (libtpu 0.0.46.1 cannot compile MaxText's DeepSeek-V3 train step with it).
    v5p: the recipe's 33 flags less the chip config libtpu 0.0.46 rejects under that spelling, the other 32, with the
    recipe's own scoped VMEM limit.  v5p_fsdp_extra: SparseCore
    offload of 2D and 3D all-gathers, neither in v5p."""
    names = lambda flags: {x.split("=", 1)[0] for x in flags}
    for name in FLAG_SETS:
      with self.subTest(flag_set=name):
        with open(os.path.join(PKG_DIR, "libtpu_flags", f"{name}.flags")) as f:
          lines = f.read().splitlines()
        flags = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        self.assertEqual(flags, flag_set(name))
        self.assertEqual([x for ln in lines if not ln.startswith("#") for x in ln.split()], flags)
        for x in flags:
          self.assertRegex(x, r"^--[A-Za-z0-9_]+=\S+$")
        self.assertEqual(len(names(flags)), len(flags))
    tpu7x, v5p, extra = map(flag_set, FLAG_SETS)
    self.assertEqual(tpu7x, TPU7X)
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65536", tpu7x)
    self.assertNotIn("--xla_tpu_enable_ici_ag_pipelining", names(tpu7x))
    self.assertEqual(len(v5p), 32)
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=81920", v5p)
    self.assertNotIn("--2a886c8_chip_config_name", names(v5p))
    self.assertEqual(
        sorted(extra),
        [
            "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
            "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true",
        ],
    )
    self.assertFalse(names(extra) & names(v5p))

  def test_flag_sets(self):
    """LIBTPU_INIT_ARGS from MPMD_FLAG_SET, alike in all three launchers: tpu7x (the default, also when empty) is the
    string main's launchers exported, byte for byte; a set is its flags, then the MPMD flag; none is the MPMD flag
    alone.  A LIBTPU_INIT_ARGS from the environment is not used."""
    want = {
        None: composed(TPU7X),
        "": composed(TPU7X),
        "tpu7x": composed(TPU7X),
        "none": composed([]),
        "v5p": composed(flag_set("v5p")),
        "v5p_fsdp_extra": composed(flag_set("v5p_fsdp_extra")),
    }
    self.assertEqual(want[None], " ".join(TPU7X) + f" {MPMD} ")
    for script in LAUNCHERS:
      for name, flags in want.items():
        with self.subTest(script=script, flag_set=name):
          r = self._run(script, "--stages", "2", MPMD_FLAG_SET=name, LIBTPU_INIT_ARGS="--xla_from_before=1")
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          self.assertEqual(_seen(r)["LIBTPU_INIT_ARGS"], flags)

  def test_unknown_flag_set_runs_nothing(self):
    """A set name without a file in libtpu_flags/ -- a misspelling, the file's own name, a path -- stops each launcher
    before anything runs (exit 1), naming the sets there are; ds_ep_multihost.sh makes no directory, writes no log."""
    for script in LAUNCHERS:
      for name in ("tpu8", "tpu7x.flags", "../libtpu_flags/tpu7x"):
        with self.subTest(script=script, flag_set=name):
          r = self._run(script, "--stages", "2", MPMD_FLAG_SET=name)
          self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
          self.assertIn(f"{script}: MPMD_FLAG_SET={name}: no such libtpu flag set (none, ", r.stderr)
          for known in FLAG_SETS:
            self.assertIn(f", {known}", r.stderr)
          self.assertEqual(r.stdout, "")
          self.assertFalse(os.path.exists(self.run_dir))
          self.assertEqual(self._work_dirs(), [])

  def test_override_and_extra(self):
    """LIBTPU_INIT_ARGS_OVERRIDE replaces the set's flags, even when empty, and goes in as given; EXTRA_LIBTPU_INIT_ARGS
    comes last, as given (a flag's last mention wins: here the scoped VMEM limit).  The v5p combination the sets
    document: v5p, the MPMD flag, then v5p_fsdp_extra."""
    v5p, extra = flag_set("v5p"), flag_set("v5p_fsdp_extra")
    vmem = "--xla_tpu_scoped_vmem_limit_kib=98304"
    cases = (
        (dict(LIBTPU_INIT_ARGS_OVERRIDE=""), composed("")),
        (dict(MPMD_FLAG_SET="v5p", LIBTPU_INIT_ARGS_OVERRIDE="--xla_a=1  --xla_b=2"), composed("--xla_a=1  --xla_b=2")),
        (dict(EXTRA_LIBTPU_INIT_ARGS=vmem), composed(TPU7X, vmem)),
        (dict(MPMD_FLAG_SET="v5p", EXTRA_LIBTPU_INIT_ARGS=" ".join(extra)), composed(v5p, " ".join(extra))),
    )
    for script in LAUNCHERS:
      for env, flags in cases:
        with self.subTest(script=script, env=env):
          r = self._run(script, "--stages", "2", **env)
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          self.assertEqual(_seen(r)["LIBTPU_INIT_ARGS"], flags)

  def test_multihost_flag_composition(self):
    """ds_ep_multihost.sh's LIBTPU_INIT_ARGS on a v5p comparison's step, as the runs gave it (the recipe's flags as
    LIBTPU_INIT_ARGS_OVERRIDE and v5p_fsdp_extra's as EXTRA_LIBTPU_INIT_ARGS, each read with the `grep -v '^#' | tr`
    their headers show) and as MPMD_FLAG_SET=v5p gives it: the recipe's 32 flags, the MPMD flag, then
    v5p_fsdp_extra's 2 -- every flag once, 35 -- in the log's second line as in the run's environment.  The platform
    is said, not guessed: an nvidia-smi on the host's PATH leaves a TPU step its libtpu flags, and no XLA_FLAGS."""
    read = lambda name: subprocess.run(
        ["bash", "-c", f"grep -v '^#' {shlex.quote(os.path.join(PKG_DIR, 'libtpu_flags', name))} | tr '\\n' ' '"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    every = flag_set("v5p") + [MPMD] + flag_set("v5p_fsdp_extra")
    self.assertEqual(len(every), 35)
    self.assertEqual(len({x.split("=", 1)[0] for x in every}), 35)
    smi = os.path.join(self.tmp, "bin", "nvidia-smi")
    os.makedirs(os.path.dirname(smi))
    with open(smi, "w") as f:
      f.write('#!/bin/sh\necho "GPU 0: NVIDIA B200"\n')
    os.chmod(smi, 0o755)
    on_path = f'{os.path.dirname(smi)}{os.pathsep}{os.environ["PATH"]}'
    for how, env in (
        (
            "as the runs gave it",
            dict(LIBTPU_INIT_ARGS_OVERRIDE=read("v5p.flags"), EXTRA_LIBTPU_INIT_ARGS=read("v5p_fsdp_extra.flags")),
        ),
        (
            "a set and its additions",
            dict(MPMD_FLAG_SET="v5p", EXTRA_LIBTPU_INIT_ARGS=" ".join(flag_set("v5p_fsdp_extra"))),
        ),
        (
            "beside an nvidia-smi",
            dict(MPMD_FLAG_SET="v5p", PATH=on_path, EXTRA_LIBTPU_INIT_ARGS=" ".join(flag_set("v5p_fsdp_extra"))),
        ),
    ):
      with self.subTest(how=how):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", **env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        seen = _seen(r)
        self.assertEqual(seen["LIBTPU_INIT_ARGS"].split(), every)
        self.assertEqual(seen["XLA_FLAGS"], "<unset>")
        self.assertEqual(self._log(self.run_dir, 0)[1], f'LIBTPU_INIT_ARGS={seen["LIBTPU_INIT_ARGS"]}')

  def test_multihost_gpu(self):
    """On GPU (PLATFORM=gpu, as a GPU job's entry script sets it, or JAX_PLATFORMS=cuda... / gpu...) ds_ep_multihost.sh sets
    XLA_FLAGS from gpu.xla_flags (its non-comment lines joined by spaces; XLA_FLAGS_OVERRIDE, even empty, in their
    place) plus EXTRA_XLA_FLAGS, and leaves LIBTPU_INIT_ARGS alone: no libtpu flag set is read, so an unknown
    MPMD_FLAG_SET, or a checkout without libtpu_flags/, stops nothing.  The log's second line is XLA_FLAGS."""
    with open(os.path.join(PKG_DIR, "gpu.xla_flags")) as f:
      gpu = "".join(ln.rstrip("\n") + " " for ln in f if not ln.startswith("#"))
    co = os.path.join(self.tmp, "checkout", *testing.PACKAGE.split("."))
    os.makedirs(co)
    for s in ("ds_ep_multihost.sh", "gpu.xla_flags"):
      shutil.copy(os.path.join(PKG_DIR, s), co)
    cases = (
        (dict(PLATFORM="gpu"), f"{gpu} "),
        (dict(JAX_PLATFORMS="cuda"), f"{gpu} "),
        (dict(JAX_PLATFORMS="gpu,cpu", EXTRA_XLA_FLAGS="--xla_y=2"), f"{gpu} --xla_y=2"),
        (dict(PLATFORM="gpu", XLA_FLAGS_OVERRIDE="--xla_x=1", EXTRA_XLA_FLAGS="--xla_y=2"), "--xla_x=1 --xla_y=2"),
        (dict(PLATFORM="gpu", XLA_FLAGS_OVERRIDE=""), " "),
        (dict(PLATFORM="gpu", MPMD_FLAG_SET="tpu8", EXTRA_LIBTPU_INIT_ARGS="--xla_e=1"), f"{gpu} "),
    )
    for script in (os.path.join(PKG_DIR, "ds_ep_multihost.sh"), os.path.join(co, "ds_ep_multihost.sh")):
      for env, flags in cases:
        with self.subTest(script=script, env=env):
          r = self._run(script, "--stages", "2", XLA_FLAGS="--xla_before=1", **env)
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          seen = _seen(r)
          self.assertEqual((seen["XLA_FLAGS"], seen["LIBTPU_INIT_ARGS"]), (flags, "<unset>"))
          self.assertEqual(self._log(self.run_dir, 0)[1], f"XLA_FLAGS={flags}")

  def test_module_and_environment(self):
    """Each launcher runs its module from the directory the package imports from (testing.import_root: the repository
    root here), wherever it is called from, with the caller's PYTHONPATH and arguments as they are: ds_ep_run.sh
    ds_ep_pipeline (whose own compile cache default holds), ds_ep_aot.sh ds_ep_aot without hash randomization,
    ds_ep_multihost.sh ds_ep_pipeline --distributed with its run record (in a directory of the invocation's own, which
    keeps only the log once the record is copied), unbuffered, without hash randomization, with a compile cache of its
    own by default, and '$RUN_DIR' in an argument replaced by this invocation's directory."""
    args = ["--stages", "2", "two words", ""]
    # PYTHONHASHSEED, PYTHONUNBUFFERED and JAX_CACHE_DIR, when the caller sets none of them
    run_env = {
        "ds_ep_run.sh": ("<unset>", "<unset>", "<unset>"),
        "ds_ep_aot.sh": ("0", "<unset>", "<unset>"),
        "ds_ep_multihost.sh": ("0", "1", "/tmp/jax_cache"),
    }
    for script in LAUNCHERS:
      for inherited in (None, os.pathsep.join(["/some/where", "/else"])):
        with self.subTest(script=script, pythonpath=inherited):
          r = self._run(script, *args, PYTHONPATH=inherited)
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          seen = _seen(r)
          record = [self._record(r)] if script == "ds_ep_multihost.sh" else []
          self.assertEqual(seen["ARGS"], PREFIX[script] + record + args)
          self._only_logs_left(len(record))
          self.assertEqual(os.path.realpath(seen["CWD"]), os.path.realpath(ROOT))
          self.assertEqual(seen["PYTHONPATH"], inherited or "<unset>")
          self.assertEqual((seen["PYTHONHASHSEED"], seen["PYTHONUNBUFFERED"], seen["JAX_CACHE_DIR"]), run_env[script])
      with self.subTest(script=script, jax_cache_dir="given"):
        cache = os.path.join(self.tmp, "cache")
        self.assertEqual(_seen(self._run(script, "--stages", "2", JAX_CACHE_DIR=cache))["JAX_CACHE_DIR"], cache)
      with self.subTest(script=script, argument="$RUN_DIR"):
        got = _seen(self._run(script, "--profile", "$RUN_DIR/xprof"))["ARGS"][-2:]
        here = f"{self.run_dir}/{self.tag}" if script == "ds_ep_multihost.sh" else "$RUN_DIR"
        self.assertEqual(got, ["--profile", f"{here}/xprof"])

  def test_maxtext_dir(self):
    """MAXTEXT_DIR puts that checkout's src/ first on PYTHONPATH for ds_ep_run.sh and ds_ep_aot.sh, which stop before
    the run where `maxtext` does not resolve under it; a relative one is taken from the caller's directory, not
    through an exported CDPATH (a `cd` would go to, and print, a MaxText of that name elsewhere).  The MaxText a run
    takes is a checkout of its own here (a stand-in); where the package is MaxText's own (in MaxText), it is the
    checkout that holds the package, whose maxtext resolves first from the directory the launchers run in."""
    absent = os.path.join(self.tmp, "maxtext")
    os.makedirs(os.path.join(absent, "src"))
    for script in ("ds_ep_run.sh", "ds_ep_aot.sh"):
      with self.subTest(script=script, maxtext="absent"):
        r = self._run(script, "--topology", "v6e-16", MAXTEXT_DIR=absent)
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"{script}: `maxtext` resolves to", r.stderr)
        self.assertNotIn("FAKE", r.stdout)
    own = not testing.is_maxtext("maxtext")  # the package is MaxText's: its checkout is the one to take
    mt = os.path.dirname(testing.import_root()) if own else absent
    decoy = os.path.join(self.tmp, "elsewhere", os.path.basename(mt))
    for d in [decoy] + ([] if own else [mt]):
      os.makedirs(os.path.join(d, "src", "maxtext"))
      open(os.path.join(d, "src", "maxtext", "__init__.py"), "w").close()
    for script in ("ds_ep_run.sh", "ds_ep_aot.sh"):
      for given, cwd, env in (
          (mt, None, {}),
          (os.path.basename(mt), os.path.dirname(mt), {"CDPATH": os.path.dirname(decoy)}),
      ):
        with self.subTest(script=script, maxtext_dir=given, **env):
          r = self._run(script, "--topology", "v6e-16", cwd=cwd, MAXTEXT_DIR=given, PYTHONPATH="/else", **env)
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          seen = _seen(r)
          self.assertEqual(seen["ARGS"], PREFIX[script] + ["--topology", "v6e-16"])
          self.assertEqual(seen["PYTHONPATH"], f"{mt}/src{os.pathsep}/else")

  def test_root_from_the_scripts_place(self):
    """The directory the package imports from is the script's, one level up for each component of the package's
    name, wherever it is called from: copies of the launchers and the flag sets in a scratch checkout run from that
    checkout, with its sets -- here one of blank lines, indented flags and a comment after a flag."""
    co = os.path.join(self.tmp, "checkout")
    d = os.path.join(co, *testing.PACKAGE.split("."))
    shutil.copytree(os.path.join(PKG_DIR, "libtpu_flags"), os.path.join(d, "libtpu_flags"))
    for script in LAUNCHERS:
      shutil.copy(os.path.join(PKG_DIR, script), d)
    with open(os.path.join(d, "libtpu_flags", "mine.flags"), "w") as f:
      f.write(
          "# a set of this checkout's own\n\n  --xla_mine=1   # a comment\n" "\t--xla_tpu_scoped_vmem_limit_kib=4096\n"
      )
    for script in LAUNCHERS:
      with self.subTest(script=script):
        r = self._run(os.path.join(d, script), "--stages", "2", MPMD_FLAG_SET="mine", cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        seen = _seen(r)
        self.assertEqual(os.path.realpath(seen["CWD"]), os.path.realpath(co))
        self.assertEqual(seen["LIBTPU_INIT_ARGS"], composed(["--xla_mine=1", "--xla_tpu_scoped_vmem_limit_kib=4096"]))

  def test_root_through_symlinks_and_cdpath(self):
    """The script's place is where its symlinks lead: through a link to a launcher, or a relative link to the package's
    directory, each launcher runs from this checkout, with its flag sets.  So does a call by relative path from the
    directory the package imports from under an exported CDPATH that finds the package (a `cd` would print it)."""
    links = os.path.join(self.tmp, "links")
    os.makedirs(links)
    pkg_link = os.path.join(links, "pkg")
    os.symlink(os.path.relpath(PKG_DIR, links), pkg_link)
    for script in LAUNCHERS:
      os.symlink(os.path.join(PKG_DIR, script), os.path.join(links, script))
      calls = {
          "script link": (os.path.join(links, script), self.tmp, {}),
          "directory link": (os.path.join(pkg_link, script), self.tmp, {}),
          "CDPATH": (os.path.relpath(os.path.join(PKG_DIR, script), ROOT), ROOT, {"CDPATH": f"{self.tmp}{os.pathsep}."}),
      }
      for how, (path, cwd, env) in calls.items():
        with self.subTest(script=script, via=how):
          r = self._run(path, "--stages", "2", cwd=cwd, MPMD_FLAG_SET="v5p", **env)
          self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
          seen = _seen(r)
          self.assertEqual(os.path.realpath(seen["CWD"]), os.path.realpath(ROOT))
          self.assertEqual(seen["LIBTPU_INIT_ARGS"], composed(flag_set("v5p")))

  def test_multihost_run_records(self):
    """The log -- its first line the host and what it runs, its second the flags, its third the directory it and the
    records live in (where the log stays after the copy), its last the exit status -- goes to RUN_DIR's logs/, the run
    records of process 0 and the stage leaders to json/; a failed run copies no record, not even one an earlier run of
    its tag on this host wrote (test_multihost_stale_files: nor one in /tmp, another user's or not)."""
    tag = self.tag
    run_a, run_b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
    r = self._run("ds_ep_multihost.sh", "--stages", "2", "--profile", "$RUN_DIR/xprof", RUN_DIR=run_a, FAKE_LEADERS="2 3")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    work = os.path.dirname(self._record(r))
    self.assertEqual(
        sorted(os.listdir(f"{run_a}/{tag}/json")), [f"mpmd_{tag}.json", f"mpmd_{tag}.p2.json", f"mpmd_{tag}.p3.json"]
    )
    self.assertEqual(os.listdir(f"{run_a}/{tag}/logs"), [f"mpmd_{tag}_host0.log"])
    log = self._log(run_a, 0)
    self.assertRegex(
        log[0],
        rf"^host 0 \([^)]+\): {re.escape(PIPELINE)} --distributed --stages 2 --profile "
        rf"{re.escape(run_a)}/{tag}/xprof$",
    )
    self.assertEqual(log[1], f"LIBTPU_INIT_ARGS={composed(TPU7X)}")
    self.assertEqual(log[2], f"log and run records: {work} (copied to {run_a}/{tag} at the end)")
    self.assertEqual(log[-1], "host 0 exit 0")
    with open(os.path.join(work, f"mpmd_{tag}_host0.log")) as f:
      self.assertEqual(f.read().splitlines(), log)  # the log kept is the log copied
    r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR=run_b, FAKE_RC="3")
    self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
    self.assertEqual(os.listdir(f"{run_b}/{tag}"), ["logs"])
    self.assertEqual(self._log(run_b, 0)[-1], "host 0 exit 3")
    self._only_logs_left(2)

  def test_multihost_stale_files(self):
    """Files in /tmp under the names main's launcher wrote there, as an earlier run or another user of the same tag
    leaves them: a log and stage leaders' records, read-only (a unit test cannot make another user's files, which the
    owner of a sticky /tmp alone may remove), and a directory where process 0's record would go, which `rm -f` cannot
    remove either.  They neither stop the run nor reach RUN_DIR, which holds this run's log and records only, and are
    left as they were: the run writes in a directory made for it alone."""
    tag = self.tag
    stale = {
        f"/tmp/mpmd_{tag}_host0.log": "another user's log of an earlier run\n",
        f"/tmp/mpmd_{tag}.p2.json": '{"step_ms": 9.0}',
        f"/tmp/mpmd_{tag}.p16.json": '{"step_ms": 9.0}',
    }
    for path, text in stale.items():
      with open(path, "w") as f:
        f.write(text)
      os.chmod(path, 0o444)
    os.mkdir(f"/tmp/mpmd_{tag}.json")
    for rc in ("0", "3"):
      with self.subTest(rc=rc):
        run_dir = os.path.join(self.tmp, f"run{rc}")
        r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR=run_dir, FAKE_RC=rc, FAKE_LEADERS="2")
        self.assertEqual(r.returncode, int(rc), r.stdout + r.stderr)
        self.assertEqual(r.stderr, "")
        log = self._log(run_dir, 0)
        self.assertRegex(log[0], rf"^host 0 \([^)]+\): {re.escape(PIPELINE)} --distributed --stages 2$")
        self.assertEqual(log[-1], f"host 0 exit {rc}")
        if rc == "0":
          records = {}
          for name in os.listdir(f"{run_dir}/{tag}/json"):
            with open(os.path.join(run_dir, tag, "json", name)) as f:
              records[name] = f.read()
          self.assertEqual(
              records, {f"mpmd_{tag}.json": '{"step_ms": 1.0}\n', f"mpmd_{tag}.p2.json": '{"step_ms": 1.0}\n'}
          )
        else:  # a failed run writes no record: nothing to copy
          self.assertEqual(os.listdir(f"{run_dir}/{tag}"), ["logs"])
        for path, text in stale.items():
          with open(path) as f:
            self.assertEqual(f.read(), text)
        self.assertEqual(os.listdir(f"/tmp/mpmd_{tag}.json"), [])
        self._only_logs_left()

  def test_multihost_work_directory(self):
    """The log and the run records live in a directory made for the invocation under TMPDIR (default /tmp), this
    user's only.  Once the copy succeeds the run records (process 0's, a stage leader's) are removed and the log stays
    there, the log copied (a watcher that reads these logs finds it there); when the copy fails (here a RUN_DIR under
    a file) the records stay too and the directory is named on stderr; the run's exit status either way."""
    r = self._run("ds_ep_multihost.sh", "--stages", "2", TMPDIR=None, FAKE_LEADERS="2")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    work = os.path.dirname(_seen(r)["ARGS"][len(PREFIX["ds_ep_multihost.sh"])])
    self.assertRegex(work, r"^/tmp/mpmd\.[A-Za-z0-9]{8}$")
    self.addCleanup(shutil.rmtree, work, ignore_errors=True)  # in /tmp itself: none left behind
    self.assertEqual(
        sorted(os.listdir(f"{self.run_dir}/{self.tag}/json")), [f"mpmd_{self.tag}.json", f"mpmd_{self.tag}.p2.json"]
    )
    self.assertEqual(os.listdir(work), [f"mpmd_{self.tag}_host0.log"])
    self.assertEqual(stat.S_IMODE(os.stat(work).st_mode), 0o700)
    with open(os.path.join(work, f"mpmd_{self.tag}_host0.log")) as f:
      self.assertEqual(f.read().splitlines(), self._log(self.run_dir, 0))
    blocked = os.path.join(self.tmp, "a-file")
    with open(blocked, "w"):
      pass
    for rc in ("0", "3"):
      with self.subTest(copy="fails", rc=rc):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR=os.path.join(blocked, "runs"), FAKE_RC=rc)
        self.assertEqual(r.returncode, int(rc), r.stdout + r.stderr)
        work = os.path.dirname(self._record(r))
        self.assertIn(
            f"ds_ep_multihost.sh: copying to {blocked}/runs/{self.tag} failed: this host's log and run "
            f"records stay in {work}\n",
            r.stderr,
        )
        self.assertEqual(stat.S_IMODE(os.stat(work).st_mode), 0o700)
        kept = [f"mpmd_{self.tag}.json"] if rc == "0" else []
        self.assertEqual(sorted(os.listdir(work)), kept + [f"mpmd_{self.tag}_host0.log"])
        with open(os.path.join(work, f"mpmd_{self.tag}_host0.log")) as f:
          self.assertEqual(f.read().splitlines()[-1], f"host 0 exit {rc}")
        shutil.rmtree(work)

  def test_multihost_without_a_work_directory(self):
    """No directory for the log and records (here a TMPDIR that does not exist; a full or read-only one alike): the host
    runs all the same, without --json, says so on stderr, copies nothing and exits with the run's status, where it
    stopped before the run and left its peers waiting until MPMD_INIT_TIMEOUT."""
    missing = os.path.join(self.tmp, "missing")
    for rc in ("0", "3"):
      with self.subTest(rc=rc):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", TMPDIR=missing, FAKE_RC=rc)
        self.assertEqual(r.returncode, int(rc), r.stdout + r.stderr)
        self.assertEqual(_seen(r)["ARGS"], PREFIX["ds_ep_multihost.sh"][:-1] + ["--stages", "2"])
        self.assertIn(
            f"ds_ep_multihost.sh: no directory under {missing} for this host's log and run records: running "
            "without them",
            r.stderr,
        )
        self.assertIn(f"ds_ep_multihost.sh: nothing to copy to {self.run_dir}/{self.tag}", r.stderr)
        self.assertIn(f"log and run records: none (copied to {self.run_dir}/{self.tag} at the end)", r.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.run_dir, self.tag)))
        self.assertFalse(os.path.exists(missing))

  def test_multihost_to_gcs(self):
    """A gs:// RUN_DIR: the same files, as objects under the bucket's prefix."""
    gcs = os.path.join(self.tmp, "gcs")
    r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR="gs://bucket/some/runs", FAKE_GCS=gcs, FAKE_LEADERS="2")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    prefix = f"bucket/some/runs/{self.tag}"
    got = sorted(os.path.relpath(os.path.join(d, f), gcs) for d, _, files in os.walk(gcs) for f in files)
    self.assertEqual(
        got,
        [
            f"{prefix}/json/mpmd_{self.tag}.json",
            f"{prefix}/json/mpmd_{self.tag}.p2.json",
            f"{prefix}/logs/mpmd_{self.tag}_host0.log",
        ],
    )
    self.assertIn(f"uploaded 3 files to gs://{prefix}\n", r.stdout)

  def test_multihost_run_dir_as_given(self):
    """'$RUN_DIR' in an argument becomes RUN_DIR/STEP_TAG character for character, an '&' or a space in it included,
    so it names the directory the log goes to.  (The '&' needs the replacement quoted under bash 5.2 and later, whose
    patsub_replacement turns an unquoted '&' into the matched text.)"""
    run_dir = os.path.join(self.tmp, "a&b c")
    r = self._run("ds_ep_multihost.sh", "--profile", "$RUN_DIR/xprof", "--out=$RUN_DIR", RUN_DIR=run_dir)
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    here = f"{run_dir}/{self.tag}"
    self.assertEqual(_seen(r)["ARGS"][-3:], ["--profile", f"{here}/xprof", f"--out={here}"])
    self.assertEqual(self._log(run_dir, 0)[-1], "host 0 exit 0")

  def test_multihost_run_dir_trailing_slash(self):
    """A RUN_DIR with trailing '/'s names the directory it names without them: no '//' in the name of an object under
    a gs:// RUN_DIR, nor in what '$RUN_DIR' in an argument becomes."""
    gcs = os.path.join(self.tmp, "gcs")
    r = self._run("ds_ep_multihost.sh", "--profile", "$RUN_DIR/xprof", RUN_DIR="gs://bucket/runs/", FAKE_GCS=gcs)
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    here = f"gs://bucket/runs/{self.tag}"
    self.assertEqual(_seen(r)["ARGS"][-2:], ["--profile", f"{here}/xprof"])
    uploads = sorted(line[len("UPLOAD ") :] for line in r.stdout.splitlines() if line.startswith("UPLOAD "))
    self.assertEqual(uploads, [f"{here}/json/mpmd_{self.tag}.json", f"{here}/logs/mpmd_{self.tag}_host0.log"])
    self.assertIn(f"uploaded 2 files to {here}\n", r.stdout)
    run_dir = os.path.join(self.tmp, "local")
    r = self._run("ds_ep_multihost.sh", "--profile", "$RUN_DIR/xprof", RUN_DIR=run_dir + "//")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    self.assertEqual(_seen(r)["ARGS"][-2:], ["--profile", f"{run_dir}/{self.tag}/xprof"])
    self.assertEqual(self._log(run_dir, 0)[-1], "host 0 exit 0")

  def test_multihost_host_ids(self):
    """The host id names the log, whose first line says where the host is on several slices (a watcher of the logs
    groups by it).  On one slice it is TPU_WORKER_ID, else JOB_COMPLETION_INDEX, else 0, also under a one-slice job's
    MEGASCALE_*; on several, slice x hosts per slice (TPU_WORKER_HOSTNAMES' entries, the empty ones left out) +
    worker: every host of 2 x 4 its own, worker 1 of slice 1 host 5, and up to 8 x 16, in decimal.  MAXTEXT_MPMD_*
    take the place of the TPU job's variables."""
    hosts, many = "h0,h1,h2,h3", ",".join(f"h{i}" for i in range(16))
    cases = [
        ({}, 0, ""),
        (dict(TPU_WORKER_ID="3"), 3, ""),
        (dict(JOB_COMPLETION_INDEX="5"), 5, ""),
        (dict(TPU_WORKER_ID="2", JOB_COMPLETION_INDEX="5"), 2, ""),
        (dict(TPU_WORKER_ID="3", TPU_WORKER_HOSTNAMES=hosts, MEGASCALE_SLICE_ID="0", MEGASCALE_NUM_SLICES="1"), 3, ""),
        (dict(TPU_WORKER_ID="3", TPU_WORKER_HOSTNAMES=hosts, MEGASCALE_NUM_SLICES="1"), 3, ""),
        *[
            (
                dict(
                    TPU_WORKER_ID=str(w), MEGASCALE_SLICE_ID=str(k), MEGASCALE_NUM_SLICES="2", TPU_WORKER_HOSTNAMES=hosts
                ),
                4 * k + w,
                f"slice {k} of 2, worker {w} of 4",
            )
            for k in (0, 1)
            for w in range(4)
        ],
        (
            dict(TPU_WORKER_ID="15", MEGASCALE_SLICE_ID="7", MEGASCALE_NUM_SLICES="8", TPU_WORKER_HOSTNAMES=many),
            127,
            "slice 7 of 8, worker 15 of 16",
        ),
        (
            dict(TPU_WORKER_ID="08", MEGASCALE_SLICE_ID="1", MEGASCALE_NUM_SLICES="8", TPU_WORKER_HOSTNAMES=many),
            24,
            "slice 1 of 8, worker 8 of 16",
        ),
        (
            dict(TPU_WORKER_ID="0", MEGASCALE_SLICE_ID="3", MEGASCALE_NUM_SLICES="4", TPU_WORKER_HOSTNAMES="a,,b,"),
            6,
            "slice 3 of 4, worker 0 of 2",
        ),
        (
            dict(
                MAXTEXT_MPMD_SLICE_ID="2",
                MAXTEXT_MPMD_WORKER_ID="1",
                MAXTEXT_MPMD_NUM_SLICES="3",
                MAXTEXT_MPMD_HOSTS_PER_SLICE="4",
                TPU_WORKER_ID="0",
                MEGASCALE_SLICE_ID="0",
                TPU_WORKER_HOSTNAMES=many,
            ),
            9,
            "slice 2 of 3, worker 1 of 4",
        ),
    ]
    for i, (env, host, where) in enumerate(cases):
      with self.subTest(env=env):
        run_dir = os.path.join(self.tmp, f"ids{i}")
        r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR=run_dir, **env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(os.listdir(f"{run_dir}/{self.tag}/logs"), [f"mpmd_{self.tag}_host{host}.log"])
        at = re.escape(f"{where}, ") if where else ""
        self.assertRegex(
            self._log(run_dir, host)[0], rf"^host {host} \({at}[^,)]+\): {re.escape(PIPELINE)} --distributed --stages 2$"
        )

  def test_multihost_refuses_bad_ids(self):
    """Ids that would collide or pass the job's hosts stop the script before anything runs (exit 2, no directory, no
    log): a value that is not a number; on several slices a worker past the slice's hosts, or no slice id (every slice
    would be slice 0); a slice past the slice count (also a slice id without one)."""
    two = dict(MEGASCALE_SLICE_ID="1", MEGASCALE_NUM_SLICES="2", TPU_WORKER_HOSTNAMES="a,b")
    no_slice = "no slice id (MEGASCALE_SLICE_ID) in a job of 2 slices: every slice would be slice 0"
    cases = [
        (dict(TPU_WORKER_ID="x"), "not a number: 'x'"),
        (dict(TPU_WORKER_ID="1", MEGASCALE_NUM_SLICES="2", TPU_WORKER_HOSTNAMES="a,b"), no_slice),
        (dict(TPU_WORKER_ID="1", **dict(two, MEGASCALE_SLICE_ID="")), no_slice),
        (dict(MAXTEXT_MPMD_NUM_SLICES="2", MAXTEXT_MPMD_WORKER_ID="1", MAXTEXT_MPMD_HOSTS_PER_SLICE="2"), no_slice),
        (dict(MEGASCALE_SLICE_ID="-1"), "not a number: '-1'"),
        (
            dict(TPU_WORKER_ID="1", **dict(two, MEGASCALE_SLICE_ID="2")),
            "slice 2 of a job of 2 (MEGASCALE_NUM_SLICES; 1 when unset): its ids would pass the job's 4 hosts",
        ),
        (dict(TPU_WORKER_ID="1", MEGASCALE_SLICE_ID="1", TPU_WORKER_HOSTNAMES="a,b"), "slice 1 of a job of 1 "),
        (
            dict(TPU_WORKER_ID="4", **two),
            "worker 4 of a slice of 2 hosts (TPU_WORKER_HOSTNAMES, or a GCE TPU VM's worker-network-endpoints): the ids of"
            " 2 slices would collide",
        ),
    ]
    for env, msg in cases:
      with self.subTest(env=env):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", **env)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn(f"ds_ep_multihost.sh: {msg}", r.stderr)
        self.assertEqual(r.stdout, "")
        self.assertFalse(os.path.exists(self.run_dir))
        self.assertEqual(self._work_dirs(), [])

  def test_multihost_refuses_pathways(self):
    """Under Pathways ('proxy' JAX_PLATFORMS's first platform, any case or spacing, as pathways.is_proxy reads it)
    ds_ep_multihost.sh starts nothing -- before even the flag set (an unknown one would exit 1), the host ids, the
    work directory, the log, the run and the copy -- and says to run the one client through ds_ep_run.sh (exit 2).
    'proxy' further down the list is no Pathways run: the host runs as ever."""
    for platforms in ("proxy", "proxy,cpu", " Proxy ", "PROXY,tpu"):
      with self.subTest(JAX_PLATFORMS=platforms):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", JAX_PLATFORMS=platforms, MPMD_FLAG_SET="tpu8")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertEqual(r.stdout, "")
        self.assertRegex(
            r.stderr,
            r"^ds_ep_multihost\.sh: JAX_PLATFORMS=.* is Pathways: one client process drives every device, so there is "
            r"no host of a multi-controller run \(--distributed\) to start: run the one client through ds_ep_run\.sh\n$",
        )
        self.assertFalse(os.path.exists(self.run_dir))
        self.assertEqual(self._work_dirs(), [])
    r = self._run("ds_ep_multihost.sh", "--stages", "2", JAX_PLATFORMS="cpu,proxy")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    self.assertEqual(_seen(r)["ARGS"][:2], ["-u", "-m"])
    self.assertIn("--distributed", _seen(r)["ARGS"])
    self._only_logs_left()

  def test_multihost_log_on_several_slices(self):
    """Under a step agent (a process on each host that runs a job's steps through the launcher, exporting WID, the id
    it took for the host) whose WID is not this host's id -- one that numbers hosts by TPU_WORKER_ID -- the log's
    second line says so: worker 1 of slice 1 of two 4-host slices is host 5, and an agent's WID 1 is not.  Nothing
    outside an agent, under one whose WID is the host's id, or on one slice."""
    two = dict(MEGASCALE_SLICE_ID="1", MEGASCALE_NUM_SLICES="2", TPU_WORKER_ID="1", TPU_WORKER_HOSTNAMES="h0,h1,h2,h3")
    stale = (
        "warning: the agent's WID 1 is not this host's id 5 (slice x hosts per slice + worker): an agent that "
        "numbers hosts by TPU_WORKER_ID -- the slices' step logs collide, and a late or restarted agent skips steps"
    )
    for env, host, second in (
        (two, 5, "LIBTPU_INIT_ARGS="),
        (dict(two, WID="5"), 5, "LIBTPU_INIT_ARGS="),
        (dict(two, WID="05"), 5, "LIBTPU_INIT_ARGS="),
        (dict(two, WID="1"), 5, stale),
        (dict(TPU_WORKER_ID="1", WID="1"), 1, "LIBTPU_INIT_ARGS="),
    ):
      with self.subTest(env=env):
        r = self._run("ds_ep_multihost.sh", "--stages", "2", **env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        log = self._log(self.run_dir, host)
        self.assertRegex(log[0], rf'^host {host} \({"slice 1 of 2, worker 1 of 4, " if host == 5 else ""}[^,)]+\): ')
        self.assertTrue(log[1].startswith(second), log[1])

  def test_multihost_module_and_timeout(self):
    """MODULE runs another entry point the same way, given only the arguments (no --distributed, no run record);
    TIMEOUT kills a run that outlives it (KILL: exit 137), and the host still copies its log."""
    r = self._run("ds_ep_multihost.sh", "--x", "1", MODULE="some.entry_point")
    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
    self.assertEqual(_seen(r)["ARGS"], ["-u", "-m", "some.entry_point", "--x", "1"])
    self.assertRegex(self._log(self.run_dir, 0)[0], r"^host 0 \([^)]+\): some\.entry_point --x 1$")
    self.assertEqual(os.listdir(os.path.join(self.run_dir, self.tag)), ["logs"])
    r = self._run("ds_ep_multihost.sh", "--stages", "2", TIMEOUT="1", FAKE_SLEEP="60")
    self.assertEqual(r.returncode, 137, r.stdout + r.stderr)
    self.assertEqual(self._log(self.run_dir, 0)[-1], "host 0 exit 137")

  def test_multihost_log_is_best_effort(self):
    """A log the host can no longer write stops nothing -- here, from the middle of the run on, a symlink into a
    directory that does not exist, which root cannot write either (a full disk, in a run): the run runs, the record it
    wrote is still copied to RUN_DIR (the log is not: its path leads nowhere, and the link stays where the log would),
    and the script exits with the run's status.  (A log another user left in /tmp under this log's name:
    test_multihost_stale_files.)"""
    for rc in ("0", "3"):
      with self.subTest(rc=rc):
        run_dir = os.path.join(self.tmp, f"middle{rc}")
        log = f"mpmd_{self.tag}_host0.log"
        r = self._run("ds_ep_multihost.sh", "--stages", "2", RUN_DIR=run_dir, FAKE_RC=rc, FAKE_UNWRITABLE=log)
        self.assertEqual(r.returncode, int(rc), r.stdout + r.stderr)
        self.assertEqual(_seen(r)["ARGS"], PREFIX["ds_ep_multihost.sh"] + [self._record(r), "--stages", "2"])
        self.assertIn(f"host 0 exit {rc}\n", r.stdout)
        self.assertIn("tee: ", r.stderr)
        if rc == "0":
          self.assertIn(f"uploaded 1 files to {run_dir}/{self.tag}\n", r.stdout)
          self.assertEqual(os.listdir(os.path.join(run_dir, self.tag)), ["json"])
          self.assertEqual(os.listdir(os.path.join(run_dir, self.tag, "json")), [f"mpmd_{self.tag}.json"])
        else:  # a failed run writes no record: nothing to copy
          self.assertIn(f"uploaded 0 files to {run_dir}/{self.tag}\n", r.stdout)
          self.assertFalse(os.path.exists(run_dir))
        self._only_logs_left()

  @unittest.skipUnless(shutil.which("curl"), "needs curl")
  def test_multihost_host_ids_from_metadata(self):
    """Where the environment names no worker -- a GCE TPU VM's holds none of the ids -- they come from the VM's metadata
    server, as jax.distributed's GCE TPU VM detection takes them (GceTpuCluster): the worker is agent-worker-number,
    the hosts a slice the entries of worker-network-endpoints (in either of its formats), MEGASCALE_* tpu-env's unless
    set.  So the id is the process id jax derives there: worker 1 of 2 hosts host 1 on one slice, host 3 on slice 1
    of 2.  A worker in the environment keeps the script off the server, and so does TPU_SKIP_MDS_QUERY (as it keeps
    jax); with no server, or no worker number there (a GCE VM that is no TPU VM), or not a number, the worker is 0."""
    from jax._src.clusters import cloud_tpu_cluster as ctc

    one = {
        "agent-worker-number": "1",
        "worker-network-endpoints": "t1v-n-0-w-0,t1v-n-0-w-1",
        "tpu-env": "ACCELERATOR_TYPE: 'v5p-16'\nTPU_WORKER_ID: '1'\n",
    }
    two = {
        "agent-worker-number": "1",
        "worker-network-endpoints": "t1v-n-1-w-0:0:10.0.1.2,t1v-n-1-w-1:1:10.0.1.3",
        "tpu-env": "MEGASCALE_NUM_SLICES: '2'\nMEGASCALE_SLICE_ID: '1'\nTPU_WORKER_ID: '1'\n",
    }
    everything = ["agent-worker-number", "worker-network-endpoints", "tpu-env"]
    # (metadata, environment, host id, where, the keys the script asks, whether jax's process id is checked)
    cases = [
        (one, {}, 1, "", everything, True),
        (two, {}, 3, "slice 1 of 2, worker 1 of 2", everything, True),
        (two, dict(MEGASCALE_SLICE_ID="0"), 1, "slice 0 of 2, worker 1 of 2", everything, True),
        (two, dict(TPU_WORKER_ID="0"), 0, "", [], False),
        (two, dict(TPU_SKIP_MDS_QUERY=""), 0, "", [], False),
        ({}, {}, 0, "", ["agent-worker-number"], False),
        ({"agent-worker-number": "<html>a portal</html>"}, {}, 0, "", ["agent-worker-number"], False),
        (None, {}, 0, "", [], False),
    ]
    for i, (attrs, env, host, where, asked, jax_too) in enumerate(cases):
      with self.subTest(metadata=attrs, env=env):
        if attrs is None:  # nothing listens: a port held by a socket that does not accept
          sock = self.enterContext(socket.socket())
          sock.bind(("127.0.0.1", 0))
          address, server = f"127.0.0.1:{sock.getsockname()[1]}", None
        else:
          server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Metadata)
          server.attrs, server.asked = attrs, []
          threading.Thread(target=server.serve_forever, daemon=True).start()
          self.addCleanup(server.server_close)
          self.addCleanup(server.shutdown)
          address = f"127.0.0.1:{server.server_address[1]}"
        run_dir = os.path.join(self.tmp, f"md{i}")
        more = {"TPU_SKIP_MDS_QUERY": None, "GCE_METADATA_IP": address, "RUN_DIR": run_dir, **env}
        r = self._run("ds_ep_multihost.sh", "--stages", "2", **more)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(os.listdir(f"{run_dir}/{self.tag}/logs"), [f"mpmd_{self.tag}_host{host}.log"])
        at = re.escape(f"{where}, ") if where else ""
        self.assertRegex(self._log(run_dir, host)[0], rf"^host {host} \({at}[^,)]+\): ")
        if server is not None:
          self.assertEqual(sorted(server.asked), sorted(asked))
        if jax_too:
          get = lambda key, attrs=attrs: (attrs[key], 200)
          with mock.patch.dict(os.environ, env), mock.patch.object(ctc, "get_metadata", get):
            for k in ("MEGASCALE_SLICE_ID", "MEGASCALE_NUM_SLICES"):
              if k not in env:
                os.environ.pop(k, None)
            self.assertEqual(ctc.GceTpuCluster.get_process_id(), host)


if __name__ == "__main__":
  unittest.main()
