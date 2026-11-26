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

"""xprof on synthetic GPU xplanes: the GPU CAPTURES reading, each rule on a timeline small enough to check by hand.

Built with xprof's own messages, as jax 0.12's profiler writes a GPU capture -- streams as lines, kernels named
after themselves, string stats as ref_values, the scope-range tree on a plane of its own -- so no tensorflow, no
jax and no GPU: the file runs in a fraction of a second on a CPU.

    python3 -m pytest -q tests/unit/experimental/mpmd/jax/xprof_gpu_test.py
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest

from maxtext.experimental.mpmd.jax import xprof

US = 10**6  # picoseconds per microsecond
T0_NS = 56_943_319  # every stream line's timestamp, as one GPU plane's are
GEMM = "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_nn"
NCCL_AG = "ncclDevKernel_AllGather_RING(ncclDevKernelArgsStorage<4096ul>)"
NCCL_RS = "ncclDevKernel_ReduceScatter_bf16_RING(ncclDevKernelArgsStorage<4096ul>)"
NCCL_AR_I32 = "ncclDevKernel_AllReduce_i32_RING(ncclDevKernelArgsStorage<4096ul>)"
NCCL_SR = "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)"
P2P_512K = "kind_src:device kind_dst:device size:524288 dest:2 async:1"
D2D_4K = "kind_src:device kind_dst:device size:4096 dest:0 async:1"
D2H_128M = "kind_src:device kind_dst:pinned size:134217728 dest:0 async:1"
H2D_128M = "kind_src:pinned kind_dst:device size:134217728 dest:0 async:1"


class GpuPlane:
  """One `/device:GPU:N` plane under construction.  An event's string stats are ref_values (each string a stat
  metadata name of its own), its plane's str_values, as in a real file."""

  def __init__(self, space, pl, thread):
    self.space, self.pl, self.thread = space, pl, thread
    self.st, self.ev, self.lines = {}, {}, {}

  def sid(self, name):
    if name not in self.st:
      k = len(self.st) + 1
      self.pl.stat_metadata[k].id = k
      self.pl.stat_metadata[k].name = name
      self.st[name] = k
    return self.st[name]

  def stat(self, msg, name, v, ref=True):
    s = msg.stats.add(metadata_id=self.sid(name))
    if isinstance(v, str):
      if ref:
        s.ref_value = self.sid(v)
      else:
        s.str_value = v
    elif isinstance(v, float):
      s.double_value = v
    else:
      s.int64_value = v

  def stream(self, n):
    if n not in self.lines:
      self.lines[n] = self.pl.lines.add(id=n, name=f"Stream #{n}(Compute)", timestamp_ns=T0_NS)
    return self.lines[n]

  def event(self, stream, kernel, start_us, dur_us, stats=()):
    if kernel not in self.ev:
      k = len(self.ev) + 1
      self.pl.event_metadata[k].id = k
      self.pl.event_metadata[k].name = kernel
      self.ev[kernel] = k
    ln = self.stream(stream)
    e = ln.events.add(
        metadata_id=self.ev[kernel],
        offset_ps=int(round(start_us * US)) - T0_NS * 1000,
        duration_ps=int(round(dur_us * US)),
    )
    for n, v in dict(stats).items():
      if v is not None:
        self.stat(e, n, v)
    return e

  def run(self, module, pid, kernels):
    """One execution of program `module` (`pid`): kernels (stream, name, start us, dur us, hlo_op, name stack[,
    stats]), each under a scope of its own below the run's root -- an NCCL kernel's one level deeper, under its
    async wrapper's, as XLA nests them."""
    root = self.space.scope(self.thread)
    for stream, kernel, s, d, op, path, *extra in kernels:
      parent = self.space.scope(self.thread, root) if "nccl" in kernel.lower() else root
      self.event(
          stream,
          kernel,
          s,
          d,
          {
              "hlo_op": op,
              "hlo_module": module,
              "program_id": pid,
              "name": path,
              "tf_op": "XlaModule:",
              "scope_range_id": self.space.scope(self.thread, parent),
              **(extra[0] if extra else {}),
          },
      )


class GpuSpace:
  """A synthetic GPU host file; `tree` False leaves out the scope-range tree (an older jaxlib's capture)."""

  def __init__(self, tree=True):
    self.sp = xprof._xplane_pb2().XSpace()
    self.count = {}
    self.tree = self.sp.planes.add(name=xprof.SCOPE_TREE) if tree else None
    if tree:  # its one named stat metadata, beside the edges
      self.tree.stat_metadata[7 << 48].id = 7 << 48
      self.tree.stat_metadata[7 << 48].name = "process_id"
      self.tree.stats.add(metadata_id=7 << 48, int64_value=489986)
    env = self.sp.planes.add(name="Task Environment")
    env.stat_metadata[1].name = "profile_start_time"
    env.stats.add(metadata_id=1, int64_value=1_794_208_437_784_689_711)

  def scope(self, thread, parent=None):
    """A fresh scope range id on `thread` (in its top bits, as XLA's), recorded under `parent` in the tree."""
    n = self.count[thread] = self.count.get(thread, 0) + 1
    sid = (thread << 48) + n
    if parent is not None and self.tree is not None:
      self.tree.stats.add(metadata_id=sid, int64_value=parent)
    return sid

  def gpu(self, n):
    g = GpuPlane(self, self.sp.planes.add(name=f"/device:GPU:{n}"), thread=n + 1)
    g.stat(g.pl, "device_vendor", "Nvidia", ref=False)
    g.stat(g.pl, "gpu_device_name", "NVIDIA A100-SXM4-80GB", ref=False)
    g.stat(g.pl, "memory_bandwidth", 2_039_040_000_000)
    return g

  def write(self, d, name="host.xplane.pb"):
    path = os.path.join(d, name)
    with open(path, "wb") as f:
      f.write(self.sp.SerializeToString())
    return path


def pipeline(tree=True, steps=2, offload=False):
  """A two-stage pipeline in miniature, one GPU a stage, `steps` steps 10 ms apart; in us from each step's start:

    GPU:0  jit_fwd_c0(3)    [0, 630]: loop fusion [0, 100], cuBLAS gemm [120, 300], gmm_wi_0 [320, 400] on stream
                            50; beside them on 215 the EP all-gather [200, 500] (scoped) and an unscoped
                            reduce-scatter [520, 600]; a D2D copy [610, 630]
           jit_fwd_c2(148)  [1000, 1400] and [1405, 1805]: two executions back to back, two fusions each
           MemcpyP2P        [1850, 1900], 512 KiB, outside every program: device_put to the next stage
           jit_bwd_c0(483)  [2000, 3000]: gmm_wo_dx [2000, 2600]; a combined EP + FSDP all-gather [2100, 2300]
                            inside it; an i32 all-reduce [2620, 2660], a collective-permute [2700, 2800], XLA's
                            one-warp all-reduce [2820, 2840] (stream 316), a memset [2900, 3000]
    GPU:1  jit_fwd_c1(21)   [1900, 2300]
           SendRecv         [3800, 4000], outside every program: a cross-process receive
           jit_bwd_c1(22)   [4000, 4500]

  `offload` adds --offload-residuals' round trip on GPU:0, outside every program too: 128 MiB of jit_fwd_c0's
  residuals to pinned host memory (MemcpyD2H [650, 800]) and back for jit_bwd_c0 (MemcpyH2D [1840, 1990]).
  """
  s = GpuSpace(tree)
  g0, g1 = s.gpu(0), s.gpu(1)
  for k in range(steps):
    b = 10000 * k
    g0.run(
        "jit_fwd_c0",
        3,
        [
            (50, "loop_gather_fusion", b, 100, "loop_gather_fusion", "jit(fwd_c0)/shard_map"),
            (50, GEMM, b + 120, 180, "custom-call.13", "jit(fwd_c0)/shard_map/dot_general"),
            (215, NCCL_AG, b + 200, 300, "all-gather.2", "jit(fwd_c0)/shard_map/ep_all_gather/mb0/all_gather"),
            (
                50,
                "_ragged_dot_kernel",
                b + 320,
                80,
                "triton_kernel_call.18",
                "jit(fwd_c0)/shard_map/tokamax/gmm_wi_0/triton_kernel_call",
            ),
            (215, NCCL_RS, b + 520, 80, "reduce_scatter.14.1", None),
            (50, "MemcpyD2D", b + 610, 20, "copy.27", None, {"memcpy_details": D2D_4K}),
        ],
    )
    for t in (b + 1000, b + 1405):
      g0.run(
          "jit_fwd_c2",
          148,
          [
              (50, "fusion_86", t, 200, "fusion.86", "jit(fwd_c2)/shard_map/add"),
              (50, "fusion_87", t + 200, 200, "fusion.87", "jit(fwd_c2)/shard_map/mul"),
          ],
      )
    g0.event(58, "MemcpyP2P", b + 1850, 50, {"memcpy_details": P2P_512K})
    if offload:
      g0.event(54, "MemcpyD2H", b + 650, 150, {"memcpy_details": D2H_128M})
      g0.event(55, "MemcpyH2D", b + 1840, 150, {"memcpy_details": H2D_128M})
    g0.run(
        "jit_bwd_c0",
        483,
        [
            (
                50,
                "_ragged_dot_kernel",
                b + 2000,
                600,
                "triton_kernel_call.3",
                "jit(bwd_c0)/shard_map/transpose(jvp(tokamax))/gmm_wo_dx/triton_kernel_call",
            ),
            (
                215,
                NCCL_AG,
                b + 2100,
                200,
                "all-gather.3",
                "jit(bwd_c0)/shard_map/(jvp(ep_all_gather)/mb1/all_gather:fsdp_all_gather/all_gather)",
            ),
            (215, NCCL_AR_I32, b + 2620, 40, "all-reduce.5", None),
            (215, NCCL_SR, b + 2700, 100, "collective-permute.2", None),
            (316, "psum_7_0_fusion_1", b + 2820, 20, "psum.7.0-fusion.1", None),
            (
                50,
                "Memset 3",
                b + 2900,
                100,
                "axis_index.20.0",
                None,
                {"Memset_details": "kind:device num_bytes:4 async:1"},
            ),
        ],
    )
    g1.run("jit_fwd_c1", 21, [(50, "fusion_9", b + 1900, 400, "fusion.9", "jit(fwd_c1)/shard_map/add")])
    g1.event(60, NCCL_SR, b + 3800, 200)
    g1.run("jit_bwd_c1", 22, [(50, "fusion_10", b + 4000, 500, "fusion.10", "jit(bwd_c1)/shard_map/add")])
  return s


def trace(space):
  with tempfile.TemporaryDirectory() as d:
    return xprof.Trace(space.write(d))


def run(cmd, tr, **kw):
  """A subcommand's stdout, with the arg defaults `main` would have parsed."""
  args = argparse.Namespace(device=None, module="jit_bwda_c0", step=-1, width=110, flops=0, bytes=0, min_us=20.0)
  vars(args).update(kw)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    cmd(tr, args)
  return buf.getvalue()


def us(x):
  return x / xprof.PS


class ClassifyTest(unittest.TestCase):
  """What a GPU event is, off its name and hlo_op -- the only things a GPU trace says about it."""

  def test_nccl_kernels_name_their_collective(self):
    cases = [
        (NCCL_AG, "all-gather.2", "all-gather"),
        (NCCL_RS, "reduce_scatter.14.1", "reduce-scatter"),
        ("ncclDevKernel_AllReduce_f32_RING(ncclDevKernelArgsStorage<4096ul>)", "psum.7.1", "all-reduce"),
        (
            "ncclKernel_AllGather_RING_LL_Sum_int8_t(ncclDevComm*, unsigned long, ncclWork*)",
            "all-gather.1",
            "all-gather",
        ),  # an older NCCL's spelling
        ("ncclDevKernel_Broadcast_RING(ncclDevKernelArgsStorage<4096ul>)", "x.1", "collective-broadcast"),
        ("ncclDevKernel_Reduce_f32_RING(ncclDevKernelArgsStorage<4096ul>)", "reduce.1", "reduce"),
    ]
    for kernel, op, want in cases:
      self.assertEqual(xprof.gpu_collective(kernel, op), want, kernel)

  def test_send_recv_is_a_permute_or_an_all_to_all_by_its_hlo_op(self):
    self.assertEqual(xprof.gpu_collective(NCCL_SR, "collective-permute.2"), "collective-permute")
    self.assertEqual(xprof.gpu_collective(NCCL_SR, "all-to-all.3"), "all-to-all")
    self.assertEqual(xprof.gpu_collective(NCCL_SR, "ragged-all-to-all.1"), "all-to-all")
    self.assertEqual(xprof.gpu_collective(NCCL_SR, None), "collective-permute")  # PJRT's, in no program

  def test_a_generic_kernel_takes_its_hlo_op(self):
    gen = "ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)"
    self.assertEqual(xprof.gpu_collective(gen, "all-reduce-start.3"), "all-reduce")
    self.assertEqual(xprof.gpu_collective(gen, "fusion.3"), "other")

  def test_xlas_own_collective_kernels_and_compute(self):
    # XLA's one-warp all-reduce kernel names no library; its hlo_op, a JAX psum's, says what it is
    self.assertEqual(xprof.gpu_collective("psum_7_0_fusion_1", "psum.7.0-fusion.1"), "all-reduce")
    self.assertIsNone(xprof.gpu_collective(GEMM, "custom-call.13"))
    self.assertIsNone(xprof.gpu_collective("loop_add_fusion", "loop_add_fusion"))
    self.assertIsNone(xprof.gpu_collective("_ragged_dot_kernel", "triton_kernel_call.18"))

  def test_hlo_opcode(self):
    self.assertEqual(xprof.hlo_opcode("all_gather.26.1"), "all-gather")
    self.assertEqual(xprof.hlo_opcode("all-reduce-start.3"), "all-reduce")
    self.assertEqual(xprof.hlo_opcode("psum.7.0-fusion.1"), "psum")
    self.assertEqual(xprof.hlo_opcode(None), "")

  def test_copies_and_memsets(self):
    self.assertEqual(xprof.gpu_op_kind("MemcpyP2P", {"memcpy_details": P2P_512K}), ("memcpy", "P2P"))
    self.assertEqual(xprof.gpu_op_kind("MemcpyD2D", {"hlo_op": "all-gather.2"}), ("memcpy", "D2D"))  # not a collective
    self.assertEqual(xprof.gpu_op_kind("Memcpy HtoD (Pinned -> Device)", {}), ("memcpy", "H2D"))
    self.assertEqual(
        xprof.gpu_op_kind("Memcpy", {"memcpy_details": "kind_src:device kind_dst:host size:8"}), ("memcpy", "D2H")
    )
    self.assertEqual(xprof.gpu_op_kind("Memset 3", {"Memset_details": "kind:device num_bytes:4"}), ("memset", None))
    self.assertEqual(xprof.gpu_op_kind(NCCL_AG, {}), ("collective", "all-gather"))
    self.assertEqual(xprof.gpu_op_kind(GEMM, {"hlo_op": "custom-call.13"}), ("compute", None))
    self.assertEqual(xprof.gpu_bytes({"memcpy_details": P2P_512K}), 524288)
    self.assertEqual(xprof.gpu_bytes({"Memset_details": "kind:device num_bytes:4 async:1"}), 4)
    self.assertIsNone(xprof.gpu_bytes({}))

  def test_the_name_stack_is_the_name_stat_not_the_kernels_tf_op(self):
    self.assertEqual(xprof.gpu_path({"name": "jit(f)/a", "tf_op": "XlaModule:"}), "jit(f)/a")
    self.assertIsNone(xprof.gpu_path({"tf_op": "XlaModule:"}))
    self.assertEqual(xprof.gpu_path({"tf_op": "jit(f)/b"}), "jit(f)/b")


class ExecutionsTest(unittest.TestCase):
  """`gpu_executions`: which execution of which program each event belongs to."""

  @staticmethod
  def k(start, mod, prog, sid):
    st = {"hlo_module": mod, "program_id": prog} if mod else {}
    if sid is not None:
      st["scope_range_id"] = sid
    return (start, start + 1, 50, "k", st)

  def test_the_tree_root_is_the_execution_and_splits_back_to_back_runs(self):
    # Two executions of one program, nothing between them on the timeline: root 1 holds 2 and the NCCL kernel 4
    # (under its async wrapper 3); root 5 holds 6.  An event of no program is in no execution.
    tree = {2: 1, 3: 1, 4: 3, 6: 5}
    ks = [self.k(0, "jit_f", 7, 2), self.k(1, "jit_f", 7, 4), self.k(2, None, None, None), self.k(3, "jit_f", 7, 6)]
    self.assertEqual(xprof.gpu_executions(ks, tree), ([[0, 1], [3]], 0))

  def test_without_the_tree_consecutive_events_of_a_program_are_one_execution(self):
    ks = [
        self.k(0, "jit_f", 7, 2),
        self.k(1, "jit_f", 7, 4),
        self.k(3, "jit_f", 7, 6),
        self.k(4, "jit_g", 8, 8),
        self.k(5, "jit_f", 7, 9),
    ]
    self.assertEqual(xprof.gpu_executions(ks, {}), ([[0, 1, 2], [3], [4]], 5))

  def test_a_root_shared_by_two_programs_does_not_merge_them_and_a_cycle_ends(self):
    tree = {2: 1, 3: 1, 10: 11, 11: 10}
    ks = [self.k(0, "jit_f", 7, 2), self.k(1, "jit_g", 8, 3), self.k(2, "jit_h", 9, 10)]
    groups, approx = xprof.gpu_executions(ks, tree)
    self.assertEqual((groups, approx), ([[0], [1], [2]], 0))

  def test_the_scope_tree_plane_skips_its_named_stat(self):
    space = GpuSpace()
    space.scope(1, 5)
    self.assertEqual(xprof.scope_tree(space.tree), {(1 << 48) + 1: 5})


class GpuTraceTest(unittest.TestCase):
  """`Trace._read_gpu`: the TPU's lines, built out of a GPU plane's streams."""

  def test_modules_are_executions_named_as_a_tpu_module_is(self):
    tr = trace(pipeline())
    self.assertEqual(tr.platform, "GPU")
    self.assertEqual(sorted(tr.devices), [0, 1])
    names = [m.name for m in tr.modules(0)]
    self.assertEqual(names, ["jit_fwd_c0(3)", "jit_fwd_c2(148)", "jit_fwd_c2(148)", "jit_bwd_c0(483)"] * 2)
    # the two back-to-back runs, 5 us apart, are two executions; a module spans its events over every stream
    m = tr.modules(0)
    self.assertEqual([(us(x.start), us(x.dur)) for x in m[:4]], [(0, 630), (1000, 400), (1405, 400), (2000, 1000)])
    self.assertEqual((m[0].base, m[0].program_id), ("jit_fwd_c0", "3"))
    self.assertEqual(len(m[0].stats["kernels"]), 6)
    self.assertEqual(tr.n_steps(), 2)
    self.assertTrue(tr.is_mpmd())
    self.assertEqual(tr.notes, [])

  def test_ops_carry_their_hlo_op_and_name_stack(self):
    tr = trace(pipeline())
    ops = tr.ops(0)
    self.assertEqual(len(ops), 2 * (6 + 2 + 2 + 6))  # every event inside a program
    ag = [o for o in ops if o.name == "all-gather.2"][0]
    self.assertEqual(ag.base, "all-gather")
    self.assertEqual(ag.stats["tf_op"], "jit(fwd_c0)/shard_map/ep_all_gather/mb0/all_gather")  # not 'XlaModule:'
    self.assertEqual((ag.stats["kind"], ag.stats["collective"], ag.stats["stream"]), ("collective", "all-gather", 215))
    self.assertEqual(xprof.scope_of(ag), "ep_all_gather")
    rs = [o for o in ops if o.name == "reduce_scatter.14.1"][0]
    self.assertNotIn("tf_op", rs.stats)  # XLA dropped its name stack
    d2d = [o for o in ops if o.name == "copy.27"][0]
    self.assertEqual((d2d.stats["kind"], d2d.stats["memcpy"], d2d.stats["bytes"]), ("memcpy", "D2D", 4096))
    self.assertEqual(ops, sorted(ops, key=lambda e: (e.start, -e.dur)))

  def test_what_runs_outside_every_program_is_a_transfer(self):
    tr = trace(pipeline())
    self.assertEqual(
        [(x.name, x.stats.get("memcpy"), x.stats.get("bytes")) for x in tr.devices[0]["Transfers"]],
        [("MemcpyP2P", "P2P", 524288)] * 2,
    )
    self.assertEqual(
        [(x.stats["kind"], x.stats["collective"]) for x in tr.devices[1]["Transfers"]],
        [("collective", "collective-permute")] * 2,
    )

  def test_tpu_only_stats_degrade_to_none(self):
    tr = trace(pipeline())
    self.assertEqual(tr.peaks[0], (None, 2.03904e12))  # memory_bandwidth; no FLOP/s peak on a GPU plane
    self.assertIsNone(tr.n_steps_line[0])
    self.assertEqual(tr.plane_stats[0]["gpu_device_name"], "NVIDIA A100-SXM4-80GB")
    self.assertEqual(tr.sparsecore, [])

  def test_overlap_is_counted_once(self):
    # The NCCL stream runs beside the compute stream: a sum of op time counts the overlap twice.
    tr = trace(pipeline(steps=1))
    self.assertEqual(us(sum(o.dur for o in tr.ops(0))), 760 + 800 + 1060)
    self.assertEqual(us(xprof.merged(tr.ops(0))), 580 + 800 + 860)

  def test_without_the_tree_back_to_back_executions_merge_with_a_note(self):
    tr = trace(pipeline(tree=False))
    self.assertEqual([m.name for m in tr.modules(0)], ["jit_fwd_c0(3)", "jit_fwd_c2(148)", "jit_bwd_c0(483)"] * 2)
    self.assertEqual(us(tr.modules(0)[1].dur), 805)
    self.assertEqual(len(tr.notes), 1)
    self.assertIn("back-to-back executions of a program read as one", tr.notes[0])
    err = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
      xprof.main(["modules", pipeline(tree=False).write(d)])
    self.assertIn("# 36 GPU events have no scope", err.getvalue())  # every event in a program

  def test_derived_lines_are_not_streams(self):
    # A plane that went through xprof's converter carries its own `XLA Modules` line: not read.  Its event is a copy
    # of a real kernel (stats and all, its scope among them), 1 s after the step: read as a stream, it would join
    # that kernel's execution and stretch jit_fwd_c0 to 1,000,100 us -- with no scope tree, add an execution.
    spans = lambda tr: [(m.name, us(m.start), us(m.dur)) for m in tr.modules(0)]
    for tree in (True, False):
      s = pipeline(tree=tree, steps=1)
      pl = next(p for p in s.sp.planes if p.name == "/device:GPU:0")
      want = trace(s)
      src = next(e for l in pl.lines if l.name.startswith("Stream #") for e in l.events if e.stats)
      ev = pl.lines.add(name="XLA Modules", timestamp_ns=T0_NS).events.add()
      ev.CopyFrom(src)
      ev.offset_ps = src.offset_ps + 10**12
      got = trace(s)
      self.assertEqual(spans(got), spans(want), f"tree={tree}")
      self.assertEqual(len(got.ops(0)), len(want.ops(0)), f"tree={tree}")

  def test_a_tpu_plane_wins_and_neither_is_a_message(self):
    s = pipeline(steps=1)
    tpu = s.sp.planes.add(name="/device:TPU:0")
    tpu.event_metadata[1].name = "jit_f(1)"
    tpu.lines.add(name="XLA Modules").events.add(metadata_id=1, offset_ps=0, duration_ps=10)
    tr = trace(s)
    self.assertEqual((tr.platform, list(tr.devices)), ("TPU", [0]))
    s = GpuSpace()
    with self.assertRaises(SystemExit) as cm:
      trace(s)
    self.assertIn("no /device:TPU:* planes", str(cm.exception))
    self.assertIn("nor /device:GPU:*", str(cm.exception))


class GpuCommandTest(unittest.TestCase):
  """The subcommands on a GPU capture: labels, and the numbers `pipeline`'s docstring gives by hand."""

  @classmethod
  def setUpClass(cls):
    cls.tr = trace(pipeline())

  def test_labels(self):
    self.assertEqual(self.tr.label(1), "GPU:1")
    with self.assertRaises(SystemExit) as cm:
      self.tr.device_list(5)
    self.assertIn("no GPU:5 in", str(cm.exception))
    self.assertIn("it has GPU:0, GPU:1", str(cm.exception))
    out = run(xprof.cmd_timeline, self.tr, step=0, width=45)
    rows = {l.split()[0]: l.split("|")[1] for l in out.splitlines() if l.startswith("GPU:")}
    self.assertEqual(sorted(rows), ["GPU:0", "GPU:1"])  # 100 us a column
    self.assertEqual(rows["GPU:0"], "FFFFFFF   FFFFFFFFF bbbbbbbbbbb" + " " * 14)
    self.assertNotIn(
        "TPU:",
        run(xprof.cmd_modules, self.tr)
        + run(xprof.cmd_gaps, self.tr)
        + run(xprof.cmd_scopes, self.tr)
        + run(xprof.cmd_summary, self.tr),
    )

  def test_summary(self):
    out = run(xprof.cmd_summary, self.tr)
    self.assertIn("2 device planes; 2 step(s)", out)
    self.assertIn("GPU:0 has no `Steps` line", out)
    # step 0 is [0, 4500]: GPU:0's four modules span 2430 us of it, its own window 3000
    self.assertIn("GPU:0         4         3      2.43ms      3.00ms    81.0%     54.0%      2.07ms", out)
    # the op union counts the NCCL stream's overlap once, over a 14.5 ms clock
    self.assertIn("  GPU:0      4.48ms busy    30.9%  (occ/own 34.5% of its own 13.00 ms)", out)
    # kernels against the modules' spans: the 50 + 140 us of gaps inside them, a step
    self.assertIn("  GPU:0      4.48ms of kernels in 4.86ms of programs (92.2%); collective kernels 1.48ms of it", out)
    self.assertIn("outside any program: 2 MemcpyP2P -- 0.10ms, 1.0 MB copied", out)
    self.assertIn("outside any program: 2 ncclDevKernel_SendRecv -- 0.40ms", out)
    self.assertIn("IS the bubble", out)  # the two GPUs share no program: MPMD

  def test_summary_keeps_host_copies_off_the_boundary(self):
    # --offload-residuals' copies run outside every program too, but only the P2P copies and the send/recv are the
    # stage boundary: the host copies, two of 150 us and 128 MiB a step, get a line apart
    out = run(xprof.cmd_summary, trace(pipeline(offload=True)))
    bnd = "\n      (PJRT's copies between GPUs and cross-process send/recv: the stage boundary)"
    self.assertIn("outside any program: 2 MemcpyP2P -- 0.10ms, 1.0 MB copied" + bnd, out)
    self.assertIn("outside any program: 2 ncclDevKernel_SendRecv -- 0.40ms" + bnd, out)
    self.assertIn(
        "outside any program: 2 MemcpyD2H, 2 MemcpyH2D -- 0.60ms, 536.9 MB copied\n      (the rest, not "
        "the boundary: PJRT's copies to and from host memory, --offload-residuals' among them)",
        out,
    )
    self.assertEqual(out.count("the stage boundary)"), 2)

  def test_idle_measures_modules_and_then_kernels(self):
    out = run(xprof.cmd_idle, self.tr)
    # modules leave [630, 1000], [1400, 1405], [1805, 1900] and [3000, 4000] of step 0's 4.5 ms to nobody
    self.assertIn("step 0  [4.50 ms]  nobody working for 1.470 ms (32.67%) in 4 hole(s)", out)
    self.assertIn("GPU:0 fwd          -> GPU:1 fwd", out)
    self.assertIn("GPU:0 bwd          -> GPU:1 bwd", out)
    # the kernels leave 1.66 ms: the holes plus the 190 us of gaps inside the modules; GPU:1's spinning receive
    # [3800, 4000] is in no program, so it is no work
    self.assertIn("kernel level: no kernel on any GPU for 1.660 ms (36.89%), 0.190 ms of it inside a program", out)
    self.assertNotIn("profiler start-up", out)

  def test_ops_is_a_union_and_names_grouped_matmuls(self):
    out = run(xprof.cmd_ops, self.tr, device=0, module="jit_fwd_c0")
    self.assertIn("GPU:0  jit_fwd_c0  n=2  mean module 630.0 us", out)
    self.assertIn("kernels cover 92.06% of the module", out)
    rows = {
        l.split()[0]: l.split()[1:]
        for l in out.splitlines()
        if l.startswith("  ")
        and l.split()[0] in ("all-gather", "custom-call", "gmm_wi_0", "reduce_scatter", "copy", "loop_gather_fusion")
    }
    self.assertEqual(rows["all-gather"], ["1.0", "300.0", "47.6%"])
    self.assertEqual(rows["gmm_wi_0"], ["1.0", "80.0", "12.7%"])  # off its scope, not `triton_kernel_call`
    self.assertEqual(rows["custom-call"], ["1.0", "180.0", "28.6%"])
    self.assertEqual(len(rows), 6)
    self.assertNotIn("inside the `while`", out)

  def test_scopes(self):
    lines = run(xprof.cmd_scopes, self.tr).splitlines()
    fwd = lines.index("  jit_fwd_c0  x1")
    # the gather's 300 us, 120 of it with no compute kernel beside it ([300, 320] and [400, 500])
    self.assertEqual(
        lines[fwd + 1].split(),
        ["ep_all_gather", "1.0", "ops", "0.30", "GPU", "ms", "0.12", "exposed", "ms", "per", "instance"],
    )
    self.assertEqual(lines[fwd + 2].split()[:8], ["reduce-scatter", "(no", "scope)", "1.0", "ops", "0.08", "GPU", "ms"])
    self.assertEqual(lines[fwd + 3], "      kernels per instance: gmm_wi_0 1")
    self.assertEqual(lines[lines.index("  jit_fwd_c2  x2") + 1], "  jit_bwd_c0  x1")
    bwd = {
        " ".join(l.split()[:-10]): l.split()[-10:]
        for l in lines[lines.index("  jit_bwd_c0  x1") + 1 :]
        if l.endswith("per instance")
    }
    self.assertEqual(bwd["fsdp_all_gather+ep_all_gather"][:5], ["1.0", "ops", "0.20", "GPU", "ms"])
    self.assertEqual(bwd["fsdp_all_gather+ep_all_gather"][5], "0.00")  # all of it inside gmm_wo_dx
    self.assertEqual(bwd["all-reduce (no scope)"][:3] + bwd["all-reduce (no scope)"][5:6], ["2.0", "ops", "0.06", "0.06"])
    self.assertEqual(bwd["collective-permute (no scope)"][2], "0.10")
    self.assertIn("      kernels per instance: gmm_wo_dx 1", lines)
    self.assertEqual(
        lines[lines.index("GPU:1") + 1 : lines.index("GPU:1") + 4],
        ["  jit_fwd_c1  x1", "  jit_bwd_c1  x1", "  step total GPU ms: none"],
    )

  def test_gpu_scope_reads_a_combined_collective_part_by_part(self):
    ev = lambda t: xprof.Event(0, 1, "all-gather.3", {"tf_op": t})
    both = "jit(b)/shard_map/(transpose(jvp(ep_reduce_scatter))/mb1/all_gather:jvp(ep_all_gather)/mb0/all_gather)"
    self.assertEqual(xprof.gpu_scope(ev(both)), "ep_all_gather+transpose(ep_reduce_scatter)")
    self.assertEqual(xprof.gpu_scope(ev("jit(b)/shard_map/(fsdp_all_gather/a:fsdp_all_gather/b)")), "fsdp_all_gather")
    self.assertEqual(xprof.gpu_scope(ev("jit(f)/shard_map/ep_all_gather/mb0/all_gather")), "ep_all_gather")
    self.assertIsNone(xprof.gpu_scope(ev("jit(f)/shard_map/(a:b)")))
    # scope_of alone hands the combined kernel to whichever scope comes first in SCOPES
    self.assertEqual(xprof.scope_of(ev(both)), "ep_all_gather")

  def test_costs(self):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
      got = json.loads(run(xprof.cmd_costs, self.tr))
    self.assertEqual(got, {"F0": 0.63, "F1": 0.4, "F2": 0.4, "B0": 1.0, "B1": 0.5, "F": 0.4, "B": 0.5})
    self.assertIn("# no A cost", err.getvalue())

  def test_roofline_scores_the_bandwidth_side(self):
    out = run(xprof.cmd_roofline, self.tr, device=0, module="jit_fwd_c2", flops=4e10, bytes=4.0782e8)
    self.assertIn("GPU:0  jit_fwd_c2  n=4", out)
    self.assertIn("peak from the trace: 2039.04 GB/s (memory_bandwidth)", out)
    self.assertIn("measured: 400.0 us per instance", out)
    self.assertIn("compute:    100.0 TF/s  (no peak in the trace to score it against)", out)
    self.assertIn("= 50.0% of peak", out)
    self.assertIn("bandwidth floor 200.0 us; measured is 2.00x that", out)
    self.assertNotIn("intensity:", out)

  def test_gaps_and_modules(self):
    out = run(xprof.cmd_gaps, self.tr)
    self.assertIn("step lead-in (waiting for the fill)", out)
    row = [l for l in out.splitlines() if l.startswith("GPU:0")][0]
    self.assertIn(f"{2 * (4.5 - 2.43):.2f}ms", row)  # each step's window less its modules
    out = run(xprof.cmd_modules, self.tr, device=0)
    rows = {l.split()[0]: l.split()[1:] for l in out.splitlines() if l.strip().startswith("jit")}
    self.assertEqual(rows["jit_fwd_c2(148)"][:3], ["4", "1.60ms", "400.0us"])


if __name__ == "__main__":
  unittest.main()
