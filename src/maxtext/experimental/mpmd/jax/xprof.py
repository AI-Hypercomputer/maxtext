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

"""Read an MPMD xplane trace from the command line, because the UI cannot.

    python3 -m maxtext.experimental.mpmd.jax.xprof summary  DIR_OR_XPLANE
    python3 -m maxtext.experimental.mpmd.jax.xprof timeline DIR_OR_XPLANE
    python3 -m maxtext.experimental.mpmd.jax.xprof modules  DIR_OR_XPLANE [-d 0]
    python3 -m maxtext.experimental.mpmd.jax.xprof gaps     DIR_OR_XPLANE [-d 0]
    python3 -m maxtext.experimental.mpmd.jax.xprof idle     DIR_OR_XPLANE [--min-us 20]
    python3 -m maxtext.experimental.mpmd.jax.xprof ops      DIR_OR_XPLANE -m jit_bwda_c0 [-d 0]
    python3 -m maxtext.experimental.mpmd.jax.xprof roofline DIR_OR_XPLANE -m jit_bwda_c0
                                           [--flops N --bytes N]
    python3 -m maxtext.experimental.mpmd.jax.xprof scopes   DIR_OR_XPLANE [-d 0] [--step -1]
    python3 -m maxtext.experimental.mpmd.jax.xprof costs    DIR_OR_XPLANE [-d 0] > costs.json

`DIR_OR_XPLANE` is a `.xplane.pb` or any directory above one, so a profile
directory straight out of `ds_ep_pipeline --profile` works; a TPU or a GPU
capture (GPU CAPTURES below says what differs).  `-d` restricts to one
device; without it `ops` and `roofline` pool every device's instances of the
module.

`-m` takes a program's name as `modules` lists it, without the id in
parentheses: `jit_<kind>_c<chunk>` and what else the program carries, e.g.
`jit_fwd_c0_agself`, `jit_bwda_c0_rs2p0` (the scheme is
`ds_ep.ChunkPrograms.program`'s).  A name no program has is taken as a
substring, which must then be in one program's name only: in several, the
command exits and lists them with their instance counts rather than pool
programs that differ (see `_select`).  The default, `jit_bwda_c0`, is chunk
0's plain accumulating backward: a backward past the first microbatch's that
carries no FSDP gather or reduce-scatter and recomputes its forward (not a
stash backward, `_st`).  Not every plan has one -- at S=2, V=2, M=4 under the
default flags each of chunk 0's backwards carries part of chunk 2's
reduce-scatter -- and then the default lists chunk 0's accumulating
backwards, or reads the one there is.  It runs on stage 0, which holds the
first chip (in every layout but physical_origin_last), the one a
single-process capture traces by default; and under its own name: the trace
keys a program by its HLO fingerprint, so identical programs of different
chunks show under whichever name loaded first, but chunk 0's hold the
embedding and match no other.

WHY THIS FILE EXISTS
--------------------
A TensorBoard profile of an SPMD program is one program on N devices, so every
aggregate the UI computes ("step time", "FLOP utilization", the roofline, the
Overview page) is an average over N copies of the same thing, which is the
thing you wanted.  MPMD breaks that assumption in four specific ways, and the
UI does not warn you about any of them:

  1. THERE ARE NO STEPS.  The `Steps` line on an MPMD device has one event per
     *module execution* -- 64 of them for a 2-step capture, 48 on the last
     stage, whose second chunk is one fused forward+backward -- not one per
     training step.  Anything the UI derives from step boundaries (Step Time
     graph, Input Pipeline Analyzer, "steps/sec") is therefore reporting on a
     unit that does not exist.  `summary` recovers the real boundaries from
     the module-name sequence, which repeats once per step on every device.
     Do not try to recover them from the widest gaps instead: on these
     captures the widest all-device-idle gap is sometimes a turnaround inside
     a step and sometimes a hole in the drain.  See `Trace.n_steps`.

  2. EVERY DEVICE RUNS A DIFFERENT PROGRAM.  A device-level aggregate is an
     average over four unrelated executables, which is why `summary` and `gaps`
     always break out per device -- the *shape* across devices is the finding,
     not the mean.  `ops` and `roofline` pool the instances of one program
     over the devices that run it: a stage's devices, and other stages' where
     the trace lists their identical programs under the same name (see 3).
     They never pool programs whose names differ (`_select`).  Pass `-d` when
     you want one stage.

  3. THE TRACE VIEWER SHOWS A WALL OF LOOK-ALIKE BOXES.  One step is dozens
     of module executions per device.  A program's name carries its chunk,
     `_c<chunk>`, but nothing in it carries the microbatch: that is the box's
     place in the step.  Nor is the name always the chunk's own -- the trace
     keys a program by its HLO fingerprint, which leaves names out, so
     identical programs of different chunks all show under whichever name
     loaded first (on a TPU7x-8 capture at PP=4, V=2, stages 1-3 ran every
     chunk but the fused last one under stage 1's names).  The device row
     gives the stage, the order within the step the chunk and the microbatch.
     `timeline` prints the whole step as one line of text per device instead.

  4. OP-LEVEL GAPS SHATTER THE IDLE INTO CONFETTI.  Measured on `XLA Ops`,
     TPU:0 of a v6e-4 capture at mbs 256 has 1,207 gaps; 1,144 of them sit *inside* a
     module, are every one sub-microsecond, and sum to 0.006 ms.  The other 63
     are the module-level gaps and carry the whole 30.9 ms.  The total idle is
     the same either way -- what op granularity destroys is the *distribution*,
     and "idle spread over ~1,200 small gaps" is exactly the fingerprint of
     host dispatch, so reading the trace at op level invents a dispatch problem
     that is not there.  `gaps` therefore works on `XLA Modules` and classifies
     each gap by the pair of modules it separates.  A bubble puts each stage's
     idle in a DIFFERENT place -- stage 0 waits mid-step for the gradient,
     stage 3 waits at the step edges -- while a dispatch stall has no reason to
     land differently per device, and tracks the driver thread.

Peaks are read from the trace's own plane stats, never from a spec sheet.
Note that these captures carry no per-op cost model -- no `flops` or
`bytes_accessed` stat is emitted on any line -- so arithmetic intensity cannot
be read off the trace and `roofline` takes it as an argument.

GPU CAPTURES
------------
A capture with `/device:GPU:N` planes and no TPU one reads the same way, and
the reports call its devices `GPU:N`.  A GPU plane has none of the TPU's
lines: its lines are CUDA streams -- `Stream #50(Memset,Compute,MemcpyD2D)`,
the NCCL kernels on a stream of their own, PJRT's copies on others -- and each
event is one kernel, memcpy or memset, named after itself, with the program it
ran in among its stats.  `Trace._read_gpu` builds the TPU's lines out of them:

  XLA Modules  one event per execution of a program, named `hlo_module(program_id)`
               as a TPU module is, from its first event to its last over every
               stream.  Which execution an event belongs to comes off the
               `/host:__ScopeRangeCallStack__` plane (`gpu_executions`): XLA opens
               one annotation scope per executable run and nests each thunk's
               inside it, so the root of an event's `scope_range_id` chain is its
               execution.  Exact, where cutting at gaps is not: the four forwards
               of a chunk run back to back with sub-microsecond gaps between them,
               as small as the gaps inside one.
  XLA Ops      one event per kernel, memcpy and memset inside a program, named by
               its `hlo_op` (the kernel name where it has none), its `name` stat --
               the JAX name stack, `jit(fwd_c0)/shard_map/ep_all_gather/mb0/
               all_gather` -- kept as `tf_op`, where a TPU op carries it (a GPU
               kernel's own `tf_op` is `XlaModule:`).  Nothing nests -- a `while` is
               its body's kernels again -- but the NCCL stream runs BESIDE the
               compute stream, so op time is a union (`merged`), never a sum, and
               `ops` and `scopes` take every kernel of the execution, not `children`.
  Transfers    whatever runs outside every program: PJRT's copies -- `device_put`
               between local GPUs is MemcpyP2P, to and from pinned host memory
               (--offload-residuals) MemcpyD2H / H2D -- and its cross-process
               NCCL send/recv.  The P2P copies and the send/recv are the stage
               boundary; `summary` prints the rest apart.

There is no `Steps` line, no SparseCore, and no peak FLOP/s stat (the plane
gives `memory_bandwidth`, so `roofline` scores the bandwidth side alone).  And
a GPU module spans the gaps between its kernels, where a host launching kernels
slower than they run shows up, so `summary` and `idle` put the kernels' own
union beside the module figures.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import functools
import glob
import json
import math
import os
import re
import statistics
import sys

PS = 1e6  # picoseconds per microsecond
PS_MS = 1e9  # picoseconds per millisecond


@functools.cache
def _xplane_pb2():
  """The xplane protobuf messages (XSpace and the rest), built once a process
  from tsl/profiler/protobuf/xplane.proto's field numbers -- on first use:
  parsing needs them, running the pipeline does not.

  Never tensorflow's `xplane_pb2`, even where tensorflow is installed.  The two
  agree on every field, so a capture parses and serializes to the same bytes
  through either (xprof_pb_test.MessagesTest), and importing tensorflow costs
  a process some 10 s and 0.7 GB: this module's, every worker xprof_align
  spawns and a pipeline host exporting its trace.
  """
  import types
  from google.protobuf import descriptor_pb2 as D, descriptor_pool, message_factory

  F = D.FieldDescriptorProto
  O, R = F.LABEL_OPTIONAL, F.LABEL_REPEATED
  I, S, B, M, DB, U = (F.TYPE_INT64, F.TYPE_STRING, F.TYPE_BYTES, F.TYPE_MESSAGE, F.TYPE_DOUBLE, F.TYPE_UINT64)
  pk = ".tensorflow.profiler."
  fd = D.FileDescriptorProto(name="xplane_mpmd.proto", package="tensorflow.profiler", syntax="proto3")

  def msg(name, fields, oneof=None, parent=None):
    m = (parent.nested_type if parent else fd.message_type).add(name=name)
    if oneof:
      m.oneof_decl.add(name=oneof)
    for n, num, t, lab, *r in fields:
      f = m.field.add(name=n, number=num, type=t, label=lab)
      if r and r[0]:
        f.type_name = r[0]
      if len(r) > 1:
        f.oneof_index = 0
    return m

  msg("XSpace", [("planes", 1, M, R, pk + "XPlane"), ("errors", 2, S, R), ("warnings", 3, S, R), ("hostnames", 4, S, R)])
  pl = msg(
      "XPlane",
      [
          ("id", 1, I, O),
          ("name", 2, S, O),
          ("lines", 3, M, R, pk + "XLine"),
          ("event_metadata", 4, M, R, pk + "XPlane.EventMetadataEntry"),
          ("stat_metadata", 5, M, R, pk + "XPlane.StatMetadataEntry"),
          ("stats", 6, M, R, pk + "XStat"),
      ],
  )
  for nm, v in (("EventMetadataEntry", "XEventMetadata"), ("StatMetadataEntry", "XStatMetadata")):
    msg(nm, [("key", 1, I, O), ("value", 2, M, O, pk + v)], parent=pl).options.map_entry = True
  msg(
      "XLine",
      [
          ("id", 1, I, O),
          ("display_id", 10, I, O),
          ("name", 2, S, O),
          ("display_name", 11, S, O),
          ("timestamp_ns", 3, I, O),
          ("duration_ps", 9, I, O),
          ("events", 4, M, R, pk + "XEvent"),
      ],
  )
  msg(
      "XEvent",
      [
          ("metadata_id", 1, I, O),
          ("offset_ps", 2, I, O, None, 0),
          ("num_occurrences", 5, I, O, None, 0),
          ("duration_ps", 3, I, O),
          ("stats", 4, M, R, pk + "XStat"),
      ],
      oneof="data",
  )
  msg(
      "XStat",
      [
          ("metadata_id", 1, I, O),
          ("double_value", 2, DB, O, None, 0),
          ("uint64_value", 3, U, O, None, 0),
          ("int64_value", 4, I, O, None, 0),
          ("str_value", 5, S, O, None, 0),
          ("bytes_value", 6, B, O, None, 0),
          ("ref_value", 7, U, O, None, 0),
      ],
      oneof="value",
  )
  msg(
      "XEventMetadata",
      [
          ("id", 1, I, O),
          ("name", 2, S, O),
          ("display_name", 4, S, O),
          ("metadata", 3, B, O),
          ("stats", 5, M, R, pk + "XStat"),
          ("child_id", 6, I, R),
      ],
  )
  msg("XStatMetadata", [("id", 1, I, O), ("name", 2, S, O), ("description", 3, S, O)])
  pool = descriptor_pool.DescriptorPool()
  pool.Add(fd)
  return types.SimpleNamespace(
      **{
          n: message_factory.GetMessageClass(pool.FindMessageTypeByName("tensorflow.profiler." + n))
          for n in ("XSpace", "XPlane", "XLine", "XEvent", "XStat", "XEventMetadata", "XStatMetadata")
      }
  )


def has_device_planes(path):
  """Does the xplane file at `path` hold a device plane (/device:TPU:n or /device:GPU:n)?  Found in its bytes, where a
  plane's name is written as it is, without parsing a capture of up to a few GB."""
  import mmap

  with open(path, "rb") as f:
    if os.fstat(f.fileno()).st_size == 0:
      return False
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
      return m.find(b"/device:TPU:") >= 0 or m.find(b"/device:GPU:") >= 0


# A capture directory's name as jax, pathwaysutils and ds_ep_pipeline name a trace session: a time to the second,
# YYYY_MM_DD_HH_MM_SS, zero-padded, in the naming host's local time (jax takes it when its trace stops, ds_ep_pipeline
# before the profiled steps: pathways.session_id), so that the names of the sessions named in one time zone sort in
# time order (XProf lists sessions by them).
SESSION_NAME = re.compile(r"\d{4}(_\d\d){5}")


def find_xplane(path):
  """Accept a .xplane.pb, or any directory above one, and find it.

  `jax.profiler.trace` writes a fresh `plugins/profile/<timestamp>/` sibling on
  every run, so re-profiling into the same directory leaves several captures
  behind.  Take the newest, and say which -- picking the oldest silently, as
  this used to, means a second run appears to have changed nothing.  Where
  every capture directory is named as a session (SESSION_NAME) the newest is
  the session named last: a copy from gs:// (gcloud storage cp, as a Pathways
  profile needs) gives each file the time it was downloaded, not the time it
  was captured.  (Names given in different time zones -- a cluster's session
  copied beside a workstation's -- do not sort so.)  Otherwise the newest is
  the file written last (its mtime).

  One capture can be several files: a Pathways session (ds_ep_pipeline
  --profile gs://... under JAX_PLATFORMS=proxy, copied here) holds the
  client's own trace (<host>.xplane.pb, client.*.xplane.pb: no device plane)
  beside each traced worker host's (server-<host>.0.0.xplane.pb: the device
  planes) and a small one of the resource manager's.  So within the newest
  capture's directory a file with device planes wins (`has_device_planes`),
  the newest of them.
  """
  if os.path.isfile(path):
    return path
  # escaped: a capture directory may be named after a label with glob characters in it, `hybrid[rand]`,
  # and an unescaped `[...]` is a glob character class that matches nothing here
  hits = glob.glob(os.path.join(glob.escape(path), "**", "*.xplane.pb"), recursive=True)
  if not hits:
    raise SystemExit(f"no *.xplane.pb under {path}")
  capture = {h: os.path.basename(os.path.dirname(h)) for h in hits}  # its directory's name
  by_name = all(SESSION_NAME.fullmatch(name) for name in capture.values())
  hits.sort(key=lambda h: (capture[h] if by_name else "", os.path.getmtime(h), h))
  pick = hits[-1]
  same = [h for h in hits if os.path.dirname(h) == os.path.dirname(pick)]
  if len(same) > 1:
    pick = next((h for h in reversed(same) if has_device_planes(h)), pick)
  if len(hits) > 1:
    if by_name:
      what = "file with device planes" if len(same) > 1 else "file"
      what = f"the newest session's {what} (sessions by name: when each was named)"
    else:
      what = "the newest" if len(same) == 1 else "the newest capture's file with device planes"
    print(f"# {len(hits)} traces under {path}, using {what}:", file=sys.stderr)
    for h in hits:
      print(f'#   {"->" if h == pick else "  "} {h}', file=sys.stderr)
  return pick


# ------------------------------------------------------------------- GPU planes
# A GPU capture's executions are read off this plane (`gpu_executions`): its stats are the scope-range tree, one a
# scope, metadata id the scope's id and value its parent's.  Its only named stat metadata is `process_id`.
SCOPE_TREE = "/host:__ScopeRangeCallStack__"
# The stats of a GPU event anything here reads.  A kernel carries about ten more (correlation_id, occupancy,
# kernel_details, ...), and at ~10^5 kernels a pipeline step, decoding them would be most of the parse.
GPU_STATS = ("hlo_op", "hlo_module", "program_id", "name", "tf_op", "scope_range_id", "memcpy_details", "Memset_details")
# What a collective kernel's `hlo_op` can read, as `hlo_opcode` leaves it: XLA's opcodes, and the JAX primitive
# names its MLIR-spelled instructions keep (`psum.7.1` is an all-reduce, `psum.7.0-fusion.1` XLA's own one-warp
# all-reduce kernel).  send / recv are a collective-permute's halves: `collective-permute` holds send/recv.
_GPU_OPCODES = {
    "all-gather": "all-gather",
    "reduce-scatter": "reduce-scatter",
    "all-reduce": "all-reduce",
    "all-to-all": "all-to-all",
    "ragged-all-to-all": "all-to-all",
    "collective-permute": "collective-permute",
    "send": "collective-permute",
    "recv": "collective-permute",
    "collective-broadcast": "collective-broadcast",
    "psum": "all-reduce",
    "psum-scatter": "reduce-scatter",
    "ppermute": "collective-permute",
}
# NCCL names its device kernels after the collective -- ncclDevKernel_AllGather_RING,
# ncclDevKernel_ReduceScatter_bf16_RING, ncclKernel_AllReduce_RING_LL_Sum_float -- matched with the underscores out,
# longest first (allreduce before reduce).  send/recv (None) is a collective-permute, or an all-to-all, which NCCL
# runs as grouped send/recv: `hlo_op` decides.
_NCCL = (
    ("allgather", "all-gather"),
    ("reducescatter", "reduce-scatter"),
    ("allreduce", "all-reduce"),
    ("alltoall", "all-to-all"),
    ("sendrecv", None),
    ("send", None),
    ("recv", None),
    ("broadcast", "collective-broadcast"),
    ("reduce", "reduce"),
)


def gpu_kernels(pl, lines=None):
  """Every event on a `/device:GPU:N` plane's streams, as (start ps, end ps, stream id, name, {stat: value}).

  Sorted by (start, -duration), as `Trace._events` sorts.  `lines` defaults to
  the plane's own; a caller that parses the lines out of the file's bytes itself
  passes those, with `pl` the plane's metadata.  Only `Stream #` lines are streams (a plane
  that went through xprof's converter also has derived lines, its own `XLA
  Modules` among them), only `GPU_STATS` are read, and a `ref_value` -- how a
  GPU plane stores every string stat -- is decoded to the stat-metadata name it
  points at.
  """
  smd = {k: m.name for k, m in pl.stat_metadata.items()}
  want = {k: n for k, n in smd.items() if n in GPU_STATS}
  emd = pl.event_metadata
  out = []
  for ln in pl.lines if lines is None else lines:
    if not ln.name.startswith("Stream #"):
      continue
    base = ln.timestamp_ns * 1000
    for e in ln.events:
      st = {}
      for s in e.stats:
        n = want.get(s.metadata_id)
        if n is not None:
          w = s.WhichOneof("value")
          v = getattr(s, w) if w else None
          st[n] = smd.get(v, "") if w == "ref_value" else v
      s0 = base + e.offset_ps
      out.append((s0, s0 + e.duration_ps, ln.id, emd[e.metadata_id].name if e.metadata_id in emd else "", st))
  out.sort(key=lambda k: (k[0], -k[1]))
  return out


def scope_tree(pl):
  """The `SCOPE_TREE` plane -> {scope range id: its parent's}."""
  named = {k for k, m in pl.stat_metadata.items() if m.name}
  return {s.metadata_id: s.int64_value for s in pl.stats if s.metadata_id not in named}


def gpu_executions(kernels, tree):
  """`gpu_kernels`' events grouped one index list per program execution, in order of first event; and how many
  events were grouped without the tree.

  An event's execution is the root of its `scope_range_id` chain in `tree`:
  XLA's GPU executable opens one annotation scope per run and every thunk's
  nests inside it -- a kernel's scope is its thunk's, an NCCL kernel's one
  level deeper, under its async wrapper's.  Measured on a 2-step A100 capture
  (5,074 kernels) and a GB200 one (11,912): every event had a root, each root
  one (hlo_module, program_id), and the roots split the back-to-back
  executions of one program (`jit_fwd_c2` x4 a step) that nothing on the
  device timeline does.  The key keeps (hlo_module, program_id) anyway, so a
  root shared by two programs could not merge them.

  Without the tree (older jaxlibs), or for an event whose scope it lacks, a
  run of consecutive events of one (hlo_module, program_id) is one execution
  -- how xprof's own derived `XLA Modules` line read GPU traces before the
  tree: back-to-back executions of one program merge, and two programs on two
  streams at once cut each other into pieces.  Events with no `hlo_module`
  (PJRT's copies, its cross-process NCCL send/recv) are in no execution.
  """
  roots, groups, order, run, approx = {}, {}, [], None, 0
  for i, (_, _, _, _, st) in enumerate(kernels):
    mod = st.get("hlo_module")
    if not mod:
      continue
    prog, sid = st.get("program_id"), st.get("scope_range_id")
    if sid is not None and sid in tree:
      r = roots.get(sid)
      if r is None:
        r, n = sid, 0
        while r in tree and n < 64:  # a cycle cannot hang the walk
          r, n = tree[r], n + 1
        roots[sid] = r
      key, run = ("scope", r, mod, prog), None
    else:
      approx += 1
      if run is None or run[2:] != (mod, prog):
        run = ("run", len(order), mod, prog)
      key = run
    g = groups.get(key)
    if g is None:
      g = groups[key] = []
      order.append(key)
    g.append(i)
  return [groups[k] for k in order], approx


def hlo_opcode(hlo_op):
  """`all_gather.26.1` -> `all-gather`, `all-reduce-start.3` -> `all-reduce`: a GPU event's `hlo_op` with its
  ordinals, async suffix and MLIR underscores off -- the nearest thing to an opcode a GPU trace carries."""
  op = (hlo_op or "").split(".")[0].replace("_", "-")
  for suf in ("-start", "-done", "-update"):
    if op.endswith(suf):
      return op[: -len(suf)]
  return op


def gpu_collective(kernel, hlo_op):
  """The collective a GPU kernel runs (`_GPU_OPCODES`' values, 'reduce', or 'other'), None for any other kernel.

  NCCL's kernels (and NVSHMEM's) are collectives whatever their `hlo_op`, their
  name saying which (`_NCCL`); a kernel of no library is one if its `hlo_op`
  is (XLA's own collective kernels).  Memcpys and memsets are `gpu_op_kind`'s.
  """
  k = kernel.lower()
  op = _GPU_OPCODES.get(hlo_opcode(hlo_op))
  if "nccl" not in k and "nvshmem" not in k:
    return op
  body = k.split("(")[0].replace("_", "")
  for pat, kind in _NCCL:
    if pat in body:
      return kind or (op if op == "all-to-all" else "collective-permute")
  return op or "other"  # ncclDevKernel_Generic: hlo_op or nothing


def gpu_op_kind(kernel, stats):
  """('memcpy', 'P2P' / 'D2D' / 'H2D' / 'D2H'), ('memset', None), ('collective', `gpu_collective`) or
  ('compute', None) of one GPU event."""
  if "memcpy_details" in stats or kernel.startswith("Memcpy"):
    k = kernel[6:].split()[0].upper().replace("TO", "2") if kernel.startswith("Memcpy") and kernel[6:].split() else ""
    if k not in ("P2P", "D2D", "H2D", "D2H", "H2H"):
      m = re.search(r"kind_src:(\w+) kind_dst:(\w+)", stats.get("memcpy_details") or "")
      k = "2".join("D" if x == "device" else "H" for x in m.groups()) if m else "D2D"
    return "memcpy", k
  if "Memset_details" in stats or kernel.startswith("Memset"):
    return "memset", None
  c = gpu_collective(kernel, stats.get("hlo_op"))
  return ("collective", c) if c else ("compute", None)


def gpu_bytes(stats):
  """A memcpy's `size:` (a memset's `num_bytes:`), or None."""
  m = re.search(r"\b(?:size|num_bytes):(\d+)", stats.get("memcpy_details") or stats.get("Memset_details") or "")
  return int(m.group(1)) if m else None


def gpu_path(stats):
  """A GPU event's JAX name stack: its `name` stat, else its `tf_op` unless that is the `XlaModule:` every kernel
  carries; None for neither."""
  t = stats.get("tf_op")
  return stats.get("name") or (t if t and not str(t).startswith("XlaModule") else None)


def gpu_module_name(stats):
  """`jit_fwd_c0(3)`: a GPU execution named as a TPU module is, its program id in parentheses."""
  p = stats.get("program_id")
  return f'{stats.get("hlo_module")}({p})' if p is not None else str(stats.get("hlo_module"))


class Event:
  """One module or op execution, with its XLA cost-model stats attached."""

  __slots__ = ("start", "dur", "name", "stats")

  def __init__(self, start, dur, name, stats):
    self.start, self.dur, self.name, self.stats = start, dur, name, stats

  @property
  def end(self):
    return self.start + self.dur

  @property
  def base(self):
    """`jit_bwda_c0(1731438548)` -> `jit_bwda_c0`; `%fusion.3 = f32[..]` -> `fusion`."""
    n = self.name.split(" = ")[0].lstrip("%")
    return n.split("(")[0].split(".")[0]

  @property
  def program_id(self):
    """The `(12345)` in a module name: the program's id.

    Not what tells chunks apart: a pipeline program's name carries its chunk
    (`_c<chunk>`), and for programs with identical HLO, which show under one
    name, the device and the order within the step do.  Two programs under
    one base name differ here; `is_mpmd` reads it, SPMD running one
    executable everywhere.
    """
    if "(" in self.name and self.name.endswith(")"):
      return self.name[self.name.rindex("(") + 1 : -1]
    return ""


class Trace:
  """One xplane file, flattened to what an MPMD question actually needs."""

  # The only lines anything here reads.  Materialising the rest is not free:
  # `counters_0` is 43,641 zero-duration events per device, 95% of the events
  # in the file, and about a second of parse time that is thrown away.
  DEVICE_LINES = ("XLA Modules", "XLA Ops", "Steps")
  # 'GPU' on a capture read by `_read_gpu`.  A class attribute, so a Trace built
  # without a file (`Trace.__new__`, as xprof_test's are) is a TPU one.
  platform = "TPU"
  notes = ()

  def __init__(self, path):
    self.path = path
    xplane_pb2 = _xplane_pb2()
    from google.protobuf.message import DecodeError

    space = xplane_pb2.XSpace()
    with open(path, "rb") as f:
      try:
        space.ParseFromString(f.read())
      except DecodeError as e:
        # Easy to hit: the `RUN.txt` and the `*.trace.json.gz` that sit beside
        # every capture are both plausible things to name on the command line.
        raise SystemExit(f"{path} is not a serialized XSpace: {e}") from e
    self.devices = {}  # dev -> {line name -> [Event]}
    self.peaks = {}  # dev -> (peak_flops, peak_hbm_bw)
    self.plane_stats = {}  # dev -> {plane stat name -> value}
    self.n_steps_line = {}  # dev -> len(Steps), to show how wrong it is
    self.host = {}  # line name -> [Event], the driver thread
    self.sparsecore = []  # plane names set aside, see below
    self.notes = []  # what a reading had to approximate; `main` prints them
    gpu, tree = [], {}  # GPU planes, read only where there is no TPU one
    for pl in space.planes:
      if pl.name.startswith("/device:TPU"):
        # `/device:TPU:0 SparseCore 0` is a plane too, on chips that have
        # SparseCores (v6e has two per chip).  It has its own line types --
        # `Sparse Core Steps`, `TEC 0..15` -- and none of the `DEVICE_LINES`,
        # so everything below is about the TensorCore planes, keyed by the
        # bare device number.  MoE gathers and scatters are what land there.
        tail = pl.name.rsplit(":", 1)[1]
        if not tail.isdigit():
          self.sparsecore.append(pl.name)
          continue
        d = int(tail)
        meta = self._op_metadata(pl)
        self.devices[d] = {ln.name: self._events(pl, ln, meta) for ln in pl.lines if ln.name in self.DEVICE_LINES}
        self.n_steps_line[d] = len(self.devices[d].get("Steps", []))
        st = {pl.stat_metadata[s.metadata_id].name: s for s in pl.stats}
        # e.g. `has_megacore`: on v5p two TensorCore planes are ONE chip
        self.plane_stats[d] = {
            k: getattr(v, v.WhichOneof("value")) if v.WhichOneof("value") else None for k, v in st.items()
        }
        self.peaks[d] = (
            st["peak_teraflops_per_second"].double_value * 1e12 if "peak_teraflops_per_second" in st else None,
            st["peak_hbm_bw_gigabytes_per_second"].double_value * 1e9
            if "peak_hbm_bw_gigabytes_per_second" in st
            else None,
        )
      elif pl.name == "/host:CPU":
        self.host = {ln.name: self._events(pl, ln) for ln in pl.lines if ln.name.startswith(("python", "train.py"))}
      elif pl.name.startswith("/device:GPU:"):
        gpu.append(pl)
      elif pl.name == SCOPE_TREE:
        tree = scope_tree(pl)
    if not self.devices and gpu:
      self._read_gpu(gpu, tree)
    if not self.devices:
      raise SystemExit(f"no /device:TPU:* planes in {path}, nor /device:GPU:*")

  def _read_gpu(self, planes, tree):
    """The GPU planes -> the per-device lines the TPU planes carry natively (GPU CAPTURES, in the module docstring).

    `XLA Ops` holds every event inside a program and `Transfers` the ones
    outside every program, each an Event named by its `hlo_op` (else its own
    name) with stats `kernel` (that name), `stream` (the line id), `kind`
    (`gpu_op_kind`'s, its detail under the same key: `collective`
    'all-gather', `memcpy` 'P2P'), `hlo_module` / `program_id`, `tf_op` (the
    JAX name stack, `gpu_path`) and `bytes` (a copy's).  An `XLA Modules` event
    keeps its execution's events, in start order, as `kernels`.  Peaks:
    (None, `memory_bandwidth`) -- a GPU plane carries no FLOP/s peak.
    """
    self.platform = "GPU"
    approx = 0
    for pl in planes:
      tail = pl.name.rsplit(":", 1)[1]
      if not tail.isdigit():
        continue
      d = int(tail)
      ks = gpu_kernels(pl)
      evs = []
      for s, e, stream, name, st in ks:
        kind, what = gpu_op_kind(name, st)
        stats = {"kernel": name, "stream": stream, "kind": kind}
        if what:
          stats[kind] = what
        if st.get("hlo_module"):  # in a program, as `gpu_executions` reads it
          stats["hlo_module"] = st["hlo_module"]
          if st.get("program_id") is not None:
            stats["program_id"] = st["program_id"]
        path, nb = gpu_path(st), gpu_bytes(st)
        if path:
          stats["tf_op"] = path
        if nb is not None:
          stats["bytes"] = nb
        evs.append(Event(s, e - s, st.get("hlo_op") or name, stats))
      groups, n = gpu_executions(ks, tree)
      approx += n
      mods = []
      for idx in groups:
        mine = [evs[i] for i in idx]
        lo, hi = min(m.start for m in mine), max(m.end for m in mine)
        mods.append(
            Event(
                lo, hi - lo, gpu_module_name(ks[idx[0]][4]), {"hlo_module": ks[idx[0]][4]["hlo_module"], "kernels": mine}
            )
        )
      mods.sort(key=lambda ev: (ev.start, -ev.dur))
      self.devices[d] = {
          "XLA Modules": mods,
          "XLA Ops": [ev for ev in evs if "hlo_module" in ev.stats],
          "Transfers": [ev for ev in evs if "hlo_module" not in ev.stats],
      }
      self.n_steps_line[d] = None
      smd, ps = {k: m.name for k, m in pl.stat_metadata.items()}, {}
      for s in pl.stats:  # gpu_device_name, clock_rate, core_count, memory_bandwidth, ...
        w = s.WhichOneof("value")
        v = getattr(s, w) if w else None
        ps[smd.get(s.metadata_id, "?")] = smd.get(v) if w == "ref_value" else v
      self.plane_stats[d] = ps
      bw = ps.get("memory_bandwidth")
      self.peaks[d] = (None, float(bw) if bw else None)
    if approx:
      self.notes.append(
          f"{approx} GPU events have no scope in the {SCOPE_TREE} tree: each run of consecutive "
          "events of one program counts as one execution there, so back-to-back executions of a "
          "program read as one (`gpu_executions`)"
      )

  def label(self, d):
    """`TPU:0` or `GPU:0`: the device as every report names it."""
    return f"{self.platform}:{d}"

  # Stats of an op's METADATA (not of each event) worth attaching to its
  # events.  `tf_op` is the HLO op_name, i.e. the JAX name stack: where
  # `jax.named_scope` lands -- `fsdp_all_gather` and the rest (see
  # `cmd_scopes`) -- and what the trace viewer's "Framework Name Scope" row is
  # built from.  `hlo_category` is XLA's own class of the op ("convolution
  # fusion", "all-reduce-scatter fusion", ...): the only marker that a TPU
  # collective fused into `fusion.N` is a collective at all.
  OP_METADATA = ("tf_op", "hlo_category")

  @classmethod
  def _op_metadata(cls, pl):
    """Op metadata id -> {stat: value} for the `OP_METADATA` stats it carries.

    They are stats of the op's *metadata*, not of each event, so `_events`
    would never see them.
    """
    ids = {m.name: k for k, m in pl.stat_metadata.items()}
    want = {ids[n]: n for n in cls.OP_METADATA if n in ids}
    out = {}
    if not want:
      return out
    for mid, em in pl.event_metadata.items():
      for s in em.stats:
        name = want.get(s.metadata_id)
        if name is not None:
          out.setdefault(mid, {})[name] = s.str_value or (
              pl.stat_metadata[s.ref_value].name if s.ref_value in pl.stat_metadata else ""
          )
    return out

  @staticmethod
  def _events(pl, ln, meta=None):
    out = []
    for e in ln.events:
      stats = {}
      for s in e.stats:
        nm = pl.stat_metadata[s.metadata_id].name
        stats[nm] = getattr(s, s.WhichOneof("value")) if s.WhichOneof("value") else None
      if meta and e.metadata_id in meta:
        stats.update(meta[e.metadata_id])
      out.append(Event(ln.timestamp_ns * 1000 + e.offset_ps, e.duration_ps, pl.event_metadata[e.metadata_id].name, stats))
    out.sort(key=lambda ev: (ev.start, -ev.dur))
    return out

  def modules(self, d):
    return self.devices[d].get("XLA Modules", [])

  def ops(self, d):
    return self.devices[d].get("XLA Ops", [])

  def device_list(self, d):
    """`-d` resolved to a list of devices, with a diagnostic on a bad index."""
    if d is None:
      return sorted(self.devices)
    if d not in self.devices:
      have = ", ".join(self.label(x) for x in sorted(self.devices))
      raise SystemExit(f"no {self.label(d)} in {self.path} -- it has {have}")
    return [d]

  def n_steps(self):
    """How many training steps the capture holds, from module-name periodicity.

    A step runs the same modules in the same order every time, so each device's
    module-name sequence is k identical blocks and k is the step count.  That
    is exact and needs no timing threshold, which matters because the obvious
    alternative -- cut at the widest gaps in the union of the device timelines,
    where the end-of-step `block_until_ready` shows up -- is wrong:

      * On the mbs-256 gpipe capture the forward->backward turnaround leaves
        all four devices idle for 108 and 130 us, and a threshold under those
        reports 4 steps for a 2-step run.
      * On the mbs-4096 1f1b capture the widest union gap (646 us) is inside
        the drain, not at the edge -- the real edge is the *narrowest* of the
        five gaps over 100 us (646, 640, 638, 637, 582) -- so cutting at the
        widest reports walls of 476 and 14 ms for a run whose steps are both
        245 ms.

    There is no threshold between those two, which is why this does not use
    one.  Returns 1 if nothing repeats -- a single-step capture, or one the
    profiler truncated; `summary` prints the union gaps next to the boundaries
    it chose so a truncated capture is visible rather than silent.
    """
    if getattr(self, "_n_steps", None) is not None:
      return self._n_steps
    seqs = [[m.name for m in self.modules(d)] for d in sorted(self.devices)]
    seqs = [s for s in seqs if s]
    if not seqs:
      self._n_steps = 0
      return 0
    g = 0
    for s in seqs:  # k must divide every device's count
      g = math.gcd(g, len(s))
    best = 1
    for k in range(2, g + 1):
      if g % k == 0 and all(all(nm == s[i % (len(s) // k)] for i, nm in enumerate(s)) for s in seqs):
        best = k
    self._n_steps = best
    return best

  def is_mpmd(self):
    """Do the devices run different programs, or one program replicated?

    Worth asking before printing any of the pipeline readings, because this
    tool gets pointed at the SPMD baseline too: on a MaxText capture "step
    lead-in (waiting for the fill)" and "that spread IS the bubble" are not so
    much wrong as meaningless, and a reader has no way to tell.

    The test is whether any program id runs on more than one device.  Names
    will not do it: MaxText's TPU:0 additionally runs the `jit__multi_slice`
    staging programs, so a name-multiset comparison calls it MPMD.  Program
    ids are exact -- SPMD replicates one executable, so all four devices share
    its id, while every MPMD chunk lives on exactly one device.
    """
    ids = [{m.program_id for m in self.modules(d)} for d in self.devices]
    ids = [s for s in ids if s]
    return len(ids) > 1 and not set.intersection(*ids)

  def steps(self):
    """Each step as a (start, end) window in picoseconds.

    The window runs from the first module to start on ANY device to the last to
    finish, so it contains the pipeline fill and drain.  That is the
    denominator a pipeline should be judged on: a stage that goes quiet early
    is idle, not finished.
    """
    k = self.n_steps()
    if not k:
      return []
    out = []
    for i in range(k):
      lo, hi = None, None
      for d in self.devices:
        mods = self.modules(d)
        if not mods:
          continue
        blk = len(mods) // k
        a, b = mods[i * blk].start, mods[(i + 1) * blk - 1].end
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
      out.append((lo, hi))
    return out

  def union_gaps(self):
    """Every gap in the union of the device timelines, longest first.

    `steps` does not use these -- but a threshold-based reading would, and
    `summary` prints them beside the boundaries it chose so you can see for
    yourself that the widest gap is not always the step edge.
    """
    spans = sorted((m.start, m.end) for d in self.devices for m in self.modules(d))
    if not spans:
      return []
    gaps, cur = [], spans[0][1]
    for s, e in spans[1:]:
      if s > cur:
        gaps.append(s - cur)
      cur = max(cur, e)
    return sorted(gaps, reverse=True)

  def all_idle(self, lo, hi):
    """Intervals within [lo, hi) when NO device is running any module.

    Deliberately not what `gaps` measures, and the difference is the whole
    point.  A pipeline bubble is one stage waiting while another works: it is
    the price of the schedule, it is bounded by `(S-1)/(M*V + S-1)`, and you
    buy throughput with it.  An interval where *every* chip is idle buys
    nothing.  Reading one for the other is easy -- the trace viewer draws them
    identically -- and it is why a 30.9 ms per-device idle figure and a 0.16 ms
    nobody-is-working figure can both be true of the same step.

    Returns `(start, end, before, after)` per interval, where `before` is the
    `(device, event)` whose end opened it and `after` the `(device, event)`
    pairs that close it, so the caller can name the handoff it sits in.
    """
    # The clip is redundant given the overlap filter -- checked against all 38
    # step windows of 19 v6e-4 captures, it changes nothing -- and it is
    # here anyway so that every span is visibly inside the window at the point
    # you read it, rather than only if you also reason about the filter.
    spans = sorted(
        (max(m.start, lo), min(m.end, hi), d, m)
        for d in self.devices
        for m in self.modules(d)
        if m.end > lo and m.start < hi
    )
    out, cur, prev = [], lo, None
    for s, e, d, m in spans:
      if s > cur:
        out.append((cur, s, prev, [(dd, mm) for ss, _, dd, mm in spans if ss == s]))
      if e > cur:
        cur, prev = e, (d, m)
    if hi > cur:
      out.append((cur, hi, prev, []))
    return out

  def framing(self):
    """The capture's outermost host frames: what is program, what is profiler.

    A trace file is not a recording of your program, it is a recording of the
    process that hosted it, and on these captures the program is the minority
    of it -- 102 ms of device work inside a 627 ms file, the other 525 ms being
    `jax.profiler.trace`'s own `__enter__` and `__exit__`.  Measured over the
    19 v6e-4 captures the profiler's share of the file runs 40% to 84%,
    mean 59%; quote the range, not the 84%, which is the worst single file.
    Opened in a viewer, that is most of the width, every pixel of it with four
    idle chips, and it looks exactly like a catastrophic pipeline stall.  It is
    not; it is the profiler starting up and then serializing what it captured.

    Teardown dominates and does not scale with the step: 397-499 ms across a
    2.3x step-size and 4x model sweep, because what it serializes is ~52 MB of
    which 83-86% is the fixed `counters_0` sampling.  Do not go looking for the
    cause of it in your own code.

    Returns `(start_up, work_lo, work_hi, tear_down, frames)` in picoseconds,
    with `frames` the outermost host frames in order for the caller to print.
    """
    if not self.host:
      return 0, None, None, 0, []
    line = max(self.host.values(), key=lambda v: sum(o.name.startswith("PjitFunction") for o in v))
    frames = children(line, min(o.start for o in line), max(o.end for o in line) + 1) if line else []
    lo = min((m.start for d in self.devices for m in self.modules(d)), default=None)
    hi = max((m.end for d in self.devices for m in self.modules(d)), default=None)
    up = sum(f.dur for f in frames if "__enter__" in f.name)
    down = sum(f.dur for f in frames if "__exit__" in f.name)
    return up, lo, hi, down, frames

  def driver_busy(self, lo, hi):
    """Wall the Python driver thread spends issuing work, within [lo, hi].

    One thread issues every module execution and every transfer in the whole
    pipeline, so its duty cycle is the ceiling MPMD dispatch can support: at
    100% the next step cannot be enqueued until the current one is done.

    The thread's whole life is six outermost frames: `jax.profiler.trace`'s
    `__enter__`, then per step a frame of the driving script that issues (a
    `<lambda>` on these captures) and an `$api.py block_until_ready` that
    waits, then `__exit__`.  Count the issuing frames -- the ones holding a
    `PjitFunction` -- clipped to the window.

    Do not measure this as the sum of `PjRtCApiLoadedExecutable::Execute`
    instead: that is the innermost leaf of a dispatch and misses the tracing,
    the argument handling and all 112 transfer sends per step, which on the
    mbs-256 capture is half of the real figure.  Nor as the residual
    `window - block_until_ready`, which is the same number on fifteen of the
    nineteen captures and wrong on the other four: the window comes from the
    device planes and this line from the host plane, and where the two clocks
    are far apart -- 4.2 ms on mbs-256 gpipe, the first capture its process
    took, whose `start_trace` ran 47 ms instead of 26 -- the window opens
    before the first dispatch and the residual charges the profiler's own
    start-up as work, reporting 14.9 ms against 10.7 for the step after it.
    """
    # `self.host` is keyed by whatever the plane calls the interpreter line --
    # `python3` on these captures, but that is the binary's name, and a host
    # plane can carry one line per thread.  Take the line doing the dispatching.
    line = max(self.host.values(), default=[], key=lambda v: sum(o.name.startswith("PjitFunction") for o in v))
    if not line:
      return 0
    busy = 0
    for e in children(line, min(x.start for x in line), max(x.end for x in line) + 1):
      if not any(o.name.startswith("PjitFunction") and e.start <= o.start and o.end <= e.end for o in line):
        continue
      s, t = max(e.start, lo), min(e.end, hi)
      if t > s:
        busy += t - s
    return busy


# ------------------------------------------------------------------ containment
def children(events, start, end):
  """Direct children of [start, end): outermost non-overlapping contained ops.

  The `XLA Ops` line is flat but the events nest by time, so a plain sum
  double-counts a `while` against its own body.  One greedy left-to-right pass
  over events sorted by (start, -duration) keeps only the outermost ones.

  `events` must already be sorted that way -- `Trace._events` does it.  Passing
  an unsorted list silently returns the wrong answer, because the pass relies on
  a container arriving before anything it contains.
  """
  out, cur = [], -1
  for e in events:
    if e.start >= start and e.end <= end and e.dur > 0 and e.start >= cur:
      out.append(e)
      cur = e.end
  return out


def merged(events):
  """Wall covered by `events`, overlaps counted once.

  Module busy does not compare across MPMD and SPMD: MaxText runs the whole
  training step as one program, so the sum of its `XLA Modules` durations is
  99.6% of the capture by construction, whatever the chip is doing.  The union
  of the `XLA Ops` intervals is the same measurement on both.

  `events` must be sorted by start -- `Trace._events` does it.
  """
  tot, cur = 0, None
  for e in events:
    if cur is None or e.start > cur[1]:
      tot += cur[1] - cur[0] if cur else 0
      cur = [e.start, e.end]
    else:
      cur[1] = max(cur[1], e.end)
  return tot + (cur[1] - cur[0] if cur else 0)


def _merge_spans(spans, lo=-math.inf, hi=math.inf):
  """(start, end) `spans`, in any order, clipped to [lo, hi) and merged: sorted disjoint [start, end] pairs."""
  out = []
  for s, e in sorted((max(s, lo), min(e, hi)) for s, e in spans if e > lo and s < hi):
    if out and s <= out[-1][1]:
      out[-1][1] = max(out[-1][1], e)
    else:
      out.append([s, e])
  return out


def _span_union(spans, lo, hi):
  """Wall covered by (start, end) `spans` inside [lo, hi), overlaps counted once."""
  return sum(e - s for s, e in _merge_spans(spans, lo, hi))


def _overlap(union, s, e):
  """How much of [s, e) the `_merge_spans` pairs `union` cover."""
  return sum(min(b, e) - max(a, s) for a, b in union if a < e and b > s)


def _select(tr, devs, want):
  """Instances of module `want` on `devs`, as (device, event) pairs.

  An exact base-name match wins if there is one; failing that, `want` is
  taken as a substring, but only where it picks out one program.  A substring
  pools the program you asked for with every one whose name extends it, and
  reports a blend that is none of them: on one v6e-4 capture `-m
  jit__bwd` caught the 14 first-microbatch backwards (1151.8 us each) *and*
  the 98 accumulating `jit__bwda` (2015.5 us), and reported 1907.6 us.  The
  pipeline's names extend each other the same way -- `jit_fwd_c0` and
  `jit_fwd_c0_agself`, `jit_bwda_c0` and `jit_bwda_c0_rs2p0` -- and a capture
  need not hold the plain program at all: at S=2, V=2, M=4 under the default
  flags every backward of chunk 0 carries part of chunk 2's reduce-scatter, so
  `-m jit_bwda_c0` would pool `jit_bwda_c0_rs2p0`, `_rs2p5` and `_rs2p6`.  A
  substring in several programs' names therefore exits, listing them with
  their instance counts for you to pick one.  One in a single program's name
  -- a bare program id, or `jit_bwda_c0` where chunk 0's only accumulating
  backward is `jit_bwda_c0_rsself` -- reads that program, and stderr names it.
  Exact-wins means the name you read off `modules` selects exactly that row.
  """
  pairs = [(d, m) for d in devs for m in tr.modules(d)]
  exact = [(d, m) for d, m in pairs if m.base == want]
  if exact:
    return exact
  sub = [(d, m) for d, m in pairs if want in m.name]
  bases = collections.Counter(m.base for _, m in sub)
  if len(bases) > 1:
    rows = "\n".join(f"  {n:6d}x  {b}" for b, n in sorted(bases.items()))
    raise SystemExit(
        f"no program on device(s) {devs} is named {want!r}, and {len(bases)} have it in their names -- "
        f"pooled, they would report a blend that is none of them; pick one with -m:\n{rows}"
    )
  if sub:
    print(
        f"# no program on device(s) {devs} is named {want!r}: reading {sub[0][1].base}, whose name has it",
        file=sys.stderr,
    )
  return sub


def uncontended(durs):
  """A stall inside a module inflates the mean; the median survives it.

  Returns (n, mean, median, p10) in the same unit as the input.  `p10` is the
  nearest-rank 10th percentile, `ceil(0.10 n)`th smallest -- for the n=14 and
  n=28 samples these captures give, that is the 2nd and 3rd, not the 1st and
  2nd that truncation would pick.
  """
  if not durs:
    return (0, 0.0, 0.0, 0.0)
  s = sorted(durs)
  return (len(s), sum(s) / len(s), statistics.median(s), s[-(-len(s) // 10) - 1])


# ------------------------------------------------------------------- subcommands
def cmd_summary(tr, args):
  steps = tr.steps()
  gaps = tr.union_gaps()
  print(f"{tr.path}\n")
  counts = ", ".join(f"{len(tr.modules(d))}" for d in sorted(tr.devices))
  print(
      f"{len(tr.devices)} device planes; {len(steps)} step(s), because the "
      f"module-name sequence\n  ({counts} modules per device) is that many "
      "identical repeats. No timing threshold."
  )
  for d in sorted(tr.devices):
    if tr.n_steps_line[d] is None:  # a GPU plane: no such line
      print(
          f"  {tr.label(d)} has no `Steps` line; its modules are its program executions, read off its "
          f"kernels\n  (GPU CAPTURES in this file's docstring)"
      )
      break
    print(
        f"  the `Steps` line on TPU:{d} claims " f'{tr.n_steps_line[d]} "steps" -- that is module executions, ignore it'
    )
    break
  between = [b[0] - a[1] for a, b in zip(steps, steps[1:])]
  if between:
    rest = list(gaps)
    for g in between:  # a boundary IS a union gap; drop it
      if g in rest:
        rest.remove(g)
    print("  all devices go quiet for " + ", ".join(f"{g / PS_MS:.2f}ms" for g in between) + " between steps")
    decoys = [g for g in rest if g >= min(between)]
    if decoys:
      print(
          f"  {len(decoys)} gap(s) INSIDE a step are just as wide (up to "
          f"{max(decoys) / PS:.0f}us) -- a threshold would\n  have split "
          "there and been wrong; see `Trace.n_steps`"
      )
    elif rest:
      print(
          f"  the other {len(rest)} union gaps are all narrower, "
          f"largest {rest[0] / PS:.1f}us -- transfer latency, not step edges"
      )
  elif gaps:
    # One step means nothing repeated, which is either the truth or a capture
    # the profiler cut short.  Say so, and show the widest hole: if that is a
    # step edge, every per-step number below is really several steps merged.
    print(
        f"  only 1 step, so no boundary to show. The widest all-device gap is "
        f"{gaps[0] / PS:.1f}us -- if that\n  is a step edge the capture is "
        'truncated and this "1 step" is really several; see `Trace.n_steps`'
    )
  print()
  hdr = (
      f'{"":>6s} {"modules":>8s} {"programs":>9s} {"busy":>10s} '
      f'{"own span":>10s} {"occ/own":>8s} {"occ/step":>9s} {"idle":>10s}'
  )
  print(hdr)
  print("-" * len(hdr))
  for si, (lo, hi) in enumerate(steps):
    if len(steps) > 1:
      print(f"step {si}  [{(hi - lo) / PS_MS:.2f} ms wall]")
    for d in sorted(tr.devices):
      mods = [m for m in tr.modules(d) if m.start >= lo and m.end <= hi]
      if not mods:
        continue
      busy = sum(m.dur for m in mods)
      span = max(m.end for m in mods) - min(m.start for m in mods)
      prog = len({m.program_id for m in mods})
      print(
          f"{tr.label(d)}  {len(mods):8d} {prog:9d} {busy / PS_MS:9.2f}ms "
          f"{span / PS_MS:9.2f}ms {100 * busy / span:7.1f}% "
          f"{100 * busy / (hi - lo):8.1f}% {(hi - lo - busy) / PS_MS:9.2f}ms"
      )
    drv = tr.driver_busy(lo, hi)
    if drv:
      print(
          f"  driver thread issuing work: {drv / PS_MS:.2f} ms "
          f"= {100 * drv / (hi - lo):.1f}% of the step; the rest it is "
          "waiting, mostly in\n  `block_until_ready` -- only mostly, because "
          "this line comes off the host plane\n  and the step window off the "
          "device planes, and the two clocks can sit several\n  ms apart (see "
          "`driver_busy`). One Python thread issues every execution and\n  "
          "every transfer, so dispatch binds as this approaches 100%."
      )
    print()
  print(
      "occ/own divides by the device's own first-to-last module and so hides"
      "\nthe drain -- the last stage finishes early and simply stops being "
      "measured.\nocc/step divides by the whole step and is the number a "
      "pipeline is judged on."
  )
  if tr.is_mpmd():
    print("The two agree on stage 0 and diverge on the last stage; that spread " "IS the bubble.")
  print()
  if steps:
    lo = min(m.start for d in tr.devices for m in tr.modules(d))
    hi = max(m.end for d in tr.devices for m in tr.modules(d))
    com, own = [], []
    for d in sorted(tr.devices):
      mods = tr.modules(d)
      if not mods:  # a plane the capture caught idle
        continue
      busy = sum(m.dur for m in mods)
      com.append(busy / (hi - lo))
      own.append(busy / (max(m.end for m in mods) - min(m.start for m in mods)))
    n = len(com)
    print(
        f"whole capture ({(hi - lo) / PS_MS:.2f} ms, {len(steps)} steps), "
        "mean over devices -- the busy share on the common clock and on each device's own window:"
    )
    print(f"  common clock       {100 * sum(com) / n:5.1f}% busy, " f"{100 - 100 * sum(com) / n:4.1f}% idle")
    print(f"  per-device window  {100 * sum(own) / n:5.1f}% busy, " f"{100 - 100 * sum(own) / n:4.1f}% idle\n")
    ops = {d: tr.ops(d) for d in sorted(tr.devices) if tr.ops(d)}
    if ops:
      olo = min(e.start for v in ops.values() for e in v)
      ohi = max(e.end for v in ops.values() for e in v)
      print(
          f"per device, from the union of the `XLA Ops` intervals over a "
          f"{(ohi - olo) / PS_MS:.2f} ms clock.\nThat is the measure to "
          "compare against an SPMD capture -- see `merged`."
          + ("\nNote again how much occ/own flatters the stage with the " "longest drain:" if tr.is_mpmd() else "")
      )
      for d, v in ops.items():
        busy = merged(v)
        span = max(e.end for e in v) - min(e.start for e in v)
        print(
            f"  {tr.label(d)}  {busy / PS_MS:8.2f}ms busy  "
            f"{100 * busy / (ohi - olo):6.1f}%  "
            f"(occ/own {100 * busy / span:.1f}% of its own {span / PS_MS:.2f} ms)"
        )
      print()
  if tr.platform == "GPU":
    _gpu_summary(tr)
  print(
      "programs per device, by name and (program id). A pipeline program's name carries its chunk, _c<chunk>,\n"
      "but programs with identical HLO all show under one name, whichever loaded first: for those the device\n"
      "and the order within the step tell the chunks apart."
  )
  for d in sorted(tr.devices):
    c = collections.Counter(m.name for m in tr.modules(d))
    print(f"  {tr.label(d)}")
    for nm, n in sorted(c.items(), key=lambda kv: -kv[1]):
      print(f"      {n:4d}x  {nm}")


def _gpu_summary(tr):
  """`summary`'s GPU lines: each device's kernel time against its programs' spans, and what ran outside them --
  the stage boundary apart from the rest."""
  print(
      "inside the programs, per device: their kernels' union against the programs' own spans.  A GPU\n"
      "module spans the gaps between its kernels: a host launching kernels slower than they run, or a\n"
      "stream waiting on another, is the difference"
  )
  for d in sorted(tr.devices):
    mods = tr.modules(d)
    if mods:
      span = sum(m.dur for m in mods)
      kern = sum(merged(m.stats["kernels"]) for m in mods)
      coll = sum(merged([k for k in m.stats["kernels"] if k.stats.get("kind") == "collective"]) for m in mods)
      print(
          f"  {tr.label(d)}  {kern / PS_MS:8.2f}ms of kernels in {span / PS_MS:.2f}ms of programs "
          f"({100 * kern / span:.1f}%); collective kernels {coll / PS_MS:.2f}ms of it"
      )
    # The stage boundary is the copies between GPUs and the send/recv: the copies to and from host memory
    # (--offload-residuals') run outside every program too.
    parts = ([], [])
    for x in tr.devices[d].get("Transfers", []):
      parts[x.stats.get("memcpy") != "P2P" and x.stats["kind"] != "collective"].append(x)
    for xs, what in zip(
        parts,
        (
            "PJRT's copies between GPUs and cross-process send/recv: the stage boundary",
            "the rest, not the boundary: PJRT's copies to and from host memory, " "--offload-residuals' among them",
        ),
    ):
      if not xs:
        continue
      kinds = collections.Counter(
          f'Memcpy{x.stats["memcpy"]}' if x.stats.get("memcpy") else x.stats["kernel"].split("(")[0] for x in xs
      )
      nb = sum(x.stats.get("bytes") or 0 for x in xs)
      print(
          f'      outside any program: {", ".join(f"{n} {k}" for k, n in kinds.most_common())}'
          f" -- {merged(xs) / PS_MS:.2f}ms" + (f", {nb / 1e6:.1f} MB copied" if nb else "") + f"\n      ({what})"
      )
  print()


CHARS = [
    ("fwd", "F"),
    ("_fba", "A"),  # (and a split last chunk's AI, fbai / fbi: the same program less its layers' dW)
    ("_fb", "A"),
    ("_bwdi", "I"),  # the split backward's halves, before '_bwd', a prefix of both
    ("_bwdw", "W"),
    ("_bwda", "B"),
    ("_bwd", "b"),
    ("loss", "L"),
    ("multi_slice", "s"),
    ("gather", "G"),
    ("reduce", "R"),
    ("update", "U"),
    ("train_step", "T"),
]


def _char(name):
  for pat, ch in CHARS:
    if pat in name:
      return ch
  return "o"


def cmd_timeline(tr, args):
  steps = tr.steps()
  if not steps:
    raise SystemExit(f"no XLA Modules events in {tr.path}")
  if args.width < 1:
    raise SystemExit(f"--width must be at least 1, got {args.width}")
  i = args.step if args.step >= 0 else args.step + len(steps)
  if not 0 <= i < len(steps):
    raise SystemExit(
        f"--step {args.step} out of range: this capture has "
        f"{len(steps)} step(s), so 0..{len(steps) - 1} "
        "(or -1 for the last)"
    )
  lo, hi = steps[i]
  width = args.width
  scale = (hi - lo) / width
  print(f"{tr.path}")
  print(
      f"step {i} of {len(steps)}: {(hi - lo) / PS_MS:.2f} ms over {width} " f"columns, {scale / PS:.1f} us per column\n"
  )
  for d in sorted(tr.devices):
    row = [" "] * width
    for m in tr.modules(d):
      if m.end <= lo or m.start >= hi:
        continue
      a = max(0, int((m.start - lo) / scale))
      b = min(width - 1, int((m.end - lo) / scale))
      for i in range(a, b + 1):
        row[i] = _char(m.name)
    busy = sum(m.dur for m in tr.modules(d) if m.start >= lo and m.end <= hi)
    print(f'{tr.label(d)} |{"".join(row)}| {100 * busy / (hi - lo):5.1f}% busy')
  print("\n  F = forward   b = backward (first microbatch, no accumulate)")
  print("  B = backward + accumulate    A = fused fwd+bwd (last chunk)")
  print("  I / W = a split backward's input / weight gradients (zbh1, zbh2)")
  print("  L = MaxText's single fused step   s = multi_slice   space = IDLE")
  print(
      "\n  A blank run on an early device while a later device is busy is the"
      "\n  pipeline bubble. A blank column across ALL devices is a step edge"
      "\n  or a host stall -- check the driver line in `summary`."
  )


def cmd_modules(tr, args):
  devs = tr.device_list(args.device)
  print(tr.path)
  for d in devs:
    by = collections.defaultdict(list)
    for m in tr.modules(d):
      by[m.name].append(m.dur / PS)
    print(f"\n{tr.label(d)}")
    print(f'  {"module":52s} {"n":>4s} {"total":>9s} {"mean":>9s} ' f'{"median":>9s} {"p10":>9s}')
    for nm, durs in sorted(by.items(), key=lambda kv: -sum(kv[1])):
      n, mean, med, p10 = uncontended(durs)
      print(f"  {nm[:52]:52s} {n:4d} {sum(durs) / 1e3:8.2f}ms " f"{mean:8.1f}us {med:8.1f}us {p10:8.1f}us")
    print(
        "  mean > median means some instances absorbed a stall (a receive "
        "wait\n  lands INSIDE the module, not in the gap before it). Compare "
        "medians."
    )


def cmd_gaps(tr, args):
  """Where each device's idle goes, measured against the whole step.

  Gaps *between* a device's modules are only part of its idle: the last stage
  also sits out the fill at the head of the step and the drain at the tail,
  and that is most of what it does.  Counting only the in-between gaps -- the
  bug this used to have -- divides by the device's own first-to-last module
  and so reports the stage with the worst bubble as the one with the least
  idle.  Lead-in and drain are therefore buckets like any other, and the total
  reconciles exactly with `summary`'s occ/step idle column.
  """
  steps = tr.steps()
  devs = tr.device_list(args.device)
  print(f"{tr.path}\n")
  print(
      f"{len(steps)} step(s), {sum(hi - lo for lo, hi in steps) / PS_MS:.2f} "
      "ms of wall between the first and last module of each."
  )
  print(f'{"":>6s} {"total idle":>11s}   classified by the module pair it ' "separates")
  fill, drain = (
      ("step lead-in (waiting for the fill)", "step drain (waiting for stage 0)")
      if tr.is_mpmd()
      else ("step lead-in (before its first module)", "step drain (after its last module)")
  )
  for d in devs:
    buckets = collections.Counter()
    tot = 0
    for lo, hi in steps:
      mods = [m for m in tr.modules(d) if m.start >= lo and m.end <= hi]
      if not mods:
        continue
      for key, g in (
          [(fill, mods[0].start - lo)]
          + [(f"{_kind(a.name)} -> {_kind(b.name)}", b.start - a.end) for a, b in zip(mods, mods[1:])]
          + [(drain, hi - mods[-1].end)]
      ):
        if g > 0:
          buckets[key] += g
          tot += g
    print(f"\n{tr.label(d)}  {tot / PS_MS:10.2f}ms")
    for k, v in buckets.most_common():
      print(f"         {v / PS_MS:10.2f}ms  {100 * v / tot:5.1f}%  {k}")
  if len(steps) > 1:
    off = sum(b[0] - a[1] for a, b in zip(steps, steps[1:]))
    print(
        f"\n  Excluded: {off / PS_MS:.2f} ms between steps, when every device "
        "is idle at once.\n  That is the driver's end-of-step "
        "`block_until_ready`, not pipeline idle."
    )
  if tr.is_mpmd():
    print(
        "\n  A stage that waits mid-step (bwd -> bwda) is waiting on the "
        "gradient coming\n  back down the pipe; a stage that waits in lead-in "
        "and drain is waiting for\n  stage 0 at the step edges. The bubble "
        "rotates around the pipeline, so the\n  SHAPE of this table is the "
        "finding -- the totals are nearly flat.\n  Idle that is flat across "
        "devices AND tracks the driver line is host dispatch."
    )
  else:
    print(
        "\n  Every device here runs the same program, so there is no bubble "
        "to read a\n  shape out of: the stages are replicas and their idle is "
        "collective wait.\n  Compare the totals with an MPMD capture's, not "
        "the rows with each other."
    )


def _kind(name):
  """A module name's program kind for the summary's transitions: the first of these it holds (each before the ones
  it extends: '_bwdwa' before '_bwdw' before '_bwd', '_fbai' and '_fbi' before '_fba' and '_fb'), else its name."""
  for pat in (
      "fwd",
      "_fbai",
      "_fbi",
      "_fba",
      "_fb",
      "_bwdi",
      "_bwdwa",
      "_bwdw",
      "_bwda",
      "_bwd",
      "loss",
      "multi_slice",
      "gather",
      "reduce",
      "update",
  ):
    if pat in name:
      return pat.lstrip("_")
  return name.split("(")[0]


def cmd_idle(tr, args):
  """Time when NO device is working: the idle a bubble does not explain.

  `gaps` answers "when is this stage idle", and the answer is dominated by the
  bubble, which is the schedule working as designed.  This answers "when is
  the machine idle", which is a different and much smaller number, and every
  millisecond of it is recoverable in principle.  Three things produce it and
  they want opposite responses, so they are reported apart:

    profiler start-up and teardown -- not your program at all, and 40-84% of
      the file (mean 59% over 19 v6e-4 captures);
    the gap between steps -- one host round trip, and flat in absolute terms
      across a 16x microbatch sweep, which is how you know it is host and not
      data.  Issuing both steps before waiting on either took this from 406
      us per boundary to 7.5 us; a capture that still shows hundreds of
      microseconds here was taken by a loop that blocked;
    holes inside a step -- a handoff nothing else overlaps, in the drain,
      where the last microbatch's gradient walks back one stage at a time with
      nothing left to hide behind.  One such hop per step is the floor.

  What that leaves, at S=4 V=2 M=8 E4096 mbs256: `search`, the schedule the
  code actually picks, idles every chip for 0.05% of the step.  MaxText SPMD
  on the same shape is 0.01%.  `1f1b` is 1.90% and `gpipe` 0.09%, but both are
  9-11 ms/step slower overall, so the idle is not what is wrong with them.
  """
  up, wlo, whi, down, frames = tr.framing()
  steps = tr.steps()
  print(f"{tr.path}\n")
  if frames and wlo is not None:
    flo = min(min(f.start for f in frames), wlo)
    fhi = max(max(f.end for f in frames), whi)
    print(
        f"capture framing -- {(fhi - flo) / PS_MS:.2f} ms of file, of which "
        f"{(whi - wlo) / PS_MS:.2f} ms "
        f"({100 * (whi - wlo) / (fhi - flo):.1f}%) has any device running:"
    )
    if tr.platform == "GPU" and not up + down:
      print(
          f"  ({len(frames)} outermost frames on the driver thread, none of them the profiler's: a GPU "
          "capture runs\n  without the Python tracer (profiling.profile_options), so there is no start-up "
          "or teardown\n  to split out, and no frame to list)"
      )
    else:
      for f in frames:
        tag = (
            "profiler start-up"
            if "__enter__" in f.name
            else "profiler teardown, serializing the capture"
            if "__exit__" in f.name
            else "issuing a step"
            if "lambda" in f.name
            else "waiting"
        )
        print(f"  {(f.start - wlo) / PS_MS:+9.2f} ms  for " f"{f.dur / PS_MS:8.2f} ms   {tag}")
      if up + down:
        print(
            f"  {(up + down) / PS_MS:.2f} ms of that is profiler overhead with "
            "the chips idle. It is not\n  pipeline idle and it is not in any "
            "timed region -- but it is most of what a\n  trace viewer draws, "
            "so do not read the file's own occupancy as the program's."
        )
    print()

  in_step, n_holes = 0, 0
  for si, (lo, hi) in enumerate(steps):
    holes = [h for h in tr.all_idle(lo, hi)]
    tot = sum(b - a for a, b, _, _ in holes)
    in_step += tot
    n_holes += len(holes)
    print(
        f"step {si}  [{(hi - lo) / PS_MS:.2f} ms]  nobody working for "
        f"{tot / PS_MS:.3f} ms ({100 * tot / (hi - lo):.2f}%) "
        f"in {len(holes)} hole(s)"
    )
    for a, b, before, after in sorted(holes, key=lambda h: h[0] - h[1]):
      if b - a < args.min_us * PS:
        continue
      drv = tr.driver_busy(a, b)
      src = f"{tr.label(before[0])} {_kind(before[1].name)}" if before else "lead-in"
      dst = ", ".join(sorted({f"{tr.label(d)} {_kind(m.name)}" for d, m in after})) or "end of step"
      print(
          f"    {(b - a) / PS:8.1f} us  at {100 * (a - lo) / (hi - lo):5.1f}%"
          f" of the step   {src:18s} -> {dst:18s}"
          f"  driver issuing {100 * drv / (b - a):4.1f}%"
      )
    if tr.platform == "GPU":
      # Measured on modules, a hole misses the gaps INSIDE a GPU module (GPU CAPTURES): the kernel-level figure
      # adds them, and every op lies inside its module's span, so the difference is exactly those gaps.
      kern = _span_union([(o.start, o.end) for d in tr.devices for o in tr.ops(d)], lo, hi)
      mods = _span_union([(m.start, m.end) for d in tr.devices for m in tr.modules(d)], lo, hi)
      print(
          f"    kernel level: no kernel on any GPU for {(hi - lo - kern) / PS_MS:.3f} ms "
          f"({100 * (hi - lo - kern) / (hi - lo):.2f}%), {(mods - kern) / PS_MS:.3f} ms of it inside a "
          "program --\n    the gaps between its kernels, where the host launching them slowly shows"
      )
  if steps:
    print(
        f"  (holes under {args.min_us:g} us are counted above but not "
        "listed; they are the\n  1.1 us between two back-to-back modules on "
        "one device, not a stall.)\n"
    )

  between = sum(b[0] - a[1] for a, b in zip(steps, steps[1:]))
  if len(steps) > 1:
    print(
        f"between steps: {between / PS_MS:.3f} ms over "
        f"{len(steps) - 1} boundary(ies), "
        f"{between / (len(steps) - 1) / PS:.1f} us each."
    )
    for a, b in zip(steps, steps[1:]):
      drv = tr.driver_busy(a[1], b[0])
      print(
          f"    driver issuing {100 * drv / (b[0] - a[1]):.1f}% of it; the "
          "rest is the round trip from\n    the last module completing to "
          "the next step being enqueued."
      )
  span = sum(hi - lo for lo, hi in steps) + between
  if span:
    print(
        f"\nall-chip idle over the {span / PS_MS:.2f} ms the steps occupy: "
        f"{(in_step + between) / PS_MS:.3f} ms = "
        f"{100 * (in_step + between) / span:.2f}%"
        f"\n  {in_step / PS_MS:.3f} ms inside steps, "
        f"{between / PS_MS:.3f} ms between them."
    )
  print(
      "\n  Compare with `gaps`, which will report far more idle per device. "
      "Both are\n  right: that is the bubble, one stage waiting while "
      "another works, and it is\n  what the schedule buys throughput with. "
      "This number is the part nobody is\n  using, and it is the only idle "
      "worth calling waste."
  )


def cmd_ops(tr, args):
  """Decompose a module into its top-level ops, pooled over devices.

  With no `-d`, every device's instances of the program go into one pool: a
  stage's devices run the same program, and stages the trace lists under one
  name run identical ones.  Pooling is what makes the mean stable enough to
  compare across shapes; programs whose names differ stay apart (`_select`).
  Pass `-d` to isolate one stage.
  """
  devs = tr.device_list(args.device)
  sel = _select(tr, devs, args.module)
  if not sel:
    raise SystemExit(f"no module matching {args.module!r} on device(s) {devs}")
  print(tr.path)
  n = len(sel)
  tot = sum(m.dur for _, m in sel)
  agg, cnt = collections.Counter(), collections.Counter()
  inner, icnt = collections.Counter(), collections.Counter()
  covered = 0
  gpu = tr.platform == "GPU"
  for d, m in sel:
    if gpu:
      # Nothing nests on a GPU, and which kernels are this execution's is known (`Trace._read_gpu`); the NCCL
      # stream overlaps the compute stream, so coverage is their union.
      kids = [k for k in m.stats["kernels"] if k.dur > 0]
      covered += merged(kids)
      for k in kids:
        key = _gpu_gmm(k) or k.base
        agg[key] += k.dur
        cnt[key] += 1
      continue
    ops = tr.ops(d)
    kids = children(ops, m.start, m.end)
    covered += sum(k.dur for k in kids)
    for k in kids:
      agg[k.base] += k.dur
      cnt[k.base] += 1
      if k.base == "while":
        # The body, not the `while` over again.  `children` keeps anything the
        # window contains and the window *is* `k`, so `k` matches itself first
        # and shadows everything inside it.  Drop whatever exactly spans the
        # window -- by span rather than by identity, so that a duplicate event
        # at the same bounds cannot shadow the body either.
        body = [o for o in ops if o.start != k.start or o.end != k.end]
        for c in children(body, k.start, k.end):
          inner[c.base] += c.dur
          icnt[c.base] += 1
  who = tr.label(devs[0]) if len(devs) == 1 else f"{len(devs)} devices pooled"
  print(f"\n{who}  {args.module}  n={n}  mean module {tot / n / PS:.1f} us")
  if gpu:
    print(
        f"  kernels cover {100 * covered / tot:.2f}% of the module (their union: nothing nests on a GPU, but "
        "the\n  NCCL stream runs beside the compute stream, so the shares below can sum past it)"
    )
  else:
    print(
        f"  top-level ops cover {100 * covered / tot:.2f}% of the module "
        "(a containment stack, so a `while` is charged once)"
    )
  print(f'  {"op":40s} {"n/mod":>7s} {"us/mod":>9s} {"share":>7s}')
  for k, v in agg.most_common():
    if v / n / PS < 0.05:
      continue
    print(f"  {k[:40]:40s} {cnt[k] / n:7.1f} {v / n / PS:9.1f} " f"{100 * v / tot:6.1f}%")
  if inner:
    print("  -- inside the `while`:")
    for k, v in inner.most_common():
      if v / n / PS < 0.05:
        continue
      print(f"     {k[:37]:37s} {icnt[k] / n:7.1f} {v / n / PS:9.1f} " f"{100 * v / tot:6.1f}%")
  if gpu:
    print(
        "  An op is its hlo_op, ordinal off: custom-call a cuBLAS or cuDNN call, gmm_* a grouped-matmul\n"
        "  kernel (named off its scope, `_gpu_gmm`), the collectives NCCL kernels on their own stream.\n"
        "  The module time the kernels leave uncovered is the gaps between them (`summary`)."
    )
    return
  print(
      "  Rows are the top-level ops by XLA name: the pipeline's grouped-matmul kernels by their own (gmm_wi_0,"
      "\n  gmm_wo_dx, gmm_wo_dw, ...), its FSDP and EP collectives anonymous (collective-permute-*, all-gather-*,"
      "\n  reduce-scatter-*, fusion) -- `scopes` finds those by their named scope. A `while` is charged once,"
      "\n  its body listed under it."
  )


def cmd_roofline(tr, args):
  """Score a module against the chip peaks the trace itself reports.

  These captures carry no per-op cost model: no `flops` or `bytes_accessed`
  stat is emitted on any line, so there is nothing to read an arithmetic
  intensity off.  The counts have to come from a byte model instead -- pass
  `--flops` and `--bytes` for ONE module instance and this
  divides them by the measured time.  Given neither, it prints the peaks and
  stops rather than inventing an intensity.

  A GPU plane's one peak is its `memory_bandwidth`: there the bandwidth side is
  scored and the compute side only measured -- no ridge, no verdict.
  """
  devs = tr.device_list(args.device)
  print(tr.path)
  pk_f, pk_b = tr.peaks[devs[0]]
  bw_only = tr.platform == "GPU" and pk_f is None and pk_b is not None
  if (pk_f is None or pk_b is None) and not bw_only:
    raise SystemExit("this plane carries no peak_teraflops_per_second / " "peak_hbm_bw_gigabytes_per_second stat")
  sel = [m for _, m in _select(tr, devs, args.module)]
  if not sel:
    raise SystemExit(f"no module matching {args.module!r} on device(s) {devs}")
  n = len(sel)
  mean = sum(m.dur for m in sel) / n / 1e12  # picoseconds -> seconds
  who = tr.label(devs[0]) if len(devs) == 1 else f"{len(devs)} devices pooled"
  print(f"\n{who}  {args.module}  n={n}")
  if bw_only:
    print(
        f"  peak from the trace: {pk_b / 1e9:.2f} GB/s (memory_bandwidth); a GPU plane carries no FLOP/s "
        "peak,\n  so there is no ridge and no compute-bound verdict: the bandwidth side alone"
    )
    print(f"  measured: {mean * 1e6:.1f} us per instance (first kernel to last, gaps included)")
    if args.flops:
      print(f"  compute: {args.flops / mean / 1e12:8.1f} TF/s  (no peak in the trace to score it against)")
    if args.bytes:
      print(f"  memory:  {args.bytes / mean / 1e9:8.1f} GB/s  = {100 * args.bytes / mean / pk_b:.1f}% of peak")
      print(f"  bandwidth floor {args.bytes / pk_b * 1e6:.1f} us; measured is {mean * pk_b / args.bytes:.2f}x that")
    if not args.flops and not args.bytes:
      print("\n  No --flops/--bytes given; supply the analytic counts for one module instance.")
    return
  print(f"  peaks from the trace: {pk_f / 1e12:.1f} TF/s, " f"{pk_b / 1e9:.2f} GB/s, ridge {pk_f / pk_b:.0f} FLOP/byte")
  print(f"  measured: {mean * 1e6:.1f} us per instance")
  if not args.flops and not args.bytes:
    print(
        "\n  No --flops/--bytes given, and this trace carries no per-op cost"
        "\n  model, so there is no intensity to report. Supply the analytic"
        "\n  counts for one module instance: the FLOPs of the layers it runs"
        "\n  at this microbatch (a remat backward recomputes their forward"
        "\n  too) and the HBM bytes it moves -- weights, gradient accumulator,"
        "\n  activations."
    )
    return
  f, b = float(args.flops or 0), float(args.bytes or 0)
  if f:
    print(f"  compute: {f / mean / 1e12:8.1f} TF/s  " f"= {100 * f / mean / pk_f:.1f}% of peak")
  if b:
    print(f"  memory:  {b / mean / 1e9:8.1f} GB/s  " f"= {100 * b / mean / pk_b:.1f}% of peak")
  if f and b:
    ai = f / b
    floor = max(f / pk_f, b / pk_b)
    side = "compute" if ai > pk_f / pk_b else "BANDWIDTH"
    print(f"  intensity: {ai:.1f} FLOP/byte against a {pk_f / pk_b:.0f} ridge " f"-> {side}-bound")
    print(
        f"  roofline floor {floor * 1e6:.1f} us; measured is "
        f"{mean / floor:.2f}x that ({100 * floor / mean:.1f}% of attainable)"
    )
    if ai <= pk_f / pk_b:
      print(
          "\n  AI below the ridge => bandwidth-bound. Weight and accumulator"
          "\n  bytes do not grow with the microbatch while FLOPs do, so more"
          "\n  tokens per microbatch raise AI and wider layers do not."
      )


# `ds_ep.py` runs its FSDP, EP and grad-norm collectives under these
# `jax.named_scope`s, and names each grouped-matmul kernel.
SCOPES = ("fsdp_all_gather", "fsdp_reduce_scatter", "ep_all_gather", "ep_reduce_scatter", "grad_norm_all_reduce")
CONTAINERS = ("while", "conditional", "call")


def leaves(ops, start, end, starts=None):
  """Top-level ops of [start, end), looking through `while`/`call` bodies.

  `ops` sorted as `Trace._events` leaves them; pass `starts`, the list of
  their start times, when calling once per module over one device's ops.
  """
  if starts is None:
    starts = [o.start for o in ops]
  window = ops[bisect.bisect_left(starts, start) : bisect.bisect_left(starts, end)]
  out = []
  for k in children(window, start, end):
    if k.base in CONTAINERS:
      # the body, not the container again (see `cmd_ops`)
      body = [o for o in window if k.start <= o.start < k.end and (o.start != k.start or o.end != k.end)]
      out.extend(leaves(body, k.start, k.end))
    else:
      out.append(k)
  return out


def _gpu_gmm(o):
  """A GPU kernel's grouped-matmul name -- `gmm_wi_0`, `gmm_wo_dx` -- off its name stack, or None.

  ds_ep scopes each grouped matmul `<impl>/<name>` (`ds_ep._kernel_scope`).  On
  a TPU the kernel then takes the name; on a GPU it is `triton_kernel_call`
  (tokamax's Triton kernels), at `.../tokamax/gmm_wi_0/triton_kernel_call`, so
  the name is the scope right above a kernel call's primitive (`*_call`:
  triton_kernel_call, pallas_call, mosaic_gpu_call).  An XLA fusion inside the
  scope -- a cast, a pad -- is not the kernel.
  """
  parts = (o.stats.get("tf_op") or "").split("/")
  if len(parts) >= 2 and parts[-1].endswith("_call") and parts[-2].startswith(("gmm_", "tgmm_")):
    return parts[-2]
  # tokamax's Mosaic GPU kernels (gmm_impl 'tokamax', the forward and dlhs products by default) sit deeper:
  # `.../tokamax/gmm_wi_1_dx/eval_jaxpr/ragged_dot_sm100/mpmd_map` -- the nearest enclosing gmm_ scope names them.
  if parts and (parts[-1] in _GPU_KERNEL_PRIMS or parts[-1].endswith("_call")):
    for q in reversed(parts[:-1]):
      if q.startswith(("gmm_", "tgmm_")):
        return q
  return None


# The primitive names a GPU kernel's name stack ends in when it is a hand-written kernel, not an XLA fusion:
# Mosaic GPU's (tokamax's sm90 / sm100 grouped matmuls) next to the `*_call` ones.
_GPU_KERNEL_PRIMS = ("mpmd_map",)


def gpu_scope(ev):
  """`scope_of` for a GPU kernel, whose name stack lists every op XLA combined into it -- `prefix/(a:b:c)` --
  where a combiner merged collectives: the parts' scopes, in `SCOPES` order, joined with `+` where they differ.

  `ep_all_gather+fsdp_all_gather` is a microbatch's EP gather riding one NCCL
  call with the FSDP gathers; `scope_of` alone would hand the whole kernel to
  whichever of `SCOPES` comes first.
  """
  t = ev.stats.get("tf_op") or ""
  i = t.find("/(")
  if i < 0 or not t.endswith(")"):
    return scope_of(ev)
  got = []
  for part in t[i + 2 : -1].split(":"):
    sc = scope_of(Event(0, 0, "", {"tf_op": t[: i + 1] + part}))
    if sc and sc not in got:
      got.append(sc)
  order = SCOPES + tuple(f"transpose({sc})" for sc in SCOPES)
  return "+".join(sorted(got, key=lambda sc: order.index(sc) if sc in order else len(order))) or None


def scope_of(ev):
  """Which of `SCOPES` an op ran under, or None.

  In a backward the scope reads `transpose(jvp(ep_all_gather))`, and the
  transpose of a collective is its dual -- that one is a reduce-scatter -- so
  it comes back as `transpose(ep_all_gather)`, its own row.
  """
  t = ev.stats.get("tf_op") or ""
  for sc in SCOPES:
    i = t.find(sc)
    if i >= 0:
      wrap = t[t.rfind("/", 0, i) + 1 : i]  # the transforms around this scope
      return f"transpose({sc})" if "transpose(" in wrap else sc
  return None


def _async_part(ev):
  """An async collective's start / update / done (incl. XLA's async-call
  wrappers, `all-gather.2.cloned.1.call-done`), or None."""
  raw = ev.name.split(" = ")[0].lstrip("%")
  for part in raw.split("."):
    if part.endswith(("-start", "-update", "-done")):
      return raw.split(".call-")[0] if ".call-" in raw else raw
  return None


def cmd_scopes(tr, args):
  """Per program: the named-scope collectives it carries, and its gmm kernels.

  The FSDP gather and reduce-scatter have no program of their own -- they are
  folded into forwards, backwards and the optimizer program -- and on the
  `XLA Ops` row they are anonymous `collective-permute-*` / `all-gather-*` /
  fusions.  Their `tf_op` carries the scope, so this finds them the way the
  trace viewer's "Framework Name Scope" row draws them.

  `ops` is the number of top-level TensorCore ops under the scope, `TC ms` the
  TensorCore time they take, and `wait ms` the part of it in an async
  collective's start / update / done: what the TensorCore spends issuing and
  blocked on the collective (a permute's peer rendezvous is paid in its
  start).  The rest of `TC ms` is the assembly or the add.  Collectives
  offloaded to SparseCore move their bytes on the SparseCore planes; only
  their start / done are here.

  On a GPU `ops` counts every kernel under the scope and `GPU ms` is their
  time; `exposed ms` is the part of its collective (NCCL) kernels' time with
  no compute kernel of the program running beside them -- what the TPU's
  `wait` is, the collective the device does not hide.  A collective XLA
  combined from several scopes reads `a+b` (`gpu_scope`); one whose name stack
  XLA dropped -- the pipelined reduce-scatters, the DP reduce -- gets a row of
  its collective's own, `reduce-scatter (no scope)`.
  """
  steps = tr.steps()
  if not steps:
    raise SystemExit(f"no XLA Modules events in {tr.path}")
  k = len(steps)
  i = args.step if args.step >= 0 else args.step + k
  if not 0 <= i < k:
    raise SystemExit(f"--step {args.step} out of range: 0..{k - 1}")
  lo, hi = steps[i]
  print(tr.path)
  print(f"step {i} of {k}, {(hi - lo) / PS_MS:.2f} ms\n")
  rows = SCOPES + tuple(f"transpose({sc})" for sc in SCOPES)
  unit, wait = ("GPU", "exposed") if tr.platform == "GPU" else ("TC", "wait")
  for d in tr.device_list(args.device):
    ops = tr.ops(d)
    starts = [o.start for o in ops]
    mods = tr.modules(d)
    blk = len(mods) // k
    progs = collections.OrderedDict()
    # Step i is the i-th block of the device's module sequence, as in
    # `Trace.steps` -- not whatever overlaps the union window, which would
    # also pick up a neighbouring step's first or last program.
    if tr.platform == "GPU":
      _gpu_scopes(mods[i * blk : (i + 1) * blk], progs)
    else:
      for m in mods[i * blk : (i + 1) * blk]:
        p = progs.setdefault(
            m.base,
            {
                "n": 0,
                "sc": collections.Counter(),
                "tc": collections.Counter(),
                "wait": collections.Counter(),
                "gmm": collections.Counter(),
            },
        )
        p["n"] += 1
        pending = {}  # async-call prefix -> scope of its start
        for o in leaves(ops, m.start, m.end, starts):
          if o.base.startswith(("gmm_", "tgmm_")):
            p["gmm"][o.base] += 1
          part = _async_part(o)
          sc = scope_of(o)
          if part is not None:
            # an async `.call-update` carries no tf_op: it is its start's
            if sc is None:
              sc = pending.get(part)
            else:
              pending[part] = sc
          if sc:
            p["sc"][sc] += 1
            p["tc"][sc] += o.dur
            if part is not None:
              p["wait"][sc] += o.dur
    tot = collections.Counter()
    # a TPU's scopes are all `rows`; a GPU's add combined ones and collectives of no scope
    shown = rows + tuple(sorted({sc for p in progs.values() for sc in p["sc"]} - set(rows)))
    w = 30 if tr.platform != "GPU" else max([30] + [len(sc) for p in progs.values() for sc in p["sc"]])
    print(tr.label(d))
    for name, p in progs.items():
      n = p["n"]
      print(f"  {name}  x{n}")
      for sc in shown:
        if p["sc"][sc]:
          tot[sc] += p["tc"][sc]
          print(
              f'      {sc:{w}s} {p["sc"][sc] / n:6.1f} ops  '
              f'{p["tc"][sc] / n / PS_MS:7.2f} {unit} ms  '
              f'{p["wait"][sc] / n / PS_MS:6.2f} {wait} ms   per instance'
          )
      if p["gmm"]:
        g = "  ".join(f"{kn} {v / n:g}" for kn, v in sorted(p["gmm"].items()))
        print(f"      kernels per instance: {g}")
    print(
        f"  step total {unit} ms: "
        + (", ".join(f"{sc} {tot[sc] / PS_MS:.2f}" for sc in shown if tot[sc]) or "none")
        + "\n"
    )


def _gpu_scopes(mods, progs):
  """`cmd_scopes`' per-program counters off GPU executions `mods`, into `progs`: every kernel of each (nothing
  nests), its scope `gpu_scope`'s or, for a collective with none, `<collective> (no scope)`; `wait` the collective
  kernels' time with no compute kernel of the execution beside it; grouped matmuls named by `_gpu_gmm`."""
  for m in mods:
    p = progs.setdefault(
        m.base,
        {
            "n": 0,
            "sc": collections.Counter(),
            "tc": collections.Counter(),
            "wait": collections.Counter(),
            "gmm": collections.Counter(),
        },
    )
    p["n"] += 1
    ks = m.stats.get("kernels", [])
    comp = _merge_spans((o.start, o.end) for o in ks if o.stats.get("kind") == "compute")
    for o in ks:
      g = _gpu_gmm(o)
      if g:
        p["gmm"][g] += 1
      coll = o.stats.get("collective")
      sc = gpu_scope(o) or (f"{coll} (no scope)" if coll else None)
      if sc:
        p["sc"][sc] += 1
        p["tc"][sc] += o.dur
        if coll:
          p["wait"][sc] += o.dur - _overlap(comp, o.start, o.end)


# `ds_ep` program names: jit_{fwd,bwd,bwda,fb,fba}_c<chunk>, the split
# backward's halves under the zero-bubble schedules (jit_bwdi_c<chunk>, the
# input gradient I; jit_bwdw_c<chunk> / jit_bwdwa_c<chunk>, the weight
# gradients W, the chunk's first and accumulating -- the last chunk's too where
# it splits, its fused program then jit_fbi_c<last> / jit_fbai_c<last>, AI),
# and the host variants that also carry FSDP gathers (_ag<chunk>..., _agself)
# and / or reduce-scatters (_rs<chunk>..., _rsself) -- under --fsdp-scope
# repeat in a backward or the fused last chunk too (bwd_c3_rs5p5_ag1p4,
# fba_c3_ag1p5).  Stash variants (_st) and F&B pairs (__) are left out.
_PROG = re.compile(r"^jit_(fwd|bwdwa|bwdw|bwdi|bwda|bwd|fbai|fbi|fba|fb)_c(\d+)((?:_(?:ag|rs)(?:self|\d+(?:p\d+)?))*)$")
_KIND = {
    "fwd": "F",
    "bwd": "B",
    "bwda": "B",
    "bwdi": "I",
    "bwdw": "W",
    "bwdwa": "W",
    "fb": "A",
    "fba": "A",
    "fbi": "AI",
    "fbai": "AI",
}


def program_costs(tr, devs):
  """Per-chunk program medians (ms) as a `--costs` table, pooled over `devs`
  (every device of a stage runs the same programs).

    F<c>   median plain `fwd_c<c>`
    B<c>   median plain `bwda_c<c>` -- the accumulating backward, which is
           every microbatch's but the first
    I<c>   under a split backward (zbh1 / zbh2), median `bwdi_c<c>`, its
           input gradient
    W<c>   ... median plain `bwdwa_c<c>`, its accumulating weight gradients;
           and B<c>, which such a chunk never runs, their sum
    A      median `fba_c<last>`, the fused last chunk
    AI     where the last chunk splits too (--zb-split-last), median
           `fbai_c<last>`, its forward, head, loss and input gradient
    W<last>, WA  ... median `bwdwa_c<last>`, its layers' weight gradients;
           and A, which it never runs, AI + WA
    F, B   the median over chunks >= 1: what a chunk without its own key gets
    I, W   the median over the split middle chunks
    dF_ag  median extra ms of a forward that also carries a gather
    dB_rs  median extra ms of a backward that also carries a reduce-scatter
    dB_ag, dB_ag+rs, dA_ag   the same for what a backward or the fused last
           chunk carries under --fsdp-scope repeat, where measured
    update the slowest stage's median optimizer program

  A host variant is its plain program plus a delta, so it is folded in as
  one: its delta against the same chunk's plain median goes into the
  d<kind>_<what it carries> key, and a chunk that ran only host variants
  gets their median minus that delta -- or, where its kind has none (the
  fused last chunk when every one of its programs carries a gather), minus
  the median delta of the same collectives in the other kinds.  The
  schedulers read F/B/A (`schedule.cost_table`); the rest is for the
  reader, and is ignored there.
  """
  durs = collections.defaultdict(list)  # (prog, chunk, host) -> [ms]; host 'ag', 'rs', 'ag+rs' or None
  upd = collections.defaultdict(list)
  for d in devs:
    for m in tr.modules(d):
      mt = _PROG.match(m.base)
      if mt:
        host = "+".join(sorted(set(re.findall(r"_(ag|rs)", mt.group(3))))) or None
        durs[(mt.group(1), int(mt.group(2)), host)].append(m.dur / PS_MS)
      elif m.base.startswith("jit_update"):
        upd[m.base].append(m.dur / PS_MS)
  med = {k: statistics.median(v) for k, v in durs.items()}
  delta = collections.defaultdict(list)  # (kind, host) -> [ms over the same chunk's plain program]
  for (prog, c, host), v in durs.items():
    if host and (prog, c, None) in med:
      delta[(_KIND[prog], host)] += [x - med[(prog, c, None)] for x in v]
  dm = {k: statistics.median(v) for k, v in delta.items()}

  def host_delta(kind, host):
    if (kind, host) in dm:
      return dm[(kind, host)]
    pooled = [x for (_, h), v in delta.items() if h == host for x in v]
    return statistics.median(pooled) if pooled else 0.0

  def chunk_cost(c, progs):
    for prog in progs:
      if (prog, c, None) in med:
        return med[(prog, c, None)]
      v = [x - host_delta(_KIND[prog], h) for (p, c2, h), xs in durs.items() if p == prog and c2 == c and h for x in xs]
      if v:
        return statistics.median(v)
    return None

  chunks = sorted({c for _, c, _ in durs})
  last = max((c for p, c, _ in durs if _KIND[p] in ("A", "AI")), default=None)
  out = {}
  for key, progs in (("F", ("fwd",)), ("B", ("bwda", "bwd")), ("I", ("bwdi",)), ("W", ("bwdwa", "bwdw"))):
    for c in chunks:
      x = None if c == last else chunk_cost(c, progs)
      if x is not None:
        out[f"{key}{c}"] = x
  for c in chunks:  # a split chunk's backward, which it ran as I and W only: what sched.cost_table reads for its B
    if f"B{c}" not in out and f"I{c}" in out and f"W{c}" in out:
      out[f"B{c}"] = out[f"I{c}"] + out[f"W{c}"]
  if last is not None:
    out["A"] = chunk_cost(last, ("fba", "fb"))
    ai, wa = chunk_cost(last, ("fbai", "fbi")), chunk_cost(last, ("bwdwa", "bwdw"))
    if ai is not None:  # the split last chunk: AI and its W, which together run what A does
      out["AI"] = ai
    if wa is not None:
      out[f"W{last}"] = out["WA"] = wa
    if out["A"] is None and ai is not None and wa is not None:
      out["A"] = ai + wa
    if out["A"] is None:
      del out["A"]
  for key in ("F", "B", "I", "W"):
    rest = [v for k, v in out.items() if k[0] == key and k[1:].isdigit() and k != f"{key}0" and k != f"{key}{last}"]
    if rest:
      out[key] = statistics.median(rest)
  for (kind, host), v in sorted(dm.items()):
    out[f"d{kind}_{host}"] = v
  if upd:
    out["update"] = max(statistics.median(v) for v in upd.values())
  return {k: round(v, 2) for k, v in out.items()}


def cmd_costs(tr, args):
  """Per-chunk program medians as JSON on stdout, for `--costs @file.json`.

  A capture of some stages only -- one chip, profiling.CHIPS, is the default
  where one process holds them all -- has no fused last chunk (A), and maybe
  no chunk past the first (F, B): the table still prints, and stderr names
  what is missing, which --costs would take from default_costs."""
  table = program_costs(tr, tr.device_list(args.device))
  if not any(k[0] in "FBA" for k in table):
    raise SystemExit(f"no ds_ep programs (jit_fwd_c*, jit_bwda_c*, jit_fba_c*) in {tr.path}")
  missing = [k for k in ("A", "F", "B") if k not in table]
  if missing:
    print(
        f'# no {", ".join(missing)} cost in {tr.path}: its devices hold only some stages (A is the fused '
        "last chunk, F and B the chunks past the first); --costs would price them from default_costs.  "
        "Capture with --profile-chips covering every stage's chips (the default traces one).",
        file=sys.stderr,
    )
  print(json.dumps(table, indent=1))


def main(argv=None):
  p = argparse.ArgumentParser(
      prog="python3 -m maxtext.experimental.mpmd.jax.xprof",
      description=__doc__.split("WHY THIS FILE EXISTS")[0],
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  p.add_argument("cmd", choices=["summary", "timeline", "modules", "gaps", "idle", "ops", "roofline", "scopes", "costs"])
  p.add_argument("path", help="a .xplane.pb, or any directory above one")
  p.add_argument("-d", "--device", type=int, default=None)
  p.add_argument(
      "-m",
      "--module",
      default="jit_bwda_c0",
      help="program name as `modules` lists it, without the (id): exact if a program has it, else a "
      "substring of one program's name -- of several, the command exits listing them (default "
      "jit_bwda_c0, chunk 0's plain accumulating backward, one past the first microbatch's that "
      "carries no FSDP collective: not every plan has one)",
  )
  p.add_argument("--step", type=int, default=-1, help="which step, negative counts from the end (default -1)")
  p.add_argument("--width", type=int, default=110, help="timeline columns")
  p.add_argument("--min-us", type=float, default=20.0, help="idle: list holes at least this wide (default 20)")
  p.add_argument("--flops", type=float, default=0, help="roofline: analytic FLOPs in ONE module instance")
  p.add_argument("--bytes", type=float, default=0, help="roofline: analytic bytes moved by ONE module instance")
  a = p.parse_args(argv)
  tr = Trace(find_xplane(a.path))
  for note in tr.notes:
    print(f"# {note}", file=sys.stderr)
  {
      "summary": cmd_summary,
      "timeline": cmd_timeline,
      "modules": cmd_modules,
      "gaps": cmd_gaps,
      "idle": cmd_idle,
      "ops": cmd_ops,
      "roofline": cmd_roofline,
      "scopes": cmd_scopes,
      "costs": cmd_costs,
  }[a.cmd](tr, a)


if __name__ == "__main__":
  main()
