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

"""Ahead-of-time compile of the MPMD DeepSeek pipeline for a TPU slice this
host does not have, before submitting it: every program of every stage,
lowered and compiled against the slice's topology, and what it needs in HBM.

    bash src/maxtext/experimental/mpmd/jax/ds_ep_aot.sh --topology v6e-256 --stages 4 --fsdp 64 --ep 1 \
        --layout physical --layers 8 --experts 16 --mb 8 --gbs 512 [any ds_ep_pipeline flag]

ds_ep_aot.sh runs this module under the run's libtpu flags, which change what
compiles; `python3 -m maxtext.experimental.mpmd.jax.ds_ep_aot` alone compiles under whatever
LIBTPU_INIT_ARGS holds, with a warning when that is not the run's set.

The pipeline is built as `ds_ep_pipeline --distributed` builds it on the
slice -- the same flags, plan and stage meshes, here over the topology's
devices (`jax.experimental.topologies`) -- and `Pipeline.precompile` compiles
each stage's programs as that stage's processes would.  What the
multi-controller run refuses at startup is checked first, on the topology's
hosts (`multihost_problems`).  The report, per stage and per device: the
estimated peak -- `hbm_walk`'s, what the step holds when the stage's
heaviest program runs (the persistent state, the compute weights, the
gradient accumulators and reduced gradients, the live residuals and stash
sets, that program's temporaries and new outputs), plus the generated code
of every program the stage loads -- against the device's HBM, with the
totals of each kind for scale.  The exit status is non-zero if a compiled
stage's estimate does not leave `--headroom` of HBM free (HEADROOM, 10%:
what the walk leaves out).

It needs a TPU backend in this process (any TPU VM, e.g. a v6e-4): the
Pallas kernels pick interpret mode from `jax.default_backend()`, and a CPU
backend would compile those, not the TPU ones.  `--batch-split` on Lineage's
layers (ds_ep_pipeline's default impl for DeepSeek-V3; a Qwen3.5-family
model's batch split is ds_ep.split_layers) needs Lineage importable
(ds_ep_lineage.import_lineage; checked first): its programs import it when they
trace -- but for `--transfers-only`, which compiles none of them.

`--num-slices N` compiles for N slices of the topology (MaxText's
compile_topology_num_slices), e.g. one stage per slice on 2 x v5p-128 under
the v5p flags (the v5p set, and v5p_fsdp_extra's two flags added to it):

    F=src/maxtext/experimental/mpmd/jax/libtpu_flags/v5p_fsdp_extra.flags
    MPMD_FLAG_SET=v5p EXTRA_LIBTPU_INIT_ARGS="$(grep -v '^#' $F | tr '\n' ' ')" \
      bash src/maxtext/experimental/mpmd/jax/ds_ep_aot.sh --topology v5p-128 --num-slices 2 --stages 2 --fsdp 4 --ep 16 \
        --layout slices --transport ppermute ...

The boundary copies between slices compile as send / recv over DCN.
libtpu's compile-only multi-slice topology compiles no program without a
device on its first slice, so such a program compiles moved down until its
lowest slice is the first, each device to the one at the same chip
coordinates (`shift_to_first_slice`): a stage on another slice onto the
first slice's alike devices (`first_slice_meshes`), a copy between two
other slices with its lower one onto the first, a copy within another slice
onto the first (`first_slice_channel`) -- the same programs but for the
device assignment.  A stage over devices of several slices does not compile
there: its collectives span part of the slices, where an all-gather failed
a RET_CHECK and a reduce-scatter crashed the process (SIGILL; libtpu
0.0.46), so main refuses it unless --transfers-only.

`--pathways` compiles for a Pathways run instead -- one client driving every
stage (ds_ep_pipeline under JAX_PLATFORMS=proxy), not one process per host:
stages may share hosts, and what main checks first is what such a run
refuses (`pathways_problems`: --transport relay or xfer, a stage over part of
a chip's TensorCores) rather than what multi-controller JAX does
(`multihost_problems`); the plan is the one the Pathways client makes, no
stage's collectives synchronous (the Pathways servers' compiler offloads them
on chip (0,0,0)'s sub-meshes too; ds_ep_pipeline.Pipeline's `proxy`), and
under --transport ppermute no gradient-norm hop to compile (the client's are
jax.device_puts).  Here libtpu compiles, not the Pathways servers' image:

    bash src/maxtext/experimental/mpmd/jax/ds_ep_aot.sh --pathways --topology tpu7x-4x4x4 --num-slices 2 \
        --layout slices --stages 4 --vv 1 --fsdp 8 --ep 8 ...
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import pathways
from maxtext.experimental.mpmd.jax import transfer

# name -> (topology, chip_config, chips_per_host_bounds, wrap, HBM bytes per device)
# (MaxText accelerator_to_spec_map.py; HBM: the device's bytes_limit)
TOPOLOGIES = {
    "v6e-16": ("v6e:4x4", "default", (2, 2, 1), (False, False, False), 33546041344),
    "v6e-32": ("v6e:4x8", "default", (2, 2, 1), (False, False, False), 33546041344),
    "v6e-64": ("v6e:8x8", "default", (2, 2, 1), (False, False, False), 33546041344),
    "v6e-128": ("v6e:8x16", "default", (2, 2, 1), (False, True, False), 33546041344),
    "v6e-256": ("v6e:16x16", "default", (2, 2, 1), (True, True, False), 33546041344),
    # v5p (megacore: one device per chip).  HBM: what XLA compiles a program for the slice against on libtpu 0.0.46,
    # "available HBM (95.73G)" (G = GiB, as v6e's 31.24G = 33546041344, a v6e-4's): 96 GiB less 279 MiB and 512 B a
    # chip of the slice, exactly (from the bytes its compile OOMs exceed by).
    "v5p-8": ("v5:2x2x1", "megacore", (2, 2, 1), (False, False, False), 102_786_660_352),
    "v5p-16": ("v5:2x2x2", "megacore", (2, 2, 1), (False, False, False), 102_786_658_304),
    "v5p-32": ("v5:2x2x4", "megacore", (2, 2, 1), (False, False, False), 102_786_654_208),
    "v5p-64": ("v5:2x4x4", "megacore", (2, 2, 1), (False, False, False), 102_786_646_016),
    "v5p-128": ("v5:4x4x4", "megacore", (2, 2, 1), (True, True, True), 102_786_629_632),
    "v5p-256": ("v5:4x4x8", "megacore", (2, 2, 1), (True, True, True), 102_786_596_864),
    # TPU7x: two TensorCores a chip, each a device (8 a host).  HBM: the largest one-device program the
    # compile-only 4 x 8x8x8 topology compiles on libtpu 0.0.46, temporaries + code + arguments + outputs,
    # bisected to 4 KiB ("Used 94.74G of 94.74G hbm" one step above).
    "tpu7x-8x8x8": ("tpu7x:8x8x8", "default", (2, 2, 1), (True, True, True), 101_722_758_976),
    # 64 chips, 128 devices on 16 hosts (MaxText's tpu7x-128).  HBM: the largest one-device program (an int8
    # argument and its sum) the compile-only 1 x and 8 x 4x4x4 topologies compile on libtpu 0.0.46, arguments +
    # outputs + code, bisected to 4 KiB (the same probe gives 101_723_929_088 on 4 x 8x8x8, 1.1 MB above that
    # entry's figure).
    "tpu7x-4x4x4": ("tpu7x:4x4x4", "default", (2, 2, 1), (True, True, True), 101_724_389_376),
    # Two cubes, 128 chips, 32 hosts (MaxText's tpu7x-256); HBM as the 8x8x8's.
    "tpu7x-4x4x8": ("tpu7x:4x4x8", "default", (2, 2, 1), (True, True, True), 101_722_758_976),
}
G = 2**30


def topology_devices(name, num_slices=1):
  """The compile-only devices of `num_slices` slices of topology `name`.

  With jax 0.11.1 and libtpu 0.0.46 the devices of slice k carry
  slice_index k, ids (k + 1) * 100000 + i and processes of their own (16 a
  slice on a v5p-128), and every slice repeats the chip coordinates; one
  slice's devices carry no slice_index at all."""
  from jax.experimental import topologies

  topo, chip, hosts, wrap, _ = TOPOLOGIES[name]
  return topologies.get_topology_desc(
      platform="tpu",
      topology_name=topo,
      chip_config_name=chip,
      chips_per_host_bounds=hosts,
      num_slices=num_slices,
      wrap=wrap,
  ).devices


def shift_to_first_slice(meshes, devices):
  """The meshes of one program -- a stage's, or a boundary copy's two --
  where libtpu's compile-only multi-slice topology compiles it -> (meshes,
  moved?).  `devices`: all of the topology's.

  That topology (libtpu 0.0.46, under jax 0.11.1) compiles no program
  without a device on its first slice -- an elementwise one no more than a
  stage's, or a copy between two other slices: "INVALID_ARGUMENT: Slice is
  empty".  One device there is enough: a copy between the first slice and
  another compiles, as does a program over part of the first slice.  The
  slices of a topology are alike, so such a program moves k slices down, k
  putting the lowest slice it touches on the first: each device to the one
  at the same chip coordinates and core on the slice k below its own, the
  meshes in the same order.  It compiles to the same program but for the
  device assignment -- a copy between slices s and t to one between s - k and
  t - k; two stages on one slice stay disjoint.  The meshes come back as
  they are (the same objects) when they touch the first slice already, or
  the devices are of one slice."""
  ks = {dsp.slice_index(d) for d in devices}
  if len(ks) < 2 or None in ks:
    return list(meshes), False
  k = min(dsp.slice_index(d) for m in meshes for d in m.devices.flat) - min(ks)
  if not k:
    return list(meshes), False
  where = lambda d, k=0: (dsp.slice_index(d) - k, tuple(d.coords), getattr(d, "core_on_chip", 0))
  at = {where(d): d for d in devices}
  return [
      Mesh(np.array([at[where(d, k)] for d in m.devices.flat], dtype=object).reshape(m.devices.shape), m.axis_names)
      for m in meshes
  ], True


def first_slice_meshes(meshes, devices):
  """`meshes` with each stage on one slice other than the first moved onto
  the first slice's devices at the same chip coordinates
  (`shift_to_first_slice`) -> (meshes, [the stages moved]).  A stage over
  devices of several slices stays where it is: its collectives, over part of
  the slices, do not compile on the compile-only topology anyway (an
  all-gather failed a RET_CHECK, a reduce-scatter crashed the process with
  SIGILL; libtpu 0.0.46)."""
  out, moved = [], []
  for s, m in enumerate(meshes):
    if len({dsp.slice_index(d) for d in m.devices.flat}) == 1:
      (m2,), shifted = shift_to_first_slice([m], devices)
      if shifted:
        m = m2
        moved.append(s)
    out.append(m)
  return out, moved


def first_slice_channel(ch, devices):
  """A boundary copy's PermuteChannel `ch` where the compile-only topology
  compiles it -> (channel, moved?): its two stages' meshes moved together
  (`shift_to_first_slice`), a channel of the same name, avals and specs
  over them when they move, else `ch` itself."""
  (a, b), moved = shift_to_first_slice([ch.src[0].mesh, ch.dst[0].mesh], devices)
  if not moved:
    return ch, False
  on = lambda m, shardings: ch.treedef.unflatten([NamedSharding(m, s.spec) for s in shardings])
  return transfer.PermuteChannel(ch.treedef.unflatten(ch.avals), on(a, ch.src), on(b, ch.dst), name=ch.name), True


def compile_boundaries(p, devices):
  """Compile `p`'s boundary-copy programs (--transport ppermute: each runs
  on both its stages' devices) that touch a stage `p` compiles (`p.local`),
  each where the compile-only topology of `devices` compiles it
  (`first_slice_channel`) -> [((a, b), channel compiled, Compiled, lowering
  s, compile s, moved?)]."""
  out = []
  for (s, t), ch in [*p.channels.items(), *p.sq_channels.items()]:
    if s in p.local or t in p.local:
      ch, moved = first_slice_channel(ch, devices)
      t0 = time.perf_counter()
      lo = ch.prog.lower(*ch.lower_args())
      t1 = time.perf_counter()
      out.append(((s, t), ch, lo.compile(), t1 - t0, time.perf_counter() - t1, moved))
  return out


def print_moved_copies(done):
  """Say which of `compile_boundaries`' copies compiled moved -> [their names]."""
  names = [ch.name for _, ch, _, _, _, moved in done if moved]
  if names:
    print(
        f'  copies {", ".join(names)}: compiled moved down until their lower slice is the first, at the same '
        "chip coordinates (libtpu's compile-only topology compiles no program without a device on its first "
        'slice: "Slice is empty")',
        flush=True,
    )
  return names


def dump_hlo(d, compiled, compiled_stage):
  """Each compiled program's HLO text into directory d, as
  s<stage>_<name>.hlo.txt (s<a>-<b>_...: a copy over stages a and b)."""
  os.makedirs(d, exist_ok=True)
  for (name, c), s in zip(compiled, compiled_stage):
    tag = "-".join(map(str, s)) if isinstance(s, tuple) else s
    with open(os.path.join(d, f"s{tag}_{name}.hlo.txt"), "w") as f:
      f.write(c.as_text())


def _shard_bytes(aval):
  """Bytes of one device's shard of a ShapeDtypeStruct with a sharding."""
  shape = aval.sharding.shard_shape(tuple(aval.shape))
  return math.prod(shape) * jnp.dtype(aval.dtype).itemsize


def _tree_bytes(tree):
  return sum(_shard_bytes(a) for a in jax.tree.leaves(tree))


# Default share of a device's HBM `report` keeps free past the estimated peak:
# the estimate leaves out run-ahead (the next step's first programs enqueued
# while this one's last run) and, unless asked (`hbm_walk`'s `inputs`), the
# activations in flight between stages.
# Above the largest excess measured so far: 4 x v5p-128's stash plan at M=16
# peaked 3.3-6.3 GiB (3.5-6.6% of 95.7) over the estimate of the time, which
# summed every buffer.
HEADROOM = 0.1


def residual_set_bytes(p, s):
  """{chunk: bytes per device of one (chunk, microbatch) residual set on
  stage s}: its input (chunk 0's tokens, else an activation) and the
  residuals its forward returns (`ChunkPrograms.res_bytes`): its `n_res`
  layer inputs, from which the backward recomputes the chunk, or under
  --save-residuals the VJP's residual leaves (`res_avals`), on which the
  backward evaluates the VJP instead."""
  act, tok = _shard_bytes(p._avals(s, "act")), _shard_bytes(p._avals(s, "tok"))
  return {
      c.index: (tok if c.first else act) + getattr(p.progs[c.index], "res_bytes", p.progs[c.index].n_res * act)
      for c in p.stage_chunks[s]
  }


def state_bytes(p, s):
  """(storage, mu): bytes per device of stage s's params and of AdamW's mu
  (its nu takes the params' dtype: storage again)."""
  mu_dtype = p.stages[s].adam["mu_dtype"]
  storage = mu = 0
  for c in p.stage_chunks[s]:
    tree = p._avals(s, "storage", c)
    storage += _tree_bytes(tree)
    mu += _tree_bytes(jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, mu_dtype, sharding=a.sharding), tree))
  return storage, mu


def live_residuals(p):
  """Per stage, the peak of (chunk, microbatch) residual sets alive at once
  in program order -- a forward's input and residuals live until its
  backward, or a split backward's I (the fused last chunk keeps none) ->
  [(peak sets, peak bytes per device)], of sets of `residual_set_bytes`.
  What a W reads of them is its W buffer's (`hbm_walk`)."""
  out = []
  for s in range(p.S):
    size = residual_set_bytes(p, s)
    sets = live = peak_sets = peak = 0
    for c, k, m in p.order:
      if p.place[c] != s or c == p.C - 1 or k == "W":
        continue
      sign = 1 if k == "F" else -1
      sets, live = sets + sign, live + sign * size[c]
      peak_sets, peak = max(peak_sets, sets), max(peak, live)
    out.append((peak_sets, peak))
  return out


def _program_memory(exe):
  """(temporaries, outputs not aliased to a donated argument) of a compiled
  program, in bytes; (0, 0) for one not compiled (a jitted function)."""
  m = exe.memory_analysis() if hasattr(exe, "memory_analysis") else None
  if m is None:
    return 0, 0
  return m.temp_size_in_bytes, max(0, m.output_size_in_bytes - m.alias_size_in_bytes)


def hbm_walk(p, s, trace=False, inputs=False):
  """Stage s's HBM per device through one step, program by program, from the
  plan: what is live when each of its programs runs -> dict(peak (bytes),
  at (the program at the peak), parts (bytes by kind at the peak),
  uncompiled (the programs it priced without temporaries: no compiled
  executable), loop_end_weights (the gathered weights the plan still held
  when the microbatch loop ended, which the loop then drops)), and with
  `trace` the (program, parts) of every program in turn.

  Live throughout: the storage params, AdamW's nu (their dtype) and mu.
  Then, as the step loop holds them (`Pipeline._run`):
    weights   the gathered compute weights: the update's gather_slots from
              the step's start (every chunk without the fusion plan: the
              loop gathers them first), a prefetch host's leaf group from its
              program on, until fsdp_scope 'repeat' releases the chunk
              (`release_at`) or the loop ends -- `step` consumes wc0, so they
              are gone by the update;
    gacc      a chunk's accumulator, from its first gradient write until its
              reduce-scatter takes it (a host's leaf group at a time, its own
              last backward under rs_self, the stage's final reduce) -- a
              split last chunk's in its two parts, the leaves its F (AI)
              keeps from its first F, its layers' from its first W;
    grads     the reduced gradients those produce, until the update;
    residuals each (chunk, microbatch)'s input and residuals from its
              forward to its backward, or its split backward's I
              (`residual_set_bytes`: its layers' inputs, or under
              --save-residuals the VJP's residuals; `live_residuals`), stash
              sets too;
    wbuf      a split backward's W buffer (`Pipeline.wbuf_bytes`: the
              crossing set and the arguments of I its W reads, a residual or
              the cotangent among them) from its I to its W -- a split last
              chunk's from its F (AI) to its W;
    inputs    with `inputs` (else 0), the boundary inputs filed for the
              stage, a microbatch's activation shard each: an activation from
              the forward that sends it until the forward that takes it, a
              cotangent from the backward (or fused last chunk) that sends it
              until the backward that takes it (`Pipeline._file`,
              `_Box.pop`), the running program's own included.
  A program adds, while it runs, its temporaries and the outputs it does not
  write into a donated argument (`memory_analysis`, where `precompile`
  compiled it; nothing otherwise).  An I is a backward to all of this but the
  accumulator: its W is the chunk's gradient writer, and hosts what a
  backward hosts (reduce-scatters, rs_self; a split last chunk's W hosts
  nothing).  The update program closes the
  step: persistent state, the reduced gradients (donated to it), its
  temporaries and the next step's gathered weights.  Left out: the boundary inputs
  unless `inputs` -- tens of MiB each on the TPU targets here (4096 tokens a
  device: 56 MiB at DeepSeek-V3's width, 64 MiB at Qwen3.8's, half that
  under CP 2), up to M - 1 of them at once
  on a stage under breadth or gpipe (the fused last chunk's cotangents,
  piling up on the previous chunk's stage through its forwards): under 1 GiB
  at M = 16, inside `HEADROOM`; but 256 MiB a device for a 16-sequence
  Qwen3.8 microbatch at fsdp 1 x EP 4, and under --interleave-group G > S
  about G - S more of them wait at the first and the last stage
  (`schedule.megatron_interleaved`), as they do at a stage fed by a much
  cheaper chunk -- and the programs' generated code (`report` adds it)."""
  A, chs = p._avals, p.stage_chunks[s]
  st = p.stages[s]
  leaf_bytes = lambda kind, c: [_shard_bytes(x) for x in jax.tree.leaves(A(s, kind, c))]
  w_leaf = {c.index: leaf_bytes("compute", c) for c in chs}
  g_leaf = {c.index: leaf_bytes("gacc", c) for c in chs}
  r_leaf = {c.index: leaf_bytes("grads", c) for c in chs}
  storage, mu = state_bytes(p, s)
  res_size = residual_set_bytes(p, s)
  # --offload-residuals: a set lives in pinned host memory between its forward and its backward, and on the device
  # only while that backward runs (Pipeline._restore brings it back for it)
  offload = getattr(p, "offload_residuals", False)
  running_res = dict(res_size)
  if offload:
    res_size = {c: 0 for c in res_size}
  persistent = 2 * storage + mu
  act = _shard_bytes(A(s, "act"))  # one boundary input on the stage: an activation or its cotangent
  weights = {}  # chunk -> live bytes of its gathered weights
  gacc, grads = {}, 0
  live = dict(residuals=0, stash=0, wbuf=0, inputs=0)
  wbuf = getattr(p, "wbuf_bytes", {})
  # a split last chunk's accumulator: AI writes the leaves it keeps, the chunk's first W the deferred ones (its layers')
  last = p.C - 1 if getattr(p, "split_last", False) else None
  deferred = ds_ep.deferred_leaves(p.chunks[last]) if last is not None else ()
  last_w0 = next((j for j, (c, k, _) in enumerate(p.order) if c == last and k == "W"), None)
  if p.overlap:
    # what the update gathers for the next step, as the stage's programs say (with self-prefetch only the first
    # chunk's leaves besides its routed experts, which 0F0 gathers)
    for i, tree in zip(p.gather_slots[s], st._gathered_specs([A(s, "compute", c) for c in chs])):
      weights[chs[i].index] = _tree_bytes(tree)
  else:
    for c in chs:
      weights[c.index] = sum(w_leaf[c.index])
  best = dict(peak=-1, at=None, parts=None, uncompiled=[], loop_end_weights=None)
  if trace:
    best["trace"] = []

  def priced(at, exe):
    """`_program_memory` of program `at`, noting it when nothing compiled it."""
    if not hasattr(exe, "memory_analysis"):
      best["uncompiled"].append(at)
    return _program_memory(exe)

  def check(at, temp=0, out=0):
    parts = dict(
        persistent=persistent,
        weights=sum(weights.values()),
        gacc=sum(gacc.values()),
        grads=grads,
        residuals=live["residuals"],
        stash=live["stash"],
        wbuf=live["wbuf"],
        inputs=live["inputs"],
        temp=temp,
        new_outputs=out,
    )
    total = sum(parts.values())
    if total > best["peak"]:
      best.update(peak=total, at=at, parts=parts)
    if trace:
      best["trace"].append((at, parts))

  def boundary(o):
    """With `inputs`, op o's change to the inputs waiting on stage s, as the step loop takes and files them
    (`Pipeline._take_*` / `_put_*`): a forward of chunk c > 0 takes its activation on its own stage, and files chunk
    c + 1's on place[c + 1] (the fused last chunk: chunk c - 1's cotangent on place[c - 1]); a backward, or a split
    one's I, takes its cotangent and files chunk c - 1's (chunk 0's files none; a W neither: what it reads of the
    cotangent is its W buffer's)."""
    if not inputs:
      return
    c, C = o.c, p.C
    if o.kind == "F":
      took, to = c > 0, (p.place[c + 1] if c != C - 1 else p.place[c - 1] if c > 0 else None)
    elif o.kind in ("B", "I") and c != C - 1:
      took, to = True, (p.place[c - 1] if c > 0 else None)
    else:  # the fused last chunk's no-op backward, a W
      return
    live["inputs"] += act * ((to == s) - (took and o.s == s))

  check("step start")
  if not p.overlap:  # the loop gathers every chunk first: its outputs are the weights above
    check(f"gather_s{s}", priced(f"gather_s{s}", st.gather)[0])
  for op in p.ops():
    if op.s != s:  # another stage's: only the inputs it files for this one
      boundary(op)
      continue
    if op.desc is None or op.j - 1 in p.fuse:  # a no-op, a pair's second half
      continue
    pair = [op] + ([p.ops()[op.j + 1]] if op.j in p.fuse else [])
    back = sum(running_res[o.c] for o in pair if o.kind in ("B", "I")) if offload else 0
    live["residuals"] += back
    check(p.op_name(op.j), *priced(p.op_name(op.j), p._exe.get((op.c, op.pair or op.desc))))
    live["residuals"] -= back
    for o in pair:
      boundary(o)
      c = o.c
      if o.kind == "F" and c != p.C - 1:
        live["residuals"] += res_size[c]
        if o.stash:
          live["stash"] += p.stash_bytes[o.stash][c]
      elif o.kind == "F" and getattr(p, "split_last", False):  # the split last chunk's AI: its W buffer
        live["wbuf"] += wbuf[c]
      elif o.kind in ("B", "I"):
        live["residuals"] -= res_size[c]
        if o.stash:
          live["stash"] -= p.stash_bytes[o.stash][c]
        if o.kind == "I":
          live["wbuf"] += wbuf[c]
      elif o.kind == "W":
        live["wbuf"] -= wbuf[c]
      if p._writes(c, o.kind) and o.j == p.first_writer[c]:
        gacc[c] = sum(b for b, d in zip(g_leaf[c], deferred, strict=True) if not d) if c == last else sum(g_leaf[c])
      if o.j == last_w0:  # the split last chunk's first W: its accumulator's other part
        gacc[c] += sum(b for b, d in zip(g_leaf[c], deferred, strict=True) if d)
      if o.rs:
        b, g = o.rs
        gacc[b] -= sum(g_leaf[b][i] for i in g)
        grads += sum(r_leaf[b][i] for i in g)
      if o.rs_self is not None:
        gacc.pop(c)
        grads += sum(r_leaf[c])
      if o.self_pf:
        weights[c] += sum(w_leaf[c][i] for i in p.routed_leaves)
      if o.pf:
        b, g = o.pf
        weights[b] = weights.get(b, 0) + sum(w_leaf[b][i] for i in g)
      for b in p.release_at.get(o.j, ()):
        weights.pop(b, None)
  best["loop_end_weights"] = sum(weights.values())
  weights.clear()  # the loop drops its weights before the final reduces (`step` consumes wc0)
  fs = p.final_slots[s] if p.overlap else tuple(range(len(chs)))
  if fs:
    red = st._final.get((fs, p.n_partial_sq[s])) if p.overlap else st.reduce
    check(f"reduce_s{s}", *priced(f"reduce_s{s}", red))
    for i in fs:
      gacc.pop(chs[i].index, None)
      grads += sum(r_leaf[chs[i].index])
  check(f"update_s{s}", *priced(f"update_s{s}", st.update_gather if p.overlap else st.update))
  return best


def stage_programs(p):
  """{stage: [(program name, its memory analysis)]} of what `precompile`
  compiled; a boundary copy's program (compiled_stage (a, b)) runs on both
  its stages' devices and counts on both."""
  out = collections.defaultdict(list)
  for (name, c), s in zip(getattr(p, "compiled", ()), getattr(p, "compiled_stage", ())):
    m = c.memory_analysis()
    for t in s if isinstance(s, tuple) else (s,):
      out[t].append((name, m))
  return out


def report(p, hbm, headroom=HEADROOM, inputs=False):
  """Per stage this process compiles (`p.local`): what it holds in HBM per
  device at the step's peak, estimated, against `hbm` bytes -> [row].

  The estimate is `hbm_walk`'s peak -- what is live when the stage's
  heaviest program runs, the program's own temporaries and new outputs
  included, and with `inputs` the boundary inputs waiting on the stage
  (`live_inputs_gib`; 0 without) -- plus the generated code of every program
  the stage loads.  It fits when it leaves `headroom` of `hbm` free
  (HEADROOM: what the walk leaves out).  The other fields are totals for
  scale -- every chunk's compute weights, accumulators and gradients (the
  walk's `live_*` are what the plan holds of them at the peak), the peak
  residual sets, a split backward's peak W buffers (`wbuf_gib`:
  Pipeline.peak_wbuf_gib), the largest temporaries of any one program -- and
  `est_total_gib`, all of those summed: the estimate before the walk.
  `uncompiled`: the programs the walk priced without temporaries, none where
  `precompile` compiled the stage."""
  if not 0 <= headroom < 1:
    raise ValueError(f"headroom is a share of HBM in [0, 1), not {headroom}")
  A = p._avals
  rows = []
  res = live_residuals(p)
  stash = p.peak_stash_gib() if p.modes else [0.0] * p.S
  wbufs = p.peak_wbuf_gib() if getattr(p, "split_chunks", None) else [0.0] * p.S
  by_stage = stage_programs(p)
  for s in sorted(p.local) if p.multi or p.aot else range(p.S):  # one controller (mem_probe's) prices every stage
    chs = p.stage_chunks[s]
    storage, mu = state_bytes(p, s)
    compute = sum(_tree_bytes(A(s, "compute", c)) for c in chs)
    gacc = sum(_tree_bytes(A(s, "gacc", c)) for c in chs)
    grads = sum(_tree_bytes(A(s, "grads", c)) for c in chs)
    sets, resid = res[s]
    temps = sorted(((m.temp_size_in_bytes, n) for n, m in by_stage[s] if m is not None), reverse=True)
    code = sum(m.generated_code_size_in_bytes for _, m in by_stage[s] if m is not None)
    walk = hbm_walk(p, s, inputs=inputs)
    peak = walk["peak"] + code
    total = storage * 2 + mu + compute + gacc + grads + resid + (stash[s] + wbufs[s]) * G + (temps[0][0] if temps else 0)
    lv = {k: v / G for k, v in walk["parts"].items()}
    rows.append(
        dict(
            stage=s,
            storage_gib=storage / G,
            adam_gib=(storage + mu) / G,
            compute_gib=compute / G,
            gacc_gib=gacc / G,
            grads_gib=grads / G,
            residual_sets=sets,
            residuals_gib=resid / G,
            stash_gib=stash[s],
            wbuf_gib=wbufs[s],
            max_temp_gib=(temps[0][0] / G if temps else 0),
            max_temp_program=(temps[0][1] if temps else None),
            code_gib=code / G,
            peak_at=walk["at"],
            live_weights_gib=lv["weights"],
            live_gacc_gib=lv["gacc"],
            live_grads_gib=lv["grads"],
            live_residuals_gib=lv["residuals"],
            live_stash_gib=lv["stash"],
            live_wbuf_gib=lv["wbuf"],
            live_inputs_gib=lv["inputs"],
            live_temp_gib=lv["temp"] + lv["new_outputs"],
            persistent_gib=lv["persistent"],
            est_peak_gib=peak / G,
            est_total_gib=total / G,
            hbm_gib=hbm / G,
            headroom_gib=headroom * hbm / G,
            fits=peak < (1 - headroom) * hbm,
            uncompiled=walk["uncompiled"],
            top_temps=[(n, round(t / G, 3)) for t, n in temps[:6]],
        )
    )
  return rows


def exit_unless_fits(rows, hbm):
  """Exit non-zero, naming them, if any stage of the report does not fit:
  its estimated peak leaves less than its headroom of `hbm` bytes per
  device free."""
  bad = [r["stage"] for r in rows if not r["fits"]]
  if bad:
    room = max(r["headroom_gib"] for r in rows)
    sys.exit(
        f"ds_ep_aot: stages {bad} do not fit: estimated peak over the {hbm / G:.2f} GiB of HBM per device "
        f"less {room:.2f} GiB of headroom"
    )


def multihost_problems(meshes, transport):
  """What `ds_ep_pipeline --distributed` refuses at startup on these stage
  meshes, and a Pipeline built ahead of time never checks (one process, no
  runtime channels) -> [message]: stages sharing processes; else, under
  transport 'xfer', each boundary and process `transfer.xfer_peers` rejects
  (one pulling from, or giving to, more than one peer host), over every two
  stages (the gradient norm's scalar copies link them all)."""
  out = [f"stages {a} and {b} share processes {procs}" for a, b, procs in dsp.shared_stage_hosts(meshes)]
  if transport == "xfer" and not out:
    pids = sorted({d.process_index for m in meshes for d in m.devices.flat})
    for a in range(len(meshes)):
      for b in range(len(meshes)):
        if a == b:
          continue
        for pid in pids:
          try:
            transfer.xfer_peers(meshes[a], meshes[b], pid)
          except ValueError as e:
            out.append(f"s{a}->s{b}: {e}")
  return out


def pathways_problems(meshes, transport):
  """What a Pathways run of these stage meshes refuses -- one client process driving every device (ds_ep_pipeline
  under JAX_PLATFORMS=proxy) -> [message]: --transport relay or xfer (main refuses them there, they move arrays between
  processes: pathways.check_transport's message, as main and mem_probe give it), and each stage over part of a chip's
  TensorCores (`dsp.split_chip_stages`: the Pathways servers refuse it, "a subslice must use all logical devices of
  each chip").  Stages sharing hosts are no problem there."""
  out = []
  try:
    pathways.check_transport(transport)
  except ValueError as e:
    out.append(str(e))
  split = collections.defaultdict(list)
  for s, (k, coords), held, of in dsp.split_chip_stages(meshes):
    split[s].append((k, coords, held, of))
  for s, chips in split.items():
    k, coords, held, of = chips[0]
    on = "" if k is None else f" of slice {k}"
    out.append(
        f"stage {s} holds {held} of the {of} TensorCores of chip {','.join(map(str, coords))}{on}"
        + (f" (and part of {len(chips) - 1} more chips)" if len(chips) > 1 else "")
        + ": a subslice must use all logical devices of each chip"
    )
  return out


def stage_extent(m, devs):
  """Where stage mesh `m` sits on the slice of `devs`: its hosts and chip coordinates, with the z range on a slice
  more than one chip deep (v5p-16 up; every TPU device has 3 coords, z 0 throughout a v6e or a v5p-8)."""
  procs = sorted({d.process_index for d in m.devices.flat})
  xs = sorted({d.coords[0] for d in m.devices.flat})
  ys = sorted({d.coords[1] for d in m.devices.flat})
  zs = sorted({d.coords[2] for d in m.devices.flat})
  z = f" z {zs[0]}..{zs[-1]}" if any(d.coords[2] for d in devs) else ""
  return f"hosts {procs[0]}..{procs[-1]} ({len(procs)}), chips x {xs[0]}..{xs[-1]} y {ys[0]}..{ys[-1]}{z}"


def main(argv=None):
  ap = dsp.make_parser()
  ap.add_argument("--topology", default="v6e-256", choices=sorted(TOPOLOGIES))
  ap.add_argument("--threads", type=int, default=32, help="compile threads")
  ap.add_argument(
      "--stages-to-compile", default="all", help="'all' or a comma list: the stages whose programs to compile"
  )
  ap.add_argument("--hlo-dir", default="", help="dump every compiled program's HLO text here")
  ap.add_argument(
      "--headroom",
      type=float,
      default=HEADROOM,
      help="share of HBM the estimated peak must leave free to fit (what the estimate leaves out: "
      "run-ahead, activations in flight between stages); default %(default)s",
  )
  ap.add_argument(
      "--transfers-only", action="store_true", help="compile only the boundary-copy programs (--transport ppermute)"
  )
  ap.add_argument(
      "--num-slices",
      type=int,
      default=1,
      help="slices of --topology (MaxText's compile_topology_num_slices): one stage per slice with "
      "--layout slices (a stage over devices of several slices does not compile on libtpu "
      "0.0.46's compile-only topology), and with --transport ppermute the boundary copies "
      "between slices compile too (send / recv over DCN)",
  )
  ap.add_argument(
      "--pathways",
      action="store_true",
      help="compile for a Pathways run -- one client driving every stage (JAX_PLATFORMS=proxy) -- not for "
      "multi-controller JAX: stages may share hosts but not chips (a stage over part of a chip's TensorCores is "
      "refused there), --transport native or ppermute, and the plan the Pathways client makes (no stage's "
      "collectives synchronous: the Pathways servers' compiler offloads them on chip (0,0,0)'s sub-meshes too)",
  )
  a = ap.parse_args(argv)
  if a.num_slices < 1:
    sys.exit(f"--num-slices must be at least 1, not {a.num_slices}")
  if not 0 <= a.headroom < 1:
    sys.exit(f"--headroom is a share of HBM in [0, 1), not {a.headroom}")
  if a.pathways and (bad := pathways_problems([], a.transport)):  # before anything builds, as main says it there
    sys.exit(f"ds_ep_aot --pathways: {bad[0]}")
  # No persistent compile cache: jax cannot load an executable compiled for
  # compile-only topology devices (every read fails, "PjRtCompiler must be
  # constructed with a Client", and recompiles), so it would only grow.
  jax.config.update("jax_enable_compilation_cache", False)
  if jax.default_backend() != "tpu":
    sys.exit(
        f"ds_ep_aot needs a TPU backend in this process (Pallas kernels choose interpret mode from "
        f"jax.default_backend(), here {jax.default_backend()!r}); run it on a TPU VM"
    )
  # --batch-split on Lineage's layers: its programs import it (--transfers-only compiles none of them)
  if not a.transfers_only and (why := dsp.lineage_missing(a)) is not None:
    sys.exit(f"ds_ep_aot: --batch-split: {why}")
  devs = topology_devices(a.topology, a.num_slices)
  where = a.topology if a.num_slices == 1 else f"{a.num_slices} x {a.topology}"
  if a.num_slices > 1 and len({dsp.slice_index(d) for d in devs} - {None}) != a.num_slices:
    sys.exit(
        f"ds_ep_aot: jax.experimental.topologies built no {where} topology here: its devices are on "
        f"slices {sorted({str(dsp.slice_index(d)) for d in devs})}"
    )
  hbm = TOPOLOGIES[a.topology][4]
  if a.stages * a.fsdp * a.ep != len(devs):
    sys.exit(f"{a.stages} stages x fsdp {a.fsdp} x ep {a.ep} != the {len(devs)} devices of {where}")
  meshes = dsp.stage_meshes(a.stages, a.fsdp, a.ep, devices=devs, layout=a.layout)
  if span := dsp.slice_spanning_stages(meshes):
    # Before anything compiles (the pipeline's build may probe a stage's
    # collectives): over part of the topology's slices they do not compile.
    on = lambda s: sorted({dsp.slice_index(d) for d in meshes[s].devices.flat})
    why = (
        ", ".join(f"stage {s} spans slices {on(s)}" for s in span) + f" of {a.num_slices}: their FSDP and EP "
        "collectives cross DCN (one stage per slice is --layout slices)"
    )
    if not a.transfers_only and any(len(on(s)) < a.num_slices for s in span):
      sys.exit(
          f"ds_ep_aot: {why}; libtpu 0.0.46's compile-only topology compiles no collective over part of its "
          "slices (an all-gather failed a RET_CHECK, a reduce-scatter crashed the process: SIGILL). "
          "--transfers-only compiles the boundary copies alone"
      )
    warnings.warn(why, RuntimeWarning)
  if a.transfers_only and a.ep_collectives == "auto":
    # Only the boundary copies compile, so the stages' collective flavours do not matter: pin the choice the
    # pipeline's build asks libtpu (sparsecore_offload_available compiles an all-gather over each stage, which
    # over part of the slices does not compile -- the refusal above).
    a = argparse.Namespace(**{**vars(a), "ep_collectives": "xla"})
    print("--transfers-only: the stages' collectives taken as xla, unprobed (no stage program compiles)", flush=True)
  lineage_sc = None
  if a.transfers_only and dsp.uses_lineage(a):
    # the same for Lineage's SparseCore regions (the pipeline's build asks libtpu over each whole stage): the device
    # rule's answers, unprobed
    sync = frozenset() if a.pathways else None  # a Pathways run: no stage synchronous (sync_collective_stages)
    lineage_sc = tuple(bool(dsp.rule_offloads(meshes, s, ds_ep.FE, sync)) for s in range(len(meshes)))
  proxy = True if a.pathways else None  # None: the pipeline asks the devices (a compile-only topology's: not Pathways)
  t0 = time.perf_counter()
  cmeshes, moved = first_slice_meshes(meshes, devs)
  if moved:
    # The programs of the stages on other slices than the first are built on
    # the first slice's alike devices (q), with no boundary copies (two stages
    # may share devices there).  The pipeline over the real stages (p: the
    # plan, the summary, the copies between slices) takes q's per-stage EP
    # collective flavours, and under --batch-split on Lineage's layers q's
    # answers for their SparseCore regions: libtpu's probe
    # (sparsecore_offload_available) cannot compile on those slices either.
    # The FSDP flavour is one for every stage, which p's build resolves from
    # the same flags as q's.
    q = dsp.build(
        argparse.Namespace(**{**vars(a), "transport": "native"}),
        meshes=cmeshes,
        aot=True,
        stage_lineage_sparsecore=lineage_sc,
        proxy=proxy,
    )
    p = dsp.build(
        argparse.Namespace(**{**vars(a), "ep_collectives": ",".join(c.ep_collectives for c in q.stage_scfg)}),
        meshes=meshes,
        aot=True,
        stage_lineage_sparsecore=q.lineage_sc,
        proxy=proxy,
    )
  else:
    p = q = dsp.build(a, meshes=meshes, aot=True, stage_lineage_sparsecore=lineage_sc, proxy=proxy)
  flags = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
  print(
      f"AOT for {where} ({len(devs)} x {devs[0].device_kind}, "
      f"{len({d.process_index for d in devs})} hosts); built in {time.perf_counter() - t0:.1f}s",
      flush=True,
  )
  print(f"LIBTPU_INIT_ARGS={flags!r}", flush=True)
  if "--xla_max_concurrent_async_collective_permutes=" not in flags:  # the one every launcher adds
    warnings.warn(
        "LIBTPU_INIT_ARGS is not the run's flag set, which changes what compiles (run "
        "src/maxtext/experimental/mpmd/jax/ds_ep_aot.sh: the libtpu flag set MPMD_FLAG_SET names, plus the MPMD one)",
        RuntimeWarning,
    )
  # multi-controller JAX's warning (its native copy hangs between slices); not asked when planned for one Pathways
  # client (--pathways), whose boundary copies the Pathways servers carry (the note)
  if not a.pathways and (msg := dsp.cross_slice_warning(meshes, a.transport)):
    warnings.warn(msg, RuntimeWarning)
  elif note := dsp.cross_slice_note(meshes, a.transport, proxy=proxy):
    print(f"  boundary copies: {note}", flush=True)
  elif a.transport == "relay" and len({dsp.slice_index(d) for m in meshes for d in m.devices.flat} - {None}) > 1:
    # it crosses them, on the hosts: the boundaries are no programs (ppermute's are, and compile below)
    print(
        "  boundary copies: --transport relay moves them between the slices on the hosts at run time, so none "
        "compiles here",
        flush=True,
    )
  print(p.summary(), flush=True)
  for s, m in enumerate(meshes):
    on = sorted({str(dsp.slice_index(d)) for d in m.devices.flat})
    sl = f'slice {",".join(on)}, ' if a.num_slices > 1 else ""
    print(f"  stage {s}: {sl}{stage_extent(m, devs)}", flush=True)
  if moved:
    print(
        f"  stages {moved}: their programs compile on slice {min(map(dsp.slice_index, devs))}'s devices at the "
        "same chip coordinates (libtpu's compile-only topology compiles no program without a device on its "
        'first slice: "Slice is empty")',
        flush=True,
    )
  if a.pathways:
    bad, who = pathways_problems(meshes, a.transport), "a Pathways run (one client driving every stage)"
    print(
        "  planned for one Pathways client (--pathways): stages may share hosts, no stage's collectives synchronous",
        flush=True,
    )
  else:
    bad, who = multihost_problems(meshes, a.transport), "`ds_ep_pipeline --distributed`"
  if bad:
    sys.exit(
        f"ds_ep_aot: {who} refuses this on {where} at startup:\n  "
        + "\n  ".join(bad[:8])
        + (f"\n  ... ({len(bad)} in all)" if len(bad) > 8 else "")
    )
  if a.stages_to_compile != "all":
    p.local = q.local = frozenset(int(x) for x in a.stages_to_compile.split(","))
    if not p.local <= set(range(p.S)):
      sys.exit(f"--stages-to-compile {a.stages_to_compile}: the stages are 0..{p.S - 1}")
  if a.transfers_only:
    if not any(s in p.local or t in p.local for s, t in [*p.channels, *p.sq_channels]):
      sys.exit(
          f"--transfers-only: no boundary-copy programs on stages {sorted(p.local)} (they exist with "
          "--transport ppermute)"
      )
    t0 = time.perf_counter()
    done = compile_boundaries(p, devs)
    for _, ch, c, _, _, _ in done:
      m = c.memory_analysis()
      print(
          f"  {ch.name}: compiled over {dict(ch.union.shape)}; temp {m.temp_size_in_bytes / 2**20:.1f} MiB, "
          f"args {m.argument_size_in_bytes / 2**20:.1f} MiB, out {m.output_size_in_bytes / 2**20:.1f} MiB",
          flush=True,
      )
    print(f"compiled {len(done)} boundary-copy programs in {time.perf_counter() - t0:.1f}s", flush=True)
    print_moved_copies(done)
    if a.hlo_dir:
      dump_hlo(a.hlo_dir, [(getattr(ch.prog, "__name__", "?"), c) for _, ch, c, *_ in done], [st for st, *_ in done])
    return []
  n, dt = q.precompile(threads=a.threads)
  copies_moved = []
  if q is not p:
    # the copies between the real stages, which q has none of: they count on both their stages
    t0 = time.perf_counter()
    done = compile_boundaries(p, devs)
    for st, ch, c, tl, tc, _ in done:
      q.compiled.append((getattr(ch.prog, "__name__", "?"), c))
      q.compiled_stage.append(st)
      q.compile_times.append((tl, tc))
      n += 1
    dt += time.perf_counter() - t0
    copies_moved = print_moved_copies(done)
  tl = sum(x for x, _ in q.compile_times)
  tc = sum(y for _, y in q.compile_times)
  print(
      f"compiled {n} programs for stages {sorted(q.local)} in {dt:.1f}s (summed over threads: lowering "
      f"{tl:.0f}s, XLA compile {tc:.0f}s)",
      flush=True,
  )
  if a.hlo_dir:
    dump_hlo(a.hlo_dir, q.compiled, q.compiled_stage)
  rows = report(q, hbm, a.headroom)
  for r in rows:
    print(
        f"  stage {r['stage']}: per device GiB -- storage {r['storage_gib']:.2f} (x2 with nu), mu "
        f"{r['adam_gib'] - r['storage_gib']:.2f}; at the peak ({r['peak_at']}): compute weights "
        f"{r['live_weights_gib']:.2f} (of {r['compute_gib']:.2f}), grad acc {r['live_gacc_gib']:.2f} (of "
        f"{r['gacc_gib']:.2f}), grads {r['live_grads_gib']:.2f}, residuals {r['live_residuals_gib']:.2f}, stash "
        f"{r['live_stash_gib']:.2f}, "
        + (f"W buffers {r['live_wbuf_gib']:.2f} (peak {r['wbuf_gib']:.2f}), " if r["wbuf_gib"] else "")
        + f"the program's temporaries + new outputs {r['live_temp_gib']:.2f}; code "
        f"{r['code_gib']:.2f} => est. peak {r['est_peak_gib']:.2f} of {r['hbm_gib']:.2f} less "
        f"{r['headroom_gib']:.2f} headroom {'OK' if r['fits'] else 'DOES NOT FIT'} (every buffer summed: "
        f"{r['est_total_gib']:.2f})",
        flush=True,
    )
    print(f"           largest temporaries: {r['top_temps']}", flush=True)
    if r["uncompiled"]:
      u = r["uncompiled"]
      print(
          f"           WARNING: {len(u)} programs priced without temporaries (not compiled): "
          f"{', '.join(u[:8])}{' ...' if len(u) > 8 else ''}",
          flush=True,
      )
  if a.json:
    dsp.resolved_config(a, p)
    with open(a.json, "w") as f:
      json.dump(
          {
              "config": vars(a),
              "libtpu_init_args": flags,
              "summary": p.summary(),
              "programs": n,
              "compile_s": dt,
              "stages": rows,
              "compiled_on_first_slice": moved,
              "copies_moved": copies_moved,
          },
          f,
          indent=1,
          default=str,
      )
  exit_unless_fits(rows, hbm)
  return rows


if __name__ == "__main__":
  main()
