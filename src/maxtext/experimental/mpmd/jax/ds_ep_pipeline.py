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

"""MPMD pipeline of DeepSeek-V3 sparse layers with FSDP x EP inside each stage
(or, with `--model qwen3.8-2.4t-a95b` / another QWEN_MODELS name, of a Qwen3.5-family
model's: Gated DeltaNet and gated-attention MoE layers, ds_ep.layer_fwd with
ds_ep.qwen_mixer_part in attention's place).

    S stages, each its own jitted programs on a disjoint ('fsdp','expert')
    sub-mesh; the model is cut into C = S*V chunks placed circularly
    (chunk c on stage c % S), which with one layer per chunk is exactly
    MaxText's circular-pipeline layer map (layer = repeat * S + stage) --
    or, with `placement='v'` (V = 2), V-shaped: stage s holds chunks s and
    2S-1-s, so the embedding and the head share stage 0 and the turn at
    chunk S-1 -> S stays on one stage.

One training step, per stage:

    gather   storage (fp32, FSDP-sharded) -> compute (bf16, gathered)   once
    F / B    per microbatch, in `schedule` order, against the gathered
             weights; weight gradients accumulate per device, unreduced
    reduce   one reduce-scatter of the accumulated gradients              once
    update   AdamW with global-norm clipping (one scalar hop per stage)   once

Each (chunk, microbatch) runs in a mode, 'r' by default: its forward keeps
only the layer inputs and its backward recomputes the layers (MaxText's remat
'full').  `--schedule stash` gives some of them a stash mode instead
(ds_ep.STASH_POLICIES): the forward also keeps what the policy names and the
backward skips recomputing it, as many at once as a per-stage HBM budget
allows (schedule.stash_search).

An F&B pair runs the forward of one microbatch and the backward of another,
both of one stage, as ONE program (`ds_ep.compose` of the two parts), so XLA
may overlap one's collectives and SparseCore work with the other's compute:
`--schedule interleaved` runs its steady state's F and B as pairs on the
stages `--pair-stages` names (none unless it is given: Megatron-LM's combined
1F1B, one microbatch's expert-parallel collectives beside another's compute).
It takes `--stash` too: the stash modes of its fixed order under the budget
(schedule.stash_modes).

The driver is one Python thread that only enqueues: nothing in `step` blocks,
so every device's FIFO queue is kept full and idle time on a device can only
come from a data dependency (a pipeline bubble) -- never from the host.

    python3 -m maxtext.experimental.mpmd.jax.ds_ep_pipeline --bench --profile /tmp/xprof/mpmd
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import importlib.util
import io
import itertools
import json
import numbers
import os
import queue
import re
import shutil
import socket
import stat
import statistics
import subprocess
import sys
import threading
import time
import warnings

# The PJRT run-ahead cap must be raised before the TPU client exists.  Under Pathways (JAX_PLATFORMS=proxy) there is
# no PJRT client here to cap (the IFRT proxy client takes no options; in local runs the client ran ~90 steps ahead
# without blocking), nor a jax compile cache to set up (pathways.initialize turns it off: Pathways caches compiled
# programs under its servers' --gcs_scratch_location).
from maxtext.experimental.mpmd.jax import pathways  # noqa: E402  (imports neither jax nor pathwaysutils)
from maxtext.experimental.mpmd.jax import transfer  # noqa: E402  (imports jax, but no backend init)

if __name__ == "__main__" and os.environ.get("JAX_PLATFORMS", "") != "cpu" and not pathways.is_proxy():
  transfer.set_max_inflight(1024)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

if __name__ == "__main__" and not pathways.is_proxy():
  # Programs are re-lowered every run; the persistent cache makes the second
  # run of an unchanged program a disk read instead of a multi-minute compile.
  # Its key strips op names (locations), so after a change that alters only
  # names -- adding, removing or renaming a named scope -- point JAX_CACHE_DIR
  # somewhere fresh, or every program whose IR is otherwise unchanged keeps its
  # old names in the trace.
  jax.config.update("jax_compilation_cache_dir", os.environ.get("JAX_CACHE_DIR", os.path.expanduser("~/jax_cache")))
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def cache_every_process():
  """Under jax.distributed jax writes persistent-cache entries from process 0
  only (compiler._cache_write: contention on a shared filesystem such as GCS),
  so every other process -- every other stage -- compiles all its programs
  again on every run.  Where each process's JAX_CACHE_DIR is a disk of its own
  (a GPU node's local SSD: a GPU job's entry script sets
  MPMD_CACHE_EVERY_PROCESS=1) every process may write its own entries: the
  check reads jax's process id, so the write runs as process 0's does.  Best
  effort: a jax without these internals keeps its own rule.  The write is
  jax's own (compilation_cache.put_executable_and_time, behind the same
  minimum compile time), without the process check."""
  try:
    from jax._src import compilation_cache as _cc
    from jax._src import compiler as _compiler
    from jax._src import config as _config
  except ImportError:
    return False
  write = getattr(_compiler, "_cache_write", None)
  put = getattr(_cc, "put_executable_and_time", None)
  if write is None or put is None or getattr(write, "_mpmd_every_process", False):
    return False

  def every_process_write(cache_key, compile_time_secs, module_name, backend, executable):
    if compile_time_secs < _config.persistent_cache_min_compile_time_secs.value:
      return
    try:
      put(cache_key, module_name, executable, backend, int(compile_time_secs))
    except Exception as e:  # jax warns and goes on likewise
      warnings.warn(f"Error writing persistent compilation cache entry for '{module_name}': " f"{type(e).__name__}: {e}")

  every_process_write._mpmd_every_process = True
  _compiler._cache_write = every_process_write
  return True


if __name__ == "__main__" and os.environ.get("MPMD_CACHE_EVERY_PROCESS", "") == "1" and not pathways.is_proxy():
  cache_every_process()
import numpy as np  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig  # noqa: E402
from maxtext.experimental.mpmd.jax.qwen35.config import GDN_CONV_IMPLS, GDN_SCAN_IMPLS, Qwen35Config  # noqa: E402
from maxtext.experimental.mpmd.jax import ds_ep  # noqa: E402
from maxtext.experimental.mpmd.jax import ds_ep_lineage  # noqa: E402  (Lineage itself loads when a program traces)
from maxtext.experimental.mpmd.jax import profiling  # noqa: E402
from maxtext.experimental.mpmd.jax import schedule as sched  # noqa: E402
from maxtext.experimental.mpmd.jax import xprof_align  # noqa: E402


# ------------------------------------------------------------------- meshes
LAYOUTS = ("fsdp_on_chip", "ep_on_chip", "snake", "physical", "physical_origin_last", "slices")


def libtpu_flag(flag, args=None):
  """Boolean libtpu flag `flag` (its name, no dashes) as LIBTPU_INIT_ARGS (or `args`) sets it:
  True, False, or None when the variable does not name the flag (libtpu's own default then
  holds).  The last mention wins, in any absl spelling: one dash or two, --flag, --flag=VALUE,
  --noflag."""
  args = os.environ.get("LIBTPU_INIT_ARGS", "") if args is None else args
  val = None
  for tok in args.split():
    m = re.fullmatch(r"--?(no)?" + re.escape(flag) + r"(?:=(.*))?", tok)
    if m is None:
      continue
    if m.group(1):
      val = False
    elif m.group(2) is None:
      val = True
    else:  # anything absl would not read as true counts as off
      val = m.group(2).lower() in ("true", "t", "1", "yes", "y")
  return val


def slice_index(d):
  """Device d's slice: its `slice_index` (CPU devices under jax.distributed
  carry 0), None where it has none (CPU in one process, a compile-only
  topology of one slice)."""
  return getattr(d, "slice_index", None)


def slice_groups(devices=None):
  """The devices (default: every device of the job) by slice -> [[device]],
  in slice order, each slice's devices in (process, id) order.

  TPU devices of a multi-slice job, or of a multi-slice compile-only topology
  (`ds_ep_aot --num-slices`), carry their slice's `slice_index`.  Devices of
  one slice -- slice 0, or none at all (CPU in one process; a one-slice
  topology) -- are one group, unless MPMD_FAKE_SLICES=N stands in for N
  slices: the devices in (process, id) order, cut into N equal contiguous
  groups.  A host belongs to one slice, so no group may split a process --
  unless one process holds them all: a one-process run, as the multi-process
  tests' reference is, or a Pathways client's (pathways.multi_controller of
  the devices: then no process is looked at).  Devices of several slices
  refuse MPMD_FAKE_SLICES (a leftover setting must not re-cut a multi-slice
  job); CPU devices under jax.distributed carry slice 0, and take it.
  """
  devs = list(jax.devices() if devices is None else devices)
  multi = pathways.multi_controller(devs)  # several processes' devices (multi-controller, a topology's hosts)
  order = (lambda d: (d.process_index, d.id)) if multi else (lambda d: d.id)  # (process, id); one process's: id
  fake = os.environ.get("MPMD_FAKE_SLICES", "")
  if fake:
    n = int(fake) if fake.isdigit() else 0
    if n < 1:
      raise ValueError(f"MPMD_FAKE_SLICES={fake!r}: want a positive number of slices")
    if len(own := sorted({k for k in map(slice_index, devs) if k is not None})) > 1:
      raise ValueError(f"MPMD_FAKE_SLICES={fake}: these devices are on slices {own} of their own; unset it")
    if len(devs) % n:
      raise ValueError(f"MPMD_FAKE_SLICES={n}: {len(devs)} devices do not cut into {n} equal slices")
    devs, k = sorted(devs, key=order), len(devs) // n
    groups = [devs[i * k : (i + 1) * k] for i in range(n)]
    if multi:
      procs = [{d.process_index for d in g} for g in groups]
      split = sorted({q for i in range(n) for j in range(i) for q in procs[i] & procs[j]})
      if split:
        raise ValueError(
            f"MPMD_FAKE_SLICES={n}: slices of {k} devices would split processes {split} (a host belongs to one slice)"
        )
    return groups
  groups = collections.defaultdict(list)
  for d in devs:
    groups[slice_index(d)].append(d)
  return [sorted(groups[k], key=order) for k in sorted(groups, key=lambda k: -1 if k is None else k)]


def slice_spanning_stages(meshes):
  """The stages whose devices lie on more than one slice -> [stage]: their
  FSDP and EP collectives cross DCN.  --layout slices makes none; the layouts
  that cut stages in device-id order make one wherever a stage's share of
  the ids straddles two slices (whether a multi-slice job's runtime ids are
  grouped by slice is unverified; a compile-only topology's are)."""
  return [s for s, m in enumerate(meshes) if len({slice_index(d) for d in m.devices.flat} - {None}) > 1]


# Numbers the key-value store keys of each multi-controller pipeline's offload answers (`offload_answers`):
# every process builds its pipelines in one order, so the n-th's are under one key on every process.
_OFFLOAD_KEYS = itertools.count()


# The boundary transports that cross TPU slices: the host relay (every MPMD
# run of PP over DCN on 2 x v5p-128, 2026-10) and the collective-permute program.
# Between slices jax's native copy hangs and jax.experimental.transfer's
# receiver crashes within a few pulls (2026-10, jax 0.11.1, libtpu 0.0.46).
CROSS_SLICE_TRANSPORTS = ("relay", "ppermute")


def stage_slices(meshes):
  """The slices the stages' devices are on -> sorted [slice_index] (none for devices without one)."""
  return sorted({k for m in meshes for k in map(slice_index, m.devices.flat) if k is not None})


def on_pathways(meshes):
  """Are the meshes' devices Pathways' (the IFRT proxy's: `pathways.is_proxy_device`)?"""
  return bool(meshes) and pathways.is_proxy_device(meshes[0].devices.flat[0])


def cross_slice_warning(meshes, transport):
  """What to warn of when the stages span TPU slices and `transport` is not
  one that crosses them (CROSS_SLICE_TRANSPORTS) -> str, else None.  Under
  Pathways none: there the one process's native copy is an IFRT CopyArrays
  that the Pathways servers carry between the slices -- not jax's cross-host
  copy, which hung -- and the host transports need --distributed, which
  Pathways refuses (`cross_slice_note` says how the boundaries cross)."""
  across = stage_slices(meshes)
  if len(across) < 2 or transport in CROSS_SLICE_TRANSPORTS or on_pathways(meshes):
    return None
  return (
      f"the stages span slices {across} and --transport {transport} crosses them: between TPU slices jax's "
      "native copy hangs and jax.experimental.transfer's receiver crashes within a few pulls "
      "(seen with jax 0.11.1, libtpu 0.0.46); --transport relay or ppermute crosses them"
  )


def cross_slice_note(meshes, transport, proxy=None):
  """Under Pathways, when the stages span TPU slices: how the boundary copies
  cross them -> str, else None.  The native copy is one IFRT CopyArrays a
  boundary, device to device, moved by the Pathways servers -- inside a
  slice at McJAX's own rate (186 GB/s between two 2-chip halves of a
  tpu7x-8, local docker Pathways); between slices unmeasured here (partir's
  compiled CopyArrays crossed DCN) -- and ppermute a collective-permute
  program over both stages' slices.  `proxy`: whether the stages run under
  Pathways (None: `on_pathways`)."""
  across = stage_slices(meshes)
  if len(across) < 2 or not (on_pathways(meshes) if proxy is None else proxy):
    return None
  how = {
      "native": "one IFRT CopyArrays, which the Pathways servers carry between the slices",
      "ppermute": "a collective-permute program over both stages' slices",
  }.get(transport, f"--transport {transport}")
  return f"the stages span slices {across}: under Pathways each boundary copy (--transport {transport}) is {how}"


def chip_strips(devices, k, name="the devices"):
  """`devices` -- one slice's, each with its chip `coords` -- cut into k strips of whole chips -> [[device]], each in
  `devices`' order: along the last chip axis (x, y, z) k divides, strip j the j-th k-th of the chips along it, every
  TensorCore of each (--layout slices at k stages a slice).  Along z, a 3D slice's last axis, a host (2x2x1 chips on
  TPU7x and v5p) is one chip deep, so each strip there holds whole hosts.  A ValueError (`name`: the devices') where
  they are not every TensorCore of a solid box of chips (a fake slice cut through a chip or across a box), or where
  no chip axis of the box is a multiple of k."""
  coords = [tuple(d.coords) for d in devices]
  chips = collections.Counter(coords)
  lo = np.min(coords, axis=0)
  dims = [int(n) for n in np.max(coords, axis=0) - lo + 1]
  box = "x".join(map(str, dims))
  if len(chips) != int(np.prod(dims)) or len(set(chips.values())) != 1:
    raise ValueError(
        f"layout 'slices': {name}'s {len(devices)} devices are not every TensorCore of a solid box of chips "
        f"({len(chips)} chips in a {box} box; MPMD_FAKE_SLICES on a TPU must cut whole boxes)"
    )
  axes = [a for a, n in enumerate(dims) if n % k == 0]
  if not axes:
    ks = sorted({j for n in dims for j in range(2, n + 1) if n % j == 0})
    raise ValueError(
        f"layout 'slices' at {k} stages a slice cuts each slice's chips into {k} strips of whole chips along a chip "
        f"axis {k} divides (every TensorCore of a chip in one stage: Pathways refuses a stage over part of a chip's): "
        f"{name}'s chips are {box}, no axis a multiple of {k} "
        + (f"(k = {' or '.join(map(str, ks))} stages a slice cut it)" if ks else "(one stage a slice)")
    )
  a = axes[-1]
  width = dims[a] // k
  return [[d for d, c in zip(devices, coords) if (c[a] - lo[a]) // width == j] for j in range(k)]


def stage_meshes(n_stages=2, fsdp=2, ep=2, devices=None, layout="ep_on_chip"):
  """One ('fsdp','expert') mesh per stage over disjoint devices.

  'ep_on_chip' puts each EP pair on the two TensorCores of one chip (the
  ring-of-experts all-gather / reduce-scatter runs every layer and
  microbatch), and FSDP across chips (its gather / scatter runs once a step).
  It is the layout MaxText's pipeline-large-moe mesh gets on this VM.
  'fsdp_on_chip' is the transpose: FSDP over a chip's two cores, EP across
  chips.  Which is better is a question of bytes: at 128 experts a device
  receives ~13 GB of FSDP gather + reduce-scatter a step against ~5 GB of
  ring-of-experts exchange, and fsdp_on_chip is 1603 vs 1699 ms/step.
  Stage s takes chips [s*k, (s+1)*k) in device-id order, so stage boundaries
  pair device i with device i + fsdp*ep.

  'snake' orders the devices along a boustrophedon path over the chip grid
  (both cores of a chip, then the next chip along the row, rows alternating
  direction).  With one or two devices per stage that keeps every stage
  boundary on a direct link: on a 2x2 grid, (0,0) -> (1,0) -> (1,1) -> (0,1).

  'physical' takes the stages from the rows of mesh_utils' topology-aware
  (n_stages, fsdp*ep) device mesh over the whole slice (MaxText's mesh with
  allow_split_physical_axes): on a v6e-256 at 4 stages, four 16x4 strips of
  the 16x16 torus, each 16 whole hosts -- the custom 4x64 mesh.  Each stage's
  devices keep that mesh's order, fsdp-major.

  'physical_origin_last' is 'physical' with the strips rotated cyclically
  until the strip holding chip (0,0,0) runs the LAST stage.  libtpu never
  offloads a collective of a proper sub-box holding that chip to SparseCore
  (`sync_collective_stages`), so under 'physical' the first stage -- which
  opens and closes every step -- runs its FSDP all-gathers and reduce-scatters
  synchronously on the TensorCore.  Rotated, the synchronous strip runs the
  last stage, which waits ~100 ms for its first input and idles ~290 ms in the
  drain (v6e-256, M=16).  On a slice that wraps around across the strips (the
  v6e-256 torus) every stage boundary stays between neighbouring strips; on
  one that does not (v6e-16 to v6e-128) the rotation puts stages S-2 and S-1
  on the slice's two far ends.  A device set without chip (0,0,0) is left in
  'physical' order.  Under Pathways the rotation buys nothing: its compiler
  offloads the origin strip's collectives too.  'snake' and both 'physical'
  layouts place by chip coordinates, which every slice repeats, so they
  refuse devices of several slices.

  'slices' runs k stages on each slice (`slice_groups`), n_stages = k x the
  slices: stages k s .. k s + k - 1 are slice s, in order, and fsdp x ep is a
  k-th of its devices; a boundary between slices crosses DCN.  At k = 1 stage
  s is all of slice s, its mesh mesh_utils' topology-aware (fsdp, ep) mesh
  over its own slice alone (MaxText's ICI mesh of one slice, with
  allow_split_physical_axes).  At k > 1 the slice's chips are cut into k
  strips of whole chips along the last chip axis k divides (`chip_strips`;
  a slice with none is refused), each strip's mesh mesh_utils' topology-aware
  (fsdp, ep) mesh over its own devices, as a whole slice's is at k = 1: two
  stages a slice on 2 x tpu7x-4x4x4 are each a 4x4x2 half of its slice (z),
  four are 4x4x1 quarters, whole hosts each.  Both TensorCores of a TPU7x
  chip are in one stage -- Pathways refuses a stage over part of a chip's
  ("a subslice must use all logical devices of each chip"), which the rows
  of mesh_utils' (k, fsdp x ep) mesh over a TPU7x slice were, its k axis on
  the cores.  Devices without chip coordinates (CPU, with MPMD_FAKE_SLICES)
  keep (process, id) order, cut into k contiguous strips.  A whole slice runs
  its collectives on SparseCore, and so does a strip away from the slice's
  chip (0,0,0); the strip holding it -- the first of its slice -- keeps them
  synchronous under McJAX (`sync_collective_stages`; Pathways offloads
  them).  Of the boundary transports `--transport relay` (the host relay:
  every MPMD run of PP over DCN on 2 x v5p-128) and `--transport ppermute`
  cross TPU slices; jax's native copy and jax.experimental.transfer do not
  (`CROSS_SLICE_TRANSPORTS`).
  """
  if layout not in LAYOUTS:
    raise ValueError(f"unknown layout {layout!r}: expected one of {LAYOUTS}")
  devices = list(devices if devices is not None else jax.devices())
  per = fsdp * ep
  if len(devices) < n_stages * per:
    raise ValueError(f"need {n_stages * per} devices, have {len(devices)}")
  if layout in ("snake", "physical", "physical_origin_last"):
    across = sorted({k for k in map(slice_index, devices) if k is not None})
    if len(across) > 1:
      raise ValueError(
          f"layout {layout!r} places stages by chip coordinates, which every slice repeats: these "
          f"devices span slices {across} (one stage per slice is --layout slices)"
      )
  if layout == "slices":
    groups = slice_groups(devices)
    k = n_stages // len(groups) if n_stages % len(groups) == 0 else 0
    if not k or any(len(g) != k * per for g in groups):
      raise ValueError(
          f"layout 'slices' runs k stages on each slice (n_stages a multiple of the slices): {n_stages} stages "
          f"of fsdp x ep = {per} devices, but the devices are {len(groups)} slice(s) of {[len(g) for g in groups]}"
      )
    from jax.experimental import mesh_utils

    meshes = []
    for i, g in enumerate(groups):
      if all(getattr(d, "coords", None) is not None for d in g):
        strips = [(f"slice {i}'s", g)]
        if k > 1:
          strips = [(f"slice {i}'s strip {j}'s", s) for j, s in enumerate(chip_strips(g, k, f"slice {i}"))]
        for what, devs in strips:
          try:
            arr = mesh_utils.create_device_mesh((fsdp, ep), devs, allow_split_physical_axes=True)
          except AssertionError as e:  # its message lists every device of the slice, the box's dims last
            why = str(e) if len(str(e)) < 300 else f"{str(e)[:120]} ... {str(e)[-120:]}"
            raise ValueError(
                f"layout 'slices': mesh_utils cannot build {what} {(fsdp, ep)} mesh, its "
                f"chips not a solid box (MPMD_FAKE_SLICES on a TPU must cut whole boxes): {why}"
            ) from e
          meshes.append(Mesh(np.asarray(arr, dtype=object).reshape(fsdp, ep), ("fsdp", "expert")))
      else:
        arr = np.array(g, dtype=object).reshape(k, per)
        meshes += [Mesh(arr[j].reshape(fsdp, ep), ("fsdp", "expert")) for j in range(k)]
    return meshes
  if layout in ("physical", "physical_origin_last"):
    from jax.experimental import mesh_utils

    if len(devices) != n_stages * per:
      raise ValueError(f"layout {layout!r} cuts the whole slice: {n_stages} x {per} devices, have " f"{len(devices)}")
    grid = mesh_utils.create_device_mesh((n_stages, per), devices, allow_split_physical_axes=True)
    rows = [list(grid[s]) for s in range(n_stages)]
    if layout == "physical_origin_last":
      o = next((s for s, row in enumerate(rows) if any(map(on_origin_chip, row))), None)
      if o is not None:
        rows = rows[o + 1 :] + rows[: o + 1]
    return [Mesh(np.array(row, dtype=object).reshape(fsdp, ep), ("fsdp", "expert")) for row in rows]
  devices = sorted(devices, key=lambda d: d.id)
  if layout == "snake":

    def snake(d):
      x, y = tuple(getattr(d, "coords", (d.id, 0)))[:2]
      return (y, x if y % 2 == 0 else -x, getattr(d, "core_on_chip", 0))

    devices = sorted(devices, key=snake)
  meshes = []
  for s in range(n_stages):
    blk = devices[s * per : (s + 1) * per]
    arr = np.array(blk, dtype=object).reshape(fsdp, ep)
    if layout == "fsdp_on_chip":
      arr = np.array(blk, dtype=object).reshape(ep, fsdp).T
    meshes.append(Mesh(arr, ("fsdp", "expert")))
  return meshes


def typed_meshes(meshes, shard_mode):
  """`meshes` with the axis types `shard_mode` (ds_ep.StageConfig.shard_mode)
  runs on: Explicit under 'explicit' -- the gathered weights and gradient
  accumulators carry reduced / unreduced types, which only Explicit axes
  take -- else Auto.  A mesh that has them already is returned as it is."""
  want = AxisType.Explicit if shard_mode == "explicit" else AxisType.Auto
  return [
      m if all(t == want for t in m.axis_types) else Mesh(m.devices, m.axis_names, axis_types=(want,) * len(m.axis_names))
      for m in meshes
  ]


def on_origin_chip(d):
  """Is device d on its slice's chip (0,0,0)?  By the chip coordinates where
  the device has them (TPU, compile-only topology devices too), else device 0."""
  coords = getattr(d, "coords", None)
  return all(x == 0 for x in coords) if coords is not None else d.id == 0


def _is_tpu(d):
  return getattr(d, "platform", "") == "tpu" or str(getattr(d, "device_kind", "")).startswith("TPU")


def allocates_at_issue(d):
  """Does device d's client allocate a program's outputs when the program is issued, from the issuing thread -- a TPU
  VM's own (McJAX), not the IFRT proxy's (Pathways)?  There `Pipeline._do_I` paces the W buffers (pace_i)."""
  return _is_tpu(d) and not pathways.is_proxy_device(d)


def pin_model_mismatch(devices):
  """The kinds of the TPU devices among `devices` that no ds_ep.PinModel, which
  --prefetch-pin auto plans with, was calibrated on (ds_ep.PIN_MODELS) -> a
  sorted list: none on v5p or v6e, nor off TPU."""
  return sorted({str(d.device_kind) for d in devices if _is_tpu(d)} - set(ds_ep.PIN_MODELS))


def sync_collective_stages(meshes):
  """The stages whose all-gathers / reduce-scatters libtpu runs synchronously
  on the TensorCore -> frozenset.

  It offloads them to SparseCore, where they run async next to the program's
  compute, for a box of the slice away from its origin and for the whole
  slice, but never for a proper subset holding chip (0,0,0) ("Skipping
  offloading collectives to SparseCore because the device assignment does not
  use all devices"): AOT on the v6e:16x16 topology and on the local v6e-4
  (libtpu 0.0.46) compiled the full-row-wide origin strips of the 'physical'
  layouts to synchronous all-gathers; TPU7x-8 behaves the same.  (On the
  v6e:16x16 topology an origin box narrower than a full 16-chip row did not
  compile at all, 2026-10.)  On the v6e-256 runs (PP=4 x
  FSDP=64, 'physical') that cost the first stage ~70 ms of TensorCore time per
  step.  A stage that is a whole slice -- each stage of `--layout slices` at
  one stage a slice, every one of which holds its slice's chip (0,0,0) -- is
  offloaded (at k stages a slice, the strip holding it is not): on a
  compile-only 2 x v5p-128 topology (libtpu 0.0.46, the v5p recipe's flags)
  slice 0's all-gathers compiled to SparseCore, those of the half of it
  holding (0,0,0) to synchronous ones.  (There the other half compiled
  synchronously too, where on one slice alone it is offloaded; never run, so
  not modelled here.)  A rule: Pipeline plans by libtpu's own answer where
  a stage's processes probed it (`offload_answers`), by this where none did
  -- ahead of time, off TPU -- or a probe failed.  A slice's devices are
  counted, by `slice_index`, among all the devices of the meshes' client --
  the job's (jax.devices()), or a compile-only topology's (ds_ep_aot) -- so
  a stage is a whole slice only if it holds every device the slice has,
  whatever the other stages hold; device objects of neither (a test's) count
  among the meshes' own.  A stage of one device has no FSDP collective to
  offload.  Decided from the devices alone (`on_origin_chip`), so every
  process of a multi-controller run agrees; a single stage over part of a
  slice, and other device sets libtpu does not offload (not a solid box),
  are not detected.

  Under Pathways (the devices the IFRT proxy's: `pathways.is_proxy_device`)
  none: the Pathways server's compiler offloads them on chip (0,0,0)'s sub-
  meshes too -- and so Pipeline plans for a Pathways run ahead of time
  (`proxy`: ds_ep_aot --pathways).  On local docker Pathways (tpu7x-8,
  images 20261117 and 20261124) the probe answered offloaded for all-gathers
  over fsdp, expert and both on the 2-chip half holding chip (0,0,0) --
  which McJAX's libtpu 0.0.50.1 keeps synchronous on the same chips -- and
  the trace showed them on SparseCore.  The probe still decides wherever it
  is asked.
  """
  devs = [d for m in meshes for d in m.devices.flat]
  if len(meshes) < 2 or not _is_tpu(devs[0]) or pathways.is_proxy_device(devs[0]):
    return frozenset()
  client = getattr(devs[0], "client", None)
  every = set(client.devices() if client is not None else jax.devices())
  pool = every if set(devs) <= every else set(devs)
  size = collections.Counter(map(slice_index, pool))

  def whole_slice(m):
    ks = {slice_index(d) for d in m.devices.flat}
    return len(ks) == 1 and m.devices.size == size[ks.pop()]

  return frozenset(
      s
      for s, m in enumerate(meshes)
      if m.devices.size > 1 and any(map(on_origin_chip, m.devices.flat)) and not whole_slice(m)
  )


def rule_offloads(meshes, s, axes, sync=None):
  """The device rule's answer to `offload_answers`' question: a TPU stage
  offloads an all-gather over `axes` (a mesh axis name or a tuple of them)
  unless `sync_collective_stages` lists it, and if one of the axes is wider
  than a device (`sync`: that set, if at hand)."""
  names = (axes,) if isinstance(axes, str) else tuple(axes)
  sync = sync_collective_stages(meshes) if sync is None else sync
  return _is_tpu(meshes[s].devices.flat[0]) and s not in sync and any(meshes[s].shape[a] > 1 for a in names)


def offload_answers(meshes, asks, kv=None, key="mpmd/offload", pid=None, timeout_s=900):
  """libtpu's answer to each ask (stage, axes) -- does it offload an all-gather
  over `axes` on that stage's mesh to SparseCore, where it runs async
  (`ds_ep.sparsecore_offload_available`: one probe compile)? -> {(stage,
  axes): (answer, why)}: why None for libtpu's answer, else why it is the
  device rule's (`rule_offloads`) instead.

  The processes of a stage probe its asks (every one of them, as they compile
  its programs).  Without `kv`, every stage's are this process's (one process,
  or ahead of time).  With it -- multi-controller JAX: the coordination
  service's key-value store -- each stage's lowest process publishes its
  answers under `key` and every process plans from those, its own stage's
  too, so that all of them build one plan from libtpu's answers rather than
  from the device rule, which a new topology or libtpu can break (a half of a
  compile-only v5p-128 slice broke it once).  A probe that fails answers by
  the rule, saying so: an advisory compile must not stop one stage's
  processes while the others go on and wait for them in the step."""
  sync = sync_collective_stages(meshes)

  def probe(s, axes):
    try:
      return bool(ds_ep.sparsecore_offload_available(meshes[s], axes)), None
    except Exception as e:  # what it is, in a line
      why = f'its libtpu probe failed ({type(e).__name__}: {" ".join(str(e).split())[:160]})'
      return rule_offloads(meshes, s, axes, sync), why

  if kv is None:  # one controller (or ahead of time): every stage's asks are this process's
    return {(s, axes): probe(s, axes) for s, axes in asks}
  # multi-controller JAX: each stage's processes probe its asks, its lowest one publishes the answers
  pid = jax.process_index() if pid is None else pid
  procs = [{d.process_index for d in m.devices.flat} for m in meshes]
  name = lambda s, axes: f'{key}/{s}/{axes if isinstance(axes, str) else ",".join(axes)}'
  out = {}
  for s, axes in asks:
    if pid not in procs[s]:
      continue
    out[(s, axes)] = probe(s, axes)
    if pid == min(procs[s]):
      got, why = out[(s, axes)]
      kv.key_value_set(name(s, axes), f'{int(got)} {why or ""}'.rstrip())
  for s, axes in asks:
    if pid != min(procs[s]):
      mine, v = out.get((s, axes)), kv.blocking_key_value_get(name(s, axes), timeout_s * 1000)
      out[(s, axes)] = (v[0] == "1", v[2:] or None)
      if mine is not None and mine != out[(s, axes)]:
        warnings.warn(
            f"stage {s}: this process answers {mine} to the libtpu probe over {axes}, its lead process "
            f"{min(procs[s])} {out[(s, axes)]}: every process plans by the lead's",
            RuntimeWarning,
        )
  return out


def offload_check(meshes, s, got, why=None):
  """Stage s's answer from `offload_answers` -- does libtpu offload an
  all-gather over the whole stage to SparseCore (`got`), and if it is the
  rule's, why -- against the device rule (`sync_collective_stages`) -> a log
  line, a warning where they disagree or the probe failed.

  main logs it for each stage whose answer was asked -- its FSDP
  collectives are lax ones, or, under batch split on Lineage's layers, its
  Regions trace Lineage's SparseCore regions by the answer -- on every
  layout: every process plans by the answer, so a rule gone wrong costs
  nothing, but says the rule needs mending."""
  rule = s not in sync_collective_stages(meshes)
  said = lambda x: "offloaded to SparseCore" if x else "synchronous"
  if why is not None:
    return (
        f"warning: stage {s}: {why}: every process plans its collectives {said(got)}, by the device rule "
        "(sync_collective_stages)"
    )
  if got == rule:
    return f"stage {s}: libtpu probe: its collectives {said(got)}, as the device rule (sync_collective_stages) " "has it"
  return (
      f"warning: stage {s}: libtpu probe: its collectives {said(got)}, where the device rule "
      f"(sync_collective_stages) has them {said(rule)}: every process plans by the probe"
  )


def shared_stage_hosts(meshes):
  """[(a, b, processes)] for every two stages whose devices share processes.
  Under multi-controller JAX every stage needs hosts of its own; compiling
  ahead of time for a topology checks the same on its devices."""
  owners = [frozenset(d.process_index for d in m.devices.flat) for m in meshes]
  return [(b, a, sorted(owners[a] & owners[b])) for a in range(len(meshes)) for b in range(a) if owners[a] & owners[b]]


def split_chip_stages(meshes):
  """[(stage, (slice, coords), held, of)] for each chip a stage holds part of the devices of -- `held` of its `of`
  TensorCores -- stage by stage, each stage's chips in mesh order: what Pathways refuses ("Devices use 1 of the 2
  logical devices on chip 0,0,0; a subslice must use all logical devices of each chip"), as it refused --layout
  slices at two stages a TPU7x slice before `chip_strips` (each stage one core of every chip).  A chip's devices are
  counted among all of the meshes' client's, as `sync_collective_stages` counts a slice's; devices without chip
  coordinates (CPU) have none to split."""
  devs = [d for m in meshes for d in m.devices.flat]
  if not devs or any(getattr(d, "coords", None) is None for d in devs):
    return []
  client = getattr(devs[0], "client", None)
  every = set(client.devices() if client is not None else jax.devices())
  pool = every if set(devs) <= every else set(devs)

  def chip(d):
    return slice_index(d), tuple(d.coords)

  size = collections.Counter(chip(d) for d in pool if getattr(d, "coords", None) is not None)
  out = []
  for s, m in enumerate(meshes):
    held = collections.Counter(map(chip, m.devices.flat))
    out += [(s, c, n, size[c]) for c, n in held.items() if n < size[c]]
  return out


def describe_mesh(mesh):
  rows = []
  for d in mesh.devices.flat:
    rows.append(f'{d.id}@{getattr(d, "coords", "?")}/c{getattr(d, "core_on_chip", "?")}')
  return f"{dict(mesh.shape)} " + " ".join(rows)


# ----------------------------------------------------------------- schedule
# The named schedules (--schedule).  build_schedule makes ORDER_SCHEDULES from
# the placement alone; Pipeline makes the rest itself: stash sizes its stash
# sets by building their programs, and interleaved (Megatron's per-stage
# order) hands out stash modes and pairs its steady state where asked.
ORDER_SCHEDULES = ("gpipe", "1f1b", "search", "timed", "breadth", "zbh1", "zbh2")
# The zero-bubble schedules (Qi et al., "Zero Bubble Pipeline Parallelism") on one chunk a stage, each middle chunk's
# backward split into I (the input gradient, with the recompute it needs) and W (the weight gradients, run later):
# ds_ep.ChunkPrograms.build_split.  ZB-H1 holds 1F1B's residual sets and ZB-H2 about twice them, each plus up to
# zb_wbuf_cap pending W buffers a stage -- memory 1F1B does not hold: a W buffer is a microbatch's crossing set and
# the arguments its W reads, about 3.6 GiB a device per DeepSeek-V3 MoE layer at 4096 tokens a device (7.25 GiB for
# a chunk of two; the summary's 'split backward' line gives a run's).  Each the fastest in simulation of its
# candidates (sched.zb_schedule) under the program costs.  Chunk 0's backward stays whole (it has no input gradient
# to split off), and so does the fused last chunk, unless zb_split_last (--zb-split-last on) splits it too: its F
# runs the forward, head, loss and input gradient (AI), a W of its own its layers' weight gradients, later
# (ds_ep.ChunkPrograms.build_split_last).  At 2 stages without it nothing splits, and zbh1 is 1F1B's order.
ZB_SCHEDULES = ("zbh1", "zbh2")
SCHEDULES = ORDER_SCHEDULES + ("stash", "interleaved")
# The schedule a Pipeline runs when given none (schedule None, no --schedule),
# on either placement: breadth, MaxText's order.  A stage runs a chunk's M
# microbatches back to back, so a microbatch's crossings overlap the other
# microbatches' forwards, and the stages wait on the boundary only at the fill
# and the drain (2 x v5p-128, relay over DCN: 5521 ms a step, not gpipe's
# 8799).  `search`, the default before, picks the same order on circular
# placement at two or more chunks a stage, or at a few small M (3 to 6) one
# with the same bubble and residual sets.  At one chunk a stage it picks 1F1B,
# which keeps S residual sets rather than M but waits on the boundary: breadth
# is 8% faster at 2 stages and M=8 with 55 ms crossings.  On V placement
# search's bubble is shorter (16% against breadth's 27% at 4 stages and M=8,
# F=1, B=2); --schedule search buys it back, under the step scope.
DEFAULT_SCHEDULE = "breadth"


def default_fsdp_scope(schedule, overlap):
  """The FSDP scope a Pipeline runs when given none (fsdp_scope None, no
  --fsdp-scope): 'repeat' under breadth, as MaxText's circular pipeline does
  with pipeline_fsdp_ag_per_repeat (a chunk's weights gathered for each
  visit, about two chunks' weights held at a time), where the fusion plan
  that repeat needs is on; 'step' otherwise."""
  return "repeat" if schedule == "breadth" and overlap else "step"


# build_schedule's per-program costs (ms) for 'timed' when given none: the
# TPU7x-8 calibration of the pipeline's first DeepSeek-V3 benchmark configuration (2026-09), never
# re-measured (not at 128 experts, not on other hardware).  Only their ratios
# matter.  F0 / B0 are chunk 0's forward / backward (they carry the
# embedding), A the last chunk's fused forward + backward, xfer one
# stage-boundary copy.  Pipeline passes its own table (`default_costs`: the
# v6e medians on a v6e, these elsewhere; `costs` / --costs override it).
TIMED_COSTS = sched.TPU7X_COSTS


def normalize_schedule(schedule):
  """A custom schedule -- any sequence of (chunk, kind, microbatch) triples,
  e.g. a list of lists from JSON -> a list of (int, str, int) tuples."""
  out = []
  for i, a in enumerate(schedule):
    try:
      c, k, m = a
    except (TypeError, ValueError):
      raise ValueError(f"schedule[{i}] = {a!r}: expected a (chunk, kind, microbatch) triple") from None
    if not all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in (c, m)):
      raise ValueError(f"schedule[{i}] = {a!r}: chunk and microbatch must be ints")
    if k not in sched.KINDS:
      raise ValueError(f"schedule[{i}] = {a!r}: kind must be 'F', 'B', 'I' or 'W'")
    out.append((int(c), str(k), int(m)))
  return out


def validate_schedule(order, n_chunks, n_micro, place=None, *, fuse=frozenset()):
  """Raise ValueError unless `order` runs F(c, m) and B(c, m) exactly once for
  every chunk c and microbatch m, each after what it consumes: F(c, m) after
  F(c-1, m); B(c, m) after F(c, m) and B(c+1, m).  The last chunk's F runs its
  fused forward + backward, so its B is a no-op -- but it must still appear,
  after that F.

  Then `sched.validate`, the rules the step loop relies on (a
  `sched.ScheduleError`, also a ValueError), so the two never disagree about
  an F / B order: every chunk's forwards and gradient writes in microbatch
  order, the last chunk's no-op B right after its F in its stage's sequence,
  and the F&B pairs `fuse` names (which need `place`).  Without `place` each
  chunk counts as its own stage: the per-stage rule is then checked per chunk
  only, which every order legal on the real placement also passes.

  An order with a split backward -- I(c, m) and then W(c, m) in place of a
  middle chunk's B(c, m), the zero-bubble schedules' -- is `sched.validate`'s
  alone, whose rules cover I and W: an I after F(c, m) and after the B or I
  of (c + 1, m), its W after it, none for chunk 0 or the fused last chunk,
  and every chunk's W's (its gradient writes) in microbatch order.
  """
  C, M = n_chunks, n_micro
  if any(k in ("I", "W") for _, k, _ in order):
    if fuse and place is None:
      raise ValueError("F&B pairs (fuse) run two ops of one stage: validate_schedule needs place")
    sched.validate(order, list(range(C)) if place is None else place, M, fuse=fuse)
    return
  seen = set()
  for i, (c, k, m) in enumerate(order):
    at = f"schedule[{i}] = ({c}, {k!r}, {m})"
    if k not in ("F", "B"):
      raise ValueError(f"{at}: kind must be 'F', 'B', 'I' or 'W'")
    if not (0 <= c < C and 0 <= m < M):
      raise ValueError(f"{at}: out of range (chunks 0..{C - 1}, microbatches 0..{M - 1})")
    if (c, k, m) in seen:
      raise ValueError(f"{at}: {k}({c},{m}) appears twice")
    if k == "F" and c > 0 and (c - 1, "F", m) not in seen:
      raise ValueError(f"{at}: F({c},{m}) before F({c - 1},{m}), whose activation it consumes")
    if k == "B" and (c, "F", m) not in seen:
      raise ValueError(f"{at}: B({c},{m}) before F({c},{m})")
    if k == "B" and c < C - 1 and (c + 1, "B", m) not in seen:
      raise ValueError(f"{at}: B({c},{m}) before B({c + 1},{m}), whose input gradient it consumes")
    seen.add((c, k, m))
  missing = [f"{k}({c},{m})" for c in range(C) for k in "FB" for m in range(M) if (c, k, m) not in seen]
  if missing:
    raise ValueError(
        f"schedule lacks {len(missing)} of its {2 * C * M} actions: "
        + ", ".join(missing[:6])
        + (" ..." if len(missing) > 6 else "")
    )
  if fuse and place is None:
    raise ValueError("F&B pairs (fuse) run two ops of one stage: validate_schedule needs place")
  sched.validate(order, list(range(C)) if place is None else place, M, fuse=fuse)


def build_schedule(schedule, n_chunks, n_micro, place, costs=None, *, zb_wbuf_cap=1, zb_split_last=False):
  """-> (order, name): a validated list of (chunk, kind, microbatch).

  `schedule` is one of ORDER_SCHEDULES or a custom sequence of triples (the
  other SCHEDULES are Pipeline's).  `costs`: the program costs of 'timed'
  and of the zero-bubble schedules (any `sched.cost_table` key; TIMED_COSTS
  fills in the rest, and sched.zb_costs a split chunk's I and W from its B
  where the table has neither).  `zb_wbuf_cap`: zbh1 / zbh2's pending W
  buffers a stage (None or 0: no cap); `zb_split_last`: theirs split the
  fused last chunk too (a W of its own a microbatch: sched.splits_last).
  zbh1 / zbh2 run on one chunk a stage: any other placement is a ValueError.
  """
  C = n_chunks
  if isinstance(schedule, (list, tuple)):
    order, name = normalize_schedule(schedule), "custom"
  elif schedule == "gpipe":
    order, name = sched.gpipe(C, n_micro, place), "gpipe"
  elif schedule == "1f1b":
    order, name = sched.one_f_one_b(C, n_micro, place), "1f1b"
  elif schedule == "breadth":
    order, name = sched.breadth_first(C, n_micro, place), "breadth"
  elif schedule == "search":
    res = sched.search(C, n_micro, place)
    order, name = res[0], f"search {res[1]}"
  elif schedule == "timed":
    # Event-driven search with per-program costs.  The last chunk is one
    # fused program.
    k = dict(TIMED_COSTS)
    k.update(costs or {})
    res = sched.timed_search(C, n_micro, place, sched.cost_table(k, C), k["xfer"], fuse_last=True)
    order, name = res[0], f"timed {res[1]} sim={res[2]:.1f}"
  elif schedule in ZB_SCHEDULES:
    if list(place) != list(range(C)):
      raise ValueError(
          f"schedule {schedule!r}: ZB-H1 / ZB-H2 run one chunk a stage (chunks_per_stage 1, --vv 1), not placement "
          f"{list(place)}; a custom order of I and W runs on any"
      )
    cap = None if zb_wbuf_cap in (None, 0) else zb_wbuf_cap
    k = sched.zb_costs({**TIMED_COSTS, **(costs or {})}, C, last=zb_split_last)
    order, tag, mk = sched.zb_schedule(
        C,
        n_micro,
        sched.cost_table(k, C),
        k["xfer"],
        mem="h1" if schedule == "zbh1" else "h2",
        wbuf_cap=cap,
        split_last=zb_split_last,
    )
    # the W-buffer cap where a chunk splits; at 2 stages none does (chunk 0 and the fused last chunk keep their
    # backwards whole), and no W buffer is held -- but the last chunk's, split too
    if all(k != "W" for _, k, _ in order):
      wb = " (no split chunk)"
    else:
      wb = (" split-last" if zb_split_last else "") + ("" if cap is None else f" wbuf<={cap}")
    name = f"{schedule} {' '.join(map(str, tag))}{wb} sim={mk:.1f}"
    # Every candidate splits (sched.zb_schedule): never slower than 1F1B where a split chunk's I + W is its B (and a
    # split last chunk's AI + W its A), as zb_costs prices them unless measured.  Measured, they can be more (TPU7x,
    # DeepSeek-V3 widths: 3.6-6.3%), an overhead each microbatch that at enough microbatches outgrows the bubble the
    # W's fill -- so say so where the costs say it.
    f1b = sched.timed_simulate(
        sched.one_f_one_b(C, n_micro, place), place, sched.cost_table(k, C), k["xfer"], fuse_last=True
    )[0]
    if mk > f1b * (1 + 1e-6):
      warnings.warn(
          f"schedule {schedule!r} ({name}): its microbatch loop simulates slower than 1F1B's ({mk:.1f} against "
          f"{f1b:.1f}) under these program costs, which price a split chunk's I + W above its B (or the split last "
          f"chunk's AI + W above its A) by more than the bubble its W's fill at {n_micro} microbatches repays: "
          "compare a --schedule 1f1b run",
          RuntimeWarning,
      )
  elif schedule in SCHEDULES:
    raise ValueError(
        f"schedule {schedule!r} is built by Pipeline, not build_schedule: Pipeline sizes "
        "its stash sets (stash) or merges its per-stage streams (interleaved)"
    )
  else:
    raise ValueError(
        f"unknown schedule {schedule!r}: expected one of {SCHEDULES} or a sequence of "
        "(chunk, kind, microbatch) triples"
    )
  order = normalize_schedule(order)
  validate_schedule(order, C, n_micro, place)
  return order, name


def zb_refusals(what, scfg, fsdp_scope, *, order=None, modes=None, fuse=()):
  """Raise ValueError, naming every problem at once, where the split backward of `what` -- zbh1 / zbh2, or a custom
  `order` with I and W ops (with its `modes` and `fuse`) -- cannot run: batch split on Lineage's layers, one custom
  VJP of Lineage's whole region, with no input-gradient half to cut (ds_ep.ChunkPrograms.build_split); fsdp_scope
  'repeat', which gathers a chunk's weights for each visit of its stage, where a chunk's W's, which read none, would
  make visits of their own; a stash mode on a split chunk, whose backward the split does not cut (the recompute one
  only); and an F&B pair whose backward is an I, or whose forward is a split last chunk's AI (no program pairs
  either)."""
  bad = []
  if scfg.lineage_split:
    bad.append(
        "batch_split on Lineage's layers (batch_split_impl 'lineage'): one custom VJP of Lineage's whole region, "
        "with no input-gradient half to cut"
    )
  if fsdp_scope == "repeat":
    bad.append(
        "fsdp_scope 'repeat' (a chunk's weights gathered for each visit): a chunk's W's, which read no weights, would "
        "make visits of their own, each one gathered for nothing (fsdp_scope 'step')"
    )
  if order is not None:
    split = {c for c, k, _ in order if k in ("I", "W")}
    held = sorted(tuple(key) for key, v in (modes or {}).items() if v != "r" and tuple(key)[0] in split)
    if held:
      bad.append(
          f"stash modes on split chunks {held[:4]}{' ...' if len(held) > 4 else ''}: the split cuts the recompute "
          "backward, not a stash one"
      )
    pairs = sorted(j for j in fuse if order[j + 1][1] == "I")
    if pairs:
      names = ", ".join("&".join(f"{c}{k}{m}" for c, k, m in order[j : j + 2]) for j in pairs[:3])
      bad.append(f"F&B pairs of a forward and an I ({names}{', ...' if len(pairs) > 3 else ''}): none is built")
    C = 1 + max(c for c, _, _ in order)
    if sched.splits_last(order, C) and (ai := sorted(j for j in fuse if order[j][0] == C - 1)):
      names = ", ".join("&".join(f"{c}{k}{m}" for c, k, m in order[j : j + 2]) for j in ai[:3])
      bad.append(
          f"F&B pairs of the split last chunk's forward (AI: its forward, head, loss and input gradient) and a "
          f"backward ({names}{', ...' if len(ai) > 3 else ''}): none is built"
      )
  if bad:
    raise ValueError(f"{what}: the split backward (I / W) cannot run this: " + "; ".join(bad))


# ----------------------------------------------------------------- pipeline
@dataclasses.dataclass(frozen=True)
class Op:
  """One entry of the order as the step loop runs it: the program to call
  (`desc`, a tuple of `ds_ep.pdesc` parts, built by `progs[c].program`; None
  for the fused last chunk's no-op backward) and what the loop does around
  it.  `acc`: accumulate into the chunk's gradient buffer (every gacc writer
  but the chunk's first) -- on an I, the split backward's input-gradient
  half, the variant of the W it feeds (one I program feeds either, unless
  in-place dW cut another: ds_ep.Split.i_acc); on a split last chunk's F
  (AI) and W, into the part each writes, every one but its first; `stash`:
  the (c, m) mode's policy, None for remat; `self_pf`, `pf`, `rs`,
  `rs_self`: the FSDP collectives the overlap plan folds into this program
  (`pf` rides in a forward; under
  fsdp_scope 'repeat' also in a backward or the fused last chunk; `rs` and
  `rs_self` in a gradient writer, a B or a W); `pin`, its prefetch pins
  (`Pipeline._plan_pins`).  `pair`, on the first op of an F&B pair (j in
  `Pipeline.fuse`): the descriptor of the ONE program both run -- this op's
  part, then the backward's with `peer=` its chunk's programs; the second op
  keeps its own `desc`, which the loop does not call."""

  j: int
  c: int
  kind: str
  m: int
  s: int
  desc: tuple = None
  acc: bool = False
  stash: str = None
  self_pf: bool = False
  pf: tuple = None  # (chunk, leaf ids) gathered here (Pipeline.pf_n[j]: in how many ops in all)
  rs: tuple = None  # (chunk, leaf ids) reduce-scattered here
  rs_self: int = None  # this chunk's own reduce-scatter, summing that many partial sumsqs
  pair: tuple = None  # the F&B pair's composed descriptor (on its F)
  pin: tuple = None  # ((point, layer, chunk, leaf id), ...): the gathers this forward pins


class _Box(dict):
  """The act / cotangents storage of a step: what a boundary copy files (`Pipeline._file`), popped by the op that
  consumes it."""


class _RelayBox(_Box):
  """_Box under multi-controller JAX: its `pop` resolves a relay receive (transfer.resolve), so a boundary's bytes are
  waited for by the op that consumes them, not by the copy that filed them."""

  def pop(self, k):
    return transfer.resolve(super().pop(k))


class _Step:
  """What one step's microbatch loop carries from one program to the next."""

  def __init__(self, p, params, wc, batch):
    S, C = p.S, p.C
    self.toks, self.tgts = batch[:2]
    self.mtp_toks = batch[2] if p.mtp else None
    self.params, self.wc = params, wc
    self.res, self.stash = {}, {}
    # A split backward's W buffers, from its I to its W: {(c, m): (the crossing set, the I arguments W reads)}.
    self.wbuf = {}
    # Under Pipeline.pace_i, each split chunk's latest W buffer this step, the crossing set its next I (AI) waits on.
    self.last_i = {}
    # A split last chunk's accumulator, until its last W: [its F's (AI's) part, its W's part], each a tuple of leaves.
    self.gacc_parts = {}
    # act / cotangents cross stages: under multi-controller JAX a relay receive filed there is resolved when it is
    # popped.
    box = _RelayBox if p.multi else _Box
    self.act, self.cotangents = box(), box()
    self.gacc = [None] * C
    self.lacc = None
    # A chunk's gathered leaves as its prefetch hosts deliver them, and how
    # many hosts that gather still waits for (`Pipeline._install`); one
    # gather of a chunk at a time, so both are keyed by chunk.
    self.wparts, self.pending = {}, {}
    self.gparts, self.gpending = {}, dict(p.rs_groups)  # reduce-scatter groups still to come
    self.gacc_leaves = {}
    self.grads = [[None] * len(p.stage_chunks[s]) for s in range(S)]
    self.sq = [[] for _ in range(S)]
    self.stage_tot = [None] * S


class Pipeline:

  def __init__(
      self,
      scfg: ds_ep.StageConfig,
      n_stages=2,
      chunks_per_stage=4,
      n_micro=8,
      global_batch=32,
      seq=4096,
      schedule=None,
      meshes=None,
      layout="ep_on_chip",
      overlap=True,
      pf_split=4,
      rs_split=4,
      rs_self=True,
      costs=None,
      self_prefetch=True,
      placement="circular",
      stash="none",
      stash_budget_gib=4.5,
      stash_caps=None,
      init_by_chunk=False,
      modes=None,
      rs_reserve=None,
      pair_stages=None,
      pair_gain=None,
      fuse=None,
      stage_ep_collectives=None,
      aot=False,
      transport="native",
      layer_split=None,
      cp=None,
      fsdp_scope=None,
      prefetch_pin="off",
      offload_residuals=False,
      norm_reduce="hops",
      interleave_group=None,
      stage_lineage_sparsecore=None,
      direct_experts=False,
      zb_wbuf_cap=1,
      zb_split_last=False,
      proxy=None,
  ):
    """`placement`: 'circular' or 'v' (chunks_per_stage 2).  `schedule`: one
    of SCHEDULES, a custom order, or None for DEFAULT_SCHEDULE (breadth).
    `stash`: the
    policy `--schedule stash` gives the (chunk, microbatch)s it stashes, as
    many at once as `stash_budget_gib` of live stash sets per stage allows
    (see schedule.py's memory note: the stash backwards' extra temporaries
    come on top, and the 4.5 GiB default is a placeholder until a TPU
    `--mem-report` sweep sets it), optionally under `stash_caps`
    live sets per chunk -- or, under `interleaved`, the stash modes of its
    fixed order under the same budget and caps (`sched.stash_modes`; there a
    chunk left off `stash_caps` is uncapped, where `stash` never stashes it).  `modes` ({(c, m): policy}) and `fuse` (the
    order indices j whose op runs with order[j+1] as one F&B program) ride
    along with a custom order (`schedule` as a list).  `init_by_chunk`:
    `init` seeds each chunk's weights from its index alone, so they do not
    depend on the placement.  `rs_reserve`: let the reduce-scatter plan keep
    each stage's last gradient write free for rs_self (`_plan_overlap`);
    None = under V and for `interleaved` (no earlier plan to keep), so circular
    breadth / search / timed / stash keep the plans they have always made.
    `pair_stages`: the stages `interleaved` pairs on -- 'all' every stage, ()
    none, else the stages listed; None, its default, none: its order stays
    Megatron's own unless its pairs are asked for
    (`sched.megatron_interleaved`).  No other schedule takes any (a custom
    order names its pairs in `fuse`);
    `pair_gain`: what a pair saves over its two programs run back to back
    (ms; None: the cost table's 'pair_gain', else 0), which the simulation
    prices.  `stage_ep_collectives`: one of
    'xla' / 'ppermute' per stage, overriding scfg.ep_collectives
    (`--ep-collectives ppermute,xla`).  `stage_lineage_sparsecore`: under
    batch split on Lineage's layers, whether each stage runs Lineage's
    SparseCore regions (a bool per stage), overriding libtpu's answer (ds_ep_aot
    hands the pipeline over the real slices the answers of the one it compiles
    on the first slice's devices).  `rs_self`: which stages may fold a
    chunk's reduce-scatter into its own last backward -- 'all' (True), 'none'
    (False) or 'first': only the stage holding chunk 0, whose last backward
    ends the step; elsewhere a program's outputs leave only when it ends, so
    the folded reduce-scatter holds the cotangent the stage upstream waits
    for, and a standalone reduce after it does not (`_plan_overlap`).  Under
    'first' the other stages' last backward hosts no reduce-scatter at all,
    another chunk's neither: those groups move to earlier backwards or to the
    standalone reduce.
    A stage whose collectives run synchronously (`sync_collective_stages`),
    other than chunk 0's, reduce-scatters every chunk in one program after
    its last backward, where it idles in the drain, not in its backwards.
    `fsdp_scope`: how long a chunk's gathered (compute) weights live.
    'step': from their gather to the end of the microbatch loop, every chunk
    of the stage at once by the backward phase.  'repeat': one visit -- a run
    of the chunk's ops in its stage's sequence (`_plan_repeat`) -- at a time:
    gathered inside the stage's previous visit, released after the visit's
    last op, gathered again for the next one, so a stage holds about two
    chunks' weights instead of all of them, for (V-1)/V more of a step's
    weight all-gather under a breadth-first order (MaxText's circular
    pipeline with `pipeline_fsdp_ag_per_repeat`).  It needs the fusion plan
    (`overlap`), no F&B pairs, and an order that visits each chunk at most
    twice a step (`schedule='breadth'`).  None: `default_fsdp_scope`,
    'repeat' under breadth with the fusion plan, else 'step'.
    `prefetch_pin`: 'off', 'auto' or a SPEC (ds_ep.parse_prefetch_pin): which
    gathers a gathering forward -- a prefetch host, the self-prefetch -- ties
    to which points of its layers (`_plan_pins`).  Under batch split on
    Lineage's layers (scfg.lineage_split) there are no stash modes, no stash
    scheduler and no pins: refused here, before anything is built.

    The split backward: the zero-bubble schedules (ZB_SCHEDULES, at
    chunks_per_stage 1), or a custom order of I and W ops at any, run each
    middle chunk's backward the order names so as I -- the input gradient,
    which the stage upstream waits on -- and later W, the weight gradients
    (ds_ep.ChunkPrograms.build_split; `split_chunks`): the W's are the
    chunk's gradient writers, in microbatch order, so the gradients are the
    B's bit for bit wherever XLA compiles the halves' operations as it does
    the fused backward's.  `zb_wbuf_cap`: zbh1 / zbh2's pending W buffers a
    stage (None or 0: no cap), each one microbatch's crossing set and the
    arguments its W reads (`wbuf_bytes`).  `zb_split_last`: zbh1 / zbh2 split
    the fused last chunk too (a custom order does where it gives the last
    chunk W's): its F runs the forward, the head, the loss and the input
    gradient (AI) and leaves a W buffer, its W its layers' weight gradients
    (ds_ep.ChunkPrograms.build_split_last) -- its accumulator two parts, AI's
    leaves and W's, merged after its last W for the stage's reduce.  Not under
    batch split on Lineage's layers, fsdp_scope 'repeat', a stash mode on a
    split chunk or an F&B pair holding an I or a split last chunk's AI
    (`zb_refusals`), nor zb_split_last on a last chunk without layers: refused
    before anything is built.

    Under multi-controller JAX (`jax.distributed`, one process per host) a
    process runs the stages that hold its devices (`local`): it compiles and
    dispatches only their programs, and walks the whole order so that it
    takes part in every cross-stage copy, in order (`_send`).  `multi` says
    so (pathways.multi_controller(), the one rule: jax.process_count() > 1):
    every piece of that machinery runs under `multi` only.  In one
    controller -- a Pathways client, which drives every device, or one host's
    own process -- every stage is local and none of it runs.  `aot`: the
    meshes are of a topology this process does not own -- every program can
    be lowered and compiled (`precompile`), nothing is placed or run.
    `proxy`: plan as one Pathways client driving the stages (None: whether
    the meshes' devices are the IFRT proxy's, `on_pathways`) -- no stage's
    collectives synchronous by the device rule (`sync_collective_stages`'
    answer for Pathways' devices: its servers' compiler offloads them on chip
    (0,0,0)'s sub-meshes too); ahead of time, where libtpu's probe would
    answer as a TPU VM's own (a compile-only topology's), every ask answered
    by that rule, unprobed (ds_ep_aot --pathways).
    `transport`: how a boundary copy moves -- 'native', jax's own (a batched
    device-to-device copy in one process, its cross-host copy across
    processes: transfer.Channel / CrossProcessChannel); 'ppermute', a
    collective-permute program over the two stages' devices
    (transfer.PermuteChannel), which the two stages run together; 'xfer',
    jax.experimental.transfer pulls between host pairs (transfer.XferChannel,
    asynchronous, over DCN; one process per host only); or 'relay', a host
    relay (transfer.RelayChannel: D2H, TCP between the hosts, H2D, in
    threads of the context `relay_ctx`; asynchronous, one process per host
    only), whose receive is a future the consuming op resolves (`_RelayBox`,
    `step`); `relay_ctx.timing()` says where its time went and
    `close()` shuts it down.

    A custom order is normalized to (int, 'F' | 'B' | 'I' | 'W', int) tuples
    and checked (`validate_schedule`) before anything is built; every order,
    named or custom, then passes `sched.validate` before the plan is made."""

    def replace(c, **changes):
      # dataclasses.replace re-runs StageConfig.__post_init__: do not repeat
      # the padded-tiling warning `scfg` already gave.
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=ds_ep.TILING_WARNING)
        return dataclasses.replace(c, **changes)

    if cp is not None and cp != scfg.cp:  # given: it overrides scfg.cp (build sets scfg.cp, not this)
      scfg = replace(scfg, cp=cp)
    cfg = scfg.model
    self.scfg, self.S, self.V, self.M = scfg, n_stages, chunks_per_stage, n_micro
    self.overlap, self.pf_split, self.rs_split = overlap, pf_split, rs_split
    rs_self = {True: "all", False: "none"}.get(rs_self, rs_self)
    if rs_self not in ("all", "first", "none"):
      raise ValueError(f"rs_self must be 'all', 'first' or 'none' (or a bool), got {rs_self!r}")
    self.rs_self_enabled, self.rs_self_policy = rs_self != "none", rs_self
    if fsdp_scope not in (None, "step", "repeat"):
      raise ValueError(f"fsdp_scope must be 'step' or 'repeat', got {fsdp_scope!r}")
    if fsdp_scope == "repeat" and not overlap:
      raise ValueError(
          "fsdp_scope 'repeat' gathers and releases a chunk's weights inside the per-microbatch "
          "programs: it needs the fusion plan (overlap=True; not --no-overlap)"
      )
    if scfg.lineage_split:
      # What a chunk on Lineage's layers has no program for (ds_ep.ChunkPrograms refuses each when it is built):
      # refused here, before anything is.
      bad = []
      what = (
          ([f"stash {stash!r}"] if stash != "none" else [])
          + (["modes"] if modes else [])
          + (["schedule 'stash'"] if isinstance(schedule, str) and schedule == "stash" else [])
      )
      if what:
        bad.append(
            f"{' / '.join(what)}: a chunk on Lineage's layers runs them in one custom VJP whose residuals are "
            "Lineage's own, which no stash policy names: no stash modes (and no stash scheduler: 'search' or "
            "'timed' search the order)"
        )
      if ds_ep.parse_prefetch_pin(prefetch_pin) != "off":
        bad.append(
            f"prefetch_pin {prefetch_pin!r}: pins tie gathers to points of ds_ep's own layers (hn / xs / y), "
            "which Lineage's layers have none of: 'off'"
        )
      if bad:
        raise ValueError(
            "batch_split on Lineage's layers (batch_split_impl 'lineage') cannot run this: " + "; ".join(bad)
        )
    # The zero-bubble schedules: one chunk a stage, and what no split backward runs under, refused before anything
    # is built (a custom order of I and W ops once it is normalized, below).
    if isinstance(schedule, str) and schedule in ZB_SCHEDULES:
      if chunks_per_stage != 1:
        raise ValueError(
            f"schedule {schedule!r}: ZB-H1 / ZB-H2 run one chunk a stage (chunks_per_stage 1, --vv 1), not "
            f"{chunks_per_stage}; a custom order of I and W ops runs at any"
        )
      zb_refusals(f"schedule {schedule!r}", scfg, fsdp_scope)
    if zb_wbuf_cap is not None and (isinstance(zb_wbuf_cap, bool) or int(zb_wbuf_cap) != zb_wbuf_cap or zb_wbuf_cap < 0):
      raise ValueError(f"zb_wbuf_cap must be None or an int >= 0 (0 or None: no cap), got {zb_wbuf_cap!r}")
    # zbh1 / zbh2: the W buffers a stage may hold pending -- each a split chunk's crossing set and the arguments its W
    # reads, for one microbatch (`wbuf_bytes`) -- None for no cap
    self.zb_wbuf_cap = None if zb_wbuf_cap in (None, 0) else int(zb_wbuf_cap)
    if zb_split_last not in (True, False):
      raise ValueError(f"zb_split_last must be a bool, got {zb_split_last!r}")
    if zb_split_last and not (isinstance(schedule, str) and schedule in ZB_SCHEDULES):
      raise ValueError(
          f"zb_split_last splits the fused last chunk of zbh1 / zbh2's orders, not of schedule {schedule!r} (a custom "
          "order splits it by giving it W ops of its own)"
      )
    self.zb_split_last = bool(zb_split_last)
    C = n_stages * chunks_per_stage
    # The model's first_num_dense_layers dense layers run first (ds_ep.ChunkSpec.dense).  `layer_split` splits the
    # MoE layers, the dense ones all in chunk 0 -- or, given as every decoder layer a chunk (its sum
    # num_decoder_layers, which the MoE layers' never is while there are dense layers), all the layers in order,
    # the dense ones spanning the first chunks.
    self.n_dense = getattr(cfg, "first_num_dense_layers", 0)  # a Qwen3.5-family config has none
    L = cfg.num_decoder_layers - self.n_dense
    self.every_layer = bool(self.n_dense and layer_split is not None and sum(layer_split) == cfg.num_decoder_layers)
    # The first and the last chunk may hold no layer (a layer split's 0 there): the embedding's alone, the head's
    # alone -- a stage of its own for a head that would make its stage the pipeline's slowest (Qwen3.8's 248320-word
    # head, a third of a two-layer stage's time).
    ends = (
        0
        if layer_split is None or not len(layer_split)
        else sum(1 for i in {0, len(layer_split) - 1} if tuple(layer_split)[i] == 0)
    )
    if self.n_dense < 0 or L < 0 or (not self.every_layer and L < C - ends):
      raise ValueError(
          f"{cfg.num_decoder_layers} layers, {self.n_dense} of them dense: {C} chunks (stages x "
          "chunks_per_stage) need >= 1 MoE layer each -- the first and the last may hold none under a "
          "layer split with a 0 there (or one that counts every layer)"
      )
    if self.n_dense and (stash != "none" or modes):
      # every chunk but the last builds the policy's programs, and dense layers have none
      raise ValueError(
          f"{self.n_dense} dense layers: the stash programs take MoE layers only, so no stash policy "
          "with dense layers (--stash none)"
      )
    self.mtp = getattr(cfg, "mtp_num_layers", 0)  # a Qwen3.5-family config has none
    if self.mtp not in (0, 1):
      raise ValueError(f"mtp_num_layers {self.mtp}: the pipeline runs 0 or 1 MTP layer")
    if self.mtp and scfg.cp > 1:
      raise ValueError("MTP shifts tokens along the sequence: not under context parallelism (--cp 1)")
    if layer_split is None:
      if L % C:
        raise ValueError(f"{L} layers do not split into {C} chunks")
      layer_split = (L // C,) * C
    else:
      layer_split = tuple(layer_split)
      if (
          len(layer_split) != C
          or sum(layer_split) != (cfg.num_decoder_layers if self.every_layer else L)
          or not all(
              isinstance(n, numbers.Integral) and not isinstance(n, bool) and n >= (0 if i in (0, C - 1) else 1)
              for i, n in enumerate(layer_split)
          )
      ):
        raise ValueError(
            f"layer_split {layer_split}: want {C} chunks (stages x chunks_per_stage) of >= 1 layer each "
            "(the first and the last may hold none: the embedding's alone -- with the dense layers, "
            "if any -- and the head's alone), "
            f"{L} layers in all"
            + (
                f" (the MoE layers, the dense ones in chunk 0; or "
                f"{cfg.num_decoder_layers}, every layer, the "
                f"{self.n_dense} dense ones first)"
                if self.n_dense
                else ""
            )
        )
    # Layers per chunk, in chunk order: uneven where the head's chunk (or the
    # embedding's) should hold fewer, to even out the stages.
    self.layer_split = layer_split
    if global_batch % n_micro:
      raise ValueError("global_batch must divide into n_micro microbatches")
    self.mbs, self.seq, self.gbs = global_batch // n_micro, seq, global_batch
    # Context parallelism (scfg.cp): a microbatch of mbs sequences runs as
    # rows = cp x mbs rows of seq / cp tokens, chunk-major per sequence (row
    # cp x i + r is chunk r of sequence i), exactly one row a stage device;
    # every activation and boundary copy is [rows, row_len, emb].
    self.cp = scfg.cp
    if seq % self.cp:
      raise ValueError(f"cp={self.cp} does not split seq={seq} into whole chunks")
    self.rows, self.row_len = self.mbs * self.cp, seq // self.cp
    if self.cp > 1 and self.rows != scfg.fsdp * scfg.ep:
      raise ValueError(
          f"cp={self.cp}: a microbatch of {self.mbs} sequences is {self.rows} rows; context "
          f"parallelism wants exactly one a stage device (fsdp*ep={scfg.fsdp * scfg.ep}), so "
          f"{scfg.fsdp * scfg.ep // self.cp} sequences a microbatch"
      )
    if self.mbs % (scfg.fsdp * scfg.ep) and self.cp == 1:
      raise ValueError(f"microbatch {self.mbs} must divide by fsdp*ep={scfg.fsdp * scfg.ep}")
    self.C = C
    self.placement = placement
    self.place = sched.placement(n_stages, chunks_per_stage, placement)
    self.init_by_chunk = init_by_chunk
    if schedule is None:
      schedule = DEFAULT_SCHEDULE
    self.schedule = schedule if isinstance(schedule, str) else "custom"  # what run records name
    if fsdp_scope is None:
      fsdp_scope = default_fsdp_scope(schedule, overlap)
    self.fsdp_scope = fsdp_scope
    # Residuals in host memory: each (chunk, microbatch)'s forward residuals
    # (its input, and the first chunk's layer inputs) go to the same sharding
    # in pinned host memory right after the forward and come back right
    # before its backward (`_put_F` / `_take_B`), so the HBM a step holds no
    # longer grows with the microbatches a schedule keeps in flight.  On a
    # GB200 the host is the Grace CPU behind NVLink-C2C: a 134 MB residual
    # (Qwen3.8, 2 rows of 4096 x 8192 bf16) moves in ~0.3 ms each way.
    self.offload_residuals = bool(offload_residuals)
    # How the stages share their gradient sums of squares for the global-norm
    # clip: 'hops', a scalar copy from every stage to every other one
    # (S(S-1) a step, each a channel of its own -- on GPU a NCCL communicator
    # of its own the first time: 2070 at 46 stages), or 'allreduce', one sum
    # over every device of the job in one program (`_global_sumsq`), whose
    # total each stage's update takes in place of the S partials -- under
    # multi-controller JAX: in one controller every stage's sum is this
    # process's, and the hops run.
    if norm_reduce not in ("hops", "allreduce"):
      raise ValueError(f"norm_reduce must be 'hops' or 'allreduce', got {norm_reduce!r}")
    self.norm_reduce = norm_reduce
    if interleave_group is not None and (not isinstance(schedule, str) or schedule != "interleaved"):
      raise ValueError(
          f"interleave_group {interleave_group} is the interleaved schedule's microbatch group, not "
          f"schedule {schedule!r}'s"
      )
    if isinstance(schedule, (list, tuple)):  # fail before anything is built
      # e.g. a list of lists from JSON -> (int, str, int) tuples: the plan
      # looks ops up by value (`order.index((0, 'F', 0))`)
      schedule = normalize_schedule(schedule)
      validate_schedule(schedule, C, n_micro, self.place, fuse=frozenset(fuse or ()))
      if any(k in ("I", "W") for _, k, _ in schedule):
        zb_refusals(
            "a custom order of I and W ops", scfg, fsdp_scope, order=schedule, modes=modes, fuse=frozenset(fuse or ())
        )
    first_layer = np.cumsum((0,) + layer_split)
    if self.every_layer:  # chunk c holds layers first_layer[c].. in order, the dense ones first
      spans = [range(first_layer[c], first_layer[c + 1]) for c in range(C)]
      dense = [sum(1 for l in r if l < self.n_dense) for r in spans]
      moe = [tuple(l - self.n_dense for l in r if l >= self.n_dense) for r in spans]
    else:  # the MoE layers split, the dense ones in chunk 0
      dense = [self.n_dense if c == 0 else 0 for c in range(C)]
      moe = [tuple(range(first_layer[c], first_layer[c + 1])) for c in range(C)]
    self.chunks = [
        ds_ep.ChunkSpec(
            index=c,
            layers=moe[c],
            first=(c == 0),
            last=(c == C - 1),
            stage=self.place[c],
            dense=dense[c],
            routed_bias=getattr(cfg, "routed_bias", False),
            mtp=self.mtp if c == C - 1 else 0,
            kinds=ds_ep.layer_kinds(cfg, moe[c], scfg.layer_tree == "per_layer"),
            dense_kinds=ds_ep.dense_layer_kinds(dense[c], scfg.layer_tree == "per_layer"),
        )
        for c in range(C)
    ]
    if self.zb_split_last and not (self.chunks[-1].layers or self.chunks[-1].dense):
      raise ValueError(
          f"zb_split_last: the last chunk holds no layer (layer_split {list(layer_split)}): its weight gradients, the "
          "head's and the decoder norm's, stay in its F (AI), with nothing for a W to defer"
      )
    if ds_ep.is_qwen(cfg) and (stash != "none" or modes):
      # STASH_POLICIES name MLA's residuals, and the stash costs (sched.STASH_BWD_TEMP) are DeepSeek's.  The stash
      # schedule without a policy (--stash none) is a remat-only order search, on the costs --schedule timed uses.
      raise ValueError(
          "stash modes are not implemented for the Qwen3.5 family: its layers recompute in the " "backward (--stash none)"
      )
    # Prefetch pins (`_plan_pins`, after the fusion plan), checked here, before anything is built: every
    # name of a SPEC must select a leaf of this model, and give every leaf one point (ds_ep.pin_of), whichever
    # leaves the plan's forwards gather.
    self.prefetch_pin = ds_ep.parse_prefetch_pin(prefetch_pin)
    if ds_ep.is_qwen(cfg) and self.prefetch_pin == "auto":
      raise ValueError(
          "prefetch_pin 'auto' plans from MLA's FLOPs and DeepSeek's measured window rates "
          "(ds_ep.pin_windows); for the Qwen3.5 family pin by SPEC, or not at all (off)"
      )
    self.leaf_names = [ds_ep.leaf_names(ch) for ch in self.chunks]
    if isinstance(self.prefetch_pin, tuple):
      bad = [
          n
          for _, _, n in self.prefetch_pin
          if not any(ds_ep.pin_selects(n, m, c) for c, names in enumerate(self.leaf_names) for m in names)
      ]
      if bad:
        every = sorted({m for names in self.leaf_names for m in names})
        raise ValueError(
            f"--prefetch-pin: {bad} select(s) no weight of this model: its leaves are {every}, each "
            f"optionally prefixed with its chunk (c0. .. c{C - 1}.)"
        )
      for c, names in enumerate(self.leaf_names):
        for m in names:
          ds_ep.pin_of(self.prefetch_pin, m, c)
    self.meshes = typed_meshes(meshes or stage_meshes(n_stages, scfg.fsdp, scfg.ep, layout=layout), scfg.shard_mode)
    # The stages this process compiles and dispatches (all of them in one
    # process, and when compiling ahead of time for a topology), and the ones
    # it places arrays on (none ahead of time).
    self.aot = bool(aot)
    # Multi-controller JAX (pathways.multi_controller: never under Pathways, nor in one host's own process): the one
    # switch for what only several processes need -- the global-order walk, the key-value store, the barriers, the
    # cross-process copies.
    self.multi = pathways.multi_controller() and not self.aot
    if transport not in ("native", "ppermute", "xfer", "relay"):
      raise ValueError(f"transport {transport!r}: want 'native', 'ppermute', 'xfer' or 'relay'")
    if transport == "xfer" and not (self.multi or self.aot):
      raise ValueError(
          "transport 'xfer' (jax.experimental.transfer between hosts) needs one process per " "host (--distributed)"
      )
    if transport == "relay" and not (self.multi or self.aot):
      raise ValueError("transport 'relay' (a host relay between processes) needs one process per host " "(--distributed)")
    self.transport = transport
    if self.multi:
      for b, a, procs in shared_stage_hosts(self.meshes)[:1]:
        raise ValueError(
            f"stages {b} and {a} share processes {procs}: under multi-controller JAX every "
            "stage needs hosts of its own (a boundary copy between stages of one process set "
            "is a reshard jax refuses)"
        )
      # The processes holding each stage's devices.  A process may hold no stage (a pipeline over part of the
      # slice): it still walks every copy of the order (`_do_remote`), and runs nothing.
      pid = jax.process_index()
      self.stage_procs = [frozenset(d.process_index for d in m.devices.flat) for m in self.meshes]
      self.local = frozenset(s for s in range(n_stages) if pid in self.stage_procs[s])
      self._gsq = None  # norm_reduce 'allreduce': `_global_sumsq`'s program, built at its first step
    else:  # one controller (or ahead of time): this process, process 0, holds every stage
      self.stage_procs = [frozenset({0})] * n_stages
      self.local = frozenset(range(n_stages))
    self.placed = frozenset() if self.aot else self.local
    self.stage_chunks = [[ch for ch in self.chunks if ch.stage == s] for s in range(n_stages)]
    self.slot = {ch.index: self.stage_chunks[ch.stage].index(ch) for ch in self.chunks}
    denom = float(global_batch * seq)
    # Per-stage collective flavours.  libtpu offloads all-gather / reduce-
    # scatter to SparseCore (async) on some sub-meshes and refuses on others
    # (any part of a slice containing chip (0,0,0), but not the whole slice);
    # `offload_answers` asks it, over the axes the collective in question
    # spans: each stage's processes probe, and across processes its lowest
    # one's answers are every process's.  `stage_ep_collectives` (one of
    # 'xla' / 'ppermute' per stage) overrides scfg.ep_collectives, e.g. to
    # force the TPU7x benchmark runs' ['ppermute', 'xla'] whatever the probe answers.
    if stage_ep_collectives is not None:
      stage_ep_collectives = tuple(stage_ep_collectives)
      if len(stage_ep_collectives) != n_stages or not set(stage_ep_collectives) <= {"xla", "ppermute"}:
        raise ValueError(
            f"stage_ep_collectives: one of 'xla' / 'ppermute' per stage ({n_stages}), got " f"{stage_ep_collectives!r}"
        )
    if stage_lineage_sparsecore is not None:
      stage_lineage_sparsecore = tuple(stage_lineage_sparsecore)
      if len(stage_lineage_sparsecore) != n_stages or not all(isinstance(v, bool) for v in stage_lineage_sparsecore):
        raise ValueError(f"stage_lineage_sparsecore: a bool per stage ({n_stages}), got {stage_lineage_sparsecore!r}")
    # The asks, the same on every process (the flags decide them): EP's under
    # 'auto' at EP 2 (elsewhere the permute flavour does not apply; nor on
    # Lineage's layers, which run EP collectives of their own), and, but ahead
    # of time, FSDP's on the TPU stages of more than one device under 'xla':
    # which run them synchronously (sync_stages).
    multi_device_tpu = lambda s: _is_tpu(self.meshes[s].devices.flat[0]) and self.meshes[s].devices.size > 1
    asks = [
        (s, ds_ep.EP_AXIS)
        for s in range(n_stages)
        if stage_ep_collectives is None and scfg.ep_collectives == "auto" and scfg.ep == 2 and not scfg.lineage_split
    ]
    asks += [
        (s, ds_ep.FE) for s in range(n_stages) if scfg.fsdp_collectives == "xla" and not self.aot and multi_device_tpu(s)
    ]
    # Batch split on Lineage's layers: does each stage run Lineage's SparseCore regions (ds_ep_lineage.Region's
    # `sparsecore`)?  Where libtpu offloads no collective its compiler aborts on them, so they trace as TensorCore
    # code there (ds_ep_lineage.no_sparsecore).  The question is the FSDP one over the whole stage, asked alike --
    # ahead of time too, as a compile-only slice answers it -- so every process of a stage traces its Regions under
    # its lead's answer, and a probe that fails (a compile-only topology off its first slice: ds_ep_aot) answers by
    # the device rule; `stage_lineage_sparsecore` answers instead.
    if scfg.lineage_split and stage_lineage_sparsecore is None:
      asks += [(s, ds_ep.FE) for s in range(n_stages) if (s, ds_ep.FE) not in asks]
    self.proxy = on_pathways(self.meshes) if proxy is None else bool(proxy)
    if self.aot and self.proxy:
      # Ahead of time for a Pathways run (ds_ep_aot --pathways): a compile-only topology's libtpu answers as a TPU
      # VM's own, where the Pathways servers' compiler offloads on chip (0,0,0)'s sub-meshes too -- the device rule's
      # answers under Pathways, unprobed.
      why = "planned for one Pathways client: its device rule (sync_collective_stages), unprobed"
      self.offloads = {(s, axes): (rule_offloads(self.meshes, s, axes, frozenset()), why) for s, axes in asks}
    elif self.multi and asks:  # each stage's lead process answers for its stage, through the key-value store
      from jax._src import distributed

      key = f"mpmd/offload/{next(_OFFLOAD_KEYS)}"
      self.offloads = offload_answers(self.meshes, asks, kv=distributed.global_state.client, key=key)
    else:
      self.offloads = offload_answers(self.meshes, asks)
    sc = lambda s, axes: self.offloads[(s, axes)][0]
    self.stage_scfg = []
    for s in range(n_stages):
      c = scfg
      if stage_ep_collectives is not None:
        c = replace(c, ep_collectives=stage_ep_collectives[s])
      elif scfg.ep_collectives == "auto":
        # 'auto' uses the lax collectives where libtpu offloads them and async
        # collective-permutes elsewhere.  With EP != 2 the permute flavour does
        # not apply (and at EP=1 the lax collectives are no-ops), so only probe
        # when there is a choice -- and there is none on Lineage's layers, which
        # run no ds_ep EP collective: 'xla', unused.
        c = replace(
            c, ep_collectives="xla" if (scfg.ep != 2 or scfg.lineage_split or sc(s, ds_ep.EP_AXIS)) else "ppermute"
        )
      self.stage_scfg.append(c)
    # Stages whose FSDP collectives are lax collectives libtpu keeps synchronous
    # (collective-permutes are async DMAs on every sub-mesh): by its answer
    # where it was asked over the whole stage, else by the device rule.
    rule = frozenset() if self.proxy else sync_collective_stages(self.meshes)  # Pathways': none (its docstring)
    asked = lambda s: (s, ds_ep.FE) in self.offloads and multi_device_tpu(s) and not self.aot
    self.sync_stages = frozenset(
        s
        for s in range(n_stages)
        if self.stage_scfg[s].fsdp_collectives == "xla" and (not sc(s, ds_ep.FE) if asked(s) else s in rule)
    )
    self.lineage_sc = None
    if scfg.lineage_split:
      self.lineage_sc = (
          stage_lineage_sparsecore
          if stage_lineage_sparsecore is not None
          else tuple(bool(sc(s, ds_ep.FE)) for s in range(n_stages))
      )
    self.progs = [
        ds_ep.ChunkPrograms(
            self.stage_scfg[ch.stage],
            ch,
            self.meshes[ch.stage],
            self.rows,
            self.row_len,
            denom,
            lineage_sparsecore=None if self.lineage_sc is None else self.lineage_sc[ch.stage],
        )
        for ch in self.chunks
    ]
    # Which chunks each stage gathers inside its optimizer program (the rest
    # are prefetched inside forwards): the stage holding chunk 0 must start
    # the forward right after its update, so it gathers only its first chunk
    # there; every other stage waits for upstream activations anyway and
    # gathers all of its chunks in that window.
    first_stage = self.place[0]
    self.gather_slots = [
        tuple(range(len(self.stage_chunks[s]))) if (overlap and s != first_stage) else (0,) for s in range(n_stages)
    ]
    # Routed experts read in place: at fsdp 1 a routed expert's compute copy
    # would be its storage's bytes again -- stored over 'expert' alone, with
    # no axis to gather over -- so where the two dtypes agree the stage's
    # programs take the storage leaves themselves: no gather and no second
    # copy for the step (12 GiB a Qwen3.8 layer at EP 4, 5.25 a DeepSeek-V3
    # one).  {chunk: its leaves read so (ds_ep.routed_leaf_ids)}.  The plans'
    # gathers leave them out; `_whole` and `_install` take them from the
    # step's params.  Shard mode auto only (an explicit one types the copy).
    self.direct = {}
    if direct_experts and scfg.fsdp == 1 and scfg.shard_mode == "auto":
      for ch in self.chunks:
        sdt = ds_ep.leaf_dtypes(self.stage_scfg[ch.stage], ch, "storage")
        cdt = ds_ep.leaf_dtypes(self.stage_scfg[ch.stage], ch, "compute")
        ids = tuple(i for i in ds_ep.routed_leaf_ids(ch) if sdt[i] == cdt[i])
        if ids:
          self.direct[ch.index] = ids
    # Self-prefetch: the stage that starts the forward gathers its first
    # chunk's routed experts inside that chunk's first forward (they are not
    # needed until the MoE half), so the update program only gathers the rest.
    self.self_prefetch = bool(overlap and self_prefetch)
    self.routed_leaves = None
    first_leaves = [None] * n_stages
    if self.self_prefetch and (not self.chunks[0].n_moe or 0 in self.direct):
      # chunk 0 is dense layers only, or its routed experts are read in place: none to gather late
      self.self_prefetch = False
    if self.self_prefetch:
      specs = jax.tree.leaves(ds_ep.chunk_storage_specs(self.chunks[0]), is_leaf=ds_ep._is_spec)
      self.routed_leaves = tuple(i for i, sp in enumerate(specs) if len(sp) > 1 and sp[1] == "expert")
      first_leaves[first_stage] = tuple(i for i in range(len(specs)) if i not in self.routed_leaves)
    self.first_leaves = first_leaves

    # -- schedule (after the chunk programs: the stash schedulers size a stash
    # set by building its programs; the stage programs come after it, since
    # the prefetch plan it decides can move a chunk's gather into them)
    if stash not in ds_ep.STASH_POLICIES:
      raise ValueError(f"stash {stash!r}: want one of {sorted(ds_ep.STASH_POLICIES)}")
    self.stash_policy, self.stash_budget_gib = stash, stash_budget_gib
    # Program costs (ms) for the timed and stash schedulers and the summary's
    # simulation: `default_costs`, any key of `costs` over them (per-chunk
    # keys too, see `sched.cost_table`), then each chunk's share of an
    # uneven `layer_split` (`split_costs`: --costs' F / B / A scale like the
    # defaults').
    table = dict(default_costs(self.meshes[0].devices.flat[0].device_kind, scfg))
    table.update(costs or {})
    # A chunk's dense layers cost about a MoE layer each (DeepSeek-V3's dense MLP has the active FLOPs of its
    # experts and shared expert).
    self.costs = split_costs(table, tuple(ch.dense + ch.n_moe for ch in self.chunks), remat_last=scfg.remat_fused_last)
    self.modes = {}
    # F&B pairs run as one program: order[j] and order[j+1] for j in fuse
    self.fuse = frozenset()
    self.pair_gain = float(self.costs.get("pair_gain", 0.0) if pair_gain is None else pair_gain)
    # -> what megatron_interleaved takes: None every stage, () none, else the stages
    if pair_stages is not None and not (isinstance(schedule, str) and schedule == "interleaved"):
      raise ValueError(
          f"pair_stages {pair_stages!r}: only schedule 'interleaved' pairs F and B on stages (a custom order "
          "names its pairs in `fuse`)"
      )
    if isinstance(pair_stages, str) and pair_stages == "all":
      pair_stages = None
    elif pair_stages is not None:
      pair_stages = tuple(sorted(set(pair_stages)))
      if any(not 0 <= s < n_stages for s in pair_stages):
        raise ValueError(f"pair_stages {pair_stages}: stages are 0..{n_stages - 1}")
    else:
      pair_stages = ()  # Megatron's order as it is, unless its pairs are asked for
    self.pair_stages = pair_stages
    if isinstance(schedule, (list, tuple)):
      self.order = list(schedule)  # normalized above
      self.modes = {tuple(k): v for k, v in (modes or {}).items() if v != "r"}
      self.fuse = frozenset(fuse or ())
      if stash != "none" and not self.modes:
        raise ValueError(f"stash {stash!r} with a custom order: say which (chunk, microbatch)s " "stash in `modes`")
      self.schedule_name = (
          "custom"
          + (f" ({len(self.modes)} stash)" if self.modes else "")
          + (f" ({len(self.fuse)} pairs)" if self.fuse else "")
      )
    elif modes or fuse:
      raise ValueError("modes and fuse ride along with a custom order (schedule as a list)")
    elif stash != "none" and schedule not in ("stash", "interleaved"):
      raise ValueError(f"stash {stash!r} needs schedule stash or interleaved (or a custom order with modes)")
    elif schedule == "interleaved":
      # Megatron-LM's interleaved 1F1B, with its steady state's F&B pairs if
      # asked for, as per-stage streams; `merge` lists them as one order
      # (`fuse`: each pair's F), and with `stash` the order's stash modes are
      # handed out under the budget.
      cost, xfer, g = sched.cost_table(self.costs, C), self.costs["xfer"], self.pair_gain
      if placement != "circular":
        raise ValueError(f"schedule interleaved runs on placement 'circular', not {placement!r}")
      # interleave_group: Megatron's microbatch_group_size_per_vp_stage (None: n_stages).  At S per group a
      # microbatch's forward comes round to rank 0's next chunk just in time when the boundary copies cost
      # nothing; on GB200 nodes (a 470 MB DeepSeek-V3 activation a hop, ~3 ms over RDMA) every chunk switch
      # stalled the 16-stage pipeline ~200 ms, twice in each 4.6 s round (megatron_interleaved).
      streams = sched.megatron_interleaved(
          n_stages, chunks_per_stage, n_micro, group=interleave_group, pair_stages=pair_stages
      )
      head = (
          "interleaved (Megatron"
          + ("" if interleave_group in (None, n_stages) else f", groups of {interleave_group}")
          + ")"
      )
      self.order, self.fuse, _ = sched.merge(streams, self.place, cost, xfer, g, n_micro=n_micro)
      est = ""
      if stash != "none":
        nbytes = [self.progs[c].build_stash(stash).bytes if c != C - 1 else 0 for c in range(C)]
        est = " (stash costs estimated)" if self._stash_costs(stash) else ""
        self.modes = sched.stash_modes(
            self.order, self.place, {stash: nbytes}, int(stash_budget_gib * 2**30), stash_caps, policy=stash
        )
      mk = sched.timed_simulate(
          self.order, self.place, cost, xfer, fuse_last=True, modes=self.modes, fuse=self.fuse, pair_gain=g
      )[0]
      on = "every stage" if pair_stages is None else f"stages {list(pair_stages)}"
      # (it names its pairs only where they were asked for: an unpaired run keeps its name)
      pairs = "" if pair_stages == () else f" pairs {len(self.fuse)} ({on}, gain {g:g} ms)"
      self.schedule_name = (
          head
          + pairs
          + (f" stash {stash} budget {stash_budget_gib:g} GiB" if stash != "none" else "")
          + f" sim={mk:.1f}{est}"
      )
    elif schedule == "stash":
      # Each forward stashes while its stage's live stash sets fit the budget
      # (and its chunk is under its cap); the order is searched with the
      # stash backwards' costs, measured or `sched.stash_estimates`.
      pol = None if stash == "none" else stash
      nbytes = [self.progs[c].build_stash(pol).bytes if pol and c != C - 1 else 0 for c in range(C)]
      est = " (stash costs estimated)" if pol and self._stash_costs(pol) else ""
      caps = stash_caps if pol else [0] * C
      if caps is not None:
        caps = list(caps) + [0] * (C - len(caps))  # the last chunk keeps nothing
      res = sched.stash_search(
          C,
          n_micro,
          self.place,
          sched.cost_table(self.costs, C),
          self.costs["xfer"],
          nbytes,
          int(stash_budget_gib * 2**30),
          policy=pol or "r",
          caps=caps,
      )
      self.order, self.modes = res[0], res[1]
      self.schedule_name = (
          f"stash {stash} budget {stash_budget_gib:g} GiB caps {res[2][0]} " f"{res[2][1]} sim={res[3]:.1f}{est}"
      )
    elif schedule in ORDER_SCHEDULES:
      # gpipe / 1f1b / breadth / search from the placement alone; timed, the
      # event-driven search, and zbh1 / zbh2, the zero-bubble schedules, with
      # this pipeline's program costs (ms, only their ratios matter; the last
      # chunk is one fused program)
      self.order, self.schedule_name = build_schedule(
          schedule,
          C,
          n_micro,
          self.place,
          costs=self.costs,
          zb_wbuf_cap=self.zb_wbuf_cap,
          zb_split_last=self.zb_split_last,
      )
    else:
      raise ValueError(
          f"unknown schedule {schedule!r}: expected one of {SCHEDULES} or a custom order "
          "(a sequence of (chunk, kind, microbatch) triples)"
      )
    # The fusion plan and the step loop take the order on trust; a bad one
    # would fail halfway through a step, after buffers were donated.
    sched.validate(self.order, self.place, n_micro, modes=self.modes, fuse=self.fuse)
    if scfg.lineage_split and (ab := sorted(j for j in self.fuse if self.order[j][0] == C - 1)):
      # (A custom order's: interleaved never pairs the last chunk's fused program.)
      # An F&B pair whose F is the last chunk's fused forward + backward pairs two of Lineage's backwards in one
      # program, which XLA compiles with an all-reduce of both chunks' layer gradients over 'fsdp' (bf16; 257 MiB a
      # program at D 1024, 2 + 2 MoE layers, TPU7x, libtpu 0.0.46), each kept on one device after: the values stay
      # right, the program does a synchronous collective it should not.  Not in the lowered program, nor in a pair
      # with one Lineage backward or of ds_ep's own layers.
      raise ValueError(
          f"batch_split on Lineage's layers: F&B pairs of the last chunk's fused program with another "
          f"chunk's backward ({', '.join(self.op_name(j) + '&' + self.op_name(j + 1) for j in ab[:3])}"
          f"{', ...' if len(ab) > 3 else ''}) run two of Lineage's backwards in one program, which XLA "
          "compiles with an all-reduce of their weight gradients over 'fsdp': leave them out of `fuse`"
      )
    # -- the split backward (the zero-bubble schedules): the chunks whose backwards the order runs as I and W build
    # their two halves (ds_ep.ChunkPrograms.build_split: one trace of the chunk's backward, one I, and the W of its
    # first gradient write, which starts its accumulator -- the others add to it; a split last chunk's AI and W,
    # build_split_last), on every process -- what one microbatch holds from its I (AI) to its W (`wbuf_bytes`: the
    # crossing set and the arguments W reads, per device) is in every process's summary and plan.  The simulations
    # price I and W from B where the table has neither, a split last chunk's W from A (sched.zb_costs).
    self.split_chunks = sorted({c for c, k, _ in self.order if k in ("I", "W")})
    # `_do_I`; a plan for one Pathways client (ds_ep_aot --pathways, compile-only TPU devices) does not pace, as its run
    self.pace_i = bool(self.split_chunks) and not self.proxy and allocates_at_issue(self.meshes[0].devices.flat[0])
    self.split_last = C - 1 in self.split_chunks
    self.wbuf_bytes = {}
    for c in self.split_chunks:
      sps = [self.split_of(c, acc) for acc in ((False, True) if n_micro > 1 else (False,))]
      self.wbuf_bytes[c] = max(sp.bytes + sp.kept_bytes for sp in sps)
    if self.split_chunks:
      self.costs = sched.zb_costs(self.costs, C, last=self.split_last)
    pols = set(self.modes.values()) | ({stash} if stash != "none" else set())
    for pol in sorted(pols):
      if pol not in ds_ep.STASH_POLICIES or pol == "none":
        raise ValueError(f"mode {pol!r}: want a stash policy of {sorted(ds_ep.STASH_POLICIES)}")
    # Bytes of one stash set per chunk and policy (per device and microbatch),
    # which also builds the policy's programs.
    self.stash_bytes = {
        pol: [self.progs[c].build_stash(pol).bytes if c != C - 1 else 0 for c in range(C)] for pol in sorted(pols)
    }
    for pol in self.stash_bytes:
      self._stash_costs(pol)

    # -- which ops carry which chunk's gather.  fsdp_scope 'step'
    # (`_prefetch_hosts`): forwards, once a step.  A chunk that no earlier
    # forward on its stage can carry (at n_micro=1 the first stage's only
    # forward before its second chunk is 0F0, busy with self-prefetch; under
    # V the head's too) is gathered by the previous step's update program
    # instead, with the stage's first chunk.  'repeat' (`_plan_repeat`): once
    # a visit, inside the stage's previous visit, and released after it.
    # `pf_gathers`: [(chunk, [(order index, leaf ids), ...])], a gather each.
    self.release_at, self.visits = {}, None
    if overlap and fsdp_scope == "repeat":
      if self.fuse:
        raise ValueError(
            "fsdp_scope 'repeat' folds its gathers into single-op programs: an order with F&B "
            "pairs (interleaved with pair_stages, a custom fuse) has no place for them"
        )
      self.pf_plan = {}
      self.pf_gathers, self.release_at, self.visits, self.gather_slots, self.pf_fallback = self._plan_repeat()
    else:
      self.pf_plan, self.pf_fallback = self._prefetch_hosts() if overlap else ({}, [])
      for b in self.pf_fallback:
        s = self.place[b]
        self.gather_slots[s] = tuple(sorted(self.gather_slots[s] + (self.slot[b],)))
      self.pf_gathers = list(self.pf_plan.items())
    self.stages = [
        ds_ep.StagePrograms(
            self.stage_scfg[s],
            self.stage_chunks[s],
            self.meshes[s],
            n_stages=n_stages,
            gather_slots=self.gather_slots[s],
            first_leaves=first_leaves[s],
            skip_leaves=[self.direct.get(ch.index, ()) for ch in self.stage_chunks[s]],
        )
        for s in range(n_stages)
    ]

    # -- channels: an activation channel for every two stages the placement
    # links (chunk c's stage and chunk c+1's, both ways); a chunk boundary
    # within one stage (the turn of a V) is no copy at all (`_send`).  Across
    # processes, every process holds every channel and walks every copy (the
    # stages' own processes act on it), and so does the gradient norm's scalar
    # hop between every two stages (`_hop`).  Ahead of time only the ppermute
    # channels exist: they are programs to compile; the others are runtime copies
    # (and the hops' only for a multi-controller run: one Pathways client's --
    # ds_ep_aot --pathways, `proxy` -- are jax.device_puts under ppermute).
    act_aval = jax.ShapeDtypeStruct((self.rows, self.row_len, cfg.emb_dim), cfg.dtype)
    act = P(ds_ep.FE, None, None)
    linked = {(self.place[c], self.place[c + 1]) for c in range(C - 1)}
    linked |= {(b, a) for a, b in linked}
    self.channels, self.sq_channels = {}, {}
    # The host-side transports (multi-controller only: refused above in one controller), each with one context per
    # pipeline: xfer's transfer server, or the relay's listener, streams and threads (`close`).
    host = self.copy_ctx = None
    if self.multi and transport in ("xfer", "relay"):
      host = {
          "xfer": (transfer.xfer_context, transfer.XferChannel),
          "relay": (transfer.RelayContext, transfer.RelayChannel),
      }[transport]
      self.copy_ctx = host[0]()
    for a in range(n_stages):
      for b in range(n_stages):
        if a == b or (self.aot and transport != "ppermute"):
          continue
        src, dst = NamedSharding(self.meshes[a], act), NamedSharding(self.meshes[b], act)
        sq = (
            jax.ShapeDtypeStruct((), jnp.float32),
            NamedSharding(self.meshes[a], P()),
            NamedSharding(self.meshes[b], P()),
        )
        if transport == "ppermute":
          if (a, b) in linked:
            self.channels[(a, b)] = transfer.PermuteChannel(act_aval, src, dst, name=f"s{a}->s{b}")
          if self.multi or (self.aot and not self.proxy):  # ahead of time: ds_ep_aot's multi-controller target
            self.sq_channels[(a, b)] = transfer.PermuteChannel(*sq, name=f"sumsq s{a}->s{b}")
        elif not self.multi:  # one controller: jax's batched device-to-device copy
          if (a, b) in linked:
            self.channels[(a, b)] = transfer.Channel(act_aval, src, dst, name=f"s{a}->s{b}", strict=True)
          if on_pathways(self.meshes):
            # The gradient norm's hops too (`_hop`): through the IFRT proxy a prebuilt Channel's send costs
            # ~105-155 us of host time, a jax.device_put ~150-240 us, and a step makes S(S-1) of them -- one
            # batched copy either way (local docker Pathways, tpu7x-8).
            self.sq_channels[(a, b)] = transfer.Channel(*sq, name=f"sumsq s{a}->s{b}", strict=True)
        elif host:
          # the channel's id (xfer's uuid prefix), the same on every process; its sumsq hop's is the next
          k = 2 * (a * n_stages + b)
          if (a, b) in linked:
            self.channels[(a, b)] = host[1](act_aval, src, dst, self.copy_ctx, k, name=f"s{a}->s{b}")
          self.sq_channels[(a, b)] = host[1](*sq, self.copy_ctx, k + 1, name=f"sumsq s{a}->s{b}")
        else:
          if (a, b) in linked:
            self.channels[(a, b)] = transfer.CrossProcessChannel(act_aval, src, dst, name=f"s{a}->s{b}")
          self.sq_channels[(a, b)] = transfer.CrossProcessChannel(*sq, name=f"sumsq s{a}->s{b}")
    # MTP's copy of the token embedding (the last chunk's, ds_ep.mtp_shapes)
    # and chunk 0's table are one parameter: each step their two gradients
    # are exchanged and summed before the norm and the update (`_tie`), over
    # a channel each way between the two stages -- ahead of time, as above,
    # only the ppermute ones, programs to compile.
    self.tie_channels = {}
    a0, b0 = self.place[0], self.place[C - 1]
    if self.mtp and a0 != b0 and not (self.aot and transport != "ppermute"):
      emb_aval = jax.ShapeDtypeStruct((cfg.vocab_size, cfg.emb_dim), scfg.rs_dtype)
      emb = P(None, ds_ep.FE)
      for k, (a, b) in enumerate(((a0, b0), (b0, a0))):
        src, dst = NamedSharding(self.meshes[a], emb), NamedSharding(self.meshes[b], emb)
        name = f"tied embedding s{a}->s{b}"
        if transport == "ppermute":
          self.tie_channels[(a, b)] = transfer.PermuteChannel(emb_aval, src, dst, name=name)
        elif not self.multi:
          self.tie_channels[(a, b)] = transfer.Channel(emb_aval, src, dst, name=name, strict=True)
        elif host:
          self.tie_channels[(a, b)] = host[1](emb_aval, src, dst, self.copy_ctx, 2 * n_stages * n_stages + k, name=name)
        else:
          self.tie_channels[(a, b)] = transfer.CrossProcessChannel(emb_aval, src, dst, name=name)
    for k, chn in self.channels.items():
      if getattr(chn, "n_slow", 0):
        raise RuntimeError(f"{chn}: boundary would bounce through the host")

    # -- per-stage constants, committed once
    rep = lambda s, x: jax.device_put(x, NamedSharding(self.meshes[s], P()))
    self.seeds = [[rep(s, jnp.int32(m)) for m in range(n_micro)] if self._places(s) else None for s in range(n_stages)]

    # -- who writes each chunk's gradient accumulator, and when
    shp = lambda c: ds_ep.chunk_shapes(scfg, self.chunks[c])
    self.treedef = {c: jax.tree.structure(shp(c), is_leaf=ds_ep._is_shape) for c in range(C)}
    self.n_leaves = {c: self.treedef[c].num_leaves for c in range(C)}
    self.first_writer, self.last_writer = {}, {}
    for j, (c, k, m) in enumerate(self.order):
      if self._writes(c, k):
        self.first_writer.setdefault(c, j)
        self.last_writer[c] = j

    # interleaved has no earlier plan to keep, so it reserves under circular
    # placement too
    self.rs_reserve = (placement == "v" or schedule == "interleaved") if rs_reserve is None else bool(rs_reserve)
    # the fusion plan (empty without overlap)
    self.pf, self.rs_at, self.rs_groups, self.rs_self = {}, {}, {}, {}
    self.pf_n = {}  # host order index -> how many hosts its gather has
    self.self_pf_at, self.final_slots, self.n_partial_sq = None, [()] * n_stages, [0] * n_stages
    if overlap:
      self._plan_overlap()
    self.pins, self.unpinned = self._plan_pins() if overlap else ({}, {})
    self._ops = {}  # overlap -> [Op]
    self._issue = {}  # overlap -> the order's indices as the step loop issues them (issue_order)
    self._exe = {}  # (chunk, desc) -> compiled program (precompile)
    self.step_gnorms = ()  # the last step's gnorm of every local stage (`step`)
    self.on_grads = None  # step: called with the reduced gradients before the update (--grad-digest)

  def _stash_costs(self, pol):
    """Give `self.costs` a stash backward and forward cost of `pol` for every
    non-last chunk: `sched.stash_estimates` wherever the table has none (the
    keys `sched.cost_table` would read, d-keys included).  -> the keys estimated."""
    guess, added = sched.stash_estimates(self.costs, pol, self.C), []
    for c in range(self.C - 1):
      for kind in ("B", "F"):
        s = f"{kind}s"
        if not any(
            n in self.costs for n in (f"{s}{c}:{pol}", f"d{s}{c}:{pol}", f"{s}:{pol}", f"d{s}:{pol}", f"{s}{c}", s)
        ):
          self.costs[f"{s}{c}:{pol}"] = guess[f"{s}{c}:{pol}"]
          added.append(f"{s}{c}:{pol}")
    return added

  @property
  def relay_ctx(self):
    """The relay's context under transport 'relay' (its settings:
    `describe()`; where its time went: `timing()`), else None."""
    return self.copy_ctx if self.transport == "relay" else None

  def _runs(self, s):
    """Does this process compile and run stage s's programs: in one controller -- a Pathways client, one host's own
    process -- every stage, `local` not looked at; under multi-controller JAX its own stages (`local`); ahead of time
    the stages ds_ep_aot compiles (`local` again: --stages-to-compile)."""
    return not (self.multi or self.aot) or s in self.local

  def _places(self, s):
    """Does this process put stage s's arrays (weights, batches, seeds) on its devices: in one controller every stage,
    `placed` not looked at; under multi-controller JAX its own stages; ahead of time none."""
    return not (self.multi or self.aot) or s in self.placed

  def _writes(self, c, k):
    """Does op (c, k, .) write chunk c's gradient accumulator?  A B or a W -- a split last chunk's W its deferred
    part -- and the fused last chunk's F."""
    return k == "W" or (k == "B" and c != self.C - 1) or (k == "F" and c == self.C - 1)

  def split_of(self, c, acc=False):
    """Chunk c's split for the zero-bubble schedules (ds_ep.Split, of `acc`'s W): a middle chunk's backward cut into I
    and W (ChunkPrograms.build_split), the fused last chunk's program into AI and W (build_split_last)."""
    pr = self.progs[c]
    return pr.build_split_last(acc) if c == self.C - 1 else pr.build_split(acc)

  def _send(self, a, b, x):
    """Stage a's activation / cotangent x -> stage b.  On the same stage (the
    turn of a V placement) it is already where it is needed: no copy, and
    nothing a later donation could clobber -- x has no other owner.

    Under multi-controller JAX (`multi`) every process calls this for every
    boundary copy of the order, in order: x is None where stage a is not
    local (its process passes no source), and the result is None where stage
    b is not -- and under transport 'relay' a RelayFuture where it is, which
    the consumer's pop resolves (`_RelayBox`)."""
    if a == b:
      return x
    y = self.channels[(a, b)].send_flat([x])[0]
    return y if not self.multi or b in self.local else None

  def _hop(self, t, s, x):
    """Stage t's gradient sum of squares (a replicated scalar) -> stage s, for
    its global-norm clip.  Every process, every pair, in order, when the
    stages are on different processes; None where s is not local (under
    transport 'relay' a RelayFuture where it is, which `step` resolves for
    the update).  In one process a jax.device_put -- under Pathways with the
    native transport a prebuilt Channel's send (cheaper through the proxy;
    ppermute builds none, so a jax.device_put there too): one batched copy
    either way."""
    if not self.multi:
      if (t, s) in self.sq_channels:
        return self.sq_channels[(t, s)].send_flat([x])[0]
      return jax.device_put(x, NamedSharding(self.meshes[s], P()))
    y = self.sq_channels[(t, s)].send_flat([x])[0]
    return y if s in self.local else None

  # --------------------------------------------------------- fusion plan
  def _prefetch_hosts(self):
    """-> ({chunk: [(order index, leaf ids), ...]}, [chunks with no host]).

    Any forward on the stage that runs before chunk b's first forward can
    carry a slice of b's gather: take the earliest free ones (1 to
    `pf_split`, as many as there are), deadline-first.  Chunks gathered in
    the update program (gather_slots) need none; the self-prefetched 0F0
    hosts nothing else, and neither half of an F&B pair hosts anything.  A
    chunk left without a host is the caller's to gather in the update
    program.
    """
    C, order, place = self.C, self.order, self.place
    paired = set(self.fuse) | {j + 1 for j in self.fuse}
    used = set()
    if self.self_prefetch:
      j = order.index((0, "F", 0))
      if j in paired:
        raise ValueError(f"0F0 (order[{j}]) gathers its own routed experts: it cannot be half of " "an F&B pair")
      used.add(j)
    plan, homeless = {}, []
    for s in range(self.S):
      fwd_slots = [
          j for j, (c, k, m) in enumerate(order) if k == "F" and place[c] == s and c != C - 1 and j not in paired
      ]
      need = [
          (order.index((ch.index, "F", 0)), ch.index)
          for i, ch in enumerate(self.stage_chunks[s])
          if i not in self.gather_slots[s]
      ]
      for deadline, b in sorted(need):
        hosts = [j for j in fwd_slots if j < deadline and j not in used][: self.pf_split]
        if not hosts:
          homeless.append(b)
          continue
        groups = ds_ep.leaf_groups(ds_ep.chunk_shapes(self.scfg, self.chunks[b]), len(hosts), skip=self.direct.get(b, ()))
        plan[b] = list(zip(hosts, groups))
        used.update(j for j, _ in plan[b])
    return plan, homeless

  def _visits(self):
    """Each stage's visits: maximal runs of one chunk's ops in the stage's
    sequence (the fused last chunk's no-op backwards left out) -> per stage
    [(chunk, [order index, ...]), ...]."""
    out = []
    for s in range(self.S):
      vs = []
      for j, (c, k, m) in enumerate(self.order):
        if self.place[c] != s or (c == self.C - 1 and k == "B"):
          continue
        if vs and vs[-1][0] == c:
          vs[-1][1].append(j)
        else:
          vs.append((c, [j]))
      out.append(vs)
    return out

  def _plan_repeat(self):
    """fsdp_scope 'repeat': gather a chunk's weights once per visit (`_visits`)
    -> (pf_gathers, release_at, visits, gather_slots, fallback).

    A stage's first visit takes its chunk from the previous step's update
    program (gather_slots; chunk 0's routed experts, with self-prefetch,
    inside 0F0).  Every later visit's gather rides in the last `pf_split`
    ops (1 to `pf_split`, as many as there are) of the nearest earlier visit
    that has a free one -- a forward, a backward, or the fused last chunk;
    not 0F0, which gathers its own experts -- split into leaf groups of
    equal bytes; never in an op of an earlier visit of the same chunk.  In a
    backward visit its last ops are the ones the reduce-scatter plan reaches
    last (it takes the first ones after a chunk's last gradient write), so
    the next chunk's weights arrive as the previous chunk's accumulator
    goes.  A visit with no such op keeps its chunk's weights from the
    previous visit instead (or, for a chunk's first visit, has the update
    program gather them: `fallback`, the summary's to explain).  `release_at`
    {order index: (chunks,)}: the weights
    of a visit's chunk are dropped after its last op, unless the chunk's
    next visit keeps them.

    Refuses an order that visits a chunk more than twice a step (a
    depth-first one: `gpipe` / `1f1b` at V > 1 and M > S switch chunks
    every S microbatches or so): `schedule='breadth'` visits each chunk once
    forward and once backward."""
    order, C = self.order, self.C
    visits = self._visits()
    for s, vs in enumerate(visits):
      n = collections.Counter(c for c, _ in vs)
      bad = {c: k for c, k in sorted(n.items()) if k > 2}
      if bad:
        raise ValueError(
            f"fsdp_scope 'repeat' gathers a chunk's weights once per visit (a run of its ops on its "
            f"stage); this order visits stage {s}'s chunks {bad} (chunk: visits) more than twice a "
            "step -- use a breadth-first order (schedule 'breadth')"
        )
    self_pf = order.index((0, "F", 0)) if self.self_prefetch else None
    gathers, release, slots, fallback = [], {}, [], []
    for s, vs in enumerate(visits):
      slots.append((self.slot[vs[0][0]],))
      used, keep = set(), set()
      for i in range(1, len(vs)):
        c = vs[i][0]
        prev = max((k for k in range(i) if vs[k][0] == c), default=None)
        hosts, n = [], max(self.pf_split, 0)  # pf_split 0: no op hosts a gather, as under 'step'
        for k in range(i - 1, -1 if prev is None else prev, -1):
          free = [j for j in vs[k][1] if j != self_pf and j not in used]
          if free and n:
            hosts = free[-n:]
            break
        if not hosts:
          if prev is None:
            slots[s] = tuple(sorted(set(slots[s]) | {self.slot[c]}))
            fallback.append(c)
          else:
            keep.add(i)
          continue
        groups = ds_ep.leaf_groups(ds_ep.chunk_shapes(self.scfg, self.chunks[c]), len(hosts), skip=self.direct.get(c, ()))
        gathers.append((c, list(zip(hosts, groups))))
        used.update(j for j, _ in gathers[-1][1])
      for i, (c, js) in enumerate(vs):
        nxt = next((k for k in range(i + 1, len(vs)) if vs[k][0] == c), None)
        if nxt not in keep:
          release.setdefault(js[-1], []).append(c)
    return gathers, {j: tuple(cs) for j, cs in release.items()}, visits, slots, sorted(fallback)

  def _plan_overlap(self):
    """Fold the once-per-step FSDP collectives into the per-microbatch programs.

    gather:  chunk k_{i+1}'s weights are all-gathered inside the first
             `pf_split` forwards (1 to `pf_split`, as many as there are) of
             other chunks on the same stage that run before k_{i+1}'s first
             forward, split into leaf groups of equal bytes (`_prefetch_hosts`,
             decided in __init__).  The stage's first chunk is gathered by the
             previous step's update program -- its routed experts, with
             self-prefetch, inside its own first forward F(0, 0) instead --
             and so is any chunk no earlier forward can carry (`pf_fallback`).
             Under fsdp_scope 'repeat', once per visit instead
             (`_plan_repeat`).
    reduce:  chunk k's reduce-scatter rides in the next `rs_split` backward
             programs on its stage after k's last gradient write, again split
             by leaf groups; whatever has no later backward (the chunk that
             finishes last) folds it into its own last backward (rs_self) or,
             failing that, gets a small standalone program.  With
             `rs_reserve` (by default under V and interleaved) that last
             backward is kept free for rs_self when another chunk would
             take it.  A split chunk's backward programs here are its W's,
             its gradient writers (an I writes none, and the stage upstream
             waits on its outputs): a W hosts what a backward hosts, rs_self
             in a chunk's last W included -- but a split last chunk's W's,
             which host nothing, as its fused program does not: its
             reduce-scatter waits for its accumulator's two parts, merged
             after its last W (the stage's final reduce).

    Hosts may run in stash mode (the composer builds the stash variants);
    no half of an F&B pair hosts anything.  The plan is keyed by order index.
    """
    C, order, place = self.C, self.order, self.place
    shp = lambda c: ds_ep.chunk_shapes(self.scfg, self.chunks[c])
    paired = set(self.fuse) | {j + 1 for j in self.fuse}
    if self.self_prefetch:
      self.self_pf_at = order.index((0, "F", 0))
    for b, hosts in self.pf_gathers:  # from _prefetch_hosts / _plan_repeat
      for j, g in hosts:
        self.pf[j] = (b, g)
        self.pf_n[j] = len(hosts)
    if self.scfg.grad_reduce == "microbatch":
      # Every backward already reduce-scattered its own gradient; the stage's
      # final program only computes the norm the global clip needs.
      self.final_slots = [tuple(range(len(self.stage_chunks[s]))) for s in range(self.S)]
      for s in range(self.S):
        self.stages[s].reduce_final(self.final_slots[s], 0)
      return
    last, first = self.last_writer, self.first_writer

    def can_rs_self(j):
      c, k, _ = order[j]
      return c != C - 1 and k in ("B", "W") and j != first[c] and j not in paired

    def hosts_on(s, reserve):
      """Stage s's reduce-scatter hosts -> ({host: (chunk, leaf ids)},
      {chunk: groups}, leftover slots), chunks taken in the order they finish.
      `reserve`: keep the stage's last gradient write out of it."""
      rs_at, n_groups, left, taken = {}, {}, [], set()
      stop = max(last[c] for c in last if place[c] == s)
      for c in sorted((c for c in last if place[c] == s), key=lambda c: last[c]):
        hosts = [
            j
            for j in range(last[c] + 1, len(order))
            if order[j][1] in ("B", "W")  # a gradient writer: a B, or a split chunk's W
            and order[j][0] != C - 1
            and place[order[j][0]] == s
            and j not in taken
            and j not in paired
            and not (reserve and j == stop)
        ][: self.rs_split]
        if not hosts:
          left.append(self.slot[c])
          continue
        groups = ds_ep.leaf_groups(shp(c), len(hosts))
        n_groups[c] = len(groups)
        for j, g in zip(hosts, groups):
          taken.add(j)
          rs_at[j] = (c, g)
      return rs_at, n_groups, tuple(sorted(left)), stop

    for s in range(self.S):
      if s in self.sync_stages and s != place[0]:
        # Synchronous collectives (no SparseCore offload on this strip): a
        # reduce-scatter folded into a backward lengthens it, and with it the
        # cotangent the stage upstream waits for.  This stage idles in the
        # drain after its last backward: reduce every chunk there instead.
        self.final_slots[s] = tuple(range(len(self.stage_chunks[s])))
        continue
      rs_at, n_groups, left, stop = hosts_on(s, False)
      # rs_self: a stage whose only leftover is the chunk that finishes last
      # folds that reduce-scatter into the chunk's own last backward (a plain
      # accumulating one) instead of a trailing standalone program; it sums the
      # stage's partial sumsqs, which all come from earlier programs.  With
      # rs_reserve (V's default), if another chunk's groups took that backward
      # (under V the head finishes on stage 0 before chunk 0 does), keep it free
      # for rs_self instead -- where the other chunk still finds hosts without
      # it.  Off for circular: it would change the plans circular placement has
      # always made (timed and search at the first TPU7x benchmark
      # configurations, 1f1b / gpipe, and the greedy order since removed: one
      # standalone program fewer, one host fewer for the other chunk, and the
      # stage sumsq summed in another order -- the global norm moves in the last
      # bit).
      rs_self_here = self.rs_self_enabled and (self.rs_self_policy == "all" or s == place[0])
      if self.rs_reserve and rs_self_here and can_rs_self(stop):
        alt = hosts_on(s, True)
        if (len(left) != 1 or stop in rs_at) and alt[2] == (self.slot[order[stop][0]],):
          rs_at, n_groups, left, stop = alt
      elif self.rs_self_policy == "first" and s != place[0]:
        # The point of 'first': the backward that emits this stage's last
        # cotangent carries no reduce-scatter -- not another chunk's either.
        rs_at, n_groups, left, stop = hosts_on(s, True)
      self.rs_at.update(rs_at)
      self.rs_groups.update(n_groups)
      self.n_partial_sq[s] = sum(n_groups.values())
      self.final_slots[s] = left
      if rs_self_here and can_rs_self(stop) and stop not in rs_at and left == (self.slot[order[stop][0]],):
        self.rs_self[stop] = self.n_partial_sq[s]
        self.final_slots[s] = ()
    for s in range(self.S):
      if self.final_slots[s]:
        self.stages[s].reduce_final(self.final_slots[s], self.n_partial_sq[s])  # build it now

  def gathering_forwards(self):
    """{order index: [(chunk, leaf id)]}: the forwards of the fusion plan that
    gather -- each prefetch host the leaves of its group, the self-prefetch
    chunk 0's routed experts.  Not the backwards and fused last chunk that
    host a gather under fsdp_scope 'repeat': no pin reaches into those."""
    hosts = {
        j: [(b, i) for i in g]
        for j, (b, g) in self.pf.items()
        if self.order[j][1] == "F" and self.order[j][0] != self.C - 1
    }
    if self.self_pf_at is not None:
      hosts.setdefault(self.self_pf_at, []).extend((0, i) for i in self.routed_leaves)
    return dict(sorted(hosts.items()))

  def joint_gather_allreduce(self):
    """Does libtpu compile a stage's gather over both mesh axes to an
    all-reduce (`ds_ep.pin_gather_ms`)?  In a program compiled for several
    slices it does (its all-gather decomposer), unless LIBTPU_INIT_ARGS
    prefers async all-gathers (--xla_tpu_prefer_async_allgather_to_allreduce,
    the libtpu set 'prefag' of the 2 x v5p-128 runs), read as libtpu reads it: its last
    mention, in any absl spelling (libtpu_flag).  The slices are
    counted among all the devices of the meshes' client -- the job's, or a
    compile-only topology's, whose second slice's stages ds_ep_aot compiles
    on the first slice's devices -- else among the meshes' own.  Not on v6e
    (ds_ep.NATIVE_JOINT_GATHER_KINDS), whose multi-slice programs gather such
    a leaf natively."""
    devs = [d for m in self.meshes for d in m.devices.flat]
    if all(str(getattr(d, "device_kind", "")) in ds_ep.NATIVE_JOINT_GATHER_KINDS for d in devs):
      return False
    client = getattr(devs[0], "client", None)
    pool = list(client.devices()) if client is not None else devs
    slices = {slice_index(d) for d in pool} - {None}
    prefer = libtpu_flag("xla_tpu_prefer_async_allgather_to_allreduce") is True
    return len(slices) > 1 and not prefer

  def pin_model(self, stage):
    """The ds_ep.PinModel 'auto' plans `stage`'s forwards with: its devices'
    kind's (ds_ep.PIN_MODELS: v5p's or v6e's), else -- off TPU, a kind none
    was calibrated on, or a stage of mixed kinds -- v5p's, ds_ep.PIN_MODEL
    (`_plan_pins` warns on a TPU)."""
    kinds = {str(getattr(d, "device_kind", "")) for d in self.meshes[stage].devices.flat}
    kind = kinds.pop() if len(kinds) == 1 else None
    return ds_ep.PIN_MODELS.get(kind, ds_ep.PIN_MODEL)

  def _plan_pins(self):
    """({order index: ((point, layer, chunk, leaf id), ...)}, {order index:
    why none}): the prefetch pins of every forward that gathers
    (`gathering_forwards`) under `prefetch_pin` -- a SPEC's points for the
    leaves it names (ds_ep.pin_of), or 'auto''s plan (ds_ep.plan_pins: the
    leaves' gathers and the forward's compute windows estimated with the host
    stage's `pin_model`, each gather in the window it loads least) -- and why
    each other gathering forward has none, for the summary.  A forward in
    stash mode is left unpinned: it runs its layers inside jax.vjp, whose
    trace a pin would leave -- and so, under fsdp_scope 'repeat', is a
    backward or the fused last chunk that hosts a gather (listed with the
    unpinned forwards).  'auto' prices async SparseCore collectives
    (ds_ep.PinModel): it plans only for the stages whose FSDP collectives are
    lax ones ('xla') libtpu offloads -- not for collective-permutes, nor for
    `sync_stages`, whose gathers block the TensorCore wherever they sit: a
    pin there would hide nothing, only hold the gathered weights longer (and
    compile the program with barrier removal off)."""
    spec = self.prefetch_pin
    if spec == "off":
      return {}, {}
    tokens = self.rows * self.row_len // (self.scfg.fsdp * self.scfg.ep)  # a device's: whole rows of seq / cp
    joint = self.joint_gather_allreduce()
    out, why = {}, {}
    fwd = self.gathering_forwards()
    for j in self.pf.keys() - fwd.keys():
      why[j] = "the fused last chunk" if self.order[j][0] == self.C - 1 else "a backward"
    for j, leaves in fwd.items():
      c, _, m = self.order[j]
      host = self.chunks[c]
      if self.modes.get((c, m)):
        why[j] = "stash mode"
      elif spec == "auto" and self.stage_scfg[host.stage].fsdp_collectives != "xla":
        why[j] = "collective-permute FSDP collectives"
      elif spec == "auto" and host.stage in self.sync_stages:
        why[j] = "synchronous FSDP collectives"
      elif spec == "auto" and not host.n_moe:
        why[j] = "dense layers only: no pin points"
      if j in why:
        continue
      if spec == "auto":
        own = tuple((b, i) for b, i in leaves if b == c)
        plan = ds_ep.plan_pins(
            self.stage_scfg[host.stage],
            host,
            [(self.chunks[b], i) for b, i in leaves],
            tokens,
            joint_allreduce=joint,
            own=own,
            model=self.pin_model(host.stage),
            seq=self.seq,
        )
      else:
        plan = []
        for b, i in leaves:
          hit = ds_ep.pin_of(spec, self.leaf_names[b][i], b)
          if hit is None or not ds_ep.has_gather(self.chunks[b], i):
            continue
          if hit[1] >= host.n_moe:  # the points are in MoE layers: a dense layer has none
            raise ValueError(
                f"--prefetch-pin: {self.leaf_names[b][i]} (chunk {b}) is pinned in layer {hit[1]} of "
                f"{self.op_name(j)}, whose chunk {c} has {host.n_moe} MoE layer(s)"
            )
          plan.append((*hit, b, i))
      if plan:
        out[j] = tuple(sorted(plan))
      else:
        why[j] = "no window holds one of its gathers" if spec == "auto" else "none of its gathered leaves named"
    if spec == "auto" and out:
      odd = pin_model_mismatch(self.meshes[self.chunks[self.order[j][0]].stage].devices.flat[0] for j in out)
      if odd:
        warnings.warn(
            f"--prefetch-pin auto: its costs (ds_ep.PIN_MODELS) are calibrated on v5p and v6e, not "
            f"{', '.join(odd)}: it estimates the windows and the gathers at v5p's rates, and on a faster "
            "chip may tie more to a window than its compute covers (every leaf at hn replayed 4.3 ms slower "
            "on v5p); check the plan in the summary, or pin with a SPEC",
            RuntimeWarning,
        )
    return out, why

  def pin_summary(self):
    """The prefetch pins in a line ('' under 'off', or without the fusion
    plan): each pinned forward's points and the leaves pinned there, by name,
    then every gathering program left unpinned and why (`_plan_pins`) --
    'none' first where no forward took a pin, so the log of a run asked for
    pins says when it ran without them."""
    if self.prefetch_pin == "off" or not self.overlap:
      return ""
    parts = []
    for j, plan in sorted(self.pins.items()):
      by = collections.defaultdict(list)
      for pt, layer, c, i in plan:
        by[(ds_ep.PIN_POINTS.index(pt), layer)].append(f"c{c}.{self.leaf_names[c][i]}")
      parts.append(
          f"{self.op_name(j)} "
          + " ".join(f'{ds_ep.PIN_POINTS[p]}{f"@{l}" if l else ""}=' + ",".join(v) for (p, l), v in sorted(by.items()))
      )
    if not parts:
      parts = ["none" if self.unpinned else "none (no forward gathers)"]
    rest = ", ".join(f"{self.op_name(j)} ({why})" for j, why in sorted(self.unpinned.items()))
    what = "auto" if self.prefetch_pin == "auto" else "spec"
    return f"           prefetch pins ({what}): " + "; ".join(parts) + (f"; unpinned: {rest}" if rest else "")

  # ------------------------------------------------------- op descriptors
  def ops(self, overlap=None):
    """The order as the step loop runs it (cached): one `Op` per entry, with
    the descriptor of its program -- the part of its kind in its mode, plus
    whatever the fusion plan folds into it when `overlap`."""
    overlap = self.overlap if overlap is None else bool(overlap)
    if overlap and not self.overlap:
      raise ValueError("this pipeline was built without overlap: it has no fusion plan")
    if overlap in self._ops:
      return self._ops[overlap]
    pd, C, out = ds_ep.pdesc, self.C, []
    # A gather rides after the op's own part (and a backward's reduce-scatter)
    # a gather of the whole chunk takes no leaf list (the program the first plans named); one host's group of fewer
    # leaves -- the chunk's routed experts read in place (`direct`) -- names them
    gather = lambda j, pf: (
        pd(
            "G", other=self.chunks[pf[0]], leaves=pf[1] if self.pf_n[j] > 1 or len(pf[1]) < self.n_leaves[pf[0]] else None
        ),
    )
    w_at = {(c, m): j for j, (c, k, m) in enumerate(self.order) if k == "W"}  # each split (chunk, microbatch)'s W
    # a split last chunk's accumulator is two parts, its F's (AI's) and its W's, each started by its first writer
    last_w0 = next((j for j, (c, k, _) in enumerate(self.order) if c == C - 1 and k == "W"), None)
    for j, (c, k, m) in enumerate(self.order):
      s, md = self.place[c], self.modes.get((c, m))
      acc = self._writes(c, k) and j != self.first_writer[c]
      pf = self.pf.get(j) if overlap else None
      if c == C - 1:
        if k == "B":
          out.append(Op(j, c, k, m, s))
        elif k == "F":
          kind = "AI" if self.split_last else "A"
          out.append(Op(j, c, k, m, s, (pd(kind, acc=acc),) + (gather(j, pf) if pf else ()), acc, pf=pf))
        else:  # the split last chunk's W: its layers' weight gradients, its accumulator's deferred part
          acc = j != last_w0
          out.append(Op(j, c, k, m, s, (pd("W", acc=acc),) + (gather(j, pf) if pf else ()), acc, pf=pf))
        continue
      if k == "F":
        self_pf = overlap and j == self.self_pf_at
        pin = self.pins.get(j) if overlap else None
        desc = (pd("F", stash=md, self_leaves=self.routed_leaves if self_pf else None, pin=pin),)
        if pf:
          desc += gather(j, pf)
        out.append(Op(j, c, k, m, s, desc, stash=md, self_pf=self_pf, pf=pf, pin=pin))
      elif k == "B":
        rs = self.rs_at.get(j) if overlap else None
        rs_self = self.rs_self.get(j) if overlap else None
        desc = (pd("B", stash=md, acc=acc, rs_self=rs_self),)
        if rs:
          desc += (pd("R", other=self.chunks[rs[0]], leaves=rs[1] if self.rs_groups[rs[0]] > 1 else None),)
        if pf:
          desc += gather(j, pf)
        out.append(Op(j, c, k, m, s, desc, acc, md, rs=rs, rs_self=rs_self, pf=pf))
      elif k == "I":
        # the split backward's input gradient: dx and the W buffer -- one program for either W, unless in-place dW
        # cut the accumulating W its own I (ds_ep.Split.i_acc)
        acc = w_at[(c, m)] != self.first_writer[c]
        desc = (pd("I", acc=self.progs[c].build_split(acc).i_acc),) + (gather(j, pf) if pf else ())
        out.append(Op(j, c, k, m, s, desc, acc, pf=pf))
      else:
        # W: the weight gradients, the split chunk's one gradient writer -- so, as a B is, a host of reduce-scatters
        # (another chunk's, or with rs_self its own), and a better one: nothing waits on its outputs
        rs = self.rs_at.get(j) if overlap else None
        rs_self = self.rs_self.get(j) if overlap else None
        desc = (pd("W", acc=acc, rs_self=rs_self),)
        if rs:
          desc += (pd("R", other=self.chunks[rs[0]], leaves=rs[1] if self.rs_groups[rs[0]] > 1 else None),)
        if pf:
          desc += gather(j, pf)
        out.append(Op(j, c, k, m, s, desc, acc, rs=rs, rs_self=rs_self, pf=pf))
    for j in sorted(self.fuse):
      # One program of both: the F's part (A for the last chunk) and then the
      # backward's, taken from its own chunk (`peer`) on the same stage mesh.
      # The plan folds nothing into either half.
      f, b = out[j], out[j + 1]
      if len(f.desc) != 1 or len(b.desc) != 1 or f.self_pf or b.rs_self is not None:
        raise AssertionError(f"the fusion plan put a collective into the pair at order[{j}]")
      ((kind, kw),) = b.desc
      peer = self.progs[b.c] if b.c != f.c else None
      out[j] = dataclasses.replace(f, pair=f.desc + (pd(kind, peer=peer, **dict(kw)),))
    self._ops[overlap] = out
    return out

  def issue_order(self, overlap=None):
    """The indices of `ops(overlap)` in the order the step loop issues them (cached): the order's own, but each
    stage's trailing W's -- its W's after its last other op -- after every other op, in the order's order.  Every
    stage runs its ops in the same order either way, so the results are the same bit for bit; an order without W's is
    issued as it is.

    Nothing in the loop waits on a trailing W: its outputs are its chunk's accumulator, what it reduce-scatters and
    its stage's sum of squares, all read after the loop.  Under McJAX the one dispatch thread allocates a program's
    outputs as it issues it, and where a split stage's HBM runs out it does at its last W, which folds the
    reduce-scatter (rs_self) and writes the gradients into new buffers: libtpu waits for the stage's queued programs
    to finish and defragments (TpuClient::DefragmentMemory: 0.6-0.7 s of the dispatch thread at DeepSeek-V3 widths
    on TPU7x without `_pace`), and every op issued after the W waits with it -- in the order's own position, the
    other stages' last backwards.  Issued last, it holds back only what follows the loop."""
    overlap = self.overlap if overlap is None else bool(overlap)
    if overlap not in self._issue:
      ops = self.ops(overlap)
      last = {}  # stage -> its last op but a W
      for op in ops:
        if op.kind != "W":
          last[op.s] = op.j
      tail = [op.j for op in ops if op.kind == "W" and op.j > last.get(op.s, -1)]
      moved = set(tail)
      self._issue[overlap] = [op.j for op in ops if op.j not in moved] + tail
    return self._issue[overlap]

  def program(self, op):
    """The callable `op` runs: its precompiled executable, else the jitted
    composition of its parts (`ChunkPrograms.program`).  For the first op of
    an F&B pair that is the pair's one program (`op.pair`)."""
    desc = op.pair or op.desc
    exe = self._exe.get((op.c, desc))
    return exe if exe is not None else self.progs[op.c].program(desc)

  def dispatched(self, overlap=None):
    """(chunk, descriptor) of every per-microbatch program one step
    dispatches, in order: an F&B pair once (its first op's chunk and the
    composed descriptor), the fused last chunk's no-op backward never."""
    return [(op.c, op.pair or op.desc) for op in self.ops(overlap) if op.desc is not None and op.j - 1 not in self.fuse]

  def program_at(self, j, overlap=None):
    return self.program(self.ops(overlap)[j])

  # ---------------------------------------------------------- precompile
  def _avals(self, s, kind, chunk=None, slots=None):
    """Abstract (sharded) arguments, for ahead-of-time lowering."""
    scfg, mesh = self.scfg, self.meshes[s]
    cfg = scfg.model
    sds = lambda shape, dt, spec: jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))

    def tree(shapes, specs, dt):
      if not isinstance(dt, dict):
        dt = jax.tree.map(lambda _: dt, shapes, is_leaf=ds_ep._is_shape)
      return jax.tree.map(lambda sh, sp, d: sds(sh, d, sp), shapes, specs, dt, is_leaf=ds_ep._is_shape)

    ch = chunk
    if kind == "compute":
      return tree(
          ds_ep.chunk_shapes(scfg, ch), ds_ep.chunk_wc_specs(scfg, ch), ds_ep.leaf_dtype_tree(scfg, ch, "compute")
      )
    if kind == "storage":
      return tree(ds_ep.chunk_shapes(scfg, ch), ds_ep.chunk_storage_specs(ch), ds_ep.leaf_dtype_tree(scfg, ch, "storage"))
    if kind == "grads":  # the expert-count leaves stay fp32 (ds_ep.scatter_tree)
      dts = jax.tree.map(lambda k: jnp.dtype(jnp.float32 if k else scfg.rs_dtype), ds_ep.count_mask(ch))
      return tree(ds_ep.chunk_shapes(scfg, ch), ds_ep.chunk_storage_specs(ch), dts)
    if kind == "gacc":
      return tree(ds_ep.acc_shapes(scfg, ch), ds_ep.acc_specs(scfg, ch), scfg.acc_dtype)
    if kind == "act":
      return sds((self.rows, self.row_len, cfg.emb_dim), cfg.dtype, P(ds_ep.FE, None, None))
    if kind == "tok":
      return sds((self.rows, self.row_len), jnp.int32, P(ds_ep.FE, None))
    if kind == "scalar_i":
      return sds((), jnp.int32, P())
    if kind == "scalar_f":
      return sds((), jnp.float32, P())
    if kind == "loss":
      return sds((scfg.fsdp * scfg.ep,), jnp.float32, P(ds_ep.FE))
    raise ValueError(kind)

  def _desc_avals(self, c, desc):
    """Abstract arguments of `progs[c].program(desc)`: each part's, in turn
    (the argument lists of `ChunkPrograms.part`)."""
    A, out = self._avals, []
    for kind, kw in desc:
      kw = dict(kw)
      pr = kw.get("peer", self.progs[c])
      ch = pr.ch
      s = ch.stage
      w, x, i32 = A(s, "compute", ch), A(s, "tok" if ch.first else "act"), A(s, "scalar_i")
      if kind == "F" and "self_leaves" in kw:
        idx = kw["self_leaves"]
        cl, sl = jax.tree.leaves(w), jax.tree.leaves(A(s, "storage", ch))
        out += [[cl[i] for i in range(len(cl)) if i not in idx], x, i32, [sl[i] for i in idx]]
      elif kind == "F":
        out += [w, x, i32]
      elif kind == "B":
        out += [w, x, tuple(pr.res_avals)]
        out += [pr.stash_avals[kw["stash"]]] if "stash" in kw else []
        out += [A(s, "act")] + ([A(s, "gacc", ch)] if kw.get("acc") else []) + [i32]
        out += [[A(s, "scalar_f")] * kw["rs_self"]] if "rs_self" in kw else []
      elif kind == "I":  # the split backward's halves (ds_ep.Split): I takes a B's arguments but the accumulator
        out += [w, x, tuple(pr.res_avals), A(s, "act"), i32]
      elif kind == "W" and ch.last:  # ... the split last chunk's W: the accumulator's deferred part, its leaves
        sp = pr.build_split_last(kw.get("acc", False))
        gl = jax.tree.leaves(A(s, "gacc", ch))
        out += [sp.crossing_avals, sp.kept_avals] + ([tuple(gl[i] for i in sp.w_leaves)] if kw.get("acc") else [])
      elif kind == "W":  # ... W the crossing set and the arguments of I it reads, then a B's accumulator and sumsqs
        sp = pr.build_split(kw.get("acc", False))
        out += [sp.crossing_avals, sp.kept_avals] + ([A(s, "gacc", ch)] if kw.get("acc") else [])
        out += [[A(s, "scalar_f")] * kw["rs_self"]] if "rs_self" in kw else []
      elif kind in ("A", "AI"):  # the fused last chunk, or split (AI: the accumulator's kept part, its leaves)
        out += [w, x, A(s, "tok")] + [A(s, "tok")] * bool(pr.mtp)
        g = A(s, "gacc", ch)
        if kind == "AI":
          gl = jax.tree.leaves(g)
          g = tuple(gl[i] for i in pr.build_split_last(kw.get("acc", False)).i_leaves)
        out += ([g, A(s, "loss")] if kw.get("acc") else []) + [i32]
      elif kind in ("G", "R"):
        lv = jax.tree.leaves(A(s, "storage" if kind == "G" else "gacc", kw["other"]))
        out.append([lv[i] for i in kw.get("leaves", range(len(lv)))])
      else:
        raise ValueError(f"no avals for part {kind!r}")
    return tuple(out)

  def precompile(self, threads=64, agree=None):
    """Lower every program, one at a time in job order in this thread,
    compile them concurrently (XLA compiles release the GIL), and swap the compiled
    executables in for the jitted functions.

    The per-microbatch programs are the distinct descriptors `dispatched`
    lists -- every mode, host variant and F&B pair the order uses, each once --
    and the stage programs of the update / reduce side.  Calling a
    `Compiled` costs the same dispatch as a jit call, and holding it
    sidesteps the jit / persistent caches re-keying a program whose lowered
    HLO is identical (observed: a few programs recompiled on first use).

    Only the compiles run in the thread pool, each as soon as its program is
    lowered: tracing is not thread-safe here (on 2 x TPU7x-4x4x4 one host
    even raised "Store occupied" in tokamax splash attention's custom VJP,
    jax 0.11.1).  tokamax's splash kernel keeps
    its mask tables in a functools.lru_cache, which two threads missing at
    once each fill with a copy of their own -- so a program traced beside
    another can hold two equal copies of a table where a program traced
    alone holds one: the same values, other constants, another executable
    (TPU v4: another fingerprint; on CPU most programs of a tokamax pipeline
    lowered to one of several HLOs from run to run,
    ds_ep_test.PrecompileIsDeterministic).  Across the processes of a stage
    that is another program at one launch, and the TPU runtime halts it ('an
    unexpected peer shows up in the launch group with a different launch
    id').  jax's splash kernel is a jit, which jax traces once per process
    whatever the threads; but a trace jax caches keeps the source locations
    of whichever program traced it first, and a Mosaic kernel carries them in
    its body (megablox's, lowered for TPU: their text differed from run to
    run too, their executables not).

    `program_prints` keeps (name, stage, lowered, executable) per program:
    the sha256 of its lowered text with the text's size and constant / func
    counts ({'sha', 'bytes', 'consts', 'funcs'}), and the sha256 of its
    executable's fingerprint, the runtime's launch-id seed (None off TPU:
    `executable_print`).  `agree(prints, what)`, if given, gets this
    process's [(name, stage, fingerprint, stats)] as soon as every program is
    lowered (what='lowered', the compiles running on) and once they are
    compiled ('executables'), before anything is installed: main passes
    `check_program_agreement` across processes, so that a stage whose
    processes hold different programs stops before its first launch, naming
    them.
    """
    import concurrent.futures as cf

    try:  # private: without it every program shares this thread's context, so each compiles here
      from jax._src.interpreters.mlir import _thread_local_context as mlir_ctx
    except ImportError:
      mlir_ctx = None
    jobs = []  # (fn, args, install(compiled), stage)
    A = self._avals

    def attr(obj, name):
      return lambda c: setattr(obj, name, c)

    # every distinct per-microbatch program the step loop dispatches (on this
    # process's stages)
    for key in dict.fromkeys(self.dispatched()):
      if not self._runs(self.place[key[0]]):
        continue

      def put(cmp, key=key):
        self._exe[key] = cmp

      jobs.append((self.progs[key[0]].program(key[1]), self._desc_avals(*key), put, self.place[key[0]]))
    for s, st in enumerate(self.stages):
      if not self._runs(s):
        continue
      chs = self.stage_chunks[s]
      params = [A(s, "storage", c) for c in chs]
      grads = [A(s, "grads", c) for c in chs]
      opt = {
          "mu": jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, st.adam["mu_dtype"], sharding=a.sharding), params),
          "nu": params,
          "count": A(s, "scalar_i"),
      }
      sqs = [A(s, "scalar_f") for _ in range(self.S)]
      if self.overlap:
        fs, npq = self.final_slots[s], self.n_partial_sq[s]

        def put_final(cmp, st=st, fs=fs, npq=npq):
          st._final[(fs, npq)] = cmp

        jobs.append((st.gather_first, (params,), attr(st, "gather_first"), s))
        if fs:
          jobs.append(
              (
                  st.reduce_final(fs, npq),
                  ([A(s, "gacc", chs[i]) for i in fs], [A(s, "scalar_f") for _ in range(npq)]),
                  put_final,
                  s,
              )
          )
        jobs.append((st.update_gather, (params, opt, grads, sqs, A(s, "scalar_f")), attr(st, "update_gather"), s))
      else:
        # every step gathers all the stage's compute weights; init_state the first ones
        jobs.append((st.gather, (params,), attr(st, "gather"), s))
        jobs.append((st.gather_first, (params,), attr(st, "gather_first"), s))
        jobs.append((st.reduce, ([A(s, "gacc", c) for c in chs],), attr(st, "reduce"), s))
        jobs.append((st.update, (params, opt, grads, sqs, A(s, "scalar_f")), attr(st, "update"), s))
    # the boundary copies' collective-permute programs (transport 'ppermute') of
    # this process's stages: each runs on both stages' devices
    for (a, b), ch in [*self.channels.items(), *self.sq_channels.items(), *self.tie_channels.items()]:
      if isinstance(ch, transfer.PermuteChannel) and (self._runs(a) or self._runs(b)):
        jobs.append((ch.prog, ch.lower_args(), attr(ch, "prog"), (a, b)))
    # MTP's tied embedding (`_tie`): the sum of its two gradients on chunk 0's stage and on the last chunk's
    if self.mtp:
      cfg = self.scfg.model
      for s, keep in ((self.place[0], True), (self.place[self.C - 1], False)):
        if not self._runs(s):
          continue
        emb = jax.ShapeDtypeStruct(
            (cfg.vocab_size, cfg.emb_dim), self.scfg.rs_dtype, sharding=NamedSharding(self.meshes[s], P(None, ds_ep.FE))
        )

        def put_tie(cmp, key=("tie", s, keep)):
          self._exe[key] = cmp

        jobs.append((self._tie_sum(s, keep), (emb, emb, A(s, "scalar_f")), put_tie, s))
    t0 = time.perf_counter()
    self.compiled = []  # (program name, Compiled), for mem_report
    self.compiled_stage = []  # the stage of each: an int, or (a, b) for a boundary copy over both
    self.compile_times = []  # (lower s, compile s) of each
    self.program_prints = []  # (name, stage, lowered, executable) of each
    names = [getattr(f, "__name__", "?") for f, _, _, _ in jobs]

    def compile_(lo):
      t1 = time.perf_counter()
      return lo.compile(), time.perf_counter() - t1

    # The first failure ends it, naming the program, as soon as it is seen
    # (a compile's before the next lowering): no waiting for the compiles
    # still running or queued (no `with`, whose exit would join them).  The
    # rest install in job order.
    ex = cf.ThreadPoolExecutor(threads)
    futs, done, lowered, t_lower = {}, [None] * len(jobs), [None] * len(jobs), [None] * len(jobs)

    def stop(i, e, what="compiling"):
      ex.shutdown(wait=False, cancel_futures=True)
      if i is not None:
        e.add_note(f"precompile: {what} {names[i]} (stage {jobs[i][3]}) failed")

    def collect(block):
      """The finished compiles' results (every one's, if `block`)."""
      for fu in cf.as_completed(list(futs)) if block else [fu for fu in futs if fu.done()]:
        i = futs.pop(fu)
        try:
          done[i] = fu.result()
        except BaseException as e:
          stop(i, e)
          raise

    if agree is not None:  # multi-controller JAX (main: check_program_agreement across processes)

      def agreed(what, prints):
        try:
          agree(prints, what)
        except BaseException as e:
          stop(None, e)
          raise

    for i, (f, a, _, _) in enumerate(jobs):
      collect(block=False)
      t1 = time.perf_counter()
      try:
        if mlir_ctx is not None:
          mlir_ctx.set_local(None)  # a context of its own: its compile thread is then its only user
        lo = f.lower(*a)
        lowered[i] = lowered_print(lo)  # before its compile reads the module in another thread
      except BaseException as e:
        stop(i, e, "lowering")
        raise
      t_lower[i] = time.perf_counter() - t1
      if mlir_ctx is None:  # the next lowering would share its context: no compile beside it
        try:
          done[i] = compile_(lo)
        except BaseException as e:
          stop(i, e)
          raise
      else:
        futs[ex.submit(compile_, lo)] = i
      del lo
    if agree is not None:
      agreed("lowered", [(n, s, lw["sha"], lw) for n, (_, _, _, s), lw in zip(names, jobs, lowered)])
    collect(block=True)
    ex.shutdown()
    execs = [executable_print(c) for c, _ in done]
    if agree is not None:
      agreed("executables", [(n, s, x, None) for n, (_, _, _, s), x in zip(names, jobs, execs) if x is not None])
    for n, (_, _, put, s), (c, tc), tl, lw, x in zip(names, jobs, done, t_lower, lowered, execs):
      self.compiled.append((n, c))
      self.compiled_stage.append(s)
      self.compile_times.append((tl, tc))
      self.program_prints.append((n, s, lw, x))
      put(c)
    return len(jobs), time.perf_counter() - t0

  def mem_report(self, state=None, top=8):
    """Compile-time memory of the largest programs, and the persistent state
    per device -- where HBM goes, before anything runs."""
    G = 2**30
    rows = []
    for name, c in getattr(self, "compiled", []):
      m = c.memory_analysis()
      if m is not None:
        rows.append((m.temp_size_in_bytes, m.argument_size_in_bytes, m.output_size_in_bytes, m.alias_size_in_bytes, name))
    lines = [
        f"  largest program temporaries (GiB per device): "
        + ", ".join(f"{n} {t / G:.2f}" for t, _, _, _, n in sorted(rows, reverse=True)[:top])
    ]
    if state:
      for label, tree in state.items():
        per = collections.Counter()
        for a in jax.tree.leaves(tree):
          for sh in getattr(a, "addressable_shards", []):
            per[sh.device.id] += sh.data.nbytes
        if per:
          lines.append(
              f"  {label:28s} per device GiB: max {max(per.values()) / G:6.2f}  "
              + " ".join(f"{d}:{v / G:.1f}" for d, v in sorted(per.items()))
          )
    return "\n".join(lines)

  def plan_summary(self):
    if not self.overlap:
      return "  overlap: off (stage-level gather / reduce programs)" + (
          "; --prefetch-pin: no forward gathers, nothing to pin" if self.prefetch_pin != "off" else ""
      )
    name = self.op_name
    pf = ", ".join(f"{name(j)}<-AG c{b}[{len(g)}]" for j, (b, g) in sorted(self.pf.items()))
    rs = ", ".join(f"{name(j)}<-RS c{c}[{len(g)}]" for j, (c, g) in sorted(self.rs_at.items()))
    if self.scfg.grad_reduce == "microbatch":
      rs = "every backward reduce-scatters its own microbatch gradient (grad_reduce=microbatch)"

    fin = "; ".join(
        f"s{s}: RS c{[self.stage_chunks[s][i].index for i in f]}" for s, f in enumerate(self.final_slots) if f
    )
    fin += "".join(f"; {name(j)}<-RS self" for j in sorted(self.rs_self))
    upd = ", ".join(f"s{s}: c{[self.stage_chunks[s][i].index for i in g]}" for s, g in enumerate(self.gather_slots))
    if self.pf_fallback:
      why = (
          "no earlier forward on the stage to prefetch in"
          if self.fsdp_scope == "step"
          else "no op of an earlier visit on the stage to gather in"
      )
      upd += f" (c{self.pf_fallback}: {why})"
    if self.self_prefetch:
      upd += f"; routed experts of c0 gathered inside {name(self.self_pf_at)}"
    if self.direct:
      upd += f"; routed experts of c{sorted(self.direct)} read in place (fsdp 1: no compute copy)"

    rel = ""
    if self.fsdp_scope == "repeat":
      rel = "\n           fsdp_scope repeat (weights per visit): released after " + ", ".join(
          f"{name(j)} c{list(cs)}" for j, cs in sorted(self.release_at.items())
      )
    pins = self.pin_summary()
    return f"  overlap: {pf}\n           {rs}\n           standalone: {fin}; " f"gathered in update: {upd}{rel}" + (
        f"\n{pins}" if pins else ""
    )

  def op_name(self, j):
    """order[j] as the summaries print it: '2F3', or '2F3s' in stash mode."""
    c, k, m = self.order[j]
    return f"{c}{k}{m}" + ("s" if (c, m) in self.modes and k in ("F", "B") else "")

  def peak_stash_gib(self, temps=False):
    """Peak live stash per stage (GiB per device), walking the order -- what
    the budget holds.  `temps`: plus what each stash backward holds in
    temporaries over a remat one (`sched.STASH_BWD_TEMP`), while its own set
    is still live: the bound on how much the step's peak grows over a remat
    step (schedule.py's stash note)."""
    size = lambda c, md: 0 if md == "r" else self.stash_bytes[md][c]
    temp = lambda c, md: 0 if md == "r" else sched.STASH_BWD_TEMP[md] * self.stash_bytes[md][c]
    return [b / 2**30 for b in sched.peak_bytes(self.order, self.place, self.modes, size, temp if temps else None)]

  def peak_wbuf_gib(self):
    """Peak live W buffers per stage (GiB per device), walking the order: a split chunk's I -- a split last chunk's F
    (AI) -- leaves one (`wbuf_bytes`: its crossing set and the arguments its W reads) and its W consumes it -- what
    zb_wbuf_cap bounds in number (sched.peak_wbufs)."""
    live, peak = [0] * self.S, [0] * self.S
    for c, k, _ in self.order:
      s = self.place[c]
      if k == "I" or (k == "F" and c == self.C - 1 and self.split_last):
        live[s] += self.wbuf_bytes[c]
        peak[s] = max(peak[s], live[s])
      elif k == "W":
        live[s] -= self.wbuf_bytes[c]
    return [b / 2**30 for b in peak]

  def zb_sims(self):
    """The microbatch loop simulated (sched.timed_simulate, this pipeline's program costs) for this order and, on
    the same placement, 1F1B's and breadth's -> {name: ms}: what the zero-bubble schedules are measured against."""
    C, M, place = self.C, self.M, self.place
    cost = sched.cost_table(self.costs, C)
    orders = {"order": self.order, "1f1b": sched.one_f_one_b(C, M, place), "breadth": sched.breadth_first(C, M, place)}
    return {
        name: round(sched.timed_simulate(o, place, cost, self.costs["xfer"], fuse_last=True)[0], 3)
        for name, o in orders.items()
    }

  def zb_record(self):
    """The run record's 'zb' block (main), under a split backward (None otherwise): the split chunks; each one's W
    buffer (GiB per microbatch per device) and what it holds -- the crossing set's bytes, the arguments of I its W
    reads (Split.kept) -- and the cut's size; the cap on pending W buffers; whether the step loop paces the I's (a
    TPU VM's own client, McJAX: `pace_i`; Pathways' is not paced); per stage the peak pending W buffers and their
    GiB, per chunk the peak residual sets; the I / W costs the schedule was priced with; and the simulated microbatch
    loop of this order against 1F1B's and breadth's (`zb_sims`)."""
    if not self.split_chunks:
      return None
    chunks = {}
    cost = sched.cost_table(self.costs, self.C)
    for c in self.split_chunks:
      sp = self.split_of(c)
      last = c == self.C - 1  # the split last chunk: AI (its F + I) and W
      chunks[str(c)] = {
          "wbuf_gib": self.wbuf_bytes[c] / 2**30,
          "crossing_bytes": sp.bytes,
          "kept_bytes": sp.kept_bytes,
          "kept": [list(k) for k in sp.kept],
          "crossing": len(sp.crossing_local),
          "stats": dict(sp.stats),
          "costs": (
              {"A": cost(c, "F"), "AI": cost(c, "F") + cost(c, "I"), "W": cost(c, "W")}
              if last
              else {k: self.costs.get(f"{k}{c}", self.costs.get(k)) for k in ("B", "I", "W")}
          ),
      }
      if last:
        chunks[str(c)]["deferred_leaves"] = len(sp.w_leaves)
    return {
        "schedule": self.schedule_name,
        "split_chunks": list(self.split_chunks),
        "split_last": self.split_last,
        "wbuf_cap": self.zb_wbuf_cap,
        "paced": self.pace_i,
        "chunks": chunks,
        "peak_wbufs": sched.peak_wbufs(self.order, self.place),
        "peak_wbuf_gib": self.peak_wbuf_gib(),
        "peak_residual_sets": sched.peak_residuals(self.order, self.C),
        "sim_ms": self.zb_sims(),
    }

  def dispatches(self):
    """-> (programs, cross-stage copies) one step enqueues, counted off the
    order and the plan: a program per op (a pair is one; the fused last
    chunk's no-op backward none), the stage programs, the activation /
    cotangent copies between stages and the sumsq hops before the update."""
    C, place = self.C, self.place
    ops = [op for op in self.ops() if op.desc is not None]
    progs = len(self.dispatched())
    if self.overlap:
      progs += self.S + sum(1 for f in self.final_slots if f)  # update_gather, reduce_final
    else:
      progs += 2 * self.S  # reduce, update
    copies = self.S * (self.S - 1)
    for op in ops:
      if op.kind == "F" and op.c != C - 1:
        copies += place[op.c + 1] != op.s
      if op.c != 0 and (op.kind in ("B", "I") or (op.kind == "F" and op.c == C - 1)):
        copies += place[op.c - 1] != op.s
    return progs, copies

  # ------------------------------------------------------------------ data
  def put_batch(self, tokens, targets):
    """tokens/targets: int32 [gbs, seq] host arrays -> per-microbatch device
    arrays, [rows, row_len] each: under context parallelism every sequence
    cut into its cp chunks, one row each (`cp_rows`).  -> (tokens on chunk
    0's stage, targets on the last chunk's), and with MTP the tokens on the
    last chunk's stage third (its loss embeds the next token)."""
    tok = P(ds_ep.FE, None)
    first, last = self.chunks[0].stage, self.chunks[-1].stage
    put = lambda a, s: [
        jax.device_put(self.cp_rows(a[m * self.mbs : (m + 1) * self.mbs]), NamedSharding(self.meshes[s], tok))
        if self._places(s)
        else None
        for m in range(self.M)
    ]
    return (put(tokens, first), put(targets, last)) + ((put(tokens, last),) if self.mtp else ())

  def cp_rows(self, a):
    """[n, seq, ...] -> [n x cp, seq / cp, ...]: row cp x i + r is chunk r of
    sequence i (the identity without context parallelism)."""
    a = np.asarray(a) if not isinstance(a, jax.Array) else a
    return a if self.cp == 1 else a.reshape(a.shape[0] * self.cp, self.row_len, *a.shape[2:])

  def synthetic_batch(self, seed=0):
    rng = np.random.default_rng(seed)
    V = self.scfg.model.vocab_size
    toks = rng.integers(0, V, size=(self.gbs, self.seq), dtype=np.int32)
    tgts = np.roll(toks, -1, axis=1)
    return self.put_batch(toks, tgts)

  def maxtext_tokens(self, impl=None):
    """MaxText's synthetic batch (dataset_type=synthetic), bit for bit: [gbs,
    seq + 1] tokens from randint(key 0) under the PRNG MaxText draws it with
    (`impl`, default MAXTEXT_BATCH_PRNG: threefry2x32, the same bits on every
    backend); inputs are tokens[:, :-1] and targets tokens[:, 1:]."""
    key = jax.random.key(0, impl=impl or MAXTEXT_BATCH_PRNG)
    return np.asarray(jax.random.randint(key, (self.gbs, self.seq + 1), 0, self.scfg.model.vocab_size, dtype=jnp.int32))

  def token_batch(self, tokens):
    """[gbs, seq + 1] integer tokens in [0, vocab) -> the batch: inputs
    tokens[:, :-1], targets tokens[:, 1:]."""
    tokens = np.asarray(tokens)
    if not np.issubdtype(tokens.dtype, np.integer):
      raise ValueError(f"want integer tokens, got {tokens.dtype}")
    if tokens.shape != (self.gbs, self.seq + 1):
      raise ValueError(f"want [gbs, seq + 1] = [{self.gbs}, {self.seq + 1}] tokens, got {list(tokens.shape)}")
    vocab = self.scfg.model.vocab_size
    if tokens.min() < 0 or tokens.max() >= vocab:
      raise ValueError(f"tokens outside [0, {vocab}): min {tokens.min()}, max {tokens.max()}")
    tokens = tokens.astype(np.int32)
    return self.put_batch(tokens[:, :-1], tokens[:, 1:])

  def chunk_trees_from(self, src):
    """Chunk-order weight trees from files src/chunk<c>.npz (src local or
    gs://; `write_chunk_files`' format: keys are '.'-joined tree paths,
    bfloat16 stored as uint16), for the chunks of this process's stages only
    (None for the rest) -> trees for `params_from_chunks`.  maxtext_parity.py
    writes them from a MaxText init, next to the MaxText checkpoint of the
    same weights."""
    trees = []
    for ch in self.chunks:
      if not self._places(ch.stage):
        trees.append(None)
        continue
      tree = {}
      with np.load(io.BytesIO(xprof_align.read_bytes(f'{src.rstrip("/")}/chunk{ch.index}.npz'))) as z:
        for k in z.files:
          v = z[k]
          v = v.view(jnp.bfloat16) if v.dtype == np.uint16 else v
          node = tree
          *path, leaf = k.split(".")
          for q in path:
            node = node.setdefault(q, {})
          node[leaf] = v
      trees.append(tree)
    return trees

  def init(self, key):
    """-> (params, opts, wc0): wc0[s] lists stage s's pre-gathered chunks (gather_slots).

    Chunk c's weights come from fold_in(key, c) under `init_by_chunk` -- the
    same whatever the placement, so a circular and a V pipeline start from
    one model -- and otherwise from fold_in(fold_in(key, stage), c), as they
    always have."""
    # MTP: chunk 0's token embedding and the last chunk's copy, one table
    embed_key = jax.random.fold_in(key, 0x7AB1E) if self.mtp else None
    params = [
        st.init(key if self.init_by_chunk else jax.random.fold_in(key, s), embed_key=embed_key)
        if self._places(s)
        else None
        for s, st in enumerate(self.stages)
    ]
    return self.init_state(params)

  def init_state(self, params):
    """Storage params per stage -> (params, fresh optimizer state, wc0);
    None for a stage another process runs."""
    opts = [None if p is None else st.init_opt(p) for st, p in zip(self.stages, params)]
    wc0 = [None if p is None else st.gather_first(p) for st, p in zip(self.stages, params)]
    return params, opts, wc0

  def chunk_trees(self, params):
    """Per-stage params (or grads) -> one tree per chunk, in chunk order
    (None for the chunks of a stage another process runs)."""
    return [None if params[ch.stage] is None else params[ch.stage][self.slot[ch.index]] for ch in self.chunks]

  def params_from_chunks(self, trees):
    """Chunk-order trees (host arrays, or another pipeline's `chunk_trees`,
    whatever its placement) -> per-stage storage params of this pipeline:
    the weight transplant between placements and schedules."""

    def put(a, sh, dt):
      if isinstance(a, jax.Array):
        return jax.device_put(a if a.dtype == dt else a.astype(dt), sh)
      return jax.device_put(np.asarray(a).astype(dt), sh)  # no stop on a default device

    def check(ch, t):
      # the model's leaves and shapes, before any copy: a file of another
      # model must not load (device_put takes any divisible shape)
      flat = lambda tree: {
          jax.tree_util.keystr(k): tuple(np.shape(v)) for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]
      }
      got, want = flat(t), flat(self._avals(ch.stage, "storage", ch))
      missing, extra = sorted(want.keys() - got.keys()), sorted(got.keys() - want.keys())
      if missing or extra:
        raise ValueError(
            f"chunk {ch.index}: the weights are not this model's: missing {missing[:4]}, " f"unexpected {extra[:4]}"
        )
      bad = [(k, got[k], want[k]) for k in want if got[k] != want[k]]
      if bad:
        raise ValueError(
            f"chunk {ch.index}: weight shapes are not this model's: "
            + ", ".join(f"{k} {g} (model {w})" for k, g, w in bad[:4])
        )

    params = [[None] * len(self.stage_chunks[s]) if self._places(s) else None for s in range(self.S)]
    for ch, t in zip(self.chunks, trees, strict=True):
      s, i = ch.stage, self.slot[ch.index]
      if self._places(s):
        check(ch, t)
        params[s][i] = jax.tree.map(put, t, self.stages[s].s_sh[i], ds_ep.leaf_dtype_tree(self.scfg, ch, "storage"))
    return params

  def lr_arrays(self, lr):
    return [
        jax.device_put(jnp.float32(lr), NamedSharding(m, P())) if self._places(s) else None
        for s, m in enumerate(self.meshes)
    ]

  # ------------------------------------------------------------ one step
  def loss_and_grads(self, params, batch, wc=None):
    """Microbatch loop only.  Returns (gaccs per stage, loss partials).
    Under shard_mode 'explicit' the gaccs are typed unreduced: no host reads
    or scales one (jax refuses); a stage's `reduce` sums them."""
    if wc is None:
      wc = [self.stages[s].gather(params[s]) if self._runs(s) else None for s in range(self.S)]
    st = self._run(params, [None if w is None else list(w) for w in wc], batch, overlap=False)
    return [[st.gacc[ch.index] for ch in self.stage_chunks[s]] if self._runs(s) else None for s in range(self.S)], st.lacc

  def loss_and_grads_overlap(self, params, wc0, batch, consume=False):
    """Microbatch loop with the FSDP gather / reduce-scatter folded in.

    Returns (grads per stage [storage layout, by slot], stage sumsq, loss partials).
    `consume`: once the loop holds `wc0`'s weights, set its entries to None
    (a list's, in place), so the caller's reference does not keep them alive
    after the loop has released them (fsdp_scope 'repeat') or finished with
    them -- `step` consumes its wc0.
    """
    wc = [[None] * len(self.stage_chunks[s]) for s in range(self.S)]
    for s in range(self.S):
      for i, w in zip(self.gather_slots[s], wc0[s] or ()):
        wc[s][i] = self._whole(s, i, w, params)
    w = None  # else the loop's variable keeps the last chunk it took past its release
    if consume and isinstance(wc0, list):
      wc0[:] = [None] * len(wc0)
    st = self._run(params, wc, batch, overlap=True)
    wc[:] = [None] * self.S  # st.wc: the loop's weights, not needed by the final reduces below
    stage_sq = []
    for s in range(self.S):
      fs = self.final_slots[s]
      if self.multi and s not in self.local:
        stage_sq.append(None)
        continue
      if not fs:
        stage_sq.append(st.stage_tot[s])
        continue
      g, tot = self.stages[s].reduce_final(fs, self.n_partial_sq[s])(
          [st.gacc[self.stage_chunks[s][i].index] for i in fs], st.sq[s]
      )
      for i, gi in zip(fs, g):
        st.grads[s][i] = gi
      stage_sq.append(tot)
    return st.grads, stage_sq, st.lacc

  def _run(self, params, wc, batch, overlap):
    """THE step loop: every op of the order in turn (`issue_order`: each
    stage's trailing W's last), through its handler.

    `wc[s][slot]` holds the compute weights gathered so far (the rest arrive
    inside prefetch hosts when `overlap`).  Each handler takes the op's inputs
    from the step state, calls its program and files the outputs: the next
    chunk's input on its stage, the residuals (and stash) for the backward,
    the gradient accumulators, the gathered / reduced leaves of other chunks.
    An F&B pair is one call of both ops' programs composed (`_do_pair`).
    -> the step state.
    """
    st = _Step(self, params, wc, batch)
    handlers = {"F": self._do_F, "B": self._do_B, "I": self._do_I, "W": self._do_W}
    ops, C = self.ops(overlap), self.C

    skip = False
    for op in (ops[j] for j in self.issue_order(overlap)):
      if skip:  # the second half of a pair, run with the first
        skip = False
        continue
      if self.multi and op.s not in self.local:  # another process's stage: only its boundary copies
        self._do_remote(st, op, ops)
        skip = op.j in self.fuse
      elif op.j in self.fuse:
        self._do_pair(st, op, ops[op.j + 1])
        skip = True
      elif op.c == C - 1:
        if op.kind in ("F", "W"):  # the fused program (split: AI), or a split last chunk's W
          (self._do_A if op.kind == "F" else self._do_W)(st, op)
          if overlap:  # the plan's releases: only its gathers bring the weights back
            self._release(st, op)
        # its B is the no-op the fused program already ran
      else:
        handlers[op.kind](st, op)
        if overlap:
          self._release(st, op)
    if st.wbuf or st.gacc_parts:  # every I's W ran: the order is checked (sched.validate), so this is the loop's own
      raise AssertionError(
          f"W buffers {sorted(st.wbuf)} or accumulator parts {sorted(st.gacc_parts)} left after the loop"
      )
    st.last_i.clear()  # (`_pace`'s: each chunk's last crossing set, which its W has read, not held past the loop)
    return st

  # Each kind's handler is its two halves: `_take_K` pops the op's arguments
  # off the step state (in its part's argument order), `_put_K` files the
  # outputs of its part.  A pair runs one program on both argument lists.
  def _do_F(self, st, op):
    """A forward of a non-last chunk: -> the next chunk's input, the
    residuals (and the stash, in stash mode), and whatever it gathers."""
    args = self._take_F(st, op)
    self._put_F(st, op, args, list(self.program(op)(*args)))

  def _do_A(self, st, op):
    """The last chunk's forward: forward, head, loss and backward in one
    program -> the upstream cotangent, its gradient, the loss partials.  Split
    (AI): its gradient but its layers', and its W buffer for its W -- paced as
    an I is (`_do_I`)."""
    self._pace(st, op)
    args = self._take_A(st, op)
    self._put_A(st, op, args, list(self.program(op)(*args)))

  def _do_B(self, st, op):
    """A backward of a non-last chunk: -> the upstream cotangent (none for
    chunk 0) and its gradient; plus whatever it reduce-scatters (another
    chunk's groups, or with rs_self its own finished gradient)."""
    args = self._take_B(st, op)
    self._put_B(st, op, args, list(self.program(op)(*args)))

  def _do_pair(self, st, op_f, op_b):
    """An F&B pair: order[j] (an F, or the fused last chunk's A) and
    order[j+1] (a B of another microbatch on the same stage) as ONE program,
    `compose` of both parts (`op_f.pair`) on the two argument lists one after
    the other; its outputs are the B's, then the F's (ds_ep.composed_order;
    an A&B pair's, the A's, then the B's), filed as `_do_F` / `_do_A` and
    `_do_B` file theirs, the F's first: in the order `_do_remote` issues
    their boundary copies.  The plan folds no collective into a pair, so the
    B's are its dx (but chunk 0's) and its gacc."""
    last = op_f.c == self.C - 1
    fa = self._take_A(st, op_f) if last else self._take_F(st, op_f)
    ba = self._take_B(st, op_b)
    out = list(self.program(op_f)(*fa, *ba))
    nb = 1 + (op_b.c != 0)
    fwd_out, bwd_out = (out[:-nb], out[-nb:]) if last else (out[nb:], out[:nb])
    (self._put_A if last else self._put_F)(st, op_f, fa, fwd_out)
    self._put_B(st, op_b, ba, bwd_out)

  def _file(self, box, key, a, b, x):
    """File stage a's output x as box[key], the input of an op on stage b
    (`_send`) -- under multi-controller JAX where stage b is local."""
    y = self._send(a, b, x)
    if not self.multi or y is not None:
      box[key] = y

  def _do_remote(self, st, op, ops):
    """An op of a stage another process runs (multi-controller): no program,
    only the boundary copies its handler would issue, in the same order
    (`_put_F` / `_put_A` / `_put_B` / `_put_I`; an F&B pair's F, then its B),
    without a source -- filing whatever arrives for a local stage.  A W
    copies nothing."""
    C, place = self.C, self.place
    for o in (op, ops[op.j + 1]) if op.j in self.fuse else (op,):
      c, m, s = o.c, o.m, o.s
      if o.kind == "F" and c != C - 1:
        self._file(st.act, (c + 1, m), s, place[c + 1], None)
      elif c != 0 and ((o.kind in ("B", "I") and c != C - 1) or (o.kind == "F" and c == C - 1)):
        self._file(st.cotangents, (c - 1, m), s, place[c - 1], None)

  def _take_F(self, st, op):
    c, m, s = op.c, op.m, op.s
    x = st.toks[m] if c == 0 else st.act.pop((c, m))
    args = [st.wc[s][self.slot[c]], x, self.seeds[s][m]]
    if op.self_pf:  # w is the chunk's other leaves; gather its routed experts here
      sl = jax.tree.leaves(st.params[s][self.slot[c]])
      args.append([sl[i] for i in self.routed_leaves])
    return args + self._pf_args(st, op)

  def _pf_args(self, st, op):
    """The storage leaves of the gather `op` carries (its 'G' part's one
    argument), else nothing."""
    if not op.pf:
      return []
    b, g = op.pf
    sl = jax.tree.leaves(st.params[op.s][self.slot[b]])
    return [[sl[i] for i in g]]

  def _whole(self, s, i, w, params):
    """Stage s's slot-i entry of the gathered weights (`gather_first`'s or
    `update_gather`'s) as the programs take it: as given, but a chunk with
    leaves read in place (`direct`), whose gather delivered its other leaves
    as a flat list (StagePrograms.keep): those, with the direct ones from
    `params`, the storage leaves themselves -- no copy."""
    c = self.stage_chunks[s][i].index
    if c not in self.direct or w is None:
      return w
    full = list(jax.tree.leaves(params[s][i]))
    for j, v in zip(self.stages[s].keep[i], w, strict=True):
      full[j] = v
    return jax.tree.unflatten(self.treedef[c], full)

  def _install(self, st, op, leaves):
    """File the leaves `op`'s gather delivered; once the last of its hosts
    has, the chunk's weights are whole: install them in st.wc.  A chunk's
    leaves read in place (`direct`) are its storage's, there from the start."""
    b, g = op.pf
    parts = st.wparts.get(b)
    if parts is None:
      parts = st.wparts[b] = [None] * self.n_leaves[b]
      if b in self.direct:
        sl = jax.tree.leaves(st.params[op.s][self.slot[b]])
        for i in self.direct[b]:
          parts[i] = sl[i]
    for i, v in zip(g, leaves):
      parts[i] = v
    left = st.pending.get(b, self.pf_n[op.j]) - 1
    if left:
      st.pending[b] = left
    else:
      st.pending.pop(b, None)
      st.wc[op.s][self.slot[b]] = jax.tree.unflatten(self.treedef[b], st.wparts.pop(b))

  def _release(self, st, op):
    """fsdp_scope 'repeat': drop the weights of the chunks whose visit `op`
    ends (`release_at`).  The programs already issued keep what they use
    until they have run; nothing else holds them (`step` consumes wc0)."""
    for b in self.release_at.get(op.j, ()):
      st.wc[op.s][self.slot[b]] = None

  def _put_F(self, st, op, args, out):
    c, m, s = op.c, op.m, op.s
    slot = self.slot
    w, x = args[0], args[1]
    y, r = out.pop(0), out.pop(0)
    if op.stash:
      st.stash[(c, m)] = out.pop(0)
    if op.self_pf:
      full = [None] * self.n_leaves[c]
      for i, v in zip(self.first_leaves[s], w):
        full[i] = v
      for i, v in zip(self.routed_leaves, out.pop(0)):
        full[i] = v
      st.wc[s][slot[c]] = jax.tree.unflatten(self.treedef[c], full)
    if op.pf:
      self._install(st, op, out.pop(0))
    st.res[(c, m)] = self._offload((x, r)) if self.offload_residuals else (x, r)
    self._file(st.act, (c + 1, m), s, self.place[c + 1], y)

  @staticmethod
  def _offload(tree):
    """`tree`'s arrays copied to their shardings in pinned host memory
    (asynchronously, behind the program that wrote them) -> (the host copies,
    the device shardings to bring them back to)."""
    leaves, tdef = jax.tree.flatten(tree)
    shardings = [a.sharding for a in leaves]
    host = [jax.device_put(a, sh.with_memory_kind("pinned_host")) for a, sh in zip(leaves, shardings)]
    return ("offloaded", tdef, host, shardings)

  @staticmethod
  def _restore(held):
    """`_offload`'s copies back on their devices (asynchronously; the
    backward that takes them waits for the copy); anything else as it is."""
    if not (isinstance(held, tuple) and len(held) == 4 and held[0] == "offloaded"):
      return held
    _, tdef, host, shardings = held
    return jax.tree.unflatten(tdef, [jax.device_put(a, sh) for a, sh in zip(host, shardings)])

  def _take_A(self, st, op):
    c, m, s = op.c, op.m, op.s
    x = st.toks[m] if c == 0 else st.act.pop((c, m))
    args = [st.wc[s][self.slot[c]], x, st.tgts[m]] + ([st.mtp_toks[m]] if self.mtp else [])
    if op.acc:  # the accumulator -- split (AI), its kept part -- and the loss partials
      args += [st.gacc_parts[c][0] if self.split_last else st.gacc[c], st.lacc]
    return args + [self.seeds[s][m]] + self._pf_args(st, op)

  def _put_A(self, st, op, args, out):
    """File the fused last chunk's outputs: its accumulator (split: AI's part) and the loss partials, the upstream
    cotangent (but a first chunk's) -- and, split, its W buffer: the crossing set and the arguments its W reads."""
    c, m, s = op.c, op.m, op.s
    if op.pf:  # the gather's leaves come last
      self._install(st, op, out.pop())
    if self.split_last:  # AI: the accumulator's kept part and a W buffer, the crossing set after the loss
      cross = out.pop()
      nt = 1 + bool(self.mtp)  # the targets [, the tokens]
      i_args = args[: 2 + nt] + [args[2 + nt + 2 * op.acc]]  # (w, x, tgt[, tok], seed)
      st.wbuf[(c, m)] = (cross, self.split_of(c, op.acc).kept_of(i_args))
      if self.pace_i:
        st.last_i[c] = cross
      parts = st.gacc_parts.setdefault(c, [None, None])
      out, (parts[0], st.lacc) = out[:-2], out[-2:]
    elif c == 0:
      st.gacc[c], st.lacc = out
      out = []
    else:
      out, (st.gacc[c], st.lacc) = out[:-2], out[-2:]
    if c != 0:
      (dx,) = out
      self._file(st.cotangents, (c - 1, m), s, self.place[c - 1], dx)

  def _take_B(self, st, op):
    c, m, s = op.c, op.m, op.s
    x, r = self._restore(st.res.pop((c, m)))
    args = [st.wc[s][self.slot[c]], x, r]
    args += [st.stash.pop((c, m))] if op.stash else []
    args.append(st.cotangents.pop((c, m)))
    args += [st.gacc[c]] if op.acc else []
    args.append(self.seeds[s][m])
    return args + self._acc_tail_args(st, op) + self._pf_args(st, op)

  def _acc_tail_args(self, st, op):
    """A gradient writer's (a B's or a W's) arguments after its own part's leading ones: with rs_self the stage's
    partial sums of squares, and the leaves of another chunk's accumulator it reduce-scatters (op.rs, which it takes
    off the step state)."""
    args = []
    if op.rs_self is not None:
      args.append(st.sq[op.s])
    if op.rs:
      o, g = op.rs
      if o not in st.gacc_leaves:
        st.gacc_leaves[o] = jax.tree.leaves(st.gacc[o])
        st.gacc[o] = None
      gl = st.gacc_leaves[o]
      args.append([gl[i] for i in g])
      for i in g:
        gl[i] = None
    return args

  def _put_acc(self, st, op, out):
    """File a gradient writer's (a B's or a W's) outputs off the end of `out`: its gather's leaves, its
    reduce-scatter's gradients and partial sumsq, and its accumulator -- or with rs_self the chunk's reduced
    gradients and the stage's sumsq.  What is left in `out` is a B's dx (none for chunk 0, nor for a W)."""
    c, s = op.c, op.s
    if op.pf:  # the gather's leaves come last (after a reduce-scatter's)
      self._install(st, op, out.pop())
    if op.rs:
      o, g = op.rs
      got, part_sq = out[-2:]
      del out[-2:]
      st.sq[s].append(part_sq)
      parts = st.gparts.setdefault(o, [None] * self.n_leaves[o])
      for i, v in zip(g, got):
        parts[i] = v
      st.gpending[o] -= 1
      if not st.gpending[o]:
        st.grads[s][self.slot[o]] = jax.tree.unflatten(self.treedef[o], st.gparts.pop(o))
        del st.gacc_leaves[o]
    if op.rs_self is not None:
      st.grads[s][self.slot[c]], st.stage_tot[s] = out[-2:]
      del out[-2:]
      st.gacc[c] = None
    else:
      st.gacc[c] = out.pop()

  def _put_B(self, st, op, args, out):
    c, m, s = op.c, op.m, op.s
    self._put_acc(st, op, out)
    if c != 0:
      self._file(st.cotangents, (c - 1, m), s, self.place[c - 1], out[0])

  # The split backward (the zero-bubble schedules; ds_ep.ChunkPrograms.build_split): an I takes what its B would
  # but the accumulator, files the upstream cotangent and leaves its W buffer on the stage -- the crossing set it
  # returns and the references of the arguments W reads (Split.kept, which I does not donate: no copy) -- and its W
  # takes the buffer and is the chunk's gradient writer: the first W starts the accumulator, the others add to it in
  # microbatch order (sched.validate), as the B's they replace would, and a W hosts what a B hosts (a split last
  # chunk's W nothing: `_plan_overlap`).
  def _do_I(self, st, op):
    """The input-gradient half of a split backward: -> the upstream cotangent and the chunk's W buffer.  Issued once
    the chunk's previous I has finished, where the client allocates outputs at issue (`_pace`)."""
    self._pace(st, op)
    args = self._take_I(st, op)
    self._put_I(st, op, args, list(self.program(op)(*args)))

  def _pace(self, st, op):
    """Under McJAX (`pace_i`), wait for the chunk's previous W buffer this step to be written before issuing the
    next one's program: an I, or a split last chunk's AI.

    A TPU VM's client allocates a program's outputs when the one dispatch thread issues it, and the loop runs ahead
    of the chips, so the W buffers of I's issued ahead would be held from then on: the split stages' HBM fills
    whatever --zb-wbuf-cap says (87.5 GiB of 94.7 a device at DeepSeek-V3 widths on TPU7x, where the HBM walk says
    56), each I waits in Execute for a W to free some, and the stage's last W, which writes the reduce-scattered
    gradients into new buffers, finds none: libtpu waits for the stage to drain and defragments, and the copies out
    of the stage wait for that too.  Paced, an I is issued with the buffers its order holds pending and its own; the
    stage still has the ops between the two (a W, an F) to run meanwhile.  Pathways' client issues asynchronously
    and is not paced."""
    if self.pace_i:
      prev = st.last_i.pop(op.c, None)
      if prev is not None:
        jax.block_until_ready(prev)

  def _do_W(self, st, op):
    """The weight-gradient half of a split backward: its W buffer (and the accumulator) -> the accumulator, and
    whatever it reduce-scatters (another chunk's groups, or with rs_self its own finished gradient)."""
    args = self._take_W(st, op)
    self._put_W(st, op, args, list(self.program(op)(*args)))

  def _take_I(self, st, op):
    c, m, s = op.c, op.m, op.s
    x, r = self._restore(st.res.pop((c, m)))
    args = [st.wc[s][self.slot[c]], x, r, st.cotangents.pop((c, m)), self.seeds[s][m]]
    return args + self._pf_args(st, op)

  def _put_I(self, st, op, args, out):
    c, m, s = op.c, op.m, op.s
    if op.pf:  # the gather's leaves come last
      self._install(st, op, out.pop())
    dx, cross = out
    st.wbuf[(c, m)] = (cross, self.progs[c].build_split(op.acc).kept_of(args[:5]))
    if self.pace_i:
      st.last_i[c] = cross
    self._file(st.cotangents, (c - 1, m), s, self.place[c - 1], dx)

  def _take_W(self, st, op):
    c, m = op.c, op.m
    cross, kept = st.wbuf.pop((c, m))
    if op.acc:  # the accumulator -- a split last chunk's W, its deferred part
      args = [cross, kept, st.gacc_parts[c][1] if c == self.C - 1 else st.gacc[c]]
    else:
      args = [cross, kept]
    return args + self._acc_tail_args(st, op) + self._pf_args(st, op)

  def _put_W(self, st, op, args, out):
    """File a W's outputs: a middle chunk's as a B's (`_put_acc`); a split last chunk's deferred part, and after its
    last W the chunk's whole accumulator (`_merge_parts`)."""
    del args  # (the _put_K signature: a W files nothing of its arguments)
    c = op.c
    if c == self.C - 1:  # the split last chunk's deferred part; after its last W, its whole accumulator
      if op.pf:  # the gather's leaves come last
        self._install(st, op, out.pop())
      parts = st.gacc_parts.setdefault(c, [None, None])
      parts[1] = out.pop()
      if op.j == self.last_writer[c]:
        st.gacc[c] = self._merge_parts(c, *st.gacc_parts.pop(c))
    else:
      self._put_acc(st, op, out)
    if out:
      raise AssertionError(f"{self.op_name(op.j)}: {len(out)} outputs left after its accumulator")

  def _merge_parts(self, c, kept, deferred):
    """A split last chunk's accumulator from its two parts (ds_ep.Split.i_leaves / w_leaves): a regroup of the
    leaves, no device work."""
    sp = self.split_of(c)
    leaves = [None] * self.n_leaves[c]
    for idx, part in ((sp.i_leaves, kept), (sp.w_leaves, deferred)):
      for i, v in zip(idx, part, strict=True):
        leaves[i] = v
    return jax.tree.unflatten(self.treedef[c], leaves)

  def step(self, params, opts, wc0, batch, lrs):
    """One full training step -> (params, opts, wc0, loss_partials, gnorm).

    With the fusion plan, like params and opts (which the update programs
    donate), a list wc0 is used up: its entries are set to None as the
    microbatch loop takes them (`loss_and_grads_overlap(consume=True)`), so
    a chunk's gathered weights are freed when the loop is done with them --
    not held by the caller through the update, which gathers the next
    step's.  Without it (overlap=False) wc0 is not used: the loop gathers
    every chunk itself.

    Under multi-controller JAX the per-stage lists hold None for the stages
    another process runs, the loss partials are None unless the last chunk's
    stage is local, and gnorm comes from a local stage (None on a process
    that holds no stage).  `self.step_gnorms` keeps every local stage's
    gnorm, an output of its update program and never donated: blocking on
    them (and on the loss partials) waits for this process's whole step."""
    S = self.S
    if self.overlap:
      grads, sq, lacc = self.loss_and_grads_overlap(params, wc0, batch, consume=True)
    else:
      gaccs, lacc = self.loss_and_grads(params, batch)
      red = [self.stages[s].reduce(gaccs[s]) if self._runs(s) else (None, None) for s in range(S)]
      grads = [r[0] for r in red]
      sq = [r[1] for r in red]
    if self.mtp:
      grads, sq = self._tie(grads, sq)
    if self.on_grads is not None:
      self.on_grads(grads)
    new_p, new_o, new_w, gns, gn = [], [], [], [], None
    # Every hop goes out before any update, in the same order on every
    # process, so no update waits on another stage's: under 'ppermute' a hop
    # is a program of both its stages, which an update queued ahead of it on
    # one would hold back from the other, and under 'relay' a hop's receive
    # is a future its update resolves (`transfer.resolve` passes the other
    # transports' arrays through).
    if self.multi and self.norm_reduce == "allreduce":
      total = self._global_sumsq(sq)
      zeros = self._gsq["zeros"]
      hops = {(t, s): zeros.get(s) for s in range(S) for t in range(S) if t != s}
      sq = [total.get(s) for s in range(S)]
    else:
      hops = {(t, s): self._hop(t, s, sq[t]) for s in range(S) for t in range(S) if t != s}
    for s in range(S):
      mine = [hops[(t, s)] if t != s else sq[t] for t in range(S)]
      if self.multi:
        if s not in self.local:
          new_p.append(None)
          new_o.append(None)
          new_w.append(None)
          continue
        mine = [transfer.resolve(x) for x in mine]
      if self.overlap:
        p, o, g, w0 = self.stages[s].update_gather(params[s], opts[s], grads[s], mine, lrs[s])
      else:
        p, o, g = self.stages[s].update(params[s], opts[s], grads[s], mine, lrs[s])
        w0 = None
      new_p.append(p)
      new_o.append(o)
      new_w.append(w0)
      gns.append(g)
      gn = g if gn is None else gn
    self.step_gnorms = tuple(gns)
    return new_p, new_o, new_w, lacc, gn

  def _global_sumsq(self, sq):
    """norm_reduce 'allreduce': the stages' sums of squares (replicated
    scalars on their meshes, None for other processes' stages) -> {local stage:
    their total, replicated on its mesh}, by one program over every device of
    the job that every process runs at the same point of its step: each
    stage's first device contributes its stage's value, every other device 0.
    The update then takes [total, 0, ..., 0] for the S partials (`step`)."""
    if self._gsq is None:
      devs = jax.devices()
      gmesh = Mesh(np.array(devs), ("d",))
      first = {self.meshes[s].devices.flat[0].id: s for s in range(self.S)}
      # On GPU the sum stays a NCCL all-reduce: XLA's default for a clique
      # inside one NVLink domain is its Triton one-shot kernel, whose
      # compile fails at 64 ranks over the 16 hosts of a GB200 NVL72 rack
      # ("size mismatch when packing elements for LLVM struct", jax 0.12.0.1)
      # -- and a scalar once a step has nothing to gain from it.
      opts = {"xla_gpu_experimental_use_collective_kernels": ""} if devs[0].platform == "gpu" else None
      prog = jax.jit(lambda v: jnp.sum(v), out_shardings=NamedSharding(gmesh, P()), compiler_options=opts)
      zero = {d.id: jax.device_put(np.zeros((1,), np.float32), d) for d in jax.local_devices()}
      zeros = {s: jax.device_put(np.float32(0), NamedSharding(self.meshes[s], P())) for s in self.local}
      self._gsq = dict(mesh=gmesh, first=first, prog=prog, zero=zero, zeros=zeros)
    g = self._gsq
    parts = []
    for d in g["mesh"].local_devices:
      s = g["first"].get(d.id)
      if s is None:
        parts.append(g["zero"][d.id])
      else:
        (shard,) = [x.data for x in sq[s].addressable_shards if x.device == d]
        parts.append(shard.reshape(1))
    v = jax.make_array_from_single_device_arrays((len(g["mesh"].devices.flat),), NamedSharding(g["mesh"], P("d")), parts)
    tot = g["prog"](v)
    by_dev = {x.device.id: x.data for x in tot.addressable_shards}
    # this process's shards only: a stage may span processes (a one-stage, data-parallel layout over every node)
    return {
        s: jax.make_array_from_single_device_arrays(
            (), NamedSharding(self.meshes[s], P()), [by_dev[d.id] for d in self.meshes[s].devices.flat if d.id in by_dev]
        )
        for s in self.local
    }

  def _tie(self, grads, sq):
    """MTP's tied token embedding: chunk 0's table and the last chunk's copy
    (ds_ep.mtp_shapes) are one parameter, so each takes the sum of their two
    reduced gradients, and the global norm counts that sum once -- chunk 0's
    stage adds |sum|^2 - |own|^2 to its sum of squares, the last chunk's
    takes |own|^2 off.  Both then run the same update from the same state:
    the copies stay equal.  Every process walks both copies in order (the
    channels' contract); -> (grads, sq) with the local stages' replaced."""
    a, b, C = self.place[0], self.place[self.C - 1], self.C
    ia, ib = self.slot[0], self.slot[C - 1]
    get = lambda s, i, path: functools.reduce(lambda t, k: t[k], path, grads[s][i]) if self._runs(s) else None
    pa, pb = ("token_embedder", "embedding"), ("mtp", "token_embedder", "embedding")
    ga, gb = get(a, ia, pa), get(b, ib, pb)
    if a == b:
      other_a, other_b = gb, ga
    else:
      other_a = self.tie_channels[(b, a)].send_flat([gb])[0]
      other_b = self.tie_channels[(a, b)].send_flat([ga])[0]
    grads, sq = list(grads), list(sq)

    def put(s, i, path, g):
      tree = node = dict(grads[s][i])
      for k in path[:-1]:
        node[k] = dict(node[k])
        node = node[k]
      node[path[-1]] = g
      grads[s] = list(grads[s])
      grads[s][i] = tree

    for s, i, path, own, other, keep in ((a, ia, pa, ga, other_a, True), (b, ib, pb, gb, other_b, False)):
      if self._runs(s):
        g, sq[s] = self._tie_sum(s, keep)(own, transfer.resolve(other) if self.multi else other, sq[s])
        put(s, i, path, g)
    return grads, sq

  def _tie_sum(self, s, keep):
    """Stage s's program of `_tie`: (own, other, sumsq) -> (own + other, the
    corrected sumsq); `keep`: this stage's norm counts the sum (chunk 0's)."""
    key = ("tie", s, keep)
    if key not in self._exe:
      mesh, dt = self.meshes[s], self.scfg.rs_dtype
      emb = P(None, ds_ep.FE)

      def body(own, other, sq):
        g = (own.astype(jnp.float32) + other.astype(jnp.float32)).astype(dt)
        norm = lambda x: jax.lax.psum(jnp.sum(jnp.square(x.astype(jnp.float32))), ds_ep.FE)
        return g, sq - norm(own) + (norm(g) if keep else 0.0)

      ns = lambda spec: NamedSharding(mesh, spec)
      self._exe[key] = jax.jit(
          ds_ep._named(
              jax.shard_map(body, mesh=mesh, in_specs=(emb, emb, P()), out_specs=(emb, P()), check_vma=False),
              f"tie_s{s}" if keep else f"tie_mtp_s{s}",
          ),
          in_shardings=(ns(emb), ns(emb), ns(P())),
          out_shardings=(ns(emb), ns(P())),
      )
    return self._exe[key]

  def close(self):
    """Shut down what runs besides the programs: under transport 'relay'
    the relay's sockets and threads (`relay_ctx.close`, which flushes
    first).  Every receive must have been consumed -- after the last step,
    on every process.  Idempotent; nothing to do in one controller (no
    relay there)."""
    if self.multi and self.relay_ctx is not None:
      self.relay_ctx.close()

  # --------------------------------------------------------------- report
  def lineage_sparsecore(self):
    """{stage: does it run Lineage's SparseCore regions} under batch_split on Lineage's layers ({} otherwise): the
    answers its chunks' Regions trace under, every stage's (`__init__`: libtpu's, agreed across processes, the
    device rule's where a probe failed, or `stage_lineage_sparsecore`) -- read, never probed here.  A stage holding
    chip (0,0,0) of a slice it is part of says no: its Regions trace Lineage's SparseCore regions as TensorCore
    code."""
    return {} if self.lineage_sc is None else dict(enumerate(self.lineage_sc))

  def summary(self):
    cfg = self.scfg.model
    sync = (
        f"; synchronous on stage {sorted(self.sync_stages)} (its strip holds chip (0,0,0): no " f"SparseCore offload)"
        if self.sync_stages
        else ""
    )
    marks, remote = [""] * self.S, ""
    if self.multi or self.aot:
      # Another process's stage (multi-controller JAX; or a stage ds_ep_aot's --stages-to-compile leaves out) is
      # never probed here: its flavours are libtpu's rule from the devices (`sc`), marked '*'; its leader logs what
      # it resolved.
      other = frozenset(range(self.S)) - self.local
      if other:
        marks, remote = ["*" if s in other else "" for s in range(self.S)], " (* another process's stage: inferred)"
    flav = lambda kind: [getattr(c, kind) + marks[s] for s, c in enumerate(self.stage_scfg)]
    dense_at = " 0" if not self.every_layer else f"s {[c.index for c in self.chunks if c.dense]}"
    ep = f'EP collectives per stage: {flav("ep_collectives")}'
    if self.scfg.lineage_split:
      # every chunk's layers are Lineage's, which run EP collectives of their own: ds_ep's flavours go unused
      sc = self.lineage_sparsecore()
      on, off = sorted(s for s, v in sc.items() if v), sorted(s for s, v in sc.items() if not v)
      ep = (
          "EP collectives: Lineage's own ("
          + "; ".join(
              ([f"on SparseCore on stage {on}"] if on else [])
              + ([f"as TensorCore code on stage {off}, where libtpu offloads no collective"] if off else [])
          )
          + ")"
      )
    rows = self.rows // (self.scfg.fsdp * self.scfg.ep)
    if ds_ep.is_qwen(cfg):
      n_attn = sum(map(cfg.is_attention_layer, range(cfg.num_decoder_layers)))
      model = (
          f"MPMD Qwen3.5 family: {cfg.num_decoder_layers} MoE layers ({cfg.num_decoder_layers - n_attn} "
          f"Gated DeltaNet, {n_attn} gated attention)"
      )
    else:
      model = f"MPMD DeepSeek sparse: {cfg.num_decoder_layers - self.n_dense} MoE layers"
    lines = [
        f'{ep}; FSDP collectives per stage: {flav("fsdp_collectives")}{sync}{remote}',
        f"{model}"
        f'{f" after {self.n_dense} dense (mlp {cfg.mlp_dim}, in chunk{dense_at})" if self.n_dense else ""}'
        f'{f" + {self.mtp} MTP" if self.mtp else ""}, '
        f"{self.C} chunks "
        f'(V={self.V}{"" if len(set(self.layer_split)) == 1 else f", layers {list(self.layer_split)}"}), '
        f"{self.S} stages x (fsdp={self.scfg.fsdp}, ep={self.scfg.ep}), "
        f"M={self.M} x mbs={self.mbs} x seq={self.seq}"
        f'{f" (cp={self.cp}: {self.rows} rows of {self.row_len})" if self.cp > 1 else ""}, '
        f"schedule={self.schedule_name}"
        + (
            ", fsdp_scope=repeat (a chunk's weights gathered per visit, released after it)"
            if self.fsdp_scope == "repeat"
            else ""
        )
        + (
            ", shard_mode=explicit (Explicit mesh axes; weight copies reduced, gradient accumulators unreduced)"
            if self.scfg.shard_mode == "explicit"
            else ""
        )
        + (
            ", expert_weights=sliced (megablox reads per-layer slices, which XLA copies first)"
            if self.scfg.expert_weights == "sliced" and not self.scfg.lineage_split
            else ""
        )
        + (
            ", rs_axes=each (a leaf stored over both mesh axes reduce-scattered one axis at a time, but the "
            "vocab leaves)"
            if self.scfg.rs_axes == "each"
            else ""
        )
        + (
            ", gather_axes=each (a leaf stored over both mesh axes gathered one axis at a time)"
            if self.scfg.gather_axes == "each"
            else ""
        )
        + (
            f", batch_split=lineage (each device's {rows} rows of a microbatch as two halves through MaxText's "
            "Lineage layers, ds_ep_lineage: one half's EP collectives beside the other half's compute; the "
            f"chunks' programs global-view jits; capacity factor {self.scfg.lineage_capacity_factor:g})"
            if self.scfg.lineage_split
            else ""
        )
        + (
            f", batch_split=native (ds_ep.split_layers; each "
            f"device's {rows} rows of a microbatch as two halves, EP collectives beside the other half's "
            "compute; the halves' expert weight "
            + (
                "gradients accumulated in the dW kernel)"
                if self.scfg.split_dw == "acc"
                else "gradients summed after their kernels)"
            )
            if self.scfg.batch_split and not self.scfg.lineage_split
            else ""
        ),
    ]
    for s, m in enumerate(self.meshes):
      lines.append(f"  stage {s}: {describe_mesh(m)}  chunks {[c.index for c in self.stage_chunks[s]]}")
    nparam = ds_ep.num_params(self.scfg, self.chunks)
    progs, copies = self.dispatches()
    mk, busy = sched.timed_simulate(
        self.order,
        self.place,
        self.costs,
        self.costs["xfer"],
        fuse_last=True,
        modes=self.modes,
        fuse=self.fuse,
        pair_gain=self.pair_gain,
    )
    bub = sched.bubble(self.order, self.place, self.costs, modes=self.modes, fuse=self.fuse, pair_gain=self.pair_gain)
    lines.append(
        f"  params {nparam / 1e9:.2f}B; placement {self.placement}; bubble(sim F=1,B=2) "
        f"{100 * sched.bubble(self.order, self.place):.1f}%, with the program costs and "
        f"modes {100 * bub:.1f}%; microbatch loop {mk:.1f} (sim, stage busy "
        f"{[round(b, 1) for b in busy]}); dispatches/step {progs} programs + {copies} copies"
    )
    if self.split_chunks:
      sims = self.zb_sims()
      cap = "no cap" if self.zb_wbuf_cap is None else f"cap {self.zb_wbuf_cap}"
      lines.append(
          f"  split backward (I / W) on chunks {self.split_chunks}: W buffer "
          f"{[round(self.wbuf_bytes[c] / 2**30, 3) for c in self.split_chunks]} GiB per microbatch per device; peak "
          f"live W buffers per stage {sched.peak_wbufs(self.order, self.place)} ({cap}) = "
          f"{[round(g, 3) for g in self.peak_wbuf_gib()]} GiB per device; peak residual sets per chunk "
          f"{sched.peak_residuals(self.order, self.C)}; microbatch loop (sim) {sims['order']:.1f} against 1f1b "
          f"{sims['1f1b']:.1f} and breadth {sims['breadth']:.1f}"
      )
    if self.modes:
      nset = self.C - 1 if self.C > 1 else 0
      per = ", ".join(f"{pol} {max(b) / 2**30:.3f}" for pol, b in self.stash_bytes.items())
      lines.append(
          f"  stash: {len(self.modes)} of {nset * self.M} (chunk, microbatch)s "
          f"({per} GiB per set per device); peak live stash per stage (GiB per device) "
          f"{[round(g, 2) for g in self.peak_stash_gib()]}, budget {self.stash_budget_gib:g}; "
          f"with the stash backwards' extra temporaries the step peak grows by up to "
          f"{[round(g, 2) for g in self.peak_stash_gib(temps=True)]}"
      )
    if self.fuse:
      per = collections.Counter(self.place[self.order[j][0]] for j in self.fuse)
      lines.append(
          f"  F&B pairs: {len(self.fuse)} programs ("
          + ", ".join(f"s{s} {per[s]}" for s in range(self.S))
          + f"), each simulated {self.pair_gain:g} ms under its two halves back to back; "
          + (
              "no pair hosts an FSDP gather, and every backward, a pair's too, "
              "reduce-scatters its own microbatch gradient (grad_reduce=microbatch)"
              if self.scfg.grad_reduce == "microbatch"
              else "no pair hosts an FSDP collective"
          )
      )
    for s in range(self.S):
      names = [
          self.op_name(j) + ("&" if j in self.fuse else " ")
          for j, (c, _, _) in enumerate(self.order)
          if self.place[c] == s
      ]
      lines.append(f"  s{s}: " + "".join(names).rstrip())
    lines.append(self.plan_summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI
# `--model`: deepseek3 is the shape flags' own defaults (DeepSeek-V3 MoE layers); a
# Qwen3.5-family model, by MaxText's model_name, sets those defaults to its
# config (`ModelParser`), so `--model qwen3.8-2.4t-a95b` alone is the whole model
# and any shape flag given still wins (`--layers 8 --experts 64`: fewer layers
# and experts at its per-layer dims).  The DeepSeek-only flags (--q-lora,
# --kv-lora) default to None there and are refused if given.
QWEN_MODELS = {
    # MaxText's configs/models/qwen3.8-2.4t-a95b.yml: 2.42T parameters, 95.3B active.  It sets no use_mrope, so
    # base.yml's false: the partial RoPE (PartialRotaryEmbedding), not the Qwen3.5 ymls' text mRoPE.
    "qwen3.8-2.4t-a95b": dict(
        layers=92,
        emb=8192,
        heads=64,
        kv_heads=4,
        head_dim=256,
        moe_mlp=2048,
        experts=512,
        topk=10,
        vocab=248320,
        gdn_key_heads=16,
        gdn_value_heads=128,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        qwen_rope="partial",
    ),
    "qwen3.5-397b-a17b": dict(
        layers=60,
        emb=4096,
        heads=32,
        kv_heads=2,
        head_dim=256,
        moe_mlp=1024,
        experts=512,
        topk=10,
        vocab=248320,
        gdn_key_heads=16,
        gdn_value_heads=64,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        qwen_rope="mrope",
    ),
    "qwen3.5-35b-a3b": dict(
        layers=40,
        emb=2048,
        heads=16,
        kv_heads=2,
        head_dim=256,
        moe_mlp=512,
        experts=256,
        topk=8,
        vocab=248320,
        gdn_key_heads=16,
        gdn_value_heads=32,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        qwen_rope="mrope",
    ),
}
MODELS = ("deepseek3",) + tuple(QWEN_MODELS)
QWEN_ONLY_FLAGS = (
    "kv_heads",
    "head_dim",
    "gdn_key_heads",
    "gdn_value_heads",
    "gdn_key_head_dim",
    "gdn_value_head_dim",
    "qwen_rope",
)
DEEPSEEK_ONLY_FLAGS = ("q_lora", "kv_lora", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")


PLATFORMS = ("auto", "tpu", "gpu", "cpu")
# The flags whose parser defaults are TPU's kernels and collectives, and what a
# GPU run takes instead when it leaves them out (ModelParser): cuDNN's flash
# attention, tokamax's grouped matmuls on a compact local buffer (no MaxText
# SparseCore sort, no megablox), a gathered embedding, NCCL collectives.  CPU
# keeps TPU's defaults (the tests pin them).
GPU_DEFAULTS = {
    "attn": "cudnn",
    "gmm_impl": "tokamax",
    "ragged_sort": "off",
    "embed": "gather",
    "fsdp_collectives": "xla",
    "ep_collectives": "xla",
    "layer_tree": "per_layer",
    "norm_reduce": "allreduce",
    "gdn_conv": "shift",
    "gdn_scan": "triton",
    "direct_experts": "on",
    "combine_impl": "triton",
    "route_weights": "experts",
}


def resolve_platform(name="auto"):
  """--platform: 'tpu', 'gpu' or 'cpu' as given, else, without starting a jax
  backend (jax.distributed comes first): the first platform the process is
  pinned to (jax.config's jax_platforms, else JAX_PLATFORMS; cuda / rocm / gpu
  -> 'gpu'), else 'tpu' where libtpu is installed, 'gpu' where a jax CUDA /
  ROCm plugin is, 'cpu' otherwise."""
  if name != "auto":
    return name
  pinned = _config_platforms() or os.environ.get("JAX_PLATFORMS", "")
  first = str(pinned).split(",")[0].strip().lower()
  if first:
    return "gpu" if first in ("cuda", "rocm", "gpu") else first

  def has(mod):
    try:
      return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
      return False

  if has("libtpu"):
    return "tpu"
  if any(has(m) for m in ("jax_cuda13_plugin", "jax_cuda12_plugin", "jax_rocm7_plugin", "jax_rocm60_plugin")):
    return "gpu"
  return "cpu"


def _config_platforms():
  """jax.config's jax_platforms: what jax.config.update pinned, or JAX_PLATFORMS when jax was imported."""
  return getattr(jax.config, "jax_platforms", None)


class ModelParser(argparse.ArgumentParser):
  """The base of `make_parser`'s parser (`_Parser`): for each parse, `--model`
  picks the shape flags' defaults -- the flags' own for deepseek3, its
  QWEN_MODELS entry for a Qwen3.5-family model (with the DeepSeek-only flags at
  None) -- so every caller of `parse_args` gets the model's dims for the flags
  it does not give.  The model is the one the parse ends with, by argparse's
  own precedence: the arguments' --model, else a given `namespace`'s, else the
  default.  The defaults are the parser's own again once the parse returns."""

  def parse_known_args(self, args=None, namespace=None):
    args = sys.argv[1:] if args is None else list(args)  # read twice: a one-shot iterable once
    # Peek into a namespace holding only the given one's model and platform (no other attribute of the caller's to
    # touch): argparse's precedence -- the arguments', else the namespace's, else the default -- for both.
    given = {k: getattr(namespace, k) for k in ("model", "platform") if hasattr(namespace, k)}
    first = super().parse_known_args(args, argparse.Namespace(**given) if given else None)[0]
    model = first.model
    over = dict(QWEN_MODELS[model], **{k: None for k in DEEPSEEK_ONLY_FLAGS}) if model in QWEN_MODELS else {}
    # --platform: a GPU run's kernel and collective defaults (GPU_DEFAULTS)
    platform = resolve_platform(getattr(first, "platform", "auto"))
    if platform == "gpu":
      over.update(GPU_DEFAULTS)
    saved = {act: act.default for act in self._actions if act.dest in over}
    try:
      for act in saved:
        act.default = over[act.dest]
      return super().parse_known_args(args, namespace)
    finally:
      for act, default in saved.items():
        act.default = default


def resolve_inplace_dw(v, tokamax_gmm="mosaic"):
  """--inplace-dw auto -> 'all' where gpu_kernels.ragged_dw's tcgen05 kernel takes an fp32 accumulator (an sm_100
  device, MOSAIC_F32_ACC: 1.78 ms against 2.67 for tokamax's dW plus the add at DeepSeek-V3's EP 8 shapes) and
  runs it (--tokamax-gmm mosaic: ds_ep._ragged_dw gives the others ragged_dw's Triton kernel, 2.69 ms, or lax's
  form), else 'on'.  Every process of a run sees the same GPUs, so they all resolve it alike."""
  if v != "auto":
    return v
  if tokamax_gmm != "mosaic":
    return "on"
  try:
    from maxtext.experimental.mpmd.jax import gpu_kernels

    fast = jax.default_backend() == "gpu" and gpu_kernels.mosaic_supported() and gpu_kernels.MOSAIC_F32_ACC
  except Exception:  # pylint: disable=broad-except
    fast = False
  return "all" if fast else "on"


def _or_default(v, default):
  """A flag's value, or `default` where it was not given (None) -- an explicit 0 stays 0 instead of becoming the
  default (--qk-rope-head-dim 0 is MLA without RoPE; model_config refuses the head dims that leave no attention)."""
  return default if v is None else v


def model_config(a):
  """The model config the flags describe: a DeepSeekConfig, or a Qwen35Config under a Qwen3.5-family --model."""
  name = getattr(a, "model", "deepseek3")
  if name in QWEN_MODELS:
    return qwen_model_config(a)
  if name != "deepseek3":
    raise ValueError(f"--model {name!r}: want one of {MODELS}")
  given = [f'--{k.replace("_", "-")}' for k in QWEN_ONLY_FLAGS if getattr(a, k, None) is not None]
  if given:
    raise ValueError(f"{given}: Qwen3.5-family flags, but --model is deepseek3")
  dense, groups, topk_groups = (
      getattr(a, "dense_layers", 0),
      getattr(a, "routing_groups", -1),
      getattr(a, "topk_groups", -1),
  )
  bias, rate = getattr(a, "routed_bias", "off") == "on", getattr(a, "bias_update_rate", 0.0)
  if (groups > 0) != (topk_groups > 0):
    raise ValueError("--routing-groups and --topk-groups come together (MaxText n_routing_groups / topk_routing_group)")
  if groups > 0 and (a.experts % groups or topk_groups > groups or a.topk > topk_groups * (a.experts // groups)):
    raise ValueError(
        f"--routing-groups {groups} --topk-groups {topk_groups}: want groups dividing --experts "
        f"{a.experts}, at most {groups} groups kept, holding --topk {a.topk} experts"
    )
  if rate and not bias:
    raise ValueError("--bias-update-rate updates the routed bias: needs --routed-bias on")
  if getattr(a, "routing", "random") == "random" and (bias or groups > 0):
    raise ValueError(
        "--routed-bias / --routing-groups shape top-k routing: they need --routing topk (random routing "
        "ignores them, as MaxText use_random_routing does)"
    )
  base2 = getattr(a, "sa_base2_exp", None)  # None: tokamax's base-2, MaxText's default
  if base2 == "on" and a.attn != "tokamax_splash":
    raise ValueError(f"--sa-base2-exp on is tokamax_splash's base-2 softmax; --attn {a.attn} computes the exact one")
  base2 = base2 != "off"
  nope, rope, v = (
      _or_default(getattr(a, k, None), d)
      for k, d in (("qk_nope_head_dim", 128), ("qk_rope_head_dim", 64), ("v_head_dim", 128))
  )
  if min(nope, rope) < 0 or nope + rope < 1 or v < 1:
    # v 0: an empty attention output, every layer's attention adds zeros; nope + rope 0: no query / key channels
    raise ValueError(
        f"--qk-nope-head-dim {nope} --qk-rope-head-dim {rope} --v-head-dim {v}: want a value head dim "
        "of at least 1 and non-negative query / key head dims summing to at least 1 "
        "(--qk-rope-head-dim 0 is MLA without RoPE)"
    )
  dkv = getattr(a, "sa_block_dkv", None) or a.sa_block
  return DeepSeekConfig(
      vocab_size=a.vocab,
      emb_dim=a.emb,
      num_decoder_layers=a.layers,
      first_num_dense_layers=dense,
      num_query_heads=a.heads,
      num_kv_heads=a.heads,
      q_lora_rank=a.q_lora,
      kv_lora_rank=a.kv_lora,
      qk_nope_head_dim=nope,
      qk_rope_head_dim=rope,
      v_head_dim=v,
      mlp_dim=getattr(a, "mlp", None) or a.moe_mlp,
      moe_mlp_dim=a.moe_mlp,
      num_experts=a.experts,
      num_experts_per_tok=a.topk,
      shared_experts=1,
      routed_bias=bias,
      routed_bias_update_rate=rate,
      n_routing_groups=groups,
      topk_routing_group=topk_groups,
      mtp_num_layers=getattr(a, "mtp", 0),
      mtp_loss_scaling_factor=getattr(a, "mtp_loss_scale", 0.1),
      float32_gate_logits=getattr(a, "float32_gate_logits", "off") == "on",
      load_balance_loss_weight=getattr(a, "lb_loss_weight", 0.0),
      moe_use_megatron_seq_aux_loss=getattr(a, "lb_loss_weight", 0.0) > 0,
      attention_kernel=a.attn,
      sa_block_q=a.sa_block,
      sa_block_kv=a.sa_block,
      sa_block_kv_compute=a.sa_block_kv_compute,
      sa_block_q_dkv=dkv,
      sa_block_kv_dkv=dkv,
      sa_block_kv_dkv_compute=getattr(a, "sa_block_kv_dkv_compute", None) or dkv,
      sa_cost_estimate_flops_fwd=a.cost_estimate,
      sa_cost_estimate_flops_bwd=a.cost_estimate,
      sa_use_base2_exp=base2,
      max_target_length=a.seq,
      remat_policy="full",
      dtype=jnp.dtype(getattr(a, "dtype", "bfloat16")),
      weight_dtype=jnp.dtype(getattr(a, "weight_dtype", "float32")),
  )


def qwen_model_config(a) -> Qwen35Config:
  """A Qwen3.5-family `--model` as the maxtext.experimental.mpmd.jax.qwen35 config ds_ep's Qwen layers read:
  MaxText's qwen3_5 decoder (3 Gated DeltaNet layers, then one gated full-attention layer; every layer MoE with one
  gated shared expert), the dims from the flags (`ModelParser` defaults them to the model's).  --float32-gate-logits
  is MaxText's for qwen3_5 too: GateLogit's logits in fp32, from an fp32 compute copy of the router's kernel.  The
  model flags only the DeepSeek-V3 layers implement are refused: MLA's ranks, dense layers and routing groups, which
  MaxText's qwen3_5 does not have either (get_topk routes by groups only for the model names of
  common_types.DEEPSEEK_V3_ROUTING_MODEL_PREFIXES and GLM5's decoder block -- never a qwen3_5 one); the routed bias's
  update rate (MaxText balances loss-free in its DeepSeek decoder blocks only); and the routed bias, the aux loss and
  MTP, which this pipeline's Qwen layers do not implement (MaxText's qwen3_5 takes a routed bias and a load-balance
  loss, but --routed-bias is DeepSeek-V3's, which picks the experts without weighing them, and --lb-loss-weight its
  sequence-wise aux loss)."""
  given = [f'--{k.replace("_", "-")}' for k in DEEPSEEK_ONLY_FLAGS if getattr(a, k, None) is not None]
  given += [
      flag
      for flag, on in (
          ("--dense-layers", getattr(a, "dense_layers", 0)),
          ("--mlp", getattr(a, "mlp", None) is not None),
          ("--routed-bias", getattr(a, "routed_bias", "off") == "on"),
          ("--bias-update-rate", getattr(a, "bias_update_rate", 0.0)),
          ("--routing-groups", getattr(a, "routing_groups", -1) > 0),
          ("--topk-groups", getattr(a, "topk_groups", -1) > 0),
          ("--lb-loss-weight", getattr(a, "lb_loss_weight", 0.0)),
          ("--mtp", getattr(a, "mtp", 0)),
      )
      if on
  ]
  if given:
    raise ValueError(f"{given}: only the DeepSeek-V3 layers implement these, but --model is {a.model}")
  missing = [f'--{k.replace("_", "-")}' for k in QWEN_ONLY_FLAGS if getattr(a, k, None) is None]
  if missing:
    raise ValueError(f"--model {a.model}: {missing} unset (make_parser().parse_args defaults them to the model)")
  base2 = getattr(a, "sa_base2_exp", None)  # None: tokamax's base-2, MaxText's default
  if base2 == "on" and a.attn != "tokamax_splash":
    raise ValueError(f"--sa-base2-exp on is tokamax_splash's base-2 softmax; --attn {a.attn} computes the exact one")
  dkv = getattr(a, "sa_block_dkv", None) or a.sa_block
  rd = int(a.head_dim * Qwen35Config.partial_rotary_factor)
  # Text positions make every mRoPE section the same rotation; any split of rotary_dim // 2 into three
  # non-negative parts is valid: the most even one (qwen3.5's (11, 11, 10) at head_dim 256).
  q, r = divmod(rd // 2, 3)
  sections = tuple(q + (i < r) for i in range(3))
  return Qwen35Config(
      vocab_size=a.vocab,
      emb_dim=a.emb,
      num_decoder_layers=a.layers,
      num_query_heads=a.heads,
      num_kv_heads=a.kv_heads,
      head_dim=a.head_dim,
      mrope_section=sections,
      gdn_num_key_heads=a.gdn_key_heads,
      gdn_num_value_heads=a.gdn_value_heads,
      gdn_key_head_dim=a.gdn_key_head_dim,
      gdn_value_head_dim=a.gdn_value_head_dim,
      moe_mlp_dim=a.moe_mlp,
      num_experts=a.experts,
      num_experts_per_tok=a.topk,
      shared_experts=1,
      attention_kernel=a.attn,
      use_mrope=a.qwen_rope == "mrope",
      float32_gate_logits=getattr(a, "float32_gate_logits", "off") == "on",
      sa_block_q=a.sa_block,
      sa_block_kv=a.sa_block,
      sa_block_kv_compute=a.sa_block_kv_compute,
      sa_block_q_dkv=dkv,
      sa_block_kv_dkv=dkv,
      sa_block_kv_dkv_compute=getattr(a, "sa_block_kv_dkv_compute", None) or dkv,
      sa_cost_estimate_flops_fwd=a.cost_estimate,
      sa_cost_estimate_flops_bwd=a.cost_estimate,
      sa_use_base2_exp=base2 != "off",
      gdn_conv_impl=getattr(a, "gdn_conv", "conv"),
      gdn_scan_impl=getattr(a, "gdn_scan", "scan"),
      max_target_length=a.seq,
      dtype=jnp.dtype(getattr(a, "dtype", "bfloat16")),
      weight_dtype=jnp.dtype(getattr(a, "weight_dtype", "float32")),
  )


# Measured cost tables `--costs` can name (see schedule.py's measured costs):
# 'v6e' was measured with padded tiles (tk = D in every kernel), 'v6e_tiles' with `--tiles v6e`.
COST_TABLES = {"tpu7x": sched.TPU7X_COSTS, "v6e": sched.V6E_COSTS, "v6e_tiles": sched.V6E_TILES_COSTS}


def load_costs(spec):
  """`--costs`: a JSON object of program costs, `@path` to a file holding
  one (what `python3 -m maxtext.experimental.mpmd.jax.xprof costs DIR` prints), or the name
  of a table in COST_TABLES (to price a CPU dry run as the TPU would).
  Anything else -- a table name COST_TABLES does not have, say -- is a
  ValueError naming the flag and the tables."""
  if not spec:
    return None
  if spec in COST_TABLES:
    return dict(COST_TABLES[spec])
  if spec.startswith("@"):
    with open(spec[1:]) as f:
      spec = f.read()
  elif not spec.lstrip().startswith("{"):  # a JSON object is the one form left
    raise ValueError(f"--costs {spec!r}: not a JSON object, @file or one of the tables {', '.join(COST_TABLES)}")
  return {k: v for k, v in json.loads(spec).items() if not k.startswith("_")}  # '_source' etc.: notes


def split_costs(table, layer_split, remat_last=False):
  """A cost table of even chunks -> the same for chunks of `layer_split` layers.  Every key describes an even
  split's chunk, and each chunk but the last costs in proportion to its layers: its F / B -- the table's F{c} /
  B{c} where it has them (a per-chunk key is that chunk's own cost and wins over F / B), else chunk 0's F0 / B0,
  else F / B -- its split backward's I / W where measured (I{c} / W{c}, else I / W; sched.zb_costs fills in the
  others from the scaled B{c}), and its stash keys, the ones `sched.cost_table` reads for it, scaled as chunk c's
  own (a d-key stays one, added to the scaled F{c} / B{c}).  The last chunk's fused A keeps its head and norm and
  moves its layers' share: B a chunk's worth, as its backward recomputes nothing (StageConfig.remat_fused_last
  False), or F + B under `remat_last`; the head's share (A less an even chunk's layers) is held at 0 or more, so a
  table whose A undercuts its layers never prices the last chunk below its layers.  Split (--zb-split-last), its W
  -- its layers' weight gradients, W{C-1} or WA where measured, else A - AI from a measured AI -- scales with its
  layers, and a measured AI moves as A does less what W takes of the move."""
  if len(set(layer_split)) == 1:
    return table
  n = sum(layer_split) / len(layer_split)
  out = dict(table)
  pols = sorted(
      {
          k.split(":", 1)[1]
          for k in table
          if ":" in k and k.split(":", 1)[0].lstrip("d").rstrip("0123456789") in ("Fs", "Bs")
      }
  )
  for c, k in enumerate(layer_split[:-1]):
    for kind in ("F", "B"):
      base = table.get(f"{kind}{c}", table.get(f"{kind}0" if c == 0 else kind, table[kind]))
      out[f"{kind}{c}"] = base * k / n
      s = f"{kind}s"
      for names in [[f"{s}{c}:{p}", f"d{s}{c}:{p}", f"{s}:{p}", f"d{s}:{p}"] for p in pols] + [[f"{s}{c}", s]]:
        have = next((x for x in names if x in table), None)  # the key cost_table would read for chunk c
        if have is not None:
          out[names[1] if have.startswith("d") else names[0]] = table[have] * k / n
    for kind in ("I", "W") if c else ():  # a middle chunk's measured split (chunk 0's backward never splits)
      base = table.get(f"{kind}{c}", table.get(kind))
      if base is not None:
        out[f"{kind}{c}"] = base * k / n
  L, r = len(layer_split) - 1, layer_split[-1] / n
  layers = table["B"] + (table["F"] if remat_last else 0.0)  # an even fused chunk's layers
  out["A"] = max(table["A"] - layers, 0.0) + layers * r
  w_key = next((x for x in (f"W{L}", "WA") if x in table), None)  # the split last chunk's W, where measured
  w = table[w_key] if w_key else table["A"] - table["AI"] if "AI" in table else None
  if w_key:
    out[w_key] = w * r
  if "AI" in table:
    out["AI"] = table["AI"] + (out["A"] - table["A"]) - (w * r - w)
  return out


def default_costs(device_kind, scfg):
  """The program costs `--schedule timed / stash / interleaved` start from
  (any `--costs` key overrides them): TPU7x ones (TPU7X_COSTS) anywhere but
  a v6e, and on a v6e the medians measured with the megablox tiles `scfg`
  runs (ds_ep.TILES):

    v6e        V6E_TILES_COSTS, measured with them.
    clamped    V6E_TILES_COSTS, though never measured: the schedulers read
               only the costs' ratios, and what skews V6E_COSTS' ratios is the
               padding -- the padded kernels are one of a forward's three
               grouped matmuls but three of a backward's six, so its B/F is
               2.67 against the unpadded 2.37.  clamped pads nothing, like the
               v6e tiles; it differs from them in tm / tn, which every kernel
               shares (config A: 1447 ms/step clamped, 1328 v6e, 1607 padded).
               Its stash deltas were measured unpadded too.
    other      V6E_COSTS, measured with padded tiles (any other tiling).

  Under batch split's split_dw 'acc' the presets count with their wi dW tm
  capped at 256, as `build` and StageConfig cap them (split_acc_wi_tiling).
  """
  if "v6" not in device_kind.lower():
    return sched.TPU7X_COSTS
  tiling = (tuple(scfg.wi_tiling), tuple(scfg.wo_tiling))
  presets = [ds_ep.TILES[k](scfg.model) for k in ("v6e", "clamped")]
  if scfg.batch_split and scfg.batch_split_impl == "native" and scfg.split_dw == "acc":  # (split_acc_wi_tiling)
    presets = [(ds_ep.split_acc_wi_tiling(wi), tuple(wo)) for wi, wo in presets]
  if tiling in [(tuple(wi), tuple(wo)) for wi, wo in presets]:
    return sched.V6E_TILES_COSTS
  return sched.V6E_COSTS


def resolve_tilings(a, cfg: DeepSeekConfig | Qwen35Config):
  """(wi_tiling, wo_tiling): the --tiles preset (ds_ep.TILES; wi's dW tm
  capped at 256 under --batch-split on ds_ep.split_layers -- --batch-split-impl
  native, a Qwen3.5-family model's default -- with --split-dw acc,
  ds_ep.split_acc_wi_tiling), then --wi-tiling / --wo-tiling.

  Attributes a hand-built namespace lacks fall back to the defaults.
  """
  wi, wo = ds_ep.TILES[getattr(a, "tiles", None) or "clamped"](cfg)
  if getattr(a, "batch_split", False) and batch_split_impl(a) == "native" and getattr(a, "split_dw", "acc") == "acc":
    wi = ds_ep.split_acc_wi_tiling(wi)
  if getattr(a, "wi_tiling", None):
    wi = tuple(a.wi_tiling)
  if getattr(a, "wo_tiling", None):
    wo = tuple(a.wo_tiling)
  return tuple(wi), tuple(wo)


def native_transport_aborts(device_kind):
  """Does jax's cross-host copy (--transport native under --distributed)
  abort every receiver on this device kind?  On TPU v6e and v5p with jax
  0.11.1 / libtpu 0.0.46 it does (a v6e-256 and a v5p-128, 2026-10:
  std::bad_alloc in the receive); libtpu reports v5p as 'TPU v5' (Pathways:
  'TPU v5p'), v5e as 'TPU v5 lite'."""
  kind = str(device_kind).lower()
  return "v6" in kind or kind in ("tpu v5", "tpu v5p")


def float32_ragged_sort_problem(a, scfg):
  """The message to stop with if the stage config `scfg` (from the parsed
  flags `a`) computes in float32 with the ragged sort where MaxText's
  ragged_gather_reduce (maxtext/kernels/ragged/ragged_gather_reduce_v2.py)
  runs its JAX fallback -- off TPU (GPU, CPU), or on a TPU without
  SparseCores, as that function chooses -- else None.  The fallback returns
  bfloat16 whatever it is given, so the sort's backward hands a bfloat16
  gradient to a float32 activation, and the first backward program fails to
  lower on a bare AssertionError from jax's autodiff.  DeepSeek-V3's layers
  and a Qwen3.5-family model's run the same dispatch; Lineage's (--batch-split
  on them) run their own, never the ragged sort."""
  if not scfg.use_ragged_sort or scfg.lineage_split or jnp.dtype(scfg.model.dtype) == jnp.bfloat16:
    return None
  from jax.experimental.pallas import tpu as pltpu

  if pltpu.is_tpu_device() and pltpu.get_tpu_info().sparse_core is not None:
    return None
  return (
      f"--dtype {a.dtype} with the ragged sort (--ragged-sort {a.ragged_sort} at --ep {a.ep}) needs a TPU with "
      f"SparseCores, not {jax.devices()[0].device_kind}: elsewhere MaxText's ragged_gather_reduce returns "
      "bfloat16 and the backward fails; add --ragged-sort off (the argsort dispatch, float32 anywhere)"
  )


def qwen_model(a):
  """Do the flags `a` name a Qwen3.5-family --model (QWEN_MODELS), whose layers Lineage does not build?"""
  return getattr(a, "model", "deepseek3") in QWEN_MODELS


def on_gpu(a):
  """Do the flags `a` run on GPU (--platform gpu, or auto resolving to it: `resolve_platform`)?"""
  return resolve_platform(getattr(a, "platform", "auto")) == "gpu"


def model_batch_split_impl(a):
  """The --batch-split-impl the flags' --model makes the default: lineage for deepseek3, native (ds_ep.split_layers)
  for a Qwen3.5-family model, whose layers Lineage does not build -- as ds_ep.StageConfig resolves batch_split_impl
  None by its model -- and for any model on GPU (`on_gpu`), where Lineage's kernels (Pallas TPU splash and gmmv2,
  SparseCore dispatch and combine) do not run and split_layers runs the GPU kernels."""
  return "native" if qwen_model(a) or on_gpu(a) else "lineage"


def batch_split_impl(a):
  """--batch-split-impl of the flags `a`, else (None: left out, or not in a namespace made by hand) the model's
  (`model_batch_split_impl`)."""
  impl = getattr(a, "batch_split_impl", None)
  return model_batch_split_impl(a) if impl is None else impl


def uses_lineage(a):
  """Do the flags `a` batch-split on Lineage's layers (--batch-split, --batch-split-impl lineage: DeepSeek-V3's
  default)?"""
  return bool(getattr(a, "batch_split", False)) and batch_split_impl(a) == "lineage"


# The flags that pick among ds_ep's own layer kernels, which --batch-split on Lineage's layers does not run
# (ds_ep_lineage.IGNORED_STAGE_KNOBS / IGNORED_MODEL_FIELDS): the routed experts' grouped matmuls and their tilings,
# the dispatch, the EP collectives, the halves' dW combine, the experts' weight reads, where the routing weights
# apply, the attention kernel, its softmax (Lineage's splash is base-2 always: --sa-base2-exp on is what it computes
# anyway) and its cost estimate.
LINEAGE_IGNORED_FLAGS = (
    "gmm_impl",
    "tiles",
    "wi_tiling",
    "wo_tiling",
    "strict_tiling",
    "ragged_sort",
    "ep_collectives",
    "split_dw",
    "expert_weights",
    "route_weights",
    "attn",
    "sa_base2_exp",
    "cost_estimate",
)


def lineage_ignored(a):
  """The LINEAGE_IGNORED_FLAGS `a` sets away from their defaults (--sa-base2-exp: off) where --batch-split runs
  Lineage's layers -> ['--flag value', ...]; [] under any other flags, and for a Qwen3.5-family model (which `build`
  refuses there).  Accepted, as they still mean what they say to an unsplit run of the same flags; main warns of
  each."""
  if not uses_lineage(a) or qwen_model(a):
    return []
  ap, out = make_parser(), []
  for k in LINEAGE_IGNORED_FLAGS:
    d = ap.get_default(k)
    v = getattr(a, k, d)
    if v == d or (k == "sa_base2_exp" and v != "off"):
      continue
    flag = "--" + k.replace("_", "-")
    out.append(flag if v is True else f"{flag} {' '.join(map(str, v)) if isinstance(v, (list, tuple)) else v}")
  return out


def lineage_unused(a, given=()):
  """The other direction: the flags only --batch-split on Lineage's layers reads, given where it does not run (not
  `uses_lineage`) -> ['--flag value: why', ...]; [] under any other flags.  Accepted; main warns of each.  Given:
  set away from its default, or named in `given`, the flags (dests) a caller knows were given -- ds_ep_parity's
  --lineage-capacity-factor, whose default None tells it apart; main's flags cannot be told apart at their defaults
  (--lineage-capacity-factor 2, --batch-split-impl the model's: lineage, or native for a Qwen3.5-family model)."""
  if uses_lineage(a):
    return []
  split, impl, model_impl = bool(getattr(a, "batch_split", False)), batch_split_impl(a), model_batch_split_impl(a)
  out, d = [], make_parser().get_default("lineage_capacity_factor")
  if "lineage_capacity_factor" in given or getattr(a, "lineage_capacity_factor", d) != d:
    # split, not on Lineage's layers: on split_layers, given (DeepSeek-V3) or the model's default (the Qwen3.5 family)
    why = (
        "without --batch-split"
        if not split
        else f"by --batch-split-impl {impl}"
        if impl != model_impl
        else f"by --model {a.model}'s batch split, ds_ep.split_layers"
    )
    out.append(
        f"--lineage-capacity-factor {a.lineage_capacity_factor:g}: not used {why} (Lineage's MoE chunk size: "
        "only --batch-split on Lineage's layers reads it)"
    )
  if not split and ("batch_split_impl" in given or impl != model_impl):
    out.append(f"--batch-split-impl {impl}: not used without --batch-split (it picks the batch " "split's layers)")
  return out


def lineage_partial(a):
  """The flags `a` gives that --batch-split on Lineage's layers applies in part only -> ['--flag value: how', ...];
  [] under any other flags, and for a Qwen3.5-family model (which `build` refuses there).  Not
  LINEAGE_IGNORED_FLAGS (they still change what runs); main warns of each."""
  if not uses_lineage(a) or qwen_model(a) or getattr(a, "float32_gate_logits", "off") != "on":
    return []
  return [
      "--float32-gate-logits on: on Lineage's layers only the expert selection runs in fp32 (Lineage's "
      "router_dtype): the router kernel's fp32 compute copy is rounded to bf16 first, and its gradient comes back "
      "in bf16 (bf16 operands, fp32 accumulation)"
  ]


def split_defaults(a):
  """The flags whose default --batch-split decides, filled in where left out (None) -> `a`, in place.  First
  --batch-split-impl, by the model (`batch_split_impl`): lineage for DeepSeek-V3, native for a Qwen3.5-family
  model, whose layers Lineage does not build.  On Lineage's layers (`uses_lineage`) --routing is topk -- they route
  top-k only, there is no RNG in them -- and --shard-mode explicit: their code reads its layouts from sharding
  types, so the chunks' programs run at global view on Explicit stage meshes (typed_meshes;
  ds_ep.StageConfig.batch_split_impl).  Else random and auto, as ever (so under a Qwen3.5-family model's batch
  split too).  A flag given stands: one Lineage cannot run is refused by `build` (`lineage_problems`).
  make_parser's parser fills them in as it parses; `build` again for a namespace made by hand."""
  a.batch_split_impl = batch_split_impl(a)
  lineage = uses_lineage(a)
  if getattr(a, "routing", None) is None:
    a.routing = "topk" if lineage else "random"
  if getattr(a, "shard_mode", None) is None:
    a.shard_mode = "explicit" if lineage else "auto"
  return a


def lineage_problems(a, cfg: DeepSeekConfig | Qwen35Config):
  """What the flags `a` ask of --batch-split on Lineage's layers that they cannot run, named by flag, every one at
  once, the model's too (ds_ep_lineage.check of `cfg`, which `a` made) -> [str]; [] when they run.  StageConfig
  refuses the stage's settings itself (ds_ep.StageConfig._check_lineage; a one-device stage, which shard_mode
  explicit cannot run, apart), Pipeline the stash and pins, ChunkPrograms an odd number of rows a device and a chunk
  without a MoE layer: these say so before anything is built, in the flags' terms.  A Qwen3.5-family model is
  refused alone: Lineage builds none of its layers, so no other flag matters."""
  if ds_ep.is_qwen(cfg):
    return [
        f"--model {a.model}: Lineage builds DeepSeek-V3's layers only, none of a Qwen3.5-family model's (Gated "
        "DeltaNet, gated attention, the sigmoid-gated shared expert)"
    ]
  bad = []
  if a.routing == "random":
    bad.append(
        "--routing random: Lineage routes top-k only, there is no RNG in it (leave --routing out: topk under "
        "--batch-split)"
    )
  if a.shard_mode == "auto":
    bad.append(
        "--shard-mode auto: Lineage's code reads its layouts from sharding types, so the chunks' programs run "
        "at global view on Explicit stage meshes (leave --shard-mode out: explicit under --batch-split)"
    )
  if a.fsdp * a.ep == 1:  # (whatever --shard-mode says: Lineage's layers need explicit)
    bad.append(
        f"--fsdp {a.fsdp} --ep {a.ep}: a one-device stage, which the Explicit stage meshes Lineage's layers "
        "run on cannot be (fsdp x ep > 1: jax lowers a one-device stage's shard_maps inline, which refuses "
        "their unreduced outputs)"
    )
  if getattr(a, "grad_reduce", "step") == "microbatch":
    bad.append(
        "--grad-reduce microbatch: the programs add Lineage's per-device partial weight gradients into "
        "unreduced accumulators, reduce-scattered once a step (step)"
    )
  if getattr(a, "ragged_buffer_factor", -1.0) > 0:
    bad.append(
        f"--ragged-buffer-factor {a.ragged_buffer_factor:g}: Lineage is dropless, > 0 drops the tokens past "
        "the ring's capped dispatch buffer (<= 0)"
    )
  if getattr(a, "cp", 1) > 1:
    bad.append(
        f"--cp {a.cp}: the batch split halves the rows a device holds, context parallelism holds one (and "
        "Lineage's splash attention shards heads, never the sequence)"
    )
  stash = [f"--stash {a.stash}"] if getattr(a, "stash", "none") != "none" else []
  if a.schedule == "stash":
    stash.append("--schedule stash")
  if stash:
    bad.append(
        f"{' / '.join(stash)}: a chunk runs Lineage's layers in one custom VJP whose residuals are Lineage's "
        "own, which no stash policy names (--schedule search or timed search the order without stash)"
    )
  if ds_ep.parse_prefetch_pin(getattr(a, "prefetch_pin", "off")) != "off":  # as Pipeline reads it (None: off)
    bad.append(
        f"--prefetch-pin {a.prefetch_pin}: pins tie gathers to points of ds_ep's own layers (hn / xs / y), "
        "which Lineage's layers have none of"
    )
  if cfg.num_experts % a.ep:
    bad.append(
        f"--experts {cfg.num_experts} is not a multiple of --ep {a.ep}: Lineage splits the experts evenly " "over EP"
    )
  n_dev, rows = a.fsdp * a.ep, a.gbs // a.mb
  if getattr(a, "cp", 1) == 1 and a.gbs % a.mb == 0 and rows % n_dev == 0 and (rows // n_dev) % 2:  # (else Pipeline's)
    bad.append(
        f"--gbs {a.gbs} / --mb {a.mb}: {rows // n_dev} row(s) a device a microbatch (fsdp x ep "
        f"{n_dev}): the batch split runs each device's rows as two halves, so it wants an even number"
    )
  split, n_dense = parse_layer_split(getattr(a, "layer_split", "")), cfg.first_num_dense_layers
  if split is not None and n_dense and sum(split) == cfg.num_decoder_layers:  # every layer counted
    if dense_only := [c for c, end in enumerate(np.cumsum(split)) if end <= n_dense]:
      bad.append(
          f"--layer-split {a.layer_split}: chunk(s) {dense_only} hold dense layers only (it counts every "
          f"layer, the {n_dense} dense ones first): Lineage batch-splits MoE layers, so every chunk needs one"
      )
  try:
    ds_ep_lineage.check(cfg)
  except ValueError as e:
    bad.append(str(e))
  return bad


def lineage_missing(a):
  """Why --batch-split on Lineage's layers cannot run here -- Lineage is not importable: the ImportError's message
  (ds_ep_lineage.import_lineage) -- else None (also
  under any other flags, and for a Qwen3.5-family model, which `build` refuses on Lineage's layers whatever is
  installed).  The pipeline builds without it, its programs import it when they trace: main asks first."""
  if not uses_lineage(a) or qwen_model(a):
    return None
  try:
    ds_ep_lineage.import_lineage()
  except ImportError as e:
    return f"{e} (off TPU, --batch-split-impl native runs without it)"
  return None


def native_split_refusal(a, devices=None):
  """Why the flags `a` cannot batch-split DeepSeek-V3 on ds_ep.split_layers (--batch-split --batch-split-impl
  native) on `devices`, the devices the pipeline runs on (default jax.devices(), which starts the backend: build
  asks after main's init_distributed): on TPU devices -- a TPU VM's own, or Pathways' through the IFRT proxy, which
  report the platform tpu -- Lineage's layers run DeepSeek-V3's batch split.  Else None (also under any other
  flags; for a Qwen3.5-family model, whose only batch split split_layers is; and on other devices, where Lineage's
  TPU kernels do not run: split_layers is DeepSeek-V3's batch split on GPU, its default there, and on CPU).  The
  devices decide, not --platform, which picks the flags' defaults before any backend starts and resolves auto to
  tpu wherever libtpu is installed: a CPU process with libtpu installed (MaxText's CPU CI) runs on CPU devices.
  `build` refuses it."""
  if not getattr(a, "batch_split", False) or batch_split_impl(a) != "native" or qwen_model(a):
    return None
  if not any(_is_tpu(d) for d in (jax.devices() if devices is None else devices)):
    return None
  return (
      "--batch-split-impl native: on TPU DeepSeek-V3's batch split runs Lineage's layers (--batch-split-impl "
      "lineage, its default); ds_ep.split_layers batch-splits DeepSeek-V3 off TPU only (on GPU, its default "
      "there, and on CPU), and the Qwen3.5 family anywhere"
  )


def lineage_off_tpu(a, devices=None):
  """Why --batch-split on Lineage's layers cannot run on `devices` (default jax.devices(), which initializes the
  backend: main asks after init_distributed) -- they are not all TPU devices, and Lineage's kernels (Pallas TPU
  splash and gmmv2, SparseCore dispatch and combine) have no other path: the first program's trace would fail --
  else None (also under any other flags, and for a Qwen3.5-family model, which `build` refuses on Lineage's layers
  on any backend).  ds_ep_aot refuses a non-TPU backend itself."""
  if not uses_lineage(a) or qwen_model(a):
    return None
  devs = jax.devices() if devices is None else devices
  if not (off := sorted({str(getattr(d, "platform", "?")) for d in devs if not _is_tpu(d)})):
    return None
  return (
      f"Lineage's layers run TPU kernels only (Pallas splash and gmmv2, SparseCore dispatch and combine), and "
      f"this process's devices are {' / '.join(off)} ones (--batch-split-impl native runs here)"
  )


def ep_collectives_arg(s):
  """--ep-collectives: auto, xla or ppermute for every stage, or one of xla /
  ppermute per stage, comma-separated (e.g. ppermute,xla)."""
  if s in ds_ep.EP_COLLECTIVES or ("," in s and set(s.split(",")) <= {"xla", "ppermute"}):
    return s
  raise argparse.ArgumentTypeError(
      f"{s!r}: expected auto, xla, ppermute, or one of xla / ppermute per " "stage, comma-separated (e.g. ppermute,xla)"
  )


def prefetch_pin_arg(s):
  """--prefetch-pin: off, auto or a SPEC, checked for its form here
  (ds_ep.parse_prefetch_pin) and for its leaf names when the pipeline is
  built."""
  try:
    ds_ep.parse_prefetch_pin(s)
  except ValueError as e:
    raise argparse.ArgumentTypeError(str(e)) from None
  return s.strip()


def float_dtype_arg(s):
  """--rs-dtype, --acc-dtype, --weight-dtype: any name jnp.dtype reads (bfloat16, float32, f4, ...) of a floating
  dtype, kept as given.  An integer one would truncate every reduced gradient, accumulator or master weight to 0:
  a gradient norm of 0 and a loss that never moves, with no error."""
  try:
    dt = jnp.dtype(s)
  except TypeError:
    raise argparse.ArgumentTypeError(f"{s!r} is not a dtype: want a floating one, e.g. bfloat16 or float32") from None
  if not jnp.issubdtype(dt, jnp.floating):
    raise argparse.ArgumentTypeError(f"want a floating dtype, e.g. bfloat16 or float32, not {s!r}")
  return s


def resolved_config(a, p):
  """Write what pipeline `p` resolved back into its arguments `a`, so a
  run's recorded config (run.json, ds_ep_aot's --json) names what ran: the
  placement --placement defaulted to, the schedule --schedule did, the
  scope --fsdp-scope did."""
  a.placement, a.schedule, a.fsdp_scope = p.placement, p.schedule, p.fsdp_scope


def build(a, meshes=None, aot=False, stage_lineage_sparsecore=None, proxy=None, **scfg_overrides):
  """The Pipeline the flags `a` (make_parser's, or a namespace made by hand: missing flags take their defaults)
  describe, on `meshes` (default: stage_meshes of --layout; `aot`: a topology this process does not own; `proxy`:
  planned for one Pathways client, Pipeline's), with `scfg_overrides` replacing StageConfig fields and
  `stage_lineage_sparsecore` libtpu's per-stage answers for Lineage's SparseCore regions (Pipeline; ds_ep_aot's).
  It refuses --pair-stages / --pair-gain outside --schedule interleaved and --batch-split-impl native for DeepSeek-V3
  on TPU (`native_split_refusal`), and under --batch-split on Lineage's layers every flag they cannot run that the
  flags alone decide, at once (`lineage_problems`), before building anything."""
  a = split_defaults(argparse.Namespace(**vars(a)))  # a copy: the caller's flags stay as given
  cfg = model_config(a)
  if a.schedule != "interleaved" and (getattr(a, "pair_stages", None), getattr(a, "pair_gain", None)) != (None, None):
    raise ValueError(
        "--pair-stages / --pair-gain: the F&B pairs of --schedule interleaved's steady state, which no other "
        f"schedule runs (--schedule {a.schedule or DEFAULT_SCHEDULE})"
    )
  if (why := native_split_refusal(a, None if meshes is None else [m.devices.flat[0] for m in meshes])) is not None:
    raise ValueError(why)
  if uses_lineage(a) and (bad := lineage_problems(a, cfg)):
    raise ValueError(
        "--batch-split runs Lineage's layers (--batch-split-impl lineage), which cannot run this: "
        + "; ".join(bad)
        + (
            " (leave --batch-split-impl out: a Qwen3.5-family model's default, " "native, runs ds_ep.split_layers)"
            if ds_ep.is_qwen(cfg)
            else " (off TPU, --batch-split-impl native runs ds_ep.split_layers)"
        )
    )
  wi_tiling, wo_tiling = resolve_tilings(a, cfg)
  ep_coll = getattr(a, "ep_collectives", "auto").split(",")
  inplace_dw = resolve_inplace_dw(getattr(a, "inplace_dw", "on"), getattr(a, "tokamax_gmm", "mosaic"))
  scfg = ds_ep.StageConfig(
      model=cfg,
      fsdp=a.fsdp,
      ep=a.ep,
      wi_tiling=wi_tiling,
      wo_tiling=wo_tiling,
      strict_tiling=getattr(a, "strict_tiling", False),
      gmm_impl=getattr(a, "gmm_impl", "megablox"),
      tokamax_gmm=getattr(a, "tokamax_gmm", "mosaic"),
      inplace_dw=inplace_dw in ("on", "all"),
      inplace_dw_any_dtype=inplace_dw == "all",
      buffer_masks=getattr(a, "buffer_masks", "on") == "on",
      combine_impl=getattr(a, "combine_impl", "xla"),
      route_weights=getattr(a, "route_weights", "combine"),
      lean_experts=getattr(a, "lean_experts", "on") == "on",
      layer_tree=getattr(a, "layer_tree", "stacked"),
      rs_dtype=jnp.dtype(a.rs_dtype),
      acc_dtype=jnp.dtype(a.acc_dtype),
      grad_dtype=(None if getattr(a, "grad_dtype", "auto") == "auto" else jnp.dtype(a.grad_dtype)),
      fsdp_collectives=(
          a.fsdp_collectives
          if a.fsdp_collectives != "auto"
          else "ppermute"
          if (a.fsdp in (1, 2) and a.ep in (1, 2) and (a.fsdp, a.ep) != (1, 1))
          else "xla"
      ),
      remat_fused_last=a.remat_last,
      save_residuals=getattr(a, "save_residuals", "off") == "on",
      ep_collectives=ep_coll[0] if len(ep_coll) == 1 else "auto",
      grad_reduce=a.grad_reduce,
      random_routing=(a.routing == "random"),
      use_ragged_sort=(
          (a.ep > 1 and getattr(a, "gmm_impl", "megablox") != "tokamax")
          if a.ragged_sort == "auto"
          else a.ragged_sort == "on"
      ),
      iota_embed=(getattr(a, "embed", "iota") == "iota"),
      adam_math=getattr(a, "adam_math", "fp32"),
      grad_clip=getattr(a, "grad_clip", 1.0),
      ragged_buffer_factor=getattr(a, "ragged_buffer_factor", -1.0),
      cp=getattr(a, "cp", 1),
      shard_mode=getattr(a, "shard_mode", "auto"),
      rs_axes=getattr(a, "rs_axes", "joint"),
      gather_axes=getattr(a, "gather_axes", "joint"),
      batch_split=getattr(a, "batch_split", False),
      batch_split_impl=a.batch_split_impl,
      lineage_capacity_factor=getattr(a, "lineage_capacity_factor", ds_ep_lineage.CAPACITY_FACTOR),
      split_dw=getattr(a, "split_dw", "acc"),
      expert_weights=getattr(a, "expert_weights", "stacked"),
  )
  if scfg_overrides:
    scfg = dataclasses.replace(scfg, **scfg_overrides)
  caps = getattr(a, "stash_caps", "")
  placement = getattr(a, "placement", None) or "circular"
  return Pipeline(
      scfg,
      n_stages=a.stages,
      chunks_per_stage=a.vv,
      n_micro=a.mb,
      global_batch=a.gbs,
      seq=a.seq,
      schedule=a.schedule,
      layout=a.layout,
      overlap=not a.no_overlap,
      pf_split=a.pf_split,
      rs_split=a.rs_split,
      rs_self=getattr(a, "rs_self", "all"),
      rs_reserve={"auto": None, "on": True, "off": False}[getattr(a, "rs_reserve", "auto")],
      costs=load_costs(a.costs),
      placement=placement,
      stash=getattr(a, "stash", "none"),
      stash_budget_gib=getattr(a, "stash_budget_gib", 4.5),
      stash_caps=[int(x) for x in caps.split(",")] if caps else None,
      init_by_chunk=getattr(a, "init_by_chunk", False),
      offload_residuals=getattr(a, "offload_residuals", "off") == "on",
      norm_reduce=getattr(a, "norm_reduce", "hops"),
      interleave_group=getattr(a, "interleave_group", None) or None,
      direct_experts=getattr(a, "direct_experts", "off") == "on",
      zb_wbuf_cap=getattr(a, "zb_wbuf_cap", 1),
      zb_split_last=getattr(a, "zb_split_last", "off") == "on",
      proxy=proxy,
      pair_stages=parse_pair_stages(getattr(a, "pair_stages", None)),
      pair_gain=getattr(a, "pair_gain", None),
      stage_ep_collectives=ep_coll if len(ep_coll) > 1 else None,
      meshes=meshes,
      aot=aot,
      transport=getattr(a, "transport", "native"),
      layer_split=parse_layer_split(getattr(a, "layer_split", "")),
      fsdp_scope=getattr(a, "fsdp_scope", None),
      prefetch_pin=getattr(a, "prefetch_pin", "off"),
      stage_lineage_sparsecore=stage_lineage_sparsecore,
  )


def parse_layer_split(spec):
  """--layer-split '3,1' -> (3, 1); '' -> None (an even split)."""
  if not spec:
    return None
  try:
    return tuple(int(x) for x in spec.split(","))
  except ValueError:
    raise ValueError(f"--layer-split {spec!r}: want comma-separated layer counts, one a chunk (e.g. 3,1)") from None


def parse_pair_stages(spec):
  """`--pair-stages` -> Pipeline's pair_stages: 'all' (every stage), 'none'
  (no pairs) or a comma list of stages, e.g. '1' or '0,1'; None, the flag
  left out, is interleaved's default (none)."""
  if spec is None or spec == "all":
    return spec
  if spec in ("none", ""):
    return ()
  return tuple(int(x) for x in str(spec).split(","))


# The PRNG MaxText's synthetic batch is drawn with: SyntheticDataIterator's jax.random.key(0, impl="threefry2x32").
MAXTEXT_BATCH_PRNG = "threefry2x32"


def _git(path):
  """{'sha', 'branch', 'dirty'} of the git checkout holding `path`; {} if none."""

  def git(*args):
    return subprocess.run(
        ["git", "-C", path, *args], capture_output=True, text=True, timeout=10, check=True
    ).stdout.strip()

  try:
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }
  except (OSError, subprocess.SubprocessError):
    return {}


def _dist_version(*names):
  for n in names:
    try:
      return importlib.metadata.version(n)
    except importlib.metadata.PackageNotFoundError:
      pass
  return None


def _import_root():
  """The directory the package imports from: ds_ep's directory, one level up for each component of its package's
  dotted name -- a MaxText checkout's src/ (git finds the checkout from it).  By ds_ep's name, not this module's,
  which is __main__ in a `python3 -m` run."""
  root = os.path.dirname(os.path.abspath(ds_ep.__file__))
  for _ in ds_ep.__name__.split(".")[:-1]:
    root = os.path.dirname(root)
  return root


def provenance(scfg: ds_ep.StageConfig | None = None, tiles: str | None = None):
  """What produced a run's numbers (best effort): the maxtext package the
  kernels come from -- located with find_spec, not imported, so this is the
  checkout PYTHONPATH / the editable install actually resolve -- and its git
  state; 'mpmd', the directory this package imports from (`_import_root`) and
  its git state, the revision of the pipeline's own code; the jax / jaxlib /
  libtpu / tokamax versions, LIBTPU_INIT_ARGS, the devices, and the resolved
  grouped-matmul tilings; under batch split on Lineage's layers the Lineage
  package's path and git state too (ds_ep_lineage.provenance, which imports
  it: None, with the error, where it does not import) and its capacity factor.
  Under Pathways the mode too (`pathways.describe`: the proxy client, the
  server's build and the image, MPMD_PATHWAYS_IMAGE): the libtpu version and
  LIBTPU_INIT_ARGS are then the client's, which compiles nothing -- the server
  image's compiler and the servers' flags did."""
  try:
    spec = importlib.util.find_spec("maxtext")
  except (ImportError, ValueError):
    spec = None
  mt = None
  if spec is not None:
    path = os.path.dirname(spec.origin) if spec.origin else next(iter(spec.submodule_search_locations or ()), None)
    mt = {"path": path, **(_git(path) if path else {})}
  here = _import_root()
  devs = jax.devices()
  out = {
      "maxtext": mt,
      "mpmd": {"path": here, **_git(here)},
      "versions": {
          "jax": jax.__version__,
          "jaxlib": _dist_version("jaxlib"),
          "libtpu": _dist_version("libtpu", "libtpu-nightly"),
          "tokamax": _dist_version("tokamax"),
      },
      "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS"),
      "devices": f"{len(devs)} x {devs[0].device_kind}" if devs else None,
  }
  if devs and pathways.is_proxy_device(devs[0]):
    out["pathways"] = pathways.describe(devs[0].client)
  if devs and devs[0].platform == "gpu":
    out["versions"].update(
        {
            k: _dist_version(*d)
            for k, d in (
                ("cuda_plugin", ("jax-cuda13-plugin", "jax-cuda12-plugin")),
                ("cudnn", ("nvidia-cudnn-cu13", "nvidia-cudnn-cu12")),
                ("nccl", ("nvidia-nccl-cu13", "nvidia-nccl-cu12")),
                ("cublas", ("nvidia-cublas", "nvidia-cublas-cu12")),
                ("triton", ("triton",)),
                ("jax_triton", ("jax-triton",)),
            )
        }
    )
    out["XLA_FLAGS"] = os.environ.get("XLA_FLAGS")
  if scfg is not None:
    out.update(gmm_impl=scfg.gmm_impl, tiles=tiles, wi_tiling=scfg.wi_tiling, wo_tiling=scfg.wo_tiling)
    if scfg.batch_split:
      out["batch_split_impl"] = scfg.batch_split_impl
    if scfg.lineage_split:
      out.update(ds_ep_lineage.provenance(), lineage_capacity_factor=scfg.lineage_capacity_factor)
  return out


def provenance_line(pv):
  """`provenance` as one line."""

  def repo(r):
    if not r:
      return "not found"
    sha = (r.get("sha") or "?")[:12]
    return f"{r['path']} @ {sha} ({r.get('branch', '?')}{', dirty' if r.get('dirty') else ''})"

  v = pv["versions"]
  line = (
      f"provenance: maxtext {repo(pv['maxtext'])} | mpmd {repo(pv['mpmd'])} | "
      + " ".join(f"{k} {v[k]}" for k in v)
      + f" | {pv['devices']}"
  )
  if "lineage" in pv:  # batch split on Lineage's layers: no grouped-matmul tiling of ds_ep's runs
    line += (
        f" | batch split on Lineage {repo(pv['lineage'])}"
        + (f" ({pv['error']})" if pv.get("error") else "")
        + f", capacity factor {pv['lineage_capacity_factor']:g}"
    )
  elif "gmm_impl" in pv:
    line += (
        f" | gmm {pv['gmm_impl']} tiles {pv['tiles']} wi={','.join(map(str, pv['wi_tiling']))} "
        f"wo={','.join(map(str, pv['wo_tiling']))}"
        + (f" | batch split {pv['batch_split_impl']}" if "batch_split_impl" in pv else "")
    )
  if "XLA_FLAGS" in pv:
    return line + f" | XLA_FLAGS={pv['XLA_FLAGS']!r}"
  if "pathways" in pv:  # the client's libtpu and flags compile nothing here: the server image's compiler does
    pw = pv["pathways"]
    return (
        line
        + f" | Pathways: {pw['backend']}, server {pw['server']!r}, image {pw['image'] or 'unknown (MPMD_PATHWAYS_IMAGE)'}"
        + f" | client LIBTPU_INIT_ARGS={pv['LIBTPU_INIT_ARGS']!r} (planning only)"
    )
  return line + f" | LIBTPU_INIT_ARGS={pv['LIBTPU_INIT_ARGS']!r}"


def maxtext_training_flops(cfg: DeepSeekConfig, tokens: int, seq: int):
  """`(learnable-weight FLOPs, causal-attention FLOPs)` of one training step
  over `tokens` tokens of `seq`-long sequences, by MaxText's own accounting.

  Ported from maxtext/utils/maxtext_utils.py (calculate_tflops_training_per_device,
  DeepSeek branch: calculate_mla_tflops_per_device +
  calculate_routed_and_shared_ffn_tflops_per_device, dense layers included), so
  a TFLOP/s printed here and MaxText's `TFLOP/s/device` log line divide the
  SAME numerator.  fwd + 2 x bwd (the `* 3`); remat recompute is not counted
  (by either side), and neither is the embedding lookup (a gather) -- only the
  unembedding.  A Qwen3.5-family config: `qwen_training_flops`.
  """
  if ds_ep.is_qwen(cfg):
    return qwen_training_flops(cfg, tokens, seq)
  b, D, H, V = tokens, cfg.emb_dim, cfg.num_query_heads, cfg.vocab_size
  qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
  ffn = lambda f: 2 * b * f * D * len(cfg.mlp_activations) + 2 * b * f * D
  moe = (
      2 * b * D * cfg.num_experts
      + ffn(cfg.shared_expert_mlp_dim) * cfg.shared_experts
      + ffn(cfg.moe_mlp_dim) * cfg.num_experts_per_tok
  )
  L, n_dense, mtp = cfg.num_decoder_layers, cfg.first_num_dense_layers, cfg.mtp_num_layers
  if cfg.q_lora_rank == 0:
    q = 2 * b * D * H * qk
  else:
    q = 2 * b * (D * cfg.q_lora_rank + cfg.q_lora_rank * H * qk)
  kv = (
      2
      * b
      * (D * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) + cfg.kv_lora_rank * H * (cfg.qk_nope_head_dim + cfg.v_head_dim))
  )
  attn = 2 * b * seq * H * (qk + cfg.v_head_dim) / 2
  proj = 2 * b * D * H * cfg.v_head_dim
  emb = 2 * b * D * V
  # an MTP layer as MaxText counts it: one more MoE layer's FFN, attention projections and attention, not its
  # [2 x emb -> emb] projection or its pass through the shared head (maxtext_utils.calculate_tflops_training_per_device)
  weights = (
      ffn(cfg.mlp_dim) * n_dense + moe * (L - n_dense) + (q + kv + proj) * L + emb + (moe + q + kv + proj) * mtp
  ) * 3
  return float(weights), float(attn * (L + mtp) * 3)


def qwen_training_flops(cfg: Qwen35Config, tokens: int, seq: int):
  """`maxtext_training_flops` of a Qwen3.5-family model: MaxText's own
  accounting for decoder_block qwen3_5 (maxtext/utils/maxtext_utils.py,
  calculate_tflops_training_per_device's QWEN3_5 branch with
  calculate_routed_and_shared_ffn_tflops_per_device and
  calculate_gated_delta_net_flops_per_device).  Per layer the MoE (router,
  shared expert, top-k routed experts); per full-attention layer q / k / v
  (not the query's gate half) and the output projection, and causal attention;
  per Gated DeltaNet layer its projections and conv, and 8 Hv Dk Dv a token
  for the delta rule (counted with attention); the unembedding.  x 3 for fwd
  + bwd; remat recompute not counted."""
  b, D, V = tokens, cfg.emb_dim, cfg.vocab_size
  H, kv, hd = cfg.num_query_heads, cfg.num_kv_heads, cfg.head_dim
  ffn = lambda f: 2 * b * f * D * len(cfg.mlp_activations) + 2 * b * f * D
  moe = (
      2 * b * D * cfg.num_experts
      + ffn(cfg.shared_expert_mlp_dim) * cfg.shared_experts
      + ffn(cfg.moe_mlp_dim) * cfg.num_experts_per_tok
  )
  qkv = 2 * b * D * (H + 2 * kv) * hd
  attn = 4 * b * seq * H * hd / 2  # causal
  proj = 2 * b * D * H * hd
  emb = 2 * b * D * V
  K, Vd, hv = cfg.gdn_key_dim, cfg.gdn_value_dim, cfg.gdn_num_value_heads
  gdn_w = 2 * b * D * (2 * K + 2 * Vd + 2 * hv) + 2 * b * D * Vd + 2 * b * cfg.gdn_conv_kernel_dim * (2 * K + Vd)
  gdn_core = b * hv * cfg.gdn_key_head_dim * cfg.gdn_value_head_dim * 8
  L = cfg.num_decoder_layers
  n_full = L // cfg.inhomogeneous_layer_cycle_interval
  n_lin = L - n_full
  weights = (moe * L + emb + (qkv + proj) * n_full + gdn_w * n_lin) * 3
  return float(weights), float((attn * n_full + gdn_core * n_lin) * 3)


def maxtext_tflops_per_device(p: Pipeline):
  """Per-device training TFLOP per step, by MaxText's own accounting
  (`maxtext_training_flops`: dense layers and MTP as MaxText counts them)."""
  w, a = maxtext_training_flops(p.scfg.model, p.gbs * p.seq, p.seq)
  return (w + a) / (p.S * p.scfg.fsdp * p.scfg.ep) / 1e12


def peak_hbm():
  """{device id: peak HBM in use so far, GiB} -- every (local) device, not
  just the worst: the stages hold different chunks (and stash), so they differ.
  A device whose backend keeps no statistics it can report is left out: the
  IFRT proxy's (Pathways) raise ("MemoryStats is only supported for
  addressable PjRt devices"), so there it is {}."""
  out = {}
  for d in jax.local_devices():
    stats, why = pathways.memory_stats(d)
    if why is None:
      out[d.id] = round(stats.get("peak_bytes_in_use", 0) / 2**30, 2)
  return out


def peak_hbm_line(per):
  """main's line for `peak_hbm`'s answer `per` (after "peak HBM"): the GiB per device, or why there are none."""
  if per:
    return f"(GiB per device) {per}"
  return "not reported: the devices keep no memory statistics this client can read (through the IFRT proxy, Pathways)"


class _Parser(ModelParser):
  """make_parser's parser: `--model` picks the shape flags' defaults (ModelParser), and what it parses comes back
  with the defaults --batch-split decides filled in (`split_defaults`: the impl by the model, then --routing and
  --shard-mode by the impl), so every parse of these flags -- main's, ds_ep_aot's, a test's -- names what runs."""

  def parse_known_args(self, args=None, namespace=None):
    a, rest = super().parse_known_args(args, namespace)
    return split_defaults(a), rest


def make_parser():
  ap = _Parser()
  ap.add_argument(
      "--model",
      default="deepseek3",
      choices=list(MODELS),
      help="the layer family and its dims: deepseek3 (DeepSeek-V3 MoE layers, the shape flags as given) "
      "or a Qwen3.5-family model by its MaxText name (qwen3.8-2.4t-a95b: 92 layers, 3 Gated "
      "DeltaNet then 1 gated attention, every layer MoE), whose dims become the shape flags' "
      "defaults (QWEN_MODELS); any shape flag given still wins",
  )
  ap.add_argument(
      "--platform",
      default="auto",
      choices=list(PLATFORMS),
      help="the platform whose defaults the kernel and collective flags take (auto: JAX_PLATFORMS, else "
      "libtpu -> tpu, a jax CUDA plugin -> gpu).  On gpu "
      + ", ".join(f'--{k.replace("_", "-")} {v}' for k, v in GPU_DEFAULTS.items())
      + "; tpu and cpu keep the TPU defaults",
  )
  ap.add_argument("--stages", type=int, default=2)
  ap.add_argument("--vv", type=int, default=2, help="chunks per stage")
  ap.add_argument("--mb", type=int, default=8)
  ap.add_argument("--gbs", type=int, default=32)
  ap.add_argument("--seq", type=int, default=4096)
  ap.add_argument("--fsdp", type=int, default=2)
  ap.add_argument("--ep", type=int, default=2)
  ap.add_argument("--layers", type=int, default=4)
  ap.add_argument("--emb", type=int, default=7168)
  ap.add_argument("--heads", type=int, default=128)
  ap.add_argument("--q-lora", type=int, default=1536)
  ap.add_argument("--kv-lora", type=int, default=512)
  ap.add_argument(
      "--qk-nope-head-dim",
      type=int,
      default=128,
      help="MLA's query / key head dims without and with RoPE and its value head dim (DeepSeek-V3's "
      "128 / 64 / 128).  On sm80 cuDNN's flash attention takes equal query / key and value head "
      "dims of at most 128 (e.g. 64 + 64 / 128); sm90 and later take DeepSeek-V3's 192 / 128.  "
      "--qk-rope-head-dim 0 is MLA without RoPE; a value head dim below 1, or query / key head dims "
      "negative or summing to 0, are refused",
  )
  ap.add_argument("--qk-rope-head-dim", type=int, default=64)
  ap.add_argument("--v-head-dim", type=int, default=128)
  ap.add_argument("--moe-mlp", type=int, default=2048)
  ap.add_argument(
      "--dense-layers",
      type=int,
      default=0,
      help="DeepSeek-V3's first_num_dense_layers: dense decoder layers (MLP width --mlp) before the MoE "
      "layers, all in chunk 0 (or spanning the first chunks under a --layer-split that counts every "
      "layer); --layers counts them (real DeepSeek-V3: --layers 61 --dense-layers 3 --mlp 18432)",
  )
  ap.add_argument(
      "--mlp",
      type=int,
      default=None,
      help="the dense layers' MLP width (MaxText base_mlp_dim; DeepSeek-V3 18432; default --moe-mlp)",
  )
  ap.add_argument("--experts", type=int, default=128)
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--vocab", type=int, default=32768)
  # DeepSeek-V3 as MaxText's deepseek3-671b with the MLPerf recipe's routing, aux loss and MTP: --layers 61
  # --dense-layers 3 --mlp 18432 --routed-bias on --bias-update-rate 0.001 --routing-groups 8 --topk-groups 4
  # --float32-gate-logits on --lb-loss-weight 0.01 --mtp 1 --routing topk
  ap.add_argument(
      "--routed-bias",
      default="off",
      choices=["off", "on"],
      help="MaxText routed_bias: a bias on every router's scores picks the experts (not their weights)",
  )
  ap.add_argument(
      "--bias-update-rate",
      type=float,
      default=0.0,
      help="MaxText routed_bias_update_rate: after each step every routed bias moves by rate x sign(mean "
      "load - load), the loads counted over the step's tokens (DeepSeek-V3: 0.001)",
  )
  ap.add_argument(
      "--routing-groups",
      type=int,
      default=-1,
      help="MaxText n_routing_groups: top-k picks among each token's --topk-groups best groups of "
      "experts (a group scored by its two best scores; DeepSeek-V3: 8)",
  )
  ap.add_argument("--topk-groups", type=int, default=-1, help="MaxText topk_routing_group (DeepSeek-V3: 4)")
  ap.add_argument(
      "--float32-gate-logits",
      default="off",
      choices=["off", "on"],
      help="MaxText float32_gate_logits: the routers in fp32, from an fp32 compute copy of their kernels "
      "(GateLogit casts the stored kernel), their routed biases fp32 in storage too.  Under "
      "--batch-split on Lineage's layers only the expert selection runs in fp32 (Lineage's "
      "router_dtype): the kernel copy is rounded to bf16 first, and its gradient comes back in bf16 "
      "(bf16 operands, fp32 accumulation; warned of)",
  )
  ap.add_argument(
      "--lb-loss-weight",
      type=float,
      default=0.0,
      help="MaxText load_balance_loss_weight with moe_use_megatron_seq_aux_loss: every MoE layer's "
      "sequence-wise aux loss (Megatron-LM seq_aux_loss), summed into the objective (DeepSeek-V3 "
      "recipe: 0.01)",
  )
  ap.add_argument(
      "--mtp",
      type=int,
      default=0,
      choices=[0, 1],
      help="MaxText mtp_num_layers: a multi-token-prediction MoE layer after the head, on the last chunk; "
      "its embedding is the input's (mtp_reuse_input_embedding)",
  )
  ap.add_argument("--mtp-loss-scale", type=float, default=0.1, help="MaxText mtp_loss_scaling_factor")
  ap.add_argument("--kv-heads", type=int, default=None, help="Qwen3.5 family: gated attention's kv heads (GQA)")
  ap.add_argument("--head-dim", type=int, default=None, help="Qwen3.5 family: gated attention's head dim")
  ap.add_argument("--gdn-key-heads", type=int, default=None, help="Qwen3.5 family: Gated DeltaNet key heads")
  ap.add_argument("--gdn-value-heads", type=int, default=None, help="Qwen3.5 family: Gated DeltaNet value heads")
  ap.add_argument("--gdn-key-head-dim", type=int, default=None, help="Qwen3.5 family: Gated DeltaNet key head dim")
  ap.add_argument("--gdn-value-head-dim", type=int, default=None, help="Qwen3.5 family: Gated DeltaNet value head dim")
  ap.add_argument(
      "--qwen-rope",
      default=None,
      choices=["partial", "mrope"],
      help="Qwen3.5 family: gated attention's rotary embedding, MaxText's use_mrope -- partial, "
      "PartialRotaryEmbedding (bf16 sin/cos and rotation: qwen3.8-2.4t-a95b's yml, which leaves "
      "use_mrope false), or mrope, the text mRoPE (fp32 sin/cos and rotation: the qwen3.5 ymls); "
      "default the --model's (maxtext.experimental.mpmd.jax.qwen35.attention)",
  )
  ap.add_argument(
      "--attn",
      default="tokamax_splash",
      help="the attention kernel (DeepSeekConfig.attention_kernel; a Qwen3.5-family model's gated "
      "attention's too): tokamax_splash (tokamax's splash, the benchmark's), splash (MaxText's), "
      "dot_product or reference (the port's CPU kernel).  Not under --batch-split on Lineage's "
      "layers, which run Lineage's own splash (base-2 softmax)",
  )
  ap.add_argument(
      "--sa-block",
      type=int,
      default=2048,
      help="tokamax_splash's and splash's forward q and kv blocks (MaxText sa_block_q, sa_block_kv); "
      "under --batch-split on Lineage's layers Lineage's splash's too, whatever --attn says (it "
      "clamps each of the six blocks to --seq)",
  )
  ap.add_argument(
      "--sa-block-kv-compute",
      type=int,
      default=512,
      help="tokamax_splash's and splash's forward kv compute block (MaxText sa_block_kv_compute; 512 "
      "tuned for tokamax_splash); under --batch-split on Lineage's layers Lineage's splash's too, "
      "which keeps its SV diagonal skip only where this equals --sa-block (both clamped to --seq): "
      "the default 512 below --sa-block 2048 turns it off (Lineage's own blocks: 2048, all six)",
  )
  ap.add_argument(
      "--sa-block-dkv",
      type=int,
      default=None,
      help="tokamax_splash, and Lineage's splash under --batch-split on its layers: the fused backward's "
      "q and kv blocks (MaxText sa_block_q_dkv, sa_block_kv_dkv); default --sa-block",
  )
  ap.add_argument(
      "--sa-block-kv-dkv-compute",
      type=int,
      default=None,
      help="tokamax_splash, and Lineage's splash under --batch-split on its layers (whose QK diagonal "
      "skip needs it equal to --sa-block-dkv): the fused backward's kv compute block (MaxText "
      "sa_block_kv_dkv_compute); default --sa-block-dkv",
  )
  ap.add_argument(
      "--sa-base2-exp",
      default=None,
      choices=["on", "off"],
      help="tokamax_splash: its base-2 softmax (MaxText sa_use_base2_exp, on in MaxText's base.yml), on "
      "by default as tokamax always ran: MaxText's numerics -- the kernels scale the bf16 query by "
      "log2(e) rounded to bf16 (1.4453125), a 1.0018x sharper softmax with ~0.3%% larger attention "
      "gradients.  off: the exact softmax, as --attn splash computes (for parity against MaxText "
      "with sa_use_base2_exp=false), ~1.2%% slower in the kernel (fwd + bwd at 4096 x 128 heads on "
      "TPU7x, 11.49 vs 11.34 ms).  on with another --attn is refused.  Not under --batch-split on "
      "Lineage's layers, whose splash is base-2 always (off is ignored there, with a warning)",
  )
  ap.add_argument("--cost-estimate", type=int, default=5_000_000_000_000)
  ap.add_argument(
      "--tiles",
      default="clamped",
      choices=sorted(ds_ep.TILES),
      help="megablox tilings (ds_ep.TILES): clamped (every tk <= the k it tiles), "
      "v6e (swept on TPU v6e at emb 7168, moe_mlp 2048).  Not under --batch-split on Lineage's "
      "layers (--batch-split-impl lineage), which run no megablox kernel",
  )
  ap.add_argument(
      "--wi-tiling",
      type=int,
      nargs=9,
      default=None,
      metavar="T",
      help="wi_0 / wi_1 megablox tiling (fwd tm tk tn, dlhs tm tk tn, drhs tm tk tn); "
      "overrides --tiles (not under --batch-split on Lineage's layers)",
  )
  ap.add_argument(
      "--wo-tiling",
      type=int,
      nargs=9,
      default=None,
      metavar="T",
      help="wo megablox tiling, as --wi-tiling; overrides --tiles (not under --batch-split on Lineage's " "layers)",
  )
  ap.add_argument(
      "--strict-tiling", action="store_true", help="refuse (instead of warning about) a tile larger than the k it tiles"
  )
  ap.add_argument(
      "--gmm-impl",
      default="megablox",
      choices=list(ds_ep.GMM_IMPLS),
      help="routed-expert grouped matmuls: megablox (MaxText's kernels, the tilings "
      "above), ragged_dot (pure JAX: XLA's ragged dot) or tokamax (GPU: "
      "tokamax's / gpu_kernels' grouped matmuls on a compact local buffer, --tokamax-gmm; the GPU "
      "default).  Not under --batch-split on Lineage's layers, whose experts run Lineage's own gmmv2 "
      "kernels",
  )
  ap.add_argument(
      "--tokamax-gmm",
      default="mosaic",
      choices=list(ds_ep.TOKAMAX_GMM),
      help="--gmm-impl tokamax: the backend of the forward and dlhs grouped matmuls (mosaic: Mosaic GPU, "
      "on sm100 gpu_kernels.grouped_matmul -- the weight read as stored, tuned tiles -- else "
      "tokamax's, with dW on Triton but an in-place one (--inplace-dw, or mb0's under --split-dw acc) "
      "on gpu_kernels' tcgen05 kernel on sm100, where --inplace-dw auto puts an fp32 accumulator's "
      "in place too; tokamax_mosaic: tokamax's Mosaic kernels everywhere, dW on Triton; triton: all "
      "three on Triton; xla: lax throughout, for CPU)",
  )
  ap.add_argument(
      "--gdn-conv",
      default="conv",
      choices=list(GDN_CONV_IMPLS),
      help="Qwen: the Gated DeltaNet's causal depthwise conv as lax.conv_general_dilated (conv: MaxText's op, "
      "bitwise) or as its 4 shifted multiply-adds in fp32 (shift: the GPU default -- XLA:GPU fuses "
      "them, where cuDNN's grouped conv runs ~10x slower than its bytes); DeepSeek ignores it",
  )
  ap.add_argument(
      "--gdn-scan",
      default="scan",
      choices=list(GDN_SCAN_IMPLS),
      help="Qwen: the Gated DeltaNet's chunked delta rule with its inter-chunk recurrence as a lax.scan "
      "(scan: MaxText's) or on Triton kernels (triton: the GPU default -- at the model's chunks of "
      "64 the whole rule, gdn_triton.FUSED: the WY factors, the triangular solve, the intra-chunk "
      "output and their backward as per-chunk kernels around the recurrence's, a GB200 layer's "
      "forward + backward 27.4 -> 10.7 ms; at a chunk FUSED does not take, the recurrence alone, "
      "the chunk outputs batched in XLA; runs per device, as the stage programs do); DeepSeek "
      "ignores it",
  )
  ap.add_argument(
      "--layer-tree",
      default="stacked",
      choices=list(ds_ep.LAYER_TREES),
      help="a DeepSeek chunk's layer weights, MoE and dense: stacked ([n, ...] leaves; TPU's, megablox "
      "reads a layer's experts in place) or per_layer (one subtree a layer, as a Qwen chunk's: no "
      "gradient stacking in every backward, in-place dW on every chunk; the GPU default)",
  )
  ap.add_argument(
      "--inplace-dw",
      default="auto",
      choices=["auto", "on", "off", "all"],
      help="--gmm-impl tokamax: add the routed experts' weight gradients into the step accumulator inside "
      "the dW kernel (on; where the accumulator dtype is the compute dtype), also into an accumulator "
      "of its own dtype (an fp32 one) on every chunk but the last (all: on sm_100 under --tokamax-gmm "
      "mosaic the tcgen05 kernel, elsewhere gpu_kernels.ragged_dw's Triton one (slower than "
      "tokamax's dW plus an add on an A100), and under --tokamax-gmm xla lax's form), or write each "
      "to a buffer and add it after (off).  auto: all where the tcgen05 kernel takes an fp32 "
      "accumulator (sm_100 under --tokamax-gmm mosaic), else on",
  )
  ap.add_argument(
      "--buffer-masks",
      default="on",
      choices=["on", "off"],
      help="--gmm-impl tokamax: zero the compact expert buffer's rows past its groups after the dispatch, "
      "every grouped matmul and on each gradient (on), or skip those buffer-wide passes (off): every "
      "reader of the rows is bounded by the group sizes or gathers valid rows only "
      "(StageConfig.buffer_masks)",
  )
  ap.add_argument(
      "--combine-impl",
      default="xla",
      choices=["xla", "triton"],
      help="gmm_impl 'tokamax': the per-token sums over a token's routed choices (the combine, the "
      "dispatch's gradient): 'xla' walks all T x K choices, 'triton' (GPU; the GPU default) loads only "
      "the rows this shard has (StageConfig.combine_impl; 64 B200, DeepSeek-V3 dims: 15.33 -> "
      "14.86 s a step)",
  )
  ap.add_argument(
      "--route-weights",
      default="combine",
      choices=list(ds_ep.ROUTE_WEIGHTS),
      help="where the routing weights multiply the routed experts' output: in the combine's weighted sum "
      "(combine: MaxText's), or on the rows of the gated activation before the wo product, the "
      "combine then an unweighted sum (experts: Megatron-LM's; the GPU default) -- whose backward "
      "needs no expert output row, so --save-residuals keeps none (896 MiB a DeepSeek-V3 layer at one "
      "sequence a GPU, EP 8, --ragged-buffer-factor 2).  Rounds the weighted activation to the "
      "compute dtype before wo.  experts needs --gmm-impl tokamax or --ragged-sort off "
      "(StageConfig.route_weights)",
  )
  ap.add_argument(
      "--lean-experts",
      default="on",
      choices=["on", "off"],
      help="with --save-residuals on: the routed experts' grouped matmuls keep what their lhs is recomputed "
      "from (on: the gathered tokens and the gated product's inputs, the fewest bytes) or the lhs "
      "itself (off: the buffer rows and the product; no row gather, and one evaluation of the gated "
      "product fewer, in the backward, for ~0.7 GiB more a DeepSeek-V3 layer-microbatch at one "
      "sequence a GPU, EP 8, --ragged-buffer-factor 2)",
  )
  ap.add_argument("--rs-dtype", default="bfloat16", type=float_dtype_arg)
  ap.add_argument(
      "--grad-dtype",
      default="auto",
      choices=["auto", "float32", "bfloat16"],
      help="the dtype the reduced gradient enters AdamW (and the clip) in, cast from --rs-dtype: "
      "MaxText grad_dtype, which its trainer applies after the reduction (auto: under --adam-math "
      "optax each weight's dtype, as MaxText's gradients take their parameter's; else --rs-dtype, "
      "no cast).  --rs-dtype bfloat16 --grad-dtype float32 --adam-math optax is MaxText's "
      "weight_dtype=float32 grad_dtype=float32, which reduce-scatters in bf16 and casts after (but "
      "for an fp32 router's kernel, which MaxText reduces in fp32).  --adam-math fp32 computes in "
      "fp32 whatever this is.  MaxText casts fp32 gradients only: with bf16 weights keep auto",
  )
  ap.add_argument("--acc-dtype", default="float32", type=float_dtype_arg)
  ap.add_argument(
      "--weight-dtype",
      default="bfloat16",
      type=float_dtype_arg,
      help="master weights and AdamW nu (MaxText weight_dtype); mu is always bf16",
  )
  ap.add_argument(
      "--dtype",
      default="bfloat16",
      choices=["bfloat16", "float32"],
      help="activations and compute copies (MaxText dtype): float32 for parity checks with the "
      "pure-JAX kernels (--gmm-impl ragged_dot --ragged-sort off, --attn dot_product)",
  )
  ap.add_argument(
      "--remat-last", action="store_true", help="also recompute the fused last chunk (MaxText-identical FLOPs)"
  )
  ap.add_argument(
      "--save-residuals",
      default="off",
      choices=["on", "off"],
      help="every chunk's forward keeps its backward's residuals (its layers under jax.vjp; Megatron-LM's "
      "default) instead of recomputing the chunk's forward inside its backward: no forward computed "
      "twice, for the residuals' memory until the backward (StageConfig.save_residuals)",
  )
  ap.add_argument(
      "--direct-experts",
      default="off",
      choices=["on", "off"],
      help="at --fsdp 1 the stages read their routed experts in place (on; the GPU default): stored over "
      "'expert' alone, a compute copy would be their storage's bytes again (12 GiB a Qwen3.8 "
      "layer at EP 4), so no gather makes one; off gathers that copy as at --fsdp > 1 (TPU's plans)",
  )
  ap.add_argument(
      "--interleave-group",
      type=int,
      default=0,
      help="--schedule interleaved: microbatches a chunk runs before the next chunk (Megatron-LM's "
      "microbatch_group_size_per_vp_stage; 0: --stages).  Above --stages each chunk switch has "
      "slack against the boundary copies' latency, for (V-1) x the excess more warmup forwards "
      "held, and about the excess more boundary inputs waiting at the first and last stage (which "
      "--offload-residuals does not move); --mb must be a multiple of it",
  )
  ap.add_argument(
      "--schedule",
      default=None,
      choices=list(SCHEDULES),
      help="microbatch order (see Pipeline); default breadth (DEFAULT_SCHEDULE), on either "
      "placement.  timed is the event-driven search over "
      "per-program costs (--costs); zbh1 / zbh2 the zero-bubble schedules at --vv 1, each middle chunk's backward "
      "split into its input and weight gradients (I and W, the W's run later; the fused last chunk's too with "
      "--zb-split-last on), priced by --costs too: ZB-H1 holds 1F1B's residual sets and ZB-H2 about twice them, each "
      "plus up to --zb-wbuf-cap pending W buffers a stage (a W buffer: about 3.6 GiB a device per DeepSeek-V3 MoE "
      "layer at 4096 tokens a device); stash the HBM-budgeted "
      "stash scheduler (--stash, "
      "--stash-budget-gib); interleaved Megatron-LM's "
      "interleaved 1F1B (circular placement, --vv >= 2, --mb a multiple of --stages; F&B pairs "
      "in its steady state on --pair-stages); breadth "
      "MaxText's circular-pipeline order: each stage runs a chunk's forward for every "
      "microbatch before its next chunk, the backwards the same in reverse (what search picks at "
      "--stages 4, --vv 2 and 4, --mb 8; --fsdp-scope repeat needs an order like it, each chunk "
      "visited at most twice a step).  gpipe at --vv > 1 is depth-first, not interleaved",
  )
  ap.add_argument(
      "--zb-wbuf-cap",
      type=int,
      default=1,
      help="--schedule zbh1 / zbh2: the W buffers a stage may hold pending (0: no cap), on top of the residual sets "
      "(1F1B's under zbh1).  Each is one microbatch's split chunk's crossing set and the arguments its W reads -- "
      "the summary's 'split backward' line gives their GiB, about 3.6 GiB a device a DeepSeek-V3 MoE layer at 4096 "
      "tokens a device -- so 1 by default, which keeps most of the schedules' gain in simulation",
  )
  ap.add_argument(
      "--zb-split-last",
      default="off",
      choices=["off", "on"],
      help="--schedule zbh1 / zbh2: split the fused last chunk too -- its F runs the forward, the head, the loss and "
      "the input gradient (AI), a W of its own each microbatch its layers' weight gradients, later -- so the last "
      "stage holds W buffers too, under --zb-wbuf-cap (7.2 GiB a device for two DeepSeek-V3 MoE layers at 4096 "
      "tokens a device).  Off by default; turn it on at --stages 2, where no middle chunk splits and it is the only "
      "split (without it zbh1 is 1F1B's order).  Measured (README, 'Zero-bubble schedules'): faster in every pair "
      "on one TPU7x-8, slower under zbh2 at four stages on 2 x TPU7x-4x4x4",
  )
  ap.add_argument(
      "--placement",
      default=None,
      choices=["circular", "v"],
      help="chunk -> stage map: circular (chunk c on c %% S, MaxText's; the default) or v (--vv 2: "
      "stage s holds chunks s and 2S-1-s)",
  )
  ap.add_argument(
      "--pair-stages",
      default=None,
      help="--schedule interleaved: the stages that run its steady state's F(k+W) and B(k) as one "
      "program, one microbatch's EP collectives beside another's compute (Megatron-LM's combined "
      "1F1B): all, none, or a list such as 1 or 0,1 (default none, Megatron's order as it is)",
  )
  ap.add_argument(
      "--pair-gain",
      type=float,
      default=None,
      help="--schedule interleaved: ms an F&B pair program saves over its two halves back to back, for "
      "the simulation (default: the cost table's pair_gain, else 0)",
  )
  ap.add_argument(
      "--stash",
      default="none",
      choices=[k for k in ds_ep.STASH_POLICIES if k != "all"],
      help="--schedule stash / interleaved: the policy of the "
      "stash-mode (chunk, microbatch)s (ds_ep.STASH_POLICIES; none under --batch-split)",
  )
  ap.add_argument(
      "--stash-budget-gib",
      type=float,
      default=4.5,
      help="--stash: live stash sets allowed per stage, GiB per device, on top of "
      "a remat step (a placeholder until a --mem-report sweep on the TPU sets it); "
      "the stash backwards' extra temporaries come on top (the summary prints both)",
  )
  ap.add_argument(
      "--stash-caps",
      default="",
      help="--stash: live stash sets per chunk, e.g. 1,3,2 (default: searched under "
      "--schedule stash, none on a fixed order). A chunk left off the list never "
      "stashes under --schedule stash and is uncapped on a fixed order (interleaved)",
  )
  ap.add_argument(
      "--norm-reduce",
      default="hops",
      choices=["hops", "allreduce"],
      help="how stages share their gradient sums of squares for the global-norm clip under --distributed: "
      "hops (a scalar copy between every two stages, S(S-1) a step) or allreduce (one sum over every "
      "device of the job; the GPU default; in one process the hops run; refused under Pathways)",
  )
  ap.add_argument(
      "--offload-residuals",
      default="off",
      choices=["off", "on"],
      help="keep each microbatch's forward residuals in pinned host memory until its backward (on: the "
      "HBM a step holds stops growing with the microbatches in flight; a GB200's Grace host sits "
      "behind NVLink-C2C)",
  )
  ap.add_argument(
      "--init-by-chunk",
      action="store_true",
      help="seed each chunk from its index alone: the same weights under any placement",
  )
  ap.add_argument(
      "--dry-run",
      action="store_true",
      help="build the pipeline and print the schedule, the plan and the peak stash; " "compile and run nothing",
  )
  ap.add_argument(
      "--layout",
      default="fsdp_on_chip",
      choices=list(LAYOUTS),
      help="fsdp_on_chip | ep_on_chip (MaxText's placement) | snake (PP=4/8) | physical "
      "(MaxText's mesh over a whole slice: stage s = row s) | physical_origin_last "
      "(physical, rotated so the strip holding device 0 -- the one libtpu keeps "
      "synchronous -- runs the last stage) | slices (k stages per TPU slice: --stages = k x "
      "the number of slices, fsdp x ep = a k-th of the devices of one; each stage's mesh topology-aware "
      "over its own slice, or at k > 1 over a strip of its whole chips along the last chip axis k divides (both "
      "TensorCores of a chip in one stage, as Pathways requires); its collectives on SparseCore, but in the strip "
      "holding chip (0,0,0) under McJAX; boundaries between slices cross DCN, which "
      "--transport relay (run on 2 and 4 slices) and ppermute (2) cross and native and xfer "
      "do not; on CPU MPMD_FAKE_SLICES=N cuts the devices into N slices)",
  )
  ap.add_argument("--no-precompile", action="store_true")
  ap.add_argument(
      "--embed",
      default="iota",
      choices=["iota", "gather"],
      help="token embedding: 'iota' = one_hot @ table (MaxText use_iota_embed=true, the "
      "baseline's), 'gather' = table[tokens] (the port's)",
  )
  ap.add_argument(
      "--ragged-sort",
      default="auto",
      choices=["auto", "on", "off"],
      help="auto: on iff EP > 1 (MaxText rejects use_ragged_sort at EP=1).  Not under --batch-split "
      "on Lineage's layers, which dispatch their own way",
  )
  ap.add_argument(
      "--cp",
      type=int,
      default=1,
      choices=[1, 2, 4],
      help="context parallelism: each microbatch sequence cut into this many chunks, one row of seq / cp "
      "tokens a stage device (cp x mbs = fsdp x ep); attention gathers the other chunks' kv "
      "latents over the expert axis",
  )
  ap.add_argument(
      "--layer-split",
      default="",
      help="layers per chunk in chunk order, comma-separated (e.g. 3,1: the head's chunk holds fewer); "
      "default an even split.  With --dense-layers it counts the MoE layers, the dense ones all in "
      "chunk 0 -- or, summing to --layers, every layer, in order, so the dense ones may span the "
      "first chunks (e.g. --layers 5 --dense-layers 2 --layer-split 1,2,1,1).  The schedulers' "
      "program costs (--costs) scale with each chunk's layers, a dense one as a MoE one",
  )
  ap.add_argument(
      "--ragged-buffer-factor",
      type=float,
      default=-1.0,
      help="MaxText ragged_buffer_factor: the ring of experts' dispatch buffer, <= 0 the worst case "
      "(dropless), > 0 that many times the balanced load (drops tokens past it; needs the "
      "ragged sort).  > 0 is refused under --batch-split on Lineage's layers, which are dropless "
      "(--lineage-capacity-factor sizes their expert loop)",
  )
  ap.add_argument(
      "--routing",
      default=None,
      choices=["random", "topk"],
      help="MaxText use_random_routing=true / false (MaxText requires topk at EP=1).  Default random, "
      "but topk under --batch-split on Lineage's layers, which route top-k only (random is "
      "refused there)",
  )
  ap.add_argument(
      "--batch-split",
      action="store_true",
      help="Lineage's batch split: each device's rows of a microbatch run as two halves through a "
      "chunk's MoE layers in phases, one half's EP all-gather / reduce-scatter beside the other "
      "half's attention (a Qwen3.5-family layer's Gated DeltaNet or gated attention) or experts -- "
      "by default on MaxText's Lineage layers for DeepSeek-V3 (TPU only), on ds_ep.split_layers for "
      "a Qwen3.5-family --model, whose layers Lineage does not build, and on GPU "
      "(--batch-split-impl).  A microbatch is two of the unsplit pipeline's, so it wants an even "
      "number of rows a device (gbs / mb / (fsdp x ep)), and --cp 1 and no --stash; and, as neither "
      "implementation keeps expert counts or an aux loss, no --bias-update-rate, --lb-loss-weight or "
      "--mtp.  A chunk without a MoE layer (the embedding's or the head's alone, or dense layers "
      "only) has nothing to split: split_layers runs it unsplit, Lineage's layers refuse it.  On "
      "Lineage's layers --routing defaults to topk and --shard-mode to explicit",
  )
  ap.add_argument(
      "--batch-split-impl",
      default=None,
      choices=list(ds_ep.BATCH_SPLIT_IMPLS),
      help="--batch-split: whose code runs the chunks' decoder layers; default the --model's: lineage "
      "for deepseek3, native for a Qwen3.5-family model (split_defaults).  lineage: MaxText's "
      "Lineage layers (ds_ep_lineage; its modules import typeguard and jaxtyping): its MLA and splash "
      "attention, router, SparseCore dispatch "
      "and combine and gmmv2 experts, chained as its driver chains them, the chunks' programs "
      "global-view jits on Explicit stage meshes.  TPU only: its kernels have no CPU or GPU path "
      "(refused there, but for a --dry-run).  It takes --shard-mode explicit and --routing topk "
      "(their defaults there), so a stage of more than one device (fsdp x ep > 1), --grad-reduce "
      "step, a dropless --ragged-buffer-factor, --cp 1, no --stash or --schedule stash, no "
      "--prefetch-pin, an even number of rows "
      "a device, a MoE layer in every chunk, and a model Lineage builds (ds_ep_lineage.check: "
      "DeepSeek-V3's layers only, bf16, sigmoid scores, ...); --gmm-impl, the tilings, --ragged-sort, "
      "--ep-collectives, --split-dw, --expert-weights, --attn, --sa-base2-exp off and "
      "--cost-estimate do not apply to its layers, --float32-gate-logits on only in part (each "
      "warned of when given).  native: ds_ep.split_layers, ds_ep's own layers over MaxText's "
      "ring-of-experts and megablox kernels -- DeepSeek-V3's off TPU only (on GPU its default; refused on "
      "TPU, where lineage runs its batch split: native_split_refusal), the Qwen3.5 family's only batch "
      "split, its mixer in attention's place (ds_ep.StageConfig.batch_split_impl)",
  )
  ap.add_argument(
      "--lineage-capacity-factor",
      type=float,
      default=ds_ep_lineage.CAPACITY_FACTOR,
      help="--batch-split on Lineage's layers: Lineage's MoE chunk size, in units of a device's balanced "
      "load of routed slots.  Dropless at any value: it sets how many trips its expert loop takes "
      "(MaxText's DeepSeek-V3 Lineage recipe: 3.0; ds_ep.StageConfig.lineage_capacity_factor)",
  )
  ap.add_argument(
      "--split-dw",
      default="acc",
      choices=list(ds_ep.SPLIT_DW),
      help="--batch-split on ds_ep.split_layers (--batch-split-impl native, a Qwen3.5-family model's "
      "default): acc adds mb0's routed-expert weight gradients into mb1's inside megablox's dW "
      "kernel (tgmm existing_out; the --tiles preset's wi dW tm capped at 256 for the VMEM it takes, "
      "ds_ep.split_acc_wi_tiling); sum gives each half its own copy and adds the two after "
      "(ds_ep.StageConfig.split_dw).  Lineage's layers accumulate the halves' expert gradients in "
      "their own kernels",
  )
  ap.add_argument(
      "--expert-weights",
      default="stacked",
      choices=list(ds_ep.EXPERT_WEIGHTS),
      help="how the routed experts' megablox kernels read a layer's weights: stacked, in place in the "
      "chunk's stacked leaves; sliced, from a per-layer slice XLA copies into a buffer first "
      "(ds_ep.StageConfig.expert_weights).  Not under --batch-split on Lineage's layers",
  )
  ap.add_argument(
      "--grad-reduce",
      default="step",
      choices=["step", "microbatch"],
      help="'microbatch': reduce-scatter every microbatch's gradient (MaxText's "
      "pipeline_fsdp_ag_per_repeat cadence); 'step': once per step (the only one under --shard-mode "
      "explicit, so under --batch-split on Lineage's layers)",
  )
  ap.add_argument(
      "--shard-mode",
      default=None,
      choices=list(ds_ep.SHARD_MODES),
      help="auto: Auto stage-mesh axes, nothing typed (a gradient accumulator carries a leading axis over "
      "the mesh axes its per-device partial sums are still to be reduced over); explicit: Explicit "
      "axes, the gathered weights typed reduced and the accumulators unreduced over those axes -- "
      "same programs, collectives and folding (ds_ep.StageConfig.shard_mode; needs --grad-reduce "
      "step).  Default auto, but explicit under --batch-split on Lineage's layers, whose code reads "
      "its layouts from sharding types (auto is refused there)",
  )
  ap.add_argument(
      "--rs-axes",
      default="joint",
      choices=list(ds_ep.RS_AXES),
      help="--fsdp-collectives xla: a gradient leaf stored over both mesh axes reduce-scatters in one "
      "psum_scatter over both (joint) or one axis at a time, fsdp first, but for the embedding and "
      "the head (each: on a v5p-128 stage of fsdp 4 x ep 16 the joint one compiles to a full-size "
      "all-reduce; on v6e and TPU7x to a reduce-scatter: ds_ep.StageConfig.rs_axes)",
  )
  ap.add_argument(
      "--gather-axes",
      default="joint",
      choices=list(ds_ep.GATHER_AXES),
      help="--fsdp-collectives xla: a weight stored over both mesh axes all-gathers in one all_gather over "
      "both (joint) or one axis at a time, the minor first (each: on a stage of a multi-slice v5p "
      "program the joint one compiles to a zero buffer + a full-size all-reduce, each to native "
      "SparseCore all-gathers; the same values bit for bit: ds_ep.StageConfig.gather_axes)",
  )
  ap.add_argument(
      "--prefetch-pin",
      default="off",
      type=prefetch_pin_arg,
      metavar="off|auto|SPEC",
      help="tie the gathers folded into a forward (a prefetch host's, the self-prefetch's) to points of "
      "its layers with optimization barriers, so they run inside its compute windows: hn (the MoE "
      "input, after attention), xs (the sorted tokens, before the expert gmms), y (gmm_wo's output, "
      "before the unsort).  auto: planned per program from estimated SparseCore gather and compute "
      "times, on the stages whose FSDP collectives are lax ones (xla) libtpu offloads to SparseCore "
      "(not a stage it runs them on synchronously); "
      "SPEC: point[@layer]=leaf,...;... by leaf name (ds_ep.leaf_names: moe.wi_0, moe.gate, "
      "self_attention.out, logits_dense, ...; a name selects the leaves under it too, of every "
      "chunk, or of one with a chunk prefix: c3.moe.wi_0; a Qwen3.5-family leaf is one layer's, "
      "named with its key in the chunk: 01.mlp.routed_experts.wi_0), e.g. "
      "'hn=moe.wi_0;y=moe.wo,logits_dense'; 'auto' is DeepSeek's only.  A pinned program compiles "
      "with xla_tpu_aggressive_opt_barrier_removal DISABLED (ds_ep.PIN_COMPILER_OPTIONS); off: no "
      "pins (the only one under --batch-split on Lineage's layers, which have no points to tie)",
  )
  ap.add_argument(
      "--ep-collectives",
      default="auto",
      type=ep_collectives_arg,
      metavar="auto|xla|ppermute|C0,C1,...",
      help="ring-of-experts gather / reduce-scatter.  auto: per stage, the lax "
      "collectives where libtpu offloads them to SparseCore, collective-permutes "
      "elsewhere; xla / ppermute: every stage; or one per stage, comma-separated "
      "(ppermute,xla is what auto picked in the TPU7x benchmark runs).  Not under --batch-split "
      "on Lineage's layers, which run their own: on SparseCore where libtpu offloads the stage's "
      "collectives, as TensorCore code elsewhere",
  )
  ap.add_argument(
      "--fsdp-scope",
      default=None,
      choices=["step", "repeat"],
      help="how long a chunk's gathered weights live.  step: gathered once a step, every chunk of "
      "the stage held by the backward phase.  repeat: per visit (a run of the chunk's ops on "
      "its stage) -- gathered inside the stage's previous visit, released after it, gathered "
      "again for the backward: about two chunks' weights at a time, for (V-1)/V more weight "
      "all-gather a step (MaxText's circular pipeline with pipeline_fsdp_ag_per_repeat).  "
      "Needs --schedule breadth, the default (or another order that "
      "visits each chunk at most twice a step), the fusion plan (no --no-overlap) and no F&B "
      "pairs.  Default (default_fsdp_scope): repeat under breadth with the fusion plan, else "
      "step",
  )
  ap.add_argument(
      "--costs",
      default="",
      help="program costs (ms) for --schedule timed / stash / interleaved: JSON "
      '(e.g. \'{"F0": 18.5, "F": 16.5, "B0": 51.3, "B": 48.6, "A": 58.3, "xfer": 1.0}\': '
      "F0/B0 chunk 0, A the fused last chunk, xfer a stage-boundary copy; per-chunk and "
      "stash keys as schedule.cost_table reads them), @file.json as written by `xprof "
      "costs`, or a table name: tpu7x, v6e (measured with padded tiles), v6e_tiles "
      "(with --tiles v6e).  Default: on a v6e the table of --tiles (v6e_tiles for v6e and "
      "clamped, v6e for custom tilings: default_costs), TPU7x ones elsewhere; "
      "its keys are overridden by these.  Under an uneven --layer-split every key is an even "
      "split's chunk, each chunk's scaled to its layers (split_costs)",
  )
  ap.add_argument(
      "--rs-self",
      default="all",
      choices=["all", "first", "none"],
      help="which stages fold their last chunk's reduce-scatter into its own last backward: "
      "all, or first -- only the stage holding chunk 0, whose last backward ends the "
      "step (elsewhere the folded reduce-scatter holds back the cotangent the stage "
      "upstream waits for; a standalone reduce right after it does not).  Under first "
      "the other stages' last backward hosts no reduce-scatter at all: other chunks' "
      "groups move to earlier backwards or to the standalone reduce after it",
  )
  ap.add_argument(
      "--rs-reserve",
      default="auto",
      choices=["auto", "on", "off"],
      help="keep each stage's last backward free for its own reduce-scatter (rs_self) "
      "when another chunk's would take it; auto: under --placement v and for "
      "--schedule interleaved (circular breadth / search / timed / stash "
      "keep the plans they have always made, which can end a stage with a standalone reduce-scatter: on "
      "folds it). "
      "It frees one backward, so it folds a single leftover reduce-scatter: at 3+ "
      "chunks per stage a stage can still end with a standalone one",
  )
  ap.add_argument(
      "--pf-split",
      type=int,
      default=4,
      help="ops each chunk gather is spread over: forwards before the chunk's first one; under "
      "--fsdp-scope repeat the last ops of the stage's previous visit (0: none -- the update "
      "gathers the chunk, or under repeat a visit keeps its chunk's weights from the last)",
  )
  ap.add_argument("--rs-split", type=int, default=4, help="backwards each chunk reduce-scatter is spread over")
  ap.add_argument(
      "--fsdp-collectives",
      default="auto",
      choices=["auto", "ppermute", "xla"],
      help="auto: collective-permutes (async TensorCore DMA) where fsdp, ep <= 2 (at fsdp = ep "
      "= 1 there is nothing to gather: xla), lax collectives (xla) otherwise",
  )
  ap.add_argument("--no-overlap", action="store_true", help="stage-level gather / reduce programs instead of fused ones")
  ap.add_argument("--steps", type=int, default=10)
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--bench", action="store_true")
  ap.add_argument(
      "--profile",
      default="",
      help="trace the --profile-steps steps after the warm-up (and --bench) into this XProf logdir; under "
      "Pathways (JAX_PLATFORMS=proxy) a gs:// one, where the Pathways workers write their device planes",
  )
  ap.add_argument("--profile-steps", type=int, default=2)
  ap.add_argument("--lr", type=float, default=1e-5)
  ap.add_argument(
      "--adam-math",
      default="fp32",
      choices=["fp32", "optax"],
      help="the AdamW update's arithmetic: fp32 (whatever the storage dtype), or optax's chain "
      "as MaxText runs it, every op in its leaf's dtype (bf16 moments and learning rate "
      "with bf16 weights): for parity runs against MaxText.  The clip compares this "
      "pipeline's fp32 global norm (MaxText's optax reduces it in bf16), so the update "
      "matches MaxText's bit for bit while the norm stays below the clip, or both round "
      "to the same bf16 value",
  )
  ap.add_argument(
      "--grad-clip",
      type=float,
      default=1.0,
      help="the global-norm clip's threshold (MaxText's gradient_clipping_threshold); 0: no clip",
  )
  ap.add_argument(
      "--data",
      default="synthetic",
      help="synthetic (numpy random tokens, targets the tokens rolled by one) | maxtext "
      "(MaxText's dataset_type=synthetic batch, bit for bit, drawn as MaxText draws it, under "
      "threefry2x32) | "
      "npy:PATH ([gbs, seq + 1] integer tokens in [0, vocab), local or gs://; inputs "
      "[:, :-1], targets [:, 1:])",
  )
  ap.add_argument(
      "--init-from",
      default="",
      help="load the weights from DIR/chunk<c>.npz (local or gs://; maxtext_parity.py writes "
      "them from a MaxText init) instead of initializing them",
  )
  ap.add_argument(
      "--max-ahead",
      type=int,
      default=None,
      help="in the --bench and --profile loops, dispatch at most this many steps beyond the "
      "last one finished (0: block after every step; default: no bound, but under Pathways "
      f"(JAX_PLATFORMS=proxy) {PATHWAYS_MAX_AHEAD}, PATHWAYS_MAX_AHEAD -- a large number for practically none)",
  )
  ap.add_argument(
      "--json",
      default="",
      help="write the run record here (process 0), a local path or gs:// (a Pathways head pod's disk goes "
      "with its pod); under --distributed the processes of "
      "--json-procs write theirs next to it, .p<process> before the extension (run.json -> "
      "run.p3.json).  With --profile it is written before the profiled steps as well, so that a run cut short "
      "while profiling keeps it; a local file is replaced whole (written beside it, then renamed, its mode kept; "
      "a symlinked record replaces the file the link names), while this process's stdout or stderr by any name "
      "(/dev/stdout, a link to it, the log's own path) or another of its descriptors (/dev/fd/N) takes each record "
      "after what was printed there before, and a named pipe or another target that is not a regular file is written "
      "in place, and only after the profiled steps (a pipe keeps nothing across a crash, and its reader may stop at the "
      "first record's end)",
  )
  ap.add_argument(
      "--json-procs",
      default="leaders",
      help="--distributed: which processes write a run record besides process 0: leaders "
      "(each stage's first), all, or a comma list.  A value other than leaders is refused under Pathways (one "
      "client: --json)",
  )
  ap.add_argument(
      "--mem-report",
      action="store_true",
      help="print per-program compile-time memory and the persistent state per device",
  )
  ap.add_argument(
      "--grad-digest",
      type=int,
      default=0,
      metavar="N",
      help="diagnostics: every process logs a digest of each gradient leaf of its local chunks (its "
      "shards, reduced, before the update) in the first N warmup steps -- to find, between two runs "
      "of one setup, the first gradients that differ. Blocks each of those steps once",
  )
  ap.add_argument(
      "--distributed",
      action="store_true",
      help="multi-controller JAX: jax.distributed.initialize() (one process per host; a "
      "process runs the stages on its devices and takes part in every boundary copy)",
  )
  ap.add_argument(
      "--transport",
      default="native",
      choices=["native", "ppermute", "xfer", "relay"],
      help="boundary copies: native (jax's copy: batched D2D in one process, its cross-host copy across "
      "processes -- on TPU v6e and v5p with jax 0.11.1 / libtpu 0.0.46 every receiving host aborts "
      "in it), ppermute (a collective-permute program over "
      "the two stages' devices, which both run: ICI inside a slice), xfer "
      "(jax.experimental.transfer between host pairs, async, over DCN) or relay (a host relay: D2H, "
      "TCP between the hosts, H2D, async, no libtpu cross-host path -- for across slices; "
      "MPMD_RELAY_STREAMS / _CHUNK_MB / _TIMEOUT / _SOCKBUF_MB / _IP / _PORT tune it, "
      "transfer.RelayContext; each process logs where its time went, transfer.RELAY_MEASURES, after "
      "the warm-up and after --bench / --profile)",
  )
  ap.add_argument(
      "--profile-procs",
      default="leaders",
      help="--profile on which processes: leaders (each stage's first), all, or a comma list; across processes "
      "they write one XProf session, on one clock (xprof_align).  A value other than leaders is refused under "
      "Pathways (--profile-hosts picks its traced worker hosts)",
  )
  ap.add_argument(
      "--profile-hlo-proto",
      choices=("auto", "on", "off"),
      default="auto",
      help="--profile: whether the trace carries the compiled programs' HLO protos (its "
      "/host:metadata plane; the XProf UI's op and graph views read them, xprof.py does not).  "
      "auto, the default on every platform, TPU too: off when this process's lowered programs "
      "pass HLO_PROTO_AUTO_BYTES (1 GiB) of text -- a trace past protobuf's 2 GB is not written at "
      "all, and a one-stage layout's 383 DeepSeek-V3 programs' protos alone were 2.6 GB; on: "
      "always, as every trace did before this flag",
  )
  ap.add_argument(
      "--profile-chips",
      type=profiling.chip_count,
      default=None,
      help="--profile: how many chips each traced process profiles, its first ones (libtpu "
      "tpu_num_chips_to_profile_per_task), or 'all'.  Default profiling.CHIPS, 1: one TPU a traced process, "
      "one a slice under --layout slices; a process holding several stages (the one process of a "
      "run, say) shows its first chip's stage only, unless this covers the others' chips.  Under "
      "Pathways (JAX_PLATFORMS=proxy) the default is all: every chip of each traced worker host",
  )
  ap.add_argument(
      "--profile-hosts",
      type=profiling.positive_int,
      default=None,
      help="--profile under Pathways (JAX_PLATFORMS=proxy): how many Pathways worker hosts trace "
      "(pathwaysutils' max_num_hosts; a multi-controller run picks its traced processes with "
      "--profile-procs).  Default $MPMD_PROFILE_HOSTS, else 1",
  )
  return ap


def load_tokens(path):
  """[gbs, seq + 1] int tokens from a .npy file (local or gs://)."""
  return np.load(io.BytesIO(xprof_align.read_bytes(path)))


def missing_input(a, p):
  """The message to stop with if a file the parsed flags `a` name for this
  process of pipeline `p` to read does not exist, else None: --init-from's
  chunk<c>.npz for this process's stages (the files
  `Pipeline.chunk_trees_from` reads), --data npy:'s file; local or gs://.
  main asks before anything compiles, so a mistyped path fails in seconds,
  not after the precompile."""
  exists = lambda path: xprof_align._blob(path).exists() if path.startswith("gs://") else os.path.exists(path)
  if a.init_from:
    for ch in p.chunks:
      path = f'{a.init_from.rstrip("/")}/chunk{ch.index}.npz'
      if (not p.multi or ch.stage in p.placed) and not exists(path):  # in one controller every chunk's
        return f"--init-from {a.init_from}: no such file {path}"
  if a.data.startswith("npy:") and not exists(a.data[len("npy:") :]):
    return f"--data {a.data}: no such file"
  return None


def missing_input_anywhere(msg, timeout_s=900):
  """Multi-controller: this process's `missing_input` answer `msg` -> the
  message every process stops with if any process lacks a file, else None.

  Each process checks only the files it reads (its stages' chunks, maybe in a
  directory local to its host), so a file some hosts lack would stop only
  those: the others would wait for them at the next barrier until the
  coordination service aborted every process, minutes later, without saying
  why.  So each process lacking a file publishes its message through the
  key-value store (no TPU program); after a barrier every process reads them
  all, so all of them stop together, each naming the processes lacking a file
  and the file."""
  from jax._src import distributed

  client = distributed.global_state.client
  if msg:
    client.key_value_set(f"mpmd/inputs/{jax.process_index()}", msg)
  client.wait_at_barrier("mpmd_inputs", timeout_s * 1000)
  groups = collections.defaultdict(list)  # message -> the processes giving it
  for key, val in client.key_value_dir_get("mpmd/inputs/"):
    groups[val].append(int(key.rsplit("/", 1)[1]))
  if not groups:
    return None
  return (
      f"input files missing on {sum(map(len, groups.values()))} of {jax.process_count()} processes, so every "
      "process stops: "
      + "; ".join(
          f'{m} (process{"es" if len(q) > 1 else ""} {_ranges(q)})'
          for m, q in sorted(groups.items(), key=lambda kv: min(kv[1]))
      )
  )


def write_chunk_files(out, trees):
  """Chunk-order weight trees (nested dicts of arrays, as `chunk_trees`
  gives) -> out/chunk<c>.npz (out local or gs://), the files
  `Pipeline.chunk_trees_from` reads (--init-from): keys are the '.'-joined
  tree paths, bfloat16 travels as its uint16 bits."""

  def flat(node, prefix=()):
    for k, v in node.items():
      if not isinstance(k, str) or "." in k:
        raise ValueError(f'chunk tree key {k!r} at {".".join(prefix) or "the root"}: want a str without "."')
      if isinstance(v, dict):
        yield from flat(v, prefix + (k,))
      else:
        yield ".".join(prefix + (k,)), v

  for c, tree in enumerate(trees):
    arrays = {}
    for k, v in flat(tree):
      v = np.asarray(v)
      arrays[k] = v.view(np.uint16) if v.dtype == jnp.bfloat16 else v
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    xprof_align.write_bytes(f'{out.rstrip("/")}/chunk{c}.npz', buf.getvalue())


def init_distributed():
  """jax.distributed.initialize(), before the backend exists.

  The cross-host copy flags are plain flags jax does not read from the
  environment: take them from it here (CPU needs the socket address; on TPU
  they select the DCN path instead of the plugin's own).  Cloud TPU / GKE
  detect the rest; elsewhere (a CPU test on one machine)
  JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES and JAX_PROCESS_ID say who is who.
  Under Pathways (JAX_PLATFORMS=proxy) a SystemExit: one client drives every
  device there, and jax.distributed would look for a cluster in its pod."""
  if pathways.is_proxy():
    raise SystemExit(
        "--distributed: JAX_PLATFORMS=proxy is Pathways, where ONE client process drives every device of the job "
        "(jax.process_count() is 1) and the pipeline runs its one-process path: run without --distributed"
    )

  def flag(v, name):
    # jax's own spellings of a boolean environment variable (config.bool_env)
    v = v.strip().lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
      return True
    if v in ("n", "no", "f", "false", "off", "0"):
      return False
    raise ValueError(f"{name.upper()}={v!r}: want 1/0, true/false, yes/no or on/off")

  for name, cast in (
      ("jax_cross_host_transfer_socket_address", str),
      ("jax_cross_host_transport_addresses", str),
      ("jax_force_dcn_cross_host_transfers", flag),
      ("jax_cross_host_transfer_transfer_size", int),
      ("jax_cross_host_transfer_timeout_seconds", int),
  ):
    if os.environ.get(name.upper()):
      v = os.environ[name.upper()]
      jax.config.update(name, cast(v, name) if cast is flag else cast(v))
  kw = {
      k: int(os.environ[e])
      for k, e in (("num_processes", "JAX_NUM_PROCESSES"), ("process_id", "JAX_PROCESS_ID"))
      if e in os.environ
  }
  # MPMD_COORD_PORT: a coordinator port of this run's own (TPU pods), so that
  # a host late from an earlier run cannot join this one.  jax's cluster
  # detection puts it on worker 0 (after waiting for that name to resolve).
  # On several slices that is MEGASCALE_COORDINATOR_ADDRESS's host (slice 0's
  # worker 0; jax drops that address's port) at this same port, and jax
  # numbers the processes slice x len(worker list) + worker of slices x
  # len(worker list) (jax 0.11.1's clusters/cloud_tpu_cluster.py; jax's main
  # of 2026-10-11 differs only in reading the worker list from
  # TPU_PROCESS_ADDRESSES_PATH's file first; ds_ep_test's
  # MultiController.test_multislice_coordinator).  A JAX_COORDINATOR_PORT
  # already set wins (a step running two programs can give each its own),
  # with a warning: in the pod's environment it would give every step one port.
  if os.environ.get("MPMD_COORD_PORT"):
    port = os.environ.setdefault("JAX_COORDINATOR_PORT", os.environ["MPMD_COORD_PORT"])
    if port != os.environ["MPMD_COORD_PORT"]:
      print(
          f"init_distributed: warning: JAX_COORDINATOR_PORT={port} takes the place of this step's "
          f'MPMD_COORD_PORT={os.environ["MPMD_COORD_PORT"]}: runs on one port can join each other\'s '
          "coordinator",
          flush=True,
      )
  # GPU pods (a GKE GPU JobSet): nothing autodetects the coordinator
  # there, so the JobSet names its pod 0 (MAXTEXT_MPMD_COORD_HOST) and each step
  # its port (MPMD_COORD_PORT, 8500 + step), as jax would put it on TPU.
  if not os.environ.get("JAX_COORDINATOR_ADDRESS") and os.environ.get("MAXTEXT_MPMD_COORD_HOST"):
    port = os.environ.get("JAX_COORDINATOR_PORT") or os.environ.get("MPMD_COORD_PORT") or "8482"
    os.environ["JAX_COORDINATOR_ADDRESS"] = f"{os.environ['MAXTEXT_MPMD_COORD_HOST']}:{port}"
  # Pods of a fresh slice start minutes apart (each pulls the image): wait
  # for all of them, not jax's default 300 s.
  kw["initialization_timeout"] = int(os.environ.get("MPMD_INIT_TIMEOUT", "1800"))
  jax.distributed.initialize(**kw)


def multi_controller_only(a):
  """What of `a` (make_parser's flags) asks for multi-controller JAX alone -> [message], in this order: --transport
  relay or xfer (pathways.check_transport's message, its one copy), --distributed, --json-procs and --profile-procs
  at a value other than their default (leaders), --norm-reduce allreduce -- each saying what to use instead.  main
  and mem_probe refuse them under Pathways (JAX_PLATFORMS=proxy) before the backend starts, in the same words: one
  client process drives every device there (pathways.multi_controller() is False), so none of them could act, and
  taking one silently would hide that.  Elsewhere they stay as they are: a one-host run takes them (--norm-reduce
  allreduce, the GPU default, runs the hops in one process; --distributed initializes a one-host slice's one process).
  The multi-controller environment is not refused, since it may be set cluster-wide: MPMD_COORD_PORT,
  JAX_COORDINATOR_*, MPMD_INIT_TIMEOUT, JAX_CROSS_HOST_*, MPMD_RELAY_* and MPMD_XFER_* are read only under
  --distributed or the relay / xfer, refused here, and the import of this module skips MPMD_CACHE_EVERY_PROCESS under
  the proxy."""
  out = []
  try:
    pathways.check_transport(getattr(a, "transport", "native"))
  except ValueError as e:
    out.append(str(e))
  if getattr(a, "distributed", False):
    out.append(
        "--distributed: under Pathways (JAX_PLATFORMS=proxy) one client process drives every device, so there is no "
        "jax.distributed to initialize: run without it"
    )
  for flag, dest, instead in (
      ("--json-procs", "json_procs", "leave it out: the client writes the one run record, --json"),
      ("--profile-procs", "profile_procs", "--profile-hosts N (or MPMD_PROFILE_HOSTS) traces N Pathways worker hosts"),
  ):
    if (procs := getattr(a, dest, "leaders")) != "leaders":
      out.append(
          f"{flag} {procs}: picks processes of a multi-controller run (--distributed); under Pathways one client "
          f"process drives every device: {instead}"
      )
  if getattr(a, "norm_reduce", "hops") == "allreduce":
    out.append(
        "--norm-reduce allreduce: one sum over every device of a multi-controller run (--distributed); under "
        "Pathways one client holds every stage's sum of squares and hops it to the others, a scalar copy each: "
        "--norm-reduce hops (the default)"
    )
  return out


def plan_fingerprint(p, extra=()):
  """Everything that decides which boundary copies and gradient-norm hops a
  process issues, and how they move: the order and its fusion plan, the
  placement, the transport, the stage meshes (which device of which process
  holds each shard, and their axis types), the shard mode, the activation's
  shape and dtype; what else of the plan decides a stage's programs, which
  its processes must compile alike: the prefetch pins (an 'auto' plan also
  reads LIBTPU_INIT_ARGS and the client's slices) and how the gathers and
  reduce-scatters split their collectives; plus `extra` (main adds every
  flag but --json, the run loop's step counts among them) -> a tuple with a
  deterministic repr."""
  cfg = p.scfg.model
  return (
      p.order,
      tuple(p.place),
      p.placement,
      p.transport,
      p.overlap,
      sorted(p.modes.items()),
      sorted(p.fuse),
      sorted(p.pf.items()),
      sorted(p.pf_n.items()),
      p.fsdp_scope,
      sorted(p.release_at.items()),
      sorted(p.rs_at.items()),
      sorted(p.rs_self.items()),
      p.final_slots,
      p.gather_slots,
      p.mbs,
      p.seq,
      p.cp,
      p.S,
      p.C,
      cfg.emb_dim,
      str(jnp.dtype(cfg.dtype)),
      p.scfg.shard_mode,
      [m.device_ids.tolist() for m in p.meshes],
      [tuple(m.axis_names) for m in p.meshes],
      [tuple(str(t) for t in m.axis_types) for m in p.meshes],
      sorted(p.pins.items()),
      p.scfg.gather_axes,
      p.scfg.rs_axes,
      tuple(extra),
  )


def host_barrier(name, timeout_s=1800):
  """Multi-controller: wait until every process gets here, through the
  coordination service (a host barrier: no TPU program).  main holds every
  process at one after the compiles and one after init, so no stage starts
  programs that its peers' devices (or, under --transport ppermute, the next
  stage's) must join while another process is still compiling: on 2 x
  TPU7x-4x4x4 a stage's first programs, launched while the other stage still
  compiled, failed the slices (SLICE_FAILURE_SW_INJECT_ERROR), and a fused
  last chunk halted its stage's cores on a launch-id mismatch."""
  from jax._src import distributed

  distributed.global_state.client.wait_at_barrier(name, timeout_s * 1000)


def check_plan_agreement(p, timeout_s=900, extra=()):
  """Multi-controller: stop every process unless all of them built the same
  plan (`plan_fingerprint`).

  Each process walks the order and issues every boundary copy in it; two
  processes with different plans (a schedule search that is not
  deterministic, different flags) would pair copies wrongly -- silently --
  or wait on copies that never come, and two of one stage that compiled its
  programs differently (other prefetch pins) would hang in their
  collectives.  Each process publishes its hash through the coordination
  service's key-value store (no TPU program); after a barrier every process
  compares all of them, so all of them stop together, each naming the
  processes that differ."""
  from jax._src import distributed

  h = hashlib.sha256(repr(plan_fingerprint(p, extra)).encode()).hexdigest()
  client = distributed.global_state.client
  client.key_value_set(f"mpmd/plan/{jax.process_index()}", h)
  client.wait_at_barrier("mpmd_plan_agreement", timeout_s * 1000)
  groups = collections.defaultdict(list)
  for key, val in client.key_value_dir_get("mpmd/plan/"):
    groups[val].append(int(key.rsplit("/", 1)[1]))
  if len(groups) > 1:
    raise RuntimeError(
        f"the processes built {len(groups)} different plans ("
        + "; ".join(f"{v[:12]}: processes {sorted(q)}" for v, q in groups.items())
        + "): they would issue different boundary copies"
    )
  return h


def lowered_print(lowered):
  """A lowered program's print, cheap to compare across processes and runs:
  {'sha': the sha256 of its text (16 hex digits), 'bytes', 'consts',
  'funcs': the text's size and its constant / function counts} -- the counts
  say at a glance how two prints of one program differ (one constant more:
  a table copied where the other shares it)."""
  txt = lowered.as_text()
  return {
      "sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
      "bytes": len(txt),
      "consts": txt.count("stablehlo.constant"),
      "funcs": txt.count("func.func"),
  }


def executable_print(compiled):
  """The sha256 (16 hex digits) of a compiled TPU program's executable
  fingerprint, from which jaxlib derives the id of each of its launches (with
  the count of launches before) -- or None off TPU, where nothing launches by
  id and the fingerprint is not even stable (CPU: a program compiled twice in
  one process can print two ways; on TPU a fresh compile, a second one and
  one read back from the persistent cache print alike), and where the
  backend gives none (an executable for a compile-only topology may refuse
  to)."""
  try:
    exe = compiled.runtime_executable()
    if {d.platform for d in exe.local_devices()} != {"tpu"}:
      return None
    fp = getattr(exe, "fingerprint", None)
  except Exception:  # an UNIMPLEMENTED from the backend: no print, nothing to compare
    return None
  if fp is None:
    return None
  return hashlib.sha256(fp if isinstance(fp, bytes) else str(fp).encode()).hexdigest()[:16]


def _stage_key(stage):
  """A program's stage as the key-value store keys it: '1', or '0-1' for a
  boundary copy over stages 0 and 1."""
  return "-".join(map(str, stage)) if isinstance(stage, tuple) else str(stage)


def _ranges(ids):
  """[0, 1, 2, 5, 7, 8] -> '0-2,5,7-8'."""
  ids, out = sorted(ids), []
  for i in ids:
    if out and i == out[-1][1] + 1:
      out[-1][1] = i
    else:
      out.append([i, i])
  return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


# Numbers the key-value store keys of each `check_program_agreement`: every process makes its checks in one order.
_PROGRAM_KEYS = itertools.count()


def check_program_agreement(p, prints, what, timeout_s=3600, show=6):
  """Multi-controller: stop every process, before any program launches,
  unless each program is the same on all the processes that run it.

  `prints` are this process's [(name, stage, fingerprint, stats)]
  (`Pipeline.precompile`'s `agree`): what='lowered', the sha256 of each
  lowered text with its `lowered_print` stats, or 'executables', of each
  executable's fingerprint.  A stage's programs run on all its processes (a
  boundary copy's, stage (a, b), on both stages'), and processes that launch
  different programs at one launch -- a precompile racing in a kernel's
  cache, a compiler that is not deterministic -- halt the TPU runtime long
  after the cause ('an unexpected peer shows up in the launch group with a
  different launch id than the current group leader').  So each process
  publishes a digest of its programs per stage, and its prints, in the
  coordination service's key-value store (no TPU program); after a barrier
  every process compares every stage's digests and, where they differ,
  fetches the prints of one process per digest: all of them stop together,
  each naming the programs that differ, their prints and the processes
  holding each.  -> {stage key: digest}."""
  from jax._src import distributed

  client = distributed.global_state.client
  pid, n = jax.process_index(), next(_PROGRAM_KEYS)
  key = f"mpmd/programs/{what}/{n}"
  seen, named, mine = collections.Counter(), [], collections.defaultdict(list)
  for name, stage, fp, st in prints:  # a name twice on a stage: name, name#1, ...
    sk = _stage_key(stage)
    k = seen[(name, sk)]
    seen[(name, sk)] += 1
    named.append((f"{name}#{k}" if k else name, sk, fp, st))
    mine[sk].append((named[-1][0], fp))
  for sk, progs in mine.items():
    client.key_value_set(f"{key}/digest/{sk}/{pid}", hashlib.sha256(json.dumps(sorted(progs)).encode()).hexdigest()[:16])
  client.key_value_set(f"{key}/prints/{pid}", json.dumps(named))
  client.wait_at_barrier(f"mpmd_programs_{what}_{n}", timeout_s * 1000)
  digests = collections.defaultdict(lambda: collections.defaultdict(set))  # stage key -> digest -> processes
  for k, v in client.key_value_dir_get(f"{key}/digest/"):
    sk, q = k.rsplit("/", 2)[1:]
    digests[sk][v].add(int(q))

  def procs(sk):
    return frozenset().union(*(p.stage_procs[int(s)] for s in sk.split("-")))

  bad = sorted(
      (sk for sk in digests if len(digests[sk]) > 1 or set().union(*digests[sk].values()) != procs(sk)),
      key=lambda sk: [int(s) for s in sk.split("-")],
  )
  if not bad:
    return {sk: next(iter(d)) for sk, d in digests.items()}
  cache = {}

  def prints_of(q):
    if q not in cache:
      cache[q] = {
          (nm, sk): (fp, st)
          for nm, sk, fp, st in json.loads(client.blocking_key_value_get(f"{key}/prints/{q}", timeout_s * 1000))
      }
    return cache[q]

  def said(fp, st):
    return fp if not st else f'{fp} ({st["bytes"]} bytes, {st["consts"]} constants, {st["funcs"]} funcs)'

  lines = []
  for sk in bad:
    groups = sorted(digests[sk].items(), key=lambda kv: min(kv[1]))
    missing = procs(sk) - set().union(*digests[sk].values())
    lines.append(
        f'stage {sk}: {len(groups)} program set{"s" if len(groups) > 1 else ""} ('
        + "; ".join(f"{d}: processes {_ranges(q)}" for d, q in groups)
        + (f"; none from processes {_ranges(missing)}" if missing else "")
        + ")"
    )
    reps = [min(q) for _, q in groups]
    names = sorted({nm for r in reps for nm, s in prints_of(r) if s == sk})
    differ = [nm for nm in names if len({prints_of(r).get((nm, sk), (None,))[0] for r in reps}) > 1]
    for nm in differ[:show]:
      lines.append(
          f"  {nm}: "
          + "; ".join(
              f'{said(*prints_of(r)[(nm, sk)]) if (nm, sk) in prints_of(r) else "not lowered"} on processes '
              f"{_ranges(q)}"
              for r, (_, q) in zip(reps, groups)
          )
      )
    if len(differ) > show:
      lines.append(f"  ... and {len(differ) - show} more: {differ[show:]}")
  raise RuntimeError(
      f"the processes hold different {what} programs, which would halt the TPU runtime at their first "
      'launch ("a different launch id than the current group leader"):\n' + "\n".join(lines)
  )


def grad_digest_lines(p, grads, step):
  """This process's reduced gradients (`Pipeline.step`'s, before the update)
  -> a line per local chunk: each leaf's sha256 (8 hex digits) over its local
  shards' bytes, in device order -- equal between two runs of one setup only
  where each shard is bit for bit."""
  out = []
  for s in sorted(p.local) if p.multi else range(p.S):
    for ch, tree in zip(p.stage_chunks[s], grads[s]):
      leaves = []
      for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]:
        h = hashlib.sha256()
        for sh in sorted(x.addressable_shards, key=lambda sh: sh.device.id):
          h.update(np.ascontiguousarray(np.asarray(sh.data)).reshape(-1).view(np.uint8))
        leaves.append(f'{jax.tree_util.keystr(path, simple=True, separator=".")}={h.hexdigest()[:8]}')
      out.append(f"grad digest step {step} c{ch.index}: " + " ".join(leaves))
  return out


def program_prints_lines(p, stage):
  """Stage `stage`'s programs as precompile printed them (`program_prints`):
  one line per program -- the lowered text's sha256, size and constant
  count, and the executable's sha256."""
  rows = [(n, lw, x) for n, s, lw, x in getattr(p, "program_prints", []) if s == stage]
  digest = hashlib.sha256(json.dumps(sorted((n, lw["sha"], x) for n, lw, x in rows)).encode()).hexdigest()[:16]
  out = [
      f"stage {_stage_key(stage)}: {len(rows)} programs, digest {digest} (each: sha256 of the lowered text, its "
      "bytes and constants; of the executable fingerprint)"
  ]
  out += [f'  {n} {lw["sha"]} {lw["bytes"]}B {lw["consts"]}c {x or "-"}' for n, lw, x in rows]
  return out


def shared_profile_session(timeout_s=900):
  """Multi-controller --profile: (session name, origin ns since the epoch)
  every process agrees on.  After a barrier -- the profiled steps start from
  a drained pipeline on every stage -- process 0 takes its clock less 10 ms
  (host clocks agree to far less: NTP) as the origin every traced host
  rebases onto, and names the session after it in local time, as jax names
  a one-process session (pathways.session_id): a logdir's sessions go by
  their names (XProf, xprof.find_xplane)."""
  from jax._src import distributed

  client = distributed.global_state.client
  client.wait_at_barrier("mpmd_profile_start", timeout_s * 1000)
  if jax.process_index() == 0:
    origin = time.time_ns() - 10_000_000
    session = pathways.session_id(origin / 1e9)
    client.key_value_set("mpmd/profile_session", f"{session} {origin}")
  else:
    session, origin = client.blocking_key_value_get("mpmd/profile_session", timeout_s * 1000).split()
  return session, int(origin)


# --profile-hlo-proto auto's bound on this process's lowered program text: their HLO protos in a trace run ~1.2x it
# (one-stage layouts on GB200: the Qwen3.8 16-layer one's 174 programs, 0.53 GB of text, 0.63 GB of protos;
# DeepSeek-V3's 383, 2.14 GB and 2.63 GB, whose trace was then not written), and protobuf writes no message past 2 GB.
HLO_PROTO_AUTO_BYTES = 1 << 30


def profile_hlo_proto(mode, p):
  """--profile-hlo-proto -> does this process's trace keep its programs' HLO protos.  'auto': yes unless the
  lowered text of the programs it compiled (`program_prints`: none without a precompile) passes
  HLO_PROTO_AUTO_BYTES."""
  if mode != "auto":
    return mode == "on"
  return sum(lw.get("bytes", 0) for _, _, lw, _ in getattr(p, "program_prints", ())) <= HLO_PROTO_AUTO_BYTES


def proc_set(spec, p, flag="--profile-procs"):
  """--profile-procs / --json-procs -> a set of processes: leaders (each
  stage's first process), all, or a comma list of process ids (SystemExit on
  anything else; every process parses the same flags, so all stop together)."""
  if spec == "all":
    return set(range(jax.process_count()))
  if spec == "leaders":
    return {min(p.stage_procs[s]) for s in range(p.S)}
  try:
    got = {int(x) for x in spec.split(",")}
  except ValueError:
    raise SystemExit(f"{flag}: leaders, all or a comma list of process ids, not {spec!r}") from None
  if bad := sorted(got - set(range(jax.process_count()))):
    raise SystemExit(f"{flag}: no process {bad} (there are {jax.process_count()})")
  return got


# --max-ahead's default under Pathways: the steps the client may dispatch beyond the last one finished.  Nothing else
# bounds it there -- the IFRT proxy client takes no PJRT options (transfer.set_max_inflight is a local client's) --
# and a local Pathways run queued ~90 steps, so that a worker running out of memory surfaced only as a dropped proxy
# connection, steps after the step that caused it.  run_steps waits for the steps to finish beside the dispatch
# (`_Finished`), so bounded or not a short step takes the same time there.
PATHWAYS_MAX_AHEAD = 2


class _Finished:
  """run_steps' bound on running ahead: a thread of its own waits, in order, for each step's outputs it is handed
  (`watch`) to be ready, and `wait` returns once the oldest step not yet waited for has finished -- or raises what
  waiting for it raised.  Through the IFRT proxy even a wait for a step long finished is a round trip: made in the
  dispatch loop, as a jax.block_until_ready a step, it slowed short steps by a quarter (local Pathways, tpu7x-8, M=1
  S=2: 2.0 ms a step unbounded, 2.6 so bounded); made here, beside the dispatch, by nothing measurable (2.0)."""

  def __init__(self):
    self._todo = queue.SimpleQueue()
    self._done = threading.Semaphore(0)
    self._errors = []
    threading.Thread(target=self._run, name="mpmd-run-ahead", daemon=True).start()

  def _run(self):
    while (outs := self._todo.get()) is not None:
      try:
        jax.block_until_ready(outs)
      # the step's own error, whatever it is: `wait` raises it in the dispatch loop
      except BaseException as e:  # pylint: disable=broad-exception-caught
        self._errors.append(e)
      del outs
      self._done.release()

  def watch(self, outs):
    self._todo.put(outs)

  def wait(self):
    self._done.acquire()  # pylint: disable=consider-using-with  # one finished step, counted off: no release
    if self._errors:
      raise self._errors[0]

  def close(self):
    self._todo.put(None)


def run_steps(p, state, n, batch, lrs, max_ahead=None):
  """n steps of `p` back to back (`Pipeline.step`), with at most `max_ahead`
  of them dispatched beyond the last one this process has seen finish --
  every local stage's update and the loss partials, waited for beside the
  dispatch (`_Finished`); no bound for None -> (params, opts, wc0, lacc,
  gnorms) after the last.

  `state` is [params, opts, wc0], emptied first, so that the caller holds no
  reference to what the first step consumes.  The gathered weights wc0 are not
  donated: a caller that kept them for the whole loop held a second copy (2.3
  GB a stage-0 device on 2 x v6e-128 at E384), and on 31 GiB chips that
  fragmented HBM so that stage 0's first chunk-0 backward, which allocates
  the chunk's gradient accumulators, stopped to defragment it in nearly every
  run-ahead step (85-104 ms, 2026-10).  The loop that blocks every step
  rebinds them each step, so it never had the copy."""
  params, opts, wc0 = state
  state.clear()
  lacc = gn = None
  finished = None if max_ahead is None else _Finished()
  try:
    for k in range(n):
      if finished is not None and k > max_ahead:
        finished.wait()  # step k - max_ahead - 1
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      if finished is not None:
        finished.watch((p.step_gnorms, lacc))  # never donated
  finally:
    if finished is not None:
      finished.close()
  return params, opts, wc0, lacc, gn


def json_target(path):
  """What the local --json path `path` is -> (descriptor, stat): one of this process's file descriptors to write the
  record through, or None; and os.stat(path), through every link, or None where nothing is there (or it was not
  asked, the path naming a descriptor).  Raises the OSError of a stat that fails otherwise (write_json returns it)."""
  if m := re.fullmatch(
      rf"/dev/std(in|out|err)|/(?:dev|proc/(?:self|thread-self|{os.getpid()}))/fd/(\d+)", os.path.abspath(path)
  ):
    # By its name, never resolved: with stdout `> run.log`, /dev/stdout -> /proc/self/fd/1 -> run.log, a regular file,
    # and a temp file renamed onto it replaced the log (the descriptor writing on into the unlinked file, and the next
    # record going to a new file named what /proc/self/fd/1 then gives, 'run.log (deleted)').
    return int(m[2]) if m[2] else {"in": 0, "out": 1, "err": 2}[m[1]], None
  try:
    st = os.stat(path)  # through every link: a symlinked record, say
  except FileNotFoundError:
    return None, None
  # Stdout or stderr by a name the pattern misses -- a link to /dev/stdout, a directory link to /dev, the log's own
  # path -- names the log its descriptor is open on: replaced, the log was lost as with /dev/stdout resolved.
  for k in 1, 2:
    try:
      if os.path.samestat(os.fstat(k), st):
        return k, st
    except OSError:  # k closed
      pass
  return None, st


def write_json(path, record, provisional=False):
  """`record` as JSON at `path`, local or gs:// (--json's run record) -> None, or the error that kept it from being
  written.  A local file is written whole or not at all: under a temporary name beside it, then renamed over it
  (os.replace, atomic within a filesystem), so a run cut short while writing it, or a write that fails (a full disk),
  leaves the record written before -- main's of the steps before a profile, say -- as it was.  Where `path` is a
  symlink the file it names is the one replaced (the link stays), and an existing record keeps its permission bits.

  Two kinds of local target are not replaced.  One of this process's descriptors -- by the name /dev/stdout,
  /dev/stderr, /dev/fd/N, /proc/self/fd/N, /proc/thread-self/fd/N or /proc/<pid>/fd/N, and stdout and stderr by any
  other name os.stat finds to be the file descriptor 1 or 2 is open on (a link to /dev/stdout, the log's own path) --
  is written through the descriptor, whatever it is open on, after what the process wrote there before, with a
  newline after the record: with stdout `> run.log` or `>> run.log`, --json /dev/stdout (a link to it, run.log)
  leaves the log every line and each record in its place.  (A descriptor other than 1 and 2 by another name, a link
  to /dev/fd/3 say, is taken for the file it is open on: a regular file there is replaced.)  A target that is no
  regular file (a named pipe, a device) is written in place, and gets no `provisional` record (main's before the
  profile, which the one after it replaces): it keeps nothing across a crash, and each write opens it anew, so a pipe
  whose reader read the first record to its end and went left the second write's open waiting for another reader
  without end."""
  data = json.dumps(record, indent=1, default=str)
  if path.startswith("gs://"):  # a Pathways head pod's disk goes with its pod; a GCS object is replaced whole
    try:
      xprof_align.write_bytes(path, data.encode())
    # google.api_core's errors are no OSError
    except Exception as e:  # pylint: disable=broad-exception-caught
      return e
    return None
  try:
    fd, st = json_target(path)
    if fd is not None:
      for s in sys.stdout, sys.stderr:  # what the process printed so far comes before the record
        if s is not None:  # None: the process started with that descriptor closed (`>&-`), nothing printed to flush
          s.flush()
      with os.fdopen(os.dup(fd), "w", encoding="utf-8") as f:
        f.write(data + "\n")
      return None
    if st is not None and not stat.S_ISREG(st.st_mode):  # a named pipe, a device: nothing to replace
      if provisional:
        return None
      with open(path, "w", encoding="utf-8") as f:
        f.write(data)
      return None
    path = os.path.realpath(path)  # a symlink's file is replaced beside itself, the link kept
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}.tmp")
    try:
      with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
      if st is not None:  # an existing record keeps its permission bits (the temp file has the umask's)
        os.chmod(tmp, stat.S_IMODE(st.st_mode))
      os.replace(tmp, path)
    finally:
      try:
        os.remove(tmp)  # still there only if the write or the rename failed
      except OSError:
        pass
  except OSError as e:
    return e
  return None


def main(argv=None):
  # Under Pathways (JAX_PLATFORMS=proxy): pathwaysutils, before anything starts a backend (else nothing).
  proxy = pathways.initialize()
  warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
  a = make_parser().parse_args(argv)
  if a.data not in ("synthetic", "maxtext") and not a.data.startswith("npy:"):
    raise SystemExit(f"--data must be synthetic, maxtext or npy:PATH, not {a.data!r}")
  if a.max_ahead is not None and a.max_ahead < 0:
    raise SystemExit(f"--max-ahead must be >= 0 (leave it out for no bound), not {a.max_ahead}")
  # Checked now, not where they are used, after the compile and warm-up (--bench divides by --steps there).
  if a.bench and a.steps < 1:
    raise SystemExit(f"--steps must be >= 1 with --bench (the steps it times), not {a.steps}")
  if a.profile and a.profile_steps < 1:
    raise SystemExit(f"--profile-steps must be >= 1 with --profile (the steps it traces), not {a.profile_steps}")
  if a.profile_hosts is not None and not proxy:
    raise SystemExit(
        "--profile-hosts is Pathways' (JAX_PLATFORMS=proxy) count of traced worker hosts; a multi-controller run "
        "picks its traced processes with --profile-procs"
    )
  # Under Pathways what only multi-controller JAX takes: refused before the backend starts (and before the write
  # probe and the Lineage checks below), not by Pipeline or init_distributed after them.
  if proxy and (bad := multi_controller_only(a)):
    raise SystemExit("\n".join(bad))
  if proxy and a.max_ahead is None:  # the client's run-ahead bounded there (PATHWAYS_MAX_AHEAD); the record says so
    a.max_ahead = PATHWAYS_MAX_AHEAD
  # A gs:// path that names no bucket stops the run now, in every mode (a dry run too), before
  # jax.distributed.initialize: every process parses the same flags, so all stop together (across processes the
  # probe's answer is only a warning, and their traces would go into a local directory 'gs:').
  for flag, path in (("--profile", a.profile), ("--json", a.json)):
    try:
      pathways.check_bucket(path, flag)
    except ValueError as e:
      raise SystemExit(str(e)) from None
  profile_hosts = None
  if a.profile and proxy:  # a trace only goes to gs:// there: said now, not after the compile and the steps
    try:
      pathways.check_trace_dir(a.profile)
    except ValueError as e:
      raise SystemExit(str(e)) from None
    env_hosts = os.environ.get("MPMD_PROFILE_HOSTS", "1")
    try:
      profile_hosts = a.profile_hosts or profiling.positive_int(env_hosts)
    except (ValueError, argparse.ArgumentTypeError) as e:
      raise SystemExit(f"MPMD_PROFILE_HOSTS={env_hosts!r}: a number of hosts, at least 1 ({e})") from None
  unwritable = None  # under --distributed, why this client cannot write a gs:// --profile: acted on once initialized
  if a.profile and str(a.profile).startswith("gs://") and not a.dry_run:
    # Tried now, in any mode: a bucket nobody can write would hang a trace written straight into it when it stops
    # (one process, Pathways or not; under Pathways the proxy with it).  Under --distributed the answer waits for
    # jax.distributed.initialize, since a process stopping before it would keep the others waiting: then one process
    # (a one-host slice) stops as well, while across processes each traced one writes a local directory first and an
    # export error only costs the profile (a warning).  Not on a dry run, which traces nothing.
    try:
      pathways.probe_trace_dir(a.profile)
    except ValueError as e:
      if not a.distributed:
        raise SystemExit(str(e)) from None
      unwritable = str(e)
  if (why := lineage_missing(a)) is not None:
    # the pipeline would build and only its first program's trace fail: say so before anything starts (a dry run
    # traces nothing, so it only warns)
    if not a.dry_run:
      raise SystemExit(f"--batch-split: {why}")
    print(f"warning: --batch-split: {why}", flush=True)
  if a.distributed:  # multi-controller JAX: jax.distributed first (a one-host slice runs one process)
    init_distributed()
    if unwritable is not None:
      if not pathways.multi_controller():  # one process: its trace goes straight into gs://, as without --distributed
        raise SystemExit(unwritable)
      print(
          f"warning: {unwritable} -- not across {jax.process_count()} processes: a traced one writes its trace into "
          "a local directory first, and keeps it there when the export fails",
          flush=True,
      )
  # this process's id: jax's under multi-controller JAX (whoever started jax.distributed), 0 in one controller
  pid = jax.process_index() if pathways.multi_controller() else 0
  if (why := lineage_off_tpu(a)) is not None:
    # the same: Lineage's kernels have no CPU / GPU path (asked only now: asking initializes the backend, which
    # jax.distributed.initialize refuses to follow)
    if not a.dry_run:
      raise SystemExit(f"--batch-split: {why}")
    if pid == 0:
      print(f"warning: --batch-split: {why}", flush=True)

  p = build(a)
  # Pathways or not, from here on the Pipeline's answer (its devices' client: on_pathways), which JAX_PLATFORMS's
  # first platform (pathways.is_proxy) only predicted before the backend started.
  if on_pathways(p.meshes) != proxy:
    p.close()
    raise SystemExit(
        f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r} said {'' if proxy else 'not '}Pathways, but the "
        f"pipeline's devices are {'not ' if proxy else ''}the IFRT proxy's "
        f"({getattr(p.meshes[0].devices.flat[0].client, 'runtime_type', '?')}): give JAX_PLATFORMS=proxy for "
        "Pathways, a platform of this host's own otherwise"
    )
  resolved_config(a, p)
  popts = None
  if a.profile:
    # Under Pathways no process is a stage's, so each traced worker host traces every chip unless told otherwise,
    # under a session name known here (the line naming where the profile went gives it).
    chips = profiling.EVERY_CHIP if proxy and a.profile_chips is None else a.profile_chips
    popts = profiling.profile_options(chips, platform=jax.devices()[0].platform)
    if proxy:
      popts.session_id = pathways.session_id()
  last = p.chunks[-1].stage
  # In one controller everything prints as it always has: this process runs every stage (and the last), writes the
  # record and traces.  Across processes (multi-controller JAX) process 0 prints the run's header, each stage's first
  # process its stage's numbers, tagged with the stage, and the last chunk's stage the loss; the processes of
  # --json-procs write records, those of --profile-procs trace (flags every process parses alike, checked before any
  # compile or step).
  if p.multi:
    traced_procs = proc_set(a.profile_procs, p, "--profile-procs") if a.profile else None
    json_procs = proc_set(a.json_procs, p, "--json-procs") | {0}
    lead = {s: min(p.stage_procs[s]) for s in range(p.S)}
    mine = sorted(p.local)
    is_lead = any(lead[s] == pid for s in mine)
    runs_last = last in p.local
    tag = f'[s{",".join(map(str, mine))} p{pid}] ' if mine else f"[idle p{pid}] "
  else:
    traced_procs = json_procs = None
    lead, mine, is_lead, runs_last, tag = dict.fromkeys(range(p.S), pid), list(range(p.S)), True, True, ""

  def log(msg, when=True):
    if when:
      print(tag + msg, flush=True)

  for flag in lineage_ignored(a):
    log(
        f"warning: {flag}: not used by --batch-split on Lineage's layers, which run their own expert kernels, "
        "dispatch, EP collectives and splash attention (ds_ep_lineage.IGNORED_STAGE_KNOBS / IGNORED_MODEL_FIELDS)",
        pid == 0,
    )
  for line in lineage_unused(a) + lineage_partial(a):
    log(f"warning: {line}", pid == 0)
  pv = provenance(p.scfg, tiles=("custom" if (a.wi_tiling or a.wo_tiling) else a.tiles))
  log(provenance_line(pv), pid == 0)
  log(p.summary(), pid == 0)
  if p.multi:
    # every flag goes into the check: the step counts decide how many copies
    # each process walks, and every stage's processes must train the same data.
    # But --json, the file a process writes its record to, decides nothing of
    # the plan, and a launcher may name one in each host's own directory.
    h = check_plan_agreement(p, extra=sorted((k, v) for k, v in vars(a).items() if k != "json"))
    idle = sorted(set(range(jax.process_count())) - set().union(*p.stage_procs))
    log(
        f"multi-controller: {jax.process_count()} processes; this one runs stage {mine} with "
        f"{len(jax.local_devices())} local devices; stage leaders {lead}; plan {h[:12]} agreed",
        is_lead,
    )
    log(
        f'{len(idle)} process{"es hold" if len(idle) > 1 else " holds"} no stage ({p.S} stages x '
        f"{a.fsdp * a.ep} devices of {jax.device_count()}): only taking part in the boundary copies",
        pid == 0 and bool(idle),
    )
    log("this process holds no stage: it only takes part in the boundary copies", not mine)
    if p.relay_ctx is not None:  # its settings, and the host's that its throughput depends on
      log(p.relay_ctx.describe(), is_lead)
    if p.transport == "native" and native_transport_aborts(jax.devices()[0].device_kind):
      log(
          "warning: --transport native across v6e / v5p hosts: jax's cross-host receive aborts there (jax 0.11.1, "
          "libtpu 0.0.46): --transport xfer, ppermute or, across slices, relay",
          pid == 0,
      )
  # Stages over several slices: across processes (jax's own copies between them: one host never spans slices), or in
  # the one process of a Pathways run, which holds them all (each copy one IFRT CopyArrays: the note).
  if p.multi and (msg := cross_slice_warning(p.meshes, p.transport)):
    log(f"warning: {msg}", pid == 0)
  if msg := cross_slice_note(p.meshes, p.transport):
    log(msg, pid == 0)
  if span := slice_spanning_stages(p.meshes):
    log(
        f"warning: stages {span} span TPU slices, so their FSDP and EP collectives cross DCN: --layout "
        f"{a.layout} does not place by slice (one stage per slice is --layout slices)",
        pid == 0,
    )
  if proxy and (msg := pathways.client_flags_note()):
    log(msg)
  # Each stage's collectives, and libtpu's SparseCore answer against the device rule where it was asked: across
  # processes, and in the one process of a Pathways run, whose server compiles otherwise than McJAX's libtpu (it
  # offloads where McJAX keeps a stage synchronous, so its plan can differ).
  if p.multi or proxy:
    for s in mine:
      c = p.stage_scfg[s]
      ep = (
          f"Lineage's own ({'on SparseCore' if p.lineage_sc[s] else 'as TensorCore code'})"
          if p.scfg.lineage_split
          else c.ep_collectives
      )  # (on Lineage's layers ds_ep's flavour runs nowhere: the summary's line)
      log(f"stage {s}: EP collectives {ep}, FSDP collectives {c.fsdp_collectives}", lead[s] == pid)
      if (s, ds_ep.FE) in p.offloads and _is_tpu(p.meshes[s].devices.flat[0]) and p.meshes[s].devices.size > 1:
        log(offload_check(p.meshes, s, *p.offloads[(s, ds_ep.FE)]), lead[s] == pid)  # the plan's answer
  if a.dry_run:
    p.close()
    rec = {"config": vars(a), "summary": p.summary(), "provenance": pv}
    if p.split_chunks:
      rec["zb"] = p.zb_record()
    return rec
  # Before the precompile, which would stop lowering the first backward; the processes share flags and device
  # kind, so all of them stop here together.
  if msg := float32_ragged_sort_problem(a, p.scfg):
    p.close()
    raise SystemExit(msg)
  msg = missing_input(a, p)  # before the precompile: a mistyped path fails in seconds
  if p.multi and (a.init_from or a.data.startswith("npy:")):  # flags alike on all processes: all reach its barrier
    msg = missing_input_anywhere(msg)
  if msg:
    p.close()
    raise SystemExit(msg)
  if not a.no_precompile:
    # across processes every stage's programs must be the same on all its processes: checked before any launches
    n, dt = p.precompile(agree=functools.partial(check_program_agreement, p) if p.multi else None)
    log(f"precompiled {n} programs in {dt:.1f}s" + (", each the same on all its processes" if p.multi else ""), is_lead)
    for s in dict.fromkeys(s for _, s, _, _ in p.program_prints):  # a boundary copy's: under its first local stage
      log(
          "\n".join(program_prints_lines(p, s)),
          not p.multi or lead[s if isinstance(s, int) else next(x for x in s if x in p.local)] == pid,
      )
  if p.multi:
    t0 = time.perf_counter()
    host_barrier("mpmd_precompiled")
    log(f"every process compiled: waited {time.perf_counter() - t0:.1f}s at the barrier", is_lead)
  t0 = time.perf_counter()
  if a.init_from:
    if p.scfg.layer_tree == "per_layer" and not ds_ep.is_qwen(p.scfg.model):
      raise SystemExit(
          f"--init-from {a.init_from}: chunk files hold a DeepSeek chunk's layers stacked; run with "
          "--layer-tree stacked"
      )
    try:
      params, opts, wc0 = p.init_state(p.params_from_chunks(p.chunk_trees_from(a.init_from)))
    except ValueError as e:
      raise SystemExit(f"--init-from {a.init_from}: {e}") from None
  else:
    params, opts, wc0 = p.init(jax.random.key(0))
  if a.data == "synthetic":
    batch = p.synthetic_batch()
  else:
    impl = MAXTEXT_BATCH_PRNG if a.data == "maxtext" else None
    toks = p.maxtext_tokens(impl) if impl else load_tokens(a.data[len("npy:") :])
    try:
      batch = p.token_batch(toks)
    except ValueError as e:
      raise SystemExit(f"--data {a.data}: {e}") from None
    digest = hashlib.sha256(np.ascontiguousarray(np.asarray(toks, np.int32)[:, :-1]).tobytes()).hexdigest()[:16]
    log(f"data: {a.data}{f' (under {impl})' if impl else ''}, inputs sha256 {digest}", is_lead)
  lrs = p.lr_arrays(a.lr)
  jax.block_until_ready((params, opts, batch))
  log(f"init {time.perf_counter() - t0:.1f}s" + (f" (weights from {a.init_from})" if a.init_from else ""), is_lead)
  if p.multi:
    host_barrier("mpmd_initialized")
  if a.mem_report:
    log(
        p.mem_report(
            {"params (storage)": params, "optimizer (mu, nu)": opts, "pre-gathered weights (wc0)": wc0, "batch": batch}
        ),
        is_lead,
    )

  losses, gnorms = [], []
  if a.grad_digest:
    seen = itertools.count()

    def on_grads(grads):
      k = next(seen)
      if k < a.grad_digest:
        log("\n".join(grad_digest_lines(p, grads, k)), bool(mine))

    p.on_grads = on_grads
  t0 = time.perf_counter()
  for i in range(a.warmup):
    params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
    read = None
    if proxy:  # one client runs every stage (gn and lacc are its own)
      # one round trip for both (each costs a few ms through the IFRT proxy): the same values as the reads below
      read = [float(x) for x in jax.device_get((gn, jnp.sum(lacc)))]
    gnorms.append(None if gn is None else read[0] if read else float(gn))  # the global norm: every stage has it
    # With bf16 weights MaxText reports its norms as bf16 scalars: compare those with this one rounded.
    gnorm = "" if gn is None else f"gnorm {gnorms[-1]:.6f} (bf16 {float(jnp.bfloat16(gnorms[-1])):.6g})"
    if runs_last:  # the sum is a program on the last stage: all its processes run it
      l = read[1] if read else float(jnp.sum(lacc))
      losses.append(l)
      log(f"warmup step {i}: loss {l:.7f} {gnorm}  ({time.perf_counter() - t0:.1f}s)", lead[last] == pid)
    else:
      log(f"warmup step {i}: {gnorm}  ({time.perf_counter() - t0:.1f}s)", is_lead)
  p.on_grads = None  # --grad-digest digests warmup steps only: none of the timed or profiled ones
  if a.mem_report:
    log(f"peak HBM after warmup {peak_hbm_line(peak_hbm())}", is_lead)
  out = {
      "config": vars(a),
      "summary": p.summary(),
      "provenance": pv,
      "warmup_losses": losses,
      "warmup_gnorms": gnorms,
      "warmup_gnorms_bf16": [None if g is None else float(jnp.bfloat16(g)) for g in gnorms],
  }
  if p.split_chunks:  # the split backward: its chunks, W buffers and simulated gain (Pipeline.zb_record)
    out["zb"] = p.zb_record()
  if getattr(p, "program_prints", None):  # this process's programs, to compare across runs
    out["programs"] = [(n, _stage_key(s), lw["sha"], x) for n, s, lw, x in p.program_prints]
  if p.multi:
    out.update(process=pid, stages=mine, processes=jax.process_count())

    def relay_window(key, what):
      """Where this process's relay time went (transfer.RELAY_MEASURES) since the last window -> out[key], then
      forgotten: one window per kind of step -- warm-up, run-ahead (--bench's steady state), blocked, profiled.
      Taken once every process is past the window's steps: each process's receives of them are in when its steps
      return (a consumer resolves its receive before it is dispatched), so then so is every send of them, but for
      its last write's bookkeeping, and none of the next window's has started.  Multi-controller only (the relay
      needs --distributed): the callers ask `p.multi`."""
      if p.relay_ctx is None:
        return
      from jax._src import distributed

      distributed.global_state.client.wait_at_barrier(f"mpmd_relay_{key}", 30 * 60 * 1000)
      out[key] = p.relay_ctx.timing(reset=True)
      log(f"relay timing, {what}: {transfer.relay_timing_line(out[key])}", bool(mine))

  def write_record(provisional=False):
    """--json: this process's record, `out` as it stands -> None, or the error that kept it from being written (main
    raises it after its last barrier: the other processes need not wait it out).  Under multi-controller JAX the
    processes of --json-procs write theirs, each but process 0's beside --json (run.json -> run.p3.json).
    `provisional`: the record before the profile, which a local target written in place (a named pipe, a device:
    write_json) does not get."""
    if not a.json:
      return None
    if not p.multi:
      return write_json(a.json, out, provisional)
    if pid not in json_procs:
      return None
    root, ext = os.path.splitext(a.json)
    return write_json(a.json if pid == 0 else f'{root}.p{pid}{ext or ".json"}', out, provisional)

  if p.multi:
    relay_window("relay_warmup", "warmup steps")
  ahead = "run-ahead" if a.max_ahead is None else f"at most {a.max_ahead} step(s) ahead"
  if a.bench:
    # steady state: keep the queues full, block once at the end
    jax.block_until_ready(params)
    t0 = time.perf_counter()
    state = [params, opts, wc0]
    del params, opts, wc0  # run_steps holds the only references
    params, opts, wc0, lacc, gn = run_steps(p, state, a.steps, batch, lrs, a.max_ahead)
    jax.block_until_ready((params, opts, lacc))
    dt = (time.perf_counter() - t0) / a.steps
    tf = maxtext_tflops_per_device(p)
    loss = f"  loss {float(jnp.sum(lacc)):.5f}" if runs_last else ""
    log(
        f"step {dt * 1e3:.2f} ms  (mean of {a.steps}, {ahead}){loss}  "
        f"TFLOP/s/device {tf / dt:.1f} (MaxText accounting, {tf:.2f} TFLOP/step/device)  "
        f"tokens/s/device {p.gbs * p.seq / dt / (p.S * a.fsdp * a.ep):.0f}",
        is_lead,
    )
    out.update(step_ms=dt * 1e3, tflops_per_device=tf / dt, tflop_per_step_per_device=tf)
    if p.multi:
      relay_window("relay", "the run-ahead steps")
    # per-step latency with a block after each
    lat = []
    for _ in range(max(3, a.steps)):
      t1 = time.perf_counter()
      params, opts, wc0, lacc, gn = p.step(params, opts, wc0, batch, lrs)
      jax.block_until_ready((params, opts, lacc))
      lat.append(time.perf_counter() - t1)
    # Blocking after every step is also a way to RUN: with stash sets holding
    # HBM, run-ahead lets the next step's first programs contend for memory
    # with this step's tail, and a blocked loop can be the faster one.
    med = statistics.median(lat)
    log(
        f"blocked step latency {min(lat) * 1e3:.2f} ms (min of {len(lat)}); "
        f"mean {sum(lat) / len(lat) * 1e3:.2f} ms, median {med * 1e3:.2f}, max {max(lat) * 1e3:.2f} "
        f"(a loop that blocks every step)",
        is_lead,
    )
    out["blocked_step_mean_ms"] = sum(lat) / len(lat) * 1e3
    out["blocked_step_ms"] = min(lat) * 1e3
    out["blocked_step_median_ms"] = med * 1e3
    out["blocked_steps_ms"] = [x * 1e3 for x in lat]
    if p.multi:
      relay_window("relay_blocked", "the blocked steps")
    per = peak_hbm()
    log(f"peak HBM {peak_hbm_line(per)}", is_lead)
    out["peak_hbm_gib"] = max(per.values()) if per else (None if proxy else 0.0)  # None: not reported (Pathways)
    out["peak_hbm_gib_per_device"] = per
  if a.profile:
    # The record so far (warm-up, --bench) goes out before the profile as well: a run cut short while profiling keeps
    # it (a GKE job preempted during its profiled steps lost its numbers so).  The write after the profile adds what
    # the profile reports.
    if (failed := write_record(provisional=True)) is not None:
      why = pathways.storage_error(failed)
      log(f"warning: --json {a.json}: not written before the profile ({why}); it is tried again after the profile")
    jax.block_until_ready(params)
    # Every process runs the profiled steps (they share every boundary copy);
    # only --profile-procs trace them, --profile-chips chips each (popts:
    # one TPU, one a slice under --layout slices), named in stage order
    # (profiling.trace_label).  Across processes the traced hosts
    # write ONE XProf session, each rebased to an origin process 0 picks
    # (xprof_align): every stage's TPUs on one timeline in the trace viewer.
    traced = not p.multi or pid in traced_procs
    if traced and not profile_hlo_proto(a.profile_hlo_proto, p):
      popts.enable_hlo_proto = False
      log(f"--profile-hlo-proto {a.profile_hlo_proto}: the trace leaves out the programs' HLO protos", True)
    if p.multi:
      session, origin = shared_profile_session()
      local = xprof_align.local_trace_dir() if traced else None
    target = local if p.multi else a.profile
    if traced and proxy:
      # Pathways: the client's trace and `profile_hosts` worker hosts' (their device planes), one session in gs://,
      # with a settle after the start (pathways.trace)
      tracer = pathways.trace(target, popts, max_num_hosts=profile_hosts)
    else:
      tracer = jax.profiler.trace(target, profiler_options=popts) if traced else contextlib.nullcontext()
    trace_error = None
    try:
      with tracer:
        state = [params, opts, wc0]
        del params, opts, wc0
        params, opts, wc0, lacc, gn = run_steps(p, state, a.profile_steps, batch, lrs, a.max_ahead)
        jax.block_until_ready((params, opts, lacc))
    # A Pathways trace that did not start (its steps then did not run) or did not stop: the profile is a by-product,
    # its loss must not lose the run record (the steps' own errors are no TraceError, and go on up).
    except pathways.TraceError as e:
      trace_error = str(e)
    if p.multi:
      relay_window("relay_profile", "the profiled steps")
    where = a.profile
    if proxy and trace_error is None:
      where = f'{a.profile.rstrip("/")}/plugins/profile/{popts.session_id}'
      out["profile_session"] = where
      chips = popts.advanced_configuration.get("tpu_num_chips_to_profile_per_task", "all")
      where += f" ({profile_hosts} Pathways worker host(s), {chips} chip(s) each)"
    elif proxy:
      out["profile_error"] = trace_error
      where = f"no profile: --profile {a.profile}: {trace_error}"
    if p.multi and traced:
      # The profile is a by-product: an export error (say, a bucket this host
      # cannot write) must not lose the run record or keep the others waiting.
      try:
        where = xprof_align.export_into_session(
            local,
            a.profile,
            session,
            origin,
            host=profiling.trace_label(mine[0] if mine else None, pid, p.S, jax.process_count(), socket.gethostname()),
        )
        shutil.rmtree(local, ignore_errors=True)
      except Exception as e:
        where = f"{local} only: exporting into {a.profile} failed ({pathways.storage_error(e)})"
    log(f"warning: {where}" if trace_error else f"profile written under {where}", traced)
  err = write_record()
  if p.multi:
    # Nobody leaves (and takes the coordination service with it, process 0)
    # before everyone is done.
    from jax._src import distributed

    distributed.global_state.client.wait_at_barrier("mpmd_main_done", 30 * 60 * 1000)
  # every process is past its last step, every relay receive consumed: its streams can end
  p.close()
  if err is not None:
    raise err
  return out


if __name__ == "__main__":
  main()
