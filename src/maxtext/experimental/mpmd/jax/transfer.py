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

"""Cross-mesh device-to-device transfer layer for the pure-JAX MPMD pipeline.

The driver moves one activation (or cotangent) pytree per stage boundary per
microbatch.  `jax.device_put` works, but costs 58-72 us of host Python per
call on a v5p host (against Channel's 13; v5p-8, jax 0.11.1, 2026-10-07), and
host dispatch is the binding cost of the whole design.  A `Channel`
resolves one boundary ONCE at setup (device lists, shardings, copy semantics)
and then issues the entire boundary as a single
`xc.batched_copy_array_to_devices_with_sharding` call.

Rules baked in here, all of them learned the hard way:
  * `ArrayCopySemantics.DONATE_INPUT` on a cross-mesh copy SEGFAULTS.  Only
    REUSE_INPUT is ever used; free the source by dropping the Python reference
    AFTER the copy is issued (PJRT keeps it alive until the copy completes).
  * A put whose per-device index map differs between src and dst silently
    round-trips through the host (`x._value`) and caches a numpy copy forever.
    It is still numerically correct, so no value check catches it -- hence
    `same_index_map` / `assert_d2d` at SETUP and `no_host_bounce()` in CI.
  * Index maps must be compared as ORDERED VALUES; the maps are keyed by Device
    and stage sub-meshes are disjoint, so dict equality is always False.
  * The batched copy does NOT check that the arrays it is handed carry the
    sharding they were resolved for.  Hand it shards laid out differently and it
    reinterprets them under the destination layout and returns a structurally
    inconsistent Array: no exception, no host bounce, WRONG NUMBERS downstream.
    That is why `send` validates every fast-path leaf: ~0.16 us/leaf measured,
    ~1% of the ~12 us/leaf send.  Slow-but-correct has three guards, so
    fast-and-wrong gets one too.

Across processes (one per host) a boundary is a CrossProcessChannel (jax's
cross-host copy), a PermuteChannel (a collective-permute program both stages
run), an XferChannel (jax.experimental.transfer pulls) or a RelayChannel: the
one deliberate host round trip, D2H -> TCP between the hosts -> H2D, for
where libtpu's own cross-host paths fail (across MegaScale slices).
"""

from __future__ import annotations
import bisect
import collections
import contextlib
import functools
import hmac
import math
import os
import queue
import secrets
import socket
import struct
import sys
import threading
import time
import warnings
import zlib

import numpy as np
import jax
from jax.sharding import Sharding
from jaxlib import xla_client as xc

from maxtext.experimental.mpmd.jax import pathways

try:  # private, version-guarded
  from jax._src import array as _jax_array
except Exception:  # pragma: no cover - jax internals
  _jax_array = None

# ------------------------------------------------------------ version guards
_BATCHED_COPY = getattr(xc, "batched_copy_array_to_devices_with_sharding", None)
BATCHED_COPY_AVAILABLE = _BATCHED_COPY is not None
REUSE_INPUT = xc.ArrayCopySemantics.REUSE_INPUT  # never DONATE_INPUT: segfaults


class ShardingMismatchError(ValueError):
  """A cross-mesh put would silently take the host-round-trip reshard path."""


class HostBounceError(RuntimeError):
  """A device->host transfer happened inside a `no_host_bounce()` region."""


# ------------------------------------------------------------- the predicate
def _index_values(sharding, shape):
  return tuple(sharding.addressable_devices_indices_map(tuple(shape)).values())


def _private_fastpath_verdict(src_sharding, dst_sharding, ndim):
  """jax's own fastpath predicate, or None if this build does not have it.

  jax 0.11.1's `_sharding_indices_and_eq(src, dst, ndim)` returns a plain bool
  (HLO shardings equal AND addressable device counts equal) despite the name.
  """
  fn = getattr(_jax_array, "_sharding_indices_and_eq", None)
  if fn is None:
    return None
  try:
    res = fn(src_sharding, dst_sharding, ndim)
  except Exception:  # NotImplementedError, signature drift
    return None
  return res if isinstance(res, bool) else None


@functools.lru_cache(maxsize=4096)
def _same_index_map_cached(src_sharding, dst_sharding, shape):
  src, dst = _index_values(src_sharding, shape), _index_values(dst_sharding, shape)
  mine = len(src) == len(dst) and src == dst
  theirs = _private_fastpath_verdict(src_sharding, dst_sharding, len(shape))
  if theirs is not None and theirs != mine:
    warnings.warn(
        f"same_index_map disagrees with jax._src.array._sharding_indices_and_eq "
        f"for {src_sharding} -> {dst_sharding} shape={shape}: ours={mine} "
        f"jax={theirs}; taking the conservative AND.",
        RuntimeWarning,
    )
    return mine and theirs
  return mine


def same_index_map(src_sharding, dst_sharding, shape) -> bool:
  """True iff a put src->dst is a pure device-to-device copy (no reshard).

  Compares the ORDERED VALUES of `addressable_devices_indices_map(shape)` --
  never the dicts, which are keyed by Device and therefore never equal across
  two disjoint stage sub-meshes -- plus equal addressable device counts.
  """
  return _same_index_map_cached(src_sharding, dst_sharding, tuple(shape))


def _try_same_index_map(src_sharding, dst_sharding, shape) -> bool:
  """`same_index_map`, but a sharding that cannot produce an index map is False.

  Used where an exception must not escape (classification, hot-path
  validation): if D2D cannot be PROVEN, it must be assumed not to hold.
  """
  if src_sharding is None or dst_sharding is None:
    return False
  try:
    return same_index_map(src_sharding, dst_sharding, shape)
  except Exception:  # unusual sharding, no index map
    return False


def _fmt_index(idx):
  def one(s):
    if isinstance(s, slice):
      a = "" if s.start is None else s.start
      b = "" if s.stop is None else s.stop
      return f"{a}:{b}"
    return str(s)

  if not isinstance(idx, tuple):
    idx = (idx,)
  return "(" + ", ".join(one(s) for s in idx) + ")"


def assert_d2d(src, dst_sharding, name: str = "transfer", max_report: int = 4):
  """Raise unless `jax.device_put(src, dst_sharding)` is a true D2D copy.

  `src` is a jax.Array (or anything with `.sharding` and `.shape`, e.g. a
  ShapeDtypeStruct with a sharding).  Call this at SETUP for every boundary,
  never in the hot loop.
  """
  src_sharding = getattr(src, "sharding", None)
  if src_sharding is None:
    raise ShardingMismatchError(
        f"{name}: source has no .sharding ({type(src).__name__}); a host array " "cannot be device-to-device copied."
    )
  shape = tuple(src.shape)
  if same_index_map(src_sharding, dst_sharding, shape):
    return True
  sv, dv = _index_values(src_sharding, shape), _index_values(dst_sharding, shape)
  if len(sv) != len(dv):
    detail = f"addressable device counts differ: src={len(sv)} dst={len(dv)}"
  else:
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(sv, dv)) if a != b]
    detail = "mismatching per-device shard indices [i: src -> dst]: " + ", ".join(
        f"{i}: {_fmt_index(a)} -> {_fmt_index(b)}" for i, a, b in bad[:max_report]
    )
    if len(bad) > max_report:
      detail += f", ... ({len(bad)} of {len(sv)} devices differ)"
  raise ShardingMismatchError(
      f"{name}: cross-mesh put for shape={shape} would RESHARD, not copy, and "
      f"silently round-trip through the host (x._value, ~21 ms for 16 MB, and "
      f"the numpy copy is cached in x._npy_value forever).\n"
      f'  src spec: {getattr(src_sharding, "spec", src_sharding)}  '
      f"({src_sharding})\n"
      f'  dst spec: {getattr(dst_sharding, "spec", dst_sharding)}  '
      f"({dst_sharding})\n"
      f"  {detail}\n"
      f"  Fix: give both sub-meshes the same axis names/sizes and use the same "
      f"PartitionSpec on both sides of the boundary."
  )


# ------------------------------------------------------------------- Channel
def _is_extended(dtype) -> bool:
  try:
    return jax.dtypes.issubdtype(dtype, jax.dtypes.extended)
  except Exception:
    return False


def _axis_type_class(mesh):
  """What a mesh's axis types mean for the arrays on it, for comparing two.

  A mesh whose axes all have one type -- all Auto, all Explicit -- types an
  array that way whatever its rank or shape, so that one type is the class.
  A mixed mesh is compared axis by axis, by name.
  """
  types = set(mesh.axis_types)
  if len(types) == 1:
    return types.pop()
  return frozenset(zip(mesh.axis_names, mesh.axis_types))


def _device_list(sharding):
  dl = getattr(sharding, "_internal_device_list", None)
  return None if dl is None else dl.addressable_device_list


class Channel:
  """A pre-resolved, reusable transfer channel for ONE stage boundary.

  Built once at setup from `(avals, src_shardings, dst_shardings)` -- pytrees of
  the same structure, or a single Sharding to broadcast over all leaves.  All
  per-leaf work (fast/slow classification, device lists, copy semantics) is done
  in `__init__`; `send` is a flat list build plus one batched copy call.

      ch = Channel(y_avals, src_shardings, dst_shardings, name='s0->s1')
      assert ch.n_slow == 0
      y_next = ch.send(y)          # one host call for the whole boundary

  `strict=True` rejects a resharding boundary at BUILD time, whatever else is
  also wrong with the leaf.  `validate=True` (the default) rejects, at SEND
  time, an array that does not carry the sharding the channel was resolved for.

  `check_axis_types=True` (the default) rejects, at BUILD time, a leaf whose
  src and dst meshes type it differently.  The copy itself does not care
  (index maps and HLO shardings ignore axis types), but its output carries
  the DST sharding, and so the dst's types.  An Explicit activation would
  arrive Auto-typed and trace its consumer untyped, and nothing would say so.
  Meshes of different rank are compared by mode, not axis by axis: two
  all-Auto meshes type nothing whatever their shapes (`_axis_type_class`).
  Pass False only if re-typing is the intent.
  """

  __slots__ = (
      "name",
      "treedef",
      "avals",
      "src",
      "dst",
      "n_fast",
      "n_slow",
      "n_leaves",
      "degraded",
      "reasons",
      "reshards",
      "validate",
      "_fast_pos",
      "_slow_pos",
      "_dev_lists",
      "_shardings",
      "_copy_semantics",
      "_checks",
      "_fallbacks",
      "last_fallback_error",
  )

  def __init__(
      self, avals, src_shardings, dst_shardings, name="channel", strict=False, validate=True, check_axis_types=True
  ):
    flat_avals, treedef = jax.tree.flatten(avals)
    self.name, self.treedef, self.avals = name, treedef, tuple(flat_avals)
    self.n_leaves = len(flat_avals)
    self.validate = bool(validate)
    self.src = self._broadcast(src_shardings, "src_shardings")
    self.dst = self._broadcast(dst_shardings, "dst_shardings")
    if check_axis_types:
      self._check_axis_types()

    self._fast_pos, self._slow_pos, self.reasons, self.reshards = [], [], [], []
    self._dev_lists, self._shardings = [], []
    for i, aval in enumerate(flat_avals):
      why, reshards = self._classify(aval, self.src[i], self.dst[i])
      self.reasons.append(why)
      self.reshards.append(reshards)
      # Independent of `why`: a resharding boundary is rejected under strict=
      # even when the leaf is ALSO slow for an earlier reason (extended dtype,
      # missing batched symbol), which used to mask it.
      if strict and reshards:
        assert_d2d(_ShapedSrc(aval, self.src[i]), self.dst[i], name=f"{name}[leaf {i}]")
      if why is None:
        self._fast_pos.append(i)
        self._dev_lists.append(_device_list(self.dst[i]))
        self._shardings.append(self.dst[i])
      else:
        self._slow_pos.append(i)
    self._copy_semantics = [REUSE_INPUT] * len(self._fast_pos)
    # Pre-bound (position, declared sharding, declared shape) for the hot-path
    # validation, so it is three compares and no attribute walks per leaf.
    self._checks = [] if not self.validate else [(i, self.src[i], tuple(flat_avals[i].shape)) for i in self._fast_pos]
    self.n_fast, self.n_slow = len(self._fast_pos), len(self._slow_pos)
    self.degraded = self.n_fast == 0 and self.n_leaves > 0
    self._fallbacks, self.last_fallback_error = 0, None

  def _check_axis_types(self):
    """Raise if any leaf's src and dst meshes would type it differently."""
    for i, (src, dst) in enumerate(zip(self.src, self.dst)):
      sm, dm = getattr(src, "mesh", None), getattr(dst, "mesh", None)
      if getattr(sm, "axis_types", None) is None or getattr(dm, "axis_types", None) is None:
        continue
      if _axis_type_class(sm) != _axis_type_class(dm):
        raise ShardingMismatchError(
            f"{self.name}[leaf {i}]: src mesh axis types "
            f"{dict(zip(sm.axis_names, sm.axis_types))} != dst "
            f"{dict(zip(dm.axis_names, dm.axis_types))}.  The copy "
            "would silently re-type the array to the dst's types (an Explicit "
            "array arriving Auto-typed traces its consumer untyped).  Build "
            "both sub-meshes with the same shard_mode, or pass "
            "check_axis_types=False if re-typing is the intent."
        )

  def _broadcast(self, shardings, what):
    if isinstance(shardings, Sharding):
      return [shardings] * self.n_leaves
    flat = self.treedef.flatten_up_to(shardings)
    if len(flat) != self.n_leaves:
      raise ValueError(f"{self.name}: {what} has {len(flat)} leaves, avals have " f"{self.n_leaves}")
    return list(flat)

  def _classify(self, aval, src_sharding, dst_sharding):
    """`(reason, reshards)`; reason is None iff the leaf rides the batched path.

    EVERY condition is evaluated and all of them are reported, joined by '; '.
    In particular the index-map check runs even when an earlier condition has
    already forced the slow path: `strict=` must reject a resharding boundary
    whether or not the leaf is also an extended dtype or the jaxlib is missing
    the batched symbol.  Returning only the first reason silently accepted
    those boundaries, and they really do round-trip through the host.
    """
    reasons = []
    if not BATCHED_COPY_AVAILABLE:
      reasons.append("no batched_copy_array_to_devices_with_sharding in this " "jaxlib")
    if _is_extended(aval.dtype):
      reasons.append(f"extended dtype {aval.dtype} rejected by the batched call")
    if _device_list(dst_sharding) is None:
      reasons.append(f"dst sharding {dst_sharding} has no addressable device " "list")
    reshards = not _try_same_index_map(src_sharding, dst_sharding, aval.shape)
    if reshards:
      reasons.append("reshard: src and dst index maps differ (host round trip)")
    return ("; ".join(reasons) if reasons else None), reshards

  def check(self, tree=None):
    """Setup-time assertion that every leaf is a real D2D copy.

    With no argument, checks the declared src shardings; with a pytree, checks
    the shardings the arrays actually carry.  `send` validates the fast-path
    leaves on every call anyway (see `_reject_leaf`); this stays as the
    setup-time, whole-boundary form and is the only check the SLOW leaves get.
    """
    leaves = self.avals if tree is None else self.treedef.flatten_up_to(tree)
    srcs = self.src if tree is None else [getattr(a, "sharding", None) for a in leaves]
    for i, (s, leaf) in enumerate(zip(srcs, leaves)):
      if s is None:
        raise ShardingMismatchError(
            f"{self.name}[leaf {i}]: {type(leaf).__name__} has no .sharding; a "
            "host array cannot be device-to-device copied."
        )
      assert_d2d(_ShapedSrc(_Aval(leaf.shape, self.avals[i].dtype), s), self.dst[i], name=f"{self.name}[leaf {i}]")
    return True

  def _reject_leaf(self, i, x):
    """Raise: leaf `i` does not carry the sharding this channel was resolved for.

    Called only off the hot path, after the cheap identity/equality compare in
    `send_flat` has already failed.  Re-checks the real predicate first, at the
    array's OWN shape: a different sharding object with the same per-device
    index map, or a differently shaped array that still shards identically on
    both sides (a short last microbatch), is a perfectly good source for the
    batched copy and is accepted rather than rejected.
    """
    s = getattr(x, "sharding", None)
    shape = tuple(getattr(x, "shape", ()))
    if _try_same_index_map(s, self.dst[i], shape):
      return  # equivalent layout, still D2D
    aval = self.avals[i]
    if s is None:
      detail = f"leaf is a {type(x).__name__} with no .sharding -- a host " "array cannot be device-to-device copied"
    else:
      sv, dv = None, None
      try:
        sv, dv = _index_values(s, shape), _index_values(self.dst[i], shape)
      except Exception:  # pragma: no cover - exotic sharding
        pass
      if sv is None or dv is None:
        detail = "the actual sharding cannot produce an index map"
      elif len(sv) != len(dv):
        detail = f"addressable device counts differ: actual={len(sv)} " f"dst={len(dv)}"
      else:
        bad = [(j, a, b) for j, (a, b) in enumerate(zip(sv, dv)) if a != b]
        detail = "per-device shard indices [i: actual -> dst]: " + ", ".join(
            f"{j}: {_fmt_index(a)} -> {_fmt_index(b)}" for j, a, b in bad[:4]
        )
        if len(bad) > 4:
          detail += f", ... ({len(bad)} of {len(sv)} devices differ)"
    raise ShardingMismatchError(
        f"{self.name}[leaf {i}]: send() was handed an array that does NOT carry "
        f"the sharding this channel was built for, and the batched copy does "
        f"not check.  It would reinterpret the source shards under the "
        f"destination layout and return a structurally inconsistent Array that "
        f"produces WRONG NUMBERS downstream -- no exception, no warning, no "
        f"host bounce.\n"
        f'  declared src: {getattr(self.src[i], "spec", self.src[i])}  '
        f"({self.src[i]})\n"
        f'  actual   src: {getattr(s, "spec", s)}  ({s})\n'
        f'  dst         : {getattr(self.dst[i], "spec", self.dst[i])}  '
        f"({self.dst[i]})\n"
        f"  declared shape {tuple(aval.shape)}, actual shape {shape}\n"
        f"  {detail}\n"
        f"  Fix: produce the array with the declared sharding (jit "
        f"out_shardings=...), or rebuild the Channel from the shardings you "
        f"actually have.  Channel(..., validate=False) skips this check "
        f"(~0.16 us/leaf, ~1% of a send) and restores the silent corruption."
    )

  def send_flat(self, leaves):
    """Hot path: list of arrays in tree order -> list of arrays on the dst mesh."""
    out = list(leaves)
    if self._fast_pos:
      for i, want_sharding, want_shape in self._checks:
        # ~0.16 us/leaf: a tuple compare plus an identity (then equality)
        # compare.  Anything that fails it goes to the slow diagnostic, which
        # re-checks the real predicate before deciding to raise.
        x = out[i]
        try:
          if x.shape != want_shape or ((s := x.sharding) is not want_sharding and s != want_sharding):
            self._reject_leaf(i, x)
        except AttributeError:  # not a jax.Array at all
          self._reject_leaf(i, x)
      srcs = [out[i] for i in self._fast_pos]
      try:
        copies = _BATCHED_COPY(srcs, self._dev_lists, self._shardings, self._copy_semantics)
        # strict=: a short return would silently leave trailing leaves holding
        # their SOURCE array, and feeding a mesh-A array into a mesh-B jit works
        # exactly once before raising INVALID_ARGUMENT forever after.
        for i, c in zip(self._fast_pos, copies, strict=True):
          out[i] = c
      except Exception as e:  # never lose a step to an API drift
        self._fallbacks += 1
        self.last_fallback_error = e
        if self._fallbacks == 1:
          warnings.warn(
              f"{self.name}: batched copy failed ({type(e).__name__}: " f"{e}); falling back to jax.device_put per leaf.",
              RuntimeWarning,
          )
        for i in self._fast_pos:
          out[i] = jax.device_put(leaves[i], self.dst[i])
    for i in self._slow_pos:
      out[i] = jax.device_put(leaves[i], self.dst[i])
    return out

  def send(self, tree):
    """Move one pytree across this boundary.  Returns the same tree structure."""
    return jax.tree.unflatten(self.treedef, self.send_flat(self.treedef.flatten_up_to(tree)))

  def __repr__(self):
    return (
        f"<Channel {self.name} leaves={self.n_leaves} fast={self.n_fast} "
        f"slow={self.n_slow}"
        f'{" reshard=" + str(sum(self.reshards)) if any(self.reshards) else ""}'
        f'{"" if self.validate else " UNVALIDATED"}'
        f'{" DEGRADED" if self.degraded else ""}>'
    )


# ------------------------------------------- multi-controller (one process per host)
def process_indices(sharding) -> frozenset:
  """The processes that own `sharding`'s devices."""
  return frozenset(d.process_index for d in sharding.device_set)


def is_cross_process(src_sharding, dst_sharding) -> bool:
  """Under multi-controller JAX: do the two shardings live on different
  processes?  (Always False in one process.)"""
  return jax.process_count() > 1 and process_indices(src_sharding) != process_indices(dst_sharding)


def _global_index_values(sharding, shape):
  """Per-device shard indices over ALL devices, in device-assignment order --
  `addressable_devices_indices_map` sees only this process's devices, which
  for a cross-process boundary is one side or neither."""
  return tuple(sharding.devices_indices_map(tuple(shape)).values())


class CrossProcessChannel:
  """One stage boundary between stages that live on different processes.

  jax's cross-host copy (`batched_copy_array_to_devices_with_sharding` with
  the shardings on different processes, jax/_src/dispatch.py) takes its
  transfer keys from a counter that EVERY process advances on EVERY such call,
  whether or not it sends or receives (pjrt_client.cc `CreateNewTransferKey`).
  So every process issues every cross-process send, in the same order: one
  that skips a send pairs every later transfer with the wrong one, silently
  (verified on CPU, jax 0.11.1: a bystander that skips one copy makes its next
  copy deliver the previous copy's array).  `send_flat` takes the source
  arrays where this process owns them and None elsewhere (a placeholder with
  no local shards stands in); the destination stage's processes get its
  array back, a process owning neither side an array with no local shards.

  Checked at build time, as jax does before taking that path
  (`_is_supported_cross_host_transfer`): the same device kind and HLO sharding
  on both sides, and -- the copy pairs shard k of the source with shard k of
  the destination -- the same global per-device index map, with every pair on
  two different processes (jax copies across processes only when every shard
  crosses; a pair within one process would fail on that process alone, at
  send time).  A boundary that would reshard is rejected: jax refuses
  cross-process reshards.
  """

  __slots__ = (
      "name",
      "treedef",
      "avals",
      "src",
      "dst",
      "n_leaves",
      "validate",
      "owns_src",
      "owns_dst",
      "_dev_lists",
      "_copy_semantics",
      "_placeholders",
  )

  def __init__(self, avals, src_shardings, dst_shardings, name="xchannel", validate=True):
    if not BATCHED_COPY_AVAILABLE:
      raise RuntimeError(f"{name}: no batched_copy_array_to_devices_with_sharding in this jaxlib")
    flat_avals, treedef = jax.tree.flatten(avals)
    self.name, self.treedef, self.avals = name, treedef, tuple(flat_avals)
    self.n_leaves = len(flat_avals)
    self.validate = bool(validate)
    bc = lambda s: [s] * self.n_leaves if isinstance(s, Sharding) else list(treedef.flatten_up_to(s))
    self.src, self.dst = bc(src_shardings), bc(dst_shardings)
    pid = jax.process_index()
    for i, (aval, s, d) in enumerate(zip(flat_avals, self.src, self.dst)):
      where = f"{name}[leaf {i}]"
      if not is_cross_process(s, d):
        raise ValueError(f"{where}: source and destination are on the same processes; use Channel")
      if _is_extended(aval.dtype):
        raise ShardingMismatchError(f"{where}: extended dtype {aval.dtype} cannot be copied across processes")
      kinds = {x.device_kind for x in s.device_set} | {x.device_kind for x in d.device_set}
      if len(kinds) != 1:
        raise ShardingMismatchError(f"{where}: device kinds differ across the boundary: {kinds}")
      nd = len(aval.shape)
      if s._to_xla_hlo_sharding(nd) != d._to_xla_hlo_sharding(nd):
        raise ShardingMismatchError(f"{where}: HLO shardings differ ({s} vs {d}): the copy would reshard")
      sv, dv = _global_index_values(s, aval.shape), _global_index_values(d, aval.shape)
      if len(sv) != len(dv):
        raise ShardingMismatchError(
            f"{where}: device counts differ: {len(sv)} source devices, " f"{len(dv)} destination devices"
        )
      if sv != dv:
        bad = [(j, a, b) for j, (a, b) in enumerate(zip(sv, dv)) if a != b]
        raise ShardingMismatchError(
            f"{where}: per-device shard indices differ [i: src -> dst]: "
            + ", ".join(f"{j}: {_fmt_index(a)} -> {_fmt_index(b)}" for j, a, b in bad[:4])
        )
      same = [
          j
          for j, (x, y) in enumerate(zip(s._device_assignment, d._device_assignment))
          if x.process_index == y.process_index
      ]
      if same:
        raise ShardingMismatchError(
            f"{where}: shards {same[:4]} would stay on one process (process "
            f"{s._device_assignment[same[0]].process_index}); jax copies across processes only when "
            "every shard crosses"
        )
    self.owns_src = [any(x.process_index == pid for x in s.device_set) for s in self.src]
    self.owns_dst = [any(x.process_index == pid for x in d.device_set) for d in self.dst]
    self._dev_lists = [d._internal_device_list for d in self.dst]
    self._copy_semantics = [REUSE_INPUT] * self.n_leaves
    self._placeholders = [None] * self.n_leaves

  def placeholder(self, i):
    """The source of leaf i as a process that owns none of it passes it: the
    global shape and sharding, no local shards."""
    if self._placeholders[i] is None:
      a = self.avals[i]
      self._placeholders[i] = jax.make_array_from_single_device_arrays(tuple(a.shape), self.src[i], [], dtype=a.dtype)
    return self._placeholders[i]

  def send_flat(self, leaves):
    """Every process, every call: leaves (arrays, or None where this process
    owns no shard of the source) -> arrays on the destination stage."""
    srcs = []
    for i, x in enumerate(leaves):
      if x is None:
        if self.owns_src[i]:
          raise ValueError(
              f"{self.name}[leaf {i}]: this process owns shards of the source, so it must " "pass the array, not None"
          )
        x = self.placeholder(i)
      elif self.validate and (
          tuple(x.shape) != tuple(self.avals[i].shape) or x.dtype != self.avals[i].dtype or x.sharding != self.src[i]
      ):
        # the receiver allocates the declared aval: a mismatch would fail there, not here
        raise ShardingMismatchError(
            f"{self.name}[leaf {i}]: send() was handed a {x.dtype}{list(x.shape)} array sharded "
            f"{x.sharding}, not the declared {self.avals[i].dtype}{list(self.avals[i].shape)} on "
            f"{self.src[i]}"
        )
      srcs.append(x)
    out = _BATCHED_COPY(srcs, self._dev_lists, self.dst, self._copy_semantics)
    return list(out)

  def send(self, tree):
    return jax.tree.unflatten(self.treedef, self.send_flat(self.treedef.flatten_up_to(tree)))

  def __repr__(self):
    return (
        f"<CrossProcessChannel {self.name} leaves={self.n_leaves} "
        f"src procs={sorted(process_indices(self.src[0]))[:4]}.. "
        f"dst procs={sorted(process_indices(self.dst[0]))[:4]}..>"
    )


# (device, shard shape, dtype) -> the zero buffer PermuteChannel's destination
# side passes as its half of the program's input (`PermuteChannel._dummy`)
_PERMUTE_DUMMIES = {}


class PermuteChannel:
  """One stage boundary as a collective-permute program over the union of
  the two stages' devices: shard i of the source stage goes to device i of
  the destination, over ICI inside one slice; between two slices (one stage
  per slice: ds_ep_pipeline --layout slices) XLA makes it a send / recv over
  DCN, the one cross-slice device copy known to work on TPU (at every size
  tried, 2026-10; RelayChannel crosses slices through the hosts instead).

  The alternative to jax's cross-host copy where that does not work: on TPU
  v6e and v5p (jax 0.11.1, libtpu 0.0.46) its native receive aborts the process
  (std::bad_alloc; in EnqueueCrossHostReceive on v6e), and on v6e the DCN
  socket path hangs.
  The program runs on both stages' devices, so the two meet at it: the
  source stage cannot run ahead of a destination still busy with earlier
  work (an async copy could).

  The program's input is ONE array over the union mesh, replicated along its
  leading axis: the source stage's devices hold the real shards (zero-copy
  views of the source array), the destination's a dummy buffer each; the
  body permutes along that axis, and the destination's output shards ARE its
  stage array (again zero-copy).  Under multi-controller JAX
  (pathways.multi_controller) only the processes of the two stages call
  `send_flat`, in the order every process walks, each with the shards it
  holds; anyone else gets None back at once.  In one controller -- a Pathways
  client, one host's run -- both sides are this process's, and no process is
  looked at.  Same-shape meshes and identical specs on both sides (checked).
  """

  AXIS = "_xfer"
  __slots__ = (
      "name",
      "treedef",
      "avals",
      "src",
      "dst",
      "n_leaves",
      "union",
      "u_sh",
      "prog",
      "multi",
      "owns_src",
      "owns_dst",
      "_dummies",
      "_dst_devices",
  )

  def __init__(self, avals, src_shardings, dst_shardings, name="pchannel"):
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh, NamedSharding

    flat_avals, treedef = jax.tree.flatten(avals)
    self.name, self.treedef, self.avals = name, treedef, tuple(flat_avals)
    self.n_leaves = len(flat_avals)
    bc = lambda s: [s] * self.n_leaves if isinstance(s, Sharding) else list(treedef.flatten_up_to(s))
    self.src, self.dst = bc(src_shardings), bc(dst_shardings)
    ma, mb = self.src[0].mesh, self.dst[0].mesh
    if ma.axis_names != mb.axis_names or ma.devices.shape != mb.devices.shape:
      raise ShardingMismatchError(f"{name}: the stage meshes differ ({dict(ma.shape)} vs {dict(mb.shape)})")
    if set(ma.device_ids.flat) & set(mb.device_ids.flat):
      raise ValueError(f"{name}: the two stages share devices")
    for i, (s, d) in enumerate(zip(self.src, self.dst)):
      if s.mesh != ma or d.mesh != mb or s.spec != d.spec:
        raise ShardingMismatchError(
            f"{name}[leaf {i}]: want one mesh per side and the same spec on both " f"({s} vs {d})"
        )
    self.union = Mesh(np.stack([ma.devices, mb.devices]), (self.AXIS,) + tuple(ma.axis_names))
    self.u_sh = [NamedSharding(self.union, s.spec) for s in self.src]
    specs = tuple(s.spec for s in self.src)
    ax = self.AXIS

    def body(*xs):
      return tuple(lax.ppermute(x, ax, [(0, 1)]) for x in xs)

    sm = jax.shard_map(body, mesh=self.union, in_specs=specs, out_specs=specs, check_vma=False)

    def xfer(*xs):
      return sm(*xs)

    xfer.__name__ = xfer.__qualname__ = "xfer_" + name.replace("->", "_to_").replace(" ", "_")
    self.prog = jax.jit(xfer, in_shardings=tuple(self.u_sh), out_shardings=tuple(self.u_sh))
    self.multi = pathways.multi_controller()
    if self.multi:  # which sides' devices this process holds
      pid = jax.process_index()
      self.owns_src = any(x.process_index == pid for x in ma.devices.flat)
      self.owns_dst = any(x.process_index == pid for x in mb.devices.flat)
    else:  # one controller: both
      self.owns_src = self.owns_dst = True
    self._dst_devices = frozenset(mb.devices.flat)
    self._dummies = None

  def lower_args(self):
    """Abstract arguments of `prog`, for ahead-of-time compiles."""
    return tuple(jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=u) for a, u in zip(self.avals, self.u_sh))

  def _dummy(self):
    # The program never writes or donates its input, so one zero buffer per
    # (device, shard shape, dtype) serves every channel into a stage.
    import numpy as np

    if self._dummies is None:
      devs = list(self.union.devices[1].flat)
      if self.multi:  # the destination's devices this process holds
        pid = jax.process_index()
        devs = [d for d in devs if d.process_index == pid]

      def zeros(shape, dtype, d):
        key = (d, shape, np.dtype(dtype))
        if key not in _PERMUTE_DUMMIES:
          _PERMUTE_DUMMIES[key] = jax.device_put(np.zeros(shape, dtype), d)
        return _PERMUTE_DUMMIES[key]

      self._dummies = [
          [zeros(tuple(s.shard_shape(tuple(a.shape))), a.dtype, d) for d in devs] for a, s in zip(self.avals, self.dst)
      ]
    return self._dummies

  def send_flat(self, leaves):
    """The two stages' processes, every boundary copy in order: leaves on the
    source side (None on the destination's) -> the destination stage's arrays
    there, None elsewhere (multi-controller JAX).  In one controller: the
    leaves -> the destination stage's arrays."""
    if self.multi and not (self.owns_src or self.owns_dst):
      return [None] * self.n_leaves
    ins = []
    for i, a in enumerate(self.avals):
      shards = []
      if not self.multi or self.owns_src:
        x = leaves[i]
        if x is None or tuple(x.shape) != tuple(a.shape) or x.dtype != a.dtype or x.sharding != self.src[i]:
          raise ShardingMismatchError(
              f"{self.name}[leaf {i}]: the source side must pass the declared "
              f"{a.dtype}{list(a.shape)} array on {self.src[i]}, got {x!r}"[:400]
          )
        shards += [s.data for s in x.addressable_shards]
      if not self.multi or self.owns_dst:
        shards += self._dummy()[i]
      ins.append(jax.make_array_from_single_device_arrays(tuple(a.shape), self.u_sh[i], shards))
    outs = self.prog(*ins)
    if self.multi and not self.owns_dst:
      return [None] * self.n_leaves
    return [
        jax.make_array_from_single_device_arrays(
            tuple(a.shape), d, [s.data for s in o.addressable_shards if s.device in self._dst_devices]
        )
        for a, d, o in zip(self.avals, self.dst, outs)
    ]

  def send(self, tree):
    return jax.tree.unflatten(self.treedef, self.send_flat(self.treedef.flatten_up_to(tree)))

  def __repr__(self):
    return f"<PermuteChannel {self.name} leaves={self.n_leaves} over {dict(self.union.shape)}>"


def _host_ip():
  """This host's address peers can reach (the pod IP on GKE hostNetwork)."""
  import socket

  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
    return s.getsockname()[0]
  except OSError:
    return socket.gethostbyname(socket.gethostname())
  finally:
    s.close()


class XferContext:
  """This process's `jax.experimental.transfer` server, its peers' addresses
  (published once through the coordination service's key-value store) and
  its connections to them, shared by every XferChannel of the process.

  One per process: its address is published under a key of the process, so
  get it with `xfer_context()` rather than constructing another."""

  def __init__(self, n_transport=None, parallel_copies=None, transfer_size=None, timeout_s=1800):
    if _XFER_CONTEXT is not None:
      raise RuntimeError(
          "this process already has an XferContext (its address is published once): " "use transfer.xfer_context()"
      )
    # MPMD_XFER_TRANSPORTS / _COPIES / _CHUNK_MB tune the sockets (defaults measured on v6e-256:
    # 8.95 GB/s per host for a 1 GiB boundary)
    n_transport = n_transport or int(os.environ.get("MPMD_XFER_TRANSPORTS", "16"))
    parallel_copies = parallel_copies or int(os.environ.get("MPMD_XFER_COPIES", "32"))
    transfer_size = transfer_size or int(float(os.environ.get("MPMD_XFER_CHUNK_MB", "16")) * 2**20)
    from jax.experimental import transfer as jxt
    from jax._src import distributed

    self.jxt, self.timeout_ms = jxt, timeout_s * 1000
    self.client = distributed.global_state.client
    ip = os.environ.get("MPMD_XFER_IP") or _host_ip()
    backend = jax.local_devices()[0].client
    # never an empty transport_addresses list: the sender CHECK-fails on one (jax 0.11.1)
    self.srv = jxt.start_transfer_server(
        backend,
        f"{ip}:0",
        [f"{ip}:0"] * n_transport,
        max_num_parallel_copies=parallel_copies,
        transfer_size=transfer_size,
    )
    self.pid = jax.process_index()
    self.client.key_value_set(f"mpmd/xfer_addr/{self.pid}", self.srv.address())
    self._conns = {}

  def conn(self, peer):
    if peer not in self._conns:
      addr = self.client.blocking_key_value_get(f"mpmd/xfer_addr/{peer}", self.timeout_ms)
      self._conns[peer] = self.srv.connect(addr)
    return self._conns[peer]


_XFER_CONTEXT = None


def xfer_context(**kw):
  """This process's XferContext, started on first use (`kw` go to that first
  construction; later calls return the same context)."""
  global _XFER_CONTEXT
  if _XFER_CONTEXT is None:
    ctx = XferContext(**kw)
    _XFER_CONTEXT = ctx
  return _XFER_CONTEXT


# --------------------------- boundaries between hosts, shard k to device k (Xfer, Relay)
def _same_mesh_boundary(name, avals, src, dst, as_bytes=False):
  """XferChannel's and RelayChannel's checks at build time: shard k of the
  source stage goes to device k of the destination, so identical stage
  meshes (axis names and shape), one mesh a side, the same spec on both
  sides, and the same per-device index map -> the two meshes.  `as_bytes`
  (the relay, which sends a shard's bytes) also refuses extended dtypes."""
  ma, mb = src[0].mesh, dst[0].mesh
  if ma.axis_names != mb.axis_names or ma.devices.shape != mb.devices.shape:
    raise ShardingMismatchError(f"{name}: the stage meshes differ ({dict(ma.shape)} vs {dict(mb.shape)})")
  for i, (aval, s, d) in enumerate(zip(avals, src, dst)):
    if s.mesh != ma or d.mesh != mb or s.spec != d.spec:
      raise ShardingMismatchError(f"{name}[leaf {i}]: want one mesh per side and the same spec on both")
    if as_bytes and _is_extended(aval.dtype):
      raise ShardingMismatchError(f"{name}[leaf {i}]: extended dtype {aval.dtype} cannot travel as bytes")
    if _global_index_values(s, aval.shape) != _global_index_values(d, aval.shape):
      raise ShardingMismatchError(f"{name}[leaf {i}]: per-device shard indices differ between the stages")
  return ma, mb


def _check_sources(name, leaves, avals, src):
  """XferChannel's and RelayChannel's source side, at each send: every leaf
  the declared array -- its shape, its dtype, and its declared sharding or
  one equivalent to it (the same devices holding the same slices, as
  P('d') is to P('d', None)), since its shards are taken by device."""
  for i, (x, a, s) in enumerate(zip(leaves, avals, src)):
    if (
        x is None
        or tuple(x.shape) != tuple(a.shape)
        or x.dtype != a.dtype
        or (x.sharding != s and not _equivalent(x.sharding, s, len(a.shape)))
    ):
      raise ShardingMismatchError(
          f"{name}[leaf {i}]: the source side must pass the declared " f"{a.dtype}{list(a.shape)} array on {s}"
      )


def _equivalent(a, b, ndim):
  try:
    return a.is_equivalent_to(b, ndim)
  except Exception:  # a sharding without the comparison: not proven equivalent
    return False


def _host_routes(src_mesh, dst_mesh, pid, what):
  """Shard k of the source stage goes from the process holding it to the
  process holding device k of the destination -> for process `pid`, (the
  destination's devices in mesh order, {destination process: the positions
  `pid` sends there}, {source process: the positions `pid` fills from
  there}).  Raises ValueError for stages sharing devices, or a process
  holding devices of both (`what` moves data between hosts only)."""
  fa, fb = list(src_mesh.devices.flat), list(dst_mesh.devices.flat)
  if set(fa) & set(fb):
    raise ValueError("the two stages share devices")
  sends, recvs = {}, {}
  for k, (x, y) in enumerate(zip(fa, fb)):
    if x.process_index == pid:
      sends.setdefault(y.process_index, []).append(k)
    if y.process_index == pid:
      recvs.setdefault(x.process_index, []).append(k)
  if sends and recvs:
    raise ValueError(
        f"process {pid} holds devices of both stages; {what} moves data between hosts "
        "only (use Channel within a process)"
    )
  return fb, sends, recvs


def xfer_peers(src_mesh, dst_mesh, pid):
  """XferChannel's host pairing for process `pid` -> (source mesh positions
  it offers, destination devices it receives on, the one process it pulls
  from or None, the one process it gives to or None).  Raises ValueError
  where the boundary cannot be one host pair per host: a process holding
  both sides, one taking its shards from several source hosts, or one
  giving its shards to several destination hosts.  A function of the meshes
  and `pid` alone, so a check ahead of time can run it for every process."""
  fb, sends, recvs = _host_routes(src_mesh, dst_mesh, pid, "XferChannel")
  if len(recvs) > 1:
    raise ValueError(
        f"process {pid} would receive from processes {sorted(recvs)}; each host must take all "
        "its shards from one source host"
    )
  if len(sends) > 1:
    raise ValueError(
        f"process {pid} would send to processes {sorted(sends)}; each host must give all its "
        "shards to one destination host"
    )
  src_pos = sorted(k for ks in sends.values() for k in ks)
  dst_pos = sorted(k for ks in recvs.values() for k in ks)
  return src_pos, [fb[k] for k in dst_pos], next(iter(recvs), None), next(iter(sends), None)


class XferChannel:
  """One stage boundary over `jax.experimental.transfer` (DCN sockets):
  each source host offers its local shards under a uuid (`await_pull`), and
  the destination host holding the same mesh positions pulls them -- matched
  by the uuid, not by call order, so only the two hosts of each pair take
  part, and the copy is asynchronous on both sides (neither stage waits for
  the other at the boundary, unlike PermuteChannel).

  The uuid is the channel's index and a per-channel sequence number both
  sides advance on every send, in the order every process walks.  Shards
  travel as single-device arrays sorted by mesh position; the receiver
  reassembles its stage array zero-copy.  Requires identical stage meshes
  (axis names, shape, specs, so shard k is the same slice on both sides) and
  every destination host's devices taking their shards from ONE source host
  (`xfer_peers`), checked at build time.

  The source returns as soon as its offer is queued and the peer pulls it
  later, from the same buffers: never donate or delete a sent array (drop the
  reference instead; the transfer server keeps the buffers until pulled).
  """

  __slots__ = (
      "name",
      "idx",
      "treedef",
      "avals",
      "src",
      "dst",
      "n_leaves",
      "ctx",
      "seq",
      "owns_src",
      "owns_dst",
      "peer",
      "_src_pos",
      "_dst_devs",
      "_pull_avals",
  )

  def __init__(self, avals, src_shardings, dst_shardings, ctx, idx, name="xchannel"):
    flat_avals, treedef = jax.tree.flatten(avals)
    self.name, self.idx, self.ctx, self.seq = name, idx, ctx, 0
    self.treedef, self.avals, self.n_leaves = treedef, tuple(flat_avals), len(flat_avals)
    bc = lambda s: [s] * self.n_leaves if isinstance(s, Sharding) else list(treedef.flatten_up_to(s))
    self.src, self.dst = bc(src_shardings), bc(dst_shardings)
    ma, mb = _same_mesh_boundary(name, flat_avals, self.src, self.dst)
    pid = jax.process_index()
    try:
      self._src_pos, self._dst_devs, self.peer, _ = xfer_peers(ma, mb, pid)
    except ValueError as e:
      raise ValueError(f"{name}: {e}") from None
    self.owns_src, self.owns_dst = bool(self._src_pos), bool(self._dst_devs)
    from jax.sharding import SingleDeviceSharding

    self._pull_avals = [
        jax.ShapeDtypeStruct(d.shard_shape(tuple(a.shape)), a.dtype, sharding=SingleDeviceSharding(dev))
        for a, d in zip(self.avals, self.dst)
        for dev in self._dst_devs
    ]

  def send_flat(self, leaves):
    """Every process, every boundary copy in order (the two stages' hosts
    act; the rest only return None): source arrays where owned -> the
    destination stage's arrays there, None elsewhere."""
    if not (self.owns_src or self.owns_dst):
      return [None] * self.n_leaves
    uuid = (self.idx << 40) | self.seq
    if self.owns_src:
      _check_sources(self.name, leaves, self.avals, self.src)
      flat = []
      for i, x in enumerate(leaves):
        by_dev = {s.device: s.data for s in x.addressable_shards}
        flat += [by_dev[dev] for dev in (list(self.src[i].mesh.devices.flat)[k] for k in self._src_pos)]
      self.seq += 1  # only a send that goes out advances the sequence both sides keep
      self.ctx.srv.await_pull(uuid, flat)
      return [None] * self.n_leaves
    self.seq += 1
    got = self.ctx.conn(self.peer).pull(uuid, self._pull_avals)
    nd, out = len(self._dst_devs), []
    for i, (a, d) in enumerate(zip(self.avals, self.dst)):
      out.append(jax.make_array_from_single_device_arrays(tuple(a.shape), d, list(got[i * nd : (i + 1) * nd])))
    return out

  def send(self, tree):
    return jax.tree.unflatten(self.treedef, self.send_flat(self.treedef.flatten_up_to(tree)))

  def __repr__(self):
    return f"<XferChannel {self.name} #{self.idx} leaves={self.n_leaves}>"


# ---------------------------------------------------------------- host relay
# Across MegaScale slices libtpu's cross-host paths fail: jax's copy hangs,
# jax.experimental.transfer's receiver aborts on about the 8th pull, and the
# collective-permute that works is a rendezvous coupling the two stages'
# queues.  The relay leaves the bytes to the hosts: device-to-host copy,
# TCP between the two hosts, host-to-device copy, each end in threads of its
# own.  Every message is a fixed header, then `length` bytes:
#   DATA   bytes [offset, offset + length) of shard (leaf, mesh position) of
#          receive (channel id, sequence number); `total` is the shard's size
#          and `sig` a CRC of its dtype and shape;
#   HELLO  opens a stream: channel id = the sender's process, seq = the
#          stream's index, total = its number of streams, sig = a CRC of the
#          context's key; the receiving context's token follows (random per
#          context, published with its address).  A connection that does not
#          open with these within _RELAY_HELLO_S is refused;
#   ERROR  the sender could not make the send (channel id, seq); the reason
#          follows as text;
#   ABORT  the sender gives up on this process before all its streams are
#          open, so the rest never come: every receive still missing its
#          shards fails.  The reason follows as text.
_RELAY_MAGIC = b"MRL1"
_RELAY_HDR = struct.Struct("<4sB3xIQIIQQQI")  # magic kind | cid seq leaf pos offset length total sig
_DATA, _HELLO, _ERROR, _ABORT = 0, 1, 2, 3
_RELAY_TEXT_MAX = 1 << 16  # an ERROR's or ABORT's reason, at most
_RELAY_HELLO_S = 60.0  # a new connection's HELLO must come within (at most the timeout)
# This process's RelayContexts without a key of their own, in creation order: the n-th publishes
# under mpmd/relay/<n>.  A context takes its number under the lock, once its address is published.
_relay_lock = threading.Lock()
_relay_contexts = 0
# RelayContext.timing's measures, in seconds: a sample per shard sent
# (send_*), per receive completed (recv_wire, recv_dispatch), and per
# receive its consumer resolved (recv_wait).  None of them holds a receive's
# host-to-device copies: each is issued (jax.device_put returns before the
# copy is done, on TPU) and runs on asynchronously, and the consumer's
# program waits for it on the device.  A context counts, sums and takes the
# largest of every sample since its last reset, and keeps the newest
# _RELAY_SAMPLES of each per channel for the percentiles: at 64
# microbatches of 4 shards, the last 64 steps.
RELAY_MEASURES = {
    "send_host": "a shard queued -> on the host: the wait for its producing program (and for the shards "
    "queued to that peer before it), then its device-to-host copy",
    "send_wire": "on the host -> its last chunk written to a socket",
    "recv_wire": "a receive's first chunk in -> its last byte in",
    "recv_dispatch": "its last byte in -> its arrays assembled: its last host-to-device copy issued (each "
    "shard's is at its own last byte), not waited for",
    "recv_wait": "how long its consumer blocked in RelayFuture.result(): 0 when its arrays were assembled, "
    "their host-to-device copies perhaps still running",
}
_RELAY_SAMPLES = 1 << 14
_RELAY_POOL = 4  # receive buffers a context keeps per (channel, leaf, mesh position)
# The relay's three data-moving steps as profiler events, so a trace shows them by name (a trace's
# boundary time can be measured off them): a chunk written to its socket, a chunk's payload read off its
# socket, a completed shard's host-to-device copy issued (the copy itself runs on after the event).
# A TraceAnnotation costs 0.3-0.5 us when no profile is being taken, against about a millisecond a
# 4 MiB chunk spends on the wire.
RELAY_TRACE = {"send": "relay_send", "recv": "relay_recv", "place": "relay_place"}


class RelayError(RuntimeError):
  """A relay transfer failed: a stream broke or carried a malformed message,
  the sending process could not make the send, or a peer closed its streams
  (or gave up opening them) with shards of a receive still unsent."""


class RelayTimeoutError(RelayError, TimeoutError):
  """A relay receive was not complete after MPMD_RELAY_TIMEOUT seconds."""


def relay_routes(src_mesh, dst_mesh, pid):
  """RelayChannel's routes for process `pid` -> ({destination process: the
  mesh positions whose source shards `pid` sends there}, {source process: the
  mesh positions whose destination devices `pid` fills from there}).  Shard k
  of the source stage goes to device k of the destination, from the process
  holding the one to the process holding the other: any number of peers each
  way (unlike `xfer_peers`).  Raises ValueError for stages sharing devices,
  or a process holding devices of both (a copy within one process is a
  Channel's).  A function of the meshes and `pid` alone."""
  _, sends, recvs = _host_routes(src_mesh, dst_mesh, pid, "RelayChannel")
  return sends, recvs


def _relay_sig(dtype, shape):
  return zlib.crc32(f"{np.dtype(dtype).name}{list(shape)}".encode())


def _relay_setting(arg, name, default):
  """A RelayContext setting: `arg` unless None, else environment variable
  `name` unless unset or blank, else `default` -- so that 0 means the same
  given either way (RelayContext refuses what it cannot run with)."""
  if arg is not None:
    return arg
  v = os.environ.get(name, "").strip()
  return v if v else default


def _host_view(x):
  """Whether np.asarray(x), x a jax array, views its device buffer (CPU)
  rather than copying it to host memory (TPU, GPU)."""
  try:
    return all(d.platform == "cpu" for d in x.devices())
  except AttributeError:
    return True


def host_net_settings():
  """This host's settings the relay's throughput depends on, as the kernel
  prints them (None where unreadable): transparent huge pages (a receive
  buffer not taken from the pool is fresh memory) and the TCP buffer limits
  (autotuning grows a socket's buffers up to tcp_[rw]mem's last field; an
  explicit MPMD_RELAY_SOCKBUF_MB is capped at [rw]mem_max)."""

  def read(path):
    try:
      with open(path) as f:
        return " ".join(f.read().split())
    except OSError:
      return None

  return {
      "thp": read("/sys/kernel/mm/transparent_hugepage/enabled"),
      "tcp_rmem": read("/proc/sys/net/ipv4/tcp_rmem"),
      "tcp_wmem": read("/proc/sys/net/ipv4/tcp_wmem"),
      "rmem_max": read("/proc/sys/net/core/rmem_max"),
      "wmem_max": read("/proc/sys/net/core/wmem_max"),
  }


def _relay_socket(sockbuf):
  """A TCP socket with Nagle off, and SO_SNDBUF / SO_RCVBUF of `sockbuf` bytes
  if asked (0: left to the kernel).  An explicit size switches the kernel's
  autotuning off and is capped at net.core.[rw]mem_max unless the process may
  force it: on the TPU VM this was written on that cap is 212992 bytes while
  autotuning grows a receive buffer to tcp_rmem's 300 MB, so the default
  leaves it alone."""
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  if sockbuf:
    force = (32, 33) if sys.platform.startswith("linux") else (None, None)  # SO_{SND,RCV}BUFFORCE
    for opt, f in zip((socket.SO_SNDBUF, socket.SO_RCVBUF), force):
      try:
        if f is None:
          raise OSError
        s.setsockopt(socket.SOL_SOCKET, f, sockbuf)
      except OSError:  # not privileged: the capped request
        s.setsockopt(socket.SOL_SOCKET, opt, sockbuf)
  return s


def _relay_connect(addr, sockbuf, timeout_s):
  """A stream to a peer's listener (retrying a refusal for up to a minute),
  left BLOCKING -- one sendmsg per chunk, the GIL released for all of it --
  with the kernel's SO_SNDTIMEO bounding a peer that stops reading."""
  deadline = time.monotonic() + min(timeout_s, 60.0)
  while True:
    s = _relay_socket(sockbuf)
    s.settimeout(10.0)
    try:
      s.connect(addr)
      break
    except OSError:
      s.close()
      if time.monotonic() > deadline:
        raise
      time.sleep(0.1)
  s.settimeout(None)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.pack("ll", int(timeout_s), int(timeout_s % 1 * 1e6)))
  return s


def _send_all(sock, *bufs):
  """Write every byte of `bufs`, in order (a blocking socket takes a whole
  sendmsg at once unless interrupted)."""
  views = [v for v in (memoryview(b).cast("B") for b in bufs) if v.nbytes]
  while views:
    n = sock.sendmsg(views)
    while n:
      if n >= views[0].nbytes:
        n -= views.pop(0).nbytes
      else:
        views[0], n = views[0][n:], 0


def _recv_exact(sock, buf, eof_ok=False):
  """Fill `buf` from `sock` (MSG_WAITALL: one syscall, and one GIL release,
  per chunk).  -> False if the stream ended before its first byte and
  `eof_ok`; an end anywhere else is a RelayError."""
  view = memoryview(buf).cast("B")
  n, got = view.nbytes, 0
  while got < n:
    k = sock.recv_into(view[got:], n - got, socket.MSG_WAITALL)
    if not k:
      if got == 0 and eof_ok:
        return False
      raise RelayError(f"the stream ended {got} bytes into a {n}-byte read")
    got += k
  return True


def _relay_text(kind, cid, seq, leaf, pos, text):
  """An ERROR or ABORT message: its header and its reason (cut short)."""
  msg = text.encode(errors="replace")[:_RELAY_TEXT_MAX]
  return _RELAY_HDR.pack(_RELAY_MAGIC, kind, cid, seq, leaf, pos, 0, len(msg), 0, 0), memoryview(msg)


def _take_span(spans, lo, hi):
  """Add bytes [lo, hi), lo < hi, to `spans` -- a sorted list of disjoint
  (start, end) runs, merged where they meet -> False, leaving it as it was,
  if they overlap a run already there."""
  i = bisect.bisect_left(spans, (lo, lo))  # the first run starting at lo or later
  if (i and spans[i - 1][1] > lo) or (i < len(spans) and spans[i][0] < hi):
    return False
  if i < len(spans) and spans[i][0] == hi:
    hi = spans.pop(i)[1]
  if i and spans[i - 1][1] == lo:
    i -= 1
    lo = spans.pop(i)[0]
  spans.insert(i, (lo, hi))
  return True


def _refs(lst, i):
  return sys.getrefcount(lst[i])


_ONLY_LISTED = _refs([object()], 0)  # `_refs` of an object nothing but its list holds


class _RelayRecv:
  """One receive of a RelayChannel on this process (its index and sequence
  number): the shards arrived so far, each straight into its buffer and, at
  its last byte, onto its destination device (`placed`); then the
  destination arrays, or the error its RelayFutures raise.  `spans`: per
  shard, the byte ranges taken by chunks in or coming in, as disjoint runs
  (`_take_span`); `got`, how many of those bytes are in.  `claimed`: the
  channel's send_flat has handed out its futures (data may come first).
  t_first / t_last: when its first chunk and its last byte came in;
  `waited`: its consumer's wait is counted (RelayContext.timing)."""

  __slots__ = (
      "ch",
      "seq",
      "bufs",
      "spans",
      "got",
      "done",
      "placed",
      "claimed",
      "event",
      "out",
      "err",
      "t_first",
      "t_last",
      "waited",
  )

  def __init__(self, ch, seq):
    self.ch, self.seq = ch, seq
    self.bufs, self.spans, self.got, self.done, self.placed = {}, {}, {}, set(), {}
    self.claimed = self.waited = False
    self.event, self.out, self.err = threading.Event(), None, None
    self.t_first = self.t_last = None

  def what(self):
    return f"{self.ch.name} (relay channel #{self.ch.idx}) seq {self.seq}"

  def missing(self):
    """{source process: how many of its shards of this receive are not in}."""
    out = {}
    for key, (_, _, q) in self.ch._recv_spec.items():
      if key not in self.done:
        out[q] = out.get(q, 0) + 1
    return out


class RelayFuture:
  """Leaf `i` of a relay receive, which reader threads complete.
  `result()` waits for it -- MPMD_RELAY_TIMEOUT seconds unless told
  otherwise, then RelayTimeoutError naming the channel, the sequence number
  and the processes whose shards are missing -- and returns the destination
  array, or raises what failed the receive.  `resolve` is what a consumer
  calls on whatever a boundary handed it; how long the first leaf resolved
  kept it waiting goes into the context's timing (recv_wait)."""

  __slots__ = ("_r", "_i")

  def __init__(self, r, i):
    self._r, self._i = r, i

  def done(self):
    return self._r.event.is_set()

  def result(self, timeout=None):
    r = self._r
    ctx = r.ch.ctx
    timeout = ctx.timeout_s if timeout is None else timeout
    t0 = time.monotonic()
    if not r.event.wait(timeout):
      with ctx._lock:
        miss, need = r.missing(), len(r.ch._recv_spec)
      if not miss:
        raise RelayTimeoutError(f"{r.what()}: all {need} shards in after {timeout:g} s, not yet all on their " "devices")
      raise RelayTimeoutError(
          f"{r.what()}: {need - sum(miss.values())} of {need} shards in after {timeout:g} s, waiting on "
          + ", ".join(f'process {q} ({n} shard{"s" if n > 1 else ""})' for q, n in sorted(miss.items()))
      )
    if r.err is not None:
      raise r.err.with_traceback(None)  # one error object, raised by every future of it
    if not r.waited:
      r.waited = True
      ctx._note("recv_wait", r.ch.name, time.monotonic() - t0)
    return r.out[self._i]

  def __repr__(self):
    return f'<RelayFuture {self._r.what()} leaf {self._i}{"" if self.done() else " pending"}>'


def resolve(x):
  """`x` itself, or -- a RelayFuture -- the array it stands for, blocking
  until it has arrived."""
  return x.result() if type(x) is RelayFuture else x


class _RelayUnsent:
  """A send this process does not make (`RelayContext._send`: an earlier
  one failed), queued for a peer in place of its shards: the peer gets an
  ERROR for it, in order behind what was queued there before."""

  __slots__ = ("cid", "seq", "why")

  def __init__(self, cid, seq, why):
    self.cid, self.seq, self.why = cid, seq, why


class _RelayPeer:
  """A RelayContext's streams to one destination process and the threads
  feeding them: a stager that takes the queued shards in order, waits for
  each one's device-to-host copy and cuts it into chunks (zero-copy views),
  and one writer per stream sending whichever chunk is next, so one shard
  spreads over every stream.  `pending` counts the shards and chunks not yet
  written, for `flush`.  A failed stream fails them all: the stager and the
  writers then only drain, every stream is shut (the peer sees them end and
  fails whatever it still expects from here), and this process's next send
  raises, once it has told its destinations it is not coming
  (`RelayContext._send`).  If the streams do not all open, the ones that did
  carry an ABORT before they end: the peer counts a sender gone only once
  every stream its HELLO announced has ended."""

  def __init__(self, ctx, pid):
    self.ctx, self.pid = ctx, pid
    self.items = queue.SimpleQueue()
    self.chunks = queue.Queue(maxsize=4 * ctx.streams)  # bounds the host memory staged ahead
    self.cv = threading.Condition()
    self.pending, self.sent, self.failed = 0, 0, None
    self.socks, self.writers = [], []
    self.stager = threading.Thread(target=self._stage, name=f"relay-stage-p{pid}", daemon=True)
    self.stager.start()

  def put(self, item):
    with self.cv:
      self.pending += 1
    self.items.put(item)

  def _done(self, n=1, nbytes=0, rec=None):
    """`n` shards or chunks are off the queue, `nbytes` of them written; a
    written chunk's `rec` is its shard's [chunks unwritten, when it reached
    the host, channel name]: the last one notes send_wire, before a flush
    can see the shard written."""
    with self.cv:
      self.pending -= n
      self.sent += nbytes
      if rec is not None:
        rec[0] -= 1
        if not rec[0]:
          self.ctx._note("send_wire", rec[2], time.monotonic() - rec[1])
      if not self.pending:
        self.cv.notify_all()

  def _fail(self, e):
    err = e if isinstance(e, RelayError) else RelayError(f"relay to process {self.pid}: {type(e).__name__}: {e}")
    with self.cv:
      first = self.failed is None
      if first:
        self.failed = err
      self.cv.notify_all()
    if first:
      self.ctx._sender_failed(err)
      self._shut()

  def abort(self, err):
    """close() at its deadline with this peer's threads still at work (a
    peer that stopped reading, say): shut the streams, so that the writers
    stop at once, the stager only drains, and the peer sees the streams
    end -- unlike `_fail`, without failing this process's sends."""
    with self.cv:
      if self.failed is None:
        self.failed = err
      self.cv.notify_all()
    self._shut()

  def _shut(self):
    for s in list(self.socks):
      try:
        s.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass

  def _connect(self):
    ctx = self.ctx
    addr, token = ctx.kv.blocking_key_value_get(f"{ctx.key}/addr/{self.pid}", int(ctx.timeout_s * 1000)).split()
    host, port = addr.rsplit(":", 1)
    token = bytes.fromhex(token)  # the peer's: each HELLO carries it
    for i in range(ctx.streams):
      s = _relay_connect((host, int(port)), ctx.sockbuf, ctx.timeout_s)
      self.socks.append(s)
      _send_all(
          s, _RELAY_HDR.pack(_RELAY_MAGIC, _HELLO, ctx.pid, i, 0, 0, 0, len(token), ctx.streams, ctx._key_sig), token
      )
    for i, s in enumerate(self.socks):
      t = threading.Thread(target=self._write, args=(s,), name=f"relay-write-p{self.pid}-{i}", daemon=True)
      t.start()
      # listed once started, as _accept does its readers: close() may `join` meanwhile, and a thread not
      # yet started cannot be joined
      with self.cv:
        self.writers.append(t)

  def _stage(self):
    try:
      self._connect()
    except BaseException as e:  # the peer never published, or never accepted
      err = RelayError(f"relay to process {self.pid}: connecting failed: {type(e).__name__}: {e}")
      self.ctx._sender_failed(err)  # this process's next send raises, before the peer hears
      # With fewer streams than its HELLO announced ever ending, the peer would never count this
      # process gone, and would wait out its timeout: an ABORT on each stream that did open.
      for s in self.socks:
        try:
          _send_all(s, *_relay_text(_ABORT, self.ctx.pid, 0, 0, 0, str(err)))
        except OSError:
          pass
      self._fail(err)
      for s in self.socks:  # no writer was started
        s.close()
    while self._cut_next():
      pass
    for _ in self.writers:
      self.chunks.put(None)

  def _cut_next(self):
    """The stager, the next queued shard (-> False at the end): wait for its
    copy to host, then queue its chunks for the writers -- or tell the peer
    it cannot come.  Its own frame holds the shard, so that it can let go."""
    item = self.items.get()
    if item is None:
      return False
    if self.failed is not None:
      self._done()
      return True
    if type(item) is _RelayUnsent:  # a send not made: the peer's receive of it fails at once
      self.chunks.put((*_relay_text(_ERROR, item.cid, item.seq, 0, 0, item.why), None, None))
      return True
    cid, seq, leaf, pos, data, nbytes, sig, name, t_queued = item
    del item
    try:
      host = np.asarray(data)  # waits for the producing program and the copy to host
      flat = memoryview(host.reshape(-1).view(np.uint8))
      if flat.nbytes != nbytes:
        raise RelayError(f"the shard has {flat.nbytes} bytes, the channel declared {nbytes}")
    except BaseException as e:
      # The program that produced it failed, most likely: tell the peer, whose receive fails at
      # once instead of timing out, and fail this process's next send.
      why = f"{type(e).__name__}: {e}"
      self.ctx._sender_failed(RelayError(f"{name} seq {seq}: could not send to process {self.pid}: {why}"))
      self.chunks.put((*_relay_text(_ERROR, cid, seq, leaf, pos, why), None, None))
      return True
    t_host = time.monotonic()
    self.ctx._note("send_host", name, t_host - t_queued)
    # On CPU `host` views the device buffer, so `data` rides along with the chunks; anywhere
    # else `host` is a copy, and letting go of `data` now frees the device buffer as soon as
    # the pipeline drops the array too, not once the last chunk is written.
    keep = data if _host_view(data) else None
    del data
    n = max(1, -(-nbytes // self.ctx.chunk))
    rec = [n, t_host, name]
    with self.cv:
      self.pending += n - 1
    for j in range(n):
      off = j * self.ctx.chunk
      ln = min(self.ctx.chunk, nbytes - off)
      self.chunks.put(
          (
              _RELAY_HDR.pack(_RELAY_MAGIC, _DATA, cid, seq, leaf, pos, off, ln, nbytes, sig),
              flat[off : off + ln],
              keep,
              rec,
          )
      )
    return True

  def _write(self, s):
    while True:
      c = self.chunks.get()
      if c is None:
        break
      hdr, payload, keep, rec = c
      del c
      ok = self.failed is None
      if ok:
        try:
          with jax.profiler.TraceAnnotation(RELAY_TRACE["send"], bytes=payload.nbytes):
            _send_all(s, hdr, payload)
        except BlockingIOError:  # SO_SNDTIMEO: the peer's reader no longer drains
          ok = False
          self._fail(
              RelayError(
                  f"relay to process {self.pid}: no progress in {self.ctx.timeout_s:g} s "
                  "(MPMD_RELAY_TIMEOUT): the peer stopped reading"
              )
          )
        except BaseException as e:
          ok = False
          self._fail(e)
      self._done(1, payload.nbytes if ok else 0, rec if ok else None)
      del payload, keep, rec  # a written chunk holds nothing until the next one
    try:
      s.shutdown(socket.SHUT_WR)  # the peer's reader sees the stream end between messages
    except OSError:
      pass
    s.close()

  def flush(self, timeout):
    """Wait until everything queued is written -> True, or False (timeout, failure)."""
    with self.cv:
      return self.cv.wait_for(lambda: not self.pending or self.failed is not None, timeout) and self.failed is None

  def stop(self):
    self.items.put(None)

  def join(self, deadline):
    """Wait until `deadline` for the stager, then for the writers it started
    (every one of them, once the stager has ended)."""
    self.stager.join(max(0.0, deadline - time.monotonic()))
    with self.cv:
      writers = list(self.writers)
    for t in writers:
      t.join(max(0.0, deadline - time.monotonic()))

  def running(self):
    """The names of this peer's threads still running."""
    with self.cv:
      return [t.name for t in (self.stager, *self.writers) if t.is_alive()]


class RelayContext:
  """This process's end of the host relay its RelayChannels move boundaries
  over.  A TCP listener on the address (MPMD_RELAY_IP, which must be one of
  this host's, or the host's primary IPv4, and MPMD_RELAY_PORT or a free
  port) that the process publishes once through the coordination service's
  key-value store, as XferContext does -- with a random token, which every
  stream's HELLO must carry.  Per peer it sends to, MPMD_RELAY_STREAMS
  connections (default 8), opened at the first send and fed by threads of
  their own (`_RelayPeer`, chunks of MPMD_RELAY_CHUNK_MB, default 4); per
  accepted connection, a reader thread that receives each chunk straight
  into its shard's buffer, puts a shard on its local destination device at
  its last byte, and completes a receive's futures once all its shards are
  there.  Sequence numbers, sizes and dtype/shape CRCs are checked on
  arrival; a malformed or broken stream fails every receive outstanding, a
  peer whose streams all end (or that gave up opening them) fails those
  still missing its shards.  On the sending side a send that fails (a
  broken stream, a shard that never reached the host) makes every later one
  raise, each after an ERROR to its destinations, whose receives then fail
  at once.  MPMD_RELAY_SOCKBUF_MB (default 0, the kernel's autotuning) sizes
  the socket buffers; MPMD_RELAY_TIMEOUT (default 1800 s) bounds a wait for
  a receive, an address, a stuck peer.  `timing()` says where the time went
  (RELAY_MEASURES), `describe()` what the settings are.

  One per Pipeline.  Every process creates its contexts in the same order,
  so the n-th of each publishes and looks up under one key (mpmd/relay/<n>,
  or `key`; one that fails to start takes no number, and shuts its
  listener).  `kv` (default jax.distributed's client) and `pid` (default
  jax.process_index()) are there for tests in one process.  `close()`
  flushes and shuts the sockets and threads down."""

  def __init__(
      self, kv=None, pid=None, key=None, ip=None, port=None, streams=None, chunk_mb=None, timeout_s=None, sockbuf_mb=None
  ):
    env = os.environ.get
    if kv is None:
      from jax._src import distributed

      kv = distributed.global_state.client
      if kv is None:
        raise RuntimeError(
            "RelayContext: jax.distributed is not initialized (peers find this process " "through its key-value store)"
        )
    self.kv = kv
    self.pid = jax.process_index() if pid is None else int(pid)
    streams = int(_relay_setting(streams, "MPMD_RELAY_STREAMS", 8))
    chunk_mb = float(_relay_setting(chunk_mb, "MPMD_RELAY_CHUNK_MB", 4))
    timeout_s = float(_relay_setting(timeout_s, "MPMD_RELAY_TIMEOUT", 1800))
    sockbuf_mb = float(_relay_setting(sockbuf_mb, "MPMD_RELAY_SOCKBUF_MB", 0))
    for bad, what in (
        (streams < 1, f"{streams} streams per peer (MPMD_RELAY_STREAMS): want at least 1"),
        (int(chunk_mb * 2**20) < 1, f"{chunk_mb:g} MiB chunks (MPMD_RELAY_CHUNK_MB): want at least a byte"),
        (not timeout_s > 0, f"a {timeout_s:g} s timeout (MPMD_RELAY_TIMEOUT): want more than 0"),
        (
            not sockbuf_mb >= 0,
            f"{sockbuf_mb:g} MiB socket buffers (MPMD_RELAY_SOCKBUF_MB): want 0 (the " "kernel's autotuning) or more",
        ),
    ):
      if bad:  # before anything is bound or started
        raise ValueError(f"RelayContext: {what}")
    self.streams, self.chunk, self.timeout_s = streams, int(chunk_mb * 2**20), timeout_s
    self.sockbuf = int(sockbuf_mb * 2**20)
    self._lock = threading.Lock()
    self._cv = threading.Condition(self._lock)
    self._channels, self._recvs, self._floor, self._retired = {}, {}, {}, {}
    self._dead = set()  # (channel, seq) failed: whatever of it still comes is drained
    self._pool = {}  # (channel, leaf, mesh position) -> its receive buffers (`_buffer`)
    self._peers, self._streams_from, self._gone, self._gone_why, self._readers = {}, {}, set(), {}, []
    self._fatal = self._send_err = None
    self._closing = self._closed = False
    self.received = 0
    self._tlock = threading.Lock()  # the timing samples' own: never held while taking another
    self._times = {}  # (measure, channel name) -> [samples, their sum, their largest,
    #   the newest _RELAY_SAMPLES of them], in seconds
    self._listener = None
    # A stream's HELLO must carry it: the peers read it with the address, from the key-value store,
    # so a connection from anywhere else -- another job, a port scan -- is refused.
    self._token = secrets.token_bytes(16)
    global _relay_contexts
    with _relay_lock:  # a number is used up only once this context has published under it
      self.key = key or f"mpmd/relay/{_relay_contexts}"
      self._key_sig = zlib.crc32(self.key.encode())
      self._lsock = _relay_socket(self.sockbuf)  # accepted streams inherit its buffer sizes
      try:
        self._lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = ip or env("MPMD_RELAY_IP") or _host_ip()
        self._lsock.bind((host, int(port if port is not None else env("MPMD_RELAY_PORT") or 0)))
        self._lsock.listen(256)
        self.address = f"{host}:{self._lsock.getsockname()[1]}"
        self._listener = threading.Thread(target=self._accept, name="relay-listen", daemon=True)
        self._listener.start()
        kv.key_value_set(f"{self.key}/addr/{self.pid}", f"{self.address} {self._token.hex()}")
      except BaseException:  # nobody could close this context: shut its listener here
        self._closing = self._closed = True
        try:
          self._lsock.shutdown(socket.SHUT_RDWR)
        except OSError:
          pass
        self._lsock.close()
        if self._listener is not None:
          self._listener.join(1.0)
        raise
      if key is None:
        _relay_contexts += 1

  # ------------------------------------------------ the channels' side
  def register(self, ch):
    """RelayChannel's constructor: the receives of channel `ch.idx` are its."""
    with self._cv:
      if not 0 <= ch.idx < 2**32:
        raise ValueError(f"{ch.name}: relay channel id {ch.idx} is not a uint32")
      if ch.idx in self._channels:
        raise ValueError(f"{ch.name}: relay channel id {ch.idx} is taken by {self._channels[ch.idx].name}")
      self._channels[ch.idx] = ch
      self._cv.notify_all()

  def check(self):
    """Raise if this context is closed: it sends nothing more (its peers see
    its streams end).  A send that failed fails the next one in `_send`."""
    if self._closing or self._closed:
      raise RelayError(f"relay context {self.key} (process {self.pid}) is closed")

  def _send(self, ch, seq, leaves):
    """The source side of send `seq` of `ch`: each shard's copy to host
    starts now, and it is queued for the peer holding its destination.  Once
    a send of this context has failed, none is made: each destination of
    this one gets an ERROR for it instead, in order behind what was queued
    for it -- so that its receive fails at once rather than wait out
    MPMD_RELAY_TIMEOUT, and the numbers both sides keep stay in step -- and
    it raises what failed."""
    err = self._send_err
    if err is not None:
      for q, _ in ch._send_items:
        self._peer(q).put(_RelayUnsent(ch.idx, seq, f"an earlier send failed ({err})"))
      raise err.with_traceback(None)
    t = time.monotonic()
    by_dev = [{s.device: s.data for s in x.addressable_shards} for x in leaves]
    for q, items in ch._send_items:
      p = self._peer(q)
      for i, k, dev in items:
        d = by_dev[i][dev]
        d.copy_to_host_async()
        p.put((ch.idx, seq, i, k, d, ch._nbytes[i], ch._sigs[i], ch.name, t))

  def _peer(self, q):
    with self._lock:
      if q not in self._peers:
        self._peers[q] = _RelayPeer(self, q)
      return self._peers[q]

  def _sender_failed(self, err):
    with self._lock:
      if self._send_err is None:
        self._send_err = err

  def _claim(self, ch, seq):
    """The destination side of send `seq` of `ch` -> its _RelayRecv (made
    here, or by its first chunk), failed at once if it cannot complete."""
    with self._lock:
      r = self._recvs.get((ch.idx, seq))
      if r is None:
        r = self._recvs[(ch.idx, seq)] = _RelayRecv(ch, seq)
      r.claimed = True
      if not r.event.is_set():
        if self._fatal is not None:
          self._fail_recv(r, self._fatal)
        elif self._closing:
          self._fail_recv(r, RelayError(f"{r.what()}: relay context {self.key} (process {self.pid}) is " "closed"))
        else:
          gone = sorted(q for q in r.missing() if q in self._gone)
          if gone:
            why = "".join(f"; process {q}: {self._gone_why[q]}" for q in gone if q in self._gone_why)
            self._fail_recv(
                r,
                RelayError(
                    f"{r.what()}: process(es) {gone} closed their relay streams " f"with shards of it unsent{why}"
                ),
            )
      if r.event.is_set():
        self._retire(r)
      return r

  # ---------------------------------------------------- the reader side
  def _accept(self):
    while True:
      try:
        conn, _ = self._lsock.accept()
      except OSError as e:
        if self._closing:
          return  # close() shut the listener down
        # EMFILE, ENOBUFS, ECONNABORTED...: a listener that stopped here would leave every stream
        # still to come unread, its writers and this process's receives waiting out their timeout
        warnings.warn(f"relay listener {self.address}: accept failed ({type(e).__name__}: {e}); retrying", RuntimeWarning)
        time.sleep(0.1)
        continue
      conn.settimeout(min(self.timeout_s, _RELAY_HELLO_S))  # until its HELLO is in (`_hello`)
      conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      t = threading.Thread(target=self._read, args=(conn,), name="relay-read", daemon=True)
      with self._lock:
        if self._closing:
          conn.close()
          continue
        # started under the lock, since close() joins what it finds in _readers and a thread
        # not yet started cannot be joined (start() waits for the thread to run, not for the lock)
        self._readers.append((conn, t))
        t.start()

  def _hello(self, conn):
    """A new connection's first message -> (the process it is a stream of,
    that process's number of streams), or None: anything but a HELLO with
    this context's key CRC and its token, within _RELAY_HELLO_S, is refused
    with a warning (a stray connection is no reason to stop the relay); one
    that ends before its first byte, quietly."""
    hdr = bytearray(_RELAY_HDR.size)
    try:
      if not _recv_exact(conn, hdr, eof_ok=True):
        return None
      magic, kind, q, _, _, _, _, ln, n, sig = _RELAY_HDR.unpack(hdr)
      why = f"magic {bytes(magic)!r}, kind {kind}, key CRC {sig:#x}"
      if magic == _RELAY_MAGIC and kind == _HELLO and sig == self._key_sig:
        token = bytearray(len(self._token))
        if ln == len(token):
          _recv_exact(conn, token)
          if hmac.compare_digest(bytes(token), self._token):
            conn.settimeout(None)
            return q, n
        why = f"a HELLO as process {q} without this context's token"
    except (OSError, RelayError) as e:
      why = f"no HELLO: {type(e).__name__}: {e}"
    if not self._closing:
      warnings.warn(
          f"relay listener {self.address}: refused a connection that is not a stream of this " f"context ({why})",
          RuntimeWarning,
      )
    return None

  def _read(self, conn):
    hdr = bytearray(_RELAY_HDR.size)
    try:
      hello = self._hello(conn)
      if hello is None:
        return
      peer, n = hello
      with self._lock:
        self._streams_from.setdefault(peer, [n, 0])
      try:
        while _recv_exact(conn, hdr, eof_ok=True):
          magic, kind, cid, seq, leaf, pos, off, ln, total, sig = _RELAY_HDR.unpack(hdr)
          if magic != _RELAY_MAGIC:
            raise RelayError(f"relay stream from process {peer}: bad magic {bytes(magic)!r} (the framing " "is lost)")
          if kind == _DATA:
            view = self._into(peer, cid, seq, leaf, pos, off, ln, total, sig)
            with jax.profiler.TraceAnnotation(RELAY_TRACE["recv"], bytes=ln):
              _recv_exact(conn, view)
            del view
            self._got(cid, seq, leaf, pos, ln)
          elif kind in (_ERROR, _ABORT):
            if ln > _RELAY_TEXT_MAX:
              raise RelayError(f"relay stream from process {peer}: a {ln}-byte reason (the framing is lost)")
            msg = bytearray(ln)
            _recv_exact(conn, msg)
            if kind == _ERROR:
              self._remote_error(peer, cid, seq, msg.decode(errors="replace"))
            else:
              self._peer_aborted(peer, msg.decode(errors="replace"))
          else:
            raise RelayError(f"relay stream from process {peer}: unknown message kind {kind}")
      except BaseException as e:
        if not self._closing:
          self._fail_all(
              e if isinstance(e, RelayError) else RelayError(f"relay stream from process {peer}: {type(e).__name__}: {e}")
          )
      finally:
        self._stream_ended(peer)
    finally:
      conn.close()

  def _channel(self, cid, peer):
    with self._cv:
      self._cv.wait_for(lambda: cid in self._channels or self._closing or self._fatal is not None, self.timeout_s)
      if self._fatal is not None:
        raise self._fatal
      if cid not in self._channels:
        raise RelayError(f"process {peer} sent data of relay channel #{cid}, which process {self.pid} has " "not built")
      return self._channels[cid]

  def _into(self, peer, cid, seq, leaf, pos, off, ln, total, sig):
    """Where DATA chunk (cid, seq, leaf, pos, off, ln) from `peer` goes, once
    checked against what the channel expects from it -> a writable view.  Its
    bytes are taken for it here (`spans`), before they are read: no byte of
    a shard comes twice, whichever streams its chunks take."""
    ch = self._channel(cid, peer)
    with self._lock:
      if self._fatal is not None:
        raise self._fatal
      r = self._recvs.get((cid, seq))
      if r is None:
        if (cid, seq) in self._dead:  # failed (and retired) already: drain
          return memoryview(np.empty(ln, np.uint8))
        if seq < self._floor.get(cid, 0) or seq in self._retired.get(cid, ()):
          raise RelayError(
              f"{ch.name} (relay channel #{cid}): process {peer} sent seq {seq} again, after "
              "this process completed that receive"
          )
        r = self._recvs[(cid, seq)] = _RelayRecv(ch, seq)
      where = f"{r.what()} leaf {leaf} position {pos} from process {peer}"
      spec = ch._recv_spec.get((leaf, pos))
      if spec is None or spec[2] != peer:
        raise RelayError(f"{where}: not a shard this process takes from that process")
      if (total, sig) != spec[:2]:
        raise RelayError(
            f"{where}: {total} bytes, dtype/shape CRC {sig:#010x}; this process expects "
            f"{spec[0]} bytes, CRC {spec[1]:#010x}: the two processes built the channel "
            "with different avals"
        )
      key = (leaf, pos)
      spans = r.spans.setdefault(key, [])
      if key in r.done or off + ln > total or (ln and not _take_span(spans, off, off + ln)):
        raise RelayError(
            f"{where}: bytes [{off}, {off + ln}) of a {total}-byte shard "
            + ("run past its end" if off + ln > total else "overlap bytes already in")
            + ": duplicate or overlong data"
        )
      if r.event.is_set():  # failed already, not yet retired: drain
        return memoryview(np.empty(ln, np.uint8))
      if r.t_first is None:
        r.t_first = time.monotonic()
      buf = r.bufs.get(key)
      if buf is None:
        buf = r.bufs[key] = self._buffer(cid, key, total)
        r.got[key] = 0
      return memoryview(buf)[off : off + ln]

  def _buffer(self, cid, key, total):  # under the lock
    """A host buffer for shard `key` of a receive of channel `cid`: a pooled
    one nothing else holds any more, else a new one, pooled while there is
    room (_RELAY_POOL).  Memory the allocator cannot recycle is faulted in
    and zeroed page by page: copying 58.7 MB into it took 9-13 ms against
    3.3 ms into a buffer in use before, on the THP=always VM this was
    written on (31-33 ms with THP off).  "Nothing else" counts jax:
    jax.device_put keeps what it reads, on TPU until its host-to-device copy
    is done, and on CPU, where the placed array is a view of the buffer, as
    long as that array lives; and every view of it, a reader's included.
    Nothing can take a reference to a buffer nothing holds but through the
    pool, under the lock: a free one stays free."""
    pool = self._pool.setdefault((cid, *key), [])
    for i in range(len(pool)):
      if _refs(pool, i) <= _ONLY_LISTED:
        return pool[i]
    buf = np.empty(total, np.uint8)
    if len(pool) < _RELAY_POOL:
      pool.append(buf)
    return buf

  def _got(self, cid, seq, leaf, pos, ln):
    """DATA chunk (cid, seq, leaf, pos) of `ln` bytes is in.  The chunk that
    completes a shard puts it on its device -- on this reader thread, out of
    the lock, while the other streams keep coming in -- and the one whose
    shard is a receive's last there assembles its arrays (`_finish`)."""
    with self._lock:
      self.received += ln
      r = self._recvs.get((cid, seq))
      key = (leaf, pos)
      if r is None or r.event.is_set() or key in r.done:  # done: a zero-byte shard's chunk again
        return
      size = len(r.bufs[key])
      r.got[key] += ln
      if r.got[key] < size:  # no byte twice (`_into`): its bytes add up to its size once all are in
        return
      r.done.add(key)
      buf = r.bufs.pop(key)
      if len(r.done) == len(r.ch._recv_spec):
        r.t_last = time.monotonic()
    try:
      x = r.ch._place(key, buf)
    except BaseException as e:
      with self._lock:
        if not r.event.is_set():
          self._fail_recv(
              r,
              RelayError(
                  f"{r.what()}: placing leaf {leaf} position {pos} on its device " f"failed: {type(e).__name__}: {e}"
              ),
          )
          if r.claimed:
            self._retire(r)
      return
    del buf
    with self._lock:
      if r.event.is_set():  # failed meanwhile
        return
      r.placed[key] = x
      if len(r.placed) < len(r.ch._recv_spec):
        return
      placed = r.placed
    self._finish(r, placed)

  def _finish(self, r, placed):
    """Every shard of receive r is on its device: its arrays (no copy)."""
    try:
      out, err = r.ch._assemble(placed), None
    except BaseException as e:
      out, err = None, RelayError(f"{r.what()}: assembling the received shards failed: {type(e).__name__}: " f"{e}")
    t = time.monotonic()
    with self._lock:
      if r.event.is_set():  # failed meanwhile (`_fail_all`): that stands
        return
      if err is not None:
        self._fail_recv(r, err)
      else:  # noted before a consumer can see it complete
        self._note("recv_wire", r.ch.name, r.t_last - r.t_first)
        self._note("recv_dispatch", r.ch.name, t - r.t_last)
        r.out, r.placed = out, None
        r.event.set()
      if r.claimed:
        self._retire(r)

  def _remote_error(self, peer, cid, seq, msg):
    ch = self._channel(cid, peer)
    with self._lock:
      r = self._recvs.get((cid, seq))
      if r is None:
        if seq < self._floor.get(cid, 0) or seq in self._retired.get(cid, ()):
          return
        r = self._recvs[(cid, seq)] = _RelayRecv(ch, seq)
      if not r.event.is_set():
        self._fail_recv(r, RelayError(f"{r.what()}: process {peer} could not send it: {msg}"))
        if r.claimed:
          self._retire(r)

  def _stream_ended(self, peer):
    """One of `peer`'s streams ended.  Once every stream its HELLO announced
    has (each after all that was sent on it is read), what is still missing
    its shards never comes.  One stream's end alone says nothing: another may
    still carry the rest of a receive."""
    with self._lock:
      st = self._streams_from[peer]
      st[1] += 1
      if st[1] < st[0] or self._closing:
        return
      self._gone.add(peer)
      self._fail_missing(
          peer, lambda r, n: f"{r.what()}: process {peer} closed its relay streams with {n} " "shard(s) of it unsent"
      )

  def _peer_aborted(self, peer, msg):
    """An ABORT from `peer`: some of its streams never opened, so the rest of
    what it sends here never comes (`_RelayPeer._stage`)."""
    with self._lock:
      self._gone.add(peer)
      self._gone_why[peer] = msg
      self._fail_missing(
          peer,
          lambda r, n: f"{r.what()}: process {peer} gave up on its relay streams to "
          f"this process with {n} shard(s) of it unsent: {msg}",
      )

  def _fail_missing(self, peer, why):  # under the lock
    for r in list(self._recvs.values()):
      n = r.missing().get(peer) if not r.event.is_set() else None
      if n:
        self._fail_recv(r, RelayError(why(r, n)))
        if r.claimed:
          self._retire(r)

  def _fail_all(self, err):
    with self._cv:
      if self._fatal is None:
        self._fatal = err
      for r in list(self._recvs.values()):
        if not r.event.is_set():
          self._fail_recv(r, err)
          if r.claimed:
            self._retire(r)
      self._cv.notify_all()

  def _fail_recv(self, r, err):  # under the lock
    """Fail receive r: its futures raise `err`, and whatever of it still
    comes is drained (`_into`), not taken for a new receive or a repeat."""
    r.err, r.bufs, r.placed = err, None, None
    r.event.set()
    self._dead.add((r.ch.idx, r.seq))

  def _retire(self, r):
    """Forget a receive both claimed and finished (under the lock); its
    number is remembered, so a later chunk of it is an error, not a new
    receive."""
    cid = r.ch.idx
    if self._recvs.get((cid, r.seq)) is r:
      del self._recvs[(cid, r.seq)]
    done = self._retired.setdefault(cid, set())
    done.add(r.seq)
    f = self._floor.get(cid, 0)
    while f in done:
      done.discard(f)
      f += 1
    self._floor[cid] = f

  # ------------------------------------------------------------- the rest
  def _note(self, measure, channel, secs):
    with self._tlock:
      t = self._times.get((measure, channel))
      if t is None:
        t = self._times[(measure, channel)] = [0, 0.0, 0.0, collections.deque(maxlen=_RELAY_SAMPLES)]
      t[0] += 1
      t[1] += secs
      t[2] = max(t[2], secs)
      t[3].append(secs)

  def timing(self, reset=False):
    """Where the relay's time went on this process (RELAY_MEASURES) ->
    {'all': {measure: summary}, 'channels': {channel name: {measure:
    summary}}}, a summary {'n': the samples since the last reset, 'total_s':
    their sum, 'max_ms': their largest, 'p50_ms' / 'p99_ms': percentiles of
    the newest _RELAY_SAMPLES of them per channel, 'kept': how many those
    are}.  `reset`: forget the samples (say, after warm-up)."""
    with self._tlock:
      snap = {k: (n, total, top, list(q)) for k, (n, total, top, q) in self._times.items()}
      if reset:
        self._times.clear()

    def summary(n, total, top, xs):
      p50, p99 = np.percentile(np.asarray(xs, np.float64), [50, 99])
      return dict(n=n, p50_ms=float(p50) * 1e3, p99_ms=float(p99) * 1e3, max_ms=top * 1e3, total_s=total, kept=len(xs))

    every, by_ch = {}, {}
    for (m, ch), (n, total, top, xs) in sorted(snap.items()):
      e = every.setdefault(m, [0, 0.0, 0.0, []])
      e[0], e[1], e[2] = e[0] + n, e[1] + total, max(e[2], top)
      e[3].extend(xs)
      by_ch.setdefault(ch, {})[m] = summary(n, total, top, xs)
    return {"all": {m: summary(*e) for m, e in every.items()}, "channels": by_ch}

  def describe(self):
    """The relay's settings and this host's (`host_net_settings`), one line."""
    buf = (
        f"{self.sockbuf / 2**20:g} MiB socket buffers"
        if self.sockbuf
        else "socket buffers left to the kernel's autotuning"
    )
    return (
        f"relay {self.key} p{self.pid} at {self.address}: {self.streams} streams per peer, "
        f"{self.chunk / 2**20:g} MiB chunks, {buf}, timeout {self.timeout_s:g} s; this host: "
        + ", ".join(f"{k}: {v}" for k, v in host_net_settings().items())
    )

  def stats(self):
    """{'sent': bytes written to peers, 'received': bytes read from them,
    'unclaimed': receives that arrived (in part) with no send_flat for them
    yet, 'waiting': claimed receives not complete}."""
    with self._lock:
      return dict(
          sent=sum(p.sent for p in self._peers.values()),
          received=self.received,
          unclaimed=sorted(r.what() for r in self._recvs.values() if not r.claimed),
          waiting=sorted(r.what() for r in self._recvs.values() if r.claimed),
      )

  def close(self, timeout_s=10.0):
    """Flush every queued send, half-close the streams (the peers' readers see
    them end between messages), give the peers' streams into this process
    the same time to end, then stop the listener and every thread -- all
    within `timeout_s`.  A peer whose threads are still at work then (one
    that stopped reading, say) has its streams shut: its threads end within
    a second more, and it sees the streams end.  A receive claimed but not
    complete fails at once: nothing completes it once the streams are gone.
    Idempotent.  Warns, never raises (it runs on the way out, where it must
    not hide what failed), about sends left unwritten, receives claimed but
    not complete, receives that arrived with nobody to claim them, and
    threads that did not end."""
    if self._closed:
      return
    self._closed = True
    deadline = time.monotonic() + timeout_s
    peers = list(self._peers.values())
    unflushed = [p.pid for p in peers if not p.flush(max(0.0, deadline - time.monotonic()))]
    with self._cv:
      self._closing = True
      waiting = sorted(
          (r for r in self._recvs.values() if r.claimed and not r.event.is_set()), key=lambda r: (r.ch.idx, r.seq)
      )
      for r in waiting:
        self._fail_recv(
            r,
            RelayError(
                f"{r.what()}: relay context {self.key} (process {self.pid}) is closed " "and the receive was not complete"
            ),
        )
        self._retire(r)
      self._cv.notify_all()
      peers, readers = list(self._peers.values()), list(self._readers)
    for p in peers:
      p.stop()
    for p in peers:
      p.join(deadline)
    stuck = [p for p in peers if p.running()]
    for p in stuck:  # else, until SO_SNDTIMEO, its writers would hold their buffers and shards
      p.abort(
          RelayError(f"relay context {self.key} (process {self.pid}) closed with sends to process {p.pid} " "unwritten")
      )
    for p in stuck:
      p.join(time.monotonic() + 1.0)
    for _, t in readers:
      t.join(max(0.0, deadline - time.monotonic()))
    for s in (self._lsock, *(c for c, _ in readers)):
      try:
        s.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass
    self._lsock.close()
    for c, t in readers:
      t.join(1.0)
      c.close()
    self._listener.join(1.0)
    with self._lock:
      self._pool.clear()  # what a live array or a reader still holds goes with it
    st = self.stats()
    running = [n for p in peers for n in p.running()]
    running += [t.name for t in (self._listener, *(t for _, t in readers)) if t.is_alive()]
    bad = []
    if unflushed:
      bad.append(f"sends to process(es) {unflushed} not all written")
    if waiting:
      bad.append(f"{len(waiting)} claimed receive(s) not complete (failed now), e.g. {waiting[0].what()}")
    if st["unclaimed"]:
      bad.append(
          f'{len(st["unclaimed"])} receive(s) arrived that no send_flat claimed, e.g. '
          f'{st["unclaimed"][0]} (the processes walked different orders?)'
      )
    if self._send_err is not None:
      bad.append(f"a send failed: {self._send_err}")
    if running:  # say, a stager still waiting for its shard's program, or for an address
      bad.append(f'threads still running: {", ".join(running)}')
    if bad:
      warnings.warn(f"RelayContext {self.key} (process {self.pid}) closed with " + "; ".join(bad), RuntimeWarning)

  def __repr__(self):
    return (
        f"<RelayContext {self.key} p{self.pid} at {self.address} streams={self.streams} "
        f"chunk={self.chunk >> 10}K peers={sorted(self._peers)}>"
    )


def relay_timing_line(t):
  """RelayContext.timing()'s summary in one line: each measure's p50 / p99,
  and the channels whose consumers waited longest in all."""
  a = t["all"]
  if not a:
    return "no transfers"
  pp = lambda m: f'p50 {a[m]["p50_ms"]:.2f} / p99 {a[m]["p99_ms"]:.2f} ms' if m in a else "none"
  parts = []
  if "send_host" in a:
    parts.append(f'sent {a["send_host"]["n"]} shards (on the host {pp("send_host")}, written {pp("send_wire")})')
  if "recv_wire" in a:
    parts.append(
        f'received {a["recv_wire"]["n"]} (wire {pp("recv_wire")}, dispatched to the devices ' f'{pp("recv_dispatch")})'
    )
  if "recv_wait" in a:
    w = sorted(((c["recv_wait"]["total_s"], ch) for ch, c in t["channels"].items() if "recv_wait" in c), reverse=True)
    parts.append(
        f'consumers waited {pp("recv_wait")}, {a["recv_wait"]["total_s"]:.3f} s in all ('
        + ", ".join(f"{ch} {s:.3f} s" for s, ch in w[:4])
        + (", ..." if len(w) > 4 else "")
        + ")"
    )
  return "; ".join(parts)


class RelayChannel:
  """One stage boundary over the host relay (RelayContext).  Each source
  host starts its shards' copies to host memory and queues them for the host
  holding the same mesh positions of the destination stage, which receives
  them in threads of its own and places each on its device as it completes.
  Nothing goes through libtpu's cross-host machinery or a collective, and
  neither stage waits for the other at the boundary: the source's send_flat
  returns once its shards are queued; the destination's returns a
  RelayFuture per leaf at once, and only `resolve` -- its consumer -- waits
  for the bytes.

  The calling contract is XferChannel's and PermuteChannel's: every process
  calls send_flat for every copy, in the one global order (the two stages'
  processes act, everyone else gets None back).  Messages carry the
  channel's index and a sequence number both sides advance on every send,
  so a receive takes its own data whatever arrives first.  Shard k of the
  source mesh goes to device k of the destination: identical meshes, specs
  and per-device index maps, checked at build time; any host pairing
  (`relay_routes`).  A sent array is held, not copied, until its shards
  are on the host (on CPU, until they are on the wire): never donate or
  delete one (drop the reference)."""

  __slots__ = (
      "name",
      "idx",
      "treedef",
      "avals",
      "src",
      "dst",
      "n_leaves",
      "ctx",
      "seq",
      "owns_src",
      "owns_dst",
      "sends",
      "recvs",
      "_send_items",
      "_recv_pos",
      "_recv_devs",
      "_recv_spec",
      "_dev_at",
      "_shapes",
      "_dtypes",
      "_nbytes",
      "_sigs",
  )

  def __init__(self, avals, src_shardings, dst_shardings, ctx, idx, name="rchannel"):
    flat_avals, treedef = jax.tree.flatten(avals)
    self.name, self.idx, self.ctx, self.seq = name, idx, ctx, 0
    self.treedef, self.avals, self.n_leaves = treedef, tuple(flat_avals), len(flat_avals)
    bc = lambda s: [s] * self.n_leaves if isinstance(s, Sharding) else list(treedef.flatten_up_to(s))
    self.src, self.dst = bc(src_shardings), bc(dst_shardings)
    ma, mb = _same_mesh_boundary(name, flat_avals, self.src, self.dst, as_bytes=True)
    try:
      self.sends, self.recvs = relay_routes(ma, mb, ctx.pid)
    except ValueError as e:
      raise ValueError(f"{name}: {e}") from None
    self.owns_src, self.owns_dst = bool(self.sends), bool(self.recvs)
    fa, fb = list(ma.devices.flat), list(mb.devices.flat)
    self._shapes = [tuple(d.shard_shape(tuple(a.shape))) for a, d in zip(self.avals, self.dst)]
    self._dtypes = [np.dtype(a.dtype) for a in self.avals]
    self._nbytes = [math.prod(s) * t.itemsize for s, t in zip(self._shapes, self._dtypes)]
    self._sigs = [_relay_sig(t, s) for t, s in zip(self._dtypes, self._shapes)]
    self._send_items = [
        (q, [(i, k, fa[k]) for i in range(self.n_leaves) for k in ks]) for q, ks in sorted(self.sends.items())
    ]
    src_of = {k: q for q, ks in self.recvs.items() for k in ks}
    # mesh-position order: the destination sharding's addressable order
    self._recv_pos = sorted(src_of)
    self._recv_devs = [fb[k] for k in self._recv_pos]
    self._dev_at = dict(zip(self._recv_pos, self._recv_devs))
    self._recv_spec = {
        (i, k): (self._nbytes[i], self._sigs[i], src_of[k]) for i in range(self.n_leaves) for k in self._recv_pos
    }
    ctx.register(self)

  def _place(self, key, buf):
    """A reader thread, at the last byte of shard `key` = (leaf, mesh
    position): its bytes as numpy -> its destination device."""
    i, k = key
    with jax.profiler.TraceAnnotation(RELAY_TRACE["place"], bytes=buf.nbytes):
      return jax.device_put(buf.view(self._dtypes[i]).reshape(self._shapes[i]), self._dev_at[k])

  def _assemble(self, placed):
    """Once every shard of a receive is on its device ({(leaf, mesh
    position): single-device array}): the destination arrays."""
    return [
        jax.make_array_from_single_device_arrays(tuple(a.shape), d, [placed[(i, k)] for k in self._recv_pos])
        for i, (a, d) in enumerate(zip(self.avals, self.dst))
    ]

  def send_flat(self, leaves):
    """Every process, every boundary copy in order: source arrays where owned
    -> a RelayFuture per leaf on the destination stage's processes, None
    elsewhere (and on the source side, which returns once its shards are
    queued).  After a send of the context failed, the source side raises,
    once it has told the destinations this one is not coming."""
    if not (self.owns_src or self.owns_dst):
      return [None] * self.n_leaves
    if self.owns_src:
      _check_sources(self.name, leaves, self.avals, self.src)
      self.ctx.check()  # closed: no number, no message (the peers see the streams end)
      seq, self.seq = self.seq, self.seq + 1
      self.ctx._send(self, seq, leaves)
      return [None] * self.n_leaves
    seq, self.seq = self.seq, self.seq + 1
    r = self.ctx._claim(self, seq)
    return [RelayFuture(r, i) for i in range(self.n_leaves)]

  def send(self, tree):
    return jax.tree.unflatten(self.treedef, self.send_flat(self.treedef.flatten_up_to(tree)))

  def __repr__(self):
    return f"<RelayChannel {self.name} #{self.idx} leaves={self.n_leaves}>"


class _Aval:
  """Minimal shape/dtype carrier (avoids importing jax.core here)."""

  __slots__ = ("shape", "dtype")

  def __init__(self, shape, dtype):
    self.shape, self.dtype = tuple(shape), dtype


class _ShapedSrc:
  """Adapter so `assert_d2d` can be used on an (aval, sharding) pair."""

  __slots__ = ("shape", "dtype", "sharding")

  def __init__(self, aval, sharding):
    self.shape, self.dtype, self.sharding = tuple(aval.shape), aval.dtype, sharding


# ------------------------------------------------------------ CI: no D2H at all
@contextlib.contextmanager
def no_host_bounce():
  """Make every device->host fetch raise `HostBounceError` inside the block.

  Patches `jax._src.array.ArrayImpl._value`, which is the single funnel for
  `np.asarray`, `float()`, `print`, and the silent reshard round trip.  Note on
  the CPU backend a SINGLE-shard array also satisfies the Python buffer
  protocol, so `np.asarray` on it bypasses `_value` entirely; multi-device
  sharded arrays (the pipeline case) always go through `_value`.
  """
  if _jax_array is None:  # pragma: no cover
    raise RuntimeError("jax._src.array unavailable; cannot install the guard")
  cls = _jax_array.ArrayImpl
  original = cls.__dict__.get("_value")
  if original is None:  # pragma: no cover
    raise RuntimeError("ArrayImpl._value not found; jax internals changed")

  def _raise(self):
    raise HostBounceError(
        f"device->host transfer of a {self.shape} {self.dtype} array inside "
        "no_host_bounce(): something called np.asarray/float/print on a device "
        "array, or a device_put resharded instead of copying."
    )

  setattr(cls, "_value", property(_raise))
  try:
    yield
  finally:
    setattr(cls, "_value", original)


# --------------------------------------------------------------- run-ahead cap
# The PJRT client-creation option that caps enqueued-but-incomplete executions
# per device.  It is NOT a jax-side name: it appears nowhere in jax/jaxlib
# python (`grep -rn max_inflight` over both packages returns nothing) and is
# read by the plugin.  `libtpu.so` contains the literal string, which is the
# only evidence available in a CPU-only process; `option_name_evidence()`
# re-checks that at runtime instead of taking it on faith.
MAX_INFLIGHT_OPTION = "max_inflight_computations"

# Platforms whose PJRT client is created through a factory that actually reads
# `jax_pjrt_client_create_options`.  In jax 0.11.1 the config is funnelled
# through `xla_bridge._options_from_jax_configs`, which has exactly two callers:
# `xla_bridge.make_tpu_client` and `xla_bridge.make_pjrt_c_api_client`.
# The built-in CPU client comes from
# `xla_client.make_cpu_client()`, whose signature has no options argument at
# all, so on CPU this whole knob is inert -- and a healthy `jax.devices()` after
# setting a garbage option proves exactly nothing.  The CUDA / ROCm plugins are
# C-API clients (make_pjrt_c_api_client), whose PJRT_Client_Create takes the
# option too: the GPU client caps a device's enqueued executions at 32 as well.
OPTION_AWARE_PLATFORMS = ("tpu", "cuda", "gpu", "rocm")


def pjrt_client_options(plugin: str = "tpu"):
  """The exact options dict jax would hand to `plugin`'s PJRT client right now.

  Thin wrapper over `jax._src.xla_bridge._options_from_jax_configs(plugin)` --
  the single place `jax_pjrt_client_create_options` is turned into client
  options.  Returns None if that private helper is gone (jax internals drift).
  """
  try:
    from jax._src import xla_bridge as xb

    fn = getattr(xb, "_options_from_jax_configs", None)
    return None if fn is None else dict(fn(plugin))
  except Exception:  # pragma: no cover - jax internals
    return None


def pinned_platforms() -> tuple[str, ...]:
  """Platforms this process is pinned to, without initializing any backend."""
  val = getattr(jax.config, "jax_platforms", None) or os.environ.get("JAX_PLATFORMS", "")
  return tuple(p.strip().lower() for p in str(val).split(",") if p.strip())


def option_name_evidence():
  """`(found, where)` for the literal option name in an installed PJRT plugin.

  Scans `libtpu.so` for `MAX_INFLIGHT_OPTION`.  `found is None` means no plugin
  binary was found to scan (nothing proven either way), False means the name is
  absent from the plugin that would consume it -- i.e. a typo.
  """
  import mmap
  # Fixed locations only.  This used to add a recursive `/usr/lib/**` glob --
  # evaluated whether or not the pip wheel had already been found -- and it
  # runs in every `TransferTest.test_transfer_layer` (transfer_test.py, through
  # `check_transfer_layer`): a walk over every file under /usr/lib, following
  # directory symlinks, on every run of that test.
  pats = [
      os.path.join(os.path.dirname(os.path.dirname(jax.__file__)), "libtpu", "libtpu.so"),
      "/usr/lib/libtpu.so",
      "/usr/local/lib/libtpu.so",
  ]
  for path in pats:
    if not os.path.exists(path):
      continue
    try:
      with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
          return m.find(MAX_INFLIGHT_OPTION.encode()) >= 0, path
    except Exception:  # pragma: no cover - unreadable
      continue
  return None, None


def set_max_inflight(n: int = 128, plugin: str = "tpu") -> bool:
  """Raise the PJRT per-device run-ahead queue above its default of 32.

  The 33rd enqueued-but-incomplete execution BLOCKS the single driver thread,
  and a 4-stage x 16-microbatch fwd+bwd schedule sits exactly on that cliff.

  MUST be called before the first backend initialization (i.e. before any
  `jax.devices()` / first jit).

  Returns True ONLY when the value was (a) accepted by the config system, (b)
  observed, as an int, in the very options dict `plugin`'s client factory reads
  (`pjrt_client_options`), and (c) this process is actually going to build that
  kind of client.  Otherwise it warns and returns False -- never an exception,
  except for a nonsense `n`, which is a caller bug.  In particular it returns
  False on a CPU-pinned process: `make_cpu_client` takes no options, so the cap
  is not applied and reporting success there would be a lie.
  """
  if isinstance(n, bool) or not isinstance(n, int):
    raise TypeError(
        f"set_max_inflight(n): n must be an int, got {n!r} "
        f"({type(n).__name__}); PJRT silently ignores an option it "
        "cannot parse."
    )
  if n <= 0:
    raise ValueError(f"set_max_inflight(n): n must be positive, got {n}")
  try:
    from jax._src import xla_bridge as xb

    if xb.backends_are_initialized():
      warnings.warn(
          "set_max_inflight() called after backend initialization; "
          "PJRT client options are read only at client creation, so "
          "this is a no-op. Call it before the first jax.devices().",
          RuntimeWarning,
      )
      return False
  except Exception:  # pragma: no cover
    pass
  cur = getattr(jax.config, "jax_pjrt_client_create_options", None)
  opts = {}
  if isinstance(cur, dict):
    opts.update(cur)
  elif isinstance(cur, str) and cur:
    # jax ships a default "k1:v1;k2:v2" string (ml_framework_name/version);
    # keep those entries instead of clobbering them with a bare dict.
    for kv in cur.split(";"):
      if kv.count(":") == 1:
        k, v = kv.split(":")
        opts[k] = v
  keep = dict(opts)
  opts[MAX_INFLIGHT_OPTION] = n
  try:
    jax.config.update("jax_pjrt_client_create_options", opts)
  except Exception as e:
    warnings.warn(f"set_max_inflight({n}) rejected by this jax build: " f"{type(e).__name__}: {e}", RuntimeWarning)
    return False

  seen = pjrt_client_options(plugin)
  if seen is None:
    warnings.warn(
        "set_max_inflight(): jax._src.xla_bridge."
        "_options_from_jax_configs is gone, so there is no way to "
        "verify the option reaches the PJRT client; treating as not "
        "applied.",
        RuntimeWarning,
    )
    return False
  if seen.get(MAX_INFLIGHT_OPTION) != n or not isinstance(seen.get(MAX_INFLIGHT_OPTION), int):
    warnings.warn(
        f"set_max_inflight({n}): the option did not survive into the "
        f"{plugin} client options ({seen!r}); not applied.",
        RuntimeWarning,
    )
    return False
  dropped = [k for k, v in keep.items() if seen.get(k) != v]
  if dropped:  # pragma: no cover - defensive
    warnings.warn(
        f"set_max_inflight({n}) clobbered pre-existing PJRT client " f"options {dropped}; not applied.", RuntimeWarning
    )
    return False
  plats = pinned_platforms()
  if "proxy" in plats and not any(p in OPTION_AWARE_PLATFORMS for p in plats):
    warnings.warn(
        f"set_max_inflight({n}) stored the option, but JAX_PLATFORMS={','.join(plats)} is Pathways: the IFRT proxy "
        "client takes no PJRT client options, so the run-ahead cap is NOT applied here -- how far the client runs "
        "ahead is the Pathways workers' (ds_ep_pipeline does not call this under the proxy; a local Pathways run "
        "queued ~90 steps without blocking).",
        RuntimeWarning,
    )
    return False
  if plats and not any(p in OPTION_AWARE_PLATFORMS for p in plats):
    warnings.warn(
        f"set_max_inflight({n}) stored the option and it does reach "
        f"_options_from_jax_configs({plugin!r}), but JAX_PLATFORMS="
        f'{",".join(plats)} pins this process to a backend whose client is '
        f"built by xla_client.make_cpu_client(), which takes no options: the "
        f"run-ahead cap is NOT applied here. Only {OPTION_AWARE_PLATFORMS} "
        "read jax_pjrt_client_create_options.",
        RuntimeWarning,
    )
    return False
  return True


def block_all(pytree):
  """The single end-of-step barrier.  One call, not a loop of per-array waits."""
  return jax.block_until_ready(pytree)


# -------------------------------------------------------------------- smoke
def smoke_test():
  """Fail loudly if the batched-copy fast path is gone after a jaxlib bump."""
  if not BATCHED_COPY_AVAILABLE:
    raise RuntimeError(
        "xla_client.batched_copy_array_to_devices_with_sharding is MISSING in "
        f"this jaxlib; Channel would silently degrade to jax.device_put for "
        "every leaf, costing ~3x the host dispatch. Pin jaxlib or update "
        "transfer.py."
    )
  import numpy as np
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

  devs = jax.devices()
  half = max(1, len(devs) // 2)
  src_mesh = Mesh(np.array(devs[:half]), ("fsdp",))
  dst_mesh = Mesh(np.array(devs[half:] or devs[:half]), ("fsdp",))
  spec = P("fsdp", None)
  x_np = np.arange(8 * half * 4, dtype=np.float32).reshape(8 * half, 4)
  x = jax.device_put(x_np, NamedSharding(src_mesh, spec))
  dst = NamedSharding(dst_mesh, spec)
  if not same_index_map(x.sharding, dst, x.shape):
    raise RuntimeError("smoke_test: identical specs on equal-sized meshes no " "longer produce identical index maps")
  ch = Channel(jax.ShapeDtypeStruct(x.shape, x.dtype), x.sharding, dst, name="smoke")
  if ch.n_fast != 1:
    raise RuntimeError(f"smoke_test: fast path not taken ({ch.reasons})")
  y = block_all(ch.send(x))
  if not np.array_equal(np.asarray(y), x_np):
    raise RuntimeError("smoke_test: batched copy returned wrong values")
  if y.sharding != dst:
    raise RuntimeError(f"smoke_test: wrong output sharding {y.sharding}")
  return True
