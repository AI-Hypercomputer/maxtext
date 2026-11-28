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

"""Automatic in-place XSpace pipeline timeline decoder and augmenter for MaxText.

Decodes multi-stage pipeline-parallel TPU XPlane traces (built-in ``gpipe`` and
``circular``, plus any schedule registered via ``register_pipeline_schedule``),
aligns cross-host clocks onto Process 0's native clock domain using global
all-reduce anchors, and appends two stacked ``/device:CUSTOM:`` planes in-place
into Process 0's ``.xplane.pb`` file while preserving 100% of existing planes
untouched:

  - ``/device:CUSTOM:1. Pipeline Stages (...)``
  - ``/device:CUSTOM:2. Bubbles & Comm Waits (By Stage)``

Extending for a new pipeline schedule:
  Register a schedule builder with ``register_pipeline_schedule`` so that
  ``PipelineOracle`` can map ``(stage, iteration/tick, pass)`` to
  ``(microbatch, repeat_or_chunk, valid)``::

    from maxtext.utils import pipeline_trace_lens as ptl

    def build_my_schedule(num_stages, num_microbatches, num_repeats=1, forwarding_delay=1):
      n_iter = num_microbatches * num_repeats + forwarding_delay * (num_stages - 1)

      def fwd_slot(iteration, stage):
        k = iteration - forwarding_delay * stage
        if 0 <= k < num_microbatches * num_repeats:
          return k % num_microbatches, k // num_microbatches, True
        return 0, 0, False

      return ptl.PipelineSchedule(
          name="my_schedule",
          mode="fwd_bwd",
          num_stages=num_stages,
          num_microbatches=num_microbatches,
          num_repeats=num_repeats,
          forwarding_delay=forwarding_delay,
          n_iter=n_iter,
          fwd_slot=fwd_slot,
      )

    ptl.register_pipeline_schedule("my_schedule", build_my_schedule)

If an unregistered schedule or unexpected loop structure is encountered, the
decoder falls back gracefully without crashing or mislabeling microbatches: it
still aligns stage clocks across hosts, emits stacked per-stage rows with raw
``iter 0 .. T-1`` tick spans and ``pp_wait`` / ``moe/fsdp wait`` spans, and
points to ``docs/guides/optimization/pipeline_parallelism_profiling.md`` on the
``1 Step & Hardware Map`` row.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import concurrent.futures
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import socket
import statistics
import time
from typing import Any

from etils import epath
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
import numpy as np

GENERATOR_NAME = "MaxText Pipeline Trace Lens"
STAGE_TRACES_SUBDIR = "_pp_stage_traces"
DEVELOPER_DOCS_PATH = "docs/guides/optimization/pipeline_parallelism_profiling.md"
PS_PER_US = 1e6

# ---------------------------------------------------------------------------
# Self-contained XSpace / XPlane protobuf schema (matches upstream xplane.proto)
# ---------------------------------------------------------------------------

_FD = descriptor_pb2.FieldDescriptorProto
_XPLANE_PKG = "tensorflow.profiler"
_XPLANE_SCHEMA = {
    "XSpace": (
        [
            ("planes", 1, "msg", True, "XPlane", None),
            ("errors", 2, "str", True, None, None),
            ("warnings", 3, "str", True, None, None),
            ("hostnames", 4, "str", True, None, None),
        ],
        [],
        [],
    ),
    "XPlane": (
        [
            ("id", 1, "i64", False, None, None),
            ("name", 2, "str", False, None, None),
            ("lines", 3, "msg", True, "XLine", None),
            ("stats", 6, "msg", True, "XStat", None),
        ],
        [],
        [
            ("event_metadata", 4, "XEventMetadata"),
            ("stat_metadata", 5, "XStatMetadata"),
        ],
    ),
    "XLine": (
        [
            ("id", 1, "i64", False, None, None),
            ("display_id", 10, "i64", False, None, None),
            ("name", 2, "str", False, None, None),
            ("display_name", 11, "str", False, None, None),
            ("timestamp_ns", 3, "i64", False, None, None),
            ("duration_ps", 9, "i64", False, None, None),
            ("events", 4, "msg", True, "XEvent", None),
        ],
        [],
        [],
    ),
    "XEvent": (
        [
            ("metadata_id", 1, "i64", False, None, None),
            ("offset_ps", 2, "i64", False, None, 0),
            ("num_occurrences", 5, "i64", False, None, 0),
            ("duration_ps", 3, "i64", False, None, None),
            ("stats", 4, "msg", True, "XStat", None),
        ],
        ["data"],
        [],
    ),
    "XStat": (
        [
            ("metadata_id", 1, "i64", False, None, None),
            ("double_value", 2, "dbl", False, None, 0),
            ("uint64_value", 3, "u64", False, None, 0),
            ("int64_value", 4, "i64", False, None, 0),
            ("str_value", 5, "str", False, None, 0),
            ("bytes_value", 6, "bytes", False, None, 0),
            ("ref_value", 7, "u64", False, None, 0),
        ],
        ["value"],
        [],
    ),
    "XEventMetadata": (
        [
            ("id", 1, "i64", False, None, None),
            ("name", 2, "str", False, None, None),
            ("display_name", 4, "str", False, None, None),
            ("metadata", 3, "bytes", False, None, None),
            ("stats", 5, "msg", True, "XStat", None),
            ("child_id", 6, "i64", True, None, None),
        ],
        [],
        [],
    ),
    "XStatMetadata": (
        [
            ("id", 1, "i64", False, None, None),
            ("name", 2, "str", False, None, None),
            ("description", 3, "str", False, None, None),
        ],
        [],
        [],
    ),
}
_PROTO_TYPES = {
    "i64": _FD.TYPE_INT64,
    "u64": _FD.TYPE_UINT64,
    "dbl": _FD.TYPE_DOUBLE,
    "str": _FD.TYPE_STRING,
    "bytes": _FD.TYPE_BYTES,
    "msg": _FD.TYPE_MESSAGE,
}


def _build_xplane_descriptor():
  """Builds an in-memory FileDescriptorProto matching xplane.proto."""
  fdp = descriptor_pb2.FileDescriptorProto(
      name="maxtext/utils/xplane_schema.proto",
      package=_XPLANE_PKG,
      syntax="proto3",
  )
  for msg_name, (fields, oneofs, maps) in _XPLANE_SCHEMA.items():
    msg = fdp.message_type.add(name=msg_name)
    for oneof_name in oneofs:
      msg.oneof_decl.add(name=oneof_name)
    for name, num, ftype, repeated, type_name, oneof in fields:
      field = msg.field.add(
          name=name,
          number=num,
          type=_PROTO_TYPES[ftype],
          label=_FD.LABEL_REPEATED if repeated else _FD.LABEL_OPTIONAL,
      )
      if type_name:
        field.type_name = f".{_XPLANE_PKG}.{type_name}"
      if oneof is not None:
        field.oneof_index = oneof
    for name, num, val_type in maps:
      entry_name = "".join(part.capitalize() for part in name.split("_")) + "Entry"
      entry = msg.nested_type.add(name=entry_name)
      entry.options.map_entry = True
      entry.field.add(name="key", number=1, type=_FD.TYPE_INT64, label=_FD.LABEL_OPTIONAL)
      entry.field.add(
          name="value",
          number=2,
          type=_FD.TYPE_MESSAGE,
          label=_FD.LABEL_OPTIONAL,
          type_name=f".{_XPLANE_PKG}.{val_type}",
      )
      msg.field.add(
          name=name,
          number=num,
          type=_FD.TYPE_MESSAGE,
          label=_FD.LABEL_REPEATED,
          type_name=f".{_XPLANE_PKG}.{msg_name}.{entry_name}",
      )
  return fdp


_XPLANE_POOL = descriptor_pool.DescriptorPool()
_XPLANE_POOL.Add(_build_xplane_descriptor())


def _xplane_cls(name: str):
  return message_factory.GetMessageClass(_XPLANE_POOL.FindMessageTypeByName(f"{_XPLANE_PKG}.{name}"))


XSpace = _xplane_cls("XSpace")
XPlane = _xplane_cls("XPlane")
XLine = _xplane_cls("XLine")
XEvent = _xplane_cls("XEvent")
XStat = _xplane_cls("XStat")
XEventMetadata = _xplane_cls("XEventMetadata")
XStatMetadata = _xplane_cls("XStatMetadata")


# ---------------------------------------------------------------------------
# XPlane writing helpers (stats, dedup metadata, flows, XProf palette colors)
# ---------------------------------------------------------------------------

FLOW_IN = 1
FLOW_OUT = 2
FLOW_INOUT = 3
_FLOW_MASK = (1 << 56) - 1

PASS_METADATA_NAMES = {
    "F": "Forward",
    "B": "Backward",
    "W": "WeightGrad",
    "BW": "BW",
    "Tail": "Tail",
    "Iter": "Pipeline Iter (Fallback)",
}
_PRIME_SUFFIX = {0: "", 1: "\u2032", 2: "\u2033"}

# XProf Trace Viewer colors each event slice by a hash of its display title mod 23.
# Appending invisible zero-width characters keeps one consistent palette color per
# pass/bubble type while still showing compact labels like "F3" or "B0'".
_ZERO_WIDTH_CHARS = ("\u200b", "\u200c", "\u200d", "\u2060")
_COLOR_SLOTS = {
    "F": 7,
    "B": 5,
    "W": 1,
    "BW": 18,
    "Tail": 6,
    "dummy": 10,
    "skipped": 4,
    "pp_wait": 0,
    "other_wait": 8,
    "tickA": 17,
    "tickB": 22,
    "phase": 14,
}
_COLOR_CACHE: dict[tuple[str, int], str] = {}


def flow_value(flow_id: int, direction: int, category: int = 0) -> int:
  return ((category & 0x3F) << 58) | ((int(flow_id) & _FLOW_MASK) << 2) | (direction & 3)


def _catapult_hash(text: str) -> int:
  h = 0
  for ch in text:
    h = (h + 37 * h + 11 * ord(ch)) % 0xFFFFFFFF
  return h


def _colorize(title: str, key: str) -> str:
  """Appends zero-width chars so XProf hashes `title` to the target palette slot."""
  target = _COLOR_SLOTS.get(key)
  if target is None:
    return title
  cache_key = (title, target)
  if cache_key in _COLOR_CACHE:
    return _COLOR_CACHE[cache_key]
  if _catapult_hash(title) % 23 == target:
    _COLOR_CACHE[cache_key] = title
    return title
  for length in range(1, 5):
    for combo in itertools.product(_ZERO_WIDTH_CHARS, repeat=length):
      candidate = title + "".join(combo)
      if _catapult_hash(candidate) % 23 == target:
        _COLOR_CACHE[cache_key] = candidate
        return candidate
  _COLOR_CACHE[cache_key] = title
  return title


class _TypedStat:
  """Typed stat wrapper for values whose proto field type must be forced."""

  def __init__(self, kind: str, value: Any):
    self.kind = kind
    self.value = value


def _u64(val: int) -> _TypedStat:
  return _TypedStat("uint64_value", int(val))


class _PlaneBuilder:
  """Incremental builder for one XPlane with deduplicated stat/event metadata."""

  def __init__(self, plane):
    self.plane = plane
    self._stat_md: dict[str, int] = {}
    self._event_md: dict[tuple[str, str | None], int] = {}
    self._lines: dict[int, Any] = {}

  def stat_id(self, name: str) -> int:
    """Returns a deduplicated XStatMetadata id for `name`."""
    sid = self._stat_md.get(name)
    if sid is None:
      sid = len(self._stat_md) + 1
      self._stat_md[name] = sid
      self.plane.stat_metadata[sid].id = sid
      self.plane.stat_metadata[sid].name = name
    return sid

  def _fill_stat(self, stat, name: str, value: Any):
    """Populates one XStat message with the appropriate typed value field."""
    stat.metadata_id = self.stat_id(name)
    if isinstance(value, _TypedStat):
      setattr(stat, value.kind, value.value)
    elif isinstance(value, bool):
      stat.int64_value = int(value)
    elif isinstance(value, int):
      if value > 0x7FFFFFFFFFFFFFFF:
        stat.uint64_value = value
      else:
        stat.int64_value = value
    elif isinstance(value, float):
      stat.double_value = value
    elif isinstance(value, bytes):
      stat.bytes_value = value
    else:
      stat.str_value = str(value)

  def add_stats(self, container, stats: dict[str, Any] | None):
    """Appends non-None key/value stats onto `container`."""
    for key, val in (stats or {}).items():
      if val is None:
        continue
      self._fill_stat(container.stats.add(), key, val)

  def event_md(self, name: str, display_name: str | None = None, stats: dict[str, Any] | None = None) -> int:
    """Returns a deduplicated XEventMetadata id for `(name, display_name)`."""
    key = (name, display_name)
    mid = self._event_md.get(key)
    if mid is None:
      mid = len(self._event_md) + 1
      self._event_md[key] = mid
      md = self.plane.event_metadata[mid]
      md.id = mid
      md.name = name
      if display_name:
        md.display_name = display_name
      self.add_stats(md, stats)
    return mid

  def line(self, lid: int, name: str, display_name: str | None = None):
    """Gets or creates an XLine on this plane with `timestamp_ns=0`."""
    ln = self._lines.get(lid)
    if ln is None:
      ln = self.plane.lines.add()
      ln.id = lid
      ln.display_id = lid
      ln.name = name
      if display_name:
        ln.display_name = display_name
      ln.timestamp_ns = 0
      self._lines[lid] = ln
    return ln

  def event(self, line, md_id: int, offset_ps: int, dur_ps: int, stats: dict[str, Any] | None = None):
    if offset_ps <= 0:
      raise ValueError("offset_ps must be > 0 (XProf drops events at offset 0)")
    ev = line.events.add()
    ev.metadata_id = md_id
    ev.offset_ps = int(offset_ps)
    ev.duration_ps = max(int(dur_ps), 1)
    self.add_stats(ev, stats)
    return ev


# ---------------------------------------------------------------------------
# Pluggable schedule registry & oracle
# ---------------------------------------------------------------------------

_NEVER_MATCH_RE = re.compile(r"(?!)")
_GENERIC_SUBTICK_RE = re.compile(r"subtick_([01])_([fbw])")


@dataclasses.dataclass
class PipelineSchedule:
  """Schedule descriptor returned by a builder registered via ``register_pipeline_schedule``.

  Two execution models are supported:

  1. ``mode == "fwd_bwd"`` (e.g. ``gpipe``, ``circular``):
     One forward pipeline loop (or unrolled sequence) of ``n_iter`` iterations
     followed by one backward loop of ``n_iter`` iterations.
     Supply ``n_iter`` and ``fwd_slot(iteration, stage) -> (mb, repeat, valid)``.
     ``bwd_slot`` defaults to reversing ``fwd_slot(n_iter - 1 - iteration, stage)``.

  2. ``mode == "phased"`` (e.g. multi-phase interleaved F/B/W schedules):
     Multiple named phase loops, each executing macros of 2 sub-ticks with
     independent F/B/W/Tail passes. Supply ``num_ticks``, ``num_macros``,
     ``phases``, ``pass_slot(tick, stage, kind) -> (mb, chunk, valid)``, and
     optionally ``scheduled_fn``, ``b_mode_fn``, ``w_uniformly_invalid_fn``,
     ``phase_regex``, ``subtick_regex``, ``tail_scope``.
  """

  name: str
  mode: str  # "fwd_bwd" | "phased"
  num_stages: int
  num_microbatches: int
  num_repeats: int = 1
  forwarding_delay: int = 1
  n_iter: int = 0
  num_ticks: int = 0
  num_macros: int = 0
  fwd_slot: Callable[[int, int], tuple[int, int, bool]] | None = None
  bwd_slot: Callable[[int, int], tuple[int, int, bool]] | None = None
  phases: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  pass_slot: Callable[[int, int, str], tuple[int, int, bool]] | None = None
  scheduled_fn: Callable[[int, str], bool] | None = None
  b_mode_fn: Callable[[int], str] | None = None
  w_uniformly_invalid_fn: Callable[[int], bool] | None = None
  layer_index_fn: Callable[[int, int], int] | None = None
  phase_regex: re.Pattern[str] | None = None
  subtick_regex: re.Pattern[str] | None = None
  tail_scope: str = ""


ScheduleBuilder = Callable[..., PipelineSchedule]
_SCHEDULE_REGISTRY: dict[str, ScheduleBuilder] = {}
_SCHEDULE_ALIASES: dict[str, str] = {}


def _normalize_schedule_name(name: str) -> str:
  return str(name or "").strip().lower()


def register_pipeline_schedule(
    name: str,
    builder_fn: ScheduleBuilder,
    *,
    aliases: Sequence[str] = (),
) -> None:
  """Registers a pipeline schedule builder for XProf stage trace decoding.

  Args:
    name: Canonical schedule name (matched case-insensitively against the
      sidecar's ``pipeline.schedule`` field).
    builder_fn: Callable ``(num_stages, num_microbatches, num_repeats=1,
      forwarding_delay=1) -> PipelineSchedule``.
    aliases: Optional sequence of alternative short names that resolve to
      ``name`` (for example ``("circ",)`` for ``"circular"``).
  """
  canonical = _normalize_schedule_name(name)
  if not canonical:
    raise ValueError("Schedule name must be non-empty.")
  _SCHEDULE_REGISTRY[canonical] = builder_fn
  for alias in aliases:
    norm_alias = _normalize_schedule_name(alias)
    if norm_alias:
      _SCHEDULE_ALIASES[norm_alias] = canonical


def unregister_pipeline_schedule(name: str) -> None:
  """Removes a registered pipeline schedule (and its aliases). Intended for tests."""
  canonical = _SCHEDULE_ALIASES.get(_normalize_schedule_name(name), _normalize_schedule_name(name))
  _SCHEDULE_REGISTRY.pop(canonical, None)
  for alias in [k for k, v in _SCHEDULE_ALIASES.items() if v == canonical]:
    _SCHEDULE_ALIASES.pop(alias, None)


def get_registered_pipeline_schedules() -> tuple[str, ...]:
  """Returns the sorted canonical names of all registered pipeline schedules."""
  return tuple(sorted(_SCHEDULE_REGISTRY))


def _resolve_schedule_name(name: str) -> str:
  norm = _normalize_schedule_name(name)
  return _SCHEDULE_ALIASES.get(norm, norm)


def _build_gpipe_or_circular_schedule(
    schedule_name: str,
    num_stages: int,
    num_microbatches: int,
    num_repeats: int = 1,
    forwarding_delay: int = 1,
) -> PipelineSchedule:
  """Built-in GPipe (`num_repeats == 1`) and Circular (`num_repeats > 1`) schedule builder."""
  s = int(num_stages)
  m = int(num_microbatches)
  r = int(num_repeats)
  d = int(forwarding_delay)
  n_iter = m * r + d * (s - 1)
  total_slots = m * r

  def _fwd_slot(iteration: int, stage: int) -> tuple[int, int, bool]:
    k = iteration - d * stage
    if 0 <= k < total_slots:
      return k % m, k // m, True
    if k < 0:
      return 0, 0, False
    return k % m, min(k // m, max(0, r - 1)), False

  def _bwd_slot(iteration: int, stage: int) -> tuple[int, int, bool]:
    return _fwd_slot(n_iter - 1 - iteration, stage)

  return PipelineSchedule(
      name=schedule_name,
      mode="fwd_bwd",
      num_stages=s,
      num_microbatches=m,
      num_repeats=r,
      forwarding_delay=d,
      n_iter=n_iter,
      num_ticks=2 * n_iter,
      fwd_slot=_fwd_slot,
      bwd_slot=_bwd_slot,
      layer_index_fn=lambda stage, repeat: repeat * s + stage,
  )


def _build_gpipe_schedule(
    num_stages: int, num_microbatches: int, num_repeats: int = 1, forwarding_delay: int = 1
) -> PipelineSchedule:
  return _build_gpipe_or_circular_schedule("gpipe", num_stages, num_microbatches, num_repeats, forwarding_delay)


def _build_circular_schedule(
    num_stages: int, num_microbatches: int, num_repeats: int = 1, forwarding_delay: int = 1
) -> PipelineSchedule:
  return _build_gpipe_or_circular_schedule("circular", num_stages, num_microbatches, num_repeats, forwarding_delay)


register_pipeline_schedule("gpipe", _build_gpipe_schedule)
register_pipeline_schedule("circular", _build_circular_schedule, aliases=("circ",))


class PipelineOracle:
  """Maps (stage, loop iteration or sub-tick, pass) to (microbatch, chunk/repeat, valid).

  Uses the schedule registry populated by ``register_pipeline_schedule``. If a
  schedule name is not registered, enters graceful fallback mode
  (``is_fallback == True``) rather than crashing or mislabeling microbatches.
  """

  def __init__(
      self, schedule: str, num_stages: int, num_microbatches: int, num_repeats: int = 1, forwarding_delay: int = 1
  ):
    canonical = _resolve_schedule_name(schedule)
    self.schedule = canonical or str(schedule or "unknown")
    self.num_stages = int(num_stages)
    self.num_microbatches = int(num_microbatches)
    self.num_repeats = int(num_repeats)
    self.forwarding_delay = int(forwarding_delay)
    self.spec_obj: PipelineSchedule | None = None
    self.fallback_reason: str | None = None

    builder = _SCHEDULE_REGISTRY.get(canonical)
    if builder is None:
      self.mode = "fallback"
      self.is_phased = False
      self.is_fallback = True
      self.n_iter = 0
      self.num_ticks = 0
      self.num_macros = 0
      self.phases: list[dict[str, Any]] = []
      self.fallback_reason = (
          f"unregistered pipeline schedule '{self.schedule}' "
          f"(registered: {', '.join(get_registered_pipeline_schedules()) or 'none'})"
      )
      return

    spec = builder(
        num_stages=self.num_stages,
        num_microbatches=self.num_microbatches,
        num_repeats=self.num_repeats,
        forwarding_delay=self.forwarding_delay,
    )
    self.spec_obj = spec
    self.schedule = spec.name or canonical
    self.mode = spec.mode
    self.is_phased = spec.mode == "phased"
    self.is_fallback = False
    self.num_repeats = int(spec.num_repeats)
    self.n_iter = int(spec.n_iter)
    self.num_ticks = int(spec.num_ticks) or (2 * self.n_iter)
    self.num_macros = int(spec.num_macros)
    self.phases = list(spec.phases)

  def fwd_slot(self, iteration: int, stage: int) -> tuple[int, int, bool]:
    """Forward iteration `iteration` on `stage` -> `(mb, repeat, valid)`."""
    assert self.spec_obj is not None and self.spec_obj.fwd_slot is not None
    return self.spec_obj.fwd_slot(iteration, stage)

  def bwd_slot(self, iteration: int, stage: int) -> tuple[int, int, bool]:
    """Backward iteration `iteration` on `stage` -> `(mb, repeat, valid)`."""
    assert self.spec_obj is not None
    if self.spec_obj.bwd_slot is not None:
      return self.spec_obj.bwd_slot(iteration, stage)
    return self.fwd_slot(self.n_iter - 1 - iteration, stage)

  def spec(self, tick: int):
    macro = tick // 2
    for ph in self.phases:
      if ph["start_macro"] <= macro < ph["start_macro"] + ph["length"]:
        return ph, (ph["a"] if tick % 2 == 0 else ph["b"])
    return None, None

  def scheduled(self, tick: int, kind: str) -> bool:
    if self.spec_obj is not None and self.spec_obj.scheduled_fn is not None:
      return bool(self.spec_obj.scheduled_fn(tick, kind))
    _, sp = self.spec(tick)
    if sp is None:
      return False
    return {
        "F": sp.has_f,
        "B": sp.b_mode != "none",
        "W": sp.w_pop or sp.b_mode in ("coupled", "fused"),
        "Tail": sp.has_tail,
    }[kind]

  def pass_slot(self, tick: int, stage: int, kind: str) -> tuple[int, int, bool]:
    """Returns `(mb, chunk, valid)` for a phased-schedule pass at `(tick, stage)`."""
    assert self.spec_obj is not None and self.spec_obj.pass_slot is not None
    return self.spec_obj.pass_slot(tick, stage, kind)

  def b_mode(self, tick: int) -> str:
    if self.spec_obj is not None and self.spec_obj.b_mode_fn is not None:
      return str(self.spec_obj.b_mode_fn(tick))
    _, sp = self.spec(tick)
    return sp.b_mode if sp is not None else "none"

  def w_uniformly_invalid(self, tick: int) -> bool:
    if not self.scheduled(tick, "W"):
      return False
    if self.spec_obj is not None and self.spec_obj.w_uniformly_invalid_fn is not None:
      return bool(self.spec_obj.w_uniformly_invalid_fn(tick))
    return all(not self.pass_slot(tick, st, "W")[2] for st in range(self.num_stages))

  def layer_index(self, stage: int, repeat_or_chunk: int) -> int:
    """Returns the monotonic layer index used to link cross-stage microbatch flow arrows."""
    if self.spec_obj is not None and self.spec_obj.layer_index_fn is not None:
      return int(self.spec_obj.layer_index_fn(stage, repeat_or_chunk))
    if self.is_phased:
      return stage if repeat_or_chunk == 0 else (2 * self.num_stages - 1 - stage)
    return repeat_or_chunk * self.num_stages + stage

  def phase_regex(self) -> re.Pattern[str]:
    if self.spec_obj is not None and self.spec_obj.phase_regex is not None:
      return self.spec_obj.phase_regex
    scopes = [re.escape(str(ph["scope"])) for ph in self.phases if ph.get("scope")]
    if scopes:
      scopes = sorted(set(scopes), key=len, reverse=True)
      return re.compile(r"(?<![a-z0-9_])(" + "|".join(scopes) + r")(?![a-z0-9_])")
    return _NEVER_MATCH_RE

  def subtick_regex(self) -> re.Pattern[str]:
    if self.spec_obj is not None and self.spec_obj.subtick_regex is not None:
      return self.spec_obj.subtick_regex
    return _GENERIC_SUBTICK_RE

  def tail_scope(self) -> str:
    if self.spec_obj is not None and self.spec_obj.tail_scope:
      return self.spec_obj.tail_scope
    return ""


# ---------------------------------------------------------------------------
# XPlane reader
# ---------------------------------------------------------------------------

_TPU_PLANE_RE = re.compile(r"^/device:TPU:(\d+)$")
_MESH_RG_RE = re.compile(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}")
_IOTA_RG_RE = re.compile(r"replica_groups=\[(\d+),(\d+)\]<=\[([\d,]+)\]")
_STP_RE = re.compile(r"source_target_pairs=\{((?:\{\d+,\d+\},?)*)\}")
_LIST_RG_RE = re.compile(r"replica_groups=\{(\{[\d,]*\}(?:,\{[\d,]*\})*)\}")


def _is_global_replica_group(full_name: str) -> bool:
  """Returns True iff a collective's replica_groups span all devices in one group."""
  m = _MESH_RG_RE.search(full_name)
  if m:
    sizes = {}
    for kv in m.group(1).split(","):
      if "=" in kv:
        k, v = kv.split("=")
        sizes[k.strip().strip("'\"")] = int(v)
    total = math.prod(sizes.values()) if sizes else 0
    reduced = [a.strip().strip("'\"") for a in m.group(2).split(",") if a.strip()]
    return 1 < total == math.prod(sizes.get(a, 1) for a in reduced)
  m = _IOTA_RG_RE.search(full_name)
  if m:
    return int(m.group(1)) == 1 and int(m.group(2)) == math.prod(int(x) for x in m.group(3).split(","))
  m = _LIST_RG_RE.search(full_name)
  if m:
    groups = re.findall(r"\{([\d,]*)\}", m.group(1))
    return len(groups) == 1 and len(groups[0].split(",")) > 1
  return False


def _extract_stats(container, stat_md) -> dict[str, Any]:
  """Decodes scalar stats from an XPlane/XEvent/XEventMetadata container."""
  out: dict[str, Any] = {}
  for stat in container.stats:
    md = stat_md.get(stat.metadata_id)
    if md is None:
      continue
    kind = stat.WhichOneof("value")
    if kind is None or kind == "bytes_value":
      continue
    val = getattr(stat, kind)
    if kind == "ref_value":
      ref = stat_md.get(val)
      val = ref.name if ref is not None else val
    out[md.name] = val
  return out


class _OpInfo:
  """Metadata summary for one deduplicated XLA Op entry."""

  __slots__ = ("name", "tf", "cat", "src", "is_global", "stp")

  def __init__(self, name: str, tf: str, cat: str, src: str, is_global: bool):
    self.name = name
    self.tf = tf
    self.cat = cat
    self.src = src
    self.is_global = is_global
    self.stp: list[tuple[int, ...]] | None = None


class _DevicePlane:
  """Decoded TensorCore plane with step module windows and flat op arrays (absolute ps)."""

  def __init__(self, plane_name: str, ordinal: int, pstats: dict[str, Any]):
    self.plane = plane_name
    self.ordinal = ordinal
    self.device_id_stat = pstats.get("device_id")
    self.global_chip_id = pstats.get("global_chip_id")
    self.device_type = pstats.get("device_type_string")
    self.has_megacore = pstats.get("has_megacore")
    self.modules: list[tuple[int, int]] = []
    self.start: list[int] = []
    self.dur: list[int] = []
    self.mid: list[int] = []
    self.info: dict[int, _OpInfo] = {}


class _HostTrace:
  """Decoded per-host trace summary."""

  def __init__(self, path: str):
    self.path = path
    self.hostname = ""
    self.profile_start_ns = 0
    self.step_nums: list[int] = []
    self.devices: list[_DevicePlane] = []


def _extract_cpu_step_nums(plane, base_ps: int) -> list[int]:
  """Extracts chronologically ordered `step_num` annotations from `/host:CPU` `train` events."""
  smd = plane.stat_metadata
  train_ids = {k for k, md in plane.event_metadata.items() if md.name == "train"}
  if not train_ids:
    return []
  rows = []
  for ln in plane.lines:
    line_base = base_ps + ln.timestamp_ns * 1000
    for ev in ln.events:
      if ev.metadata_id not in train_ids:
        continue
      st = _extract_stats(ev, smd)
      if "step_num" in st:
        rows.append((line_base + ev.offset_ps, int(st["step_num"])))
  return [s for _, s in sorted(rows)]


def _parse_host_trace(path: str, space, module_prefix: str = "jit_train_step") -> _HostTrace:
  """Extracts TPU planes, step windows, and XLA Ops from an already-parsed XSpace."""
  host_trace = _HostTrace(path)
  host_trace.hostname = space.hostnames[0] if len(space.hostnames) else os.path.basename(path).split(".xplane.pb")[0]
  for plane in space.planes:
    if plane.name == "Task Environment":
      stats = _extract_stats(plane, plane.stat_metadata)
      host_trace.profile_start_ns = int(stats.get("profile_start_time", 0) or 0)
  base_ps = host_trace.profile_start_ns * 1000
  for plane in space.planes:
    if plane.name == "/host:CPU":
      host_trace.step_nums = _extract_cpu_step_nums(plane, base_ps)
  for plane in space.planes:
    match = _TPU_PLANE_RE.match(plane.name)
    if not match:
      continue
    smd, emd = plane.stat_metadata, plane.event_metadata
    dp = _DevicePlane(plane.name, int(match.group(1)), _extract_stats(plane, smd))
    lines = {ln.name: ln for ln in plane.lines}
    mod_line = lines.get("XLA Modules")
    if mod_line is not None:
      line_base = base_ps + mod_line.timestamp_ns * 1000
      for ev in mod_line.events:
        md = emd.get(ev.metadata_id)
        name = md.name if md is not None else ""
        if name.startswith(module_prefix):
          dp.modules.append((line_base + ev.offset_ps, line_base + ev.offset_ps + ev.duration_ps))
      dp.modules.sort()
    ops_line = lines.get("XLA Ops")
    if ops_line is not None:
      line_base = base_ps + ops_line.timestamp_ns * 1000
      st_list, du_list, mi_list = dp.start, dp.dur, dp.mid
      for ev in ops_line.events:
        st_list.append(line_base + ev.offset_ps)
        du_list.append(ev.duration_ps)
        mi_list.append(ev.metadata_id)
      for mid in set(mi_list):
        md = emd.get(mid)
        if md is None:
          dp.info[mid] = _OpInfo(str(mid), "", "", "", False)
          continue
        st = _extract_stats(md, smd)
        name = md.display_name or md.name
        cat = str(st.get("hlo_category", "") or "")
        is_global = False
        low = name.lower()
        if ("all-reduce" in low or "all-reduce" in cat) and "replica_groups=" in md.name:
          is_global = _is_global_replica_group(md.name)
        oi = _OpInfo(
            name,
            str(st.get("tf_op", "") or ""),
            cat,
            str(st.get("source", "") or ""),
            is_global,
        )
        if name.startswith("collective-permute-start"):
          stp_match = _STP_RE.search(md.name)
          if stp_match:
            oi.stp = [tuple(int(x) for x in pr.split(",")) for pr in re.findall(r"\{(\d+,\d+)\}", stp_match.group(1))]
        dp.info[mid] = oi
    host_trace.devices.append(dp)
  return host_trace


def _load_host_trace(path: str, module_prefix: str = "jit_train_step") -> _HostTrace:
  raw = epath.Path(path).read_bytes()
  space = XSpace.FromString(raw)
  return _parse_host_trace(path, space, module_prefix=module_prefix)


# ---------------------------------------------------------------------------
# Per-device-step decoder
# ---------------------------------------------------------------------------

_MATMUL_CATS = ("convolution fusion", "convolution", "custom-call", "custom fusion")


def _comm_kind(info: _OpInfo) -> tuple[str | None, bool]:
  """Returns (family, is_wait) where family is pp | pp_other | moe | fsdp, or (None, False)."""
  name = info.name.lower()
  cat = info.cat.lower()
  base = name.split(".")[0]
  is_cp = cat.startswith("collective-permute") or base.startswith("collective-permute")
  fam = None
  if is_cp:
    fam = "pp" if ("ppermute" in info.tf or "pipeline" in info.src) else "pp_other"
  elif base.startswith(("ragged-all-to-all", "all-to-all")) or cat in ("all-to-all", "ragged-all-to-all"):
    fam = "moe"
  elif base.startswith(
      ("all-gather", "reduce-scatter", "all-reduce", "psum", "reduce_scatter", "all_gather", "all_reduce")
  ) or cat in ("all-reduce", "all-gather", "reduce-scatter", "all-reduce-scatter fusion"):
    fam = "fsdp"
  if fam is None:
    return None, False
  is_start = (
      cat.endswith(("-start", "-update"))
      or base.endswith(("-start", "-update"))
      or "call-start" in name
      or "call-update" in name
      or cat in ("async-start", "async-update")
  )
  return fam, not is_start


class _Unit:
  """One decoded pipeline loop iteration or phased-schedule macro."""

  __slots__ = ("phase", "kind", "iter", "macro", "t0", "t1", "acc", "slots", "loop_dir")

  def __init__(self, phase: str, kind: str, iteration: int, macro: int, t0: int, t1: int, loop_dir: str | None = None):
    self.phase = phase
    self.kind = kind
    self.iter = iteration
    self.macro = macro
    self.t0 = t0
    self.t1 = t1
    self.acc: collections.Counter[str] = collections.Counter()
    self.slots: list[_Slot] = []
    self.loop_dir = loop_dir


class _Slot:
  """One scheduled forward, backward, weight-grad, tail, or raw fallback slot inside a unit."""

  __slots__ = (
      "pass_",
      "tick",
      "subtick",
      "mb",
      "chunk",
      "valid",
      "compute",
      "span",
      "kind",
      "uniform_invalid",
      "waits",
  )

  def __init__(self, pass_: str, tick: int, subtick: int, mb: int, chunk: int, valid: bool):
    self.pass_ = pass_
    self.tick = tick
    self.subtick = subtick
    self.mb = mb
    self.chunk = chunk
    self.valid = valid
    self.compute = 0
    self.span: tuple[int, int] | None = None
    self.kind = "raw_iter" if pass_ == "Iter" else ("useful" if valid else "dummy")
    self.uniform_invalid = False
    self.waits: collections.Counter[str] = collections.Counter()


class _DeviceStep:
  """Decoded pipeline units, anchors, and gaps for one device and one training step."""

  def __init__(self, stage: int, step: int, win: tuple[int, int]):
    self.stage = stage
    self.step = step
    self.win = win
    self.units: list[_Unit] = []
    self.gaps: list[tuple[int, int, str, collections.Counter[str]]] = []
    self.fallback_reason: str | None = None
    self.anchors: dict[tuple[str, int], int] = {}
    self.pp_instances: dict[tuple[str, int], tuple[int, int]] = {}


def _lower_bound(arr: list[int], target: int) -> int:
  lo, hi = 0, len(arr)
  while lo < hi:
    mid = (lo + hi) // 2
    if arr[mid] < target:
      lo = mid + 1
    else:
      hi = mid
  return lo


def _build_op_tree(starts: list[int], durs: list[int], mids: list[int], info: dict[int, _OpInfo]):
  """Builds a containment tree of XLA Ops and computes self-time and enclosing while-loops."""
  order = sorted(range(len(starts)), key=lambda i: (starts[i], -durs[i]))
  n = len(order)
  start = [starts[i] for i in order]
  dur = [durs[i] for i in order]
  mid = [mids[i] for i in order]
  depth = [0] * n
  child_dur = [0] * n
  enclosing = [-1] * n
  is_while = [info[m].name.startswith("while") for m in mid]
  stack: list[int] = []
  for k in range(n):
    s = start[k]
    while stack and start[stack[-1]] + dur[stack[-1]] <= s:
      stack.pop()
    depth[k] = len(stack)
    if stack:
      parent = stack[-1]
      child_dur[parent] += dur[k]
      enclosing[k] = parent if is_while[parent] else enclosing[parent]
    stack.append(k)
  self_time = [max(0, dur[k] - child_dur[k]) for k in range(n)]
  return start, dur, mid, depth, self_time, enclosing, is_while


def _acc_comm(unit: _Unit, inf: _OpInfo, self_ps: int) -> bool:
  fam, is_wait = _comm_kind(inf)
  if fam is None:
    return False
  if is_wait:
    unit.acc[("pp" if fam.startswith("pp") else fam) + "_wait"] += self_ps
  else:
    unit.acc["bookkeeping"] += self_ps
  return True


def _account_gpipe(unit: _Unit, slot: _Slot, desc: list[int], t0: int, t1: int, start, selft, infos):
  """Attributes self-time inside one GPipe/Circular/fallback iteration to compute, waits, and bookkeeping."""
  busy = 0
  for k in desc:
    s0 = start[k]
    if s0 < t0:
      continue
    if s0 >= t1:
      break
    inf = infos[k]
    s_ps = selft[k]
    busy += s_ps
    if _acc_comm(unit, inf, s_ps):
      continue
    tf = inf.tf
    nm = inf.name
    body = tf.split("while/body/", 1)[1] if "while/body/" in tf else ""
    if "pp_stage_compute" in tf:
      slot.compute += s_ps
    elif "pp_shift" in tf or "pp_bubble_select" in tf:
      unit.acc["bookkeeping"] += s_ps
    elif body and "vmap(" in body:
      slot.compute += s_ps
    elif inf.cat in _MATMUL_CATS or (inf.cat == "loop fusion" and tf.rstrip(":").endswith("/while")):
      slot.compute += s_ps
    elif "pipeline_module" in tf:
      unit.acc["bookkeeping"] += s_ps
    elif not tf:
      if inf.cat.startswith(("async", "copy", "data formatting")) or nm.startswith(("copy", "slice-start", "slice-done")):
        unit.acc["bookkeeping"] += s_ps
      else:
        slot.compute += s_ps
  unit.acc["idle"] = max(0, (t1 - t0) - busy)
  slot.span = (t0, t1)
  slot.waits = collections.Counter({k: v for k, v in unit.acc.items() if k.endswith("_wait")})


def _account_phased(unit: _Unit, ks: list[int], macro: int, oracle: PipelineOracle, stage: int, start, dur, selft, infos):
  """Attributes self-time inside one phased-schedule macro across sub-ticks and F/B/W/Tail slots."""
  t0, t1 = unit.t0, unit.t1
  subtick_re = oracle.subtick_regex()
  tail_scope = oracle.tail_scope()
  busy = 0
  per: collections.Counter[tuple[str, int]] = collections.Counter()
  spans: dict[tuple[str, int], tuple[int, int]] = {}
  pending: list[tuple[int, int]] = []
  tagged_seq: list[tuple[int, tuple[str, int]]] = []
  slot_wait: dict[tuple[str, int], collections.Counter[str]] = collections.defaultdict(collections.Counter)
  for k in ks:
    inf = infos[k]
    s_ps = selft[k]
    s0 = start[k]
    busy += s_ps
    tf = inf.tf
    match = subtick_re.search(tf)
    nm = inf.name
    fam, is_wait = _comm_kind(inf)
    if fam is not None and is_wait and match:
      wkey = ({"f": "F", "b": "B", "w": "W"}[match.group(2)], int(match.group(1)))
      slot_wait[wkey][("pp" if fam.startswith("pp") else fam) + "_wait"] += s_ps
    if _acc_comm(unit, inf, s_ps):
      continue
    if match:
      key = ({"f": "F", "b": "B", "w": "W"}[match.group(2)], int(match.group(1)))
    elif tail_scope and tail_scope in tf:
      key = ("Tail", 1)
    else:
      key = None
    if key is not None:
      per[key] += s_ps
      sp = spans.get(key)
      spans[key] = (s0, s0 + dur[k]) if sp is None else (min(sp[0], s0), max(sp[1], s0 + dur[k]))
      tagged_seq.append((s0, key))
      continue
    is_mm = inf.cat in _MATMUL_CATS
    collapsed = (not tf) or tf.rstrip(":").endswith(("/while", "pipeline_module"))
    if is_mm and collapsed:
      pending.append((k, s_ps))
    elif "pipeline_module" in tf:
      unit.acc["bookkeeping"] += s_ps
    elif not tf and not (
        inf.cat.startswith(("async", "copy", "data formatting")) or nm.startswith(("copy", "slice-start", "slice-done"))
    ):
      pending.append((k, s_ps))
    elif not tf:
      unit.acc["bookkeeping"] += s_ps
  if pending and tagged_seq:
    ts = [x[0] for x in tagged_seq]
    for k, s_ps in pending:
      idx = _lower_bound(ts, start[k])
      cands = []
      if idx > 0:
        cands.append((start[k] - ts[idx - 1], tagged_seq[idx - 1][1]))
      if idx < len(ts):
        cands.append((ts[idx] - start[k], tagged_seq[idx][1]))
      per[min(cands)[1]] += s_ps
  unit.acc["idle"] = max(0, (t1 - t0) - busy)
  for subtick in (0, 1):
    tick = 2 * macro + subtick
    if tick >= oracle.num_ticks:
      continue
    bmode = oracle.b_mode(tick)
    for kind in ("F", "B", "W", "Tail"):
      if not oracle.scheduled(tick, kind):
        if per.get((kind, subtick)):
          unit.acc["bookkeeping"] += per[(kind, subtick)]
        continue
      if kind == "W" and bmode in ("coupled", "fused"):
        continue
      mb, chunk, valid = oracle.pass_slot(tick, stage, kind)
      if kind == "W" and oracle.w_uniformly_invalid(tick):
        sl = _Slot("W", tick, subtick, -1, -1, False)
        sl.uniform_invalid = True
        sl.compute = per.get(("W", subtick), 0)
        sl.span = spans.get(("W", subtick))
        sl.waits = slot_wait.get(("W", subtick), collections.Counter())
        unit.slots.append(sl)
        continue
      name = kind
      comp = per.get((kind, subtick), 0)
      sp = spans.get((kind, subtick))
      if kind == "B" and bmode in ("coupled", "fused"):
        name = "BW"
        comp += per.get(("W", subtick), 0)
        sw = spans.get(("W", subtick))
        if sw is not None:
          sp = sw if sp is None else (min(sp[0], sw[0]), max(sp[1], sw[1]))
      sl = _Slot(name, tick, subtick, mb, chunk, valid)
      sl.compute = comp
      sl.span = sp
      sl.waits = collections.Counter(slot_wait.get((kind, subtick), {}))
      if name == "BW":
        sl.waits.update(slot_wait.get(("W", subtick), {}))
      unit.slots.append(sl)


def _append_unrolled_phase(ph, pre, start, dur, infos, subtick_re, macro_units) -> int:
  """Splits straight-line unrolled phased ops into per-macro intervals at subtick 1->0 boundaries."""
  bounds = [start[pre[0]]]
  last_p = None
  for k in pre:
    match = subtick_re.search(infos[k].tf)
    if not match:
      continue
    subtick = int(match.group(1))
    if last_p == 1 and subtick == 0:
      bounds.append(start[k])
    last_p = subtick
  e_end = max(start[k] + dur[k] for k in pre)
  if len(bounds) == ph["length"]:
    bounds.append(e_end)
    for it in range(ph["length"]):
      macro_units.append((ph, ph["start_macro"] + it, bounds[it], bounds[it + 1], "unrolled"))
  else:
    macro_units.append((ph, ph["start_macro"], start[pre[0]], e_end, "peel_only"))
  return e_end


def _segment_unrolled_fwd_bwd_direction(
    ks: list[int], start: list[int], dur: list[int], infos: list[_OpInfo], direction: str
) -> list[tuple[int, int]]:
  """Segments unrolled (`scan_pipeline_iterations=False`) pipeline ops into per-iteration spans."""
  if not ks:
    return []
  # Forward order inside one iteration: bubble -> compute -> shift.
  # Backward (transpose) order: shift -> compute -> bubble.
  order = {"bubble": 0, "compute": 1, "shift": 2} if direction == "fwd" else {"shift": 0, "compute": 1, "bubble": 2}
  bounds: list[int] = []
  last_rank = -1
  for k in ks:
    tf = infos[k].tf
    if "pp_bubble_select" in tf:
      tag = "bubble"
    elif "pp_shift" in tf:
      tag = "shift"
    elif "pp_stage_compute" in tf:
      tag = "compute"
    else:
      continue
    rank = order[tag]
    if not bounds:
      bounds.append(start[k])
    elif rank < last_rank:
      bounds.append(start[k])
    last_rank = rank
  if not bounds:
    return []
  e_end = max(start[k] + dur[k] for k in ks)
  bounds.append(e_end)
  return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def _decode_fallback_units(
    a: int,
    b: int,
    whiles0: list[int],
    bodies: dict[int, list[int]],
    segment_fn,
    descendants_fn,
    start: list[int],
    selft: list[int],
    infos: list[_OpInfo],
) -> list[_Unit]:
  """Emits raw `iter 0 .. T-1` units without guessing microbatch labels."""
  pipe_whiles = []
  for w in whiles0:
    kids = bodies.get(w, [])
    if any(("pipeline" in infos[k].tf or "pp_stage_compute" in infos[k].tf or "ppermute" in infos[k].tf) for k in kids):
      pipe_whiles.append((w, kids))
  if not pipe_whiles:
    pipe_whiles = [(w, bodies.get(w, [])) for w in whiles0 if bodies.get(w)]

  units: list[_Unit] = []
  raw_iter = 0
  for loop_idx, (w, kids) in enumerate(pipe_whiles):
    _, iters = segment_fn(w, kids, None)
    desc = descendants_fn(w)
    for t0, t1 in iters:
      u = _Unit(f"loop_{loop_idx}", "fallback_loop", raw_iter, raw_iter, t0, t1, "fallback")
      sl = _Slot("Iter", raw_iter, 0, raw_iter, 0, True)
      u.slots.append(sl)
      _account_gpipe(u, sl, desc, t0, t1, start, selft, infos)
      units.append(u)
      raw_iter += 1

  if not units and b > a:
    u = _Unit("step_body", "fallback_span", 0, 0, a, b, "fallback")
    sl = _Slot("Iter", 0, 0, 0, 0, True)
    u.slots.append(sl)
    all_ks = list(range(len(start)))
    _account_gpipe(u, sl, all_ks, a, b, start, selft, infos)
    units.append(u)
  return units


def _build_fwd_bwd_units(
    iters: list[tuple[int, int]],
    direction: str,
    kind: str,
    desc: list[int],
    oracle: PipelineOracle,
    stage: int,
    start: list[int],
    selft: list[int],
    infos: list[_OpInfo],
) -> list[_Unit]:
  """Builds `_Unit` entries for one forward or backward GPipe/Circular pass."""
  out: list[_Unit] = []
  for it, (t0, t1) in enumerate(iters):
    if direction == "fwd":
      mb, rep, ok = oracle.fwd_slot(it, stage)
      tick = it
    else:
      mb, rep, ok = oracle.bwd_slot(it, stage)
      tick = oracle.n_iter + it
    u = _Unit("pipeline_" + direction, kind, it, it, t0, t1, direction)
    sl = _Slot("F" if direction == "fwd" else "B", tick, 0, mb, rep, ok)
    u.slots.append(sl)
    _account_gpipe(u, sl, desc, t0, t1, start, selft, infos)
    out.append(u)
  return out


def _decode_fwd_bwd_units(
    a: int,
    b: int,
    n: int,
    whiles0: list[int],
    bodies: dict[int, list[int]],
    segment_fn,
    descendants_fn,
    oracle: PipelineOracle,
    stage: int,
    start: list[int],
    dur: list[int],
    selft: list[int],
    infos: list[_OpInfo],
    ds: _DeviceStep,
) -> list[_Unit]:
  """Decodes a `fwd_bwd` schedule (scanned `while` loops or unrolled `scan_pipeline_iterations=False`)."""
  loops = []
  for w in whiles0:
    kids = bodies.get(w, [])
    if not any(("pipeline_module/while/body" in infos[k].tf or "pp_stage_compute" in infos[k].tf) for k in kids):
      continue
    tagged = [k for k in kids if ("pipeline_module/while" in infos[k].tf or "pp_stage_compute" in infos[k].tf)]
    ntr = sum(1 for k in tagged if "transpose(" in infos[k].tf)
    loops.append((w, kids, "bwd" if ntr > len(tagged) / 2 else "fwd"))

  fwd_loops = [lp for lp in loops if lp[2] == "fwd"]
  bwd_loops = [lp for lp in loops if lp[2] == "bwd"]
  if len(fwd_loops) == 1 and len(bwd_loops) == 1:
    loop_trips = []
    for w, kids, direction in loops:
      trip, iters = segment_fn(w, kids, oracle.n_iter)
      loop_trips.append((w, direction, trip, iters))
    if all(trip == oracle.n_iter for _, _, trip, _ in loop_trips):
      units: list[_Unit] = []
      for w, direction, _, iters in loop_trips:
        units.extend(
            _build_fwd_bwd_units(iters, direction, "loop", descendants_fn(w), oracle, stage, start, selft, infos)
        )
      return units
    mismatch = ", ".join(f"{d}={tr}" for _, d, tr, _ in loop_trips)
    ds.fallback_reason = f"unexpected loop trip count ({mismatch}, expected {oracle.n_iter})"
    return _decode_fallback_units(a, b, whiles0, bodies, segment_fn, descendants_fn, start, selft, infos)

  if not loops:
    # Handle scan_pipeline_iterations=False (unrolled Python loop tagged with pp_bubble_select/pp_stage_compute/pp_shift).
    unrolled_ks = [
        k
        for k in range(n)
        if ("pp_stage_compute" in infos[k].tf or "pp_shift" in infos[k].tf or "pp_bubble_select" in infos[k].tf)
    ]
    fwd_ks = [k for k in unrolled_ks if "transpose(" not in infos[k].tf]
    bwd_ks = [k for k in unrolled_ks if "transpose(" in infos[k].tf]
    fwd_iters = _segment_unrolled_fwd_bwd_direction(fwd_ks, start, dur, infos, "fwd")
    bwd_iters = _segment_unrolled_fwd_bwd_direction(bwd_ks, start, dur, infos, "bwd")
    if len(fwd_iters) == oracle.n_iter and len(bwd_iters) == oracle.n_iter:
      all_ks = list(range(n))
      units = []
      for direction, iters in (("fwd", fwd_iters), ("bwd", bwd_iters)):
        units.extend(_build_fwd_bwd_units(iters, direction, "unrolled", all_ks, oracle, stage, start, selft, infos))
      return units
    ds.fallback_reason = (
        f"unexpected pipeline loop structure (0 while loops; unrolled fwd={len(fwd_iters)}, "
        f"bwd={len(bwd_iters)}, expected {oracle.n_iter})"
    )
    return _decode_fallback_units(a, b, whiles0, bodies, segment_fn, descendants_fn, start, selft, infos)

  ds.fallback_reason = f"expected 1 fwd + 1 bwd pipeline loop, found {[lp[2] for lp in loops]}"
  return _decode_fallback_units(a, b, whiles0, bodies, segment_fn, descendants_fn, start, selft, infos)


def _decode_device_step(
    dp: _DevicePlane, win: tuple[int, int], oracle: PipelineOracle, stage: int, step_label: int
) -> _DeviceStep:
  """Decodes one training step window on one TPU device plane into pipeline units and gaps."""
  a, b = win
  idx = [i for i, s in enumerate(dp.start) if a <= s < b]
  st = [dp.start[i] for i in idx]
  du = [dp.dur[i] for i in idx]
  mi = [dp.mid[i] for i in idx]
  info = dp.info
  start, dur, mid, depth, selft, encl, is_while = _build_op_tree(st, du, mi, info)
  n = len(start)
  ds = _DeviceStep(stage, step_label, win)
  infos = [info[m] for m in mid]

  occ: collections.Counter[str] = collections.Counter()
  cp_start: dict[tuple[str, int], int] = {}
  for k in range(n):
    inf = infos[k]
    if inf.is_global and not inf.cat.endswith("-start"):
      key = inf.name
      ds.anchors[(key, occ[key])] = start[k] + dur[k]
      occ[key] += 1
    nm = inf.name
    if nm.startswith("collective-permute-start"):
      suf = nm[len("collective-permute-start") :]
      o = occ["S" + suf]
      occ["S" + suf] += 1
      cp_start[(suf, o)] = start[k]
    elif nm.startswith("collective-permute-done"):
      suf = nm[len("collective-permute-done") :]
      o = occ["D" + suf]
      occ["D" + suf] += 1
      ds.pp_instances[(suf, o)] = (cp_start.get((suf, o), start[k]), start[k] + dur[k])

  bodies: dict[int, list[int]] = collections.defaultdict(list)
  for k in range(n):
    if encl[k] >= 0:
      bodies[encl[k]].append(k)
  whiles0 = [k for k in range(n) if depth[k] == 0 and is_while[k]]

  def end_ps(k: int) -> int:
    return start[k] + dur[k]

  def segment(w: int, kids: list[int], expected: int | None):
    cnt = collections.Counter(mid[k] for k in kids)
    if not cnt:
      return 0, []
    hist = collections.Counter(cnt.values())
    trip = hist.most_common(1)[0][0]
    if expected and trip != expected and hist.get(expected, 0) > 0:
      trip = expected
    first: dict[int, int] = {}
    for k in kids:
      first.setdefault(mid[k], start[k])
    anchor = min((m for m, c in cnt.items() if c == trip), key=lambda m: first[m])
    starts = [start[k] for k in kids if mid[k] == anchor]
    bounds = [start[w]] + starts[1:] + [end_ps(w)]
    return trip, list(zip(bounds[:-1], bounds[1:]))

  def descendants(w: int) -> list[int]:
    hi = end_ps(w)
    out = []
    k = w + 1
    while k < n and start[k] < hi:
      out.append(k)
      k += 1
    return out

  units: list[_Unit] = []
  if oracle.is_fallback:
    ds.fallback_reason = oracle.fallback_reason or f"unregistered schedule '{oracle.schedule}'"
    units = _decode_fallback_units(a, b, whiles0, bodies, segment, descendants, start, selft, infos)
  elif not oracle.is_phased:
    units = _decode_fwd_bwd_units(
        a, b, n, whiles0, bodies, segment, descendants, oracle, stage, start, dur, selft, infos, ds
    )
  else:
    phase_re = oracle.phase_regex()
    subtick_re = oracle.subtick_regex()
    loop_by_scope: dict[str, tuple[int, list[int]]] = {}
    for w in whiles0:
      kids = bodies.get(w, [])
      counts: collections.Counter[str] = collections.Counter()
      for k in kids:
        match = phase_re.search(infos[k].tf)
        if match:
          counts[match.group(1)] += 1
      if counts:
        sc = counts.most_common(1)[0][0]
        if sc not in loop_by_scope or dur[w] > dur[loop_by_scope[sc][0]]:
          loop_by_scope[sc] = (w, kids)
    main_loops = {v[0] for v in loop_by_scope.values()}
    top_phase: dict[str, list[int]] = collections.defaultdict(list)
    for k in range(n):
      if depth[k] == 0 and k not in main_loops:
        match = phase_re.search(infos[k].tf)
        if match:
          top_phase[match.group(1)].append(k)
    plan = [(ph, loop_by_scope.get(ph["scope"])) for ph in oracle.phases]
    macro_units = []
    prev_end = a
    for pi, (ph, wk) in enumerate(plan):
      sc = ph["scope"]
      nxt = b
      for _, wk2 in plan[pi + 1 :]:
        if wk2 is not None:
          nxt = start[wk2[0]]
          break
      cand = top_phase.get(sc, [])
      if wk is None:
        pre = [k for k in cand if prev_end <= start[k] < nxt]
        if pre:
          prev_end = _append_unrolled_phase(ph, pre, start, dur, infos, subtick_re, macro_units)
        continue
      w, kids = wk
      hist = collections.Counter(collections.Counter(mid[k] for k in kids).values())
      cands = [ph["length"] - x for x in (0, 1, 2) if ph["length"] - x >= 1 and hist.get(ph["length"] - x)]
      mode = hist.most_common(1)[0][0] if hist else 0
      exp = mode if mode in cands else (max(cands) if cands else None)
      trip, iters = segment(w, kids, exp)
      pre = [k for k in cand if prev_end <= start[k] and end_ps(k) <= start[w]]
      post = [k for k in cand if end_ps(w) <= start[k] < nxt]
      n_peel = ph["length"] - trip
      tag_pre = sum(selft[k] for k in pre if subtick_re.search(infos[k].tf))
      tag_post = sum(selft[k] for k in post if subtick_re.search(infos[k].tf))
      nf = nl = 0
      if n_peel >= 2:
        nf = nl = 1
      elif n_peel == 1:
        nf, nl = (1, 0) if tag_pre >= tag_post else (0, 1)
      if nf and pre:
        first_tagged = next((k for k in pre if subtick_re.search(infos[k].tf)), pre[0])
        macro_units.append((ph, ph["start_macro"], start[first_tagged], start[w], "peel_first"))
      for it, (t0, t1) in enumerate(iters):
        macro_units.append((ph, ph["start_macro"] + nf + it, t0, t1, "loop"))
      prev_end = end_ps(w)
      if nl and post:
        last_tagged = [k for k in post if subtick_re.search(infos[k].tf)] or post
        e_end = max(end_ps(k) for k in last_tagged)
        macro_units.append((ph, ph["start_macro"] + ph["length"] - 1, end_ps(w), e_end, "peel_last"))
        prev_end = e_end
    if macro_units:
      for ph, j, t0, t1, kind in macro_units:
        u = _Unit(ph["scope"], kind, j, j, t0, t1)
        ks = [k for k in range(_lower_bound(start, t0), n) if start[k] < t1 and k not in main_loops]
        _account_phased(u, ks, j, oracle, stage, start, dur, selft, infos)
        units.append(u)
    else:
      ds.fallback_reason = f"no phased scopes matched for schedule '{oracle.schedule}'"
      units = _decode_fallback_units(a, b, whiles0, bodies, segment, descendants, start, selft, infos)

  units.sort(key=lambda u: u.t0)
  ds.units = units

  def acc_window(t0: int, t1: int) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    busy = 0
    for k in range(_lower_bound(start, t0), n):
      if start[k] >= t1:
        break
      inf = infos[k]
      fam, is_wait = _comm_kind(inf)
      s_ps = selft[k]
      if fam in ("pp", "pp_other", "moe", "fsdp") and is_wait:
        counts[("pp" if fam.startswith("pp") else fam) + "_wait"] += s_ps
      elif fam is not None:
        counts["bookkeeping"] += s_ps
      else:
        counts["compute"] += s_ps
      busy += s_ps
    counts["idle"] = max(0, (t1 - t0) - busy)
    return counts

  for u0, u1 in zip(units[:-1], units[1:]):
    if u1.t0 > u0.t1:
      cls = "tail" if (not oracle.is_phased and u0.loop_dir == "fwd" and u1.loop_dir == "bwd") else "gap"
      ds.gaps.append((u0.t1, u1.t0, cls, acc_window(u0.t1, u1.t0)))
  return ds


# ---------------------------------------------------------------------------
# Trace and metadata discovery (supports local paths and gs:// via etils.epath)
# ---------------------------------------------------------------------------


def _path_str(p: str | os.PathLike[str]) -> str:
  return str(epath.Path(p))


def _is_remote_path(path: str | os.PathLike[str]) -> bool:
  return str(path).startswith(("gs://", "s3://", "az://", "gcs://"))


def list_profile_xplanes(root_dir: str | os.PathLike[str], exclude_stage_subdir: bool = False) -> list[str]:
  """Lists `.xplane.pb` files under `root_dir` (checking `plugins/profile/*` and direct children)."""
  root = epath.Path(root_dir)
  if not root.exists():
    return []
  found: set[str] = set()
  profile_root = root / "plugins" / "profile"
  if profile_root.exists():
    for entry in sorted(profile_root.iterdir(), key=str):
      if entry.is_dir():
        for child in sorted(entry.iterdir(), key=str):
          if child.name.endswith(".xplane.pb"):
            found.add(str(child))
      elif entry.name.endswith(".xplane.pb"):
        found.add(str(entry))
  for child in sorted(root.iterdir(), key=str):
    if child.name.endswith(".xplane.pb"):
      found.add(str(child))
  results = sorted(found)
  if exclude_stage_subdir:
    marker = f"/{STAGE_TRACES_SUBDIR}/"
    results = [p for p in results if marker not in p and not p.endswith(f"/{STAGE_TRACES_SUBDIR}")]
  return results


def find_host0_xplane(
    output_dir: str | os.PathLike[str],
    existing_paths: set[str] | None = None,
    hostname: str | None = None,
) -> str | None:
  """Locates Process 0's newly created `.xplane.pb` under `output_dir` (excluding `_pp_stage_traces`)."""
  all_files = list_profile_xplanes(output_dir, exclude_stage_subdir=True)
  if not all_files:
    return None
  existing = existing_paths or set()
  candidates = [p for p in all_files if p not in existing] or all_files
  host = hostname or socket.gethostname()
  host_matches = [p for p in candidates if os.path.basename(p) == f"{host}.xplane.pb"]
  if host_matches:
    return sorted(host_matches)[-1]
  return sorted(candidates)[-1]


def _load_sidecar_json(path: epath.Path) -> dict[str, Any] | None:
  """Reads and validates a pipeline trace metadata JSON sidecar, returning None on mismatch."""
  try:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
  except Exception:  # pylint: disable=broad-except
    return None
  if isinstance(data, dict) and str(data.get("schema", "")).startswith("pp_trace_metadata") and "local_devices" in data:
    data["_path"] = str(path)
    return data
  return None


def _collect_dir_sidecars(directory: epath.Path, seen: set[str], metas: list[dict[str, Any]]) -> None:
  """Loads valid metadata sidecar JSONs directly inside `directory`."""
  for child in sorted(directory.iterdir(), key=str):
    if not (child.name.endswith(".json") and child.is_file()):
      continue
    sp = str(child)
    if sp in seen:
      continue
    seen.add(sp)
    m = _load_sidecar_json(child)
    if m is not None:
      metas.append(m)


def _collect_sidecars(dirs_or_files: list[str | os.PathLike[str]]) -> list[dict[str, Any]]:
  """Collects valid pipeline trace metadata JSON files from the given paths."""
  seen: set[str] = set()
  metas: list[dict[str, Any]] = []
  for item in dirs_or_files:
    p = epath.Path(item)
    if not p.exists():
      continue
    if p.is_file():
      sp = str(p)
      if sp not in seen and sp.endswith(".json"):
        seen.add(sp)
        m = _load_sidecar_json(p)
        if m is not None:
          metas.append(m)
      continue
    _collect_dir_sidecars(p, seen, metas)
    stage_root = p / STAGE_TRACES_SUBDIR
    if stage_root.exists() and stage_root.is_dir():
      for proc_dir in sorted(stage_root.iterdir(), key=str):
        if proc_dir.is_dir():
          _collect_dir_sidecars(proc_dir, seen, metas)
  return metas


def discover_stage_traces(
    host0_xplane_path: str | os.PathLike[str],
    search_dir: str | os.PathLike[str] | None = None,
    extra_xplane_paths: list[str | os.PathLike[str]] | None = None,
    metadata_paths: list[str | os.PathLike[str]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
  """Discovers all stage `.xplane.pb` files (including Process 0) and their sidecar metadata."""
  host0_str = _path_str(host0_xplane_path)
  files: list[str] = [host0_str]
  seen_files: set[str] = {host0_str}

  root = epath.Path(search_dir) if search_dir is not None else None
  if root is None:
    # Infer root directory above `plugins/profile/<ts>/<host>.xplane.pb`.
    p = epath.Path(host0_str).parent
    if p.parent.name == "profile" and p.parent.parent.name == "plugins":
      root = p.parent.parent.parent
    else:
      root = p

  if extra_xplane_paths is not None:
    for xp in extra_xplane_paths:
      xp_str = _path_str(xp)
      if xp_str not in seen_files:
        seen_files.add(xp_str)
        files.append(xp_str)
  elif root is not None and root.exists():
    stage_root = root / STAGE_TRACES_SUBDIR
    if stage_root.exists() and stage_root.is_dir():
      for proc_dir in sorted(stage_root.iterdir(), key=str):
        if not proc_dir.is_dir():
          continue
        proc_xplanes = list_profile_xplanes(proc_dir, exclude_stage_subdir=False)
        if proc_xplanes:
          latest = proc_xplanes[-1]
          if latest not in seen_files:
            seen_files.add(latest)
            files.append(latest)
    else:
      # Multi-host flat layout (e.g. upload_all_profiler_results=True): pick latest per hostname.
      by_host: dict[str, str] = {}
      for xp in list_profile_xplanes(root, exclude_stage_subdir=True):
        by_host[os.path.basename(xp)] = xp
      for xp in sorted(by_host.values()):
        if xp not in seen_files:
          seen_files.add(xp)
          files.append(xp)

  meta_sources: list[str | os.PathLike[str]] = []
  if root is not None:
    meta_sources.append(root)
  host0_parent = epath.Path(host0_str).parent
  if str(host0_parent) != str(root):
    meta_sources.append(host0_parent)
  for xp in files[1:]:
    meta_sources.append(epath.Path(xp).parent)
  for mp in metadata_paths or []:
    meta_sources.append(mp)
  metas = _collect_sidecars(meta_sources)
  return files, metas


def _dir_prefix(path_str: str) -> str:
  parent = str(epath.Path(path_str).parent).rstrip("/")
  return parent + "/"


def _sidecar_for(path_str: str, hostname: str, metas: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
  """Matches a trace file to its sidecar by closest unique ancestor dir, else unique hostname."""
  best = None
  best_len = -1
  pdir = _dir_prefix(path_str)
  for meta in metas:
    mdir = _dir_prefix(meta["_path"])
    if pdir.startswith(mdir) and len(mdir) > best_len:
      same_dir = [m for m in metas if _dir_prefix(m["_path"]) == mdir]
      if len(same_dir) == 1:
        best = meta
        best_len = len(mdir)
  if best is not None:
    return best, "sidecar_dir"
  hits = [m for m in metas if m.get("hostname") == hostname]
  if len(hits) == 1:
    return hits[0], "sidecar_hostname"
  return None, None


def _schedule_from_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
  """Extracts pipeline schedule parameters from a trace metadata sidecar."""
  if meta is None:
    return {}
  pipe = meta.get("pipeline")
  if isinstance(pipe, dict):
    repeats = int(pipe.get("num_repeats", 1) or 1)
    sched = str(pipe.get("schedule") or "").strip()
    if not sched:
      sched = "circular" if repeats > 1 else "gpipe"
    elif sched == "gpipe" and repeats > 1:
      sched = "circular"
    return {
        "schedule": sched,
        "num_stages": pipe.get("num_stages"),
        "num_microbatches": pipe.get("num_microbatches"),
        "num_repeats": repeats,
        "forwarding_delay": pipe.get("forwarding_delay", 1) or 1,
    }
  return {}


def _partition_stage_map(meta: dict[str, Any] | None) -> dict[int, tuple[int, int]] | None:
  """Maps SPMD partition index to `(global_device_id, stage)` using sidecar `mesh_device_ids`."""
  if not meta or "mesh_device_ids" not in meta or meta["mesh_device_ids"] is None:
    return None
  arr = np.array(meta["mesh_device_ids"])
  ax = meta.get("stage_axis", meta.get("stage_axis_index"))
  if ax is None:
    axes = meta.get("mesh_axes") or []
    ax = axes.index("stage") if "stage" in axes else None
  if ax is None:
    return None
  return {p: (int(arr[idx]), int(idx[ax])) for p, idx in enumerate(np.ndindex(arr.shape))}


# ---------------------------------------------------------------------------
# Alignment, model building, hardware mapping, and in-place XSpace emission
# ---------------------------------------------------------------------------


def _median(vals: list[float]) -> float:
  return float(statistics.median(vals)) if vals else 0.0


def _pct(vals: list[float], q: float) -> float:
  if not vals:
    return 0.0
  sv = sorted(vals)
  idx = min(len(sv) - 1, max(0, int(round(q * (len(sv) - 1)))))
  return float(sv[idx])


def _fmt_ranges(nums) -> str:
  """Formats sorted integers as compact contiguous ranges (e.g. `0-31`)."""
  xs = sorted({int(x) for x in nums if x is not None})
  if not xs:
    return "?"
  out = []
  a = b = xs[0]
  for x in xs[1:]:
    if x == b + 1:
      b = x
    else:
      out.append(f"{a}" if a == b else f"{a}\u2013{b}")
      a = b = x
  out.append(f"{a}" if a == b else f"{a}\u2013{b}")
  return ",".join(out)


def _short_host_map(hosts: list[str]) -> dict[str, str]:
  cand = {h: (h.rsplit("-", 1)[-1] if len(h) > 16 and "-" in h else h) for h in hosts}
  counts = collections.Counter(cand.values())
  return {h: (cand[h] if counts[cand[h]] == 1 else h) for h in hosts}


def _block_label(blk: dict[str, Any], is_phased: bool) -> str:
  if blk["pass"] == "Iter":
    return f"iter {blk['tick']}"
  c = blk["chunk"] if is_phased else blk["repeat"]
  p = {"Tail": "T"}.get(blk["pass"], blk["pass"])
  return f"{p}{blk['mb']}{_PRIME_SUFFIX.get(c, '^' + str(c))}"


def _sanitize_events(events: list[list[Any]]) -> list[list[Any]]:
  """Ensures events on one XLine are properly nested/disjoint for XProf Trace Viewer."""
  events = sorted(events, key=lambda e: (e[0], -e[1]))
  out: list[list[Any]] = []
  stack: list[list[Any]] = []
  for ev in events:
    s, d = ev[0], ev[1]
    while stack and stack[-1][0] + stack[-1][1] <= s:
      stack.pop()
    if stack:
      top_end = stack[-1][0] + stack[-1][1]
      if s + d > top_end:
        d = top_end - s
        if d <= 0:
          continue
        ev = [s, d, ev[2]]
    out.append(ev)
    stack.append(ev)
  return out


def _flow_id(step: int, chain: str, mb: int, leg: int) -> int:
  digest = hashlib.blake2b(f"{int(step)}:{chain}:{int(mb)}:{int(leg)}".encode("utf-8"), digest_size=8).digest()
  return (int.from_bytes(digest, "little") & ((1 << 48) - 1)) + 1000


def _build_flow_chains(blocks: list[dict[str, Any]], edges: dict[str, list[Any]], rel_ps_fn):
  """Connects cross-stage microbatch edges into continuous monotonic XFlow chains."""
  next_e: dict[int, tuple[int, int]] = {}
  has_prev: set[int] = set()
  for ei in range(len(edges["type"])):
    sb, db = edges["src_block"][ei], edges["dst_block"][ei]
    next_e[sb] = (db, ei)
    has_prev.add(db)
  flow_tag: dict[int, tuple[int, int]] = {}
  edge_info: dict[int, dict[str, Any]] = {}
  for ei in range(len(edges["type"])):
    sb = edges["src_block"][ei]
    edge_info[sb] = {
        "out_pp_to": f"S{edges['dst_stage'][ei]}",
        "out_pp_wait_us": round(edges["wait_us"][ei], 2),
        "out_pp_gap_us": round(blocks[edges["dst_block"][ei]]["start_us"] - blocks[sb]["end_us"], 2),
    }
  heads = sorted(sb for sb in next_e if sb not in has_prev)
  for head in heads:
    chain = [head]
    cur = head
    while cur in next_e:
      nxt, _ = next_e[cur]
      if nxt in chain:
        break
      chain.append(nxt)
      cur = nxt
    if len(chain) < 2:
      continue
    b0 = blocks[chain[0]]
    fam = "F" if b0["pass"] == "F" else "B"
    legs = [[chain[0]]]
    for idx in chain[1:]:
      prev = legs[-1][-1]
      if rel_ps_fn(blocks[idx]["start_us"]) <= rel_ps_fn(blocks[prev]["start_us"]):
        legs.append([idx])
      else:
        legs[-1].append(idx)
    for leg_i, nodes in enumerate(legs):
      if len(nodes) < 2:
        continue
      fid = _flow_id(b0["step"], fam, b0["mb"], leg_i)
      for k, blk_i in enumerate(nodes):
        if k == 0:
          flow_tag[blk_i] = (fid, FLOW_OUT)
        elif k == len(nodes) - 1:
          flow_tag[blk_i] = (fid, FLOW_IN)
        else:
          flow_tag[blk_i] = (fid, FLOW_INOUT)
  return flow_tag, edge_info


def _layout_phased_slots(unit: _Unit, us_fn, dev: dict[str, Any]) -> dict[int, tuple[float, float]]:
  """Packs phased-schedule slots into non-overlapping sub-tick display intervals by measured compute."""
  t0, t1 = us_fn(unit.t0, dev), us_fn(unit.t1, dev)
  by_p: dict[int, list[_Slot]] = {0: [], 1: []}
  for sl in unit.slots:
    if not sl.uniform_invalid:
      by_p.setdefault(sl.subtick, []).append(sl)
  c0 = sum(sl.compute for sl in by_p[0]) / PS_PER_US
  c1 = sum(sl.compute for sl in by_p[1]) / PS_PER_US
  e0 = max((us_fn(sl.span[1], dev) for sl in by_p[0] if sl.span), default=None)
  s1 = min((us_fn(sl.span[0], dev) for sl in by_p[1] if sl.span), default=None)
  if e0 is not None and s1 is not None and e0 <= s1:
    split = (e0 + s1) / 2.0
  else:
    split = t0 + (t1 - t0) * (c0 / (c0 + c1) if (c0 + c1) > 0 else 0.5)
  out: dict[int, tuple[float, float]] = {}
  pass_order = ("F", "B", "BW", "W", "Tail", "Iter")
  for subtick, (w0, w1) in ((0, (t0, split)), (1, (split, t1))):
    sls = sorted(
        by_p.get(subtick, []),
        key=lambda sl: (us_fn(sl.span[0], dev) if sl.span else 1e30, pass_order.index(sl.pass_)),
    )
    tot = sum(sl.compute for sl in sls) / PS_PER_US
    scale = min(1.0, (w1 - w0) / tot) if tot > 0 else 1.0
    cur = w0
    for sl in sls:
      width = sl.compute / PS_PER_US * scale
      out[id(sl)] = (cur, cur + width)
      cur += width
  return out


def _relabel_device_step(ds: _DeviceStep, stage: int, oracle: PipelineOracle) -> _DeviceStep:
  """Projects a measured device-step onto an unmeasured stage using oracle labels."""
  nd = copy.copy(ds)
  nd.stage = stage
  new_units = []
  for u in ds.units:
    u2 = copy.copy(u)
    u2.slots = []
    for sl in u.slots:
      s2 = copy.copy(sl)
      if sl.pass_ == "Iter" or oracle.is_fallback or ds.fallback_reason:
        s2.kind = "raw_iter"
      elif oracle.is_phased:
        if not sl.uniform_invalid:
          kind = "B" if sl.pass_ == "BW" else sl.pass_
          mb, chunk, ok = oracle.pass_slot(sl.tick, stage, kind)
          s2.mb, s2.chunk, s2.valid = mb, chunk, ok
        s2.kind = "useful" if s2.valid else "dummy"
      else:
        mb, rep, ok = oracle.fwd_slot(u.iter, stage) if u.loop_dir == "fwd" else oracle.bwd_slot(u.iter, stage)
        s2.mb, s2.chunk, s2.valid = mb, rep, ok
        s2.kind = "useful" if s2.valid else "dummy"
      u2.slots.append(s2)
    new_units.append(u2)
  nd.units = new_units
  return nd


def _classify_dummy_skip(blocks: list[dict[str, Any]]):
  """Classifies invalid slots as SPMD dummy compute vs skipped idle based on measured compute."""
  med: dict[tuple[int, int, str], list[float]] = collections.defaultdict(list)
  for b in blocks:
    if b["pass"] == "Iter":
      continue
    if b["valid"]:
      med[(b["dev"], b["step"], b["pass"])].append(b["compute_us"])
  medv = {k: _median(v) for k, v in med.items()}
  allp: dict[str, list[float]] = collections.defaultdict(list)
  for (_, _, pass_name), val in medv.items():
    allp[pass_name].append(val)
  for b in blocks:
    if b["pass"] == "Iter":
      b["kind"] = "raw_iter"
      continue
    if b["valid"]:
      b["kind"] = "useful"
      continue
    m = medv.get((b["dev"], b["step"], b["pass"]))
    if m is None:
      m = _median(allp.get(b["pass"], [])) or 0.0
    b["kind"] = "dummy" if (m > 0 and b["compute_us"] >= 0.5 * m) else "skipped"


def _infer_cores_per_chip(meta0, metas, results, devs) -> int:
  """Infers the number of TensorCores (JAX devices) per physical TPU chip."""
  cores = [d["core"] for d in devs if d.get("core") is not None]
  for m in [meta0] + list(metas or []):
    for ld in (m or {}).get("local_devices") or []:
      if ld.get("core_on_chip") is not None:
        cores.append(ld["core_on_chip"])
  if cores:
    return max(int(c) for c in cores) + 1
  max_grp = 0
  for m in [meta0] + list(metas or []):
    by_c: collections.Counter[tuple[int, ...]] = collections.Counter()
    for ld in (m or {}).get("local_devices") or []:
      if ld.get("coords") is not None:
        by_c[tuple(ld["coords"])] += 1
    if by_c:
      max_grp = max([max_grp, *by_c.values()])
  if max_grp > 0:
    return max_grp
  for r in results or []:
    by_chip: collections.Counter[int] = collections.Counter()
    for dv in r.get("devices", []):
      gc = dv.get("global_chip_id")
      if isinstance(gc, int) and gc >= 0:
        by_chip[gc] += 1
    if by_chip:
      max_grp = max([max_grp, *by_chip.values()])
  if max_grp > 0:
    return max_grp
  for r in results or []:
    for dv in r.get("devices", []):
      dt = str(dv.get("device_type") or "").lower()
      hm = dv.get("has_megacore")
      if "tpu7x" in dt:
        return 1 if hm else 2
      if "v6" in dt or "v5e" in dt or hm:
        return 1
  return 1


def _stage_hardware_map(num_stages: int, meta0, metas, results, devs, rep_map) -> dict[int, dict[str, Any]]:
  """Computes per-stage Core/Chip/Host membership and representative core metadata."""
  cpc = _infer_cores_per_chip(meta0, metas, results, devs)
  pmap = None
  gid_meta: dict[int, dict[str, Any]] = {}
  dph = len((meta0 or {}).get("local_devices") or []) or None
  for m in [meta0] + list(metas or []):
    if not m:
      continue
    if pmap is None:
      pmap = _partition_stage_map(m)
    pi = m.get("process_index")
    hn = m.get("hostname")
    for ld in m.get("local_devices") or []:
      gid = ld.get("id")
      if isinstance(gid, int):
        gid_meta[gid] = {
            "process_index": pi,
            "hostname": hn,
            "coords": ld.get("coords"),
            "core": ld.get("core_on_chip"),
        }
  raw_hosts = sorted({d["hkey"] for d in devs if d.get("measured")})
  short_map = _short_host_map(raw_hosts)
  dev_by_idx = {d["idx"]: d for d in devs}

  def _hdisp(d):
    return short_map.get(d["hkey"], d["hkey"])

  meas_stages = sorted({d["stage"] for d in devs if d.get("measured")})
  proj_src = meas_stages[0] if meas_stages else 0
  out: dict[int, dict[str, Any]] = {}
  for s in range(num_stages):
    ms = [d for d in devs if d["stage"] == s and d.get("measured")]
    measured = len(ms) > 0
    if pmap:
      cores = sorted({gid for _, (gid, st) in pmap.items() if st == s})
    elif ms and all(isinstance(d.get("global_device_id"), int) for d in ms):
      cores = sorted({d["global_device_id"] for d in ms})
    else:
      cores = []
    if cores:
      chips = sorted({g // cpc for g in cores})
    else:
      chips = sorted({int(d["global_chip_id"]) for d in ms if isinstance(d.get("global_chip_id"), int)})
    hosts = list(dict.fromkeys(_hdisp(d) for d in ms))
    procs = sorted({d["process_index"] for d in ms if d.get("process_index") is not None})
    if not procs and cores:
      procs = sorted(
          {gid_meta[g]["process_index"] for g in cores if g in gid_meta and gid_meta[g].get("process_index") is not None}
      )
    expected_hosts = len(hosts)
    if cores and dph:
      expected_hosts = max(expected_hosts, (len(cores) + dph - 1) // dph)
    rep = dev_by_idx.get(rep_map.get(s)) or (ms[0] if ms else next((d for d in devs if d["stage"] == s), None))
    rep_gid = rep.get("global_device_id") if rep else None
    rep_plane = rep["plane"] if rep else "/device:TPU:0"
    rep_host = _hdisp(rep) if rep and rep.get("measured") else None

    cores_lbl = (f"Core {cores[0]}" if len(cores) == 1 else f"Cores {_fmt_ranges(cores)}") if cores else ""
    chips_lbl = (f"Chip {chips[0]}" if len(chips) == 1 else f"Chips {_fmt_ranges(chips)}") if chips else ""
    if measured:
      if expected_hosts > len(hosts):
        hosts_lbl = f"{len(hosts)}/{expected_hosts} hosts"
      else:
        hosts_lbl = f"{len(hosts)} host" if len(hosts) == 1 else f"{len(hosts)} hosts"
    else:
      hosts_lbl = ""

    if cores_lbl and chips_lbl and measured:
      row_title = f"Stage {s} \u00b7 {cores_lbl} ({chips_lbl} \u00b7 {hosts_lbl})"
    elif cores_lbl and chips_lbl and not measured:
      row_title = f"Stage {s} \u00b7 {cores_lbl} ({chips_lbl}) \u00b7 PROJECTED"
    elif not measured:
      row_title = f"Stage {s} \u00b7 PROJECTED (from Stage {proj_src})"
    else:
      nc = len(ms)
      row_title = f"Stage {s} \u00b7 {nc} core{'s' if nc != 1 else ''} ({hosts_lbl})"

    procs_str = f"Procs {_fmt_ranges(procs)}" if procs else ""
    hosts_list_str = ",".join(hosts[:8]) + (",..." if len(hosts) > 8 else "") if hosts else ""
    if cores_lbl and (chips_lbl or hosts_list_str):
      inner = " \u00b7 ".join(x for x in (chips_lbl, f"Hosts {hosts_list_str}" if hosts_list_str else "", procs_str) if x)
      stage_map_str = f"{cores_lbl} ({inner})"
    elif not measured:
      stage_map_str = f"PROJECTED (from Stage {proj_src})"
    else:
      stage_map_str = row_title
    if not measured and cores_lbl:
      stage_map_str += " \u00b7 PROJECTED"

    rep_core_label = (
        f"Core {rep_gid}"
        if isinstance(rep_gid, int)
        else (f"Dev {rep['local_device']}" if rep and rep.get("local_device") is not None else "Core ?")
    )
    out[s] = {
        "stage": s,
        "measured": measured,
        "projected_from": None if measured else proj_src,
        "cores_per_chip": cpc,
        "cores": cores,
        "chips": chips,
        "short_hosts": hosts,
        "procs": procs,
        "n_cores": len(cores) if cores else len(ms),
        "n_measured_cores": len(ms),
        "n_chips": len(chips),
        "cores_str": cores_lbl,
        "chips_str": chips_lbl,
        "row_title": row_title,
        "stage_map": stage_map_str,
        "rep_core_label": rep_core_label,
        "rep_plane": rep_plane,
        "rep_host": rep_host,
    }
  return out


def _pipeline_subtitle(num_stages: int, num_devices: int | None, hw: dict[int, dict[str, Any]], cpc: int) -> str:
  """Builds the hardware coverage subtitle for the Pipeline Stages custom plane."""
  measured_s = sorted(s for s in range(num_stages) if hw[s]["measured"])
  proj_s = sorted(s for s in range(num_stages) if not hw[s]["measured"])
  if proj_s:
    m_lbl = ("Stage " if len(measured_s) == 1 else "Stages ") + _fmt_ranges(measured_s) + " Measured"
    p_lbl = ("Stage " if len(proj_s) == 1 else "Stages ") + _fmt_ranges(proj_s) + " Projected"
    return f"{m_lbl} \u00b7 {p_lbl}"
  tot_cores = num_devices or sum(hw[s]["n_cores"] for s in range(num_stages))
  tot_chips = sum(hw[s]["n_chips"] for s in range(num_stages)) or (tot_cores // cpc if tot_cores else 0)
  meas_cores = sum(hw[s]["n_measured_cores"] for s in range(num_stages))
  if tot_cores and meas_cores < tot_cores:
    return f"{meas_cores}/{tot_cores} Cores Measured / {tot_chips} Chips"
  return f"All {tot_cores} Core{'s' if tot_cores != 1 else ''} / {tot_chips} Chip{'s' if tot_chips != 1 else ''}"


def _write_bytes_atomic(path: epath.Path, data: bytes) -> None:
  """Writes `data` atomically to `path` (temp+os.replace locally; atomic object write on GCS)."""
  if _is_remote_path(path):
    path.write_bytes(data)
    return
  tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
  try:
    tmp_path.write_bytes(data)
    os.replace(os.fspath(tmp_path), os.fspath(path))
  except Exception:
    if tmp_path.exists():
      tmp_path.unlink()
    raise


def _is_pipeline_lens_plane(plane) -> bool:
  if not plane.name.startswith("/device:CUSTOM:"):
    return False
  stats = _extract_stats(plane, plane.stat_metadata)
  return stats.get("generator") == GENERATOR_NAME


def _decode_host_file(job) -> dict[str, Any]:
  """Loads and decodes all mapped TPU device planes from one host `.xplane.pb` file."""
  path, plane_stage, cfg, preloaded_host = job
  host_trace = preloaded_host if preloaded_host is not None else _load_host_trace(path)
  orc = PipelineOracle(
      cfg["schedule"],
      cfg["num_stages"],
      cfg["num_microbatches"],
      cfg["num_repeats"],
      cfg["forwarding_delay"],
  )
  out: dict[str, Any] = {
      "path": path,
      "hostname": host_trace.hostname,
      "profile_start_ns": host_trace.profile_start_ns,
      "step_nums": host_trace.step_nums,
      "devices": [],
  }
  for dp in host_trace.devices:
    stage = plane_stage.get(dp.plane)
    if stage is None:
      continue
    steps = []
    for k, (a, b) in enumerate(dp.modules):
      step_label = host_trace.step_nums[k] if k < len(host_trace.step_nums) else k
      steps.append(_decode_device_step(dp, (a, b), orc, stage, step_label))
    stp_names = {}
    for inf in dp.info.values():
      if inf.stp:
        stp_names[inf.name[len("collective-permute-start") :]] = inf.stp
    out["devices"].append(
        {
            "plane": dp.plane,
            "ordinal": dp.ordinal,
            "device_id_stat": dp.device_id_stat,
            "global_chip_id": dp.global_chip_id,
            "device_type": dp.device_type,
            "has_megacore": dp.has_megacore,
            "steps": steps,
            "stp": stp_names,
        }
    )
  return out


def _check_ppermute_causality(pmap, devs: list[dict[str, Any]], keep_k: list[int]) -> tuple[int, int]:
  """Checks aligned ppermute causality (`dst_done >= src_start`) across measured partner pairs."""
  if not pmap:
    return 0, 0
  gid_dev = {d["global_device_id"]: d for d in devs if isinstance(d.get("global_device_id"), int)}
  pdev = {p: v[0] for p, v in pmap.items()}
  part_of_gid = {g: p for p, g in pdev.items()}
  checked = 0
  viol = 0
  for d in devs:
    gid = d.get("global_device_id")
    if not isinstance(gid, int) or gid not in part_of_gid:
      continue
    mp = part_of_gid[gid]
    for k in keep_k:
      for (suf, o), (_, de) in d["steps"][k].pp_instances.items():
        for src, dst in d["stp"].get(suf) or ():
          if dst != mp or src == dst:
            continue
          sd = gid_dev.get(pdev.get(src))
          if sd is None or k >= len(sd["steps"]):
            continue
          inst = sd["steps"][k].pp_instances.get((suf, o))
          if inst is None:
            continue
          checked += 1
          if de + d["off_ps"] < inst[0] + sd["off_ps"]:
            viol += 1
  return checked, viol


def augment_host0_xplane_in_place(
    host0_xplane_path: str | os.PathLike[str],
    *,
    search_dir: str | os.PathLike[str] | None = None,
    extra_xplane_paths: list[str | os.PathLike[str]] | None = None,
    metadata_paths: list[str | os.PathLike[str]] | None = None,
    schedule_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Augments Process 0's `.xplane.pb` in-place with decoded cross-stage pipeline planes.

  Preserves 100% of existing planes, lines, events, and stats in Process 0's
  `XSpace` untouched (`/device:TPU:*`, SparseCore, `/host:CPU`, `Task Environment`,
  `/host:metadata` HLO protos) and appends two `/device:CUSTOM:` planes
  (`1. Pipeline Stages` and `2. Bubbles & Comm Waits`). All newly added
  pipeline events are expressed in Process 0's native relative picosecond clock
  so they align directly with Process 0's native `/device:TPU:0` `XLA Ops`.

  Args:
    host0_xplane_path: Path to Process 0's `.xplane.pb` (local or `gs://`).
    search_dir: Optional root profile directory containing `_pp_stage_traces/`
      and/or sibling stage traces and metadata JSON files.
    extra_xplane_paths: Optional explicit list of additional stage `.xplane.pb` files.
    metadata_paths: Optional explicit list of metadata JSON files or directories.
    schedule_overrides: Optional dict overriding `schedule`, `num_stages`,
      `num_microbatches`, `num_repeats`, or `forwarding_delay`.

  Returns:
    Summary dict with counts, alignment p95, causality stats, fallback status, and byte sizes.
  """
  t_start = time.time()
  host0_path = epath.Path(host0_xplane_path)
  host0_str = str(host0_path)
  raw_host0 = host0_path.read_bytes()
  bytes_before = len(raw_host0)
  host0_space = XSpace.FromString(raw_host0)
  del raw_host0

  # Remove only previously generated Pipeline Trace Lens planes (for idempotency),
  # leaving every original TPU/Host/Metadata/Megascale plane untouched.
  kept_planes = [p for p in host0_space.planes if not _is_pipeline_lens_plane(p)]
  if len(kept_planes) != len(host0_space.planes):
    del host0_space.planes[:]
    host0_space.planes.extend(kept_planes)

  host0_parsed = _parse_host_trace(host0_str, host0_space)
  host0_base_ps = host0_parsed.profile_start_ns * 1000

  files, metas = discover_stage_traces(
      host0_str,
      search_dir=search_dir,
      extra_xplane_paths=extra_xplane_paths,
      metadata_paths=metadata_paths,
  )
  meta0 = metas[0] if metas else None
  for m in metas:
    if m.get("process_index") == 0:
      meta0 = m
      break

  cfg = _schedule_from_meta(meta0)
  if schedule_overrides:
    for key, val in schedule_overrides.items():
      if val is not None:
        cfg[key] = val
  if not cfg.get("schedule"):
    cfg["schedule"] = "circular" if int(cfg.get("num_repeats") or 1) > 1 else "gpipe"
  elif cfg.get("schedule") == "gpipe" and int(cfg.get("num_repeats") or 1) > 1:
    cfg["schedule"] = "circular"
  missing = [k for k in ("schedule", "num_stages", "num_microbatches") if not cfg.get(k)]
  if missing:
    raise ValueError(f"Cannot infer pipeline schedule from metadata; missing fields: {', '.join(missing)}")
  cfg["num_stages"] = int(cfg["num_stages"])
  cfg["num_microbatches"] = int(cfg["num_microbatches"])
  cfg["num_repeats"] = int(cfg.get("num_repeats") or 1)
  cfg["forwarding_delay"] = int(cfg.get("forwarding_delay") or 1)
  num_stages = cfg["num_stages"]
  orc = PipelineOracle(cfg["schedule"], num_stages, cfg["num_microbatches"], cfg["num_repeats"], cfg["forwarding_delay"])
  cfg["num_repeats"] = orc.num_repeats

  file_meta: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
  for f in files:
    host_guess = host0_parsed.hostname if f == host0_str else os.path.basename(f).split(".xplane.pb")[0]
    file_meta[f] = _sidecar_for(f, host_guess, metas)

  file_order = sorted(
      files,
      key=lambda f: (0 if f == host0_str else 1, (file_meta[f][0] or {}).get("process_index", 1 << 20), f),
  )
  plane_maps: dict[str, dict[str, int | None]] = {}
  for f in file_order:
    m, _ = file_meta[f]
    pm: dict[str, int | None] = {}
    if m is not None:
      for k, ld in enumerate(m.get("local_devices") or []):
        if ld.get("stage") is not None:
          pm[f"/device:TPU:{k}"] = int(ld["stage"])
    if not pm:
      pm = {f"/device:TPU:{k}": None for k in range(16)}
    plane_maps[f] = pm

  unknown_files = [f for f in files if all(v is None for v in plane_maps[f].values())]
  if unknown_files:
    if len(files) == 1 and len(host0_parsed.devices) >= num_stages:
      ords = sorted(d.ordinal for d in host0_parsed.devices)
      plane_maps[host0_str] = {f"/device:TPU:{o}": (i * num_stages) // len(ords) for i, o in enumerate(ords)}
    else:
      for f in unknown_files:
        if f == host0_str:
          ords = sorted(d.ordinal for d in host0_parsed.devices)
          plane_maps[f] = {f"/device:TPU:{o}": 0 for o in ords}

  jobs = [(f, plane_maps[f], cfg, host0_parsed if f == host0_str else None) for f in file_order]

  # Use threads (never fork!) because this runs inside a live multithreaded JAX/TPU process.
  if len(jobs) <= 1:
    results = [_decode_host_file(jobs[0])]
  else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
      results = list(pool.map(_decode_host_file, jobs))

  # Align all host files onto Process 0's exact clock domain (Process 0 offset == 0).
  ref_res = next((r for r in results if r["path"] == host0_str), results[0])
  ref_dev = ref_res["devices"][0] if ref_res["devices"] else None
  offsets_ps: dict[str, int] = {}
  diffs_all: dict[str, list[float]] = {}
  for r in results:
    f = r["path"]
    if f == host0_str or ref_dev is None:
      offsets_ps[f] = 0
      diffs_all[f] = []
      continue
    diffs: list[int] = []
    for dv in r["devices"]:
      for k, ds in enumerate(dv["steps"]):
        if k >= len(ref_dev["steps"]):
          break
        rd = ref_dev["steps"][k]
        common = [key for key in ds.anchors if key in rd.anchors]
        if common:
          diffs.extend(rd.anchors[key] - ds.anchors[key] for key in common)
        else:
          diffs.append(rd.win[1] - ds.win[1])
    diffs.sort()
    offsets_ps[f] = diffs[len(diffs) // 2] if diffs else 0
    diffs_all[f] = [abs(d - offsets_ps[f]) / PS_PER_US for d in diffs]

  all_res_us = [x for vals in diffs_all.values() for x in vals]
  align_p50 = round(_pct(all_res_us, 0.5), 3)
  align_p95 = round(_pct(all_res_us, 0.95), 3)
  align_max = round(max(all_res_us), 3) if all_res_us else 0.0
  align_badge = "green" if align_p95 <= 20 else ("amber" if align_p95 <= 100 else "red")

  def _hkey(r: dict[str, Any]) -> str:
    m = file_meta[r["path"]][0]
    pi = m.get("process_index") if m else None
    dup = sum(1 for other in results if other["hostname"] == r["hostname"]) > 1
    return f"{r['hostname']}" + (f"/p{pi}" if pi is not None and dup else "")

  # Build flat measured device list.
  devs: list[dict[str, Any]] = []
  for r in results:
    f = r["path"]
    meta, _ = file_meta[f]
    for dv in r["devices"]:
      k = dv["device_id_stat"] if dv["device_id_stat"] is not None else dv["ordinal"]
      ld = (meta or {}).get("local_devices") or []
      ldv = ld[k] if (meta and k is not None and 0 <= k < len(ld)) else {}
      devs.append(
          {
              "file": f,
              "host": r["hostname"],
              "hkey": _hkey(r),
              "process_index": (meta or {}).get("process_index"),
              "plane": dv["plane"],
              "local_device": k,
              "global_device_id": ldv.get("id"),
              "global_chip_id": dv.get("global_chip_id"),
              "coords": ldv.get("coords"),
              "core": ldv.get("core_on_chip"),
              "stage": dv["steps"][0].stage if dv["steps"] else None,
              "steps": dv["steps"],
              "measured": True,
              "off_ps": offsets_ps.get(f, 0),
              "stp": dv["stp"],
          }
      )
  devs = [d for d in devs if d["stage"] is not None and d["steps"]]
  if not devs:
    raise ValueError("No TPU device planes with jit_train_step modules and stage mappings were decoded.")

  devs.sort(
      key=lambda d: (
          d["stage"],
          d["process_index"] if d["process_index"] is not None else 1 << 20,
          d["host"],
          d["local_device"] if d["local_device"] is not None else 0,
      )
  )
  for i, d in enumerate(devs):
    d["idx"] = i

  fallback_reasons = [ds.fallback_reason for d in devs for ds in d["steps"] if ds.fallback_reason]
  fallback_active = orc.is_fallback or bool(fallback_reasons)
  fallback_reason = orc.fallback_reason or (fallback_reasons[0] if fallback_reasons else "")

  nsteps = min(len(d["steps"]) for d in devs)
  t0_ps = min(d["steps"][0].win[0] + d["off_ps"] for d in devs)

  def _us(abs_ps: int, d: dict[str, Any]) -> float:
    return (abs_ps + d["off_ps"] - t0_ps) / PS_PER_US

  def _host0_rel_ps(us_val: float) -> int:
    aligned_abs_ps = t0_ps + int(round(us_val * PS_PER_US))
    return max(1000, aligned_abs_ps - host0_base_ps)

  def _host0_span_ps(start_us: float, end_us: float) -> tuple[int, int] | None:
    s_raw = (t0_ps + int(round(start_us * PS_PER_US))) - host0_base_ps
    e_raw = (t0_ps + int(round(end_us * PS_PER_US))) - host0_base_ps
    if e_raw <= 1000:
      return None
    s_clamped = max(1000, s_raw)
    dur_ps = max(1000, e_raw - s_clamped)
    return s_clamped, dur_ps

  step_labels = [devs[0]["steps"][k].step for k in range(nsteps)]
  durs = [(_median([(d["steps"][k].win[1] - d["steps"][k].win[0]) / PS_PER_US for d in devs]), k) for k in range(nsteps)]
  sorted_durs = sorted(durs)
  sel_step = step_labels[sorted_durs[(len(sorted_durs) - 1) // 2][1]]
  keep_k = list(range(nsteps))

  blocks: list[dict[str, Any]] = []
  units: list[dict[str, Any]] = []
  uniform_w: list[dict[str, Any]] = []
  dev_steps: list[dict[str, Any]] = []

  def _emit_device_step(d: dict[str, Any], k: int, ds: _DeviceStep, measured: bool):
    step = ds.step
    a, b = ds.win
    step_fallback = bool(ds.fallback_reason or orc.is_fallback)
    rec = {
        "step": step,
        "k": k,
        "dev": d["idx"],
        "stage": d["stage"],
        "t0_us": _us(a, d),
        "t1_us": _us(b, d),
        "dur_us": (b - a) / PS_PER_US,
        "measured": measured,
        "acc": collections.Counter(),
        "gap_acc": collections.Counter(),
    }
    for u in ds.units:
      ui = len(units)
      uu = {
          "step": step,
          "k": k,
          "dev": d["idx"],
          "stage": d["stage"],
          "phase": u.phase,
          "kind": u.kind,
          "macro": u.macro,
          "iter": u.iter,
          "start_us": _us(u.t0, d),
          "end_us": _us(u.t1, d),
          "acc": {kk: v / PS_PER_US for kk, v in u.acc.items()},
          "slots": [],
          "loop_dir": u.loop_dir,
          "measured": measured,
      }
      units.append(uu)
      rec["acc"].update({kk: v / PS_PER_US for kk, v in u.acc.items()})
      use_phased_layout = orc.is_phased and not step_fallback
      lay = (
          _layout_phased_slots(u, _us, d)
          if use_phased_layout
          else {id(sl): (_us(u.t0, d), _us(u.t1, d)) for sl in u.slots}
      )
      for sl in u.slots:
        if sl.uniform_invalid:
          uniform_w.append(
              {"step": step, "dev": d["idx"], "stage": d["stage"], "tick": sl.tick, "compute_us": sl.compute / PS_PER_US}
          )
          continue
        bi = len(blocks)
        ls, le = lay.get(id(sl), (_us(u.t0, d), _us(u.t1, d)))
        sp = (_us(sl.span[0], d), _us(sl.span[1], d)) if sl.span else None
        blk = {
            "step": step,
            "k": k,
            "dev": d["idx"],
            "stage": d["stage"],
            "phase": u.phase,
            "tick": sl.tick,
            "macro": u.macro,
            "subtick": sl.subtick,
            "pass": sl.pass_,
            "mb": sl.mb,
            "chunk": sl.chunk if use_phased_layout else 0,
            "repeat": 0 if use_phased_layout else sl.chunk,
            "valid": bool(sl.valid),
            "kind": sl.kind,
            "start_us": ls,
            "end_us": le,
            "span": sp,
            "compute_us": sl.compute / PS_PER_US,
            "pp_wait_us": sl.waits.get("pp_wait", 0) / PS_PER_US,
            "moe_wait_us": sl.waits.get("moe_wait", 0) / PS_PER_US,
            "fsdp_wait_us": sl.waits.get("fsdp_wait", 0) / PS_PER_US,
            "unit": ui,
            "measured": measured,
        }
        blocks.append(blk)
        uu["slots"].append(bi)
    for x, y, cls, acc in ds.gaps:
      rec["gap_acc"][cls] += (y - x) / PS_PER_US
      for kk, v in acc.items():
        rec["gap_acc"][cls + ":" + kk] += v / PS_PER_US
    dev_steps.append(rec)

  for d in devs:
    for k in keep_k:
      _emit_device_step(d, k, d["steps"][k], measured=True)

  # Project any unmeasured stage by SPMD symmetry so all S stages render.
  measured_stages = sorted({d["stage"] for d in devs})
  projected: list[dict[str, Any]] = []
  if len(measured_stages) < num_stages:
    rep_stage = measured_stages[0]
    rep = next(d for d in devs if d["stage"] == rep_stage)
    for s in range(num_stages):
      if s in measured_stages:
        continue
      pd = dict(rep)
      pd.update(
          {
              "stage": s,
              "measured": False,
              "idx": len(devs) + len(projected),
              "global_device_id": None,
              "host": f"(projected from {rep['host']})",
              "hkey": f"proj-s{s}",
              "projected_from": rep["idx"],
          }
      )
      projected.append(pd)
      for k in keep_k:
        _emit_device_step(pd, k, _relabel_device_step(rep["steps"][k], s, orc), measured=False)

  all_devs = devs + projected
  _classify_dummy_skip(blocks)

  # Representative device index per stage & cross-stage flow edges (skipped in fallback mode).
  rep_dev: dict[int, int] = {}
  for d in all_devs:
    rep_dev.setdefault(d["stage"], d["idx"])
  repset = set(rep_dev.values())
  blk_lookup: dict[tuple[int, str, int, int], int] = {}
  if not fallback_active:
    for i, b in enumerate(blocks):
      if b["dev"] in repset and b["valid"] and b["pass"] in ("F", "B", "BW"):
        chain = "F" if b["pass"] == "F" else "B"
        layer = orc.layer_index(b["stage"], b["chunk"] if orc.is_phased else b["repeat"])
        blk_lookup[(b["step"], chain, b["mb"], layer)] = i

  edges: dict[str, list[Any]] = {
      "type": [],
      "src_block": [],
      "dst_block": [],
      "wait_us": [],
      "src_stage": [],
      "dst_stage": [],
  }
  for (step, chain, mb, layer), i in sorted(blk_lookup.items()):
    nl = layer + 1 if chain == "F" else layer - 1
    j = blk_lookup.get((step, chain, mb, nl))
    if j is None:
      continue
    bi, bj = blocks[i], blocks[j]
    edges["type"].append(chain)
    edges["src_block"].append(i)
    edges["dst_block"].append(j)
    edges["wait_us"].append(round(bj["pp_wait_us"], 3))
    edges["src_stage"].append(bi["stage"])
    edges["dst_stage"].append(bj["stage"])

  # Per-device-step and per-step summary metrics.
  by_ds: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
  for b in blocks:
    by_ds[(b["dev"], b["step"])].append(b)
  uw_by: dict[tuple[int, int], float] = collections.defaultdict(float)
  for w in uniform_w:
    uw_by[(w["dev"], w["step"])] += w["compute_us"]
  per_dev_step = []
  for rec in dev_steps:
    bl = by_ds[(rec["dev"], rec["step"])]
    acc, ga = rec["acc"], rec["gap_acc"]
    useful = sum(b["compute_us"] for b in bl if b["kind"] in ("useful", "raw_iter") and b["pass"] != "Tail")
    dummy = sum(b["compute_us"] for b in bl if b["kind"] == "dummy") + uw_by[(rec["dev"], rec["step"])]
    skip = sum(b["compute_us"] for b in bl if b["kind"] == "skipped")
    pp_w = acc.get("pp_wait", 0.0) + sum(
        v for kk, v in ga.items() if kk.endswith(":pp_wait") and kk.startswith(("gap", "tail"))
    )
    moe_w = acc.get("moe_wait", 0.0) + sum(
        v for kk, v in ga.items() if kk.endswith(":moe_wait") and kk.startswith(("gap", "tail"))
    )
    fsdp_w = acc.get("fsdp_wait", 0.0) + sum(
        v for kk, v in ga.items() if kk.endswith(":fsdp_wait") and kk.startswith(("gap", "tail"))
    )
    idle = acc.get("idle", 0.0) + sum(v for kk, v in ga.items() if kk in ("gap:idle", "tail:idle"))
    pp_bub = sum(b["pp_wait_us"] for b in bl if not b["valid"]) + ga.get("gap:pp_wait", 0.0)
    bubble = dummy + skip + pp_bub + idle
    dur = rec["dur_us"]
    per_dev_step.append(
        {
            "step": rec["step"],
            "measured": rec["measured"],
            "t0_us": rec["t0_us"],
            "dur_us": dur,
            "useful_us": useful,
            "dummy_us": dummy,
            "pp_wait_us": pp_w,
            "moe_wait_us": moe_w,
            "fsdp_wait_us": fsdp_w,
            "bubble_us": bubble,
        }
    )

  per_step = []
  for st in sorted({r["step"] for r in per_dev_step}):
    ps = [r for r in per_dev_step if r["step"] == st and r["measured"]] or [r for r in per_dev_step if r["step"] == st]
    dur_m = _median([r["dur_us"] for r in ps])
    useful_m = _median([r["useful_us"] for r in ps])
    dummy_m = _median([r["dummy_us"] for r in ps])
    bubble_m = _median([r["bubble_us"] for r in ps])
    per_step.append(
        {
            "step": st,
            "devices": len(ps),
            "t0_us": min(r["t0_us"] for r in ps),
            "dur_us": dur_m,
            "useful_us": useful_m,
            "dummy_us": dummy_m,
            "pp_wait_us": _median([r["pp_wait_us"] for r in ps]),
            "moe_wait_us": _median([r["moe_wait_us"] for r in ps]),
            "fsdp_wait_us": _median([r["fsdp_wait_us"] for r in ps]),
            "bubble_fraction": bubble_m / dur_m if dur_m else 0.0,
            "lockstep_bubble_fraction": dummy_m / (useful_m + dummy_m) if (useful_m + dummy_m) else 0.0,
        }
    )

  caus_checked, caus_viol = _check_ppermute_causality(_partition_stage_map(meta0), devs, keep_k)

  # Infer global device count for hardware subtitle.
  num_devices = None
  for m in [meta0] + list(metas or []):
    if not m:
      continue
    num_devices = num_devices or m.get("global_device_count")
    if not num_devices and m.get("mesh_device_ids") is not None:
      num_devices = int(np.array(m["mesh_device_ids"]).size)
    if not num_devices and m.get("mesh_shape"):
      num_devices = int(np.prod(m["mesh_shape"]))
    if not num_devices and m.get("process_count") and m.get("local_devices"):
      num_devices = int(m["process_count"]) * len(m["local_devices"])
    if num_devices:
      break

  hw = _stage_hardware_map(num_stages, meta0, metas, results, all_devs, rep_dev)
  cpc = _infer_cores_per_chip(meta0, metas, results, all_devs)
  subtitle = _pipeline_subtitle(num_stages, num_devices, hw, cpc)

  # Pick next free /device:CUSTOM: plane id (< 200).
  next_custom_id = max((p.id for p in host0_space.planes if p.name.startswith("/device:CUSTOM:")), default=-1) + 1
  if next_custom_id + 1 >= 200:
    raise ValueError(f"Insufficient free /device:CUSTOM: plane IDs (next={next_custom_id})")

  def _add_custom_plane(pid: int, name: str, stats: dict[str, Any]) -> _PlaneBuilder:
    plane = host0_space.planes.add()
    plane.id = pid
    plane.name = name
    builder = _PlaneBuilder(plane)
    plane_stats = dict(stats)
    plane_stats["generator"] = GENERATOR_NAME
    builder.add_stats(plane, plane_stats)
    return builder

  ev_lines: list[tuple[_PlaneBuilder, Any, list[list[Any]]]] = []
  planes_added = 0

  # ---------------- Plane A: 1. Pipeline Stages ----------------
  plane_a_stats: dict[str, Any] = {
      "schedule": orc.schedule,
      "num_stages": num_stages,
      "num_microbatches": cfg["num_microbatches"],
      "num_devices": num_devices or 0,
      "fallback_mode": int(fallback_active),
  }
  if fallback_active:
    plane_a_stats["fallback_reason"] = fallback_reason
    plane_a_stats["developer_docs"] = DEVELOPER_DOCS_PATH

  p0 = _add_custom_plane(
      next_custom_id,
      f"/device:CUSTOM:1. Pipeline Stages ({subtitle})",
      plane_a_stats,
  )
  next_custom_id += 1
  planes_added += 1

  ln_map = p0.line(1, "1 Step & Hardware Map")
  ln_ph = p0.line(2, "2 Phase \u00b7 Tick")

  stage_maps = {f"stage_{s}_map": hw[s]["stage_map"] for s in range(num_stages)}
  evs_map: list[list[Any]] = []
  for row in per_step:
    span = _host0_span_ps(row["t0_us"], row["t0_us"] + row["dur_us"])
    if span is None:
      continue
    if fallback_active:
      txt = (
          f"step {row['step']} \u00b7 {row['dur_us'] / 1000:.1f} ms \u00b7 pp wait {row['pp_wait_us'] / 1000:.1f} ms "
          f"\u00b7 align p95 {align_p95:.1f} \u00b5s \u00b7 [raw iter fallback: register schedule in "
          f"pipeline_trace_lens.py \u2014 see {DEVELOPER_DOCS_PATH}]"
      )
    else:
      txt = (
          f"step {row['step']} \u00b7 {row['dur_us'] / 1000:.1f} ms \u00b7 bubble {row['bubble_fraction'] * 100:.1f}% "
          f"(dummy {row['dummy_us'] / 1000:.1f} ms) \u00b7 pp wait {row['pp_wait_us'] / 1000:.1f} ms "
          f"\u00b7 align p95 {align_p95:.1f} \u00b5s"
      )
    st_stats: dict[str, Any] = {
        "step": int(row["step"]),
        "step_ms": round(row["dur_us"] / 1000, 3),
        "bubble_pct": round(row["bubble_fraction"] * 100, 3),
        "lockstep_bubble_pct": round(row["lockstep_bubble_fraction"] * 100, 3),
        "dummy_ms": round(row["dummy_us"] / 1000, 3),
        "pp_wait_ms": round(row["pp_wait_us"] / 1000, 3),
        "moe_wait_ms": round(row["moe_wait_us"] / 1000, 3),
        "fsdp_wait_ms": round(row["fsdp_wait_us"] / 1000, 3),
        "useful_ms": round(row["useful_us"] / 1000, 3),
        "selected": int(row["step"] == sel_step),
        "devices_measured": row["devices"],
        "align_p50_us": align_p50,
        "align_p95_us": align_p95,
        "align_max_us": align_max,
        "align_badge": align_badge,
        "anchors_used": len(all_res_us),
        "causality_checked": caus_checked,
        "causality_violations": caus_viol,
    }
    if fallback_active:
      st_stats["fallback_mode"] = 1
      st_stats["fallback_reason"] = fallback_reason
      st_stats["developer_docs"] = DEVELOPER_DOCS_PATH
      st_stats["note"] = (
          f"Raw iteration fallback active ({fallback_reason}). Register this schedule via "
          f"register_pipeline_schedule() in pipeline_trace_lens.py (see {DEVELOPER_DOCS_PATH})."
      )
    st_stats.update(stage_maps)
    st_stats["offsets_us"] = ", ".join(f"{_hkey(r)}={offsets_ps[r['path']] / PS_PER_US:+.1f}" for r in results)
    evs_map.append([span[0], span[1], (p0.event_md("Step", txt), st_stats)])
  ev_lines.append((p0, ln_map, evs_map))

  ref_stage = next((s for s in range(num_stages) if hw[s]["measured"]), 0)
  rd_ph = rep_dev.get(ref_stage)
  evs_ph: list[list[Any]] = []
  ph_span: dict[tuple[int, str], tuple[float, float]] = collections.OrderedDict()
  for u in units:
    if u["dev"] != rd_ph:
      continue
    span = _host0_span_ps(u["start_us"], u["end_us"])
    if span is None:
      continue
    acc = u["acc"]
    is_phased_u = orc.is_phased and not fallback_active
    tick_title = f"{u['phase']} \u00b7 {'macro' if is_phased_u else 'iter'} {u['macro']}" + (
        "" if u["kind"] in ("loop", "fallback_loop") else f" ({u['kind']})"
    )
    evs_ph.append(
        [
            span[0],
            span[1],
            (
                p0.event_md("Tick", _colorize(tick_title, "tickA" if u["macro"] % 2 == 0 else "tickB")),
                {
                    "phase": u["phase"],
                    "macro": int(u["macro"]),
                    "kind": u["kind"],
                    "step": int(u["step"]),
                    "compute_ms": round(sum(blocks[j]["compute_us"] for j in u["slots"]) / 1000, 4),
                    "bookkeeping_us": round(acc.get("bookkeeping", 0.0), 2),
                    "idle_us": round(acc.get("idle", 0.0), 2),
                },
            ),
        ]
    )
    key = (u["step"], u["phase"])
    prev = ph_span.get(key)
    ph_span[key] = (
        (u["start_us"], u["end_us"]) if prev is None else (min(prev[0], u["start_us"]), max(prev[1], u["end_us"]))
    )
  for (stp, ph), (x0, x1) in ph_span.items():
    span = _host0_span_ps(x0, x1)
    if span is not None:
      evs_ph.append([span[0], span[1] + 1000, (p0.event_md("Phase", _colorize(ph, "phase")), {"step": int(stp)})])
  ev_lines.append((p0, ln_ph, evs_ph))

  agg: dict[tuple[int, int, int, int, str], list[float]] = collections.defaultdict(list)
  for bk in blocks:
    if bk["measured"]:
      agg[(bk["step"], bk["stage"], bk["tick"], bk["subtick"], bk["pass"])].append(bk["compute_us"])
  flow_tag, edge_info = _build_flow_chains(blocks, edges, _host0_rel_ps)
  bidx = {id(bk): i for i, bk in enumerate(blocks)}

  # ---------------- Plane B: 2. Bubbles & Comm Waits (By Stage) ----------------
  p1 = _add_custom_plane(
      next_custom_id,
      "/device:CUSTOM:2. Bubbles & Comm Waits (By Stage)",
      {"num_stages": num_stages},
  )
  planes_added += 1

  for s in range(num_stages):
    rd = rep_dev.get(s)
    hs = hw[s]
    measured = hs["measured"]
    ln_s = p0.line(10 + s, f"3.{s} {hs['row_title']}")
    ln_b = p1.line(1 + s, f"1.{s} Stage {s} Bubbles & Waits" + ("" if measured else " [PROJECTED]"))
    evs_s: list[list[Any]] = []
    evs_b: list[list[Any]] = []
    my_blocks = [bk for bk in blocks if bk["dev"] == rd] if rd is not None else []
    rep_core_str = (
        f"{hs['rep_core_label']} ({hs['rep_plane']} on {hs['rep_host']})"
        if measured
        else f"projected from Stage {hs['projected_from']}"
    )
    if measured and hs["cores"]:
      hcsv = ",".join(hs["short_hosts"])
      nc = hs["n_cores"]
      reps_str = f"All {nc} core{'s' if nc != 1 else ''} ({hs['cores_str']}, {hs['chips_str']}" + (
          f", Hosts {hcsv})" if hcsv else ")"
      )
    elif measured:
      reps_str = f"{hs['n_cores']} core{'s' if hs['n_cores'] != 1 else ''} ({','.join(hs['short_hosts'])})"
    else:
      reps_str = f"PROJECTED from Stage {hs['projected_from']}" + (
          f" ({hs['cores_str']}, {hs['chips_str']})" if hs["cores"] else ""
      )

    for bk in my_blocks:
      bi = bidx[id(bk)]
      span = _host0_span_ps(bk["start_us"], bk["end_us"])
      if span is None:
        continue
      a_vals = agg.get((bk["step"], s, bk["tick"], bk["subtick"], bk["pass"]), [])
      is_raw_iter = bk["pass"] == "Iter"
      lab = _block_label(bk, orc.is_phased and not fallback_active)
      st: dict[str, Any] = {
          "step": int(bk["step"]),
          "stage": s,
          "tick": int(bk["tick"]),
          "macro": int(bk["macro"]),
          "phase": bk["phase"],
          "kind": bk["kind"],
          "compute_ms": round(bk["compute_us"] / 1000, 4),
          "pp_wait_us": round(bk["pp_wait_us"], 2),
          "moe_wait_us": round(bk["moe_wait_us"], 2),
          "fsdp_wait_us": round(bk["fsdp_wait_us"], 2),
          "rep_core": rep_core_str,
          "represents": reps_str,
          "source": "measured" if bk["measured"] else "projected",
      }
      if is_raw_iter:
        st["fallback_mode"] = 1
        st["developer_docs"] = DEVELOPER_DOCS_PATH
      else:
        st["mb"] = int(bk["mb"])
        st["subtick"] = int(bk["subtick"])
        st["valid"] = int(bk["valid"])
        st["chunk" if orc.is_phased else "repeat"] = int(bk["chunk"] if orc.is_phased else bk["repeat"])
      if a_vals and measured:
        sa = sorted(a_vals)
        mn, md_, mx = sa[0] / 1000.0, sa[len(sa) // 2] / 1000.0, sa[-1] / 1000.0
        st["intra_stage_compute_min_med_max_ms"] = f"{mn:.2f} / {md_:.2f} / {mx:.2f}"
        st["intra_stage_spread_ms"] = round(mx - mn, 3)
        st["devices"] = len(sa)
      if bk["span"]:
        st["span_start_ms"] = round(_host0_rel_ps(bk["span"][0]) / 1e9, 4)
        st["span_end_ms"] = round(_host0_rel_ps(bk["span"][1]) / 1e9, 4)
      if is_raw_iter:
        color_key = "tickA" if bk["tick"] % 2 == 0 else "tickB"
        md = p0.event_md(PASS_METADATA_NAMES["Iter"], _colorize(lab, color_key))
        evs_s.append([span[0], span[1], (md, st)])
      elif bk["valid"]:
        if bi in flow_tag:
          fid, fdir = flow_tag[bi]
          st["flow"] = _u64(flow_value(fid, fdir))
        if bi in edge_info:
          st.update(edge_info[bi])
        md = p0.event_md(PASS_METADATA_NAMES.get(bk["pass"], bk["pass"]), _colorize(lab, bk["pass"]))
        evs_s.append([span[0], span[1], (md, st)])
      else:
        name = "SPMD Dummy Compute" if bk["kind"] == "dummy" else "Skip Idle"
        disp_s = (
            (f"\u2205 {bk['pass']}" if span[1] >= 20_000_000_000 else "\u2205") if bk["kind"] == "dummy" else "\u2205"
        )
        evs_s.append([span[0], span[1], (p0.event_md(name, _colorize(disp_s, bk["kind"])), st)])
        disp_b = f"\u2205 {lab}" if bk["kind"] == "dummy" else f"skip {lab}"
        evs_b.append([span[0], span[1], (p1.event_md(name, _colorize(disp_b, bk["kind"])), dict(st))])

    for u in units:
      if u["dev"] != rd:
        continue
      acc = u["acc"]
      ppw = acc.get("pp_wait", 0.0)
      ow = acc.get("moe_wait", 0.0) + acc.get("fsdp_wait", 0.0)
      t1 = u["end_us"]
      cur = max(u["start_us"], t1 - (ppw + ow))
      out_notes = []
      for bi in u["slots"]:
        if bi in edge_info:
          ei = edge_info[bi]
          out_notes.append(
              f"{_block_label(blocks[bi], orc.is_phased)}->{ei['out_pp_to']}(wait={ei['out_pp_wait_us']:.0f}us)"
          )
      if ow > 1.0:
        span_ow = _host0_span_ps(cur, cur + ow)
        if span_ow is not None:
          evs_b.append(
              [
                  span_ow[0],
                  span_ow[1],
                  (
                      p1.event_md("MoE/FSDP Wait", _colorize(f"moe/fsdp {ow / 1000:.2f} ms", "other_wait")),
                      {
                          "moe_wait_us": round(acc.get("moe_wait", 0.0), 2),
                          "fsdp_wait_us": round(acc.get("fsdp_wait", 0.0), 2),
                          "step": int(u["step"]),
                          "tick": int(u["macro"]),
                          "phase": u["phase"],
                          "source": "measured" if measured else "projected",
                      },
                  ),
              ]
          )
      if ppw > 1.0:
        span_pp = _host0_span_ps(cur + ow, cur + ow + ppw)
        if span_pp is not None:
          w_st: dict[str, Any] = {
              "pp_wait_us": round(ppw, 2),
              "step": int(u["step"]),
              "tick": int(u["macro"]),
              "phase": u["phase"],
              "source": "measured" if measured else "projected",
          }
          if out_notes:
            w_st["outgoing_pp_waits"] = ", ".join(out_notes)
          evs_b.append(
              [
                  span_pp[0],
                  span_pp[1],
                  (p1.event_md("PP ppermute Wait", _colorize(f"pp wait {ppw / 1000:.2f} ms", "pp_wait")), w_st),
              ]
          )
    ev_lines.append((p0, ln_s, evs_s))
    ev_lines.append((p1, ln_b, evs_b))

  events_added = 0
  for pb, ln, evs in ev_lines:
    for s0, d_ps, (md, st) in _sanitize_events(evs):
      if s0 <= 0:
        continue
      pb.event(ln, md, s0, d_ps, st)
      events_added += 1

  serialized = host0_space.SerializeToString()
  _write_bytes_atomic(host0_path, serialized)
  return {
      "host0_xplane_path": host0_str,
      "schedule": orc.schedule,
      "num_stages": num_stages,
      "num_microbatches": cfg["num_microbatches"],
      "stages_measured": measured_stages,
      "planes_added": planes_added,
      "events_added": events_added,
      "align_p95_us": align_p95,
      "causality_checked": caus_checked,
      "causality_violations": caus_viol,
      "fallback_mode": fallback_active,
      "fallback_reason": fallback_reason,
      "bytes_before": bytes_before,
      "bytes_after": len(serialized),
      "elapsed_s": round(time.time() - t_start, 3),
  }
