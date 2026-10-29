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

"""Event parser and data structures for TPU ICI-SDC and sflag straggler logs.

This module parses both raw TPU driver/runtime logs (`TPU_LOG_DIR`) and
`libtpu.sdk.debug` event-log callback records into a single normalized `Event`
dataclass using only the Python 3 standard library.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# Regular expressions for raw log lines and callback event messages
# ---------------------------------------------------------------------------

# Matches optional host wrapper prefix: `SDC_LOG: [host=0] ...` or `[host=0]`
_HOST_PREFIX_RE = re.compile(r"\[host=(?P<host>[^\]]+)\]")

# Matches bracketed core identifier in standard, compact, tuple, or launch-tagged form:
#   [job.0x3x0_TC0@1]
#   [1x0x0_TC0@2:7e388229:-]
#   [0x3x0_TC0]
#   [(1,2)_TC0@1]
_CORE_BRACKET_RE = re.compile(
    r"\[(?:(?P<job>[A-Za-z0-9_.-]+)\.)?"
    r"(?P<coord>"
    r"\d+x\d+(?:x\d+)?(?:_[A-Za-z]+\d+)?"
    r"|\(\s*\d+\s*,\s*\d+(?:\s*,\s*\d+)?\s*\)(?:_[A-Za-z]+\d+)?"
    r"|core_?\d+|chip_?\d+"
    r")"
    r"(?:@(?P<run_id>\d+))?"
    r"(?::(?P<launch_id>[0-9a-fA-F]+):[^\]]*)?\]"
)

# Parse chip (x, y, z) and TensorCore index from `XxYxZ_TCc`, `XxY`, `(X,Y)_TCc`, etc.
_COORD_3D_RE = re.compile(r"^(?P<x>\d+)x(?P<y>\d+)(?:x(?P<z>\d+))?(?:_(?P<type>[A-Za-z]+)(?P<tc>\d+))?$")
_COORD_TUPLE_RE = re.compile(
    r"^\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)(?:\s*,\s*(?P<z>\d+))?\s*\)" r"(?:_(?P<type>[A-Za-z]+)(?P<tc>\d+))?$"
)

# 1a. Hardware non-compact ICI-SDC transfer timeout line (`llo_sdc_checker_emitters.cc`):
#     `ICI transfer wait on SEND|RECV flag timed out on 0x3x0_TC0 at iteration 9 on core with global core index 24`
_XFER_TIMEOUT_HW_RE = re.compile(
    r"ICI transfer wait on (?P<dir>SEND|RECV|send|recv) flag timed out on "
    r"(?P<core>.+?) at iteration (?P<iter>\d+)"
    r"(?:\s+on core with global core index (?P<g>\d+))?",
    re.IGNORECASE,
)

# 1b. Explicit src->dst ICI-SDC transfer timeout line:
#     `ICI SDC Test transfer RECV timeout : 0x3x1_TC0 -> 0x3x0_TC0 - 4194304 bytes - iteration 9`
_XFER_TIMEOUT_PAIR_RE = re.compile(
    r"ICI SDC Test transfer (?P<dir>SEND|RECV|send|recv) timeout\s*:\s*"
    r"(?P<src>\d+x\d+x\d+_TC\d+)\s*->\s*(?P<dst>\d+x\d+x\d+_TC\d+)\s*-\s*"
    r"(?P<bytes>\d+)\s*bytes\s*-\s*iteration\s*(?P<iter>\d+)"
)

# 2a. Hardware compact-codegen ICI-SDC transfer timeout line (`llo_sdc_checker_emitters.cc`):
#     `ICI SDC Test transfer recv timeout (dynamic) at iteration 9 on core with global core index 26`
_XFER_TIMEOUT_COMPACT_HW_RE = re.compile(
    r"ICI SDC Test transfer (?P<dir>send|recv|SEND|RECV) timeout \(dynamic\) "
    r"at iteration (?P<iter>\d+)"
    r"(?:\s+on core with global core index (?P<g>\d+))?",
    re.IGNORECASE,
)

# 2b. Pair compact-codegen ICI-SDC transfer timeout line:
#     `ICI SDC Test transfer recv timeout (dynamic) : 25 -> 26 - 4194304 bytes - iteration 9`
_XFER_TIMEOUT_COMPACT_PAIR_RE = re.compile(
    r"ICI SDC Test transfer (?P<dir>send|recv|SEND|RECV) timeout \(dynamic\)\s*:\s*"
    r"(?P<src_g>\d+)\s*->\s*(?P<dst_g>\d+)\s*-\s*"
    r"(?P<bytes>\d+)\s*bytes\s*-\s*iteration\s*(?P<iter>\d+)"
)

# 3a. Hardware `ComputeAndDumpIcBw` completion & delay_iterations (P) line:
#     `IciSdcTest :: ICI BW: 162.109 (GiB/s), clock: 1250 (MHz), data: 32 (MiB),`
#     ` time: 0.197398 (ms), delay iterations: 101`
_DELAY_ITER_HW_RE = re.compile(
    r"IciSdcTest\s*::\s*ICI BW:\s*(?P<bw>-?\d+(?:\.\d+)?)\s*\(GiB/s\)\s*,\s*"
    r"clock:\s*(?P<clock>-?\d+(?:\.\d+)?)\s*\(MHz\)\s*,\s*"
    r"data:\s*(?P<data>-?\d+(?:\.\d+)?)\s*\(MiB\)\s*,\s*"
    r"time:\s*(?P<ms>-?\d+(?:\.\d+)?)\s*\(ms\)\s*,\s*"
    r"delay iterations:\s*(?P<P>-?\d+)"
    r"(?:.*?\bon core\s+(?P<core>\S+))?"
    r"(?:.*?\bglobal core index[:\s]+(?P<g>\d+))?",
    re.IGNORECASE,
)

# 3b. Compact completion & delay_iterations (P) line:
#     `IciSdcTest :: ICI BW: 100.785942 GiB/s, time: 193.733000 ms (global core index: 61) (delay iterations: 10)`
_DELAY_ITER_SHORT_RE = re.compile(
    r"IciSdcTest\s*::\s*ICI BW:\s*(?P<bw>[0-9.]+)\s*GiB/s,\s*"
    r"time:\s*(?P<ms>[0-9.]+)\s*ms\s*"
    r"\(global core index:\s*(?P<g>\d+)\)"
    r"(?:\s*\(delay iterations:\s*(?P<P>-?\d+)\))?"
)

# 4. Approach 2 passive sflag timeout line (`Wait timeout on sflag` or `Short timeout on sflag`):
_SFLAG_TIMEOUT_RE = re.compile(
    r"(?:Wait|Short) timeout on sflag\s*\{(?P<alloc>[^\}]+)\}\s*"
    r"'(?P<label>[^']*)'\s*"
    r"for value\s*\{(?P<for_val>-?\d+)\}\s*"
    r"after\s*(?P<timeout_ms>\d+)\s*milliseconds,\s*"
    r"latest value seen\s*\{(?P<latest_val>-?\d+)\},\s*"
    r"hlo:\s*(?P<hlo>\S+)"
    r"(?:\s*::\s*program tracker\s*::\s*depth:(?P<depth>-?\d+)\s*"
    r"meta_id:(?P<meta_id>-?\d+)\s*(?P<levels>.*?))?"
    r"(?:\s*::\s*send sflag tracker\s*::\s*unfinished send sflags:\s*"
    r"(?P<unfinished>-?\d+)(?:,\s*\{(?P<send_list>[^\}]+)\})?)?$"
)

_ALLOC_ID_OFFSET_RE = re.compile(r"#allocation(?P<alloc_id>\d+)(?:,\s*offset:\s*(?P<offset>\d+)\s*bytes)?")
_LEVEL0_RE = re.compile(r"level:0\s*\{(?P<lvl0>[^\}]+)\}")

# 5a. Hardware `IciSdcTestEmitter` compile-time config line:
#     `IciSdcTestEmitter :: Emitting code for buffer_size_chunks:1024 ... max_distance:1 pipeline_depth:13 ...`
_EMITTER_CFG_HW_RE = re.compile(
    r"IciSdcTestEmitter\s*::\s*Emitting code for\s+.*?" r"max_distance:(?P<d>\d+)\s+pipeline_depth:(?P<D>\d+)"
)

# 5b. Key-value `IciSdcTestEmitter` config line:
#     `IciSdcTestEmitter: max_distance=1, pipeline_depth=13, send_timeout_cycles=6250000, recv_timeout_cycles=6250000`
_EMITTER_CFG_KV_RE = re.compile(
    r"IciSdcTestEmitter:\s*max_distance=(?P<d>\d+),\s*"
    r"pipeline_depth=(?P<D>\d+)"
    r"(?:,\s*send_timeout_cycles=(?P<stc>\d+),\s*recv_timeout_cycles=(?P<rtc>\d+))?"
)

# 6a. Per-device `DEVICE_MAP:` line:
#     `DEVICE_MAP: [host=0] id=0 process_index=0 coords=(0, 0, 0) core_on_chip=0`
#     `DEVICE_MAP: idx=0 id=0 process_index=0 coords=[0, 0, 0] core_on_chip=0 kind=TPU7x`
_DEVICE_MAP_SINGLE_RE = re.compile(
    r"DEVICE_MAP:\s*(?:\[host=(?P<host_id>[^\]]+)\]\s*)?"
    r"(?:idx=\d+\s+)?id=(?P<g>\d+)\s+process_index=(?P<proc>\d+)\s+"
    r"coords=[\(\[](?P<x>\d+),\s*(?P<y>\d+),\s*(?P<z>\d+)[\)\]]\s+"
    r"core_on_chip=(?P<tc>\d+)"
)

# 6b. Semicolon-separated `DEVICE_MAP:` line:
#     `DEVICE_MAP: 0:0,0,0,0,0;1:0,0,0,1,0;...`
_DEVICE_MAP_LIST_RE = re.compile(r"DEVICE_MAP:\s*(?P<entries>[0-9,:;]+)")
_DEVICE_MAP_ENTRY_RE = re.compile(r"(?P<g>\d+):(?P<x>\d+),(?P<y>\d+),(?P<z>\d+),(?P<tc>\d+),(?P<proc>\d+)")

# 7. Environment line (`ENV: [host=0] LIBTPU_INIT_ARGS=...`):
_ENV_LINE_RE = re.compile(r"ENV:\s*(?:\[host=(?P<host_id>[^\]]+)\]\s*)?(?P<key>[A-Za-z0-9_]+)=(?P<val>.*)$")

# 8. Serialized SDK callback line (`SDK_CALLBACK_EVENT: ...`):
_SDK_CALLBACK_META_PAYLOAD_RE = re.compile(
    r"SDK_CALLBACK_EVENT:\s*"
    r"(?:process_index=(?P<proc>\d+)\s+)?"
    r"(?:phase=(?P<phase>\S+)\s+)?"
    r"(?:step=(?P<step>\d+)\s+)?"
    r"(?:idx_in_phase=\d+\s+)?"
    r"metadata=(?P<meta>\{.*?\})\s+payload=(?P<payload>\{.*\})\s*$"
)
_SDK_CALLBACK_LINE_RE = re.compile(
    r"SDK_CALLBACK_EVENT:\s*(?:process_index=(?P<proc>\d+)\s+.*?\bmetadata=)?(?P<payload>\{.*\})\s*$"
)
_NDARRAY_VALS_RE = re.compile(r"val=\[(?P<vals>[0-9,\s-]+)\]")


def parse_coord_details(
    coord: str | None,
) -> tuple[tuple[int, int, int] | None, int | None, bool, bool]:
  """Extracts `(chip_xyz, tc_idx, is_2d_format, has_tc_suffix)` from a core label."""
  if not coord:
    return None, None, False, False
  s = coord.strip()
  for pat in (_COORD_3D_RE, _COORD_TUPLE_RE):
    m = pat.match(s)
    if m:
      x = int(m.group("x"))
      y = int(m.group("y"))
      z_s = m.group("z")
      z = int(z_s) if z_s is not None else 0
      tc_s = m.group("tc")
      tc = int(tc_s) if tc_s is not None else 0
      return (x, y, z), tc, (pat is _COORD_TUPLE_RE and z_s is None), (tc_s is not None)
  return None, None, False, False


def parse_coord_parts(coord: str | None) -> tuple[tuple[int, int, int] | None, int | None]:
  """Extracts `(x, y, z)` chip coordinates and `tc_idx` from a core label."""
  chip, tc, _, _ = parse_coord_details(coord)
  return chip, tc


def format_coord(
    x: int,
    y: int,
    z: int,
    tc: int,
    is_2d_format: bool = False,
    has_tc_suffix: bool = True,
) -> str:
  """Formats `(x, y, z, tc)` into `XxYxZ_TCc` (or `(X,Y)` when `is_2d_format=True`)."""
  base = f"({x},{y})" if is_2d_format else f"{x}x{y}x{z}"
  return f"{base}_TC{tc}" if has_tc_suffix else base


def normalize_core_str(raw: str | None) -> str | None:
  """Normalizes whitespace inside tuple coordinates like `(0, 1)` -> `(0,1)`."""
  if not raw:
    return None
  s = raw.strip()
  chip, tc, is_2d, has_tc = parse_coord_details(s)
  if chip is not None and tc is not None:
    return format_coord(chip[0], chip[1], chip[2], tc, is_2d_format=is_2d, has_tc_suffix=has_tc)
  return s


def normalize_alloc_label(raw_alloc: str) -> tuple[str, int | None, int | None]:
  """Normalizes an sflag allocation descriptor and extracts `(alloc_id, offset)`."""
  cleaned = raw_alloc.strip().strip("[]").strip()
  m = _ALLOC_ID_OFFSET_RE.search(cleaned)
  if not m:
    return cleaned, None, None
  alloc_id = int(m.group("alloc_id"))
  offset = int(m.group("offset")) if m.group("offset") is not None else None
  if offset is not None:
    return f"#allocation{alloc_id}@{offset}", alloc_id, offset
  return f"#allocation{alloc_id}", alloc_id, None


@dataclass
class Event:
  """Unified parsed event across Approach-1 ICI-SDC and Approach-2 sflag logs."""

  kind: str
  file: str
  line_no: int
  raw_line: str
  host: str = ""
  coord: str | None = None
  chip_coord: tuple[int, int, int] | None = None
  tc_idx: int | None = None
  run_id: int | None = None
  norm_run_id: int | None = None
  launch_id: str | None = None
  global_idx: int | None = None

  # Approach-1 transfer timeout fields
  direction: str | None = None
  src_coord: str | None = None
  dst_coord: str | None = None
  src_global_idx: int | None = None
  dst_global_idx: int | None = None
  bytes: int | None = None
  iteration: int | None = None

  # Approach-1 completion / delay_iterations fields
  bw_gibs: float | None = None
  time_ms: float | None = None
  delay_p: int | None = None

  # Approach-2 sflag wait timeout fields
  alloc_label: str | None = None
  alloc_id: int | None = None
  offset: int | None = None
  for_val: int | None = None
  timeout_ms: int | None = None
  latest_val: int | None = None
  hlo_op: str | None = None
  meta_id: int | None = None
  level0: str | None = None
  unfinished_sends: int | None = None
  truncated_eof: bool = False

  # Emitter configuration fields
  pipeline_depth: int | None = None
  max_distance: int | None = None
  send_timeout_cycles: int | None = None
  recv_timeout_cycles: int | None = None

  # Device map entries: tuple of (global_idx, x, y, z, tc, process_index)
  device_map_entries: tuple[tuple[int, int, int, int, int, int], ...] | None = None


@dataclass
class ParsedLogSet:
  """Container of parsed `Event` records and cross-host normalization maps."""

  events: list[Event] = field(default_factory=list)
  files_scanned: list[str] = field(default_factory=list)
  core_to_global: dict[str, int] = field(default_factory=dict)
  global_to_core: dict[int, str] = field(default_factory=dict)
  core_to_host: dict[str, str] = field(default_factory=dict)
  run_id_offsets: dict[str, int] = field(default_factory=dict)
  env_vars: dict[str, str] = field(default_factory=dict)


def _infer_host_from_line_or_file(line: str, filepath: str, default_host: str = "") -> str:
  """Infers host identifier from `[host=N]` or filename `host_NN_...`."""
  m_host = _HOST_PREFIX_RE.search(line)
  if m_host:
    h_val = m_host.group("host").strip()
    return f"host_{int(h_val):02d}" if h_val.isdigit() else h_val
  if default_host:
    return default_host
  base = os.path.basename(filepath)
  m_file = re.match(r"^(?:host|pod)_(\d+)_", base)
  if m_file:
    return f"host_{int(m_file.group(1)):02d}"
  m_worker = re.search(r"(?:worker|task)[_-]?(\d+)", base)
  if m_worker:
    return f"host_{int(m_worker.group(1)):02d}"
  return ""


def _extract_core_context(line: str) -> tuple[str | None, int | None, str | None, str]:
  """Extracts `(coord, run_id, launch_id, job_host)` from bracketed core prefix."""
  for m in _CORE_BRACKET_RE.finditer(line):
    raw_coord = m.group("coord")
    if raw_coord.startswith("host="):
      continue
    coord = normalize_core_str(raw_coord)
    run_id = int(m.group("run_id")) if m.group("run_id") is not None else None
    launch_id = m.group("launch_id")
    job_host = m.group("job") or ""
    return coord, run_id, launch_id, job_host
  return None, None, None, ""


def parse_line(
    line: str,
    file: str = "<memory>",
    line_no: int = 1,
    default_host: str = "",
) -> list[Event]:
  """Parses a single raw log line into zero or more `Event` objects."""
  stripped = line.strip()
  if not stripped:
    return []

  # Handle serialized SDK callback lines first (`SDK_CALLBACK_EVENT: ...`).
  if "SDK_CALLBACK_EVENT:" in stripped:
    m_mp = _SDK_CALLBACK_META_PAYLOAD_RE.search(stripped)
    if m_mp:
      try:
        meta_d = ast.literal_eval(m_mp.group("meta"))
        pay_d = ast.literal_eval(m_mp.group("payload"))
        if isinstance(meta_d, dict) and isinstance(pay_d, dict):
          combined = dict(meta_d)
          combined["payload"] = pay_d
          if m_mp.group("step") is not None:
            combined.setdefault("step", int(m_mp.group("step")))
          cb_host = f"host_{int(m_mp.group('proc')):02d}" if m_mp.group("proc") is not None else default_host
          return parse_sdk_callback_event(combined, file=file, line_no=line_no, default_host=cb_host)
      except (ValueError, SyntaxError):
        pass
    m_cb = _SDK_CALLBACK_LINE_RE.search(stripped)
    if m_cb:
      try:
        payload = ast.literal_eval(m_cb.group("payload"))
        if isinstance(payload, dict):
          cb_host = f"host_{int(m_cb.group('proc')):02d}" if m_cb.group("proc") is not None else default_host
          return parse_sdk_callback_event(payload, file=file, line_no=line_no, default_host=cb_host)
      except (ValueError, SyntaxError):
        pass

  coord, run_id, launch_id, job_host = _extract_core_context(stripped)
  host = _infer_host_from_line_or_file(stripped, file, default_host=default_host) or job_host
  chip_coord, tc_idx = parse_coord_parts(coord)

  # 1a. Hardware non-compact ICI transfer timeout (`ICI transfer wait on SEND|RECV flag timed out on ...`)
  if "ICI transfer wait on" in stripped and "[XLA_SDC_CHECKER]" not in stripped:
    m_hw = _XFER_TIMEOUT_HW_RE.search(stripped)
    if m_hw:
      direction = m_hw.group("dir").upper()
      body_core = normalize_core_str(m_hw.group("core"))
      eff_coord = coord or body_core
      eff_chip, eff_tc = parse_coord_parts(eff_coord)
      g_idx = int(m_hw.group("g")) if m_hw.group("g") is not None else None
      return [
          Event(
              kind="XFER_TIMEOUT",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=eff_coord,
              chip_coord=eff_chip,
              tc_idx=eff_tc,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              global_idx=g_idx,
              direction=direction,
              src_coord=eff_coord if direction == "SEND" else None,
              dst_coord=eff_coord if direction == "RECV" else None,
              iteration=int(m_hw.group("iter")),
          )
      ]

  # 1b & 2. `ICI SDC Test transfer ...` (pair or compact dynamic)
  if "ICI SDC Test transfer" in stripped:
    m_pair = _XFER_TIMEOUT_PAIR_RE.search(stripped)
    if m_pair:
      direction = m_pair.group("dir").upper()
      src_coord = normalize_core_str(m_pair.group("src"))
      dst_coord = normalize_core_str(m_pair.group("dst"))
      eff_coord = coord or (src_coord if direction == "SEND" else dst_coord)
      eff_chip, eff_tc = parse_coord_parts(eff_coord)
      return [
          Event(
              kind="XFER_TIMEOUT",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=eff_coord,
              chip_coord=eff_chip,
              tc_idx=eff_tc,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              direction=direction,
              src_coord=src_coord,
              dst_coord=dst_coord,
              bytes=int(m_pair.group("bytes")),
              iteration=int(m_pair.group("iter")),
          )
      ]

    m_comp_pair = _XFER_TIMEOUT_COMPACT_PAIR_RE.search(stripped)
    if m_comp_pair:
      direction = m_comp_pair.group("dir").upper()
      src_g = int(m_comp_pair.group("src_g"))
      dst_g = int(m_comp_pair.group("dst_g"))
      eff_g = src_g if direction == "SEND" else dst_g
      return [
          Event(
              kind="XFER_TIMEOUT_COMPACT",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=coord,
              chip_coord=chip_coord,
              tc_idx=tc_idx,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              global_idx=eff_g,
              direction=direction,
              src_coord=coord if direction == "SEND" else None,
              dst_coord=coord if direction == "RECV" else None,
              src_global_idx=src_g,
              dst_global_idx=dst_g,
              bytes=int(m_comp_pair.group("bytes")),
              iteration=int(m_comp_pair.group("iter")),
          )
      ]

    m_comp_hw = _XFER_TIMEOUT_COMPACT_HW_RE.search(stripped)
    if m_comp_hw:
      direction = m_comp_hw.group("dir").upper()
      g_idx = int(m_comp_hw.group("g")) if m_comp_hw.group("g") is not None else None
      eff_coord = coord or (f"core_{g_idx}" if g_idx is not None else None)
      eff_chip, eff_tc = parse_coord_parts(eff_coord)
      return [
          Event(
              kind="XFER_TIMEOUT_COMPACT",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=eff_coord,
              chip_coord=eff_chip,
              tc_idx=eff_tc,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              global_idx=g_idx,
              direction=direction,
              src_coord=eff_coord if direction == "SEND" else None,
              dst_coord=eff_coord if direction == "RECV" else None,
              src_global_idx=g_idx if direction == "SEND" else None,
              dst_global_idx=g_idx if direction == "RECV" else None,
              iteration=int(m_comp_hw.group("iter")),
          )
      ]

  # 3. ICI BW & delay iterations (P)
  if "IciSdcTest" in stripped and "ICI BW:" in stripped:
    m_delay_hw = _DELAY_ITER_HW_RE.search(stripped)
    if m_delay_hw:
      eff_coord = coord or normalize_core_str(m_delay_hw.group("core"))
      eff_chip, eff_tc = parse_coord_parts(eff_coord)
      g_idx = int(m_delay_hw.group("g")) if m_delay_hw.group("g") is not None else None
      return [
          Event(
              kind="DELAY_ITER",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=eff_coord,
              chip_coord=eff_chip,
              tc_idx=eff_tc,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              global_idx=g_idx,
              bw_gibs=float(m_delay_hw.group("bw")),
              time_ms=float(m_delay_hw.group("ms")),
              delay_p=int(m_delay_hw.group("P")),
          )
      ]
    m_delay_short = _DELAY_ITER_SHORT_RE.search(stripped)
    if m_delay_short:
      return [
          Event(
              kind="DELAY_ITER",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              coord=coord,
              chip_coord=chip_coord,
              tc_idx=tc_idx,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              global_idx=int(m_delay_short.group("g")),
              bw_gibs=float(m_delay_short.group("bw")),
              time_ms=float(m_delay_short.group("ms")),
              delay_p=(int(m_delay_short.group("P")) if m_delay_short.group("P") is not None else 0),
          )
      ]

  # 4. Approach-2 sflag wait/short timeout
  if "timeout on sflag" in stripped:
    cleaned = re.sub(r"===\s*TAIL\s+.*$", "", stripped).strip()
    truncated = cleaned != stripped or ("program tracker" in cleaned and "unfinished send sflags" not in cleaned)
    m_sf = _SFLAG_TIMEOUT_RE.search(cleaned)
    if m_sf:
      alloc_label, alloc_id, offset = normalize_alloc_label(m_sf.group("alloc"))
      levels_str = (m_sf.group("levels") or "").strip()
      m_lvl0 = _LEVEL0_RE.search(levels_str)
      level0 = m_lvl0.group("lvl0") if m_lvl0 else None
      return [
          Event(
              kind="SFLAG_TIMEOUT",
              file=file,
              line_no=line_no,
              raw_line=cleaned,
              host=host,
              coord=coord,
              chip_coord=chip_coord,
              tc_idx=tc_idx,
              run_id=run_id,
              norm_run_id=run_id,
              launch_id=launch_id,
              alloc_label=alloc_label,
              alloc_id=alloc_id,
              offset=offset,
              for_val=int(m_sf.group("for_val")),
              timeout_ms=int(m_sf.group("timeout_ms")),
              latest_val=int(m_sf.group("latest_val")),
              hlo_op=m_sf.group("hlo").strip(),
              meta_id=int(m_sf.group("meta_id")) if m_sf.group("meta_id") is not None else None,
              level0=level0,
              unfinished_sends=(int(m_sf.group("unfinished")) if m_sf.group("unfinished") is not None else None),
              truncated_eof=truncated,
          )
      ]

  # 5. Emitter configuration
  if "IciSdcTestEmitter" in stripped:
    m_cfg_hw = _EMITTER_CFG_HW_RE.search(stripped)
    if m_cfg_hw:
      return [
          Event(
              kind="EMITTER_CFG",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              max_distance=int(m_cfg_hw.group("d")),
              pipeline_depth=int(m_cfg_hw.group("D")),
          )
      ]
    m_cfg_kv = _EMITTER_CFG_KV_RE.search(stripped)
    if m_cfg_kv:
      return [
          Event(
              kind="EMITTER_CFG",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              max_distance=int(m_cfg_kv.group("d")),
              pipeline_depth=int(m_cfg_kv.group("D")),
              send_timeout_cycles=(int(m_cfg_kv.group("stc")) if m_cfg_kv.group("stc") is not None else None),
              recv_timeout_cycles=(int(m_cfg_kv.group("rtc")) if m_cfg_kv.group("rtc") is not None else None),
          )
      ]

  # 6. Device map
  if "DEVICE_MAP:" in stripped:
    m_dm_single = _DEVICE_MAP_SINGLE_RE.search(stripped)
    if m_dm_single:
      entry = (
          int(m_dm_single.group("g")),
          int(m_dm_single.group("x")),
          int(m_dm_single.group("y")),
          int(m_dm_single.group("z")),
          int(m_dm_single.group("tc")),
          int(m_dm_single.group("proc")),
      )
      return [
          Event(
              kind="DEVICE_MAP",
              file=file,
              line_no=line_no,
              raw_line=stripped,
              host=host,
              device_map_entries=(entry,),
          )
      ]
    m_dm_list = _DEVICE_MAP_LIST_RE.search(stripped)
    if m_dm_list:
      entries: list[tuple[int, int, int, int, int, int]] = []
      for m_ent in _DEVICE_MAP_ENTRY_RE.finditer(m_dm_list.group("entries")):
        entries.append(
            (
                int(m_ent.group("g")),
                int(m_ent.group("x")),
                int(m_ent.group("y")),
                int(m_ent.group("z")),
                int(m_ent.group("tc")),
                int(m_ent.group("proc")),
            )
        )
      if entries:
        return [
            Event(
                kind="DEVICE_MAP",
                file=file,
                line_no=line_no,
                raw_line=stripped,
                host=host,
                device_map_entries=tuple(entries),
            )
        ]

  return []


def _interpolate_user_tag(user_tag: str, payload: dict[str, Any]) -> str:
  """Substitutes `$0`, `$1`, ... placeholders in `user_tag` using `payload`."""
  if "$" not in user_tag:
    return user_tag
  vals: list[int] = []
  lit = str(payload.get("literal", ""))
  m_v = _NDARRAY_VALS_RE.search(lit)
  if m_v:
    vals = [int(x.strip()) for x in m_v.group("vals").split(",") if x.strip()]
  elif "expected_sflag_value" in payload and "latest_sflag_value_seen" in payload:
    vals = [
        int(payload["expected_sflag_value"]),
        int(payload["latest_sflag_value_seen"]),
        int(payload.get("progress_tracker_current_depth", 1)),
        int(payload.get("progress_tracker_meta_id", -1)),
    ]
  elif "global_core_id" in payload and "iteration" in payload:
    vals = [int(payload["global_core_id"]), int(payload["iteration"])]

  if not vals:
    return user_tag

  def _repl(m: re.Match[str]) -> str:
    idx = int(m.group(1))
    return str(vals[idx]) if idx < len(vals) else m.group(0)

  return re.sub(r"\$(\d+)", _repl, user_tag)


def parse_sdk_callback_event(
    record: dict[str, Any],
    file: str = "<callback>",
    line_no: int = 1,
    default_host: str = "",
) -> list[Event]:
  """Parses a `libtpu.sdk.debug` event_log callback dict into `Event` objects."""
  meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else record
  payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
  msg = str(meta.get("message") or meta.get("user_tag") or record.get("message") or "")
  if not msg:
    return []
  msg = _interpolate_user_tag(msg, payload)

  chip_raw = meta.get("chip") or record.get("chip")
  core_raw = meta.get("core") if meta.get("core") is not None else record.get("core")
  core_loc = meta.get("core_location") or record.get("core_location")
  step_val = record.get("step") if isinstance(record.get("step"), int) else None
  fallback_coord: str | None = None
  if isinstance(core_loc, str) and core_loc.strip():
    fallback_coord = normalize_core_str(core_loc)
  elif isinstance(chip_raw, (tuple, list)) and len(chip_raw) == 3 and isinstance(core_raw, int):
    fallback_coord = format_coord(int(chip_raw[0]), int(chip_raw[1]), int(chip_raw[2]), int(core_raw))

  evs = parse_line(msg, file=file, line_no=line_no, default_host=default_host)
  for ev in evs:
    if fallback_coord and (ev.coord is None or ev.coord.startswith("core_")):
      ev.coord = fallback_coord
      ev.chip_coord, ev.tc_idx = parse_coord_parts(fallback_coord)
    if step_val is not None and ev.run_id is None:
      ev.run_id = step_val
      ev.norm_run_id = step_val
  return evs


def normalize_parsed_log_set(parsed: ParsedLogSet) -> ParsedLogSet:
  """Resolves `G <-> XxYxZ_TCc` bijections and normalizes `@run_id` offsets."""
  for ev in parsed.events:
    if ev.kind == "DEVICE_MAP" and ev.device_map_entries:
      for g, x, y, z, tc, proc in ev.device_map_entries:
        c = format_coord(x, y, z, tc)
        parsed.core_to_global[c] = g
        parsed.global_to_core[g] = c
        if c not in parsed.core_to_host:
          parsed.core_to_host[c] = f"host_{proc:02d}"
    elif ev.kind in ("DELAY_ITER", "XFER_TIMEOUT", "XFER_TIMEOUT_COMPACT"):
      if ev.coord and not ev.coord.startswith("core_") and ev.global_idx is not None:
        parsed.core_to_global[ev.coord] = ev.global_idx
        parsed.global_to_core[ev.global_idx] = ev.coord
    if ev.coord and not ev.coord.startswith("core_") and ev.host and ev.coord not in parsed.core_to_host:
      parsed.core_to_host[ev.coord] = ev.host

  for ev in parsed.events:
    if ev.kind == "XFER_TIMEOUT_COMPACT":
      if (ev.src_coord is None or ev.src_coord.startswith("core_")) and ev.src_global_idx is not None:
        ev.src_coord = parsed.global_to_core.get(ev.src_global_idx, f"core_{ev.src_global_idx}")
      if (ev.dst_coord is None or ev.dst_coord.startswith("core_")) and ev.dst_global_idx is not None:
        ev.dst_coord = parsed.global_to_core.get(ev.dst_global_idx, f"core_{ev.dst_global_idx}")
      if ev.coord is None or ev.coord.startswith("core_"):
        if ev.global_idx is not None and ev.global_idx in parsed.global_to_core:
          ev.coord = parsed.global_to_core[ev.global_idx]
        elif ev.direction == "SEND":
          ev.coord = ev.src_coord
        else:
          ev.coord = ev.dst_coord
      if ev.coord and ev.chip_coord is None:
        ev.chip_coord, ev.tc_idx = parse_coord_parts(ev.coord)
    elif ev.kind in ("XFER_TIMEOUT", "DELAY_ITER"):
      if ev.coord and ev.global_idx is None:
        ev.global_idx = parsed.core_to_global.get(ev.coord)
      if ev.src_coord and ev.src_global_idx is None:
        ev.src_global_idx = parsed.core_to_global.get(ev.src_coord)
      if ev.dst_coord and ev.dst_global_idx is None:
        ev.dst_global_idx = parsed.core_to_global.get(ev.dst_coord)

  min_delay_run: dict[str, int] = {}
  for ev in parsed.events:
    if ev.kind == "DELAY_ITER" and ev.coord and ev.run_id is not None:
      prev = min_delay_run.get(ev.coord)
      if prev is None or ev.run_id < prev:
        min_delay_run[ev.coord] = ev.run_id

  offsets: dict[str, int] = {}
  if len(min_delay_run) >= 2:
    base_run = min(min_delay_run.values())
    offsets = {c: r - base_run for c, r in min_delay_run.items() if r - base_run > 0}
  else:
    min_sflag_run: dict[str, int] = {}
    for ev in parsed.events:
      if (
          ev.kind == "SFLAG_TIMEOUT"
          and ev.coord
          and ev.run_id is not None
          and not ev.launch_id
          and ev.hlo_op
          and ev.hlo_op != "<no-hlo-instruction>"
      ):
        prev = min_sflag_run.get(ev.coord)
        if prev is None or ev.run_id < prev:
          min_sflag_run[ev.coord] = ev.run_id
    if len(min_sflag_run) >= 2:
      counts_by_min: dict[int, int] = {}
      for r in min_sflag_run.values():
        counts_by_min[r] = counts_by_min.get(r, 0) + 1
      mode_min = max(counts_by_min.items(), key=lambda kv: (kv[1], -kv[0]))[0]
      offsets = {c: r - mode_min for c, r in min_sflag_run.items() if r > mode_min}

  parsed.run_id_offsets = offsets
  for ev in parsed.events:
    if ev.run_id is not None and ev.coord:
      ev.norm_run_id = ev.run_id - offsets.get(ev.coord, 0)
    else:
      ev.norm_run_id = ev.run_id

  return parsed


def parse_lines(
    lines: Iterable[str],
    file: str = "<memory>",
    default_host: str = "",
    normalize: bool = True,
) -> ParsedLogSet:
  """Parses an iterable of log lines into a `ParsedLogSet`."""
  parsed = ParsedLogSet(files_scanned=[file])
  for idx, line in enumerate(lines, start=1):
    m_env = _ENV_LINE_RE.search(line)
    if m_env:
      parsed.env_vars[m_env.group("key")] = m_env.group("val").strip()
    evs = parse_line(line, file=file, line_no=idx, default_host=default_host)
    if evs:
      parsed.events.extend(evs)
  if normalize:
    normalize_parsed_log_set(parsed)
  return parsed


def _collect_dir_log_files(root: str, s_files: list[str]) -> list[str]:
  """Collects log files from a single directory, deduplicating `host_NN_` retries."""
  skip_exts = (".py", ".sh", ".yaml", ".yml", ".md", ".json", ".pyc")
  skip_names = {"SUMMARY.txt", "README.md", "BUILD"}
  host_logs = [fn for fn in s_files if fn.startswith("host_") and fn.endswith(".log")]
  if host_logs:
    by_host_id: dict[int, list[tuple[float, str]]] = {}
    other_logs: list[str] = []
    for fn in host_logs:
      full = os.path.join(root, fn)
      if not os.path.isfile(full):
        continue
      m_h = re.match(r"^host_(\d+)_", fn)
      if m_h:
        hid = int(m_h.group(1))
        by_host_id.setdefault(hid, []).append((os.path.getmtime(full), full))
      else:
        other_logs.append(full)
    out = [sorted(by_host_id[hid])[-1][1] for hid in sorted(by_host_id)]
    out.extend(other_logs)
    return out

  out_files: list[str] = []
  for fn in s_files:
    if fn in skip_names or fn.endswith(skip_exts) or fn.endswith("_combined.log"):
      continue
    full = os.path.join(root, fn)
    if os.path.isfile(full):
      out_files.append(full)
  return out_files


def collect_log_files(paths: Sequence[str | os.PathLike[str]]) -> list[str]:
  """Collects log files from file/directory paths, deduplicating multi-pod retries."""
  collected: list[str] = []
  for p_raw in paths:
    p = os.fspath(p_raw)
    if os.path.isfile(p):
      collected.append(p)
    elif os.path.isdir(p):
      for root, dirs, files in os.walk(p):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", "raw_pod_logs"))
        collected.extend(_collect_dir_log_files(root, sorted(files)))
  return collected


def parse_paths(paths: Sequence[str | os.PathLike[str]]) -> ParsedLogSet:
  """Parses all log files across the given file or directory paths."""
  files = collect_log_files(paths)
  parsed = ParsedLogSet(files_scanned=files)
  for filepath in files:
    default_host = _infer_host_from_line_or_file("", filepath)
    try:
      with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
          m_env = _ENV_LINE_RE.search(line)
          if m_env:
            parsed.env_vars[m_env.group("key")] = m_env.group("val").strip()
          evs = parse_line(line, file=filepath, line_no=idx, default_host=default_host)
          if evs:
            parsed.events.extend(evs)
    except OSError:
      continue
  return normalize_parsed_log_set(parsed)
