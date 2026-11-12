# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-grained stage recorder, jax.monitoring listener, barrier/RAM/HBM trackers, and multi-host merger."""

from collections.abc import Callable, Sequence
import contextlib
import datetime
import functools
import json
import os
import re
import socket
import statistics
import threading
import time
from typing import Any

STAGES_LINE_PREFIX = "BENCH_STAGES_JSON "

DURATION_MONITORING_EVENTS: tuple[str, ...] = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
    "/jax/orbax/read/blocking_duration_secs",
    "/jax/orbax/read/total_duration_secs",
    "/jax/orbax/read/worker/total_duration_secs",
    "/jax/orbax/write/blocking_d2h_duration_secs",
    "/jax/orbax/write/blocking_duration_secs",
    "/jax/orbax/write/background_ts_duration_secs",
    "/jax/orbax/write/background_ocdbt_merge_secs",
    "/jax/orbax/write/background_meta_write_duration_secs",
    "/jax/checkpoint/sync_global_devices_duration_sec",
)

SCALAR_MONITORING_EVENTS: tuple[str, ...] = (
    "/jax/orbax/read/safetensors/bytes_read",
    "/jax/orbax/read/safetensors/num_reads",
    "/jax/orbax/read/safetensors/storage_reads",
    "/jax/orbax/read/worker/io/requested/gbytes",
    "/jax/orbax/write/gbytes",
)


class StageRecorder:
  """Thread-safe stage timer supporting inclusive/exclusive durations, wrappers, and custom metrics."""

  def __init__(self, clock: Callable[[], float] | None = None, wall_clock: Callable[[], float] | None = None):
    self._clock = clock or time.perf_counter
    self._wall_clock = wall_clock or time.time
    self._lock = threading.RLock()
    self._local = threading.local()
    self._stages: dict[str, dict[str, Any]] = {}
    self._barriers: dict[str, dict[str, Any]] = {}
    self._patches: list[tuple[Any, str, Any]] = []
    self._active_wrappers: list[str] = []
    self._last_entered_stage: str | None = None

  def _stack(self) -> list[dict[str, Any]]:
    stack = getattr(self._local, "stack", None)
    if stack is None:
      stack = []
      self._local.stack = stack
    return stack

  def current_stage(self) -> str | None:
    """Returns the currently executing top-of-stack stage for this thread or last entered global stage."""
    stack = self._stack()
    if stack:
      return stack[-1]["name"]
    return self._last_entered_stage

  def record_value(self, name: str, inclusive_s: float, exclusive_s: float | None = None) -> None:
    """Accumulates a pre-measured stage duration directly."""
    excl = inclusive_s if exclusive_s is None else exclusive_s
    now_wall = self._wall_clock()
    with self._lock:
      entry = self._stages.setdefault(
          name,
          {
              "inclusive_s": 0.0,
              "exclusive_s": 0.0,
              "count": 0,
              "first_start_epoch_s": now_wall,
              "last_end_epoch_s": now_wall,
              "metrics": {},
          },
      )
      entry["inclusive_s"] += float(inclusive_s)
      entry["exclusive_s"] += float(excl)
      entry["count"] += 1
      entry["last_end_epoch_s"] = now_wall

  def add_metric(self, stage_name: str, key: str, value: float | int) -> None:
    """Adds a numeric counter or byte total onto `stage_name`."""
    with self._lock:
      entry = self._stages.setdefault(
          stage_name,
          {
              "inclusive_s": 0.0,
              "exclusive_s": 0.0,
              "count": 0,
              "first_start_epoch_s": None,
              "last_end_epoch_s": None,
              "metrics": {},
          },
      )
      metrics = entry["metrics"]
      metrics[key] = metrics.get(key, 0) + value

  def record_barrier(self, barrier_name: str, duration_s: float) -> None:
    """Records one named global barrier wait."""
    with self._lock:
      b = self._barriers.setdefault(barrier_name, {"total_s": 0.0, "count": 0, "max_s": 0.0})
      b["total_s"] += float(duration_s)
      b["count"] += 1
      b["max_s"] = max(b["max_s"], float(duration_s))

  @contextlib.contextmanager
  def stage(self, name: str, *, log_start: bool = False):
    """Times execution inside `with recorder.stage(name):`, deducting nested child durations."""
    stack = self._stack()
    t0 = self._clock()
    w0 = self._wall_clock()
    frame = {"name": name, "child_s": 0.0}
    stack.append(frame)
    prev_global = self._last_entered_stage
    if threading.current_thread() is threading.main_thread():
      self._last_entered_stage = name
    if log_start:
      print(f"[STAGE START] {name}", flush=True)
    try:
      yield
    finally:
      dt = max(0.0, self._clock() - t0)
      w1 = self._wall_clock()
      stack.pop()
      if stack:
        stack[-1]["child_s"] += dt
      if threading.current_thread() is threading.main_thread():
        self._last_entered_stage = stack[-1]["name"] if stack else prev_global
      excl = max(0.0, dt - frame["child_s"])
      with self._lock:
        entry = self._stages.setdefault(
            name,
            {
                "inclusive_s": 0.0,
                "exclusive_s": 0.0,
                "count": 0,
                "first_start_epoch_s": w0,
                "last_end_epoch_s": w1,
                "metrics": {},
            },
        )
        entry["inclusive_s"] += dt
        entry["exclusive_s"] += excl
        entry["count"] += 1
        if entry["first_start_epoch_s"] is None:
          entry["first_start_epoch_s"] = w0
        entry["last_end_epoch_s"] = w1

  def wrap(
      self,
      target_obj: Any,
      attr_name: str,
      stage_name: str,
      *,
      reentrant_guard: bool = False,
      label: str | None = None,
  ) -> bool:
    """Safely monkeypatches `target_obj.attr_name` to execute inside `recorder.stage(stage_name)`."""
    if target_obj is None or not hasattr(target_obj, attr_name):
      return False
    orig = getattr(target_obj, attr_name)
    if getattr(orig, "__mesa_stage_wrapped__", False):
      return True
    wrapper_id = label or f"{getattr(target_obj, '__name__', type(target_obj).__name__)}.{attr_name}"
    state = threading.local()

    @functools.wraps(orig)
    def _wrapped(*args, **kwargs):
      if reentrant_guard and getattr(state, "depth", 0) > 0:
        return orig(*args, **kwargs)
      state.depth = getattr(state, "depth", 0) + 1
      try:
        with self.stage(stage_name):
          return orig(*args, **kwargs)
      finally:
        state.depth -= 1

    setattr(_wrapped, "__mesa_stage_wrapped__", True)
    setattr(target_obj, attr_name, _wrapped)
    with self._lock:
      self._patches.append((target_obj, attr_name, orig))
      if wrapper_id not in self._active_wrappers:
        self._active_wrappers.append(wrapper_id)
    return True

  def restore_all(self) -> None:
    """Restores all attributes patched via `wrap()`."""
    with self._lock:
      while self._patches:
        target_obj, attr_name, orig = self._patches.pop()
        setattr(target_obj, attr_name, orig)

  def snapshot(self) -> dict[str, Any]:
    """Returns a deep-copied snapshot of stages, barriers, and active wrapper names."""
    with self._lock:
      return {
          "stages": {k: {**v, "metrics": dict(v["metrics"])} for k, v in self._stages.items()},
          "barriers": {k: dict(v) for k, v in self._barriers.items()},
          "active_timing_wrappers": list(self._active_wrappers),
      }


def wrap_sync_global_devices(multihost_utils_mod: Any, recorder: StageRecorder) -> bool:
  """Times `multihost_utils.sync_global_devices(name)` calls by barrier name."""
  if multihost_utils_mod is None or not hasattr(multihost_utils_mod, "sync_global_devices"):
    return False
  orig = multihost_utils_mod.sync_global_devices
  if getattr(orig, "__mesa_barrier_wrapped__", False):
    return True

  @functools.wraps(orig)
  def _timed_barrier(name, *args, **kwargs):
    t0 = recorder._clock()  # pylint: disable=protected-access
    try:
      return orig(name, *args, **kwargs)
    finally:
      recorder.record_barrier(str(name), max(0.0, recorder._clock() - t0))  # pylint: disable=protected-access

  setattr(_timed_barrier, "__mesa_barrier_wrapped__", True)
  multihost_utils_mod.sync_global_devices = _timed_barrier
  with recorder._lock:  # pylint: disable=protected-access
    recorder._patches.append((multihost_utils_mod, "sync_global_devices", orig))  # pylint: disable=protected-access
    if "multihost_utils.sync_global_devices" not in recorder._active_wrappers:  # pylint: disable=protected-access
      recorder._active_wrappers.append("multihost_utils.sync_global_devices")  # pylint: disable=protected-access
  return True


class MonitoringCollector:
  """Collects JAX/Orbax `jax.monitoring` duration & scalar events bucketed by active stage."""

  def __init__(self, recorder: StageRecorder):
    self._recorder = recorder
    self._lock = threading.Lock()
    self._events: dict[str, dict[str, Any]] = {}
    self._stage_events: dict[str, dict[str, dict[str, Any]]] = {}
    self._monitoring_mod: Any = None
    self._registered_durations: list[tuple[str, Any]] = []
    self._registered_scalars: list[tuple[str, Any]] = []
    self._active_events: list[str] = []
    self._allowed_events: set[str] = set()
    self._duration_event_set: set[str] = set()

  def _on_event(self, event_name: str, value: float, **_metadata: Any) -> None:
    if self._allowed_events and event_name not in self._allowed_events:
      return
    fval = float(value)
    # Guard against JAX backends emitting an unsubtracted Unix epoch timestamp (~1.7e9) as a duration sample.
    if event_name in self._duration_event_set and (fval < 0.0 or fval > 1e7):
      return
    self._record(event_name, fval)

  def _record(self, event_name: str, value: float) -> None:
    """Thread-safely records a monitoring duration/scalar sample for the active stage."""
    stage_name = self._recorder.current_stage() or "unscoped"
    val = float(value)
    with self._lock:
      g = self._events.setdefault(event_name, {"sum": 0.0, "count": 0, "max": val})
      g["sum"] += val
      g["count"] += 1
      g["max"] = max(g["max"], val)

      st_bucket = self._stage_events.setdefault(stage_name, {})
      s = st_bucket.setdefault(event_name, {"sum": 0.0, "count": 0})
      s["sum"] += val
      s["count"] += 1

  def register(
      self,
      monitoring_mod: Any,
      duration_events: Sequence[str] = DURATION_MONITORING_EVENTS,
      scalar_events: Sequence[str] = SCALAR_MONITORING_EVENTS,
  ) -> list[str]:
    """Attaches listeners to `monitoring_mod` (`jax.monitoring`); returns tracked event names."""
    if monitoring_mod is None:
      return []
    self._monitoring_mod = monitoring_mod
    self._duration_event_set = set(duration_events)
    self._allowed_events = set(duration_events) | set(scalar_events)
    reg_dur = getattr(monitoring_mod, "register_event_duration_secs_listener", None)
    reg_scl = getattr(monitoring_mod, "register_scalar_listener", None)

    if callable(reg_dur):
      try:
        reg_dur(self._on_event)
        self._registered_durations.append(("", self._on_event))
        self._active_events.extend(duration_events)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    if callable(reg_scl):
      try:
        reg_scl(self._on_event)
        self._registered_scalars.append(("", self._on_event))
        self._active_events.extend(scalar_events)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return list(self._active_events)

  def unregister(self) -> None:
    """Detaches registered listeners when JAX exposes unregister helpers."""
    mod = self._monitoring_mod
    if mod is None:
      return
    self._allowed_events = set()
    unreg_dur = getattr(mod, "unregister_event_duration_secs_listener", None) or getattr(
        mod, "unregister_event_duration_listener", None
    )
    if callable(unreg_dur):
      for _, cb in self._registered_durations:
        try:
          unreg_dur(cb)  # pylint: disable=not-callable
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    self._registered_durations.clear()
    unreg_scl = getattr(mod, "unregister_scalar_listener", None)
    if callable(unreg_scl):
      for _, cb in self._registered_scalars:
        try:
          unreg_scl(cb)  # pylint: disable=not-callable
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    self._registered_scalars.clear()

  def snapshot(self) -> dict[str, Any]:
    """Returns copies of global and stage-bucketed monitoring counters."""
    with self._lock:
      return {
          "events": {k: dict(v) for k, v in self._events.items()},
          "by_stage": {st: {ev: dict(v) for ev, v in evs.items()} for st, evs in self._stage_events.items()},
          "registered_events": list(self._active_events),
      }


def read_process_rss_gb() -> float | None:
  """Reads `VmRSS` from `/proc/self/status` in GB (`10^9` bytes)."""
  try:
    with open("/proc/self/status", "r", encoding="utf-8") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          kb = int(line.split()[1])
          return round((kb * 1024) / 1e9, 4)
  except (OSError, ValueError):
    return None
  return None


def snapshot_hbm_gb(jax_mod: Any) -> dict[str, float] | None:
  """Reads local device `memory_stats()` in GB (`10^9` bytes); returns None on non-HBM backends."""
  if jax_mod is None:
    return None
  try:
    devices = jax_mod.local_devices()
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  usable_vals: list[float] = []
  used_vals: list[float] = []
  peak_vals: list[float] = []
  for dev in devices:
    stats_fn = getattr(dev, "memory_stats", None)
    if not callable(stats_fn):
      continue
    try:
      st = stats_fn()
    except Exception:  # pylint: disable=broad-exception-caught
      st = None
    if not isinstance(st, dict):
      continue
    if st.get("bytes_limit"):
      usable_vals.append(float(st["bytes_limit"]) / 1e9)
    if "bytes_in_use" in st:
      used_vals.append(float(st["bytes_in_use"]) / 1e9)
    if "peak_bytes_in_use" in st:
      peak_vals.append(float(st["peak_bytes_in_use"]) / 1e9)
  if not usable_vals and not used_vals and not peak_vals:
    return None
  usable_gb = max(usable_vals) if usable_vals else 0.0
  used_gb = max(used_vals) if used_vals else 0.0
  peak_gb = max(peak_vals) if peak_vals else used_gb
  pct = round((peak_gb / usable_gb) * 100.0, 2) if usable_gb > 0 else 0.0
  return {
      "usable_per_device_gb": round(usable_gb, 4),
      "used_per_device_gb": round(used_gb, 4),
      "peak_per_device_gb": round(peak_gb, 4),
      "peak_percent_of_usable": pct,
  }


def emit_host_stages_line(payload: dict[str, Any], stream: Any = None) -> str:
  """Serializes and prints one single-line `BENCH_STAGES_JSON ...` marker."""
  line = STAGES_LINE_PREFIX + json.dumps(payload, separators=(",", ":"), sort_keys=True)
  if stream is None:
    print(line, flush=True)
  else:
    stream.write(line + "\n")
    stream.flush()
  return line


_STAGES_RE = re.compile(r"^BENCH_STAGES_JSON\s+(\{.*\})\s*$", re.MULTILINE)


def parse_host_stages_lines(text: str) -> list[dict[str, Any]]:
  """Parses all `BENCH_STAGES_JSON {...}` lines out of worker or slice logs."""
  out: list[dict[str, Any]] = []
  for m in _STAGES_RE.finditer(text):
    try:
      parsed = json.loads(m.group(1))
      if isinstance(parsed, dict):
        out.append(parsed)
    except ValueError:
      continue
  return out


def _host_label(payload: dict[str, Any], fallback_idx: int) -> str:
  return str(payload.get("host_name") or payload.get("pod_name") or f"slice_{payload.get('slice_id', fallback_idx)}")


def merge_host_stages(host_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
  """Merges per-host `BENCH_STAGES_JSON` payloads into min/median/max/slowest-host summaries."""
  if not host_payloads:
    return {
        "host_count": 0,
        "stages_summary": {},
        "stages_headline_max": {},
        "barriers_summary": {},
        "monitoring_summary": {},
        "memory_summary": {},
        "active_timing_wrappers": [],
        "hosts": [],
    }

  all_stage_names: list[str] = []
  all_barrier_names: list[str] = []
  all_wrappers: list[str] = []
  all_event_names: list[str] = []

  for p in host_payloads:
    for st in p.get("stages", {}):
      if st not in all_stage_names:
        all_stage_names.append(st)
    for b in p.get("barriers", {}):
      if b not in all_barrier_names:
        all_barrier_names.append(b)
    for w in p.get("active_timing_wrappers", []):
      if w not in all_wrappers:
        all_wrappers.append(w)
    for ev in p.get("monitoring", {}).get("events", {}):
      if ev not in all_event_names:
        all_event_names.append(ev)

  stages_summary: dict[str, dict[str, Any]] = {}
  stages_headline_max: dict[str, float] = {}
  for st in all_stage_names:
    samples: list[tuple[float, str, int | None]] = []
    metric_totals: dict[str, list[float]] = {}
    for idx, p in enumerate(host_payloads):
      entry = p.get("stages", {}).get(st)
      if not isinstance(entry, dict):
        continue
      val = float(entry.get("inclusive_s", 0.0))
      samples.append((val, _host_label(p, idx), p.get("slice_id")))
      for mk, mv in entry.get("metrics", {}).items():
        if isinstance(mv, (int, float)) and not isinstance(mv, bool):
          metric_totals.setdefault(mk, []).append(float(mv))
    if not samples:
      continue
    vals = [s[0] for s in samples]
    slowest_val, slowest_host, slowest_sid = max(samples, key=lambda x: x[0])
    agg_metrics = {mk: {"sum": sum(mvs), "max": max(mvs)} for mk, mvs in metric_totals.items()}
    stages_summary[st] = {
        "min_s": min(vals),
        "median_s": statistics.median(vals),
        "max_s": slowest_val,
        "slowest_host": slowest_host,
        "slowest_slice_id": slowest_sid,
        "metrics": agg_metrics,
    }
    stages_headline_max[st] = slowest_val

  barriers_summary: dict[str, dict[str, Any]] = {}
  for b_name in all_barrier_names:
    b_samples: list[tuple[float, str]] = []
    for idx, p in enumerate(host_payloads):
      b_entry = p.get("barriers", {}).get(b_name)
      if isinstance(b_entry, dict):
        b_samples.append((float(b_entry.get("total_s", 0.0)), _host_label(p, idx)))
    if b_samples:
      b_vals = [x[0] for x in b_samples]
      max_b, max_b_host = max(b_samples, key=lambda x: x[0])
      barriers_summary[b_name] = {
          "min_s": min(b_vals),
          "median_s": statistics.median(b_vals),
          "max_s": max_b,
          "slowest_host": max_b_host,
      }

  monitoring_summary: dict[str, dict[str, float]] = {}
  for ev in all_event_names:
    ev_sums = [
        float(p.get("monitoring", {}).get("events", {}).get(ev, {}).get("sum", 0.0))
        for p in host_payloads
        if ev in p.get("monitoring", {}).get("events", {})
    ]
    if ev_sums:
      monitoring_summary[ev] = {
          "sum_over_hosts": sum(ev_sums),
          "max_over_hosts": max(ev_sums),
          "median_over_hosts": statistics.median(ev_sums),
      }

  ram_peaks = [
      (float(p["memory"]["peak_host_ram_container_gb"]), _host_label(p, idx))
      for idx, p in enumerate(host_payloads)
      if isinstance(p.get("memory", {}).get("peak_host_ram_container_gb"), (int, float))
  ]
  rss_peaks = [
      float(p["memory"]["peak_process_rss_gb"])
      for p in host_payloads
      if isinstance(p.get("memory", {}).get("peak_process_rss_gb"), (int, float))
  ]
  ram_sizes = [
      float(p["memory"]["host_ram_size_gb"])
      for p in host_payloads
      if isinstance(p.get("memory", {}).get("host_ram_size_gb"), (int, float))
  ]
  mem_summary: dict[str, Any] = {}
  if ram_peaks:
    max_ram, max_ram_host = max(ram_peaks, key=lambda x: x[0])
    mem_summary["memory_peak_host_ram_container_gb"] = round(max_ram, 2)
    mem_summary["memory_peak_host_ram_host_name"] = max_ram_host
  if rss_peaks:
    mem_summary["memory_peak_process_rss_gb"] = round(max(rss_peaks), 2)
  if ram_sizes:
    mem_summary["memory_host_ram_size_gb"] = round(max(ram_sizes), 2)

  for hbm_key in (
      "memory_tpu_hbm_usable_per_device_gb",
      "memory_tpu_hbm_used_after_read_per_device_gb",
      "memory_tpu_hbm_peak_per_device_gb",
      "memory_tpu_hbm_used_after_load_per_device_gb",
      "memory_tpu_hbm_peak_percent_of_usable",
  ):
    hbm_vals = [
        float(p["memory"][hbm_key]) for p in host_payloads if isinstance(p.get("memory", {}).get(hbm_key), (int, float))
    ]
    if hbm_vals:
      mem_summary[hbm_key] = round(max(hbm_vals), 2)

  return {
      "host_count": len(host_payloads),
      "stages_summary": stages_summary,
      "stages_headline_max": stages_headline_max,
      "barriers_summary": barriers_summary,
      "monitoring_summary": monitoring_summary,
      "memory_summary": mem_summary,
      "active_timing_wrappers": all_wrappers,
      "hosts": list(host_payloads),
  }


def utc_now_iso() -> str:
  """Returns current UTC ISO timestamp with Z suffix."""
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_host_name() -> str:
  return os.environ.get("POD_NAME") or socket.gethostname()
