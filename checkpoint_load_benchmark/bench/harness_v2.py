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

"""Phase 2 instrumentation helpers, memory samplers, checksums, and v2 JSON builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import datetime
import functools
import gc
import glob
import os
import re
import signal
import sys
import threading
import time
from typing import Any

_BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCH_DIR not in sys.path:
  sys.path.insert(0, _BENCH_DIR)

from bench import metadata  # pylint: disable=wrong-import-position
from bench import platforms  # pylint: disable=wrong-import-position
from bench import schema_v2  # pylint: disable=wrong-import-position
from bench import stages  # pylint: disable=wrong-import-position


def utc_now_iso() -> str:
  """Returns current UTC timestamp in ISO-8601 `Z` format."""
  return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_container_mem_gb() -> tuple[float, float, float]:
  """Reads `(used_gb, avail_gb, total_gb)` combining `/proc/meminfo` and cgroup limits."""
  used_gb = 0.0
  avail_gb = 250.0
  total_gb = 250.0
  try:
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
      mem: dict[str, int] = {}
      for line in f:
        parts = line.split()
        if len(parts) >= 2:
          mem[parts[0].rstrip(":")] = int(parts[1])
      total_gb = (mem.get("MemTotal", 0) * 1024) / 1e9
      avail_gb = (mem.get("MemAvailable", mem.get("MemFree", 0)) * 1024) / 1e9
      used_gb = max(0.0, total_gb - avail_gb)
    for cur_path, max_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
      if not os.path.exists(cur_path):
        continue
      try:
        with open(cur_path, "r", encoding="utf-8") as cf:
          cg_used_gb = int(cf.read().strip()) / 1e9
        if os.path.exists(max_path):
          with open(max_path, "r", encoding="utf-8") as mf:
            raw_max = mf.read().strip()
          if raw_max.isdigit():
            cg_max_gb = int(raw_max) / 1e9
            if 10.0 < cg_max_gb < total_gb:
              total_gb = cg_max_gb
        if cg_used_gb > used_gb:
          used_gb = cg_used_gb
          avail_gb = max(0.0, total_gb - used_gb)
      except OSError:
        pass
      break
  except OSError:
    pass
  return used_gb, avail_gb, total_gb


class ResourceSampler:
  """Samples host container RAM, process RSS, and optional TPU HBM per stage in the background."""

  def __init__(
      self,
      recorder: stages.StageRecorder,
      *,
      jax_mod: Any = None,
      interval_s: float = 0.5,
      hbm_every_n: int = 2,
      ram_reader: Callable[[], tuple[float, float, float]] | None = None,
      rss_reader: Callable[[], float | None] | None = None,
      hbm_reader: Callable[[Any], dict[str, float] | None] | None = None,
  ):
    self._recorder = recorder
    self._jax_mod = jax_mod
    self._interval_s = interval_s
    self._hbm_every_n = max(1, hbm_every_n)
    self._ram_reader = ram_reader or read_container_mem_gb
    self._rss_reader = rss_reader or stages.read_process_rss_gb
    self._hbm_reader = hbm_reader or stages.snapshot_hbm_gb
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._lock = threading.Lock()
    self.peak_container_ram_gb = 0.0
    self.peak_process_rss_gb = 0.0
    self.host_ram_size_gb = 0.0
    self.ram_by_stage_gb: dict[str, dict[str, float]] = {}
    self.hbm_usable_gb: float | None = None
    self.hbm_peak_sampled_gb: float = 0.0
    self.hbm_peak_hardware_gb: float = 0.0
    self.hbm_used_after_read_gb: float | None = None
    self.hbm_used_after_load_gb: float | None = None
    self._sample_once(check_hbm=True)

  def _sample_once(self, *, check_hbm: bool = False) -> None:
    """Captures container RAM, RSS, and optional TPU HBM snapshots for the active stage."""
    used_gb, _, total_gb = self._ram_reader()
    rss_gb = self._rss_reader() or 0.0
    stage_name = self._recorder.current_stage() or "unscoped"
    hbm_snap = None
    if check_hbm and self._jax_mod is not None:
      try:
        hbm_snap = self._hbm_reader(self._jax_mod)
      except Exception:  # pylint: disable=broad-exception-caught
        hbm_snap = None

    with self._lock:
      self.host_ram_size_gb = max(self.host_ram_size_gb, total_gb)
      self.peak_container_ram_gb = max(self.peak_container_ram_gb, used_gb)
      self.peak_process_rss_gb = max(self.peak_process_rss_gb, rss_gb)
      st = self.ram_by_stage_gb.setdefault(stage_name, {"peak_container_ram_gb": 0.0, "peak_rss_gb": 0.0})
      st["peak_container_ram_gb"] = max(st["peak_container_ram_gb"], used_gb)
      st["peak_rss_gb"] = max(st["peak_rss_gb"], rss_gb)

      if hbm_snap:
        usable = hbm_snap.get("usable_per_device_gb")
        if usable and (self.hbm_usable_gb is None or usable > self.hbm_usable_gb):
          self.hbm_usable_gb = usable
        self.hbm_peak_sampled_gb = max(self.hbm_peak_sampled_gb, float(hbm_snap.get("max_bytes_in_use_gb", 0.0) or 0.0))
        self.hbm_peak_hardware_gb = max(
            self.hbm_peak_hardware_gb, float(hbm_snap.get("max_peak_bytes_in_use_gb", 0.0) or 0.0)
        )

  def _run(self) -> None:
    step = 0
    while not self._stop_event.wait(self._interval_s):
      step += 1
      self._sample_once(check_hbm=step % self._hbm_every_n == 0)

  def start(self) -> None:
    """Starts background memory sampling."""
    if self._thread is not None:
      return
    self._thread = threading.Thread(target=self._run, name="mesa-res-sampler", daemon=True)
    self._thread.start()

  def stop(self) -> None:
    """Stops background memory sampling and captures one final sample."""
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
      self._thread = None
    self._sample_once(check_hbm=True)

  def record_hbm_milestone(self, milestone: str, jax_mod: Any = None) -> None:
    """Captures an immediate HBM snapshot at `after_read` or `after_load`."""
    jm = jax_mod or self._jax_mod
    if jm is None:
      return
    try:
      snap = self._hbm_reader(jm)
    except Exception:  # pylint: disable=broad-exception-caught
      return
    if not snap:
      return
    with self._lock:
      used_now = float(snap.get("max_bytes_in_use_gb", 0.0) or 0.0)
      peak_now = float(snap.get("max_peak_bytes_in_use_gb", 0.0) or 0.0)
      usable = snap.get("usable_per_device_gb")
      if usable and self.hbm_usable_gb is None:
        self.hbm_usable_gb = usable
      self.hbm_peak_sampled_gb = max(self.hbm_peak_sampled_gb, used_now)
      self.hbm_peak_hardware_gb = max(self.hbm_peak_hardware_gb, peak_now)
      if milestone == "after_read":
        self.hbm_used_after_read_gb = round(used_now, 3)
      elif milestone == "after_load":
        self.hbm_used_after_load_gb = round(used_now, 3)

  def memory_payload(self) -> dict[str, Any]:
    """Returns serialized memory dictionary suitable for `BENCH_STAGES_JSON`."""
    with self._lock:
      peak_hbm = max(self.hbm_peak_hardware_gb, self.hbm_peak_sampled_gb)
      out: dict[str, Any] = {
          "peak_host_ram_container_gb": round(self.peak_container_ram_gb, 3),
          "peak_process_rss_gb": round(self.peak_process_rss_gb, 3),
          "host_ram_size_gb": round(self.host_ram_size_gb, 2),
          "by_stage_gb": {k: dict(v) for k, v in self.ram_by_stage_gb.items()},
      }
      if self.hbm_usable_gb is not None:
        out["memory_tpu_hbm_usable_per_device_gb"] = round(self.hbm_usable_gb, 3)
        if self.hbm_usable_gb > 0 and peak_hbm > 0:
          out["memory_tpu_hbm_peak_percent_of_usable"] = round((peak_hbm / self.hbm_usable_gb) * 100.0, 2)
      if peak_hbm > 0:
        out["memory_tpu_hbm_peak_per_device_gb"] = round(peak_hbm, 3)
      if self.hbm_used_after_read_gb is not None:
        out["memory_tpu_hbm_used_after_read_per_device_gb"] = self.hbm_used_after_read_gb
      if self.hbm_used_after_load_gb is not None:
        out["memory_tpu_hbm_used_after_load_per_device_gb"] = self.hbm_used_after_load_gb
      return out


def install_setup_hooks(
    recorder: stages.StageRecorder,
    *,
    maxtext_utils_mod: Any = None,
    model_creation_mod: Any = None,
) -> None:
  """Attaches non-invasive timing wrappers around MaxText mesh and abstract state builders."""
  if maxtext_utils_mod is not None:
    recorder.wrap(maxtext_utils_mod, "get_mesh_from_config", "setup_device_mesh_build_seconds")
    recorder.wrap(maxtext_utils_mod, "get_abstract_state", "setup_abstract_train_state_build_seconds")
  if model_creation_mod is not None:
    recorder.wrap(model_creation_mod, "create_nnx_abstract_model", "setup_abstract_model_build_seconds")


def install_safetensors_dynamic_hooks(
    recorder: stages.StageRecorder,
    *,
    load_dynamic_mod: Any,
    tensor_handling_mod: Any,
    ocp_v1_mod: Any = None,
    jax_mod: Any = None,
) -> dict[str, Any]:
  """Installs fine-grained `safetensors_dynamic` stage wrappers and returns a slowest-param state dict."""
  slowest_state: dict[str, Any] = {"slowest_s": 0.0, "slowest_name": ""}

  if ocp_v1_mod is not None:
    recorder.wrap(ocp_v1_mod, "pytree_metadata", "safetensors_dynamic_step2_read_file_headers_and_plan_seconds")
    recorder.wrap(ocp_v1_mod, "load_pytree", "safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds")
  if load_dynamic_mod is not None:
    recorder.wrap(
        load_dynamic_mod,
        "_leading_axis_shardings",
        "safetensors_dynamic_step2_read_file_headers_and_plan_seconds",
        reentrant_guard=True,
    )
  if jax_mod is not None and hasattr(jax_mod, "device_put"):
    orig_dp = jax_mod.device_put
    if not getattr(orig_dp, "__mesa_sd_h2d_wrapped__", False):

      @functools.wraps(orig_dp)
      def _sd_device_put(*args, **kwargs):
        cur = recorder.current_stage()
        if cur in (
            "safetensors_dynamic_step2_load_total_seconds",
            "safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds",
        ):
          with recorder.stage("safetensors_dynamic_step2_copy_host_ram_to_tpu_hbm_seconds"):
            return orig_dp(*args, **kwargs)
        return orig_dp(*args, **kwargs)

      setattr(_sd_device_put, "__mesa_sd_h2d_wrapped__", True)
      jax_mod.device_put = _sd_device_put
      recorder._patches.append((jax_mod, "device_put", orig_dp))  # pylint: disable=protected-access
      recorder._active_wrappers.append("jax.device_put[sd_load]")  # pylint: disable=protected-access

  if tensor_handling_mod is not None:
    recorder.wrap(
        tensor_handling_mod,
        "apply_hook_fns",
        "safetensors_dynamic_step2_transform_hooks_seconds",
        reentrant_guard=True,
    )
    recorder.wrap(
        tensor_handling_mod,
        "_reshard",
        "safetensors_dynamic_step2_reshard_to_target_layout_seconds",
        reentrant_guard=True,
    )
    for fn_name in ("_binary_chunked_stack", "_concatenate_pieces"):
      recorder.wrap(
          tensor_handling_mod,
          fn_name,
          "safetensors_dynamic_step2_stack_layers_and_experts_seconds",
          reentrant_guard=True,
      )
    recorder.wrap(
        tensor_handling_mod,
        "_cast_and_free",
        "safetensors_dynamic_step2_cast_to_load_dtype_seconds",
        reentrant_guard=True,
    )
    recorder.wrap(
        tensor_handling_mod,
        "_gather_to_host",
        "safetensors_dynamic_step2_gather_to_host_fallback_seconds",
        reentrant_guard=True,
    )
    recorder.wrap(
        tensor_handling_mod,
        "_device_put",
        "safetensors_dynamic_step2_copy_host_ram_to_tpu_hbm_seconds",
        reentrant_guard=True,
    )
  for target_mod in (load_dynamic_mod, tensor_handling_mod):
    _wrap_get_hf_loading_function(recorder, target_mod, slowest_state)
  return slowest_state


def _wrap_get_hf_loading_function(
    recorder: stages.StageRecorder,
    target_mod: Any,
    slowest_state: dict[str, Any],
) -> None:
  """Wraps `target_mod.get_hf_loading_function` to record the slowest parameter transform."""
  if target_mod is None or not hasattr(target_mod, "get_hf_loading_function"):
    return
  orig_get_loader = target_mod.get_hf_loading_function
  if getattr(orig_get_loader, "__mesa_param_wrapped__", False):
    return

  @functools.wraps(orig_get_loader)
  def _wrapped_get_loader(*args, **kwargs):
    loader_fn = orig_get_loader(*args, **kwargs)
    mt_key = args[5] if len(args) > 5 else kwargs.get("mt_key", "")
    key_str = ",".join(str(x) for x in mt_key) if isinstance(mt_key, tuple) else str(mt_key)

    @functools.wraps(loader_fn)
    def _timed_loader():
      t0 = time.perf_counter()
      try:
        return loader_fn()
      finally:
        dt = time.perf_counter() - t0
        if dt > slowest_state["slowest_s"]:
          slowest_state["slowest_s"] = dt
          slowest_state["slowest_name"] = key_str

    return _timed_loader

  setattr(_wrapped_get_loader, "__mesa_param_wrapped__", True)
  target_mod.get_hf_loading_function = _wrapped_get_loader
  recorder._patches.append((target_mod, "get_hf_loading_function", orig_get_loader))  # pylint: disable=protected-access
  mod_label = getattr(target_mod, "__name__", type(target_mod).__name__)
  recorder._active_wrappers.append(f"{mod_label}.get_hf_loading_function")  # pylint: disable=protected-access


def install_to_maxtext_step2_hooks(
    recorder: stages.StageRecorder,
    *,
    to_maxtext_mod: Any,
    conv_utils_mod: Any,
    jax_mod: Any = None,
) -> None:
  """Attaches stage wrappers for `to_maxtext` Step 2 CPU conversion."""
  if conv_utils_mod is not None:
    recorder.wrap(conv_utils_mod, "apply_hook_fns", "to_maxtext_step2_transform_hooks_seconds", reentrant_guard=True)
  if to_maxtext_mod is not None:
    recorder.wrap(
        to_maxtext_mod,
        "_process_and_stack_weights",
        "to_maxtext_step2_stack_layers_and_experts_seconds",
        reentrant_guard=True,
    )
  if jax_mod is not None:
    if hasattr(jax_mod, "make_array_from_process_local_data"):
      recorder.wrap(
          jax_mod,
          "make_array_from_process_local_data",
          "to_maxtext_step2_place_on_cpu_devices_seconds",
          reentrant_guard=True,
      )
    recorder.wrap(jax_mod, "device_put", "to_maxtext_step2_place_on_cpu_devices_seconds", reentrant_guard=True)
  recorder.wrap(gc, "collect", "to_maxtext_step2_garbage_collection_seconds", reentrant_guard=True)


def install_to_maxtext_step3_hooks(
    recorder: stages.StageRecorder,
    *,
    checkpointing_mod: Any,
    ocp_v1_mod: Any = None,
    ocp_mod: Any = None,
) -> None:
  """Attaches stage wrappers for `to_maxtext` Step 3 Orbax restore."""
  if ocp_v1_mod is not None:
    recorder.wrap(ocp_v1_mod, "pytree_metadata", "to_maxtext_step3_read_orbax_metadata_seconds", reentrant_guard=True)
  if ocp_mod is not None and hasattr(ocp_mod, "Checkpointer"):
    recorder.wrap(
        ocp_mod.Checkpointer,
        "metadata",
        "to_maxtext_step3_read_orbax_metadata_seconds",
        reentrant_guard=True,
    )
  if checkpointing_mod is not None:
    recorder.wrap(checkpointing_mod, "load_checkpoint_metadata", "to_maxtext_step3_read_orbax_metadata_seconds")
    recorder.wrap(checkpointing_mod, "_flat_moe_restore_plan", "to_maxtext_step3_plan_read_seconds")
    recorder.wrap(
        checkpointing_mod,
        "load_params_from_path",
        "to_maxtext_step3_read_from_gcs_to_tpu_hbm_seconds",
        reentrant_guard=True,
    )


def compute_local_shard_gb(pytree_or_dict: Any) -> float:
  """Sums addressable local shard byte sizes across all JAX arrays in `pytree_or_dict` in GB (`10^9`)."""
  seen_bufs: set[int] = set()
  total_bytes = 0

  def _visit(obj: Any) -> None:
    nonlocal total_bytes
    if isinstance(obj, Mapping):
      for v in obj.values():
        _visit(v)
      return
    if isinstance(obj, (list, tuple)):
      for v in obj:
        _visit(v)
      return
    shards = getattr(obj, "addressable_shards", None)
    if shards is not None:
      for shard in shards:
        data = getattr(shard, "data", shard)
        buf_id = id(data)
        if buf_id not in seen_bufs and hasattr(data, "nbytes"):
          seen_bufs.add(buf_id)
          total_bytes += int(data.nbytes)
      return
    if hasattr(obj, "nbytes"):
      buf_id = id(obj)
      if buf_id not in seen_bufs:
        seen_bufs.add(buf_id)
        total_bytes += int(obj.nbytes)

  _visit(pytree_or_dict)
  return round(total_bytes / 1e9, 4)


def _flatten_params(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
  """Flattens nested param tree into sorted `(dotted_path, array)` leaves."""
  if hasattr(tree, "to_pure_dict"):
    tree = tree.to_pure_dict()
  out: list[tuple[str, Any]] = []
  if isinstance(tree, Mapping):
    for k in sorted(tree, key=str):
      sub = f"{prefix}.{k}" if prefix else str(k)
      out.extend(_flatten_params(tree[k], sub))
  elif isinstance(tree, (list, tuple)):
    for idx, v in enumerate(tree):
      sub = f"{prefix}.{idx}" if prefix else str(idx)
      out.extend(_flatten_params(v, sub))
  elif hasattr(tree, "shape") and hasattr(tree, "dtype"):
    out.append((prefix, tree))
  return out


def select_checksum_indices(num_leaves: int, max_leaves: int = 8) -> list[int]:
  """Selects up to `max_leaves` deterministic indices spanning start, middle, and end."""
  if num_leaves <= 0:
    return []
  if num_leaves <= max_leaves:
    return list(range(num_leaves))
  head = list(range(min(4, max_leaves)))
  tail_count = min(2, max_leaves - len(head))
  tail = list(range(num_leaves - tail_count, num_leaves)) if tail_count > 0 else []
  mid_slots = max_leaves - len(head) - len(tail)
  mid = []
  if mid_slots > 0:
    lo, hi = head[-1] + 1, tail[0] if tail else num_leaves
    if hi > lo:
      step = max(1, (hi - lo) // (mid_slots + 1))
      mid = [lo + step * (i + 1) for i in range(mid_slots) if lo + step * (i + 1) < hi]
  return sorted(set(head + mid + tail))


def compute_parameter_checksums(
    raw_params: Any,
    jax_mod: Any,
    *,
    max_leaves: int = 8,
) -> dict[str, dict[str, Any]]:
  """Computes distributed scalar reductions (sum, L2, abs-max) over sample parameter leaves."""
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

  leaves = _flatten_params(raw_params)
  chosen = select_checksum_indices(len(leaves), max_leaves=max_leaves)
  if not chosen:
    return {}

  @jax_mod.jit
  def _leaf_stats(arr):
    f = arr.astype(jnp.float32)
    return jnp.sum(f), jnp.sqrt(jnp.sum(f * f)), jnp.max(jnp.abs(f))

  results: dict[str, dict[str, Any]] = {}
  for idx in chosen:
    name, arr = leaves[idx]
    norm_name = name.removeprefix("params.")
    if hasattr(arr, "shape") and arr.size == 0:
      continue
    s_val, l2_val, amax_val = _leaf_stats(arr)
    results[norm_name] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sum": round(float(jax_mod.device_get(s_val)), 6),
        "l2": round(float(jax_mod.device_get(l2_val)), 6),
        "abs_max": round(float(jax_mod.device_get(amax_val)), 6),
    }
  return results


def compare_parameter_checksums(
    cs_a: Mapping[str, Mapping[str, Any]],
    cs_b: Mapping[str, Mapping[str, Any]],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> list[str]:
  """Compares sample parameter checksums between two pipelines; returns mismatch descriptions."""
  norm_a = {k.removeprefix("params."): v for k, v in cs_a.items()}
  norm_b = {k.removeprefix("params."): v for k, v in cs_b.items()}
  shared = sorted(set(norm_a) & set(norm_b))
  if not shared:
    return ["No overlapping parameter keys between checksum dicts"]
  mismatches: list[str] = []
  for k in shared:
    a, b = norm_a[k], norm_b[k]
    if list(a.get("shape", [])) != list(b.get("shape", [])):
      mismatches.append(f"{k}: shape {a.get('shape')} != {b.get('shape')}")
      continue
    for metric in ("sum", "l2", "abs_max"):
      va, vb = float(a[metric]), float(b[metric])
      tol = atol + rtol * max(abs(va), abs(vb))
      if abs(va - vb) > tol:
        mismatches.append(f"{k} {metric}: {va} vs {vb} (diff={abs(va - vb):.4g} > tol={tol:.4g})")
  return mismatches


def inspect_gcs_or_local_safetensors(path: str) -> dict[str, Any]:
  """Safely inspects `.safetensors` headers from a local folder or `gs://` prefix."""
  if not path:
    return {}
  try:
    if path.startswith("gs://"):
      from etils import epath  # pylint: disable=import-outside-toplevel

      st_paths = [str(p) for p in sorted(epath.Path(path).glob("*.safetensors"))]
    elif os.path.isdir(path):
      st_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    elif path.endswith(".safetensors") and os.path.exists(path):
      st_paths = [path]
    else:
      return {}
    if not st_paths:
      return {}
    return metadata.inspect_safetensors_headers(st_paths)
  except Exception:  # pylint: disable=broad-exception-caught
    return {}


def prepare_clean_jax_compilation_cache() -> str:
  """Clears default local JAX persistent compilation cache directories unless `KEEP_JAX_CACHE=1`."""
  if os.environ.get("KEEP_JAX_CACHE", "0") == "1":
    return os.environ.get("BENCH_CACHE_STATE", "warm")
  candidates = [
      "/tmp/jax_cache",
      os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip(),
  ]
  for cand in candidates:
    if cand and not cand.startswith("gs://") and os.path.isdir(cand):
      import shutil  # pylint: disable=import-outside-toplevel

      shutil.rmtree(cand, ignore_errors=True)
  return os.environ.get("BENCH_CACHE_STATE", "empty")


def classify_failure(
    err_text: str,
    log_texts: Sequence[str] = (),
    *,
    exit_code: int | None = None,
    not_tpu_info: Mapping[str, Any] | None = None,
) -> tuple[str, str, bool, str]:
  """Classifies a failed worker run into `(failure_category, signal_name, was_preempted, error_summary)`."""
  combined = f"{err_text}\n" + "\n".join(log_texts)
  first_line = next((ln.strip() for ln in err_text.splitlines() if ln.strip()), "failed")

  sig_name = ""
  if exit_code in (-9, 137):
    sig_name = "SIGKILL"
  elif exit_code in (-15, 143):
    sig_name = "SIGTERM"
  elif exit_code is not None and exit_code < 0:
    try:
      sig_name = signal.Signals(-exit_code).name
    except ValueError:
      sig_name = f"SIG{-exit_code}"

  if not_tpu_info is not None or "[NOT-TPU]" in combined:
    return "NOT-TPU", sig_name, False, first_line
  if (
      "TIMEOUT:" in combined
      or "[TIMEOUT_WATCHDOG]" in combined
      or "exceeded PHASE_TIMEOUT_S" in combined
      or "DEADLINE_EXCEEDED" in combined
  ):
    return "TIMEOUT", sig_name, False, first_line
  if "RESOURCE_EXHAUSTED" in combined or "pjrt.OOM" in combined or "HLO temporaries" in combined:
    return "OOM-HBM", sig_name, False, first_line
  if sig_name == "SIGTERM" or "Preempted" in combined or "Evicted" in combined or "spot preemption" in combined.lower():
    return "PREEMPTED", sig_name, True, first_line
  if (
      "[OOM_WATCHDOG]" in combined
      or "OOMKilled" in combined
      or sig_name == "SIGKILL"
      or "holds the whole converted state" in combined
      or "exceeds ~" in combined
      and "usable" in combined
      or re.search(r"needs all \d+ shards .* staged", combined)
  ):
    return "OOM-HOST", sig_name, False, first_line
  return "ERROR", sig_name, False, first_line


def infer_failed_stage(merged_stages: Mapping[str, Any], log_texts: Sequence[str]) -> str:
  """Infers which stage was executing when a run failed."""
  hosts = merged_stages.get("hosts") or []
  for h in hosts:
    if h.get("failed_stage"):
      return str(h["failed_stage"])
    if h.get("active_stage"):
      return str(h["active_stage"])
  last_stage = ""
  for text in log_texts:
    for match in re.finditer(r"(?:BENCH_STAGE_START:|\[STAGE START\]\s+)([A-Za-z0-9_.:-]+)", text):
      last_stage = match.group(1)
  return last_stage


def _mon_sum(merged_stages: Mapping[str, Any], event_name: str, factor: float = 1.0) -> float | str:
  val = merged_stages.get("monitoring_summary", {}).get(event_name, {}).get("max_over_hosts")
  if isinstance(val, (int, float)):
    return round(float(val) * factor, 4)
  return ""


def _stage_max(merged_stages: Mapping[str, Any], stage_name: str) -> float | str:
  val = merged_stages.get("stages_headline_max", {}).get(stage_name)
  if isinstance(val, (int, float)):
    return round(float(val), 4)
  return ""


def _stage_metric_sum(merged_stages: Mapping[str, Any], stage_name: str, metric_key: str) -> float | str:
  val = merged_stages.get("stages_summary", {}).get(stage_name, {}).get("metrics", {}).get(metric_key, {}).get("sum")
  if isinstance(val, (int, float)):
    return round(float(val), 4)
  return ""


def collect_merged_stages_from_logs(log_texts: Sequence[str]) -> dict[str, Any]:
  """Parses all `BENCH_STAGES_JSON:` lines from local + remote slice logs and merges them."""
  payloads: list[dict[str, Any]] = []
  for txt in log_texts:
    payloads.extend(stages.parse_host_stages_lines(txt))
  return stages.merge_host_stages(payloads)


def build_v2_phase_record(
    *,
    phase: str,
    model_name: str,
    scan_mode: str,
    platform_name: str,
    num_slices: int,
    num_hosts_per_slice: int,
    topology: str,
    raw_worker_data: Mapping[str, Any] | None,
    merged_stages: Mapping[str, Any],
    env_meta: Mapping[str, Any],
    code_ver: Mapping[str, Any],
    hw_info: Mapping[str, Any],
    pkg_versions: Mapping[str, str | None],
    started_at_utc: str,
    finished_at_utc: str,
    wall_clock_s: float,
    run_status: str,
    cache_state: str,
    pipeline_config: Mapping[str, Any] | None = None,
    header_info: Mapping[str, Any] | None = None,
    failure_info: Mapping[str, Any] | None = None,
    pod_names: Sequence[str] = (),
    node_names: Sequence[str] = (),
    log_urls: Mapping[str, str] | None = None,
    parameter_checksums: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Builds a complete v2 result JSON dictionary validating against `schema_v2.validate_v2_phase_record`."""
  pipeline = "safetensors_dynamic" if phase == "safetensors_dynamic" else "to_maxtext"
  models_cfg = schema_v2.load_models_v2()
  models_map = {m["model_name"]: m for m in models_cfg["models"]}
  model_info = models_map.get(model_name, {})
  profile = platforms.get_profile(platform_name) if platform_name in platforms.PROFILES else None

  dev_per_slice = profile.devices_per_slice(topology) if profile else 8
  chips_per_host = profile.chips_per_host(topology) if profile else 4
  dev_per_chip = profile.tpu_devices_per_chip if profile else 2
  total_hosts = num_slices * num_hosts_per_slice
  total_devices = num_slices * dev_per_slice

  raw = dict(raw_worker_data or {})
  fail = dict(failure_info or {})
  hdr = dict(header_info or {})
  p_cfg = dict(pipeline_config or {})
  l_urls = dict(log_urls or {})
  mem_sum = dict(merged_stages.get("memory_summary") or {})

  rec: dict[str, Any] = {
      "schema_version": schema_v2.SCHEMA_VERSION,
      "phase_name": phase,
      "run_id": env_meta.get("run_id") or "",
      "experiment_name": env_meta.get("experiment_name") or "",
      "repetition_index": env_meta.get("repetition_index") or 1,
      "queue_attempt_number": env_meta.get("queue_attempt_number") or 1,
      "jobset_restart_attempt": env_meta.get("jobset_restart_attempt") or 0,
      "phase_started_at_utc": started_at_utc,
      "phase_finished_at_utc": finished_at_utc,
      "tpu_platform": platform_name,
      "gke_cluster_name": env_meta.get("gke_cluster_name") or (profile.gke_cluster_name if profile else ""),
      "gcp_project_id": env_meta.get("gcp_project_id") or (profile.gcp_project_id if profile else ""),
      "gcp_region": env_meta.get("gcp_region") or (profile.gcp_region if profile else ""),
      "gcp_zone": env_meta.get("gcp_zone") or "",
      "gke_version": env_meta.get("gke_version") or "",
      "gke_node_pool_name": env_meta.get("gke_node_pool_name") or "",
      "capacity_type": env_meta.get("capacity_type") or "",
      "kueue_local_queue_name": env_meta.get("kueue_local_queue_name") or (profile.kueue_queue or "" if profile else ""),
      "host_machine_type": env_meta.get("host_machine_type")
      or (profile.host_machine_type_by_topology.get(topology, "") if profile else ""),
      "tpu_accelerator_type": env_meta.get("tpu_accelerator_type") or "",
      "number_of_slices": num_slices,
      "slice_topology": topology,
      "hosts_per_slice": num_hosts_per_slice,
      "tpu_chips_per_host": chips_per_host,
      "tpu_devices_per_chip": dev_per_chip,
      "tpu_devices_per_slice": dev_per_slice,
      "total_hosts": total_hosts,
      "total_tpu_devices": total_devices,
      "interconnect_type": (
          "ICI+DCN"
          if (num_slices > 1 and num_hosts_per_slice > 1)
          else ("DCN" if num_slices > 1 else ("ICI" if num_hosts_per_slice > 1 else "Single-host"))
      ),
      "host_ram_gb_measured": mem_sum.get("memory_host_ram_size_gb") or hw_info.get("host_ram_gb_measured") or "",
      "host_vcpu_count": hw_info.get("host_vcpu_count") or "",
      "container_image": env_meta.get("container_image") or "",
      **pkg_versions,
      "git_branch_name": code_ver.get("git_branch_name", "unknown"),
      "git_commit": code_ver.get("git_commit", "unknown"),
      "git_working_tree_clean": code_ver.get("git_working_tree_clean"),
      "code_tarball_sha256": code_ver.get("code_tarball_sha256") or "",
      "active_timing_wrappers": ",".join(merged_stages.get("active_timing_wrappers", [])),
      "model_name": model_name,
      "hf_repo_id": model_info.get("hf_repo_id", ""),
      "model_parameter_count_billions": model_info.get("params_billions", ""),
      "model_architecture": model_info.get("architecture", ""),
      "model_test_tier": model_info.get("tier", ""),
      "scan_layers_mode": scan_mode,
      "number_of_parameter_leaves": raw.get("num_param_leaves", ""),
      "hf_checkpoint_dtype": hdr.get("hf_checkpoint_dtype") or model_info.get("hf_dtype", ""),
      "loaded_checkpoint_dtype": p_cfg.get("loaded_checkpoint_dtype") or model_info.get("loaded_dtype", ""),
      "maxtext_load_weight_dtype": p_cfg.get("maxtext_load_weight_dtype") or model_info.get("weight_dtype", ""),
      "to_maxtext_orbax_save_dtype": p_cfg.get("to_maxtext_orbax_save_dtype", ""),
      "checkpoint_weights_kind": "real",
      "checkpoint_size_gb": hdr.get("checkpoint_size_gb") or model_info.get("loaded_bf16_gb", ""),
      "checkpoint_number_of_files": hdr.get("checkpoint_number_of_files") or model_info.get("num_files", ""),
      "checkpoint_number_of_tensors": hdr.get("checkpoint_number_of_tensors", ""),
      "pipeline_name": pipeline,
      "device_mesh_shape": p_cfg.get("device_mesh_shape", ""),
      "ici_fsdp_parallelism": p_cfg.get("ici_fsdp_parallelism", ""),
      "dcn_parallelism_axis": p_cfg.get("dcn_parallelism_axis", ""),
      "dcn_parallelism_size": p_cfg.get("dcn_parallelism_size", ""),
      "checkpoint_storage_concurrent_gb": p_cfg.get("checkpoint_storage_concurrent_gb", ""),
      "safetensors_read_concurrent_bytes": p_cfg.get("safetensors_read_concurrent_bytes", ""),
      "jax_compilation_cache_state": cache_state,
      "harness_environment_flags": p_cfg.get("harness_environment_flags", ""),
      "run_status": run_status,
      "failure_category": fail.get("failure_category", ""),
      "failed_stage_name": fail.get("failed_stage_name", ""),
      "process_exit_code": fail.get("process_exit_code", ""),
      "process_signal": fail.get("process_signal", ""),
      "error_summary": fail.get("error_summary", ""),
      "was_preempted": bool(fail.get("was_preempted", False)),
      "pod_names": ",".join(pod_names) if pod_names else (env_meta.get("pod_name") or ""),
      "node_names": ",".join(node_names) if node_names else (env_meta.get("node_name") or ""),
      "phase_wall_clock_seconds": round(wall_clock_s, 4),
      **l_urls,
  }

  # Attach TPU backend check measurements.
  for tpu_k in (
      "jax_backend_platform",
      "tpu_device_kind",
      "total_tpu_devices_measured",
      "tpu_devices_per_host_measured",
      "tpu_hbm_per_device_gb_measured",
  ):
    if tpu_k in raw and raw[tpu_k] is not None:
      rec[tpu_k] = raw[tpu_k]
  if isinstance(raw.get("tpu_hbm_per_device_gb_measured"), (int, float)) and total_devices:
    rec["total_tpu_hbm_gb"] = round(float(raw["tpu_hbm_per_device_gb_measured"]) * total_devices, 2)

  # Map setup + pipeline stage columns directly from merged stage keys + headline keys.
  setup_keys = (
      "setup_process_start_to_harness_code_seconds",
      "setup_python_imports_seconds",
      "setup_jax_distributed_initialize_seconds",
      "setup_tpu_backend_initialize_seconds",
      "setup_tpu_check_seconds",
      "setup_maxtext_config_build_seconds",
      "setup_device_mesh_build_seconds",
      "setup_abstract_model_build_seconds",
      "setup_abstract_train_state_build_seconds",
      "setup_delete_old_gcs_cache_seconds",
      "setup_wait_for_all_hosts_before_load_seconds",
      "setup_total_before_load_seconds",
      "teardown_write_results_and_exit_seconds",
  )
  for sk in setup_keys:
    val = _stage_max(merged_stages, sk)
    if val != "":
      rec[sk] = val
    elif sk in raw:
      rec[sk] = raw[sk]

  for k, v in raw.items():
    if k not in rec:
      rec[k] = v

  for mem_k in (
      "memory_peak_host_ram_container_gb",
      "memory_peak_host_ram_host_name",
      "memory_peak_process_rss_gb",
      "memory_host_ram_size_gb",
      "memory_tpu_hbm_usable_per_device_gb",
      "memory_tpu_hbm_used_after_read_per_device_gb",
      "memory_tpu_hbm_peak_per_device_gb",
      "memory_tpu_hbm_used_after_load_per_device_gb",
      "memory_tpu_hbm_peak_percent_of_usable",
  ):
    if mem_k in mem_sum:
      rec[mem_k] = mem_sum[mem_k]
  if "memory_peak_host_ram_container_gb" in mem_sum:
    rec["peak_host_ram_gb"] = mem_sum["memory_peak_host_ram_container_gb"]
  if "memory_host_ram_size_gb" in mem_sum:
    rec["total_host_ram_gb"] = mem_sum["memory_host_ram_size_gb"]

  if pipeline == "safetensors_dynamic":
    _populate_sd_stage_fields(rec, merged_stages, raw)
  else:
    _populate_tm_stage_fields(rec, phase, merged_stages, raw)

  if parameter_checksums:
    rec["parameter_checksums"] = dict(parameter_checksums)
  rec["stages_merged"] = {
      "host_count": merged_stages.get("host_count", 0),
      "stages_summary": merged_stages.get("stages_summary", {}),
      "barriers_summary": merged_stages.get("barriers_summary", {}),
      "monitoring_summary": merged_stages.get("monitoring_summary", {}),
  }
  return rec


def _populate_sd_stage_fields(
    rec: dict[str, Any],
    merged_stages: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> None:
  """Populates fine-grained `safetensors_dynamic` stage and throughput fields in `rec`."""
  sd_stage_names = (
      "safetensors_dynamic_step1_verify_gcs_cache_seconds",
      "safetensors_dynamic_step1_other_hosts_wait_seconds",
      "safetensors_dynamic_step1_total_seconds",
      "safetensors_dynamic_step2_build_hf_to_maxtext_mappings_seconds",
      "safetensors_dynamic_step2_read_file_headers_and_plan_seconds",
      "safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds",
      "safetensors_dynamic_step2_copy_host_ram_to_tpu_hbm_seconds",
      "safetensors_dynamic_step2_wait_for_other_hosts_during_load_seconds",
      "safetensors_dynamic_step2_load_total_seconds",
      "safetensors_dynamic_step2_transform_hooks_seconds",
      "safetensors_dynamic_step2_reshard_to_target_layout_seconds",
      "safetensors_dynamic_step2_stack_layers_and_experts_seconds",
      "safetensors_dynamic_step2_cast_to_load_dtype_seconds",
      "safetensors_dynamic_step2_gather_to_host_fallback_seconds",
      "safetensors_dynamic_step2_xla_compile_seconds",
      "safetensors_dynamic_step2_transform_total_seconds",
      "safetensors_dynamic_step2_harness_memory_cleanup_seconds",
      "safetensors_dynamic_step2_final_weight_check_and_sync_seconds",
      "safetensors_dynamic_step2_total_seconds",
  )
  for k in sd_stage_names:
    if k not in rec or rec[k] in ("", None):
      v = _stage_max(merged_stages, k)
      if v != "":
        rec[k] = v

  g_host_cnt = sum(
      int(h.get("stages", {}).get("safetensors_dynamic_step2_gather_to_host_fallback_seconds", {}).get("count", 0) or 0)
      for h in merged_stages.get("hosts", [])
  )
  if g_host_cnt > 0:
    rec["safetensors_dynamic_step2_gather_to_host_fallback_count"] = g_host_cnt
  elif "safetensors_dynamic_step2_gather_to_host_fallback_count" not in rec:
    rec["safetensors_dynamic_step2_gather_to_host_fallback_count"] = int(raw.get("gather_fallback_count", 0) or 0)

  if not rec.get("safetensors_dynamic_step1_other_hosts_wait_seconds"):
    st1_barrier = merged_stages.get("barriers_summary", {}).get("dynamic_hf_download_complete", {}).get("max_s")
    if isinstance(st1_barrier, (int, float)):
      rec["safetensors_dynamic_step1_other_hosts_wait_seconds"] = round(float(st1_barrier), 4)
  if not rec.get("safetensors_dynamic_step2_wait_for_other_hosts_during_load_seconds"):
    st2_barrier = merged_stages.get("barriers_summary", {}).get("safetensors_dynamic_load_complete", {}).get(
        "max_s"
    ) or _mon_sum(merged_stages, "/jax/checkpoint/sync_global_devices_duration_sec")
    if isinstance(st2_barrier, (int, float)) and float(st2_barrier) > 0:
      rec["safetensors_dynamic_step2_wait_for_other_hosts_during_load_seconds"] = round(float(st2_barrier), 4)

  xla_sum = 0.0
  for ev in (
      "/jax/core/compile/jaxpr_trace_duration",
      "/jax/core/compile/jaxpr_to_mlir_module_duration",
      "/jax/core/compile/backend_compile_duration",
  ):
    val = _mon_sum(merged_stages, ev)
    if isinstance(val, (int, float)):
      xla_sum += float(val)
  if xla_sum > 0:
    rec["safetensors_dynamic_step2_xla_compile_seconds"] = round(xla_sum, 4)

  # Fill headline-consistent totals & read throughput metrics.
  if "safetensors_dynamic_step1_total_seconds" not in rec and "hf_to_gcs_s" in raw:
    rec["safetensors_dynamic_step1_total_seconds"] = round(float(raw["hf_to_gcs_s"]), 4)
  if "safetensors_dynamic_step2_total_seconds" not in rec and "safetensors_and_convert_s" in raw:
    rec["safetensors_dynamic_step2_total_seconds"] = round(float(raw["safetensors_and_convert_s"]), 4)

  read_gb = _mon_sum(merged_stages, "/jax/orbax/read/safetensors/bytes_read", factor=1e-9)
  if read_gb == "" and raw.get("bytes_read_gb") is not None:
    read_gb = round(float(raw["bytes_read_gb"]), 4)
  if isinstance(read_gb, (int, float)) and float(read_gb) > 0:
    rec["safetensors_dynamic_step2_read_from_gcs_to_host_ram_gb"] = round(float(read_gb), 4)
    read_s = rec.get("safetensors_dynamic_step2_read_from_gcs_to_host_ram_seconds")
    if isinstance(read_s, (int, float)) and float(read_s) > 0:
      rec["safetensors_dynamic_step2_read_from_gcs_to_host_ram_gb_per_second"] = round(float(read_gb) / float(read_s), 4)
  needed_gb = _stage_metric_sum(
      merged_stages, "safetensors_dynamic_step2_load_total_seconds", "bytes_needed_by_this_host_gb"
  )
  if needed_gb == "" and raw.get("bytes_needed_by_this_host_gb"):
    needed_gb = round(float(raw["bytes_needed_by_this_host_gb"]), 4)
  if isinstance(needed_gb, (int, float)) and float(needed_gb) > 0:
    rec["safetensors_dynamic_step2_bytes_needed_by_this_host_gb"] = round(float(needed_gb), 4)
    if isinstance(read_gb, (int, float)) and float(read_gb) > 0:
      rec["safetensors_dynamic_step2_bytes_read_divided_by_bytes_needed_ratio"] = round(
          float(read_gb) / float(needed_gb), 4
      )


def _populate_tm_stage_fields(
    rec: dict[str, Any],
    phase: str,
    merged_stages: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> None:
  """Populates fine-grained `to_maxtext` Step 1-3 stage and throughput fields in `rec`."""
  for k in merged_stages.get("stages_headline_max", {}):
    if k.startswith("to_maxtext_") and k not in rec:
      rec[k] = round(float(merged_stages["stages_headline_max"][k]), 4)
  if phase == "to_maxtext_step1" and "hf_to_local_s" in raw:
    rec.setdefault("to_maxtext_step1_total_seconds", round(float(raw["hf_to_local_s"]), 4))
  elif phase == "to_maxtext_step2" and "convert_to_orbax_s" in raw:
    rec.setdefault("to_maxtext_step2_total_seconds", round(float(raw["convert_to_orbax_s"]), 4))
    rec.setdefault("to_maxtext_step2_host_machine_type", rec.get("host_machine_type", ""))
    for mon_ev, tm_key in (
        ("/jax/orbax/write/blocking_d2h_duration_secs", "to_maxtext_step2_orbax_save_copy_device_to_host_seconds"),
        ("/jax/orbax/write/blocking_duration_secs", "to_maxtext_step2_orbax_save_write_to_gcs_seconds"),
        ("/jax/orbax/write/background_ocdbt_merge_secs", "to_maxtext_step2_orbax_save_merge_ocdbt_seconds"),
        ("/jax/orbax/write/background_meta_write_duration_secs", "to_maxtext_step2_orbax_save_commit_seconds"),
    ):
      mv = _mon_sum(merged_stages, mon_ev)
      if mv != "":
        rec.setdefault(tm_key, mv)
    save_gb = _mon_sum(merged_stages, "/jax/orbax/write/gbytes")
    if save_gb != "":
      rec.setdefault("to_maxtext_step2_orbax_save_written_gb", save_gb)
      save_s = rec.get("to_maxtext_step2_orbax_save_total_seconds") or raw.get("orbax_save_to_gcs_s")
      if isinstance(save_s, (int, float)) and float(save_s) > 0:
        rec["to_maxtext_step2_orbax_save_gb_per_second"] = round(float(save_gb) / float(save_s), 4)
  elif phase == "to_maxtext_step3" and "load_orbax_s" in raw:
    rec.setdefault("to_maxtext_step3_total_seconds", round(float(raw["load_orbax_s"]), 4))
    rec.setdefault("to_maxtext_load_only_seconds", round(float(raw["load_orbax_s"]), 4))
    if not rec.get("to_maxtext_step3_wait_for_other_hosts_inside_orbax_seconds"):
      orb_barrier = _mon_sum(merged_stages, "/jax/checkpoint/sync_global_devices_duration_sec")
      if isinstance(orb_barrier, (int, float)) and float(orb_barrier) > 0:
        rec["to_maxtext_step3_wait_for_other_hosts_inside_orbax_seconds"] = round(float(orb_barrier), 4)
    read_gb = _mon_sum(merged_stages, "/jax/orbax/read/worker/io/requested/gbytes") or raw.get("bytes_read_gb", "")
    if isinstance(read_gb, (int, float)) and float(read_gb) > 0:
      rec["to_maxtext_step3_read_from_gcs_to_tpu_hbm_gb"] = round(float(read_gb), 4)
      read_s = rec.get("to_maxtext_step3_read_from_gcs_to_tpu_hbm_seconds") or raw.get("load_orbax_s")
      if isinstance(read_s, (int, float)) and float(read_s) > 0:
        rec["to_maxtext_step3_read_from_gcs_to_tpu_hbm_gb_per_second"] = round(float(read_gb) / float(read_s), 4)
