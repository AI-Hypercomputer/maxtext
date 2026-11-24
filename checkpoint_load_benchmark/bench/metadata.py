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

"""Software, hardware, CODE_VERSION, downward-API, and safetensors-header metadata collectors."""

from collections.abc import Mapping, Sequence
import importlib.metadata
import json
import os
import platform
import struct
from typing import Any

DEFAULT_PACKAGES: tuple[tuple[str, str], ...] = (
    ("jax", "jax_version"),
    ("jaxlib", "jaxlib_version"),
    ("libtpu", "libtpu_version"),
    ("flax", "flax_version"),
    ("orbax-checkpoint", "orbax_checkpoint_version"),
    ("transformers", "transformers_version"),
    ("safetensors", "safetensors_version"),
    ("numpy", "numpy_version"),
)

_SCALE_SUFFIXES = ("_scale_inv", ".scale", "_scale")


def collect_package_versions(packages: Sequence[tuple[str, str]] = DEFAULT_PACKAGES) -> dict[str, str | None]:
  """Collects installed Python and ML package versions without importing heavy backends."""
  versions: dict[str, str | None] = {"python_version": platform.python_version()}
  for dist_name, key_name in packages:
    try:
      versions[key_name] = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
      versions[key_name] = None
  return versions


def _read_proc_meminfo_gb() -> float | None:
  try:
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
      for line in f:
        if line.startswith("MemTotal:"):
          kb = int(line.split()[1])
          return round((kb * 1024) / 1e9, 2)
  except OSError:
    return None
  return None


def _read_cgroup_memory_max_gb() -> float | None:
  """Returns the container cgroup memory limit in GB (`10^9`), or None if unbounded."""
  for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
      with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
      if raw == "max":
        continue
      val = int(raw)
      # Unbounded cgroup v1 limits often report ~2^63-1; ignore limits >= 1 PiB.
      if 0 < val < 10**15:
        return round(val / 1e9, 2)
    except (OSError, ValueError):
      continue
  return None


def collect_host_hardware_info() -> dict[str, Any]:
  """Collects host RAM capacity (GB, 10^9 bytes) and vCPU count from the local node/container."""
  proc_gb = _read_proc_meminfo_gb()
  cg_gb = _read_cgroup_memory_max_gb()
  if proc_gb is not None and cg_gb is not None:
    ram_gb = min(proc_gb, cg_gb)
  else:
    ram_gb = proc_gb if proc_gb is not None else cg_gb
  return {
      "host_ram_gb_measured": ram_gb,
      "host_vcpu_count": os.cpu_count(),
  }


def load_code_version_metadata(tree_root: str | None = None) -> dict[str, Any]:
  """Reads CODE_VERSION.json from `tree_root` (or repo root) and returns v2 code metadata."""
  if tree_root is None:
    tree_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  cv_path = os.path.join(tree_root, "CODE_VERSION.json")
  raw: dict[str, Any] = {}
  try:
    with open(cv_path, "r", encoding="utf-8") as f:
      parsed = json.load(f)
    if isinstance(parsed, dict):
      raw = parsed
  except (OSError, ValueError):
    pass

  sha = (
      raw.get("code_tarball_sha256")
      or os.environ.get("BENCH_CODE_TARBALL_SHA256")
      or os.environ.get("CODE_TARBALL_SHA256")
  )
  return {
      "git_branch_name": raw.get("git_branch_name", "unknown"),
      "git_commit": raw.get("git_commit", "unknown"),
      "git_commit_short": raw.get("git_commit_short", "unknown"),
      "git_working_tree_clean": raw.get("git_working_tree_clean"),
      "git_commit_pushed": raw.get("git_commit_pushed"),
      "code_tarball_sha256": sha,
      "code_version": raw if raw else None,
  }


def _opt_int(val: str | None) -> int | None:
  if val is None or val == "":
    return None
  try:
    return int(val)
  except ValueError:
    return None


def collect_env_cluster_metadata(env: Mapping[str, str] | None = None) -> dict[str, Any]:
  """Reads cluster, platform, downward-API, and run metadata from environment variables."""
  src = env if env is not None else os.environ
  return {
      "run_id": src.get("BENCH_RUN_ID") or None,
      "experiment_name": src.get("BENCH_EXPERIMENT_NAME") or None,
      "repetition_index": _opt_int(src.get("BENCH_REPETITION_INDEX")),
      "queue_attempt_number": _opt_int(src.get("BENCH_QUEUE_ATTEMPT")),
      "jobset_restart_attempt": _opt_int(src.get("JOBSET_RESTART_ATTEMPT") or src.get("BENCH_JOBSET_RESTART_ATTEMPT")),
      "tpu_platform": src.get("BENCH_PLATFORM") or None,
      "gke_cluster_name": src.get("BENCH_CLUSTER") or None,
      "gcp_project_id": src.get("BENCH_PROJECT") or None,
      "gcp_region": src.get("BENCH_REGION") or None,
      "gcp_zone": src.get("BENCH_ZONE") or src.get("TPU_ZONE") or None,
      "gke_version": src.get("BENCH_GKE_VERSION") or None,
      "gke_node_pool_name": src.get("BENCH_NODE_POOL") or None,
      "capacity_type": src.get("BENCH_CAPACITY_TYPE") or None,
      "kueue_local_queue_name": src.get("BENCH_KUEUE_QUEUE") or None,
      "host_machine_type": src.get("BENCH_HOST_MACHINE_TYPE") or None,
      "tpu_accelerator_type": src.get("BENCH_TPU_ACCELERATOR") or src.get("TPU_ACCELERATOR_TYPE") or None,
      "container_image": src.get("BENCH_IMAGE") or None,
      "pod_name": src.get("POD_NAME") or src.get("HOSTNAME") or None,
      "pod_namespace": src.get("POD_NAMESPACE") or None,
      "node_name": src.get("NODE_NAME") or None,
      "jax_compilation_cache_state": src.get("BENCH_CACHE_STATE") or "empty",
  }


def _read_safetensors_header(path: str) -> tuple[dict[str, Any], int | None]:
  """Reads only the 8-byte header size + JSON header from a local or GCS `.safetensors` file."""
  if path.startswith("gs://"):
    from etils import epath  # pylint: disable=import-outside-toplevel

    fp = epath.Path(path)
    with fp.open("rb") as f:
      hdr_len = struct.unpack("<Q", f.read(8))[0]
      hdr = json.loads(f.read(hdr_len).decode("utf-8"))
    try:
      file_size = int(fp.stat().length)
    except Exception:  # pylint: disable=broad-exception-caught
      file_size = None
    return hdr, file_size

  file_size = os.path.getsize(path)
  with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hdr_len).decode("utf-8"))
  return hdr, file_size


def format_safetensors_dtype_summary(bytes_by_dtype: Mapping[str, int], has_scales: bool = False) -> str:
  """Formats tensor byte breakdown into canonical Sheet dtype notation (`BF16`, `F8_E4M3+scales`, ...)."""
  if not bytes_by_dtype:
    return ""
  ordered = [dtype for dtype, _ in sorted(bytes_by_dtype.items(), key=lambda kv: (-kv[1], kv[0]))]
  # Map DeepSeek packed INT8/uint8 + F8_E4M3 + scales cleanly when scale tensors exist.
  normalized: list[str] = []
  for dt in ordered:
    mapped = "FP4" if (has_scales and dt in ("I8", "U8")) else dt
    if mapped not in normalized:
      normalized.append(mapped)
  if has_scales and "scales" not in normalized:
    normalized.append("scales")
  return "+".join(normalized)


def inspect_safetensors_headers(paths: Sequence[str]) -> dict[str, Any]:
  """Reads `.safetensors` headers (without downloading weights) and computes count/size/dtype metrics."""
  bytes_by_dtype: dict[str, int] = {}
  total_data_bytes = 0
  total_file_bytes = 0
  have_all_file_sizes = True
  num_tensors = 0
  has_scales = False

  for path in sorted(paths):
    hdr, file_size = _read_safetensors_header(path)
    if file_size is None:
      have_all_file_sizes = False
    else:
      total_file_bytes += file_size

    max_end = 0
    for tensor_name, info in hdr.items():
      if tensor_name == "__metadata__" or not isinstance(info, dict):
        continue
      num_tensors += 1
      dtype = str(info.get("dtype", "UNKNOWN"))
      offsets = info.get("data_offsets") or [0, 0]
      nbytes = max(0, int(offsets[1]) - int(offsets[0]))
      max_end = max(max_end, int(offsets[1]))
      if tensor_name.endswith(_SCALE_SUFFIXES):
        has_scales = True
      else:
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + nbytes
    total_data_bytes += max_end

  # When all scale tensors were separated out and non-scale weights exist, keep scale metadata noted cleanly.
  if not bytes_by_dtype and has_scales:
    bytes_by_dtype["scales"] = total_data_bytes

  size_bytes = total_file_bytes if (have_all_file_sizes and paths) else total_data_bytes
  return {
      "checkpoint_number_of_files": len(paths),
      "checkpoint_number_of_tensors": num_tensors,
      "checkpoint_size_gb": round(size_bytes / 1e9, 4),
      "checkpoint_bytes": size_bytes,
      "bytes_by_dtype": bytes_by_dtype,
      "has_scales": has_scales,
      "hf_checkpoint_dtype": format_safetensors_dtype_summary(bytes_by_dtype, has_scales=has_scales),
  }
