# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional memory and request diagnostics for evaluation servers."""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _host_rss_bytes() -> int:
  """Return resident host memory in bytes, or -1 when unavailable."""
  try:
    with open("/proc/self/status", "r", encoding="utf-8") as status_file:
      for line in status_file:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024
  except OSError:
    pass

  try:
    import resource  # pylint: disable=import-outside-toplevel

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024
  except Exception:  # pylint: disable=broad-except
    return -1


def _hbm_stats() -> dict[str, int]:
  """Return aggregate JAX device-memory statistics."""
  try:
    import jax  # pylint: disable=import-outside-toplevel

    devices = jax.local_devices()
    stats = {"in_use": 0, "peak": 0, "limit": 0, "ndev": len(devices)}
    for device in devices:
      memory = device.memory_stats() or {}
      stats["in_use"] += memory.get("bytes_in_use", 0)
      stats["peak"] += memory.get("peak_bytes_in_use", 0)
      stats["limit"] += memory.get("bytes_limit", 0)
    return stats
  except Exception:  # pylint: disable=broad-except
    return {}


def _kv_cache_usage(llm: Any) -> float | None:
  """Return vLLM KV-cache utilization when its metrics API is available."""
  try:
    get_metrics = getattr(getattr(llm, "llm_engine", None), "get_metrics", None)
    if not callable(get_metrics):
      return None
    for metric in get_metrics():
      name = getattr(metric, "name", "")
      if "kv_cache_usage" in name or "gpu_cache_usage" in name:
        return float(getattr(metric, "value"))
  except Exception:  # pylint: disable=broad-except
    pass
  return None


def memory_summary(llm: Any) -> str:
  """Return a compact host, HBM, and KV-cache memory snapshot."""
  hbm = _hbm_stats()
  return (
      f"host_rss={_host_rss_bytes()} "
      f"hbm_in_use={hbm.get('in_use', -1)} hbm_peak={hbm.get('peak', -1)} "
      f"hbm_limit={hbm.get('limit', -1)} ndev={hbm.get('ndev', -1)} "
      f"kv_usage={_kv_cache_usage(llm)}"
  )


def log_request_diagnostics(
    llm: Any,
    request_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
) -> None:
  """Log periodic request and memory diagnostics."""
  if request_count % 100 != 0 and finish_reason != "length":
    return
  logger.info(
      "req=%d prompt_tok=%d completion_tok=%d finish=%s %s",
      request_count,
      prompt_tokens,
      completion_tokens,
      finish_reason,
      memory_summary(llm),
  )


class MemoryMonitor:
  """Optional periodic memory logger controlled by EVAL_MEM_MONITOR_SEC."""

  def __init__(self, llm: Any, rank: int):
    self._llm = llm
    self._rank = rank
    self._stop = threading.Event()
    self._thread: threading.Thread | None = None

  def start(self) -> None:
    """Start the monitor when EVAL_MEM_MONITOR_SEC is positive."""
    try:
      interval = float(os.environ.get("EVAL_MEM_MONITOR_SEC", "0") or 0)
    except ValueError:
      interval = 0.0
    if interval <= 0:
      return

    def _loop() -> None:
      while not self._stop.wait(interval):
        logger.info("MEM_MONITOR rank=%d %s", self._rank, memory_summary(self._llm))

    self._thread = threading.Thread(target=_loop, daemon=True, name="mem-monitor")
    self._thread.start()
    logger.info("Memory monitor started (rank=%d, every %.0fs).", self._rank, interval)

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=5)
      self._thread = None
