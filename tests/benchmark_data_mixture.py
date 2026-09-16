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

"""Multi-Domain PyGrain Dataset Mixture Benchmark & Profiling Harness.

This standalone harness replicates the MaxText data input pipeline for
multi-domain dataset mixtures (e.g. 35 domains) and profiles:
1. Thread & thread pool counts per process (parent + mp_prefetch worker children).
2. Process RSS memory footprint trajectory over steps (100–1,000 steps).
3. Step processing latency (ms/step) and batch throughput (samples/sec & tokens/sec).
4. Checkpoint state structure (get_state/set_state) and worker-count elasticity (W=2 <-> W=8).

Supports direct comparison between:
- Baseline: grain.IterDataset.mix (Stock MaxText, creates 35 thread pools per worker)
- Candidate: grain.MapDataset.mix (Proposed fix, creates 1 thread pool per worker)
- Elastic: grain.MapDataset.mix + ElasticIterator

Usage:
  # Compare baseline vs candidate across worker counts
  python tests/benchmark_data_mixture.py --mode=compare --steps=200 --worker-counts=2,4,8

  # Profile memory trajectory over 500 steps
  python tests/benchmark_data_mixture.py --mode=profile --pipeline=map_mix --steps=500 --worker-count=8

  # Test checkpoint serialization and worker transition elasticity
  python tests/benchmark_data_mixture.py --mode=checkpoint

  # Run all benchmarks and export JSON report
  python tests/benchmark_data_mixture.py --mode=all --output-json=benchmark_report.json
"""

import argparse
import dataclasses
import functools
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure MaxText repo and src are on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
  import psutil
  HAS_PSUTIL = True
except ImportError:
  HAS_PSUTIL = False

from absl import flags as absl_flags

if not absl_flags.FLAGS.is_parsed():
  absl_flags.FLAGS(sys.argv, known_only=True)

import grain.python as grain
from grain.experimental import BestFitPackIterDataset, FirstFitPackIterDataset, ElasticIterator

from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import grain_tokenizer
from maxtext.input_pipeline import tokenizer
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_TEST_ASSETS_ROOT


# ==============================================================================
# Process & Resource Tracker
# ==============================================================================

@dataclasses.dataclass
class ProcessMetrics:
  """Snapshot of process resource utilization."""
  timestamp: float
  step: int
  main_pid: int
  main_rss_mb: float
  main_threads: int
  num_worker_processes: int
  worker_pids: List[int]
  worker_rss_mb: List[float]
  worker_thread_counts: List[int]
  total_rss_mb: float
  total_threads: int
  total_fds: int


class ResourceTracker:
  """Tracks memory RSS, OS threads, and FDs across parent and child processes."""

  def __init__(self, main_pid: Optional[int] = None):
    self.main_pid = main_pid or os.getpid()
    self._page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

  def _read_proc_statm(self, pid: int) -> float:
    """Reads RSS memory in MB from /proc/<pid>/statm."""
    try:
      with open(f"/proc/{pid}/statm", "r") as f:
        fields = f.read().strip().split()
        rss_pages = int(fields[1])
        return (rss_pages * self._page_size) / (1024 * 1024)
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
      return 0.0

  def _read_proc_threads(self, pid: int) -> int:
    """Reads OS thread count from /proc/<pid>/status."""
    try:
      with open(f"/proc/{pid}/status", "r") as f:
        for line in f:
          if line.startswith("Threads:"):
            return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      pass
    return 0

  def _read_proc_fds(self, pid: int) -> int:
    """Counts open file descriptors in /proc/<pid>/fd."""
    try:
      return len(os.listdir(f"/proc/{pid}/fd"))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      return 0

  def _get_child_pids(self) -> List[int]:
    """Finds all child worker process PIDs."""
    if HAS_PSUTIL:
      try:
        parent = psutil.Process(self.main_pid)
        return [p.pid for p in parent.children(recursive=True)]
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    # Fallback scanning /proc
    child_pids = []
    try:
      for entry in os.listdir("/proc"):
        if entry.isdigit():
          pid = int(entry)
          try:
            with open(f"/proc/{pid}/stat", "r") as f:
              stat_line = f.read()
              ppid = int(stat_line.split()[3])
              if ppid == self.main_pid:
                child_pids.append(pid)
          except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    except Exception:
      pass
    return child_pids

  def sample(self, step: int = 0) -> ProcessMetrics:
    """Captures a point-in-time resource snapshot."""
    main_rss = self._read_proc_statm(self.main_pid)
    main_threads = self._read_proc_threads(self.main_pid)
    main_fds = self._read_proc_fds(self.main_pid)

    child_pids = self._get_child_pids()
    worker_rss = []
    worker_threads = []
    total_fds = main_fds

    for cpid in child_pids:
      crss = self._read_proc_statm(cpid)
      cthr = self._read_proc_threads(cpid)
      cfds = self._read_proc_fds(cpid)
      worker_rss.append(crss)
      worker_threads.append(cthr)
      total_fds += cfds

    total_rss = main_rss + sum(worker_rss)
    total_threads = main_threads + sum(worker_threads)

    return ProcessMetrics(
        timestamp=time.time(),
        step=step,
        main_pid=self.main_pid,
        main_rss_mb=round(main_rss, 2),
        main_threads=main_threads,
        num_worker_processes=len(child_pids),
        worker_pids=child_pids,
        worker_rss_mb=[round(r, 2) for r in worker_rss],
        worker_thread_counts=worker_threads,
        total_rss_mb=round(total_rss, 2),
        total_threads=total_threads,
        total_fds=total_fds,
    )


# ==============================================================================
# Pipeline Construction Helpers
# ==============================================================================

def resolve_mixture_config(mixture_config_path: Optional[str]) -> Tuple[Dict[str, Dict[str, Any]], str]:
  """Loads or generates mixture config and resolves file paths."""
  if mixture_config_path and os.path.exists(mixture_config_path):
    with open(mixture_config_path, "r", encoding="utf-8") as f:
      config = json.load(f)
    actual_path = mixture_config_path
  else:
    # Try default test fixture
    fixture_path = REPO_ROOT / "tests" / "assets" / "test_35_domain_mixture.json"
    if fixture_path.exists():
      with open(fixture_path, "r", encoding="utf-8") as f:
        config = json.load(f)
      actual_path = str(fixture_path)
    else:
      from tests.generate_mixture_config import generate_mixture_config
      actual_path = str(fixture_path)
      config = generate_mixture_config(
          num_domains=35,
          output_path=actual_path,
      )

  # Resolve relative paths against REPO_ROOT
  resolved_config = {}
  for domain_name, entry in config.items():
    raw_path = entry["path"]
    if not os.path.isabs(raw_path) and not raw_path.startswith("gs://"):
      resolved_path = str((REPO_ROOT / raw_path).resolve())
    else:
      resolved_path = raw_path
    resolved_config[domain_name] = {
        "path": resolved_path,
        "weight": float(entry["weight"]),
    }

  return resolved_config, actual_path


def create_dataset_from_pattern(pattern: str) -> grain.MapDataset:
  """Creates an ArrayRecord MapDataset source from a glob pattern."""
  import glob
  files = glob.glob(str(Path(pattern).expanduser().resolve()))
  if not files:
    raise FileNotFoundError(f"No files found matching dataset pattern: {pattern}")
  source = grain.ArrayRecordDataSource(files)
  return grain.MapDataset.source(source)


def apply_maxtext_transforms(
    dataset: Union[grain.MapDataset, grain.IterDataset],
    tokenizer_model: Any,
    pad_id: int,
    max_target_length: int = 1024,
    packing_type: str = "best_fit",
    batch_size: int = 8,
    worker_count: int = 2,
    per_worker_buffer_size: int = 2,
    is_elastic: bool = False,
) -> Union[grain.MapDataset, grain.IterDataset]:
  """Applies ParseFeatures, TokenizeAndTrim, Rekey, Packing, Batching, and mp_prefetch."""
  # 1. Parse proto features
  dataset = dataset.map(input_pipeline_utils.ParseFeatures(data_columns=["text"], tokenize=True))
  dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(column_names=["text"], tokenize=True))

  # 2. Tokenize and Trim
  dataset = dataset.map(grain_tokenizer.TokenizeAndTrim("text", max_target_length, tokenizer_model))

  # 3. Rekey to inputs and targets
  dataset = dataset.map(input_pipeline_utils.Rekey({"inputs": "text", "targets": "text"}))

  if is_elastic:
    # ElasticIterator applies batching and multiprocessing internally
    dataset = dataset.map(input_pipeline_utils.PadOrTrimToMaxLength(max_target_length, pad_id))
    dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=0))
    return dataset

  # 4. Packing
  if packing_type == "best_fit":
    length_struct = {"inputs": max_target_length, "targets": max_target_length}
    dataset = BestFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=batch_size)
    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
  elif packing_type == "first_fit":
    length_struct = {"inputs": max_target_length, "targets": max_target_length}
    dataset = FirstFitPackIterDataset(dataset, length_struct=length_struct, num_packing_bins=batch_size)
    rekey_dict = {
        "targets_segmentation": "targets_segment_ids",
        "inputs_segmentation": "inputs_segment_ids",
        "targets_position": "targets_positions",
        "inputs_position": "inputs_positions",
    }
    dataset = dataset.map(input_pipeline_utils.Rekey(rekey_dict))
  elif packing_type == "none":
    dataset = dataset.map(input_pipeline_utils.PadOrTrimToMaxLength(max_target_length, pad_id))
  else:
    raise ValueError(f"Unsupported packing type: {packing_type}")

  # 5. Batching
  batch_fn = functools.partial(grain.experimental.batch_and_pad, batch_size=batch_size, pad_value=pad_id)
  dataset = dataset.batch(batch_size, batch_fn=batch_fn)

  # 6. Next-token prediction shift
  dataset = dataset.map(input_pipeline_utils.ShiftData(ignored_ids=[pad_id], axis=1))

  # 7. Multiprocessing prefetch
  if worker_count > 0:
    mp_opts = grain.MultiprocessingOptions(
        num_workers=worker_count,
        per_worker_buffer_size=per_worker_buffer_size,
    )
    dataset = dataset.mp_prefetch(mp_opts)

  return dataset


def build_pipeline_iter_mix(
    mixture_config: Dict[str, Dict[str, Any]],
    tokenizer_model: Any,
    pad_id: int,
    max_target_length: int = 1024,
    packing_type: str = "best_fit",
    batch_size: int = 8,
    worker_count: int = 2,
    per_worker_buffer_size: int = 2,
    host_index: int = 0,
    host_count: int = 1,
    num_epoch: int = 1,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_threads_per_domain: int = 1,
    domain_prefetch_buffer_size: int = 1,
) -> grain.IterDataset:
  """Builds the baseline pipeline using grain.IterDataset.mix (Stock MaxText)."""
  datasets_dict = {}
  weights_dict = {}
  total_weight = sum(entry["weight"] for entry in mixture_config.values())

  for domain_name, entry in mixture_config.items():
    ds = create_dataset_from_pattern(entry["path"])
    if shuffle:
      ds = ds.shuffle(seed=shuffle_seed)
    ds = ds.repeat(num_epoch)
    ds = ds[host_index::host_count]
    # Stock MaxText applies to_iter_dataset to every domain individually
    ds = ds.to_iter_dataset(
        read_options=grain.ReadOptions(
            num_threads=num_threads_per_domain,
            prefetch_buffer_size=domain_prefetch_buffer_size,
        )
    )
    datasets_dict[domain_name] = ds
    weights_dict[domain_name] = entry["weight"] / total_weight

  # Mix at IterDataset level (creates 35 sub-iterators per worker)
  mixed_iter_ds = grain.IterDataset.mix(datasets_dict, weights_dict)

  return apply_maxtext_transforms(
      mixed_iter_ds,
      tokenizer_model=tokenizer_model,
      pad_id=pad_id,
      max_target_length=max_target_length,
      packing_type=packing_type,
      batch_size=batch_size,
      worker_count=worker_count,
      per_worker_buffer_size=per_worker_buffer_size,
      is_elastic=False,
  )


def build_pipeline_map_mix(
    mixture_config: Dict[str, Dict[str, Any]],
    tokenizer_model: Any,
    pad_id: int,
    max_target_length: int = 1024,
    packing_type: str = "best_fit",
    batch_size: int = 8,
    worker_count: int = 2,
    per_worker_buffer_size: int = 2,
    host_index: int = 0,
    host_count: int = 1,
    num_epoch: int = 1,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_threads_total: int = 1,
    total_prefetch_buffer_size: int = 1,
) -> grain.IterDataset:
  """Builds the candidate pipeline using grain.MapDataset.mix (Proposed fix)."""
  dataset_list = []
  weights_list = []

  for entry in mixture_config.values():
    ds = create_dataset_from_pattern(entry["path"])
    if shuffle:
      ds = ds.shuffle(seed=shuffle_seed)
    ds = ds.repeat(num_epoch)
    dataset_list.append(ds)
    weights_list.append(float(entry["weight"]))

  total_weight = sum(weights_list)
  normalized_weights = [w / total_weight for w in weights_list]

  # 1. Mix at MapDataset level (pure index mapping, 0 threads)
  mixed_map_ds = grain.MapDataset.mix(dataset_list, weights=normalized_weights)

  # 2. Shard and convert to IterDataset ONCE for the entire mixture
  mixed_map_ds = mixed_map_ds[host_index::host_count]
  mixed_iter_ds = mixed_map_ds.to_iter_dataset(
      read_options=grain.ReadOptions(
          num_threads=num_threads_total,
          prefetch_buffer_size=total_prefetch_buffer_size,
      )
  )

  return apply_maxtext_transforms(
      mixed_iter_ds,
      tokenizer_model=tokenizer_model,
      pad_id=pad_id,
      max_target_length=max_target_length,
      packing_type=packing_type,
      batch_size=batch_size,
      worker_count=worker_count,
      per_worker_buffer_size=per_worker_buffer_size,
      is_elastic=False,
  )


def build_pipeline_elastic_map_mix(
    mixture_config: Dict[str, Dict[str, Any]],
    tokenizer_model: Any,
    pad_id: int,
    max_target_length: int = 1024,
    batch_size: int = 8,
    worker_count: int = 2,
    per_worker_buffer_size: int = 2,
    host_index: int = 0,
    host_count: int = 1,
    num_epoch: int = 1,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    num_threads_total: int = 1,
    total_prefetch_buffer_size: int = 1,
) -> ElasticIterator:
  """Builds the candidate pipeline using MapDataset.mix + ElasticIterator."""
  dataset_list = []
  weights_list = []

  for entry in mixture_config.values():
    ds = create_dataset_from_pattern(entry["path"])
    if shuffle:
      ds = ds.shuffle(seed=shuffle_seed)
    ds = ds.repeat(num_epoch)
    dataset_list.append(ds)
    weights_list.append(float(entry["weight"]))

  total_weight = sum(weights_list)
  normalized_weights = [w / total_weight for w in weights_list]

  mixed_map_ds = grain.MapDataset.mix(dataset_list, weights=normalized_weights)

  preprocessed_map_ds = apply_maxtext_transforms(
      mixed_map_ds,
      tokenizer_model=tokenizer_model,
      pad_id=pad_id,
      max_target_length=max_target_length,
      packing_type="none",
      batch_size=batch_size,
      worker_count=worker_count,
      per_worker_buffer_size=per_worker_buffer_size,
      is_elastic=True,
  )

  mp_options = (
      grain.MultiprocessingOptions(
          num_workers=worker_count,
          per_worker_buffer_size=per_worker_buffer_size,
      )
      if worker_count > 0
      else None
  )

  return ElasticIterator(
      preprocessed_map_ds,
      global_batch_size=batch_size * host_count,
      shard_options=grain.ShardOptions(
          shard_index=host_index,
          shard_count=host_count,
      ),
      read_options=grain.ReadOptions(
          num_threads=num_threads_total,
          prefetch_buffer_size=total_prefetch_buffer_size,
      ),
      multiprocessing_options=mp_options,
  )


# ==============================================================================
# Benchmarking & Profiling Runner
# ==============================================================================

def run_pipeline_benchmark(
    pipeline_name: str,
    pipeline_ds: Union[grain.IterDataset, ElasticIterator],
    num_steps: int = 100,
    warmup_steps: int = 10,
    sample_every_n: int = 10,
    batch_size: int = 8,
    max_target_length: int = 1024,
    verbose: bool = False,
) -> Dict[str, Any]:
  """Executes steps on the pipeline and tracks latency, throughput, threads, and memory."""
  tracker = ResourceTracker()
  initial_metrics = tracker.sample(step=0)

  print(f"\n[{pipeline_name}] Starting warmup ({warmup_steps} steps)...")
  iterator = iter(pipeline_ds)

  # Warmup
  for w in range(warmup_steps):
    _ = next(iterator)

  warmup_metrics = tracker.sample(step=warmup_steps)
  print(
      f"[{pipeline_name}] Warmup complete. Workers: {warmup_metrics.num_worker_processes}, "
      f"Total Threads: {warmup_metrics.total_threads}, Initial RSS: {warmup_metrics.total_rss_mb:.1f} MB"
  )

  step_times_ms: List[float] = []
  memory_trajectory: List[ProcessMetrics] = [warmup_metrics]

  start_benchmark_time = time.time()
  print(f"[{pipeline_name}] Running {num_steps} benchmark steps...")

  for step in range(1, num_steps + 1):
    t0 = time.perf_counter()
    batch = next(iterator)
    t1 = time.perf_counter()
    step_time_ms = (t1 - t0) * 1000.0
    step_times_ms.append(step_time_ms)

    if step % sample_every_n == 0 or step == num_steps:
      current_metrics = tracker.sample(step=step)
      memory_trajectory.append(current_metrics)
      if verbose:
        print(
            f"  Step {step:4d}/{num_steps}: {step_time_ms:6.1f} ms | "
            f"RSS: {current_metrics.total_rss_mb:7.1f} MB (Main: {current_metrics.main_rss_mb:.1f} MB, "
            f"Workers: {sum(current_metrics.worker_rss_mb):.1f} MB) | "
            f"Threads: {current_metrics.total_threads:3d} (Worker avg: "
            f"{np.mean(current_metrics.worker_thread_counts) if current_metrics.worker_thread_counts else 0:.1f})"
        )

  total_benchmark_time = time.time() - start_benchmark_time

  # Close / cleanup iterator
  if hasattr(iterator, "close"):
    try:
      iterator.close()
    except Exception:
      pass
  del iterator
  del pipeline_ds
  gc.collect()

  # Compute statistics
  step_times_arr = np.array(step_times_ms)
  total_samples = num_steps * batch_size
  total_tokens = total_samples * max_target_length
  samples_per_sec = total_samples / total_benchmark_time
  tokens_per_sec = total_tokens / total_benchmark_time

  final_metrics = memory_trajectory[-1]
  initial_rss = warmup_metrics.total_rss_mb
  final_rss = final_metrics.total_rss_mb
  peak_rss = max(m.total_rss_mb for m in memory_trajectory)
  rss_delta = final_rss - initial_rss
  rss_slope_mb_per_step = rss_delta / num_steps if num_steps > 0 else 0.0

  results = {
      "pipeline_name": pipeline_name,
      "num_steps": num_steps,
      "warmup_steps": warmup_steps,
      "batch_size": batch_size,
      "max_target_length": max_target_length,
      "total_time_sec": round(total_benchmark_time, 3),
      "latency_ms": {
          "mean": round(float(np.mean(step_times_arr)), 2),
          "std": round(float(np.std(step_times_arr)), 2),
          "p50": round(float(np.percentile(step_times_arr, 50)), 2),
          "p90": round(float(np.percentile(step_times_arr, 90)), 2),
          "p99": round(float(np.percentile(step_times_arr, 99)), 2),
          "min": round(float(np.min(step_times_arr)), 2),
          "max": round(float(np.max(step_times_arr)), 2),
      },
      "throughput": {
          "samples_per_sec": round(samples_per_sec, 2),
          "tokens_per_sec": round(tokens_per_sec, 2),
      },
      "memory_mb": {
          "initial_rss": initial_rss,
          "final_rss": final_rss,
          "peak_rss": peak_rss,
          "growth_mb": round(rss_delta, 2),
          "growth_rate_mb_per_step": round(rss_slope_mb_per_step, 4),
      },
      "threads": {
          "total_threads_final": final_metrics.total_threads,
          "main_threads": final_metrics.main_threads,
          "num_worker_processes": final_metrics.num_worker_processes,
          "worker_thread_counts": final_metrics.worker_thread_counts,
          "worker_threads_per_proc": (
              round(float(np.mean(final_metrics.worker_thread_counts)), 1)
              if final_metrics.worker_thread_counts
              else 0
          ),
      },
      "memory_trajectory": [dataclasses.asdict(m) for m in memory_trajectory],
  }

  return results


# ==============================================================================
# Checkpoint & Elasticity Validator
# ==============================================================================

def test_checkpoint_compatibility(
    mixture_config: Dict[str, Dict[str, Any]],
    tokenizer_model: Any,
    pad_id: int,
    batch_size: int = 4,
    max_target_length: int = 128,
) -> Dict[str, Any]:
  """Validates checkpoint get_state/set_state structure and worker elasticity (W=2 <-> W=8)."""
  print("\n" + "=" * 80)
  print("STAGE 3: CHECKPOINT STRUCTURE & WORKER-COUNT ELASTICITY VALIDATION")
  print("=" * 80)

  results: Dict[str, Any] = {}

  for pipeline_type in ["iter_mix", "map_mix"]:
    print(f"\n--- Testing Checkpoints for {pipeline_type} ---")
    builder_fn = build_pipeline_iter_mix if pipeline_type == "iter_mix" else build_pipeline_map_mix

    # 1. Build pipeline with W=2 and advance 10 steps
    ds_w2 = builder_fn(
        mixture_config,
        tokenizer_model=tokenizer_model,
        pad_id=pad_id,
        batch_size=batch_size,
        max_target_length=max_target_length,
        worker_count=2,
    )
    iter_w2 = iter(ds_w2)
    for _ in range(10):
      _ = next(iter_w2)

    # 2. Extract State & Inspect Schema
    state_w2 = iter_w2.get_state()
    state_repr = str(state_w2)
    state_size_bytes = len(state_repr.encode("utf-8"))

    # Count sub-iterator states or domain entries in state dict
    state_keys = list(state_w2.keys()) if isinstance(state_w2, dict) else []
    print(f"[{pipeline_type}] State size: {state_size_bytes:,} bytes, top-level keys: {state_keys}")

    # 3. Test Save/Restore with Same Worker Count (W=2 -> W=2)
    ds_w2_restore = builder_fn(
        mixture_config,
        tokenizer_model=tokenizer_model,
        pad_id=pad_id,
        batch_size=batch_size,
        max_target_length=max_target_length,
        worker_count=2,
    )
    iter_w2_restore = iter(ds_w2_restore)
    iter_w2_restore.set_state(state_w2)

    batch_orig = next(iter_w2)
    batch_restored = next(iter_w2_restore)

    deterministic = bool(np.array_equal(batch_orig["inputs"], batch_restored["inputs"]))
    print(f"[{pipeline_type}] Same-worker restore (W=2 -> W=2) determinism: {deterministic}")

    # 4. Test Worker Elasticity Transition (W=2 -> W=8)
    elasticity_w2_to_w8_success = False
    elasticity_w2_to_w8_error = None

    try:
      ds_w8 = builder_fn(
          mixture_config,
          tokenizer_model=tokenizer_model,
          pad_id=pad_id,
          batch_size=batch_size,
          max_target_length=max_target_length,
          worker_count=8,
      )
      iter_w8 = iter(ds_w8)
      iter_w8.set_state(state_w2)
      _ = next(iter_w8)
      elasticity_w2_to_w8_success = True
      print(f"[{pipeline_type}] Worker count transition (W=2 -> W=8): SUCCESS")
    except Exception as e:
      elasticity_w2_to_w8_error = f"{type(e).__name__}: {e}"
      print(f"[{pipeline_type}] Worker count transition (W=2 -> W=8): FAILED ({elasticity_w2_to_w8_error})")

    # 5. Test Worker Elasticity Transition (W=8 -> W=2)
    ds_w8_base = builder_fn(
        mixture_config,
        tokenizer_model=tokenizer_model,
        pad_id=pad_id,
        batch_size=batch_size,
        max_target_length=max_target_length,
        worker_count=8,
    )
    iter_w8_base = iter(ds_w8_base)
    for _ in range(10):
      _ = next(iter_w8_base)
    state_w8 = iter_w8_base.get_state()

    elasticity_w8_to_w2_success = False
    elasticity_w8_to_w2_error = None
    try:
      ds_w2_target = builder_fn(
          mixture_config,
          tokenizer_model=tokenizer_model,
          pad_id=pad_id,
          batch_size=batch_size,
          max_target_length=max_target_length,
          worker_count=2,
      )
      iter_w2_target = iter(ds_w2_target)
      iter_w2_target.set_state(state_w8)
      _ = next(iter_w2_target)
      elasticity_w8_to_w2_success = True
      print(f"[{pipeline_type}] Worker count transition (W=8 -> W=2): SUCCESS")
    except Exception as e:
      elasticity_w8_to_w2_error = f"{type(e).__name__}: {e}"
      print(f"[{pipeline_type}] Worker count transition (W=8 -> W=2): FAILED ({elasticity_w8_to_w2_error})")

    results[pipeline_type] = {
        "state_size_bytes": state_size_bytes,
        "same_worker_restore_deterministic": deterministic,
        "transition_w2_to_w8": {
            "success": elasticity_w2_to_w8_success,
            "error": elasticity_w2_to_w8_error,
        },
        "transition_w8_to_w2": {
            "success": elasticity_w8_to_w2_success,
            "error": elasticity_w8_to_w2_error,
        },
    }

  return results


# ==============================================================================
# Summary Formatter
# ==============================================================================

def print_comparison_table(all_results: List[Dict[str, Any]]):
  """Prints a clean tabular comparison of benchmark results."""
  header = (
      f"{'Pipeline':<18} | {'Workers':<7} | {'Threads/Wk':<10} | {'Tot Thr':<7} | "
      f"{'Peak RSS':<10} | {'Growth/Step':<12} | {'Lat (ms)':<9} | {'Throughput':<12}"
  )
  separator = "-" * len(header)
  print("\n" + separator)
  print("MAXTEXT 35-DOMAIN DATA MIXTURE BENCHMARK RESULTS SUMMARY")
  print(separator)
  print(header)
  print(separator)

  for res in all_results:
    pipe_name = res["pipeline_name"]
    workers = res["threads"]["num_worker_processes"]
    thr_per_worker = res["threads"]["worker_threads_per_proc"]
    tot_thr = res["threads"]["total_threads_final"]
    peak_rss = f"{res['memory_mb']['peak_rss']:.1f} MB"
    growth = f"{res['memory_mb']['growth_rate_mb_per_step']:.3f} MB/st"
    lat_p50 = f"{res['latency_ms']['p50']:.1f}"
    throughput = f"{res['throughput']['samples_per_sec']:.1f} smp/s"

    print(
        f"{pipe_name:<18} | {workers:<7d} | {thr_per_worker:<10.1f} | {tot_thr:<7d} | "
        f"{peak_rss:<10} | {growth:<12} | {lat_p50:<9} | {throughput:<12}"
    )

  print(separator + "\n")


# ==============================================================================
# Main Entrypoint
# ==============================================================================

def main():
  parser = argparse.ArgumentParser(
      description="MaxText Multi-Domain PyGrain Dataset Mixture Benchmark & Profiling Harness"
  )
  parser.add_argument(
      "--mode",
      type=str,
      choices=["profile", "compare", "checkpoint", "all"],
      default="all",
      help="Execution mode (default: all)",
  )
  parser.add_argument(
      "--pipeline",
      type=str,
      choices=["iter_mix", "map_mix", "elastic_map_mix", "both"],
      default="both",
      help="Pipeline implementation to test (default: both)",
  )
  parser.add_argument(
      "--worker-counts",
      type=str,
      default="2,8",
      help="Comma-separated list of worker counts to evaluate (default: '2,8')",
  )
  parser.add_argument(
      "--steps",
      type=int,
      default=100,
      help="Number of benchmark steps to run per configuration (default: 100)",
  )
  parser.add_argument(
      "--warmup-steps",
      type=int,
      default=10,
      help="Number of warmup steps (default: 10)",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=8,
      help="Local batch size (default: 8)",
  )
  parser.add_argument(
      "--max-target-length",
      type=int,
      default=1024,
      help="Sequence length (default: 1024)",
  )
  parser.add_argument(
      "--packing-type",
      type=str,
      choices=["best_fit", "first_fit", "none"],
      default="best_fit",
      help="Packing strategy (default: best_fit)",
  )
  parser.add_argument(
      "--mixture-config",
      type=str,
      default=None,
      help="Path to 35-domain mixture JSON config file",
  )
  parser.add_argument(
      "--sample-every-n",
      type=int,
      default=10,
      help="Sample memory and threads every N steps (default: 10)",
  )
  parser.add_argument(
      "--output-json",
      type=str,
      default="benchmark_report.json",
      help="File path to write benchmark JSON report (default: benchmark_report.json)",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Enable verbose step-by-step logging",
  )

  args = parser.parse_args()

  # Load / resolve mixture config
  mixture_config, config_path = resolve_mixture_config(args.mixture_config)
  print(f"Loaded {len(mixture_config)} mixture domains from: {config_path}")

  # Initialize tokenizer
  tokenizer_path = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default")
  if not os.path.exists(tokenizer_path):
    tokenizer_path = str(REPO_ROOT / "src" / "maxtext" / "assets" / "tokenizers" / "tokenizer.default")

  tokenizer_model = tokenizer.build_tokenizer(
      tokenizer_path=tokenizer_path,
      tokenizer_type="sentencepiece",
      add_bos=True,
      add_eos=True,
      hf_access_token="",
  )
  pad_id = getattr(tokenizer_model, "pad_id", 0) or 0

  worker_counts = [int(w.strip()) for w in args.worker_counts.split(",") if w.strip()]
  all_benchmark_results = []
  checkpoint_results = {}

  if args.mode in ("profile", "compare", "all"):
    pipelines_to_run = []
    if args.pipeline in ("iter_mix", "both"):
      pipelines_to_run.append("iter_mix")
    if args.pipeline in ("map_mix", "both"):
      pipelines_to_run.append("map_mix")
    if args.pipeline == "elastic_map_mix":
      pipelines_to_run.append("elastic_map_mix")

    print("\n" + "=" * 80)
    print("STAGES 1 & 2: RESOURCE & THROUGHPUT PROFILING")
    print(f"Worker Counts: {worker_counts} | Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.max_target_length}")
    print("=" * 80)

    for w in worker_counts:
      for pipe_type in pipelines_to_run:
        label = f"{pipe_type}_W{w}"
        print(f"\n>>> Constructing {label} (worker_count={w})...")

        if pipe_type == "iter_mix":
          ds = build_pipeline_iter_mix(
              mixture_config,
              tokenizer_model=tokenizer_model,
              pad_id=pad_id,
              max_target_length=args.max_target_length,
              packing_type=args.packing_type,
              batch_size=args.batch_size,
              worker_count=w,
          )
        elif pipe_type == "map_mix":
          ds = build_pipeline_map_mix(
              mixture_config,
              tokenizer_model=tokenizer_model,
              pad_id=pad_id,
              max_target_length=args.max_target_length,
              packing_type=args.packing_type,
              batch_size=args.batch_size,
              worker_count=w,
          )
        elif pipe_type == "elastic_map_mix":
          ds = build_pipeline_elastic_map_mix(
              mixture_config,
              tokenizer_model=tokenizer_model,
              pad_id=pad_id,
              max_target_length=args.max_target_length,
              batch_size=args.batch_size,
              worker_count=w,
          )

        res = run_pipeline_benchmark(
            pipeline_name=label,
            pipeline_ds=ds,
            num_steps=args.steps,
            warmup_steps=args.warmup_steps,
            sample_every_n=args.sample_every_n,
            batch_size=args.batch_size,
            max_target_length=args.max_target_length,
            verbose=args.verbose,
        )
        all_benchmark_results.append(res)

    print_comparison_table(all_benchmark_results)

  if args.mode in ("checkpoint", "all"):
    checkpoint_results = test_checkpoint_compatibility(
        mixture_config=mixture_config,
        tokenizer_model=tokenizer_model,
        pad_id=pad_id,
        batch_size=args.batch_size,
        max_target_length=128,  # Lightweight sequence length for state checks
    )

  # Final report export
  report = {
      "timestamp": time.time(),
      "num_domains": len(mixture_config),
      "mixture_config_path": config_path,
      "benchmark_runs": all_benchmark_results,
      "checkpoint_validation": checkpoint_results,
  }

  if args.output_json:
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
      output_path = REPO_ROOT / output_path
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)
    print(f"\nFull benchmark report exported to: {output_path}")

  print("\nBenchmark completed successfully.")


if __name__ == "__main__":
  main()
