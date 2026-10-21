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

"""Benchmark GCS QPS of MaxText PyGrain + ArrayRecord input pipelines (Text & Multimodal).

Runs on CPU VMs without requiring TPUs (`JAX_PLATFORMS=cpu`), comparing:
- Modality:
  1. `text`: Pre-training pipeline (`pretrain_preprocessing_pipeline`) with sequence packing
     (`FirstFitPackIterDataset`) over variable-length log-normal documents.
  2. `multimodal`: Vision SFT pipeline (`vision_sft_preprocessing_pipeline`) with `gemma3-4b`
     image decoding, resizing/tiling, and token fusion.
- Storage Backend (both 100% measured from real C++ execution):
  1. `direct_gcs`: Reads via `gs://...` using C++ `riegeli::gcs::GcsReader` + `google-cloud-cpp`
     against `InstrumentedGCSServer`, classifying every HTTP GET deterministically by exact
     byte offset (`footer_offset`) into:
     - `init_probe_shard_open` (`ArrayRecordDataSource.__init__` `num_records()` probe)
     - `lazy_worker_shard_open` (`_ensure_reader_exists` per-worker lazy reader open)
     - `record_read` (`[0, footer_offset)` data record chunk read)
  2. `gcsfuse`: Reads via POSIX `riegeli::FdReader` (`pread64`), traced live via
     `PosixGCSFuseTracer` (`strace -f -y -e trace=pread64`) across the parent process and
     all `mp_prefetch` child worker processes, and replayed through a stateful GCSFuse
     streaming read-ahead and local file-cache state machine.
- Phases:
  Separates `init` (`ArrayRecordDataSource.__init__`), `first_batch` (Step 0 worker spawn &
  prefetch fill), `warmup` (`--warmup_steps` steps), and `steady` (`--steps` measured steps),
  reporting both **True Steady-State QPS (Record-Read Only)** and **Early-Step QPS (including
  Lazy Shard Opens)**.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import tempfile
import time

from absl import flags

flags.FLAGS.mark_as_parsed()

# Force CPU execution before importing JAX so the benchmark runs on any GCE CPU VM
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from benchmarks.input_pipeline import gcs_qps_tracer
from benchmarks.input_pipeline import generate_synthetic_arrayrecords
from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing

DEFAULT_GCS_BUCKET_QPS_LIMIT = 50_000


def _build_maxtext_config(
    modality: str,
    train_files_pattern: str,
    per_host_batch_size: int,
    max_target_length: int,
    grain_worker_count: int,
    grain_per_worker_buffer_size: int,
    grain_num_threads: int,
    grain_prefetch_buffer_size: int,
    enable_data_shuffling: bool = True,
):
  """Initializes a MaxText `pyconfig` for either `text` pre-training or `multimodal` vision training."""
  maxtext_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  if modality == "text":
    base_yml = os.path.join(maxtext_root, "src", "maxtext", "configs", "base.yml")
    tok_path = os.path.join(maxtext_root, "src", "maxtext", "assets", "tokenizers", "tokenizer.default")
    argv = [
        "",
        base_yml,
        "run_name=qps_bench_text",
        f"per_device_batch_size={per_host_batch_size}",
        f"max_target_length={max_target_length}",
        "dataset_type=grain",
        "grain_file_type=arrayrecord",
        f"grain_train_files={train_files_pattern}",
        'train_data_columns=["text"]',
        "tokenize_train_data=True",
        f"tokenizer_path={tok_path}",
        "packing=True",
        "grain_packing_type=first_fit",
        "num_epoch=1000",
        f"enable_data_shuffling={enable_data_shuffling}",
        f"grain_worker_count={grain_worker_count}",
        f"grain_per_worker_buffer_size={grain_per_worker_buffer_size}",
        f"grain_num_threads={grain_num_threads}",
        f"grain_prefetch_buffer_size={grain_prefetch_buffer_size}",
        "enable_checkpointing=False",
        "skip_jax_distributed_system=True",
    ]
  elif modality == "multimodal":
    sft_yml = os.path.join(maxtext_root, "src", "maxtext", "configs", "post_train", "sft-vision-chartqa.yml")
    tok_path = os.path.join(maxtext_root, "src", "maxtext", "assets", "tokenizers", "tokenizer.gemma3")
    argv = [
        "",
        sft_yml,
        "run_name=qps_bench_multimodal",
        "model_name=gemma3-4b",
        f"per_device_batch_size={per_host_batch_size}",
        f"max_target_length={max_target_length}",
        f"max_prefill_predict_length={max_target_length // 2}",
        "dataset_type=grain",
        "grain_file_type=arrayrecord",
        f"grain_train_files={train_files_pattern}",
        'train_data_columns=["question","answer"]',
        "train_image_column=image",
        "tokenize_train_data=True",
        f"tokenizer_path={tok_path}",
        "use_tunix_gradient_accumulation=False",
        "num_epoch=1000",
        f"enable_data_shuffling={enable_data_shuffling}",
        f"grain_worker_count={grain_worker_count}",
        f"grain_per_worker_buffer_size={grain_per_worker_buffer_size}",
        f"grain_num_threads={grain_num_threads}",
        f"grain_prefetch_buffer_size={grain_prefetch_buffer_size}",
        "enable_checkpointing=False",
        "skip_jax_distributed_system=True",
    ]
  else:
    raise ValueError(f"Unsupported modality: {modality}")

  return pyconfig.initialize(argv)


def _create_host_iterator(config, host_index: int, emulated_host_count: int):
  """Builds the exact Grain dataset + preprocessing iterator for a single emulated host."""
  pipeline_fn = grain_data_processing._get_pipeline_fn(config)  # pylint: disable=protected-access
  dataset_config = grain_data_processing._build_dataset_config(  # pylint: disable=protected-access
      config,
      num_samples=None,
      seed=config.data_shuffle_seed,
      split_ratio=None,
      split_index=0,
  )
  ds = grain_data_processing.get_datasets(
      data_file_pattern=config.grain_train_files,
      data_file_type=config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=config.num_epoch,
      dataloading_host_index=host_index,
      dataloading_host_count=emulated_host_count,
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      grain_index_storage_option=getattr(config, "grain_index_storage_option", None),
      mixture_config_path=config.grain_train_mixture_config_path,
      elastic=config.grain_use_elastic_iterator,
      dataset_config=dataset_config,
      split="train",
  )
  kwargs = {
      "config": config,
      "data_columns": config.train_data_columns,
      "tokenize": config.tokenize_train_data,
      "grain_worker_count": config.grain_worker_count,
      "grain_per_worker_buffer_size": config.grain_per_worker_buffer_size,
  }
  if config.use_sft and config.use_multimodal:
    kwargs["image_column"] = config.train_image_column
  processed_ds = pipeline_fn(dataset=ds, **kwargs)
  return iter(processed_ds)


def _estimate_consumed_records_in_batch(batch: dict, modality: str) -> int:
  """Counts how many raw ArrayRecord examples were packed/batched into `batch`."""
  if modality == "text" and "inputs_segmentation" in batch:
    seg = np.asarray(batch["inputs_segmentation"])
    max_seg_per_row = np.max(seg, axis=-1)
    return int(np.sum(np.maximum(1, max_seg_per_row)))
  return int(batch["inputs"].shape[0])


def run_single_benchmark(
    modality: str,
    backend: str,
    dataset_stats: dict,
    gcs_server: gcs_qps_tracer.InstrumentedGCSServer,
    warmup_steps: int,
    steps: int,
    per_host_batch_size: int,
    max_target_length: int,
    grain_worker_count: int,
    grain_per_worker_buffer_size: int,
    grain_num_threads: int,
    grain_prefetch_buffer_size: int,
    emulated_host_count: int,
    concurrent_emulated_hosts: int,
    target_tpu_step_time_s: float,
    pace_to_tpu_step_time: bool = False,
    gcsfuse_read_ahead_kb: int = 1024,
) -> dict:
  """Runs one `(modality, backend)` benchmark configuration and returns exact QPS metrics."""
  gcs_server.reset_stats()

  if backend == "direct_gcs":
    train_files_pattern = f"gs://{gcs_server.bucket_name}/{modality}/train.array_record-*"
  elif backend == "gcsfuse":
    train_files_pattern = dataset_stats["glob_pattern"]
  else:
    raise ValueError(f"Unknown backend: {backend}")

  config = _build_maxtext_config(
      modality=modality,
      train_files_pattern=train_files_pattern,
      per_host_batch_size=per_host_batch_size,
      max_target_length=max_target_length,
      grain_worker_count=grain_worker_count,
      grain_per_worker_buffer_size=grain_per_worker_buffer_size,
      grain_num_threads=grain_num_threads,
      grain_prefetch_buffer_size=grain_prefetch_buffer_size,
      enable_data_shuffling=True,
  )

  posix_tracer = None
  if backend == "gcsfuse":
    posix_tracer = gcs_qps_tracer.PosixGCSFuseTracer(
        target_dir=dataset_stats["dir"],
        read_ahead_kb=gcsfuse_read_ahead_kb,
    )
    posix_tracer.start()

  io_before_init = gcs_qps_tracer.read_process_tree_io()
  gcs_server.set_phase("init", step_idx=-1)
  if posix_tracer is not None:
    posix_tracer.mark_phase("init")
  t_init_start = time.perf_counter()

  with ThreadPoolExecutor(max_workers=concurrent_emulated_hosts) as pool:
    # Phase 1: `init` (`ArrayRecordDataSource.__init__` probes `num_records()` across all shards)
    iterators = list(
        pool.map(
            lambda h_idx: _create_host_iterator(config, h_idx, emulated_host_count),
            range(concurrent_emulated_hosts),
        )
    )
    t_init_end = time.perf_counter()
    init_time_s = t_init_end - t_init_start

    # Phase 2: `first_batch` (Step 0: worker spawn + initial prefetch buffer fill)
    gcs_server.set_phase("first_batch", step_idx=0)
    if posix_tracer is not None:
      posix_tracer.mark_phase("first_batch")
    t_first_start = time.perf_counter()
    first_batches = list(pool.map(next, iterators))
    t_first_end = time.perf_counter()
    first_batch_time_s = t_first_end - t_first_start
    total_raw_records_consumed = sum(_estimate_consumed_records_in_batch(b, modality) for b in first_batches)

    # Phase 3: `warmup` (`warmup_steps` steps to separate initial buffer/shard warmup from steady state)
    gcs_server.set_phase("warmup", step_idx=0)
    if posix_tracer is not None:
      posix_tracer.mark_phase("warmup")
    t_warmup_start = time.perf_counter()
    warmup_raw_records = 0
    for w_idx in range(warmup_steps):
      gcs_server.set_phase("warmup", step_idx=w_idx)
      t0 = time.perf_counter()
      w_batches = list(pool.map(next, iterators))
      recs = sum(_estimate_consumed_records_in_batch(b, modality) for b in w_batches)
      warmup_raw_records += recs
      total_raw_records_consumed += recs
      if pace_to_tpu_step_time:
        elapsed = time.perf_counter() - t0
        if elapsed < target_tpu_step_time_s:
          time.sleep(target_tpu_step_time_s - elapsed)
    t_warmup_end = time.perf_counter()
    warmup_wall_s = t_warmup_end - t_warmup_start

    io_before_steady = gcs_qps_tracer.read_process_tree_io()

    # Phase 4: `steady` (`steps` measured steps)
    gcs_server.set_phase("steady", step_idx=0)
    if posix_tracer is not None:
      posix_tracer.mark_phase("steady")
    step_latencies = []
    steady_state_raw_records = 0

    t_steady_start = time.perf_counter()
    for step_idx in range(steps):
      gcs_server.set_phase("steady", step_idx=step_idx)
      t0 = time.perf_counter()
      batches = list(pool.map(next, iterators))
      t1 = time.perf_counter()
      step_latencies.append(t1 - t0)
      recs = sum(_estimate_consumed_records_in_batch(b, modality) for b in batches)
      total_raw_records_consumed += recs
      steady_state_raw_records += recs
      if pace_to_tpu_step_time:
        elapsed = time.perf_counter() - t0
        if elapsed < target_tpu_step_time_s:
          time.sleep(target_tpu_step_time_s - elapsed)
    t_steady_end = time.perf_counter()

  io_after_all = gcs_qps_tracer.read_process_tree_io()
  steady_state_wall_s = max(1e-9, t_steady_end - t_steady_start)
  total_wall_s = max(1e-9, t_steady_end - t_init_start)
  total_steps_executed = 1 + warmup_steps + steps
  steady_steps = max(1, steps)

  avg_record_bytes = dataset_stats["avg_record_bytes"]
  useful_payload_bytes_total = total_raw_records_consumed * avg_record_bytes
  useful_payload_bytes_steady = steady_state_raw_records * avg_record_bytes

  raw_records_per_step_per_host = (steady_state_raw_records / steady_steps) / concurrent_emulated_hosts
  posix_syscr_total = (io_after_all["syscr"] - io_before_init["syscr"]) / concurrent_emulated_hosts
  posix_syscr_steady = (io_after_all["syscr"] - io_before_steady["syscr"]) / concurrent_emulated_hosts

  if backend == "direct_gcs":
    summary = gcs_server.summarize()
    gcsfuse_cached_gets_per_step = None
    gcsfuse_cached_init_gets = None
  else:
    assert posix_tracer is not None
    summary = posix_tracer.stop_and_summarize()
    num_shards = dataset_stats["num_shards"]
    gcsfuse_cached_init_gets = float(
        num_shards * math.ceil((dataset_stats["total_bytes"] / num_shards) / (16 * 1024 * 1024))
    )
    gcsfuse_cached_gets_per_step = 0.0

  init_probe_open_gets_per_host = summary["init_probe_shard_open_gets"] / concurrent_emulated_hosts
  lazy_worker_open_gets_per_host = summary["lazy_worker_shard_open_gets"] / concurrent_emulated_hosts
  exact_shard_open_gets_per_host = summary["exact_shard_open_gets"] / concurrent_emulated_hosts
  exact_record_read_gets_per_host = summary["exact_record_read_gets"] / concurrent_emulated_hosts

  init_and_first_gets_per_host = summary["init_and_first_batch_gets"] / concurrent_emulated_hosts
  warmup_gets_per_host = summary["warmup_gets"] / concurrent_emulated_hosts
  steady_total_gets_per_host = summary["steady_total_gets"] / concurrent_emulated_hosts
  steady_record_read_gets_per_host = summary["steady_record_read_gets"] / concurrent_emulated_hosts
  steady_lazy_open_gets_per_host = summary["steady_lazy_shard_open_gets"] / concurrent_emulated_hosts
  total_gets_per_host = summary["total_media_get_requests"] / concurrent_emulated_hosts
  unique_shards_touched = summary["unique_shards_touched"]

  # Key distinction:
  # 1. `true_record_read_gets_per_step_per_host`: Pure data record GETs/step once shards are open.
  # 2. `total_gets_per_step_per_host`: Includes any remaining lazy shard opens during the step window.
  true_record_read_gets_per_step_per_host = steady_record_read_gets_per_host / steady_steps
  total_gets_per_step_per_host = steady_total_gets_per_host / steady_steps

  requested_bytes_per_host = summary["total_requested_bytes"] / concurrent_emulated_hosts
  transferred_bytes_per_host = summary["total_transferred_bytes"] / concurrent_emulated_hosts
  useful_per_host = useful_payload_bytes_total / concurrent_emulated_hosts
  read_amp_requested = requested_bytes_per_host / max(1.0, useful_per_host)
  read_amp_transferred = transferred_bytes_per_host / max(1.0, useful_per_host)

  # Cluster-wide projections for `emulated_host_count` hosts
  cluster_startup_burst_gets = init_and_first_gets_per_host * emulated_host_count
  cluster_shard_open_only_gets = exact_shard_open_gets_per_host * emulated_host_count
  cluster_init_probe_only_gets = init_probe_open_gets_per_host * emulated_host_count
  cluster_lazy_open_only_gets = lazy_worker_open_gets_per_host * emulated_host_count

  cluster_true_steady_qps = (true_record_read_gets_per_step_per_host * emulated_host_count) / max(
      1e-6, target_tpu_step_time_s
  )
  cluster_early_step_qps = (total_gets_per_step_per_host * emulated_host_count) / max(
      1e-6, target_tpu_step_time_s
  )
  mean_cpu_step_time_s = float(np.mean(step_latencies)) if step_latencies else first_batch_time_s

  candidate_step_times = [0.25, 0.40, 0.80] if modality == "text" else [0.40, 0.80, 1.50]
  step_time_sensitivity = {}
  for st in candidate_step_times:
    true_qps = (true_record_read_gets_per_step_per_host * emulated_host_count) / st
    early_qps = (total_gets_per_step_per_host * emulated_host_count) / st
    step_time_sensitivity[f"{st:.2f}s"] = {
        "cluster_true_steady_qps": round(true_qps, 1),
        "pct_of_50k_limit_true_steady": round(100.0 * true_qps / DEFAULT_GCS_BUCKET_QPS_LIMIT, 1),
        "cluster_early_step_qps_incl_lazy_opens": round(early_qps, 1),
        "pct_of_50k_limit_early_step": round(100.0 * early_qps / DEFAULT_GCS_BUCKET_QPS_LIMIT, 1),
    }

  return {
      "modality": modality,
      "backend": backend,
      "warmup_steps": warmup_steps,
      "steps": steps,
      "total_steps_executed": total_steps_executed,
      "per_host_batch_size": per_host_batch_size,
      "max_target_length": max_target_length,
      "grain_worker_count": grain_worker_count,
      "grain_num_threads": grain_num_threads,
      "emulated_host_count": emulated_host_count,
      "concurrent_emulated_hosts": concurrent_emulated_hosts,
      "target_tpu_step_time_s": target_tpu_step_time_s,
      "pace_to_tpu_step_time": pace_to_tpu_step_time,
      "timing": {
          "init_time_s": round(init_time_s, 4),
          "first_batch_time_s": round(first_batch_time_s, 4),
          "warmup_wall_s": round(warmup_wall_s, 4),
          "steady_state_wall_s": round(steady_state_wall_s, 4),
          "total_wall_s": round(total_wall_s, 4),
          "step_time_mean_s": round(mean_cpu_step_time_s, 4),
          "step_time_p50_s": round(float(np.percentile(step_latencies, 50)), 4) if step_latencies else 0.0,
          "step_time_p90_s": round(float(np.percentile(step_latencies, 90)), 4) if step_latencies else 0.0,
          "step_time_p99_s": round(float(np.percentile(step_latencies, 99)), 4) if step_latencies else 0.0,
          "host_batches_per_sec": round(steady_steps / steady_state_wall_s, 2),
          "host_examples_per_sec": round((steady_steps * per_host_batch_size) / steady_state_wall_s, 2),
          "host_payload_mb_per_sec": round(
              (useful_payload_bytes_steady / concurrent_emulated_hosts) / (1024**2) / steady_state_wall_s, 2
          ),
      },
      "per_host_io": {
          "unique_shards_touched": unique_shards_touched,
          "raw_records_per_step_per_host": round(raw_records_per_step_per_host, 2),
          "init_probe_shard_open_gets_per_host": round(init_probe_open_gets_per_host, 1),
          "lazy_worker_shard_open_gets_per_host": round(lazy_worker_open_gets_per_host, 1),
          "exact_shard_open_gets_per_host": round(exact_shard_open_gets_per_host, 1),
          "exact_record_read_gets_per_host": round(exact_record_read_gets_per_host, 1),
          "init_and_first_step_gets": round(init_and_first_gets_per_host, 1),
          "warmup_gets": round(warmup_gets_per_host, 1),
          "steady_state_total_gets": round(steady_total_gets_per_host, 1),
          "steady_state_record_read_gets": round(steady_record_read_gets_per_host, 1),
          "steady_state_lazy_shard_open_gets": round(steady_lazy_open_gets_per_host, 1),
          "total_gets": round(total_gets_per_host, 1),
          "true_record_read_gets_per_step_per_host": round(true_record_read_gets_per_step_per_host, 2),
          "total_gets_per_step_per_host": round(total_gets_per_step_per_host, 2),
          "posix_read_syscalls_total": round(posix_syscr_total, 1),
          "posix_read_syscalls_per_step": round(posix_syscr_steady / steady_steps, 2),
          "useful_payload_mib": round(useful_per_host / (1024**2), 2),
          "gcs_requested_mib": round(requested_bytes_per_host / (1024**2), 2),
          "gcs_transferred_mib": round(transferred_bytes_per_host / (1024**2), 2),
          "read_amplification_requested_x": round(read_amp_requested, 2),
          "read_amplification_transferred_x": round(read_amp_transferred, 2),
          "range_type_breakdown": summary["range_type_breakdown"],
          "request_category_breakdown": summary["request_category_breakdown"],
      },
      "large_scale_cluster_projection": {
          "emulated_host_count": emulated_host_count,
          "global_batch_size": per_host_batch_size * emulated_host_count,
          "cluster_init_probe_shard_open_gets": int(round(cluster_init_probe_only_gets)),
          "cluster_lazy_worker_shard_open_gets": int(round(cluster_lazy_open_only_gets)),
          "cluster_total_shard_open_gets": int(round(cluster_shard_open_only_gets)),
          "cluster_startup_burst_gets": int(round(cluster_startup_burst_gets)),
          "cluster_startup_qps_over_5s_window": round(cluster_startup_burst_gets / 5.0, 1),
          "cluster_true_record_read_gets_per_step": int(
              round(true_record_read_gets_per_step_per_host * emulated_host_count)
          ),
          "cluster_total_gets_per_step_incl_lazy_opens": int(
              round(total_gets_per_step_per_host * emulated_host_count)
          ),
          "cluster_true_steady_qps_at_tpu_step_time": round(cluster_true_steady_qps, 1),
          "cluster_early_step_qps_at_tpu_step_time": round(cluster_early_step_qps, 1),
          "pct_of_50k_limit_true_steady": round(
              100.0 * cluster_true_steady_qps / DEFAULT_GCS_BUCKET_QPS_LIMIT, 1
          ),
          "pct_of_50k_limit_early_step": round(
              100.0 * cluster_early_step_qps / DEFAULT_GCS_BUCKET_QPS_LIMIT, 1
          ),
          "step_time_sensitivity": step_time_sensitivity,
          "gcsfuse_file_cache_warm_steady_qps": 0.0 if gcsfuse_cached_gets_per_step == 0.0 else None,
          "gcsfuse_file_cache_cold_fill_total_gets": (
              int(round(gcsfuse_cached_init_gets * emulated_host_count))
              if gcsfuse_cached_init_gets is not None
              else None
          ),
      },
  }


def _print_comparison_report(results: list[dict], dataset_meta: dict[str, dict]) -> None:
  """Prints a structured report separating Init Probe, Lazy Shard Opens, and True Record-Read QPS."""
  print("\n" + "=" * 148)
  print("MAXTEXT PYGRAIN + ARRAYRECORD INPUT PIPELINE QPS BENCHMARK REPORT (EXACT BYTE-OFFSET CLASSIFICATION)")
  print("=" * 148)
  for mod, meta in dataset_meta.items():
    print(
        f"Dataset [{mod.upper()}]: {meta['num_shards']} shards, {meta['total_records']} records, "
        f"avg record = {meta['avg_record_bytes'] / 1024:.1f} KiB, "
        f"total size = {meta['total_bytes'] / (1024**2):.2f} MiB ({meta['writer_options']})"
    )
  print("-" * 148)
  header = (
      f"{'Modality':<10} | {'Backend':<12} | {'Recs/Stp':>8} | {'InitProbe':>9} | {'LazyOpen':>8} | "
      f"{'RecGET/Stp':>10} | {'TotGET/Stp':>10} | {'ReadAmp(Xfer)':>13} | {'Startup Burst':>13} | "
      f"{'True Steady QPS':>15} | {'Early Step QPS':>14}"
  )
  print(header)
  print("-" * 148)
  for r in results:
    io = r["per_host_io"]
    cl = r["large_scale_cluster_projection"]
    print(
        f"{r['modality']:<10} | {r['backend']:<12} | "
        f"{io['raw_records_per_step_per_host']:>8.2f} | "
        f"{io['init_probe_shard_open_gets_per_host']:>9.1f} | "
        f"{io['lazy_worker_shard_open_gets_per_host']:>8.1f} | "
        f"{io['true_record_read_gets_per_step_per_host']:>10.2f} | "
        f"{io['total_gets_per_step_per_host']:>10.2f} | "
        f"{io['read_amplification_transferred_x']:>12.1f}x | "
        f"{cl['cluster_startup_burst_gets']:>13,d} | "
        f"{cl['cluster_true_steady_qps_at_tpu_step_time']:>15,.1f} | "
        f"{cl['cluster_early_step_qps_at_tpu_step_time']:>14,.1f}"
    )
    if r["backend"] == "gcsfuse":
      warm_open_per_host = cl["gcsfuse_file_cache_cold_fill_total_gets"] / r["emulated_host_count"]
      print(
          f"{r['modality']:<10} | {'gcsfuse_warm':<12} | "
          f"{io['raw_records_per_step_per_host']:>8.2f} | "
          f"{warm_open_per_host:>9.1f} | "
          f"{0.0:>8.1f} | "
          f"{0.0:>10.2f} | "
          f"{0.0:>10.2f} | "
          f"{1.0:>12.1f}x | "
          f"{cl['gcsfuse_file_cache_cold_fill_total_gets']:>13,d} | "
          f"{0.0:>15,.1f} | "
          f"{0.0:>14,.1f}"
      )
  print("-" * 148)
  print("STEP-TIME SENSITIVITY PROJECTION (True Steady-State Record-Read QPS across TPU Step Times):")
  for r in results:
    cl = r["large_scale_cluster_projection"]
    sens_parts = [
        f"t_step={k}: {v['cluster_true_steady_qps']:,.1f} QPS ({v['pct_of_50k_limit_true_steady']:.1f}% of 50k)"
        for k, v in cl["step_time_sensitivity"].items()
    ]
    print(f"  - {r['modality']:<10} [{r['backend']:<10}]: " + "  |  ".join(sens_parts))
  print("=" * 148 + "\n")


def main():
  parser = argparse.ArgumentParser(
      description="Benchmark MaxText PyGrain + ArrayRecord GCS QPS (Text & Multimodal, Direct GCS vs GCSFuse)."
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="",
      help="Directory for synthetic ArrayRecord datasets (created in a temp dir if omitted).",
  )
  parser.add_argument(
      "--modalities",
      type=str,
      default="text,multimodal",
      help="Comma-separated modalities to benchmark: text,multimodal",
  )
  parser.add_argument(
      "--backends",
      type=str,
      default="direct_gcs,gcsfuse",
      help="Comma-separated storage backends to compare: direct_gcs,gcsfuse",
  )
  parser.add_argument("--num_shards", type=int, default=64, help="Number of .array_record shards per dataset.")
  parser.add_argument("--text_records_per_shard", type=int, default=256, help="Records per shard for text dataset.")
  parser.add_argument(
      "--multimodal_records_per_shard", type=int, default=64, help="Records per shard for multimodal dataset."
  )
  parser.add_argument(
      "--warmup_steps",
      type=int,
      default=5,
      help="Number of warmup steps after first_batch before measuring steady-state steps.",
  )
  parser.add_argument("--steps", type=int, default=10, help="Number of measured steady-state steps per benchmark.")
  parser.add_argument("--per_host_batch_size", type=int, default=4, help="Batch size loaded per host per step.")
  parser.add_argument("--max_target_length", type=int, default=2048, help="Sequence length (`max_target_length`).")
  parser.add_argument(
      "--grain_worker_count",
      type=int,
      default=0,
      help="PyGrain mp_prefetch worker count (0 = in-process; >0 = multiprocess workers).",
  )
  parser.add_argument("--grain_per_worker_buffer_size", type=int, default=2)
  parser.add_argument("--grain_num_threads", type=int, default=4, help="PyGrain ReadOptions.num_threads.")
  parser.add_argument(
      "--grain_prefetch_buffer_size", type=int, default=16, help="PyGrain ReadOptions.prefetch_buffer_size."
  )
  parser.add_argument(
      "--emulated_host_count",
      type=int,
      default=2048,
      help="Number of TPU dataloading hosts to emulate in cluster sharding (`dataloading_host_count`).",
  )
  parser.add_argument(
      "--concurrent_emulated_hosts",
      type=int,
      default=4,
      help="Number of emulated host iterators to run concurrently in lockstep on this VM.",
  )
  parser.add_argument(
      "--text_target_tpu_step_time_s",
      type=float,
      default=0.40,
      help="Target TPU step duration (seconds) for projecting cluster-wide QPS in text pre-training.",
  )
  parser.add_argument(
      "--multimodal_target_tpu_step_time_s",
      type=float,
      default=0.80,
      help="Target TPU step duration (seconds) for projecting cluster-wide QPS in multimodal training.",
  )
  parser.add_argument(
      "--pace_to_tpu_step_time",
      action="store_true",
      help="If set, paces `next(iterator)` calls to `target_tpu_step_time_s`.",
  )
  parser.add_argument(
      "--output_json",
      type=str,
      default="",
      help="Optional path to save detailed JSON benchmark results.",
  )
  args = parser.parse_args()

  cleanup_temp = False
  if not args.data_dir:
    args.data_dir = tempfile.mkdtemp(prefix="maxtext_qps_bench_")
    cleanup_temp = True

  modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
  backends = [b.strip() for b in args.backends.split(",") if b.strip()]

  dataset_meta = {}
  if "text" in modalities:
    dataset_meta["text"] = generate_synthetic_arrayrecords.generate_text_dataset(
        args.data_dir,
        num_shards=args.num_shards,
        records_per_shard=args.text_records_per_shard,
    )
  if "multimodal" in modalities:
    dataset_meta["multimodal"] = generate_synthetic_arrayrecords.generate_multimodal_dataset(
        args.data_dir,
        num_shards=args.num_shards,
        records_per_shard=args.multimodal_records_per_shard,
    )

  gcs_server = gcs_qps_tracer.InstrumentedGCSServer(root_dir=args.data_dir, bucket_name="qps-bench")
  endpoint = gcs_server.start()
  print(f"Started Instrumented GCS Endpoint at {endpoint} serving {args.data_dir}")

  all_results = []
  try:
    for modality in modalities:
      tpu_step_s = (
          args.text_target_tpu_step_time_s if modality == "text" else args.multimodal_target_tpu_step_time_s
      )
      for backend in backends:
        print(
            f"\n>>> Running benchmark: modality={modality}, backend={backend}, "
            f"emulated_hosts={args.emulated_host_count} (concurrent={args.concurrent_emulated_hosts}), "
            f"per_host_batch_size={args.per_host_batch_size}, warmup_steps={args.warmup_steps}, steps={args.steps} ..."
        )
        res = run_single_benchmark(
            modality=modality,
            backend=backend,
            dataset_stats=dataset_meta[modality],
            gcs_server=gcs_server,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            per_host_batch_size=args.per_host_batch_size,
            max_target_length=args.max_target_length,
            grain_worker_count=args.grain_worker_count,
            grain_per_worker_buffer_size=args.grain_per_worker_buffer_size,
            grain_num_threads=args.grain_num_threads,
            grain_prefetch_buffer_size=args.grain_prefetch_buffer_size,
            emulated_host_count=args.emulated_host_count,
            concurrent_emulated_hosts=args.concurrent_emulated_hosts,
            target_tpu_step_time_s=tpu_step_s,
            pace_to_tpu_step_time=args.pace_to_tpu_step_time,
        )
        all_results.append(res)
  finally:
    gcs_server.stop()

  _print_comparison_report(all_results, dataset_meta)

  if args.output_json:
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
      json.dump({"datasets": dataset_meta, "benchmarks": all_results}, f, indent=2)
    print(f"Saved detailed JSON report to {args.output_json}")

  if cleanup_temp:
    import shutil  # pylint: disable=import-outside-toplevel

    shutil.rmtree(args.data_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
