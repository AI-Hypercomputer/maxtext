# Copyright 2023-2026 Google LLC
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

"""Standalone benchmark comparing MaxText dataset initialization with and without FileInstruction optimization.

This script executes the end-to-end dataset iterator initialization using Colocated Python
over Pathways, measuring the time taken for:
  1. With FileInstruction optimization (Coordinator pre-extracts FileInstruction manifests
     and serializes them to worker sidecars, bypassing per-shard GCS index inspection).
  2. Without FileInstruction optimization (Baseline: Coordinator sends raw file patterns/paths,
     forcing every worker sidecar to read array_record headers/indices directly from GCS).

Usage:
  python3 -m maxtext.benchmarks.benchmark_file_instruction src/maxtext/configs/base.yml \\
      model_name=llama3.1-8b \\
      num_benchmark_runs=10
"""

import functools
import os
import statistics
import sys
import time
from typing import Sequence

from absl import app
import pathwaysutils
import jax
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import multihost_dataloading
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils


def init_with_file_instructions(config, mesh, process_indices) -> tuple[float, float, float]:
  """Runs end-to-end dataset iterator initialization with FileInstruction optimization.

  Returns:
    (total_time_sec, coordinator_extraction_time_sec, remote_init_and_fetch_time_sec)
  """
  t0 = time.perf_counter()

  # Step 1: Coordinator manifest extraction
  t_extract_start = time.perf_counter()
  file_instructions = grain_data_processing.extract_file_instructions(
      config.grain_train_files, config.grain_data_source_max_workers
  )
  extract_time = time.perf_counter() - t_extract_start

  # Step 2: Build get_ds_fn with FileInstruction objects
  get_ds_fn = functools.partial(
      grain_data_processing.get_datasets,
      file_instructions,
      config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=config.num_epoch,
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      mixture_config_path=config.grain_train_mixture_config_path,
      elastic=config.grain_use_elastic_iterator,
  )

  pipeline_fn = grain_data_processing._get_pipeline_fn(config)
  preprocessing_fn = functools.partial(
      pipeline_fn,
      config=config,
      data_columns=config.train_data_columns,
      tokenize=config.tokenize_train_data,
      grain_worker_count=config.grain_worker_count,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )

  if config.grain_use_elastic_iterator:
    preprocessing_fn = functools.partial(
        grain_data_processing._make_elastic_iterator, config=config, preprocessing_fn=preprocessing_fn
    )

  global_shape = (config.global_batch_size_to_load, config.max_target_length)

  # Step 3: Create RemoteIteratorWrapper (triggers colocated python init on worker sidecars)
  t_remote_start = time.perf_counter()
  iterator = multihost_dataloading.RemoteIteratorWrapper(
      get_ds_fn,
      preprocessing_fn,
      mesh,
      global_shape,
      checkpoint_path=config.checkpoint_dir,
      elastic=config.grain_use_elastic_iterator,
  )

  # Step 4: Fetch first batch to ensure pipeline is fully started end-to-end
  _ = next(iterator)
  remote_time = time.perf_counter() - t_remote_start

  total_time = time.perf_counter() - t0
  return total_time, extract_time, remote_time


def init_without_file_instructions(config, mesh, process_indices) -> tuple[float, float, float]:
  """Runs end-to-end dataset iterator initialization WITHOUT FileInstruction optimization (Baseline).

  Returns:
    (total_time_sec, 0.0, remote_init_and_fetch_time_sec)
  """
  t0 = time.perf_counter()

  # Step 1: Raw file pattern (no coordinator extraction)
  train_files = config.grain_train_files

  # Step 2: Build get_ds_fn with raw pattern
  get_ds_fn = functools.partial(
      grain_data_processing.get_datasets,
      train_files,
      config.grain_file_type,
      shuffle=config.enable_data_shuffling,
      shuffle_seed=config.data_shuffle_seed,
      shuffle_buffer_size=config.grain_shuffle_buffer_size,
      num_epoch=config.num_epoch,
      grain_worker_count=config.grain_worker_count,
      grain_num_threads=config.grain_num_threads,
      grain_prefetch_buffer_size=config.grain_prefetch_buffer_size,
      grain_data_source_max_workers=config.grain_data_source_max_workers,
      mixture_config_path=config.grain_train_mixture_config_path,
      elastic=config.grain_use_elastic_iterator,
  )

  pipeline_fn = grain_data_processing._get_pipeline_fn(config)
  preprocessing_fn = functools.partial(
      pipeline_fn,
      config=config,
      data_columns=config.train_data_columns,
      tokenize=config.tokenize_train_data,
      grain_worker_count=config.grain_worker_count,
      grain_per_worker_buffer_size=config.grain_per_worker_buffer_size,
  )

  if config.grain_use_elastic_iterator:
    preprocessing_fn = functools.partial(
        grain_data_processing._make_elastic_iterator, config=config, preprocessing_fn=preprocessing_fn
    )

  global_shape = (config.global_batch_size_to_load, config.max_target_length)

  # Step 3: Create RemoteIteratorWrapper (triggers colocated python init on worker sidecars)
  t_remote_start = time.perf_counter()
  iterator = multihost_dataloading.RemoteIteratorWrapper(
      get_ds_fn,
      preprocessing_fn,
      mesh,
      global_shape,
      checkpoint_path=config.checkpoint_dir,
      elastic=config.grain_use_elastic_iterator,
  )

  # Step 4: Fetch first batch to ensure pipeline is fully started end-to-end
  _ = next(iterator)
  remote_time = time.perf_counter() - t_remote_start

  total_time = time.perf_counter() - t0
  return total_time, 0.0, remote_time


def print_report(
    times_with: list[float],
    times_without: list[float],
    extract_times_with: list[float],
    remote_times_with: list[float],
    remote_times_without: list[float],
):
  """Prints a clean, structured benchmark report with summary statistics."""
  mean_with = statistics.mean(times_with)
  std_with = statistics.stdev(times_with) if len(times_with) > 1 else 0.0
  min_with = min(times_with)
  max_with = max(times_with)

  mean_without = statistics.mean(times_without)
  std_without = statistics.stdev(times_without) if len(times_without) > 1 else 0.0
  min_without = min(times_without)
  max_without = max(times_without)

  speedup = mean_without / mean_with if mean_with > 0 else 0.0
  reduction_pct = ((mean_without - mean_with) / mean_without * 100) if mean_without > 0 else 0.0

  mean_extract_with = statistics.mean(extract_times_with)
  mean_remote_with = statistics.mean(remote_times_with)
  mean_remote_without = statistics.mean(remote_times_without)

  report = []
  report.append("=" * 80)
  report.append("MAXTEXT DATASET ITERATOR INITIALIZATION BENCHMARK REPORT")
  report.append("=" * 80)
  report.append(f"Number of Trials: {len(times_with)}")
  report.append("")
  report.append(f"{'Trial':<8} | {'With FileInstruction (s)':<26} | {'Without FileInstruction (s)':<28}")
  report.append("-" * 70)
  for i in range(len(times_with)):
    report.append(f"#{i+1:<7} | {times_with[i]:<26.4f} | {times_without[i]:<28.4f}")
  report.append("-" * 70)
  report.append("")
  report.append("SUMMARY METRICS (Mean +/- Std):")
  report.append(f"  • With FileInstruction (Optimized):   {mean_with:.4f} s +/- {std_with:.4f} s  [min: {min_with:.4f}s, max: {max_with:.4f}s]")
  report.append(f"      - Coordinator manifest extract:   {mean_extract_with:.4f} s")
  report.append(f"      - Remote sidecar init + 1st batch: {mean_remote_with:.4f} s")
  report.append("")
  report.append(f"  • Without FileInstruction (Baseline): {mean_without:.4f} s +/- {std_without:.4f} s  [min: {min_without:.4f}s, max: {max_without:.4f}s]")
  report.append(f"      - Coordinator manifest extract:   0.0000 s")
  report.append(f"      - Remote sidecar init + 1st batch: {mean_remote_without:.4f} s")
  report.append("")
  report.append("KEY TAKEAWAYS:")
  report.append(f"  • End-to-End Speedup:       {speedup:.2f}x faster")
  report.append(f"  • Total Time Reduction:     {reduction_pct:.1f}%")
  report.append(f"  • Worker Sidecar Speedup:   {mean_remote_without / mean_remote_with:.2f}x faster in sidecars")
  report.append("=" * 80)

  print("\n".join(report))


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()

  # Extract custom benchmark args before pyconfig validation
  num_runs = int(os.environ.get("NUM_BENCHMARK_RUNS", "10"))
  filtered_argv = []
  for arg in argv:
    if arg.startswith("num_benchmark_runs="):
      num_runs = int(arg.split("=", 1)[1])
    elif arg.startswith("--num_benchmark_runs="):
      num_runs = int(arg.split("=", 1)[1])
    else:
      filtered_argv.append(arg)

  config = pyconfig.initialize(filtered_argv)
  mesh = maxtext_utils.get_mesh_from_config(config)
  process_indices = tuple(range(jax.process_count()))

  max_logging.log(f"Starting MaxText FileInstruction benchmark ({num_runs} iterations each)...")
  max_logging.log(f"Dataset files: {config.grain_train_files}")

  # Warmup
  max_logging.log("Running warmup for both implementations...")
  try:
    _ = init_with_file_instructions(config, mesh, process_indices)
    _ = init_without_file_instructions(config, mesh, process_indices)
    max_logging.log("Warmup complete.")
  except Exception as e:
    max_logging.log(f"Warmup warning/error: {e}")

  times_with = []
  extract_times_with = []
  remote_times_with = []

  times_without = []
  extract_times_without = []
  remote_times_without = []

  max_logging.log(f"\n--- Running {num_runs} trials WITH FileInstruction Optimization ---")
  for i in range(num_runs):
    total, extract, remote = init_with_file_instructions(config, mesh, process_indices)
    times_with.append(total)
    extract_times_with.append(extract)
    remote_times_with.append(remote)
    max_logging.log(f"  [With FileInstruction #{i+1}/{num_runs}] Total: {total:.4f}s (Extract: {extract:.4f}s, Remote+Batch: {remote:.4f}s)")

  max_logging.log(f"\n--- Running {num_runs} trials WITHOUT FileInstruction Optimization (Baseline) ---")
  for i in range(num_runs):
    total, extract, remote = init_without_file_instructions(config, mesh, process_indices)
    times_without.append(total)
    extract_times_without.append(extract)
    remote_times_without.append(remote)
    max_logging.log(f"  [Without FileInstruction #{i+1}/{num_runs}] Total: {total:.4f}s (Remote+Batch: {remote:.4f}s)")

  print_report(
      times_with=times_with,
      times_without=times_without,
      extract_times_with=extract_times_with,
      remote_times_with=remote_times_with,
      remote_times_without=remote_times_without,
  )


if __name__ == "__main__":
  app.run(main)
