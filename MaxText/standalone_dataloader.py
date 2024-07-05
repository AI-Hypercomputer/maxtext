"""
Copyright 2023 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
""" Standalone data loader - only loads data for each training step, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more
import jax
import os

import max_logging

from typing import Sequence
import datetime
from absl import app

import time
import random

import pyconfig
from train import validate_train_config, setup_train_loop, setup_mesh_and_model

from torch_datasets.parquet import ParquetDataset
from torch.utils.data import DataLoader


def split_list(lst, n):
  """Splits a list into roughly equal sized sublists and pads.

  Args:
    lst: The list to split.
    n: The desired number of sublists.

  Returns:
    A list of sublists.
  """
  # Calculate the size of each sublist.
  size = len(lst) // n

  # Create the sublists.
  sublists = [lst[i * size : (i + 1) * size] for i in range(n)]

  last_idx = n * size

  if last_idx >= len(lst):
    return sublists

  remainder = len(lst) - last_idx

  for i in range(remainder):
    sublists[i].append(lst[last_idx + i])

  # Padding to make sure all nodes are loading the same amount of
  # files. Needed to make sure no deadlocking when the workload
  # is distributed unevenly.
  max_length = max([len(each) for each in sublists])
  for each in sublists:
    while len(each) < max_length:
      each.append(random.choice(lst))

  return sublists


def parquet_data_loader(config):
  batch_size = config.local_batch_size

  # [Short-term]: we know what the mount path is, and we know what the dataset is.
  # Therefore, we save the listing process which will create a listing storm.
  # TODO: create an option to create a list storm if opted in.
  parquet_files = [f"/mnt/gcsfuse/{i:04d}.parquet" for i in range(10000)]
  if config.dataset_bucket == "xai-hf-dataset-parquet-10g":
    parquet_files = [f"/mnt/gcsfuse/{i:05d}.parquet" for i in range(12000)]

  worker_id = jax.process_index()

  sublists = split_list(parquet_files, jax.process_count())

  dataset = ParquetDataset(
      allocated_parquet_files=sublists[worker_id],
      batch_size=batch_size,
      columns=["outputs", "image_base64_str"],
  )
  data_loader = DataLoader(
      dataset=dataset,
      num_workers=config.data_loader_num_workers,
      batch_size=batch_size,
      prefetch_factor=config.prefetch_factor,
  )
  return data_loader


def data_load_loop(config):
  """Main data loader loop.
  Loads batches of data for each training step.
  """
  # We seem to need the mesh to be setup for the distributed barrier to work.
  # Therefore, simply calling this function but using our own iterator.
  setup_mesh_and_model(config)
  data_loader = parquet_data_loader(config)

  # Record per-step per-epoch data loading times.
  data_loading_time = [[] for _ in range(config.epochs)]

  # Record per-step times, which includes data loading, simulated computation time, and barrier wait time.
  step_time = [[] for _ in range(config.epochs)]

  # Record per-epoch total time.
  epoch_times = []

  training_start = datetime.datetime.now()
  max_logging.log(
      f"STANDALONE DATALOADER : Started training loop on host {jax.process_index()}"
  )
  for i in range(config.epochs):
    epoch_start = datetime.datetime.now()
    step_count = 0
    step_data_loading_start = datetime.datetime.now()
    step_start = datetime.datetime.now()
    for _ in data_loader:
      step_data_loading_end = datetime.datetime.now()
      max_logging.log(
          f"STANDALONE DATALOADER : Host {jax.process_index()} got a batch in {step_data_loading_end - step_data_loading_start} seconds on epoch {i} step {step_count}"
      )

      data_loading_time[i].append(
          (step_data_loading_end - step_data_loading_start).total_seconds()
      )

      # Simulate Accelerator Computation Time.
      if config.per_step_computation_time:
        time.sleep(config.per_step_computation_time)

      barrier_start = datetime.datetime.now()
      max_logging.log(
          f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} started on step {step_count} at {barrier_start}"
      )

      jax.experimental.multihost_utils.sync_global_devices(
          "Barrier before proceeding to the next step"
      )

      barrier_end = datetime.datetime.now()
      max_logging.log(
          f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} completed on step {step_count} at {barrier_end}, lasted {barrier_end - barrier_start} seconds"
      )

      step_count += 1

      # Count the per-step time.
      step_end = datetime.datetime.now()
      step_time[i].append((step_end - step_start).total_seconds())
      step_start = step_end
      # Reset the start time of computing data loading.
      step_data_loading_start = datetime.datetime.now()

    epoch_end = datetime.datetime.now()
    max_logging.log(
        f"STANDALONE DATALOADER : Host {jax.process_index()} completed epoch {i} using {epoch_end - epoch_start} seconds"
    )
    epoch_times.append((epoch_end - epoch_start).total_seconds())

  training_end = datetime.datetime.now()
  time_to_load_first_batch = training_end - training_start

  max_logging.log(
      f"STANDALONE DATALOADER Metrics: Training completed in {time_to_load_first_batch} seconds, on host {jax.process_index()}"
  )
  for i in range(config.epochs):
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-epoch total times for epoch {i} on host {jax.process_index()}: {epoch_times[i]}."
    )
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-step data loading times for epoch {i} on host {jax.process_index()}: {data_loading_time[i]}."
    )
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-step times for epoch {i} on host {jax.process_index()}: {step_time[i]}.",
    )


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_cpu_enable_gloo_collectives", True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config
  # validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  max_logging.log(f"Found {jax.process_count()} processes.")
  max_logging.log(f"Found {jax.devices()} devices.")
  if config.dataset_type == "tfds":
    os.environ["TFDS_DATA_DIR"] = config.dataset_path
  data_load_loop(config)


if __name__ == "__main__":
  app.run(main)
