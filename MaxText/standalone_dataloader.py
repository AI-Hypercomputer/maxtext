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

from typing import Sequence, Type
from types import MappingProxyType
import datetime
from absl import app

import time
import random

import pyconfig
from train import validate_train_config, setup_train_loop, setup_mesh_and_model
import storage_utils

from torch_datasets.parquet import FileParallelRandomRead, FileParallelSequentialRead
from torch.utils.data import DataLoader, IterableDataset

TOTAL_TRAINING_TIME_DIRECTORY = "total_training_time"
PER_STEP_DATA_LOADING_TIME_DIRECTORY = "per_step_data_loading_time"
PER_STEP_TIME_DIRECTORY = "per_step_time"
PER_EPOCH_TIME_DIRECTORY = "per_epoch_time"

DATA_LOADER_STRATEGIES_BY_NAME = MappingProxyType({
    'FileParallelSequentialRead': FileParallelSequentialRead,
    'FileParallelRandomRead': FileParallelRandomRead, 
})
STEP_BARRIER_MSG = "Synchronize all processes within a step"

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


def data_loader_strategy_type(config) -> Type[IterableDataset]:
  name = config.data_loader_strategy_name
  if name in DATA_LOADER_STRATEGIES_BY_NAME:
    return DATA_LOADER_STRATEGIES_BY_NAME[name]
  raise ValueError(f'data_loader_strategy of \'{name}\' is not one of the following '
                   f'supported strategies: {DATA_LOADER_STRATEGIES_BY_NAME.keys()}')

def list_files_walk(start_path='.'):
    dataset_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            dataset_files.append(os.path.join(root, file))
    return sorted(dataset_files)

def parquet_data_loader(config):
  batch_size = config.local_batch_size

  # On each node, we "walk" the directory to get the list of files within the dataset.
  parquet_files = list_files_walk(config.dataset_directory)

  worker_id = jax.process_index()

  sublists = split_list(parquet_files, jax.process_count())

  strategy_type = data_loader_strategy_type(config)

  dataset = strategy_type(
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

def step_barrier_wait(msg, step_count):
  barrier_start = datetime.datetime.now()
  max_logging.log(
      f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} started on step {step_count} at {barrier_start}"
  )

  jax.experimental.multihost_utils.sync_global_devices(
      msg
  )

  barrier_end = datetime.datetime.now()
  max_logging.log(
      f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} completed on step {step_count} at {barrier_end}, lasted {(barrier_end - barrier_start).total_seconds()} seconds"
  )
  
def measure_epoch_time(epoch, epoch_start, epoch_times):
  epoch_end = datetime.datetime.now()
  max_logging.log(
      f"STANDALONE DATALOADER : Host {jax.process_index()} completed epoch {epoch} using {(epoch_end - epoch_start).total_seconds()} seconds"
  )
  epoch_times.append([jax.process_index(), epoch, (epoch_end - epoch_start).total_seconds()])
  

def data_load_loop(config):
  """Main data loader loop.
  Loads batches of data for each training step.
  """
  # We seem to need the mesh to be setup for the distributed barrier to work.
  # Therefore, simply calling this function but using our own iterator.
  setup_mesh_and_model(config)
  data_loader = parquet_data_loader(config)
  
  max_steps = config.max_steps

  # Record per-step per-epoch data loading times.
  per_step_data_loading_time = []

  # Record per-step times, which includes data loading, simulated computation time, and barrier wait time.
  step_time = []

  # Record per-epoch total time.
  epoch_times = []

  jax.experimental.multihost_utils.sync_global_devices("Barrier before training steps start")
  training_start = datetime.datetime.now()
  max_logging.log(
      f"STANDALONE DATALOADER : Started training loop on host {jax.process_index()}"
  )
  global_steps = 0
  for i in range(config.epochs):
    epoch_start = datetime.datetime.now()
    local_steps = 0
    step_data_loading_start = datetime.datetime.now()
    step_start = datetime.datetime.now()
    for _ in data_loader:
      step_data_loading_end = datetime.datetime.now()
      data_loading_interval = (step_data_loading_end - step_data_loading_start).total_seconds()
      max_logging.log(
          f"STANDALONE DATALOADER : Host {jax.process_index()} got a batch in {data_loading_interval} seconds on epoch {i} step {local_steps}"
      )
      per_step_data_loading_time.append(
          [jax.process_index(), i, local_steps, data_loading_interval]
      )
      
      if jax.process_index() == 0:
        if data_loading_interval < config.per_step_interval:
          time.sleep(config.per_step_interval - data_loading_interval)
        step_barrier_wait(STEP_BARRIER_MSG, local_steps)
      else:
        step_barrier_wait(STEP_BARRIER_MSG, local_steps)

      # Measure the per-step time.
      step_end = datetime.datetime.now()
      step_time.append([jax.process_index(), i, local_steps, (step_end - step_start).total_seconds()])
      step_start = step_end
      # Reset the start time of computing data loading.
      step_data_loading_start = datetime.datetime.now()
      
      local_steps += 1
      global_steps += 1
      if max_steps > 0 and global_steps >= max_steps:
        max_logging.log(
            f"STANDALONE DATALOADER : {global_steps} global steps reached. Stopped training."
        )
        break
    else:
      # Only executed if the inner loop did NOT break.
      measure_epoch_time(epoch=i, epoch_start=epoch_start, epoch_times=epoch_times)
      continue
    # Although the training has reached the global steps, we'd still want to
    # log the current epoch time.
    measure_epoch_time(epoch=i, epoch_start=epoch_start, epoch_times=epoch_times)
    break

  training_end = datetime.datetime.now()
  training_time = (training_end - training_start).total_seconds()

  max_logging.log(
      f"STANDALONE DATALOADER Metrics: Training completed in {training_time} seconds, on host {jax.process_index()}"
  )
  max_logging.log(
      f"STANDALONE DATALOADER Metrics: Per-epoch total times on host {jax.process_index()}: {epoch_times}."
  )
  max_logging.log(
      f"STANDALONE DATALOADER Metrics: Per-step data loading times on host {jax.process_index()}: {per_step_data_loading_time}."
  )
  max_logging.log(
      f"STANDALONE DATALOADER Metrics: Per-step times on host {jax.process_index()}: {step_time}.",
  )
  
  if config.gcs_metrics_bucket:
    max_logging.log(f"Uploading metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    base_name = f"{jax.process_index()}.csv"
    # Upload total training time.
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, TOTAL_TRAINING_TIME_DIRECTORY, base_name), 
        [[jax.process_index(), training_time]]
    )
    
    # Upload per-step data loading time.
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, PER_STEP_DATA_LOADING_TIME_DIRECTORY, base_name), 
        per_step_data_loading_time
    )
    
    # Upload per-step total time.
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, PER_STEP_TIME_DIRECTORY, base_name), 
        step_time
    )
    
    # Upload per epoch time.
    storage_utils.upload_csv(
        config.gcs_metrics_bucket, 
        os.path.join(config.run_name, PER_EPOCH_TIME_DIRECTORY, base_name), 
        epoch_times
    )
    
    max_logging.log(f"Finished uploading metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}")
    
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
