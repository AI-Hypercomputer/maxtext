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
import numpy as np

import math
import time
import tensorflow as tf

import pyconfig
from train import (
    validate_train_config,
    setup_train_loop,
)

from torch_datasets.parquet import ParquetDataset
from torch.utils.data import DataLoader


def parquet_data_loader():
    batch_size = 32

    # [Short-term]: we know what the mount path is, and we know what the dataset names are.
    # Therefore, we save the listing process which will create a listing storm.
    parquet_files = [f"/mnt/gcsfuse/{i:04d}.parquet" for i in range(10000)]

    per_worker = int(math.ceil(len(parquet_files) / float(jax.process_count())))

    worker_id = jax.process_index()
    start = worker_id * per_worker
    end = min(start + per_worker, len(parquet_files))

    dataset = ParquetDataset(
        allocated_parquet_files=parquet_files[start:end],
        batch_size=batch_size,
        columns=["outputs", "image_base64_str"],
    )
    data_loader = DataLoader(dataset=dataset, num_workers=2, batch_size=batch_size)
    return data_loader


def data_load_loop(config):
    """Main data loader loop.
    Loads batches of data for each training step.
    """
    # We seem to need the mesh to be setup for the distributed barrier to work.
    # Therefore, simply calling this function but using our own iterator.
    setup_train_loop(config)
    # data_loader = parquet_data_loader()

    # training_start = datetime.datetime.now()
    # max_logging.log(
    #     f"STANDALONE DATALOADER : Started training loop on host {jax.process_index()}"
    # )
    # for i in range(config.epochs):
    #     epoch_start = datetime.datetime.now()
    #     step_count = 0
    #     step_data_loading_start = datetime.datetime.now()
    #     for _ in data_loader:
    #         step_data_loading_end = datetime.datetime.now()
    #         max_logging.log(
    #             f"STANDALONE DATALOADER : Host {jax.process_index()} got a batch in {step_data_loading_end - step_data_loading_start} seconds on epoch {i} step {step_count}"
    #         )

    #         # Simulate Accelerator Computation Time.
    #         # time.sleep(10)

    #         barrier_start = datetime.datetime.now()
    #         max_logging.log(
    #             f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} started on step {step_count} at {barrier_start}"
    #         )

    #         jax.experimental.multihost_utils.sync_global_devices(
    #             "Barrier before proceeding to the next step"
    #         )

    #         barrier_end = datetime.datetime.now()
    #         max_logging.log(
    #             f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} completed on step {step_count} at {barrier_end}, lasted {barrier_end - barrier_start} seconds"
    #         )
    #         step_count += 1
    #         # We are only interested in counting the per-step data loading time which *excludes* the computation and barrier time.
    #         step_data_loading_start = step_data_loading_end

    #     epoch_end = datetime.datetime.now()
    #     max_logging.log(
    #         f"STANDALONE DATALOADER : Host {jax.process_index()} completed epoch {i} using {epoch_end - epoch_start} seconds"
    #     )

    # training_end = datetime.datetime.now()
    # time_to_load_first_batch = training_end - training_start

    # max_logging.log(
    #     f"STANDALONE DATALOADER : Training completed in {time_to_load_first_batch} seconds, on host {jax.process_index()}"
    # )


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_cpu_enable_gloo_collectives", True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    pyconfig.initialize(argv)
    config = pyconfig.config
    validate_train_config(config)
    max_logging.log(f"Found {jax.device_count()} devices.")
    max_logging.log(f"Found {jax.process_count()} processes.")
    max_logging.log(f"Found {jax.devices()} devices.")
    if config.dataset_type == "tfds":
        os.environ["TFDS_DATA_DIR"] = config.dataset_path
    data_load_loop(config)


if __name__ == "__main__":
    app.run(main)
