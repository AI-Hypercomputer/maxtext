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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
""" Standalone data loader - only loads data for each training step, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more
import os
from typing import Sequence
import datetime

from absl import app

import numpy as np

import jax

from MaxText import max_logging
from MaxText import pyconfig
from MaxText.data_loader import DataLoader
from MaxText.train import validate_train_config, get_first_step, setup_train_loop


def data_load_loop(config, state=None):
  """Main data loader loop.
  Loads batches of data for each training step.
  """
  _, _, _, _, mesh, _, data_iterator, _, state = setup_train_loop(config, recorder=None)
  data_loader = DataLoader(config, mesh, data_iterator, None)

  example_batch = None

  start = datetime.datetime.now()
  start_step = get_first_step(state)
  example_batch = data_loader.load_next_batch()
  jax.block_until_ready(example_batch)
  first_end = datetime.datetime.now()
  time_to_load_first_batch = first_end - start
  if jax.process_index() == 0:
    max_logging.log(
        f"STANDALONE DATALOADER : First step completed in {time_to_load_first_batch.seconds} seconds, on host 0"
    )

  for _ in np.arange(start_step + 1, config.steps):
    example_batch = data_loader.load_next_batch()

  jax.block_until_ready(example_batch)  # wait until the last batch is read
  end = datetime.datetime.now()
  if jax.process_index() == 0:
    max_logging.log(f"STANDALONE DATALOADER : {config.steps} batches loaded in {(end-start).seconds} seconds, on host 0")
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_cpu_enable_gloo_collectives", True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  max_logging.log(f"Found {jax.process_count()} processes.")
  max_logging.log(f"Found {jax.devices()} devices.")
  if config.dataset_type in ("tfds", "c4_mlperf"):
    os.environ["TFDS_DATA_DIR"] = config.dataset_path
  data_load_loop(config)


if __name__ == "__main__":
  app.run(main)
