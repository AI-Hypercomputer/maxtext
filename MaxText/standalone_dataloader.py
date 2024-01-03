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

from typing import Sequence
import datetime
from absl import app
import numpy as np

import pyconfig
from train import validate_train_config, get_first_step, load_next_batch, setup_train_loop

from jax.experimental.compilation_cache import compilation_cache as cc


def data_load_loop(config, state=None):
  """Main data loader loop.
    Loads batches of data for each training step.
  """
  _, _, _, _, _, _, _, data_iterator, state = setup_train_loop(config)

  example_batch = None

  start = datetime.datetime.now()
  start_step = get_first_step(state)

  # Actual data loading steps
  for step in np.arange(start_step, config.steps):
    example_batch = load_next_batch(data_iterator, example_batch, config)
    # print("Step ", step, " finished in ", new_time - last_step_completion)
    if step==0:
      new_time = datetime.datetime.now()
      print("First step completed in ", new_time-start," seconds")

  end = datetime.datetime.now()
  print(config.steps," batches loaded in ", end-start ," seconds, on host ", jax.process_index())
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_train_config(config)
  cc.initialize_cache(os.path.expanduser(config.jax_cache_dir))
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  data_load_loop(config)



if __name__ == "__main__":
  app.run(main)
