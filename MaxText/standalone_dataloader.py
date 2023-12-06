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
""" Standalone data loader - only loads data for each training step."""

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
  _, _, _, _, _, _, data_iterator, state = setup_train_loop(config)
  
  example_batch = None
  last_step_completion = datetime.datetime.now()
  print("last_step_completion is ", last_step_completion)
  
  start_step = get_first_step(state)
  start = datetime.datetime.now()
  # Actual training steps
  for step in np.arange(start_step, config.steps):
    print("Step ", step, " is starting at ", last_step_completion)
    example_batch = load_next_batch(data_iterator, example_batch, config)
    print(" Shape of example batch: ", example_batch)
    new_time = datetime.datetime.now()
    last_step_completion = new_time
  end = datetime.datetime.now()
  print("NEW : data_loading_test.py: preprocessing_pipeline ROSHANI loading batches done in ", end-start ," seconds. " )
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
#   jax.config.update('jax_platform_name', 'cpu')
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  # os.environ["JAX_PLATFORMS"] = "cpu"
  print(f"ROSHANI Found {jax.device_count()} devices.")
  print(f"ROSHANI Found {jax.process_count()} processes.")
  # print(f"ROSHANI ... \n Found {jax.devices()} devices.")
  cc.initialize_cache(os.path.expanduser("~/jax_cache"))
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  data_load_loop(config)



if __name__ == "__main__":
  app.run(main)
