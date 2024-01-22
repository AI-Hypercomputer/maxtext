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
"""Standalone checkpointer - only saves and restores checkpoints at regular intervals, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os

from typing import Sequence
from absl import app
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np

import checkpointing
import max_utils
import max_logging
import pyconfig
from train import setup_mesh_and_model, get_first_step, validate_train_config

from layers import models

from jax.experimental.compilation_cache import compilation_cache as cc

Transformer = models.Transformer

def checkpoint_loop(config, state=None):
  """Main Checkpointing loop.
  Saves checkpoints.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  init_rng, writer, checkpoint_manager, mesh, model, _, tx = setup_mesh_and_model(config)

  unboxed_abstract_state, state_mesh_annotations = max_utils.get_abstract_state(model, tx,
                                                config, init_rng, mesh, is_training=True)
  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("dummy1")
  checkpoint_load_start = datetime.datetime.now()
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, _ = checkpointing.load_state_if_possible(checkpoint_manager,
                                                config.load_parameters_path,
                                                config.load_full_state_path,
                                                unboxed_abstract_state,
                                                mesh,
                                                state_mesh_annotations)
  jax.block_until_ready(state)
  checkpoint_load_end = datetime.datetime.now()
  if state is not None: # Checkpoint was available for restore
    if jax.process_index() == 0:
      max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint restored in : {checkpoint_load_end - checkpoint_load_start}")
  else: # Checkpoint was unavailable, state needs to be initialized
    state, state_mesh_annotations = max_utils.setup_training_state(model,
          tx, config, init_rng, mesh, checkpoint_manager)

  start_step = get_first_step(state) # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("dummy2")
      if checkpoint_manager.save(step, state):
        checkpoint_manager.wait_until_finished()
        end_time = datetime.datetime.now()
        if jax.process_index() == 0:
          max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint saved in {end_time - start_time} ,step {step}, on host 0")

  writer.close()
  return state

def main(argv: Sequence[str]) -> None:
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_train_config(config)
  cc.initialize_cache(os.path.expanduser(config.jax_cache_dir))
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  checkpoint_loop(config)


if __name__ == "__main__":
  app.run(main)
