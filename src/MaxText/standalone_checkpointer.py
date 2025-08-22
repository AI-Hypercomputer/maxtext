# Copyright 2023–2025 Google LLC
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with
"""Standalone checkpointer - only saves and restores checkpoints at regular intervals, accesses storage needs."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import datetime
import os
from typing import Sequence

from absl import app

import numpy as np

import jax
from jax import numpy as jnp

from flax.linen import partitioning as nn_partitioning

import MaxText as mt
from MaxText import checkpointing
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import max_logging
from MaxText import pyconfig
from MaxText.train import get_first_step, validate_train_config
from MaxText.layers import models

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
  model = mt.from_pretrained(config)
  mesh = model.mesh
  init_rng, checkpoint_manager, _, tx = train_utils.create_training_tools(config, model, mesh)

  unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
  # A barrier to sync all hosts before starting to restore checkpoint
  jax.experimental.multihost_utils.sync_global_devices("Barrier before load")
  checkpoint_load_start = datetime.datetime.now()
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    state, _ = checkpointing.load_state_if_possible(
        checkpoint_manager,
        None,
        config.load_parameters_path,
        config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb,
        unboxed_abstract_state,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    if state:
      state = state["items"]

  jax.block_until_ready(state)
  checkpoint_load_end = datetime.datetime.now()
  if state is not None:  # Checkpoint was available for restore
    if jax.process_index() == 0:
      max_logging.log(f"STANDALONE CHECKPOINTER : Checkpoint restored in : {checkpoint_load_end - checkpoint_load_start}")
  else:  # Checkpoint was unavailable, state needs to be initialized
    state, _, _, _ = maxtext_utils.setup_training_state(model, None, tx, config, init_rng, mesh, checkpoint_manager)
  state = add_entropy_to_checkpoint(state)

  start_step = get_first_step(state)  # this is the start_step for training
  for step in np.arange(start_step, config.steps):
    if checkpoint_manager is not None:
      start_time = datetime.datetime.now()
      # A barrier to sync all hosts before starting to save checkpoint
      jax.experimental.multihost_utils.sync_global_devices("Barrier before save")
      if checkpointing.save_checkpoint(checkpoint_manager, int(step), state):
        checkpoint_manager.wait_until_finished()
        end_time = datetime.datetime.now()
        if jax.process_index() == 0:
          max_logging.log(
              f"STANDALONE CHECKPOINTER : Checkpoint saved in {end_time - start_time} ,step {step}, on host 0"
          )

  return state


def add_entropy_to_checkpoint(state):
  """Introduce randomness in checkpoints. This is useful to simulate real checkpoints, without training.
  Args:
    state: Initial state
  Returns:
    state: Returns state with entropy added to the optimizer state.
  """
  opt_0 = state.opt_state[0]
  opt_0 = opt_0._replace(mu=jax.tree_util.tree_map(lambda k: jnp.cos(1000 * k), state.params))
  opt_0 = opt_0._replace(nu=jax.tree_util.tree_map(lambda k: jnp.sin(1000 * k), state.params))
  new_opt = [opt_0] + list(state.opt_state[1:])
  state = state.replace(opt_state=new_opt)
  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_cpu_enable_gloo_collectives", True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  validate_train_config(config)
  print(f"Found {jax.device_count()} devices.")
  print(f"Found {jax.process_count()} processes.")
  print(f"Found {jax.devices()} devices.")
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  checkpoint_loop(config)


if __name__ == "__main__":
  app.run(main)
