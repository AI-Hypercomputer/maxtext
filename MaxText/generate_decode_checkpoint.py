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
"""Generate a decode checkpoint with no opt_state from a training checkpoint."""

import sys

import checkpointing
import jax
import max_logging
import max_utils
import maxtext_utils
import pyconfig

from absl import app
from etils import epath
from jax.sharding import Mesh
from jax import random
from typing import Sequence
from layers import model

Transformer = model.Transformer


def _read_checkpoint(config, checkpoint_manager, is_training):
  """Read training checkpoint at path defined by load_parameters_path."""
  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  # Model and Optimizer definition
  model = Transformer(config, mesh)
  rng = random.PRNGKey(0)
  if is_training:
    learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
    tx = maxtext_utils.get_optimizer(config, learning_rate_schedule)
    state, state_mesh_notations = max_utils.setup_training_state(
      model, tx, config, rng, mesh, checkpoint_manager
      )
  else:
    state, state_mesh_notations = max_utils.setup_decode_state(
      model, config, rng, mesh, checkpoint_manager
      )
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"Read checkpoint is_training: {is_training}")
  max_logging.log(f"Number of model params={num_params/10**9:.3f} billion")
  return state, state_mesh_notations

def read_decode_checkpoint(config, checkpoint_manager):
  is_training = False
  return _read_checkpoint(config, checkpoint_manager, is_training)

def read_training_checkpoint(config, checkpoint_manager):
  is_training = True
  return _read_checkpoint(config, checkpoint_manager, is_training)

def _save_decode_checkpoint(config, state, checkpoint_manager):
  """Generate checkpoint for decode from the training_state."""
  with jax.spmd_mode('allow_all'):
    decode_state = max_utils.init_decode_state(state.apply_fn, state.params)
  if checkpoint_manager is not None:
    if checkpoint_manager.save(0, decode_state):
      max_logging.log(f"saved an decode checkpoint at {config.checkpoint_dir}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

def generate_decode_checkpoint(config):
  """
  Generate an decode checkpoint from a given training checkpoint.
  - Training checkpoint is saved at config.load_parameters_path.
  - Inference checkpoint will be saved at config.checkpoint_dir.
  """
  assert config.checkpoint_dir, "checkpoint_dir not configured"
  # Remove any old checkpoint
  path = epath.Path(config.checkpoint_dir)
  if path.exists():
    if jax.process_index() == 0:
      path.rmtree()

  assert config.load_parameters_path, "load_parameters_path not configured"
  assert epath.Path(config.load_parameters_path).exists(), "no checkpoint at load_parameters_path"

  # Create a checkpoint manager to save decode checkpoint at config.checkpoint_dir
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.save_period,
  )
  # Read training state from config.load_paramaters_path
  max_logging.log(f"Read training checkpoint from: {config.load_parameters_path}")
  training_state, _ = read_training_checkpoint(config, checkpoint_manager)
  assert training_state.opt_state != {}, "missing opt_state in training checkpoint"

  # Save decode state to config.checkpoint_dir at step 0
  max_logging.log(f"Save decode checkpoint at: {config.checkpoint_dir}")
  _save_decode_checkpoint(config, training_state, checkpoint_manager)

  # Read  newly saved decode state from config.checkpoint_dir at step 0
  max_logging.log(f"Read decode checkpoint from: {config.checkpoint_dir}")
  decode_state, _ = read_decode_checkpoint(config, checkpoint_manager)
  assert decode_state.opt_state == {}, "non null opt_state in decode checkpoint"
  num_training_params = max_utils.calculate_num_params_from_pytree(training_state.params)
  num_decode_params = max_utils.calculate_num_params_from_pytree(decode_state.params)
  assert num_training_params == num_decode_params, "count mismatch in params in checkpoints"

  max_logging.log(f"Successfully generated decode checkpoint at: {config.checkpoint_dir}0/default")
  return True


def main(argv: Sequence[str]) -> None:
  print(argv)
  pyconfig.initialize(argv)
  generate_decode_checkpoint(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
