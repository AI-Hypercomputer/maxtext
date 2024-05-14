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
"""Transforms a "full state" including optimizer state to a bfloat16 "parameter state" without optimizer state.
   This typically used for turning a state output by training.py into a state than can be consumed by decode.py.

   The input "fullstate" is passed in via:
     load_full_state_path.
   The output "parameter state" is output to the checkpoint directory. Additionally it is cast down to bf16.
"""

import checkpointing
import jax
import max_logging
import max_utils
import optimizers
import pyconfig

from absl import app
from etils import epath
from jax.sharding import Mesh
from jax import random
from typing import Sequence
from layers import models, quantizations
from train import save_checkpoint

Transformer = models.Transformer


def _possibly_unroll_params(config, training_state, training_state_annotations, mesh):
  """If input layers are scanned, and force_unroll is set,
  return modify training_state and train_state_annotations to be "unrolled".
  Otherwise do nothing."""
  if not config.scan_layers or not config.force_unroll:
    return

  training_state_layers = training_state.params["params"]["decoder"]["layers"]
  training_state_annotations_layers = training_state_annotations.params["params"]["decoder"]["layers"]

  def new_pspec(x):
    return jax.sharding.PartitionSpec(*x[0 : config.param_scan_axis] + x[config.param_scan_axis + 1 :])

  new_per_layer_state_annotation = jax.tree_util.tree_map(new_pspec, training_state_annotations_layers)
  new_per_layer_state_sharding = jax.tree_util.tree_map(
    lambda x: jax.sharding.NamedSharding(mesh, x), new_per_layer_state_annotation)

  for i in range(config.num_decoder_layers):

    def slice_ith(input_layers):
      return jax.tree_util.tree_map(lambda x: jax.numpy.take(x, i, axis=config.param_scan_axis), input_layers)

    new_layer = jax.jit(slice_ith, out_shardings=new_per_layer_state_sharding)(training_state_layers)

    training_state.params["params"]["decoder"][f"layers_{i}"] = new_layer
    training_state_annotations.params["params"]["decoder"][f"layers_{i}"] = new_per_layer_state_annotation

  del training_state.params["params"]["decoder"]["layers"]
  del training_state_annotations.params["params"]["decoder"]["layers"]

  jax.tree_util.tree_map(lambda x: x.delete(), training_state_layers)


def _read_train_checkpoint(config, checkpoint_manager, mesh):
  """Read training checkpoint at path defined by load_full_state_path."""
  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant)
  rng = random.PRNGKey(0)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  state, state_mesh_notations, _ = max_utils.setup_training_state(model, None, tx, config, rng, mesh, checkpoint_manager)
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"In input checkpoint Number of model params={num_params/1e9:.3f} billion")
  return state, state_mesh_notations


def _save_decode_checkpoint(config, state, checkpoint_manager):
  """Generate checkpoint for decode from the training_state."""
  with jax.spmd_mode("allow_all"):
    decode_state = max_utils.init_decode_state(
      None, jax.tree_util.tree_map(lambda x: x.astype(jax.numpy.bfloat16), state.params))
  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, 0, decode_state):
      max_logging.log(f"saved an decode checkpoint at {config.checkpoint_dir}")
  checkpoint_manager.wait_until_finished()


def generate_decode_checkpoint(config):
  """
  Generate an decode checkpoint from a given training checkpoint.
  - Training checkpoint is loaded from config.load_full_state_path.
  - Inference checkpoint will be saved at the config's checkpoint directory.
  """

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  assert config.checkpoint_dir, "checkpoint_dir not configured"
  # Remove any old checkpoint
  path = epath.Path(config.checkpoint_dir)
  if path.exists():
    if jax.process_index() == 0:
      path.rmtree()

  # Create a checkpoint manager to save decode checkpoint at config.checkpoint_dir
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
  )
  # Read training state from config.load_paramaters_path
  max_logging.log(f"Read training checkpoint from: {config.load_full_state_path}")
  training_state, training_state_annotations = _read_train_checkpoint(config, checkpoint_manager, mesh)
  assert training_state.opt_state != {}, "missing opt_state in training checkpoint"

  _possibly_unroll_params(config, training_state, training_state_annotations, mesh)

  # Save decode state to config's checkpoint directory at step 0
  max_logging.log(f"Save decode checkpoint at: {config.checkpoint_dir}")
  _save_decode_checkpoint(config, training_state, checkpoint_manager)
  max_logging.log(f"Successfully generated decode checkpoint at: {config.checkpoint_dir}0/items")
  return True


def main(argv: Sequence[str]) -> None:
  print(argv)
  pyconfig.initialize(argv)
  generate_decode_checkpoint(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
