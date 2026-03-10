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
"""Transforms a "full state" including optimizer state to a bfloat16 "parameter state" without optimizer state.

This typically used for turning a state output by training.py into a state than can be consumed by decode.py.

The input "fullstate" is passed in via ``load_full_state_path``.

The output "parameter state" is output to the checkpoint directory. Additionally it is cast down to bf16.
"""

import os.path
from typing import Sequence

from absl import app
from etils import epath
import jax
from jax import random
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common import checkpointing
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.utils import gcs_utils
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils

Transformer = models.transformer_as_linen


def _possibly_unroll_params(config, training_state, training_state_annotations, mesh):
  """Unroll scanned input layers when force_unroll is set."""
  if not config.scan_layers or not config.force_unroll:
    return

  def unroll_layer_group(num_layers, layer_name="layers"):
    """Helper function to unroll layers (e.g. dense or MoE) into individual layers."""
    layers = training_state.params["params"]["decoder"].get(layer_name, None)
    layers_annotations = training_state_annotations.params["params"]["decoder"].get(layer_name, None)

    if layers is None or layers_annotations is None:
      raise ValueError(f"Missing {layer_name} in training_state or training_state_annotations.")

    def new_pspec(x):
      return jax.sharding.PartitionSpec(*(x[0 : config.param_scan_axis] + x[config.param_scan_axis + 1 :]))

    new_layer_annotation = jax.tree_util.tree_map(new_pspec, layers_annotations)
    new_layer_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), new_layer_annotation)

    for i in range(num_layers):

      def slice_ith(input_layers):
        return jax.tree_util.tree_map(lambda x: jax.numpy.take(x, i, axis=config.param_scan_axis), input_layers)

      # pylint: disable=not-callable
      new_layer = jax.jit(slice_ith, out_shardings=new_layer_sharding)(layers)

      training_state.params["params"]["decoder"][f"{layer_name}_{i}"] = new_layer
      training_state_annotations.params["params"]["decoder"][f"{layer_name}_{i}"] = new_layer_annotation

    # Remove the original layer collection
    del training_state.params["params"]["decoder"][layer_name]
    del training_state_annotations.params["params"]["decoder"][layer_name]

    jax.tree_util.tree_map(lambda x: x.delete(), layers)

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    # Unroll dense and MoE layers separately
    unroll_layer_group(config.first_num_dense_layers, layer_name="dense_layers")
    unroll_layer_group(config.num_decoder_layers - config.first_num_dense_layers, layer_name="moe_layers")
  else:
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")


def _read_train_checkpoint(config, checkpoint_manager, mesh):
  """Read training checkpoint at path defined by load_full_state_path."""
  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant, MODEL_MODE_TRAIN)
  rng = random.PRNGKey(0)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  state, state_mesh_notations, _, _ = maxtext_utils.setup_training_state(
      model, None, tx, config, rng, mesh, checkpoint_manager
  )
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"In input checkpoint Number of model params={num_params/1e9:.3f} billion")
  return state, state_mesh_notations


def _generate_lora_decode_checkpoints(config, mesh):
  """Read lora checkpoints checkpoint at path defined by load_full_state_path."""
  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant, MODEL_MODE_TRAIN)
  rng = random.PRNGKey(0)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)

  lora_adapters = gcs_utils.gcs_list_directories(config.lora_input_adapters_path)
  for lora_id in lora_adapters:
    # Expected lora_checkpoint_dir = <checkpoint_dir>/loras/<lora_id>
    lora_checkpoint_dir = os.path.join(config.checkpoint_dir, "loras", lora_id, "")

    lora_adapter_path = os.path.join(config.lora_input_adapters_path, lora_id, "")

    # Create a checkpoint manager to save decode checkpoint at lora_checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        lora_checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
    )

    lora_config, lora_state, lora_state_annotations = lora_utils.setup_initial_lora_state(
        model, None, tx, config, rng, mesh, checkpoint_manager, lora_adapter_path
    )

    _possibly_unroll_params(config, lora_state, lora_state_annotations, mesh)

    gcs_utils.write_dict_to_gcs_json(lora_config, os.path.join(lora_checkpoint_dir, "adapter_config.json"))

    # Save decode state to config's checkpoint directory at step 0
    _save_decode_checkpoint(config, lora_state, checkpoint_manager)
    max_logging.log(f"Successfully saved LoRA checkpoint at: {os.path.join(lora_checkpoint_dir, '0', 'items')}")


def _save_decode_checkpoint(config, state, checkpoint_manager):
  """Generate checkpoint for decode from the training_state."""
  decode_state = maxtext_utils.init_decode_state(
      None, jax.tree_util.tree_map(lambda x: x.astype(jax.numpy.bfloat16), state.params)
  )
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, decode_state):
      max_logging.log(f"saved an decode checkpoint at {config.checkpoint_dir}")
  checkpoint_manager.wait_until_finished()


def generate_decode_checkpoint(config):
  """
  Generate an decode checkpoint from a given training checkpoint.

  * Training checkpoint is loaded from config.load_full_state_path.
  * Inference checkpoint will be saved at the config's checkpoint directory.
  """

  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  assert config.checkpoint_dir, "checkpoint_dir not configured"
  # Remove any old checkpoint
  path = epath.Path(config.checkpoint_dir)
  if path.exists():
    if jax.process_index() == 0:
      path.rmtree()

  # Create a checkpoint manager to save decode checkpoint at config.checkpoint_dir
  base_checkpoint_dir = config.checkpoint_dir

  if config.lora_input_adapters_path:
    base_checkpoint_dir += "base/"

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      base_checkpoint_dir,
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
  max_logging.log(f"Save decode checkpoint at: {base_checkpoint_dir}")
  _save_decode_checkpoint(config, training_state, checkpoint_manager)
  max_logging.log(f"Successfully generated decode checkpoint at: {base_checkpoint_dir}0/items")

  if config.lora_input_adapters_path:
    _generate_lora_decode_checkpoints(config, mesh)

  return True


def main(argv: Sequence[str]) -> None:
  print(argv)
  config = pyconfig.initialize(argv)
  generate_decode_checkpoint(config)


if __name__ == "__main__":
  app.run(main)
