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

import functools
import os.path
from typing import Sequence

from absl import app
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common import checkpointing
from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN
from maxtext.layers import quantizations
from maxtext.layers import train_state_nnx
from maxtext.models import models
from maxtext.optimizers import optimizers
from maxtext.utils import gcs_utils
from maxtext.utils import lora_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils import train_utils


def _possibly_unroll_params(config, training_state, training_state_annotations, mesh):
  """Unroll scanned input layers when force_unroll is set."""
  if not config.scan_layers or not config.force_unroll:
    return
  if config.pure_nnx:
    _possibly_unroll_params_nnx(config, training_state, training_state_annotations, mesh)
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


def _possibly_unroll_params_nnx(config, state, state_mesh_shardings, mesh):
  """NNX equivalent of _possibly_unroll_params.

  `state` is a flat `nnx.State` (post-split TrainStateNNX) with `state.model`
  as a sub-State whose tree mirrors the model module hierarchy. Slices
  `state.model.decoder[layer_name]` into per-index `layer_name_0..N` siblings
  and removes the original collection. Mirrors the same operation on
  `state_mesh_shardings` so downstream sharding stays correct.
  """
  decoder_state = state.model.decoder
  decoder_shardings = state_mesh_shardings.model.decoder

  def unroll_layer_group(num_layers, layer_name="layers"):
    layers = decoder_state.get(layer_name, None)
    layers_shardings = decoder_shardings.get(layer_name, None)
    if layers is None or layers_shardings is None:
      raise ValueError(f"Missing {layer_name} in NNX state.model.decoder or state_mesh_shardings.")

    def drop_scan_axis(named_sharding):
      ps = named_sharding.spec
      return jax.sharding.PartitionSpec(*(ps[0 : config.param_scan_axis] + ps[config.param_scan_axis + 1 :]))

    new_layer_pspec = jax.tree_util.tree_map(
        drop_scan_axis, layers_shardings, is_leaf=lambda x: isinstance(x, jax.sharding.NamedSharding)
    )
    new_layer_sharding = jax.tree_util.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), new_layer_pspec)

    for i in range(num_layers):

      def slice_ith(input_layers):
        return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=config.param_scan_axis), input_layers)

      # pylint: disable=not-callable
      new_layer = jax.jit(slice_ith, out_shardings=new_layer_sharding)(layers)

      decoder_state[f"{layer_name}_{i}"] = new_layer
      decoder_shardings[f"{layer_name}_{i}"] = new_layer_sharding

    decoder_state.pop(layer_name)
    decoder_shardings.pop(layer_name)
    jax.tree_util.tree_map(lambda x: x.delete() if hasattr(x, "delete") else None, layers)

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    unroll_layer_group(config.first_num_dense_layers, layer_name="dense_layers")
    unroll_layer_group(config.num_decoder_layers - config.first_num_dense_layers, layer_name="moe_layers")
  else:
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")


def _read_train_checkpoint(config, checkpoint_manager, mesh):
  """Read training checkpoint at path defined by load_full_state_path."""
  rng = random.PRNGKey(0)
  if config.pure_nnx:
    rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=rng)
    model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
    _, tx = train_utils.create_training_optimizer(config, model)
    _create_model_partial, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

    def init_state_fn():
      nnx_model = _create_model_partial()
      optimizer = nnx.Optimizer(nnx_model, tx, wrt=nnx.Param)
      return train_state_nnx.TrainStateNNX(nnx_model, optimizer)

  else:
    quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(config, mesh, quant, MODEL_MODE_TRAIN)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    tx = optimizers.get_optimizer(config, learning_rate_schedule)
    init_state_fn = functools.partial(maxtext_utils.init_initial_state, model, tx, config, True, rng)

  state, state_mesh_notations, state_mesh_shardings, _ = maxtext_utils.setup_training_state(
      None, config, mesh, checkpoint_manager, init_state_fn
  )
  if config.pure_nnx:
    # On NNX, state is a flat nnx.State; params live under state.model and the
    # legacy notations are unused (callers receive shardings directly).
    num_params = max_utils.calculate_num_params_from_pytree(state.model)
    max_logging.log(f"In input checkpoint Number of model params={num_params/1e9:.3f} billion")
    return state, state_mesh_shardings
  num_params = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"In input checkpoint Number of model params={num_params/1e9:.3f} billion")
  return state, state_mesh_notations


def _generate_lora_decode_checkpoints(config, mesh):
  """Read lora checkpoints checkpoint at path defined by load_full_state_path."""
  if config.pure_nnx:
    _generate_lora_decode_checkpoints_nnx(config, mesh)
    return
  quant = quantizations.configure_quantization(config)
  model = models.transformer_as_linen(config, mesh, quant, MODEL_MODE_TRAIN)
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
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
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
  if config.pure_nnx:
    _save_decode_checkpoint_nnx(config, state, checkpoint_manager)
    return
  decode_state = maxtext_utils.init_decode_state(
      None, jax.tree_util.tree_map(lambda x: x.astype(jax.numpy.bfloat16), state.params)
  )
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, decode_state):
      max_logging.log(f"saved an decode checkpoint at {config.checkpoint_dir}")
  checkpoint_manager.wait_until_finished()


def _save_decode_checkpoint_nnx(config, state, checkpoint_manager):
  """Save a bf16 NNX-format param-only decode checkpoint.

  The on-disk shape mirrors what a vanilla NNX-trained checkpoint produces: a
  plain dict tree of arrays (one per nnx.Param), with no Linen-style "params"
  wrapper. This is the shape `from_pretrained` reads via its NNX-detection
  branch (see model_creation_utils._adjust_target_for_moe_fusion / "is_nnx_checkpoint").
  """
  pure_model = state.model.to_pure_dict() if hasattr(state.model, "to_pure_dict") else dict(state.model)
  bf16_model = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), pure_model)
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, bf16_model):
      max_logging.log(f"saved an NNX decode checkpoint at {config.checkpoint_dir}")
    checkpoint_manager.wait_until_finished()


def _possibly_unroll_lora_params_nnx(config, lora_state, lora_state_annotations, mesh):
  """Unroll scanned LoRA delta layers when force_unroll is set on the NNX path.

  `lora_state` is a Linen-style `TrainState` (returned by `get_lora_abstract_state_nnx`)
  whose `.params` is single-nested (`{"decoder": {...}}`, no outer `params` wrap)
  and whose leaves at target attention paths are `lora_a.kernel`/`lora_b.kernel`.
  """
  if not config.scan_layers or not config.force_unroll:
    return

  decoder_params = lora_state.params["decoder"]
  decoder_annotations = lora_state_annotations.params["decoder"]

  def unroll_layer_group(num_layers, layer_name="layers"):
    layers = decoder_params.get(layer_name)
    layers_annotations = decoder_annotations.get(layer_name)
    if layers is None or layers_annotations is None:
      return  # No LoRA on this layer group; nothing to unroll.

    def new_pspec(x):
      return jax.sharding.PartitionSpec(*(x[0 : config.param_scan_axis] + x[config.param_scan_axis + 1 :]))

    new_layer_annotation = jax.tree_util.tree_map(new_pspec, layers_annotations)
    new_layer_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), new_layer_annotation)

    for i in range(num_layers):

      def slice_ith(input_layers):
        return jax.tree_util.tree_map(lambda x: jnp.take(x, i, axis=config.param_scan_axis), input_layers)

      # pylint: disable=not-callable
      new_layer = jax.jit(slice_ith, out_shardings=new_layer_sharding)(layers)
      decoder_params[f"{layer_name}_{i}"] = new_layer
      decoder_annotations[f"{layer_name}_{i}"] = new_layer_annotation

    del decoder_params[layer_name]
    del decoder_annotations[layer_name]
    jax.tree_util.tree_map(lambda x: x.delete() if hasattr(x, "delete") else None, layers)

  if config.decoder_block == DecoderBlockType.DEEPSEEK:
    unroll_layer_group(config.first_num_dense_layers, layer_name="dense_layers")
    unroll_layer_group(config.num_decoder_layers - config.first_num_dense_layers, layer_name="moe_layers")
  else:
    unroll_layer_group(config.num_decoder_layers, layer_name="layers")


def _save_lora_decode_checkpoint_nnx(config, lora_state, checkpoint_manager):
  """Save a bf16 LoRA-only decode checkpoint (NNX path).

  `lora_state.params` is single-nested (NNX-derived shape). The on-disk
  format mirrors the Linen LoRA decode shape so existing serving consumers
  can keep reading it: a `TrainState` wrapper with `params` set to the
  bf16-cast LoRA delta tree. The base model is loaded separately at serve
  time via `apply_lora_on_base_params_nnx`.
  """
  decode_state = maxtext_utils.init_decode_state(
      None, jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), lora_state.params)
  )
  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, decode_state):
      max_logging.log(f"saved a LoRA decode checkpoint at {config.checkpoint_dir}")
    checkpoint_manager.wait_until_finished()


def _generate_lora_decode_checkpoints_nnx(config, mesh):
  """NNX-shaped sibling of `_generate_lora_decode_checkpoints`.

  Builds the NNX abstract base model so `setup_initial_lora_state`
  produces an NNX-derived `lora_state`, then runs an NNX-shape unroll/save.
  """
  rng = random.PRNGKey(0)
  rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=rng)
  model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
  _, tx = train_utils.create_training_optimizer(config, model)

  lora_adapters = gcs_utils.gcs_list_directories(config.lora_input_adapters_path)
  for lora_id in lora_adapters:
    lora_checkpoint_dir = os.path.join(config.checkpoint_dir, "loras", lora_id, "")
    lora_adapter_path = os.path.join(config.lora_input_adapters_path, lora_id, "")

    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        lora_checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
    )

    lora_config, lora_state, lora_state_annotations = lora_utils.setup_initial_lora_state(
        model, None, tx, config, rng, mesh, checkpoint_manager, lora_adapter_path
    )

    _possibly_unroll_lora_params_nnx(config, lora_state, lora_state_annotations, mesh)

    gcs_utils.write_dict_to_gcs_json(lora_config, os.path.join(lora_checkpoint_dir, "adapter_config.json"))

    _save_lora_decode_checkpoint_nnx(config, lora_state, checkpoint_manager)
    max_logging.log(f"Successfully saved LoRA checkpoint at: {os.path.join(lora_checkpoint_dir, '0', 'items')}")


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
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
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
