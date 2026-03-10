# Copyright 2023–2026 Google LLC
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


r"""Layerwise quantization for large models

Provides a utility to load and quantize a checkpoint layer by layer. Currently, it supports DeepSeek-family models only.

Example cmd:

python3 -m MaxText.layerwise_quantization  src/maxtext/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${LOAD_PARAMS_PATH} \
  model_name=deepseek2-16b ici_fsdp_parallelism=1 ici_autoregressive_parallelism=1 \
  ici_tensor_parallelism=-1 scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
  attention=dot_product quantization=int8 async_checkpointing=false enable_single_controller=true \
  tokenizer_type=huggingface megablox=false sparse_matmul=false \
  save_quantized_params_path=${SAVE_PARAMS_PATH} checkpoint_storage_use_ocdbt=False \
  checkpoint_storage_use_zarr3=False

"""

import os
from typing import Any, Sequence

from absl import app
from aqt.jax.v2 import aqt_tensor
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.common import checkpointing
from maxtext.layers import quantizations
from maxtext.models import deepseek, models
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
import orbax.checkpoint as ocp
from tqdm import tqdm
from maxtext.configs import pyconfig

IGNORE = ocp.PLACEHOLDER
PRNGKeyType = Any
DictKey = jax.tree_util.DictKey


def get_original_path_key(aqt_k_tuple: tuple[DictKey, ...]) -> tuple[DictKey, ...] | None:
  """
  Maps an AQT PyTree path (tuple of keys) to its corresponding original parameter path.
  Only returns a path if it corresponds to a parameter to be removed.
  """
  aqt_k = list(aqt_k_tuple)
  str_path = jax.tree_util.keystr(aqt_k_tuple)
  if "AqtEinsum_" in str_path:
    return None
  if "AqtDotGeneral_" not in str_path:
    return None
  aqt_module_index = -1
  for i, key in enumerate(aqt_k):
    if isinstance(key, DictKey) and key.key.startswith("AqtDotGeneral_"):
      aqt_module_index = i
      break
  if aqt_module_index == -1:
    return None
  if (
      len(aqt_k) > aqt_module_index + 2
      and isinstance(aqt_k[aqt_module_index + 1], DictKey)
      and aqt_k[aqt_module_index + 1].key == "qrhs"
      and isinstance(aqt_k[aqt_module_index + 2], DictKey)
      and aqt_k[aqt_module_index + 2].key == "frozen"
  ):
    parent_path = tuple(aqt_k[:aqt_module_index])
    return parent_path + (DictKey("kernel"),)
  return None


def get_quantized_param_paths(aqt_params: Any, params: Any) -> set[tuple[DictKey, ...]]:
  """
  Identifies the set of paths in the original params tree that have been quantized.
  """

  def is_qtensor(x):
    return isinstance(x, aqt_tensor.QTensor)

  aqt_param_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_params, is_leaf=is_qtensor)
  if not aqt_param_flat:
    return set()
  param_tree_flat_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
  params_path_set: set[tuple[DictKey, ...]] = {tuple(k) for k, _ in param_tree_flat_with_path}
  original_param_paths_to_remove: set[tuple[DictKey, ...]] = set()
  for aqt_k_tuple, _ in aqt_param_flat:
    original_k_tuple = get_original_path_key(aqt_k_tuple)
    if original_k_tuple is None:
      continue
    if original_k_tuple in params_path_set:
      original_param_paths_to_remove.add(original_k_tuple)
      continue
    params_keys_str = {jax.tree_util.keystr(k) for k in params_path_set}
    raise ValueError(
        f"Mapped AQT path {jax.tree_util.keystr(aqt_k_tuple)} to {jax.tree_util.keystr(original_k_tuple)},"
        f" but not found in params. Available: {params_keys_str}"
    )
  return original_param_paths_to_remove


def remove_quantized_params(params: Any, aqt_vars: Any) -> Any:
  """Replaces the values in the original params tree that are now quantized with empty dicts."""
  quantized_param_path_set = get_quantized_param_paths(aqt_vars, params)
  if not quantized_param_path_set:
    return params

  def _map_fn(path, value):
    return {} if tuple(path) in quantized_param_path_set else value

  return jax.tree_util.tree_map_with_path(_map_fn, params)


# --- Function to restructure NNX-Run AQT tree to match PURE LINEN saved format ---
def insert_deepseekmoeblock_scope(aqt_layer_tree: dict[str, Any]) -> dict[str, Any]:
  """
  Moves top-level AqtEinsum_* entries into the existing 'DeepSeekMoeBlock_0'
  dict to match the pure Linen AQT structure.
  """
  if not isinstance(aqt_layer_tree, dict):
    return aqt_layer_tree

  new_tree = dict(aqt_layer_tree)  # Start with a copy

  einsum_items = {key: new_tree.pop(key) for key in list(new_tree.keys()) if key.startswith("AqtEinsum_")}

  if einsum_items:
    if "DeepSeekMoeBlock_0" not in new_tree:
      # This case indicates the MoE block itself was missing, which is unexpected
      max_logging.log("Error: 'DeepSeekMoeBlock_0' not found in AQT vars for MoE layer.")
      new_tree["DeepSeekMoeBlock_0"] = {}
    elif not isinstance(new_tree["DeepSeekMoeBlock_0"], dict):
      max_logging.log(f"Error: 'DeepSeekMoeBlock_0' is not a dict, type: {type(new_tree['DeepSeekMoeBlock_0'])}")
      new_tree["DeepSeekMoeBlock_0"] = {}

    # Merge einsum_items into the DeepSeekMoeBlock_0 dict
    new_tree["DeepSeekMoeBlock_0"].update(einsum_items)

  return new_tree


class LayerwiseQuantization:
  """
  Layerwise quantization for large models.
  """

  def __init__(self, config: Any, rng: PRNGKeyType):
    self.config = config
    self.rng = rng

    # TODO(ranlihao): Remove this assertion once the Layerwise quantization is supported for other decoder blocks.
    assert (
        config.decoder_block == common_types.DecoderBlockType.DEEPSEEK
    ), f"Layerwise quantization is only supported for {common_types.DecoderBlockType.DEEPSEEK}\
      , but got {config.decoder_block}."
    # Mesh definition
    devices_array = maxtext_utils.create_device_mesh(config=config)
    self._mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    # Model and quantization config
    self.quant = quantizations.configure_quantization(config)
    model = models.transformer_as_linen(
        config, mesh=self._mesh, quant=self.quant, model_mode=common_types.MODEL_MODE_TRAIN
    )
    self.unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(
        model, None, self.config, self.rng, self._mesh, False
    )

  def load_and_quantize(self) -> None:
    """
    Load parameters layer by layer and quantize them.
    """
    quantized_params = {}
    quantized_params["params"] = {"decoder": {}}
    quantized_params["aqt"] = {"decoder": {}}
    config = self.config
    self.quant.quant_mode = quantizations.get_quant_mode("convert")
    model_mode = common_types.MODEL_MODE_PREFILL
    _, rng_quant_params = jax.random.split(self.rng)

    layers = [
        deepseek.DeepSeekDenseLayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=nnx.Rngs(self.rng)
        ),
        deepseek.DeepSeekMoELayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=nnx.Rngs(self.rng)
        ),
    ]
    layer_prefixes = [
        "dense_layers",
        "moe_layers",
    ]
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    num_layers_list = [
        config.first_num_dense_layers,
        num_moe_layers,
    ]

    def model_apply(_p, _rng, layer):
      return layer.apply(
          _p | {"aqt": {}},
          jnp.ones((1, self.config.max_prefill_predict_length, self.config.base_emb_dim), dtype=jnp.int32),
          None,
          jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          True,
          model_mode=model_mode,
          rngs={"params": _rng},
          mutable=True,
      )

    for layer, num_layers, layer_prefix in zip(layers, num_layers_list, layer_prefixes):
      for index in tqdm(range(num_layers)):
        layer_name = f"{layer_prefix}_{index}"
        max_logging.log(f"Processing layer: {layer_name}")

        params = self._load_layer(layer_name)
        params["params"] = params["params"]["decoder"][layer_name]

        _, new_vars = model_apply(params, rng_quant_params, layer)

        if "aqt" not in new_vars:
          max_logging.log(
              f"Warning: 'aqt' not found in new_vars for {layer_name}. Skipping AQT processing for this layer."
          )
          quantized_params["params"]["decoder"][layer_name] = params["params"]  # Keep original params
          continue

        aqt_vars = new_vars["aqt"]

        try:
          removed_params = remove_quantized_params(params["params"], aqt_vars)
          quantized_params["params"]["decoder"][layer_name] = removed_params
        except Exception as e:
          max_logging.log(f"ERROR: Failed to remove quantized params for {layer_name}: {e}")
          max_logging.log(f"Dumping params['params'] keys for {layer_name}:")
          jax.tree_util.tree_map_with_path(
              lambda path, _: max_logging.log(f"  {jax.tree_util.keystr(path)}"), params["params"]
          )
          max_logging.log(f"Dumping new_vars['aqt'] keys for {layer_name}:")
          jax.tree_util.tree_map_with_path(lambda path, _: max_logging.log(f"  {jax.tree_util.keystr(path)}"), aqt_vars)
          raise

        # Restructure the aqt_vars for this layer to match pure Linen format for saving
        if layer_prefix == "moe_layers":
          structured_aqt = insert_deepseekmoeblock_scope(aqt_vars)
        else:
          structured_aqt = aqt_vars
        quantized_params["aqt"]["decoder"][layer_name] = structured_aqt

    unquantized_layers = ["decoder_norm", "logits_dense"]
    for unquantized_layer in unquantized_layers:
      params = self._load_layer(unquantized_layer)
      quantized_params["params"]["decoder"][unquantized_layer] = params["params"]["decoder"][unquantized_layer]
    quantized_params["params"]["token_embedder"] = self._load_layer("token_embedder")["params"]["token_embedder"]

    maxtext_utils.save_quantized_checkpoint_if_configured(self.config, quantized_params)

  def _load_layer(self, layer_name):
    """Loads a specific layer's parameters from the checkpoint."""

    config = self.config
    with nn_partitioning.axis_rules(config.logical_axis_rules):

      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          self._create_partial_abstract_params(self.unboxed_abstract_state.params, layer_name),
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )
    return params

  def _create_partial_abstract_params(self, abstract_unboxed_params, layer):
    """Creates a partial abstract params structure using ocp.PLACEHOLDER."""

    def _should_keep(path, _):
      # True if the layer name is part of the path
      return any(isinstance(key, jax.tree_util.DictKey) and key.key == layer for key in path)

    def _map_fn(path, value):
      if not _should_keep(path, value):
        return IGNORE
      if isinstance(value, jax.ShapeDtypeStruct):
        zeros_array = jnp.zeros(value.shape, value.dtype)
        if value.sharding is not None:
          try:
            return jax.device_put(zeros_array, value.sharding)
          except Exception as e:  # pylint: disable=broad-except
            max_logging.log(f"Error applying sharding for path {path}: {e}")
            return zeros_array
        return zeros_array
      return value

    return jax.tree_util.tree_map_with_path(_map_fn, abstract_unboxed_params)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  validate_config(config)
  max_utils.print_system_information()
  rng = jax.random.PRNGKey(1234)
  quantization = LayerwiseQuantization(config, rng)
  quantization.load_and_quantize()


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
