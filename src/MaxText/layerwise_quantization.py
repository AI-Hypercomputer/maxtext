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


r"""Layerwise quantization for large models

Provides a utility to load and quantize a checkpoint layer by layer. Currently, it supports DeepSeek-family models only.

Example cmd:

python3 -m MaxText.layerwise_quantization  src/MaxText/configs/base.yml \
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

from tqdm import tqdm
import jax
import jax.numpy as jnp
from absl import app

from flax.linen import partitioning as nn_partitioning
from flax import nnx

from MaxText import checkpointing
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText import common_types
from MaxText.layers import models, quantizations, deepseek
import orbax.checkpoint as ocp


IGNORE = ocp.PLACEHOLDER
PRNGKeyType = Any


class LayerwiseQuantization:
  """
  Layerwise quantization for large models.
  """

  def __init__(self, config: Any):
    self.config = config

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
    rng = jax.random.PRNGKey(1234)
    self.unboxed_abstract_state, _, _ = maxtext_utils.get_abstract_state(model, None, self.config, rng, self._mesh, False)

  def load_and_quantize(self, rng: None | PRNGKeyType = None) -> None:
    """
    Load parameters layer by layer and quantize them.
    """

    quantized_params = {}
    quantized_params["params"] = {"decoder": {}}
    quantized_params["aqt"] = {"decoder": {}}

    config = self.config

    self.quant.quant_mode = quantizations.get_quant_mode("convert")

    model_mode = common_types.MODEL_MODE_PREFILL
    _, rng_quant_params = jax.random.split(rng)
    rngs = nnx.Rngs(rng)

    layers = [
        deepseek.DeepSeekDenseLayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=rngs
        ),
        deepseek.DeepSeekMoELayerToLinen(
            config=config, mesh=self._mesh, quant=self.quant, model_mode=model_mode, rngs=rngs
        ),
    ]
    layer_prefixes = ["dense_layers", "moe_layers"]
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
    num_layers_list = [config.first_num_dense_layers, num_moe_layers]

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
        print(f"Quantizing layer {layer_prefix}_{index}")

        layer_name = f"{layer_prefix}_{index}"
        params = self._load_layer(layer_name)

        params["params"] = params["params"]["decoder"][layer_name]

        _, new_vars = model_apply(params, rng_quant_params, layer)

        quantized_params["aqt"]["decoder"][layer_name] = new_vars["aqt"]
        quantized_params["params"]["decoder"][layer_name] = quantizations.remove_quantized_params(
            params["params"], new_vars["aqt"]
        )

    # Load and save the layers that should not be quantized.
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
    """Creates a partial abstract params structure using ocp.PLACEHOLDER.

    Args:
        abstract_params: The full abstract params structure (e.g., from a TrainState).
        layer: The layer name to keep in the abstract params.

    Returns:
        A new abstract params structure with ocp.PLACEHOLDER for skipped nodes.
    """

    def _should_keep(path, _):
      if layer in [x.key for x in path]:
        return True
      return False

    def _map_fn(path, value):
      if _should_keep(path, value):
        return value
      return IGNORE

    return jax.tree_util.tree_map_with_path(_map_fn, abstract_unboxed_params)


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  validate_config(config)
  max_utils.print_system_information()

  quantization = LayerwiseQuantization(config)
  rng = jax.random.PRNGKey(1234)

  # load_and_quantize will load a checkpoint and quantize if the following parameters are set:
  # quantization=$valid_quantization_type \
  # save_quantized_params_path=$gsbucket_path \
  # checkpoint_is_quantized=false (default)
  quantization.load_and_quantize(rng)


def validate_config(config):
  assert (
      config.load_full_state_path == ""
  ), "Operation on full states not supported! Convert to parameter checkpoint first."


if __name__ == "__main__":
  app.run(main)
