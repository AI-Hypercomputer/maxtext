# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script converts a HuggingFace model checkpoint to a MaxText-compatible
Orbax checkpoint.

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma2-2b").
              Must be a key in `MaxText.utils.ckpt_conversion.utils.utils.HF_IDS`.
  base_output_directory: (Optional) The directory where the converted HuggingFace
                         checkpoint will be saved. Can be a local path, a GCS
                         path (gs://...), or a HuggingFace Hub repo ID (hf://...).
                         Defaults to "./mt_output/".
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
               This must match the training configuration of the checkpoint.

Environment Variables:
  HF_AUTH_TOKEN: (Required) HuggingFace authentication token, needed to
                 download models from HuggingFace Hub.

Example Usage:
  To convert a gemma2-2b model and save it to a specific directory:

  HF_AUTH_TOKEN="hf_YOUR_TOKEN" python MaxText/utils/ckpt_conversion/to_maxtext.py \
    --model_name="gemma2-2b" \
    --base_output_directory="/path/to/your/output/directory" \
    --scan_layers=False

  For models with scanned layers (e.g., some custom architectures), you might
  need to set scan_layers=True and param_scan_axis accordingly.
"""

import os
import sys
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from absl import app
from flax.training import train_state
from transformers import AutoConfig, AutoModelForCausalLM

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText import optimizers
from MaxText.layers import models, quantizations
from MaxText.train import save_checkpoint
from MaxText.utils.ckpt_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.utils import apply_hook_fns, HF_IDS


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  config = pyconfig.initialize(argv)
  # check the supported model ids
  if config.model_name not in HF_IDS:
    raise ValueError(f"Unsupported model name: {config.model_name}. Supported models are: {list(HF_IDS.keys())}")

  model_id = HF_IDS[config.model_name]
  max_utils.print_system_information()
  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  # Setup JAX distributed system and mesh
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  hf_token = config.hf_access_token
  # Load HuggingFace model, config, and state_dict
  max_logging.log(f"Loading HuggingFace model: {model_id}...")
  hf_config_obj = AutoConfig.from_pretrained(model_id, token=hf_token)
  hf_model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token)
  hf_state_dict_numpy = hf_model.state_dict()
  for k, v in hf_state_dict_numpy.items():
    hf_state_dict_numpy[k] = v.numpy()
  del hf_model
  max_logging.log("HuggingFace model loaded and converted to NumPy.")

  # Initialize MaxText model, optimizer, and abstract state
  rng = jax.random.PRNGKey(config.init_weights_seed)
  quant = quantizations.configure_quantization(config)
  maxtext_model_flax = models.Transformer(config, mesh, quant=quant)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      output_directory,
      enable_checkpointing=True,
      use_async=False,  # Synchronous saving for simplicity in conversion script
      save_interval_steps=1,  # Save at step 0
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )

  abstract_state, _, _, _ = maxtext_utils.setup_training_state(
      maxtext_model_flax, None, tx, config, rng, mesh, checkpoint_manager
  )
  abstract_params_tree = abstract_state.params["params"]
  abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(abstract_params_tree)
  max_logging.log("MaxText abstract model and state initialized.")

  # Get parameter mappings and hooks
  # example of param mapping (gemma2, maxtext:huggingface):
  # "params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale":
  #                                                                    f"model.layers.{global_layer_idx}.input_layernorm.weight",

  model_key = config.model_name
  param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_obj.to_dict(), config.scan_layers)

  # Example of Hook FN mapping, to perform reshape:
  # f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
  hook_fn_map_mt = HOOK_FNS[model_key](hf_config_obj.to_dict(), config.scan_layers, saving_to_hf=False)
  max_logging.log("Parameter mappings and hooks obtained.")

  # Transform weights
  max_logging.log("Starting weight transformation...")
  final_mt_weights = []

  for path_tuple, abstract_leaf_value in abstract_params_flat:
    mt_param_key = "params-" + "-".join(k.key for k in path_tuple)
    mt_target_shape_final = abstract_leaf_value.shape

    hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key)
    if hf_source_keys_or_key is None:
      raise ValueError(f"MaxText parameter {mt_param_key} not found in mapping. Check PARAM_MAPPING for {model_key}.")

    hook_fn_list_or_fn = hook_fn_map_mt.get(mt_param_key)
    final_mt_tensor_numpy = None

    if isinstance(hf_source_keys_or_key, list):  # MaxText param is scanned, built from multiple HF params
      tensors_to_stack = []
      # Determine the shape of an individual slice for hooks
      mt_slice_shape_list = list(mt_target_shape_final)
      del mt_slice_shape_list[config.param_scan_axis]
      mt_slice_shape = tuple(mt_slice_shape_list)

      for hf_key_single in hf_source_keys_or_key:
        if hf_key_single not in hf_state_dict_numpy:
          raise ValueError(f"HuggingFace key {hf_key_single} (for MaxText {mt_param_key}) not found in HF state_dict.")
        hf_tensor_numpy = hf_state_dict_numpy[hf_key_single]
        # The target_shape for the hook should be the shape of the MaxText slice it produces
        processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fn_list_or_fn)
        tensors_to_stack.append(processed_hf_tensor)
      final_mt_tensor_numpy = np.stack(tensors_to_stack, axis=config.param_scan_axis)
    else:  # Single HF source key
      hf_key_single = hf_source_keys_or_key
      if hf_key_single not in hf_state_dict_numpy:
        raise ValueError(f"HuggingFace key {hf_key_single} (for MaxText {mt_param_key}) not found in HF state_dict.")
      hf_tensor_numpy = hf_state_dict_numpy[hf_key_single]
      # The target_shape for the hook is the final MaxText parameter shape
      final_mt_tensor_numpy = apply_hook_fns(hf_tensor_numpy, mt_target_shape_final, hook_fn_list_or_fn)

    if final_mt_tensor_numpy.shape != mt_target_shape_final:
      raise ValueError(
          f"Shape mismatch for {mt_param_key}: "
          f"Expected {mt_target_shape_final}, got {final_mt_tensor_numpy.shape} "
          f"from HF key(s) {hf_source_keys_or_key} after hooks."
      )
    final_mt_weights.append(final_mt_tensor_numpy)

  del abstract_params_flat, hf_state_dict_numpy
  max_logging.log("Weight transformation complete.")

  # Create final MaxText parameters tree
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  del final_mt_weights, abstract_params_treedef

  # Create TrainState for saving.
  final_params_for_state = {"params": jax_weights}
  final_save_state = train_state.TrainState(step=0, apply_fn=None, params=final_params_for_state, tx=None, opt_state={})
  del final_params_for_state

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, 0, final_save_state):
      max_logging.log("saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

  max_logging.log(f"Conversion complete. Checkpoint saved to {output_directory}")


if __name__ == "__main__":
  app.run(main)
