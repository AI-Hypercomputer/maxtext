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

  HF_AUTH_TOKEN="hf_YOUR_TOKEN" python src/MaxText/utils/ckpt_conversion/to_maxtext.py \
    --model_name="gemma2-2b" \
    --base_output_directory="/path/to/your/output/directory" \
    --scan_layers=False

  For models with scanned layers (e.g., some custom architectures), you might
  need to set scan_layers=True and param_scan_axis accordingly.
"""

import os
import sys
from typing import Sequence, List, Dict, Any
import time
import argparse

import numpy as np
import jax
from absl import app
from flax.training import train_state
from transformers import AutoConfig, AutoModelForCausalLM

import logging
import ml_dtypes
import pathlib
import psutil

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText import optimizers
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations
from MaxText.checkpointing import save_checkpoint
from MaxText.utils.ckpt_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.utils import apply_hook_fns, HF_IDS
import jax.numpy as jnp
import torch
from tqdm import tqdm
from safetensors import safe_open


jax.config.update("jax_platform_name", "cpu")

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"


CAST_DTYPE = ml_dtypes.bfloat16
mem_info = psutil.Process()


# Only skip the MTP weights that are shared with the main model.
# The MTP block in MaxText will reuse the main embedding and output head.
MTP_KEYS_TO_SKIP = [
    "model.layers.61.embed_tokens.weight",
    "model.layers.61.shared_head.norm.weight",
    "model.layers.61.shared_head.head.weight",
]


def is_key_allowed(key, banned_keys) -> bool:
  """
  Checks if a key is NOT in a list of banned keys.
  """
  return key not in banned_keys


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]], hf_state_dict: Dict[str, np.ndarray], hook_fns: Any
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles the complex case for scanned MoE layers, producing a tensor
  with the shape (num_experts, num_layers, ...).

  Args:
      hf_source_keys: A nested (2D) list of Hugging Face parameter names.
                      Outer list iterates experts, inner list iterates layers.
      hf_state_dict: The dictionary of loaded Hugging Face weights.
      hook_fns: The hook function(s) to apply to each individual weight.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  """
  all_expert_tensors = []
  # Outer loop iterates through experts
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    # Inner loop iterates through layers for the current expert
    for hf_key_single in layer_keys_for_expert:
      if hf_key_single not in hf_state_dict:
        raise ValueError(f"HuggingFace key {hf_key_single} not found in state_dict.")
      hf_tensor_numpy = hf_state_dict[hf_key_single].to(torch.float16).numpy()
      # For this case, the hook function does not require the target_shape.
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, None, hook_fns)
      layer_tensors_for_expert.append(processed_hf_tensor)

    print("hi")
    # First, stack all layers for the current expert. This creates the 'layer' axis.
    stacked_expert_tensor = np.stack(layer_tensors_for_expert, axis=0)
    all_expert_tensors.append(stacked_expert_tensor)

  # Second, stack all the expert tensors. This creates the 'expert' axis.
  return np.stack(all_expert_tensors, axis=0)


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str], hf_state_dict: Dict[str, np.ndarray], hook_fns: Any, target_shape: tuple, config
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along a single axis.

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      hf_state_dict: The dictionary of loaded Hugging Face weights.
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  """
  tensors_to_stack = []
  # Heuristic to determine if we are stacking layers or experts.
  # If the number of items to stack equals the number of layers, it's a standard
  # scanned layer, and we use the configured param_scan_axis. Otherwise, it's
  # an unscanned MoE layer, and we stack along the expert axis (0).
  # TODO(shuningjin)
  axis_to_stack = config.param_scan_axis if len(hf_source_keys) == config.base_num_decoder_layers else 0
  axis_to_stack = config.param_scan_axis

  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # We calculate it by removing the stacking dimension from the final target shape.
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  for hf_key_single in hf_source_keys:
    if hf_key_single not in hf_state_dict:
      raise ValueError(f"HuggingFace key {hf_key_single} not found in state_dict.")
    hf_tensor_numpy = hf_state_dict[hf_key_single].to(torch.float16).numpy()
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
    tensors_to_stack.append(processed_hf_tensor)

  # Stack all processed tensors along the determined axis.
  return np.stack(tensors_to_stack, axis=axis_to_stack)


def get_abstract_param(model, config):
  key = jax.random.PRNGKey(0)
  # input_shape = (1, 1)  # (batch, length)
  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  abstract_vars = jax.eval_shape(
      model.init,
      {"params": key, "dropout": key, "aqt": key},
      jnp.ones(input_shape, dtype=jnp.int32),
      jnp.ones(input_shape, dtype=jnp.int32),
      encoder_images=None,
  )
  return abstract_vars


def main(argv: Sequence[str], local_argv: argparse.Namespace) -> None:
  # jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # # Check if the user is using an Instruct version. If so, use the base model architecture
  # for i, arg in enumerate(argv):
  #   if arg.startswith("model_name="):
  #     model_name_arg = argv[i].split("=")[1]
  #     model_name_original = model_name_arg
  #     if "-Instruct" in model_name_arg:
  #       max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
  #       model_name_arg = model_name_arg.replace("-Instruct", "")
  #       argv[i] = f"model_name={model_name_arg}"
  #     break

  config = pyconfig.initialize(argv)
  # # check the supported model ids
  # if model_name_original not in HF_IDS:
  #   raise ValueError(f"Unsupported model name: {model_name_original}. Supported models are: {list(HF_IDS.keys())}")

  if local_argv.hf_model_path:
    # use local model
    model_id = local_argv.hf_model_path
  # else:
  #   model_id = HF_IDS[model_name_original]

  max_utils.print_system_information()
  if not config.base_output_directory:
    output_directory = f"/tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  # # Setup JAX distributed system and mesh
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  hf_token = config.hf_access_token
  # # Load HuggingFace model, config, and state_dict
  # max_logging.log(f"Loading HuggingFace model: {model_id}...")
  # start = time.time()
  hf_config_obj = AutoConfig.from_pretrained(model_id, token=hf_token)
  # hf_model = AutoModelForCausalLM.from_pretrained(
  #     model_id,
  #     token=hf_token,
  #     # low_cpu_mem_usage=True,
  #     device_map="cpu",
  #     dtype=torch.float16,
  #     cache_dir="/home/shuningjin/deepseek3-671b/hf-671b-bf16-cache",
  # )

  # # local_files_only=True
  # # dtype=torch.bfloat16

  # hf_state_dict_numpy = hf_model.state_dict()
  # for k, v in tqdm(hf_state_dict_numpy.items(), total=len(hf_state_dict_numpy)):
  #   # print(v.dtype)
  #   hf_state_dict_numpy[k] = v.numpy()
  #   # hf_state_dict_numpy[k] = v.to(torch.float32).numpy().astype(CAST_DTYPE)
  #   hf_state_dict_numpy[k] = v.to(torch.float16).numpy()
  #   # print(hf_state_dict_numpy[k].dtype)
  # del hf_model
  # max_logging.log(f"HuggingFace model loaded and converted to NumPy. {time.time() - start: .2f} second")

  # logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log(f"Create checkpoint manager...")
  start = time.time()
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      output_directory,
      enable_checkpointing=True,
      use_async=False,  # Synchronous saving for simplicity in conversion script
      save_interval_steps=1,  # Save at step 0
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  max_logging.log(f"Finish creating checkpoint manager. {time.time() - start: .2f} second")

  def init1():
    # Initialize MaxText model, optimizer, and abstract state
    max_logging.log(f"MaxText abstract model and state initializing...")
    start = time.time()

    # Initialize MaxText model, optimizer, and abstract state
    rng = jax.random.PRNGKey(config.init_weights_seed)
    quant = quantizations.configure_quantization(config)
    maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
    tx = optimizers.get_optimizer(config, learning_rate_schedule)

    abstract_state, _, _, _ = maxtext_utils.setup_training_state(
        maxtext_model_flax, None, tx, config, rng, mesh, checkpoint_manager
    )
    abstract_params_tree = abstract_state.params["params"]
    abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(abstract_params_tree)

    max_logging.log(f"Elapse. MaxText abstract model and state initialized. {time.time() - start: .2f} second")
    return abstract_params_flat, abstract_params_treedef

  def init2():
    max_logging.log(f"MaxText abstract model and state initializing...")
    start = time.time()
    quant = quantizations.configure_quantization(config)
    maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    abstract_params_tree = get_abstract_param(maxtext_model_flax, config)["params"]

    abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(
        abstract_params_tree,  # is_leaf=lambda x: not isinstance(x, dict)
    )
    print(abstract_params_flat)
    max_logging.log(f"Elapse. MaxText abstract model and state initialized. {time.time() - start: .2f} second")
    return abstract_params_flat, abstract_params_treedef

  print("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  abstract_params_flat, abstract_params_treedef = init1()
  print("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  sys.exit(1)

  ckpt_paths = sorted(pathlib.Path(model_id).glob("[!.]*.safetensors"))
  hf_state_dict_numpy = {}
  max_logging.log(f"Loading {len(ckpt_paths)} checkpoint ...")
  for i, ckpt_path in tqdm(enumerate(ckpt_paths), total=len(ckpt_paths)):
    # max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        # parts = key.split(".")
        # layer = int(parts[2]) if "layers" in key else 0
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if is_key_allowed(key, MTP_KEYS_TO_SKIP):
          # mapped_key = hf_to_maxtext_mapping(
          #     layer, num_experts, first_num_dense_layers, base_num_decoder_layers, has_mtp
          # ).get(key)
          hf_state_dict_numpy[key] = f.get_tensor(key)  # .to(torch.float16).numpy()

  logging.info("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # Get parameter mappings and hooks
  # example of param mapping (gemma2, maxtext:huggingface):
  # "params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale":
  #   f"model.layers.{global_layer_idx}.input_layernorm.weight",

  model_key = config.model_name
  param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_obj.to_dict(), config.scan_layers)

  # Example of Hook FN mapping, to perform reshape:
  # f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
  hook_fn_map_mt = HOOK_FNS[model_key](hf_config_obj.to_dict(), config.scan_layers, saving_to_hf=False)
  max_logging.log("Parameter mappings and hooks obtained.")

  # Transform weights
  max_logging.log("Starting weight transformation...")
  start = time.time()
  final_mt_weights = []

  for path_tuple, abstract_leaf_value in tqdm(abstract_params_flat, total=len(abstract_params_flat)):

    # key_parts = [k.key for k in path_tuple[:-1]]
    key_parts = [k.key for k in path_tuple]
    mt_param_key = "params-" + "-".join(key_parts)
    mt_target_shape_final = abstract_leaf_value.shape
    print(mt_param_key)
    print(mt_target_shape_final)

    # key_parts = [k.key for k in path_tuple]
    # mt_param_key = "params-" + "-".join(key_parts)
    # mt_target_shape_final = abstract_leaf_value.shape

    hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key)
    if hf_source_keys_or_key is None:
      raise ValueError(f"MaxText parameter {mt_param_key} not found in mapping.")

    hook_fn_list_or_fn = hook_fn_map_mt.get(mt_param_key)
    final_mt_tensor_numpy = None

    if not isinstance(hf_source_keys_or_key, list):
      # Case 1: Simple 1-to-1 mapping
      hf_key_single = hf_source_keys_or_key
      if hf_key_single not in hf_state_dict_numpy:
        raise ValueError(f"HuggingFace key {hf_key_single} not found in state_dict.")
      hf_tensor_numpy = hf_state_dict_numpy[hf_key_single].to(torch.float16).numpy()
      final_mt_tensor_numpy = apply_hook_fns(hf_tensor_numpy, mt_target_shape_final, hook_fn_list_or_fn)
    else:
      # It's a stacked parameter, so dispatch to a helper function.
      is_multi_axis_stacked = isinstance(hf_source_keys_or_key[0], list)

      if is_multi_axis_stacked:
        # Case 2: Multi-Axis Stacked (Scanned MoE)
        final_mt_tensor_numpy = _build_multi_axis_stacked_tensor(
            hf_source_keys_or_key, hf_state_dict_numpy, hook_fn_list_or_fn
        )
      else:
        # Case 3: Single-Axis Stacked (Standard Scanned or Unscanned MoE)
        final_mt_tensor_numpy = _build_single_axis_stacked_tensor(
            hf_source_keys_or_key, hf_state_dict_numpy, hook_fn_list_or_fn, mt_target_shape_final, config
        )

    if final_mt_tensor_numpy.shape != mt_target_shape_final:
      raise ValueError(
          f"Shape mismatch for {mt_param_key}: "
          f"Expected {mt_target_shape_final}, got {final_mt_tensor_numpy.shape} "
          f"from HF key(s) {hf_source_keys_or_key} after hooks."
      )
    final_mt_weights.append(final_mt_tensor_numpy)

  del abstract_params_flat, hf_state_dict_numpy
  max_logging.log(f"Elapse: Weight transformation complete. {time.time() - start: .2f} second")

  # Create final MaxText parameters tree
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  print(abstract_params_treedef)
  del final_mt_weights, abstract_params_treedef

  # Create TrainState for saving.
  max_logging.log("Saving checkpoint...")
  start = time.time()
  final_params_for_state = {"params": jax_weights}
  final_save_state = train_state.TrainState(step=0, apply_fn=None, params=final_params_for_state, tx=None, opt_state={})
  # print(final_params_for_state)
  del final_params_for_state

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, 0, final_save_state):
      max_logging.log("saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

  max_logging.log(f"Elapse. Conversion complete. Checkpoint saved to {output_directory}. {time.time() - start: .2f} second")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--hf_model_path", type=str, required=False, default="")
  local_args, _ = parser.parse_known_args()
  # Remove local args for pyconfig
  model_args = [s for s in sys.argv if not s.startswith("--hf_model_path")]
  main(model_args, local_args)
