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
import gc
import tempfile
import multiprocessing as mp
from typing import Sequence, List, Dict, Any, Tuple
from tqdm import tqdm

import numpy as np
import jax
import psutil
from absl import app
from flax.training import train_state
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
from transformers import AutoConfig
import json
from huggingface_hub import snapshot_download
from safetensors import safe_open

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import models, quantizations
from MaxText.checkpointing import save_checkpoint
from MaxText.utils.ckpt_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.utils import apply_hook_fns, HF_IDS

jax.config.update("jax_platform_name", "cpu")


def transform_and_save_tensor(args: Tuple[int, str, Tuple, Any, str, Dict, str, str, Dict, bool]) -> str:
  """
  Worker function to transform a single tensor in a separate process.
  This function receives all necessary data, performs the conversion,
  saves the result to a temporary file, and returns the file path.
  """
  (
      i,
      mt_param_key,
      mt_target_shape_final,
      hf_source_keys_or_key,
      model_path,
      weight_map,
      temp_dir,
      model_name,
      hf_config_dict,
      scan_layers,
  ) = args

  # Recreate hook function map inside the worker to avoid pickling issues
  hook_fn_map_mt = HOOK_FNS[model_name](hf_config_dict, scan_layers, saving_to_hf=False)
  hook_fn_list_or_fn = hook_fn_map_mt.get(mt_param_key)

  # Each process needs to open its own file handles
  local_hf_safetensor_handles = {}

  def get_hf_tensor_local(hf_key: str) -> np.ndarray:
    if hf_key not in weight_map:
      raise ValueError(f"HuggingFace key '{hf_key}' not found in the model's weight map.")
    filename = weight_map[hf_key]
    if filename not in local_hf_safetensor_handles:
      local_hf_safetensor_handles[filename] = safe_open(os.path.join(model_path, filename), framework="pt", device="cpu")
    return local_hf_safetensor_handles[filename].get_tensor(hf_key).float().numpy()

  final_mt_tensor_numpy = None
  temp_path = os.path.join(temp_dir, f"tensor_{i}.npy")

  if not isinstance(hf_source_keys_or_key, list):
    hf_key_single = hf_source_keys_or_key
    hf_tensor_numpy = get_hf_tensor_local(hf_key_single)
    final_mt_tensor_numpy = apply_hook_fns(hf_tensor_numpy, mt_target_shape_final, hook_fn_list_or_fn)
    np.save(temp_path, final_mt_tensor_numpy)
  else:
    is_multi_axis_stacked = isinstance(hf_source_keys_or_key[0], list)
    if is_multi_axis_stacked:
      # Build the complex MoE tensor incrementally on disk
      _build_multi_axis_stacked_tensor(
          hf_source_keys_or_key, get_hf_tensor_local, hook_fn_list_or_fn, mt_target_shape_final, temp_path
      )
    else:
      # This is the common case for large stacked tensors.
      # We build it incrementally on disk and don't return a numpy array.
      _build_single_axis_stacked_tensor(
          hf_source_keys_or_key,
          get_hf_tensor_local,
          hook_fn_list_or_fn,
          mt_target_shape_final,
          pyconfig.Config(model_name=model_name, scan_layers=scan_layers),  # Pass a minimal config
          temp_path,
      )

  if final_mt_tensor_numpy is not None:
    if final_mt_tensor_numpy.shape != mt_target_shape_final:
      raise ValueError(
          f"Shape mismatch for {mt_param_key}: "
          f"Expected {mt_target_shape_final}, got {final_mt_tensor_numpy.shape} "
          f"from HF key(s) {hf_source_keys_or_key} after hooks."
      )
    del final_mt_tensor_numpy
    gc.collect()

  # The OS will close file handles when the process exits.
  # No explicit close is needed for safetensor handles.

  return temp_path


class MemoryMonitorTqdm(tqdm):
  """Custom tqdm class that displays memory usage in the progress bar."""

  def format_meter(
      self,
      n,
      total,
      elapsed,
      postfix=None,
      **extra_kwargs,
  ):
    """Override to add memory usage info to the postfix."""
    # Get memory info
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    memory_percent = memory.percent

    # Create memory postfix
    memory_info = f"RAM: {used_gb:.1f}/{total_gb:.1f}GB ({memory_percent:.1f}%)"

    # Add memory info to postfix
    if postfix:
      if isinstance(postfix, dict):
        postfix["memory"] = memory_info
      else:
        postfix = f"{postfix}, {memory_info}"
    else:
      postfix = memory_info

    return super().format_meter(n=n, total=total, elapsed=elapsed, postfix=postfix, **extra_kwargs)

def _get_hf_model(model_id: str, token: str):
  """Loads the HuggingFace model based on model_id."""
  # Some models require special classes to import
  if model_id in ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]:
    from transformers import Qwen3OmniMoeForConditionalGeneration  # pylint: disable=import-outside-toplevel

    hf_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, token=token)
  else:
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
  return hf_model

def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    get_tensor_fn: callable,
    hook_fns: Any,
    target_shape: tuple,
    temp_file_path: str,
) -> None:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      hf_state_dict: The dictionary of loaded Hugging Face weights.
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      temp_file_path: Path to a temporary file to store the intermediate tensor.
  """
  # Create a memory-mapped array on disk to hold the final tensor
  final_memmapped_tensor = np.memmap(temp_file_path, dtype=np.float32, mode="w+", shape=target_shape)

  # Outer loop iterates through experts
  for expert_idx, layer_keys_for_expert in enumerate(hf_source_keys):
    # Inner loop iterates through layers for the current expert
    for layer_idx, hf_key_single in enumerate(layer_keys_for_expert):
      hf_tensor_numpy = get_tensor_fn(hf_key_single)
      # For this case, the hook function does not require the target_shape.
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, None, hook_fns)

      # Write the processed tensor directly into its slice of the on-disk array
      final_memmapped_tensor[expert_idx, layer_idx] = processed_hf_tensor

  # Ensure all data is written to disk
  final_memmapped_tensor.flush()
  del final_memmapped_tensor
  gc.collect()


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    get_tensor_fn: callable,
    hook_fns: Any,
    target_shape: tuple,
    config,
    temp_file_path: str,
) -> None:
  """Builds a MaxText tensor by stacking HF weights along a single axis.

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      hf_state_dict: The dictionary of loaded Hugging Face weights.
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.
      temp_file_path: Path to a temporary file to store the intermediate tensor.
  """
  # Heuristic to determine if we are stacking layers or experts.
  axis_to_stack = config.param_scan_axis if len(hf_source_keys) == config.base_num_decoder_layers else 0

  # Create a memory-mapped array on disk to hold the final tensor
  final_memmapped_tensor = np.memmap(temp_file_path, dtype=np.float32, mode="w+", shape=target_shape)

  # Calculate the shape of an individual slice
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  for i, hf_key_single in enumerate(hf_source_keys):
    hf_tensor_numpy = get_tensor_fn(hf_key_single)
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
    # Create a slice object to write to the correct index in the memmapped array
    slice_obj = [slice(None)] * len(target_shape)
    slice_obj[axis_to_stack] = i
    final_memmapped_tensor[tuple(slice_obj)] = processed_hf_tensor

  # Ensure the data is written to disk
  final_memmapped_tensor.flush()
  del final_memmapped_tensor
  gc.collect()


def print_ram_usage(stage=""):
  """Helper function to print current RAM usage."""
  memory = psutil.virtual_memory()
  used_gb = memory.used / (1024**3)
  total_gb = memory.total / (1024**3)
  memory_percent = memory.percent
  max_logging.log(f"[{stage}] RAM Usage: {used_gb:.2f}/{total_gb:.2f} GB ({memory_percent:.1f}%)")


def main(argv: Sequence[str]) -> None:
  print_ram_usage("Script Start")
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # Check if the user is using an Instruct version. If so, use the base model architecture
  for i, arg in enumerate(argv):
    if arg.startswith("model_name="):
      model_name_arg = argv[i].split("=")[1]
      model_name_original = model_name_arg
      if "-Instruct" in model_name_arg:
        max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
        model_name_arg = model_name_arg.replace("-Instruct", "")
        argv[i] = f"model_name={model_name_arg}"
      break

  config = pyconfig.initialize(argv)
  # check the supported model ids
  if model_name_original not in HF_IDS:
    raise ValueError(f"Unsupported model name: {model_name_original}. Supported models are: {list(HF_IDS.keys())}")

  model_id = HF_IDS[model_name_original]
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
  hf_model = _get_hf_model(model_id, token=hf_token)
  hf_state_dict_numpy = hf_model.state_dict()
  for k, v in hf_state_dict_numpy.items():
    hf_state_dict_numpy[k] = v.numpy()
  del hf_model
  max_logging.log("HuggingFace model loaded and converted to NumPy.")

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      output_directory,
      enable_checkpointing=True,
      use_async=False,  # Synchronous saving for simplicity in conversion script
      save_interval_steps=1,  # Save at step 0
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )

  max_logging.log("MaxText abstract model and state initializing...") 
  quant = quantizations.configure_quantization(config)
  maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  # Get abstract model structure (name, shape) without materializing the weights to save memory
  with maxtext_model_flax.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_params_tree = maxtext_utils.get_abstract_param(maxtext_model_flax, config)["params"]
  # Get abstract_params_flat from abstract_params_tree
  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_params_tree)
  # Get abstract_params_treedef from transformed abstract_params_tree
  # Otherwise the param name has extra "value" after unflatten
  abstract_params_tree = jax.tree.map(
      lambda _: 0,
      abstract_params_tree,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  abstract_params_treedef = jax.tree_util.tree_structure(abstract_params_tree)
  del abstract_params_tree

  max_logging.log("MaxText abstract model and state initialized.")

  # Extract the metadata we need: the shapes and the tree structure
  abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(abstract_state.params["params"])
  param_shapes = {"params-" + "-".join([k.key for k in p]): v.shape for p, v in abstract_params_flat}

  # Immediately delete the large JAX objects to free memory before the main loop
  del abstract_state, maxtext_model_flax, tx, abstract_params_flat
  gc.collect()
  print_ram_usage("After JAX Metadata Extraction and Cleanup")

  # Transform weights
  max_logging.log("Starting weight transformation...")
  print_ram_usage("Before Weight Transformation Loop")
  temp_files = []
  temp_dir = tempfile.mkdtemp()

  # Prepare arguments for the worker processes
  tasks = []
  hf_config_dict = hf_config_obj.to_dict()
  for i, mt_param_key in enumerate(param_shapes):
    hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key)
    if hf_source_keys_or_key is None:
      raise ValueError(f"MaxText parameter {mt_param_key} not found in mapping.")

    tasks.append(
        (
            i,
            mt_param_key,
            param_shapes[mt_param_key],  # Pass the correct shape
            hf_source_keys_or_key,
            model_path,
            weight_map,
            temp_dir,
            config.model_name,
            hf_config_dict,
            config.scan_layers,
        )
    )

  # Use a process pool to transform tensors in parallel
  try:
    # Using 'spawn' start method for better cross-platform compatibility and safety
    ctx = mp.get_context("spawn")
    with ctx.Pool() as pool:
      with MemoryMonitorTqdm(
          total=len(tasks),
          desc="Transforming weights",
          unit="param",
      ) as pbar:
        for temp_path in pool.imap_unordered(transform_and_save_tensor, tasks):
          temp_files.append(temp_path)
          pbar.update(1)

    # Ensure temp_files are sorted correctly by tensor index
    temp_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    print_ram_usage("After Weight Transformation Loop")
    max_logging.log("All tensors converted and saved to temporary files.")
    max_logging.log("Reloading tensors to build final checkpoint.")

    final_mt_weights = []
    for temp_path in temp_files:
      final_mt_weights.append(np.load(temp_path, mmap_mode="r"))

    print_ram_usage("After Reloading Tensors")

  finally:
    # Clean up all temporary files and the directory
    for temp_path in temp_files:
      if os.path.exists(temp_path):
        os.remove(temp_path)
    if os.path.exists(temp_dir):
      os.rmdir(temp_dir)

  # Create final MaxText parameters tree
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  del final_mt_weights, abstract_params_treedef
  gc.collect()
  print_ram_usage("After Creating JAX Tree")

  # Create TrainState for saving.
  final_params_for_state = {"params": jax_weights}
  final_save_state = train_state.TrainState(step=0, apply_fn=None, params=final_params_for_state, tx=None, opt_state={})
  del final_params_for_state
  print_ram_usage("After Creating TrainState")

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, 0, final_save_state):
      max_logging.log("saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

  max_logging.log(f"Conversion complete. Checkpoint saved to {output_directory}")
  print_ram_usage("Script End")


if __name__ == "__main__":
  app.run(main)
