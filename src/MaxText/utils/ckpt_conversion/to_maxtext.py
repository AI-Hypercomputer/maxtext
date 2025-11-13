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
  lazy_load: (bool) If True, uses an on-demand loading strategy to minimize RAM
             usage during conversion. Recommended if, 2 * model_size (GB) >= system RAM
             Defaults to False.

Environment Variables:
  HF_AUTH_TOKEN: (Required) HuggingFace authentication token, needed to
                 download models from HuggingFace Hub.

Example Usage:
  To convert a gemma2-2b model and save it to a specific directory:

    /usr/bin/time -v python src/MaxText/utils/ckpt_conversion/to_maxtext.py \
    MaxText/configs/base.yml model_name="gemma2-2b" \
    base_output_directory="/path/to/your/output/directory" \
    hf_access_token=$HF_TOKEN hardware=cpu skip_jax_distributed_system=True \
    scan_layers=False

  For models with scanned layers (e.g., some custom architectures), you might
  need to set scan_layers=True and param_scan_axis accordingly.

  To convert a 70B model with minimal RAM usage:

   /usr/bin/time -v python src/MaxText/utils/ckpt_conversion/to_maxtext.py \
    MaxText/configs/base.yml model_name="meta-llama/Llama-3.1-70B" \
    base_output_directory="gs://my-bucket/maxtext-checkpoints" \
    hf_access_token=$HF_TOKEN hardware=cpu skip_jax_distributed_system=True \
    --lazy_load_tensors=True
"""

import argparse
import os
import sys
import json
import threading
from functools import partial
from typing import Sequence, List, Any, Callable
from MaxText.inference_utils import str2bool
import numpy as np
import jax
import psutil
from flax.training import train_state
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

from orbax.checkpoint import type_handlers
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


def print_ram_usage(stage=""):
  memory = psutil.virtual_memory()
  max_logging.log(
      f"[{stage}] RAM Usage: {memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB ({memory.percent:.1f}%)"
  )


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


class LazyHFLoader:
  """
  Loads Hugging Face weights on-demand to minimize RAM usage.

  This class is the core of the "lazy loading" feature. Instead of loading the
  entire model into memory at once, it reads the model's index file (e.g.,
  `model.safetensors.index.json`) to understand the mapping between tensor names
  and the shard files they belong to.

  When a specific tensor is requested via `get_tensor`, this class:
  1. Identifies the correct shard file.
  2. Downloads the shard file if not already cached by `huggingface_hub`.
  3. Opens the shard and extracts *only* the requested tensor into memory.

  This approach is highly memory-efficient, especially for `safetensors`, as
  it avoids loading entire multi-gigabyte shard files when only a small piece
  is needed. A threading lock (`_ram_lock`) is used to ensure that memory-intensive
  file-opening operations are serialized to prevent RAM spikes, while downloads
  can still occur in parallel.
  """

  def __init__(self, model_id, token):
    self.model_id = model_id
    self.token = token
    self.shard_map = {}
    self.current_shard_name = None
    self.current_shard_content = {}
    # Use a lock to serialize heavy RAM operations, but NOT downloads
    self._ram_lock = threading.Lock()
    self._initialize_index()

  def __getstate__(self):
    """Allows pickling/copying by excluding the non-pickleable lock."""
    state = self.__dict__.copy()
    del state["_ram_lock"]
    return state

  def __setstate__(self, state):
    """Restores state after pickling/copying and recreates a new lock."""
    self.__dict__.update(state)
    self._ram_lock = threading.Lock()

  def _initialize_index(self):
    """Fetches and parses the Hugging Face model index file to build a shard map."""
    files = list_repo_files(self.model_id, token=self.token)

    # Prefer safetensors
    if "model.safetensors.index.json" in files:
      index_file = "model.safetensors.index.json"
    elif "model.safetensors" in files:
      # Single file case
      self.shard_map = {None: "model.safetensors"}
      return
    else:
      raise ValueError("Could not find recognized model weights (safetensors) in HF repo.")

    # Download and parse the index
    max_logging.log(f"Loading index file: {index_file}")
    index_path = hf_hub_download(repo_id=self.model_id, filename=index_file, token=self.token)
    with open(index_path, "r", encoding="utf-8") as f:
      index_data = json.load(f)
    self.shard_map = index_data["weight_map"]

  def get_tensor(self, key: str) -> np.ndarray:
    """
    Retrieves a specific tensor by name, lazily loading its shard if necessary.

    This is the main entry point for accessing model weights. It determines
    which shard file contains the tensor, ensures it's downloaded, and then
    reads the tensor data.

    For safetensors, this is extremely efficient as it memory-maps the file
    and reads only the required tensor's data from disk.
    """
    # Handle single-file models (shard map key might be None or we just know the filename)
    shard_name = self.shard_map.get(key)
    if shard_name is None and None in self.shard_map:
      shard_name = self.shard_map[None]
    elif shard_name is None:
      # Fallback: sometimes keys in index don't perfectly match requested keys if there are prefix mismatches.
      # You might need advanced fuzzy matching here if you encounter errors.
      raise ValueError(f"Key {key} not found in HF checkpoint index.")

    # STEP 1: Download outside the lock.
    # multiple threads can download different shards at the same time.
    local_path = hf_hub_download(repo_id=self.model_id, filename=shard_name, token=self.token)

    # STEP 2: Lock ONLY the reading into RAM.
    # This prevents multiple threads from simultaneously allocating large chunks of RAM.
    with self._ram_lock:
      with safe_open(local_path, framework="np", device="cpu") as f:
        return f.get_tensor(key)


class LazyTensor:
  """
  A proxy object that looks like a NumPy array but delays actual loading
  and transformation until __array__ is called (e.g., by Orbax during save).
  """

  def __init__(self, load_fn: Callable[[], np.ndarray], shape: tuple, dtype, name: str = "unknown"):
    self._load_fn = load_fn
    self.shape = shape
    self.dtype = np.dtype(dtype)
    self.ndim = len(shape)
    self.name = name

  @property
  def size(self):
    """Total number of elements in the tensor."""
    return np.prod(self.shape)

  @property
  def nbytes(self):
    """Return estimated nbytes so Orbax doesn't need to load the real array to find out."""
    return self.size * self.dtype.itemsize

  @property
  def itemsize(self):
    return self.dtype.itemsize

  def __array__(self, dtype=None):
    """
    Materializes the tensor data.

    When this method is invoked, it finally calls the `_load_fn` that was
    provided during initialization. This function executes the actual loading
    and transformation of the tensor from the Hugging Face checkpoint. The
    resulting NumPy array is then returned to the caller.
    """
    # This method is called just-in-time by Orbax when saving this specific leaf.
    try:
      arr = self._load_fn()
    except Exception as e:
      max_logging.log(f"FATAL ERROR: Failed to load tensor '{self.name}' (shape {self.shape}). Error: {e}")
      # Re-raise the original exception so it doesn't get masked by "object __array__..."
      raise

    # Ensure it's a standard numpy array (converts JAX arrays if necessary)
    if not isinstance(arr, np.ndarray):
      arr = np.array(arr)

    if dtype is not None and arr.dtype != dtype:
      return arr.astype(dtype)
    return arr

  def __repr__(self):
    return f"LazyTensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class LazyTensorHandler(type_handlers.NumpyHandler):
  """Custom Orbax handler for LazyTensor to avoid typestr collision with np.ndarray."""

  def typestr(self):
    return "LazyTensor"


# Register LazyTensor with the custom handler.
# It's safe to register this globally even if eager loading is used.
type_handlers.register_type_handler(LazyTensor, LazyTensorHandler())


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]], tensor_getter_fn: Callable[[str], np.ndarray], hook_fns: Any
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles the complex case for scanned MoE layers, producing a tensor
  with the shape (num_experts, num_layers, ...).

  Args:
      hf_source_keys: A nested (2D) list of Hugging Face parameter names.
                      Outer list iterates experts, inner list iterates layers.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
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
      hf_tensor_numpy = tensor_getter_fn(hf_key_single)
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, None, hook_fns)
      layer_tensors_for_expert.append(processed_hf_tensor)
    all_expert_tensors.append(np.stack(layer_tensors_for_expert, axis=0))
  return np.stack(all_expert_tensors, axis=0)


def _build_single_axis_stacked_tensor(
    hf_source_keys: List[str],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along a single axis.

  This function handles both standard scanned layers (e.g., attention) and
  unscanned MoE layers (which are stacked along the expert axis).

  Args:
      hf_source_keys: A 1D list of Hugging Face parameter names.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
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
  axis_to_stack = config.param_scan_axis if len(hf_source_keys) == config.base_num_decoder_layers else 0

  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # We calculate it by removing the stacking dimension from the final target shape.
  mt_slice_shape_list = list(target_shape)
  del mt_slice_shape_list[axis_to_stack]
  mt_slice_shape = tuple(mt_slice_shape_list)

  for hf_key_single in hf_source_keys:
    hf_tensor_numpy = tensor_getter_fn(hf_key_single)
    processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
    tensors_to_stack.append(processed_hf_tensor)

  # Stack all processed tensors along the determined axis.
  return np.stack(tensors_to_stack, axis=axis_to_stack)


def _get_hf_model(model_id: str, token: str):
  """Loads the HuggingFace model based on model_id (Eager mode only)."""
  # Some models require special classes to import
  if model_id in ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]:
    from transformers import Qwen3OmniMoeForConditionalGeneration  # pylint: disable=import-outside-toplevel

    hf_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, token=token)
  else:
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
  return hf_model


def main(args: Sequence[str], test_args: Sequence[str]) -> None:
  # Check if the user is using an Instruct version. If so, use the base model architecture
  for i, arg in enumerate(args):
    if arg.startswith("model_name="):
      model_name_arg = args[i].split("=")[1]
      model_name_original = model_name_arg
      if "-Instruct" in model_name_arg:
        max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
        model_name_arg = model_name_arg.replace("-Instruct", "")
        args[i] = f"model_name={model_name_arg}"
      break

  config = pyconfig.initialize(args)
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

  use_lazy_load = test_args.lazy_load_tensors

  if use_lazy_load and config.use_multimodal:
    raise ValueError("lazy loading of HF tensors is not supported for multimodal models yet.")

  hf_state_dict_numpy = None
  hf_loader = None

  if use_lazy_load:
    max_logging.log(f"Lazy loading ENABLED. Initializing LazyHFLoader for: {model_id}...")
    hf_loader = LazyHFLoader(model_id, hf_token)
    hf_config_obj = AutoConfig.from_pretrained(model_id, token=hf_token)
    print_ram_usage("After LazyLoader init")
  else:
    max_logging.log(f"Lazy loading DISABLED. Loading full HuggingFace model: {model_id}...")
    hf_config_obj = AutoConfig.from_pretrained(model_id, token=hf_token)
    hf_model = _get_hf_model(model_id, token=hf_token)
    hf_state_dict_numpy = hf_model.state_dict()
    # Convert all to numpy immediately in eager mode
    for k, v in hf_state_dict_numpy.items():
      hf_state_dict_numpy[k] = v.numpy()
    del hf_model
    max_logging.log("HuggingFace model loaded and converted to NumPy.")
    print_ram_usage("After full HF model load")

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      output_directory,
      enable_checkpointing=True,
      use_async=False,  # Synchronous saving for simplicity in conversion script
      save_interval_steps=1,  # Save at step 0
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )

  max_logging.log("Initializing MaxText abstract model...")
  quant = quantizations.configure_quantization(config)
  maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  # Get abstract model structure (name, shape) without materializing the weights to save memory
  with maxtext_model_flax.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_params_tree = maxtext_utils.get_abstract_param(maxtext_model_flax, config)["params"]

  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_params_tree)
  # Standardize abstract tree for later unflattening
  abstract_params_tree = jax.tree.map(
      lambda _: 0,
      abstract_params_tree,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  abstract_params_treedef = jax.tree_util.tree_structure(abstract_params_tree)
  del abstract_params_tree

  max_logging.log("MaxText abstract model and state initialized.")

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

  max_logging.log("Starting weight transformation...")
  final_mt_weights = []

  # Define the appropriate tensor getter based on mode
  if use_lazy_load:
    tensor_getter = hf_loader.get_tensor
  else:

    def _eager_getter(key):
      if key not in hf_state_dict_numpy:
        raise ValueError(f"HuggingFace key {key} not found in state_dict.")
      return hf_state_dict_numpy[key]

    tensor_getter = _eager_getter

  for path_tuple, abstract_leaf_value in MemoryMonitorTqdm(
      abstract_params_flat, desc="Transforming weights", unit="param", leave=True, dynamic_ncols=True
  ):
    key_parts = [k.key for k in path_tuple if hasattr(k, "key")]
    mt_param_key = "params-" + "-".join(key_parts)
    mt_target_shape_final = abstract_leaf_value.shape

    hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key)

    if hf_source_keys_or_key is None:
      raise ValueError(f"MaxText parameter {mt_param_key} not found in mapping.")

    hook_fn = hook_fn_map_mt.get(mt_param_key)

    # Determine the loading function for this specific parameter
    load_fn = None
    if not isinstance(hf_source_keys_or_key, list):
      # Case 1: Simple 1-to-1 mapping
      def _loader(getter, key, shape, hook):
        return apply_hook_fns(getter(key), shape, hook)

      load_fn = partial(_loader, tensor_getter, hf_source_keys_or_key, mt_target_shape_final, hook_fn)
    else:
      # Stacked mapping
      if isinstance(hf_source_keys_or_key[0], list):
        # Case 2: Multi-Axis Stacked
        load_fn = partial(_build_multi_axis_stacked_tensor, hf_source_keys_or_key, tensor_getter, hook_fn)
      else:
        # Case 3: Single-Axis Stacked
        load_fn = partial(
            _build_single_axis_stacked_tensor,
            hf_source_keys_or_key,
            tensor_getter,
            hook_fn,
            mt_target_shape_final,
            config,
        )

    # Execute based on mode
    if use_lazy_load:
      # In lazy mode, we don't execute the loading/transformation function
      # immediately. Instead, we wrap it in a `LazyTensor` object. This
      # object acts as a placeholder that holds all the information needed
      # to load the tensor later (the `load_fn`, shape, dtype).
      # The actual data will only be loaded when Orbax calls `__array__`
      # on this object during the saving process.
      final_mt_weights.append(LazyTensor(load_fn, mt_target_shape_final, abstract_leaf_value.dtype, name=mt_param_key))
    else:
      # In eager mode, we execute the function immediately to get the
      # NumPy array and append it to our list of weights.
      final_mt_tensor_numpy = load_fn()
      if final_mt_tensor_numpy.shape != mt_target_shape_final:
        raise ValueError(
            f"Shape mismatch for {mt_param_key}: Expected {mt_target_shape_final}, got {final_mt_tensor_numpy.shape}"
        )
      final_mt_weights.append(final_mt_tensor_numpy)

  del abstract_params_flat, hf_state_dict_numpy
  max_logging.log("Weight transformation preparation complete.")
  print_ram_usage("Before creating full JAX tree")

  # Create final MaxText parameters tree
  jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
  del final_mt_weights, abstract_params_treedef

  # Create TrainState for saving.
  final_params_for_state = {"params": jax_weights}
  final_save_state = train_state.TrainState(step=0, apply_fn=None, params=final_params_for_state, tx=None, opt_state={})
  del final_params_for_state

  print_ram_usage("Before saving")
  if checkpoint_manager is not None:
    if use_lazy_load:
      max_logging.log("Starting checkpoint save (loading weights just-in-time)...")
    else:
      max_logging.log("Starting checkpoint save...")

    if save_checkpoint(checkpoint_manager, 0, final_save_state):
      max_logging.log("saved a checkpoint at step 0")

    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()

  print_ram_usage("Program Ends")
  max_logging.log(f"Conversion complete. Checkpoint saved to {output_directory}")


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--lazy_load_tensors",
      type=str2bool,
      required=False,
      default=False,
      help="Whether to use lazy loading of HF tensors.",
  )
  local_args, _ = parser.parse_known_args()
  model_args = sys.argv
  to_remove_args = ["--lazy_load_tensors"]
  for a in to_remove_args:
    model_args = [s for s in model_args if not s.startswith(a)]
  main(model_args, local_args)
