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

"""Converts a HuggingFace model checkpoint to a MaxText-compatible Orbax checkpoint.

This script supports three conversion modes:
1. Base: Converts a standard Hugging Face model to MaxText format.
2. Adapter: Converts a standalone Hugging Face LoRA adapter to MaxText PEFT format.
   (Requires `hf_lora_adapter_path` in config, and `load_parameters_path` should be empty)
3. Merged: Merges a Hugging Face LoRA adapter into the base weights during conversion.
   (Requires both `hf_lora_adapter_path` and `load_parameters_path` to be set/not empty)

Key Parameters (to be set in the config file or as command-line overrides):
  model_name: (Required) The name of the model to convert (e.g., "gemma3-4b").
              Must be a key in `maxtext.utils.globals.HF_IDS`.
  base_output_directory: (Optional) The directory where the converted checkpoint
                         will be saved. Can be a local or GCS path.
  load_parameters_path: (Optional) For Merged mode, path to the MaxText base weights.
  hf_lora_adapter_path: (Optional) For Adapter or Merged mode, path to the HF LoRA adapter.
  scan_layers: (bool) Whether the MaxText model was trained with scanned layers.
  --lazy_load_tensors: (bool) If True, uses an on-demand loading strategy to minimize RAM
             usage during conversion. Recommended for large models.
  --hf_model_path: (Optional) Specifies a local or remote directory containing the base HF weights.
  --save_dtype: (Optional) Data type of saved weights. Default to `bfloat16`.

Environment Variables:
  HF_AUTH_TOKEN: (Required) HuggingFace authentication token.

Example Usage:
  To merge a HF LoRA adapter into base weights and save as a MaxText checkpoint:

   python -m maxtext.checkpoint_conversion.to_maxtext \
    maxtext/configs/base.yml model_name="gemma3-4b" \
    load_parameters_path="gs://my-bucket/maxtext-base-weights" \
    hf_lora_adapter_path="my-user/my-lora-adapter" \
    base_output_directory="gs://my-bucket/maxtext-merged-output" \
    hf_access_token=${HF_TOKEN?} hardware=cpu skip_jax_distributed_system=True \
    scan_layers=True
"""

import argparse
from functools import partial
import json
import os
import sys
import threading
import time
from typing import Any, Callable, List, Sequence
import absl
import ml_dtypes
import flax.linen as nn
from huggingface_hub import hf_hub_download, list_repo_files
import jax
from maxtext.configs import pyconfig
from maxtext.configs.types import DType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.utils import MemoryMonitorTqdm, apply_hook_fns, load_hf_dict_from_transformers, load_hf_dict_from_safetensors, print_peak_memory, print_ram_usage, save_weights_to_checkpoint, validate_and_filter_param_map_keys
from maxtext.inference.inference_utils import str2bool
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import max_logging, max_utils, maxtext_utils
from maxtext.utils.globals import HF_IDS
import numpy as np
from orbax.checkpoint import type_handlers
from safetensors import safe_open

try:
  import torch
except ImportError:
  torch = None


absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


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

  def __init__(self, model_id, token, revision=None):
    self.model_id = model_id
    self.token = token
    self.revision = revision
    # Whether loads from local directory
    self.is_local = os.path.isdir(self.model_id)
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
    if self.is_local:
      files = os.listdir(self.model_id)
    else:
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
    if self.is_local:
      index_path = os.path.join(self.model_id, index_file)
    else:
      index_path = hf_hub_download(
          repo_id=self.model_id,
          filename=index_file,
          token=self.token,
          revision=self.revision,
      )
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

    if self.is_local:
      local_path = os.path.join(self.model_id, shard_name)
    else:
      # STEP 1: Download outside the lock.
      # multiple threads can download different shards at the same time.
      local_path = hf_hub_download(
          repo_id=self.model_id,
          filename=shard_name,
          token=self.token,
          revision=self.revision,
      )

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

  def __init__(
      self,
      load_fn: Callable[[], np.ndarray],
      shape: tuple,
      dtype,
      name: str = "unknown",
  ):
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

    if not isinstance(self.shape, list) and arr.shape != self.shape:
      raise ValueError(f"Shape mismatch for tensor '{self.name}'. Expected {self.shape}, but got {arr.shape}.")

    # Ensure it's a standard numpy array (converts JAX arrays if necessary)
    if not isinstance(arr, np.ndarray):
      arr = np.array(arr)

    if dtype is not None and arr.dtype != dtype:
      return arr.astype(dtype)
    return arr

  def __repr__(self):
    return f"LazyTensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"


class LazyTensorHandler(type_handlers.NumpyHandler):
  """
  Custom Orbax handler for LazyTensor.

  It masquerades as a standard NumpyHandler so that the resulting checkpoint
  has the standard 'array_metadatas' structure and can be loaded by
  standard MaxText instances.
  """

  async def serialize(self, value, *args, **kwargs):
    # MATERIALIZE: Trigger the lazy load (__array__) explicitly before saving.
    # This ensures the parent NumpyHandler receives a real np.ndarray.
    if hasattr(value, "__array__"):
      value = np.array(value)

    return await super().serialize(value, *args, **kwargs)


# Register LazyTensor with the custom handler.
# It's safe to register this globally even if eager loading is used.
type_handlers.register_type_handler(LazyTensor, LazyTensorHandler(), override=True)


def get_maxtext_model_info(config):
  """Initializes the abstract MaxText model and returns parameter mapping information.

  Args:
    config: The MaxText configuration object.

  Returns:
    maxtext_abstract_dict: A dictionary mapping MaxText parameter keys to a tuple
      (index, target_shape), where 'index' is the position of the parameter in the
      flattened parameter list.
    abstract_params_treedef: The tree structure definition of the abstract model parameters.
  """
  # Setup JAX distributed system and mesh
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  max_logging.log("Initializing MaxText abstract model...")
  quant = quantizations.configure_quantization(config)
  maxtext_model_flax = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  # Get abstract model structure (name, shape) without materializing the weights to save memory
  abstract_params_tree = maxtext_utils.get_abstract_param(maxtext_model_flax, config)["params"]

  abstract_params_flat, abstract_params_treedef = jax.tree_util.tree_flatten_with_path(
      abstract_params_tree, is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned)
  )

  max_logging.log("MaxText abstract model and state initialized.")

  # preprocess state
  maxtext_abstract_dict = {}
  for mt_target_idx, (path_tuple, abstract_leaf_value) in enumerate(abstract_params_flat):
    key_parts = []
    for k in path_tuple:
      # JAX path components can be DictKey(key), GetItemKey(key), or SequenceKey(idx).
      # We prefer string keys. If we see an integer or digit-string index, we assume it's
      # a layer/block index and join it with the previous part using '_', matching
      # MaxText's Linen-style naming convention (e.g., layers_0).
      val = getattr(k, "key", getattr(k, "idx", None))
      if val is None:
        val = str(k)

      val_str = str(val)
      if (isinstance(val, int) or val_str.isdigit()) and key_parts:
        key_parts[-1] = f"{key_parts[-1]}_{val_str}"
      else:
        key_parts.append(val_str)

    mt_param_key = "params-" + "-".join(key_parts)
    if isinstance(abstract_leaf_value, nn.LogicallyPartitioned):
      mt_target_shape = abstract_leaf_value.value.shape
    else:
      mt_target_shape = abstract_leaf_value.shape
    maxtext_abstract_dict[mt_param_key] = (mt_target_idx, mt_target_shape)

  return maxtext_abstract_dict, abstract_params_treedef


def _build_multi_axis_stacked_tensor(
    hf_source_keys: List[List[str]],
    tensor_getter_fn: Callable[[str], np.ndarray],
    hook_fns: Any,
    target_shape: tuple,
    config,
) -> np.ndarray:
  """Builds a MaxText tensor by stacking HF weights along two axes (experts and layers).

  This function handles the complex case for scanned MoE layers, producing a tensor
  with the shape (num_experts, num_layers, ...).

  Args:
      hf_source_keys: A nested (2D) list of Hugging Face parameter names.
                      Outer list iterates experts, inner list iterates layers.
      tensor_getter_fn: A callable that takes a HF key and returns the tensor (as numpy array).
      hook_fns: The hook function(s) to apply to each individual weight.
      target_shape: The final shape of the target MaxText tensor.
      config: The MaxText pyconfig object.

  Returns:
      The final, assembled NumPy array for the MaxText parameter.
  """
  all_expert_tensors = []
  # The hook function needs the shape of an individual slice, not the full stacked tensor.
  # For multi-axis stacking (experts, layers, ...), the slice shape is target_shape[2:]
  mt_slice_shape = target_shape[2:]

  # Outer loop iterates through experts
  for layer_keys_for_expert in hf_source_keys:
    layer_tensors_for_expert = []
    # Inner loop iterates through layers for the current expert
    for hf_key_single in layer_keys_for_expert:
      hf_tensor_numpy = tensor_getter_fn(hf_key_single)
      processed_hf_tensor = apply_hook_fns(hf_tensor_numpy, mt_slice_shape, hook_fns)
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

  if config.scan_layers:
    # If it's a standard scanned layer, we use the configured param_scan_axis.
    axis_to_stack = config.param_scan_axis
  else:
    # Otherwise, if an unscanned MoE layer, and we stack along the expert axis (0).
    axis_to_stack = 0

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


def _get_hf_loading_function(hf_source_keys_or_key, tensor_getter, hook_fn, mt_target_shape_or_shapes, config):
  """Determine the loading function for HF keys.
  HF keys can take four forms:
    Case 1: Unscanned (single string)
    Case 2: Scanned (list of strings)
    Case 3: Unscanned with expert stacking (list of strings)
    Case 4: Scanned with expert stacking (nested list of strings)
  """
  load_fn = None
  if not isinstance(hf_source_keys_or_key, list):
    # Case 1: Single hf key (str)
    def _loader(getter, key, shape, hook):
      return apply_hook_fns(getter(key), shape, hook)

    load_fn = partial(
        _loader,
        tensor_getter,
        hf_source_keys_or_key,
        mt_target_shape_or_shapes,
        hook_fn,
    )
  # Stacked mapping
  elif not isinstance(hf_source_keys_or_key[0], list):
    # Case 2 or 3: Single-Axis Stacked hf keys (un-nested list)
    load_fn = partial(
        _build_single_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  else:
    # isinstance(hf_source_keys_or_key[0], list)
    # Case 4: Multi-Axis Stacked hf keys (nested list)
    load_fn = partial(
        _build_multi_axis_stacked_tensor,
        hf_source_keys_or_key,
        tensor_getter,
        hook_fn,
        mt_target_shape_or_shapes,
        config,
    )
  return load_fn


def _get_maxtext_indices_and_shapes(mt_param_key_or_keys, maxtext_abstract_dict):
  """Resolves MaxText key(s) to target indices and shapes.

  The index is the parameter's order in `maxtext_abstract_dict.keys()`.
  This function handles two forms of MaxText keys:
  - `atomic_mt_key`: A single string representing one MaxText parameter that map to HF parameter(s).
  - `composite_mt_key`: A tuple of strings for multiple MaxText parameters that map to HF parameter(s).
  """
  is_composite_mt_key = isinstance(mt_param_key_or_keys, tuple)
  # atomic_mt_key
  if not is_composite_mt_key:
    mt_target_idx, mt_target_shape = maxtext_abstract_dict[mt_param_key_or_keys]
    return mt_target_idx, mt_target_shape
  # composite_mt_key
  mt_target_indices, mt_target_shapes = [], []
  for mt_param_key in mt_param_key_or_keys:
    mt_target_idx, mt_target_shape = maxtext_abstract_dict[mt_param_key]
    mt_target_indices.append(mt_target_idx)
    mt_target_shapes.append(mt_target_shape)
  return mt_target_indices, mt_target_shapes


def _get_maxtext_weight(
    load_fn,
    mt_target_idx_or_indices,
    mt_target_shape_or_shapes,
    mt_param_key_or_keys,
    final_mt_weights,
    save_dtype,
    use_lazy_load,
):
  """Loads Hugging Face parameters and converts them to MaxText parameters.

  This function handles loading based on tensor mode (eager or lazy) and
  processes MaxText keys, which can be `atomic_mt_key` or `composite_mt_key`.
  """
  is_composite_mt_key = isinstance(mt_param_key_or_keys, tuple)
  if not use_lazy_load:
    # Case 1: Eager mode
    # In eager mode, we execute the function immediately to get the
    # NumPy array and append it to our list of weights.
    final_mt_tensor_numpy = load_fn()
    if not is_composite_mt_key:
      # Case 1.1: Eager mode, `atomic_mt_key`
      final_mt_weights[mt_target_idx_or_indices] = final_mt_tensor_numpy
      if final_mt_tensor_numpy.shape != mt_target_shape_or_shapes:
        raise ValueError(
            f"Shape mismatch for {mt_param_key_or_keys}: Expected {mt_target_shape_or_shapes}, "
            f"got {final_mt_tensor_numpy.shape}"
        )
    else:
      # Case 1.2: Eager mode, `composite_mt_key`
      # The hook returns a tensor that can be split in last dim.
      # In eager mode, we can just split the materialized tensor.
      for i, mt_target_idx in enumerate(mt_target_idx_or_indices):
        final_mt_weights[mt_target_idx] = final_mt_tensor_numpy[..., i]
        if final_mt_weights[mt_target_idx].shape != mt_target_shape_or_shapes[i]:
          raise ValueError(
              f"Shape mismatch for {mt_param_key_or_keys[i]}: Expect {mt_target_shape_or_shapes[i]}, "
              f"got {final_mt_weights[mt_target_idx].shape}"
          )
  else:
    # Case 2: Lazy mode
    # In lazy mode, we don't execute the loading/transformation function
    # immediately. Instead, we wrap it in a `LazyTensor` object. This
    # object acts as a placeholder that holds all the information needed
    # to load the tensor later (the `load_fn`, shape, dtype).
    # The actual data will only be loaded when Orbax calls `__array__`
    # on this object during the saving process.
    final_mt_tensor_numpy = LazyTensor(
        load_fn,
        mt_target_shape_or_shapes,
        save_dtype,
        name=mt_param_key_or_keys,
    )
    if not is_composite_mt_key:
      # Case 2.1: Lazy mode, `atomic_mt_key`
      final_mt_weights[mt_target_idx_or_indices] = final_mt_tensor_numpy
    else:
      # Case 2.2: Lazy mode, `composite_mt_key`
      # For a composite key, the hook returns a tensor that can be split in last dim.
      # For lazy loading, we can't split the tensor until it's loaded.
      # We create multiple LazyTensors, each responsible for loading the
      # full source tensor but then slicing its piece. Parent HF tensor is loaded repeatedly.
      for i, mt_target_idx in enumerate(mt_target_idx_or_indices):

        def _slicing_loader(base_loader, slice_idx):
          return np.array(base_loader)[..., slice_idx]

        # Each LazyTensor gets a new load_fn that wraps the original and applies the slice.
        slicing_load_fn = partial(_slicing_loader, final_mt_tensor_numpy, i)
        final_mt_weights[mt_target_idx] = LazyTensor(
            slicing_load_fn,
            mt_target_shape_or_shapes[i],
            save_dtype,
            name=mt_param_key_or_keys[i],
        )


def convert_hf_lora_key_to_maxtext(hf_key: str, param_mapping: dict) -> tuple[str | None, int | None]:
  """Convert HF LoRA key to MaxText parameter path and optional layer index."""
  hf_param_key = hf_key.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
  hf_param_key = hf_param_key.replace(".lora_A", "").replace(".lora_B", "")

  if hf_param_key.startswith("base_model.model."):
    hf_param_key = hf_param_key[len("base_model.model.") :]

  if hf_param_key.startswith("language_model.model."):
    hf_param_key = "model.language_model." + hf_param_key[len("language_model.model.") :]

  for mt_key, hf_keys in param_mapping.items():
    if isinstance(hf_keys, str):
      if hf_keys == hf_param_key:
        return mt_key, None
      continue

    if not hf_keys:
      continue

    if isinstance(hf_keys[0], list):
      for i, sub_list in enumerate(hf_keys):
        for j, hf_k in enumerate(sub_list):
          if hf_k == hf_param_key:
            return mt_key, (i, j)
    else:
      for i, hf_k in enumerate(hf_keys):
        if hf_k == hf_param_key:
          return mt_key, i

  return None, None


def _process_and_stack_weights(
    indexed_weights: dict[str, Any],
    is_scanned: bool,
    num_layers: int,
    axis_to_stack: int,
    target_dtype: np.dtype,
    mt_key: str,
    suffix: str,
    config: Any,
) -> np.ndarray:
  """Transposes and optionally stacks weights across layers."""
  # Llama 3.1 models require a specific layout transformation for their RoPE embeddings
  needs_llama31_rope_shuffle = config.rope_type == "llama3.1" or "llama3.1" in config.model_name.lower()
  is_2d_indexed = any(isinstance(k, tuple) for k in indexed_weights.keys())

  for idx in list(indexed_weights.keys()):
    w = indexed_weights[idx].T

    if needs_llama31_rope_shuffle:
      if "query-kernel" in mt_key and suffix == "kernel_lora_b":
        w = w * (1.0 / np.sqrt(config.head_dim))

      if ("query-kernel" in mt_key or "key-kernel" in mt_key) and suffix == "kernel_lora_b":
        num_heads = config.num_query_heads if "query-kernel" in mt_key else config.num_kv_heads
        head_dim = config.head_dim
        orig_shape = w.shape

        work_val = w.reshape(orig_shape[0], num_heads, head_dim)
        half = head_dim // 2

        first_half = work_val[..., :half]
        second_half = work_val[..., half:]
        interleaved = np.stack([first_half, second_half], axis=-1).reshape(work_val.shape)
        w = interleaved.reshape(orig_shape)

    indexed_weights[idx] = w

  if not is_scanned:
    return np.array(indexed_weights[0], dtype=target_dtype)

  if is_2d_indexed:
    num_experts = max(k[0] for k in indexed_weights.keys()) + 1
    num_layers_2d = max(k[1] for k in indexed_weights.keys()) + 1

    sample_weight = next(iter(indexed_weights.values()))
    weights_array = np.zeros((num_experts, num_layers_2d) + sample_weight.shape, dtype=target_dtype)

    for (e_idx, l_idx), w in indexed_weights.items():
      weights_array[e_idx, l_idx] = w.astype(target_dtype)

    return weights_array

  weights_list = [None] * num_layers
  for idx, w in indexed_weights.items():
    if isinstance(idx, int) and idx < num_layers:
      weights_list[idx] = w

  sample_weight = next((w for w in weights_list if w is not None), None)
  if sample_weight is None:
    return np.array([], dtype=target_dtype)

  for i in range(num_layers):
    if weights_list[i] is None:
      weights_list[i] = np.zeros_like(sample_weight)

  return np.stack(weights_list, axis=axis_to_stack).astype(target_dtype)


def convert_lora_to_maxtext_adapter(
    config,
    lora_weights: dict[str, Any],
    save_dtype: str = "bfloat16",
) -> dict[str, Any]:
  """Converts HF LoRA weights to MaxText adapter format."""
  model_key = config.model_name
  if "-Instruct" in model_key:
    max_logging.log("Warning: You want an Instruct version, so we are using the base model architecture instead.")
    model_key = model_key.replace("-Instruct", "")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()
  param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_dict, config, config.scan_layers)

  mt_adapter_tree = {}
  mapped_count = 0
  target_dtype = ml_dtypes.bfloat16 if save_dtype == "bfloat16" else np.float32

  collected_weights = {}

  for hf_key, weight in lora_weights.items():
    mt_key, index = convert_hf_lora_key_to_maxtext(hf_key, param_map_mt_to_hf)

    if mt_key:
      if hasattr(weight, "numpy"):
        # bfloat16 to numpy direct conversion is not fully supported in all PyTorch versions
        if weight.dtype == torch.bfloat16:
          weight = weight.to(torch.float32)
        weight = weight.detach().cpu().numpy()
      suffix = "kernel_lora_a" if "lora_A" in hf_key or "lora_a" in hf_key else "kernel_lora_b"

      if isinstance(mt_key, tuple):
        mt_key = mt_key[0]  # Fallback for composite keys, though LoRA usually doesn't target them directly

      if mt_key not in collected_weights:
        collected_weights[mt_key] = {}
      if suffix not in collected_weights[mt_key]:
        collected_weights[mt_key][suffix] = {}

      idx = index if index is not None else 0
      collected_weights[mt_key][suffix][idx] = weight
      mapped_count += 1

  for mt_key, suffixes in collected_weights.items():
    clean_mt_key = mt_key.replace("-kernel", "")
    parts = clean_mt_key.split("-")
    if parts[0] == "params":
      parts = parts[1:]

    for suffix, indexed_weights in suffixes.items():
      is_scanned = isinstance(param_map_mt_to_hf.get(mt_key), list)
      num_layers = len(param_map_mt_to_hf[mt_key]) if is_scanned else 1

      final_weight = _process_and_stack_weights(
          indexed_weights, is_scanned, num_layers, config.param_scan_axis, target_dtype, mt_key, suffix, config
      )

      current = mt_adapter_tree
      for part in parts:
        if part not in current:
          current[part] = {}
        current = current[part]
      current[suffix] = {"value": final_weight}

  max_logging.log(f"Successfully mapped {mapped_count} out of {len(lora_weights)} LoRA parameters")
  return mt_adapter_tree


def _setup_merge_mode_getter(tensor_getter, config, hf_lora_adapter_path, revision):
  """Helper function to intercept the tensor_getter and inject LoRA weights dynamically."""
  max_logging.log("LoRA adapter path provided and load_parameters_path provided. Merging LoRA into base weights.")
  hf_access_token = config.hf_access_token
  lora_weights = load_hf_dict_from_safetensors(hf_lora_adapter_path, hf_access_token, revision)

  # Load adapter config to get scaling factor
  if os.path.isdir(hf_lora_adapter_path):
    config_path = os.path.join(hf_lora_adapter_path, "adapter_config.json")
  else:
    config_path = hf_hub_download(hf_lora_adapter_path, "adapter_config.json", token=hf_access_token)
  with open(config_path, "r", encoding="utf-8") as f:
    adapter_config = json.load(f)

  lora_alpha = adapter_config.get("lora_alpha", 8)
  lora_rank = adapter_config.get("r", 8)
  scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0

  base_to_lora = {}
  for k, w in lora_weights.items():
    if hasattr(w, "numpy"):
      if w.dtype == torch.bfloat16:
        w = w.to(torch.float32)
      w = w.detach().cpu().numpy()

    hf_param_key = k.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
    hf_param_key = hf_param_key.replace(".lora_A", "").replace(".lora_B", "")

    if hf_param_key.startswith("base_model.model."):
      hf_param_key = hf_param_key[len("base_model.model.") :]
    if hf_param_key.startswith("language_model.model."):
      hf_param_key = "model.language_model." + hf_param_key[len("language_model.model.") :]

    if hf_param_key not in base_to_lora:
      base_to_lora[hf_param_key] = {}

    if "lora_A" in k or "lora_a" in k:
      base_to_lora[hf_param_key]["A"] = w
    else:
      base_to_lora[hf_param_key]["B"] = w

  original_getter = tensor_getter

  def _merged_getter(key):
    base_w = original_getter(key)
    if key in base_to_lora:
      lora_dict = base_to_lora[key]
      if "A" in lora_dict and "B" in lora_dict:
        lora_a = np.array(lora_dict["A"], dtype=np.float32)
        lora_b = np.array(lora_dict["B"], dtype=np.float32)

        if lora_a.ndim > 2 or lora_b.ndim > 2:
          # Use einsum for multi-dimensional LoRA weights to contract on rank dimension
          delta = np.einsum("...ir,rj...->...ij...", lora_b, lora_a) * scaling
        else:
          delta = np.matmul(lora_b, lora_a) * scaling

        if hasattr(base_w, "dtype"):
          original_dtype = base_w.dtype
        else:
          original_dtype = np.float32

        if delta.shape != base_w.shape and delta.size == base_w.size:
          delta = delta.reshape(base_w.shape)

        base_w = np.array(base_w, dtype=np.float32) + delta
        return base_w.astype(original_dtype)

    return base_w

  return _merged_getter


def main(
    args: Sequence[str],
    lazy_load_tensors: bool = False,
    eager_load_method: str = "safetensors",
    hf_model_path: str | None = None,
    revision: str | None = None,
    save_dtype: str = "bfloat16",
    simulated_cpu_devices_count: int = 16,
) -> None:
  overall_start = time.time()
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

  # check the supported model ids
  if model_name_original not in HF_IDS:
    raise ValueError(
        f"Unsupported model name: {model_name_original}.\
                      Supported models are: {list(HF_IDS.keys())}"
    )

  model_id = hf_model_path or HF_IDS[model_name_original]

  # Initialize maxtext config
  config = pyconfig.initialize(args)
  max_utils.print_system_information()

  if not config.base_output_directory:
    output_directory = f"tmp/{config.run_name}"
  else:
    output_directory = config.base_output_directory

  hf_token = config.hf_access_token

  hf_lora_adapter_path = config.hf_lora_adapter_path

  is_adapter_only = bool(hf_lora_adapter_path and not config.load_parameters_path)
  is_merge_mode = bool(hf_lora_adapter_path and config.load_parameters_path)

  if is_adapter_only:
    max_logging.log("LoRA adapter path provided and load_parameters_path NOT provided. Converting LoRA adapter ONLY.")
    hf_access_token = config.hf_access_token
    lora_weights = load_hf_dict_from_safetensors(hf_lora_adapter_path, hf_access_token, revision)

    model_name_for_path = model_name_original or config.model_name
    jax_weights = convert_lora_to_maxtext_adapter(config, lora_weights, save_dtype)
    adapter_name = os.path.basename(os.path.normpath(hf_lora_adapter_path))
    output_directory = os.path.join(output_directory, model_name_for_path, adapter_name)
  else:

    if lazy_load_tensors and config.use_multimodal:
      raise ValueError("lazy loading of HF tensors is not supported for multimodal models yet.")

    hf_state_dict_numpy = None
    hf_loader = None

    # Define the appropriate tensor getter based on mode
    if lazy_load_tensors:
      max_logging.log(f"Lazy loading ENABLED. Initializing LazyHFLoader for: {model_id}...")
      hf_loader = LazyHFLoader(model_id, hf_token, revision=revision)

      print_ram_usage("After LazyLoader init")
      tensor_getter = hf_loader.get_tensor
    else:
      max_logging.log(f"Lazy loading DISABLED. Loading full HuggingFace model: {model_id}...")

      # Eager load methods:
      # - Method 1: transformers_class.from_pretrained(..., dtype="auto")
      # - Method 2: safetensors.safe_open(..., framework="pt")
      #
      # Comparison:
      # - Both methods result in the same dtype (usually bfloat16) and model structure
      #   for most models (e.g., DeepSeek-V2), with similar loading times.
      # - Exception: Gemma-3 uses different internal naming (prefixes) between
      #   Method 1 and Method 2. Current MaxText 'param_mapping' for Gemma-3 assumes
      #   the Transformers-style structure (Method 1).
      # - The 'safetensors' method is a necessary fallback for:
      #   1. "Day-0" models where the official Transformers code hasn't been merged yet
      #      (e.g., DeepSeek-V3.2 during its initial release).
      #   2. Weights omitted by official Transformers class
      #      (e.g., Multi-Token Prediction weights (`layers.61`) in DeepSeek-V3).
      #
      # Recommendation:
      # - Use 'safetensors' as the default. Since transformers 5.8.0, model initialization
      #   changed and the 'transformers' method may produce different key structures.
      # - Use 'transformers' only if explicitly needed for backward-compatible key mapping (e.g. Gemma3).
      if eager_load_method == "transformers":
        max_logging.log("Eager load with Transformers backend, from_pretrained with auto dtype")
        # For auto mode, loaded dtype is the same as `dtype` specified in config.json (or `torch_dtype` for older version)
        # e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json#L54
        hf_state_dict_numpy = load_hf_dict_from_transformers(model_id, token=hf_token, revision=revision, dtype="auto")
      elif eager_load_method == "safetensors":
        max_logging.log("Eager load with Safetensors backend, safe_open with pt framework")
        # For safe_open, loaded dtype is the same as original safetensor
        # e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/model.safetensors.index.json
        hf_state_dict_numpy = load_hf_dict_from_safetensors(model_id, token=hf_token, revision=revision, framework="pt")
      else:
        raise NotImplementedError

      unique_dtypes = {tensor.dtype for tensor in hf_state_dict_numpy.values()}
      max_logging.log(f"HuggingFace model loaded. dtypes: {unique_dtypes}")
      print_ram_usage("After full HF model load")

      def _eager_getter(key):
        if key not in hf_state_dict_numpy:
          raise ValueError(f"HuggingFace key {key} not found in state_dict.")
        v = hf_state_dict_numpy[key]
        # target dtype is "float32"
        if save_dtype == DType.FLOAT32:
          return v.to(torch.float32).numpy()
        # target dtype is "bfloat16"
        elif save_dtype == DType.BFLOAT16:
          # - torch.bfloat16 -> torch.float32 -> np.float32 -> ml_dtypes.bfloat16
          #   As numpy doesn't accept bfloat16 directly, we convert to float32 first
          # - torch.float16 -> np.float16 -> ml_dtypes.bfloat16
          # - torch.float32 -> np.float32 -> ml_dtypes.bfloat16
          if v.dtype == torch.bfloat16:
            v = v.to(torch.float32)
          return v.numpy().astype(ml_dtypes.bfloat16)
        raise NotImplementedError(f"Save dtype {save_dtype} is not currently implemented.")

      tensor_getter = _eager_getter

    if is_merge_mode:
      tensor_getter = _setup_merge_mode_getter(tensor_getter, config, hf_lora_adapter_path, revision)

    # Get parameter mappings and hooks
    model_key = config.model_name
    # load config
    hf_config_obj = HF_MODEL_CONFIGS[model_key]
    hf_config_dict = hf_config_obj.to_dict()
    # example of param mapping (gemma2, maxtext:huggingface):
    # "params-decoder-layers_{maxtext_layer_idx}-pre_self_attention_norm_global-scale":
    #   f"model.layers.{global_layer_idx}.input_layernorm.weight",
    param_map_mt_to_hf = PARAM_MAPPING[model_key](hf_config_dict, config, config.scan_layers)
    # Example of Hook FN mapping, to perform reshape:
    # f"params-decoder-layers_{maxtext_layer_idx}-self_attention_global-key-kernel": reshape_kernel,
    hook_fn_map_mt = HOOK_FNS[model_key](hf_config_dict, config, config.scan_layers, saving_to_hf=False)
    max_logging.log("Parameter mappings and hooks obtained.")

    maxtext_abstract_dict, abstract_params_treedef = get_maxtext_model_info(config)

    # Weight transformation
    max_logging.log("Starting weight transformation...")
    start = time.time()
    # Stores MaxText weights: numpy.ndarray
    final_mt_weights = [None] * len(maxtext_abstract_dict)

    # Preprocess key
    filtered_map_keys = validate_and_filter_param_map_keys(param_map_mt_to_hf.keys(), maxtext_abstract_dict.keys())

    for mt_param_key_or_keys in MemoryMonitorTqdm(
        filtered_map_keys,
        desc="Transforming weights",
        unit="param",
        leave=True,
        dynamic_ncols=True,
        smoothing=0,
    ):
      if not lazy_load_tensors:
        max_logging.log(f"maxtext param: {mt_param_key_or_keys}")

      hf_source_keys_or_key = param_map_mt_to_hf.get(mt_param_key_or_keys)
      if hf_source_keys_or_key is None:
        raise ValueError(f"MaxText parameter {mt_param_key_or_keys} not found in mapping.")
      hook_fn = hook_fn_map_mt.get(mt_param_key_or_keys)

      # Step 1: Resolves MaxText key(s) to target indices and shapes
      # based on MaxText key form (`atomic_mt_key` or `composite_mt_key`)
      mt_target_idx_or_indices, mt_target_shape_or_shapes = _get_maxtext_indices_and_shapes(
          mt_param_key_or_keys, maxtext_abstract_dict
      )

      # Step 2: Determine the loading function for hf key
      # based on hf_key form (unscanned, scanned, unscanned with expert stacking, or scanned with expert stacking)
      load_fn = _get_hf_loading_function(
          hf_source_keys_or_key,
          tensor_getter,
          hook_fn,
          mt_target_shape_or_shapes,
          config,
      )

      # Step 3: Load hf keys and convert to maxtext keys
      # based on tensor load mode (lazy, eager) and MaxText key form (`atomic_mt_key` or `composite_mt_key`)
      _get_maxtext_weight(
          load_fn,
          mt_target_idx_or_indices,
          mt_target_shape_or_shapes,
          mt_param_key_or_keys,
          final_mt_weights,
          save_dtype,
          lazy_load_tensors,
      )

    del hf_state_dict_numpy
    max_logging.log("Weight transformation preparation complete.")
    max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")
    print_ram_usage("Before creating full JAX tree")

    # Create final MaxText parameters tree
    jax_weights = jax.tree_util.tree_unflatten(abstract_params_treedef, final_mt_weights)
    del final_mt_weights, abstract_params_treedef

  print_ram_usage("Before saving")
  if lazy_load_tensors and not is_adapter_only:
    max_logging.log("Starting checkpoint save (loading weights just-in-time)...")
  else:
    max_logging.log("Starting checkpoint save...")

  # Save the converted weights to a MaxText checkpoint.
  # If simulated_cpu_devices_count > 1, weights are promoted from NumPy to JAX arrays
  # and sharded across virtual devices.
  save_weights_to_checkpoint(
      output_directory,
      jax_weights,
      simulated_cpu_devices_count,
      config.checkpoint_storage_use_ocdbt,
      config.checkpoint_storage_use_zarr3,
  )

  print_ram_usage("Program Ends")
  if is_adapter_only:
    max_logging.log(f"LoRA adapter conversion completed successfully. Saved to {output_directory}")
  else:
    max_logging.log(f"Conversion complete. Checkpoint saved to {output_directory}")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # Define local parser
  parser = argparse.ArgumentParser()
  # Lazy load uses `safetensors.safe_open` with np
  parser.add_argument(
      "--lazy_load_tensors",
      type=str2bool,
      required=False,
      default=False,
      help="Whether to use lazy loading of HF tensors",
  )
  # Eager load uses `transformers_class.from_pretrained` with auto dtype or `safetensors.safe_open` with pt.
  # The two methods are interchangeable in most cases.
  # Must use "transformers" for gemma3-4b due to mapping compatibility.
  # Must use "safetensors" for models without official transformers support, like DeepSeek-V3.2.
  # Must use "safetensors" for weights omitted by transformers class,
  #   like Multi-Token Prediction weights (`layers.61`) in DeepSeek-V3.
  parser.add_argument(
      "--eager_load_method",
      type=str,
      required=False,
      default="safetensors",
      choices=["transformers", "safetensors"],
      help="Backend to use for eager loading: `transformers_class.from_pretrained` or `safetensors.safe_open` with pt",
  )
  # If not specified, default to maxtext.utils.globals.HF_IDS[model_name]
  parser.add_argument(
      "--hf_model_path",
      type=str,
      required=False,
      default=None,
      help="Customized remote HF repo, or local path to HF model",
  )
  # If hf_model_path is set to a local path, this is ignored.
  parser.add_argument(
      "--revision",
      type=str,
      required=False,
      default=None,
      help="Specific Hugging Face revision (branch/tag/commit)",
  )
  parser.add_argument(
      "--save_dtype",
      type=str,
      required=False,
      default="bfloat16",
      choices=["float32", "bfloat16"],
      help="Save MaxText weights in specified dtype",
  )
  # Determines the logical sharding of the output checkpoint by partitioning
  # weights across virtual XLA devices.
  # - Even on a single CPU host, JAX can simulate multiple devices (e.g., 16)
  # - If set to 1, sharding is skipped.
  # - Sharding is preferred. For downstream loading on TPU pods, this helps prevent OOM and speedup.
  #
  # Example: Embedding Layer shape=(151936, 1024)
  # Case 1: simulated_cpu_devices_count=16 (Sharded)
  #   sharding: NamedShardingMetadata(shape=[16], ...)
  #   storage:  chunk_shape=(9496, 1024)  <-- 1/16th of rows per chunk
  # Case 2: simulated_cpu_devices_count=1 (Monolith)
  #   sharding: None
  #   storage:  chunk_shape=(151936, 1024) <-- Full layer in one chunk
  parser.add_argument(
      "--simulated_cpu_devices_count", type=int, required=False, default=16, help="Sharding of checkpoint"
  )
  # Parse local arguments
  # Parse known args returns the namespace AND the list of remaining arguments
  local_args, remaining_args = parser.parse_known_args()
  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set jax environment
  jax.config.update("jax_platforms", "cpu")
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"
  main(
      args=model_args,
      lazy_load_tensors=local_args.lazy_load_tensors,
      eager_load_method=local_args.eager_load_method,
      hf_model_path=local_args.hf_model_path,
      revision=local_args.revision,
      save_dtype=local_args.save_dtype,
      simulated_cpu_devices_count=local_args.simulated_cpu_devices_count,
  )
