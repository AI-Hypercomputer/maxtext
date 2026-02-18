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

""" Checkpoint conversion utility functions. """

import contextlib
import io
import os
import tempfile
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from tqdm import tqdm
import resource

import jax
from jax.experimental import multihost_utils

from jaxtyping import Array

import numpy as np

from google.cloud.storage import Client, transfer_manager

from safetensors.numpy import save_file as numpy_save_file
from safetensors.numpy import save as numpy_save
from safetensors.flax import save as save_flax_to_bytes

from huggingface_hub import HfApi, repo_exists

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import AutoModelForCausalLM
import torch

from maxtext.utils import max_logging
import psutil

from etils import epath
import orbax.checkpoint as ocp


SAFE_TENSORS_CONFIG_FILE = "config.json"
SAFE_TENSORS_WEIGHTS_FILE = "model.safetensors"
SAFE_TENSORS_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024 * 3  # 3GB default


# Mapping from MaxText model key to Hugging Face tokenizer identifiers
HF_IDS = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "gemma2-27b": "google/gemma-2-27b",
    "gemma3-4b": "google/gemma-3-4b-it",  # hf multi-modal should also support the pure-text
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-70b-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama3.1-405b": "meta-llama/Llama-3.1-405B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "qwen3-480b-a35b": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek2-16b": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek3-671b": "deepseek-ai/DeepSeek-V3",
    "deepseek3.2-671b": "deepseek-ai/DeepSeek-V3.2-Exp",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-omni-30b-a3b": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "olmo3-7b": "allenai/Olmo-3-7B-Instruct",
    "olmo3-7b-pt": "allenai/Olmo-3-1025-7B",
    "olmo3-32b": "allenai/Olmo-3-32B-Think",
}


def _get_local_directory(output_dir: str) -> str:
  """Determines the local directory for saving files."""
  if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
    # Fallback to a generic temp directory name if used directly
    local_dir = os.path.join(os.path.expanduser("~"), ".cache", "maxtext_hf_conversion_temp", "temp_files")
  else:
    local_dir = output_dir
  os.makedirs(local_dir, exist_ok=True)
  return local_dir


def validate_and_filter_param_map_keys(param_map_keys, maxtext_state_keys):
  """Validates param_mapping coverage and filters unused keys, for to_maxtext and to_huggingface.

  Preprocess maxtext keys for transformation.
  - Ensures every MaxText checkpoint key (`maxtext_state_keys`) is covered by
    the flattened param_mapping.
  - Keys in the param_mapping that are not present in the checkpoint (common for
    multi-variant maps like gemma3, qwen3, deepseek) are skipped.

  Args:
    param_map_keys: MaxText keys from the `PARAM_MAPPING`. These can be:
      - `atomic_mt_key`: A single string representing one MaxText parameter that map to HF parameter(s).
      - `composite_mt_key`: A tuple of strings representing multiple MaxText parameters that map to HF parameter(s).
    maxtext_state_keys: Set of MaxText keys loaded from the Orbax checkpoint.

  Returns:
    A list of 'filtered' mapping keys (strings or tuples) that are fully present
    and valid based on `maxtext_state_keys`.

  Raises:
    ValueError: If `maxtext_state_keys` is NOT a subset of the flattened
      `param_map_keys`.
  """
  flattened_map_keys = set()
  for key in param_map_keys:
    if isinstance(key, tuple):
      flattened_map_keys.update(key)
    else:
      flattened_map_keys.add(key)

  # 1 Validate: every maxtext state key must be covered by param map
  missing_keys = maxtext_state_keys - flattened_map_keys
  if missing_keys:
    raise ValueError(
        "maxtext_state_dict must be a subset of flattened param_map"
        + f"\nparam map\n{param_map_keys}"
        + f"\nmaxtext:\n{maxtext_state_keys}"
        + f"\nmissing keys:\n{missing_keys}"
    )

  # 2 Filter: param map may have extra keys
  extra_keys = flattened_map_keys - maxtext_state_keys
  if extra_keys:
    max_logging.log(f"Warning: extra keys in param_map are skipped: {extra_keys}")

  # skip extra keys in param map
  filtered_map_keys = []
  for key in param_map_keys:
    if (isinstance(key, str) and key in maxtext_state_keys) or (
        isinstance(key, tuple) and all(k in maxtext_state_keys for k in key)
    ):
      filtered_map_keys.append(key)
  return filtered_map_keys


def apply_hook_fns(weight, target_shape, hook_fns):
  """Apply hook functions, essential for to_maxtext and to_huggingface"""
  # If hook is unsepecified, use identity
  if hook_fns is None:
    return weight
  if not isinstance(hook_fns, list):
    hook_fns = [hook_fns]
  # Apply a list of hooks, be careful of order
  for hook_fn in hook_fns:
    weight = hook_fn(weight, target_shape)
  return weight


def convert_jax_weight_to_numpy(weight: "jax.Array", dtype_str: None | str = None) -> np.ndarray:
  """Converts a JAX array to a NumPy array with the specified dtype, used in to_huggingface.

  Args:
    weight: The input JAX array, potentially sharded across devices.
    dtype_str: The target NumPy dtype as a string (e.g., 'float32', 'bfloat16').
      If None, the dtype of the input JAX array is preserved. Defaults to None.

  Returns:
    A NumPy array containing the data from `weight`, cast to `dtype_str` if provided.
  """
  final_dtype_str = str(weight.dtype) if dtype_str is None else dtype_str
  # JAX dtypes like 'bfloat16', 'float32' are understood by np.dtype()
  target_np_dtype = np.dtype(final_dtype_str)
  expected_shape = weight.shape

  # Gather the array across devices if it's sharded.
  # process_allgather typically returns the array on the host.
  weight = multihost_utils.process_allgather(weight)

  # Convert JAX array to NumPy array.
  np_array = np.array(weight)

  # Cast to the target NumPy dtype if it's different.
  if np_array.dtype != target_np_dtype:
    np_array = np_array.astype(target_np_dtype)

  return np_array.reshape(expected_shape)  # Reshape for safety, though usually preserved.


def _process(hf_path, processed_slice, output_weights, current_hook_fns, hf_shape_map):
  """Applies hooks, converts a JAX slice to NumPy, and appends it to the output list, used in to_huggingface"""
  if hf_path not in hf_shape_map:
    raise ValueError(f"HF path '{hf_path}' not found in hf_shape_map.")
  target_hf_shape = hf_shape_map[hf_path]
  # If hook is unsepecified, use identity
  if current_hook_fns:
    processed_slice = apply_hook_fns(processed_slice, target_hf_shape, current_hook_fns)
  numpy_slice = convert_jax_weight_to_numpy(processed_slice).squeeze()
  if numpy_slice.shape != tuple(target_hf_shape):
    raise ValueError(f"Shape mismatch for {hf_path}: Expect {target_hf_shape}, got {numpy_slice.shape}")
  output_weights.append((hf_path, numpy_slice))


def process_maxtext_param(
    maxtext_param_key: str | tuple[str, ...],
    maxtext_param_weight: jax.Array | list[jax.Array],
    param_map: dict[str, Any],
    hook_fn_map: dict[str, Any],
    hf_shape_map: dict[str, Any],
    maxtext_config: Any,
) -> list[tuple[str, np.ndarray]]:
  """Processes a single MaxText parameter (or a group of parameters) for conversion, used in to_huggingface.

  This function is responsible for taking a MaxText parameter and transforming
  it into one or more Hugging Face compatible parameters. It handles various
  scenarios based on
  - the MaxText key form (`atomic_mt_key` or `composite_mt_key`)
  - and the Hugging Face value form (unscanned string, scanned list of strings,
    unscanned with expert stacking, or scanned with expert stacking).
  Note: We assume composite_mt_key can only occur for unscanned/scanned HF keys, but not those with expert stacking.

  Args:
    maxtext_param_key: The key identifying the MaxText parameter(s). Can be
      an `atomic_mt_key` (str) or a `composite_mt_key` (tuple of str) mapping
      to HF parameter(s).
    maxtext_param_weight: The actual weight(s) of the MaxText parameter(s).
      This can be a single `jax.Array` for an `atomic_mt_key` or a list of
      `jax.Array` for a `composite_mt_key`.
    param_map: A dictionary mapping MaxText parameter keys to their corresponding
      Hugging Face target path(s).
    hook_fn_map: A dictionary mapping MaxText parameter keys to transformation
      functions (hooks) that should be applied to the weights.
    hf_shape_map: A dictionary mapping Hugging Face parameter paths to their
      expected shapes.
    maxtext_config: The MaxText configuration object, used to determine
      details like `param_scan_axis` and `base_num_decoder_layers`.

  Returns:
    A list of tuples, where each tuple contains:
    - hf_path (str): The Hugging Face parameter path.
    - hf_weight (np.ndarray): The transformed Hugging Face compatible weight.
  """
  max_logging.log(f"maxtext param: {maxtext_param_key}")

  if maxtext_param_key not in param_map:
    raise ValueError(f"MaxText param key '{maxtext_param_key}' not found in param_map.")
  hf_target_paths = param_map[maxtext_param_key]
  if not hf_target_paths:
    raise ValueError(f"No HF target paths found for MaxText key '{maxtext_param_key}'")

  # If maxtext_param_key is not in hook_fn_map, current_hook_fns is None, indicating identity (no transformation)
  current_hook_fns = hook_fn_map.get(maxtext_param_key)

  # This list will store tuples of (hf_path, hf_weight)
  output_weights = []

  # Case 1: Unscanned
  if not isinstance(hf_target_paths, list):
    max_logging.log("\tunscan")
    hf_path = hf_target_paths
    _process(hf_path, maxtext_param_weight, output_weights, current_hook_fns, hf_shape_map)
    return output_weights

  # Stacked MaxText weight
  # This now handles three cases:
  # 2. Standard scanned layers (1D list of targets from a tensor stacked only on the layer axis)
  # 3. Unscanned MoE layers (1D list of targets from a tensor stacked only on the expert axis)
  # 4. Scanned MoE layers (2D list of targets from a tensor stacked on expert and layer axes)

  if not isinstance(hf_target_paths[0], list):
    # Case 2 or 3: The source tensor is stacked on a single axis.
    # i.e., hf_target_paths is an (un-nested) list
    # We determine if it's standard scanned (stack on layer axis) or unscanned MoE (stack on expert axis).
    if maxtext_config.scan_layers:
      max_logging.log("\tscan")
      # Case 2: Standard scanned layer.
      # The tensor is stacked ONLY on the layer axis.
      axis_to_slice = maxtext_config.param_scan_axis
    else:
      max_logging.log("\tunscan moe")
      # Case 3: Unscanned MoE layer, e.g., from 'layers_0-moe_block-wi_0'.
      # The tensor is stacked ONLY on the expert axis. Assuming expert is axis 0.
      axis_to_slice = 0

    # Iterate through the slices of the MaxText weight along the determined stacking axis.
    # Handles MaxText key forms (`atomic_mt_key` and `composite_mt_key`)
    for i, hf_path in enumerate(hf_target_paths):
      if isinstance(maxtext_param_weight, list):
        # This handles `composite_mt_key` mappings where `maxtext_param_weight` is a list of tensors.
        # Each tensor in the list is sliced independently along the `axis_to_slice`.
        weight_slice = [jax.lax.index_in_dim(x, i, axis=axis_to_slice, keepdims=False) for x in maxtext_param_weight]
      else:
        # For `atomic_mt_key` mappings, slice the single MaxText tensor.
        weight_slice = jax.lax.index_in_dim(maxtext_param_weight, i, axis=axis_to_slice, keepdims=False)
      _process(hf_path, weight_slice, output_weights, current_hook_fns, hf_shape_map)

    return output_weights

  # Multi axis stacked: isinstance(hf_target_paths[0], list)
  max_logging.log("\tscan moe")
  # Case 4: Scanned MoE layer, e.g., from 'layers-moe_block-wi_0'.
  # The tensor is stacked on expert and layer axes. We slice experts first, then layers.
  # MaxText format is (experts, layers, ...), so expert axis is 0, layer axis is 1.
  expert_axis_to_slice = 0

  # Outer loop for experts
  for expert_idx, expert_paths_for_layer in enumerate(hf_target_paths):
    # Slice along the expert axis to get the tensor for the current expert across all layers.
    expert_tensor_slice = jax.lax.index_in_dim(
        maxtext_param_weight, expert_idx, axis=expert_axis_to_slice, keepdims=False
    )
    # Inner loop for layers
    for layer_idx, hf_path in enumerate(expert_paths_for_layer):
      # Slice the expert tensor along the layer axis to get the final individual weight.
      # axis is 0 on the new sliced tensor
      layer_tensor_slice = jax.lax.index_in_dim(expert_tensor_slice, layer_idx, axis=0, keepdims=False)
      _process(hf_path, layer_tensor_slice, output_weights, current_hook_fns, hf_shape_map)

  return output_weights


def create_huggingface_hub_repo_if_not_exist(repo_id, repo_type):
  if not repo_exists(repo_id, repo_type=repo_type):
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        exist_ok=True,
        private=True,
    )
    max_logging.log(f"\n Created new HuggingFace Hub {repo_type} repo: {repo_id}.")


def save_config_file(
    config,
    local_path_to_save_to: str,
    output_dir_final: str,
    file_name: str,
    remove_local_copy_after_upload: bool = False,
):
  """Saves the model configuration file(config.json)."""
  if jax.process_index() == 0:
    config.architectures = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]]
    if output_dir_final.startswith("hf://"):
      max_logging.log(f"  Serializing {file_name} to memory for Hugging Face Hub upload...")
      json_string = config.to_json_string()
      json_bytes = json_string.encode("utf-8")
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(json_bytes) as f:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model",
        )
      max_logging.log(f"  Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      # local storage
      actual_local_file_path = os.path.join(local_path_to_save_to, file_name)
      config.to_json_file(actual_local_file_path)
      max_logging.log(f"   Saved {file_name} to {actual_local_file_path}")
      # upload
      if output_dir_final.startswith("gs://"):
        upload_file_to_gcs(
            actual_local_file_path,
            os.path.join(output_dir_final, file_name),
            remove_local_file_after_upload=remove_local_copy_after_upload,
        )


def shard_checkpoint(
    weights_dict: dict[str, Array],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    weights_name: str = "model.safetensors",
) -> tuple[dict[str, dict[str, Array]], None | dict]:
  """Shards a model checkpoint into smaller pieces based on size constraints.

  Args:
      weights_dict: Model weights dictionary to shard
      max_shard_size: Maximum size in bytes for each shard
      weights_name: Base filename for the shards

  Returns:
      tuple of (sharded weights dict, optional index dict)
      Index contains metadata and weight mapping information
  """
  # Track current shard and accumulated sizes
  current_shard: dict[str, Array] = {}
  shards: list[dict[str, Array]] = [current_shard]
  current_size = 0
  total_size = 0

  # Iterate through weights in sorted order for deterministic sharding
  for key, tensor in sorted(weights_dict.items()):
    weight_size = tensor.size * tensor.itemsize
    # Start new shard if current one would exceed size limit
    if (current_size + weight_size > max_shard_size) and len(current_shard.items()):
      current_shard = {}
      shards.append(current_shard)
      current_size = 0

    # Add weight to current shard and update sizes
    current_shard[key] = tensor
    current_size += weight_size
    total_size += weight_size

  # Return single shard without index if no sharding needed
  if len(shards) == 1:
    return {weights_name: shards[0]}, None

  # Generate shard filenames and build index
  shard_dict = {}
  weight_map = {}

  for idx, shard in enumerate(shards, 1):
    # Create numbered shard filename
    shard_name = weights_name.replace(".safetensors", f"-{idx:05d}-of-{len(shards):05d}.safetensors")
    shard_dict[shard_name] = shard

    # Map each weight to its shard file
    for key in shard:
      weight_map[key] = shard_name

  return shard_dict, {
      "metadata": {"total_size": total_size},
      "weight_map": weight_map,
  }


def save_safetensor_file(
    state_dict,
    local_dir_to_save_to: str,
    output_dir_final: str,
    file_name: str,
):
  """Saves a single safetensor file, from memory to remote when uploading"""
  if jax.process_index() == 0:
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    if "model.safetensors" in state_dict and isinstance(state_dict["model.safetensors"], dict):
      state_dict = state_dict["model.safetensors"]

    if output_dir_final.startswith("gs://"):
      cloud_path = os.path.join(output_dir_final, file_name)
      upload_state_dict_to_gcs(state_dict=state_dict, gs_bucket_path=cloud_path)
    elif output_dir_final.startswith("hf://"):
      max_logging.log(f"  Serializing {file_name} to memory for Hugging Face Hub upload...")
      serialized_content = save_flax_to_bytes(state_dict, metadata={"format": "pt"})
      # Upload in-memory; skip local storage
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(serialized_content) as f:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model",
        )
      max_logging.log(f"  Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      # local storage
      local_path = os.path.join(local_dir_to_save_to, file_name)
      numpy_save_file(state_dict, local_path, metadata={"format": "pt"})
      max_logging.log(f"   Saved {file_name} to {local_path}")


def save_index_file(
    index: dict,
    local_dir_to_save_to: str,
    output_dir_final: str,
    file_name: str,
    remove_local_copy_after_upload: bool = False,
):
  """Saves the model index json file (model.safetensors.index.json)."""
  if jax.process_index() == 0:
    local_path = os.path.join(local_dir_to_save_to, file_name)

    if output_dir_final.startswith("hf://"):
      max_logging.log(f"   Serialized {file_name} to memory for Hugging Face Hub upload.")
      json_bytes = json.dumps(index, indent=2).encode("utf-8")
      repo_id = output_dir_final.lstrip("hf://")
      api = HfApi()
      with io.BytesIO(json_bytes) as f:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model",
        )
      max_logging.log(f"   Successfully uploaded {file_name} to HF repo: {repo_id}")
    else:
      with open(local_path, "wt", encoding="utf8") as f:
        json.dump(index, f, indent=2)
      max_logging.log(f"   Saved {file_name} to {local_path}")
      if output_dir_final.startswith("gs://"):
        upload_file_to_gcs(
            local_path,
            os.path.join(output_dir_final, file_name),
            remove_local_file_after_upload=remove_local_copy_after_upload,
        )
        max_logging.log(f"   Successfully uploaded {file_name} to GCS: {output_dir_final}")


def save_weight_files(
    shards,
    index,
    local_dir_to_save_to: str,
    output_dir_final: str,
    parallel_threads=8,
    remove_local_copy_after_upload: bool = False,
):
  """Saves weight files and index if needed.

  Requires local system to have at least `parallel_threads * DEFAULT_MAX_SHARD_SIZE`
  free disk space, as each thread will maintain a local cache of its shard during processing.
  """
  if index is None:
    # 'shards' is actually the single state_dict here
    save_safetensor_file(shards, local_dir_to_save_to, output_dir_final, SAFE_TENSORS_WEIGHTS_FILE)
  else:
    # Save sharded weights in parallel
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
      shard_items = list(shards.items())
      futures = [
          executor.submit(
              save_safetensor_file,
              shard_dict,
              local_dir_to_save_to,
              output_dir_final,
              shard_name,
          )
          for shard_name, shard_dict in shard_items
      ]
      for future in futures:
        future.result()

    # Save index file
    save_index_file(
        index, local_dir_to_save_to, output_dir_final, SAFE_TENSORS_INDEX_FILE, remove_local_copy_after_upload
    )


@contextlib.contextmanager
def get_local_save_path_manager(output_dir: str):
  """
  Context manager to provide a local path for saving files.
  If output_dir is remote (GCS/HF), a temporary local directory is created.
  If output_dir is local, it's used directly.
  Yields:
      tuple: (path_to_use_for_saving: str, is_temporary: bool)
  """
  if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
    with tempfile.TemporaryDirectory(prefix="maxtext_hf_save_") as temp_dir:
      max_logging.log(f"   Using temporary local staging directory: {temp_dir}")
      yield temp_dir, True  # path, is_temporary
  else:
    os.makedirs(output_dir, exist_ok=True)
    max_logging.log(f"   Using local directory: {output_dir}")
    yield output_dir, False  # path, is_temporary


def save_model_files(
    weight_arrays: dict,
    config,  # HF config object
    tokenizer: None | Any,  # transformers.PreTrainedTokenizerBase
    processor,
    output_dir: str,
    parallel_threads=8,
):
  """
  Saves model files (config and weights) to the specified directory.
  When uploading to GCS/HF hub,
          *.safetensors are uploaded from memory to remote, no local storage is used to save disk usage
  """

  if output_dir.startswith("hf://"):
    create_huggingface_hub_repo_if_not_exist(repo_id=output_dir.lstrip("hf://"), repo_type="model")
    repo_id = output_dir.lstrip("hf://")
  else:
    repo_id = None

  max_logging.log(f"\n-> Saving model and tokenizer (if provided) to {output_dir}...")

  with get_local_save_path_manager(output_dir) as (current_save_path, is_temp_path):
    remove_local_copy = is_temp_path

    if jax.process_index() == 0:
      files_to_upload = []
      if processor is not None:
        max_logging.log(f"    Saving image processor files to {current_save_path}...")
        saved_image_processor_files = processor.save_pretrained(current_save_path)
        max_logging.log(f"    Processor files saved locally: {saved_image_processor_files}")
      elif tokenizer is not None:
        max_logging.log(f"    Saving tokenizer files to {current_save_path}...")
        saved_tokenizer_files = tokenizer.save_pretrained(current_save_path)
        max_logging.log(f"    Tokenizer files saved locally: {saved_tokenizer_files}")
      files_to_upload = [os.path.join(current_save_path, f) for f in os.listdir(current_save_path)]

      if output_dir.startswith("gs://"):
        for local_file_path in files_to_upload:
          if not os.path.exists(local_file_path):
            max_logging.log(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to GCS.")
            continue
          file_name = os.path.basename(local_file_path)
          upload_file_to_gcs(
              local_file_path,
              os.path.join(output_dir, file_name),
              remove_local_file_after_upload=remove_local_copy,
          )
      elif output_dir.startswith("hf://") and repo_id:
        api = HfApi()
        for local_file_path in files_to_upload:
          if not os.path.exists(local_file_path):
            max_logging.log(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to HF Hub.")
            continue
          file_name = os.path.basename(local_file_path)
          api.upload_file(
              path_or_fileobj=local_file_path,
              path_in_repo=file_name,
              repo_id=repo_id,
              repo_type="model",
          )
          if remove_local_copy:
            os.remove(local_file_path)
            max_logging.log(f"   Removed local copy: {local_file_path}")

      # Save config.json
      save_config_file(config, current_save_path, output_dir, SAFE_TENSORS_CONFIG_FILE, remove_local_copy)

    # Save .safetensors files (sharding can be outside process guard if weights are replicated)
    # The actual file saving within save_weight_files is guarded.
    # Unwrap nested dict if needed
    shards, index = shard_checkpoint(weight_arrays)
    save_weight_files(shards, index, current_save_path, output_dir, parallel_threads, remove_local_copy)

  if jax.process_index() == 0:
    max_logging.log(f"✅ Model and tokenizer (if provided) successfully processed for {output_dir}")


def upload_state_dict_to_gcs(state_dict: dict, gs_bucket_path: str):
  """Uploads a state_dict from memory to Google Cloud Storage.

  Args:
      state_dict: A PyTorch model's state_dict.
      gs_bucket_path: GCS destination (e.g., "gs://my-bucket/models/model.pt").
  """
  # TODO(shuningjin): max retries exceeded when uploading hf checkpoint for deepseek3-671b, b/457821616
  # Standardize bucket path format
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name, *blob_path_parts = gs_bucket_path.split("/")
  blob_name = "/".join(blob_path_parts)

  # 1. Serialize the state_dict to an in-memory byte buffer
  data = numpy_save(state_dict, metadata={"format": "pt"})
  buffer = io.BytesIO(data)
  buffer.seek(0)  # Rewind the buffer to the beginning

  # 2. Upload the bytes to GCS
  storage_client = Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)

  print(f"-> Uploading in-memory state_dict to {gs_bucket_path}...")
  blob.upload_from_file(buffer, content_type="application/octet-stream", timeout=600)
  print(f"✅ Uploaded to {bucket.name}/{blob_name}")


def upload_file_to_gcs(local_file: str, gs_bucket_path: str, remove_local_file_after_upload=False):
  """Uploads a single file to Google Cloud Storage.

  Args:
      local_file: Path to local file
      gs_bucket_path: GCS destination (e.g. "gs://my-bucket/path/file.txt" or "my-bucket/path/file.txt")
  """
  # Standardize bucket path format
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name = gs_bucket_path.split("/")[0]
  blob_name = gs_bucket_path[len(bucket_name) :].lstrip("/")

  max_logging.log(f"-> Uploading {local_file} to {gs_bucket_path}...")
  # Upload file
  storage_client = Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  # set large timeout to support large safetensors
  blob.upload_from_filename(local_file, timeout=600)

  max_logging.log(f"✅ Uploaded {local_file} to {bucket.name}/{blob_name}")

  if remove_local_file_after_upload:
    os.remove(local_file)
    max_logging.log(f"✅ Deleted {local_file}")


def upload_folder_to_gcs(local_folder: str, gs_bucket_path: str, num_workers: int = 4):
  """Uploads all files from a local folder to Google Cloud Storage.

  Args:
      local_folder: Path to local folder (e.g. "data/images")
      gs_bucket_path: GCS destination (e.g. "gs://my-bucket/images" or "my-bucket/images")
      num_workers: Number of parallel upload workers
  """
  start_time = time.time()

  # Standardize bucket path format
  gs_bucket_path = gs_bucket_path.removeprefix("gs://")
  bucket_name = gs_bucket_path.split("/")[0]
  destination_dir = gs_bucket_path[len(bucket_name) :]
  if destination_dir.startswith("/"):
    destination_dir = destination_dir[1:]
  # Ensure destination ends with "/"
  if destination_dir != "" and not destination_dir.endswith("/"):
    destination_dir += "/"

  # Get files to upload
  files_in_local_folder = os.listdir(local_folder)
  # Set up GCS client
  storage_client = Client()
  bucket = storage_client.bucket(bucket_name)

  # Upload files in parallel
  results = transfer_manager.upload_many_from_filenames(
      bucket,
      files_in_local_folder,
      source_directory=local_folder,
      max_workers=num_workers,
      blob_name_prefix=destination_dir,
      timeout=600,
      deadline=None,
  )

  # Report results
  for name, result in zip(files_in_local_folder, results):
    if isinstance(result, Exception):
      max_logging.log(f"Failed to upload {name}: {result}")
    else:
      max_logging.log(f"✅ Uploaded {name} to {bucket.name}/{destination_dir}{name}")

  max_logging.log(f"Upload completed in {time.time() - start_time}s")


def print_ram_usage(stage=""):
  memory = psutil.virtual_memory()
  max_logging.log(
      f"[{stage}] RAM Usage: {memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB ({memory.percent:.1f}%)"
  )


def print_peak_memory():
  # Returns peak usage in Kilobytes on Linux
  peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  max_logging.log(f"Peak Memory: {peak_memory_kb / 1024**2:.2f} GB")


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


def load_orbax_checkpoint(config) -> dict:
  """Loads a full Orbax checkpoint from disk with unsharded arrays.

  Args:
    config: MaxText config containing checkpoint storage settings

  Returns:
    Dictionary containing the full checkpoint structure
  """
  # Create Orbax checkpointer
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
  )

  # Get checkpoint metadata
  checkpoint_path = epath.Path(config.load_parameters_path)
  metadata = ckptr.metadata(checkpoint_path)

  # Create a mesh with all devices for unsharded restoration
  devices = np.array(jax.devices()).reshape((-1,))
  single_device_mesh = jax.sharding.Mesh(devices, ("x",))

  def create_restore_args(tree_metadata):
    """Create restore args for unsharded restoration."""
    if hasattr(tree_metadata, "shape"):
      return ocp.ArrayRestoreArgs(sharding=jax.sharding.NamedSharding(single_device_mesh, jax.sharding.PartitionSpec()))
    elif isinstance(tree_metadata, dict):
      return {k: create_restore_args(v) for k, v in tree_metadata.items()}
    else:
      return None

  restore_args = jax.tree_util.tree_map(
      lambda x: create_restore_args(x) if hasattr(x, "shape") else None,
      metadata.item_metadata.tree,
      is_leaf=lambda x: hasattr(x, "shape"),
  )

  # Restore the entire checkpoint
  return ckptr.restore(checkpoint_path, restore_args=restore_args)


def extract_nnx_weights(weights_dict: dict) -> dict[str, np.ndarray]:
  """Extract weights from NNX checkpoint structure.

  NNX checkpoints have structure: {'decoder': {'decoder_norm': {'scale': {'value': array}}}}
  This function flattens it to: {'params-decoder-decoder_norm-scale': array}

  Args:
    weights_dict: NNX checkpoint weights dictionary

  Returns:
    Dictionary mapping parameter names to weight arrays
  """
  result = {}
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(weights_dict)
  for path_tuple, leaf_value in leaves_with_paths:
    path_keys = [k.key for k in path_tuple]
    # Skip NNX RNG state variables (not model weights)
    if "to_nnx__rngs" in path_keys or any(k.endswith("_rngs") for k in path_keys):
      continue
    # Skip if this is the "value" key itself - we want the parent path
    if path_keys[-1] == "value":
      path_keys = path_keys[:-1]
    maxtext_param_key = "params-" + "-".join(path_keys)
    if not isinstance(leaf_value, (jax.Array, np.ndarray)):
      raise ValueError(f"Leaf value for {maxtext_param_key} is not an array. Type: {type(leaf_value)}.")
    result[maxtext_param_key] = leaf_value
  return result


def extract_linen_weights(weights_dict: dict) -> dict[str, np.ndarray]:
  """Extract weights from Linen checkpoint structure.

  Linen checkpoints have structure: {'params': {'decoder': {'decoder_norm': {'scale': array}}}}
  This function flattens it to: {'params-decoder-decoder_norm-scale': array}

  Args:
    weights_dict: Linen checkpoint weights dictionary

  Returns:
    Dictionary mapping parameter names to weight arrays
  """
  result = {}
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(weights_dict)
  for path_tuple, leaf_value in leaves_with_paths:
    path_keys = [k.key for k in path_tuple]
    # Construct maxtext_param_key from path_tuple
    maxtext_param_key = "params-" + "-".join(path_keys)
    if not isinstance(leaf_value, (jax.Array, np.ndarray)):
      raise ValueError(f"Leaf value for {maxtext_param_key} is not an array. Type: {type(leaf_value)}.")
    result[maxtext_param_key] = leaf_value
  return result


def detect_and_extract_checkpoint(checkpoint_dict: dict) -> dict[str, np.ndarray]:
  """Detect checkpoint type (Linen vs NNX) and extract weights.

  Handles multiple NNX checkpoint variants:
  - Linen: {'params': {'params': {'decoder': {...}, 'token_embedder': ... {WEIGHT_ARRAY}}}}
  - NNX-SFT: {'decoder': {...}, 'token_embedder': ... {'value': WEIGHT_ARRAY}}
  - NNX-RL: {'base': {'decoder': {...}, 'token_embedder': ... {'value': WEIGHT_ARRAY}}}

  Currently, we align all extracted weights to MaxText-Linen naming convention
  like "params-decoder-decoder_norm-scale". This allows reusing the same param_mapping
  for both Linen and NNX checkpoints.

  Args:
    checkpoint_dict: Raw checkpoint dictionary from Orbax

  Returns:
    Dictionary mapping MaxText parameter names to weight arrays
  """
  # Detect checkpoint type by structure
  actual_weights_dict = checkpoint_dict.get("params")

  if actual_weights_dict is None:
    # NNX checkpoint: structure is directly at the root
    # Check for NNX-RL variant with 'base' wrapper
    if "base" in checkpoint_dict and isinstance(checkpoint_dict["base"], dict):
      # NNX-RL: {'base': {'decoder': ..., 'token_embedder': ...}}
      max_logging.log("Detected NNX-RL checkpoint structure (with 'base' wrapper)")
      return extract_nnx_weights(checkpoint_dict["base"])
    else:
      # NNX-SFT: {'decoder': ..., 'token_embedder': ...}
      max_logging.log("Detected NNX-SFT checkpoint structure")
      return extract_nnx_weights(checkpoint_dict)
  else:
    # Linen checkpoint: check if there's a nested 'params' key
    if isinstance(actual_weights_dict, dict) and "params" in actual_weights_dict:
      actual_weights_dict = actual_weights_dict["params"]
      max_logging.log("Detected Linen checkpoint structure")
    else:
      max_logging.log("Detected Linen checkpoint structure (single params layer)")
    return extract_linen_weights(actual_weights_dict)


def get_hf_dict_from_pretrained(model_id: str, token: str, revision: str = None, dtype=None):
  """Loads the HuggingFace model based on model_id (Eager mode only), used in to_maxtext"""
  if model_id in ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]:
    from transformers import Qwen3OmniMoeForConditionalGeneration  # pylint: disable=import-outside-toplevel

    model_class = Qwen3OmniMoeForConditionalGeneration
  else:
    model_class = AutoModelForCausalLM

  if dtype is None:
    hf_model = model_class.from_pretrained(model_id, token=token, revision=revision)
  else:
    hf_model = model_class.from_pretrained(model_id, token=token, revision=revision, dtype=dtype)

  return hf_model.state_dict()
