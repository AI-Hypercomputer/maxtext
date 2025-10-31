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

import jax
import jax.tree_util
from jax.experimental import multihost_utils

from jaxtyping import Array

import numpy as np

from google.cloud.storage import Client, transfer_manager

from safetensors.numpy import save_file as numpy_save_file
from safetensors.numpy import save as numpy_save
from safetensors.flax import save as save_flax_to_bytes

from huggingface_hub import HfApi, repo_exists

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from MaxText import max_logging


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
    "deepseek3-test": "deepseek-ai/DeepSeek-V3",
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


def _process(hf_path, processed_slice, output_weights, current_hook_fns, shape_map_local):
  target_hf_shape = shape_map_local[hf_path]
  if current_hook_fns:
    # otherwise identity
    processed_slice = apply_hook_fns(processed_slice, target_hf_shape, current_hook_fns)
  numpy_slice = convert_jax_weight_to_numpy(processed_slice).squeeze()
  assert len(target_hf_shape) == len(
      numpy_slice.shape
  ), f"shape mismatch {len(target_hf_shape)} and {len(numpy_slice.shape)}"
  output_weights.append((hf_path, numpy_slice))


def process_leaf_param(
    path_tuple: Any,
    leaf_value: jax.Array,
    param_map_local: dict[str, Any],
    shape_map_local: dict[str, Any],
    hook_fn_map_local: dict[str, Any],
    current_config: Any,
) -> list[tuple[str, np.ndarray]]:
  """Processes a single leaf from the MaxText parameter tree."""

  # Construct maxtext_param_key from path_tuple
  key_parts = []
  for p_entry in path_tuple:
    if isinstance(p_entry, jax.tree_util.DictKey):
      key_parts.append(p_entry.key)
    else:
      max_logging.log(f"Warning: Path tuple {path_tuple} contains non-DictKey entry '{p_entry}'. Skipping this path.")
      return []  # Skip this parameter

  maxtext_param_key = "params-" + "-".join(key_parts)
  print(f"mt param: {maxtext_param_key}")

  if not isinstance(leaf_value, (jax.Array, np.ndarray)):
    max_logging.log(f"Warning: Leaf value for {maxtext_param_key} is not an array. Type: {type(leaf_value)}. Skipping.")
    return []

  if maxtext_param_key not in param_map_local:
    max_logging.log(f"Warning: MaxText param key '{maxtext_param_key}' not found in param_map. Skipping.")
    return []

  hf_target_paths = param_map_local[maxtext_param_key]
  if not isinstance(hf_target_paths, list):
    hf_target_paths = [hf_target_paths]
  print(f"\thf param: {hf_target_paths}")

  if not hf_target_paths:
    max_logging.log(f"Warning: No HF target paths found for MaxText key '{maxtext_param_key}'. Skipping.")
    return []

  current_hook_fns = hook_fn_map_local.get(maxtext_param_key)
  output_weights = []

  # TODO(shuningjin): it could be scan with one layer
  # TODO(shuningjin): it could be scan with one layer with one expert
  # Case 0: Unscan, or Scan with single layer
  if len(hf_target_paths) == 1:
    print("\tunscan or scan single layer")
    hf_path = hf_target_paths[0]
    if hf_path not in shape_map_local:
      max_logging.log(
          f"Warning: HF path '{hf_path}' not found in shape_map for MaxText key '{maxtext_param_key}'. Skipping."
      )
      return []
    _process(hf_path, leaf_value, output_weights, current_hook_fns, shape_map_local)
    return output_weights

  # Stacked MaxText weight
  # This now handles three cases:
  # 1. Scanned MoE layers (2D list of targets from a tensor stacked on expert and layer axes)
  # 2. Unscanned MoE layers (1D list of targets from a tensor stacked only on the expert axis)
  # 3. Standard scanned layers (1D list of targets from a tensor stacked only on the layer axis)
  is_scanned_moe_layer = isinstance(hf_target_paths[0], list)

  if is_scanned_moe_layer:
    print("\tscan moe")
    # Case 1: Scanned MoE layer, e.g., from 'layers-moe_block-wi_0'.
    # The tensor is stacked on expert and layer axes. We slice experts first, then layers.
    # MaxText format is (experts, layers, ...), so expert axis is 0, layer axis is 1.
    expert_axis_to_slice = 0

    # Outer loop for experts
    for expert_idx, expert_paths_for_layer in enumerate(hf_target_paths):
      # Slice along the expert axis to get the tensor for the current expert across all layers.
      expert_tensor_slice = jax.lax.index_in_dim(leaf_value, expert_idx, axis=expert_axis_to_slice, keepdims=False)
      # Inner loop for layers
      for layer_idx, hf_path in enumerate(expert_paths_for_layer):
        if hf_path not in shape_map_local:
          max_logging.log(
              f"Warning: HF path '{hf_path}' not found in shape_map for MaxText key '{maxtext_param_key}'. Skipping."
          )
          continue

        # Slice the expert tensor along the layer axis to get the final individual weight.
        # axis is 0 on the new sliced tensor
        layer_tensor_slice = jax.lax.index_in_dim(expert_tensor_slice, layer_idx, axis=0, keepdims=False)
        _process(hf_path, layer_tensor_slice, output_weights, current_hook_fns, shape_map_local)

    return output_weights

  # Case 2 or 3: The source tensor is stacked on a single axis.
  # We determine if it's an unscanned MoE (expert axis) or standard scanned (layer axis).
  is_unscanned_moe = "moe_block" in maxtext_param_key and any(
      f"_{i}-" in maxtext_param_key for i in range(current_config.base_num_decoder_layers)
  )

  if is_unscanned_moe:
    print("\tunscan moe")
    # Case 2: Unscanned MoE layer, e.g., from 'layers_0-moe_block-wi_0'.
    # The tensor is stacked ONLY on the expert axis.
    axis_to_slice = 0  # Assuming expert is axis 0.
  else:
    print("\tscan")
    # Case 3: Standard scanned layer.
    # The tensor is stacked ONLY on the layer axis.
    axis_to_slice = current_config.param_scan_axis

  for i, hf_path in enumerate(hf_target_paths):
    if hf_path not in shape_map_local:
      max_logging.log(
          f"Warning: HF path '{hf_path}' not found in shape_map for MaxText key '{maxtext_param_key}'. Skipping."
      )
      continue

    weight_slice = jax.lax.index_in_dim(leaf_value, i, axis=axis_to_slice, keepdims=False)
    _process(hf_path, weight_slice, output_weights, current_hook_fns, shape_map_local)

  return output_weights


def convert_jax_weight_to_numpy(weight: "jax.Array", dtype_str: None | str = None) -> np.ndarray:
  """Converts a JAX array to a NumPy array with the specified dtype."""
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


def apply_hook_fns(weight, target_shape, hook_fns):
  if hook_fns is None:
    return weight
  if not isinstance(hook_fns, list):
    hook_fns = [hook_fns]
  for hook_fn in hook_fns[::-1]:
    weight = hook_fn(weight, target_shape)
  return weight


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
