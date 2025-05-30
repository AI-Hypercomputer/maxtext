"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import contextlib
import time
import os
import jax
import jax.numpy as jnp 
from jaxtyping import Array
import numpy as np
import json
from jax.experimental import multihost_utils
from typing import Optional, List, Dict, Tuple, Callable, Any
from google.cloud.storage import Client, transfer_manager
from safetensors.numpy import save_file as numpy_save_file
from huggingface_hub import HfApi, repo_exists
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from concurrent.futures import ThreadPoolExecutor 



SAFE_TENSORS_CONFIG_FILE = "config.json"
SAFE_TENSORS_WEIGHTS_FILE = "model.safetensors"
SAFE_TENSORS_INDEX_FILE = "model.safetensors.index.json"
DEFAULT_MAX_SHARD_SIZE = 1024 * 1024 * 1024 * 3  # 3GB default


def _get_local_directory(output_dir: str) -> str:
    """Determines the local directory for saving files."""
    # This function is largely superseded by get_local_save_path_manager
    # but kept for now if any other part of the codebase uses it directly.
    # For save_model_files, get_local_save_path_manager is preferred.
    if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
        # Fallback to a generic temp directory name if used directly
        local_dir = os.path.join(os.path.expanduser("~/.cache/maxtext_hf_conversion_temp"), "temp_files")
    else:
        local_dir = output_dir
    os.makedirs(local_dir, exist_ok=True)
    return local_dir

def convert_jax_weight_to_numpy(
    weight: "jax.Array", dtype_str: Optional[str] = None
) -> np.ndarray:
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

    return np_array.reshape(expected_shape) # Reshape for safety, though usually preserved.

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
        print(f"\n Created new HuggingFace Hub {repo_type} repo: {repo_id}.")

def save_config_file(
    config, 
    local_path_to_save_to: str, 
    output_dir_final: str, 
    file_name: str, 
    remove_local_copy_after_upload: bool = False
):
    """Saves the model configuration file(config.json)."""
    if jax.process_index() == 0:
        actual_local_file_path = os.path.join(local_path_to_save_to, file_name)
        config.architectures = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]]
        config.to_json_file(actual_local_file_path)
        print(f"   Saved {file_name} to {actual_local_file_path}")

        if output_dir_final.startswith("gs://"):
            upload_file_to_gcs(
                actual_local_file_path,
                os.path.join(output_dir_final, file_name),
                remove_local_file_after_upload=remove_local_copy_after_upload,
            )
        elif output_dir_final.startswith("hf://"):
            repo_id = output_dir_final.lstrip("hf://")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=actual_local_file_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
            )
            if remove_local_copy_after_upload:
                os.remove(actual_local_file_path)
                print(f"   Removed local copy: {actual_local_file_path}")

def shard_checkpoint(
    weights_dict: Dict[str, Array],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    weights_name: str = "model.safetensors",
) -> Tuple[Dict[str, Dict[str, Array]], Optional[Dict]]:
    """Shards a model checkpoint into smaller pieces based on size constraints.

    Args:
        weights_dict: Model weights dictionary to shard
        max_shard_size: Maximum size in bytes for each shard
        weights_name: Base filename for the shards

    Returns:
        Tuple of (sharded weights dict, optional index dict)
        Index contains metadata and weight mapping information
    """
    # Track current shard and accumulated sizes
    current_shard: Dict[str, Array] = {}
    shards: List[Dict[str, Array]] = [current_shard]
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
        shard_name = weights_name.replace(
            ".safetensors", f"-{idx:05d}-of-{len(shards):05d}.safetensors"
        )
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
    remove_local_copy_after_upload: bool = False
):
    """Saves a single safetensor file."""
    if jax.process_index() == 0:
        state_dict = {k: v for k, v in state_dict.items() if v is not None}
        local_path = os.path.join(local_dir_to_save_to, file_name)
        numpy_save_file(state_dict, local_path, metadata={"format": "pt"})
        print(f"   Saved {file_name} to {local_path}")

        if output_dir_final.startswith("gs://"):
            cloud_path = os.path.join(output_dir_final, file_name)
            upload_file_to_gcs(
                local_path, cloud_path, 
                remove_local_file_after_upload=remove_local_copy_after_upload
            )
        elif output_dir_final.startswith("hf://"):
            repo_id = output_dir_final.lstrip("hf://")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model"
            )
            if remove_local_copy_after_upload:
                os.remove(local_path)
                print(f"   Removed local copy: {local_path}")

def save_index_file(
    index: dict, 
    local_dir_to_save_to: str, 
    output_dir_final: str, 
    file_name: str, 
    remove_local_copy_after_upload: bool = False
):
    """Saves the model index json file (model.safetensors.index.json)."""
    if jax.process_index() == 0:
        local_path = os.path.join(local_dir_to_save_to, file_name)
        with open(local_path, "w") as f:
            json.dump(index, f)
        print(f"   Saved {file_name} to {local_path}")

        if output_dir_final.startswith("gs://"):
            upload_file_to_gcs(
                local_path,
                os.path.join(output_dir_final, file_name),
                remove_local_file_after_upload=remove_local_copy_after_upload,
            )
        elif output_dir_final.startswith("hf://"):
            repo_id = output_dir_final.lstrip("hf://")
            api = HfApi()
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
            )
            if remove_local_copy_after_upload:
                os.remove(local_path)
                print(f"   Removed local copy: {local_path}")

def save_weight_files(
    shards, 
    index, 
    local_dir_to_save_to: str, 
    output_dir_final: str, 
    parallel_threads=8, 
    remove_local_copy_after_upload: bool = False
):
    """Saves weight files and index if needed.

    Requires local system to have at least `parallel_threads * DEFAULT_MAX_SHARD_SIZE`
    free disk space, as each thread will maintain a local cache of its shard during processing.
    """
    if index is None:
        # 'shards' is actually the single state_dict here
        save_safetensor_file(
            shards, local_dir_to_save_to, output_dir_final, 
            SAFE_TENSORS_WEIGHTS_FILE, remove_local_copy_after_upload
        )
    else:
        # Save sharded weights in parallel
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            shard_items = list(shards.items())
            futures = [
                executor.submit(
                    save_safetensor_file, shard_dict, local_dir_to_save_to, 
                    output_dir_final, shard_name, remove_local_copy_after_upload
                )
                for shard_name, shard_dict in shard_items
            ]
            for future in futures:
                future.result()

        # Save index file
        save_index_file(
            index, local_dir_to_save_to, output_dir_final, 
            SAFE_TENSORS_INDEX_FILE, remove_local_copy_after_upload
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
    import tempfile # Local import to keep it contained
    if output_dir.startswith("gs://") or output_dir.startswith("hf://"):
        with tempfile.TemporaryDirectory(prefix="maxtext_hf_save_") as temp_dir:
            print(f"   Using temporary local staging directory: {temp_dir}")
            yield temp_dir, True # path, is_temporary
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"   Using local directory: {output_dir}")
        yield output_dir, False # path, is_temporary

def save_model_files(
    weight_arrays: Dict, 
    config,  # HF config object
    tokenizer: Optional[Any], # transformers.PreTrainedTokenizerBase
    output_dir: str, 
    parallel_threads=8
):
    """Saves model files (config and weights) to the specified directory."""

    if output_dir.startswith("hf://"):
        create_huggingface_hub_repo_if_not_exist(
            repo_id=output_dir.lstrip("hf://"), repo_type="model"
        )
        repo_id = output_dir.lstrip("hf://")
    else:
        repo_id = None

    print(f"\n-> Saving model and tokenizer (if provided) to {output_dir}...")

    with get_local_save_path_manager(output_dir) as (current_save_path, is_temp_path):
        remove_local_copy = is_temp_path

        if jax.process_index() == 0:
            # Save tokenizer files
            if tokenizer is not None:
                print(f"   Saving tokenizer files to {current_save_path}...")
                # save_pretrained returns a tuple of saved file names (full paths)
                saved_tokenizer_files = tokenizer.save_pretrained(current_save_path)
                print(f" Tokenizer files saved locally: {saved_tokenizer_files}")

                if output_dir.startswith("gs://"):
                    for local_file_path in saved_tokenizer_files:
                        if not os.path.exists(local_file_path):
                            print(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to GCS.")
                            continue
                        file_name = os.path.basename(local_file_path)
                        upload_file_to_gcs(
                            local_file_path,
                            os.path.join(output_dir, file_name),
                            remove_local_file_after_upload=remove_local_copy,
                        )
                elif output_dir.startswith("hf://") and repo_id:
                    api = HfApi()
                    for local_file_path in saved_tokenizer_files:
                        if not os.path.exists(local_file_path):
                            print(f"   Warning: Tokenizer file {local_file_path} not found locally. Skipping upload to HF Hub.")
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
                            print(f"   Removed local copy: {local_file_path}")
            
            # Save config.json
            save_config_file(config, current_save_path, output_dir, 
                             SAFE_TENSORS_CONFIG_FILE, remove_local_copy)

        # Save .safetensors files (sharding can be outside process guard if weights are replicated)
        # The actual file saving within save_weight_files is guarded.
        shards, index = shard_checkpoint(weight_arrays)
        save_weight_files(shards, index, current_save_path, output_dir, 
                          parallel_threads, remove_local_copy)

    if jax.process_index() == 0:
        print(f"✅ Model and tokenizer (if provided) successfully processed for {output_dir}")

def upload_file_to_gcs(
    local_file: str, gs_bucket_path: str, remove_local_file_after_upload=False
):
    """Uploads a single file to Google Cloud Storage.

    Args:
        local_file: Path to local file
        gs_bucket_path: GCS destination (e.g. "gs://my-bucket/path/file.txt" or "my-bucket/path/file.txt")
    """
    # Standardize bucket path format
    gs_bucket_path = gs_bucket_path.removeprefix("gs://")
    bucket_name = gs_bucket_path.split("/")[0]
    blob_name = gs_bucket_path[len(bucket_name) :].lstrip("/")

    print(f"-> Uploading {local_file} to {gs_bucket_path}...")
    # Upload file
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # set large timeout to support large safetensors
    blob.upload_from_filename(local_file, timeout=600)

    print(f"✅ Uploaded {local_file} to {bucket.name}/{blob_name}")

    if remove_local_file_after_upload:
        os.remove(local_file)
        print(f"✅ Deleted {local_file}")


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
    # Ensure destination ends with "/"
    destination_dir = gs_bucket_path[len(bucket_name) :]
    if destination_dir.startswith("/"):
        destination_dir = destination_dir[1:]
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
            print(f"Failed to upload {name}: {result}")
        else:
            print(f"✅ Uploaded {name} to {bucket.name}/{destination_dir}{name}")

    print(f"Upload completed in {time.time() - start_time}s")