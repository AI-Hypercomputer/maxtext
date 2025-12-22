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
JAX_PROCESS_INDEX=0 JAX_PROCESS_IDX=0 JOB_INDEX=0 JOB_COMPLETION_INDEX=0 PROCESSES_IN_JOB=1 JAX_PROCESS_COUNT=1 JAX_PROCESS_INDEX=0 JAX_COORDINATOR_ADDRESS='127.0.0.1' JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_conversion.test_checkpoints src/MaxText/configs/base.yml \
    model_name=mixtral-8x7b \
    use_multimodal=false \
    scan_layers=true hardware=cpu --converted_ckpt=gs://runner-maxtext-logs/ranran/mixtral-8x7b/maxtext_to_hf_unscan_debug5
"""

import argparse
import os
import sys
import json
import threading
from functools import partial
from typing import Sequence, List, Any, Callable
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
from MaxText.inference_utils import str2bool
from MaxText.layers import models, quantizations
from MaxText.checkpointing import save_checkpoint
from MaxText.utils.ckpt_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from MaxText.utils.ckpt_conversion.utils.utils import apply_hook_fns, HF_IDS
from safetensors import safe_open
import torch
import gcsfs
from safetensors.torch import load
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# def load_hf(gcs_path, fs):
#   fs = gcsfs.GCSFileSystem()
#   search_pattern = f"{gcs_path.rstrip('/')}/*.safetensors"
#   safetensor_files = fs.glob(search_pattern)
#   safetensor_files = [f"gs://{f}" for f in safetensor_files]

#   hf_tensor = {}
#   for st_f in safetensor_files:
#     with fs.open(st_f, "rb") as f:
#       file_bytes = f.read()
#     loaded_tensors = load(file_bytes)
#     for key, tensor in loaded_tensors.items():
#       hf_tensor[key] = tensor.numpy()
#     del file_bytes
#     del loaded_tensors
#     print(f"file is read: {st_f}")
#   return hf_tensor

def load_shard(gcs_path, fs):
    """Worker function to read and process a single file."""
    print(f"Start processing: {gcs_path}")
    
    # 1. Read bytes (IO Bound)
    with fs.open(gcs_path, "rb") as f:
        file_bytes = f.read()
    
    # 2. Parse Safetensors (CPU Bound)
    loaded_tensors = load(file_bytes)
    
    # 3. Convert to Numpy
    # Using a dict comprehension is slightly faster than a loop
    shard_dict = {}
    for key, tensor in loaded_tensors.items():
        # Optimization: Only cast if strictly necessary
        if tensor.dtype != torch.float16:
            tensor = tensor.to(torch.float16)
        shard_dict[key] = tensor.numpy()
        
    # Explicit cleanup to help GC in high-concurrency settings
    del file_bytes
    del loaded_tensors
    
    print(f"Finished: {gcs_path}")
    return shard_dict

def load_hf_fast(hf_checkpoint_folder, max_workers=12):
    fs = gcsfs.GCSFileSystem()
    search_pattern = f"{hf_checkpoint_folder.rstrip('/')}/*.safetensors"
    
    # Get file list
    safetensor_files = fs.glob(search_pattern)
    safetensor_files = [f"gs://{f}" for f in safetensor_files]
    
    print(f"Found {len(safetensor_files)} files. Starting parallel load with {max_workers} workers...")

    final_tensor = {}
    
    # 4. Use ThreadPool to download/process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(load_shard, f, fs): f 
            for f in safetensor_files
        }
        
        # Gather results as they finish
        for future in as_completed(future_to_file):
            try:
                shard_data = future.result()
                # 5. Merge into main dictionary
                final_tensor.update(shard_data)
            except Exception as e:
                print(f"generated an exception: {e}")

    return final_tensor

def _get_hf_model(model_id: str, token: str):
  """Loads the HuggingFace model based on model_id (Eager mode only)."""
  # Some models require special classes to import
  if model_id in ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]:
    from transformers import Qwen3OmniMoeForConditionalGeneration  # pylint: disable=import-outside-toplevel

    hf_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, token=token)
  else:
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
  return hf_model


def compare_numpy_dicts(d1, d2, rtol=1e-2, atol=1e-2):
    # 1. Compare Keys
    if d1.keys() != d2.keys():
        print("❌ Keys do not match.")
        print(f"In d1 only: {set(d1.keys()) - set(d2.keys())}")
        print(f"In d2 only: {set(d2.keys()) - set(d1.keys())}")
        return False

    print("✅ Keys match.")
    
    # 2. Compare Values
    for key in d1:
        print(f"checking key: {key}")
        arr1 = d1[key]
        arr2 = d2[key]
        
        # Check Shape
        if arr1.shape != arr2.shape:
            print(f"❌ Shape mismatch for '{key}': {arr1.shape} vs {arr2.shape}")
            break
        print("✅ Key: {key} shape match.")
            
        # Check Values (using allclose for float tolerance)
        if not np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(arr1 - arr2))
            print(f"❌ Value mismatch for '{key}'. Max diff: {max_diff}")
            break
        print("✅ Key: {key} value match.")

    print("✅ All values match!")


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
      # print(f"Key: {k} | Shape: {hf_state_dict_numpy[k].shape}")
    del hf_model
    max_logging.log("HuggingFace model loaded and converted to NumPy.")
    print_ram_usage("After full HF model load")

    print(f"test_args.converted_ckpt: {test_args.converted_ckpt}")
    maxtext_tensor = load_hf_fast(test_args.converted_ckpt)
    compare_numpy_dicts(hf_state_dict_numpy, maxtext_tensor)


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
  parser.add_argument(
      "--converted_ckpt",
      type=str,
      default="",
      help="The original huggingface checkpoint",
  )
  local_args, _ = parser.parse_known_args()
  model_args = sys.argv
  to_remove_args = ["--lazy_load_tensors", "--converted_ckpt"]
  for a in to_remove_args:
    model_args = [s for s in model_args if not s.startswith(a)]
  main(model_args, local_args)
