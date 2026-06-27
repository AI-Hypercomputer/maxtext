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

"""Dynamic loading of HuggingFace checkpoints during training/eval workloads directly in the target format."""

import concurrent.futures
import gc
import multiprocessing
import os
import random
import time
import numpy as np

from flax import nnx
from flax import traverse_util
from google.cloud import storage
from huggingface_hub import HfFileSystem
import jax
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils

from maxtext.utils import max_logging
from maxtext.utils.globals import HF_IDS
from maxtext.checkpoint_conversion.utils.tensor_handling import get_hf_loading_function
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS


def build_gcs_cache_worker(fpath, gcs_cache_dir, hf_access_token):
    fs = HfFileSystem(token=hf_access_token)
    time.sleep(random.uniform(0.0, 5.0))
    
    bucket_name = gcs_cache_dir.replace("gs://", "").split("/")[0]
    blob_prefix = gcs_cache_dir.replace("gs://", "").split("/", 1)[1] if "/" in gcs_cache_dir.replace("gs://", "") else ""
    blob_name = os.path.join(blob_prefix, os.path.basename(fpath))
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        max_logging.log(f"[Worker] Cache hit for {os.path.basename(fpath)}.")
        return
        
    t0 = time.time()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with fs.open(fpath, "rb") as remote_f:
                blob.chunk_size = 1024 * 1024 * 32  # 32MB chunks
                blob.upload_from_file(remote_f, client=storage_client)
            print(f"[Worker] Cached {os.path.basename(fpath)} in {time.time() - t0:.1f}s", flush=True)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                max_logging.log(f"Error fetching {fpath} to GCS: {e}. Retrying in 15 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(15)
            else:
                max_logging.log(f"Failed to fetch {fpath} to GCS after {max_retries} attempts.")
                raise


def get_hf_config_and_mappings(config):
  """Gets HF config and parameter mapping based on the MaxText config."""
  model_key = config.model_name
  if "-Instruct" in model_key:
    model_key = model_key.replace("-Instruct", "")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()

  param_map_mt_to_hf = param_mapping.PARAM_MAPPING[model_key](
      hf_config_dict, config, scan_layers=config.scan_layers
  )
  hook_fn_map_mt = param_mapping.HOOK_FNS[model_key](
      hf_config_dict, config, scan_layers=config.scan_layers, saving_to_hf=False
  )
  return param_map_mt_to_hf, hook_fn_map_mt


def load_sharded_hf_state(path, devices=None):
  """Loads HF state with maximal sharding across TPU mesh to avoid host OOM."""
  t0 = time.time()
  context = ocp_v1.Context(
      checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS,
      safetensors_options=ocp_v1.options.SafetensorsOptions(ignore_load_sharding=False),
  )
  with context:
    metadata = ocp_v1.metadata(path)
    simple_abstract_state = metadata.metadata
    
    # Distributed Sharded Download: Tell JAX to shard the HF Safetensors download 
    # across the entire TPU mesh to avoid Host OOM.
    current_global_devices = devices if devices is not None else jax.devices()
    shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state, devices=current_global_devices)

    def combine_sharding(sds, single_sharding):
      return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=single_sharding)

    sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
    
    max_logging.log("Reading raw Safetensors into memory (Distributed Sharded GCS Download)...")
    hf_state = ocp_v1.load(path, sharded_abstract_state)
    max_logging.log(f"load_sharded_hf_state took {time.time() - t0:.2f}s")
    return hf_state


def transform_hf_state_to_mt_state(
    hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, config
):
  """Transforms HF state into MaxText state by applying param mappings and mathematical hooks."""
  t0 = time.time()
  def tensor_getter(key):
    return hf_state.pop(key)

  flat_target = traverse_util.flatten_dict(target_tree, sep=".")
  flat_restored = flat_target.copy()

  mapped_count = 0
  keys_missed = []
  max_logging.log("Starting fast in-memory Distributed Transformations...")

  for mt_key, hf_source in param_map_mt_to_hf.items():
    mt_name = mt_key.replace("params-", "").replace("-", ".")
    
    # Determine the correct key in flat_target
    check_name = mt_name
    if check_name not in flat_target:
        if ("params." + mt_name) in flat_target:
            check_name = "params." + mt_name
        elif mt_key.replace("-", ".") in flat_target:
            check_name = mt_key.replace("-", ".")
    
    if check_name not in flat_target:
      keys_missed.append(mt_name)
      continue
      
    target_shape = flat_target[check_name].shape
    hook_fn = hook_fn_map_mt.get(mt_key)

    load_fn = get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_shape,
        config,
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    unsharded_array = load_fn()
    
    # Ensure it's Sharded explicitly matching the JAX model expectations
    target_sharding = flat_target[check_name].sharding
    
    if isinstance(unsharded_array, jax.Array):
      if target_sharding.device_set == unsharded_array.sharding.device_set:
        max_logging.log(f"Loaded {check_name} via TPU-to-TPU direct resharding.")
        flat_restored[check_name] = jax.device_put(unsharded_array, device=target_sharding)
      else:
        max_logging.log(f"Loaded {check_name} via JAX JIT TPU-to-TPU resharding.")
        flat_restored[check_name] = jax.jit(
            lambda x: x, out_shardings=target_sharding
        )(unsharded_array)
    else:
      if jax.process_count() > 1 and not target_sharding.is_fully_addressable:
        max_logging.log(f"Loaded {check_name} via Host CPU callback fallback (NumPy array).")
        flat_restored[check_name] = jax.make_array_from_callback(
            unsharded_array.shape, target_sharding, lambda index, source=unsharded_array: source[index]
        )
      else:
        max_logging.log(f"Loaded {check_name} via Host CPU device_put.")
        flat_restored[check_name] = jax.device_put(unsharded_array, device=target_sharding)
    del unsharded_array
    
    max_logging.log(f"Transformed {check_name} from {hf_source} in {time.time() - t_layer:.4f}s")
    mapped_count += 1
    
    if mapped_count % 10 == 0:
        gc.collect()
      
  if mapped_count == 0:
    max_logging.log(f"All transformations missed! Sample missed mt_names: {keys_missed[:5]}")
    max_logging.log(f"Sample flat_target keys: {list(flat_target.keys())[:5]}")
  
  max_logging.log(f"Successfully mapped {mapped_count} parameters.")
  restored_params = traverse_util.unflatten_dict(flat_restored, sep=".")
  
  if "params" in restored_params:
      restored_params = restored_params["params"]
  
  max_logging.log(f"transform_hf_state_to_mt_state took {time.time() - t0:.2f}s")
  
  return {"params": restored_params}


def _get_global_mesh(target_tree):
  flat_target = traverse_util.flatten_dict(target_tree, sep=".")
  for val in flat_target.values():
    if hasattr(val, "sharding") and val.sharding is not None:
      return val.sharding.mesh
  return None


def load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, config):
  """Main entry point to dynamically build and load safetensors into MaxText format.
  
  Splits execution into:
  1. Deriving Mappings
  2. Loading Sharded arrays directly to TPUs
  3. Processing the transformations natively on TPUs
  """
  if config is None:
    raise ValueError("config must be provided for safetensors_dynamic loading.")
  
  model_name = config.model_name
  if "-Instruct" in model_name:
    model_name = model_name.replace("-Instruct", "")

  if not path:
    if model_name not in HF_IDS:
      raise ValueError(f"Unsupported model name for automatic HF repo resolution: {model_name}.")
    path = HF_IDS[model_name]

  if path.startswith("hf://"):
    path = path[5:]

  if not path.startswith("gs://") and not os.path.isdir(path):
    fs = HfFileSystem(token=config.hf_access_token)
    repo_id = path
    
    files = fs.glob(f"{repo_id}/*.safetensors")
    
    host_id = jax.process_index()

    if hasattr(config, "base_output_directory") and config.base_output_directory.startswith("gs://"):
        gcs_cache_dir = f"{config.base_output_directory}/hf_cache/{repo_id.replace('/', '_')}"
        path = gcs_cache_dir
        
        # Only Host 0 downloads to the shared GCS cache
        if host_id == 0:            
            max_logging.log(f"Dynamic HF Hub Fast DL: Host 0 is downloading to shared GCS Cache: {gcs_cache_dir}")
            t_gcs_start = time.time()
            
            # List existing blobs to avoid spawning processes for already cached files
            storage_client = storage.Client()
            bucket_name = gcs_cache_dir.replace("gs://", "").split("/")[0]
            blob_prefix = gcs_cache_dir.replace("gs://", "").split("/", 1)[1] if "/" in gcs_cache_dir.replace("gs://", "") else ""
            
            existing_blobs = {blob.name for blob in storage_client.list_blobs(bucket_name, prefix=blob_prefix)}
            
            files_to_download = []
            for fpath in files:
                expected_blob_name = os.path.join(blob_prefix, os.path.basename(fpath))
                if expected_blob_name not in existing_blobs:
                    files_to_download.append(fpath)
            
            if files_to_download:
                with concurrent.futures.ProcessPoolExecutor(max_workers=32, mp_context=multiprocessing.get_context("spawn")) as executor:
                    futures = [
                        executor.submit(build_gcs_cache_worker, fpath, gcs_cache_dir, config.hf_access_token) 
                        for fpath in files_to_download
                    ]
                    
                    while futures:
                        done, futures = concurrent.futures.wait(futures, timeout=10)
                        
                        # Raise any exceptions if a worker failed
                        for f in done:
                            f.result()
                            
            t_gcs_end = time.time()
            max_logging.log(f"GCS caching complete in {t_gcs_end - t_gcs_start:.2f}s. Downloaded {len(files_to_download)} missing files.")
            
        # Global barrier: all hosts wait for Host 0 to finish downloading to the shared GCS bucket
        max_logging.log(f"Host {host_id} waiting for GCS cache at {gcs_cache_dir} to be populated by Host 0...")
        jax.experimental.multihost_utils.sync_global_devices("dynamic_hf_download_complete")
        max_logging.log(f"Host {host_id} detected GCS cache is ready!")

    else:
        raise ValueError("base_output_directory with gs:// prefix is required for huggingface downloads.")

  t_total = time.time()
  param_map_mt_to_hf, hook_fn_map_mt = get_hf_config_and_mappings(config)
  max_logging.log(f"[1/3] Mappings derived in {time.time() - t_total:.2f}s")

  target_tree = (
      abstract_unboxed_pre_state.to_pure_dict()
      if isinstance(abstract_unboxed_pre_state, nnx.State)
      else abstract_unboxed_pre_state.params
  )
  
  t1 = time.time()
  hf_state = load_sharded_hf_state(path, devices=None)
  max_logging.log(f"[2/3] Distributed Sharded GCS load completed in {time.time() - t1:.2f}s")
  
  t2 = time.time()
  restored_params = transform_hf_state_to_mt_state(
      hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, config
  )
  max_logging.log(f"[3/3] CPU Transformations completed in {time.time() - t2:.2f}s")
  max_logging.log(f"Total safetensors_dynamic duration: {time.time() - t_total:.2f}s")
  
  return None, restored_params

