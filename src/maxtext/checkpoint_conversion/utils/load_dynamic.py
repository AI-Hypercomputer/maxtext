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

import jax
from flax import traverse_util
from flax import nnx
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils

from maxtext.utils import max_logging
from maxtext.checkpoint_conversion.utils.tensor_handling import _get_hf_loading_function
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
import time


def get_hf_config_and_mappings(maxtext_config):
  """Gets HF config and parameter mapping based on the MaxText config."""
  model_key = maxtext_config.model_name
  if "-Instruct" in model_key:
    model_key = model_key.replace("-Instruct", "")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()

  param_map_mt_to_hf = param_mapping.PARAM_MAPPING[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers
  )
  hook_fn_map_mt = param_mapping.HOOK_FNS[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers, saving_to_hf=False
  )
  return param_map_mt_to_hf, hook_fn_map_mt


def load_sharded_hf_state(path):
  """Loads HF state with maximal sharding across TPU mesh to avoid host OOM."""
  t0 = time.time()
  context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
  with context:
    metadata = ocp_v1.pytree_metadata(path)
    simple_abstract_state = metadata.metadata
    
    # Distributed Sharded Download: Tell JAX to shard the HF Safetensors download 
    # across the entire TPU mesh to avoid Host OOM.
    current_global_devices = jax.devices()
    shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state, devices=current_global_devices)

    def combine_sharding(sds, single_sharding):
      return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=single_sharding)

    sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
    
    max_logging.log("Reading raw Safetensors into memory (Distributed Sharded GCS Download)...")
    hf_state = ocp_v1.load_pytree(path, sharded_abstract_state)
    max_logging.log(f"load_sharded_hf_state took {time.time() - t0:.2f}s")
    return hf_state


def transform_hf_state_to_mt_state(
    hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
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

    load_fn = _get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_shape,
        maxtext_config,
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    unsharded_array = load_fn()
    
    # Ensure it's Sharded explicitly matching the JAX model expectations
    target_sharding = flat_target[check_name].sharding
    flat_restored[check_name] = jax.device_put(unsharded_array, device=target_sharding, donate=True)
    
    max_logging.log(f"Transformed {check_name} from {hf_source} in {time.time() - t_layer:.4f}s")
    mapped_count += 1
      
  if mapped_count == 0:
    max_logging.log(f"All transformations missed! Sample missed mt_names: {keys_missed[:5]}")
    max_logging.log(f"Sample flat_target keys: {list(flat_target.keys())[:5]}")
  
  max_logging.log(f"Successfully mapped {mapped_count} parameters.")
  restored_params = traverse_util.unflatten_dict(flat_restored, sep=".")
  
  if "params" in restored_params:
      restored_params = restored_params["params"]
  
  max_logging.log(f"transform_hf_state_to_mt_state took {time.time() - t0:.2f}s")
  
  return {"params": restored_params}


def load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config):
  """Main entry point to dynamically build and load safetensors into MaxText format.
  
  Splits execution into:
  1. Deriving Mappings
  2. Loading Sharded arrays directly to TPUs
  3. Processing the transformations natively on TPUs
  """
  if maxtext_config is None:
    raise ValueError("maxtext_config must be provided for safetensors_dynamic loading.")
  
  import os
  from maxtext.utils.globals import HF_IDS

  model_name = maxtext_config.model_name
  if "-Instruct" in model_name:
    model_name = model_name.replace("-Instruct", "")

  if not path:
    if model_name not in HF_IDS:
      raise ValueError(f"Unsupported model name for automatic HF repo resolution: {model_name}.")
    path = HF_IDS[model_name]

  if path.startswith("hf://"):
    path = path[5:]

  if not path.startswith("gs://") and not os.path.isdir(path):
    from huggingface_hub import HfFileSystem
    import concurrent.futures
    import json
    import jax
    
    fs = HfFileSystem(token=maxtext_config.hf_access_token)
    repo_id = path
    
    files = fs.glob(f"{repo_id}/*.safetensors")
    local_dir = f"/tmp/hf_checkpoints/{repo_id.replace('/', '_')}"
    os.makedirs(local_dir, exist_ok=True)
    
    process_count = max(1, jax.process_count())
    host_id = jax.process_index()
    HEADER_NUM_BYTES = 8
    
    max_logging.log(f"Dynamic HF Hub Fast DL: Resolving metadata and partial chunks via HTTP Range Requests for Host {host_id}/{process_count}")
    import random
    import time

    def fetch_shard(fpath):
        time.sleep(random.uniform(0.0, 5.0))
        local_path = os.path.join(local_dir, os.path.basename(fpath))
        with fs.open(fpath, "rb") as remote_f:
            header_size_bytes = remote_f.read(HEADER_NUM_BYTES)
            header_size = int.from_bytes(header_size_bytes, byteorder="little")
            header_bytes = remote_f.read(header_size)
            header = json.loads(header_bytes)
            
            data_start_offset = HEADER_NUM_BYTES + header_size
            
            tensors = {k: v for k, v in header.items() if k != "__metadata__"}
            sorted_tensors = sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0])
            
            with open(local_path, "wb") as local_f:
                local_f.write(header_size_bytes)
                local_f.write(header_bytes)
                
                if not sorted_tensors:
                    return
                
                total_size = sorted_tensors[-1][1]["data_offsets"][1]
                current_bundle = 0
                cumulative_size = 0
                host_start_offset = None
                host_end_offset = None
                
                for name, info in sorted_tensors:
                    start, end = info["data_offsets"]
                    tensor_size = end - start
                    if current_bundle < process_count - 1:
                        ideal = (current_bundle + 1) * (total_size / process_count)
                        dist_if_cut = abs(cumulative_size - ideal)
                        dist_if_keep = abs((cumulative_size + tensor_size) - ideal)
                        if dist_if_cut < dist_if_keep and cumulative_size > 0:
                            current_bundle += 1
                    
                    if current_bundle == host_id:
                        if host_start_offset is None:
                            host_start_offset = start
                        host_end_offset = end
                        
                    cumulative_size += tensor_size
                
                if host_start_offset is not None:
                    chunk_size = host_end_offset - host_start_offset
                    remote_f.seek(data_start_offset + host_start_offset)
                    local_f.seek(data_start_offset + host_start_offset)
                    
                    buffer_size = 1024 * 1024 * 16
                    bytes_remaining = chunk_size
                    while bytes_remaining > 0:
                        sz = min(buffer_size, bytes_remaining)
                        buf = remote_f.read(sz)
                        if not buf:
                            break
                        local_f.write(buf)
                        bytes_remaining -= len(buf)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fetch_shard, files))
        
    path = local_dir

  t_total = time.time()
  param_map_mt_to_hf, hook_fn_map_mt = get_hf_config_and_mappings(maxtext_config)
  max_logging.log(f"[1/3] Mappings derived in {time.time() - t_total:.2f}s")

  target_tree = (
      abstract_unboxed_pre_state.to_pure_dict()
      if isinstance(abstract_unboxed_pre_state, nnx.State)
      else abstract_unboxed_pre_state.params
  )
  
  t1 = time.time()
  hf_state = load_sharded_hf_state(path)
  max_logging.log(f"[2/3] Distributed Sharded GCS load completed in {time.time() - t1:.2f}s")
  
  t2 = time.time()
  restored_params = transform_hf_state_to_mt_state(
      hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
  )
  max_logging.log(f"[3/3] CPU Transformations completed in {time.time() - t2:.2f}s")
  max_logging.log(f"Total safetensors_dynamic duration: {time.time() - t_total:.2f}s")
  
  return None, restored_params
