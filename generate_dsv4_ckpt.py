# Copyright 2026 Google LLC
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

"""Script to generate and save a dummy checkpoint for DeepSeek-V4-Flash model."""

import sys
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.common import checkpointing

def get_path_string(path):
  keys = []
  for p in path:
    if hasattr(p, "key"):
      keys.append(str(p.key))
    elif hasattr(p, "idx"):
      keys.append(str(p.idx))
    else:
      keys.append(str(p).strip("'\""))
  return "/".join(keys)

def main():
  # Initialize the configuration for deepseek_v4-flash
  argv = sys.argv
  if len(argv) < 2:
    argv = ['', 'src/maxtext/configs/base.yml', 'model_name=deepseek_v4-flash', 'override_model_config=True', 'skip_jax_distributed_system=True', 'weight_dtype=bfloat16', 'scan_layers=False', 'num_experts=8']
  
  print("Initializing configuration...")
  config = pyconfig.initialize(argv)
  
  print("Creating device mesh...")
  mesh = maxtext_utils.get_mesh_from_config(config)
  
  print("Creating model architecture...")
  model = model_creation_utils.create_model(config, mesh)
  
  print("Getting abstract parameters...")
  abstract_vars = maxtext_utils.get_abstract_param(model, config)
  abstract_params = abstract_vars["params"]
  
  print("Materializing sharded zero arrays...")
  def materialize_sharded_zero_array(x):
    sharding = getattr(x, 'sharding', None)
    arr = jnp.zeros(x.shape, dtype=x.dtype)
    if sharding is not None:
      arr = jax.device_put(arr, sharding)
    return arr

  params = jax.tree.map(materialize_sharded_zero_array, abstract_params)
  
  print(f"Saving dummy checkpoint to {config.checkpoint_dir}...")
  checkpointing.save_params_to_path(
      config.checkpoint_dir,
      params,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3
  )
  
  print("\n--- Key-Value Pairs (Weight Names and Shapes) ---")
  flat_params = jax.tree_util.tree_flatten_with_path(abstract_params)[0]
  for path, x in flat_params:
    name = get_path_string(path)
    print(f"{name}: {x.shape}")
  print("-------------------------------------------------\n")
  
  print("Checkpoint generated and saved successfully.")

if __name__ == "__main__":
  main()
