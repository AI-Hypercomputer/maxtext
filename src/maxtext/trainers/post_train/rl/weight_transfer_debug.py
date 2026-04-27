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

"""
Weight Transfer Benchmark for the RL Trainer.
Similar to reshard_debug.py but only considers weight transfer.
"""

from __future__ import annotations

import os

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import time
import pathwaysutils


from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.utils import max_logging, max_utils
from tunix.rl import reshard
from tunix.sft.utils import show_hbm_usage

import argparse

def main(args):
  print("DEBUG: Entering main", flush=True)
  pathwaysutils.initialize()
  print("DEBUG: After pathwaysutils.initialize()", flush=True)
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  # max_utils.print_system_information()
  print("DEBUG: Skipped print_system_information", flush=True)

  devices = jax.devices()
  print(f"DEBUG: Found {len(devices)} devices", flush=True)
  req_devices = args.device_size * 2
  if len(devices) < req_devices:
    raise ValueError(f"Need at least {req_devices} devices for this benchmark, got {len(devices)}")

  max_logging.log(f"Using specific {args.device_size}/{args.device_size} split for trainer/sampler on {req_devices} chips.")
  trainer_devices = devices[:args.device_size]
  sampler_devices = devices[args.device_size:req_devices]

  import numpy as np
  trainer_mesh = Mesh(np.array(trainer_devices).reshape(args.device_size), ('tensor',))
  sampler_mesh = Mesh(np.array(sampler_devices).reshape(args.device_size), ('tensor',))

  max_logging.log(f"Trainer mesh shape: {trainer_mesh.shape}")
  max_logging.log(f"Sampler mesh shape: {sampler_mesh.shape}")

  # Define array size
  total_bytes = args.data_size_gb * 1024 * 1024 * 1024
  element_size = 2  # bfloat16
  total_elements = total_bytes // element_size

  num_trainer_chips = args.device_size
  elements_per_chip = total_elements // num_trainer_chips

  max_logging.log(f"Total elements: {total_elements}")
  max_logging.log(f"Elements per chip: {elements_per_chip}")

  # Create a 2D array where axis 0 is sharded across trainer chips (size 8)
  shape = (num_trainer_chips, elements_per_chip)
  
  # Shard only across ici_tensor_parallelism ('tensor' axis)
  trainer_sharding = NamedSharding(trainer_mesh, P('tensor', None))
  sampler_sharding = NamedSharding(sampler_mesh, P('tensor', None))

  key = jax.random.PRNGKey(42)

  # Create data on trainer mesh
  with trainer_mesh:
    def _create_data(k):
      return jax.random.uniform(k, shape, dtype=jnp.bfloat16)
    create_data = jax.jit(_create_data, out_shardings=trainer_sharding)
    data_trainer = create_data(key)
    jax.block_until_ready(data_trainer)
    max_logging.log("Created data on trainer mesh.")
    show_hbm_usage("HBM after data creation on trainer:")

  # Benchmark transfer
  num_loops = 10
  times = []

  for i in range(num_loops):
    start_time = time.time()
    # Transfer using reshard_pytree
    data_rollout = reshard.reshard_pytree(data_trainer, sampler_sharding)
    jax.block_until_ready(data_rollout)
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    max_logging.log(f"Loop {i}: Transfer time: {elapsed:.4f}s")

  max_logging.log(f"Weight Transfer Benchmark completed.")
  max_logging.log(f"Average Transfer Time: {sum(times)/len(times):.4f}s")
  max_logging.log(f"Max Transfer Time: {max(times):.4f}s")
  max_logging.log(f"Min Transfer Time: {min(times):.4f}s")

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size_gb', type=int, default=10, help='Total data size in GiB')
    parser.add_argument('--device_size', type=int, default=4, help='Number of devices for both trainer and sampler')
    args, unknown = parser.parse_known_args(sys.argv[1:])
    main(args)
