# Copyright 2023–2026 Google LLC
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
Weight Transfer Benchmark Script

This script benchmarks the latency of transferring a 20GiB array from trainer chips
to sampler chips using Pathways.
"""

from __future__ import annotations
import time
import os
from typing import Sequence
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pathwaysutils
from absl import app
from maxtext.utils import max_logging
from maxtext.configs import pyconfig
from maxtext.utils import max_utils
from tunix.sft.utils import show_hbm_usage

def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  max_utils.print_system_information()

  config = pyconfig.initialize_pydantic(argv)
  
  devices = jax.devices()
  num_devices = len(devices)
  max_logging.log(f"Total devices detected: {num_devices}")
  
  # Default fractions if not provided in config
  trainer_fraction = getattr(config, "trainer_devices_fraction", 0.5)
  sampler_fraction = getattr(config, "sampler_devices_fraction", 0.5)
  transfer_size_gb = getattr(config, "batch_size", 10)
  num_loops = getattr(config, "num_batches", 10)
  
  num_trainer_devices = int(num_devices * trainer_fraction)
  num_sampler_devices = int(num_devices * sampler_fraction)
  
  max_logging.log(f"Allocating {num_trainer_devices} trainer devices and {num_sampler_devices} sampler devices.")
  
  if num_trainer_devices + num_sampler_devices > num_devices:
    raise ValueError("Sum of trainer and sampler devices exceeds total available devices.")
    
  trainer_devices = devices[:num_trainer_devices]
  sampler_devices = devices[num_devices - num_sampler_devices :]
  
  max_logging.log(f"Trainer devices: {[d.id for d in trainer_devices]}")
  max_logging.log(f"Sampler devices: {[d.id for d in sampler_devices]}")
  
  # Create meshes
  # Assuming 1D sharding for simplicity
  trainer_mesh = Mesh(trainer_devices, ('data',))
  sampler_mesh = Mesh(sampler_devices, ('data',))
  
  # 20 GiB in bfloat16
  num_elements = (transfer_size_gb * 1024**3) // 2
  
  # Shape must be divisible by number of trainer devices.
  # We use a shape where the first axis is the number of devices.
  shape = (len(trainer_devices), num_elements // len(trainer_devices))
  
  max_logging.log(f"Creating array of shape {shape} on Trainer mesh...")
  
  show_hbm_usage(f"HBM before loading data to trainer devices:")
  trainer_sharding = NamedSharding(trainer_mesh, P('data', None))
  
  dummy_data = jax.device_put(jnp.ones(shape, dtype=jnp.bfloat16), trainer_sharding)
  jax.block_until_ready(dummy_data)
  show_hbm_usage(f"HBM after loading data to trainer devices:")
    
  max_logging.log("Array created on Trainer mesh.")
  
  sampler_sharding = NamedSharding(sampler_mesh, P('data', None))
  
  # Transfer benchmark
  latencies = []
  
  max_logging.log(f"Starting weight transfer benchmark for {num_loops} loops...")
  
  for i in range(num_loops):
    start_time = time.time()
    transferred_data = jax.device_put(dummy_data, sampler_sharding)
    jax.block_until_ready(transferred_data)
    end_time = time.time()
    
    latency = end_time - start_time
    latencies.append(latency)
    max_logging.log(f"Loop {i+1}: {latency:.4f}s")
    show_hbm_usage(f"HBM after transfer {i+1}:")
    
  avg_latency = sum(latencies) / num_loops
  max_logging.log(f"Average weight transfer latency: {avg_latency:.4f}s")

if __name__ == "__main__":
  app.run(main)
