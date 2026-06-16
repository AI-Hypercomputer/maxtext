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
This script re-shards a MaxText checkpoint on CPU, assuming linen format.
- The Orbax checkpoint is streamed from storage directly into the target sharded layout on a simulated CPU mesh,
  and then saved to a new checkpoint.
- The goal is to pre-shard checkpoints (source) to accelerate loading on TPUs (target) by reducing re-sharding overhead.
  E.g., when target sharding is fsdp=64, checkpoint loading time varies across source sharding (fsdp=16 < ep=16)

Key Parameters:
- `--simulated_cpu_devices_count` (defaults to 16). Examples:
  - **Suitable for most cases**: `--simulated_cpu_devices_count=16 ici_fsdp_parallelism=16`
  - More customization: `--simulated_cpu_devices_count=32 ici_fsdp_parallelism=16 ici_expert_parallelism=2`
- `weight_dtype`: The dtype used to load and save the checkpoint. **Highly recommend** using `weight_dtype=bfloat16`.
- `load_parameters_path`: The input checkpoint path (GCS or local).
- `base_output_directory`: The output directory (GCS or local).
  - The output checkpoint path will be `<base_output_directory>/0/items`

Memory Requirements:
- For X billion parameters, needs slightly over 2X GB RAM (each param takes 2 bytes with `weight_dtype=bfloat16`).
- Note: We only hold one model copy in memory, as the re-sharding happens dynamically during the read operation.
  Additional buffer memory is needed mainly for the I/O streaming overhead, usually small compared to model weight.
- Example: DeepSeek-V3 with MTP layers has 685B parameters, uses 1.37 TB for weights, and hits a peak RAM of ~1.45 TB.

Example Commands:

python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \
  model_name=deepseek2-16b attention=dot_product mla_naive_kvcache=false \
  scan_layers=True load_parameters_path=<input_ckpt_path> \
  base_output_directory=<output_ckpt_dir> \
  weight_dtype=bfloat16 \
  checkpoint_storage_concurrent_gb=1024 checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \
  skip_jax_distributed_system=True ici_fsdp_parallelism=16 \
  --simulated_cpu_devices_count=16

python3 -m maxtext.checkpoint_conversion.reshard_checkpoint \
  model_name=deepseek3-671b mtp_num_layers=1 mtp_loss_scaling_factor=0.1 attention=dot_product mla_naive_kvcache=false \
  scan_layers=True load_parameters_path=<input_ckpt_path> \
  base_output_directory=<output_ckpt_dir> \
  weight_dtype=bfloat16 \
  checkpoint_storage_concurrent_gb=1024 checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \
  skip_jax_distributed_system=True ici_fsdp_parallelism=16 ici_expert_parallelism=2 \
  --simulated_cpu_devices_count=32
"""


import argparse
import os
import sys
import time
from typing import Sequence
from absl import app

import jax
from flax.training import train_state

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils, max_logging
from maxtext.common import checkpointing
from maxtext.checkpoint_conversion.utils.utils import print_peak_memory


def main(argv: Sequence[str]) -> None:
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()
  max_logging.log(f"Load and save checkpoint with weight dtype: {config.weight_dtype}")

  # 1. Engine sets up the mesh based on config
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)

  # 2. Load parameters and reshard with the mesh
  start = time.time()
  params = engine.load_params(rng_load_params)
  max_logging.log(f"Elapse for checkpoint load (with reshard): {(time.time() - start) / 60:.2f} min")

  # 3. Save checkpoint
  start = time.time()
  save_ckpt_directory = config.base_output_directory

  # Dummy configs for the checkpoint_manager
  step_number = 0
  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      save_ckpt_directory,
      enable_checkpointing,
      async_checkpointing,
      save_interval_steps,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )
  if checkpoint_manager is None:
    raise RuntimeError("Failed to create Orbax checkpoint manager.")

  state_new = train_state.TrainState(
      step=step_number, apply_fn=None, params=params, tx=None, opt_state={}  # type: ignore
  )

  if checkpointing.save_checkpoint(checkpoint_manager, step_number, state_new):
    save_ckpt_path = os.path.join(save_ckpt_directory, str(step_number), "items")
    max_logging.log(f"Saved checkpoint: {save_ckpt_path}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    checkpoint_manager.wait_until_finished()

  max_logging.log(f"Elapse for checkpoint save: {(time.time() - start) / 60:.2f} min")
  print_peak_memory()


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Suppress TensorFlow logging

  # Define local parser
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--simulated_cpu_devices_count",
      type=int,
      required=False,
      default=16,
      help="Number of simulated CPU devices for sharding the checkpoint",
  )

  # Parse known args returns the namespace AND the list of remaining arguments
  local_args, remaining_args = parser.parse_known_args()

  # Reconstruct model_args (script name + the args MaxText needs)
  model_args = [sys.argv[0]] + remaining_args

  # Set JAX environment
  jax.config.update("jax_platforms", "cpu")
  # Simulate CPU devices as virtual mesh
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={local_args.simulated_cpu_devices_count}"

  app.run(main, argv=model_args)
