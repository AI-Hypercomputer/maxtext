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
RL Resharding Debug Script

This script initializes actor and reference models, creates an RLCluster,
and calls sync_weights() to trigger parameter resharding from the actor
mesh to the rollout mesh. Its purpose is to isolate and debug resharding
in Tunix, removing data loading, checkpoint loading, sampling, RL training
and evaluation logic.

It is adapted from the RL training script.

Usage Example:
python3 -m src.MaxText.rl.reshard_debug src/maxtext/configs/rl.yml \
  model_name=llama3.1-8b \
  tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
  run_name=$WORKLOAD \
  base_output_directory=$OUTPUT_PATH \
  hf_access_token=$HF_TOKEN
"""

from __future__ import annotations
from typing import Sequence

import collections
import jax
import json
import logging
import os
import pathwaysutils

from absl import app
from absl import logging as absl_logging
from etils import epath
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh
from transformers import AutoTokenizer
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

# for vLLM we can skip JAX precompilation with this flag, it makes startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

from MaxText import pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from MaxText.rl.evaluate_rl import evaluate
from MaxText.rl import utils_rl
from MaxText.input_pipeline.instruction_data_processing import load_template_from_file
from maxtext.utils import max_logging, max_utils, maxtext_utils, model_creation_utils


def get_maxtext_model(config, devices=None):
  """
  Load MaxText model with Tunix adapter.
  # Note: pass the path to your scanned checkpoint for 'load_parameters_path'.
  # To create a scanned checkpoint, you can use /maxtext/src/MaxText/utils/ckpt_conversion/to_maxtext.py and if
  # using Pathways, please set `checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False`
  # python src/MaxText/utils/ckpt_conversion/to_maxtext.py \
  #  --model_name="gemma2-2b" \
  #  --base_output_directory="/path/to/your/output/directory" \
  #  --scan_layers=True \
  # --checkpoint_storage_use_ocdbt=False\
  # checkpoint_storage_use_zarr3=False
  # Please ensure that you pass the full path ending in `/0/items` for load_parameters_path to train_rl.py i.e.,
  # load_parameters_path=/path/to/your/output/directory/0/items
  """
  model, mesh = model_creation_utils.create_nnx_model(config, devices=devices)
  with mesh:
    use_no_op_mappings = "maxtext_config" in config.vllm_additional_config
    tunix_model = TunixMaxTextAdapter(base_model=model, use_no_op_mappings=use_no_op_mappings)
    tunix_model.config = None
  return tunix_model, mesh


def setup_configs_and_devices(argv: list[str]):
  """Setup device allocation and configs for training and inference."""
  config = pyconfig.initialize_pydantic(argv)
  devices = jax.devices()
  if config.num_trainer_slices == -1 and config.num_samplers_slices == -1:
    max_logging.log("Running RL on a single slice")
    num_vms = len(devices) // config.chips_per_vm
    trainer_devices = devices
    sampler_devices = devices
    if num_vms >= 2 and config.use_pathways:
      # Multiple hosts with Pathways - potentially split devices for trainer and sampler
      # based on trainer_devices_fraction and sampler_devices_fraction
      max_logging.log(f"{num_vms} VMs detected, allocating trainer and sampler devices, and using Pathways.")
      num_devices = len(devices)
      num_trainer_devices = int(num_devices * config.trainer_devices_fraction)
      num_sampler_devices = int(num_devices * config.sampler_devices_fraction)
      trainer_devices = devices[:num_trainer_devices]
      sampler_devices = devices[num_devices - num_sampler_devices :]
      if config.trainer_devices_fraction != 1.0:
        max_logging.log(f"Using first {len(trainer_devices)} devices as Trainer devices")
      if config.sampler_devices_fraction != 1.0:
        max_logging.log(f"Using last {len(sampler_devices)} devices as Sampler devices")
    trainer_config = config
    sampler_config = config
  elif config.num_trainer_slices > 0 and config.num_samplers_slices > 0:
    max_logging.log("Running RL with Multislice")
    devices_by_slice = collections.defaultdict(list)
    for d in devices:
      devices_by_slice[d.slice_index].append(d)
    slice_indices = sorted(devices_by_slice.keys())

    if len(slice_indices) < config.num_trainer_slices + config.num_samplers_slices:
      raise ValueError("Not enough slices for trainer and samplers")

    trainer_devices = []
    for i in range(config.num_trainer_slices):
      trainer_devices.extend(devices_by_slice[slice_indices[i]])

    sampler_devices = []
    for i in range(config.num_trainer_slices, config.num_trainer_slices + config.num_samplers_slices):
      sampler_devices.extend(devices_by_slice[slice_indices[i]])

    trainer_devices_per_slice = len(trainer_devices) // config.num_trainer_slices
    trainer_fsdp = trainer_devices_per_slice
    tp = config.ici_tensor_parallelism
    if tp > 1:
      if trainer_devices_per_slice % tp != 0:
        raise ValueError(
            f"trainer_devices_per_slice ({trainer_devices_per_slice}) must be divisible by tensor parallelism ({tp})"
        )
      if config.ici_fsdp_parallelism != -1 and config.ici_fsdp_parallelism * tp != trainer_devices_per_slice:
        raise ValueError(
            f"ici_fsdp_parallelism ({config.ici_fsdp_parallelism}) * ici_tensor_parallelism ({tp}) must equal "
            f"devices_per_slice ({trainer_devices_per_slice})"
        )
      trainer_fsdp = trainer_devices_per_slice // tp

    trainer_update = {
        "num_slices": config.num_trainer_slices,
        "ici_fsdp_parallelism": trainer_fsdp,
        "ici_tensor_parallelism": tp,
        "dcn_data_parallelism": config.num_trainer_slices,
    }

    sampler_update = {
        "num_slices": config.num_samplers_slices,
        "ici_fsdp_parallelism": len(sampler_devices) // config.num_samplers_slices,
        "ici_tensor_parallelism": -1,
        "dcn_data_parallelism": config.num_samplers_slices,
    }

    trainer_config = pyconfig.initialize_pydantic(argv, **trainer_update)
    sampler_config = pyconfig.initialize_pydantic(argv, **sampler_update)

  else:
    raise ValueError("num_trainer_slices and num_samplers_slices should be both -1 or positive")

  return trainer_config, sampler_config, trainer_devices, sampler_devices


def get_rollout_kwargs_for_data_parallelism(sampler_config, num_sampler_devices):
  """Get rollout kwargs for vLLM rollout when using data parallelism."""
  dp = sampler_config.rollout_data_parallelism
  if dp == -1:
    return {}

  rollout_kwargs = {}
  tp = sampler_config.rollout_tensor_parallelism

  if tp == -1:
    if num_sampler_devices % dp != 0:
      raise ValueError(
          f"num_sampler_devices({num_sampler_devices}) must be divisible by "
          f"rollout_data_parallelism({dp}) "
          f"when rollout_tensor_parallelism is -1."
      )
    tp = num_sampler_devices // dp
  elif tp * dp != num_sampler_devices:
    raise ValueError(
        f"rollout_tensor_parallelism({tp}) * "
        f"rollout_data_parallelism({dp}) "
        f"!= len(sampler_devices)({num_sampler_devices})"
    )
  rollout_kwargs["tensor_parallel_size"] = tp
  rollout_kwargs["data_parallel_size"] = dp
  rollout_kwargs["rollout_vllm_async_scheduling"] = True

  return rollout_kwargs


def reshard_debug(trainer_config, sampler_config, trainer_devices, sampler_devices):
  """
  Run RL training with the provided configuration.

  Args:
    trainer_config: MaxText configuration for the trainer.
    sampler_config: MaxText configuration for the sampler.
    trainer_devices: JAX devices for the trainer.
    sampler_devices: JAX devices for the sampler.
  """
  if not trainer_config.debug.rl:
    # Apply filter to suppress noisy logs
    noise_filter = max_logging.NoisyLogFilter()
    logging.getLogger().addFilter(noise_filter)
    absl_logging.get_absl_logger().addFilter(noise_filter)

  max_logging.log("Starting RL Resharding Debug Script")

  # Number of training steps.
  max_train_steps = 1
  # Create model tokenizer
  model_tokenizer = AutoTokenizer.from_pretrained(trainer_config.tokenizer_path)

  # Load reference model
  max_logging.log("Creating reference model and also meshes for reference and rollout")
  reference_model, reference_mesh = get_maxtext_model(trainer_config, trainer_devices)
  devices_array = maxtext_utils.create_device_mesh(sampler_config, sampler_devices)
  # if trainer_devices=sampler_devices, then rollout_mesh=reference_mesh
  # else rollout_mesh uses sampler_devices
  rollout_mesh = Mesh(devices_array, sampler_config.mesh_axes)
  if trainer_config.debug.rl:
    max_logging.log("Reference Model initialized successfully")
    nnx.display(reference_model)
    max_logging.log(f"Reference mesh shape: {reference_mesh.shape}")

    # Sanity check that weights are loaded correctly.
    _maxtext_state_flatten = nnx.state(reference_model).flat_state()
    maxtext_state_flatten = {".".join(str(key) for key in keys): v for keys, v in _maxtext_state_flatten}
    max_logging.log(
        f"maxtext_state_flatten[base.token_embedder.embedding].value=\
          {maxtext_state_flatten['base.token_embedder.embedding'][...]}"
    )

  # TODO: @mazumdera: change this to use lora
  # TODO: @xfgu: instead of restoring a second time from GCS, can we just copy reference_model
  # Load policy model
  max_logging.log("Creating policy model with same config as reference model on trainer mesh")
  actor_model, actor_mesh = get_maxtext_model(trainer_config, trainer_devices)

  if trainer_config.debug.rl:
    max_logging.log("Policy Model initialized successfully")
    nnx.display(actor_model)
    max_logging.log(f"Policy mesh shape: {actor_mesh.shape}")

  # Setup optimizer
  optimizer = utils_rl.get_optimizer(trainer_config, max_train_steps)

  # Parse vllm_additional_config
  rollout_additional_config = None
  if trainer_config.vllm_additional_config:
    if isinstance(trainer_config.vllm_additional_config, dict):
      # It's already parsed into a dict
      rollout_additional_config = trainer_config.vllm_additional_config
    elif isinstance(trainer_config.vllm_additional_config, str):
      # It's a string, so we need to parse it
      try:
        rollout_additional_config = json.loads(trainer_config.vllm_additional_config)
      except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse additional_config JSON: {e}") from e

    max_logging.log(f"Parsed additional config: {rollout_additional_config}")

  # RL Cluster config
  # Note that we use vLLM as the rollout engine.
  # and we are using Tensor Parallelism for rollout
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: actor_mesh,
          rl_cluster_lib.Role.REFERENCE: reference_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      role_to_logical_axis_rule={
          rl_cluster_lib.Role.ACTOR: trainer_config.logical_axis_rules,
          rl_cluster_lib.Role.REFERENCE: trainer_config.logical_axis_rules,
      },
      rollout_engine="vllm",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=trainer_config.eval_interval,
          max_steps=max_train_steps,
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
          max_prompt_length=trainer_config.max_prefill_predict_length,
          kv_cache_size=trainer_config.max_target_length + trainer_config.kv_cache_buffer,
          temperature=trainer_config.decode_sampling_temperature,
          top_p=trainer_config.decode_sampling_nucleus_p,
          top_k=trainer_config.decode_sampling_top_k,
          rollout_vllm_model_version=trainer_config.tokenizer_path,
          rollout_vllm_hbm_utilization=trainer_config.hbm_utilization_vllm,
          rollout_vllm_tpu_backend_type="jax",
          rollout_vllm_swap_space_size_gb=trainer_config.swap_space_vllm_gb,
          rollout_vllm_hf_config_path=trainer_config.vllm_hf_config_path,
          rollout_vllm_additional_config=rollout_additional_config,
          rollout_vllm_init_with_random_weights=True,
          rollout_vllm_enable_dp_attention=trainer_config.enable_dp_attention,
          rollout_vllm_max_num_batched_tokens=trainer_config.max_num_batched_tokens,
          rollout_vllm_max_num_seqs=trainer_config.max_num_seqs,
          **get_rollout_kwargs_for_data_parallelism(sampler_config, len(sampler_devices)),
      ),
  )

  # Create RL cluster
  max_logging.log("Creating RL cluster...")

  pkg_dir = os.environ.get("MAXTEXT_PKG_DIR", MAXTEXT_PKG_DIR)
  vllm_config_path = epath.Path(pkg_dir) / "configs" / "vllm.yml"
  argv_list = ["", str(vllm_config_path), "log_config=False"]
  vllm_config = pyconfig.initialize(argv_list)

  with nn_partitioning.axis_rules(vllm_config.logical_axis_rules):
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=model_tokenizer,
        cluster_config=cluster_config,
    )

  max_logging.log(
      "Calling rl_cluster.sync_weights() to reshard actor weights to rollout mesh..."
  )
  with reference_mesh, nn_partitioning.axis_rules(trainer_config.logical_axis_rules):
    rl_cluster.sync_weights()
    jax.block_until_ready(rl_cluster.rollout._sampler.transformer_state)
  max_logging.log("Resharding via sync_weights() completed.")


def main(argv: Sequence[str]) -> None:
  """Main function to run RL training.

  Args:
    argv: Command-line arguments.
  """
  pathwaysutils.initialize()
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_utils.print_system_information()
  trainer_config, sampler_config, trainer_devices, sampler_devices = setup_configs_and_devices(argv)
  reshard_debug(trainer_config, sampler_config, trainer_devices, sampler_devices)


if __name__ == "__main__":
  app.run(main)
