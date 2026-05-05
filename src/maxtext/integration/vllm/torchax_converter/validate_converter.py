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

"""Validate MaxText to vLLM weight conversion for supported models.

This module provides a config-driven validation entrypoint that:
1. loads a MaxText model from a standard MaxText config,
2. converts its weights into the vLLM layout,
3. loads the matching vLLM model, and
4. assigns the converted weights before running a short generation check.

	python -m maxtext.integration.vllm.torchax_converter.validate_converter \
			src/maxtext/configs/base.yml model_name=qwen3-30b-a3b \
			tokenizer_type=huggingface tokenizer_path=Qwen/Qwen3-30B-A3B \
			load_parameters_path=<your_maxtext_checkpoint_path> run_name=qwen3_converter_validation \
			per_device_batch_size=1 max_prefill_predict_length=8 max_target_length=16 steps=1 \
			scan_layers=true skip_jax_distributed_system=true weight_dtype=bfloat16 \
			rollout_tensor_parallelism=4 hbm_utilization_vllm=0.6 async_scheduling=false \
			prompt="Paris is" hf_access_token=<token>

  For multislice (e.g. 2x128-device slices), additionally pass:
        num_trainer_slices=1 num_samplers_slices=1

Currently this validator supports qwen3 converter flows.
"""

import gc
import logging
import os
from typing import Sequence

from absl import app
import jax
from flax import nnx
from tunix.rl.reshard import reshard_pytree
from vllm import LLM
from vllm import SamplingParams
import pathwaysutils

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.utils import model_creation_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

vllm_model_name_mapping = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    # Add more mappings as needed
}


def _setup_jax_compilation_cache():
  jax.config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_enable_compilation_cache", True)


def _setup_vllm_environment():
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["JAX_RANDOM_WEIGHTS"] = "False"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _clean_device_memory():
  logging.info("Cleaning JAX device memory...")
  gc.collect()
  for array in jax.live_arrays():
    array.delete()
  logging.info("Device memory cleanup complete.")


def validate_converter(argv) -> None:
  """Run end-to-end validation for MaxText to vLLM weight conversion.

  Device/config split mirrors train_rl.py:
    - trainer_config uses ici_* parallelism for the MaxText mesh
    - sampler_config uses rollout_* parallelism for the vLLM mesh
  Single-slice (num_trainer_slices == -1): trainer and sampler share all devices.
  Multislice: first num_trainer_slices slices go to MaxText, the next
  num_samplers_slices slices go to vLLM.
  """
  trainer_config, sampler_config, trainer_devices, sampler_devices = (
      model_creation_utils.setup_configs_and_devices(argv)
  )

  if not trainer_config.model_name.startswith("qwen3"):
    raise ValueError(
        "validate_converter.py currently supports qwen3 models only. "
        f"Got {trainer_config.model_name}."
    )

  # In single-slice mode setup_configs_and_devices returns the same object for both.
  multislice = trainer_devices is not sampler_devices

  logging.info("Creating MaxText model...")
  model, mesh = model_creation_utils.from_pretrained(
      trainer_config,
      devices=trainer_devices,
      model_mode=MODEL_MODE_AUTOREGRESSIVE,
  )
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {trainer_config.model_name}")
  print(f"Mesh: {mesh}")

  print("=" * 80)
  print("Converting weights to vLLM format")
  print("=" * 80)
  model_state = {"base": nnx.state(model)}
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, "shape") and hasattr(leaf, "sharding"):
      path_str = jax.tree_util.keystr(path)
      logging.info("Name: %s, shape: %s", path_str, leaf.shape)
      logging.info("\tSharding: %s", leaf.sharding)

  converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
  with timer("Overall Conversion"):
    vllm_state = converter.convert(model_state)
  del model_state
  gc.collect()

  print("=" * 80)
  print("Loading vLLM model for generation test...")
  print("=" * 80)
  # vLLM parallelism is driven by rollout_* params from sampler_config.
  vllm_kwargs = dict(
      model=vllm_model_name_mapping[trainer_config.model_name],
      max_model_len=trainer_config.max_target_length,
      data_parallel_size=sampler_config.rollout_data_parallelism,
      tensor_parallel_size=sampler_config.rollout_tensor_parallelism,
      gpu_memory_utilization=getattr(sampler_config, "hbm_utilization_vllm", 0.5),
      async_scheduling=getattr(sampler_config, "async_scheduling", False),
  )
  if multislice:
    # Pin vLLM to its assigned sampler devices so it doesn't overlap with trainer.
    vllm_kwargs["additional_config"] = {
        "sharding": {
            "sharding_strategy": {
                "device_indexes": [d.id for d in sampler_devices],
            }
        }
    }
  llm = LLM(**vllm_kwargs)
  print("\n" + "=" * 80)
  llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state

  with timer(f"Assigning {len(vllm_state)} weights to vLLM model"):
    for key, weight in vllm_state.items():
      weight_array = weight.value if hasattr(weight, "value") else weight
      dst_sharding = llm_state[key].sharding
      assert (
          llm_state[key].shape == weight_array.shape
      ), f"Shape mismatch for {key}: expected {llm_state[key].shape}, got {weight_array.shape}"
      llm_state[key] = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)

  sampling_params = SamplingParams(temperature=0.0, max_tokens=trainer_config.max_target_length - trainer_config.max_prefill_predict_length)
  prompt = getattr(trainer_config, "prompt", trainer_config.prompt)
  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate(prompt, sampling_params=sampling_params))


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  print(f"JAX devices: {jax.devices()}")
  _setup_jax_compilation_cache()
  _setup_vllm_environment()
  _clean_device_memory()

  validate_converter(argv)


if __name__ == "__main__":
  app.run(main)
