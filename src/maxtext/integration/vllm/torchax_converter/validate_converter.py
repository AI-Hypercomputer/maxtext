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
			src/maxtext/configs/post_train/rl.yml model_name=qwen3-30b-a3b \
			tokenizer_type=huggingface tokenizer_path=Qwen/Qwen3-30B-A3B \
			load_parameters_path=<your_maxtext_checkpoint_path> run_name=qwen3_converter_validation \
			per_device_batch_size=1 max_prefill_predict_length=8 max_target_length=16 steps=1 \
			scan_layers=true skip_jax_distributed_system=true weight_dtype=bfloat16 \
			rollout_tensor_parallelism=4 hbm_utilization_vllm=0.6 async_scheduling=false \
			prompt="Paris is" hf_access_token=<token> use_chat_template=true
  For multislice (e.g. 2x128-device slices), additionally pass:
        num_trainer_slices=1 num_samplers_slices=1

Currently this validator supports: qwen3-30b-a3b, qwen3-30b-a3b-base, qwen3-235b-a22b, gemma4-26b.
"""

import gc
import logging
import os
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
from flax import nnx
import transformers
from tunix.rl.reshard import reshard_pytree
from vllm import LLM
from vllm import SamplingParams
import pathwaysutils

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.utils import model_creation_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

vllm_model_name_mapping = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "gemma4-26b": "google/gemma-4-26B-A4B",
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

def save_dict_to_file(state_dict, filename):
  with open(filename, "w", encoding="utf-8") as f:
    for key in sorted(state_dict.keys()):
      f.write(f"{key}: {state_dict[key].shape}\n")


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

  if trainer_config.model_name not in vllm_model_name_mapping:
    raise ValueError(
        f"validate_converter.py does not support model '{trainer_config.model_name}'. "
        f"Supported models: {sorted(vllm_model_name_mapping.keys())}"
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

  if trainer_config.model_name.startswith("gemma4"):
    converter = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
  else:
    converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
  with timer("Overall Conversion"):
    vllm_state = converter.convert(model_state)
  # Explicitly delete MaxText device buffers before resharding. Python del + gc
  # is not enough — Pathways holds buffers in its object store independently of
  # Python GC, so we must call .delete() on each array to free HBM.
  for arr in jax.tree_util.tree_leaves(model_state):
    if hasattr(arr, "delete"):
      arr.delete()
  del model_state, model, mesh, converter
  gc.collect()

  print("=" * 80)
  print("Loading vLLM model for generation test...")
  print("=" * 80)
  # vLLM parallelism is driven by rollout_* params from sampler_config.
  # load_format="dummy" skips loading real weights — converted MaxText weights
  # are assigned afterwards, so real HF weights are never needed here.
  vllm_kwargs = dict(
      model=vllm_model_name_mapping[trainer_config.model_name],
      max_model_len=trainer_config.max_target_length,
      load_format="dummy",
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
  # save_dict_to_file(llm_state, "vllm_model_state.txt")
  # save_dict_to_file(vllm_state, "converted_vllm_state.txt")

  _embed_key = "vllm_model.model.embed_tokens.weight"
  _embed_before = float(jnp.mean(jnp.abs(jnp.array(llm_state[_embed_key])))) if _embed_key in llm_state else None
  logging.info("embed_tokens mean-abs BEFORE assignment: %s", _embed_before)

  with timer(f"Assigning {len(vllm_state)} weights to vLLM model"):
    for key, weight in vllm_state.items():
      weight_array = weight.value if hasattr(weight, "value") else weight
      dst_sharding = llm_state[key].sharding
      assert (
          llm_state[key].shape == weight_array.shape
      ), f"Shape mismatch for {key}: expected {llm_state[key].shape}, got {weight_array.shape}"
      llm_state[key] = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)

  _embed_after = float(jnp.mean(jnp.abs(jnp.array(llm_state[_embed_key])))) if _embed_key in llm_state else None
  logging.info("embed_tokens mean-abs AFTER assignment: %s", _embed_after)
  if _embed_before is not None and _embed_after is not None:
    logging.info("Weight assignment changed embed_tokens: %s", abs(_embed_before - _embed_after) > 1e-6)

  sampling_params = SamplingParams(temperature=0.0, max_tokens=trainer_config.max_target_length - trainer_config.max_prefill_predict_length)
  prompt = getattr(trainer_config, "prompt", "Paris is")
  if getattr(trainer_config, "use_chat_template", False):
    tokenizer_path = getattr(trainer_config, "tokenizer_path", None) or vllm_model_name_mapping[trainer_config.model_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=getattr(trainer_config, "hf_access_token", None),
    )
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
  elif trainer_config.model_name.startswith("gemma4") and not prompt.startswith("<bos>"):
    prompt = "<bos>" + prompt

  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False))


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  print(f"JAX devices: {jax.devices()}")
  _setup_jax_compilation_cache()
  _setup_vllm_environment()
  _clean_device_memory()

  validate_converter(argv)


if __name__ == "__main__":
  app.run(main)
