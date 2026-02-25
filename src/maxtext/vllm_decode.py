#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
An example script to perform decoding using vLLM via Tunix or via MaxText on vLLM.

Example usage with Tunix:
  python3 -m maxtext.vllm_decode maxtext/configs/base.yml \
    model_name=llama3.1-8b tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
    tokenizer_type=huggingface hf_access_token=<your_hf_token> \
    load_parameters_path=<your_checkpoint_path> \
    per_device_batch_size=1 run_name=vllm_decode_test max_target_length=64 \
    use_chat_template=False prompt="Suggest some famous landmarks in London." \
    decode_sampling_temperature=0.0 decode_sampling_nucleus_p=1.0 decode_sampling_top_k=0.0 \
    --use_tunix \
  
Or without Tunix using the MaxText vLLM integration:
  python3 -m maxtext.vllm_decode maxtext/configs/base.yml \
    model_name=qwen3-30b-a3b \
    tokenizer_path=Qwen/Qwen3-30B-A3B \
    vllm_hf_config_path=src/MaxText/integration/vllm/maxtext_vllm_adapter \
    load_parameters_path=<your_checkpoint_path> \
    ici_tensor_parallelism=4 \
    hbm_utilization_vllm=0.5 \
    prompt="Suggest some famous landmarks in London."
"""

import os
from typing import Any, Sequence

from absl import app
from absl import flags
from flax.linen import partitioning as nn_partitioning
import jax
import transformers

from maxtext.utils import model_creation_utils
from maxtext.utils import max_logging
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.common.common_types import Config
from maxtext.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout
from vllm import LLM
from vllm.sampling_params import SamplingParams
from MaxText import pyconfig

os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["NEW_MODEL_DESIGN"] = "1"


# --- DEFINE FLAGS GLOBALLY ---
FLAGS = flags.FLAGS

flags.DEFINE_bool("use_tunix", False, "Whether to use Tunix for vLLM decoding.")
flags.DEFINE_integer("seed", 42, "Random seed for sampling.")


def decode_with_vllm(config: Config) -> None:
  """Decode using vLLM with a MaxText model implementation.

  Args:
    config: MaxText config.
  """
  # Prepare vLLM Arguments
  vllm_args = {
      "model": config.tokenizer_path,
      "max_model_len": config.max_target_length,
      "tensor_parallel_size": config.ici_tensor_parallelism,
      "data_parallel_size": config.ici_data_parallelism,
      "hf_config_path": config.vllm_hf_config_path,
      "hf_overrides": config.vllm_hf_overrides,
      "gpu_memory_utilization": config.hbm_utilization_vllm,
      "additional_config": {
          "maxtext_config": {
              "model_name": config.model_name,
              "weight_dtype": "bfloat16",
              "allow_split_physical_axes": True,
              "debug_sharding": config.debug_sharding,
          },
          "sharding": {
              "sharding_strategy": {
                  "enable_dp_attention": config.enable_dp_attention,
              },
          },
      },
  }

  if config.load_parameters_path:
    vllm_args["additional_config"]["maxtext_config"]["load_parameters_path"] = config.load_parameters_path
  else:
    vllm_args["load_format"] = "dummy"

  enable_expert_parallel = config.ici_expert_parallelism > 1
  if enable_expert_parallel:
    vllm_args["additional_config"]["sharding"]["sharding_strategy"]["expert_parallelism"] = config.ici_expert_parallelism
    vllm_args["enable_expert_parallel"] = enable_expert_parallel

  max_logging.log(
      f"Initializing LLM with DP={config.ici_data_parallelism}, TP={config.ici_tensor_parallelism} "
      f"and EP={config.ici_expert_parallelism if enable_expert_parallel else 1}..."
  )

  vllm_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "inference", "vllm.yml")
  argv_list = ["", str(vllm_config_path), "log_config=False"]
  vllm_config = pyconfig.initialize(argv_list)

  with nn_partitioning.axis_rules(vllm_config.logical_axis_rules):
    llm = LLM(**vllm_args)

  max_logging.log("Generating output...")
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      token=config.hf_access_token,
  )

  prompts = [config.prompt]
  if config.use_chat_template:
    # Format the prompt using chat template if specified
    messages = [
        {"role": "user", "content": config.prompt},
    ]
    input_with_chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Set to False to get the string
        add_generation_prompt=True,
        add_special_tokens=False,  # Prevent adding special tokens
    )
    prompts = [input_with_chat_template]

  max_prompt_length = max(len(tokenizer.encode(p)) for p in prompts)
  max_tokens_to_generate = config.max_target_length - max_prompt_length
  if max_tokens_to_generate <= 0:
    raise ValueError(
        f"max_target_length ({config.max_target_length}) must be greater than max_prompt_length ({max_prompt_length})"
    )

  sampling_params = SamplingParams(
      temperature=config.decode_sampling_temperature,
      max_tokens=max_tokens_to_generate,
      top_k=config.decode_sampling_top_k,
      top_p=config.decode_sampling_nucleus_p,
      seed=FLAGS.seed,
  )

  outputs = llm.generate(prompts, sampling_params)

  # max_logging.log Outputs
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    max_logging.log(f"Prompt: {prompt}, Generated text: {generated_text}")


def decode_with_tunix(
    config: Config,
    model: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Decode using vLLM with a MaxText model via Tunix adapter.

  Args:
    config: MaxText config.
    model: The MaxText model instance.
    mesh: The JAX mesh for parallelism.
  """
  # Wrap the model for Tunix
  tunix_model = TunixMaxTextAdapter(base_model=model)

  # Load the tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      token=config.hf_access_token,
      model_max_length=config.max_target_length,
  )
  tokenizer.bos_token = None

  prompts = [config.prompt]
  if config.use_chat_template:
    # Format the prompt using chat template if specified
    messages = [
        {"role": "user", "content": config.prompt},
    ]
    input_with_chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Set to False to get the string
        add_generation_prompt=True,
        add_special_tokens=False,  # Prevent adding special tokens
    )
    prompts = [input_with_chat_template]

  max_prompt_length = max(len(tokenizer.encode(p)) for p in prompts)
  max_tokens_to_generate = config.max_target_length - max_prompt_length
  if max_tokens_to_generate <= 0:
    raise ValueError(
        f"max_target_length ({config.max_target_length}) must be greater than max_prompt_length ({max_prompt_length})"
    )

  # Create vLLM rollout for inference
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=max_tokens_to_generate,
      max_prompt_length=max_prompt_length,
      temperature=config.decode_sampling_temperature,
      top_p=config.decode_sampling_nucleus_p,
      top_k=config.decode_sampling_top_k,
  )
  vllm_rollout = VllmRollout(
      model=tunix_model,
      tokenizer=tokenizer,
      # The cache_config_or_size sets the absolute maximum sequence length.
      # We add 256 as a safety buffer to account for tokens added by
      # other special formatting, which is not part of max_prompt_length.
      cache_config_or_size=max_prompt_length + max_tokens_to_generate + 256,
      mesh=mesh,
      model_version=config.tokenizer_path,
      hbm_utilization=0.8,
      # Initialize vllm model with random weights to speed up bootstrap time.
      # Actual model weights will be loaded later.
      init_with_random_weights=True,
      tpu_backend_type="jax",
  )

  # Generate text
  output = vllm_rollout.generate(prompts, rollout_config)
  max_logging.log(f"Prompt: {config.prompt}")
  max_logging.log(f"Output: {output.text[0]}")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  config = pyconfig.initialize(argv)

  if FLAGS.use_tunix:
    maxtext_model, mesh = model_creation_utils.create_nnx_model(config)
    decode_with_tunix(config, model=maxtext_model, mesh=mesh)
  else:
    decode_with_vllm(config)


if __name__ == "__main__":
  app.run(main)
