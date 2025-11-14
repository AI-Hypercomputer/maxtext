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
  python3 -m MaxText.vllm_decode MaxText/configs/base.yml \
    model_name=llama3.1-8b tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
    tokenizer_type=huggingface hf_access_token=<your_hf_token> \
    load_parameters_path=<your_checkpoint_path> \
    per_device_batch_size=1 run_name=vllm_decode_test max_target_length=64 \
    use_chat_template=False prompt="Suggest some famous landmarks in London." \
    decode_sampling_temperature=0.0 decode_sampling_nucleus_p=1.0 decode_sampling_top_k=0.0 \
    --use_tunix \
  
Or without Tunix using the MaxText vLLM integration:
  python3 -m MaxText.vllm_decode \
    --model-name qwen3-30b-a3b \
    --hf-model-name Qwen/Qwen3-30B-A3B \
    --hf-config-path src/MaxText/integration/vllm/maxtext_vllm_adapter \
    --load-parameters-path <your_checkpoint_path> \
    --ici_data_parallelism 1 \
    --ici-tensor-parallelism 4 \
    --ici-expert-parallelism 1 \
    --max-model-len 4096 \
    --max-num-batched-tokens 262144 \
    --gpu-memory-utilization 0.5 \
    --prompt "Suggest some famous landmarks in London." \
"""

import os
from typing import Any, Sequence

from absl import app
from absl import flags
import jax
import transformers

from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.common_types import Config
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout
from vllm import LLM
from vllm.sampling_params import SamplingParams

os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["NEW_MODEL_DESIGN"] = "1"


# --- DEFINE FLAGS GLOBALLY ---
FLAGS = flags.FLAGS

# Parallelism
flags.DEFINE_integer("ici_data_parallelism", 1, "Size of the data parallelism dimension.")
flags.DEFINE_integer("ici_tensor_parallelism", 1, "Size of the non-expert tensor parallelism dimension.")
flags.DEFINE_integer("ici_expert_parallelism", 1, "Size of the MoE expert parallelism dimension.")

# Model
flags.DEFINE_string("model_name", "qwen3-30b-a3b", "Model name for MaxText.")
flags.DEFINE_string("hf_model_name", "Qwen/Qwen3-30B-A3B", "Path to the Hugging Face model.")
flags.DEFINE_string("hf_config_path", None, "Path to the local Hugging Face model config.")
flags.DEFINE_string("load_parameters_path", None, "Path to load model parameters from.")
flags.DEFINE_bool("enable_expert_parallel", False, "Whether to enable expert parallelism.")

# Length/Throughput
flags.DEFINE_integer("max_target_length", 1024, "Maximum total context length (MCL).")
flags.DEFINE_integer("max_prefill_length", 512, "Maximum prefill length.")
flags.DEFINE_float("gpu_memory_utilization", 0.72, "Fraction of GPU memory to be used for the model executor.")

# Decoding
flags.DEFINE_bool("use_tunix", False, "Whether to use Tunix for vLLM decoding.")
flags.DEFINE_string("prompt", "Suggest some famous landmarks in London.", "The prompt to decode.")
flags.DEFINE_integer("decode_sampling_temperature", 0, "Temperature for sampling.")
flags.DEFINE_integer("decode_sampling_nucleus_p", 1, "Nucleus sampling probability.")
flags.DEFINE_integer("decode_sampling_top_k", 1, "Top-k sampling probability.")

# Mark required flags
flags.mark_flag_as_required("hf_config_path")


def decode_with_vllm(
    model_name: str,
    hf_model_name: str,
    hf_config_path: str,
    load_parameters_path: str,
    ici_data_parallelism: int,
    ici_tensor_parallelism: int,
    ici_expert_parallelism: int,
    max_prefill_length: int,
    max_target_length: int,
    gpu_memory_utilization: float,
    enable_expert_parallel: bool,
    prompt: str,
    decode_sampling_temperature: float,
    decode_sampling_nucleus_p: float,
    decode_sampling_top_k: float,
) -> None:
  """Decode using vLLM with a MaxText model implementation.

  Args:
    model_name: Name of the model for MaxText.
    hf_model_name: Path to the Hugging Face model.
    hf_config_path: Path to the local Hugging Face model config.
    load_parameters_path: Path to load model parameters from.
    ici_data_parallelism: Size of the data parallelism dimension.
    ici_tensor_parallelism: Size of the non-expert tensor parallelism dimension.
    ici_expert_parallelism: Size of the MoE expert parallelism dimension.
    max_prefill_length: Maximum prefill length.
    max_target_length: Maximum total context length (MCL).
    gpu_memory_utilization: Fraction of GPU memory to be used for the model executor.
    enable_expert_parallel: Whether to enable expert parallelism.
    prompt: The prompt to decode.
    decode_sampling_temperature: Temperature for sampling.
    decode_sampling_nucleus_p: Nucleus sampling probability.
    decode_sampling_top_k: Top-k sampling probability.
  """

  # Prepare vLLM Arguments
  vllm_args = {}
  vllm_args["additional_config"] = {}

  # Core vLLM Arguments
  vllm_args["model"] = hf_model_name
  vllm_args["max_model_len"] = max_target_length
  vllm_args["tensor_parallel_size"] = ici_tensor_parallelism
  vllm_args["data_parallel_size"] = ici_data_parallelism
  vllm_args["enable_expert_parallel"] = enable_expert_parallel
  vllm_args["hf_config_path"] = hf_config_path
  vllm_args["gpu_memory_utilization"] = gpu_memory_utilization

  if load_parameters_path is None:
    vllm_args["load_format"] = "dummy"

  # Prepare MaxText and sharding configs (Parallelism is dynamic)
  vllm_args["additional_config"]["maxtext_config"] = {
      "model_name": model_name,
      "max_target_length": max_target_length,
      "weight_dtype": "bfloat16",
      "allow_split_physical_axes": True,
      "load_parameters_path": load_parameters_path,
  }

  vllm_args["additional_config"]["sharding"] = {
      "sharding_strategy": {
          "tensor_parallelism": ici_tensor_parallelism,
          "expert_parallelism": ici_expert_parallelism,
          "data_parallelism": ici_data_parallelism,
      },
  }

  if enable_expert_parallel:
    vllm_args["additional_config"]["sharding"]["sharding_strategy"].update({"expert_parallelism": ici_expert_parallelism})

  # Initialize and Run LLM
  max_tokens = max_target_length - max_prefill_length
  sampling_params = SamplingParams(
      temperature=decode_sampling_temperature,
      max_tokens=max_tokens,
      top_k=decode_sampling_top_k,
      top_p=decode_sampling_nucleus_p,
  )

  print(
      f"Initializing LLM with DP={vllm_args['data_parallel_size']}, TP={vllm_args['tensor_parallel_size']} "
      f"and EP={ici_expert_parallelism if enable_expert_parallel else 0}..."
  )
  llm = LLM(**vllm_args)

  print("Generating output...")
  outputs = llm.generate([prompt], sampling_params)

  # Print Outputs
  for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def decode_with_tunix(
    config: Config,
    model: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Decode using vLLM with a MaxText model."""
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
  print(f"Prompt: {config.prompt}")
  print(f"Output: {output.text[0]}")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  if FLAGS.use_tunix:
    config = pyconfig.initialize(argv)
    maxtext_model, mesh = model_creation_utils.create_nnx_model(config)
    decode_with_tunix(config, model=maxtext_model, mesh=mesh)
  else:
    decode_with_vllm(
        model_name=FLAGS.model_name,
        hf_model_name=FLAGS.hf_model_name,
        hf_config_path=FLAGS.hf_config_path,
        load_parameters_path=FLAGS.load_parameters_path,
        ici_data_parallelism=FLAGS.ici_data_parallelism,
        ici_tensor_parallelism=FLAGS.ici_tensor_parallelism,
        ici_expert_parallelism=FLAGS.ici_expert_parallelism,
        max_target_length=FLAGS.max_target_length,
        max_prefill_length=FLAGS.max_prefill_length,
        gpu_memory_utilization=FLAGS.gpu_memory_utilization,
        enable_expert_parallel=FLAGS.enable_expert_parallel,
        prompt=FLAGS.prompt,
        decode_sampling_temperature=FLAGS.decode_sampling_temperature,
        decode_sampling_nucleus_p=FLAGS.decode_sampling_nucleus_p,
        decode_sampling_top_k=FLAGS.decode_sampling_top_k,
    )


if __name__ == "__main__":
  app.run(main)
