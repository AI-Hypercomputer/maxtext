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
An example script to perform decoding using vLLM with a MaxText model.

Example command:
  python3 -m MaxText.vllm_decode MaxText/configs/sft.yml \
    model_name=llama3.1-8b tokenizer_path=meta-llama/Llama-3.1-8B-Instruct \
    tokenizer_type=huggingface hf_access_token=<your_hf_token> \
    load_parameters_path=<your_checkpoint_path> \
    per_device_batch_size=1 run_name=vllm_decode_test \
    use_chat_template=True prompt="Suggest some famous landmarks in London." \
    decode_sampling_temperature=0.0 decode_sampling_nucleus_p=1.0 decode_sampling_top_k=0.0
"""

import os
from typing import Any, Sequence

from absl import app
import jax
import transformers

from MaxText import model_creation_utils
from MaxText import pyconfig
from MaxText.common_types import Config
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout

os.environ["SKIP_JAX_PRECOMPILE"] = "1"


def decode(
    config: Config,
    model: Any,
    mesh: jax.sharding.Mesh,
) -> None:
  """Decode using vLLM with a MaxText model."""
  # Wrap the model for Tunix
  tunix_model = TunixMaxTextAdapter(base_model=model)

  # Load the tokenizer and format the prompt
  model_tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
  model_tokenizer.bos_token = None

  # Format the prompt using chat template if specified
  prompts = [config.prompt]
  if config.use_chat_template:
    messages = [
        {"role": "user", "content": config.prompt},
    ]
    formatted_prompt_string = model_tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Set to False to get the string
        add_generation_prompt=True,
        add_special_tokens=False,  # Prevent adding special tokens
    )
    print("Formatted prompt string:", formatted_prompt_string)
    prompts = [formatted_prompt_string]

  # Create vLLM rollout for inference
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=config.max_target_length - config.max_prefill_predict_length,
      max_prompt_length=config.max_prefill_predict_length,
      temperature=config.decode_sampling_temperature,
      top_p=config.decode_sampling_nucleus_p,
      top_k=config.decode_sampling_top_k,
  )
  vllm_rollout = VllmRollout(
      model=tunix_model,
      tokenizer=model_tokenizer,
      cache_config_or_size=config.max_target_length,  # Max sequence length
      mesh=mesh,
      model_version=config.tokenizer_path,
      hbm_utilization=0.8,
      init_with_random_weights=True,  # Use random weights
      tpu_backend_type="jax",
  )

  # Generate text
  output = vllm_rollout.generate(prompts, rollout_config)
  print("Generated text:")
  print(output.text[0])


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )

  config = pyconfig.initialize(argv)
  maxtext_model, mesh = model_creation_utils.create_nnx_model(config)
  decode(config, model=maxtext_model, mesh=mesh)


if __name__ == "__main__":
  app.run(main)
