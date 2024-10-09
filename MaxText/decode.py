# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utility for running inference on a single stream"""

import jax

import max_utils
import maxengine

import os
import pyconfig
import sys


def main(config):
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
  assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"

  # Split RNG before calling prefill
  rng, rng_prefill = jax.random.split(rng)
  prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill)
  slot = 0

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  sampled_tokens_list.append(first_token)
  for _ in steps:
    rng, rng_generate = jax.random.split(rng)
    decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  output = tokenizer_model.decode(results)
  print(f"Input `{text}` -> `{output}`")

  if config.autoregressive_decode_assert != "":
    assert (
        output == config.autoregressive_decode_assert
    ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg)
