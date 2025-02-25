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

from typing import Sequence
from absl import app
import numpy as np
import transformers

def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_config(config)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)

  # text = config.prompt
  #Anisha
  # texts = ["I love to", "Sky is the"]
  texts = [config.prompt]
  outputs = []
  metadata = engine.get_tokenizer()
  # tokenizer_model = engine.build_tokenizer(metadata)
  tokenizer_model = tokenizer_model = transformers.AutoTokenizer.from_pretrained(
    config.tokenizer_path,
    add_bos_token=config.add_bos,
    add_eos_token=False, #config.add_eos,
    model_max_length=config.max_target_length,
    legacy=False,
    token=config.hf_access_token,
  )
  batch_tokens = []
  batch_true_lengths = []
  for text in texts:
    # tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
    # assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
    # assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    tokens = tokenizer_model(
    text,
    truncation=True,      # Enable truncation if the text exceeds max_length
    max_length=512,        
    )
    batch_tokens.append(tokens['input_ids'])
    true_length = len(tokens['input_ids'])
    batch_true_lengths.append(true_length)

  batch_tokens = jax.numpy.array(batch_tokens)
  batch_true_lengths = jax.numpy.array(batch_true_lengths)
  # batch_tokens = jax.numpy.repeat(batch_tokens, repeats=5, axis=0)  # shape [B*G, L_prompt]
  # batch_true_lengths = jax.numpy.repeat(batch_true_lengths,repeats=5,axis=0)

  for i in range(batch_tokens.shape[0]):
    tokens = batch_tokens[i]
    true_length = batch_true_lengths[i]

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
    print(f"Input `{tokenizer_model.decode(tokens.tolist())}` -> `{output}`")
    outputs.append(output)

  if config.autoregressive_decode_assert != "":
    assert (
        output == config.autoregressive_decode_assert
    ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  app.run(main)
