"""
 Copyright 2025 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Shared Benchmark config for v5e orchestrations."""

# pylint: disable=ungrouped-imports
import jax

from MaxText import max_utils
from MaxText import maxengine

import os
from MaxText import pyconfig

from typing import Sequence
from absl import app
import datetime
from jetstream.engine import token_utils

_WARMUP_ITERS = 2
_BENCHMARK_ITERS = 5


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  # set it to max complete prompt length that has to be benchmarked with chunked prefill
  max_prefill_length = config.max_prefill_predict_length
  tokens, _ = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[max_prefill_length])

  chunk_size = config.prefill_chunk_size
  # set this to array of acceptable chunk sizes
  prefill_lengths = [1024, 2048, 4096, 8192]

  # chunked first to separate time of chunked and tokenized.
  common_prefix_tokens = []
  padded_input_tokens = []
  input_true_lengths = []
  # use whole tokens with padding for long context to max_prefill_length
  for start_pos in range(0, len(tokens), chunk_size):
    input_token = tokens[start_pos : min(len(tokens), start_pos + chunk_size)]
    padded_input_token, input_true_length = token_utils.pad_tokens(
        input_token,
        tokenizer_model.bos_id,
        tokenizer_model.pad_id,
        is_bos=False,
        prefill_lengths=prefill_lengths,
        max_prefill_length=max_prefill_length,
        jax_padding=True,
    )
    common_prefix_tokens.append(tokens[0 : min(len(tokens), start_pos + chunk_size)])
    padded_input_tokens.append(padded_input_token)
    input_true_lengths.append(input_true_length)

  rng, _ = jax.random.split(rng)

  def run_chunked_prefill():
    prefill_result = None
    existing_prefix = None
    for common_prefix_token, padded_input_token, input_true_length in zip(
        common_prefix_tokens, padded_input_tokens, input_true_lengths
    ):
      prefill_result, _ = engine.prefill(
          params=params,
          existing_prefix=existing_prefix,
          padded_tokens=padded_input_token,
          true_length=input_true_length,
          rng=rng,
      )
      existing_prefix = maxengine.ExistingPrefix(
          cache=prefill_result["cache"],
          common_prefix_tokens=common_prefix_token,
      )

    return prefill_result

  for _ in range(_WARMUP_ITERS):
    start = datetime.datetime.now()
    prefill_result = run_chunked_prefill()
    jax.block_until_ready(prefill_result)
    end = datetime.datetime.now()
    print("time taken for chunk prefill warmup: ", end - start)

  for _ in range(_BENCHMARK_ITERS):
    start = datetime.datetime.now()
    prefill_result = run_chunked_prefill()
    jax.block_until_ready(prefill_result)
    end = datetime.datetime.now()
    print("time taken for chunk prefill ", end - start)


if __name__ == "__main__":
  app.run(main)
