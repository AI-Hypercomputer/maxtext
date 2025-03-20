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

import max_utils
import maxengine

import os
import pyconfig

from typing import Sequence
from absl import app
import jax.numpy as jnp
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
  # set it to max complete prompt length that has to be bechmarked with chunked prefill
  max_prefill_length = 7000
  tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[max_prefill_length])

  chunk_size = 2048
  # set this to array of acceptable chunk sizes
  prefill_lengths = [1024, 2048, 4096, 7000]
  # prefill_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

  padded_tokens, true_lengths, positions = token_utils.chunk_and_pad_tokens(
      tokens, tokenizer_model.bos_id, tokenizer_model.pad_id, False, prefill_lengths, max_prefill_length, chunk_size, True
  )
  rng, _ = jax.random.split(rng)

  def run_chunked_prefill():
    prefill_result = None
    next_pos = 0
    for chunk_num, _ in enumerate(padded_tokens):
      if prefill_result is None:
        prefill_result, _ = engine.prefill(
            params=params,
            padded_tokens=padded_tokens[chunk_num],
            true_length=true_lengths[chunk_num],
            positions=positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=tokens,
            previous_chunk=prefill_result,
            rng=rng,
        )
      else:
        prefill_result, _ = engine.prefill(
            params=params | {"cache": prefill_result["cache"]},
            padded_tokens=padded_tokens[chunk_num],
            true_length=true_lengths[chunk_num],
            positions=positions[chunk_num],
            complete_prompt_true_length=true_length,
            complete_padded_prompt=tokens,
            previous_chunk=prefill_result,
            rng=rng,
        )
      true_length_array = jnp.expand_dims(jnp.arange(0, chunk_num * chunk_size + true_lengths[chunk_num]), 0)
      prefill_result["true_length_array"] = true_length_array
      prefill_result["next_pos"] = jnp.full((1, 1), next_pos + true_lengths[chunk_num], dtype=jnp.int32)
      next_pos = next_pos + true_lengths[chunk_num]
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
