# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmark script for evaluating chunked prefill performance.

This script benchmarks the `chunked_prefill.do_chunked_prefill` function
from the JetStream engine, integrated with MaxText's `MaxEngine`.
It measures the average time taken for chunked prefill over several iterations,
both with and without prefix caching enabled.

Key functionalities:
1.  Initializes MaxEngine with chunked prefill enabled.
2.  Loads model parameters and tokenizer.
3.  Tokenizes a prompt and splits it into chunks based on `config.prefill_chunk_size`.
4.  Benchmarks the standard chunked prefill operation (without caching).
5.  Initializes and populates a `PrefixCache` instance.
6.  Benchmarks chunked prefill with varying levels of prefix cache hits.
7.  Benchmarks chunked prefill including the time taken to save the final result
    to the prefix cache.

Configuration options like `use_chunked_prefill`, `prefill_chunk_size`,
`prefix_caching_hbm_byte`, `prefix_caching_dram_byte`, and
`inference_microbenchmark_prefix_cache_entries_num` control the benchmark behavior.
"""


import os
from typing import Any, Sequence
import datetime

import jax

from jetstream.engine import chunked_prefill
from jetstream.engine import engine_api
from jetstream.engine import prefix_cache

from absl import app

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig

_WARMUP_ITERS = 2
_BENCHMARK_ITERS = 5


def fill_prefix_cache(
    prefix_cache_inst: prefix_cache.PrefixCache,
    prefix: Any,
    cache_num: int,
    hit_tokens: jax.Array,
    padded_length: int,
) -> None:
  """Fill prefix cache to a specified number of entries.

  The cache will be filled with `cache_num` dummy entries, each using a
  copy of the provided `prefix` but associated with a unique dummy token key.
  Finally, an entry using `hit_tokens` as the key and the `prefix` as the
  value is added.

  Args:
    prefix_cache_inst: The PrefixCache instance to fill.
    prefix: The KVCache (value) to store in the cache entries.
    cache_num: The number of dummy entries to add before the target entry.
    hit_tokens: The token sequence (key) for the target cache entry.
    padded_length: The sequence length (including padding) that corresponds
      to the provided `prefix` KVCache.
  """
  true_length = len(hit_tokens)
  key_to_hit = tuple(hit_tokens.tolist())

  def copy_prefix():
    return jax.tree.map(lambda x: x.copy(), prefix)

  # --- Fill the cache with dummy entries ---
  print("Filling cache with", cache_num, "dummy entries...")
  for i in range(cache_num):
    # Create a unique dummy key, ensuring it's different from key_to_hit
    # and has the same length for consistency (though not strictly required by Trie).
    # Adding a large offset makes collisions highly unlikely.
    dummy_key = tuple(int(token) + 1000 + i * true_length for token in key_to_hit)

    # Create the final Value object for the dummy entry, associating
    # the copied prefix with the *dummy* key.
    dummy_value_with_key = prefix_cache.Value(
        prefix=copy_prefix(),
        true_length=true_length,
        padded_length=padded_length,
        tokens=dummy_key,  # Use the dummy key here
    )

    # Save the dummy entry to the cache
    prefix_cache_inst.save(dummy_key, dummy_value_with_key)
    # Block to make sure the cache is synced
    load_result = prefix_cache_inst.load(dummy_key)
    assert load_result is not None
    jax.block_until_ready(load_result.prefix)
    del load_result

  print("Finished filling cache with", cache_num, "dummy entries.")

  # --- Add the actual target entry ---
  print("Adding the target entry with key length ", len(key_to_hit), "...", sep="")

  value_to_hit = prefix_cache.Value(
      prefix=copy_prefix(),
      true_length=true_length,
      padded_length=padded_length,
      tokens=key_to_hit,
  )
  prefix_cache_inst.save(key_to_hit, value_to_hit)
  load_result = prefix_cache_inst.load(key_to_hit)
  assert load_result is not None
  jax.block_until_ready(load_result.prefix)
  del load_result
  print("Finished adding the target entry.")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  prefix_caching_hbm_byte = config.prefix_caching_hbm_byte
  prefix_caching_dram_byte = config.prefix_caching_dram_byte
  inference_microbenchmark_prefix_cache_entries_num = config.inference_microbenchmark_prefix_cache_entries_num

  engine = maxengine.MaxEngine(config)

  if not engine.use_chunked_prefill:
    raise ValueError("Engine must be configured with use_chunked_prefill=True")

  params = engine.load_params()

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  # set it to max complete prompt length that has to be benchmarked with chunked prefill
  max_prefill_length = config.max_prefill_predict_length
  tokens, _ = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[max_prefill_length])

  chunk_size = config.prefill_chunk_size

  chunked_tokens_list = chunked_prefill.gen_chunked_padded_tokens(
      tokens=tokens,
      chunk_size=chunk_size,
      tokenizer=tokenizer_model,
      jax_padding=True,
  )

  def run_chunked_prefill_utility():
    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine,
        prefill_params=params,
        chunked_tokens_list=chunked_tokens_list,
    )
    return prefill_result

  print("Starting warmup...")
  for i in range(_WARMUP_ITERS):
    start = datetime.datetime.now()
    prefill_result = run_chunked_prefill_utility()
    jax.block_until_ready(prefill_result)
    end = datetime.datetime.now()
    print("  Warmup iteration", i + 1, "time:", end - start)

  print("\nStarting benchmark...")
  total_time = datetime.timedelta()
  for i in range(_BENCHMARK_ITERS):
    start = datetime.datetime.now()
    prefill_result = run_chunked_prefill_utility()
    jax.block_until_ready(prefill_result)
    end = datetime.datetime.now()
    iter_time = end - start
    total_time += iter_time
    print("  Benchmark iteration", i + 1, "time:", iter_time)

  average_time = total_time / _BENCHMARK_ITERS
  print("\nAverage time taken for chunked prefill over", _BENCHMARK_ITERS, "iterations:", average_time)

  # Run prefix caching benchmark
  prefill_result = run_chunked_prefill_utility()
  prefix_cache_inst = prefix_cache.PrefixCache(prefix_caching_hbm_byte, prefix_caching_dram_byte)
  fill_prefix_cache(
      prefix_cache_inst,
      prefill_result["cache"],
      inference_microbenchmark_prefix_cache_entries_num,
      tokens,
      max_prefill_length,
  )

  def run_chunked_prefill_with_prefix_caching(cache_hit_chunk: int, need_save: bool):
    # Load to simulated the time consuming for reading the cache
    # TODO: Separate test case load from DRAM
    tokens_list = tokens.tolist()
    existing_prefix = prefix_cache.load_existing_prefix(prefix_cache_inst, tuple(tokens_list), chunk_size)
    assert existing_prefix is not None
    # Simulate prefix cache hit with chunked sized
    if cache_hit_chunk > 0:
      existing_prefix = engine_api.ExistingPrefix(
          cache=existing_prefix[0].cache, common_prefix_tokens=tokens[: cache_hit_chunk * chunk_size]
      )
    else:
      existing_prefix = None

    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine,
        prefill_params=params,
        chunked_tokens_list=chunked_tokens_list[cache_hit_chunk:],
        existing_prefix=existing_prefix,
    )
    # Simulate save to cache
    if need_save:
      # Assume directly call save will happen
      prefix_cache_inst.save(
          tuple(tokens_list),
          prefix_cache.Value(
              prefix=jax.tree.map(lambda x: x.copy(), prefill_result["cache"]),
              true_length=len(tokens_list),
              padded_length=len(tokens_list),
              tokens=tuple(tokens_list),
          ),
      )

    return prefill_result

  for cache_hit_chunk in range(len(chunked_tokens_list)):
    for need_save in [True, False]:
      print("\nBenchmark prefix caching cache_hit_chunk=", cache_hit_chunk, " need_save=", need_save, sep="")
      for i in range(_WARMUP_ITERS):
        start = datetime.datetime.now()
        prefill_result = run_chunked_prefill_with_prefix_caching(cache_hit_chunk, need_save)
        jax.block_until_ready(prefill_result)
        end = datetime.datetime.now()
        print("  Warmup iteration", i + 1, "time:", end - start)

      total_time = datetime.timedelta()
      for i in range(_BENCHMARK_ITERS):
        start = datetime.datetime.now()
        prefill_result = run_chunked_prefill_with_prefix_caching(cache_hit_chunk, need_save)
        jax.block_until_ready(prefill_result)
        end = datetime.datetime.now()
        iter_time = end - start
        total_time += iter_time
        print("  Benchmark iteration", i + 1, "time:", iter_time)

      average_time = total_time / _BENCHMARK_ITERS
      print("\nAverage time taken for prefix caching chunked prefill:", average_time)


if __name__ == "__main__":
  app.run(main)
