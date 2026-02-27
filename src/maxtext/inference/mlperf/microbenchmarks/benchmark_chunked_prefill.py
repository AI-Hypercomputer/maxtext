# Copyright 2023–2025 Google LLC
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


import datetime
import functools
import os
from typing import Any, Callable, Sequence

import jax

from jetstream.core import prefix_cache
from jetstream.engine import chunked_prefill

from absl import app

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.utils import max_utils

_WARMUP_ITERS = 2
_BENCHMARK_ITERS = 5


@jax.jit
def _copy(cache):
  def _array_copy(x):
    return x.copy()

  return jax.tree.map(_array_copy, cache)


def _run_benchmark_loop(func_to_benchmark: Callable[[], Any], iters: int, label: str) -> datetime.timedelta:
  """Runs warmup and benchmark iterations for a given function.

  Args:
    func_to_benchmark: The function to benchmark.
    iters: The number of benchmark iterations to run.
    label: A string label for printing output.

  Returns:
    The average time taken per iteration.
  """
  print(f"\nStarting warmup for {label}...")
  for i in range(_WARMUP_ITERS):
    start = datetime.datetime.now()
    result = func_to_benchmark()
    jax.block_until_ready(result)
    end = datetime.datetime.now()
    print(f"  Warmup iteration {i+1} time: {end - start}")

  print(f"\nStarting benchmark for {label}...")
  total_time = datetime.timedelta()
  for i in range(iters):
    start = datetime.datetime.now()
    result = func_to_benchmark()
    jax.block_until_ready(result)
    end = datetime.datetime.now()
    iter_time = end - start
    total_time += iter_time
    print(f"  Benchmark iteration {i+1} time: {iter_time}")

  average_time = total_time / iters
  print(f"\nAverage time taken for {label} over {iters} iterations: {average_time}")
  return average_time


def benchmark_chunked_prefill(
    engine: maxengine.MaxEngine,
    params: Any,
    chunked_tokens_list: list[chunked_prefill.ChunkedTokens],
):
  """Benchmarks chunked prefill without prefix caching.

  Args:
    engine: The MaxEngine instance.
    params: The model parameters.
    chunked_tokens_list: A list of ChunkedTokens objects representing the
      input sequence split into chunks.

  Returns:
    The average time taken for chunked prefill.
  """

  def run_chunked_prefill_utility():
    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine,
        prefill_params=params,
        chunked_tokens_list=chunked_tokens_list,
    )
    return prefill_result

  # Benchmark standard chunked prefill (no caching)
  average_time = _run_benchmark_loop(run_chunked_prefill_utility, _BENCHMARK_ITERS, "standard chunked prefill")
  return average_time


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
    return _copy(prefix)

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
  jax.effects_barrier()
  print("Finished adding the target entry.")


def create_prefix_cache(
    engine: maxengine.MaxEngine,
    params: Any,
    tokens: jax.Array,
    chunked_tokens_list: list[chunked_prefill.ChunkedTokens],
    prefix_caching_hbm_byte: int,
    prefix_caching_dram_byte: int,
    max_prefill_length: int,
) -> prefix_cache.PrefixCache:
  """Creates and populates a prefix cache.

  Args:
    engine: The MaxEngine instance.
    params: The model parameters.
    tokens: The input token sequence.
    chunked_tokens_list: A list of ChunkedTokens objects representing the
      input sequence split into chunks.
    prefix_caching_hbm_byte: The size of the HBM layer in the prefix cache.
    prefix_caching_dram_byte: The size of the DRAM layer in the prefix cache.
    max_prefill_length: The maximum length of the prefill sequence.

  Returns:
    A populated PrefixCache instance.
  """
  print("\n--- Creating and Populating Prefix Cache ---")

  # Run chunked prefill to get the initial prefix
  prefill_result, _ = chunked_prefill.do_chunked_prefill(
      prefill_engine=engine,
      prefill_params=params,
      chunked_tokens_list=chunked_tokens_list,
  )

  prefix_cache_inst = prefix_cache.PrefixCache(prefix_caching_hbm_byte, prefix_caching_dram_byte)
  # Fill cache to the max dram size
  prefill_result_cache_byte_size = jax.tree.reduce(
      lambda acc, array: acc + array.nbytes,
      prefill_result["cache"],
      0,
  )
  cache_entries_num = prefix_caching_dram_byte // prefill_result_cache_byte_size
  fill_prefix_cache(
      prefix_cache_inst,
      prefill_result["cache"],
      cache_entries_num,
      tokens,
      max_prefill_length,
  )
  return prefix_cache_inst


def benchmark_prefix_cache_loop(
    engine: maxengine.MaxEngine,
    params: Any,
    tokens: jax.Array,
    chunked_tokens_list: list[chunked_prefill.ChunkedTokens],
    prefix_cache_inst: prefix_cache.PrefixCache,
    chunk_size: int,
    run_time: list[int],
):
  """Benchmarks chunked prefill with prefix caching.

  This function benchmarks the performance of chunked prefill with varying
  cache hit rates, including the impact of saving new prefixes to the cache.

  Args:
    engine: The MaxEngine instance.
    params: The model parameters.
    tokens: The input token sequence.
    chunked_tokens_list: A list of ChunkedTokens objects representing the
      input sequence split into chunks.
    prefix_cache_inst: The PrefixCache instance to use.
    chunk_size: The chunk size used for prefilling.
    run_time: Length 1 int list (e.g. [1]) for pointer to counter. Use to prevent existing key collisions.
  """

  print("\n--- Starting Prefix Cache Benchmark ---")

  def run_chunked_prefill_with_prefix_caching(cache_hit_chunk: int, need_save: bool):
    tokens_list = tokens.tolist()

    # Load from cache (simulates reading)
    existing_prefix, _ = prefix_cache.load_existing_prefix_and_get_remain_tokens(prefix_cache_inst, tokens, chunk_size)
    assert existing_prefix is not None, "Should hit in benchmark"

    # Perform chunked prefill on remaining tokens
    prefill_result, _ = chunked_prefill.do_chunked_prefill(
        prefill_engine=engine,
        prefill_params=params,
        chunked_tokens_list=chunked_tokens_list[cache_hit_chunk:],  # Pass only the remaining chunks
        existing_prefix=existing_prefix if cache_hit_chunk > 0 else None,  # Pass existing prefix if hit
    )

    # Simulate save to cache
    if need_save:

      # Assume save will happen
      run_time[0] += 1
      prefix_cache_inst.save(
          tuple(tokens_list + [run_time[0]]),  # Prevent key existed.
          prefix_cache.Value(
              prefix=_copy(prefill_result["cache"]),
              true_length=len(tokens_list),
              padded_length=len(tokens_list),
              tokens=tuple(tokens_list),
          ),
      )

    return prefill_result

  for cache_hit_chunk in range(len(chunked_tokens_list)):
    for need_save in [True, False]:
      label = f"prefix caching (hit_chunks={cache_hit_chunk}, save={need_save})"
      benchmark_func = functools.partial(
          run_chunked_prefill_with_prefix_caching,
          cache_hit_chunk=cache_hit_chunk,
          need_save=need_save,
      )
      _run_benchmark_loop(benchmark_func, _BENCHMARK_ITERS, label)


def benchmark_prefix_cache(
    engine: maxengine.MaxEngine,
    params: Any,
    tokens: jax.Array,
    chunked_tokens_list: list[chunked_prefill.ChunkedTokens],
    prefix_caching_hbm_byte: int,
    prefix_caching_dram_byte: int,
    chunk_size: int,
    max_prefill_length: int,
):
  """Benchmarks chunked prefill with prefix caching.

  This function first creates and populates a prefix cache, then benchmarks
  the performance of chunked prefill with varying cache hit rates, including
  the impact of saving new prefixes to the cache.

  Args:
    engine: The MaxEngine instance.
    params: The model parameters.
    tokens: The input token sequence.
    chunked_tokens_list: A list of ChunkedTokens objects representing the
      input sequence split into chunks.
    prefix_caching_hbm_byte: The size of the HBM layer in the prefix cache.
    prefix_caching_dram_byte: The size of the DRAM layer in the prefix cache.
    chunk_size: The chunk size used for prefilling.
    max_prefill_length: The maximum length of the prefill sequence.
  """
  print("\n--- Starting Prefix Cache Benchmark ---")

  prefix_cache_inst = create_prefix_cache(
      engine,
      params,
      tokens,
      chunked_tokens_list,
      prefix_caching_hbm_byte,
      prefix_caching_dram_byte,
      max_prefill_length,
  )

  run_time = [0]
  benchmark_prefix_cache_loop(engine, params, tokens, chunked_tokens_list, prefix_cache_inst, chunk_size, run_time)


def prepare_setting(argv: Sequence[str]):
  """
    Constructs the necessary components for benchmarking chunked prefill with prefix caching.
  q
    Args:
      argv: The command-line arguments.

    Returns:
      engine (maxengine.MaxEngine): The MaxEngine instance.
      params (Any): The model parameters.
      tokens (jax.Array): The input token sequence.
      chunked_tokens_list (list[chunked_prefill.ChunkedTokens]): A list of ChunkedTokens objects representing the
        input sequence split into chunks.
      prefix_caching_hbm_byte (int): The size of the HBM layer in the prefix cache.
      prefix_caching_dram_byte (int): The size of the DRAM layer in the prefix cache.
      chunk_size (int): The chunk size used for prefilling.
      max_prefill_length (int): The maximum length of the prefill sequence.
  """
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  prefix_caching_hbm_byte = config.prefix_caching_hbm_byte
  prefix_caching_dram_byte = config.prefix_caching_dram_byte
  max_prefill_length = config.max_prefill_predict_length
  text = config.prompt
  chunk_size = config.prefill_chunk_size

  engine = maxengine.MaxEngine(config)

  if not engine.use_chunked_prefill:
    raise ValueError("Engine must be configured with use_chunked_prefill=True")

  params = engine.load_params()

  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  # set it to max complete prompt length that has to be benchmarked with chunked prefill
  tokens, _ = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[max_prefill_length])

  chunked_tokens_list = chunked_prefill.gen_chunked_padded_tokens(
      tokens=tokens,
      chunk_size=chunk_size,
      tokenizer=tokenizer_model,
      jax_padding=True,
  )

  return (
      engine,
      params,
      tokens,
      chunked_tokens_list,
      prefix_caching_hbm_byte,
      prefix_caching_dram_byte,
      chunk_size,
      max_prefill_length,
  )


def main(argv: Sequence[str]) -> None:
  (
      engine,
      params,
      tokens,
      chunked_tokens_list,
      prefix_caching_hbm_byte,
      prefix_caching_dram_byte,
      chunk_size,
      max_prefill_length,
  ) = prepare_setting(argv)

  benchmark_chunked_prefill(
      engine,
      params,
      chunked_tokens_list,
  )

  benchmark_prefix_cache(
      engine,
      params,
      tokens,
      chunked_tokens_list,
      prefix_caching_hbm_byte,
      prefix_caching_dram_byte,
      chunk_size,
      max_prefill_length,
  )


if __name__ == "__main__":
  app.run(main)
