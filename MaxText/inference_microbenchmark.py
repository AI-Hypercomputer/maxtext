"""
Copyright 2024 Google LLC

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

"""Inference microbenchmark for prefill and autoregressive steps."""
import datetime
import jax
import json

from absl import app
from collections.abc import MutableMapping

from jetstream.engine import token_utils

from MaxText import max_utils
from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import prefix_cache
from MaxText import profiler
from MaxText import pyconfig

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

_WARMUP_ITERS = 2
_FLATTEN_MICROBENCHMARK_RESULTS = False
# pylint: disable=too-many-positional-arguments


def prefix_cache_benchmark(
    prefix, prefill_length: int, true_length: int, common_prefix_proportion: float, prefix_cache_entries_num: int, iters: int
):
  """Handles running prefix cache benchmark, and printing results.

  Create different key with half of prefill_length common prefix insert into cache.
  The value is not relevant to the cache for now. Just copy the prefix for every cache entry.
  1. Fill the prefix cache to full capacity.
  2. Benchmark save prefix cache with evicting time average by prefix_cache_entries_num.
  3. Benchmark fetch_longest_common_prefix_key average by iters.
  4. Benchmark load prefix cache time average by iters.

  Args:
    prefix: prefix return from prefill function
    prefill_length: prefill token length after padding
    true_length: true prefill token length
    common_prefix_proportion: [0., 1.] common prefix proportion to the prefill_length
    prefix_cache_entries_num: number of prefix cache entries insert into PrefixCache
    iters: repeat time to test fetch_longest_common_prefix_key and load from cache
  """

  print(f"Prefix Cache benchmark results for prefill length {prefill_length}:\n")

  value: prefix_cache.Value = prefix_cache.Value(
      prefix=prefix,
      true_length=true_length,
      padded_length=prefill_length,
      tokens=tuple(i for i in range(prefill_length)),
  )

  def copy_jax_array(x):
    return x.copy()

  def clone_value():
    return prefix_cache.Value(
        prefix=jax.tree.map(copy_jax_array, value.prefix),
        true_length=value.true_length,
        padded_length=value.padded_length,
        tokens=value.tokens,
        prefix_size_bytes=value.prefix_size_bytes,
        device=value.device,
    )

  prefix_size_bytes_gb = value.prefix_size_bytes / 1024 / 1024 / 1024
  max_bytes = prefix_cache_entries_num * value.prefix_size_bytes
  # TODO(yuyanpeng): test hierarchical cache
  prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
  common_len = int(prefill_length * common_prefix_proportion)
  remain_len = prefill_length - common_len
  common_prefix_key = tuple(i for i in range(common_len))

  # Fill the prefix caching
  new_value_list = []
  for c_idx in range(prefix_cache_entries_num):
    # Add 100 to make sure filled prefix caching will not share the common_prefix_key.
    # The later save prefix part will evict all of them.
    key = tuple(100 + i + c_idx * prefill_length for i in range(prefill_length))
    new_value = clone_value()
    prefix_cache_inst.save(key, new_value)
    new_value_list.append(new_value)
  jax.block_until_ready(new_value_list)
  del new_value_list

  # Save prefix
  new_value = None
  save_sec = 0
  for c_idx in range(iters):
    key = common_prefix_key + tuple(i + c_idx * remain_len for i in range(remain_len))
    # values are not relevant for caching now, just clone the same tokens and values for test
    new_value = clone_value()
    jax.block_until_ready(new_value)
    start = datetime.datetime.now()
    prefix_cache_inst.save(key, new_value)
    end = datetime.datetime.now()
    save_sec += (end - start).total_seconds()
  del new_value
  save_avg_ms = save_sec * 1000 / iters

  # Fetch longest prefix key
  key_load = common_prefix_key + tuple(i + prefix_cache_entries_num * remain_len for i in range(remain_len))
  matched_key = None
  fetch_sec = 0
  for _ in range(iters):
    start = datetime.datetime.now()
    matched_key = prefix_cache_inst.fetch_longest_common_prefix_key(key_load)
    end = datetime.datetime.now()
    fetch_sec += (end - start).total_seconds()
  fetch_avg_ms = fetch_sec * 1000 / iters

  assert matched_key is not None

  # Load prefix
  load_sec = 0
  value_load = None
  for _ in range(iters):
    start = datetime.datetime.now()
    loaded_value = prefix_cache_inst.load(matched_key)
    jax.block_until_ready(loaded_value)
    end = datetime.datetime.now()
    load_sec += (end - start).total_seconds()
  del value_load
  load_avg_ms = load_sec * 1000 / iters

  print(
      f"PrefixCaching results:\n"
      f"\tPer prefix size bytes: {prefix_size_bytes_gb:.3f} GB\n"
      f"\tAverage save cache time: {save_avg_ms:.3f} ms\n"
      f"\tAverage fetch longest prefix time: {fetch_avg_ms:.3f} ms\n"
      f"\tAverage load cache time: {load_avg_ms:.3f} ms\n\n\n"
  )
  del prefix_cache_inst


def prefill_benchmark_loop(engine_prefill, params, tokens, true_length, iters, num_samples: int | None = None):
  """Inner loop for benchmarking prefill step."""
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(iters):
    rng, rng_prefill = jax.random.split(rng)
    if num_samples is None:
      prefill_result, _ = engine_prefill(params, tokens, true_length, rng_prefill)
    else:
      prefill_result, _ = engine_prefill[num_samples](params, tokens, true_length, rng_prefill, None)
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  del prefill_result
  return (end - start).total_seconds()


def prefill_benchmark(config, engine_prefill, params, tokens, true_length, num_model_params, iters):
  """Handles warmup, running prefill benchmark, and printing results."""
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(_WARMUP_ITERS):
    rng, rng_prefill = jax.random.split(rng)
    prefill_result, _ = engine_prefill(params, tokens, true_length, rng_prefill)
  jax.block_until_ready(prefill_result)
  del prefill_result

  print(f"Prefill benchmark results for length {tokens.size}:\n")
  time_in_s = prefill_benchmark_loop(engine_prefill, params, tokens, true_length, iters)
  prefill_average_ms = 1000 * time_in_s / iters
  prefill_tflops_per_device, _, _ = maxtext_utils.calculate_prefill_tflops_per_device(num_model_params, tokens.size, config)
  tflops_per_sec_per_device = prefill_tflops_per_device / prefill_average_ms * 1000.0
  print(
      f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
      f"\tPrefill total TFLOPs/device: {prefill_tflops_per_device:.3f}\n"
      f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n\n\n\n"
  )
  result_dict = {
      "time_in_ms": prefill_average_ms,
      "total_tflops_per_device": prefill_tflops_per_device,
      "tflops_per_sec_per_device": tflops_per_sec_per_device,
  }
  return result_dict


def prefill_multisampling_benchmark(config, engine_prefill_multisampling, params, tokens, true_length, iters):
  """Handles warmup, running prefill benchmark, and printing results."""
  rng = jax.random.PRNGKey(1234)
  prefill_result = None
  for _ in range(_WARMUP_ITERS):
    rng, rng_prefill = jax.random.split(rng)
    for num_samples in config.inference_microbenchmark_num_samples:
      prefill_result, _ = engine_prefill_multisampling[num_samples](params, tokens, true_length, rng_prefill, None)
  jax.block_until_ready(prefill_result)
  del prefill_result

  print(f"Multi-sampling prefill benchmark results for length {tokens.size}:\n")
  result_dict = {}
  for num_samples in config.inference_microbenchmark_num_samples:
    time_in_s = prefill_benchmark_loop(engine_prefill_multisampling, params, tokens, true_length, iters, num_samples)
    multisampling_prefill_average_ms = 1000 * time_in_s / iters
    print(
        f"\nNum samples: {num_samples}\n" f"\tPrefill step average time: {multisampling_prefill_average_ms:.3f} ms\n\n\n\n"
    )
    result_dict[num_samples] = {
        "time_in_ms": multisampling_prefill_average_ms,
    }
  return result_dict


def prefill_insert_benchmark_loop(
    config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters, profile_name
):
  """Inner loop for benchmarking prefill and insert step."""
  prof = profiler.Profiler(config)
  prof.activate(optional_postfix=profile_name)
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  for i in range(iters):
    rng, rng_prefill = jax.random.split(rng)
    decode_state = engine_insert(tokens, true_length, rng_prefill, decode_state, int(i % total_slots), params)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  prof.deactivate()
  return (end - start).total_seconds(), decode_state


def prefill_insert_benchmark(config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters):
  """Handles warmup, running insert benchmark, and printing results."""
  rng = jax.random.PRNGKey(1234)
  for i in range(_WARMUP_ITERS):
    rng, rng_prefill = jax.random.split(rng)
    decode_state = engine_insert(tokens, true_length, rng_prefill, decode_state, int(i % total_slots), params)
  jax.block_until_ready(decode_state)

  print(f"Prefill and insert benchmark results for length {tokens.size}:\n")
  time_in_s, decode_state = prefill_insert_benchmark_loop(
      config, engine_insert, decode_state, params, total_slots, tokens, true_length, iters, f"prefill_insert_{tokens.size}"
  )
  prefill_insert_average_ms = time_in_s / iters * 1000.0
  print(f"\tPrefill + Insert step average time: {prefill_insert_average_ms:.3f} ms\n\n\n\n")
  result_dict = {"time_in_ms": prefill_insert_average_ms}
  return result_dict, decode_state


def ar_benchmark_loop(config, engine_generate, params, decode_state, iters, profile_name):
  """Inner loop for benchmarking ar step."""
  prof = profiler.Profiler(config)
  prof.activate(optional_postfix=profile_name)
  start = datetime.datetime.now()
  rng = jax.random.PRNGKey(1234)
  for _ in range(iters):
    rng, rng_generate = jax.random.split(rng)
    decode_state, _ = engine_generate(params, decode_state, rng_generate)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  prof.deactivate()
  return (end - start).total_seconds(), decode_state


def ar_benchmark(config, engine_generate, params, decode_state, global_batch_size, cache_size, model_size, iters):
  """Handles warmup, running ar benchmark, and printing results."""
  rng = jax.random.PRNGKey(1234)
  for _ in range(_WARMUP_ITERS):
    rng, rng_generate = jax.random.split(rng)
    decode_state, _ = engine_generate(params, decode_state, rng_generate)
  jax.block_until_ready(decode_state)

  time_in_s, decode_state = ar_benchmark_loop(
      config, engine_generate, params, decode_state, iters, profile_name="autoregress"
  )
  seconds_per_step = time_in_s / iters
  ar_average_ms = seconds_per_step * 1000
  total_throughput = global_batch_size / seconds_per_step

  GB_per_step_per_device = (model_size + cache_size) / 1e9 / jax.device_count()
  bw_per_device = GB_per_step_per_device / seconds_per_step
  print(
      f"AutoRegressive results:\n"
      f"\tAR step average time: {ar_average_ms:.3f} ms\n"
      f"\tAR step average time per seq: {ar_average_ms/global_batch_size:.3f} ms\n"
      f"\tAR global batch size: {global_batch_size}\n"
      f"\tAR throughput: {total_throughput:.3f} tokens/second\n"
      f"\tAR memory bandwidth per device: {bw_per_device:.3f} GB/s\n\n\n"
  )

  result_dict = {
      "step_in_ms": ar_average_ms,
      "step_in_ms_per_seq": ar_average_ms / global_batch_size,
      "global_batch_size": global_batch_size,
      "total_throughput_tokens_per_second": total_throughput,
      "bw_per_device_GB_per_second": bw_per_device,
  }
  return result_dict, decode_state


def collate_results(config, results, model_size, cache_size, num_model_params, incl_config=False):
  """Adds model/cache size info and optionally config info to results."""
  results["sizes"] = {
      "model_size_in_gb": model_size / 1e9,
      "cache_size_in_gb": cache_size / 1e9,
      "model_params_in_billions": num_model_params / 1e9,
  }
  if incl_config:
    results["config"] = {}
    for k, v in dict(config.get_keys()).items():
      results["config"][k] = str(v) if k == "dtype" else v  # json fails with original dtype
  return results


def flatten_dict(dictionary, prefix="", sep="_"):
  results = []
  for k, v in dictionary.items():
    new_key = str(prefix) + sep + str(k) if prefix else k
    if isinstance(v, MutableMapping):
      results.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      results.append((new_key, v))
  return dict(results)


def write_results(results, filename, flatten_microbenchmark_results):
  """Write the results microbenchmark results to a json file."""
  if flatten_microbenchmark_results:
    results["flattened_results"] = flatten_dict(results)
  if filename:
    with open(filename, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
  return results


def print_results_for_analyze(results):
  """Print results."""
  print("\nFor usage in analyze_sharegpt.py :")

  if "prefill" in results:
    prefill_bucket_size_to_ms = {}
    for k, v in results["prefill"].items():
      prefill_bucket_size_to_ms[int(k)] = round(v["time_in_ms"], 3)
    print(f"PREFILL_BUCKET_SIZE_TO_MS = {prefill_bucket_size_to_ms}")

  if "prefill-multisampling" in results:
    multi_sampling_prefill_bucket_size_to_ms = {}
    for prefill_length, result_dict in results["prefill-multisampling"].items():
      multi_sampling_prefill_bucket_size_to_ms[int(prefill_length)] = {}
      for num_samples, v in result_dict.items():
        multi_sampling_prefill_bucket_size_to_ms[int(prefill_length)][num_samples] = round(v["time_in_ms"], 3)
    print(f"MULTISAMPLING_PREFILL_BUCKET_SIZE_TO_MS = {multi_sampling_prefill_bucket_size_to_ms}")

  if "insert" in results:
    insert_bucket_size_to_ms = {}
    for k, v in results["insert"].items():
      insert_bucket_size_to_ms[int(k)] = round(v["time_in_ms"], 3)
    print(f"INSERT_BUCKET_SIZE_TO_MS = {insert_bucket_size_to_ms}")

  if "autoregressive" in results:
    print(f"SYSTEM_TIME_PER_DECODE_TOKEN_MS = {results['autoregressive']['step_in_ms_per_seq']}")


def summarize_prefill_result(engine_prefill, params, tokens, true_length):
  """Summarize Prefill result."""
  print(f"Prefill result of length {tokens.size}:\n")
  rng = jax.random.PRNGKey(1234)
  prefill_result, _ = engine_prefill(params, tokens, true_length, rng)
  jax.block_until_ready(prefill_result)
  num_prefill_logits_params, total_prefill_logits_size, avg_prefill_logits_param_size = max_utils.summarize_pytree_data(
      prefill_result["logits"], name="Prefill Logits", raw=True
  )
  num_prefill_cache_params, total_prefill_cache_size, avg_prefill_cache_param_size = max_utils.summarize_pytree_data(
      prefill_result["cache"], name="Prefill Cache"
  )
  del prefill_result
  return {
      "num_logits_params": num_prefill_logits_params,
      "total_logits_size": total_prefill_logits_size,
      "avg_logits_param_size": avg_prefill_logits_param_size,
      "num_cache_params": num_prefill_cache_params,
      "total_cache_size": total_prefill_cache_size,
      "avg_cache_param_size": avg_prefill_cache_param_size,
  }


def run_benchmarks(config):
  """Run microbenchmarks."""
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)
  prefill_lengths = [int(l) for l in config.inference_microbenchmark_prefill_lengths.split(",")]
  stages_to_benchmark = config.inference_microbenchmark_stages.split(",")
  benchmark_loop_iters = config.inference_microbenchmark_loop_iters

  text = config.prompt
  metadata = engine.get_tokenizer()
  tokenizer_model = engine.build_tokenizer(metadata)
  rng, rng_init_decode = jax.random.split(rng)

  generate_executable, params, decode_state_executable = engine.aot_compile(params, pass_rng_shape=True)
  decode_state = decode_state_executable(rng_init_decode)

  _, cache_size, _ = max_utils.summarize_pytree_data(decode_state["cache"], name="Cache")
  num_model_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")

  benchmark_results = {}
  if "prefill" in stages_to_benchmark:
    benchmark_results["prefill-result-sizes"] = {}
    benchmark_results["prefill"] = {}
    benchmark_results["insert"] = {}
    prefill_tokens = {}
    prefill_true_lengths = {}
    prefill_executable = {}
    prefill_insert_executable = {}
    i32_scalar = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))

    for prefill_length in prefill_lengths:
      prefill_tokens[prefill_length], prefill_true_lengths[prefill_length] = tokenizer_model.encode(
          text, is_bos=True, prefill_lengths=[prefill_length]
      )

      key_shape = jax.ShapeDtypeStruct([prefill_length], jax.numpy.dtype("int32"))
      prefill_executable[prefill_length] = (
          jax.jit(
              engine.prefill_aot,
              in_shardings=(engine.param_layouts, None, None, None),
          ).lower(params, key_shape, i32_scalar, rng_shape)
      ).compile(compiler_options=None)

      prefill_insert_executable[prefill_length] = (
          jax.jit(
              engine.prefill_insert,
              in_shardings=(None, None, None, engine.decode_state_layouts, None, engine.param_layouts),
              out_shardings=(engine.decode_state_layouts),
              donate_argnames=("decode_state",),
          ).lower(key_shape, i32_scalar, rng_shape, engine.decode_state_shapes, i32_scalar, params)
      ).compile(compiler_options=None)

      benchmark_results["prefill-result-sizes"][prefill_length] = summarize_prefill_result(
          prefill_executable[prefill_length], params, prefill_tokens[prefill_length], prefill_true_lengths[prefill_length]
      )

    if "prefix_cache" in stages_to_benchmark:
      for prefill_length in prefill_lengths:
        rng_cache = jax.random.PRNGKey(1234)
        prefill_result, _ = prefill_executable[prefill_length](
            params, prefill_tokens[prefill_length], prefill_true_lengths[prefill_length], rng_cache
        )
        prefix_cache_benchmark(
            prefill_result,
            prefill_length,
            prefill_true_lengths[prefill_length],
            config.inference_microbenchmark_prefix_cache_common_prefix_proportion,
            config.inference_microbenchmark_prefix_cache_entries_num,
            benchmark_loop_iters,
        )
        del prefill_result

    for prefill_length in prefill_lengths:
      benchmark_results["prefill"][prefill_length] = prefill_benchmark(
          config,
          prefill_executable[prefill_length],
          params,
          prefill_tokens[prefill_length],
          prefill_true_lengths[prefill_length],
          num_model_params,
          benchmark_loop_iters,
      )

      prefill_insert_time, decode_state = prefill_insert_benchmark(
          config,
          prefill_insert_executable[prefill_length],
          decode_state,
          params,
          engine.max_concurrent_decodes,
          prefill_tokens[prefill_length],
          prefill_true_lengths[prefill_length],
          benchmark_loop_iters,
      )
      benchmark_results["insert"][prefill_length] = {}
      benchmark_results["insert"][prefill_length]["time_in_ms"] = (
          prefill_insert_time["time_in_ms"] - benchmark_results["prefill"][prefill_length]["time_in_ms"]
      )

  if "prefill-multisampling" in stages_to_benchmark:
    benchmark_results["prefill-multisampling"] = {}
    multisampling_prefill_executable = {}
    i32_scalar = jax.ShapeDtypeStruct((), int)
    rng_shape = jax.ShapeDtypeStruct([4], jax.numpy.dtype("uint32"))
    # Compile the program in advance.
    for prefill_length in prefill_lengths:
      key_shape = jax.ShapeDtypeStruct([prefill_length], jax.numpy.dtype("int32"))
      multisampling_prefill_executable[prefill_length] = {}
      for num_samples in config.inference_microbenchmark_num_samples:
        multisampling_prefill_executable[prefill_length][num_samples] = (
            jax.jit(
                engine.prefill_multisampling_aot,
                in_shardings=(engine.param_layouts, None, None, None, None),
                static_argnames=("num_samples",),
            ).lower(params, key_shape, i32_scalar, rng_shape, num_samples, None)
        ).compile(compiler_options=None)

    for prefill_length in prefill_lengths:
      benchmark_results["prefill-multisampling"][prefill_length] = prefill_multisampling_benchmark(
          config,
          multisampling_prefill_executable[prefill_length],
          params,
          prefill_tokens[prefill_length],
          prefill_true_lengths[prefill_length],
          benchmark_loop_iters,
      )

  if "generate" in stages_to_benchmark:
    benchmark_results["autoregressive"], decode_state = ar_benchmark(
        config,
        generate_executable,
        params,
        decode_state,
        engine.max_concurrent_decodes,
        cache_size,
        model_size,
        benchmark_loop_iters,
    )

  results = collate_results(config, benchmark_results, model_size, cache_size, num_model_params)
  print_results_for_analyze(results)
  if config.inference_microbenchmark_log_file_path:
    write_results(
        results,
        filename=config.inference_microbenchmark_log_file_path,
        flatten_microbenchmark_results=_FLATTEN_MICROBENCHMARK_RESULTS,
    )
  return results


def main(argv):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  run_benchmarks(pyconfig.initialize(argv))


if __name__ == "__main__":
  app.run(main)
