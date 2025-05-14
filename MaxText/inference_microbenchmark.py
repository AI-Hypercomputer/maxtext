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
import time
import jax
import json
import statistics
# import numpy as np 

from absl import app
from collections.abc import MutableMapping

from jetstream.engine import token_utils

from MaxText import max_utils
from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import profiler
from MaxText import pyconfig

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

_WARMUP_ITERS = 20
_FLATTEN_MICROBENCHMARK_RESULTS = False

# Define stage names as constants
STAGE_PREFILL = "prefill"
STAGE_INSERT = "insert"
STAGE_PREFILL_INSERT = "prefill_insert"
STAGE_GENERATE = "generate"


def _get_iteration_stats(times_sec: list[float]) -> dict:
    """Calculates statistics from a list of iteration times in seconds."""
    if not times_sec:
        return {
            "mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0, "std_dev_ms": 0,
            "iterations": 0, "total_time_s": 0,
        }
    
    times_ms = [t * 1000 for t in times_sec]
    stats_dict = {
        "mean_ms": statistics.mean(times_ms) if times_ms else 0,
        "median_ms": statistics.median(times_ms) if times_ms else 0,
        "min_ms": min(times_ms) if times_ms else 0,
        "max_ms": max(times_ms) if times_ms else 0,
        "std_dev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        "iterations": len(times_ms),
        "total_time_s": sum(times_sec),
    }
    return stats_dict


# pylint: disable=too-many-positional-arguments

# --- Prefill Benchmark ---
def prefill_benchmark_loop(engine, params, tokens, true_length, iters, shared_rng_key):
    """Inner loop for benchmarking prefill step, returns individual times."""
    iter_times_s = []
    current_rng = shared_rng_key
    prefill_result = None
    for _ in range(iters):
        current_rng, rng_prefill = jax.random.split(current_rng)
        start_time = time.perf_counter()
        prefill_result, _ = engine.prefill(
            params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill, slot=0,
        )
        jax.block_until_ready(prefill_result)
        end_time = time.perf_counter()
        iter_times_s.append(end_time - start_time)
    del prefill_result 
    return iter_times_s


def prefill_benchmark(config, engine, params, tokens, true_length, num_model_params, iters, shared_rng_key):
    """Handles warmup, running prefill benchmark, and printing results."""
    print(f"\nRunning Prefill Benchmark for length {tokens.size}...")
    current_rng = shared_rng_key
    prefill_result_warmup = None
    for _ in range(_WARMUP_ITERS):
        current_rng, rng_prefill_warmup = jax.random.split(current_rng)
        prefill_result_warmup, _ = engine.prefill(
            params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill_warmup, slot=0,
        )
    jax.block_until_ready(prefill_result_warmup)
    del prefill_result_warmup

    iter_times_s = prefill_benchmark_loop(engine, params, tokens, true_length, iters, current_rng)
    stats = _get_iteration_stats(iter_times_s)

    prefill_tflops_per_device, _, _ = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_params, tokens.size, config
    )
    tflops_per_sec_per_device = (
        (prefill_tflops_per_device / (stats["mean_ms"] / 1000.0)) if stats["mean_ms"] > 0 else 0
    )

    print(f"Prefill Benchmark Results (length {tokens.size}):")
    print(f"\tIterations: {stats['iterations']}")
    print(f"\tMean time: {stats['mean_ms']:.3f} ms")
    print(f"\tMedian time: {stats['median_ms']:.3f} ms")
    print(f"\tMin time: {stats['min_ms']:.3f} ms")
    print(f"\tMax time: {stats['max_ms']:.3f} ms")
    print(f"\tStd Dev time: {stats['std_dev_ms']:.3f} ms")
    print(f"\tTotal TFLOPs/device (for one prefill): {prefill_tflops_per_device:.3f}")
    print(f"\tTFLOPs/sec/device: {tflops_per_sec_per_device:.3f}")

    result_dict = {
        "stats": stats,
        "total_tflops_per_device": prefill_tflops_per_device,
        "tflops_per_sec_per_device": tflops_per_sec_per_device,
    }
    return result_dict

# --- Insert Benchmark ---
def insert_benchmark_loop(engine, prefill_result_for_insert, initial_decode_state, total_slots, iters, profile_config, profile_name_prefix):
    iter_times_s = []
    current_decode_state = initial_decode_state
    
    prof = profiler.Profiler(profile_config) 
    prof.activate(optional_postfix=f"{profile_name_prefix}_insert_loop")

    for i in range(iters):
        start_time = time.perf_counter()
        current_decode_state = engine.insert(prefill_result_for_insert, current_decode_state, slot=0)
        jax.block_until_ready(current_decode_state)
        end_time = time.perf_counter()
        iter_times_s.append(end_time - start_time)
    
    prof.deactivate()
    return iter_times_s, current_decode_state


def insert_benchmark(config, engine, params, tokens, true_length, initial_decode_state, total_slots, iters, shared_rng_key):
    print(f"\nRunning Insert Benchmark for prefill length {tokens.size}...")
    
    current_rng, rng_prefill_setup = jax.random.split(shared_rng_key)
    prefill_result_for_insert, _ = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill_setup, slot=0,
    )
    jax.block_until_ready(prefill_result_for_insert)

    warmup_decode_state = initial_decode_state 
    for i in range(_WARMUP_ITERS):
        warmup_decode_state = engine.insert(prefill_result_for_insert, warmup_decode_state, slot=0)
    jax.block_until_ready(warmup_decode_state)

    iter_times_s, final_decode_state = insert_benchmark_loop(
        engine, prefill_result_for_insert, warmup_decode_state, 
        total_slots, iters, config, f"insert_{tokens.size}"
    )
    del prefill_result_for_insert 
    stats = _get_iteration_stats(iter_times_s)

    print(f"Insert Benchmark Results (for prefill length {tokens.size}):")
    print(f"\tIterations: {stats['iterations']}")
    print(f"\tMean time: {stats['mean_ms']:.3f} ms")
    print(f"\tMedian time: {stats['median_ms']:.3f} ms")
    print(f"\tMin time: {stats['min_ms']:.3f} ms")
    print(f"\tMax time: {stats['max_ms']:.3f} ms")
    print(f"\tStd Dev time: {stats['std_dev_ms']:.3f} ms")
    
    result_dict = {"stats": stats}
    return result_dict, final_decode_state


# --- Prefill + Insert Benchmark ---
def prefill_and_insert_benchmark_loop(
    engine, params, tokens, true_length, initial_decode_state, total_slots, iters, shared_rng_key, profile_config, profile_name_prefix
):
    iter_times_s = []
    current_rng = shared_rng_key
    current_decode_state = initial_decode_state
    prefill_result = None

    prof = profiler.Profiler(profile_config)
    prof.activate(optional_postfix=f"{profile_name_prefix}_prefill_insert_loop")

    for i in range(iters):
        current_rng, rng_prefill = jax.random.split(current_rng)
        start_time = time.perf_counter()
        prefill_result, _ = engine.prefill(
            params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill, slot=0,
        )
        current_decode_state = engine.insert(prefill_result, current_decode_state, slot=0)
        jax.block_until_ready(current_decode_state) 
        end_time = time.perf_counter()
        iter_times_s.append(end_time - start_time)
        
    del prefill_result 
    prof.deactivate()
    return iter_times_s, current_decode_state


def prefill_and_insert_benchmark(
    config, engine, params, tokens, true_length, initial_decode_state, total_slots, num_model_params, iters, shared_rng_key
):
    print(f"\nRunning Prefill+Insert Benchmark for length {tokens.size}...")
    
    current_rng = shared_rng_key
    warmup_decode_state = initial_decode_state
    prefill_result_warmup = None
    for i in range(_WARMUP_ITERS):
        current_rng, rng_prefill_warmup = jax.random.split(current_rng)
        prefill_result_warmup, _ = engine.prefill(
            params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill_warmup, slot=0,
        )
        warmup_decode_state = engine.insert(prefill_result_warmup, warmup_decode_state, slot=0)
    jax.block_until_ready(warmup_decode_state)
    del prefill_result_warmup

    iter_times_s, final_decode_state = prefill_and_insert_benchmark_loop(
        engine, params, tokens, true_length, warmup_decode_state, 
        total_slots, iters, current_rng, config, f"prefill_insert_{tokens.size}"
    )
    stats = _get_iteration_stats(iter_times_s)

    prefill_tflops_per_device, _, _ = maxtext_utils.calculate_prefill_tflops_per_device(
        num_model_params, tokens.size, config
    )
    tflops_per_sec_per_device = (
        (prefill_tflops_per_device / (stats["mean_ms"] / 1000.0)) if stats["mean_ms"] > 0 else 0
    )
    
    print(f"Prefill+Insert Benchmark Results (length {tokens.size}):")
    print(f"\tIterations: {stats['iterations']}")
    print(f"\tMean time: {stats['mean_ms']:.3f} ms")
    print(f"\tMedian time: {stats['median_ms']:.3f} ms")
    print(f"\tMin time: {stats['min_ms']:.3f} ms")
    print(f"\tMax time: {stats['max_ms']:.3f} ms")
    print(f"\tStd Dev time: {stats['std_dev_ms']:.3f} ms")
    print(f"\tTotal TFLOPs/device (for prefill part): {prefill_tflops_per_device:.3f}")
    print(f"\tTFLOPs/sec/device (based on combined time): {tflops_per_sec_per_device:.3f}")

    result_dict = {
        "stats": stats,
        "prefill_total_tflops_per_device": prefill_tflops_per_device,
        "tflops_per_sec_per_device": tflops_per_sec_per_device,
    }
    return result_dict, final_decode_state


# --- Autoregressive (Generate) Benchmark ---
def ar_benchmark_loop(engine, params, initial_decode_state, iters, shared_rng_key, profile_config, profile_name):
    iter_times_s = []
    current_rng = shared_rng_key
    current_decode_state = initial_decode_state
    prof = profiler.Profiler(profile_config)
    prof.activate(optional_postfix=profile_name)
    
    for _ in range(iters):
        current_rng, rng_generate = jax.random.split(current_rng)
        start_time = time.perf_counter()
        current_decode_state, _ = engine.generate(params, current_decode_state, rng=rng_generate)
        jax.block_until_ready(current_decode_state) 
        end_time = time.perf_counter()
        iter_times_s.append(end_time - start_time)
        
    prof.deactivate()
    return iter_times_s, current_decode_state


def ar_benchmark(config, engine, params, initial_decode_state, global_batch_size, cache_byte_size, model_byte_size, iters, shared_rng_key):
    print(f"\nRunning Autoregressive (Generate) Benchmark...")
    
    current_rng = shared_rng_key
    warmup_decode_state = initial_decode_state
    for _ in range(_WARMUP_ITERS):
        current_rng, rng_generate_warmup = jax.random.split(current_rng)
        warmup_decode_state, _ = engine.generate(params, warmup_decode_state, rng=rng_generate_warmup)
    jax.block_until_ready(warmup_decode_state)

    iter_times_s, final_decode_state = ar_benchmark_loop(
        engine, params, warmup_decode_state, iters, current_rng, config, profile_name="autoregress"
    )
    stats = _get_iteration_stats(iter_times_s)
    
    mean_time_s_per_step = stats["mean_ms"] / 1000.0 if stats["mean_ms"] > 0 else float('inf')
    
    total_throughput_tps = global_batch_size / mean_time_s_per_step if mean_time_s_per_step > 0 else 0
    
    bytes_per_step_total = model_byte_size + cache_byte_size 
    gb_per_step_per_device = (bytes_per_step_total / 1e9) / jax.device_count() if jax.device_count() > 0 else 0
    bw_per_device_gb_s = gb_per_step_per_device / mean_time_s_per_step if mean_time_s_per_step > 0 else 0

    print("Autoregressive (Generate) Benchmark Results:")
    print(f"\tIterations: {stats['iterations']}")
    print(f"\tMean step time: {stats['mean_ms']:.3f} ms")
    print(f"\tMedian step time: {stats['median_ms']:.3f} ms")
    print(f"\tMin step time: {stats['min_ms']:.3f} ms")
    print(f"\tMax step time: {stats['max_ms']:.3f} ms")
    print(f"\tStd Dev step time: {stats['std_dev_ms']:.3f} ms")
    print(f"\tMean step time per sequence: {(stats['mean_ms']/global_batch_size if global_batch_size > 0 else 0):.3f} ms")
    print(f"\tGlobal batch size (concurrent sequences): {global_batch_size}")
    print(f"\tThroughput: {total_throughput_tps:.3f} tokens/sec (total)")
    print(f"\tMemory Bandwidth per device: {bw_per_device_gb_s:.3f} GB/s (approx.)")

    result_dict = {
        "stats": stats,
        "stats_per_seq": {
            "mean_ms_per_seq": (stats['mean_ms']/global_batch_size if global_batch_size > 0 else 0),
        },
        "global_batch_size": global_batch_size,
        "total_throughput_tokens_per_second": total_throughput_tps,
        "bw_per_device_GB_per_second": bw_per_device_gb_s,
    }
    return result_dict, final_decode_state


# --- Result Aggregation and Output ---
def collate_results(config, results, model_size_bytes, cache_size_bytes, num_model_params, incl_config=False):
    """Adds model/cache size info and optionally config info to results."""
    results["model_and_cache_sizes"] = {
        "model_size_in_gb": model_size_bytes / 1e9,
        "cache_size_in_gb": cache_size_bytes / 1e9, 
        "model_params_in_billions": num_model_params / 1e9,
    }
    if incl_config:
        results["config_summary"] = {}
        # Use config.get_keys() which is the method provided by MaxText's PyConfig
        # to get a dictionary of configuration keys and values.
        config_items = config.get_keys() # This returns a dict-like object or a dict
        
        for k, v in dict(config_items).items(): # Ensure it's a dict for .items()
            # Ensure dtype is string, and other values are passed as is if simple
            if k == "dtype":
                results["config_summary"][k] = str(v)
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                 results["config_summary"][k] = v
            else:
                # For more complex objects, convert to string to avoid JSON serialization errors
                results["config_summary"][k] = str(v)
    return results


def flatten_dict(dictionary, prefix="", sep="_"):
    """Flattens a nested dictionary."""
    results_list = [] 
    for k, v in dictionary.items():
        new_key = str(prefix) + sep + str(k) if prefix else str(k)
        if isinstance(v, MutableMapping):
            results_list.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            results_list.append((new_key, v))
    return dict(results_list)


def write_results(results, filename, flatten_microbenchmark_results):
    """Write the results microbenchmark results to a json file."""
    if flatten_microbenchmark_results:
        results["flattened_summary"] = flatten_dict(results) 
    if filename:
        print(f"\nWriting benchmark results to: {filename}")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def print_results_for_analyze(results):
    """Print results in a format potentially usable by other scripts (e.g., analyze_sharegpt.py)."""
    print("\nSummary for analyze_sharegpt.py (using mean times):")

    if STAGE_PREFILL in results:
        prefill_bucket_size_to_ms = {}
        for k, v_dict in results[STAGE_PREFILL].items(): 
             if isinstance(v_dict, dict) and "stats" in v_dict and "mean_ms" in v_dict["stats"]:
                prefill_bucket_size_to_ms[int(k)] = round(v_dict["stats"]["mean_ms"], 3)
        if prefill_bucket_size_to_ms:
            print(f"PREFILL_BUCKET_SIZE_TO_MS = {prefill_bucket_size_to_ms}")

    if STAGE_INSERT in results:
        insert_bucket_size_to_ms = {}
        for k, v_dict in results[STAGE_INSERT].items(): 
            if isinstance(v_dict, dict) and "stats" in v_dict and "mean_ms" in v_dict["stats"]:
                insert_bucket_size_to_ms[int(k)] = round(v_dict["stats"]["mean_ms"], 3)
        if insert_bucket_size_to_ms:
            print(f"INSERT_BUCKET_SIZE_TO_MS = {insert_bucket_size_to_ms}")
    elif STAGE_PREFILL_INSERT in results and STAGE_PREFILL in results:
        print(f"Note: INSERT_BUCKET_SIZE_TO_MS derived from {STAGE_PREFILL_INSERT} and {STAGE_PREFILL} because {STAGE_INSERT} not run or has no results.")
        derived_insert_bucket_size_to_ms = {}
        for k_len_str, pi_v_dict in results[STAGE_PREFILL_INSERT].items():
            if k_len_str in results[STAGE_PREFILL]:
                p_v_dict = results[STAGE_PREFILL][k_len_str]
                if (isinstance(pi_v_dict, dict) and "stats" in pi_v_dict and "mean_ms" in pi_v_dict["stats"] and
                    isinstance(p_v_dict, dict) and "stats" in p_v_dict and "mean_ms" in p_v_dict["stats"]):
                    derived_insert_ms = pi_v_dict["stats"]["mean_ms"] - p_v_dict["stats"]["mean_ms"]
                    derived_insert_bucket_size_to_ms[int(k_len_str)] = round(max(0, derived_insert_ms), 3)
        if derived_insert_bucket_size_to_ms:
             print(f"DERIVED_INSERT_BUCKET_SIZE_TO_MS = {derived_insert_bucket_size_to_ms}")


    if STAGE_GENERATE in results:
        ar_data = results[STAGE_GENERATE]
        if isinstance(ar_data, dict):
            ar_stats_per_seq = ar_data.get("stats_per_seq", {})
            mean_ms_per_seq = ar_stats_per_seq.get("mean_ms_per_seq", 0)
            print(f"SYSTEM_TIME_PER_DECODE_TOKEN_MS = {round(mean_ms_per_seq, 3)}")


def summarize_prefill_result_data(engine, params, tokens, true_length, shared_rng_key):
    """Summarize Prefill result (logits and cache sizes)."""
    print(f"\nSummarizing Prefill Result Data for prefill length {tokens.size}:")
    current_rng, rng_prefill_summary = jax.random.split(shared_rng_key)
    prefill_result_obj, _ = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill_summary, slot=0,
    )
    jax.block_until_ready(prefill_result_obj)
    
    summary = {}
    if "logits" in prefill_result_obj:
        # raw=True: summarize_pytree_data returns values, doesn't print summary itself
        num_logits_params, total_logits_size, avg_logits_param_size = max_utils.summarize_pytree_data(
            prefill_result_obj["logits"], name="Prefill Logits", raw=True 
        )
        summary["logits_summary"] = {
            "num_params": num_logits_params,
            "total_size_bytes": total_logits_size,
            "avg_param_size_bytes": avg_logits_param_size,
        }

    if "cache" in prefill_result_obj: 
        # No raw=True: summarize_pytree_data will print its summary using the name
        num_cache_params, total_cache_size, avg_cache_param_size = max_utils.summarize_pytree_data(
            prefill_result_obj["cache"], name="Prefill Cache Output" 
        )
        summary["cache_output_summary"] = {
            "num_params": num_cache_params,
            "total_size_bytes": total_cache_size,
            "avg_param_size_bytes": avg_cache_param_size,
        }
    del prefill_result_obj
    return summary


# --- Main Benchmark Runner ---
def run_benchmarks(config): 
    """Run microbenchmarks based on configuration."""
    engine = maxengine.MaxEngine(config)
    
    master_rng_seed = 1234 
    top_level_rng = jax.random.PRNGKey(master_rng_seed)

    top_level_rng, rng_load_params = jax.random.split(top_level_rng)
    params = engine.load_params(rng_load_params)
    
    prefill_lengths_str = config.inference_microbenchmark_prefill_lengths
    prefill_lengths = [int(l) for l in prefill_lengths_str.split(",") if l.strip()] if prefill_lengths_str else []
    
    stages_to_benchmark_str = config.inference_microbenchmark_stages
    stages_to_benchmark = [s.strip() for s in stages_to_benchmark_str.split(",") if s.strip()] if stages_to_benchmark_str else []
    
    benchmark_loop_iters = config.inference_microbenchmark_loop_iters
    text_prompt = config.prompt

    metadata = engine.get_tokenizer()
    vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

    top_level_rng, rng_init_decode = jax.random.split(top_level_rng)
    master_decode_state = engine.init_decode_state(rng=rng_init_decode)

    print("\nSummarizing Model Parameters:")
    _, model_byte_size, _ = max_utils.summarize_pytree_data(params, name="Model Parameters")
    print("\nSummarizing Max Cache Capacity:")
    _, max_cache_byte_size, _ = max_utils.summarize_pytree_data(master_decode_state["cache"], name="Max Cache Capacity")
    
    num_model_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    benchmark_results = {}
    tokenized_prompts = {}
    if any(s in stages_to_benchmark for s in [STAGE_PREFILL, STAGE_INSERT, STAGE_PREFILL_INSERT]):
        for length in prefill_lengths:
            current_max_prefill_length = length 
            padded_tokens, true_length = token_utils.tokenize_and_pad(
                text_prompt, vocab, is_bos=True, prefill_lengths=[length], max_prefill_length=current_max_prefill_length
            )
            tokenized_prompts[length] = (padded_tokens, true_length)

    if any(s in stages_to_benchmark for s in [STAGE_PREFILL, STAGE_INSERT, STAGE_PREFILL_INSERT]):
        if prefill_lengths: 
            benchmark_results["prefill_result_summaries"] = {}
            for pl_length in prefill_lengths:
                if pl_length in tokenized_prompts:
                    tokens, true_len = tokenized_prompts[pl_length]
                    top_level_rng, rng_for_summary = jax.random.split(top_level_rng)
                    benchmark_results["prefill_result_summaries"][str(pl_length)] = summarize_prefill_result_data(
                        engine, params, tokens, true_len, rng_for_summary
                    )

    current_run_rng = top_level_rng 

    if STAGE_PREFILL in stages_to_benchmark:
        benchmark_results[STAGE_PREFILL] = {}
        for pl_length in prefill_lengths:
            if pl_length in tokenized_prompts:
                tokens, true_len = tokenized_prompts[pl_length]
                current_run_rng, stage_rng = jax.random.split(current_run_rng)
                benchmark_results[STAGE_PREFILL][str(pl_length)] = prefill_benchmark(
                    config, engine, params, tokens, true_len,
                    num_model_params, benchmark_loop_iters, stage_rng
                )

    temp_decode_state_for_stages = master_decode_state 

    if STAGE_PREFILL_INSERT in stages_to_benchmark:
        benchmark_results[STAGE_PREFILL_INSERT] = {}
        for pl_length in prefill_lengths:
            if pl_length in tokenized_prompts:
                tokens, true_len = tokenized_prompts[pl_length]
                current_run_rng, stage_rng = jax.random.split(current_run_rng)
                results, temp_decode_state_for_stages = prefill_and_insert_benchmark(
                    config, engine, params, tokens, true_len,
                    temp_decode_state_for_stages, 
                    engine.max_concurrent_decodes,
                    num_model_params, benchmark_loop_iters, stage_rng
                )
                benchmark_results[STAGE_PREFILL_INSERT][str(pl_length)] = results 
        master_decode_state = temp_decode_state_for_stages 

    if STAGE_INSERT in stages_to_benchmark:
        benchmark_results[STAGE_INSERT] = {}
        for pl_length in prefill_lengths:
            if pl_length in tokenized_prompts:
                tokens, true_len = tokenized_prompts[pl_length]
                current_run_rng, stage_rng = jax.random.split(current_run_rng)
                results, temp_decode_state_for_stages = insert_benchmark(
                    config, engine, params, tokens, true_len,
                    temp_decode_state_for_stages, 
                    engine.max_concurrent_decodes,
                    benchmark_loop_iters, stage_rng
                )
                benchmark_results[STAGE_INSERT][str(pl_length)] = results 
        master_decode_state = temp_decode_state_for_stages 

    if STAGE_GENERATE in stages_to_benchmark:
        current_run_rng, stage_rng = jax.random.split(current_run_rng)
        ar_results, master_decode_state = ar_benchmark( 
            config, engine, params, master_decode_state,
            engine.max_concurrent_decodes, max_cache_byte_size, model_byte_size,
            benchmark_loop_iters, stage_rng
        )
        benchmark_results[STAGE_GENERATE] = ar_results

    final_results = collate_results(
        config, benchmark_results, model_byte_size, max_cache_byte_size, num_model_params, incl_config=True
    )
    
    print_results_for_analyze(final_results)
    
    log_file_path = config.inference_microbenchmark_log_file_path
    if log_file_path: 
        write_results(
            final_results,
            filename=log_file_path,
            flatten_microbenchmark_results=_FLATTEN_MICROBENCHMARK_RESULTS,
        )
    
    return final_results


def main(argv):
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    config = pyconfig.initialize(argv) 

    print(f"JAX process index: {jax.process_index()}, host count: {jax.process_count()}")
    print(f"JAX local devices: {jax.local_devices()}")
    print(f"JAX global devices: {jax.devices()}")

    if jax.process_count() > 1:
       print("Multi-process JAX environment detected.")
       jax.distributed.initialize() 
    elif jax.device_count() > jax.local_device_count(): 
       print("Multi-host JAX environment detected (single process per host).")
       jax.distributed.initialize()

    run_benchmarks(config)


if __name__ == "__main__":
    app.run(main)