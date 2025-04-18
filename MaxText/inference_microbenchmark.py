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
import jax.numpy as jnp
import json
from typing import Any, Dict, Tuple, List, Optional
import uuid # Needed for type hints potentially used by engine methods

from absl import app
from collections.abc import MutableMapping

from jetstream.engine import token_utils
from jetstream.engine import engine_api # For types like ResultTokens
from jetstream.engine import tokenizer_api # For types

from MaxText import max_utils
from MaxText import maxengine
from MaxText import maxtext_utils
from MaxText import prefix_cache
from MaxText import profiler
from MaxText import pyconfig

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

_WARMUP_ITERS = 10 # Increased warmup iterations to be safer with implicit JIT
_FLATTEN_MICROBENCHMARK_RESULTS = False

# Define types for clarity
Params = Any
DecodeState = Any
Config = Any # Represents the pyconfig object or similar config structure
Prefix = Any # From engine_api potentially
ExistingPrefix = Any # From engine_api potentially

# --- Utility Functions ---

def flatten_dict(dictionary: MutableMapping, prefix: str = "", sep: str = "_") -> Dict[str, Any]:
    """Flattens a nested dictionary."""
    results = []
    for k, v in dictionary.items():
        new_key = str(prefix) + sep + str(k) if prefix else str(k)
        if isinstance(v, MutableMapping):
            results.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            results.append((new_key, v))
    return dict(results)

def write_results(results: Dict, filename: str, flatten_microbenchmark_results: bool):
    """Write the results microbenchmark results to a json file."""
    if flatten_microbenchmark_results:
        flat_results = {}
        for key, value in results.items():
             if isinstance(value, MutableMapping) and key not in ["config", "sizes"]:
                 flat_results.update(flatten_dict(value, prefix=key))
             else:
                 flat_results[key] = value
        results["flattened_results"] = flat_results

    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            # Ensure config dictionary is cleaned for JSON
            if "config" in results and isinstance(results["config"], dict):
                 results["config"] = {k: str(v) if isinstance(v, jnp.dtype) else v for k, v in results["config"].items()}
            json.dump(results, f, indent=2, default=str) # Use default=str for safety
    return results

def print_results_for_analyze(results: Dict):
    """Print results formatted for analyze_sharegpt.py."""
    print("\nFor usage in analyze_sharegpt.py :")

    prefill_results = results.get("prefill", {})
    if isinstance(prefill_results, dict) and any(isinstance(k, (int, str)) and k != "result_sizes" for k in prefill_results.keys()):
         prefill_bucket_size_to_ms = {
             # Ensure keys are integers before using them
             int(k): round(v["time_in_ms"], 3)
             for k, v in prefill_results.items() if k != "result_sizes" and isinstance(v, dict) and "time_in_ms" in v
         }
         if prefill_bucket_size_to_ms:
              print(f"PREFILL_BUCKET_SIZE_TO_MS = {prefill_bucket_size_to_ms}")

    multisampling_results = results.get("prefill_multisampling", {})
    if isinstance(multisampling_results, dict) and multisampling_results:
        multi_sampling_prefill_bucket_size_to_ms = {}
        for prefill_length, result_dict in multisampling_results.items():
            # Ensure result_dict is a dictionary
            if isinstance(result_dict, dict):
                multi_sampling_prefill_bucket_size_to_ms[int(prefill_length)] = {
                    int(num_samples): round(v["time_in_ms"], 3)
                    for num_samples, v in result_dict.items() if isinstance(v, dict) and "time_in_ms" in v
                }
        if multi_sampling_prefill_bucket_size_to_ms:
             print(f"MULTISAMPLING_PREFILL_BUCKET_SIZE_TO_MS = {multi_sampling_prefill_bucket_size_to_ms}")

    insert_results = results.get("prefill_insert", {})
    if isinstance(insert_results, dict) and insert_results:
        insert_bucket_size_to_ms = {
            int(k): round(v["time_in_ms"], 3)
            for k, v in insert_results.items() if isinstance(v, dict) and "time_in_ms" in v
        }
        if insert_bucket_size_to_ms:
            print(f"INSERT_BUCKET_SIZE_TO_MS = {insert_bucket_size_to_ms}")

    ar_results = results.get("autoregressive", {})
    if isinstance(ar_results, dict):
        step_ms_per_seq = ar_results.get("step_in_ms_per_seq")
        if step_ms_per_seq is not None:
             try:
                 print(f"SYSTEM_TIME_PER_DECODE_TOKEN_MS = {float(step_ms_per_seq):.3f}")
             except (ValueError, TypeError):
                 print(f"Warning: Could not format SYSTEM_TIME_PER_DECODE_TOKEN_MS value: {step_ms_per_seq}")


def collate_results(config: Config, results: Dict, model_size: float, cache_size: float, num_model_params: int, incl_config: bool = False) -> Dict:
    """Adds model/cache size info and optionally config info to results."""
    results["sizes"] = {
        "model_size_in_gb": model_size / 1e9,
        "cache_size_in_gb": cache_size / 1e9,
        "model_params_in_billions": num_model_params / 1e9,
    }
    if incl_config:
        results["config"] = {}
        # Handle config being an object or a dict
        try:
            # Attempt to get config dictionary, handle different config types
            if hasattr(config, 'get_keys') and callable(config.get_keys):
                 config_dict = config.get_keys()
            elif isinstance(config, dict):
                 config_dict = config
            elif hasattr(config, '__dict__'):
                 config_dict = vars(config)
            else:
                 config_dict = {} # Fallback if config type is unknown

            for k, v in config_dict.items():
                # Filter out non-serializable types if necessary, or convert them
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                     results["config"][k] = v
                elif isinstance(v, jnp.dtype):
                     results["config"][k] = str(v)
                # Add more types or handle complex objects as needed
                # else:
                #     results["config"][k] = f"<non-serializable type: {type(v).__name__}>"

        except Exception as e:
            print(f"Warning: Could not serialize config object fully: {e}")
            results["config"]["error"] = "Could not serialize full config"

    return results

def summarize_prefill_result_data(prefill_result: Dict) -> Dict:
    """Summarizes the data sizes within a prefill result dictionary."""
    logits_data = prefill_result.get("logits")
    cache_data = prefill_result.get("cache")
    summary = {}

    if logits_data is not None:
         try:
             # Removed print_stats=False
             num_logits_params, total_logits_size, avg_logits_param_size = max_utils.summarize_pytree_data(
                 logits_data, name="Prefill Logits", raw=True
             )
             summary.update({
                 "num_logits_params": num_logits_params,
                 "total_logits_size_gb": total_logits_size / 1e9,
                 "avg_logits_param_size_mb": avg_logits_param_size / 1e6,
             })
         except Exception as e:
             print(f"Warning: Could not summarize logits data: {e}")

    if cache_data is not None:
         try:
             # Removed print_stats=False
             num_cache_params, total_cache_size, avg_cache_param_size = max_utils.summarize_pytree_data(
                 cache_data, name="Prefill Cache"
             )
             summary.update({
                 "num_cache_params": num_cache_params,
                 "total_cache_size_gb": total_cache_size / 1e9,
                 "avg_cache_param_size_mb": avg_cache_param_size / 1e6,
             })
         except Exception as e:
             print(f"Warning: Could not summarize cache data: {e}")
    return summary


# --- Benchmarking Functions ---

def prefix_cache_benchmark(
    engine: maxengine.MaxEngine,
    params: Params,
    tokens: jax.Array,
    true_length: int,
    prefill_length: int,
    common_prefix_proportion: float,
    prefix_cache_entries_num: int,
    iters: int
):
    """Handles running prefix cache benchmark, and printing results."""
    print(f"\nPrefix Cache benchmark running for prefill length {prefill_length}...")

    # Run prefill once to get the cache structure/data needed
    print("  Running prefill once to get cache data...")
    rng_cache = jax.random.PRNGKey(1234)
    try:
        # Assuming slot 0 is suitable for getting the representative cache structure
        prefix_result, _ = engine.prefill(
            params=params,
            padded_tokens=tokens,
            true_length=true_length,
            rng=rng_cache,
            slot=0
        )
        jax.block_until_ready(prefix_result)
        print("  Prefill complete.")
    except Exception as e:
        print(f"Error during initial prefill for prefix cache benchmark: {e}")
        return # Cannot proceed without initial cache data

    prefix_cache_data = prefix_result.get("cache")
    if prefix_cache_data is None:
        print("Error: Prefill result did not contain 'cache'. Cannot run prefix cache benchmark.")
        return

    try:
        # Removed print_stats=False
        _, prefix_size_bytes, _ = max_utils.summarize_pytree_data(prefix_cache_data, name="Prefill Cache Data")
    except Exception as e:
        print(f"Warning: Could not calculate size of prefetched cache data: {e}")
        prefix_size_bytes = 1 # Avoid division by zero, but size is unknown

    # Create a representative value structure for the PrefixCache API
    value_for_cache = prefix_cache.Value(
        prefix=prefix_cache_data,
        true_length=true_length,
        padded_length=prefill_length,
        tokens=tuple(i for i in range(prefill_length)), # Dummy tokens
    )

    def copy_jax_array(x):
      # Handles non-jax arrays gracefully
      if isinstance(x, jax.Array):
        return x.copy()
      return x

    def clone_value(val_template):
      # Ensure deep copy of JAX arrays within the structure
      return jax.tree_util.tree_map(copy_jax_array, val_template, is_leaf=lambda x: isinstance(x, jax.Array))

    prefix_size_bytes_gb = prefix_size_bytes / 1024 / 1024 / 1024
    max_bytes = prefix_cache_entries_num * prefix_size_bytes
    # Ensure max_bytes is non-negative
    max_bytes = max(0, max_bytes)
    try:
        prefix_cache_inst = prefix_cache.PrefixCache(hbm_bytes=max_bytes, dram_bytes=max_bytes)
    except Exception as e:
         print(f"Error creating PrefixCache instance: {e}")
         return

    common_len = int(prefill_length * common_prefix_proportion)
    remain_len = prefill_length - common_len
    common_prefix_key = tuple(i for i in range(common_len))

    print(f"  Filling cache with {prefix_cache_entries_num} entries...")
    fill_values = []
    start_fill = time.perf_counter()
    try:
        for c_idx in range(prefix_cache_entries_num):
            # Use a unique key that won't clash with later test keys
            key = common_prefix_key + tuple(1000 + i + c_idx * remain_len for i in range(remain_len))
            # Add a non-common part to ensure uniqueness if common_prefix_proportion is 1.0
            if not remain_len:
                key += (c_idx,)
            new_value = clone_value(value_for_cache)
            prefix_cache_inst.save(key, new_value)
            fill_values.append(new_value) # Keep track to block later
        # Block until all save operations involving JAX arrays are complete
        jax.block_until_ready(fill_values)
    except Exception as e:
        print(f"Error during prefix cache fill: {e}")
        return # Cannot proceed
    finally:
        end_fill = time.perf_counter()
        print(f"  Cache fill took: {end_fill - start_fill:.3f} s")
        del fill_values # Release memory


    # Benchmark Save
    print(f"  Benchmarking save ({iters} iterations)...")
    save_values = []
    save_sec_total = 0
    try:
        for c_idx in range(iters):
            # Create keys that definitely have the common prefix
            key = common_prefix_key + tuple(i + c_idx * remain_len for i in range(remain_len))
            new_value = clone_value(value_for_cache)
            start = time.perf_counter()
            prefix_cache_inst.save(key, new_value)
            jax.block_until_ready(new_value) # Block on the saved value's arrays
            end = time.perf_counter()
            save_sec_total += (end - start)
            save_values.append(new_value) # Keep track only for potential deletion
    except Exception as e:
        print(f"Error during prefix cache save benchmark: {e}")
        save_avg_ms = -1.0 # Indicate error
    else:
        save_avg_ms = save_sec_total * 1000 / iters if iters > 0 else 0.0
    finally:
        del save_values


    # Benchmark Fetch longest prefix key
    print(f"  Benchmarking fetch ({iters} iterations)...")
    # Create a key to query that should match one of the saved keys
    key_load = common_prefix_key + tuple(i + 0 * remain_len for i in range(remain_len)) # Match the first saved key
    matched_key = None
    fetch_sec_total = 0
    try:
        for _ in range(iters):
            start = time.perf_counter()
            matched_key = prefix_cache_inst.fetch_longest_common_prefix_key(key_load)
            end = time.perf_counter() # Fetching key is likely CPU bound
            fetch_sec_total += (end - start)
            # Check immediately if key wasn't found on first try
            if matched_key is None and _ == 0:
                 raise ValueError(f"Failed to fetch a matching key for {key_load} on first attempt.")
    except Exception as e:
         print(f"Error during prefix cache fetch benchmark: {e}")
         fetch_avg_ms = -1.0
    else:
         fetch_avg_ms = fetch_sec_total * 1000 / iters if iters > 0 else 0.0
         if matched_key is None: # Final check
             print(f"Warning: Failed to fetch a matching key for {key_load} during benchmark.")


    # Benchmark Load prefix
    print(f"  Benchmarking load ({iters} iterations)...")
    load_sec_total = 0
    loaded_value_holder = None # To hold the result for blocking
    load_avg_ms = -1.0 # Default to error
    if matched_key is not None: # Only proceed if fetch was successful
        try:
            for _ in range(iters):
                start = time.perf_counter()
                loaded_value = prefix_cache_inst.load(matched_key)
                # Block until loading (potentially involving device transfer) is complete
                jax.block_until_ready(loaded_value)
                end = time.perf_counter()
                load_sec_total += (end - start)
                loaded_value_holder = loaded_value # Keep the last loaded value
        except Exception as e:
             print(f"Error during prefix cache load benchmark: {e}")
        else:
             load_avg_ms = load_sec_total * 1000 / iters if iters > 0 else 0.0
        finally:
             del loaded_value_holder
    else:
        print("Skipping load benchmark because fetch failed.")


    print(
        f"PrefixCaching results for prefill length {prefill_length}:\n"
        f"\tPer prefix size bytes: {prefix_size_bytes_gb:.3f} GB\n"
        f"\tAverage save cache time: {save_avg_ms:.3f} ms\n"
        f"\tAverage fetch longest prefix time: {fetch_avg_ms:.3f} ms\n"
        f"\tAverage load cache time: {load_avg_ms:.3f} ms\n"
    )
    # Clean up resources if possible
    del prefix_cache_inst
    del prefix_result


def prefill_benchmark(
    config: Config,
    engine: maxengine.MaxEngine,
    params: Params,
    tokens: jax.Array,
    true_length: int,
    num_model_params: int,
    iters: int,
    slot: int = 0 # Added slot parameter with a default value
) -> Dict[str, float]:
    """Handles warmup, running prefill benchmark using engine.prefill, and printing results."""
    print(f"\nPrefill benchmark running for length {tokens.size} (slot {slot})...") # Log slot used
    rng = jax.random.PRNGKey(1234)
    result_dict = { # Initialize with defaults in case of errors
        "time_in_ms": -1.0,
        "total_tflops_per_device": 0.0,
        "tflops_per_sec_per_device": 0.0,
        "slot": slot
    }

    # Warmup - Call engine.prefill directly, passing the slot
    print(f"  Warmup ({_WARMUP_ITERS} iterations)...")
    warmup_result = None
    try:
        for _ in range(_WARMUP_ITERS):
            rng, rng_prefill = jax.random.split(rng)
            # Pass slot to engine.prefill
            warmup_result, _ = engine.prefill(
                params=params,
                padded_tokens=tokens,
                true_length=true_length,
                rng=rng_prefill,
                slot=slot # Pass the slot here
            )
        jax.block_until_ready(warmup_result) # Block on the prefix part
    except Exception as e:
        print(f"Error during prefill warmup: {e}")
        return result_dict # Return default error values
    finally:
        del warmup_result

    # Measurement - Call engine.prefill directly, passing the slot
    print(f"  Measuring ({iters} iterations)...")
    rng = jax.random.PRNGKey(1234) # Reset RNG for consistent measurement
    total_time_s = 0.0
    last_result = None
    try:
        for i in range(iters):
            rng, rng_prefill = jax.random.split(rng)
            start_time = time.perf_counter()
            # Call the engine's public prefill method with the slot
            prefill_prefix, prefill_tokens_out = engine.prefill(
                params=params,
                padded_tokens=tokens,
                true_length=true_length,
                rng=rng_prefill,
                slot=slot # Pass the slot here
            )
            # Block on the outputs (prefix contains the cache/logits)
            jax.block_until_ready((prefill_prefix, prefill_tokens_out))
            end_time = time.perf_counter()
            total_time_s += (end_time - start_time)
            last_result = prefill_prefix # Store last result internally
    except Exception as e:
        print(f"Error during prefill measurement: {e}")
        return result_dict # Return default error values
    finally:
        del last_result # Clean up last result after loop

    if iters <= 0:
        return result_dict # Avoid division by zero

    prefill_average_ms = 1000 * total_time_s / iters
    try:
        # Ensure config object has necessary attributes for TFLOPS calculation
        prefill_tflops_per_device, _, _ = maxtext_utils.calculate_prefill_tflops_per_device(num_model_params, tokens.size, config)
        tflops_per_sec_per_device = prefill_tflops_per_device / (prefill_average_ms / 1000.0) if prefill_average_ms > 0 else 0.0
    except AttributeError as e:
         print(f"Warning: Could not calculate TFLOPS due to missing config attribute: {e}")
         prefill_tflops_per_device = 0.0
         tflops_per_sec_per_device = 0.0
    except Exception as e:
        print(f"Warning: Could not calculate TFLOPS: {e}")
        prefill_tflops_per_device = 0.0
        tflops_per_sec_per_device = 0.0


    print(
        f"Prefill results (length {tokens.size}, slot {slot}):\n" # Log slot in results
        f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
        f"\tPrefill total TFLOPs/device: {prefill_tflops_per_device:.3f}\n"
        f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n"
    )
    result_dict = {
        "time_in_ms": prefill_average_ms,
        "total_tflops_per_device": prefill_tflops_per_device,
        "tflops_per_sec_per_device": tflops_per_sec_per_device,
        "slot": slot # Optionally record the slot used in results
    }
    return result_dict


def prefill_multisampling_benchmark(
    config: Config,
    engine: maxengine.MaxEngine,
    params: Params,
    tokens: jax.Array,
    true_length: int,
    iters: int
) -> Dict[int, Dict[str, float]]:
    """Handles warmup, running multi-sampling prefill benchmark using engine.prefill_multisampling."""
    print(f"\nMulti-sampling Prefill benchmark running for length {tokens.size}...")
    rng = jax.random.PRNGKey(1234)
    try:
         num_samples_list = sorted(list(config.inference_microbenchmark_num_samples))
    except AttributeError:
         print("Error: Config missing 'inference_microbenchmark_num_samples'. Skipping multi-sampling benchmark.")
         return {}
    result_dict = {} # Initialize results

    # Warmup
    print(f"  Warmup ({_WARMUP_ITERS} iterations for each sample count)...")
    warmup_result = None
    try:
        for _ in range(_WARMUP_ITERS):
            rng, rng_prefill = jax.random.split(rng)
            for num_samples in num_samples_list:
                 # Call engine's multi-sampling method
                 warmup_result, _ = engine.prefill_multisampling(
                     params=params,
                     padded_tokens=tokens,
                     true_length=true_length,
                     rng=rng_prefill,
                     num_samples=num_samples
                 )
        jax.block_until_ready(warmup_result) # Block only after the last warmup call
    except Exception as e:
         print(f"Error during multi-sampling prefill warmup: {e}")
         # Populate result dict with error indication for all sample counts
         for num_samples in num_samples_list:
             result_dict[num_samples] = {"time_in_ms": -1.0}
         return result_dict
    finally:
         del warmup_result

    # Measurement
    print(f"  Measuring ({iters} iterations for each sample count)...")
    for num_samples in num_samples_list:
        print(f"    Num samples: {num_samples}")
        rng_measure = jax.random.PRNGKey(5678) # Use a different seed for measurement phase
        total_time_s = 0.0
        last_result = None
        try:
            for i in range(iters):
                 rng_measure, rng_prefill_iter = jax.random.split(rng_measure)
                 start_time = time.perf_counter()
                 # Call engine's multi-sampling method
                 last_result, _ = engine.prefill_multisampling(
                     params=params,
                     padded_tokens=tokens,
                     true_length=true_length,
                     rng=rng_prefill_iter,
                     num_samples=num_samples
                 )
                 jax.block_until_ready(last_result) # Block on the prefix part
                 end_time = time.perf_counter()
                 total_time_s += (end_time - start_time)
        except Exception as e:
             print(f"Error during multi-sampling prefill measurement for {num_samples} samples: {e}")
             result_dict[num_samples] = {"time_in_ms": -1.0}
             continue # Move to next sample count
        finally:
             del last_result # Clean up result from this num_samples run

        if iters <= 0:
             multisampling_prefill_average_ms = 0.0
        else:
             multisampling_prefill_average_ms = 1000 * total_time_s / iters

        print(
            f"\tPrefill step average time: {multisampling_prefill_average_ms:.3f} ms"
        )
        result_dict[num_samples] = {
            "time_in_ms": multisampling_prefill_average_ms,
        }

    print("\n")
    return result_dict


def prefill_insert_benchmark(
    config: Config,
    engine: maxengine.MaxEngine,
    initial_decode_state: DecodeState,
    params: Params,
    total_slots: int,
    tokens: jax.Array,
    true_length: int,
    iters: int
) -> Tuple[Dict[str, float], DecodeState]:
    """Handles warmup, running prefill+insert benchmark using engine.prefill_insert."""
    print(f"\nPrefill+Insert benchmark running for length {tokens.size}...")
    rng = jax.random.PRNGKey(1234)
    # Create a copy to avoid modifying the initial state used by other benchmarks if run concurrently
    decode_state = jax.tree_util.tree_map(lambda x: x, initial_decode_state)
    result_output = {"time_in_ms": -1.0} # Default error result

    # Warmup
    print(f"  Warmup ({_WARMUP_ITERS} iterations)...")
    try:
        for i in range(_WARMUP_ITERS):
            rng, rng_prefill = jax.random.split(rng)
            slot = int(i % total_slots) if total_slots > 0 else 0
            # Call engine's prefill_insert method
            decode_state = engine.prefill_insert(
                padded_tokens=tokens,
                true_length=true_length,
                rng=rng_prefill,
                decode_state=decode_state, # Pass state for potential donation
                slot=slot,
                params=params
            )
        jax.block_until_ready(decode_state)
    except Exception as e:
         print(f"Error during prefill_insert warmup: {e}")
         return result_output, initial_decode_state # Return error and original state


    # Measurement
    print(f"  Measuring ({iters} iterations)...")
    prof = profiler.Profiler(config)
    profile_name = f"prefill_insert_{tokens.size}"
    prof.activate(optional_postfix=profile_name)

    rng = jax.random.PRNGKey(1234) # Reset RNG for measurement
    total_time_s = 0.0
    try:
        for i in range(iters):
             rng, rng_prefill_iter = jax.random.split(rng)
             slot = int(i % total_slots) if total_slots > 0 else 0
             start_time = time.perf_counter()
             # Call engine's prefill_insert method
             decode_state = engine.prefill_insert(
                 padded_tokens=tokens,
                 true_length=true_length,
                 rng=rng_prefill_iter,
                 decode_state=decode_state, # Pass state for potential donation
                 slot=slot,
                 params=params
             )
             jax.block_until_ready(decode_state) # Block on the returned state
             end_time = time.perf_counter()
             total_time_s += (end_time - start_time)
    except Exception as e:
         print(f"Error during prefill_insert measurement: {e}")
         prof.deactivate()
         return result_output, decode_state # Return error and current (potentially modified) state

    prof.deactivate()

    if iters <= 0:
        prefill_insert_average_ms = 0.0
    else:
        prefill_insert_average_ms = total_time_s / iters * 1000.0

    print(f"\tPrefill + Insert step average time: {prefill_insert_average_ms:.3f} ms\n")
    result_output = {"time_in_ms": prefill_insert_average_ms}
    return result_output, decode_state # Return the final state


def ar_benchmark(
    config: Config,
    engine: maxengine.MaxEngine,
    params: Params,
    initial_decode_state: DecodeState,
    global_batch_size: int,
    cache_size: float,
    model_size: float,
    iters: int
) -> Tuple[Dict[str, float], DecodeState]:
    """Handles warmup, running autoregressive benchmark using engine.generate."""
    print("\nAutoregressive benchmark running...")
    rng = jax.random.PRNGKey(1234)
    # Create a copy to avoid modifying the initial state used by other benchmarks
    decode_state = jax.tree_util.tree_map(lambda x: x, initial_decode_state)
    result_output = { # Default error results
        "step_in_ms": -1.0,
        "step_in_ms_per_seq": -1.0,
        "global_batch_size": global_batch_size,
        "total_throughput_tokens_per_second": 0.0,
        "bw_per_device_GB_per_second": 0.0,
    }

    # Warmup - Call engine.generate
    print(f"  Warmup ({_WARMUP_ITERS} iterations)...")
    try:
        for _ in range(_WARMUP_ITERS):
            rng, rng_generate = jax.random.split(rng)
            # engine.generate returns Tuple[DecodeState, ResultTokens]
            decode_state, _ = engine.generate(params=params, decode_state=decode_state, rng=rng_generate)
        jax.block_until_ready(decode_state) # Block on the state part
    except Exception as e:
        print(f"Error during autoregressive warmup: {e}")
        return result_output, initial_decode_state # Return error and original state

    # Measurement - Call engine.generate
    print(f"  Measuring ({iters} iterations)...")
    prof = profiler.Profiler(config)
    profile_name = "autoregress"
    prof.activate(optional_postfix=profile_name)

    rng = jax.random.PRNGKey(1234) # Reset RNG for measurement
    total_time_s = 0.0
    ar_results_tokens = None # To hold non-donated output
    try:
        for i in range(iters):
            rng, rng_generate_iter = jax.random.split(rng)
            start_time = time.perf_counter()
            # Call engine's generate method
            decode_state, ar_results_tokens = engine.generate(params=params, decode_state=decode_state, rng=rng_generate_iter)
            # Block on both outputs
            jax.block_until_ready((decode_state, ar_results_tokens))
            end_time = time.perf_counter()
            total_time_s += (end_time - start_time)
    except Exception as e:
         print(f"Error during autoregressive measurement: {e}")
         prof.deactivate()
         del ar_results_tokens
         return result_output, decode_state # Return error and current state
    finally:
         prof.deactivate()
         del ar_results_tokens # Clean up last non-donated result

    if iters <= 0:
        return result_output, decode_state # Avoid division by zero

    seconds_per_step = total_time_s / iters
    ar_average_ms = seconds_per_step * 1000
    total_throughput = global_batch_size / seconds_per_step if seconds_per_step > 0 else 0.0

    device_count = jax.device_count()
    if device_count == 0:
        bw_per_device = 0.0
        print("Warning: jax.device_count() is 0. Cannot calculate bandwidth per device.")
    else:
        GB_per_step_total = (model_size + cache_size) / 1e9
        bw_total = GB_per_step_total / seconds_per_step if seconds_per_step > 0 else 0.0
        bw_per_device = bw_total / device_count

    print(
        f"AutoRegressive results:\n"
        f"\tAR step average time: {ar_average_ms:.3f} ms\n"
        f"\tAR step average time per seq: {ar_average_ms / global_batch_size if global_batch_size > 0 else 0.0:.3f} ms\n"
        f"\tAR global batch size: {global_batch_size}\n"
        f"\tAR throughput: {total_throughput:.3f} tokens/second\n"
        f"\tAR memory bandwidth per device: {bw_per_device:.3f} GB/s\n"
    )

    result_output = {
        "step_in_ms": ar_average_ms,
        "step_in_ms_per_seq": ar_average_ms / global_batch_size if global_batch_size > 0 else 0.0,
        "global_batch_size": global_batch_size,
        "total_throughput_tokens_per_second": total_throughput,
        "bw_per_device_GB_per_second": bw_per_device,
    }
    return result_output, decode_state # Return the final state


# --- Main Execution Logic ---

def run_benchmarks(config: Config):
    """Sets up and runs the requested microbenchmarks."""
    print("Initializing MaxEngine...")
    try:
        engine = maxengine.MaxEngine(config)
    except Exception as e:
        print(f"Fatal: Failed to initialize MaxEngine: {e}")
        return None # Cannot proceed

    rng = jax.random.PRNGKey(1234)
    rng, rng_load, rng_init_state = jax.random.split(rng, 3)

    print("Loading parameters...")
    try:
        params = engine.load_params(rng=rng_load)
        print("Parameters loaded.")
    except Exception as e:
        print(f"Fatal: Failed to load parameters: {e}")
        return None

    print("Initializing decode state...")
    try:
        initial_decode_state = engine.init_decode_state(rng=rng_init_state)
        jax.block_until_ready(initial_decode_state) # Ensure state is ready
        print("Decode state initialized.")
    except Exception as e:
        print(f"Fatal: Failed to initialize decode state: {e}")
        return None

    # Calculate sizes
    print("Calculating model and cache sizes...")
    try:
         # Removed print_stats=False
         num_model_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")
         # Removed print_stats=False
         _, cache_size, _ = max_utils.summarize_pytree_data(initial_decode_state["cache"], name="Cache")
         print(f"Model Size: {model_size/1e9:.3f} GB, Cache Size: {cache_size/1e9:.3f} GB, Num Params: {num_model_params/1e9:.3f} B")
    except Exception as e:
         print(f"Warning: Could not calculate model/cache sizes: {e}")
         num_model_params, model_size, cache_size = 0, 0.0, 0.0 # Set defaults

    # Prepare tokens
    print("Tokenizing prompts...")
    try:
        # Access config attributes safely
        prefill_lengths_str = getattr(config, 'inference_microbenchmark_prefill_lengths', None)
        if prefill_lengths_str is None:
             raise ValueError("Config missing 'inference_microbenchmark_prefill_lengths'")
        prefill_lengths = [int(l) for l in prefill_lengths_str.split(",")]

        text = getattr(config, 'prompt', None)
        if text is None:
             raise ValueError("Config missing 'prompt'")

        metadata = engine.get_tokenizer()
        vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
        prefill_tokens: Dict[int, jax.Array] = {}
        prefill_true_lengths: Dict[int, int] = {}
        for length in prefill_lengths:
            tokens, true_len = token_utils.tokenize_and_pad(
                text, vocab, is_bos=True, prefill_lengths=[length]
            )
            prefill_tokens[length] = tokens
            prefill_true_lengths[length] = true_len
        print("Tokenization complete.")
    except (AttributeError, ValueError) as e:
         print(f"Fatal: Missing configuration value or error during setup: {e}")
         return None
    except Exception as e:
         print(f"Fatal: Error during tokenization setup: {e}")
         return None


    try:
        stages_to_benchmark = config.inference_microbenchmark_stages.split(",")
        benchmark_loop_iters = config.inference_microbenchmark_loop_iters
    except AttributeError as e:
         print(f"Fatal: Missing configuration value (stages or loop iters): {e}")
         return None

    benchmark_results = {}
    # Make a copy of the initial state to be potentially modified by benchmarks
    current_decode_state = jax.tree_util.tree_map(lambda x: x, initial_decode_state)

    # --- Run Benchmarks ---
    print("\nStarting benchmark runs...")

    # Prefill Result Size Summary (Run once per length using engine.prefill)
    if "prefill" in stages_to_benchmark or "prefix_cache" in stages_to_benchmark:
        benchmark_results["prefill_result_sizes"] = {}
        print("\nSummarizing prefill result sizes...")
        rng_summ = jax.random.PRNGKey(9876)
        for length in prefill_lengths:
             print(f"  Prefill length: {length} (using slot 0 for summary)")
             rng_summ, rng_prefill_summ = jax.random.split(rng_summ)
             try:
                 prefill_result_summ, _ = engine.prefill(
                     params=params,
                     padded_tokens=prefill_tokens[length],
                     true_length=prefill_true_lengths[length],
                     rng=rng_prefill_summ,
                     slot=0 # Use slot 0 for size summary
                 )
                 jax.block_until_ready(prefill_result_summ)
                 summary_data = summarize_prefill_result_data(prefill_result_summ)
                 benchmark_results["prefill_result_sizes"][length] = summary_data
                 if summary_data.get('total_logits_size_gb') is not None:
                     print(f"    Logits Size: {summary_data['total_logits_size_gb']:.3f} GB")
                 if summary_data.get('total_cache_size_gb') is not None:
                     print(f"    Cache Size: {summary_data['total_cache_size_gb']:.3f} GB")
                 del prefill_result_summ # Clean up
             except Exception as e:
                 print(f"Warning: Failed to get prefill result for size summary (length {length}): {e}")
                 benchmark_results["prefill_result_sizes"][length] = {"error": str(e)}


    # Prefix Cache Benchmark
    if "prefix_cache" in stages_to_benchmark:
        prefix_cache_prop = getattr(config, 'inference_microbenchmark_prefix_cache_common_prefix_proportion', 0.5)
        prefix_cache_num = getattr(config, 'inference_microbenchmark_prefix_cache_entries_num', 1)
        for length in prefill_lengths:
            prefix_cache_benchmark(
                engine,
                params,
                prefill_tokens[length],
                prefill_true_lengths[length],
                length, # prefill_length
                prefix_cache_prop,
                prefix_cache_num,
                benchmark_loop_iters,
            )

    # Prefill Benchmark (using engine.prefill)
    if "prefill" in stages_to_benchmark:
        benchmark_results["prefill"] = {}
        for length in prefill_lengths:
            # Call prefill_benchmark, passing slot=0 explicitly
            prefill_time_result = prefill_benchmark(
                config,
                engine,
                params,
                prefill_tokens[length],
                prefill_true_lengths[length],
                num_model_params,
                benchmark_loop_iters,
                slot=0 # Pass slot 0 for the pure prefill benchmark
            )
            benchmark_results["prefill"][length] = prefill_time_result

    # Prefill + Insert Benchmark
    if "prefill_insert" in stages_to_benchmark:
        benchmark_results["prefill_insert"] = {}
        # Ensure engine.max_concurrent_decodes is available
        try:
             max_slots = engine.max_concurrent_decodes
        except AttributeError:
             print("Warning: engine.max_concurrent_decodes not found, defaulting to 1 slot.")
             max_slots = 1

        for length in prefill_lengths:
             # Pass the *current* decode state, it will be updated
             prefill_insert_time_result, current_decode_state = prefill_insert_benchmark(
                 config,
                 engine,
                 current_decode_state, # Use and update the state
                 params,
                 max_slots,
                 prefill_tokens[length],
                 prefill_true_lengths[length],
                 benchmark_loop_iters,
             )
             benchmark_results["prefill_insert"][length] = prefill_insert_time_result

    # Prefill Multisample Benchmark
    if "prefill-multisampling" in stages_to_benchmark:
         benchmark_results["prefill-multisampling"] = {}
         for length in prefill_lengths:
             benchmark_results["prefill-multisampling"][length] = prefill_multisampling_benchmark(
                 config,
                 engine,
                 params,
                 prefill_tokens[length],
                 prefill_true_lengths[length],
                 benchmark_loop_iters,
             )

    # Autoregressive Benchmark
    if "generate" in stages_to_benchmark:
        # Uses the state potentially modified by prefill_insert
        try:
             max_slots = engine.max_concurrent_decodes
        except AttributeError:
             print("Warning: engine.max_concurrent_decodes not found, defaulting to 1 slot for global_batch_size.")
             max_slots = 1

        ar_time_result, current_decode_state = ar_benchmark(
            config,
            engine,
            params,
            current_decode_state, # Use and update the state
            max_slots, # Use max_concurrent_decodes as global_batch_size
            cache_size,
            model_size,
            benchmark_loop_iters,
        )
        benchmark_results["autoregressive"] = ar_time_result

    # --- Finalize and Output ---
    print("\nBenchmark runs finished.")
    # Pass config object directly to collate_results
    results = collate_results(config, benchmark_results, model_size, cache_size, num_model_params)
    print_results_for_analyze(results)

    log_file_path = getattr(config, 'inference_microbenchmark_log_file_path', None)
    if log_file_path:
        print(f"Writing results to: {log_file_path}")
        write_results(
            results,
            filename=log_file_path,
            flatten_microbenchmark_results=_FLATTEN_MICROBENCHMARK_RESULTS,
        )
    else:
         print("No log file path specified (inference_microbenchmark_log_file_path), results not written to file.")

    return results


# Updated main function
def main(argv):
    # jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    try:
        # Capture the config object returned by initialize
        config = pyconfig.initialize(argv)
        if config is None:
             print("Fatal: pyconfig.initialize returned None.")
             # Attempt to access FLAGS if using absl.flags as a fallback
             if hasattr(pyconfig, 'FLAGS'):
                 print("Attempting to use pyconfig.FLAGS as config.")
                 config = pyconfig.FLAGS
             else:
                 return 1 # Indicate error

    except Exception as e:
         print(f"Fatal: Failed to initialize configuration: {e}")
         # Attempt fallback to FLAGS if init fails and FLAGS exist
         if hasattr(pyconfig, 'FLAGS'):
             print("Attempting fallback to pyconfig.FLAGS.")
             config = pyconfig.FLAGS
         else:
             return 1 # Indicate error

    # Ensure config is not None before proceeding
    if config is None:
        print("Fatal: Configuration object is None after initialization and fallback attempts.")
        return 1

    # Pass the config object to run_benchmarks
    run_benchmarks(config)
    return 0 # Indicate success


if __name__ == "__main__":
    app.run(main)