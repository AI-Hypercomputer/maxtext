#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""
Script to automate running MaxText inference microbenchmarks for various configurations,
extract multiple performance metrics (throughput, mean/median times),
consolidate redundant metrics, and identify the best performing one.
Full stdout/stderr for each run are saved to log files.
Includes a --dry-run option.
"""

import subprocess
import itertools
import re
import json
import os
import shlex
import argparse
from typing import Dict, List, Any, Tuple, Callable
import jax

jax.config.update("jax_compilation_cache_dir", "/mnt/disks/persist/jax_cache")

# --- Configuration Section ---

PYTHON_EXECUTABLE = "python3"
MAXTEXT_MODULE_TO_RUN = "MaxText.inference_microbenchmark"
BASE_CONFIG_FILE = "MaxText/configs/base.yml" # Ensure this path is correct

COMMON_PARAMS: Dict[str, Any] = {
    "tokenizer_path": "assets/tokenizer.llama2",
    "max_prefill_predict_length": 2048,
    "max_target_length": 8192,
    "model_name": "llama2-70b",
    "ici_fsdp_parallelism": 1,
    "ici_autoregressive_parallelism": 1,
    "ici_tensor_parallelism": -1,
    "scan_layers": False,
    "weight_dtype": "bfloat16",
    "checkpoint_is_quantized": True,
}

SCENARIOS: List[Dict[str, Any]] = [
    # {
    #     "name": "int8kv_dot_product",
    #     "base_params": {"quantize_kvcache": True, "attention": "dot_product", "quantization": "int8"},
    #     "sweep_params": {"per_device_batch_size": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
    # },
    # {
    #     "name": "bf16kv_dot_product",
    #     "base_params": {"quantize_kvcache": False, "attention": "dot_product", "quantization": "int8"},
    #     "sweep_params": {"per_device_batch_size": [3, 4, 5, 6, 7, 8, 9, 10]},
    # },
    {
        "name": "bf16kv_paged_attention",
        "base_params": {"quantize_kvcache": False, "attention": "paged", "pagedattn_tokens_per_page": 32, "quantization": "int8"},
        "sweep_params": {"per_device_batch_size": [4, 5, 6, 7, 8, 9, 10], "pagedattn_num_pages": [13000]}
    }
]

# Define raw metrics to extract. Consolidation will happen in parse_performance_metrics
METRICS_TO_EXTRACT: List[Tuple[str, str, Callable[[str], Any], int]] = [
    ("raw_decode_concise_throughput_tps", r"^\s*Mean Decode Throughput:\s*([\d.]+)\s*tokens/sec", float, re.IGNORECASE | re.MULTILINE),
    ("raw_decode_concise_mean_step_time_ms", r"^\s*Mean Step Time:\s*([\d.]+)\s*ms", float, re.IGNORECASE | re.MULTILINE),
    ("raw_decode_detail_throughput_tps_total", r"Autoregressive \(Generate\) Benchmark Results:.*?^\s*Throughput:\s*([\d.]+)\s*tokens/sec \(total\)", float, re.IGNORECASE | re.DOTALL | re.MULTILINE),
    ("raw_decode_detail_mean_step_time_ms", r"Autoregressive \(Generate\) Benchmark Results:.*?^\s*Mean step time:\s*([\d.]+)\s*ms", float, re.IGNORECASE | re.DOTALL | re.MULTILINE),
    ("raw_decode_detail_median_step_time_ms", r"Autoregressive \(Generate\) Benchmark Results:.*?^\s*Median step time:\s*([\d.]+)\s*ms", float, re.IGNORECASE | re.DOTALL | re.MULTILINE),
]

# Primary metric key used for ranking the "best" configuration (this is a clean, consolidated key)
PRIMARY_OPTIMIZATION_METRIC_KEY = "decode_throughput_tps"
OPTIMIZE_HIGHER_IS_BETTER: bool = True
BENCHMARK_TIMEOUT_SECONDS: int = 1800

# --- Helper Functions ---

def generate_run_name(_scenario_ref_name: str,
                      current_params: Dict[str, Any]) -> str:
    model_name_str = str(current_params.get("model_name", "model")).replace('-', '_')
    name_elements = [model_name_str]
    attention_type = str(current_params.get("attention", "unknown_attn"))
    name_elements.append(attention_type)
    kv_cache_quant = "int8kv" if current_params.get("quantize_kvcache") else "bf16kv"
    name_elements.append(kv_cache_quant)
    if "per_device_batch_size" in current_params:
        name_elements.append(f"pdbs{current_params['per_device_batch_size']}")
    prefill_len = current_params.get("max_prefill_predict_length")
    target_len = current_params.get("max_target_length")
    if prefill_len is not None and target_len is not None:
        name_elements.append(f"{prefill_len}pf_{target_len}tg")
    elif prefill_len is not None:
        name_elements.append(f"{prefill_len}pf")
    elif target_len is not None:
        name_elements.append(f"{target_len}tg")
    if attention_type == "paged":
        if "pagedattn_num_pages" in current_params:
            name_elements.append(f"pgnp{current_params['pagedattn_num_pages']}")
    safe_suffix = "_".join(str(p) for p in name_elements if p)
    safe_suffix = safe_suffix.replace('/', '_').replace(' ', '_')
    return f"microbenchmark_sweep/{safe_suffix}"


def construct_command(scenario_base_params: Dict[str, Any],
                      iteration_params: Dict[str, Any],
                      scenario_name_for_run: str) -> List[str]:
    cmd_parts = [PYTHON_EXECUTABLE, "-m", MAXTEXT_MODULE_TO_RUN, BASE_CONFIG_FILE]
    final_params = {**COMMON_PARAMS, **scenario_base_params, **iteration_params}
    final_params["run_name"] = generate_run_name(scenario_name_for_run, final_params)

    for key, value in final_params.items():
        if isinstance(value, bool):
            cmd_parts.append(f"{key}={str(value).lower()}")
        else:
            cmd_parts.append(f"{key}={value}")
    return cmd_parts

def safe_write_log_file(log_path: str, content: str, log_type: str = "log"):
    if not content:
        return
    try:
        with open(log_path, "w", encoding="utf-8") as f_log:
            f_log.write(content)
    except Exception as e_log:
        print(f"Warning: Could not write {log_type} to {log_path}: {e_log}")


def parse_performance_metrics(output: str) -> Dict[str, Any]:
    """Parses multiple defined metrics from the benchmark output and consolidates them."""
    raw_metrics: Dict[str, Any] = {} 
    final_metrics: Dict[str, Any] = {} 

    for key, pattern_str, converter, flags in METRICS_TO_EXTRACT:
        match = re.search(pattern_str, output, flags)
        if match:
            try:
                raw_metrics[key] = converter(match.group(1))
            except (ValueError, IndexError):
                print(f"Warning: Could not convert/extract value for {key} using pattern {pattern_str}")
    
    # Consolidate Throughput: Prefer "concise" ("raw_decode_concise_throughput_tps"), then "detail"
    concise_tps = raw_metrics.get("raw_decode_concise_throughput_tps")
    detail_tps = raw_metrics.get("raw_decode_detail_throughput_tps_total")
    if concise_tps is not None:
        final_metrics["decode_throughput_tps"] = concise_tps
    elif detail_tps is not None:
        final_metrics["decode_throughput_tps"] = detail_tps

    # Consolidate Mean Step Time: Prefer "concise", then "detail"
    concise_mean_time = raw_metrics.get("raw_decode_concise_mean_step_time_ms")
    detail_mean_time = raw_metrics.get("raw_decode_detail_mean_step_time_ms")
    if concise_mean_time is not None:
        final_metrics["decode_mean_step_time_ms"] = concise_mean_time
    elif detail_mean_time is not None:
        final_metrics["decode_mean_step_time_ms"] = detail_mean_time
        
    # Median Step Time (only one source defined in METRICS_TO_EXTRACT for it)
    if "raw_decode_detail_median_step_time_ms" in raw_metrics:
        final_metrics["decode_median_step_time_ms"] = raw_metrics["raw_decode_detail_median_step_time_ms"]
            
    if not final_metrics:
        print("Warning: No performance metrics extracted and consolidated from output.")
    return final_metrics

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Automate MaxText inference microbenchmarks with dry-run capability."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without actually running them."
    )
    args = parser.parse_args()

    all_results: List[Dict[str, Any]] = []
    commands_generated_count = 0
    
    output_base_dir = "microbenchmark_sweep"
    if not os.path.exists(output_base_dir) and not args.dry_run:
        try:
            os.makedirs(output_base_dir)
            print(f"Created output directory: {output_base_dir}")
        except OSError as e:
            print(f"Warning: Could not create base directory '{output_base_dir}': {e}")

    if args.dry_run:
        print("--- DRY RUN MODE: Commands will be printed but not executed. ---")
    else:
        print("--- LIVE RUN MODE: Executing benchmarks. ---")
        print("Note on Memory Management: Each benchmark configuration is run as a separate process.")
        print("System memory (including JAX/TPU memory associated with a process) is typically reclaimed")
        print("by the OS and runtime when the process terminates, isolating runs from each other.")
        print(f"Full stdout/stderr for each run will be saved to files in the '{output_base_dir}' directory.")

    for scenario in SCENARIOS:
        scenario_name_from_config = scenario['name']
        if not args.dry_run:
            print(f"\n--- Processing Scenario Group: {scenario_name_from_config} ---")
        
        scenario_base_params = scenario.get('base_params', {})
        sweep_params_config = scenario.get('sweep_params', {})
        
        sweep_param_names = list(sweep_params_config.keys())
        value_lists = [sweep_params_config[name] for name in sweep_param_names]
        
        for values_combination in itertools.product(*value_lists):
            commands_generated_count += 1
            current_iteration_params = dict(zip(sweep_param_names, values_combination))
            
            command_args_list = construct_command(scenario_base_params, current_iteration_params, scenario_name_from_config)
            command_str_for_log = shlex.join(command_args_list)
            
            generated_full_run_name_path = ""
            for arg in command_args_list:
                if arg.startswith("run_name="):
                    generated_full_run_name_path = arg.split("=", 1)[1]
                    break
            run_name_slug = os.path.basename(generated_full_run_name_path) if generated_full_run_name_path else f"unknown_run_{commands_generated_count}"

            if args.dry_run:
                print(f"\n[DRY-RUN] Would execute ({commands_generated_count}): {command_str_for_log}")
                print(f"          Log files would be named based on slug: {run_name_slug}")
                all_results.append({
                    "scenario_group_name": scenario_name_from_config,
                    "params": {**COMMON_PARAMS, **scenario_base_params, **current_iteration_params},
                    "performance_metrics": "N/A (Dry Run)",
                    "run_name_used_in_command": generated_full_run_name_path,
                    "log_file_slug": run_name_slug,
                    "command": command_str_for_log
                })
                continue

            print(f"\nExecuting ({commands_generated_count}): {command_str_for_log}")
            print(f"          Log files will be named based on slug: {run_name_slug} (in '{output_base_dir}')")
            
            result_entry: Dict[str, Any] = {
                "scenario_group_name": scenario_name_from_config,
                "params": {**COMMON_PARAMS, **scenario_base_params, **current_iteration_params},
                "performance_metrics": {},
                "run_name_used_in_command": generated_full_run_name_path,
                "log_file_slug": run_name_slug,
                "command": command_str_for_log,
                "status": "Not Run",
                "stdout_log": None, "stderr_log": None,
                "timeout_stdout_log": None, "timeout_stderr_log": None,
                "stdout_snippet_on_parse_error": None, "stderr_snippet_on_error": None,
            }

            try:
                process = subprocess.run(command_args_list,
                                         capture_output=True, text=True, check=False,
                                         timeout=BENCHMARK_TIMEOUT_SECONDS,
                                         encoding='utf-8', errors='replace')
                stdout = process.stdout
                stderr = process.stderr

                if stdout:
                    stdout_log_path = os.path.join(output_base_dir, f"{run_name_slug}.stdout.log")
                    safe_write_log_file(stdout_log_path, stdout, "stdout log")
                    result_entry["stdout_log"] = stdout_log_path
                if stderr:
                    stderr_log_path = os.path.join(output_base_dir, f"{run_name_slug}.stderr.log")
                    safe_write_log_file(stderr_log_path, stderr, "stderr log")
                    result_entry["stderr_log"] = stderr_log_path

                if process.returncode != 0:
                    print(f"Warning: Command exited with non-zero status ({process.returncode}). See '{result_entry.get('stderr_log', 'stderr log file')}' for details.")
                    result_entry["status"] = f"Exec Error (code {process.returncode})"
                    result_entry["stderr_snippet_on_error"] = stderr[:1000].strip() if stderr else ""
                else:
                    parsed_metrics = parse_performance_metrics(stdout)
                    if parsed_metrics: # Check if dict is not empty
                        print(f"Parsed Metrics: {json.dumps(parsed_metrics, indent=2)}")
                        result_entry["performance_metrics"] = parsed_metrics
                        result_entry["status"] = "Success"
                    else:
                        print(f"Failed to parse any performance metrics for this run. See '{result_entry.get('stdout_log', 'stdout log file')}' for full output.")
                        result_entry["status"] = "Parse Error"
                        result_entry["stdout_snippet_on_parse_error"] = stdout[:2000].strip() if stdout else ""
            
            except subprocess.TimeoutExpired as e:
                print(f"Command timed out (limit: {BENCHMARK_TIMEOUT_SECONDS}s): {command_str_for_log}")
                result_entry["status"] = "Timeout"
                if hasattr(e, 'stdout') and e.stdout:
                    timeout_stdout_log_path = os.path.join(output_base_dir, f"{run_name_slug}.timeout.stdout.log")
                    safe_write_log_file(timeout_stdout_log_path, e.stdout, "timeout stdout log")
                    result_entry["timeout_stdout_log"] = timeout_stdout_log_path
                if hasattr(e, 'stderr') and e.stderr:
                    timeout_stderr_log_path = os.path.join(output_base_dir, f"{run_name_slug}.timeout.stderr.log")
                    safe_write_log_file(timeout_stderr_log_path, e.stderr, "timeout stderr log")
                    result_entry["timeout_stderr_log"] = timeout_stderr_log_path
            except Exception as e:
                print(f"An unexpected error occurred while running/parsing: {e}")
                result_entry["status"] = "Unexpected Script Error"
            
            all_results.append(result_entry)

    if args.dry_run:
        print(f"\n--- DRY RUN COMPLETE ---")
        print(f"Total commands that would have been generated: {commands_generated_count}")
        return

    print("\n\n--- Overall Results Summary ---")
    if not all_results:
        print("No benchmark runs were attempted or completed.")
        return

    valid_results = [
        r for r in all_results
        if isinstance(r.get("performance_metrics"), dict) and
           r["performance_metrics"].get(PRIMARY_OPTIMIZATION_METRIC_KEY) is not None
    ]

    if not valid_results:
        print(f"No valid performance results (with primary optimization metric '{PRIMARY_OPTIMIZATION_METRIC_KEY}') obtained.")
    else:
        def get_opt_metric_value(res_dict):
            metrics_dict = res_dict.get("performance_metrics", {})
            opt_val = metrics_dict.get(PRIMARY_OPTIMIZATION_METRIC_KEY)
            if opt_val is not None:
                return opt_val
            return 0 if OPTIMIZE_HIGHER_IS_BETTER else float('inf')

        sorted_results = sorted(valid_results, key=get_opt_metric_value, reverse=OPTIMIZE_HIGHER_IS_BETTER)
        best_result = sorted_results[0]
        
        print(f"\n🏆 Best Configuration (Optimizing for '{PRIMARY_OPTIMIZATION_METRIC_KEY}'):")
        best_opt_value = get_opt_metric_value(best_result)
        print(f"  Primary Optimization Metric Value ({PRIMARY_OPTIMIZATION_METRIC_KEY}): {best_opt_value:.3f}")
        print(f"  Scenario Group: {best_result['scenario_group_name']}")
        print(f"  Run Name (in command): {best_result['run_name_used_in_command']}")
        print(f"  Log File Slug: {best_result['log_file_slug']}")
        if best_result.get("stdout_log"): print(f"  Stdout Log: {best_result['stdout_log']}")
        if best_result.get("stderr_log"): print(f"  Stderr Log: {best_result['stderr_log']}")
        
        print(f"  All Parsed Metrics for Best Run:")
        if isinstance(best_result.get("performance_metrics"), dict):
            for m_key, m_val in best_result['performance_metrics'].items():
                print(f"    {m_key}: {m_val:.3f}" if isinstance(m_val, float) else f"    {m_key}: {m_val}")
        
        print(f"  Key Parameters for Best Run:")
        best_params = best_result['params']
        print(f"    Model: {best_params.get('model_name')}")
        print(f"    Attention: {best_params.get('attention')}")
        print(f"    KV Cache Quant: {'int8' if best_params.get('quantize_kvcache') else 'bf16'}")
        print(f"    Prefill Length: {best_params.get('max_prefill_predict_length')}")
        print(f"    Target Length: {best_params.get('max_target_length')}")
        print(f"    Per Device BS: {best_params.get('per_device_batch_size')}")
        if best_params.get("attention") == "paged":
            print(f"    Paged Attn Num Pages: {best_params.get('pagedattn_num_pages')}")

    print("\n--- All Run Details (Summary) ---")
    results_file = os.path.join(output_base_dir, "microbenchmark_sweep_results.json")
    try:
        serializable_results = []
        for res_item in all_results:
            s_item = res_item.copy()
            if "params" in s_item: s_item["params"] = dict(s_item["params"])
            serializable_results.append(s_item)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Full results (including all parsed metrics and log paths) saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results to JSON file '{results_file}': {e}")

    print("\n{:<28} | {:<7} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<60}".format(
        "Scenario Group", "PDBS", "PagedNP", "Attention", "PrefillL", "TargetL", 
        "TPS(Opt)", "MeanTime", "MedianTime", "Run Name (Command)"
    ))
    print("-" * 180)
    for res in all_results:
        params = res["params"]
        metrics = res.get("performance_metrics", {}) # metrics is now the cleaned dict
        
        opt_metric_val_str = "N/A"
        if isinstance(metrics, dict):
            opt_val = metrics.get(PRIMARY_OPTIMIZATION_METRIC_KEY) # Use the new consolidated key
            if opt_val is not None:
                opt_metric_val_str = f"{opt_val:.2f}"
        elif isinstance(metrics, str): 
            opt_metric_val_str = metrics

        # Use the new consolidated keys for mean and median time
        mean_time_val = metrics.get('decode_mean_step_time_ms')
        median_time_val = metrics.get('decode_median_step_time_ms')

        mean_time_str = f"{mean_time_val:.2f}" if isinstance(mean_time_val, float) else "N/A"
        median_time_str = f"{median_time_val:.2f}" if isinstance(median_time_val, float) else "N/A"

        print("{:<28} | {:<7} | {:<7} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<60}".format(
            res.get("scenario_group_name", "N/A"),
            str(params.get("per_device_batch_size", "N/A")),
            str(params.get("pagedattn_num_pages", "N/A") if params.get("attention") == "paged" else "N/A"),
            str(params.get("attention", "N/A")),
            str(params.get("max_prefill_predict_length", "N/A")),
            str(params.get("max_target_length", "N/A")),
            opt_metric_val_str,
            mean_time_str,
            median_time_str,
            res.get("run_name_used_in_command", "N/A")
        ))

if __name__ == "__main__":
    main()