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
Standalone script to predict optimal MaxText inference configurations
and provide theoretical roofline performance estimates.

NOTE: The Paged Attention memory calculation herein ASSUMES an
implementation where K/V Tensors are instantiated PER LAYER,
with a shape determined by the 'pagedattn_num_pages' config setting.

This version iterates over Target Effective KV Capacity Ratios to
explore trade-offs between KV cache size and batch size.
"""

import argparse
import math
import sys
from typing import Dict, Any, Tuple, List, MutableMapping

# Attempt to import MaxText components.
try:
    from MaxText import pyconfig
except ImportError:
    print("Error: MaxText components (pyconfig) not found. Ensure MaxText is in your PYTHONPATH or installed.")
    sys.exit(1)

EFFECTIVE_BYTES_INT8_WEIGHTS = 1.008
EFFECTIVE_BYTES_INT8_KV_CACHE = 1.002

def get_bytes_per_dtype(
    base_dtype_str, is_weight=False, is_kv_cache=False, config=None
    ):
     if config is None:
         # print(f"Warning: get_bytes_per_dtype called without config. Using base_dtype_str: {base_dtype_str}")
         dtype_to_check = base_dtype_str
     elif is_weight:
         if getattr(config, 'checkpoint_is_quantized', False) and getattr(config, 'quantization', None) == 'int8':
              return EFFECTIVE_BYTES_INT8_WEIGHTS
         dtype_to_check = config.weight_dtype
     elif is_kv_cache:
         if getattr(config, 'quantize_kvcache', False):
              kv_quant_dtype = getattr(config, 'kv_quant_dtype', 'int8')
              if kv_quant_dtype == 'int8':
                   return EFFECTIVE_BYTES_INT8_KV_CACHE
              dtype_to_check = kv_quant_dtype
         else:
              dtype_to_check = config.dtype
     else:
         dtype_to_check = base_dtype_str # Should be config.dtype

     if dtype_to_check == "bfloat16" or dtype_to_check == "float16": return 2
     if dtype_to_check == "float32": return 4
     if dtype_to_check == "int8": return 1
     # print(f"Warning: Unknown dtype_to_check '{dtype_to_check}' in get_bytes_per_dtype. Assuming 2 bytes (bf16).")
     return 2

def calculate_num_params_from_config(config, model_dims):
    vocab_size = config.vocab_size
    emb_dim = model_dims["emb_dim"]
    mlp_dim = model_dims["mlp_dim"]
    num_decoder_layers = model_dims["num_decoder_layers"]
    num_query_heads = model_dims["num_query_heads"]
    num_kv_heads = model_dims["num_kv_heads"]
    head_dim = model_dims["head_dim"]

    if emb_dim <= 0 or num_decoder_layers <= 0:
         raise ValueError(f"emb_dim ({emb_dim}) and num_decoder_layers ({num_decoder_layers}) must be positive.")
    if num_query_heads <=0 or num_kv_heads <=0 or head_dim <=0:
          raise ValueError(f"Attention dims (num_query_heads {num_query_heads}, num_kv_heads {num_kv_heads}, head_dim {head_dim}) must be positive.")

    params = vocab_size * emb_dim
    if not config.logits_via_embedding:
        params += emb_dim * vocab_size
    layer_attn_params = (emb_dim * num_query_heads * head_dim) + \
                         (emb_dim * num_kv_heads * head_dim) * 2 + \
                         (num_query_heads * head_dim * emb_dim)
    mlp_activations_list = getattr(config, 'mlp_activations', ["gelu"])
    num_mlp_up_projections = 2 if isinstance(mlp_activations_list, list) and len(mlp_activations_list) > 1 else 1
    layer_mlp_params = (emb_dim * mlp_dim * num_mlp_up_projections) + \
                         (mlp_dim * emb_dim)
    layer_norm_params = (2 * emb_dim) * 2
    params_per_layer = layer_attn_params + layer_mlp_params + layer_norm_params
    params += num_decoder_layers * params_per_layer
    params += 2 * emb_dim # For final layernorm
    return int(params)

def estimate_model_weights_memory(config, num_total_params):
    bytes_per_param = get_bytes_per_dtype(None, is_weight=True, config=config)
    return num_total_params * bytes_per_param

def get_effective_model_dims(config):
    scale = getattr(config, 'global_parameter_scale', 1)
    base_emb_dim = getattr(config, 'base_emb_dim', 0); base_num_query_heads = getattr(config, 'base_num_query_heads', 0)
    base_num_kv_heads = getattr(config, 'base_num_kv_heads', 0); base_mlp_dim = getattr(config, 'base_mlp_dim', 0)
    base_num_decoder_layers = getattr(config, 'base_num_decoder_layers', 0); head_dim_cfg = getattr(config, 'head_dim', 0)
    emb_dim = getattr(config, 'emb_dim',0) or base_emb_dim * scale
    num_query_heads = getattr(config, 'num_query_heads',0) or base_num_query_heads * scale
    num_kv_heads = getattr(config, 'num_kv_heads',0) or base_num_kv_heads * scale
    if num_kv_heads == 0 and num_query_heads > 0 : num_kv_heads = num_query_heads
    mlp_dim = getattr(config, 'mlp_dim',0) or base_mlp_dim * scale
    num_decoder_layers = getattr(config, 'num_decoder_layers',0) or base_num_decoder_layers
    head_dim = head_dim_cfg
    if head_dim <= 0 :
        if num_query_heads > 0 and emb_dim > 0: head_dim = emb_dim // num_query_heads
        elif emb_dim > 0 and base_num_query_heads > 0 and scale > 0 :
             scaled_num_query_heads = base_num_query_heads * scale
             if scaled_num_query_heads > 0: head_dim = emb_dim // scaled_num_query_heads
             else: head_dim = 128;
        else: head_dim = 128;
    dims = {"emb_dim": int(emb_dim), "num_query_heads": int(num_query_heads), "num_kv_heads": int(num_kv_heads),
            "mlp_dim": int(mlp_dim), "num_decoder_layers": int(num_decoder_layers), "head_dim": int(head_dim)}
    if not all(dims[k] > 0 for k in dims):
        print("CRITICAL Warning: Some vital model dimensions are zero/negative. Estimates will be incorrect or fail.")
    return dims

def estimate_total_activation_memory(global_batch_size, seq_len, config, model_dims, verbose=True):
    if global_batch_size == 0 or seq_len == 0: return 0
    bytes_per_element = get_bytes_per_dtype(config.dtype, is_weight=False, is_kv_cache=False, config=config)
    num_layers = model_dims["num_decoder_layers"]
    if num_layers == 0: return 0
    layer_input_mem = global_batch_size * seq_len * model_dims["emb_dim"] * bytes_per_element
    attn_output_mem = global_batch_size * seq_len * model_dims["num_query_heads"] * model_dims["head_dim"] * bytes_per_element
    mlp_hidden_mem = global_batch_size * seq_len * model_dims["mlp_dim"] * bytes_per_element
    remat_policy = getattr(config, 'remat_policy', 'full')
    live_per_layer_approx = 0
    if remat_policy == 'full': live_per_layer_approx = 1.5 * max(layer_input_mem, attn_output_mem, mlp_hidden_mem)
    elif 'offloaded' in remat_policy : live_per_layer_approx = max(layer_input_mem, attn_output_mem, mlp_hidden_mem) * 0.15
    elif remat_policy == 'minimal': live_per_layer_approx = layer_input_mem + attn_output_mem + mlp_hidden_mem
    else: live_per_layer_approx = layer_input_mem + max(attn_output_mem, mlp_hidden_mem) * 0.7
    effective_layers_for_memory = 0
    scan_layers_flag = getattr(config, 'scan_layers', True)
    if remat_policy == 'full': effective_layers_for_memory = 2.0 if scan_layers_flag else 3.0 
    elif scan_layers_flag: effective_layers_for_memory = 2.5 
    else:
        if remat_policy == 'minimal':
            if verbose: print("Warning: scan_layers=False and remat_policy='minimal' implies very high activation memory.")
            effective_layers_for_memory = num_layers
        else: effective_layers_for_memory = num_layers * 0.6 
    total_activation_mem = live_per_layer_approx * effective_layers_for_memory * 1.25
    return total_activation_mem

def estimate_kv_cache_memory_traditional(global_batch_size, max_len, config, model_dims):
    if global_batch_size == 0 or max_len == 0 : return 0
    if model_dims["num_decoder_layers"] == 0 or model_dims["num_kv_heads"] == 0 or model_dims["head_dim"] == 0: return 0
    bytes_per_element = get_bytes_per_dtype(config.dtype, is_kv_cache=True, config=config)
    single_cache_layer_elements = 2 * global_batch_size * max_len * model_dims["num_kv_heads"] * model_dims["head_dim"]
    return model_dims["num_decoder_layers"] * single_cache_layer_elements * bytes_per_element

def estimate_kv_cache_memory_paged(num_pages_dimension, config, model_dims):
    if num_pages_dimension == 0: return 0
    tokens_pg = getattr(config, 'pagedattn_tokens_per_page', 32)
    bytes_kv_elem = get_bytes_per_dtype(config.dtype, is_kv_cache=True, config=config)
    mem_one_tensor_per_layer = model_dims["num_kv_heads"] * \
                               num_pages_dimension * \
                               tokens_pg * \
                               model_dims["head_dim"] * \
                               bytes_kv_elem
    total_paged_kv_mem = model_dims["num_decoder_layers"] * 2 * mem_one_tensor_per_layer
    return total_paged_kv_mem

def calculate_inference_flops(config, model_dims, num_model_params, batch_size, seq_len, is_prefill_step):
    if batch_size == 0: return 0
    if num_model_params == 0 and (model_dims["num_decoder_layers"] == 0) : return 0
    learnable_weight_flops_per_seq = 2 * num_model_params * (seq_len if is_prefill_step else 1)
    q_len = seq_len if is_prefill_step else 1
    kv_len = seq_len if is_prefill_step else config.max_target_length 
    attn_qkt_flops_per_layer_per_seq = model_dims["num_query_heads"] * q_len * model_dims["head_dim"] * kv_len * 2
    attn_av_flops_per_layer_per_seq = model_dims["num_query_heads"] * q_len * kv_len * model_dims["head_dim"] * 2
    softmax_flops_per_layer_per_seq = model_dims["num_query_heads"] * q_len * kv_len * 5
    total_attention_flops_per_seq = model_dims["num_decoder_layers"] * \
                                     (attn_qkt_flops_per_layer_per_seq + softmax_flops_per_layer_per_seq + attn_av_flops_per_layer_per_seq)
    total_flops_per_seq = learnable_weight_flops_per_seq + total_attention_flops_per_seq
    return total_flops_per_seq * batch_size

def calculate_memory_access_bytes(config, model_dims, num_model_params, batch_size, seq_len,
                                   is_prefill_step, activation_footprint_bytes_total_batch,
                                   kv_rw_bytes_this_step_total_batch):
    if batch_size == 0: return 0
    bytes_per_weight_param_eff = get_bytes_per_dtype(None, is_weight=True, config=config)
    weight_bytes_read = num_model_params * bytes_per_weight_param_eff 
    remat_policy = getattr(config, 'remat_policy', 'full')
    activation_traffic_factor = 1.0 if remat_policy == 'full' else 1.5
    activation_bytes_rw = activation_footprint_bytes_total_batch * activation_traffic_factor
    total_bytes = weight_bytes_read + activation_bytes_rw + kv_rw_bytes_this_step_total_batch
    return total_bytes

def get_roofline_estimates(config, model_dims, num_model_params, suggested_params, num_devices, hw_specs, cli_args):
    if not suggested_params or suggested_params.get("per_device_batch_size", 0) == 0:
        return None
    global_batch_size = int(suggested_params["per_device_batch_size"] * num_devices)
    if global_batch_size == 0: return None

    peak_tflops_device = hw_specs["device_peak_tflops_eff"]
    peak_bw_device_gb_s = hw_specs["device_hbm_bandwidth_gb_s"]
    kernel_efficiency = hw_specs["kernel_efficiency_factor"]
    estimates = {}
    bytes_per_element_act = get_bytes_per_dtype(config.dtype, config=config)

    prefill_len = config.max_prefill_predict_length
    if prefill_len <= 0: prefill_len = config.max_target_length
    if prefill_len > 0:
        act_footprint_prefill = estimate_total_activation_memory(global_batch_size, prefill_len, config, model_dims, verbose=False)
        prefill_flops = calculate_inference_flops(config, model_dims, num_model_params, global_batch_size, prefill_len, True)
        kv_bytes_written_this_step = estimate_kv_cache_memory_traditional(global_batch_size, prefill_len, config, model_dims)
        mem_access_bytes_prefill = calculate_memory_access_bytes(config, model_dims, num_model_params, global_batch_size,
                                                                 prefill_len, True, act_footprint_prefill, kv_bytes_written_this_step)
        oi_prefill = prefill_flops / mem_access_bytes_prefill if mem_access_bytes_prefill > 0 else 0
        perf_from_bw_prefill = oi_prefill * peak_bw_device_gb_s * 1e-3
        roof_perf_device_prefill = min(peak_tflops_device, perf_from_bw_prefill)
        roof_perf_total_prefill = roof_perf_device_prefill * num_devices * kernel_efficiency
        time_est_s_prefill = prefill_flops / (roof_perf_total_prefill * 1e12) if roof_perf_total_prefill > 0 else float('inf')
        throughput_tps_prefill = (global_batch_size * prefill_len) / time_est_s_prefill if time_est_s_prefill > 0 and time_est_s_prefill != float('inf') else 0
        estimates["prefill"] = {
            "stage_name": "Prefill", "seq_len": prefill_len, "total_flops_tf": prefill_flops / 1e12,
            "total_mem_access_gib": mem_access_bytes_prefill / (1024**3),
            "operational_intensity_f_b": oi_prefill, "roofline_perf_total_tf_s": roof_perf_total_prefill,
            "est_latency_ms": time_est_s_prefill * 1000, "est_throughput_tps": throughput_tps_prefill,
            "bound": "Compute" if peak_tflops_device <= perf_from_bw_prefill else "HBM Bandwidth"}
        if cli_args.roofline_device_ici_bw_gb_s > 0 and getattr(config, 'ici_tensor_parallelism', 1) > 1 and cli_args.global_activation_ici_transfer_fraction > 0:
            ici_bytes = act_footprint_prefill * cli_args.global_activation_ici_transfer_fraction 
            total_sys_ici_bw = cli_args.roofline_device_ici_bw_gb_s * 1e9 * num_devices
            if total_sys_ici_bw > 0:
                time_ici_s = ici_bytes / total_sys_ici_bw
                estimates["prefill"]["est_ici_transfer_time_ms"] = time_ici_s * 1000
                estimates["prefill"]["ici_bytes_transferred_gib"] = ici_bytes / (1024**3)

    decode_ctx_len = config.max_target_length
    if decode_ctx_len > 0:
        act_footprint_decode = estimate_total_activation_memory(global_batch_size, 1, config, model_dims, verbose=False)
        decode_flops = calculate_inference_flops(config, model_dims, num_model_params, global_batch_size, 1, False)
        kv_read_bytes_decode_step = estimate_kv_cache_memory_traditional(global_batch_size, decode_ctx_len, config, model_dims)
        kv_write_bytes_decode_step = estimate_kv_cache_memory_traditional(global_batch_size, 1, config, model_dims)
        kv_rw_bytes_decode_step = kv_read_bytes_decode_step + kv_write_bytes_decode_step
        mem_access_bytes_decode = calculate_memory_access_bytes(config, model_dims, num_model_params, global_batch_size,
                                                                1, False, act_footprint_decode, kv_rw_bytes_decode_step)
        oi_decode = decode_flops / mem_access_bytes_decode if mem_access_bytes_decode > 0 else 0
        perf_from_bw_decode = oi_decode * peak_bw_device_gb_s * 1e-3
        roof_perf_device_decode = min(peak_tflops_device, perf_from_bw_decode)
        roof_perf_total_decode = roof_perf_device_decode * num_devices * kernel_efficiency
        time_est_s_decode = decode_flops / (roof_perf_total_decode * 1e12) if roof_perf_total_decode > 0 else float('inf')
        throughput_tps_decode = global_batch_size / time_est_s_decode if time_est_s_decode > 0 and time_est_s_decode != float('inf') else 0
        estimates["decode"] = {
            "stage_name": "Decode", "context_len_for_kv": decode_ctx_len, "total_flops_tf_per_step": decode_flops / 1e12,
            "total_mem_access_gib_per_step": mem_access_bytes_decode / (1024**3),
            "operational_intensity_f_b": oi_decode, "roofline_perf_total_tf_s": roof_perf_total_decode,
            "est_latency_ms_per_token": time_est_s_decode * 1000, "est_throughput_tps": throughput_tps_decode,
            "bound": "Compute" if peak_tflops_device <= perf_from_bw_decode else "HBM Bandwidth"}
        if cli_args.roofline_device_ici_bw_gb_s > 0 and getattr(config, 'ici_tensor_parallelism', 1) > 1 and cli_args.global_activation_ici_transfer_fraction > 0:
             ici_bytes = act_footprint_decode * cli_args.global_activation_ici_transfer_fraction
             total_sys_ici_bw = cli_args.roofline_device_ici_bw_gb_s * 1e9 * num_devices
             if total_sys_ici_bw > 0:
                 time_ici_s = ici_bytes / total_sys_ici_bw
                 estimates["decode"]["est_ici_transfer_time_ms"] = time_ici_s * 1000
                 estimates["decode"]["ici_bytes_transferred_gib"] = ici_bytes / (1024**3)
    if not estimates: return None
    
    print_str = f"\n" + "---" * 15 + \
                f"\nTheoretical Roofline Performance Estimates (Using Scenario with TargetRatio=1.0 or Override)" + \
                f"\n(Peak Device: {hw_specs['device_peak_tflops_eff']:.0f} TFLOP/s, {peak_bw_device_gb_s:.0f} GB/s HBM | Kernel Eff: {kernel_efficiency:.2f}"
    if cli_args.roofline_device_ici_bw_gb_s > 0 and getattr(config, 'ici_tensor_parallelism', 1) > 1 and cli_args.global_activation_ici_transfer_fraction > 0:
        print_str += f" | ICI BW: {cli_args.roofline_device_ici_bw_gb_s:.0f} GB/s per dev, ICI Frac: {cli_args.global_activation_ici_transfer_fraction:.2f})"
    else: print_str += ")"
    print_str += "\n" + "---" * 15
    print(print_str)
    for stage_key, est in estimates.items():
        is_prefill = est["stage_name"] == "Prefill"
        latency_key = "est_latency_ms" if is_prefill else "est_latency_ms_per_token"
        flops_key = "total_flops_tf" if is_prefill else "total_flops_tf_per_step"
        mem_key = "total_mem_access_gib" if is_prefill else "total_mem_access_gib_per_step"
        seq_len_key_display = "seq_len" if is_prefill else "context_len_for_kv"
        latency_unit = " (full prefill)" if is_prefill else " (per token)"
        print(f"\n  {est['stage_name'].upper()} STAGE (@ Global Batch Size: {global_batch_size}):")
        print(f"     {'Prefill Length' if is_prefill else 'KV Cache Context Length'}: {est[seq_len_key_display]} tokens")
        print(f"     Total FLOPs: {est[flops_key]:.3f} TFLOPs")
        print(f"     Total HBM R/W: {est[mem_key]:.3f} GiB")
        print(f"     Operational Intensity (HBM): {est['operational_intensity_f_b']:.2f} FLOPs/Byte")
        print(f"     Est. Total Roofline Perf: {est['roofline_perf_total_tf_s']:.2f} TFLOP/s ({est['bound']}-Bound)")
        current_stage_latency_ms = est[latency_key]
        print(f"     Est. Latency: {current_stage_latency_ms:.3f} ms{latency_unit}")
        print(f"     Est. Throughput (total): {est['est_throughput_tps']:.1f} tokens/sec")
        if "est_ici_transfer_time_ms" in est:
            print(f"     Est. ICI Bytes Transferred (info): {est['ici_bytes_transferred_gib']:.3f} GiB")
            print(f"     Est. ICI Transfer Time (info): {est['est_ici_transfer_time_ms']:.3f} ms")
            if est['est_ici_transfer_time_ms'] > current_stage_latency_ms * 0.25 :
                 print(f"     WARNING: Estimated ICI transfer time ({est['est_ici_transfer_time_ms']:.3f} ms) is significant compared to " +
                       f"HBM/Compute-limited latency ({current_stage_latency_ms:.3f} ms) and might be an unmodeled bottleneck.")
    print("---" * 15);
    return estimates

def suggest_optimized_config_params_for_capacity_ratio(
     base_config, num_devices, hbm_per_device_bytes, num_model_params,
     safety_margin_factor, target_oom_safety_factor, max_bs_search_limit,
     effective_kv_capacity_ratio,
     base_page_manager_min_overcommit_factor,
     cli_pagedattn_num_pages_override,
     verbose=True
     ) -> Tuple[Dict, Dict]:

    config = base_config
    model_dims = get_effective_model_dims(config)
    if not all(model_dims[k] > 0 for k in model_dims):
         if verbose: print("Error: Invalid model dimensions.")
         return {}, model_dims

    total_hbm = num_devices * hbm_per_device_bytes
    model_weights_mem = estimate_model_weights_memory(config, num_model_params)
    static_safety_margin_bytes = total_hbm * safety_margin_factor
    available_for_dynamic_components = total_hbm - model_weights_mem - static_safety_margin_bytes
    if available_for_dynamic_components <= 0:
        if verbose: print(f"\nERROR: Not enough memory for weights & static margin.");
        return {}, model_dims
    target_dynamic_memory_usage = available_for_dynamic_components * target_oom_safety_factor
    if target_dynamic_memory_usage <=0:
         if verbose: print("\nERROR: Target dynamic memory usage is zero or negative.");
         return {}, model_dims

    seq_len_act_est = config.max_target_length
    if hasattr(config, 'max_prefill_predict_length') and config.max_prefill_predict_length > 0:
         seq_len_act_est = max(config.max_prefill_predict_length, config.max_target_length)
    seq_len_act_est = max(1, seq_len_act_est)
    
    # Refined Heuristic for initial high_bs_per_device
    est_act_single_global = estimate_total_activation_memory(1, seq_len_act_est, config, model_dims, verbose=False)
    est_kv_single_global = 0
    if config.attention == "paged":
        tokens_pg_heuristic = getattr(config, 'pagedattn_tokens_per_page', 32)
        eff_len_heuristic = max(1, config.max_target_length * effective_kv_capacity_ratio)
        pages_for_one_item_eff_len = math.ceil((1 * eff_len_heuristic) / tokens_pg_heuristic)
        P_target_for_one_item_heuristic = math.ceil(pages_for_one_item_eff_len * 1.05) # Match buffer logic
        
        # Cost of P_target_for_one_item_heuristic pages (dimension size) for K/V tensors
        est_kv_single_global = estimate_kv_cache_memory_paged(P_target_for_one_item_heuristic, config, model_dims)
    else:
        est_kv_single_global = estimate_kv_cache_memory_traditional(1, config.max_target_length, config, model_dims)

    total_mem_one_global_item = est_act_single_global + est_kv_single_global
    high_bs_calc = max_bs_search_limit * num_devices
    if total_mem_one_global_item > 0:
      high_bs_calc = math.floor(target_dynamic_memory_usage / total_mem_one_global_item)
    elif verbose: print("Warning: Estimated memory per batch item (for heuristic) is zero.")

    high_bs_per_device = math.floor(high_bs_calc / num_devices)
    high_bs_per_device = min(high_bs_per_device, max_bs_search_limit)
    high_bs_per_device = max(1, high_bs_per_device)
    
    search_target_desc = f"Target KV Capacity Ratio: {effective_kv_capacity_ratio*100:.1f}%"
    if cli_pagedattn_num_pages_override is not None:
        search_target_desc = f"CLI Override pagedattn_num_pages={cli_pagedattn_num_pages_override}"
    if verbose:
      print(f"\n--- Memory Budgeting ({search_target_desc}) ---")
      print(f"Target dynamic memory: {target_dynamic_memory_usage / (1024**3):.2f} GiB")
      print(f"Binary search for per_device_batch_size: low=1, high_initial_estimate={high_bs_per_device} (capped at {max_bs_search_limit})")

    low_bs = 1; current_best_bs = 0; current_best_paged_params = {}
    while low_bs <= high_bs_per_device:
        try_bs = low_bs + (high_bs_per_device - low_bs) // 2; try_bs = max(1, try_bs)
        if verbose: print(f"\n  Trying per_device_batch_size: {try_bs} (Global: {try_bs * num_devices})")
        current_global_bs = try_bs * num_devices
        if current_global_bs == 0: high_bs_per_device = try_bs -1; continue

        act_mem = estimate_total_activation_memory(current_global_bs, seq_len_act_est, config, model_dims, verbose=False)
        if verbose: print(f"     Est. activation memory: {act_mem / (1024**3):.3f} GiB")
        rem_kv_budget_total = target_dynamic_memory_usage - act_mem
        if rem_kv_budget_total < 0:
            if verbose: print(f"     NO FIT: Activations ({act_mem/(1024**3):.2f} GiB) > dynamic budget.")
            high_bs_per_device = try_bs - 1; continue

        fits_criteria = False; P_final_to_provision_for_this_bs = 0
        if config.attention == "paged":
            tokens_pg = getattr(config, 'pagedattn_tokens_per_page', 32); tokens_pg = max(1, tokens_pg)
            P_hbm = 0 # Max pages HBM can hold for the K/V tensors per layer (dimension size)
            # Cost if 'num_pages' dimension of K/V tensors (per layer) increases by 1
            mem_one_page_dim_one_layer_kv = (2 * tokens_pg * model_dims["num_kv_heads"] * model_dims["head_dim"]) * \
                                           get_bytes_per_dtype(config.dtype, is_kv_cache=True, config=config)
            kv_mem_cost_per_page_dim_in_all_layers = model_dims["num_decoder_layers"] * mem_one_page_dim_one_layer_kv
            if kv_mem_cost_per_page_dim_in_all_layers > 0:
                P_hbm = math.floor(rem_kv_budget_total / kv_mem_cost_per_page_dim_in_all_layers)
            P_hbm = max(0, P_hbm)

            if cli_pagedattn_num_pages_override is not None:
                P_final_to_provision_for_this_bs = cli_pagedattn_num_pages_override
                if P_final_to_provision_for_this_bs <= P_hbm: fits_criteria = True
                if verbose: print(f"     Using CLI override: {P_final_to_provision_for_this_bs} pages. HBM allows {P_hbm}. Fit: {fits_criteria}")
            else:
                _max_target_len_runtime = max(1, config.max_target_length)
                effective_max_len_stat = _max_target_len_runtime * effective_kv_capacity_ratio
                pages_needed_for_target_pool_effective_len = math.ceil((current_global_bs * effective_max_len_stat) / tokens_pg)
                P_target_statistical_pool_pages = math.ceil(pages_needed_for_target_pool_effective_len * 1.05)
                
                pm_floor_effective_max_len = _max_target_len_runtime * effective_kv_capacity_ratio
                pages_per_sequence_for_pm_floor = math.ceil(pm_floor_effective_max_len / tokens_pg)
                total_pages_for_pm_floor_at_eff_len = current_global_bs * pages_per_sequence_for_pm_floor
                min_pages_for_pm_floor_stat_effective = (total_pages_for_pm_floor_at_eff_len / base_page_manager_min_overcommit_factor) * 1.05
                P_min_page_manager_floor_effective = math.ceil(max(min_pages_for_pm_floor_stat_effective, current_global_bs * 1.0, 10.0))
                
                P_final_to_provision_for_this_bs = P_target_statistical_pool_pages
                if verbose:
                    print(f"     Targeting {effective_kv_capacity_ratio*100:.1f}% eff.cap (eff.len {effective_max_len_stat:.0f}) -> {P_final_to_provision_for_this_bs} pages desired for K/V tensors.")
                    print(f"     HBM supports K/V tensors up to: {P_hbm} pages dimension.")
                    print(f"     PageManager floor (eff.len {pm_floor_effective_max_len:.0f}, factor {base_page_manager_min_overcommit_factor:.2f}): {P_min_page_manager_floor_effective} pages.")
                if P_final_to_provision_for_this_bs <= P_hbm and P_final_to_provision_for_this_bs >= P_min_page_manager_floor_effective:
                    fits_criteria = True
                    if verbose: print(f"     ==> Paged Attn FIT: Provisioning {P_final_to_provision_for_this_bs} pages.")
                else:
                    if verbose:
                        details = []
                        if P_final_to_provision_for_this_bs > P_hbm: details.append(f"Target pages ({P_final_to_provision_for_this_bs}) > HBM capacity ({P_hbm})")
                        if P_final_to_provision_for_this_bs < P_min_page_manager_floor_effective: details.append(f"Target pages ({P_final_to_provision_for_this_bs}) < PM floor ({P_min_page_manager_floor_effective})")
                        print(f"     Paged Attn NO FIT: " + "; ".join(details))
            if fits_criteria:
                 temp_paged_params = {"pagedattn_tokens_per_page": tokens_pg, "pagedattn_num_pages": int(P_final_to_provision_for_this_bs)}
        else: # Standard KV Cache
            kv_mem = estimate_kv_cache_memory_traditional(current_global_bs, config.max_target_length, config, model_dims)
            if act_mem + kv_mem < target_dynamic_memory_usage: fits_criteria = True
            if verbose: print(f"     Std KV: Act_mem {act_mem/(1024**3):.2f} + KV_mem {kv_mem/(1024**3):.2f} = {(act_mem+kv_mem)/(1024**3):.2f} GiB. Target: {target_dynamic_memory_usage/(1024**3):.2f} GiB. Fit: {fits_criteria}")

        if fits_criteria:
            current_best_bs = try_bs
            if config.attention == "paged": current_best_paged_params = temp_paged_params.copy()
            low_bs = try_bs + 1
        else: high_bs_per_device = try_bs - 1
    
    suggested_params = {}
    if current_best_bs > 0:
        suggested_params["per_device_batch_size"] = float(current_best_bs)
        if config.attention == "paged":
            if not current_best_paged_params :
                if verbose: print("ERROR: Paged attention FIT for BS > 0 but paged params not captured.")
                suggested_params["pagedattn_num_pages"] = 0
            else: suggested_params.update(current_best_paged_params)
    elif verbose:
         err_desc = f"KV Capacity Ratio {effective_kv_capacity_ratio*100:.1f}%"
         if cli_pagedattn_num_pages_override is not None: err_desc = f"CLI override pagedattn_num_pages={cli_pagedattn_num_pages_override}"
         print(f"\nERROR: Could not find a suitable batch size for {err_desc}")
    return suggested_params, model_dims

def main():
    parser = argparse.ArgumentParser(description="MaxText Standalone Memory & Roofline Predictor")
    parser.add_argument("config_path", type=str, help="Path to base MaxText YAML config")
    parser.add_argument("config_overrides", nargs='*', help="MaxText config overrides (e.g., key=value)")
    parser.add_argument("--num_devices", type=int, required=True, help="Total devices")
    parser.add_argument("--device_hbm_gib", type=float, required=True, help="HBM per device (GiB)")
    parser.add_argument("--static_safety_margin_factor", type=float, default=0.10, help="Static overhead margin")
    parser.add_argument("--target_oom_safety_factor", type=float, default=0.90, help="Target dynamic memory usage")
    parser.add_argument("--num_model_params_billions", type=float, default=0, help="Model params (B). 0 to auto-calc")
    parser.add_argument("--max_bs_search_limit", type=int, default=256, help="Practical upper limit for per-device BS in binary search.")
    parser.add_argument("--verbose-search", action='store_true', help="Print the details of the binary search for each ratio.")
    parser.add_argument("--roofline_device_peak_tflops_bf16", type=float, default=0, help="Peak BF16 TFLOP/s per device")
    parser.add_argument("--roofline_device_peak_tflops_int8", type=float, default=0, help="Peak INT8 TFLOP/s per device")
    parser.add_argument("--roofline_device_hbm_bandwidth_gb_s", type=float, default=0, help="Peak HBM GB/s per device")
    parser.add_argument("--roofline_kernel_efficiency_factor", type=float, default=0.85, help="Kernel efficiency (0.0-1.0)")
    parser.add_argument("--roofline_device_ici_bw_gb_s", type=float, default=0, help="Effective BiDi ICI BW per device (GB/s) (for info only)")
    parser.add_argument("--global_activation_ici_transfer_fraction", type=float, default=0.0, help="Fraction of total activation volume assumed to cross ICI if tensor parallel (for info only, 0 to disable)")
    parser.add_argument("--kv_capacity_ratios", type=str, default="1.0,0.75,0.5", help="Comma-separated list of effective KV capacity ratios to test (e.g., 1.0,0.75,0.5)")
    parser.add_argument("--page_manager_floor_factor", type=float, default=1.25, help="Overcommit factor for PageManager's absolute minimum page requirement, applied to the *effective* sequence length.")
    args = parser.parse_args()

    cli_pagedattn_num_pages_override = None; temp_config_overrides = []
    for override in args.config_overrides:
        if override.startswith("pagedattn_num_pages="):
            try:
                cli_pagedattn_num_pages_override = int(override.split("=")[1])
                print(f"INFO: CLI override for pagedattn_num_pages found: {cli_pagedattn_num_pages_override}.")
                temp_config_overrides.append(override)
            except: print(f"Warning: Could not parse pagedattn_num_pages: {override}"); temp_config_overrides.append(override)
        else: temp_config_overrides.append(override)
    args.config_overrides = temp_config_overrides

    pyconfig_argv = [sys.argv[0], args.config_path] + args.config_overrides
    print(f"\nInitializing MaxText config:\n Base: {args.config_path}\n Overrides: {args.config_overrides}")
    try: config = pyconfig.initialize(pyconfig_argv)
    except Exception as e: sys.exit(f"Error initializing MaxText config: {e}")

    if args.device_hbm_gib <= 0: sys.exit("Error: --device_hbm_gib must be positive.")
    hbm_per_device_bytes = int(args.device_hbm_gib * (1024**3))
    try: model_dims_for_calc = get_effective_model_dims(config)
    except Exception as e: sys.exit(f"Error getting model dimensions: {e}")
    if not all(model_dims_for_calc[k] > 0 for k in model_dims_for_calc): sys.exit("Error: Zero/negative vital model dimensions.")

    num_params = int(args.num_model_params_billions * 1e9) if args.num_model_params_billions > 0 else 0
    if num_params == 0:
        try: num_params = calculate_num_params_from_config(config, model_dims_for_calc)
        except Exception as e: sys.exit(f"Error calculating num_model_params: {e}")
    if num_params <= 0: sys.exit("Error: Num model params is not positive.")
    print(f"\nCalculated/Used model params: {num_params / 1e9:.3f} B")
    
    try: effective_kv_capacity_ratios_to_test = [float(r.strip()) for r in args.kv_capacity_ratios.split(',') if r.strip()]
    except ValueError as e: sys.exit(f"Error parsing --kv_capacity_ratios: {e}.")
    if not effective_kv_capacity_ratios_to_test or not all(0.0 < r <= 2.0 for r in effective_kv_capacity_ratios_to_test):
         sys.exit("KV capacity ratios must be positive and realistically <= 2.0.")

    all_results_by_ratio: MutableMapping[float, Dict] = {}; final_model_dims = model_dims_for_calc
    is_paged = getattr(config, 'attention', 'dot_product') == 'paged'
    if not is_paged: effective_kv_capacity_ratios_to_test = [1.0]; print("\nINFO: attention != paged. Ratio sweep N/A.")
    elif cli_pagedattn_num_pages_override is not None: effective_kv_capacity_ratios_to_test = [-1.0]; print(f"\nINFO: pagedattn_num_pages overridden. Using fixed page count.")

    # Removed the "TESTING SCENARIOS" print line
    for ratio_key in effective_kv_capacity_ratios_to_test:
        current_ratio_for_logic = ratio_key if ratio_key != -1.0 else 1.0
        loop_target_desc = f"Effective KV Capacity Ratio: {current_ratio_for_logic*100:.1f}%"
        if cli_pagedattn_num_pages_override is not None and ratio_key == -1.0:
            loop_target_desc = f"CLI Override pagedattn_num_pages={cli_pagedattn_num_pages_override}"
        if not args.verbose_search: print(f". ({loop_target_desc})", end="")
        
        current_suggested_params, _ = suggest_optimized_config_params_for_capacity_ratio(
            config, args.num_devices, hbm_per_device_bytes, num_params,
            args.static_safety_margin_factor, args.target_oom_safety_factor,
            args.max_bs_search_limit, current_ratio_for_logic,
            args.page_manager_floor_factor, cli_pagedattn_num_pages_override,
            verbose=args.verbose_search)
        all_results_by_ratio[ratio_key] = current_suggested_params.copy()
    if not args.verbose_search : print(" Done.\n") # Newline after dots if not verbose

    print(f"\n{'='*30}\n      SUMMARY OF RESULTS\n{'='*30}")
    header_parts = [f"{'Scenario':<18}", f"{'Per_Dev_BS':<12}", f"{'Global_BS':<10}"]
    separator_parts = [f"{'-'*17}", f"{'-'*12}", f"{'-'*10}"]
    if is_paged:
       header_parts.extend([f"{'PagedNumPages':<13}", f"{'ActualCoverage(%)':<18}"])
       separator_parts.extend([f"{'-'*13}", f"{'-'*18}"])
    print(" | ".join(header_parts)); print("-+-".join(separator_parts))
     
    tokens_per_page_conf = getattr(config, 'pagedattn_tokens_per_page', 32)
    max_target_len_conf = config.max_target_length

    for ratio_key_to_print in effective_kv_capacity_ratios_to_test:
         results = all_results_by_ratio.get(ratio_key_to_print, {})
         bs_val = results.get("per_device_batch_size", 0.0)
         scenario_label = f"Ratio {ratio_key_to_print*100:.1f}%"
         if ratio_key_to_print == -1.0 and cli_pagedattn_num_pages_override is not None: scenario_label = f"Override"
         elif not is_paged and ratio_key_to_print == 1.0: scenario_label = "Non-Paged"
         row_parts = [f"{scenario_label:<18}", f"{bs_val:<12.1f}"]
         gbs_val = int(bs_val * args.num_devices) if bs_val > 0 else 0
         row_parts.append(f"{gbs_val:<10}")
         if is_paged:
             pages_val = results.get("pagedattn_num_pages", "N/A")
             row_parts.append(f"{str(pages_val):<13}")
             actual_coverage_str = "N/A"
             if isinstance(pages_val, int) and pages_val > 0 and gbs_val > 0 and max_target_len_conf > 0 and tokens_per_page_conf > 0:
                 total_token_capacity_in_pages = pages_val * tokens_per_page_conf
                 theoretical_max_tokens_needed_at_Smax = gbs_val * max_target_len_conf
                 if theoretical_max_tokens_needed_at_Smax > 0:
                     actual_coverage_percentage = (total_token_capacity_in_pages / theoretical_max_tokens_needed_at_Smax) * 100
                     actual_coverage_str = f"{actual_coverage_percentage:.1f}"
                 else: actual_coverage_str = "Inf" 
             elif isinstance(pages_val, int) and (pages_val == 0 or gbs_val == 0) : actual_coverage_str = "0.0"
             elif not results : actual_coverage_str = "N/A"
             row_parts.append(f"{actual_coverage_str:<18}")
         elif not is_paged: row_parts.extend([f"{'N/A':<13}", f"{'N/A':<18}"])
         print(" | ".join(row_parts))
    print(f"{'='*(len(' | '.join(header_parts)))}\n")
    
    roofline_params_to_use = all_results_by_ratio.get(1.0, {}) 
    if not roofline_params_to_use and -1.0 in all_results_by_ratio:
        roofline_params_to_use = all_results_by_ratio.get(-1.0, {})
        if roofline_params_to_use and roofline_params_to_use.get("per_device_batch_size", 0.0) > 0: print(f"INFO: Using results from CLI Override for Roofline.")
    elif not roofline_params_to_use and effective_kv_capacity_ratios_to_test:
        first_valid_ratio = next((r for r in effective_kv_capacity_ratios_to_test if r != -1.0 and all_results_by_ratio.get(r, {}).get("per_device_batch_size", 0.0) > 0), None)
        if first_valid_ratio is not None:
            roofline_params_to_use = all_results_by_ratio.get(first_valid_ratio,{})
            print(f"INFO: Using results from TargetRatio {first_valid_ratio*100:.1f}% for Roofline as 1.0 was not available/successful.")
            
    run_roofline = (args.roofline_device_peak_tflops_bf16 > 0 or args.roofline_device_peak_tflops_int8 > 0) and args.roofline_device_hbm_bandwidth_gb_s > 0
    if run_roofline:
      if roofline_params_to_use and roofline_params_to_use.get("per_device_batch_size", 0) > 0:
           eff_tflops = args.roofline_device_peak_tflops_bf16
           is_int8_compute = getattr(config, 'quantization', None) == 'int8' or \
                             (getattr(config, 'quantize_kvcache', False) and getattr(config, 'kv_quant_dtype', 'int8') == 'int8')
           if is_int8_compute and args.roofline_device_peak_tflops_int8 > 0: eff_tflops = args.roofline_device_peak_tflops_int8; print(f"INFO: Using INT8 peak TFLOP/s ({eff_tflops}) for roofline.")
           elif args.roofline_device_peak_tflops_bf16 > 0 : print(f"INFO: Using BF16 peak TFLOP/s ({eff_tflops}) for roofline.")
           else: eff_tflops = 0; print("Warning: Roofline TFLOP/s not determined, defaulting to 0.")
           hw_specs = {"device_peak_tflops_eff": eff_tflops, "device_hbm_bandwidth_gb_s": args.roofline_device_hbm_bandwidth_gb_s, "kernel_efficiency_factor": args.roofline_kernel_efficiency_factor}
           get_roofline_estimates(config, final_model_dims, num_params, roofline_params_to_use, args.num_devices, hw_specs, args)
      else: print("\nSkipping roofline: Could not find a valid configuration from tested scenarios for roofline.")
    else: print("\nSkipping roofline: peak TFLOP/s or HBM bandwidth not provided/zero via CLI args.")

if __name__ == "__main__":
     main()