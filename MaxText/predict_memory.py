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
based on estimated memory usage.
"""

import argparse
import math
import sys
from MaxText import pyconfig
from MaxText.configs import models

def get_bytes_per_dtype(dtype_str):
    """Returns the number of bytes for a given dtype string."""
    if dtype_str == "bfloat16":
        return 2
    elif dtype_str == "float16":
        return 2
    elif dtype_str == "float32":
        return 4
    elif dtype_str == "int8":
        return 1
    elif dtype_str == "int4":
        return 0.5
    else:
        raise ValueError(f"dtype ({dtype_str}) is not currently known.")

def calculate_num_params_from_config(config, model_dims):
    """
    Estimates the number of parameters for a Llama-like model from config.
    This is a simplified calculation.
    """
    vocab_size = config.vocab_size
    emb_dim = model_dims["emb_dim"]
    mlp_dim = model_dims["mlp_dim"]
    num_decoder_layers = model_dims["num_decoder_layers"]
    num_query_heads = model_dims["num_query_heads"]
    num_kv_heads = model_dims["num_kv_heads"]
    head_dim = model_dims["head_dim"]

    if emb_dim <= 0 or num_decoder_layers <= 0 :
        raise ValueError(f"emb_dim ({emb_dim}) and num_decoder_layers ({num_decoder_layers}) must be positive for param calculation.")

    params = 0

    # Embedding layer (shared with output logits if logits_via_embedding=True)
    params += vocab_size * emb_dim

    # Output Logits layer (if not shared)
    if not config.logits_via_embedding:
        params += emb_dim * vocab_size

    # Per Decoder Layer
    # Attention projections (Q, K, V, O)
    # Q: emb_dim * num_query_heads * head_dim
    # K: emb_dim * num_kv_heads * head_dim
    # V: emb_dim * num_kv_heads * head_dim
    # O: num_query_heads * head_dim * emb_dim
    layer_attn_params = (emb_dim * num_query_heads * head_dim) + \
                        (emb_dim * num_kv_heads * head_dim) * 2 + \
                        (num_query_heads * head_dim * emb_dim)
    
    # MLP parameters (Llama-style: Gate, Up, Down)
    # W_gate: emb_dim * mlp_dim
    # W_up:   emb_dim * mlp_dim
    # W_down: mlp_dim * emb_dim
    if isinstance(config.mlp_activations, list) and len(config.mlp_activations) > 1: # Gated MLP like SwiGLU
        layer_mlp_params = (emb_dim * mlp_dim) * 2 + (mlp_dim * emb_dim)
    else: # Simple MLP
        layer_mlp_params = (emb_dim * mlp_dim) + (mlp_dim * emb_dim)

    # Normalization layers (RMSNorm: 1 parameter per channel, i.e., emb_dim)
    # Typically 2 per decoder layer (input_layernorm, post_attention_layernorm)
    layer_norm_params = 2 * emb_dim
    
    params_per_layer = layer_attn_params + layer_mlp_params + layer_norm_params
    params += num_decoder_layers * params_per_layer

    # Final normalization layer (after all decoder layers)
    params += emb_dim # norm_f

    return int(params)


def estimate_model_weights_memory(config, num_total_params):
    """Estimates memory for model weights."""
    bytes_per_param = get_bytes_per_dtype(config.weight_dtype)
    model_weights_bytes = num_total_params * bytes_per_param
    return model_weights_bytes

def get_effective_model_dims(config):
    """Gets model dimensions, applying global_parameter_scale if present."""
    # These base dimensions are usually overridden by model_name configs
    # but we use them as a fallback if a model_name isn't perfectly matched
    # or for generic calculations.
    scale = getattr(config, 'global_parameter_scale', 1)
    
    base_emb_dim = getattr(config, 'base_emb_dim', 0)
    base_num_query_heads = getattr(config, 'base_num_query_heads', 0)
    base_num_kv_heads = getattr(config, 'base_num_kv_heads', 0)
    base_mlp_dim = getattr(config, 'base_mlp_dim', 0)
    base_num_decoder_layers = getattr(config, 'base_num_decoder_layers', 0)
    
    # head_dim is often explicitly set and not scaled by global_parameter_scale
    head_dim = getattr(config, 'head_dim', 0)

    emb_dim = getattr(config, 'emb_dim', base_emb_dim * scale) # Prefer direct if set (e.g. by model config)
    num_query_heads = getattr(config, 'num_query_heads', base_num_query_heads * scale)
    num_kv_heads = getattr(config, 'num_kv_heads', base_num_kv_heads * scale)
    mlp_dim = getattr(config, 'mlp_dim', base_mlp_dim * scale)
    num_decoder_layers = getattr(config, 'num_decoder_layers', base_num_decoder_layers) # Layer count usually not scaled

    if head_dim <= 0 and num_query_heads > 0 and emb_dim > 0:
        head_dim = emb_dim // num_query_heads
    elif head_dim <= 0 : # A default if completely unspecified
        print("Warning: head_dim is undefined, defaulting to 128. This might be incorrect.")
        head_dim = 128

    dims = {
        "emb_dim": int(emb_dim),
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads),
        "mlp_dim": int(mlp_dim),
        "num_decoder_layers": int(num_decoder_layers),
        "head_dim": int(head_dim),
    }
    print("\nEffective Model Dimensions for Calculation:")
    for k, v in dims.items():
        print(f"  {k}: {v}")
    if any(v <= 0 for v in dims.values()):
        print("Warning: Some model dimensions are zero or negative. Estimates might be inaccurate.")
        print("Ensure your model_name is correctly processed or base dimensions are set.")
    return dims


def estimate_total_activation_memory(global_batch_size, seq_len, config, model_dims):
    """Estimates activation memory, considering rematerialization policy."""
    if global_batch_size == 0 or seq_len == 0: return 0

    bytes_per_element = get_bytes_per_dtype(config.dtype)
    num_layers = model_dims["num_decoder_layers"]
    if num_layers == 0: return 0

    layer_input_mem = global_batch_size * seq_len * model_dims["emb_dim"] * bytes_per_element
    attn_output_mem = global_batch_size * seq_len * model_dims["num_query_heads"] * model_dims["head_dim"] * bytes_per_element
    mlp_hidden_mem = global_batch_size * seq_len * model_dims["mlp_dim"] * bytes_per_element
    
    live_per_layer_approx = 0
    remat_policy = getattr(config, 'remat_policy', 'full') # Default to full if not set

    if remat_policy == 'full':
        live_per_layer_approx = layer_input_mem + max(attn_output_mem, mlp_hidden_mem) # Input + largest intermediate
    elif 'offloaded' in remat_policy:
        live_per_layer_approx = layer_input_mem * 0.15 # Device footprint for offloaded activations
    elif remat_policy == 'minimal':
        live_per_layer_approx = layer_input_mem + attn_output_mem + mlp_hidden_mem # Simplified sum
    elif 'save_dot_except_mlp' in remat_policy or 'save_qkv_proj' in remat_policy :
        live_per_layer_approx = layer_input_mem + attn_output_mem # Assume MLP part is rematted
    else: # Other custom/partial remat policies
        live_per_layer_approx = layer_input_mem + max(attn_output_mem, mlp_hidden_mem) * 0.7 # Intermediate guess

    # scan_layers reduces memory proportional to layers, effectively making it a small constant factor of layers.
    # If not scanning layers, memory can be proportional to num_layers * live_per_layer_approx.
    effective_layers_for_memory = 2.0 if config.scan_layers else num_layers
    
    # Adjust if not scanning layers but remat is aggressive
    if not config.scan_layers and remat_policy == 'full':
        effective_layers_for_memory = num_layers * 0.3 # Heuristic: full remat reduces layer stacking effect
    elif not config.scan_layers and 'offloaded' in remat_policy:
         effective_layers_for_memory = num_layers * 0.05


    total_activation_mem = live_per_layer_approx * effective_layers_for_memory
    total_activation_mem *= 1.15 # Small buffer for miscellaneous activations / JAX overheads
    return total_activation_mem

def estimate_kv_cache_memory(global_batch_size, max_len, config, model_dims):
    """Estimates memory for the standard (non-paged) KV cache."""
    if global_batch_size == 0: return 0
    bytes_per_element = get_bytes_per_dtype(config.dtype) # KV cache usually same dtype as activations
    
    single_cache_layer_elements = 2 * global_batch_size * max_len * \
                                  model_dims["num_kv_heads"] * model_dims["head_dim"]
    
    total_kv_cache_bytes = model_dims["num_decoder_layers"] * \
                             single_cache_layer_elements * bytes_per_element
    return total_kv_cache_bytes

def estimate_paged_kv_cache_memory(config, model_dims):
    """Estimates memory for the paged KV cache based on its total configured size."""
    bytes_per_element = get_bytes_per_dtype(config.dtype)

    pagedattn_num_pages = getattr(config, 'pagedattn_num_pages', 0)
    pagedattn_tokens_per_page = getattr(config, 'pagedattn_tokens_per_page', 0)

    if pagedattn_num_pages <= 0 or pagedattn_tokens_per_page <= 0:
        # This means paged KV cache is not meaningfully configured for memory calculation
        return 0

    total_elements_in_paged_kv = pagedattn_num_pages * \
                                 pagedattn_tokens_per_page * \
                                 model_dims["num_kv_heads"] * \
                                 model_dims["head_dim"]
                                 
    paged_kv_cache_bytes = total_elements_in_paged_kv * bytes_per_element
    return paged_kv_cache_bytes

def suggest_optimized_config_params(
    base_config,
    num_devices,
    hbm_per_device_bytes,
    num_model_params,
    safety_margin_factor, # For static overheads
    target_oom_safety_factor # How much of the remaining dynamic memory to target
):
    """
    Suggests per_device_batch_size and paged attention parameters.
    Returns a dictionary of suggested config values.
    """
    print("\n" + "="*40)
    print("MaxText Memory-Aware Configuration Predictor")
    print("="*40)

    config = base_config # Use the fully resolved config
    model_dims = get_effective_model_dims(config)

    model_weights_mem = estimate_model_weights_memory(config, num_model_params)
    print(f"\n--- System & Model Basics ---")
    print(f"HBM per device: {hbm_per_device_bytes / (1024**3):.2f} GiB")
    print(f"Number of devices: {num_devices}")
    total_hbm = num_devices * hbm_per_device_bytes
    print(f"Total HBM across all devices: {total_hbm / (1024**3):.2f} GiB")
    print(f"Model Datatype (weights): {config.weight_dtype}")
    print(f"Model Datatype (activations/KV): {config.dtype}")
    print(f"Total model parameters: {num_model_params / 1e9:.2f} Billion")
    print(f"Estimated model weights memory (total): {model_weights_mem / (1024**3):.2f} GiB")
    if num_devices > 0 :
        print(f"Estimated model weights memory (per device, if perfectly sharded): {(model_weights_mem / num_devices) / (1024**3):.2f} GiB/device")


    static_safety_margin_bytes = total_hbm * safety_margin_factor
    available_for_dynamic_components = total_hbm - model_weights_mem - static_safety_margin_bytes

    if available_for_dynamic_components <= 0:
        print("\nERROR: Not enough memory even for model weights and static safety margin.")
        print(f"  Total HBM: {total_hbm / (1024**3):.2f} GiB")
        print(f"  Model Weights: {model_weights_mem / (1024**3):.2f} GiB")
        print(f"  Static Safety Margin ({safety_margin_factor*100}%): {static_safety_margin_bytes / (1024**3):.2f} GiB")
        return {}

    target_dynamic_memory_usage = available_for_dynamic_components * target_oom_safety_factor
    
    print(f"\n--- Memory Budgeting ---")
    print(f"Memory after weights & static margin (for Activations, KV-Cache): {available_for_dynamic_components / (1024**3):.2f} GiB")
    print(f"Targeting ~{target_oom_safety_factor*100:.0f}% of this for dynamic data: {target_dynamic_memory_usage / (1024**3):.2f} GiB")
    print(f"  (Remaining {(1-target_oom_safety_factor)*100:.0f}% or {available_for_dynamic_components*(1-target_oom_safety_factor) / (1024**3):.2f} GiB for XLA overheads, fragmentation, etc.)")

    suggested_params = {}
    
    # Determine sequence length for activation estimation (worst case of prefill or target)
    # max_prefill_predict_length, max_target_length
    seq_len_for_activation_est = config.max_target_length
    if hasattr(config, 'max_prefill_predict_length') and config.max_prefill_predict_length > 0:
        seq_len_for_activation_est = max(config.max_prefill_predict_length, config.max_target_length)
    print(f"Using sequence length for activation estimation: {seq_len_for_activation_est} (max of prefill/target lengths)")
    print(f"Using sequence length for KV cache capacity: {config.max_target_length}")
    print(f"Using remat_policy: {getattr(config, 'remat_policy', 'full')}, scan_layers: {config.scan_layers}")


    if config.attention == "paged":
        print("\n--- Optimizing for Paged Attention ---")
        
        pagedattn_tokens_per_page = getattr(config, 'pagedattn_tokens_per_page', 0)
        if not isinstance(pagedattn_tokens_per_page, int) or pagedattn_tokens_per_page <= 0:
            print(f"  Config 'pagedattn_tokens_per_page' ({pagedattn_tokens_per_page}) is invalid or not set, using default: 64.")
            pagedattn_tokens_per_page = 64
        suggested_params["pagedattn_tokens_per_page"] = pagedattn_tokens_per_page

        pagedattn_max_pages_per_group_cfg = getattr(config, 'pagedattn_max_pages_per_group', -1)
        calc_max_pages_per_group = math.ceil(config.max_target_length / pagedattn_tokens_per_page)
        if not isinstance(pagedattn_max_pages_per_group_cfg, int) or pagedattn_max_pages_per_group_cfg <= 0:
            pagedattn_max_pages_per_group = calc_max_pages_per_group
        else:
            pagedattn_max_pages_per_group = max(pagedattn_max_pages_per_group_cfg, calc_max_pages_per_group)
        suggested_params["pagedattn_max_pages_per_group"] = pagedattn_max_pages_per_group
        
        print(f"  Using pagedattn_tokens_per_page: {pagedattn_tokens_per_page}")
        print(f"  Calculated pagedattn_max_pages_per_group: {pagedattn_max_pages_per_group} (for max_target_length {config.max_target_length})")

        potential_batch_sizes = sorted(list(set([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64] + [int(config.per_device_batch_size)])), reverse=True)
        found_batch_size = 0

        for bs_per_device in potential_batch_sizes:
            if bs_per_device <=0: continue
            current_global_batch = bs_per_device * num_devices
            print(f"\n  Trying per_device_batch_size: {bs_per_device} (Global: {current_global_batch})")

            activation_mem_est = estimate_total_activation_memory(
                current_global_batch, seq_len_for_activation_est, config, model_dims
            )
            print(f"    Estimated activation memory: {activation_mem_est / (1024**3):.2f} GiB")

            if activation_mem_est >= target_dynamic_memory_usage:
                print(f"    Activation memory ({activation_mem_est / (1024**3):.2f} GiB) alone meets/exceeds target dynamic memory ({target_dynamic_memory_usage/(1024**3):.2f} GiB).")
                continue

            remaining_for_paged_kv = target_dynamic_memory_usage - activation_mem_est
            print(f"    Memory remaining for Paged KV Cache: {remaining_for_paged_kv / (1024**3):.2f} GiB")

            bytes_per_element_kv = get_bytes_per_dtype(config.dtype)
            memory_per_physical_page = model_dims["num_kv_heads"] * \
                                       pagedattn_tokens_per_page * \
                                       model_dims["head_dim"] * \
                                       bytes_per_element_kv

            if memory_per_physical_page <= 0:
                print("    Error: Memory per physical page is zero or negative. Check model dimensions or pagedattn_tokens_per_page.")
                continue

            num_pages_can_allocate = int(remaining_for_paged_kv / memory_per_physical_page) if remaining_for_paged_kv >0 else 0
            
            if num_pages_can_allocate <= 0 :
                 print(f"    Cannot allocate any pages with remaining memory {remaining_for_paged_kv / (1024**3):.2f} GiB.")
                 continue
            print(f"    Can allocate ~{num_pages_can_allocate} pages for KV cache.")

            # Min pages needed: at least enough for the batch to have their first page, and ideally more.
            # A very loose check: enough pages for each request to have *some* context to avoid immediate full stalls.
            # A more practical limit is whether num_pages_can_allocate is substantial.
            min_practical_pages = current_global_batch * pagedattn_max_pages_per_group * 0.1 # e.g. 10% of theoretical max if all full
            
            if num_pages_can_allocate > min_practical_pages and num_pages_can_allocate > current_global_batch:
                cfg_pagedattn_num_pages = getattr(config, 'pagedattn_num_pages', -1)
                if cfg_pagedattn_num_pages > 0: # If user provided a value for num_pages in input config
                     final_num_pages = min(num_pages_can_allocate, cfg_pagedattn_num_pages)
                     if final_num_pages < num_pages_can_allocate:
                         print(f"    User-defined config.pagedattn_num_pages ({cfg_pagedattn_num_pages}) is a tighter constraint. Using {final_num_pages}.")
                else: # User did not specify, so we use our calculated value
                     final_num_pages = num_pages_can_allocate
                
                # Check if we have enough pages for all concurrent requests if they were to fill up
                # This is a strong check, system might work with fewer due to request staggering
                if final_num_pages < current_global_batch * pagedattn_max_pages_per_group:
                    print(f"    Warning: Suggested pagedattn_num_pages ({final_num_pages}) is less than theoretical max needed "
                          f"({current_global_batch * pagedattn_max_pages_per_group}) if all {current_global_batch} "
                          f"requests fill to max_target_length. System relies on page reuse.")
                
                if final_num_pages > 0:
                    suggested_params["per_device_batch_size"] = float(bs_per_device)
                    suggested_params["pagedattn_num_pages"] = final_num_pages
                    found_batch_size = bs_per_device
                    print(f"    ==> Found configuration: per_device_batch_size={bs_per_device}, pagedattn_num_pages={final_num_pages}")
                    break 
            else:
                print(f"    Calculated pages ({num_pages_can_allocate}) not sufficient or practical for this batch size.")
        
        if not found_batch_size:
            print("\nERROR: Could not find a suitable paged attention configuration for any batch size tried.")
            print("  Consider increasing HBM, reducing safety_margin_factor, model size, or max_target_length.")
            return {}

    else: # Standard KV Cache
        print("\n--- Optimizing for Standard KV Attention ---")
        potential_batch_sizes = sorted(list(set([1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64] + [int(config.per_device_batch_size)])), reverse=True)
        found_batch_size = 0

        for bs_per_device in potential_batch_sizes:
            if bs_per_device <=0: continue
            current_global_batch = bs_per_device * num_devices
            print(f"\n  Trying per_device_batch_size: {bs_per_device} (Global: {current_global_batch})")

            activation_mem_est = estimate_total_activation_memory(
                current_global_batch, seq_len_for_activation_est, config, model_dims
            )
            kv_cache_mem_est = estimate_kv_cache_memory(
                current_global_batch, config.max_target_length, config, model_dims
            )
            total_dynamic_needed = activation_mem_est + kv_cache_mem_est

            print(f"    Estimated activation memory: {activation_mem_est / (1024**3):.2f} GiB")
            print(f"    Estimated standard KV Cache memory: {kv_cache_mem_est / (1024**3):.2f} GiB")
            print(f"    Total dynamic memory needed: {total_dynamic_needed / (1024**3):.2f} GiB")

            if total_dynamic_needed < target_dynamic_memory_usage:
                suggested_params["per_device_batch_size"] = float(bs_per_device)
                found_batch_size = bs_per_device
                print(f"    ==> Found configuration: per_device_batch_size={bs_per_device}")
                break
            else:
                print(f"    Total dynamic memory needed exceeds target ({target_dynamic_memory_usage/(1024**3):.2f} GiB).")

        if not found_batch_size:
            print("\nERROR: Could not find a suitable standard KV cache configuration for any batch size tried.")
            print("  Consider increasing HBM, reducing safety_margin_factor, model size, or max_target_length.")
            return {}
            
    print("\n" + "="*40)
    print("End of MaxText Memory-Aware Configuration Predictor")
    print("="*40 + "\n")
    return suggested_params

def main():
    parser = argparse.ArgumentParser(description="MaxText Standalone Memory Configuration Predictor")
    parser.add_argument("config_path", type=str, help="Path to the base MaxText YAML config file (e.g., MaxText/configs/base.yml)")
    parser.add_argument("config_overrides", nargs='*', help="MaxText config overrides (e.g., model_name=llama2-70b per_device_batch_size=1 ...)")
    
    # Custom arguments for this script
    parser.add_argument("--num_devices", type=int, required=True, help="Total number of devices (e.g., TPU cores)")
    parser.add_argument("--device_hbm_gib", type=float, required=True, help="HBM per device in GiB (e.g., 16 for TPU v6e, 32 for TPU v4)")
    parser.add_argument("--static_safety_margin_factor", type=float, default=0.10, help="Safety margin for static overheads (OS, runtime, etc.) (0.0 to 1.0)")
    parser.add_argument("--target_oom_safety_factor", type=float, default=0.90, help="Target using this fraction of available dynamic memory (0.0 to 1.0 to prevent OOM)")
    parser.add_argument("--num_model_params_billions", type=float, default=0, help="Number of model parameters in billions. If 0, script will try to calculate from config (approximate).")


    args = parser.parse_args()

    # Prepare argv for pyconfig
    pyconfig_argv = [sys.argv[0]] + [args.config_path] + args.config_overrides
    
    print("Initializing MaxText config with:")
    print(f"  Base config: {args.config_path}")
    print(f"  Overrides: {args.config_overrides}")
    
    # This will print all loaded config values
    config = pyconfig.initialize(pyconfig_argv)

    if args.device_hbm_gib <= 0:
        print("Error: --device_hbm_gib must be positive.")
        sys.exit(1)
    hbm_per_device_bytes = int(args.device_hbm_gib * (1024**3))

    model_dims = get_effective_model_dims(config) # Call this early to show what config results in

    num_params = 0
    if args.num_model_params_billions > 0:
        num_params = int(args.num_model_params_billions * 1e9)
        print(f"Using provided number of model parameters: {num_params / 1e9:.2f} Billion")
    else:
        print("Attempting to calculate number of model parameters from configuration (approximate)...")
        try:
            num_params = calculate_num_params_from_config(config, model_dims)
            print(f"Calculated model parameters from config: {num_params / 1e9:.3f} Billion")
            if num_params == 0 :
                 print("Warning: Calculated number of parameters is 0. Ensure model dimensions in config are correct.")
        except Exception as e:
            print(f"Error calculating num_model_params from config: {e}")
            print("Please provide --num_model_params_billions or ensure config defines model structure.")
            sys.exit(1)
    
    if num_params <= 0:
        print("Error: Number of model parameters is not positive. Cannot proceed.")
        sys.exit(1)


    suggested = suggest_optimized_config_params(
        config,
        args.num_devices,
        hbm_per_device_bytes,
        num_params,
        args.static_safety_margin_factor,
        args.target_oom_safety_factor
    )

    if suggested:
        print("\n--- Suggested Configuration Values ---")
        for key, value in suggested.items():
            # These are the parameters this script primarily tries to optimize
            if key in ["per_device_batch_size", "pagedattn_num_pages", "pagedattn_tokens_per_page", "pagedattn_max_pages_per_group"]:
                 print(f"  ** {key}: {value} ** (Suggested Optimal)")
            else:
                 print(f"  {key}: {value}") # Should mainly be paged attention params derived from others
        
        print("\nTo use these suggestions, pass them as command-line overrides when running MaxText:")
        suggestion_overrides = []
        for key, value in suggested.items():
            suggestion_overrides.append(f"{key}={value}")
        print(f"Example: python3 MaxText/...py ... {' '.join(suggestion_overrides)}")

    else:
        print("\nNo suitable configuration could be suggested with the given constraints.")

if __name__ == "__main__":
    main()