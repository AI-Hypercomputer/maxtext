# from transformers import AutoModelForCausalLM
# import torch

# hf_model_path= "/home/shuningjin/deepseek3-671b/hf-671b-bf16"
# hf_model1 = AutoModelForCausalLM.from_pretrained(hf_model_path, dtype=torch.bfloat16)

# hf_model_path= "/home/shuningjin/deepseek3-671b/deepseek3-671b-hf-2025-10-31-16-41-10"
# hf_model2 = AutoModelForCausalLM.from_pretrained(hf_model_path, dtype=torch.bfloat16)



import os
# jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"


import torch
from transformers import AutoModelForCausalLM
import gc
from tqdm import tqdm


# Define model paths
hf_model_path1 = "/home/shuningjin/deepseek3-671b/hf-671b-bf16"
hf_model_path2 = "/home/shuningjin/deepseek3-671b/deepseek3-671b-hf-2025-10-31-16-41-10"



# --- Load Model 1 State Dict ---
print(f"Loading model 1 from: {hf_model_path1}")
try:
    # low_cpu_mem_usage=True is crucial for large models
    model1 = AutoModelForCausalLM.from_pretrained(
        hf_model_path1, 
        dtype=torch.bfloat16, 
       # low_cpu_mem_usage=True
    )
    # Get state_dict on CPU
    sd1 = {k: v.cpu() for k, v in model1.state_dict().items()}
    print("Model 1 state_dict loaded.")
except Exception as e:
    print(f"Error loading model 1: {e}")
    exit()
finally:
    # Delete model to free memory
    if 'model1' in locals():
        del model1
    gc.collect()
    print("Model 1 cleared from memory.")


# --- Load Model 2 State Dict ---
print(f"\nLoading model 2 from: {hf_model_path2}")
try:
    model2 = AutoModelForCausalLM.from_pretrained(
        hf_model_path2, 
        dtype=torch.bfloat16, 
      #  low_cpu_mem_usage=True
    )
    # Get state_dict on CPU
    sd2 = {k: v.cpu() for k, v in model2.state_dict().items()}
    print("Model 2 state_dict loaded.")
except Exception as e:
    print(f"Error loading model 2: {e}")
    exit()
finally:
    # Delete model to free memory
    if 'model2' in locals():
        del model2
    gc.collect()
    print("Model 2 cleared from memory.")


# --- Compare State Dicts ---
print("\n--- Comparing Model Weights ---")
keys1 = set(sd1.keys())
keys2 = set(sd2.keys())

# Check for keys present in one model but not the other
missing_in_2 = keys1 - keys2
if missing_in_2:
    print(f"WARNING: Keys in model 1 but not in model 2 ({len(missing_in_2)}):")
    for k in list(missing_in_2)[:5]: # Print first 5
        print(f"  {k}")

missing_in_1 = keys2 - keys1
if missing_in_1:
    print(f"WARNING: Keys in model 2 but not in model 1 ({len(missing_in_1)}):")
    for k in list(missing_in_1)[:5]: # Print first 5
        print(f"  {k}")

# Compare common parameters
common_keys = sorted(list(keys1 & keys2))
diff_summary = {}
total_abs_diff = 0.0
global_max_diff = 0.0

print(f"\nComparing {len(common_keys)} common parameters...")

with torch.no_grad():
    for key in tqdm(common_keys, total=len(common_keys)):
        p1 = sd1[key]
        p2 = sd2[key]

        # Check for shape mismatches
        if p1.shape != p2.shape:
            print(f"ERROR: Shape mismatch for key {key}: {p1.shape} (model 1) vs {p2.shape} (model 2)")
            continue
        
        # Calculate difference. 
        # Convert to float32 for more stable difference calculation
        diff_tensor = torch.abs(p1 - p2)
        
        mean_abs_diff = diff_tensor.mean().item()
        max_abs_diff = diff_tensor.max().item()
        
        total_abs_diff += diff_tensor.sum().item()
        if max_abs_diff > global_max_diff:
            global_max_diff = max_abs_diff
            
        # Store non-zero differences
        if mean_abs_diff > 0:
            diff_summary[key] = {
                'mean_abs_diff': mean_abs_diff,
                'max_abs_diff': max_abs_diff
            }
        print(f"{key}: {max_abs_diff:.2f}")

        
        # Free memory for this tensor
        del sd1[key], sd2[key]

print("\n--- Comparison Summary ---")
print(f"Total parameters compared: {len(common_keys)}")
print(f"Parameters with differences: {len(diff_summary)}")
print(f"Total Absolute Difference (Sum): {total_abs_diff:e}")
print(f"Global Maximum Absolute Difference: {global_max_diff:e}")

print("\n--- Top 5 Differing Parameters (by Mean Absolute Difference) ---")
# Sort diff_summary by mean_abs_diff in descending order
sorted_diffs = sorted(diff_summary.items(), key=lambda item: item[1]['mean_abs_diff'], reverse=True)

for key, metrics in sorted_diffs[:5]:
    print(f"  {key}:")
    print(f"    Mean Abs Diff: {metrics['mean_abs_diff']:e}")
    print(f"    Max Abs Diff:  {metrics['max_abs_diff']:e}")

print("\nComparison complete.")