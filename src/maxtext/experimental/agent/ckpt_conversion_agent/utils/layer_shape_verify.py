# Copyright 2026 Google LLC
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


import argparse
import sys
import os
import json
import torch
import numpy as np

# Add maxtext to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING, HOOK_FNS
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import load_hf_dict_from_safetensors, apply_hook_fns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gemma3-4b")
    parser.add_argument("--scan_layers", action="store_true")
    parser.add_argument("--use_multimodal", action="store_true")
    args = parser.parse_args()

    print(f"Loading HF weights from {args.hf_path}...")
    hf_weights = load_hf_dict_from_safetensors(args.hf_path, None, None)

    # Load tracing json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_family = args.model_name.split("-")[0] # e.g. gemma3
    tracing_json_path = os.path.join(script_dir, f"{model_family}_tracing.json")
    if not os.path.exists(tracing_json_path):
        print(f"Tracing JSON not found at {tracing_json_path}")
        sys.exit(1)
        
    with open(tracing_json_path, "r") as f:
        tracing_data = json.load(f)
        
    mt_parameters = tracing_data.get("mt_parameters", {})

    hf_config_obj = HF_MODEL_CONFIGS[args.model_name]
    hf_config_dict = hf_config_obj.to_dict()

    param_mapping = PARAM_MAPPING[args.model_name](hf_config_dict, None, scan_layers=args.scan_layers)
    hook_fns = HOOK_FNS[args.model_name](hf_config_dict, None, scan_layers=args.scan_layers, saving_to_hf=False)

    print("Verifying layer shapes...")
    total_checked = 0
    
    for mt_key, hf_keys in param_mapping.items():
        if mt_key not in mt_parameters:
            print(f"Skip {mt_key}, not in tracing.json")
            continue
            
        target_shape = tuple(mt_parameters[mt_key])
        hook_fn = hook_fns.get(mt_key, lambda x, shape: x)
        
        try:
            if isinstance(hf_keys, str):
                hf_t = hf_weights[hf_keys]
                if isinstance(hf_t, torch.Tensor):
                    hf_t = hf_t.float().numpy()
                jax_t = apply_hook_fns(hf_t, target_shape, hook_fn)
                expected_shape = target_shape
                
            elif isinstance(hf_keys, list):
                if len(hf_keys) == 0:
                    continue
                if isinstance(hf_keys[0], list):
                    hf_t = hf_weights[hf_keys[0][0]]
                else:
                    hf_t = hf_weights[hf_keys[0]]
                
                if isinstance(hf_t, torch.Tensor):
                    hf_t = hf_t.float().numpy()
                
                slice_shape_list = list(target_shape)
                axis_to_stack = 0 if not args.scan_layers else 1
                if len(slice_shape_list) > axis_to_stack:
                    del slice_shape_list[axis_to_stack]
                slice_shape = tuple(slice_shape_list)
                
                jax_t = apply_hook_fns(hf_t, slice_shape, hook_fn)
                expected_shape = slice_shape
                
            elif isinstance(hf_keys, tuple):
                hf_t_list = []
                for k in hf_keys:
                    v = hf_weights[k]
                    if isinstance(v, torch.Tensor):
                        v = v.float().numpy()
                    hf_t_list.append(v)
                hf_t = tuple(hf_t_list)
                jax_t = apply_hook_fns(hf_t, target_shape, hook_fn)
                expected_shape = target_shape
            else:
                continue
                
        except KeyError as e:
            print(f"HF key not found in weights: {e}")
            continue
        except Exception as e:
            print(f"Error applying hook for {mt_key}: {e}")
            print(f"Mismatch/Wrong hook function encountered at {mt_key}. Exiting.")
            sys.exit(1)

        actual_shape = getattr(jax_t, "shape", None)
        if actual_shape is None and isinstance(jax_t, np.ndarray):
            actual_shape = jax_t.shape
        elif actual_shape is None and hasattr(jax_t, "shape"):
            actual_shape = jax_t.shape
            
        if actual_shape is None:
            print(f"Could not determine shape for {mt_key}")
            continue

        if tuple(actual_shape) != expected_shape:
            print(f"Shape mismatch at {mt_key}: expected {expected_shape}, got {tuple(actual_shape)} from applying hook on {hf_keys}")
            print(f"Mismatch encountered. Exiting immediately.")
            sys.exit(1)
            
        total_checked += 1

    print(f"Successfully verified {total_checked} layer shapes.")

if __name__ == "__main__":
    main()
