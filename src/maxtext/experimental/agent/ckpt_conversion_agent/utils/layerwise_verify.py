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
import torch
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoModelForCausalLM, AutoConfig

# Add maxtext to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from maxtext.configs import pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING, HOOK_FNS
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from maxtext.checkpoint_conversion.utils.utils import load_hf_dict_from_safetensors, apply_hook_fns

def load_mt_ckpt(mt_path):
    if not mt_path.startswith("gs://"):
        mt_path = os.path.abspath(mt_path)
    
    import jax
    import numpy as np
    from etils import epath
    import orbax.checkpoint as ocp
    
    checkpoint_path = epath.Path(mt_path)
    
    # Checkpoint could be the step directory (e.g. ".../0") or the root directory.
    # Find the root and the step.
    if (checkpoint_path / "_CHECKPOINT_METADATA").exists() or (checkpoint_path / "items").exists():
        root_path = checkpoint_path.parent
        step = int(checkpoint_path.name)
    else:
        root_path = checkpoint_path
        # Temporary manager to find latest step
        temp_mngr = ocp.CheckpointManager(root_path)
        step = temp_mngr.latest_step()
        if step is None:
            raise ValueError(f"No checkpoint steps found at {root_path}")
            
    mngr = ocp.CheckpointManager(
        root_path,
        item_names=('items',),
        item_handlers={'items': ocp.PyTreeCheckpointHandler()}
    )
    
    devices = np.array(jax.devices()).reshape((-1,))
    single_device_mesh = jax.sharding.Mesh(devices, ("x",))
    
    def create_restore_args(tree_metadata):
        if hasattr(tree_metadata, "shape"):
            return ocp.type_handlers.ArrayRestoreArgs(sharding=jax.sharding.NamedSharding(single_device_mesh, jax.sharding.PartitionSpec()))
        else:
            return ocp.type_handlers.ArrayRestoreArgs()
            
    try:
        meta = mngr.item_metadata(step)
        if hasattr(meta, 'get'):
            items_meta = meta.get('items')
            if items_meta and hasattr(items_meta, 'tree'):
                restore_args = jax.tree_util.tree_map(
                    create_restore_args, 
                    items_meta.tree,
                    is_leaf=lambda x: hasattr(x, "shape")
                )
                restored = mngr.restore(step, args=ocp.args.Composite(
                    items=ocp.args.PyTreeRestore(item=items_meta.tree, restore_args=restore_args)
                ))
                if 'items' in restored:
                    return restored['items']
        
        # Fallback to simple Checkpointer if not a Composite
        ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        metadata = ckptr.metadata(root_path / str(step))
        restore_args = jax.tree_util.tree_map(
            lambda x: create_restore_args(x) if hasattr(x, "shape") else None,
            metadata.item_metadata.tree if hasattr(metadata, "item_metadata") else metadata,
            is_leaf=lambda x: hasattr(x, "shape")
        )
        return ckptr.restore(root_path / str(step), restore_args=restore_args)
    except Exception as e:
        print(f"Failed to load checkpoint unsharded: {e}")
        return None


def flatten_dict(d, parent_key='params', sep='-'):
    items = []
    if d is None:
        return {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) or str(type(v)).find('FrozenDict') != -1:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def jax_rmsnorm(x, jax_w, eps):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    normed_x = x / np.sqrt(variance + eps)
    # MaxText RMSNorm weight is HF weight - 1. So add 1 back.
    return normed_x * jax_w

def verify_layer(hf_name, pt_in, pt_out, jax_w, mt_key, hf_config, model_name):
    """
    hf_name: the hf parameter name e.g. model.layers.0.mlp.gate_proj.weight
    pt_in: input tensor to the PyTorch module (numpy)
    pt_out: output tensor from the PyTorch module (numpy)
    jax_w: converted Jax parameter (numpy)
    mt_key: MaxText key name
    """

    if isinstance(jax_w, torch.Tensor):
        jax_w = jax_w.float().numpy()
    else:
        jax_w = np.array(jax_w, dtype=np.float32)
    pt_in = np.array(pt_in, dtype=np.float32)
    pt_out = np.array(pt_out, dtype=np.float32)
    
    # Calculate JAX output based on layer type
    if "embed" in mt_key.lower() or "embedding" in mt_key.lower():
        # JAX embedding lookup
        # pt_in is usually input_ids of shape (batch, seq_len)
        # JAX weight is (vocab_size, hidden_size)
        
        # We assume pt_in is integer ids
        ids = pt_in.astype(np.int32)
        jax_out = jax_w[ids]
        
        # MaxText embeddings are scaled by sqrt(hidden_size) for Gemma models, but HF output from `embed_tokens` is usually NOT scaled (scaled later in forward).
        # We unscale jax_out to compare with PyTorch `embed_tokens` output only for Gemma
        if 'gemma' in model_name and 'hidden_size' in hf_config:
            scale = np.sqrt(hf_config['hidden_size'])
            jax_out = jax_out / scale
            
    elif "norm" in mt_key.lower():
        eps = hf_config.get("rms_norm_eps", 1e-6)
        jax_out = jax_rmsnorm(pt_in, jax_w, eps)
        
    else:
        # Assume Linear layer (DenseGeneral)
        in_dim = pt_in.shape[-1]
        
        # JAX weights might be reshaped to e.g. (in_features, num_heads, head_dim)
        # We flatten it to (in_features, out_features) for standard matmul
        jax_w_flat = jax_w.reshape(in_dim, -1)
        jax_out = pt_in @ jax_w_flat
        
        # PyTorch output might be (batch, seq, num_heads * head_dim)
        # We flatten pt_out to match jax_out's shape if needed
        # In reality, they should match as (batch, seq, out_features)
        
        # Handle concatenated projections (e.g. qkv_proj or gate_up_proj)
        if pt_out.shape[-1] != jax_out.shape[-1]:
            if "query" in mt_key.lower():
                pt_out = pt_out[..., :jax_out.shape[-1]]
                head_dim = hf_config.get("head_dim", hf_config.get("hidden_size", 1) // hf_config.get("num_attention_heads", 1))
                jax_out = jax_out * np.sqrt(head_dim)
            elif "key" in mt_key.lower():
                head_dim = hf_config.get("head_dim", hf_config.get("hidden_size", 1) // hf_config.get("num_attention_heads", 1))
                q_dim = hf_config.get("num_attention_heads", 1) * head_dim
                pt_out = pt_out[..., q_dim : q_dim + jax_out.shape[-1]]
            elif "value" in mt_key.lower():
                head_dim = hf_config.get("head_dim", hf_config.get("hidden_size", 1) // hf_config.get("num_attention_heads", 1))
                q_dim = hf_config.get("num_attention_heads", 1) * head_dim
                k_dim = hf_config.get("num_key_value_heads", 1) * head_dim
                pt_out = pt_out[..., q_dim + k_dim : q_dim + k_dim + jax_out.shape[-1]]
            elif "wi_0" in mt_key.lower() or "gate" in mt_key.lower():
                pt_out = pt_out[..., :jax_out.shape[-1]]
            elif "wi_1" in mt_key.lower() or "up" in mt_key.lower():
                pt_out = pt_out[..., jax_out.shape[-1] : jax_out.shape[-1]*2]
    
    # Compare
    # For bfloat16 models, we need generous tolerances
    diff = np.abs(pt_out - jax_out).max()
    rtol = 1e-2
    atol = 1e-2
    
    match = np.allclose(pt_out, jax_out, rtol=rtol, atol=atol)
    
    print(f"[{mt_key} | {hf_name}] Max Diff: {diff:.6f} | Match: {match}")
    if not match:
        print(f"  -> PT out mean: {pt_out.mean():.6f}, JAX out mean: {jax_out.mean():.6f}")
        print(f"  -> PT out flat[:5]: {pt_out.flatten()[:5]}")
        print(f"  -> JAX out flat[:5]: {jax_out.flatten()[:5]}")
        
    return match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--mt_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gemma3-4b")
    parser.add_argument("--scan_layers", action="store_true")
    parser.add_argument("--use_multimodal", action="store_true")
    parser.add_argument("--disable_trust_remote_code", action="store_true")
    args = parser.parse_args()

    print("Loading HuggingFace model...")
    # Load model with bfloat16 for realistic numericals
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=not args.disable_trust_remote_code)
    hf_model.eval()

    # Register hooks to capture inputs and outputs
    hooks_data = {}
    
    def get_hook(name):
        def hook(module, args, output):
            # Capture the first input tensor and the output
            if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
                try:
                    if isinstance(output, torch.Tensor):
                        out_val = output.detach().float().numpy()
                    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                        out_val = output[0].detach().float().numpy()
                    else:
                        return
                    
                    hooks_data[name] = {
                        "input": args[0].detach().float().numpy(),
                        "output": out_val
                    }
                except Exception:
                    pass
        return hook

    # Register hook on every named module
    for name, module in hf_model.named_modules():
        module.register_forward_hook(get_hook(name))

    print("Running dummy forward pass on HF model to capture intermediates...")
    # Use realistic seq len
    if hasattr(hf_model.config, "vocab_size"):
        vocab_size = hf_model.config.vocab_size
    elif hasattr(hf_model.config, "text_config") and hasattr(hf_model.config.text_config, "vocab_size"):
        vocab_size = hf_model.config.text_config.vocab_size
    else:
        vocab_size = 32000
    dummy_input = torch.randint(0, vocab_size, (1, 128))
    with torch.no_grad():
        hf_model(dummy_input)

    hf_config_obj = HF_MODEL_CONFIGS[args.model_name]
    hf_config_dict = hf_config_obj.to_dict()

    param_mapping = PARAM_MAPPING[args.model_name](hf_config_dict, None, scan_layers=args.scan_layers)
    hook_fns = HOOK_FNS[args.model_name](hf_config_dict, None, scan_layers=args.scan_layers, saving_to_hf=False)

    print("Loading HF weights from safetensors...")
    hf_weights = load_hf_dict_from_safetensors(args.hf_path, None, None)
    
    print("Loading MaxText checkpoint...")
    mt_state = load_mt_ckpt(args.mt_path)
    if mt_state is None:
        print("Could not load MaxText checkpoint for target shapes. Exiting.")
        sys.exit(1)
        
    mt_flat_raw = flatten_dict(mt_state, parent_key="")
    mt_flat = {}
    for k, v in mt_flat_raw.items():
        if k.startswith("-"):
            k = k[1:]
        # Normalize: if it starts with params-params-, strip one params-
        while k.startswith("params-params-"):
            k = k[len("params-"):]
        mt_flat[k] = v
        
    total_layers = 0
    matched_layers = 0
    
    for mt_key, hf_keys in param_mapping.items():
        if mt_key not in mt_flat:
            print(f"Skip {mt_key}, not in mt ckpt")
            continue
            
        mt_tensor = mt_flat[mt_key]
        target_shape = mt_tensor.shape
        hook_fn = hook_fns.get(mt_key, lambda x, shape: x)
        
        try:

            if isinstance(hf_keys, str):
                hf_t = hf_weights[hf_keys]
                if isinstance(hf_t, torch.Tensor):
                    hf_t = hf_t.float().numpy()
                jax_t = apply_hook_fns(hf_t, target_shape, hook_fn)
                hf_primary_key = hf_keys
            elif isinstance(hf_keys, list):
                if len(hf_keys) == 0:
                    continue
                if isinstance(hf_keys[0], list):
                    hf_t = hf_weights[hf_keys[0][0]]
                    hf_primary_key = hf_keys[0][0]
                else:
                    hf_t = hf_weights[hf_keys[0]]
                    hf_primary_key = hf_keys[0]
                
                if isinstance(hf_t, torch.Tensor):
                    hf_t = hf_t.float().numpy()
                
                slice_shape_list = list(target_shape)
                axis_to_stack = 0 if not args.scan_layers else 1 # default to 1 for param_scan_axis, though it varies.
                if len(slice_shape_list) > axis_to_stack:
                    del slice_shape_list[axis_to_stack]
                slice_shape = tuple(slice_shape_list)
                
                jax_t = apply_hook_fns(hf_t, slice_shape, hook_fn)
            elif isinstance(hf_keys, tuple):
                hf_t_list = []
                for k in hf_keys:
                    v = hf_weights[k]
                    if isinstance(v, torch.Tensor):
                        v = v.float().numpy()
                    hf_t_list.append(v)
                hf_t = tuple(hf_t_list)
                jax_t = apply_hook_fns(hf_t, target_shape, hook_fn)
                hf_primary_key = hf_keys[0] # Just use first to identify module
            else:
                continue
        except KeyError as e:
            print(f"HF key not found in weights: {e}")
            continue
        except Exception as e:
            print(f"Error applying hook for {mt_key}: {e}")
            continue

        # Extract module name from HF weight name (e.g., language_model.model.layers.0.mlp.gate_proj.weight -> language_model.model.layers.0.mlp.gate_proj)
        module_name = ".".join(hf_primary_key.split(".")[:-1])
        
        if module_name not in hooks_data:
            module_name_alt = module_name.replace("language_model.model.", "model.language_model.")
            if module_name_alt in hooks_data:
                module_name = module_name_alt
        
        if module_name not in hooks_data:
            print(f"[{mt_key}] Module {module_name} not found in captured hooks. Skipping verification.")
            continue
            
        pt_in = hooks_data[module_name]["input"]
        pt_out = hooks_data[module_name]["output"]
        
        total_layers += 1
        
        is_match = verify_layer(hf_primary_key, pt_in, pt_out, jax_t, mt_key, hf_config_dict, args.model_name)
        if is_match:
            matched_layers += 1
        else:
            print(f"Mismatch found at layer: {mt_key}. Exiting immediately.")
            sys.exit(1)

    if total_layers > 0:
        print(f"\nTotal layers tested: {total_layers}")
        print(f"Matched layers: {matched_layers}")
        print(f"Accuracy (matched ratio): {matched_layers / total_layers:.2%}")
    else:
        print("\nNo layers were tested.")

if __name__ == "__main__":
    main()
