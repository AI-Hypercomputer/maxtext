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
import json

# Add maxtext to python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from transformers import AutoModelForCausalLM, AutoConfig

def load_mt_ckpt(mt_path):
    if not mt_path.startswith("gs://"):
        mt_path = os.path.abspath(mt_path)
    
    import jax
    import numpy as np
    from etils import epath
    import orbax.checkpoint as ocp
    
    checkpoint_path = epath.Path(mt_path)
    
    if (checkpoint_path / "_CHECKPOINT_METADATA").exists() or (checkpoint_path / "items").exists():
        root_path = checkpoint_path.parent
        step = int(checkpoint_path.name)
    else:
        root_path = checkpoint_path
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

def get_shape(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    elif isinstance(x, (tuple, list)):
        return [get_shape(i) for i in x]
    return str(type(x))

def main():
    parser = argparse.ArgumentParser(description="Trace HuggingFace model and MaxText checkpoint to extract layer names and shapes.")
    parser.add_argument("--hf_path", type=str, default=None, help="Path to HF model")
    parser.add_argument("--mt_path", type=str, default=None, help="Path to MaxText checkpoint")
    parser.add_argument("--mt_args", type=str, default=None, help="MaxText pyconfig args string (e.g. 'models/phi4.yml scan_layers=False') to trace model abstract parameters.")
    parser.add_argument("--output_file", type=str, default="tracing_result.json")
    parser.add_argument("--unroll_scan_layers", action="store_true", help="Unroll scanned layers in MT checkpoint")
    args = parser.parse_args()

    result = {
        "hf_modules": {},
        "hf_parameters": {},
        "mt_parameters": {}
    }

    if args.hf_path:
        print(f"Loading HuggingFace model from {args.hf_path}...")
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        hf_model.eval()

        print("Registering hooks and collecting parameters...")
        for name, param in hf_model.named_parameters():
            result["hf_parameters"][name] = list(param.shape)

        for name, module in hf_model.named_modules():
            mod_params = {}
            for pn, pv in module.named_parameters(recurse=False):
                mod_params[pn] = list(pv.shape)
            
            result["hf_modules"][name] = {
                "type": module.__class__.__name__,
                "parameters": mod_params,
                "input_shape": None,
                "output_shape": None
            }

        def get_hook(name):
            def hook(module, args, output):
                try:
                    result["hf_modules"][name]["input_shape"] = get_shape(args)
                    result["hf_modules"][name]["output_shape"] = get_shape(output)
                except Exception as e:
                    print(f"Error in hook for {name}: {e}")
            return hook

        for name, module in hf_model.named_modules():
            module.register_forward_hook(get_hook(name))

        print("Running dummy forward pass...")
        if hasattr(hf_model.config, "vocab_size"):
            vocab_size = hf_model.config.vocab_size
        elif hasattr(hf_model.config, "text_config") and hasattr(hf_model.config.text_config, "vocab_size"):
            vocab_size = hf_model.config.text_config.vocab_size
        else:
            vocab_size = 32000
        dummy_input = torch.randint(0, vocab_size, (1, 128))
        with torch.no_grad():
            hf_model(dummy_input)
            
        # Determine num_layers if hf_model is loaded
    num_layers = None
    if args.hf_path:
        # We can parse config directly if hf_model is not easily accessible globally
        config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "n_layer"):
            num_layers = config.n_layer

    if args.mt_args:
        print(f"Loading MaxText model from args: {args.mt_args}...")
        import jax
        from maxtext import pyconfig
        from maxtext.inference.maxengine import maxengine

        pyconfig_argv = [""] + args.mt_args.split()
        config = pyconfig.initialize(pyconfig_argv)
        
        engine = maxengine.MaxEngine(config)
        rng = jax.random.PRNGKey(1234)
        _, rng_load_params = jax.random.split(rng)
        loaded_params_from_engine = engine.load_params(rng_load_params)
        
        actual_weights_dict = loaded_params_from_engine.get("params")
        if actual_weights_dict is None:
            actual_weights_dict = loaded_params_from_engine
            
        leaves_with_paths = jax.tree_util.tree_leaves_with_path(actual_weights_dict)

        for path_tuple, leaf_value in leaves_with_paths:
            key_parts = []
            for p_entry in path_tuple:
                if isinstance(p_entry, jax.tree_util.DictKey):
                    key_parts.append(p_entry.key)
                elif isinstance(p_entry, jax.tree_util.SequenceKey):
                    key_parts.append(str(p_entry.idx))
                else:
                    key_parts.append(f"__unhandled_key_{type(p_entry).__name__}__")
            
            k = "-".join(key_parts)
            
            if hasattr(leaf_value, "shape"):
                shape = list(leaf_value.shape)
            else:
                shape = "unknown"
            
            if args.unroll_scan_layers and "layers-" in k and num_layers is not None and num_layers in shape:
                layer_dim = shape.index(num_layers)
                new_shape = list(shape)
                new_shape.pop(layer_dim)
                for i in range(num_layers):
                    new_k = k.replace("layers-", f"layers_{i}-")
                    result["mt_parameters"][new_k] = new_shape
            else:
                result["mt_parameters"][k] = shape

    elif args.mt_path:
            print(f"Loading MaxText checkpoint from {args.mt_path}...")
            mt_state = load_mt_ckpt(args.mt_path)
            if mt_state:
                mt_flat_raw = flatten_dict(mt_state, parent_key="")
            for k, v in mt_flat_raw.items():
                if k.startswith("-"):
                    k = k[1:]
                while k.startswith("params-params-"):
                    k = k[len("params-"):]
                
                # Try to get shape
                if hasattr(v, "shape"):
                    shape = list(v.shape)
                else:
                    shape = "unknown"
                
                if args.unroll_scan_layers and "layers-" in k and num_layers is not None and num_layers in shape:
                    layer_dim = shape.index(num_layers)
                    new_shape = list(shape)
                    new_shape.pop(layer_dim)
                    for i in range(num_layers):
                        new_k = k.replace("layers-", f"layers_{i}-")
                        result["mt_parameters"][new_k] = new_shape
                else:
                    result["mt_parameters"][k] = shape

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
