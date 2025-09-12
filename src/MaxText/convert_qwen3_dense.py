"""
Copyright 2025 Google LLC
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

r"""Convert weights from a Qwen3 dense model to a MaxText one.

This script follows a two-stage conversion process (map-then-transform) to generate
a MaxText checkpoint compatible with scanned model layers.

Example cmd:

python3 -m MaxText.convert_qwen3_dense --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt> --model_size qwen3-4b-Thinking-2507
"""

import argparse
import gc
import os
import pathlib
import sys  # Added for exception hook

import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from MaxText import llama_or_mistral_ckpt, max_logging
from MaxText.inference_utils import str2bool

max_logging.log("Script imports complete. Defining globals.")

CAST_DTYPE = ml_dtypes.bfloat16


def _pt_to_np(pt_weight, cast_dtype=CAST_DTYPE):
  """Helper function to convert a PyTorch tensor to a NumPy array with a specified dtype."""
  return pt_weight.to(torch.float32).numpy().astype(cast_dtype)


# Static model parameters dictionary for Qwen3 dense models
MODEL_PARAMS_DICT = {
    "qwen3-4b-Thinking-2507": {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 2560,
        "head_dim": 128,
        "intermediate_size": 9728,
    }
}


# Static mapping for non-layer weights
NON_LAYER_MAPPING = {
    "model.embed_tokens.weight": "token_embedder.embedding",
    "model.norm.weight": "decoder.decoder_norm.scale",
}

def hf_to_maxtext_mapping(layer_idx: int) -> dict:
    """Creates a mapping from HF weight names to MaxText weight names for a single layer."""
    # This function is definition-only, no logging needed here unless it fails to be called.
    mapping = {
        # Layer-specific mappings for a scanned dense model
        f"model.layers.{layer_idx}.input_layernorm.weight": (
            f"decoder.layers.{layer_idx}.pre_self_attention_layer_norm.scale"
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": (
            f"decoder.layers.{layer_idx}.post_self_attention_layer_norm.scale"
        ),
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"decoder.layers.{layer_idx}.self_attention.query.kernel",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"decoder.layers.{layer_idx}.self_attention.key.kernel",
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"decoder.layers.{layer_idx}.self_attention.value.kernel",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"decoder.layers.{layer_idx}.self_attention.out.kernel",
        f"model.layers.{layer_idx}.self_attn.q_norm.weight": f"decoder.layers.{layer_idx}.self_attention.query_norm.scale",
        f"model.layers.{layer_idx}.self_attn.k_norm.weight": f"decoder.layers.{layer_idx}.self_attention.key_norm.scale",
        # Dense MLP mappings
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"decoder.layers.{layer_idx}.mlp.wi_0.kernel",
        f"model.layers.{layer_idx}.mlp.up_proj.weight": f"decoder.layers.{layer_idx}.mlp.wi_1.kernel",
        f"model.layers.{layer_idx}.mlp.down_proj.weight": f"decoder.layers.{layer_idx}.mlp.wo.kernel",
    }
    return mapping


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
    """Converts a Hugging Face Qwen3 dense checkpoint to a MaxText compatible format."""
    max_logging.log("Starting convert_hf_to_maxtext function...")
    num_layers = model_params["num_hidden_layers"]
    hidden_size = model_params["hidden_size"]
    num_heads = model_params["num_attention_heads"]
    num_kv_heads = model_params["num_key_value_heads"]
    head_dim = model_params["head_dim"]
    intermediate_size = model_params["intermediate_size"]
    max_logging.log(f"Model parameters set: {num_layers} layers, {hidden_size} hidden_size.")

    # Part 1: Load all weights from safetensors into a flat dictionary with MaxText names
    max_logging.log("Beginning Part 1: Loading Safetensors.")
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
    if not ckpt_paths:
        max_logging.log(f"No .safetensors files found at path: {base_model_path}", type="ERROR")
        raise FileNotFoundError(f"No .safetensors files found at path: {base_model_path}")
    max_logging.log(f"Found {len(ckpt_paths)} safetensors files.")
    
    chkpt_vars = {}
    layer_mappings = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        max_logging.log(f"Opening file {i+1}: {ckpt_path.name}")
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in NON_LAYER_MAPPING:
                    maxtext_key = NON_LAYER_MAPPING[key]
                    chkpt_vars[maxtext_key] = f.get_tensor(key)
                elif "layers" in key:
                    layer_idx_str = key.split(".")[2]
                    if layer_idx_str.isdigit():
                        layer_idx = int(layer_idx_str)
                        if layer_idx not in layer_mappings:
                            layer_mappings[layer_idx] = hf_to_maxtext_mapping(layer_idx)
                        
                        maxtext_key = layer_mappings[layer_idx].get(key)
                        if maxtext_key:
                            chkpt_vars[maxtext_key] = f.get_tensor(key)
                else:
                    max_logging.log(f"Skipping key: {key}", type="WARNING")
    
    max_logging.log(f"Finished loading tensors. Total keys loaded into chkpt_vars: {len(chkpt_vars)}")
    if not chkpt_vars:
        max_logging.log("No tensors were loaded. Check mapping function and checkpoint files.", type="ERROR")
        raise ValueError("Loaded 0 tensors. Checkpoint files may be empty or mapping is incorrect.")

    # Part 2: Initialize, populate, and transform the weights for MaxText
    max_logging.log("Beginning Part 2: Initializing MaxText weights structure.")
    maxtext_weights = {
        "decoder": {
            "layers": {
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
                "self_attention": {
                    "query": {"kernel": None},
                    "key": {"kernel": None},
                    "value": {"kernel": None},
                    "out": {"kernel": None},
                    "query_norm": {"scale": None},
                    "key_norm": {"scale": None},
                },
                "mlp": {
                    "wi_0": {"kernel": None},
                    "wi_1": {"kernel": None},
                    "wo": {"kernel": None},
                },
            },
            "decoder_norm": {"scale": None},
        },
        "token_embedder": {"embedding": None},
    }
    max_logging.log("MaxText weights dictionary initialized.")

    max_logging.log("Populating non-layer weights...")
    maxtext_weights["token_embedder"]["embedding"] = _pt_to_np(chkpt_vars["token_embedder.embedding"], cast_dtype=CAST_DTYPE)
    maxtext_weights["decoder"]["decoder_norm"]["scale"] = _pt_to_np(chkpt_vars["decoder.decoder_norm.scale"], cast_dtype=CAST_DTYPE)
    max_logging.log("Non-layer weights populated.")

    max_logging.log("Allocating stacked NumPy arrays for layer weights...")
    ln = maxtext_weights["decoder"]["layers"]
    s_attn = ln["self_attention"]
    mlp = ln["mlp"]

    # Pre-allocate stacked arrays
    ln["pre_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=CAST_DTYPE)
    ln["post_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=CAST_DTYPE)
    s_attn["query"]["kernel"] = np.zeros((num_layers, hidden_size, num_heads, head_dim), dtype=CAST_DTYPE)
    s_attn["key"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=CAST_DTYPE)
    s_attn["value"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=CAST_DTYPE)
    s_attn["out"]["kernel"] = np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=CAST_DTYPE)
    s_attn["query_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=CAST_DTYPE)
    s_attn["key_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=CAST_DTYPE)
    mlp["wi_0"]["kernel"] = np.zeros((num_layers, hidden_size, intermediate_size), dtype=CAST_DTYPE)
    mlp["wi_1"]["kernel"] = np.zeros((num_layers, hidden_size, intermediate_size), dtype=CAST_DTYPE)
    mlp["wo"]["kernel"] = np.zeros((num_layers, intermediate_size, hidden_size), dtype=CAST_DTYPE)
    max_logging.log("All NumPy arrays allocated successfully.")

    # Loop through layers and populate the stacked arrays
    max_logging.log(f"Starting layer weight stacking loop for {num_layers} layers...")
    for l in tqdm(range(num_layers), desc="Stacking layer weights"):
        ln["pre_self_attention_layer_norm"]["scale"][l, :] = _pt_to_np(
            chkpt_vars[f"decoder.layers.{l}.pre_self_attention_layer_norm.scale"]
        )
        ln["post_self_attention_layer_norm"]["scale"][l, :] = _pt_to_np(
            chkpt_vars[f"decoder.layers.{l}.post_self_attention_layer_norm.scale"]
        )
        s_attn["query_norm"]["scale"][l, :] = _pt_to_np(
            chkpt_vars[f"decoder.layers.{l}.self_attention.query_norm.scale"]
        )
        s_attn["key_norm"]["scale"][l, :] = _pt_to_np(
            chkpt_vars[f"decoder.layers.{l}.self_attention.key_norm.scale"]
        )

        s_attn["query"]["kernel"][l, ...] = (
            _pt_to_np(chkpt_vars[f"decoder.layers.{l}.self_attention.query.kernel"])
            .T.reshape(hidden_size, num_heads, head_dim)
        )
        s_attn["key"]["kernel"][l, ...] = (
            _pt_to_np(chkpt_vars[f"decoder.layers.{l}.self_attention.key.kernel"])
            .T.reshape(hidden_size, num_kv_heads, head_dim)
        )
        s_attn["value"]["kernel"][l, ...] = (
            _pt_to_np(chkpt_vars[f"decoder.layers.{l}.self_attention.value.kernel"])
            .T.reshape(hidden_size, num_kv_heads, head_dim)
        )
        s_attn["out"]["kernel"][l, ...] = (
            _pt_to_np(chkpt_vars[f"decoder.layers.{l}.self_attention.out.kernel"])
            .T.reshape(num_heads, head_dim, hidden_size)
        )

        mlp["wi_0"]["kernel"][l, ...] = _pt_to_np(chkpt_vars[f"decoder.layers.{l}.mlp.wi_0.kernel"]).T
        mlp["wi_1"]["kernel"][l, ...] = _pt_to_np(chkpt_vars[f"decoder.layers.{l}.mlp.wi_1.kernel"]).T
        mlp["wo"]["kernel"][l, ...] = _pt_to_np(chkpt_vars[f"decoder.layers.{l}.mlp.wo.kernel"]).T

    max_logging.log("Finished layer weight stacking loop.")

    # Final transformations for scanned weights (swap layer and feature axes)
    max_logging.log("Transposing layer weights for MaxText scanned format...")
    ln["pre_self_attention_layer_norm"]["scale"] = np.transpose(ln["pre_self_attention_layer_norm"]["scale"], axes=(1, 0))
    ln["post_self_attention_layer_norm"]["scale"] = np.transpose(ln["post_self_attention_layer_norm"]["scale"], axes=(1, 0))
    s_attn["query_norm"]["scale"] = np.transpose(s_attn["query_norm"]["scale"], axes=(1, 0))
    s_attn["key_norm"]["scale"] = np.transpose(s_attn["key_norm"]["scale"], axes=(1, 0))

    s_attn["query"]["kernel"] = np.transpose(s_attn["query"]["kernel"], axes=(1, 0, 2, 3))
    s_attn["key"]["kernel"] = np.transpose(s_attn["key"]["kernel"], axes=(1, 0, 2, 3))
    s_attn["value"]["kernel"] = np.transpose(s_attn["value"]["kernel"], axes=(1, 0, 2, 3))
    s_attn["out"]["kernel"] = np.transpose(s_attn["out"]["kernel"], axes=(1, 0, 2, 3))

    mlp["wi_0"]["kernel"] = np.transpose(mlp["wi_0"]["kernel"], axes=(1, 0, 2))
    mlp["wi_1"]["kernel"] = np.transpose(mlp["wi_1"]["kernel"], axes=(1, 0, 2))
    mlp["wo"]["kernel"] = np.transpose(mlp["wo"]["kernel"], axes=(1, 0, 2))
    
    max_logging.log("Finished final transposition.")

    gc.collect()
    max_logging.log("Cleanup complete. Returning final weights dictionary.")
    return maxtext_weights


def main(args):
    """Main function to run the conversion."""
    try:
        max_logging.log(f"Starting main() with args: {args}")
        # Set up JAX simulated environment for checkpoint saving
        max_logging.log("Setting JAX environment variables...")
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
        max_logging.log("JAX environment variables set.")

        if args.model_size not in MODEL_PARAMS_DICT:
            max_logging.log(f"Model size '{args.model_size}' not found.", type="ERROR")
            raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")

        max_logging.log(f"Looking up model params for: {args.model_size}")
        model_params = MODEL_PARAMS_DICT[args.model_size]
        max_logging.log(f"Starting conversion for Qwen3 dense model size: {args.model_size}")
        
        max_logging.log("Calling convert_hf_to_maxtext...")
        jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params)
        max_logging.log("convert_hf_to_maxtext returned successfully.")
        
        max_logging.log(f"Conversion complete. Calling save_weights_to_checkpoint at {args.maxtext_model_path}")
        llama_or_mistral_ckpt.save_weights_to_checkpoint(
            args.maxtext_model_path, jax_weights, args.simulated_cpu_devices_count, args.use_ocdbt, args.use_zarr3
        )
        max_logging.log("Checkpoint saved successfully.")
    
    except Exception as e:
        max_logging.log(f"A Python exception occurred: {e}")
        max_logging.log("Script failed due to an exception.")
        raise e # Re-raise the exception to get the full traceback
    
    finally:
        max_logging.log("Main function 'finally' block reached. Script is exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3 dense HF weights to MaxText.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3 checkpoint files.")
    parser.add_argument(
        "--maxtext_model_path", type=str, required=True, help="Path to save the MaxText checkpoint (local or GCS)."
    )
    parser.add_argument(
        "--model_size", type=str, required=True, choices=MODEL_PARAMS_DICT.keys(), help="The model size to convert."
    )
    parser.add_argument(
        "--simulated_cpu_devices_count", type=int, default=16, help="Number of simulated CPU devices for saving."
    )
    parser.add_argument("--use-ocdbt", type=str2bool, default=True, help="Use OCDBT format for saving.")
    parser.add_argument("--use-zarr3", type=str2bool, default=True, help="Use Zarr3 format for saving.")

    parsed_args = parser.parse_args()
    main(parsed_args)