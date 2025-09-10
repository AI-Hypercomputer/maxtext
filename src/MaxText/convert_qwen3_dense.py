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

import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from MaxText import llama_or_mistral_ckpt, max_logging
from MaxText.inference_utils import str2bool

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


def hf_to_maxtext_mapping(layer_idx: int) -> dict:
    """Creates a mapping from HF weight names to MaxText weight names for a single layer."""
    mapping = {
        # Non-layer weights (only need to be defined once)
        "model.embed_tokens.weight": "token_embedder.embedding",
        "model.norm.weight": "decoder.decoder_norm.scale",
    }
    # Layer-specific mappings for a scanned dense model
    mapping.update({
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
    })

    return mapping


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
    """Converts a Hugging Face Qwen3 dense checkpoint to a MaxText compatible format."""
    num_layers = model_params["num_hidden_layers"]
    hidden_size = model_params["hidden_size"]
    num_heads = model_params["num_attention_heads"]
    num_kv_heads = model_params["num_key_value_heads"]
    head_dim = model_params["head_dim"]
    intermediate_size = model_params["intermediate_size"]

    # Part 1: Load all weights from safetensors into a flat dictionary with MaxText names
    ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
    chkpt_vars = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)}...")
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Determine layer index from key
                layer_idx = 0
                if "layers" in key:
                    layer_idx_str = key.split(".")[2]
                    if layer_idx_str.isdigit():
                        layer_idx = int(layer_idx_str)

                # Get the corresponding MaxText key
                maxtext_key = hf_to_maxtext_mapping(layer_idx).get(key)
                if maxtext_key:
                    chkpt_vars[maxtext_key] = f.get_tensor(key)

    # Part 2: Initialize, populate, and transform the weights for MaxText
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

    max_logging.log("Populating non-layer weights...")
    maxtext_weights["token_embedder"]["embedding"] = _pt_to_np(chkpt_vars["token_embedder.embedding"])
    maxtext_weights["decoder"]["decoder_norm"]["scale"] = _pt_to_np(chkpt_vars["decoder.decoder_norm.scale"])

    max_logging.log("Allocating and stacking layer weights...")
    ln = maxtext_weights["decoder"]["layers"]
    s_attn = ln["self_attention"]
    mlp = ln["mlp"]

    # Pre-allocate stacked arrays with the 'layer' dimension first for efficiency
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

    # Loop through layers and populate the stacked arrays
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

    gc.collect()
    return maxtext_weights