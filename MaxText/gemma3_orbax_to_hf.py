#!/usr/bin/env python
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

This script converts a maxtext Gemma3 checkpoint (in Flax/Orbax format)
into a Hugging Face–compatible PyTorch checkpoint.
It assumes that the maxtext checkpoint was produced by the conversion
script from Keras → maxtext and that the maxtext parameter tree has the following structure:

  {
    "params": {
      "token_embedder": {
         "embedding": <np.ndarray>  # shape (vocab_size, embed_dim), already scaled by sqrt(embed_dim)
      },
      "decoder": {
         "decoder_norm": { "scale": <np.ndarray> },  # final norm; stored as (original_scale + 1)
         "layers": {
             "self_attention": {
                "query": { "kernel": <np.ndarray> },   # shape (num_layers, num_heads, hidden_size, head_dim)
                "key":   { "kernel": <np.ndarray> },     # shape (num_layers, num_kv_heads, hidden_size, head_dim)
                "value": { "kernel": <np.ndarray> },      # shape (num_layers, num_kv_heads, hidden_size, head_dim)
                "out":   { "kernel": <np.ndarray> }       # shape (num_layers, num_heads, head_dim, hidden_size)
             },
             "mlp": {
                "wi_0": { "kernel": <np.ndarray> },      # gate_proj weight; shape (num_layers, intermediate_dim, hidden_size)
                "wi_1": { "kernel": <np.ndarray> },      # up_proj weight; shape (num_layers, intermediate_dim, hidden_size)
                "wo":   { "kernel": <np.ndarray> }       # down_proj weight; shape (num_layers, hidden_size, intermediate_dim)
             },
             "pre_self_attention_norm": { "scale": <np.ndarray> },    # shape (num_layers, hidden_size) (stored as original+1)
             "post_self_attention_norm": { "scale": <np.ndarray> },
             "pre_ffw_norm": { "scale": <np.ndarray> },
             "post_ffw_norm": { "scale": <np.ndarray> },
         }
      }
    }
  }

The resulting Hugging Face checkpoint will be a flat state dict whose keys follow
the HF Gemma3 model naming (for example, "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", etc.).

Usage:
    JAX_PLATFORMS=cpu python MaxText/gemma3_orbax_to_hf.py MaxText/configs/base.yml \
        base_output_directory=/tmp/output \
        load_parameters_path=/path/to/maxtext/checkpoint \
        model_name='gemma3-4b' \
        hf_model_path=/path/to/save/hf_model.bin \
        model_size=4b
"""

import math
import numpy as np
import torch
import jax
from absl import app
import pyconfig
import maxengine
from transformers import AutoConfig, Gemma3ForCausalLM
from tqdm import tqdm
from typing import Sequence

jax.config.update("jax_platform_name", "cpu")

# Maximum vocabulary size.
GEMMA_VOCAB_SIZE = 262144

def load_hf_model(model_size):
    """
    Load the Hugging Face Gemma3 model based on the provided model size.
    """
    if model_size == "4b":
        config = AutoConfig.from_pretrained("google/gemma-3-4b-it").text_config
        config.vocab_size = GEMMA_VOCAB_SIZE
        model = Gemma3ForCausalLM(config)
    elif model_size == "12b":
        config = AutoConfig.from_pretrained("google/gemma-3-12b-it").text_config
        config.vocab_size = GEMMA_VOCAB_SIZE
        model = Gemma3ForCausalLM(config)
    elif model_size == "27b":
        config = AutoConfig.from_pretrained("google/gemma-3-27b-it").text_config
        config.vocab_size = GEMMA_VOCAB_SIZE
        model = Gemma3ForCausalLM(config)
    else:
        raise NotImplementedError(f"Model size {model_size} not supported.")
    return model

def load_maxtext_params(config):
    """
    Loads model parameters from the given maxtext checkpoint using the MaxEngine.
    This function temporarily sets the necessary paths from the configuration,
    calls load_params, and returns the inner parameters.
    """
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng_load_params)
    return params['params']

def reverse_scale(arr, scale):
    """
    Reverse the scaling applied in maxtext. For the query weight,
    maxtext has the scaling factor baked into the weight, so we reverse it here.
    """
    return arr * np.sqrt(scale)

def get_query_pre_attn_scalar(config) -> float:
    """
    Returns the scalar to reverse the query pre-attention scaling.
    """
    if config.model_name in ["gemma3-4b", "gemma3-12b"]:
        return config.head_dim ** -0.5
    elif config.model_name == "gemma3-27b":
        return (config.base_emb_dim // config.base_num_query_heads) ** -0.5
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")

def convert_maxtext_to_hf(config, model_size):
    """
    Converts the maxtext checkpoint parameters for Gemma3 to a Hugging Face–compatible
    state dict.
    """
    params = load_maxtext_params(config)
    hf_state = {}

    # --- Token Embedding ---
    # In maxtext, the token embeddings are scaled by sqrt(embed_dim)
    embed = np.array(params["token_embedder"]["embedding"])[:GEMMA_VOCAB_SIZE]
    _, embed_dim = embed.shape
    hf_embed = embed / math.sqrt(embed_dim)
    hf_state["model.embed_tokens.weight"] = torch.tensor(
        hf_embed.astype(np.float32), dtype=torch.bfloat16
    )

    # --- Final (Decoder) Norm ---
    # The final norm was stored as (original_scale + 1) in maxtext
    final_norm = np.array(params["decoder"]["decoder_norm"]["scale"])
    hf_state["model.norm.weight"] = torch.tensor(
        (final_norm - 1).astype(np.float32), dtype=torch.bfloat16
    )

    # --- Layers Conversion ---
    layers = params["decoder"]["layers"]

    # Assume the self_attention.query kernel has shape:
    # (num_layers, num_heads, hidden_size, head_dim)
    self_attn_query = np.array(layers["self_attention"]["query"]["kernel"])
    print(self_attn_query.shape)
    total_layers = self_attn_query.shape[1]

    query_pre_attn_scalar = get_query_pre_attn_scalar(config)

    for layer_idx in tqdm(range(total_layers), desc="Converting layers"):
        prefix = f"model.layers.{layer_idx}."

        # Self-Attention Parameters
        # --- q_proj.weight ---
        q = reverse_scale(
            np.array(layers["self_attention"]["query"]["kernel"])[:, layer_idx],
            query_pre_attn_scalar,
        )
        q = q.reshape(self_attn_query.shape[0], -1).T
        hf_state[prefix + "self_attn.q_proj.weight"] = torch.tensor(
            q.astype(np.float32), dtype=torch.bfloat16
        )

        # --- k_proj.weight ---
        k = np.array(layers["self_attention"]["key"]["kernel"])[:, layer_idx]
        k = k.reshape(self_attn_query.shape[0], -1).T
        hf_state[prefix + "self_attn.k_proj.weight"] = torch.tensor(
            k.astype(np.float32), dtype=torch.bfloat16
        )

        # --- v_proj.weight ---
        v = np.array(layers["self_attention"]["value"]["kernel"])[:, layer_idx]
        v = v.reshape(self_attn_query.shape[0], -1).T
        hf_state[prefix + "self_attn.v_proj.weight"] = torch.tensor(
            v.astype(np.float32), dtype=torch.bfloat16
        )

        # --- o_proj.weight ---
        o = np.array(layers["self_attention"]["out"]["kernel"])[:, layer_idx]
        # Assume o has shape (num_heads, head_dim, hidden_size); transpose to (hidden_size, num_heads * head_dim)
        o = o.transpose((2, 0, 1)).reshape(self_attn_query.shape[0], -1)
        hf_state[prefix + "self_attn.o_proj.weight"] = torch.tensor(
            o.astype(np.float32), dtype=torch.bfloat16
        )

        # MLP Block Parameters
        # --- gate_proj.weight (from mlp.wi_0) ---
        gate = np.array(layers["mlp"]["wi_0"]["kernel"])[:, layer_idx].T
        hf_state[prefix + "mlp.gate_proj.weight"] = torch.tensor(
            gate.astype(np.float32), dtype=torch.bfloat16
        )
        # --- up_proj.weight (from mlp.wi_1) ---
        up = np.array(layers["mlp"]["wi_1"]["kernel"])[:, layer_idx].T
        hf_state[prefix + "mlp.up_proj.weight"] = torch.tensor(
            up.astype(np.float32), dtype=torch.bfloat16
        )
        # --- down_proj.weight (from mlp.wo) ---
        down = np.array(layers["mlp"]["wo"]["kernel"])[:, layer_idx].T
        hf_state[prefix + "mlp.down_proj.weight"] = torch.tensor(
            down.astype(np.float32), dtype=torch.bfloat16
        )

        # Norm Layers
        # In maxtext, norm scales are stored as (original_scale + 1). Subtract 1 for HF.
        q_norm = np.array(layers["self_attention"]["query_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "self_attn.q_norm.weight"] = torch.tensor(
            q_norm.astype(np.float32), dtype=torch.bfloat16
        )

        k_norm = np.array(layers["self_attention"]["key_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "self_attn.k_norm.weight"] = torch.tensor(
            k_norm.astype(np.float32), dtype=torch.bfloat16
        )

        inp_norm = np.array(layers["pre_self_attention_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "input_layernorm.weight"] = torch.tensor(
            inp_norm.astype(np.float32), dtype=torch.bfloat16
        )
        post_attn_norm = np.array(layers["post_self_attention_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "post_attention_layernorm.weight"] = torch.tensor(
            post_attn_norm.astype(np.float32), dtype=torch.bfloat16
        )
        pre_ffw_norm = np.array(layers["pre_ffw_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "pre_feedforward_layernorm.weight"] = torch.tensor(
            pre_ffw_norm.astype(np.float32), dtype=torch.bfloat16
        )
        post_ffw_norm = np.array(layers["post_ffw_norm"]["scale"])[:, layer_idx] - 1
        hf_state[prefix + "post_feedforward_layernorm.weight"] = torch.tensor(
            post_ffw_norm.astype(np.float32), dtype=torch.bfloat16
        )

    # --- LM Head ---
    # Typically the LM head is tied to the token embedding.
    hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"][:GEMMA_VOCAB_SIZE].clone()

    return hf_state

def main(argv: Sequence[str]):
    config = pyconfig.initialize(argv[:-2])
    hf_model_path = argv[-2].split("=")[1]
    model_size = argv[-1].split("=")[1]
    print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

    print("Loading Hugging Face model...")
    hf_model = load_hf_model(model_size)

    print("Checkpoint loaded; converting parameters...")
    hf_state_dict = convert_maxtext_to_hf(config, model_size)

    print("Conversion complete; saving Hugging Face checkpoint to", hf_model_path)
    hf_model.save_pretrained(hf_model_path, state_dict=hf_state_dict)
    print("Done.")

if __name__ == "__main__":
    app.run(main)
