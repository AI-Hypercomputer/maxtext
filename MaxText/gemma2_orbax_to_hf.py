#!/usr/bin/env python
"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

This script converts a maxtext Gemma2 checkpoint (in Flax/Orbax format)
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
             "self_attention_local": {
                "query": { "kernel": <np.ndarray> },   # shape (num_local, num_heads, hidden_size, head_dim)
                "key":   { "kernel": <np.ndarray> },   # shape (num_local, num_kv_heads, hidden_size, head_dim)
                "value": { "kernel": <np.ndarray> },   # shape (num_local, num_kv_heads, hidden_size, head_dim)
                "out":   { "kernel": <np.ndarray> }    # shape (num_local, num_heads, head_dim, hidden_size)
             },
             "self_attention_global": {
                "query": { "kernel": <np.ndarray> },   # shape (num_global, num_heads, hidden_size, head_dim)
                "key":   { "kernel": <np.ndarray> },   # shape (num_global, num_kv_heads, hidden_size, head_dim)
                "value": { "kernel": <np.ndarray> },   # shape (num_global, num_kv_heads, hidden_size, head_dim)
                "out":   { "kernel": <np.ndarray> }    # shape (num_global, num_heads, head_dim, hidden_size)
             },
             "mlp_local": {
                "wi_0": { "kernel": <np.ndarray> },    # gate_proj weight, shape (num_local, intermediate_dim, hidden_size)
                "wi_1": { "kernel": <np.ndarray> },    # up_proj weight, shape (num_local, intermediate_dim, hidden_size)
                "wo":   { "kernel": <np.ndarray> }     # down_proj weight, shape (num_local, hidden_size, intermediate_dim)
             },
             "mlp_global": {
                "wi_0": { "kernel": <np.ndarray> },
                "wi_1": { "kernel": <np.ndarray> },
                "wo":   { "kernel": <np.ndarray> }
             },
             "pre_self_attention_norm_local": { "scale": <np.ndarray> },   # shape (num_local, hidden_size) (stored as original+1)
             "post_self_attention_norm_local": { "scale": <np.ndarray> },
             "pre_ffw_norm_local": { "scale": <np.ndarray> },
             "post_ffw_norm_local": { "scale": <np.ndarray> },
             "pre_self_attention_norm_global": { "scale": <np.ndarray> },  # same for global parts
             "post_self_attention_norm_global": { "scale": <np.ndarray> },
             "pre_ffw_norm_global": { "scale": <np.ndarray> },
             "post_ffw_norm_global": { "scale": <np.ndarray> },
         }
      }
    }
  }

The resulting Hugging Face checkpoint will be a flat state dict whose keys follow
the HF Gemma2 model naming (for example, "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", etc.).

Usage:
    python convert_maxtext_to_hf.py \
         --maxtext_checkpoint /path/to/maxtext/checkpoint \
         --hf_output /path/to/save/hf_model.bin \
         --model_size 9b
"""

import math
import numpy as np
import torch
import jax
from absl import app
import pyconfig
import maxengine
from transformers import AutoConfig, Gemma2ForCausalLM
from generate_param_only_checkpoint import _read_train_checkpoint
from tqdm import tqdm
from typing import Sequence
jax.config.update("jax_platform_name", "cpu")

GEMMA_VOCAB_SIZE = 256000


def load_hf_model(model_size):
  """
  Load the model that we are interested in from HuggingFace

  """
  if model_size == "2b":
    config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
    model = Gemma2ForCausalLM(config)
  elif model_size == "9b":
    config = AutoConfig.from_pretrained("google/gemma-2-9b")
    model = Gemma2ForCausalLM(config)
  elif model_size == "27b":
    config = AutoConfig.from_pretrained("google/gemma-2-27b-it")
    model = Gemma2ForCausalLM(config)
  else:
    raise NotImplementedError

  return model


def load_maxtext_params(config):
    """
    Loads model parameters from the given checkpoint_path using load_model_state.
    This function temporarily sets config.load_parameters_path and config.load_full_state_path,
    calls load_model_state, and then returns the inner parameters (discarding optimizer state).
    """
    engine = maxengine.MaxEngine(config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load_params = jax.random.split(rng)
    params = engine.load_params(rng_load_params)

    return params['params']


def unpermute_from_match_maxtext_rope(arr):
  """
  Function to get the RoPE values in correct ordering
  """
  split_size = arr.shape[-1] // 2  # Assuming half for evens, half for odds
  evens = arr[..., :split_size]
  odds = arr[..., split_size:]
  return jax.numpy.stack([evens, odds], axis=len(arr.shape)).reshape(arr.shape)

def reverse_scale(arr,scale):
  """
  MaxText has the scaling factor included into the weights,
  we reverse it when writing out the HuggingFace checkpoint
  """
  return arr * np.sqrt(scale)


def convert_maxtext_to_hf(config, model_size):
    params = load_maxtext_params(config)
    hf_state = {}

    # --- Token embedding ---
    # In the maxtext checkpoint, the token embeddings were scaled by sqrt(embed_dim)
    embed = np.array(params["token_embedder"]["embedding"])[:GEMMA_VOCAB_SIZE]
    _, embed_dim = embed.shape
    hf_embed = embed / math.sqrt(embed_dim)
    hf_state["model.embed_tokens.weight"] = torch.tensor(
        np.array(hf_embed, dtype=np.float32), dtype=torch.bfloat16
    )

    # --- Final (decoder) norm ---
    # In maxtext, the final norm was stored as (original_scale + 1)
    final_norm = np.array(params["decoder"]["decoder_norm"]["scale"])
    hf_state["model.norm.weight"] = torch.tensor(
        np.array(final_norm - 1, dtype=np.float32), dtype=torch.bfloat16
    )

    # --- Layers conversion ---
    layers = params["decoder"]["layers"]

    # The maxtext checkpoint groups parameters into a local part (for even-indexed HF layers)
    # and a global part (for odd-indexed HF layers). We assume that the number of local layers equals
    # the length of the arrays under self_attention_local, and similarly for global.
    local_query = np.array(layers["self_attention_local"]["query"]["kernel"])  # shape: (num_local, num_heads, hidden_size, head_dim)
    num_local = local_query.shape[1]
    if model_size in ("2b", "9b"):
       query_pre_attn_scalar = local_query.shape[3]
    elif model_size in ("27b",):
       query_pre_attn_scalar = local_query.shape[0] // local_query.shape[2]
    global_query = np.array(layers["self_attention_global"]["query"]["kernel"])  # shape: (num_global, num_heads, hidden_size, head_dim)
    num_global = global_query.shape[1]
    total_layers = num_local + num_global

    # For each Hugging Face layer, use local parameters for even indices and global for odd indices.
    for layer_idx in tqdm(range(total_layers), desc='converting layers'):
        if layer_idx % 2 == 0:
            # EVEN: local layer
            i = layer_idx // 2
            prefix = f"model.layers.{layer_idx}."
            # Attention parameters from local branch:
            # q_proj.weight: from local query kernel; expected HF shape: (num_heads * head_dim, hidden_size)
            q = reverse_scale(np.array(layers["self_attention_local"]["query"]["kernel"])[:, i], query_pre_attn_scalar).reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.q_proj.weight"] = torch.tensor(
                np.array(q, dtype=np.float32), dtype=torch.bfloat16
            )

            # k_proj.weight: from local key kernel; expected shape: (num_kv_heads * head_dim, hidden_size)
            k = np.array(layers["self_attention_local"]["key"]["kernel"])[:, i]  # (num_kv_heads, hidden_size, head_dim)
            k = k.reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.k_proj.weight"] = torch.tensor(
                np.array(k, dtype=np.float32), dtype=torch.bfloat16
            )

            # v_proj.weight: from local value kernel; expected shape: (num_kv_heads * head_dim, hidden_size)
            v = np.array(layers["self_attention_local"]["value"]["kernel"])[:, i]
            v = v.reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.v_proj.weight"] = torch.tensor(
                np.array(v, dtype=np.float32), dtype=torch.bfloat16
            )

            # o_proj.weight: from local out kernel; expected HF shape: (hidden_size, num_heads * head_dim)
            o = np.array(layers["self_attention_local"]["out"]["kernel"])[:, i]  # (num_heads, head_dim, hidden_size)
            o = o.transpose((2, 0, 1)).reshape(local_query.shape[0], -1)
            hf_state[prefix + "self_attn.o_proj.weight"] = torch.tensor(
                np.array(o, dtype=np.float32), dtype=torch.bfloat16
            )

            # MLP block (local)
            # gate_proj.weight from wi_0; expected shape: (intermediate_dim, hidden_size)
            gate = np.array(layers["mlp_local"]["wi_0"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.gate_proj.weight"] = torch.tensor(
                np.array(gate, dtype=np.float32), dtype=torch.bfloat16
            )
            # up_proj.weight from wi_1; expected shape: (intermediate_dim, hidden_size)
            up = np.array(layers["mlp_local"]["wi_1"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.up_proj.weight"] = torch.tensor(
                np.array(up, dtype=np.float32), dtype=torch.bfloat16
            )
            # down_proj.weight from wo; expected shape: (hidden_size, intermediate_dim)
            down = np.array(layers["mlp_local"]["wo"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.down_proj.weight"] = torch.tensor(
                np.array(down, dtype=np.float32), dtype=torch.bfloat16
            )

            # Norm layers (local)
            # In maxtext the norm scales were stored as (original_scale + 1); subtract 1 for HF.
            inp_norm = np.array(layers["pre_self_attention_norm_local"]["scale"])[:, i] - 1
            hf_state[prefix + "input_layernorm.weight"] = torch.tensor(
                np.array(inp_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            post_attn_norm = np.array(layers["post_self_attention_norm_local"]["scale"])[:, i] - 1
            hf_state[prefix + "post_attention_layernorm.weight"] = torch.tensor(
                np.array(post_attn_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            pre_ffw_norm = np.array(layers["pre_ffw_norm_local"]["scale"])[:, i] - 1
            hf_state[prefix + "pre_feedforward_layernorm.weight"] = torch.tensor(
                np.array(pre_ffw_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            post_ffw_norm = np.array(layers["post_ffw_norm_local"]["scale"])[:, i] - 1
            hf_state[prefix + "post_feedforward_layernorm.weight"] = torch.tensor(
                np.array(post_ffw_norm, dtype=np.float32), dtype=torch.bfloat16
            )
        else:
            # ODD: global layer
            i = layer_idx // 2
            prefix = f"model.layers.{layer_idx}."
            q = reverse_scale(np.array(layers["self_attention_global"]["query"]["kernel"])[:, i], query_pre_attn_scalar).reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.q_proj.weight"] = torch.tensor(
                np.array(q, dtype=np.float32), dtype=torch.bfloat16
            )

            # k_proj.weight: from local key kernel; expected shape: (num_kv_heads * head_dim, hidden_size)
            k = np.array(layers["self_attention_global"]["key"]["kernel"])[:, i]  # (num_kv_heads, hidden_size, head_dim)
            k = k.reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.k_proj.weight"] = torch.tensor(
                np.array(k, dtype=np.float32), dtype=torch.bfloat16
            )

            # v_proj.weight: from local value kernel; expected shape: (num_kv_heads * head_dim, hidden_size)
            v = np.array(layers["self_attention_global"]["value"]["kernel"])[:, i]
            v = v.reshape(local_query.shape[0], -1).T
            hf_state[prefix + "self_attn.v_proj.weight"] = torch.tensor(
                np.array(v, dtype=np.float32), dtype=torch.bfloat16
            )

            # o_proj.weight: from local out kernel; expected HF shape: (hidden_size, num_heads * head_dim)
            o = np.array(layers["self_attention_global"]["out"]["kernel"])[:, i]  # (num_heads, head_dim, hidden_size)
            o = o.transpose((2, 0, 1)).reshape(local_query.shape[0], -1)
            hf_state[prefix + "self_attn.o_proj.weight"] = torch.tensor(
                np.array(o, dtype=np.float32), dtype=torch.bfloat16
            )

            # MLP block (local)
            # gate_proj.weight from wi_0; expected shape: (intermediate_dim, hidden_size)
            gate = np.array(layers["mlp_global"]["wi_0"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.gate_proj.weight"] = torch.tensor(
                np.array(gate, dtype=np.float32), dtype=torch.bfloat16
            )
            # up_proj.weight from wi_1; expected shape: (intermediate_dim, hidden_size)
            up = np.array(layers["mlp_global"]["wi_1"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.up_proj.weight"] = torch.tensor(
                np.array(up, dtype=np.float32), dtype=torch.bfloat16
            )
            # down_proj.weight from wo; expected shape: (hidden_size, intermediate_dim)
            down = np.array(layers["mlp_global"]["wo"]["kernel"])[:, i].T
            hf_state[prefix + "mlp.down_proj.weight"] = torch.tensor(
                np.array(down, dtype=np.float32), dtype=torch.bfloat16
            )

            # Norm layers (local)
            # In maxtext the norm scales were stored as (original_scale + 1); subtract 1 for HF.
            inp_norm = np.array(layers["pre_self_attention_norm_global"]["scale"])[:, i] - 1
            hf_state[prefix + "input_layernorm.weight"] = torch.tensor(
                np.array(inp_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            post_attn_norm = np.array(layers["post_self_attention_norm_global"]["scale"])[:, i] - 1
            hf_state[prefix + "post_attention_layernorm.weight"] = torch.tensor(
                np.array(post_attn_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            pre_ffw_norm = np.array(layers["pre_ffw_norm_global"]["scale"])[:, i] - 1
            hf_state[prefix + "pre_feedforward_layernorm.weight"] = torch.tensor(
                np.array(pre_ffw_norm, dtype=np.float32), dtype=torch.bfloat16
            )
            post_ffw_norm = np.array(layers["post_ffw_norm_global"]["scale"])[:, i] - 1
            hf_state[prefix + "post_feedforward_layernorm.weight"] = torch.tensor(
                np.array(post_ffw_norm, dtype=np.float32), dtype=torch.bfloat16
            )

    # --- LM head ---
    # Typically the LM head is tied to the token embedding.
    hf_state["lm_head.weight"] = hf_state["model.embed_tokens.weight"][:GEMMA_VOCAB_SIZE].clone()

    return hf_state


def main(argv: Sequence[str]):
    pyconfig.initialize(argv[:-2])

    hf_model_path = argv[-2].split("=")[1]
    model_size = argv[-1].split("=")[1]
    print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")

    config = pyconfig.config

    print("Load hf checkpoint")
    hf_model = load_hf_model(model_size)

    print("Checkpoint loaded; converting parameters...")
    hf_state_dict = convert_maxtext_to_hf(config, model_size)

    print("Conversion complete; saving Hugging Face checkpoint to", hf_model_path)
    hf_model.save_pretrained(hf_model_path, state_dict=hf_state_dict)
    print("Done.")


if __name__ == "__main__":
    app.run(main)

