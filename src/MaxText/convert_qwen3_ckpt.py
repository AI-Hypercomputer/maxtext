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

r"""Convert weights from a Qwen3 style model to a MaxText one.

This script rigorously follows the two-stage conversion process (map-then-transform)
required for generating a MaxText checkpoint compatible with scanned model layers.

Example cmd:

python3 -m MaxText.convert_qwen3_ckpt --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt> --model_size qwen3-8b
"""

import argparse
import gc
import os
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from MaxText import llama_or_mistral_ckpt, max_logging
from MaxText.inference_utils import str2bool

# Static model parameters dictionary
MODEL_PARAMS_DICT = {
    "qwen3-8b": {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
        "intermediate_size": 12288,
    }
}


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
  """Converts a Hugging Face Qwen3 checkpoint to a MaxText compatible format."""
  num_layers = model_params["num_hidden_layers"]
  hidden_size = model_params["hidden_size"]
  num_heads = model_params["num_attention_heads"]
  num_kv_heads = model_params["num_key_value_heads"]
  head_dim = model_params["head_dim"]
  intermediate_size = model_params["intermediate_size"]

  # Part 1: Load all weights from safetensors - keep original HF keys
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)}...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        chkpt_vars[key] = f.get_tensor(key)

  # Part 2: Initialize MaxText weight structure
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
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # Part 3: Process non-layer weights
  max_logging.log("Processing token embeddings")
  maxtext_weights["token_embedder"]["embedding"] = (
      chkpt_vars["model.embed_tokens.weight"].to(torch.float16).numpy()
  )

  max_logging.log("Processing decoder norm")
  maxtext_weights["decoder"]["decoder_norm"]["scale"] = (
      chkpt_vars["model.norm.weight"].to(torch.float16).numpy()
  )

  max_logging.log("Processing logits dense")
  maxtext_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["lm_head.weight"].to(torch.float16).numpy().transpose()
  )

  # Part 4: Process layer weights - using stacking approach
  max_logging.log("Processing self attention layers")
  s_attn = maxtext_weights["decoder"]["layers"]["self_attention"]
  ln = maxtext_weights["decoder"]["layers"]
  mlp = ln["mlp"]

  # Pre-allocate arrays with layer dimension first
  s_attn["query"]["kernel"] = np.zeros((num_layers, hidden_size, num_heads, head_dim), dtype=np.float16)
  s_attn["key"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
  s_attn["value"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
  s_attn["out"]["kernel"] = np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=np.float16)
  s_attn["query_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=np.float16)
  s_attn["key_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=np.float16)

  ln["pre_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=np.float16)
  ln["post_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=np.float16)

  mlp["wi_0"]["kernel"] = np.zeros((num_layers, hidden_size, intermediate_size), dtype=np.float16)
  mlp["wi_1"]["kernel"] = np.zeros((num_layers, hidden_size, intermediate_size), dtype=np.float16)
  mlp["wo"]["kernel"] = np.zeros((num_layers, intermediate_size, hidden_size), dtype=np.float16)

  # Fill in layer weights
  # pylint: disable=unsupported-assignment-operation
  for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
    # Attention projections - transpose and reshape
    wq = chkpt_vars[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].to(torch.float16).numpy().transpose()
    wk = chkpt_vars[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].to(torch.float16).numpy().transpose()
    wv = chkpt_vars[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].to(torch.float16).numpy().transpose()
    wo = chkpt_vars[f"model.layers.{layer_idx}.self_attn.o_proj.weight"].to(torch.float16).numpy()

    # Reshape: [hidden_size, num_heads * head_dim] -> [hidden_size, num_heads, head_dim]
    s_attn["query"]["kernel"][layer_idx, ...] = wq.reshape(hidden_size, num_heads, head_dim)
    s_attn["key"]["kernel"][layer_idx, ...] = wk.reshape(hidden_size, num_kv_heads, head_dim)
    s_attn["value"]["kernel"][layer_idx, ...] = wv.reshape(hidden_size, num_kv_heads, head_dim)

    # Output projection: [num_heads * head_dim, hidden_size] -> [num_heads, head_dim, hidden_size]
    s_attn["out"]["kernel"][layer_idx, ...] = wo.reshape(num_heads, head_dim, hidden_size)

    # Query and Key norms
    s_attn["query_norm"]["scale"][layer_idx, ...] = (
        chkpt_vars[f"model.layers.{layer_idx}.self_attn.q_norm.weight"].to(torch.float16).numpy()
    )
    s_attn["key_norm"]["scale"][layer_idx, ...] = (
        chkpt_vars[f"model.layers.{layer_idx}.self_attn.k_norm.weight"].to(torch.float16).numpy()
    )

    # Layer norms
    ln["pre_self_attention_layer_norm"]["scale"][layer_idx, :] = (
        chkpt_vars[f"model.layers.{layer_idx}.input_layernorm.weight"].to(torch.float16).numpy()
    )
    ln["post_self_attention_layer_norm"]["scale"][layer_idx, :] = (
        chkpt_vars[f"model.layers.{layer_idx}.post_attention_layernorm.weight"].to(torch.float16).numpy()
    )

    # MLP weights - transpose
    mlp["wi_0"]["kernel"][layer_idx, ...] = (
        chkpt_vars[f"model.layers.{layer_idx}.mlp.gate_proj.weight"].to(torch.float16).numpy().transpose()
    )
    mlp["wi_1"]["kernel"][layer_idx, ...] = (
        chkpt_vars[f"model.layers.{layer_idx}.mlp.up_proj.weight"].to(torch.float16).numpy().transpose()
    )
    mlp["wo"]["kernel"][layer_idx, ...] = (
        chkpt_vars[f"model.layers.{layer_idx}.mlp.down_proj.weight"].to(torch.float16).numpy().transpose()
    )

  # Part 5: Transpose for scanned format (swap layer and feature dimensions)
  max_logging.log("Transposing for MaxText scanned format...")

  # Attention kernels: [layers, hidden_size, heads, head_dim] -> [hidden_size, layers, heads, head_dim]
  s_attn["query"]["kernel"] = np.transpose(s_attn["query"]["kernel"], axes=(1, 0, 2, 3))
  s_attn["key"]["kernel"] = np.transpose(s_attn["key"]["kernel"], axes=(1, 0, 2, 3))
  s_attn["value"]["kernel"] = np.transpose(s_attn["value"]["kernel"], axes=(1, 0, 2, 3))

  # Output kernel: [layers, heads, head_dim, hidden_size] -> [heads, layers, head_dim, hidden_size]
  s_attn["out"]["kernel"] = np.transpose(s_attn["out"]["kernel"], axes=(1, 0, 2, 3))

  # Norms: [layers, dim] -> [dim, layers]
  s_attn["query_norm"]["scale"] = np.transpose(s_attn["query_norm"]["scale"], axes=(1, 0))
  s_attn["key_norm"]["scale"] = np.transpose(s_attn["key_norm"]["scale"], axes=(1, 0))
  ln["pre_self_attention_layer_norm"]["scale"] = np.transpose(ln["pre_self_attention_layer_norm"]["scale"], axes=(1, 0))
  ln["post_self_attention_layer_norm"]["scale"] = np.transpose(ln["post_self_attention_layer_norm"]["scale"], axes=(1, 0))

  # MLP kernels: [layers, dim1, dim2] -> [dim1, layers, dim2]
  mlp["wi_0"]["kernel"] = np.transpose(mlp["wi_0"]["kernel"], axes=(1, 0, 2))
  mlp["wi_1"]["kernel"] = np.transpose(mlp["wi_1"]["kernel"], axes=(1, 0, 2))
  mlp["wo"]["kernel"] = np.transpose(mlp["wo"]["kernel"], axes=(1, 0, 2))

  gc.collect()
  return maxtext_weights


def main(args):
  """Main function to run the conversion."""
  # Set up JAX simulated environment
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  if args.model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")

  model_params = MODEL_PARAMS_DICT[args.model_size]
  max_logging.log(f"Starting conversion for Qwen3 model size: {args.model_size}")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params)
  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path, jax_weights, args.simulated_cpu_devices_count, args.use_ocdbt, args.use_zarr3
  )
  max_logging.log("Checkpoint saved successfully.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Qwen3 HF weights to MaxText.")
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
