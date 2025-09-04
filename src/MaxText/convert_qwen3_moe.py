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

r"""Convert weights from a Qwen3-MoE style model to a MaxText one.

This script rigorously follows the two-stage conversion process (map-then-transform)
required for generating a MaxText checkpoint compatible with scanned model layers.

Example cmd:

python3 -m MaxText.convert_qwen3_moe_ckpt --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt> --model_size qwen3-235b-a22b
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
from maxtext.src.maxtext.inference_utils import str2bool

# Static model parameters dictionary
MODEL_PARAMS_DICT = {
    "qwen3-235b-a22b": {
        "num_hidden_layers": 94,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "hidden_size": 4096,
        "head_dim": 128,
        "num_experts": 128,
        "moe_intermediate_size": 1536,
    }
}


def hf_to_maxtext_mapping(layer_idx: int, num_experts: int) -> dict:
  """Creates a mapping from HF weight names to MaxText weight names."""
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "model.norm.weight": "decoder.decoder_norm.scale",
      "lm_head.weight": "decoder.logits_dense.kernel",
  }
  # Layer-specific mappings for a pure MoE/scanned model
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
      f"model.layers.{layer_idx}.mlp.gate.weight": f"decoder.layers.{layer_idx}.moe_block.gate.kernel",
  })

  # MoE expert mappings
  for i in range(num_experts):
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight"] = (
        f"decoder.layers.{layer_idx}.moe_block.{i}.wi_0"
    )
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.{i}.wi_1"
    mapping[f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight"] = f"decoder.layers.{layer_idx}.moe_block.{i}.wo"

  return mapping


def convert_hf_to_maxtext(base_model_path: str, model_params: dict) -> dict:
  """Converts a Hugging Face Qwen3-MoE checkpoint to a MaxText compatible format."""
  num_layers = model_params["num_hidden_layers"]
  num_experts = model_params["num_experts"]
  hidden_size = model_params["hidden_size"]
  num_heads = model_params["num_attention_heads"]
  num_kv_heads = model_params["num_key_value_heads"]
  head_dim = model_params["head_dim"]
  moe_intermediate_size = model_params["moe_intermediate_size"]

  # Part 1: Load all weights from safetensors into a flat dictionary with MaxText names
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)}...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        if "layers" not in key and "embed_tokens" not in key and "norm" not in key and "lm_head" not in key:
          continue

        layer_idx_str = key.split(".")[2] if "layers" in key else "0"
        layer_idx = int(layer_idx_str) if layer_idx_str.isdigit() else 0

        maxtext_key = hf_to_maxtext_mapping(layer_idx, num_experts).get(key)
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
              "moe_block": {
                  "gate": {"kernel": None},
                  "wi_0": None,
                  "wi_1": None,
                  "wo": None,
              },
          },
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  max_logging.log("Populating non-layer weights...")
  maxtext_weights["token_embedder"]["embedding"] = chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy()
  maxtext_weights["decoder"]["decoder_norm"]["scale"] = chkpt_vars["decoder.decoder_norm.scale"].to(torch.float16).numpy()
  maxtext_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["decoder.logits_dense.kernel"].to(torch.float16).numpy().transpose()
  )

  max_logging.log("Allocating and stacking layer weights...")
  ln = maxtext_weights["decoder"]["layers"]
  s_attn = ln["self_attention"]
  moe = ln["moe_block"]

  # Pre-allocate stacked arrays with the 'layer' dimension first
  ln["pre_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=np.float16)
  ln["post_self_attention_layer_norm"]["scale"] = np.zeros((num_layers, hidden_size), dtype=np.float16)
  s_attn["query"]["kernel"] = np.zeros((num_layers, hidden_size, num_heads, head_dim), dtype=np.float16)
  s_attn["key"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
  s_attn["value"]["kernel"] = np.zeros((num_layers, hidden_size, num_kv_heads, head_dim), dtype=np.float16)
  s_attn["out"]["kernel"] = np.zeros((num_layers, num_heads, head_dim, hidden_size), dtype=np.float16)
  s_attn["query_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=np.float16)
  s_attn["key_norm"]["scale"] = np.zeros((num_layers, head_dim), dtype=np.float16)
  moe["gate"]["kernel"] = np.zeros((num_layers, hidden_size, num_experts), dtype=np.float16)
  moe["wi_0"] = np.zeros((num_experts, num_layers, hidden_size, moe_intermediate_size), dtype=np.float16)
  moe["wi_1"] = np.zeros((num_experts, num_layers, hidden_size, moe_intermediate_size), dtype=np.float16)
  moe["wo"] = np.zeros((num_experts, num_layers, moe_intermediate_size, hidden_size), dtype=np.float16)

  # Loop through layers and populate the stacked arrays
  # pylint: disable=unsupported-assignment-operation
  for l in tqdm(range(num_layers), desc="Stacking layer weights"):
    ln["pre_self_attention_layer_norm"]["scale"][l, :] = (
        chkpt_vars[f"decoder.layers.{l}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
    )
    ln["post_self_attention_layer_norm"]["scale"][l, :] = (
        chkpt_vars[f"decoder.layers.{l}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
    )

    s_attn["query"]["kernel"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.query.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
        .reshape(hidden_size, num_heads, head_dim)
    )
    s_attn["key"]["kernel"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.key.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
        .reshape(hidden_size, num_kv_heads, head_dim)
    )
    s_attn["value"]["kernel"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.value.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
        .reshape(hidden_size, num_kv_heads, head_dim)
    )
    s_attn["out"]["kernel"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.out.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
        .reshape(num_heads, head_dim, hidden_size)
    )

    s_attn["query_norm"]["scale"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.query_norm.scale"].to(torch.float16).numpy()
    )
    s_attn["key_norm"]["scale"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.self_attention.key_norm.scale"].to(torch.float16).numpy()
    )

    moe["gate"]["kernel"][l, ...] = (
        chkpt_vars[f"decoder.layers.{l}.moe_block.gate.kernel"].to(torch.float16).numpy().transpose()
    )
    for i in range(num_experts):
      moe["wi_0"][i, l, ...] = chkpt_vars[f"decoder.layers.{l}.moe_block.{i}.wi_0"].to(torch.float16).numpy().transpose()
      moe["wi_1"][i, l, ...] = chkpt_vars[f"decoder.layers.{l}.moe_block.{i}.wi_1"].to(torch.float16).numpy().transpose()
      moe["wo"][i, l, ...] = chkpt_vars[f"decoder.layers.{l}.moe_block.{i}.wo"].to(torch.float16).numpy().transpose()

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

  moe["gate"]["kernel"] = np.transpose(moe["gate"]["kernel"], axes=(1, 0, 2))

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
  max_logging.log(f"Starting conversion for Qwen3-MoE model size: {args.model_size}")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params)
  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path, jax_weights, args.simulated_cpu_devices_count, args.use_ocdbt, args.use_zarr3
  )
  max_logging.log("Checkpoint saved successfully.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Qwen3-MoE HF weights to MaxText.")
  parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3-MoE checkpoint files.")
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
