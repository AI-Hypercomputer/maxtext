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

r"""Convert weights from a Qwen3-Next style model to a MaxText one.

This script rigorously follows the two-stage conversion process (map-then-transform)
required for generating a MaxText checkpoint compatible with scanned model layers.

The Qwen3-Next model has heterogeneous layers, with full attention occurring
at a specified interval and linear attention otherwise.

Example cmd:

python3 -m convert_qwen3_next --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt>
"""

import argparse
import gc
import json
import os
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from MaxText.utils.ckpt_scripts import llama_or_mistral_ckpt
from MaxText import max_logging
from MaxText.inference_utils import str2bool

def load_hf_config(base_model_path: str) -> dict:
  """Loads the HF config.json file."""
  config_path = os.path.join(base_model_path, "config.json")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"config.json not found in {base_model_path}")
  with open(config_path, "r", encoding="utf-8") as f:
    return json.load(f)

def hf_to_maxtext_mapping(layer_idx: int, config: dict) -> dict:
  """Creates a mapping from HF weight names to MaxText weight names for a specific layer."""
  mapping = {
      "model.embed_tokens.weight": "params.params.token_embedder.embedding",
      "model.norm.weight": "params.params.decoder.decoder_norm.scale",
      "lm_head.weight": "params.params.decoder.logits_dense.kernel",
  }

  num_experts = config["num_experts"]
  full_attention_interval = config["full_attention_interval"]

  # Common layer components
  layer_prefix = f"model.layers.{layer_idx}"
  maxtext_layer_prefix = f"params.params.decoder.layers.{layer_idx}"

  mapping.update(
      {
          f"{layer_prefix}.input_layernorm.weight": f"{maxtext_layer_prefix}.input_layernorm.scale",
          f"{layer_prefix}.post_attention_layernorm.weight": f"{maxtext_layer_prefix}.post_attention_layernorm.scale",
      }
  )

  # MoE Mappings
  mapping.update({
      f"{layer_prefix}.mlp.shared_expert.down_proj.weight": f"{maxtext_layer_prefix}.mlp.shared_expert.wo.kernel",
      f"{layer_prefix}.mlp.shared_expert.gate_proj.weight": f"{maxtext_layer_prefix}.mlp.shared_expert.wi_0.kernel",
      f"{layer_prefix}.mlp.shared_expert.up_proj.weight": f"{maxtext_layer_prefix}.mlp.shared_expert.wi_1.kernel",
      f"{layer_prefix}.mlp.shared_expert_gate.weight": f"{maxtext_layer_prefix}.mlp.shared_expert_gate.kernel",
      f"{layer_prefix}.mlp.gate.weight": f"{maxtext_layer_prefix}.mlp.routed_experts.gate.kernel",
  })
  for i in range(num_experts):
    mapping[f"{layer_prefix}.mlp.experts.{i}.gate_proj.weight"] = f"{maxtext_layer_prefix}.mlp.routed_experts.{i}.wi_0.kernel"
    mapping[f"{layer_prefix}.mlp.experts.{i}.up_proj.weight"] = f"{maxtext_layer_prefix}.mlp.routed_experts.{i}.wi_1.kernel"
    mapping[f"{layer_prefix}.mlp.experts.{i}.down_proj.weight"] = f"{maxtext_layer_prefix}.mlp.routed_experts.{i}.wo.kernel"

  # Attention Mappings - Conditional
  is_full_attn = (layer_idx + 1) % full_attention_interval == 0

  if is_full_attn:
    mapping.update({
        f"{layer_prefix}.self_attn.q_proj.weight": f"{maxtext_layer_prefix}.attention.attention.query.kernel",
        f"{layer_prefix}.self_attn.k_proj.weight": f"{maxtext_layer_prefix}.attention.attention.key.kernel",
        f"{layer_prefix}.self_attn.v_proj.weight": f"{maxtext_layer_prefix}.attention.attention.value.kernel",
        f"{layer_prefix}.self_attn.o_proj.weight": f"{maxtext_layer_prefix}.attention.attention.out.kernel",
        f"{layer_prefix}.self_attn.q_norm.weight": f"{maxtext_layer_prefix}.attention.attention.query_norm.scale",
        f"{layer_prefix}.self_attn.k_norm.weight": f"{maxtext_layer_prefix}.attention.attention.key_norm.scale",
    })
  else: # Linear Attention
    mapping.update({
        f"{layer_prefix}.linear_attn.A_log": f"{maxtext_layer_prefix}.attention.A_log",
        f"{layer_prefix}.linear_attn.conv1d.weight": f"{maxtext_layer_prefix}.attention.conv1d.kernel",
        f"{layer_prefix}.linear_attn.dt_bias": f"{maxtext_layer_prefix}.attention.dt_bias",
        f"{layer_prefix}.linear_attn.in_proj_ba.weight": f"{maxtext_layer_prefix}.attention.in_proj_ba.kernel",
        f"{layer_prefix}.linear_attn.in_proj_qkvz.weight": f"{maxtext_layer_prefix}.attention.in_proj_qkvz.kernel",
        f"{layer_prefix}.linear_attn.norm.weight": f"{maxtext_layer_prefix}.attention.norm.rms_norm.scale",
        f"{layer_prefix}.linear_attn.out_proj.weight": f"{maxtext_layer_prefix}.attention.out_proj.kernel",
    })

  return mapping

def convert_hf_to_maxtext(base_model_path: str, config: dict) -> dict:
  """Converts a Hugging Face Qwen3-Next checkpoint to a MaxText compatible format."""
  num_layers = config["num_hidden_layers"]
  num_experts = config["num_experts"]
  hidden_size = config["hidden_size"]
  full_attention_interval = config["full_attention_interval"]

  # Dimensions for Full Attention
  num_heads = config["num_attention_heads"]
  num_kv_heads = config["num_key_value_heads"]
  head_dim = config["head_dim"]

  # MoE dimensions
  moe_intermediate_size = config["moe_intermediate_size"]

  # Part 1: Load all weights from safetensors into a flat dictionary with MaxText names
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  chkpt_vars = {}
  print(f"Found {len(ckpt_paths)} checkpoint files.")

  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)}: {ckpt_path.name}")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      layer_map_cache = {}
      for key in f.keys():
        if key.startswith("mtp."): # Skip MTP layers
          continue

        if not key.startswith("model.") and not key.startswith("lm_head."):
          continue

        layer_idx = -1
        if "layers" in key:
          try:
            layer_idx_str = key.split(".")[2]
            if layer_idx_str.isdigit():
              layer_idx = int(layer_idx_str)
          except IndexError:
            pass

        if layer_idx != -1:
          if layer_idx not in layer_map_cache:
            layer_map_cache[layer_idx] = hf_to_maxtext_mapping(layer_idx, config)
          current_map = layer_map_cache[layer_idx]
        else: # Non-layer weights
          current_map = hf_to_maxtext_mapping(0, config)

        maxtext_key = current_map.get(key)
        if maxtext_key:
          chkpt_vars[maxtext_key] = f.get_tensor(key)

  # Part 2: Initialize, populate, and transform the weights for MaxText
  maxtext_weights = {
      "params": {
          "params": {
              "decoder": {
                  "layers": {}, # This will be populated layer by layer
                  "decoder_norm": {"scale": None},
                  "logits_dense": {"kernel": None},
              },
              "token_embedder": {"embedding": None},
          }
      }
  }

  max_logging.log("Populating non-layer weights...")
  maxtext_weights["params"]["params"]["token_embedder"]["embedding"] = chkpt_vars["params.params.token_embedder.embedding"].to(torch.float32).numpy()
  maxtext_weights["params"]["params"]["decoder"]["decoder_norm"]["scale"] = chkpt_vars["params.params.decoder.decoder_norm.scale"].to(torch.float32).numpy()
  maxtext_weights["params"]["params"]["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["params.params.decoder.logits_dense.kernel"].to(torch.float32).numpy().transpose()
  )

  max_logging.log("Allocating and stacking layer weights...")

  # Pre-allocate lists to store layer weights
  layer_weights = [{} for _ in range(num_layers)]

  for l in tqdm(range(num_layers), desc="Processing layers"):
    is_full_attn = (l + 1) % full_attention_interval == 0
    layer_prefix = f"params.params.decoder.layers.{l}"
    l_weights = layer_weights[l]

    # Common
    l_weights["input_layernorm.scale"] = chkpt_vars[f"{layer_prefix}.input_layernorm.scale"].to(torch.float32).numpy()
    l_weights["post_attention_layernorm.scale"] = chkpt_vars[f"{layer_prefix}.post_attention_layernorm.scale"].to(torch.float32).numpy()

    # MoE
    l_weights["mlp.shared_expert.wo.kernel"] = chkpt_vars[f"{layer_prefix}.mlp.shared_expert.wo.kernel"].to(torch.float32).numpy().transpose()
    l_weights["mlp.shared_expert.wi_0.kernel"] = chkpt_vars[f"{layer_prefix}.mlp.shared_expert.wi_0.kernel"].to(torch.float32).numpy().transpose()
    l_weights["mlp.shared_expert.wi_1.kernel"] = chkpt_vars[f"{layer_prefix}.mlp.shared_expert.wi_1.kernel"].to(torch.float32).numpy().transpose()
    l_weights["mlp.shared_expert_gate.kernel"] = chkpt_vars[f"{layer_prefix}.mlp.shared_expert_gate.kernel"].to(torch.float32).numpy().transpose()
    l_weights["mlp.routed_experts.gate.kernel"] = chkpt_vars[f"{layer_prefix}.mlp.routed_experts.gate.kernel"].to(torch.float32).numpy().transpose()

    expert_wi_0 = np.zeros((num_experts, hidden_size, moe_intermediate_size), dtype=np.float32)
    expert_wi_1 = np.zeros((num_experts, hidden_size, moe_intermediate_size), dtype=np.float32)
    expert_wo = np.zeros((num_experts, moe_intermediate_size, hidden_size), dtype=np.float32)
    for i in range(num_experts):
      expert_wi_0[i, ...] = chkpt_vars[f"{layer_prefix}.mlp.routed_experts.{i}.wi_0.kernel"].to(torch.float32).numpy().transpose()
      expert_wi_1[i, ...] = chkpt_vars[f"{layer_prefix}.mlp.routed_experts.{i}.wi_1.kernel"].to(torch.float32).numpy().transpose()
      expert_wo[i, ...] = chkpt_vars[f"{layer_prefix}.mlp.routed_experts.{i}.wo.kernel"].to(torch.float32).numpy().transpose()
    l_weights["mlp.routed_experts.wi_0.kernel"] = expert_wi_0
    l_weights["mlp.routed_experts.wi_1.kernel"] = expert_wi_1
    l_weights["mlp.routed_experts.wo.kernel"] = expert_wo

    # Attention
    if is_full_attn:
      # Q PROJ - GATED
      q_proj_hf = chkpt_vars[f"{layer_prefix}.attention.attention.query.kernel"].to(torch.float32)
      q_proj_hf = q_proj_hf.transpose(0, 1) # -> (hidden_size, num_heads * head_dim * 2)
      q_proj_hf = q_proj_hf.reshape(hidden_size, num_heads, head_dim * 2) # -> (hidden_size, num_heads, head_dim * 2)
      query_up, query_gate = torch.chunk(q_proj_hf, 2, dim=-1) # Each is (hidden_size, num_heads, head_dim)
      l_weights["attention.attention.query.kernel"] = query_up.numpy()

      # K and V PROJ - Assuming not gated
      l_weights["attention.attention.key.kernel"] = chkpt_vars[f"{layer_prefix}.attention.attention.key.kernel"].to(torch.float32).numpy().transpose().reshape(hidden_size, num_kv_heads, head_dim)
      l_weights["attention.attention.value.kernel"] = chkpt_vars[f"{layer_prefix}.attention.attention.value.kernel"].to(torch.float32).numpy().transpose().reshape(hidden_size, num_kv_heads, head_dim)

      # OUT PROJ
      l_weights["attention.attention.out.kernel"] = chkpt_vars[f"{layer_prefix}.attention.attention.out.kernel"].to(torch.float32).numpy().transpose().reshape(num_heads, head_dim, hidden_size)
      # Norms
      l_weights["attention.attention.query_norm.scale"] = chkpt_vars[f"{layer_prefix}.attention.attention.query_norm.scale"].to(torch.float32).numpy()
      l_weights["attention.attention.key_norm.scale"] = chkpt_vars[f"{layer_prefix}.attention.attention.key_norm.scale"].to(torch.float32).numpy()
    else: # Linear
      l_weights["attention.A_log"] = chkpt_vars[f"{layer_prefix}.attention.A_log"].to(torch.float32).numpy()
      l_weights["attention.conv1d.kernel"] = chkpt_vars[f"{layer_prefix}.attention.conv1d.kernel"].to(torch.float32).numpy().transpose(2, 1, 0) # [C_K, 1, H * 2]
      l_weights["attention.dt_bias"] = chkpt_vars[f"{layer_prefix}.attention.dt_bias"].to(torch.float32).numpy()
      l_weights["attention.in_proj_ba.kernel"] = chkpt_vars[f"{layer_prefix}.attention.in_proj_ba.kernel"].to(torch.float32).numpy().transpose()
      l_weights["attention.in_proj_qkvz.kernel"] = chkpt_vars[f"{layer_prefix}.attention.in_proj_qkvz.kernel"].to(torch.float32).numpy().transpose()
      l_weights["attention.norm.rms_norm.scale"] = chkpt_vars[f"{layer_prefix}.attention.norm.rms_norm.scale"].to(torch.float32).numpy()
      l_weights["attention.out_proj.kernel"] = chkpt_vars[f"{layer_prefix}.attention.out_proj.kernel"].to(torch.float32).numpy().transpose()

  # Stack weights for MaxText scanned layers
  stacked_weights = {}

  def stack(key_path):
    data = [np.expand_dims(layer_weights[l][key_path], axis=0) for l in range(num_layers)]
    return np.concatenate(data, axis=0)

  def get_dtype_shape(key_path, layer_idx=0):
      return layer_weights[layer_idx][key_path].dtype, layer_weights[layer_idx][key_path].shape

  # Common
  stacked_weights["input_layernorm.scale"] = stack("input_layernorm.scale")
  stacked_weights["post_attention_layernorm.scale"] = stack("post_attention_layernorm.scale")

  # MoE
  stacked_weights["mlp.shared_expert.wo.kernel"] = stack("mlp.shared_expert.wo.kernel")
  stacked_weights["mlp.shared_expert.wi_0.kernel"] = stack("mlp.shared_expert.wi_0.kernel")
  stacked_weights["mlp.shared_expert.wi_1.kernel"] = stack("mlp.shared_expert.wi_1.kernel")
  stacked_weights["mlp.shared_expert_gate.kernel"] = stack("mlp.shared_expert_gate.kernel")
  stacked_weights["mlp.routed_experts.gate.kernel"] = stack("mlp.routed_experts.gate.kernel")
  # expert weights are already stacked as (E, ...) in layer_weights
  stacked_weights["mlp.routed_experts.wi_0.kernel"] = stack("mlp.routed_experts.wi_0.kernel")
  stacked_weights["mlp.routed_experts.wi_1.kernel"] = stack("mlp.routed_experts.wi_1.kernel")
  stacked_weights["mlp.routed_experts.wo.kernel"] = stack("mlp.routed_experts.wo.kernel")


  # Attention - Initialize with zeros, then fill
  full_attn_idx = full_attention_interval - 1
  f32_dtype = np.float32

  # Linear shapes (layer 0)
  _, shape = get_dtype_shape("attention.A_log", 0); stacked_weights["attention.A_log"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.conv1d.kernel", 0); stacked_weights["attention.conv1d.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.dt_bias", 0); stacked_weights["attention.dt_bias"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.in_proj_ba.kernel", 0); stacked_weights["attention.in_proj_ba.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.in_proj_qkvz.kernel", 0); stacked_weights["attention.in_proj_qkvz.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.norm.rms_norm.scale", 0); stacked_weights["attention.norm.rms_norm.scale"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.out_proj.kernel", 0); stacked_weights["attention.out_proj.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)

  # Full attn shapes (layer full_attn_idx)
  _, shape = get_dtype_shape("attention.attention.query.kernel", full_attn_idx); stacked_weights["attention.attention.query.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.attention.key.kernel", full_attn_idx); stacked_weights["attention.attention.key.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.attention.value.kernel", full_attn_idx); stacked_weights["attention.attention.value.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.attention.out.kernel", full_attn_idx); stacked_weights["attention.attention.out.kernel"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.attention.query_norm.scale", full_attn_idx); stacked_weights["attention.attention.query_norm.scale"] = np.zeros((num_layers, *shape), dtype=f32_dtype)
  _, shape = get_dtype_shape("attention.attention.key_norm.scale", full_attn_idx); stacked_weights["attention.attention.key_norm.scale"] = np.zeros((num_layers, *shape), dtype=f32_dtype)

  for l in range(num_layers):
    is_full_attn = (l + 1) % full_attention_interval == 0
    if is_full_attn:
      stacked_weights["attention.attention.query.kernel"][l] = layer_weights[l]["attention.attention.query.kernel"]
      stacked_weights["attention.attention.key.kernel"][l] = layer_weights[l]["attention.attention.key.kernel"]
      stacked_weights["attention.attention.value.kernel"][l] = layer_weights[l]["attention.attention.value.kernel"]
      stacked_weights["attention.attention.out.kernel"][l] = layer_weights[l]["attention.attention.out.kernel"]
      stacked_weights["attention.attention.query_norm.scale"][l] = layer_weights[l]["attention.attention.query_norm.scale"]
      stacked_weights["attention.attention.key_norm.scale"][l] = layer_weights[l]["attention.attention.key_norm.scale"]
    else: # Linear
      stacked_weights["attention.A_log"][l] = layer_weights[l]["attention.A_log"]
      stacked_weights["attention.conv1d.kernel"][l] = layer_weights[l]["attention.conv1d.kernel"]
      stacked_weights["attention.dt_bias"][l] = layer_weights[l]["attention.dt_bias"]
      stacked_weights["attention.in_proj_ba.kernel"][l] = layer_weights[l]["attention.in_proj_ba.kernel"]
      stacked_weights["attention.in_proj_qkvz.kernel"][l] = layer_weights[l]["attention.in_proj_qkvz.kernel"]
      stacked_weights["attention.norm.rms_norm.scale"][l] = layer_weights[l]["attention.norm.rms_norm.scale"]
      stacked_weights["attention.out_proj.kernel"][l] = layer_weights[l]["attention.out_proj.kernel"]

  # Final transformations for scanned weights (swap layer and feature axes)
  max_logging.log("Transposing layer weights for MaxText scanned format...")
  def transpose_for_scan(arr, key_name=""):
    print(f"Attempting to transpose key: {key_name}, shape: {arr.shape}, ndim: {arr.ndim}")
    if arr.ndim == 2:
        print(f"  Applying ndim=2 transpose to {key_name}")
        return np.transpose(arr, axes=(1, 0)) # (L, F) -> (F, L)

    if key_name.startswith("mlp.routed_experts"):
        if key_name == "mlp.routed_experts.gate.kernel":
            print(f"  Applying ndim=3 transpose to {key_name}")
            return np.transpose(arr, axes=(1, 2, 0)) # (L, H, N_E) -> (H, N_E, L)
        else: # wi_0, wi_1, wo
            print(f"  Matched mlp.routed_experts weights for {key_name}, shape: {arr.shape}, ndim: {arr.ndim}")
            if arr.ndim == 4:
                return np.transpose(arr, axes=(1, 2, 3, 0)) # (L, E, F1, F2) -> (E, F1, F2, L)
            else:
                raise ValueError(f"Shape mismatch for {key_name}: Expected 4D, got {arr.ndim} with shape {arr.shape}")
    elif key_name == "attention.attention.out.kernel":
        print(f"  Applying attention.attention.out.kernel transpose to {key_name}")
        return np.transpose(arr, axes=(3, 1, 2, 0)) # (L, NumHeads, HeadDim, Hidden) -> (Hidden, NumHeads, HeadDim, L)
    elif arr.ndim == 3:
        print(f"  Applying default ndim=3 transpose to {key_name}")
        return np.transpose(arr, axes=(1, 2, 0)) # (L, F1, F2) -> (F1, F2, L)
    elif arr.ndim == 4: # Catch-all for other 4D arrays
        print(f"  Applying default ndim=4 transpose to {key_name}")
        return np.transpose(arr, axes=(1, 2, 3, 0)) # (L, F1, F2, F3) -> (F1, F2, F3, L)
    else:
      print(f"  WARNING: No transpose rule matched for key {key_name} with shape {arr.shape}")
      return arr

  final_layers = {}
  for key, value in stacked_weights.items():
    keys = key.split('.')
    d = final_layers
    for k in keys[:-1]:
      if k not in d: d[k] = {}
      d = d[k]
    d[keys[-1]] = transpose_for_scan(value, key)

  maxtext_weights["params"]["params"]["decoder"]["layers"] = final_layers

  gc.collect()
  return maxtext_weights

def main(args):
  """Main function to run the conversion."""
  os.environ["JAX_PLATFORMS"] = "cpu"
  if args.simulated_cpu_devices_count > 0:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  hf_config = load_hf_config(args.base_model_path)
  max_logging.log(f"Starting conversion for Qwen3-Next model with config: {args.base_model_path}")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, hf_config)

  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      maxtext_model_path=args.maxtext_model_path, # <--- Corrected
      jax_weights=jax_weights,
      device_count=args.simulated_cpu_devices_count, # <--- Corrected
      use_ocdbt=args.use_ocdbt,
      use_zarr3=args.use_zarr3
  )
  max_logging.log("Checkpoint saved successfully.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Qwen3-Next HF weights to MaxText.")
  parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3-Next checkpoint files.")
  parser.add_argument(
      "--maxtext_model_path", type=str, required=True, help="Path to save the MaxText checkpoint (local or GCS)."
  )
  parser.add_argument(
      "--simulated_cpu_devices_count", type=int, default=16, help="Number of simulated CPU devices for saving. Set to 0 to disable."
  )
  parser.add_argument("--use-ocdbt", type=str2bool, default=True, help="Use OCDBT format for saving.")
  parser.add_argument("--use-zarr3", type=str2bool, default=True, help="Use Zarr3 format for saving.")

  parsed_args = parser.parse_args()
  main(parsed_args)