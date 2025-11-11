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

r"""Convert weights from a Qwen3-Next style model to an UNSCANNED MaxText one.

Example cmd:

python3 -m MaxText.utils.ckpt_scripts.convert_qwen3_next_unscanned --base_model_path . \
    --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt> --model_size qwen3-next-80b-a3b
"""

import argparse
import gc
import os
import pathlib
import numpy as np
import torch
import jax.numpy as jnp
from safetensors import safe_open
from tqdm import tqdm
import ml_dtypes # Required for jnp.bfloat16

from MaxText.utils.ckpt_scripts import llama_or_mistral_ckpt
from MaxText import max_logging
from MaxText.inference_utils import str2bool

# Static model parameters dictionary
MODEL_PARAMS_DICT = {
    "qwen3-next-80b-a3b": {
        "num_layers": 48,
        "num_q_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 256,
        "emb_dim": 2048,
        "vocab_size": 151936,
        "moe_intermediate_size": 512,  # base_moe_mlp_dim
        "num_experts": 512,
        "num_experts_per_tok": 10,
        # Qwen3-Next Specific Parameters for Linear Attention (Gated Delta Net)
        "inhomogeneous_layer_cycle_interval": 4,
        "gdn_conv_kernel_dim": 4,
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 128,
        "gdn_num_key_heads": 16,
        "gdn_num_value_heads": 32,
    }
}

def to_np_bfloat16(tensor):
  """Converts a torch tensor to a numpy array with bfloat16 dtype."""
  return tensor.to(torch.float32).numpy().astype(jnp.bfloat16)

def convert_hf_to_maxtext(base_model_path: str, model_params: dict, args) -> dict:
  """Converts a Hugging Face Qwen3-Next checkpoint to a MaxText compatible format."""
  num_layers = model_params["num_layers"]
  num_experts = model_params["num_experts"]
  emb_dim = model_params["emb_dim"]
  num_q_heads = model_params["num_q_heads"]
  num_kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["head_dim"]
  inhomogeneous_layer_cycle_interval = model_params["inhomogeneous_layer_cycle_interval"]
  moe_intermediate_size = model_params["moe_intermediate_size"]

  num_layers_to_convert = args.num_layers_to_convert if args.num_layers_to_convert > 0 else num_layers
  num_experts_to_convert = args.num_experts_to_convert if args.num_experts_to_convert > 0 else num_experts

  # Part 1: Load weights from safetensors
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("model-*-of-*.safetensors"))
  chkpt_vars = {}
  max_logging.log(f"Loading {len(ckpt_paths)} checkpoint files...")
  for i, ckpt_path in enumerate(tqdm(ckpt_paths, desc="Loading HF Checkpoints")):
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
          chkpt_vars[key] = f.get_tensor(key)
  gc.collect()
  max_logging.log("HF weights loaded.")

  # Part 2: Initialize, populate, and transform weights
  maxtext_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }
  for i in range(num_layers_to_convert):
      maxtext_weights["decoder"][f"layers_{i}"] = {}

  # Non-layer weights
  max_logging.log("Populating non-layer weights...")
  if "model.embed_tokens.weight" in chkpt_vars:
    # HF: [vocab_size, emb_dim] -> MaxText: [vocab_size, emb_dim] - No transformation
    maxtext_weights["token_embedder"]["embedding"] = to_np_bfloat16(chkpt_vars["model.embed_tokens.weight"])
  if "model.norm.weight" in chkpt_vars:
    # HF: [emb_dim] -> MaxText: [emb_dim] - No transformation
    maxtext_weights["decoder"]["decoder_norm"]["scale"] = to_np_bfloat16(chkpt_vars["model.norm.weight"])
  if "lm_head.weight" in chkpt_vars:
    # HF: [vocab_size, emb_dim] -> MaxText: [emb_dim, vocab_size] - Transpose
    maxtext_weights["decoder"]["logits_dense"]["kernel"] = to_np_bfloat16(chkpt_vars["lm_head.weight"]).transpose()

  for l in tqdm(range(num_layers_to_convert), desc="Processing Layers"):
    layer_key = f"layers_{l}"
    ln = maxtext_weights["decoder"][layer_key]
    # HF: [2048] -> MaxText: [2048,] - No transformation
    ln["input_layernorm"] = {"scale": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.input_layernorm.weight"])}
    # HF: [2048] -> MaxText: [2048,] - No transformation
    ln["post_attention_layernorm"] = {"scale": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.post_attention_layernorm.weight"])}

    ln["attention"] = {}
    attn_block = ln["attention"]
    is_full_attention_layer = (l + 1) % inhomogeneous_layer_cycle_interval == 0
    if is_full_attention_layer:
        attn_block["attention"] = {}
        attn_params = attn_block["attention"]
        # HF: [8192, 2048] -> Transpose [2048, 8192] -> Reshape to [2048, 16, 512]
        q_kernel = to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.q_proj.weight"]).transpose()
        attn_params["query"] = {"kernel": q_kernel.reshape(emb_dim, num_q_heads, head_dim * 2)}
        # HF: [512, 2048] -> Transpose [2048, 512] -> Reshape to [2048, 2, 256]
        k_kernel = to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.k_proj.weight"]).transpose()
        attn_params["key"] = {"kernel": k_kernel.reshape(emb_dim, num_kv_heads, head_dim)}
        # HF: [512, 2048] -> Transpose [2048, 512] -> Reshape to [2048, 2, 256]
        v_kernel = to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.v_proj.weight"]).transpose()
        attn_params["value"] = {"kernel": v_kernel.reshape(emb_dim, num_kv_heads, head_dim)}
        # HF: [2048, 4096] -> Transpose to [4096, 2048]
        attn_params["out"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.o_proj.weight"]).transpose()}
        # HF: [256] -> MaxText: [256,]
        attn_params["query_norm"] = {"scale": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.q_norm.weight"])}
        # HF: [256] -> MaxText: [256,]
        attn_params["key_norm"] = {"scale": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.self_attn.k_norm.weight"])}
    else: # Gated Delta Net
        # HF: [12288, 2048] -> Transpose to [2048, 12288]
        attn_block["in_proj_qkvz"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_qkvz.weight"]).transpose()}
        # HF: [64, 2048] -> Transpose to [2048, 64]
        attn_block["in_proj_ba"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_ba.weight"]).transpose()}
        # HF: [8192, 1, 4] -> Transpose(2, 1, 0) to [4, 1, 8192]
        attn_block["conv1d"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.conv1d.weight"]).transpose(2, 1, 0)}
        # HF: [32] -> MaxText: [32,]
        attn_block["A_log"] = to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.A_log"])
        # HF: [32] -> MaxText: [32,]
        attn_block["dt_bias"] = to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.dt_bias"])
        # HF: [128] -> MaxText: [128,]
        attn_block["norm"] = {"rms_norm": {"scale": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.norm.weight"])}}
        # HF: [2048, 4096] -> Transpose to [4096, 2048]
        attn_block["out_proj"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.linear_attn.out_proj.weight"]).transpose()}

    # MoE
    ln["mlp"] = {}
    mlp_block = ln["mlp"]
    mlp_block["routed_experts"] = {}
    # HF: [512, 2048] -> Transpose to [2048, 512]
    mlp_block["routed_experts"]["gate"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.gate.weight"]).transpose()}

    expert_wi_0, expert_wi_1, expert_wo = [], [], []
    for i in range(num_experts_to_convert):
        # HF: [512, 2048] -> Transpose to [2048, 512]
        expert_wi_0.append(to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.gate_proj.weight"]).transpose())
        # HF: [512, 2048] -> Transpose to [2048, 512]
        expert_wi_1.append(to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.up_proj.weight"]).transpose())
        # HF: [2048, 512] -> Transpose to [512, 2048]
        expert_wo.append(to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.down_proj.weight"]).transpose())

    # Stack Experts: [512, 2048, 512]
    mlp_block["routed_experts"]["wi_0"] = np.stack(expert_wi_0, axis=0)
    # Stack Experts: [512, 2048, 512]
    mlp_block["routed_experts"]["wi_1"] = np.stack(expert_wi_1, axis=0)
    # Stack Experts: [512, 512, 2048]
    mlp_block["routed_experts"]["wo"] = np.stack(expert_wo, axis=0)

    mlp_block["shared_expert"] = {}
    # HF: [512, 2048] -> Transpose to [2048, 512]
    mlp_block["shared_expert"]["wi_0"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.shared_expert.gate_proj.weight"]).transpose()}
    # HF: [512, 2048] -> Transpose to [2048, 512]
    mlp_block["shared_expert"]["wi_1"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.shared_expert.up_proj.weight"]).transpose()}
    # HF: [2048, 512] -> Transpose to [512, 2048]
    mlp_block["shared_expert"]["wo"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.shared_expert.down_proj.weight"]).transpose()}
    # HF: [1, 2048] -> Transpose to [2048, 1]
    mlp_block["shared_expert_gate"] = {"kernel": to_np_bfloat16(chkpt_vars[f"model.layers.{l}.mlp.shared_expert_gate.weight"]).transpose()}

  gc.collect()
  return maxtext_weights

def main(args):
  """Main function to run the conversion."""
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  if args.model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")

  model_params = MODEL_PARAMS_DICT[args.model_size]
  max_logging.log(f"Starting conversion for Qwen3-Next model size: {args.model_size} (UNSCANNED)")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params, args)
  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")

  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path, jax_weights, args.simulated_cpu_devices_count, args.use_ocdbt, args.use_zarr3
  )
  max_logging.log("Checkpoint saved successfully.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Qwen3-Next HF weights to MaxText UNSCANNED.")
  parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3-Next checkpoint files.")
  parser.add_argument("--maxtext_model_path", type=str, required=True, help="Path to save the MaxText checkpoint (local or GCS).")
  parser.add_argument("--model_size", type=str, required=True, choices=MODEL_PARAMS_DICT.keys(), help="The model size to convert.")

  # Dry run options
  parser.add_argument("--num_layers_to_convert", type=int, default=-1, help="Number of layers to convert for a dry run. -1 for all.")
  parser.add_argument("--num_experts_to_convert", type=int, default=-1, help="Number of experts to convert for a dry run. -1 for all.")

  # Saving options
  parser.add_argument("--simulated_cpu_devices_count", type=int, default=16, help="Number of simulated CPU devices for saving.")
  parser.add_argument("--use_ocdbt", type=str2bool, default=True, help="Use OCDBT format for saving.")
  parser.add_argument("--use_zarr3", type=str2bool, default=True, help="Use Zarr3 format for saving.")

  parsed_args = parser.parse_args()
  main(parsed_args)