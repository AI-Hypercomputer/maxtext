# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Convert weights from a Qwen3-Next style model to a MaxText one.
This script rigorously follows the two-stage conversion process (map-then-transform)
required for generating a MaxText checkpoint compatible with the model structure,
specifically for scanned heterogeneous layers.
Example cmd:
python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_qwen3_next_scanned \
  --base-model-path <path/to/hf/ckpt> \
  --maxtext_model_path gs://<gcs_bucket>/<path/to/save/ckpt> \
  --model_size qwen3-next-80b-a3b
"""

import argparse
import gc
import os
import pathlib
import ml_dtypes
import numpy as np
import torch
import jax.numpy as jnp
from safetensors import safe_open
from functools import partial
from tqdm import tqdm

from maxtext.checkpoint_conversion.standalone_scripts import llama_or_mistral_ckpt
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging

MODEL_PARAMS_DICT = {
    "qwen3-next-80b-a3b": {
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        # MoE Params
        "num_experts": 512,
        "moe_intermediate_size": 512,
        # Gated Attention (GA) params (layer_3)
        "head_dim": 256,
        "ga_num_q_heads": 16,
        "ga_num_kv_heads": 2,
        "ga_o_proj_input_dim": 4096,
        # Gated DeltaNet (GDN) params (layers_0, _1, _2)
        "gdn_num_value_heads": 32,
        "gdn_num_key_heads": 16,
        "inhomogeneous_layer_cycle_interval": 4,
        "gdn_conv_kernel_dim": 4,
        "gdn_key_head_dim": 128,
        "gdn_value_head_dim": 128,
        "gdn_a_log_dim": 32,
        "gdn_conv_features": 8192,
        "gdn_norm_dim": 128,
        "gdn_out_proj_input_dim": 4096,
    },
}

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def to_np_bfloat16(tensor):
  """Converts a torch tensor to a numpy array with bfloat16 dtype."""
  return tensor.to(torch.float32).numpy().astype(CAST_DTYPE)


def hf_to_maxtext_mapping(layer_idx: int, num_experts: int, inhomogeneous_layer_cycle_interval: int) -> dict:
  """Creates a mapping from HF weight names to MaxText weight names for a specific layer."""

  # 1. Define base prefixes to shorten line lengths
  block_idx = layer_idx % inhomogeneous_layer_cycle_interval
  hf_prefix = f"model.layers.{layer_idx}"
  mt_prefix = f"decoder.layers.layer_{block_idx}"
  mt_attn_prefix = f"{mt_prefix}.attention"
  mt_mlp_prefix = f"{mt_prefix}.mlp"

  # 2. Initialize mapping with global weights and standard layer norms
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "model.norm.weight": "decoder.decoder_norm.scale",
      "lm_head.weight": "decoder.logits_dense.kernel",
      f"{hf_prefix}.input_layernorm.weight": f"{mt_prefix}.input_layernorm.scale",
      f"{hf_prefix}.post_attention_layernorm.weight": f"{mt_prefix}.post_attention_layernorm.scale",
  }

  # 3. Handle Attention Logic (Full vs Linear)
  is_full_attention_layer = (layer_idx + 1) % inhomogeneous_layer_cycle_interval == 0

  if is_full_attention_layer:
    mapping.update(
        {
            f"{hf_prefix}.self_attn.q_proj.weight": f"{mt_attn_prefix}.attention.query.kernel",
            f"{hf_prefix}.self_attn.k_proj.weight": f"{mt_attn_prefix}.attention.key.kernel",
            f"{hf_prefix}.self_attn.v_proj.weight": f"{mt_attn_prefix}.attention.value.kernel",
            f"{hf_prefix}.self_attn.o_proj.weight": f"{mt_attn_prefix}.attention.out.kernel",
            f"{hf_prefix}.self_attn.q_norm.weight": f"{mt_attn_prefix}.attention.query_norm.scale",
            f"{hf_prefix}.self_attn.k_norm.weight": f"{mt_attn_prefix}.attention.key_norm.scale",
        }
    )
  else:
    mapping.update(
        {
            f"{hf_prefix}.linear_attn.in_proj_qkvz.weight": f"{mt_attn_prefix}.in_proj_qkvz.kernel",
            f"{hf_prefix}.linear_attn.in_proj_ba.weight": f"{mt_attn_prefix}.in_proj_ba.kernel",
            f"{hf_prefix}.linear_attn.conv1d.weight": f"{mt_attn_prefix}.conv1d.kernel",
            f"{hf_prefix}.linear_attn.A_log": f"{mt_attn_prefix}.A_log",
            f"{hf_prefix}.linear_attn.dt_bias": f"{mt_attn_prefix}.dt_bias",
            f"{hf_prefix}.linear_attn.norm.weight": f"{mt_attn_prefix}.norm.rms_norm.scale",
            f"{hf_prefix}.linear_attn.out_proj.weight": f"{mt_attn_prefix}.out_proj.kernel",
        }
    )

  # 4. Handle MLP (Gates and Shared Experts)
  mapping.update(
      {
          f"{hf_prefix}.mlp.gate.weight": f"{mt_mlp_prefix}.routed_experts.gate.kernel",
          f"{hf_prefix}.mlp.shared_expert.gate_proj.weight": f"{mt_mlp_prefix}.shared_expert.wi_0.kernel",
          f"{hf_prefix}.mlp.shared_expert.up_proj.weight": f"{mt_mlp_prefix}.shared_expert.wi_1.kernel",
          f"{hf_prefix}.mlp.shared_expert.down_proj.weight": f"{mt_mlp_prefix}.shared_expert.wo.kernel",
          f"{hf_prefix}.mlp.shared_expert_gate.weight": f"{mt_mlp_prefix}.shared_expert_gate.kernel",
      }
  )

  # 5. Handle Routed Experts Loop
  for i in range(num_experts):
    # Note: Ensure these don't require '.kernel' suffix (common in Flax, but absent in your original code)
    mapping[f"{hf_prefix}.mlp.experts.{i}.gate_proj.weight"] = f"{mt_mlp_prefix}.routed_experts.{i}.wi_0"
    mapping[f"{hf_prefix}.mlp.experts.{i}.up_proj.weight"] = f"{mt_mlp_prefix}.routed_experts.{i}.wi_1"
    mapping[f"{hf_prefix}.mlp.experts.{i}.down_proj.weight"] = f"{mt_mlp_prefix}.routed_experts.{i}.wo"

  return mapping


def init_maxtext_weights(model_params, num_layers_to_convert, num_experts_to_convert):
  """Initializes an empty pytree for the hf weights to be loaded in"""
  emb_dim = model_params["emb_dim"]
  num_q_heads = model_params["num_q_heads"]
  num_kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["head_dim"]
  moe_intermediate_size = model_params["moe_intermediate_size"]
  # num_experts = model_params["num_experts"]
  cycle = model_params["inhomogeneous_layer_cycle_interval"]
  num_stacked_layers = num_layers_to_convert // cycle

  gdn_num_v_heads = model_params["gdn_num_value_heads"]
  gdn_key_dim = model_params["gdn_num_key_heads"] * model_params["gdn_key_head_dim"]
  gdn_value_dim = gdn_num_v_heads * model_params["gdn_value_head_dim"]
  gdn_conv_dim = gdn_key_dim * 2 + gdn_value_dim
  gdn_conv_kernel_dim = model_params["gdn_conv_kernel_dim"]

  weights = {
      "decoder": {
          "layers": {},
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  for i in range(cycle):
    layer_key = f"layer_{i}"
    layer_struct = {
        "input_layernorm": {"scale": np.zeros((emb_dim, num_stacked_layers), dtype=jnp.bfloat16)},
        "post_attention_layernorm": {"scale": np.zeros((emb_dim, num_stacked_layers), dtype=jnp.bfloat16)},
        "attention": {},
        "mlp": {
            "routed_experts": {
                "gate": {"kernel": np.zeros((emb_dim, num_stacked_layers, num_experts_to_convert), dtype=jnp.bfloat16)},
                "wi_0": np.zeros(
                    (num_experts_to_convert, num_stacked_layers, emb_dim, moe_intermediate_size), dtype=jnp.bfloat16
                ),
                "wi_1": np.zeros(
                    (num_experts_to_convert, num_stacked_layers, emb_dim, moe_intermediate_size), dtype=jnp.bfloat16
                ),
                "wo": np.zeros(
                    (num_experts_to_convert, num_stacked_layers, moe_intermediate_size, emb_dim), dtype=jnp.bfloat16
                ),
            },
            "shared_expert": {
                "wi_0": {"kernel": np.zeros((emb_dim, num_stacked_layers, moe_intermediate_size), dtype=jnp.bfloat16)},
                "wi_1": {"kernel": np.zeros((emb_dim, num_stacked_layers, moe_intermediate_size), dtype=jnp.bfloat16)},
                "wo": {"kernel": np.zeros((moe_intermediate_size, num_stacked_layers, emb_dim), dtype=jnp.bfloat16)},
            },
            "shared_expert_gate": {"kernel": np.zeros((emb_dim, num_stacked_layers, 1), dtype=jnp.bfloat16)},
        },
    }

    is_full_attention_layer = (i + 1) % cycle == 0
    if is_full_attention_layer:
      layer_struct["attention"] = {
          "attention": {
              "query": {"kernel": np.zeros((emb_dim, num_stacked_layers, num_q_heads, head_dim * 2), dtype=jnp.bfloat16)},
              "key": {"kernel": np.zeros((emb_dim, num_stacked_layers, num_kv_heads, head_dim), dtype=jnp.bfloat16)},
              "value": {"kernel": np.zeros((emb_dim, num_stacked_layers, num_kv_heads, head_dim), dtype=jnp.bfloat16)},
              "out": {"kernel": np.zeros((num_q_heads * head_dim, num_stacked_layers, emb_dim), dtype=jnp.bfloat16)},
              "query_norm": {"scale": np.zeros((head_dim, num_stacked_layers), dtype=jnp.bfloat16)},
              "key_norm": {"scale": np.zeros((head_dim, num_stacked_layers), dtype=jnp.bfloat16)},
          }
      }
    else:
      layer_struct["attention"] = {
          "in_proj_qkvz": {
              "kernel": np.zeros((emb_dim, num_stacked_layers, gdn_key_dim * 2 + gdn_value_dim * 2), dtype=jnp.bfloat16)
          },
          "in_proj_ba": {"kernel": np.zeros((emb_dim, num_stacked_layers, gdn_num_v_heads * 2), dtype=jnp.bfloat16)},
          "conv1d": {"kernel": np.zeros((gdn_conv_kernel_dim, num_stacked_layers, 1, gdn_conv_dim), dtype=jnp.bfloat16)},
          "A_log": np.zeros((gdn_num_v_heads, num_stacked_layers), dtype=jnp.bfloat16),
          "dt_bias": np.zeros((gdn_num_v_heads, num_stacked_layers), dtype=jnp.bfloat16),
          "norm": {
              "rms_norm": {
                  "scale": np.zeros((model_params["gdn_value_head_dim"], num_stacked_layers), dtype=jnp.bfloat16)
              }
          },
          "out_proj": {"kernel": np.zeros((gdn_value_dim, num_stacked_layers, emb_dim), dtype=jnp.bfloat16)},
      }
    weights["decoder"]["layers"][layer_key] = layer_struct
  return weights


def _get_hf_tensor(maxtext_key_suffix, hf_map, l, chkpt_vars):
  for hf_key, mt_key in hf_map.items():
    if mt_key.endswith(maxtext_key_suffix):
      if hf_key in chkpt_vars:
        return chkpt_vars[hf_key]
      else:
        raise ValueError(f"HF Key {hf_key} not found in chkpt_vars for MaxText suffix: {maxtext_key_suffix} in layer {l}")
  raise ValueError(f"Could not find HF key for MaxText suffix: {maxtext_key_suffix} in layer {l}")


def convert_hf_to_maxtext(base_model_path: str, model_params: dict, args) -> dict:
  """Converts a Hugging Face Qwen3-Next checkpoint to a MaxText compatible format."""
  num_layers = model_params["num_layers"]
  num_experts = model_params["num_experts"]
  emb_dim = model_params["emb_dim"]
  num_q_heads = model_params["num_q_heads"]
  num_kv_heads = model_params["num_kv_heads"]
  head_dim = model_params["head_dim"]
  inhomogeneous_layer_cycle_interval = model_params["inhomogeneous_layer_cycle_interval"]
  cycle = inhomogeneous_layer_cycle_interval

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
  maxtext_weights = init_maxtext_weights(model_params, num_layers_to_convert, num_experts_to_convert)

  # Non-layer weights
  max_logging.log("Populating non-layer weights...")
  if "model.embed_tokens.weight" in chkpt_vars:
    # HF: [vocab_size, emb_dim] -> MaxText: [vocab_size, emb_dim]
    maxtext_weights["token_embedder"]["embedding"] = to_np_bfloat16(chkpt_vars["model.embed_tokens.weight"])
  if "model.norm.weight" in chkpt_vars:
    # HF: [emb_dim] -> MaxText: [emb_dim]
    maxtext_weights["decoder"]["decoder_norm"]["scale"] = to_np_bfloat16(chkpt_vars["model.norm.weight"])
  if "lm_head.weight" in chkpt_vars:
    # HF: [vocab_size, emb_dim] -> MaxText: [emb_dim, vocab_size] (Transposed)
    maxtext_weights["decoder"]["logits_dense"]["kernel"] = to_np_bfloat16(chkpt_vars["lm_head.weight"]).transpose()

  max_logging.log(f"Populating layer weights for {num_layers_to_convert} layers...")

  for l in tqdm(range(num_layers_to_convert), desc="Processing Layers"):
    block_idx = l % cycle
    stack_idx = l // cycle
    layer_key = f"layer_{block_idx}"
    hf_map = hf_to_maxtext_mapping(l, num_experts, cycle)

    get_hf_tensor = partial(_get_hf_tensor, hf_map=hf_map, l=l, chkpt_vars=chkpt_vars)

    ln = maxtext_weights["decoder"]["layers"][layer_key]

    # Layernorms
    # HF: [emb_dim] -> slice of MaxText: [emb_dim, num_stacked_layers]
    ln["input_layernorm"]["scale"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".input_layernorm.scale"))
    # HF: [emb_dim] -> slice of MaxText: [emb_dim, num_stacked_layers]
    ln["post_attention_layernorm"]["scale"][:, stack_idx] = to_np_bfloat16(
        get_hf_tensor(".post_attention_layernorm.scale")
    )

    attn_block = ln["attention"]
    is_full_attention_layer = (l + 1) % cycle == 0
    if is_full_attention_layer:
      attn_params = attn_block["attention"]
      # HF: [8192, 2048] -> Transpose [2048, 8192] -> Reshape ->
      #   slice of MaxText: [emb_dim, num_stacked_layers, num_q_heads, head_dim * 2]
      q_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.query.kernel")).transpose()
      attn_params["query"]["kernel"][:, stack_idx, :, :] = q_kernel.reshape(emb_dim, num_q_heads, head_dim * 2)
      # HF: [512, 2048] -> Transpose [2048, 512] -> Reshape ->
      #   slice of MaxText: [emb_dim, num_stacked_layers, num_kv_heads, head_dim]
      k_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.key.kernel")).transpose()
      attn_params["key"]["kernel"][:, stack_idx, :, :] = k_kernel.reshape(emb_dim, num_kv_heads, head_dim)
      # HF: [512, 2048] -> Transpose [2048, 512] -> Reshape ->
      #   slice of MaxText: [emb_dim, num_stacked_layers, num_kv_heads, head_dim]
      v_kernel = to_np_bfloat16(get_hf_tensor(".attention.attention.value.kernel")).transpose()
      attn_params["value"]["kernel"][:, stack_idx, :, :] = v_kernel.reshape(emb_dim, num_kv_heads, head_dim)
      # HF: [2048, 4096] -> Transpose -> slice of MaxText: [num_q_heads * head_dim, num_stacked_layers, emb_dim]
      attn_params["out"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.attention.out.kernel")
      ).transpose()
      # HF: [256] -> slice of MaxText: [head_dim, num_stacked_layers]
      attn_params["query_norm"]["scale"][:, stack_idx] = to_np_bfloat16(
          get_hf_tensor(".attention.attention.query_norm.scale")
      )
      # HF: [256] -> slice of MaxText: [head_dim, num_stacked_layers]
      attn_params["key_norm"]["scale"][:, stack_idx] = to_np_bfloat16(
          get_hf_tensor(".attention.attention.key_norm.scale")
      )
    else:  # Gated Delta Net
      # HF: [12288, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 12288]
      attn_block["in_proj_qkvz"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.in_proj_qkvz.kernel")
      ).transpose()
      # HF: [64, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 64]
      attn_block["in_proj_ba"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.in_proj_ba.kernel")
      ).transpose()
      # HF: [8192, 1, 4] -> Transpose(2,1,0) -> slice of MaxText: [gdn_conv_kernel_dim, num_stacked_layers, 1, gdn_conv_dim]
      conv1d_kernel = to_np_bfloat16(get_hf_tensor(".attention.conv1d.kernel"))
      attn_block["conv1d"]["kernel"][:, stack_idx, :, :] = conv1d_kernel.transpose(2, 1, 0)
      # HF: [32] -> slice of MaxText: [32, num_stacked_layers]
      attn_block["A_log"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".attention.A_log"))
      # HF: [32] -> slice of MaxText: [32, num_stacked_layers]
      attn_block["dt_bias"][:, stack_idx] = to_np_bfloat16(get_hf_tensor(".attention.dt_bias"))
      # HF: [128] -> slice of MaxText: [128, num_stacked_layers]
      attn_block["norm"]["rms_norm"]["scale"][:, stack_idx] = to_np_bfloat16(
          get_hf_tensor(".attention.norm.rms_norm.scale")
      )
      # HF: [2048, 4096] -> Transpose -> slice of MaxText: [4096, num_stacked_layers, 2048]
      attn_block["out_proj"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
          get_hf_tensor(".attention.out_proj.kernel")
      ).transpose()

    # MoE
    mlp_block = ln["mlp"]
    # HF: [512, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 512]
    mlp_block["routed_experts"]["gate"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.routed_experts.gate.kernel")
    ).transpose()
    # HF: [512, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 512]
    mlp_block["shared_expert"]["wi_0"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.shared_expert.wi_0.kernel")
    ).transpose()
    # HF: [512, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 512]
    mlp_block["shared_expert"]["wi_1"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.shared_expert.wi_1.kernel")
    ).transpose()
    # HF: [2048, 512] -> Transpose -> slice of MaxText: [512, num_stacked_layers, 2048]
    mlp_block["shared_expert"]["wo"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.shared_expert.wo.kernel")
    ).transpose()
    # HF: [1, 2048] -> Transpose -> slice of MaxText: [emb_dim, num_stacked_layers, 1]
    mlp_block["shared_expert_gate"]["kernel"][:, stack_idx, :] = to_np_bfloat16(
        get_hf_tensor(".mlp.shared_expert_gate.kernel")
    ).transpose()

    for i in range(num_experts_to_convert):
      # HF: [512, 2048] -> Transpose -> slice of MaxText: [num_experts, num_stacked_layers, emb_dim, moe_intermediate_size]
      mlp_block["routed_experts"]["wi_0"][i, stack_idx, :, :] = to_np_bfloat16(
          get_hf_tensor(f".mlp.routed_experts.{i}.wi_0")
      ).transpose()
      # HF: [512, 2048] -> Transpose -> slice of MaxText: [num_experts, num_stacked_layers, emb_dim, moe_intermediate_size]
      mlp_block["routed_experts"]["wi_1"][i, stack_idx, :, :] = to_np_bfloat16(
          get_hf_tensor(f".mlp.routed_experts.{i}.wi_1")
      ).transpose()
      # HF: [2048, 512] -> Transpose -> slice of MaxText: [num_experts, num_stacked_layers, moe_intermediate_size, emb_dim]
      mlp_block["routed_experts"]["wo"][i, stack_idx, :, :] = to_np_bfloat16(
          get_hf_tensor(f".mlp.routed_experts.{i}.wo")
      ).transpose()
  gc.collect()
  return maxtext_weights


def main(args):
  """Main function to run the conversion."""
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"

  if args.model_size not in MODEL_PARAMS_DICT:
    raise ValueError(f"Model size '{args.model_size}' not found in MODEL_PARAMS_DICT.")

  model_params = MODEL_PARAMS_DICT[args.model_size]
  max_logging.log(f"Starting conversion for Qwen3-Next model size: {args.model_size}")
  jax_weights = convert_hf_to_maxtext(args.base_model_path, model_params, args)
  max_logging.log(f"Conversion complete. Saving MaxText checkpoint to {args.maxtext_model_path}")

  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path, jax_weights, args.simulated_cpu_devices_count, args.use_ocdbt, args.use_zarr3
  )
  max_logging.log("Checkpoint saved successfully.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert Qwen3-Next HF weights to MaxText.")
  parser.add_argument("--base_model_path", type=str, required=True, help="Path to the HF Qwen3-Next checkpoint files.")
  parser.add_argument(
      "--maxtext_model_path", type=str, required=True, help="Path to save the MaxText checkpoint (local or GCS)."
  )
  parser.add_argument(
      "--model_size", type=str, required=True, choices=MODEL_PARAMS_DICT.keys(), help="The model size to convert."
  )

  # Saving options
  parser.add_argument(
      "--simulated_cpu_devices_count", type=int, default=16, help="Number of simulated CPU devices for saving."
  )
  parser.add_argument("--use_ocdbt", type=str2bool, default=True, help="Use OCDBT format for saving.")
  parser.add_argument("--use_zarr3", type=str2bool, default=True, help="Use Zarr3 format for saving.")

  parsed_args = parser.parse_args()
  main(parsed_args)
