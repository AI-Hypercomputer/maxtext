# Copyright 2023–2025 Google LLC
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

r"""Convert weights from a DeepSeek V4 style model to a MaxText one.

Example cmd:

python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_v4_ckpt \
    --base_model_path <path/to/meta/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek4-284b
"""

import argparse
import pathlib
import os
import gc
import logging
import absl
import numpy as np
import torch
import psutil
from tqdm import tqdm
from safetensors import safe_open

from maxtext.checkpoint_conversion.utils.utils import save_weights_to_checkpoint
from maxtext.utils import max_logging

absl.logging.set_verbosity(absl.logging.INFO)

MODEL_PARAMS_DICT = {
    "deepseek4-284b": {
        "num_pre_layers": 3,
        "layers_per_block": 20,  # The nn.scan iterates 20 times over a block of 2 layers
        "num_experts": 256,
        "base_emb_dim": 4096,
        "base_num_query_heads": 64,
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }
}


def _get_expert_stack(chkpt_vars, hf_prefix, num_experts, weight_name, dtype=torch.float16):
  """Helper to stack expert weights."""
  stack = []
  for k in range(num_experts):
    key = f"{hf_prefix}.ffn.experts.{k}.{weight_name}.weight"
    if key in chkpt_vars:
      stack.append(chkpt_vars[key].to(dtype).numpy().transpose())
  if stack:
    return np.stack(stack, axis=0)
  return None


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info) -> dict:
  """Convert Huggingface Checkpoint to Jax."""
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}

  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        chkpt_vars[key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
          "hc_head": {"hc_base": None, "hc_fn": None, "hc_scale": None},
          "pre_layers": {f"layers_{i}": {} for i in range(model_params["num_pre_layers"])},
          "scanned_blocks": {
              f"layers_{i}": {} for i in range(2)  # 2 layers in the scanned block: layers_0 (HCA) and layers_1 (CSA)
          },
      },
      "token_embedder": {"embedding": None},
  }

  # 1. Base components
  max_logging.log("Processing base embeddings and heads")
  if "embed.weight" in chkpt_vars:
    jax_weights["token_embedder"]["embedding"] = chkpt_vars["embed.weight"].to(torch.float16).numpy()
  if "norm.weight" in chkpt_vars:
    jax_weights["decoder"]["decoder_norm"]["scale"] = chkpt_vars["norm.weight"].to(torch.float16).numpy()
  if "head.weight" in chkpt_vars:
    jax_weights["decoder"]["logits_dense"]["kernel"] = chkpt_vars["head.weight"].to(torch.float16).numpy().transpose()

  # Output head hyper connections
  if "hc_head_base" in chkpt_vars:
    jax_weights["decoder"]["hc_head"]["hc_base"] = chkpt_vars["hc_head_base"].to(torch.float16).numpy()
    jax_weights["decoder"]["hc_head"]["hc_scale"] = chkpt_vars["hc_head_scale"].to(torch.float16).numpy()
    jax_weights["decoder"]["hc_head"]["hc_fn"] = chkpt_vars["hc_head_fn"].to(torch.float16).numpy().transpose()

  # 2. Pre-Layers (Layers 0, 1, 2)
  max_logging.log("Processing Pre-Layers")
  for layer_idx in range(model_params["num_pre_layers"]):
    pre_layer = jax_weights["decoder"]["pre_layers"][f"layers_{layer_idx}"]
    hf_prefix = f"layers.{layer_idx}"

    # Attention
    attn = {}
    if f"{hf_prefix}.attn.q_norm.weight" in chkpt_vars:
      attn["sinks"] = chkpt_vars[f"{hf_prefix}.attn.attn_sink"].to(torch.float16).numpy()
      attn["q_norm"] = {"scale": chkpt_vars[f"{hf_prefix}.attn.q_norm.weight"].to(torch.float16).numpy()}
      attn["kv_norm"] = {"scale": chkpt_vars[f"{hf_prefix}.attn.kv_norm.weight"].to(torch.float16).numpy()}
      attn["wq_a"] = {"kernel": chkpt_vars[f"{hf_prefix}.attn.wq_a.weight"].to(torch.float16).numpy().transpose()}

      # wq_b comes as [32768, 1024]. MaxText expects [1024, 64, 512]. 64 * 512 = 32768.
      wq_b_pt = chkpt_vars[f"{hf_prefix}.attn.wq_b.weight"].to(torch.float16).numpy().transpose()
      attn["wq_b"] = {"kernel": wq_b_pt.reshape(1024, 64, 512)}

      # wkv comes as [512, 4096]. MaxText expects [4096, 1, 512].
      wkv_pt = chkpt_vars[f"{hf_prefix}.attn.wkv.weight"].to(torch.float16).numpy().transpose()
      attn["wkv"] = {"kernel": wkv_pt.reshape(4096, 1, 512)}

      # wo_a is [8192, 4096]. MaxText o_a_proj is [8, 4096, 1024].
      wo_a_pt = chkpt_vars[f"{hf_prefix}.attn.wo_a.weight"].to(torch.float16).numpy().transpose()
      attn["o_a_proj"] = {"kernel": wo_a_pt.reshape(8, 4096, 1024)}

      attn["o_b_proj"] = {"kernel": chkpt_vars[f"{hf_prefix}.attn.wo_b.weight"].to(torch.float16).numpy().transpose()}

    pre_layer["self_attention"] = attn

    if f"{hf_prefix}.attn_norm.weight" in chkpt_vars:
      pre_layer["pre_self_attention_layer_norm"] = {
          "scale": chkpt_vars[f"{hf_prefix}.attn_norm.weight"].to(torch.float16).numpy()
      }
    if f"{hf_prefix}.ffn_norm.weight" in chkpt_vars:
      pre_layer["post_self_attention_layer_norm"] = {
          "scale": chkpt_vars[f"{hf_prefix}.ffn_norm.weight"].to(torch.float16).numpy()
      }

    # MoE
    moe = {"MoeBlock_0": {}, "shared_experts": {}}
    if f"{hf_prefix}.ffn.gate.weight" in chkpt_vars:
      moe["MoeBlock_0"]["gate"] = {
          "kernel": chkpt_vars[f"{hf_prefix}.ffn.gate.weight"].to(torch.float16).numpy().transpose()
      }
      if f"{hf_prefix}.ffn.gate.bias" in chkpt_vars:
        moe["MoeBlock_0"]["gate"]["bias"] = chkpt_vars[f"{hf_prefix}.ffn.gate.bias"].to(torch.float16).numpy()
      if f"{hf_prefix}.ffn.gate.tid2eid" in chkpt_vars:
        moe["MoeBlock_0"]["tid2eid"] = chkpt_vars[f"{hf_prefix}.ffn.gate.tid2eid"].to(torch.float16).numpy()

      w1 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w1")
      w2 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w2")
      w3 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w3")

      if w1 is not None:
        moe["MoeBlock_0"]["wi_0"] = w1
      if w2 is not None:
        moe["MoeBlock_0"]["wo"] = w2
      if w3 is not None:
        moe["MoeBlock_0"]["wi_1"] = w3

      if f"{hf_prefix}.ffn.shared_experts.w1.weight" in chkpt_vars:
        moe["shared_experts"]["wi_0"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w1.weight"].to(torch.float16).numpy().transpose()
        }
        moe["shared_experts"]["wo"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w2.weight"].to(torch.float16).numpy().transpose()
        }
        moe["shared_experts"]["wi_1"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w3.weight"].to(torch.float16).numpy().transpose()
        }

    pre_layer["mlp"] = moe

  # 3. Scanned Blocks (Layers 3 to N)
  max_logging.log("Processing Scanned Layers")
  layers_per_block = model_params["layers_per_block"]
  start_layer = model_params["num_pre_layers"]

  # In MaxText, the scan block runs 20 times (layers_per_block).
  # Each iteration executes layers_0 (HCA, ratio 128) then layers_1 (CSA, ratio 4).
  # This implicitly skips MTP nodes which are completely ignored by this scan.
  for block_idx in range(2):  # 0 for layers_0 (Odd HF layers), 1 for layers_1 (Even HF layers)
    # Accumulators for stacking along the scan dimension
    collected_layers = []

    for local_l in range(layers_per_block):
      # Interleave the layers! block_idx=0 gets +0, block_idx=1 gets +1
      hf_l = start_layer + 2 * local_l + block_idx
      hf_prefix = f"layers.{hf_l}"

      layer_dict = {"self_attention": {}, "mlp": {"MoeBlock_0": {}, "shared_experts": {}}}

      # --- Attention Base ---
      if f"{hf_prefix}.attn.q_norm.weight" in chkpt_vars:
        layer_dict["self_attention"]["sinks"] = chkpt_vars[f"{hf_prefix}.attn.attn_sink"].to(torch.float16).numpy()
        layer_dict["self_attention"]["q_norm"] = {
            "scale": chkpt_vars[f"{hf_prefix}.attn.q_norm.weight"].to(torch.float16).numpy()
        }
        layer_dict["self_attention"]["kv_norm"] = {
            "scale": chkpt_vars[f"{hf_prefix}.attn.kv_norm.weight"].to(torch.float16).numpy()
        }
        layer_dict["self_attention"]["wq_a"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.attn.wq_a.weight"].to(torch.float16).numpy().transpose()
        }

        wq_b_pt = chkpt_vars[f"{hf_prefix}.attn.wq_b.weight"].to(torch.float16).numpy().transpose()
        layer_dict["self_attention"]["wq_b"] = {"kernel": wq_b_pt.reshape(1024, 64, 512)}

        wkv_pt = chkpt_vars[f"{hf_prefix}.attn.wkv.weight"].to(torch.float16).numpy().transpose()
        layer_dict["self_attention"]["wkv"] = {"kernel": wkv_pt.reshape(4096, 1, 512)}

        wo_a_pt = chkpt_vars[f"{hf_prefix}.attn.wo_a.weight"].to(torch.float16).numpy().transpose()
        layer_dict["self_attention"]["o_a_proj"] = {"kernel": wo_a_pt.reshape(8, 4096, 1024)}

        layer_dict["self_attention"]["o_b_proj"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.attn.wo_b.weight"].to(torch.float16).numpy().transpose()
        }

        if f"{hf_prefix}.attn_norm.weight" in chkpt_vars:
          layer_dict["pre_self_attention_layer_norm"] = {
              "scale": chkpt_vars[f"{hf_prefix}.attn_norm.weight"].to(torch.float16).numpy()
          }
        if f"{hf_prefix}.ffn_norm.weight" in chkpt_vars:
          layer_dict["post_self_attention_layer_norm"] = {
              "scale": chkpt_vars[f"{hf_prefix}.ffn_norm.weight"].to(torch.float16).numpy()
          }

      # --- Compressors ---
      if block_idx == 0:
        # HCA Compressor (Odd Layers)
        if f"{hf_prefix}.attn.compressor.ape" in chkpt_vars:
          layer_dict["self_attention"]["hca_compressor"] = {
              "position_bias": chkpt_vars[f"{hf_prefix}.attn.compressor.ape"].to(torch.float16).numpy(),
              "kv_norm": {"scale": chkpt_vars[f"{hf_prefix}.attn.compressor.norm.weight"].to(torch.float16).numpy()},
              "gate_proj": {
                  "kernel": chkpt_vars[f"{hf_prefix}.attn.compressor.wgate.weight"].to(torch.float16).numpy().transpose()
              },
              "kv_proj": {
                  "kernel": chkpt_vars[f"{hf_prefix}.attn.compressor.wkv.weight"].to(torch.float16).numpy().transpose()
              },
          }
      else:
        # CSA Compressor + Indexer (Even Layers)
        if f"{hf_prefix}.attn.compressor.ape" in chkpt_vars:
          csa_comp = {
              "position_bias": chkpt_vars[f"{hf_prefix}.attn.compressor.ape"].to(torch.float16).numpy(),
              "kv_norm": {"scale": chkpt_vars[f"{hf_prefix}.attn.compressor.norm.weight"].to(torch.float16).numpy()},
              "gate_proj": {
                  "kernel": chkpt_vars[f"{hf_prefix}.attn.compressor.wgate.weight"].to(torch.float16).numpy().transpose()
              },
              "kv_proj": {
                  "kernel": chkpt_vars[f"{hf_prefix}.attn.compressor.wkv.weight"].to(torch.float16).numpy().transpose()
              },
          }
          if f"{hf_prefix}.attn.indexer.compressor.ape" in chkpt_vars:
            csa_comp["indexer"] = {
                "position_bias": chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.ape"].to(torch.float16).numpy(),
                "kv_norm": {
                    "scale": chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.norm.weight"].to(torch.float16).numpy()
                },
                "gate_proj": {
                    "kernel": chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.wgate.weight"]
                    .to(torch.float16)
                    .numpy()
                    .transpose()
                },
                "kv_proj": {
                    "kernel": chkpt_vars[f"{hf_prefix}.attn.indexer.compressor.wkv.weight"]
                    .to(torch.float16)
                    .numpy()
                    .transpose()
                },
                "weights_proj": {
                    "kernel": chkpt_vars[f"{hf_prefix}.attn.indexer.weights_proj.weight"]
                    .to(torch.float16)
                    .numpy()
                    .transpose()
                },
                "q_proj": {
                    "kernel": chkpt_vars[f"{hf_prefix}.attn.indexer.wq_b.weight"].to(torch.float16).numpy().transpose()
                },
            }
          layer_dict["self_attention"]["csa_compressor"] = csa_comp

      # --- MoE ---
      if f"{hf_prefix}.ffn.gate.weight" in chkpt_vars:
        layer_dict["mlp"]["MoeBlock_0"]["gate"] = {
            "kernel": chkpt_vars[f"{hf_prefix}.ffn.gate.weight"].to(torch.float16).numpy().transpose()
        }
        if f"{hf_prefix}.ffn.gate.bias" in chkpt_vars:
          layer_dict["mlp"]["MoeBlock_0"]["gate"]["bias"] = (
              chkpt_vars[f"{hf_prefix}.ffn.gate.bias"].to(torch.float16).numpy()
          )

        w1 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w1")
        w2 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w2")
        w3 = _get_expert_stack(chkpt_vars, hf_prefix, model_params["num_experts"], "w3")

        if w1 is not None:
          layer_dict["mlp"]["MoeBlock_0"]["wi_0"] = w1
        if w2 is not None:
          layer_dict["mlp"]["MoeBlock_0"]["wo"] = w2
        if w3 is not None:
          layer_dict["mlp"]["MoeBlock_0"]["wi_1"] = w3

        if f"{hf_prefix}.ffn.shared_experts.w1.weight" in chkpt_vars:
          layer_dict["mlp"]["shared_experts"]["wi_0"] = {
              "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w1.weight"].to(torch.float16).numpy().transpose()
          }
          layer_dict["mlp"]["shared_experts"]["wo"] = {
              "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w2.weight"].to(torch.float16).numpy().transpose()
          }
          layer_dict["mlp"]["shared_experts"]["wi_1"] = {
              "kernel": chkpt_vars[f"{hf_prefix}.ffn.shared_experts.w3.weight"].to(torch.float16).numpy().transpose()
          }

      collected_layers.append(layer_dict)

    # Helper to recursively stack lists of dicts
    def recursive_stack(list_of_dicts, path=""):
      if not list_of_dicts:
        return {}
      stacked = {}
      for k in list_of_dicts[0].keys():
        current_path = f"{path}/{k}" if path else k
        if isinstance(list_of_dicts[0][k], dict):
          stacked[k] = recursive_stack([d[k] for d in list_of_dicts if k in d], current_path)
        elif isinstance(list_of_dicts[0][k], np.ndarray):
          logging.info(f"Stacking array for block {block_idx}: {current_path} ...")
          # Follow the old script convention perfectly:
          # 1. Pre-allocate np.zeros with the layer dimension at axis 0
          sample_shape = list_of_dicts[0][k].shape
          stack_shape = (len(list_of_dicts),) + sample_shape
          arr = np.zeros(stack_shape, dtype=list_of_dicts[0][k].dtype)
          
          # 2. Assign layer by layer
          for layer_idx, d in enumerate(list_of_dicts):
            arr[layer_idx, ...] = d[k]
            
          # 3. Explicitly transpose axis 0 (layer) to axis 1
          # This identically matches np.transpose(..., axes=(1, 0, 2)) from the legacy scripts
          axes = list(range(len(stack_shape)))
          axes[0], axes[1] = axes[1], axes[0]
          stacked[k] = np.transpose(arr, axes=tuple(axes))
      return stacked

    if collected_layers:
      stacked_layer = recursive_stack(collected_layers)
      jax_weights["decoder"]["scanned_blocks"][f"layers_{block_idx}"] = stacked_layer

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def _convert_to_jax_weights(base_model_path, model_size, mem_info) -> dict:
  model_params = MODEL_PARAMS_DICT[model_size]
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert DeepSeek V4 model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--dry_run", action="store_true", help="Run without saving the checkpoint")
  args = parser.parse_args()

  mem_info = psutil.Process()
  jax_weights = _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info)

  if args.dry_run:
    max_logging.log("Dry run mode: weights mapped successfully in memory.")

    # Just to prove everything is there, let's print the shapes of layer 0
    def print_shapes(d, indent=""):
      for k, v in d.items():
        if isinstance(v, dict):
          print(f"{indent}{k}:")
          print_shapes(v, indent + "  ")
        elif isinstance(v, np.ndarray):
          print(f"{indent}{k}: {v.shape}")

    print("\n--- MaxText PyTree Shapes ---")
    print("decoder/pre_layers/layers_0:")
    if "layers_0" in jax_weights["decoder"]["pre_layers"]:
      print_shapes(jax_weights["decoder"]["pre_layers"]["layers_0"], "  ")
    print("\ndecoder/scanned_blocks/layers_1 (CSA):")
    if "layers_1" in jax_weights["decoder"]["scanned_blocks"]:
      print_shapes(jax_weights["decoder"]["scanned_blocks"]["layers_1"], "  ")
    return

  save_weights_to_checkpoint(args.maxtext_model_path, jax_weights, 16, True, True)


if __name__ == "__main__":
  main()
