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

r"""Convert weights from a DeepSeek style model to a MaxText one.

Example cmd:

python3 -m MaxText.convert_deepseek_unscanned_ckpt --base_model_path <path/to/meta/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek2-16b
"""

# pylint: disable=line-too-long
# pylint: disable=unsupported-assignment-operation
# pytype: disable=unsupported-operands

import argparse
import pathlib
import os
import gc
import logging

import numpy as np
import torch
import psutil
from tqdm import tqdm

from MaxText import llama_or_mistral_ckpt
from MaxText import max_logging
from MaxText.inference_utils import str2bool
from safetensors import safe_open

MODEL_PARAMS_DICT = {
    "deepseek2-16b": {
        "num_layers": 27,
        "first_num_dense_layers": 1,
        "base_num_query_heads": 16,
        "base_emb_dim": 2048,
        "num_experts": 64,
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
    "deepseek3-671b": {
        "num_layers": 61,
        "first_num_dense_layers": 3,
        "base_num_query_heads": 128,
        "base_emb_dim": 7168,
        "num_experts": 256,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
}


# Only skip the MTP weights that are shared with the main model.
# The MTP block in MaxText will reuse the main embedding and output head.
MTP_KEYS_TO_SKIP = [
    # "model.layers.61.embed_tokens.weight",
    # "model.layers.61.shared_head.norm.weight",
    # "model.layers.61.shared_head.head.weight",
]


def is_key_allowed(key, banned_keys) -> bool:
  """
  Checks if a key is NOT in a list of banned keys.
  """
  return key not in banned_keys


def hf_to_maxtext_mapping(layer_idx, num_experts, first_num_dense_layers, num_main_layers) -> dict:
  """
  Generates a mapping between Hugging Face (HF) and MaxText model weight names.
  HF MLP is using self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)).


  Args:
      layer_idx: The index of the current layer.
      num_experts: The number of experts in MoE layers.
      first_num_dense_layers: The number of initial dense layers.
      num_main_layers: The total number of main layers in the model.

  Returns:
      A dictionary mapping HF weight names to MaxText weight names.
  """
  mapping = {
      "model.layers.61.embed_tokens.weight": "token_embedder.embedding",
      "model.layers.61.shared_head.head.weight": "logits_dense.kernel",
      "model.layers.61.shared_head.norm.weight": "decoder_norm.scale",
  }

  if layer_idx < first_num_dense_layers:
    # Dense layers mapping
    mapping.update(
        {
            f"model.layers.{layer_idx}.input_layernorm.weight": f"dense_layers.{layer_idx}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"dense_layers.{layer_idx}.post_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"dense_layers.{layer_idx}.self_attention.query.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"dense_layers.{layer_idx}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"dense_layers.{layer_idx}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.up_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_1.kernel",
            f"model.layers.{layer_idx}.mlp.down_proj.weight": f"dense_layers.{layer_idx}.mlp.wo.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.kv_norm.scale",
        }
    )
  elif layer_idx < num_main_layers:
    # MoE layers mapping
    moe_layer_idx_in_maxtext = layer_idx - first_num_dense_layers
    for expert_idx in range(num_experts):
      mapping.update(
          {
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wo",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_0",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_1",
          }
      )
    mapping.update(
        {
            f"model.layers.{layer_idx}.mlp.gate.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel",
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias",
            f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wo.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.kv_norm.scale",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.query.kernel",
            f"model.layers.{layer_idx}.input_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.post_self_attention_layer_norm.scale",
        }
    )
  elif layer_idx == num_main_layers:
    mapping.update(
        {
            f"model.layers.{layer_idx}.enorm.weight": "mtp_block.mtp_layer_1.mtp_1_embedding_norm.scale",
            f"model.layers.{layer_idx}.hnorm.weight": "mtp_block.mtp_layer_1.mtp_1_hidden_state_norm.scale",
            f"model.layers.{layer_idx}.eh_proj.weight": "mtp_block.mtp_layer_1.mtp_1_projection.kernel",
        }
    )
    for expert_idx in range(num_experts):
      mapping.update(
          {
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wo",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_0",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight": f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_1",
          }
      )
    mapping.update(
        {
            f"model.layers.{layer_idx}.mlp.gate.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel",
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias",
            f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wo.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_a.kernel",
            f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.q_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.kv_norm.scale",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.out.kernel",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.query.kernel",
            f"model.layers.{layer_idx}.input_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": "mtp_block.mtp_layer_1.mtp_1_transformer_layer.post_self_attention_layer_norm.scale",
        }
    )
  return mapping


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info) -> dict:
  """Convert Huggingface Checkpoint to Jax."""
  base_num_decoder_layers = model_params["num_layers"]
  first_num_dense_layers = model_params["first_num_dense_layers"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_emb_dim = model_params["base_emb_dim"]
  num_experts = model_params["num_experts"]
  q_lora_rank = model_params["q_lora_rank"]
  kv_lora_rank = model_params["kv_lora_rank"]
  qk_nope_head_dim = model_params["qk_nope_head_dim"]
  qk_rope_head_dim = model_params["qk_rope_head_dim"]
  v_head_dim = model_params["v_head_dim"]

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  # We now process the main model layers + 1 MTP layer
  num_total_layers_to_process = base_num_decoder_layers + 1

  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key and parts[2].isdigit() else 0

        if layer >= num_total_layers_to_process:
          continue

        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")

        # This will now correctly map keys for all layers including the MTP layer
        mapped_key = hf_to_maxtext_mapping(layer, num_experts, first_num_dense_layers, base_num_decoder_layers).get(key)
        if mapped_key:
          chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # # decoder norm scale ###########################################
  # max_logging.log("Processing decoder norm scale")
  # jax_weights["decoder"]["decoder_norm"]["scale"] = chkpt_vars["decoder_norm.scale"].to(torch.float16).numpy()
  # logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # # logits dense #################################################
  # max_logging.log("Processing logits dense")
  # jax_weights["decoder"]["logits_dense"]["kernel"] = chkpt_vars["logits_dense.kernel"].to(torch.float16).numpy().transpose()
  # logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # # token embedding ##############################################
  # max_logging.log("Processing token embeddings")
  # jax_weights["token_embedder"]["embedding"] = chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy()
  # logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layers = {
  #     "dense_layers": first_num_dense_layers,
  #     "moe_layers": base_num_decoder_layers - first_num_dense_layers,
  # }
  # # self attention and normalization ###############################################
  # max_logging.log("Processing self attention and normalization in dense layer")
  # for layer_key, layer_value in layers.items():
  #   for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
  #     layer_name = f"{layer_key}_{layer_idx}"
  #     if layer_key == "dense_layers":
  #       jax_weights["decoder"].update(
  #           {
  #               layer_name: {
  #                   "mlp": {
  #                       "wi_0": {"kernel": None},
  #                       "wi_1": {"kernel": None},
  #                       "wo": {"kernel": None},
  #                   },
  #                   "self_attention": {
  #                       "kv_norm": {"scale": None},
  #                       "wkv_a": {"kernel": None},
  #                       "wkv_b": {"kernel": None},
  #                       "out": {"kernel": None},
  #                   },
  #                   "pre_self_attention_layer_norm": {"scale": None},
  #                   "post_self_attention_layer_norm": {"scale": None},
  #               },
  #           }
  #       )
  #     else:
  #       jax_weights["decoder"].update(
  #           {
  #               layer_name: {
  #                   "DeepSeekMoeBlock_0": {
  #                       "MoeBlock_0": {
  #                           "wi_0": None,
  #                           "wi_1": None,
  #                           "wo": None,
  #                           "gate": {"kernel": None},
  #                       },
  #                       "shared_experts": {
  #                           "wi_0": {"kernel": None},
  #                           "wi_1": {"kernel": None},
  #                           "wo": {"kernel": None},
  #                       },
  #                   },
  #                   "self_attention": {
  #                       "kv_norm": {"scale": None},
  #                       "wkv_a": {"kernel": None},
  #                       "wkv_b": {"kernel": None},
  #                       "out": {"kernel": None},
  #                   },
  #                   "pre_self_attention_layer_norm": {"scale": None},
  #                   "post_self_attention_layer_norm": {"scale": None},
  #               },
  #           }
  #       )
  #     self_attention = jax_weights["decoder"][layer_name]["self_attention"]
  #     pre_self_attention_layer_norm = jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"]
  #     post_self_attention_layer_norm = jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"]

  #     pre_self_attention = (
  #         chkpt_vars[f"{layer_key}.{layer_idx}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  #     )
  #     post_self_attention = (
  #         chkpt_vars[f"{layer_key}.{layer_idx}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  #     )
  #     kv_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.kv_norm.scale"].to(torch.float16).numpy().transpose()
  #     wkv_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_a.kernel"].to(torch.float16).numpy().transpose()
  #     wkv_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_b.kernel"].to(torch.float16).numpy().transpose()
  #     out = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.out.kernel"].to(torch.float16).numpy().transpose()
  #     if q_lora_rank != 0:
  #       q_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.q_norm.scale"].to(torch.float16).numpy()
  #       wq_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_a.kernel"].to(torch.float16).numpy().transpose()
  #       wq_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_b.kernel"].to(torch.float16).numpy().transpose()
  #     else:
  #       query = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"].to(torch.float16).numpy().transpose()

  #     # reshape to match maxtext
  #     wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
  #     out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
  #     if q_lora_rank != 0:
  #       wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
  #     else:
  #       query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

  #     if q_lora_rank != 0:
  #       self_attention.update(
  #           {
  #               "q_norm": {"scale": None},
  #               "wq_a": {"kernel": None},
  #               "wq_b": {"kernel": None},
  #           }
  #       )
  #     else:
  #       self_attention.update({"query": {"kernel": None}})

  #     self_attention["kv_norm"]["scale"] = kv_norm
  #     self_attention["wkv_a"]["kernel"] = wkv_a
  #     self_attention["wkv_b"]["kernel"] = wkv_b
  #     self_attention["out"]["kernel"] = out
  #     pre_self_attention_layer_norm["scale"] = pre_self_attention
  #     post_self_attention_layer_norm["scale"] = post_self_attention
  #     if q_lora_rank != 0:
  #       self_attention["q_norm"]["scale"] = q_norm
  #       self_attention["wq_a"]["kernel"] = wq_a
  #       self_attention["wq_b"]["kernel"] = wq_b
  #     else:
  #       self_attention["query"]["kernel"] = query

  #     jax_weights["decoder"][layer_name]["self_attention"] = self_attention
  #     jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
  #     jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"] = post_self_attention_layer_norm

  # # layer weights ################################################
  # max_logging.log("Processing layer weights")
  # for layer_key, layer_value in layers.items():
  #   for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
  #     if layer_key == "dense_layers":
  #       layer_name = f"{layer_key}_{layer_idx}"
  #       mlp = jax_weights["decoder"][layer_name]["mlp"]
  #       wi_0 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"].to(torch.float16).numpy().transpose()
  #       wi_1 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"].to(torch.float16).numpy().transpose()
  #       wo = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"].to(torch.float16).numpy().transpose()
  #       mlp["wi_0"]["kernel"] = wi_0
  #       mlp["wi_1"]["kernel"] = wi_1
  #       mlp["wo"]["kernel"] = wo
  #       jax_weights["decoder"][layer_name]["mlp"] = mlp
  #     else:
  #       layer_name = f"{layer_key}_{layer_idx}"
  #       moe = jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]
  #       if q_lora_rank != 0:
  #         gate_bias = (
  #             chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"]
  #             .to(torch.float16)
  #             .numpy()
  #             .transpose()
  #         )
  #       gate = (
  #           chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"]
  #           .to(torch.float16)
  #           .numpy()
  #           .transpose()
  #       )
  #       shared_wi_0 = (
  #           chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"]
  #           .to(torch.float16)
  #           .numpy()
  #           .transpose()
  #       )
  #       shared_wi_1 = (
  #           chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"]
  #           .to(torch.float16)
  #           .numpy()
  #           .transpose()
  #       )
  #       shared_wo = (
  #           chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wo.kernel"]
  #           .to(torch.float16)
  #           .numpy()
  #           .transpose()
  #       )

  #       if q_lora_rank != 0:
  #         moe["MoeBlock_0"]["gate"]["bias"] = gate_bias
  #       moe["MoeBlock_0"]["gate"]["kernel"] = gate
  #       moe["shared_experts"]["wi_0"]["kernel"] = shared_wi_0
  #       moe["shared_experts"]["wi_1"]["kernel"] = shared_wi_1
  #       moe["shared_experts"]["wo"]["kernel"] = shared_wo

  #       for k in tqdm(range(num_experts), desc="experts", leave=False):
  #         wi_0 = (
  #             chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"]
  #             .to(torch.float16)
  #             .numpy()
  #             .transpose()
  #         )
  #         wi_1 = (
  #             chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"]
  #             .to(torch.float16)
  #             .numpy()
  #             .transpose()
  #         )
  #         wo = (
  #             chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"]
  #             .to(torch.float16)
  #             .numpy()
  #             .transpose()
  #         )

  #         if moe["MoeBlock_0"]["wi_0"] is None:
  #           stack_shape = (num_experts,)
  #           moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
  #           moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
  #           moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
  #         moe["MoeBlock_0"]["wi_0"][k, ...] = wi_0
  #         moe["MoeBlock_0"]["wi_1"][k, ...] = wi_1
  #         moe["MoeBlock_0"]["wo"][k, ...] = wo

  #       jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"] = moe

  # ### START OF CORRECTED MTP BLOCK ###
  max_logging.log("Processing MTP block")

  # Define prefixes for easier key access from chkpt_vars
  mtp_prefix = "mtp_block.mtp_layer_1"
  mtp_transformer_prefix = f"{mtp_prefix}.mtp_1_transformer_layer"
  sa_prefix = f"{mtp_transformer_prefix}.self_attention"
  mlp_prefix = f"{mtp_transformer_prefix}.DeepSeekMoeBlock_0"

  # Step 1: Initialize the entire MTP structure in the final weights dictionary
  jax_weights["mtp_block"] = {
      "mtp_layer_1": {
          "mtp_1_embedding_norm": {"scale": None},
          "mtp_1_hidden_state_norm": {"scale": None},
          "mtp_1_projection": {"kernel": None},
          "mtp_1_transformer_layer": {
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
              "self_attention": {
                  "kv_norm": {"scale": None},
                  "wkv_a": {"kernel": None},
                  "wkv_b": {"kernel": None},
                  "out": {"kernel": None},
              },
              "DeepSeekMoeBlock_0": {
                  "MoeBlock_0": {
                      "wi_0": None,
                      "wi_1": None,
                      "wo": None,
                      "gate": {"kernel": None},
                  },
                  "shared_experts": {
                      "wi_0": {"kernel": None},
                      "wi_1": {"kernel": None},
                      "wo": {"kernel": None},
                  },
              },
          },
      }
  }
  # Add conditional keys based on model config
  if q_lora_rank != 0:
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_transformer_layer"]["self_attention"].update(
        {
            "q_norm": {"scale": None},
            "wq_a": {"kernel": None},
            "wq_b": {"kernel": None},
        }
    )
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_transformer_layer"]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"][
        "bias"
    ] = None
  else:
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_transformer_layer"]["self_attention"].update({"query": {"kernel": None}})

  # Step 2: Get references to the nested dictionaries for easier population
  mtp_jax = jax_weights["mtp_block"]["mtp_layer_1"]
  mtp_transformer_jax = mtp_jax["mtp_1_transformer_layer"]
  self_attention = mtp_transformer_jax["self_attention"]
  moe_mtp = mtp_transformer_jax["DeepSeekMoeBlock_0"]

  # Step 3: Populate the structure with weights from chkpt_vars
  # Populate unique MTP weights
  mtp_jax["mtp_1_embedding_norm"]["scale"] = chkpt_vars[f"{mtp_prefix}.mtp_1_embedding_norm.scale"].to(torch.float16).numpy()
  mtp_jax["mtp_1_hidden_state_norm"]["scale"] = (
      chkpt_vars[f"{mtp_prefix}.mtp_1_hidden_state_norm.scale"].to(torch.float16).numpy()
  )
  mtp_jax["mtp_1_projection"]["kernel"] = (
      chkpt_vars[f"{mtp_prefix}.mtp_1_projection.kernel"].to(torch.float16).numpy().transpose()
  )

  # Populate transformer layer norms
  mtp_transformer_jax["pre_self_attention_layer_norm"]["scale"] = (
      chkpt_vars[f"{mtp_transformer_prefix}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  )
  mtp_transformer_jax["post_self_attention_layer_norm"]["scale"] = (
      chkpt_vars[f"{mtp_transformer_prefix}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  )

  # Populate Self-Attention Block
  kv_norm = chkpt_vars[f"{sa_prefix}.kv_norm.scale"].to(torch.float16).numpy().transpose()
  wkv_a = chkpt_vars[f"{sa_prefix}.wkv_a.kernel"].to(torch.float16).numpy().transpose()
  wkv_b = chkpt_vars[f"{sa_prefix}.wkv_b.kernel"].to(torch.float16).numpy().transpose()
  out = chkpt_vars[f"{sa_prefix}.out.kernel"].to(torch.float16).numpy().transpose()
  if q_lora_rank != 0:
    q_norm = chkpt_vars[f"{sa_prefix}.q_norm.scale"].to(torch.float16).numpy()
    wq_a = chkpt_vars[f"{sa_prefix}.wq_a.kernel"].to(torch.float16).numpy().transpose()
    wq_b = chkpt_vars[f"{sa_prefix}.wq_b.kernel"].to(torch.float16).numpy().transpose()
  else:
    query = chkpt_vars[f"{sa_prefix}.query.kernel"].to(torch.float16).numpy().transpose()

  wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
  out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
  if q_lora_rank != 0:
    wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
  else:
    query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

  self_attention["kv_norm"]["scale"] = kv_norm
  self_attention["wkv_a"]["kernel"] = wkv_a
  self_attention["wkv_b"]["kernel"] = wkv_b
  self_attention["out"]["kernel"] = out
  if q_lora_rank != 0:
    self_attention["q_norm"]["scale"] = q_norm
    self_attention["wq_a"]["kernel"] = wq_a
    self_attention["wq_b"]["kernel"] = wq_b
  else:
    self_attention["query"]["kernel"] = query

  # Populate MLP/MoE Block
  moe_block_mtp = moe_mtp["MoeBlock_0"]
  if q_lora_rank != 0:
    moe_block_mtp["gate"]["bias"] = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.gate.bias"].to(torch.float16).numpy().transpose()
  moe_block_mtp["gate"]["kernel"] = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.gate.kernel"].to(torch.float16).numpy().transpose()
  moe_mtp["shared_experts"]["wi_0"]["kernel"] = (
      chkpt_vars[f"{mlp_prefix}.shared_experts.wi_0.kernel"].to(torch.float16).numpy().transpose()
  )
  moe_mtp["shared_experts"]["wi_1"]["kernel"] = (
      chkpt_vars[f"{mlp_prefix}.shared_experts.wi_1.kernel"].to(torch.float16).numpy().transpose()
  )
  moe_mtp["shared_experts"]["wo"]["kernel"] = (
      chkpt_vars[f"{mlp_prefix}.shared_experts.wo.kernel"].to(torch.float16).numpy().transpose()
  )

  for k in tqdm(range(num_experts), desc="mtp experts", leave=False):
    wi_0 = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wi_0"].to(torch.float16).numpy().transpose()
    wi_1 = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wi_1"].to(torch.float16).numpy().transpose()
    wo = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wo"].to(torch.float16).numpy().transpose()

    if moe_block_mtp["wi_0"] is None:
      stack_shape = (num_experts,)
      moe_block_mtp["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
      moe_block_mtp["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
      moe_block_mtp["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
    moe_block_mtp["wi_0"][k, ...] = wi_0
    moe_block_mtp["wi_1"][k, ...] = wi_1
    moe_block_mtp["wo"][k, ...] = wo
  # ### END OF CORRECTED MTP BLOCK ###

  # # ### START OF MINIMAL ADDITION FOR MTP ###
  # max_logging.log("Processing MTP block")
  # # Initialize the MTP structure in the final weights
  # jax_weights["mtp_block"] = {"mtp_layer_1": {}}
  # mtp_jax = jax_weights["mtp_block"]["mtp_layer_1"]

  # # Define prefixes for easier key access from chkpt_vars
  # mtp_prefix = "mtp_block.mtp_layer_1"
  # mtp_transformer_prefix = f"{mtp_prefix}.mtp_1_transformer_layer"

  # # Process unique MTP weights
  # mtp_jax["mtp_1_embedding_norm"] = {"scale": chkpt_vars[f"{mtp_prefix}.mtp_1_embedding_norm.scale"].to(torch.float16).numpy()}
  # mtp_jax["mtp_1_hidden_state_norm"] = {"scale": chkpt_vars[f"{mtp_prefix}.mtp_1_hidden_state_norm.scale"].to(torch.float16).numpy()}
  # mtp_jax["mtp_1_projection"] = {"kernel": chkpt_vars[f"{mtp_prefix}.mtp_1_projection.kernel"].to(torch.float16).numpy().transpose()}

  # # Initialize the MTP transformer layer structure
  # mtp_transformer_jax = {
  #     "pre_self_attention_layer_norm": {"scale": chkpt_vars[f"{mtp_transformer_prefix}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()},
  #     "post_self_attention_layer_norm": {"scale": chkpt_vars[f"{mtp_transformer_prefix}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()},
  #     "self_attention": {},
  #     "DeepSeekMoeBlock_0": { "MoeBlock_0": {"gate": {}}, "shared_experts": {} }
  # }

  # self_attention = mtp_jax["mtp_1_transformer_layer"]["self_attention"]
  # pre_self_attention_layer_norm = mtp_jax["mtp_1_transformer_layer"]["pre_self_attention_layer_norm"]
  # post_self_attention_layer_norm = mtp_jax["mtp_1_transformer_layer"]["post_self_attention_layer_norm"]

  # sa_prefix = f"{mtp_transformer_prefix}.self_attention"
  # pre_self_attention = chkpt_vars[f"{mtp_transformer_prefix}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  # post_self_attention = chkpt_vars[f"{mtp_transformer_prefix}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
  # kv_norm = chkpt_vars[f"{sa_prefix}.kv_norm.scale"].to(torch.float16).numpy().transpose()
  # wkv_a = chkpt_vars[f"{sa_prefix}.wkv_a.kernel"].to(torch.float16).numpy().transpose()
  # wkv_b = chkpt_vars[f"{sa_prefix}.wkv_b.kernel"].to(torch.float16).numpy().transpose()
  # out = chkpt_vars[f"{sa_prefix}.out.kernel"].to(torch.float16).numpy().transpose()
  # if q_lora_rank != 0:
  #   q_norm = chkpt_vars[f"{sa_prefix}.q_norm.scale"].to(torch.float16).numpy()
  #   wq_a = chkpt_vars[f"{sa_prefix}.wq_a.kernel"].to(torch.float16).numpy().transpose()
  #   wq_b = chkpt_vars[f"{sa_prefix}.wq_b.kernel"].to(torch.float16).numpy().transpose()
  # else:
  #   query = chkpt_vars[f"{sa_prefix}.query.kernel"].to(torch.float16).numpy().transpose()

  # wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
  # out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
  # if q_lora_rank != 0:
  #   wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
  # else:
  #   query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

  # if q_lora_rank != 0:
  #   self_attention.update({"q_norm": {"scale": None}, "wq_a": {"kernel": None}, "wq_b": {"kernel": None}})
  # else:
  #   self_attention.update({"query": {"kernel": None}})

  # self_attention["kv_norm"]["scale"] = kv_norm
  # self_attention["wkv_a"]["kernel"] = wkv_a
  # self_attention["wkv_b"]["kernel"] = wkv_b
  # self_attention["out"]["kernel"] = out
  # pre_self_attention_layer_norm["scale"] = pre_self_attention
  # post_self_attention_layer_norm["scale"] = post_self_attention
  # if q_lora_rank != 0:
  #   self_attention["q_norm"]["scale"] = q_norm
  #   self_attention["wq_a"]["kernel"] = wq_a
  #   self_attention["wq_b"]["kernel"] = wq_b
  # else:
  #   self_attention["query"]["kernel"] = query

  # # MLP/MoE Block (1:1 logic from main loop)
  # moe_mtp = mtp_transformer_jax["DeepSeekMoeBlock_0"]
  # mlp_prefix = f"{mtp_transformer_prefix}.DeepSeekMoeBlock_0"
  # if q_lora_rank != 0:
  #     moe_mtp["MoeBlock_0"]["gate"]["bias"] = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.gate.bias"].to(torch.float16).numpy().transpose()
  # moe_mtp["MoeBlock_0"]["gate"]["kernel"] = chkpt_vars[f"{mlp_prefix}.MoeBlock_0.gate.kernel"].to(torch.float16).numpy().transpose()
  # moe_mtp["shared_experts"] = {
  #     "wi_0": {"kernel": chkpt_vars[f"{mlp_prefix}.shared_experts.wi_0.kernel"].to(torch.float16).numpy().transpose()},
  #     "wi_1": {"kernel": chkpt_vars[f"{mlp_prefix}.shared_experts.wi_1.kernel"].to(torch.float16).numpy().transpose()},
  #     "wo": {"kernel": chkpt_vars[f"{mlp_prefix}.shared_experts.wo.kernel"].to(torch.float16).numpy().transpose()}
  # }
  # moe_mtp["MoeBlock_0"]["wi_0"] = None
  # for k in tqdm(range(num_experts), desc="mtp experts", leave=False):
  #     wi_0 = (chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wi_0"].to(torch.float16).numpy().transpose())
  #     wi_1 = (chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wi_1"].to(torch.float16).numpy().transpose())
  #     wo = (chkpt_vars[f"{mlp_prefix}.MoeBlock_0.{k}.wo"].to(torch.float16).numpy().transpose())

  #     if moe_mtp["MoeBlock_0"]["wi_0"] is None:
  #       stack_shape = (num_experts,)
  #       moe_mtp["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
  #       moe_mtp["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
  #       moe_mtp["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
  #     moe_mtp["MoeBlock_0"]["wi_0"][k, ...] = wi_0
  #     moe_mtp["MoeBlock_0"]["wi_1"][k, ...] = wi_1
  #     moe_mtp["MoeBlock_0"]["wo"][k, ...] = wo

  # mtp_jax["mtp_1_transformer_layer"] = mtp_transformer_jax
  # ### END OF MINIMAL ADDITION ###

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def _convert_to_jax_weights(base_model_path, model_size, mem_info) -> dict:
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText.
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert and save DeepSeek model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--simulated_cpu_devices_count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  mem_info = psutil.Process()
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path,
      _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info),
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )


if __name__ == "__main__":
  main()
