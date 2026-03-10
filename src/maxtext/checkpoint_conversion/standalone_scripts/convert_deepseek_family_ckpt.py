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

r"""Convert weights from a DeepSeek style model to a MaxText one.

This script now includes support for converting Multi-Token Prediction (MTP) weights.

Example cmd:

python3 -m maxtext.checkpoint_conversion.standalone_scripts.convert_deepseek_family_ckpt \
    --base_model_path <path/to/meta/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek3-671b --enable_mtp
"""

# pylint: disable=line-too-long
# pylint: disable=unsupported-assignment-operation
# pytype: disable=unsupported-operands

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

from maxtext.checkpoint_conversion.standalone_scripts import llama_or_mistral_ckpt
from maxtext.inference.inference_utils import str2bool
from maxtext.utils import max_logging

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


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
        "has_mtp": False,
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
        "has_mtp": True,
    },
    "kimi-k2-1t": {
        "num_layers": 61,
        "first_num_dense_layers": 1,
        "base_num_query_heads": 64,
        "base_emb_dim": 7168,
        "num_experts": 384,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "has_mtp": False,
    },
}


# Only skip the MTP weights that are shared with the main model.
# The MTP block in MaxText will reuse the main embedding and output head.
MTP_KEYS_TO_SKIP = [
    "model.layers.61.embed_tokens.weight",
    "model.layers.61.shared_head.norm.weight",
    "model.layers.61.shared_head.head.weight",
]


def is_key_allowed(key, banned_keys) -> bool:
  """
  Checks if a key is NOT in a list of banned keys.
  """
  return key not in banned_keys


def hf_to_maxtext_mapping(layer_idx, num_experts, first_num_dense_layers, num_main_layers, has_mtp=False) -> dict:
  """
  Generates a mapping between Hugging Face (HF) and MaxText model weight names.
  HF MLP is using self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)).


  Args:
      layer_idx: The index of the current layer.
      num_experts: The number of experts in MoE layers.
      first_num_dense_layers: The number of initial dense layers.
      num_main_layers: The total number of main layers in the model.
      has_mtp: Whether the model has an MTP head.

  Returns:
      A dictionary mapping HF weight names to MaxText weight names.
  """
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "lm_head.weight": "logits_dense.kernel",
      "model.norm.weight": "decoder_norm.scale",
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
  elif has_mtp and layer_idx == num_main_layers:
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


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info, enable_mtp=False) -> dict:
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
  has_mtp = model_params.get("has_mtp", False) and enable_mtp

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key else 0
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if is_key_allowed(key, MTP_KEYS_TO_SKIP):
          mapped_key = hf_to_maxtext_mapping(
              layer, num_experts, first_num_dense_layers, base_num_decoder_layers, has_mtp
          ).get(key)
          if mapped_key:
            chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "decoder": {
          "dense_layers": {
              "mlp": {
                  "wi_0": {"kernel": None},
                  "wi_1": {"kernel": None},
                  "wo": {"kernel": None},
              },
              "self_attention": {
                  "kv_norm": {"scale": None},
                  "wkv_a": {"kernel": None},
                  "wkv_b": {"kernel": None},
                  "out": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
          "moe_layers": {
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
              "self_attention": {
                  "kv_norm": {"scale": None},
                  "wkv_a": {"kernel": None},
                  "wkv_b": {"kernel": None},
                  "out": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  jax_weights["decoder"]["decoder_norm"]["scale"] = chkpt_vars["decoder_norm.scale"].to(torch.float16).numpy()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")
  jax_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["logits_dense.kernel"].to(torch.float16).numpy().transpose()
  )
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")
  jax_weights["token_embedder"]["embedding"] = chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  layers = {
      "dense_layers": first_num_dense_layers,
      "moe_layers": base_num_decoder_layers - first_num_dense_layers,
  }
  # self attention and normalization ###############################################
  max_logging.log("Processing self attention and normalization in dense layer")
  for layer_key, layer_value in layers.items():
    self_attention = jax_weights["decoder"][f"{layer_key}"]["self_attention"]
    pre_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"]
    post_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"]

    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
      pre_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      post_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      kv_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.kv_norm.scale"].to(torch.float16).numpy().transpose()
      wkv_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_a.kernel"].to(torch.float16).numpy().transpose()
      wkv_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_b.kernel"].to(torch.float16).numpy().transpose()
      out = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.out.kernel"].to(torch.float16).numpy().transpose()
      if q_lora_rank != 0:
        q_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.q_norm.scale"].to(torch.float16).numpy()
        wq_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_a.kernel"].to(torch.float16).numpy().transpose()
        wq_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_b.kernel"].to(torch.float16).numpy().transpose()
      else:
        query = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"].to(torch.float16).numpy().transpose()

      # reshape to match maxtext
      wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
      out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
      if q_lora_rank != 0:
        wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
      else:
        query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

      # initialization
      if self_attention["kv_norm"]["scale"] is None:
        stack_shape = (layer_value,)
        self_attention["kv_norm"]["scale"] = np.zeros(stack_shape + kv_norm.shape, dtype=np.float16)
        self_attention["wkv_a"]["kernel"] = np.zeros(stack_shape + wkv_a.shape, dtype=np.float16)
        self_attention["wkv_b"]["kernel"] = np.zeros(stack_shape + wkv_b.shape, dtype=np.float16)
        self_attention["out"]["kernel"] = np.zeros(stack_shape + out.shape, dtype=np.float16)
        pre_self_attention_layer_norm["scale"] = np.zeros(stack_shape + pre_self_attention.shape, dtype=np.float16)
        post_self_attention_layer_norm["scale"] = np.zeros(stack_shape + post_self_attention.shape, dtype=np.float16)
        if q_lora_rank != 0:
          self_attention.update(
              {
                  "q_norm": {"scale": None},
                  "wq_a": {"kernel": None},
                  "wq_b": {"kernel": None},
              }
          )
          self_attention["q_norm"]["scale"] = np.zeros(stack_shape + q_norm.shape, dtype=np.float16)
          self_attention["wq_a"]["kernel"] = np.zeros(stack_shape + wq_a.shape, dtype=np.float16)
          self_attention["wq_b"]["kernel"] = np.zeros(stack_shape + wq_b.shape, dtype=np.float16)
        else:
          self_attention.update({"query": {"kernel": None}})
          self_attention["query"]["kernel"] = np.zeros(stack_shape + query.shape, dtype=np.float16)

      self_attention["kv_norm"]["scale"][layer_idx, ...] = kv_norm
      self_attention["wkv_a"]["kernel"][layer_idx, ...] = wkv_a
      self_attention["wkv_b"]["kernel"][layer_idx, ...] = wkv_b
      self_attention["out"]["kernel"][layer_idx, ...] = out
      pre_self_attention_layer_norm["scale"][layer_idx, ...] = pre_self_attention
      post_self_attention_layer_norm["scale"][layer_idx, ...] = post_self_attention
      if q_lora_rank != 0:
        self_attention["q_norm"]["scale"][layer_idx, ...] = q_norm
        self_attention["wq_a"]["kernel"][layer_idx, ...] = wq_a
        self_attention["wq_b"]["kernel"][layer_idx, ...] = wq_b
      else:
        self_attention["query"]["kernel"][layer_idx, ...] = query

    # re-order to fit maxtext
    self_attention["kv_norm"]["scale"] = np.transpose(self_attention["kv_norm"]["scale"], axes=(1, 0))
    self_attention["wkv_a"]["kernel"] = np.transpose(self_attention["wkv_a"]["kernel"], axes=(1, 0, 2))
    self_attention["wkv_b"]["kernel"] = np.transpose(self_attention["wkv_b"]["kernel"], axes=(1, 0, 2, 3))
    self_attention["out"]["kernel"] = np.transpose(self_attention["out"]["kernel"], axes=(1, 0, 2, 3))
    pre_self_attention_layer_norm["scale"] = np.transpose(pre_self_attention_layer_norm["scale"], axes=(1, 0))
    post_self_attention_layer_norm["scale"] = np.transpose(post_self_attention_layer_norm["scale"], axes=(1, 0))
    if q_lora_rank != 0:
      self_attention["q_norm"]["scale"] = np.transpose(self_attention["q_norm"]["scale"], axes=(1, 0))
      self_attention["wq_a"]["kernel"] = np.transpose(self_attention["wq_a"]["kernel"], axes=(1, 0, 2))
      self_attention["wq_b"]["kernel"] = np.transpose(self_attention["wq_b"]["kernel"], axes=(1, 0, 2, 3))
    else:
      self_attention["query"]["kernel"] = np.transpose(self_attention["query"]["kernel"], axes=(1, 0, 2, 3))

    jax_weights["decoder"][f"{layer_key}"]["self_attention"] = self_attention
    jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
    jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"] = post_self_attention_layer_norm
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")
  for layer_key, layer_value in layers.items():
    if layer_key == "dense_layers":
      mlp = jax_weights["decoder"][f"{layer_key}"]["mlp"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        wi_0 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"].to(torch.float16).numpy().transpose()
        wi_1 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"].to(torch.float16).numpy().transpose()
        wo = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"].to(torch.float16).numpy().transpose()
        # initialization
        if mlp["wo"]["kernel"] is None:
          stack_shape = (layer_value,)
          mlp["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
          mlp["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
          mlp["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
        mlp["wi_0"]["kernel"][layer_idx, ...] = wi_0
        mlp["wi_1"]["kernel"][layer_idx, ...] = wi_1
        mlp["wo"]["kernel"][layer_idx, ...] = wo

      mlp["wi_0"]["kernel"] = np.transpose(mlp["wi_0"]["kernel"], axes=(1, 0, 2))
      mlp["wi_1"]["kernel"] = np.transpose(mlp["wi_1"]["kernel"], axes=(1, 0, 2))
      mlp["wo"]["kernel"] = np.transpose(mlp["wo"]["kernel"], axes=(1, 0, 2))
      jax_weights["decoder"][f"{layer_key}"]["mlp"] = mlp
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    else:
      moe = jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        if q_lora_rank != 0:
          gate_bias = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
        gate = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_0 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_1 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wo = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wo.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )

        # initialization
        if moe["MoeBlock_0"]["gate"]["kernel"] is None:
          stack_shape = (layer_value,)
          if q_lora_rank != 0:
            moe["MoeBlock_0"]["gate"].update({"bias": None})
            moe["MoeBlock_0"]["gate"]["bias"] = np.zeros(stack_shape + gate_bias.shape, dtype=np.float16)
          moe["MoeBlock_0"]["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=np.float16)
          moe["shared_experts"]["wi_0"]["kernel"] = np.zeros(stack_shape + shared_wi_0.shape, dtype=np.float16)
          moe["shared_experts"]["wi_1"]["kernel"] = np.zeros(stack_shape + shared_wi_1.shape, dtype=np.float16)
          moe["shared_experts"]["wo"]["kernel"] = np.zeros(stack_shape + shared_wo.shape, dtype=np.float16)

        if q_lora_rank != 0:
          moe["MoeBlock_0"]["gate"]["bias"][layer_idx, ...] = gate_bias
        moe["MoeBlock_0"]["gate"]["kernel"][layer_idx, ...] = gate
        moe["shared_experts"]["wi_0"]["kernel"][layer_idx, ...] = shared_wi_0
        moe["shared_experts"]["wi_1"]["kernel"][layer_idx, ...] = shared_wi_1
        moe["shared_experts"]["wo"]["kernel"][layer_idx, ...] = shared_wo

      # re-order
      if q_lora_rank != 0:
        moe["MoeBlock_0"]["gate"]["bias"] = np.transpose(moe["MoeBlock_0"]["gate"]["bias"], axes=(1, 0))
      moe["MoeBlock_0"]["gate"]["kernel"] = np.transpose(moe["MoeBlock_0"]["gate"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_0"]["kernel"] = np.transpose(moe["shared_experts"]["wi_0"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_1"]["kernel"] = np.transpose(moe["shared_experts"]["wi_1"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wo"]["kernel"] = np.transpose(moe["shared_experts"]["wo"]["kernel"], axes=(1, 0, 2))

      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        for k in tqdm(range(num_experts), desc="experts", leave=False):
          wi_0 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wi_1 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wo = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )

          if moe["MoeBlock_0"]["wi_0"] is None:
            stack_shape = (num_experts, layer_value)
            moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wi_0"][k, layer_idx, ...] = wi_0
          moe["MoeBlock_0"]["wi_1"][k, layer_idx, ...] = wi_1
          moe["MoeBlock_0"]["wo"][k, layer_idx, ...] = wo

      jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"] = moe
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # MTP Layer Processing ################################################
  if has_mtp:
    max_logging.log("Processing MTP Layer")

    # MTP unique components
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_embedding_norm"]["scale"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_embedding_norm.scale"].to(torch.float16).numpy()
    )
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_hidden_state_norm"]["scale"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_hidden_state_norm.scale"].to(torch.float16).numpy()
    )
    jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_projection"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_projection.kernel"].to(torch.float16).numpy().transpose()
    )

    # MTP internal transformer layer - Attention and Norms
    mtp_transformer_layer = jax_weights["mtp_block"]["mtp_layer_1"]["mtp_1_transformer_layer"]
    mtp_transformer_layer["pre_self_attention_layer_norm"]["scale"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.pre_self_attention_layer_norm.scale"]
        .to(torch.float16)
        .numpy()
    )
    mtp_transformer_layer["post_self_attention_layer_norm"]["scale"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.post_self_attention_layer_norm.scale"]
        .to(torch.float16)
        .numpy()
    )

    mtp_attn_block = mtp_transformer_layer["self_attention"]
    mtp_attn_block["kv_norm"]["scale"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.kv_norm.scale"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    mtp_attn_block["wkv_a"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_a.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    wkv_b = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wkv_b.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    out = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.out.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    mtp_attn_block["wkv_b"]["kernel"] = np.reshape(
        wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)]
    )
    mtp_attn_block["out"]["kernel"] = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
    if q_lora_rank != 0:
      mtp_attn_block.update(
          {
              "q_norm": {"scale": None},
              "wq_a": {"kernel": None},
              "wq_b": {"kernel": None},
          }
      )
    else:
      mtp_attn_block.update({"query": {"kernel": None}})
    if q_lora_rank != 0:
      mtp_attn_block["q_norm"]["scale"] = (
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.q_norm.scale"]
          .to(torch.float16)
          .numpy()
      )
      mtp_attn_block["wq_a"]["kernel"] = (
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_a.kernel"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )
      wq_b = (
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.wq_b.kernel"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )
      mtp_attn_block["wq_b"]["kernel"] = np.reshape(
          wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)]
      )
    else:
      query = (
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.self_attention.query.kernel"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )
      mtp_attn_block["query"]["kernel"] = np.reshape(
          query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)]
      )

    # MTP internal transformer layer - MoE Block
    moe = mtp_transformer_layer["DeepSeekMoeBlock_0"]
    moe["MoeBlock_0"]["gate"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    moe["shared_experts"]["wi_0"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    moe["shared_experts"]["wi_1"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    moe["shared_experts"]["wo"]["kernel"] = (
        chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.shared_experts.wo.kernel"]
        .to(torch.float16)
        .numpy()
        .transpose()
    )
    if q_lora_rank != 0:
      moe["MoeBlock_0"]["gate"].update({"bias": None})
      moe["MoeBlock_0"]["gate"]["bias"] = (
          chkpt_vars["mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )

    experts_wi0, experts_wi1, experts_wo = [], [], []
    for k in range(num_experts):
      experts_wi0.append(
          chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )
      experts_wi1.append(
          chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )
      experts_wo.append(
          chkpt_vars[f"mtp_block.mtp_layer_1.mtp_1_transformer_layer.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"]
          .to(torch.float16)
          .numpy()
          .transpose()
      )

    moe["MoeBlock_0"]["wi_0"] = np.stack(experts_wi0)
    moe["MoeBlock_0"]["wi_1"] = np.stack(experts_wi1)
    moe["MoeBlock_0"]["wo"] = np.stack(experts_wo)

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def _convert_to_jax_weights(base_model_path, model_size, mem_info, enable_mtp=False) -> dict:
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText.

  Args:
      base_model_path: Path to the Hugging Face model checkpoint.
      model_size: Model size key in MODEL_PARAMS_DICT.
      mem_info: A process instance used for memory tracking.
      enable_mtp: Whether to enable MTP conversion.

  Returns:
      The converted JAX weights.
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info, enable_mtp)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert and save DeepSeek model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--simulated_cpu_devices_count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  parser.add_argument("--enable_mtp", type=str2bool, required=False, default=False, help="Enable MTP layer conversion.")
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  mem_info = psutil.Process()
  llama_or_mistral_ckpt.save_weights_to_checkpoint(
      args.maxtext_model_path,
      _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info, args.enable_mtp),
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )


if __name__ == "__main__":
  main()
