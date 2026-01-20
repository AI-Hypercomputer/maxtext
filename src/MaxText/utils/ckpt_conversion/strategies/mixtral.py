# Copyright 2023–2026 Google LLC
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

"""Mixtral family parameter mapping strategies."""

import numpy as np
from typing import Any, Dict

from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class MixtralMapper(ParamMapperStrategy):
  """Strategy for Mixtral models."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Generates the mapping of parameter names from MaxText to Hugging Face for Mixtral."""
    mapping = {}

    # Top-level, non-layer-specific parameters
    mapping["params-token_embedder-embedding"] = "model.embed_tokens.weight"
    mapping["params-decoder-decoder_norm-scale"] = "model.norm.weight"
    mapping["params-decoder-logits_dense-kernel"] = "lm_head.weight"

    num_experts = maxtext_config.num_experts

    if scan_layers:
      # Initialize lists for scanned layer weights
      mapping.update(
          {
              "params-decoder-layers-self_attention-query-kernel": [],
              "params-decoder-layers-self_attention-key-kernel": [],
              "params-decoder-layers-self_attention-value-kernel": [],
              "params-decoder-layers-self_attention-out-kernel": [],
              "params-decoder-layers-pre_self_attention_layer_norm-scale": [],
              "params-decoder-layers-post_self_attention_layer_norm-scale": [],
              "params-decoder-layers-MoeBlock_0-gate-kernel": [],
              "params-decoder-layers-MoeBlock_0-wi_0": [],
              "params-decoder-layers-MoeBlock_0-wi_1": [],
              "params-decoder-layers-MoeBlock_0-wo": [],
          }
      )

      for i in range(hf_config["num_hidden_layers"]):
        hf_prefix = f"model.layers.{i}"
        # Attention weights
        mapping["params-decoder-layers-self_attention-query-kernel"].append(f"{hf_prefix}.self_attn.q_proj.weight")
        mapping["params-decoder-layers-self_attention-key-kernel"].append(f"{hf_prefix}.self_attn.k_proj.weight")
        mapping["params-decoder-layers-self_attention-value-kernel"].append(f"{hf_prefix}.self_attn.v_proj.weight")
        mapping["params-decoder-layers-self_attention-out-kernel"].append(f"{hf_prefix}.self_attn.o_proj.weight")

        # RMSNorm weights
        mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"].append(f"{hf_prefix}.input_layernorm.weight")
        mapping["params-decoder-layers-post_self_attention_layer_norm-scale"].append(
            f"{hf_prefix}.post_attention_layernorm.weight"
        )

        # MoE gate
        mapping["params-decoder-layers-MoeBlock_0-gate-kernel"].append(f"{hf_prefix}.block_sparse_moe.gate.weight")

      # Outer loop as experts and inner loop as layers to align with logic in _build_multi_axis_stacked_tensor()
      for j in range(num_experts):
        w1_layers = []
        w3_layers = []
        w2_layers = []

        for i in range(hf_config["num_hidden_layers"]):
          hf_prefix = f"model.layers.{i}"
          w1_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight")
          w3_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight")
          w2_layers.append(f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight")

        mapping["params-decoder-layers-MoeBlock_0-wi_0"].append(w1_layers)
        mapping["params-decoder-layers-MoeBlock_0-wi_1"].append(w3_layers)
        mapping["params-decoder-layers-MoeBlock_0-wo"].append(w2_layers)

    else:
      for i in range(hf_config["num_hidden_layers"]):
        maxtext_prefix = f"params-decoder-layers_{i}"
        hf_prefix = f"model.layers.{i}"

        # Attention weights
        mapping[f"{maxtext_prefix}-self_attention-query-kernel"] = f"{hf_prefix}.self_attn.q_proj.weight"
        mapping[f"{maxtext_prefix}-self_attention-key-kernel"] = f"{hf_prefix}.self_attn.k_proj.weight"
        mapping[f"{maxtext_prefix}-self_attention-value-kernel"] = f"{hf_prefix}.self_attn.v_proj.weight"
        mapping[f"{maxtext_prefix}-self_attention-out-kernel"] = f"{hf_prefix}.self_attn.o_proj.weight"

        # RMSNorm weights
        mapping[f"{maxtext_prefix}-pre_self_attention_layer_norm-scale"] = f"{hf_prefix}.input_layernorm.weight"
        mapping[f"{maxtext_prefix}-post_self_attention_layer_norm-scale"] = f"{hf_prefix}.post_attention_layernorm.weight"

        # MoE gate
        mapping[f"{maxtext_prefix}-MoeBlock_0-gate-kernel"] = f"{hf_prefix}.block_sparse_moe.gate.weight"

        # MoE expert weights (1 MaxText param -> 8 HF params)
        w1_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w1.weight" for j in range(num_experts)]
        w3_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w3.weight" for j in range(num_experts)]
        w2_experts = [f"{hf_prefix}.block_sparse_moe.experts.{j}.w2.weight" for j in range(num_experts)]

        mapping[f"{maxtext_prefix}-MoeBlock_0-wi_0"] = w1_experts
        mapping[f"{maxtext_prefix}-MoeBlock_0-wi_1"] = w3_experts
        mapping[f"{maxtext_prefix}-MoeBlock_0-wo"] = w2_experts

    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Generates parameter conversion hooks for Mixtral between MaxText and Hugging Face."""
    hooks = {}

    def reshape_and_transpose_attention(x, target_shape):
      """MaxText: [hidden, n_heads, h_dim] <-> HF: [n_heads * h_dim, hidden]"""
      if saving_to_hf:
        # (H, N, D) -> (H, N*D) -> (N*D, H)
        return x.reshape(hf_config["hidden_size"], -1).transpose()
      else:
        # (N*D, H) -> (H, N*D) -> (H, N, D)
        return x.transpose().reshape(target_shape)

    def reshape_kernel(x, target_shape):
      return x.transpose()

    def scale_query_layer(input_tensor, target_shape):
      if saving_to_hf:
        depth_scale = np.dtype("float32").type(np.sqrt(maxtext_config.head_dim))
        return (input_tensor * depth_scale).astype(input_tensor.dtype)
      else:
        depth_scale = np.dtype("float32").type(1 / np.sqrt(maxtext_config.head_dim))
        return (input_tensor * depth_scale).astype(input_tensor.dtype)

    # hook order does not affect result
    query_hook_chain = [reshape_and_transpose_attention, scale_query_layer]

    if scan_layers:
      plan = [
          ("params-decoder-layers-self_attention-query-kernel", query_hook_chain),
          ("params-decoder-layers-self_attention-key-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers-self_attention-value-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers-self_attention-out-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers-MoeBlock_0-wi_0", reshape_kernel),
          ("params-decoder-layers-MoeBlock_0-wi_1", reshape_kernel),
          ("params-decoder-layers-MoeBlock_0-wo", reshape_kernel),
          ("params-decoder-layers-MoeBlock_0-gate-kernel", reshape_kernel),
      ]
    else:
      plan = [
          ("params-decoder-layers_{i}-self_attention-query-kernel", query_hook_chain),
          ("params-decoder-layers_{i}-self_attention-key-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers_{i}-self_attention-value-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers_{i}-self_attention-out-kernel", reshape_and_transpose_attention),
          ("params-decoder-layers_{i}-MoeBlock_0-wi_0", reshape_kernel),
          ("params-decoder-layers_{i}-MoeBlock_0-wi_1", reshape_kernel),
          ("params-decoder-layers_{i}-MoeBlock_0-wo", reshape_kernel),
          ("params-decoder-layers_{i}-MoeBlock_0-gate-kernel", reshape_kernel),
      ]
    plan.append(("params-decoder-logits_dense-kernel", reshape_kernel))

    for maxtext_pattern, op_func in plan:
      if "{i}" in maxtext_pattern:
        for i in range(hf_config["num_hidden_layers"]):
          hooks[maxtext_pattern.format(i=i)] = op_func
      else:
        hooks[maxtext_pattern] = op_func
    return hooks
