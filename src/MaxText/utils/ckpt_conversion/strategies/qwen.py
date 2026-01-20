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

"""Qwen family parameter mapping strategies."""

from typing import Any, Dict, Callable
import numpy as np
from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class QwenMapper(ParamMapperStrategy):
  """Strategy for Qwen3 models."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns mapping from MaxText to HuggingFace Qwen3 weight paths.

    Handles both dense and Mixture-of-Experts (MoE) model variants.
    """
    n_layers = hf_config["num_hidden_layers"]
    num_experts = hf_config.get("num_experts", 0)

    mapping = {
        "params-token_embedder-embedding": "model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "model.norm.weight",
        "params-decoder-logits_dense-kernel": "lm_head.weight",
    }

    if scan_layers:
      # This block handles scanned layers for both dense and MoE models.
      mapping.update(
          {
              "params-decoder-layers-pre_self_attention_layer_norm-scale": [
                  f"model.layers.{i}.input_layernorm.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-query-kernel": [
                  f"model.layers.{i}.self_attn.q_proj.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-key-kernel": [
                  f"model.layers.{i}.self_attn.k_proj.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-value-kernel": [
                  f"model.layers.{i}.self_attn.v_proj.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-out-kernel": [
                  f"model.layers.{i}.self_attn.o_proj.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-query_norm-scale": [
                  f"model.layers.{i}.self_attn.q_norm.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-self_attention-key_norm-scale": [
                  f"model.layers.{i}.self_attn.k_norm.weight" for i in range(n_layers)
              ],
              "params-decoder-layers-post_self_attention_layer_norm-scale": [
                  f"model.layers.{i}.post_attention_layernorm.weight" for i in range(n_layers)
              ],
          }
      )
      if num_experts > 1:
        # For scanned MoE, we create a nested list: [[e0_l0, e0_l1..], [e1_l0, e1_l1..]..]
        # This follows the (experts, layers, ...) tensor layout.
        mapping.update(
            {
                "params-decoder-layers-moe_block-gate-kernel": [
                    f"model.layers.{i}.mlp.gate.weight" for i in range(n_layers)
                ],
                "params-decoder-layers-moe_block-wi_0": [
                    [f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight" for l in range(n_layers)]
                    for e in range(num_experts)
                ],
                "params-decoder-layers-moe_block-wi_1": [
                    [f"model.layers.{l}.mlp.experts.{e}.up_proj.weight" for l in range(n_layers)]
                    for e in range(num_experts)
                ],
                "params-decoder-layers-moe_block-wo": [
                    [f"model.layers.{l}.mlp.experts.{e}.down_proj.weight" for l in range(n_layers)]
                    for e in range(num_experts)
                ],
            }
        )
      else:  # Dense MLP
        mapping.update(
            {
                "params-decoder-layers-mlp-wi_0-kernel": [
                    f"model.layers.{i}.mlp.gate_proj.weight" for i in range(n_layers)
                ],
                "params-decoder-layers-mlp-wi_1-kernel": [
                    f"model.layers.{i}.mlp.up_proj.weight" for i in range(n_layers)
                ],
                "params-decoder-layers-mlp-wo-kernel": [
                    f"model.layers.{i}.mlp.down_proj.weight" for i in range(n_layers)
                ],
            }
        )
    else:  # unscanned layers
      for i in range(n_layers):
        # Common Attention and Norms
        mapping.update(
            {
                f"params-decoder-layers_{i}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
                f"params-decoder-layers_{i}-self_attention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
                f"params-decoder-layers_{i}-self_attention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
                f"params-decoder-layers_{i}-self_attention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
                f"params-decoder-layers_{i}-self_attention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
                f"params-decoder-layers_{i}-self_attention-query_norm-scale": f"model.layers.{i}.self_attn.q_norm.weight",
                f"params-decoder-layers_{i}-self_attention-key_norm-scale": f"model.layers.{i}.self_attn.k_norm.weight",
                f"params-decoder-layers_{i}-post_self_attention_layer_norm-scale": (
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ),
            }
        )
        if num_experts > 1:
          # For each unscanned MoE layer, map the MaxText parameter to a 1D list of all expert weights for that layer.
          mapping.update(
              {
                  f"params-decoder-layers_{i}-moe_block-gate-kernel": f"model.layers.{i}.mlp.gate.weight",
                  f"params-decoder-layers_{i}-moe_block-wi_0": [
                      f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight" for j in range(num_experts)
                  ],
                  f"params-decoder-layers_{i}-moe_block-wi_1": [
                      f"model.layers.{i}.mlp.experts.{j}.up_proj.weight" for j in range(num_experts)
                  ],
                  f"params-decoder-layers_{i}-moe_block-wo": [
                      f"model.layers.{i}.mlp.experts.{j}.down_proj.weight" for j in range(num_experts)
                  ],
              }
          )
        else:  # Dense MLP
          mapping.update(
              {
                  f"params-decoder-layers_{i}-mlp-wi_0-kernel": f"model.layers.{i}.mlp.gate_proj.weight",
                  f"params-decoder-layers_{i}-mlp-wi_1-kernel": f"model.layers.{i}.mlp.up_proj.weight",
                  f"params-decoder-layers_{i}-mlp-wo-kernel": f"model.layers.{i}.mlp.down_proj.weight",
              }
          )
    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Creates parameter transformation functions for Qwen3."""
    n_layers = hf_config["num_hidden_layers"]
    num_experts = hf_config.get("num_experts", 0)

    def pad_embedding_layer(input_tensor, target_shape):
      """Pads or truncates embedding layer to match target vocab size."""
      source_vocab_size = input_tensor.shape[0]
      target_vocab_size = target_shape[0]

      if source_vocab_size == target_vocab_size:
        return input_tensor

      if saving_to_hf:  # MaxText to HF, truncate
        return input_tensor[:target_vocab_size, :]
      else:  # HF to MaxText, pad
        padded_tensor = np.zeros(target_shape, dtype=input_tensor.dtype)
        padded_tensor[:source_vocab_size, :] = input_tensor
        return padded_tensor

    def reshape_kernel(input_tensor, target_shape):
      """Reshapes and transposes kernel weights between MaxText and HF."""
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      else:
        return input_tensor.T.reshape(target_shape)

    mapping = {
        "params-token_embedder-embedding": pad_embedding_layer,
        "params-decoder-logits_dense-kernel": reshape_kernel,
    }

    kernel_hooks = [
        "self_attention-query-kernel",
        "self_attention-key-kernel",
        "self_attention-value-kernel",
        "self_attention-out-kernel",
        "mlp-wi_0-kernel",
        "mlp-wi_1-kernel",
        "mlp-wo-kernel",
    ]
    moe_kernel_hooks = [
        "moe_block-gate-kernel",
        "moe_block-wi_0-kernel",
        "moe_block-wi_1-kernel",
        "moe_block-wo-kernel",
        "moe_block-wi_0",
        "moe_block-wi_1",
        "moe_block-wo",
    ]

    if scan_layers:
      for key in kernel_hooks:
        mapping[f"params-decoder-layers-{key}"] = reshape_kernel
      if num_experts > 1:
        for key in moe_kernel_hooks:
          mapping[f"params-decoder-layers-{key}"] = reshape_kernel
    else:
      for i in range(n_layers):
        for key in kernel_hooks:
          mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
        if num_experts > 1:
          for key in moe_kernel_hooks:
            mapping[f"params-decoder-layers_{i}-{key}"] = reshape_kernel
    return mapping

  def get_vllm_hooks(self) -> Dict[str, Callable]:
    """Creates parameter transformation functions for Qwen3 in VLLM."""
    return {}


class QwenOmniMoeMapper(QwenMapper):
  """Strategy for Qwen3 Omni models."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns mapping from MaxText to HuggingFace Qwen3-Omni weight paths.

    Combines mappings from different modalities (text, vision, audio, etc.).
    """
    # Collect all modality mappings
    mapping = {}

    # Text mapping with "thinker." prefix, reusing QwenMapper logic
    # In Omni, the text config is nested under "thinker_config" -> "text_config"
    text_config = hf_config["thinker_config"]["text_config"]
    # Reuse QwenMapper but with the extracted text config
    text_mapping = super().get_mapping(text_config, maxtext_config, scan_layers)

    # Add "thinker." prefix to text mapping values
    for key, value in text_mapping.items():
      # If list (scanned), apply prefix to each element
      if isinstance(value, list):
        new_val = []
        for v in value:
          # Handle nested lists for MoE experts
          if isinstance(v, list):
            new_val.append([f"thinker.{x}" for x in v])
          else:
            new_val.append(f"thinker.{v}")
        mapping[key] = new_val
      else:
        mapping[key] = f"thinker.{value}"

    # TODO(hengtaoguo): Add vision, audio, and other modality mappings here similarly
    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Creates parameter transformation functions for Qwen3-Omni."""
    mapping = {}

    # Text hooks, reusing QwenMapper logic
    text_config = hf_config["thinker_config"]["text_config"]
    text_hooks = super().get_hooks(text_config, maxtext_config, scan_layers, saving_to_hf)
    mapping.update(text_hooks)

    # TODO(hengtaoguo): Add vision, audio, and other modality mappings here similarly
    return mapping
