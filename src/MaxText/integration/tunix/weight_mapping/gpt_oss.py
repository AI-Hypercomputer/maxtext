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

"""Mapping MaxText GPT-OSS (MoE) weights to vLLM/tpu-inference keys."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class GPT_OSS_VLLM_MAPPING:
  """
  Mapping definition from MaxText GPT-OSS (Scanned/Interleaved) to vLLM JAX NNX.
  Supports:
  - Modulo Interleaving (e.g., Block 0 -> Layers 0, 2, 4...)
  """

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights.
    Returns:
        None, as LoRA mappings are not defined for this model.
    """
    return None

  @staticmethod
  def to_hf_hook_fns():
    """Returns hook functions to fuse interleaved weights."""
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    """Returns keys that need to be transposed."""
    return {}

  @staticmethod
  def to_hf_mapping(
      layer_cycle_interval: int = 2, total_num_layers: int = 36, interleave_style: str = "modulo"
  ) -> Dict[str, Tuple[str, Tuple[Optional[str], ...]]]:
    """Returns the weight mapping for the model.
    Args:
        layer_cycle_interval: The interval at which layers are cycled.
        total_num_layers: The total number of layers in the model.
        interleave_style: The style of interleaving used for the layers.
    Returns:
        A dictionary mapping MaxText parameter names to vLLM parameter names.
    """

    mapping = {}

    # --- 1. Global Parameters ---
    mapping.update(
        {
            "base.token_embedder.embedding": ("embedder.input_embedding_table_VD", ("model", None)),
            "base.decoder.decoder_norm.scale": ("final_norm.scale", (None,)),
            "base.decoder.logits_dense.kernel": ("lm_head.input_embedding_table_DV", (None, "model")),
        }
    )

    # --- 2. Layer Mapping Loop ---
    layers_per_block = total_num_layers // layer_cycle_interval

    for block_idx in range(layer_cycle_interval):
      src_block = f"base.decoder.layers.layers_{block_idx}"
      if interleave_style == "modulo":
        target_indices = range(block_idx, total_num_layers, layer_cycle_interval)
      else:
        start = block_idx * layers_per_block
        target_indices = range(start, start + layers_per_block)

      regex_indices = "|".join(map(str, target_indices))

      layer_regex = f"layers\\.({regex_indices})"
      # --- 3. Block Mappings (Standard) ---
      mapping.update(
          {
              f"{src_block}.pre_self_attention_layer_norm.scale": (
                  f"{layer_regex}.pre_attention_norm.scale",
                  (None, "layer"),
              ),
              f"{src_block}.post_self_attention_layer_norm.scale": (f"{layer_regex}.pre_mlp_norm.scale", (None, "layer")),
              f"{src_block}.GptOssAttention.query.kernel": (
                  f"{layer_regex}.attn.kernel_q_DNH",
                  (None, "layer", "model", None),
              ),
              f"{src_block}.GptOssAttention.key.kernel": (
                  f"{layer_regex}.attn.kernel_k_DKH",
                  (None, "layer", "model", None),
              ),
              f"{src_block}.GptOssAttention.value.kernel": (
                  f"{layer_regex}.attn.kernel_v_DKH",
                  (None, "layer", "model", None),
              ),
              f"{src_block}.GptOssAttention.out.kernel": (
                  f"{layer_regex}.attn.kernel_o_proj_NHD",
                  ("model", "layer", None, None),
              ),
              f"{src_block}.GptOssAttention.query.bias": (f"{layer_regex}.attn.bias_q_NH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.key.bias": (f"{layer_regex}.attn.bias_k_KH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.value.bias": (f"{layer_regex}.attn.bias_v_KH", (None, "layer", None)),
              f"{src_block}.GptOssAttention.out.bias": (f"{layer_regex}.attn.bias_o_D", (None, "layer")),
              f"{src_block}.GptOssAttention.sinks": (f"{layer_regex}.attn.sinks_N", (None, "layer")),
          }
      )

      # MoE Router
      mapping.update(
          {
              f"{src_block}.GptOssMlp.gate.kernel": (
                  f"{layer_regex}.custom_module.router.kernel_DE",
                  (None, "layer", "model"),
              ),
              f"{src_block}.GptOssMlp.gate.bias": (f"{layer_regex}.custom_module.router.bias_E", ("model", "layer")),
          }
      )

      # --- MOE EXPERTS ---
      # Separate gate_proj (wi_0) and up_proj (wi_1) kernels and biases.

      # MLP Gate Projection (wi_0)
      mapping.update(
          {
              f"{src_block}.GptOssMlp.wi_0": (f"{layer_regex}.custom_module.gate_proj_kernel", ("model", "layer", None)),
              f"{src_block}.GptOssMlp.wi_0_bias": (f"{layer_regex}.custom_module.gate_proj_bias", ("model", "layer")),
          }
      )

      # MLP Up Projection (wi_1)
      mapping.update(
          {
              f"{src_block}.GptOssMlp.wi_1": (f"{layer_regex}.custom_module.up_proj_kernel", ("model", "layer", None)),
              f"{src_block}.GptOssMlp.wi_1_bias": (f"{layer_regex}.custom_module.up_proj_bias", ("model", "layer")),
          }
      )

      # MLP Down Projection (wo)
      mapping.update(
          {
              f"{src_block}.GptOssMlp.wo": (f"{layer_regex}.custom_module.mlp2_weight_EFD", ("model", "layer", None)),
              f"{src_block}.GptOssMlp.wo_bias": (f"{layer_regex}.custom_module.mlp2_bias_ED", ("model", "layer")),
          }
      )

    return mapping
