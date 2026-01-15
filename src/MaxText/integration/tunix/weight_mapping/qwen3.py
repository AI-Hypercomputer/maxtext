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

"""Defines the weight mapping from MaxText's Qwen3 model to a vLLM-compatible format.

This module provides the `QWEN3_VLLM_MAPPING` dataclass, which contains all the
necessary configurations to convert MaxText's Qwen3 model weights into a
format that can be loaded by HuggingFace's vLLM. This includes:
- A direct mapping of parameter names.
- Sharding specifications for distributed environments.
"""

from dataclasses import dataclass


@dataclass
class QWEN3_VLLM_MAPPING:
  """Mapping MaxText Qwen3-8 weights to vLLM's Qwen3-8 weights."""

  @staticmethod
  def to_hf_hook_fns():
    """Returns a dictionary of hook functions to be applied to MaxText weights.

    Returns:
      An empty dictionary, as no hook functions are needed for this mapping.
    """

    return {}

  @staticmethod
  def to_hf_transpose_keys():
    """Returns a list of keys for weights that need to be transposed.

    Returns:
      An empty dictionary, as no keys require transposition for this mapping.
    """
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights.

    Returns:
      None, as LoRA mappings are not defined for this model.
    """
    return None

  @staticmethod
  def to_hf_mapping():
    """Mapping from MaxText model to HuggingFace vLLM model.

    Currently, the param mapping conforms to the Tunix API, which combines the
    param name & sharding in one dictionary.
    This is subject to change in the future where we can decouple the two.
    """
    return {
        # Token embeddings - shard vocab dimension
        "base.token_embedder.embedding": (
            "model.embed.embedding",
            ("model", None),
        ),
        # Final layer norm - no sharding needed
        "base.decoder.decoder_norm.scale": (
            "model.norm.scale",
            (None,),
        ),
        # LM head (logits projection) - shard vocab dimension
        "base.decoder.logits_dense.kernel": (
            "model.lm_head",
            (None, "model"),
        ),
        # Layer-specific mappings (scanned -> unscanned)
        # MLP components - shard hidden dimensions
        "base.decoder.layers.mlp.wi_0.kernel": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wi_1.kernel": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wo.kernel": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", "layer", None),
        ),
        # Layer norms - no sharding needed
        "base.decoder.layers.pre_self_attention_layer_norm.scale": (
            "model.layers.*.input_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_layer_norm.scale": (
            "model.layers.*.post_attention_layernorm.scale",
            (None, "layer"),
        ),
        # Attention components - shard head dimensions
        "base.decoder.layers.self_attention.query.kernel": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.key.kernel": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.value.kernel": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", "layer", None, None),
        ),
        "base.decoder.layers.self_attention.query_norm.scale": (
            "model.layers.*.self_attn.q_norm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.key_norm.scale": (
            "model.layers.*.self_attn.k_norm.scale",
            (None, "layer"),
        ),
    }
