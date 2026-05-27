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

"""Defines the weight mapping from MaxText's Gemma3 model to a vLLM-compatible format."""

from dataclasses import dataclass


@dataclass
class GEMMA3_VLLM_MAPPING:
  """Mapping MaxText Gemma3 weights to vLLM's Gemma3 weights."""

  @staticmethod
  def to_hf_hook_fns():
    """Returns a dictionary of hook functions to be applied to MaxText weights."""
    return {}

  @staticmethod
  def to_hf_transpose_keys():
    """Returns a list of keys for weights that need to be transposed."""
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights."""
    return None

  @staticmethod
  def to_hf_mapping():
    """Mapping from MaxText model to HuggingFace vLLM model.

    Returns:
      A dictionary mapping MaxText parameter names to HuggingFace parameter names and sharding.
    """
    return {
        # Token embeddings - shard vocab dimension
        "base.token_embedder.embedding": (
            "model.language_model.embed_tokens.kernel",
            ("model", None),
        ),
        # Final layer norm - no sharding needed
        "base.decoder.decoder_norm.scale": (
            "model.language_model.norm.scale",
            (None,),
        ),
        # Layer norms - no sharding needed
        "base.decoder.layers.pre_self_attention_norm.scale": (
            "model.language_model.layers.*.input_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_norm.scale": (
            "model.language_model.layers.*.post_attention_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.query_norm.scale": (
            "model.language_model.layers.*.self_attn.q_norm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.key_norm.scale": (
            "model.language_model.layers.*.self_attn.k_norm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.pre_ffw_norm.scale": (
            "model.language_model.layers.*.pre_feedforward_layernorm.scale",
            (None, "layer"),
        ),
        "base.decoder.layers.post_ffw_norm.scale": (
            "model.language_model.layers.*.post_feedforward_layernorm.scale",
            (None, "layer"),
        ),
        # MLP components - shard hidden dimensions
        "base.decoder.layers.mlp.wi_0.kernel": (
            "model.language_model.layers.*.mlp.gate_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wi_1.kernel": (
            "model.language_model.layers.*.mlp.up_proj.kernel",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.mlp.wo.kernel": (
            "model.language_model.layers.*.mlp.down_proj.kernel",
            ("model", "layer", None),
        ),
        # Attention components - shard head dimensions
        "base.decoder.layers.self_attention.query.kernel": (
            "model.language_model.layers.*.self_attn.q_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.key.kernel": (
            "model.language_model.layers.*.self_attn.k_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.value.kernel": (
            "model.language_model.layers.*.self_attn.v_proj.kernel",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.language_model.layers.*.self_attn.o_proj.kernel",
            ("model", "layer", None, None),
        ),
    }

