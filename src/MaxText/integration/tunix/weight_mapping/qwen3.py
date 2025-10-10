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

"""Defines the weight mapping from Maxtext to Qwen3 model to a vLLM compatible format.
This module provides the weight mapping for the Qwen3 model. It maps the
Maxtext model to the Qwen3 model to a vLLM compatible format.

"""
from dataclasses import dataclass
import numpy as np


@dataclass
class QWEN3_VLLM_MAPPING:
  """Mapping MaxText Qwen3 weights to vLLM's Qwen3 weights."""

  @staticmethod
  def to_hf_hook_fns():
    """Defines and returns hook functions for Qwen3 weight transformations.

    These hooks are applied during the conversion from MaxText to a
    HuggingFace-compatible format for vLLM. They handle transformations
    like vocabulary padding/truncation and kernel reshaping.

    Returns:
      A dictionary where keys are MaxText parameter names and values are
      the corresponding transformation functions.
    """

    def pad_embedding_to_hf(input_tensor, target_shape):
      """Truncates MaxText embedding layer to match HF vocab size."""
      target_vocab_size = target_shape[0]
      # When saving to HF, we truncate the padded vocab from MaxText
      return input_tensor[:target_vocab_size, :]

    def reshape_kernel_to_hf(input_tensor, target_shape):
      """Reshapes and transposes kernel weights from MaxText to HF."""
      flipped_target_shape = np.flip(np.array(target_shape))
      return input_tensor.reshape(flipped_target_shape).T

    hook_fns = {
        "base.token_embedder.embedding": pad_embedding_to_hf,
        "base.decoder.logits_dense.kernel": reshape_kernel_to_hf,
        # Attention Kernels
        "base.decoder.layers.self_attention.query.kernel": reshape_kernel_to_hf,
        "base.decoder.layers.self_attention.key.kernel": reshape_kernel_to_hf,
        "base.decoder.layers.self_attention.value.kernel": reshape_kernel_to_hf,
        "base.decoder.layers.self_attention.out.kernel": reshape_kernel_to_hf,
    }

    # Add hooks for dense MLP
    dense_hooks = {
        "base.decoder.layers.mlp.wi_0.kernel": reshape_kernel_to_hf,
        "base.decoder.layers.mlp.wi_1.kernel": reshape_kernel_to_hf,
        "base.decoder.layers.mlp.wo.kernel": reshape_kernel_to_hf,
    }
    hook_fns.update(dense_hooks)

    return hook_fns

  @staticmethod
  def to_hf_transpose_keys():
    """Returns a list of keys for weights that need to be transposed."""
    # All kernel transformations are handled by the reshape hook function,
    # so no simple transpositions are needed here.
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights."""
    return None

  @staticmethod
  def to_hf_mapping():
    """
    Defines the mapping from MaxText Qwen3 parameters to HuggingFace vLLM format.

    This mapping includes the target parameter name and the sharding
    specification for distributed execution. It dynamically adjusts for dense
    models

    Returns:
      A dictionary mapping MaxText names to (vLLM_name, sharding_spec) tuples.
    """
    # Base mapping for parameters outside the decoder layers
    mapping = {
        # Token embeddings - shard vocab dimension
        "base.token_embedder.embedding": (
            "model.embed_tokens.weight",
            ("model", None),
        ),
        # Final layer norm - replicated
        "base.decoder.decoder_norm.scale": ("model.norm.weight", (None,)),
        # LM head (logits projection) - shard vocab dimension
        "base.decoder.logits_dense.kernel": ("lm_head.weight", (None, "model")),
    }

    # Layer-specific mappings (scanned -> unscanned with wildcards)
    layer_mapping = {
        # Layer norms - replicated
        "base.decoder.layers.pre_self_attention_layer_norm.scale": (
            "model.layers.*.input_layernorm.weight",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_layer_norm.scale": (
            "model.layers.*.post_attention_layernorm.weight",
            (None, "layer"),
        ),
        # Attention components - shard head dimensions
        "base.decoder.layers.self_attention.query.kernel": (
            "model.layers.*.self_attn.q_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.key.kernel": (
            "model.layers.*.self_attn.k_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.value.kernel": (
            "model.layers.*.self_attn.v_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.layers.*.self_attn.o_proj.weight",
            ("model", "layer", None, None),
        ),
        # Qwen3 Specific: QK Norms - replicated
        "base.decoder.layers.self_attention.query_norm.scale": (
            "model.layers.*.self_attn.q_norm.weight",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.key_norm.scale": (
            "model.layers.*.self_attn.k_norm.weight",
            (None, "layer"),
        ),
    }
    mapping.update(layer_mapping)

    # Add MLP mappings for dense model
    dense_mapping = {
        # Dense MLP components - shard hidden dimensions
        "base.decoder.layers.mlp.wi_0.kernel": ("model.layers.*.mlp.gate_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wi_1.kernel": ("model.layers.*.mlp.up_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wo.kernel": ("model.layers.*.mlp.down_proj.weight", ("model", "layer", None)),
    }
    mapping.update(dense_mapping)

    return mapping
