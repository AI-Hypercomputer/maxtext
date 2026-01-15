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

"""Defines the weight mapping from MaxText's Llama3 model to a vLLM-compatible format.

This module provides the `LLAMA3_VLLM_MAPPING` dataclass, which contains all the
necessary configurations to convert MaxText's Llama3 model weights into a
format that can be loaded by HuggingFace's vLLM. This includes:
- A direct mapping of parameter names.
- Sharding specifications for distributed environments.
- Hook functions for complex transformations (e.g., RoPE reordering).
"""

from dataclasses import dataclass

import numpy as np

import jax


@dataclass
class LLAMA3_VLLM_MAPPING:
  """Mapping MaxText Llama 2 and Llama 3 weights to vLLM's Llama 2 and Llama 3 weights."""

  @staticmethod
  def to_hf_hook_fns():
    """Defines and returns hook functions for weight transformations.

    These hooks are applied to specific weights during the conversion
    from MaxText to a HuggingFace-compatible format. They handle
    transformations like RoPE reordering and query scaling that are not
    simple re-mappings.

    Returns:
      A dictionary where keys are MaxText parameter names and values are
      the corresponding transformation functions.
    """

    def reorder_rope(arr):
      """Reorders Rotary Position Embedding (RoPE) weights.

      This function is necessary because MaxText and HuggingFace's vLLM
      implementations may have different orderings for RoPE dimensions.
      It splits the last dimension into even and odd indices and
      concatenates them.

      Args:
        arr: The input weight array.

      Returns:
        The reordered weight array.
      """
      evens = arr[..., ::2]
      odds = arr[..., 1::2]
      return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

    def transform_query_kernel(arr):
      """Transforms the query kernel.

      This involves scaling the kernel by the square root of the head
      dimension and then applying RoPE reordering.

      Args:
        arr: The query kernel weight array.

      Returns:
        The transformed query kernel array.
      """
      head_dim = arr.shape[-1]
      depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
      arr = arr * depth_scale
      return reorder_rope(arr)

    hook_fns = {
        "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
        "base.decoder.layers.self_attention.key.kernel": reorder_rope,
    }
    return hook_fns

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
    """
    Mapping from MaxText model to HuggingFace vLLM model.

    Currently, the param mapping conforms to the Tunix API, which combines the param name & sharding in one dictionary.
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
    }
