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

from dataclasses import dataclass
import numpy as np
import jax


@dataclass
class LLAMA3_VLLM_MAPPING:
  """Mapping MaxText Llama 2 and Llama 3 weights to vLLM's Llama 2 and Llama 3 weights."""

  def to_hf_hook_fns():

    def reorder_rope(arr):
      evens = arr[..., ::2]
      odds = arr[..., 1::2]
      return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

    def transform_query_kernel(arr):
      head_dim = arr.shape[-1]
      depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
      arr = arr * depth_scale
      return reorder_rope(arr)

    hook_fns = {
        "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
        "base.decoder.layers.self_attention.key.kernel": reorder_rope,
    }
    return hook_fns

  def to_hf_transpose_keys():
    return {}

  def lora_to_hf_mappings():
    return None

  def to_hf_mapping():
    """
    Mapping from MaxText model to HuggingFace vLLM model.
    Currently the param mapping conforms to the Tunix API, which combines the param name and sharding in one dictionary.
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
