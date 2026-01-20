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

"""Llama family parameter mapping strategies."""

import numpy as np
import jax
from typing import Any, Dict, Callable

from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class LlamaMapper(ParamMapperStrategy):
  """Strategy for Llama 3.1 models."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Returns a dictionary mapping from MaxText parameter names to HuggingFace LLaMA3.1 parameter names."""
    n_layers = hf_config["num_hidden_layers"]

    mapping = {
        "params-token_embedder-embedding": "model.embed_tokens.weight",
        "params-decoder-logits_dense-kernel": "lm_head.weight",
        "params-decoder-decoder_norm-scale": "model.norm.weight",
    }

    if scan_layers:
      mapping["params-decoder-layers-self_attention-query-kernel"] = [
          f"model.layers.{layer_idx}.self_attn.q_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-self_attention-key-kernel"] = [
          f"model.layers.{layer_idx}.self_attn.k_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-self_attention-value-kernel"] = [
          f"model.layers.{layer_idx}.self_attn.v_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-self_attention-out-kernel"] = [
          f"model.layers.{layer_idx}.self_attn.o_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-mlp-wi_0-kernel"] = [
          f"model.layers.{layer_idx}.mlp.gate_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-mlp-wi_1-kernel"] = [
          f"model.layers.{layer_idx}.mlp.up_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-mlp-wo-kernel"] = [
          f"model.layers.{layer_idx}.mlp.down_proj.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-pre_self_attention_layer_norm-scale"] = [
          f"model.layers.{layer_idx}.input_layernorm.weight" for layer_idx in range(n_layers)
      ]
      mapping["params-decoder-layers-post_self_attention_layer_norm-scale"] = [
          f"model.layers.{layer_idx}.post_attention_layernorm.weight" for layer_idx in range(n_layers)
      ]
    else:
      for layer_idx in range(n_layers):
        mapping[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = (
            f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        )
        mapping[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = (
            f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        )
        mapping[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = (
            f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        )
        mapping[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = (
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        )
        mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        mapping[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        mapping[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        mapping[f"params-decoder-layers_{layer_idx}-pre_self_attention_layer_norm-scale"] = (
            f"model.layers.{layer_idx}.input_layernorm.weight"
        )
        mapping[f"params-decoder-layers_{layer_idx}-post_self_attention_layer_norm-scale"] = (
            f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        )

    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Creates parameter transformation functions for converting between MaxText and HuggingFace formats."""
    nlayers = hf_config["num_hidden_layers"]

    def scale_query_layer(input_tensor, target_shape):
      if saving_to_hf:
        depth_scale = np.dtype("float32").type(np.sqrt(hf_config["head_dim"]))
        original_dtype = input_tensor.dtype
        output_tensor = input_tensor.astype(np.float32) * depth_scale
        return output_tensor.astype(original_dtype)
      else:
        depth_scale = np.dtype("float32").type(1 / np.sqrt(hf_config["head_dim"]))
        original_dtype = input_tensor.dtype
        output_tensor = input_tensor.astype(np.float32) * depth_scale
        return output_tensor.astype(original_dtype)

    def adjust_rope(input_tensor, target_shape):
      arr = input_tensor
      if saving_to_hf:
        # Convert from MaxText's interleaved layout to HF's concatenated layout
        evens = arr[..., ::2]
        odds = arr[..., 1::2]
        return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)
      else:
        # Convert from HF's concatenated layout to MaxText's interleaved layout
        half_dim = arr.shape[-1] // 2
        first_half = arr[..., :half_dim]
        second_half = arr[..., half_dim:]
        return jax.numpy.stack([first_half, second_half], axis=-1).reshape(arr.shape)

    def reshape_kernel(input_tensor, target_shape):
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).transpose()
      else:
        return input_tensor.transpose().reshape(target_shape)

    # caveat: hook order does affect result
    # to_huggingface
    query_hook_chain = [scale_query_layer, adjust_rope, reshape_kernel]
    key_hook_chain = [adjust_rope, reshape_kernel]
    # to_maxtext
    if not saving_to_hf:
      query_hook_chain.reverse()
      key_hook_chain.reverse()

    hook_fns = {}

    hook_fns["params-decoder-logits_dense-kernel"] = reshape_kernel

    if scan_layers:
      hook_fns.update(
          {
              "params-decoder-layers-self_attention-query-kernel": query_hook_chain,
              "params-decoder-layers-self_attention-key-kernel": key_hook_chain,
              "params-decoder-layers-self_attention-value-kernel": reshape_kernel,
              "params-decoder-layers-self_attention-out-kernel": reshape_kernel,
              "params-decoder-layers-mlp-wi_0-kernel": reshape_kernel,
              "params-decoder-layers-mlp-wi_1-kernel": reshape_kernel,
              "params-decoder-layers-mlp-wo-kernel": reshape_kernel,
          }
      )
    else:
      for layer_idx in range(nlayers):
        hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-query-kernel"] = query_hook_chain
        hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-key-kernel"] = key_hook_chain
        hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-value-kernel"] = reshape_kernel
        hook_fns[f"params-decoder-layers_{layer_idx}-self_attention-out-kernel"] = reshape_kernel
        hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_0-kernel"] = reshape_kernel
        hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wi_1-kernel"] = reshape_kernel
        hook_fns[f"params-decoder-layers_{layer_idx}-mlp-wo-kernel"] = reshape_kernel
    return hook_fns

  def get_vllm_hooks(self) -> Dict[str, Callable]:
    """Defines and returns hook functions for NNX to VLLM weight transformations."""

    def reorder_rope(arr):
      """Reorders Rotary Position Embedding (RoPE) weights for vLLM."""
      evens = arr[..., ::2]
      odds = arr[..., 1::2]
      return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)

    def transform_query_kernel(arr):
      """Transforms the query kernel (scale + rope reorder)."""
      head_dim = arr.shape[-1]
      depth_scale = np.dtype("float32").type(np.sqrt(head_dim))
      arr = arr * depth_scale
      return reorder_rope(arr)

    hook_fns = {
        "base.decoder.layers.self_attention.query.kernel": transform_query_kernel,
        "base.decoder.layers.self_attention.key.kernel": reorder_rope,
    }
    return hook_fns
