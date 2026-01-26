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

"""GPT-OSS family parameter mapping strategies."""

from typing import Any, Dict
import numpy as np
from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class GptOssMapper(ParamMapperStrategy):
  """Strategy for GPT-OSS models."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Generates mapping from MaxText gpt-oss to Hugging Face weight paths.

    Notes:
    - Handles the inhomogeneous scan block structure, based on `inhomogeneous_layer_cycle_interval`
    - Handles `composite_mt_key`: multiple MaxText keys map to HF key(s)
    """
    n_layers = hf_config["num_hidden_layers"]
    layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

    # Base mapping for non-layer parameters (targeting standard HF keys)
    mapping = {
        "params-token_embedder-embedding": "model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "model.norm.weight",
        "params-decoder-logits_dense-kernel": "lm_head.weight",
    }

    if scan_layers:
      # Scan over blocks
      for block_idx in range(layer_cycle_interval):
        # Identify all original HF layer indices that collapse into this block
        hf_indices = range(block_idx, n_layers, layer_cycle_interval)
        prefix = f"params-decoder-layers-layers_{block_idx}"
        block_mapping = {
            # Layer Norms
            f"{prefix}-pre_self_attention_layer_norm-scale": [
                f"model.layers.{i}.input_layernorm.weight" for i in hf_indices
            ],
            f"{prefix}-post_self_attention_layer_norm-scale": [
                f"model.layers.{i}.post_attention_layernorm.weight" for i in hf_indices
            ],
            # GptOssAttention
            f"{prefix}-GptOssAttention-query-kernel": [f"model.layers.{i}.self_attn.q_proj.weight" for i in hf_indices],
            f"{prefix}-GptOssAttention-query-bias": [f"model.layers.{i}.self_attn.q_proj.bias" for i in hf_indices],
            f"{prefix}-GptOssAttention-key-kernel": [f"model.layers.{i}.self_attn.k_proj.weight" for i in hf_indices],
            f"{prefix}-GptOssAttention-key-bias": [f"model.layers.{i}.self_attn.k_proj.bias" for i in hf_indices],
            f"{prefix}-GptOssAttention-value-kernel": [f"model.layers.{i}.self_attn.v_proj.weight" for i in hf_indices],
            f"{prefix}-GptOssAttention-value-bias": [f"model.layers.{i}.self_attn.v_proj.bias" for i in hf_indices],
            f"{prefix}-GptOssAttention-out-kernel": [f"model.layers.{i}.self_attn.o_proj.weight" for i in hf_indices],
            f"{prefix}-GptOssAttention-out-bias": [f"model.layers.{i}.self_attn.o_proj.bias" for i in hf_indices],
            f"{prefix}-GptOssAttention-sinks": [f"model.layers.{i}.self_attn.sinks" for i in hf_indices],
            # GptOssMlp
            # 1. Gate/Router
            f"{prefix}-GptOssMlp-gate-kernel": [f"model.layers.{i}.mlp.router.weight" for i in hf_indices],
            f"{prefix}-GptOssMlp-gate-bias": [f"model.layers.{i}.mlp.router.bias" for i in hf_indices],
            # 2. Experts (Down Projection)
            f"{prefix}-GptOssMlp-wo": [f"model.layers.{i}.mlp.experts.down_proj" for i in hf_indices],
            f"{prefix}-GptOssMlp-wo_bias": [f"model.layers.{i}.mlp.experts.down_proj_bias" for i in hf_indices],
            # 3. Experts (Gate/Up Fused Projection)
            # `composite_mt_key`: Multiple MaxText keys map to HF key(s).
            (f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1"): [
                f"model.layers.{i}.mlp.experts.gate_up_proj" for i in hf_indices
            ],
            (f"{prefix}-GptOssMlp-wi_0_bias", f"{prefix}-GptOssMlp-wi_1_bias"): [
                f"model.layers.{i}.mlp.experts.gate_up_proj_bias" for i in hf_indices
            ],
        }
        mapping.update(block_mapping)

    else:
      # Unscan
      for i in range(n_layers):
        prefix = f"params-decoder-layers_{i}"
        layer_mapping = {
            # Layer Norms
            f"{prefix}-pre_self_attention_layer_norm-scale": f"model.layers.{i}.input_layernorm.weight",
            f"{prefix}-post_self_attention_layer_norm-scale": f"model.layers.{i}.post_attention_layernorm.weight",
            # GptOssAttention
            f"{prefix}-GptOssAttention-query-kernel": f"model.layers.{i}.self_attn.q_proj.weight",
            f"{prefix}-GptOssAttention-query-bias": f"model.layers.{i}.self_attn.q_proj.bias",
            f"{prefix}-GptOssAttention-key-kernel": f"model.layers.{i}.self_attn.k_proj.weight",
            f"{prefix}-GptOssAttention-key-bias": f"model.layers.{i}.self_attn.k_proj.bias",
            f"{prefix}-GptOssAttention-value-kernel": f"model.layers.{i}.self_attn.v_proj.weight",
            f"{prefix}-GptOssAttention-value-bias": f"model.layers.{i}.self_attn.v_proj.bias",
            f"{prefix}-GptOssAttention-out-kernel": f"model.layers.{i}.self_attn.o_proj.weight",
            f"{prefix}-GptOssAttention-out-bias": f"model.layers.{i}.self_attn.o_proj.bias",
            f"{prefix}-GptOssAttention-sinks": f"model.layers.{i}.self_attn.sinks",
            # GptOssMlp
            # 1. Gate/Router
            f"{prefix}-GptOssMlp-gate-kernel": f"model.layers.{i}.mlp.router.weight",
            f"{prefix}-GptOssMlp-gate-bias": f"model.layers.{i}.mlp.router.bias",
            # 2. Experts (Down Projection)
            f"{prefix}-GptOssMlp-wo": f"model.layers.{i}.mlp.experts.down_proj",
            f"{prefix}-GptOssMlp-wo_bias": f"model.layers.{i}.mlp.experts.down_proj_bias",
            # 3. Experts (Gate/Up Fused Projection)
            # `composite_mt_key`: Multiple MaxText keys map to HF key(s).
            (f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1"): f"model.layers.{i}.mlp.experts.gate_up_proj",
            (
                f"{prefix}-GptOssMlp-wi_0_bias",
                f"{prefix}-GptOssMlp-wi_1_bias",
            ): f"model.layers.{i}.mlp.experts.gate_up_proj_bias",
        }
        mapping.update(layer_mapping)

    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Transformation hooks for gpt-oss parameters."""

    def transpose(input_tensor, target_shape=None):
      return input_tensor.T

    def reshape_kernel(input_tensor, target_shape):
      """Reshapes and transposes kernel weights between MaxText and HF."""
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      else:
        return input_tensor.T.reshape(target_shape)

    def reshape_bias(input_tensor, target_shape=None):
      """Reshapes biases between MaxText 2D (heads, dim) and HF 1D (hidden)."""
      if saving_to_hf:
        # MaxText [heads, head_dim] -> HF [hidden_dim] (flatten)
        return input_tensor.reshape(target_shape)
      else:
        # HF [hidden_dim] -> MaxText [heads, head_dim]
        return input_tensor.reshape(target_shape)

    def interleave(input_tensor, target_shape=None):
      """
      Handles `composite_mt_key`: maxtext (wi_0, wi_1) <-> hf (wi_0_1)
      - if saving_to_hf: (wi_0, wi_1) -> wi_0_1
        - input_tensor is a list of two tensors, tensor ORDER must be same as key order
        - return a single tensor
      - otherwise: wi_0_1 -> (wi_0, wi_1)
        - input_tensor is a single tensor
        - return two tensors stack at LAST index -1, tensor ORDER must be same as key order
      """
      if saving_to_hf:
        wi_0, wi_1 = input_tensor
        wi_0_1 = np.empty(target_shape, dtype=wi_0.dtype)
        wi_0_1[..., ::2] = wi_0
        wi_0_1[..., 1::2] = wi_1
        return wi_0_1
      else:
        wi_0_1 = input_tensor
        wi_0 = wi_0_1[..., ::2]
        wi_1 = wi_0_1[..., 1::2]
        return np.stack([wi_0, wi_1], axis=-1)

    n_layers = hf_config["num_hidden_layers"]
    layer_cycle_interval = maxtext_config.inhomogeneous_layer_cycle_interval

    hooks = {"params-decoder-logits_dense-kernel": transpose}

    indices = range(layer_cycle_interval) if scan_layers else range(n_layers)
    for idx in indices:
      prefix = f"params-decoder-layers-layers_{idx}" if scan_layers else f"params-decoder-layers_{idx}"
      # Attention Kernels & Biases
      for key in ["query", "key", "value"]:
        hooks[f"{prefix}-GptOssAttention-{key}-kernel"] = reshape_kernel
        hooks[f"{prefix}-GptOssAttention-{key}-bias"] = reshape_bias
      hooks[f"{prefix}-GptOssAttention-out-kernel"] = reshape_kernel
      # MLP Kernels & Biases
      hooks[f"{prefix}-GptOssMlp-gate-kernel"] = transpose
      # `composite_mt_key`: A hook for combining multiple MaxText params.
      hooks[(f"{prefix}-GptOssMlp-wi_0", f"{prefix}-GptOssMlp-wi_1")] = interleave
      hooks[(f"{prefix}-GptOssMlp-wi_0_bias", f"{prefix}-GptOssMlp-wi_1_bias")] = interleave

    return hooks
