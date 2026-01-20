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

"""DeepSeek family parameter mapping strategies."""

from typing import Any, Dict, Callable

import numpy as np

from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy


class DeepSeekMapper(ParamMapperStrategy):
  """Strategy for DeepSeek models (e.g. DeepSeek-V3)."""

  def get_mapping(self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool) -> Dict[str, Any]:
    """Generates a parameter mapping from MaxText to HuggingFace Deepseek weight paths.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.

    Returns:
      A dictionary mapping keys to HuggingFace parameter names.
    """
    # TODO(shuningjin): add unscan support, b/457820735
    if not scan_layers:
      raise NotImplementedError("This conversion only supports scanned MaxText models.")

    # Extract hf configuration parameters, without mtp
    num_main_layers = hf_config["num_hidden_layers"]
    first_num_dense_layers = hf_config["first_k_dense_replace"]
    num_experts = hf_config.get("n_routed_experts", 0)

    # Mapping for non-layer-specific weights
    mapping = {
        "params-token_embedder-embedding": "model.embed_tokens.weight",
        "params-decoder-decoder_norm-scale": "model.norm.weight",
        "params-decoder-logits_dense-kernel": "lm_head.weight",
    }
    # Attention keys are shared by both dense and MoE
    attention_keys = {
        "pre_self_attention_layer_norm-scale": "input_layernorm.weight",
        "post_self_attention_layer_norm-scale": "post_attention_layernorm.weight",
        "self_attention-wkv_a-kernel": "self_attn.kv_a_proj_with_mqa.weight",
        "self_attention-kv_norm-scale": "self_attn.kv_a_layernorm.weight",
        "self_attention-wkv_b-kernel": "self_attn.kv_b_proj.weight",
        "self_attention-out-kernel": "self_attn.o_proj.weight",
        # v3
        "self_attention-wq_a-kernel": "self_attn.q_a_proj.weight",
        "self_attention-q_norm-scale": "self_attn.q_a_layernorm.weight",
        "self_attention-wq_b-kernel": "self_attn.q_b_proj.weight",
        # v2
        "self_attention-query-kernel": "self_attn.q_proj.weight",
    }
    # Dense Layers
    dense_layer_keys = attention_keys | {
        "mlp-wi_0-kernel": "mlp.gate_proj.weight",
        "mlp-wi_1-kernel": "mlp.up_proj.weight",
        "mlp-wo-kernel": "mlp.down_proj.weight",
    }
    for maxtext_key, hf_key in dense_layer_keys.items():
      mapping[f"params-decoder-dense_layers-{maxtext_key}"] = [
          f"model.layers.{i}.{hf_key}" for i in range(first_num_dense_layers)
      ]

    # MoE Layers
    moe_layer_keys = attention_keys | {
        "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel": "mlp.shared_experts.gate_proj.weight",
        "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel": "mlp.shared_experts.up_proj.weight",
        "DeepSeekMoeBlock_0-shared_experts-wo-kernel": "mlp.shared_experts.down_proj.weight",
        "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel": "mlp.gate.weight",
        # v3
        "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias": "mlp.gate.e_score_correction_bias",
    }
    for maxtext_key, hf_key in moe_layer_keys.items():
      mapping[f"params-decoder-moe_layers-{maxtext_key}"] = [
          f"model.layers.{i}.{hf_key}" for i in range(first_num_dense_layers, num_main_layers)
      ]

    # MoE Experts (nested list mapping: [[e0_l0, e0_l1..], [e1_l0, e1_l1..]..])
    moe_expert_keys = {
        "DeepSeekMoeBlock_0-MoeBlock_0-wi_0": "gate_proj.weight",
        "DeepSeekMoeBlock_0-MoeBlock_0-wi_1": "up_proj.weight",
        "DeepSeekMoeBlock_0-MoeBlock_0-wo": "down_proj.weight",
    }
    for maxtext_key, hf_key in moe_expert_keys.items():
      mapping[f"params-decoder-moe_layers-{maxtext_key}"] = [
          [f"model.layers.{l}.mlp.experts.{e}.{hf_key}" for l in range(first_num_dense_layers, num_main_layers)]
          for e in range(num_experts)
      ]
    return mapping

  def get_hooks(
      self, hf_config: Dict[str, Any], maxtext_config: Any, scan_layers: bool, saving_to_hf: bool
  ) -> Dict[str, Any]:
    """Creates parameter transformation functions for Deepseek.

    Args:
      hf_config: The HuggingFace configuration dictionary.
      maxtext_config: The MaxText configuration object.
      scan_layers: Boolean indicating if layers are scanned.
      saving_to_hf: Boolean indicating direction (True for MT->HF, False for HF->MT).

    Returns:
      A dictionary mapping keys to hook functions.
    """
    # TODO(shuningjin): support hf->orbax(scan), b/457820372
    if not saving_to_hf:
      raise NotImplementedError("This conversion only supports saving_to_hf")
    # TODO(shuningjin): add unscan support, b/457820735
    if not scan_layers:
      raise NotImplementedError("This conversion only supports scanned MaxText models.")

    def reshape_kernel(input_tensor, target_shape):
      """Reshapes and transposes kernel weights between MaxText and HF."""
      if saving_to_hf:
        flipped_target_shape = np.flip(np.array(target_shape))
        return input_tensor.reshape(flipped_target_shape).T
      else:
        return input_tensor.T.reshape(target_shape)

    mapping = {
        "params-decoder-logits_dense-kernel": reshape_kernel,
    }
    # all keys that need the reshape hook
    params_need_reshape = {
        # Dense Layers
        "params-decoder-dense_layers-self_attention-query-kernel",
        "params-decoder-dense_layers-self_attention-wq_a-kernel",
        "params-decoder-dense_layers-self_attention-wq_b-kernel",
        "params-decoder-dense_layers-self_attention-wkv_a-kernel",
        "params-decoder-dense_layers-self_attention-wkv_b-kernel",
        "params-decoder-dense_layers-self_attention-out-kernel",
        "params-decoder-dense_layers-mlp-wi_0-kernel",
        "params-decoder-dense_layers-mlp-wi_1-kernel",
        "params-decoder-dense_layers-mlp-wo-kernel",
        # MoE Layers
        "params-decoder-moe_layers-self_attention-query-kernel",
        "params-decoder-moe_layers-self_attention-wq_a-kernel",
        "params-decoder-moe_layers-self_attention-wq_b-kernel",
        "params-decoder-moe_layers-self_attention-wkv_a-kernel",
        "params-decoder-moe_layers-self_attention-wkv_b-kernel",
        "params-decoder-moe_layers-self_attention-out-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-shared_experts-wi_0-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-shared_experts-wi_1-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-shared_experts-wo-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_0",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wi_1",
        "params-decoder-moe_layers-DeepSeekMoeBlock_0-MoeBlock_0-wo",
    }

    for key in params_need_reshape:
      mapping[key] = reshape_kernel
    return mapping

  def get_vllm_hooks(self) -> Dict[str, Callable]:
    """Creates parameter transformation functions for Deepseek VLLM integration.

    Returns:
      A dictionary mapping keys to hook functions.
    """
    return {}
