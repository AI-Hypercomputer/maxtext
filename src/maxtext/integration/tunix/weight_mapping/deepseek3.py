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

"""Mapping MaxText Deepseek (MoE) weights to vLLM/tpu-inference keys."""

from dataclasses import dataclass


@dataclass
class DEEPSEEK_VLLM_MAPPING:
  """Mapping MaxText Deepseek-V3 weights to Tunix/vLLM NNX keys."""

  @staticmethod
  def to_hf_hook_fns():
    def flatten_3d_to_2d(val):
      # Converts (Rank, Heads, HeadDim) -> (Rank, Heads * HeadDim)
      if val.ndim == 3:
        return val.reshape(val.shape[0], -1)
      return val

    return {
        # MaxText MLA weights are 3D (Rank, Heads, HeadDim).
        # tpu-inference expects 2D (Rank, Heads*HeadDim) before it splits them.
        "base.decoder.layers.self_attention.wq_b.kernel": flatten_3d_to_2d,
        "base.decoder.layers.self_attention.wkv_b.kernel": flatten_3d_to_2d,
        "base.decoder.layers.self_attention.out.kernel": flatten_3d_to_2d,
    }

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
    """Returns the weight mapping for the model."""
    mapping = {
        # --- Base Model Params ---
        # Map to HF names to be safe with loader regexes
        "base.token_embedder.embedding": ("model.embed_tokens.weight", ("model", None)),
        "base.decoder.decoder_norm.scale": ("model.norm.weight", (None,)),
        "base.decoder.logits_dense.kernel": ("lm_head.weight", (None, "model")),
        # MLA LAYERS (Map to HF Keys to trigger loader splitting logic)
        # Norms
        "base.decoder.layers.pre_self_attention_layer_norm.scale": (
            "model.layers.*.input_layernorm.weight",
            (None, "layer"),
        ),
        "base.decoder.layers.post_self_attention_layer_norm.scale": (
            "model.layers.*.post_attention_layernorm.weight",
            (None, "layer"),
        ),
        # MLA Norms
        "base.decoder.layers.self_attention.kv_norm.scale": (
            "model.layers.*.self_attn.kv_a_layernorm.weight",
            (None, "layer"),
        ),
        "base.decoder.layers.self_attention.q_norm.scale": (
            "model.layers.*.self_attn.q_a_layernorm.weight",
            (None, "layer"),
        ),
        # MLA Projections
        # We use HF names here so `DeepSeekV3WeightLoader` detects "kv_b_proj"
        # and performs the necessary split into k_b and v_b for the MLA kernel.
        "base.decoder.layers.self_attention.wq_a.kernel": (
            "model.layers.*.self_attn.q_a_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.wq_b.kernel": (
            "model.layers.*.self_attn.q_b_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.wkv_a.kernel": (
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.wkv_b.kernel": (
            "model.layers.*.self_attn.kv_b_proj.weight",
            (None, "layer", "model", None),
        ),
        "base.decoder.layers.self_attention.out.kernel": (
            "model.layers.*.self_attn.o_proj.weight",
            ("model", "layer", None, None),
        ),
        # DENSE MLP LAYERS (Map to vllm keys for safety/consistency)
        "base.decoder.layers.mlp.wi_0.kernel": ("model.layers.*.mlp.gate_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wi_1.kernel": ("model.layers.*.mlp.up_proj.weight", (None, "layer", "model")),
        "base.decoder.layers.mlp.wo.kernel": ("model.layers.*.mlp.down_proj.weight", ("model", "layer", None)),
        # MOE LAYERS (Map to INTERNAL keys to bypass loader stacking)
        # Since MaxText experts are already fused/stacked, we map directly to the
        # internal `tpu-inference` param names. The loader will fail to find
        # "experts.{i}" in the name and fall back to loading these directly,
        # which is exactly what we want for performance.
        # Shared Experts
        "base.decoder.layers.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel": (
            "layers.*.shared_experts.kernel_gating_DF",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel": (
            "layers.*.shared_experts.kernel_up_proj_DF",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.DeepSeekMoeBlock_0.shared_experts.wo.kernel": (
            "layers.*.shared_experts.kernel_down_proj_FD",
            ("model", "layer", None),
        ),
        # Router
        "base.decoder.layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel": (
            "layers.*.custom_module.router.kernel_DE",
            (None, "layer", "model"),
        ),
        "base.decoder.layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias": (
            "layers.*.custom_module.router.bias_E",
            (None, "layer", "model"),
        ),
        # Routed Experts (Fused)
        "base.decoder.layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_0": (
            "layers.*.custom_module.kernel_gating_EDF",
            ("expert", "layer", None, "model"),
        ),
        "base.decoder.layers.DeepSeekMoeBlock_0.MoeBlock_0.wi_1": (
            "layers.*.custom_module.kernel_up_proj_EDF",
            ("expert", "layer", None, "model"),
        ),
        "base.decoder.layers.DeepSeekMoeBlock_0.MoeBlock_0.wo": (
            "layers.*.custom_module.kernel_down_proj_EFD",
            ("expert", "layer", "model", None),
        ),
        # MTP BLOCK (Included for completeness, but typically skipped by current loader)
        "base.mtp_block.mtp_layer_1.embedding_norm.scale": ("mtp_block.layer.pre_norm.scale", (None,)),
        "base.mtp_block.mtp_layer_1.hidden_state_norm.scale": ("mtp_block.layer.post_norm.scale", (None,)),
        "base.mtp_block.mtp_layer_1.projection_layer.kernel": ("mtp_block.layer.projection.kernel", (None, "model")),
    }
    return mapping
