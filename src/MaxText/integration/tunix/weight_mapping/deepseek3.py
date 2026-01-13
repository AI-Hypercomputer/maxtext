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
    # Hook for Input Projections (wq_b): Flattens last two dimensions
    # Source: (Latent, Heads, HeadDim) -> Target: (Latent, Heads * HeadDim)
    def flatten_input_proj(val):
      if val.ndim == 3:
        return val.reshape(val.shape[0], -1)
      return val

    # Hook for Output Projections (out): Flattens first two dimensions
    # Source: (Heads, HeadDim, Embed) -> Target: (Heads * HeadDim, Embed)
    def flatten_output_proj(val):
      if val.ndim == 3:
        return val.reshape(-1, val.shape[-1])
      return val

    # Hook for WKV_B: Slices the K component (first 128 dims)
    # Source: (Latent, Heads, K+V) -> Target: (Latent, Heads, K)
    def slice_wkv_b(val):
      # MaxText fused wkv_b is typically (512, 128, 256) [Latent, Heads, HeadDim]
      if val.ndim == 3 and val.shape[-1] == 256:
          return val[:, :, :128]
      return val

    return {
        # Dense Layers
        "base.decoder.dense_layers.self_attention.wq_b.kernel": flatten_input_proj,
        "base.decoder.dense_layers.self_attention.wkv_b.kernel": slice_wkv_b,
        "base.decoder.dense_layers.self_attention.out.kernel": flatten_output_proj,
        
        # MoE Layers
        "base.decoder.moe_layers.self_attention.wq_b.kernel": flatten_input_proj,
        "base.decoder.moe_layers.self_attention.wkv_b.kernel": slice_wkv_b,
        "base.decoder.moe_layers.self_attention.out.kernel": flatten_output_proj,
    }

  @staticmethod
  def to_hf_transpose_keys():
    return {}

  @staticmethod
  def lora_to_hf_mappings():
    return None

  @staticmethod
  def to_hf_mapping():
    # 1. Generate regex for layers
    # dense_tgt matches layers 0, 1, 2
    dense_tgt = f"layers\\.({'|'.join(map(str, range(0, 3)))})\\."
    # moe_tgt matches layers 3 to 60
    moe_tgt = f"layers\\.({'|'.join(map(str, range(3, 61)))})\\."

    # Strict regex to match EITHER the base key OR base.array.qvalue
    # Does NOT match .array.scale (preventing ShapeMismatch on scales)
    def q_path(path):
        return path + r"(?:\.array\.qvalue)?"

    mt_dense = "base.decoder.dense_layers"
    mt_moe = "base.decoder.moe_layers"

    mapping = {
        # Base Params
        "base.token_embedder.embedding": ("embedder.input_embedding_table_VD", ("model", None)),
        "base.decoder.decoder_norm.scale": ("final_norm.scale", (None,)),
        "base.decoder.logits_dense.kernel": ("lm_head.input_embedding_table_DV", (None, "model")),

        # ==============================================================================
        # DENSE LAYERS (0-2)
        # ==============================================================================
        f"{mt_dense}.pre_self_attention_layer_norm.scale": (f"{dense_tgt}pre_attention_norm.scale", (None, "layer")),
        f"{mt_dense}.post_self_attention_layer_norm.scale": (f"{dense_tgt}pre_mlp_norm.scale", (None, "layer")),
        f"{mt_dense}.self_attention.kv_norm.scale": (f"{dense_tgt}attn.kv_rms_norm.scale", (None, "layer")),
        f"{mt_dense}.self_attention.q_norm.scale": (f"{dense_tgt}attn.q_rms_norm.scale", (None, "layer")),
        
        # Attention
        f"{mt_dense}.self_attention.wq_a.kernel": (q_path(f"{dense_tgt}attn.kernel_q_down_proj_DA"), (None, "layer", "vocab")),
        f"{mt_dense}.self_attention.wq_b.kernel": (q_path(f"{dense_tgt}attn.kernel_q_up_proj_AP"), (None, "layer", "model")),
        f"{mt_dense}.self_attention.wkv_a.kernel": (q_path(f"{dense_tgt}attn.kernel_kv_down_proj_DA"), (None, "layer", "vocab")),
        
        # Map wkv_b to both K and V up-projections. 
        # The Regex `kernel_[kv]_up_proj_ANH` matches both 'kernel_k_...' and 'kernel_v_...'
        f"{mt_dense}.self_attention.wkv_b.kernel": (q_path(f"{dense_tgt}attn.kernel_[kv]_up_proj_ANH"), (None, "layer", "model")),
        
        f"{mt_dense}.self_attention.out.kernel": (q_path(f"{dense_tgt}attn.kernel_o_proj_RD"), ("model", "layer", None)),
        
        # MLP
        f"{mt_dense}.mlp.wi_0.kernel": (q_path(f"{dense_tgt}custom_module.kernel_gating_DF"), (None, "layer", "model")),
        f"{mt_dense}.mlp.wi_1.kernel": (q_path(f"{dense_tgt}custom_module.kernel_up_proj_DF"), (None, "layer", "model")),
        f"{mt_dense}.mlp.wo.kernel": (q_path(f"{dense_tgt}custom_module.kernel_down_proj_FD"), ("model", "layer", None)),

        # ==============================================================================
        # MOE LAYERS (3-60)
        # ==============================================================================
        f"{mt_moe}.pre_self_attention_layer_norm.scale": (f"{moe_tgt}pre_attention_norm.scale", (None, "layer")),
        f"{mt_moe}.post_self_attention_layer_norm.scale": (f"{moe_tgt}pre_mlp_norm.scale", (None, "layer")),
        f"{mt_moe}.self_attention.kv_norm.scale": (f"{moe_tgt}attn.kv_rms_norm.scale", (None, "layer")),
        f"{mt_moe}.self_attention.q_norm.scale": (f"{moe_tgt}attn.q_rms_norm.scale", (None, "layer")),

        # Attention
        f"{mt_moe}.self_attention.wq_a.kernel": (q_path(f"{moe_tgt}attn.kernel_q_down_proj_DA"), (None, "layer", "vocab")),
        f"{mt_moe}.self_attention.wq_b.kernel": (q_path(f"{moe_tgt}attn.kernel_q_up_proj_AP"), (None, "layer", "model")),
        f"{mt_moe}.self_attention.wkv_a.kernel": (q_path(f"{moe_tgt}attn.kernel_kv_down_proj_DA"), (None, "layer", "vocab")),
        
        # Map wkv_b to both K and V up-projections
        f"{mt_moe}.self_attention.wkv_b.kernel": (q_path(f"{moe_tgt}attn.kernel_[kv]_up_proj_ANH"), (None, "layer", "model")),

        f"{mt_moe}.self_attention.out.kernel": (q_path(f"{moe_tgt}attn.kernel_o_proj_RD"), ("model", "layer", None)),

        # Shared Experts
        f"{mt_moe}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel": (q_path(f"{moe_tgt}shared_experts.kernel_gating_DF"), (None, "layer", "model")),
        f"{mt_moe}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel": (q_path(f"{moe_tgt}shared_experts.kernel_up_proj_DF"), (None, "layer", "model")),
        f"{mt_moe}.DeepSeekMoeBlock_0.shared_experts.wo.kernel": (q_path(f"{moe_tgt}shared_experts.kernel_down_proj_FD"), ("model", "layer", None)),

        # Router
        f"{mt_moe}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel": (f"{moe_tgt}custom_module.router.kernel_DE", (None, "layer", "model")),
        f"{mt_moe}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias": (f"{moe_tgt}custom_module.router.bias_E", (None, "layer", "model")),

        # Routed Experts
        f"{mt_moe}.DeepSeekMoeBlock_0.MoeBlock_0.wi_0": (q_path(f"{moe_tgt}custom_module.kernel_gating_EDF"), ("expert", "layer", None, "model")),
        f"{mt_moe}.DeepSeekMoeBlock_0.MoeBlock_0.wi_1": (q_path(f"{moe_tgt}custom_module.kernel_up_proj_EDF"), ("expert", "layer", None, "model")),
        f"{mt_moe}.DeepSeekMoeBlock_0.MoeBlock_0.wo": (q_path(f"{moe_tgt}custom_module.kernel_down_proj_EFD"), ("expert", "layer", "model", None)),
    }

    return mapping