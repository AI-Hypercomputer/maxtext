# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLOPs calculation per device for GLM-5.3-Flash training."""

from typing import Any
from maxtext.m3.models.glm5_3_flash import glm5_config


def calculate_glm5_3_flash_tflops_per_device(
    config: Any,
    log: bool = True,
) -> tuple[float, float, float]:
  """Calculates per-device training TFLOPs for the hybrid GLM-5.3-Flash model.

  Accounts for:
  - 4-stream Manifold-Constrained Hyper-Connections (mHC)
  - Hybrid 34 KDA (linear attention) and 11 NoPE-DSA-MLA (full attention
    with DSA indexer)
  - 3 Dense + 42 MoE (top-8 routed + 1 shared expert) clamped-SwiGLU layers
  - Optional Layer-45 NextN Multi-Token Prediction (MTP) block
  - Vocabulary unembedding projection

  Note on DSA Attention & Indexer FLOPs:
  - When use_indexer=True and seq_len > index_topk, glm5_flops models
    theoretical sparse DSA attention FLOPs (multiplier = ratio - 0.5 * ratio^2),
    whereas the current dense_mask reference implementation in
    modeling_glm5_3_flash.py
    materializes dense S x S attention scores before applying the top-k mask.
  - The frozen DSA indexer (stop_gradient in JAX and @torch.no_grad() in HF)
    has no backward pass and is counted at 1.0x (forward-only) rather than 3.0x.

  Args:
    config: A MaxText configuration object.
    log: Whether to print FLOPs breakdown.

  Returns:
    (total_tflops, learnable_weight_tflops, attention_tflops)
  """
  glm_cfg = glm5_config.Glm5Config.from_maxtext_config(config)

  b_dim = config.per_device_batch_size
  s_dim = config.max_target_length
  batch_len = b_dim * s_dim
  emb_dim = glm_cfg.hidden_size
  vocab_size = glm_cfg.vocab_size

  num_layers = glm_cfg.num_hidden_layers
  if glm_cfg.layer_types is None:
    raise ValueError("layer_types must not be None on Glm5Config.")
  layer_types = glm_cfg.layer_types
  num_kda_layers = sum(1 for lt in layer_types if lt == "linear_attention")
  num_mla_layers = sum(1 for lt in layer_types if lt == "full_attention")

  first_k_dense = glm_cfg.first_k_dense_replace
  num_dense_layers = min(num_layers, first_k_dense)
  num_moe_layers = max(0, num_layers - num_dense_layers)

  # 1. Unembedding (FLOPs for output projection)
  embedding_flops = 2 * batch_len * emb_dim * vocab_size

  # 2. 4-stream mHC (2 mHC blocks per layer: attn_hc and ffn_hc)
  hc_mult = glm_cfg.mhc_expansion_rate
  mix = (2 + hc_mult) * hc_mult
  mhc_block_flops = 2 * batch_len * emb_dim * (hc_mult * mix + hc_mult * hc_mult)
  total_mhc_flops = 2 * mhc_block_flops * num_layers

  # 3. FFN FLOPs (Dense + MoE with clamped SwiGLU)
  intermediate_size = glm_cfg.intermediate_size
  dense_ffn_per_layer = 6 * batch_len * emb_dim * intermediate_size
  total_dense_ffn_flops = num_dense_layers * dense_ffn_per_layer

  moe_intermediate_size = glm_cfg.moe_intermediate_size
  n_routed_experts = glm_cfg.n_routed_experts
  num_experts_per_tok = glm_cfg.num_experts_per_tok
  n_shared_experts = glm_cfg.n_shared_experts

  router_gate_flops = 2 * batch_len * emb_dim * n_routed_experts
  shared_expert_flops = n_shared_experts * (6 * batch_len * emb_dim * moe_intermediate_size)
  routed_expert_flops = num_experts_per_tok * (6 * batch_len * emb_dim * moe_intermediate_size)
  moe_ffn_per_layer = router_gate_flops + shared_expert_flops + routed_expert_flops
  total_moe_ffn_flops = num_moe_layers * moe_ffn_per_layer

  # 4. KDA Attention (Linear Attention)
  num_k_heads = glm_cfg.linear_num_key_heads
  num_v_heads = glm_cfg.linear_num_value_heads
  k_head_dim = glm_cfg.linear_key_head_dim
  v_head_dim = glm_cfg.linear_value_head_dim
  conv_kernel_dim = glm_cfg.linear_conv_kernel_dim
  k_dim = num_k_heads * k_head_dim
  v_dim = num_v_heads * v_head_dim

  kda_proj_flops = (
      2 * batch_len * (emb_dim * (2 * k_dim + 2 * v_dim + num_k_heads + 2 * k_head_dim) + k_head_dim * (k_dim + v_dim))
  )
  kda_conv_flops = 2 * batch_len * conv_kernel_dim * (2 * k_dim + v_dim)
  kda_weight_flops_per_layer = kda_proj_flops + kda_conv_flops
  kda_attn_flops_per_layer = 8 * batch_len * num_v_heads * k_head_dim * v_head_dim
  total_kda_weight_flops = num_kda_layers * kda_weight_flops_per_layer
  total_kda_attn_flops = num_kda_layers * kda_attn_flops_per_layer

  # 5. NoPE-DSA-MLA Attention (Full Attention with DSA Lightning Indexer)
  num_attn_heads = glm_cfg.num_attention_heads
  q_lora_rank = glm_cfg.q_lora_rank
  kv_lora_rank = glm_cfg.kv_lora_rank
  qk_nope_head_dim = glm_cfg.qk_nope_head_dim
  qk_rope_head_dim = glm_cfg.qk_rope_head_dim
  mla_v_head_dim = glm_cfg.v_head_dim
  qk_head_dim_sum = qk_nope_head_dim + qk_rope_head_dim

  q_flops = 2 * batch_len * (emb_dim * q_lora_rank + q_lora_rank * num_attn_heads * qk_head_dim_sum)
  kv_flops = (
      2
      * batch_len
      * (
          emb_dim * (kv_lora_rank + qk_rope_head_dim)
          + kv_lora_rank * num_attn_heads * (qk_nope_head_dim + mla_v_head_dim)
      )
  )
  out_flops = 2 * batch_len * (num_attn_heads * mla_v_head_dim) * emb_dim
  mla_weight_flops_per_layer = q_flops + kv_flops + out_flops

  use_indexer = glm_cfg.use_indexer
  index_topk = glm_cfg.index_topk
  index_n_heads = glm_cfg.index_n_heads
  index_head_dim = glm_cfg.index_head_dim

  if use_indexer and s_dim > index_topk:
    indexer_proj = (
        2
        * batch_len
        * (q_lora_rank * index_n_heads * index_head_dim + 2 * emb_dim * index_head_dim + emb_dim * index_n_heads)
    )
    indexer_weight_flops_per_layer = float(indexer_proj)
    ratio = index_topk / s_dim
    multiplier = ratio - 0.5 * ratio**2
    indexer_qk_flops = (2 * batch_len * s_dim * index_n_heads * index_head_dim) / glm_cfg.index_kpool
    indexer_weight_flops = (2 * batch_len * s_dim * index_n_heads) / glm_cfg.index_kpool
    indexer_scoring = (indexer_qk_flops + indexer_weight_flops) / 2
    indexer_attn_flops_per_layer = float(indexer_scoring)
    mla_attn_flops_per_layer = 2 * batch_len * s_dim * num_attn_heads * (qk_head_dim_sum + mla_v_head_dim) * multiplier
  else:
    indexer_weight_flops_per_layer = 0.0
    indexer_attn_flops_per_layer = 0.0
    mla_attn_flops_per_layer = (2 * batch_len * s_dim * num_attn_heads * (qk_head_dim_sum + mla_v_head_dim)) / 2

  total_mla_weight_flops = num_mla_layers * mla_weight_flops_per_layer
  total_mla_attn_flops = num_mla_layers * mla_attn_flops_per_layer

  # Combine and convert to TFLOPs (multiply by 3 for fwd + 2 bwd)
  total_learnable_weight_flops = (
      embedding_flops
      + total_mhc_flops
      + total_dense_ffn_flops
      + total_moe_ffn_flops
      + total_kda_weight_flops
      + total_mla_weight_flops
  )
  total_trainable_attn_flops = total_kda_attn_flops + total_mla_attn_flops

  # 6. NextN Multi-Token Prediction (MTP) block FLOPs (no mHC; eh_proj instead)
  mtp_indexer_weight_flops = 0.0
  mtp_indexer_attn_flops = 0.0
  if glm_cfg.num_nextn_predict_layers > 0:
    mtp_layers = glm_cfg.num_nextn_predict_layers
    eh_proj_flops = 2 * batch_len * (2 * emb_dim) * emb_dim
    mtp_weight_flops = mtp_layers * (eh_proj_flops + mla_weight_flops_per_layer + moe_ffn_per_layer + embedding_flops)
    mtp_attn_flops = mtp_layers * mla_attn_flops_per_layer
    total_learnable_weight_flops += mtp_weight_flops
    total_trainable_attn_flops += mtp_attn_flops
    mtp_indexer_weight_flops = mtp_layers * indexer_weight_flops_per_layer
    mtp_indexer_attn_flops = mtp_layers * indexer_attn_flops_per_layer

  total_indexer_weight_flops = num_mla_layers * indexer_weight_flops_per_layer + mtp_indexer_weight_flops
  total_indexer_attn_flops = num_mla_layers * indexer_attn_flops_per_layer + mtp_indexer_attn_flops

  grad_accum = config.gradient_accumulation_steps
  learnable_weight_tflops = ((3.0 * total_learnable_weight_flops + total_indexer_weight_flops) / 10**12) * grad_accum
  attention_tflops = ((3.0 * total_trainable_attn_flops + total_indexer_attn_flops) / 10**12) * grad_accum
  total_tflops = learnable_weight_tflops + attention_tflops

  if log:
    print(
        f"GLM-5.3-Flash per train step:\nTotal TFLOPs: {total_tflops:.2f}"
        " \nsplit as"
        f" {100 * learnable_weight_tflops / max(total_tflops, 1e-12):.2f}%"
        " learnable weight flops and"
        f" {100 * attention_tflops / max(total_tflops, 1e-12):.2f}% attention"
        f" flops;\nlearnable weight {learnable_weight_tflops:.2f} TFLOPs,"
        f" attention {attention_tflops:.2f} TFLOPs"
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
