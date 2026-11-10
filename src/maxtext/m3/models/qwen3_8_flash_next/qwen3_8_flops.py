# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLOPs calculation per device for Qwen3.8-Flash-Next training.

Registered in `maxtext.m3.models.MODEL_FLOPS_REGISTRY`, so
`maxtext_utils.calculate_tflops_training_per_device` dispatches here for
`use_m3_model=true` runs of `qwen3.8-flash-next` and `qwen3.8-flash-next-mini`
(mirroring `glm5_3_flash/glm5_flops.py`).

Self-contained on purpose: `utils:maxtext_utils` already depends on `m3/models`,
so importing it from here would be a dependency cycle. The MoE FFN, Gated
DeltaNet, indexer mask-ratio and unembedding terms therefore replicate the
legacy helpers (`calculate_routed_and_shared_ffn_tflops_per_device`,
`calculate_gated_delta_net_flops_per_device`, `calculate_indexer_mask_ratio`),
with the Qwen3.8 shared expert sized by `mlp_dim`
(HF `shared_expert_intermediate_size`) rather than `moe_mlp_dim`.
"""

from typing import Any


def _ffn_matmul_flops(config: Any, mlp_dim: int) -> float:
  """Forward matmul FLOPs of one gated FFN at width `mlp_dim`: `len(mlp_activations)` up-projections + 1 down."""
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  ffn1_flops = 2 * B * S * mlp_dim * E * len(config.mlp_activations)
  ffn2_flops = 2 * B * S * mlp_dim * E
  return ffn1_flops + ffn2_flops


def _moe_ffn_flops_all_layers(config: Any) -> float:
  """Router gate + shared expert (`mlp_dim`) + top-k routed experts (`moe_mlp_dim`), summed over all decoder layers.

  Every Qwen3.8-Flash-Next decoder layer is an MoE layer, so there is no dense-FFN term.
  """
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  gate_flops = 2 * B * S * E * config.num_experts
  shared_experts_flops = _ffn_matmul_flops(config, config.mlp_dim) * config.shared_experts
  routed_experts_flops = _ffn_matmul_flops(config, config.moe_mlp_dim) * config.num_experts_per_tok
  return (gate_flops + shared_experts_flops + routed_experts_flops) * config.num_decoder_layers


def _gated_delta_net_flops_per_layer(config: Any) -> tuple[float, float]:
  """FLOPs for a single Gated Delta Net (linear attention) layer, split into (weight, attention) parts.

  Ref: Megatron's gated-delta-net accounting
  https://github.com/NVIDIA/Megatron-LM/blob/8f1c2f8ae53b4e3f32c0ae7f397d8b38a675eaa2/megatron/training/training.py#L513
  Core complexity follows the recurrent state-update view (4 ops * 2 FLOPs = 8).
  """
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim

  H_k = config.gdn_num_key_heads
  H_v = config.gdn_num_value_heads
  D_k = config.gdn_key_head_dim
  D_v = config.gdn_value_head_dim
  K_conv = config.gdn_conv_kernel_dim

  K_dim = H_k * D_k
  V_dim = H_v * D_v

  # 1. Projections (learnable weights): in_proj_qkvz (2*K + 2*V) + in_proj_ba (2*H_v), x2 for FMA.
  flops_qkvz_ba = 2 * B * S * E * (2 * K_dim + 2 * V_dim + 2 * H_v)
  # out_proj
  flops_out = 2 * B * S * E * V_dim
  flops_projections = flops_qkvz_ba + flops_out

  # 2. Depthwise convolution (learnable weights), x2 for FMA.
  flops_conv = 2 * B * S * K_conv * (2 * K_dim + V_dim)

  # 3. Core Gated Delta Net: 4 O(D^2) ops per token in the recurrent update
  #    (KK^T, VK^T, S(a(I-bKK^T)), SQ), x2 for FMA -> 8 * H * D^2 per token.
  flops_core_per_token = H_v * (D_k * D_v) * 8
  flops_core = B * S * flops_core_per_token

  gdn_weight_flops = flops_projections + flops_conv
  gdn_attn_flops = flops_core
  return gdn_weight_flops, gdn_attn_flops


def _indexer_mask_ratio(indexer_topk: int, max_target_length: int) -> float:
  """Sparse-to-dense ratio of a causal top-K indexer: (T*K - 0.5*K^2) / T^2 = (K/T) - 0.5*(K/T)^2."""
  T = float(max_target_length)
  K = float(indexer_topk)
  ratio = K / T
  return ratio - (0.5 * ratio**2)


def calculate_qwen3_8_flash_next_tflops_per_device(config: Any, log: bool = True) -> tuple[float, float, float]:
  """Calculates per-device training TFLOPs for the Qwen3.8-Flash-Next hybrid architecture.

  Accounts for:
  - Interleaved 3:1 GatedDeltaNet (GDN) linear attention + QSA (Gated Query + MQA
    Micro-Block Indexer) sparse attention layers.
  - 4-stream Hyper-Connections (HCBlock around attention and MoE per decoder layer
    + final HCMixer).
  - Layer-2 Per-Layer Embedding (PLE) block (`w_ple`, `gate_proj`, causal conv1d).
  - 512-expert MoE with sigmoid-gated shared expert (`shared_expert_gate`) in every block.
  - Multi-Token Prediction (MTP) block (`mtp_fc` projection over `[inputs_embeds, 4-stream hidden]`
    + 1 full QSA-MoE decoder layer + `mtp_hc_mixer` + shared `lm_head` unembedding matmul).
  - Vocabulary unembedding projection (the embedding lookup is a gather with no dense math).

  Like `maxtext_utils.calculate_tflops_training_per_device`, the result is multiplied by 3 for
  forward + backward, scaled by `gradient_accumulation_steps`, and includes the extra DPO
  reference-model forward pass when `use_dpo` is set.

  Args:
    config: A MaxText configuration object.
    log: Whether to print the FLOPs breakdown.

  Returns:
    (total_tflops, learnable_weight_tflops, attention_tflops)
  """
  B = config.per_device_batch_size
  S = config.max_target_length
  E = config.emb_dim
  num_layers = config.num_decoder_layers
  cycle_interval = config.inhomogeneous_layer_cycle_interval
  num_qsa_layers = num_layers // cycle_interval
  num_gdn_layers = num_layers - num_qsa_layers

  # MoE FFN across all layers, and the unembedding projection.
  total_ffn_flops_all_layers = _moe_ffn_flops_all_layers(config)
  embedding_flops = 2 * B * S * E * config.vocab_size

  # 1. Gated Delta Net (Linear Attention) FLOPs per GDN layer
  gdn_weight_flops_per_layer, gdn_attn_flops_per_layer = _gated_delta_net_flops_per_layer(config)

  # 2. QSA (Gated Query Attention + MQA Micro-Block Indexer) FLOPs per QSA layer
  H_q = config.num_query_heads
  H_kv = config.num_kv_heads
  D_head = config.head_dim
  index_n_heads = getattr(config, "index_n_heads", 0)
  index_kv_heads = getattr(config, "index_kv_heads", 1)
  index_head_dim = getattr(config, "index_head_dim", 0)
  compress_ratio = getattr(config, "index_select_block_size", 4) or 4
  token_budget = getattr(config, "index_select_budget", 0)

  # QSA projections:
  # - q_proj projects E -> 2 * H_q * D_head (includes output gate)
  # - k_proj + v_proj project E -> 2 * H_kv * D_head
  # - o_proj projects H_q * D_head -> E
  # - indexer_q_proj + indexer_k_proj project E -> (index_n_heads + index_kv_heads) * index_head_dim
  qsa_q_flops = 2 * B * S * E * (2 * H_q * D_head)
  qsa_kv_flops = 2 * B * S * E * (2 * H_kv * D_head)
  qsa_o_flops = 2 * B * S * (H_q * D_head) * E
  idx_proj_flops = 2 * B * S * E * (index_n_heads + index_kv_heads) * index_head_dim
  qsa_weight_flops_per_layer = qsa_q_flops + qsa_kv_flops + qsa_o_flops + idx_proj_flops

  # QSA attention + MQA micro-block indexer non-weight FLOPs:
  # - Indexer block-compressed score matmul (causal half):
  #   2 * B * S * (S / compress_ratio) * index_n_heads * index_head_dim / 2
  # - Indexer head reduction (causal half):
  #   2 * B * S * (S / compress_ratio) * index_n_heads / 2
  # - Sparse attention over selected token budget (causal window ratio):
  num_blocks = S / compress_ratio
  idx_score_flops = B * S * num_blocks * index_n_heads * index_head_dim
  idx_reduce_flops = B * S * num_blocks * index_n_heads
  effective_budget = min(token_budget, S) if token_budget > 0 else S
  qsa_mask_ratio = _indexer_mask_ratio(effective_budget, S)
  qsa_sparse_attn_flops = 4 * B * (S**2) * H_q * D_head * qsa_mask_ratio
  qsa_attn_flops_per_layer = qsa_sparse_attn_flops + idx_score_flops + idx_reduce_flops

  # 3. 4-stream Hyper-Connections (HCBlock x2 per decoder layer + final HCMixer)
  hc_count = getattr(config, "hc_count", 1)
  hc_lowrank = getattr(config, "hc_lowrank", 0)
  hc_dim = hc_count * E
  if hc_count > 1 and hc_lowrank > 0:
    # Each HCBlock:
    # - pre_Compress: [B, S, hc_C*E] x [hc_C*E, hc_r] -> 2 * B * S * hc_dim * hc_lowrank
    # - post_Compress: [B, S, hc_C*E] x [hc_C*E, hc_r] -> 2 * B * S * hc_dim * hc_lowrank
    # - stream mixing (h_pre / h_res / h_post): 2 * B * S * hc_dim * hc_count
    hc_block_flops = 2 * B * S * hc_dim * (2 * hc_lowrank + hc_count)
    hc_layer_flops = 2 * hc_block_flops
    # Final HCMixer:
    # - pre_Compress: [B, S, hc_C*E] x [hc_C*E, hc_r]
    # - Expand + weighted sum: 2 * B * S * hc_dim * hc_lowrank
    hc_mixer_flops = 2 * B * S * hc_dim * (2 * hc_lowrank)
    total_hc_flops = num_layers * hc_layer_flops + hc_mixer_flops
  else:
    hc_layer_flops = 0
    hc_mixer_flops = 0
    total_hc_flops = 0

  # 4. Shared expert scalar gate (`shared_expert_gate`: [E, 1]) per MoE layer
  shared_expert_gate_flops_per_layer = 2 * B * S * E * getattr(config, "shared_experts", 0)
  total_shared_gate_flops = num_layers * shared_expert_gate_flops_per_layer

  # 5. Layer-2 Per-Layer Embedding (PLE) block (`w_ple`, `gate_proj`, causal conv1d)
  if getattr(config, "use_ple_layer", False):
    ple_dim = getattr(config, "ple_embed_dim", 0) or E
    ple_conv_k = getattr(config, "ple_conv_kernel_size", 4)
    # w_ple: [ple_dim, hc_dim], gate_proj: [E, ple_dim], conv1d depthwise: [ple_conv_k, hc_dim]
    ple_weight_flops = 2 * B * S * (ple_dim * hc_dim + ple_dim * E + ple_conv_k * hc_dim)
    # Elementwise gated mixing: 2 * B * S * hc_dim
    ple_attn_flops = 2 * B * S * hc_dim
  else:
    ple_weight_flops = 0
    ple_attn_flops = 0

  # 6. Multi-Token Prediction (MTP) block
  # In Qwen3.8-Flash-Next, each MTP step projects [inputs_embeds (E), 4-stream hidden (hc_dim)]
  # -> E via `mtp_fc`, then runs 1 full QSA + MoE decoder layer (with 2 HCBlocks) + `mtp_hc_mixer`
  # + `self.lm_head(mtp_h)` unembedding matmul. The `lm_head` weights are shared with the main
  # head, but each MTP step executes its own [B*S, E] x [E, V] matmul (`embedding_flops`).
  mtp_num_layers = getattr(config, "mtp_num_layers", 0)
  if mtp_num_layers > 0:
    mtp_fc_flops = 2 * B * S * (1 + hc_count) * E * E
    moe_layer_flops = (total_ffn_flops_all_layers / num_layers) + shared_expert_gate_flops_per_layer
    mtp_weight_flops_per_layer = (
        mtp_fc_flops + qsa_weight_flops_per_layer + hc_layer_flops + moe_layer_flops + hc_mixer_flops + embedding_flops
    )
    mtp_weight_flops = mtp_num_layers * mtp_weight_flops_per_layer
    mtp_attn_flops = mtp_num_layers * qsa_attn_flops_per_layer
  else:
    mtp_weight_flops = 0
    mtp_attn_flops = 0

  total_weight_flops = (
      total_ffn_flops_all_layers
      + total_shared_gate_flops
      + embedding_flops
      + num_qsa_layers * qsa_weight_flops_per_layer
      + num_gdn_layers * gdn_weight_flops_per_layer
      + total_hc_flops
      + ple_weight_flops
      + mtp_weight_flops
  )
  total_attn_flops = (
      num_qsa_layers * qsa_attn_flops_per_layer
      + num_gdn_layers * gdn_attn_flops_per_layer
      + ple_attn_flops
      + mtp_attn_flops
  )

  # x3 for forward + backward, to TFLOPs, then scale by gradient accumulation.
  learnable_weight_tflops = total_weight_flops * 3 / 10**12
  attention_tflops = total_attn_flops * 3 / 10**12
  learnable_weight_tflops = learnable_weight_tflops * config.gradient_accumulation_steps
  attention_tflops = attention_tflops * config.gradient_accumulation_steps

  # DPO includes one additional (reference-model) forward pass per gradient accumulation step.
  if config.use_dpo:
    reference_model_tflops = learnable_weight_tflops / 3
    reference_model_attention_tflops = attention_tflops / 3
    attention_tflops = attention_tflops + reference_model_attention_tflops
  else:
    reference_model_tflops = 0

  total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops

  if log:
    print(
        f"{config.model_name} per train step:\n",
        f"Total TFLOPs: {total_tflops:.2f} \n",
        f"split as {100 * learnable_weight_tflops / total_tflops:.2f}% learnable weight flops",
        f"and {100 * attention_tflops / total_tflops:.2f}% attention flops",
    )
  return total_tflops, learnable_weight_tflops, attention_tflops
