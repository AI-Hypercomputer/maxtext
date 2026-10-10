# Copyright 2026 Google LLC
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

"""Configuration dataclasses for the Lineage DSv3 model.

`DSv3Config` bundles everything the model-level API (`dsv3_model.py`) needs:
model dimensions, sharding, kernel/runtime knobs and the training objective.
Framework adapters build it once (e.g. `lineage_adapter.from_maxtext_config`)
and nothing below the adapter reads a framework config.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_sharding
from maxtext.experimental.lineage import quantization

# Quantization flag value -> `QuantConfig` (None runs in bf16).
QUANT_CONFIGS: dict[str, quantization.QuantConfig | None] = {
    "none": None,
    "fp8_full": quantization.FP8_FULL,
}


def quant_config_from_name(name: str | None) -> quantization.QuantConfig | None:
  """Returns the `QuantConfig` selected by a quantization flag value.

  Args:
    name: One of `QUANT_CONFIGS`; None is treated as "none".

  Returns:
    The quantization config, or None for bf16.

  Raises:
    ValueError: If `name` is not a known quantization.
  """
  name = "none" if name is None else name
  if name not in QUANT_CONFIGS:
    raise ValueError(f"quantization must be one of {sorted(QUANT_CONFIGS)}, got: {name!r}")
  return QUANT_CONFIGS[name]


@dataclasses.dataclass(frozen=True)
class DSv3ModelConfig:
  """Model dimensions and dtypes.

  Attributes:
    vocab_size: Vocabulary size `V`.
    emb_dim: Model dim `D`.
    num_dense_layers: Dense (MLP) decoder layers; at least 2.
    num_sparse_layers: Sparse (MoE) decoder layers; at least 3.
    num_mtp_layers: Multi-token prediction depths `K` (0 disables MTP).
    num_query_heads: Attention heads `H`.
    num_kv_heads: Key/value heads (equal to `num_query_heads` for DSv3).
    q_lora_rank: Compressed query dim.
    kv_lora_rank: Compressed key/value dim.
    qk_head_dim: Non-positional q/k head dim.
    rope_head_dim: RoPE head dim.
    v_head_dim: Value head dim.
    mscale: YaRN attention scale.
    mlp_dim: Dense layer MLP hidden dim.
    moe_mlp_dim: Routed and shared expert hidden dim.
    num_experts: Routed experts `E`.
    num_experts_per_tok: Routed experts per token.
    n_routing_groups: Device-limited routing groups.
    topk_routing_group: Groups selected per token.
    routed_scaling_factor: Scale of the routed expert outputs.
    rope_theta: RoPE base frequency.
    max_position_embeddings: YaRN extended context length.
    original_max_position_embeddings: YaRN original context length.
    rope_factor: YaRN scaling factor.
    beta_fast: YaRN fast rotation threshold.
    beta_slow: YaRN slow rotation threshold.
    norm_epsilon: RMSNorm epsilon.
    tied_head: Compute logits against the embedding table instead of a separate
      LM head kernel (`logits_via_embedding`).
    iota_embed: Look up embeddings with a one-hot matmul (`use_iota_embed`).
    dtype: Activation / compute dtype.
    weight_dtype: Parameter storage dtype.
    router_dtype: Expert-selection dtype (fp32 under `float32_gate_logits`), or
      None to route in `dtype`.
    router_bias_dtype: Storage dtype of the routed gate bias.
    head_dot_in_fp32: Run the LM head matmul in fp32 (`logits_dot_in_fp32`).
    cast_logits_to_fp32: Cast the logits to fp32 before the loss.
  """

  vocab_size: int
  emb_dim: int
  num_dense_layers: int
  num_sparse_layers: int
  num_mtp_layers: int
  num_query_heads: int
  num_kv_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_head_dim: int
  rope_head_dim: int
  v_head_dim: int
  mscale: float
  mlp_dim: int
  moe_mlp_dim: int
  num_experts: int
  num_experts_per_tok: int
  n_routing_groups: int
  topk_routing_group: int
  routed_scaling_factor: float
  rope_theta: int
  max_position_embeddings: int
  original_max_position_embeddings: int
  rope_factor: int
  beta_fast: int
  beta_slow: int
  norm_epsilon: float
  tied_head: bool = False
  iota_embed: bool = False
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  weight_dtype: jax.typing.DTypeLike = jnp.float32
  router_dtype: jax.typing.DTypeLike | None = None
  router_bias_dtype: jax.typing.DTypeLike = jnp.bfloat16
  head_dot_in_fp32: bool = False
  cast_logits_to_fp32: bool = True

  @property
  def topk_in_group(self) -> int:
    """Experts selected per routing group."""
    return self.num_experts_per_tok // self.topk_routing_group

  @property
  def logits_dtype(self) -> jax.typing.DTypeLike:
    """Dtype of the logits handed to the loss."""
    if self.cast_logits_to_fp32 or self.head_dot_in_fp32:
      return jnp.float32
    return self.dtype


@dataclasses.dataclass(frozen=True)
class DSv3KernelConfig:
  """Kernel and runtime knobs of the Lineage layers.

  Attributes:
    max_target_length: Sequence length `T` (sizes the Splash causal mask).
    sa_block_q: Splash attention forward query block.
    sa_block_kv: Splash attention forward key/value block.
    sa_block_kv_compute: Splash attention forward key/value compute block.
    sa_block_q_dkv: Splash attention backward query block.
    sa_block_kv_dkv: Splash attention backward key/value block.
    sa_block_kv_dkv_compute: Splash attention backward key/value compute block.
    sa_q_layout: Splash attention query layout.
    sa_k_layout: Splash attention key layout.
    sa_v_layout: Splash attention value layout.
    qk_diag_skip: Skip masked QK diagonal blocks in Splash attention.
    qk_diag_grid: Diagonal grid of the QK diagonal skip.
    sv_diag_skip: Skip masked SV diagonal blocks in Splash attention.
    capacity_factor: MoE chunk capacity relative to the balanced load.
    ragged_buffer_factor: Cross-layer ragged activation bank safety factor.
    quant: Quantization of the routed experts and MLA, or None for bf16.
    max_async_overlap_transform: Register the max-async-overlap XLA transform
      around the decoder.
    expert_permutation: Expert permutation algorithm, one of
      `dsv3_expert_shuffle.EXPERT_PERMUTATION_ALGORITHMS`.
  """

  max_target_length: int
  sa_block_q: int = 2048
  sa_block_kv: int = 2048
  sa_block_kv_compute: int = 2048
  sa_block_q_dkv: int = 2048
  sa_block_kv_dkv: int = 2048
  sa_block_kv_dkv_compute: int = 2048
  sa_q_layout: Any = "SEQ_MINOR"
  sa_k_layout: Any = "SEQ_MINOR"
  sa_v_layout: Any = "HEAD_DIM_MINOR"
  qk_diag_skip: bool = True
  qk_diag_grid: int = 8
  sv_diag_skip: bool = True
  capacity_factor: float = 2.0
  ragged_buffer_factor: float = 1.2
  quant: quantization.QuantConfig | None = None
  max_async_overlap_transform: bool = True
  expert_permutation: str = "none"


@dataclasses.dataclass(frozen=True)
class DSv3TrainingConfig:
  """Training objective knobs.

  Attributes:
    load_balance_loss_weight: Weight of the MoE load balance loss (0 disables).
    megatron_seq_aux_loss: Use Megatron-LM's sigmoid-router seq_aux_loss, one
      loss per MoE layer summed over layers; otherwise the Switch-style loss
      averaged over layers.
    routed_bias_update_rate: Loss-free load balancing update rate of the routed
      gate bias (0 disables the update).
    z_loss_multiplier: z-loss coefficient of the main cross-entropy.
    mtp_loss_scaling_factor: Scale of the MTP loss.
    mtp_eval_target_module: 1-based MTP depth whose predictions are reported for
      the acceptance rate in eval (0 disables).
    mtp_reuse_input_embedding: Shift the main decoder's token embeddings for the
      MTP depths instead of looking the shifted tokens up again.
  """

  load_balance_loss_weight: float = 0.0
  megatron_seq_aux_loss: bool = False
  routed_bias_update_rate: float = 0.0
  z_loss_multiplier: float = 0.0
  mtp_loss_scaling_factor: float = 0.1
  mtp_eval_target_module: int = 0
  mtp_reuse_input_embedding: bool = False


@dataclasses.dataclass(frozen=True)
class DSv3Config:
  """Full configuration of the Lineage DSv3 model.

  Attributes:
    model: Model dimensions and dtypes.
    sharding: Axis mapping and weight/activation specs.
    kernels: Kernel and runtime knobs.
    training: Training objective knobs.
  """

  model: DSv3ModelConfig
  sharding: dsv3_sharding.DSv3ShardingConfig
  kernels: DSv3KernelConfig
  training: DSv3TrainingConfig = DSv3TrainingConfig()

  def validate(self) -> None:
    """Checks the constraints of the Lineage layers.

    Raises:
      ValueError: On the first violated constraint.
    """
    m, k, t = self.model, self.kernels, self.training
    if m.num_dense_layers < 2:
      raise ValueError("Lineage's dense layer scan needs at least 2 dense layers, got" f" {m.num_dense_layers}.")
    if m.num_sparse_layers < 3:
      raise ValueError("Lineage's sparse layer scan needs at least 3 sparse layers, got" f" {m.num_sparse_layers}.")
    if m.num_mtp_layers < 0:
      raise ValueError(f"num_mtp_layers must be >= 0, got {m.num_mtp_layers}.")
    if m.topk_routing_group <= 0 or (m.num_experts_per_tok % m.topk_routing_group):
      raise ValueError(
          f"num_experts_per_tok ({m.num_experts_per_tok}) must be a positive"
          f" multiple of topk_routing_group ({m.topk_routing_group})."
      )
    if m.num_kv_heads != m.num_query_heads:
      raise ValueError(f"num_kv_heads ({m.num_kv_heads}) must equal num_query_heads" f" ({m.num_query_heads}).")
    if k.capacity_factor <= 0:
      raise ValueError(f"capacity_factor must be > 0, got {k.capacity_factor}.")
    if k.ragged_buffer_factor < 1.0:
      raise ValueError(f"ragged_buffer_factor must be >= 1, got {k.ragged_buffer_factor}.")
    if k.expert_permutation not in (dsv3_expert_shuffle.EXPERT_PERMUTATION_ALGORITHMS):
      raise ValueError(
          "expert_permutation must be one of"
          f" {dsv3_expert_shuffle.EXPERT_PERMUTATION_ALGORITHMS}, got"
          f" {k.expert_permutation!r}."
      )
    if t.mtp_eval_target_module < 0 or (t.mtp_eval_target_module > m.num_mtp_layers):
      raise ValueError(
          f"mtp_eval_target_module ({t.mtp_eval_target_module}) must be in" f" [0, num_mtp_layers={m.num_mtp_layers}]."
      )
    missing = [axis for axis in dsv3_sharding.LOGICAL_AXES if axis not in self.sharding.axis_mapping]
    if missing:
      raise ValueError(f"sharding.axis_mapping lacks logical axes {missing}.")
