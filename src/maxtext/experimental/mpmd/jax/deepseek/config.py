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

"""Config for the pure-JAX DeepSeek-V3 model.

Field names and defaults mirror MaxText's resolved config so that a MaxText
`pyconfig` object can be translated 1:1 (see `from_maxtext`).  Only the
knobs that the DeepSeek-V3 training forward pass actually reads are kept;
everything that MaxText leaves off by default (quantization, pipeline
parallelism, attention sinks, soft caps, indexer/DSA, ...) is simply absent.

How the model shards is one more field, `shard` (a `ShardPolicy`): the
default policy, the program this port has always traced.  `SharedLayerConfig`
names the fields that the functions other models borrow from this package
read.
"""

import dataclasses
import math
import typing
from typing import Any, Protocol

import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.sharding_mode import ShardPolicy


@dataclasses.dataclass(frozen=True)
class DeepSeekConfig:
  """Resolved DeepSeek-V3 hyperparameters."""

  # --- vocabulary / shape ---------------------------------------------------
  vocab_size: int = 129280
  emb_dim: int = 64
  num_decoder_layers: int = 61
  first_num_dense_layers: int = 3

  # --- MLA ------------------------------------------------------------------
  num_query_heads: int = 4
  num_kv_heads: int = 4
  q_lora_rank: int = 32
  kv_lora_rank: int = 16
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128

  # --- MLP / MoE ------------------------------------------------------------
  mlp_dim: int = 64
  moe_mlp_dim: int = 64
  mlp_activations: tuple[str, ...] = ("silu", "linear")
  num_experts: int = 16
  num_experts_per_tok: int = 8
  shared_experts: int = 1
  routed_score_func: str = "sigmoid"
  routed_bias: bool = True
  routed_scaling_factor: float = 2.5
  routed_bias_update_rate: float = 0.0
  # Group-limited routing (moe.expert_group_mask): the experts are cut into
  # n_routing_groups groups, each scored by its two best post-bias scores, and
  # a token picks its top-k among the topk_routing_group best groups.  -1: off.
  n_routing_groups: int = -1
  topk_routing_group: int = -1
  # Multi-token prediction (multi_token_prediction.py): mtp_num_layers extra
  # MoE layers after the decoder, each predicting one token further, their
  # mean cross entropy scaled by mtp_loss_scaling_factor into the loss.
  mtp_num_layers: int = 0
  mtp_loss_scaling_factor: float = 0.1
  load_balance_loss_weight: float = 0.0
  # The aux loss of a sigmoid router as Megatron-LM's seq_aux_loss
  # (moe.megatron_seq_aux_loss, DeepSeek-V3 eqs 17-20): per sequence, summed
  # over the MoE layers (MTP's too); else MaxText's Switch-style loss.
  moe_use_megatron_seq_aux_loss: bool = False
  # `dense` reproduces moe.dense_matmul (einsum over every expert);
  # `ragged` reproduces moe.sparse_matmul (sort + grouped matmul).
  moe_impl: str = "ragged"
  # MaxText's default grouped matmul is the megablox Pallas kernel, which
  # accumulates every tile in fp32 and rounds to bf16 once, on the way out.  Its
  # `jax_ragged_dot_gmm` fallback (`megablox=false use_tokamax_gmm=false`) hands
  # `preferred_element_type=bf16` to XLA instead.  The two differ by up to one
  # bf16 ulp per grouped matmul, so the port has to follow whichever is active.
  gmm_fp32_accum: bool = True
  # Reproduce megablox's `tgmm` weight gradient exactly, at `num_experts` times
  # the FLOPs -- see `moe._grouped_matmul`.  For parity runs, not for training.
  gmm_exact_drhs: bool = False
  moe_use_direct_token_gather: bool = False
  float32_weight_sum: bool = True
  float32_gate_logits: bool = False

  # --- normalization --------------------------------------------------------
  normalization_layer_epsilon: float = 1.0e-6

  # --- YaRN RoPE ------------------------------------------------------------
  rope_max_timescale: float = 10_000.0
  max_position_embeddings: int = 163840
  original_max_position_embeddings: int = 4096
  rope_factor: float = 40.0
  beta_fast: float = 32.0
  beta_slow: float = 1.0
  mscale: float = 1.0
  rope_interleave: bool = True
  rope_truncate: bool = True
  rope_attention_scaling: bool = False

  # --- numerics -------------------------------------------------------------
  dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32
  matmul_precision: str = "default"
  cast_logits_to_fp32: bool = True
  logits_dot_in_fp32: bool = False
  dense_init_scale: float = 1.0

  # --- attention kernel -----------------------------------------------------
  # 'splash'      -- the TPU Pallas kernel MaxText autoselects (fp32 softmax);
  # 'dot_product' -- MaxText's `apply_attention_dot` (bf16 softmax), which is
  #                  what MaxText itself falls back to below 128 tokens;
  # 'reference'   -- a portable fp32 stand-in for the splash kernel;
  # 'tokamax_splash' -- tokamax's splash kernel with the fused backward
  #                  (MaxText `use_tokamax_splash=true`), which the MPMD
  #                  ds_ep pipeline runs;
  # 'cudnn'       -- cuDNN's fused flash attention (GPU; unpacked rows, no
  #                  context parallelism): ds_ep_pipeline's GPU default.
  attention_kernel: str = "splash"
  sa_block_q: int = 512
  sa_block_kv: int = 512
  sa_block_kv_compute: int = 512
  # 'tokamax_splash' only (MaxText `use_tokamax_splash=true`, fused bwd).
  # 0 means "same as the forward block"; -1 cost estimates mean "unset".
  sa_block_q_dkv: int = 0
  sa_block_kv_dkv: int = 0
  sa_block_kv_dkv_compute: int = 0
  sa_cost_estimate_flops_fwd: int = -1
  sa_cost_estimate_flops_bwd: int = -1
  # 'tokamax_splash' only: tokamax's base-2 softmax (MaxText `sa_use_base2_exp`,
  # true in its base.yml, as here; tokamax always ran it before the field).  Not
  # the exact softmax: with a bf16 query the kernels scale it by the Python
  # float log2(e), which JAX rounds to bf16 1.4453125, so the softmax runs at
  # 1.0018x its logits while the fused backward's dq and dk both leave that
  # factor out (DeepSeek-V3's layer 0 at init: the output and dv ~0.3%, dq and
  # dk ~0.25% larger than the exact softmax's).  False computes the exact one,
  # ~1.2% slower in the kernel (fwd + bwd, DeepSeek-V3 attention at 4096 tokens
  # x 128 heads on one TPU7x device: 11.49 vs 11.34 ms).  `from_maxtext`
  # carries MaxText's setting, but its kernel resolution maps MaxText's flash
  # attention to 'splash' (the exact softmax): it matters where a caller picks
  # 'tokamax_splash', e.g. ds_ep_pipeline.
  sa_use_base2_exp: bool = True

  # --- training -------------------------------------------------------------
  per_device_batch_size: int = 1
  max_target_length: int = 2048
  scan_layers: bool = True
  remat_policy: str = "full"

  # --- sharding -------------------------------------------------------------
  # How the model shards (`src/maxtext/experimental/mpmd/jax/sharding_mode.py`): the default
  # policy, today's program, the only one there is.  It is frozen like this
  # class, which therefore stays hashable.
  shard: ShardPolicy = ShardPolicy()

  # --- derived --------------------------------------------------------------
  @property
  def qk_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  @property
  def num_moe_layers(self) -> int:
    return self.num_decoder_layers - self.first_num_dense_layers

  @property
  def softmax_scale(self) -> float:
    """MLA softmax scale, folded into the query (`attention_mla.MLA.mla_query_projection`)."""
    scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      scale = scale * mscale * mscale
    return scale

  @property
  def shared_expert_mlp_dim(self) -> int:
    """`moe.get_shared_expert_mlp_dim` -- moe_mlp_dim unless overridden."""
    return self.moe_mlp_dim

  def replace(self, **kwargs) -> "DeepSeekConfig":
    return dataclasses.replace(self, **kwargs)


@typing.runtime_checkable
class SharedLayerConfig(Protocol):
  """The config fields read by the functions of this package other models call.

  The Qwen3.5 port calls a handful of these whose arithmetic is the same for
  both models, so any config handed to them needs exactly these fields:
  `DeepSeekConfig` has them, and so must every other model's config.
  `deepseek_model_test` collects every `config.<field>` the functions'
  sources mention and checks the set equals this one, and it traces them
  with an object that has these fields and nothing else.

    layers.precision_of                     matmul_precision
    layers.precision_or_none                (precision_of)
    layers.rms_norm                         dtype, normalization_layer_epsilon
                                            (+ precision_or_none)
    layers.dense_general                    dtype (+ precision_of)
    layers.embed_lookup                     dtype
    layers.mlp_block                        mlp_activations, dtype
    moe.routed_experts                      moe_impl
    moe._experts_dense                      dtype, num_experts,
                                            float32_weight_sum
    moe._experts_ragged                     dtype, num_experts,
                                            num_experts_per_tok,
                                            moe_use_direct_token_gather,
                                            gmm_fp32_accum, gmm_exact_drhs,
                                            float32_weight_sum (+ precision_of,
                                            precision_or_none)
    attention.attention_op                  attention_kernel; splash
                                            reads num_query_heads and
                                            sa_block_{q,kv,kv_compute};
                                            reference (+ precision_or_none)
    attention.dot_product_attention         (precision_of)
    attention.tokamax_splash_attention      sa_block_* incl. the *_dkv blocks,
                                            sa_cost_estimate_flops_{fwd,bwd},
                                            sa_use_base2_exp
    model.remat_policy_for                  remat_policy
  """

  dtype: Any
  matmul_precision: str
  normalization_layer_epsilon: float
  mlp_activations: tuple
  moe_impl: str
  num_experts: int
  num_experts_per_tok: int
  moe_use_direct_token_gather: bool
  gmm_fp32_accum: bool
  gmm_exact_drhs: bool
  float32_weight_sum: bool
  attention_kernel: str
  num_query_heads: int
  sa_block_q: int
  sa_block_kv: int
  sa_block_kv_compute: int
  sa_block_q_dkv: int
  sa_block_kv_dkv: int
  sa_block_kv_dkv_compute: int
  sa_cost_estimate_flops_fwd: int
  sa_cost_estimate_flops_bwd: int
  sa_use_base2_exp: bool
  remat_policy: str


SHARED_LAYER_FIELDS = frozenset(SharedLayerConfig.__annotations__)


def resolve_attention_kernel(cfg) -> str:
  """`attention_op.AttentionOp.apply_attention` dispatch, train.

  `autoselected` picks the TPU splash kernel unless the sequence is shorter than
  128 tokens, in which case MaxText silently drops to the bf16 dot-product path.
  """
  kernel = cfg.attention
  if kernel == "dot_product":
    return "dot_product"
  if kernel in ("autoselected", "flash"):
    if int(cfg.max_target_length) < 128:
      return "dot_product"
    return "splash"
  raise ValueError(f"unsupported attention kernel for this port: {kernel}")


def from_maxtext(cfg) -> DeepSeekConfig:
  """Builds a DeepSeekConfig from an initialized MaxText pyconfig."""
  activations = tuple(cfg.mlp_activations)
  return DeepSeekConfig(
      attention_kernel=resolve_attention_kernel(cfg),
      sa_block_q=cfg.sa_block_q,
      sa_block_kv=cfg.sa_block_kv,
      sa_block_kv_compute=cfg.sa_block_kv_compute,
      sa_use_base2_exp=bool(cfg.sa_use_base2_exp),
      scan_layers=cfg.scan_layers,
      remat_policy=cfg.remat_policy,
      vocab_size=cfg.vocab_size,
      emb_dim=cfg.emb_dim,
      num_decoder_layers=cfg.num_decoder_layers,
      first_num_dense_layers=cfg.first_num_dense_layers,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      q_lora_rank=cfg.q_lora_rank,
      kv_lora_rank=cfg.kv_lora_rank,
      qk_nope_head_dim=cfg.qk_nope_head_dim,
      qk_rope_head_dim=cfg.qk_rope_head_dim,
      v_head_dim=cfg.v_head_dim,
      mlp_dim=cfg.mlp_dim,
      moe_mlp_dim=cfg.moe_mlp_dim,
      mlp_activations=activations,
      num_experts=cfg.num_experts,
      num_experts_per_tok=cfg.num_experts_per_tok,
      shared_experts=cfg.shared_experts,
      routed_score_func=cfg.routed_score_func,
      routed_bias=cfg.routed_bias,
      routed_scaling_factor=cfg.routed_scaling_factor,
      routed_bias_update_rate=cfg.routed_bias_update_rate,
      n_routing_groups=cfg.n_routing_groups,
      topk_routing_group=cfg.topk_routing_group,
      mtp_num_layers=cfg.mtp_num_layers,
      mtp_loss_scaling_factor=cfg.mtp_loss_scaling_factor,
      load_balance_loss_weight=cfg.load_balance_loss_weight,
      moe_use_megatron_seq_aux_loss=bool(cfg.moe_use_megatron_seq_aux_loss),
      moe_impl="ragged" if cfg.sparse_matmul else "dense",
      gmm_fp32_accum=bool(cfg.megablox or cfg.use_tokamax_gmm),
      moe_use_direct_token_gather=cfg.moe_use_direct_token_gather,
      float32_weight_sum=cfg.float32_weight_sum,
      float32_gate_logits=cfg.float32_gate_logits,
      normalization_layer_epsilon=cfg.normalization_layer_epsilon,
      rope_max_timescale=cfg.rope_max_timescale,
      max_position_embeddings=cfg.max_position_embeddings,
      original_max_position_embeddings=cfg.original_max_position_embeddings,
      rope_factor=cfg.rope_factor,
      beta_fast=cfg.beta_fast,
      beta_slow=cfg.beta_slow,
      mscale=cfg.mscale,
      rope_interleave=cfg.rope_interleave,
      rope_truncate=cfg.rope_truncate,
      rope_attention_scaling=cfg.rope_attention_scaling,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      matmul_precision=cfg.matmul_precision,
      cast_logits_to_fp32=cfg.cast_logits_to_fp32,
      logits_dot_in_fp32=cfg.logits_dot_in_fp32,
      dense_init_scale=cfg.dense_init_scale,
      per_device_batch_size=int(cfg.per_device_batch_size),
      max_target_length=cfg.max_target_length,
  )
