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

"""Model Garden config for Qwen3.8-Flash-Next (GDN/QSA hybrid MoE with PLE and MTP)."""

from typing import Any, Literal

from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator


class Qwen3_8FlashNextConfig(ModelConfig):  # pylint: disable=invalid-name
  """Frozen HuggingFace-native architecture configuration for Qwen3.8-Flash-Next."""

  model_type: str = Field("qwen3_next", description="HuggingFace model_type identifier.")
  model_name: str = Field("qwen3.8-flash-next", description="Canonical model identifier.")
  vocab_size: int = Field(248320, gt=0)
  hidden_size: int = Field(2560, gt=0)
  num_hidden_layers: int = Field(48, gt=0)
  num_attention_heads: int = Field(24, gt=0)
  num_key_value_heads: int = Field(2, gt=0)
  head_dim: int = Field(256, gt=0)
  rms_norm_eps: float = Field(1e-6, gt=0.0)
  tie_word_embeddings: bool = False

  # Hyper-Connections (HC)
  hc_count: int = Field(4, gt=0)
  hc_lowrank: int = Field(320, gt=0)

  # MoE parameters
  shared_expert_intermediate_size: int = Field(640, gt=0)
  moe_intermediate_size: int = Field(640, gt=0)
  n_routed_experts: int = Field(512, gt=0)
  num_experts_per_tok: int = Field(10, gt=0)
  n_shared_experts: int = Field(1, ge=0)
  routed_scaling_factor: float = Field(1.0, gt=0.0)
  norm_topk_prob: bool = True
  router_aux_loss_coef: float = Field(0.001, ge=0.0)

  # Gated DeltaNet (GDN) linear attention
  full_attention_interval: int = Field(4, gt=0)
  linear_conv_kernel_dim: int = Field(4, gt=0)
  linear_key_head_dim: int = Field(128, gt=0)
  linear_value_head_dim: int = Field(128, gt=0)
  linear_num_key_heads: int = Field(16, gt=0)
  linear_num_value_heads: int = Field(48, gt=0)
  gdn_chunk_size: int = Field(64, gt=0)
  gdn_output_gate_activation: str = Field("sigmoid")
  use_qk_norm_in_gdn: bool = True
  gdn_chunk_inverse: Literal["solve", "block_doubling"] = Field("block_doubling")

  # RoPE & interleaved M-RoPE
  rope_theta: float = Field(10_000_000.0, gt=0.0)
  partial_rotary_factor: float = Field(0.25, gt=0.0, le=1.0)
  mrope_section: tuple[int, int, int] = Field((11, 11, 10))

  # QSA & Indexer
  indexer_n_heads: int = Field(4, gt=0)
  indexer_kv_heads: int = Field(1, gt=0)
  indexer_head_dim: int = Field(128, gt=0)
  indexer_compress_ratio: int = Field(4, gt=0)
  indexer_budget: int = Field(2048, gt=0)
  qsa_indexer_mask: Literal["gather", "reshape"] = Field("gather")
  qsa_attention_kernel: Literal["dense", "flash"] = Field("dense")

  # Per-Layer Embedding (PLE)
  use_ple_layer: bool = True
  ple_layer_id: int = Field(1, ge=0)
  ple_embed_dim: int = Field(2560, gt=0)
  ple_conv_kernel_size: int = Field(4, gt=0)
  ngram_size: int = Field(3, gt=0)
  heads_per_ngram: int = Field(8, gt=0)
  ngram_vocab_size_base: int = Field(20_000_000, gt=0)
  make_ngram_vocab_size_divisible_by: int = Field(128, gt=0)
  split_ngram_parts: int = Field(128, gt=0)
  ple_seed: int = Field(1234)
  eos_token_id: int = Field(248044, ge=0)

  # Multi-Token Prediction (MTP)
  num_nextn_predict_layers: int = Field(1, ge=0, le=1)
  mtp_loss_scaling_factor: float = Field(0.1, ge=0.0)

  @model_validator(mode="after")
  def _validate_qwen3_8_invariants(self) -> "Qwen3_8FlashNextConfig":
    """Validates hybrid layer cycle, RoPE/M-RoPE geometry, MoE, indexer, and PLE invariants."""
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if self.num_hidden_layers % self.full_attention_interval != 0:
      raise ValueError(
          f"num_hidden_layers ({self.num_hidden_layers}) must be divisible by "
          f"full_attention_interval ({self.full_attention_interval})."
      )
    rotary_dim_f = self.head_dim * self.partial_rotary_factor
    if not rotary_dim_f.is_integer() or int(rotary_dim_f) % 2 != 0:
      raise ValueError(f"head_dim * partial_rotary_factor ({rotary_dim_f:g}) must be an even integer for RoPE.")
    rotary_dim = int(rotary_dim_f)
    if any(s < 0 for s in self.mrope_section) or sum(self.mrope_section) != rotary_dim // 2:
      raise ValueError(
          f"mrope_section ({self.mrope_section}) entries must be non-negative and sum to "
          f"rotary_dim // 2 ({rotary_dim // 2})."
      )
    if self.indexer_n_heads % self.indexer_kv_heads != 0:
      raise ValueError(
          f"indexer_n_heads ({self.indexer_n_heads}) must be divisible by " f"indexer_kv_heads ({self.indexer_kv_heads})."
      )
    if self.use_ple_layer and not 0 <= self.ple_layer_id < self.num_hidden_layers:
      raise ValueError(
          f"ple_layer_id ({self.ple_layer_id}) must be in [0, {self.num_hidden_layers}) when use_ple_layer is True."
      )
    if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
      raise ValueError(f"num_experts_per_tok ({self.num_experts_per_tok}) must be in [1, {self.n_routed_experts}].")
    if self.linear_num_value_heads % self.linear_num_key_heads != 0:
      raise ValueError(
          f"linear_num_value_heads ({self.linear_num_value_heads}) must be divisible by "
          f"linear_num_key_heads ({self.linear_num_key_heads})."
      )
    if self.eos_token_id >= self.vocab_size:
      raise ValueError(f"eos_token_id ({self.eos_token_id}) must be < vocab_size ({self.vocab_size}).")
    return self

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults for this Qwen3.8-Flash-Next variant."""
    return {
        "decoder_block": "qwen3_next",
        "base_emb_dim": self.hidden_size,
        "base_num_decoder_layers": self.num_hidden_layers,
        "base_num_query_heads": self.num_attention_heads,
        "base_num_kv_heads": self.num_key_value_heads,
        "head_dim": self.head_dim,
        "vocab_size": self.vocab_size,
        "normalization_layer_epsilon": self.rms_norm_eps,
        "logits_via_embedding": self.tie_word_embeddings,
        "hc_count": self.hc_count,
        "hc_lowrank": self.hc_lowrank,
        "base_mlp_dim": self.shared_expert_intermediate_size,
        "base_moe_mlp_dim": self.moe_intermediate_size,
        "num_experts": self.n_routed_experts,
        "num_experts_per_tok": self.num_experts_per_tok,
        "shared_experts": self.n_shared_experts,
        "routed_scaling_factor": self.routed_scaling_factor,
        "norm_topk_prob": self.norm_topk_prob,
        "load_balance_loss_weight": self.router_aux_loss_coef,
        "inhomogeneous_layer_cycle_interval": self.full_attention_interval,
        "gdn_conv_kernel_dim": self.linear_conv_kernel_dim,
        "gdn_key_head_dim": self.linear_key_head_dim,
        "gdn_value_head_dim": self.linear_value_head_dim,
        "gdn_num_key_heads": self.linear_num_key_heads,
        "gdn_num_value_heads": self.linear_num_value_heads,
        "gdn_chunk_size": self.gdn_chunk_size,
        "gdn_output_gate_activation": self.gdn_output_gate_activation,
        "use_qk_norm_in_gdn": self.use_qk_norm_in_gdn,
        "rope_max_timescale": self.rope_theta,
        "partial_rotary_factor": self.partial_rotary_factor,
        "mrope_section": list(self.mrope_section),
        "index_n_heads": self.indexer_n_heads,
        "index_kv_heads": self.indexer_kv_heads,
        "index_head_dim": self.indexer_head_dim,
        "index_select_block_size": self.indexer_compress_ratio,
        "index_select_budget": self.indexer_budget,
        "use_ple_layer": self.use_ple_layer,
        "ple_layer_id": self.ple_layer_id,
        "ple_embed_dim": self.ple_embed_dim,
        "ple_conv_kernel_size": self.ple_conv_kernel_size,
        "ngram_size": self.ngram_size,
        "heads_per_ngram": self.heads_per_ngram,
        "ngram_vocab_size_base": self.ngram_vocab_size_base,
        "make_ngram_vocab_size_divisible_by": self.make_ngram_vocab_size_divisible_by,
        "split_ngram_parts": self.split_ngram_parts,
        "ple_seed": self.ple_seed,
        "ple_eos_token_id": self.eos_token_id,
        "mtp_num_layers": self.num_nextn_predict_layers,
        "mtp_loss_scaling_factor": self.mtp_loss_scaling_factor,
        "use_m3_model": True,
        "param_scan_axis": 0,
        "enable_dropout": False,
    }

  @classmethod
  def qwen3_8_flash_next(cls, **overrides: Any) -> "Qwen3_8FlashNextConfig":
    """Returns the full Qwen3.8-Flash-Next (180B) architecture configuration."""
    return cls(**overrides)

  @classmethod
  def qwen3_8_flash_next_mini(cls, **overrides: Any) -> "Qwen3_8FlashNextConfig":
    """Returns the 4-layer / 1-cycle CPU/CI testing variant (`qwen3.8-flash-next-mini`)."""
    defaults: dict[str, Any] = {
        "model_name": "qwen3.8-flash-next-mini",
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 1024,
        "hc_count": 4,
        "hc_lowrank": 32,
        "shared_expert_intermediate_size": 64,
        "moe_intermediate_size": 64,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "gdn_chunk_size": 16,
        "mrope_section": (3, 3, 2),
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 32,
        "indexer_compress_ratio": 4,
        "indexer_budget": 16,
        "ple_embed_dim": 256,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 4,
        "split_ngram_parts": 4,
        "eos_token_id": 3,
    }
    defaults.update(overrides)
    return cls(**defaults)
