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

"""Model Garden config for the GraniteMoE model family."""

from typing import Any, Literal

from maxtext.configs.types import DType
from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator


class GraniteMoeModelConfig(ModelConfig):
  """Frozen HuggingFace-native architecture configuration for GraniteMoE."""

  model_type: str = Field("granitemoe", description="HuggingFace model_type identifier.")
  model_name: str = Field("granite-3.0-1b-a400m", description="Canonical GraniteMoE model identifier.")
  vocab_size: int = Field(49155, gt=0, description="Vocabulary size.")
  hidden_size: int = Field(1024, gt=0, description="Model hidden / embedding dimension.")
  intermediate_size: int = Field(512, gt=0, description="Per-expert SwiGLU intermediate dimension.")
  num_hidden_layers: int = Field(24, gt=0, description="Number of transformer decoder layers.")
  num_attention_heads: int = Field(16, gt=0, description="Number of attention query heads.")
  num_key_value_heads: int = Field(8, gt=0, description="Number of grouped-query KV heads.")
  num_local_experts: int = Field(32, gt=0, description="Number of routed MoE experts per layer.")
  num_experts_per_tok: int = Field(8, gt=0, description="Active top-k experts per token.")
  rms_norm_eps: float = Field(1e-6, gt=0.0, description="RMSNorm epsilon.")
  rope_theta: float = Field(10000.0, gt=0.0, description="RoPE base period / max timescale.")
  attention_bias: bool = Field(False, description="Include bias terms in Q/K/V/O projections.")
  attention_dropout: float = Field(0.0, ge=0.0, lt=1.0, description="Attention dropout probability.")
  hidden_act: Literal["silu"] = Field("silu", description="Expert MLP activation function.")
  tie_word_embeddings: bool = Field(True, description="Whether input and output embeddings are tied.")
  embedding_multiplier: float = Field(12.0, description="Scaling factor applied to input embeddings.")
  attention_multiplier: float = Field(0.015625, gt=0.0, description="Attention logit scaling factor.")
  residual_multiplier: float = Field(0.22, description="Residual branch scaling multiplier.")
  logits_scaling: float = Field(6.0, description="Output logits divisor (must be non-zero).")
  router_aux_loss_coef: float = Field(0.0, ge=0.0, description="MoE load-balancing auxiliary loss weight.")
  compute_dtype: DType = Field(DType.FLOAT32, description="Forward/backward activation compute dtype.")

  @model_validator(mode="after")
  def _validate_granitemoe_invariants(self) -> "GraniteMoeModelConfig":
    """Validates GraniteMoE head divisibility, MoE expert bounds, and non-zero logit scaling."""
    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})."
      )
    head_dim = self.hidden_size // self.num_attention_heads
    if head_dim % 2 != 0:
      raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")
    if self.num_attention_heads % self.num_key_value_heads != 0:
      raise ValueError(
          f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
          f"num_key_value_heads ({self.num_key_value_heads})."
      )
    if not 1 <= self.num_experts_per_tok <= self.num_local_experts:
      raise ValueError(f"num_experts_per_tok ({self.num_experts_per_tok}) must be in [1, {self.num_local_experts}].")
    if self.logits_scaling == 0.0:
      raise ValueError("logits_scaling must be non-zero.")
    return self

  @property
  def head_dim(self) -> int:
    return self.hidden_size // self.num_attention_heads

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults for `granite-3.0-1b-a400m`."""
    return {
        "model_name": self.model_name,
        "use_m3_model": True,
        "param_scan_axis": 0,
        "base_emb_dim": self.hidden_size,
        "base_num_query_heads": self.num_attention_heads,
        "base_num_kv_heads": self.num_key_value_heads,
        "base_mlp_dim": self.intermediate_size,
        "base_moe_mlp_dim": self.intermediate_size,
        "base_num_decoder_layers": self.num_hidden_layers,
        "head_dim": self.head_dim,
        "num_experts": self.num_local_experts,
        "num_experts_per_tok": self.num_experts_per_tok,
        "mlp_activations": ["silu", "linear"],
        "vocab_size": self.vocab_size,
        "normalization_layer_epsilon": self.rms_norm_eps,
        "rope_max_timescale": self.rope_theta,
        "logits_via_embedding": self.tie_word_embeddings,
        "load_balance_loss_weight": self.router_aux_loss_coef,
        "enable_dropout": False,
        "tokenizer_type": "huggingface",
    }

  def to_granitemoe_config(
      self,
      *,
      scan_layers: bool = True,
      remat_policy: str = "none",
      matmul_precision: str = "default",
  ) -> Any:
    """Converts this frozen config into `modeling_granitemoe.GraniteMoeConfig`."""
    from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeConfig  # pylint: disable=import-outside-toplevel

    return GraniteMoeConfig(
        vocab_size=self.vocab_size,
        hidden_size=self.hidden_size,
        intermediate_size=self.intermediate_size,
        num_hidden_layers=self.num_hidden_layers,
        num_attention_heads=self.num_attention_heads,
        num_key_value_heads=self.num_key_value_heads,
        num_local_experts=self.num_local_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        rms_norm_eps=self.rms_norm_eps,
        rope_theta=self.rope_theta,
        attention_bias=self.attention_bias,
        attention_dropout=self.attention_dropout,
        hidden_act=self.hidden_act,
        tie_word_embeddings=self.tie_word_embeddings,
        embedding_multiplier=self.embedding_multiplier,
        attention_multiplier=self.attention_multiplier,
        residual_multiplier=self.residual_multiplier,
        logits_scaling=self.logits_scaling,
        router_aux_loss_coef=self.router_aux_loss_coef,
        dtype=self.compute_dtype.value,
        weight_dtype=self.param_dtype.value,
        scan_layers=scan_layers,
        remat_policy=remat_policy,
        matmul_precision=matmul_precision,
    )

  @classmethod
  def granite_3_0_1b_a400m(cls, **overrides: Any) -> "GraniteMoeModelConfig":
    """Returns the Granite-3.0-1B-A400M architecture configuration."""
    return cls(**overrides)
