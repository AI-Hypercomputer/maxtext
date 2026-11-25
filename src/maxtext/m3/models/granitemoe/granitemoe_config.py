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

from typing import Any, ClassVar, Literal, TYPE_CHECKING

from maxtext.configs.components import DType
from maxtext.m3.configs.training_config import TrainingConfig
from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator

if TYPE_CHECKING:  # Annotation only: importing `modeling_granitemoe` loads jax/flax.
  from maxtext.m3.models.granitemoe import modeling_granitemoe

# Remat policies `modeling_granitemoe.GraniteMoeConfig` accepts.
_GRANITEMOE_REMAT_POLICIES = ("none", "full", "minimal")


class GraniteMoeModelConfig(ModelConfig):
  """Frozen HuggingFace-native architecture configuration for GraniteMoE."""

  model_type: str = Field("granitemoe", description="HuggingFace model_type identifier.")
  vocab_size: int = Field(49155, gt=0, description="Vocabulary size.")
  hidden_size: int = Field(..., gt=0, description="Model hidden / embedding dimension.")
  intermediate_size: int = Field(..., gt=0, description="Per-expert SwiGLU intermediate dimension.")
  num_hidden_layers: int = Field(..., gt=0, description="Number of transformer decoder layers.")
  num_attention_heads: int = Field(..., gt=0, description="Number of attention query heads.")
  num_key_value_heads: int = Field(..., gt=0, description="Number of grouped-query KV heads.")
  num_local_experts: int = Field(..., gt=0, description="Number of routed MoE experts per layer.")
  num_experts_per_tok: int = Field(..., gt=0, description="Active top-k experts per token.")
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
  router_aux_loss_coef: float = Field(
      0.0, ge=0.0, description="MoE load-balance loss weight folded into `aux_loss` (scaled by `aux_loss_weight`)."
  )
  compute_dtype: DType = Field(DType.FLOAT32, description="Forward/backward activation compute dtype.")
  param_scan_axis: int = Field(0, ge=0, description="Scanned-layer stacking axis (m3-native checkpoints use 0).")

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

  def supported_mesh_axes(self) -> frozenset[str]:
    """Mesh axes GraniteMoE implements (`MODEL_SUPPORTED_MESH_AXES`)."""
    return frozenset({"data", "fsdp", "tensor", "expert"})

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    """Returns dims that must divide evenly by the `tensor` and `expert` mesh axis sizes."""
    return {
        "tensor": {"num_attention_heads": self.num_attention_heads, "num_key_value_heads": self.num_key_value_heads},
        "expert": {"num_local_experts": self.num_local_experts},
    }

  PYCONFIG_CONSTANTS: ClassVar[dict[str, Any]] = {
      "use_m3_model": True,
      "mlp_activations": ["silu", "linear"],
      "enable_dropout": False,
      "tokenizer_type": "huggingface",
  }
  PYCONFIG_KEYS: ClassVar[dict[str, str]] = {
      "model_name": "model_name",
      "param_scan_axis": "param_scan_axis",
      "base_emb_dim": "hidden_size",
      "base_num_query_heads": "num_attention_heads",
      "base_num_kv_heads": "num_key_value_heads",
      "base_mlp_dim": "intermediate_size",
      "base_moe_mlp_dim": "intermediate_size",
      "base_num_decoder_layers": "num_hidden_layers",
      "head_dim": "head_dim",  # Derived (`hidden_size // num_attention_heads`).
      "num_experts": "num_local_experts",
      "num_experts_per_tok": "num_experts_per_tok",
      "vocab_size": "vocab_size",
      "normalization_layer_epsilon": "rms_norm_eps",
      "rope_max_timescale": "rope_theta",
      "logits_via_embedding": "tie_word_embeddings",
      "load_balance_loss_weight": "router_aux_loss_coef",
  }

  def to_granitemoe_config(self, train_cfg: TrainingConfig | None = None) -> "modeling_granitemoe.GraniteMoeConfig":
    """Converts this frozen config into `modeling_granitemoe.GraniteMoeConfig`.

    Args:
      train_cfg: Runtime settings (`transforms.scan_layers`, `transforms.remat_policy`, `matmul_precision`).
        `None` keeps `GraniteMoeConfig`'s own runtime defaults.
    """
    from maxtext.m3.models.granitemoe.modeling_granitemoe import GraniteMoeConfig  # pylint: disable=import-outside-toplevel

    runtime: dict[str, Any] = {}
    if train_cfg is not None:
      remat_policy = train_cfg.transforms.remat_policy.value
      if remat_policy not in _GRANITEMOE_REMAT_POLICIES:
        raise ValueError(f"GraniteMoE supports remat_policy in {_GRANITEMOE_REMAT_POLICIES}; got {remat_policy!r}.")
      runtime = {
          "scan_layers": train_cfg.transforms.scan_layers,
          "remat_policy": remat_policy,
          "matmul_precision": train_cfg.matmul_precision.value,
      }

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
        **runtime,
    )

  @classmethod
  def granite_3_0_1b_a400m(cls) -> "GraniteMoeModelConfig":
    """Returns the `granite-3.0-1b-a400m` architecture configuration."""
    return cls(
        model_name="granite-3.0-1b-a400m",
        hidden_size=1024,
        intermediate_size=512,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_local_experts=32,
        num_experts_per_tok=8,
    )
