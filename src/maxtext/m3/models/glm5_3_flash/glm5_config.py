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

"""Model Garden config for GLM-5.3-Flash (hybrid KDA/MLA + MoE)."""

from typing import TYPE_CHECKING, Any, Literal

from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator

if TYPE_CHECKING:
  from maxtext.m3.configs.glm5_3_flash_config import Glm5Config


def _default_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
  """Generates the 3:1 KDA:MLA hybrid attention schedule."""
  pattern = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
  return tuple(pattern[i % 4] for i in range(num_hidden_layers))


class Glm5ModelConfig(ModelConfig):
  """Frozen HuggingFace-native architecture configuration for GLM-5.3-Flash."""

  model_type: str = Field("glm5_next_text", description="HuggingFace text_config model_type identifier.")
  model_name: str = Field("glm5.3-flash", description="Canonical GLM-5.3-Flash model identifier.")
  vocab_size: int = Field(154880, gt=0)
  hidden_size: int = Field(4096, gt=0)
  intermediate_size: int = Field(12288, gt=0)
  num_hidden_layers: int = Field(45, gt=0)
  first_k_dense_replace: int = Field(3, ge=0)
  layer_types: tuple[str, ...] | None = Field(
      None,
      description="Per-layer attention type tuple ('linear_attention' or 'full_attention').",
  )
  indexer_types: tuple[str, ...] | None = Field(
      None,
      description="Per-layer sparse indexer type tuple (currently 'full').",
  )
  mtp_indexer_type: Literal["full"] = Field("full", description="Sparse indexer type for MTP layers.")

  # Hybrid KDA linear attention parameters
  linear_num_key_heads: int = Field(64, gt=0)
  linear_num_value_heads: int = Field(64, gt=0)
  linear_key_head_dim: int = Field(128, gt=0)
  linear_value_head_dim: int = Field(128, gt=0)
  linear_conv_kernel_dim: int = Field(4, gt=0)
  linear_lower_bound: float | None = Field(-5.0)
  kda_implementation: Literal["xla", "xla_chunked", "mosaic"] = Field("mosaic")

  # MLA / DSA attention parameters (NoPE: qk_rope_head_dim == 0)
  num_attention_heads: int = Field(64, gt=0)
  num_key_value_heads: int = Field(64, gt=0)
  q_lora_rank: int = Field(1536, ge=0)
  kv_lora_rank: int = Field(512, gt=0)
  qk_nope_head_dim: int = Field(256, gt=0)
  qk_rope_head_dim: int = Field(0, ge=0)
  v_head_dim: int = Field(256, gt=0)
  use_indexer: bool = True
  dsa_attention_implementation: Literal["dense_mask"] = Field("dense_mask")
  index_n_heads: int = Field(32, gt=0)
  index_head_dim: int = Field(128, gt=0)
  index_topk: int = Field(2048, gt=0)
  index_kpool: int = Field(4, ge=1)
  index_kpool_always_select_tail: bool = True
  indexer_loss_scaling_factor: float = Field(0.0, ge=0.0)

  # MoE & Hyper-Connection parameters (native HuggingFace field names)
  moe_intermediate_size: int = Field(2048, gt=0)
  n_routed_experts: int = Field(288, gt=0)
  num_experts_per_tok: int = Field(8, gt=0)
  n_shared_experts: int = Field(1, ge=0)
  n_group: int = Field(1, ge=1)
  topk_group: int = Field(1, ge=1)
  routed_scaling_factor: float = Field(2.5, gt=0.0)
  norm_topk_prob: bool = True
  scoring_func: str = Field("sigmoid")
  swiglu_limit: float = Field(10.0, gt=0.0)
  routed_bias_update_rate: float = Field(0.0, ge=0.0)
  router_aux_loss_coef: float = Field(0.001, ge=0.0)
  hc_mult: int = Field(4, gt=0)
  hc_sinkhorn_iters: int = Field(20, gt=0)
  hc_eps: float = Field(1e-6, gt=0.0)
  rms_norm_eps: float = Field(1e-5, gt=0.0)
  tie_word_embeddings: bool = False
  num_nextn_predict_layers: int = Field(1, ge=0, le=1)
  mtp_loss_scaling_factor: float = Field(0.1, ge=0.0)

  @model_validator(mode="after")
  def _validate_and_populate_schedules(self) -> "Glm5ModelConfig":
    """Populates default hybrid layer/indexer schedules and validates MLA/KDA/MoE invariants."""
    if self.layer_types is None:
      l_types = _default_layer_types(self.num_hidden_layers)
    else:
      l_types = tuple("full_attention" if lt == "deepseek_sparse_attention" else lt for lt in self.layer_types)
    object.__setattr__(self, "layer_types", l_types)

    i_types = ("full",) * self.num_hidden_layers if self.indexer_types is None else tuple(self.indexer_types)
    object.__setattr__(self, "indexer_types", i_types)

    if len(l_types) != self.num_hidden_layers:
      raise ValueError(f"len(layer_types) ({len(l_types)}) must equal num_hidden_layers ({self.num_hidden_layers}).")
    if len(i_types) != self.num_hidden_layers:
      raise ValueError(f"len(indexer_types) ({len(i_types)}) must equal num_hidden_layers ({self.num_hidden_layers}).")
    for idx, lt in enumerate(l_types):
      if lt not in ("linear_attention", "full_attention"):
        raise ValueError(f"layer_types[{idx}] invalid: {lt!r}.")
    for idx, it in enumerate(i_types):
      if it != "full":
        raise ValueError(f"indexer_types[{idx}] invalid: {it!r}.")
    if self.qk_rope_head_dim != 0:
      raise ValueError(f"GLM-5.3-Flash requires NoPE (qk_rope_head_dim=0), got {self.qk_rope_head_dim}.")
    if self.num_key_value_heads != self.num_attention_heads:
      raise ValueError(
          f"num_key_value_heads ({self.num_key_value_heads}) must equal "
          f"num_attention_heads ({self.num_attention_heads})."
      )
    if self.linear_num_key_heads != self.linear_num_value_heads:
      raise ValueError("linear_num_key_heads must equal linear_num_value_heads.")
    if self.linear_key_head_dim != self.linear_value_head_dim:
      raise ValueError("linear_key_head_dim must equal linear_value_head_dim.")
    if self.index_topk % self.index_kpool != 0:
      raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")
    if self.kda_implementation in ("xla_chunked", "mosaic") and self.linear_lower_bound is None:
      raise ValueError(f"linear_lower_bound cannot be None when kda_implementation={self.kda_implementation!r}.")
    if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
      raise ValueError(f"num_experts_per_tok ({self.num_experts_per_tok}) must be in [1, {self.n_routed_experts}].")
    return self

  @property
  def qk_head_dim(self) -> int:
    """Derived query/key head dimension (`qk_nope_head_dim + qk_rope_head_dim`)."""
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def to_pyconfig_dict(self) -> dict[str, Any]:
    """Returns `MaxTextConfig` field defaults matching `glm5.3-flash.yml`."""
    return {
        "base_emb_dim": self.hidden_size,
        "base_num_query_heads": self.num_attention_heads,
        "base_num_kv_heads": self.num_key_value_heads,
        "base_mlp_dim": self.intermediate_size,
        "base_moe_mlp_dim": self.moe_intermediate_size,
        "base_num_decoder_layers": self.num_hidden_layers,
        "vocab_size": self.vocab_size,
        "normalization_layer_epsilon": self.rms_norm_eps,
        "num_experts": self.n_routed_experts,
        "num_experts_per_tok": self.num_experts_per_tok,
        "shared_experts": self.n_shared_experts,
        "routed_scaling_factor": self.routed_scaling_factor,
        "first_num_dense_layers": self.first_k_dense_replace,
        "q_lora_rank": self.q_lora_rank,
        "kv_lora_rank": self.kv_lora_rank,
        "qk_nope_head_dim": self.qk_nope_head_dim,
        "qk_rope_head_dim": self.qk_rope_head_dim,
        "v_head_dim": self.v_head_dim,
        # Required by MaxTextConfig validator when use_indexer is true:
        "attention_type": "mla",
        "attention": "dot_product",
        "use_indexer": self.use_indexer,
        "indexer_n_heads": self.index_n_heads,
        "indexer_head_dim": self.index_head_dim,
        "indexer_topk": self.index_topk,
        "indexer_loss_scaling_factor": self.indexer_loss_scaling_factor,
        "routed_score_func": self.scoring_func,
        "routed_bias_update_rate": self.routed_bias_update_rate,
        "norm_topk_prob": self.norm_topk_prob,
        "n_routing_groups": self.n_group,
        "topk_routing_group": self.topk_group,
        "mhc_expansion_rate": self.hc_mult,
        "sinkhorn_iterations": self.hc_sinkhorn_iters,
        "mtp_num_layers": self.num_nextn_predict_layers,
        "mtp_loss_scaling_factor": self.mtp_loss_scaling_factor,
        "load_balance_loss_weight": self.router_aux_loss_coef,
        "mlp_activations_limit": self.swiglu_limit,
        "packing": False,
        "use_m3_model": True,
    }

  def to_glm5_config(
      self,
      *,
      scan_layers: bool = True,
      remat_policy: str = "full",
      num_vocab_tiling: int = 1,
  ) -> "Glm5Config":
    """Converts this frozen Pydantic config into a `Glm5Config` dataclass for `modeling_glm5_3_flash.py`."""
    from maxtext.m3.configs.glm5_3_flash_config import Glm5Config  # pylint: disable=import-outside-toplevel

    return Glm5Config(
        vocab_size=self.vocab_size,
        model_name=self.model_name or "glm5.3-flash",
        hidden_size=self.hidden_size,
        num_hidden_layers=self.num_hidden_layers,
        first_k_dense_replace=self.first_k_dense_replace,
        layer_types=self.layer_types,
        indexer_types=self.indexer_types,
        mtp_indexer_type=self.mtp_indexer_type,
        linear_num_key_heads=self.linear_num_key_heads,
        linear_num_value_heads=self.linear_num_value_heads,
        linear_key_head_dim=self.linear_key_head_dim,
        linear_value_head_dim=self.linear_value_head_dim,
        linear_conv_kernel_dim=self.linear_conv_kernel_dim,
        linear_lower_bound=self.linear_lower_bound,
        kda_implementation=self.kda_implementation,
        num_attention_heads=self.num_attention_heads,
        num_key_value_heads=self.num_key_value_heads,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        qk_nope_head_dim=self.qk_nope_head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        use_indexer=self.use_indexer,
        dsa_attention_implementation=self.dsa_attention_implementation,
        index_n_heads=self.index_n_heads,
        index_head_dim=self.index_head_dim,
        index_topk=self.index_topk,
        index_kpool=self.index_kpool,
        index_kpool_always_select_tail=self.index_kpool_always_select_tail,
        indexer_loss_scaling_factor=self.indexer_loss_scaling_factor,
        intermediate_size=self.intermediate_size,
        moe_intermediate_size=self.moe_intermediate_size,
        n_routed_experts=self.n_routed_experts,
        num_experts_per_tok=self.num_experts_per_tok,
        n_shared_experts=self.n_shared_experts,
        n_group=self.n_group,
        topk_group=self.topk_group,
        routed_scaling_factor=self.routed_scaling_factor,
        norm_topk_prob=self.norm_topk_prob,
        scoring_func=self.scoring_func,
        mlp_clamp_limit=self.swiglu_limit,
        routed_bias_update_rate=self.routed_bias_update_rate,
        moe_lb_loss_weight=self.router_aux_loss_coef,
        mhc_expansion_rate=self.hc_mult,
        mhc_sinkhorn_iters=self.hc_sinkhorn_iters,
        mhc_eps=self.hc_eps,
        rms_norm_eps=self.rms_norm_eps,
        num_nextn_predict_layers=self.num_nextn_predict_layers,
        mtp_loss_scaling_factor=self.mtp_loss_scaling_factor,
        scan_layers=scan_layers,
        dtype=self.compute_dtype.value,
        weight_dtype=self.param_dtype.value,
        remat_policy=remat_policy,
        num_vocab_tiling=num_vocab_tiling,
    )

  @classmethod
  def glm5_3_flash(cls, **overrides: Any) -> "Glm5ModelConfig":
    """Returns the full GLM-5.3-Flash architecture configuration."""
    return cls(**overrides)
