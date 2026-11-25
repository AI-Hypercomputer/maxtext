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

from typing import Any, ClassVar, Literal, TYPE_CHECKING

from maxtext.m3.configs.training_config import TrainingConfig
from maxtext.m3.models.model_config import ModelConfig
from pydantic import Field, model_validator

if TYPE_CHECKING:  # Annotation only: importing `glm5_3_flash_config` loads jax/flax.
  from maxtext.m3.configs.glm5_3_flash_config import Glm5Config

# Remat policies `create_glm5_3_flash_model` implements.
_GLM5_REMAT_POLICIES = ("none", "full")


def _default_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
  """Generates the 3:1 KDA:MLA hybrid attention schedule."""
  pattern = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
  return tuple(pattern[i % 4] for i in range(num_hidden_layers))


class Glm5ModelConfig(ModelConfig):
  """Frozen HuggingFace-native architecture configuration for GLM-5.3-Flash."""

  model_type: str = Field("glm5_next_text", description="HuggingFace text_config model_type identifier.")
  vocab_size: int = Field(154880, gt=0)
  hidden_size: int = Field(..., gt=0)
  intermediate_size: int = Field(..., gt=0)
  num_hidden_layers: int = Field(..., gt=0)
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
  linear_num_key_heads: int = Field(..., gt=0)
  linear_num_value_heads: int = Field(..., gt=0)
  linear_key_head_dim: int = Field(128, gt=0)
  linear_value_head_dim: int = Field(128, gt=0)
  linear_conv_kernel_dim: int = Field(4, gt=0)
  linear_lower_bound: float | None = Field(-5.0)
  # Runtime kernel selection (numerics/performance only; no effect on parameters or checkpoints), not architecture.
  # TODO(m3): move to a TrainingConfig kernel section once one exists.
  kda_implementation: Literal["xla", "xla_chunked", "mosaic"] = Field("mosaic")

  # MLA / DSA attention parameters (NoPE: qk_rope_head_dim == 0)
  num_attention_heads: int = Field(..., gt=0)
  num_key_value_heads: int = Field(..., gt=0)
  q_lora_rank: int = Field(..., ge=0)
  kv_lora_rank: int = Field(512, gt=0)
  qk_nope_head_dim: int = Field(..., gt=0)
  qk_rope_head_dim: int = Field(0, ge=0)
  v_head_dim: int = Field(256, gt=0)
  use_indexer: bool = True
  dsa_attention_implementation: Literal["dense_mask"] = Field("dense_mask")  # Runtime kernel selection (see above).
  index_n_heads: int = Field(32, gt=0)
  index_head_dim: int = Field(128, gt=0)
  index_topk: int = Field(2048, gt=0)
  index_kpool: int = Field(4, ge=1)
  index_kpool_always_select_tail: bool = True
  indexer_loss_scaling_factor: float = Field(
      0.0,
      ge=0.0,
      description="Indexer KL loss weight folded into `aux_loss` (scaled again by `TrainingConfig.aux_loss_weight`).",
  )

  # MoE & Hyper-Connection parameters (native HuggingFace field names)
  moe_intermediate_size: int = Field(2048, gt=0)
  n_routed_experts: int = Field(..., gt=0)
  num_experts_per_tok: int = Field(8, gt=0)
  n_shared_experts: int = Field(1, ge=0)
  n_group: int = Field(1, ge=1)
  topk_group: int = Field(1, ge=1)
  routed_scaling_factor: float = Field(2.5, gt=0.0)
  norm_topk_prob: bool = True
  scoring_func: str = Field("sigmoid")
  swiglu_limit: float = Field(10.0, gt=0.0)
  # The aux-loss-free router-bias step size is a runtime update rule: `TrainingConfig.routed_bias_update_rate`.
  router_aux_loss_coef: float = Field(
      0.001, ge=0.0, description="MoE load-balance loss weight folded into `aux_loss` (scaled by `aux_loss_weight`)."
  )
  hc_mult: int = Field(4, gt=0)
  hc_sinkhorn_iters: int = Field(20, gt=0)
  hc_eps: float = Field(1e-6, gt=0.0)
  rms_norm_eps: float = Field(1e-5, gt=0.0)
  tie_word_embeddings: bool = False
  num_nextn_predict_layers: int = Field(1, ge=0, le=1)
  mtp_loss_scaling_factor: float = Field(
      0.1,
      ge=0.0,
      description="MTP loss weight folded into `aux_loss` (scaled again by `TrainingConfig.aux_loss_weight`).",
  )

  @model_validator(mode="after")
  def _validate_schedules_and_invariants(self) -> "Glm5ModelConfig":
    """Validates the resolved hybrid layer/indexer schedules and MLA/KDA/MoE invariants."""
    l_types = self.resolved_layer_types
    i_types = self.resolved_indexer_types
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

  # Derived schedules are plain properties (not stored, not `@computed_field`) so the frozen config is never mutated
  # after validation and `Glm5ModelConfig(**cfg.model_dump())` round trips still work under `extra="forbid"`.
  @property
  def resolved_layer_types(self) -> tuple[str, ...]:
    """Per-layer attention schedule: `layer_types` with HF `deepseek_sparse_attention` mapped to `full_attention`,
    or the default 3:1 KDA:MLA pattern when unset."""
    if self.layer_types is None:
      return _default_layer_types(self.num_hidden_layers)
    return tuple("full_attention" if lt == "deepseek_sparse_attention" else lt for lt in self.layer_types)

  @property
  def resolved_indexer_types(self) -> tuple[str, ...]:
    """Per-layer sparse indexer schedule: `indexer_types`, or all `'full'` when unset."""
    return ("full",) * self.num_hidden_layers if self.indexer_types is None else tuple(self.indexer_types)

  @property
  def qk_head_dim(self) -> int:
    """Derived query/key head dimension (`qk_nope_head_dim + qk_rope_head_dim`)."""
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def supported_mesh_axes(self) -> frozenset[str]:
    """Mesh axes GLM-5.3-Flash implements (`MODEL_SUPPORTED_MESH_AXES`)."""
    return frozenset({"data", "fsdp", "tensor", "expert"})

  def mesh_divisibility_constraints(self) -> dict[str, dict[str, int]]:
    """Returns dims that must divide evenly by the `tensor` and `expert` mesh axis sizes."""
    return {
        "tensor": {"num_attention_heads": self.num_attention_heads, "num_key_value_heads": self.num_key_value_heads},
        "expert": {"n_routed_experts": self.n_routed_experts},
    }

  PYCONFIG_CONSTANTS: ClassVar[dict[str, Any]] = {
      # Required by the `MaxTextConfig` validator when `use_indexer` is true.
      "attention_type": "mla",
      "attention": "dot_product",
      "packing": False,
      "use_m3_model": True,
  }
  PYCONFIG_KEYS: ClassVar[dict[str, str]] = {
      "base_emb_dim": "hidden_size",
      "base_num_query_heads": "num_attention_heads",
      "base_num_kv_heads": "num_key_value_heads",
      "base_mlp_dim": "intermediate_size",
      "base_moe_mlp_dim": "moe_intermediate_size",
      "base_num_decoder_layers": "num_hidden_layers",
      "vocab_size": "vocab_size",
      "normalization_layer_epsilon": "rms_norm_eps",
      "logits_via_embedding": "tie_word_embeddings",
      "num_experts": "n_routed_experts",
      "num_experts_per_tok": "num_experts_per_tok",
      "shared_experts": "n_shared_experts",
      "routed_scaling_factor": "routed_scaling_factor",
      "first_num_dense_layers": "first_k_dense_replace",
      "q_lora_rank": "q_lora_rank",
      "kv_lora_rank": "kv_lora_rank",
      "qk_nope_head_dim": "qk_nope_head_dim",
      "qk_rope_head_dim": "qk_rope_head_dim",
      "v_head_dim": "v_head_dim",
      "use_indexer": "use_indexer",
      "indexer_n_heads": "index_n_heads",
      "indexer_head_dim": "index_head_dim",
      "indexer_topk": "index_topk",
      "indexer_loss_scaling_factor": "indexer_loss_scaling_factor",
      "routed_score_func": "scoring_func",
      "norm_topk_prob": "norm_topk_prob",
      "n_routing_groups": "n_group",
      "topk_routing_group": "topk_group",
      "mhc_expansion_rate": "hc_mult",
      "sinkhorn_iterations": "hc_sinkhorn_iters",
      "mtp_num_layers": "num_nextn_predict_layers",
      "mtp_loss_scaling_factor": "mtp_loss_scaling_factor",
      "load_balance_loss_weight": "router_aux_loss_coef",
      "mlp_activations_limit": "swiglu_limit",
  }

  # TODO(m3): Transitional. Once `modeling_glm5_3_flash.py` consumes `Glm5ModelConfig` directly (shared m3 model
  # interface), delete this converter and the `Glm5Config` dataclass, and move `Glm5Sharding` next to the model.
  def to_glm5_config(self, train_cfg: TrainingConfig | None = None) -> "Glm5Config":
    """Converts this frozen Pydantic config into a `Glm5Config` dataclass for `modeling_glm5_3_flash.py`.

    Args:
      train_cfg: Runtime settings (`transforms.scan_layers`, `transforms.remat_policy`, `num_vocab_tiling`,
        `routed_bias_update_rate`). `None` keeps `Glm5Config`'s own runtime defaults.
    """
    from maxtext.m3.configs.glm5_3_flash_config import Glm5Config  # pylint: disable=import-outside-toplevel

    runtime: dict[str, Any] = {}
    if train_cfg is not None:
      remat_policy = train_cfg.transforms.remat_policy.value
      if remat_policy not in _GLM5_REMAT_POLICIES:
        raise ValueError(f"GLM-5.3-Flash supports remat_policy in {_GLM5_REMAT_POLICIES}; got {remat_policy!r}.")
      runtime = {
          "scan_layers": train_cfg.transforms.scan_layers,
          "remat_policy": remat_policy,
          "num_vocab_tiling": train_cfg.num_vocab_tiling,
          "routed_bias_update_rate": train_cfg.routed_bias_update_rate,
      }

    return Glm5Config(
        vocab_size=self.vocab_size,
        model_name=self.model_name or "glm5.3-flash",
        hidden_size=self.hidden_size,
        num_hidden_layers=self.num_hidden_layers,
        first_k_dense_replace=self.first_k_dense_replace,
        layer_types=self.resolved_layer_types,
        indexer_types=self.resolved_indexer_types,
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
        moe_lb_loss_weight=self.router_aux_loss_coef,
        mhc_expansion_rate=self.hc_mult,
        mhc_sinkhorn_iters=self.hc_sinkhorn_iters,
        mhc_eps=self.hc_eps,
        rms_norm_eps=self.rms_norm_eps,
        num_nextn_predict_layers=self.num_nextn_predict_layers,
        mtp_loss_scaling_factor=self.mtp_loss_scaling_factor,
        dtype=self.compute_dtype.value,
        weight_dtype=self.param_dtype.value,
        **runtime,
    )

  @classmethod
  def glm5_3_flash(cls) -> "Glm5ModelConfig":
    """Returns the `glm5.3-flash` architecture configuration."""
    return cls(
        model_name="glm5.3-flash",
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=45,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_routed_experts=288,
        q_lora_rank=1536,
        qk_nope_head_dim=256,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
    )
