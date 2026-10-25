# Copyright 2023-2026 Google LLC
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

"""Configuration and logical-axis sharding rules for GLM-5.3-Flash."""

import dataclasses
from typing import Any

import jax
from maxtext.m3.core import sharding as m3_sharding
import numpy as np


def default_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
  """Generates 3:1 KDA:MLA hybrid schedule for num_hidden_layers."""
  sch = ("linear_attention",) * 3 + ("full_attention",)
  return tuple(sch[i % 4] for i in range(num_hidden_layers))


def default_indexer_types(num_hidden_layers: int) -> tuple[str, ...]:
  """Generates all-'full' indexer types for num_hidden_layers."""
  return ("full",) * num_hidden_layers


def validate_no_packing(decoder_segment_ids: Any) -> None:
  """Host-side only (np.asarray): raises ValueError if decoder_segment_ids has packed segments (> 1)."""
  if decoder_segment_ids is None:
    return
  if bool((np.asarray(decoder_segment_ids) > 1).any()):
    msg = f"Packed sequences unsupported (FU-D5): {decoder_segment_ids}."
    raise ValueError(msg)


_GLM_ONLY_FIELDS = frozenset(
    (
        "layer_types indexer_types linear_num_key_heads linear_num_value_heads"
        " linear_key_head_dim linear_value_head_dim linear_conv_kernel_dim"
        " kda_implementation rope_parameters index_kpool"
        " index_kpool_always_select_tail linear_lower_bound"
        " mhc_eps enable_debug_stats"
    ).split()
)


MTP_LAYER_INDEX = 45  # Hub safetensors layer index of the NextN MTP layer.


@dataclasses.dataclass(frozen=True)
class Glm5Config:
  """GLM-5.3-Flash config.

  All fields map from MaxTextConfig via from_maxtext_config except GLM-only
  fields owned here: layer_types, indexer_types, linear_num_key_heads,
  linear_num_value_heads, linear_key_head_dim, linear_value_head_dim,
  linear_conv_kernel_dim, kda_implementation, qk_head_dim (derived from
  qk_nope_head_dim + qk_rope_head_dim), rope_parameters, index_kpool,
  index_kpool_always_select_tail, linear_lower_bound, mhc_eps,
  enable_debug_stats.
  """

  vocab_size: int = 154880
  model_name: str = "glm5.3-flash"
  hidden_size: int = 4096
  num_hidden_layers: int = 45
  first_k_dense_replace: int = 3
  layer_types: tuple[str, ...] | None = None
  indexer_types: tuple[str, ...] | None = None
  linear_num_key_heads: int = 64
  linear_num_value_heads: int = 64
  linear_key_head_dim: int = 128
  linear_value_head_dim: int = 128
  linear_conv_kernel_dim: int = 4
  linear_lower_bound: float | None = -5.0
  kda_implementation: str = "mosaic"
  num_attention_heads: int = 64
  num_key_value_heads: int = 64
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 256
  qk_rope_head_dim: int = 0
  v_head_dim: int = 256
  rope_parameters: Any = None
  use_indexer: bool = True
  index_n_heads: int = 32
  index_head_dim: int = 128
  index_topk: int = 2048
  # Matches zai-org/GLM-5.3-Flash Hub config.json ("index_kpool": 4), which
  # overrides HF Glm5NextTextConfig.__init__ default (16).
  index_kpool: int = 4
  index_kpool_always_select_tail: bool = True
  indexer_loss_scaling_factor: float = 0.0
  intermediate_size: int = 12288
  moe_intermediate_size: int = 2048
  n_routed_experts: int = 288
  num_experts_per_tok: int = 8
  n_shared_experts: int = 1
  n_group: int = 1
  topk_group: int = 1
  routed_scaling_factor: float = 2.5
  norm_topk_prob: bool = True
  scoring_func: str = "sigmoid"
  mlp_clamp_limit: float = 10.0
  routed_bias_update_rate: float = 0.0
  # Explicit training default matching HF router_aux_loss_coef=0.001 (applied
  # when output_router_logits=True); sole MoE balancing mechanism while
  # routed_bias_update_rate=0.0 (FU-D4).
  moe_lb_loss_weight: float = 0.001
  mhc_expansion_rate: int = 4
  mhc_sinkhorn_iters: int = 20
  mhc_eps: float = 1e-6
  rms_norm_eps: float = 1e-5
  num_nextn_predict_layers: int = 1
  mtp_loss_scaling_factor: float = 0.1
  packing: bool = False
  scan_layers: bool = True
  dtype: str = "bfloat16"
  weight_dtype: str = "float32"
  remat_policy: str = "full"
  enable_debug_stats: bool = False

  def __post_init__(self) -> None:
    if isinstance(self.dtype, (np.dtype, type)):
      object.__setattr__(self, "dtype", np.dtype(self.dtype).name)
    if isinstance(self.weight_dtype, (np.dtype, type)):
      object.__setattr__(self, "weight_dtype", np.dtype(self.weight_dtype).name)
    l_types = default_layer_types(self.num_hidden_layers) if self.layer_types is None else tuple(self.layer_types)
    i_types = default_indexer_types(self.num_hidden_layers) if self.indexer_types is None else tuple(self.indexer_types)
    object.__setattr__(self, "layer_types", l_types)
    object.__setattr__(self, "indexer_types", i_types)

    r_pol = self.remat_policy
    if not isinstance(r_pol, str):
      raise TypeError(f"Glm5Config.remat_policy must be a str, got {type(r_pol).__name__}.")
    if r_pol not in ("none", "full"):
      raise ValueError(f"Glm5Config.remat_policy must be 'none' or 'full'; got {r_pol!r}.")

    if self.packing:
      raise ValueError("packing=True unsupported for GLM-5.3-Flash (FU-D5).")
    n = self.num_hidden_layers
    if len(l_types) != n:
      raise ValueError(f"len(layer_types)={len(l_types)}!={n}.")
    if len(i_types) != n:
      raise ValueError(f"len(indexer_types)={len(i_types)}!={n}.")
    for i, lt in enumerate(l_types):
      if lt not in {"linear_attention", "full_attention"}:
        raise ValueError(f"layer_types[{i}] invalid: {lt!r}.")
    for i, it in enumerate(i_types):
      if it != "full":
        raise ValueError(f"Shared indexer unsupported: {it!r}.")
    rbur = self.routed_bias_update_rate
    if rbur != 0.0:
      raise ValueError(f"routed_bias_update_rate must be 0 (FU-D4): {rbur}.")
    ilsf = self.indexer_loss_scaling_factor
    if ilsf != 0.0:
      raise ValueError(f"indexer_loss_scaling_factor!=0 (FU-D3): {ilsf}.")
    kda = self.kda_implementation
    if not isinstance(kda, str) or kda not in {"xla", "xla_chunked", "mosaic"}:
      raise ValueError(f"kda_implementation invalid: {kda!r}.")
    if self.kda_implementation in {"xla_chunked", "mosaic"} and self.linear_lower_bound is None:
      raise ValueError("linear_lower_bound cannot be None when kda_implementation is" f" {self.kda_implementation!r}.")
    if self.index_kpool < 1:
      raise ValueError(f"index_kpool must be >= 1, got {self.index_kpool}.")
    if self.index_topk % self.index_kpool != 0:
      raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool" f" ({self.index_kpool}).")
    if self.mlp_clamp_limit <= 0.0:
      raise ValueError(f"mlp_clamp_limit must be > 0.0; got {self.mlp_clamp_limit}.")
    if self.qk_rope_head_dim != 0 or self.rope_parameters is not None:
      raise ValueError(f"NoPE violation: {self.qk_rope_head_dim=}, {self.rope_parameters=}.")
    if self.num_key_value_heads != self.num_attention_heads:
      raise ValueError(
          f"num_key_value_heads ({self.num_key_value_heads}) must equal"
          f" num_attention_heads ({self.num_attention_heads})."
      )
    kh, vh = self.linear_num_key_heads, self.linear_num_value_heads
    if kh != vh:
      raise ValueError(f"linear_num_key_heads ({kh}) must equal linear_num_value_heads" f" ({vh}).")
    kd, vd = self.linear_key_head_dim, self.linear_value_head_dim
    if kd != vd:
      raise ValueError(f"linear_key_head_dim ({kd}) must equal linear_value_head_dim ({vd}).")

  @property
  def qk_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  @property
  def linear_num_heads(self) -> int:
    return self.linear_num_key_heads

  @property
  def linear_head_dim(self) -> int:
    return self.linear_key_head_dim

  @classmethod
  def mini_config(cls, **overrides: Any) -> "Glm5Config":
    """Constructs a 5-layer mini Glm5Config covering L0..L4 permutations."""
    n = overrides.pop("num_hidden_layers", 5)
    lt = overrides.pop("layer_types", default_layer_types(n))
    it = overrides.pop("indexer_types", default_indexer_types(n))
    remat_policy = overrides.pop("remat_policy", "none")
    return cls(
        num_hidden_layers=n,
        layer_types=lt,
        indexer_types=it,
        remat_policy=remat_policy,
        **overrides,
    )

  @classmethod
  def from_maxtext_config(cls, config: Any, **overrides: Any) -> "Glm5Config":
    """Maps a MaxText configuration object into a Glm5Config instance without silent fallbacks."""
    if isinstance(config, cls):
      if not overrides:
        return config
      rep = dict(overrides)
      if "num_hidden_layers" in rep and rep["num_hidden_layers"] != config.num_hidden_layers:
        new_n = rep["num_hidden_layers"]
        if "layer_types" not in rep:
          rep["layer_types"] = default_layer_types(new_n)
        if "indexer_types" not in rep:
          rep["indexer_types"] = default_indexer_types(new_n)
      return dataclasses.replace(config, **rep)
    c = config
    if hasattr(c, "use_m3_model") and not c.use_m3_model:
      raise ValueError(
          "glm5.3-flash requires use_m3_model=True; legacy decoder_block is not" " supported for GLM-5.3-Flash."
      )
    renames = {
        "hidden_size": "emb_dim",
        "num_hidden_layers": "num_decoder_layers",
        "num_attention_heads": "num_query_heads",
        "num_key_value_heads": "num_kv_heads",
        "intermediate_size": "mlp_dim",
        "moe_intermediate_size": "moe_mlp_dim",
        "first_k_dense_replace": "first_num_dense_layers",
        "n_routed_experts": "num_experts",
        "n_shared_experts": "shared_experts",
        "rms_norm_eps": "normalization_layer_epsilon",
        "index_n_heads": "indexer_n_heads",
        "index_head_dim": "indexer_head_dim",
        "index_topk": "indexer_topk",
        "scoring_func": "routed_score_func",
        "n_group": "n_routing_groups",
        "topk_group": "topk_routing_group",
        "mhc_sinkhorn_iters": "sinkhorn_iterations",
        "num_nextn_predict_layers": "mtp_num_layers",
        "moe_lb_loss_weight": "load_balance_loss_weight",
        "mlp_clamp_limit": "mlp_activations_limit",
    }
    kwargs = {}
    for f in dataclasses.fields(cls):
      if f.name in _GLM_ONLY_FIELDS:
        if hasattr(c, f.name):
          kwargs[f.name] = getattr(c, f.name)
      else:
        kwargs[f.name] = getattr(c, renames.get(f.name, f.name))
    if "remat_policy" in kwargs and kwargs["remat_policy"] is None:
      kwargs["remat_policy"] = "none"
    kwargs.update(overrides)
    if "num_hidden_layers" in overrides:
      if "layer_types" not in overrides:
        kwargs["layer_types"] = default_layer_types(kwargs["num_hidden_layers"])
      if "indexer_types" not in overrides:
        kwargs["indexer_types"] = default_indexer_types(kwargs["num_hidden_layers"])
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class Glm5Sharding(m3_sharding.Sharding):
  """Logical-axis sharding rules for GLM-5.3-Flash."""

  fsdp_axis: str = "fsdp"
  dp_axis: str = "data"
  ep_axis: str = "expert"
  tp_axis: str = "tensor"
  use_tp: bool = False
  ep_in_batch: bool = False
  ep_size: int = 1

  @classmethod
  def from_mesh(cls, mesh: jax.sharding.Mesh) -> "Glm5Sharding":
    """Constructs Glm5Sharding from a jax.sharding.Mesh inspecting axis names."""
    axis_names = set(mesh.axis_names)
    required = {"data", "fsdp", "tensor", "expert"}
    if not required.issubset(axis_names):
      raise ValueError(
          f"Unsupported mesh axis_names {mesh.axis_names}; expected ('data',"
          f" 'fsdp', 'tensor', 'expert'), missing {required - axis_names}."
      )
    for ax in mesh.axis_names:
      if ax not in required and mesh.shape[ax] > 1:
        raise NotImplementedError(
            f"Mesh axis {ax!r} has size {mesh.shape[ax]} > 1, which is not yet" " supported in Glm5Sharding."
        )
    use_tp = mesh.shape["tensor"] > 1
    ep_size = int(mesh.shape["expert"])
    return cls(use_tp=use_tp, ep_size=ep_size)

  def map_axis(
      self,
      axis: str,
      tensor_name: str,
      tensor_type: m3_sharding.TensorType,
  ) -> m3_sharding.MeshAxes:
    """Maps a logical axis name to mesh axis names for parameters or activations."""
    act = tensor_type == m3_sharding.TensorType.ACTIVATION
    if axis == "batch":
      if act:
        return (self.dp_axis, self.fsdp_axis, self.ep_axis) if self.ep_in_batch else (self.dp_axis, self.fsdp_axis)
      return None
    if axis == "expert":
      return None if act else self.ep_axis
    if axis in {"embed", "q_lora", "kv_lora"}:
      return None if act else self.fsdp_axis
    if axis in {"vocab", "heads", "kv", "mlp"}:
      return self.tp_axis if self.use_tp else None
    if axis == "norm" or (axis == "length" and act):
      return None
    msg = f"Unknown axis {axis!r} for {tensor_name!r} ({tensor_type})."
    raise KeyError(msg)
