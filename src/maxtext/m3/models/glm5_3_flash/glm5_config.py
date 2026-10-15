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
        " kda_implementation qk_head_dim rope_parameters index_kpool"
        " mlp_clamp_limit moe_lb_loss_weight mhc_eps enable_debug_stats"
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
  mlp_clamp_limit, moe_lb_loss_weight, mhc_eps, enable_debug_stats.
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
  kda_implementation: str = "mosaic"
  num_attention_heads: int = 64
  num_key_value_heads: int = 64
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 256
  qk_rope_head_dim: int = 0
  v_head_dim: int = 256
  qk_head_dim: int = 256
  head_dim: int = 128
  rope_parameters: Any = None
  use_indexer: bool = True
  index_n_heads: int = 32
  index_head_dim: int = 128
  index_topk: int = 2048
  index_kpool: int = 4
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
  routed_bias: bool = True
  mlp_clamp_limit: float = 10.0
  routed_bias_update_rate: float = 0.0
  moe_lb_loss_weight: float = 0.001
  mhc_expansion_rate: int = 4
  mhc_sinkhorn_iters: int = 20
  mhc_eps: float = 1e-6
  rms_norm_eps: float = 1e-5
  num_nextn_predict_layers: int = 1
  mtp_loss_scaling_factor: float = 0.1
  dropout_rate: float = 0.0
  packing: bool = False
  scan_layers: bool = True
  dtype: str = "bfloat16"
  weight_dtype: str = "float32"
  enable_debug_stats: bool = False

  def __post_init__(self) -> None:
    l_types = self.layer_types
    if l_types is None:
      l_types = default_layer_types(self.num_hidden_layers)
    i_types = self.indexer_types
    if i_types is None:
      i_types = default_indexer_types(self.num_hidden_layers)
    object.__setattr__(self, "layer_types", l_types)
    object.__setattr__(self, "indexer_types", i_types)

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
    if self.qk_rope_head_dim != 0 or self.rope_parameters is not None:
      raise ValueError(f"NoPE violation: {self.qk_rope_head_dim=}.")

  @classmethod
  def mini_config(cls, **overrides: Any) -> "Glm5Config":
    """Constructs a 5-layer mini Glm5Config covering L0..L4 permutations."""
    n = overrides.pop("num_hidden_layers", 5)
    lt = overrides.pop("layer_types", default_layer_types(n))
    it = overrides.pop("indexer_types", default_indexer_types(n))
    return cls(
        num_hidden_layers=n,
        layer_types=lt,
        indexer_types=it,
        **overrides,
    )

  @classmethod
  def from_maxtext_config(cls, config: Any, **overrides: Any) -> "Glm5Config":
    """Maps a MaxText configuration object into a Glm5Config instance without silent fallbacks."""
    if isinstance(config, cls):
      return config
    c = config
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
    }
    kwargs = {}
    for f in dataclasses.fields(cls):
      if f.name not in _GLM_ONLY_FIELDS:
        kwargs[f.name] = getattr(c, renames.get(f.name, f.name))
    kwargs["qk_head_dim"] = int(c.qk_nope_head_dim) + int(c.qk_rope_head_dim)
    kwargs.update(overrides)
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class Glm5Sharding(m3_sharding.Sharding):
  """Logical-axis sharding rules for GLM-5.3-Flash."""

  fsdp_axis: str = "fsdp"
  dp_axis: str = "dp"
  ep_axis: str = "ep"
  tp_axis: str = "tp"
  use_tp: bool = False

  def map_axis(
      self,
      axis: str,
      tensor_name: str,
      tensor_type: m3_sharding.TensorType,
  ) -> m3_sharding.MeshAxes:
    """Maps a logical axis name to mesh axis names for parameters or activations."""
    act = tensor_type == m3_sharding.TensorType.ACTIVATION
    if axis == "batch":
      return (self.dp_axis, self.fsdp_axis) if act else None
    if axis == "expert":
      return None if act else self.ep_axis
    if axis in {"mlp_moe", "embed"}:
      return None if act else self.fsdp_axis
    if axis in {"vocab", "heads", "kv", "mlp"}:
      return self.tp_axis if self.use_tp else None
    if axis == "norm" or (axis == "length" and act):
      return None
    msg = f"Unknown axis {axis!r} for {tensor_name!r} ({tensor_type})."
    raise KeyError(msg)
