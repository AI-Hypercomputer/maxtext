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

"""Generic JAX PyTree types for Flint weights, shardings, batches, and metrics."""

from __future__ import annotations

import dataclasses
from typing import Generic, NamedTuple, TypeVar

import jax
import jaxtyping
import numpy as np

Array = jaxtyping.Array
Float = jaxtyping.Float
Int = jaxtyping.Int

T = TypeVar(
    "T",
    jax.Array,
    np.ndarray,
    jax.ShapeDtypeStruct,
    jax.sharding.NamedSharding,
    jax.sharding.PartitionSpec,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintAttnWeights(Generic[T]):
  """Hybrid GQA attention weights, QK norms, and learnable FP32 sinks.

  2D projections use `[in_features, out_features]` layout; 1D vectors (`q_norm`,
  `k_norm`, `sinks`) gain leading `[N_sb, ...]` when stacked for `lax.scan`.
  """

  q_proj: T
  k_proj: T
  v_proj: T
  o_proj: T
  q_norm: T
  k_norm: T
  sinks: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintMLPWeights(Generic[T]):
  """Biasless SwiGLU MLP weights (`[in_features, out_features]`)."""

  gate_proj: T
  up_proj: T
  down_proj: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintRouterWeights(Generic[T]):
  """Biasless MoE router gate (`[..., hidden_size, num_experts]`)."""

  gate: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintRoutedExpertsWeights(Generic[T]):
  """Packed routed SwiGLU expert weights.

  `gate_up_proj` has shape `[..., E, hidden_size, 2 * moe_intermediate_size]`
  with `gate_proj` in the first half and `up_proj` in the second half along
  `-1`.
  `down_proj` has shape `[..., E, moe_intermediate_size, hidden_size]`.
  """

  gate_up_proj: T
  down_proj: T


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintMoEWeights(Generic[T]):
  """MoE block weights combining router, shared expert, and routed experts."""

  router: FlintRouterWeights[T]
  shared_expert: FlintMLPWeights[T]
  routed_experts: FlintRoutedExpertsWeights[T]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintDenseLayerWeights(Generic[T]):
  """Weights for a dense prefix decoder layer (`layers 0..1`)."""

  input_layernorm: T
  self_attn: FlintAttnWeights[T]
  post_attention_layernorm: T
  mlp: FlintMLPWeights[T]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintSparseLayerWeights(Generic[T]):
  """Weights for a sparse MoE decoder layer (`layers 2..48`)."""

  input_layernorm: T
  self_attn: FlintAttnWeights[T]
  post_attention_layernorm: T
  moe: FlintMoEWeights[T]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintSuperblockWeights(Generic[T]):
  """Stacked 4-layer MoE superblock (`sliding_0..2`, `global_3`) for `scan`."""

  sliding_0: FlintSparseLayerWeights[T]
  sliding_1: FlintSparseLayerWeights[T]
  sliding_2: FlintSparseLayerWeights[T]
  global_3: FlintSparseLayerWeights[T]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintDecoderWeights(Generic[T]):
  """Hybrid decoder weights supporting both scanned and unscanned execution.

  `scanned_blocks` is `None` when `scan_layers=False` or `num_superblocks == 0`.
  """

  prefix_dense: tuple[FlintDenseLayerWeights[T], ...]
  scanned_blocks: FlintSuperblockWeights[T] | None
  unscanned_moe: tuple[FlintSparseLayerWeights[T], ...]
  tail_moe: tuple[FlintSparseLayerWeights[T], ...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintParams(Generic[T]):
  """Full Flint causal language model parameter PyTree."""

  embed: T
  decoder: FlintDecoderWeights[T]
  final_norm: T
  lm_head: T


class FlintBatch(NamedTuple):
  """Token batch with decoupled document `segment_ids` and `loss_mask`."""

  tokens: Int[Array, "batch seq"]
  targets: Int[Array, "batch seq"]
  positions: Int[Array, "batch seq"]
  segment_ids: Int[Array, "batch seq"]
  loss_mask: Int[Array, "batch seq"]


class FlintMoeAux(NamedTuple):
  """Per-layer MoE routing auxiliary statistics and top-k assignments."""

  expert_counts: Float[Array, "num_experts"]
  prob_sum: Float[Array, "num_experts"]
  topk_ids: Int[Array, "... top_k"]
  topk_weights: Float[Array, "... top_k"]


class FlintTrainMetrics(NamedTuple):
  """Scalar training metrics emitted per optimizer step."""

  loss: Float[Array, ""]
  ce_loss: Float[Array, ""]
  aux_loss: Float[Array, ""]
  grad_norm: Float[Array, ""]
  learning_rate: Float[Array, ""]
