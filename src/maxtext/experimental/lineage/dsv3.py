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

"""High level DeepSeekV3 model running dense layers then sparse layers."""

from collections.abc import Mapping
from typing import Any

import jax
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_dense_layers
from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_sparse_layers
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import quantization
from maxtext.experimental.lineage.xla_transforms import max_async_overlap_transform as max_async_overlap_transform_module


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3WeightsPytree,
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    mscale: float,
    kv_lora_rank: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mesh: jax.sharding.Mesh,
    norm_fn: dsv3_dense_layers.NormFn,
    rope_fn: dsv3_dense_layers.RopeFn,
    gmm_fn: dsv3_sparse_layers.GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert",
    capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    max_async_overlap_transform: bool = False,
    ragged_buffer_factor: float,
    quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "L E"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Executes full DSv3 model (dense layers followed by sparse layers).

  Args:
    x: Input tokens.
    w: Full DSv3 weights containing stacked `dense` and `sparse` layer weights.
    yarn_freqs: YaRN frequencies.
    splash_kernel: Initialized Splash kernel.
    num_experts: Total number of routed experts.
    num_experts_per_tok: Number of experts selected per token.
    routed_scaling_factor: Scaling factor for routed token scores.
    n_routing_groups: Number of routing groups.
    topk_routing_group: Number of routing groups to choose for node-limited
      routing.
    topk_in_group: Number of selected experts in each routing group.
    qk_head_dim: Non-positional key/query dimension per head.
    rope_head_dim: RoPE dimension per head.
    num_query_heads: Number of query heads.
    mscale: mscale, used for scaling.
    kv_lora_rank: LoRA rank of key/value projection.
    max_position_embeddings: Maximum position embeddings, used for scaling.
    original_max_position_embeddings: Original maximum position embeddings.
    rope_factor: RoPE factor.
    mesh: JAX mesh over which data and model are sharded.
    norm_fn: Normalization function.
    rope_fn: RoPE function.
    gmm_fn: GMM function.
    axis_mapping: Mapping from logical to physical mesh axes.
    expert_axis_name: Expert axis name.
    capacity_factor: Each MoE chunk processes `capacity_factor` times the
      average number of slots routed to local experts per device (local tokens
      times `num_experts_per_tok`), looping over chunks as needed. `1.0`
      suffices for a single chunk under perfectly balanced routing, and the
      number of expert shards guarantees a single chunk.
    router_dtype: Optional dtype for MoE expert selection; see
      `dsv3_router.expert_selection`.
    segment_ids: Optional segment IDs for sequence packing.
    max_async_overlap_transform: Whether to register the max async overlap XLA
      transform.
    ragged_buffer_factor: Safety factor for cross-layer ragged activation bank.
    quant: Optional quantization config.
    expert_permutations: Optional per-layer expert shuffle, an integer array of
      shape `(num_layers, num_experts)`, where `num_layers` is at least the
      number of sparse layers (extra trailing rows, e.g. for MTP layers, are
      ignored here). Row `j` must be a permutation of `range(num_experts)`. With
      `L` experts per expert shard and `p` the entry at `[j, i]`, logical expert
      `i` of sparse layer `j` is computed at physical position `p`, i.e. as
      local expert `p % L` of expert shard `p // L`. The shuffle is applied on
      the fly to the routed expert weights (and undone on their gradients), so
      weights, gradients and `aux` stay in logical expert order and results are
      numerically equivalent up to summation order. `None` performs no shuffle.

  Returns:
    A tuple of (output_tokens, aux). Note that aux is still local and has not
    been reduced across devices to global stats.
  """
  if max_async_overlap_transform:
    max_async_overlap_transform_module.register_transform()

  if expert_permutations is not None:
    num_sparse_layers = jax.tree.leaves(w.sparse)[0].shape[0]
    dsv3_expert_shuffle.validate_expert_permutations(
        expert_permutations,
        num_experts=num_experts,
        num_layers=num_sparse_layers,
    )
    expert_permutations = expert_permutations[:num_sparse_layers]

  x = dsv3_dense_layers.dsv3_dense_layers(
      x,
      w.dense,
      yarn_freqs,
      splash_kernel,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      mscale=mscale,
      kv_lora_rank=kv_lora_rank,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mesh=mesh,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
      quant=quant,
  )
  x, aux = dsv3_sparse_layers.dsv3_sparse_layers(
      x,
      w.sparse,
      yarn_freqs,
      splash_kernel,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      mscale=mscale,
      kv_lora_rank=kv_lora_rank,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mesh=mesh,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      gmm_fn=gmm_fn,
      axis_mapping=axis_mapping,
      expert_axis_name=expert_axis_name,
      capacity_factor=capacity_factor,
      router_dtype=router_dtype,
      segment_ids=segment_ids,
      ragged_buffer_factor=ragged_buffer_factor,
      quant=quant,
      expert_permutations=expert_permutations,
  )
  return x, aux
