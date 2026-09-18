# Copyright 2023–2026 Google LLC
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

"""DeepSeekV3 scan over multiple sparse layers."""

from collections.abc import Mapping
import functools
from typing import Any, Protocol

import jax
import jaxtyping as jt
from maxtext.models.deepseek_lineage import dsv3_sparse_layer
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops
import typeguard


class NormFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*input_dims"],
      scale: jt.Num[jax.Array, "..."] | None = None,
  ) -> jt.Num[jax.Array, "*input_dims"]:
    ...


class RopeFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*BT N D"],
      freqs: tuple[
          jt.Num[jax.Array, "*BT 1 D"],
          jt.Num[jax.Array, "*BT 1 D"],
      ],
  ) -> jt.Num[jax.Array, "*BT N D"]:
    ...


class GmmFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "BT D"],
      w: jt.Num[jax.Array, "E D F"],
      group_sizes: jt.Num[jax.Array, "E"],
  ) -> jt.Num[jax.Array, "BT F"]:
    ...


@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layers(
    x: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
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
    norm_fn: NormFn,
    rope_fn: RopeFn,
    gmm_fn: GmmFn,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert",
    capacity_factor: float = 8.0,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Executes multiple DSv3 sparse (MoE) layers.

  Args:
    x: Input tokens.
    w: DSv3 weights, stacked for all layers on the first dimension.
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
    kv_lora_rank: LoRA rank of the key/value projection.
    max_position_embeddings: Maximum position embeddings, used for scaling.
    original_max_position_embeddings: Original maximum position embeddings, used
      for scaling.
    rope_factor: RoPE factor, used for scaling.
    mesh: JAX mesh over which the data and model are sharded.
    norm_fn: Normalization function.
    rope_fn: RoPE function.
    gmm_fn: GMM function.
    axis_mapping: Mapping from logical to physical mesh axes.
    expert_axis_name: Expert axis name.
    capacity_factor: Capacity factor determining the destination buffer size for
      dispatch: `num_experts_per_tok` guarantees single-iteration processing
        without looping under worst-case imbalance; `1.0` guarantees forward
        progress under worst-case imbalance (looping may occur); in perfectly
        balanced routing, `num_experts_per_tok // ep` suffices for single pass.
    segment_ids: Optional segment IDs for sequence packing.

  Returns:
    A tuple of (output_tokens, aux).
  """

  mb0, mb1 = ops.split_microbatches(x, mesh=mesh)
  yarn_freqs_0, yarn_freqs_1 = ops.split_microbatches(yarn_freqs, mesh=mesh)
  segment_ids_0, segment_ids_1 = ops.split_microbatches(segment_ids, mesh=mesh)

  _collect_w = functools.partial(
      dsv3_sparse_layer.collect_w,
      axis_mapping=axis_mapping,
  )
  _prologue = functools.partial(
      dsv3_sparse_layer.dsv3_sparse_layer_prologue,
      yarn_freqs=(yarn_freqs_0, yarn_freqs_1),
      splash_kernel=splash_kernel,
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
      segment_ids=(segment_ids_0, segment_ids_1),
  )
  _scan_body = functools.partial(
      dsv3_sparse_layer.dsv3_sparse_layer_scan_body,
      yarn_freqs=(yarn_freqs_0, yarn_freqs_1),
      splash_kernel=splash_kernel,
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
      segment_ids=(segment_ids_0, segment_ids_1),
  )
  _epilogue = functools.partial(
      dsv3_sparse_layer.dsv3_sparse_layer_epilogue,
      yarn_freqs=(yarn_freqs_0, yarn_freqs_1),
      splash_kernel=splash_kernel,
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
      segment_ids=(segment_ids_0, segment_ids_1),
  )
  (out_mb0, out_mb1), aux = ops.w_collect_pipelined_scan_prologue_epilogue(
      _prologue, _scan_body, _epilogue, (mb0, mb1), w, _collect_w
  )
  out = ops.merge_microbatches(out_mb0, out_mb1, mesh=mesh)
  return out, aux
