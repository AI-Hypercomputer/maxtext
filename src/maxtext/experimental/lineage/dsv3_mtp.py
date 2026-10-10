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

"""High level DeepSeekV3 Multi-Token Prediction (MTP) layer module."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_dense_layers
from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_sparse_layer
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization


def collect_ehproj_weights(
    w: dsv3_types.DSv3EHProjWeightsPytree,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3EHProjWeightsPytree:
  """Collects EHProj weights along mesh axes."""
  return dsv3_types.DSv3EHProjWeightsPytree(
      enorm_scale=ops.collect_along_axis(w.enorm_scale, ("dcn", "fsdp_attention", "attention"), axis_mapping),
      hnorm_scale=ops.collect_along_axis(w.hnorm_scale, ("dcn", "fsdp_attention", "attention"), axis_mapping),
      eh_proj=ops.collect_along_axis(w.eh_proj, ("dcn", "fsdp_attention", "attention"), axis_mapping),
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_ehproj(
    h_prev: jt.Num[jax.Array, "B T D"],
    emb: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3EHProjWeightsPytree,
    *,
    norm_fn: dsv3_dense_layers.NormFn,
) -> jt.Num[jax.Array, "B T D"]:
  """Executes the DSv3 EHProj (embedding and hidden state normalization and projection).

  Args:
    h_prev: Output activations from the previous layer / base model.
    emb: Token embeddings from the embedding layer.
    w: DSv3 EHProj layer weights (enorm_scale, hnorm_scale, eh_proj).
    norm_fn: Normalization function.

  Returns:
    Projected activations (h_prime) of shape (B, T, D).
  """
  with jax.named_scope("enorm"):
    enorm = norm_fn(emb, w.enorm_scale)
  with jax.named_scope("hnorm"):
    hnorm = norm_fn(h_prev, w.hnorm_scale)
  eh = jnp.concatenate([enorm, hnorm], axis=-1)
  assert w.eh_proj is not None
  with jax.named_scope("eh_proj"):
    h_prime = jnp.tensordot(eh, w.eh_proj, axes=1)
  return h_prime


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mtp_layer(
    h_prev: jt.Num[jax.Array, "B T D"],
    emb: jt.Num[jax.Array, "B T D"],
    w: dsv3_types.DSv3MTPWeightsPytree,
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
    gmm_fn: Any,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    expert_axis_name: str = "expert",
    capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    ragged_buffer_factor: float,
    quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "E"] | None = None,
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3RouterAux[jax.Array]]:
  """Executes a single DSv3 Multi-Token Prediction (MTP) layer.

  Args:
    h_prev: Output activations from the previous layer / base model.
    emb: Token embeddings from the embedding layer.
    w: DSv3 MTP layer weights.
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
    capacity_factor: Capacity factor determining the MoE chunk size.
    router_dtype: Optional dtype for MoE expert selection; see
      `dsv3_router.expert_selection`.
    segment_ids: Optional segment IDs for sequence packing.
    ragged_buffer_factor: Capacity factor for cross-layer ragged activation
      bank.
    quant: Optional quantization config.
    expert_permutations: Optional expert shuffle for this layer, an integer
      permutation of `range(num_experts)` placing logical expert `i` at physical
      position `expert_permutations[i]` (see `dsv3.dsv3`). Applied on the fly to
      the routed expert weights; `w`, its gradient and `aux` stay in logical
      expert order.

  Returns:
    A tuple of (output_tokens, aux).
  """
  if expert_permutations is not None:
    dsv3_expert_shuffle.validate_expert_permutations(expert_permutations, num_experts=num_experts)

  def _collect_routed(
      routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
      axis_mapping: Mapping[str, str | tuple[str, ...]],
  ) -> dsv3_types.DSv3MoERoutedExpertWeightsPytree:
    """Collects routed expert weights, shuffling experts before the FSDP AG."""
    if expert_permutations is None:
      return ops.collect_along_axis(routed, ("dcn", "fsdp_moe"), axis_mapping)
    routed = ops.collect_along_axis(routed, ("dcn",), axis_mapping)
    with jax.named_scope("shuffle_experts"):
      routed = dsv3_expert_shuffle.shuffle_experts(
          routed,
          expert_permutations,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      )
    return ops.collect_along_axis(routed, ("fsdp_moe",), axis_mapping)

  def _collect_w(
      w: dsv3_types.DSv3MTPWeightsPytree,
      axis_mapping: Mapping[str, str | tuple[str, ...]],
  ) -> dsv3_types.DSv3MTPWeightsPytree:
    """Collects MTP layer weights along mesh axes."""
    return dsv3_types.DSv3MTPWeightsPytree(
        ehproj=collect_ehproj_weights(w.ehproj, axis_mapping),
        sparse=dsv3_types.DSv3SparseLayerWeightsPytree(
            pre_attn_norm_scale=ops.collect_along_axis(
                w.sparse.pre_attn_norm_scale,
                ("dcn", "fsdp_attention", "attention"),
                axis_mapping,
            ),
            mla=dsv3_types.DSv3MLAWeightsPytree(
                q_down=ops.collect_along_axis(
                    w.sparse.mla.q_down,
                    ("dcn", "fsdp_attention", "attention"),
                    axis_mapping,
                ),
                q_up=ops.collect_along_axis(w.sparse.mla.q_up, ("dcn", "fsdp_attention"), axis_mapping),
                q_norm_scale=ops.collect_along_axis(
                    w.sparse.mla.q_norm_scale,
                    ("dcn", "fsdp_attention"),
                    axis_mapping,
                ),
                kv_down=ops.collect_along_axis(
                    w.sparse.mla.kv_down,
                    ("dcn", "fsdp_attention", "attention"),
                    axis_mapping,
                ),
                k_up=ops.collect_along_axis(w.sparse.mla.k_up, ("dcn", "fsdp_attention"), axis_mapping),
                v_up=ops.collect_along_axis(w.sparse.mla.v_up, ("dcn", "fsdp_attention"), axis_mapping),
                kv_norm_scale=ops.collect_along_axis(
                    w.sparse.mla.kv_norm_scale,
                    ("dcn", "fsdp_attention"),
                    axis_mapping,
                ),
                out=ops.collect_along_axis(w.sparse.mla.out, ("dcn", "fsdp_attention"), axis_mapping),
            ),
            post_attn_norm_scale=ops.collect_along_axis(
                w.sparse.post_attn_norm_scale,
                ("dcn", "fsdp_attention", "attention"),
                axis_mapping,
            ),
            moe=dsv3_types.DSv3MoEWeightsPytree(
                router=dsv3_types.DSv3MoERouterWeightsPytree(
                    kernel=ops.collect_along_axis(
                        w.sparse.moe.router.kernel,
                        ("dcn", "expert", "fsdp_moe"),
                        axis_mapping,
                    ),
                    bias=w.sparse.moe.router.bias,
                ),
                # Don't collect along the expert axis for routed weights.
                routed=_collect_routed(w.sparse.moe.routed, axis_mapping),
                shared=ops.collect_along_axis(
                    w.sparse.moe.shared,
                    ("dcn", "expert", "fsdp_moe"),
                    axis_mapping,
                ),
            ),
        ),
    )

  w = _collect_w(
      w,
      axis_mapping=axis_mapping,
  )
  h_prime = dsv3_ehproj(
      h_prev=h_prev,
      emb=emb,
      w=w.ehproj,
      norm_fn=norm_fn,
  )
  with jax.named_scope("sparse_layer"):
    assert w.sparse is not None
    return dsv3_sparse_layer.dsv3_sparse_layer(
        h_prime,
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
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        capacity_factor=capacity_factor,
        router_dtype=router_dtype,
        segment_ids=segment_ids,
        ragged_buffer_factor=ragged_buffer_factor,
        quant=quant,
        expert_permutations=expert_permutations,
    )
