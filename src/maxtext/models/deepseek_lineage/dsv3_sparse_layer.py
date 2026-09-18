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

"""Single DeepSeekV3 sparse layer."""

from collections.abc import Mapping
import functools
import math
from typing import Any, Protocol

import jax
import jax.experimental.compute_on

try:
  from jax.experimental.overlap import program_order
except ImportError:
  try:
    from jax._src.pjit import program_order
  except ImportError:

    def program_order(*unused_args, **unused_kwargs):
      return lambda fn: fn


import jax.numpy as jnp
import jaxtyping as jt
import typeguard

from maxtext.models.deepseek_lineage import dsv3_experts
from maxtext.models.deepseek_lineage import dsv3_mla
from maxtext.models.deepseek_lineage import dsv3_router
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops

compute_on = jax.experimental.compute_on.compute_on


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


def _get_local_dim_size(
    dim_size: int,
    dim_pspec: Any,
    mesh: jax.sharding.Mesh,
) -> int:
  if dim_pspec is None:
    return dim_size
  elif isinstance(dim_pspec, str):
    return dim_size // mesh.shape[dim_pspec]
  elif isinstance(dim_pspec, tuple):
    return dim_size // math.prod(mesh.shape[name] for name in dim_pspec)
  return dim_size


def _get_mesh_axis_size(
    axis_name: str | tuple[str, ...],
    mesh: jax.sharding.Mesh,
) -> int:
  if isinstance(axis_name, str):
    return mesh.shape[axis_name]
  elif isinstance(axis_name, tuple):
    return math.prod(mesh.shape[name] for name in axis_name)
  return 1


def _offload_to_host(x: jax.Array | None) -> jax.Array | None:
  if x is None:
    return None
  return jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("pinned_host"))


def _load_to_device(x: jax.Array | None) -> jax.Array | None:
  if x is None:
    return None
  return jax.device_put(x, jax.typeof(x).sharding.with_memory_kind("device"))


def _merge_router_aux(
    aux0: dsv3_types.DSv3RouterAux[Any],
    aux1: dsv3_types.DSv3RouterAux[Any],
    *,
    mesh: jax.sharding.Mesh,
) -> dsv3_types.DSv3RouterAux[Any]:
  """Combines router auxiliary data from two microbatches."""
  group_sizes = None
  if aux0.group_sizes is not None and aux1.group_sizes is not None:
    group_sizes = aux0.group_sizes + aux1.group_sizes
  elif aux0.group_sizes is not None:
    group_sizes = aux0.group_sizes
  elif aux1.group_sizes is not None:
    group_sizes = aux1.group_sizes

  return dsv3_types.DSv3RouterAux(
      group_sizes=group_sizes,
      selected_experts=ops.merge_microbatches(aux0.selected_experts, aux1.selected_experts, mesh=mesh),
      logits=ops.merge_microbatches(aux0.logits, aux1.logits, mesh=mesh),
  )


@jax.named_call
def ubatch_wait(*args: Any) -> Any:
  """Passes inputs to outputs without doing anything."""
  if len(args) == 1:
    return args[0]
  return args


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_mla(
    x: jt.Num[jax.Array, "B T D"],
    pre_attn_norm_scale: jt.Num[jax.Array, "..."] | None,
    w_mla: dsv3_types.DSv3MLAWeightsPytree,
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    *,
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
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T Cq"],
        jt.Num[jax.Array, "B T Ckv_plus_R"],
        jt.Num[jax.Array, "B N T"],
        jt.Num[jax.Array, "B T N V"],
    ],
]:
  """Pre-attention norm and MLA."""
  orig_x = x
  with jax.named_scope("pre_attn_norm"):
    norm_x = norm_fn(x, pre_attn_norm_scale)
  out, mla_residuals = dsv3_mla.dsv3_mla_fwd(
      norm_x,
      yarn_freqs,
      splash_kernel,
      w_mla,
      segment_ids=segment_ids,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  res = (
      orig_x,
      *mla_residuals,
  )
  return (out, orig_x), res


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_norm_and_metadata(
    out_proj_out: jt.Num[jax.Array, "B T D"],
    orig_x: jt.Num[jax.Array, "B T D"],
    post_attn_norm_scale: jt.Num[jax.Array, "..."] | None,
    w_router: dsv3_types.DSv3MoERouterWeightsPytree,
    *,
    norm_fn: NormFn,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    topk_in_group: int,
    expert_axis_name: str = "expert",
    max_capacity: int,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[
    tuple[
        tuple[
            dsv3_router.RouterMetadata[jax.Array],
            dsv3_router.ChunkMetadata[jax.Array],
            jt.Num[jax.Array, "BT k"],
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        dsv3_types.DSv3RouterAux[jax.Array],
    ],
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
]:
  """Post-attention norm, routing metadata, chunk 0 metadata, and auxiliary loss."""
  mla_out = out_proj_out + orig_x
  with jax.named_scope("post_attn_norm"):
    moe_in = norm_fn(mla_out, post_attn_norm_scale)
  router_metadata, coeffs, aux = dsv3_router.dsv3_routing_metadata(
      moe_in,
      w_router,
      num_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      routed_scaling_factor=routed_scaling_factor,
      n_routing_groups=n_routing_groups,
      topk_routing_group=topk_routing_group,
      topk_in_group=topk_in_group,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  chunk0_metadata = dsv3_router.dsv3_chunk_metadata(
      router_metadata.send_sizes,
      jnp.int32(0),
      max_capacity,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  return ((router_metadata, chunk0_metadata, coeffs, moe_in, mla_out), aux), (
      out_proj_out,
      orig_x,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_permute(
    moe_in: jt.Num[jax.Array, "B T D"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    chunk0_metadata: dsv3_router.ChunkMetadata[jax.Array],
    coeffs: jt.Num[jax.Array, "BT k"],
    mla_out: jt.Num[jax.Array, "B T D"],
    *,
    num_experts_per_tok: int = 8,
    mesh: jax.sharding.Mesh,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "BT_k D0 D1"],
        dsv3_router.RouterMetadata[jax.Array],
        dsv3_router.ChunkMetadata[jax.Array],
        jt.Num[jax.Array, "BT k"],
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    tuple[
        dsv3_router.RouterMetadata[jax.Array],
        jt.Num[jax.Array, "BT k"],
        dsv3_router.ChunkMetadata[jax.Array],
    ],
]:
  """Token permute."""
  permuted_x = dsv3_router.dsv3_permute(
      moe_in,
      router_metadata,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
  )
  permute_out = (
      permuted_x,
      router_metadata,
      chunk0_metadata,
      coeffs,
      moe_in,
      mla_out,
  )
  return permute_out, (router_metadata, coeffs, chunk0_metadata)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_dispatch(
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    chunk0_metadata: dsv3_router.ChunkMetadata[jax.Array],
    coeffs: jt.Num[jax.Array, "BT k"],
    moe_in: jt.Num[jax.Array, "B T D"],
    mla_out: jt.Num[jax.Array, "B T D"],
    *,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "kNBT D0 D1"],
        dsv3_router.RouterMetadata[jax.Array],
        dsv3_router.ChunkMetadata[jax.Array],
        jt.Num[jax.Array, "BT k"],
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "BT_k D0 D1"],
    ],
    tuple[()],
]:
  """Token dispatch across devices."""
  chunk0_routed = dsv3_router.dsv3_dispatch_chunk(
      permuted_x,
      chunk0_metadata.chunk_send_sizes,
      router_metadata.send_sizes,
      max_capacity=max_capacity,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  dispatch_out = (
      chunk0_routed,
      router_metadata,
      chunk0_metadata,
      coeffs,
      moe_in,
      mla_out,
      permuted_x,
  )
  return dispatch_out, ()


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_expert(
    chunk0_routed: jt.Num[jax.Array, "kNBT D0 D1"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    chunk0_metadata: dsv3_router.ChunkMetadata[jax.Array],
    coeffs: jt.Num[jax.Array, "BT k"],
    moe_in: jt.Num[jax.Array, "B T D"],
    mla_out: jt.Num[jax.Array, "B T D"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    w_shared: dsv3_types.DSv3MoESharedExpertWeightsPytree,
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    *,
    gmm_fn: GmmFn,
    mesh: jax.sharding.Mesh,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "kNBT D0 D1"],
        jt.Num[jax.Array, "B T D"],
        dsv3_router.RouterMetadata[jax.Array],
        dsv3_router.ChunkMetadata[jax.Array],
        jt.Num[jax.Array, "BT k"],
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "BT_k D0 D1"],
    ],
    tuple[jt.Num[jax.Array, "kNBT ..."],],
]:
  """Routed and shared expert computation for chunk 0."""
  shared_x = dsv3_experts.dsv3_shared_expert(moe_in, w_shared)
  chunk0_routed_out, chunk0_gate_out = dsv3_experts.dsv3_routed_experts_chunk0(
      chunk0_routed,
      w_routed,
      chunk0_metadata.group_sizes,
      gmm_fn=gmm_fn,
      mesh=mesh,
  )
  expert_out = (
      chunk0_routed_out,
      shared_x,
      router_metadata,
      chunk0_metadata,
      coeffs,
      mla_out,
      permuted_x,
  )
  return expert_out, (chunk0_gate_out,)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_iterative_combine_impl(
    initial_combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    initial_next_expert: jt.Num[jax.Array, ""],
    *,
    gmm_fn: GmmFn,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "BT_k D0 D1"]:
  """Forward pass while_loop for chunks k >= 1 in shmap."""
  local_num_experts = full_send_sizes_3d.shape[-1]

  def cond_fn(state):
    curr_expert, _ = state
    return curr_expert < local_num_experts

  def body_fn(state):
    curr_expert, combined_buf = state
    chunk_meta = dsv3_router.dsv3_chunk_metadata_impl(
        full_send_sizes_3d,
        curr_expert,
        max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )
    chunk_routed = dsv3_router.dsv3_dispatch_chunk_impl(
        permuted_x,
        chunk_meta.chunk_send_sizes,
        full_send_sizes_3d,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )
    chunk_routed = dsv3_experts.dsv3_routed_experts_impl(
        chunk_routed,
        w_routed,
        chunk_meta.group_sizes,
        gmm_fn=gmm_fn,
    )
    updated_combined = dsv3_router.dsv3_combine_chunk_impl(
        chunk_routed,
        combined_buf,
        chunk_meta.chunk_send_sizes,
        full_send_sizes_3d,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )
    return (chunk_meta.next_expert, updated_combined)

  _, total_combined_x = jax.lax.while_loop(cond_fn, body_fn, (initial_next_expert, initial_combined_x))
  return total_combined_x


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_iterative_combine(
    initial_combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    initial_next_expert: jt.Num[jax.Array, ""],
    *,
    gmm_fn: GmmFn,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> jt.Num[jax.Array, "BT_k D0 D1"]:
  """Forward pass running while_loop for chunks k >= 1."""
  full_send_sizes_3d = router_metadata.send_sizes
  token_axis = dsv3_router.dsv3_get_token_axis(
      jax.typeof(permuted_x).sharding.spec, num_model_dims=2
  ) or dsv3_router.dsv3_get_token_axis(jax.typeof(initial_combined_x).sharding.spec, num_model_dims=2)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)

  return jax.shard_map(
      functools.partial(
          dsv3_iterative_combine_impl,
          gmm_fn=gmm_fn,
          max_capacity=max_capacity,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=token_spec,
  )(
      initial_combined_x,
      permuted_x,
      w_routed,
      full_send_sizes_3d,
      initial_next_expert,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_combine(
    chunk0_routed: jt.Num[jax.Array, "kNBT D0 D1"],
    shared_x: jt.Num[jax.Array, "B T D"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    chunk0_metadata: dsv3_router.ChunkMetadata[jax.Array],
    coeffs: jt.Num[jax.Array, "BT k"],
    mla_out: jt.Num[jax.Array, "B T D"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    *,
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    gmm_fn: GmmFn,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[
    tuple[
        jt.Num[jax.Array, "BT_k D0 D1"],
        jt.Num[jax.Array, "B T D"],
        dsv3_router.RouterMetadata[jax.Array],
        jt.Num[jax.Array, "BT k"],
        jt.Num[jax.Array, "B T D"],
    ],
    tuple[()],
]:
  """Combines chunk 0 and runs iterative combine for remaining chunks."""
  combined_x_0 = dsv3_router.dsv3_combine_chunk(
      chunk0_routed,
      jnp.empty_like(permuted_x),
      chunk0_metadata.chunk_send_sizes,
      router_metadata.send_sizes,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  total_combined_x = dsv3_iterative_combine(
      combined_x_0,
      permuted_x,
      w_routed,
      router_metadata,
      chunk0_metadata.next_expert,
      gmm_fn=gmm_fn,
      max_capacity=max_capacity,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  combine_out = (
      total_combined_x,
      shared_x,
      router_metadata,
      coeffs,
      mla_out,
  )
  return combine_out, ()


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_unpermute(
    combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    shared_x: jt.Num[jax.Array, "B T D"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    coeffs: jt.Num[jax.Array, "BT k"],
    mla_out: jt.Num[jax.Array, "B T D"],
    *,
    num_experts_per_tok: int = 8,
    local_seq_length: int,
    mesh: jax.sharding.Mesh,
    out_specs: jax.sharding.PartitionSpec,
) -> tuple[
    tuple[jt.Num[jax.Array, "B T D"]],
    tuple[jt.Num[jax.Array, "BT_k D0 D1"]],
]:
  """Global unpermute and residual additions."""
  unpermuted_x = dsv3_router.dsv3_unpermute(
      combined_x,
      coeffs,
      router_metadata,
      num_experts_per_tok=num_experts_per_tok,
      local_seq_length=local_seq_length,
      mesh=mesh,
      out_specs=out_specs,
  )
  moe_out = unpermuted_x + shared_x
  out = mla_out + moe_out
  return (out,), (combined_x,)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def ubatch_unpermute_bwd(
    grad_out: jt.Num[jax.Array, "B T D"],
    combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    coeffs: jt.Num[jax.Array, "BT k"],
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    *,
    num_experts_per_tok: int = 8,
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT_k D0 D1"],
    jt.Num[jax.Array, "B T D"],
    jt.Num[jax.Array, "BT k"],
    jt.Num[jax.Array, "B T D"],
]:
  """Backward pass for unpermute and residual additions."""
  grad_combined_x, grad_coeffs = dsv3_router.dsv3_unpermute_bwd(
      grad_out,
      combined_x,
      coeffs,
      router_metadata,
      num_experts_per_tok=num_experts_per_tok,
      mesh=mesh,
  )
  grad_shared_x = grad_out
  grad_mla_out = grad_out
  return grad_combined_x, grad_shared_x, grad_coeffs, grad_mla_out


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 17)))
def _dsv3_sparse_layer_prologue_vjp(
    x: tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
    ],
    splash_kernel: Any,
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
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    jt.Num[jax.Array, "B T D"],
]:
  """Custom VJP wrapper for DSv3 sparse layer prologue."""
  return _dsv3_sparse_layer_prologue_fwd(
      x,
      w,
      yarn_freqs,
      splash_kernel,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      axis_mapping,
      segment_ids,
  )[0]


def _dsv3_sparse_layer_prologue_fwd(
    x,
    w,
    yarn_freqs,
    splash_kernel,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    axis_mapping,
    segment_ids=(None, None),
):
  """Forward pass for DSv3 sparse layer prologue."""
  mb0, mb1 = x
  yarn_freqs_0, _ = yarn_freqs
  segment_ids_0, _ = segment_ids

  outputs_mla_0, res_mla = ubatch_mla(
      mb0,
      w.pre_attn_norm_scale,
      w.mla,
      yarn_freqs_0,
      splash_kernel,
      segment_ids=segment_ids_0,
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
  )
  mb1_wait = ubatch_wait(mb1)

  (
      orig_x,
      q_down,
      kv_down,
      context,
      splash_out,
  ) = res_mla

  (
      q_down,
      kv_down,
      context,
  ) = jax.tree.map(
      _offload_to_host,
      (q_down, kv_down, context),
  )
  res = (
      orig_x,
      yarn_freqs,
      splash_kernel,
      q_down,
      kv_down,
      context,
      splash_out,
      w,
      segment_ids_0,
  )
  return (outputs_mla_0, mb1_wait), res


def _dsv3_sparse_layer_prologue_bwd(
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    axis_mapping,
    segment_ids,
    res,
    grad_outputs,
):
  """Backward pass for DSv3 sparse layer prologue."""
  del segment_ids
  grad_outputs_mla_0, grad_mb1 = grad_outputs
  grad_out_proj_out, grad_orig_x = grad_outputs_mla_0
  (
      orig_x,
      yarn_freqs,
      splash_kernel,
      q_down,
      kv_down,
      context,
      splash_out,
      w,
      segment_ids_0,
  ) = res

  (
      q_down,
      kv_down,
      context,
  ) = jax.tree.map(
      _load_to_device,
      (q_down, kv_down, context),
  )

  yarn_freqs_0, yarn_freqs_1 = yarn_freqs

  # MLA Backward using saved residuals and remat of q, k, v
  with jax.named_scope("pre_attn_norm"):
    norm_x = norm_fn(orig_x, w.pre_attn_norm_scale)

  grad_norm_x, grad_w_mla, grad_yarn_freqs = dsv3_mla.dsv3_mla_bwd(
      grad_out_proj_out,
      norm_x,
      q_down,
      kv_down,
      context,
      splash_out,
      w.mla,
      yarn_freqs_0,
      splash_kernel,
      segment_ids=segment_ids_0,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )

  # Backprop through pre_attn_norm and accumulate x gradient
  def _fwd_pre_norm(x, pre_attn_norm_scale):
    with jax.named_scope("pre_attn_norm"):
      return norm_fn(x, pre_attn_norm_scale)

  _, vjp_pre_norm = jax.vjp(_fwd_pre_norm, orig_x, w.pre_attn_norm_scale)
  grad_x_norm, grad_pre_attn_norm_scale = vjp_pre_norm(grad_norm_x)
  grad_mb0 = grad_x_norm + grad_orig_x

  grad_mb1_out = ubatch_wait(grad_mb1)

  grad_w = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale,
      mla=grad_w_mla,
      post_attn_norm_scale=None,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=dsv3_types.DSv3MoERouterWeightsPytree(),
          routed=dsv3_types.DSv3MoERoutedExpertWeightsPytree(),
          shared=dsv3_types.DSv3MoESharedExpertWeightsPytree(),
      ),
  )

  grad_yarn_freqs_1 = jax.tree.map(jnp.zeros_like, yarn_freqs_1)
  grad_yarn_freqs_out = (grad_yarn_freqs, grad_yarn_freqs_1)

  return (grad_mb0, grad_mb1_out), grad_w, grad_yarn_freqs_out, None


_dsv3_sparse_layer_prologue_vjp.defvjp(
    _dsv3_sparse_layer_prologue_fwd,
    _dsv3_sparse_layer_prologue_bwd,
)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layer_prologue(
    x: tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
    ],
    splash_kernel: Any,
    *,
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
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    jt.Num[jax.Array, "B T D"],
]:
  """Executes the prologue of a single DSv3 sparse (MoE) layer."""
  return _dsv3_sparse_layer_prologue_vjp(
      x,
      w,
      yarn_freqs,
      splash_kernel,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      axis_mapping,
      segment_ids,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_iterative_dispatch_bwd_impl(
    grad_combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    full_send_sizes_3d: jt.Num[jax.Array, "N N E"],
    initial_next_expert: jt.Num[jax.Array, ""],
    initial_grad_permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    initial_grad_w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    *,
    gmm_fn: GmmFn,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[
    jt.Num[jax.Array, "BT_k D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
]:
  """Local implementation of iterative dispatch backward while_loop for chunks k >= 1 in shard_map."""
  local_num_experts = full_send_sizes_3d.shape[-1]

  def cond_fn(state):
    curr_expert, _, _ = state
    return curr_expert < local_num_experts

  def body_fn(state):
    curr_expert, grad_permuted_buf, grad_w_accum = state
    chunk_meta = dsv3_router.dsv3_chunk_metadata_impl(
        full_send_sizes_3d,
        curr_expert,
        max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )
    chunk_routed, chunk_grad_routed = dsv3_router.dsv3_aggregated_dispatches_impl(
        permuted_x,
        grad_combined_x,
        chunk_meta.chunk_send_sizes,
        full_send_sizes_3d,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )

    def _expert_forward(inputs, weights):
      return dsv3_experts.dsv3_routed_experts_impl(inputs, weights, chunk_meta.group_sizes, gmm_fn=gmm_fn)

    _, vjp_fn = jax.vjp(_expert_forward, chunk_routed, w_routed)
    chunk_grad_routed_in, chunk_grad_w = vjp_fn(chunk_grad_routed)

    updated_grad_permuted = dsv3_router.dsv3_combine_chunk_impl(
        chunk_grad_routed_in,
        grad_permuted_buf,
        chunk_meta.chunk_send_sizes,
        full_send_sizes_3d,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
    )

    def _accumulate_grad_w(accum, grad):
      return accum + grad

    updated_grad_w = jax.tree_util.tree_map(_accumulate_grad_w, grad_w_accum, chunk_grad_w)
    return (chunk_meta.next_expert, updated_grad_permuted, updated_grad_w)

  _, final_grad_permuted_x, final_grad_w_routed = jax.lax.while_loop(
      cond_fn,
      body_fn,
      (initial_next_expert, initial_grad_permuted_x, initial_grad_w_routed),
  )
  return final_grad_permuted_x, final_grad_w_routed


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_iterative_dispatch_bwd(
    grad_combined_x: jt.Num[jax.Array, "BT_k D0 D1"],
    permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    router_metadata: dsv3_router.RouterMetadata[jax.Array],
    initial_next_expert: jt.Num[jax.Array, ""],
    initial_grad_permuted_x: jt.Num[jax.Array, "BT_k D0 D1"],
    initial_grad_w_routed: dsv3_types.DSv3MoERoutedExpertWeightsPytree,
    *,
    gmm_fn: GmmFn,
    max_capacity: int,
    expert_axis_name: str,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[
    jt.Num[jax.Array, "BT_k D0 D1"],
    dsv3_types.DSv3MoERoutedExpertWeightsPytree,
]:
  """Backward pass looping over remaining chunks in shard_map."""
  full_send_sizes_3d = router_metadata.send_sizes
  token_axis = dsv3_router.dsv3_get_token_axis(
      jax.typeof(permuted_x).sharding.spec, num_model_dims=2
  ) or dsv3_router.dsv3_get_token_axis(jax.typeof(grad_combined_x).sharding.spec, num_model_dims=2)
  token_spec = jax.sharding.PartitionSpec(token_axis, None, None)
  token_spec = ops.physical_pspec(token_spec, axis_mapping)
  out_specs = (
      token_spec,
      jax.tree_util.tree_map(lambda g: jax.typeof(g).sharding.spec, initial_grad_w_routed),
  )

  return jax.shard_map(
      functools.partial(
          dsv3_iterative_dispatch_bwd_impl,
          gmm_fn=gmm_fn,
          max_capacity=max_capacity,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
      ),
      mesh=mesh,
      out_specs=out_specs,
  )(
      grad_combined_x,
      permuted_x,
      w_routed,
      full_send_sizes_3d,
      initial_next_expert,
      initial_grad_permuted_x,
      initial_grad_w_routed,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(4, 26)))
def _dsv3_sparse_layer_epilogue_vjp(
    outputs_mla: tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
    ],
    splash_kernel: Any,
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
    expert_axis_name: str,
    capacity_factor: float,
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Custom VJP wrapper for DSv3 sparse layer epilogue."""
  return _dsv3_sparse_layer_epilogue_fwd(
      outputs_mla,
      w,
      yarn_freqs,
      splash_kernel,
      num_experts,
      num_experts_per_tok,
      routed_scaling_factor,
      n_routing_groups,
      topk_routing_group,
      topk_in_group,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      gmm_fn,
      axis_mapping,
      expert_axis_name,
      capacity_factor,
      segment_ids,
  )[0]


def _dsv3_sparse_layer_epilogue_fwd(
    outputs_mla,
    w,
    yarn_freqs,
    splash_kernel,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    topk_in_group,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    gmm_fn,
    axis_mapping,
    expert_axis_name,
    capacity_factor,
    segment_ids=(None, None),
):
  """Forward pass for DSv3 sparse layer epilogue."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  num_devices = _get_mesh_axis_size(physical_expert_axis_name, mesh)
  outputs_mla_curr_0, mb1 = outputs_mla
  out_proj_out_curr_0, _ = outputs_mla_curr_0
  x_pspec = jax.typeof(out_proj_out_curr_0).sharding.spec
  local_batch_size = _get_local_dim_size(out_proj_out_curr_0.shape[0], x_pspec[0], mesh)
  local_seq_length = _get_local_dim_size(out_proj_out_curr_0.shape[1], x_pspec[1], mesh)
  max_capacity = int(capacity_factor * (local_batch_size * local_seq_length) * num_devices)
  _, yarn_freqs_1 = yarn_freqs
  _, segment_ids_1 = segment_ids

  @jax.named_call
  @program_order(enforce=False)
  def phase1(outputs_mla_curr_0, mb1):
    out_proj_out_curr_0, orig_x_curr_0 = outputs_mla_curr_0
    (
        (
            router_metadata_curr_0,
            chunk0_metadata_curr_0,
            coeffs_curr_0,
            moe_in_curr_0,
            mla_out_curr_0,
        ),
        aux_curr_0,
    ), res_metadata_curr_0 = ubatch_norm_and_metadata(
        out_proj_out_curr_0,
        orig_x_curr_0,
        w.post_attn_norm_scale,
        w.moe.router,
        norm_fn=norm_fn,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups,
        topk_routing_group=topk_routing_group,
        topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
        max_capacity=max_capacity,
    )
    outputs_permute_curr_0, res_permute_curr_0 = ubatch_permute(
        moe_in_curr_0,
        router_metadata_curr_0,
        chunk0_metadata_curr_0,
        coeffs_curr_0,
        mla_out_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )
    mb1_wait = ubatch_wait(mb1)
    return (outputs_permute_curr_0, mb1_wait, aux_curr_0), (
        *res_metadata_curr_0,
        *res_permute_curr_0,
    )

  @jax.named_call
  @program_order(enforce=False)
  def phase2(outputs_permute_curr_0, mb1):
    outputs_dispatch_curr_0, _ = ubatch_dispatch(
        *outputs_permute_curr_0,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    outputs_mla_curr_1, res_mla_curr_1 = ubatch_mla(
        mb1,
        w.pre_attn_norm_scale,
        w.mla,
        yarn_freqs_1,
        splash_kernel,
        segment_ids=segment_ids_1,
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
    )
    return (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase3(outputs_dispatch_curr_0, outputs_mla_curr_1):
    outputs_dispatch_wait_0 = ubatch_wait(outputs_dispatch_curr_0)
    out_proj_out_curr_1, orig_x_curr_1 = outputs_mla_curr_1
    (
        (
            router_metadata_curr_1,
            chunk0_metadata_curr_1,
            coeffs_curr_1,
            moe_in_curr_1,
            mla_out_curr_1,
        ),
        aux_curr_1,
    ), res_metadata_curr_1 = ubatch_norm_and_metadata(
        out_proj_out_curr_1,
        orig_x_curr_1,
        w.post_attn_norm_scale,
        w.moe.router,
        norm_fn=norm_fn,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups,
        topk_routing_group=topk_routing_group,
        topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
        max_capacity=max_capacity,
    )
    outputs_permute_curr_1, res_permute_curr_1 = ubatch_permute(
        moe_in_curr_1,
        router_metadata_curr_1,
        chunk0_metadata_curr_1,
        coeffs_curr_1,
        mla_out_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )
    return (outputs_dispatch_wait_0, outputs_permute_curr_1, aux_curr_1), (
        *res_metadata_curr_1,
        *res_permute_curr_1,
    )

  @jax.named_call
  @program_order(enforce=False)
  def phase4(outputs_dispatch_curr_0, outputs_permute_curr_1):
    outputs_expert_curr_0, res_expert_curr_0 = ubatch_expert(
        *outputs_dispatch_curr_0,
        w_shared=w.moe.shared,
        w_routed=w.moe.routed,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    outputs_dispatch_curr_1, _ = ubatch_dispatch(
        *outputs_permute_curr_1,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (outputs_expert_curr_0, outputs_dispatch_curr_1), res_expert_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase5(outputs_expert_curr_0, outputs_dispatch_curr_1):
    outputs_combine_curr_0, _ = ubatch_combine(
        *outputs_expert_curr_0,
        w_routed=w.moe.routed,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    outputs_expert_curr_1, res_expert_curr_1 = ubatch_expert(
        *outputs_dispatch_curr_1,
        w_shared=w.moe.shared,
        w_routed=w.moe.routed,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    return (outputs_combine_curr_0, outputs_expert_curr_1), res_expert_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase6(outputs_combine_curr_0, outputs_expert_curr_1):
    outputs_unpermute_curr_0, res_unpermute_curr_0 = ubatch_unpermute(
        *outputs_combine_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        local_seq_length=local_seq_length,
        mesh=mesh,
        out_specs=x_pspec,
    )
    (out_mb0,) = outputs_unpermute_curr_0
    outputs_expert_wait_1 = ubatch_wait(outputs_expert_curr_1)
    return (out_mb0, outputs_expert_wait_1), res_unpermute_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase7(out_mb0, outputs_expert_curr_1):
    out_mb0_wait = ubatch_wait(out_mb0)
    outputs_combine_curr_1, _ = ubatch_combine(
        *outputs_expert_curr_1,
        w_routed=w.moe.routed,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (out_mb0_wait, outputs_combine_curr_1), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(out_mb0_wait, outputs_combine_curr_1):
    out_mb0_final = ubatch_wait(out_mb0_wait)
    outputs_unpermute_curr_1, res_unpermute_curr_1 = ubatch_unpermute(
        *outputs_combine_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        local_seq_length=local_seq_length,
        mesh=mesh,
        out_specs=x_pspec,
    )
    (out_mb1,) = outputs_unpermute_curr_1
    return (out_mb0_final, out_mb1), res_unpermute_curr_1

  @program_order(enforce=True)
  def _epilogue_fwd(outputs_mla_curr_0, mb1):
    (outputs_permute_curr_0, mb1_wait, aux_curr_0), res_phase1_curr_0 = phase1(outputs_mla_curr_0, mb1)
    (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1 = phase2(outputs_permute_curr_0, mb1_wait)
    (outputs_dispatch_wait_0, outputs_permute_curr_1, aux_curr_1), (res_phase3_curr_1) = phase3(
        outputs_dispatch_curr_0, outputs_mla_curr_1
    )
    (outputs_expert_curr_0, outputs_dispatch_curr_1), res_expert_curr_0 = phase4(
        outputs_dispatch_wait_0, outputs_permute_curr_1
    )
    (outputs_combine_curr_0, outputs_expert_curr_1), res_expert_curr_1 = phase5(
        outputs_expert_curr_0, outputs_dispatch_curr_1
    )
    (out_mb0, outputs_expert_wait_1), res_unpermute_curr_0 = phase6(outputs_combine_curr_0, outputs_expert_curr_1)
    (out_mb0_wait, outputs_combine_curr_1), _ = phase7(out_mb0, outputs_expert_wait_1)
    (out_mb0_final, out_mb1), res_unpermute_curr_1 = phase8(out_mb0_wait, outputs_combine_curr_1)

    aux = _merge_router_aux(aux_curr_0, aux_curr_1, mesh=mesh)
    return ((out_mb0_final, out_mb1), aux), (
        res_phase1_curr_0,
        res_expert_curr_0,
        res_unpermute_curr_0,
        res_mla_curr_1,
        res_phase3_curr_1,
        res_expert_curr_1,
        res_unpermute_curr_1,
    )

  ((out_mb0_final, out_mb1), aux), (
      res_phase1_curr_0,
      res_expert_curr_0,
      res_unpermute_curr_0,
      res_mla_curr_1,
      res_phase3_curr_1,
      res_expert_curr_1,
      res_unpermute_curr_1,
  ) = _epilogue_fwd(outputs_mla_curr_0, mb1)

  (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
  ) = res_phase1_curr_0
  (chunk0_gate_out_curr_0,) = res_expert_curr_0
  (total_combined_x_curr_0,) = res_unpermute_curr_0

  (
      _,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
  ) = res_mla_curr_1
  (
      out_proj_out_curr_1,
      orig_x_curr_1,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
  ) = res_phase3_curr_1
  (chunk0_gate_out_curr_1,) = res_expert_curr_1
  (total_combined_x_curr_1,) = res_unpermute_curr_1

  (
      out_proj_out_curr_0,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      out_proj_out_curr_1,
      total_combined_x_curr_1,
      chunk0_gate_out_curr_0,
      chunk0_gate_out_curr_1,
  ) = jax.tree.map(
      _offload_to_host,
      (
          out_proj_out_curr_0,
          q_down_curr_1,
          kv_down_curr_1,
          context_curr_1,
          out_proj_out_curr_1,
          total_combined_x_curr_1,
          chunk0_gate_out_curr_0,
          chunk0_gate_out_curr_1,
      ),
  )

  res = (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
      chunk0_gate_out_curr_0,
      total_combined_x_curr_0,
      orig_x_curr_1,
      yarn_freqs,
      splash_kernel,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
      out_proj_out_curr_1,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
      chunk0_gate_out_curr_1,
      total_combined_x_curr_1,
      w,
      segment_ids_1,
  )
  return ((out_mb0_final, out_mb1), aux), res


def _dsv3_sparse_layer_epilogue_bwd(
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    topk_in_group,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    gmm_fn,
    axis_mapping,
    expert_axis_name,
    capacity_factor,
    segment_ids,
    res,
    grad_outputs,
):
  """Backward pass for DSv3 sparse layer epilogue."""
  del segment_ids
  grad_out, grad_aux = grad_outputs
  grad_out_mb0, grad_out_mb1 = grad_out
  (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
      chunk0_gate_out_curr_0,
      total_combined_x_curr_0,
      orig_x_curr_1,
      yarn_freqs,
      splash_kernel,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
      out_proj_out_curr_1,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
      chunk0_gate_out_curr_1,
      total_combined_x_curr_1,
      w,
      segment_ids_1,
  ) = res

  (
      out_proj_out_curr_0,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      out_proj_out_curr_1,
      total_combined_x_curr_1,
      chunk0_gate_out_curr_0,
      chunk0_gate_out_curr_1,
  ) = jax.tree.map(
      _load_to_device,
      (
          out_proj_out_curr_0,
          q_down_curr_1,
          kv_down_curr_1,
          context_curr_1,
          out_proj_out_curr_1,
          total_combined_x_curr_1,
          chunk0_gate_out_curr_0,
          chunk0_gate_out_curr_1,
      ),
  )

  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  num_devices = _get_mesh_axis_size(physical_expert_axis_name, mesh)
  x_pspec = jax.typeof(out_proj_out_curr_0).sharding.spec
  local_batch_size = _get_local_dim_size(out_proj_out_curr_0.shape[0], x_pspec[0], mesh)
  local_seq_length = _get_local_dim_size(out_proj_out_curr_0.shape[1], x_pspec[1], mesh)
  max_capacity = int(capacity_factor * (local_batch_size * local_seq_length) * num_devices)

  yarn_freqs_0, yarn_freqs_1 = yarn_freqs

  if getattr(grad_aux, "logits", None) is not None:
    grad_logits_0, grad_logits_1 = ops.split_microbatches(grad_aux.logits, mesh=mesh)
    grad_aux_0 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=getattr(grad_aux, "group_sizes", None),
        selected_experts=getattr(grad_aux, "selected_experts", None),
        logits=grad_logits_0,
    )
    grad_aux_1 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=getattr(grad_aux, "group_sizes", None),
        selected_experts=getattr(grad_aux, "selected_experts", None),
        logits=grad_logits_1,
    )
  else:
    grad_aux_0 = grad_aux
    grad_aux_1 = grad_aux

  mla_out_curr_0 = out_proj_out_curr_0 + orig_x_curr_0
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_0 = norm_fn(mla_out_curr_0, w.post_attn_norm_scale)

  mla_out_curr_1 = out_proj_out_curr_1 + orig_x_curr_1
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_1 = norm_fn(mla_out_curr_1, w.post_attn_norm_scale)

  @jax.named_call
  @program_order(enforce=False)
  def phase1(grad_out_mb0, grad_out_mb1):
    """Phase 1: mb1 unpermute pullback, mb0 wait."""
    grad_out_mb0_wait = ubatch_wait(grad_out_mb0)
    (
        grad_combined_x_1,
        grad_shared_x_1,
        grad_coeffs_1,
        grad_mla_out_unpermute_1,
    ) = ubatch_unpermute_bwd(
        grad_out_mb1,
        total_combined_x_curr_1,
        coeffs_curr_1,
        router_metadata_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )

    def _fwd_permute_1(moe_in):
      return dsv3_router.dsv3_permute(
          moe_in,
          router_metadata_curr_1,
          num_experts_per_tok=num_experts_per_tok,
          mesh=mesh,
      )

    permuted_x_1, vjp_permute_1 = jax.vjp(_fwd_permute_1, moe_in_curr_1)
    return (
        grad_out_mb0_wait,
        permuted_x_1,
        grad_combined_x_1,
        grad_shared_x_1,
        vjp_permute_1,
        grad_coeffs_1,
        grad_mla_out_unpermute_1,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase2(grad_out_mb0_wait, permuted_x_1, grad_combined_x_1):
    """Phase 2: mb0 unpermute pullback; mb1 dispatch remat & combine transpose."""
    (
        grad_combined_x_0,
        grad_shared_x_0,
        grad_coeffs_0,
        grad_mla_out_unpermute_0,
    ) = ubatch_unpermute_bwd(
        grad_out_mb0_wait,
        total_combined_x_curr_0,
        coeffs_curr_0,
        router_metadata_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )

    def _fwd_permute_0(moe_in):
      return dsv3_router.dsv3_permute(
          moe_in,
          router_metadata_curr_0,
          num_experts_per_tok=num_experts_per_tok,
          mesh=mesh,
      )

    permuted_x_0, vjp_permute_0 = jax.vjp(_fwd_permute_0, moe_in_curr_0)

    chunk0_routed_1, chunk0_grad_routed_1 = dsv3_router.dsv3_aggregated_dispatches(
        permuted_x_1,
        grad_combined_x_1,
        chunk0_metadata_curr_1.chunk_send_sizes,
        router_metadata_curr_1.send_sizes,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (
        (
            permuted_x_0,
            grad_combined_x_0,
            grad_shared_x_0,
            vjp_permute_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        (chunk0_routed_1, chunk0_grad_routed_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(mb0_state, mb1_state):
    """Phase 3: wait."""
    return ubatch_wait((mb0_state, mb1_state)), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(
      permuted_x_0,
      grad_combined_x_0,
      chunk0_routed_1,
      chunk0_grad_routed_1,
      grad_shared_x_1,
  ):
    """Phase 4: mb0 dispatch remat & combine transpose; mb1 expert backward."""
    chunk0_routed_0, chunk0_grad_routed_0 = dsv3_router.dsv3_aggregated_dispatches(
        permuted_x_0,
        grad_combined_x_0,
        chunk0_metadata_curr_0.chunk_send_sizes,
        router_metadata_curr_0.send_sizes,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    def _shared_expert_fwd_1(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)

    _, vjp_shared_1 = jax.vjp(_shared_expert_fwd_1, moe_in_curr_1, w.moe.shared)
    grad_moe_in_shared_1, grad_w_shared_1 = vjp_shared_1(grad_shared_x_1)

    chunk0_grad_routed_in_1, grad_w_routed_0_1 = dsv3_experts.dsv3_routed_experts_chunk0_bwd(
        chunk0_grad_routed_1,
        chunk0_routed_1,
        chunk0_gate_out_curr_1,
        w.moe.routed,
        chunk0_metadata_curr_1.group_sizes,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    return (
        (chunk0_routed_0, chunk0_grad_routed_0),
        (
            chunk0_grad_routed_in_1,
            grad_w_routed_0_1,
            grad_moe_in_shared_1,
            grad_w_shared_1,
        ),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(
      chunk0_routed_0,
      chunk0_grad_routed_0,
      grad_shared_x_0,
      chunk0_grad_routed_in_1,
      grad_w_routed_0_1,
      permuted_x_1,
      grad_combined_x_1,
  ):
    """Phase 5: mb0 expert backward; mb1 dispatch transpose & iterative dispatch backward."""

    def _shared_expert_fwd_0(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)

    _, vjp_shared_0 = jax.vjp(_shared_expert_fwd_0, moe_in_curr_0, w.moe.shared)
    grad_moe_in_shared_0, grad_w_shared_0 = vjp_shared_0(grad_shared_x_0)

    chunk0_grad_routed_in_0, grad_w_routed_0_0 = dsv3_experts.dsv3_routed_experts_chunk0_bwd(
        chunk0_grad_routed_0,
        chunk0_routed_0,
        chunk0_gate_out_curr_0,
        w.moe.routed,
        chunk0_metadata_curr_0.group_sizes,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )

    grad_permuted_x_0_1 = dsv3_router.dsv3_combine_chunk(
        chunk0_grad_routed_in_1,
        jnp.empty_like(permuted_x_1),
        chunk0_metadata_curr_1.chunk_send_sizes,
        router_metadata_curr_1.send_sizes,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    final_grad_permuted_x_1, final_grad_w_routed_1 = dsv3_iterative_dispatch_bwd(
        grad_combined_x_1,
        permuted_x_1,
        w.moe.routed,
        router_metadata_curr_1,
        chunk0_metadata_curr_1.next_expert,
        grad_permuted_x_0_1,
        grad_w_routed_0_1,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (
        (
            chunk0_grad_routed_in_0,
            grad_w_routed_0_0,
            grad_moe_in_shared_0,
            grad_w_shared_0,
        ),
        (final_grad_permuted_x_1, final_grad_w_routed_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(
      chunk0_grad_routed_in_0,
      grad_w_routed_0_0,
      permuted_x_0,
      grad_combined_x_0,
      final_grad_permuted_x_1,
      vjp_permute_1,
      grad_moe_in_shared_1,
      grad_coeffs_1,
      grad_aux_1,
  ):
    """Phase 6: mb0 dispatch transpose & iterative dispatch backward; mb1 router & post_norm pullback."""
    grad_permuted_x_0_0 = dsv3_router.dsv3_combine_chunk(
        chunk0_grad_routed_in_0,
        jnp.empty_like(permuted_x_0),
        chunk0_metadata_curr_0.chunk_send_sizes,
        router_metadata_curr_0.send_sizes,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    final_grad_permuted_x_0, final_grad_w_routed_0 = dsv3_iterative_dispatch_bwd(
        grad_combined_x_0,
        permuted_x_0,
        w.moe.routed,
        router_metadata_curr_0,
        chunk0_metadata_curr_0.next_expert,
        grad_permuted_x_0_0,
        grad_w_routed_0_0,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    (grad_moe_in_permute_1,) = vjp_permute_1(final_grad_permuted_x_1)
    grad_moe_in_expert_1 = grad_moe_in_permute_1 + grad_moe_in_shared_1

    def _fwd_router_1(moe_in, w_router):
      _, coeffs, aux = dsv3_router.dsv3_routing_metadata(
          moe_in,
          w_router,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          mesh=mesh,
      )
      return coeffs, aux.logits

    _, vjp_router_1 = jax.vjp(_fwd_router_1, moe_in_curr_1, w.moe.router)
    grad_moe_in_router_1, grad_w_router_1 = vjp_router_1((grad_coeffs_1, grad_aux_1.logits))
    grad_moe_in_total_1 = grad_moe_in_expert_1 + grad_moe_in_router_1

    def _fwd_post_norm_1(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)

    _, vjp_post_norm_1 = jax.vjp(_fwd_post_norm_1, mla_out_curr_1, w.post_attn_norm_scale)
    grad_mla_out_norm_1, grad_post_attn_norm_scale_1 = vjp_post_norm_1(grad_moe_in_total_1)
    return (
        (final_grad_permuted_x_0, final_grad_w_routed_0),
        (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(mb0_state, grad_mla_out_total_1):
    """Phase 7: mb0 wait; mb1 MLA backward."""
    mb0_wait = ubatch_wait(mb0_state)
    with jax.named_scope("pre_attn_norm"):
      norm_x_1 = norm_fn(orig_x_curr_1, w.pre_attn_norm_scale)

    grad_norm_x_1, grad_w_mla_1, grad_yarn_freqs_1 = dsv3_mla.dsv3_mla_bwd(
        grad_mla_out_total_1,
        norm_x_1,
        q_down_curr_1,
        kv_down_curr_1,
        context_curr_1,
        splash_out_curr_1,
        w.mla,
        yarn_freqs_1,
        splash_kernel,
        segment_ids=segment_ids_1,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        rope_head_dim=rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        mscale=mscale,
        norm_fn=norm_fn,
        rope_fn=rope_fn,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

    def _fwd_pre_norm_1(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)

    _, vjp_pre_norm_1 = jax.vjp(_fwd_pre_norm_1, orig_x_curr_1, w.pre_attn_norm_scale)
    grad_x_norm_1, grad_pre_attn_norm_scale_1 = vjp_pre_norm_1(grad_norm_x_1)
    grad_mb1 = grad_x_norm_1 + grad_mla_out_total_1

    return (
        mb0_wait,
        (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(
      final_grad_permuted_x_0,
      vjp_permute_0,
      grad_moe_in_shared_0,
      grad_coeffs_0,
      grad_aux_0,
      grad_mb1,
  ):
    """Phase 8: mb0 router & post_norm pullback; mb1 wait."""
    (grad_moe_in_permute_0,) = vjp_permute_0(final_grad_permuted_x_0)
    grad_moe_in_expert_0 = grad_moe_in_permute_0 + grad_moe_in_shared_0

    def _fwd_router_0(moe_in, w_router):
      _, coeffs, aux = dsv3_router.dsv3_routing_metadata(
          moe_in,
          w_router,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          mesh=mesh,
      )
      return coeffs, aux.logits

    _, vjp_router_0 = jax.vjp(_fwd_router_0, moe_in_curr_0, w.moe.router)
    grad_moe_in_router_0, grad_w_router_0 = vjp_router_0((grad_coeffs_0, grad_aux_0.logits))
    grad_moe_in_total_0 = grad_moe_in_expert_0 + grad_moe_in_router_0

    def _fwd_post_norm_0(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)

    _, vjp_post_norm_0 = jax.vjp(_fwd_post_norm_0, mla_out_curr_0, w.post_attn_norm_scale)
    grad_mla_out_norm_0, grad_post_attn_norm_scale_0 = vjp_post_norm_0(grad_moe_in_total_0)

    grad_mb1_wait = ubatch_wait(grad_mb1)

    return (
        (grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0),
        grad_mb1_wait,
    ), ()

  @program_order(enforce=True)
  def _epilogue_bwd(grad_out_mb0, grad_out_mb1):
    (
        (
            grad_out_mb0_wait,
            permuted_x_1,
            grad_combined_x_1,
            grad_shared_x_1,
            vjp_permute_1,
            grad_coeffs_1,
            grad_mla_out_unpermute_1,
        ),
        _,
    ) = phase1(grad_out_mb0, grad_out_mb1)

    (
        (
            permuted_x_0,
            grad_combined_x_0,
            grad_shared_x_0,
            vjp_permute_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        (chunk0_routed_1, chunk0_grad_routed_1),
    ), _ = phase2(grad_out_mb0_wait, permuted_x_1, grad_combined_x_1)

    (
        (
            permuted_x_0_wait,
            grad_combined_x_0_wait,
            grad_shared_x_0_wait,
            vjp_permute_0_wait,
            grad_coeffs_0_wait,
            grad_mla_out_unpermute_0_wait,
        ),
        (chunk0_routed_1_wait, chunk0_grad_routed_1_wait),
    ), _ = phase3(
        (
            permuted_x_0,
            grad_combined_x_0,
            grad_shared_x_0,
            vjp_permute_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        (chunk0_routed_1, chunk0_grad_routed_1),
    )

    (
        (chunk0_routed_0, chunk0_grad_routed_0),
        (
            chunk0_grad_routed_in_1,
            grad_w_routed_0_1,
            grad_moe_in_shared_1,
            grad_w_shared_1,
        ),
    ), _ = phase4(
        permuted_x_0_wait,
        grad_combined_x_0_wait,
        chunk0_routed_1_wait,
        chunk0_grad_routed_1_wait,
        grad_shared_x_1,
    )

    (
        (
            chunk0_grad_routed_in_0,
            grad_w_routed_0_0,
            grad_moe_in_shared_0,
            grad_w_shared_0,
        ),
        (final_grad_permuted_x_1, final_grad_w_routed_1),
    ), _ = phase5(
        chunk0_routed_0,
        chunk0_grad_routed_0,
        grad_shared_x_0_wait,
        chunk0_grad_routed_in_1,
        grad_w_routed_0_1,
        permuted_x_1,
        grad_combined_x_1,
    )

    (
        (final_grad_permuted_x_0, final_grad_w_routed_0),
        (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1),
    ), _ = phase6(
        chunk0_grad_routed_in_0,
        grad_w_routed_0_0,
        permuted_x_0_wait,
        grad_combined_x_0_wait,
        final_grad_permuted_x_1,
        vjp_permute_1,
        grad_moe_in_shared_1,
        grad_coeffs_1,
        grad_aux_1,
    )

    grad_mla_out_total_1 = grad_mla_out_unpermute_1 + grad_mla_out_norm_1
    (
        mb0_wait_7,
        (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1),
    ), _ = phase7(
        (
            final_grad_permuted_x_0,
            final_grad_w_routed_0,
            vjp_permute_0_wait,
            grad_moe_in_shared_0,
            grad_coeffs_0_wait,
            grad_mla_out_unpermute_0_wait,
        ),
        grad_mla_out_total_1,
    )

    (
        final_grad_permuted_x_0_w,
        final_grad_w_routed_0_w,
        vjp_permute_0_w,
        grad_moe_in_shared_0_w,
        grad_coeffs_0_w,
        grad_mla_out_unpermute_0_w,
    ) = mb0_wait_7

    (
        (grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0),
        grad_mb1_wait,
    ), _ = phase8(
        final_grad_permuted_x_0_w,
        vjp_permute_0_w,
        grad_moe_in_shared_0_w,
        grad_coeffs_0_w,
        grad_aux_0,
        grad_mb1,
    )

    grad_mla_out_total_0 = grad_mla_out_unpermute_0_w + grad_mla_out_norm_0

    return (
        grad_mla_out_total_0,
        grad_mb1_wait,
        grad_pre_attn_norm_scale_1,
        grad_w_mla_1,
        grad_yarn_freqs_1,
        grad_post_attn_norm_scale_0,
        grad_post_attn_norm_scale_1,
        grad_w_router_0,
        grad_w_router_1,
        final_grad_w_routed_0_w,
        final_grad_w_routed_1,
        grad_w_shared_0,
        grad_w_shared_1,
    )

  (
      grad_mla_out_total_0,
      grad_mb1_wait,
      grad_pre_attn_norm_scale_1,
      grad_w_mla_1,
      grad_yarn_freqs_1,
      grad_post_attn_norm_scale_0,
      grad_post_attn_norm_scale_1,
      grad_w_router_0,
      grad_w_router_1,
      final_grad_w_routed_0,
      final_grad_w_routed_1,
      grad_w_shared_0,
      grad_w_shared_1,
  ) = _epilogue_bwd(grad_out_mb0, grad_out_mb1)

  grad_carry = (
      (grad_mla_out_total_0, grad_mla_out_total_0),
      grad_mb1_wait,
  )

  grad_w = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale_1,
      mla=grad_w_mla_1,
      post_attn_norm_scale=grad_post_attn_norm_scale_0 + grad_post_attn_norm_scale_1,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=jax.tree.map(jnp.add, grad_w_router_0, grad_w_router_1),
          routed=jax.tree.map(jnp.add, final_grad_w_routed_0, final_grad_w_routed_1),
          shared=jax.tree.map(jnp.add, grad_w_shared_0, grad_w_shared_1),
      ),
  )

  grad_yarn_freqs_0 = jax.tree.map(jnp.zeros_like, yarn_freqs_0)
  grad_yarn_freqs = (grad_yarn_freqs_0, grad_yarn_freqs_1)

  return grad_carry, grad_w, grad_yarn_freqs, None


_dsv3_sparse_layer_epilogue_vjp.defvjp(
    _dsv3_sparse_layer_epilogue_fwd,
    _dsv3_sparse_layer_epilogue_bwd,
)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layer_epilogue(
    outputs_mla: tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
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
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        jt.Num[jax.Array, "B T D"],
        jt.Num[jax.Array, "B T D"],
    ],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Executes the epilogue of a single DSv3 sparse (MoE) layer."""
  return _dsv3_sparse_layer_epilogue_vjp(
      outputs_mla,
      w,
      yarn_freqs,
      splash_kernel,
      num_experts,
      num_experts_per_tok,
      routed_scaling_factor,
      n_routing_groups,
      topk_routing_group,
      topk_in_group,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      gmm_fn,
      axis_mapping,
      expert_axis_name,
      capacity_factor,
      segment_ids,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(5, 27)))
def _dsv3_sparse_layer_scan_body_vjp(
    outputs_mla_curr: tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    w_curr: dsv3_types.DSv3SparseLayerWeightsPytree,
    w_next: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
    ],
    splash_kernel: Any,
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
    expert_axis_name: str,
    capacity_factor: float,
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Custom VJP wrapper for DSv3 sparse layer scan body."""
  return _dsv3_sparse_layer_scan_body_fwd(
      outputs_mla_curr,
      w_curr,
      w_next,
      yarn_freqs,
      splash_kernel,
      num_experts,
      num_experts_per_tok,
      routed_scaling_factor,
      n_routing_groups,
      topk_routing_group,
      topk_in_group,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      gmm_fn,
      axis_mapping,
      expert_axis_name,
      capacity_factor,
      segment_ids,
  )[0]


def _dsv3_sparse_layer_scan_body_fwd(
    outputs_mla_curr,
    w_curr,
    w_next,
    yarn_freqs,
    splash_kernel,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    topk_in_group,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    gmm_fn,
    axis_mapping,
    expert_axis_name,
    capacity_factor,
    segment_ids=(None, None),
):
  """Forward pass for DSv3 sparse layer scan body."""
  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  num_devices = _get_mesh_axis_size(physical_expert_axis_name, mesh)
  outputs_mla_curr_0, orig_x_curr_1 = outputs_mla_curr
  out_proj_out_curr_0, _ = outputs_mla_curr_0
  x_pspec = jax.typeof(out_proj_out_curr_0).sharding.spec
  local_batch_size = _get_local_dim_size(out_proj_out_curr_0.shape[0], x_pspec[0], mesh)
  local_seq_length = _get_local_dim_size(out_proj_out_curr_0.shape[1], x_pspec[1], mesh)
  max_capacity = int(capacity_factor * (local_batch_size * local_seq_length) * num_devices)
  yarn_freqs_0, yarn_freqs_1 = yarn_freqs
  segment_ids_0, segment_ids_1 = segment_ids

  @jax.named_call
  @program_order(enforce=False)
  def phase1(outputs_mla_curr_0, orig_x_curr_1):
    """Phase 1: mb0 norm, metadata, permute; mb1 wait."""
    out_proj_out_curr_0, orig_x_curr_0 = outputs_mla_curr_0
    (
        (
            router_metadata_curr_0,
            chunk0_metadata_curr_0,
            coeffs_curr_0,
            moe_in_curr_0,
            mla_out_curr_0,
        ),
        aux_curr_0,
    ), res_metadata_curr_0 = ubatch_norm_and_metadata(
        out_proj_out_curr_0,
        orig_x_curr_0,
        w_curr.post_attn_norm_scale,
        w_curr.moe.router,
        norm_fn=norm_fn,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups,
        topk_routing_group=topk_routing_group,
        topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
        max_capacity=max_capacity,
    )
    outputs_permute_curr_0, res_permute_curr_0 = ubatch_permute(
        moe_in_curr_0,
        router_metadata_curr_0,
        chunk0_metadata_curr_0,
        coeffs_curr_0,
        mla_out_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )
    mb1_wait = ubatch_wait(orig_x_curr_1)
    return (outputs_permute_curr_0, mb1_wait, aux_curr_0), (
        *res_metadata_curr_0,
        *res_permute_curr_0,
    )

  @jax.named_call
  @program_order(enforce=False)
  def phase2(outputs_permute_curr_0, mb1):
    """Phase 2: mb0 dispatch; mb1 MLA."""
    outputs_dispatch_curr_0, _ = ubatch_dispatch(
        *outputs_permute_curr_0,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    outputs_mla_curr_1, res_mla_curr_1 = ubatch_mla(
        mb1,
        w_curr.pre_attn_norm_scale,
        w_curr.mla,
        yarn_freqs_1,
        splash_kernel,
        segment_ids=segment_ids_1,
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
    )
    return (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase3(outputs_dispatch_curr_0, outputs_mla_curr_1):
    """Phase 3: mb0 wait; mb1 norm, metadata, permute."""
    outputs_dispatch_wait_0 = ubatch_wait(outputs_dispatch_curr_0)
    out_proj_out_curr_1, orig_x_curr_1 = outputs_mla_curr_1
    (
        (
            router_metadata_curr_1,
            chunk0_metadata_curr_1,
            coeffs_curr_1,
            moe_in_curr_1,
            mla_out_curr_1,
        ),
        aux_curr_1,
    ), res_metadata_curr_1 = ubatch_norm_and_metadata(
        out_proj_out_curr_1,
        orig_x_curr_1,
        w_curr.post_attn_norm_scale,
        w_curr.moe.router,
        norm_fn=norm_fn,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        routed_scaling_factor=routed_scaling_factor,
        n_routing_groups=n_routing_groups,
        topk_routing_group=topk_routing_group,
        topk_in_group=topk_in_group,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
        max_capacity=max_capacity,
    )
    outputs_permute_curr_1, res_permute_curr_1 = ubatch_permute(
        moe_in_curr_1,
        router_metadata_curr_1,
        chunk0_metadata_curr_1,
        coeffs_curr_1,
        mla_out_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )
    return (outputs_dispatch_wait_0, outputs_permute_curr_1, aux_curr_1), (
        *res_metadata_curr_1,
        *res_permute_curr_1,
    )

  @jax.named_call
  @program_order(enforce=False)
  def phase4(outputs_dispatch_curr_0, outputs_permute_curr_1):
    """Phase 4: mb0 expert; mb1 dispatch."""
    outputs_expert_curr_0, res_expert_curr_0 = ubatch_expert(
        *outputs_dispatch_curr_0,
        w_shared=w_curr.moe.shared,
        w_routed=w_curr.moe.routed,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    outputs_dispatch_curr_1, _ = ubatch_dispatch(
        *outputs_permute_curr_1,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (outputs_expert_curr_0, outputs_dispatch_curr_1), res_expert_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase5(outputs_expert_curr_0, outputs_dispatch_curr_1):
    """Phase 5: mb0 combine & loop; mb1 expert."""
    outputs_combine_curr_0, _ = ubatch_combine(
        *outputs_expert_curr_0,
        w_routed=w_curr.moe.routed,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    outputs_expert_curr_1, res_expert_curr_1 = ubatch_expert(
        *outputs_dispatch_curr_1,
        w_shared=w_curr.moe.shared,
        w_routed=w_curr.moe.routed,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    return (outputs_combine_curr_0, outputs_expert_curr_1), res_expert_curr_1

  @jax.named_call
  @program_order(enforce=False)
  def phase6(outputs_combine_curr_0, outputs_expert_curr_1):
    """Phase 6: mb0 unpermute; mb1 wait."""
    outputs_unpermute_curr_0, res_unpermute_curr_0 = ubatch_unpermute(
        *outputs_combine_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        local_seq_length=local_seq_length,
        mesh=mesh,
        out_specs=x_pspec,
    )
    (x_next_0,) = outputs_unpermute_curr_0
    outputs_expert_wait_1 = ubatch_wait(outputs_expert_curr_1)
    return (x_next_0, outputs_expert_wait_1), res_unpermute_curr_0

  @jax.named_call
  @program_order(enforce=False)
  def phase7(x_next_0, outputs_expert_curr_1):
    """Phase 7: mb0 next layer MLA; mb1 combine & loop."""
    outputs_mla_next_0, res_mla_next_0 = ubatch_mla(
        x_next_0,
        w_next.pre_attn_norm_scale,
        w_next.mla,
        yarn_freqs_0,
        splash_kernel,
        segment_ids=segment_ids_0,
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
    )
    outputs_combine_curr_1, _ = ubatch_combine(
        *outputs_expert_curr_1,
        w_routed=w_curr.moe.routed,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (outputs_mla_next_0, outputs_combine_curr_1), res_mla_next_0

  @jax.named_call
  @program_order(enforce=False)
  def phase8(outputs_mla_next_0, outputs_combine_curr_1):
    """Phase 8: mb0 wait; mb1 unpermute."""
    outputs_mla_next_0_wait = ubatch_wait(outputs_mla_next_0)
    outputs_unpermute_curr_1, res_unpermute_curr_1 = ubatch_unpermute(
        *outputs_combine_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        local_seq_length=local_seq_length,
        mesh=mesh,
        out_specs=x_pspec,
    )
    (x_next_1,) = outputs_unpermute_curr_1
    return (outputs_mla_next_0_wait, x_next_1), res_unpermute_curr_1

  @program_order(enforce=True)
  def _scan_body_fwd(outputs_mla_curr_0, mb1):
    (outputs_permute_curr_0, mb1_wait, aux_curr_0), res_phase1_curr_0 = phase1(outputs_mla_curr_0, mb1)
    (outputs_dispatch_curr_0, outputs_mla_curr_1), res_mla_curr_1 = phase2(outputs_permute_curr_0, mb1_wait)
    (outputs_dispatch_wait_0, outputs_permute_curr_1, aux_curr_1), (res_phase3_curr_1) = phase3(
        outputs_dispatch_curr_0, outputs_mla_curr_1
    )
    (outputs_expert_curr_0, outputs_dispatch_curr_1), res_expert_curr_0 = phase4(
        outputs_dispatch_wait_0, outputs_permute_curr_1
    )
    (outputs_combine_curr_0, outputs_expert_curr_1), res_expert_curr_1 = phase5(
        outputs_expert_curr_0, outputs_dispatch_curr_1
    )
    (x_next_0, outputs_expert_wait_1), res_unpermute_curr_0 = phase6(outputs_combine_curr_0, outputs_expert_curr_1)
    (outputs_mla_next_0, outputs_combine_curr_1), res_mla_next_0 = phase7(x_next_0, outputs_expert_wait_1)
    (outputs_mla_next_0_wait, x_next_1), res_unpermute_curr_1 = phase8(outputs_mla_next_0, outputs_combine_curr_1)

    aux_curr = _merge_router_aux(aux_curr_0, aux_curr_1, mesh=mesh)
    return (
        (outputs_mla_next_0_wait, x_next_1),
        aux_curr,
    ), (
        res_phase1_curr_0,
        res_mla_curr_1,
        res_phase3_curr_1,
        res_expert_curr_0,
        res_expert_curr_1,
        res_unpermute_curr_0,
        res_mla_next_0,
        res_unpermute_curr_1,
    )

  (
      (outputs_mla_next_0_wait, x_next_1),
      aux_curr,
  ), (
      res_phase1_curr_0,
      res_mla_curr_1,
      res_phase3_curr_1,
      res_expert_curr_0,
      res_expert_curr_1,
      res_unpermute_curr_0,
      res_mla_next_0,
      res_unpermute_curr_1,
  ) = _scan_body_fwd(outputs_mla_curr_0, orig_x_curr_1)

  (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
  ) = res_phase1_curr_0

  (
      orig_x_curr_1,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
  ) = res_mla_curr_1

  (
      out_proj_out_curr_1,
      _,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
  ) = res_phase3_curr_1

  (chunk0_gate_out_curr_0,) = res_expert_curr_0
  (chunk0_gate_out_curr_1,) = res_expert_curr_1
  (total_combined_x_curr_0,) = res_unpermute_curr_0

  (
      orig_x_next_0,
      q_down_next_0,
      kv_down_next_0,
      context_next_0,
      splash_out_next_0,
  ) = res_mla_next_0

  (total_combined_x_curr_1,) = res_unpermute_curr_1

  (
      out_proj_out_curr_0,
      out_proj_out_curr_1,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      total_combined_x_curr_1,
      q_down_next_0,
      kv_down_next_0,
      context_next_0,
      chunk0_gate_out_curr_0,
      chunk0_gate_out_curr_1,
  ) = jax.tree.map(
      _offload_to_host,
      (
          out_proj_out_curr_0,
          out_proj_out_curr_1,
          q_down_curr_1,
          kv_down_curr_1,
          context_curr_1,
          total_combined_x_curr_1,
          q_down_next_0,
          kv_down_next_0,
          context_next_0,
          chunk0_gate_out_curr_0,
          chunk0_gate_out_curr_1,
      ),
  )

  res = (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
      chunk0_gate_out_curr_0,
      total_combined_x_curr_0,
      out_proj_out_curr_1,
      orig_x_curr_1,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
      chunk0_gate_out_curr_1,
      total_combined_x_curr_1,
      w_curr,
      w_next,
      yarn_freqs,
      splash_kernel,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
      orig_x_next_0,
      q_down_next_0,
      kv_down_next_0,
      context_next_0,
      splash_out_next_0,
  )
  return ((outputs_mla_next_0_wait, x_next_1), aux_curr), res


def _dsv3_sparse_layer_scan_body_bwd(
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    topk_in_group,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    mscale,
    kv_lora_rank,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mesh,
    norm_fn,
    rope_fn,
    gmm_fn,
    axis_mapping,
    expert_axis_name,
    capacity_factor,
    segment_ids,
    res,
    grad_outputs,
):
  """Backward pass for DSv3 sparse layer scan body."""
  grad_carry, grad_aux_curr = grad_outputs
  grad_outputs_mla_next_0, grad_x_next_1 = grad_carry

  (
      out_proj_out_curr_0,
      orig_x_curr_0,
      router_metadata_curr_0,
      coeffs_curr_0,
      chunk0_metadata_curr_0,
      chunk0_gate_out_curr_0,
      total_combined_x_curr_0,
      out_proj_out_curr_1,
      orig_x_curr_1,
      router_metadata_curr_1,
      coeffs_curr_1,
      chunk0_metadata_curr_1,
      chunk0_gate_out_curr_1,
      total_combined_x_curr_1,
      w_curr,
      w_next,
      yarn_freqs,
      splash_kernel,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      splash_out_curr_1,
      orig_x_next_0,
      q_down_next_0,
      kv_down_next_0,
      context_next_0,
      splash_out_next_0,
  ) = res

  (
      out_proj_out_curr_0,
      out_proj_out_curr_1,
      q_down_curr_1,
      kv_down_curr_1,
      context_curr_1,
      total_combined_x_curr_1,
      q_down_next_0,
      kv_down_next_0,
      context_next_0,
      chunk0_gate_out_curr_0,
      chunk0_gate_out_curr_1,
  ) = jax.tree.map(
      _load_to_device,
      (
          out_proj_out_curr_0,
          out_proj_out_curr_1,
          q_down_curr_1,
          kv_down_curr_1,
          context_curr_1,
          total_combined_x_curr_1,
          q_down_next_0,
          kv_down_next_0,
          context_next_0,
          chunk0_gate_out_curr_0,
          chunk0_gate_out_curr_1,
      ),
  )

  mla_out_curr_0 = out_proj_out_curr_0 + orig_x_curr_0
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_0 = norm_fn(mla_out_curr_0, w_curr.post_attn_norm_scale)

  mla_out_curr_1 = out_proj_out_curr_1 + orig_x_curr_1
  with jax.named_scope("post_attn_norm"):
    moe_in_curr_1 = norm_fn(mla_out_curr_1, w_curr.post_attn_norm_scale)

  physical_expert_axis_name = axis_mapping.get(expert_axis_name, expert_axis_name)
  num_devices = _get_mesh_axis_size(physical_expert_axis_name, mesh)
  x_pspec = jax.typeof(moe_in_curr_0).sharding.spec
  local_batch_size = _get_local_dim_size(moe_in_curr_0.shape[0], x_pspec[0], mesh)
  local_seq_length = _get_local_dim_size(moe_in_curr_0.shape[1], x_pspec[1], mesh)
  max_capacity = int(capacity_factor * (local_batch_size * local_seq_length) * num_devices)

  yarn_freqs_0, yarn_freqs_1 = yarn_freqs
  segment_ids_0, segment_ids_1 = segment_ids

  if getattr(grad_aux_curr, "logits", None) is not None:
    grad_logits_0, grad_logits_1 = ops.split_microbatches(grad_aux_curr.logits, mesh=mesh)
    grad_aux_0 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=grad_aux_curr.group_sizes,
        selected_experts=grad_aux_curr.selected_experts,
        logits=grad_logits_0,
    )
    grad_aux_1 = dsv3_types.DSv3RouterAux[Any](
        group_sizes=grad_aux_curr.group_sizes,
        selected_experts=grad_aux_curr.selected_experts,
        logits=grad_logits_1,
    )
  else:
    grad_aux_0 = grad_aux_curr
    grad_aux_1 = grad_aux_curr

  @jax.named_call
  @program_order(enforce=False)
  def phase1(grad_outputs_mla_next_0, grad_x_next_1):
    """Phase 1: mb0 wait; mb1 unpermute pullback & permute remat."""
    grad_outputs_mla_next_0_wait = ubatch_wait(grad_outputs_mla_next_0)
    (
        grad_combined_x_1,
        grad_shared_x_1,
        grad_coeffs_1,
        grad_mla_out_unpermute_1,
    ) = ubatch_unpermute_bwd(
        grad_x_next_1,
        total_combined_x_curr_1,
        coeffs_curr_1,
        router_metadata_curr_1,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )

    def _fwd_permute_1(moe_in):
      return dsv3_router.dsv3_permute(
          moe_in,
          router_metadata_curr_1,
          num_experts_per_tok=num_experts_per_tok,
          mesh=mesh,
      )

    permuted_x_1, vjp_permute_1 = jax.vjp(_fwd_permute_1, moe_in_curr_1)
    return (
        grad_outputs_mla_next_0_wait,
        permuted_x_1,
        grad_combined_x_1,
        grad_shared_x_1,
        vjp_permute_1,
        grad_coeffs_1,
        grad_mla_out_unpermute_1,
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase2(grad_outputs_mla_next_0_wait, permuted_x_1, grad_combined_x_1):
    """Phase 2: mb0 next layer mla backward; mb1 dispatch remat & combine transpose."""
    grad_out_proj_out_next_0, grad_orig_x_next_0 = grad_outputs_mla_next_0_wait
    with jax.named_scope("pre_attn_norm"):
      norm_x_next_0 = norm_fn(orig_x_next_0, w_next.pre_attn_norm_scale)

    grad_norm_x_next_0, grad_w_mla_next_0, grad_yarn_freqs_next_0 = dsv3_mla.dsv3_mla_bwd(
        grad_out_proj_out_next_0,
        norm_x_next_0,
        q_down_next_0,
        kv_down_next_0,
        context_next_0,
        splash_out_next_0,
        w_next.mla,
        yarn_freqs_0,
        splash_kernel,
        segment_ids=segment_ids_0,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        rope_head_dim=rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        mscale=mscale,
        norm_fn=norm_fn,
        rope_fn=rope_fn,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

    def _fwd_pre_norm_next_0(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)

    _, vjp_pre_norm_next_0 = jax.vjp(_fwd_pre_norm_next_0, orig_x_next_0, w_next.pre_attn_norm_scale)
    grad_x_norm_next_0, grad_pre_attn_norm_scale_next_0 = vjp_pre_norm_next_0(grad_norm_x_next_0)
    grad_x_next_0 = grad_x_norm_next_0 + grad_orig_x_next_0

    chunk0_routed_1, chunk0_grad_routed_1 = dsv3_router.dsv3_aggregated_dispatches(
        permuted_x_1,
        grad_combined_x_1,
        chunk0_metadata_curr_1.chunk_send_sizes,
        router_metadata_curr_1.send_sizes,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (
        (
            grad_x_next_0,
            grad_pre_attn_norm_scale_next_0,
            grad_w_mla_next_0,
            grad_yarn_freqs_next_0,
        ),
        (chunk0_routed_1, chunk0_grad_routed_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase3(grad_x_next_0, chunk0_routed_1, chunk0_grad_routed_1):
    """Phase 3: mb0 unpermute pullback & permute remat; mb1 wait."""
    (
        grad_combined_x_0,
        grad_shared_x_0,
        grad_coeffs_0,
        grad_mla_out_unpermute_0,
    ) = ubatch_unpermute_bwd(
        grad_x_next_0,
        total_combined_x_curr_0,
        coeffs_curr_0,
        router_metadata_curr_0,
        num_experts_per_tok=num_experts_per_tok,
        mesh=mesh,
    )

    def _fwd_permute_0(moe_in):
      return dsv3_router.dsv3_permute(
          moe_in,
          router_metadata_curr_0,
          num_experts_per_tok=num_experts_per_tok,
          mesh=mesh,
      )

    permuted_x_0, vjp_permute_0 = jax.vjp(_fwd_permute_0, moe_in_curr_0)
    chunk0_routed_1_wait, chunk0_grad_routed_1_wait = ubatch_wait((chunk0_routed_1, chunk0_grad_routed_1))
    return (
        (
            permuted_x_0,
            grad_combined_x_0,
            grad_shared_x_0,
            vjp_permute_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        (chunk0_routed_1_wait, chunk0_grad_routed_1_wait),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase4(
      permuted_x_0,
      grad_combined_x_0,
      chunk0_routed_1,
      chunk0_grad_routed_1,
      grad_shared_x_1,
  ):
    """Phase 4: mb0 dispatch remat & combine transpose; mb1 expert backward."""
    chunk0_routed_0, chunk0_grad_routed_0 = dsv3_router.dsv3_aggregated_dispatches(
        permuted_x_0,
        grad_combined_x_0,
        chunk0_metadata_curr_0.chunk_send_sizes,
        router_metadata_curr_0.send_sizes,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    def _shared_expert_fwd_1(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)

    _, vjp_shared_1 = jax.vjp(_shared_expert_fwd_1, moe_in_curr_1, w_curr.moe.shared)
    grad_moe_in_shared_1, grad_w_shared_1 = vjp_shared_1(grad_shared_x_1)

    chunk0_grad_routed_in_1, grad_w_routed_0_1 = dsv3_experts.dsv3_routed_experts_chunk0_bwd(
        chunk0_grad_routed_1,
        chunk0_routed_1,
        chunk0_gate_out_curr_1,
        w_curr.moe.routed,
        chunk0_metadata_curr_1.group_sizes,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )
    return (
        (chunk0_routed_0, chunk0_grad_routed_0),
        (
            chunk0_grad_routed_in_1,
            grad_w_routed_0_1,
            grad_moe_in_shared_1,
            grad_w_shared_1,
        ),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase5(
      chunk0_routed_0,
      chunk0_grad_routed_0,
      grad_shared_x_0,
      chunk0_grad_routed_in_1,
      grad_w_routed_0_1,
      permuted_x_1,
      grad_combined_x_1,
  ):
    """Phase 5: mb0 expert backward; mb1 dispatch transpose & iterative dispatch backward."""

    def _shared_expert_fwd_0(moe_in, w_shared):
      return dsv3_experts.dsv3_shared_expert(moe_in, w_shared)

    _, vjp_shared_0 = jax.vjp(_shared_expert_fwd_0, moe_in_curr_0, w_curr.moe.shared)
    grad_moe_in_shared_0, grad_w_shared_0 = vjp_shared_0(grad_shared_x_0)

    chunk0_grad_routed_in_0, grad_w_routed_0_0 = dsv3_experts.dsv3_routed_experts_chunk0_bwd(
        chunk0_grad_routed_0,
        chunk0_routed_0,
        chunk0_gate_out_curr_0,
        w_curr.moe.routed,
        chunk0_metadata_curr_0.group_sizes,
        gmm_fn=gmm_fn,
        mesh=mesh,
    )

    grad_permuted_x_0_1 = dsv3_router.dsv3_combine_chunk(
        chunk0_grad_routed_in_1,
        jnp.empty_like(permuted_x_1),
        chunk0_metadata_curr_1.chunk_send_sizes,
        router_metadata_curr_1.send_sizes,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    final_grad_permuted_x_1, final_grad_w_routed_1 = dsv3_iterative_dispatch_bwd(
        grad_combined_x_1,
        permuted_x_1,
        w_curr.moe.routed,
        router_metadata_curr_1,
        chunk0_metadata_curr_1.next_expert,
        grad_permuted_x_0_1,
        grad_w_routed_0_1,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
    return (
        (
            chunk0_grad_routed_in_0,
            grad_w_routed_0_0,
            grad_moe_in_shared_0,
            grad_w_shared_0,
        ),
        (final_grad_permuted_x_1, final_grad_w_routed_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase6(
      chunk0_grad_routed_in_0,
      grad_w_routed_0_0,
      permuted_x_0,
      grad_combined_x_0,
      final_grad_permuted_x_1,
      vjp_permute_1,
      grad_moe_in_shared_1,
      grad_coeffs_1,
      grad_aux_1,
  ):
    """Phase 6: mb0 dispatch transpose & iterative dispatch backward; mb1 router & post_norm pullback."""
    grad_permuted_x_0_0 = dsv3_router.dsv3_combine_chunk(
        chunk0_grad_routed_in_0,
        jnp.empty_like(permuted_x_0),
        chunk0_metadata_curr_0.chunk_send_sizes,
        router_metadata_curr_0.send_sizes,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    final_grad_permuted_x_0, final_grad_w_routed_0 = dsv3_iterative_dispatch_bwd(
        grad_combined_x_0,
        permuted_x_0,
        w_curr.moe.routed,
        router_metadata_curr_0,
        chunk0_metadata_curr_0.next_expert,
        grad_permuted_x_0_0,
        grad_w_routed_0_0,
        gmm_fn=gmm_fn,
        max_capacity=max_capacity,
        expert_axis_name=expert_axis_name,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )

    (grad_moe_in_permute_1,) = vjp_permute_1(final_grad_permuted_x_1)
    grad_moe_in_expert_1 = grad_moe_in_permute_1 + grad_moe_in_shared_1

    def _fwd_router_1(moe_in, w_router):
      _, coeffs, aux = dsv3_router.dsv3_routing_metadata(
          moe_in,
          w_router,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          mesh=mesh,
      )
      return coeffs, aux.logits

    _, vjp_router_1 = jax.vjp(_fwd_router_1, moe_in_curr_1, w_curr.moe.router)
    grad_moe_in_router_1, grad_w_router_1 = vjp_router_1((grad_coeffs_1, grad_aux_1.logits))
    grad_moe_in_total_1 = grad_moe_in_expert_1 + grad_moe_in_router_1

    def _fwd_post_norm_1(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)

    _, vjp_post_norm_1 = jax.vjp(_fwd_post_norm_1, mla_out_curr_1, w_curr.post_attn_norm_scale)
    grad_mla_out_norm_1, grad_post_attn_norm_scale_1 = vjp_post_norm_1(grad_moe_in_total_1)
    return (
        (final_grad_permuted_x_0, final_grad_w_routed_0),
        (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase7(mb0_state, grad_mla_out_total_1):
    """Phase 7: mb0 wait; mb1 pre_norm and MLA backward."""
    mb0_wait = ubatch_wait(mb0_state)
    with jax.named_scope("pre_attn_norm"):
      norm_x_curr_1 = norm_fn(orig_x_curr_1, w_curr.pre_attn_norm_scale)

    grad_norm_x_1, grad_w_mla_1, grad_yarn_freqs_1 = dsv3_mla.dsv3_mla_bwd(
        grad_mla_out_total_1,
        norm_x_curr_1,
        q_down_curr_1,
        kv_down_curr_1,
        context_curr_1,
        splash_out_curr_1,
        w_curr.mla,
        yarn_freqs_1,
        splash_kernel,
        segment_ids=segment_ids_1,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=qk_head_dim,
        rope_head_dim=rope_head_dim,
        num_query_heads=num_query_heads,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        mscale=mscale,
        norm_fn=norm_fn,
        rope_fn=rope_fn,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

    def _fwd_pre_norm_1(x, pre_attn_norm_scale):
      with jax.named_scope("pre_attn_norm"):
        return norm_fn(x, pre_attn_norm_scale)

    _, vjp_pre_norm_1 = jax.vjp(_fwd_pre_norm_1, orig_x_curr_1, w_curr.pre_attn_norm_scale)
    grad_x_norm_1, grad_pre_attn_norm_scale_1 = vjp_pre_norm_1(grad_norm_x_1)
    grad_mb1 = grad_x_norm_1 + grad_mla_out_total_1

    return (
        mb0_wait,
        (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1),
    ), ()

  @jax.named_call
  @program_order(enforce=False)
  def phase8(
      final_grad_permuted_x_0,
      vjp_permute_0,
      grad_moe_in_shared_0,
      grad_coeffs_0,
      grad_aux_0,
      grad_mb1,
  ):
    """Phase 8: mb0 router & post_norm pullback; mb1 wait."""
    (grad_moe_in_permute_0,) = vjp_permute_0(final_grad_permuted_x_0)
    grad_moe_in_expert_0 = grad_moe_in_permute_0 + grad_moe_in_shared_0

    def _fwd_router_0(moe_in, w_router):
      _, coeffs, aux = dsv3_router.dsv3_routing_metadata(
          moe_in,
          w_router,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          topk_in_group=topk_in_group,
          expert_axis_name=expert_axis_name,
          axis_mapping=axis_mapping,
          mesh=mesh,
      )
      return coeffs, aux.logits

    _, vjp_router_0 = jax.vjp(_fwd_router_0, moe_in_curr_0, w_curr.moe.router)
    grad_moe_in_router_0, grad_w_router_0 = vjp_router_0((grad_coeffs_0, grad_aux_0.logits))
    grad_moe_in_total_0 = grad_moe_in_expert_0 + grad_moe_in_router_0

    def _fwd_post_norm_0(mla_out, post_attn_norm_scale):
      with jax.named_scope("post_attn_norm"):
        return norm_fn(mla_out, post_attn_norm_scale)

    _, vjp_post_norm_0 = jax.vjp(_fwd_post_norm_0, mla_out_curr_0, w_curr.post_attn_norm_scale)
    grad_mla_out_norm_0, grad_post_attn_norm_scale_0 = vjp_post_norm_0(grad_moe_in_total_0)

    grad_mb1_wait = ubatch_wait(grad_mb1)

    return (
        (grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0),
        grad_mb1_wait,
    ), ()

  @program_order(enforce=True)
  def _scan_body_bwd(grad_outputs_mla_next_0, grad_x_next_1):
    (
        (
            grad_outputs_mla_next_0_wait,
            permuted_x_1,
            grad_combined_x_1,
            grad_shared_x_1,
            vjp_permute_1,
            grad_coeffs_1,
            grad_mla_out_unpermute_1,
        ),
        _,
    ) = phase1(grad_outputs_mla_next_0, grad_x_next_1)

    (
        (
            grad_x_next_0,
            grad_pre_attn_norm_scale_next_0,
            grad_w_mla_next_0,
            grad_yarn_freqs_next_0,
        ),
        (chunk0_routed_1, chunk0_grad_routed_1),
    ), _ = phase2(grad_outputs_mla_next_0_wait, permuted_x_1, grad_combined_x_1)

    (
        (
            permuted_x_0,
            grad_combined_x_0,
            grad_shared_x_0,
            vjp_permute_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        (chunk0_routed_1_wait, chunk0_grad_routed_1_wait),
    ), _ = phase3(grad_x_next_0, chunk0_routed_1, chunk0_grad_routed_1)

    (
        (chunk0_routed_0, chunk0_grad_routed_0),
        (
            chunk0_grad_routed_in_1,
            grad_w_routed_0_1,
            grad_moe_in_shared_1,
            grad_w_shared_1,
        ),
    ), _ = phase4(
        permuted_x_0,
        grad_combined_x_0,
        chunk0_routed_1_wait,
        chunk0_grad_routed_1_wait,
        grad_shared_x_1,
    )

    (
        (
            chunk0_grad_routed_in_0,
            grad_w_routed_0_0,
            grad_moe_in_shared_0,
            grad_w_shared_0,
        ),
        (final_grad_permuted_x_1, final_grad_w_routed_1),
    ), _ = phase5(
        chunk0_routed_0,
        chunk0_grad_routed_0,
        grad_shared_x_0,
        chunk0_grad_routed_in_1,
        grad_w_routed_0_1,
        permuted_x_1,
        grad_combined_x_1,
    )

    (
        (final_grad_permuted_x_0, final_grad_w_routed_0),
        (grad_mla_out_norm_1, grad_post_attn_norm_scale_1, grad_w_router_1),
    ), _ = phase6(
        chunk0_grad_routed_in_0,
        grad_w_routed_0_0,
        permuted_x_0,
        grad_combined_x_0,
        final_grad_permuted_x_1,
        vjp_permute_1,
        grad_moe_in_shared_1,
        grad_coeffs_1,
        grad_aux_1,
    )

    grad_mla_out_total_1 = grad_mla_out_unpermute_1 + grad_mla_out_norm_1
    (
        mb0_wait_7,
        (grad_mb1, grad_pre_attn_norm_scale_1, grad_w_mla_1, grad_yarn_freqs_1),
    ), _ = phase7(
        (
            final_grad_permuted_x_0,
            final_grad_w_routed_0,
            vjp_permute_0,
            grad_moe_in_shared_0,
            grad_coeffs_0,
            grad_mla_out_unpermute_0,
        ),
        grad_mla_out_total_1,
    )

    (
        final_grad_permuted_x_0_w,
        final_grad_w_routed_0_w,
        vjp_permute_0_w,
        grad_moe_in_shared_0_w,
        grad_coeffs_0_w,
        grad_mla_out_unpermute_0_w,
    ) = mb0_wait_7

    (
        (grad_mla_out_norm_0, grad_post_attn_norm_scale_0, grad_w_router_0),
        grad_mb1_wait,
    ), _ = phase8(
        final_grad_permuted_x_0_w,
        vjp_permute_0_w,
        grad_moe_in_shared_0_w,
        grad_coeffs_0_w,
        grad_aux_0,
        grad_mb1,
    )

    grad_mla_out_total_0 = grad_mla_out_unpermute_0_w + grad_mla_out_norm_0

    grad_yarn_freqs = (grad_yarn_freqs_next_0, grad_yarn_freqs_1)

    return (
        grad_mla_out_total_0,
        grad_mb1_wait,
        grad_pre_attn_norm_scale_1,
        grad_w_mla_1,
        grad_yarn_freqs,
        grad_post_attn_norm_scale_0,
        grad_post_attn_norm_scale_1,
        grad_w_router_0,
        grad_w_router_1,
        final_grad_w_routed_0_w,
        final_grad_w_routed_1,
        grad_w_shared_0,
        grad_w_shared_1,
        grad_pre_attn_norm_scale_next_0,
        grad_w_mla_next_0,
    )

  (
      grad_mla_out_total_0,
      grad_mb1_wait,
      grad_pre_attn_norm_scale_1,
      grad_w_mla_1,
      grad_yarn_freqs,
      grad_post_attn_norm_scale_0,
      grad_post_attn_norm_scale_1,
      grad_w_router_0,
      grad_w_router_1,
      final_grad_w_routed_0,
      final_grad_w_routed_1,
      grad_w_shared_0,
      grad_w_shared_1,
      grad_pre_attn_norm_scale_next_0,
      grad_w_mla_next_0,
  ) = _scan_body_bwd(grad_outputs_mla_next_0, grad_x_next_1)

  grad_outputs_mla_curr = (
      (grad_mla_out_total_0, grad_mla_out_total_0),
      grad_mb1_wait,
  )

  grad_w_curr = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale_1,
      mla=grad_w_mla_1,
      post_attn_norm_scale=grad_post_attn_norm_scale_0 + grad_post_attn_norm_scale_1,
      moe=dsv3_types.DSv3MoEWeightsPytree(
          router=jax.tree.map(jnp.add, grad_w_router_0, grad_w_router_1),
          routed=jax.tree.map(jnp.add, final_grad_w_routed_0, final_grad_w_routed_1),
          shared=jax.tree.map(jnp.add, grad_w_shared_0, grad_w_shared_1),
      ),
  )

  grad_w_next = dsv3_types.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=grad_pre_attn_norm_scale_next_0,
      mla=grad_w_mla_next_0,
      post_attn_norm_scale=None,
      moe=dsv3_types.DSv3MoEWeightsPytree(),
  )

  return (
      grad_outputs_mla_curr,
      grad_w_curr,
      grad_w_next,
      grad_yarn_freqs,
      None,
  )


_dsv3_sparse_layer_scan_body_vjp.defvjp(
    _dsv3_sparse_layer_scan_body_fwd,
    _dsv3_sparse_layer_scan_body_bwd,
)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layer_scan_body(
    outputs_mla_curr: tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    w_curr: dsv3_types.DSv3SparseLayerWeightsPytree,
    w_next: dsv3_types.DSv3SparseLayerWeightsPytree,
    yarn_freqs: tuple[
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
        tuple[
            jt.Num[jax.Array, "B T 1 R"],
            jt.Num[jax.Array, "B T 1 R"],
        ],
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
    segment_ids: tuple[
        jt.Num[jax.Array, "B T"] | None,
        jt.Num[jax.Array, "B T"] | None,
    ] = (None, None),
) -> tuple[
    tuple[
        tuple[
            jt.Num[jax.Array, "B T D"],
            jt.Num[jax.Array, "B T D"],
        ],
        jt.Num[jax.Array, "B T D"],
    ],
    dsv3_types.DSv3RouterAux[jax.Array],
]:
  """Executes the scan body (layer i epilogue + layer i+1 prologue)."""
  return _dsv3_sparse_layer_scan_body_vjp(
      outputs_mla_curr,
      w_curr,
      w_next,
      yarn_freqs,
      splash_kernel,
      num_experts,
      num_experts_per_tok,
      routed_scaling_factor,
      n_routing_groups,
      topk_routing_group,
      topk_in_group,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      mscale,
      kv_lora_rank,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mesh,
      norm_fn,
      rope_fn,
      gmm_fn,
      axis_mapping,
      expert_axis_name,
      capacity_factor,
      segment_ids,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_sparse_layer(
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
) -> tuple[jt.Num[jax.Array, "B T D"], dsv3_types.DSv3RouterAux[jax.Array]]:
  """Executes a single DSv3 sparse (MoE) layer.

  Args:
    x: Input tokens.
    w: DSv3 weights.
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
    rope_fn: Rope function.
    gmm_fn: GmmFn,
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
  outputs_mla = dsv3_sparse_layer_prologue(
      (mb0, mb1),
      w,
      (yarn_freqs_0, yarn_freqs_1),
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
      segment_ids=(segment_ids_0, segment_ids_1),
  )
  (out_mb0, out_mb1), aux = dsv3_sparse_layer_epilogue(
      outputs_mla,
      w,
      (yarn_freqs_0, yarn_freqs_1),
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
      segment_ids=(segment_ids_0, segment_ids_1),
  )
  out = ops.merge_microbatches(out_mb0, out_mb1, mesh=mesh)
  return out, aux


@jt.jaxtyped(typechecker=typeguard.typechecked)
def collect_w(
    w: dsv3_types.DSv3SparseLayerWeightsPytree,
    *,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> dsv3_types.DSv3SparseLayerWeightsPytree:
  """Collects sparse layer weights across mesh axes."""

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [1]}},
  )
  def _collect_mla_w(pre_attn_norm_scale, mla):
    return (
        ops.collect_along_axis(
            pre_attn_norm_scale,
            ("fsdp_attention", "attention"),
            axis_mapping,
        ),
        dsv3_types.DSv3MLAWeightsPytree(
            q_down=ops.collect_along_axis(mla.q_down, ("fsdp_attention", "attention"), axis_mapping),
            q_up=ops.collect_along_axis(mla.q_up, "fsdp_attention", axis_mapping),
            q_norm_scale=ops.collect_along_axis(mla.q_norm_scale, "fsdp_attention", axis_mapping),
            kv_down=ops.collect_along_axis(mla.kv_down, ("fsdp_attention", "attention"), axis_mapping),
            k_up=ops.collect_along_axis(mla.k_up, "fsdp_attention", axis_mapping),
            v_up=ops.collect_along_axis(mla.v_up, "fsdp_attention", axis_mapping),
            kv_norm_scale=ops.collect_along_axis(mla.kv_norm_scale, "fsdp_attention", axis_mapping),
            out=ops.collect_along_axis(mla.out, "fsdp_attention", axis_mapping),
        ),
    )

  @compute_on(
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [1]}},
  )
  def _collect_other_w(post_attn_norm_scale, router, routed, shared):
    return (
        ops.collect_along_axis(
            post_attn_norm_scale,
            ("fsdp_attention", "attention"),
            axis_mapping,
        ),
        ops.collect_along_axis(router, ("expert", "fsdp_moe"), axis_mapping),
        # Don't collect along the expert axis for routed weights.
        ops.collect_along_axis(routed, "fsdp_moe", axis_mapping),
        ops.collect_along_axis(shared, ("expert", "fsdp_moe"), axis_mapping),
    )

  @program_order(enforce=True)
  def _collect_w(w):
    pre_attn_norm_scale, mla = _collect_mla_w(w.pre_attn_norm_scale, w.mla)
    post_attn_norm_scale, router, routed, shared = _collect_other_w(
        w.post_attn_norm_scale, w.moe.router, w.moe.routed, w.moe.shared
    )
    return dsv3_types.DSv3SparseLayerWeightsPytree(
        pre_attn_norm_scale=pre_attn_norm_scale,
        mla=mla,
        post_attn_norm_scale=post_attn_norm_scale,
        moe=dsv3_types.DSv3MoEWeightsPytree(
            router=router,
            routed=routed,
            shared=shared,
        ),
    )

  return _collect_w(w)
