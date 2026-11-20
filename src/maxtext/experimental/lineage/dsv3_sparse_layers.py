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

"""DeepSeekV3 scan over multiple sparse layers."""

from collections.abc import Callable, Mapping
import functools
from typing import Any, Protocol

import jax
import jaxtyping as jt
import typeguard

from maxtext.experimental.lineage import dsv3_expert_shuffle
from maxtext.experimental.lineage import dsv3_sparse_layer
from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization


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


GmmFn = dsv3_sparse_layer.GmmFn


def _with_expert_permutations(fn: Callable[..., Any], *, accepts: bool = True) -> Callable[..., Any]:
  """Adapts `fn` to the pipelined scan's per-layer `w_aux` keyword.

  Args:
    fn: Collect or layer function.
    accepts: Whether `fn` takes an `expert_permutations` keyword. If not, the
      per-layer permutation is dropped (e.g. for the prologue, which runs no
      router, and the backward passes, which replay the saved routing).

  Returns:
    `fn` taking `w_aux` and forwarding it as `expert_permutations`.
  """

  @functools.wraps(fn)
  def wrapped(*args, w_aux=None, **kwargs):
    if accepts:
      kwargs["expert_permutations"] = w_aux
    return fn(*args, **kwargs)

  return wrapped


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
    capacity_factor: float = 2.0,
    router_dtype: jax.typing.DTypeLike | None = None,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
    ragged_buffer_factor: float,
    quant: quantization.QuantConfig | None = None,
    expert_permutations: jt.Int[jax.Array, "L E"] | None = None,
    mla_offload_fields: tuple[str, ...] = (),
    moe_offload_fields: tuple[str, ...] = (),
    offload_bank: bool = False,
    offload_w_grad: bool = False,
    stream_param_cast: bool = False,
    compute_dtype: jax.typing.DTypeLike | None = None,
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
    capacity_factor: Each MoE chunk processes `capacity_factor` times the
      average number of slots routed to local experts per device (local tokens
      times `num_experts_per_tok`), looping over chunks as needed. `1.0`
      suffices for a single chunk under perfectly balanced routing, and the
      number of expert shards guarantees a single chunk.
    router_dtype: Optional dtype for MoE expert selection; see
      `dsv3_router.expert_selection`.
    segment_ids: Optional segment IDs for sequence packing.
    ragged_buffer_factor: Safety factor for cross-layer ragged activation bank.
    quant: Optional quantization config.
    expert_permutations: Optional per-layer expert shuffle of shape
      `(num_layers, num_experts)`. With `L` experts per expert shard and `p` the
      entry at `[j, i]`, logical expert `i` of layer `j` is placed at physical
      position `p`, i.e. local slot `p % L` of expert shard `p // L`. The routed
      expert weights are shuffled on the fly before their FSDP all-gather (and
      their gradients unshuffled after the FSDP reduce-scatter), so `w`, its
      gradient, and `aux` all stay in logical expert order. `None` performs no
      shuffle.

  Returns:
    A tuple of (output_tokens, aux).
  """

  num_layers = jax.tree.leaves(w)[0].shape[0]
  if expert_permutations is not None:
    dsv3_expert_shuffle.validate_expert_permutations(expert_permutations, num_experts=num_experts, num_layers=num_layers)
    if expert_permutations.shape[0] != num_layers:
      raise ValueError(f"expert_permutations has {expert_permutations.shape[0]} rows," f" expected {num_layers}")
  mb0, mb1 = ops.split_microbatches(x, mesh=mesh)
  yarn_freqs_0, yarn_freqs_1 = ops.split_microbatches(yarn_freqs, mesh=mesh)
  segment_ids_0, segment_ids_1 = ops.split_microbatches(segment_ids, mesh=mesh)

  quantize_enabled = bool(quant and (quant.routed_experts or quant.mla or quant.shared_mlp))

  def _cast_sparse_layer(w_layer):
    if not stream_param_cast or compute_dtype is None:
      return w_layer

    def _cast_leaf(path, a):
      if a is None:
        return None
      if path and getattr(path[-1], "name", None) == "bias":
        return a
      return a.astype(compute_dtype)

    return jax.tree_util.tree_map_with_path(_cast_leaf, w_layer, is_leaf=lambda a: a is None)

  def _collect_w_dcn(w_layer):
    if stream_param_cast:
      w_layer = _cast_sparse_layer(w_layer)
    w_dcn = dsv3_sparse_layer.collect_w_dcn(w_layer, axis_mapping=axis_mapping)
    if stream_param_cast and quantize_enabled:
      assert quant is not None
      w_dcn = dsv3_sparse_layer.quantize_weights(w_dcn, quant=quant)
    return w_dcn

  def _reduce_w_dcn(grad_w, like):
    # Return reduced gradients in compute/grad dtype (`bfloat16`) inside the
    # backward scan so the 58-layer stacked accumulator takes 10.5 GiB instead
    # of 21.0 GiB `float32`; `w_collect_pipelined_scan_prologue_epilogue_bwd`
    # restores the primal `w` leaf dtypes once outside the scan where XLA fuses
    # the `bf16 -> f32` cast directly into Optax AdamW.
    return dsv3_sparse_layer.reduce_w_dcn(grad_w, like, axis_mapping=axis_mapping)

  _collect_w_ici_fwd = functools.partial(
      dsv3_sparse_layer.collect_w_ici,
      axis_mapping=axis_mapping,
      split=True,
      expert_axis_name=expert_axis_name,
  )
  _collect_w_ici_bwd = functools.partial(
      dsv3_sparse_layer.collect_w_ici,
      axis_mapping=axis_mapping,
      split=False,
      expert_axis_name=expert_axis_name,
  )
  w_first = _collect_w_ici_fwd(_collect_w_dcn(jax.tree.map(lambda _w: _w[0], w)))
  (bank_0, bank_offset_0), (bank_1, bank_offset_1) = dsv3_sparse_layer.init_activation_bank(
      mb0,
      w_first,
      num_layers=num_layers,
      num_experts_per_tok=num_experts_per_tok,
      ragged_buffer_factor=ragged_buffer_factor,
      capacity_factor=capacity_factor,
      expert_axis_name=expert_axis_name,
      axis_mapping=axis_mapping,
      mesh=mesh,
  )
  _prologue = ops.PipelinedScanLayer(
      fwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_prologue_fwd,
          yarn_freqs=(yarn_freqs_0, yarn_freqs_1),
          splash_kernel=splash_kernel,
          banks=(bank_0, bank_1),
          bank_offsets=(bank_offset_0, bank_offset_1),
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
          quant=quant,
      ),
      bwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_prologue_bwd,
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
          quant=quant,
      ),
  )

  def _scan_body_fwd(*args, **kwargs):
    out, res = dsv3_sparse_layer.dsv3_sparse_layer_scan_body_fwd(*args, **kwargs)
    assert res.mb0.moe is not None
    moe_0 = res.mb0.moe._replace(
        out_proj_out=res.mb0.moe.out_proj_out + res.mb0.moe.orig_x,
        orig_x=None,  # pyrefly: ignore[bad-argument-type]
    )
    return out, res._replace(mb0=res.mb0._replace(moe=moe_0))

  _scan_body = ops.PipelinedScanLayer(
      fwd=functools.partial(
          _scan_body_fwd,
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
          router_dtype=router_dtype,
          segment_ids=(segment_ids_0, segment_ids_1),
          quant=quant,
      ),
      bwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_scan_body_bwd,
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
          router_dtype=router_dtype,
          segment_ids=(segment_ids_0, segment_ids_1),
          quant=quant,
      ),
  )
  _epilogue = ops.PipelinedScanLayer(
      fwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_epilogue_fwd,
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
          router_dtype=router_dtype,
          segment_ids=(segment_ids_0, segment_ids_1),
          quant=quant,
      ),
      bwd=functools.partial(
          dsv3_sparse_layer.dsv3_sparse_layer_epilogue_bwd,
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
          router_dtype=router_dtype,
          segment_ids=(segment_ids_0, segment_ids_1),
          quant=quant,
      ),
  )
  _reduce_w_ici_sc1 = functools.partial(
      dsv3_sparse_layer.reduce_w_ici_sc1,
      axis_mapping=axis_mapping,
  )
  _reduce_w_ici_sc0 = functools.partial(dsv3_sparse_layer.reduce_w_ici_sc0, axis_mapping=axis_mapping)
  collect_fn_pre = None
  reduce_fn_pre = None
  if expert_permutations is not None:
    # The pipelined scan slices the per-layer permutation alongside `w` and
    # hands it to the collect and layer functions as `w_aux`. The shuffle is a
    # TensorCore collective that the pipelined scan issues one step ahead of
    # the ICI collect (and its transpose one step after the first stage of the
    # ICI reduction), so that neither waits on the other.
    collect_fn_pre = _with_expert_permutations(
        functools.partial(
            dsv3_sparse_layer.shuffle_w_routed,
            axis_mapping=axis_mapping,
            expert_axis_name=expert_axis_name,
        )
    )
    reduce_fn_pre = _with_expert_permutations(
        functools.partial(
            dsv3_sparse_layer.unshuffle_w_routed,
            axis_mapping=axis_mapping,
            expert_axis_name=expert_axis_name,
        )
    )
    _collect_w_ici_fwd = _with_expert_permutations(_collect_w_ici_fwd, accepts=False)
    _collect_w_ici_bwd = _with_expert_permutations(_collect_w_ici_bwd, accepts=False)
    _reduce_w_ici_sc1 = _with_expert_permutations(_reduce_w_ici_sc1, accepts=False)
    _reduce_w_ici_sc0 = _with_expert_permutations(_reduce_w_ici_sc0, accepts=False)
    _prologue = ops.PipelinedScanLayer(
        fwd=_with_expert_permutations(_prologue.fwd, accepts=False),
        bwd=_with_expert_permutations(_prologue.bwd, accepts=False),
    )
    _scan_body = ops.PipelinedScanLayer(
        fwd=_with_expert_permutations(_scan_body.fwd),
        bwd=_with_expert_permutations(_scan_body.bwd, accepts=False),
    )
    _epilogue = ops.PipelinedScanLayer(
        fwd=_with_expert_permutations(_epilogue.fwd),
        bwd=_with_expert_permutations(_epilogue.bwd, accepts=False),
    )
  (out_mb0, out_mb1), aux = ops.w_collect_pipelined_scan_prologue_epilogue(
      _prologue,
      _scan_body,
      _epilogue,
      (mb0, mb1),
      w,
      _collect_w_dcn,
      _collect_w_ici_fwd,
      _reduce_w_dcn,
      _reduce_w_ici_sc1,
      _reduce_w_ici_sc0,
      _collect_w_ici_bwd,
      # Quantizes the routed expert and MLA weights of all layers once, unless
      # `stream_param_cast` quantizes one layer at a time inside `_collect_w_dcn`.
      quantize_fn=(
          functools.partial(dsv3_sparse_layer.quantize_weights, quant=quant)
          if (quantize_enabled and not stream_param_cast)
          else None
      ),
      offload_fn=functools.partial(
          dsv3_sparse_layer.offload_residuals,
          mla_fields=mla_offload_fields,
          moe_fields=moe_offload_fields,
          offload_bank=offload_bank,
      ),
      load_fn=functools.partial(
          dsv3_sparse_layer.load_residuals,
          mla_fields=mla_offload_fields,
          moe_fields=moe_offload_fields,
          offload_bank=offload_bank,
      ),
      w_aux=expert_permutations,
      collect_fn_pre=collect_fn_pre,
      reduce_fn_pre=reduce_fn_pre,
      offload_w_grad=offload_w_grad,
  )
  out = ops.merge_microbatches(out_mb0, out_mb1, mesh=mesh)
  return out, aux
