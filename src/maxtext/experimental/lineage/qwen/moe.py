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

"""Pure-JAX Qwen MoE entrypoints (shared expert, routing, dispatch, GMM, combine)."""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen import moe_kernels
from maxtext.experimental.lineage.qwen import ops
from maxtext.experimental.lineage.qwen.config import QwenConfig
from maxtext.kernels.ragged import ragged_gather_tc as tc
from maxtext.kernels.ragged import ragged_sort
from maxtext.kernels.ragged import ragged_sort_tc

Fp8SteWeight = moe_kernels.Fp8SteWeight
gather_routed_weight_sc1 = moe_kernels.gather_routed_weight_sc1
routed_moe_forked_3d_fp8 = moe_kernels.routed_moe_forked_3d_fp8


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GatheredMoeWeights:
  """Gathered per-sublayer routed and shared-expert MoE weights."""

  router_gate: jax.Array
  wi: jax.Array | moe_kernels.Fp8SteWeight
  wo: jax.Array | moe_kernels.Fp8SteWeight
  shared_wi_0: Any
  shared_wi_1: Any
  shared_wo: Any
  shared_gate: jax.Array


class DispatchedMoeTokens(NamedTuple):
  """All-gathered EP tokens (`2D` or `3D` / `Fp8DispatchedTokens`) and routing."""

  tokens_ep: jax.Array | ops.Fp8DispatchedTokens
  topk_weights_ep: jax.Array
  topk_ids_ep: jax.Array


def order_dispatched_after(
    disp: DispatchedMoeTokens,
    deps: jax.Array | ops.Fp8DispatchedTokens | tuple[object, ...],
) -> DispatchedMoeTokens:
  """Attaches forward-only ordering tie(s) onto `disp.tokens_ep` (4B tie when FP8)."""
  dep_tuple = deps if isinstance(deps, tuple) else (deps,)
  tok_ep = disp.tokens_ep
  for dep in dep_tuple:
    tok_ep = ops.order_after(tok_ep, dep)
  return DispatchedMoeTokens(
      tokens_ep=tok_ep,
      topk_weights_ep=disp.topk_weights_ep,
      topk_ids_ep=disp.topk_ids_ep,
  )


def gather_moe_weights(
    p: Any,
    wi_absmax: jax.Array | None,
    wo_absmax: jax.Array | None,
    cfg: QwenConfig,
) -> GatheredMoeWeights:
  """Gathers routed and shared-expert MoE weights over `'fsdp'` on SC1."""
  is_fe = cfg.dense_sharding_mode == "fsdp_expert"
  router_fe = cfg.dense_proj_fsdp_expert(cfg.num_experts)
  shared_fe = cfg.dense_proj_fsdp_expert(cfg.shared_mlp_dim)
  use_shared_fp8 = cfg.dense_fp8_matmul or cfg.dense_fp8_shared_expert
  if cfg.replicate_shared_gate:
    shared_gate_w = ops.replicate_small_param(p.shared_gate, cfg)
  else:
    shared_gate_w = ops.gather_dense_weight(
        p.shared_gate, cfg, sharded_axis=0, is_fsdp_expert=is_fe
    )
  return GatheredMoeWeights(
      router_gate=ops.gather_dense_weight(
          p.router_gate, cfg, sharded_axis=0, is_fsdp_expert=router_fe
      ),
      wi=moe_kernels.gather_routed_weight_sc1(p.wi, wi_absmax, cfg, is_wi=True),
      wo=moe_kernels.gather_routed_weight_sc1(
          p.wo, wo_absmax, cfg, is_wi=False
      ),
      shared_wi_0=ops.gather_dense_weight(
          p.shared_wi_0,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          use_fp8=use_shared_fp8,
      ),
      shared_wi_1=ops.gather_dense_weight(
          p.shared_wi_1,
          cfg,
          sharded_axis=0,
          is_fsdp_expert=shared_fe,
          use_fp8=use_shared_fp8,
      ),
      shared_wo=ops.gather_dense_weight(
          p.shared_wo,
          cfg,
          sharded_axis=1,
          is_fsdp_expert=shared_fe,
          use_fp8=use_shared_fp8,
      ),
      shared_gate=shared_gate_w,
  )


def shared_expert_and_residual(
    x_post_mixer: jax.Array,
    normed_post_mixer: jax.Array,
    w_moe: GatheredMoeWeights,
) -> jax.Array:
  """Computes `x_post_mixer + sigmoid(shared_gate(normed)) * shared_mlp(normed)`."""
  mm = ops.dense_fp8_matmul if hasattr(ops, "dense_fp8_matmul") else jnp.matmul
  g0 = mm(normed_post_mixer, w_moe.shared_wi_0)
  u0 = mm(normed_post_mixer, w_moe.shared_wi_1)
  act = (jax.nn.silu(g0.astype(jnp.float32)) * u0.astype(jnp.float32)).astype(
      x_post_mixer.dtype
  )
  shared_out = mm(act, w_moe.shared_wo)
  gate_prob = jax.nn.sigmoid(
      jnp.matmul(normed_post_mixer, w_moe.shared_gate).astype(jnp.float32)
  )
  gated_shared = (shared_out.astype(jnp.float32) * gate_prob).astype(
      x_post_mixer.dtype
  )
  return x_post_mixer + gated_shared


def moe_local_route_only(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
    sublayer_idx: int = 0,
    batch_offset: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> tuple[jax.Array, jax.Array]:
  """Computes local top-k routing `(topk_weights, topk_ids)` ahead of Shared Expert."""
  bsz, seq_len, emb_dim = normed_tokens.shape
  flat_tokens = normed_tokens.reshape(bsz * seq_len, emb_dim)

  if cfg.float32_gate_logits:
    logits = jnp.matmul(
        flat_tokens.astype(jnp.float32), router_gate.astype(jnp.float32)
    )
  else:
    logits = jnp.matmul(flat_tokens, router_gate).astype(jnp.float32)
  if route_after is not None:
    deps = route_after if isinstance(route_after, tuple) else (route_after,)
    for dep in deps:
      logits = ops.order_after(logits, dep)

  probs = jax.nn.softmax(logits, axis=-1)
  if cfg.routing_mode == "router":
    topk_weights, topk_ids = jax.lax.top_k(probs, cfg.num_experts_per_tok)
  elif cfg.routing_mode == "random":
    eff_layer_idx = sublayer_idx if layer_idx is None else layer_idx
    topk_ids = ops.sample_random_topk_ids(
        (bsz, seq_len),
        cfg,
        layer_idx=eff_layer_idx,
        step=step,
        batch_offset=batch_offset,
        router_logits_f32=logits,
    )
    topk_weights = ops.dense_select_topk_probs(probs, topk_ids, cfg.num_experts)
  else:
    raise ValueError(f"Unsupported routing_mode={cfg.routing_mode!r}")

  if cfg.norm_topk_prob:
    denom = jnp.sum(topk_weights, axis=-1, keepdims=True)
    topk_weights = topk_weights / jnp.where(denom > 0.0, denom, 1.0)
  return (
      checkpoint_name(topk_weights, "moe_routing"),
      checkpoint_name(topk_ids, "moe_routing"),
  )


def moe_dispatch_from_routed(
    normed_tokens: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    cfg: QwenConfig,
    *,
    order_tokens_on_router: bool = True,
) -> DispatchedMoeTokens:
  """Launches SC0-pinned EP token (`2D` or zero-relayout `3D`) + routing all-gathers."""
  moe_kernels.assert_global_mixed_precision_contract(cfg)
  bsz, seq_len, emb_dim = normed_tokens.shape
  use_3d = (
      cfg.moe_use_tc_ragged and cfg.moe_3d_ep_tokens and (emb_dim % 128 == 0)
  )
  tokens_disp = (
      normed_tokens.reshape(bsz * seq_len, emb_dim // 128, 128)
      if use_3d
      else normed_tokens.reshape(bsz * seq_len, emb_dim)
  )
  if order_tokens_on_router:
    tokens_disp = ops.order_after(tokens_disp, topk_weights)

  use_3d_fp8_silu = (
      use_3d
      and cfg.moe_use_ragged_silu_3d_fp8
      and cfg.moe_fp8_token_dispatch
      and cfg.moe_fp8_weights
      and cfg.moe_fp8_gmm_compute
  )
  tokens_ep = (
      ops.ep_token_dispatch_3d_fp8(tokens_disp, cfg)
      if use_3d_fp8_silu
      else ops.ep_token_dispatch(tokens_disp, cfg)
  )
  weights_ep = ops.ep_routing_weights_dispatch(topk_weights, cfg)
  ids_ep = ops.ep_routing_ids_dispatch(topk_ids, cfg)
  return DispatchedMoeTokens(
      tokens_ep=tokens_ep,
      topk_weights_ep=weights_ep,
      topk_ids_ep=ids_ep,
  )


def moe_local_route_and_dispatch(
    normed_tokens: jax.Array,
    router_gate: jax.Array,
    cfg: QwenConfig,
    *,
    route_after: jax.Array | tuple[jax.Array, ...] | None = None,
    sublayer_idx: int = 0,
    batch_offset: int = 0,
    layer_idx: jax.Array | int | None = None,
    step: jax.Array | int = 0,
) -> DispatchedMoeTokens:
  """Computes local top-k routing and launches SC0-pinned EP dispatch all-gathers."""
  topk_weights, topk_ids = moe_local_route_only(
      normed_tokens,
      router_gate,
      cfg,
      route_after=route_after,
      sublayer_idx=sublayer_idx,
      batch_offset=batch_offset,
      layer_idx=layer_idx,
      step=step,
  )
  return moe_dispatch_from_routed(
      normed_tokens,
      topk_weights,
      topk_ids,
      cfg,
      order_tokens_on_router=(route_after is not None),
  )


def moe_expert_gmm_and_unsort(
    dispatched: DispatchedMoeTokens,
    wi: jax.Array | moe_kernels.Fp8SteWeight,
    wo: jax.Array | moe_kernels.Fp8SteWeight,
    cfg: QwenConfig,
    *,
    return_weights: bool = False,
    forward_tie: object | None = None,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
    use_external_bwd_rw: bool = False,
    use_external_bwd_disp: bool = False,
    return_routing: bool = False,
):
  """Executes routed MoE local sort -> FP8/BF16 `wi` GMM -> `wo` GMM -> unsort."""
  moe_kernels.assert_global_mixed_precision_contract(cfg)
  ubatch_tokens_local = dispatched.tokens_ep.shape[0] // cfg.ep_size
  buf_size = cfg.ragged_buffer_size(ubatch_tokens_local)
  total_ep_slots = dispatched.tokens_ep.shape[0] * cfg.num_experts_per_tok
  if buf_size > total_ep_slots:
    raise ValueError(
        f"Packed local ragged buffer requires buf_size ({buf_size}) <="
        f" total_ep_slots ({total_ep_slots})."
    )
  fallback = not cfg.use_pallas_kernels

  if isinstance(dispatched.tokens_ep, ops.Fp8DispatchedTokens):
    if not (
        isinstance(wi, moe_kernels.Fp8SteWeight)
        and isinstance(wo, moe_kernels.Fp8SteWeight)
    ):
      raise ValueError(
          "Fp8DispatchedTokens requires Fp8SteWeight for both `wi` and `wo`."
      )
    num_tokens_ep = dispatched.tokens_ep.shape[0]
    topk = cfg.num_experts_per_tok
    e_local = cfg.local_num_experts
    topk_ids_flat = dispatched.topk_ids_ep.reshape(-1).astype(jnp.int32)
    w_flat = jnp.ravel(dispatched.topk_weights_ep)
    ep_idx = ops.ep_axis_index(cfg)
    if cfg.moe_local_65key_sort:
      # General `(e_local + 1)`-key local sort (`33` keys at EP=16, `65` keys
      # at EP=8): maps local experts to `0 .. e_local - 1` and all non-local
      # experts to sentinel `e_local`. Uses 1-array `packed_stable_sort` on the
      # full `N`-slot routing array + `tc_gather_1d_unique` on the `buf_size`
      # local slice so neither forward `gather_fusion` nor backward `scatter`
      # is emitted while avoiding the 2-array `N`-element weight sort.
      local_expert = topk_ids_flat - ep_idx * e_local
      sort_key = jnp.where(
          (local_expert >= 0) & (local_expert < e_local),
          local_expert,
          e_local,
      )
      if cfg.use_pallas_kernels:
        sorted_keys, sort_indices = tc.packed_stable_sort(sort_key, e_local)
      else:
        sort_indices = jnp.argsort(sort_key, stable=True).astype(jnp.int32)
        sorted_keys = jnp.take(sort_key, sort_indices, axis=0)
      slot_order = sort_indices[:buf_size]
      sk_win = sorted_keys[:buf_size]
      ends = jnp.searchsorted(
          sk_win,
          jnp.arange(1, e_local + 1, dtype=jnp.int32),
          side="left",
          method="scan_unrolled",
      ).astype(jnp.int32)
      starts = jnp.concatenate(
          [jnp.zeros((1,), dtype=jnp.int32), ends[:-1]], axis=0
      )
      local_gs = ends - starts
      count = ends[-1]
      w_rows = tc.tc_gather_1d_unique(
          w_flat,
          slot_order,
          count,
          out_dtype=jnp.float32,
          use_pallas=cfg.use_pallas_kernels,
      )
    else:
      if cfg.use_pallas_kernels:
        _, topk_argsort_indices = tc.packed_stable_sort(
            topk_ids_flat, cfg.num_experts - 1
        )
      else:
        topk_argsort_indices = jnp.argsort(topk_ids_flat, stable=True).astype(
            jnp.int32
        )
      group_sizes_global = jax.nn.one_hot(
          topk_ids_flat, cfg.num_experts, dtype=jnp.int32
      ).sum(axis=0)
      group_offsets = jnp.cumulative_sum(
          group_sizes_global, include_initial=True
      )
      start_raw = group_offsets[ep_idx * e_local].astype(jnp.int32)
      end_raw = group_offsets[(ep_idx + 1) * e_local].astype(jnp.int32)
      count = jnp.minimum(end_raw - start_raw, buf_size).astype(jnp.int32)
      slot_order = jax.lax.dynamic_slice_in_dim(
          jnp.pad(topk_argsort_indices, (0, buf_size)),
          start_raw,
          buf_size,
          axis=0,
      )
      w_rows = tc.tc_gather_1d_unique(
          w_flat,
          slot_order,
          count,
          out_dtype=jnp.float32,
          use_pallas=cfg.use_pallas_kernels,
      )
      local_gs = jax.lax.dynamic_slice_in_dim(
          group_sizes_global, ep_idx * e_local, e_local, axis=0
      )
      prefix_excl = jnp.cumsum(local_gs) - local_gs
      local_gs = jnp.minimum(local_gs, jnp.maximum(0, buf_size - prefix_excl))

    token_ids_sorted = jax.lax.stop_gradient(slot_order // topk)
    if cfg.use_pallas_kernels:
      reduce_meta = tc.ragged_gather_reduce_tc_metadata(
          token_ids_sorted,
          0,
          count,
          top_k=topk,
          block_size=cfg.tc_reduce_block_size,
          num_out_tokens=num_tokens_ep,
          packed_sort=True,
      )
    else:
      reduce_meta = None
    local_gs = checkpoint_name(local_gs, "moe_routing")
    token_ids_sorted = checkpoint_name(token_ids_sorted, "moe_routing")
    count = checkpoint_name(count, "moe_routing")
    slot_order = checkpoint_name(slot_order, "moe_routing")
    if reduce_meta is not None:
      reduce_meta = jax.tree.map(
          lambda v: (
              checkpoint_name(v, "moe_routing")
              if isinstance(v, jax.Array)
              else v
          ),
          reduce_meta,
      )
    w_rows = checkpoint_name(w_rows, "moe_routing")
    saved_routing = (
        ops.MoeSavedRouting(
            slot_order=slot_order,
            local_gs=local_gs,
            count=count,
            reduce_meta=reduce_meta,
            w_rows=w_rows.astype(dispatched.topk_weights_ep.dtype),
        )
        if return_routing
        else None
    )
    if cfg.moe_use_forked_3d_gmmv2:
      if use_external_bwd_rw:
        if return_weights:
          partial_ep, wi_ret, wo_ret, rw_slot = (
              moe_kernels.routed_moe_forked_3d_fp8(
                  dispatched.tokens_ep,
                  w_rows,
                  wi,
                  wo,
                  local_gs,
                  token_ids_sorted,
                  count,
                  reduce_meta,
                  cfg,
                  buf_size=buf_size,
                  return_rhs=True,
                  forward_tie=forward_tie,
                  wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
                  sublayer_idx=sublayer_idx,
                  use_external_bwd_rw=True,
                  use_external_bwd_disp=use_external_bwd_disp,
              )
          )
          if forward_tie is None:
            return partial_ep, wi_ret, wo_ret, rw_slot
          return partial_ep, forward_tie, wi_ret, wo_ret, rw_slot
        partial_ep, rw_slot = moe_kernels.routed_moe_forked_3d_fp8(
            dispatched.tokens_ep,
            w_rows,
            wi,
            wo,
            local_gs,
            token_ids_sorted,
            count,
            reduce_meta,
            cfg,
            buf_size=buf_size,
            return_rhs=False,
            forward_tie=forward_tie,
            wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
            sublayer_idx=sublayer_idx,
            use_external_bwd_rw=True,
            use_external_bwd_disp=use_external_bwd_disp,
        )
        if forward_tie is not None:
          return partial_ep, forward_tie, rw_slot
        return partial_ep, rw_slot
      if return_weights:
        partial_ep, wi_ret, wo_ret = moe_kernels.routed_moe_forked_3d_fp8(
            dispatched.tokens_ep,
            w_rows,
            wi,
            wo,
            local_gs,
            token_ids_sorted,
            count,
            reduce_meta,
            cfg,
            buf_size=buf_size,
            return_rhs=True,
            forward_tie=forward_tie,
            wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
            sublayer_idx=sublayer_idx,
        )
        if forward_tie is None:
          return partial_ep, wi_ret, wo_ret
        return partial_ep, forward_tie, wi_ret, wo_ret
      partial_ep = moe_kernels.routed_moe_forked_3d_fp8(
          dispatched.tokens_ep,
          w_rows,
          wi,
          wo,
          local_gs,
          token_ids_sorted,
          count,
          reduce_meta,
          cfg,
          buf_size=buf_size,
          return_rhs=False,
          forward_tie=forward_tie,
          wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
          sublayer_idx=sublayer_idx,
      )
      if return_routing:
        return partial_ep, saved_routing
      if forward_tie is not None:
        return partial_ep, forward_tie
      return partial_ep

    if return_weights:
      partial_ep, wi_ret, wo_ret = moe_kernels.routed_moe_3d_fp8_silu(
          dispatched.tokens_ep,
          w_rows,
          wi,
          wo,
          local_gs,
          token_ids_sorted,
          count,
          reduce_meta,
          cfg,
          buf_size=buf_size,
          return_rhs=True,
          forward_tie=forward_tie,
      )
      if forward_tie is None:
        return partial_ep, wi_ret, wo_ret
      return partial_ep, forward_tie, wi_ret, wo_ret
    partial_ep = moe_kernels.routed_moe_3d_fp8_silu(
        dispatched.tokens_ep,
        w_rows,
        wi,
        wo,
        local_gs,
        token_ids_sorted,
        count,
        reduce_meta,
        cfg,
        buf_size=buf_size,
        return_rhs=False,
        forward_tie=forward_tie,
    )
    if return_routing:
      return partial_ep, saved_routing
    if forward_tie is not None:
      return partial_ep, forward_tie
    return partial_ep

  if cfg.moe_use_tc_ragged:
    use_3d = cfg.moe_3d_ep_tokens and (dispatched.tokens_ep.ndim == 3)
    ep_idx = ops.ep_axis_index(cfg)

    if cfg.use_pallas_kernels:
      sorted_tokens, group_sizes_global, _, routing = (
          ragged_sort_tc.ring_ragged_sort_tc(
              dispatched.tokens_ep,
              dispatched.topk_ids_ep,
              cfg.num_experts,
              cfg.num_experts_per_tok,
              cfg.ep_axis_names,
              cfg.ep_size,
              buf_size,
              gather_block_size=cfg.tc_gather_block_size,
              reduce_block_size=cfg.tc_reduce_block_size,
              mask_padding=True,
              keep_3d=use_3d,
              window_routing=True,
          )
      )
      local_gs = jax.lax.dynamic_slice_in_dim(
          group_sizes_global,
          ep_idx * cfg.local_num_experts,
          cfg.local_num_experts,
          axis=0,
      )
      prefix_excl = jnp.cumsum(local_gs) - local_gs
      local_gs = jnp.minimum(local_gs, jnp.maximum(0, buf_size - prefix_excl))
      w_rows = ragged_sort_tc.tc_buffer_row_weights(
          routing, jnp.ravel(dispatched.topk_weights_ep), buf_size
      )
    else:
      topk = cfg.num_experts_per_tok
      topk_ids_flat = dispatched.topk_ids_ep.reshape(-1).astype(jnp.int32)
      slot_order_all = jnp.argsort(topk_ids_flat, stable=True).astype(jnp.int32)
      group_sizes_global = jnp.sum(
          jax.nn.one_hot(topk_ids_flat, cfg.num_experts, dtype=jnp.int32),
          axis=0,
      )
      group_offsets = jnp.cumulative_sum(
          group_sizes_global, include_initial=True
      )
      start = group_offsets[ep_idx * cfg.local_num_experts]
      end = group_offsets[(ep_idx + 1) * cfg.local_num_experts]
      count = jnp.minimum(end - start, buf_size)
      win_slots = jax.lax.dynamic_slice_in_dim(
          jnp.pad(slot_order_all, (0, buf_size)), start, buf_size, axis=0
      )
      valid = jnp.arange(buf_size, dtype=jnp.int32) < count
      tok_idx = jax.lax.stop_gradient(jnp.where(valid, win_slots // topk, 0))
      sorted_tokens = jnp.take(dispatched.tokens_ep, tok_idx, axis=0)
      mask_bcast = valid.reshape(-1, *([1] * (sorted_tokens.ndim - 1)))
      sorted_tokens = jnp.where(mask_bcast, sorted_tokens, 0)
      local_gs = jax.lax.dynamic_slice_in_dim(
          group_sizes_global,
          ep_idx * cfg.local_num_experts,
          cfg.local_num_experts,
          axis=0,
      )
      prefix_excl = jnp.cumsum(local_gs) - local_gs
      local_gs = jnp.minimum(local_gs, jnp.maximum(0, buf_size - prefix_excl))
      w_flat = jnp.ravel(dispatched.topk_weights_ep).astype(jnp.float32)
      w_rows = jnp.where(
          valid, jnp.take(w_flat, jnp.where(valid, win_slots, 0), axis=0), 0.0
      )

    if return_weights:
      act, gate_up, wi_ret = moe_kernels.wi_gmm_silu(
          sorted_tokens, wi, local_gs, cfg, return_rhs=True
      )
      weighted_down, wo_ret = moe_kernels.wo_weighted_gmm(
          jax.lax.stop_gradient(act),
          gate_up,
          w_rows,
          wo,
          local_gs,
          cfg,
          return_rhs=True,
          out_is_3d=use_3d,
          forward_tie=forward_tie,
      )
    else:
      act, gate_up = moe_kernels.wi_gmm_silu(
          sorted_tokens, wi, local_gs, cfg, return_rhs=False
      )
      wi_ret = None
      weighted_down = moe_kernels.wo_weighted_gmm(
          jax.lax.stop_gradient(act),
          gate_up,
          w_rows,
          wo,
          local_gs,
          cfg,
          return_rhs=False,
          out_is_3d=use_3d,
          forward_tie=forward_tie,
      )
      wo_ret = None

    if cfg.use_pallas_kernels:
      partial_ep = ragged_sort_tc.ring_ragged_unsort_tc(
          weighted_down,
          routing,
          cfg.num_experts_per_tok,
          jnp.ravel(dispatched.topk_weights_ep),
          gather_block_size=cfg.tc_gather_block_size,
          mask_padding=True,
          prescaled=True,
          out_3d=use_3d,
      )
    else:
      num_tokens_ep = dispatched.tokens_ep.shape[0]
      safe_dst = jnp.where(valid, tok_idx, num_tokens_ep)
      padded_out = jnp.zeros(
          (num_tokens_ep + 1, *weighted_down.shape[1:]),
          dtype=weighted_down.dtype,
      )
      partial_ep = padded_out.at[safe_dst].add(
          jnp.where(mask_bcast, weighted_down, 0)
      )[:num_tokens_ep]

    if return_weights:
      if forward_tie is None:
        return partial_ep, wi_ret, wo_ret
      return partial_ep, forward_tie, wi_ret, wo_ret
    if forward_tie is not None:
      return partial_ep, forward_tie
    return partial_ep

  sorted_tokens, group_sizes_global, revert_indices, argsort_indices = (
      ragged_sort.ring_ragged_sort(
          dispatched.tokens_ep,
          dispatched.topk_ids_ep,
          cfg.num_experts,
          cfg.num_experts_per_tok,
          cfg.ep_axis_names,
          cfg.ep_size,
          buffer_size=buf_size,
          enforce_gather_fallback=fallback,
          enforce_gather_reduce_fallback=fallback,
          use_single_sparsecore=cfg.ragged_single_sc,
          index_math=cfg.moe_routing_index_math,
      )
  )

  ep_idx = ops.ep_axis_index(cfg)
  local_gs = jax.lax.dynamic_slice_in_dim(
      group_sizes_global,
      ep_idx * cfg.local_num_experts,
      cfg.local_num_experts,
      axis=0,
  )
  prefix_excl = jnp.cumsum(local_gs) - local_gs
  remaining = jnp.maximum(0, buf_size - prefix_excl)
  local_gs = jnp.minimum(local_gs, remaining)

  if return_weights:
    act, gate_up, wi_ret = moe_kernels.wi_gmm_silu(
        sorted_tokens, wi, local_gs, cfg, return_rhs=True
    )
  else:
    act, gate_up = moe_kernels.wi_gmm_silu(
        sorted_tokens, wi, local_gs, cfg, return_rhs=False
    )
    wi_ret = None

  act_sg = jax.lax.stop_gradient(act)
  down_proj_fn, down_proj_bwd_fn = moe_kernels.make_wo_callbacks(cfg)

  unsort_ret = ragged_sort.ring_ragged_unsort_down_projection(
      act_sg,
      wo,
      down_proj_fn,
      (local_gs,),
      group_sizes_global,
      revert_indices,
      cfg.num_experts_per_tok,
      cfg.local_num_experts,
      cfg.ep_axis_names,
      topk_weights=jnp.ravel(dispatched.topk_weights_ep),
      topk_argsort_indices=argsort_indices,
      forward_tie=forward_tie,
      enforce_gather_fallback=fallback,
      enforce_gather_reduce_fallback=fallback,
      use_single_sparsecore=cfg.ragged_single_sc,
      index_math=cfg.moe_routing_index_math,
      dswiglu_gate_up=gate_up,
      down_projection_bwd_dswiglu=down_proj_bwd_fn,
      spread_masked_indices=cfg.ragged_unsort_spread_masked_indices,
      return_wo=return_weights,
  )

  if return_weights:
    if forward_tie is None:
      partial_ep, wo_ret = unsort_ret
      return partial_ep, wi_ret, wo_ret
    partial_ep, tie_out, wo_ret = unsort_ret
    return partial_ep, tie_out, wi_ret, wo_ret

  return unsort_ret


def moe_expert_gmm_and_unsort_from_routing(
    tokens_ep: ops.Fp8DispatchedTokens,
    w_rows_f32: jax.Array,
    saved_routing: ops.MoeSavedRouting,
    wi: moe_kernels.Fp8SteWeight,
    wo: moe_kernels.Fp8SteWeight,
    cfg: QwenConfig,
    *,
    return_weights: bool = False,
    wrap_partial_fp8_bwd: bool = False,
    sublayer_idx: int = 0,
):
  """Executes routed MoE GMM+unsort directly from FWD `saved_routing` (zero sort/metadata)."""
  moe_kernels.assert_global_mixed_precision_contract(cfg)
  buf_size = saved_routing.slot_order.shape[0]
  token_ids_sorted = jax.lax.stop_gradient(
      saved_routing.slot_order // cfg.num_experts_per_tok
  )
  if cfg.moe_use_forked_3d_gmmv2:
    return moe_kernels.routed_moe_forked_3d_fp8(
        tokens_ep,
        w_rows_f32,
        wi,
        wo,
        saved_routing.local_gs,
        token_ids_sorted,
        saved_routing.count,
        saved_routing.reduce_meta,
        cfg,
        buf_size=buf_size,
        return_rhs=return_weights,
        wrap_partial_fp8_bwd=wrap_partial_fp8_bwd,
        sublayer_idx=sublayer_idx,
    )
  return moe_kernels.routed_moe_3d_fp8_silu(
      tokens_ep,
      w_rows_f32,
      wi,
      wo,
      saved_routing.local_gs,
      token_ids_sorted,
      saved_routing.count,
      saved_routing.reduce_meta,
      cfg,
      buf_size=buf_size,
      return_rhs=return_weights,
  )


def unsort_moe_row_weights(
    g_w_rows: jax.Array,
    saved_routing: ops.MoeSavedRouting,
    num_tokens_ep: int,
    cfg: QwenConfig,
) -> jax.Array:
  """Maps `[buf_size]` buffer-row weight gradients back to `[num_tokens_ep, K]`."""
  return ragged_sort_tc.tc_unsort_row_weights(
      g_w_rows,
      saved_routing.slot_order,
      saved_routing.count,
      saved_routing.reduce_meta,
      num_out_tokens=num_tokens_ep,
      top_k=cfg.num_experts_per_tok,
      out_dtype=saved_routing.w_rows.dtype,
  )


def moe_combine_and_add_base(
    partial_ep: jax.Array | ops.MoePartialTokens,
    base_residual: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Combines partial EP outputs on SC0 (BF16+ enforced) and adds `base_residual`."""
  moe_kernels.assert_global_mixed_precision_contract(cfg)
  y_arr = (
      partial_ep.y_ep
      if isinstance(partial_ep, ops.MoePartialTokens)
      else partial_ep
  )
  ops.assert_bf16_or_fp32_reduction(y_arr, "moe_combine_and_add_base")
  combined_flat = ops.ep_token_combine(partial_ep, cfg)
  return base_residual + combined_flat.reshape(base_residual.shape)
