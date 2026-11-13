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

"""Shared and native `jax.lax.ragged_dot` routed MoE experts for Flint."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_gmm_v2
from maxtext.experimental.flint40b import flint_mlp
from maxtext.experimental.flint40b import flint_moe_kernels
from maxtext.experimental.flint40b import flint_router
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops


PreGatheredRouted = jax.Array | tuple[jax.Array, jax.Array]


def gather_routed_gate_up(
    routed_shard: flint_types.FlintRoutedExpertsWeights[jax.Array],
    cfg: flint_config.FlintConfig,
) -> PreGatheredRouted:
  """All-gathers `routed_shard.gate_up_proj` across `('fsdp', 'cp')` (`SC1`)."""
  gu_axis = 0 if cfg.kernels.shard_routed_on_expert_axis else 1
  if cfg.kernels.use_fp8:
    return ops.prefetch_routed_weight_fp8(
        routed_shard.gate_up_proj,
        cfg,
        sharded_axis=gu_axis,
        role="gate_up",
        rhs_is_transposed=False,
    )
  return ops.gather_routed_weight(routed_shard.gate_up_proj, cfg, sharded_axis=gu_axis, role="gate_up")


def gather_routed_down(
    routed_shard: flint_types.FlintRoutedExpertsWeights[jax.Array],
    cfg: flint_config.FlintConfig,
) -> PreGatheredRouted:
  """All-gathers `routed_shard.down_proj` across `('fsdp', 'cp')` (`SC0`)."""
  if cfg.kernels.transpose_routed_down_ag:
    down_shard_t = jnp.swapaxes(routed_shard.down_proj, 1, 2)
    if cfg.kernels.use_fp8:
      return ops.prefetch_routed_weight_fp8(
          down_shard_t,
          cfg,
          sharded_axis=1,
          role="down",
          rhs_is_transposed=True,
      )
    return ops.gather_routed_weight(down_shard_t, cfg, sharded_axis=1, role="down")
  down_axis = 0 if cfg.kernels.shard_routed_on_expert_axis else 2
  if cfg.kernels.use_fp8:
    return ops.prefetch_routed_weight_fp8(
        routed_shard.down_proj,
        cfg,
        sharded_axis=down_axis,
        role="down",
        rhs_is_transposed=False,
    )
  return ops.gather_routed_weight(routed_shard.down_proj, cfg, sharded_axis=down_axis, role="down")


def _ragged_dot_fp8_xla(
    lhs_q: jax.Array,
    rhs_q: jax.Array,
    gs_i32: jax.Array,
    *,
    transpose_rhs: bool,
) -> jax.Array:
  """Runs 2D x 3D grouped matmul on FP8 operands (`f32` upcast on CPU)."""
  is_cpu = jax.default_backend() == "cpu"
  lhs_in = lhs_q.astype(jnp.float32) if is_cpu else lhs_q
  rhs_in = rhs_q.astype(jnp.float32) if is_cpu else rhs_q
  prec = jax.lax.Precision.HIGHEST if is_cpu else None
  if transpose_rhs:
    return jax.lax.ragged_dot_general(
        lhs=lhs_in,
        rhs=rhs_in,
        group_sizes=gs_i32,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((1,), (2,)), ((), ())),
            lhs_ragged_dimensions=(0,),
            rhs_group_dimensions=(0,),
        ),
        precision=prec,
        preferred_element_type=jnp.float32,
    )
  return jax.lax.ragged_dot(
      lhs=lhs_in,
      rhs=rhs_in,
      group_sizes=gs_i32,
      precision=prec,
      preferred_element_type=jnp.float32,
  )


def _tgmm_xla(
    lhs_in: jax.Array,
    rhs_in: jax.Array,
    gs_i32: jax.Array,
) -> jax.Array:
  """Runs grouped `lhs_in.T @ rhs_in -> [G, K, N]` (`f32` upcast on CPU)."""
  is_cpu = jax.default_backend() == "cpu"
  lhs_arg = lhs_in.astype(jnp.float32) if is_cpu else lhs_in
  rhs_arg = rhs_in.astype(jnp.float32) if is_cpu else rhs_in
  prec = jax.lax.Precision.HIGHEST if is_cpu else None
  return jax.lax.ragged_dot_general(
      lhs=lhs_arg,
      rhs=rhs_arg,
      group_sizes=gs_i32,
      ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
          dot_dimension_numbers=(((0,), (0,)), ((), ())),
          lhs_ragged_dimensions=(0,),
          rhs_group_dimensions=(),
      ),
      precision=prec,
      preferred_element_type=jnp.float32,
  )


def _row_group_scale(
    num_rows: int,
    gs_i32: jax.Array,
    w_scale: jax.Array,
) -> jax.Array:
  """Maps per-group `w_scale` (`f32[E, 1, C]`) to per-row `f32[num_rows, C]`."""
  cum_ends = jnp.cumsum(gs_i32)
  row_g = jnp.minimum(
      jnp.searchsorted(cum_ends, jnp.arange(num_rows, dtype=jnp.int32), side="right"),
      gs_i32.shape[0] - 1,
  )
  return w_scale[row_g, 0, :]


def _routed_ragged_dot_fp8(
    lhs: jax.Array,
    w_shard: jax.Array,
    q_w: jax.Array,
    w_scale: jax.Array,
    group_sizes: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    sharded_axis: int,
    role: str,
    rhs_is_transposed: bool,
) -> jax.Array:
  """Differentiable FP8 grouped matmul with BF16 SparseCore grad RS."""
  comp_dt = cfg.train.compute_jnp_dtype
  orig_lhs_dtype = lhs.dtype
  orig_shard_dtype = w_shard.dtype
  use_pallas = cfg.kernels.moe_impl == "pallas_gmm_v2"
  gmm_tm = cfg.kernels.gmm_tile_m
  tgmm_tm = cfg.kernels.tgmm_tile_m
  q_w_sg = jax.lax.stop_gradient(q_w)
  w_sc_sg = jax.lax.stop_gradient(w_scale)

  def _eval_fwd(
      lhs_arr: jax.Array,
      q_w_g: jax.Array,
      w_sc_g: jax.Array,
      gs_i32: jax.Array,
  ) -> jax.Array:
    if use_pallas:
      return flint_gmm_v2.gmm_v2(
          lhs_arr,
          q_w_g,
          gs_i32,
          preferred_element_type=comp_dt,
          transpose_rhs=rhs_is_transposed,
          rhs_scale=w_sc_g,
          rhs_scale_on_k=False,
          lhs_qtype=jnp.float8_e4m3fn,
          tile_m_override=gmm_tm,
      )
    q_lhs, lhs_scale = ops.quantize_fp8_axis(lhs_arr, reduce_axis=1, qtype=jnp.float8_e4m3fn)
    raw = _ragged_dot_fp8_xla(q_lhs, q_w_g, gs_i32, transpose_rhs=rhs_is_transposed)
    row_sc = _row_group_scale(lhs_arr.shape[0], gs_i32, w_sc_g)
    return ((raw * lhs_scale) * row_sc).astype(comp_dt)

  @jax.custom_vjp
  def _op(
      lhs_arr: jax.Array,
      w_s: jax.Array,
      q_w_g: jax.Array,
      w_sc_g: jax.Array,
      gs_arr: jax.Array,
  ) -> jax.Array:
    del w_s
    return _eval_fwd(lhs_arr, q_w_g, w_sc_g, gs_arr.astype(jnp.int32))

  def _fwd(
      lhs_arr: jax.Array,
      w_s: jax.Array,
      q_w_g: jax.Array,
      w_sc_g: jax.Array,
      gs_arr: jax.Array,
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    del w_s
    gs_i32 = gs_arr.astype(jnp.int32)
    out = _eval_fwd(lhs_arr, q_w_g, w_sc_g, gs_i32)
    return out, (lhs_arr, q_w_g, w_sc_g, gs_i32)

  def _bwd(
      residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
      do: jax.Array,
  ) -> tuple[jax.Array, jax.Array, None, None, None]:
    lhs_arr, q_w_g, w_sc_g, gs_i32 = residuals
    if use_pallas:
      dlhs = flint_gmm_v2.gmm_v2(
          do,
          q_w_g,
          gs_i32,
          preferred_element_type=orig_lhs_dtype,
          transpose_rhs=not rhs_is_transposed,
          rhs_scale=w_sc_g,
          rhs_scale_on_k=True,
          lhs_qtype=jnp.float8_e4m3fn,
          tile_m_override=gmm_tm,
      )
      if rhs_is_transposed:
        dw_full = flint_gmm_v2.tgmm_v2(
            do.astype(comp_dt),
            lhs_arr.astype(comp_dt),
            gs_i32,
            num_actual_groups=q_w_g.shape[0],
            preferred_element_type=comp_dt,
            tile_m_override=tgmm_tm,
        )
      else:
        dw_full = flint_gmm_v2.tgmm_v2(
            lhs_arr.astype(comp_dt),
            do.astype(comp_dt),
            gs_i32,
            num_actual_groups=q_w_g.shape[0],
            preferred_element_type=comp_dt,
            tile_m_override=tgmm_tm,
        )
    else:
      row_sc = _row_group_scale(do.shape[0], gs_i32, w_sc_g)
      do_scaled = (do.astype(jnp.float32) * row_sc).astype(comp_dt)
      q_do_dlhs, do_dlhs_scale = ops.quantize_fp8_axis(
          do_scaled,
          reduce_axis=1,
          qtype=jnp.float8_e4m3fn,
      )
      raw_dlhs = _ragged_dot_fp8_xla(q_do_dlhs, q_w_g, gs_i32, transpose_rhs=not rhs_is_transposed)
      dlhs = (raw_dlhs * do_dlhs_scale).astype(orig_lhs_dtype)

      if rhs_is_transposed:
        dw_full = _tgmm_xla(do.astype(comp_dt), lhs_arr.astype(comp_dt), gs_i32).astype(comp_dt)
      else:
        dw_full = _tgmm_xla(lhs_arr.astype(comp_dt), do.astype(comp_dt), gs_i32).astype(comp_dt)

    dw_shard = ops.reduce_scatter_routed_grad(
        dw_full,
        cfg,
        sharded_axis=sharded_axis,
        role=role,
        out_dtype=orig_shard_dtype,
    )
    return dlhs, dw_shard, None, None, None

  _op.defvjp(_fwd, _bwd)
  return _op(lhs, w_shard, q_w_sg, w_sc_sg, group_sizes)


def _pack_expert_assignments(
    ids_ep: jax.Array,
    expert_start: int | jax.Array,
    num_local_experts: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float | None,
    *,
    slot_major: bool = False,
    zero_scatter: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sorts `(token, slot)` assignments by local expert ID for `ragged_dot`.

  Args:
    ids_ep: Selected global expert IDs of shape `[num_ep_tokens, top_k]`.
    expert_start: First global expert index owned by this `'ep'` rank.
    num_local_experts: Number of experts `E_local = E // ep_size` on this rank.
    num_experts: Total global number of experts `E`.
    top_k: Number of experts selected per token `K`.
    capacity_factor: Optional capacity multiplier; if `None`, runs dropless.
    slot_major: When `True`, flattens `[top_k, num_ep_tokens]` (`ids_ep.T`) so
      input/output `[K, N, D]` broadcast and sum stay `(8, 128)`-tile aligned.
    zero_scatter: When `True`, computes `raw_counts` from monotonic `sorted_eid`
      via `jnp.searchsorted` instead of an HBM `jnp.bincount` scatter.

  Returns:
    Tuple `(perm, sorted_valid, group_sizes)` where `perm` (`int32[N_ep * K]`)
    permutes flattened assignments so local experts `0..E_local-1` are packed
    contiguously at the front and invalid/dropped assignments sit at the tail,
    `sorted_valid` (`bool[N_ep * K]`) marks active assignments, and
    `group_sizes` (`int32[E_local]`) holds per-expert row counts.
  """
  num_ep_tokens = ids_ep.shape[0]
  total_assign = num_ep_tokens * top_k
  raw_ids = jnp.swapaxes(ids_ep, 0, 1) if slot_major else ids_ep
  local_eid = raw_ids.astype(jnp.int32).reshape(-1) - expert_start
  valid = (local_eid >= 0) & (local_eid < num_local_experts)
  safe_key = jnp.where(valid, local_eid, num_local_experts)
  perm = jnp.argsort(safe_key, stable=True).astype(jnp.int32)
  sorted_eid = safe_key[perm]
  if zero_scatter:
    edges = jnp.searchsorted(
        sorted_eid,
        jnp.arange(num_local_experts + 1, dtype=jnp.int32),
        side="left",
    )
    raw_counts = (edges[1:] - edges[:-1]).astype(jnp.int32)
  else:
    raw_counts = jnp.bincount(sorted_eid, length=num_local_experts + 1)[:num_local_experts].astype(jnp.int32)

  if capacity_factor is None:
    sorted_valid = sorted_eid < num_local_experts
    return perm, sorted_valid, raw_counts

  cap = max(
      1,
      int(math.ceil(capacity_factor * num_ep_tokens * top_k / num_experts)),
  )
  group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(raw_counts)[:-1]])
  safe_eid_clamped = jnp.minimum(sorted_eid, num_local_experts - 1)
  pos_in_expert = jnp.arange(total_assign, dtype=jnp.int32) - group_starts[safe_eid_clamped]
  keep = (sorted_eid < num_local_experts) & (pos_in_expert < cap)
  key2 = jnp.where(keep, sorted_eid, num_local_experts)
  perm2 = jnp.argsort(key2, stable=True).astype(jnp.int32)
  perm = perm[perm2]
  sorted_valid = keep[perm2]
  group_sizes = jnp.minimum(raw_counts, cap)
  return perm, sorted_valid, group_sizes


def flint_routed_experts_forward(
    x_local: jax.Array,
    routed_shard: flint_types.FlintRoutedExpertsWeights[jax.Array],
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    cfg: flint_config.FlintConfig,
    *,
    pre_gathered_gate_up: PreGatheredRouted | None = None,
    pre_gathered_down: PreGatheredRouted | None = None,
) -> jax.Array:
  """Computes routed SwiGLU experts via native `ragged_dot` or `pallas_gmm_v2`.

  Executes inside `jax.shard_map(..., check_vma=False)` over
  `('fsdp', 'cp', 'ep')`. Supports `ep_size == 1` and `ep_size > 1` using
  `all_gather` of tokens/assignments over `'ep'` followed by local expert
  filtering, grouped matmul, weighted unpermutation, and `psum_scatter` over
  `'ep'`.

  Args:
    x_local: Local token activations of shape `[..., hidden_size]`.
    routed_shard: Local `FlintRoutedExpertsWeights` shards.
    topk_ids: Selected expert indices of shape `[..., top_k]`.
    topk_weights: Normalized routing weights of shape `[..., top_k]`.
    cfg: Validated `FlintConfig`.
    pre_gathered_gate_up: Optional prefetched `gate_up_proj` local experts.
    pre_gathered_down: Optional prefetched `down_proj` local experts.

  Returns:
    Local routed-expert output tensor with shape `x_local.shape` and dtype
    `cfg.train.compute_jnp_dtype`.
  """
  num_experts = cfg.model.num_experts
  ep_size = cfg.sharding.ep_size
  num_local_experts = num_experts // ep_size
  top_k = cfg.model.num_experts_per_tok
  moe_inter = cfg.model.moe_intermediate_size
  hidden_size = cfg.model.hidden_size
  comp_dt = cfg.train.compute_jnp_dtype
  prec = jax.lax.Precision.HIGHEST if comp_dt == jnp.float32 else None
  all_valid = ep_size == 1 and cfg.kernels.capacity_factor is None
  if cfg.kernels.use_sc_moe_gather_reduce and not all_valid and flint_moe_kernels.is_tpu_device():
    raise ValueError("use_sc_moe_gather_reduce=True on TPU requires ep_size == 1 and" " capacity_factor is None.")
  use_direct_sc = cfg.kernels.use_sc_moe_gather_reduce and all_valid
  slot_major = False if use_direct_sc else cfg.kernels.moe_slot_major_pack
  compact_maps = cfg.kernels.compact_router_maps

  gate_up_local = pre_gathered_gate_up if pre_gathered_gate_up is not None else gather_routed_gate_up(routed_shard, cfg)
  down_local = pre_gathered_down if pre_gathered_down is not None else gather_routed_down(routed_shard, cfg)

  x_flat = x_local.astype(comp_dt).reshape(-1, hidden_size)
  ids_flat = topk_ids.reshape(-1, top_k)
  w_flat = topk_weights.reshape(-1, top_k)

  if ep_size > 1:
    x_ep = jax.lax.all_gather(x_flat, "ep", axis=0, tiled=True)
    ids_ep = jax.lax.all_gather(ids_flat, "ep", axis=0, tiled=True)
    w_ep = jax.lax.all_gather(w_flat, "ep", axis=0, tiled=True)
    expert_start: int | jax.Array = jax.lax.axis_index("ep") * num_local_experts
  else:
    x_ep = x_flat
    ids_ep = ids_flat
    w_ep = w_flat
    expert_start = 0

  if cfg.kernels.use_sc_counting_sort:
    if not all_valid or slot_major:
      if flint_moe_kernels.is_tpu_device():
        raise ValueError(
            "use_sc_counting_sort=True on TPU requires ep_size == 1," " capacity_factor is None, and slot_major == False."
        )
      sc_sort = None
    else:
      sc_sort = flint_moe_kernels.counting_sort_permute_sc(ids_ep.astype(jnp.int32).reshape(-1), num_local_experts)
      if sc_sort is None and flint_moe_kernels.is_tpu_device():
        raise ValueError(
            "use_sc_counting_sort=True failed TPU preconditions for"
            f" ids_ep={ids_ep.shape}, num_local_experts={num_local_experts}."
        )
  else:
    sc_sort = None
  if sc_sort is not None:
    perm, inv_perm, group_sizes = sc_sort
    sorted_valid: jax.Array | None = None
  else:
    perm, sorted_valid_raw, group_sizes = _pack_expert_assignments(
        ids_ep=ids_ep,
        expert_start=expert_start,
        num_local_experts=num_local_experts,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=cfg.kernels.capacity_factor,
        slot_major=slot_major,
        zero_scatter=cfg.kernels.moe_zero_scatter_router,
    )
    inv_perm = jnp.argsort(perm).astype(jnp.int32)
    sorted_valid = None if all_valid else sorted_valid_raw

  perm = flint_moe_kernels.checkpoint_router_map(perm, compact=compact_maps)
  inv_perm = flint_moe_kernels.checkpoint_router_map(inv_perm, compact=compact_maps)
  group_sizes = flint_moe_kernels.checkpoint_router_map(group_sizes, compact=compact_maps)
  if sorted_valid is not None:
    sorted_valid = flint_moe_kernels.checkpoint_router_map(sorted_valid, compact=compact_maps)

  num_ep_tokens = x_ep.shape[0]
  zero_comp = jnp.zeros((), dtype=comp_dt)
  if use_direct_sc:
    sorted_x = flint_moe_kernels.route_tokens_direct(x_ep, perm, inv_perm, top_k, True)
  elif cfg.kernels.moe_bijective_permute:
    if slot_major:
      x_rep = jnp.broadcast_to(x_ep[None, :, :], (top_k, num_ep_tokens, hidden_size)).reshape(
          top_k * num_ep_tokens, hidden_size
      )
    else:
      x_rep = jnp.broadcast_to(x_ep[:, None, :], (num_ep_tokens, top_k, hidden_size)).reshape(
          num_ep_tokens * top_k, hidden_size
      )
    sorted_x = ops.bijective_permute(x_rep, perm, inv_perm)
    if sorted_valid is not None:
      sorted_x = jnp.where(sorted_valid[:, None], sorted_x, zero_comp)
  else:
    ar_tok = jnp.arange(num_ep_tokens, dtype=jnp.int32)
    token_idx = jnp.tile(ar_tok, top_k) if slot_major else jnp.repeat(ar_tok, top_k)
    sorted_x = x_ep[token_idx[perm]]
    if sorted_valid is not None:
      sorted_x = jnp.where(sorted_valid[:, None], sorted_x, zero_comp)
  sorted_x = jax.ad_checkpoint.checkpoint_name(sorted_x, "moe_sorted_x")

  if cfg.kernels.use_fp8:
    assert isinstance(gate_up_local, tuple)
    gu_axis = 0 if cfg.kernels.shard_routed_on_expert_axis else 1
    gate_up_out = _routed_ragged_dot_fp8(
        sorted_x,
        routed_shard.gate_up_proj,
        gate_up_local[0],
        gate_up_local[1],
        group_sizes,
        cfg,
        sharded_axis=gu_axis,
        role="gate_up",
        rhs_is_transposed=False,
    )
  elif cfg.kernels.moe_impl == "pallas_gmm_v2":
    assert isinstance(gate_up_local, jax.Array)
    gate_up_out = flint_gmm_v2.pallas_ragged_dot(
        sorted_x,
        gate_up_local,
        group_sizes,
        comp_dt,
        cfg.kernels.gmm_tile_m,
        cfg.kernels.tgmm_tile_m,
    )
  else:
    assert isinstance(gate_up_local, jax.Array)
    gate_up_out = jax.lax.ragged_dot(
        lhs=sorted_x,
        rhs=gate_up_local,
        group_sizes=group_sizes,
        precision=prec,
        preferred_element_type=comp_dt,
    )
  if sorted_valid is not None:
    gate_up_out = jnp.where(sorted_valid[:, None], gate_up_out, zero_comp)
  gate_up_out = jax.ad_checkpoint.checkpoint_name(gate_up_out, "moe_gate_up")
  gate = gate_up_out[:, :moe_inter]
  up = gate_up_out[:, moe_inter:]
  act = jax.nn.silu(gate) * up
  if sorted_valid is not None:
    act = jnp.where(sorted_valid[:, None], act, zero_comp)

  if cfg.kernels.use_fp8:
    assert isinstance(down_local, tuple)
    if cfg.kernels.transpose_routed_down_ag:
      down_w_shard = jnp.swapaxes(routed_shard.down_proj, 1, 2)
      down_axis = 1
      down_transposed = True
    else:
      down_w_shard = routed_shard.down_proj
      down_axis = 0 if cfg.kernels.shard_routed_on_expert_axis else 2
      down_transposed = False
    sorted_out = _routed_ragged_dot_fp8(
        act,
        down_w_shard,
        down_local[0],
        down_local[1],
        group_sizes,
        cfg,
        sharded_axis=down_axis,
        role="down",
        rhs_is_transposed=down_transposed,
    )
  elif cfg.kernels.moe_impl == "pallas_gmm_v2":
    assert isinstance(down_local, jax.Array)
    if cfg.kernels.transpose_routed_down_ag:
      sorted_out = flint_gmm_v2.pallas_ragged_dot_transposed_rhs(
          act,
          down_local,
          group_sizes,
          comp_dt,
          cfg.kernels.gmm_tile_m,
          cfg.kernels.tgmm_tile_m,
      )
    else:
      sorted_out = flint_gmm_v2.pallas_ragged_dot(
          act,
          down_local,
          group_sizes,
          comp_dt,
          cfg.kernels.gmm_tile_m,
          cfg.kernels.tgmm_tile_m,
      )
  elif cfg.kernels.transpose_routed_down_ag:
    assert isinstance(down_local, jax.Array)
    sorted_out = jax.lax.ragged_dot_general(
        lhs=act,
        rhs=down_local,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((1,), (2,)), ((), ())),
            lhs_ragged_dimensions=(0,),
            rhs_group_dimensions=(0,),
        ),
        precision=prec,
        preferred_element_type=comp_dt,
    )
  else:
    assert isinstance(down_local, jax.Array)
    sorted_out = jax.lax.ragged_dot(
        lhs=act,
        rhs=down_local,
        group_sizes=group_sizes,
        precision=prec,
        preferred_element_type=comp_dt,
    )
  if sorted_valid is not None:
    sorted_out = jnp.where(sorted_valid[:, None], sorted_out, zero_comp)
  sorted_out = jax.ad_checkpoint.checkpoint_name(sorted_out, "moe_down_out")

  w_1d = (jnp.swapaxes(w_ep, 0, 1) if slot_major else w_ep).reshape(-1)
  if use_direct_sc:
    out_ep = flint_moe_kernels.weighted_combine_direct(sorted_out, perm, inv_perm, w_1d, top_k, True)
  else:
    if cfg.kernels.moe_bijective_permute:
      w_perm = ops.bijective_permute(w_1d, perm, inv_perm)
    else:
      w_perm = w_1d[perm]
    if sorted_valid is not None:
      w_perm = jnp.where(sorted_valid, w_perm, jnp.zeros((), dtype=w_ep.dtype))
    sorted_w = w_perm.astype(comp_dt)[:, None]
    weighted = sorted_out * sorted_w
    if cfg.kernels.moe_bijective_permute:
      unperm = ops.bijective_permute(weighted, inv_perm, perm)
    else:
      unperm = weighted[inv_perm]
    if slot_major:
      out_ep = unperm.reshape(top_k, num_ep_tokens, hidden_size).sum(axis=0)
    else:
      out_ep = unperm.reshape(num_ep_tokens, top_k, hidden_size).sum(axis=1)
  if ep_size > 1:
    out_flat = jax.lax.psum_scatter(out_ep, "ep", scatter_dimension=0, tiled=True)
  else:
    out_flat = out_ep
  return out_flat.reshape(x_local.shape)


def flint_moe_forward(
    x_local: jax.Array,
    moe_shard: flint_types.FlintMoEWeights[jax.Array],
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    pre_gathered_gate_up: PreGatheredRouted | None = None,
    pre_gathered_down: PreGatheredRouted | None = None,
    compute_aux_stats: bool = True,
) -> tuple[jax.Array, flint_types.FlintMoeAux]:
  """Runs the complete Flint MoE block (`router + shared_expert + routed`).

  Args:
    x_local: Local token activations of shape `[..., hidden_size]`.
    moe_shard: Local `FlintMoEWeights` parameter shards.
    loss_mask_local: Optional local token validity mask of shape
      `x_local.shape[:-1]`, or `None` when all tokens are valid.
    cfg: Validated `FlintConfig`.
    pre_gathered_gate_up: Optional prefetched `gate_up_proj` local experts.
    pre_gathered_down: Optional prefetched `down_proj` local experts.
    compute_aux_stats: Whether to compute `expert_counts` and `prob_sum`.

  Returns:
    Tuple `(moe_out, aux)` where `moe_out` (`x_local.shape`,
    `cfg.train.compute_jnp_dtype`) equals `routed_out + shared_out` and `aux` is
    the per-layer `FlintMoeAux`.
  """
  aux = flint_router.flint_router_forward(
      x_local,
      moe_shard.router,
      loss_mask_local,
      cfg,
      compute_aux_stats=compute_aux_stats,
  )
  shared_out = flint_mlp.flint_mlp_forward(x_local, moe_shard.shared_expert, cfg)
  routed_out = flint_routed_experts_forward(
      x_local,
      moe_shard.routed_experts,
      aux.topk_ids,
      aux.topk_weights,
      cfg,
      pre_gathered_gate_up=pre_gathered_gate_up,
      pre_gathered_down=pre_gathered_down,
  )
  return routed_out + shared_out, aux
