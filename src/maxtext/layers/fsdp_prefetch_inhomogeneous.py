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

"""FSDP weight prefetching + deferred grad RS for inhomogeneous scanned blocks (e.g. Qwen3.5)."""

# pylint: disable=protected-access

import functools
import math
import re

from flax import nnx
import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch
from maxtext.layers.moe import _resolve_ep_axis_name, combine_ep_ubatch_bwd_fn, combine_ep_ubatch_fn
from maxtext.utils import max_logging, sharding


def _batchsplit_shardings_and_ep(mesh, cfg):
  rules = sharding.get_logical_axis_rules() or cfg.logical_axis_rules
  act_pspec = sharding.logical_to_mesh_axes(
      ("activation_batch", "activation_norm_length", None), mesh=mesh, rules=rules
  )
  act_sharding = jax.sharding.NamedSharding(mesh, act_pspec)
  ep_axis = _resolve_ep_axis_name(mesh, cfg)
  ep_axes = ep_axis if isinstance(ep_axis, tuple) else (ep_axis,)
  num_ep = math.prod(mesh.shape.get(a, 1) for a in ep_axes)
  return act_sharding, num_ep


def _maybe_constrain(x, s, shard_mode):
  return jax.lax.with_sharding_constraint(x, s) if shard_mode != ShardMode.EXPLICIT else x


def _combine_add_residual(base_mb1, unperm_mb1, mesh, cfg, act_sharding, shard_mode):
  routed_mb1 = combine_ep_ubatch_fn(unperm_mb1, mesh, cfg)
  if getattr(cfg, "float32_weight_sum", False):
    out = (base_mb1.astype(jnp.float32) + routed_mb1.astype(jnp.float32)).astype(cfg.dtype)
  else:
    out = base_mb1 + routed_mb1
  return _maybe_constrain(out, act_sharding, shard_mode)


def _index_at(tree, idx):
  """Layer `idx` (a traced index) of a layer-stacked pytree."""
  return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False), tree)


_SKIP_TRIP_COUNT_ONE = {"skip-simplify-while-loops_trip-count-one": "true"}


def _sublayer_scan(body, carry, xs, reverse=False, unroll=1):
  """`jax.lax.scan` over sub-layers that XLA keeps as a while-loop, even with a single trip."""
  with xla_metadata.set_xla_metadata(**_SKIP_TRIP_COUNT_ONE):
    if unroll == 1:
      return jax.lax.scan(body, carry, xs, reverse=reverse)
    if unroll != 2:
      raise ValueError(f"unroll must be 1 or 2, got {unroll}.")
    length = jax.tree_util.tree_leaves(xs)[0].shape[0]
    trips, rem = divmod(length, 2)
    split = rem if reverse else 2 * trips
    head = jax.tree.map(lambda x: x[:split], xs)
    rest = jax.tree.map(lambda x: x[split:], xs)
    xs_paired, xs_odd = (rest, head) if reverse else (head, rest)

    def two_steps(carry, xs2):
      first, second = (1, 0) if reverse else (0, 1)
      carry, y_a = body(carry, jax.tree.map(lambda x: x[first], xs2))
      carry, y_b = body(carry, jax.tree.map(lambda x: x[second], xs2))
      ys2 = (y_b, y_a) if reverse else (y_a, y_b)
      return carry, jax.tree.map(lambda a, b: jnp.stack((a, b)), *ys2)

    xs_paired = jax.tree.map(lambda x: x.reshape(trips, 2, *x.shape[1:]), xs_paired)
    carry, ys = jax.lax.scan(two_steps, carry, xs_paired, reverse=reverse)
    ys = jax.tree.map(lambda y: y.reshape(2 * trips, *y.shape[2:]), ys)
    if rem:
      carry, ys_odd = jax.lax.scan(body, carry, xs_odd, reverse=reverse)
      ys = _concat0(ys_odd, ys) if reverse else _concat0(ys, ys_odd)
    return carry, ys


def _concat0(*trees):
  """Concatenates layer-stacked pytrees along the leading axis."""
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def _setup_groups(graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs):
  """`_prefetch_setup` for each sub-layer group; returns per-group tuples (gather_fn, rs_fn, ...)."""
  setups = [
      fsdp_prefetch._prefetch_setup(
          g, tuple(x[:, 0] for x in p), tuple(x[:, 0] for x in s), diff_args_leaves, diff_kwargs_leaves, ctx
      )
      for g, p, s, ctx in zip(graphdefs, params_leaves, state_leaves, ctxs)
  ]
  _, _, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns = zip(*setups)
  params = [jax.tree_util.tree_unflatten(ctx.params_treedef, p) for p, ctx in zip(params_leaves, ctxs)]
  state = [jax.tree_util.tree_unflatten(ctx.state_treedef, s) for s, ctx in zip(state_leaves, ctxs)]
  return params, state, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns


def _offload_saved_group(saved_group, layout, shard_mode):
  if layout is None:
    return saved_group
  return tuple(
      fsdp_prefetch._maybe_offload_to_host(sv, off, shard_mode)
      for sv, off in zip(saved_group, layout.offload_mask)
  )


def _onload_saved_group(saved_group, layout, shard_mode):
  if layout is None:
    return saved_group
  return tuple(
      fsdp_prefetch._maybe_onload_to_device(sv, off, shard_mode)
      for sv, off in zip(saved_group, layout.offload_mask)
  )


def _init_bwd_layouts(num_groups, ys, params, state, gather, layout_fns, offload_y, shard_mode):
  layouts = {}
  for i in range(num_groups):
    y_sample = fsdp_prefetch._maybe_onload_to_device(
        jax.tree.map(lambda x: x[0, 0], ys[i]), offload_y, shard_mode
    )
    w_sample = jax.eval_shape(gather[i], jax.tree.map(lambda x: x[0, 0], params[i]))
    w_dummy = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), w_sample)
    s_sample = jax.tree.map(lambda x: x[0, 0], state[i])
    layouts[i] = layout_fns[i](y_sample, w_dummy, s_sample)
  return layouts


def _sublayer_schedule(sizes):
  """Returns list of `(group_idx, local_idx)` pairs for all sub-layers in one block."""
  return [(gi, k) for gi, n in enumerate(sizes) for k in range(n)]


def _compute_flat_g8_absmax(flat_params, sched, ctxs):
  """Computes hoisted whole-stack per-channel absmax outside `jax.lax.scan` (1 whole-stack AG, 0 inside scan)."""
  out = []
  for j, (gi, _) in enumerate(sched):
    ctx = ctxs[gi]
    p_leaves = jax.tree_util.tree_leaves(flat_params[j])
    out.append(
        fsdp_prefetch.compute_stacked_gather8_absmax(
            p_leaves, ctx.layer_shardings, ctx.mesh, ctx.dtype, ctx.fp8_gather8
        )
    )
  return tuple(out)


def _setup_flat_sublayers(graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs, sizes):
  """Sets up per-group functions, plus a dedicated j=0 deferred setup when `defer_grad_reduce_scatter=True`."""
  del sizes
  defer_j0 = bool(ctxs[0].defer_grad_reduce_scatter)
  immed_ctxs = tuple(c._replace(defer_grad_reduce_scatter=False, defer_host_offload=False) for c in ctxs)
  params, state, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns = _setup_groups(
      graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, immed_ctxs
  )
  j0_setup = None
  if defer_j0:
    j0_ctx = ctxs[0]._replace(defer_grad_reduce_scatter=True, defer_host_offload=False)
    j0_setup = fsdp_prefetch._prefetch_setup(
        graphdefs[0],
        tuple(x[:, 0] for x in params_leaves[0]),
        tuple(x[:, 0] for x in state_leaves[0]),
        diff_args_leaves,
        diff_kwargs_leaves,
        j0_ctx,
    )
  return params, state, gather, rs_fns, layout_fns, fwd_layers, layer_vjp_fns, j0_setup


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7, 8))
def _custom_vjp_flat_prefetch_pipeline(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  out, _ = _custom_vjp_flat_prefetch_pipeline_fwd(
      graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
  )
  return out


def _custom_vjp_flat_prefetch_pipeline_fwd(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  """Flat-unrolled inhomogeneous block scan (`whiles=0` inside the outer block scan, 0 exposed HOST_DMA!)."""
  sched = _sublayer_schedule(sizes)
  m_sub = len(sched)
  offload_y = ctxs[0].offload_layer_input
  shard_mode = ctxs[0].shard_mode
  params, state, gather, _, layout_fns, fwd_layers, _, j0_setup = _setup_flat_sublayers(
      graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs, sizes
  )
  layouts = {}
  j0_layout_box = []

  def fwd_sub(j, gi, y, w, layer_state):
    if j == 0 and j0_setup is not None:
      _, _, _, _, j0_layout_fn, j0_fwd, _ = j0_setup
      if not j0_layout_box:
        j0_layout_box.append(j0_layout_fn(y, w, layer_state))
      return j0_fwd(y, w, layer_state, j0_layout_box[0])
    if gi not in layouts:
      layouts[gi] = layout_fns[gi](y, w, layer_state)
    return fwd_layers[gi](y, w, layer_state, layouts[gi])

  def gather_for(j, gi):
    if j == 0 and j0_setup is not None:
      return j0_setup[2]
    return gather[gi]

  # Pre-slice per-sublayer [length, ...] trees and hoist gather8 absmax outside the 23-block scan!
  flat_params = tuple(jax.tree.map(lambda x, k=k: x[:, k], params[gi]) for gi, k in sched)
  flat_state = tuple(jax.tree.map(lambda x, k=k: x[:, k], state[gi]) for gi, k in sched)
  flat_g8_absmax = _compute_flat_g8_absmax(flat_params, sched, ctxs)
  batch_split = bool(ctxs[0].batch_split)

  mesh = ctxs[0].mesh
  cfg = ctxs[0].config
  act_sharding, num_ep = _batchsplit_shardings_and_ep(mesh, cfg)

  if batch_split:

    def body_bs(carry, xs):
      (x_mb0, unperm_mb1, base_mb1), w0_cur = carry
      b, b_flat_state, b_flat_params, b_flat_g8 = xs
      ys_flat = []
      saved_flat = []
      w0_next = None

      for j, (gi, _) in enumerate(sched):
        if j == 0:
          w = w0_cur
        else:
          (x_mb0, unperm_mb1, base_mb1), p_j, g8_j = jax.lax.optimization_barrier(
              ((x_mb0, unperm_mb1, base_mb1), b_flat_params[j], b_flat_g8[j])
          )
          w = gather_for(j, gi)(p_j, g8_absmax=g8_j)

        x_mb1 = _combine_add_residual(base_mb1, unperm_mb1, mesh, cfg, act_sharding, shard_mode)
        if j == m_sub - 1:
          nxt_idx = (b + 1) % length
          p_nxt0 = _index_at(flat_params[0], nxt_idx)
          g8_nxt0 = _index_at(flat_g8_absmax[0], nxt_idx)
          x_mb1, p_nxt0, g8_nxt0 = jax.lax.optimization_barrier((x_mb1, p_nxt0, g8_nxt0))
          w0_next = gather_for(0, 0)(p_nxt0, g8_absmax=g8_nxt0)

        y_saved = fsdp_prefetch._maybe_offload_to_host((x_mb0, x_mb1), offload_y, shard_mode)
        (x_mb0, unperm_mb1, base_mb1), saved_j = fwd_sub(
            j, gi, (x_mb0, unperm_mb1, base_mb1, x_mb1), w, b_flat_state[j]
        )
        ys_flat.append(y_saved)
        saved_flat.append(saved_j)

      return ((x_mb0, unperm_mb1, base_mb1), w0_next), (tuple(ys_flat), tuple(saved_flat))

    w0 = gather_for(0, 0)(
        fsdp_prefetch._slice_at(flat_params[0], 0),
        g8_absmax=fsdp_prefetch._slice_at(flat_g8_absmax[0], 0),
    )
    x_in = _maybe_constrain(x_in, act_sharding, shard_mode)
    r_in = x_in.reshape(x_in.shape[0] // 2, 2, *x_in.shape[1:])
    x_mb0_init = _maybe_constrain(r_in[:, 0], act_sharding, shard_mode)
    base_mb1_init = _maybe_constrain(r_in[:, 1], act_sharding, shard_mode)
    unperm_zero = _maybe_constrain(
        jax.lax.broadcast(
            jnp.zeros((), dtype=base_mb1_init.dtype),
            (base_mb1_init.shape[0] * num_ep, *base_mb1_init.shape[1:]),
        ),
        act_sharding,
        shard_mode,
    )
    xs = (jnp.arange(length), flat_state, flat_params, flat_g8_absmax)
    ((x_mb0_final, unperm_mb1_final, base_mb1_final), _), (ys_flat_stacked, saved_flat_stacked) = jax.lax.scan(
        body_bs, ((x_mb0_init, unperm_zero, base_mb1_init), w0), xs
    )
    x_mb1_final = _combine_add_residual(base_mb1_final, unperm_mb1_final, mesh, cfg, act_sharding, shard_mode)
    y = _maybe_constrain(
        jnp.stack([x_mb0_final, x_mb1_final], axis=1).reshape(x_in.shape), act_sharding, shard_mode
    )
    residuals = (
        ys_flat_stacked,
        saved_flat_stacked,
        flat_g8_absmax,
        params_leaves,
        state_leaves,
        diff_args_leaves,
        diff_kwargs_leaves,
    )
    return (y, state_leaves), residuals

  def body(carry, xs):
    y, w0_cur = carry
    b, b_flat_state, b_flat_params, b_flat_g8 = xs
    ys_flat = []
    saved_flat = []
    w0_next = None

    for j, (gi, _) in enumerate(sched):
      if j == 0:
        w = w0_cur
      else:
        y, p_j, g8_j = jax.lax.optimization_barrier((y, b_flat_params[j], b_flat_g8[j]))
        w = gather_for(j, gi)(p_j, g8_absmax=g8_j)

      if j == m_sub - 1:
        nxt_idx = (b + 1) % length
        p_nxt0 = _index_at(flat_params[0], nxt_idx)
        g8_nxt0 = _index_at(flat_g8_absmax[0], nxt_idx)
        y, p_nxt0, g8_nxt0 = jax.lax.optimization_barrier((y, p_nxt0, g8_nxt0))
        w0_next = gather_for(0, 0)(p_nxt0, g8_absmax=g8_nxt0)

      y_saved = fsdp_prefetch._maybe_offload_to_host(y, offload_y, shard_mode)
      y, saved_j = fwd_sub(j, gi, y, w, b_flat_state[j])
      ys_flat.append(y_saved)
      saved_flat.append(saved_j)

    return (y, w0_next), (tuple(ys_flat), tuple(saved_flat))

  w0 = gather_for(0, 0)(
      fsdp_prefetch._slice_at(flat_params[0], 0),
      g8_absmax=fsdp_prefetch._slice_at(flat_g8_absmax[0], 0),
  )
  x_in = _maybe_constrain(x_in, act_sharding, shard_mode)
  xs = (jnp.arange(length), flat_state, flat_params, flat_g8_absmax)
  (y, _), (ys_flat_stacked, saved_flat_stacked) = jax.lax.scan(body, (x_in, w0), xs)
  residuals = (
      ys_flat_stacked,
      saved_flat_stacked,
      flat_g8_absmax,
      params_leaves,
      state_leaves,
      diff_args_leaves,
      diff_kwargs_leaves,
  )
  return (y, state_leaves), residuals


def _custom_vjp_flat_prefetch_pipeline_bwd(graphdefs, length, ctxs, sizes, res, g_out):
  """Flat-unrolled backward pass (unpinned BWD gathers + optional zero-d_acc cross-block deferred RS on j=0)."""
  ys_flat_stacked, saved_flat_stacked, flat_g8_absmax, params_leaves, state_leaves, d_args, d_kwargs = res
  g, _ = g_out
  sched = _sublayer_schedule(sizes)
  m_sub = len(sched)
  num_groups = len(graphdefs)
  offload_y = ctxs[0].offload_layer_input
  shard_mode = ctxs[0].shard_mode
  use_sg = ctxs[0].use_scheduling_group
  defer_j0 = bool(ctxs[0].defer_grad_reduce_scatter)
  batch_split = bool(ctxs[0].batch_split)
  g_orig_shape = g.shape
  mesh = ctxs[0].mesh
  cfg = ctxs[0].config
  act_sharding, num_ep = _batchsplit_shardings_and_ep(mesh, cfg) if batch_split else (None, 1)

  def _expand_bwd_y(y_dev):
    if not batch_split:
      return y_dev
    x0_d, x1_d = y_dev
    unperm_d = _maybe_constrain(
        jax.lax.broadcast(jnp.zeros((), dtype=x1_d.dtype), (x1_d.shape[0] * num_ep, *x1_d.shape[1:])),
        act_sharding,
        shard_mode,
    )
    return (x0_d, unperm_d, x1_d, x1_d)

  params, state, gather, rs_fns, layout_fns, _, layer_vjp_fns, j0_setup = _setup_flat_sublayers(
      graphdefs, params_leaves, state_leaves, d_args, d_kwargs, ctxs, sizes
  )
  flat_params = tuple(jax.tree.map(lambda x, k=k: x[:, k], params[gi]) for gi, k in sched)
  flat_state = tuple(jax.tree.map(lambda x, k=k: x[:, k], state[gi]) for gi, k in sched)

  layouts = {}
  for j, (gi, _) in enumerate(sched):
    if gi in layouts:
      continue
    y_sample = _expand_bwd_y(
        fsdp_prefetch._maybe_onload_to_device(
            fsdp_prefetch._slice_at(ys_flat_stacked[j], 0), offload_y, shard_mode
        )
    )
    w_sample = jax.eval_shape(
        gather[gi],
        fsdp_prefetch._slice_at(flat_params[j], 0),
        fsdp_prefetch._slice_at(flat_g8_absmax[j], 0),
    )
    w_dummy = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), w_sample)
    s_sample = fsdp_prefetch._slice_at(flat_state[j], 0)
    layouts[gi] = layout_fns[gi](y_sample, w_dummy, s_sample)

  j0_layout = None
  if defer_j0:
    _, _, j0_gather, j0_rs, j0_layout_fn, _, j0_vjp = j0_setup
    y0_sample = _expand_bwd_y(
        fsdp_prefetch._maybe_onload_to_device(
            fsdp_prefetch._slice_at(ys_flat_stacked[0], 0), offload_y, shard_mode
        )
    )
    w0_sample = jax.eval_shape(
        j0_gather,
        fsdp_prefetch._slice_at(flat_params[0], 0),
        fsdp_prefetch._slice_at(flat_g8_absmax[0], 0),
    )
    w0_dummy = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), w0_sample)
    s0_sample = fsdp_prefetch._slice_at(flat_state[0], 0)
    j0_layout = j0_layout_fn(y0_sample, w0_dummy, s0_sample)

  def gather_for(j, gi):
    if j == 0 and defer_j0:
      return j0_setup[2]
    return gather[gi]

  last_gi, _ = sched[-1]
  w_last = gather_for(m_sub - 1, last_gi)(
      fsdp_prefetch._slice_at(flat_params[-1], length - 1),
      g8_absmax=fsdp_prefetch._slice_at(flat_g8_absmax[-1], length - 1),
      pin_sc_override=-1,
  )
  if batch_split:
    g = _maybe_constrain(g, act_sharding, shard_mode)
    r_g = g.reshape(g.shape[0] // 2, 2, *g.shape[1:])
    g_mb0 = _maybe_constrain(r_g[:, 0], act_sharding, shard_mode)
    g_mb1 = _maybe_constrain(r_g[:, 1], act_sharding, shard_mode)
    g_unperm_init = combine_ep_ubatch_bwd_fn(g_mb1, mesh, cfg)
    g = (g_mb0, g_unperm_init, g_mb1)

  if not defer_j0:

    def body_stage1(carry, xs):
      g, w = carry
      b, b_ys_flat, b_saved_flat, b_flat_state, b_flat_params, b_flat_g8 = xs
      # Onload all sub-layer y inputs at the top of the block so PCIe DMA overlaps all 4 sub-layers' TC compute!
      b_ys_dev = tuple(fsdp_prefetch._maybe_onload_to_device(yj, offload_y, shard_mode) for yj in b_ys_flat)
      dws_flat = [None] * m_sub

      for j in reversed(range(m_sub)):
        gi, _ = sched[j]
        if j > 0:
          prv_gi, _ = sched[j - 1]
          p_prv = b_flat_params[j - 1]
          g8_prv = b_flat_g8[j - 1]
          gather_prv = gather_for(j - 1, prv_gi)
        else:
          prv_gi, _ = sched[-1]
          prv_idx = (b - 1) % length
          p_prv = _index_at(flat_params[-1], prv_idx)
          g8_prv = _index_at(flat_g8_absmax[-1], prv_idx)
          gather_prv = gather_for(m_sub - 1, prv_gi)

        g, p_prv, g8_prv = jax.lax.optimization_barrier((g, p_prv, g8_prv))
        w_prv = gather_prv(p_prv, g8_absmax=g8_prv, pin_sc_override=-1)
        dy_j, dw_j = layer_vjp_fns[gi](
            _expand_bwd_y(b_ys_dev[j]), w, b_flat_state[j], g, b_saved_flat[j], layouts[gi]
        )
        g = dy_j[:3] if batch_split else dy_j
        dw_j = rs_fns[gi](dw_j)
        # Do NOT put dw_j or w_prv in an end-of-sub-layer barrier: j-1's top barrier already orders `g`, while
        # dw_j RS and w_prv dual-SC gathers overlap sub-layer j-1 natively without blocking compute!
        w = w_prv
        dws_flat[j] = dw_j

      return (g, w), tuple(dws_flat)

    xs = (jnp.arange(length), ys_flat_stacked, saved_flat_stacked, flat_state, flat_params, flat_g8_absmax)
    (g, _), dws_flat_all = jax.lax.scan(body_stage1, (g, w_last), xs, reverse=True)
  else:
    # Stage 2 Flat: j=3, 2, 1 overlap their RS natively inside the flat block (0 carry HBM!), while j=0 defers its
    # tagged MoE gradients across the block boundary into block b-1's j=3..1 and returns via shifted scan output!
    _, _, j0_gather, j0_rs, _, _, j0_vjp = j0_setup

    def extract_tagged(dw):
      leaves, treedef = jax.tree_util.tree_flatten(dw)
      flags = j0_rs.tag_flags(leaves)
      return tuple(x for x, f in zip(leaves, flags) if f), treedef, flags

    def first_j0_tagged(g_sample):
      y0 = _expand_bwd_y(
          fsdp_prefetch._maybe_onload_to_device(fsdp_prefetch._slice_at(ys_flat_stacked[0], 0), offload_y, shard_mode)
      )
      sv0 = fsdp_prefetch._slice_at(saved_flat_stacked[0], 0)
      st0 = fsdp_prefetch._slice_at(flat_state[0], 0)
      w0 = j0_gather(
          fsdp_prefetch._slice_at(flat_params[0], 0),
          g8_absmax=fsdp_prefetch._slice_at(flat_g8_absmax[0], 0),
          pin_sc_override=-1,
      )
      _, dw0 = j0_vjp(y0, w0, st0, g_sample, sv0, j0_layout)
      dw0_untagged = j0_rs(dw0, "untagged")
      tagged_leaves, _, _ = extract_tagged(dw0_untagged)
      return tagged_leaves

    pend_avals = jax.eval_shape(first_j0_tagged, g)

    def _zero_for_aval(a):
      z = jnp.zeros(a.shape, a.dtype)
      if getattr(a, "sharding", None) is not None and len(a.sharding.spec) > 0:
        z = jax.lax.with_sharding_constraint(z, a.sharding)
      return z

    pend_init = tuple(_zero_for_aval(a) for a in pend_avals)
    template_dw_j0 = jax.tree.map(lambda x: jnp.zeros_like(x[0]), flat_params[0])

    def flush_j0_tagged(pend_tagged, sg_id=None):
      leaves, treedef = jax.tree_util.tree_flatten(template_dw_j0)
      flags = j0_rs.tag_flags(leaves)
      it_t = iter(pend_tagged)
      full_in = [next(it_t) if f else x for x, f in zip(leaves, flags)]
      reduced = j0_rs(jax.tree_util.tree_unflatten(treedef, full_in), "tagged", sg_id=sg_id)
      red_leaves = jax.tree_util.tree_leaves(reduced)
      return tuple(r for r, f in zip(red_leaves, flags) if f)

    def body_stage2(carry, xs):
      g, w, pend_j0 = carry
      b, b_ys_flat, b_saved_flat, b_flat_state, b_flat_params, b_flat_g8 = xs
      b_ys_dev = tuple(fsdp_prefetch._maybe_onload_to_device(yj, offload_y, shard_mode) for yj in b_ys_flat)
      dws_flat = [None] * m_sub
      flushed_prev_j0 = None
      new_pend_j0 = None

      for j in reversed(range(m_sub)):
        gi, _ = sched[j]
        if j > 0:
          prv_gi, _ = sched[j - 1]
          p_prv = b_flat_params[j - 1]
          g8_prv = b_flat_g8[j - 1]
          gather_prv = gather_for(j - 1, prv_gi)
        else:
          prv_gi, _ = sched[-1]
          prv_idx = (b - 1) % length
          p_prv = _index_at(flat_params[-1], prv_idx)
          g8_prv = _index_at(flat_g8_absmax[-1], prv_idx)
          gather_prv = gather_for(m_sub - 1, prv_gi)

        if j == m_sub - 1:
          g, p_prv, g8_prv, pend_j0 = jax.lax.optimization_barrier((g, p_prv, g8_prv, pend_j0))
          sg_id = fsdp_prefetch._next_sg_id() if use_sg else None
          flushed_prev_j0 = flush_j0_tagged(pend_j0, sg_id=sg_id)
          w_prv = gather_prv(p_prv, g8_absmax=g8_prv, sg_id=sg_id, pin_sc_override=-1)
        elif j == 0:
          # Wait for flushed_prev_j0 only right before j=0 runs, giving RS(pend_j0) all 3 sub-layers (j=3, 2, 1)
          # to overlap while ensuring pend_j0 and new_pend_j0 never coexist in HBM!
          g, p_prv, g8_prv, flushed_prev_j0 = jax.lax.optimization_barrier((g, p_prv, g8_prv, flushed_prev_j0))
          w_prv = gather_prv(p_prv, g8_absmax=g8_prv, pin_sc_override=-1)
        else:
          g, p_prv, g8_prv = jax.lax.optimization_barrier((g, p_prv, g8_prv))
          w_prv = gather_prv(p_prv, g8_absmax=g8_prv, pin_sc_override=-1)

        if j == 0:
          dy_0, dw_0 = j0_vjp(_expand_bwd_y(b_ys_dev[0]), w, b_flat_state[0], g, b_saved_flat[0], j0_layout)
          g = dy_0[:3] if batch_split else dy_0
          dw_0_untagged = j0_rs(dw_0, "untagged")
          new_pend_j0, _, flags0 = extract_tagged(dw_0_untagged)
          dw0_untagged_leaves = [x for x, f in zip(jax.tree_util.tree_leaves(dw_0_untagged), flags0) if not f]
          dws_flat[0] = tuple(dw0_untagged_leaves)
          g, new_pend_j0 = jax.lax.optimization_barrier((g, new_pend_j0))
        else:
          dy_j, dw_j = layer_vjp_fns[gi](
              _expand_bwd_y(b_ys_dev[j]), w, b_flat_state[j], g, b_saved_flat[j], layouts[gi]
          )
          g = dy_j[:3] if batch_split else dy_j
          dw_j = rs_fns[gi](dw_j)
          dws_flat[j] = dw_j
        w = w_prv

      return (g, w, new_pend_j0), (tuple(dws_flat), flushed_prev_j0)

    xs = (jnp.arange(length), ys_flat_stacked, saved_flat_stacked, flat_state, flat_params, flat_g8_absmax)
    (g, _, final_pend_j0), (dws_flat_raw, flushed_shifted) = jax.lax.scan(
        body_stage2, (g, w_last, pend_init), xs, reverse=True
    )
    first_j0 = flush_j0_tagged(final_pend_j0)
    full_j0_tagged = tuple(
        jnp.concatenate([jnp.expand_dims(f0, 0), fs[:-1]], axis=0) for f0, fs in zip(first_j0, flushed_shifted)
    )
    _, treedef0 = jax.tree_util.tree_flatten(template_dw_j0)
    flags0 = j0_rs.tag_flags(jax.tree_util.tree_leaves(template_dw_j0))
    it_t = iter(full_j0_tagged)
    it_u = iter(dws_flat_raw[0])
    dw0_rebuilt = jax.tree_util.tree_unflatten(treedef0, [next(it_t) if f else next(it_u) for f in flags0])
    dws_flat_all = (dw0_rebuilt,) + dws_flat_raw[1:]

  if batch_split:
    dx_mb0, _, dx_mb1 = g
    g = _maybe_constrain(
        jnp.stack([dx_mb0, dx_mb1], axis=1).reshape(g_orig_shape), act_sharding, shard_mode
    )

  # Re-stack [length, sizes[gi], ...] per group after scan!
  d_params = []
  offset = 0
  for gi in range(num_groups):
    n = sizes[gi]
    group_slices = dws_flat_all[offset : offset + n]
    d_params.append(jax.tree.map(lambda *xs_: jnp.stack(xs_, axis=1), *group_slices))
    offset += n
  d_params = tuple(d_params)

  params_bar_leaves = tuple(tuple(jax.tree_util.tree_leaves(d)) for d in d_params)
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in d_args)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in d_kwargs)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_flat_prefetch_pipeline.defvjp(
    _custom_vjp_flat_prefetch_pipeline_fwd, _custom_vjp_flat_prefetch_pipeline_bwd
)


def _prefetch_sublayer_names(layers):
  names_fn = getattr(layers, "prefetch_sublayer_names", None)
  if names_fn is None:
    return None
  names = tuple(names_fn())
  return names if len(names) >= 2 else None


def _unstack_variable(var, num_local: int, var_scan_axis: int):
  if not isinstance(var, nnx.Variable):
    return (var,) * num_local
  val = var.get_value()
  meta = var.get_metadata()
  updates = {}
  for key in ("sharding", "out_sharding", "kernel_axes", "sharding_names"):
    axes = getattr(var, key, None)
    if axes is None:
      axes = meta.get(key)
    if isinstance(axes, (jax.sharding.PartitionSpec, tuple, list)) and "local_layers" in axes:
      filtered = [a for a in axes if a != "local_layers"]
      updates[key] = (
          jax.sharding.PartitionSpec(*filtered) if isinstance(axes, jax.sharding.PartitionSpec) else tuple(filtered)
      )
  ndim = getattr(val, "ndim", 0)
  if meta.get(nnx.PARTITION_NAME) == "local_layers":
    local_axis = var_scan_axis if ndim > var_scan_axis else 0
  else:
    inner_axis = var_scan_axis if ndim - 1 > var_scan_axis else 0
    outer_axis = var_scan_axis if ndim > var_scan_axis else 0
    local_axis = inner_axis + 1 if inner_axis >= outer_axis else inner_axis
  if ndim <= local_axis or val.shape[local_axis] != num_local:
    stripped = var.replace(**updates) if updates else var
    return (stripped,) * num_local
  slices = jnp.unstack(val, axis=local_axis)
  return tuple(var.replace(value=s, **updates) for s in slices)


def _unstack_scanned_sublayers(sublayers, num_local: int, scan_axis: int):
  graphdef, params, state = nnx.split(sublayers, nnx.Param, ...)

  def unstack_state(tree, var_scan_axis):
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
    if not leaves:
      return [tree] * num_local
    unstacked = [_unstack_variable(leaf, num_local, var_scan_axis) for leaf in leaves]
    return [jax.tree_util.tree_unflatten(treedef, [u[i] for u in unstacked]) for i in range(num_local)]

  params_per_stage = unstack_state(params, scan_axis)
  state_per_stage = unstack_state(state, 0)
  return [(graphdef, p, s) for p, s in zip(params_per_stage, state_per_stage)]


def sublayer_stages(layers):
  num_local = getattr(layers, "num_local", 0)
  num_global = getattr(layers, "num_global", 0)
  local_layers = getattr(layers, "local_layers", None)
  global_layer = getattr(layers, "global_layer", None)
  if (local_layers is not None or global_layer is not None) and num_local + num_global >= 2:
    scan_axis = getattr(getattr(layers, "config", None), "param_scan_axis", 0)
    stages = []
    if local_layers is not None and num_local > 0:
      stages.extend(_unstack_scanned_sublayers(local_layers, num_local, scan_axis))
    if global_layer is not None and num_global == 1:
      stages.append(nnx.split(global_layer, nnx.Param, ...))
  else:
    names = _prefetch_sublayer_names(layers)
    if names is None:
      return None
    stages = [nnx.split(getattr(layers, name), nnx.Param, ...) for name in names]
  if len(stages) < 2:
    return None
  block_params = nnx.state(layers, nnx.Param)
  stage_param_size = sum(x.size for _, p, _ in stages for x in jax.tree_util.tree_leaves(p))
  block_param_size = sum(x.size for x in jax.tree_util.tree_leaves(block_params))
  if stage_param_size != block_param_size:
    max_logging.warning("prefetch_fsdp_weights: the block has params outside its sub-layers; prefetching whole blocks.")
    return None
  return stages


def split_batchsplit_args_kwargs(full_bsz: int, args, kwargs, mesh, config):
  """Splits `[full_bsz, ...]` positional/keyword arrays into `(mb0, mb1)` 2-tuples with preserved shardings."""
  rules = sharding.get_logical_axis_rules() or config.logical_axis_rules
  seg_pspec = sharding.logical_to_mesh_axes(("activation_batch", "activation_norm_length"), mesh=mesh, rules=rules)
  seg_sharding = jax.sharding.NamedSharding(mesh, seg_pspec)

  def _split_arg(a):
    if isinstance(a, jax.Array) and a.ndim >= 2 and a.shape[0] == full_bsz:
      if a.ndim == 2 and config.shard_mode != ShardMode.EXPLICIT:
        a = jax.lax.with_sharding_constraint(a, seg_sharding)
      r = a.reshape(full_bsz // 2, 2, *a.shape[1:])
      a0, a1 = r[:, 0], r[:, 1]
      if a0.ndim == 2 and config.shard_mode != ShardMode.EXPLICIT:
        a0 = jax.lax.with_sharding_constraint(a0, seg_sharding)
        a1 = jax.lax.with_sharding_constraint(a1, seg_sharding)
      return (a0, a1)
    return a

  return tuple(_split_arg(a) for a in args), {k: _split_arg(v) for k, v in kwargs.items()}


def apply_layers_with_fsdp_prefetch(
    layers, stages, x_in, *args, length: int, mesh, config, remat_policy=None, prevent_cse: bool = False, **kwargs
):
  if length == 0:
    return x_in, layers, None
  act_sharding, _ = _batchsplit_shardings_and_ep(mesh, config)
  x_in = _maybe_constrain(x_in, act_sharding, config.shard_mode)
  if getattr(config, "prefetch_fsdp_batch_split", False):
    args, kwargs = split_batchsplit_args_kwargs(x_in.shape[0], args, kwargs, mesh, config)
  _, _, _, diff_args_leaves, diff_kwargs_leaves, ctx = fsdp_prefetch._prepare_prefetch_call(
      layers,
      args,
      kwargs,
      length=length,
      mesh=mesh,
      config=config,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
      defer_host_offload=False,
  )

  def _norm_graphdef(gd) -> str:
    return re.sub(r" at 0x[0-9a-fA-F]+>", ">", repr(gd))

  groups = []
  for stage_graphdef, p, s in stages:
    p_shardings = fsdp_prefetch._resolve_layer_shardings_from_params(p, mesh, config)
    p, s = fsdp_prefetch._layer_major(p, s, config.param_scan_axis, length)
    p_leaves, p_treedef = jax.tree_util.tree_flatten(p)
    s_leaves, s_treedef = jax.tree_util.tree_flatten(s)
    shapes = [x.shape for x in p_leaves + s_leaves]
    gd_key = _norm_graphdef(stage_graphdef)
    if groups:
      g = groups[-1]
      if (
          g[6] == gd_key
          and g[1] == p_treedef
          and g[2] == s_treedef
          and [x.shape for x in g[3][0] + g[4][0]] == shapes
          and g[5] == p_shardings
      ):
        g[3].append(p_leaves)
        g[4].append(s_leaves)
        continue
    groups.append([stage_graphdef, p_treedef, s_treedef, [p_leaves], [s_leaves], p_shardings, gd_key])

  max_logging.warning(
      f"[FSDP_PREFETCH] Inhomogeneous sub-layer grouping: {[len(g[3]) for g in groups]} sub-layers across {length} blocks."
  )

  def stack(per_sublayer):
    return tuple(jnp.stack(xs, axis=1) for xs in zip(*per_sublayer))

  final_y, _ = _custom_vjp_flat_prefetch_pipeline(
      tuple(g[0] for g in groups),
      x_in,
      tuple(stack(g[3]) for g in groups),
      tuple(stack(g[4]) for g in groups),
      diff_args_leaves,
      diff_kwargs_leaves,
      length,
      tuple(
          ctx._replace(
              params_treedef=g[1],
              state_treedef=g[2],
              scan_axis=0,
              layer_shardings=g[5],
              defer_host_offload=False,
          )
          for g in groups
      ),
      tuple(len(g[3]) for g in groups),
  )
  return final_y, layers, None
