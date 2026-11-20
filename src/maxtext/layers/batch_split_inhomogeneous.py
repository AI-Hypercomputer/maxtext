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

"""Generalized Option B Batch-Split scheduling for inhomogeneous scanned MoE blocks.

Handles repeating heterogeneous MoE block stacks (e.g., Qwen3-Next / Qwen3.5 `[3 GDN + 1 FullAttn]`
and DeepSeek-V4 `[1 HCA + 1 CSA]` with 4D mHC streams) one sub-layer at a time inside nested
while-loops, wrapping weight-free SparseCore EP collectives (`SC0`) and FSDP prefetch / deferred
gradient reduce-scatters (`SC1`) seamlessly across heterogeneous sub-layer and block boundaries.

Strictly isolated behind `config.enable_batch_split=True`; leaves all existing scan and
`fsdp_prefetch*` codepaths untouched when disabled.
"""

# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
from typing import Any, Callable

from flax import nnx
import jax
from jax.experimental.layout import Layout, with_layout_constraint
import jax.numpy as jnp
from maxtext.common.common_types import ShardMode
from maxtext.layers import batch_split
from maxtext.layers import fsdp_prefetch
from maxtext.layers import fsdp_prefetch_inhomogeneous
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import sharding


_index_at = fsdp_prefetch_inhomogeneous._index_at
_sublayer_scan = fsdp_prefetch_inhomogeneous._sublayer_scan
_concat0 = fsdp_prefetch_inhomogeneous._concat0
_update_at = fsdp_prefetch_inhomogeneous._update_at


@dataclass(frozen=True)
class _GroupCtx:
  """Static per-group metadata for `_custom_vjp_nested_batch_split`."""

  mesh: Any
  shard_mode: ShardMode
  debug_sharding: bool
  metadata_axis_name: str
  static_args_leaves: tuple
  static_kwargs_leaves: tuple
  params_treedef: Any
  state_treedef: Any
  args_treedef: Any
  kwargs_treedef: Any
  use_fsdp_prefetch: bool
  defer_grad_reduce_scatter: bool
  offload_layer_inputs: bool
  sparse_core_id: int | None
  remat_policy: Any = None
  prevent_cse: bool = True
  replicated_red_axes: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _InhomogeneousBwdBundle:
  """Captured backward jaxprs and variable masks for one sub-layer group."""

  attn_w_indices: tuple[int, ...]
  exp_w_indices: tuple[int, ...]
  exp_accum_sub_indices: tuple[int, ...]
  shared_w_indices: tuple[int, ...]
  post_ctx_indices: tuple[int, ...]
  hs_treedef: Any
  route_in_mask: tuple[bool, ...]
  route_in_treedef: Any
  route_ag_mask: tuple[bool, ...]
  route_ag_treedef: Any
  has_seq_lb: bool
  has_exp_lb: bool
  has_x_scale: bool
  has_bias: bool
  has_deferred_combine_bwd: bool
  attn_route_subfn: batch_split._RematSubfn
  experts_subfn: batch_split._RematSubfn
  experts_acc_subfn: batch_split._RematSubfn | None
  shared_subfn: batch_split._RematSubfn
  dispatch_fwd: Any
  dispatch_bwd: Any
  combine_bwd: Any
  post_combine_bwd: Any


def _batch_split_sublayer_names(layers) -> tuple[str, ...] | None:
  """Returns named sub-layer attributes in execution order if opted in, else None."""
  names_fn = getattr(layers, "batch_split_sublayer_names", None)
  if names_fn is None:
    names_fn = getattr(layers, "prefetch_sublayer_names", None)
  if names_fn is None:
    return None
  names = tuple(names_fn())
  return names if len(names) >= 2 else None


def sublayer_stages(layers):
  """Returns `(stages, stage_info)` if `layers` is an inhomogeneous block whose sub-layers all support batch-split.

  `stage_info` describes how to sow `moe_lb_loss` and `moe_bias_updates` back onto `layers` matching
  the structure expected by `train.py`:
    - `("hybrid", num_local, num_global)` for `local_layers` + `global_layer` blocks (Qwen3-Next / Qwen3.5).
    - `("named", names)` for named sub-layer blocks (`DeepSeek4ScannableBlock`).
  """
  num_local = getattr(layers, "num_local", 0)
  num_global = getattr(layers, "num_global", 0)
  local_layers = getattr(layers, "local_layers", None)
  global_layer = getattr(layers, "global_layer", None)
  if (local_layers is not None or global_layer is not None) and num_local + num_global >= 2:
    if local_layers is not None and num_local > 0 and not batch_split.supports_batch_split(local_layers):
      return None
    if global_layer is not None and num_global == 1 and not batch_split.supports_batch_split(global_layer):
      return None
    scan_axis = getattr(getattr(layers, "config", None), "param_scan_axis", 0)
    stages = []
    if local_layers is not None and num_local > 0:
      stages.extend(fsdp_prefetch_inhomogeneous._unstack_scanned_sublayers(local_layers, num_local, scan_axis))
    if global_layer is not None and num_global == 1:
      stages.append(nnx.split(global_layer, nnx.Param, ...))
    stage_info = ("hybrid", num_local, num_global)
  else:
    names = _batch_split_sublayer_names(layers)
    if names is None:
      return None
    sublayers = [getattr(layers, name) for name in names]
    if not all(batch_split.supports_batch_split(sub) for sub in sublayers):
      return None
    stages = [nnx.split(sub, nnx.Param, ...) for sub in sublayers]
    stage_info = ("named", names)

  if len(stages) < 2:
    return None
  block_params = nnx.state(layers, nnx.Param)
  stage_param_size = sum(x.size for _, p, _ in stages for x in jax.tree_util.tree_leaves(p))
  block_param_size = sum(x.size for x in jax.tree_util.tree_leaves(block_params))
  if stage_param_size != block_param_size:
    max_logging.info("enable_batch_split: inhomogeneous block has parameters outside its sub-layers; falling back.")
    return None
  return stages, stage_info


def _setup_one_group(
    graphdef,
    params_leaves_stacked,
    state_leaves_stacked,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    ctx: _GroupCtx,
):
  """Builds gather/RS closures and pure batch-split sub-functions for one `[length, n, ...]` group."""
  p_slice0 = [x[:, 0] for x in params_leaves_stacked]
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, params_leaves_stacked)
  state = jax.tree_util.tree_unflatten(ctx.state_treedef, state_leaves_stacked)

  if ctx.use_fsdp_prefetch:
    layer_shardings = [fsdp_prefetch._per_layer_sharding(p, ctx.mesh) for p in p_slice0]
    pin_layout_flags = [
        fsdp_prefetch._needs_minor_dim_layout_pin(p, s, ctx.mesh) for p, s in zip(p_slice0, layer_shardings)
    ]
    rep_red_axes = ctx.replicated_red_axes if len(ctx.replicated_red_axes) == len(p_slice0) else ((),) * len(p_slice0)

    def gather_fn(layer_params):
      leaves, treedef = jax.tree_util.tree_flatten(layer_params)
      out = []
      for x, pin, red_ax in zip(leaves, pin_layout_flags, rep_red_axes):
        if red_ax:
          x = batch_split._cast_replicated_reduced(x, ctx.mesh, red_ax)
        else:
          x = fsdp_prefetch._apply_sharding_hint(
              x, ctx.mesh, ctx.shard_mode, ctx.debug_sharding, sparse_core_id=ctx.sparse_core_id
          )
        if pin:
          x = with_layout_constraint(x, Layout(tuple(range(x.ndim))))
        out.append(x)
      return jax.tree_util.tree_unflatten(treedef, out)

    def rs_one(g, s, tag, red_ax):
      if s is None:
        return g
      if red_ax:
        out = batch_split._batch_split_replicated_all_reduce(g, ctx.mesh, red_ax, sparse_core_id=ctx.sparse_core_id)
        if out is not None:
          return out
      if tag:
        out = batch_split._batch_split_fsdp_reduce_scatter(
            g, ctx.mesh, s.spec, sharding.remove_fsdp_from_sharding(s).spec, sparse_core_id=ctx.sparse_core_id
        )
        if out is not None:
          return out
      scatter = fsdp_prefetch._fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
      if scatter is not None and fsdp_prefetch._is_unreduced_over(
          g, fsdp_prefetch._expected_unreduced_axes(ctx.mesh, s.spec, scatter[1])
      ):
        g = fsdp_prefetch._pin_scatter_dim_major(g, scatter[0])
      return sharding.maybe_shard_with_name(g, s, ctx.shard_mode, debug_sharding=ctx.debug_sharding)

    tag_cache = {}

    def tag_flags(leaves):
      if "flags" not in tag_cache:
        flags = []
        for g, s, red_ax in zip(leaves, layer_shardings, rep_red_axes):
          if red_ax:
            eligible = fsdp_prefetch._is_unreduced_over(g, red_ax)
          else:
            scatter = None if s is None else fsdp_prefetch._fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
            eligible = scatter is not None and fsdp_prefetch._is_unreduced_over(
                g, fsdp_prefetch._expected_unreduced_axes(ctx.mesh, s.spec, scatter[1])
            )
          flags.append(bool(ctx.defer_grad_reduce_scatter and eligible))
        tag_cache["flags"] = flags
      return tag_cache["flags"]

    def rs_fn(layer_grads, which="all"):
      leaves, treedef = jax.tree_util.tree_flatten(layer_grads)
      flags = tag_flags(leaves)
      out = [
          rs_one(g, s, tag, red_ax) if which == "all" or (which == "tagged") == tag else g
          for g, s, tag, red_ax in zip(leaves, layer_shardings, flags, rep_red_axes)
      ]
      return jax.tree_util.tree_unflatten(treedef, out)

    rs_fn.tag_flags = tag_flags
  else:

    def gather_fn(layer_params):
      return layer_params

    def rs_fn(layer_grads, which="all"):
      del which
      return layer_grads

    rs_fn.tag_flags = lambda leaves: [False] * len(leaves)

  def _reconstruct_call_args(d_args, d_kwargs):
    args_leaves = [s if s is not None else d for d, s in zip(d_args, ctx.static_args_leaves)]
    args_tuple = jax.tree_util.tree_unflatten(ctx.args_treedef, args_leaves)
    kwargs_leaves = [s if s is not None else d for d, s in zip(d_kwargs, ctx.static_kwargs_leaves)]
    kwargs_dict = dict(jax.tree_util.tree_unflatten(ctx.kwargs_treedef, kwargs_leaves))
    return args_tuple, kwargs_dict

  mb0_consts = (tuple(mb0_args_leaves), tuple(mb0_kwargs_leaves))
  mb1_consts = (tuple(mb1_args_leaves), tuple(mb1_kwargs_leaves))

  def _merge_layer(layer_weights, layer_state):
    w_clean, s_clean = maxtext_utils_nnx.nnx_remove_scan_axis((layer_weights, layer_state), ctx.metadata_axis_name)
    if ctx.metadata_axis_name != "layers":
      w_clean, s_clean = maxtext_utils_nnx.nnx_remove_scan_axis((w_clean, s_clean), "layers")
    if "local_layers" not in (ctx.metadata_axis_name, "layers"):
      w_clean, s_clean = maxtext_utils_nnx.nnx_remove_scan_axis((w_clean, s_clean), "local_layers")
    return nnx.merge(graphdef, w_clean, s_clean)

  def attn_route_pure(x, layer_weights, layer_state, d_args, d_kwargs):
    args_tuple, kwargs_dict = _reconstruct_call_args(d_args, d_kwargs)
    layer = _merge_layer(layer_weights, layer_state)
    return layer.batch_split_attn_and_route(x, *args_tuple, **kwargs_dict)

  def dispatch_pure(moe_in, route_in, layer_state, template_w):
    layer = _merge_layer(template_w, layer_state)
    return layer.batch_split_dispatch(moe_in, route_in)

  def dispatch_bwd_pure(moe_in, route_in, g_x_val_ag, g_route_ag_diff, layer_state, template_w):
    layer = _merge_layer(template_w, layer_state)
    return layer.batch_split_dispatch_bwd(moe_in, route_in, g_x_val_ag, g_route_ag_diff)

  def experts_pure(
      moe_in_ref, x_val_ag, x_scale_ag, route_ag, layer_weights, layer_state, d_args, d_kwargs, return_wgrad_hooks=False
  ):
    _, kwargs_dict = _reconstruct_call_args(d_args, d_kwargs)
    layer = _merge_layer(layer_weights, layer_state)
    exp_fn = layer.batch_split_experts
    dec_tok = kwargs_dict.get("decoder_input_tokens")
    if return_wgrad_hooks:
      if "return_wgrad_hooks" in inspect.signature(exp_fn).parameters:
        return exp_fn(moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=dec_tok, return_wgrad_hooks=True)
      return (*exp_fn(moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=dec_tok), ())
    return exp_fn(moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=dec_tok)

  def shared_pure(moe_in, layer_weights, layer_state):
    layer = _merge_layer(layer_weights, layer_state)
    return layer.batch_split_shared_expert(moe_in)

  def combine_pure(y_partial, moe_in_ref, layer_state, template_w):
    layer = _merge_layer(template_w, layer_state)
    return layer.batch_split_combine(y_partial, moe_in_ref)

  def post_combine_pure(mlp_sum, h_skip, layer_state, template_w):
    layer = _merge_layer(template_w, layer_state)
    post_fn = getattr(layer, "batch_split_post_combine", None)
    if post_fn is not None:
      return batch_split._without_qwix_interception(post_fn)(mlp_sum, h_skip)
    return mlp_sum + h_skip

  def combine_bwd_pure(g_routed, moe_in_ref, layer_state, template_w):
    layer = _merge_layer(template_w, layer_state)
    cb_fn = layer.batch_split_combine_bwd
    if "defer_dequant" in inspect.signature(cb_fn).parameters:
      return cb_fn(g_routed, moe_in_ref, defer_dequant=True)
    return cb_fn(g_routed, moe_in_ref)

  def finalize_bias_pure(summed_counts, layer_state, template_w):
    if summed_counts is None:
      return None
    layer = _merge_layer(template_w, layer_state)
    return layer.batch_split_finalize_bias_updates(summed_counts)

  return (
      params,
      state,
      gather_fn,
      rs_fn,
      mb0_consts,
      mb1_consts,
      attn_route_pure,
      dispatch_pure,
      dispatch_bwd_pure,
      experts_pure,
      shared_pure,
      combine_pure,
      post_combine_pure,
      combine_bwd_pure,
      finalize_bias_pure,
  )


def _build_group_bwd_bundle(
    *,
    x_sample,
    w_sample,
    s_sample,
    d_args_sample,
    d_kwargs_sample,
    attn_route_pure,
    dispatch_pure,
    dispatch_bwd_pure,
    experts_pure,
    shared_pure,
    post_combine_pure,
    combine_bwd_pure,
    finalize_bias_pure,
    remat_policy: Any = None,
    prevent_cse: bool = True,
) -> tuple[_InhomogeneousBwdBundle, tuple]:
  """Probes shapes via `jax.eval_shape` and captures all backward jaxprs for one group."""

  def _probe_fwd(x_, w_, s_, da_, dk_):
    hs, m_in, r_in, s_lb = attn_route_pure(x_, w_, s_, da_, dk_)
    xv, xs, r_ag = dispatch_pure(m_in, r_in, s_, w_)
    y_p, e_lb, cnt = experts_pure(m_in, xv, xs, r_ag, w_, s_, da_, dk_)
    sh = shared_pure(m_in, w_, s_)
    bias = finalize_bias_pure(cnt, s_, w_)
    return hs, m_in, r_in, s_lb, xv, xs, r_ag, y_p, e_lb, sh, bias

  (
      h_skip_sample,
      moe_in_sample,
      route_in_sample,
      seq_lb_sample,
      x_val_ag_sample,
      x_scale_ag_sample,
      route_ag_sample,
      y_part_sample,
      exp_lb_sample,
      shared_sample,
      bias_sample,
  ) = jax.eval_shape(_probe_fwd, x_sample, w_sample, s_sample, d_args_sample, d_kwargs_sample)

  w_leaves_all, w_treedef = jax.tree_util.tree_flatten(w_sample)
  s_leaves_all, s_treedef = jax.tree_util.tree_flatten(s_sample)
  n_w = len(w_leaves_all)

  route_in_diff, route_in_nondiff, route_in_mask, route_in_treedef = batch_split._split_inexact(route_in_sample)
  route_ag_diff, route_ag_nondiff, route_ag_mask, route_ag_treedef = batch_split._split_inexact(route_ag_sample)
  has_seq_lb = seq_lb_sample is not None
  has_exp_lb = exp_lb_sample is not None
  has_x_scale = x_scale_ag_sample is not None
  has_bias = bias_sample is not None

  # 1. Discover weight subsets read by each sub-function.
  def _attn_probe(x, w_l, s_l, da, dk):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    return attn_route_pure(x, w_t, s_t, da, dk)

  attn_probe_jaxpr = jax.make_jaxpr(_attn_probe)(x_sample, w_leaves_all, s_leaves_all, d_args_sample, d_kwargs_sample)
  attn_w_indices = batch_split._used_var_indices(attn_probe_jaxpr, slice(1, 1 + n_w))

  def _exp_probe(m_ref, xv, xs, r_diff, r_nd, w_l, s_l, da, dk):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    r_ag = batch_split._merge_inexact(r_diff, r_nd, route_ag_mask, route_ag_treedef)
    y_p, e_lb, _ = experts_pure(m_ref, xv, xs, r_ag, w_t, s_t, da, dk)
    return (y_p, e_lb) if has_exp_lb else (y_p,)

  exp_probe_prefix = 1 + 1 + (1 if has_x_scale else 0) + len(route_ag_diff) + len(route_ag_nondiff)
  exp_probe_jaxpr = jax.make_jaxpr(_exp_probe)(
      moe_in_sample,
      x_val_ag_sample,
      x_scale_ag_sample,
      route_ag_diff,
      route_ag_nondiff,
      w_leaves_all,
      s_leaves_all,
      d_args_sample,
      d_kwargs_sample,
  )
  exp_w_indices = batch_split._used_var_indices(exp_probe_jaxpr, slice(exp_probe_prefix, exp_probe_prefix + n_w))

  def _shared_probe(m_in, w_l, s_l):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    return shared_pure(m_in, w_t, s_t)

  shared_probe_jaxpr = jax.make_jaxpr(_shared_probe)(moe_in_sample, w_leaves_all, s_leaves_all)
  shared_w_indices = batch_split._used_var_indices(shared_probe_jaxpr, slice(1, 1 + n_w))

  def _substitute_weights(w_sub, indices):
    merged = list(w_leaves_all)
    for idx, val in zip(indices, w_sub):
      merged[idx] = val
    return jax.tree_util.tree_unflatten(w_treedef, merged)

  # 2. Trace weight-free post_combine_bwd and discover minimal saved context (`post_ctx_indices`).
  hs_leaves_sample, hs_treedef = jax.tree_util.tree_flatten(h_skip_sample)
  g_x_sds = jax.ShapeDtypeStruct(x_sample.shape, x_sample.dtype, sharding=getattr(jax.typeof(x_sample), "sharding", None))

  def _post_bwd_probe(mlp_sum, hs_l, s_l, g_x):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)

    def _inner(ms_, hsl_):
      hs_t = jax.tree_util.tree_unflatten(hs_treedef, hsl_)
      return post_combine_pure(ms_, hs_t, s_t, w_t)

    _, vjp_fn = jax.vjp(_inner, mlp_sum, hs_l)
    g_ms, g_hsl = batch_split._without_qwix_interception(vjp_fn)(g_x)
    return (g_ms, *g_hsl)

  n_post_cands = 1 + len(hs_leaves_sample)
  post_probe_jaxpr = jax.make_jaxpr(_post_bwd_probe)(moe_in_sample, hs_leaves_sample, s_leaves_all, g_x_sds)
  post_ctx_indices = batch_split._used_var_indices(post_probe_jaxpr, slice(0, n_post_cands))
  all_post_samples = (moe_in_sample, *hs_leaves_sample)
  post_ctx_sample = tuple(all_post_samples[idx] for idx in post_ctx_indices)

  def _post_bwd_fn(post_ctx_sub, s_l, g_x):
    cands = [jnp.zeros(c.shape, c.dtype) for c in all_post_samples]
    for idx, val in zip(post_ctx_indices, post_ctx_sub):
      cands[idx] = val
    return _post_bwd_probe(cands[0], tuple(cands[1:]), s_l, g_x)

  post_combine_bwd_jaxpr = jax.make_jaxpr(_post_bwd_fn)(post_ctx_sample, s_leaves_all, g_x_sds)
  g_mlp_sum_sds = batch_split._aval_to_sds(post_combine_bwd_jaxpr.out_avals[0])
  g_hs_leaves_sds = tuple(batch_split._aval_to_sds(a) for a in post_combine_bwd_jaxpr.out_avals[1:])
  g_h_skip_sds = jax.tree_util.tree_unflatten(hs_treedef, g_hs_leaves_sds)

  # 3. Trace combine_bwd_jaxpr: (g_routed, moe_in_ref, s_leaves) -> g_y_partial or (qval, scale)
  def _combine_bwd_fn(g_routed, m_ref, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    return combine_bwd_pure(g_routed, m_ref, s_t, w_t)

  combine_bwd_jaxpr = jax.make_jaxpr(_combine_bwd_fn)(g_mlp_sum_sds, moe_in_sample, s_leaves_all)
  has_deferred_combine_bwd = len(combine_bwd_jaxpr.out_avals) == 2
  if has_deferred_combine_bwd:
    qval_sds = batch_split._aval_to_sds(combine_bwd_jaxpr.out_avals[0])
    g_y_part_sds = jax.ShapeDtypeStruct(qval_sds.shape, y_part_sample.dtype, sharding=getattr(qval_sds, "sharding", None))
  else:
    g_y_part_sds = batch_split._aval_to_sds(combine_bwd_jaxpr.out_avals[0])

  # 4. Trace experts_subfn under `remat_policy`.
  w_exp_sample = tuple(w_leaves_all[i] for i in exp_w_indices)
  g_exp_lb_sds = batch_split._to_sds(exp_lb_sample) if has_exp_lb else None

  def _exp_diff_and_aux(d_args, o_args):
    if has_x_scale:
      xv_, xs_, rd_, ws_ = d_args
    else:
      xv_, rd_, ws_ = d_args
      xs_ = None
    m_ref_, r_nd_, s_l_, da_, dk_ = o_args
    w_t = _substitute_weights(ws_, exp_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    r_ag = batch_split._merge_inexact(rd_, r_nd_, route_ag_mask, route_ag_treedef)
    y_p, e_lb, cnt = experts_pure(m_ref_, xv_, xs_, r_ag, w_t, s_t, da_, dk_)
    d_outs = (y_p, e_lb) if has_exp_lb else (y_p,)
    return d_outs, cnt

  exp_diff_samples = (
      (x_val_ag_sample, x_scale_ag_sample, route_ag_diff, w_exp_sample)
      if has_x_scale
      else (x_val_ag_sample, route_ag_diff, w_exp_sample)
  )
  exp_other_samples = (moe_in_sample, route_ag_nondiff, s_leaves_all, d_args_sample, d_kwargs_sample)
  exp_cot_samples = (g_y_part_sds, g_exp_lb_sds) if has_exp_lb else (g_y_part_sds,)
  experts_subfn = batch_split._trace_remat_subfn(
      _exp_diff_and_aux,
      exp_diff_samples,
      exp_other_samples,
      exp_cot_samples,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  exp_bwd_out_sds = jax.tree_util.tree_unflatten(
      experts_subfn.bwd_out_treedef, [batch_split._aval_to_sds(a) for a in experts_subfn.bwd_jaxpr.out_avals]
  )
  if has_x_scale:
    g_x_val_ag_sds, _, g_route_ag_diff_sds, exp_bwd_dw_sds = exp_bwd_out_sds
  else:
    g_x_val_ag_sds, g_route_ag_diff_sds, exp_bwd_dw_sds = exp_bwd_out_sds

  # 4b. Optional in-place expert wgrad accumulation (`tgmm_v2` `partial_sum` via `return_wgrad_hooks`).
  probe_box: dict[str, tuple[int, ...]] = {"sub_indices": ()}

  def _probe_acc_hooks(m_ref, xv, xs, r_diff, r_nd, w_sub, s_l, da, dk):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = _substitute_weights(w_sub, exp_w_indices)
    r_ag = batch_split._merge_inexact(r_diff, r_nd, route_ag_mask, route_ag_treedef)
    _, _, _, hooks = experts_pure(m_ref, xv, xs, r_ag, w_t, s_t, da, dk, return_wgrad_hooks=True)
    if not hooks:
      return ()
    id_to_sub_idx = {id(t): idx for idx, t in enumerate(w_sub)}
    sub_indices = []
    seen = set()
    for param_tracer, hook in hooks:
      sub_idx = id_to_sub_idx.get(id(param_tracer))
      if sub_idx is None or sub_idx in seen:
        return ()
      target_sds = exp_bwd_dw_sds[sub_idx]
      if hook.shape != target_sds.shape or hook.dtype != target_sds.dtype:
        return ()
      seen.add(sub_idx)
      sub_indices.append(sub_idx)
    probe_box["sub_indices"] = tuple(sub_indices)
    return ()

  jax.eval_shape(
      _probe_acc_hooks,
      moe_in_sample,
      x_val_ag_sample,
      x_scale_ag_sample,
      route_ag_diff,
      route_ag_nondiff,
      w_exp_sample,
      s_leaves_all,
      d_args_sample,
      d_kwargs_sample,
  )
  exp_accum_sub_indices = probe_box["sub_indices"]
  experts_acc_subfn = None
  if exp_accum_sub_indices:
    dw_accum_sds = tuple(exp_bwd_dw_sds[j] for j in exp_accum_sub_indices)

    def _exp_acc_diff_and_aux(d_args, o_args):
      if has_x_scale:
        xv_, xs_, rd_, ws_ = d_args
      else:
        xv_, rd_, ws_ = d_args
        xs_ = None
      m_ref_, r_nd_, s_l_, da_, dk_ = o_args
      w_t = _substitute_weights(ws_, exp_w_indices)
      s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
      r_ag = batch_split._merge_inexact(rd_, r_nd_, route_ag_mask, route_ag_treedef)
      y_p, e_lb, cnt, hooks = experts_pure(m_ref_, xv_, xs_, r_ag, w_t, s_t, da_, dk_, return_wgrad_hooks=True)
      hook_vals = tuple(h for _, h in hooks)
      d_outs = (y_p, e_lb, hook_vals) if has_exp_lb else (y_p, hook_vals)
      return d_outs, cnt

    exp_acc_cot_samples = (g_y_part_sds, g_exp_lb_sds, dw_accum_sds) if has_exp_lb else (g_y_part_sds, dw_accum_sds)
    experts_acc_subfn = batch_split._trace_remat_subfn(
        _exp_acc_diff_and_aux,
        exp_diff_samples,
        exp_other_samples,
        exp_acc_cot_samples,
        remat_policy=remat_policy,
        prevent_cse=prevent_cse,
        fwd_keep_diff_count=2 if has_exp_lb else 1,
    )

  # 5. Trace dispatch_fwd_jaxpr & dispatch_bwd_jaxpr
  def _dispatch_fwd_fn(m_in, r_diff, r_nd, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    r_in = batch_split._merge_inexact(r_diff, r_nd, route_in_mask, route_in_treedef)
    xv, xs, r_ag = dispatch_pure(m_in, r_in, s_t, w_t)
    r_ag_d, r_ag_nd, _, _ = batch_split._split_inexact(r_ag)
    return (xv, xs, r_ag_d), r_ag_nd

  dispatch_fwd_jaxpr = jax.make_jaxpr(_dispatch_fwd_fn)(moe_in_sample, route_in_diff, route_in_nondiff, s_leaves_all)

  def _dispatch_bwd_fn(m_in, r_diff, r_nd, g_xv, g_rag_d, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    r_in = batch_split._merge_inexact(r_diff, r_nd, route_in_mask, route_in_treedef)
    return dispatch_bwd_pure(m_in, r_in, g_xv, g_rag_d, s_t, w_t)

  dispatch_bwd_jaxpr = jax.make_jaxpr(_dispatch_bwd_fn)(
      moe_in_sample, route_in_diff, route_in_nondiff, g_x_val_ag_sds, g_route_ag_diff_sds, s_leaves_all
  )
  g_moe_in_sds = batch_split._aval_to_sds(dispatch_bwd_jaxpr.out_avals[0])
  g_route_in_diff_sds = tuple(batch_split._aval_to_sds(a) for a in dispatch_bwd_jaxpr.out_avals[1:])

  # 6. Trace shared_subfn under `remat_policy`.
  w_shared_sample = tuple(w_leaves_all[i] for i in shared_w_indices)

  def _shared_diff_and_aux(d_args, o_args):
    m_, ws_ = d_args
    (s_l_,) = o_args
    w_t = _substitute_weights(ws_, shared_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    return shared_pure(m_, w_t, s_t), ()

  shared_subfn = batch_split._trace_remat_subfn(
      _shared_diff_and_aux,
      (moe_in_sample, w_shared_sample),
      (s_leaves_all,),
      g_mlp_sum_sds,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  # 7. Trace attn_route_subfn under `remat_policy`.
  w_attn_sample = tuple(w_leaves_all[i] for i in attn_w_indices)
  g_seq_lb_sds = batch_split._to_sds(seq_lb_sample) if has_seq_lb else None

  def _attn_diff_and_aux(d_args, o_args):
    x_, ws_ = d_args
    s_l_, da_, dk_ = o_args
    w_t = _substitute_weights(ws_, attn_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    hs, m_in, r_in, s_lb = attn_route_pure(x_, w_t, s_t, da_, dk_)
    r_d, r_nd, _, _ = batch_split._split_inexact(r_in)
    d_outs = (hs, m_in, r_d, s_lb) if has_seq_lb else (hs, m_in, r_d)
    return d_outs, r_nd

  attn_cot_samples = (
      (g_h_skip_sds, g_moe_in_sds, g_route_in_diff_sds, g_seq_lb_sds)
      if has_seq_lb
      else (g_h_skip_sds, g_moe_in_sds, g_route_in_diff_sds)
  )
  attn_route_subfn = batch_split._trace_remat_subfn(
      _attn_diff_and_aux,
      (x_sample, w_attn_sample),
      (s_leaves_all, d_args_sample, d_kwargs_sample),
      attn_cot_samples,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  bundle = _InhomogeneousBwdBundle(
      attn_w_indices=attn_w_indices,
      exp_w_indices=exp_w_indices,
      exp_accum_sub_indices=exp_accum_sub_indices,
      shared_w_indices=shared_w_indices,
      post_ctx_indices=post_ctx_indices,
      hs_treedef=hs_treedef,
      route_in_mask=route_in_mask,
      route_in_treedef=route_in_treedef,
      route_ag_mask=route_ag_mask,
      route_ag_treedef=route_ag_treedef,
      has_seq_lb=has_seq_lb,
      has_exp_lb=has_exp_lb,
      has_x_scale=has_x_scale,
      has_bias=has_bias,
      has_deferred_combine_bwd=has_deferred_combine_bwd,
      attn_route_subfn=attn_route_subfn,
      experts_subfn=experts_subfn,
      experts_acc_subfn=experts_acc_subfn,
      shared_subfn=shared_subfn,
      dispatch_fwd=dispatch_fwd_jaxpr,
      dispatch_bwd=dispatch_bwd_jaxpr,
      combine_bwd=combine_bwd_jaxpr,
      post_combine_bwd=post_combine_bwd_jaxpr,
  )
  sample_sds = (y_part_sample, shared_sample, h_skip_sample)
  return bundle, sample_sds


def _select_post_ctx(indices: tuple[int, ...], mlp_sum, h_skip) -> tuple:
  if not indices:
    return ()
  hs_leaves = jax.tree_util.tree_leaves(h_skip)
  cands = (mlp_sum, *hs_leaves)
  return tuple(cands[idx] for idx in indices)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 9, 10, 11, 12))
def _custom_vjp_nested_batch_split(
    graphdefs,
    x0_in,
    x1_in,
    params_leaves,
    state_leaves,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    length,
    ctxs,
    sizes,
    bwd_cache,
):
  """Runs `length` scanned inhomogeneous blocks with Option B batch-split + FSDP prefetch."""
  out, _ = _custom_vjp_nested_batch_split_fwd(
      graphdefs,
      x0_in,
      x1_in,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
      length,
      ctxs,
      sizes,
      bwd_cache,
  )
  return out


def _custom_vjp_nested_batch_split_fwd(
    graphdefs,
    x0_in,
    x1_in,
    params_leaves,
    state_leaves,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    length,
    ctxs,
    sizes,
    bwd_cache: Callable,
):
  """Forward pass: nested `[entry: 0] -> [mid: 1..n-2] -> [exit: n-1]` per-sublayer Option B schedule."""
  num_groups = len(graphdefs)
  offload_res = bool(ctxs and ctxs[0].offload_layer_inputs)
  setups = [
      _setup_one_group(g, p, s, mb0_args_leaves, mb0_kwargs_leaves, mb1_args_leaves, mb1_kwargs_leaves, ctx)
      for g, p, s, ctx in zip(graphdefs, params_leaves, state_leaves, ctxs)
  ]
  (
      params,
      state,
      gather,
      _,
      mb0_consts_list,
      mb1_consts_list,
      attn_route_pures,
      dispatch_pures,
      dispatch_bwd_pures,
      experts_pures,
      shared_pures,
      combine_pures,
      post_combine_pures,
      combine_bwd_pures,
      finalize_bias_pures,
  ) = zip(*setups)

  mb0_da, mb0_dk = mb0_consts_list[0]
  mb1_da, mb1_dk = mb1_consts_list[0]
  w_templates = [fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(p, 0), 0) for p in params]
  s_templates = [fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(s, 0), 0) for s in state]

  # Capture per-group backward jaxprs up front (while Qwix/qpl interceptors are active) and get initial carry shapes.
  bundles = []
  sample_sds_per_group = []
  for i in range(num_groups):
    w_sample_i = gather[i](w_templates[i])
    b_i, sds_i = _build_group_bwd_bundle(
        x_sample=x0_in,
        w_sample=w_sample_i,
        s_sample=s_templates[i],
        d_args_sample=mb0_da,
        d_kwargs_sample=mb0_dk,
        attn_route_pure=attn_route_pures[i],
        dispatch_pure=dispatch_pures[i],
        dispatch_bwd_pure=dispatch_bwd_pures[i],
        experts_pure=experts_pures[i],
        shared_pure=shared_pures[i],
        post_combine_pure=post_combine_pures[i],
        combine_bwd_pure=combine_bwd_pures[i],
        finalize_bias_pure=finalize_bias_pures[i],
        remat_policy=ctxs[i].remat_policy,
        prevent_cse=ctxs[i].prevent_cse,
    )
    bundles.append(b_i)
    sample_sds_per_group.append(sds_i)
  bwd_cache(lambda: tuple(bundles))

  def _zeros_from_sds(tree):
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=getattr(s, "sharding", None)), tree)

  yp1_init, sh1_init, hs1_init = _zeros_from_sds(sample_sds_per_group[-1])

  def _fwd_attn_route(i: int, x_k, w_k, w_leaves, s_k, da, dk):
    b_i = bundles[i]
    if b_i.attn_route_subfn.has_saved:
      w_sub = tuple(w_leaves[idx] for idx in b_i.attn_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      d_outs, r_nd, saved = batch_split._eval_remat_fwd(b_i.attn_route_subfn, (x_k, w_sub), (s_l, da, dk))
      hs, m_in, r_d = d_outs[:3]
      s_lb = d_outs[3] if b_i.has_seq_lb else None
      r_in = batch_split._merge_inexact(r_d, r_nd, b_i.route_in_mask, b_i.route_in_treedef)
      return hs, m_in, r_in, s_lb, saved
    hs, m_in, r_in, s_lb = attn_route_pures[i](x_k, w_k, s_k, da, dk)
    return hs, m_in, r_in, s_lb, ()

  def _fwd_experts(i: int, subfn: batch_split._RematSubfn, m_in, xv, xs, r_ag, w_k, w_leaves, s_k, da, dk):
    b_i = bundles[i]
    if subfn.has_saved:
      w_sub = tuple(w_leaves[idx] for idx in b_i.exp_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      r_d, r_nd, _, _ = batch_split._split_inexact(r_ag)
      d_args = (xv, xs, r_d, w_sub) if b_i.has_x_scale else (xv, r_d, w_sub)
      o_args = (m_in, r_nd, s_l, da, dk)
      d_outs, cnt, saved = batch_split._eval_remat_fwd(subfn, d_args, o_args)
      y_part = d_outs[0]
      e_lb = d_outs[1] if b_i.has_exp_lb else None
      return y_part, e_lb, cnt, saved
    y_part, e_lb, cnt = experts_pures[i](m_in, xv, xs, r_ag, w_k, s_k, da, dk)
    return y_part, e_lb, cnt, ()

  def _fwd_shared(i: int, m_in, w_k, w_leaves, s_k):
    b_i = bundles[i]
    if b_i.shared_subfn.has_saved:
      w_sub = tuple(w_leaves[idx] for idx in b_i.shared_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      sh_out, _, saved = batch_split._eval_remat_fwd(b_i.shared_subfn, (m_in, w_sub), (s_l,))
      return sh_out, saved
    return shared_pures[i](m_in, w_k, s_k), ()

  def _run_fwd_sublayer(i: int, i_prev: int, x0_k, yp1_prev, sh1_prev, hs1_prev, w_k, s_k, is_first=None):
    with jax.named_scope(f"batch_split_inhom_fwd_g{i}"):
      b_i = bundles[i]
      exp_mb0_subfn = b_i.experts_acc_subfn if b_i.experts_acc_subfn is not None else b_i.experts_subfn
      w_leaves = jax.tree_util.tree_leaves(w_k)
      # F1: mb1 combine(prev) [SC0 RS, weight-free] || mb0 attn+route(curr) [TC].
      routed_1_prev = combine_pures[i_prev](yp1_prev, sh1_prev, s_templates[i_prev], w_templates[i_prev])
      hs0, moe_in_0, route_in_0, seq_lb_0, attn_sv_0 = _fwd_attn_route(i, x0_k, w_k, w_leaves, s_k, mb0_da, mb0_dk)
      routed_1_prev, hs0, moe_in_0_f2, route_in_0_f2 = batch_split._phase_barrier(
          (routed_1_prev, hs0, moe_in_0, route_in_0)
      )
      mlp_sum_1_prev = routed_1_prev + sh1_prev
      x1_combined = post_combine_pures[i_prev](mlp_sum_1_prev, hs1_prev, s_templates[i_prev], w_templates[i_prev])
      x1_k = jnp.where(is_first, x1_in, x1_combined) if is_first is not None else x1_combined
      post_ctx_1_prev = _select_post_ctx(bundles[i_prev].post_ctx_indices, mlp_sum_1_prev, hs1_prev)

      # F2: mb0 dispatch(curr) [SC0 AG] || mb1 attn+route(curr) [TC].
      x_val_0, x_scale_0, route_ag_0 = dispatch_pures[i](moe_in_0_f2, route_in_0_f2, s_k, w_templates[i])
      hs1, moe_in_1, route_in_1, seq_lb_1, attn_sv_1 = _fwd_attn_route(i, x1_k, w_k, w_leaves, s_k, mb1_da, mb1_dk)
      x_val_0, x_scale_0, route_ag_0, hs1, moe_in_1_f3, route_in_1_f3 = batch_split._phase_barrier(
          (x_val_0, x_scale_0, route_ag_0, hs1, moe_in_1, route_in_1)
      )
      # F3: mb1 dispatch(curr) [SC0 AG] || mb0 experts(curr) [TC].
      x_val_1, x_scale_1, route_ag_1 = dispatch_pures[i](moe_in_1_f3, route_in_1_f3, s_k, w_templates[i])
      y_part_0, exp_lb_0, cnt_0, exp_sv_0 = _fwd_experts(
          i, exp_mb0_subfn, moe_in_0_f2, x_val_0, x_scale_0, route_ag_0, w_k, w_leaves, s_k, mb0_da, mb0_dk
      )
      y_part_0, moe_in_0_f4, x_val_1, x_scale_1, route_ag_1, moe_in_1_f4 = batch_split._phase_barrier(
          (y_part_0, moe_in_0_f2, x_val_1, x_scale_1, route_ag_1, moe_in_1_f3)
      )
      # F4: mb0 combine(curr) [SC0 RS] + mb0 shared(curr) [TC] || mb1 experts(curr) [TC] + mb1 shared(curr) [TC].
      routed_0 = combine_pures[i](y_part_0, moe_in_0_f4, s_k, w_templates[i])
      shared_0, sh_sv_0 = _fwd_shared(i, moe_in_0_f4, w_k, w_leaves, s_k)
      y_part_1, exp_lb_1, cnt_1, exp_sv_1 = _fwd_experts(
          i, b_i.experts_subfn, moe_in_1_f4, x_val_1, x_scale_1, route_ag_1, w_k, w_leaves, s_k, mb1_da, mb1_dk
      )
      shared_1, sh_sv_1 = _fwd_shared(i, moe_in_1_f4, w_k, w_leaves, s_k)
      routed_0, shared_0, y_part_1, shared_1 = batch_split._phase_barrier((routed_0, shared_0, y_part_1, shared_1))
      mlp_sum_0 = routed_0 + shared_0
      x0_next = post_combine_pures[i](mlp_sum_0, hs0, s_k, w_templates[i])
      post_ctx_0 = _select_post_ctx(b_i.post_ctx_indices, mlp_sum_0, hs0)

      lb_k = batch_split._combine_layer_lb(seq_lb_0, seq_lb_1, exp_lb_0, exp_lb_1)
      summed_cnt_k = None if cnt_0 is None else (cnt_0 + cnt_1)
      bias_k = finalize_bias_pures[i](summed_cnt_k, s_k, w_templates[i])
      res_k = (x0_k, x1_k, moe_in_0, route_in_0, moe_in_1, route_in_1, post_ctx_0, post_ctx_1_prev)
      saved_k = (attn_sv_0, attn_sv_1, exp_sv_0, exp_sv_1, sh_sv_0, sh_sv_1)
      return (x0_next, y_part_1, shared_1, hs1), res_k, saved_k, lb_k, bias_k

  def _sample_res_sds(i: int):
    i_prev = (i - 1) % num_groups
    w_s = gather[i](w_templates[i])
    return jax.eval_shape(
        lambda x0_, yp1_, sh1_, hs1_: _run_fwd_sublayer(i, i_prev, x0_, yp1_, sh1_, hs1_, w_s, s_templates[i], None)[1],
        x0_in,
        yp1_init,
        sh1_init,
        hs1_init,
    )

  res_buf_init = (
      tuple(
          batch_split._tree_host_buffer(_zeros_from_sds(_sample_res_sds(i)), length * sizes[i]) for i in range(num_groups)
      )
      if offload_res
      else ()
  )

  def body(carry, xs):
    act_carry, w, res_buf = carry
    b, b_state, b_params = xs
    res_groups, saved_groups, lb_groups, bias_groups = [], [], [], []
    for i in range(num_groups):
      n = sizes[i]
      i_prev_boundary = (i - 1) % num_groups
      res_buf_i = res_buf[i] if offload_res else None
      if i + 1 < num_groups:
        nxt_first, gather_nxt = fsdp_prefetch._slice_at(b_params[i + 1], 0), gather[i + 1]
      else:
        nxt_first = fsdp_prefetch._slice_at(_index_at(params[0], (b + 1) % length), 0)
        gather_nxt = gather[0]

      parts_res, parts_saved, parts_lb, parts_bias = [], [], [], []

      def _run_single_trip(
          act_c,
          w_c,
          rbuf_i,
          k_idx: int,
          i_prev_step: int,
          p_next_param,
          gather_next_fn,
          check_first: bool,
          i: int = i,
          n: int = n,
      ):
        is_first_flag = (b == 0) if check_first else None

        def single_step(c_in, xs_in):
          ac, wc, rb = c_in
          s_step, p_nxt = xs_in
          w_nxt = gather_next_fn(p_nxt)
          x0_c, yp1_c, sh1_c, hs1_c = ac
          ac_out, r_step, sv_step, lb_step, bias_step = _run_fwd_sublayer(
              i, i_prev_step, x0_c, yp1_c, sh1_c, hs1_c, wc, s_step, is_first_flag
          )
          if offload_res:
            rb = batch_split._tree_host_write(rb, r_step, b * n + k_idx)
            return (ac_out, wc, rb), (w_nxt, None, sv_step, lb_step, bias_step)
          return (ac_out, wc, rb), (w_nxt, r_step, sv_step, lb_step, bias_step)

        step_xs = fsdp_prefetch._expand0((fsdp_prefetch._slice_at(b_state[i], k_idx), p_next_param))
        (act_out, _, rbuf_out), (w_nxt_stacked, r_st, sv_st, lb_st, bias_st) = _sublayer_scan(
            single_step, (act_c, w_c, rbuf_i), step_xs
        )
        return act_out, fsdp_prefetch._slice_at(w_nxt_stacked, 0), rbuf_out, r_st, sv_st, lb_st, bias_st

      if n == 1:
        act_carry, w, res_buf_i, r_0, sv_0, lb_0, bias_0 = _run_single_trip(
            act_carry, w, res_buf_i, 0, i_prev_boundary, nxt_first, gather_nxt, check_first=(i == 0)
        )
        if not offload_res:
          parts_res.append(r_0)
        parts_saved.append(sv_0)
        parts_lb.append(lb_0)
        parts_bias.append(bias_0)
      else:
        # Entry (k = 0): predecessor is previous group `i_prev_boundary`, prefetches k = 1 of group `i`.
        act_carry, w, res_buf_i, r_0, sv_0, lb_0, bias_0 = _run_single_trip(
            act_carry,
            w,
            res_buf_i,
            0,
            i_prev_boundary,
            fsdp_prefetch._slice_at(b_params[i], 1),
            gather[i],
            check_first=(i == 0),
        )
        if not offload_res:
          parts_res.append(r_0)
        parts_saved.append(sv_0)
        parts_lb.append(lb_0)
        parts_bias.append(bias_0)

        if n >= 3:
          ordered = n - 2 >= 2

          def mid_step(c_mid, k_mid, i=i, n=n, p=b_params[i], s=b_state[i], ordered=ordered):
            ac, wc, rb = c_mid
            p_next = _index_at(p, k_mid + 1)
            if ordered:
              ac, p_next = jax.lax.optimization_barrier((ac, p_next))
            w_next = gather[i](p_next)
            x0_c, yp1_c, sh1_c, hs1_c = ac
            ac_out, r_k, sv_k, lb_k, bias_k = _run_fwd_sublayer(
                i, i, x0_c, yp1_c, sh1_c, hs1_c, wc, _index_at(s, k_mid), None
            )
            if offload_res:
              rb = batch_split._tree_host_write(rb, r_k, b * n + k_mid)
              return (ac_out, w_next, rb), (None, sv_k, lb_k, bias_k)
            return (ac_out, w_next, rb), (r_k, sv_k, lb_k, bias_k)

          (act_carry, w, res_buf_i), (r_mid, sv_mid, lb_mid, bias_mid) = _sublayer_scan(
              mid_step, (act_carry, w, res_buf_i), jnp.arange(1, n - 1), unroll=2 if ordered else 1
          )
          if not offload_res:
            parts_res.append(r_mid)
          parts_saved.append(sv_mid)
          parts_lb.append(lb_mid)
          parts_bias.append(bias_mid)

        # Exit (k = n - 1): predecessor is same group `i`, prefetches first sub-layer of next group.
        act_carry, w, res_buf_i, r_last, sv_last, lb_last, bias_last = _run_single_trip(
            act_carry, w, res_buf_i, n - 1, i, nxt_first, gather_nxt, check_first=False
        )
        if not offload_res:
          parts_res.append(r_last)
        parts_saved.append(sv_last)
        parts_lb.append(lb_last)
        parts_bias.append(bias_last)

      if offload_res:
        res_buf = res_buf[:i] + (res_buf_i,) + res_buf[i + 1 :]
      else:
        res_groups.append(_concat0(*parts_res))
      saved_groups.append(_concat0(*parts_saved))
      lb_groups.append(None if parts_lb[0] is None else _concat0(*parts_lb))
      bias_groups.append(None if parts_bias[0] is None else _concat0(*parts_bias))

    return (act_carry, w, res_buf), (tuple(res_groups), tuple(saved_groups), tuple(lb_groups), tuple(bias_groups))

  w_first = gather[0](w_templates[0])
  xs = (jnp.arange(length, dtype=jnp.int32), tuple(state), tuple(params))
  ((x0_final, yp1_last, sh1_last, hs1_last), _, res_buf), (res_all, saved_all, lb_all, bias_all) = jax.lax.scan(
      body, ((x0_in, yp1_init, sh1_init, hs1_init), w_first, res_buf_init), xs
  )

  # Epilogue: complete mb1 combine + post_combine for the final sub-layer (group `num_groups - 1`, weight-free).
  last_g = num_groups - 1
  routed_1_last = combine_pures[last_g](yp1_last, sh1_last, s_templates[last_g], w_templates[last_g])
  mlp_sum_1_last = routed_1_last + sh1_last
  x1_final = post_combine_pures[last_g](mlp_sum_1_last, hs1_last, s_templates[last_g], w_templates[last_g])
  post_ctx_1_last = _select_post_ctx(bundles[last_g].post_ctx_indices, mlp_sum_1_last, hs1_last)

  if offload_res:
    res_all = res_buf
  residuals = (
      res_all,
      saved_all,
      post_ctx_1_last,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
  )
  return (x0_final, x1_final, lb_all, bias_all), residuals


def _custom_vjp_nested_batch_split_bwd(graphdefs, length, ctxs, sizes, bwd_cache: Callable, res, g_out):
  """Backward pass: nested `[entry: n-1] -> [mid: n-2..1] -> [exit: 0]` reverse Option B schedule."""
  (
      res_all,
      saved_all,
      post_ctx_1_last,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
  ) = res
  g_x0_final, g_x1_final, g_lb_all, _ = g_out

  num_groups = len(graphdefs)
  offload_res = bool(ctxs and ctxs[0].offload_layer_inputs)
  bundles: tuple[_InhomogeneousBwdBundle, ...] = bwd_cache(None)

  setups = [
      _setup_one_group(g, p, s, mb0_args_leaves, mb0_kwargs_leaves, mb1_args_leaves, mb1_kwargs_leaves, ctx)
      for g, p, s, ctx in zip(graphdefs, params_leaves, state_leaves, ctxs)
  ]
  params, state, gather, rs_fns, mb0_consts_list, mb1_consts_list, *_ = zip(*setups)
  mb0_da, mb0_dk = mb0_consts_list[0]
  mb1_da, mb1_dk = mb1_consts_list[0]

  g_x0_final, g_x1_final, params = jax.lax.optimization_barrier((g_x0_final, g_x1_final, tuple(params)))

  def _run_post_combine_bwd(i: int, post_ctx, s_k, g_x):
    b_i = bundles[i]
    s_l = jax.tree_util.tree_leaves(s_k)
    flat_out = batch_split._eval_jaxpr_flat(b_i.post_combine_bwd, post_ctx, s_l, g_x)
    g_mlp_sum = flat_out[0]
    g_hs = jax.tree_util.tree_unflatten(b_i.hs_treedef, flat_out[1:])
    return g_mlp_sum, g_mlp_sum, g_hs

  def _run_combine_bwd(i: int, g_routed, m_ref, s_k):
    s_l = jax.tree_util.tree_leaves(s_k)
    out = batch_split._eval_jaxpr_flat(bundles[i].combine_bwd, g_routed, m_ref, s_l)
    if bundles[i].has_deferred_combine_bwd:
      return (out[0], out[1])
    return out[0]

  def _run_dispatch_fwd(i: int, m_in, r_in, s_k):
    b_i = bundles[i]
    r_diff, r_nd, _, _ = batch_split._split_inexact(r_in)
    s_l = jax.tree_util.tree_leaves(s_k)
    flat_out = batch_split._eval_jaxpr_flat(b_i.dispatch_fwd, m_in, r_diff, r_nd, s_l)
    pos = 0
    xv = flat_out[pos]
    pos += 1
    xs = flat_out[pos] if b_i.has_x_scale else None
    if b_i.has_x_scale:
      pos += 1
    n_rd = sum(b_i.route_ag_mask)
    n_rnd = len(b_i.route_ag_mask) - n_rd
    rag_diff = tuple(flat_out[pos : pos + n_rd])
    pos += n_rd
    rag_nd = tuple(flat_out[pos : pos + n_rnd])
    return (xv, xs, rag_diff, rag_nd)

  def _run_dispatch_bwd(i: int, m_in, r_in, g_xv, g_xs, g_rag_diff, s_k):
    del g_xs
    r_diff, r_nd, _, _ = batch_split._split_inexact(r_in)
    s_l = jax.tree_util.tree_leaves(s_k)
    flat_out = batch_split._eval_jaxpr_flat(bundles[i].dispatch_bwd, m_in, r_diff, r_nd, g_xv, g_rag_diff, s_l)
    return flat_out[0], tuple(flat_out[1:])

  def _run_experts_bwd(i: int, m_ref, disp, w_leaves, s_k, da, dk, exp_saved, g_yp, g_elb, dw_partial_from_mb1=None):
    if isinstance(g_yp, tuple):
      g_yp = (g_yp[0].astype(jnp.float32) * g_yp[1]).astype(m_ref.dtype)
    b_i = bundles[i]
    xv, xs, rag_diff, rag_nd = disp
    w_sub = tuple(w_leaves[idx] for idx in b_i.exp_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    d_args = (xv, xs, rag_diff, w_sub) if b_i.has_x_scale else (xv, rag_diff, w_sub)
    o_args = (m_ref, rag_nd, s_l, da, dk)
    if dw_partial_from_mb1 is not None and b_i.experts_acc_subfn is not None:
      dw_part = tuple(dw_partial_from_mb1[j] for j in b_i.exp_accum_sub_indices)
      cot = (g_yp, g_elb, dw_part) if b_i.has_exp_lb else (g_yp, dw_part)
      grads = batch_split._eval_remat_bwd(b_i.experts_acc_subfn, d_args, o_args, exp_saved, cot)
    else:
      cot = (g_yp, g_elb) if b_i.has_exp_lb else (g_yp,)
      grads = batch_split._eval_remat_bwd(b_i.experts_subfn, d_args, o_args, exp_saved, cot)
    if b_i.has_x_scale:
      g_xv, g_xs, g_rag_diff, dw_sub = grads
    else:
      g_xv, g_rag_diff, dw_sub = grads
      g_xs = None
    return (g_xv, g_xs, g_rag_diff), dw_sub

  def _run_shared_bwd(i: int, m_in, w_leaves, s_k, sh_saved, g_sh):
    b_i = bundles[i]
    w_sub = tuple(w_leaves[idx] for idx in b_i.shared_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    g_m, dw_sub = batch_split._eval_remat_bwd(b_i.shared_subfn, (m_in, w_sub), (s_l,), sh_saved, g_sh)
    return g_m, dw_sub

  def _run_attn_route_bwd(i: int, x_in_k, w_leaves, s_k, da, dk, attn_saved, g_hs, g_m_in, g_r_diff, g_slb):
    b_i = bundles[i]
    w_sub = tuple(w_leaves[idx] for idx in b_i.attn_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    cot = (g_hs, g_m_in, g_r_diff, g_slb) if b_i.has_seq_lb else (g_hs, g_m_in, g_r_diff)
    g_x, dw_sub = batch_split._eval_remat_bwd(b_i.attn_route_subfn, (x_in_k, w_sub), (s_l, da, dk), attn_saved, cot)
    return g_x, dw_sub

  def _assemble_dw(i: int, w_leaves, dw_exp_0, dw_exp_1, dw_sh_0, dw_sh_1, dw_attn_0, dw_attn_1):
    b_i = bundles[i]
    n_w_total = len(w_leaves)
    accum: list[Any] = [None] * n_w_total
    exp_acc_set = frozenset(b_i.exp_accum_sub_indices)

    def _add_into(indices, g0_tuple, g1_tuple, acc_skip=frozenset()):
      for sub_i, (idx, g0, g1) in enumerate(zip(indices, g0_tuple, g1_tuple)):
        g_sum = g0 if sub_i in acc_skip else (g0 + g1)
        accum[idx] = g_sum if accum[idx] is None else (accum[idx] + g_sum)

    _add_into(b_i.exp_w_indices, dw_exp_0, dw_exp_1, exp_acc_set)
    _add_into(b_i.shared_w_indices, dw_sh_0, dw_sh_1)
    _add_into(b_i.attn_w_indices, dw_attn_0, dw_attn_1)
    for idx in range(n_w_total):
      if accum[idx] is None:
        accum[idx] = jnp.zeros_like(w_leaves[idx])
    return jax.tree_util.tree_unflatten(ctxs[i].params_treedef, accum)

  def _bwd_sublayer(
      i: int,
      i_prev: int,
      bwd_act_carry,
      g_lb_k,
      w_k,
      s_k,
      res_k,
      saved_k,
      prev_moe_in_1,
      prev_route_in_1,
      s_prev,
  ):
    """Runs Option B phases B4 -> B3 -> B2 -> B1 for one sub-layer of group `i`."""
    with jax.named_scope(f"batch_split_inhom_bwd_g{i}"):
      g_x0_out, g_sh1_curr, g_hs1_curr, g_yp1_curr, disp_1_curr = bwd_act_carry
      x0_k, x1_k, moe_in_0, route_in_0, moe_in_1, route_in_1, post_ctx_0, post_ctx_1_prev = res_k
      attn_sv_0, attn_sv_1, exp_sv_0, exp_sv_1, sh_sv_0, sh_sv_1 = saved_k
      w_leaves = jax.tree_util.tree_leaves(w_k)
      b_i = bundles[i]
      g_lb_half = None if g_lb_k is None else (0.5 * g_lb_k)
      g_exp_lb = g_lb_half if b_i.has_exp_lb else None
      g_seq_lb = g_lb_half if b_i.has_seq_lb else None

      # B4: mb0 post_combine_bwd + SC0 AG (combine_bwd + dispatch_remat) || TC (mb1 experts_bwd + mb1/mb0 shared_bwd).
      g_routed_0, g_sh0, g_hs0 = _run_post_combine_bwd(i, post_ctx_0, s_k, g_x0_out)
      g_yp0 = _run_combine_bwd(i, g_routed_0, moe_in_0, s_k)
      disp_0 = _run_dispatch_fwd(i, moe_in_0, route_in_0, s_k)
      g_disp_1, dw_exp_1 = _run_experts_bwd(
          i, moe_in_1, disp_1_curr, w_leaves, s_k, mb1_da, mb1_dk, exp_sv_1, g_yp1_curr, g_exp_lb
      )
      g_moe_1_sh, dw_sh_1 = _run_shared_bwd(i, moe_in_1, w_leaves, s_k, sh_sv_1, g_sh1_curr)
      g_moe_0_sh, dw_sh_0 = _run_shared_bwd(i, moe_in_0, w_leaves, s_k, sh_sv_0, g_sh0)
      g_yp0, disp_0, g_disp_1, dw_exp_1, g_moe_1_sh, g_moe_0_sh = batch_split._phase_barrier(
          (g_yp0, disp_0, g_disp_1, dw_exp_1, g_moe_1_sh, g_moe_0_sh)
      )

      # B3: SC0 RS (mb1 dispatch_bwd) || TC (mb0 experts_bwd accumulating onto dw_exp_1 in place).
      g_moe_1_rt, g_route_1_diff = _run_dispatch_bwd(i, moe_in_1, route_in_1, *g_disp_1, s_k)
      g_disp_0, dw_exp_0 = _run_experts_bwd(
          i, moe_in_0, disp_0, w_leaves, s_k, mb0_da, mb0_dk, exp_sv_0, g_yp0, g_exp_lb, dw_partial_from_mb1=dw_exp_1
      )
      g_moe_1_rt, g_route_1_diff, g_disp_0, dw_exp_0 = batch_split._phase_barrier(
          (g_moe_1_rt, g_route_1_diff, g_disp_0, dw_exp_0)
      )

      # B2: SC0 RS (mb0 dispatch_bwd) || TC (mb1 attn_route_bwd).
      g_moe_0_rt, g_route_0_diff = _run_dispatch_bwd(i, moe_in_0, route_in_0, *g_disp_0, s_k)
      g_moe_1 = g_moe_1_rt + g_moe_1_sh
      g_x1_in, dw_attn_1 = _run_attn_route_bwd(
          i, x1_k, w_leaves, s_k, mb1_da, mb1_dk, attn_sv_1, g_hs1_curr, g_moe_1, g_route_1_diff, g_seq_lb
      )
      g_moe_0_rt, g_route_0_diff, g_x1_in, dw_attn_1 = batch_split._phase_barrier(
          (g_moe_0_rt, g_route_0_diff, g_x1_in, dw_attn_1)
      )

      # B1: mb1 post_combine_bwd(prev) + SC0 AG (combine_bwd(prev) + dispatch_remat(prev)) || TC (mb0 attn_route_bwd).
      g_routed_1_prev, g_sh1_prev, g_hs1_prev = _run_post_combine_bwd(i_prev, post_ctx_1_prev, s_prev, g_x1_in)
      g_yp1_prev = _run_combine_bwd(i_prev, g_routed_1_prev, prev_moe_in_1, s_prev)
      disp_1_prev = _run_dispatch_fwd(i_prev, prev_moe_in_1, prev_route_in_1, s_prev)

      g_moe_0 = g_moe_0_rt + g_moe_0_sh
      g_x0_in, dw_attn_0 = _run_attn_route_bwd(
          i, x0_k, w_leaves, s_k, mb0_da, mb0_dk, attn_sv_0, g_hs0, g_moe_0, g_route_0_diff, g_seq_lb
      )
      g_yp1_prev, disp_1_prev, g_x0_in, dw_attn_0 = batch_split._phase_barrier(
          (g_yp1_prev, disp_1_prev, g_x0_in, dw_attn_0)
      )

      dw_k = _assemble_dw(i, w_leaves, dw_exp_0, dw_exp_1, dw_sh_0, dw_sh_1, dw_attn_0, dw_attn_1)
      next_act_carry = (g_x0_in, g_sh1_prev, g_hs1_prev, g_yp1_prev, disp_1_prev)
      return next_act_carry, g_x1_in, dw_k

  def deferred(i: int, dw):
    leaves, treedef = jax.tree_util.tree_flatten(dw)
    flags = rs_fns[i].tag_flags(leaves)
    return jax.tree_util.tree_unflatten(treedef, [x if f else jnp.zeros((0,), x.dtype) for x, f in zip(leaves, flags)])

  def write(d_acc, i: int, dw, b, k, which: str):
    acc_leaves, treedef = jax.tree_util.tree_flatten(d_acc[i])
    d_leaves = jax.tree_util.tree_leaves(dw)
    flags = rs_fns[i].tag_flags(d_leaves)
    out = [_update_at(a, d, b, k) if (which == "tagged") == f else a for a, d, f in zip(acc_leaves, d_leaves, flags)]
    return d_acc[:i] + (jax.tree_util.tree_unflatten(treedef, out),) + d_acc[i + 1 :]

  def flush(d_acc, owner, pend):
    i, b, k = owner
    return write(d_acc, i, rs_fns[i](pend, "tagged"), b, k, "tagged")

  pending_avals = {}

  def pending_zeros(i: int):
    if i not in pending_avals:
      b_i = bundles[i]
      w0 = gather[i](jax.tree.map(lambda x: x[0, 0], params[i]))
      w_leaves = jax.tree_util.tree_leaves(w0)
      exp_avals = b_i.experts_subfn.bwd_jaxpr.out_avals
      dw_exp_sds = tuple(batch_split._aval_to_sds(a) for a in exp_avals[len(exp_avals) - len(b_i.exp_w_indices) :])
      dw_sh_sds = tuple(batch_split._aval_to_sds(a) for a in b_i.shared_subfn.bwd_jaxpr.out_avals[1:])
      dw_attn_sds = tuple(batch_split._aval_to_sds(a) for a in b_i.attn_route_subfn.bwd_jaxpr.out_avals[1:])
      pending_avals[i] = jax.eval_shape(
          lambda wl, de, ds, da: deferred(i, rs_fns[i](_assemble_dw(i, wl, de, de, ds, ds, da, da), "untagged")),
          w_leaves,
          dw_exp_sds,
          dw_sh_sds,
          dw_attn_sds,
      )
    return jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype, out_sharding=s.sharding), pending_avals[i])

  def _read_res(i: int, b, k, b_res_i=None):
    if offload_res:
      return batch_split._tree_host_read(res_all[i], b * sizes[i] + k)
    if b_res_i is not None:
      return _index_at(b_res_i, k) if isinstance(k, jax.Array) else fsdp_prefetch._slice_at(b_res_i, k)
    return fsdp_prefetch._slice_at(_index_at(res_all[i], b), k)

  # Prologue: mb1 post_combine_bwd + combine_bwd + dispatch_remat for the final sub-layer `(length - 1, last_g, last_k)`.
  last_g = num_groups - 1
  last_k = sizes[last_g] - 1
  s_last = fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(state[last_g], length - 1), last_k)
  res_last = _read_res(last_g, length - 1, last_k)
  g_routed_1_init, g_sh1_init, g_hs1_init = _run_post_combine_bwd(last_g, post_ctx_1_last, s_last, g_x1_final)
  g_yp1_init = _run_combine_bwd(last_g, g_routed_1_init, res_last[4], s_last)
  disp_1_init = _run_dispatch_fwd(last_g, res_last[4], res_last[5], s_last)
  act_carry_init = (g_x0_final, g_sh1_init, g_hs1_init, g_yp1_init, disp_1_init)

  def body(carry, xs):
    act_carry, g_x1_latest, w, d_acc, pend = carry
    b, b_res, b_saved, b_g_lb, b_state, b_params = xs
    owner = (0, (b + 1) % length, 0)

    for i in reversed(range(num_groups)):
      n = sizes[i]
      i_prev_boundary = (i - 1) % num_groups
      if i > 0:
        prv_last_param = fsdp_prefetch._slice_at(b_params[i - 1], sizes[i - 1] - 1)
        prv_last_state = fsdp_prefetch._slice_at(b_state[i - 1], sizes[i - 1] - 1)
        prv_last_res = _read_res(i - 1, b, sizes[i - 1] - 1, None if offload_res else b_res[i - 1])
        gather_prv = gather[i - 1]
      else:
        b_clamped = jnp.maximum(b - 1, 0)
        prv_last_param = fsdp_prefetch._slice_at(_index_at(params[-1], (b - 1) % length), sizes[-1] - 1)
        prv_last_state = fsdp_prefetch._slice_at(_index_at(state[-1], b_clamped), sizes[-1] - 1)
        prv_last_res = _read_res(num_groups - 1, b_clamped, sizes[-1] - 1)
        gather_prv = gather[-1]

      def run_boundary(
          c_in,
          k_idx: int,
          i_prev_step: int,
          p_prev,
          s_prev_step,
          prev_m1,
          prev_r1,
          gather_fn_prev,
          cur_owner,
          cur_pend,
          i: int = i,
          n: int = n,
      ):
        ac_in, gx1_in, w_in, d_acc_in = c_in
        res_arg = b * n + k_idx if offload_res else fsdp_prefetch._slice_at(b_res[i], k_idx)
        sv_arg = fsdp_prefetch._slice_at(b_saved[i], k_idx)
        g_lb_step = None if b_g_lb[i] is None else fsdp_prefetch._slice_at(b_g_lb[i], k_idx)
        s_step = fsdp_prefetch._slice_at(b_state[i], k_idx)
        xs_step = fsdp_prefetch._expand0((res_arg, sv_arg, g_lb_step, s_step, p_prev, s_prev_step, prev_m1, prev_r1))

        def step(sc, s_xs):
          ac_s, gx1_s, w_s, da_s, p_in, _ = sc
          del gx1_s
          r_k, sv_k, glb_k, st_k, pp_k, sp_k, pm1_k, pr1_k = s_xs
          if offload_res:
            r_k = batch_split._tree_host_read(res_all[i], r_k)
          da_s = flush(da_s, cur_owner, p_in)
          w_prv = gather_fn_prev(pp_k)
          ac_next, gx1_next, dw_k = _bwd_sublayer(i, i_prev_step, ac_s, glb_k, w_s, st_k, r_k, sv_k, pm1_k, pr1_k, sp_k)
          dw_k = rs_fns[i](dw_k, "untagged")
          da_s = write(da_s, i, dw_k, b, k_idx, "untagged")
          return (ac_next, gx1_next, w_s, da_s, p_in, deferred(i, dw_k)), w_prv

        sc_init = (ac_in, gx1_in, w_in, d_acc_in, cur_pend, pending_zeros(i))
        (ac_out, gx1_out, _, d_acc_out, _, new_pend), w_prv_st = _sublayer_scan(step, sc_init, xs_step)
        return (ac_out, gx1_out, fsdp_prefetch._slice_at(w_prv_st, 0), d_acc_out), new_pend

      if n == 1:
        (act_carry, g_x1_latest, w, d_acc), pend = run_boundary(
            (act_carry, g_x1_latest, w, d_acc),
            0,
            i_prev_boundary,
            prv_last_param,
            prv_last_state,
            prv_last_res[4],
            prv_last_res[5],
            gather_prv,
            owner,
            pend,
        )
        owner = (i, b, 0)
      else:
        # Entry (k = n - 1): predecessor is k = n - 2 of group `i`.
        res_nm2 = _read_res(i, b, n - 2, None if offload_res else b_res[i])
        (act_carry, g_x1_latest, w, d_acc), pend = run_boundary(
            (act_carry, g_x1_latest, w, d_acc),
            n - 1,
            i,
            fsdp_prefetch._slice_at(b_params[i], n - 2),
            fsdp_prefetch._slice_at(b_state[i], n - 2),
            res_nm2[4],
            res_nm2[5],
            gather[i],
            owner,
            pend,
        )
        owner = (i, b, n - 1)

        if n >= 3:
          ordered = n - 2 >= 2
          res_group = None if offload_res else b_res[i]
          sv_group = b_saved[i]
          glb_group = b_g_lb[i]

          def mid_bwd(
              c_mid,
              k_mid,
              i=i,
              n=n,
              p=b_params[i],
              s=b_state[i],
              rg=res_group,
              svg=sv_group,
              glbg=glb_group,
              ordered=ordered,
          ):
            ac_m, gx1_m, w_m, da_m, pend_m = c_mid
            del gx1_m
            p_prev = _index_at(p, k_mid - 1)
            s_prev_m = _index_at(s, k_mid - 1)
            r_k = _read_res(i, b, k_mid, rg)
            sv_k = _index_at(svg, k_mid)
            r_km1 = _read_res(i, b, k_mid - 1, rg)
            glb_k = None if glbg is None else _index_at(glbg, k_mid)
            if ordered:
              ac_m, r_k, sv_k, p_prev = jax.lax.optimization_barrier((ac_m, r_k, sv_k, p_prev))
            da_m = flush(da_m, (i, b, k_mid + 1), pend_m)
            w_prev = gather[i](p_prev)
            ac_next, gx1_next, dw_k = _bwd_sublayer(
                i, i, ac_m, glb_k, w_m, _index_at(s, k_mid), r_k, sv_k, r_km1[4], r_km1[5], s_prev_m
            )
            dw_k = rs_fns[i](dw_k, "untagged")
            if ordered:
              ac_next, gx1_next, dw_k = jax.lax.optimization_barrier((ac_next, gx1_next, dw_k))
            da_m = write(da_m, i, dw_k, b, k_mid, "untagged")
            return (ac_next, gx1_next, w_prev, da_m, deferred(i, dw_k)), None

          (act_carry, g_x1_latest, w, d_acc, pend), _ = _sublayer_scan(
              mid_bwd,
              (act_carry, g_x1_latest, w, d_acc, pend),
              jnp.arange(1, n - 1),
              reverse=True,
              unroll=2 if ordered else 1,
          )
          owner = (i, b, 1)

        # Exit (k = 0): predecessor is `prv_last` of group `i_prev_boundary`.
        (act_carry, g_x1_latest, w, d_acc), pend = run_boundary(
            (act_carry, g_x1_latest, w, d_acc),
            0,
            i_prev_boundary,
            prv_last_param,
            prv_last_state,
            prv_last_res[4],
            prv_last_res[5],
            gather_prv,
            owner,
            pend,
        )
        owner = (i, b, 0)

    return (act_carry, g_x1_latest, w, d_acc, pend), None

  w_init_bwd = gather[last_g](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[last_g], length - 1), last_k))
  d_acc_init = tuple(jax.tree.map(jnp.zeros_like, p) for p in params)
  pend_init = pending_zeros(0)
  res_xs = () if offload_res else res_all
  xs = (jnp.arange(length, dtype=jnp.int32), res_xs, saved_all, g_lb_all, tuple(state), tuple(params))
  ((g_x0_out, *_), g_x1_out, _, d_acc, pend), _ = jax.lax.scan(
      body, (act_carry_init, jnp.zeros_like(g_x1_final), w_init_bwd, d_acc_init, pend_init), xs, reverse=True
  )
  d_acc = flush(d_acc, (0, 0, 0), pend)

  params_bar_leaves = tuple(tuple(jax.tree_util.tree_leaves(d)) for d in d_acc)
  mb0_args_bar = tuple(jnp.zeros_like(x) for x in mb0_args_leaves)
  mb0_kwargs_bar = tuple(jnp.zeros_like(x) for x in mb0_kwargs_leaves)
  mb1_args_bar = tuple(jnp.zeros_like(x) for x in mb1_args_leaves)
  mb1_kwargs_bar = tuple(jnp.zeros_like(x) for x in mb1_kwargs_leaves)
  return (g_x0_out, g_x1_out, params_bar_leaves, None, mb0_args_bar, mb0_kwargs_bar, mb1_args_bar, mb1_kwargs_bar)


_custom_vjp_nested_batch_split.defvjp(_custom_vjp_nested_batch_split_fwd, _custom_vjp_nested_batch_split_bwd)


def _sow_inhomogeneous_aux(layers, config, stage_info, groups_stage_indices, lb_all, bias_all) -> None:
  """Sows `moe_lb_loss` and `moe_bias_updates` onto `layers`'s sub-layer modules matching regular scan."""
  sow_lb = config.load_balance_loss_weight > 0.0 and any(x is not None for x in lb_all)
  sow_bias = config.routed_bias and config.routed_bias_update_rate > 0.0 and any(x is not None for x in bias_all)
  if not sow_lb and not sow_bias:
    return

  kind = stage_info[0]
  if kind == "hybrid":
    _, num_local, num_global = stage_info
    pos = 0
    if num_local > 0:
      # Group 0 holds all `num_local` local layers stacked `[length, num_local, ...]`.
      if sow_lb and lb_all[0] is not None:
        layers.local_layers.moe_lb_loss = nnx.Intermediate(lb_all[0])
      if sow_bias and bias_all[0] is not None:
        if hasattr(layers.local_layers, "moe_bias_updates"):
          delattr(layers.local_layers, "moe_bias_updates")
        layers.local_layers.sow(nnx.Intermediate, "moe_bias_updates", bias_all[0])
      pos = 1
    if num_global == 1 and pos < len(lb_all):
      if sow_lb and lb_all[pos] is not None:
        layers.global_layer.moe_lb_loss = nnx.Intermediate(lb_all[pos][:, 0])
      if sow_bias and bias_all[pos] is not None:
        if hasattr(layers.global_layer, "moe_bias_updates"):
          delattr(layers.global_layer, "moe_bias_updates")
        layers.global_layer.sow(nnx.Intermediate, "moe_bias_updates", bias_all[pos][:, 0])
  elif kind == "named":
    _, names = stage_info
    for g_idx, stage_idxs in enumerate(groups_stage_indices):
      for sub_k, stage_idx in enumerate(stage_idxs):
        sub_mod = getattr(layers, names[stage_idx])
        if sow_lb and lb_all[g_idx] is not None:
          if hasattr(sub_mod, "moe_lb_loss"):
            delattr(sub_mod, "moe_lb_loss")
          sub_mod.sow(nnx.Intermediate, "moe_lb_loss", lb_all[g_idx][:, sub_k])
        if sow_bias and bias_all[g_idx] is not None:
          if hasattr(sub_mod, "moe_bias_updates"):
            delattr(sub_mod, "moe_bias_updates")
          sub_mod.sow(nnx.Intermediate, "moe_bias_updates", bias_all[g_idx][:, sub_k])


def apply_layers_with_batch_split(
    layers,
    stages_and_info,
    x_in,
    *args,
    length: int,
    mesh,
    config,
    metadata_axis_name: str = "layers",
    remat_policy: Any = None,
    prevent_cse: bool = True,
    **kwargs,
):
  """Runs a scanned inhomogeneous block stack with per-sublayer Option B batch-split + FSDP prefetch."""
  if length == 0:
    return x_in, layers, None

  stages, stage_info = stages_and_info
  uses_qwix = hasattr(layers, "qwix_path") or (
      bool(getattr(config, "quantization", "")) and bool(getattr(config, "use_qwix_quantization", False))
  )
  use_fsdp_prefetch = bool(getattr(config, "prefetch_fsdp_weights", False)) and fsdp_prefetch.mesh_uses_fsdp(mesh)
  if use_fsdp_prefetch and config.shard_mode == ShardMode.EXPLICIT and uses_qwix:
    if not fsdp_prefetch.qwix_supports_explicit_sharding():
      raise ValueError(
          "enable_batch_split=True with prefetch_fsdp_weights=True, shard_mode='explicit', and Qwix quantization"
          " requires a qwix version that includes google/qwix#455 (explicit sharding support)."
      )

  fsdp_prefetch._register_overlap_transform(mesh)

  sig = inspect.signature(layers.__class__.__call__)
  valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

  check_vma = bool(config.check_vma) if config.shard_mode == ShardMode.EXPLICIT else False
  x_pspec = batch_split._infer_activation_pspec(x_in, mesh, config)
  global_batch = x_in.shape[0]
  x0_in, x1_in = batch_split._split_array_microbatches(x_in, x_pspec, mesh, check_vma)

  def _maybe_split_leaf(leaf):
    if batch_split._is_diff_leaf(leaf) and leaf.ndim >= 1 and leaf.shape[0] == global_batch:
      pspec = batch_split._infer_activation_pspec(leaf, mesh, config)
      return batch_split._split_array_microbatches(leaf, pspec, mesh, check_vma)
    return leaf, leaf

  args_leaves, args_treedef = jax.tree_util.tree_flatten(tuple(args))
  kwargs_leaves, kwargs_treedef = jax.tree_util.tree_flatten(tuple(valid_kwargs.items()))

  dummy_arr = jnp.zeros((0,), dtype=jnp.float32)
  mb0_args_leaves, mb1_args_leaves, static_args_leaves = [], [], []
  for leaf in args_leaves:
    if batch_split._is_diff_leaf(leaf):
      l0, l1 = _maybe_split_leaf(leaf)
      mb0_args_leaves.append(l0)
      mb1_args_leaves.append(l1)
      static_args_leaves.append(None)
    else:
      mb0_args_leaves.append(dummy_arr)
      mb1_args_leaves.append(dummy_arr)
      static_args_leaves.append(leaf)

  mb0_kwargs_leaves, mb1_kwargs_leaves, static_kwargs_leaves = [], [], []
  for leaf in kwargs_leaves:
    if batch_split._is_diff_leaf(leaf):
      l0, l1 = _maybe_split_leaf(leaf)
      mb0_kwargs_leaves.append(l0)
      mb1_kwargs_leaves.append(l1)
      static_kwargs_leaves.append(None)
    else:
      mb0_kwargs_leaves.append(dummy_arr)
      mb1_kwargs_leaves.append(dummy_arr)
      static_kwargs_leaves.append(leaf)

  # Group consecutive sub-layers with identical structure (`[length, sizes[i], ...]`), preserving param_scan_axis on return.
  groups = []  # [graphdef, p_treedef, s_treedef, [p_leaves...], [s_leaves...], [stage_idx...], orig_scan_ranks]
  scan_axis = config.param_scan_axis
  for stage_idx, (stage_graphdef, p, s) in enumerate(stages):
    orig_p_leaves, _ = jax.tree_util.tree_flatten(p)
    p, s = fsdp_prefetch._layer_major(p, s, scan_axis, length)
    p_leaves, p_treedef = jax.tree_util.tree_flatten(p)
    s_leaves, s_treedef = jax.tree_util.tree_flatten(s)
    shapes = [x.shape for x in p_leaves + s_leaves]
    if groups:
      g = groups[-1]
      if (
          g[0] == stage_graphdef
          and g[1] == p_treedef
          and g[2] == s_treedef
          and [x.shape for x in g[3][0] + g[4][0]] == shapes
      ):
        g[3].append(p_leaves)
        g[4].append(s_leaves)
        g[5].append(stage_idx)
        continue
    groups.append(
        [stage_graphdef, p_treedef, s_treedef, [p_leaves], [s_leaves], [stage_idx], [x.ndim for x in orig_p_leaves]]
    )

  def stack_sublayers(per_sublayer):
    return tuple(jnp.stack(xs, axis=1) for xs in zip(*per_sublayer))

  bwd_box = {}

  def bwd_cache(build=None):
    if build is not None:
      bwd_box["bundles"] = build()
    if "bundles" not in bwd_box:
      raise RuntimeError("batch_split_inhomogeneous backward bundles were not populated during forward trace.")
    return bwd_box["bundles"]

  sparse_core_id = fsdp_prefetch._default_sparse_core_id(mesh)
  meta_axes = tuple(dict.fromkeys((metadata_axis_name, "layers", "local_layers")))
  group_rep_red_axes = []
  for g in groups:
    if use_fsdp_prefetch and config.shard_mode == ShardMode.EXPLICIT:
      p0_leaves = g[3][0]
      layer_shardings_probe = [fsdp_prefetch._per_layer_sharding(p, mesh) for p in p0_leaves]
      p0_sds_leaves = [
          jax.ShapeDtypeStruct(p.shape[1:] if getattr(p, "ndim", 0) > 0 else p.shape, p.dtype, sharding=s)
          for p, s in zip(p0_leaves, layer_shardings_probe)
      ]
      s0_sds_tree = jax.eval_shape(
          lambda sl, td=g[2]: fsdp_prefetch._slice_at(jax.tree_util.tree_unflatten(td, sl), 0), g[4][0]
      )
      rep_ax = batch_split._probe_replicated_reduction_axes(
          per_layer_params_leaves=p0_sds_leaves,
          params_treedef=g[1],
          layer_shardings=layer_shardings_probe,
          s0_tree=s0_sds_tree,
          x0_sample=x0_in,
          mb0_args_leaves=mb0_args_leaves,
          mb0_kwargs_leaves=mb0_kwargs_leaves,
          static_args_leaves=tuple(static_args_leaves),
          static_kwargs_leaves=tuple(static_kwargs_leaves),
          args_treedef=args_treedef,
          kwargs_treedef=kwargs_treedef,
          graphdef=g[0],
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
          sparse_core_id=sparse_core_id,
          x_pspec=x_pspec,
          metadata_axis_names=meta_axes,
      )
    else:
      rep_ax = ()
    group_rep_red_axes.append(rep_ax)

  ctxs = tuple(
      _GroupCtx(
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
          metadata_axis_name=metadata_axis_name,
          static_args_leaves=tuple(static_args_leaves),
          static_kwargs_leaves=tuple(static_kwargs_leaves),
          params_treedef=g[1],
          state_treedef=g[2],
          args_treedef=args_treedef,
          kwargs_treedef=kwargs_treedef,
          use_fsdp_prefetch=use_fsdp_prefetch,
          defer_grad_reduce_scatter=bool(getattr(config, "prefetch_fsdp_defer_grad_reduce_scatter", False))
          and use_fsdp_prefetch,
          offload_layer_inputs=getattr(config, "decoder_layer_input", "device") == "offload",
          sparse_core_id=sparse_core_id,
          remat_policy=remat_policy,
          prevent_cse=prevent_cse,
          replicated_red_axes=rep_ax,
      )
      for g, rep_ax in zip(groups, group_rep_red_axes)
  )

  x0_final, x1_final, lb_all, bias_all = _custom_vjp_nested_batch_split(
      tuple(g[0] for g in groups),
      x0_in,
      x1_in,
      tuple(stack_sublayers(g[3]) for g in groups),
      tuple(stack_sublayers(g[4]) for g in groups),
      tuple(mb0_args_leaves),
      tuple(mb0_kwargs_leaves),
      tuple(mb1_args_leaves),
      tuple(mb1_kwargs_leaves),
      length,
      ctxs,
      tuple(len(g[3]) for g in groups),
      bwd_cache,
  )

  out_pspec = batch_split._infer_activation_pspec(x0_final, mesh, config)
  final_y = batch_split._merge_array_microbatches(x0_final, x1_final, out_pspec, mesh, check_vma)
  _sow_inhomogeneous_aux(layers, config, stage_info, [g[5] for g in groups], lb_all, bias_all)
  return final_y, layers, None
