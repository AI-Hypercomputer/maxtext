# Copyright 2023-2026 Google LLC
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

"""Generalized Batch-Split scheduling for scanned homogeneous MoE decoder layers.

Splits the per-device batch along axis 0 into two microbatches (`mb0`, `mb1`) and
interleaves TensorCore compute (`attn_and_route`, `experts`, `shared_expert`) with
SparseCore-pinned Expert-Parallel collectives (`SC0`: `dispatch` AG, `combine` RS)
and SparseCore-pinned FSDP weight prefetch / deferred gradient reduce-scatter (`SC1`)
via an explicit `@jax.custom_vjp` scan (`unroll=2`).

Wrap-around schedule (Option B):
  - Forward (layer L, holding only W[L] + prefetched W[L+1]):
      F1(L): mb0.attn_and_route(L) [TC] || mb1.combine(L-1) [SC0 RS, weight-free]
      F2(L): mb0.dispatch(L) [SC0 AG]   || mb1.attn_and_route(L) [TC]
      F3(L): mb0.experts(L) [TC]        || mb1.dispatch(L) [SC0 AG]
      F4(L): mb0.combine(L) [SC0 RS] + mb0.shared(L) [TC]
             || mb1.experts(L) [TC] + mb1.shared(L) [TC]
  - Backward (layer L, holding W[L] + prefetched W[L-1] + deferred RS(dW[L+1])):
      B4(L): mb1.experts_bwd(L) + mb1.shared_bwd(L) + mb0.shared_bwd(L) [TC]
             || mb0.combine_bwd(L) [SC0 AG] + mb0.dispatch_remat(L) [SC0 AG]
      B3(L): mb0.experts_bwd(L) [TC]    || mb1.dispatch_bwd(L) [SC0 RS]
      B2(L): mb1.attn_route_bwd(L) [TC] || mb0.dispatch_bwd(L) [SC0 RS]
      B1(L): mb0.attn_route_bwd(L) [TC]
             || mb1.combine_bwd(L-1) [SC0 AG] + mb1.dispatch_remat(L-1) [SC0 AG]

This module is strictly isolated behind `config.enable_batch_split=True` and leaves
all existing scan, `fsdp_prefetch`, `use_batch_split_schedule`, and `use_lineage`
codepaths untouched when disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import math
from typing import Any, Callable

from flax import nnx
import jax
from jax.experimental.layout import Layout, with_layout_constraint
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from maxtext.common.common_types import ShardMode
from maxtext.layers import fsdp_prefetch
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import sharding


_REQUIRED_ADAPTER_METHODS = (
    "batch_split_supported",
    "batch_split_attn_and_route",
    "batch_split_dispatch",
    "batch_split_dispatch_bwd",
    "batch_split_experts",
    "batch_split_shared_expert",
    "batch_split_combine",
    "batch_split_combine_bwd",
    "batch_split_finalize_bias_updates",
)


def supports_batch_split(layers) -> bool:
  """Returns True if `layers` implements the batch-split adapter and supports the current config."""
  if not all(hasattr(layers, m) for m in _REQUIRED_ADAPTER_METHODS):
    return False
  return bool(layers.batch_split_supported())


def _axes_for_dim(dim_spec) -> tuple[str, ...]:
  if dim_spec is None:
    return ()
  if isinstance(dim_spec, str):
    return (dim_spec,)
  return tuple(a for a in dim_spec if a is not None)


def _batch_shard_factor(pspec: P, mesh) -> int:
  if not pspec:
    return 1
  return math.prod(mesh.shape.get(a, 1) for a in _axes_for_dim(pspec[0]))


def _infer_activation_pspec(arr: jax.Array, mesh, config) -> P:
  """Returns the PartitionSpec to use for per-device microbatch splitting of `arr`."""
  if config.shard_mode == ShardMode.EXPLICIT:
    s = getattr(jax.typeof(arr), "sharding", None)
    if isinstance(s, NamedSharding):
      return s.spec
  if arr.ndim == 4:
    logical = ("activation_batch", "activation_length", None, "activation_embed")
  elif arr.ndim == 3:
    logical = ("activation_batch", "activation_length", "activation_embed")
  elif arr.ndim == 2:
    logical = ("activation_batch", "activation_length")
  else:
    logical = ("activation_batch",) + (None,) * max(0, arr.ndim - 1)
  pspec = sharding.logical_to_mesh_axes(logical, mesh)
  # Drop any mesh axis on dim 0 that does not divide the global batch dimension.
  if pspec and pspec[0] is not None:
    batch_axes = _axes_for_dim(pspec[0])
    factor = math.prod(mesh.shape.get(a, 1) for a in batch_axes)
    if factor > 0 and arr.shape[0] % factor != 0:
      pspec = P(None, *pspec[1:])
  return pspec


def can_split_batch(x_in: jax.Array, mesh, config) -> bool:
  """Returns True if `x_in` has an even per-device batch size >= 2 along axis 0."""
  if not isinstance(x_in, jax.Array) or x_in.ndim < 2:
    return False
  pspec = _infer_activation_pspec(x_in, mesh, config)
  factor = _batch_shard_factor(pspec, mesh)
  if factor <= 0 or x_in.shape[0] % factor != 0:
    return False
  local_bs = x_in.shape[0] // factor
  return local_bs >= 2 and local_bs % 2 == 0


def _split_array_microbatches(arr: jax.Array, pspec: P, mesh, check_vma: bool) -> tuple[jax.Array, jax.Array]:
  """Splits `arr` in half along axis 0 per device, preserving sharding and sample correspondence."""

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=pspec,
      out_specs=(pspec, pspec),
      check_vma=check_vma,
  )
  def _split(x):
    half = x.shape[0] // 2
    return x[:half], x[half:]

  return _split(arr)


def _merge_array_microbatches(mb0: jax.Array, mb1: jax.Array, pspec: P, mesh, check_vma: bool) -> jax.Array:
  """Merges `(mb0, mb1)` back along axis 0 per device, restoring exact global sample order."""

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(pspec, pspec),
      out_specs=pspec,
      check_vma=check_vma,
  )
  def _merge(a, b):
    return jnp.concatenate([a, b], axis=0)

  return _merge(mb0, mb1)


def _is_diff_leaf(x) -> bool:
  return isinstance(x, jax.Array) or (hasattr(jax.core, "Tracer") and isinstance(x, jax.core.Tracer))


def _is_inexact_array(x) -> bool:
  return (_is_diff_leaf(x) or isinstance(x, jax.ShapeDtypeStruct)) and jnp.issubdtype(x.dtype, jnp.inexact)


def _split_inexact(tree):
  """Splits a pytree into (inexact_leaves, non_inexact_leaves, mask, treedef)."""
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  mask = tuple(_is_inexact_array(x) for x in leaves)
  diff = tuple(x for x, m in zip(leaves, mask) if m)
  nondiff = tuple(x for x, m in zip(leaves, mask) if not m)
  return diff, nondiff, mask, treedef


def _merge_inexact(diff, nondiff, mask, treedef):
  di, ni = iter(diff), iter(nondiff)
  leaves = [next(di) if m else next(ni) for m in mask]
  return jax.tree_util.tree_unflatten(treedef, leaves)


def _aval_to_sds(aval) -> jax.ShapeDtypeStruct:
  """Converts a JAX `ShapedArray` abstract value to a `ShapeDtypeStruct` preserving sharding and memory space."""
  sharding_attr = getattr(aval, "sharding", None)
  mem_space = getattr(aval, "memory_space", None)
  if sharding_attr is not None:
    if (
        mem_space == jax.memory.Space.Host
        and isinstance(sharding_attr, NamedSharding)
        and sharding_attr.memory_space != jax.memory.Space.Host
    ):
      sharding_attr = sharding_attr.with_memory_space(jax.memory.Space.Host)
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=sharding_attr)
  if mem_space == jax.memory.Space.Host:
    return jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=jax.memory.Space.Host)
  return jax.ShapeDtypeStruct(aval.shape, aval.dtype)


def _zeros_like_with_sharding(x):
  return jnp.zeros_like(x)


def _used_var_indices(closed_jaxpr, invar_slice: slice) -> tuple[int, ...]:
  """Returns indices within `invar_slice` of jaxpr invars actually referenced by any equation or output."""
  target_vars = closed_jaxpr.jaxpr.invars[invar_slice]
  # Key by object id because jaxpr eqns may contain unhashable `Literal` invars.
  var_to_idx = {id(v): i for i, v in enumerate(target_vars)}
  used = set()
  for eqn in closed_jaxpr.jaxpr.eqns:
    for v in eqn.invars:
      idx = var_to_idx.get(id(v))
      if idx is not None:
        used.add(idx)
  for v in closed_jaxpr.jaxpr.outvars:
    idx = var_to_idx.get(id(v))
    if idx is not None:
      used.add(idx)
  return tuple(sorted(used))


_PASSTHROUGH_PRIMITIVES = frozenset(
    {
        "name",
        "copy",
        "copy_p",
        "convert_element_type",
        "sharding_constraint",
        "reshard",
        "mesh_cast",
        "stop_gradient",
    }
)


def _is_literal(v) -> bool:
  return hasattr(v, "val")


def _propagate_passthrough_origins(jaxpr) -> dict[int, int]:
  """Maps `id(var) -> invar_index` for jaxpr vars equal to an input via passthrough primitives."""
  origin = {id(v): i for i, v in enumerate(jaxpr.invars)}
  for eqn in jaxpr.eqns:
    pname = eqn.primitive.name
    if pname == "optimization_barrier" and len(eqn.invars) == len(eqn.outvars):
      for sv, ov in zip(eqn.invars, eqn.outvars):
        if not _is_literal(sv) and id(sv) in origin and sv.aval.shape == ov.aval.shape and sv.aval.dtype == ov.aval.dtype:
          origin[id(ov)] = origin[id(sv)]
      continue
    if len(eqn.invars) != 1 or len(eqn.outvars) != 1:
      continue
    sv, ov = eqn.invars[0], eqn.outvars[0]
    if _is_literal(sv) or id(sv) not in origin:
      continue
    if sv.aval.shape != ov.aval.shape or sv.aval.dtype != ov.aval.dtype:
      continue
    is_pass = pname in _PASSTHROUGH_PRIMITIVES
    if not is_pass and pname == "reduce_precision" and jnp.issubdtype(sv.aval.dtype, jnp.floating):
      finfo = jnp.finfo(sv.aval.dtype)
      is_pass = eqn.params.get("exponent_bits") == finfo.nexp and eqn.params.get("mantissa_bits") == finfo.nmant
    if is_pass:
      origin[id(ov)] = origin[id(sv)]
  return origin


@dataclass(frozen=True)
class _RematSubfn:
  """Captured forward (optional saved-residual) and backward ClosedJaxprs for one sub-function."""

  has_saved: bool
  fwd_jaxpr: Any
  fwd_out_treedef: Any
  bwd_jaxpr: Any
  bwd_out_treedef: Any
  saved_sds: tuple


def _without_qwix_interception(fn: Callable) -> Callable:
  """Disables Qwix op interception while `fn` runs, matching `Transformer.__call__` exit before backward."""
  try:
    from qwix._src import interception as qwix_interception  # pylint: disable=import-outside-toplevel
  except ImportError:
    return fn
  return qwix_interception.disable_interceptions(fn)


def _trace_remat_subfn(
    fn_diff_and_aux: Callable,
    diff_samples: tuple,
    other_samples: tuple,
    cot_samples: Any,
    *,
    remat_policy: Any = None,
    prevent_cse: bool = True,
    fwd_keep_diff_count: int | None = None,
) -> _RematSubfn:
  """Traces forward + backward ClosedJaxprs under `remat_policy`, stripping input-passthrough residuals."""
  bwd_treedef_box: list[Any] = []

  def _plain_bwd(d_args, o_args, cot):
    # Wrap primal in `jax.checkpoint` so Qwix intercepts only primal ops (not JAX internal JVP rules
    # like `solve_triangular`), then run `vjp_fn` with Qwix interception disabled just like `jax.grad`
    # outside `Transformer.__call__` (already-captured `@custom_vjp` `dot_general_qt` keeps its FP8 bwd).
    primal_ckpt = jax.checkpoint(lambda *da: fn_diff_and_aux(da, o_args)[0], prevent_cse=False)
    _, vjp_fn = jax.vjp(primal_ckpt, *d_args)
    grads = _without_qwix_interception(vjp_fn)(cot)
    g_leaves, g_treedef = jax.tree_util.tree_flatten(grads)
    if not bwd_treedef_box:
      bwd_treedef_box.append(g_treedef)
    return g_leaves

  if remat_policy is None:
    bwd_jaxpr = jax.make_jaxpr(_plain_bwd)(diff_samples, other_samples, cot_samples)
    return _RematSubfn(
        has_saved=False,
        fwd_jaxpr=None,
        fwd_out_treedef=None,
        bwd_jaxpr=bwd_jaxpr,
        bwd_out_treedef=bwd_treedef_box[0],
        saved_sds=(),
    )

  ckpt_fn = jax.checkpoint(
      fn_diff_and_aux,
      policy=remat_policy,
      prevent_cse=prevent_cse,
  )
  flat_samples, all_treedef = jax.tree_util.tree_flatten((diff_samples, other_samples))
  res_treedef_box: list[Any] = []
  n_out_box: list[int] = []

  def _probe_fwd(*flat_in):
    d_args, o_args = jax.tree_util.tree_unflatten(all_treedef, flat_in)
    d_outs, vjp_fn, aux_outs = jax.vjp(lambda *da: ckpt_fn(da, o_args), *d_args, has_aux=True)
    res_leaves, res_treedef = jax.tree_util.tree_flatten(vjp_fn)
    out_leaves, _ = jax.tree_util.tree_flatten((d_outs, aux_outs))
    if not res_treedef_box:
      res_treedef_box.append(res_treedef)
      n_out_box.append(len(out_leaves))
    return (*out_leaves, *res_leaves)

  probe_closed = jax.make_jaxpr(_probe_fwd)(*flat_samples)
  origin = _propagate_passthrough_origins(probe_closed.jaxpr)
  res_outvars = probe_closed.jaxpr.outvars[n_out_box[0] :]
  res_sources = []
  saved_res_indices = []
  saved_sds = []
  for r_idx, v in enumerate(res_outvars):
    if _is_literal(v):
      res_sources.append(("lit", v.val))
    elif id(v) in origin:
      res_sources.append(("arg", origin[id(v)]))
    else:
      res_sources.append(None)
      saved_res_indices.append(r_idx)
      saved_sds.append(_aval_to_sds(v.aval))

  if not saved_res_indices:
    bwd_jaxpr = jax.make_jaxpr(_plain_bwd)(diff_samples, other_samples, cot_samples)
    return _RematSubfn(
        has_saved=False,
        fwd_jaxpr=None,
        fwd_out_treedef=None,
        bwd_jaxpr=bwd_jaxpr,
        bwd_out_treedef=bwd_treedef_box[0],
        saved_sds=(),
    )

  fwd_treedef_box: list[Any] = []
  saved_res_indices_t = tuple(saved_res_indices)

  def _remat_fwd(*flat_in):
    d_args, o_args = jax.tree_util.tree_unflatten(all_treedef, flat_in)
    d_outs, vjp_fn, aux_outs = jax.vjp(lambda *da: ckpt_fn(da, o_args), *d_args, has_aux=True)
    res_leaves = jax.tree_util.tree_leaves(vjp_fn)
    saved = tuple(res_leaves[idx] for idx in saved_res_indices_t)
    kept_d_outs = d_outs[:fwd_keep_diff_count] if fwd_keep_diff_count is not None else d_outs
    ret_leaves, ret_treedef = jax.tree_util.tree_flatten((kept_d_outs, aux_outs, saved))
    if not fwd_treedef_box:
      fwd_treedef_box.append(ret_treedef)
    return ret_leaves

  fwd_jaxpr = jax.make_jaxpr(_remat_fwd)(*flat_samples)
  res_treedef = res_treedef_box[0]
  res_sources_t = tuple(res_sources)

  def _remat_bwd(flat_in, saved, cot):
    saved_iter = iter(saved)
    res_leaves = [next(saved_iter) if s is None else (s[1] if s[0] == "lit" else flat_in[s[1]]) for s in res_sources_t]
    vjp_fn = jax.tree_util.tree_unflatten(res_treedef, res_leaves)
    grads = _without_qwix_interception(vjp_fn)(cot)
    g_leaves, g_treedef = jax.tree_util.tree_flatten(grads)
    if not bwd_treedef_box:
      bwd_treedef_box.append(g_treedef)
    return g_leaves

  bwd_jaxpr = jax.make_jaxpr(_remat_bwd)(tuple(flat_samples), tuple(saved_sds), cot_samples)
  return _RematSubfn(
      has_saved=True,
      fwd_jaxpr=fwd_jaxpr,
      fwd_out_treedef=fwd_treedef_box[0],
      bwd_jaxpr=bwd_jaxpr,
      bwd_out_treedef=bwd_treedef_box[0],
      saved_sds=tuple(saved_sds),
  )


def _eval_remat_fwd(subfn: _RematSubfn, diff_args: tuple, other_args: tuple):
  flat_out = _eval_jaxpr_flat(subfn.fwd_jaxpr, diff_args, other_args)
  return jax.tree_util.tree_unflatten(subfn.fwd_out_treedef, flat_out)


def _eval_remat_bwd(subfn: _RematSubfn, diff_args: tuple, other_args: tuple, saved: tuple, cot: Any):
  if subfn.has_saved:
    flat_in = jax.tree_util.tree_leaves((diff_args, other_args))
    flat_out = _eval_jaxpr_flat(subfn.bwd_jaxpr, flat_in, saved, cot)
  else:
    flat_out = _eval_jaxpr_flat(subfn.bwd_jaxpr, diff_args, other_args, cot)
  return jax.tree_util.tree_unflatten(subfn.bwd_out_treedef, flat_out)


def _tree_host_buffer(tree, n: int):
  """Allocates a per-leaf pinned host buffer of leading length `n`."""
  return jax.tree.map(lambda x: fsdp_prefetch._host_stack_buffer(x, n), tree)  # pylint: disable=protected-access


def _tree_host_write(buf_tree, val_tree, idx):
  return jax.tree.map(
      lambda b, v: fsdp_prefetch._host_stack_write(b, v, idx), buf_tree, val_tree  # pylint: disable=protected-access
  )


def _tree_host_read(buf_tree, idx):
  return jax.tree.map(lambda b: fsdp_prefetch._host_stack_read(b, idx), buf_tree)  # pylint: disable=protected-access


@dataclass(frozen=True)
class _BatchSplitCtx:
  """Static (non-differentiable) metadata carried into `_custom_vjp_batch_split_pipeline`."""

  mesh: Any
  shard_mode: ShardMode
  debug_sharding: bool
  scan_axis: int
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
  bwd_cache: Callable
  remat_policy: Any = None
  prevent_cse: bool = True
  replicated_red_axes: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class _CapturedBwdBundle:
  """Backward sub-function jaxprs captured during forward tracing (preserving Qwix interceptors)."""

  attn_w_indices: tuple[int, ...]
  exp_w_indices: tuple[int, ...]
  exp_accum_sub_indices: tuple[int, ...]
  shared_w_indices: tuple[int, ...]
  route_in_mask: tuple[bool, ...]
  route_in_treedef: Any
  route_ag_mask: tuple[bool, ...]
  route_ag_treedef: Any
  has_seq_lb: bool
  has_exp_lb: bool
  has_x_scale: bool
  has_deferred_combine_bwd: bool
  attn_route_subfn: _RematSubfn
  experts_subfn: _RematSubfn
  experts_acc_subfn: _RematSubfn | None
  shared_subfn: _RematSubfn
  dispatch_fwd: Any  # ClosedJaxpr
  dispatch_bwd: Any  # ClosedJaxpr
  combine_bwd: Any  # ClosedJaxpr


def _cast_replicated_reduced(w, mesh, red_axes: tuple[str, ...]):
  """Casts a replicated 2D+ weight to `reduced=frozenset(red_axes)` (zero runtime cost)."""
  if not red_axes:
    return w
  spec = P(*(None,) * w.ndim)
  w_sharding = getattr(jax.typeof(w), "sharding", None)
  if isinstance(w_sharding, NamedSharding) and tuple(w_sharding.spec) != tuple(spec):
    w = jax.sharding.reshard(w, NamedSharding(mesh, spec))
  ax_arg = red_axes if len(red_axes) > 1 else red_axes[0]
  return jax.shard_map(
      lambda z: jax.lax.pcast(z, ax_arg, to="reduced"),
      mesh=mesh,
      in_specs=spec,
      out_specs=P(*(None,) * w.ndim, reduced=frozenset(red_axes)),
  )(w)


def _dcn_mesh_axes(mesh) -> frozenset[str]:
  """Returns mesh axis names of size > 1 that span multiple slices (DCN)."""
  dcn: set[str] = set()
  axis_names = getattr(mesh, "axis_names", None)
  devices = getattr(mesh, "devices", None)
  if (
      axis_names is not None
      and devices is not None
      and getattr(devices, "size", 0) > 0
      and devices.ndim == len(axis_names)
  ):
    for i, name in enumerate(axis_names):
      if devices.shape[i] <= 1:
        continue
      rows = iter(devices.swapaxes(0, i).reshape(devices.shape[i], -1))
      ref = [getattr(d, "slice_index", 0) for d in next(rows)]
      if any([getattr(d, "slice_index", 0) for d in row] != ref for row in rows):
        dcn.add(name)
  return frozenset(dcn)


def _batch_split_replicated_all_reduce(g, mesh, red_axes: tuple[str, ...], sparse_core_id=None):
  """Reduces an `unreduced` gradient of a replicated 2D+ weight on SparseCore (ICI) then DCN."""
  if not red_axes or not fsdp_prefetch._is_unreduced_over(g, red_axes):  # pylint: disable=protected-access
    return None
  spec = P(*(None,) * g.ndim)
  g_spec = jax.typeof(g).sharding.spec
  dcn_axes = _dcn_mesh_axes(mesh)
  ici_axes = tuple(a for a in red_axes if a not in dcn_axes)
  cross_axes = tuple(a for a in red_axes if a in dcn_axes)

  def psum_fn(x):
    if ici_axes:
      ici_arg = ici_axes if len(ici_axes) > 1 else ici_axes[0]
      x = fsdp_prefetch._on_sparse_core(  # pylint: disable=protected-access
          functools.partial(jax.lax.psum, axis_name=ici_arg),
          sparse_core_id,
          x,
      )(x)
    if cross_axes:
      x = jax.lax.psum(x, cross_axes if len(cross_axes) > 1 else cross_axes[0])
    return x

  return jax.shard_map(
      psum_fn,
      mesh=mesh,
      in_specs=P(*(None,) * g.ndim, unreduced=frozenset(red_axes), unreduced_kind=g_spec.unreduced_kind),
      out_specs=spec,
  )(g)


def _per_layer_sharding_at_axis(stacked, mesh, scan_axis: int):
  """Sharding of one layer slice of `stacked` with scan dimension at `scan_axis`, or None."""
  s = getattr(jax.typeof(stacked), "sharding", None) or getattr(stacked, "sharding", None)
  if not isinstance(s, NamedSharding) or s.spec is None:
    return None
  spec_t = tuple(s.spec)
  if scan_axis < len(spec_t):
    spec_t = spec_t[:scan_axis] + spec_t[scan_axis + 1 :]
  return NamedSharding(mesh, P(*spec_t))


def _candidate_reduction_axis_sets(x_pspec: P, mesh) -> list[tuple[str, ...]]:
  """Returns ordered non-empty mesh-axis tuples `((batch+seq axes), (batch axes))` for `reduced` probing."""
  batch_axes = tuple(a for a in _axes_for_dim(x_pspec[0] if x_pspec else None) if mesh.shape.get(a, 1) > 1)
  seq_axes = tuple(
      a
      for a in (_axes_for_dim(x_pspec[1]) if len(x_pspec) > 1 else ())
      if mesh.shape.get(a, 1) > 1 and a not in batch_axes
  )
  out: list[tuple[str, ...]] = []
  for ax in (batch_axes + seq_axes, batch_axes):
    if ax and ax not in out:
      out.append(ax)
  return out


def _probe_replicated_reduction_axes(
    *,
    per_layer_params_leaves,
    params_treedef,
    layer_shardings,
    s0_tree,
    x0_sample,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    static_args_leaves,
    static_kwargs_leaves,
    args_treedef,
    kwargs_treedef,
    graphdef,
    mesh,
    shard_mode: ShardMode,
    debug_sharding: bool,
    sparse_core_id: int | None,
    x_pspec: P,
    metadata_axis_names: tuple[str, ...] = ("layers",),
) -> tuple[tuple[str, ...], ...]:
  """Probes replicated 2D+ parameter leaves to find safe `reduced=...` axes under `ShardMode.EXPLICIT`."""
  n_leaves = len(per_layer_params_leaves)
  result: list[tuple[str, ...]] = [()] * n_leaves
  if shard_mode != ShardMode.EXPLICIT:
    return tuple(result)
  cand_axis_sets = _candidate_reduction_axis_sets(x_pspec, mesh)
  if not cand_axis_sets:
    return tuple(result)
  remaining: list[int] = []
  for idx, (p, s) in enumerate(zip(per_layer_params_leaves, layer_shardings)):
    if (
        getattr(p, "ndim", 0) >= 2
        and s is not None
        and all(e is None for e in s.spec)
        and not s.spec.reduced
        and not s.spec.unreduced
        and fsdp_prefetch._fsdp_scatter_dim(p.ndim, mesh, s.spec) is None  # pylint: disable=protected-access
    ):
      remaining.append(idx)
  if not remaining:
    return tuple(result)

  def _eval_with_overrides(overrides: dict[int, tuple[str, ...]]):
    def _vjp_probe(p_l, s0_t, x_s, d_a, d_k):
      gathered_l = []
      for idx, (x, s) in enumerate(zip(p_l, layer_shardings)):
        red_ax = overrides.get(idx, ())
        if red_ax:
          x = _cast_replicated_reduced(x, mesh, red_ax)
        else:
          x = fsdp_prefetch._apply_sharding_hint(  # pylint: disable=protected-access
              x, mesh, shard_mode, debug_sharding, src_sharding=s, sparse_core_id=sparse_core_id
          )
        gathered_l.append(x)
      args_t = jax.tree_util.tree_unflatten(
          args_treedef, [st if st is not None else d for d, st in zip(d_a, static_args_leaves)]
      )
      kwargs_d = dict(
          jax.tree_util.tree_unflatten(
              kwargs_treedef, [st if st is not None else d for d, st in zip(d_k, static_kwargs_leaves)]
          )
      )

      def _fwd_all(wl):
        wt = jax.tree_util.tree_unflatten(params_treedef, wl)
        wc, sc = wt, s0_t
        for ax_name in metadata_axis_names:
          wc, sc = maxtext_utils_nnx.nnx_remove_scan_axis((wc, sc), ax_name)
        lyr = nnx.merge(graphdef, wc, sc)
        hs, m_in, r_in, s_lb = lyr.batch_split_attn_and_route(x_s, *args_t, **kwargs_d)
        xv, xs, r_ag = lyr.batch_split_dispatch(m_in, r_in)
        yp, e_lb, _ = lyr.batch_split_experts(
            m_in, xv, xs, r_ag, decoder_input_tokens=kwargs_d.get("decoder_input_tokens")
        )
        sh = lyr.batch_split_shared_expert(m_in)
        hs_d, _, _, _ = _split_inexact(hs)
        r_d, _, _, _ = _split_inexact(r_in)
        outs = [*hs_d, m_in, *r_d, yp, sh]
        if s_lb is not None:
          outs.append(s_lb)
        if e_lb is not None:
          outs.append(e_lb)
        return tuple(outs)

      outs, vjp_fn = jax.vjp(jax.checkpoint(_fwd_all, prevent_cse=False), gathered_l)
      cots = tuple(jnp.zeros(o.shape, o.dtype, out_sharding=getattr(jax.typeof(o), "sharding", None)) for o in outs)
      return _without_qwix_interception(vjp_fn)(cots)[0]

    return jax.eval_shape(
        _vjp_probe, per_layer_params_leaves, s0_tree, x0_sample, tuple(mb0_args_leaves), tuple(mb0_kwargs_leaves)
    )

  for cand_axes in cand_axis_sets:
    if not remaining:
      break
    base = {i: result[i] for i in range(n_leaves) if result[i]}
    try:
      dw_sds = _eval_with_overrides({**base, **{idx: cand_axes for idx in remaining}})
      for idx in remaining:
        if fsdp_prefetch._is_unreduced_over(dw_sds[idx], cand_axes):  # pylint: disable=protected-access
          result[idx] = cand_axes
    except Exception:  # pylint: disable=broad-except
      if len(remaining) > 1:
        for idx in remaining:
          try:
            dw_sds = _eval_with_overrides({**base, idx: cand_axes})
            if fsdp_prefetch._is_unreduced_over(dw_sds[idx], cand_axes):  # pylint: disable=protected-access
              result[idx] = cand_axes
              base[idx] = cand_axes
          except Exception:  # pylint: disable=broad-except
            pass
    remaining = [idx for idx in remaining if not result[idx]]
  return tuple(result)


def _batch_split_fsdp_reduce_scatter(g, mesh, src_spec, gathered_spec, pad_minor=False, sparse_core_id=None):
  """FSDP reduce-scatter that moves any non-major scatter dim to axis 0 before SparseCore psum_scatter.

  Workaround for TPU7x SparseCore `psum_scatter` when `scatter_dimension != 0` on 3D+ tensors
  (e.g. `self_attention.out.kernel` of shape `(heads, head_dim, emb)` sharded on `dim=2`):
  moving `dim` to axis 0 across the `compute_on` boundary guarantees a major-most `scatter_dimension=0`
  SparseCore reduce-scatter for all ranks, not only 2D.
  """
  scatter = fsdp_prefetch._fsdp_scatter_dim(g.ndim, mesh, src_spec)  # pylint: disable=protected-access
  if scatter is None or not fsdp_prefetch._is_unreduced_over(  # pylint: disable=protected-access
      g, fsdp_prefetch._expected_unreduced_axes(mesh, src_spec, scatter[1])  # pylint: disable=protected-access
  ):
    return None
  dim, ax = scatter
  eff_sc_id = None if any(a in _dcn_mesh_axes(mesh) for a in ax) else sparse_core_id
  extra = fsdp_prefetch._extra_batch_axes(mesh, src_spec, ax)  # pylint: disable=protected-access
  spec = tuple(src_spec) + (None,) * (g.ndim - len(tuple(src_spec)))
  gathered = tuple(gathered_spec) + (None,) * (g.ndim - len(tuple(gathered_spec)))
  g_spec = jax.typeof(g).sharding.spec
  minor_dim = 1 - dim if pad_minor and g.ndim == 2 else None
  orig_minor = g.shape[minor_dim] if minor_dim is not None and gathered[minor_dim] is None else None
  padded_minor = fsdp_prefetch._sc_rs_padded_minor_size(orig_minor) if orig_minor is not None else None  # pylint: disable=protected-access
  if padded_minor is None or padded_minor == orig_minor:
    g = fsdp_prefetch._pin_scatter_dim_major(g, dim)  # pylint: disable=protected-access

  def psum_scatter(x):
    if padded_minor is not None and padded_minor > orig_minor:
      pad_width = [(0, 0)] * x.ndim
      pad_width[minor_dim] = (0, padded_minor - orig_minor)
      x = fsdp_prefetch._pad_unreduced(x, pad_width, tuple(ax) + tuple(extra))  # pylint: disable=protected-access
      x = fsdp_prefetch._pin_scatter_dim_major(x, dim)  # pylint: disable=protected-access
    if dim != 0 and eff_sc_id is not None and eff_sc_id >= 0:
      x = jnp.moveaxis(x, dim, 0)
      x = fsdp_prefetch._on_sparse_core(  # pylint: disable=protected-access
          functools.partial(
              jax.lax.psum_scatter, axis_name=ax if len(ax) > 1 else ax[0], scatter_dimension=0, tiled=True
          ),
          eff_sc_id,
          x,
      )(x)
      x = jnp.moveaxis(x, 0, dim)
    else:
      x = fsdp_prefetch._on_sparse_core(  # pylint: disable=protected-access
          functools.partial(
              jax.lax.psum_scatter, axis_name=ax if len(ax) > 1 else ax[0], scatter_dimension=dim, tiled=True
          ),
          eff_sc_id,
          x,
      )(x)
    if extra:
      x = jax.lax.psum(x, extra if len(extra) > 1 else extra[0])
    if padded_minor is not None and padded_minor > orig_minor:
      x = jax.lax.slice_in_dim(x, 0, orig_minor, axis=minor_dim)
    return x

  return jax.shard_map(
      psum_scatter,
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec(
          *gathered, unreduced=frozenset(ax) | frozenset(extra), unreduced_kind=g_spec.unreduced_kind
      ),
      out_specs=jax.sharding.PartitionSpec(*spec),
  )(g)


def _batch_split_setup(
    graphdef,
    params_leaves,
    state_leaves,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    ctx: _BatchSplitCtx,
):
  """Rebuilds params/state and returns per-layer helpers + FSDP gather/RS closures."""
  if ctx.scan_axis != 0:
    params_leaves = [
        jnp.moveaxis(p, ctx.scan_axis, 0) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_leaves
    ]
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, params_leaves)
  state = jax.tree_util.tree_unflatten(ctx.state_treedef, state_leaves)

  if ctx.use_fsdp_prefetch:
    layer_shardings = [fsdp_prefetch._per_layer_sharding(p, ctx.mesh) for p in params_leaves]  # pylint: disable=protected-access
    pin_layout_flags = [
        fsdp_prefetch._needs_minor_dim_layout_pin(p, s, ctx.mesh)  # pylint: disable=protected-access
        for p, s in zip(params_leaves, layer_shardings)
    ]
    rep_red_axes = (
        ctx.replicated_red_axes if len(ctx.replicated_red_axes) == len(params_leaves) else ((),) * len(params_leaves)
    )

    def gather_fn(layer_params):
      leaves, treedef = jax.tree_util.tree_flatten(layer_params)
      out = []
      for x, pin, red_ax in zip(leaves, pin_layout_flags, rep_red_axes):
        if red_ax:
          x = _cast_replicated_reduced(x, ctx.mesh, red_ax)
        else:
          x = fsdp_prefetch._apply_sharding_hint(  # pylint: disable=protected-access
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
        out = _batch_split_replicated_all_reduce(g, ctx.mesh, red_ax, sparse_core_id=ctx.sparse_core_id)
        if out is not None:
          return out
      if tag:
        out = _batch_split_fsdp_reduce_scatter(
            g, ctx.mesh, s.spec, sharding.remove_fsdp_from_sharding(s).spec, sparse_core_id=ctx.sparse_core_id
        )
        if out is not None:
          return out
      scatter = fsdp_prefetch._fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)  # pylint: disable=protected-access
      if scatter is not None and fsdp_prefetch._is_unreduced_over(  # pylint: disable=protected-access
          g, fsdp_prefetch._expected_unreduced_axes(ctx.mesh, s.spec, scatter[1])  # pylint: disable=protected-access
      ):
        g = fsdp_prefetch._pin_scatter_dim_major(g, scatter[0])  # pylint: disable=protected-access
      return sharding.maybe_shard_with_name(g, s, ctx.shard_mode, debug_sharding=ctx.debug_sharding)

    tag_cache = {}

    def tag_flags(leaves):
      if "flags" not in tag_cache:
        flags = []
        for g, s, red_ax in zip(leaves, layer_shardings, rep_red_axes):
          if red_ax:
            eligible = fsdp_prefetch._is_unreduced_over(g, red_ax)  # pylint: disable=protected-access
          else:
            scatter = None if s is None else fsdp_prefetch._fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)  # pylint: disable=protected-access
            eligible = scatter is not None and fsdp_prefetch._is_unreduced_over(  # pylint: disable=protected-access
                g, fsdp_prefetch._expected_unreduced_axes(ctx.mesh, s.spec, scatter[1])  # pylint: disable=protected-access
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
    w_clean, s_clean = maxtext_utils_nnx.nnx_remove_scan_axis((layer_weights, layer_state), "layers")
    return nnx.merge(graphdef, w_clean, s_clean)

  def _merge_weightless_layer(layer_state, template_w):
    # Weight-free ops (dispatch, combine, combine_bwd, finalize_bias_updates) never read parameters;
    # pass template_w (or zeros/placeholders) solely so nnx.merge builds the layer shell.
    return _merge_layer(template_w, layer_state)

  def attn_route_pure(x, layer_weights, layer_state, d_args, d_kwargs):
    args_tuple, kwargs_dict = _reconstruct_call_args(d_args, d_kwargs)
    layer = _merge_layer(layer_weights, layer_state)
    return layer.batch_split_attn_and_route(x, *args_tuple, **kwargs_dict)

  def dispatch_pure(moe_in, route_in, layer_state, template_w):
    layer = _merge_weightless_layer(layer_state, template_w)
    return layer.batch_split_dispatch(moe_in, route_in)

  def dispatch_bwd_pure(moe_in, route_in, g_x_val_ag, g_route_ag_diff, layer_state, template_w):
    layer = _merge_weightless_layer(layer_state, template_w)
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
    layer = _merge_weightless_layer(layer_state, template_w)
    return layer.batch_split_combine(y_partial, moe_in_ref)

  def combine_bwd_pure(g_out, moe_in_ref, layer_state, template_w):
    layer = _merge_weightless_layer(layer_state, template_w)
    cb_fn = layer.batch_split_combine_bwd
    if "defer_dequant" in inspect.signature(cb_fn).parameters:
      return cb_fn(g_out, moe_in_ref, defer_dequant=True)
    return cb_fn(g_out, moe_in_ref)

  def finalize_bias_pure(summed_counts, layer_state, template_w):
    if summed_counts is None:
      return None
    layer = _merge_weightless_layer(layer_state, template_w)
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
      combine_bwd_pure,
      finalize_bias_pure,
  )


def _to_sds(x) -> jax.ShapeDtypeStruct:
  if isinstance(x, jax.ShapeDtypeStruct):
    return x
  sharding_attr = getattr(jax.typeof(x), "sharding", getattr(x, "sharding", None))
  if sharding_attr is not None:
    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding_attr)
  return jax.ShapeDtypeStruct(x.shape, x.dtype)


def _build_captured_bwd_bundle(
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
    combine_bwd_pure,
    remat_policy: Any = None,
    prevent_cse: bool = True,
) -> _CapturedBwdBundle:
  """Traces all backward (and remat-saved forward) sub-functions once during forward."""

  def _probe_fwd(x_, w_, s_, da_, dk_):
    hs, m_in, r_in, s_lb = attn_route_pure(x_, w_, s_, da_, dk_)
    xv, xs, r_ag = dispatch_pure(m_in, r_in, s_, w_)
    y_p, e_lb, _ = experts_pure(m_in, xv, xs, r_ag, w_, s_, da_, dk_)
    return hs, m_in, r_in, s_lb, xv, xs, r_ag, y_p, e_lb

  (
      h_skip_sample,
      moe_in_sample,
      route_in_sample,
      seq_lb_sample,
      x_val_ag_sample,
      x_scale_ag_sample,
      route_ag_sample,
      _y_part_sample,
      exp_lb_sample,
  ) = jax.eval_shape(_probe_fwd, x_sample, w_sample, s_sample, d_args_sample, d_kwargs_sample)

  w_leaves_all, w_treedef = jax.tree_util.tree_flatten(w_sample)
  s_leaves_all, s_treedef = jax.tree_util.tree_flatten(s_sample)
  n_w = len(w_leaves_all)

  route_in_diff, route_in_nondiff, route_in_mask, route_in_treedef = _split_inexact(route_in_sample)
  route_ag_diff, route_ag_nondiff, route_ag_mask, route_ag_treedef = _split_inexact(route_ag_sample)
  has_seq_lb = seq_lb_sample is not None
  has_exp_lb = exp_lb_sample is not None
  has_x_scale = x_scale_ag_sample is not None

  # 1. Discover which weight leaves each sub-function reads so each VJP differentiates only its own weights.
  def _attn_probe(x, w_l, s_l, da, dk):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    return attn_route_pure(x, w_t, s_t, da, dk)

  attn_probe_jaxpr = jax.make_jaxpr(_attn_probe)(x_sample, w_leaves_all, s_leaves_all, d_args_sample, d_kwargs_sample)
  attn_w_indices = _used_var_indices(attn_probe_jaxpr, slice(1, 1 + n_w))

  def _exp_probe(m_ref, xv, xs, r_diff, r_nd, w_l, s_l, da, dk):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    r_ag = _merge_inexact(r_diff, r_nd, route_ag_mask, route_ag_treedef)
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
  exp_w_indices = _used_var_indices(exp_probe_jaxpr, slice(exp_probe_prefix, exp_probe_prefix + n_w))

  def _shared_probe(m_in, w_l, s_l):
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_l)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    return shared_pure(m_in, w_t, s_t)

  shared_probe_jaxpr = jax.make_jaxpr(_shared_probe)(moe_in_sample, w_leaves_all, s_leaves_all)
  shared_w_indices = _used_var_indices(shared_probe_jaxpr, slice(1, 1 + n_w))

  def _substitute_weights(w_sub, indices):
    merged = list(w_leaves_all)
    for idx, val in zip(indices, w_sub):
      merged[idx] = val
    return jax.tree_util.tree_unflatten(w_treedef, merged)

  # 2. Trace combine_bwd_jaxpr: (g_out, moe_in_ref, s_leaves) -> g_y_partial or (qval, scale)
  def _combine_bwd_fn(g_out, m_ref, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    return combine_bwd_pure(g_out, m_ref, s_t, w_t)

  g_x_sds = _to_sds(x_sample)
  combine_bwd_jaxpr = jax.make_jaxpr(_combine_bwd_fn)(g_x_sds, moe_in_sample, s_leaves_all)
  has_deferred_combine_bwd = len(combine_bwd_jaxpr.out_avals) == 2
  if has_deferred_combine_bwd:
    qval_sds = _aval_to_sds(combine_bwd_jaxpr.out_avals[0])
    g_y_part_sds = jax.ShapeDtypeStruct(
        qval_sds.shape, _y_part_sample.dtype, sharding=getattr(qval_sds, "sharding", None)
    )
  else:
    g_y_part_sds = _aval_to_sds(combine_bwd_jaxpr.out_avals[0])

  # 3. Trace experts_subfn (mb1 in F4->B4, and mb0 fallback) under `remat_policy`.
  w_exp_sample = tuple(w_leaves_all[i] for i in exp_w_indices)
  g_exp_lb_sds = _to_sds(exp_lb_sample) if has_exp_lb else None

  def _exp_diff_and_aux(d_args, o_args):
    if has_x_scale:
      xv_, xs_, rd_, ws_ = d_args
    else:
      xv_, rd_, ws_ = d_args
      xs_ = None
    m_ref_, r_nd_, s_l_, da_, dk_ = o_args
    w_t = _substitute_weights(ws_, exp_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    r_ag = _merge_inexact(rd_, r_nd_, route_ag_mask, route_ag_treedef)
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
  experts_subfn = _trace_remat_subfn(
      _exp_diff_and_aux,
      exp_diff_samples,
      exp_other_samples,
      exp_cot_samples,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  # Extract exact cotangent avals for dispatch_bwd directly from experts_subfn.bwd_jaxpr outputs!
  exp_bwd_out_sds = jax.tree_util.tree_unflatten(
      experts_subfn.bwd_out_treedef, [_aval_to_sds(a) for a in experts_subfn.bwd_jaxpr.out_avals]
  )
  if has_x_scale:
    g_x_val_ag_sds, _, g_route_ag_diff_sds, exp_bwd_dw_sds = exp_bwd_out_sds
  else:
    g_x_val_ag_sds, g_route_ag_diff_sds, exp_bwd_dw_sds = exp_bwd_out_sds

  # 3b. Optional in-place expert wgrad accumulation (`tgmm_v2` `partial_sum` via `return_wgrad_hooks`).
  probe_box: dict[str, tuple[int, ...]] = {"sub_indices": ()}

  def _probe_acc_hooks(m_ref, xv, xs, r_diff, r_nd, w_sub, s_l, da, dk):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = _substitute_weights(w_sub, exp_w_indices)
    r_ag = _merge_inexact(r_diff, r_nd, route_ag_mask, route_ag_treedef)
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
      r_ag = _merge_inexact(rd_, r_nd_, route_ag_mask, route_ag_treedef)
      y_p, e_lb, cnt, hooks = experts_pure(m_ref_, xv_, xs_, r_ag, w_t, s_t, da_, dk_, return_wgrad_hooks=True)
      hook_vals = tuple(h for _, h in hooks)
      d_outs = (y_p, e_lb, hook_vals) if has_exp_lb else (y_p, hook_vals)
      return d_outs, cnt

    exp_acc_cot_samples = (g_y_part_sds, g_exp_lb_sds, dw_accum_sds) if has_exp_lb else (g_y_part_sds, dw_accum_sds)
    experts_acc_subfn = _trace_remat_subfn(
        _exp_acc_diff_and_aux,
        exp_diff_samples,
        exp_other_samples,
        exp_acc_cot_samples,
        remat_policy=remat_policy,
        prevent_cse=prevent_cse,
        fwd_keep_diff_count=2 if has_exp_lb else 1,
    )

  # 4. Trace dispatch_fwd_jaxpr & explicit SC0 dispatch_bwd_jaxpr:
  def _dispatch_fwd_fn(m_in, r_diff, r_nd, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    r_in = _merge_inexact(r_diff, r_nd, route_in_mask, route_in_treedef)
    xv, xs, r_ag = dispatch_pure(m_in, r_in, s_t, w_t)
    r_ag_d, r_ag_nd, _, _ = _split_inexact(r_ag)
    return (xv, xs, r_ag_d), r_ag_nd

  dispatch_fwd_jaxpr = jax.make_jaxpr(_dispatch_fwd_fn)(moe_in_sample, route_in_diff, route_in_nondiff, s_leaves_all)

  def _dispatch_bwd_fn(m_in, r_diff, r_nd, g_xv, g_rag_d, s_l):
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l)
    w_t = jax.tree_util.tree_unflatten(w_treedef, w_leaves_all)
    r_in = _merge_inexact(r_diff, r_nd, route_in_mask, route_in_treedef)
    return dispatch_bwd_pure(m_in, r_in, g_xv, g_rag_d, s_t, w_t)

  dispatch_bwd_jaxpr = jax.make_jaxpr(_dispatch_bwd_fn)(
      moe_in_sample, route_in_diff, route_in_nondiff, g_x_val_ag_sds, g_route_ag_diff_sds, s_leaves_all
  )
  g_moe_in_sds = _aval_to_sds(dispatch_bwd_jaxpr.out_avals[0])
  g_route_in_diff_sds = tuple(_aval_to_sds(a) for a in dispatch_bwd_jaxpr.out_avals[1:])

  # 5. Trace shared_subfn under `remat_policy`.
  w_shared_sample = tuple(w_leaves_all[i] for i in shared_w_indices)

  def _shared_diff_and_aux(d_args, o_args):
    m_, ws_ = d_args
    (s_l_,) = o_args
    w_t = _substitute_weights(ws_, shared_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    return shared_pure(m_, w_t, s_t), ()

  shared_subfn = _trace_remat_subfn(
      _shared_diff_and_aux,
      (moe_in_sample, w_shared_sample),
      (s_leaves_all,),
      g_x_sds,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  # 6. Trace attn_route_subfn under `remat_policy`.
  w_attn_sample = tuple(w_leaves_all[i] for i in attn_w_indices)
  g_h_skip_sds = _to_sds(h_skip_sample)
  g_seq_lb_sds = _to_sds(seq_lb_sample) if has_seq_lb else None

  def _attn_diff_and_aux(d_args, o_args):
    x_, ws_ = d_args
    s_l_, da_, dk_ = o_args
    w_t = _substitute_weights(ws_, attn_w_indices)
    s_t = jax.tree_util.tree_unflatten(s_treedef, s_l_)
    h_skip, m_in, r_in, s_lb = attn_route_pure(x_, w_t, s_t, da_, dk_)
    r_d, r_nd, _, _ = _split_inexact(r_in)
    d_outs = (h_skip, m_in, r_d, s_lb) if has_seq_lb else (h_skip, m_in, r_d)
    return d_outs, r_nd

  attn_cot_samples = (
      (g_h_skip_sds, g_moe_in_sds, g_route_in_diff_sds, g_seq_lb_sds)
      if has_seq_lb
      else (g_h_skip_sds, g_moe_in_sds, g_route_in_diff_sds)
  )
  attn_route_subfn = _trace_remat_subfn(
      _attn_diff_and_aux,
      (x_sample, w_attn_sample),
      (s_leaves_all, d_args_sample, d_kwargs_sample),
      attn_cot_samples,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
  )

  if remat_policy is not None:
    exp0_subfn = experts_acc_subfn if experts_acc_subfn is not None else experts_subfn
    all_saved_sds = (
        *attn_route_subfn.saved_sds,
        *attn_route_subfn.saved_sds,
        *exp0_subfn.saved_sds,
        *experts_subfn.saved_sds,
        *shared_subfn.saved_sds,
        *shared_subfn.saved_sds,
    )
    saved_bytes = sum(s.size * s.dtype.itemsize for s in all_saved_sds)
    max_logging.info(
        f"enable_batch_split: remat_policy saves {len(all_saved_sds)} residual leaf(s) per layer across mb0+mb1"
        f" (attn={len(attn_route_subfn.saved_sds)}x2, exp={len(exp0_subfn.saved_sds)}+{len(experts_subfn.saved_sds)},"
        f" shared={len(shared_subfn.saved_sds)}x2; {saved_bytes / 2**20:.1f} MiB global shape/layer)."
    )

  return _CapturedBwdBundle(
      attn_w_indices=attn_w_indices,
      exp_w_indices=exp_w_indices,
      exp_accum_sub_indices=exp_accum_sub_indices,
      shared_w_indices=shared_w_indices,
      route_in_mask=route_in_mask,
      route_in_treedef=route_in_treedef,
      route_ag_mask=route_ag_mask,
      route_ag_treedef=route_ag_treedef,
      has_seq_lb=has_seq_lb,
      has_exp_lb=has_exp_lb,
      has_x_scale=has_x_scale,
      has_deferred_combine_bwd=has_deferred_combine_bwd,
      attn_route_subfn=attn_route_subfn,
      experts_subfn=experts_subfn,
      experts_acc_subfn=experts_acc_subfn,
      shared_subfn=shared_subfn,
      dispatch_fwd=dispatch_fwd_jaxpr,
      dispatch_bwd=dispatch_bwd_jaxpr,
      combine_bwd=combine_bwd_jaxpr,
  )


def _eval_jaxpr_flat(closed_jaxpr, *args):
  flat_args = jax.tree_util.tree_leaves(args)
  return jax.core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *flat_args)


def _combine_layer_lb(seq_lb_0, seq_lb_1, exp_lb_0, exp_lb_1):
  if seq_lb_0 is not None:
    return 0.5 * (seq_lb_0 + seq_lb_1)
  if exp_lb_0 is not None:
    return 0.5 * (exp_lb_0 + exp_lb_1)
  return None


def _phase_barrier(tree):
  """Enforces inter-phase ordering (`program_order` barrier) across an arbitrary PyTree."""
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  if not leaves:
    return tree
  return jax.tree_util.tree_unflatten(treedef, jax.lax.optimization_barrier(leaves))


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 9, 10))
def _custom_vjp_batch_split_pipeline(
    graphdef,
    x0_in,
    x1_in,
    params_leaves,
    state_leaves,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    length,
    ctx,
):
  """Runs `length` scanned MoE layers with Option B batch-split + FSDP prefetch."""
  out, _ = _custom_vjp_batch_split_pipeline_fwd(
      graphdef,
      x0_in,
      x1_in,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
      length,
      ctx,
  )
  return out


def _custom_vjp_batch_split_pipeline_fwd(
    graphdef,
    x0_in,
    x1_in,
    params_leaves,
    state_leaves,
    mb0_args_leaves,
    mb0_kwargs_leaves,
    mb1_args_leaves,
    mb1_kwargs_leaves,
    length,
    ctx: _BatchSplitCtx,
):
  """Forward pass: Option B 4-phase schedule with mb1 combine wrapped across layer boundaries."""
  (
      params,
      state,
      gather_fn,
      _,
      mb0_consts,
      mb1_consts,
      attn_route_pure,
      dispatch_pure,
      dispatch_bwd_pure,
      experts_pure,
      shared_pure,
      combine_pure,
      combine_bwd_pure,
      finalize_bias_pure,
  ) = _batch_split_setup(
      graphdef,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
      ctx,
  )
  mb0_da, mb0_dk = mb0_consts
  mb1_da, mb1_dk = mb1_consts
  w_template = fsdp_prefetch._slice_at(params, 0)  # pylint: disable=protected-access

  w_0 = gather_fn(fsdp_prefetch._slice_at(params, 0))  # pylint: disable=protected-access
  s_0 = fsdp_prefetch._slice_at(state, 0)  # pylint: disable=protected-access

  # Capture backward (and remat-saved forward) sub-function jaxprs once while Qwix/qpl interceptors are active.
  bundle: _CapturedBwdBundle = ctx.bwd_cache(
      lambda: _build_captured_bwd_bundle(
          x_sample=x0_in,
          w_sample=w_0,
          s_sample=s_0,
          d_args_sample=mb0_da,
          d_kwargs_sample=mb0_dk,
          attn_route_pure=attn_route_pure,
          dispatch_pure=dispatch_pure,
          dispatch_bwd_pure=dispatch_bwd_pure,
          experts_pure=experts_pure,
          shared_pure=shared_pure,
          combine_bwd_pure=combine_bwd_pure,
          remat_policy=ctx.remat_policy,
          prevent_cse=ctx.prevent_cse,
      ),
  )
  exp_mb0_subfn = bundle.experts_acc_subfn if bundle.experts_acc_subfn is not None else bundle.experts_subfn
  exp_mb1_subfn = bundle.experts_subfn

  def _fwd_attn_route(x_k, w_k, w_leaves, s_k, da, dk):
    if bundle.attn_route_subfn.has_saved:
      w_sub = tuple(w_leaves[i] for i in bundle.attn_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      d_outs, r_nd, saved = _eval_remat_fwd(bundle.attn_route_subfn, (x_k, w_sub), (s_l, da, dk))
      h_skip, m_in, r_d = d_outs[:3]
      s_lb = d_outs[3] if bundle.has_seq_lb else None
      r_in = _merge_inexact(r_d, r_nd, bundle.route_in_mask, bundle.route_in_treedef)
      return h_skip, m_in, r_in, s_lb, saved
    h_skip, m_in, r_in, s_lb = attn_route_pure(x_k, w_k, s_k, da, dk)
    return h_skip, m_in, r_in, s_lb, ()

  def _fwd_experts(subfn: _RematSubfn, m_in, xv, xs, r_ag, w_k, w_leaves, s_k, da, dk):
    if subfn.has_saved:
      w_sub = tuple(w_leaves[i] for i in bundle.exp_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      r_d, r_nd, _, _ = _split_inexact(r_ag)
      d_args = (xv, xs, r_d, w_sub) if bundle.has_x_scale else (xv, r_d, w_sub)
      o_args = (m_in, r_nd, s_l, da, dk)
      d_outs, cnt, saved = _eval_remat_fwd(subfn, d_args, o_args)
      y_part = d_outs[0]
      e_lb = d_outs[1] if bundle.has_exp_lb else None
      return y_part, e_lb, cnt, saved
    y_part, e_lb, cnt = experts_pure(m_in, xv, xs, r_ag, w_k, s_k, da, dk)
    return y_part, e_lb, cnt, ()

  def _fwd_shared(m_in, w_k, w_leaves, s_k):
    if bundle.shared_subfn.has_saved:
      w_sub = tuple(w_leaves[i] for i in bundle.shared_w_indices)
      s_l = jax.tree_util.tree_leaves(s_k)
      sh_out, _, saved = _eval_remat_fwd(bundle.shared_subfn, (m_in, w_sub), (s_l,))
      return sh_out, saved
    return shared_pure(m_in, w_k, s_k), ()

  def _run_layer_0(w_0_in, s_0_in):
    with jax.named_scope("batch_split_fwd_layer0"):
      w_leaves = jax.tree_util.tree_leaves(w_0_in)
      # F1(0): mb0 attn+route (no prior mb1 combine).
      h_skip_0, moe_in_0, route_in_0, seq_lb_0, attn_sv_0 = _fwd_attn_route(
          x0_in, w_0_in, w_leaves, s_0_in, mb0_da, mb0_dk
      )
      h_skip_0, moe_in_0_f2, route_in_0_f2, x1_f2 = _phase_barrier((h_skip_0, moe_in_0, route_in_0, x1_in))
      # F2(0): mb0 dispatch [SC0 AG] || mb1 attn+route [TC].
      x_val_0, x_scale_0, route_ag_0 = dispatch_pure(moe_in_0_f2, route_in_0_f2, s_0_in, w_template)
      h_skip_1, moe_in_1, route_in_1, seq_lb_1, attn_sv_1 = _fwd_attn_route(
          x1_f2, w_0_in, w_leaves, s_0_in, mb1_da, mb1_dk
      )
      x_val_0, x_scale_0, route_ag_0, h_skip_1, moe_in_1_f3, route_in_1_f3 = _phase_barrier(
          (x_val_0, x_scale_0, route_ag_0, h_skip_1, moe_in_1, route_in_1)
      )
      # F3(0): mb1 dispatch [SC0 AG] || mb0 experts [TC].
      x_val_1, x_scale_1, route_ag_1 = dispatch_pure(moe_in_1_f3, route_in_1_f3, s_0_in, w_template)
      y_part_0, exp_lb_0, cnt_0, exp_sv_0 = _fwd_experts(
          exp_mb0_subfn, moe_in_0_f2, x_val_0, x_scale_0, route_ag_0, w_0_in, w_leaves, s_0_in, mb0_da, mb0_dk
      )
      y_part_0, moe_in_0_f4, x_val_1, x_scale_1, route_ag_1, moe_in_1_f4 = _phase_barrier(
          (y_part_0, moe_in_0_f2, x_val_1, x_scale_1, route_ag_1, moe_in_1_f3)
      )
      # F4(0): mb0 combine [SC0 RS] + mb0 shared [TC] || mb1 experts [TC] + mb1 shared [TC].
      routed_0 = combine_pure(y_part_0, moe_in_0_f4, s_0_in, w_template)
      shared_0, sh_sv_0 = _fwd_shared(moe_in_0_f4, w_0_in, w_leaves, s_0_in)
      y_part_1, exp_lb_1, cnt_1, exp_sv_1 = _fwd_experts(
          exp_mb1_subfn, moe_in_1_f4, x_val_1, x_scale_1, route_ag_1, w_0_in, w_leaves, s_0_in, mb1_da, mb1_dk
      )
      shared_1, sh_sv_1 = _fwd_shared(moe_in_1_f4, w_0_in, w_leaves, s_0_in)
      # F5(0): await mb0 combine only after mb1 experts + shared finish so combine_done never stalls TC early.
      routed_0, shared_0, y_part_1, shared_1 = _phase_barrier((routed_0, shared_0, y_part_1, shared_1))
      x0_next = routed_0 + (h_skip_0 + shared_0)
      skip_shared_1 = h_skip_1 + shared_1
      lb_0 = _combine_layer_lb(seq_lb_0, seq_lb_1, exp_lb_0, exp_lb_1)
      summed_cnt_0 = None if cnt_0 is None else (cnt_0 + cnt_1)
      bias_0 = finalize_bias_pure(summed_cnt_0, s_0_in, w_template)
      res_0_out = (x0_in, x1_in, moe_in_0, route_in_0, moe_in_1, route_in_1)
      saved_0_out = (attn_sv_0, attn_sv_1, exp_sv_0, exp_sv_1, sh_sv_0, sh_sv_1)
      return x0_next, y_part_1, skip_shared_1, res_0_out, saved_0_out, lb_0, bias_0

  def _run_layer_k(x0_k, y_part_1_prev, skip_shared_1_prev, w_k, s_k):
    with jax.named_scope("batch_split_fwd_layer"):
      w_leaves = jax.tree_util.tree_leaves(w_k)
      # F1(k): mb1 combine(k-1) [SC0 RS, weight-free] || mb0 attn+route(k) [TC].
      routed_1_prev = combine_pure(y_part_1_prev, x0_k, s_k, w_template)
      h_skip_0, moe_in_0, route_in_0, seq_lb_0, attn_sv_0 = _fwd_attn_route(x0_k, w_k, w_leaves, s_k, mb0_da, mb0_dk)
      routed_1_prev, h_skip_0, moe_in_0_f2, route_in_0_f2 = _phase_barrier(
          (routed_1_prev, h_skip_0, moe_in_0, route_in_0)
      )
      x1_k = routed_1_prev + skip_shared_1_prev
      # F2(k): mb0 dispatch(k) [SC0 AG] || mb1 attn+route(k) [TC].
      x_val_0, x_scale_0, route_ag_0 = dispatch_pure(moe_in_0_f2, route_in_0_f2, s_k, w_template)
      h_skip_1, moe_in_1, route_in_1, seq_lb_1, attn_sv_1 = _fwd_attn_route(x1_k, w_k, w_leaves, s_k, mb1_da, mb1_dk)
      x_val_0, x_scale_0, route_ag_0, h_skip_1, moe_in_1_f3, route_in_1_f3 = _phase_barrier(
          (x_val_0, x_scale_0, route_ag_0, h_skip_1, moe_in_1, route_in_1)
      )
      # F3(k): mb1 dispatch(k) [SC0 AG] || mb0 experts(k) [TC].
      x_val_1, x_scale_1, route_ag_1 = dispatch_pure(moe_in_1_f3, route_in_1_f3, s_k, w_template)
      y_part_0, exp_lb_0, cnt_0, exp_sv_0 = _fwd_experts(
          exp_mb0_subfn, moe_in_0_f2, x_val_0, x_scale_0, route_ag_0, w_k, w_leaves, s_k, mb0_da, mb0_dk
      )
      y_part_0, moe_in_0_f4, x_val_1, x_scale_1, route_ag_1, moe_in_1_f4 = _phase_barrier(
          (y_part_0, moe_in_0_f2, x_val_1, x_scale_1, route_ag_1, moe_in_1_f3)
      )
      # F4(k): mb0 combine(k) [SC0 RS] + mb0 shared(k) [TC] || mb1 experts(k) [TC] + mb1 shared(k) [TC].
      routed_0 = combine_pure(y_part_0, moe_in_0_f4, s_k, w_template)
      shared_0, sh_sv_0 = _fwd_shared(moe_in_0_f4, w_k, w_leaves, s_k)
      y_part_1, exp_lb_1, cnt_1, exp_sv_1 = _fwd_experts(
          exp_mb1_subfn, moe_in_1_f4, x_val_1, x_scale_1, route_ag_1, w_k, w_leaves, s_k, mb1_da, mb1_dk
      )
      shared_1, sh_sv_1 = _fwd_shared(moe_in_1_f4, w_k, w_leaves, s_k)
      # F5(k): await mb0 combine only after mb1 experts + shared finish so combine_done never stalls TC early.
      routed_0, shared_0, y_part_1, shared_1 = _phase_barrier((routed_0, shared_0, y_part_1, shared_1))
      x0_next = routed_0 + (h_skip_0 + shared_0)
      skip_shared_1 = h_skip_1 + shared_1
      lb_k = _combine_layer_lb(seq_lb_0, seq_lb_1, exp_lb_0, exp_lb_1)
      summed_cnt_k = None if cnt_0 is None else (cnt_0 + cnt_1)
      bias_k = finalize_bias_pure(summed_cnt_k, s_k, w_template)
      res_k = (x0_k, x1_k, moe_in_0, route_in_0, moe_in_1, route_in_1)
      saved_k = (attn_sv_0, attn_sv_1, exp_sv_0, exp_sv_1, sh_sv_0, sh_sv_1)
      return x0_next, y_part_1, skip_shared_1, res_k, saved_k, lb_k, bias_k

  w_curr = gather_fn(fsdp_prefetch._slice_at(params, 1)) if length > 1 else None  # pylint: disable=protected-access
  x0_curr, y_part_1_curr, skip_shared_1_curr, res_0, saved_0, lb_0, bias_0 = _run_layer_0(w_0, s_0)

  res_mid = None
  saved_mid = None
  lb_mid = None
  bias_mid = None
  if length > 2:
    ordered = (length - 2) // 2 >= 2
    offload_mid = ctx.offload_layer_inputs and ordered

    def body(carry, xs):
      x0_k, yp1_prev, ss1_prev, w_k, res_buf = carry
      s_k, next_idx = xs
      params_next = fsdp_prefetch._slice_at(params, next_idx)  # pylint: disable=protected-access
      if ordered:
        x0_k, yp1_prev, ss1_prev, params_next = jax.lax.optimization_barrier((x0_k, yp1_prev, ss1_prev, params_next))
      w_next = gather_fn(params_next)  # SC1 AG(k+1) || layer k.
      x0_n, yp1_k, ss1_k, res_k, saved_k, lb_k, bias_k = _run_layer_k(x0_k, yp1_prev, ss1_prev, w_k, s_k)
      if offload_mid:
        res_buf = _tree_host_write(res_buf, res_k, next_idx - 2)
        return (x0_n, yp1_k, ss1_k, w_next, res_buf), (None, saved_k, lb_k, bias_k)
      return (x0_n, yp1_k, ss1_k, w_next, res_buf), (res_k, saved_k, lb_k, bias_k)

    res_buf = _tree_host_buffer(res_0, length - 2) if offload_mid else None
    (x0_curr, y_part_1_curr, skip_shared_1_curr, w_curr, res_buf), (res_mid, saved_mid, lb_mid, bias_mid) = jax.lax.scan(
        body,
        (x0_curr, y_part_1_curr, skip_shared_1_curr, w_curr, res_buf),
        (
            fsdp_prefetch._slice_range(state, 1, length - 1),  # pylint: disable=protected-access
            jnp.arange(2, length, dtype=jnp.int32),
        ),
        unroll=2,
    )
    if offload_mid:
      res_mid = res_buf

  if length > 1:
    s_last = fsdp_prefetch._slice_at(state, length - 1)  # pylint: disable=protected-access
    x0_final, y_part_1_last, skip_shared_1_last, res_last, saved_last, lb_last, bias_last = _run_layer_k(
        x0_curr, y_part_1_curr, skip_shared_1_curr, w_curr, s_last
    )
    # Epilogue: complete mb1 combine for the final layer (weight-free).
    x1_final = combine_pure(y_part_1_last, x0_final, s_last, w_template) + skip_shared_1_last
  else:
    x0_final = x0_curr
    x1_final = combine_pure(y_part_1_curr, x0_final, s_0, w_template) + skip_shared_1_curr
    res_last = res_0
    saved_last = saved_0
    lb_last = lb_0
    bias_last = bias_0

  def _stack_layers(v0, vmid, vlast):
    if v0 is None:
      return None
    if length == 1:
      return jnp.expand_dims(v0, 0)
    if length == 2:
      return jnp.stack([v0, vlast], axis=0)
    return jnp.concatenate([jnp.expand_dims(v0, 0), vmid, jnp.expand_dims(vlast, 0)], axis=0)

  stacked_lb = _stack_layers(lb_0, lb_mid, lb_last)
  stacked_bias = _stack_layers(bias_0, bias_mid, bias_last)

  residuals = (
      res_0,
      res_mid,
      res_last,
      saved_0,
      saved_mid,
      saved_last,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
  )
  return (x0_final, x1_final, stacked_lb, stacked_bias), residuals


def _custom_vjp_batch_split_pipeline_bwd(graphdef, length, ctx: _BatchSplitCtx, res, g_out):
  """Backward pass: Option B 4-phase backward with mb1 combine_bwd + dispatch_remat wrapped."""
  (
      res_0,
      res_mid,
      res_last,
      saved_0,
      saved_mid,
      saved_last,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
  ) = res
  g_x0_final, g_x1_final, g_stacked_lb, _ = g_out

  (
      params,
      state,
      gather_fn,
      rs_fn,
      mb0_consts,
      mb1_consts,
      *_,
  ) = _batch_split_setup(
      graphdef,
      params_leaves,
      state_leaves,
      mb0_args_leaves,
      mb0_kwargs_leaves,
      mb1_args_leaves,
      mb1_kwargs_leaves,
      ctx,
  )
  mb0_da, mb0_dk = mb0_consts
  mb1_da, mb1_dk = mb1_consts
  bundle: _CapturedBwdBundle = ctx.bwd_cache(None)
  n_w_total = len(params_leaves)
  w_treedef = ctx.params_treedef

  def _run_combine_bwd(g_x, m_ref, s_k):
    s_l = jax.tree_util.tree_leaves(s_k)
    out = _eval_jaxpr_flat(bundle.combine_bwd, g_x, m_ref, s_l)
    if bundle.has_deferred_combine_bwd:
      return (out[0], out[1])
    return out[0]

  def _run_dispatch_fwd(m_in, r_in, s_k):
    r_diff, r_nd, _, _ = _split_inexact(r_in)
    s_l = jax.tree_util.tree_leaves(s_k)
    flat_out = _eval_jaxpr_flat(bundle.dispatch_fwd, m_in, r_diff, r_nd, s_l)
    pos = 0
    xv = flat_out[pos]
    pos += 1
    xs = flat_out[pos] if bundle.has_x_scale else None
    if bundle.has_x_scale:
      pos += 1
    n_rd = sum(bundle.route_ag_mask)
    n_rnd = len(bundle.route_ag_mask) - n_rd
    rag_diff = tuple(flat_out[pos : pos + n_rd])
    pos += n_rd
    rag_nd = tuple(flat_out[pos : pos + n_rnd])
    return (xv, xs, rag_diff, rag_nd)

  def _run_dispatch_bwd(m_in, r_in, g_xv, g_xs, g_rag_diff, s_k):
    del g_xs  # Explicit dispatch_bwd unscales via static/absmax calibration from m_in.
    r_diff, r_nd, _, _ = _split_inexact(r_in)
    s_l = jax.tree_util.tree_leaves(s_k)
    flat_out = _eval_jaxpr_flat(bundle.dispatch_bwd, m_in, r_diff, r_nd, g_xv, g_rag_diff, s_l)
    return flat_out[0], tuple(flat_out[1:])

  def _run_experts_bwd(m_ref, disp, w_leaves, s_k, da, dk, exp_saved, g_yp, g_elb, dw_partial_from_mb1=None):
    if isinstance(g_yp, tuple):
      g_yp = (g_yp[0].astype(jnp.float32) * g_yp[1]).astype(m_ref.dtype)
    xv, xs, rag_diff, rag_nd = disp
    w_sub = tuple(w_leaves[i] for i in bundle.exp_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    d_args = (xv, xs, rag_diff, w_sub) if bundle.has_x_scale else (xv, rag_diff, w_sub)
    o_args = (m_ref, rag_nd, s_l, da, dk)
    if dw_partial_from_mb1 is not None and bundle.experts_acc_subfn is not None:
      dw_part = tuple(dw_partial_from_mb1[j] for j in bundle.exp_accum_sub_indices)
      cot = (g_yp, g_elb, dw_part) if bundle.has_exp_lb else (g_yp, dw_part)
      grads = _eval_remat_bwd(bundle.experts_acc_subfn, d_args, o_args, exp_saved, cot)
    else:
      cot = (g_yp, g_elb) if bundle.has_exp_lb else (g_yp,)
      grads = _eval_remat_bwd(bundle.experts_subfn, d_args, o_args, exp_saved, cot)
    if bundle.has_x_scale:
      g_xv, g_xs, g_rag_diff, dw_sub = grads
    else:
      g_xv, g_rag_diff, dw_sub = grads
      g_xs = None
    return (g_xv, g_xs, g_rag_diff), dw_sub

  def _run_shared_bwd(m_in, w_leaves, s_k, sh_saved, g_sh):
    w_sub = tuple(w_leaves[i] for i in bundle.shared_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    g_m, dw_sub = _eval_remat_bwd(bundle.shared_subfn, (m_in, w_sub), (s_l,), sh_saved, g_sh)
    return g_m, dw_sub

  def _run_attn_route_bwd(x_in_k, w_leaves, s_k, da, dk, attn_saved, g_h_skip, g_m_in, g_r_diff, g_slb):
    w_sub = tuple(w_leaves[i] for i in bundle.attn_w_indices)
    s_l = jax.tree_util.tree_leaves(s_k)
    cot = (g_h_skip, g_m_in, g_r_diff, g_slb) if bundle.has_seq_lb else (g_h_skip, g_m_in, g_r_diff)
    g_x, dw_sub = _eval_remat_bwd(bundle.attn_route_subfn, (x_in_k, w_sub), (s_l, da, dk), attn_saved, cot)
    return g_x, dw_sub

  def _assemble_dw(w_leaves, dw_exp_0, dw_exp_1, dw_sh_0, dw_sh_1, dw_attn_0, dw_attn_1):
    """Sums mb0 + mb1 gradients locally per sub-function and places them into the layer dW tree."""
    accum: list[Any] = [None] * n_w_total
    exp_acc_set = frozenset(bundle.exp_accum_sub_indices)

    def _add_into(indices, g0_tuple, g1_tuple, acc_skip=frozenset()):
      for sub_i, (idx, g0, g1) in enumerate(zip(indices, g0_tuple, g1_tuple)):
        g_sum = g0 if sub_i in acc_skip else (g0 + g1)
        accum[idx] = g_sum if accum[idx] is None else (accum[idx] + g_sum)

    _add_into(bundle.exp_w_indices, dw_exp_0, dw_exp_1, exp_acc_set)
    _add_into(bundle.shared_w_indices, dw_sh_0, dw_sh_1)
    _add_into(bundle.attn_w_indices, dw_attn_0, dw_attn_1)
    # Any parameter leaf untouched by all sub-functions gets an explicit zero matching w_leaves[i].
    for i in range(n_w_total):
      if accum[i] is None:
        accum[i] = jnp.zeros_like(w_leaves[i])
    return jax.tree_util.tree_unflatten(w_treedef, accum)

  def _bwd_layer(
      g_x0_out,
      g_x1_out,
      g_yp1_curr,
      disp_1_curr,
      g_lb_k,
      w_k,
      s_k,
      res_k,
      saved_k,
      prev_mb1_res,
  ):
    """Runs Option B phases B4 -> B3 -> B2 -> B1 for one layer."""
    with jax.named_scope("batch_split_bwd_layer"):
      x0_k, x1_k, moe_in_0, route_in_0, moe_in_1, route_in_1 = res_k
      attn_sv_0, attn_sv_1, exp_sv_0, exp_sv_1, sh_sv_0, sh_sv_1 = saved_k
      w_leaves = jax.tree_util.tree_leaves(w_k)
      g_lb_half = None if g_lb_k is None else (0.5 * g_lb_k)
      g_exp_lb = g_lb_half if bundle.has_exp_lb else None
      g_seq_lb = g_lb_half if bundle.has_seq_lb else None

      # B4(k): SC0 AG (mb0 combine_bwd + dispatch_remat) || TC (mb1 experts_bwd + mb1/mb0 shared_bwd).
      g_yp0 = _run_combine_bwd(g_x0_out, moe_in_0, s_k)
      disp_0 = _run_dispatch_fwd(moe_in_0, route_in_0, s_k)
      g_disp_1, dw_exp_1 = _run_experts_bwd(
          moe_in_1, disp_1_curr, w_leaves, s_k, mb1_da, mb1_dk, exp_sv_1, g_yp1_curr, g_exp_lb
      )
      g_moe_1_sh, dw_sh_1 = _run_shared_bwd(moe_in_1, w_leaves, s_k, sh_sv_1, g_x1_out)
      g_moe_0_sh, dw_sh_0 = _run_shared_bwd(moe_in_0, w_leaves, s_k, sh_sv_0, g_x0_out)
      g_yp0, disp_0, g_disp_1, dw_exp_1, g_moe_1_sh, g_moe_0_sh = _phase_barrier(
          (g_yp0, disp_0, g_disp_1, dw_exp_1, g_moe_1_sh, g_moe_0_sh)
      )

      # B3(k): SC0 RS (mb1 dispatch_bwd) || TC (mb0 experts_bwd accumulating onto dw_exp_1 in place).
      g_moe_1_rt, g_route_1_diff = _run_dispatch_bwd(moe_in_1, route_in_1, *g_disp_1, s_k)
      g_disp_0, dw_exp_0 = _run_experts_bwd(
          moe_in_0, disp_0, w_leaves, s_k, mb0_da, mb0_dk, exp_sv_0, g_yp0, g_exp_lb, dw_partial_from_mb1=dw_exp_1
      )
      g_moe_1_rt, g_route_1_diff, g_disp_0, dw_exp_0 = _phase_barrier((g_moe_1_rt, g_route_1_diff, g_disp_0, dw_exp_0))

      # B2(k): SC0 RS (mb0 dispatch_bwd) || TC (mb1 attn_route_bwd).
      g_moe_0_rt, g_route_0_diff = _run_dispatch_bwd(moe_in_0, route_in_0, *g_disp_0, s_k)
      g_moe_1 = g_moe_1_rt + g_moe_1_sh
      g_x1_in, dw_attn_1 = _run_attn_route_bwd(
          x1_k, w_leaves, s_k, mb1_da, mb1_dk, attn_sv_1, g_x1_out, g_moe_1, g_route_1_diff, g_seq_lb
      )
      g_moe_0_rt, g_route_0_diff, g_x1_in, dw_attn_1 = _phase_barrier((g_moe_0_rt, g_route_0_diff, g_x1_in, dw_attn_1))

      # B1(k): SC0 AG (mb1 combine_bwd(k-1) + dispatch_remat(k-1)) || TC (mb0 attn_route_bwd).
      if prev_mb1_res is not None:
        prev_moe_in_1, prev_route_in_1 = prev_mb1_res
        g_yp1_next = _run_combine_bwd(g_x1_in, prev_moe_in_1, s_k)
        disp_1_next = _run_dispatch_fwd(prev_moe_in_1, prev_route_in_1, s_k)
      else:
        g_yp1_next, disp_1_next = None, None

      g_moe_0 = g_moe_0_rt + g_moe_0_sh
      g_x0_in, dw_attn_0 = _run_attn_route_bwd(
          x0_k, w_leaves, s_k, mb0_da, mb0_dk, attn_sv_0, g_x0_out, g_moe_0, g_route_0_diff, g_seq_lb
      )
      g_yp1_next, disp_1_next, g_x0_in, dw_attn_0 = _phase_barrier((g_yp1_next, disp_1_next, g_x0_in, dw_attn_0))

      dw_k = _assemble_dw(w_leaves, dw_exp_0, dw_exp_1, dw_sh_0, dw_sh_1, dw_attn_0, dw_attn_1)
      return g_x0_in, g_x1_in, g_yp1_next, disp_1_next, dw_k

  def write(d_acc, dw_tree, idx, which):
    acc_leaves, treedef = jax.tree_util.tree_flatten(d_acc)
    d_leaves = jax.tree_util.tree_leaves(dw_tree)
    flags = rs_fn.tag_flags(d_leaves) if which != "all" else [None] * len(d_leaves)
    out = [
        jax.lax.dynamic_update_index_in_dim(a, d, idx, axis=0) if which == "all" or (which == "tagged") == f else a
        for a, d, f in zip(acc_leaves, d_leaves, flags)
    ]
    return jax.tree_util.tree_unflatten(treedef, out)

  def deferred(dw_tree):
    leaves, treedef = jax.tree_util.tree_flatten(dw_tree)
    flags = rs_fn.tag_flags(leaves)
    return jax.tree_util.tree_unflatten(treedef, [x if f else jnp.zeros((0,), x.dtype) for x, f in zip(leaves, flags)])

  last = length - 1
  ordered = length > 2 and ((length - 2) // 2 >= 2)
  offload_mid = length > 2 and ctx.offload_layer_inputs and ordered

  params_last = fsdp_prefetch._slice_at(params, last)  # pylint: disable=protected-access
  params_prev_init = fsdp_prefetch._slice_at(params, last - 1) if length > 1 else None  # pylint: disable=protected-access
  if length > 1:
    g_x0_final, g_x1_final, params_last, params_prev_init = jax.lax.optimization_barrier(
        (g_x0_final, g_x1_final, params_last, params_prev_init)
    )
  else:
    g_x0_final, g_x1_final, params_last = jax.lax.optimization_barrier((g_x0_final, g_x1_final, params_last))

  w_last = gather_fn(params_last)
  w_prev = gather_fn(params_prev_init) if length > 1 else None  # SC1 AG(last - 1) || bwd(last).
  s_last = fsdp_prefetch._slice_at(state, last)  # pylint: disable=protected-access

  # Prologue: mb1 combine_bwd + dispatch_remat for the final layer.
  g_yp1_init = _run_combine_bwd(g_x1_final, res_last[4], s_last)
  disp_1_init = _run_dispatch_fwd(res_last[4], res_last[5], s_last)

  if length == 1:
    prev_mb1_for_last = None
  elif length == 2:
    prev_mb1_for_last = (res_0[4], res_0[5])
  elif offload_mid:
    res_lm2 = _tree_host_read(res_mid, length - 3)
    prev_mb1_for_last = (res_lm2[4], res_lm2[5])
  else:
    prev_mb1_for_last = (
        fsdp_prefetch._slice_at(res_mid[4], length - 3),  # pylint: disable=protected-access
        fsdp_prefetch._slice_at(res_mid[5], length - 3),  # pylint: disable=protected-access
    )

  g_lb_last = None if g_stacked_lb is None else g_stacked_lb[last]
  g_x0, g_x1, g_yp1, disp_1, dw_last = _bwd_layer(
      g_x0_final,
      g_x1_final,
      g_yp1_init,
      disp_1_init,
      g_lb_last,
      w_last,
      s_last,
      res_last,
      saved_last,
      prev_mb1_for_last,
  )

  d_params = jax.tree.map(jnp.zeros_like, params)
  if length == 1:
    d_params = write(d_params, rs_fn(dw_last), last, "all")
  else:
    d_params = write(d_params, rs_fn(dw_last, "untagged"), last, "untagged")
    pending = deferred(dw_last)

    if length > 2:
      res_0_mb1 = (res_0[4], res_0[5])

      def body(carry, xs):
        g0_k, g1_k, gyp1_k, d1_k, w_k, d_acc, pend = carry
        res_k, saved_k, s_k, g_lb_k, k_idx = xs
        if offload_mid:
          res_k = _tree_host_read(res_mid, k_idx - 1)
          prev_mb1_mid = _tree_host_read((res_mid[4], res_mid[5]), jnp.maximum(k_idx - 2, 0))
        else:
          prev_idx = jnp.maximum(k_idx - 2, 0)
          prev_mb1_mid = (
              fsdp_prefetch._slice_at(res_mid[4], prev_idx),  # pylint: disable=protected-access
              fsdp_prefetch._slice_at(res_mid[5], prev_idx),  # pylint: disable=protected-access
          )
        prev_mb1 = jax.tree.map(lambda r0, rm: jnp.where(k_idx == 1, r0, rm), res_0_mb1, prev_mb1_mid)
        params_prev = fsdp_prefetch._slice_at(params, k_idx - 1)  # pylint: disable=protected-access
        if ordered:
          g0_k, g1_k, gyp1_k, d1_k, res_k, saved_k, params_prev = jax.lax.optimization_barrier(
              (g0_k, g1_k, gyp1_k, d1_k, res_k, saved_k, params_prev)
          )
        d_acc = write(d_acc, rs_fn(pend, "tagged"), k_idx + 1, "tagged")  # SC1 RS(k+1) || bwd(k).
        w_km1 = gather_fn(params_prev)  # SC1 AG(k-1) || bwd(k).
        g0_km1, g1_km1, gyp1_km1, d1_km1, dw_k = _bwd_layer(
            g0_k, g1_k, gyp1_k, d1_k, g_lb_k, w_k, s_k, res_k, saved_k, prev_mb1
        )
        dw_k = rs_fn(dw_k, "untagged")
        if ordered:
          g0_km1, g1_km1, gyp1_km1, d1_km1, dw_k = jax.lax.optimization_barrier((g0_km1, g1_km1, gyp1_km1, d1_km1, dw_k))
        d_acc = write(d_acc, dw_k, k_idx, "untagged")
        return (g0_km1, g1_km1, gyp1_km1, d1_km1, w_km1, d_acc, deferred(dw_k)), None

      res_xs = jnp.zeros((length - 2, 0), jnp.int8) if offload_mid else res_mid
      g_lb_mid = None if g_stacked_lb is None else g_stacked_lb[1 : length - 1]
      (g_x0, g_x1, g_yp1, disp_1, w_prev, d_params, pending), _ = jax.lax.scan(
          body,
          (g_x0, g_x1, g_yp1, disp_1, w_prev, d_params, pending),
          (
              res_xs,
              saved_mid,
              fsdp_prefetch._slice_range(state, 1, length - 1),  # pylint: disable=protected-access
              g_lb_mid,
              jnp.arange(1, length - 1, dtype=jnp.int32),
          ),
          reverse=True,
          unroll=2,
      )

    d_params = write(d_params, rs_fn(pending, "tagged"), 1, "tagged")  # SC1 RS(1) || bwd(0).
    s_0 = fsdp_prefetch._slice_at(state, 0)  # pylint: disable=protected-access
    g_lb_0 = None if g_stacked_lb is None else g_stacked_lb[0]
    g_x0, g_x1, _, _, dw_0 = _bwd_layer(g_x0, g_x1, g_yp1, disp_1, g_lb_0, w_prev, s_0, res_0, saved_0, None)
    d_params = write(d_params, rs_fn(dw_0), 0, "all")

  params_bar_leaves = jax.tree_util.tree_leaves(d_params)
  if ctx.scan_axis != 0:
    params_bar_leaves = [
        jnp.moveaxis(p, 0, ctx.scan_axis) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_bar_leaves
    ]
  params_bar_leaves = tuple(params_bar_leaves)

  mb0_args_bar = tuple(jnp.zeros_like(x) for x in mb0_args_leaves)
  mb0_kwargs_bar = tuple(jnp.zeros_like(x) for x in mb0_kwargs_leaves)
  mb1_args_bar = tuple(jnp.zeros_like(x) for x in mb1_args_leaves)
  mb1_kwargs_bar = tuple(jnp.zeros_like(x) for x in mb1_kwargs_leaves)
  return (g_x0, g_x1, params_bar_leaves, None, mb0_args_bar, mb0_kwargs_bar, mb1_args_bar, mb1_kwargs_bar)


_custom_vjp_batch_split_pipeline.defvjp(_custom_vjp_batch_split_pipeline_fwd, _custom_vjp_batch_split_pipeline_bwd)


def apply_layers_with_batch_split(
    layers,
    x_in,
    *args,
    length: int,
    mesh,
    config,
    remat_policy: Any = None,
    prevent_cse: bool = True,
    **kwargs,
):
  """Runs a homogeneous scanned NNX MoE layer stack with Option B generalized batch-split.

  Args:
    layers: The scanned (layer-stacked) NNX MoE module implementing the batch-split adapter.
    x_in: The input activation of the first layer, shape `(batch, seq, emb)`.
    *args: Positional arguments forwarded to each layer call.
    length: Number of layers in the stack.
    mesh: The device mesh.
    config: The MaxText config.
    remat_policy: Optional JAX checkpoint policy (None for full remat).
    prevent_cse: Forwarded to `jax.checkpoint` when `remat_policy` is non-None.
    **kwargs: Keyword arguments forwarded to each layer call.

  Returns:
    `(final_y, layers, None)` with `moe_lb_loss` and `moe_bias_updates` sown onto `layers`
    when enabled.
  """
  if length == 0:
    return x_in, layers, None

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

  fsdp_prefetch._register_overlap_transform(mesh)  # pylint: disable=protected-access
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  scan_axis = config.param_scan_axis
  params, state = fsdp_prefetch._layer_major(params, state, scan_axis, length, move_scan_axis=False)  # pylint: disable=protected-access

  sig = inspect.signature(layers.__class__.__call__)
  valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

  check_vma = bool(config.check_vma) if config.shard_mode == ShardMode.EXPLICIT else False
  if config.shard_mode == ShardMode.EXPLICIT and x_in.ndim == 3:
    rules = sharding.get_logical_axis_rules() or getattr(config, "logical_axis_rules", ())
    norm_pspec = sharding.logical_to_mesh_axes(
        ("activation_batch", "activation_norm_length", "activation_embed"), mesh, rules=rules
    )
    if (
        norm_pspec is not None
        and len(norm_pspec) == 3
        and all(
            d % math.prod(mesh.shape.get(a, 1) for a in _axes_for_dim(spec)) == 0
            for d, spec in zip(x_in.shape, norm_pspec)
        )
    ):
      x_in = sharding.maybe_shard_with_pspec(x_in, norm_pspec, mesh, config.shard_mode)
  x_pspec = _infer_activation_pspec(x_in, mesh, config)
  global_batch = x_in.shape[0]
  x0_in, x1_in = _split_array_microbatches(x_in, x_pspec, mesh, check_vma)

  def _maybe_split_leaf(leaf):
    if _is_diff_leaf(leaf) and leaf.ndim >= 1 and leaf.shape[0] == global_batch:
      pspec = _infer_activation_pspec(leaf, mesh, config)
      return _split_array_microbatches(leaf, pspec, mesh, check_vma)
    return leaf, leaf

  args_leaves, args_treedef = jax.tree_util.tree_flatten(tuple(args))
  kwargs_leaves, kwargs_treedef = jax.tree_util.tree_flatten(tuple(valid_kwargs.items()))

  dummy_arr = jnp.zeros((0,), dtype=jnp.float32)
  mb0_args_leaves, mb1_args_leaves, static_args_leaves = [], [], []
  for leaf in args_leaves:
    if _is_diff_leaf(leaf):
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
    if _is_diff_leaf(leaf):
      l0, l1 = _maybe_split_leaf(leaf)
      mb0_kwargs_leaves.append(l0)
      mb1_kwargs_leaves.append(l1)
      static_kwargs_leaves.append(None)
    else:
      mb0_kwargs_leaves.append(dummy_arr)
      mb1_kwargs_leaves.append(dummy_arr)
      static_kwargs_leaves.append(leaf)

  params_leaves, params_treedef = jax.tree_util.tree_flatten(params)
  state_leaves, state_treedef = jax.tree_util.tree_flatten(state)

  bwd_box = {}

  def bwd_cache(build=None):
    if build is not None:
      bwd_box["bundle"] = build()
    if "bundle" not in bwd_box:
      raise RuntimeError("batch_split backward bundle was not populated during forward trace.")
    return bwd_box["bundle"]

  sparse_core_id = fsdp_prefetch._default_sparse_core_id(mesh)  # pylint: disable=protected-access
  replicated_red_axes: tuple[tuple[str, ...], ...] = ()
  if use_fsdp_prefetch and config.shard_mode == ShardMode.EXPLICIT:
    layer_shardings_probe = [_per_layer_sharding_at_axis(p, mesh, scan_axis) for p in params_leaves]
    p0_sds_leaves = [
        jax.ShapeDtypeStruct(
            p.shape[:scan_axis] + p.shape[scan_axis + 1 :] if getattr(p, "ndim", 0) > scan_axis else p.shape,
            p.dtype,
            sharding=s,
        )
        for p, s in zip(params_leaves, layer_shardings_probe)
    ]
    s0_sds_tree = jax.eval_shape(lambda st: fsdp_prefetch._slice_at(st, 0), state)  # pylint: disable=protected-access
    replicated_red_axes = _probe_replicated_reduction_axes(
        per_layer_params_leaves=p0_sds_leaves,
        params_treedef=params_treedef,
        layer_shardings=layer_shardings_probe,
        s0_tree=s0_sds_tree,
        x0_sample=x0_in,
        mb0_args_leaves=mb0_args_leaves,
        mb0_kwargs_leaves=mb0_kwargs_leaves,
        static_args_leaves=tuple(static_args_leaves),
        static_kwargs_leaves=tuple(static_kwargs_leaves),
        args_treedef=args_treedef,
        kwargs_treedef=kwargs_treedef,
        graphdef=graphdef,
        mesh=mesh,
        shard_mode=config.shard_mode,
        debug_sharding=config.debug_sharding,
        sparse_core_id=sparse_core_id,
        x_pspec=x_pspec,
    )

  ctx = _BatchSplitCtx(
      mesh=mesh,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      scan_axis=scan_axis,
      static_args_leaves=tuple(static_args_leaves),
      static_kwargs_leaves=tuple(static_kwargs_leaves),
      params_treedef=params_treedef,
      state_treedef=state_treedef,
      args_treedef=args_treedef,
      kwargs_treedef=kwargs_treedef,
      use_fsdp_prefetch=use_fsdp_prefetch,
      defer_grad_reduce_scatter=bool(getattr(config, "prefetch_fsdp_defer_grad_reduce_scatter", False))
      and use_fsdp_prefetch,
      offload_layer_inputs=getattr(config, "decoder_layer_input", "device") == "offload",
      sparse_core_id=sparse_core_id,
      bwd_cache=bwd_cache,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
      replicated_red_axes=replicated_red_axes,
  )

  x0_final, x1_final, stacked_lb, stacked_bias = _custom_vjp_batch_split_pipeline(
      graphdef,
      x0_in,
      x1_in,
      tuple(params_leaves),
      tuple(state_leaves),
      tuple(mb0_args_leaves),
      tuple(mb0_kwargs_leaves),
      tuple(mb1_args_leaves),
      tuple(mb1_kwargs_leaves),
      length,
      ctx,
  )

  out_pspec = _infer_activation_pspec(x0_final, mesh, config)
  final_y = _merge_array_microbatches(x0_final, x1_final, out_pspec, mesh, check_vma)

  if config.load_balance_loss_weight > 0.0 and stacked_lb is not None:
    if hasattr(layers, "moe_lb_loss"):
      delattr(layers, "moe_lb_loss")
    layers.sow(nnx.Intermediate, "moe_lb_loss", stacked_lb)
  if config.routed_bias and config.routed_bias_update_rate > 0.0 and stacked_bias is not None:
    if hasattr(layers, "moe_bias_updates"):
      delattr(layers, "moe_bias_updates")
    layers.sow(nnx.Intermediate, "moe_bias_updates", stacked_bias)

  return final_y, layers, None
