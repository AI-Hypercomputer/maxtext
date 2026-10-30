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

"""FSDP weight prefetching for scanned decoder layers (ShardMode.AUTO + hoisted gather8 + fwd SC[0] pinning)."""

import collections
import contextlib
import dataclasses
import functools
import inspect

from flax import nnx
import jax
from jax import ad_checkpoint
from jax.experimental import xla_metadata
from jax.experimental.compute_on import compute_on
from jax.experimental.layout import Layout, with_layout_constraint
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import ShardMode
from maxtext.layers import quantizations
from maxtext.utils import max_logging, maxtext_utils_nnx, sharding


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class PrefetchedFp8Codes:
  """Prefetched FP8 codes + per-channel scale for gather8 MoE weights (zero bf16 carry overhead)."""

  codes: jax.Array
  scale: jax.Array


def _maybe_scheduling_group(group_id: int | None):
  if group_id is None:
    return contextlib.nullcontext()
  return xla_metadata.set_xla_metadata(_scheduling_group_id=group_id)


def _apply_sharding_hint(
    w, src_sharding: NamedSharding | None, mesh, shard_mode, debug_sharding=False, pin_sc_id: int = -1
):
  """All-gathers `w` over the FSDP mesh axes by resharding it to its FSDP-stripped sharding."""
  w_sharding = src_sharding or getattr(jax.typeof(w), "sharding", None) or getattr(w, "sharding", None)
  if not isinstance(w_sharding, NamedSharding) or w_sharding.spec is None or len(w_sharding.spec) == 0:
    return w
  src_sharding = NamedSharding(mesh, w_sharding.spec)
  gathered_sharding = sharding.remove_fsdp_from_sharding(src_sharding)
  if gathered_sharding.spec == src_sharding.spec:
    return w
  if shard_mode == ShardMode.EXPLICIT:
    out = _explicit_fsdp_all_gather(
        w,
        mesh,
        src_sharding.spec,
        gathered_sharding.spec,
        reduced=True,
        pin_sc_id=pin_sc_id,
    )
    if out is not None:
      return out
    return sharding.maybe_shard_with_name(w, gathered_sharding, shard_mode=shard_mode, debug_sharding=debug_sharding)

  # ShardMode.AUTO: use native GSPMD sharding constraints (matching moe.py) to preserve native tiled layouts & fusions.
  w = jax.lax.with_sharding_constraint(w, src_sharding)
  if pin_sc_id >= 0:

    @functools.partial(
        compute_on,
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [pin_sc_id]}},
    )
    def _pinned_gather(x):
      return jax.lax.with_sharding_constraint(x, gathered_sharding)

    return _pinned_gather(w)
  return jax.lax.with_sharding_constraint(w, gathered_sharding)


def _explicit_fsdp_all_gather(w, mesh, src_spec, gathered_spec, reduced=False, pin_sc_id: int = -1):
  """All-gathers `w` over its FSDP mesh axes with an explicit shard_map all_gather (ShardMode.EXPLICIT only)."""
  spec = tuple(src_spec) + (None,) * (w.ndim - len(tuple(src_spec)))
  dims = []
  for i, entry in enumerate(spec):
    axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
    axes = tuple(a for a in axes if mesh.shape[a] > 1)
    fsdp = sharding.fsdp_axes_for_dim(axes)
    if not fsdp:
      continue
    if axes[len(axes) - len(fsdp) :] != fsdp:
      return None
    dims.append((i, fsdp))
  if not dims:
    return None

  def _maybe_pin_sc(fn):
    if pin_sc_id >= 0 and len(dims) == 1 and len(dims[0][1]) == 1:
      return compute_on(
          compute_type="tpu_sparsecore",
          out_memory_spaces=jax.memory.Space.Device,
          compiler_options={"sparse_core_config": {"core_ids": [pin_sc_id]}},
      )(fn)
    return fn

  if reduced and len(dims) == 1:
    (dim, ax) = dims[0]
    gathered = tuple(gathered_spec) + (None,) * (w.ndim - len(tuple(gathered_spec)))
    in_spec = []
    for entry in spec:
      entry_axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
      stripped = set(sharding.fsdp_axes_for_dim(entry_axes))
      kept = tuple(a for a in entry_axes if a not in stripped or mesh.shape[a] > 1)
      in_spec.append(kept if len(kept) > 1 else (kept[0] if kept else None))
    in_spec = jax.sharding.PartitionSpec(*in_spec)
    if tuple(in_spec) != spec:
      w = jax.sharding.reshard(w, NamedSharding(mesh, in_spec))

    def gather_reduced(x):
      return jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=dim, tiled=True, to="reduced")

    with xla_metadata.set_xla_metadata(xla_explicit_fsdp="yes"):
      fn = jax.shard_map(
          gather_reduced,
          mesh=mesh,
          in_specs=in_spec,
          out_specs=jax.sharding.PartitionSpec(*gathered, reduced=frozenset(ax)),
      )
      return _maybe_pin_sc(fn)(w)

  def gather(x):
    for i, ax in dims:
      x = jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=i, tiled=True)
    return x

  with xla_metadata.set_xla_metadata(xla_explicit_fsdp="yes"):
    fn = jax.shard_map(
        gather, mesh=mesh, in_specs=jax.sharding.PartitionSpec(*spec), out_specs=gathered_spec, check_vma=False
    )
    return _maybe_pin_sc(fn)(w)


def _fsdp_scatter_dim(ndim, mesh, src_spec):
  """Returns (dim, fsdp_axes) of the single dimension a gradient is reduce-scattered over, else None."""
  spec = tuple(src_spec) + (None,) * (ndim - len(tuple(src_spec)))
  dims = []
  for i, entry in enumerate(spec):
    axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
    axes = tuple(a for a in axes if mesh.shape[a] > 1)
    fsdp = sharding.fsdp_axes_for_dim(axes)
    if not fsdp:
      continue
    if axes[len(axes) - len(fsdp) :] != fsdp:
      return None
    dims.append((i, fsdp))
  if len(dims) != 1:
    return None
  dim, fsdp_axes = dims[0]
  return dim, fsdp_axes


def _is_unreduced_over(g, axes):
  g_spec = getattr(getattr(jax.typeof(g), "sharding", None), "spec", None)
  return g_spec is not None and frozenset(g_spec.unreduced) == frozenset(axes)


def _is_auto_deferred_partial_sum(g, target_shape, mesh, src_spec):
  """Detects ShardMode.AUTO deferred partial sum where dim `scatter_dim` is scaled by FSDP size."""
  scatter = _fsdp_scatter_dim(g.ndim, mesh, src_spec)
  if scatter is None or g.ndim != len(target_shape):
    return False
  dim, fsdp_axes = scatter
  fsdp_prod = 1
  for a in fsdp_axes:
    fsdp_prod *= mesh.shape[a]
  expected = list(target_shape)
  expected[dim] *= fsdp_prod
  return tuple(g.shape) == tuple(expected)


def _pin_scatter_dim_major(g, dim):
  """Lays `g` out with dimension `dim` major-most before it is reduce-scattered over `dim`."""
  return with_layout_constraint(g, Layout((dim,) + tuple(i for i in range(g.ndim) if i != dim)))


def _explicit_fsdp_reduce_scatter(g, mesh, src_spec, gathered_spec, target_shape=None):
  """Reduce-scatters a gathered-weight gradient `g` over its FSDP mesh axes (never compute_on-wrapped!)."""
  scatter = _fsdp_scatter_dim(g.ndim, mesh, src_spec)
  if scatter is None:
    return None
  dim, ax = scatter
  spec = tuple(src_spec) + (None,) * (g.ndim - len(tuple(src_spec)))
  gathered = tuple(gathered_spec) + (None,) * (g.ndim - len(tuple(gathered_spec)))

  if _is_unreduced_over(g, ax):
    g_spec = jax.typeof(g).sharding.spec
    g = _pin_scatter_dim_major(g, dim)

    def psum_scatter_explicit(x):
      return jax.lax.psum_scatter(x, ax if len(ax) > 1 else ax[0], scatter_dimension=dim, tiled=True)

    return jax.shard_map(
        psum_scatter_explicit,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(*gathered, unreduced=frozenset(ax), unreduced_kind=g_spec.unreduced_kind),
        out_specs=jax.sharding.PartitionSpec(*spec),
    )(g)

  if target_shape is not None and _is_auto_deferred_partial_sum(g, target_shape, mesh, src_spec):
    g = jax.lax.with_sharding_constraint(g, NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)))

    def psum_scatter_auto(x):
      return jax.lax.psum_scatter(x, ax if len(ax) > 1 else ax[0], scatter_dimension=dim, tiled=True)

    return jax.shard_map(
        psum_scatter_auto,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(*spec),
        out_specs=jax.sharding.PartitionSpec(*spec),
        check_vma=False,
    )(g)

  return None


def _per_layer_sharding(stacked, mesh):
  """Sharding of one layer slice `stacked[i]` of a scanned (layer-major) parameter, or None."""
  s = getattr(jax.typeof(stacked), "sharding", None) or getattr(stacked, "sharding", None)
  if not isinstance(s, NamedSharding) or s.spec is None or len(s.spec) == 0:
    return None
  return NamedSharding(mesh, jax.sharding.PartitionSpec(*tuple(s.spec)[1:]))


def _resolve_layer_shardings_from_params(params, mesh, config) -> tuple[NamedSharding | None, ...]:
  """Resolves physical per-layer NamedShardings from nnx.Variable logical metadata (works in ShardMode.AUTO & EXPLICIT)."""
  rules = getattr(config, "logical_axis_rules", None)
  default_scan_ax = getattr(config, "param_scan_axis", 0)
  var_leaves = jax.tree_util.tree_leaves(params, is_leaf=lambda v: isinstance(v, nnx.Variable))
  out = []
  for var in var_leaves:
    if isinstance(var, nnx.Variable):
      val = var.get_value()
      meta = var.get_metadata()
      names = meta.get("sharding_names") or meta.get("out_sharding")
      if names is not None:
        scan_ax = meta.get("param_scan_axis", default_scan_ax)
        if isinstance(names, tuple) and len(names) > scan_ax and names[scan_ax] == "layers":
          per_names = names[:scan_ax] + names[scan_ax + 1 :]
        else:
          per_names = tuple(n for n in names if n != "layers")
        out.append(sharding.create_sharding(mesh, per_names, rules=rules))
        continue
      out.append(_per_layer_sharding(val, mesh))
    else:
      out.append(_per_layer_sharding(var, mesh))
  return tuple(out)


def _is_routed_moe_gather8_leaf(stacked, layer_sharding, mesh, fp8_gather8: bool) -> bool:
  """Whether `stacked` is a 3D routed MoE weight `[L, E, D, F]` sharded `P(('expert', 'fsdp'), None, None)`."""
  if not fp8_gather8:
    return False
  ndim = getattr(stacked, "ndim", 0) - 1
  if ndim != 3 or layer_sharding is None or layer_sharding.spec is None or len(layer_sharding.spec) < 1:
    return False
  spec = tuple(layer_sharding.spec) + (None,) * (3 - len(tuple(layer_sharding.spec)))
  axes0 = tuple(a for a in sharding.mesh_axes_for_dim(spec[0]) if mesh.shape.get(a, 1) > 1)
  axes1 = tuple(a for a in sharding.mesh_axes_for_dim(spec[1]) if mesh.shape.get(a, 1) > 1)
  axes2 = tuple(a for a in sharding.mesh_axes_for_dim(spec[2]) if mesh.shape.get(a, 1) > 1)
  fsdp0 = sharding.fsdp_axes_for_dim(axes0)
  return bool(axes0 and axes0[0] == "expert" and fsdp0 and not axes1 and not axes2)


def compute_stacked_gather8_absmax(stacked_leaves, layer_shardings, mesh, dtype, fp8_gather8: bool) -> tuple[jax.Array, ...]:
  """Computes hoisted whole-stack per-channel absmax (replicated across FSDP, 1 AG outside scan) for gather8 leaves."""
  out = []
  for p, s in zip(stacked_leaves, layer_shardings):
    if not _is_routed_moe_gather8_leaf(p, s, mesh, fp8_gather8):
      continue
    src_stacked_spec = jax.sharding.PartitionSpec(None, *tuple(s.spec))
    p_sharded = jax.lax.with_sharding_constraint(p, NamedSharding(mesh, src_stacked_spec))
    w_cast = jax.lax.stop_gradient(p_sharded.astype(dtype))
    # p has shape [length, E, D_in, D_out]; channel_axis=2 in 3D weight corresponds to axis 3 in stacked 4D weight,
    # so we reduce over axis 2 (D_in, which is unsharded: zero all-reduce!).
    absmax = jnp.max(jnp.abs(w_cast), axis=2, keepdims=True)
    gathered_s = sharding.remove_fsdp_from_sharding(s)
    gathered_stacked_spec = jax.sharding.PartitionSpec(None, *tuple(gathered_s.spec))
    absmax_g = jax.lax.with_sharding_constraint(absmax, NamedSharding(mesh, gathered_stacked_spec))
    out.append(absmax_g)
  return tuple(out)


_PrefetchCtx = collections.namedtuple(
    "_PrefetchCtx",
    [
        "mesh",
        "shard_mode",
        "debug_sharding",
        "dtype",
        "scan_axis",
        "static_args_leaves",
        "static_kwargs_leaves",
        "params_treedef",
        "state_treedef",
        "args_treedef",
        "kwargs_treedef",
        "remat_policy",
        "prevent_cse",
        "defer_grad_reduce_scatter",
        "fp8_gather8",
        "pin_sparse_core_id",
        "pin_moe_only",
        "use_scheduling_group",
        "offload_layer_input",
        "layer_shardings",
        "defer_host_offload",
        "batch_split",
        "config",
    ],
)

_PASSTHROUGH_PRIMITIVES = frozenset(
    {"name", "copy", "copy_p", "convert_element_type", "sharding_constraint", "reshard", "mesh_cast", "stop_gradient"}
)


class _ResidualLayout(collections.namedtuple("_ResidualLayout", ["treedef", "src", "saved_avals", "offload_mask"])):
  """Where each residual leaf of `jax.vjp(remat(layer), y, w)` comes from."""


def _keep_remat_on_device(remat_policy):
  """Converts any Offloadable decision in `remat_policy` to Saveable so inner while-loops stay in Device memory."""
  if remat_policy is None:
    return None

  def policy(prim, *args, **params):
    decision = remat_policy(prim, *args, **params)
    if isinstance(decision, ad_checkpoint.Offloadable):
      return ad_checkpoint.Saveable
    return decision

  return policy


def _is_literal(v):
  return hasattr(v, "val")


def _is_passthrough(eqn):
  if len(eqn.invars) != 1 or len(eqn.outvars) != 1:
    return False
  name = eqn.primitive.name
  if name in _PASSTHROUGH_PRIMITIVES:
    return True
  if name == "reduce_precision":
    dtype = eqn.invars[0].aval.dtype
    if not jnp.issubdtype(dtype, jnp.floating):
      return False
    finfo = jnp.finfo(dtype)
    return eqn.params.get("exponent_bits") == finfo.nexp and eqn.params.get("mantissa_bits") == finfo.nmant
  return False


def _vjp_residual_layout(
    f, y, w, layer_state, consts, split_w_fn, make_anchors_fn, offload_mask_override=None
):
  """Traces `jax.vjp(f, y, w)` abstractly and classifies its residual leaves."""
  y_leaves, y_treedef = jax.tree_util.tree_flatten(y)
  dense_leaves, fp8_items = split_w_fn(w)
  anc_leaves = list(make_anchors_fn())
  fp8_leaves, fp8_treedef = jax.tree_util.tree_flatten(fp8_items)
  s_leaves, s_treedef = jax.tree_util.tree_flatten(layer_state)
  c_leaves, c_treedef = jax.tree_util.tree_flatten(consts)
  treedef_box = []

  def residual_leaves(y_l, wd_l, anc_l, wf_l, s_l, c_l):
    s_ = jax.tree_util.tree_unflatten(s_treedef, s_l)
    c_ = jax.tree_util.tree_unflatten(c_treedef, c_l)
    fp8_ = jax.tree_util.tree_unflatten(fp8_treedef, wf_l)
    _, vjp_fn = jax.vjp(
        lambda yl, wdl, ancl: f(jax.tree_util.tree_unflatten(y_treedef, yl), wdl, tuple(ancl), fp8_, s_, c_),
        y_l,
        wd_l,
        anc_l,
    )
    leaves, treedef = jax.tree_util.tree_flatten(vjp_fn)
    treedef_box.append(treedef)
    return leaves

  closed = jax.make_jaxpr(residual_leaves)(y_leaves, dense_leaves, anc_leaves, fp8_leaves, s_leaves, c_leaves)
  jaxpr = closed.jaxpr
  origin = {}
  n_y = len(y_leaves)
  n_wd = len(dense_leaves)
  n_anc = len(anc_leaves)
  n_wf = len(fp8_leaves)
  n_s = len(s_leaves)
  for i, v in enumerate(jaxpr.invars):
    if i < n_y:
      origin[v] = ("y", i)
    elif i < n_y + n_wd:
      origin[v] = ("wd", i - n_y)
    elif i < n_y + n_wd + n_anc:
      origin[v] = ("anc", i - n_y - n_wd)
    elif i < n_y + n_wd + n_anc + n_wf:
      origin[v] = ("wf", i - n_y - n_wd - n_anc)
    elif i < n_y + n_wd + n_anc + n_wf + n_s:
      origin[v] = ("s", i - n_y - n_wd - n_anc - n_wf)
    else:
      origin[v] = ("c", i - n_y - n_wd - n_anc - n_wf - n_s)
  for eqn in jaxpr.eqns:
    if _is_passthrough(eqn):
      src_var, out_var = eqn.invars[0], eqn.outvars[0]
      if (
          not _is_literal(src_var)
          and src_var in origin
          and src_var.aval.shape == out_var.aval.shape
          and src_var.aval.dtype == out_var.aval.dtype
      ):
        origin[out_var] = origin[src_var]

  src, saved_avals = [], []
  for v in jaxpr.outvars:
    s = None if _is_literal(v) else origin.get(v)
    src.append(s)
    if s is None:
      saved_avals.append(v.aval)
  if offload_mask_override is not None:
    offload_mask = tuple(offload_mask_override)
  else:
    offload_mask = (False,) * len(saved_avals)
  return _ResidualLayout(treedef_box[0], tuple(src), tuple(saved_avals), offload_mask)


def _split_residuals(vjp_fn, layout):
  leaves = jax.tree_util.tree_leaves(vjp_fn)
  assert len(leaves) == len(layout.src), (len(leaves), len(layout.src))
  return tuple(leaf for leaf, s in zip(leaves, layout.src) if s is None)


def _rebuild_vjp(layout, saved, y, w, layer_state, consts, split_w_fn, make_anchors_fn):
  dense_leaves, fp8_items = split_w_fn(w)
  groups = {
      "y": jax.tree_util.tree_leaves(y),
      "wd": dense_leaves,
      "anc": list(make_anchors_fn()),
      "wf": jax.tree_util.tree_leaves(fp8_items),
      "s": jax.tree_util.tree_leaves(layer_state),
      "c": jax.tree_util.tree_leaves(consts),
  }
  saved_iter = iter(saved)
  leaves = [next(saved_iter) if s is None else groups[s[0]][s[1]] for s in layout.src]
  return jax.tree_util.tree_unflatten(layout.treedef, leaves)


def _log_saved_residuals(layout, w):
  saved_bytes = sum(a.size * a.dtype.itemsize for a in layout.saved_avals)
  host_cnt = sum(1 for a in layout.saved_avals if getattr(a, "memory_space", None) == jax.memory.Space.Host)
  host_bytes = sum(
      a.size * a.dtype.itemsize for a in layout.saved_avals if getattr(a, "memory_space", None) == jax.memory.Space.Host
  )
  max_logging.warning(
      f"[FSDP_PREFETCH] remat policy saves {len(layout.saved_avals)} residual(s) per sub-layer"
      f" ({saved_bytes / 2**20:.2f} MiB global, {host_cnt} host-offloaded = {host_bytes / 2**20:.2f} MiB global)."
  )
  w_shapes = {tuple(x.shape) for x in jax.tree_util.tree_leaves(w)}
  weight_like = [tuple(a.shape) for a in layout.saved_avals if tuple(a.shape) in w_shapes]
  if weight_like:
    max_logging.warning(
        f"[FSDP_PREFETCH] WARNING: remat policy saves residuals shaped like gathered weights {weight_like}!"
    )


_SG_COUNTER = [100]


def _next_sg_id() -> int:
  _SG_COUNTER[0] += 1
  return _SG_COUNTER[0]


def _prefetch_setup(graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx):
  """Rebuilds params/state and returns (params, state, gather_fn, rs_fn, layout_fn, fwd_layer, layer_vjp_fn)."""
  if ctx.scan_axis != 0:
    params_leaves = [
        jnp.moveaxis(p, ctx.scan_axis, 0) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_leaves
    ]
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, params_leaves)
  state = jax.tree_util.tree_unflatten(ctx.state_treedef, state_leaves)
  if ctx.layer_shardings is not None:
    layer_shardings = list(ctx.layer_shardings)
  else:
    layer_shardings = [_per_layer_sharding(p, ctx.mesh) for p in params_leaves]
  layer_shapes = [tuple(p.shape[1:]) for p in params_leaves]
  layer_dtypes = [p.dtype for p in params_leaves]
  consts = (tuple(diff_args_leaves), tuple(diff_kwargs_leaves))

  gather8_flags = [
      _is_routed_moe_gather8_leaf(p, s, ctx.mesh, ctx.fp8_gather8) for p, s in zip(params_leaves, layer_shardings)
  ]
  num_fsdp_gathers = sum(
      1
      for s in layer_shardings
      if s is not None and len(s.spec) > 0 and sharding.remove_fsdp_from_sharding(s).spec != s.spec
  )
  max_logging.warning(
      f"[FSDP_PREFETCH] setup: {num_fsdp_gathers}/{len(layer_shardings)} FSDP-gathered leaves,"
      f" gather8={sum(gather8_flags)}/{len(gather8_flags)},"
      f" pin_sparse_core_id={ctx.pin_sparse_core_id}, defer_rs={ctx.defer_grad_reduce_scatter}."
  )

  def gather_fn(
      layer_params,
      g8_absmax: tuple[jax.Array, ...] | None = None,
      sg_id: int | None = None,
      pin_sc_override: int | None = None,
  ):
    """All-gathers one layer's FSDP-sharded params (with hoisted gather8 MoE FP8 codes gather and ZERO internal barrier)."""
    leaves, treedef = jax.tree_util.tree_flatten(layer_params)
    active_sg = (_next_sg_id() if sg_id is None else sg_id) if ctx.use_scheduling_group else None
    eff_pin = ctx.pin_sparse_core_id if pin_sc_override is None else pin_sc_override
    it_g8 = iter(g8_absmax) if g8_absmax is not None else None
    out = []
    with _maybe_scheduling_group(active_sg):
      for x, s, g8 in zip(leaves, layer_shardings, gather8_flags):
        if g8:
          gathered_s = sharding.remove_fsdp_from_sharding(s)
          x_sharded = jax.lax.with_sharding_constraint(x, s)
          if it_g8 is not None:
            absmax_g = jax.lax.with_sharding_constraint(next(it_g8), gathered_s)
            absmax_s = jax.lax.with_sharding_constraint(absmax_g, s)
            codes_s, _ = quantizations.quantize_weight_codes_per_channel(
                jax.lax.stop_gradient(x_sharded.astype(ctx.dtype)), absmax=absmax_s
            )
            codes_s = jax.lax.with_sharding_constraint(codes_s, s)
            codes_g = jax.lax.stop_gradient(
                _apply_sharding_hint(
                    codes_s, s, ctx.mesh, ctx.shard_mode, ctx.debug_sharding, pin_sc_id=eff_pin
                )
            )
            fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
            scale_g = jax.lax.stop_gradient(
                jax.lax.with_sharding_constraint(
                    jnp.maximum(absmax_g / fp8_max, 1e-12).astype(jnp.float32), gathered_s
                )
            )
          else:
            codes_s, scale_s = quantizations.quantize_weight_codes_per_channel(
                jax.lax.stop_gradient(x_sharded.astype(ctx.dtype))
            )
            codes_s = jax.lax.with_sharding_constraint(codes_s, s)
            scale_s = jax.lax.with_sharding_constraint(scale_s, s)
            codes_g = jax.lax.stop_gradient(
                _apply_sharding_hint(
                    codes_s, s, ctx.mesh, ctx.shard_mode, ctx.debug_sharding, pin_sc_id=eff_pin
                )
            )
            scale_g = jax.lax.stop_gradient(
                _apply_sharding_hint(scale_s, s, ctx.mesh, ctx.shard_mode, ctx.debug_sharding, pin_sc_id=-1)
            )
          out.append(PrefetchedFp8Codes(codes=codes_g, scale=scale_g))
        else:
          dense_pin = -1 if ctx.pin_moe_only else eff_pin
          x_g = _apply_sharding_hint(
              x, s, ctx.mesh, ctx.shard_mode, ctx.debug_sharding, pin_sc_id=dense_pin
          )
          out.append(x_g)
    # CRITICAL: Do NOT call optimization_barrier(res) here! An internal barrier forces XLA to wait for
    # all-gather-done BEFORE starting fwd_sub(j) / layer_vjp(j), serializing the collective!
    return jax.tree_util.tree_unflatten(treedef, out)

  def split_w_for_ad(w):
    raw_leaves = jax.tree_util.tree_leaves(w, is_leaf=lambda v: isinstance(v, PrefetchedFp8Codes))
    dense = tuple(v for v, g8 in zip(raw_leaves, gather8_flags) if not g8)
    fp8 = tuple(v for v, g8 in zip(raw_leaves, gather8_flags) if g8)
    return dense, fp8

  def _make_fp8_grad_transform(idx: int):
    s = layer_shardings[idx]
    gathered_s = sharding.remove_fsdp_from_sharding(s)

    if ctx.defer_grad_reduce_scatter and ctx.shard_mode != ShardMode.EXPLICIT:
      # w_ste entered sparse_matmul shard_map with P(('expert', 'fsdp'), None, None), so g exits with 0 collectives!
      def to_deferred_partial(g):
        return jax.lax.with_sharding_constraint(g, s)

      return to_deferred_partial

    # Stage 1: shard_map transpose already emitted psum('fsdp'); constraining to `s` in rs_one fuses into 1 RS!
    def immediate_rs(g):
      return jax.lax.with_sharding_constraint(g, gathered_s)

    return immediate_rs

  fp8_grad_transforms = {idx: _make_fp8_grad_transform(idx) for idx, g8 in enumerate(gather8_flags) if g8}

  def make_fp8_anchors():
    anchors = []
    for idx, g8 in enumerate(gather8_flags):
      if not g8:
        continue
      s = layer_shardings[idx]
      shp = list(layer_shapes[idx])
      if ctx.defer_grad_reduce_scatter and ctx.shard_mode != ShardMode.EXPLICIT:
        scatter = _fsdp_scatter_dim(len(shp), ctx.mesh, s.spec)
        if scatter is not None:
          dim, ax = scatter
          for a in ax:
            shp[dim] *= ctx.mesh.shape[a]
      z = jax.lax.broadcast(jnp.zeros((), dtype=layer_dtypes[idx]), tuple(shp))
      z = jax.lax.with_sharding_constraint(z, s)
      anchors.append(z)
    return tuple(anchors)

  def assemble_w_for_ad(dense_leaves, anchors, fp8_items):
    it_d = iter(dense_leaves)
    it_a = iter(anchors)
    it_f = iter(fp8_items)
    merged = []
    defer_auto = bool(ctx.defer_grad_reduce_scatter and ctx.shard_mode != ShardMode.EXPLICIT)
    for idx, g8 in enumerate(gather8_flags):
      if not g8:
        merged.append(next(it_d))
      else:
        anc = next(it_a)
        fp8 = next(it_f)
        merged.append(
            quantizations.dequantize_weight_codes_ste(
                anc,
                fp8.codes,
                fp8.scale,
                grad_transform=fp8_grad_transforms[idx],
                w_ste_sharding=layer_shardings[idx] if defer_auto else None,
            )
        )
    return jax.tree_util.tree_unflatten(ctx.params_treedef, merged)

  def merge_dw_from_ad(d_dense_leaves, d_anchors):
    it_d = iter(d_dense_leaves)
    it_a = iter(d_anchors)
    merged = [next(it_a) if g8 else next(it_d) for g8 in gather8_flags]
    return jax.tree_util.tree_unflatten(ctx.params_treedef, merged)

  def layer_pure(y, dense_leaves, anchors, fp8_items, layer_state, layer_consts):
    """One layer call with every array input explicit (assembling STE weights inside remat so wf stays an invar)."""
    layer_weights = assemble_w_for_ad(dense_leaves, anchors, fp8_items)
    d_args, d_kwargs = layer_consts
    args_leaves = [s if s is not None else d for d, s in zip(d_args, ctx.static_args_leaves)]
    args_tuple = jax.tree_util.tree_unflatten(ctx.args_treedef, args_leaves)
    kwargs_leaves = [s if s is not None else d for d, s in zip(d_kwargs, ctx.static_kwargs_leaves)]
    kwargs = dict(jax.tree_util.tree_unflatten(ctx.kwargs_treedef, kwargs_leaves))
    layer_weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((layer_weights, layer_state), "layers")
    layer = nnx.merge(graphdef, layer_weights, layer_state)
    out = layer.call_batchsplit(y, *args_tuple, **kwargs) if ctx.batch_split else layer(y, *args_tuple, **kwargs)
    return out[0] if isinstance(out, tuple) else out

  def layer_fn(y, dense_leaves, anchors, fp8_items, layer_state):
    return layer_pure(y, dense_leaves, anchors, fp8_items, layer_state, consts)

  def rs_one(g, s, shp, tag):
    """Reduce-scatters one gathered-weight gradient leaf back to its FSDP parameter sharding `s`."""
    if s is None or len(s.spec) == 0 or not any(s.spec):
      return g
    if tag:
      out = _explicit_fsdp_reduce_scatter(
          g, ctx.mesh, s.spec, sharding.remove_fsdp_from_sharding(s).spec, target_shape=shp
      )
      if out is not None:
        return out
    scatter = _fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
    if scatter is not None and (
        _is_unreduced_over(g, scatter[1]) or _is_auto_deferred_partial_sum(g, shp, ctx.mesh, s.spec)
    ):
      out = _explicit_fsdp_reduce_scatter(
          g, ctx.mesh, s.spec, sharding.remove_fsdp_from_sharding(s).spec, target_shape=shp
      )
      if out is not None:
        return out
    return sharding.maybe_shard_with_name(g, s, ctx.shard_mode, debug_sharding=ctx.debug_sharding)

  tag_cache = {}

  def tag_flags(leaves):
    if "flags" not in tag_cache:
      flags = []
      for g, s, shp in zip(leaves, layer_shardings, layer_shapes):
        scatter = None if (s is None or len(s.spec) == 0 or not any(s.spec)) else _fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
        eligible = scatter is not None and (
            _is_unreduced_over(g, scatter[1]) or _is_auto_deferred_partial_sum(g, shp, ctx.mesh, s.spec)
        )
        flags.append(bool(ctx.defer_grad_reduce_scatter and eligible))
      tag_cache["flags"] = flags
    return tag_cache["flags"]

  def rs_fn(layer_grads, which="all", sg_id: int | None = None):
    leaves, treedef = jax.tree_util.tree_flatten(layer_grads)
    flags = tag_flags(leaves)
    active_sg = (
        (_next_sg_id() if sg_id is None else sg_id) if (ctx.use_scheduling_group and which == "tagged") else None
    )
    with _maybe_scheduling_group(active_sg):
      out = [
          rs_one(g, s, shp, tag) if which == "all" or (which == "tagged") == tag else g
          for g, s, shp, tag in zip(leaves, layer_shardings, layer_shapes, flags)
      ]
    # CRITICAL: No internal optimization_barrier(res) here so reduce-scatter-start overlaps subsequent backward compute!
    return jax.tree_util.tree_unflatten(treedef, out)

  rs_fn.tag_flags = tag_flags

  remat_layer = None
  orig_remat_layer = None
  if ctx.remat_policy is not None:
    if ctx.defer_host_offload:
      remat_layer = jax.checkpoint(layer_pure, policy=_keep_remat_on_device(ctx.remat_policy), prevent_cse=ctx.prevent_cse)
      orig_remat_layer = jax.checkpoint(layer_pure, policy=ctx.remat_policy, prevent_cse=ctx.prevent_cse)
    else:
      remat_layer = jax.checkpoint(layer_pure, policy=ctx.remat_policy, prevent_cse=ctx.prevent_cse)

  def layout_fn(y, w, layer_state):
    if remat_layer is None:
      return None
    offload_mask_override = None
    if orig_remat_layer is not None:
      orig_layout = _vjp_residual_layout(
          orig_remat_layer, y, w, layer_state, consts, split_w_for_ad, make_fp8_anchors
      )
      offload_mask_override = tuple(a.memory_space == jax.memory.Space.Host for a in orig_layout.saved_avals)
    layout = _vjp_residual_layout(
        remat_layer,
        y,
        w,
        layer_state,
        consts,
        split_w_for_ad,
        make_fp8_anchors,
        offload_mask_override=offload_mask_override,
    )
    _log_saved_residuals(layout, w)
    return layout

  def fwd_layer(y, w, layer_state, layout):
    dense_l, fp8_items = split_w_for_ad(w)
    anchors = make_fp8_anchors()
    if remat_layer is None:
      return layer_fn(y, dense_l, anchors, fp8_items, layer_state), ()
    y_leaves, y_treedef = jax.tree_util.tree_flatten(y)
    out, vjp_fn = jax.vjp(
        lambda yl, wdl, ancl: remat_layer(
            jax.tree_util.tree_unflatten(y_treedef, yl), wdl, ancl, fp8_items, layer_state, consts
        ),
        y_leaves,
        dense_l,
        anchors,
    )
    return out, _split_residuals(vjp_fn, layout)

  def layer_vjp_fn(y_in, w, layer_state, g_y, saved, layout):
    y_leaves, y_treedef = jax.tree_util.tree_flatten(y_in)
    if remat_layer is None:
      dense_l, fp8_items = split_w_for_ad(w)
      anchors = make_fp8_anchors()
      _, vjp_fn = jax.vjp(
          lambda yl, wdl, ancl: layer_fn(
              jax.tree_util.tree_unflatten(y_treedef, yl), wdl, ancl, fp8_items, layer_state
          ),
          y_leaves,
          dense_l,
          anchors,
      )
    else:
      vjp_fn = _rebuild_vjp(layout, saved, y_in, w, layer_state, consts, split_w_for_ad, make_fp8_anchors)
    dy_leaves, d_dense, d_anchors = vjp_fn(g_y)
    dy = jax.tree_util.tree_unflatten(y_treedef, dy_leaves)
    return dy, merge_dw_from_ad(d_dense, d_anchors)

  return params, state, gather_fn, rs_fn, layout_fn, fwd_layer, layer_vjp_fn


def _slice_at(tree, idx):
  return jax.tree.map(lambda x: x[idx], tree)


def _slice_range(tree, start, end):
  return jax.tree.map(lambda x: x[start:end], tree)


def _expand0(tree):
  return jax.tree.map(lambda x: jnp.expand_dims(x, 0), tree)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7))
def _custom_vjp_prefetch_pipeline(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  out, _ = _custom_vjp_prefetch_pipeline_fwd(
      graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
  )
  return out


def _maybe_offload_to_host(tree, enabled: bool, shard_mode=ShardMode.AUTO):
  if not enabled:
    return tree
  if shard_mode == ShardMode.EXPLICIT:
    return jax.tree.map(lambda t: jax.device_put(t, jax.typeof(t).sharding.with_memory_kind("pinned_host")), tree)
  return jax.tree.map(lambda t: jax.device_put(t, jax.memory.Space.Host), tree)


def _maybe_onload_to_device(tree, enabled: bool, shard_mode=ShardMode.AUTO):
  if not enabled:
    return tree
  if shard_mode == ShardMode.EXPLICIT:
    return jax.tree.map(lambda t: jax.device_put(t, jax.typeof(t).sharding.with_memory_kind("device")), tree)
  return jax.tree.map(lambda t: jax.device_put(t, jax.memory.Space.Device), tree)


def _custom_vjp_prefetch_pipeline_fwd(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  params, state, gather_fn, _, layout_fn, fwd_layer, _ = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )
  w_0 = gather_fn(_slice_at(params, 0))
  layout = layout_fn(x_in, w_0, _slice_at(state, 0))

  if length == 1:
    y_out, saved_0 = fwd_layer(x_in, w_0, _slice_at(state, 0), layout)
    residuals = (
        _expand0(_maybe_offload_to_host(x_in, ctx.offload_layer_input, ctx.shard_mode)),
        _expand0(saved_0),
        params_leaves,
        state_leaves,
        diff_args_leaves,
        diff_kwargs_leaves,
    )
    return (y_out, state_leaves), residuals

  def body(carry, xs):
    y_k, w_k = carry
    p_next, s_k = xs
    y_k, p_next = jax.lax.optimization_barrier((y_k, p_next))
    w_next = gather_fn(p_next)
    y_next, saved_k = fwd_layer(y_k, w_k, s_k, layout)
    y_next, w_next = jax.lax.optimization_barrier((y_next, w_next))
    return (y_next, w_next), (_maybe_offload_to_host(y_k, ctx.offload_layer_input, ctx.shard_mode), saved_k)

  xs = (_slice_range(params, 1, length), _slice_range(state, 0, length - 1))
  (y_last_in, w_last), (ys_prefix, saved_prefix) = jax.lax.scan(body, (x_in, w_0), xs, unroll=2)
  y_out, saved_last = fwd_layer(y_last_in, w_last, _slice_at(state, length - 1), layout)
  all_y_inputs = jax.tree.map(
      lambda ys, yl: jnp.concatenate([ys, jnp.expand_dims(yl, 0)], axis=0),
      ys_prefix,
      _maybe_offload_to_host(y_last_in, ctx.offload_layer_input, ctx.shard_mode),
  )
  all_saved = jax.tree.map(lambda ss, sl: jnp.concatenate([ss, jnp.expand_dims(sl, 0)], axis=0), saved_prefix, saved_last)
  residuals = (all_y_inputs, all_saved, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves)
  return (y_out, state_leaves), residuals


def _custom_vjp_prefetch_pipeline_bwd(graphdef, length, ctx, res, g_out):
  all_y_inputs, all_saved, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves = res
  g, _ = g_out
  params, state, gather_fn, rs_fn, layout_fn, _, layer_vjp_fn = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )
  g, params = jax.lax.optimization_barrier((g, params))
  w_last = gather_fn(_slice_at(params, length - 1), pin_sc_override=-1)
  y_last_in = _maybe_onload_to_device(_slice_at(all_y_inputs, length - 1), ctx.offload_layer_input, ctx.shard_mode)
  layout = layout_fn(y_last_in, w_last, _slice_at(state, length - 1))

  if length == 1:
    g, dw_0 = layer_vjp_fn(y_last_in, w_last, _slice_at(state, 0), g, _slice_at(all_saved, 0), layout)
    d_params = _expand0(rs_fn(dw_0))
  else:

    def body(carry, xs):
      g_k, w_k = carry
      y_k, saved_k, p_prev, s_k = xs
      g_k, y_k, saved_k, p_prev = jax.lax.optimization_barrier((g_k, y_k, saved_k, p_prev))
      w_prev = gather_fn(p_prev, pin_sc_override=-1)
      y_k_dev = _maybe_onload_to_device(y_k, ctx.offload_layer_input, ctx.shard_mode)
      g_prev, dw_k = layer_vjp_fn(y_k_dev, w_k, s_k, g_k, saved_k, layout)
      dw_k = rs_fn(dw_k)
      g_prev, w_prev = jax.lax.optimization_barrier((g_prev, w_prev))
      return (g_prev, w_prev), dw_k

    xs = (
        _slice_range(all_y_inputs, 1, length),
        _slice_range(all_saved, 1, length),
        _slice_range(params, 0, length - 1),
        _slice_range(state, 1, length),
    )
    (g, w_0), d_params_suffix = jax.lax.scan(body, (g, w_last), xs, reverse=True, unroll=2)
    y_0_dev = _maybe_onload_to_device(_slice_at(all_y_inputs, 0), ctx.offload_layer_input, ctx.shard_mode)
    g, dw_0 = layer_vjp_fn(y_0_dev, w_0, _slice_at(state, 0), g, _slice_at(all_saved, 0), layout)
    dw_0 = rs_fn(dw_0)
    d_params = jax.tree.map(lambda d0, ds: jnp.concatenate([jnp.expand_dims(d0, 0), ds], axis=0), dw_0, d_params_suffix)

  params_bar_leaves = jax.tree_util.tree_leaves(d_params)
  if ctx.scan_axis != 0:
    params_bar_leaves = [
        jnp.moveaxis(p, 0, ctx.scan_axis) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_bar_leaves
    ]
  params_bar_leaves = tuple(params_bar_leaves)
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in diff_args_leaves)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in diff_kwargs_leaves)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_prefetch_pipeline.defvjp(_custom_vjp_prefetch_pipeline_fwd, _custom_vjp_prefetch_pipeline_bwd)

_SAVE_SUBLAYER_INPUTS = jax.checkpoint_policies.save_only_these_names("decoder_layer_input")


def _also_save_sublayer_inputs(remat_policy):
  if remat_policy is None:
    return _SAVE_SUBLAYER_INPUTS

  def policy(prim, *args, **params):
    decision = remat_policy(prim, *args, **params)
    if decision is False or decision is ad_checkpoint.Recompute:
      return _SAVE_SUBLAYER_INPUTS(prim, *args, **params)
    return decision

  return policy


def mesh_uses_fsdp(mesh) -> bool:
  return any(mesh.shape.get(axis, 1) > 1 for axis in sharding.FSDP_MESH_AXES)


def _layer_major(params, state, scan_axis, length, move_scan_axis=True):
  if move_scan_axis and scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, params)
  params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
  state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)
  return params, state


def _prepare_prefetch_call(
    layers, args, kwargs, *, length, mesh, config, remat_policy, prevent_cse, defer_host_offload: bool = False
):
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  layer_shardings = _resolve_layer_shardings_from_params(params, mesh, config)
  scan_axis = config.param_scan_axis
  params, state = _layer_major(params, state, scan_axis, length, move_scan_axis=False)

  sig = inspect.signature(layers.__class__.__call__)
  valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

  def _is_diff_leaf(x):
    return isinstance(x, jax.Array) or (hasattr(jax.core, "Tracer") and isinstance(x, jax.core.Tracer))

  args_leaves, args_treedef = jax.tree_util.tree_flatten(tuple(args))
  kwargs_leaves, kwargs_treedef = jax.tree_util.tree_flatten(tuple(valid_kwargs.items()))

  dummy_arr = jnp.zeros((0,), dtype=jnp.float32)

  diff_args_leaves = tuple(x if _is_diff_leaf(x) else dummy_arr for x in args_leaves)
  static_args_leaves = tuple(None if _is_diff_leaf(x) else x for x in args_leaves)

  diff_kwargs_leaves = tuple(x if _is_diff_leaf(x) else dummy_arr for x in kwargs_leaves)
  static_kwargs_leaves = tuple(None if _is_diff_leaf(x) else x for x in kwargs_leaves)

  params_leaves, params_treedef = jax.tree_util.tree_flatten(params)
  state_leaves, state_treedef = jax.tree_util.tree_flatten(state)

  fp8_gather8 = bool(
      getattr(config, "fp8_moe_fake_quant", False) and getattr(config, "fp8_fake_quant_hoist", "off") == "gather8"
  )
  dli = str(getattr(config, "decoder_layer_input", ""))
  offload_layer_input = getattr(config, "remat_policy", "") == "custom" and ("offload" in dli.lower())

  ctx = _PrefetchCtx(
      mesh=mesh,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      dtype=config.dtype,
      scan_axis=scan_axis,
      static_args_leaves=static_args_leaves,
      static_kwargs_leaves=static_kwargs_leaves,
      params_treedef=params_treedef,
      state_treedef=state_treedef,
      args_treedef=args_treedef,
      kwargs_treedef=kwargs_treedef,
      remat_policy=remat_policy,
      prevent_cse=prevent_cse,
      defer_grad_reduce_scatter=bool(getattr(config, "prefetch_fsdp_defer_grad_reduce_scatter", False)),
      fp8_gather8=fp8_gather8,
      pin_sparse_core_id=int(getattr(config, "prefetch_fsdp_pin_sparse_core_id", -1)),
      pin_moe_only=bool(getattr(config, "prefetch_fsdp_pin_moe_only", False)),
      use_scheduling_group=bool(getattr(config, "prefetch_fsdp_use_scheduling_group", False)),
      offload_layer_input=offload_layer_input,
      layer_shardings=layer_shardings,
      defer_host_offload=defer_host_offload,
      batch_split=bool(getattr(config, "prefetch_fsdp_batch_split", False)),
      config=config,
  )
  return graphdef, tuple(params_leaves), tuple(state_leaves), diff_args_leaves, diff_kwargs_leaves, ctx


def apply_layers_with_fsdp_prefetch(
    layers,
    x_in,
    *args,
    length: int,
    mesh,
    config,
    remat_policy=None,
    prevent_cse: bool = False,
    skip_block_remat: bool = False,
    **kwargs,
):
  if length == 0:
    return x_in, layers, None
  if skip_block_remat:
    remat_policy = _also_save_sublayer_inputs(remat_policy)
  graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx = _prepare_prefetch_call(
      layers, args, kwargs, length=length, mesh=mesh, config=config, remat_policy=remat_policy, prevent_cse=prevent_cse
  )
  final_y, _ = _custom_vjp_prefetch_pipeline(
      graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
  )
  return final_y, layers, None
