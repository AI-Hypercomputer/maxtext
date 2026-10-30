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

"""FSDP weight prefetching for a scanned (layer-stacked) decoder layer stack.

Enabled with `prefetch_fsdp_weights=True` (requires `scan_layers=True`). Instead of letting XLA
all-gather each layer's FSDP-sharded weights right before the layer runs, the layer loop is written
by hand so that the all-gather (AG) of layer k+1's weights overlaps the compute of layer k in the
forward pass, and the AG of layer k-1's weights overlaps the recompute + vjp of layer k in the
backward pass, followed by a reduce-scatter (RS) of layer k's weight gradients.

The all-gathers are explicit `all_gather`s tagged with the `xla_explicit_fsdp` XLA metadata. On TPUs
with SparseCores, run with
  --xla_tpu_enable_fsdp_latency_hiding_scheduler=true
  --xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0
so that XLA's FSDP scheduler runs the tagged gathers on a dedicated SparseCore.

See `_custom_vjp_prefetch_pipeline` for the design.
"""

import collections
import functools
import inspect

from flax import nnx
import jax
from jax import ad_checkpoint
from jax.experimental import xla_metadata
from jax.experimental.layout import Layout, with_layout_constraint
import jax.numpy as jnp
from jax.sharding import NamedSharding
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_logging, maxtext_utils_nnx, sharding


def _apply_sharding_hint(w, mesh, shard_mode, debug_sharding=False):
  """All-gathers `w` over the FSDP mesh axes by resharding it to its FSDP-stripped sharding.

  Leaves without a NamedSharding are returned unchanged.
  """
  w_sharding = getattr(jax.typeof(w), "sharding", None) or getattr(w, "sharding", None)
  if not isinstance(w_sharding, NamedSharding) or w_sharding.spec is None:
    return w
  src_sharding = NamedSharding(mesh, w_sharding.spec)
  if shard_mode != ShardMode.EXPLICIT:
    # In auto mode, pin the source to its FSDP-sharded layout so the gather is not hoisted onto the
    # stored parameter (which would materialize the whole gathered stack).
    w = jax.lax.with_sharding_constraint(w, src_sharding)
  gathered_sharding = sharding.remove_fsdp_from_sharding(src_sharding)
  out = _explicit_fsdp_all_gather(
      w, mesh, src_sharding.spec, gathered_sharding.spec, reduced=shard_mode == ShardMode.EXPLICIT
  )
  if out is not None:
    return out
  return sharding.maybe_shard_with_name(w, gathered_sharding, shard_mode=shard_mode, debug_sharding=debug_sharding)


def _explicit_fsdp_all_gather(w, mesh, src_spec, gathered_spec, reduced=False):
  """All-gathers `w` over its FSDP mesh axes with an explicit all_gather tagged `xla_explicit_fsdp`.

  On TPUs that offload all-gathers to the SparseCore, the tag lets XLA's FSDP scheduler
  (--xla_tpu_enable_fsdp_latency_hiding_scheduler with --xla_tpu_explicit_fsdp_dedicated_sparse_core_id)
  run these gathers on a dedicated SparseCore. Otherwise they share the SparseCore queue with the
  layer's own SparseCore work (e.g. MoE token gathers) and must finish before that work can.

  With `reduced` (explicit sharding mode, one gathered dimension), the result is typed `reduced` over
  the FSDP axes. The gradient of a reduced value is `unreduced` (per-device partial sums), so the
  backward pass emits no all-reduce for it and `_explicit_fsdp_reduce_scatter` performs the only
  reduction, tagged so XLA's FSDP scheduler can overlap it like the prefetch all-gathers.

  Returns None (the caller falls back to resharding) if the FSDP axes of a dimension are not its
  minor-most mesh axes, since a tiled gather would then not produce the target layout.
  """
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

  if reduced and len(dims) == 1:
    (dim, ax) = dims[0]
    gathered = tuple(gathered_spec) + (None,) * (w.ndim - len(tuple(gathered_spec)))
    # Size-1 FSDP (co-)axes (e.g. `fsdp_transpose`, `context`), which `ax` skips but `gathered_spec`
    # strips, are first resharded away (a no-op on the data); otherwise the output would be typed as
    # varying over axes the out_specs replicate.
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
      return jax.shard_map(
          gather_reduced,
          mesh=mesh,
          in_specs=in_spec,
          out_specs=jax.sharding.PartitionSpec(*gathered, reduced=frozenset(ax)),
      )(w)

  def gather(x):
    for i, ax in dims:
      x = jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=i, tiled=True)
    return x

  with xla_metadata.set_xla_metadata(xla_explicit_fsdp="yes"):
    return jax.shard_map(
        gather, mesh=mesh, in_specs=jax.sharding.PartitionSpec(*spec), out_specs=gathered_spec, check_vma=False
    )(w)


def _fsdp_scatter_dim(ndim, mesh, src_spec):
  """Returns (dim, fsdp_axes) of the single dimension a gradient is reduce-scattered over, else None.

  None if no dimension or more than one is FSDP-sharded (chained FSDP collectives must be independent
  for XLA's FSDP scheduler), or if the FSDP axes of the dimension are not its minor-most mesh axes.
  """
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


def _pin_scatter_dim_major(g, dim):
  """Lays `g` out with dimension `dim` major-most before it is reduce-scattered over `dim`.

  XLA does not offload a reduce-scatter over a physically minor dimension to the SparseCore as one and
  instead rewrites it into an all-reduce plus slice. That happens for kernels sharded on their last
  dimension (e.g. a [F, D/fsdp] output projection) and for gradients of transposed matmuls (laid out
  {0, 1}) unless their layout is pinned.
  """
  return with_layout_constraint(g, Layout((dim,) + tuple(i for i in range(g.ndim) if i != dim)))


def _explicit_fsdp_reduce_scatter(g, mesh, src_spec, gathered_spec):
  """Reduce-scatters a gathered-weight gradient `g` over its FSDP mesh axes, tagged `xla_explicit_fsdp`.

  The inverse of the `reduced` `_explicit_fsdp_all_gather`: `g` is `unreduced` over the FSDP axes
  (each device holds a partial sum) with the gathered sharding `gathered_spec`, and the result is the
  summed gradient with the parameter's FSDP sharding `src_spec`. The tag lets XLA's FSDP scheduler
  run the reduce-scatter on the dedicated FSDP SparseCore (next to the prefetch all-gathers) and
  complete it with the next FSDP all-gather, so it overlaps the following layer's backward.

  Returns None (the caller falls back to resharding) if `g` is not unreduced over exactly the FSDP
  axes of a single dimension: reduce-scattering an already-reduced gradient would sum it again.
  """
  scatter = _fsdp_scatter_dim(g.ndim, mesh, src_spec)
  if scatter is None or not _is_unreduced_over(g, scatter[1]):
    return None
  dim, ax = scatter
  spec = tuple(src_spec) + (None,) * (g.ndim - len(tuple(src_spec)))
  gathered = tuple(gathered_spec) + (None,) * (g.ndim - len(tuple(gathered_spec)))
  g_spec = jax.typeof(g).sharding.spec
  g = _pin_scatter_dim_major(g, dim)

  def psum_scatter(x):
    return jax.lax.psum_scatter(x, ax if len(ax) > 1 else ax[0], scatter_dimension=dim, tiled=True)

  with xla_metadata.set_xla_metadata(xla_explicit_fsdp="yes"):
    return jax.shard_map(
        psum_scatter,
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec(*gathered, unreduced=frozenset(ax), unreduced_kind=g_spec.unreduced_kind),
        out_specs=jax.sharding.PartitionSpec(*spec),
    )(g)


def _per_layer_sharding(stacked, mesh):
  """Sharding of one layer slice `stacked[i]` of a scanned (layer-major) parameter, or None."""
  s = getattr(jax.typeof(stacked), "sharding", None) or getattr(stacked, "sharding", None)
  if not isinstance(s, NamedSharding) or s.spec is None:
    return None
  return NamedSharding(mesh, jax.sharding.PartitionSpec(*tuple(s.spec)[1:]))


def _needs_minor_dim_layout_pin(stacked, layer_sharding, mesh):
  """Whether a per-layer weight is 3D+ and FSDP-sharded on its minor-most dimension."""
  ndim = getattr(stacked, "ndim", 0) - 1
  if ndim < 3 or layer_sharding is None or layer_sharding.spec is None or len(layer_sharding.spec) < ndim:
    return False
  entry = layer_sharding.spec[ndim - 1]
  axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
  axes = tuple(a for a in axes if mesh.shape.get(a, 1) > 1)
  return bool(sharding.fsdp_axes_for_dim(axes))


_PrefetchCtx = collections.namedtuple(
    "_PrefetchCtx",
    [
        "mesh",
        "shard_mode",
        "debug_sharding",
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
    ],
)

# Primitives that pass their (single) operand through unchanged. A remat residual produced only by a
# chain of these from a layer input (e.g. `checkpoint_name(inputs, "decoder_layer_input")`) is that
# input, so it is resupplied in the backward pass instead of being saved a second time.
_PASSTHROUGH_PRIMITIVES = frozenset(
    {"name", "copy", "copy_p", "convert_element_type", "sharding_constraint", "reshard", "mesh_cast"}
)


class _ResidualLayout(collections.namedtuple("_ResidualLayout", ["treedef", "src", "saved_avals"])):
  """Where each residual leaf of `jax.vjp(remat(layer), y, w)` comes from.

  `src[i]` is `("y",)`, `("w", j)`, `("s", j)` or `("c", j)` when residual leaf `i` is the layer
  input, the j-th gathered weight leaf, the j-th layer-state leaf or the j-th layer-argument leaf
  (all of which the backward pass has at hand), and None when it is an activation saved by the
  remat policy. Only the None leaves are carried from the forward to the backward pass.
  """


def _is_literal(v):
  """Whether a jaxpr atom is a Literal (constant) rather than a Var.

  Only Literals carry a `val`. The Literal type itself is only public under `jax.extend.core`, which
  is not a dependency available to every MaxText build.
  """
  return hasattr(v, "val")


def _is_passthrough(eqn):
  """Whether a jaxpr equation returns its single operand unchanged (see `_PASSTHROUGH_PRIMITIVES`).

  `reduce_precision` to the operand's own precision is an identity that remat inserts around
  saved values, so it counts as a passthrough too.
  """
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


def _vjp_residual_layout(f, y, w, layer_state, consts):
  """Traces `jax.vjp(f, y, w)` abstractly and classifies its residual leaves (see `_ResidualLayout`)."""
  w_leaves = jax.tree_util.tree_leaves(w)
  s_leaves = jax.tree_util.tree_leaves(layer_state)
  c_leaves = jax.tree_util.tree_leaves(consts)
  treedef_box = []

  def residual_leaves(y, w_leaves, s_leaves, c_leaves):
    w_ = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(w), w_leaves)
    s_ = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(layer_state), s_leaves)
    c_ = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(consts), c_leaves)
    _, vjp_fn = jax.vjp(lambda y_, w2: f(y_, w2, s_, c_), y, w_)
    leaves, treedef = jax.tree_util.tree_flatten(vjp_fn)
    treedef_box.append(treedef)
    return leaves

  closed = jax.make_jaxpr(residual_leaves)(y, w_leaves, s_leaves, c_leaves)
  jaxpr = closed.jaxpr
  origin = {}
  for i, v in enumerate(jaxpr.invars):
    if i == 0:
      origin[v] = ("y",)
    elif i <= len(w_leaves):
      origin[v] = ("w", i - 1)
    elif i <= len(w_leaves) + len(s_leaves):
      origin[v] = ("s", i - 1 - len(w_leaves))
    else:
      origin[v] = ("c", i - 1 - len(w_leaves) - len(s_leaves))
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
  return _ResidualLayout(treedef_box[0], tuple(src), tuple(saved_avals))


def _split_residuals(vjp_fn, layout):
  """The residual leaves of `vjp_fn` that the remat policy saves (layout src None)."""
  leaves = jax.tree_util.tree_leaves(vjp_fn)
  assert len(leaves) == len(layout.src), (len(leaves), len(layout.src))
  return tuple(leaf for leaf, s in zip(leaves, layout.src) if s is None)


def _rebuild_vjp(layout, saved, y, w, layer_state, consts):
  """Reassembles the vjp function from the saved residuals and the inputs available in the backward."""
  groups = {
      "y": [y],
      "w": jax.tree_util.tree_leaves(w),
      "s": jax.tree_util.tree_leaves(layer_state),
      "c": jax.tree_util.tree_leaves(consts),
  }
  saved_iter = iter(saved)
  leaves = [next(saved_iter) if s is None else groups[s[0]][s[1] if len(s) > 1 else 0] for s in layout.src]
  return jax.tree_util.tree_unflatten(layout.treedef, leaves)


def _log_saved_residuals(layout, w):
  """Logs the per-layer bytes saved by the remat policy and flags residuals shaped like a weight."""
  saved_bytes = sum(a.size * a.dtype.itemsize for a in layout.saved_avals)
  max_logging.info(
      f"prefetch_fsdp_weights: remat policy saves {len(layout.saved_avals)} residual(s) per layer"
      f" ({saved_bytes / 2**20:.1f} MiB per layer, per device-global shape)."
  )
  w_shapes = {tuple(x.shape) for x in jax.tree_util.tree_leaves(w)}
  weight_like = [tuple(a.shape) for a in layout.saved_avals if tuple(a.shape) in w_shapes]
  if weight_like:
    max_logging.info(
        f"prefetch_fsdp_weights: WARNING: the remat policy saves residuals shaped like gathered weights {weight_like};"
        " these are kept for every layer between the forward and backward pass."
    )


def _prefetch_setup(graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx):
  """Rebuilds params/state and returns (params, state, gather_fn, rs_fn, layout_fn, fwd_layer, layer_vjp_fn).

  `layout_fn(y, w, layer_state)` returns the `_ResidualLayout` of one layer (None for full remat).
  `fwd_layer(y, w, layer_state, layout) -> (y_out, saved)` runs one layer in the forward pass and
  returns the activations its remat policy saves (empty for full remat). `layer_vjp_fn(y, w,
  layer_state, g, saved, layout) -> (dy, dw)` differentiates one layer in the backward pass from
  those activations and the gathered weights `w`, recomputing everything else.
  """
  if ctx.scan_axis != 0:
    params_leaves = [
        jnp.moveaxis(p, ctx.scan_axis, 0) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_leaves
    ]
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, params_leaves)
  state = jax.tree_util.tree_unflatten(ctx.state_treedef, state_leaves)
  layer_shardings = [_per_layer_sharding(p, ctx.mesh) for p in params_leaves]
  consts = (tuple(diff_args_leaves), tuple(diff_kwargs_leaves))

  pin_layout_flags = [_needs_minor_dim_layout_pin(p, s, ctx.mesh) for p, s in zip(params_leaves, layer_shardings)]

  def gather_fn(layer_params):
    """All-gathers one layer's FSDP-sharded params (bf16[E, D/fsdp, F] -> bf16[E, D, F])."""
    leaves, treedef = jax.tree_util.tree_flatten(layer_params)
    out = []
    for x, pin in zip(leaves, pin_layout_flags):
      x = _apply_sharding_hint(x, ctx.mesh, ctx.shard_mode, ctx.debug_sharding)
      if pin:
        # 3D+ weights gathered on their minor-most dimension (e.g. [E, F, D/fsdp] where D/fsdp is not a
        # multiple of 128) exit all-gather in layout {1, 0, 2}. Pinning only those weights to default
        # major-to-minor layout keeps their relayout on the gather output (offloaded to SparseCore)
        # without forcing unnecessary SparseCore data-format copies on weights gathered on major dims.
        x = with_layout_constraint(x, Layout(tuple(range(x.ndim))))
      out.append(x)
    return jax.tree_util.tree_unflatten(treedef, out)

  def layer_pure(y, layer_weights, layer_state, layer_consts):
    """One layer call with every array input explicit (the layer arguments come from `layer_consts`)."""
    d_args, d_kwargs = layer_consts
    args_leaves = [s if s is not None else d for d, s in zip(d_args, ctx.static_args_leaves)]
    args_tuple = jax.tree_util.tree_unflatten(ctx.args_treedef, args_leaves)
    kwargs_leaves = [s if s is not None else d for d, s in zip(d_kwargs, ctx.static_kwargs_leaves)]
    kwargs = dict(jax.tree_util.tree_unflatten(ctx.kwargs_treedef, kwargs_leaves))
    # Per-layer slices still carry the scan axis in their sharding metadata; drop it so the
    # merged layer sees the same metadata as in the regular scan path.
    layer_weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((layer_weights, layer_state), "layers")
    layer = nnx.merge(graphdef, layer_weights, layer_state)
    out = layer(y, *args_tuple, **kwargs)
    return out[0] if isinstance(out, tuple) else out

  def layer_fn(y, layer_weights, layer_state):
    return layer_pure(y, layer_weights, layer_state, consts)

  def rs_one(g, s, tag):
    """Reduce-scatters one gathered-weight gradient leaf back to its FSDP parameter sharding `s`."""
    if s is None:
      return g
    if tag:
      out = _explicit_fsdp_reduce_scatter(g, ctx.mesh, s.spec, sharding.remove_fsdp_from_sharding(s).spec)
      if out is not None:
        return out
    scatter = _fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
    if scatter is not None and _is_unreduced_over(g, scatter[1]):
      g = _pin_scatter_dim_major(g, scatter[0])
    return sharding.maybe_shard_with_name(g, s, ctx.shard_mode, debug_sharding=ctx.debug_sharding)

  tag_cache = {}

  def tag_flags(leaves):
    """Per-leaf choice of the tagged, deferred (dedicated SparseCore) reduce-scatter; the same for every layer."""
    if "flags" not in tag_cache:
      flags = []
      for g, s in zip(leaves, layer_shardings):
        scatter = None if s is None else _fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
        eligible = scatter is not None and _is_unreduced_over(g, scatter[1])
        flags.append(bool(ctx.defer_grad_reduce_scatter and eligible))
      tag_cache["flags"] = flags
    return tag_cache["flags"]

  def rs_fn(layer_grads, which="all"):
    """Reduce-scatters one layer's gathered-weight gradients back to the FSDP parameter sharding.

    `which` restricts it to the "tagged" or "untagged" reduce-scatters (the others are returned as is).
    """
    leaves, treedef = jax.tree_util.tree_flatten(layer_grads)
    flags = tag_flags(leaves)
    out = [
        rs_one(g, s, tag) if which == "all" or (which == "tagged") == tag else g
        for g, s, tag in zip(leaves, layer_shardings, flags)
    ]
    return jax.tree_util.tree_unflatten(treedef, out)

  rs_fn.tag_flags = tag_flags

  # Full remat (no policy): save only each layer's input and recompute the whole layer in the backward pass.
  remat_layer = None
  if ctx.remat_policy is not None:
    remat_layer = jax.checkpoint(layer_pure, policy=ctx.remat_policy, prevent_cse=ctx.prevent_cse)

  def layout_fn(y, w, layer_state):
    if remat_layer is None:
      return None
    layout = _vjp_residual_layout(remat_layer, y, w, layer_state, consts)
    _log_saved_residuals(layout, w)
    return layout

  def fwd_layer(y, w, layer_state, layout):
    if remat_layer is None:
      return layer_fn(y, w, layer_state), ()
    out, vjp_fn = jax.vjp(lambda y_, w_: remat_layer(y_, w_, layer_state, consts), y, w)
    return out, _split_residuals(vjp_fn, layout)

  def layer_vjp_fn(y_in, w, layer_state, g_y, saved, layout):
    if remat_layer is None:
      _, vjp_fn = jax.vjp(lambda y_, w_: layer_fn(y_, w_, layer_state), y_in, w)
    else:
      vjp_fn = _rebuild_vjp(layout, saved, y_in, w, layer_state, consts)
    return vjp_fn(g_y)  # (dy_in, dW gathered)

  return params, state, gather_fn, rs_fn, layout_fn, fwd_layer, layer_vjp_fn


def _slice_at(tree, idx):
  """Layer `idx` of a layer-stacked pytree."""
  return jax.tree.map(lambda x: x[idx], tree)


def _slice_range(tree, start, end):
  """Layers `start .. end-1` of a layer-stacked pytree."""
  return jax.tree.map(lambda x: x[start:end], tree)


def _expand0(tree):
  return jax.tree.map(lambda x: jnp.expand_dims(x, 0), tree)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7))
def _custom_vjp_prefetch_pipeline(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  """Runs `length` scanned layers with FSDP weight prefetching in both the forward and backward pass.

  Forward (layer k):   AG(k+1) || compute(k). Each layer's input activation is saved, plus the
                       activations its remat policy saves (none for full remat).
  Backward (layer k):  AG(k-1) || [recompute(k) + vjp(k)], then reduce-scatter dW(k).

  Gathered weights are never saved between forward and backward: with a non-full remat policy the
  forward pass saves the policy's residuals of `jax.vjp(jax.checkpoint(layer, policy))` minus the
  layer inputs, and the backward pass rebuilds the vjp from them and the re-gathered weights.

  Both passes unroll their scan loop by 2 (`unroll=2`), so each unrolled while-loop body runs two
  consecutive layers and alternates between two gathered-weight buffers. The forward holds at most
  two layers of gathered weights (current + prefetched), and the backward at most three (W(k),
  W(k-1) and dW(k) until its reduce-scatter).
  """
  out, _ = _custom_vjp_prefetch_pipeline_fwd(
      graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
  )
  return out


def _custom_vjp_prefetch_pipeline_fwd(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  """Forward pass: AG(k+1) is issued while layer k computes; the gathered weights ride in the carry."""
  params, state, gather_fn, _, layout_fn, fwd_layer, _ = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )
  w_0 = gather_fn(_slice_at(params, 0))
  layout = layout_fn(x_in, w_0, _slice_at(state, 0))
  w_curr = gather_fn(_slice_at(params, 1)) if length > 1 else None  # AG(1) || fwd(0).
  y, saved_0 = fwd_layer(x_in, w_0, _slice_at(state, 0), layout)
  ys_mid = None  # Inputs of layers 1 .. length-2 (residuals for the backward pass).
  saved_mid = None  # Remat-policy residuals of layers 1 .. length-2.
  if length > 2:
    # Only order within the unrolled loop when the unroll=2 while-loop has >= 2 trips; XLA unrolls
    # single-trip while-loops into the parent computation, where FSDP gathers must not depend on
    # earlier FSDP gathers without a while-loop barrier in between.
    ordered = (length - 2) // 2 >= 2

    def body(carry, xs):
      y_k, w_k = carry
      state_k, next_idx = xs
      params_next = _slice_at(params, next_idx)
      if ordered:
        # In the 2nd unrolled step of each iteration, waits for the 1st step's layer_fn(y_k, w_k)
        # so W(k) is dead before AG(k+2) starts and AG(k+2) can reuse W(k)'s buffer.
        y_k, params_next = jax.lax.optimization_barrier((y_k, params_next))
      w_next = gather_fn(params_next)  # AG(k+1) || layer k.
      y_out, saved_k = fwd_layer(y_k, w_k, state_k, layout)
      return (y_out, w_next), (y_k, saved_k)

    # Iterates k = 1 .. length-2; element j of xs has state[j+1] and next_idx = j + 2.
    (y, w_curr), (ys_mid, saved_mid) = jax.lax.scan(
        body,
        (y, w_curr),
        (_slice_range(state, 1, length - 1), jnp.arange(2, length, dtype=jnp.int32)),
        unroll=2,
    )
  y_last_in = y if length > 1 else x_in
  if length > 1:
    final_y, saved_last = fwd_layer(y_last_in, w_curr, _slice_at(state, length - 1), layout)
  else:
    final_y, saved_last = y, saved_0
  residuals = (
      x_in,
      ys_mid,
      y_last_in,
      (saved_0, saved_mid, saved_last),
      params_leaves,
      state_leaves,
      diff_args_leaves,
      diff_kwargs_leaves,
  )
  return (final_y, state), residuals


def _custom_vjp_prefetch_pipeline_bwd(graphdef, length, ctx, res, g_out):
  """Backward pass: AG(k-1) is issued while layer k is rematerialized and differentiated."""
  x_in, ys_mid, y_last_in, saved, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves = res
  saved_0, saved_mid, saved_last = saved
  g, _ = g_out  # Non-parameter layer state is not differentiated.
  params, state, gather_fn, rs_fn, layout_fn, _, layer_vjp_fn = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )

  last = length - 1
  # Tie peeled backward slices to `g` so XLA cannot CSE backward all-gathers with peeled forward
  # slices or hoist backward weight slices before the forward pass, without placing an
  # optimization_barrier on the full 12-layer parameter stack.
  params_last = _slice_at(params, last)
  params_prev_init = _slice_at(params, last - 1) if length > 1 else None
  if length > 1:
    g, params_last, params_prev_init = jax.lax.optimization_barrier((g, params_last, params_prev_init))
  else:
    g, params_last = jax.lax.optimization_barrier((g, params_last))
  w_last = gather_fn(params_last)
  layout = layout_fn(y_last_in, w_last, _slice_at(state, last))
  w_prev = gather_fn(params_prev_init) if length > 1 else None  # AG(L-2) || bwd(L-1).
  g, dw = layer_vjp_fn(y_last_in, w_last, _slice_at(state, last), g, saved_last, layout)

  def write(d_acc, dw_tree, idx, which):
    """Writes the `which` ("tagged"/"untagged"/"all") leaves of one layer's reduced dW into row `idx`."""
    acc_leaves, treedef = jax.tree_util.tree_flatten(d_acc)
    d_leaves = jax.tree_util.tree_leaves(dw_tree)
    flags = rs_fn.tag_flags(d_leaves) if which != "all" else [None] * len(d_leaves)
    out = [
        jax.lax.dynamic_update_index_in_dim(a, d, idx, axis=0) if which == "all" or (which == "tagged") == f else a
        for a, d, f in zip(acc_leaves, d_leaves, flags)
    ]
    return jax.tree_util.tree_unflatten(treedef, out)

  def deferred(dw_tree):
    """The unreduced tagged leaves of dW, carried to the next step (placeholders for the others)."""
    leaves, treedef = jax.tree_util.tree_flatten(dw_tree)
    flags = rs_fn.tag_flags(leaves)
    return jax.tree_util.tree_unflatten(treedef, [x if f else jnp.zeros((0,), x.dtype) for x, f in zip(leaves, flags)])

  # Tagged (dedicated SparseCore) reduce-scatters of layer k's dW are issued at the start of the next
  # backward step (layer k-1), so each completes with that step's FSDP all-gather and overlaps its whole
  # backward. Issued at the end of their own step, XLA's FSDP scheduler would have to complete those of
  # the last step of each loop iteration before the iteration ends, exposing them. Untagged ones complete
  # within their own step.
  d_params = jax.tree.map(jnp.zeros_like, params)
  if length == 1:
    d_params = write(d_params, rs_fn(dw), last, "all")
  else:
    d_params = write(d_params, rs_fn(dw, "untagged"), last, "untagged")
    pending = deferred(dw)
    if length > 2:
      ordered = (length - 2) // 2 >= 2

      def body(carry, xs):
        g_k, w_k, d_acc, pending = carry
        y_k, saved_k, state_k, k_idx = xs
        params_prev = _slice_at(params, k_idx - 1)
        if ordered:
          # In the 2nd unrolled step of each iteration, waits for the 1st step's g_k (which is
          # barrier-tied to dw_k) so W(k) and dW(k) are dead before AG(k-2) and recompute(k-1) start.
          g_k, y_k, saved_k, params_prev = jax.lax.optimization_barrier((g_k, y_k, saved_k, params_prev))
        d_acc = write(d_acc, rs_fn(pending, "tagged"), k_idx + 1, "tagged")  # RS(k+1) || bwd(k).
        w_km1 = gather_fn(params_prev)  # AG(k-1) || bwd(k).
        g_km1, dw_k = layer_vjp_fn(y_k, w_k, state_k, g_k, saved_k, layout)
        dw_k = rs_fn(dw_k, "untagged")
        if ordered:
          g_km1, dw_k = jax.lax.optimization_barrier((g_km1, dw_k))
        d_acc = write(d_acc, dw_k, k_idx, "untagged")
        return (g_km1, w_km1, d_acc, deferred(dw_k)), None

      # Iterates k = length-2 .. 1; element j of xs is layer k = j + 1.
      (g, w_prev, d_params, pending), _ = jax.lax.scan(
          body,
          (g, w_prev, d_params, pending),
          (ys_mid, saved_mid, _slice_range(state, 1, length - 1), jnp.arange(1, length - 1, dtype=jnp.int32)),
          reverse=True,
          unroll=2,
      )
    d_params = write(d_params, rs_fn(pending, "tagged"), 1, "tagged")  # RS(1) || bwd(0).
    g, dw0 = layer_vjp_fn(x_in, w_prev, _slice_at(state, 0), g, saved_0, layout)
    d_params = write(d_params, rs_fn(dw0), 0, "all")

  params_bar_leaves = jax.tree_util.tree_leaves(d_params)
  if ctx.scan_axis != 0:
    params_bar_leaves = [
        jnp.moveaxis(p, 0, ctx.scan_axis) if getattr(p, "ndim", 0) > ctx.scan_axis else p for p in params_bar_leaves
    ]
  params_bar_leaves = tuple(params_bar_leaves)

  # diff_*_leaves hold only arrays (non-array leaves were replaced by a dummy array), and the layer
  # arguments are treated as non-differentiable, so their cotangents are zeros.
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in diff_args_leaves)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in diff_kwargs_leaves)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_prefetch_pipeline.defvjp(_custom_vjp_prefetch_pipeline_fwd, _custom_vjp_prefetch_pipeline_bwd)


_SAVE_SUBLAYER_INPUTS = jax.checkpoint_policies.save_only_these_names("decoder_layer_input")


def _also_save_sublayer_inputs(remat_policy):
  """Returns `remat_policy` (None for full remat), extended to also save every `decoder_layer_input` it recomputes."""
  if remat_policy is None:
    return _SAVE_SUBLAYER_INPUTS

  def policy(prim, *args, **params):
    decision = remat_policy(prim, *args, **params)
    if decision is False or decision is ad_checkpoint.Recompute:
      return _SAVE_SUBLAYER_INPUTS(prim, *args, **params)
    return decision

  return policy


def mesh_uses_fsdp(mesh) -> bool:
  """Whether any FSDP mesh axis of `mesh` spans more than one device.

  Without FSDP sharding there is nothing to prefetch, so callers use the regular layer scan.
  """
  return any(mesh.shape.get(axis, 1) > 1 for axis in sharding.FSDP_MESH_AXES)


def _layer_major(params, state, scan_axis, length, move_scan_axis=True):
  """Gives `params`/`state` a leading layer axis, moving the scan axis of `params` to the front.

  With `move_scan_axis=False` the scan axis stays in place; `_prefetch_setup` then moves it inside
  the custom vjp (`ctx.scan_axis`), so the transposed stack is not materialized around the pipeline.
  """
  if move_scan_axis and scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, params)
  params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
  state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)
  return params, state


def _prepare_prefetch_call(layers, args, kwargs, *, length, mesh, config, remat_policy, prevent_cse):
  """Splits a scanned layer stack and its call arguments for `jax.custom_vjp`.

  Returns (graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx), with
  the scan axis of params still at `ctx.scan_axis` (`_prefetch_setup` moves it to the front).
  """
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  scan_axis = config.param_scan_axis
  params, state = _layer_major(params, state, scan_axis, length, move_scan_axis=False)

  sig = inspect.signature(layers.__class__.__call__)
  valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

  # jax.custom_vjp requires positional arguments (1-5) to consist strictly of differentiable JAX arrays.
  # We split positional (args) and keyword (kwargs) arguments into two parallel structures:
  # 1. diff_*_leaves: Contains ONLY JAX arrays/tracers (substituting non-arrays with dummy_arr).
  # 2. static_*_leaves: Contains non-array metadata (booleans, strings, shapes), passed via static context (ctx).
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

  ctx = _PrefetchCtx(
      mesh=mesh,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
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
  """Runs a scanned NNX layer stack with FSDP weight prefetching in the forward and backward pass.

  Args:
    layers: The scanned (layer-stacked) NNX module.
    x_in: The input activation of the first layer.
    *args: Positional arguments passed to every layer call.
    length: Number of layers in the stack.
    mesh: The device mesh.
    config: The MaxText config.
    remat_policy: The `jax.checkpoint` policy of each layer (None for full remat). The forward pass
      saves the activations this policy saves (in addition to each layer's input) and the backward
      pass recomputes the rest; gathered weights are never saved.
    prevent_cse: `prevent_cse` of the per-layer `jax.checkpoint` (only used with a remat policy).
    skip_block_remat: Whether each scanned module is a block that rematerializes its own sub-layers
      (as for `NNXDecoder._apply_layers_sequentially`). The pipeline keeps its per-module remat, since
      without it the forward pass would save the weights the block's inner scans read, but the policy
      then also saves each sub-layer's `decoder_layer_input`: the backward pass rebuilds every
      sub-layer from its input once, instead of first recomputing the whole block.
    **kwargs: Keyword arguments passed to every layer call (those not accepted by the layer are dropped).

  Returns:
    (final activation, layers, None). Non-parameter layer state is read-only in this path: state
    mutated inside a layer call (e.g. sown intermediates) is not propagated back to `layers`.

  See `_custom_vjp_prefetch_pipeline`. Inhomogeneous blocks (e.g. Qwen3.5) can instead prefetch one
  sub-layer at a time with `fsdp_prefetch_inhomogeneous`.
  """
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
