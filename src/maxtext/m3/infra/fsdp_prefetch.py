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

"""Path-transparent FSDP weight prefetching for m3 `ScannedLayers`.

Enabled with `prefetch_fsdp_weights=True` (requires `scan_layers=True` and a
non-`none` remat policy). Replaces the default `unroll=1` scan execution with a
`jax.custom_vjp` `unroll=2` double-buffered pipeline that overlaps:

  * Forward layer k:  all-gather W(k+1) || compute layer k
  * Backward layer k: all-gather W(k-1) || (recompute + vjp layer k),
                      plus reduce-scatter dW(k) (or deferred dW(k+1) when
                      `prefetch_fsdp_defer_grad_reduce_scatter=True` under
                      `shard_mode=explicit`).

The transform attaches a static runner to each `ScannedLayers` instance in place
without inserting wrapper modules, keeping `nnx.state(model, nnx.Param)` paths
identical and leaving compiled HLO instructions, collectives, and HBM identical when disabled.

Known limitation: `remat_policy='minimal_offloaded'` (and offloaded `custom`
residuals) compile and match numerics via explicit pinned-host stack streaming,
but synchronous host-stack updates currently stall PCIe DMA on TPU7x (`~3.3x`
slower than unprefetched `minimal_offloaded`); pair `prefetch_fsdp_weights=True`
with `remat_policy='full'` or device-resident savepoints for peak throughput.
"""

import collections
from collections.abc import Callable
import dataclasses
import functools
from typing import Any

from flax import nnx
import jax
from jax import ad_checkpoint
from jax import checkpoint_policies as cp
from jax.experimental.layout import Layout, with_layout_constraint
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import scan as m3_scan
from maxtext.utils import max_logging

_TRAIN_MODE = "train"
FSDP_MESH_AXES = ("fsdp", "fsdp_transpose")
FSDP_CO_AXES = FSDP_MESH_AXES + ("context", "expert")
# Activation contracting axes besides FSDP that can contribute to unreduced weight grads.
_EXTRA_BATCH_AXES = ("data", "context", "expert")

_PASSTHROUGH_PRIMITIVES = frozenset(
    {"name", "copy", "copy_p", "convert_element_type", "sharding_constraint", "reshard", "mesh_cast"}
)


def fsdp_axes_for_dim(entry: Any) -> tuple[str, ...]:
  """Returns FSDP and trailing co-sharded axes on one PartitionSpec dimension."""
  axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
  first_fsdp = next((i for i, a in enumerate(axes) if a in FSDP_MESH_AXES), None)
  if first_fsdp is None:
    return ()
  return tuple(a for a in axes[first_fsdp:] if a in FSDP_CO_AXES)


def remove_fsdp_from_spec(spec: P) -> P:
  """Strips FSDP and trailing co-sharded axes from every dimension of `spec`."""
  out = []
  for entry in tuple(spec):
    axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
    drop = set(fsdp_axes_for_dim(axes))
    kept = tuple(a for a in axes if a not in drop)
    out.append(kept if len(kept) > 1 else (kept[0] if kept else None))
  return P(*out)


def mesh_uses_fsdp(mesh: Any) -> bool:
  """Whether `mesh` has an active FSDP axis (`fsdp` or `fsdp_transpose` > 1)."""
  if mesh is None or getattr(mesh, "empty", False):
    return False
  shape = getattr(mesh, "shape", None) or {}
  return any(int(shape.get(axis, 1)) > 1 for axis in FSDP_MESH_AXES)


def _is_explicit_mesh(mesh: Any) -> bool:
  axis_types = tuple(getattr(mesh, "axis_types", ()))
  return bool(axis_types) and all(t == AxisType.Explicit for t in axis_types)


def _default_sparse_core_id(mesh: Any) -> int | None:
  """Pins FSDP prefetch collectives to SparseCore 1 on dual-SparseCore TPU7x."""
  devices = getattr(mesh, "devices", None)
  if devices is None or devices.size == 0:
    return None
  device_kind = getattr(devices.flat[0], "device_kind", "").lower()
  if any(tag in device_kind for tag in ("7x", "tpu7", "v7")):
    return 1
  return None


def _has_multi_trip_loop(length: int) -> bool:
  """Returns True when the `unroll=2` middle scan has >= 2 trips (`length >= 6`).

  The pipeline peels layer 0 and layer `length - 1` outside an `unroll=2` scan
  over the remaining `length - 2` layers. When `(length - 2) // 2 < 2`
  (`length < 6`), XLA unrolls/inlines the body into `ENTRY`, where TPU7x
  `compute_on("tpu_sparsecore")` hits `UNIMPLEMENTED: ComputeEmitter` on
  `all-gather(%convert)`. Gating SparseCore pinning, intra-loop barriers, and
  host-stack offloading on `length >= 6` avoids that crash.
  """
  return length > 2 and (length - 2) // 2 >= 2


def _on_sparse_core(fn: Callable[..., Any], core_id: int | None, x: jax.Array | None = None) -> Callable[..., Any]:
  """Pins `fn` to SparseCore `core_id` when supported by the operand shape."""
  if core_id is None or core_id < 0:
    return fn
  if x is not None and (x.ndim < 2 or x.shape[-1] < 128):
    return fn
  from jax.experimental import compute_on as _co_mod  # pylint: disable=import-outside-toplevel

  _co = getattr(_co_mod, "compute_on2", _co_mod.compute_on)
  return _co(
      fn,
      compute_type="tpu_sparsecore",
      out_memory_spaces=jax.memory.Space.Device,
      compiler_options={"sparse_core_config": {"core_ids": [core_id]}},
  )


def _register_overlap_transform(mesh: Any) -> None:
  """Registers the TPU post-scheduler async overlap pass when running on TPU."""
  devices = getattr(mesh, "devices", None)
  if devices is None or devices.size == 0 or getattr(devices.flat[0], "platform", None) != "tpu":
    return
  from maxtext.layers import fsdp_prefetch_overlap_transform  # pylint: disable=import-outside-toplevel

  fsdp_prefetch_overlap_transform.register_transform(platforms="tpu")


def _sharding_axes_of_pytree(pytree: Any) -> frozenset[str]:
  """Returns mesh axis names appearing in any array leaf's PartitionSpec in `pytree`."""
  axes: set[str] = set()
  for leaf in jax.tree_util.tree_leaves(pytree):
    if not isinstance(leaf, jax.Array):
      continue
    s = getattr(jax.typeof(leaf), "sharding", None) or getattr(leaf, "sharding", None)
    spec = getattr(s, "spec", None)
    if isinstance(spec, P):
      for entry in tuple(spec):
        for a in (entry,) if isinstance(entry, str) else (entry or ()):
          axes.add(a)
  return frozenset(axes)


def _extra_batch_axes(
    mesh: Any,
    spec: P,
    fsdp_axes: tuple[str, ...],
    candidate_axes: tuple[str, ...] = _EXTRA_BATCH_AXES,
) -> tuple[str, ...]:
  used = {a for entry in tuple(spec) for a in ((entry,) if isinstance(entry, str) else (entry or ()))}
  return tuple(a for a in candidate_axes if mesh.shape.get(a, 1) > 1 and a not in used and a not in fsdp_axes)


def _expected_unreduced_axes(
    mesh: Any,
    src_spec: P,
    fsdp_axes: tuple[str, ...],
    candidate_axes: tuple[str, ...] = _EXTRA_BATCH_AXES,
) -> frozenset[str]:
  return frozenset(fsdp_axes) | frozenset(_extra_batch_axes(mesh, src_spec, fsdp_axes, candidate_axes))


def _fsdp_dims(ndim: int, mesh: Any, src_spec: P) -> list[tuple[int, tuple[str, ...]]] | None:
  """Returns `(dim, fsdp_axes)` pairs where FSDP axes are minor-most, or None if non-minor."""
  spec = tuple(src_spec) + (None,) * (ndim - len(tuple(src_spec)))
  dims: list[tuple[int, tuple[str, ...]]] = []
  for i, entry in enumerate(spec):
    axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
    axes = tuple(a for a in axes if mesh.shape.get(a, 1) > 1)
    fsdp = fsdp_axes_for_dim(axes)
    if not fsdp:
      continue
    if axes[len(axes) - len(fsdp) :] != fsdp:
      return None
    dims.append((i, fsdp))
  return dims


def _fsdp_scatter_dim(ndim: int, mesh: Any, src_spec: P) -> tuple[int, tuple[str, ...]] | None:
  dims = _fsdp_dims(ndim, mesh, src_spec)
  if not dims or len(dims) != 1:
    return None
  return dims[0]


def _is_unreduced_over(g: jax.Array, axes: frozenset[str]) -> bool:
  g_spec = getattr(getattr(jax.typeof(g), "sharding", None), "spec", None)
  return g_spec is not None and frozenset(g_spec.unreduced) == frozenset(axes)


def _pin_scatter_dim_major(g: jax.Array, dim: int) -> jax.Array:
  """Places `dim` major-most so SparseCore can emit a direct reduce-scatter."""
  return with_layout_constraint(g, Layout((dim,) + tuple(i for i in range(g.ndim) if i != dim)))


def _explicit_fsdp_all_gather(
    w: jax.Array,
    mesh: Mesh,
    src_spec: P,
    gathered_spec: P,
    *,
    reduced: bool = False,
    sparse_core_id: int | None = None,
    candidate_batch_axes: tuple[str, ...] = _EXTRA_BATCH_AXES,
) -> jax.Array | None:
  """All-gathers `w` over its FSDP axes with an explicit `shard_map` collective."""
  dims = _fsdp_dims(w.ndim, mesh, src_spec)
  if not dims:
    return None
  spec = tuple(src_spec) + (None,) * (w.ndim - len(tuple(src_spec)))

  if reduced and len(dims) == 1:
    dim, ax = dims[0]
    gathered = tuple(gathered_spec) + (None,) * (w.ndim - len(tuple(gathered_spec)))
    in_spec_list = []
    for entry in spec:
      entry_axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
      stripped = set(fsdp_axes_for_dim(entry_axes))
      kept = tuple(a for a in entry_axes if a not in stripped or mesh.shape.get(a, 1) > 1)
      in_spec_list.append(kept if len(kept) > 1 else (kept[0] if kept else None))
    in_spec = P(*in_spec_list)
    if tuple(in_spec) != spec:
      w = jax.sharding.reshard(w, NamedSharding(mesh, in_spec))
    extra = _extra_batch_axes(mesh, in_spec, ax, candidate_batch_axes)

    def all_gather_reduced(x):
      return jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=dim, tiled=True, to="reduced")

    def gather_reduced(x):
      x = _on_sparse_core(all_gather_reduced, sparse_core_id, x)(x)
      if extra:
        x = jax.lax.pcast(x, extra if len(extra) > 1 else extra[0], to="reduced")
      return x

    return jax.shard_map(
        gather_reduced,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=P(*gathered, reduced=frozenset(ax) | frozenset(extra)),
    )(w)

  def gather_all(x):
    for i, ax in dims:
      x = jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=i, tiled=True)
    return x

  def gather(x):
    return _on_sparse_core(gather_all, sparse_core_id, x)(x)

  return jax.shard_map(gather, mesh=mesh, in_specs=P(*spec), out_specs=gathered_spec, check_vma=False)(w)


def _explicit_fsdp_reduce_scatter(
    g: jax.Array,
    mesh: Mesh,
    src_spec: P,
    gathered_spec: P,
    *,
    sparse_core_id: int | None = None,
    candidate_batch_axes: tuple[str, ...] = _EXTRA_BATCH_AXES,
) -> jax.Array | None:
  """Reduce-scatters an `unreduced` gathered-weight gradient back to `src_spec`."""
  scatter = _fsdp_scatter_dim(g.ndim, mesh, src_spec)
  if scatter is None or not _is_unreduced_over(
      g, _expected_unreduced_axes(mesh, src_spec, scatter[1], candidate_batch_axes)
  ):
    return None
  dim, ax = scatter
  extra = _extra_batch_axes(mesh, src_spec, ax, candidate_batch_axes)
  spec = tuple(src_spec) + (None,) * (g.ndim - len(tuple(src_spec)))
  gathered = tuple(gathered_spec) + (None,) * (g.ndim - len(tuple(gathered_spec)))
  g_spec = jax.typeof(g).sharding.spec
  g = _pin_scatter_dim_major(g, dim)

  def psum_scatter(x):
    if dim != 0 and sparse_core_id is not None and sparse_core_id >= 0:
      # `compute_on` outlines its call boundary with row-major `{N-1, ..., 0}` layout; moving
      # `dim` to axis 0 keeps the scatter axis major-most for 2D and 3D+ SparseCore reduce-scatters.
      x = jnp.moveaxis(x, dim, 0)
      x = _on_sparse_core(
          functools.partial(
              jax.lax.psum_scatter, axis_name=ax if len(ax) > 1 else ax[0], scatter_dimension=0, tiled=True
          ),
          sparse_core_id,
          x,
      )(x)
      x = jnp.moveaxis(x, 0, dim)
    else:
      x = _on_sparse_core(
          functools.partial(
              jax.lax.psum_scatter, axis_name=ax if len(ax) > 1 else ax[0], scatter_dimension=dim, tiled=True
          ),
          sparse_core_id,
          x,
      )(x)
    if extra:
      x = jax.lax.psum(x, extra if len(extra) > 1 else extra[0])
    return x

  return jax.shard_map(
      psum_scatter,
      mesh=mesh,
      in_specs=P(*gathered, unreduced=frozenset(ax) | frozenset(extra), unreduced_kind=g_spec.unreduced_kind),
      out_specs=P(*spec),
  )(g)


def _needs_minor_dim_layout_pin(per_layer_ndim: int, layer_spec: P | None, mesh: Any) -> bool:
  if per_layer_ndim < 3 or layer_spec is None or len(layer_spec) < per_layer_ndim:
    return False
  entry = layer_spec[per_layer_ndim - 1]
  axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
  axes = tuple(a for a in axes if mesh.shape.get(a, 1) > 1)
  return bool(fsdp_axes_for_dim(axes))


def _is_literal(v: Any) -> bool:
  return hasattr(v, "val")


def _is_passthrough(eqn: Any) -> bool:
  """Returns True when `eqn` preserves value identity for residual origin tracking."""
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


class _ResidualLayout(collections.namedtuple("_ResidualLayout", ["treedef", "src", "saved_avals", "saved_offload_mask"])):
  """Origin map for each residual leaf of `jax.vjp(remat(layer), carry, w)`."""


def _split_diff_consts(
    consts: tuple[Any, ...], diff_kw_mask: tuple[bool, ...]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
  diff_c = tuple(c for c, is_diff in zip(consts, diff_kw_mask) if is_diff)
  nondiff_c = tuple(c for c, is_diff in zip(consts, diff_kw_mask) if not is_diff)
  return diff_c, nondiff_c


def _merge_diff_consts(
    diff_c: tuple[Any, ...], nondiff_c: tuple[Any, ...], diff_kw_mask: tuple[bool, ...]
) -> tuple[Any, ...]:
  it_d = iter(diff_c)
  it_n = iter(nondiff_c)
  return tuple(next(it_d) if is_diff else next(it_n) for is_diff in diff_kw_mask)


def _vjp_residual_layout(
    f: Callable[..., Any],
    carry: Any,
    w: Any,
    layer_rest: Any,
    consts: Any,
    diff_kw_mask: tuple[bool, ...] = (),
) -> _ResidualLayout:
  """Traces `jax.vjp(f, carry, w, diff_consts)` once and classifies input vs policy residuals."""
  y_leaves, y_treedef = jax.tree_util.tree_flatten(carry)
  w_leaves, w_treedef = jax.tree_util.tree_flatten(w)
  s_leaves, s_treedef = jax.tree_util.tree_flatten(layer_rest)
  c_leaves, c_treedef = jax.tree_util.tree_flatten(consts)
  treedef_box: list[Any] = []

  def residual_leaves(yl, wl, sl, cl):
    y_ = jax.tree_util.tree_unflatten(y_treedef, yl)
    w_ = jax.tree_util.tree_unflatten(w_treedef, wl)
    s_ = jax.tree_util.tree_unflatten(s_treedef, sl)
    c_ = jax.tree_util.tree_unflatten(c_treedef, cl)
    dc_, nc_ = _split_diff_consts(c_, diff_kw_mask)
    _, vjp_fn, _ = jax.vjp(
        lambda y2, w2, dc2: f(y2, w2, s_, _merge_diff_consts(dc2, nc_, diff_kw_mask)),
        y_,
        w_,
        dc_,
        has_aux=True,
    )
    leaves, treedef = jax.tree_util.tree_flatten(vjp_fn)
    treedef_box.append(treedef)
    return leaves

  closed = jax.make_jaxpr(residual_leaves)(y_leaves, w_leaves, s_leaves, c_leaves)
  jaxpr = closed.jaxpr
  n_y = len(y_leaves)
  n_w = len(w_leaves)
  n_s = len(s_leaves)
  origin: dict[Any, tuple[str, int]] = {}
  for i, v in enumerate(jaxpr.invars):
    if i < n_y:
      origin[v] = ("y", i)
    elif i < n_y + n_w:
      origin[v] = ("w", i - n_y)
    elif i < n_y + n_w + n_s:
      origin[v] = ("s", i - n_y - n_w)
    else:
      origin[v] = ("c", i - n_y - n_w - n_s)
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

  src: list[tuple[str, int] | None] = []
  saved_avals: list[Any] = []
  saved_offload_mask: list[bool] = []
  host_space = getattr(getattr(jax, "memory", None), "Space", None)
  host_space_val = getattr(host_space, "Host", None)
  for v in jaxpr.outvars:
    s = None if _is_literal(v) else origin.get(v)
    src.append(s)
    if s is None:
      saved_avals.append(v.aval)
      mem_space = getattr(v.aval, "memory_space", None)
      mem_kind = getattr(getattr(v.aval, "sharding", None), "memory_kind", None)
      is_host = (host_space_val is not None and mem_space == host_space_val) or mem_kind == "pinned_host"
      saved_offload_mask.append(bool(is_host))
  return _ResidualLayout(treedef_box[0], tuple(src), tuple(saved_avals), tuple(saved_offload_mask))


def _split_residuals(vjp_fn: Any, layout: _ResidualLayout) -> tuple[Any, ...]:
  leaves = jax.tree_util.tree_leaves(vjp_fn)
  return tuple(leaf for leaf, s in zip(leaves, layout.src) if s is None)


def _rebuild_vjp(layout: _ResidualLayout, saved: tuple[Any, ...], carry: Any, w: Any, layer_rest: Any, consts: Any):
  groups = {
      "y": jax.tree_util.tree_leaves(carry),
      "w": jax.tree_util.tree_leaves(w),
      "s": jax.tree_util.tree_leaves(layer_rest),
      "c": jax.tree_util.tree_leaves(consts),
  }
  saved_iter = iter(saved)
  leaves = [next(saved_iter) if s is None else groups[s[0]][s[1]] for s in layout.src]
  return jax.tree_util.tree_unflatten(layout.treedef, leaves)


def _slice_at(tree: Any, idx: Any) -> Any:
  return jax.tree.map(lambda x: x[idx], tree)


def _slice_range(tree: Any, start: int, end: int) -> Any:
  return jax.tree.map(lambda x: x[start:end], tree)


def _with_memory_kind(x: jax.Array, kind: str) -> jax.Array:
  s = jax.typeof(x).sharding
  if not isinstance(s, NamedSharding):
    return x
  return jax.device_put(x, s.with_memory_kind(kind))


def _host_stack_buffer(carry: Any, n: int) -> Any:
  def _alloc(y):
    return _with_memory_kind(jnp.zeros_like(jnp.broadcast_to(y[None], (n,) + y.shape)), "pinned_host")

  return jax.tree.map(_alloc, carry)


def _host_stack_write(buf: Any, carry: Any, idx: Any) -> Any:
  def _write(b, y):
    return jax.lax.dynamic_update_slice(b, _with_memory_kind(y[None], "pinned_host"), (idx,) + (0,) * y.ndim)

  return jax.tree.map(_write, buf, carry)


def _host_stack_read(buf: Any, idx: Any) -> Any:
  def _read(b):
    s = jax.lax.dynamic_slice(b, (idx,) + (0,) * (b.ndim - 1), (1,) + b.shape[1:])
    return jnp.squeeze(_with_memory_kind(s, "device"), axis=0)

  return jax.tree.map(_read, buf)


def _keep_layer_inputs_on_device(remat_policy: Any) -> Any:
  """Keeps `decoder_layer_input` on device inside `jax.checkpoint` when streamed via host stack."""
  if remat_policy is None:
    return None

  def policy(prim, *args, **params):
    decision = remat_policy(prim, *args, **params)
    if params.get("name") == "decoder_layer_input" and isinstance(decision, ad_checkpoint.Offloadable):
      return ad_checkpoint.Saveable
    return decision

  return policy


def _device_only_policy(remat_policy: Any) -> Any:
  """Converts `Offloadable` decisions to `Saveable` so the prefetch loop handles host DMA."""
  if remat_policy is None:
    return None

  def policy(prim, *args, **params):
    decision = remat_policy(prim, *args, **params)
    if isinstance(decision, ad_checkpoint.Offloadable):
      return ad_checkpoint.Saveable
    return decision

  return policy


def _merge_offload_residual_layout(raw_layout: _ResidualLayout, dev_layout: _ResidualLayout) -> _ResidualLayout:
  """Attaches `raw_layout.saved_offload_mask` to `dev_layout`, failing closed on count mismatch."""
  if len(dev_layout.saved_avals) != len(raw_layout.saved_offload_mask):
    raise ValueError(
        "Device-only remat rewrite changed the saved residual leaf count "
        f"({len(dev_layout.saved_avals)} vs {len(raw_layout.saved_offload_mask)}); "
        "cannot safely stream offloaded residuals through host-stack buffers."
    )
  return dev_layout._replace(saved_offload_mask=raw_layout.saved_offload_mask)


_PrefetchCtx = collections.namedtuple(
    "_PrefetchCtx",
    [
        "mesh",
        "explicit",
        "extra_batch_axes",
        "leaf_scan_axes",
        "layer_specs",
        "params_treedef",
        "rest_treedef",
        "has_rest",
        "kw_treedef",
        "kw_is_array",
        "static_kw_leaves",
        "diff_kw_mask",
        "remat_policy",
        "prevent_cse",
        "defer_grad_reduce_scatter",
        "offload_layer_inputs",
        "sparse_core_id",
        "layout_cache",
    ],
)


def _prefetch_setup(graphdef, params_leaves, rest_leaves, dyn_kw_leaves, ctx: _PrefetchCtx):
  """Builds per-layer gather, reduce-scatter, forward, and backward callables."""
  moved_params = [
      jnp.moveaxis(p, ax, 0) if ax != 0 and getattr(p, "ndim", 0) > ax else p
      for p, ax in zip(params_leaves, ctx.leaf_scan_axes)
  ]
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, moved_params)
  rest = jax.tree_util.tree_unflatten(ctx.rest_treedef, rest_leaves)
  layer_specs = ctx.layer_specs
  layer_shardings = [NamedSharding(ctx.mesh, s) if s is not None else None for s in layer_specs]
  pin_layout_flags = [
      _needs_minor_dim_layout_pin(getattr(p, "ndim", 1) - 1, s, ctx.mesh) for p, s in zip(moved_params, layer_specs)
  ]
  consts = tuple(dyn_kw_leaves)
  diff_kw_mask = ctx.diff_kw_mask
  diff_consts, nondiff_consts = _split_diff_consts(consts, diff_kw_mask)
  batch_axes = ctx.extra_batch_axes

  def gather_fn(layer_params):
    leaves, treedef = jax.tree_util.tree_flatten(layer_params)
    out = []
    for x, src_sharding, pin in zip(leaves, layer_shardings, pin_layout_flags):
      if src_sharding is not None:
        if not ctx.explicit:
          x = jax.lax.with_sharding_constraint(x, src_sharding)
        gathered_spec = remove_fsdp_from_spec(src_sharding.spec)
        gathered = _explicit_fsdp_all_gather(
            x,
            ctx.mesh,
            src_sharding.spec,
            gathered_spec,
            reduced=ctx.explicit and ctx.mesh.shape.get("tensor_sequence", 1) == 1,
            sparse_core_id=ctx.sparse_core_id,
            candidate_batch_axes=batch_axes,
        )
        if gathered is not None:
          x = gathered
        elif ctx.explicit:
          x = jax.sharding.reshard(x, NamedSharding(ctx.mesh, gathered_spec))
        else:
          x = jax.lax.with_sharding_constraint(x, NamedSharding(ctx.mesh, gathered_spec))
        if pin:
          x = with_layout_constraint(x, Layout(tuple(range(x.ndim))))
      out.append(x)
    return jax.tree_util.tree_unflatten(treedef, out)

  def layer_pure(carry_in, layer_weights, layer_rest, layer_consts):
    it_dyn = iter(layer_consts)
    it_sta = iter(ctx.static_kw_leaves)
    call_kw = jax.tree_util.tree_unflatten(
        ctx.kw_treedef,
        [next(it_dyn) if is_arr else next(it_sta) for is_arr in ctx.kw_is_array],
    )
    layer = nnx.merge(graphdef, layer_weights, layer_rest)
    carry_out = super(m3_scan.ScannedLayers, layer).__call__(carry_in, **call_kw)
    _, _, new_rest = nnx.split(layer, nnx.Param, ...)
    if jax.tree_util.tree_structure(new_rest) != ctx.rest_treedef:
      raise ValueError(
          "ScannedLayers with prefetch_fsdp_weights=True cannot create new non-Param state "
          "variables inside __call__ (for example via self.sow); initialize state in __init__."
      )
    return carry_out, new_rest

  def rs_one(g, src_sharding, tag):
    if src_sharding is None:
      return g
    gathered_spec = remove_fsdp_from_spec(src_sharding.spec)
    if tag:
      out = _explicit_fsdp_reduce_scatter(
          g,
          ctx.mesh,
          src_sharding.spec,
          gathered_spec,
          sparse_core_id=ctx.sparse_core_id,
          candidate_batch_axes=batch_axes,
      )
      if out is not None:
        return out
    scatter = _fsdp_scatter_dim(g.ndim, ctx.mesh, src_sharding.spec)
    if scatter is not None and _is_unreduced_over(
        g, _expected_unreduced_axes(ctx.mesh, src_sharding.spec, scatter[1], batch_axes)
    ):
      out = _explicit_fsdp_reduce_scatter(
          g,
          ctx.mesh,
          src_sharding.spec,
          gathered_spec,
          sparse_core_id=ctx.sparse_core_id,
          candidate_batch_axes=batch_axes,
      )
      if out is not None:
        return out
      g = _pin_scatter_dim_major(g, scatter[0])
    if ctx.explicit:
      return jax.sharding.reshard(g, src_sharding)
    return jax.lax.with_sharding_constraint(g, src_sharding)

  tag_cache: dict[str, list[bool]] = {}

  def tag_flags(leaves):
    if "flags" not in tag_cache:
      flags = []
      for g, s in zip(leaves, layer_shardings):
        scatter = None if s is None else _fsdp_scatter_dim(g.ndim, ctx.mesh, s.spec)
        eligible = scatter is not None and _is_unreduced_over(
            g, _expected_unreduced_axes(ctx.mesh, s.spec, scatter[1], batch_axes)
        )
        flags.append(bool(ctx.defer_grad_reduce_scatter and eligible))
      tag_cache["flags"] = flags
    return tag_cache["flags"]

  def rs_fn(layer_grads, which="all"):
    leaves, treedef = jax.tree_util.tree_flatten(layer_grads)
    flags = tag_flags(leaves)
    out = [
        rs_one(g, s, tag) if which == "all" or (which == "tagged") == tag else g
        for g, s, tag in zip(leaves, layer_shardings, flags)
    ]
    return jax.tree_util.tree_unflatten(treedef, out)

  rs_fn.tag_flags = tag_flags

  raw_remat_layer = jax.checkpoint(layer_pure, policy=ctx.remat_policy, prevent_cse=ctx.prevent_cse)
  dev_remat_layer = jax.checkpoint(layer_pure, policy=_device_only_policy(ctx.remat_policy), prevent_cse=ctx.prevent_cse)

  def layout_fn(carry_in, w, layer_rest):
    key = (
        id(graphdef),
        ctx.params_treedef,
        ctx.rest_treedef,
        ctx.kw_treedef,
        diff_kw_mask,
        tuple((jax.typeof(p).shape, jax.typeof(p).dtype) for p in jax.tree_util.tree_leaves(w)),
        tuple((jax.typeof(c).shape, jax.typeof(c).dtype) for c in consts),
    )

    def _build():
      raw_layout = _vjp_residual_layout(raw_remat_layer, carry_in, w, layer_rest, consts, diff_kw_mask)
      if any(raw_layout.saved_offload_mask):
        dev_layout = _vjp_residual_layout(dev_remat_layer, carry_in, w, layer_rest, consts, diff_kw_mask)
        return _merge_offload_residual_layout(raw_layout, dev_layout)
      return raw_layout

    return ctx.layout_cache(key, _build)

  def fwd_layer(carry_in, w, layer_rest, layout):
    if ctx.remat_policy is cp.nothing_saveable and not layout.saved_avals:
      carry_out, new_rest = layer_pure(carry_in, w, layer_rest, consts)
      return carry_out, new_rest, ()
    active_remat = dev_remat_layer if any(layout.saved_offload_mask) else raw_remat_layer
    carry_out, vjp_fn, new_rest = jax.vjp(
        lambda c_, w_, dc_: active_remat(c_, w_, layer_rest, _merge_diff_consts(dc_, nondiff_consts, diff_kw_mask)),
        carry_in,
        w,
        diff_consts,
        has_aux=True,
    )
    return carry_out, new_rest, _split_residuals(vjp_fn, layout)

  def layer_vjp_fn(carry_in, w, layer_rest, g_y, saved, layout):
    vjp_fn = _rebuild_vjp(layout, saved, carry_in, w, layer_rest, consts)
    return vjp_fn(g_y)

  return params, rest, gather_fn, rs_fn, layout_fn, fwd_layer, layer_vjp_fn


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 5, 6))
def _custom_vjp_prefetch_pipeline(graphdef, carry_in, params_leaves, rest_leaves, dyn_kw_leaves, length, ctx):
  """Runs `length >= 2` layers with double-buffered forward & backward weight prefetch."""
  out, _ = _custom_vjp_prefetch_pipeline_fwd(graphdef, carry_in, params_leaves, rest_leaves, dyn_kw_leaves, length, ctx)
  return out


def _custom_vjp_prefetch_pipeline_fwd(graphdef, carry_in, params_leaves, rest_leaves, dyn_kw_leaves, length, ctx):
  """Forward pass of the double-buffered FSDP weight-prefetch scan pipeline."""
  params, rest, gather_fn, _, layout_fn, fwd_layer, _ = _prefetch_setup(
      graphdef, params_leaves, rest_leaves, dyn_kw_leaves, ctx
  )
  w_0 = gather_fn(_slice_at(params, 0))
  layout = layout_fn(carry_in, w_0, _slice_at(rest, 0))
  w_curr = gather_fn(_slice_at(params, 1))
  y, new_rest_0, saved_0 = fwd_layer(carry_in, w_0, _slice_at(rest, 0), layout)
  ordered = _has_multi_trip_loop(length)
  offload_mask = layout.saved_offload_mask
  offload_saved = ordered and any(offload_mask)
  ys_mid = None
  saved_mid = None
  saved_host_bufs = None
  rest_mid = None
  if length > 2:
    offload_mid = ctx.offload_layer_inputs and ordered

    def body(carry, xs):
      y_k, w_k, ys_buf, s_bufs = carry
      rest_k, next_idx = xs
      params_next = _slice_at(params, next_idx)
      if ordered:
        y_k, params_next = jax.lax.optimization_barrier((y_k, params_next))
      w_next = gather_fn(params_next)
      y_out, new_rest_k, saved_k = fwd_layer(y_k, w_k, rest_k, layout)
      rest_out = new_rest_k if ctx.has_rest else None
      if offload_saved:
        new_s_bufs = []
        dev_saved_k = []
        for s_leaf, buf, is_off in zip(saved_k, s_bufs, offload_mask):
          if is_off:
            new_s_bufs.append(_host_stack_write(buf, s_leaf, next_idx - 2))
            dev_saved_k.append(None)
          else:
            new_s_bufs.append(None)
            dev_saved_k.append(s_leaf)
        s_bufs = tuple(new_s_bufs)
        saved_out = tuple(dev_saved_k)
      else:
        saved_out = saved_k
      if offload_mid:
        ys_buf = _host_stack_write(ys_buf, y_k, next_idx - 2)
        return (y_out, w_next, ys_buf, s_bufs), (None, saved_out, rest_out)
      return (y_out, w_next, ys_buf, s_bufs), (y_k, saved_out, rest_out)

    ys_buf = _host_stack_buffer(y, length - 2) if offload_mid else None
    s_bufs_init = (
        tuple(_host_stack_buffer(s_leaf, length - 2) if is_off else None for s_leaf, is_off in zip(saved_0, offload_mask))
        if offload_saved
        else None
    )
    (y, w_curr, ys_buf, saved_host_bufs), (ys_mid, saved_mid, rest_mid) = jax.lax.scan(
        body,
        (y, w_curr, ys_buf, s_bufs_init),
        (_slice_range(rest, 1, length - 1), jnp.arange(2, length, dtype=jnp.int32)),
        unroll=2,
    )
    if offload_mid:
      ys_mid = ys_buf
  y_last_in = y
  final_y, new_rest_last, saved_last = fwd_layer(y_last_in, w_curr, _slice_at(rest, length - 1), layout)
  if offload_saved:
    saved_0 = tuple(_with_memory_kind(s, "pinned_host") if is_off else s for s, is_off in zip(saved_0, offload_mask))
    saved_last = tuple(
        _with_memory_kind(s, "pinned_host") if is_off else s for s, is_off in zip(saved_last, offload_mask)
    )
    saved_mid = (saved_mid, saved_host_bufs)
  if ctx.has_rest:
    if length > 2:
      new_rest_full = jax.tree.map(
          lambda r0, rm, rl: jnp.concatenate([r0[None], rm, rl[None]], axis=0),
          new_rest_0,
          rest_mid,
          new_rest_last,
      )
    else:
      new_rest_full = jax.tree.map(lambda r0, rl: jnp.stack([r0, rl], axis=0), new_rest_0, new_rest_last)
  else:
    new_rest_full = rest
  residuals = (
      carry_in,
      ys_mid,
      y_last_in,
      (saved_0, saved_mid, saved_last),
      params_leaves,
      rest_leaves,
      dyn_kw_leaves,
  )
  return (final_y, new_rest_full), residuals


def _custom_vjp_prefetch_pipeline_bwd(graphdef, length, ctx, res, g_out):
  """Backward pass overlapping layer k-1 weight all-gather with layer k recompute and VJP."""
  carry_in, ys_mid, y_last_in, saved, params_leaves, rest_leaves, dyn_kw_leaves = res
  saved_0, saved_mid, saved_last = saved
  g, _ = g_out
  params, rest, gather_fn, rs_fn, layout_fn, _, layer_vjp_fn = _prefetch_setup(
      graphdef, params_leaves, rest_leaves, dyn_kw_leaves, ctx
  )

  last = length - 1
  params_last = _slice_at(params, last)
  params_prev_init = _slice_at(params, last - 1)
  g, params_last, params_prev_init = jax.lax.optimization_barrier((g, params_last, params_prev_init))
  w_last = gather_fn(params_last)
  layout = layout_fn(y_last_in, w_last, _slice_at(rest, last))
  ordered = _has_multi_trip_loop(length)
  offload_mask = layout.saved_offload_mask
  offload_saved = ordered and any(offload_mask)
  if offload_saved:
    saved_mid, saved_host_bufs = saved_mid
    saved_last = tuple(_with_memory_kind(s, "device") if is_off else s for s, is_off in zip(saved_last, offload_mask))
  else:
    saved_host_bufs = None
  w_prev = gather_fn(params_prev_init)
  g, dw, d_kw = layer_vjp_fn(y_last_in, w_last, _slice_at(rest, last), g, saved_last, layout)

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

  d_params = jax.tree.map(jnp.zeros_like, params)
  d_params = write(d_params, rs_fn(dw, "untagged"), last, "untagged")
  pending = deferred(dw)
  if length > 2:
    offload_mid = ctx.offload_layer_inputs and ordered

    def body(carry, xs):
      g_k, w_k, d_acc, pending_rs, d_kw_acc = carry
      y_k, saved_k, rest_k, k_idx = xs
      if offload_mid:
        y_k = _host_stack_read(ys_mid, k_idx - 1)
      if offload_saved:
        saved_k = tuple(
            _host_stack_read(buf, k_idx - 1) if is_off else s_leaf
            for s_leaf, buf, is_off in zip(saved_k, saved_host_bufs, offload_mask)
        )
      params_prev = _slice_at(params, k_idx - 1)
      if ordered:
        g_k, y_k, saved_k, params_prev = jax.lax.optimization_barrier((g_k, y_k, saved_k, params_prev))
      d_acc = write(d_acc, rs_fn(pending_rs, "tagged"), k_idx + 1, "tagged")
      w_km1 = gather_fn(params_prev)
      g_km1, dw_k, d_kw_k = layer_vjp_fn(y_k, w_k, rest_k, g_k, saved_k, layout)
      dw_k = rs_fn(dw_k, "untagged")
      d_kw_acc = tuple(acc + dk for acc, dk in zip(d_kw_acc, d_kw_k))
      if ordered:
        g_km1, dw_k, d_kw_acc = jax.lax.optimization_barrier((g_km1, dw_k, d_kw_acc))
      d_acc = write(d_acc, dw_k, k_idx, "untagged")
      return (g_km1, w_km1, d_acc, deferred(dw_k), d_kw_acc), None

    ys_xs = jnp.zeros((length - 2, 0), jnp.int8) if offload_mid else ys_mid
    (g, w_prev, d_params, pending, d_kw), _ = jax.lax.scan(
        body,
        (g, w_prev, d_params, pending, d_kw),
        (ys_xs, saved_mid, _slice_range(rest, 1, length - 1), jnp.arange(1, length - 1, dtype=jnp.int32)),
        reverse=True,
        unroll=2,
    )
  d_params = write(d_params, rs_fn(pending, "tagged"), 1, "tagged")
  if offload_saved:
    saved_0 = tuple(_with_memory_kind(s, "device") if is_off else s for s, is_off in zip(saved_0, offload_mask))
  g, dw0, d_kw0 = layer_vjp_fn(carry_in, w_prev, _slice_at(rest, 0), g, saved_0, layout)
  d_kw = tuple(acc + dk for acc, dk in zip(d_kw, d_kw0))
  d_params = write(d_params, rs_fn(dw0), 0, "all")

  params_bar_leaves = tuple(
      jnp.moveaxis(dp, 0, ax) if ax != 0 and getattr(dp, "ndim", 0) > ax else dp
      for dp, ax in zip(jax.tree_util.tree_leaves(d_params), ctx.leaf_scan_axes)
  )
  zeros_nondiff = tuple(jnp.zeros_like(x) for x, is_diff in zip(dyn_kw_leaves, ctx.diff_kw_mask) if not is_diff)
  dyn_kw_bar_leaves = _merge_diff_consts(d_kw, zeros_nondiff, ctx.diff_kw_mask)
  return g, params_bar_leaves, None, dyn_kw_bar_leaves


_custom_vjp_prefetch_pipeline.defvjp(_custom_vjp_prefetch_pipeline_fwd, _custom_vjp_prefetch_pipeline_bwd)


def _resolve_mesh(explicit_mesh: Any, params: Any) -> Any:
  """Resolves the active mesh from explicit argument, parameter shardings, or ambient abstract mesh."""
  if explicit_mesh is not None and not getattr(explicit_mesh, "empty", False):
    return explicit_mesh
  for leaf in jax.tree_util.tree_leaves(params):
    s = getattr(jax.typeof(leaf), "sharding", None) or getattr(leaf, "sharding", None)
    m = getattr(s, "mesh", None)
    if m is not None and not getattr(m, "empty", False):
      return m
  ambient = jax.sharding.get_abstract_mesh()
  if not ambient.empty:
    return ambient
  return None


def _extract_layer_specs(params: Any, param_scan_axis: int, mesh: Any, explicit: bool) -> tuple[P | None, ...]:
  """Extracts each parameter leaf's per-layer mesh PartitionSpec (with scan axis removed)."""
  specs: list[P | None] = []

  def _visit(var):
    raw = var.get_value()
    ax = min(param_scan_axis, raw.ndim - 1) if raw.ndim >= 1 else 0
    spec: P | None = None
    if explicit:
      s = getattr(jax.typeof(raw), "sharding", None) or getattr(raw, "sharding", None)
      if isinstance(s, NamedSharding) and s.spec is not None:
        entries = list(s.spec)
        entries.extend([None] * max(0, raw.ndim - len(entries)))
        if raw.ndim >= 1:
          del entries[ax]
        spec = P(*entries)
    if spec is None:
      meta_spec = var.get_metadata().get("out_sharding")
      if isinstance(meta_spec, P):
        entries = list(meta_spec)
        entries.extend([None] * max(0, raw.ndim - len(entries)))
        if raw.ndim >= 1:
          del entries[ax]
        spec = m3_sharding.project_spec_to_mesh(P(*entries), mesh, drop_explicit_unit_axes=explicit)
    if spec is not None and not any(
        fsdp_axes_for_dim(tuple(a for a in ((e,) if isinstance(e, str) else (e or ())) if mesh.shape.get(a, 1) > 1))
        for e in tuple(spec)
    ):
      spec = None
    specs.append(spec)
    return var

  jax.tree.map(_visit, params, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return tuple(specs)


def _resolve_remat_policy(remat_spec: Any) -> Callable[..., Any]:
  """Extracts a JAX checkpoint policy callable from a `Remat` wrapper or raw callable."""
  if callable(remat_spec):
    return remat_spec
  pol = getattr(remat_spec, "policy", None)
  return cp.nothing_saveable if pol is None else pol


@dataclasses.dataclass(frozen=True)
class FsdpPrefetchRunner:
  """Static, path-transparent prefetch runner attached to `ScannedLayers.fsdp_prefetch`."""

  defer_grad_reduce_scatter: bool = False
  offload_layer_inputs: bool = False
  sparse_core_id: int | None = None
  enable_overlap_transform: bool = True
  mesh: Mesh | None = None

  def __call__(self, scanned: m3_scan.ScannedLayers, carry: Any, **kwargs: Any) -> Any:
    remat_cfg = getattr(scanned, "remat", None)
    if remat_cfg is None:
      raise ValueError(
          "remat_policy='none' is not supported with prefetch_fsdp_weights=True; the FSDP weight-prefetch "
          "custom_vjp pipeline requires per-layer gradient checkpointing so gathered weights are not retained "
          "between forward and backward."
      )
    if scanned.num_layers <= 1:
      return scanned._run_regular_scan(carry, **kwargs)  # pylint: disable=protected-access

    graphdef, params, rest = nnx.split(scanned, nnx.Param, ...)
    mesh = _resolve_mesh(self.mesh, params)
    if not mesh_uses_fsdp(mesh):
      return scanned._run_regular_scan(carry, **kwargs)  # pylint: disable=protected-access

    explicit = _is_explicit_mesh(mesh)
    if self.enable_overlap_transform:
      _register_overlap_transform(mesh)
    if self.sparse_core_id is not None:
      sc_id = self.sparse_core_id
    else:
      # Only pin collectives to SparseCore 1 when `unroll=2` emits a multi-trip loop (`>= 6` layers).
      sc_id = _default_sparse_core_id(mesh) if _has_multi_trip_loop(scanned.num_layers) else None

    axis = scanned._param_scan_axis  # pylint: disable=protected-access
    layer_specs = _extract_layer_specs(params, axis, mesh, explicit)
    if not any(s is not None for s in layer_specs):
      return scanned._run_regular_scan(carry, **kwargs)  # pylint: disable=protected-access

    stripped_params = m3_scan.strip_layer_axis(params, axis)
    stacked_rest = rest
    stripped_rest = m3_scan.strip_layer_axis(rest, 0)
    params_leaves, params_treedef = jax.tree_util.tree_flatten(stripped_params)
    rest_leaves, rest_treedef = jax.tree_util.tree_flatten(stripped_rest)
    leaf_scan_axes = tuple(
        min(axis, getattr(p, "ndim", 1) - 1) if getattr(p, "ndim", 0) >= 1 else 0 for p in params_leaves
    )

    mro = type(scanned).__mro__
    layer_cls = mro[mro.index(m3_scan.ScannedLayers) + 1]
    filtered_kw = m3_scan.filter_layer_kwargs(layer_cls, kwargs)
    kw_all_leaves, kw_treedef = jax.tree_util.tree_flatten(filtered_kw)
    kw_is_array = tuple(isinstance(x, jax.Array) for x in kw_all_leaves)
    dyn_kw_leaves = tuple(x for x, is_arr in zip(kw_all_leaves, kw_is_array) if is_arr)
    static_kw_leaves = tuple(x for x, is_arr in zip(kw_all_leaves, kw_is_array) if not is_arr)
    diff_kw_mask = tuple(jax.dtypes.issubdtype(jax.typeof(x).dtype, jnp.inexact) for x in dyn_kw_leaves)
    active_axes = _sharding_axes_of_pytree((carry, dyn_kw_leaves))
    extra_batch_axes = tuple(a for a in _EXTRA_BATCH_AXES if a in active_axes and mesh.shape.get(a, 1) > 1)

    remat_policy = _resolve_remat_policy(remat_cfg)
    if self.offload_layer_inputs and _has_multi_trip_loop(scanned.num_layers):
      remat_policy = _keep_layer_inputs_on_device(remat_policy)

    layout_box: dict[Any, _ResidualLayout] = {}

    def layout_cache(key, build):
      if key not in layout_box:
        layout_box[key] = build()
      return layout_box[key]

    ctx = _PrefetchCtx(
        mesh=mesh,
        explicit=explicit,
        extra_batch_axes=extra_batch_axes,
        leaf_scan_axes=leaf_scan_axes,
        layer_specs=layer_specs,
        params_treedef=params_treedef,
        rest_treedef=rest_treedef,
        has_rest=bool(rest_leaves),
        kw_treedef=kw_treedef,
        kw_is_array=kw_is_array,
        static_kw_leaves=static_kw_leaves,
        diff_kw_mask=diff_kw_mask,
        remat_policy=remat_policy,
        prevent_cse=False,
        defer_grad_reduce_scatter=self.defer_grad_reduce_scatter,
        offload_layer_inputs=self.offload_layer_inputs,
        sparse_core_id=sc_id,
        layout_cache=layout_cache,
    )
    with jax.named_scope("fsdp_prefetch"):
      final_carry, new_rest = _custom_vjp_prefetch_pipeline(
          graphdef,
          carry,
          tuple(params_leaves),
          tuple(rest_leaves),
          tuple(dyn_kw_leaves),
          scanned.num_layers,
          ctx,
      )
    if rest_leaves:
      restored_rest = jax.tree.map(
          lambda original, updated: original.replace(updated[...]),
          stacked_rest,
          new_rest,
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      nnx.update(scanned, restored_rest)
    return final_carry


def apply_fsdp_prefetch(
    model: nnx.Module,
    *,
    defer_grad_reduce_scatter: bool = False,
    offload_layer_inputs: bool = False,
    sparse_core_id: int | None = None,
    enable_overlap_transform: bool = True,
    mesh: Mesh | None = None,
) -> nnx.Module:
  """Attaches `FsdpPrefetchRunner` to every `ScannedLayers` (or `nnx.Sequential`) in `model`."""
  runner = FsdpPrefetchRunner(
      defer_grad_reduce_scatter=defer_grad_reduce_scatter,
      offload_layer_inputs=offload_layer_inputs,
      sparse_core_id=sparse_core_id,
      enable_overlap_transform=enable_overlap_transform,
      mesh=mesh,
  )
  matched = 0
  for _, node in nnx.iter_graph(model):
    if isinstance(node, (m3_scan.ScannedLayers, nnx.Sequential)):
      node.fsdp_prefetch = runner
      matched += 1
  if matched == 0:
    max_logging.warning(
        f"apply_fsdp_prefetch found 0 ScannedLayers/Sequential containers on {type(model).__name__}; "
        "custom non-ScannedLayers loops are left unchanged."
    )
  return model


def maybe_apply_fsdp_prefetch(
    model: nnx.Module,
    config: Any,
    *,
    mesh: Any = None,
    model_mode: str = _TRAIN_MODE,
) -> nnx.Module:
  """Applies FSDP weight prefetching when `config.prefetch_fsdp_weights` is enabled for training."""
  if model_mode != _TRAIN_MODE:
    return model
  if not getattr(config, "prefetch_fsdp_weights", False):
    return model
  if not getattr(config, "scan_layers", True):
    max_logging.warning("prefetch_fsdp_weights=True has no effect with scan_layers=False; requires scan_layers=True.")
    return model
  if getattr(config, "parameter_memory_host_offload", False):
    return model
  return apply_fsdp_prefetch(
      model,
      defer_grad_reduce_scatter=bool(getattr(config, "prefetch_fsdp_defer_grad_reduce_scatter", False)),
      offload_layer_inputs=getattr(config, "decoder_layer_input", "device") == "offload",
      mesh=mesh if isinstance(mesh, Mesh) else None,
  )
