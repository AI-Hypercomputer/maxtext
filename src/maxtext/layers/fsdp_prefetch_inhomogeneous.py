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

"""FSDP weight prefetching, one sub-layer at a time, for inhomogeneous scanned blocks (e.g. Qwen3.5).

`fsdp_prefetch` prefetches whole scan steps. When one scan step is a block of sub-layers with
different parameter structures (e.g. Qwen3.5's mix of Gated DeltaNet and global attention layers),
that gathers a whole block's weights at once. A block that defines `prefetch_sublayer_names()` is
instead prefetched one sub-layer at a time here: the all-gather of sub-layer j+1 overlaps sub-layer
j, across block boundaries, in both passes, and each sub-layer runs in a while-loop nested in the
block scan, as in the block's regular nested scan. The gathers, reduce-scatters and remat policy
handling are shared with `fsdp_prefetch`.

See `_custom_vjp_nested_prefetch_pipeline` for the design.
"""

# pylint: disable=protected-access

import functools

from flax import nnx
import jax
from jax.experimental import xla_metadata
import jax.numpy as jnp
from maxtext.layers import fsdp_prefetch
from maxtext.utils import max_logging


def _index_at(tree, idx):
  """Layer `idx` (a traced index) of a layer-stacked pytree."""
  return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, idx, axis=0, keepdims=False), tree)


_SKIP_TRIP_COUNT_ONE = {"skip-simplify-while-loops_trip-count-one": "true"}


def _sublayer_scan(body, carry, xs, reverse=False):
  """`jax.lax.scan` over sub-layers that XLA keeps as a while-loop, even with a single trip.

  Each sub-layer run is its own while-loop nested in the block scan, as in the regular nested scan
  of the block (see `Qwen3NextScannableBlock`). The loop boundary keeps a prefetched all-gather
  inside the sub-layer that issues it: XLA cannot finish it early, ahead of that sub-layer's compute.
  """
  with xla_metadata.set_xla_metadata(**_SKIP_TRIP_COUNT_ONE):
    return jax.lax.scan(body, carry, xs, reverse=reverse)


def _concat0(*trees):
  """Concatenates layer-stacked pytrees along the leading axis."""
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def _setup_groups(graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs):
  """`_prefetch_setup` for each sub-layer group; returns per-group tuples (gather_fn, rs_fn, ...).

  Group params/state leaves are stacked `[length, n, ...]`; the setup sees sub-layer 0's
  `[length, ...]` slice, which has the per-sub-layer shardings.
  """
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7, 8))
def _custom_vjp_nested_prefetch_pipeline(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  """Runs `length` scanned blocks of sub-layer groups with per-sub-layer prefetching and nested scans.

  Used for inhomogeneous stacks (e.g. Qwen3.5), where one scan step is a block of sub-layers with
  different parameter structures. A block is a sequence of groups; group `i` has `sizes[i]`
  consecutive sub-layers of one structure (e.g. Qwen3.5: 3 Gated DeltaNet layers, then 1 global
  attention layer). The pipeline walks the flat sequence of sub-layers:

  Forward (sub-layer j):   AG(j+1) || compute(j), where AG(j+1) of the last sub-layer of a block is
                           sub-layer 0 of the next block.
  Backward (sub-layer j):  AG(j-1) || [recompute(j) + vjp(j)], then reduce-scatter dW(j).

  Each pass is a scan over the blocks whose body runs every sub-layer in a while-loop nested in it
  (see `_sublayer_scan`). Within a group of n sub-layers, the n-1 that prefetch a sub-layer of the
  same group (the first n-1 forward, the last n-1 backward) share one scan; the remaining one
  prefetches a sub-layer of another structure and runs in its own single-trip loop. Qwen3.5 forward:
  [GDN 0, GDN 1] -> [GDN 2] -> [attention]; backward: [attention] -> [GDN 2, GDN 1] -> [GDN 0].

  Group `i` has its own graphdef, `params_leaves[i]` and `state_leaves[i]`, each stacked
  `[length, sizes[i], ...]`. The forward pass holds at most two sub-layers of gathered weights and
  the backward pass at most three, as in `fsdp_prefetch._custom_vjp_prefetch_pipeline`. All
  sub-layer gathers but the first stay inside the block loop; the cost is one unused gather of a
  sub-layer at the end of each pass.
  """
  out, _ = _custom_vjp_nested_prefetch_pipeline_fwd(
      graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
  )
  return out


def _custom_vjp_nested_prefetch_pipeline_fwd(
    graphdefs, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctxs, sizes
):
  """Forward pass of `_custom_vjp_nested_prefetch_pipeline`: one scan step per block."""
  num_groups = len(graphdefs)
  params, state, gather, _, layout_fns, fwd_layers, _ = _setup_groups(
      graphdefs, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctxs
  )
  layouts = {}

  def fwd(i, y, w, layer_state):
    if i not in layouts:
      layouts[i] = layout_fns[i](y, w, layer_state)
    return fwd_layers[i](y, w, layer_state, layouts[i])

  def body(carry, xs):
    y, w = carry  # `w` is this block's gathered sub-layer 0.
    b, b_state, b_params = xs  # Group i's leaves are `[sizes[i], ...]`.
    ys, saved = [], []
    for i in range(num_groups):
      n = sizes[i]
      if i + 1 < num_groups:
        nxt_first, gather_nxt = fsdp_prefetch._slice_at(b_params[i + 1], 0), gather[i + 1]
      else:
        # The last block prefetches block 0's sub-layer 0, which is unused: every gather then stays
        # in the block loop instead of a peeled epilogue in the parent computation.
        nxt_first = fsdp_prefetch._slice_at(_index_at(params[0], (b + 1) % length), 0)
        gather_nxt = gather[0]
      parts_y, parts_saved = [], []
      if n >= 2:

        def same_group(carry, k, i=i, p=b_params[i], s=b_state[i]):
          y, w = carry
          w_next = gather[i](_index_at(p, k + 1))  # AG(j+1) || sub-layer j.
          y_out, saved_k = fwd(i, y, w, _index_at(s, k))
          return (y_out, w_next), (y, saved_k)

        (y, w), (ys_k, saved_k) = _sublayer_scan(same_group, (y, w), jnp.arange(n - 1))
        parts_y.append(ys_k)
        parts_saved.append(saved_k)

      def last(carry, xs_last, i=i, gather_nxt=gather_nxt):
        y, w = carry
        layer_state, nxt = xs_last
        w_next = gather_nxt(nxt)  # AG(j+1) || sub-layer j, where sub-layer j+1 is of another group.
        y_out, saved_last = fwd(i, y, w, layer_state)
        return (y_out, w), (w_next, saved_last)

      # The last sub-layer's state and next params ride in xs, so XLA cannot hoist them as loop invariants.
      last_xs = fsdp_prefetch._expand0((fsdp_prefetch._slice_at(b_state[i], n - 1), nxt_first))
      y_last_in = y
      (y, _), (w_next, saved_last) = _sublayer_scan(last, (y, w), last_xs)
      w = fsdp_prefetch._slice_at(w_next, 0)
      parts_y.append(fsdp_prefetch._expand0(y_last_in))
      parts_saved.append(saved_last)
      ys.append(_concat0(*parts_y))
      saved.append(_concat0(*parts_saved))
    return (y, w), (tuple(ys), tuple(saved))

  w = gather[0](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[0], 0), 0))
  xs = (jnp.arange(length), tuple(state), tuple(params))
  (y, _), (ys, saved) = jax.lax.scan(body, (x_in, w), xs)
  residuals = (ys, saved, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves)
  return (y, state_leaves), residuals


def _custom_vjp_nested_prefetch_pipeline_bwd(graphdefs, length, ctxs, sizes, res, g_out):
  """Backward pass of `_custom_vjp_nested_prefetch_pipeline`: one reverse scan step per block."""
  ys, saved, params_leaves, state_leaves, d_args, d_kwargs = res
  g, _ = g_out  # Non-parameter layer state is not differentiated.
  num_groups = len(graphdefs)
  params, state, gather, rs_fns, layout_fns, _, layer_vjp_fns = _setup_groups(
      graphdefs, params_leaves, state_leaves, d_args, d_kwargs, ctxs
  )
  # Tie `params` to `g` so XLA cannot CSE the backward all-gathers with the forward ones or hoist
  # backward weight slices before the forward pass.
  g, params = jax.lax.optimization_barrier((g, tuple(params)))
  layouts = {}

  def bwd(i, y, w, layer_state, g, saved_k):
    if i not in layouts:
      layouts[i] = layout_fns[i](y, w, layer_state)
    g, dw = layer_vjp_fns[i](y, w, layer_state, g, saved_k, layouts[i])
    return g, rs_fns[i](dw)

  def body(carry, xs):
    g, w = carry  # `w` is this block's gathered last sub-layer.
    b, b_ys, b_saved, b_state, b_params = xs  # Group i's leaves are `[sizes[i], ...]`.
    dws = [None] * num_groups
    for i in reversed(range(num_groups)):
      n = sizes[i]
      if i > 0:
        prv_last, gather_prv = fsdp_prefetch._slice_at(b_params[i - 1], sizes[i - 1] - 1), gather[i - 1]
      else:
        # Block 0 prefetches the last block's last sub-layer, which is unused (see the forward pass).
        prv_last = fsdp_prefetch._slice_at(_index_at(params[-1], (b - 1) % length), sizes[-1] - 1)
        gather_prv = gather[-1]
      d_rest = None
      if n >= 2:

        def same_group(carry, k, i=i, p=b_params[i], s=b_state[i], y_=b_ys[i], sv=b_saved[i]):
          g, w = carry
          w_prev = gather[i](_index_at(p, k - 1))  # AG(j-1) || bwd(j).
          g, dw = bwd(i, _index_at(y_, k), w, _index_at(s, k), g, _index_at(sv, k))
          return (g, w_prev), dw

        # Iterates k = n-1 .. 1; outputs are stacked in k order.
        (g, w), d_rest = _sublayer_scan(same_group, (g, w), jnp.arange(1, n), reverse=True)

      def first(carry, xs_first, i=i, gather_prv=gather_prv):
        g, w = carry
        y0, saved0, layer_state, prv = xs_first
        w_prev = gather_prv(prv)  # AG(j-1) || bwd(j), where sub-layer j-1 is of another group.
        g, dw = bwd(i, y0, w, layer_state, g, saved0)
        return (g, w), (w_prev, dw)

      # The first sub-layer's inputs ride in xs, so XLA cannot hoist its recompute as a loop invariant.
      first_xs = fsdp_prefetch._expand0(
          tuple(fsdp_prefetch._slice_at(t, 0) for t in (b_ys[i], b_saved[i], b_state[i])) + (prv_last,)
      )
      (g, _), (w_prev, dw0) = _sublayer_scan(first, (g, w), first_xs)
      w = fsdp_prefetch._slice_at(w_prev, 0)
      dws[i] = dw0 if d_rest is None else _concat0(dw0, d_rest)
    return (g, w), tuple(dws)

  w = gather[-1](fsdp_prefetch._slice_at(fsdp_prefetch._slice_at(params[-1], length - 1), sizes[-1] - 1))
  xs = (jnp.arange(length), ys, saved, tuple(state), tuple(params))
  (g, _), d_params = jax.lax.scan(body, (g, w), xs, reverse=True)

  # `d_params[i]` holds group i's grads stacked `[length, sizes[i], ...]`.
  params_bar_leaves = tuple(tuple(jax.tree_util.tree_leaves(d)) for d in d_params)
  diff_args_bar_leaves = tuple(jnp.zeros_like(x) for x in d_args)
  diff_kwargs_bar_leaves = tuple(jnp.zeros_like(x) for x in d_kwargs)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_nested_prefetch_pipeline.defvjp(
    _custom_vjp_nested_prefetch_pipeline_fwd, _custom_vjp_nested_prefetch_pipeline_bwd
)


def _prefetch_sublayer_names(layers):
  """Names of the sub-layers to prefetch one at a time, or None to prefetch whole scan steps.

  A block with named sub-layer attributes opts in by defining `prefetch_sublayer_names()`, returning
  its sub-layer attribute names in call order. Each sub-layer must accept the block's call arguments
  and return `(x, ...)`.
  """
  names_fn = getattr(layers, "prefetch_sublayer_names", None)
  if names_fn is None:
    return None
  names = tuple(names_fn())
  return names if len(names) >= 2 else None


def _unstack_variable(var, num_local: int, var_scan_axis: int):
  """Splits one variable of an inner-scanned sub-layer module into its `num_local` per-sub-layer slices.

  The `local_layers` axis is located from the variable's scan metadata and is dropped from its sharding
  metadata. Leaves that are not `nnx.Variable`s, and variables without a `local_layers` axis of size
  `num_local` (e.g. a scalar shared by every sub-layer), are passed to every sub-layer unchanged.
  """
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
    # Only stacked by the inner scan (the block scan re-labels the variables it stacks as "layers").
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
  """Unstacks an inner-scanned sub-layer module (`local_layers`) into `num_local` `(graphdef, params, state)` stages."""
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
  """`nnx.split(sublayer, nnx.Param, ...)` of each sub-layer to prefetch, or None.

  None means the block should prefetch whole scan steps (`fsdp_prefetch`): it does not opt in (via
  `local_layers` + `global_layer` or `_prefetch_sublayer_names`) or it has params outside those
  sub-layers.
  """
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
    max_logging.info("prefetch_fsdp_weights: the block has params outside its sub-layers; prefetching whole blocks.")
    return None
  return stages


def apply_layers_with_fsdp_prefetch(
    layers, stages, x_in, *args, length: int, mesh, config, remat_policy=None, prevent_cse: bool = False, **kwargs
):
  """Runs a scanned inhomogeneous block stack, prefetching FSDP weights one sub-layer at a time.

  Args:
    layers: The scanned (block-stacked) NNX module.
    stages: `sublayer_stages(layers)` (not None).
    x_in: The input activation of the first block.
    *args: Positional arguments passed to every sub-layer call (the block's arguments).
    length: Number of blocks in the stack.
    mesh: The device mesh.
    config: The MaxText config.
    remat_policy: The `jax.checkpoint` policy of each sub-layer (None for full remat).
    prevent_cse: `prevent_cse` of the per-sub-layer `jax.checkpoint` (only used with a remat policy).
    **kwargs: Keyword arguments passed to every sub-layer call (those not accepted by the block are dropped).

  Returns:
    (final activation, layers, None), as `fsdp_prefetch.apply_layers_with_fsdp_prefetch`.
  """
  if length == 0:
    return x_in, layers, None
  _, _, _, diff_args_leaves, diff_kwargs_leaves, ctx = fsdp_prefetch._prepare_prefetch_call(
      layers, args, kwargs, length=length, mesh=mesh, config=config, remat_policy=remat_policy, prevent_cse=prevent_cse
  )
  # Groups consecutive sub-layers of one structure (e.g. the unstacked `local_layers`), stacked
  # `[length, n, ...]`, so each group can be scanned by sub-layer.
  groups = []  # [graphdef, params_treedef, state_treedef, [params leaves per sub-layer], [state leaves ...]]
  for stage_graphdef, p, s in stages:
    p, s = fsdp_prefetch._layer_major(p, s, config.param_scan_axis, length)
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
        continue
    groups.append([stage_graphdef, p_treedef, s_treedef, [p_leaves], [s_leaves]])

  def stack(per_sublayer):
    return tuple(jnp.stack(xs, axis=1) for xs in zip(*per_sublayer))

  final_y, _ = _custom_vjp_nested_prefetch_pipeline(
      tuple(g[0] for g in groups),
      x_in,
      tuple(stack(g[3]) for g in groups),
      tuple(stack(g[4]) for g in groups),
      diff_args_leaves,
      diff_kwargs_leaves,
      length,
      tuple(ctx._replace(params_treedef=g[1], state_treedef=g[2]) for g in groups),
      tuple(len(g[3]) for g in groups),
  )
  return final_y, layers, None
