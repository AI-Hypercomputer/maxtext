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
  gathered_sharding = sharding.remove_mesh_axes_from_sharding(src_sharding, sharding.FSDP_MESH_AXES)
  out = _explicit_fsdp_all_gather(w, mesh, src_sharding.spec, gathered_sharding.spec)
  if out is not None:
    return out
  return sharding.maybe_shard_with_name(w, gathered_sharding, shard_mode=shard_mode, debug_sharding=debug_sharding)


def _explicit_fsdp_all_gather(w, mesh, src_spec, gathered_spec):
  """All-gathers `w` over its FSDP mesh axes with an explicit all_gather tagged `xla_explicit_fsdp`.

  On TPUs that offload all-gathers to the SparseCore, the tag lets XLA's FSDP scheduler
  (--xla_tpu_enable_fsdp_latency_hiding_scheduler with --xla_tpu_explicit_fsdp_dedicated_sparse_core_id)
  run these gathers on a dedicated SparseCore. Otherwise they share the SparseCore queue with the
  layer's own SparseCore work (e.g. MoE token gathers) and must finish before that work can.

  Returns None (the caller falls back to resharding) if the FSDP axes of a dimension are not its
  minor-most mesh axes, since a tiled gather would then not produce the target layout.
  """
  spec = tuple(src_spec) + (None,) * (w.ndim - len(tuple(src_spec)))
  dims = []
  for i, entry in enumerate(spec):
    axes = () if entry is None else (tuple(entry) if isinstance(entry, (tuple, list)) else (entry,))
    axes = tuple(a for a in axes if mesh.shape[a] > 1)
    fsdp = tuple(a for a in axes if a in sharding.FSDP_MESH_AXES)
    if not fsdp:
      continue
    if axes[len(axes) - len(fsdp) :] != fsdp:
      return None
    dims.append((i, fsdp))
  if not dims:
    return None

  def gather(x):
    for i, ax in dims:
      x = jax.lax.all_gather(x, ax if len(ax) > 1 else ax[0], axis=i, tiled=True)
    return x

  with xla_metadata.set_xla_metadata(xla_explicit_fsdp="yes"):
    return jax.shard_map(
        gather, mesh=mesh, in_specs=jax.sharding.PartitionSpec(*spec), out_specs=gathered_spec, check_vma=False
    )(w)


def _per_layer_sharding(stacked, mesh):
  """Sharding of one layer slice `stacked[i]` of a scanned (layer-major) parameter, or None."""
  s = getattr(jax.typeof(stacked), "sharding", None) or getattr(stacked, "sharding", None)
  if not isinstance(s, NamedSharding) or s.spec is None:
    return None
  return NamedSharding(mesh, jax.sharding.PartitionSpec(*tuple(s.spec)[1:]))


_PrefetchCtx = collections.namedtuple(
    "_PrefetchCtx",
    [
        "mesh",
        "shard_mode",
        "debug_sharding",
        "static_args_leaves",
        "static_kwargs_leaves",
        "params_treedef",
        "state_treedef",
        "args_treedef",
        "kwargs_treedef",
    ],
)


def _prefetch_setup(graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx):
  """Rebuilds params/state/layer arguments and returns (params, state, gather_fn, layer_fn, rs_fn)."""
  params = jax.tree_util.tree_unflatten(ctx.params_treedef, params_leaves)
  state = jax.tree_util.tree_unflatten(ctx.state_treedef, state_leaves)
  args_leaves = [s if s is not None else d for d, s in zip(diff_args_leaves, ctx.static_args_leaves)]
  args_tuple = jax.tree_util.tree_unflatten(ctx.args_treedef, args_leaves)
  kwargs_leaves = [s if s is not None else d for d, s in zip(diff_kwargs_leaves, ctx.static_kwargs_leaves)]
  kwargs = dict(jax.tree_util.tree_unflatten(ctx.kwargs_treedef, kwargs_leaves))
  layer_shardings = [_per_layer_sharding(p, ctx.mesh) for p in params_leaves]

  def gather_one(x):
    x = _apply_sharding_hint(x, ctx.mesh, ctx.shard_mode, ctx.debug_sharding)
    if getattr(x, "ndim", 0) >= 2:
      # Pin the gathered weight to the default layout. Otherwise XLA may pick a different layout for
      # the gathered buffer that rides in the loop carry and then relayout it with a full-size copy
      # on the TensorCore before its consumer; pinned, any relayout happens on the gather output.
      x = with_layout_constraint(x, Layout(tuple(range(x.ndim))))
    return x

  def gather_fn(layer_params):
    """All-gathers one layer's FSDP-sharded params (bf16[E, D/fsdp, F] -> bf16[E, D, F])."""
    return jax.tree.map(gather_one, layer_params)

  def layer_fn(y, layer_weights, layer_state):
    # Per-layer slices still carry the scan axis in their sharding metadata; drop it so the
    # merged layer sees the same metadata as in the regular scan path.
    layer_weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((layer_weights, layer_state), "layers")
    layer = nnx.merge(graphdef, layer_weights, layer_state)
    out = layer(y, *args_tuple, **kwargs)
    return out[0] if isinstance(out, tuple) else out

  def rs_fn(layer_grads):
    """Reduce-scatters one layer's gathered-weight gradients back to the FSDP parameter sharding."""
    leaves, treedef = jax.tree_util.tree_flatten(layer_grads)
    out = [
        g if s is None else sharding.maybe_shard_with_name(g, s, ctx.shard_mode, debug_sharding=ctx.debug_sharding)
        for g, s in zip(leaves, layer_shardings)
    ]
    return jax.tree_util.tree_unflatten(treedef, out)

  return params, state, gather_fn, layer_fn, rs_fn


def _take(tree, idx):
  """Layer `idx` (a Python int or a traced scalar) of a layer-stacked pytree."""
  return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, idx, 0, keepdims=False), tree)


def _expand0(tree):
  return jax.tree.map(lambda x: jnp.expand_dims(x, 0), tree)


def _after(deps, x):
  """Returns `x`, ordered after `deps` have been computed (via an optimization barrier)."""
  _, x = jax.lax.optimization_barrier((deps, x))
  return x


def _interleave(even, odd):
  """[P, ...] stacks of layers 0, 2, .., 2P-2 and 1, 3, .., 2P-1 -> one [2P, ...] stack in layer order."""
  return jax.tree.map(lambda e, o: jnp.stack([e, o], axis=1).reshape((2 * e.shape[0],) + e.shape[1:]), even, odd)


def _prefetch_fwd_split(length):
  """Returns (s, num_pairs): the forward runs layers 0 .. s-1 before its pair loop (s = 0 or 1), so
  the loop covers an even number of layers s .. length-2 and the last layer runs after it."""
  s = (length - 1) % 2
  return s, (length - 1 - s) // 2


def _prefetch_can_order(num_pairs):
  """Whether a pair loop may order its second gather after the loop's first layer (see `_after`).

  XLA's FSDP scheduler (for gathers tagged `xla_explicit_fsdp`) requires that no tagged gather
  depends on another tagged gather's result within one computation; only while loops separate them.
  Inside a loop body the dependency runs through the loop-carried weights, but XLA removes one-trip
  loops (and unrolling would put two iterations in one body), so only multi-trip, non-unrolled loops
  order.
  """
  return num_pairs >= 2


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 6, 7))
def _custom_vjp_prefetch_pipeline(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  """Runs `length` scanned layers with FSDP weight prefetching in both the forward and backward pass.

  Forward (layer k):   AG(k+1) || compute(k). Only each layer's input activation is saved.
  Backward (layer k):  AG(k-1) || [recompute(k) + vjp(k)], then reduce-scatter dW(k).

  Gathered weights are never saved between forward and backward. The backward always rematerializes
  each layer in full (the decoder remat policy does not apply to this path).

  Both passes scan over pairs of layers and use two gathered-weight buffers in turn (ping-pong): the
  buffer carried into an iteration is only refilled (by the prefetch two layers ahead) once the layer
  reading it is done. With a single carried buffer, the prefetch of W(k+1) would overwrite the carry
  slot while W(k) is still being read, and XLA would add a full-size copy of every gathered layer to
  double-buffer it. So the forward holds at most two layers of gathered weights (current +
  prefetched), and the backward at most three (W(k), W(k-1) and dW(k) until its reduce-scatter).
  """
  out, _ = _custom_vjp_prefetch_pipeline_fwd(
      graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
  )
  return out


def _custom_vjp_prefetch_pipeline_fwd(
    graphdef, x_in, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, length, ctx
):
  """Forward pass. Scan iteration j runs layers k = s+2j and k+1 (s = 1 if layer 0 runs before the loop).

  W(k) arrives in the loop carry. AG(k+1) is issued at the start of the iteration (overlapping layer
  k) into an iteration-local buffer, and AG(k+2) once layer k is done (overlapping layer k+1) into the
  carry slot W(k) has just freed. The last layer runs after the loop on the carried weights.
  """
  params, state, gather_fn, layer_fn, _ = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )
  s, num_pairs = _prefetch_fwd_split(length)
  y = x_in
  w = gather_fn(_take(params, 0))
  ys_head = []  # Inputs of the layers before the loop (residuals for the backward pass).
  if s:
    w_next = gather_fn(_take(params, 1))  # AG(1) || layer 0.
    ys_head.append(y)
    y = layer_fn(y, w, _take(state, 0))
    w = w_next
  ys_even = ys_odd = None  # Inputs of the loop's layers s, s+2, .. / s+1, s+3, ..
  if num_pairs:
    ordered = _prefetch_can_order(num_pairs)

    def body(carry, j):
      y, w_a = carry
      k = s + 2 * j
      w_b = gather_fn(_take(params, k + 1))  # AG(k+1) || layer k.
      y_a = y
      y = layer_fn(y, w_a, _take(state, k))
      # AG(k+2) waits for layer k's output, so for every read of W(k), and can reuse W(k)'s buffer.
      p = _take(params, k + 2)
      w_a = gather_fn(_after(y, p) if ordered else p)  # AG(k+2) || layer k+1.
      y_b = y
      y = layer_fn(y, w_b, _take(state, k + 1))
      return (y, w_a), (y_a, y_b)

    (y, w), (ys_even, ys_odd) = jax.lax.scan(body, (y, w), jnp.arange(num_pairs))
  y_last = y
  y = layer_fn(y, w, _take(state, length - 1))
  residuals = (
      (tuple(ys_head), ys_even, ys_odd, y_last),
      params_leaves,
      state_leaves,
      diff_args_leaves,
      diff_kwargs_leaves,
  )
  return (y, state), residuals


def _custom_vjp_prefetch_pipeline_bwd(graphdef, length, ctx, res, g_out):
  """Backward pass, mirroring the forward's ping-pong buffers.

  Layer L-1 (and layer L-2 if needed to leave an even number of loop layers) runs before the loop.
  Scan iteration j runs layers hi = 2j+2 and hi-1 (in that order): W(hi) arrives in the loop carry,
  AG(hi-1) is issued at the start of the iteration (overlapping bwd(hi)), and AG(hi-2) once bwd(hi) is
  done (overlapping bwd(hi-1)) into the carry slot W(hi) has just freed. Layer 0 runs after the loop
  on the carried weights. Each layer's dW is reduce-scattered after its vjp.
  """
  (ys_head, ys_even, ys_odd, y_last), params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves = res
  g, _ = g_out  # Non-parameter layer state is not differentiated.
  params, state, gather_fn, layer_fn, rs_fn = _prefetch_setup(
      graphdef, params_leaves, state_leaves, diff_args_leaves, diff_kwargs_leaves, ctx
  )
  s, _ = _prefetch_fwd_split(length)

  def layer_input(k, odd=None):
    """Saved input of layer k. For a traced k, `odd` says whether k-s is odd (static)."""
    if isinstance(k, int):
      if k == length - 1:
        return y_last
      if k < s:
        return ys_head[k]
      odd = (k - s) % 2 == 1
    return _take(ys_odd if odd else ys_even, (k - s) // 2)

  def layer_vjp(k, w, g_y, odd=None, y_in=None):
    layer_state = _take(state, k)
    y_in = layer_input(k, odd) if y_in is None else y_in
    _, vjp_fn = jax.vjp(lambda y_, w_: layer_fn(y_, w_, layer_state), y_in, w)
    return vjp_fn(g_y)  # (dy_in, dW gathered)

  # The param slices depend only on the params, so XLA would issue the first backward gathers at the
  # start of the step and keep those gathered layers alive through the whole forward pass. Tying the
  # (sharded) params to the incoming cotangent delays the backward gathers until the backward starts.
  g, params = jax.lax.optimization_barrier((g, params))

  grads = {}  # Layer -> reduce-scattered dW, for the layers outside the loop.
  last = length - 1
  w = gather_fn(_take(params, last))
  w_next = gather_fn(_take(params, last - 1)) if length > 1 else None  # AG(L-2) || bwd(L-1).
  g, dw = layer_vjp(last, w, g)
  grads[last] = rs_fn(dw)
  w = w_next
  top = 0  # Highest loop layer (the loop runs layers top .. 1).
  d_lo = d_hi = None  # Reduce-scattered dW of the loop's layers 1, 3, .. / 2, 4, .. (stacked by pair).
  if length > 1:
    peel = (length - 2) % 2  # Layer L-2 runs before the loop, leaving an even number of loop layers.
    if peel:
      k = length - 2
      w_next = gather_fn(_take(params, k - 1))  # AG(L-3) || bwd(L-2).
      g, dw = layer_vjp(k, w, g)
      grads[k] = rs_fn(dw)
      w = w_next
    top = length - 2 - peel
    num_pairs = top // 2
    if num_pairs:
      ordered = _prefetch_can_order(num_pairs)

      def body(carry, j):
        g, w_a = carry
        hi = 2 * j + 2
        w_b = gather_fn(_take(params, hi - 1))  # AG(hi-1) || bwd(hi).
        g, dw_a = layer_vjp(hi, w_a, g, odd=s == 1)
        p, y_b = _take(params, hi - 2), layer_input(hi - 1, odd=s == 0)
        if ordered:
          # Layer hi-1 (including its recompute) and AG(hi-2) wait for all of bwd(hi), dW included.
          # So W(hi) is dead when AG(hi-2) starts (it can reuse W(hi)'s buffer), and XLA does not
          # interleave the two layers, which would keep both layers' full-size dW and activations live.
          g, dw_a, y_b, p = jax.lax.optimization_barrier((g, dw_a, y_b, p))
        w_a = gather_fn(p)  # AG(hi-2) || bwd(hi-1).
        g, dw_b = layer_vjp(hi - 1, w_b, g, y_in=y_b)
        return (g, w_a), (rs_fn(dw_b), rs_fn(dw_a))

      # Iterates j = num_pairs-1 .. 0; outputs are stacked in j order.
      (g, w), (d_lo, d_hi) = jax.lax.scan(body, (g, w), jnp.arange(num_pairs), reverse=True)
    g, dw = layer_vjp(0, w, g)
    grads[0] = rs_fn(dw)

  # Per-layer grads stacked in layer order: [dW0, dW1 .. dW(top) (loop), dW(top+1) .. dW(L-1)].
  parts = [_expand0(grads[0])]
  if top:
    parts.append(_interleave(d_lo, d_hi))
  parts += [_expand0(grads[k]) for k in range(top + 1, length)]
  d_params = parts[0] if len(parts) == 1 else jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)
  params_bar_leaves = tuple(jax.tree_util.tree_leaves(d_params))

  def _zero_tangent(x):
    if isinstance(x, jax.Array) or (hasattr(jax.core, "Tracer") and isinstance(x, jax.core.Tracer)):
      return jnp.zeros_like(x)
    return None

  diff_args_bar_leaves = tuple(_zero_tangent(x) for x in diff_args_leaves)
  diff_kwargs_bar_leaves = tuple(_zero_tangent(x) for x in diff_kwargs_leaves)
  return g, params_bar_leaves, None, diff_args_bar_leaves, diff_kwargs_bar_leaves


_custom_vjp_prefetch_pipeline.defvjp(_custom_vjp_prefetch_pipeline_fwd, _custom_vjp_prefetch_pipeline_bwd)


def apply_layers_with_fsdp_prefetch(layers, x_in, *args, length: int, mesh, config, **kwargs):
  """Runs a scanned NNX layer stack with FSDP weight prefetching in the forward and backward pass.

  Args:
    layers: The scanned (layer-stacked) NNX module.
    x_in: The input activation of the first layer.
    *args: Positional arguments passed to every layer call.
    length: Number of layers in the stack.
    mesh: The device mesh.
    config: The MaxText config.
    **kwargs: Keyword arguments passed to every layer call (those not accepted by the layer are dropped).

  Returns:
    (final activation, layers, None). Non-parameter layer state is read-only in this path: state
    mutated inside a layer call (e.g. sown intermediates) is not propagated back to `layers`.

  See `_custom_vjp_prefetch_pipeline`. The backward pass always rematerializes each layer in full,
  so `remat_policy` does not apply here; only each layer's input activation is saved.
  """
  if length == 0:
    return x_in, layers, None
  if config.remat_policy not in ("full", "none"):
    max_logging.info(
        f"prefetch_fsdp_weights: remat_policy={config.remat_policy} is ignored; the prefetch"
        " pipeline saves only layer inputs and rematerializes each layer in full in the backward pass."
    )
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)

  scan_axis = config.param_scan_axis
  if scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) if x.ndim > scan_axis else x, params)
  params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
  state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)

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
      static_args_leaves=static_args_leaves,
      static_kwargs_leaves=static_kwargs_leaves,
      params_treedef=params_treedef,
      state_treedef=state_treedef,
      args_treedef=args_treedef,
      kwargs_treedef=kwargs_treedef,
  )

  final_y, _ = _custom_vjp_prefetch_pipeline(
      graphdef, x_in, tuple(params_leaves), tuple(state_leaves), diff_args_leaves, diff_kwargs_leaves, length, ctx
  )
  return final_y, layers, None
