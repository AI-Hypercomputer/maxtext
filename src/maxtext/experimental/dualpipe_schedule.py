# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-JAX layer-level 1F1B accumulation with the real training loss.

The caller owns normalization and the optimizer update and jits the enclosing
training step. This module sums loss, metrics, and gradients over microbatches.
"""

import contextlib
import itertools

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np


def _reduce_aux(stacked_aux):
  """Match normal GA: sum loss metrics, but OR/max/min TE capacity metrics."""
  reducers = {
      "te_moe_capacity_overflow": jnp.any,
      "te_moe_max_total_recv_tokens": jnp.max,
      "te_moe_recv_capacity_per_rank": jnp.min,
  }
  return {key: reducers.get(key, jnp.sum)(value, axis=0) for key, value in stacked_aux.items()}


class _ResidualLayout:
  """Static description of one layer's VJP residuals.

  A layer's pullback (from jax.vjp) closes over its inputs (weights, hidden,
  state, positions, segments) plus the activations its remat policy saves. Only
  the saved activations, and the hidden input if the pullback uses it, differ
  per microbatch; weights and state are re-read from the model at backward time,
  so they are never copied into saved residuals.
  """

  def __init__(self, treedef, sources, saved_avals, kept, leaf_avals, needs_hidden):
    self.treedef = treedef
    self.sources = sources  # per pullback leaf: input index, or None for a saved activation
    self.saved_avals = saved_avals
    self.kept = kept  # per pullback leaf: True if it is stored in the saved tuple
    self.leaf_avals = leaf_avals  # per pullback leaf: (shape, dtype), to rebuild pruned leaves
    self.needs_hidden = needs_hidden

  def matches(self, other):
    return (
        self.sources == other.sources
        and self.saved_avals == other.saved_avals
        and self.kept == other.kept
        and self.needs_hidden == other.needs_hidden
    )


def _used_pullback_leaves(treedef, leaves, out):
  """Which pullback leaves the backward pass reads, by abstractly tracing it and removing dead code.

  jax.vjp's pullback closes over every residual the remat policy marked saveable, even ones the
  backward never reads (for example the output of a layer whose gradient does not need it). The
  scan path drops those during partial evaluation; here each one would be stored per layer and
  per microbatch, so we find and drop them. Falls back to keeping everything if analysis fails.
  """
  try:
    # Private JAX module: imported lazily so a JAX change degrades to "keep every residual", not an import error.
    from jax._src.interpreters import partial_eval as pe  # pylint: disable=import-outside-toplevel

    avals = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in leaves]
    closed = jax.make_jaxpr(lambda flat, dy: jax.tree.unflatten(treedef, flat)(dy))(
        avals, jax.ShapeDtypeStruct(out.shape, out.dtype)
    )
    outvars = closed.jaxpr.outvars
    _, used_inputs = pe.dce_jaxpr(closed.jaxpr, [True] * len(outvars))
    return tuple(bool(u) for u in used_inputs[: len(leaves)])
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("dual_pipe: could not analyze pullback residual use (%r); keeping all residuals", e)
    return (True,) * len(leaves)


def _aval_key(x):
  aval = jax.typeof(x)
  return (aval.shape, aval.dtype, getattr(aval, "memory_space", None))


def _layer_passes(apply):
  """Forward/backward for one layer that save only the residuals its backward reads.

  apply may be wrapped in jax.checkpoint (the saved residuals are then what its
  remat policy keeps) or not (every residual the backward reads is saved).

  forward(weights, hidden, state, positions, segments) -> (out, saved, metrics),
  where saved holds exactly the activations the layer's remat policy keeps (on
  device or offloaded), and the hidden input only if the backward needs it.
  backward(weights, state, positions, segments, saved, dy) -> (dweights, dhidden)
  rebuilds the pullback from saved plus the model's weights/state. The layout
  is recorded when forward is traced and must be identical for every trace.

  How it works: jax.vjp returns the pullback as a pytree. Its treedef holds the
  backward jaxpr (for a checkpointed layer, the recompute of whatever the policy
  did not save, then the transposed ops), and its leaves are every array that
  jaxpr reads. A leaf is either one of the layer's own inputs or a saved
  activation. forward flattens the pullback and tells the two apart by object
  identity. It keeps only the saved activations that the backward actually reads
  (see _used_pullback_leaves), plus the hidden input when it is read, since hidden
  differs per microbatch. Weights, state, positions and segments are never stored.
  backward puts the leaf list back together in its original order: inputs from
  its arguments, activations from saved, and zeros for leaves that are never read
  (XLA removes them). It then unflattens with the recorded treedef and calls the
  result on dy, which runs the layer's recompute and backward. This is not
  jax.fwd_and_bwd, because its residuals would include the layer's weights, so
  every pool slot would hold a copy of them.

  backward always uses the first trace's treedef. The layout check compares the
  stored residuals, not the backward jaxpr, so every trace must produce the same
  backward computation. This relies on how jax.vjp flattens its pullback.
  """
  layouts = []

  def inputs_of(weights, hidden, state, positions, segments):
    # Fixed flattening order; a pullback leaf's "source" is its index in this list.
    return jax.tree.leaves((weights, hidden, state, positions, segments))

  def forward(weights, hidden, state, positions, segments):
    # Only weights and hidden are differentiated; state, positions and segments are closed-over constants.
    out, pullback, metrics = jax.vjp(lambda w, x: apply(w, x, state, positions, segments), weights, hidden, has_aux=True)
    leaves, treedef = jax.tree.flatten(pullback)
    # A leaf that is the same object as an input is re-supplied at backward time; None marks a saved activation.
    index = {id(x): i for i, x in enumerate(inputs_of(weights, hidden, state, positions, segments))}
    sources = tuple(index.get(id(leaf)) for leaf in leaves)
    used = _used_pullback_leaves(treedef, leaves, out)
    kept = tuple(source is None and u for source, u in zip(sources, used))
    saved = tuple(leaf for leaf, k in zip(leaves, kept) if k)
    # hidden directly follows the weight leaves in inputs_of and is a single array.
    hidden_index = len(jax.tree.leaves(weights))
    needs_hidden = any(source == hidden_index and u for source, u in zip(sources, used))
    if needs_hidden:
      saved = (hidden,) + saved
    leaf_avals = tuple((tuple(x.shape), x.dtype) for x in leaves)
    # saved_avals (shape, dtype, memory space) is also the key that groups layers into residual pools.
    layout = _ResidualLayout(treedef, sources, tuple(_aval_key(x) for x in saved), kept, leaf_avals, needs_hidden)
    # treedef is not compared: it holds a jaxpr object that is new on every trace.
    if layouts and not layouts[0].matches(layout):
      raise ValueError("A layer's saved-residual layout changed between traces")
    layouts.append(layout)
    return out, saved, metrics

  def backward(weights, state, positions, segments, saved, dy):
    layout = layouts[0]
    saved = list(saved)
    # When the pullback never reads the hidden input, any same-shaped placeholder
    # fills its input slot; it is not referenced by layout.sources.
    hidden = saved.pop(0) if layout.needs_hidden else dy
    inputs = inputs_of(weights, hidden, state, positions, segments)
    activations = iter(saved)
    leaves = []
    for source, kept, (shape, dtype) in zip(layout.sources, layout.kept, layout.leaf_avals):
      if source is not None:
        leaves.append(inputs[source])
      elif kept:
        leaves.append(next(activations))
      else:
        # Never read by the backward, so XLA drops this placeholder.
        leaves.append(jnp.zeros(shape, dtype))
    # Calling the rebuilt pullback runs the recompute and backward; returns (dweights, dhidden).
    return jax.tree.unflatten(layout.treedef, leaves)(dy)

  forward.layouts = layouts
  return forward, backward


def _slot_tables(microbatches, layer_pool, pool_count):
  """Assign every (microbatch, layer) residual a slot in its pool, reusing freed slots.

  Simulates the schedule's residual traffic: fill writes microbatch 0's layers
  0..L-1; fused step k of later microbatch m reads (m-1, L-1-k) and then writes
  (m, k); the drain reads microbatch M-1's layers L-1..0. A write first reuses a
  slot freed in the same step (the read just before it), else the lowest free
  slot. Layers whose groups share one residual layout share a pool, so with full
  remat every write lands in the slot just read and the pool holds exactly L
  slots. Returns int32 tables [M, L] per pool and each pool's slot count.
  """
  layers = len(layer_pool)
  free = [[] for _ in range(pool_count)]
  size = [0] * pool_count
  table = np.zeros((pool_count, microbatches, layers), np.int32)
  where = {}

  def write(m, layer):
    pool = layer_pool[layer]
    if free[pool]:
      slot = free[pool].pop()
    else:
      slot, size[pool] = size[pool], size[pool] + 1
    table[pool, m, layer] = where[(m, layer)] = slot

  def read(m, layer):
    free[layer_pool[layer]].append(where.pop((m, layer)))
    free[layer_pool[layer]].sort(reverse=True)

  for layer in range(layers):
    write(0, layer)
  for m in range(1, microbatches):
    for k in range(layers):
      read(m - 1, layers - 1 - k)
      # Prefer the slot that was just freed so same-pool steps update in place.
      pool = layer_pool[layers - 1 - k]
      if layer_pool[k] == pool:
        just_freed = int(table[pool, m - 1, layers - 1 - k])
        free[pool].remove(just_freed)
        free[pool].append(just_freed)
      write(m, k)
  return table, size


def _sum_aux(total, aux):
  return _reduce_aux(jax.tree.map(lambda a, b: jnp.stack((a, b)), total, aux))


def _merge_layer_metrics(loss_aux, layer_aux):
  if not isinstance(loss_aux, dict) or not isinstance(layer_aux, dict) or loss_aux.keys() & layer_aux.keys():
    raise ValueError("Layer metrics and loss aux must be dictionaries with distinct keys")
  return {**loss_aux, **layer_aux}


def _group_boundaries_and_segments(params, states, group_count):
  """Validate grouped layer params/state; return group boundaries and fused-step segments.

  boundaries[g] is the global index of group g's first layer (boundaries[-1] is the layer count).
  In fused step k the backward runs global layer L-1-k while the forward runs layer k, so a
  segment breaks wherever either direction crosses a group boundary. Each segment is
  (backward_group, forward_group, backward_stop, forward_start, length) with group-local
  indices: the backward visits backward_stop-1 down, the forward forward_start up. For group
  lengths (1, 3) the segments are B1/F0 (1), B1/F1 (2), B0/F1 (1).
  """
  if (
      not isinstance(params, tuple)
      or not isinstance(states, tuple)
      or len(params) != group_count
      or len(states) != group_count
  ):
    raise ValueError("Grouped layer callbacks require matching parameter/state tuples")
  boundaries = [0]
  for group_params in params:
    leaves = jax.tree.leaves(group_params)
    if not leaves or leaves[0].ndim == 0 or leaves[0].shape[0] < 1:
      raise ValueError("Each layer group needs parameters with a positive leading layer axis; omit empty groups")
    length = leaves[0].shape[0]
    if any(x.ndim == 0 or x.shape[0] != length for x in leaves):
      raise ValueError("Parameters within a group must share the leading layer length")
    boundaries.append(boundaries[-1] + length)
  total_layers = boundaries[-1]
  cuts = sorted(set(boundaries) | {total_layers - boundary for boundary in boundaries})
  segments = []
  for start, end in zip(cuts[:-1], cuts[1:]):
    forward_group = next(i for i in range(len(params)) if start < boundaries[i + 1])
    backward_group = next(i for i in range(len(params)) if total_layers - 1 - start < boundaries[i + 1])
    forward_start = start - boundaries[forward_group]
    backward_stop = total_layers - start - boundaries[backward_group]
    segments.append((backward_group, forward_group, backward_stop, forward_start, end - start))
  return boundaries, segments


def make_training_schedule(
    prefix_apply,
    layer_apply,
    loss_apply,
    grad_dtype=jnp.float32,
    constrain_residual=None,
    anchor_segments=None,
    layer_loop_context=None,
):
  """Build a fused layer-level 1F1B loss/gradient computation.

  Callbacks:
    prefix_apply(boundary_params, data) -> hidden
    layer_apply: tuple of callbacks (params, hidden, state, positions, segments)
      -> (hidden, metric_dict), with matching metric keys across all groups
    loss_apply(boundary_params, hidden, data) -> (unnormalized_loss, aux)

  The returned step accepts (layer_params, layer_state, boundary_params, data).
  Layer parameters/state have a leading layer axis; every data leaf has a
  leading microbatch axis. The data mapping must contain inputs_position and
  inputs_segmentation. State is read-only and callbacks must be deterministic.

  layer_apply is a tuple of callbacks, one per
  contiguous homogeneous group. Pass matching tuples of layer_params/state;
  each group has its own positive leading layer length and parameter/residual
  structure. Returned layer gradients have the same tuple structure. Only the
  small number of groups/segments is statically expanded, not individual layers.

  Layer metrics are nondifferentiated and disjoint from loss aux. Use {} only
  if all groups have no metrics; otherwise use neutral values for dense groups.
  Loss metrics sum; TE overflow/capacity metrics use OR/max/min across layers and microbatches.

  Layer callbacks may be wrapped in jax.checkpoint (any remat policy, including
  full remat and policies that offload to host) or not. Each layer stores only
  the activations its backward reads (for a checkpointed layer, the ones its
  policy saves) in slot pools that the old and new microbatch share (see
  _slot_tables), and layer gradients are added into the running sums in place.

  constrain_residual(x) -> x optionally pins the sharding of each device-resident
  saved residual when it is stored and when it is read back. Pool buffers have
  no sharding of their own, so without it the partitioner may pick a layout for
  a restored residual that differs from the forward's (e.g. feature- instead of
  batch-sharded), turning backward weight-gradient GEMMs into all-gather/all-to-all
  heavy variants. Residuals offloaded to host memory are left untouched.

  anchor_segments=True also orders each fused segment after the work producing
  its input, like the drain. XLA peels pipelined iterations off short segment
  loops (all-gather prefetch, and host-offload prefetch when residuals are
  offloaded); once a loop is fully peeled, its weight slices/all-gathers can
  otherwise be hoisted ahead of the previous segment and stay live across it.
  The default (None) anchors segments only when some residual is offloaded to
  host: there the extra peel unrolls 3-layer segments and anchoring costs no
  time, while with device-only residuals the loops survive and anchoring would
  expose each segment's first weight all-gather.

  layer_loop_context() -> context manager optionally entered around each per-layer scan (not the
  microbatch scans that enclose them). The trainer uses it to mark the layer loops for XLA's
  while-loop double buffering, which is what prefetches the next layer's weight all-gather across
  iterations; microbatch loops must stay out of it so the accumulation loop is not unrolled.

  Returns (loss_sum, aux_sum, layer_grads_sum, boundary_grads_sum). Boundary
  parameters are shared by prefix and head so tied embeddings receive BOTH
  contributions. All parameters stay fixed across the accumulation window.
  """
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("The dual-pipe experiment requires JAX with jax.fwd_and_bwd")

  loop_context = layer_loop_context or contextlib.nullcontext

  def layer_scan(*args, **kwargs):
    with loop_context():
      return jax.lax.scan(*args, **kwargs)

  prefix_forward, prefix_backward = jax.fwd_and_bwd(prefix_apply, argnums=(0,), jitted=False)
  if not isinstance(layer_apply, tuple):
    raise TypeError("Layer callbacks must be a tuple, one per group")
  if not layer_apply:
    raise ValueError("At least one layer group is required")

  layer_passes = tuple(_layer_passes(apply) for apply in layer_apply)
  # Saved-residual layouts recorded when each forward is traced.
  residual_layouts = tuple(forward.layouts for forward, _ in layer_passes)
  # Reuse each forward trace's saved-VJP metadata between fill and steady state.
  # These helpers inline into the caller's train_step, not separate GPU calls.
  prefix_forward = jax.jit(prefix_forward, inline=True)
  prefix_backward = jax.jit(prefix_backward, inline=True)
  forwards = tuple(jax.jit(forward, inline=True) for forward, _ in layer_passes)
  backwards = tuple(jax.jit(backward, inline=True) for _, backward in layer_passes)
  loss_and_grad = jax.jit(jax.value_and_grad(loss_apply, argnums=(0, 1), has_aux=True), inline=True)

  def accumulate(total, grads):
    return jax.tree.map(lambda a, g: a + g.astype(grad_dtype), total, grads)

  def step(layer_params, layer_state, boundary_params, microbatch_data):
    """1F1B with shared residual slot pools and in-place gradient sums.

    Each layer saves only the residuals its backward reads (its input hidden state for
    full remat). Layers whose saved residuals have the same layout share one
    pool; _slot_tables assigns slots so the new microbatch's forward reuses slots
    the old microbatch's backward has finished reading. With full remat all
    groups share one pool of L slots and every fused step writes the slot it just
    read; otherwise each pool is sized by its own peak.
    Positions/segments are shared by all layers of a microbatch and kept once.
    Layer gradients are added into the running sums one layer at a time instead
    of materializing a per-microbatch gradient stack.
    """
    microbatches = microbatch_data["inputs_position"].shape[0]
    if microbatches < 1:
      raise ValueError("At least one microbatch is required")
    if any(x.shape[0] != microbatches for x in jax.tree.leaves(microbatch_data)):
      raise ValueError("All input leaves must have the same leading microbatch axis")
    params, states = layer_params, layer_state
    boundaries, segments = _group_boundaries_and_segments(params, states, len(forwards))
    total_layers = boundaries[-1]
    group_of = [g for g in range(len(params)) for _ in range(boundaries[g + 1] - boundaries[g])]

    def take(tree, index):
      return jax.tree.map(lambda x: jax.lax.dynamic_index_in_dim(x, index, axis=0, keepdims=False), tree)

    def put(buffer, value, index):
      return jax.lax.dynamic_update_index_in_dim(buffer, value.astype(buffer.dtype), index, axis=0)

    first_data = jax.tree.map(lambda x: x[0], microbatch_data)
    later_data = jax.tree.map(lambda x: x[1:], microbatch_data)
    hidden_shape = jax.eval_shape(lambda: prefix_forward(boundary_params, first_data)[0])

    # Trace each group's forward once (abstractly) to record the layout of its saved residuals.
    saved_layouts = []
    for group in range(len(params)):
      jax.eval_shape(
          lambda g=group: forwards[g](
              take(params[g], 0),
              jnp.zeros(hidden_shape.shape, hidden_shape.dtype),
              take(states[g], 0),
              first_data["inputs_position"],
              first_data["inputs_segmentation"],
          )
      )
      saved_layouts.append(residual_layouts[group][0].saved_avals)
    pool_layouts = list(dict.fromkeys(saved_layouts))
    group_pool = [pool_layouts.index(layout) for layout in saved_layouts]
    layer_pool = [group_pool[g] for g in group_of]
    tables, pool_sizes = _slot_tables(microbatches, layer_pool, len(pool_layouts))
    anchor = anchor_segments
    if anchor is None:
      anchor = any(space == jax.memory.Space.Host for layout in pool_layouts for _, _, space in layout)
    # Slot tables stay static numpy. Every slot/layer index a layer loop uses is then a constant array
    # scanned as xs, i.e. a function of the loop's own induction variable only. XLA can resolve such
    # offsets at compile time and issue host residual reads/writes as DMA copies; offsets that depend
    # on a traced microbatch counter (or a non-constant loop-carried counter) fall back to SM kernels.
    tables_np = tables
    # With period-2 slot reuse (true for a shared pool), the steady state is two alternating phases.
    periodic = all(np.array_equal(tables_np[:, m], tables_np[:, m - 2]) for m in range(2, microbatches))

    def table_row(microbatch):
      return tables_np[:, microbatch % 2 if periodic else microbatch]

    for pool, layout in enumerate(pool_layouts):
      groups = [g for g in range(len(params)) if group_pool[g] == pool]
      logging.info(
          "dual_pipe residual pool %d: groups %s, %d slots of %s",
          pool,
          groups,
          pool_sizes[pool],
          ", ".join(
              f"{dtype}[{','.join(map(str, shape))}]@{getattr(space, 'name', space)}" for shape, dtype, space in layout
          ),
      )

    fill_count = itertools.count(1)

    is_host = [[space == jax.memory.Space.Host for _, _, space in layout] for layout in pool_layouts]
    # Host residuals ping-pong between two buffers by microbatch parity: microbatch m writes side m % 2
    # and the backward of m reads that side while m+1 fills the other. A read (an async host->device
    # copy that may span a whole loop iteration) therefore never overlaps an in-place write to the same
    # buffer, which XLA would otherwise resolve by copying the whole pool. Host memory is not the
    # constrained resource (device memory is), so these keep one slot per layer and no slot reuse.
    pool_layer_rank = np.zeros(total_layers, np.int32)
    for pool in range(len(pool_layouts)):
      ranks = [layer for layer in range(total_layers) if layer_pool[layer] == pool]
      pool_layer_rank[ranks] = np.arange(len(ranks))
    pool_layer_count = [layer_pool.count(pool) for pool in range(len(pool_layouts))]

    def new_pool(pool):
      def buffer(key, host):
        shape, dtype, _ = key
        # Contents are never read before being written. Distinct fill values keep XLA from
        # merging same-shaped buffers into one, which copy insertion would then duplicate.
        if host:
          return tuple(
              jax.device_put(jnp.full((pool_layer_count[pool],) + shape, next(fill_count), dtype), jax.memory.Space.Host)
              for _ in range(2)
          )
        return jnp.full((pool_sizes[pool],) + shape, next(fill_count), dtype)

      return tuple(buffer(key, host) for key, host in zip(pool_layouts[pool], is_host[pool]))

    def constrain(values, pool):
      if constrain_residual is None:
        return values
      return tuple(x if host else constrain_residual(x) for x, host in zip(values, is_host[pool]))

    def read(pools, group, loc, side):
      """Residuals at loc = (shared slot, layer rank); host buffers use `side`, device buffers the slot."""
      pool = group_pool[group]
      values = tuple(
          jax.lax.dynamic_index_in_dim(x[side] if host else x, loc[1] if host else loc[0], 0, False)
          for x, host in zip(pools[pool], is_host[pool])
      )
      # Move host residuals to the device here. Left in host memory, a prefetched value that crosses
      # a loop iteration is copied host->host (slowly) and only reaches the device at its use.
      values = tuple(jax.device_put(x, jax.memory.Space.Device) if host else x for x, host in zip(values, is_host[pool]))
      return constrain(values, pool)

    def write(pools, group, loc, side, saved):
      pool = group_pool[group]
      pools = list(pools)
      saved = constrain(saved, pool)
      updated = []
      for buffer, value, host in zip(pools[pool], saved, is_host[pool]):
        if host:
          pair = list(buffer)
          pair[side] = put(pair[side], value, loc[1])
          updated.append(tuple(pair))
        else:
          updated.append(put(buffer, value, loc[0]))
      pools[pool] = tuple(updated)
      return tuple(pools)

    def read_before_write(saved, pools, group):
      """Order device-residual reads before the in-place slot writes that follow.

      Without it copy insertion protects the reads by copying the whole pool. Host residuals are
      excluded: they ping-pong between buffers, so their (async, possibly long) copies need no
      ordering, and forcing them to finish here would stop them overlapping the layer's compute.
      """
      pool = group_pool[group]
      device = [i for i, host in enumerate(is_host[pool]) if not host]
      if not device:
        return saved, pools
      saved_device, buffers = jax.lax.optimization_barrier(
          (tuple(saved[i] for i in device), tuple(pools[pool][i] for i in device))
      )
      saved, updated = list(saved), list(pools[pool])
      for i, value, buffer in zip(device, saved_device, buffers):
        saved[i], updated[i] = value, buffer
      return tuple(saved), tuple(pools[:pool]) + (tuple(updated),) + tuple(pools[pool + 1 :])

    def group_locs(row, group, layers):
      """[len(layers), 2] int32 of (shared slot, layer rank in pool) for group-local `layers`."""
      layers = np.asarray(layers)
      slots = row[group_pool[group]][boundaries[group] + layers]
      return jnp.asarray(np.stack([slots, pool_layer_rank[boundaries[group] + layers]], -1), jnp.int32)

    def next_slots(slots):
      """Slot each iteration should prefetch: the following iteration's (the last repeats itself)."""
      return jnp.concatenate((slots[1:], slots[-1:]))

    def anchored(weights, dependency):
      """Weights that become available only once `dependency` is computed.

      Loop-peeled weight slices/all-gathers (and host prefetches) are indexed by constants, so the
      scheduler could hoist them to the start of the step, where they would stay live across the whole
      steady phase. Routing the stacked weights through a barrier with the previous segment's output
      pins them. (A loop-carried counter started from the barrier would do the same but hides the
      counter's init from XLA, which then cannot turn the slot offsets into DMA copies.)
      """
      if not anchor:
        return weights
      return jax.lax.optimization_barrier((weights, dependency))[0]

    def add_layer(total, grads, index):
      def add(a, g):
        return put(a, jax.lax.dynamic_index_in_dim(a, index, 0, False) + g.astype(grad_dtype), index)

      return jax.tree.map(add, total, grads)

    def forward_microbatch(data, pools, microbatch):
      """F only: store one microbatch's residuals, then head loss/backward."""
      with jax.named_scope("prefix_forward"):
        hidden, prefix_residuals = prefix_forward(boundary_params, data)
      layer_metrics = None
      row = table_row(microbatch)
      for group, (forward, weights, state) in enumerate(zip(forwards, params, states)):
        count = boundaries[group + 1] - boundaries[group]

        def body(carry, layer_data, group=group, forward=forward):
          hidden, pools = carry
          weights, state, loc = layer_data
          with jax.named_scope("forward"):
            hidden, saved, metrics = forward(weights, hidden, state, data["inputs_position"], data["inputs_segmentation"])
          pools = write(pools, group, loc, microbatch % 2, saved)
          return (hidden, pools), metrics

        (hidden, pools), metrics = layer_scan(
            body, (hidden, pools), (weights, state, group_locs(row, group, np.arange(count)))
        )
        metrics = _reduce_aux(metrics)
        layer_metrics = metrics if layer_metrics is None else _sum_aux(layer_metrics, metrics)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, dhidden) = loss_and_grad(boundary_params, hidden, data)
      return prefix_residuals, pools, dhidden, loss, _merge_layer_metrics(aux, layer_metrics), head_grads

    def backward_microbatch(pools, microbatch, dhidden, data, layer_sums):
      """B only (drain): read the residuals from the last layer to the first."""
      layer_sums = list(layer_sums)
      row = table_row(microbatch)
      for group in reversed(range(len(params))):
        count = boundaries[group + 1] - boundaries[group]
        layers = count - 1 - np.arange(count)
        group_weights, group_state = anchored((params[group], states[group]), dhidden)

        locs = group_locs(row, group, layers)
        side = microbatch % 2

        def body(carry, loop_data, group=group, group_weights=group_weights, group_state=group_state, side=side):
          dhidden, total, saved = carry
          layer, prefetch_loc = loop_data
          # Start the next layer's host->device copy now so it overlaps this layer's backward.
          next_saved = read(pools, group, prefetch_loc, side)
          weights, state = take((group_weights, group_state), layer)
          with jax.named_scope("backward"):
            dweights, dhidden = backwards[group](
                weights, state, data["inputs_position"], data["inputs_segmentation"], saved, dhidden
            )
          return (dhidden, add_layer(total, dweights, layer), next_saved), None

        (dhidden, layer_sums[group], _), _ = layer_scan(
            body,
            (dhidden, layer_sums[group], read(pools, group, locs[0], side)),
            (jnp.asarray(layers, jnp.int32), next_slots(locs)),
        )
      return dhidden, tuple(layer_sums)

    layer_total = tuple(jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), group) for group in params)
    boundary_total = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), boundary_params)
    pools = tuple(new_pool(pool) for pool in range(len(pool_layouts)))

    with jax.named_scope("fill_F0"):
      prefix_residuals, pools, dhidden, loss_total, aux_total, head_grads = forward_microbatch(first_data, pools, 0)
    boundary_total = accumulate(boundary_total, head_grads)

    def backward_forward(carry, data, microbatch):
      """B of microbatch-1 fused with F of `microbatch` (a static index)."""
      (prev_prefix_residuals, pools, prev_dhidden, prev_data, loss_sum, aux_sum, layer_sum, boundary_sum) = carry
      with jax.named_scope("prefix_forward"):
        next_hidden, next_prefix_residuals = prefix_forward(boundary_params, data)
      layer_sum = list(layer_sum)
      layer_metrics = None
      dprefix = prev_dhidden
      read_row, write_row = table_row(microbatch - 1), table_row(microbatch)
      with jax.named_scope("combined_bf_layers"):
        for backward_group, forward_group, backward_stop, forward_start, length in segments:
          backward_layers = backward_stop - 1 - np.arange(length)
          forward_layers = forward_start + np.arange(length)
          backward_params, backward_states = anchored(
              (params[backward_group], states[backward_group]), (dprefix, next_hidden)
          )
          forward_params, forward_states = (
              (backward_params, backward_states)
              if forward_group == backward_group
              else anchored((params[forward_group], states[forward_group]), (dprefix, next_hidden))
          )

          read_locs = group_locs(read_row, backward_group, backward_layers)

          def combined_layer(
              carry,
              loop_data,
              backward_group=backward_group,
              forward_group=forward_group,
              backward_params=backward_params,
              backward_states=backward_states,
              forward_params=forward_params,
              forward_states=forward_states,
          ):
            dhidden, next_hidden, pools, total, saved = carry
            backward_layer, forward_layer, prefetch_loc, write_loc = loop_data
            # Prefetch the next backward layer's residuals (host->device) so the copy overlaps
            # this iteration's compute; this iteration's backward uses the carried `saved`.
            with jax.named_scope("bwd_residual_prefetch"):
              next_saved = read(pools, backward_group, prefetch_loc, (microbatch - 1) % 2)
              # Order the read before any later write into its pool: this iteration's when both groups
              # share a pool, otherwise a later segment's, which can be unordered with it once XLA peels
              # this iteration out of the loop.
              next_saved, pools = read_before_write(next_saved, pools, backward_group)
            # Named scopes mark which direction each weight gather / copy serves in profiles.
            with jax.named_scope("bwd_weights"):
              backward_weights, backward_state = take((backward_params, backward_states), backward_layer)
            with jax.named_scope("fwd_weights"):
              next_weights, next_state = take((forward_params, forward_states), forward_layer)
            with jax.named_scope("backward"):
              dweights, dhidden = backwards[backward_group](
                  backward_weights,
                  backward_state,
                  prev_data["inputs_position"],
                  prev_data["inputs_segmentation"],
                  saved,
                  dhidden,
              )
            with jax.named_scope("forward"):
              next_hidden, next_saved_fwd, metrics = forwards[forward_group](
                  next_weights, next_hidden, next_state, data["inputs_position"], data["inputs_segmentation"]
              )
            # ...and the new microbatch's forward stores its residuals into freed slots.
            with jax.named_scope("fwd_residual_write"):
              pools = write(pools, forward_group, write_loc, microbatch % 2, next_saved_fwd)
            with jax.named_scope("bwd_grad_accumulate"):
              total = add_layer(total, dweights, backward_layer)
            return (dhidden, next_hidden, pools, total, next_saved), metrics

          # Peeled read of the first layer; the in-loop prefetches never touch a slot written earlier.
          first_saved = read(pools, backward_group, read_locs[0], (microbatch - 1) % 2)
          # A later forward write (this segment's, or a later segment's when the groups use different
          # pools) reuses the slot this read frees. Nothing else orders this read before that write once
          # XLA peels loop iterations into the enclosing computation, and copy insertion then protects the
          # read by copying the whole pool. Thread the pool through the barrier so every later write
          # depends on the read.
          first_saved, pools = read_before_write(first_saved, pools, backward_group)
          (dprefix, next_hidden, pools, layer_sum[backward_group], _), metrics = layer_scan(
              combined_layer,
              (dprefix, next_hidden, pools, layer_sum[backward_group], first_saved),
              (
                  jnp.asarray(backward_layers, jnp.int32),
                  jnp.asarray(forward_layers, jnp.int32),
                  next_slots(read_locs),
                  group_locs(write_row, forward_group, forward_layers),
              ),
          )
          metrics = _reduce_aux(metrics)
          layer_metrics = metrics if layer_metrics is None else _sum_aux(layer_metrics, metrics)

      with jax.named_scope("prefix_backward"):
        (prefix_grads,) = prefix_backward(prev_prefix_residuals, dprefix)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, next_dhidden) = loss_and_grad(boundary_params, next_hidden, data)
      aux = _merge_layer_metrics(aux, layer_metrics)
      boundary_sum = accumulate(accumulate(boundary_sum, prefix_grads), head_grads)
      return (
          next_prefix_residuals,
          pools,
          next_dhidden,
          data,
          loss_sum + loss,
          _sum_aux(aux_sum, aux),
          tuple(layer_sum),
          boundary_sum,
      )

    carry = (prefix_residuals, pools, dhidden, first_data, loss_total, aux_total, layer_total, boundary_total)
    steady = microbatches - 1
    with jax.named_scope("steady_Bi_Fnext"):
      if periodic and steady >= 2:
        # Two microbatches per iteration so both slot phases are static; pairs start at microbatch 1.
        pair_count = steady // 2
        pair_data = jax.tree.map(lambda x: x[: 2 * pair_count].reshape((pair_count, 2) + x.shape[1:]), later_data)

        def pair_step(carry, data):
          carry = backward_forward(carry, jax.tree.map(lambda x: x[0], data), 1)
          return backward_forward(carry, jax.tree.map(lambda x: x[1], data), 2), None

        carry, _ = jax.lax.scan(pair_step, carry, pair_data)
        first_leftover = 2 * pair_count + 1
      else:
        first_leftover = 1
      if not periodic and microbatches - first_leftover > 4:
        logging.warning(
            "dual_pipe: residual slot tables are not period-2, so %d microbatch steps are unrolled in the "
            "program; compile time and program size grow with gradient_accumulation_steps",
            microbatches - first_leftover,
        )
      for microbatch in range(first_leftover, microbatches):
        carry = backward_forward(carry, jax.tree.map(lambda x, m=microbatch: x[m - 1], later_data), microbatch)
    prefix_residuals, pools, dhidden, last_data, loss_total, aux_total, layer_total, boundary_total = carry
    with jax.named_scope("drain_Blast"):
      dprefix, layer_total = backward_microbatch(pools, microbatches - 1, dhidden, last_data, layer_total)
      (prefix_grads,) = prefix_backward(prefix_residuals, dprefix)
    return loss_total, aux_total, layer_total, accumulate(boundary_total, prefix_grads)

  return step
