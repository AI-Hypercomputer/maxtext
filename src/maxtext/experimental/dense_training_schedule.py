"""Pure-JAX layer-level 1F1B accumulation with the real training loss.

The caller owns normalization and the optimizer update and jits the enclosing
training step. This module sums loss, metrics, and gradients over microbatches.
"""

import jax
import jax.numpy as jnp


def make_training_schedule(
    prefix_apply, layer_apply, loss_apply, schedule="dual_pipe", grad_dtype=jnp.float32,
    *, layer_has_aux=False, reduce_aux=None,
):
  """Build a serial or fused layer-level 1F1B loss/gradient computation.

  Callbacks:
    prefix_apply(boundary_params, data) -> hidden
    layer_apply(params, hidden, state, positions, segments) -> hidden
    loss_apply(boundary_params, hidden, data) -> (unnormalized_loss, aux)

  The returned step accepts (layer_params, layer_state, boundary_params, data).
  Layer parameters/state have a leading layer axis; every data leaf has a
  leading microbatch axis. The data mapping must contain inputs_position and
  inputs_segmentation. State is read-only and callbacks must be deterministic.

  For heterogeneous stacks, layer_apply may be a tuple of callbacks, one per
  contiguous homogeneous group. Pass matching tuples of layer_params/state;
  each group has its own positive leading layer length and parameter/residual
  structure. Returned layer gradients have the same tuple structure. Only the
  small number of groups/segments is statically expanded, not individual layers.

  With layer_has_aux=True, every layer callback returns (hidden, metric_dict).
  These metrics are not differentiated and must have different keys from the
  loss callback's aux dict. reduce_aux(stacked_aux) reduces a leading axis,
  defaulting to a sum; a custom associative reducer can use any/max/min for
  overflow/capacity metrics. It also combines metrics across microbatches.

  Returns (loss_sum, aux_sum, layer_grads_sum, boundary_grads_sum). Boundary
  parameters are shared by prefix and head so tied embeddings receive BOTH
  contributions. All parameters stay fixed across the accumulation window.
  """
  if schedule not in ("serial", "dual_pipe"):
    raise ValueError(f"Unknown schedule: {schedule}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("The dual-pipe experiment requires JAX with jax.fwd_and_bwd")

  prefix_forward, prefix_backward = jax.fwd_and_bwd(prefix_apply, argnums=(0,), jitted=False)
  grouped = isinstance(layer_apply, tuple)
  layer_functions = layer_apply if grouped else (layer_apply,)
  if not layer_functions:
    raise ValueError("At least one layer group is required")
  layer_passes = tuple(
      jax.fwd_and_bwd(apply, argnums=(0, 1), has_aux=layer_has_aux, jitted=False) for apply in layer_functions
  )
  # Reuse each forward trace's saved-VJP metadata between fill and steady state.
  # These helpers inline into the caller's train_step, not separate GPU calls.
  prefix_forward = jax.jit(prefix_forward, inline=True)
  prefix_backward = jax.jit(prefix_backward, inline=True)
  forwards = tuple(jax.jit(forward, inline=True) for forward, _ in layer_passes)
  backwards = tuple(jax.jit(backward, inline=True) for _, backward in layer_passes)
  loss_and_grad = jax.jit(jax.value_and_grad(loss_apply, argnums=(0, 1), has_aux=True), inline=True)

  def step(layer_params, layer_state, boundary_params, microbatch_data):
    microbatches = microbatch_data["inputs_position"].shape[0]
    if microbatches < 1:
      raise ValueError("At least one microbatch is required")
    if any(x.shape[0] != microbatches for x in jax.tree.leaves(microbatch_data)):
      raise ValueError("All input leaves must have the same leading microbatch axis")
    params = layer_params if grouped else (layer_params,)
    states = layer_state if grouped else (layer_state,)
    if not isinstance(params, tuple) or not isinstance(states, tuple) or len(params) != len(forwards) or len(states) != len(forwards):
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

    # Break only where the forward OR reverse layer type changes. For group
    # lengths (1, 3), these are B1/F0 (1), B1/F1 (2), B0/F1 (1).
    cuts = sorted(set(boundaries) | {total_layers - boundary for boundary in boundaries})
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
      forward_group = next(i for i in range(len(params)) if start < boundaries[i + 1])
      backward_group = next(i for i in range(len(params)) if total_layers - 1 - start < boundaries[i + 1])
      forward_start = start - boundaries[forward_group]
      backward_stop = total_layers - start - boundaries[backward_group]
      segments.append((backward_group, forward_group, backward_stop, forward_start, end - start))

    def unpack_groups(values):
      return tuple(values) if grouped else values[0]

    def accumulate(total, grads):
      return jax.tree.map(lambda a, g: a + g.astype(grad_dtype), total, grads)

    def reduce_metrics(stacked):
      if reduce_aux is not None:
        return reduce_aux(stacked)
      return jax.tree.map(lambda x: jnp.sum(x, axis=0), stacked)

    def sum_aux(total, aux):
      return reduce_metrics(jax.tree.map(lambda a, b: jnp.stack((a, b)), total, aux))

    def merge_layer_metrics(loss_aux, layer_aux):
      if not layer_has_aux:
        return loss_aux
      if not isinstance(loss_aux, dict) or not isinstance(layer_aux, dict) or loss_aux.keys() & layer_aux.keys():
        raise ValueError("Layer metrics and loss aux must be dictionaries with distinct keys")
      return {**loss_aux, **layer_aux}

    def call_forward(forward, weights, hidden, state, data):
      result = forward(weights, hidden, state, data["inputs_position"], data["inputs_segmentation"])
      if layer_has_aux:
        return result
      hidden, residuals = result
      return hidden, residuals, {}

    def forward_layers(hidden, data):
      residual_groups = []
      layer_metrics = None
      for forward, weights, state in zip(forwards, params, states):
        def body(hidden, layer_data):
          weights, state = layer_data
          with jax.named_scope("forward"):
            hidden, residuals, metrics = call_forward(forward, weights, hidden, state, data)
          return hidden, (residuals, metrics)

        hidden, (residuals, metrics) = jax.lax.scan(body, hidden, (weights, state))
        residual_groups.append(residuals)
        metrics = reduce_metrics(metrics)
        layer_metrics = metrics if layer_metrics is None else sum_aux(layer_metrics, metrics)
      return hidden, tuple(residual_groups), layer_metrics

    def backward_layers(residuals, dhidden):
      gradient_groups = [None] * len(params)
      for group in reversed(range(len(params))):
        def body(dhidden, residual):
          with jax.named_scope("backward"):
            dweights, dhidden = backwards[group](residual, dhidden)
          return dhidden, dweights

        # reverse=True visits L-1..0 but stacks gradients in original order.
        dhidden, gradient_groups[group] = jax.lax.scan(body, dhidden, residuals[group], reverse=True)
      return dhidden, unpack_groups(gradient_groups)

    def forward_microbatch(data):
      with jax.named_scope("prefix_forward"):
        hidden, prefix_residuals = prefix_forward(boundary_params, data)
      hidden, residuals, layer_metrics = forward_layers(hidden, data)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, dhidden) = loss_and_grad(boundary_params, hidden, data)
      return prefix_residuals, residuals, dhidden, loss, merge_layer_metrics(aux, layer_metrics), head_grads

    first_data = jax.tree.map(lambda x: x[0], microbatch_data)
    later_data = jax.tree.map(lambda x: x[1:], microbatch_data)
    layer_total = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), layer_params)
    boundary_total = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), boundary_params)

    with jax.named_scope("fill_F0"):
      prefix_residuals, residuals, dhidden, loss_total, aux_total, head_grads = forward_microbatch(first_data)
    boundary_total = accumulate(boundary_total, head_grads)

    if schedule == "serial" or microbatches == 1:
      with jax.named_scope("serial_B0"):
        dprefix, layer_grads = backward_layers(residuals, dhidden)
        prefix_grads, = prefix_backward(prefix_residuals, dprefix)
      layer_total = accumulate(layer_total, layer_grads)
      boundary_total = accumulate(boundary_total, prefix_grads)

      def serial_microbatch(carry, data):
        loss_sum, aux_sum, layer_sum, boundary_sum = carry
        prefix_res, layer_res, dy, loss, aux, head_grads = forward_microbatch(data)
        dprefix, layer_grads = backward_layers(layer_res, dy)
        with jax.named_scope("prefix_backward"):
          prefix_grads, = prefix_backward(prefix_res, dprefix)
        boundary_sum = accumulate(accumulate(boundary_sum, head_grads), prefix_grads)
        return (loss_sum + loss, sum_aux(aux_sum, aux), accumulate(layer_sum, layer_grads), boundary_sum), None

      with jax.named_scope("serial_FB"):
        totals, _ = jax.lax.scan(
            serial_microbatch, (loss_total, aux_total, layer_total, boundary_total), later_data
        )
      return totals

    def backward_forward(carry, data):
      previous_prefix_residuals, previous_residuals, previous_dhidden, loss_sum, aux_sum, layer_sum, boundary_sum = carry
      with jax.named_scope("prefix_forward"):
        next_hidden, next_prefix_residuals = prefix_forward(boundary_params, data)
      gradient_parts = [[] for _ in params]
      residual_parts = [[] for _ in params]
      layer_metrics = None
      dprefix = previous_dhidden
      with jax.named_scope("combined_bf_layers"):
        for backward_group, forward_group, backward_stop, forward_start, length in segments:
          old_residuals = jax.tree.map(
              lambda r: r[backward_stop - length : backward_stop][::-1], previous_residuals[backward_group]
          )
          next_weights, next_state = jax.tree.map(
              lambda x: x[forward_start : forward_start + length], (params[forward_group], states[forward_group])
          )

          def combined_layer(carry, layer_data):
            dhidden, next_hidden = carry
            old_residual, next_weights, next_state = layer_data
            # B_i at layer L-1-k and F_(i+1) at layer k have independent carries.
            with jax.named_scope("backward"):
              dweights, dhidden = backwards[backward_group](old_residual, dhidden)
            with jax.named_scope("forward"):
              next_hidden, next_residual, metrics = call_forward(
                  forwards[forward_group], next_weights, next_hidden, next_state, data
              )
            return (dhidden, next_hidden), (dweights, next_residual, metrics)

          (dprefix, next_hidden), (reversed_grads, next_residuals, metrics) = jax.lax.scan(
              combined_layer, (dprefix, next_hidden), (old_residuals, next_weights, next_state)
          )
          gradient_parts[backward_group].append(jax.tree.map(lambda g: g[::-1], reversed_grads))
          residual_parts[forward_group].append(next_residuals)
          metrics = reduce_metrics(metrics)
          layer_metrics = metrics if layer_metrics is None else sum_aux(layer_metrics, metrics)

      def concatenate(parts):
        if len(parts) == 1:
          return parts[0]
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)

      # Backward segments arrive in descending group-local layer order.
      layer_grads = unpack_groups([concatenate(list(reversed(parts))) for parts in gradient_parts])
      next_residuals = tuple(concatenate(parts) for parts in residual_parts)
      with jax.named_scope("prefix_backward"):
        prefix_grads, = prefix_backward(previous_prefix_residuals, dprefix)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, next_dhidden) = loss_and_grad(boundary_params, next_hidden, data)
      aux = merge_layer_metrics(aux, layer_metrics)
      boundary_sum = accumulate(accumulate(boundary_sum, prefix_grads), head_grads)
      return (
          next_prefix_residuals,
          next_residuals,
          next_dhidden,
          loss_sum + loss,
          sum_aux(aux_sum, aux),
          accumulate(layer_sum, layer_grads),
          boundary_sum,
      ), None

    # F0, [B0 + F1], [B1 + F2], ..., B_last. Only the next microbatch's
    # residuals persist in the outer carry; old/new residuals can coexist inside.
    with jax.named_scope("steady_Bi_Fnext"):
      carry, _ = jax.lax.scan(
          backward_forward,
          (prefix_residuals, residuals, dhidden, loss_total, aux_total, layer_total, boundary_total),
          later_data,
      )
    prefix_residuals, residuals, dhidden, loss_total, aux_total, layer_total, boundary_total = carry
    with jax.named_scope("drain_Blast"):
      dprefix, layer_grads = backward_layers(residuals, dhidden)
      prefix_grads, = prefix_backward(prefix_residuals, dprefix)
    return loss_total, aux_total, accumulate(layer_total, layer_grads), accumulate(boundary_total, prefix_grads)

  return step
