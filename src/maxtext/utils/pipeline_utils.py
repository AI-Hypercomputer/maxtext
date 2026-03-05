# Copyright 2023–2025 Google LLC
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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining"""

import functools
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned


def get_fsdp_index_pytree(physical_partition_spec, axis_name="fsdp"):
  """
  Finds the index of 'fsdp' within each PartitionSpec in a Pytree.

  Args:
    physical_partition_spec: A Pytree where leaves are PartitionSpecs.
    axis_name: physical axis name for indexing

  Returns:
    A Pytree of the same structure where leaves are the integer index
    of 'fsdp' or -1 if not found.
  """

  def find_fsdp(pspec):
    # Ensure we are handling a PartitionSpec or a tuple/list of strings
    if pspec is None:
      return -1

    # PartitionSpecs are essentially tuples (e.g., PartitionSpec('data', 'fsdp'))
    for i, axis in enumerate(pspec):
      # Handle cases where an axis might be a tuple itself (e.g., ('fsdp', 'tensor'))
      if isinstance(axis, (list, tuple)):
        if axis_name in axis:
          return i
      elif axis == axis_name:
        return i
    return -1

  return jax.tree.map(find_fsdp, physical_partition_spec)


def generate_bsw_pps_from_pps(physical_partition_spec):
  """Create bsw physical partition spec from weight physical partition spec."""

  def _process_pps(path, pps):
    # Extract string keys from the JAX KeyPath elements safely
    path_keys = [getattr(p, "key", str(p)) for p in path]
    is_moe_block_0 = "MoeBlock_0" in path_keys

    # Remove the gathered axes conditionally based on the path
    processed_pps = remove_gathered_axes_from_physical_partition_spec(pps, is_moe_block_0)

    # Keep the original [1:] slicing behavior (e.g., to drop the 'stage' axis)
    return P(*processed_pps[1:])

  return jax.tree_util.tree_map_with_path(
      _process_pps,
      physical_partition_spec,
  )


def remove_gathered_axes_from_physical_partition_spec(pps, is_moe_block_0):
  """Removes 'fsdp', 'fsdp_transpose', and conditionally 'expert' from a physical PartitionSpec."""

  # Always remove fsdp and fsdp_transpose as they are always gathered
  axes_to_remove = ["fsdp", "fsdp_transpose"]

  # Only remove 'expert' if we are NOT in MoeBlock_0
  if not is_moe_block_0:
    axes_to_remove.append("expert")

  if isinstance(pps, P):
    new_spec = []
    # Iterate through each axis in the original PartitionSpec.
    for axis in pps:
      if axis is None:
        new_spec.append(None)
      elif isinstance(axis, str):
        # If the axis is in our removal list, replace it with None to signify replication.
        if axis not in axes_to_remove:
          new_spec.append(axis)
        else:
          new_spec.append(None)
      elif isinstance(axis, (list, tuple)):
        # If the axis is a collection, filter out the gathered axes.
        new_axis = [a for a in axis if a not in axes_to_remove]
        # If all elements are filtered out, new_axis becomes [], which as a tuple ()
        # correctly signals replication across those mesh axes in JAX.
        new_spec.append(tuple(new_axis))
      else:
        raise ValueError(f"Unsupported_axis_type: {type(axis)}")

    # Return a new sharding object with the modified spec.
    return P(*new_spec)

  return pps


def get_logical_spec_repeats_removed(full_logical):
  """Removes 'circular_repeats' from logical partition spec."""
  if full_logical is None:
    return None

  def _remove_from_spec(spec):
    return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

  return jax.tree.map(_remove_from_spec, full_logical)


# TODO(chengnuojin) Remove this function and its usage after pipeline nnx migration
def remove_logically_partition(weights):
  """Removes LogicallyPartitioned wrapper from weights."""

  def _remove_logically_partition_leaf(v):
    return getattr(v, "value") if isinstance(v, LogicallyPartitioned) else v

  return jax.tree.map(
      _remove_logically_partition_leaf,
      weights,
      is_leaf=lambda v: isinstance(v, LogicallyPartitioned),
  )


def remove_fsdp_from_physical_partition_spec(pps):
  """Removes 'fsdp' and 'fsdp_transpose' from a physical PartitionSpec."""
  if isinstance(pps, P):
    new_spec = []
    # Iterate through each axis in the original PartitionSpec.
    for axis in pps:
      if axis is None:
        new_spec.append(None)
      elif isinstance(axis, str):
        # If the axis is 'fsdp', replace it with None to signify replication.
        if axis not in ("fsdp", "fsdp_transpose"):
          new_spec.append(axis)
        else:
          new_spec.append(None)
      elif isinstance(axis, (list, tuple)):
        # If the axis is a collection, filter out 'fsdp'.
        new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
        new_spec.append(tuple(new_axis))
      else:
        raise ValueError(f"Unsupported_axis_type: {type(axis)}")
      # Return a new sharding object with the modified spec.
    return P(*new_spec)
  return pps


def create_scanned_function(
    model,
    length,
    deterministic=True,
    model_mode=None,
    logical_partition_spec=None,
):
  """
  Creates a scanned function with custom VJP for pipeline iterations.
  Refactored to take lightweight_state, bsw, and weights separately.
  """

  @functools.partial(jax.custom_vjp)
  def step_fn_custom(lightweight_state, bsw, weights, pos_arg, seg_arg):
    s = {**lightweight_state, "bsw": bsw, "weights": weights}
    out = model.run_one_iteration(
        s, pos_arg, seg_arg, deterministic, model_mode, logical_partition_spec=logical_partition_spec
    )
    return {k: v for k, v in out.items() if k not in ["bsw", "weights"]}, out["bsw"], out["weights"]

  def step_fn_custom_fwd(lightweight_state, bsw, weights, pos_arg, seg_arg):
    def _run(l, b, w):
      s = {**l, "bsw": b, "weights": w}
      out = model.run_one_iteration(
          s, pos_arg, seg_arg, deterministic, model_mode, logical_partition_spec=logical_partition_spec
      )
      return {k: v for k, v in out.items() if k not in ["bsw", "weights"]}, out["bsw"], out["weights"]

    _run_remat = jax.checkpoint(_run, prevent_cse=False, policy=model.get_pipeline_remat_policy())
    out, vjp_fun = jax.vjp(_run_remat, lightweight_state, bsw, weights)
    return out, vjp_fun

  def step_fn_custom_bwd(res, g_out):
    vjp_fun = res
    d_l, d_b, d_w = vjp_fun(g_out)
    return d_l, d_b, d_w, None, None

  step_fn_custom.defvjp(step_fn_custom_fwd, step_fn_custom_bwd)

  # Refactored: Now strictly takes separated states
  @functools.partial(jax.custom_vjp)
  def run_scanned_custom(lightweight_state, bsw, weights, positions, segment_ids):
    return run_scanned_custom_fwd(lightweight_state, bsw, weights, positions, segment_ids)[0]

  def run_scanned_custom_fwd(lightweight_state, bsw, weights, positions, segment_ids):
    final_lightweight, scan_vjp_fun = jax.vjp(
        lambda l, b, w: jax.lax.scan(
            lambda carry, _: (step_fn_custom(carry, b, w, positions, segment_ids)[0], None), l, None, length=length
        )[0],
        lightweight_state,
        bsw,
        weights,
    )
    # Return separated to keep the signature clean for the outer VJP
    return (final_lightweight, bsw, weights), scan_vjp_fun

  def run_scanned_custom_bwd(residuals, g_out):
    scan_vjp_fun = residuals
    g_lightweight, g_bsw, g_weights = g_out
    d_init_lightweight, d_init_bsw, d_init_weights = scan_vjp_fun(g_lightweight)

    # Accumulate any gradient signals passed down directly into bsw/weights
    d_init_bsw = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_bsw, g_bsw)
    d_init_weights = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_weights, g_weights)

    return (d_init_lightweight, d_init_bsw, d_init_weights, None, None)

  run_scanned_custom.defvjp(run_scanned_custom_fwd, run_scanned_custom_bwd)
  return run_scanned_custom


def create_run_scannable(
    model,
    run_iteration_scannable,
    length,
    deterministic,
    model_mode,
    logical_partition_spec,
    physical_partition_spec,
    positions,
    segment_ids,
):
  """Creates a scannable function for pipeline loop iterations."""

  @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
  def run_scannable(model, loop_state):
    weights = loop_state["weights"]
    loop_iteration = loop_state["loop_iteration"]
    bsw_old = loop_state["bsw"][1]

    new_bsw = model.bsw_all_gather(weights, physical_partition_spec, loop_iteration)
    bsw_pair = (bsw_old, new_bsw)
    lightweight_state = {k: v for k, v in loop_state.items() if k not in ["bsw", "weights"]}

    if model.config.scan_pipeline_iterations:
      run_scanned_custom = create_scanned_function(
          model=model,
          length=length,
          deterministic=deterministic,
          model_mode=model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      out_lightweight, out_bsw, out_weights = run_scanned_custom(
          lightweight_state, bsw_pair, weights, positions, segment_ids
      )
      out_loop_state = {**out_lightweight, "bsw": out_bsw, "weights": out_weights}
    else:
      temp_loop_state = {**lightweight_state, "bsw": bsw_pair, "weights": weights}
      for _ in range(length):
        temp_loop_state, _ = run_iteration_scannable(model, temp_loop_state)
      out_loop_state = temp_loop_state

    return out_loop_state, None

  def run_scannable_fwd(model, loop_state):
    weights = loop_state["weights"]
    loop_iteration = loop_state["loop_iteration"]
    bsw_old = loop_state["bsw"][1]

    new_bsw = model.bsw_all_gather(weights, physical_partition_spec, loop_iteration)
    bsw_pair = (bsw_old, new_bsw)
    lightweight_state = {k: v for k, v in loop_state.items() if k not in ["bsw", "weights"]}

    run_scanned_custom = create_scanned_function(
        model=model,
        length=length,
        deterministic=deterministic,
        model_mode=model_mode,
        logical_partition_spec=logical_partition_spec,
    )

    # Execute primal and capture the vjp function inside the forward pass
    (out_lightweight, _, _), vjp_fn = jax.vjp(
        run_scanned_custom, lightweight_state, bsw_pair, weights, positions, segment_ids
    )

    out_loop_state = {**out_lightweight, "bsw": bsw_pair, "weights": weights}

    # Pack vjp_fn directly into residuals along with original inputs needed for bsw_reduce_scatter
    residuals = (vjp_fn, weights, loop_iteration)
    return (out_loop_state, None), residuals

  def run_scannable_bwd(model, residuals, g_out):
    vjp_fn, weights, loop_iteration = residuals
    g_loop_state, _ = g_out

    g_lightweight = {k: v for k, v in g_loop_state.items() if k not in ["bsw", "weights"]}
    g_bsw_pair = g_loop_state.get("bsw")
    g_weights = g_loop_state.get("weights")

    # Execute the inner backward pass using the vjp_fn captured in fwd
    d_lightweight, d_bsw_pair, d_weights, _, _ = vjp_fn((g_lightweight, g_bsw_pair, g_weights))
    d_bsw_old, d_bsw_new = d_bsw_pair

    # Use jax.linear_transpose to perfectly reduce-scatter the newly gathered bsw cotangents
    d_weights_from_bsw = model.bsw_reduce_scatter(d_bsw_new, weights, physical_partition_spec, loop_iteration)

    # Combine the weight gradients
    d_weights_total = jax.tree.map(lambda a, b: a + b, d_weights, d_weights_from_bsw)

    # Route gradients for the `bsw` tuple structure (index 0 is a zeroed old-bsw gradient)
    d_bsw_tuple = (jax.tree.map(jnp.zeros_like, d_bsw_old), d_bsw_old)

    # Reconstruct the full loop_state cotangents
    d_loop_state = {**d_lightweight, "bsw": d_bsw_tuple, "weights": d_weights_total}

    return (d_loop_state,)

  run_scannable.defvjp(run_scannable_fwd, run_scannable_bwd)

  # Using nn.remat / jax.checkpoint to govern the outer loop boundary memory
  return nn.remat(
      run_scannable,
      prevent_cse=not model.config.scan_pipeline_iterations,
      policy=model.get_pipeline_remat_policy(),
  )


def create_run_repeats_scanned(run_scannable, length):
  """Creates a scanned function over the pipeline repeats."""
  return nn.scan(
      run_scannable,
      split_rngs={"random": True},
      length=length,
  )
