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
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned


def get_fsdp_index_pytree(physical_partition_spec):
  """
  Finds the index of 'fsdp' within each PartitionSpec in a Pytree.

  Args:
    physical_partition_spec: A Pytree where leaves are PartitionSpecs.

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
        if "fsdp" in axis:
          return i
      elif axis == "fsdp":
        return i
    return -1

  return jax.tree.map(find_fsdp, physical_partition_spec)


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
    run_iteration_scannable,
    length,
    variable_axes=None,
    split_rngs=None,
    deterministic=True,
    model_mode=None,
    logical_partition_spec=None,
):
  """
  Creates a scanned function with custom VJP for pipeline iterations.

  This helper encapsulates the logic for:
  1. Creating the forward scan (saving lightweight state)
  2. Creating the backward scan (recomputing heavy state and accumulating gradients)
  3. Registering the custom VJP
  """
  if variable_axes is None:
    variable_axes = {
        "summaries": 0,
        "aux_loss": 0,
        "intermediates": 0,
        "hyper_params": 0,
    }

  if split_rngs is None:
    split_rngs = {"random": True}

  run_scanned = nn.scan(
      run_iteration_scannable,
      variable_axes=variable_axes,
      split_rngs=split_rngs,
      length=length,
  )

  @functools.partial(jax.custom_vjp)
  def run_scanned_custom(loop_state, positions, segment_ids):
    final_state, _ = run_scanned(model, loop_state)
    return final_state

  def run_scanned_custom_fwd(loop_state, positions, segment_ids):
    def step_fn(lightweight_state, bsw, weights):
      # Reconstruct full state for the model
      s = {**lightweight_state, "bsw": bsw, "weights": weights}
      out = model.run_one_iteration(
          s,
          positions,
          segment_ids,
          deterministic,
          model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      # Deconstruct back to lightweight to decouple gradients
      new_lightweight = {k: v for k, v in out.items() if k not in ["bsw", "weights"]}
      return new_lightweight, out["bsw"], out["weights"]

    step_fn_remat = jax.remat(step_fn, prevent_cse=False, policy=model.get_pipeline_remat_policy())

    # Separate heavy and light state initially
    initial_lightweight = {k: v for k, v in loop_state.items() if k not in ["bsw", "weights"]}
    initial_bsw = loop_state["bsw"]
    initial_weights = loop_state["weights"]

    def scan_body_fwd(carry, _):
      lightweight_carry, bsw_carry, weights_carry = carry
      (new_lightweight, new_bsw, new_weights), vjp_fun = jax.vjp(
          step_fn_remat, lightweight_carry, bsw_carry, weights_carry
      )
      return (new_lightweight, new_bsw, new_weights), vjp_fun

    (final_lightweight, final_bsw, final_weights), vjp_funs = jax.lax.scan(
        scan_body_fwd,
        (initial_lightweight, initial_bsw, initial_weights),
        None,
        length=length,
    )

    final_state = {**final_lightweight, "bsw": final_bsw, "weights": final_weights}
    return final_state, vjp_funs

  def run_scanned_custom_bwd(residuals, g_final_state):
    vjp_funs = residuals

    # Split the gradient of the final state
    g_lightweight = {k: v for k, v in g_final_state.items() if k not in ["bsw", "weights"]}
    g_bsw = g_final_state["bsw"]
    g_weights = g_final_state["weights"]

    def scan_body_bwd(carry, vjp_fun):
      d_next_lightweight, d_next_bsw, d_next_weights = carry

      # Apply saved vjp_fun directly
      d_curr_lightweight, d_curr_bsw, d_curr_weights = vjp_fun((d_next_lightweight, d_next_bsw, d_next_weights))

      # Accumulate gradients for invariant parts
      d_bsw_accum = jax.tree.map(lambda x, y: x + y, d_next_bsw, d_curr_bsw)
      d_weights_accum = jax.tree.map(lambda x, y: x + y, d_next_weights, d_curr_weights)

      return (d_curr_lightweight, d_bsw_accum, d_weights_accum), None

    # Run backward scan
    (d_init_lightweight, d_init_bsw, d_init_weights), _ = jax.lax.scan(
        scan_body_bwd,
        (g_lightweight, g_bsw, g_weights),
        vjp_funs,
        reverse=True,
    )

    d_init_state = {**d_init_lightweight, "bsw": d_init_bsw, "weights": d_init_weights}
    return (d_init_state, None, None)

  run_scanned_custom.defvjp(run_scanned_custom_fwd, run_scanned_custom_bwd)
  return run_scanned_custom
