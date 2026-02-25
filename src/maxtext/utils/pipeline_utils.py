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

    _run_remat = jax.remat(_run, prevent_cse=False, policy=model.get_pipeline_remat_policy())
    out, vjp_fun = jax.vjp(_run_remat, lightweight_state, bsw, weights)

    return out, vjp_fun

  def step_fn_custom_bwd(res, g_out):
    vjp_fun = res
    d_l, d_b, d_w = vjp_fun(g_out)
    return d_l, d_b, d_w, None, None

  step_fn_custom.defvjp(step_fn_custom_fwd, step_fn_custom_bwd)

  def run_scanned_custom_fwd(loop_state, positions, segment_ids):
    initial_lightweight = {k: v for k, v in loop_state.items() if k not in ["bsw", "weights"]}
    bsw = loop_state["bsw"]
    weights = loop_state["weights"]

    def scan_body_fwd(carry, _):
      lightweight_carry = carry
      new_lightweight, _, _ = step_fn_custom(lightweight_carry, bsw, weights, positions, segment_ids)
      return new_lightweight, None

    final_lightweight, _ = jax.lax.scan(scan_body_fwd, initial_lightweight, None, length=length)

    final_state = {**final_lightweight, "bsw": bsw, "weights": weights}
    return final_state, (loop_state, positions, segment_ids)

  def run_scanned_custom_bwd(residuals, g_final_state):
    init_loop_state, positions, segment_ids = residuals

    initial_lightweight = {k: v for k, v in init_loop_state.items() if k not in ["bsw", "weights"]}
    bsw = init_loop_state["bsw"]
    weights = init_loop_state["weights"]

    # We evaluate jax.vjp on the scan so it utilizes the inner step_fn_custom.
    _, scan_vjp_fun = jax.vjp(
        lambda l, b, w: jax.lax.scan(
            lambda carry, _: (step_fn_custom(carry, b, w, positions, segment_ids)[0], None), l, None, length=length
        )[0],
        initial_lightweight,
        bsw,
        weights,
    )

    g_lightweight = {k: v for k, v in g_final_state.items() if k not in ["bsw", "weights"]}

    d_init_lightweight, d_init_bsw, d_init_weights = scan_vjp_fun(g_lightweight)

    d_init_bsw = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_bsw, g_final_state["bsw"])
    d_init_weights = jax.tree.map(
        lambda d, g: d + g if hasattr(d, "shape") else d, d_init_weights, g_final_state["weights"]
    )

    d_init_state = {**d_init_lightweight, "bsw": d_init_bsw, "weights": d_init_weights}
    return (d_init_state, None, None)

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

  def run_scannable(model, loop_state):
    loop_state["bsw"] = model.bsw_all_gather_over_fsdp(
        loop_state["weights"], physical_partition_spec, loop_state["loop_iteration"]
    )

    if model.config.scan_pipeline_iterations:
      run_scanned_custom = create_scanned_function(
          model=model,
          run_iteration_scannable=run_iteration_scannable,
          length=length,
          variable_axes={
              "summaries": 0,
              "aux_loss": 0,
              "intermediates": 0,
              "hyper_params": 0,
          },
          split_rngs={"random": True},
          deterministic=deterministic,
          model_mode=model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      loop_state = run_scanned_custom(loop_state, positions, segment_ids)
    else:
      for _ in range(length):
        loop_state, _ = run_iteration_scannable(model, loop_state)
    return loop_state, None

  return nn.remat(
      run_scannable,
      prevent_cse=not model.config.scan_pipeline_iterations,
      policy=model.get_pipeline_remat_policy(),
  )


def create_run_repeats_scanned(run_scannable, length):
  """Creates a scanned function over the pipeline repeats."""
  return nn.scan(
      run_scannable,
      variable_axes={
          "summaries": 0,
          "aux_loss": 0,
          "intermediates": 0,
          "hyper_params": 0,
      },
      split_rngs={"random": True},
      length=length,
  )
