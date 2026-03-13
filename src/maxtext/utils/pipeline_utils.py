# Copyright 2023–2026 Google LLC
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


def get_mesh_axis_dim_indices(physical_partition_spec, axis_name="fsdp"):
  """Finds the tensor dimension index sharded across a specific physical mesh axis.

  In JAX sharding, a PartitionSpec maps tensor dimensions to physical device mesh axes.
  This utility traverses a PyTree of PartitionSpecs and returns the integer index of the
  tensor dimension that is mapped to the target `axis_name` (e.g., finding which dimension
  is FSDP-sharded to prepare for an all-gather operation).

  Args:
    physical_partition_spec: A PyTree where leaves are `jax.sharding.PartitionSpec` objects.
    axis_name: The physical mesh axis string to search for (defaults to "fsdp").

  Returns:
    A PyTree of the exact same structure where the leaves are integers representing the
    dimension index of the target axis, or -1 if the axis is not found in that spec.
  """

  def find_axis_index(pspec):
    if pspec is None:
      return -1

    for i, axis in enumerate(pspec):
      # Handle compound mesh axes (e.g., when a dimension is sharded over ('fsdp', 'tensor'))
      if isinstance(axis, (list, tuple)):
        if axis_name in axis:
          return i
      elif axis == axis_name:
        return i
    return -1

  return jax.tree.map(find_axis_index, physical_partition_spec)


def derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove):
  """Derives the physical partition specs for weights inside the scanned pipeline loop.

  When weights enter the inner `jax.lax.scan` loop for microbatch execution, their
  sharding requirements change. This function modifies the base weight specs by:
  1. Removing physical axes that will be all-gathered during the forward pass
     (e.g., FSDP axes, and Expert axes outside of the routed MoE block).
  2. Slicing off the first dimension `[1:]`, which typically represents the
     outer pipeline stage or layer-repeat dimension that the scan operates over.

  Args:
    physical_partition_spec: A PyTree of `PartitionSpec` objects for the full model weights.
    axes_to_remove: list of physical axes to remove.

  Returns:
    A PyTree of `PartitionSpec` objects tailored for the inner scanned execution block.
  """

  def _process_pps(path, pps):
    # Safely extract string keys from the JAX KeyPath elements to identify the layer type
    path_keys = [getattr(p, "key", str(p)) for p in path]
    is_moe_block_0 = "MoeBlock_0" in path_keys

    processed_pps = remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove=axes_to_remove)

    # Drop the first dimension (usually the 'stage' or 'layer' axis handled by the scan)
    return P(*processed_pps[1:])

  return jax.tree_util.tree_map_with_path(
      _process_pps,
      physical_partition_spec,
  )


def remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove):
  """Strips FSDP and specific MoE mesh axes from a PartitionSpec.

  When FSDP or Expert-sharded weights are all-gathered for computation, the resulting
  tensor is no longer sharded across those physical mesh dimensions. This function
  removes those axes from the PartitionSpec, replacing them with `None` to indicate
  replication across that mesh dimension.

  Args:
    pps: A single `jax.sharding.PartitionSpec` object.
    is_moe_block_0: Boolean indicating if the target is the routed MoE block. The 'expert'
                    mesh axis is only retained for the routed block.
    axes_to_remove: physical axes that we should remove from current physical partition axes

  Returns:
    A new `PartitionSpec` with the gathered axes removed, or the original object if it
    was not a PartitionSpec.
  """
  if not is_moe_block_0:
    axes_to_remove.append("expert")

  if isinstance(pps, P):
    new_spec = []
    for axis in pps:
      if axis is None:
        new_spec.append(None)
      elif isinstance(axis, str):
        if axis not in axes_to_remove:
          new_spec.append(axis)
        else:
          new_spec.append(None)  # None signifies replication across the removed mesh axis
      elif isinstance(axis, (list, tuple)):
        new_axis = [a for a in axis if a not in axes_to_remove]
        new_spec.append(tuple(new_axis))
      else:
        raise ValueError(f"Unsupported_axis_type: {type(axis)}")

    return P(*new_spec)

  return pps


def strip_pipeline_repeat_logical_axis(full_logical_spec):
  """Removes 'circular_repeats' from a logical PartitionSpec PyTree.

  Args:
    full_logical_spec: A PyTree of logical PartitionSpecs (strings like 'vocab', 'embed').

  Returns:
    A PyTree with 'circular_repeats' filtered out of all logical partition tuples.
  """
  if full_logical_spec is None:
    return None

  def _remove_from_spec(spec):
    return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

  return jax.tree.map(_remove_from_spec, full_logical_spec)


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


def create_gradient_accumulation_scan(
    model,
    length,
    deterministic=True,
    model_mode=None,
    logical_partition_spec=None,
):
  """Creates a memory-efficient `jax.lax.scan` loop for pipeline microbatches with a custom VJP.

  In pipeline parallelism, scanning over microbatches normally forces JAX to save
  heavy parameter states for every iteration to compute the backward pass. This helper
  defines a custom Vector-Jacobian Product (VJP) to solve this by:

  1. Forward pass: Separating transient activations (`lightweight_state`) from heavy
     parameters (`bsw` and `weights`).
  2. Rematerialization: Recomputing the forward pass of individual steps during the backward
     pass to save memory.
  3. Backward pass: Manually accumulating gradients (`d + g`) onto the heavy parameters
     across the scanned iterations, rather than letting the standard autodiff trace them linearly.

  Args:
    model: The model instance containing the `run_one_iteration` and rematerialization logic.
    length: The number of microbatch iterations to scan over.
    deterministic: Whether to run the model in a deterministic mode (e.g., disable dropout).
    model_mode: The operational mode of the model (e.g., 'train', 'eval').
    logical_partition_spec: Rules for logical partitioning in standard tensor parallelism.

  Returns:
    A JAX custom_vjp function that executes the `length` pipeline iterations.
  """

  @functools.partial(jax.custom_vjp)
  def run_single_microbatch_custom(lightweight_state, bsw, weights, pos_arg, seg_arg):
    return run_single_microbatch_custom_fwd(lightweight_state, bsw, weights, pos_arg, seg_arg)[0]

  def run_single_microbatch_custom_fwd(lightweight_state, bsw, weights, pos_arg, seg_arg):
    def _run(l, b, w):
      out = model.run_one_iteration(
          l, b, w, pos_arg, seg_arg, deterministic, model_mode, logical_partition_spec=logical_partition_spec
      )
      return out, b, w

    # Rematerialize the inner step to save activation memory
    _run_remat = jax.remat(_run, prevent_cse=False, policy=model.get_pipeline_remat_policy())
    out, vjp_fun = jax.vjp(_run_remat, lightweight_state, bsw, weights)
    return out, vjp_fun

  def run_single_microbatch_custom_bwd(res, g_out):
    vjp_fun = res
    d_l, d_b, d_w = vjp_fun(g_out)
    return d_l, d_b, d_w, None, None

  run_single_microbatch_custom.defvjp(run_single_microbatch_custom_fwd, run_single_microbatch_custom_bwd)

  @functools.partial(jax.custom_vjp)
  def run_pipeline_microbatches_custom(loop_state, bsw, weights, positions, segment_ids):
    return run_pipeline_microbatches_custom_fwd(loop_state, bsw, weights, positions, segment_ids)[0]

  def run_pipeline_microbatches_custom_fwd(loop_state, bsw, weights, positions, segment_ids):
    final_lightweight, scan_vjp_fun = jax.vjp(
        lambda l, b, w: jax.lax.scan(
            lambda carry, _: (run_single_microbatch_custom(carry, b, w, positions, segment_ids)[0], None),
            l,
            None,
            length=length,
        )[0],
        loop_state,
        bsw,
        weights,
    )

    return (final_lightweight, bsw, weights), scan_vjp_fun

  def run_pipeline_microbatches_custom_bwd(residuals, g_final_state):
    scan_vjp_fun = residuals
    g_lightweight, g_bsw, g_weights = g_final_state
    d_init_lightweight, d_init_bsw, d_init_weights = scan_vjp_fun(g_lightweight)

    d_init_bsw = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_bsw, g_bsw)
    d_init_weights = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_weights, g_weights)

    return (d_init_lightweight, d_init_bsw, d_init_weights, None, None)

  run_pipeline_microbatches_custom.defvjp(run_pipeline_microbatches_custom_fwd, run_pipeline_microbatches_custom_bwd)
  return run_pipeline_microbatches_custom


def create_rematerialized_pipeline_stage(
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
  """Builds a memory-checkpointed execution block for a single pipeline stage.

  This function prepares the state for a specific chunk of pipeline execution by:
  1. Prefetching the required weights for the current stage/loop iteration.
  2. Executing `length` microbatches using either a memory-efficient `jax.lax.scan`
     (if `scan_pipeline_iterations` is True) or an unrolled Python `for` loop.
  3. Wrapping the entire stage block in `flax.linen.remat` to discard and recompute
     activations during the backward pass based on the model's policy.

  Args:
    model: The model instance containing configuration and prefetching logic.
    run_iteration_scannable: A fallback function for executing a single iteration unrolled.
    length: The number of microbatches to process in this stage.
    deterministic: Whether to run deterministically (e.g., disable dropout).
    model_mode: The operational mode (e.g., 'train').
    logical_partition_spec: Rules for logical tensor sharding.
    physical_partition_spec: Rules for physical device mesh mappings (used in prefetching).
    positions: Position IDs for the sequence.
    segment_ids: Segment/Attention routing IDs for the sequence.

  Returns:
    A function decorated with `nn.remat` that takes `(model, loop_state)` and returns
    the updated `loop_state`.
  """

  def execute_pipeline_stage(model, loop_state_and_bsw_and_weights):
    loop_state, bsw, weights = loop_state_and_bsw_and_weights
    # Retrieve the specific weights needed for this pipeline chunk
    bsw = model.weight_prefetching(weights, physical_partition_spec, loop_state["loop_iteration"])

    if model.config.scan_pipeline_iterations:
      scan_microbatches_fn = create_gradient_accumulation_scan(
          model=model,
          length=length,
          deterministic=deterministic,
          model_mode=model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      loop_state, bsw, weights = scan_microbatches_fn(loop_state, bsw, weights, positions, segment_ids)
    else:
      for _ in range(length):
        (loop_state, bsw, weights), _ = run_iteration_scannable(model, loop_state, bsw, weights)
    return (loop_state, bsw, weights), None

  return nn.remat(
      execute_pipeline_stage,
      prevent_cse=not model.config.scan_pipeline_iterations,
      policy=model.get_pipeline_remat_policy(),
  )


def create_flax_pipeline_scan(pipeline_stage_fn, length):
  """Wraps the pipeline stage execution in a `flax.linen.scan`.

  This lifts the pipeline stage function so it can be repeated sequentially over
  the specified length. It safely handles Flax-specific state collections, ensuring
  that metrics, intermediate values, and PRNG keys do not collide or overwrite
  each other across the loop iterations.

  Args:
    pipeline_stage_fn: The function representing a single pipeline stage
                       (usually created by `create_rematerialized_pipeline_stage`).
    length: The total number of pipeline stages/repeats to scan over.

  Returns:
    A Flax scanned function that executes the full pipeline schedule.
  """
  return nn.scan(
      pipeline_stage_fn,
      variable_axes={
          "summaries": 0,
          "aux_loss": 0,
          "intermediates": 0,
          "hyper_params": 0,
      },
      split_rngs={"random": True},
      length=length,
  )
