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

import jax
from jax.sharding import PartitionSpec as P
from flax import nnx


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

  @jax.custom_vjp
  def run_single_microbatch_custom(lightweight_state, bsw, pos_arg, seg_arg):
    return run_single_microbatch_custom_fwd(lightweight_state, bsw, pos_arg, seg_arg)[0]

  def run_single_microbatch_custom_fwd(lightweight_state, bsw, pos_arg, seg_arg):
    def _run(l, b):
      out = model.run_one_iteration(
          l, b, pos_arg, seg_arg, deterministic, model_mode, logical_partition_spec=logical_partition_spec
      )
      return out

    # Rematerialize the inner step to save activation memory
    _run_remat = jax.remat(_run, policy=model.get_pipeline_remat_policy())
    out, vjp_fun = jax.vjp(_run_remat, lightweight_state, bsw)
    return out, vjp_fun

  def run_single_microbatch_custom_bwd(res, g_out):
    vjp_fun = res
    d_l, d_b = vjp_fun(g_out)
    return d_l, d_b, None, None

  run_single_microbatch_custom.defvjp(run_single_microbatch_custom_fwd, run_single_microbatch_custom_bwd)

  @jax.custom_vjp
  def run_pipeline_microbatches_custom(loop_state, bsw, positions, segment_ids):
    return run_pipeline_microbatches_custom_fwd(loop_state, bsw, positions, segment_ids)[0]

  def run_pipeline_microbatches_custom_fwd(loop_state, bsw, positions, segment_ids):
    final_lightweight, scan_vjp_fun = jax.vjp(
        lambda l, b: jax.lax.scan(
            lambda carry, _: (run_single_microbatch_custom(carry, b, positions, segment_ids), None),
            l,
            None,
            length=length,
        )[0],
        loop_state,
        bsw,
    )

    return (final_lightweight, bsw), scan_vjp_fun

  def run_pipeline_microbatches_custom_bwd(residuals, g_final_state):
    scan_vjp_fun = residuals
    g_lightweight, g_bsw = g_final_state
    d_init_lightweight, d_init_bsw = scan_vjp_fun(g_lightweight)

    d_init_bsw = jax.tree.map(lambda d, g: d + g if hasattr(d, "shape") else d, d_init_bsw, g_bsw)

    return (d_init_lightweight, d_init_bsw, None, None)

  run_pipeline_microbatches_custom.defvjp(run_pipeline_microbatches_custom_fwd, run_pipeline_microbatches_custom_bwd)
  return run_pipeline_microbatches_custom


def is_static_param(path, v):
  """Predicate matching nnx.Param and FP8 _overwrite_with_gradient variables.

  Used throughout the pipeline to split state into trainable params vs other state.
  Must be consistent everywhere to prevent tree structure mismatches.
  """
  return isinstance(v, nnx.Param) or type(v).__name__ == "_overwrite_with_gradient"


def advance_rng_state(state, iteration):
  """Fold loop_iteration into all RNG keys to produce unique dropout masks per scan step.

  jax.lax.scan has no split_rngs mechanism (unlike Linen's nn.scan), so every
  iteration would otherwise see the same dropout mask. This mirrors the effect
  of ``nn.scan(split_rngs={"random": True})`` from the Linen pipeline.

  Only typed PRNG key variables (``RngKey``) are folded. RNG counters
  (``RngCount``) are uint32 arrays and must be left untouched -- calling
  ``jax.random.fold_in`` on raw uint32 data triggers a PRNG-impl shape
  mismatch (e.g. shape ``(N, 2)`` vs ``unsafe_rbg`` expecting ``(4,)``).

  Args:
    state: An ``nnx.State`` (or partition thereof) that may contain
        ``nnx.RngState`` variable entries whose ``.value`` is a JAX PRNG key.
    iteration: A scalar integer (the loop counter) folded into each key via
        ``jax.random.fold_in``.

  Returns:
    A new state with the same tree structure, where every typed PRNG key
    entry has a unique key derived from the original key and *iteration*.
  """

  def _fold_if_rng(x):
    if isinstance(x, nnx.Variable) and issubclass(x.type, nnx.RngState):
      val = x.value
      if jax.dtypes.issubdtype(val.dtype, jax.dtypes.prng_key):

        def folded(k):
          return jax.random.fold_in(k, iteration)

        for _ in range(val.ndim):
          folded = jax.vmap(folded)
        return x.replace(value=folded(val))
    return x

  return jax.tree.map(_fold_if_rng, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def is_spec_leaf(x):
  """Predicate matching leaves in the bsw_pps treedef, which can be either P or None (if no sharding)."""
  return isinstance(x, P) or x is None


def flatten_nnx_state(state):
  """Flatten nnx.State to (arrays, treedef, is_var_flags, var_types, var_metadata).

  Returns raw arrays and Python-only metadata for reconstruction.
  var_metadata: list of dicts with Variable field metadata (NO .raw_value).
  Captures: nothing (pure function on inputs).
  """

  def _is_var(x):
    return isinstance(x, nnx.Variable)

  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(state, is_leaf=_is_var)
  arrays = []
  is_var_flags = []
  var_types = []
  var_metadata = []
  for _, leaf in leaves_with_path:
    if isinstance(leaf, nnx.Variable):
      arrays.append(leaf.value)
      is_var_flags.append(True)
      var_types.append(type(leaf))
      var_metadata.append(dict(leaf._var_metadata))  # pylint: disable=protected-access
    else:
      arrays.append(leaf)
      is_var_flags.append(False)
      var_types.append(None)
      var_metadata.append({})
  return arrays, treedef, is_var_flags, var_types, var_metadata


def unflatten_nnx_state(arrays, treedef, is_var_flags, var_types, var_metadata):
  """Reconstruct nnx.State from flattened arrays + metadata.

  Does NOT reference any nnx.Variable objects from the original state.
  var_metadata contains only Python objects (no JAX arrays).
  Captures: nothing (pure function on inputs).
  """
  new_leaves = []
  for arr, is_var, vtype, meta in zip(arrays, is_var_flags, var_types, var_metadata):
    if is_var and vtype is not None:
      new_leaves.append(vtype(arr, **meta))
    else:
      new_leaves.append(arr)
  return treedef.unflatten(new_leaves)


def arrays_to_linen_collection(arrays, keys):
  """Convert list of arrays + key names to a Linen-style flat dict.

  Captures: nothing (pure function).
  """
  return dict(zip(keys, arrays))


def linen_collection_to_arrays(collection, keys):
  """Extract arrays from Linen-style flat dict in key order.

  Captures: nothing (pure function).
  """
  return [collection[k] for k in keys]


# ---------------------------------------------------------------------------
# Non-pytree context: holds Python-only attributes from the pipeline module.
# NNX modules are JAX pytrees — capturing them in lift.scan closures leaks
# JIT-level tracers from self.layers. This wrapper is NOT a JAX pytree,
# so JAX never tries to flatten it.
# ---------------------------------------------------------------------------


class PipelineContext:
  """Non-pytree wrapper holding pipeline methods + Python config.

  Created from an NNXCircularPipeline ONCE before entering transforms.
  Captures ONLY bound methods (which internally access config/mesh/Python attrs)
  and Python objects. No nnx.Variable or JAX arrays.
  """

  __slots__ = (
      "weight_prefetching",
      "run_one_iteration",
      "from_all_variables_to_repeat_weights",
      "from_repeat_weights_to_bsw",
  )

  def __init__(self, pipeline_module):
    self.weight_prefetching = pipeline_module.weight_prefetching
    self.run_one_iteration = pipeline_module.run_one_iteration
    self.from_all_variables_to_repeat_weights = pipeline_module.from_all_variables_to_repeat_weights
    self.from_repeat_weights_to_bsw = pipeline_module.from_repeat_weights_to_bsw
