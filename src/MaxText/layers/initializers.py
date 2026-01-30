"""Initializers."""

from typing import Callable

import jax

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

from MaxText import max_logging
from MaxText.common_types import Array, DType, Shape, PRNGKey

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = int | tuple[int, ...]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)

default_bias_init = jax.nn.initializers.constant(0.0)


def nd_dense_init(scale, mode, distribution):
  """Creates a variance-scaling initializer with dynamic in/out axes.

  This function is a factory that returns an initializer function. The returned
  function is a wrapper around `jax.nn.initializers.variance_scaling` that
  allows the `in_axis` and `out_axis` to be specified at call time, rather
  than at creation time.

  Args:
    scale: The scaling factor for the variance.
    mode: The mode for variance scaling ('fan_in', 'fan_out', 'fan_avg').
    distribution: The distribution to sample from ('normal', 'uniform', etc.).

  Returns:
    A function that takes a PRNG key, shape, dtype, in_axis, and out_axis,
    and returns an initialized array.
  """

  def init_fn(key, shape, dtype, in_axis, out_axis):
    """Initializes an array using variance scaling with specified axes."""
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn


def variable_to_logically_partitioned(variable: nnx.VariableState):
  """Wraps an NNX variable's value in `nn.LogicallyPartitioned`.

  This function inspects the metadata of an `nnx.VariableState` object. If
  sharding information ('sharding' or 'sharding_names') is present, it wraps
  the variable's value in `nn.LogicallyPartitioned` to apply the specified
  sharding constraints.

  It handles special cases for `aqt_tensor.QTensor` and variables of type
  `_overwrite_with_gradient` by returning their values directly without
  wrapping.

  Args:
    variable: The `nnx.VariableState` object to process.

  Returns:
    The variable's value, potentially wrapped in `nn.LogicallyPartitioned`.
  """
  val = variable.value

  # --- DEBUG LOGGING ---
  # Check if this looks like the problematic tensor (Rank 4, Stage dim size 2)
  is_target = False
  if hasattr(val, 'shape') and len(val.shape) == 4 and val.shape[1] == 2:
      is_target = True
  elif isinstance(val, nn.spmd.LogicallyPartitioned) and len(val.value.shape) == 4:
      is_target = True

  if is_target:
      max_logging.log("-" * 50)
      max_logging.log(f"DEBUG: variable_to_logically_partitioned hit Rank 4 tensor")
      max_logging.log(f"  Type of value: {type(val)}")
      
      metadata = variable.get_metadata()
      if 'sharding' in metadata:
          max_logging.log(f"  Metadata['sharding']: {metadata['sharding']}")
      if 'sharding_names' in metadata:
          max_logging.log(f"  Metadata['sharding_names']: {metadata['sharding_names']}")
  # ---------------------

  if isinstance(variable.value, aqt_tensor.QTensor):
    return variable.value

  # If the value is already explicitly partitioned (e.g. from nnx_pipeline),
  # return it as-is.
  if isinstance(variable.value, nn.spmd.LogicallyPartitioned):
    if is_target:
        max_logging.log("DEBUG: Returning existing LogicallyPartitioned value.")
    return variable.value

  if variable.type.__name__ == "_overwrite_with_gradient":
    return variable.value

  metadata = variable.get_metadata()
  if "sharding" in metadata or "sharding_names" in metadata:
    if "sharding_names" in metadata:
      sharding_names = metadata["sharding_names"]
    else:
      sharding_names = metadata["sharding"]
    
    # --- Auto-Patching for Pipeline Expansion ---
    # If the value rank is greater than the sharding rank, it implies pipeline expansion
    # occurred (broadcasting), but the metadata is stale. We patch the spec here.
    if hasattr(val, 'ndim') and isinstance(sharding_names, tuple):
        val_rank = val.ndim
        spec_rank = len(sharding_names)
        diff = val_rank - spec_rank
        
        if diff > 0:
            if is_target:
                max_logging.log(f"DEBUG: Rank mismatch detected (Val: {val_rank}, Spec: {spec_rank}). Patching spec.")
            
            # Prepend axes based on rank difference
            # Diff 2: [Repeats, Stage, ...] -> ('circular_repeats', 'activation_stage', ...)
            # Diff 1: [Stage, ...] -> ('activation_stage', ...)
            if diff == 2:
                sharding_names = ('circular_repeats', 'activation_stage') + sharding_names
            elif diff == 1:
                sharding_names = ('activation_stage',) + sharding_names
            
            if is_target:
                max_logging.log(f"DEBUG: New sharding_names: {sharding_names}")
    # ---------------------------------------------

    return nn.LogicallyPartitioned(  # type: ignore[wrong-keyword-args]
        variable.value,
        sharding_names,  # type: ignore[arg-type]
        mesh=metadata.get("mesh"),
        rules=metadata.get("rules"),
    )
  else:
    return variable.value