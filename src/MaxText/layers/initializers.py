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

"""Initializers."""

from typing import Callable

import jax

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

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
  if isinstance(variable.value, aqt_tensor.QTensor):
    return variable.value

  if variable.type.__name__ == "_overwrite_with_gradient":
    return variable.value

  metadata = variable.get_metadata()
  if "sharding" in metadata or "sharding_names" in metadata:
    if "sharding_names" in metadata:
      sharding_names = metadata["sharding_names"]
    else:
      sharding_names = metadata["sharding"]
    return nn.LogicallyPartitioned(  # type: ignore[wrong-keyword-args]
        variable.value,
        sharding_names,  # type: ignore[arg-type]
        mesh=metadata.get("mesh"),
        rules=metadata.get("rules"),
    )
  else:
    return variable.value
