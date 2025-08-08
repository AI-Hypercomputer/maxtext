# SPDX-License-Identifier: Apache-2.0

"""Initializers."""

from typing import Callable, Tuple, Union

import jax

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

from MaxText.common_types import Array, DType, Shape, PRNGKey

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_embed_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)

default_bias_init = jax.nn.initializers.constant(0.0)


def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""

  def init_fn(key, shape, dtype, in_axis, out_axis):
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn


def variable_to_logically_partitioned(variable: nnx.VariableState):
  if isinstance(variable.value, aqt_tensor.QTensor):
    return variable.value

  if variable.type.__name__ == "_overwrite_with_gradient":
    return variable.value

  metadata = variable.get_metadata()
  return nn.LogicallyPartitioned(  # type: ignore[wrong-keyword-args]
      variable.value,
      variable.sharding,  # type: ignore[arg-type]
      mesh=metadata.get("mesh"),
      rules=metadata.get("rules"),
  )
