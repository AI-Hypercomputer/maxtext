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
