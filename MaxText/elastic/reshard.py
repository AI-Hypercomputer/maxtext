# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resharding API for elastic training."""

from typing import Any
from typing import Callable, Optional, Sequence
import jax


def default_put_array(
    arr: jax.Array,
    dst_sharding: jax.sharding.Sharding,
    donate_input: bool,
):
  if not isinstance(dst_sharding, jax.sharding.Sharding):
    raise ValueError("`sharding` must contain only `Sharding` instances.")
  return jax.device_put(arr, dst_sharding, donate=donate_input)


def reshard(
    x: Any,
    sharding: jax.sharding.Sharding | Any,
    *,
    donate_input: bool = True,
    put_array: Optional[
        Callable[[jax.Array, Sequence[jax.sharding.Sharding], bool], jax.Array]
    ] = None,
) -> Any:
  """Reshards `x` to the specified `sharding`.

  Args:
      x: An array, scalar, or a nested Python container thereof.
      sharding: A `Sharding` or a nested `Sharding` in a Python container (must
        match the structure of `x`), specifying the target sharding.
      donate_input: If `True`, donates the input arrays to reduce memory needed
        for resharding. Donated buffers should not be reused.
      put_array: A function that takes an array, a sharding, and a boolean
        indicating whether to donate the input, and returns a copy of the array
        with the specified sharding.

  Returns:
      A copy of `x` with the specified `sharding`.
  """
  if put_array is None:
    put_array = default_put_array

  flat_x, tree_def = jax.tree_util.tree_flatten(x)
  flat_sharding = jax.api_util.flatten_axes(
      "reshard sharding", tree_def, sharding
  )

  if len(flat_x) != len(flat_sharding):
    raise ValueError("Mismatched length between `x` and `sharding`.")

  arrays = [
      put_array(arr, dst_sharding, donate_input)
      for arr, dst_sharding in zip(flat_x, flat_sharding)
  ]
  return jax.tree_util.tree_unflatten(tree_def, arrays)

