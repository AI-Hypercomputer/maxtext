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

"""Utilities for constructing stacks of scanned NNX layers."""

from collections.abc import Callable

from flax import nnx
import jax
import jax.numpy as jnp


def create_scanned_layers(
    layer_factory: Callable[[nnx.Rngs], nnx.Module],
    *,
    length: int,
    param_scan_axis: int,
    metadata_axis_name: str,
    rngs: nnx.Rngs,
) -> nnx.Module | None:
  """Constructs an NNX layer whose variables are stacked for a layer scan."""
  if length == 0:
    return None

  forked_rngs = rngs.fork(split=length)
  rngs_graphdef, rngs_state = nnx.split(forked_rngs)

  first_rng_state = jax.tree.map(lambda x: x[0], rngs_state)
  reference_layer = layer_factory(nnx.merge(rngs_graphdef, first_rng_state))
  layer_graphdef, _, _ = nnx.split(reference_layer, nnx.Param, ...)
  del reference_layer

  def scan_body(carry, rng_state_slice):
    layer = layer_factory(nnx.merge(rngs_graphdef, rng_state_slice))
    _, params, rest = nnx.split(layer, nnx.Param, ...)
    return carry, (params, rest)

  _, (stacked_params, stacked_rest) = jax.lax.scan(scan_body, None, rngs_state)

  if param_scan_axis != 0:
    stacked_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis), stacked_params)

  def add_scan_metadata(state, axis):
    def update_leaf(leaf):
      if hasattr(leaf, "replace") and hasattr(leaf, "value"):
        replace_kwargs = {}
        if hasattr(leaf, "get_metadata"):
          replace_kwargs.update(leaf.get_metadata())

        replace_kwargs[nnx.PARTITION_NAME] = metadata_axis_name
        replace_kwargs["param_scan_axis"] = axis

        for key in ["sharding", "out_sharding", "kernel_axes", "sharding_names"]:
          value = getattr(leaf, key, None)
          if value is None and key in replace_kwargs:
            value = replace_kwargs[key]

          if value is not None:
            if isinstance(value, str):
              value = (value,)
            if isinstance(value, tuple):
              logical_axes = list(value)
              if metadata_axis_name not in logical_axes:
                logical_axes.insert(min(axis, len(logical_axes)), metadata_axis_name)
                replace_kwargs[key] = tuple(logical_axes)

        return leaf.replace(**replace_kwargs)
      return leaf

    return jax.tree.map(
        update_leaf,
        state,
        is_leaf=lambda x: hasattr(x, "replace") and hasattr(x, "value"),
    )

  stacked_params = add_scan_metadata(stacked_params, param_scan_axis)
  stacked_rest = add_scan_metadata(stacked_rest, 0)
  return nnx.merge(layer_graphdef, stacked_params, stacked_rest)
