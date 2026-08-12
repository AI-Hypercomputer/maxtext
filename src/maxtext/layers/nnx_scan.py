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

"""Utilities for constructing and applying stacks of scanned NNX layers."""

from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.utils import max_utils


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

  forked_rngs = rngs.split(length) if hasattr(rngs, "split") else rngs.fork(split=length)
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
      if isinstance(leaf, nnx.Variable) and hasattr(leaf, "replace"):
        replace_kwargs = {}
        if hasattr(leaf, "get_metadata"):
          replace_kwargs.update(leaf.get_metadata())

        replace_kwargs[nnx.PARTITION_NAME] = metadata_axis_name
        replace_kwargs["param_scan_axis"] = axis

        for key in ["sharding", "out_sharding", "kernel_axes"]:
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
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  stacked_params = add_scan_metadata(stacked_params, param_scan_axis)
  stacked_rest = add_scan_metadata(stacked_rest, 0)
  return nnx.merge(layer_graphdef, stacked_params, stacked_rest)


def apply_scanned_layers(
    layers: nnx.Module,
    carry: Any,
    *,
    length: int,
    param_scan_axis: int,
    apply_fn: Callable[[nnx.Module, Any], Any],
    remat: bool = False,
    remat_policy: Callable[..., Any] | None = None,
    prevent_cse: bool = True,
    unroll: int = 1,
    parameter_memory_host_offload: bool = False,
    parameter_memory_two_layer_buffer: bool = False,
) -> Any:
  """Applies stacked NNX layers using ``jax.lax.scan``.

  This helper owns the generic NNX state and scan-axis mechanics. ``apply_fn``
  defines the model-specific module invocation and must return the next carry.
  ``remat`` is separate from ``remat_policy`` because ``None`` is JAX's full
  rematerialization policy, not an indication that rematerialization is off.

  Externally managed per-layer state, such as KV caches, is not supported by
  this scan path.

  Note:
    This is a minimal, model-agnostic scan primitive. The other NNX decoder
    paths instead go through ``NNXDecoder._apply_layers_sequentially``, a heavier
    applier that also threads external (vLLM) KV caches via a static unroll and
    re-applies scan-axis metadata. Gemma4 is currently the only caller of this
    function; unifying the two appliers is a follow-up cleanup.
  """
  if length <= 0:
    return carry

  layer_graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  if param_scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, param_scan_axis, 0), params)

  def scan_body(current_carry, scanned_state):
    current_params, current_state = scanned_state
    if parameter_memory_host_offload or parameter_memory_two_layer_buffer:
      def move_param_to_device(param, outer_param):
        param_dev = max_utils.to_device(param)
        sharding = getattr(param, "sharding", None)
        if sharding is None and hasattr(param, "aval"):
          sharding = getattr(param.aval, "sharding", None)
        if sharding is None:
          sharding = getattr(outer_param, "sharding", None)
          if sharding is None:
            val = outer_param.get_value() if isinstance(outer_param, nnx.Variable) else getattr(outer_param, "value", outer_param)
            sharding = getattr(val, "sharding", getattr(getattr(val, "aval", None), "sharding", None))
        if hasattr(sharding, "with_memory_kind"):
          mesh = getattr(sharding, "mesh", None)
          if mesh is not None and not getattr(mesh, "empty", False) and bool(getattr(mesh, "shape", None)):
            spec = getattr(sharding, "spec", None)
            ndim = getattr(param, "ndim", len(param.shape) if hasattr(param, "shape") else None)
            if spec is not None and ndim is not None:
              if len(spec) > ndim:
                target_spec = jax.sharding.PartitionSpec(*spec[-ndim:])
              elif len(spec) == ndim:
                target_spec = spec
              else:
                target_spec = jax.sharding.PartitionSpec(*(spec + (None,) * (ndim - len(spec))))
              target_sharding = jax.sharding.NamedSharding(mesh, target_spec, memory_kind="device")
            else:
              target_sharding = sharding.with_memory_kind("device")
            return jax.lax.with_sharding_constraint(param_dev, target_sharding)
        return param_dev

      current_params = jax.tree.map(
          move_param_to_device,
          current_params,
          params,
          is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.ShapeDtypeStruct)) or (hasattr(x, "shape") and hasattr(x, "dtype")),
      )
    current_layer = nnx.merge(layer_graphdef, current_params, current_state)
    next_carry = apply_fn(current_layer, current_carry)
    rng_filters = tuple(f for f in (getattr(nnx, "RngCount", None), getattr(nnx, "RngKey", None), getattr(nnx, "Intermediate", None)) if f is not None)
    if rng_filters:
      non_param_state = nnx.state(current_layer, (nnx.Not(nnx.Param), *(nnx.Not(f) for f in rng_filters)))
    else:
      non_param_state = nnx.state(current_layer, nnx.Not(nnx.Param))
    return next_carry, non_param_state

  scan_fn = jax.checkpoint(scan_body, policy=remat_policy, prevent_cse=prevent_cse) if remat else scan_body
  final_carry, scanned_state = jax.lax.scan(scan_fn, carry, (params, state), unroll=unroll)

  if bool(scanned_state):
    if param_scan_axis != 0:
      if hasattr(nnx, "split_state"):
        scanned_params, scanned_other = nnx.split_state(scanned_state, nnx.Param, ...)
      else:
        scanned_params, scanned_other = scanned_state.split(nnx.Param, ...)
      scanned_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis), scanned_params)
      if hasattr(nnx, "merge_state"):
        scanned_state = nnx.merge_state(scanned_params, scanned_other)
      elif hasattr(nnx, "State") and hasattr(nnx.State, "merge"):
        scanned_state = nnx.State.merge(scanned_params, scanned_other)
      else:
        scanned_state = nnx.merge(scanned_params, scanned_other)

    nnx.update(layers, scanned_state)
  return final_carry
