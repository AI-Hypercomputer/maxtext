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


def apply_scanned_layers(
    layers: nnx.Module,
    carry: Any,
    *,
    length: int,
    param_scan_axis: int,
    apply_fn: Callable[..., Any],
    xs: Any | None = None,
    remat: bool = False,
    remat_policy: Callable[..., Any] | None = None,
    prevent_cse: bool = True,
    unroll: int = 1,
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

  layer_graphdef, params, rest = nnx.split(layers, nnx.Param, ...)
  if param_scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, param_scan_axis, 0), params)

  # Parameters fed in as scan inputs come back out unchanged, so they must not be
  # re-emitted as scan outputs (see scan_body). Anything else the body produces --
  # including parameters materialized while tracing, such as Qwix LoRA adapters,
  # which are ``nnx.Param`` subclasses -- still has to leave the scan.
  carried_param_paths = {path for path, _ in nnx.to_flat_state(params)}

  def _ensure_stacked(x):
    if hasattr(x, "ndim") and x.ndim == 0:
      return jnp.broadcast_to(x, (length,))
    return x

  rest = jax.tree.map(_ensure_stacked, rest)

  def _strip_scan_metadata(leaf):
    if hasattr(leaf, "replace") and hasattr(leaf, "value"):  # pylint: disable=too-many-nested-blocks
      replace_kwargs = {}
      if hasattr(leaf, "get_metadata"):
        replace_kwargs.update(leaf.get_metadata())

      replace_kwargs.pop(nnx.PARTITION_NAME, None)
      replace_kwargs.pop("param_scan_axis", None)

      val = getattr(leaf, "value", None)
      val_ndim = getattr(val, "ndim", None)

      for key in ["sharding", "out_sharding", "kernel_axes", "sharding_names"]:
        value = getattr(leaf, key, None)
        if value is None and key in replace_kwargs:
          value = replace_kwargs[key]
        if value is not None:
          if isinstance(value, str):
            value = (value,)
          if isinstance(value, tuple):
            if val_ndim is not None and len(value) > val_ndim:
              filtered = tuple(
                  axis for axis in value if axis not in ("local_layers", "layers", "scanned_blocks", "decoder_layers")
              )
              if len(filtered) > val_ndim:
                filtered = filtered[:val_ndim]
              replace_kwargs[key] = filtered
      return leaf.replace(**replace_kwargs)
    return leaf

  def scan_body(current_carry, scanned_state):
    if xs is None:
      current_params, current_rest = scanned_state
      apply_args = ()
    else:
      current_params, current_rest, current_xs = scanned_state
      apply_args = (current_xs,)
    current_params = jax.tree.map(
        _strip_scan_metadata,
        current_params,
        is_leaf=lambda x: hasattr(x, "replace") and hasattr(x, "value"),
    )
    current_layer = nnx.merge(layer_graphdef, current_params, current_rest)
    next_carry = apply_fn(current_layer, current_carry, *apply_args)
    # Drop the parameters that were carried in: ``jax.lax.scan`` stacks every
    # output, so returning them would materialize a second copy of the stacked
    # layer weights. Parameters created inside the body are still returned.
    _, updated_params, updated_rest = nnx.split(current_layer, nnx.Param, ...)
    new_params = nnx.from_flat_state(
        [(path, value) for path, value in nnx.to_flat_state(updated_params) if path not in carried_param_paths]
    )
    return next_carry, (new_params, updated_rest)

  scan_fn = jax.checkpoint(scan_body, policy=remat_policy, prevent_cse=prevent_cse) if remat else scan_body
  scan_xs = (params, rest) if xs is None else (params, rest, xs)
  final_carry, (scanned_new_params, scanned_rest) = jax.lax.scan(scan_fn, carry, scan_xs, length=length, unroll=unroll)

  if param_scan_axis != 0:
    scanned_new_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis), scanned_new_params)

  nnx.update(layers, scanned_new_params, scanned_rest)
  return final_carry
