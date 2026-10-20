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

"""Post-construction scanning of repeated decoder layers or blocks.

Scan units must share an NNX graph definition and matching corresponding
variable shapes; their weight values are independent. A unit may be a single
layer or a model-specific block containing different layer types.

Models opt in through ``decoder.layer_names`` and define the scan units.
Keep this module free of model-specific branches: architecture, cyclic patterns,
and grouping belong to model definitions, not to the scan transform.
"""

from collections.abc import Sequence
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

_SHARDING_KEYS = ("out_sharding", "sharding", "sharding_names")


def _is_variable(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def _insert_axis_in_spec(spec: Any, axis: int) -> Any:
  """Inserts a replicated (None) entry for the new layer axis into a sharding spec."""
  if isinstance(spec, (P, tuple)):
    entries = list(spec)
    entries.extend([None] * max(0, axis - len(entries)))
    entries.insert(axis, None)
    return P(*entries) if isinstance(spec, P) else tuple(entries)
  return spec


def _remove_axis_from_spec(spec: Any, axis: int) -> Any:
  """Removes the scan-axis entry from a sharding spec when present."""
  if isinstance(spec, (P, tuple)) and len(spec) > axis:
    entries = list(spec)
    del entries[axis]
    return P(*entries) if isinstance(spec, P) else tuple(entries)
  return spec


def _map_sharding_metadata(state: nnx.State, fn, axis: int) -> nnx.State:
  """Applies `fn(spec, axis)` to supported sharding metadata in `state`."""

  def update(var):
    if not _is_variable(var):
      return var
    meta = var.get_metadata()
    effective_axis = axis if var.get_value().ndim >= axis else 0
    changes = {k: fn(meta[k], effective_axis) for k in _SHARDING_KEYS if meta.get(k) is not None}
    return var.replace(**changes) if changes else var

  return jax.tree.map(update, state, is_leaf=_is_variable)


def _stack_states(states: Sequence[nnx.State], param_axis: int, mesh) -> nnx.State:
  """Stacks Params on `param_axis` when rank permits; other variables use axis 0."""

  def stack(*variables):
    first = variables[0]
    if not _is_variable(first):
      return first
    axis = param_axis if isinstance(first, nnx.Param) and first.get_value().ndim >= param_axis else 0
    value = jnp.stack([v.get_value() for v in variables], axis=axis)
    meta = first.get_metadata()
    changes = {k: _insert_axis_in_spec(meta[k], axis) for k in _SHARDING_KEYS if meta.get(k) is not None}
    spec = changes.get("out_sharding", changes.get("sharding"))
    if mesh is not None and isinstance(spec, P):
      value = jax.device_put(value, NamedSharding(mesh, spec))
    return first.replace(value=value, **changes)

  return jax.tree.map(stack, *states, is_leaf=_is_variable)


class ScannedLayers(nnx.Module):
  """Scan compatible layers or blocks with independent weights.

  Units must have equal NNX graph definitions and matching corresponding variable
  shapes. Calls take an array carry and return an array of the same shape and
  dtype; additional arguments are shared across all iterations. Parameters must
  remain unchanged during calls; updated non-Param state is written back.
  """

  def __init__(self, layers: Sequence[nnx.Module], *, param_scan_axis: int = 1, mesh=None):
    if len(layers) < 2:
      raise ValueError(f"ScannedLayers needs at least two layers, got {len(layers)}.")
    graphdef = nnx.graphdef(layers[0])
    for i, layer in enumerate(layers[1:], start=1):
      if nnx.graphdef(layer) != graphdef:
        raise ValueError(f"Layer {i} is not structurally identical to layer 0; cannot scan them together.")

    stacked = nnx.merge(
        graphdef,
        _stack_states([nnx.state(l) for l in layers], param_scan_axis, mesh),
    )
    # Adopt the stacked layer's attributes so parameter paths match a single layer's.
    for name, value in vars(stacked).items():
      if not name.startswith("_"):
        setattr(self, name, value)
    self._layer_graphdef = graphdef
    self._param_scan_axis = param_scan_axis
    self.num_layers = len(layers)

  def __call__(self, y: jax.Array, *args, **kwargs) -> jax.Array:
    axis = self._param_scan_axis
    _, params, rest = nnx.split(self, nnx.Param, ...)
    params = jax.tree.map(lambda x: jnp.moveaxis(x, axis, 0) if x.ndim > axis else x, params)

    def body(carry, xs):
      layer_params, layer_rest = xs
      layer_params = _map_sharding_metadata(layer_params, _remove_axis_from_spec, axis)
      layer_rest = _map_sharding_metadata(layer_rest, _remove_axis_from_spec, 0)
      layer = nnx.merge(self._layer_graphdef, layer_params, layer_rest)
      out = layer(carry, *args, **kwargs)
      # Params come back unchanged; only emit the (possibly mutated) non-Param state.
      _, _, new_rest = nnx.split(layer, nnx.Param, ...)
      new_rest = _map_sharding_metadata(new_rest, _insert_axis_in_spec, 0)
      return out, new_rest

    y, new_rest = jax.lax.scan(body, y, (params, rest))
    if jax.tree.leaves(new_rest):
      nnx.update(self, new_rest)
    return y


def maybe_apply_scan(model: nnx.Module, config, mesh=None) -> nnx.Module:
  """Scan an opted-in decoder stack in place when enabled; no remat is applied.

  Decoders opt in by executing compatible layers or blocks through ordered
  ``layer_names``. Missing or None means unsupported; fewer than two is a no-op.
  The transform replaces these units with one module named ``layers``.
  """
  if not config.scan_layers:
    return model
  decoder = model.decoder
  layer_names = getattr(decoder, "layer_names", None)
  if layer_names is None:
    raise ValueError(
        f"{type(decoder).__name__} does not expose scan units through "
        "`layer_names`; m3 layer scanning is currently unsupported for this model."
    )
  if len(layer_names) < 2:
    return model
  layers = [getattr(decoder, name) for name in layer_names]
  scanned = ScannedLayers(layers, param_scan_axis=config.param_scan_axis, mesh=mesh)
  for name in layer_names:
    delattr(decoder, name)
  decoder.layers = scanned
  decoder.layer_names = ("layers",)
  return model
