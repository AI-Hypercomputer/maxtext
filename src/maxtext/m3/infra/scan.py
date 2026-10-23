# Copyright 2026 Google LLC
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

"""Direct stacked initialization and execution with `jax.lax.scan`.

Models supply a factory for identical layers or repeated blocks. Parameters
retain one layer's paths with an added scan axis, avoiding an unstacked copy.
This module must contain no model-specific branches.
"""

import functools
from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def _is_variable(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def _with_layer_axis(spec: Any, axis: int) -> Any:
  """Inserts a replicated entry for the layer axis into a `PartitionSpec`."""
  if not isinstance(spec, P):
    return spec
  entries = list(spec)
  entries.extend([None] * max(0, axis - len(entries)))
  entries.insert(axis, None)
  return P(*entries)


def _without_layer_axis(spec: Any, axis: int) -> Any:
  if not isinstance(spec, P) or len(spec) <= axis:
    return spec
  entries = list(spec)
  del entries[axis]
  return P(*entries)


def _stack_axis(variable: nnx.Variable, param_scan_axis: int, *, stacked: bool = False) -> int:
  """Parameters stack at `param_scan_axis` (clamped to their rank), everything else at 0.

  `stacked` says whether `variable` already carries the layer axis, in which
  case the clamp is against the original rank, one less than the current.
  """
  if not isinstance(variable, nnx.Param):
    return 0
  rank = variable.ndim - (1 if stacked else 0)
  return min(param_scan_axis, rank)


def _place_stacked(first: nnx.Variable, value: jax.Array, axis: int) -> nnx.Variable:
  """Adds layer-axis metadata and preserves the variable's placement."""
  spec = first.get_metadata().get("out_sharding")
  changes = {}
  if isinstance(spec, P):
    spec = _with_layer_axis(spec, axis)
    changes["out_sharding"] = spec
    sharding = getattr(first.get_value(), "sharding", None)
    if isinstance(sharding, NamedSharding):
      value = jax.device_put(value, NamedSharding(sharding.mesh, spec))
  return first.replace(value, **changes)


def _strip_layer_axis(state: Any, param_scan_axis: int) -> Any:
  """Returns `state` with per-layer sharding metadata, for the body's merged layer."""

  def strip(variable):
    spec = variable.get_metadata().get("out_sharding")
    if not isinstance(spec, P):
      return variable
    axis = _stack_axis(variable, param_scan_axis, stacked=True)
    return variable.replace(variable[...], out_sharding=_without_layer_axis(spec, axis))

  return jax.tree.map(strip, state, is_leaf=_is_variable)


class ScannedLayers(nnx.Module):
  """Structurally identical layers with stacked state, executed via `jax.lax.scan`.

  Construction combines this mixin with the original layer class, preserving
  parameter paths and layer helper methods. Each iteration rebuilds a layer
  from the current graphdef and its state slice, retaining later annotations.
  Parameters are read-only scan inputs; other state is updated after the scan.
  """

  def __init__(self, stacked_layer: nnx.Module, *, num_layers: int, param_scan_axis: int):  # pylint: disable=super-init-not-called
    # Adopt the stacked layer's attributes, so this module's parameter tree is
    # one layer's tree with a layer axis -- no wrapper segment.
    for name, value in vars(stacked_layer).items():
      if not name.startswith(("_pytree__", "_object__")):
        setattr(self, name, value)
    self._param_scan_axis = param_scan_axis
    self.num_layers = num_layers

  def __call__(self, carry: Any, **kwargs: Any) -> Any:
    axis = self._param_scan_axis
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    params = jax.tree.map(lambda x: jnp.moveaxis(x, min(axis, x.ndim - 1), 0), params)
    params = _strip_layer_axis(params, axis)
    stacked_rest = rest
    rest = _strip_layer_axis(rest, 0)

    def body(carry, xs):
      layer_params, layer_rest = xs
      layer = nnx.merge(graphdef, layer_params, layer_rest)
      # The layer class's `__call__`: next in the MRO after this mixin.
      carry = super(ScannedLayers, layer).__call__(carry, **kwargs)
      _, _, new_rest = nnx.split(layer, nnx.Param, ...)
      return carry, new_rest

    carry, new_rest = jax.lax.scan(body, carry, (params, rest), length=self.num_layers)
    if jax.tree.leaves(new_rest):
      # Restore the stacked metadata stripped for the body.
      new_rest = jax.tree.map(
          lambda original, updated: original.replace(updated[...]),
          stacked_rest,
          new_rest,
          is_leaf=_is_variable,
      )
      nnx.update(self, new_rest)
    return carry


@functools.cache
def _scanned_class(layer_cls: type[nnx.Module]) -> type[ScannedLayers]:
  """`Scanned<Layer>`: `ScannedLayers` first (its `__init__` / `__call__` win), the layer class second.

  Cached so every scanned run of one layer class shares a type; graphdefs
  compare equal across instances and `jit` caches are not fragmented.
  """
  return type(f"Scanned{layer_cls.__name__}", (ScannedLayers, layer_cls), {})


def init_scan(
    layer_factory: Callable[[nnx.Rngs], nnx.Module],
    *,
    num_layers: int,
    rngs: nnx.Rngs,
    param_scan_axis: int,
) -> ScannedLayers:
  """Initializes stacked layers directly, preserving sequential RNG consumption."""
  if num_layers < 1:
    raise ValueError("num_layers must be positive.")
  if not isinstance(param_scan_axis, int) or param_scan_axis < 0:
    raise ValueError("param_scan_axis must be a non-negative integer.")
  rng_graphdef, rng_state = nnx.split(rngs)

  def make_layer(state):
    layer_rngs = nnx.merge(rng_graphdef, state, copy=True)
    layer = layer_factory(layer_rngs)
    return nnx.state(layer_rngs), layer

  _, abstract_layer = jax.eval_shape(make_layer, rng_state)
  graphdef, template = nnx.split(abstract_layer)
  axes = jax.tree.map(lambda v: _stack_axis(v, param_scan_axis), template, is_leaf=_is_variable)
  mesh = jax.sharding.get_abstract_mesh()

  def output_sharding(variable):
    spec = variable.get_metadata().get("out_sharding")
    if isinstance(spec, P) and not mesh.empty:
      return NamedSharding(mesh, _with_layer_axis(spec, _stack_axis(variable, param_scan_axis)))
    return None

  shardings = jax.tree.map(output_sharding, template, is_leaf=_is_variable)

  @functools.partial(jax.jit, out_shardings=(None, shardings))
  def initialize(state):
    # Retain only RNG counters from this call; XLA eliminates its weights.
    advanced, _ = make_layer(state)

    def batch_rng(variable, after):
      value = variable[...]
      if isinstance(variable, nnx.RngCount):
        offsets = jnp.arange(num_layers, dtype=value.dtype).reshape((num_layers,) + (1,) * value.ndim)
        value = value + offsets * (after[...] - value)
      else:
        value = jnp.broadcast_to(value, (num_layers,) + value.shape)
      return variable.replace(value)

    batched = jax.tree.map(batch_rng, state, advanced, is_leaf=_is_variable)
    stacked = jax.vmap(lambda r: nnx.state(make_layer(r)[1]), out_axes=axes)(batched)
    stacked = jax.tree.map(
        lambda v: _place_stacked(v, v[...], _stack_axis(v, param_scan_axis, stacked=True)),
        stacked,
        is_leaf=_is_variable,
    )

    def advance_rng(variable, after):
      value = variable[...]
      if isinstance(variable, nnx.RngCount):
        value = value + num_layers * (after[...] - value)
      return variable.replace(value)

    return jax.tree.map(advance_rng, state, advanced, is_leaf=_is_variable), stacked

  rng_state, stacked_state = initialize(rng_state)
  nnx.update(rngs, rng_state)
  return _scanned_class(type(abstract_layer))(
      nnx.merge(graphdef, stacked_state), num_layers=num_layers, param_scan_axis=param_scan_axis
  )
