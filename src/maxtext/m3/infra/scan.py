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

"""Model-independent layer scanning, applied after construction.

Replaces `nnx.Sequential` containers with `ScannedLayers`, stacking parameters
at `config.param_scan_axis` and executing each run of structurally identical
layers with `jax.lax.scan`. Parameter paths retain one layer's submodule
layout so scanned checkpoints share `decoder/layers/<param>` directly.

Models hold their repeated blocks in an `nnx.Sequential` and run them with
`carry = self.layers(carry, **kwargs)`. Every `nnx.Sequential` in a model is
treated as such a stack, so models should not use it for anything else.
"""

from collections.abc import MutableMapping, Sequence
import functools
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def _is_variable(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def _stack_axis(variable: nnx.Variable, param_scan_axis: int, *, stacked: bool = False) -> int:
  """Parameters use the configured axis; other state stacks at axis zero."""
  if not isinstance(variable, nnx.Param):
    return 0
  rank = variable.ndim - (1 if stacked else 0)
  return min(param_scan_axis, rank)


def _stack_variables(variables: Sequence[nnx.Variable], param_scan_axis: int) -> nnx.Variable:
  """Stacks one variable from each layer, keeping its sharding metadata consistent.

  The stacked value is placed explicitly: `jnp.stack` of sharded arrays leaves
  the result's placement to XLA, which would quietly drop the per-layer FSDP
  layout. Placement is skipped when the layers were not built under a mesh or
  hold abstract `ShapeDtypeStruct` values during `eval_shape`.
  """
  first = variables[0]
  axis = _stack_axis(first, param_scan_axis)
  value = jnp.stack([v[...] for v in variables], axis=axis)
  spec = first.get_metadata().get("out_sharding")
  changes: dict[str, Any] = {}
  if isinstance(spec, P):
    entries = list(spec)
    entries.extend([None] * max(0, axis - len(entries)))
    entries.insert(axis, None)
    spec = P(*entries)
    changes["out_sharding"] = spec
    sharding = getattr(first.get_value(), "sharding", None)
    if isinstance(sharding, NamedSharding) and isinstance(value, jax.Array):
      value = jax.device_put(value, NamedSharding(sharding.mesh, spec))
  return first.replace(value, **changes)


def _strip_layer_axis(state: Any, param_scan_axis: int) -> Any:
  """Returns `state` with per-layer sharding metadata, for the body's merged layer."""

  def strip(variable: nnx.Variable) -> nnx.Variable:
    spec = variable.get_metadata().get("out_sharding")
    if not isinstance(spec, P):
      return variable
    axis = _stack_axis(variable, param_scan_axis, stacked=True)
    entries = list(spec)
    if axis < len(entries):
      del entries[axis]
    return variable.replace(variable[...], out_sharding=P(*entries))

  return jax.tree.map(strip, state, is_leaf=_is_variable)


class ScannedLayers(nnx.Module):
  """Structurally identical layers with stacked state, executed via `jax.lax.scan`.

  `scan_run` combines this mixin with the original layer class, preserving
  parameter paths and layer helper methods. Each iteration rebuilds a layer
  from the current graphdef and its state slice, retaining later annotations.
  Parameters are read-only scan inputs; other state is updated after the scan.
  """

  def __init__(self, layers: Sequence[nnx.Module], *, param_scan_axis: int = 1):  # pylint: disable=super-init-not-called
    if not isinstance(param_scan_axis, int) or param_scan_axis < 0:
      raise ValueError("param_scan_axis must be a non-negative integer.")
    if not layers:
      raise ValueError("Cannot scan an empty run of layers.")
    graphdef = nnx.graphdef(layers[0])
    for i, layer in enumerate(layers[1:], start=1):
      if nnx.graphdef(layer) != graphdef:
        raise ValueError(f"Layer {i} is not structurally identical to layer 0, so they cannot share one scan body.")
    states = [nnx.state(layer) for layer in layers]
    stacked = jax.tree.map(lambda *vs: _stack_variables(vs, param_scan_axis), *states, is_leaf=_is_variable)
    stacked_layer = nnx.merge(graphdef, stacked)
    # Adopt the stacked layer's attributes, so this module's parameter tree is
    # one layer's tree with a layer axis -- no wrapper segment.
    for name, value in vars(stacked_layer).items():
      if not name.startswith(("_pytree__", "_object__")):
        setattr(self, name, value)
    self._param_scan_axis = param_scan_axis
    self.num_layers = len(layers)

  def __call__(self, carry: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
    axis = self._param_scan_axis
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    params = jax.tree.map(lambda x: jnp.moveaxis(x, min(axis, x.ndim - 1), 0), params)
    params = _strip_layer_axis(params, axis)
    stacked_rest = rest
    rest = _strip_layer_axis(rest, 0)

    def body(carry: Any, xs: tuple[Any, Any]) -> tuple[Any, Any]:
      layer_params, layer_rest = xs
      layer = nnx.merge(graphdef, layer_params, layer_rest)
      carry = super(ScannedLayers, layer).__call__(carry, **kwargs)
      _, _, new_rest = nnx.split(layer, nnx.Param, ...)
      return carry, new_rest

    carry, new_rest = jax.lax.scan(body, carry, (params, rest), length=self.num_layers)
    if jax.tree.leaves(new_rest):
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
  """Caches the path-transparent wrapper type so runs share graphdefs and JIT caches."""
  return type(f"Scanned{layer_cls.__name__}", (ScannedLayers, layer_cls), {})


def scan_run(layers: Sequence[nnx.Module], *, param_scan_axis: int = 1) -> ScannedLayers:
  """Stacks `layers` (all of one class) into a `Scanned<Layer>` instance."""
  if not layers:
    raise ValueError("Cannot scan an empty run of layers.")
  return _scanned_class(type(layers[0]))(layers, param_scan_axis=param_scan_axis)


def _identical_runs(layers: Sequence[nnx.Module]) -> list[list[nnx.Module]]:
  """Splits `layers` into maximal consecutive runs sharing one graphdef."""
  runs: list[list[nnx.Module]] = []
  last_graphdef = None
  for layer in layers:
    graphdef = nnx.graphdef(layer)
    if runs and graphdef == last_graphdef:
      runs[-1].append(layer)
    else:
      runs.append([layer])
    last_graphdef = graphdef
  return runs


class _SequentialRuns(nnx.Sequential):
  """`nnx.Sequential` container forwarding `**kwargs` across every scanned run."""

  def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
    if not self.layers:
      return args[0] if len(args) == 1 else args
    carry: Any = self.layers[0](*args, **kwargs)
    for layer in self.layers[1:]:
      carry = layer(carry, **kwargs)
    return carry


def scan_layers(layers: nnx.Sequential, *, param_scan_axis: int = 1) -> nnx.Module:
  """Returns the scanned replacement for one `nnx.Sequential` container.

  One run of identical layers becomes a single `ScannedLayers`. Several runs
  become an `nnx.Sequential` of `ScannedLayers`, one per run.
  """
  runs = _identical_runs(list(layers.layers))
  scanned = []
  for run in runs:
    if isinstance(run[0], ScannedLayers):
      scanned.extend(run)
      continue
    scanned.append(scan_run(run, param_scan_axis=param_scan_axis))
  if len(scanned) == 1:
    return scanned[0]
  return _SequentialRuns(*scanned)


def _replace_child(model: nnx.Module, path: tuple[Any, ...], replacement: nnx.Module) -> None:
  parent: Any = model
  for key in path[:-1]:
    parent = parent[key] if isinstance(key, int) or isinstance(parent, MutableMapping) else getattr(parent, key)
  key = path[-1]
  if isinstance(key, int) or isinstance(parent, MutableMapping):
    parent[key] = replacement
  else:
    setattr(parent, key, replacement)


def apply_scan(model: nnx.Module, *, param_scan_axis: int = 1) -> nnx.Module:
  """Replaces every `nnx.Sequential` in `model` with its scanned form, in place."""
  containers = [(path, node) for path, node in nnx.iter_graph(model) if isinstance(node, nnx.Sequential)]
  for path, container in containers:
    replacement = scan_layers(container, param_scan_axis=param_scan_axis)
    if not path:
      model = replacement
    else:
      _replace_child(model, path, replacement)
  return model


def maybe_apply_scan(model: nnx.Module, config: Any) -> nnx.Module:
  """Scans the model's layers when `scan_layers` is enabled on `config`."""
  transforms = getattr(config, "transforms", None)
  scan_enabled = getattr(transforms, "scan_layers", None)
  if scan_enabled is None:
    scan_enabled = getattr(config, "scan_layers", False)
  if not scan_enabled:
    return model
  param_scan_axis = getattr(transforms, "param_scan_axis", None)
  if param_scan_axis is None:
    param_scan_axis = getattr(config, "param_scan_axis", 1)
  return apply_scan(model, param_scan_axis=int(param_scan_axis))
