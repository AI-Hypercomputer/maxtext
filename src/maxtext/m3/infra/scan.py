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
layout for compatibility with legacy scanned checkpoints.

Models hold their repeated blocks in an `nnx.Sequential` and run them with
`carry = self.layers(carry, **kwargs)`. Every `nnx.Sequential` in a model is
treated as such a stack, so models should not use it for anything else.
"""

import functools
from collections.abc import MutableMapping, Sequence
import inspect
from typing import Any

from flax import nnx
import jax
from jax import checkpoint_policies as cp
from jax.experimental import xla_metadata
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
  layout. Placement is skipped when the layers were not built under a mesh.
  """
  first = variables[0]
  axis = _stack_axis(first, param_scan_axis)
  value = jnp.stack([v[...] for v in variables], axis=axis)
  spec = first.get_metadata().get("out_sharding")
  changes = {}
  if isinstance(spec, P):
    entries = list(spec)
    entries.extend([None] * max(0, axis - len(entries)))
    entries.insert(axis, None)
    spec = P(*entries)
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
    entries = list(spec)
    del entries[axis]
    return variable.replace(variable[...], out_sharding=P(*entries))

  return jax.tree.map(strip, state, is_leaf=_is_variable)


def _filter_layer_kwargs(layer_cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
  """Filters `kwargs` to parameters accepted by `layer_cls.__call__`, matching `nnx.Sequential`."""
  if not kwargs:
    return {}
  params = list(inspect.signature(layer_cls.__call__).parameters.values())
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
    return kwargs
  # `self` and `carry` consume the first 2 positional slots.
  pos_consumed = 0
  accepted: set[str] = set()
  for p in params:
    if p.kind == inspect.Parameter.POSITIONAL_ONLY:
      pos_consumed += 1
    elif p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
      if pos_consumed < 2:
        pos_consumed += 1
      else:
        accepted.add(p.name)
    elif p.kind == inspect.Parameter.VAR_POSITIONAL:
      pos_consumed = 2
    elif p.kind == inspect.Parameter.KEYWORD_ONLY:
      accepted.add(p.name)
  return {k: v for k, v in kwargs.items() if k in accepted}


class ScannedLayers(nnx.Module):
  """Structurally identical layers with stacked state, executed via `jax.lax.scan`.

  `scan_run` combines this mixin with the original layer class, preserving
  parameter paths and layer helper methods. Each iteration rebuilds a layer
  from the current graphdef and its state slice, retaining later annotations.
  Parameters are read-only scan inputs; other state is updated after the scan.
  """

  def __init__(
      self,
      layers: Sequence[nnx.Module],
      *,
      param_scan_axis: int = 1,
      remat: Any = None,
  ):  # pylint: disable=super-init-not-called
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
    self.remat = remat

  def __call__(self, carry: Any, **kwargs: Any) -> Any:
    axis = self._param_scan_axis
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    params = jax.tree.map(lambda x: jnp.moveaxis(x, min(axis, x.ndim - 1), 0), params)
    params = _strip_layer_axis(params, axis)
    stacked_rest = rest
    rest = _strip_layer_axis(rest, 0)

    remat_cfg = getattr(self, "remat", None)
    if remat_cfg is None:
      use_remat = False
      remat_pol = None
    elif callable(remat_cfg):
      use_remat = True
      remat_pol = remat_cfg
    else:
      use_remat = True
      remat_pol = getattr(remat_cfg, "policy", None)
      if remat_pol is None:
        remat_pol = cp.nothing_saveable

    mro = type(self).__mro__
    layer_cls = mro[mro.index(ScannedLayers) + 1]
    filtered_kw = _filter_layer_kwargs(layer_cls, kwargs)
    traced_kw = {k: v for k, v in filtered_kw.items() if isinstance(v, jax.Array)}
    static_kw = {k: v for k, v in filtered_kw.items() if k not in traced_kw}

    def _layer_step(carry_in, layer_params, layer_rest, dyn_kw):
      layer = nnx.merge(graphdef, layer_params, layer_rest)
      # The layer class's `__call__`: next in the MRO after this mixin.
      carry_out = super(ScannedLayers, layer).__call__(carry_in, **dyn_kw, **static_kw)
      _, _, new_rest = nnx.split(layer, nnx.Param, ...)
      return carry_out, new_rest

    step_fn = jax.checkpoint(_layer_step, policy=remat_pol, prevent_cse=True) if use_remat else _layer_step

    def body(carry_in, xs):
      layer_params, layer_rest = xs
      return step_fn(carry_in, layer_params, layer_rest, traced_kw)

    has_offload = bool(getattr(remat_cfg, "offloaded", False) or getattr(remat_pol, "offloaded", False))
    if self.num_layers == 1 and not has_offload:
      # XLA's while-loop simplifier inlines trip-count-1 loops by default. Inlining drops the
      # loop boundary, allowing the scheduler to overlap activation temporaries with adjacent
      # layers and CSE across remat, defeating the per-layer checkpoint. Keeping the while loop
      # preserves the scan body as a memory barrier. Skip when offloading residuals to
      # `pinned_host` (`S(5)`): an un-simplified length-1 scan folds its backward `[1, ...]` ->
      # `[...]` squeeze into a cross-memory-space `bitcast` (`S(0)` -> `S(5)`) rejected by XLA.
      with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
        carry, new_rest = jax.lax.scan(body, carry, (params, rest), length=self.num_layers)
    else:
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
  """Cache the path-transparent wrapper type so runs share graphdefs and JIT caches."""
  return type(f"Scanned{layer_cls.__name__}", (ScannedLayers, layer_cls), {})


def scan_run(
    layers: Sequence[nnx.Module],
    *,
    param_scan_axis: int,
    remat: Any = None,
) -> ScannedLayers:
  """Stacks `layers` (all of one class) into a `Scanned<Layer>` instance."""
  if not layers:
    raise ValueError("Cannot scan an empty run of layers.")
  return _scanned_class(type(layers[0]))(layers, param_scan_axis=param_scan_axis, remat=remat)


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


def scan_layers(layers: nnx.Sequential, *, param_scan_axis: int) -> nnx.Module:
  """Returns the scanned replacement for one `nnx.Sequential` container.

  One run of identical layers becomes a single `ScannedLayers`. Several runs
  become an `nnx.Sequential` of `ScannedLayers`, one per run.
  """
  prior_remat = getattr(layers, "remat", None)
  runs = _identical_runs(list(layers.layers))
  scanned = []
  for run in runs:
    if isinstance(run[0], ScannedLayers):
      if prior_remat is not None:
        for item in run:
          if getattr(item, "remat", None) is None:
            item.remat = prior_remat
      scanned.extend(run)
      continue
    scanned.append(scan_run(run, param_scan_axis=param_scan_axis, remat=prior_remat))
  if len(scanned) == 1:
    return scanned[0]
  return nnx.Sequential(*scanned)


def _replace_child(model: nnx.Module, path: tuple, replacement: nnx.Module) -> None:
  parent: Any = model
  for key in path[:-1]:
    parent = parent[key] if isinstance(key, int) or isinstance(parent, MutableMapping) else getattr(parent, key)
  key = path[-1]
  if isinstance(key, int) or isinstance(parent, MutableMapping):
    parent[key] = replacement
  else:
    setattr(parent, key, replacement)


def apply_scan(model: nnx.Module, *, param_scan_axis: int) -> nnx.Module:
  """Replaces every `nnx.Sequential` in `model` with its scanned form, in place."""
  containers = [(path, node) for path, node in nnx.iter_graph(model) if isinstance(node, nnx.Sequential)]
  for path, container in containers:
    replacement = scan_layers(container, param_scan_axis=param_scan_axis)
    if not path:
      model = replacement
    else:
      _replace_child(model, path, replacement)
  return model


def maybe_apply_scan(model: nnx.Module, config) -> nnx.Module:
  """Scans the model's layers when `config.scan_layers` is set."""
  if not config.scan_layers:
    return model
  return apply_scan(model, param_scan_axis=getattr(config, "param_scan_axis", 0))
