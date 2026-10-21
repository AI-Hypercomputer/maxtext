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

"""Layer scan, applied after construction.

`maybe_apply_scan` finds every `Layers` container in a model and replaces it
with `ScannedLayers`: one module holding the layers' parameters stacked along
a new axis, applied with `jax.lax.scan`. Compile time stops growing with depth
and XLA sees one layer body instead of N copies.

Paths and shapes match legacy MaxText's scanned layout, which checkpoints and
weight mappings already understand: a `ScannedLayers` holds the *same
submodules as one layer* -- `decoder/layers/self_attention/query/kernel`, no
extra segment -- and each parameter gains a layer axis at
`config.param_scan_axis` (legacy's default of 1 puts it after `embed`, which
keeps FSDP's shard of `embed` contiguous).

A stack whose layers are not all identical -- DeepSeek's dense-then-MoE
decoder, a Qwen3-MoE with dense `mlp_only_layers` -- is split into runs of
identical layers, each scanned on its own:
`Layers([ScannedLayers(dense_run), ScannedLayers(moe_run)])`, paths
`decoder/layers/0/...` and `decoder/layers/1/...`. Legacy does the same under
the names `dense_layers` / `moe_layers`.

Remat composes: a `Remat` set on the original container is carried onto each
`ScannedLayers`, which checkpoints the scan body. That is why the fixed
composition order is remat first, then scan.
"""

import functools
from collections.abc import Sequence
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from maxtext.m3.core.layers import Layers, Remat


def _is_variable(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def _with_layer_axis(spec: Any, axis: int) -> Any:
  """Inserts a replicated entry for the layer axis into a `PartitionSpec`."""
  if not isinstance(spec, P):
    return spec
  entries = list(spec)
  entries.insert(min(axis, len(entries)), None)
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
  rank = jnp.ndim(variable[...]) - (1 if stacked else 0)
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
    spec = _with_layer_axis(spec, axis)
    changes["out_sharding"] = spec
    sharding = getattr(first[...], "sharding", None)
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
  """A run of structurally identical layers, stacked and applied with `jax.lax.scan`.

  Not instantiated directly: `scan_run` derives a `Scanned<Layer>` class per
  layer class, with this as the first base and the layer class as the second.
  An instance therefore *is* a layer -- same attributes, same helper methods --
  whose parameters carry a layer axis, and whose `__call__` (this class's)
  scans instead of running once. Inside the scan body each iteration's slice
  is merged back into the live graphdef, giving another instance of the same
  class, on which the layer class's own `__call__` runs. Because the graphdef
  is taken live rather than at construction, anything attached to the
  submodules afterwards -- qwix's path tags, for instance -- is still there.

  Only parameters are fed to the body as scan inputs; they come back out
  unchanged and are not re-emitted, which avoids materializing a second copy
  of the stacked weights. Any other state (a router bias, RNG counters) is
  carried per layer and written back after the scan.
  """

  remat: Remat | None = None

  def __init__(self, layers: Sequence[nnx.Module], *, param_scan_axis: int = 1):  # pylint: disable=super-init-not-called
    if not layers:
      raise ValueError("ScannedLayers needs at least one layer.")
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
      if not name.startswith("_"):
        setattr(self, name, value)
    self._param_scan_axis = param_scan_axis
    self.num_layers = len(layers)

  def __call__(self, carry: Any, **kwargs: Any) -> Any:
    axis = self._param_scan_axis
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    params = jax.tree.map(lambda x: jnp.moveaxis(x, min(axis, x.ndim - 1), 0), params)
    params = _strip_layer_axis(params, axis)
    rest = _strip_layer_axis(rest, 0)

    def body(carry, xs):
      layer_params, layer_rest = xs
      layer = nnx.merge(graphdef, layer_params, layer_rest)
      # The layer class's `__call__`: next in the MRO after this mixin.
      carry = super(ScannedLayers, layer).__call__(carry, **kwargs)
      _, _, new_rest = nnx.split(layer, nnx.Param, ...)
      return carry, new_rest

    if self.remat is not None:
      # `prevent_cse=False` is safe inside `scan` and gives XLA more freedom.
      body = jax.checkpoint(body, policy=self.remat.policy, prevent_cse=False)

    carry, new_rest = jax.lax.scan(body, carry, (params, rest))
    if jax.tree.leaves(new_rest):
      nnx.update(self, new_rest)
    return carry


@functools.cache
def _scanned_class(layer_cls: type[nnx.Module]) -> type[ScannedLayers]:
  """`Scanned<Layer>`: `ScannedLayers` first (its `__init__` / `__call__` win), the layer class second.

  Cached so every scanned run of one layer class shares a type; graphdefs
  compare equal across instances and `jit` caches are not fragmented.
  """
  return type(f"Scanned{layer_cls.__name__}", (ScannedLayers, layer_cls), {})


def scan_run(layers: Sequence[nnx.Module], *, param_scan_axis: int) -> ScannedLayers:
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


def scan_layers(layers: Layers, *, param_scan_axis: int) -> nnx.Module:
  """Returns the scanned replacement for one `Layers` container.

  One run of identical layers becomes a single `ScannedLayers`. Several runs
  become a `Layers` of `ScannedLayers`, one per run, each carrying the
  original container's `Remat` so the outer container applies none itself.
  """
  runs = _identical_runs(list(layers))
  scanned = []
  for run in runs:
    block = scan_run(run, param_scan_axis=param_scan_axis)
    block.remat = layers.remat
    scanned.append(block)
  if len(scanned) == 1:
    return scanned[0]
  return Layers(scanned)


def _replace_child(model: nnx.Module, path: tuple, replacement: nnx.Module) -> None:
  parent: Any = model
  for key in path[:-1]:
    parent = parent[key] if isinstance(key, int) else getattr(parent, key)
  key = path[-1]
  if isinstance(key, int):
    parent[key] = replacement
  else:
    setattr(parent, key, replacement)


def apply_scan(model: nnx.Module, *, param_scan_axis: int) -> nnx.Module:
  """Replaces every `Layers` in `model` with its scanned form, in place."""
  containers = [(path, node) for path, node in nnx.iter_graph(model) if isinstance(node, Layers)]
  for path, container in containers:
    _replace_child(model, path, scan_layers(container, param_scan_axis=param_scan_axis))
  return model


def maybe_apply_scan(model: nnx.Module, config) -> nnx.Module:
  """Scans the model's layers when `config.scan_layers` is set."""
  if not config.scan_layers:
    return model
  return apply_scan(model, param_scan_axis=config.param_scan_axis)
