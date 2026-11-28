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

"""Bridge between MaxText's NNX state and this model's parameter pytrees.

Two differences have to be reconciled:

  * MaxText stacks scanned `nnx.Param` leaves at `param_scan_axis` (1 by
    default) while non-`Param` variables stay at axis 0; we always stack at
    axis 0 so `jax.lax.scan` can consume the tree directly.
  * MaxText's router bias is a `MoEBiasVar`, which the trainer deliberately
    excludes from the gradient.  We keep it in a separate `aux` tree so that
    `jax.grad` cannot reach it either.

Flax is imported inside the functions that use it: the port itself does not
depend on it.
"""

import jax
import jax.numpy as jnp

_SCAN_STACKS = ("dense_layers", "moe_layers")
_SKIP = ("rngs",)


def _set(tree, path, value):
  node = tree
  for key in path[:-1]:
    node = node.setdefault(key, {})
  node[path[-1]] = value


def _get(tree, path):
  node = tree
  for key in path:
    node = node[key]
  return node


def _flat_state(state):
  """`(path, variable)` pairs of an `nnx.State` (`State.flat_state` is deprecated)."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  return nnx.to_flat_state(state)


def _value(var):
  """The array a variable holds, or `var` itself when it is already one.

  `get_value()` rather than `.value`, which flax deprecates.
  """
  get_value = getattr(var, "get_value", None)
  return get_value() if callable(get_value) else var


def from_nnx_state(state, param_scan_axis: int = 1):
  """Converts a `nnx.State` (from `nnx.split(model)`) into `(params, aux)`.

  Args:
    state: the NNX state of a MaxText `Transformer`.
    param_scan_axis: `config.param_scan_axis` used to build the model.

  Returns:
    `(params, aux)` -- nested dicts of `jax.Array`, with scanned stacks at
    axis 0.
  """
  params: dict = {}
  aux: dict = {}
  for path, var in _flat_state(state):
    parts = [str(p) for p in path]
    if any(skip in parts for skip in _SKIP):
      continue
    kind = type(var).__name__
    if kind not in ("Param", "MoEBiasVar"):
      continue
    value = jnp.asarray(_value(var))
    scanned = any(stack in parts for stack in _SCAN_STACKS)
    if scanned and kind == "Param" and param_scan_axis != 0:
      value = jnp.moveaxis(value, param_scan_axis, 0)
    _set(params if kind == "Param" else aux, parts, value)
  return params, aux


def tree_from_nnx_state(state, param_scan_axis: int = 1):
  """Converts a single-collection `nnx.State` (e.g. gradients) into one dict.

  Unlike `from_nnx_state` this does not look at variable types -- after
  `jax.grad` the leaves need not be typed variables -- so every leaf lands in
  one tree and every scanned stack is moved to axis 0.
  """
  out: dict = {}
  for path, var in _flat_state(state):
    parts = [str(p) for p in path]
    if any(skip in parts for skip in _SKIP):
      continue
    value = _value(var)
    if not hasattr(value, "shape"):
      continue
    value = jnp.asarray(value)
    if any(stack in parts for stack in _SCAN_STACKS) and param_scan_axis != 0:
      value = jnp.moveaxis(value, param_scan_axis, 0)
    _set(out, parts, value)
  return out


def from_nnx_model(model, param_scan_axis: int = 1):
  from flax import nnx  # pylint: disable=g-import-not-at-top

  _, state = nnx.split(model)
  return from_nnx_state(state, param_scan_axis)


def to_nnx_state(model, params, aux, param_scan_axis: int = 1):
  """Writes `(params, aux)` back into a live NNX model, in place."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  _, state = nnx.split(model)
  updates = []
  for path, var in _flat_state(state):
    parts = [str(p) for p in path]
    if any(skip in parts for skip in _SKIP):
      continue
    kind = type(var).__name__
    if kind not in ("Param", "MoEBiasVar"):
      continue
    value = _get(params if kind == "Param" else aux, parts)
    scanned = any(stack in parts for stack in _SCAN_STACKS)
    if scanned and kind == "Param" and param_scan_axis != 0:
      value = jnp.moveaxis(value, 0, param_scan_axis)
    updates.append((path, value))
  for path, value in updates:
    node = model
    for key in path[:-1]:
      node = node[int(key)] if str(key).isdigit() else getattr(node, key)
    getattr(node, path[-1]).set_value(value)
  return model


def tree_report(params, aux):
  """`path -> (shape, dtype)` for eyeballing a converted tree."""
  rows = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path({"params": params, "aux": aux})[0]:
    name = "/".join(str(k.key) for k in path)
    rows[name] = (tuple(leaf.shape), str(leaf.dtype))
  return rows
