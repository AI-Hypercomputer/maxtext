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

"""Metadata for one side of a weight conversion: weight names, shapes, dtypes, and sharding."""

import dataclasses
import enum
from typing import Any, Mapping

import numpy as np

# Wrapper levels around the model: `base` on the trainer (TunixMaxTextAdapter), `model` on the rollout.
_WRAPPER_ROOTS = ("base", "model")
_NON_WEIGHT_PREFIXES = ("cache", "rngs", "intermediates")


class Names(enum.Enum):
  """Naming convention a layout's weight names follow."""

  MAXTEXT = "maxtext"
  HF = "hf"


@dataclasses.dataclass(frozen=True)
class LeafSpec:
  """Shape, dtype, and device sharding of one weight; `sharding` is None for host data."""

  shape: tuple[int, ...]
  dtype: np.dtype
  sharding: Any | None = None


@dataclasses.dataclass(frozen=True)
class ScanInfo:
  """How decoder layers are stacked: `num_layers` in blocks of `cycle`, stacked on `scan_axis`."""

  num_layers: int
  cycle: int
  scan_axis: int


@dataclasses.dataclass(frozen=True)
class Layout:
  """One side of a conversion.

  Attributes:
    names: Naming convention of `leaves`.
    leaves: Weight name to spec as this side stores it, or None when the planner derives it.
    rules: Rules mapping the canonical (per-layer, unfused) layout to the stored layout.
    manifest: What the consumer says it expects, when it publishes that.
  """

  names: Names
  leaves: Mapping[str, LeafSpec] | None
  rules: tuple[Any, ...] = ()
  manifest: Mapping[str, LeafSpec] | None = None


def _to_pure_dict(tree: Any) -> Any:
  """Unwraps nnx.State, FrozenDict, and nnx.Variable into plain nested dicts of arrays."""
  if hasattr(tree, "to_pure_dict"):
    tree = tree.to_pure_dict()
  if hasattr(tree, "unfreeze"):
    tree = tree.unfreeze()
  if hasattr(tree, "items"):
    return {key: _to_pure_dict(value) for key, value in tree.items()}
  if hasattr(tree, "value"):
    return tree.value
  return tree


def _flatten(tree: Mapping[Any, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
  flat = {}
  for key, value in tree.items():
    path = prefix + (str(key),)
    if isinstance(value, Mapping):
      flat.update(_flatten(value, path))
    else:
      flat[path] = value
  return flat


def _is_weight_path(path: tuple[str, ...]) -> bool:
  return not any(part.lstrip("_").startswith(_NON_WEIGHT_PREFIXES) for part in path)


def _has_wrapper_root(flat: Mapping[tuple[str, ...], Any]) -> bool:
  roots = {path[0] for path in flat}
  return len(roots) == 1 and roots <= set(_WRAPPER_ROOTS) and all(len(path) > 1 for path in flat)


def flatten_leaves(tree: Any) -> dict[str, Any]:
  """Returns `{dotted weight name: leaf}` for a trainer or rollout state.

  Wrapper levels shared by every weight are stripped, so a trainer state and a rollout state
  of the same model yield the same names. KV caches, RNG state, and intermediates are dropped.
  """
  flat = {path: leaf for path, leaf in _flatten(_to_pure_dict(tree)).items() if _is_weight_path(path)}
  while _has_wrapper_root(flat):
    flat = {path[1:]: leaf for path, leaf in flat.items()}
  return {".".join(path): leaf for path, leaf in flat.items()}


def leaf_specs(tree: Any) -> dict[str, LeafSpec]:
  """Returns `{dotted weight name: LeafSpec}` without reading any tensor value."""
  return {
      name: LeafSpec(tuple(leaf.shape), np.dtype(leaf.dtype), getattr(leaf, "sharding", None))
      for name, leaf in flatten_leaves(tree).items()
  }
