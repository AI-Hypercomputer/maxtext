# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-leaf random-direction perturbation shared by every negative control.

A parity or output test calls `perturb_one_leaf` on the candidate parameters, runs the candidate again, and asserts
that the same comparison now fails.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# A perturbation outside this relative-norm band is not a valid control: too small hides nothing, too large proves
# nothing about the tolerance.
REL_NORM_BAND = (5e-3, 2e-2)


def leaf_path(path: tuple[Any, ...]) -> str:
  """Renders a jax key path as 'a/b/0'; raises on an unknown key type."""
  parts = []
  for key in path:
    if isinstance(key, (jax.tree_util.DictKey, jax.tree_util.FlattenedIndexKey)):
      parts.append(str(key.key))
    elif isinstance(key, jax.tree_util.SequenceKey):
      parts.append(str(key.idx))
    elif isinstance(key, jax.tree_util.GetAttrKey):
      parts.append(key.name)
    else:
      raise TypeError(f"Unsupported key type {type(key).__name__} in path {path!r}")
  return "/".join(parts)


def scaled_perturbation(values: np.ndarray, *, name: str, rel_scale: float, seed: int) -> np.ndarray:
  """Returns `values` moved along a random direction of norm `rel_scale * ||values||`, in the input dtype.

  The single NumPy implementation behind every parameter, checkpoint, and output negative control.

  Args:
    values: Floating-point array to perturb.
    name: Name of the array, used in error messages.
    rel_scale: Relative norm of the perturbation; must lie in `REL_NORM_BAND`.
    seed: Seed of the random direction.

  Returns:
    The perturbed array, with the dtype and shape of `values`.

  Raises:
    ValueError: If the scale is outside `REL_NORM_BAND`, `values` is integer, boolean, or all zeros, or the
      perturbation rounds away in the input dtype.
  """
  lo, hi = REL_NORM_BAND
  if not lo <= rel_scale <= hi:
    raise ValueError(f"rel_scale={rel_scale} outside {REL_NORM_BAND}")
  if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
    raise ValueError(f"{name!r} has non-float dtype {values.dtype}; pick a floating-point leaf")
  values64 = np.asarray(values, dtype=np.float64)
  norm = float(np.linalg.norm(values64))
  if norm == 0.0:
    raise ValueError(f"{name!r} is all zeros; pick a leaf the output depends on")
  direction = np.random.default_rng(seed).standard_normal(values.shape)
  direction *= rel_scale * norm / float(np.linalg.norm(direction))
  perturbed = (values64 + direction).astype(values.dtype)
  if np.array_equal(perturbed, values):
    raise ValueError(f"Perturbation of {name!r} at rel_scale={rel_scale} rounds away in dtype {values.dtype}")
  return perturbed


def perturb_one_leaf(params: Any, *, target_path: str, rel_scale: float, seed: int) -> Any:
  """Returns `params` with the leaf at `target_path` moved by `rel_scale * ||leaf||`.

  Args:
    params: Candidate parameter pytree.
    target_path: Exact `leaf_path` of a floating-point leaf the compared output depends on. Choosing it is a
      test-design decision, so it has no default.
    rel_scale: Relative norm of the perturbation; must lie in `REL_NORM_BAND`.
    seed: Seed of the random direction.

  Returns:
    A pytree with the same structure as `params` in which only `target_path` changed.

  Raises:
    ValueError: If the path is absent, or `scaled_perturbation` rejects the leaf or the scale.
  """
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
  paths = [leaf_path(p) for p, _ in leaves_with_path]
  if paths.count(target_path) != 1:
    raise ValueError(f"target_path {target_path!r} matched {paths.count(target_path)} leaves")
  idx = paths.index(target_path)
  leaf = leaves_with_path[idx][1]
  if not jnp.issubdtype(leaf.dtype, jnp.floating):
    raise ValueError(f"Leaf {target_path!r} has non-float dtype {leaf.dtype}")
  perturbed = scaled_perturbation(np.asarray(leaf), name=target_path, rel_scale=rel_scale, seed=seed)
  leaves = [value for _, value in leaves_with_path]
  leaves[idx] = jax.device_put(perturbed, leaf.sharding) if isinstance(leaf, jax.Array) else perturbed
  return jax.tree_util.tree_unflatten(treedef, leaves)
