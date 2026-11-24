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

"""One-leaf random-direction perturbation shared by every negative control.

A parity or output test calls `perturb_one_leaf` on the candidate parameters,
runs the candidate again, and asserts that the same comparison now fails.
"""

from typing import Any

import numpy as np

jax: Any
jnp: Any
try:
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
except ImportError:  # pragma: no cover - optional for pure-NumPy callers
  jax = None
  jnp = None

# A perturbation outside this relative-norm band is not a valid control: too
# small hides nothing, too large proves nothing about the tolerance.
REL_NORM_BAND = (5e-3, 2e-2)
_MIN_DIRECTION_NORM = 1e-7


def is_floating_dtype(dtype: Any) -> bool:
  """Returns True for floating-point dtypes, including ml_dtypes/jnp bfloat16."""
  if jnp is not None:
    return bool(jnp.issubdtype(dtype, jnp.floating))
  np_dt = np.dtype(dtype)
  return bool(np.issubdtype(np_dt, np.floating) or np_dt.name == "bfloat16")


def leaf_path(path: tuple[Any, ...]) -> str:
  """Renders a jax key path as '\''a/b/0'\''; raises on an unknown key type."""
  if jax is None:
    raise RuntimeError("leaf_path requires jax to be installed")
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

  The single NumPy implementation behind parameter, checkpoint, and output negative controls.

  Args:
    values: Floating-point array to perturb.
    name: Name of the array, used in error messages.
    rel_scale: Relative norm of the perturbation; both `rel_scale` and the
      post-cast realized relative norm in `values.dtype` must lie in `REL_NORM_BAND`.
    seed: Seed of the random direction.

  Returns:
    The perturbed array, with the dtype and shape of `values`.

  Raises:
    ValueError: If the requested or realized scale is outside `REL_NORM_BAND`,
      `values` is non-float, non-finite, or all zeros, or the perturbation rounds
      away in `values.dtype`.
  """
  lo, hi = REL_NORM_BAND
  if not lo <= rel_scale <= hi:
    raise ValueError(f"rel_scale={rel_scale} outside {REL_NORM_BAND}")
  if not is_floating_dtype(values.dtype):
    raise ValueError(f"{name!r} has non-float dtype {values.dtype}; pick a floating-point leaf")
  values64 = np.asarray(values, dtype=np.float64)
  if not np.all(np.isfinite(values64)):
    raise ValueError(f"{name!r} contains non-finite values")
  norm = float(np.linalg.norm(values64))
  if norm == 0.0:
    raise ValueError(f"{name!r} is all zeros; pick a leaf the output depends on")
  direction = np.random.default_rng(seed).standard_normal(values.shape)
  direction_norm = float(np.linalg.norm(direction))
  if direction_norm < _MIN_DIRECTION_NORM:
    raise ValueError(f"Degenerate direction norm {direction_norm} for seed={seed}")
  direction *= rel_scale * norm / direction_norm
  perturbed = (values64 + direction).astype(values.dtype)
  if np.array_equal(perturbed, values):
    raise ValueError(f"Perturbation of {name!r} at rel_scale={rel_scale} rounds away in dtype {values.dtype}")
  realized_rel = float(np.linalg.norm(np.asarray(perturbed, dtype=np.float64) - values64)) / norm
  if not lo <= realized_rel <= hi:
    raise ValueError(
        f"Perturbation scale outside {REL_NORM_BAND}: requested rel_scale={rel_scale}, "
        f"realized={realized_rel} for {name!r} (dtype={values.dtype})"
    )
  return perturbed


def perturb_one_leaf(params: Any, *, target_path: str, rel_scale: float, seed: int) -> Any:
  """Returns `params` with the leaf at `target_path` moved by `rel_scale * ||leaf||`.

  Args:
    params: Candidate parameter pytree.
    target_path: Exact `leaf_path` of a floating-point leaf the compared output
      depends on. Choosing it is a test-design decision, so it has no default.
    rel_scale: Relative norm of the perturbation; both `rel_scale` and the
      post-cast realized relative norm in `leaf.dtype` must lie in `REL_NORM_BAND`.
    seed: Seed of the random direction.

  Raises:
    ValueError: If the path is absent or `scaled_perturbation` rejects the leaf
      or realized perturbation norm.
  """
  if jax is None or jnp is None:
    raise RuntimeError("perturb_one_leaf requires jax to be installed")
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
  paths = [leaf_path(p) for p, _ in leaves_with_path]
  if paths.count(target_path) != 1:
    raise ValueError(f"target_path {target_path!r} matched {paths.count(target_path)} leaves")
  idx = paths.index(target_path)
  leaf = leaves_with_path[idx][1]
  perturbed = scaled_perturbation(np.asarray(leaf), name=target_path, rel_scale=rel_scale, seed=seed)
  perturbed_leaf: Any = jnp.asarray(perturbed, dtype=leaf.dtype) if isinstance(leaf, jax.Array) else perturbed
  if isinstance(leaf, jax.Array) and getattr(leaf, "committed", False):
    perturbed_leaf = jax.device_put(perturbed_leaf, leaf.sharding)
  leaves = [value for _, value in leaves_with_path]
  leaves[idx] = perturbed_leaf
  return jax.tree_util.tree_unflatten(treedef, leaves)
