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

import argparse
from collections.abc import Sequence
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# A perturbation outside this relative-norm band is not a valid control: too
# small hides nothing, too large proves nothing about the tolerance.
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


def perturb_one_leaf(params: Any, *, target_path: str, rel_scale: float, seed: int) -> Any:
  """Returns `params` with the leaf at `target_path` moved by `rel_scale * ||leaf||`.

  Args:
    params: Candidate parameter pytree.
    target_path: Exact `leaf_path` of a floating-point leaf the compared output
      depends on. Choosing it is a test-design decision, so it has no default.
    rel_scale: Relative norm of the perturbation; must lie in `REL_NORM_BAND`.
    seed: Seed of the random direction.

  Raises:
    ValueError: If the path is absent, not floating point, all zeros, or the
      scale is outside `REL_NORM_BAND`.
  """
  lo, hi = REL_NORM_BAND
  if not lo <= rel_scale <= hi:
    raise ValueError(f"rel_scale={rel_scale} outside {REL_NORM_BAND}")
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
  paths = [leaf_path(p) for p, _ in leaves_with_path]
  if paths.count(target_path) != 1:
    raise ValueError(f"target_path {target_path!r} matched {paths.count(target_path)} leaves")
  idx = paths.index(target_path)
  leaf = leaves_with_path[idx][1]
  if not jnp.issubdtype(leaf.dtype, jnp.floating):
    raise ValueError(f"Leaf {target_path!r} has non-float dtype {leaf.dtype}")
  leaf64 = np.asarray(leaf, dtype=np.float64)
  leaf_norm = float(np.linalg.norm(leaf64))
  if leaf_norm == 0.0:
    raise ValueError(f"Leaf {target_path!r} is all zeros; pick a leaf the output depends on")
  direction = np.random.default_rng(seed).standard_normal(leaf.shape)
  direction *= rel_scale * leaf_norm / np.linalg.norm(direction)
  leaves = [value for _, value in leaves_with_path]
  leaves[idx] = leaf + jnp.asarray(direction, dtype=leaf.dtype)
  return jax.tree_util.tree_unflatten(treedef, leaves)


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint for negative-control perturbation self-check."""
  parser = argparse.ArgumentParser(description="Perturb one floating-point JAX pytree leaf within REL_NORM_BAND.")
  parser.add_argument(
      "--self-test",
      action="store_true",
      help="Run a quick self-check verifying perturbation norm bounds.",
  )
  args = parser.parse_args(argv)
  if args.self_test:
    tree = {"block": {"w": jnp.ones((4, 4), dtype=jnp.float32)}}
    out = perturb_one_leaf(tree, target_path="block/w", rel_scale=1e-2, seed=0)
    rel = float(jnp.linalg.norm(out["block"]["w"] - tree["block"]["w"]) / jnp.linalg.norm(tree["block"]["w"]))
    if not REL_NORM_BAND[0] * 0.99 <= rel <= REL_NORM_BAND[1] * 1.01:
      raise AssertionError(f"Unexpected relative perturbation norm: {rel}")
    print(f"NEGCTL_SELF_TEST PASS rel_norm={rel:.6f}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
