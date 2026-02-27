"""Script for comparing parameters between two checkpoints."""

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing import Any, Dict, Sequence
from jax.tree_util import tree_flatten_with_path, keystr, tree_structure, tree_map_with_path
from absl import app
from absl import flags


_LINEN_CKPT_PATH = flags.DEFINE_string(
    "linen_ckpt_path", None, "Path to the Linen model checkpoint items directory.", required=True
)
_NNX_CKPT_PATH = flags.DEFINE_string(
    "nnx_ckpt_path", None, "Path to the NNX model checkpoint items directory.", required=True
)


def load_checkpoint_params(path: str) -> Dict[str, Any]:
  """Loads parameters from an Orbax checkpoint path."""
  print(f"Loading checkpoint from: {path}")
  checkpointer = ocp.PyTreeCheckpointer()
  restored_state = checkpointer.restore(path)
  if restored_state is None:
    raise ValueError(f"Failed to restore checkpoint from {path}")
  if isinstance(restored_state, dict) and "params" in restored_state:
    return restored_state["params"]
  return restored_state


def transform_nnx_params(nnx_params: Dict[str, Any]) -> Dict[str, Any]:
  """Applies specific transformations with verbose logging matching original format."""

  def _transform(path, leaf: jax.Array) -> jax.Array:
    key_str = keystr(path)

    if "layers" in key_str and hasattr(leaf, "ndim") and leaf.ndim >= 2:
      print(f"TRANSPOSING: {key_str} with shape {leaf.shape}")
      axes = (1, 0) + tuple(range(2, leaf.ndim))
      return jnp.transpose(leaf, axes=axes)
    else:
      if "token_embedder" in key_str:
        print(f"SKIPPING Transpose: {key_str} because it is token_embedder")
      else:
        shape = getattr(leaf, "shape", "N/A")
        print(f"SKIPPING Transpose: {key_str} with shape {shape} (ndim < 2)")
      return leaf

  print("Applying transformations to NNX params...")
  return tree_map_with_path(_transform, nnx_params)


def get_tree_structure_info(tree: Dict[str, Any]):
  """Helper only used if structures differ."""
  flat_with_path, _ = tree_flatten_with_path(tree)
  return {keystr(p): (getattr(l, "shape", "N/A"), str(getattr(l, "dtype", type(l).__name__))) for p, l in flat_with_path}


def print_structure_diff(params1, params2):
  """Prints missing/added keys if structures differ."""
  info1 = get_tree_structure_info(params1)
  info2 = get_tree_structure_info(params2)
  keys1, keys2 = set(info1.keys()), set(info2.keys())

  for k in sorted(keys2 - keys1):
    print(f"  + Added in NNX: {k}")
  for k in sorted(keys1 - keys2):
    print(f"  - Missing in NNX: {k}")


def compare_params(params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
  """
  Compares two parameter trees (e.g., JAX/Flax PyTrees) for structural and numerical equality.

  This function performs a deep comparison of two PyTrees. It first
  validates that both trees share the exact same structure. If successful, it iterates
  through every leaf node to verify:
  1. Shapes match.
  2. Data types (dtypes) match.
  3. Numerical values are close (within `jnp.allclose` tolerances).

  Args:
      params1: The first parameter dictionary or PyTree (e.g., a Linen model).
      params2: The second parameter dictionary or PyTree (e.g., an NNX model).

  Returns:
      bool: True if structure, shapes, types, and values all match; False otherwise.
  """

  if tree_structure(params1) != tree_structure(params2):
    print("[] Tree structures differ.")
    print_structure_diff(params1, params2)
    return False

  print("[] Tree structures are the same.")

  all_match = True

  def _compare_leaf(path, x, y):
    nonlocal all_match
    key_str = keystr(path)

    try:
      shape1 = getattr(x, "shape", "N/A")
      shape2 = getattr(y, "shape", "N/A")

      if shape1 != shape2:
        print(f"[{key_str}] SHAPE MISMATCH: {shape1} vs {shape2}")
        all_match = False
        return

      dtype1 = getattr(x, "dtype", type(x))
      dtype2 = getattr(y, "dtype", type(y))

      if dtype1 != dtype2:
        print(f"[{key_str}] DTYPE MISMATCH: {dtype1} vs {dtype2}")
        all_match = False
        return

      diff = x - y
      abs_diff = jnp.abs(diff)
      mean_diff_scalar = jnp.mean(abs_diff)
      max_diff_scalar = jnp.max(abs_diff)
      is_close_scalar = jnp.allclose(x, y)

      mean_diff = float(mean_diff_scalar)
      max_diff = float(max_diff_scalar)
      is_close = bool(is_close_scalar)

      print(
          f"[{key_str}] "
          f"Shape(Linen/NNX): {shape1} / {shape2} — "
          f"Mean abs diff: {mean_diff:.2e}, "
          f"Max abs diff: {max_diff:.2e}, "
          f"AllClose: {is_close}"
      )

      if not is_close:
        all_match = False

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[{key_str}] Error during comparison: {e}")
      all_match = False

  tree_map_with_path(_compare_leaf, params1, params2)

  return all_match


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  linen_ckpt_path = _LINEN_CKPT_PATH.value
  nnx_ckpt_path = _NNX_CKPT_PATH.value

  print(f"Linen Checkpoint Path: {linen_ckpt_path}")
  print(f"NNX Checkpoint Path: {nnx_ckpt_path}")

  print("Loading Linen params...")
  linen_params = load_checkpoint_params(linen_ckpt_path)
  print("Loading NNX params...")
  nnx_params = load_checkpoint_params(nnx_ckpt_path)

  if linen_params is not None and nnx_params is not None:
    nnx_params_transformed = transform_nnx_params(nnx_params)

    print("\nComparing Linen params with Transformed NNX params...")
    if compare_params(linen_params, nnx_params_transformed):
      print("\nCheckpoints are considered the same (within np.allclose tolerance) after transformation!")
    else:
      print("\nCheckpoints DIFFER after transformation.")
  else:
    print("Failed to load params from one or both checkpoints.")


if __name__ == "__main__":
  app.run(main)
