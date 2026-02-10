# Copyright 2023-2026 Google LLC
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

"""Compare checkpoint tree structures, shapes, and values.

Supports comparing any combination of Linen and NNX checkpoints:
- Linen vs NNX (cross-format comparison)
- Linen vs Linen (same-format comparison)
- NNX vs NNX (same-format comparison)

The script auto-detects the format of each checkpoint and applies the
appropriate normalization. Cross-format transformations (like layer axis
transposition) are only applied when comparing Linen vs NNX.

Key differences between Linen and NNX checkpoints:
- Linen: params/params/decoder/layers/0/... (per-layer, double nested)
- NNX: model/decoder/layers/... (stacked layers, single nested, {value: array} wrappers)

The script handles:
- Double 'params' nesting in Linen checkpoints
- 'model' key in NNX checkpoints (vs 'params' in Linen)
- {value: array} wrappers in NNX checkpoints
- Layer axis transposition (NNX stacks layers along axis 0, only for cross-format)
- RNG filtering (NNX has rngs, Linen doesn't)

Usage:
  # Compare Linen vs NNX (structure and shapes only)
  python compare_linen_nnx_checkpoint.py \
    --ckpt_path_1="gs://bucket/linen_checkpoint/0/items" \
    --ckpt_path_2="gs://bucket/nnx_checkpoint/0/items"

  # Compare NNX vs NNX
  python compare_linen_nnx_checkpoint.py \
    --ckpt_path_1="gs://bucket/nnx_checkpoint_a/0/items" \
    --ckpt_path_2="gs://bucket/nnx_checkpoint_b/0/items"

  # Compare Linen vs Linen
  python compare_linen_nnx_checkpoint.py \
    --ckpt_path_1="gs://bucket/linen_checkpoint_a/0/items" \
    --ckpt_path_2="gs://bucket/linen_checkpoint_b/0/items"

  # Compare with value checking
  python compare_linen_nnx_checkpoint.py \
    --ckpt_path_1="gs://bucket/checkpoint_a/0/items" \
    --ckpt_path_2="gs://bucket/checkpoint_b/0/items" \
    --compare_values --atol=1e-5 --rtol=1e-5
"""

import os
from typing import Any, Dict, Sequence

# MUST set before importing JAX to force CPU-only mode
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, keystr, tree_structure, tree_map_with_path
import numpy as np
from etils import epath
import orbax.checkpoint as ocp
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ckpt_path_1",
    None,
    "Path to the first checkpoint items directory. Format is auto-detected.",
    required=True,
)
flags.DEFINE_string(
    "ckpt_path_2",
    None,
    "Path to the second checkpoint items directory. Format is auto-detected.",
    required=True,
)
flags.DEFINE_boolean(
    "verbose",
    False,
    "Print detailed per-parameter information.",
)
flags.DEFINE_boolean(
    "transpose_nnx_layers",
    False,
    "Transpose NNX layer params from (layers, ...) to (...) for comparison. "
    "NNX stacks layers along axis 0, while Linen stores per-layer params. "
    "Only applied for cross-format (Linen vs NNX) comparisons.",
)
flags.DEFINE_string(
    "compare_only",
    "params",
    "Which parts to compare: 'params' for params only, 'all' for full state.",
)
flags.DEFINE_boolean(
    "ignore_rngs",
    True,
    "Ignore RNG-related paths in comparison (NNX has rngs, Linen doesn't).",
)
flags.DEFINE_boolean(
    "compare_values",
    False,
    "Also compare parameter values (not just structure and shapes).",
)
flags.DEFINE_float(
    "atol",
    1e-5,
    "Absolute tolerance for value comparison.",
)
flags.DEFINE_float(
    "rtol",
    1e-5,
    "Relative tolerance for value comparison.",
)


def log(message: str) -> None:
  """Log a message with prefix."""
  print(f"[compare_ckpt] {message}")


def is_rng_path(path: str) -> bool:
  """Check if a path is RNG-related."""
  path_lower = path.lower()
  return "rngs" in path_lower or "rng" in path_lower


def filter_rngs(tree: Dict[str, Any]) -> Dict[str, Any]:
  """Filter out RNG-related keys from a tree."""
  if not isinstance(tree, dict):
    return tree

  result = {}
  for key, value in tree.items():
    # Skip RNG-related keys
    if is_rng_path(key):
      continue
    # Recursively filter nested dicts
    if isinstance(value, dict):
      filtered = filter_rngs(value)
      if filtered:  # Only add if not empty after filtering
        result[key] = filtered
    else:
      result[key] = value
  return result


def detect_format(state: dict) -> str:
  """Detects checkpoint format from state structure ('linen' or 'nnx').

  Linen format:
    - Top-level keys: ['params', 'opt_state', 'step']
    - params/params/decoder/... (double nested)

  NNX format:
    - Top-level keys: ['model', 'optimizer'] (nnx.State style)
    - model/decoder/... with {value: array} wrappers
  """
  # Check for NNX nnx.State format (has 'model' key instead of 'params')
  if "model" in state:
    return "nnx"

  if "params" not in state:
    raise ValueError(f"Checkpoint does not contain 'params' or 'model' key. Found keys: {list(state.keys())}")

  params = state["params"]

  # Check for Linen's double 'params' nesting
  if isinstance(params, dict) and "params" in params:
    inner = params["params"]
    if isinstance(inner, dict) and ("decoder" in inner or "encoder" in inner):
      return "linen"

  # Check for NNX's flat structure (params/decoder/...)
  if isinstance(params, dict) and ("decoder" in params or "encoder" in params):
    return "nnx"

  # Try to detect by looking for {value: array} wrappers (NNX style)
  if _has_value_wrappers(params):
    return "nnx"

  raise ValueError(
      f"Could not detect checkpoint format. params keys: {list(params.keys()) if isinstance(params, dict) else type(params)}"
  )


def _has_value_wrappers(tree: Any) -> bool:
  """Check if tree contains {value: array} wrappers (NNX style)."""
  if isinstance(tree, dict):
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, (np.ndarray, jnp.ndarray)):
        return True
    for v in tree.values():
      if _has_value_wrappers(v):
        return True
  return False


def _strip_value_wrappers(tree: Any) -> Any:
  """Recursively strips {'value': array} wrappers from a tree."""
  if isinstance(tree, dict):
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, (np.ndarray, jnp.ndarray)):
        return inner
    return {k: _strip_value_wrappers(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_strip_value_wrappers(item) for item in tree)
  else:
    return tree


def _normalize_linen_params(params: dict) -> dict:
  """Normalize Linen params by removing double 'params' nesting."""
  if isinstance(params, dict) and "params" in params:
    inner = params["params"]
    if isinstance(inner, dict) and ("decoder" in inner or "encoder" in inner):
      return inner
  return params


def _normalize_nnx_params(params: dict) -> dict:
  """Normalize NNX params by stripping {value: array} wrappers."""
  return _strip_value_wrappers(params)


def load_checkpoint(checkpoint_path: str) -> dict:
  """Loads checkpoint from local or GCS path."""
  log(f"Loading checkpoint from: {checkpoint_path}")

  checkpoint_dir = epath.Path(checkpoint_path)

  # Create checkpointer and get metadata
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

  try:
    metadata = ckptr.metadata(checkpoint_dir)
    # Create a mesh with all available devices for unsharded restoration
    devices = np.array(jax.devices()).reshape((-1,))
    single_device_mesh = jax.sharding.Mesh(devices, ("x",))
    unsharded = jax.sharding.NamedSharding(single_device_mesh, jax.sharding.PartitionSpec())

    # Build restore args that restore arrays without original sharding
    restore_args = jax.tree_util.tree_map(
        lambda x: ocp.ArrayRestoreArgs(sharding=unsharded) if hasattr(x, "shape") else None,
        metadata.item_metadata.tree,
        is_leaf=lambda x: hasattr(x, "shape"),
    )
    state = ckptr.restore(checkpoint_dir, restore_args=restore_args)
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Fallback to simple restore without sharding args
    log(f"  Falling back to simple restore: {e}")
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(checkpoint_path)

  if state is None:
    raise ValueError(f"Failed to restore checkpoint from {checkpoint_path}")

  log(f"  Loaded keys: {list(state.keys())}")
  return state


def transform_nnx_params_for_comparison(nnx_params: Dict[str, Any]) -> Dict[str, Any]:
  """Transform NNX params to match Linen structure for comparison.

  NNX stacks layer parameters along axis 0 (shape: [num_layers, ...]),
  while Linen stores per-layer parameters (shape: [...]).

  This function transposes layer params from (layers, d1, d2, ...) to (d1, layers, d2, ...)
  to align with how Linen params would look if stacked.
  """

  def _transform(path, leaf: jax.Array) -> jax.Array:
    key_str = keystr(path)

    # Only transform arrays in 'layers' with ndim >= 2
    if "layers" in key_str and hasattr(leaf, "ndim") and leaf.ndim >= 2:
      # Transpose from (layers, d1, d2, ...) to (d1, layers, d2, ...)
      axes = (1, 0) + tuple(range(2, leaf.ndim))
      result = jnp.transpose(leaf, axes=axes)
      if FLAGS.verbose:
        log(f"  TRANSPOSING: {key_str} shape {leaf.shape} -> {result.shape}")
      return result
    else:
      return leaf

  log("Transforming NNX params (transposing layer dimensions)...")
  return tree_map_with_path(_transform, nnx_params)


def get_tree_structure_info(tree: Dict[str, Any]) -> Dict[str, tuple]:
  """Get structure info as dict of path -> (shape, dtype)."""
  flat_with_path, _ = tree_flatten_with_path(tree)
  return {
      keystr(p): (
          getattr(leaf, "shape", "N/A"),
          str(getattr(leaf, "dtype", type(leaf).__name__)),
      )
      for p, leaf in flat_with_path
  }


def print_structure_diff(params1: Dict, params2: Dict, name1: str = "Linen", name2: str = "NNX"):
  """Print structural differences between two param trees."""
  info1 = get_tree_structure_info(params1)
  info2 = get_tree_structure_info(params2)
  keys1, keys2 = set(info1.keys()), set(info2.keys())

  only_in_1 = sorted(keys1 - keys2)
  only_in_2 = sorted(keys2 - keys1)
  common = keys1 & keys2

  if only_in_1:
    print(f"\n--- Paths only in {name1} ({len(only_in_1)}) ---")
    for k in only_in_1:
      shape, dtype = info1[k]
      print(f"  - {k}: shape={shape}, dtype={dtype}")

  if only_in_2:
    print(f"\n--- Paths only in {name2} ({len(only_in_2)}) ---")
    for k in only_in_2:
      shape, dtype = info2[k]
      print(f"  + {k}: shape={shape}, dtype={dtype}")

  # Check for shape/dtype mismatches in common paths
  shape_mismatches = []
  dtype_mismatches = []
  for k in common:
    shape1, dtype1 = info1[k]
    shape2, dtype2 = info2[k]
    if shape1 != shape2:
      shape_mismatches.append((k, shape1, shape2))
    if dtype1 != dtype2:
      dtype_mismatches.append((k, dtype1, dtype2))

  if shape_mismatches:
    print(f"\n--- Shape mismatches ({len(shape_mismatches)}) ---")
    for k, s1, s2 in shape_mismatches:
      print(f"  {k}: {name1}={s1}, {name2}={s2}")

  if dtype_mismatches:
    print(f"\n--- Dtype mismatches ({len(dtype_mismatches)}) ---")
    for k, d1, d2 in dtype_mismatches:
      print(f"  {k}: {name1}={d1}, {name2}={d2}")

  return only_in_1, only_in_2, shape_mismatches, dtype_mismatches


def compare_params(
    params1: Dict[str, Any],
    params2: Dict[str, Any],
    verbose: bool = False,
    compare_values: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name1: str = "Ckpt1",
    name2: str = "Ckpt2",
) -> bool:
  """Compare two parameter trees for structure, shape, and optionally values.

  Returns True if tree structures, shapes, and (optionally) values match.
  """
  # First check tree structure
  if tree_structure(params1) != tree_structure(params2):
    print("\n[✗] Tree structures differ.")
    print_structure_diff(params1, params2, name1=name1, name2=name2)
    return False

  print("\n[✓] Tree structures are the same.")

  all_match = True
  num_params = 0
  shape_mismatches = []
  dtype_mismatches = []
  value_mismatches = []
  value_matches = 0

  def _compare_leaf(path, x, y):
    nonlocal all_match, num_params, shape_mismatches, dtype_mismatches, value_mismatches, value_matches
    key_str = keystr(path)
    num_params += 1

    shape1 = getattr(x, "shape", "N/A")
    shape2 = getattr(y, "shape", "N/A")
    dtype1 = getattr(x, "dtype", type(x).__name__)
    dtype2 = getattr(y, "dtype", type(y).__name__)

    # Check shape
    shape_match = shape1 == shape2
    if not shape_match:
      shape_mismatches.append((key_str, shape1, shape2))
      all_match = False

    # Check dtype
    dtype_match = str(dtype1) == str(dtype2)
    if not dtype_match:
      dtype_mismatches.append((key_str, dtype1, dtype2))
      all_match = False

    # Check values if requested and shapes match
    if compare_values and shape_match and hasattr(x, "shape") and hasattr(y, "shape"):
      try:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        is_close = bool(np.allclose(x_arr, y_arr, atol=atol, rtol=rtol))

        if is_close:
          value_matches += 1
          if verbose:
            print(f"  [✓] {key_str} | Shape: {shape1} | Values match")
        else:
          diff = np.abs(x_arr - y_arr)
          mean_diff = float(np.mean(diff))
          max_diff = float(np.max(diff))
          value_mismatches.append((key_str, mean_diff, max_diff))
          all_match = False
          if verbose:
            print(f"  [✗] {key_str} | Shape: {shape1} | Mean diff: {mean_diff:.2e}, Max diff: {max_diff:.2e}")
      except Exception as e:  # pylint: disable=broad-exception-caught
        value_mismatches.append((key_str, f"Error: {e}", ""))
        all_match = False
    elif verbose and not compare_values:
      print(f"  {key_str} | Shape: {shape1} | Dtype: {dtype1}")

  tree_map_with_path(_compare_leaf, params1, params2)

  # Print summary
  print("\n--- Summary ---")
  print(f"Total parameters: {num_params}")

  if shape_mismatches:
    print(f"\n[✗] Shape mismatches ({len(shape_mismatches)}):")
    for key_str, s1, s2 in shape_mismatches:
      print(f"  {key_str}: {name1}={s1}, {name2}={s2}")
  else:
    print("[✓] All shapes match.")

  if dtype_mismatches:
    print(f"\n[✗] Dtype mismatches ({len(dtype_mismatches)}):")
    for key_str, d1, d2 in dtype_mismatches:
      print(f"  {key_str}: {name1}={d1}, {name2}={d2}")
  else:
    print("[✓] All dtypes match.")

  if compare_values:
    if value_mismatches:
      print(f"\n[✗] Value mismatches ({len(value_mismatches)}):")
      for item in value_mismatches[:20]:  # Show first 20
        if len(item) == 3:
          key_str, mean_diff, max_diff = item
          if isinstance(mean_diff, float):
            print(f"  {key_str}: mean_diff={mean_diff:.2e}, max_diff={max_diff:.2e}")
          else:
            print(f"  {key_str}: {mean_diff}")
      if len(value_mismatches) > 20:
        print(f"  ... and {len(value_mismatches) - 20} more (use --verbose to see all)")
    else:
      print(f"[✓] All values match (atol={atol}, rtol={rtol}).")
    print(f"    Values matching: {value_matches}/{num_params}")

  return all_match


def _extract_params(state: dict, fmt: str) -> dict:
  """Extract params from a checkpoint state based on its detected format."""
  if fmt == "linen":
    return state.get("params", {})
  else:
    # NNX format: params are in 'model' key
    return state.get("model", state.get("params", {}))


def _normalize_params(params: dict, fmt: str) -> dict:
  """Normalize params based on detected format."""
  if fmt == "linen":
    return _normalize_linen_params(params)
  else:
    return _normalize_nnx_params(params)


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  ckpt_path_1 = FLAGS.ckpt_path_1
  ckpt_path_2 = FLAGS.ckpt_path_2

  print("=" * 80)
  print("Checkpoint Comparator")
  print("=" * 80)

  print(f"\nCheckpoint 1: {ckpt_path_1}")
  print(f"Checkpoint 2: {ckpt_path_2}")
  print(f"Transpose NNX layers: {FLAGS.transpose_nnx_layers}")
  print(f"Ignore RNGs: {FLAGS.ignore_rngs}")
  print(f"Compare values: {FLAGS.compare_values}")
  if FLAGS.compare_values:
    print(f"  Tolerance: atol={FLAGS.atol}, rtol={FLAGS.rtol}")

  # Load checkpoints
  print("\n" + "-" * 40)
  state_1 = load_checkpoint(ckpt_path_1)
  state_2 = load_checkpoint(ckpt_path_2)

  # Detect formats
  format_1 = detect_format(state_1)
  format_2 = detect_format(state_2)
  log(f"Detected checkpoint 1 format: {format_1}")
  log(f"Detected checkpoint 2 format: {format_2}")

  is_cross_format = format_1 != format_2
  name_1 = f"Ckpt1({format_1})"
  name_2 = f"Ckpt2({format_2})"

  # Extract and normalize params
  print("\n" + "-" * 40)
  log("Normalizing parameters...")

  if FLAGS.compare_only == "params":
    params_1 = _extract_params(state_1, format_1)
    params_2 = _extract_params(state_2, format_2)
  else:
    params_1 = state_1
    params_2 = state_2

  params_1 = _normalize_params(params_1, format_1)
  log(f"  Checkpoint 1 ({format_1}): normalized")
  params_2 = _normalize_params(params_2, format_2)
  log(f"  Checkpoint 2 ({format_2}): normalized")

  # Filter out RNG paths if requested
  if FLAGS.ignore_rngs:
    print("\n" + "-" * 40)
    log("Filtering out RNG-related paths...")
    params_1 = filter_rngs(params_1)
    params_2 = filter_rngs(params_2)

  # Transform NNX params for cross-format comparison (transpose layer dimensions)
  # Only apply when comparing Linen vs NNX, not for same-format comparisons
  if FLAGS.transpose_nnx_layers and is_cross_format:
    print("\n" + "-" * 40)
    if format_1 == "nnx":
      params_1 = transform_nnx_params_for_comparison(params_1)
    if format_2 == "nnx":
      params_2 = transform_nnx_params_for_comparison(params_2)

  # Compare
  print("\n" + "-" * 40)
  log("Comparing parameters...")

  success = compare_params(
      params_1,
      params_2,
      verbose=FLAGS.verbose,
      compare_values=FLAGS.compare_values,
      atol=FLAGS.atol,
      rtol=FLAGS.rtol,
      name1=name_1,
      name2=name_2,
  )

  # Final verdict
  print("\n" + "=" * 80)
  if success:
    print("CHECKPOINTS MATCH")
    if FLAGS.compare_values:
      print("  Tree structure, shapes, and values are identical!")
    else:
      print("  Tree structure and all shapes are identical!")
  else:
    print("CHECKPOINTS DIFFER")
    print("  See details above for mismatches.")
  print("=" * 80)

  return 0 if success else 1


if __name__ == "__main__":
  app.run(main)
