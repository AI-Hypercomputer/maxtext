# Copyright 2023-2025 Google LLC
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

"""Bidirectional conversion between Linen and NNX checkpoint formats.

Usage:
  python linen_nnx_converter.py \
    --source_path="gs://bucket/checkpoint/0/items" \
    --target_path="gs://bucket/converted/" \
    --direction=auto
"""

import argparse
import os
import re
import time
from typing import Any

# MUST set before importing JAX to force CPU-only mode
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
import orbax.checkpoint as ocp


def log(message: str) -> None:
  print(f"[linen_nnx_converter] {message}")


def detect_format(state: dict) -> str:
  """Detects checkpoint format from params structure ('linen' or 'nnx')."""
  if "params" not in state:
    raise ValueError("Checkpoint does not contain 'params' key")

  params = state["params"]
  if isinstance(params, dict) and "params" in params:
    inner = params["params"]
    if isinstance(inner, dict) and ("decoder" in inner or "encoder" in inner):
      return "linen"

  if isinstance(params, dict) and ("decoder" in params or "encoder" in params):
    return "nnx"

  if "opt_state" in state:
    opt_state = state["opt_state"]
    if _has_params_in_opt_state(opt_state):
      return "linen"
    if _has_value_wrappers(opt_state):
      return "nnx"

  raise ValueError("Could not detect checkpoint format")


def _has_params_in_opt_state(opt_state: Any) -> bool:
  if isinstance(opt_state, dict):
    if "params" in opt_state:
      return True
    for v in opt_state.values():
      if _has_params_in_opt_state(v):
        return True
  return False


def _has_value_wrappers(tree: Any) -> bool:
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
    # Check if this is a value wrapper
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, (np.ndarray, jnp.ndarray)):
        return inner
    # Recurse into dict
    return {k: _strip_value_wrappers(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_strip_value_wrappers(item) for item in tree)
  else:
    return tree


def _add_value_wrappers(tree: Any) -> Any:
  """Recursively adds {'value': array} wrappers to arrays in a tree.

  NNX models store parameters as nnx.Param(value=array), which serializes
  to {'value': array} structure. This function converts plain arrays to
  that format.
  """
  if isinstance(tree, dict):
    # If already has 'value' wrapper with an array, keep as-is
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, (np.ndarray, jnp.ndarray)):
        return tree
    # Recurse into dict
    return {k: _add_value_wrappers(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_add_value_wrappers(item) for item in tree)
  elif hasattr(tree, "shape") or isinstance(tree, (np.ndarray, jnp.ndarray)):
    # Wrap arrays in {'value': array}
    return {"value": tree}
  else:
    return tree


def _transpose_layers_axes(tree: Any, src_axis: int, dst_axis: int) -> Any:
  """Transpose the layers dimension in arrays within a tree.

  Both Linen and NNX store stacked layers at config.param_scan_axis (default: 1).
  This function is used only when converting between checkpoints with different
  param_scan_axis values (src_axis != dst_axis).
  """
  if src_axis == dst_axis:
    return tree
  if isinstance(tree, dict):
    return {k: _transpose_layers_axes(v, src_axis, dst_axis) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_transpose_layers_axes(item, src_axis, dst_axis) for item in tree)
  elif hasattr(tree, "shape") and len(tree.shape) >= 2:
    axes = list(range(len(tree.shape)))
    axes[src_axis], axes[dst_axis] = axes[dst_axis], axes[src_axis]
    result = np.transpose(np.asarray(tree), axes=axes)
    log(f"    Transposed: {tree.shape} -> {result.shape}")
    return result
  else:
    return tree


def _stack_layers(decoder: dict) -> tuple[dict, bool]:
  """Stacks per-layer parameters (layers_0, layers_1, ...) into a single 'layers' dict.

  Converts structure like:
    decoder/layers_0/mlp/wi/kernel -> [embed, mlp]
    decoder/layers_1/mlp/wi/kernel -> [embed, mlp]
  To:
    decoder/layers/mlp/wi/kernel -> [num_layers, embed, mlp]  (layers at axis 0)

  Returns:
    (result_dict, was_stacked): was_stacked is True if individual layers were found and stacked.
  """
  # Find all layers_N keys
  layer_pattern = re.compile(r"^layers_(\d+)$")
  layer_indices = {}
  other_keys = {}

  for key, value in decoder.items():
    match = layer_pattern.match(key)
    if match:
      idx = int(match.group(1))
      layer_indices[idx] = value
    else:
      other_keys[key] = value

  if not layer_indices:
    return decoder, False

  # Sort by layer index
  sorted_indices = sorted(layer_indices.keys())
  num_layers = len(sorted_indices)
  log(f"  Found {num_layers} individual layers, stacking into 'layers'")

  def stack_arrays(path: str, layers_data: list) -> Any:
    """Recursively stack arrays from multiple layers."""
    first = layers_data[0]

    if hasattr(first, "shape") or isinstance(first, (np.ndarray, jnp.ndarray)):
      # Stack all arrays along new first dimension
      stacked = np.stack([np.asarray(layers_data[i]) for i in range(len(layers_data))], axis=0)
      return stacked
    elif isinstance(first, dict):
      result = {}
      for key in first.keys():
        child_data = [layers_data[i].get(key) for i in range(len(layers_data))]
        if all(c is not None for c in child_data):
          result[key] = stack_arrays(f"{path}/{key}", child_data)
      return result
    else:
      return first

  # Stack all layers
  layers_data = [layer_indices[i] for i in sorted_indices]
  stacked_layers = stack_arrays("layers", layers_data)

  # Build result with stacked layers
  result = dict(other_keys)
  result["layers"] = stacked_layers

  return result, True


def convert_linen_to_nnx(state: dict) -> dict:
  """Converts Linen checkpoint to NNX format."""
  result = {}

  # Copy step
  if "step" in state:
    result["step"] = state["step"]

  if "params" in state:
    linen_params = state["params"]
    if isinstance(linen_params, dict) and "params" in linen_params:
      nnx_params = linen_params["params"]
      log("  params: Removed double 'params' nesting")
    else:
      nnx_params = linen_params
      log("  params: No double nesting found")

    # Strip any existing 'value' wrappers first
    stripped = _strip_value_wrappers(nnx_params)

    # Stack per-layer parameters (layers_0, layers_1, ...) into single 'layers'.
    # _stack_layers stacks at axis 0; if stacking occurred we then move to param_scan_axis=1.
    # If checkpoint is already pre-scanned (layers already at axis 1), no transpose is needed.
    for component in ("decoder", "encoder"):
      if component in stripped and isinstance(stripped[component], dict):
        stripped[component], was_stacked = _stack_layers(stripped[component])
        if was_stacked and "layers" in stripped[component]:
          log(f"  Transposing {component}/layers axes: (layers, d0, ...) -> (d0, layers, ...) to match param_scan_axis=1")
          stripped[component]["layers"] = _transpose_layers_axes(stripped[component]["layers"], src_axis=0, dst_axis=1)

    # Add 'value' wrappers for NNX format
    result["params"] = _add_value_wrappers(stripped)
    log("  params: Added 'value' wrappers for NNX format")

  if "opt_state" in state:
    result["opt_state"] = _convert_opt_state_linen_to_nnx(state["opt_state"])
    log("  opt_state: Removed 'params' level and added 'value' wrappers")

  return result


def convert_nnx_to_linen(state: dict) -> dict:
  """Converts NNX checkpoint to Linen format."""
  result = {}

  # Copy step
  if "step" in state:
    result["step"] = state["step"]

  if "params" in state:
    nnx_params = state["params"]

    # Strip value wrappers first
    stripped = _strip_value_wrappers(nnx_params)
    log("  params: Removed 'value' wrappers from arrays")

    # Both NNX and Linen store layers at param_scan_axis (default: 1), so no transposition needed.

    # Add double 'params' nesting for Linen format
    if isinstance(stripped, dict) and "params" not in stripped:
      result["params"] = {"params": stripped}
      log("  params: Added double 'params' nesting")
    else:
      result["params"] = stripped
      log("  params: Already has double nesting, copied as-is")

  if "opt_state" in state:
    result["opt_state"] = _convert_opt_state_nnx_to_linen(state["opt_state"])
    log("  opt_state: Added 'params' level and removed 'value' wrappers")

  return result


def _convert_opt_state_linen_to_nnx(opt_state: Any) -> Any:
  """Removes 'params' level and adds 'value' wrappers to arrays."""
  if isinstance(opt_state, dict):
    result = {}
    for k, v in opt_state.items():
      if k == "params":
        converted = _convert_opt_state_linen_to_nnx(v)
        if isinstance(converted, dict):
          result.update(converted)
        else:
          result[k] = converted
      else:
        result[k] = _convert_opt_state_linen_to_nnx(v)
    return result
  elif isinstance(opt_state, (list, tuple)):
    return type(opt_state)(_convert_opt_state_linen_to_nnx(item) for item in opt_state)
  elif hasattr(opt_state, "shape"):
    return {"value": opt_state}
  else:
    return opt_state


def _convert_opt_state_nnx_to_linen(opt_state: Any, depth: int = 0) -> Any:
  """Removes 'value' wrappers and adds 'params' level after mu/nu keys."""
  if isinstance(opt_state, dict):
    if set(opt_state.keys()) == {"value"}:
      inner = opt_state["value"]
      if hasattr(inner, "shape") or isinstance(inner, (np.ndarray, jnp.ndarray)):
        return inner

    result = {}
    for k, v in opt_state.items():
      converted = _convert_opt_state_nnx_to_linen(v, depth + 1)
      if k in ("mu", "nu") and isinstance(converted, dict):
        result[k] = {"params": converted}
      else:
        result[k] = converted
    return result
  elif isinstance(opt_state, (list, tuple)):
    return type(opt_state)(_convert_opt_state_nnx_to_linen(item, depth + 1) for item in opt_state)
  else:
    return opt_state


def load_checkpoint(checkpoint_path: str) -> dict:
  """Loads checkpoint from local or GCS path."""
  log(f"Loading checkpoint from: {checkpoint_path}")

  checkpoint_dir = epath.Path(checkpoint_path)

  # Create checkpointer and get metadata
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
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

  log(f"  Loaded keys: {list(state.keys())}")
  return state


def save_checkpoint(state: dict, output_path: str) -> None:
  """Saves checkpoint to local or GCS path."""
  log(f"Saving checkpoint to: {output_path}")

  output_dir = epath.Path(output_path)
  output_dir.mkdir(exist_ok=True, parents=True)

  ckptr = ocp.PyTreeCheckpointer()
  ckptr.save(output_dir, state, force=True)

  log("  Checkpoint saved successfully")


def main():
  parser = argparse.ArgumentParser(
      description="Convert between Linen and NNX checkpoint formats.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument(
      "--source_path",
      type=str,
      required=True,
      help="Path to source checkpoint (e.g., gs://bucket/checkpoint/0/items)",
  )
  parser.add_argument(
      "--target_path",
      type=str,
      required=True,
      help="Path to save converted checkpoint.",
  )
  parser.add_argument(
      "--direction",
      type=str,
      choices=["auto", "linen_to_nnx", "nnx_to_linen"],
      default="auto",
      help="Conversion direction. 'auto' detects from source.",
  )

  args = parser.parse_args()

  print("=" * 80)
  print("Linen <-> NNX Checkpoint Converter")
  print("=" * 80)

  start_time = time.time()

  state = load_checkpoint(args.source_path)

  if args.direction == "auto":
    source_format = detect_format(state)
    target_format = "nnx" if source_format == "linen" else "linen"
    log(f"Auto-detected: {source_format} -> {target_format}")
  else:
    source_format = args.direction.split("_to_")[0]
    target_format = args.direction.split("_to_")[1]
    log(f"Using specified direction: {source_format} -> {target_format}")

  log(f"Converting: {source_format} -> {target_format}")

  if source_format == "linen" and target_format == "nnx":
    converted_state = convert_linen_to_nnx(state)
  elif source_format == "nnx" and target_format == "linen":
    converted_state = convert_nnx_to_linen(state)
  else:
    raise ValueError(f"Invalid conversion: {source_format} -> {target_format}")

  save_checkpoint(converted_state, args.target_path)

  elapsed = time.time() - start_time
  print("\n" + "=" * 80)
  print(f"Conversion complete in {elapsed:.2f} seconds")
  print(f"  Source: {args.source_path}")
  print(f"  Target: {args.target_path}")
  print(f"  Direction: {source_format} -> {target_format}")
  print("=" * 80)


if __name__ == "__main__":
  main()
