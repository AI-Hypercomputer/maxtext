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

Top-level key mapping:
  Linen → NNX:
    params/params/<model>  →  model/<model>       (remove double-nesting, rename, add {value:} wrappers)
    opt_state              →  optimizer/opt_state  (remove 'params' level from mu/nu)
    step                   →  optimizer/step       (move inside optimizer)

  NNX → Linen:
    model/<model>          →  params/params/<model>  (strip {value:} wrappers, add double-nesting)
    optimizer/opt_state    →  opt_state               (add 'params' level to mu/nu)
    optimizer/step         →  step                    (move to top level)

Layer structure (--scan_layers):
  linen_to_nnx:
    scan_layers=True  (default): stack layers_N arrays → 'layers' tensor with layer dim at axis 1
    scan_layers=False:           rename layers_N → integer-keyed 'layers/{N}'

  nnx_to_linen (auto-detected):
    Stacked 'layers' tensor  → unstack along axis 1 → layers_N per-layer arrays
    Integer-keyed layers/{N} → rename to layers_N

Usage:
  python linen_nnx_converter.py \\
    --source_path="gs://bucket/checkpoint/0/items" \\
    --target_path="gs://bucket/converted/" \\
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
import numpy as np
from etils import epath
import orbax.checkpoint as ocp


def log(message: str) -> None:
  print(f"[linen_nnx_converter] {message}")


# ── Format detection ───────────────────────────────────────────────────────────


def detect_format(state: dict) -> str:
  """Detects checkpoint format ('linen' or 'nnx') from top-level keys."""
  # NNX: uses 'model' as the top-level params key
  if "model" in state:
    return "nnx"

  if "params" not in state:
    raise ValueError(f"Cannot detect checkpoint format: no 'model' or 'params' key. " f"Found: {list(state.keys())}")

  params = state["params"]

  # Linen: double-nested params/params/decoder
  if isinstance(params, dict) and "params" in params:
    inner = params["params"]
    if isinstance(inner, dict) and ("decoder" in inner or "encoder" in inner):
      return "linen"

  # Old NNX format: params/decoder (single-nested with value wrappers)
  if isinstance(params, dict) and ("decoder" in params or "encoder" in params):
    if _has_value_wrappers(params):
      return "nnx"

  if "optimizer" in state:
    return "nnx"
  if "opt_state" in state:
    return "linen"

  raise ValueError(
      f"Could not detect checkpoint format. Keys: {list(state.keys())}, "
      f"params keys: {list(params.keys()) if isinstance(params, dict) else type(params)}"
  )


# ── Value wrapper helpers ──────────────────────────────────────────────────────


def _has_value_wrappers(tree: Any) -> bool:
  """Returns True if tree contains {value: array} wrappers (NNX style)."""
  if isinstance(tree, dict):
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, np.ndarray):
        return True
    for v in tree.values():
      if _has_value_wrappers(v):
        return True
  return False


def _strip_value_wrappers(tree: Any) -> Any:
  """Recursively strips {value: array} wrappers from a tree."""
  if isinstance(tree, dict):
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, np.ndarray):
        return inner
    return {k: _strip_value_wrappers(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_strip_value_wrappers(item) for item in tree)
  else:
    return tree


def _add_value_wrappers(tree: Any) -> Any:
  """Recursively wraps leaf arrays in {value: array} (NNX nnx.Param format)."""
  if isinstance(tree, dict):
    if set(tree.keys()) == {"value"}:
      inner = tree["value"]
      if hasattr(inner, "shape") or isinstance(inner, np.ndarray):
        return tree  # Already wrapped
    return {k: _add_value_wrappers(v) for k, v in tree.items()}
  elif isinstance(tree, (list, tuple)):
    return type(tree)(_add_value_wrappers(item) for item in tree)
  elif hasattr(tree, "shape") or isinstance(tree, np.ndarray):
    return {"value": tree}
  else:
    return tree


# ── Layer structure helpers ────────────────────────────────────────────────────


def _stack_layers(decoder: dict) -> tuple[dict, bool]:
  """Stacks per-layer parameters (layers_N) into a single 'layers' dict at axis 0.

  Returns (result_dict, was_stacked).
  """
  layer_pattern = re.compile(r"^layers_(\d+)$")
  layer_indices = {}
  other_keys = {}

  for key, value in decoder.items():
    match = layer_pattern.match(key)
    if match:
      layer_indices[int(match.group(1))] = value
    else:
      other_keys[key] = value

  if not layer_indices:
    return decoder, False

  sorted_indices = sorted(layer_indices.keys())
  num_layers = len(sorted_indices)
  log(f"  Found {num_layers} individual layers, stacking into 'layers'")

  def stack_arrays(layers_data: list) -> Any:
    first = layers_data[0]
    if hasattr(first, "shape") or isinstance(first, np.ndarray):
      return np.stack([np.asarray(layers_data[i]) for i in range(len(layers_data))], axis=0)
    elif isinstance(first, dict):
      result = {}
      for key in first.keys():
        child_data = [layers_data[i].get(key) for i in range(len(layers_data))]
        if all(c is not None for c in child_data):
          result[key] = stack_arrays(child_data)
      return result
    else:
      return first

  layers_data = [layer_indices[i] for i in sorted_indices]
  stacked = stack_arrays(layers_data)

  result = dict(other_keys)
  result["layers"] = stacked
  return result, True


def _rename_layers_to_integer_keys(decoder: dict) -> dict:
  """Converts layers_N keys to integer-keyed dict under 'layers' (no stacking).

  Converts {layers_0: {...}, layers_1: {...}} → {layers: {'0': {...}, '1': {...}}}.
  Used for scan_layers=False linen→nnx conversion (Pattern C).
  """
  layer_pattern = re.compile(r"^layers_(\d+)$")
  layer_indices = {}
  other_keys = {}

  for key, value in decoder.items():
    match = layer_pattern.match(key)
    if match:
      layer_indices[int(match.group(1))] = value
    else:
      other_keys[key] = value

  if not layer_indices:
    return decoder

  sorted_indices = sorted(layer_indices.keys())
  log(f"  Found {len(sorted_indices)} individual layers, renaming to integer-keyed 'layers/N'")
  result = dict(other_keys)
  result["layers"] = {str(i): layer_indices[i] for i in sorted_indices}
  return result


def _transpose_layers_axes(tree: Any, src_axis: int, dst_axis: int) -> Any:
  """Transposes the layers dimension in arrays within a tree (src_axis ↔ dst_axis)."""
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
    log(f"    Transposed: {tree.shape} → {result.shape}")
    return result
  else:
    return tree


def _detect_num_layers(tree: Any, scan_axis: int) -> int | None:
  """Detects num_layers from the first array with ndim > scan_axis."""
  if hasattr(tree, "shape") or isinstance(tree, np.ndarray):
    shape = getattr(tree, "shape", None) or np.asarray(tree).shape
    if len(shape) > scan_axis:
      return shape[scan_axis]
    return None
  if isinstance(tree, dict):
    for v in tree.values():
      result = _detect_num_layers(v, scan_axis)
      if result is not None:
        return result
  return None


def _unstack_single_layer(tree: Any, idx: int, scan_axis: int) -> Any:
  """Extracts a single layer by indexing at scan_axis."""
  if hasattr(tree, "shape") or isinstance(tree, np.ndarray):
    arr = np.asarray(tree)
    if arr.ndim > scan_axis:
      return np.take(arr, idx, axis=scan_axis)
    return arr
  if isinstance(tree, dict):
    return {k: _unstack_single_layer(v, idx, scan_axis) for k, v in tree.items()}
  if isinstance(tree, (list, tuple)):
    return type(tree)(_unstack_single_layer(v, idx, scan_axis) for v in tree)
  return tree


def _convert_layers_to_linen_format(decoder: dict) -> dict:
  """Converts NNX 'layers' back to Linen's layers_N format (auto-detects NNX style).

  Handles:
    - Stacked tensor (Pattern B):  layers/<arrays with layer dim at axis 1>
                                   → layers_0, layers_1, ...  (unstack along axis 1)
    - Integer-keyed (Pattern C):   layers/0, layers/1, ...
                                   → layers_0, layers_1, ...  (rename)
  """
  if "layers" not in decoder:
    return decoder

  layers_val = decoder["layers"]
  other_keys = {k: v for k, v in decoder.items() if k != "layers"}

  if not isinstance(layers_val, dict):
    # Already a non-dict (shouldn't happen normally), keep as-is
    return decoder

  # Pattern C: integer-keyed per-layer dict → rename
  if all(k.isdigit() for k in layers_val.keys()):
    result = dict(other_keys)
    for idx_str, layer_data in sorted(layers_val.items(), key=lambda x: int(x[0])):
      result[f"layers_{idx_str}"] = layer_data
    log(f"  Renamed integer-keyed layers/N → layers_N ({len(layers_val)} layers)")
    return result

  # Pattern B: stacked tensor (layer dim at axis 1) → unstack
  num_layers = _detect_num_layers(layers_val, scan_axis=1)
  if num_layers is None:
    log("  WARNING: Could not detect num_layers for unstacking, keeping 'layers' as-is")
    result = dict(other_keys)
    result["layers"] = layers_val
    return result

  result = dict(other_keys)
  for i in range(num_layers):
    result[f"layers_{i}"] = _unstack_single_layer(layers_val, idx=i, scan_axis=1)
  log(f"  Unstacked scanned 'layers' → layers_N ({num_layers} layers at axis 1)")
  return result


# ── Optimizer state helpers ────────────────────────────────────────────────────


def _convert_opt_state_linen_to_nnx(opt_state: Any) -> Any:
  """Removes 'params' nesting from mu/nu in linen opt_state.

  NNX optimizer state has plain arrays (no {value:} wrappers).
  Linen opt_state mirrors the params structure (params/decoder/...),
  so we remove the 'params' level to get decoder/... directly.
  """
  if isinstance(opt_state, dict):
    result = {}
    for k, v in opt_state.items():
      if k == "params":
        # Remove this level by merging its contents up
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
  else:
    return opt_state  # Plain array or scalar — no value wrapper for opt_state


def _convert_opt_state_nnx_to_linen(opt_state: Any, depth: int = 0) -> Any:
  """Adds 'params' nesting to mu/nu, removes any stray {value:} wrappers.

  NNX optimizer mu/nu contains decoder/... directly.
  Linen expects mu/params/decoder/... (one 'params' level mirroring the params structure).
  """
  if isinstance(opt_state, dict):
    # Strip any {value:} wrappers in opt_state (shouldn't be there but handle gracefully)
    if set(opt_state.keys()) == {"value"}:
      inner = opt_state["value"]
      if hasattr(inner, "shape") or isinstance(inner, np.ndarray):
        return inner

    result = {}
    for k, v in opt_state.items():
      converted = _convert_opt_state_nnx_to_linen(v, depth + 1)
      # Add one 'params' level after mu/nu (mirrors linen's params structure)
      if k in ("mu", "nu") and isinstance(converted, dict):
        result[k] = {"params": converted}
      else:
        result[k] = converted
    return result
  elif isinstance(opt_state, (list, tuple)):
    return type(opt_state)(_convert_opt_state_nnx_to_linen(item, depth + 1) for item in opt_state)
  else:
    return opt_state


# ── Main conversion functions ──────────────────────────────────────────────────


def convert_linen_to_nnx(state: dict, scan_layers: bool = True) -> dict:
  """Converts Linen checkpoint to NNX format.

  Args:
    state: Linen checkpoint dict with keys ['params', 'opt_state', 'step'].
    scan_layers: If True (default), stack per-layer arrays and insert layer
                 dim at axis 1 (for NNX with scan_layers=True).
                 If False, rename layers_N → integer-keyed layers/N
                 (for NNX with scan_layers=False).
  """
  result = {}

  if "params" in state:
    linen_params = state["params"]
    # Remove double 'params' nesting: params/params/decoder → decoder
    if isinstance(linen_params, dict) and "params" in linen_params:
      nnx_params = linen_params["params"]
      log("  params: Removed double 'params' nesting (params/params → model)")
    else:
      nnx_params = linen_params
      log("  params: No double nesting found")

    stripped = _strip_value_wrappers(nnx_params)

    for component in ("decoder", "encoder"):
      if component in stripped and isinstance(stripped[component], dict):
        if scan_layers:
          stripped[component], was_stacked = _stack_layers(stripped[component])
          if was_stacked and "layers" in stripped[component]:
            log(f"  {component}/layers: Transposing stacked (layers, ...) → (..., layers, ...) at axis 1")
            stripped[component]["layers"] = _transpose_layers_axes(stripped[component]["layers"], src_axis=0, dst_axis=1)
        else:
          stripped[component] = _rename_layers_to_integer_keys(stripped[component])

    result["model"] = _add_value_wrappers(stripped)
    log("  model: Saved with {value:} wrappers under 'model' key")

  # optimizer: move step inside, keep opt_state
  optimizer_dict = {}
  if "step" in state:
    optimizer_dict["step"] = state["step"]
    log(f"  optimizer/step: Moved from top-level (step={state['step']})")
  if "opt_state" in state:
    optimizer_dict["opt_state"] = _convert_opt_state_linen_to_nnx(state["opt_state"])
    log("  optimizer/opt_state: Removed 'params' nesting from mu/nu")
  if optimizer_dict:
    result["optimizer"] = optimizer_dict

  return result


def convert_nnx_to_linen(state: dict) -> dict:
  """Converts NNX checkpoint to Linen format.

  Reads from 'model'/'optimizer' keys (or falls back to old 'params'/'opt_state' format).
  Layer structure is auto-detected (stacked vs integer-keyed).
  """
  result = {}

  model_key = "model" if "model" in state else "params"
  if model_key in state:
    nnx_params = state[model_key]
    stripped = _strip_value_wrappers(nnx_params)
    log(f"  {model_key}: Removed {{value:}} wrappers")

    for component in ("decoder", "encoder"):
      if component in stripped and isinstance(stripped[component], dict):
        stripped[component] = _convert_layers_to_linen_format(stripped[component])

    # Add double 'params' nesting: decoder → params/params/decoder
    result["params"] = {"params": stripped}
    log("  params: Added double 'params' nesting (model → params/params)")

  # optimizer: extract step and opt_state back to top level
  if "optimizer" in state:
    optimizer = state["optimizer"]
    if "step" in optimizer:
      result["step"] = optimizer["step"]
      log("  step: Extracted from optimizer/step to top level")
    if "opt_state" in optimizer:
      result["opt_state"] = _convert_opt_state_nnx_to_linen(optimizer["opt_state"])
      log("  opt_state: Added 'params' nesting to mu/nu")
  elif "opt_state" in state:
    # Backward compat: old format with opt_state at top level
    result["opt_state"] = _convert_opt_state_nnx_to_linen(state["opt_state"])
    log("  opt_state: Converted from top-level opt_state (old format)")

  if "step" in state and "step" not in result:
    result["step"] = state["step"]

  return result


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────


def load_checkpoint(checkpoint_path: str) -> dict:
  """Loads checkpoint from local or GCS path."""
  log(f"Loading checkpoint from: {checkpoint_path}")

  checkpoint_dir = epath.Path(checkpoint_path)
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
  metadata = ckptr.metadata(checkpoint_dir)

  devices = np.array(jax.devices()).reshape((-1,))
  single_device_mesh = jax.sharding.Mesh(devices, ("x",))
  unsharded = jax.sharding.NamedSharding(single_device_mesh, jax.sharding.PartitionSpec())

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


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
  parser = argparse.ArgumentParser(
      description="Convert between Linen and NNX checkpoint formats.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--source_path",
      type=str,
      required=True,
      help="Path to source checkpoint items directory (e.g. gs://bucket/ckpt/0/items).",
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
      help="Conversion direction. 'auto' detects from source format.",
  )
  parser.add_argument(
      "--scan_layers",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "For linen_to_nnx only: if True (default), stack per-layer arrays into a "
          "scanned 'layers' tensor with layer dim at axis 1 (for NNX with scan_layers=True). "
          "If False, rename layers_N to integer-keyed layers/N without stacking "
          "(for NNX with scan_layers=False)."
      ),
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
    log(f"Auto-detected: {source_format} → {target_format}")
  else:
    source_format = args.direction.split("_to_")[0]
    target_format = args.direction.split("_to_")[1]
    log(f"Using specified direction: {source_format} → {target_format}")

  log(f"Converting: {source_format} → {target_format}")
  if source_format == "linen":
    log(f"scan_layers={args.scan_layers}")

  if source_format == "linen" and target_format == "nnx":
    converted_state = convert_linen_to_nnx(state, scan_layers=args.scan_layers)
  elif source_format == "nnx" and target_format == "linen":
    converted_state = convert_nnx_to_linen(state)
  else:
    raise ValueError(f"Invalid conversion: {source_format} → {target_format}")

  save_checkpoint(converted_state, args.target_path)

  elapsed = time.time() - start_time
  print("\n" + "=" * 80)
  print(f"Conversion complete in {elapsed:.2f} seconds")
  print(f"  Source: {args.source_path}")
  print(f"  Target: {args.target_path}")
  print(f"  Direction: {source_format} → {target_format}")
  print("=" * 80)


if __name__ == "__main__":
  main()
