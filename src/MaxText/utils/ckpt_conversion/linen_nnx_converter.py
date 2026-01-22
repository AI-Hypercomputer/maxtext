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
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from etils import epath
import orbax.checkpoint as ocp


os.environ["JAX_PLATFORMS"] = "cpu"


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


def convert_linen_to_nnx(state: dict) -> dict:
  """Converts Linen checkpoint to NNX format."""
  result = {}

  # Copy step
  if "step" in state:
    result["step"] = state["step"]

  if "params" in state:
    linen_params = state["params"]
    if isinstance(linen_params, dict) and "params" in linen_params:
      result["params"] = linen_params["params"]
      log("  params: Removed double 'params' nesting")
    else:
      result["params"] = linen_params
      log("  params: No double nesting found, copied as-is")

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
    if isinstance(nnx_params, dict) and "params" not in nnx_params:
      result["params"] = {"params": nnx_params}
      log("  params: Added double 'params' nesting")
    else:
      result["params"] = nnx_params
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

  ckptr = ocp.PyTreeCheckpointer()
  state = ckptr.restore(epath.Path(checkpoint_path))

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
