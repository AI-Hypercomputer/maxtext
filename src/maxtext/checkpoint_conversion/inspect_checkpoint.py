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


"""
A unified tool to inspect checkpoint structures for:
1. HuggingFace/PyTorch (need load weight)
2. MaxText Model Architecture (lightweight, no weights loaded)
3. Orbax Checkpoints (lightweight, no weights loaded)

Usage Examples:
[Mode 1: HF/PyTorch]   
   python src/maxtext/checkpoint_conversion/inspect_checkpoint.py hf --path <local_hf_path> --format <safetensors | pth>
[Mode 2: MaxText Arch] 
  python src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext model_name <maxtext_model_name> scan_layers <True | False>
[Mode 3: Orbax]        
  python src/maxtext/checkpoint_conversion/inspect_checkpoint.py orbax --path <local_orbax_path | gcs_orbax_path>
"""

import argparse
import sys
import os
import re
import pathlib


def natural_sort_key(s: str):
  """Sorts strings containing numbers naturally (1, 2, 10 instead of 1, 10, 2)."""
  return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", str(s))]


def print_structure(data_dict):
  """Utility to print sorted keys and shapes from a flat dictionary."""
  for key in sorted(data_dict.keys(), key=natural_sort_key):
    print(f"key: {key} | {data_dict[key]}")


# ==============================================================================
# Mode 1: HuggingFace / PyTorch (.safetensors or .pth)
# ==============================================================================
def inspect_hf(args):
  print(f"\n--- Inspecting {args.format} files in {args.path} ---")

  ckpt_paths = sorted(pathlib.Path(args.path).glob(f"[!.]*.{args.format}"))
  if not ckpt_paths:
    sys.exit(f"No files with extension .{args.format} found in {args.path}")

  param_dict = {}

  if args.format == "safetensors":
    try:
      from safetensors import safe_open
    except ImportError:
      sys.exit("Error: 'safetensors' is required. `pip install safetensors`")

    for i, ckpt_path in enumerate(ckpt_paths):
      print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          # Storing shape directly to save memory, rather than the full tensor
          shape = f.get_tensor(k).shape
          param_dict[k] = f"shape: {shape}"

  elif args.format == "pth":
    try:
      import torch
    except ImportError:
      sys.exit("Error: 'torch' is required for this mode. `pip install torch`")

    for i, ckpt_path in enumerate(ckpt_paths):
      print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
      checkpoint = torch.load(ckpt_path, map_location="cpu")
      # Flatten logic might be needed depending on pth structure,
      # here we assume standard state_dict or handle the wrapper keys manually if needed.
      if isinstance(checkpoint, dict):
        for k, v in checkpoint.items():
          # Handle nested state dicts or wrapper keys if common in your workflow
          shape = v.shape if hasattr(v, "shape") else "Non-tensor found"
          param_dict[k] = f"shape: {shape}"

  print("\n=== Structure ===")
  print_structure(param_dict)


# ==============================================================================
# Mode 2: MaxText Architecture (On-the-fly)
# ==============================================================================
def inspect_maxtext(args, remaining_args):

  # Lazy imports
  import jax
  from maxtext.utils import max_utils, maxtext_utils
  from MaxText import pyconfig
  from maxtext.utils.globals import MAXTEXT_PKG_DIR
  from maxtext.layers import quantizations
  from maxtext.models import models

  Transformer = models.transformer_as_linen

  # Setup config
  argv = (
      # First arg is usually script name in pyconfig
      [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")]
      + remaining_args
      + ["attention=dot_product", "skip_jax_distributed_system=true"]
  )
  print(argv)

  # Initialize without heavyweight runtime
  config = pyconfig.initialize(argv)
  print(f"\n--- Inspecting MaxText Architecture: {config.model_name} (Scan: {config.scan_layers}) ---")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)

  # Get abstract params (no memory/compute)
  abstract_param = maxtext_utils.get_abstract_param(model, config)
  num_params = max_utils.calculate_num_params_from_pytree(abstract_param)

  print(f"\nTotal Parameters: {num_params} (~{num_params/1e9:.2f} B)")
  print("\n=== Structure ===")

  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_param)

  param_dict = {}
  # abstract_leaf_value: ShapeDtypeStruct(shape=(128, 58), dtype=float32)
  for path_tuple, abstract_leaf_value in abstract_params_flat:
    key_parts = [k.key for k in path_tuple if hasattr(k, "key")]
    # Construct MaxText style parameter key
    param_key = "params-" + "-".join(key_parts)
    shape = abstract_leaf_value.shape
    param_dict[param_key] = f"shape: {shape}"
    dtype = abstract_leaf_value.dtype
    param_dict[param_key] += f" | dtype: {dtype}"

  print_structure(param_dict)


# ==============================================================================
# Mode 3: Orbax Checkpoint (Saved)
# ==============================================================================
def inspect_orbax(args):
  print(f"\n--- Inspecting Orbax Checkpoint: {args.path} ---")

  # Lazy imports
  try:
    import orbax.checkpoint as ocp
    from etils import epath
  except ImportError:
    sys.exit("Error: 'orbax-checkpoint' or 'etils' not found. `pip install orbax-checkpoint etils[epath]`")

  path = epath.Path(args.path)

  # Depending on Orbax version, metadata access might vary slightly.
  # This aligns with StandardCheckpointer usage.
  metadata = ocp.StandardCheckpointer().metadata(path)
  if hasattr(metadata, "item_metadata"):
    metadata = metadata.item_metadata

  # Convert to flat dict
  dictionary = ocp.tree.to_flat_dict(metadata)

  # Filter for params only and clean up keys
  param_dict = {}
  for k, v in dictionary.items():
    # k is a tuple, join it. v is metadata object with .shape
    param_key = ".".join(k)
    if not param_key.startswith("params"):
      continue
    shape = v.shape
    param_dict[param_key] = f"shape: {shape}"
    dtype = v.dtype
    param_dict[param_key] += f" | dtype: {dtype}"
    print(v)

  print("\n=== Structure ===")
  print_structure(param_dict)


# ==============================================================================
# Main CLI Driver
# ==============================================================================
def main():
  parser = argparse.ArgumentParser(description="Consolidated Model Checkpoint Inspector")
  subparsers = parser.add_subparsers(dest="mode", required=True, help="Inspection mode: hf, maxtext, orbax")

  # Mode 1: HuggingFace / PyTorch
  parser_hf = subparsers.add_parser("hf", help="Inspect .safetensors or .pth files")
  parser_hf.add_argument("--path", type=str, required=True, help="Directory containing checkpoint files")
  parser_hf.add_argument(
      "--format", type=str, required=False, choices=["safetensors", "pth"], default="safetensors", help="File format"
  )

  # Mode 2: MaxText Architecture
  parser_mt = subparsers.add_parser("maxtext", help="Inspect MaxText theoretical architecture")

  # Mode 3: Orbax
  parser_orbax = subparsers.add_parser("orbax", help="Inspect saved Orbax checkpoint metadata")
  parser_orbax.add_argument("--path", type=str, required=True, help="Path to checkpoint items (local or GCS)")

  args, remaining_args = parser.parse_known_args()

  if args.mode == "hf":
    inspect_hf(args)
  elif args.mode == "maxtext":
    inspect_maxtext(args, remaining_args)
  elif args.mode == "orbax":
    inspect_orbax(args)


if __name__ == "__main__":
  main()
