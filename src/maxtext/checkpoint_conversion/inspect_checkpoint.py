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
  python src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext --model_name <maxtext_model_name> --scan_layers <True | False>
[Mode 3: Orbax]        
  python src/maxtext/checkpoint_conversion/inspect_checkpoint.py orbax --path <local_orbax_path | gcs_orbax_path>


pip install --no-deps -e .
python src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext --model_name deepseek-custom --scan_layers false
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
    shape = data_dict[key]
    print(f"key: {key} | shape: {shape}")


# ==============================================================================
# Mode 1: HuggingFace / PyTorch (.safetensors or .pth)
# ==============================================================================
def inspect_hf(args):
  print(f"\n--- Inspecting {args.format} files in {args.path} ---")

  # Lazy imports
  try:
    import torch
  except ImportError:
    sys.exit("Error: 'torch' is required for this mode. `pip install torch`")

  ckpt_paths = sorted(pathlib.Path(args.path).glob(f"[!.]*.{args.format}"))
  if not ckpt_paths:
    sys.exit(f"No files with extension .{args.format} found in {args.path}")

  chkpt_vars_raw = {}

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
          chkpt_vars_raw[k] = f.get_tensor(k).shape

  elif args.format == "pth":
    for i, ckpt_path in enumerate(ckpt_paths):
      print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
      checkpoint = torch.load(ckpt_path, map_location="cpu")
      # Flatten logic might be needed depending on pth structure,
      # here we assume standard state_dict or handle the wrapper keys manually if needed.
      if isinstance(checkpoint, dict):
        for k, v in checkpoint.items():
          if hasattr(v, "shape"):
            chkpt_vars_raw[k] = v.shape
          else:
            # Handle nested state dicts or wrapper keys if common in your workflow
            chkpt_vars_raw[k] = "Non-tensor found"

  print("\n=== Structure ===")
  print_structure(chkpt_vars_raw)


# ==============================================================================
# Mode 2: MaxText Architecture (On-the-fly)
# ==============================================================================
def inspect_maxtext(args):
  print(f"\n--- Inspecting MaxText Architecture: {args.model_name} (Scan: {args.scan_layers}) ---")

  # Lazy imports
  import jax
  from maxtext.utils import max_utils, maxtext_utils
  from MaxText import pyconfig
  from maxtext.utils.globals import MAXTEXT_PKG_DIR
  from maxtext.layers import quantizations
  from maxtext.models import models

  Transformer = models.transformer_as_linen

  # Setup config
  argv = [
      "",  # First arg is usually script name in pyconfig
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      f"model_name={args.model_name}",
      f"scan_layers={args.scan_layers}",
      "attention=dot_product",
      "skip_jax_distributed_system=true",
      "tokenizer_type=huggingface",
      "tokenizer_path=deepseek-ai/DeepSeek-V3.2",
      "hf_access_token=fake",
  ]

  # Initialize without heavyweight runtime
  config = pyconfig.initialize(argv)
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)

  # Get abstract params (no memory/compute)
  abstract_param = maxtext_utils.get_abstract_param(model, config)
  num_params = max_utils.calculate_num_params_from_pytree(abstract_param)

  print(f"\nTotal Parameters: {num_params} (~{num_params/1e9:.2f}B)")
  print("\n=== Structure ===")

  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_param)

  flat_shapes = {}
  for path_tuple, abstract_leaf_value in abstract_params_flat:
    key_parts = [k.key for k in path_tuple if hasattr(k, "key")]
    # Construct MaxText style parameter key
    mt_param_key = "params-" + "-".join(key_parts)
    flat_shapes[mt_param_key] = abstract_leaf_value.shape

  print_structure(flat_shapes)


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

  try:
    # Depending on Orbax version, metadata access might vary slightly.
    # This aligns with StandardCheckpointer usage.
    metadata = ocp.StandardCheckpointer().metadata(path)
    if hasattr(metadata, "item_metadata"):
      metadata = metadata.item_metadata
  except Exception as e:
    sys.exit(f"Error reading Orbax metadata: {e}")

  # Convert to flat dict
  dictionary = ocp.tree.to_flat_dict(metadata)

  # Filter for params only and clean up keys
  flat_shapes = {}
  for k, v in dictionary.items():
    # k is a tuple, join it. v is metadata object with .shape
    key_str = ".".join(k)
    if key_str.startswith("params"):
      flat_shapes[key_str] = v.shape

  print("\n=== Structure ===")
  print_structure(flat_shapes)


# ==============================================================================
# Main CLI Driver
# ==============================================================================
def main():
  parser = argparse.ArgumentParser(description="Consolidated Model Checkpoint Inspector")
  subparsers = parser.add_subparsers(dest="mode", required=True, help="Inspection mode")

  # Mode 1: HuggingFace / PyTorch
  parser_hf = subparsers.add_parser("hf", help="Inspect .safetensors or .pth files")
  parser_hf.add_argument("--path", type=str, required=True, help="Directory containing checkpoint files")
  parser_hf.add_argument(
      "--format", type=str, required=False, choices=["safetensors", "pth"], default="safetensors", help="File format"
  )

  # Mode 2: MaxText Architecture
  parser_mt = subparsers.add_parser("maxtext", help="Inspect MaxText theoretical architecture")
  parser_mt.add_argument("--model_name", type=str, required=True, help="e.g. deepseek3-671b")
  parser_mt.add_argument(
      "--scan_layers",
      type=str,
      required=False,
      default="true",
      choices=["true", "false", "True", "False"],
      help="Simulate scanned or unscanned structure",
  )

  # Mode 3: Orbax
  parser_orbax = subparsers.add_parser("orbax", help="Inspect saved Orbax checkpoint metadata")
  parser_orbax.add_argument("--path", type=str, required=True, help="Path to checkpoint items (local or GCS)")

  args = parser.parse_args()

  if args.mode == "hf":
    inspect_hf(args)
  elif args.mode == "maxtext":
    inspect_maxtext(args)
  elif args.mode == "orbax":
    inspect_orbax(args)


if __name__ == "__main__":
  main()
