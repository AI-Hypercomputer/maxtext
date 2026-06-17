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

"""A unified command-line tool to inspect checkpoint structures.

Supported formats and frameworks:
  1. HuggingFace Checkpoints: Locally downloaded file.
     - safetensors: Lightweight with near-zero cost. Parses file headers to read metadata instantly without allocating RAM.
     - pth: Per-tensor load with small cost. Deserializes the file to load weights per-tensor. 
        Incurs a time cost for deserialization, but memory footprint is constrained to individual tensors.
  2. MaxText Model Architecture: On-the-fly. 
     - Lightweight with near-zero cost. Traces JAX shapes abstractly; evaluates theoretical parameters dynamically without executing compute or allocating VRAM/HBM.
  3. Orbax Checkpoints: Pre-saved disk/GCS file.
     - Lightweight with near-zero cost. Reads structural metadata trees instantly without pulling underlying TensorStore data chunks.

Usage Examples:
  [Mode 1: HuggingFace]   
    python src/maxtext/checkpoint_conversion/inspect_checkpoint.py hf \
        --path <local_hf_path> --format <safetensors | pth>
  
  [Mode 2: MaxText Architecture] 
    python src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext \
        model_name=<maxtext_model_name> scan_layers=<True | False>
  
  [Mode 3: Orbax]        
    python src/maxtext/checkpoint_conversion/inspect_checkpoint.py orbax \
        --path <local_orbax_path | gcs_orbax_path>

Additional Flags:
  `--check_dtype`: Appends the data type (dtype) of each tensor to the output.
"""

import argparse
import sys
import os
import re
import pathlib
import absl
from maxtext.inference.inference_utils import str2bool
from maxtext.checkpoint_conversion.utils.utils import print_peak_memory


def natural_sort_key(s: str):
  """Sorts strings containing numbers naturally (e.g., 1, 2, 10 instead of 1, 10, 2)."""
  return [int(text) if text.isdigit() else text for text in re.split(r"(\d+)", str(s))]


def print_structure(data_dict):
  """Utility to format and print sorted keys and shapes from a flattened dictionary."""
  for key in sorted(data_dict.keys(), key=natural_sort_key):
    print(f"key: {key} | {data_dict[key]}")


# ==============================================================================
# Mode 1: HuggingFace Checkpoint (Locally downloaded, safetensors or pth)
# ==============================================================================


# pylint: disable=import-outside-toplevel
def _inspect_safetensors(ckpt_paths, check_dtype):
  """Inspects HuggingFace checkpoints, from local .safetensors file.

  Lightweight with near-zero cost: Bypasses heavy RAM allocation and disk I/O.
  It strictly parses the JSON header of the locally downloaded safetensors file
  to extract shapes and dtypes instantly.

  Optimization: Read metadata rather than weights via `f.get_slice(k)`.
  """
  try:
    from safetensors import safe_open
  except ImportError:
    sys.exit("Error: 'safetensors' is required. Run `pip install safetensors`")

  param_dict = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")
    with safe_open(ckpt_path, framework="pt") as f:
      for k in f.keys():
        # Optimization: `f.get_slice(k)` only reads the file header.
        # This provides instant access to metadata. Using `f.get_tensor(k)`
        # instead would force the actual tensor values into host RAM.
        slice_obj = f.get_slice(k)
        shape = slice_obj.get_shape()
        param_dict[k] = f"shape: {shape}"

        if check_dtype:
          dtype = slice_obj.get_dtype()
          param_dict[k] += f" | dtype: {dtype}"

  return param_dict


# pylint: disable=import-outside-toplevel
def _inspect_pth(ckpt_paths, check_dtype):
  """Inspects HuggingFace checkpoints, from local .pth file.

  Per-tensor load with small cost: Memory is allocated per-tensor rather than
  loading the entire massive file into RAM at once. Because it deserializes
  PyTorch's pickled format, it incurs a time cost just to extract keys and shapes.

  Optimization: Per-tensor load via `mmap=True`.
  """
  try:
    import torch
  except ImportError:
    sys.exit("Error: 'torch' is required for this mode. Run `pip install torch`")

  param_dict = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f"Loading {ckpt_path.name} ({i+1}/{len(ckpt_paths)})...")

    # `mmap=True`: ensures the file is memory-mapped, keeping the RAM footprint
    #   strictly bounded to the per-tensor level rather than pulling the whole file.
    # `weights_only=True`: prevents arbitrary code execution.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)

    # Flattening logic may be necessary depending on the .pth structure.
    # This assumes a standard state_dict; wrapper keys may need manual handling if present.
    if isinstance(checkpoint, dict):
      for k, v in checkpoint.items():
        if hasattr(v, "shape"):
          param_dict[k] = f"shape: {v.shape}"

        if check_dtype and hasattr(v, "dtype"):
          param_dict[k] += f" | dtype: {v.dtype}"

  return param_dict


def inspect_hf(args):
  """Inspects the structure and tensor shapes of HuggingFace checkpoints.

  Note: The checkpoint structure typically matches the standard `from_pretrained`
  format, with minor exceptions for specific architectures (e.g., multimodal Gemma).
  Use this to verify structure compatibility with `to_maxtext` conversion pipelines.
  """
  print(f"\n--- Inspecting {args.format} files in {args.path} ---")

  ckpt_paths = sorted(pathlib.Path(args.path).glob(f"[!.]*.{args.format}"))
  if not ckpt_paths:
    sys.exit(f"No files with extension .{args.format} found in {args.path}")

  if args.format == "safetensors":
    param_dict = _inspect_safetensors(ckpt_paths, args.check_dtype)
  elif args.format == "pth":
    param_dict = _inspect_pth(ckpt_paths, args.check_dtype)
  else:
    sys.exit(f"Unsupported format: {args.format}")

  print("\n=== Structure ===")
  print_structure(param_dict)


# ==============================================================================
# Mode 2: MaxText Architecture (On-the-fly)
# ==============================================================================


# pylint: disable=import-outside-toplevel
def inspect_maxtext(args, remaining_args):
  """Inspects MaxText model architecture, from on-the-fly execution.

  Lightweight with near-zero cost: Uses JAX's abstract tracing
  to evaluate the model's parameters dynamically. This instantly computes
  theoretical shapes and dtypes without executing actual FLOPs or allocating
  device memory (VRAM/TPU HBM).

  Optimization: Uses `jax.eval_shape`.
  """
  # Defer imports to avoid overhead when running in other modes.
  import jax
  from maxtext.utils import max_utils, maxtext_utils
  from maxtext import pyconfig
  from maxtext.utils.globals import MAXTEXT_PKG_DIR
  from maxtext.layers import quantizations
  from maxtext.models import models

  Transformer = models.transformer_as_linen

  # Configure the PyConfig environment.
  # The first argument in argv is typically the script name.
  argv = (
      [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")]
      + remaining_args
      + ["attention=dot_product", "skip_jax_distributed_system=true"]
  )
  print(argv)

  # Initialize the configuration without spinning up the heavyweight distributed runtime.
  config = pyconfig.initialize(argv)
  print(f"\n--- Inspecting MaxText Architecture: {config.model_name} (Scan: {config.scan_layers}) ---")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)

  # Extract abstract parameters. This returns a PyTree of `ShapeDtypeStruct`
  # objects, meaning zero memory payload is attached to the weights.
  abstract_param = maxtext_utils.get_abstract_param(model, config)

  # Calculate and display the total parameter count based purely on abstract shapes.
  num_params = max_utils.calculate_num_params_from_pytree(abstract_param)
  print(f"\nTotal Parameters: {num_params} (~{num_params/1e9:.2f} B)")

  print("\n=== Structure ===")

  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_param)

  param_dict = {}
  # Example of abstract_leaf_value format: ShapeDtypeStruct(shape=(128, 58), dtype=float32)
  for path_tuple, abstract_leaf_value in abstract_params_flat:
    key_parts = [k.key for k in path_tuple if hasattr(k, "key")]

    # Construct a MaxText-style parameter key (e.g., "params.layer.weight").
    param_key = "params." + ".".join(key_parts)

    shape = abstract_leaf_value.shape
    param_dict[param_key] = f"shape: {shape}"

    if args.check_dtype:
      dtype = abstract_leaf_value.dtype
      param_dict[param_key] += f" | dtype: {dtype}"

  print_structure(param_dict)


# ==============================================================================
# Mode 3: Orbax Checkpoint (Pre-saved)
# ==============================================================================


# pylint: disable=import-outside-toplevel
def inspect_orbax(args):
  """Inspects Orbax checkpoint, from pre-saved disk/GCS file.

  Lightweight with near-zero cost: Orbax separates metadata from
  heavy payload data. This method strictly parses the structural metadata tree,
  bypassing the need to pull the underlying TensorStore data chunks from disk or GCS.

  Optimization: Reads metadata rather than weights.
  """
  print(f"\n--- Inspecting Orbax Checkpoint: {args.path} ---")

  # Defer imports to avoid overhead when running in other modes.
  try:
    import orbax.checkpoint as ocp
    from etils import epath
  except ImportError:
    sys.exit("Error: 'orbax-checkpoint' or 'etils' not found. Run `pip install orbax-checkpoint etils[epath]`")

  path = epath.Path(args.path)

  # Retrieve structural metadata. Note: Depending on the Orbax version, metadata
  # access might vary slightly. This logic aligns with StandardCheckpointer usage.
  metadata = ocp.StandardCheckpointer().metadata(path)
  if hasattr(metadata, "item_metadata"):
    metadata = metadata.item_metadata

  # Flatten the nested metadata tree into a standard dictionary.
  dictionary = ocp.tree.to_flat_dict(metadata)

  # Filter strictly for parameter keys and format them.
  param_dict = {}
  for k, v in dictionary.items():
    # `k` is a tuple representing the path hierarchy; join it into a single string.
    # `v` is a metadata object containing `.shape` and `.dtype`.
    param_key = ".".join(k)
    if not param_key.startswith("params"):
      continue

    shape = v.shape
    param_dict[param_key] = f"shape: {shape}"

    if args.check_dtype:
      dtype = v.dtype
      param_dict[param_key] += f" | dtype: {dtype}"

    print(v)

  print("\n=== Structure ===")
  print_structure(param_dict)


# ==============================================================================
# Main CLI Driver
# ==============================================================================
def main():

  # Shared parser for arguments common across all modes.
  shared_parser = argparse.ArgumentParser(add_help=False)
  shared_parser.add_argument(
      "--check_dtype", type=str2bool, required=False, default=False, help="Whether to append dtype info to the output"
  )

  # Main parser and sub-parsers for distinct inspection modes.
  parser = argparse.ArgumentParser(description="Consolidated Model Checkpoint Inspector")
  subparsers = parser.add_subparsers(dest="mode", required=True, help="Inspection mode: hf, maxtext, orbax")

  # Mode 1: HuggingFace
  parser_hf = subparsers.add_parser("hf", parents=[shared_parser], help="Inspect .safetensors or .pth files")
  parser_hf.add_argument("--path", type=str, required=True, help="Directory containing checkpoint files")
  parser_hf.add_argument(
      "--format",
      type=str,
      required=False,
      choices=["safetensors", "pth"],
      default="safetensors",
      help="Checkpoint file format",
  )

  # Mode 2: MaxText Architecture
  subparsers.add_parser("maxtext", parents=[shared_parser], help="Inspect MaxText theoretical architecture")

  # Mode 3: Orbax
  parser_orbax = subparsers.add_parser("orbax", parents=[shared_parser], help="Inspect saved Orbax checkpoint metadata")
  parser_orbax.add_argument("--path", type=str, required=True, help="Path to checkpoint items (local or GCS)")

  args, remaining_args = parser.parse_known_args()

  if args.mode == "hf":
    inspect_hf(args)
  elif args.mode == "maxtext":
    inspect_maxtext(args, remaining_args)
  elif args.mode == "orbax":
    inspect_orbax(args)

  absl.logging.set_verbosity(absl.logging.INFO)
  print_peak_memory()


if __name__ == "__main__":
  main()
