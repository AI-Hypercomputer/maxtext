#!/usr/bin/env python3
# Copyright 2023–2025 Google LLC
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

"""Compare Linen and NNX model tree structures for MaxText.

This script creates abstract models (without actual checkpoints) for both
Linen and NNX implementations and compares their parameter tree structures.

Usage:
    python compare_linen_nnx_tree.py [--model gemma2-2b]
"""

import sys
import os
import argparse

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAXTEXT_SRC_DIR = os.path.join(SCRIPT_DIR, "src", "MaxText")

# Set environment variable before importing MaxText
os.environ["MAXTEXT_PKG_DIR"] = MAXTEXT_SRC_DIR

# Add MaxText to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import linen as nn
from flax import nnx

from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.layers import models
from MaxText.layers import quantizations
from MaxText.common_types import MODEL_MODE_TRAIN

# Use our computed path
MAXTEXT_PKG_DIR = MAXTEXT_SRC_DIR


def get_tree_paths(pytree, prefix=""):
  """Recursively extract all paths from a pytree."""
  paths = []

  if isinstance(pytree, dict):
    for key, value in pytree.items():
      new_prefix = f"{prefix}/{key}" if prefix else key
      paths.extend(get_tree_paths(value, new_prefix))
  elif isinstance(pytree, (list, tuple)):
    for i, value in enumerate(pytree):
      new_prefix = f"{prefix}[{i}]"
      paths.extend(get_tree_paths(value, new_prefix))
  elif hasattr(pytree, "__dict__"):
    # Handle nnx.VariableState or similar objects
    for key, value in vars(pytree).items():
      if not key.startswith("_"):
        new_prefix = f"{prefix}.{key}" if prefix else key
        paths.extend(get_tree_paths(value, new_prefix))
  else:
    # Leaf node
    if hasattr(pytree, "shape"):
      paths.append((prefix, pytree.shape, str(pytree.dtype)))
    else:
      paths.append((prefix, type(pytree).__name__, ""))

  return paths


def extract_linen_paths(vars_dict, prefix=""):
  """Extract paths from Linen variables dict using JAX tree utilities."""
  paths = []

  # Use jax.tree_util to properly flatten the pytree
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(vars_dict)

  for path_parts, leaf in leaves_with_paths:
    # Convert path parts to string path
    path_str = ""
    for part in path_parts:
      if hasattr(part, "key"):
        # DictKey or similar
        if path_str:
          path_str += "/" + str(part.key)
        else:
          path_str = str(part.key)
      elif hasattr(part, "idx"):
        # SequenceKey (list/tuple index)
        path_str += f"[{part.idx}]"
      elif isinstance(part, str):
        if path_str:
          path_str += "/" + part
        else:
          path_str = part
      else:
        if path_str:
          path_str += "/" + str(part)
        else:
          path_str = str(part)

    # Get shape info from leaf
    if hasattr(leaf, "shape"):
      paths.append((path_str, leaf.shape, str(leaf.dtype)))
    else:
      paths.append((path_str, type(leaf).__name__, ""))

  return paths


def extract_nnx_paths(state, prefix=""):
  """Extract paths from NNX state using JAX tree utilities."""
  paths = []

  # Use jax.tree_util to properly flatten the NNX state
  leaves_with_paths = jax.tree_util.tree_leaves_with_path(state)

  for path_parts, leaf in leaves_with_paths:
    # Convert path parts to string path
    path_str = ""
    for part in path_parts:
      if hasattr(part, "key"):
        # DictKey or similar
        if path_str:
          path_str += "/" + str(part.key)
        else:
          path_str = str(part.key)
      elif hasattr(part, "idx"):
        # SequenceKey (list/tuple index)
        path_str += f"[{part.idx}]"
      elif isinstance(part, str):
        if path_str:
          path_str += "/" + part
        else:
          path_str = part
      else:
        if path_str:
          path_str += "/" + str(part)
        else:
          path_str = str(part)

    # Get shape info from leaf
    if hasattr(leaf, "shape"):
      paths.append((path_str, leaf.shape, str(leaf.dtype)))
    elif hasattr(leaf, "value") and hasattr(leaf.value, "shape"):
      paths.append((path_str, leaf.value.shape, str(leaf.value.dtype)))
    else:
      paths.append((path_str, type(leaf).__name__, ""))

  return paths


def normalize_path(path, is_linen=False):
  """Normalize a path for comparison.

  Linen format: params/params/decoder/layers/0/mlp/wi_0/kernel
  NNX format: decoder/layers/0/mlp/wi_0/kernel

  This removes the double 'params' prefix from Linen paths and handles
  other minor differences.
  """
  # Remove leading 'params/params' from Linen paths
  if is_linen and path.startswith("params/params/"):
    path = path[len("params/params/") :]
  elif is_linen and path.startswith("params/"):
    path = path[len("params/") :]

  return path


def create_linen_model_abstract(cfg, mesh):
  """Create a Linen model and get its abstract parameter structure.

  Uses pure_nnx_decoder=False to get the Linen Decoder parameters.
  """
  print("\n" + "=" * 60)
  print("Creating Linen model...")
  print("=" * 60)

  # Force pure_nnx_decoder=False for Linen model to use Linen Decoder
  # We rely on the config being set correctly externally

  quant = quantizations.configure_quantization(cfg)
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)

  # Create dummy inputs
  batch_size = cfg.global_batch_size_to_train_on
  seq_len = cfg.max_target_length

  rng = jax.random.PRNGKey(0)
  dummy_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  dummy_positions = jnp.stack([jnp.arange(seq_len, dtype=jnp.int32) for _ in range(batch_size)])
  dummy_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  # Use eval_shape to get abstract structure without allocating memory
  def init_fn():
    return model.init(
        {"params": rng, "aqt": rng, "dropout": rng},
        dummy_tokens,
        dummy_positions,
        dummy_segment_ids,
        enable_dropout=False,
    )

  with mesh:
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract_vars = jax.eval_shape(init_fn)

  return abstract_vars


def create_nnx_model_abstract(cfg, mesh):
  """Create an NNX model and get its abstract parameter structure.

  Uses pure_nnx_decoder=True to get the NNX Decoder parameters.
  The NNX Transformer class with pure_nnx_decoder=True uses NNXDecoder.
  """
  print("\n" + "=" * 60)
  print("Creating NNX model...")
  print("=" * 60)

  quant = quantizations.configure_quantization(cfg)

  def create_model():
    # Create rngs inside the function to avoid trace context issues
    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = nnx.Rngs(params=params_rng, dropout=dropout_rng)
    return models.Transformer(cfg, mesh, quant=quant, rngs=rngs, model_mode=MODEL_MODE_TRAIN)

  with mesh:
    with nn.logical_axis_rules(cfg.logical_axis_rules):
      abstract_model = nnx.eval_shape(create_model)

  # Extract state from abstract model
  _, abstract_state = nnx.split(abstract_model)

  return abstract_state


def is_rng_path(path):
  """Check if a path is RNG-related."""
  return "/rngs/" in path or path.startswith("rngs/")


def compare_tree_structures(linen_vars, nnx_state, hide_rngs=True):
  """Compare the tree structures of Linen and NNX models."""
  print("\n" + "=" * 60)
  print("Comparing tree structures...")
  if hide_rngs:
    print("(RNG paths are hidden, use --show-rngs to include them)")
  print("=" * 60)

  # Extract paths from both
  linen_paths = extract_linen_paths(linen_vars)
  nnx_paths = extract_nnx_paths(nnx_state)

  # Filter out RNG paths if requested
  if hide_rngs:
    linen_paths = [(p, s, d) for p, s, d in linen_paths if not is_rng_path(p)]
    nnx_paths = [(p, s, d) for p, s, d in nnx_paths if not is_rng_path(p)]

  print(f"\nLinen total paths: {len(linen_paths)}")
  print(f"NNX total paths: {len(nnx_paths)}")

  # Normalize paths for comparison
  linen_normalized = {}
  for path, shape, dtype in linen_paths:
    norm_path = normalize_path(path, is_linen=True)
    linen_normalized[norm_path] = (path, shape, dtype)

  nnx_normalized = {}
  for path, shape, dtype in nnx_paths:
    # Don't normalize NNX paths - compare them directly
    # (The previous bug was replacing "/value" which removed the value projection layer name)
    norm_path = path
    nnx_normalized[norm_path] = (path, shape, dtype)

  # Find matches and mismatches
  linen_only = set(linen_normalized.keys()) - set(nnx_normalized.keys())
  nnx_only = set(nnx_normalized.keys()) - set(linen_normalized.keys())
  common = set(linen_normalized.keys()) & set(nnx_normalized.keys())

  print(f"\nPaths in both: {len(common)}")
  print(f"Paths only in Linen: {len(linen_only)}")
  print(f"Paths only in NNX: {len(nnx_only)}")

  # Check for shape mismatches in common paths
  shape_mismatches = []
  for path in common:
    linen_shape = linen_normalized[path][1]
    nnx_shape = nnx_normalized[path][1]
    if linen_shape != nnx_shape:
      shape_mismatches.append((path, linen_shape, nnx_shape))

  if shape_mismatches:
    print(f"\nShape mismatches: {len(shape_mismatches)}")
    for path, linen_shape, nnx_shape in shape_mismatches:
      print(f"  {path}: Linen={linen_shape}, NNX={nnx_shape}")
  else:
    print("\nNo shape mismatches in common paths!")

  return linen_normalized, nnx_normalized, linen_only, nnx_only, common


def print_tree_structure(paths_dict, name, max_depth=3):
  """Print a hierarchical view of the tree structure."""
  print(f"\n{'='*60}")
  print(f"{name} Tree Structure (depth {max_depth}):")
  print("=" * 60)

  # Build a tree representation
  tree = {}
  for norm_path, (_, shape, dtype) in paths_dict.items():
    parts = norm_path.split("/")
    current = tree
    for i, part in enumerate(parts[:-1]):
      if i >= max_depth:
        break
      if part not in current:
        current[part] = {}
      current = current[part]
    if len(parts) <= max_depth:
      leaf_name = parts[-1] if parts else "root"
      if isinstance(shape, tuple):
        current[leaf_name] = f"{shape} {dtype}"
      else:
        current[leaf_name] = f"({shape})"

  def print_tree(d, indent=0):
    for key, value in sorted(d.items()):
      if isinstance(value, dict):
        print("  " * indent + f"{key}/")
        print_tree(value, indent + 1)
      else:
        print("  " * indent + f"{key}: {value}")

  print_tree(tree)


def main():
  parser = argparse.ArgumentParser(description="Compare Linen and NNX model tree structures")
  parser.add_argument("--model", type=str, default="gemma2-2b", help="Model config to use (e.g., gemma2-2b, llama2-7b)")
  parser.add_argument("--depth", type=int, default=4, help="Max depth for tree structure printout")
  parser.add_argument("--verbose", action="store_true", help="Print detailed path information")
  parser.add_argument("--show-rngs", action="store_true", help="Show RNG-related paths (hidden by default)")
  args = parser.parse_args()

  # Initialize config
  model_config = os.path.join(MAXTEXT_PKG_DIR, "configs", "models", f"{args.model}.yml")
  if not os.path.exists(model_config):
    print(f"Model config not found: {model_config}")
    print("Available models:")
    models_dir = os.path.join(MAXTEXT_PKG_DIR, "configs", "models")
    for f in os.listdir(models_dir):
      if f.endswith(".yml"):
        print(f"  {f[:-4]}")
    return 1

  print(f"Using model config: {args.model}")

  # Create config for Linen model (uses Linen Decoder)
  cfg_linen = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      model_name=args.model,
      per_device_batch_size=1.0,
      run_name="tree_compare",
      enable_checkpointing=False,
      max_target_length=32,  # Small for faster abstract model creation
      attention="dot_product",
      pure_nnx_decoder=False,  # Use Linen Decoder for Linen model
  )

  # Create config for NNX model (uses NNX Decoder)
  cfg_nnx = pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
      model_name=args.model,
      per_device_batch_size=1.0,
      run_name="tree_compare",
      enable_checkpointing=False,
      max_target_length=32,  # Small for faster abstract model creation
      attention="dot_product",
      pure_nnx_decoder=True,  # Use NNX Decoder for NNX model
  )

  # Create mesh (same for both)
  devices_array = maxtext_utils.create_device_mesh(cfg_linen)
  mesh = Mesh(devices_array, cfg_linen.mesh_axes)

  print(f"\nModel: {args.model}")
  print(f"emb_dim: {cfg_linen.emb_dim}")
  print(f"num_decoder_layers: {cfg_linen.num_decoder_layers}")
  print(f"num_query_heads: {cfg_linen.num_query_heads}")
  print(f"num_kv_heads: {cfg_linen.num_kv_heads}")
  print(f"vocab_size: {cfg_linen.vocab_size}")

  # Create abstract models with their respective configs
  linen_vars = create_linen_model_abstract(cfg_linen, mesh)
  nnx_state = create_nnx_model_abstract(cfg_nnx, mesh)

  # Compare structures
  hide_rngs = not args.show_rngs
  linen_normalized, nnx_normalized, linen_only, nnx_only, common = compare_tree_structures(
      linen_vars, nnx_state, hide_rngs=hide_rngs
  )

  # Print tree structures
  print_tree_structure(linen_normalized, "Linen (normalized)", max_depth=args.depth)
  print_tree_structure(nnx_normalized, "NNX (normalized)", max_depth=args.depth)

  # Print differences
  if linen_only:
    print(f"\n{'='*60}")
    print("Paths ONLY in Linen:")
    print("=" * 60)
    for path in sorted(linen_only):
      _, shape, dtype = linen_normalized[path]
      print(f"  {path}: {shape} {dtype}")

  if nnx_only:
    print(f"\n{'='*60}")
    print("Paths ONLY in NNX:")
    print("=" * 60)
    for path in sorted(nnx_only):
      _, shape, dtype = nnx_normalized[path]
      print(f"  {path}: {shape} {dtype}")

  # Summary
  print(f"\n{'='*60}")
  print("SUMMARY")
  print("=" * 60)
  print(f"Model: {args.model}")
  print(f"Total Linen paths: {len(linen_normalized)}")
  print(f"Total NNX paths: {len(nnx_normalized)}")
  print(f"Common paths: {len(common)}")
  print(f"Linen-only paths: {len(linen_only)}")
  print(f"NNX-only paths: {len(nnx_only)}")

  if len(linen_only) == 0 and len(nnx_only) == 0:
    print("\n✓ All paths match between Linen and NNX!")
  else:
    print("\n✗ There are differences between Linen and NNX paths")

  return 0


if __name__ == "__main__":
  sys.exit(main())
