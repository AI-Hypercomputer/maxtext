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


"""Utilities for Muon optimizer integration and dimension number generation.

This module provides functions to automatically generate MuonDimensionNumbers
for various MaxText models. These dimension numbers are crucial for the Muon
optimizer to correctly apply its update rules.

This module can also be run as a script to inspect the generated dimension
numbers for a specific model. Example:
  python3 -m MaxText.muon_utils qwen3-4b True
"""

import os
import sys
from typing import Optional, Tuple

import flax.linen as nn
from flax import nnx
import jax
from optax.contrib._muon import MuonDimensionNumbers as mdn

from MaxText import maxtext_utils, model_creation_utils, pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models, quantizations


def _is_path_contain_any(tuples, path):
  """Checks if any element in 'tuples' is present in 'path'."""
  return any(x in path for x in tuples)


def transform_logic(path: Tuple[str, ...]) -> Optional[mdn]:
  """
  Determines Muon dimension numbers based on the parameter's hierarchical path.

  This function defines the mapping from a parameter's logical path within the model
  to its corresponding MuonDimensionNumbers (mdn). The strategy is applied in
  a specific order to handle general cases and then more specific ones, allowing
  for fall-through logic in nested structures.

  Strategy:
  1. Exclusions: Parameters not suitable for Muon (e.g., scalars, embeddings,
     unembedding) are explicitly returned as `None`.
  2. Special Weights:
     2.1 MoE Block Specific Weights
     2.2 Self-Attention Specific Weights
  3. Standard Weights: Default mapping for most other 3D weight shapes.

  Args:
    path: A tuple of strings representing the hierarchical path of the parameter.

  Returns:
    An instance of `MuonDimensionNumbers` if a specific mapping is found,
    `None` for excluded parameters, or a default `mdn` for standard weights.
  """

  # 1 Exclude parameters not suitable for Muon (scalar, embeddings, unembedding)
  if _is_path_contain_any(("scale", "bias", "embedding", "logits_dense"), path):
    return None

  # 2 Special weights
  # 2.1 Special weights: MoE, [0, L, -2, -1]
  # L (optional) stands for layer when scan_layers=True
  if "MoeBlock_0" in path:
    # exclude gate
    if _is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      return mdn((-2,), (-1,))

  # 2.2 Special weights: Self attention
  elif "self_attention" in path:
    # Attention output projection: [0, L, -2, -1]
    if "out" in path:
      return mdn((0, -2), (-1,))
    # Attention qkv projection: [0, L, -2, -1]
    # MLA, exclude wq_a / wkv_a
    elif _is_path_contain_any(("query", "key", "value", "wq_b", "wkv_b"), path):
      return mdn((0,), (-2, -1))

  # 3 Standard weights, [0, L, -1]
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  """Extraction utility via recursion."""
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(path)


def get_muon_weight_dimension_numbers(model, config, verbose=False):
  """Extract muon dimension number from model structure."""

  if isinstance(model, nnx.Module):
    _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

    def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf):
      # Convert jax.tree_util.KeyEntry path to Tuple[str, ...]
      path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
      return transform_logic(path_strings)

    # Use jax.tree_util.tree_map_with_path for NNX's potentially complex PyTree structure.
    # This is different with linen where abstract_param is a dict-based tree with nn.LogicallyPartitioned leaves.
    muon_weight_dimension_numbers = jax.tree_util.tree_map_with_path(apply_transform_nnx, abstract_param)

  else:  # Linen
    # quickly get param structure without materialization
    abstract_param = maxtext_utils.get_abstract_param(model, config)
    # get muon dimension number from param
    muon_weight_dimension_numbers = get_transform_tree(abstract_param)

  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Prints the model structure and the resulting Muon config."""

  def get_leaf_info(leaf):
    # For linen:
    # Access the shape from the inner ShapeDtypeStruct and names from the wrapper
    # Return a new tree with the same structure containing only shapes/names
    if isinstance(leaf, nn.LogicallyPartitioned):
      return {"shape": leaf.value.shape, "names": leaf.names}
    # For nnx:
    # Only return the shape because it doesn't have a wrapper.
    elif isinstance(leaf, jax.ShapeDtypeStruct):
      return {"shape": leaf.shape}
    return {"shape": "N/A"}

  info_tree = jax.tree_util.tree_map(
      get_leaf_info,
      abstract_param,
      is_leaf=lambda x: isinstance(x, (nn.LogicallyPartitioned, jax.ShapeDtypeStruct)),
  )
  print(f"\n=== Model Structure ===\n{info_tree}")
  print(f"\n=== Muon Dimension Numbers ===\n{muon_weight_dimension_numbers}")
  print("\nIs this reasonable?")


def get_model_mdn(model_name, scan_layers=True, verbose=False, pure_nnx=False):
  """Initializes a model and retrieves its Muon dimension numbers.

  This function sets up the configuration for a given model, initializes the
  transformer model, and then extracts the Muon dimension numbers for the model's
  weights. It can optionally print verbose debug information.

  Args:
    model_name: The name of the model to be initialized.
    scan_layers: Whether to use layer scanning in the model configuration.
    verbose: If True, prints detailed debugging information about the model
      structure and Muon dimension numbers.

  Returns:
    A tree structure containing the Muon dimension numbers for the model's
    parameters.
  """
  # Setup config
  argv = [
      None,
      os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
      f"model_name={model_name}",
      f"scan_layers={scan_layers}",
      "attention=dot_product",
      f"pure_nnx={pure_nnx}",
  ]
  config = pyconfig.initialize(argv)
  # Setup model
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  if pure_nnx:
    _, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
  else:
    model = models.transformer_as_linen(config, mesh=mesh, quant=quant)
  # Get dimension number
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python3 -m MaxText.muon_utils <model_name> <scan_layers:True/False>")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True, pure_nnx=False)
