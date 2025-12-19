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
import jax
from optax.contrib import MuonDimensionNumbers as mdn

from MaxText import maxtext_utils, pyconfig
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models, quantizations


Transformer = models.transformer_as_linen


def _is_path_contain_any(tuples, path):
  return any(x in path for x in tuples)


def transform_logic(path: Tuple[str, ...]) -> Optional[mdn]:
  """
  Determines Muon dimension numbers based on the parameter's hierarchical path.
  
  Strategy:
  1. Exclusions: Skip vectors/biases/embeddings (AdamW).
  2. MoE: Handle both DeepSeek style (MoeBlock_0) and Qwen3-Next style (routed_experts).
  3. Attention: 
     - "self_attention" (Llama/DeepSeek): 'out' is 4D -> (0, -2) reduction.
     - "attention" (Qwen3-Next): 'out' is 3D -> (0,) reduction.
  4. Standard: Default 3D weights -> (0,) reduction.
  """

  # 1. Exclusions
  if _is_path_contain_any(("scale", "bias", "embedding", "logits_dense", "A_log", "conv1d", "dt_bias"), path):
    return None

  # 2. MoE Weights
  # Case A: DeepSeek / Standard MoE (MoeBlock_0)
  if "MoeBlock_0" in path:
    # DeepSeek/Standard MoE experts: (Experts, Layers, In, Out) -> reduce on In (-2)
    if _is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      return mdn((-2,), (-1,))
    # Gate is usually (Layers, In, Experts) or similar standard 3D
    if "gate" in path:
      return mdn((0,), (-1,))
  
  # Case B: Qwen3-Next MoE (routed_experts)
  if "routed_experts" in path:
    # Qwen3-Next experts: (Experts, Layers, In, Out) -> reduce on In (-2)
    if _is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      return mdn((-2,), (-1,))
    # Gate
    if "gate" in path:
      return mdn((0,), (-1,))

  # 3. Attention Weights
  # Case A: Standard Llama/DeepSeek (uses "self_attention" in path)
  if "self_attention" in path:
    # Attention Output: usually 4D (Heads, Layers, HeadDim, Embed)
    # We reduce on Heads (0) and HeadDim (-2) to get back to Embed (-1)
    if "out" in path:
      return mdn((0, -2), (-1,))

    # QKV / MLA Projections
    # Input (Embed) -> Output (Heads, HeadDim)
    # Reduce on Embed (0), Output on Heads (-2) and HeadDim (-1)
    if _is_path_contain_any(("query", "key", "value", "wq_a", "wq_b", "wkv_a", "wkv_b"), path):
      return mdn((0,), (-2, -1))

  # Case B: Qwen3-Next (uses "attention" in path, but NOT "self_attention")
  # Qwen3-Next's structure is typically 'layer_x' -> 'attention' (wrapper) -> 'attention' (inner)
  elif "attention" in path:
    # Attention Output: Qwen3-Next 'out' is 3D (Hidden, Layers, Embed) -> Standard reduction
    if "out" in path:
      return mdn((0,), (-1,))
    
    # QKV Projections
    if _is_path_contain_any(("query", "key", "value"), path):
      return mdn((0,), (-2, -1))
    
    # GDN Projections (in_proj_ba, in_proj_qkvz, out_proj) -> Standard 3D
    if _is_path_contain_any(("in_proj", "out_proj"), path):
      return mdn((0,), (-1,))

  # 4. Standard Weights (Default Fallback)
  # Handles Dense layers (mlp), Shared Experts, and other 3D projections.
  # Assumes (In, Layers, Out) or similar where 0 is Input/Reduction and -1 is Output.
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  """Extraction utility via recursion."""
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(path)


def get_muon_weight_dimension_numbers(model, config, verbose=False):
  """Extract muon dimension number from model structure."""
  # quickly get param structure without materialization
  abstract_param = maxtext_utils.get_abstract_param(model, config)
  # get muon dimension number from param
  muon_weight_dimension_numbers = get_transform_tree(abstract_param)
  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Pretty prints the model structure and the resulting Muon config."""
  
  def _get_leaf_info(leaf):
    # Case 1: flax.linen.LogicallyPartitioned (Wrapped)
    if hasattr(leaf, "value") and hasattr(leaf.value, "shape"):
      return {"shape": leaf.value.shape, "names": getattr(leaf, "names", None)}
    # Case 2: jax.ShapeDtypeStruct or raw Array (Unwrapped)
    elif hasattr(leaf, "shape"):
      return {"shape": leaf.shape, "names": None}
    # Fallback
    return {"shape": "unknown", "names": None}

  # Access the shape from the inner ShapeDtypeStruct and names from the wrapper
  # Return a new tree with the same structure containing only shapes/names
  info_tree = jax.tree_util.tree_map(
      _get_leaf_info,
      abstract_param,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  print("\n=== Model Structure ===")
  print(info_tree)
  print("\n=== Muon Dimension Numbers ===")
  print(muon_weight_dimension_numbers)
  print("\nIs this reasonable?")


def get_model_mdn(model_name, scan_layers=True, verbose=False):
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
  ]
  config = pyconfig.initialize(argv)
  # Setup model
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)
  # Get dimension number
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python3 -m MaxText.muon_utils <model_name> <scan_layers:True/False>")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True)