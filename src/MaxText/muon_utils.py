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
  Revised for Qwen3-Next hybrid architecture.
  """

  # 1. Exclusions: Parameters not suitable for Muon
  # Added "dt_bias" to the exclusion list.
  if _is_path_contain_any(("scale", "bias", "embedding", "logits_dense", "A_log", "conv1d", "dt_bias"), path):
    return None

  # 2. Special weights: MoE Routed Experts
  # Qwen3-Next routed experts are 4D: (NumExperts, Layers, In, Out) or similar.
  # We want to treat Experts and Layers as batch dimensions, and perform 
  # matrix multiplication on the last two dimensions.
  if "routed_experts" in path:
    if _is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      # reduction on input_dim (-2), output on output_dim (-1)
      return mdn((-2,), (-1,))

  # 3. Special weights: Attention (Full Attention layers)
  # Qwen3-Next uses "attention" in the path for layer_3, not just "self_attention"
  if "attention" in path or "self_attention" in path:
    # Attention output projection
    # For 3D weights (Hidden, Layer, Embed), standard (0,), (-1,) works fine.
    # For 4D weights (Heads, Layer, HeadDim, Embed), we might want (0, -2).
    # We stick to standard logic for 'out' if it falls through, or define specific if needed.
    if "out" in path:
       # If the shape is 4D (Heads, Layers, HeadDim, Embed), this groups Heads and HeadDim.
       # If the shape is 3D (Hidden, Layers, Embed), 0 is Hidden, -1 is Embed.
       # The safest generic strategy for 'out' is often just mapping input->output.
       # Let's allow 'out' to fall through to the default (0,), (-1,) which works for
       # both Qwen3's 3D out_proj and standard cases.
       pass

    # Attention qkv projection
    # Standard: Input (Embed) -> Output (Heads, HeadDim)
    # We want to group Heads and HeadDim as the output axis.
    if _is_path_contain_any(("query", "key", "value", "wq_b", "wkv_b"), path):
      return mdn((0,), (-2, -1))

  # 4. Standard weights: [0, L, -1]
  # Handles Dense layers, Shared Experts, MoE Gates, and Attention Out (3D)
  # Assumes dim 0 is reduction (Fan-In) and dim -1 is Output (Fan-Out)
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