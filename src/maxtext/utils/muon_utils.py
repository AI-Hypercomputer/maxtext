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
  python3 -m maxtext.utils.muon_utils qwen3-4b True
"""

import collections.abc
import os
import sys
from typing import Optional, Tuple

import flax.linen as nn
from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils, model_creation_utils
from optax.contrib._muon import MuonDimensionNumbers as mdn


def _is_path_contain_any(tuples, path):
  """Checks if any element in 'tuples' is present in 'path'."""
  return any(x in path for x in tuples)


# Parameters excluded from Muon updates (e.g. 1D norms, embeddings, scalars)
EXCLUDED_SUBSTRINGS = (
    "scale",
    "embedding",
    "logits_dense",
    "post_beta",
    "pre_beta",
    "res_beta",
    "hc_base",
    "sinks",
    "tid2eid",
    "A_log",
    "dt_bias",
    "conv1d",
    "shared_expert_gate",
)

EXCLUDED_EXACT_SEGMENTS = {
    "bias",
}

# Attention module identifiers and tensor projection names that require
# head-aware (3D/4D) dimension specifications.
ATTENTION_BLOCK_NAMES = (
    "self_attention",
    "full_attention",
    "attention",
    "self_attn",
    "attn",
    "attention_mla",
    "GptOssAttention",
)

ATTENTION_QKV_NAMES = (
    "query",
    "key",
    "value",
    "wq_b",
    "wkv_b",
    "wkv",
)

ATTENTION_OUT_NAMES = ("out",)

MOE_BLOCK_NAMES = (
    "MoeBlock_0",
    "moe_block",
    "routed_experts",
    "GptOssMlp",
    "DeepSeekMoeBlock_0",
    "Llama4MoEBlock_0",
    "routed_moe",
)


def transform_logic(
    path: Tuple[str, ...],
    shape: Optional[Tuple[int, ...]] = None,
    include_routers: bool = True,
) -> Optional[mdn]:
  """Determines Muon dimension numbers based on parameter path and shape.

  This function maps a parameter's hierarchical path within the model
  to its corresponding MuonDimensionNumbers (mdn) specifying the reduction
  and output axes for 2D matrix orthogonalization.

  In MaxText, layer scanning places the layer scan axis at index 1
  (`param_scan_axis = 1`), resulting in shapes:
    - Standard weights / MLPs: [in_features, num_layers, out_features]
    - Attention QKV: [in_features, num_layers, num_heads, head_dim]
    - Attention Out: [num_heads, num_layers, head_dim, out_features]
    - MoE routed experts: [num_experts, num_layers, in_features, out_features]
    - Grouped linear (o_a_proj): [o_groups, num_layers, in_features_per_group, out_features_per_group]

  Strategy:
    1. Exclusions: Non-matrix, 1D, scalar, embedding, or state-space parameters
       are excluded (returns None) and optimized via AdamW.
    2. MoE Routed Experts: [num_experts, ..., in_features, out_features]
       map to reduction axis (-2,) and output axis (-1,).
       Note: gate.kernel is [in_features, (num_layers), num_experts] and uses standard (0,) and (-1,).
    3. Grouped Linear: [n_groups, ..., in_features_per_group, out_features_per_group]
       maps to reduction axis (-2,) and output axis (-1,).
    4. Head-expanded Attention: QKV and Output projections with 3D/4D shapes
       map to head-aware reduction and output axes:
       - QKV: reduction (0,), output (-2, -1)
       - Out (unflattened): reduction (0, -2), output (-1,)
    5. Standard Weights: Default 2D matrix mapping (0,) and (-1,) for MLPs,
       GDN, shared experts, MHC alpha projections, and dense projections.

  Args:
    path: Tuple of strings representing the parameter's hierarchical path.
    shape: Optional shape tuple of the parameter tensor.
    include_routers: Whether to apply Muon updates to MoE router matrices. If
      False, router weights return None and are optimized with AdamW.

  Returns:
    An instance of `optax.contrib.MuonDimensionNumbers` if a valid mapping is
    found, or `None` if the parameter is excluded from Muon updates.
  """
  # Exclude 1D / scalar parameters
  if shape is not None and len(shape) < 2:
    return None

  # Exclude non-matrix parameters, embeddings, biases, and normalization
  if any(
      segment in EXCLUDED_EXACT_SEGMENTS
      or (segment.endswith("bias") and segment != "position_bias")
      or any(x in segment for x in EXCLUDED_SUBSTRINGS)
      for segment in path
  ):
    return None

  # MoE routed expert weights: [num_experts, (num_layers), in_features, out_features]
  if _is_path_contain_any(MOE_BLOCK_NAMES, path):
    if _is_path_contain_any(("wi", "wi_0", "wi_1", "wo", "gate_up_proj"), path):
      return mdn((-2,), (-1,))
    # MoE router weights (e.g. gate.kernel): [in_features, (num_layers), num_experts]
    if _is_path_contain_any(("gate", "router"), path):
      return mdn((0,), (-1,)) if include_routers else None

  # Block-diagonal grouped linear layer (e.g. DeepSeek-V4 attention output projection):
  # [n_groups, (num_layers), in_features_per_group, out_features_per_group] -> reduce (-2,), output (-1,)
  if "o_a_proj" in path:
    return mdn((-2,), (-1,))

  # Head-expanded attention projections (3D unscanned or 4D scanned)
  if _is_path_contain_any(ATTENTION_BLOCK_NAMES, path) and (shape is None or len(shape) > 2):
    if _is_path_contain_any(ATTENTION_QKV_NAMES, path):
      # [in_features, (num_layers), num_heads, head_dim] -> reduce (0,), output (-2, -1)
      return mdn((0,), (-2, -1))
    if _is_path_contain_any(ATTENTION_OUT_NAMES, path):
      # Standard attention out projection: [num_heads, (num_layers), head_dim, out_features] -> reduce (0, -2), output (-1,)
      # Note: Qwen3-Next flattens heads into a 2D projection (in_features = num_heads * head_dim),
      # so its out projection is a standard 2D matrix [in_features, (num_layers), out_features] -> reduce (0,), output (-1,)
      if _is_path_contain_any(("self_attention", "GptOssAttention"), path) or (shape is not None and len(shape) == 4):
        return mdn((0, -2), (-1,))

  # Standard 2D matrix weights (dense MLPs, shared experts, GDN, MHC alpha, dense projections, router weights)
  # [in_features, (num_layers), out_features] -> reduce (0,), output (-1,)
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=(), include_routers: bool = True):
  """Recursively extracts `MuonDimensionNumbers` for Linen abstract parameters."""
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: get_transform_tree(v, path=path + (k,), include_routers=include_routers) for k, v in tree.items()}
  else:
    val = getattr(tree, "value", tree)
    val_shape = getattr(val, "shape", None)
    return transform_logic(path, shape=val_shape, include_routers=include_routers)


def get_muon_weight_dimension_numbers(model, config=None, verbose=False):
  """Extracts a matching pytree of `MuonDimensionNumbers` from a model."""
  include_routers = getattr(config, "muon_include_routers", True) if config is not None else True

  if isinstance(model, nnx.Module):
    _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

    def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf):
      # Convert jax.tree_util.KeyEntry path to Tuple[str, ...]
      path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
      val = leaf.get_value() if hasattr(leaf, "get_value") else leaf
      val_shape = getattr(val, "shape", None)
      return transform_logic(path_strings, shape=val_shape, include_routers=include_routers)

    # NNX abstract_param is an nnx.State (not Linen's dict of LogicallyPartitioned leaves);
    # tree_map_with_path round-trips that structure so each Param.value holds the mdn result.
    muon_weight_dimension_numbers = jax.tree_util.tree_map_with_path(
        apply_transform_nnx, nnx.to_pure_dict(abstract_param)
    )
    muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)

  else:  # Linen
    # quickly get param structure without materialization
    abstract_param = maxtext_utils.get_abstract_param(model, config)
    # get muon dimension number from param
    muon_weight_dimension_numbers = get_transform_tree(abstract_param, include_routers=include_routers)

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


def get_model_mdn(model_name, scan_layers=True, verbose=False, pure_nnx=False, include_routers=True):
  """Initializes a model and retrieves its Muon dimension numbers.

  This function sets up the configuration for a given model, initializes the
  transformer model, and then extracts the Muon dimension numbers for the model's
  weights. It can optionally print verbose debug information.

  Args:
    model_name: The name of the model to be initialized.
    scan_layers: Whether to use layer scanning in the model configuration.
    verbose: If True, prints detailed debugging information about the model
      structure and Muon dimension numbers.
    pure_nnx: Whether to use pure NNX model creation.
    include_routers: Whether to apply Muon updates to MoE router matrices.

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
      f"muon_include_routers={include_routers}",
      "skip_jax_distributed_system=True",
  ]
  if not pure_nnx:
    argv.extend(
        [
            "enable_nnx=False",
            "pure_nnx_decoder=False",
        ]
    )
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
  if pure_nnx:
    muon_weight_dimension_numbers = {"params": nnx.to_pure_dict(muon_weight_dimension_numbers)}
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  if len(sys.argv) not in (3, 4):
    print("Usage: python3 -m maxtext.utils.muon_utils <model_name> <scan_layers:True/False> [include_routers:True/False]")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  include_routers_arg = sys.argv[3].lower() == "true" if len(sys.argv) == 4 else True
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True, pure_nnx=False, include_routers=include_routers_arg)
