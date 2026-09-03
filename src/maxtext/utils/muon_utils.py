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

from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.optimizers.muon import MuonDimensionNumbers as mdn
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils import sharding as sharding_lib
from maxtext.utils.globals import MAXTEXT_PKG_DIR


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
    "q_proj",
    "k_proj",
    "v_proj",
)

ATTENTION_OUT_NAMES = ("out", "o_proj")

MOE_BLOCK_NAMES = (
    "MoeBlock_0",
    "moe_block",
    "routed_experts",
    "GptOssMlp",
)


def transform_logic(path: Tuple[str, ...], shape: Optional[Tuple[int, ...]] = None) -> Optional[mdn]:
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


def get_transform_tree(tree, mesh=None, path=()):
  """Recursively extracts `MuonDimensionNumbers` and shardings for Linen abstract parameters."""
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: get_transform_tree(v, mesh=mesh, path=path + (k,)) for k, v in tree.items()}
  else:
    val = getattr(tree, "value", tree)
    val_shape = getattr(val, "shape", None)
    dim_num = transform_logic(path, shape=val_shape)
    if dim_num is not None:
      names = getattr(tree, "names", None)
      if names is not None:
        # Prepend None for leading scanned layer dimensions if names length < tensor rank
        if val_shape is not None and len(names) < len(val_shape):
          diff = len(val_shape) - len(names)
          names = (None,) * diff + names
        # Resolve logical axis names to physical NamedSharding when a device mesh is available
        if mesh is not None:
          sharding = sharding_lib.create_sharding(mesh, names)
        else:
          sharding = names
      else:
        sharding = None
      return mdn(
          reduction_axis=dim_num.reduction_axis,
          output_axis=dim_num.output_axis,
          sharding=sharding,
      )
    return None


def get_muon_weight_dimension_numbers(
    model, config=None, mesh=None, verbose=False
):
  """Extracts a matching pytree of MuonDimensionNumbers with physical shardings from a model.

  Evaluates within an active `nn_partitioning.axis_rules` context to map logical
  partition axes to physical mesh axes. Supports both NNX and Linen models.
  """
  if mesh is None and config is not None and hasattr(config, "mesh_axes"):
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  logical_rules = (
      getattr(config, "logical_axis_rules", ()) if config is not None else ()
  )
  # Populate logical axis rules in Flax context for automatic NamedSharding resolution
  with nn_partitioning.axis_rules(logical_rules):
    if isinstance(model, nnx.Module):
      # Extract abstract parameters from the NNX model hierarchy
      _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

      # Resolve physical NamedSharding for each parameter under the active axis rules
      named_sharding_state = (
          sharding_lib.nnx_construct_named_sharding(abstract_param, mesh)
          if mesh is not None
          else None
      )
      abstract_dict = nnx.to_pure_dict(abstract_param)
      named_sharding_dict = (
          nnx.to_pure_dict(named_sharding_state)
          if named_sharding_state is not None
          else abstract_dict
      )

      def apply_transform_nnx(
          path: Tuple[jax.tree_util.KeyEntry, ...], leaf, abs_leaf
      ):
        path_strings = tuple(
            p.key for p in path if isinstance(p, jax.tree_util.DictKey)
        )
        abs_val = (
            abs_leaf.get_value() if hasattr(abs_leaf, "get_value") else abs_leaf
        )
        val_shape = getattr(abs_val, "shape", None)
        dim_num = transform_logic(path_strings, shape=val_shape)
        if dim_num is not None:
          val = leaf.get_value() if hasattr(leaf, "get_value") else leaf
          sharding = (
              val
              if isinstance(
                  val, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)
              )
              else getattr(leaf, "sharding", None)
          )
          if isinstance(sharding, jax.ShapeDtypeStruct) or not isinstance(
              sharding, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)
          ):
            sharding = getattr(leaf, "sharding", None)
          return mdn(
              reduction_axis=dim_num.reduction_axis,
              output_axis=dim_num.output_axis,
              sharding=sharding,
          )
        return None

      # Walk the parameter tree to produce a matching nnx.State of MuonDimensionNumbers
      muon_weight_dimension_numbers = jax.tree_util.tree_map_with_path(
          apply_transform_nnx, named_sharding_dict, abstract_dict
      )
      muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)

    else:  # Linen
      abstract_param = maxtext_utils.get_abstract_param(model, config)
      muon_weight_dimension_numbers = get_transform_tree(
          abstract_param, mesh=mesh
      )

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
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(
      model, config, mesh=mesh, verbose=verbose
  )
  if pure_nnx:
    muon_weight_dimension_numbers = {"params": nnx.to_pure_dict(muon_weight_dimension_numbers)}
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python3 -m maxtext.utils.muon_utils <model_name> <scan_layers:True/False>")
    sys.exit(1)
  model_name_arg = sys.argv[1]
  scan_layers_arg = sys.argv[2].lower() == "true"
  get_model_mdn(model_name_arg, scan_layers_arg, verbose=True, pure_nnx=False)
