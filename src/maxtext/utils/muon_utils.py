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
import jax
from maxtext.configs import pyconfig
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from optax.contrib._muon import MuonDimensionNumbers as mdn


def _is_path_contain_any(tuples, path):
  """Checks if any element in 'tuples' is present in 'path'."""
  return any(x in path for x in tuples)


# Parameters excluded from Muon updates (e.g. 1D norms, embeddings, scalars, routing gates)
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
    "post_alpha",
    "pre_alpha",
    "res_alpha",
)

EXCLUDED_EXACT_SEGMENTS = {
    "bias",
    "gate",
    "shared_expert_gate",
    "router",
    "moe_gate",
    "expert_gate",
    "router_weights",
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


def transform_logic(path: Tuple[str, ...], shape: Optional[Tuple[int, ...]] = None) -> Optional[mdn]:
  """Determines Muon dimension numbers based on parameter path and shape.

  This function maps a parameter's hierarchical path within the model
  to its corresponding MuonDimensionNumbers (mdn) specifying the reduction
  and output axes for 2D matrix orthogonalization.

  Negative indexing is used throughout to ensure dimension numbers remain
  invariant to leading batch and scanned-layer axes.

  Strategy:
    1. Exclusions: Non-matrix, 1D, scalar, embedding, state-space, or gating
       parameters are excluded (returns None) and optimized via AdamW.
    2. Head-expanded Attention: QKV and Output projections with 3D/4D shapes
       map to head-aware reduction and output axes.
    3. Standard Weights: Default 2D matrix mapping (-2, -1) for MLPs,
       MoE routed experts, GDN, and dense projections.

  Args:
    path: Tuple of strings representing the parameter's hierarchical path.
    shape: Optional shape tuple of the parameter tensor.

  Returns:
    An instance of `optax.contrib.MuonDimensionNumbers` if a valid mapping is
    found, or `None` if the parameter is excluded from Muon updates.
  """
  # 1. Exclude 1D / scalar parameters
  if shape is not None and len(shape) < 2:
    return None

  # 2. Exclude non-matrix parameters, embeddings, biases, normalization, and routing gates
  if any(
      segment in EXCLUDED_EXACT_SEGMENTS
      or (segment.endswith("bias") and segment != "position_bias")
      or any(x in segment for x in EXCLUDED_SUBSTRINGS)
      for segment in path
  ):
    return None

  # 3. Head-expanded attention projections (3D unscanned or 4D scanned)
  if _is_path_contain_any(ATTENTION_BLOCK_NAMES, path) and (shape is None or len(shape) > 2):
    if _is_path_contain_any(ATTENTION_QKV_NAMES, path):
      # [..., in_features, num_heads, head_dim] -> reduce (-3,), output (-2, -1)
      return mdn((-3,), (-2, -1))
    if _is_path_contain_any(ATTENTION_OUT_NAMES, path):
      # [..., num_heads, head_dim, out_features] -> reduce (-3, -2), output (-1,)
      return mdn((-3, -2), (-1,))

  # 4. Standard 2D matrix weights (dense MLPs, MoE routed experts, GDN, shared experts)
  # [..., in_features, out_features] -> reduce (-2,), output (-1,)
  return mdn((-2,), (-1,))


def get_transform_tree(tree, path=()):
  """Recursively extracts optax.contrib.MuonDimensionNumbers for Linen abstract parameters."""
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: get_transform_tree(v, path=path + (k,)) for k, v in tree.items()}
  else:
    val = getattr(tree, "value", tree)
    val_shape = getattr(val, "shape", None)
    return transform_logic(path, shape=val_shape)


def get_muon_weight_dimension_numbers(model, config=None, verbose=False):
  """Extracts a matching pytree of optax.contrib.MuonDimensionNumbers from a model."""
  if isinstance(model, nnx.Module):
    _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

    def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf):
      path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
      val = leaf.get_value() if hasattr(leaf, "get_value") else leaf
      val_shape = getattr(val, "shape", None)
      return transform_logic(path_strings, shape=val_shape)

    muon_weight_dimension_numbers = jax.tree_util.tree_map_with_path(
        apply_transform_nnx, nnx.to_pure_dict(abstract_param)
    )
    muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)

  else:  # Linen
    abstract_param = maxtext_utils.get_abstract_param(model, config)
    muon_weight_dimension_numbers = get_transform_tree(abstract_param)

  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Prints the model structure and the resulting Muon config."""

  def get_leaf_info(leaf):
    if isinstance(leaf, nn.LogicallyPartitioned):
      return {"shape": leaf.value.shape, "names": leaf.names}
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
  """Initializes a model and retrieves its Muon dimension numbers."""
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
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  if pure_nnx:
    _, model = model_creation_utils.create_nnx_abstract_model(config, mesh)
  else:
    model = models.transformer_as_linen(config, mesh=mesh, quant=quant)
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
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
