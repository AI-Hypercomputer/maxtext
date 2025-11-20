from optax.contrib import MuonDimensionNumbers as mdn
from MaxText.maxtext_utils import get_abstract_param

from typing import Tuple, Optional
import flax.linen as nn
import os
import jax

from MaxText import pyconfig, maxtext_utils
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.layers import models, quantizations

import sys

Transformer = models.transformer_as_linen


"""Example:
python3 -m MaxText.muon_utils qwen3-4b
"""

def _is_path_contain_any(tuples, path):
  return any(x in path for x in tuples)


def transform_logic(path: Tuple[str, ...]) -> Optional[mdn]:
  """
  Determines Muon dimension numbers based on parameter path.

  Strategy:
  1. Filter out exclusions (Norms, Biases, Embeddings) -> None
  2. Special weight
    - Handle MoE specific shapes
    - Handle Attention specific shapes: output proj, qkv proj
  3. Default to standard 3D weight shape -> in=0, out=-1


  TODO(shuningjin): improve comment, notation, example
  assume scan (i.e., dim 1 is layer num L), should work with unscan (without L)
  works for deepseek, llama2, gemma3
  """

  # 1 Exclude parameters not suitable for Muon (scalar, embeddings, unembedding)
  if _is_path_contain_any(("scale", "bias", "embedding", "logits_dense"), path):
    return None

  # 2 Special weight
  # 2.1 MoE, [0, L, -2, -1]
  if "MoeBlock_0" in path:
    # Exclude gate
    if _is_path_contain_any(("wi_0", "wi_1", "wo"), path):
      return mdn((-2,), (-1,))

  # 2.2 Self attention
  if "self_attention" in path:
    # Attention output projection: [0, L, -2, -1]
    if "out" in path:
      return mdn((0, -2), (-1,))

    # Attention qkv projection: [0, L, -2, -1]
    # For MLA, exclude wq_a / wkv_a
    if _is_path_contain_any(("query", "key", "value", "wq_b", "wkv_b"), path):
      return mdn((0,), (-2, -1))

  # 3 Standard weights, [0, L, -1]
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  """recursion"""
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(path)


def get_muon_weight_dimension_numbers(model, config, verbose=False):
  """extract muon dimension number from model structure"""
  # quickly get param structure without materialization
  abstract_param = get_abstract_param(model, config)
  # get muon dimension number from param
  muon_weight_dimension_numbers = get_transform_tree(abstract_param)
  if verbose:
    _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Pretty prints the model structure and the resulting Muon config."""
  # Access the shape from the inner ShapeDtypeStruct and names from the wrapper
  # Return a new tree with the same structure containing only shapes/names
  info_tree = jax.tree_util.tree_map(
      lambda leaf: {"shape": leaf.value.shape, "names": leaf.names},
      abstract_param,
      is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
  )
  print("\n=== Model Structure ===")
  print(info_tree)
  print("\n=== Muon Dimension Numbers ===")
  print(muon_weight_dimension_numbers)
  print("\nIs this reasonable?")


def get_model_mdn(model_name, verbose=False):
  # Setup config
  argv = [None, os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"), f"model_name={model_name}", "scan_layers=True"]
  config = pyconfig.initialize(argv)
  # Setup model
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh=mesh, quant=quant)
  # Run test
  muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config, verbose=verbose)
  return muon_weight_dimension_numbers


if __name__ == "__main__":
  get_model_mdn(sys.argv[1], verbose=True)
