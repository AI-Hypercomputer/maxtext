from optax.contrib import MuonDimensionNumbers as mdn
from MaxText.maxtext_utils import get_abstract_param
from typing import Tuple, Optional
from flax.linen import partitioning as nn_partitioning



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

  # 3 Standard 3D weights, [0, L, -1]
  return mdn((0,), (-1,))


def get_transform_tree(tree, path=()):
  """recursion"""
  if isinstance(tree, dict):
    return {k: get_transform_tree(v, path + (k,)) for k, v in tree.items()}
  else:
    return transform_logic(path)


def get_muon_weight_dimension_numbers(model, config):
  """extract muon dimension number from model structure"""
  # quickly get param structure without materialization
  with model.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_param = get_abstract_param(model, config)
  print(abstract_param)
  # get muon dimension number
  muon_weight_dimension_numbers = get_transform_tree(abstract_param)
  print("dimension number:", muon_weight_dimension_numbers)
  return muon_weight_dimension_numbers