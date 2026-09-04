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


"""Utilities for sharded Muon optimizer integration and dimension/sharding generation.

This module provides functions to automatically generate
ShardedMuonDimensionNumbers
with physical NamedSharding for various MaxText models.
"""

import collections.abc
from typing import Tuple

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from maxtext.optimizers.muon import ShardedMuonDimensionNumbers as smdn
from maxtext.utils import maxtext_utils
from maxtext.utils import muon_utils
from maxtext.utils import sharding as sharding_lib


def get_transform_tree(tree, mesh, config=None, path=()):
  """Recursively extracts ShardedMuonDimensionNumbers and shardings for Linen abstract parameters."""
  if isinstance(tree, (dict, collections.abc.Mapping)) or hasattr(tree, "items"):
    return {k: get_transform_tree(v, mesh=mesh, config=config, path=path + (k,)) for k, v in tree.items()}
  else:
    val = getattr(tree, "value", tree)
    val_shape = getattr(val, "shape", None)
    # Determine reduction and output axes from base muon_utils
    dim_num = muon_utils.transform_logic(path, shape=val_shape)
    if dim_num is not None:
      names = getattr(tree, "names", None)
      if names is not None:
        # Insert None for scanned layer dimensions at param_scan_axis if names length < tensor rank
        if val_shape is not None and len(names) != len(val_shape):
          diff = len(val_shape) - len(names)
          if diff != 1:
            raise ValueError(
                f"Expected names length ({len(names)}) to differ from tensor rank ({len(val_shape)}) "
                f"by exactly 1 for scanned layer dimension, got difference of {diff} for path {path}."
            )
          scan_axis = getattr(config, "param_scan_axis", 1) if config is not None else 1
          names = tuple(names)
          names = names[:scan_axis] + (None,) + names[scan_axis:]
        sharding = sharding_lib.create_sharding(mesh, names)
      else:
        sharding = None
      return smdn(
          reduction_axis=dim_num.reduction_axis,
          output_axis=dim_num.output_axis,
          sharding=sharding,
      )
    return None


def _get_leaf_value(leaf):
  """Extracts value from leaf, unwrapping get_value() if present."""
  if hasattr(leaf, "get_value"):
    return leaf.get_value()
  return leaf


def _extract_sharding(leaf):
  """Extracts NamedSharding or PartitionSpec from a leaf or its attributes."""
  val = _get_leaf_value(leaf)
  if isinstance(val, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)):
    return val
  sharding = getattr(leaf, "sharding", None)
  if isinstance(sharding, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec)):
    return sharding
  return None


def get_sharded_muon_weight_dimension_numbers(model, config=None, mesh=None, verbose=False):
  """Extracts a matching pytree of ShardedMuonDimensionNumbers with physical shardings from a model."""
  if mesh is None:
    if config is None:
      raise ValueError("Either mesh or config must be provided to get_sharded_muon_weight_dimension_numbers.")
    mesh = maxtext_utils.get_mesh_from_config(config)

  if config is not None:
    logical_rules = getattr(config, "logical_axis_rules", ())
  else:
    logical_rules = ()

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(logical_rules):
    if isinstance(model, nnx.Module):
      # Extract abstract parameters from the NNX model hierarchy
      _, abstract_param, _ = nnx.split(model, nnx.Param, ...)

      # Resolve physical NamedSharding for each parameter under the active axis rules
      named_sharding_state = sharding_lib.nnx_construct_named_sharding(abstract_param, mesh)
      abstract_dict = nnx.to_pure_dict(abstract_param)
      named_sharding_dict = nnx.to_pure_dict(named_sharding_state)

      def apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf, abs_leaf):
        path_strings = tuple(p.key for p in path if isinstance(p, jax.tree_util.DictKey))
        abs_val = _get_leaf_value(abs_leaf)
        val_shape = getattr(abs_val, "shape", None)
        dim_num = muon_utils.transform_logic(path_strings, shape=val_shape)
        if dim_num is not None:
          sharding = _extract_sharding(leaf)
          return smdn(
              reduction_axis=dim_num.reduction_axis,
              output_axis=dim_num.output_axis,
              sharding=sharding,
          )
        return None

      # Walk the parameter tree to produce a matching nnx.State of ShardedMuonDimensionNumbers
      muon_weight_dimension_numbers = jax.tree.map_with_path(apply_transform_nnx, named_sharding_dict, abstract_dict)
      muon_weight_dimension_numbers = nnx.State(muon_weight_dimension_numbers)

    else:  # Linen
      abstract_param = maxtext_utils.get_abstract_param(model, config)
      muon_weight_dimension_numbers = get_transform_tree(abstract_param, mesh=mesh, config=config)

    if verbose:
      _print_structure_debug(abstract_param, muon_weight_dimension_numbers)
    return muon_weight_dimension_numbers


def _print_structure_debug(abstract_param, muon_weight_dimension_numbers):
  """Prints the model structure and the resulting Muon config."""
  return muon_utils._print_structure_debug(abstract_param, muon_weight_dimension_numbers)  # pylint: disable=protected-access
