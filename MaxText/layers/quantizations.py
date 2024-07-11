#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools
import json
import re
from typing import Optional

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import calibration
import common_types
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, tree_unflatten
from typing import Tuple, Sequence

MAX_INT8 = 127.5
MAX_INT4 = 7.5

Array = common_types.Array
Config = common_types.Config
AxisIdxes = common_types.AxisIdxes
AxisNames = common_types.AxisNames
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    pass


def _tiling_fn(lhs, rhs, dimension_numbers, tile_size):
  del lhs, rhs

  (lhs_ca, rhs_ca), _ = dimension_numbers
  ret = tiled_dot_general.Cfg(
      lhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
      rhs=tiled_dot_general.TensorTiling(contraction_axes=[], remaining_axes=[]),
  )

  for lhs_idx, rhs_idx in zip(lhs_ca, rhs_ca):
    ret.lhs.contraction_axes.append(
        tiled_dot_general.AxisTiling(axis=lhs_idx, tile_size=tile_size, tile_count=None)
    )
    ret.rhs.contraction_axes.append(
        tiled_dot_general.AxisTiling(
            axis=rhs_idx, tile_size=tile_size, tile_count=None
        )
    )

  return ret


def _rhs_axis_metadata_wrapper(x: jnp.ndarray, tile_map, no_sharding_axis: Sequence[int], mesh_axes: Tuple[str, ...], is_tiled: bool):
  mesh_axes = list(mesh_axes)
  if is_tiled:
    # tile_map is a mapping between original rank and a list of new, tiled rank.
    if len(mesh_axes) < len(tile_map):
      mesh_axes = [None] * (len(tile_map) - len(mesh_axes)) + mesh_axes
    new_mesh_axes = [None] * len(x.shape)
    for orig_rank, new_rank in tile_map.items():
      assert new_rank
      assert len(new_rank) <= 2
      new_mesh_axes[new_rank[-1]] = mesh_axes[orig_rank]
    mesh_axes = new_mesh_axes

  if mesh_axes is not None and len(mesh_axes) > 0:
    for no_shard_idx in no_sharding_axis:
      mesh_axes[no_shard_idx] = None

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()


@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN

  def _get_mixed_precision_cfg(self):
    quant_dg = None
    is_tiled=False
    tiling_fn=None
    module_path = '/'.join(nn.module._context.module_stack[-1].path)
    for layer_name_re, layer_quant_dg in self.quant_dg.items():
      if re.fullmatch(layer_name_re, module_path):
        quant_dg, tile_size = layer_quant_dg
    if quant_dg is None:
      quant_dg, tile_size = self.quant_dg['default']
    if tile_size != -1:
      is_tiled=True
      tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    return quant_dg, is_tiled, tiling_fn

  def _get_rhs_axis_metadata_wrapper(self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(_rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled)

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn  = self.quant_dg, False, None
    rhs_axis_metadata_wrapper=self._get_rhs_axis_metadata_wrapper(
      mesh_axes, is_tiled)
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    rhs_axis_metadata_wrapper=self._get_rhs_axis_metadata_wrapper(
      mesh_axes)
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=self.quant_dg,
            lhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
        )
    )
    return aqt_einsum


@dataclass
class Fp8Quantization(Quantization):
  """Configures Fp8 quantization for NVIDIA GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.Fp8DotGeneralOp

def _get_int8_quant_config(config):
  drhs_bits = None
  drhs_accumulator_dtype = None
  drhs_local_aqt = None
  if config.quantization_local_shard_count != 0:
    drhs_bits = 8
    drhs_accumulator_dtype = jnp.int32
    drhs_local_aqt = aqt_config.LocalAqt(
      contraction_axis_shard_count=config.quantization_local_shard_count
      )
  return aqt_config.config_v3(
    fwd_bits=8,
    dlhs_bits=8,
    drhs_bits=drhs_bits,
    rng_type="jax.uniform",
    dlhs_local_aqt=None,
    drhs_local_aqt=drhs_local_aqt,
    fwd_accumulator_dtype=jnp.int32,
    dlhs_accumulator_dtype=jnp.int32,
    drhs_accumulator_dtype=drhs_accumulator_dtype,
    )


def _get_weight_only_quant_config(lhs_bits=None, rhs_bits=None):
  return aqt_config.dot_general_make(lhs_bits=lhs_bits, rhs_bits=rhs_bits)


def _get_mixed_precision_quant_config(config, config_file):
  """Set quantization params based on user configuration."""
  with open(config_file, "r") as infile:
    mixed_precision_config = json.load(infile)
  ret_config = {}
  ret_config["default"] = [aqt_config.dot_general_make(lhs_bits=None, rhs_bits=8), -1]
  for layer_name_re, layer_quantization_config in mixed_precision_config.items():
    rhs_num_bits = layer_quantization_config.get("bits", 8)
    tile_size = layer_quantization_config.get("tile_size", -1)
    scale = layer_quantization_config.get("scale", 1.0)
    aqt_dg = aqt_config.dot_general_make(lhs_bits=None, rhs_bits=rhs_num_bits)
    if scale < 1.0:
      aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(
        calibration.AbsMaxCalibration, scale=scale)
    ret_config[layer_name_re] = [aqt_dg, tile_size]
  return ret_config


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization == "int8":
    return _get_int8_quant_config(config)
  if config.quantization == "int8w":
    return _get_weight_only_quant_config(lhs_bits=None, rhs_bits=8)
  if config.quantization == "int4w":
    return _get_weight_only_quant_config(lhs_bits=None, rhs_bits=4)
  if config.quantization == "intmp":
    assert config.quant_cfg_path, "Must specify quant_cfg for mixed precision quantization"
    return _get_mixed_precision_quant_config(config, config.quant_cfg_path)
  if config.quantization == "fp8":
    return "fp8"
  raise ValueError(f"Invalid value configured for quantization {config.quantization}.")


def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)


def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)


def get_quant_mode(quant_mode_str: str = "train"):
  """Set quant mode."""
  if quant_mode_str == "train":
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == "serve":
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == "convert":
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f"Invalid quantization mode {quant_mode_str}.")
  return None


def configure_quantization(config: Config, quant_mode_str: str = "train"):
  """Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
      return Fp8Quantization()
    quant_mode = get_quant_mode(quant_mode_str)
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode)
  return None


def _get_aqt_key_paths(aqt_vars):
  """Generate a list of paths which have aqt state"""
  aqt_tree_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_vars)
  aqt_key_paths = []
  for k, _ in aqt_tree_flat:
    pruned_keys = []
    for d in list(k):
      if "AqtDotGeneral" in d.key:
        pruned_keys.append(jax.tree_util.DictKey(key="kernel"))
        break
      else:
        assert "Aqt" not in d.key, f"Unexpected Aqt op {d.key} in {k}."
        pruned_keys.append(d)
    aqt_key_paths.append(tuple(pruned_keys))
  return aqt_key_paths


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  aqt_paths = _get_aqt_key_paths(aqt_vars)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in aqt_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)

def configure_kv_quant(config):
  return None if not config.quantize_kvcache else KVQuant(config)

class KVQuant:
  axis_cfg = ""
  dtype = None

  def __init__(self, config:Config):
    assert config.quantize_kvcache
    self.axis_cfg = config.kv_quant_axis
    self.dtype = self._get_dtype(config.kv_quant_dtype)

  def _get_dtype(self, dtype_cfg: str):
    if dtype_cfg == "int4":
      return jnp.int4
    if dtype_cfg == "int8":
      return jnp.int8
    raise ValueError(f"Invalid kv_quant_dtype: {dtype_cfg}")

  def _get_max_axis(self, axis_names: AxisNames):
    if self.axis_cfg == "dkv":
      return axis_names.index(CACHE_KV)
    if self.axis_cfg == "heads_and_dkv":
      return (
        axis_names.index(CACHE_HEADS),
        axis_names.index(CACHE_KV)
        )
    raise ValueError(f"Invalid KV quant axis cfg: {self.axis_cfg}")

  def quantize(self, kv: Array, axis_names: AxisNames):
    """Quantize key/values stored in kvcache."""
    assert self.axis_cfg, 'KV quant axis cannot be None'
    max_axis = self._get_max_axis(axis_names)
    scale = jnp.max(jnp.abs(kv), axis=max_axis, keepdims=True)
    if self.dtype == jnp.int8:
      value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
      return value, scale
    if self.dtype == jnp.int4:
      value = jnp.int4(jnp.rint(kv * (MAX_INT4 / scale)))
      return value, scale
    raise ValueError(f"Invalid KV quant dtype:{self.dtype}.")


  def unquantize(self, value: Array, scale: Array, dtype: jnp.dtype):
    """Unquantize key/values stored in kvcache."""
    if self.dtype == jnp.int8:
      return value.astype(dtype) * scale / MAX_INT8
    if self.dtype == jnp.int4:
      return value.astype(dtype) * scale / MAX_INT4
    raise ValueError(f"Invalid KV quant dtype: {self.dtype}.")

