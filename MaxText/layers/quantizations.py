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
from aqt.jax.v2 import aqt_tensor
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

# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel


MAX_INT8 = 127.5
MAX_INT4 = 7.5

Array = common_types.Array
Config = common_types.Config
AxisIdxes = common_types.AxisIdxes
AxisNames = common_types.AxisNames
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
KVTensor = aqt_tensor.QTensor


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
    ret.lhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=lhs_idx, tile_size=tile_size, tile_count=None))
    ret.rhs.contraction_axes.append(tiled_dot_general.AxisTiling(axis=rhs_idx, tile_size=tile_size, tile_count=None))

  return ret


def _rhs_axis_metadata_wrapper(
    x: jnp.ndarray,
    tile_map,
    no_sharding_axis: Sequence[int],
    mesh_axes: Tuple[str, ...],
    is_tiled: bool,
    replicate_scale: bool = False,
):
  if replicate_scale:
    # Temporarily using the shape to identify the scale.
    # TODO: remove the replication once the 2d sharding quantization
    # works as expected.
    if len(x.shape) == 1:
      return nn.with_logical_partitioning((lambda: x), tuple([None for _ in mesh_axes]))()

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
      if no_shard_idx < len(mesh_axes):
        mesh_axes[no_shard_idx] = None

  return nn.with_logical_partitioning((lambda: x), mesh_axes)()


@dataclass
class AqtQuantization:
  """Configures AQT quantization github.com/google/aqt."""

  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN
  replicate_scale: bool = False

  def _get_mixed_precision_cfg(self):
    quant_dg = None
    is_tiled = False
    tiling_fn = None
    module_path = "/".join(nn.module._context.module_stack[-1].path)
    for layer_name_re, layer_quant_dg in self.quant_dg.items():
      if re.fullmatch(layer_name_re, module_path):
        quant_dg, tile_size = layer_quant_dg
    if quant_dg is None:
      quant_dg, tile_size = self.quant_dg[DEFAULT]
    if tile_size != -1:
      is_tiled = True
      tiling_fn = functools.partial(_tiling_fn, tile_size=tile_size)
    return quant_dg, is_tiled, tiling_fn

  def _get_rhs_axis_metadata_wrapper(
      self, mesh_axes: Tuple[str, ...] = (), is_tiled: bool = False, replicate_scale: bool = False
  ):
    if self.quant_mode == aqt_flax.QuantMode.CONVERT:
      return None
    return functools.partial(
        _rhs_axis_metadata_wrapper, mesh_axes=mesh_axes, is_tiled=is_tiled, replicate_scale=replicate_scale
    )

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None
    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    # module_path = "/".join(nn.module._context.module_stack[-1].path)
    # print(f"quant_dg: {quant_dg}, is_tiled: {is_tiled}, module_path: {module_path}")
    aqt_dg_cls = functools.partial(
        aqt_flax.AqtDotGeneral,
        quant_dg,
        rhs_quant_mode=self.quant_mode,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
        rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
        use_legacy_freezer=False,
        tiling_fn=tiling_fn,
    )
    return aqt_dg_cls

  def einsum(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns einsum configured with aqt params."""
    if isinstance(self.quant_dg, dict):
      quant_dg, is_tiled, tiling_fn = self._get_mixed_precision_cfg()
    else:
      quant_dg, is_tiled, tiling_fn = self.quant_dg, False, None

    rhs_axis_metadata_wrapper = self._get_rhs_axis_metadata_wrapper(
        mesh_axes, is_tiled, replicate_scale=self.replicate_scale
    )
    aqt_einsum = functools.partial(
        aqt_flax.AqtEinsum(
            cfg=quant_dg,
            rhs_quant_mode=self.quant_mode,
            lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
            rhs_freeze_mode=aqt_flax.FreezerMode.CALIBRATION_AND_VALUE,
            rhs_axis_metadata_wrapper=rhs_axis_metadata_wrapper,
            use_legacy_freezer=False,
            tiling_fn=tiling_fn,
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
    drhs_local_aqt = aqt_config.LocalAqt(contraction_axis_shard_count=config.quantization_local_shard_count)
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


def _dot_general_make(quant_cfg):
  lhs_bits = quant_cfg[_A_BITS]
  lhs_scale = quant_cfg[_A_SCALE]
  rhs_bits = quant_cfg[_W_BITS]
  rhs_scale = quant_cfg[_W_SCALE]
  aqt_dg = aqt_config.dot_general_make(lhs_bits=lhs_bits, rhs_bits=rhs_bits)
  if lhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(calibration.AbsMaxCalibration, scale=lhs_scale)
  if rhs_scale < 1.0:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(calibration.AbsMaxCalibration, scale=rhs_scale)
  return aqt_dg


def _get_default_mp_config(default=None):
  default_config = {_W_BITS: None, _A_BITS: None, _W_SCALE: 1.0, _A_SCALE: 1.0, _TILE_SIZE: -1}
  if default:
    for k in default_config.keys():
      default_config[k] = default.get(k, default_config[k])
  return default_config


def _get_mixed_precision_quant_config(mixed_precision_config):
  """Set quantization params based on user configuration."""
  ret_config = {}
  default_mp_config = _get_default_mp_config(default=mixed_precision_config.get(DEFAULT, None))
  for layer_name_re, layer_quantization_config in mixed_precision_config.items():
    # Make a copy of default_mp_config to avoid updaing original dict
    quant_config = default_mp_config.copy()
    # print(f"Mixed precision config: processing {layer_name_re} - {layer_quantization_config}, default config - {quant_config}")
    if layer_name_re != DEFAULT:
      for k in quant_config.keys():
        quant_config[k] = layer_quantization_config.get(k, default_mp_config[k])
    ret_config[layer_name_re] = [_dot_general_make(quant_config), quant_config["tile_size"]]
  return ret_config


def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == "":
    return None
  if config.quantization == "int8":
    return _get_int8_quant_config(config)
  if config.quantization == "intmp":
    assert config.quant_cfg_path, "Must specify quant_cfg for mixed precision quantization"
    with open(config.quant_cfg_path, "r") as config_file:
      mixed_precision_config = json.load(config_file)
    return _get_mixed_precision_quant_config(mixed_precision_config)
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
    replicate_scale = config.replicate_quant_scale if config.replicate_quant_scale else False
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode, replicate_scale=replicate_scale)
  return None


def match_aqt_and_unquantized_param(aqt_params, params):
  aqt_param_flat, aqt_tree_def = jax.tree_util.tree_flatten_with_path(
      aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor)
  )
  param_tree_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  aqt_paths = []
  # Orginal path of quantized AQT param path.
  param_paths = []

  for aqt_k, _ in aqt_param_flat:
    for index, (k, _) in enumerate(param_tree_flat):
      path_depth = len(k)
      # every quantized parameter has AQT.. as the leaf node
      # AqtDotGeneral and AqtEinsum replace leaf node.
      # Therefore, leaf node should be ignored for path matching
      if k[: path_depth - 1] == aqt_k[: path_depth - 1]:
        aqt_paths.append(aqt_k)
        param_paths.append(k)
        break
    # since the parameter is already added, we can delete it.
    param_tree_flat.pop(index)
  return jax.tree_util.tree_unflatten(aqt_tree_def, param_paths)


def _get_aqt_key_paths(aqt_vars, params):
  """Generate a list of paths which have aqt state"""
  aqt_to_unquantized_key_path = match_aqt_and_unquantized_param(aqt_vars, params)
  aqt_key_paths, _ = jax.tree_util.tree_flatten(aqt_to_unquantized_key_path, is_leaf=lambda x: isinstance(x, tuple))
  return list(aqt_key_paths)


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  quantized_param_paths = _get_aqt_key_paths(aqt_vars, params)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in quantized_param_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)


def configure_kv_quant(config):
  return None if not config.quantize_kvcache else KVQuant(config)


class KVQuant:
  axis_cfg = ""
  dtype = None

  def __init__(self, config: Config):
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
      return (axis_names.index(CACHE_HEADS), axis_names.index(CACHE_KV))
    raise ValueError(f"Invalid KV quant axis cfg: {self.axis_cfg}")

  def quantize(self, kv: Array, axis_names: AxisNames):
    """Quantize key/values stored in kvcache."""
    assert self.axis_cfg, "KV quant axis cannot be None"
    max_axis = self._get_max_axis(axis_names)
    scale = jnp.max(jnp.abs(kv), axis=max_axis, keepdims=True)
    if self.dtype == jnp.int8:
      value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
      return value, scale
    if self.dtype == jnp.int4:
      value = jnp.int4(jnp.rint(kv * (MAX_INT4 / scale)))
      return value, scale
    raise ValueError(f"Invalid KV quant dtype:{self.dtype}.")

  def einsum_fn_with_rhs_qtensor(
      self,
      kv: Array | aqt_tensor.QTensor,
      rhs_dequant_mode=None,
      rhs_calibration_mode=None,
  ):
    # Assumes kv is already quantized.
    einsum = jnp.einsum
    if isinstance(kv, aqt_tensor.QTensor):
      num_bits = 4 if kv.qvalue.dtype == jnp.int4 else 8
      kv_cfg = aqt_config.dot_general_make(
          lhs_bits=None,
          rhs_bits=num_bits,
          bwd_bits=None,
          use_fwd_quant=False,
      )
      if rhs_dequant_mode:
        aqt_config.set_fwd_dequant_mode(kv_cfg, rhs_dequant_mode=rhs_dequant_mode)
      if rhs_calibration_mode:
        aqt_config.set_fwd_calibration_mode(
            kv_cfg,
            rhs_calibration_mode=rhs_calibration_mode,
        )
      einsum = aqt_flax.AqtEinsum(
          rhs_quant_mode=aqt_flax.QuantMode.TRAIN,
          lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          rhs_freeze_mode=aqt_flax.FreezerMode.NONE,
          cfg=kv_cfg,
      )
    return einsum

  def einsum_fn_with_rhs_qtensor_and_dequant(self, value):
    return self.einsum_fn_with_rhs_qtensor(
        value,
        rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
        rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
    )
