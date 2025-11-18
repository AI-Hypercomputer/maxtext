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

"""Quantization library."""

import functools
import json
import re
from typing import Tuple, Sequence
from dataclasses import dataclass

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import calibration

import qwix

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, tree_unflatten

from flax.linen import fp8_ops
from flax.linen import initializers as flax_initializers
import flax.linen as nn

from MaxText.common_types import DType, Config
from MaxText.inference.kvcache import KVQuant

# Params used to define mixed precision quantization configs
DEFAULT = "__default__"  # default config
_W_BITS = "w_bits"  # Number of bits used to represent weights
_A_BITS = "a_bits"  # Number of bits used to represent activations
_W_SCALE = "w_scale"  # Clipping scale for weights
_A_SCALE = "a_scale"  # Clipping scale for activations
_TILE_SIZE = "tile_size"  # Tile size for subchannel


@dataclass
class Quantization:
  """Base class for quantization configurations"""

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""


def _tiling_fn(lhs, rhs, dimension_numbers, tile_size):
  """apply tiling function"""
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
  """right-hand-side axis metadata wrapper"""
  if replicate_scale:
    # Temporarily using the shape to identify the scale.
    # TODO: remove the replication once the 2d sharding quantization
    # works as expected.
    if len(x.shape) == 1:
      return nn.with_logical_partitioning((lambda: x), tuple(None for _ in mesh_axes))()

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
    """get configuration for mixed precision"""
    quant_dg = None
    is_tiled = False
    tiling_fn = None
    # pylint: disable=protected-access
    module_path = "/".join(nn.module._context.module_stack[-1].path)
    tile_size = -1
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
    return nn.Fp8DirectDotGeneralOp

  def einsum(self, dtype: DType = jnp.float32):
    return _Fp8EinsumWrapper(dtype=dtype)


class _Fp8EinsumWrapper(nn.Module):
  """Wrapper for nn.Fp8Einsum to handle computation dtype."""

  dtype: DType

  @nn.compact
  def __call__(self, eqn, lhs, rhs, **kwargs):
    # nn.Fp8Einsum determines compute dtype from rhs.
    # We cast rhs to the desired computation dtype.
    # nn.Fp8Einsum will then cast lhs to the same dtype.
    rhs = rhs.astype(self.dtype)
    return nn.Fp8Einsum(name="fp8_einsum")(eqn, lhs, rhs, **kwargs)


class Fp8Einsum(nn.Module):
  """An fp8 einsum op.

  Attributes:
    amax_history_length: size of the amax history.
    e4m3_dtype: e4m3 variants, e.g., e4m3fn, e4m3fnuz.
    e5m2_dtype: e5m2 variants, e.g., e5m2, e5m2fnuz.
    dtype: computation dtype.
  """

  amax_history_length: int = 1024
  e4m3_dtype: DType = jnp.float8_e4m3fn
  e5m2_dtype: DType = jnp.float8_e5m2
  dtype: DType = jnp.float32

  def setup(self) -> None:
    """init with input_amax_history, kernel_amax_history, output_grad_amax_history,
    input_scale, kernel_scale, output_grad_scale"""
    scale_args = (
        flax_initializers.ones_init(),
        jax.random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        flax_initializers.zeros_init(),
        jax.random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
    self.input_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "input_amax_history", *amax_history_args)
    self.kernel_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_amax_history", *amax_history_args)
    self.output_grad_amax_history = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_amax_history", *amax_history_args)

    self.input_scale = self.variable(OVERWRITE_WITH_GRADIENT, "input_scale", *scale_args)
    self.kernel_scale = self.variable(OVERWRITE_WITH_GRADIENT, "kernel_scale", *scale_args)
    self.output_grad_scale = self.variable(OVERWRITE_WITH_GRADIENT, "output_grad_scale", *scale_args)

  def __call__(self, eqn, *args, **kwargs):
    assert len(args) == 2
    x = args[0]
    k = args[1]

    comp_dtype = self.dtype
    k = jnp.asarray(k, comp_dtype)
    x = jnp.asarray(x, comp_dtype)

    x_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, x, self.input_scale.value, self.input_amax_history.value)
    k_qdq = fp8_ops.in_qdq(comp_dtype, self.e4m3_dtype, k, self.kernel_scale.value, self.kernel_amax_history.value)

    y_qdq = jnp.einsum(eqn, x_qdq, k_qdq, _dot_general=fp8_ops.dot_general_with_precision)

    y = fp8_ops.out_qdq(
        comp_dtype,
        self.e5m2_dtype,
        y_qdq,
        self.output_grad_scale.value,
        self.output_grad_amax_history.value,
    )
    return y


@dataclass
class NANOOFp8Quantization(Quantization):
  """Configures NANOO Fp8 quantization for AMD MI300/MI325 GPUs"""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Returns dot_general configured with aqt params."""
    return nn.NANOOFp8DotGeneralOp


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


@dataclass(frozen=True)
class ConstantBoundConfig:
  fwd_lhs_bound: float | None = None
  fwd_rhs_bound: float | None = None
  dlhs_lhs_bound: float | None = None
  dlhs_rhs_bound: float | None = None
  drhs_lhs_bound: float | None = None
  drhs_rhs_bound: float | None = None


def _build_const_scale_config(
    aqt_dg: aqt_config.DotGeneral,
    cst_bound_config: ConstantBoundConfig,
) -> aqt_config.DotGeneral:
  """Build a constant scale config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    cst_bound_config: The constant bound config.

  Returns:
    The AQT dot general config with constant scale config.
  """
  if cst_bound_config.fwd_lhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_lhs_bound
    )
  if cst_bound_config.fwd_rhs_bound is not None:
    aqt_dg.fwd.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.fwd_rhs_bound
    )
  if cst_bound_config.dlhs_lhs_bound:
    aqt_dg.dlhs.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_lhs_bound
    )

  if cst_bound_config.dlhs_rhs_bound is not None:
    aqt_dg.dlhs.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.dlhs_rhs_bound
    )

  if cst_bound_config.drhs_lhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.lhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_lhs_bound
    )

  if cst_bound_config.drhs_rhs_bound is not None:
    aqt_dg.drhs.dg_quantizer.rhs.calibration = functools.partial(
        calibration.ConstantCalibration, bound=cst_bound_config.drhs_rhs_bound
    )

  return aqt_dg


@dataclass
class PerTensorScales:
  fwd_lhs: bool = False
  fwd_rhs: bool = False
  dlhs_lhs: bool = False
  dlhs_rhs: bool = False
  drhs_lhs: bool = False
  drhs_rhs: bool = False


def _build_per_tensor_config(
    aqt_dg: aqt_config.DotGeneral,
    per_tensor_scales: PerTensorScales,
) -> aqt_config.DotGeneral:
  """Build a per tensor config for AQT dot general.

  Args:
    aqt_dg: The AQT dot general config.
    per_tensor_scales: The per tensor scales config.

  Returns:
    The AQT dot general config with per tensor config.
  """
  if per_tensor_scales.fwd_lhs:
    aqt_dg.fwd.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.fwd_rhs:
    aqt_dg.fwd.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.dlhs_lhs:
    aqt_dg.dlhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.dlhs_rhs:
    aqt_dg.dlhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.drhs_lhs:
    aqt_dg.drhs.dg_quantizer.lhs.calib_shared_axes = "per_tensor"
  if per_tensor_scales.drhs_rhs:
    aqt_dg.drhs.dg_quantizer.rhs.calib_shared_axes = "per_tensor"
  return aqt_dg


# fp8 training recipe of dynamic scaling with configurable constant_bound_config for static scaling option
def _get_aqt_fp8_default_config(config):
  """Get aqt for 8-bit floating point quantization configuration."""
  aqt_dg = aqt_config.config_v4(
      fwd_bits="e4m3",
      dlhs_bits="e5m2",
      drhs_bits="e5m2",
      use_dummy_static_bound=False,
      fwd_accumulator_dtype=jnp.bfloat16,
      dlhs_accumulator_dtype=jnp.bfloat16,
      drhs_accumulator_dtype=jnp.bfloat16,
      dlhs_use_fwd_quant=False,
      drhs_use_fwd_quant=False,
  )
  constant_bound_config = None

  if len(config.constant_bound_config) == 6:
    fwd_lhs_bound, fwd_rhs_bound, dlhs_lhs_bound, dlhs_rhs_bound, drhs_lhs_bound, drhs_rhs_bound = (
        config.constant_bound_config
    )
    constant_bound_config = ConstantBoundConfig(
        fwd_lhs_bound=fwd_lhs_bound,
        fwd_rhs_bound=fwd_rhs_bound,
        dlhs_lhs_bound=dlhs_lhs_bound,
        dlhs_rhs_bound=dlhs_rhs_bound,
        drhs_lhs_bound=drhs_lhs_bound,
        drhs_rhs_bound=drhs_rhs_bound,
    )
    aqt_dg = _build_const_scale_config(aqt_dg, constant_bound_config)

  aqt_config.set_stochastic_rounding(
      aqt_dg,
      vjp_lhs_stochastic_rounding=False,
      vjp_rhs_stochastic_rounding=False,
      implementation="jax.uniform",
  )

  per_tensor_scales = PerTensorScales(
      fwd_lhs=True,
      fwd_rhs=True,
      dlhs_lhs=True,
      dlhs_rhs=True,
      drhs_lhs=True,
      drhs_rhs=True,
  )
  return _build_per_tensor_config(aqt_dg, per_tensor_scales)


def _get_aqt_fp8_quant_config(config):
  """get aqt for 8-bit floating point quantization configuration"""
  cfg = aqt_config.config_v4(fwd_bits="e4m3", dlhs_bits=None, drhs_bits=None, fwd_accumulator_dtype=jnp.bfloat16)
  return cfg


def _dot_general_make(quant_cfg):
  """Create quantization configs for input matrices to a matmul"""
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
    default_config.update(default)
  return default_config


def _get_mixed_precision_quant_config(mixed_precision_config):
  """Set quantization params based on user configuration."""
  ret_config = {}
  default_mp_config = _get_default_mp_config(default=mixed_precision_config.get(DEFAULT, None))
  for layer_name_re, layer_quantization_config in mixed_precision_config.items():
    # Make a copy of default_mp_config to avoid updating original dict
    quant_config = default_mp_config.copy()
    # print(f"Mixed precision config: processing
    # {layer_name_re} - {layer_quantization_config}, default config - {quant_config}")
    if layer_name_re != DEFAULT:
      for k in quant_config:
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
    with open(config.quant_cfg_path, "rt", encoding="utf8") as config_file:
      mixed_precision_config = json.load(config_file)
    return _get_mixed_precision_quant_config(mixed_precision_config)
  if config.quantization == "fp8":
    return "fp8"
  if config.quantization == "nanoo_fp8":
    return "nanoo_fp8"
  if config.quantization == "aqt_fp8":
    return _get_aqt_fp8_quant_config(config)
  if config.quantization == "aqt_fp8_full":
    return _get_aqt_fp8_default_config(config)
  if config.quantization.startswith("te_"):
    return config.quantization

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
  if config.use_qwix_quantization:
    return None
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
      return Fp8Quantization()
    elif quant_cfg == "nanoo_fp8":
      return NANOOFp8Quantization()
    elif isinstance(quant_cfg, str) and quant_cfg.startswith("te_"):
      return TransformerEngineQuantization(config)
    quant_mode = get_quant_mode(quant_mode_str)
    replicate_scale = config.replicate_quant_scale if config.replicate_quant_scale else False
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode, replicate_scale=replicate_scale)
  return None


def match_aqt_and_unquantized_param(aqt_params, params):
  """match aqt and unquantized params"""
  aqt_param_flat, aqt_tree_def = jax.tree_util.tree_flatten_with_path(
      aqt_params, is_leaf=lambda x: isinstance(x, aqt_tensor.QTensor)
  )
  param_tree_flat, _ = jax.tree_util.tree_flatten_with_path(params)
  aqt_paths = []
  # Original path of quantized AQT param path.
  param_paths = []

  for aqt_k, _ in aqt_param_flat:
    index = None
    for index, (k, _) in enumerate(param_tree_flat):
      path_depth = len(k)
      # every quantized parameter has AQT.. as the leaf node
      # AqtDotGeneral and AqtEinsum replace leaf node.
      # Therefore, leaf node should be ignored for path matching
      # Note: Aqt only operates on kernels so don't pop bias parameters.
      # Ref: https://github.com/AI-Hypercomputer/maxtext/compare/main...quantize_r1
      if k[: path_depth - 1] == aqt_k[: path_depth - 1] and k[-1].key != "bias":
        aqt_paths.append(aqt_k)
        param_paths.append(k)
        break
    assert index is not None
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


class NvidaFp8Provider(qwix.QtProvider):
  """Wraps nn.Fp8DirectDotGeneralOp with Qwix's provider interface."""

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.Fp8DirectDotGeneralOp(name=op_id)(*args, **kwargs)

  def einsum(self, *args, **kwargs):
    rule, op_id = self._get_current_rule_and_op_id("einsum")
    if rule is None:
      return jnp.einsum(*args, **kwargs)
    return nn.Fp8Einsum(name=op_id)(*args, **kwargs)


class NANOOFp8Provider(qwix.QtProvider):

  def dot_general(self, *args, **kwargs):
    # Here we only check if the rule is None or not.
    rule, op_id = self._get_current_rule_and_op_id("dot_general")
    if rule is None:
      return jax.lax.dot_general(*args, **kwargs)
    return nn.NANOOFp8DotGeneralOp(name=op_id)(*args, **kwargs)


def get_quantization_rule(config: Config):
  match config.quantization:
    case "int8":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.int8,
          act_qtype=jnp.int8,
          bwd_qtype=jnp.int8,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8_full":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e5m2,
          weight_calibration_method=config.quantization_calibration_method,
          act_calibration_method=config.quantization_calibration_method,
          bwd_calibration_method=config.quantization_calibration_method,
          op_names=("dot_general", "gmm"),
      )
    case "fp8_gpu":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "fp8_nanoo":
      return qwix.QtRule(
          module_path="decoder/.*layers.*",
          weight_qtype=jnp.float8_e4m3fn,
          act_qtype=jnp.float8_e4m3fn,
          bwd_qtype=jnp.float8_e4m3fn,
          bwd_weight_grad_tile_size=1 / config.quantization_local_shard_count,
          op_names=("dot_general",),
      )
    case "":
      return None


def get_qt_provider(config):
  """Get quantization rules based on the config."""
  match config.quantization:
    case "int8":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8_full":
      return qwix.QtProvider([get_quantization_rule(config)])
    case "fp8_gpu":
      return NvidaFp8Provider([get_quantization_rule(config)])
    case "fp8_nanoo":
      return NANOOFp8Provider([get_quantization_rule(config)])
  return None


def maybe_quantize_model(model, config):
  """Quantize the model if quantization is enabled."""
  if config.use_qwix_quantization:
    quantization_provider = get_qt_provider(config)
    if quantization_provider:
      model = qwix.quantize_model(model, quantization_provider)
  return model


class TransformerEngineQuantization(Quantization):
  """Class for TransformerEngine quantization recipes."""

  def __init__(self, config):
    """Initialize TransformerEngine quantization."""

    self.quant_mode = "train"

    if not config.quantization.startswith("te_"):
      raise ValueError(f"Invalid TransformerEngine quantization config: {config.quantization}")

    self._recipe = TransformerEngineQuantization._get_recipe(config.quantization)

  def __hash__(self):
    return hash((self.quant_mode, self._recipe))

  def __eq__(self, other):
    if not isinstance(other, TransformerEngineQuantization):
      return False
    return (self.quant_mode, self._recipe) == (other.quant_mode, other._recipe)

  @staticmethod
  def _get_recipe(recipe_name: str):
    """Get the TransformerEngine recipe based on the name."""
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    RECIPES = {
        "te_fp8_delayedscaling": recipe.DelayedScaling,
        "te_fp8_currentscaling": recipe.Float8CurrentScaling,
        "te_mxfp8": recipe.MXFP8BlockScaling,
        "te_nvfp4": recipe.NVFP4BlockScaling,  # pytype: disable=module-attr
    }
    if recipe_name not in RECIPES:
      raise ValueError(f"Invalid TransformerEngine recipe: {recipe_name}")
    return RECIPES[recipe_name]()

  def get_block_size(self):
    """Get the block size for quantization for recipes that require blocks.

    If there is no block requirement for the current recipe, returns 1.
    """
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    if isinstance(self._recipe, recipe.MXFP8BlockScaling):
      return 32
    if isinstance(self._recipe, recipe.NVFP4BlockScaling):  # pytype: disable=module-attr
      return 128  # TODO(set this to 16 when unfused RHT is supported)
    return 1

  def _wrap(self, f, name=None):
    """Wraps the given function `f` to support TransformerEngine quantization.

    This method does a couple things:


    1. Wraps the given function in a context that specifies MaxText's physical mesh axes to
    TransformerEngine. This ensures our collective operations in TransformerEngine are using
    the correct axes.

    2. Wraps the given function in a Flax linen module. This module does not store any Flax
    parameters but can store Flax variables for quantizers if required by the recipe.

    3. When the wrapper is called, it provides an additional argument to the given function `f`,
    'generate_quantizer_set' as the first argument. 'generate_quantizer_set' is a function that
    can be called to generate a TransformerEngine/JAX quantizer set object used in
    TransformerEngine/JAX APIs. 'generate_quantizer_set' will generate quantizers based on the
    recipe of this TransformerEngineQuantizer object.

    Args:
      f: The function to wrap. The first argument must be 'generate_quantizer_set'.
      name: The name of this wrapped operation. If unspecified, will use `f.__name__`.

    Returns:
      A Flax linen module that wraps the given function.
    """

    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    fp8_recipe = self._recipe

    class TEWrapper(transformer_engine.jax.flax.module.TransformerEngineBase):
      """Wrapper module for TransformerEngine quantization."""

      def generate_quantizer_set(self, postfix: str = ""):
        OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"
        return super().generate_quantizer_set(  # pytype: disable=wrong-keyword-args
            postfix=postfix, variable_collection=OVERWRITE_WITH_GRADIENT, fp8_recipe=fp8_recipe
        )

      @nn.compact
      def __call__(self, *args, **kwargs):
        return f(self.generate_quantizer_set, *args, **kwargs)

    TEWrapper.__name__ = f"TEWrapper_{name if name else f.__name__}"

    return TEWrapper

  def dot_general_cls(self, mesh_axes: Tuple[str, ...] = ()):
    """Placeholder for dot_general implementation in subclasses."""
    import transformer_engine.jax  # pylint: disable=import-outside-toplevel # pytype: disable=import-error

    def te_dot_general(generate_quantizer_set, x, kernel, dims, **kwargs):
      contracting_dims, batch_dims = dims
      assert batch_dims == ((), ()), "Batch dimensions must be empty for TransformerEngine dot."

      quantizer_set = generate_quantizer_set()
      return transformer_engine.jax.dense.dense(
          x,
          kernel,
          contracting_dims=contracting_dims,
          quantizer_set=quantizer_set,
      )

    return self._wrap(te_dot_general, "dot_general")

  def einsum(self, dtype: DType = jnp.float32):
    """Placeholder for einsum implementation in subclasses."""
    # quant.einsum is only required for MoE or for inference with KVCache.
    raise ValueError("Einsum is not yet supported for TransformerEngine quantization.")
