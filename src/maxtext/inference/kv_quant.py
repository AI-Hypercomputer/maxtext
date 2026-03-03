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

"""Common quantization types and configuration logic for KV cache."""

from typing import Any

import jax
import jax.numpy as jnp

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.aqt_tensor import QTensor as KVTensor
from aqt.jax.v2.flax import aqt_flax

from maxtext.common.common_types import Array, AxisNames, Config, CACHE_HEADS, CACHE_KV

MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)


class KVQuant:
  """Class to configure quantization for KV cache."""

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
    if dtype_cfg == "fp8":
      return jnp.float8_e4m3fn
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
    if self.dtype == jnp.float8_e4m3fn:
      value = jnp.float8_e4m3fn(kv * (E4M3_MAX / scale))
      return value, scale
    raise ValueError(f"Invalid KV quant dtype:{self.dtype}.")

  def einsum_fn_with_rhs_qtensor(
      self,
      rhs_dequant_mode=None,
      rhs_calibration_mode=None,
      lhs_dequant_mode=None,
      lhs_calibration_mode=None,
  ):
    """einsum function where QTensor is the right-hand-side"""
    # Assumes kv is already quantized.
    einsum = jnp.einsum
    if self.dtype != jnp.float8_e4m3fn:
      num_bits = 4 if self.dtype == jnp.int4 else 8
      kv_cfg = aqt_config.dot_general_make(
          lhs_bits=None,
          rhs_bits=num_bits,
          bwd_bits=None,
          use_fwd_quant=False,
      )
    else:
      kv_cfg = aqt_config.config_fwd_fp8()

    if rhs_dequant_mode:
      aqt_config.set_fwd_dequant_mode(kv_cfg, rhs_dequant_mode=rhs_dequant_mode)
    if rhs_calibration_mode:
      aqt_config.set_fwd_calibration_mode(
          kv_cfg,
          rhs_calibration_mode=rhs_calibration_mode,
      )
    if lhs_dequant_mode:
      aqt_config.set_fwd_dequant_mode(kv_cfg, lhs_dequant_mode=lhs_dequant_mode)
    if lhs_calibration_mode:
      aqt_config.set_fwd_calibration_mode(
          kv_cfg,
          lhs_calibration_mode=lhs_calibration_mode,
      )
    einsum = aqt_flax.AqtEinsum(
        rhs_quant_mode=aqt_flax.QuantMode.TRAIN,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        cfg=kv_cfg,
    )
    return einsum

  def einsum_fn_with_rhs_qtensor_and_dequant(self):
    """Get einstein summation for different dequant modes."""
    if self.dtype == jnp.float8_e4m3fn:
      return self.einsum_fn_with_rhs_qtensor(
          lhs_dequant_mode=aqt_config.DequantMode.THIS_INPUT,
          lhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )
    else:
      return self.einsum_fn_with_rhs_qtensor(
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )
