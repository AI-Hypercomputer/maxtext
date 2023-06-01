# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for Jax AQTp."""

from aqt.common import aqt_config
from aqt.jax import aqt_tensor
import jax.numpy as jnp

# pylint: disable=protected-access
# pytype: disable=attribute-error


def possibly_use_quantized_variable(
    quantizer: aqt_tensor.TensorQuantizer,
    x: jnp.ndarray,
    train: bool) -> jnp.ndarray:
  """Returns quantized variable if not training and TQ.use_quantized_variable, casted to x.dtype.

  Overrides x with quantizer's quantized variable, cast to x.dtype, if
  quantizer.use_quantized variable, and return x otherwise. Note this will
  return the quantized variable even if a FloatConfig is used.

  For motivation behind these semantics, see b/219040448.

  Args:
    quantizer: TensorQuantizer for the input tensor x.
    x: lhs or rhs of conv_general.
    train: Indicates if in training or not.

  Returns:
    The input tensor x or its quantized one.
  """
  if quantizer.config is not None and quantizer.config.use_quantized_variable and not train:
    qx = quantizer.quantized_variable.value
    qx = qx.astype(x.dtype)
    return qx
  return x


def should_int8_quantize(
    lhs_quantizer: aqt_tensor.TensorQuantizer,
    rhs_quantizer: aqt_tensor.TensorQuantizer) -> bool:
  """Determines whether or not to quantize."""

  if lhs_quantizer.config is None or rhs_quantizer.config is None:
    return jnp.bool_(False)

  lhs_configs = lhs_quantizer.config.tensor_configs
  rhs_configs = rhs_quantizer.config.tensor_configs

  should_quantize = False
  for lhs_config in lhs_configs:
    for rhs_config in rhs_configs:
      if (isinstance(lhs_config.quant_config, aqt_config.IntQuantConfig) and
          isinstance(rhs_config.quant_config, aqt_config.IntQuantConfig) and
          lhs_config.quant_config.bits <= 8 and
          rhs_config.quant_config.bits <= 8):
        should_quantize |= (
            aqt_tensor.is_config_active(lhs_config,
                                        lhs_quantizer._last_update.value)
            & aqt_tensor.is_config_active(rhs_config,
                                          rhs_quantizer._last_update.value))

  return should_quantize
