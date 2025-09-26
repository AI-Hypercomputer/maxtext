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

"""Grouped matrix multiplication operations with custom VJPs."""

# pylint: disable=too-many-positional-arguments

from typing import Literal
import jax
import jax.numpy as jnp
from aqt.jax.v2 import aqt_tensor
import qwix
from qwix.pallas import QArray

from MaxText.kernels.megablox import gmm as backend
from MaxText.kernels.megablox.quantization_config import (
  QuantizationManager,
  QuantizationConfig,
)

gmm = jax.custom_vjp(
  backend.gmm,
  nondiff_argnums=(3, 4, 7, 8, 9, 10, 11, 12, 13, 14),
)


def _gmm_fwd(
  lhs: jnp.ndarray,
  rhs: jnp.ndarray | aqt_tensor.QTensor,
  group_sizes: jnp.ndarray,
  preferred_element_type: jnp.dtype = jnp.float32,
  tiling: tuple[int, int, int] = (128, 128, 128),
  group_offset: jnp.ndarray | None = None,
  existing_out: jnp.ndarray | None = None,
  transpose_rhs: bool = False,
  interpret: bool = False,
  # Kwargs for non-qwix quantization, which also serve as the fallback.
  lhs_quantize_dtype: jnp.dtype | None = None,
  rhs_quantize_dtype: jnp.dtype | None = None,
  lhs_calibration_method: str = "absmax",
  rhs_calibration_method: str = "absmax",
  # Qwix-specific quantization rule.
  quantization_rule: qwix.QtRule | None = None,
  use_qwix_quantization: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray,
        jnp.ndarray | aqt_tensor.QTensor | QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
]:
  """Forward function for GMM VJP."""
  # Create the fallback config from the function's default arguments.
  fallback_config = QuantizationConfig(
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      lhs_calibration_method=lhs_calibration_method,
      rhs_calibration_method=rhs_calibration_method,
  )

  # Instantiate the quantization manager. It will resolve the correct config internally.
  manager = QuantizationManager(
      quantization_rule=quantization_rule,
      use_qwix_quantization=use_qwix_quantization,
      fallback=fallback_config,
  )
  out = backend.gmm(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      group_offset,
      existing_out,
      transpose_rhs=transpose_rhs,
      interpret=interpret,
      use_qwix_quantization=use_qwix_quantization,
      **manager.as_kwargs("fwd"),
  )
  # Pass the manager to the backward pass.
  residual = (lhs, rhs, group_sizes, group_offset, rhs.shape[0])
  return out, residual


def _gmm_bwd(
  # Non-diff args passed from fwd
  preferred_element_type: jnp.dtype,
  tiling: tuple[int, int, int],
  transpose_rhs: bool,
  interpret: bool,
  # These args are now handled by the manager but must be kept in the
  # signature to match the VJP rule.
  lhs_quantize_dtype: Literal["int8", "int4"] | None,
  rhs_quantize_dtype: Literal["int8", "int4"] | None,
  lhs_calibration_method: str,
  rhs_calibration_method: str,
  quantization_rule: qwix.QtRule | None,
  use_qwix_quantization: bool,
  # Residuals from fwd pass
  residual: tuple,
  grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  del preferred_element_type  # Not needed in bwd pass.
  lhs, rhs, group_sizes, group_offset, num_actual_groups = residual

  fallback_config = QuantizationConfig(
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      lhs_calibration_method=lhs_calibration_method,
      rhs_calibration_method=rhs_calibration_method,
  )

  # Instantiate the quantization manager. It will resolve the correct config internally.
  manager = QuantizationManager(
      quantization_rule=quantization_rule,
      use_qwix_quantization=use_qwix_quantization,
      fallback=fallback_config,
  )
  
  # Calculate grad_lhs using the config for the 'dlhs' phase.
  grad_lhs = backend.gmm(
      grad,
      rhs,
      group_sizes,
      lhs.dtype,
      tiling,
      group_offset,
      transpose_rhs=not transpose_rhs,
      interpret=interpret,
      use_qwix_quantization=use_qwix_quantization,
      **manager.as_kwargs("dlhs"),
  )

  # Calculate grad_rhs using the config for the 'drhs' phase.
  grad_rhs = backend.tgmm(
      lhs.swapaxes(0, 1),
      grad,
      group_sizes,
      rhs.dtype,
      tiling,
      group_offset,
      num_actual_groups,
      interpret=interpret,
      use_qwix_quantization=use_qwix_quantization,
      **manager.as_kwargs("drhs"),
  )

  grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
  # Gradients for non-tensor inputs are None.
  return (grad_lhs, grad_rhs, None, None, grad)


gmm.defvjp(_gmm_fwd, _gmm_bwd)
 