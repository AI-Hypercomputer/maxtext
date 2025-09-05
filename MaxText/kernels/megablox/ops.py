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
import dataclasses
import jax
import jax.numpy as jnp

from aqt.jax.v2 import aqt_tensor

import qwix
import qwix.pallas as qpl

from MaxText.kernels.megablox import gmm as backend

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
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    lhs_calibration_method: str = "absmax",
    rhs_calibration_method: str = "absmax",
    quantization_rule: qwix.QtRule | None = None,
    use_qwix_quantization: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray,
        jnp.ndarray | aqt_tensor.QTensor,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
]:
  """Forward function for GMM VJP."""
  if use_qwix_quantization:
    lhs_quantize_dtype, rhs_quantize_dtype = None, None
    if quantization_rule is not None:
      lhs_quantize_dtype = quantization_rule.act_qtype
      rhs_quantize_dtype = quantization_rule.weight_qtype
      lhs_calibration_method = quantization_rule.act_calibration_method
      rhs_calibration_method = quantization_rule.weight_calibration_method
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
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      lhs_calibration_method=lhs_calibration_method,
      rhs_calibration_method=rhs_calibration_method,
      use_qwix_quantization=use_qwix_quantization,
  )
  return out, (lhs, rhs, group_sizes, group_offset, rhs.shape[0])


def _gmm_bwd(
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None,
    lhs_calibration_method: str,
    rhs_calibration_method: str,
    quantization_rule: qwix.QtRule | None,
    use_qwix_quantization: bool,
    residual: tuple[
        jnp.ndarray,
        jnp.ndarray | aqt_tensor.QTensor,
        jnp.ndarray,
        jnp.ndarray | None,
        int,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  if use_qwix_quantization:
    lhs_quantize_dtype, rhs_quantize_dtype = None, None
    if quantization_rule is not None:
      if quantization_rule.additional_qt_config is not None:
        lhs_quantize_dtype = quantization_rule.additional_qt_config["dlhs_lhs_qtype"]
        rhs_quantize_dtype = quantization_rule.additional_qt_config["dlhs_rhs_qtype"]
      else:
        lhs_quantize_dtype = quantization_rule.act_qtype
        rhs_quantize_dtype = quantization_rule.bwd_qtype
      lhs_calibration_method = quantization_rule.additional_qt_config["dlhs_lhs_calibration_method"]
      rhs_calibration_method = quantization_rule.additional_qt_config["dlhs_rhs_calibration_method"]
  del preferred_element_type
  lhs, rhs, group_sizes, group_offset, num_actual_groups = residual
  grad_lhs = backend.gmm(
      grad,
      rhs,
      group_sizes,
      lhs[0].dtype,
      tiling,
      group_offset,
      transpose_rhs=not transpose_rhs,
      interpret=interpret,
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      lhs_calibration_method=lhs_calibration_method,
      rhs_calibration_method=rhs_calibration_method,
      use_qwix_quantization=use_qwix_quantization,  
  )
  if use_qwix_quantization:
    lhs_quantize_dtype, rhs_quantize_dtype = None, None
    if quantization_rule is not None:
      if quantization_rule.additional_qt_config is not None:
        lhs_quantize_dtype = quantization_rule.additional_qt_config["drhs_lhs_qtype"]
        rhs_quantize_dtype = quantization_rule.additional_qt_config["drhs_rhs_qtype"]
      else:
        lhs_quantize_dtype = quantization_rule.bwd_qtype
        rhs_quantize_dtype = quantization_rule.act_qtype
      lhs_calibration_method = quantization_rule.additional_qt_config["drhs_lhs_calibration_method"]
      rhs_calibration_method = quantization_rule.additional_qt_config["drhs_rhs_calibration_method"]
  grad_rhs = backend.tgmm(
      lhs.swapaxes(0, 1),
      grad,
      group_sizes,
      rhs.dtype,
      tiling,
      group_offset,
      num_actual_groups,
      interpret=interpret,
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      lhs_calibration_method=lhs_calibration_method,
      rhs_calibration_method=rhs_calibration_method,
      use_qwix_quantization=use_qwix_quantization,
  )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs
  return grad_lhs, grad_rhs, None, None, grad


gmm.defvjp(_gmm_fwd, _gmm_bwd)
