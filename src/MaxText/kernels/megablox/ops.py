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

import functools
import dataclasses
from typing import Literal
import jax
import jax.numpy as jnp
from MaxText.kernels.megablox import backend
from tokamax._src.ops.ragged_dot import api as tokamax_api  # pylint: disable=unused-import
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend  # pylint: disable=unused-import
import qwix
import qwix.pallas as qpl


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    use_qwix_quantization: bool = False,
    use_tokamax_backend: bool = False,
):
  """Grouped matrix multiplication operation."""
  quantization_rule = None
  if use_qwix_quantization:
    # get_current_rule has to be called outside of the _gmm_fwd function.
    quantization_rule = qpl.get_current_rule("gmm")
    if quantization_rule and not isinstance(quantization_rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training.")
  else:
    # Handcraft a rule that matches the AQT's behavior.
    if lhs_quantize_dtype or rhs_quantize_dtype:
      quantization_rule = qwix.QtRule(
          weight_qtype=rhs_quantize_dtype,
          weight_calibration_method="absmax",
          act_qtype=lhs_quantize_dtype,
          act_calibration_method="absmax",
      )

  gmm_fwd_bwd = lambda *args: _gmm_fwd(*args)[0]  # pylint: disable=C3001
  gmm_fwd_bwd = jax.custom_vjp(gmm_fwd_bwd, nondiff_argnums=(3, 4, 7, 8, 9, 10))
  gmm_fwd_bwd.defvjp(_gmm_fwd, functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype))
  return gmm_fwd_bwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      group_offset,
      existing_out,
      transpose_rhs,
      interpret,
      quantization_rule,
      use_tokamax_backend,
  )


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    quantization_rule: qwix.QtRule | None = None,
    use_tokamax_backend: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
    ],
]:
  """Forward function for GMM VJP."""
  if quantization_rule:
    if quantization_rule.act_qtype:
      lhs = qpl.quantize(
          lhs,
          quantization_rule.act_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
          calibration_method=quantization_rule.act_calibration_method,
          scale_dtype=jnp.float32,
      )
    if quantization_rule.weight_qtype:
      rhs = qpl.quantize(
          rhs,
          quantization_rule.weight_qtype,
          # If only considering the fwd pass, we could also enable channelwise
          # axes for the group axis, i.e., [0, 1 or 2]. However, this makes the
          # bwd pass unable to reuse the scale easily.
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else ([1] if transpose_rhs else [2]),
          calibration_method=quantization_rule.weight_calibration_method,
          scale_dtype=jnp.float32,
      )
      # QAG is only supported for following conditions
  if use_tokamax_backend:
    print(f"before all gather {jax.typeof(rhs.qvalue)=}")
    if quantization_rule and quantization_rule.bwd_qtype:
      if quantization_rule.weight_calibration_method.startswith("fixed") and isinstance(rhs, qpl.QArray):
        rhs_qvalue = jax.lax.all_gather(rhs.qvalue, "fsdp", axis=0, tiled=True)
        rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
        # rhs = jax.lax.all_gather(rhs, "fsdp", axis=0, tiled=True)
        # print(f"after all gather {jax.typeof(rhs.qvalue)=}")
    # with set_xla_metadata(MUST_FUSE=fwd_counter):
    # print(f"{len((lhs.shape[0] // rhs.shape[0],) * rhs.shape[0])=}")
    print(f"group_sizes {jax.typeof(group_sizes)=}")
    print(f"lhs {jax.typeof(lhs.qvalue)=}")
    print(f"rhs {jax.typeof(rhs.qvalue)=}")
    out = tokamax_api.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(([1], [1]), ([], [])),
            lhs_ragged_dimensions=[0],
            rhs_group_dimensions=[0],
        ),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        implementation="mosaic",
    )
    print(f"forward pass output {jax.typeof(out)=}")
    # out = tokamax_backend.gmm(
    #     lhs=lhs,
    #     rhs=rhs,
    #     group_sizes=group_sizes,
    #     precision=jax.lax.Precision.DEFAULT,
    #     out_dtype=preferred_element_type,
    #     tiling=tiling[:3],
    #     group_offset=group_offset,
    #     transpose_rhs=transpose_rhs,
    #     interpret=interpret,
    # )
  else:
    out = backend.gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling[:3],
        group_offset,
        existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
  return out, (lhs, rhs, group_sizes, group_offset)


def _gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    quantization_rule: qwix.QtRule | None,
    use_tokamax_backend: bool,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  lhs, rhs, group_sizes, group_offset = residual
  num_actual_groups = rhs.shape[0]

  # Jargon used here:
  #  - lhs: input activation in forward pass, possibly quantized.
  #  - rhs: weight in forward pass, possibly quantized.
  #  - dout (or grad): the incoming gradient in the backward pass.
  #  - dlhs: gradient of the lhs in the backward pass, what we want to compute.
  #  - drhs: gradient of the rhs in the backward pass, what we want to compute.
  #  - dlhs_dout: the incoming gradient used to calculate dlhs.
  #  - drhs_dout: the incoming gradient used to calculate drhs.

  # dlhs_dout and drhs_dout can be different when quantization is enabled.
  # dlhs_counter = next(_counter)
  # drhs_counter = next(_counter)
  dlhs_dout = grad
  drhs_dout = grad
  if isinstance(rhs, qpl.QArray):  # qvalue: [g, k, n] scale: [1, 1, n]
    # Apply rhs.scale to dlhs_dout to avoid dequantizing or requantizing rhs.
    # We cannot apply the scale to dlhs because axis n will disappear there.
    dlhs_dout *= rhs.scale.astype(grad.dtype).reshape(1, -1)  # [1, n]
    rhs = rhs.qvalue
  if isinstance(lhs, qpl.QArray):  # qvalue: [m, k] scale: [m, 1]
    # Apply lhs.scale to drhs_dout, as axis m will disappear in drhs.
    drhs_dout *= lhs.scale.astype(grad.dtype)
    lhs = lhs.qvalue
  if quantization_rule and quantization_rule.bwd_qtype:
    # Enable backward pass quantization
    dlhs_dout = qpl.quantize(
        dlhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        calibration_method=quantization_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )
    drhs_dout = qpl.quantize(
        drhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [1],
        calibration_method=quantization_rule.bwd_calibration_method,
        scale_dtype=jnp.float32,
    )
  if use_tokamax_backend:
    # with set_xla_metadata(MUST_FUSE=dlhs_counter):
    print(f"backward: dlhs_dout {jax.typeof(dlhs_dout.qvalue)=}")
    print(f"backward: rhs {jax.typeof(rhs.qvalue)=}")
    dlhs = tokamax_api.ragged_dot_general(
        lhs=dlhs_dout,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(([1], [2]), ([], [])),
            lhs_ragged_dimensions=[0],
            rhs_group_dimensions=[0],
        ),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        implementation="mosaic",
    )
    print(f"backward: dlhs {jax.typeof(dlhs)=}")
    print(f"backward: lhs {jax.typeof(lhs.swapaxes(0, 1).qvalue)=}")
    print(f"backward: drhs_dout {jax.typeof(drhs_dout.qvalue)=}")
    drhs = tokamax_api.ragged_dot_general(
        lhs.swapaxes(0, 1),
        drhs_dout,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(([0], [0]), ([], [])),
            lhs_ragged_dimensions=[0],
            rhs_group_dimensions=[],
        ),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
        implementation="mosaic",
    )
    print(f"backward: drhs {jax.typeof(drhs)=}")
    # dlhs = tokamax_backend.gmm(
    #     lhs=dlhs_dout,
    #     rhs=rhs,
    #     group_sizes=group_sizes,
    #     precision=jax.lax.Precision.DEFAULT,
    #     out_dtype=lhs_dtype,
    #     tiling=tiling[3:6],
    #     group_offset=group_offset,
    #     transpose_rhs=not transpose_rhs,
    #     interpret=interpret,
    # )
    # drhs = tokamax_backend.tgmm(
    #     lhs=lhs.swapaxes(0, 1),
    #     rhs=drhs_dout,
    #     group_sizes=group_sizes,
    #     precision=jax.lax.Precision.DEFAULT,
    #     out_dtype=rhs_dtype,
    #     tiling=tiling[-3:],
    #     group_offset=group_offset,
    #     num_actual_groups=num_actual_groups,
    #     interpret=interpret,
    # )
    if quantization_rule and quantization_rule.bwd_qtype:
      drhs = jax.lax.psum_scatter(drhs, "fsdp", scatter_dimension=0, tiled=True)
  else:
    dlhs = backend.gmm(
        dlhs_dout,
        rhs,
        group_sizes,
        lhs_dtype,
        tiling[3:6],
        group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    drhs = backend.tgmm(
        lhs.swapaxes(0, 1),
        drhs_dout,
        group_sizes,
        rhs_dtype,
        tiling[-3:],
        group_offset,
        num_actual_groups,
        interpret=interpret,
    )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  drhs = drhs.swapaxes(1, 2) if transpose_rhs else drhs
  return dlhs, drhs, None, None, grad
