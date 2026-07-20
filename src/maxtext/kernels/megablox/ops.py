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

"""Grouped matrix multiplication operations with custom VJPs."""

# pylint: disable=too-many-positional-arguments

import dataclasses
import functools
from typing import List, Literal, Tuple
import jax
import jax.numpy as jnp
from maxtext.kernels.megablox import backend
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.layers import quantizations
from maxtext.utils import max_logging
import qwix
import qwix.pallas as qpl
import tokamax
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as tokamax_backend


DLHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([1], [2]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[0],
)

DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    input_buffer_count: tuple[int, int, int] = (2, 2, 2),
    combine_scopes: bool = False,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool | None = None,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    use_qwix_quantization: bool = False,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    lhs_vma_axes: tuple = tuple(),
    rhs_vma_axes: tuple = tuple(),
    # TODO(amandaliang): get rid of the qwix_rule in favor of Qwix's interception feature
    qwix_rule: qwix.QtRule | None = None,
    use_manual_quantization: bool = False,  # used in batchsplit
    use_gmm_v2_fwd: bool = False,
    use_gmm_v2_dlhs: bool = False,
    use_gmm_v2_drhs: bool = False,
    partial_sum: jnp.ndarray | None = None,
):
  """Grouped matrix multiplication operation."""
  if interpret is None:
    # Default to native (TPU) lowering. `jax.devices()[0]` is NOT the compile TARGET:
    # during train_compile the local backend is CPU (JAX_PLATFORMS=cpu) while the mesh
    # targets tpu7x, and interpret-mode there breaks check_vma (exposes the kernel's
    # internal dynamic_slice VMA) and balloons HBM temporaries. Callers that genuinely
    # run off-TPU (e.g. equiv_chunk_test) pass interpret based on their target mesh.
    interpret = False
  quantization_rule = None
  if use_qwix_quantization:
    # 1. for non-batchsplit, retrieve rule ("gmm") via qwix interception
    #   get_current_rule has to be called outside of the _gmm_fwd function.
    # 2. for batchsplit, explicitly pass the rule
    quantization_rule = qwix_rule if qwix_rule else qpl.get_current_rule("gmm")
    if not quantization_rule or not isinstance(quantization_rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training. " f"But get quantization_rule={quantization_rule}")
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
  gmm_fwd_bwd = jax.custom_vjp(
      gmm_fwd_bwd,
      nondiff_argnums=(3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
  )
  gmm_fwd_bwd.defvjp(_gmm_fwd, functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype))
  return gmm_fwd_bwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      input_buffer_count,
      combine_scopes,
      group_offset,
      existing_out,
      transpose_rhs,
      interpret,
      quantization_rule,
      use_tokamax_backend,
      weight_gather_axes,
      use_manual_quantization,
      lhs_vma_axes,
      rhs_vma_axes,
      use_gmm_v2_fwd,
      use_gmm_v2_dlhs,
      use_gmm_v2_drhs,
      partial_sum,
  )


def _prepare_fwd_rhs_scale(rhs: qpl.QArray, transpose_rhs: bool = False) -> jnp.ndarray:
  """Formats and broadcasts rhs scale for the V2 GMM forward kernel."""
  # Target shape: (size_group, num_quant_blocks, 1, size_n)
  if transpose_rhs:
    G, N, _ = rhs.qvalue.shape
    scale = rhs.scale
    if scale.ndim == 3:
      scale = scale.swapaxes(1, 2)
  else:
    G, _, N = rhs.qvalue.shape
    scale = rhs.scale

  if scale.ndim == 2:  # Per-Channel quantization
    rhs_scale = jnp.expand_dims(scale, axis=(1, 2))
  elif scale.ndim == 3:  # Block-wise quantization
    rhs_scale = jnp.expand_dims(scale, axis=2)
  else:  # Per-tensor quantization, (1, 1, 1, 1)
    rhs_scale = scale

  num_quant_blocks = rhs_scale.shape[1] if rhs_scale.ndim > 1 else 1
  return jnp.broadcast_to(rhs_scale, (G, num_quant_blocks, 1, N))


def _scale_grad_by_rhs_scale(
    grad: jnp.ndarray,
    rhs: qpl.QArray,
    group_sizes: jnp.ndarray,
    transpose_rhs: bool = False,
) -> jnp.ndarray:
  """Squeezes the rhs scale and multiplies it with the incoming gradient.

  Scaling is applied before the V2 GMM DLHS kernel.
  """
  rhs_scale = rhs.scale

  # 1. Squeeze the scale to 2D [g, n] based on transpose_rhs
  if rhs_scale.ndim == 3:
    squeeze_axis = 2 if transpose_rhs else 1
    if rhs_scale.shape[squeeze_axis] == 1:
      rhs_scale = rhs_scale.squeeze(axis=squeeze_axis)

  # 2. Apply scale (handle shared vs per-expert scales)
  if rhs_scale.shape[0] == 1:
    return grad * rhs_scale.astype(grad.dtype)
  else:
    repeated_scale = jnp.repeat(
        rhs_scale.astype(grad.dtype),
        group_sizes,
        axis=0,
        total_repeat_length=grad.shape[0],
    )
    return grad * repeated_scale


def _prepare_bwd_drhs_scale(drhs_dout: qpl.QArray) -> jnp.ndarray:
  """Formats and broadcasts drhs_dout scale to (1, 1, size_n) for V2 TGMM kernel."""
  scale = drhs_dout.scale
  size_n = drhs_dout.shape[1]
  # per channel: (1, n) -> (1, 1, n)
  # per tensor: (1, 1) -> (1, 1, 1)
  rhs_scale = jnp.expand_dims(scale, axis=1)
  # per-tensor quantization: broadcast (1, 1, 1) to (1, 1, size_n)
  if rhs_scale.shape[2] == 1:
    rhs_scale = jnp.broadcast_to(rhs_scale, (1, 1, size_n))
  return rhs_scale


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    input_buffer_count: tuple[int, int, int] = (2, 2, 2),
    combine_scopes: bool = False,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    quantization_rule: qwix.QtRule | None = None,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    use_manual_quantization: bool = False,
    lhs_vma_axes: tuple = tuple(),
    rhs_vma_axes: tuple = tuple(),
    use_gmm_v2_fwd: bool = False,
    use_gmm_v2_dlhs: bool = False,
    use_gmm_v2_drhs: bool = False,
    partial_sum: jnp.ndarray | None = None,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
    ],
]:
  """Forward function for GMM VJP.

  lhs: [m, k]
  rhs: [g, k, n] if transpose_rhs=False
  rhs: [g, n, k] if transpose_rhs=True
  if channelwise_axes scaling, quantize non-contracting dimension,
  i.e., m for lhs and n for rhs.
  """

  if quantization_rule:
    if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray) and not use_gmm_v2_fwd:
      lhs = qpl.quantize(  # pyrefly: ignore[bad-assignment]
          lhs,
          quantization_rule.act_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
          calibration_method=quantization_rule.act_calibration_method,  # pyrefly: ignore[bad-argument-type]
      )
    if quantization_rule.weight_qtype and not isinstance(rhs, qpl.QArray):
      if not use_manual_quantization:
        rhs = qpl.quantize(  # pyrefly: ignore[bad-assignment]
            rhs,
            quantization_rule.weight_qtype,
            # If only considering the fwd pass, we could also enable channelwise
            # axes for the group axis, i.e., [0, 1 or 2]. However, this makes the
            # bwd pass unable to reuse the scale easily.
            channelwise_axes=([] if quantization_rule.disable_channelwise_axes else ([1] if transpose_rhs else [2])),
            calibration_method=quantization_rule.weight_calibration_method,
        )
      else:
        rhs = quantizations.manual_quantize(  # pyrefly: ignore[bad-assignment]
            rhs,
            quantization_rule.weight_qtype,
            calibration_method=quantization_rule.weight_calibration_method,
        )

  if use_tokamax_backend:
    # QAG is only supported for following conditions
    if quantization_rule and quantization_rule.bwd_qtype:
      if quantization_rule.weight_calibration_method.startswith("fixed") and isinstance(rhs, qpl.QArray):
        if weight_gather_axes:
          for axis_name, axis_idx in weight_gather_axes:
            rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
            # replace the qvalue with the gathered qvalue in the QArray
            rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)

    # manual_axis_type is for gmm with shard_map check_vma=True, needs tokamax > 0.0.12
    out_kwargs = {}
    if use_manual_quantization:
      # used in batchsplit
      out_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))

    if not use_gmm_v2_fwd:  # TOKAMAX FWD GMM 1
      # Handle transpose_rhs manually as ragged_dot assumes (G, K, N)
      if transpose_rhs:
        rhs = rhs.swapaxes(1, 2)
      out = tokamax.ragged_dot(
          lhs=lhs,
          rhs=rhs,
          group_sizes=group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=preferred_element_type,
          # `group_offset` is not yet supported
          group_offset=None,
          implementation="mosaic",
          **out_kwargs,
      )
    else:  # TOKAMAX FWD GMM 2
      # preserve rhs to be returned in the residual
      # if transpose_rhs=False, [g, k, n]
      # if transpose_rhs=True, [g, n, k], explicit transpose to [g, k, n]
      rhs_operand = rhs if not transpose_rhs else rhs.swapaxes(1, 2)
      rhs_scale = None

      if isinstance(rhs, qpl.QArray):
        rhs_operand = rhs_operand.qvalue
        rhs_scale = _prepare_fwd_rhs_scale(rhs, transpose_rhs=transpose_rhs)

      custom_fwd_tiling = gmm_v2.TileSizes(
          tile_m=tiling[0],
          tile_k=tiling[1],
          tile_n=tiling[2],
      )

      out = gmm_v2.gmm_v2(
          lhs=lhs,
          rhs=rhs_operand,
          group_sizes=group_sizes,
          rhs_scale=rhs_scale,
          tile_info=custom_fwd_tiling,
          preferred_element_type=preferred_element_type,
          partial_sum=partial_sum,
          group_offset=group_offset,
      )

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
    for axis in lhs_vma_axes:
      out = jax.lax.pcast(out, axis_name=axis, to="varying")

  return out, (lhs, rhs, group_sizes, group_offset, partial_sum)


def _gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    input_buffer_count: tuple[int, int, int],
    combine_scopes: bool,
    transpose_rhs: bool,
    interpret: bool,
    quantization_rule: qwix.QtRule | None,
    use_tokamax_backend: bool,
    weight_gather_axes: List[Tuple[str, int]] | None,
    use_manual_quantization: bool,
    lhs_vma_axes: tuple,
    rhs_vma_axes: tuple,
    use_gmm_v2_fwd: bool,
    use_gmm_v2_dlhs: bool,
    use_gmm_v2_drhs: bool,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray, jnp.ndarray | None]:
  """Backward function for throughput GMM VJP."""
  del preferred_element_type
  lhs, rhs, group_sizes, group_offset, partial_sum_fwd = residual
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
  dlhs_dout = grad
  drhs_dout = grad

  # Apply rhs.scale to dlhs_dout, dlhs_dout[m, n] @ rhs_tranpose[g, n, k] = dlhs[m, k]
  # Assume channelwise scale on rhs n.
  # Apply rhs.scale to dlhs_dout to avoid dequantizing or requantizing rhs.
  # We cannot apply the scale to dlhs because axis n will disappear there.
  if isinstance(rhs, qpl.QArray):
    # rhs - qvalue: [g, k, n] scale: [1, 1, n], assume transpose_rhs=False
    if not use_gmm_v2_dlhs:
      dlhs_dout *= rhs.scale.astype(grad.dtype).reshape(1, -1)  # [1, n]
      rhs = rhs.qvalue
    else:
      # NOTE: rhs.scale is for the contracting dimension (N) in DLHS, but gmm_v2
      # only supports scaling the output dimension. Thus, we must scale dlhs_dout
      # beforehand. If the kernel supports transposing RHS internally, we can fuse
      # this scale inside the kernel.
      dlhs_dout = _scale_grad_by_rhs_scale(dlhs_dout, rhs, group_sizes, transpose_rhs)
      rhs = rhs.qvalue

  # Apply lhs.scale to drhs_dout, lhs_transpose[k, m] @ drhs_out[m, n] = drhs[g, k, n]
  # Assume channelwise scale on lhs m
  # Apply lhs.scale to drhs_dout, as axis m will disappear in drhs.
  if isinstance(lhs, qpl.QArray):
    # lhs - qvalue: [m, k] scale: [m, 1]
    drhs_dout *= lhs.scale.astype(grad.dtype)
    lhs = lhs.qvalue

  if quantization_rule and quantization_rule.bwd_qtype:
    # Enable backward pass quantization
    dlhs_dout = qpl.quantize(
        dlhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        calibration_method=quantization_rule.bwd_calibration_method,
    )
    drhs_dout = qpl.quantize(
        drhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [1],
        calibration_method=quantization_rule.bwd_calibration_method,
    )

  # TOKAMAX: DLHS (GMM) and DRHS (TGMM)
  if use_tokamax_backend:
    # manual_axis_type is for gmm with shard_map check_vma=True, needs tokamax > 0.0.12
    dlhs_kwargs = {}
    drhs_kwargs = {}
    if use_manual_quantization:
      # used in batchsplit
      dlhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))
      drhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(
          varying=frozenset(["expert"]), unreduced=frozenset(["data", "fsdp"])
      )

    # TOKAMAX DLHS
    if not use_gmm_v2_dlhs and not use_gmm_v2_fwd:  # TOKAMAX DLHS GMM 1
      # Handle transpose_rhs manually
      dlhs_rhs = rhs
      if transpose_rhs:
        dlhs_rhs = dlhs_rhs.swapaxes(1, 2)
      dlhs = tokamax.ragged_dot_general(
          lhs=dlhs_dout,
          rhs=dlhs_rhs,
          group_sizes=group_sizes,
          ragged_dot_dimension_numbers=DLHS_RAGGED_DOT_DIM_NUMS,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=lhs_dtype,
          # `group_offset` is not yet supported
          group_offset=None,
          implementation="mosaic",
          **dlhs_kwargs,
      )
    elif not use_gmm_v2_dlhs and use_gmm_v2_fwd:
      # TOKAMAX DLHS GMM 1 (with tile passing inside scheme 2+1+2)
      dlhs = tokamax_backend.gmm(
          lhs=dlhs_dout,
          rhs=rhs,
          group_sizes=group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          out_dtype=lhs_dtype,
          tiling=tiling[3:6],
          group_offset=group_offset,
          transpose_rhs=not transpose_rhs,
          interpret=interpret,
          input_buffer_count=input_buffer_count[1],
          **dlhs_kwargs,
      )
    else:  # TOKAMAX DLHS GMM 2
      # NOTE: We manually transpose RHS here because gmm_v2 lacks native transpose_rhs
      # support. Fusing this transpose into the kernel would also allow us to fuse
      # the rhs_scale application.
      dlhs_rhs = rhs if transpose_rhs else rhs.swapaxes(1, 2)
      dlhs_lhs = dlhs_dout.qvalue if isinstance(dlhs_dout, qpl.QArray) else dlhs_dout
      custom_dlhs_tiling = gmm_v2.TileSizes(
          tile_m=tiling[3],
          tile_k=tiling[4],
          tile_n=tiling[5],
      )
      dlhs = gmm_v2.gmm_v2(
          lhs=dlhs_lhs,
          rhs=dlhs_rhs,
          group_sizes=group_sizes,
          # rhs scale is already in dlhs_lhs
          rhs_scale=None,
          tile_info=custom_dlhs_tiling,
          preferred_element_type=lhs_dtype,
          group_offset=group_offset,
      )

      if isinstance(dlhs_dout, qpl.QArray):
        dlhs *= dlhs_dout.scale.astype(dlhs.dtype)

    # TOKAMAX DRHS
    if not use_gmm_v2_drhs:  # TOKAMAX DRHS TGMM 1
      drhs = tokamax.ragged_dot_general(
          lhs=lhs,
          rhs=drhs_dout,
          group_sizes=group_sizes,
          ragged_dot_dimension_numbers=DRHS_RAGGED_DOT_DIM_NUMS,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=rhs_dtype,
          # `group_offset` is not yet supported
          group_offset=None,
          implementation="mosaic",
          **drhs_kwargs,
      )
    else:  # TOKAMAX DRHS TGMM 2
      # Extract arrays
      drhs_rhs = drhs_dout.qvalue if isinstance(drhs_dout, qpl.QArray) else drhs_dout
      drhs_lhs = lhs
      # TGMM kernel requires matching sublane sizes (dtypes) for hardware packing.
      # If mismatch, cast lhs to match rhs_q.
      if drhs_lhs.dtype != drhs_rhs.dtype:
        drhs_lhs = drhs_lhs.astype(drhs_rhs.dtype)

      rhs_scale = None
      if isinstance(drhs_dout, qpl.QArray):
        # drhs_dout.scale expand to (1, 1, N) for tgmm_v2
        rhs_scale = _prepare_bwd_drhs_scale(drhs_dout)

      custom_drhs_tiling = gmm_v2.TileSizes(
          tile_m=tiling[6],
          tile_k=tiling[7],
          tile_n=tiling[8],
      )

      drhs = tgmm_v2.tgmm_v2(
          lhs=drhs_lhs,
          rhs=drhs_rhs,
          group_sizes=group_sizes,
          num_actual_groups=num_actual_groups,
          rhs_scale=rhs_scale,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=rhs_dtype,
          group_offset=group_offset,
          tile_info=custom_drhs_tiling,
      )

    if quantization_rule and quantization_rule.bwd_qtype and weight_gather_axes:
      # Scatter back in reverse order of gather
      for axis_name, axis_idx in reversed(weight_gather_axes):
        drhs = jax.lax.psum_scatter(drhs, axis_name, scatter_dimension=axis_idx, tiled=True)

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
        varying_axes=lhs_vma_axes,
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
        varying_axes=rhs_vma_axes,
    )

  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  drhs = drhs.swapaxes(1, 2) if transpose_rhs else drhs
  dpartial_sum = grad if partial_sum_fwd is not None else None
  # tokamax gmm path ignore existing_out
  d_existing_out = None if use_tokamax_backend else grad
  return dlhs, drhs, None, None, d_existing_out, dpartial_sum
