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
    use_gmm_v2: bool = False,
    use_gmm_v2_heuristic_tiling: bool = False,
    partial_sum: jnp.ndarray | None = None,
    bwd_inkernel_quant: bool = False,
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
      raise ValueError(f"Expect a QtRule for quantized training. But get quantization_rule={quantization_rule} for gmm.")
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
      nondiff_argnums=(3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18),
  )
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
      weight_gather_axes,
      use_manual_quantization,
      lhs_vma_axes,
      rhs_vma_axes,
      use_gmm_v2,
      use_gmm_v2_heuristic_tiling,
      partial_sum,
      bwd_inkernel_quant,
  )


# ==============================================================================
# Forward: FWD GMM
# ==============================================================================


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
    use_gmm_v2: bool = False,
    use_gmm_v2_heuristic_tiling: bool = False,
    partial_sum: jnp.ndarray | None = None,
    bwd_inkernel_quant: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
    ],
]:
  """Forward function for GMM VJP.

  - lhs: [m, k]
  - rhs: [g, k, n] if transpose_rhs=False. [g, n, k] if transpose_rhs=True
  """

  lhs_is_qarray = isinstance(lhs, qpl.QArray)
  rhs_is_qarray = isinstance(rhs, qpl.QArray)

  # Quantize activation and weight
  if quantization_rule:
    # pyrefly: ignore[bad-assignment]
    lhs, rhs = _fwd_quantize_activation_and_weight(
        lhs, rhs, quantization_rule, use_gmm_v2, use_manual_quantization, transpose_rhs
    )

  # Quantization All-Gather (QAG) for weight: only supported for following conditions
  if (
      use_tokamax_backend
      and quantization_rule
      and quantization_rule.bwd_qtype
      and quantization_rule.weight_calibration_method.startswith("fixed")
      and isinstance(rhs, qpl.QArray)
      and weight_gather_axes
  ):
    # pyrefly: ignore[bad-assignment]
    rhs = _fwd_gather_weight(rhs, weight_gather_axes)

  # Backend Execution Routing
  if use_tokamax_backend and not use_gmm_v2:
    out = _fwd_run_tokamax_v1(lhs, rhs, group_sizes, preferred_element_type, transpose_rhs, use_manual_quantization)
  elif use_tokamax_backend and use_gmm_v2:
    out = _fwd_run_tokamax_v2(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        use_gmm_v2_heuristic_tiling,
        group_offset,
        partial_sum,
        transpose_rhs,
        quantization_rule,
    )
  else:
    out = _fwd_run_megablox(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        transpose_rhs,
        interpret,
        lhs_vma_axes,
    )

  return out, (lhs, rhs, group_sizes, group_offset, partial_sum, lhs_is_qarray, rhs_is_qarray)  # pyrefly: ignore[bad-return]


def _fwd_quantize_activation_and_weight(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule,
    use_gmm_v2: bool,
    use_manual_quantization: bool,
    transpose_rhs: bool,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray]:
  """Handles act and weight quantization for GMM forward inputs."""
  if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray) and not use_gmm_v2:
    lhs = qpl.quantize(  # pyrefly: ignore[bad-assignment]
        lhs,
        quantization_rule.act_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        # pyrefly: ignore[bad-argument-type]
        calibration_method=quantization_rule.act_calibration_method,
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
  return lhs, rhs


def _fwd_gather_weight(rhs: qpl.QArray, weight_gather_axes: List[Tuple[str, int]]) -> qpl.QArray:
  """Applies QAG (Quantization All-Gather) to RHS weights during forward pass."""
  for axis_name, axis_idx in weight_gather_axes:
    rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
    # replace the qvalue with the gathered qvalue in the QArray
    rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
  return rhs


def _fwd_run_tokamax_v1(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    transpose_rhs: bool,
    use_manual_quantization: bool,
) -> jnp.ndarray:
  """Executes the standard Tokamax GMM V1 for forward pass."""
  # manual_axis_type is for gmm with shard_map check_vma=True, needs tokamax > 0.0.12
  out_kwargs = {}
  if use_manual_quantization:
    # used in batchsplit
    out_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))

  if transpose_rhs:
    rhs = rhs.swapaxes(1, 2)

  return tokamax.ragged_dot(
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


def _fwd_prepare_rhs_scale(rhs: qpl.QArray, transpose_rhs: bool = False) -> jnp.ndarray:
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


def _fwd_prepare_lhs_scale(quantization_rule: qwix.QtRule | None) -> jax.Array | None:
  """Extracts the static LHS (activation) scale for the GMM v2 forward pass.

  If a static scale is used, GMM v2 requires it to be from a symmetric fixed-range
  calibration (e.g., 'fixed,-max,max' or 'fixed,max'). If no static scale is
  provided, the kernel will compute a dynamic scale on the fly.

  Enforces a default (1, 1) shape for per-tensor quantization kernels.

  Args:
    quantization_rule: The Qwix quantization rule from which to extract the scale.

  Returns:
    The extracted static scale array, or None if not using purely fixed calibration.
  """
  if quantization_rule is None:
    return None

  method = quantization_rule.act_calibration_method
  qtype = quantization_rule.act_qtype

  # Use dynamic quantization, gmm_v2 calculates dynamic scale internally
  if method is None or qtype is None or not method.lower().startswith("fixed"):
    return None

  scale_val = quantizations.get_static_scale(qtype, method)

  return jnp.full((1, 1), scale_val, jnp.float32)


def _fwd_run_tokamax_v2(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple,
    use_gmm_v2_heuristic_tiling: bool,
    group_offset: jnp.ndarray | None,
    partial_sum: jnp.ndarray | None,
    transpose_rhs: bool,
    quantization_rule: qwix.QtRule | None = None,
) -> jnp.ndarray:
  """Executes the Tokamax GMM V2 backend for forward pass OUT = LHS @ RHS."""
  # if transpose_rhs=False, rhs is [g, k, n], remain unchanged
  # if transpose_rhs=True, rhs [g, n, k], explicit transpose to [g, k, n]
  rhs_operand = rhs if not transpose_rhs else rhs.swapaxes(1, 2)
  rhs_scale = None

  if isinstance(rhs, qpl.QArray):
    # pyrefly: ignore[missing-attribute]
    rhs_operand = rhs_operand.qvalue
    rhs_scale = _fwd_prepare_rhs_scale(rhs, transpose_rhs=transpose_rhs)

  lhs_operand = lhs.qvalue if isinstance(lhs, qpl.QArray) else lhs

  if use_gmm_v2_heuristic_tiling:
    fwd_tiling = gmm_v2.calculate_tiling
  else:
    fwd_tiling = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2])

  out = gmm_v2.gmm_v2(
      lhs=lhs_operand,  # pyrefly: ignore[bad-argument-type]
      rhs=rhs_operand,  # pyrefly: ignore[bad-argument-type]
      group_sizes=group_sizes,
      rhs_scale=rhs_scale,
      tile_info=fwd_tiling,
      preferred_element_type=preferred_element_type,
      partial_sum=partial_sum,
      group_offset=group_offset,
      lhs_scale=_fwd_prepare_lhs_scale(quantization_rule) if not isinstance(lhs, qpl.QArray) else None,
      maybe_quantize_lhs=not isinstance(lhs, qpl.QArray),
  )

  if isinstance(lhs, qpl.QArray):
    out = out * (lhs.scale.squeeze() if lhs.scale.size == 1 else lhs.scale).astype(out.dtype)

  return out


def _fwd_run_megablox(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple,
    group_offset: jnp.ndarray | None,
    existing_out: jnp.ndarray | None,
    transpose_rhs: bool,
    interpret: bool,
    lhs_vma_axes: tuple,
) -> jnp.ndarray:
  """Executes the Megablox backend fallback for forward pass."""
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
  return out


# ==============================================================================
# Backward: BWD DLHS GMM + BWD DRHS TGMM
# ==============================================================================


def _gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    quantization_rule: qwix.QtRule | None,
    use_tokamax_backend: bool,
    weight_gather_axes: List[Tuple[str, int]] | None,
    use_manual_quantization: bool,
    lhs_vma_axes: tuple,
    rhs_vma_axes: tuple,
    use_gmm_v2: bool,
    use_gmm_v2_heuristic_tiling: bool,
    bwd_inkernel_quant: bool,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        bool,
        bool,
    ],
    grad: jnp.ndarray,
) -> tuple[
    jnp.ndarray | qpl.QArray,
    jnp.ndarray | qpl.QArray,
    None,
    None,
    jnp.ndarray | None,
    jnp.ndarray | None,
]:
  """Backward function for throughput GMM VJP."""
  residual_lhs, residual_rhs, group_sizes, group_offset, partial_sum_fwd, lhs_is_qarray, rhs_is_qarray = residual
  num_actual_groups = residual_rhs.shape[0]

  # Jargon used here:
  #  - lhs: input activation in forward pass, possibly quantized.
  #  - rhs: weight in forward pass, possibly quantized.
  #  - dout (or grad): the incoming gradient in the backward pass.
  #  - dlhs: gradient of the lhs in the backward pass, what we want to compute.
  #  - drhs: gradient of the rhs in the backward pass, what we want to compute.
  #  - dlhs_dout: the incoming gradient used to calculate dlhs.
  #  - drhs_dout: the incoming gradient used to calculate drhs.

  # moe_bwd_inkernel_quant: run the drhs tgmm with BOTH operands quantized in-kernel
  # (per-gm-tile-per-channel), touching only rows covered by group_sizes. This subsumes three
  # dense buffer-sized XLA ops -- the per-row lhs re-quantize, the drhs_dout *= lhs.scale
  # multiply, and the per-N cotangent quantize -- and, because no reduction ever reads past the
  # valid rows, removes the NaN hazard of amax over uninitialized ragged-buffer tail rows.
  inkernel_drhs = (
      bwd_inkernel_quant
      and use_tokamax_backend
      and use_gmm_v2
      and quantization_rule is not None
      and bool(quantization_rule.bwd_qtype)
  )

  # 1. Scale Application & QArray Unwrapping
  dlhs_dout, drhs_dout, lhs, rhs = _bwd_prepare_inputs(
      grad, residual_lhs, residual_rhs, group_sizes, use_gmm_v2, transpose_rhs, quantization_rule,
      skip_lhs_quant=inkernel_drhs,
  )

  # 2. Backward Pass Quantization
  if quantization_rule:
    if inkernel_drhs:
      # the in-kernel tgmm quantizes drhs_dout ITSELF; quantize ONLY the dlhs cotangent here
      # (calling the two-sided helper would emit the dense drhs quantize just to discard it).
      if quantization_rule.bwd_qtype:
        dlhs_dout = qpl.quantize(
            # pyrefly: ignore[bad-argument-type]
            dlhs_dout,
            quantization_rule.bwd_qtype,
            channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
            calibration_method=quantization_rule.bwd_calibration_method,
        )
      if not isinstance(drhs_dout, qpl.QArray) and drhs_dout.dtype != lhs.dtype:
        # tgmm requires equal operand widths; the in-kernel path reads the RAW cotangent, so
        # carry it at the activation width (halves the kernel's cotangent read bytes vs f32).
        drhs_dout = drhs_dout.astype(lhs.dtype)
    else:
      dlhs_dout, drhs_dout = _bwd_quantize_gradient(dlhs_dout, drhs_dout, quantization_rule)

  # 3. DLHS Gradient Execution
  dlhs = _compute_dlhs(
      dlhs_dout,
      rhs,
      group_sizes,
      group_offset,
      lhs_dtype,
      tiling,
      transpose_rhs,
      use_tokamax_backend,
      use_gmm_v2,
      use_manual_quantization,
      interpret,
      lhs_vma_axes,
      use_gmm_v2_heuristic_tiling,
  )

  # 4. DRHS Gradient Execution
  drhs = _compute_drhs(
      drhs_dout,
      lhs,
      group_sizes,
      group_offset,
      num_actual_groups,
      rhs_dtype,
      tiling,
      use_tokamax_backend,
      use_gmm_v2,
      use_manual_quantization,
      weight_gather_axes,
      interpret,
      rhs_vma_axes,
      quantization_rule,
      use_gmm_v2_heuristic_tiling,
      inkernel_quant=inkernel_drhs,
  )

  # 5. Output Formatting
  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  drhs = drhs.swapaxes(1, 2) if transpose_rhs else drhs
  dpartial_sum = grad if partial_sum_fwd is not None else None
  d_existing_out = None if use_tokamax_backend else grad

  if lhs_is_qarray and isinstance(residual_lhs, qpl.QArray):
    lhs_scale = (residual_lhs.scale.squeeze() if residual_lhs.scale.size == 1 else residual_lhs.scale).astype(dlhs.dtype)
    dlhs = qpl.QArray(
        qvalue=dlhs * lhs_scale,
        scale=jnp.zeros_like(residual_lhs.scale),
        zero_point=jnp.zeros_like(residual_lhs.zero_point) if residual_lhs.zero_point is not None else None,
        qtype=residual_lhs.qtype,
    )
  if rhs_is_qarray and isinstance(residual_rhs, qpl.QArray):
    drhs = qpl.QArray(
        qvalue=drhs.astype(residual_rhs.qvalue.dtype) if drhs.dtype != residual_rhs.qvalue.dtype else drhs,
        scale=jnp.zeros_like(residual_rhs.scale),
        zero_point=jnp.zeros_like(residual_rhs.zero_point) if residual_rhs.zero_point is not None else None,
        qtype=residual_rhs.qtype,
    )

  return dlhs, drhs, None, None, d_existing_out, dpartial_sum


def _bwd_prepare_inputs(
    grad: jnp.ndarray,
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    use_gmm_v2: bool,
    transpose_rhs: bool,
    quantization_rule: qwix.QtRule | None,
    skip_lhs_quant: bool = False,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray, jnp.ndarray, jnp.ndarray]:
  """Prepares backward operands.

  `skip_lhs_quant=True` (bwd_inkernel_quant) keeps the lhs as the raw wide array: the drhs
  tgmm quantizes BOTH operands in-kernel over valid gm tiles only, so the dense per-row XLA
  quantize here (and the drhs_dout *= lhs.scale multiply below) would be buffer-sized overhead.
  """

  # dlhs_dout and drhs_dout can be different when quantization is enabled.
  dlhs_dout = grad
  drhs_dout = grad

  # Apply rhs.scale to dlhs_dout, dlhs_dout[m, n] @ rhs_transpose[g, n, k] = dlhs[m, k]
  # Assume channelwise scale on rhs n.
  # Apply rhs.scale to dlhs_dout to avoid dequantizing or requantizing rhs.
  # We cannot apply the scale to dlhs because axis n will disappear there.
  if isinstance(rhs, qpl.QArray):
    # rhs - qvalue: [g, k, n] scale: [1, 1, n], assume transpose_rhs=False
    if not use_gmm_v2:
      dlhs_dout *= rhs.scale.astype(grad.dtype).reshape(1, -1)
      rhs = rhs.qvalue
    else:
      # NOTE: rhs.scale is for the contracting dimension (N) in DLHS, but gmm_v2
      # only supports scaling the output dimension. Thus, we must scale dlhs_dout
      # beforehand.
      dlhs_dout = _dlhs_scale_grad_by_rhs_scale(dlhs_dout, rhs, group_sizes, transpose_rhs)
      rhs = rhs.qvalue

  # GMM2 FWD performs lhs quantization inside kernel, lhs is stored as unquantized dtype
  # in the residual tuple. In BWD, we explicitly quantize lhs.
  if quantization_rule and quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray) and not skip_lhs_quant:
    lhs = qpl.quantize(  # pyrefly: ignore[bad-assignment]
        lhs,
        quantization_rule.act_qtype,
        # pyrefly: ignore[bad-argument-type]
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        # pyrefly: ignore[bad-argument-type]
        calibration_method=quantization_rule.act_calibration_method,
    )

  # Assume channelwise scale on lhs m, lhs_transpose[k, m] @ drhs_out[m, n] = drhs[g, k, n]
  # Apply lhs.scale to drhs_dout, as axis m will disappear in drhs.
  if isinstance(lhs, qpl.QArray):
    # lhs - qvalue: [m, k] scale: [m, 1]
    drhs_dout = drhs_dout * (lhs.scale.squeeze() if lhs.scale.size == 1 else lhs.scale).astype(grad.dtype)
    lhs = lhs.qvalue

  return dlhs_dout, drhs_dout, lhs, rhs


def _bwd_quantize_gradient(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    drhs_dout: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray]:
  """Applies backward quantization to incoming gradients."""
  if quantization_rule.bwd_qtype:
    dlhs_dout = qpl.quantize(
        # pyrefly: ignore[bad-argument-type]
        dlhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        calibration_method=quantization_rule.bwd_calibration_method,
    )
    drhs_dout = qpl.quantize(
        # pyrefly: ignore[bad-argument-type]
        drhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [1],
        calibration_method=quantization_rule.bwd_calibration_method,
    )
  return dlhs_dout, drhs_dout


# ==============================================================================
# BWD DLHS GMM
# ==============================================================================


def _compute_dlhs(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    lhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    transpose_rhs: bool,
    use_tokamax_backend: bool,
    use_gmm_v2: bool,
    use_manual_quantization: bool,
    interpret: bool,
    lhs_vma_axes: tuple,
    use_gmm_v2_heuristic_tiling: bool,
) -> jnp.ndarray:
  """Routes execution of DLHS based on backend choices."""
  if use_tokamax_backend and not use_gmm_v2:
    return _dlhs_run_tokamax_v1(
        dlhs_dout,
        rhs,
        group_sizes,
        lhs_dtype,
        transpose_rhs,
        use_manual_quantization,
    )
  elif use_tokamax_backend and use_gmm_v2:
    return _dlhs_run_tokamax_v2(
        dlhs_dout, rhs, group_sizes, group_offset, lhs_dtype, tiling, use_gmm_v2_heuristic_tiling, transpose_rhs
    )
  else:
    return _dlhs_run_megablox(
        dlhs_dout, rhs, group_sizes, group_offset, lhs_dtype, tiling, transpose_rhs, interpret, lhs_vma_axes
    )


def _dlhs_run_tokamax_v1(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    lhs_dtype: jax.typing.DTypeLike,
    transpose_rhs: bool,
    use_manual_quantization: bool,
) -> jnp.ndarray:
  """Executes DLHS using GMM 1"""
  dlhs_kwargs = {}
  if use_manual_quantization:
    dlhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"]))

  dlhs_rhs = rhs.swapaxes(1, 2) if transpose_rhs else rhs
  return tokamax.ragged_dot_general(
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


def _dlhs_scale_grad_by_rhs_scale(
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


def _dlhs_run_tokamax_v2(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    lhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    use_gmm_v2_heuristic_tiling: bool,
    transpose_rhs: bool,
) -> jnp.ndarray:
  """Executes Tokamax GMM V2 backend for DLHS = DLHS_dout @ RHS^T."""
  # NOTE: We manually transpose RHS here because gmm_v2 lacks native transpose_rhs support.
  dlhs_rhs = rhs if transpose_rhs else rhs.swapaxes(1, 2)
  dlhs_lhs = dlhs_dout.qvalue if isinstance(dlhs_dout, qpl.QArray) else dlhs_dout

  if use_gmm_v2_heuristic_tiling:
    dlhs_tiling = gmm_v2.calculate_tiling
  else:
    dlhs_tiling = gmm_v2.TileSizes(tile_m=tiling[3], tile_k=tiling[4], tile_n=tiling[5])

  dlhs = gmm_v2.gmm_v2(
      lhs=dlhs_lhs,
      rhs=dlhs_rhs,
      group_sizes=group_sizes,
      # rhs scale is already applied to dlhs_lhs
      rhs_scale=None,
      tile_info=dlhs_tiling,
      preferred_element_type=lhs_dtype,  # pyrefly: ignore[bad-argument-type]
      group_offset=group_offset,
      maybe_quantize_lhs=not isinstance(dlhs_dout, qpl.QArray),
  )

  if isinstance(dlhs_dout, qpl.QArray):
    dlhs = dlhs * (dlhs_dout.scale.squeeze() if dlhs_dout.scale.size == 1 else dlhs_dout.scale).astype(dlhs.dtype)

  return dlhs


def _dlhs_run_megablox(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    lhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    transpose_rhs: bool,
    interpret: bool,
    lhs_vma_axes: tuple,
) -> jnp.ndarray:
  """Executes Megablox fallback for DLHS."""
  return backend.gmm(
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


# ==============================================================================
# BWD DRHS TGMM
# ==============================================================================


def _compute_drhs(
    drhs_dout: jnp.ndarray | qpl.QArray,
    lhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    num_actual_groups: int,
    rhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    use_tokamax_backend: bool,
    use_gmm_v2: bool,
    use_manual_quantization: bool,
    weight_gather_axes: List[Tuple[str, int]] | None,
    interpret: bool,
    rhs_vma_axes: tuple,
    quantization_rule: qwix.QtRule | None,
    use_gmm_v2_heuristic_tiling: bool,
    inkernel_quant: bool = False,
) -> jnp.ndarray:
  """Routes execution of DRHS based on backend choices."""
  if use_tokamax_backend and not use_gmm_v2:
    drhs = _drhs_run_tokamax_v1(drhs_dout, lhs, group_sizes, rhs_dtype, use_manual_quantization)
  elif use_tokamax_backend and use_gmm_v2:
    drhs = _drhs_run_tokamax_v2(
        drhs_dout, lhs, group_sizes, group_offset, num_actual_groups, rhs_dtype, tiling, use_gmm_v2_heuristic_tiling,  quantize_operands=inkernel_quant,
    )
  else:
    drhs = _drhs_run_megablox(
        drhs_dout, lhs, group_sizes, group_offset, num_actual_groups, rhs_dtype, tiling, interpret, rhs_vma_axes
    )

  if use_tokamax_backend and quantization_rule and quantization_rule.bwd_qtype and weight_gather_axes:
    drhs = _drhs_scatter_weight(drhs, weight_gather_axes)

  return drhs


def _drhs_scatter_weight(drhs: jnp.ndarray, weight_gather_axes: List[Tuple[str, int]]) -> jnp.ndarray:
  """Scatters the DRHS output back in the reverse order of the forward gather."""
  for axis_name, axis_idx in reversed(weight_gather_axes):
    drhs = jax.lax.psum_scatter(drhs, axis_name, scatter_dimension=axis_idx, tiled=True)
  return drhs


def _drhs_run_tokamax_v1(
    drhs_dout: jnp.ndarray | qpl.QArray,
    lhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    rhs_dtype: jax.typing.DTypeLike,
    use_manual_quantization: bool,
) -> jnp.ndarray:
  """Executes standard Tokamax ragged_dot for DRHS."""
  drhs_kwargs = {}
  if use_manual_quantization:
    drhs_kwargs["manual_axis_type"] = jax.sharding.ManualAxisType(
        varying=frozenset(["expert"]), unreduced=frozenset(["data", "fsdp"])
    )
  return tokamax.ragged_dot_general(
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


def _drhs_prepare_bwd_scale(drhs_dout: qpl.QArray) -> jnp.ndarray:
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


def _drhs_run_tokamax_v2(
    drhs_dout: jnp.ndarray | qpl.QArray,
    lhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    num_actual_groups: int,
    rhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    use_gmm_v2_heuristic_tiling: bool,
    quantize_operands: bool = False,
) -> jnp.ndarray:
  """Executes Tokamax TGMM V2 backend for DRHS = LHS^T @ DRHS_dout."""
  drhs_rhs = drhs_dout.qvalue if isinstance(drhs_dout, qpl.QArray) else drhs_dout
  drhs_lhs = lhs

  rhs_scale = None
  if isinstance(drhs_dout, qpl.QArray):
    rhs_scale = _drhs_prepare_bwd_scale(drhs_dout)

  if use_gmm_v2_heuristic_tiling:
    drhs_tiling = tgmm_v2.calculate_tgmm_tiling
  else:
    drhs_tiling = gmm_v2.TileSizes(tile_m=tiling[6], tile_k=tiling[7], tile_n=tiling[8])

  return tgmm_v2.tgmm_v2(
      lhs=drhs_lhs,
      rhs=drhs_rhs,
      group_sizes=group_sizes,
      num_actual_groups=num_actual_groups,
      rhs_scale=rhs_scale,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=rhs_dtype,  # pyrefly: ignore[bad-argument-type]
      group_offset=group_offset,
      tile_info=drhs_tiling,
      quantize_operands=quantize_operands and rhs_scale is None,
  )


def _drhs_run_megablox(
    drhs_dout: jnp.ndarray | qpl.QArray,
    lhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    num_actual_groups: int,
    rhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    interpret: bool,
    rhs_vma_axes: tuple,
) -> jnp.ndarray:
  """Executes Megablox fallback for DRHS."""
  return backend.tgmm(
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
