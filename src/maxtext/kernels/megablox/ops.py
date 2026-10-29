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
from typing import List, Literal, NamedTuple, Tuple
from absl import logging
import jax
import jax.numpy as jnp
from maxtext.kernels.megablox import backend
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.ragged import ragged_sort_tc
from maxtext.layers import quantizations
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


class TcUnsortCfg(NamedTuple):
  """Static config to fuse the TC unsort-reduce (and optionally the EP combine) into the GMM custom VJP.

  The forward reduces the GMM's expert-slot rows back to token order; the backward quantizes the smaller
  token-order cotangent first and gathers it into expert slots afterwards.
  """

  unpadded_cap: int  # GMM output rows that hold expert slots; the rest is tile padding.
  num_out_tokens: int
  topk: int
  blocks: tuple  # ragged_sort_tc blocks: (gather_block_size, flatten_block_size, keep_3d, tokens_3d).
  mask_padding: bool
  combine_rs_cfg: tuple[str, int] | None = None  # (ep_axis_name, seq_len): also fuse the EP psum_scatter.


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
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
    return_rhs: bool = False,
    return_lhs: bool = False,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    flat_fsdp_shard: tuple[int, int, int] | None = None,
    out_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
    tc_routing: object = None,
    tc_unsort_cfg: TcUnsortCfg | None = None,
):
  """Grouped matrix multiplication operation.

  A 3D lhs ([m, k // 128, 128], e.g. tokens from the TensorCore ragged sort) is
  detected from its rank; `out_is_3d` writes the output as [m, n // 128, 128].
  Both are only supported by the tokamax gmm_v2 backend; the backward pass then
  consumes / produces the matching 3D cotangents without relayout copies. With
  `out_is_3d`, a forward `partial_sum` must also be 3D; with `return_lhs`, the
  cotangent of the returned (3D) lhs is accumulated into the 3D dlhs in place.
  """
  lhs_ndim = (lhs.qvalue if isinstance(lhs, qpl.QArray) else lhs).ndim
  if (lhs_ndim == 3 or out_is_3d) and not (use_tokamax_backend and use_gmm_v2):
    raise NotImplementedError("3D lhs / out requires use_tokamax_backend=True and use_gmm_v2=True.")
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
    # Only "fp8_full" defines a Qwix rule for GMM.
    # quantization_rule is None when:
    # 1. Quantization scheme does not target GMM (e.g., "fp8", "int8").
    # 2. Module is excluded from quantization (e.g., MTP when quantize_mtp=False).
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

  if return_rhs and not (use_tokamax_backend and use_gmm_v2):
    # The cotangent of the returned rhs is accumulated into drhs via tgmm_v2 partial_sum,
    # which other backends do not support; they would silently drop it.
    raise ValueError("return_rhs=True requires use_tokamax_backend=True and use_gmm_v2=True.")

  if return_lhs and not (use_tokamax_backend and use_gmm_v2):
    # The cotangent of the returned lhs is accumulated into dlhs via gmm_v2 partial_sum,
    # which other backends do not support; they would silently drop it.
    raise ValueError("return_lhs=True requires use_tokamax_backend=True and use_gmm_v2=True.")

  lhs_dtype = preferred_element_type if isinstance(lhs, qpl.QArray) else lhs.dtype
  rhs_dtype = preferred_element_type if isinstance(rhs, qpl.QArray) else rhs.dtype
  gmm_fwd_bwd = lambda *args: _gmm_fwd(*args)[0]  # pylint: disable=C3001
  gmm_fwd_bwd = jax.custom_vjp(
      gmm_fwd_bwd,
      nondiff_argnums=(3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26),
  )
  gmm_fwd_bwd.defvjp(_gmm_fwd, functools.partial(_gmm_bwd, lhs_dtype, rhs_dtype))
  out, lhs_out, rhs_out = gmm_fwd_bwd(
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
      return_rhs,
      return_lhs,
      fuse_dlhs_scale,
      use_dlhs_transpose_rhs,
      flat_fsdp_shard,
      out_is_3d,
      fuse_3d_dlhs_ps,
      tc_routing,
      tc_unsort_cfg,
  )
  if return_lhs and return_rhs:
    return out, lhs_out, rhs_out
  if return_lhs:
    return out, lhs_out
  if return_rhs:
    return out, rhs_out
  return out


def _scale_like(scale: jnp.ndarray, x: jnp.ndarray, dtype) -> jnp.ndarray:
  """Per-tensor scale -> scalar; per-row scale [m, 1, ...] -> [m, 1, ..., 1] broadcastable to x."""
  if scale.size == 1:
    return scale.reshape(()).astype(dtype)
  if x.ndim == 3 or (scale.ndim != x.ndim and scale.size == scale.shape[0]):
    # Per-row scale whose rank may differ from x (3D [m, 1, 1] vs 2D [m, k] or vice versa).
    return scale.reshape(scale.shape[0], *([1] * (x.ndim - 1))).astype(dtype)
  return scale.astype(dtype)


# ==============================================================================
# Forward: FWD GMM
# ==============================================================================


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
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
    return_rhs: bool = False,
    return_lhs: bool = False,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    flat_fsdp_shard: tuple[int, int, int] | None = None,
    out_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
    tc_routing: object = None,
    tc_unsort_cfg: TcUnsortCfg | None = None,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray],
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        bool,
        bool,
        object,
    ],
]:
  """Forward function for GMM VJP.

  - lhs: [m, k] (or [m, k // 128, 128])
  - rhs: [g, k, n] if transpose_rhs=False. [g, n, k] if transpose_rhs=True
  - out: [m, n], or [m, n // 128, 128] if out_is_3d

  Always returns (out, lhs, rhs) so the output structure does not depend on return_lhs/return_rhs;
  gmm() drops lhs unless return_lhs=True and rhs unless return_rhs=True.
  """

  del use_dlhs_transpose_rhs  # Only affects the custom backward pass.

  # Track whether operands arrived as QArray (e.g. from token all-gather quantization
  # or gathered weights) to return matching cotangent containers in backward pass.
  lhs_is_qarray = isinstance(lhs, qpl.QArray)
  rhs_is_qarray = isinstance(rhs, qpl.QArray)
  if flat_fsdp_shard is not None:
    if rhs_is_qarray or transpose_rhs or rhs.ndim != 2 or not weight_gather_axes:
      raise NotImplementedError("flat_fsdp_shard needs a plain 2D rhs block, transpose_rhs=False and weight_gather_axes.")
    if any(axis_idx != 0 for _, axis_idx in weight_gather_axes):
      raise ValueError("flat_fsdp_shard: the flat rhs is gathered on dim 0 only.")
    if not (
        use_tokamax_backend
        and quantization_rule
        and quantization_rule.bwd_qtype
        and quantization_rule.weight_calibration_method.startswith("fixed")
    ):
      raise NotImplementedError("flat_fsdp_shard requires the tokamax QAG path (fixed weight calibration, bwd_qtype).")

  # Quantize activation and weight
  if quantization_rule:
    # pyrefly: ignore[bad-assignment]
    lhs, rhs = _fwd_quantize_activation_and_weight(
        lhs,
        rhs,
        quantization_rule,
        use_gmm_v2,
        use_manual_quantization,
        transpose_rhs,
        rhs_is_flat2d=flat_fsdp_shard is not None,
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
    rhs = _fwd_gather_weight(rhs, weight_gather_axes, gathered_shape=flat_fsdp_shard)

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
        out_is_3d=out_is_3d,
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

  if tc_unsort_cfg is not None:
    cfg = tc_unsort_cfg
    out = ragged_sort_tc._gather_reduce(  # pylint: disable=protected-access
        out[: cfg.unpadded_cap], tc_routing, cfg.num_out_tokens, cfg.topk, cfg.blocks
    )
    if cfg.combine_rs_cfg is not None:
      # Reduce-scatter the [tokens, D // 128, 128] combine over EP into the local [batch, seq, D].
      ep_axis_name, seq_len = cfg.combine_rs_cfg
      out = out.reshape(-1, seq_len, *out.shape[1:]).astype(preferred_element_type)
      out = jax.lax.psum_scatter(out, ep_axis_name, scatter_dimension=0, tiled=True)
      out = out.reshape(*out.shape[:2], -1)

  del return_rhs, return_lhs, fuse_dlhs_scale, fuse_3d_dlhs_ps  # Only used by gmm() and _gmm_bwd.
  return (out, lhs, rhs), (
      lhs,
      rhs,
      group_sizes,
      group_offset,
      partial_sum,
      lhs_is_qarray,
      rhs_is_qarray,
      tc_routing,
  )


def _fwd_quantize_activation_and_weight(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule,
    use_gmm_v2: bool,
    use_manual_quantization: bool,
    transpose_rhs: bool,
    rhs_is_flat2d: bool = False,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray]:
  """Handles act and weight quantization for GMM forward inputs."""
  if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray) and not use_gmm_v2:
    lhs = qpl.quantize(
        lhs,
        quantization_rule.act_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        # pyrefly: ignore[bad-argument-type]
        calibration_method=quantization_rule.act_calibration_method,
    )

  if quantization_rule.weight_qtype and not isinstance(rhs, qpl.QArray):
    if not use_manual_quantization:
      rhs = qpl.quantize(
          rhs,
          quantization_rule.weight_qtype,
          # If only considering the fwd pass, we could also enable channelwise
          # axes for the group axis, i.e., [0, 1 or 2]. However, this makes the
          # bwd pass unable to reuse the scale easily.
          channelwise_axes=(
              []
              if quantization_rule.disable_channelwise_axes
              else ([1] if (transpose_rhs or rhs_is_flat2d) else [2])  # flat 2D [rows, cols]: cols is the N axis
          ),
          calibration_method=quantization_rule.weight_calibration_method,
      )
    else:
      rhs = quantizations.manual_quantize(
          rhs,
          quantization_rule.weight_qtype,
          calibration_method=quantization_rule.weight_calibration_method,
      )
  return lhs, rhs


def _fwd_gather_weight(
    rhs: qpl.QArray,
    weight_gather_axes: List[Tuple[str, int]],
    gathered_shape: tuple[int, ...] | None = None,
) -> qpl.QArray:
  """Applies QAG (Quantization All-Gather) to RHS weights during forward pass.

  With `gathered_shape` (moe_flat_fsdp_weights) the gathered 2D [G * rows, cols]
  qvalue is bitcast to (G, rows, cols) and a per-channel [1, cols] scale to [1, 1, cols].
  """
  if gathered_shape is None:
    for axis_name, axis_idx in weight_gather_axes:
      rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
      # replace the qvalue with the gathered qvalue in the QArray
      rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
    return rhs

  qvalue = rhs.qvalue
  for axis_name, axis_idx in weight_gather_axes:
    qvalue = jax.lax.all_gather(qvalue, axis_name, axis=axis_idx, tiled=True)
  qvalue = qvalue.reshape(gathered_shape)

  def _expand(s):
    if s is None or s.ndim == 0 or s.ndim == len(gathered_shape):
      return s
    return s.reshape((1,) * (len(gathered_shape) - s.ndim) + tuple(s.shape))

  return dataclasses.replace(rhs, qvalue=qvalue, scale=_expand(rhs.scale), zero_point=_expand(rhs.zero_point))


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
    out_is_3d: bool = False,
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

  # When lhs is already a QArray (e.g. from quantized token all-gather), extract raw
  # qvalue buffer and skip redundant internal quantization inside gmm_v2.
  if isinstance(lhs, qpl.QArray):
    if partial_sum is not None:
      raise NotImplementedError(
          "partial_sum is not supported with pre-quantized lhs QArray because"
          " scaling output by lhs.scale would erroneously rescale partial_sum."
      )
    lhs_operand = lhs.qvalue
    lhs_scale = None
    maybe_quantize_lhs = False
  else:
    lhs_operand = lhs
    lhs_scale = _fwd_prepare_lhs_scale(quantization_rule)
    maybe_quantize_lhs = True

  if use_gmm_v2_heuristic_tiling:
    fwd_tiling = gmm_v2.calculate_tiling
  else:
    fwd_tiling = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2])

  out = gmm_v2.gmm_v2(
      lhs=lhs_operand,
      rhs=rhs_operand,
      group_sizes=group_sizes,
      rhs_scale=rhs_scale,
      tile_info=fwd_tiling,
      preferred_element_type=preferred_element_type,
      partial_sum=partial_sum,
      group_offset=group_offset,
      lhs_scale=lhs_scale,
      maybe_quantize_lhs=maybe_quantize_lhs,
      out_is_3d=out_is_3d,
  )

  # gmm_v2 only rescales output when it quantizes lhs internally; for pre-quantized QArray
  # inputs, apply lhs.scale to the accumulated output here.
  if isinstance(lhs, qpl.QArray):
    out = out * _scale_like(lhs.scale, out, out.dtype)

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
    return_rhs: bool,
    return_lhs: bool,
    fuse_dlhs_scale: bool,
    use_dlhs_transpose_rhs: bool,
    flat_fsdp_shard: tuple[int, int, int] | None,
    out_is_3d: bool,
    fuse_3d_dlhs_ps: bool,
    tc_unsort_cfg: TcUnsortCfg | None,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        bool,
        bool,
        object,
    ],
    grad: jnp.ndarray,
) -> tuple[
    jnp.ndarray | qpl.QArray,
    jnp.ndarray | qpl.QArray,
    None,
    None,
    jnp.ndarray | None,
    jnp.ndarray | None,
    None,
]:
  """Backward function for throughput GMM VJP."""
  (
      residual_lhs,
      residual_rhs,
      group_sizes,
      group_offset,
      partial_sum_fwd,
      lhs_is_qarray,
      rhs_is_qarray,
      tc_routing,
  ) = residual
  num_actual_groups = residual_rhs.shape[0]
  grad, dlhs_partial_sum, drhs_partial_sum = grad
  # With return_lhs, the cotangent of the returned lhs is the input gradient already computed by its
  # other consumers (e.g. the MoE wi_1 GMM). The DLHS GMM adds this dlhs onto it in place.
  if not return_lhs:
    dlhs_partial_sum = None
  elif isinstance(dlhs_partial_sum, qpl.QArray):
    dlhs_partial_sum = dlhs_partial_sum.qvalue
  # With return_rhs, the cotangent of the returned rhs is the weight gradient already accumulated
  # by its downstream consumers (e.g. later MoE token chunks). tgmm adds this drhs onto it in place.
  if not return_rhs:
    drhs_partial_sum = None
  else:
    if isinstance(drhs_partial_sum, qpl.QArray):
      drhs_partial_sum = drhs_partial_sum.qvalue
    if transpose_rhs:
      drhs_partial_sum = drhs_partial_sum.swapaxes(1, 2)
  # 3D layouts: the incoming grad matches the (possibly 3D) forward output and dlhs is
  # produced in the layout of the forward lhs.
  lhs_is_3d = (residual_lhs.qvalue if isinstance(residual_lhs, qpl.QArray) else residual_lhs).ndim == 3
  if tc_unsort_cfg is not None:
    _check_tc_unsort_bwd_supported(residual_lhs, residual_rhs, quantization_rule, tc_unsort_cfg)
    if tc_unsort_cfg.combine_rs_cfg is not None:
      # The fused forward returned the local [batch, seq, D] combine; go back to the TC token layout.
      grad = grad.reshape(-1, grad.shape[-1] // 128, 128)
  del out_is_3d  # grad.ndim carries it.

  # Jargon used here:
  #  - lhs: input activation in forward pass, possibly quantized.
  #  - rhs: weight in forward pass, possibly quantized.
  #  - dout (or grad): the incoming gradient in the backward pass.
  #  - dlhs: gradient of the lhs in the backward pass, what we want to compute.
  #  - drhs: gradient of the rhs in the backward pass, what we want to compute.
  #  - dlhs_dout: the incoming gradient used to calculate dlhs.
  #  - drhs_dout: the incoming gradient used to calculate drhs.

  # 1. Scale Application & QArray Unwrapping
  dlhs_dout, drhs_dout, lhs, rhs = _bwd_prepare_inputs(
      grad,
      residual_lhs,
      residual_rhs,
      group_sizes,
      use_gmm_v2,
      transpose_rhs,
      quantization_rule,
  )

  # 2. Backward Pass Quantization
  if quantization_rule:
    dlhs_dout, drhs_dout = _bwd_quantize_gradient(dlhs_dout, drhs_dout, quantization_rule)

  # 2b. Fused TC unsort: gather the (already scaled / quantized) token-order cotangent into expert slots.
  if tc_unsort_cfg is not None:
    cfg = tc_unsort_cfg
    padded_cap = (residual_lhs.qvalue if isinstance(residual_lhs, qpl.QArray) else residual_lhs).shape[0]

    def _unsort_gather_arr(arr):
      if cfg.combine_rs_cfg is not None:
        arr = jax.lax.all_gather(arr, cfg.combine_rs_cfg[0], axis=0, tiled=True)
      arr = ragged_sort_tc._gather(arr, tc_routing, cfg.unpadded_cap, cfg.blocks, cfg.mask_padding)  # pylint: disable=protected-access
      return jnp.pad(arr, [(0, padded_cap - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1))

    def _unsort_gather_maybe_qa(x):
      if isinstance(x, qpl.QArray):
        return dataclasses.replace(x, qvalue=_unsort_gather_arr(x.qvalue))
      return _unsort_gather_arr(x)

    same_dout = dlhs_dout is drhs_dout
    dlhs_dout = _unsort_gather_maybe_qa(dlhs_dout)
    drhs_dout = dlhs_dout if same_dout else _unsort_gather_maybe_qa(drhs_dout)

  lhs_scale_for_dlhs = None
  if lhs_is_qarray and isinstance(residual_lhs, qpl.QArray):
    lhs_scale_for_dlhs = residual_lhs.scale

  # 3. DLHS Gradient Execution
  dlhs, lhs_scale_applied = _compute_dlhs(
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
      partial_sum=dlhs_partial_sum,
      lhs_scale=lhs_scale_for_dlhs,
      fuse_dlhs_scale=fuse_dlhs_scale,
      use_dlhs_transpose_rhs=use_dlhs_transpose_rhs,
      dlhs_is_3d=lhs_is_3d,
      fuse_3d_dlhs_ps=fuse_3d_dlhs_ps,
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
      partial_sum=drhs_partial_sum,
      flat_rhs_shard=flat_fsdp_shard is not None,
  )

  # 5. Output Formatting
  # NOTE: If the rhs transposition is fused into the forward pass we need to
  # return the transpose of the rhs gradient that we calculated above.
  #
  # TODO(tgale, enriqueps, apaske): Fuse this transposition into the tgmm.
  drhs = drhs.swapaxes(1, 2) if transpose_rhs else drhs
  dpartial_sum = grad if partial_sum_fwd is not None else None
  d_existing_out = None if use_tokamax_backend else grad

  # If forward inputs were QArray, wrap cotangents in QArray to match JAX pytree container
  # structure expected by autodiff.
  if lhs_is_qarray and isinstance(residual_lhs, qpl.QArray):
    if not lhs_scale_applied:
      # Scale dlhs by lhs_scale to propagate chain rule through pre-quantized QArray input.
      dlhs = dlhs * _scale_like(residual_lhs.scale, dlhs, dlhs.dtype)
    dlhs = qpl.QArray(
        qvalue=dlhs,
        scale=jnp.zeros_like(residual_lhs.scale),
        zero_point=jnp.zeros_like(residual_lhs.zero_point) if residual_lhs.zero_point is not None else None,
        qtype=residual_lhs.qtype,
    )
  if rhs_is_qarray and isinstance(residual_rhs, qpl.QArray):
    # Keep drhs.qvalue in full-precision float for accurate optimizer updates (do not cast
    # or quantize weight gradients to the quantized dtype, e.g. FP8). Scale gradient is zero.
    drhs = qpl.QArray(
        qvalue=drhs,
        scale=jnp.zeros_like(residual_rhs.scale),
        zero_point=jnp.zeros_like(residual_rhs.zero_point) if residual_rhs.zero_point is not None else None,
        qtype=residual_rhs.qtype,
    )

  return dlhs, drhs, None, None, d_existing_out, dpartial_sum, None


def _check_tc_unsort_bwd_supported(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule | None,
    tc_unsort_cfg: TcUnsortCfg,
):
  """The fused TC unsort scales and quantizes the cotangent in token order, before the gather into expert slots."""
  # Per-row lhs and per-expert rhs scales are laid out by expert slot, so they can't be applied in token order.
  for name, x in (("lhs", lhs), ("rhs", rhs)):
    if isinstance(x, qpl.QArray) and x.scale.size != 1:
      raise NotImplementedError(f"tc_unsort_cfg requires a per-tensor {name} scale, got shape {x.scale.shape}.")
  if quantization_rule is None:
    return
  if (
      quantization_rule.act_qtype
      and not isinstance(lhs, qpl.QArray)
      and not quantization_rule.disable_channelwise_axes
      and not quantizations.is_static_calibration(quantization_rule.act_calibration_method)
  ):
    raise NotImplementedError("tc_unsort_cfg requires a per-tensor act scale (fixed or disable_channelwise_axes).")
  if tc_unsort_cfg.combine_rs_cfg is not None:
    # Each EP shard applies the rhs / lhs scales to its local cotangent and quantizes it before the all-gather, so
    # every scale must be the same on all shards; a dynamic one is computed from the shard's own experts and rows.
    for name, qtype, method in (
        ("weight", quantization_rule.weight_qtype, quantization_rule.weight_calibration_method),
        ("act", quantization_rule.act_qtype, quantization_rule.act_calibration_method),
        ("bwd", quantization_rule.bwd_qtype, quantization_rule.bwd_calibration_method),
    ):
      if qtype and not quantizations.is_static_calibration(method):
        raise NotImplementedError(
            f"tc_unsort_cfg with combine_rs_cfg requires a fixed {name}_calibration_method, got {method!r}."
        )
  if not quantization_rule.bwd_qtype or quantizations.is_static_calibration(quantization_rule.bwd_calibration_method):
    return
  # A dynamic per-row cotangent scale would be in token order, so it can't be gathered like the qvalue.
  if not quantization_rule.disable_channelwise_axes:
    raise NotImplementedError("tc_unsort_cfg requires a per-tensor bwd scale (fixed or disable_channelwise_axes).")


def _lhs_rhs_share_fixed_scale(
    quantization_rule: qwix.QtRule | None,
    lhs: qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
) -> bool:
  """True when lhs and rhs carry the same fixed per-tensor scale, so grad * rhs.scale == grad * lhs.scale.

  Scale equality is inferred from the rule, not the (traced) arrays: qwix builds a fixed scale from the
  calibration bound in the quantized array's dtype, so the scales match only if both operands were quantized
  under this rule's calibration at the same dtype. A pre-quantized lhs (moe_quantize_token_all_gather) is
  checked to use the rule's act_calibration_method; a scale dtype mismatch falls back to applying lhs.scale.
  """
  return (
      isinstance(rhs, qpl.QArray)
      and quantization_rule is not None
      and quantization_rule.weight_qtype == quantization_rule.act_qtype
      and quantizations.is_static_calibration(quantization_rule.weight_calibration_method)
      and quantization_rule.weight_calibration_method == quantization_rule.act_calibration_method
      and lhs.scale.size == 1
      and rhs.scale.size == 1
      and lhs.scale.dtype == rhs.scale.dtype
  )


def _bwd_prepare_inputs(
    grad: jnp.ndarray,
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    use_gmm_v2: bool,
    transpose_rhs: bool,
    quantization_rule: qwix.QtRule | None,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray, jnp.ndarray, jnp.ndarray]:
  """Prepares backward operands."""

  # dlhs_dout and drhs_dout can be different when quantization is enabled.
  dlhs_dout = grad
  drhs_dout = grad
  orig_rhs = rhs

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
  if quantization_rule and quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray):
    lhs = qpl.quantize(
        lhs,
        quantization_rule.act_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        # pyrefly: ignore[bad-argument-type]
        calibration_method=quantization_rule.act_calibration_method,
    )

  # Assume channelwise scale on lhs m, lhs_transpose[k, m] @ drhs_out[m, n] = drhs[g, k, n]
  # Apply lhs.scale to drhs_dout, as axis m will disappear in drhs.
  if isinstance(lhs, qpl.QArray):
    # lhs - qvalue: [m, k] scale: [m, 1]
    if _lhs_rhs_share_fixed_scale(quantization_rule, lhs, orig_rhs):
      drhs_dout = dlhs_dout
    else:
      drhs_dout = drhs_dout * _scale_like(lhs.scale, drhs_dout, grad.dtype)
    lhs = lhs.qvalue

  return dlhs_dout, drhs_dout, lhs, rhs


def _bwd_quantize_gradient(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    drhs_dout: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray]:
  """Applies backward quantization to incoming gradients."""
  if quantization_rule.bwd_qtype:
    bwd_cal = quantization_rule.bwd_calibration_method
    share_quantized_grad = dlhs_dout is drhs_dout and quantizations.is_static_calibration(bwd_cal)
    dlhs_dout = qpl.quantize(
        # pyrefly: ignore[bad-argument-type]
        dlhs_dout,
        quantization_rule.bwd_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        calibration_method=bwd_cal,
    )
    if share_quantized_grad:
      drhs_dout = dlhs_dout
    else:
      drhs_dout = qpl.quantize(
          # pyrefly: ignore[bad-argument-type]
          drhs_dout,
          quantization_rule.bwd_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else list(range(1, drhs_dout.ndim)),
          calibration_method=bwd_cal,
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
    partial_sum: jnp.ndarray | None = None,
    lhs_scale: jnp.ndarray | None = None,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    dlhs_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
) -> tuple[jnp.ndarray, bool]:
  """Routes execution of DLHS based on backend choices.

  Returns (dlhs, lhs_scale_applied). partial_sum (only set with return_lhs, i.e. tokamax gmm_v2) is
  added to dlhs; lhs_scale_applied tells the caller whether lhs_scale was already folded into dlhs.
  """
  if use_tokamax_backend and not use_gmm_v2:
    dlhs = _dlhs_run_tokamax_v1(
        dlhs_dout,
        rhs,
        group_sizes,
        lhs_dtype,
        transpose_rhs,
        use_manual_quantization,
    )
    return dlhs, False
  elif use_tokamax_backend and use_gmm_v2:
    return _dlhs_run_tokamax_v2(
        dlhs_dout,
        rhs,
        group_sizes,
        group_offset,
        lhs_dtype,
        tiling,
        use_gmm_v2_heuristic_tiling,
        transpose_rhs,
        partial_sum=partial_sum,
        lhs_scale=lhs_scale,
        fuse_dlhs_scale=fuse_dlhs_scale,
        use_dlhs_transpose_rhs=use_dlhs_transpose_rhs,
        out_is_3d=dlhs_is_3d,
        fuse_3d_dlhs_ps=fuse_3d_dlhs_ps,
    )
  else:
    dlhs = _dlhs_run_megablox(
        dlhs_dout, rhs, group_sizes, group_offset, lhs_dtype, tiling, transpose_rhs, interpret, lhs_vma_axes
    )
    return dlhs, False


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
  if rhs_scale.size == 1:
    return grad * rhs_scale.reshape(()).astype(grad.dtype)
  if rhs_scale.shape[0] == 1:
    scale = rhs_scale.astype(grad.dtype)
  else:
    scale = jnp.repeat(
        rhs_scale.astype(grad.dtype),
        group_sizes,
        axis=0,
        total_repeat_length=grad.shape[0],
    )
  if grad.ndim == 3:
    scale = scale.reshape(scale.shape[0], *grad.shape[1:])
  return grad * scale


def _dlhs_run_tokamax_v2(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    lhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    use_gmm_v2_heuristic_tiling: bool,
    transpose_rhs: bool,
    partial_sum: jnp.ndarray | None = None,
    lhs_scale: jnp.ndarray | None = None,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    out_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
) -> tuple[jnp.ndarray, bool]:
  """Executes Tokamax GMM V2 backend for DLHS = DLHS_dout @ RHS^T."""
  is_qarray_dout = isinstance(dlhs_dout, qpl.QArray)
  dlhs_lhs = dlhs_dout.qvalue if is_qarray_dout else dlhs_dout

  if use_gmm_v2_heuristic_tiling:
    dlhs_tiling = gmm_v2.calculate_tiling
  else:
    dlhs_tiling = gmm_v2.TileSizes(tile_m=tiling[3], tile_k=tiling[4], tile_n=tiling[5])

  if use_dlhs_transpose_rhs:
    kernel_transpose_rhs = not transpose_rhs
    dlhs_rhs = rhs
  else:
    kernel_transpose_rhs = False
    dlhs_rhs = rhs if transpose_rhs else rhs.swapaxes(1, 2)
  # Sub-byte packing only supports the standard [group, k, n] RHS layout.
  if kernel_transpose_rhs and jax.dtypes.itemsize_bits(rhs.dtype) < 8:
    dlhs_rhs = rhs.swapaxes(1, 2)
    kernel_transpose_rhs = False

  # The kernel adds partial_sum after applying its lhs_scale, so partial_sum can only go into the kernel
  # when every scale on dlhs (dlhs_dout.scale and, with partial_sum, the pre-quantized lhs scale) is
  # folded into that per-tensor lhs_scale. Otherwise scale and accumulate after the kernel.
  scales = [s for s in (dlhs_dout.scale if is_qarray_dout else None, lhs_scale) if s is not None]
  can_fuse_scale = (fuse_dlhs_scale or partial_sum is not None) and all(s.size == 1 for s in scales)
  kernel_lhs_scale = None
  kernel_partial_sum = None
  lhs_scale_applied = False
  if can_fuse_scale:
    for scale in scales:
      scale = scale.astype(jnp.float32).reshape(1, 1)
      kernel_lhs_scale = scale if kernel_lhs_scale is None else kernel_lhs_scale * scale
    lhs_scale_applied = lhs_scale is not None
    kernel_partial_sum = partial_sum

  dlhs = gmm_v2.gmm_v2(
      lhs=dlhs_lhs,
      rhs=dlhs_rhs,
      group_sizes=group_sizes,
      # rhs scale is already applied to dlhs_lhs
      rhs_scale=None,
      partial_sum=kernel_partial_sum,
      tile_info=dlhs_tiling,
      preferred_element_type=lhs_dtype,
      group_offset=group_offset,
      lhs_scale=kernel_lhs_scale,
      transpose_rhs=kernel_transpose_rhs,
      # Bypass internal quantization if incoming dlhs_dout is already a QArray, or when lhs_scale is a
      # post-matmul scale rather than a quantization scale.
      maybe_quantize_lhs=not is_qarray_dout and kernel_lhs_scale is None,
      zero_initialize=not (fuse_3d_dlhs_ps and out_is_3d and kernel_partial_sum is not None),
      out_is_3d=out_is_3d,
  )

  if not can_fuse_scale:
    # Rescale dlhs by dlhs_dout.scale when incoming gradient was pre-quantized QArray.
    if is_qarray_dout:
      dlhs = dlhs * _scale_like(dlhs_dout.scale, dlhs, dlhs.dtype)
    if partial_sum is not None:
      if lhs_scale is not None:
        dlhs = dlhs * _scale_like(lhs_scale, dlhs, dlhs.dtype)
        lhs_scale_applied = True
      dlhs = dlhs + partial_sum

  return dlhs, lhs_scale_applied


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
    partial_sum: jnp.ndarray | None = None,
    flat_rhs_shard: bool = False,
) -> jnp.ndarray:
  """Routes execution of DRHS based on backend choices."""
  if use_tokamax_backend and not use_gmm_v2:
    drhs = _drhs_run_tokamax_v1(drhs_dout, lhs, group_sizes, rhs_dtype, use_manual_quantization)
  elif use_tokamax_backend and use_gmm_v2:
    drhs = _drhs_run_tokamax_v2(
        drhs_dout,
        lhs,
        group_sizes,
        group_offset,
        num_actual_groups,
        rhs_dtype,
        tiling,
        use_gmm_v2_heuristic_tiling,
        partial_sum=partial_sum,
    )
  else:
    drhs = _drhs_run_megablox(
        drhs_dout, lhs, group_sizes, group_offset, num_actual_groups, rhs_dtype, tiling, interpret, rhs_vma_axes
    )

  if use_tokamax_backend and quantization_rule and quantization_rule.bwd_qtype and weight_gather_axes:
    drhs = _drhs_scatter_weight(drhs, weight_gather_axes, flat_rhs_shard=flat_rhs_shard)

  return drhs


def _drhs_scatter_weight(
    drhs: jnp.ndarray,
    weight_gather_axes: List[Tuple[str, int]],
    flat_rhs_shard: bool = False,
) -> jnp.ndarray:
  """Scatters the DRHS output back in the reverse order of the forward gather.

  With `flat_rhs_shard` (moe_flat_fsdp_weights) the (G, rows, cols) grad is first
  bitcast to the flat [G * rows, cols] matrix so the reduce-scatter runs on the major dim.
  """
  if flat_rhs_shard:
    drhs = drhs.reshape(-1, drhs.shape[-1])
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
  size_n = drhs_dout.qvalue.size // drhs_dout.qvalue.shape[0]
  if drhs_dout.qvalue.ndim == 3 or scale.size == 1:
    # 3D grad [m, n // 128, 128]: scale is (1, 1, 1) per-tensor or (1, n // 128, 128) per-channel.
    return jnp.broadcast_to(scale.reshape(1, 1, -1), (1, 1, size_n))
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
    partial_sum: jnp.ndarray | None = None,
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
      partial_sum=partial_sum,
      precision=jax.lax.Precision.DEFAULT,
      preferred_element_type=rhs_dtype,
      group_offset=group_offset,
      tile_info=drhs_tiling,
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


# ==============================================================================
# GMM + SwiGLU: the activation in the gmm_v2 epilogue
# ==============================================================================


def gmm_silu(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    group_offset: jnp.ndarray | int | None = None,
    use_gmm_v2_heuristic_tiling: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    fuse_dlhs_scale: bool = False,
    fuse_3d_dlhs_ps: bool = False,
    fuse_bwd_dswiglu: bool = False,
    fwd_rhs_buffer_count: int | None = None,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
  """silu(lhs @ gate) * (lhs @ up) per group, with the activation in the gmm_v2 epilogue.

  `rhs` is the prefused [g, k, 2n] weight, gate = rhs[..., :n] and up = rhs[..., n:]. The unfused path writes
  the [m, 2n] GMM output and runs the activation over every row in a separate elementwise pass; here the kernel
  applies it to the f32 accumulator of the rows it computes and writes [m, n].

  Numerics: the activation reads the f32 accumulator instead of its bf16 rounding, so `act` differs from the
  unfused value by rounding (it is the more precise of the two). The backward is the unfused one: it takes
  gate and up rounded exactly as the unfused GMM rounds its output (the kernel writes them next to act when
  differentiated), the SwiGLU VJP of `jax.nn.silu(gate) * up`, and the unchanged tokamax gmm_v2 DLHS / TGMM
  DRHS of `gmm`.

  Only the bf16 tokamax gmm_v2 path: no quantization, no partial_sum, no return_lhs / return_rhs, 2D lhs and
  output, rhs not transposed. tiling is gmm's 9-tuple; the fused kernel's tile_n counts the columns of each
  half, so tiling[2] is halved and every grid step does the unfused kernel's MXU work in the same VMEM.

  fuse_bwd_dswiglu (moe_wo_gmm_fused_dswiglu): the SwiGLU backward runs in the epilogue of the next GMM's DLHS
  (`gmm_dswiglu`), so this op returns `(act, gate_up)`: gate_up = [gate | up] as ONE [m, 2n] array (what that
  epilogue reads; written by the same kernel when differentiated), rounded as the unfused GMM rounds its output.
  The backward takes gate_up's cotangent, which is already d[gate | up], straight to gmm's DLHS / TGMM; act's
  cotangent is NOT used, so the caller must stop act's gradient and route it through gate_up. The kernel writes
  gate_up one [gate_j | up_j] block per forward n tile: with more than one tile per half it is interleaved by
  tile, and its reader must pass gmm_silu_pre_act_tile_n(tiling, 2n) as gmm_dswiglu's dact_pre_act_tile_n.

  fwd_rhs_buffer_count (wi_tile_fwd_rhs_buffer_count): the forward kernel's rhs pipeline depth (gmm_v2's
  rhs_buffer_count; None: its default, 3). Buffering only: the values are unchanged, and the backward kernels keep
  their default.
  """
  if isinstance(lhs, qpl.QArray) or isinstance(rhs, qpl.QArray) or lhs.ndim != 2 or rhs.ndim != 3:
    raise NotImplementedError("gmm_silu: plain 2D lhs and [g, k, 2n] rhs only.")
  fn = _gmm_silu_dswiglu if fuse_bwd_dswiglu else _gmm_silu
  return fn(
      lhs,
      rhs,
      group_sizes,
      group_offset,
      preferred_element_type,
      tuple(tiling),
      use_gmm_v2_heuristic_tiling,
      use_dlhs_transpose_rhs,
      fuse_dlhs_scale,
      fuse_3d_dlhs_ps,
      fwd_rhs_buffer_count,
  )


def gmm_silu_pre_act_tile_n(tiling: tuple[int, ...], size_n: int, heuristic: bool = False) -> int:
  """The fused forward's tile_n per half for gmm's 9-tuple `tiling` and a [g, k, size_n] prefused rhs.

  That is the width of the [gate_j | up_j] pieces of gmm_silu(fuse_bwd_dswiglu=True)'s gate_up: the
  dact_pre_act_tile_n its reader (gmm_dswiglu, gmm_dswiglu_bwd) must pass. gmm_silu itself tiles with it.
  """
  if heuristic:
    # gmm_v2.calculate_tiling picks the forward's tile inside the kernel from shapes this caller does not have.
    raise NotImplementedError(
        "gmm_silu's [gate | up] layout is not known statically with use_gmm_v2_heuristic_tiling: use explicit tiles."
    )
  half_n = size_n // 2
  tile_n = min(tiling[2], size_n) // 2
  if tiling[2] % 2 or (tile_n % 128 and tile_n != half_n):
    raise ValueError(f"gmm_silu: fwd tile_n {tiling[2]} must halve to a multiple of 128 (or to n={half_n}).")
  return tile_n


def _gmm_silu_kernel(
    lhs, rhs, group_sizes, group_offset, preferred_element_type, tiling, heuristic, return_pre_act, rhs_buffer_count
):
  """The fused forward: act, or (act, gate, up) with return_pre_act, or (act, [gate | up]) with "concat"."""
  if heuristic:
    tile_info = gmm_v2.calculate_tiling
  else:
    tile_n = gmm_silu_pre_act_tile_n(tiling, rhs.shape[-1])
    tile_info = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tile_n)
  # Passed only when set, so that the default call (and its kernel) is exactly the one without the option.
  buffering = {} if rhs_buffer_count is None else {"rhs_buffer_count": rhs_buffer_count}
  return gmm_v2.gmm_v2(
      lhs=lhs,
      rhs=rhs,
      group_sizes=group_sizes,
      tile_info=tile_info,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
      fuse_act="silu",
      return_pre_act=return_pre_act,
      **buffering,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def _gmm_silu(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
):
  """Runs the fused SiLU grouped matmul with custom VJP."""
  # Not differentiated: the activation alone, nothing else is written.
  del dlhs_transpose, fuse_scale, fuse_3d
  return _gmm_silu_kernel(
      lhs, rhs, group_sizes, group_offset, preferred_element_type, tiling, heuristic, False, fwd_rhs_buffer_count
  )


def _gmm_silu_fwd(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
):
  """Differentiated: the kernel also writes gate and up, the residuals of the unfused backward."""
  del dlhs_transpose, fuse_scale, fuse_3d
  act, gate, up = _gmm_silu_kernel(
      lhs, rhs, group_sizes, group_offset, preferred_element_type, tiling, heuristic, True, fwd_rhs_buffer_count
  )
  return act, (lhs, rhs, group_sizes, group_offset, gate, up)


def _swiglu(gate, up):
  """MaxText's MoE activation on the GMM output (`apply_ffn_activation`, mlp_activations silu + linear)."""
  return (jax.nn.silu(gate) * up).astype(gate.dtype)


def _gmm_silu_bwd(
    preferred_element_type, tiling, heuristic, dlhs_transpose, fuse_scale, fuse_3d, fwd_rhs_buffer_count, residual, dact
):
  """The unfused backward from the saved bf16 gate and up: the SwiGLU VJP, then gmm's DLHS / TGMM."""
  del fwd_rhs_buffer_count  # the forward kernel's buffering only
  lhs, rhs, group_sizes, group_offset, gate, up = residual
  # The unfused backward: the SwiGLU VJP on the bf16 GMM output, then gmm's own backward on [d gate | d up].
  _, swiglu_vjp = jax.vjp(_swiglu, gate, up)
  dgate, dup = swiglu_vjp(dact.astype(gate.dtype))
  grad = jnp.concatenate([dgate, dup], axis=-1)
  # gmm's own backward, as gmm's custom VJP calls it for a plain bf16 tokamax gmm_v2 GMM (no quantization, no
  # partial sum, no returned operands), with this op's DLHS options.
  dlhs, drhs, *_ = _gmm_bwd(
      lhs_dtype=lhs.dtype,
      rhs_dtype=rhs.dtype,
      preferred_element_type=preferred_element_type,
      tiling=tiling,
      transpose_rhs=False,
      interpret=False,  # the megablox backend only
      quantization_rule=None,
      use_tokamax_backend=True,
      weight_gather_axes=None,
      use_manual_quantization=False,
      lhs_vma_axes=(),
      rhs_vma_axes=(),
      use_gmm_v2=True,
      use_gmm_v2_heuristic_tiling=heuristic,
      return_rhs=False,
      return_lhs=False,
      fuse_dlhs_scale=fuse_scale,
      use_dlhs_transpose_rhs=dlhs_transpose,
      flat_fsdp_shard=None,
      out_is_3d=False,
      fuse_3d_dlhs_ps=fuse_3d,
      tc_unsort_cfg=None,
      # (lhs, rhs, group_sizes, group_offset, partial_sum_fwd, lhs_is_qarray, rhs_is_qarray, tc_routing)
      residual=(lhs, rhs, group_sizes, group_offset, None, False, False, None),
      # (d out, d returned lhs, d returned rhs)
      grad=(grad, None, None),
  )
  return dlhs, drhs, None, None


_gmm_silu.defvjp(_gmm_silu_fwd, _gmm_silu_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def _gmm_silu_dswiglu(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
):
  """gmm_silu(fuse_bwd_dswiglu=True) when not differentiated: the activation alone.

  Without a backward pass nothing reads gate_up (gmm_dswiglu and ring_ragged_unsort_down_projection read it in
  their backward only), so a zero placeholder stands in for it and XLA removes it.
  """
  del dlhs_transpose, fuse_scale, fuse_3d
  act = _gmm_silu_kernel(
      lhs, rhs, group_sizes, group_offset, preferred_element_type, tiling, heuristic, False, fwd_rhs_buffer_count
  )
  return act, jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=preferred_element_type)


def _gmm_silu_dswiglu_fwd(
    lhs,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
):
  """Differentiated: the kernel also writes gate_up = [gate | up] (one array), the input of the fused dSwiGLU."""
  del dlhs_transpose, fuse_scale, fuse_3d
  act, gate_up = _gmm_silu_kernel(
      lhs, rhs, group_sizes, group_offset, preferred_element_type, tiling, heuristic, "concat", fwd_rhs_buffer_count
  )
  return (act, gate_up), (lhs, rhs, group_sizes, group_offset)


def _gmm_silu_dswiglu_bwd(
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
    residual,
    grad_out,
):
  """gmm's DLHS / TGMM on gate_up's cotangent, which the next GMM's fused epilogue already made d[gate | up].

  act's cotangent is not read: the caller stops act's gradient (see gmm_silu), so it is zero.
  """
  del fwd_rhs_buffer_count  # the forward kernel's buffering only
  lhs, rhs, group_sizes, group_offset = residual
  _, d_gate_up = grad_out
  grad = d_gate_up.astype(preferred_element_type)
  dlhs, drhs, *_ = _gmm_bwd(
      lhs.dtype,
      rhs.dtype,
      preferred_element_type,
      tiling,
      False,  # transpose_rhs
      False,  # interpret (megablox backend only)
      None,  # quantization_rule
      True,  # use_tokamax_backend
      None,  # weight_gather_axes
      False,  # use_manual_quantization
      (),  # lhs_vma_axes
      (),  # rhs_vma_axes
      True,  # use_gmm_v2
      heuristic,
      False,  # return_rhs
      False,  # return_lhs
      fuse_scale,
      dlhs_transpose,
      None,  # flat_fsdp_shard
      False,  # out_is_3d
      fuse_3d,
      None,  # tc_unsort_cfg
      (lhs, rhs, group_sizes, group_offset, None, False, False, None),
      (grad, None, None),
  )
  return dlhs, drhs, None, None


_gmm_silu_dswiglu.defvjp(_gmm_silu_dswiglu_fwd, _gmm_silu_dswiglu_bwd)


# ==============================================================================
# GMM of a SwiGLU output: the SwiGLU backward in the DLHS epilogue
# ==============================================================================


def gmm_dswiglu_bwd(
    dout: jnp.ndarray,
    lhs_act: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | int | None,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    dact_gate_up: jnp.ndarray,
    *,
    dact_row_scale: jnp.ndarray | None = None,
    return_dact_dot: bool = False,
    use_gmm_v2_heuristic_tiling: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    dact_pre_act_tile_n: int | None = None,
    rhs_fp8_scale: jnp.ndarray | None = None,
    rhs_dtype: jnp.dtype | None = None,
):
  """The backward of `lhs_act @ rhs` (per group), lhs_act = silu(gate) * up, with the SwiGLU backward fused.

  The DLHS is gmm's tokamax gmm_v2 DLHS (u = dout @ rhs^T, same tiles, rhs transpose and zero init) with
  fuse_dact="swiglu": its epilogue turns the float32 u of each row into d[gate | up] (apply_dact_fn), reading
  dact_gate_up = [gate | up] (the [m, 2n] input of the activation), so u and d act are never written. The DRHS is
  gmm's TGMM, lhs_act^T @ dout. dact_row_scale (optional, [m]) multiplies u per row before the derivative;
  return_dact_dot also returns <u, act> per row in float32. Needs tiling[5] (the DLHS tile_n) == n.
  dact_pre_act_tile_n: None when dact_gate_up is the contiguous [gate | up] (an unfused GMM output); for the
  gate_up of gmm_silu(fuse_bwd_dswiglu=True), gmm_silu_pre_act_tile_n(its tiling, 2n): the epilogue then reads
  it in that kernel's per-tile [gate_j | up_j] layout. d[gate | up] is contiguous either way.

  Only the tokamax gmm_v2 path, as gmm_silu: no quantized activations, partial sums or explicit weight gathers. The
  weight is bf16, or FP8 codes with rhs_fp8_scale:

  rhs_fp8_scale (fp8_moe_native_bwd_dlhs): rhs is float8_e4m3fn codes with this per-channel [g, 1, n] scale, i.e.
  the weight dequantize_fp8_codes(rhs, rhs_fp8_scale, rhs_dtype). With use_dlhs_transpose_rhs and fixed tiles the
  DLHS kernel dequantizes it in VMEM (_dlhs_fp8_native_kwargs); otherwise it is dequantized here first. The results
  are bit-identical to passing the dequantized weight; drhs is in rhs_dtype.

  Returns:
    (d_gate_up, dot, drhs); dot is None without return_dact_dot.
  """
  native_dlhs = {}
  if rhs_fp8_scale is not None:
    rhs_dtype = jnp.dtype(preferred_element_type if rhs_dtype is None else rhs_dtype)
    if _use_fp8_native_dlhs(use_dlhs_transpose_rhs, use_gmm_v2_heuristic_tiling, tiling, rhs.shape):
      native_dlhs = _dlhs_fp8_native_kwargs(rhs_fp8_scale, rhs_dtype)
    else:
      rhs = dequantize_fp8_codes(rhs, rhs_fp8_scale, rhs_dtype)
  drhs_dtype = rhs_dtype if rhs_fp8_scale is not None else rhs.dtype
  if use_gmm_v2_heuristic_tiling:
    dlhs_tiling = gmm_v2.calculate_tiling
  else:
    dlhs_tiling = gmm_v2.TileSizes(tile_m=tiling[3], tile_k=tiling[4], tile_n=tiling[5])
  # As _dlhs_run_tokamax_v2 with transpose_rhs=False.
  if use_dlhs_transpose_rhs:
    kernel_transpose_rhs = True
    dlhs_rhs = rhs
  else:
    kernel_transpose_rhs = False
    dlhs_rhs = rhs.swapaxes(1, 2)
  dlhs_out = gmm_v2.gmm_v2(
      lhs=dout,
      rhs=dlhs_rhs,
      group_sizes=group_sizes,
      rhs_scale=None,
      partial_sum=None,
      tile_info=dlhs_tiling,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
      lhs_scale=None,
      transpose_rhs=kernel_transpose_rhs,
      maybe_quantize_lhs=True,
      zero_initialize=True,
      out_is_3d=False,
      fuse_dact="swiglu",
      dact_gate_up=dact_gate_up,
      dact_row_scale=dact_row_scale,
      return_dact_dot=return_dact_dot,
      dact_pre_act_tile_n=dact_pre_act_tile_n,
      **native_dlhs,
  )
  drhs = _drhs_run_tokamax_v2(
      dout,
      lhs_act,
      group_sizes,
      group_offset,
      rhs.shape[0],
      drhs_dtype,
      tiling,
      use_gmm_v2_heuristic_tiling,
      partial_sum=None,
  )
  d_gate_up, dot = dlhs_out if return_dact_dot else (dlhs_out, None)
  return d_gate_up, dot, drhs


def gmm_dswiglu(
    lhs_act: jnp.ndarray,
    gate_up: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    group_offset: jnp.ndarray | int | None = None,
    use_gmm_v2_heuristic_tiling: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    dact_pre_act_tile_n: int | None = None,
) -> jnp.ndarray:
  """`lhs_act @ rhs` per group (gmm's tokamax gmm_v2 forward) whose backward returns d gate_up, not d lhs_act.

  lhs_act must be silu(gate) * up of gate_up ([m, 2n]). The backward (gmm_dswiglu_bwd) fuses the SwiGLU backward
  into the DLHS epilogue: it returns gate_up's cotangent d[gate | up] and a ZERO cotangent for lhs_act, so the
  caller must stop lhs_act's gradient (its gradient goes through gate_up). The forward is gmm's. gate_up is the
  contiguous [gate | up] (dact_pre_act_tile_n None), or gmm_silu(fuse_bwd_dswiglu=True)'s, read in its per-tile
  layout with dact_pre_act_tile_n = gmm_silu_pre_act_tile_n (see gmm_dswiglu_bwd); its cotangent is contiguous.
  """
  return _gmm_dswiglu(
      lhs_act,
      gate_up,
      rhs,
      group_sizes,
      group_offset,
      preferred_element_type,
      tuple(tiling),
      use_gmm_v2_heuristic_tiling,
      use_dlhs_transpose_rhs,
      dact_pre_act_tile_n,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _gmm_dswiglu(
    lhs_act,
    gate_up,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    pre_act_tile_n,
):
  """gmm_dswiglu when not differentiated: gmm's forward (gate_up is only read by the backward)."""
  del gate_up, dlhs_transpose, pre_act_tile_n
  return _fwd_run_tokamax_v2(
      lhs_act, rhs, group_sizes, preferred_element_type, tiling, heuristic, group_offset, None, False
  )


def _gmm_dswiglu_fwd(
    lhs_act,
    gate_up,
    rhs,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    pre_act_tile_n,
):
  """Forward rule of _gmm_dswiglu: gmm's forward, keeping the inputs (not the output) for the backward."""
  del dlhs_transpose, pre_act_tile_n
  out = _fwd_run_tokamax_v2(
      lhs_act, rhs, group_sizes, preferred_element_type, tiling, heuristic, group_offset, None, False
  )
  return out, (lhs_act, gate_up, rhs, group_sizes, group_offset)


def _gmm_dswiglu_bwd(preferred_element_type, tiling, heuristic, dlhs_transpose, pre_act_tile_n, residual, grad):
  """gmm_dswiglu_bwd at the saved activation and [gate | up]: (0 for lhs_act, d[gate | up], d rhs)."""
  lhs_act, gate_up, rhs, group_sizes, group_offset = residual
  d_gate_up, _, drhs = gmm_dswiglu_bwd(
      grad,
      lhs_act,
      rhs,
      group_sizes,
      group_offset,
      preferred_element_type,
      tiling,
      gate_up,
      use_gmm_v2_heuristic_tiling=heuristic,
      use_dlhs_transpose_rhs=dlhs_transpose,
      dact_pre_act_tile_n=pre_act_tile_n,
  )
  # lhs_act's gradient is carried by gate_up (see gmm_dswiglu).
  return jnp.zeros_like(lhs_act), d_gate_up, drhs, None, None


_gmm_dswiglu.defvjp(_gmm_dswiglu_fwd, _gmm_dswiglu_bwd)


# ==============================================================================
# fp8_moe_native_fwd: native FP8 forward for the routed experts
# ==============================================================================
#
# With fp8_moe_fake_quant + fp8_fake_quant_hoist=gather8 the trainer gathers each routed-expert weight as
# float8_e4m3fn codes with a per-(expert, output channel) float32 scale, dequantizes them to bf16 and runs the
# GMMs in bf16. The ops below take the codes and the scale instead and run the forward GMMs on the MXU in FP8, as
# the FP8 rollout does: the weight stays in FP8 (its scale is applied to the block accumulator, whose dtype is
# acc_dtype) and the bf16 activation is quantized to float8_e4m3fn inside the kernel with a dynamic per-row absmax
# scale (per 512-column block by default; lhs_quant_mode selects other schemes).
#
# The backward runs the same bf16 DLHS / TGMM as gmm_silu(fuse_bwd_dswiglu=True) and gmm_dswiglu, at the weight
# dequantized with dequantize_weight_codes_ste's expression, (codes.astype(float32) * scale).astype(bf16), which is
# bit-identical to the bf16 weight the bf16 path multiplies by. With native_bwd_dlhs the DLHS dequantizes the codes
# inside the kernel instead (see below). Only the codes and the scale are saved for the backward (1 byte per weight
# element instead of 2).
#
# `rhs` is the gradient anchor: the bf16 weight the bf16 path would multiply by (in the trainer, the output of
# dequantize_weight_codes_ste). Its value is not read, so XLA removes the forward dequantization; the weight
# gradient is returned as its cotangent, and dequantize_weight_codes_ste carries it to the master weight
# (straight-through, as today). `rhs_codes` and `rhs_scale` get no gradient.


def _check_per_channel_scale(scale: jnp.ndarray, name: str):
  """Raises unless `scale` is a per-(expert, output channel) [g, 1, n] weight scale."""
  if scale.ndim != 3 or scale.shape[1] != 1:
    raise ValueError(f"{name}: expected a per-channel [g, 1, n] scale, got {scale.shape}.")


def _fp8_rhs_scale(scale: jnp.ndarray) -> jnp.ndarray:
  """quantize_weight_codes_per_channel's [g, 1, n] scale as gmm_v2's rhs_scale [g, 1, 1, n] (one K block)."""
  _check_per_channel_scale(scale, "fp8_moe_native_fwd")
  return scale.astype(jnp.float32)[:, :, None, :]


def dequantize_fp8_codes(codes: jnp.ndarray, scale: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
  """The weight of `codes` in `dtype`: dequantize_weight_codes_ste's forward expression (also used in the kernel)."""
  return gmm_v2.dequantize_fp8_codes(codes, scale, dtype)


# fp8_moe_native_bwd_dlhs: the backward DLHS (dout @ W^T) reads the codes and dequantizes them inside gmm_v2
# (rhs_contract_scale): per weight block, in VMEM, with dequantize_fp8_codes' expression, cached across the m tiles
# of an expert. The matmul is the same bf16 one, so the DLHS is bit-identical to the one at the dequantized weight,
# without the XLA dequantization of every expert weight (and its bf16 copy in HBM). The TGMM (weight gradient) does
# not read the weight and is unchanged. Only with the DLHS rhs transpose (the codes' [g, k, n] layout is the
# kernel's transposed [g, size_n, size_k] rhs, the scale varying along its contraction dim) and fixed tiles (the
# heuristic tiling would size its tiles for an fp8 rhs).


def _use_fp8_native_dlhs(dlhs_transpose: bool, heuristic: bool, tiling, rhs_shape) -> bool:
  """Whether the DLHS can dequantize the [g, size_n, size_k] codes in the kernel; otherwise they are dequantized first.

  Besides the rhs transpose and fixed tiles, gmm_v2's rhs_contract_scale needs whole [tile_n, tile_k] weight blocks
  with tile_n % 32 == 0 and tile_k % 128 == 0 (tiling[4], tiling[5] are the DLHS tile_k, tile_n). Called only when
  fp8_moe_native_bwd_dlhs is requested, so a fallback is logged.
  """
  if not dlhs_transpose or heuristic:
    supported = False
  else:
    tile_k, tile_n = tiling[4], tiling[5]
    size_n, size_k = rhs_shape[1], rhs_shape[2]
    supported = tile_n % 32 == 0 and tile_k % 128 == 0 and size_n % tile_n == 0 and size_k % tile_k == 0
  if not supported:
    logging.warning(
        "fp8_moe_native_bwd_dlhs: the DLHS falls back to an XLA weight dequantization (dlhs_transpose=%s,"
        " heuristic_tiling=%s, tiling=%s, rhs_shape=%s); it needs the DLHS rhs transpose, fixed tiles and whole"
        " [tile_n, tile_k] blocks with tile_n %% 32 == 0 and tile_k %% 128 == 0.",
        dlhs_transpose,
        heuristic,
        tiling,
        rhs_shape,
    )
  return supported


def _dlhs_fp8_native_kwargs(rhs_scale: jnp.ndarray, rhs_dtype) -> dict:
  """gmm_v2 kwargs of a transpose_rhs DLHS whose rhs is the codes of the per-channel [g, 1, n] rhs_scale."""
  _check_per_channel_scale(rhs_scale, "fp8_moe_native_bwd_dlhs")
  return {"rhs_contract_scale": rhs_scale, "rhs_dequant_dtype": jnp.dtype(rhs_dtype)}


def _check_fp8_rhs(rhs, rhs_codes, rhs_scale, name):
  """Raises unless rhs_codes are float8_e4m3fn of the anchor's shape with a matching [g, 1, n] scale."""
  if rhs.shape != rhs_codes.shape:
    raise ValueError(f"{name}: anchor {rhs.shape} and codes {rhs_codes.shape} must have the same shape.")
  if rhs_codes.dtype != jnp.float8_e4m3fn:
    raise ValueError(f"{name}: rhs_codes must be float8_e4m3fn, got {rhs_codes.dtype}.")
  if rhs_scale.shape != (rhs_codes.shape[0], 1, rhs_codes.shape[2]):
    raise ValueError(f"{name}: rhs_scale must be [g, 1, n] = {(rhs_codes.shape[0], 1, rhs_codes.shape[2])}.")


def _acc_kwargs(acc_dtype):
  """gmm_v2's acc_dtype kwarg, passed only when set: None keeps its default (bf16 when lhs is quantized)."""
  return {} if acc_dtype is None else {"acc_dtype": jnp.dtype(acc_dtype)}


_LHS_QUANT_MODES: dict[str, dict] = {
    "": {},
    "block512_bf16": {},
    "block512_bf16_rhs1": {"rhs_scale_first": True},
    "block512_f32": {"lhs_f32_scale": True},
    "block512_f32_hoist": {"lhs_f32_scale": True, "lhs_hoist_quant": True},
    "fullk_bf16": {"lhs_quant_block_size": 0},
    "fullk_bf16_hoist": {"lhs_quant_block_size": 0, "lhs_hoist_quant": True},
    "fullk_f32": {"lhs_quant_block_size": 0, "lhs_f32_scale": True},
    "fullk_f32_hoist": {"lhs_quant_block_size": 0, "lhs_f32_scale": True, "lhs_hoist_quant": True},
}


def _lhs_quant_kwargs(lhs_quant_mode: str) -> dict:
  """gmm_v2's in-kernel lhs quantization kwargs for a fp8_moe_native_fwd_lhs_quant mode."""
  if lhs_quant_mode not in _LHS_QUANT_MODES:
    raise ValueError(f"Unknown lhs_quant_mode {lhs_quant_mode!r}; expected one of {list(_LHS_QUANT_MODES)}.")
  return _LHS_QUANT_MODES[lhs_quant_mode]


def gmm_silu_fp8(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    rhs_codes: jnp.ndarray,
    rhs_scale: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    group_offset: jnp.ndarray | int | None = None,
    use_gmm_v2_heuristic_tiling: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    fuse_dlhs_scale: bool = False,
    fuse_3d_dlhs_ps: bool = False,
    fwd_rhs_buffer_count: int | None = None,
    acc_dtype: jnp.dtype | None = None,
    lhs_quant_mode: str = "",
    native_bwd_dlhs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """gmm_silu(fuse_bwd_dswiglu=True) with an FP8 forward: returns (act, gate_up).

  rhs: the bf16 [g, k, 2n] gradient anchor (not read); rhs_codes: its float8_e4m3fn codes; rhs_scale: [g, 1, 2n].
  Same contract as gmm_silu(fuse_bwd_dswiglu=True): the caller stops act's gradient and routes it through
  gate_up.

  acc_dtype: gmm_v2's accumulator across K blocks (None: the kernel default). lhs_quant_mode: the in-kernel lhs
  quantization scheme, a key of _LHS_QUANT_MODES (fp8_moe_native_fwd_lhs_quant). native_bwd_dlhs
  (fp8_moe_native_bwd_dlhs): the backward DLHS dequantizes the codes inside the kernel (same values). The other
  arguments are gmm_silu's.
  """
  if isinstance(lhs, qpl.QArray) or lhs.ndim != 2 or rhs.ndim != 3:
    raise NotImplementedError("gmm_silu_fp8: plain 2D lhs and [g, k, 2n] rhs only.")
  _check_fp8_rhs(rhs, rhs_codes, rhs_scale, "gmm_silu_fp8")
  return _gmm_silu_fp8(
      lhs,
      rhs,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tuple(tiling),
      use_gmm_v2_heuristic_tiling,
      use_dlhs_transpose_rhs,
      fuse_dlhs_scale,
      fuse_3d_dlhs_ps,
      fwd_rhs_buffer_count,
      None if acc_dtype is None else jnp.dtype(acc_dtype),
      jnp.dtype(rhs.dtype),
      lhs_quant_mode,
      bool(native_bwd_dlhs),
  )


def _gmm_silu_fp8_kernel(
    lhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    return_pre_act,
    rhs_buffer_count,
    acc_dtype,
    lhs_quant_mode="",
):
  """_gmm_silu_kernel with FP8 weights (codes + per-channel scale) and in-kernel FP8 lhs quantization."""
  if heuristic:
    tile_info = gmm_v2.calculate_tiling
  else:
    tile_n = gmm_silu_pre_act_tile_n(tiling, rhs_codes.shape[-1])
    tile_info = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tile_n)
  buffering = {} if rhs_buffer_count is None else {"rhs_buffer_count": rhs_buffer_count}
  return gmm_v2.gmm_v2(
      lhs=lhs,
      rhs=rhs_codes,
      group_sizes=group_sizes,
      rhs_scale=_fp8_rhs_scale(rhs_scale),
      tile_info=tile_info,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
      maybe_quantize_lhs=True,
      fuse_act="silu",
      return_pre_act=return_pre_act,
      **buffering,
      **_acc_kwargs(acc_dtype),
      **_lhs_quant_kwargs(lhs_quant_mode),
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))
def _gmm_silu_fp8(
    lhs,
    rhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
):
  """gmm_silu_fp8 when not differentiated: the activation, and a zero gate_up placeholder XLA removes."""
  del rhs, dlhs_transpose, fuse_scale, fuse_3d, rhs_dtype, native_bwd_dlhs
  act = _gmm_silu_fp8_kernel(
      lhs,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tiling,
      heuristic,
      return_pre_act=False,
      rhs_buffer_count=fwd_rhs_buffer_count,
      acc_dtype=acc_dtype,
      lhs_quant_mode=lhs_quant_mode,
  )
  return act, jnp.zeros((lhs.shape[0], rhs_codes.shape[-1]), dtype=preferred_element_type)


def _gmm_silu_fp8_fwd(
    lhs,
    rhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
):
  """Differentiated: (act, gate_up) from the FP8 kernel; saves the codes and scale, not the bf16 weight."""
  del rhs, dlhs_transpose, fuse_scale, fuse_3d, rhs_dtype, native_bwd_dlhs
  act, gate_up = _gmm_silu_fp8_kernel(
      lhs,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tiling,
      heuristic,
      return_pre_act="concat",
      rhs_buffer_count=fwd_rhs_buffer_count,
      acc_dtype=acc_dtype,
      lhs_quant_mode=lhs_quant_mode,
  )
  return (act, gate_up), (lhs, rhs_codes, rhs_scale, group_sizes, group_offset)


def _gmm_silu_fp8_bwd(
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    fuse_scale,
    fuse_3d,
    fwd_rhs_buffer_count,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
    residual,
    grad_out,
):
  """_gmm_silu_dswiglu_bwd at the dequantized bf16 weight; the weight gradient goes to the anchor."""
  del acc_dtype, lhs_quant_mode
  lhs, rhs_codes, rhs_scale, group_sizes, group_offset = residual
  if native_bwd_dlhs and _use_fp8_native_dlhs(dlhs_transpose, heuristic, tiling, rhs_codes.shape):
    # fp8_moe_native_bwd_dlhs: _gmm_silu_dswiglu_bwd's _gmm_bwd (no quantization, partial sums or 3D layouts:
    # _dlhs_run_tokamax_v2 and _compute_drhs reduce to these two calls) with the codes dequantized in the DLHS.
    _, d_gate_up = grad_out
    grad = d_gate_up.astype(preferred_element_type)
    dlhs = gmm_v2.gmm_v2(
        lhs=grad,
        rhs=rhs_codes,
        group_sizes=group_sizes,
        rhs_scale=None,
        partial_sum=None,
        tile_info=gmm_v2.TileSizes(tile_m=tiling[3], tile_k=tiling[4], tile_n=tiling[5]),
        preferred_element_type=lhs.dtype,
        group_offset=group_offset,
        lhs_scale=None,
        transpose_rhs=True,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        out_is_3d=False,
        **_dlhs_fp8_native_kwargs(rhs_scale, rhs_dtype),
    )
    drhs = _drhs_run_tokamax_v2(
        grad,
        lhs,
        group_sizes,
        group_offset,
        rhs_codes.shape[0],
        rhs_dtype,
        tiling,
        heuristic,
        partial_sum=None,
    )
    return dlhs, drhs, None, None, None, None
  rhs = dequantize_fp8_codes(rhs_codes, rhs_scale, rhs_dtype)
  dlhs, drhs, _, _ = _gmm_silu_dswiglu_bwd(
      preferred_element_type,
      tiling,
      heuristic,
      dlhs_transpose,
      fuse_scale,
      fuse_3d,
      fwd_rhs_buffer_count,
      (lhs, rhs, group_sizes, group_offset),
      grad_out,
  )
  return dlhs, drhs, None, None, None, None


_gmm_silu_fp8.defvjp(_gmm_silu_fp8_fwd, _gmm_silu_fp8_bwd)


def gmm_dswiglu_fp8(
    lhs_act: jnp.ndarray,
    gate_up: jnp.ndarray,
    rhs: jnp.ndarray,
    rhs_codes: jnp.ndarray,
    rhs_scale: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    group_offset: jnp.ndarray | int | None = None,
    use_gmm_v2_heuristic_tiling: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    dact_pre_act_tile_n: int | None = None,
    acc_dtype: jnp.dtype | None = None,
    lhs_quant_mode: str = "",
    native_bwd_dlhs: bool = False,
) -> jnp.ndarray:
  """gmm_dswiglu with an FP8 forward. rhs: the bf16 [g, k, n] anchor (not read); rhs_codes / rhs_scale [g, 1, n].

  Same contract as gmm_dswiglu: the backward returns d[gate | up] and a zero cotangent for lhs_act.

  acc_dtype, lhs_quant_mode and native_bwd_dlhs: as in gmm_silu_fp8. The other arguments are gmm_dswiglu's.
  """
  _check_fp8_rhs(rhs, rhs_codes, rhs_scale, "gmm_dswiglu_fp8")
  return _gmm_dswiglu_fp8(
      lhs_act,
      gate_up,
      rhs,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tuple(tiling),
      use_gmm_v2_heuristic_tiling,
      use_dlhs_transpose_rhs,
      dact_pre_act_tile_n,
      None if acc_dtype is None else jnp.dtype(acc_dtype),
      jnp.dtype(rhs.dtype),
      lhs_quant_mode,
      bool(native_bwd_dlhs),
  )


def _gmm_fp8_fwd_kernel(
    lhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    acc_dtype,
    lhs_quant_mode="",
    rhs_buffer_count=None,
):
  """gmm's tokamax gmm_v2 forward with FP8 weights and in-kernel FP8 lhs quantization."""
  if heuristic:
    tile_info = gmm_v2.calculate_tiling
  else:
    tile_info = gmm_v2.TileSizes(tile_m=tiling[0], tile_k=tiling[1], tile_n=tiling[2])
  buffering = {} if rhs_buffer_count is None else {"rhs_buffer_count": rhs_buffer_count}
  return gmm_v2.gmm_v2(
      lhs=lhs,
      rhs=rhs_codes,
      group_sizes=group_sizes,
      rhs_scale=_fp8_rhs_scale(rhs_scale),
      tile_info=tile_info,
      preferred_element_type=preferred_element_type,
      group_offset=group_offset,
      maybe_quantize_lhs=True,
      **buffering,
      **_acc_kwargs(acc_dtype),
      **_lhs_quant_kwargs(lhs_quant_mode),
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15))
def _gmm_dswiglu_fp8(
    lhs_act,
    gate_up,
    rhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    pre_act_tile_n,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
):
  """gmm_dswiglu_fp8 when not differentiated: the FP8 forward."""
  del gate_up, rhs, dlhs_transpose, pre_act_tile_n, rhs_dtype, native_bwd_dlhs
  return _gmm_fp8_fwd_kernel(
      lhs_act,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tiling,
      heuristic,
      acc_dtype,
      lhs_quant_mode,
  )


def _gmm_dswiglu_fp8_fwd(
    lhs_act,
    gate_up,
    rhs,
    rhs_codes,
    rhs_scale,
    group_sizes,
    group_offset,
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    pre_act_tile_n,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
):
  """Differentiated: the FP8 forward; saves the FP8 weight and scale (not a bf16 copy) for the backward."""
  del rhs, dlhs_transpose, pre_act_tile_n, rhs_dtype, native_bwd_dlhs
  out = _gmm_fp8_fwd_kernel(
      lhs_act,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tiling,
      heuristic,
      acc_dtype,
      lhs_quant_mode,
  )
  return out, (lhs_act, gate_up, rhs_codes, rhs_scale, group_sizes, group_offset)


def _gmm_dswiglu_fp8_bwd(
    preferred_element_type,
    tiling,
    heuristic,
    dlhs_transpose,
    pre_act_tile_n,
    acc_dtype,
    rhs_dtype,
    lhs_quant_mode,
    native_bwd_dlhs,
    residual,
    grad,
):
  """_gmm_dswiglu_bwd at the dequantized bf16 weight; the weight gradient goes to the anchor."""
  del acc_dtype, lhs_quant_mode
  lhs_act, gate_up, rhs_codes, rhs_scale, group_sizes, group_offset = residual
  if native_bwd_dlhs:
    # fp8_moe_native_bwd_dlhs: _gmm_dswiglu_bwd with the codes (gmm_dswiglu_bwd dequantizes them in the DLHS).
    d_gate_up, _, drhs = gmm_dswiglu_bwd(
        grad,
        lhs_act,
        rhs_codes,
        group_sizes,
        group_offset,
        preferred_element_type,
        tiling,
        gate_up,
        use_gmm_v2_heuristic_tiling=heuristic,
        use_dlhs_transpose_rhs=dlhs_transpose,
        dact_pre_act_tile_n=pre_act_tile_n,
        rhs_fp8_scale=rhs_scale,
        rhs_dtype=rhs_dtype,
    )
    return jnp.zeros_like(lhs_act), d_gate_up, drhs, None, None, None, None
  rhs = dequantize_fp8_codes(rhs_codes, rhs_scale, rhs_dtype)
  d_lhs_act, d_gate_up, drhs, _, _ = _gmm_dswiglu_bwd(
      preferred_element_type,
      tiling,
      heuristic,
      dlhs_transpose,
      pre_act_tile_n,
      (lhs_act, gate_up, rhs, group_sizes, group_offset),
      grad,
  )
  return d_lhs_act, d_gate_up, drhs, None, None, None, None


_gmm_dswiglu_fp8.defvjp(_gmm_dswiglu_fp8_fwd, _gmm_dswiglu_fp8_bwd)


def gmm_fp8_forward(
    lhs: jnp.ndarray,
    rhs_codes: jnp.ndarray,
    rhs_scale: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    group_offset: jnp.ndarray | int | None = None,
    use_gmm_v2_heuristic_tiling: bool = False,
    acc_dtype: jnp.dtype | None = None,
    lhs_quant_mode: str = "",
    fwd_rhs_buffer_count: int | None = None,
) -> jnp.ndarray:
  """The FP8 forward of gmm (no VJP): for callers that own the backward (ring_ragged_unsort_down_projection).

  rhs_codes: float8_e4m3fn [g, k, n]; rhs_scale: [g, 1, n]. acc_dtype and lhs_quant_mode: as in gmm_silu_fp8;
  fwd_rhs_buffer_count: gmm_v2's rhs_buffer_count. The other arguments are gmm's.
  """
  if rhs_codes.dtype != jnp.float8_e4m3fn:
    raise ValueError(f"gmm_fp8_forward: rhs_codes must be float8_e4m3fn, got {rhs_codes.dtype}.")
  return _gmm_fp8_fwd_kernel(
      lhs,
      rhs_codes,
      rhs_scale,
      group_sizes,
      group_offset,
      preferred_element_type,
      tuple(tiling),
      use_gmm_v2_heuristic_tiling,
      None if acc_dtype is None else jnp.dtype(acc_dtype),
      lhs_quant_mode,
      fwd_rhs_buffer_count,
  )
