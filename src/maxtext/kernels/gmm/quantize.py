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

"""Quantization orchestration, QAG gather/scatter, and custom VJP for grouped matmul.

This module handles quantization policy (Qwix rules, manual quantization,
pre-quantized QArrays), quantization all-gather/reduce-scatter (QAG), and
TensorCore unsort fusion independently of the underlying `GmmBackend`.
"""

# pylint: disable=too-many-positional-arguments

import dataclasses
import functools
from typing import List, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from maxtext.kernels.gmm import backends as gmm_backends
from maxtext.kernels.ragged import ragged_sort_tc
from maxtext.layers import quantizations
import qwix
import qwix.pallas as qpl


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


def resolve_quantization_rule(
    *,
    use_qwix_quantization: bool = False,
    qwix_rule: qwix.QtRule | None = None,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
) -> qwix.QtRule | None:
  """Resolves the active `qwix.QtRule` from interception context or explicit dtypes."""
  if use_qwix_quantization:
    # 1. For non-batchsplit, retrieve rule ("gmm") via qwix interception outside custom_vjp.
    # 2. For batchsplit, use the explicitly provided rule.
    rule = qwix_rule if qwix_rule is not None else qpl.get_current_rule("gmm")
    if rule is not None and not isinstance(rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training.")
    return rule
  if lhs_quantize_dtype is not None or rhs_quantize_dtype is not None:
    return qwix.QtRule(
        weight_qtype=rhs_quantize_dtype,
        weight_calibration_method="absmax",
        act_qtype=lhs_quantize_dtype,
        act_calibration_method="absmax",
    )
  return None


def prepare_lhs_scale(quantization_rule: qwix.QtRule | None) -> jax.Array | None:
  """Extracts the static LHS (activation) scale for backends supporting native LHS quantization.

  If a static scale is used, it must come from a symmetric fixed-range
  calibration (e.g., 'fixed,-max,max' or 'fixed,max'). Returns None for dynamic
  quantization or when no rule/act_qtype is configured.

  Enforces a default (1, 1) shape for per-tensor quantization kernels.
  """
  if quantization_rule is None:
    return None

  method = quantization_rule.act_calibration_method
  qtype = quantization_rule.act_qtype

  if method is None or qtype is None or not method.lower().startswith("fixed"):
    return None

  scale_val = quantizations.get_static_scale(qtype, method)
  return jnp.full((1, 1), scale_val, jnp.float32)


def dlhs_scale_grad_by_rhs_scale(
    grad: jnp.ndarray,
    rhs: qpl.QArray,
    group_sizes: jnp.ndarray,
) -> jnp.ndarray:
  """Squeezes the rhs scale and multiplies it with the incoming gradient before DLHS."""
  rhs_scale = rhs.scale

  # 1. Squeeze the scale [g, 1, n] to 2D [g, n]
  if rhs_scale.ndim == 3 and rhs_scale.shape[1] == 1:
    rhs_scale = rhs_scale.squeeze(axis=1)

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


def check_tc_unsort_bwd_supported(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule | None,
    tc_unsort_cfg: TcUnsortCfg,
):
  """The fused TC unsort scales and quantizes the cotangent in token order, before the gather into expert slots."""
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
  if not quantization_rule.disable_channelwise_axes:
    raise NotImplementedError("tc_unsort_cfg requires a per-tensor bwd scale (fixed or disable_channelwise_axes).")


def lhs_rhs_share_fixed_scale(
    quantization_rule: qwix.QtRule | None,
    lhs: qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
) -> bool:
  """True when lhs and rhs carry the same fixed per-tensor scale, so grad * rhs.scale == grad * lhs.scale."""
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


def _fwd_quantize_activation_and_weight(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    quantization_rule: qwix.QtRule,
    skip_act_quant: bool,
    use_manual_quantization: bool,
    rhs_is_flat2d: bool = False,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray]:
  """Handles act and weight quantization for GMM forward inputs."""
  if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray) and not skip_act_quant:
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
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else ([1] if rhs_is_flat2d else [2]),
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
  """Applies QAG (Quantization All-Gather) to RHS weights during forward pass."""
  if gathered_shape is None:
    for axis_name, axis_idx in weight_gather_axes:
      rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
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


def _drhs_scatter_weight(
    drhs: jnp.ndarray,
    weight_gather_axes: List[Tuple[str, int]],
    flat_rhs_shard: bool = False,
) -> jnp.ndarray:
  """Scatters the DRHS output back in the reverse order of the forward gather."""
  if flat_rhs_shard:
    drhs = drhs.reshape(-1, drhs.shape[-1])
  for axis_name, axis_idx in reversed(weight_gather_axes):
    drhs = jax.lax.psum_scatter(drhs, axis_name, scatter_dimension=axis_idx, tiled=True)
  return drhs


def _bwd_prepare_inputs(
    grad: jnp.ndarray,
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    dlhs_needs_repeated_rhs_scale: bool,
    quantization_rule: qwix.QtRule | None,
) -> tuple[jnp.ndarray | qpl.QArray, jnp.ndarray | qpl.QArray, jnp.ndarray, jnp.ndarray]:
  """Prepares backward operands by unwrapping QArrays and propagating forward scales."""
  dlhs_dout = grad
  drhs_dout = grad
  orig_rhs = rhs

  if isinstance(rhs, qpl.QArray):
    if not dlhs_needs_repeated_rhs_scale and rhs.scale.size == 1:
      dlhs_dout = dlhs_dout * rhs.scale.astype(grad.dtype).reshape(())
    elif not dlhs_needs_repeated_rhs_scale and rhs.scale.ndim >= 1 and rhs.scale.shape[0] == 1:
      dlhs_dout = dlhs_dout * rhs.scale.astype(grad.dtype).reshape(1, -1)
    else:
      dlhs_dout = dlhs_scale_grad_by_rhs_scale(dlhs_dout, rhs, group_sizes)
    rhs = rhs.qvalue

  # Backends that quantize lhs inside the forward kernel leave residual_lhs unquantized.
  if quantization_rule and quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray):
    lhs = qpl.quantize(
        lhs,
        quantization_rule.act_qtype,
        channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
        # pyrefly: ignore[bad-argument-type]
        calibration_method=quantization_rule.act_calibration_method,
    )

  if isinstance(lhs, qpl.QArray):
    if lhs_rhs_share_fixed_scale(quantization_rule, lhs, orig_rhs):
      drhs_dout = dlhs_dout
    else:
      drhs_dout = drhs_dout * gmm_backends.scale_like(lhs.scale, drhs_dout, grad.dtype)
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


def _quantized_gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype,  # pyrefly: ignore[bad-function-definition]
    tiling: tuple,
    group_offset: jnp.ndarray | None,
    existing_out: jnp.ndarray | None,
    interpret: bool,
    backend: gmm_backends.GmmBackend,
    quantization_rule: qwix.QtRule | None,
    weight_gather_axes: List[Tuple[str, int]] | None,
    use_manual_quantization: bool,
    lhs_vma_axes: tuple,
    rhs_vma_axes: tuple,
    partial_sum: jnp.ndarray | None,
    return_rhs: bool,
    return_lhs: bool,
    flat_fsdp_shard: tuple[int, int, int] | None,
    out_is_3d: bool,
    tc_routing: object,
    tc_unsort_cfg: TcUnsortCfg | None,
):
  """Forward pass for custom-VJP / quantized grouped matmul."""
  del return_rhs, return_lhs, rhs_vma_axes

  lhs_is_qarray = isinstance(lhs, qpl.QArray)
  rhs_is_qarray = isinstance(rhs, qpl.QArray)
  if flat_fsdp_shard is not None:
    if rhs_is_qarray or rhs.ndim != 2 or not weight_gather_axes:
      raise NotImplementedError("flat_fsdp_shard needs a plain 2D rhs block and weight_gather_axes.")
    if any(axis_idx != 0 for _, axis_idx in weight_gather_axes):
      raise ValueError("flat_fsdp_shard: the flat rhs is gathered on dim 0 only.")
    if not (
        backend.supports_qag
        and quantization_rule
        and quantization_rule.bwd_qtype
        and quantization_rule.weight_calibration_method.startswith("fixed")
    ):
      raise NotImplementedError("flat_fsdp_shard requires a QAG-capable backend (fixed weight calibration, bwd_qtype).")

  if quantization_rule:
    # pyrefly: ignore[bad-assignment]
    lhs, rhs = _fwd_quantize_activation_and_weight(
        lhs,
        rhs,
        quantization_rule,
        skip_act_quant=backend.supports_native_lhs_quant,
        use_manual_quantization=use_manual_quantization,
        rhs_is_flat2d=flat_fsdp_shard is not None,
    )

  if (
      backend.supports_qag
      and quantization_rule
      and quantization_rule.bwd_qtype
      and quantization_rule.weight_calibration_method.startswith("fixed")
      and isinstance(rhs, qpl.QArray)
      and weight_gather_axes
  ):
    # pyrefly: ignore[bad-assignment]
    rhs = _fwd_gather_weight(rhs, weight_gather_axes, gathered_shape=flat_fsdp_shard)

  lhs_scale = None
  if backend.supports_native_lhs_quant and not isinstance(lhs, qpl.QArray):
    lhs_scale = prepare_lhs_scale(quantization_rule)

  out = backend.fwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type=preferred_element_type,
      tiling=tiling,
      group_offset=group_offset,
      existing_out=existing_out,
      interpret=interpret,
      lhs_vma_axes=lhs_vma_axes,
      use_manual_quantization=use_manual_quantization,
      partial_sum=partial_sum,
      lhs_scale=lhs_scale,
      out_is_3d=out_is_3d,
  )

  if tc_unsort_cfg is not None:
    cfg = tc_unsort_cfg
    out = ragged_sort_tc._gather_reduce(  # pylint: disable=protected-access
        out[: cfg.unpadded_cap], tc_routing, cfg.num_out_tokens, cfg.topk, cfg.blocks
    )
    if cfg.combine_rs_cfg is not None:
      ep_axis_name, seq_len = cfg.combine_rs_cfg
      out = out.reshape(-1, seq_len, *out.shape[1:]).astype(preferred_element_type)
      out = jax.lax.psum_scatter(out, ep_axis_name, scatter_dimension=0, tiled=True)
      out = out.reshape(*out.shape[:2], -1)

  existing_out_dtype_ref = jnp.zeros((0,), dtype=existing_out.dtype) if existing_out is not None else None

  return (out, lhs, rhs), (
      lhs,
      rhs,
      group_sizes,
      group_offset,
      partial_sum,
      existing_out_dtype_ref,
      lhs_is_qarray,
      rhs_is_qarray,
      tc_routing,
  )


def _quantized_gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,  # pyrefly: ignore[bad-function-definition]
    tiling: tuple,
    interpret: bool,
    backend: gmm_backends.GmmBackend,
    quantization_rule: qwix.QtRule | None,
    weight_gather_axes: List[Tuple[str, int]] | None,
    use_manual_quantization: bool,
    lhs_vma_axes: tuple,
    rhs_vma_axes: tuple,
    return_rhs: bool,
    return_lhs: bool,
    flat_fsdp_shard: tuple[int, int, int] | None,
    out_is_3d: bool,
    tc_unsort_cfg: TcUnsortCfg | None,
    residual,
    grad: tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
):
  """Backward pass for custom-VJP / quantized grouped matmul."""
  del preferred_element_type, out_is_3d
  (
      residual_lhs,
      residual_rhs,
      group_sizes,
      group_offset,
      partial_sum_fwd,
      existing_out_dtype_ref,
      lhs_is_qarray,
      rhs_is_qarray,
      tc_routing,
  ) = residual
  num_actual_groups = residual_rhs.shape[0]
  grad_out, dlhs_partial_sum, drhs_partial_sum = grad

  if not return_lhs:
    dlhs_partial_sum = None
  elif isinstance(dlhs_partial_sum, qpl.QArray):
    dlhs_partial_sum = dlhs_partial_sum.qvalue

  if not return_rhs:
    drhs_partial_sum = None
  elif isinstance(drhs_partial_sum, qpl.QArray):
    drhs_partial_sum = drhs_partial_sum.qvalue

  lhs_is_3d = (residual_lhs.qvalue if isinstance(residual_lhs, qpl.QArray) else residual_lhs).ndim == 3
  if tc_unsort_cfg is not None:
    check_tc_unsort_bwd_supported(residual_lhs, residual_rhs, quantization_rule, tc_unsort_cfg)
    if tc_unsort_cfg.combine_rs_cfg is not None:
      grad_out = grad_out.reshape(-1, grad_out.shape[-1] // 128, 128)

  dlhs_dout, drhs_dout, lhs, rhs = _bwd_prepare_inputs(
      grad_out,
      residual_lhs,
      residual_rhs,
      group_sizes,
      backend.dlhs_needs_repeated_rhs_scale,
      quantization_rule,
  )

  if quantization_rule:
    dlhs_dout, drhs_dout = _bwd_quantize_gradient(dlhs_dout, drhs_dout, quantization_rule)

  if tc_unsort_cfg is not None:
    cfg = tc_unsort_cfg
    padded_cap = (residual_lhs.qvalue if isinstance(residual_lhs, qpl.QArray) else residual_lhs).shape[0]

    def _unsort_gather_arr(arr):
      if cfg.combine_rs_cfg is not None:
        arr = jax.lax.all_gather(arr, cfg.combine_rs_cfg[0], axis=0, tiled=True)
      # pylint: disable=protected-access
      arr = ragged_sort_tc._gather(arr, tc_routing, cfg.unpadded_cap, cfg.blocks, cfg.mask_padding)
      # pylint: enable=protected-access
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

  dlhs, lhs_scale_applied = backend.dlhs(
      dlhs_dout,
      rhs,
      group_sizes,
      preferred_element_type=lhs_dtype,
      tiling=tiling,
      group_offset=group_offset,
      interpret=interpret,
      lhs_vma_axes=lhs_vma_axes,
      use_manual_quantization=use_manual_quantization,
      partial_sum=dlhs_partial_sum,
      lhs_scale=lhs_scale_for_dlhs,
      out_is_3d=lhs_is_3d,
  )

  drhs = backend.drhs(
      lhs,
      drhs_dout,
      group_sizes,
      preferred_element_type=rhs_dtype,
      tiling=tiling,
      group_offset=group_offset,
      num_actual_groups=num_actual_groups,
      interpret=interpret,
      rhs_vma_axes=rhs_vma_axes,
      use_manual_quantization=use_manual_quantization,
      partial_sum=drhs_partial_sum,
  )
  if backend.supports_qag and quantization_rule and quantization_rule.bwd_qtype and weight_gather_axes:
    drhs = _drhs_scatter_weight(drhs, weight_gather_axes, flat_rhs_shard=flat_fsdp_shard is not None)

  dpartial_sum = grad_out if partial_sum_fwd is not None else None
  d_existing_out = (
      grad_out.astype(existing_out_dtype_ref.dtype)
      if backend.accumulate_existing_out_in_bwd and existing_out_dtype_ref is not None
      else None
  )

  if lhs_is_qarray and isinstance(residual_lhs, qpl.QArray):
    if not lhs_scale_applied:
      dlhs = dlhs * gmm_backends.scale_like(residual_lhs.scale, dlhs, dlhs.dtype)
    dlhs = qpl.QArray(
        qvalue=dlhs,
        scale=jnp.zeros_like(residual_lhs.scale),
        zero_point=jnp.zeros_like(residual_lhs.zero_point) if residual_lhs.zero_point is not None else None,
        qtype=residual_lhs.qtype,
    )
  if rhs_is_qarray and isinstance(residual_rhs, qpl.QArray):
    drhs = qpl.QArray(
        qvalue=drhs,
        scale=jnp.zeros_like(residual_rhs.scale),
        zero_point=jnp.zeros_like(residual_rhs.zero_point) if residual_rhs.zero_point is not None else None,
        qtype=residual_rhs.qtype,
    )

  return dlhs, drhs, None, None, d_existing_out, dpartial_sum, None


def quantized_gmm(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    *,
    backend: gmm_backends.GmmBackend,
    preferred_element_type: jax.typing.DTypeLike = jnp.float32,
    tiling: tuple = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool | None = None,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    use_qwix_quantization: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    lhs_vma_axes: tuple = (),
    rhs_vma_axes: tuple = (),
    qwix_rule: qwix.QtRule | None = None,
    use_manual_quantization: bool = False,
    partial_sum: jnp.ndarray | None = None,
    return_rhs: bool = False,
    return_lhs: bool = False,
    flat_fsdp_shard: tuple[int, int, int] | None = None,
    out_is_3d: bool = False,
    tc_routing: object = None,
    tc_unsort_cfg: TcUnsortCfg | None = None,
):
  """Executes grouped matmul with custom VJP, quantization, and optional QAG / TC unsort."""
  lhs_ndim = (lhs.qvalue if isinstance(lhs, qpl.QArray) else lhs).ndim
  if (lhs_ndim == 3 or out_is_3d) and not backend.supports_3d_lhs:
    raise NotImplementedError(f"3D lhs / out is not supported by backend {backend.name!r} (requires tokamax v2).")
  if return_rhs and not backend.supports_partial_sum:
    raise ValueError(f"return_rhs=True is not supported by backend {backend.name!r} (requires tokamax v2).")
  if return_lhs and not backend.supports_partial_sum:
    raise ValueError(f"return_lhs=True is not supported by backend {backend.name!r} (requires tokamax v2).")
  if partial_sum is not None and not backend.supports_partial_sum:
    raise NotImplementedError(f"partial_sum is not supported by backend {backend.name!r} (requires tokamax v2).")
  if tc_unsort_cfg is not None and not backend.supports_tc_unsort:
    raise NotImplementedError(f"tc_unsort_cfg is not supported by backend {backend.name!r} (requires tokamax v2).")
  if flat_fsdp_shard is not None and not backend.supports_flat_fsdp_shard:
    raise NotImplementedError(f"flat_fsdp_shard is not supported by backend {backend.name!r} (requires tokamax v2).")

  if interpret is None:
    interpret = False

  quantization_rule = resolve_quantization_rule(
      use_qwix_quantization=use_qwix_quantization,
      qwix_rule=qwix_rule,
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
  )

  lhs_dtype = preferred_element_type if isinstance(lhs, qpl.QArray) else lhs.dtype
  rhs_dtype = preferred_element_type if isinstance(rhs, qpl.QArray) else rhs.dtype

  gmm_fwd_bwd = lambda *args: _quantized_gmm_fwd(*args)[0]  # pylint: disable=C3001
  gmm_fwd_bwd = jax.custom_vjp(
      gmm_fwd_bwd,
      nondiff_argnums=(3, 4, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20),
  )
  gmm_fwd_bwd.defvjp(_quantized_gmm_fwd, functools.partial(_quantized_gmm_bwd, lhs_dtype, rhs_dtype))
  out, lhs_out, rhs_out = gmm_fwd_bwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      group_offset,
      existing_out,
      interpret,
      backend,
      quantization_rule,
      weight_gather_axes,
      use_manual_quantization,
      lhs_vma_axes,
      rhs_vma_axes,
      partial_sum,
      return_rhs,
      return_lhs,
      flat_fsdp_shard,
      out_is_3d,
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
