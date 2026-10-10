# Copyright 2026 Google LLC
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

"""Differentiable 3D-layout expert GMMs for the DSv3 routed-expert MLP.

Lineage keeps routed tokens in the 3D layout ``[BT, D0, 128]`` with
``D0 = 7168 // 128 = 56``.  These wrappers let the expert MLP consume and
produce that layout directly, so ``ragged_flatten`` / ``ragged_unflatten``
are no longer needed around the GMMs::

    x3d [BT, 56, 128] --gate_gmm--> gate_out [BT, 4096]
                                        | ragged_silu_mul
                                    act [BT, 2048] --linear_gmm--> out3d [BT,
                                    56, 128]

Every forward and backward matmul is a single Pallas kernel that does the
3D<->2D relayout in VMEM (see ``gmmv2_kernel`` / ``tgmm_kernel``):

  gate_gmm   fwd : gmm_v2 (3D lhs)
  [BT,56,128]x[E,7168,4096]
             bwd : dx   = gmm_v2 (transpose_rhs, 3D out)
             [BT,4096]x[E,7168,4096]^T
                   dWg  = tgmm_v2 (3D lhs)                  [BT,56,128]^T x
                   [BT,4096]
  linear_gmm fwd : gmm_v2 (3D out)
  [BT,2048]x[E,2048,7168]
             bwd : dact = gmm_v2 (3D lhs, transpose_rhs)
             [BT,56,128]x[E,2048,7168]^T
                   dWl  = tgmm_v2 (3D rhs)                  [BT,2048]^T x
                   [BT,56,128]

``gate_gmm`` is forward-only and writes into a caller buffer (the activation
bank); ``gate_gmm_bwd`` is its backward pass. ``ExpertGmm3d`` implements the
``GmmFn`` protocol (``.gate`` / ``.gate_bwd`` / ``.linear`` / ``.linear_bwd``)
used by ``dsv3_experts.py``. The backward passes can accumulate the weight
gradients into existing ones in place (``dw_acc``).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.lineage.gmmv2 import gmmv2_kernel
from maxtext.experimental.lineage.gmmv2 import tgmm_kernel

TileSizes = gmmv2_kernel.TileSizes
LANES = 128


@dataclasses.dataclass(frozen=True)
class ExpertMlpConfig:
  """Static configuration for the six expert-MLP kernels.

  ``None`` for any tile field means "use the kernel's auto-tiler".  Lineage
  shapes are fixed, so production should pass explicit ``TileSizes`` picked
  from the microbenchmarks (the auto-tiler does not know about the 3D
  ``tile % 1024`` constraint's performance cliffs).

  Attributes:
    gate_fwd: tiles for gate_out = x3d @ Wg.
    gate_dx: tiles for dx3d = dgate @ Wg^T.
    gate_dw: tiles for dWg = x3d^T @ dgate.
    linear_fwd: tiles for out3d = act @ Wl.
    linear_dact: tiles for dact = dout3d @ Wl^T.
    linear_dw: tiles for dWl = act^T @ dout3d.
    disable_multi_core_mode: run kernels as a plain pallas_call (needed when
      each JAX device is already a single TensorCore, e.g. in unit tests).
    zero_initialize: zero the padding rows (>= sum(group_sizes)) of every GMM
      output. Matches ``jax.lax.ragged_dot`` semantics; can be disabled if the
      consumer ignores padding rows.
    flat_weights: hand every expert weight to `gmm_v2` as its 2D ``[E * rows,
      cols]`` view (``gmm_v2(flat_rhs=True)``). The view is a bitcast, so the
      kernel reads the weight in whatever HBM tiling its producer picked (e.g.
      ``T(16,128)``); with the 3D operand XLA inserts a relayout copy to
      ``T(8,128)`` in front of every GMM instead. Values and autodiff are
      unchanged. Requires the rhs row tile (``tile_k``, or ``tile_n`` for the
      transposed dx / dact GMMs) to divide the per-expert rows.
  """

  gate_fwd: TileSizes | None = None
  gate_dx: TileSizes | None = None
  gate_dw: TileSizes | None = None
  linear_fwd: TileSizes | None = None
  linear_dact: TileSizes | None = None
  linear_dw: TileSizes | None = None
  disable_multi_core_mode: bool = False
  zero_initialize: bool = True
  flat_weights: bool = True


def _check_3d(name: str, x: jax.Array) -> None:
  if x.ndim != 3 or x.shape[-1] != LANES:
    raise ValueError(f"{name} must be 3D [M, D // 128, 128], got shape {x.shape}.")
  if x.dtype != jnp.bfloat16 and not gmmv2_kernel.is_fp8(x.dtype):
    raise ValueError(f"{name} must be bfloat16 or fp8, got {x.dtype}.")


# Output dtype of the fp8 GMMs (tokens, token gradients and activations).
_FP8_OUT_DTYPE = jnp.bfloat16


def _gmm(cfg: ExpertMlpConfig, tiles: TileSizes | None, *args, **kwargs):
  if tiles is not None:
    kwargs["tile_info"] = tiles
  return gmmv2_kernel.gmm_v2(
      *args,
      disable_multi_core_mode=cfg.disable_multi_core_mode,
      zero_initialize=cfg.zero_initialize,
      flat_rhs=cfg.flat_weights,
      **kwargs,
  )


def _tgmm(cfg: ExpertMlpConfig, tiles: TileSizes | None, *args, **kwargs):
  if tiles is not None:
    kwargs["tile_info"] = tiles
  return tgmm_kernel.tgmm_v2(
      *args,
      disable_multi_core_mode=cfg.disable_multi_core_mode,
      **kwargs,
  )


def _psum_axes(ct: jax.Array, primal: jax.Array) -> frozenset[str]:
  """Axes over which `_match_vma(ct, primal)` needs a real psum."""
  primal_mat = jax.typeof(primal).manual_axis_type
  return jax.typeof(ct).manual_axis_type.varying - primal_mat.varying - primal_mat.reduced


def _match_vma(ct: jax.Array, primal: jax.Array) -> jax.Array:
  """Gives cotangent `ct` the manual-axis type JAX expects for `primal`.

  Under ``shard_map(check_vma=True)`` a ``custom_vjp`` bwd rule must return
  cotangents whose manual-axis type is the *cotangent type* of the primal
  (``to_ct_aval``): varying axes stay varying, axes over which the primal is
  ``reduced`` become ``unreduced`` (a deferred sum, no communication) and axes
  over which the primal is invariant need a real ``psum``. JAX's own autodiff
  inserts exactly these casts when e.g. a replicated / reduced expert weight
  (``(dcn, z)`` in DSv3) meets varying activations. Our kernels simply give
  every output the lhs' (fully varying) axes, so reproduce the casts here to
  keep the semantics identical to the ``jax.lax.ragged_dot`` path.

  Args:
    ct: Cotangent array returned by the backward GMM/TGMM kernel.
    primal: Corresponding primal input array.

  Returns:
    Cotangent cast to the manual-axis type expected for `primal`.
  """
  ct_mat = jax.typeof(ct).manual_axis_type
  primal_mat = jax.typeof(primal).manual_axis_type
  if ct_mat.unreduced or ct_mat.reduced:
    raise NotImplementedError(f"Kernel cotangent has unreduced/reduced axes: {ct_mat}.")
  if primal_mat.unreduced:
    raise NotImplementedError(f"Unreduced primal inputs are not supported: {primal_mat}.")
  want_varying = primal_mat.varying
  want_unreduced = primal_mat.reduced  # to_ct: reduced -> unreduced.
  have_varying = ct_mat.varying
  # Axes we vary over but the primal is invariant over: all-reduce.
  if extra := _psum_axes(ct, primal):
    ct = jax.lax.psum(ct, tuple(sorted(extra)))
  # Axes the primal is `reduced` over: mark the sum as deferred.
  if want_unreduced:
    missing_var = want_unreduced - have_varying
    if missing_var:
      ct = jax.lax.pcast(ct, tuple(sorted(missing_var)), to="varying")
    ct = jax.lax.pcast(ct, tuple(sorted(want_unreduced)), to="unreduced")
  # Axes the primal varies over but the kernel output does not.
  if missing := want_varying - jax.typeof(ct).manual_axis_type.varying:
    ct = jax.lax.pcast(ct, tuple(sorted(missing)), to="varying")
  return ct


def _tgmm_dw(
    cfg: ExpertMlpConfig,
    tiles: TileSizes | None,
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    w: jax.Array,
    dw_acc: jax.Array | None,
    dw_dtype: jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
  """Returns `dw_acc` plus the weight gradient `lhs^T @ rhs` of `w`.

  The kernel writes into `dw_acc` in place when it can: the casts in
  `_match_vma` are per-device no-ops unless they need a psum, so adding the
  kernel's partial sums to `dw_acc` (already of w's cotangent type) is exact.

  Args:
    cfg: Kernel configuration.
    tiles: Tiles of the TGMM.
    lhs: TGMM lhs, whose manual-axis type the kernel output inherits.
    rhs: TGMM rhs.
    group_sizes: Tokens per local expert.
    w: Weight the gradient is taken of.
    dw_acc: Optional gradient of `w` to accumulate into, e.g. from another
      microbatch.
    dw_dtype: Dtype of the gradient. Defaults to w's dtype.
    **kwargs: Extra `tgmm_v2` arguments, e.g. fp8 scales.

  Returns:
    The (accumulated) gradient of `w`, with w's cotangent type.
  """
  tgmm = functools.partial(
      _tgmm,
      cfg,
      tiles,
      lhs,
      rhs,
      group_sizes,
      w.shape[0],
      preferred_element_type=w.dtype if dw_dtype is None else dw_dtype,
      **kwargs,
  )
  if dw_acc is not None and cfg.disable_multi_core_mode and not _psum_axes(lhs, w):
    return tgmm(acc=dw_acc)
  dw = _match_vma(tgmm(), w)
  return dw if dw_acc is None else dw_acc + dw


# ----------------------------------------------------------------------------
# gate GMM: 3D lhs in, 2D out.
# ----------------------------------------------------------------------------


def gate_gmm(
    cfg: ExpertMlpConfig,
    x3d: jax.Array,  # bf16 or fp8 [BT, D // 128, 128]
    w_gate: jax.Array,  # [E, D, F_gate]
    group_sizes: jax.Array,  # int32 [E]
    out: jax.Array,  # [R, F_gate]
    out_offset: jax.Array | int,
    out_scale: jax.Array | None = None,  # f32 [1, 1]
    *,
    glu_coeffs: jax.Array | None = None,  # [BT, 1]
    glu_out_dtype: jnp.dtype | None = None,
    glu_out_scale: jax.Array | None = None,  # f32 [1, 1]
) -> Any:  # [R, F_gate] (and [BT, F_gate // 2] when glu_coeffs is set)
  """Writes x[m] @ w_gate[expert(m)] to out[out_offset + m], x in 3D layout.

  Not differentiable; use `gate_gmm_bwd` for the backward pass. Only rows
  [out_offset, out_offset + round_up(sum(group_sizes), sublanes)) of `out` are
  written, the rows past sum(group_sizes) with garbage.

  Args:
    cfg: Kernel configuration. Requires `disable_multi_core_mode`.
    x3d: Routed tokens in the 3D layout.
    w_gate: Gate weights.
    group_sizes: Tokens per local expert.
    out: Buffer to write into, e.g. the activation bank. Donated.
    out_offset: Row of `out` receiving token 0. Must be a multiple of the
      sublane tiling of x3d: 16 for bf16, 32 for fp8.
    out_scale: For fp8 x3d and w_gate, the product of their per-tensor scales.
    glu_coeffs: Optional per-token routing coefficients `[BT, 1]` to fuse
      `silu(g0) * g1 * c` in the gate epilogue.
    glu_out_dtype: Optional dtype for the fused GLU output `act`.
    glu_out_scale: Optional `[1, 1]` f32 quantization scale for `act`.

  Returns:
    The updated `out`, or `(out, act)` when `glu_coeffs` is provided.
  """
  _check_3d("x3d", x3d)
  kwargs: dict[str, Any] = {}
  if out_scale is not None:
    kwargs.update(
        out_scale=out_scale,
        acc_dtype=jnp.float32,
        preferred_element_type=out.dtype,
    )
  if glu_coeffs is not None:
    kwargs["glu_coeffs"] = glu_coeffs
    kwargs["glu_out_dtype"] = glu_out_dtype
    kwargs["glu_out_scale"] = glu_out_scale
  return _gmm(
      cfg,
      cfg.gate_fwd,
      x3d,
      w_gate,
      group_sizes,
      out=out,
      out_offset=out_offset,
      **kwargs,
  )


def gate_gmm_bwd(
    cfg: ExpertMlpConfig,
    x3d: jax.Array,  # bf16 or fp8 [BT, D // 128, 128]
    w_gate: jax.Array,  # [E, D, F_gate]
    group_sizes: jax.Array,  # int32 [E]
    dgate: jax.Array,  # [BT, F_gate]
    dw_acc: jax.Array | None = None,  # [E, D, F_gate]
    *,
    w_q: jax.Array | None = None,  # fp8 [E, D, F_gate]
    dgate_scale: jax.Array | None = None,  # f32 [1, 1]
    dgate_qtype: jnp.dtype | None = None,
    dx_scale: jax.Array | None = None,  # f32 [1, 1]
    dw_scale: jax.Array | None = None,  # f32 [1, 1]
    dw_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:  # dx3d [BT, D // 128, 128], dw [E, D, F_gate]
  """Backward pass for gate_gmm given the cotangent of its BT output rows.

  With `w_q`, both GMMs run in fp8 on pre-quantized (or in-kernel quantized via
  `dgate_scale` / `dgate_qtype`) operands: dx3d = dgate @ w_q^T scaled by
  `dx_scale`, dw = x3d^T @ dgate scaled by `dw_scale`, both output in bf16 (dw
  in `dw_dtype`, or w_gate's dtype).

  Args:
    cfg: Kernel configuration.
    x3d: Routed tokens in the 3D layout (fp8 with `w_q`).
    w_gate: Gate weights, giving the dtype and type of dw.
    group_sizes: Tokens per local expert.
    dgate: Cotangent of the gating GMM output rows (fp8 or bf16 with `w_q`).
    dw_acc: Optional gradient of `w_gate` to accumulate into (see `_tgmm_dw`).
    w_q: Optional fp8 copy of w_gate; selects the fp8 path.
    dgate_scale: Optional per-tensor scale to quantize bf16 `dgate` in-kernel.
    dgate_qtype: Optional fp8 dtype to quantize `dgate` to in-kernel.
    dx_scale: Product of the dgate and w_q scales (or w_q scale when
      `dgate_scale` is given).
    dw_scale: Product of the x3d and dgate scales (or x3d scale when
      `dgate_scale` is given).
    dw_dtype: Dtype of dw_gate. Defaults to w_gate's dtype.

  Returns:
    Tuple of (dx3d, dw_gate), with dw_gate including dw_acc.
  """
  dx_kwargs: dict[str, Any]
  dw_kwargs: dict[str, Any]
  if w_q is None:
    dgate = dgate.astype(x3d.dtype)
    dx_kwargs, dw_kwargs = {}, {}
  else:
    dx_kwargs = dict(
        out_scale=dx_scale,
        acc_dtype=jnp.float32,
        preferred_element_type=_FP8_OUT_DTYPE,
    )
    dw_kwargs = dict(out_scale=dw_scale, acc_dtype=jnp.float32)
    if dgate_scale is not None:
      dx_kwargs["lhs_scale"] = dgate_scale
      dx_kwargs["lhs_quant_dtype"] = dgate_qtype
      dw_kwargs["rhs_quant_scale"] = dgate_scale
      dw_kwargs["rhs_quant_dtype"] = dgate_qtype
  # dx = dgate @ Wg^T, written straight back into the 3D layout.
  dx3d = _gmm(
      cfg,
      cfg.gate_dx,
      dgate,
      w_gate if w_q is None else w_q,
      group_sizes,
      transpose_rhs=True,
      out_is_3d=True,
      **dx_kwargs,
  )
  # dWg[e] = x[rows of e]^T @ dgate[rows of e], x consumed in the 3D layout.
  dw_gate = _tgmm_dw(
      cfg,
      cfg.gate_dw,
      x3d,
      dgate,
      group_sizes,
      w_gate,
      dw_acc,
      dw_dtype,
      **dw_kwargs,
  )
  return _match_vma(dx3d, x3d), dw_gate


# ----------------------------------------------------------------------------
# linear GMM: 2D lhs in, 3D out.
# ----------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def linear_gmm(
    cfg: ExpertMlpConfig,
    act: jax.Array,  # bf16 [BT, F]
    w_linear: jax.Array,  # [E, F, D]
    group_sizes: jax.Array,  # int32 [E]
) -> jax.Array:  # bf16 [BT, D // 128, 128]
  """out[m] = act[m] @ w_linear[expert(m)], produced in the 3D layout."""
  return _linear_gmm_fwd(cfg, act, w_linear, group_sizes)[0]


def _linear_gmm_fwd(cfg, act, w_linear, group_sizes):
  if act.ndim != 2:
    raise ValueError(f"act must be 2D [M, F], got shape {act.shape}.")
  out3d = _gmm(
      cfg,
      cfg.linear_fwd,
      act,
      w_linear,
      group_sizes,
      out_is_3d=True,
  )
  return out3d, (act, w_linear, group_sizes)


def linear_gmm_fp8(
    cfg: ExpertMlpConfig,
    act: jax.Array,  # bf16 or fp8 [BT, F]
    w_q: jax.Array,  # fp8 [E, F, D]
    group_sizes: jax.Array,  # int32 [E]
    *,
    act_scale: jax.Array,  # f32 [1, 1]
    act_qtype: jnp.dtype,
    out_scale: jax.Array,  # f32 [1, 1]
) -> jax.Array:  # bf16 [BT, D // 128, 128]
  """fp8 `linear_gmm` (forward only; see `linear_gmm_bwd(w_q=...)`).

  act is quantized to `act_qtype` with the static `act_scale` inside the
  kernel (or consumed directly if already fp8), multiplied with w_q and the f32
  result scaled by act_scale and `out_scale` (w_q's scale).

  Args:
    cfg: Kernel configuration.
    act: Activations (bf16 or pre-quantized fp8).
    w_q: fp8 linear weights.
    group_sizes: Tokens per local expert.
    act_scale: Static per-tensor scale of act.
    act_qtype: fp8 dtype act is quantized to.
    out_scale: Per-tensor scale of w_q.

  Returns:
    The bf16 output in the 3D layout.
  """
  if act.ndim != 2:
    raise ValueError(f"act must be 2D [M, F], got shape {act.shape}.")
  if gmmv2_kernel.is_fp8(act.dtype):
    return _gmm(
        cfg,
        cfg.linear_fwd,
        act,
        w_q,
        group_sizes,
        out_scale=out_scale * act_scale,
        acc_dtype=jnp.float32,
        preferred_element_type=_FP8_OUT_DTYPE,
        out_is_3d=True,
    )
  return _gmm(
      cfg,
      cfg.linear_fwd,
      act,
      w_q,
      group_sizes,
      lhs_scale=act_scale,
      lhs_quant_dtype=act_qtype,
      out_scale=out_scale,
      acc_dtype=jnp.float32,
      preferred_element_type=_FP8_OUT_DTYPE,
      out_is_3d=True,
  )


def linear_gmm_bwd(
    cfg: ExpertMlpConfig,
    act: jax.Array,  # bf16 or fp8 [BT, F]
    w_linear: jax.Array,  # [E, F, D]
    group_sizes: jax.Array,  # int32 [E]
    dout3d: jax.Array,  # [BT, D // 128, 128]
    dw_acc: jax.Array | None = None,  # [E, F, D]
    *,
    w_q: jax.Array | None = None,  # fp8 [E, F, D]
    act_scale: jax.Array | None = None,  # f32 [1, 1]
    act_qtype: jnp.dtype | None = None,
    dout_scale: jax.Array | None = None,  # f32 [1, 1]
    dout_qtype: jnp.dtype | None = None,
    dact_scale: jax.Array | None = None,  # f32 [1, 1]
    dw_scale: jax.Array | None = None,  # f32 [1, 1]
    dw_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:  # dact [BT, F], dw [E, F, D]
  """Backward pass for linear_gmm given the cotangent of its output.

  With `w_q`, both GMMs run in fp8 on dout3d (pre-quantized or quantized
  in-kernel via `dout_scale` / `dout_qtype`): dact = dout3d @ w_q^T scaled by
  `dact_scale`, and dw = act^T @ dout3d with act quantized in-kernel like in
  `linear_gmm_fp8` (or consumed directly if already fp8) and scaled by
  `dw_scale`, both output in bf16 (dw in `dw_dtype`, or w_linear's dtype).

  Args:
    cfg: Kernel configuration.
    act: Activations, the lhs of linear_gmm.
    w_linear: Linear weights, giving the dtype and type of dw.
    group_sizes: Tokens per local expert.
    dout3d: Cotangent of the linear_gmm output in the 3D layout (fp8 or bf16
      with `w_q`).
    dw_acc: Optional gradient of `w_linear` to accumulate into (see `_tgmm_dw`).
    w_q: Optional fp8 copy of w_linear; selects the fp8 path.
    act_scale: Static per-tensor scale act is quantized with.
    act_qtype: fp8 dtype act is quantized to.
    dout_scale: Optional per-tensor scale to quantize bf16 `dout3d` in-kernel.
    dout_qtype: Optional fp8 dtype to quantize `dout3d` to in-kernel.
    dact_scale: Product of the dout3d and w_q scales (or w_q scale when
      `dout_scale` is given).
    dw_scale: Scale of dout3d (or None when `dout_scale` is given).
    dw_dtype: Dtype of dw_linear. Defaults to w_linear's dtype.

  Returns:
    Tuple of (dact, dw_linear), with dw_linear including dw_acc.
  """
  dact_kwargs: dict[str, Any]
  dw_kwargs: dict[str, Any]
  if w_q is None:
    dout3d = dout3d.astype(act.dtype)
    dact_kwargs, dw_kwargs = {}, {}
  else:
    dact_kwargs = dict(
        out_scale=dact_scale,
        acc_dtype=jnp.float32,
        preferred_element_type=_FP8_OUT_DTYPE,
    )
    if gmmv2_kernel.is_fp8(act.dtype):
      assert act_scale is not None
      dw_out_scale = act_scale if dw_scale is None else dw_scale * act_scale
      dw_kwargs = dict(
          out_scale=dw_out_scale,
          acc_dtype=jnp.float32,
      )
    else:
      dw_kwargs = dict(
          lhs_scale=act_scale,
          lhs_quant_dtype=act_qtype,
          out_scale=dw_scale,
          acc_dtype=jnp.float32,
      )
    if dout_scale is not None:
      dact_kwargs["lhs_scale"] = dout_scale
      dact_kwargs["lhs_quant_dtype"] = dout_qtype
      dw_kwargs["rhs_quant_scale"] = dout_scale
      dw_kwargs["rhs_quant_dtype"] = dout_qtype
  _check_3d("dout3d", dout3d)
  # dact = dout @ Wl^T, dout consumed in the 3D layout.
  dact = _gmm(
      cfg,
      cfg.linear_dact,
      dout3d,
      w_linear if w_q is None else w_q,
      group_sizes,
      transpose_rhs=True,
      **dact_kwargs,
  )
  # dWl[e] = act[rows of e]^T @ dout[rows of e], dout consumed in 3D layout.
  dw_linear = _tgmm_dw(
      cfg,
      cfg.linear_dw,
      act,
      dout3d,
      group_sizes,
      w_linear,
      dw_acc,
      dw_dtype,
      **dw_kwargs,
  )
  return _match_vma(dact, act), dw_linear


def _linear_gmm_bwd(cfg, res, dout3d):
  act, w_linear, group_sizes = res
  dact, dw_linear = linear_gmm_bwd(cfg, act, w_linear, group_sizes, dout3d)
  return dact, dw_linear, None


linear_gmm.defvjp(_linear_gmm_fwd, _linear_gmm_bwd)


# ----------------------------------------------------------------------------
# `gmm_fn` object for dsv3_experts.py.
# ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExpertGmm3d:
  """3D-layout expert GMMs implementing `dsv3_experts.GmmFn`.

  In the real DSv3 mesh every JAX device is a single TensorCore (the
  ``"core"`` mesh axis), hence ``disable_multi_core_mode=True`` by default.
  """

  cfg: ExpertMlpConfig = ExpertMlpConfig(disable_multi_core_mode=True)
  supports_fused_glu: bool = True
  supports_dynamic_grad_scale: bool = True

  def gate(
      self,
      x: jax.Array,
      w: jax.Array,
      group_sizes: jax.Array,
      out: jax.Array,
      out_offset: jax.Array | int,
      out_scale: jax.Array | None = None,
      *,
      glu_coeffs: jax.Array | None = None,
      glu_out_dtype: jnp.dtype | None = None,
      glu_out_scale: jax.Array | None = None,
  ) -> Any:
    """Runs the gate GMM; optionally fuses the GLU epilogue via `glu_coeffs`."""
    return gate_gmm(
        self.cfg,
        x,
        w,
        group_sizes,
        out,
        out_offset,
        out_scale=out_scale,
        glu_coeffs=glu_coeffs,
        glu_out_dtype=glu_out_dtype,
        glu_out_scale=glu_out_scale,
    )

  def gate_bwd(
      self,
      x: jax.Array,
      w: jax.Array,
      group_sizes: jax.Array,
      dgate: jax.Array,
      dw_acc: jax.Array | None = None,
      *,
      w_q: jax.Array | None = None,
      dgate_scale: jax.Array | None = None,
      dgate_qtype: jnp.dtype | None = None,
      dx_scale: jax.Array | None = None,
      dw_scale: jax.Array | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    return gate_gmm_bwd(
        self.cfg,
        x,
        w,
        group_sizes,
        dgate,
        dw_acc,
        w_q=w_q,
        dgate_scale=dgate_scale,
        dgate_qtype=dgate_qtype,
        dx_scale=dx_scale,
        dw_scale=dw_scale,
        dw_dtype=dw_dtype,
    )

  def linear_bwd(
      self,
      x: jax.Array,
      w: jax.Array,
      group_sizes: jax.Array,
      dout: jax.Array,
      dw_acc: jax.Array | None = None,
      *,
      w_q: jax.Array | None = None,
      x_scale: jax.Array | None = None,
      x_qtype: jnp.dtype | None = None,
      dout_scale: jax.Array | None = None,
      dout_qtype: jnp.dtype | None = None,
      dx_scale: jax.Array | None = None,
      dw_scale: jax.Array | None = None,
      dw_dtype: jnp.dtype | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    return linear_gmm_bwd(
        self.cfg,
        x,
        w,
        group_sizes,
        dout,
        dw_acc,
        w_q=w_q,
        act_scale=x_scale,
        act_qtype=x_qtype,
        dout_scale=dout_scale,
        dout_qtype=dout_qtype,
        dact_scale=dx_scale,
        dw_scale=dw_scale,
        dw_dtype=dw_dtype,
    )

  def linear(
      self,
      x: jax.Array,
      w: jax.Array,
      group_sizes: jax.Array,
      *,
      x_scale: jax.Array | None = None,
      x_qtype: jnp.dtype | None = None,
      out_scale: jax.Array | None = None,
  ) -> jax.Array:
    if out_scale is None:
      return linear_gmm(self.cfg, x, w, group_sizes)
    assert x_scale is not None and x_qtype is not None
    return linear_gmm_fp8(
        self.cfg,
        x,
        w,
        group_sizes,
        act_scale=x_scale,
        act_qtype=x_qtype,
        out_scale=out_scale,
    )


# Tiles measured on v7x for the DSv3 shapes (24.5k valid tokens of 32k, 8 local
# experts; see gmmv2_3d_perf_test.py test_4 / test_6). `gate_dw` was measured
# with fp8 operands and 16 local experts (test_7): 1.19 ms vs 1.86 ms for the
# auto-tiler's tm256/tk7168/tn768.
DSV3_V7X_CONFIG = ExpertMlpConfig(
    gate_fwd=TileSizes(tile_m=768, tile_k=1024, tile_n=4096, bucket_base=256),
    gate_dx=TileSizes(tile_m=1024, tile_k=4096, tile_n=1024, bucket_base=128),
    gate_dw=TileSizes(tile_m=768, tile_k=1024, tile_n=4096, bucket_base=256),
    linear_fwd=TileSizes(tile_m=1024, tile_k=2048, tile_n=1024, bucket_base=128),
    linear_dact=TileSizes(tile_m=768, tile_k=1024, tile_n=2048, bucket_base=256),
    linear_dw=TileSizes(tile_m=1024, tile_k=2048, tile_n=1024, bucket_base=256),
    disable_multi_core_mode=True,
    zero_initialize=False,
)

# Model dim for which `DSV3_V7X_CONFIG` was measured.
_DSV3_EMB_DIM = 7168


def make_gmm_fn(emb_dim: int) -> ExpertGmm3d:
  """Returns the `gmm_fn` for `dsv3.dsv3(...)` / `dsv3_mtp.dsv3_mtp_layer(...)`.

  Args:
    emb_dim: model dim. For the full DSv3 7168 the measured v7x tiles
      (`DSV3_V7X_CONFIG`) are used; otherwise the kernels' auto-tiler.
  """
  if emb_dim == _DSV3_EMB_DIM:
    return ExpertGmm3d(DSV3_V7X_CONFIG)
  return ExpertGmm3d()
