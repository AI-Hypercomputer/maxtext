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

"""Per-tensor quantization configs and helpers (qwix `fp8_full`-style).

Mirrors MaxText's qwix flags::

  quantization=fp8_full
  weight_quantization_calibration_method='fixed,-224,224'
  act_quantization_calibration_method='fixed,-224,224'
  bwd_quantization_calibration_method='absmax'

`fixed,<min>,<max>` is a static scale mapping max(|min|, |max|) to the largest
finite value of the quantized dtype (224 / 448 = 0.5 for e4m3fn); `absmax` is a
dynamic scale computed from each tensor's absolute maximum.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from maxtext.experimental.lineage import ops


def _finfo_max(qtype: Any) -> float:
  return float(jnp.finfo(qtype).max)


def get_static_scale(calibration_method: str, qtype: Any) -> float:
  """Returns the scale of a `fixed,<min>,<max>` calibration for `qtype`."""
  name, *bounds = calibration_method.split(",")
  if name != "fixed" or len(bounds) != 2:
    raise ValueError("Expected a 'fixed,<min>,<max>' calibration method, got" f" {calibration_method!r}.")
  return max(abs(float(b)) for b in bounds) / _finfo_max(qtype)


@dataclasses.dataclass(frozen=True)
class GmmQuantRule:
  """Per-tensor quantization of a set of GMMs / dots and their backward passes.

  Attributes:
    weight_qtype: Dtype the weights are quantized to.
    act_qtype: Dtype the forward activations (GMM / dot lhs) are quantized to.
    bwd_qtype: Dtype the output gradients are quantized to in the backward pass.
    grad_dtype: Dtype of the weight gradients output by the backward pass.
    weight_calibration_method: Static `fixed,<min>,<max>` weight calibration.
    act_calibration_method: Static `fixed,<min>,<max>` activation calibration.
    bwd_calibration_method: Gradient calibration; only `absmax` is supported.
  """

  weight_qtype: Any = jnp.float8_e4m3fn
  act_qtype: Any = jnp.float8_e4m3fn
  bwd_qtype: Any = jnp.float8_e5m2
  grad_dtype: Any = jnp.bfloat16
  weight_calibration_method: str = "fixed,-224,224"
  act_calibration_method: str = "fixed,-224,224"
  bwd_calibration_method: str = "absmax"

  def __post_init__(self):
    if self.bwd_calibration_method != "absmax":
      raise ValueError("Only 'absmax' gradient calibration is supported, got" f" {self.bwd_calibration_method!r}.")
    # Fail at construction on malformed static calibrations.
    _ = self.weight_scale, self.act_scale

  @property
  def weight_scale(self) -> float:
    return get_static_scale(self.weight_calibration_method, self.weight_qtype)

  @property
  def act_scale(self) -> float:
    return get_static_scale(self.act_calibration_method, self.act_qtype)


@dataclasses.dataclass(frozen=True)
class MlaQuantRule(GmmQuantRule):
  """`GmmQuantRule` of the MLA projections, plus splash attention options.

  Attributes:
    splash_qk_qtype: If set, q and k are saturated to this dtype's finite range
      and cast to it (unscaled) right before the splash attention kernel, in
      both the forward and the (rematerialized) backward pass. v, the attention
      output and all gradients stay in bf16.
    q_proj_bwd_quantized: Whether the backward pass of the q down and q up
      projections runs in fp8 like `_qdot_bwd`. If False, their dgrad and wgrad
      dots run in bf16 on the dequantized weights while the forward pass stays
      quantized.
    q_prescaled_log2e: Whether q arrives already multiplied by LOG2E (folds
      LOG2E into the softmax scale before casting to FP8).
  """

  splash_qk_qtype: Any | None = None
  q_proj_bwd_quantized: bool = True
  q_prescaled_log2e: bool = False


@dataclasses.dataclass(frozen=True)
class FsdpWeightQuantRule:
  """Weight-only fp8 quantization of the routed expert FSDP all-gathers.

  The routed expert weights are quantized once per step with a static
  per-tensor scale and all-gathered in `weight_qtype`; the expert GMM kernels
  dequantize the gathered weight tiles to bf16 in VMEM, folding the scale into
  the f32 accumulator. All other weights (attention, shared experts, dense
  MLPs, norms, router) are all-gathered in bf16. All matmuls run on bf16
  operands; activations and activation gradients are never quantized. The
  routed expert weight gradients are those of the dequantized weights (a
  straight-through estimator). Every decoder weight gradient is cast to
  `grad_reduce_dtype` before its FSDP reduce-scatters and DP all-reduces.

  Attributes:
    weight_qtype: Dtype the routed expert weights are all-gathered in.
    weight_calibration_method: Static `fixed,<min>,<max>` weight calibration.
    grad_reduce_dtype: Dtype of the weight gradients and their reductions.
  """

  weight_qtype: Any = jnp.float8_e4m3fn
  weight_calibration_method: str = "fixed,-224,224"
  grad_reduce_dtype: Any = jnp.float32

  def __post_init__(self):
    _ = self.weight_scale

  @property
  def weight_scale(self) -> float:
    return get_static_scale(self.weight_calibration_method, self.weight_qtype)

  def quantize(self, w: jax.Array) -> jax.Array:
    """Returns w quantized with the static weight scale, unless it is."""
    if w.dtype == self.weight_qtype:
      return w
    return static_quantize(w, self.weight_qtype, self.weight_scale)


@dataclasses.dataclass(frozen=True)
class QuantConfig:
  """Model-wide quantization config. `None` fields stay unquantized.

  Attributes:
    routed_experts: Quantization of the sparse layers' routed-expert GMMs,
      including the token all-gather that feeds them.
    mla: Quantization of the MLA projections (q down, kv down, q up, v up and
      out; k up stays unquantized) in all layers.
    combine_qtype: If set, the forward-pass combine token reduce-scatter of the
      sparse layers runs on SparseCore 0 (`ops.reduce_scatter_sc0`) with its ICI
      ring sends quantized to this dtype (only `jnp.float8_e4m3fn` is supported)
      using one dynamic absmax scale per token shared by all expert devices; the
      local, die-to-die and output tokens stay bf16. The backward pass stays in
      bf16.
    combine_per_hop_scaling: With `combine_qtype`, scale each ring hop of the
      combine reduce-scatter with its own per-token bound (local + die-to-die
      absmax plus the bound received with the running sum) instead of one
      all-device scale per token, which removes the cross-device absmax gather.
    fsdp_weights: If set, fp8 all-gathers of the routed expert weights with bf16
      compute and f32 gradient reductions (see `FsdpWeightQuantRule`). Exclusive
      with `routed_experts` and `mla`.
    ep_reduce_dtype: If set, the expert-parallel token reductions run in this
      dtype (e.g. f32): the top-k sums of the expert outputs and of the token
        gradients are accumulated across MoE chunks in it, and the forward
        combine reduce-scatter and the backward dispatch reduce-scatter send it;
        their results are cast back to the activation dtype. The dispatch
        all-gather and the backward all-gather of the output gradients stay in
        the activation dtype.
    grad_reduce_dtype: If set, the gradients of the embedding table, the LM head
      kernel and the final norm scale, whose forward passes run in the
      activation dtype, are accumulated and reduced in this dtype (only
      `jnp.float32` is supported; see `ops.f32_grad`).
  """

  routed_experts: GmmQuantRule | None = None
  mla: GmmQuantRule | None = None
  combine_qtype: Any | None = None
  combine_per_hop_scaling: bool = False
  fsdp_weights: FsdpWeightQuantRule | None = None
  ep_reduce_dtype: Any | None = None
  grad_reduce_dtype: Any | None = None

  def __post_init__(self):
    if self.grad_reduce_dtype not in (None, jnp.float32):
      raise ValueError("grad_reduce_dtype must be None or jnp.float32, got" f" {self.grad_reduce_dtype!r}.")
    if self.fsdp_weights is not None and (self.routed_experts is not None or self.mla is not None):
      raise ValueError("fsdp_weights is exclusive with routed_experts and mla quantization.")
    if self.combine_qtype is not None and self.combine_qtype != jnp.float8_e4m3fn:
      raise ValueError("combine_qtype must be None or jnp.float8_e4m3fn, got" f" {self.combine_qtype!r}.")
    if self.combine_per_hop_scaling and self.combine_qtype is None:
      raise ValueError("combine_per_hop_scaling requires combine_qtype.")


FP8_FULL = QuantConfig(routed_experts=GmmQuantRule(), mla=GmmQuantRule())
FP8_FULL_SPLASH_QK = QuantConfig(
    routed_experts=GmmQuantRule(),
    mla=MlaQuantRule(splash_qk_qtype=jnp.float8_e4m3fn),
)
FP8_FULL_SPLASH_QK_E3 = QuantConfig(
    routed_experts=GmmQuantRule(),
    mla=MlaQuantRule(
        splash_qk_qtype=jnp.float8_e4m3fn,
        q_prescaled_log2e=True,
    ),
)
FP8_FULL_COMBINE = dataclasses.replace(FP8_FULL, combine_qtype=jnp.float8_e4m3fn)
FP8_FULL_COMBINE_PER_HOP = dataclasses.replace(FP8_FULL_COMBINE, combine_per_hop_scaling=True)
FP8_FULL_SPLASH_QK_E3_COMBINE_PER_HOP = dataclasses.replace(
    FP8_FULL_SPLASH_QK_E3,
    combine_qtype=jnp.float8_e4m3fn,
    combine_per_hop_scaling=True,
)
# bf16 compute, fp8 routed expert weight all-gathers, f32 gradient reductions.
FP8_FSDP_AG = QuantConfig(
    fsdp_weights=FsdpWeightQuantRule(),
    ep_reduce_dtype=jnp.float32,
    grad_reduce_dtype=jnp.float32,
)
# `FP8_FSDP_AG` with the expert-parallel token reductions in the activation
# dtype (bf16), e.g. to compare the f32 and bf16 EP reduce-scatters.
FP8_FSDP_AG_BF16_EP = dataclasses.replace(FP8_FSDP_AG, ep_reduce_dtype=None)


def static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  """Returns clip(x / scale) cast to `qtype` (saturating, never inf / NaN)."""
  fmax = _finfo_max(qtype)
  return jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)


def absmax_to_scale(amax: jax.Array, qtype: Any) -> jax.Array:
  """Returns the `[1, 1]` f32 quantization scale for `amax` and `qtype`."""
  fmax = _finfo_max(qtype)
  amax_f32 = amax.astype(jnp.float32)
  scale = jnp.where(amax_f32 > 0, amax_f32 / fmax, 1.0)
  return scale.reshape(1, 1)


def _xla_quantize_with_scale(x: jax.Array, scale: jax.Array, qtype: Any) -> jax.Array:
  fmax = _finfo_max(qtype)
  inv_scale = 1.0 / scale.astype(jnp.float32)
  return jnp.clip(x.astype(jnp.float32) * inv_scale, -fmax, fmax).astype(qtype)


def _can_use_ragged_pallas(x: jax.Array) -> bool:
  return (
      x.ndim in (2, 3)
      and x.shape[0] >= 64
      and x.shape[0] % 64 == 0
      and ((x.ndim == 2 and x.shape[1] % 128 == 0) or (x.ndim == 3 and x.shape[1] % 8 == 0 and x.shape[2] == 128))
  )


def quantize_with_scale(
    x: jax.Array,
    scale: jax.Array,
    qtype: Any,
    num_valid_rows: jax.Array | int | None = None,
) -> jax.Array:
  """Returns `clip(x / scale)` cast to `qtype` using `[1, 1]` f32 `scale`.

  Args:
    x: Array to quantize; rows are its leading dimension.
    scale: Per-tensor `[1, 1]` float32 quantization scale.
    qtype: Dtype to quantize to.
    num_valid_rows: Number of leading rows of `x` that hold valid values. If
      `None`, all rows of `x` are quantized.

  Returns:
    `x` quantized to `qtype` over its valid rows.
  """
  if num_valid_rows is not None and _can_use_ragged_pallas(x):
    return jax.lax.platform_dependent(
        x,
        scale,
        jnp.asarray(num_valid_rows, jnp.int32),
        tpu=lambda a, s, n: ops.ragged_quantize(a, s, qtype, n),
        default=lambda a, s, _: _xla_quantize_with_scale(a, s, qtype),
    )
  return _xla_quantize_with_scale(x, scale, qtype)


def ragged_absmax(x: jax.Array, num_valid_rows: jax.Array | int | None = None) -> jax.Array:
  """Returns the `[1, 1]` f32 absmax of the first `num_valid_rows` of `x`."""
  abs_x = jnp.abs(x.astype(jnp.float32))
  if num_valid_rows is not None:
    rows = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    abs_x = jnp.where(rows < num_valid_rows, abs_x, 0.0)
  return jnp.max(abs_x).reshape(1, 1)


def _xla_absmax_quantize(
    x: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None
) -> tuple[jax.Array, jax.Array]:
  scale = absmax_to_scale(ragged_absmax(x, num_valid_rows), qtype)
  xq = _xla_quantize_with_scale(x, scale, qtype)
  return xq, scale


def absmax_quantize(x: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None) -> tuple[jax.Array, jax.Array]:
  """Dynamically quantizes x with one scale from its first `num_valid_rows`.

  Rows past `num_valid_rows` are padding (possibly garbage) and do not affect
  the scale.

  Args:
    x: Array to quantize; rows are its leading dimension.
    qtype: Dtype to quantize to.
    num_valid_rows: Number of leading rows of x that hold real values. If None,
      all rows do.

  Returns:
    Tuple of (x / scale cast to `qtype`, f32 scale of shape [1, 1]). The scale
    is 1 if the valid rows are all zero.
  """
  if num_valid_rows is not None and _can_use_ragged_pallas(x):
    return jax.lax.platform_dependent(
        x,
        jnp.asarray(num_valid_rows, jnp.int32),
        tpu=lambda a, n: ops.ragged_absmax_quantize(a, qtype, n),
        default=lambda a, n: _xla_absmax_quantize(a, qtype, n),
    )
  return _xla_absmax_quantize(x, qtype, num_valid_rows)
