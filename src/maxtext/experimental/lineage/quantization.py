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
class QuantConfig:
  """Model-wide quantization config. `None` fields stay unquantized.

  Attributes:
    routed_experts: Quantization of the sparse layers' routed-expert GMMs,
      including the token all-gather that feeds them.
    mla: Quantization of the MLA projections (q down, kv down, q up, v up and
      out; k up stays unquantized) in all layers.
  """

  routed_experts: GmmQuantRule | None = None
  mla: GmmQuantRule | None = None


FP8_FULL = QuantConfig(routed_experts=GmmQuantRule(), mla=GmmQuantRule())


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


def absmax_quantize(
    x: jax.Array, qtype: Any, num_valid_rows: jax.Array | int | None = None
) -> tuple[jax.Array, jax.Array]:
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
