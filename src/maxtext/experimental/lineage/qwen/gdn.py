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

"""Pure-JAX Gated DeltaNet (GDN) mixer for Qwen MAX & Qwen 3.5 on TPU7x."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Protocol

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.layout import Layout  # pylint: disable=g-importing-member
from jax.experimental.layout import with_layout_constraint  # pylint: disable=g-importing-member
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen.config import QwenConfig
from maxtext.kernels.gdn.gdn_bwd_pallas import GdnPrecomputedMeta
from maxtext.kernels.gdn.gdn_bwd_pallas import gdn_decoupled_conv1d
from maxtext.kernels.gdn.gdn_bwd_pallas import precompute_gdn_meta

# Zero-host-offload HBM residual checkpoint names (`S(0)` HBM only;
# NEVER `S(5)` `pinned_host` offload).
GDN_SAVED_NAMES: tuple[str, ...] = (
    "gdn",
    "gdn_core_attn_out",
    "gdn_fwd_out",
    "gdn_fwd_conv",
    "gdn_qkv",
    "gdn_b",
    "gdn_a",
    "gdn_t_inv",
    "gdn_chunk_states",
    "gdn_m_local",
    "gdn_conv_state",
    "gdn_recurrent_state",
    "gdn_conv",
    "gdn_conv_out",
    "qkv_proj",
    "out_proj",
)

_ALLOWED_GRAD_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))


class GdnWeightCarrier(Protocol):
  """Structural protocol matching `GatheredGdnWeights` and `GdnWeights`."""

  pre_attn_norm: jax.Array
  in_proj_qkvz: Any
  in_proj_ba: jax.Array
  conv1d: jax.Array
  A_log: jax.Array  # pylint: disable=invalid-name
  dt_bias: jax.Array
  gdn_norm: jax.Array
  out_proj: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class QuantizedDenseWeightFp8:
  """Pre-quantized `float8_e4m3fn` weight codes, `float32` scale, and BF16/FP32 STE carrier."""

  codes: jax.Array
  scale: jax.Array
  w_ste: jax.Array | None = None


@jax.custom_vjp
def quantize_dense_weight_fp8(w: jax.Array) -> QuantizedDenseWeightFp8:
  """Quantizes dense `[K, ...]` weight to `float8_e4m3fn` (`gather8`) with BF16/FP32 STE bwd."""
  k_dim = w.shape[0]
  w_2d = jax.lax.stop_gradient(w.reshape(k_dim, -1))
  codes_2d, scale_2d = _quantize_axis_fp8(w_2d, axis=0, qtype=jnp.float8_e4m3fn)
  codes = jax.lax.stop_gradient(codes_2d.reshape(w.shape))
  scale = jax.lax.stop_gradient(scale_2d)
  grad_dt = (
      jnp.dtype(w.dtype)
      if jnp.dtype(w.dtype) in _ALLOWED_GRAD_DTYPES
      else jnp.dtype(jnp.bfloat16)
  )
  w_ste = jax.lax.broadcast(jnp.zeros((), dtype=grad_dt), w.shape)
  return QuantizedDenseWeightFp8(codes=codes, scale=scale, w_ste=w_ste)


def _quantize_dense_weight_fp8_fwd(
    w: jax.Array,
) -> tuple[QuantizedDenseWeightFp8, jax.Array]:
  return quantize_dense_weight_fp8(w), jnp.zeros((0,), dtype=w.dtype)


def _quantize_dense_weight_fp8_bwd(
    w_dtype_tag: jax.Array, g_q: QuantizedDenseWeightFp8
) -> tuple[jax.Array, ...]:
  if g_q.w_ste is None:
    raise ValueError(
        "QuantizedDenseWeightFp8 requires a non-None BF16/FP32 w_ste carrier"
        " for gradient propagation."
    )
  assert_bf16_or_fp32_grad(g_q.w_ste, op_name="quantize_dense_weight_fp8_bwd")
  return (g_q.w_ste.astype(w_dtype_tag.dtype),)


quantize_dense_weight_fp8.defvjp(
    _quantize_dense_weight_fp8_fwd, _quantize_dense_weight_fp8_bwd
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GdnWeights:
  """Standalone pure-JAX GDN weight PyTree (for isolated tests & benchmarks)."""

  pre_attn_norm: jax.Array
  in_proj_qkvz: jax.Array | QuantizedDenseWeightFp8
  in_proj_ba: jax.Array
  conv1d: jax.Array
  A_log: jax.Array  # pylint: disable=invalid-name
  dt_bias: jax.Array
  gdn_norm: jax.Array
  out_proj: jax.Array | QuantizedDenseWeightFp8


def is_quantized_weight_fp8(w: Any) -> bool:
  """Returns True if `w` is an FP8 `(codes, scale)` weight carrier."""
  return hasattr(w, "codes") and hasattr(w, "scale")


def assert_bf16_or_fp32_grad(grad: jax.Array, *, op_name: str) -> None:
  """Enforces Global Mixed-Precision Invariant (BF16/FP32 gradients for RS/AR)."""
  dt = jnp.dtype(grad.dtype)
  if dt not in _ALLOWED_GRAD_DTYPES:
    raise ValueError(
        f"{op_name} must emit bfloat16 or float32 gradients for downstream"
        f" ReduceScatter/AllReduce, got {dt}."
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def forward_layout_constraint(
    x: jax.Array, major_to_minor: tuple[int, ...]
) -> jax.Array:
  """Pins `with_layout_constraint` in forward only; identity in backward."""
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor))


def _fwd_layout_fwd(
    x: jax.Array, major_to_minor: tuple[int, ...]
) -> tuple[jax.Array, None]:
  return with_layout_constraint(x, Layout(major_to_minor=major_to_minor)), None


def _fwd_layout_bwd(
    major_to_minor: tuple[int, ...], _: None, cotangent: jax.Array
) -> tuple[jax.Array, ...]:
  del major_to_minor
  return (cotangent,)


forward_layout_constraint.defvjp(_fwd_layout_fwd, _fwd_layout_bwd)


def rms_norm_zero_centered(
    x: jax.Array, scale: jax.Array, eps: float = 1e-6
) -> jax.Array:
  """Qwen zero-centered RMSNorm: `(x / rms(x)) * (1.0 + scale)` in FP32."""
  orig_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + eps
  )
  normed = x_f32 * inv_rms * (1.0 + scale.astype(jnp.float32))
  return normed.astype(orig_dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def rms_norm_gated(
    x: jax.Array,
    gate_z: jax.Array,
    scale: jax.Array,
    eps: float = 1e-6,
    barrier_stages: bool = False,
) -> jax.Array:
  """Qwen gated RMSNorm: `rms_norm(x, scale) * silu(gate_z)` in FP32."""
  orig_dtype = x.dtype
  x_var = jax.lax.optimization_barrier(x).astype(jnp.float32)
  inv_rms = jax.lax.rsqrt(
      jnp.mean(jnp.square(x_var), axis=-1, keepdims=True) + eps
  )
  if barrier_stages:
    x_f, z_f = jax.lax.optimization_barrier((x, gate_z))
  else:
    x_f, z_f = x, gate_z
  normed = x_f.astype(jnp.float32) * inv_rms * scale.astype(jnp.float32)
  gated = normed * jax.nn.silu(z_f.astype(jnp.float32))
  return gated.astype(orig_dtype)


def _rms_norm_gated_fwd(
    x: jax.Array,
    gate_z: jax.Array,
    scale: jax.Array,
    eps: float,
    barrier_stages: bool,
):
  orig_dtype = x.dtype
  x_var = jax.lax.optimization_barrier(x).astype(jnp.float32)
  var_eps = jnp.mean(jnp.square(x_var), axis=-1, keepdims=True) + eps
  inv_rms = jax.lax.rsqrt(var_eps)
  if barrier_stages:
    x_f, z_f = jax.lax.optimization_barrier((x, gate_z))
  else:
    x_f, z_f = x, gate_z
  normed = x_f.astype(jnp.float32) * inv_rms * scale.astype(jnp.float32)
  gated = (normed * jax.nn.silu(z_f.astype(jnp.float32))).astype(orig_dtype)
  return gated, (x, gate_z, scale, inv_rms, var_eps)


def _rms_norm_gated_bwd(
    eps: float, barrier_stages: bool, res, g_out: jax.Array
):
  del eps
  x, gate_z, scale, inv_rms, var_eps = res
  w_f32 = scale.astype(jnp.float32)

  # Stage 1+2: reductions from BF16 inputs to tiny `d_sq` (`[B,S,H,1]`) and
  # `g_scale` (`scale.shape`), avoiding any materialized `f32[B,S,H,D]` buffer
  # while matching JAX's exact rsqrt/square VJP arithmetic order (0-ULP parity).
  scale_ndim = scale.ndim
  lead_axes = tuple(range(x.ndim - scale_ndim))
  bcast_axes = tuple(
      x.ndim - scale_ndim + i
      for i, s_dim in enumerate(scale.shape)
      if s_dim == 1 and x.shape[x.ndim - scale_ndim + i] != 1
  )
  reduce_axes = lead_axes + bcast_axes
  if barrier_stages:
    # Barrier-isolate Stage 1 (`d_inv_rms`, axis=-1) and Stage 2 (`g_scale`,
    # leading axes) so XLA cannot CSE `bf16->f32` converts across remat FWD and
    # the two incompatible reductions (which otherwise materializes 3x
    # `f32[B,S,H,D]` HBM buffers and triggers 3x FP32 layout copies).
    x_1, z_1, g_1 = jax.lax.optimization_barrier((x, gate_z, g_out))
    x_f32_1 = x_1.astype(jnp.float32)
    g_normed_1 = g_1.astype(jnp.float32) * jax.nn.silu(z_1.astype(jnp.float32))
    d_inv_rms = jnp.sum((g_normed_1 * w_f32) * x_f32_1, axis=-1, keepdims=True)
    d_var = d_inv_rms * (jnp.float32(-0.5) * (inv_rms / var_eps))
    d_sq = d_var / jnp.float32(x.shape[-1])

    x_2, z_2, g_2 = jax.lax.optimization_barrier((x, gate_z, g_out))
    x_hat_2 = x_2.astype(jnp.float32) * inv_rms
    g_normed_2 = g_2.astype(jnp.float32) * jax.nn.silu(z_2.astype(jnp.float32))
    g_scale = (
        jnp.sum(g_normed_2 * x_hat_2, axis=reduce_axes)
        .reshape(scale.shape)
        .astype(scale.dtype)
    )
  else:
    x_f32_r = x.astype(jnp.float32)
    x_hat_r = x_f32_r * inv_rms
    g_normed_r = g_out.astype(jnp.float32) * jax.nn.silu(
        gate_z.astype(jnp.float32)
    )
    d_inv_rms = jnp.sum((g_normed_r * w_f32) * x_f32_r, axis=-1, keepdims=True)
    d_var = d_inv_rms * (jnp.float32(-0.5) * (inv_rms / var_eps))
    d_sq = d_var / jnp.float32(x.shape[-1])
    g_scale = (
        jnp.sum(g_normed_r * x_hat_r, axis=reduce_axes)
        .reshape(scale.shape)
        .astype(scale.dtype)
    )

  # Stage 3: barrier-separated elementwise backward directly into BF16 `(g_x, g_z)`.
  x_b, z_b, g_b = jax.lax.optimization_barrier((x, gate_z, g_out))
  x_f32 = x_b.astype(jnp.float32)
  z_f32 = z_b.astype(jnp.float32)
  sig_z = jax.nn.sigmoid(z_f32)
  silu_z = z_f32 * sig_z
  g_f32 = g_b.astype(jnp.float32)
  g_x_hat = (g_f32 * silu_z) * w_f32
  g_x = (g_x_hat * inv_rms + d_sq * (jnp.float32(2.0) * x_f32)).astype(x.dtype)
  g_silu = g_f32 * ((x_f32 * inv_rms) * w_f32)
  g_z = (
      g_silu * sig_z + (g_silu * z_f32) * (sig_z * (jnp.float32(1.0) - sig_z))
  ).astype(gate_z.dtype)
  return g_x, g_z, g_scale


rms_norm_gated.defvjp(_rms_norm_gated_fwd, _rms_norm_gated_bwd)


def _quantize_axis_fp8(
    x: jax.Array,
    *,
    axis: int,
    qtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
  """Channelwise/rowwise dynamic FP8 quantization along `axis` (never NaN/Inf)."""
  qdt = jnp.dtype(qtype)
  qmax = float(jnp.finfo(qdt).max)
  x_f32 = x.astype(jnp.float32)
  amax = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True)
  scale = jnp.where(amax > 0.0, amax / qmax, jnp.ones_like(amax))
  codes = jnp.clip(x_f32 / scale, -qmax, qmax).astype(qdt)
  return codes, scale


@jax.custom_vjp
def _dense_fp8_dot_2d_array(x_2d: jax.Array, w_2d: jax.Array) -> jax.Array:
  """Native MXU `fp8_e4m3fn x fp8_e4m3fn` 2D matmul (`convolution` fusion)."""
  x_q, x_s = _quantize_axis_fp8(x_2d, axis=1, qtype=jnp.float8_e4m3fn)
  w_q, w_s = _quantize_axis_fp8(w_2d, axis=0, qtype=jnp.float8_e4m3fn)
  acc = jax.lax.dot_general(
      x_q,
      w_q,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  return (acc * x_s * w_s).astype(x_2d.dtype)


def _dense_fp8_dot_2d_array_fwd(
    x_2d: jax.Array, w_2d: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
  """Forward VJP rule for native MXU FP8 2D matmul."""
  x_q, x_s = _quantize_axis_fp8(x_2d, axis=1, qtype=jnp.float8_e4m3fn)
  w_q, w_s = _quantize_axis_fp8(w_2d, axis=0, qtype=jnp.float8_e4m3fn)
  acc = jax.lax.dot_general(
      x_q,
      w_q,
      (((1,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  out = (acc * x_s * w_s).astype(x_2d.dtype)
  return out, (x_2d, w_q, w_s, jnp.zeros((0,), dtype=w_2d.dtype))


def _dense_fp8_dot_2d_array_bwd(
    res: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    dy_2d: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Backward VJP rule for native MXU FP8 dgrad & wgrad."""
  x_2d, w_q, w_s, w_dtype_tag = res
  x_dtype = x_2d.dtype
  w_dtype = jnp.dtype(w_dtype_tag.dtype)
  # dgrad: fold [1, N] w_s into dy before e5m2 row quantization -> pure fp8 MXU!
  dy_w = dy_2d.astype(jnp.float32) * w_s
  dy_dgrad_q, dy_dgrad_s = _quantize_axis_fp8(
      dy_w, axis=1, qtype=jnp.float8_e5m2
  )
  dx_acc = jax.lax.dot_general(
      dy_dgrad_q,
      w_q,
      (((1,), (1,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  dx = (dx_acc * dy_dgrad_s).astype(x_dtype)

  # wgrad: channelwise e4m3fn x e5m2 along M -> BF16/FP32 dw for SC1.
  x_wgrad_q, x_wgrad_s = _quantize_axis_fp8(
      x_2d, axis=0, qtype=jnp.float8_e4m3fn
  )
  dy_wgrad_q, dy_wgrad_s = _quantize_axis_fp8(
      dy_2d, axis=0, qtype=jnp.float8_e5m2
  )
  dw_acc = jax.lax.dot_general(
      x_wgrad_q,
      dy_wgrad_q,
      (((0,), (0,)), ((), ())),
      preferred_element_type=jnp.float32,
  )
  grad_dt = (
      w_dtype if w_dtype in _ALLOWED_GRAD_DTYPES else jnp.dtype(jnp.bfloat16)
  )
  dw = (dw_acc * jnp.swapaxes(x_wgrad_s, 0, 1) * dy_wgrad_s).astype(grad_dt)
  assert_bf16_or_fp32_grad(dx, op_name="dense_fp8_matmul_dx")
  assert_bf16_or_fp32_grad(dw, op_name="dense_fp8_matmul_dw")
  return dx, dw


_dense_fp8_dot_2d_array.defvjp(
    _dense_fp8_dot_2d_array_fwd, _dense_fp8_dot_2d_array_bwd
)


def _dense_fp8_dot_2d_carrier(
    x_2d: jax.Array,
    w_codes: jax.Array,
    w_scale: jax.Array,
    w_ste: jax.Array | None,
) -> jax.Array:
  """Native MXU `fp8_e4m3fn x fp8_e4m3fn` 2D matmul for pre-quantized `gather8` weights."""
  if jnp.dtype(w_codes.dtype) != jnp.dtype(jnp.float8_e4m3fn):
    raise ValueError(
        f"Expected float8_e4m3fn weight codes, got {w_codes.dtype}."
    )
  if w_ste is None:
    x_q, x_s = _quantize_axis_fp8(x_2d, axis=1, qtype=jnp.float8_e4m3fn)
    ws_bcast = (
        w_scale.reshape(1, -1) if w_scale.size > 1 else w_scale.reshape(1, 1)
    )
    acc = jax.lax.dot_general(
        x_q,
        jax.lax.stop_gradient(w_codes),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (acc * x_s * jax.lax.stop_gradient(ws_bcast)).astype(x_2d.dtype)

  @jax.custom_vjp
  def _fn(
      x_in: jax.Array, wc: jax.Array, ws: jax.Array, ste: jax.Array
  ) -> jax.Array:
    del ste
    x_q, x_s = _quantize_axis_fp8(x_in, axis=1, qtype=jnp.float8_e4m3fn)
    ws_bcast = ws.reshape(1, -1) if ws.size > 1 else ws.reshape(1, 1)
    acc = jax.lax.dot_general(
        x_q,
        wc,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return (acc * x_s * ws_bcast).astype(x_in.dtype)

  def _fwd(x_in: jax.Array, wc: jax.Array, ws: jax.Array, ste: jax.Array):
    out = _fn(x_in, wc, ws, ste)
    return out, (x_in, wc, ws, jnp.zeros((0,), dtype=ste.dtype))

  def _bwd(res, dy_2d: jax.Array):
    x_in, wc, ws, ste_tag = res
    ste_dt = jnp.dtype(ste_tag.dtype)
    ws_bcast = ws.reshape(1, -1) if ws.size > 1 else ws.reshape(1, 1)
    dy_w = dy_2d.astype(jnp.float32) * ws_bcast.astype(jnp.float32)
    dy_dgrad_q, dy_dgrad_s = _quantize_axis_fp8(
        dy_w, axis=1, qtype=jnp.float8_e5m2
    )
    dx_acc = jax.lax.dot_general(
        dy_dgrad_q,
        wc,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    dx = (dx_acc * dy_dgrad_s).astype(x_in.dtype)

    x_wgrad_q, x_wgrad_s = _quantize_axis_fp8(
        x_in, axis=0, qtype=jnp.float8_e4m3fn
    )
    dy_wgrad_q, dy_wgrad_s = _quantize_axis_fp8(
        dy_2d, axis=0, qtype=jnp.float8_e5m2
    )
    dw_acc = jax.lax.dot_general(
        x_wgrad_q,
        dy_wgrad_q,
        (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    grad_dt = (
        ste_dt if ste_dt in _ALLOWED_GRAD_DTYPES else jnp.dtype(jnp.bfloat16)
    )
    dw = (dw_acc * jnp.swapaxes(x_wgrad_s, 0, 1) * dy_wgrad_s).astype(grad_dt)
    assert_bf16_or_fp32_grad(dx, op_name="dense_fp8_carrier_dx")
    assert_bf16_or_fp32_grad(dw, op_name="dense_fp8_carrier_dw")
    return dx, jnp.zeros_like(wc), jnp.zeros_like(ws), dw

  _fn.defvjp(_fwd, _bwd)
  return _fn(x_2d, w_codes, w_scale, w_ste)


def dense_linear(
    x: jax.Array,
    w: Any,
    *,
    use_fp8: bool = False,
) -> jax.Array:
  """Executes `[..., K] @ [K, N]` in BF16 or native MXU `fp8_e4m3fn` (`convolution` fusion)."""
  orig_prefix = x.shape[:-1]
  k_dim = x.shape[-1]
  x_2d = x.reshape(-1, k_dim)
  if is_quantized_weight_fp8(w):
    w_codes_2d = w.codes.reshape(k_dim, -1)
    w_ste_raw = getattr(w, "w_ste", None)
    w_ste_2d = w_ste_raw.reshape(k_dim, -1) if w_ste_raw is not None else None
    y_2d = _dense_fp8_dot_2d_carrier(x_2d, w_codes_2d, w.scale, w_ste_2d)
    return y_2d.reshape(*orig_prefix, w_codes_2d.shape[1])

  w_2d = w.reshape(k_dim, -1)
  if use_fp8:
    y_2d = _dense_fp8_dot_2d_array(x_2d, w_2d)
  else:
    y_2d = jnp.matmul(x_2d, w_2d)
  return y_2d.reshape(*orig_prefix, w_2d.shape[1])


def _pure_jax_gdn_cpu_fallback(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
) -> jax.Array:
  """Zero-Flax pure-JAX segmented causal Conv1D + Gated Delta Rule reference."""
  bsz, seq_len, _ = qkv.shape
  k_conv = cfg.gdn_conv_kernel_size
  qkv_f32 = qkv.astype(jnp.float32)
  w_f32 = conv_weight.astype(jnp.float32)  # [K_conv, 1, qkv_dim]

  conv_acc = jnp.zeros_like(qkv_f32)
  for lag in range(k_conv):
    tap_w = w_f32[k_conv - 1 - lag, 0, :]
    if lag == 0:
      shifted = qkv_f32
      same_seg = segment_ids > 0
    else:
      shifted = jnp.pad(qkv_f32[:, :-lag, :], ((0, 0), (lag, 0), (0, 0)))
      seg_shifted = jnp.pad(
          segment_ids[:, :-lag], ((0, 0), (lag, 0)), constant_values=-1
      )
      same_seg = (segment_ids > 0) & (segment_ids == seg_shifted)
    conv_acc = conv_acc + jnp.where(
        same_seg[:, :, None], shifted * tap_w[None, None, :], 0.0
    )
  qkv_conv = jax.nn.silu(conv_acc)

  key_dim = cfg.gdn_key_dim
  q_c, k_c, v_c = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)
  q = q_c.reshape(bsz, seq_len, cfg.gdn_num_k_heads, cfg.gdn_head_k_dim)
  k = k_c.reshape(bsz, seq_len, cfg.gdn_num_k_heads, cfg.gdn_head_k_dim)
  v = v_c.reshape(bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim)

  q_norm_sq = jnp.sum(jnp.square(q), axis=-1, keepdims=True)
  k_norm_sq = jnp.sum(jnp.square(k), axis=-1, keepdims=True)
  q = jnp.where(q_norm_sq == 0.0, 0.0, q * jax.lax.rsqrt(q_norm_sq + 1e-6)) * (
      cfg.gdn_head_k_dim**-0.5
  )
  k = jnp.where(k_norm_sq == 0.0, 0.0, k * jax.lax.rsqrt(k_norm_sq + 1e-6))

  repeats = cfg.gdn_num_v_heads // cfg.gdn_num_k_heads
  if repeats > 1:
    q = jnp.repeat(q, repeats, axis=2)
    k = jnp.repeat(k, repeats, axis=2)

  beta = jax.nn.sigmoid(b.astype(jnp.float32))
  g = -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(
      a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
  )

  init_state = jnp.zeros(
      (bsz, cfg.gdn_num_v_heads, cfg.gdn_head_k_dim, cfg.gdn_head_v_dim),
      dtype=jnp.float32,
  )
  init_prev_seg = jnp.full((bsz,), -1, dtype=jnp.int32)

  def _step(carry, tok_inputs):
    state, prev_seg = carry
    q_t, k_t, v_t, beta_t, g_t, seg_t = tok_inputs
    valid = seg_t > 0
    same_doc = valid & (seg_t == prev_seg)
    s_in = jnp.where(same_doc[:, None, None, None], state, 0.0)
    s_dec = s_in * jnp.exp(g_t)[:, :, None, None]
    kv_mem = jnp.einsum(
        "bhk,bhkv->bhv", k_t, s_dec, precision=jax.lax.Precision.HIGHEST
    )
    delta = (v_t - kv_mem) * beta_t[:, :, None]
    s_next = s_dec + jnp.einsum(
        "bhk,bhv->bhkv", k_t, delta, precision=jax.lax.Precision.HIGHEST
    )
    s_next = jnp.where(valid[:, None, None, None], s_next, 0.0)
    out_t = jnp.where(
        valid[:, None, None],
        jnp.einsum(
            "bhk,bhkv->bhv", q_t, s_next, precision=jax.lax.Precision.HIGHEST
        ),
        0.0,
    )
    return (s_next, seg_t), out_t

  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(beta, 0, 1),
      jnp.swapaxes(g, 0, 1),
      jnp.swapaxes(segment_ids, 0, 1),
  )
  _, outs_sbhd = jax.lax.scan(_step, (init_state, init_prev_seg), xs)
  return jnp.swapaxes(outs_sbhd, 0, 1).astype(qkv.dtype)


def gdn_mixer(
    x: jax.Array,
    w: GdnWeightCarrier,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    conv_natural_layout: bool = True,
    precomputed_meta: GdnPrecomputedMeta | None = None,
) -> jax.Array:
  """Executes Qwen Gated DeltaNet (`x`: `[B_mb, S, D]`) and returns `x + gdn_out`.

  Supports both Qwen MAX (`D=8192, H_k=16, H_v=128, D_k=D_v=128`) and Qwen 3.5
  (`D=4096, H_k=16, H_v=64, D_k=D_v=128`) with zero-host-offload HBM residual
  checkpoint tags (`qkv_proj`, `gdn_*`, `out_proj`), `token_minor` layout
  constraints, and optional P1 native MXU FP8 `in_proj_qkvz` / `out_proj` dots.

  Args:
    x: Input activations of shape `[B_mb, S, emb_dim]`.
    w: GDN weights PyTree matching `GdnWeightCarrier`.
    segment_ids: Packed sequence segment IDs of shape `[B_mb, S]`.
    cfg: Frozen `QwenConfig` specifying dimensions and kernel options.
    conv_natural_layout: Whether to use natural channel-minor conv1d bwd layout
      in Pallas GDN kernels.

  Returns:
    Output activations `x + gdn_out` of shape `[B_mb, S, emb_dim]`.
  """
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_gdn)
  normed = rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)

  qkvz = dense_linear(normed, w.in_proj_qkvz, use_fp8=use_fp8)
  qkvz = checkpoint_name(qkvz, "qkv_proj")
  qkv = qkvz[..., : cfg.gdn_qkv_dim]
  z = qkvz[..., cfg.gdn_qkv_dim :].reshape(
      bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim
  )

  ba = jnp.matmul(normed, w.in_proj_ba)
  b = ba[..., : cfg.gdn_num_v_heads]
  a = ba[..., cfg.gdn_num_v_heads :]

  if cfg.use_pallas_kernels:
    is_bf16 = jnp.dtype(qkv.dtype) == jnp.dtype(jnp.bfloat16)
    compute_dt = jnp.bfloat16 if is_bf16 else jnp.float32
    core_out, _ = gdn_decoupled_conv1d(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=w.conv1d,
        conv_bias=None,
        a_log=w.A_log,
        dt_bias=w.dt_bias,
        conv_state=None,
        recurrent_state=None,
        num_k_heads=cfg.gdn_num_k_heads,
        num_v_heads=cfg.gdn_num_v_heads,
        head_k_dim=cfg.gdn_head_k_dim,
        head_v_dim=cfg.gdn_head_v_dim,
        conv_kernel_size=cfg.gdn_conv_kernel_size,
        chunk_size=cfg.gdn_chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=compute_dt,
        cp_axis_name=None,
        segment_ids=segment_ids,
        conv_input_in_qkv_dtype=True,
        conv_dense_token_stack=True,
        pass1_states_and_bf16_qkv=is_bf16,
        bwd_natural_layout=cfg.gdn_bwd_natural_layout,
        conv_natural_layout=(conv_natural_layout and is_bf16),
        fwd_head_group_size=cfg.gdn_fwd_head_group_size,
        precomputed_meta=precomputed_meta,
        keep_bf16_states=cfg.gdn_keep_bf16_states,
        recompute_t_inv_bwd=cfg.gdn_recompute_t_inv,
    )
    if cfg.gdn_norm_layout_mode in ("token_minor", "bf16_z_token_minor"):
      # Pin bf16 GDN output token-minor (0, 2, 3, 1) ahead of fp32 gated norm.
      core_out = forward_layout_constraint(core_out, (0, 2, 3, 1))
      if cfg.gdn_norm_layout_mode == "bf16_z_token_minor":
        z = forward_layout_constraint(z, (0, 2, 3, 1))
  else:
    core_out = _pure_jax_gdn_cpu_fallback(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=w.conv1d,
        a_log=w.A_log,
        dt_bias=w.dt_bias,
        segment_ids=segment_ids,
        cfg=cfg,
    )

  gated_normed = rms_norm_gated(
      core_out,
      z,
      w.gdn_norm,
      cfg.norm_eps,
      cfg.gdn_rms_norm_barrier_stages,
  )
  flat = gated_normed.reshape(bsz, seq_len, cfg.gdn_value_dim)
  if cfg.use_pallas_kernels and cfg.gdn_norm_layout_mode != "natural":
    flat = forward_layout_constraint(flat, (2, 0, 1))
  proj = dense_linear(flat, w.out_proj, use_fp8=use_fp8)
  proj = checkpoint_name(proj, "out_proj")
  return x + proj


def gdn_mixer_fwd_save_res(
    x: jax.Array,
    w: GdnWeightCarrier,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    conv_natural_layout: bool = True,
    precomputed_meta: GdnPrecomputedMeta | None = None,
) -> tuple[jax.Array, tuple[Any, jax.Array, jax.Array]]:
  """Forward `gdn_mixer` returning `(y, (vjp_core, core_out, z))` with zero big-weight capture."""
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_gdn)
  normed = rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)

  qkvz = dense_linear(normed, w.in_proj_qkvz, use_fp8=use_fp8)
  qkv = qkvz[..., : cfg.gdn_qkv_dim]
  z = qkvz[..., cfg.gdn_qkv_dim :].reshape(
      bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim
  )

  ba = jnp.matmul(normed, w.in_proj_ba)
  b = ba[..., : cfg.gdn_num_v_heads]
  a = ba[..., cfg.gdn_num_v_heads :]

  def _gdn_core(qkv_in, b_in, a_in, conv_w, a_log_w, dt_bias_w):
    if cfg.use_pallas_kernels:
      is_bf16 = jnp.dtype(qkv_in.dtype) == jnp.dtype(jnp.bfloat16)
      compute_dt = jnp.bfloat16 if is_bf16 else jnp.float32
      core_o, _ = gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=conv_w,
          conv_bias=None,
          a_log=a_log_w,
          dt_bias=dt_bias_w,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=cfg.gdn_num_k_heads,
          num_v_heads=cfg.gdn_num_v_heads,
          head_k_dim=cfg.gdn_head_k_dim,
          head_v_dim=cfg.gdn_head_v_dim,
          conv_kernel_size=cfg.gdn_conv_kernel_size,
          chunk_size=cfg.gdn_chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=compute_dt,
          cp_axis_name=None,
          segment_ids=segment_ids,
          conv_input_in_qkv_dtype=True,
          conv_dense_token_stack=True,
          pass1_states_and_bf16_qkv=is_bf16,
          bwd_natural_layout=cfg.gdn_bwd_natural_layout,
          conv_natural_layout=(conv_natural_layout and is_bf16),
          fwd_head_group_size=cfg.gdn_fwd_head_group_size,
          precomputed_meta=precomputed_meta,
          keep_bf16_states=cfg.gdn_keep_bf16_states,
          recompute_t_inv_bwd=cfg.gdn_recompute_t_inv,
      )
      return core_o
    return _pure_jax_gdn_cpu_fallback(
        qkv=qkv_in,
        b=b_in,
        a=a_in,
        conv_weight=conv_w,
        a_log=a_log_w,
        dt_bias=dt_bias_w,
        segment_ids=segment_ids,
        cfg=cfg,
    )

  core_out, vjp_core = jax.vjp(
      _gdn_core, qkv, b, a, w.conv1d, w.A_log, w.dt_bias
  )
  if cfg.use_pallas_kernels and cfg.gdn_norm_layout_mode in (
      "token_minor",
      "bf16_z_token_minor",
  ):
    core_out = forward_layout_constraint(core_out, (0, 2, 3, 1))
    if cfg.gdn_norm_layout_mode == "bf16_z_token_minor":
      z = forward_layout_constraint(z, (0, 2, 3, 1))

  gated_normed = rms_norm_gated(
      core_out,
      z,
      w.gdn_norm,
      cfg.norm_eps,
      cfg.gdn_rms_norm_barrier_stages,
  )
  flat = gated_normed.reshape(bsz, seq_len, cfg.gdn_value_dim)
  if cfg.use_pallas_kernels and cfg.gdn_norm_layout_mode != "natural":
    flat = forward_layout_constraint(flat, (2, 0, 1))
  proj = dense_linear(flat, w.out_proj, use_fp8=use_fp8)
  return x + proj, (vjp_core, core_out, z)


def gdn_mixer_bwd_from_res(
    g_y: jax.Array,
    x: jax.Array,
    saved_res: tuple[Any, jax.Array, jax.Array],
    w: GdnWeightCarrier,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    conv_natural_layout: bool = True,
    precomputed_meta: GdnPrecomputedMeta | None = None,
) -> tuple[jax.Array, GdnWeights]:
  """Backward `gdn_mixer` reusing saved GDN residuals (zero GDN/in_proj/out_proj FWD remat)."""
  del segment_ids, conv_natural_layout, precomputed_meta
  vjp_core, core_out, z = saved_res
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_gdn)

  def _post_gdn(core_o, z_in, gdn_norm_w, out_proj_w):
    gated_normed = rms_norm_gated(
        core_o,
        z_in,
        gdn_norm_w,
        cfg.norm_eps,
        cfg.gdn_rms_norm_barrier_stages,
    )
    flat = gated_normed.reshape(bsz, seq_len, cfg.gdn_value_dim)
    if cfg.use_pallas_kernels and cfg.gdn_norm_layout_mode != "natural":
      flat = forward_layout_constraint(flat, (2, 0, 1))
    return dense_linear(flat, out_proj_w, use_fp8=use_fp8)

  def _pre_gdn(x_in, pre_norm_w, qkvz_w, ba_w):
    normed = rms_norm_zero_centered(x_in, pre_norm_w, cfg.norm_eps)
    qkvz = dense_linear(normed, qkvz_w, use_fp8=use_fp8)
    qkv = qkvz[..., : cfg.gdn_qkv_dim]
    z_out = qkvz[..., cfg.gdn_qkv_dim :].reshape(
        bsz, seq_len, cfg.gdn_num_v_heads, cfg.gdn_head_v_dim
    )
    ba = jnp.matmul(normed, ba_w)
    b_out = ba[..., : cfg.gdn_num_v_heads]
    a_out = ba[..., cfg.gdn_num_v_heads :]
    return qkv, z_out, b_out, a_out

  _, vjp_post = jax.vjp(_post_gdn, core_out, z, w.gdn_norm, w.out_proj)
  g_core_out, g_z, g_gdn_norm, g_out_proj = vjp_post(g_y)
  g_qkv, g_b, g_a, g_conv1d, g_a_log, g_dt_bias = vjp_core(g_core_out)
  _, vjp_pre = jax.vjp(
      _pre_gdn, x, w.pre_attn_norm, w.in_proj_qkvz, w.in_proj_ba
  )
  g_x_pre, g_pre_norm, g_qkvz_w, g_ba_w = vjp_pre((g_qkv, g_z, g_b, g_a))
  return (
      g_y + g_x_pre,
      GdnWeights(
          pre_attn_norm=g_pre_norm,
          in_proj_qkvz=g_qkvz_w,
          in_proj_ba=g_ba_w,
          conv1d=g_conv1d,
          A_log=g_a_log,
          dt_bias=g_dt_bias,
          gdn_norm=g_gdn_norm,
          out_proj=g_out_proj,
      ),
  )


def build_mixer_remat_policy(
    cfg: QwenConfig, *, save_qkv_proj_hbm: bool = False
) -> Any:
  """Builds zero-host-offload (`S(0)` HBM only) checkpoint policy for GDN & Attention."""
  if (
      cfg.offload_decoder_input
      or cfg.offload_attn_context
      or cfg.offload_query_proj
      or cfg.offload_out_proj
      or cfg.offload_gdn_qkv
  ):
    raise ValueError(
        "Zero-host-offload invariant violated: S(5) pinned_host activation"
        " offload flags must remain False."
    )
  save_names: list[str] = ["decoder_layer_input"]
  if cfg.save_key_value_proj:
    save_names.extend(["key_proj", "value_proj"])
  if cfg.save_gdn_residuals:
    save_names.extend(GDN_SAVED_NAMES)
  if save_qkv_proj_hbm or cfg.hbm_save_offload_targets:
    if "qkv_proj" not in save_names:
      save_names.append("qkv_proj")
  if cfg.save_attn_context_hbm or cfg.hbm_save_offload_targets:
    if "context" not in save_names:
      save_names.append("context")
  if cfg.save_query_proj_hbm or cfg.hbm_save_offload_targets:
    if "query_proj" not in save_names:
      save_names.append("query_proj")
  if cfg.save_out_proj_hbm or cfg.hbm_save_offload_targets:
    if "out_proj" not in save_names:
      save_names.append("out_proj")
  return jax.checkpoint_policies.save_only_these_names(*save_names)


__all__ = [
    "GDN_SAVED_NAMES",
    "GdnWeightCarrier",
    "GdnWeights",
    "QuantizedDenseWeightFp8",
    "_pure_jax_gdn_cpu_fallback",
    "assert_bf16_or_fp32_grad",
    "build_mixer_remat_policy",
    "dense_linear",
    "forward_layout_constraint",
    "gdn_mixer",
    "gdn_mixer_bwd_from_res",
    "gdn_mixer_fwd_save_res",
    "is_quantized_weight_fp8",
    "quantize_dense_weight_fp8",
    "rms_norm_gated",
    "rms_norm_zero_centered",
]
