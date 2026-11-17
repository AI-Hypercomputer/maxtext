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

"""Meta Muse-Spark (`MuseSparkForConditionalGeneration` text backbone) decoder layers.

Reference: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B
(`transformers.models.muse_spark.modeling_muse_spark`).
"""

# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import math
from typing import Any

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.common.common_types import (
    Array,
    AttentionType,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    DType,
    D_KV,
    HEAD,
    KV_HEAD,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    ShardMode,
    get_weight_dtype,
)
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.ragged import ragged_gather_tc
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils import sharding
from maxtext.utils.sharding import (
    create_sharding,
    get_logical_axis_rules,
    logical_to_mesh_axes,
    mesh_axes_for_dim,
    remove_incompatible_mesh_axes_from_partition_spec,
)


# ------------------------------------------------------------------------------
# Layer Schedule & Scan Layout Helpers
# ------------------------------------------------------------------------------


def is_full_attention_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses global NoPE attention instead of sliding-window RoPE attention.

  Matches `MuseSparkTextConfig`:
    `layer_types[i] = "full_attention" if (num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"`
    `layer_rope_theta[i] = 0 if (num_hidden_layers - 1 - i) % 4 == 0 else rope_theta`
  """
  return (config.num_decoder_layers - 1 - layer_idx) % config.inhomogeneous_layer_cycle_interval == 0


def is_moe_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses the bottlenecked Sparse MoE block."""
  return config.num_experts > 1 and layer_idx >= config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int]:
  """Returns `(num_prefix_layers, num_blocks)` for `scan_layers=True`.

  Because `(num_layers - 1 - i) % cycle == 0` places `full_attention` at the last sub-layer
  of every `cycle`-layer block aligned with the end of the stack (`i % cycle == num_layers % cycle`),
  unrolling the leading `num_prefix = num_layers % cycle` layers leaves an exact integer number
  of homogeneous `[sliding, ..., sliding, full]` MoE cycles of length `cycle` with no trailing remainder.
  If `first_num_dense_layers > num_layers % cycle`, whole cycles overlapping the dense prefix
  are also unrolled into `num_prefix`.
  """
  cycle = config.inhomogeneous_layer_cycle_interval
  n_layers = config.num_decoder_layers
  if config.num_experts <= 1 or cycle <= 0:
    return n_layers, 0
  align = n_layers % cycle
  if config.first_num_dense_layers <= align:
    num_prefix = align
  else:
    extra = config.first_num_dense_layers - align
    num_prefix = align + ((extra + cycle - 1) // cycle) * cycle
  num_prefix = min(n_layers, num_prefix)
  num_blocks = (n_layers - num_prefix) // cycle
  return num_prefix, num_blocks


# ------------------------------------------------------------------------------
# Normalization & Residual Gate Modules
# ------------------------------------------------------------------------------


def _rms_normalize(x_f32: jax.Array, epsilon: float, axis: int | tuple[int, ...] = -1) -> jax.Array:
  """Returns `x * rsqrt(mean(x^2) + epsilon)` over `axis`, in float32."""
  mean2 = jnp.mean(jax.lax.square(x_f32), axis=axis, keepdims=True)
  return x_f32 * jax.lax.rsqrt(mean2 + epsilon)


def _centered_scale_fp32(scale: jax.Array) -> jax.Array:
  """Computes `(1.0 + scale).float()` in `scale.dtype` first, matching PyTorch `(1.0 + weight).float()`."""
  return jnp.asarray(jnp.asarray(1.0, dtype=scale.dtype) + scale, jnp.float32)


def _var_eps_transpose_dx(
    d_var_eps: jax.Array, x_f32: jax.Array, axis: int | tuple[int, ...] = -1
) -> jax.Array:
  """Exact transpose of `var_eps = jnp.mean(jax.lax.square(x_f32), axis=axis, keepdims=True) + eps`."""
  if isinstance(axis, tuple):
    denom = math.prod(x_f32.shape[a] for a in axis)
  else:
    denom = x_f32.shape[axis]
  d_mean2 = jax.lax.div(d_var_eps, jnp.asarray(denom, jnp.float32))
  return d_mean2 * (2.0 * x_f32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _rms_norm_pow_no_scale_save_rstd(x: jax.Array, epsilon: float) -> jax.Array:
  """Bit-exact `MuseSparkRMSNorm(with_scale=False)` saving `(x, rstd, var_eps)` instead of float32 `[..., D]`."""
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jnp.power(var_eps, -0.5)
  return jnp.asarray(x_f32 * rstd, x.dtype)


def _rms_norm_pow_no_scale_save_rstd_fwd(x: jax.Array, epsilon: float):
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jnp.power(var_eps, -0.5)
  return jnp.asarray(x_f32 * rstd, x.dtype), (x, rstd, var_eps)


def _rms_norm_pow_no_scale_save_rstd_bwd(epsilon: float, res, g: jax.Array):
  del epsilon
  x, rstd, var_eps = res
  x_f32 = jnp.asarray(x, jnp.float32)
  _, vjp_tail = jax.vjp(lambda xf, r: jnp.asarray(xf * r, x.dtype), x_f32, rstd)
  dx_1, d_rstd = vjp_tail(g)
  _, vjp_pow = jax.vjp(lambda v: jnp.power(v, -0.5), var_eps)
  (d_var_eps,) = vjp_pow(d_rstd)
  dx_2 = _var_eps_transpose_dx(d_var_eps, x_f32)
  return (jnp.asarray(dx_1 + dx_2, x.dtype),)


_rms_norm_pow_no_scale_save_rstd.defvjp(
    _rms_norm_pow_no_scale_save_rstd_fwd, _rms_norm_pow_no_scale_save_rstd_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _rms_norm_pow_with_scale_save_rstd(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """Bit-exact `MuseSparkRMSNorm(with_scale=True)` saving `(x, scale, rstd, var_eps)`."""
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jnp.power(var_eps, -0.5)
  return jnp.asarray((x_f32 * rstd) * jnp.asarray(scale, jnp.float32), x.dtype)


def _rms_norm_pow_with_scale_save_rstd_fwd(x: jax.Array, scale: jax.Array, epsilon: float):
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jnp.power(var_eps, -0.5)
  out = jnp.asarray((x_f32 * rstd) * jnp.asarray(scale, jnp.float32), x.dtype)
  return out, (x, scale, rstd, var_eps)


def _rms_norm_pow_with_scale_save_rstd_bwd(epsilon: float, res, g: jax.Array):
  del epsilon
  x, scale, rstd, var_eps = res
  x_f32 = jnp.asarray(x, jnp.float32)
  _, vjp_tail = jax.vjp(
      lambda xf, r, s: jnp.asarray((xf * r) * jnp.asarray(s, jnp.float32), x.dtype),
      x_f32,
      rstd,
      scale,
  )
  dx_1, d_rstd, d_scale = vjp_tail(g)
  _, vjp_pow = jax.vjp(lambda v: jnp.power(v, -0.5), var_eps)
  (d_var_eps,) = vjp_pow(d_rstd)
  dx_2 = _var_eps_transpose_dx(d_var_eps, x_f32)
  return jnp.asarray(dx_1 + dx_2, x.dtype), d_scale


_rms_norm_pow_with_scale_save_rstd.defvjp(
    _rms_norm_pow_with_scale_save_rstd_fwd, _rms_norm_pow_with_scale_save_rstd_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _centered_rms_norm_save_rstd(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """Bit-exact `MuseSparkCenteredRMSNorm` saving `(x, scale, rstd, var_eps)` without backward variance reduction."""
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jax.lax.rsqrt(var_eps)
  return jnp.asarray((x_f32 * rstd) * _centered_scale_fp32(scale), x.dtype)


def _centered_rms_norm_save_rstd_fwd(x: jax.Array, scale: jax.Array, epsilon: float):
  x_f32 = jnp.asarray(x, jnp.float32)
  var_eps = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + epsilon
  rstd = jax.lax.rsqrt(var_eps)
  out = jnp.asarray((x_f32 * rstd) * _centered_scale_fp32(scale), x.dtype)
  return out, (x, scale, rstd, var_eps)


def _centered_rms_norm_save_rstd_bwd(epsilon: float, res, g: jax.Array):
  del epsilon
  x, scale, rstd, var_eps = res
  x_f32 = jnp.asarray(x, jnp.float32)
  _, vjp_tail = jax.vjp(
      lambda xf, r, s: jnp.asarray((xf * r) * _centered_scale_fp32(s), x.dtype),
      x_f32,
      rstd,
      scale,
  )
  dx_1, d_rstd, d_scale = vjp_tail(g)
  d_var_eps = d_rstd * (-0.5 * jax.lax.div(rstd, var_eps))
  dx_2 = _var_eps_transpose_dx(d_var_eps, x_f32)
  return jnp.asarray(dx_1 + dx_2, x.dtype), d_scale


_centered_rms_norm_save_rstd.defvjp(_centered_rms_norm_save_rstd_fwd, _centered_rms_norm_save_rstd_bwd)


def _is_3d_tiled_expert_tensor(x: jax.Array, scale: jax.Array) -> bool:
  return x.ndim == 3 and scale.ndim == 1 and x.shape[1] * x.shape[2] == scale.shape[0]


def _scale_for_input_layout(x: jax.Array, scale: jax.Array) -> tuple[jax.Array, int | tuple[int, ...]]:
  s_f32 = _centered_scale_fp32(scale)
  if _is_3d_tiled_expert_tensor(x, scale):
    return jnp.reshape(s_f32, (1, x.shape[1], x.shape[2])), (-2, -1)
  return s_f32, -1


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _input_scaled_rms_norm_save_rstd(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """Bit-exact `MuseSparkInputScaledRMSNorm` saving `(x, scale, rstd, var_eps)` on 2D `[..., D]` or 3D `[M, D//128, 128]`."""
  scale_f32, norm_axis = _scale_for_input_layout(x, scale)
  scaled_f32 = (jnp.asarray(x, jnp.float32) * scale_f32).astype(x.dtype).astype(jnp.float32)
  var_eps = jnp.mean(jax.lax.square(scaled_f32), axis=norm_axis, keepdims=True) + epsilon
  rstd = jax.lax.rsqrt(var_eps)
  return (scaled_f32 * rstd).astype(x.dtype)


def _input_scaled_rms_norm_save_rstd_fwd(x: jax.Array, scale: jax.Array, epsilon: float):
  scale_f32, norm_axis = _scale_for_input_layout(x, scale)
  scaled_f32 = (jnp.asarray(x, jnp.float32) * scale_f32).astype(x.dtype).astype(jnp.float32)
  var_eps = jnp.mean(jax.lax.square(scaled_f32), axis=norm_axis, keepdims=True) + epsilon
  rstd = jax.lax.rsqrt(var_eps)
  return (scaled_f32 * rstd).astype(x.dtype), (x, scale, rstd, var_eps)


def _input_scaled_rms_norm_save_rstd_bwd(epsilon: float, res, g: jax.Array):
  del epsilon
  x, scale, rstd, var_eps = res
  _, norm_axis = _scale_for_input_layout(x, scale)
  scaled_f32, vjp_scale = jax.vjp(
      lambda x_in, s_in: (jnp.asarray(x_in, jnp.float32) * _scale_for_input_layout(x_in, s_in)[0])
      .astype(x_in.dtype)
      .astype(jnp.float32),
      x,
      scale,
  )
  _, vjp_out = jax.vjp(lambda sf, r: (sf * r).astype(x.dtype), scaled_f32, rstd)
  d_scaled_1, d_rstd = vjp_out(g)
  d_var_eps = d_rstd * (-0.5 * jax.lax.div(rstd, var_eps))
  d_scaled_2 = _var_eps_transpose_dx(d_var_eps, scaled_f32, axis=norm_axis)
  return vjp_scale(d_scaled_1 + d_scaled_2)


_input_scaled_rms_norm_save_rstd.defvjp(
    _input_scaled_rms_norm_save_rstd_fwd, _input_scaled_rms_norm_save_rstd_bwd
)


def _input_scaled_rms_norm(x: jax.Array, scale: jax.Array, epsilon: float) -> jax.Array:
  """Scales `x` by `(1.0 + scale).float()` in fp32, rounds to `x.dtype`, then RMS-normalizes in fp32."""
  if x.dtype != jnp.float32:
    return _input_scaled_rms_norm_save_rstd(x, scale, epsilon)
  scale_f32, norm_axis = _scale_for_input_layout(x, scale)
  scaled = (jnp.asarray(x, jnp.float32) * scale_f32).astype(x.dtype).astype(jnp.float32)
  return _rms_normalize(scaled, epsilon, axis=norm_axis).astype(x.dtype)


def _residual_gate_impl(
    residual: jax.Array, hidden_states: jax.Array, gate: jax.Array, temperature: float, eps: float
) -> jax.Array:
  w = jnp.asarray(gate, jnp.float32) / temperature
  beta = jax.nn.sigmoid(w)
  alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-w) * (1.0 + beta), eps))
  out = alpha * jnp.asarray(residual, jnp.float32) + beta * jnp.asarray(hidden_states, jnp.float32)
  return out.astype(residual.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _residual_gate_save_compact(
    residual: jax.Array, hidden_states: jax.Array, gate: jax.Array, temperature: float, eps: float
) -> jax.Array:
  """Bit-exact `MuseSparkResidualGate` saving compact `residual.dtype` inputs instead of float32 copies."""
  return _residual_gate_impl(residual, hidden_states, gate, temperature, eps)


def _residual_gate_save_compact_fwd(
    residual: jax.Array, hidden_states: jax.Array, gate: jax.Array, temperature: float, eps: float
):
  return _residual_gate_impl(residual, hidden_states, gate, temperature, eps), (residual, hidden_states, gate)


def _residual_gate_save_compact_bwd(temperature: float, eps: float, res, g: jax.Array):
  residual, hidden_states, gate = res
  _, vjp_fn = jax.vjp(
      lambda r_in, h_in, g_in: _residual_gate_impl(r_in, h_in, g_in, temperature, eps),
      residual,
      hidden_states,
      gate,
  )
  return vjp_fn(g)


_residual_gate_save_compact.defvjp(_residual_gate_save_compact_fwd, _residual_gate_save_compact_bwd)


class MuseSparkRMSNorm(RMSNorm):
  """Muse-Spark RMSNorm (`MuseSparkRMSNorm`), using `pow(mean_sq + eps, -0.5)` in fp32 and casting to `x.dtype`."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: initializers.Initializer = jax.nn.initializers.ones,
      with_scale: bool = True,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=scale_init,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=with_scale,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    del out_sharding
    if x.dtype != jnp.float32:
      if self.with_scale and self.scale is not None:
        scale = self.scale[...]
        if self.parameter_memory_host_offload:
          scale = jax.device_put(scale, max_utils.device_space())
        return _rms_norm_pow_with_scale_save_rstd(x, scale, self.epsilon)
      return _rms_norm_pow_no_scale_save_rstd(x, self.epsilon)
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
    # Matches `MuseSparkRMSNorm._norm`: `hidden_states * torch.pow(mean_squared, -0.5)`.
    normed = x_f32 * jnp.power(mean2 + self.epsilon, -0.5)
    if self.with_scale and self.scale is not None:
      scale = self.scale[...]
      if self.parameter_memory_host_offload:
        scale = jax.device_put(scale, max_utils.device_space())
      normed = normed * jnp.asarray(scale, jnp.float32)
    return jnp.asarray(normed, x.dtype)


class MuseSparkCenteredRMSNorm(RMSNorm):
  """Muse-Spark zero-centered RMSNorm (`MuseSparkTextCenteredRMSNorm`): `normed * (1.0 + weight).float()`."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=jax.nn.initializers.zeros,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=True,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    if x.dtype != jnp.float32:
      return _centered_rms_norm_save_rstd(x, scale, self.epsilon)
    normed = _rms_normalize(jnp.asarray(x, jnp.float32), self.epsilon)
    output = normed * _centered_scale_fp32(scale)
    return jnp.asarray(output, x.dtype)


class MuseSparkInputScaledRMSNorm(RMSNorm):
  """Muse-Spark input-scaled RMSNorm (`MuseSparkTextInputScaledRMSNorm`).

  Scales the input by `(1.0 + weight).float()` in fp32, casts back to input dtype, then normalizes in fp32:
    `scaled = (x.float() * (1.0 + weight).float()).type_as(x).float()`
    `normed = (scaled * rsqrt(mean(scaled^2) + eps)).type_as(x)`
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-8,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=shard_mode,
        kernel_axes=kernel_axes,
        scale_init=jax.nn.initializers.zeros,
        parameter_memory_host_offload=parameter_memory_host_offload,
        with_scale=True,
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    return _input_scaled_rms_norm(x, scale, self.epsilon)


class MuseSparkResidualGate(nnx.Module):
  """Learnable fp32 temperature-scaled residual gate (`MuseSparkResidualGate`)."""

  def __init__(
      self,
      num_features: int,
      temperature: float = 0.3,
      eps: float = 1e-3,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      gate_init: initializers.Initializer = jax.nn.initializers.normal(stddev=0.02),
      *,
      rngs: nnx.Rngs,
  ):
    del weight_dtype  # Kept in float32 per `MuseSparkPreTrainedModel._keep_in_fp32_modules_strict`.
    self.num_features = num_features
    self.temperature = float(temperature)
    self.eps = float(eps)
    self.gate = nnx.Param(
        gate_init(rngs.params(), (num_features,), jnp.float32),
        sharding=kernel_axes,
    )

  def __call__(self, residual: jax.Array, hidden_states: jax.Array) -> jax.Array:
    gate = self.gate[...]
    if residual.dtype != jnp.float32:
      return _residual_gate_save_compact(residual, hidden_states, gate, self.temperature, self.eps)
    return _residual_gate_impl(residual, hidden_states, gate, self.temperature, self.eps)


# ------------------------------------------------------------------------------
# Gated GQA Attention
# ------------------------------------------------------------------------------


class MuseSparkAttention(attentions.Attention):
  """Muse-Spark Gated GQA Attention (`MuseSparkTextAttention`).

  Features:
  - Scaleless RMSNorm (`qk_norm`) on Q and K, followed by `query = (qk_norm(query) * qk_scale_factor).to(dtype)`.
  - Non-interleaved RoPE on sliding-window layers (`is_full_attention=False`), NoPE on full-attention layers.
  - Per-head scaleless RMSNorm (`o_norm`) on attention output before multiplying by `sigmoid(gate_proj(inputs_q))`.
  """

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: tuple[int, ...],
      inputs_kv_shape: tuple[int, ...],
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      float32_qk_product: bool | None = None,
      float32_logits: bool | None = None,
      quant: quantizations.AqtQuantization | None = None,
      kv_quant: quantizations.KVQuant | None = None,
      is_full_attention: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    attention_type = AttentionType.GLOBAL if is_full_attention else AttentionType.LOCAL_SLIDING
    sliding_window_size = 0 if is_full_attention else int(config.sliding_window_size)
    if float32_qk_product is None:
      float32_qk_product = bool(config.float32_qk_product)
    if float32_logits is None:
      float32_logits = bool(config.float32_logits)
    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        sliding_window_size=sliding_window_size,
        query_pre_attn_scalar=float(head_dim**-0.5),
        is_nope_layer=is_full_attention,
        use_bias_in_projections=config.attention_bias,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.is_full_attention = is_full_attention
    self.qk_scale_factor = float(config.muse_spark_qk_scale_factor)
    # Matches `MuseSparkTextAttention`: `gate_proj` never uses bias even when `attention_bias=True`.
    prev_use_bias = self.use_bias_in_projections
    self.use_bias_in_projections = False
    try:
      self.gate_proj = self.init_query_w(inputs_q_shape=inputs_q_shape)
    finally:
      self.use_bias_in_projections = prev_use_bias
    self.qk_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.o_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: int | None = None,
      bidirectional_mask: Any = None,
      rope_kwargs: dict | None = None,
      kv_cache: Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      **kwargs,
  ):
    del deterministic, slot, kwargs
    if model_mode == MODEL_MODE_PREFILL:
      input_axis_names = self.prefill_input_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      input_axis_names = self.input_axis_names
    else:
      input_axis_names = self.decode_input_axis_names

    inputs_q = self._maybe_shard_with_logical(inputs_q, input_axis_names)
    inputs_kv = self._maybe_shard_with_logical(inputs_kv, input_axis_names)
    qkv_sharding = create_sharding(self.mesh, input_axis_names)

    query = checkpoint_name(self.query_projection(inputs_q, out_sharding=qkv_sharding), "query_proj")
    key = checkpoint_name(self.kv_projection(inputs_kv, proj_name="key", out_sharding=qkv_sharding), "key_proj")
    value = self.kv_projection(inputs_kv, proj_name="value", out_sharding=qkv_sharding)

    query = (self.qk_norm(query) * self.qk_scale_factor).astype(query.dtype)
    key = self.qk_norm(key)

    if not self.is_nope_layer:
      if inputs_positions is None:
        inputs_positions = jnp.arange(inputs_q.shape[1], dtype=jnp.int32)[None, :]
      query = self.apply_rotary_embedding(query, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)
      key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions, rope_kwargs=rope_kwargs)

    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = (query * self.query_pre_attn_scalar).astype(query.dtype)

    if model_mode == MODEL_MODE_PREFILL:
      query = self._maybe_shard_with_logical(query, self.prefill_query_axis_names)
      key = self._maybe_shard_with_logical(key, self.prefill_key_axis_names)
      value = self._maybe_shard_with_logical(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      query = self._maybe_shard_with_logical(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = self._maybe_shard_with_logical(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = self._maybe_shard_with_logical(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    else:
      query = self._maybe_shard_with_logical(query, self.query_axis_names)
      key = self._maybe_shard_with_logical(key, self.key_axis_names)
      value = self._maybe_shard_with_logical(value, self.value_axis_names)

    value = checkpoint_name(value, "value_proj")

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") and model_mode != MODEL_MODE_TRAIN:
      batch, seq_len, num_heads, head_dim = query.shape
      attn_out, updated_kv = self.forward_serve_vllm(
          query, key, value, rpa_kv_cache=kv_cache, rpa_metadata=attention_metadata
      )
      out = attn_out.reshape(batch, seq_len, num_heads, head_dim)
      kv_cache = updated_kv
    else:
      cached_values = [None, None]
      if model_mode != MODEL_MODE_TRAIN:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      out = self.attention_op(
          query,
          key,
          value,
          decoder_segment_ids,
          inputs_positions,
          model_mode,
          cached_values,
          previous_chunk,
          bidirectional_mask,
          self.sinks,
      )

    out = checkpoint_name(out, "attention_out")
    if model_mode == MODEL_MODE_PREFILL:
      out_axis_names = self.prefill_out_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      out_axis_names = self.out_axis_names
    else:
      out_axis_names = self.decode_out_axis_names
    out = self._maybe_shard_with_logical(out, out_axis_names)

    out = self.o_norm(out)
    gate_logits = checkpoint_name(self.gate_proj(inputs_q, out_sharding=qkv_sharding), "mlpwi")
    gate = jax.nn.sigmoid(gate_logits.astype(jnp.float32)).astype(out.dtype)
    gate = self._maybe_shard_with_logical(gate, out_axis_names)
    out = out * gate

    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache


# ------------------------------------------------------------------------------
# Bottlenecked Sparse MoE (`MuseSparkTextSparseMoeBlock`)
# ------------------------------------------------------------------------------


def _checkpoint_routing_maps(top_k_index: jax.Array, top_k_weights: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Names the `[B, S, k]` routing maps `moe_routing_maps` in a lane-dense `[B, S * k // 128, 128]` layout.

  A `[B, S, k=8]` array pads its minor dimension from 8 to 128 lanes in HBM, so a remat policy that saves the routing
  maps by name would hold 16x their size. Reshaping to 128 lanes first stores them without padding. Values are
  unchanged.
  """
  b, s, k = top_k_index.shape
  if (s * k) % 128 != 0:
    return checkpoint_name(top_k_index, "moe_routing_maps"), checkpoint_name(top_k_weights, "moe_routing_maps")
  packed_shape = (b, (s * k) // 128, 128)
  index_packed = checkpoint_name(jnp.reshape(top_k_index, packed_shape), "moe_routing_maps")
  weights_packed = checkpoint_name(jnp.reshape(top_k_weights, packed_shape), "moe_routing_maps")
  return jnp.reshape(index_packed, (b, s, k)), jnp.reshape(weights_packed, (b, s, k))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _gather_topk_weights(
    routing_weights: jax.Array,
    top_k_index: jax.Array,
    valid_token_mask: jax.Array | None,
    num_experts: int,
    dense_fwd: bool = False,
) -> jax.Array:
  """Gathers `routing_weights` at `top_k_index` (`[..., E] -> [..., k]`) with a dense compare-select VJP.

  XLA's default VJP for `jnp.take_along_axis(routing_weights, gather_indices, axis=-1)` sorts and scatters into a
  `[B * S * E]` (`1,048,576`-element) array per MoE layer, and when SparseCore gather offload is disabled on the opt
  path `jnp.take_along_axis` lowers to scalar 4B HBM gathers. Comparing `top_k_index` against `0 .. E - 1` on the
  TensorCore VPU produces the bit-exact forward gather (when `dense_fwd=True`) and backward cotangent without scalar
  gathers, sorting, or scattering.
  """
  gather_indices = jnp.where(valid_token_mask, top_k_index, 0) if valid_token_mask is not None else top_k_index
  if dense_fwd:
    iota = jax.lax.broadcasted_iota(
        gather_indices.dtype, (1,) * gather_indices.ndim + (num_experts,), gather_indices.ndim
    )
    top_k_weights = jnp.sum(
        jnp.where(gather_indices[..., None] == iota, routing_weights[..., None, :], jnp.zeros((), routing_weights.dtype)),
        axis=-1,
    )
  else:
    top_k_weights = jnp.take_along_axis(routing_weights, gather_indices, axis=-1)
  if valid_token_mask is not None:
    top_k_weights = top_k_weights * valid_token_mask
  return top_k_weights


def _gather_topk_weights_fwd(routing_weights, top_k_index, valid_token_mask, num_experts, dense_fwd):
  return (
      _gather_topk_weights(routing_weights, top_k_index, valid_token_mask, num_experts, dense_fwd),
      (top_k_index, valid_token_mask),
  )


def _gather_topk_weights_bwd(num_experts, dense_fwd, res, g):
  del dense_fwd
  top_k_index, valid_token_mask = res
  if valid_token_mask is not None:
    g = jnp.where(valid_token_mask, g, jnp.zeros((), g.dtype))
  iota = jax.lax.broadcasted_iota(top_k_index.dtype, (1,) * top_k_index.ndim + (num_experts,), top_k_index.ndim)
  hit = top_k_index[..., None] == iota
  if valid_token_mask is not None:
    hit = hit & valid_token_mask[..., None]
  d_routing_weights = jnp.sum(jnp.where(hit, g[..., None], jnp.zeros((), g.dtype)), axis=-2)
  return d_routing_weights, None, None


_gather_topk_weights.defvjp(_gather_topk_weights_fwd, _gather_topk_weights_bwd)


def _count_groups(ids: jax.Array, num_groups: int) -> jax.Array:
  """Counts occurrences of `0 .. num_groups - 1` in `ids` via a vectorized compare-and-sum (`0` scatters)."""
  iota = jax.lax.broadcasted_iota(ids.dtype, (1,) * ids.ndim + (num_groups,), ids.ndim)
  return jnp.sum(ids[..., None] == iota, axis=tuple(range(ids.ndim)), dtype=jnp.int32)


def _mesh_is_tpu(mesh: Mesh) -> bool:
  """Returns True when `mesh` is made of TPU devices."""
  return mesh.devices.flat[0].platform == "tpu"


class MuseSparkRouterGate(nnx.Module):
  """Router gate holding fp32 `kernel` and `MoEBiasVar` `bias`."""

  def __init__(
      self,
      emb_dim: int,
      num_experts: int,
      use_bias: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.use_bias = use_bias
    self.kernel = nnx.Param(
        initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")(
            rngs.params(), (emb_dim, num_experts), jnp.float32, 0, 1
        ),
        sharding=("embed_router", None),
    )
    if use_bias:
      self.bias = moe.MoEBiasVar(
          jnp.zeros((num_experts,), dtype=jnp.float32),
          sharding=(None,),
      )
    else:
      self.bias = None


class MuseSparkMoE(nnx.Module):
  """Muse-Spark bottlenecked Sparse MoE block (`MuseSparkTextSparseMoeBlock`).

  Structure:
  1. Router (`MuseSparkRouterGate`) operates on the full-width `emb_dim` (8192) hidden states in fp32,
     scales router logits by `muse_spark_output_multiplier`, applies sigmoid + `e_score_correction_bias`
     for top-k selection, and normalizes the un-biased sigmoid weights over the selected top-k experts.
  2. Shared width-bottleneck down-projection `pre_expert_proj` (`emb_dim -> moe_expert_input_dim`)
     followed by `pre_expert_norm` (`MuseSparkCenteredRMSNorm`).
  3. 256 routed SwiGLU experts (`wi_0`, `wi_1`, `wo` of shape `[num_experts, moe_expert_input_dim, moe_mlp_dim]`)
     followed by per-expert-token `post_expert_norm` (`MuseSparkInputScaledRMSNorm`) before weighting by
     `top_k_weights` and summing over the `k` selected experts.
  4. Shared width-bottleneck up-projection `post_expert_proj` (`moe_expert_input_dim -> emb_dim`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.num_experts = int(config.num_experts)
    self.num_experts_per_tok = int(config.num_experts_per_tok)
    self.emb_dim = int(config.emb_dim)
    self.expert_input_dim = int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)
    self.moe_mlp_dim = int(config.moe_mlp_dim)
    self.output_multiplier = float(config.muse_spark_output_multiplier)
    self.routed_scaling_factor = float(config.routed_scaling_factor)
    self.routed_score_func = str(config.routed_score_func or "sigmoid")
    self.routed_bias = bool(config.routed_bias)
    self.norm_topk_prob = bool(config.norm_topk_prob)
    self.float32_gate_logits = bool(config.float32_gate_logits)
    if config.moe_tc_ragged_weights_on_activation:
      raise ValueError(
          "MuseSparkMoE requires `moe_tc_ragged_weights_on_activation=False` because "
          "`post_expert_norm` must normalize each expert output before `top_k_weights` scaling."
      )

    norm_weight_dtype = get_weight_dtype(config, "norm")
    expert_weight_dtype = get_weight_dtype(config, "moe_block")

    self.gate = MuseSparkRouterGate(
        emb_dim=self.emb_dim,
        num_experts=self.num_experts,
        use_bias=self.routed_bias,
        rngs=rngs,
    )
    self.pre_expert_proj = linears.DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.expert_input_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )
    self.pre_expert_norm = MuseSparkCenteredRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    rules_init = get_logical_axis_rules() or config.logical_axis_rules
    wi_init_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        logical_to_mesh_axes(("exp", "embed_moe", "mlp_moe"), mesh=mesh, rules=rules_init),
        (self.num_experts, self.expert_input_dim, self.moe_mlp_dim),
        mesh,
        allow_remove_axes=True,
    )
    wo_axis1_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        logical_to_mesh_axes(("exp", "embed_moe", "mlp_moe"), mesh=mesh, rules=rules_init),
        (self.num_experts, self.moe_mlp_dim, self.expert_input_dim),
        mesh,
        allow_remove_axes=True,
    )
    use_wo_fsdp_axis1 = not mesh_axes_for_dim(wi_init_pspec[2]) and wo_axis1_pspec[1] == wi_init_pspec[1]
    wo_sharding = ("exp", "embed_moe", "mlp_moe") if use_wo_fsdp_axis1 else ("exp", "mlp_moe", "embed_moe")
    self.wo_sharding = wo_sharding

    expert_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    self.wi_0 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), expert_weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wi_1 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), expert_weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wo = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.moe_mlp_dim, self.expert_input_dim), expert_weight_dtype, 1, 2
        ),
        sharding=wo_sharding,
    )
    self.post_expert_norm = MuseSparkInputScaledRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_expert_proj = linears.DenseGeneral(
        in_features_shape=self.expert_input_dim,
        out_features_shape=self.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )

  def _compute_load_balance_loss(self, top_k_index: jax.Array, routing_weights: jax.Array) -> jax.Array:
    """Computes sequence-wise auxiliary load-balance loss matching `RoutedMoE`."""
    if moe.uses_megatron_seq_aux_loss(self.config):
      routing_weights, top_k_index = moe.megatron_aux_loss_inputs(routing_weights, self.num_experts_per_tok)
    expert_mask = jax.nn.one_hot(top_k_index, num_classes=self.num_experts, dtype=jnp.int32)
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / self.num_experts_per_tok
    density_prob = jnp.mean(routing_weights.astype(jnp.float32), axis=1)
    return jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight

  def __call__(
      self,
      inputs: jax.Array,
      forced_routed_experts: jax.Array | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[jax.Array, jax.Array | None]:
    k = self.num_experts_per_tok
    d_in = self.expert_input_dim

    # 1. Router on full-width hidden_states
    gate_dtype = jnp.float32 if self.float32_gate_logits else inputs.dtype
    router_logits = jnp.einsum(
        "bsd,de->bse",
        inputs.astype(gate_dtype),
        self.gate.kernel[...].astype(gate_dtype),
        precision=jax.lax.Precision.HIGHEST if gate_dtype == jnp.float32 else self.config.matmul_precision,
    ).astype(jnp.float32)
    router_logits = router_logits * self.output_multiplier
    if self.routed_score_func == "sigmoid":
      routing_weights = jax.nn.sigmoid(router_logits)
    elif self.routed_score_func == "softmax":
      routing_weights = jax.nn.softmax(router_logits, axis=-1)
    else:
      routing_weights = linears._convert_to_activation_function(  # pylint: disable=protected-access
          self.routed_score_func
      )(router_logits)

    scores_for_choice = routing_weights
    if self.routed_bias and self.gate.bias is not None:
      scores_for_choice = scores_for_choice + self.gate.bias[...].astype(jnp.float32)

    if forced_routed_experts is not None:
      top_k_index = forced_routed_experts.astype(jnp.int32)
      valid_token_mask = moe.valid_expert_mask(top_k_index, self.num_experts)
    elif self.config.use_random_routing:
      # Synthetic-perf routing (as in `moe.random_routing`): uniform expert ids model balanced steady-state load
      # instead of the skew of an untrained router. The top-k weights are still gathered from the learned router
      # scores below, so the router keeps receiving gradients.
      top_k_index = jax.random.randint(
          jax.random.key(self.config.init_weights_seed),
          shape=(*scores_for_choice.shape[:-1], k),
          minval=0,
          maxval=self.num_experts,
          dtype=jnp.int32,
      )
      _, top_k_index = jax.lax.optimization_barrier((jax.lax.stop_gradient(scores_for_choice), top_k_index))
      valid_token_mask = None
    else:
      _, top_k_index = jax.lax.top_k(jax.lax.stop_gradient(scores_for_choice), k=k)
      valid_token_mask = None
    if self.config.return_routed_experts:
      self.selected_experts = nnx.Intermediate(jnp.reshape(top_k_index, (-1, k)))

    load_balance_loss = None
    if self.config.load_balance_loss_weight > 0.0:
      load_balance_loss = self._compute_load_balance_loss(top_k_index, routing_weights)

    top_k_weights = _gather_topk_weights(routing_weights, top_k_index, valid_token_mask, self.num_experts, True)
    if self.norm_topk_prob:
      top_k_weights = top_k_weights / (jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-15)
    top_k_weights = jnp.asarray(top_k_weights * self.routed_scaling_factor, jnp.float32)

    # Resolve the sharding specs before `_checkpoint_routing_maps`: with expert parallelism the routing is
    # checkpointed after the expert-parallel all-gather (per-shard sort result) instead of here.
    if model_mode == MODEL_MODE_PREFILL:
      batch_axis, length_axis = "activation_batch", "prefill_activation_norm_length"
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      batch_axis, length_axis = DECODE_BATCH, DECODE_LENGTH
    else:
      batch_axis, length_axis = "activation_batch", "activation_norm_length"

    rules = get_logical_axis_rules() or self.config.logical_axis_rules
    x_raw_pspec = logical_to_mesh_axes((batch_axis, length_axis, None), mesh=self.mesh, rules=rules)
    x_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        P(x_raw_pspec[0], x_raw_pspec[1], None),
        (inputs.shape[0], inputs.shape[1], self.expert_input_dim),
        self.mesh,
        allow_remove_axes=True,
    )
    wi_raw_pspec = logical_to_mesh_axes(("exp", "embed_moe", "mlp_moe"), mesh=self.mesh, rules=rules)
    raw_ep_axes = mesh_axes_for_dim(wi_raw_pspec[0])
    raw_ep_size = math.prod(self.mesh.shape[a] for a in raw_ep_axes)
    if self.num_experts % raw_ep_size != 0:
      raise ValueError(
          f"MuseSparkMoE requires `num_experts` ({self.num_experts}) to be divisible by "
          f"expert mesh size ({raw_ep_size})."
      )
    wi_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        wi_raw_pspec,
        (self.num_experts, self.expert_input_dim, self.moe_mlp_dim),
        self.mesh,
        allow_remove_axes=True,
    )
    wo_meta = self.wo.get_metadata()
    wo_logical = (
        wo_meta.get("out_sharding")
        or wo_meta.get("sharding_names")
        or wo_meta.get("sharding")
        or getattr(self, "wo_sharding", None)
    )
    wo_is_axis1 = tuple(wo_logical) == ("exp", "embed_moe", "mlp_moe") if wo_logical is not None else False
    wo_pspec = P(wi_pspec[0], wi_pspec[1], wi_pspec[2]) if wo_is_axis1 else P(wi_pspec[0], wi_pspec[2], wi_pspec[1])
    wo_gather_axis = 1 if wo_is_axis1 else 2
    ep_axes = mesh_axes_for_dim(wi_pspec[0])
    fsdp_axes = mesh_axes_for_dim(wi_pspec[1])
    tp_axes = mesh_axes_for_dim(wi_pspec[2])
    if math.prod(self.mesh.shape[a] for a in ep_axes) > 1:
      raise ValueError(
          "MuseSparkMoE does not support expert parallelism (`ici_expert_parallelism > 1`); "
          "shard experts with FSDP and tensor parallelism instead."
      )
    fsdp_axis_arg = fsdp_axes[0] if len(fsdp_axes) == 1 else (fsdp_axes or None)
    tp_axis_arg = tp_axes[0] if len(tp_axes) == 1 else (tp_axes or None)
    is_tpu = _mesh_is_tpu(self.mesh)
    top_k_index, top_k_weights = _checkpoint_routing_maps(top_k_index, top_k_weights)

    # 2. Width bottleneck before experts (`emb_dim -> expert_input_dim`)
    x = self.pre_expert_proj(inputs)
    # Name the projection output, not the normed latent. The norm VJP needs its input, so saving it lets remat
    # recompute only the elementwise norm instead of re-running `pre_expert_proj` (matmul + FSDP all-gather).
    x = checkpoint_name(x, "pre_expert_norm")
    x = self.pre_expert_norm(x)

    use_gmm_v2 = bool(
        self.config.use_tokamax_gmm
        and self.config.use_gmm_v2
        and is_tpu
    )
    use_heuristic_tiling = bool(self.config.use_gmm_v2_heuristic_tiling)
    is_eval = model_mode != MODEL_MODE_TRAIN
    wi_tiling = (
        self.config.eval_wi_tile_fwd_batch_seq if is_eval else self.config.wi_tile_fwd_batch_seq,
        self.config.eval_wi_tile_fwd_embed_dim if is_eval else self.config.wi_tile_fwd_embed_dim,
        self.config.eval_wi_tile_fwd_mlp_dim if is_eval else self.config.wi_tile_fwd_mlp_dim,
        self.config.wi_tile_dlhs_batch_seq,
        self.config.wi_tile_dlhs_mlp_dim,
        self.config.wi_tile_dlhs_embed_dim,
        self.config.wi_tile_drhs_batch_seq,
        self.config.wi_tile_drhs_embed_dim,
        self.config.wi_tile_drhs_mlp_dim,
    )
    wo_tiling = (
        self.config.eval_wo_tile_fwd_batch_seq if is_eval else self.config.wo_tile_fwd_batch_seq,
        self.config.eval_wo_tile_fwd_mlp_dim if is_eval else self.config.wo_tile_fwd_mlp_dim,
        self.config.eval_wo_tile_fwd_embed_dim if is_eval else self.config.wo_tile_fwd_embed_dim,
        self.config.wo_tile_dlhs_batch_seq,
        self.config.wo_tile_dlhs_embed_dim,
        self.config.wo_tile_dlhs_mlp_dim,
        self.config.wo_tile_drhs_batch_seq,
        self.config.wo_tile_drhs_mlp_dim,
        self.config.wo_tile_drhs_embed_dim,
    )

    def _clamped_gmm_tiles(
        tm: int,
        tk: int,
        tn: int,
        size_m: int,
        size_group: int,
        size_k: int,
        size_n: int,
    ) -> gmm_v2.TileSizes:
      del size_m
      if (tm, tk, tn) == (512, 1024, 1024) and size_k >= 2048 and size_n >= 2048:
        if 0 < size_group < 16:
          tm, tk, tn = 1024, 4096, 1024
        else:
          tm, tk, tn = 256, 4096, 2048
      aligned_k = gmm_v2.align_to(size_k, 128)
      aligned_n = gmm_v2.align_to(size_n, 128)
      out_tk = min(tk, aligned_k)
      out_tn = min(tn, aligned_n)
      return gmm_v2.TileSizes(tile_m=tm, tile_k=out_tk, tile_n=out_tn)

    def _clamped_tgmm_tiles(
        tm: int,
        tk: int,
        tn: int,
        size_k: int,
        size_n: int,
    ) -> gmm_v2.TileSizes:
      if (tm, tk, tn) == (512, 1024, 1024) and size_k >= 2048 and size_n >= 2048:
        tk, tn = 2048, 2048
      aligned_k = gmm_v2.align_to(size_k, 128)
      aligned_n = gmm_v2.align_to(size_n, 128)
      out_tk = min(tk, aligned_k)
      out_tn = min(tn, aligned_n)
      return gmm_v2.TileSizes(tile_m=tm, tile_k=out_tk, tile_n=out_tn)

    scale_pspec = P(None)
    in_specs_tuple = (x_pspec, x_pspec, x_pspec, wi_pspec, wi_pspec, wo_pspec, scale_pspec)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=in_specs_tuple,
        out_specs=x_pspec,
        check_vma=False,
    )
    def _local_moe_forward(x_local, idx_local, weights_local, wi_0_val, wi_1_val, wo_val, post_norm_scale):
      local_bsz, local_seq, _ = x_local.shape
      local_tokens = local_bsz * local_seq
      x_flat = jnp.reshape(x_local, (local_tokens, d_in))
      flat_expert_ids = jnp.reshape(idx_local, (local_tokens * k,))
      flat_weights = jnp.reshape(weights_local, (local_tokens * k,))
      valid_flat = None
      if valid_token_mask is not None:
        valid_flat = moe.valid_expert_mask(flat_expert_ids, self.num_experts)

      def _gather_expert_weights(rhs, lhs_dtype, gather_axis: int):
        """Casts the local expert-weight shard and all-gathers its FSDP-sharded embed axis."""
        rhs = rhs.astype(lhs_dtype)
        if fsdp_axis_arg is not None:
          rhs = jax.lax.all_gather(rhs, axis_name=fsdp_axis_arg, axis=gather_axis, tiled=True)
        return rhs

      def _ragged_dot(
          lhs,
          rhs_sharded,
          grp,
          gather_axis: int,
          tiling: tuple[int, ...],
      ):
        """Grouped matmul (`gmm_v2` or `ragged_dot`) whose backward reuses or re-gathers FSDP expert weights."""
        use_reuse_bwd_rhs = fsdp_axis_arg is not None
        rhs_dtype = rhs_sharded.dtype

        def _dot_fwd_impl(lhs_in, rhs_in, grp_in):
          _, grp_in = jax.lax.optimization_barrier((lhs_in, grp_in))
          rhs_full = _gather_expert_weights(rhs_in, lhs_in.dtype, gather_axis)
          if not use_gmm_v2:
            out = jax.lax.ragged_dot(lhs_in, rhs_full, grp_in, precision=self.config.matmul_precision)
            return out, rhs_full
          if rhs_full.shape[1] % 128 != 0 or rhs_full.shape[2] % 128 != 0:
            raise ValueError(
                "MuseSparkMoE `use_gmm_v2=True` requires local expert weight dims to be multiples of 128; "
                f"got gathered shape {rhs_full.shape}."
            )
          fwd_tile = (
              gmm_v2.calculate_tiling
              if use_heuristic_tiling
              else _clamped_gmm_tiles(
                  tiling[0],
                  tiling[1],
                  tiling[2],
                  lhs_in.shape[0],
                  rhs_full.shape[0],
                  rhs_full.shape[1],
                  rhs_full.shape[2],
              )
          )
          pad_align_m = fwd_tile.tile_m if isinstance(fwd_tile, gmm_v2.TileSizes) else tiling[0]
          lhs_pad, pad_m = max_utils.maybe_pad(lhs_in, pad_align_m)
          out_pad = gmm_v2.gmm_v2(
              lhs=lhs_pad,
              rhs=rhs_full,
              group_sizes=grp_in,
              tile_info=fwd_tile,
              preferred_element_type=lhs_in.dtype,
          )
          out = out_pad[: lhs_in.shape[0]] if pad_m > 0 else out_pad
          return out, rhs_full

        def _compute_dlhs(orig_m: int, rhs_full, grp_in, g, lhs_dtype):
          dlhs_tile = (
              gmm_v2.calculate_tiling
              if use_heuristic_tiling
              else _clamped_gmm_tiles(
                  tiling[3],
                  tiling[4],
                  tiling[5],
                  orig_m,
                  rhs_full.shape[0],
                  rhs_full.shape[2],
                  rhs_full.shape[1],
              )
          )
          pad_align_dlhs = dlhs_tile.tile_m if isinstance(dlhs_tile, gmm_v2.TileSizes) else tiling[3]
          g_pad_dlhs, pad_dlhs = max_utils.maybe_pad(g, pad_align_dlhs)
          d_lhs_pad = gmm_v2.gmm_v2(
              lhs=g_pad_dlhs,
              rhs=rhs_full,
              group_sizes=grp_in,
              tile_info=dlhs_tile,
              preferred_element_type=lhs_dtype,
              transpose_rhs=True,
          )
          return d_lhs_pad[:orig_m] if pad_dlhs > 0 else d_lhs_pad

        def _compute_drhs(lhs_in, rhs_full, grp_in, g, gather_vjp):
          drhs_tile = (
              tgmm_v2.calculate_tgmm_tiling
              if use_heuristic_tiling
              else _clamped_tgmm_tiles(tiling[6], tiling[7], tiling[8], rhs_full.shape[1], rhs_full.shape[2])
          )
          pad_align_drhs = drhs_tile.tile_m if isinstance(drhs_tile, gmm_v2.TileSizes) else tiling[6]
          lhs_pad_drhs, pad_drhs = max_utils.maybe_pad(lhs_in, pad_align_drhs)
          g_pad_drhs = jnp.pad(g, ((0, pad_drhs), (0, 0))) if pad_drhs > 0 else g
          d_rhs_full = tgmm_v2.tgmm_v2(
              lhs=lhs_pad_drhs,
              rhs=g_pad_drhs,
              group_sizes=grp_in,
              num_actual_groups=rhs_full.shape[0],
              tile_info=drhs_tile,
              preferred_element_type=rhs_full.dtype,
          )
          if gather_vjp is not None:
            (d_rhs,) = gather_vjp(d_rhs_full)
          else:
            d_rhs = jax.lax.psum_scatter(
                d_rhs_full, axis_name=fsdp_axis_arg, scatter_dimension=gather_axis, tiled=True
            ).astype(rhs_dtype)
          return d_rhs

        @jax.custom_vjp
        def _dot_fn(lhs_in, rhs_in, grp_in):
          out, _ = _dot_fwd_impl(lhs_in, rhs_in, grp_in)
          return out

        def _dot_fwd(lhs_in, rhs_in, grp_in):
          out, rhs_full = _dot_fwd_impl(lhs_in, rhs_in, grp_in)
          if use_reuse_bwd_rhs:
            return out, (lhs_in, rhs_full, grp_in)
          return out, (lhs_in, rhs_in, grp_in)

        def _dot_bwd(res, g):
          if use_reuse_bwd_rhs:
            lhs_in, rhs_full, grp_in = res
            _, grp_in = jax.lax.optimization_barrier((g, grp_in))
            gather_vjp = None
            if not use_gmm_v2:
              _, vjp_fn = jax.vjp(
                  lambda l, r_full: jax.lax.ragged_dot(
                      l,
                      r_full,
                      grp_in,
                      precision=self.config.matmul_precision,
                  ),
                  lhs_in,
                  rhs_full,
              )
              d_lhs, d_rhs_full = vjp_fn(g)
              d_rhs = jax.lax.psum_scatter(
                  d_rhs_full, axis_name=fsdp_axis_arg, scatter_dimension=gather_axis, tiled=True
              ).astype(rhs_dtype)
              return d_lhs, d_rhs, None
          else:
            lhs_in, rhs_in, grp_in = res
            _, grp_in = jax.lax.optimization_barrier((g, grp_in))
            if not use_gmm_v2:
              _, vjp_fn = jax.vjp(
                  lambda l, r: jax.lax.ragged_dot(
                      l,
                      _gather_expert_weights(r, l.dtype, gather_axis),
                      grp_in,
                      precision=self.config.matmul_precision,
                  ),
                  lhs_in,
                  rhs_in,
              )
              d_lhs, d_rhs = vjp_fn(g)
              return d_lhs, d_rhs, None
            rhs_full, gather_vjp = jax.vjp(
                lambda r: _gather_expert_weights(r, lhs_in.dtype, gather_axis),
                rhs_in,
            )

          d_lhs = _compute_dlhs(lhs_in.shape[0], rhs_full, grp_in, g, lhs_in.dtype)
          d_rhs = _compute_drhs(lhs_in, rhs_full, grp_in, g, gather_vjp)
          return d_lhs, d_rhs, None

        _dot_fn.defvjp(_dot_fwd, _dot_bwd)
        return _dot_fn(lhs, rhs_sharded, grp)


      if valid_flat is not None:
        dummy_ids = jnp.arange(local_tokens * k, dtype=flat_expert_ids.dtype) % self.num_experts
        flat_expert_ids = jnp.where(valid_flat, flat_expert_ids, dummy_ids)
      flat_token_ids = jnp.repeat(jnp.arange(local_tokens, dtype=jnp.int32), k)
      sorted_expert_ids, perm = ragged_gather_tc.packed_stable_sort(flat_expert_ids, max(0, self.num_experts - 1))
      sorted_token_ids = flat_token_ids[perm]
      sorted_weights = flat_weights[perm]
      x_sorted = x_flat[sorted_token_ids]
      group_sizes = _count_groups(sorted_expert_ids, self.num_experts)

      gate_out = _ragged_dot(x_sorted, wi_0_val, group_sizes, 1, wi_tiling)
      x_for_wi1 = x_sorted
      up_out = _ragged_dot(x_for_wi1, wi_1_val, group_sizes, 1, wi_tiling)
      act = jax.nn.silu(gate_out.astype(jnp.float32)).astype(x_local.dtype) * up_out
      expert_out = _ragged_dot(act, wo_val, group_sizes, wo_gather_axis, wo_tiling)
      if tp_axis_arg is not None:
        expert_out = jax.lax.psum(expert_out, axis_name=tp_axis_arg)

      # Per-expert-output input-scaled RMSNorm before weighting by fp32 top_k_weights
      expert_out = _input_scaled_rms_norm(
          expert_out, post_norm_scale, self.post_expert_norm.epsilon
      )

      weighted_sorted = (expert_out.astype(jnp.float32) * sorted_weights[:, None]).astype(x_local.dtype)

      # Group back by token_id while preserving ascending expert_idx order (matches HF loop order)
      _, token_group_perm = ragged_gather_tc.packed_stable_sort(sorted_token_ids, max(0, local_tokens - 1))
      by_token = jnp.reshape(weighted_sorted[token_group_perm], (local_bsz, local_seq, k, d_in))
      return jnp.sum(by_token, axis=2)

    combined = _local_moe_forward(
        x,
        top_k_index,
        top_k_weights,
        self.wi_0[...],
        self.wi_1[...],
        self.wo[...],
        self.post_expert_norm.scale[...],
    )
    combined = checkpoint_name(combined, "moe_combined")

    # 4. Width bottleneck up-projection back to emb_dim
    out = self.post_expert_proj(combined)
    # `post_feedforward_layernorm`'s VJP needs this tensor; naming it lets a custom remat policy (`mlpwo=device`)
    # skip re-running `post_expert_proj` (matmul + FSDP all-gather). All-MoE Muse Spark has no other `mlpwo` tag.
    out = checkpoint_name(out, "mlpwo")
    return out, load_balance_loss


# ------------------------------------------------------------------------------
# Decoder Layer & Scannable Block
# ------------------------------------------------------------------------------


class MuseSparkDecoderLayer(nnx.Module):
  """Single Muse-Spark transformer decoder layer (`MuseSparkTextDecoderLayer`)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      layer_idx: int = 0,
      is_full_attention_override: bool | None = None,
      is_moe_override: bool | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.is_full_attention = (
        is_full_attention_layer(layer_idx, config)
        if is_full_attention_override is None
        else bool(is_full_attention_override)
    )
    self.is_moe = is_moe_layer(layer_idx, config) if is_moe_override is None else bool(is_moe_override)

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
    norm_weight_dtype = get_weight_dtype(config, "norm")

    self.input_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.self_attention = MuseSparkAttention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        max_prefill_predict_length=config.max_prefill_predict_length,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        is_full_attention=self.is_full_attention,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.post_attention_layernorm = MuseSparkRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.post_attention_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.pre_feedforward_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    if self.is_moe:
      self.moe_block = MuseSparkMoE(
          config=config,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = linears.MlpBlock(
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          config=config,
          mesh=mesh,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )
    self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=norm_weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_feedforward_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.activation_axis_names = self._activation_axis_names_for_mode(model_mode)

  @staticmethod
  def _activation_axis_names_for_mode(model_mode: str) -> tuple[str, str, str]:
    if model_mode == MODEL_MODE_PREFILL:
      return ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      return (DECODE_BATCH, DECODE_LENGTH, "activation_embed")
    return ("activation_batch", "activation_norm_length", "activation_embed")

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kwargs
    activation_axis_names = self._activation_axis_names_for_mode(model_mode)
    inputs = sharding.maybe_shard_with_logical(
        inputs,
        activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # 1. Self-Attention sub-block
    residual = inputs
    hidden_states = self.input_layernorm(inputs)
    attn_out, kv_cache = self.self_attention(
        hidden_states,
        hidden_states,
        decoder_positions,
        decoder_segment_ids,
        model_mode=model_mode,
        deterministic=deterministic,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attn_out = self.post_attention_layernorm(attn_out)
    hidden_states = self.post_attention_residual_gate(residual, attn_out)

    # 2. Feedforward / MoE sub-block
    residual = hidden_states
    ffn_in = self.pre_feedforward_layernorm(hidden_states)
    load_balance_loss = None
    if self.is_moe:
      ffn_out, load_balance_loss = self.moe_block(
          ffn_in,
          forced_routed_experts=forced_routed_experts,
          model_mode=model_mode,
      )
    else:
      ffn_out = self.mlp(ffn_in, deterministic=deterministic)
    ffn_out = self.post_feedforward_layernorm(ffn_out)
    output = self.post_feedforward_residual_gate(residual, ffn_out)

    output = sharding.maybe_shard_with_logical(
        output,
        activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    return output, kv_cache


@jax.custom_vjp
def _release_cotangents_together(leaves):
  return leaves


def _release_cotangents_together_fwd(leaves):
  return leaves, None


def _release_cotangents_together_bwd(_, cotangents):
  d_params_and_zero = jax.lax.optimization_barrier((*cotangents[:-1], jnp.float32(0.0)))
  d_params = d_params_and_zero[:-1]
  zero = d_params_and_zero[-1]
  d_x = cotangents[-1]
  pred = zero == jnp.float32(0.0)
  d_x_out = jax.lax.select(jnp.broadcast_to(pred, d_x.shape), d_x, jnp.zeros_like(d_x))
  return ((*d_params, d_x_out),)


_release_cotangents_together.defvjp(_release_cotangents_together_fwd, _release_cotangents_together_bwd)


def _tie_expert_weight_cotangents(moe_block: "MuseSparkMoE", inputs: jax.Array) -> jax.Array:
  """Returns `inputs` with a VJP that holds its cotangent until `moe_block`'s expert weight gradients are done.

  In a scanned block the backward pass of sub-layer `j` produces the cotangent that sub-layer `j - 1` consumes. Without
  this barrier XLA can start sub-layer `j - 1` while the `[experts, ...]` weight gradients of sub-layer `j` are still
  pending, keeping both sub-layers' gathered expert weights and activations alive at once. The barrier is on the weight
  cotangents only; it does not change any value.
  """
  leaves, treedef = jax.tree.flatten(nnx.state(moe_block, nnx.Param))
  expert_idx = [
      i
      for i, x in enumerate(leaves)
      if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact) and getattr(x, "ndim", 0) >= 3
  ]
  tied = _release_cotangents_together((*(leaves[i] for i in expert_idx), inputs))
  for i, x in zip(expert_idx, tied[:-1]):
    leaves[i] = x
  nnx.update(moe_block, jax.tree.unflatten(treedef, leaves))
  return tied[-1]


class MuseSparkScannableBlock(nnx.Module):
  """Scannable inhomogeneous cycle of `inhomogeneous_layer_cycle_interval` (4) Muse-Spark MoE layers.

  Within each cycle:
  - Sub-layers `0 .. cycle - 2` use sliding-window attention (`sliding_window_size=2048`) + RoPE
    (`is_full_attention=False`).
  - Sub-layer `cycle - 1` uses global causal attention + NoPE (`is_full_attention=True`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.num_of_layers = int(config.inhomogeneous_layer_cycle_interval)
    for sub_id in range(self.num_of_layers):
      layer = MuseSparkDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          layer_idx=sub_id,
          is_full_attention_override=(sub_id == self.num_of_layers - 1),
          is_moe_override=True,
          rngs=rngs,
      )
      setattr(self, f"layers_{sub_id}", layer)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: Any = None,
      attention_metadata: Any = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kwargs
    y = inputs
    updated_kvs = []
    for sub_id in range(self.num_of_layers):
      sub_forced = forced_routed_experts[sub_id] if forced_routed_experts is not None else None
      sub_kv = kv_cache[sub_id] if kv_cache is not None else None
      sub_attn_meta = attention_metadata[sub_id] if isinstance(attention_metadata, tuple) else attention_metadata
      layer = getattr(self, f"layers_{sub_id}")
      if sub_id > 0 and kv_cache is None and not self.config.prefetch_fsdp_weights:
        y = _tie_expert_weight_cotangents(layer.moe_block, y)
      y, new_kv = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=sub_kv,
          attention_metadata=sub_attn_meta,
          forced_routed_experts=sub_forced,
      )
      if kv_cache is not None:
        updated_kvs.append(new_kv)
    return y, (tuple(updated_kvs) if kv_cache is not None else None)
