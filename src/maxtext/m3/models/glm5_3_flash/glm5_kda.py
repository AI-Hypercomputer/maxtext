# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kimi Delta Attention (KDA) module and primitives for GLM-5.3-Flash."""

from typing import cast
from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config
from tokamax._src.ops.experimental.kda import api as tokamax_kda
from tokamax._src.ops.experimental.kda import pallas_mosaic_tpu as _tokamax_kda_mosaic  # pylint: disable=unused-import  # buildcleaner: keep


def l2norm(
    x: jax.Array,
    axis: int = -1,
    eps: float = 1e-6,
) -> jax.Array:
  """L2 normalization along a specified axis: x / sqrt(sum(x^2) + eps).

  Aligns with the l2norm implementation in the FLA library and GLM-5.3-Flash HF
  reference, which uses `+ eps` inside the sqrt rather than `max(norm, eps)`.

  Args:
    x: Input array.
    axis: Axis along which to compute the L2 norm (default -1).
    eps: Epsilon value added inside the square root (default 1e-6).

  Returns:
    Normalized array with the same shape and dtype as x.
  """
  input_dtype = x.dtype
  acc_dtype = jnp.promote_types(input_dtype, jnp.float32)
  x_acc = x.astype(acc_dtype)
  norm = jnp.sqrt(jnp.sum(x_acc * x_acc, axis=axis, keepdims=True) + eps)
  out = x_acc / norm
  return out.astype(input_dtype)


class Glm5RMSNormGated(nnx.Module):
  """RMSNorm with sigmoid gating: (weight * normed) * sigmoid(gate)."""

  def __init__(
      self,
      hidden_size: int,
      eps: float,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.hidden_size = hidden_size
    self.eps = eps
    self.param_dtype = param_dtype
    self.sharding = sharding

    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(
        jnp.ones((hidden_size,), dtype=param_dtype),
        sharding=spec,
    )

  def __call__(self, hidden_states: jax.Array, gate: jax.Array) -> jax.Array:
    """Forward pass for gated RMSNorm.

    Args:
      hidden_states: Input tensor of shape [..., hidden_size].
      gate: Gate tensor of same shape as hidden_states.

    Returns:
      Gated and normalized tensor of shape [..., hidden_size].
    """
    input_dtype = hidden_states.dtype
    h_fp32 = hidden_states.astype(jnp.float32)
    gate_fp32 = gate.astype(jnp.float32)

    variance = jnp.mean(jnp.square(h_fp32), axis=-1, keepdims=True)
    normed = h_fp32 * jax.lax.rsqrt(variance + self.eps)
    w = self.weight[...].astype(jnp.float32)
    out = (w * normed) * jax.nn.sigmoid(gate_fp32)
    return out.astype(input_dtype)


class Glm5ShortConv1d(nnx.Module):
  """1D causal depthwise convolution with SiLU activation."""

  def __init__(
      self,
      dim: int,
      kernel_size: int = 4,
      *,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs | None = None,
  ):
    del rngs
    self.dim = dim
    self.kernel_size = kernel_size
    self.param_dtype = param_dtype
    self.sharding = sharding

    spec = sharding("conv1d", ("heads", None, None)) if sharding else None
    self.weight = nnx.Param(
        jnp.zeros((dim, 1, kernel_size), dtype=param_dtype),
        sharding=spec,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass for causal depthwise 1D conv + SiLU.

    Args:
      x: Input tensor of shape [batch, seq_len, dim].

    Returns:
      Output tensor of shape [batch, seq_len, dim].
    """
    b, s, _ = x.shape
    k = self.kernel_size
    pad = jnp.zeros((b, k - 1, self.dim), dtype=x.dtype)
    x_padded = jnp.concatenate([pad, x], axis=1)

    w = self.weight[...].astype(x.dtype)
    out = sum(x_padded[:, i : i + s, :] * w[:, 0, i] for i in range(k))
    return jax.nn.silu(out)


class Glm5KdaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) layer for GLM-5.3-Flash."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sharding = sharding

    kda_impl = config.kda_implementation
    valid = {"xla", "xla_chunked", "mosaic"}
    if not isinstance(kda_impl, str) or kda_impl not in valid:
      msg = f"kda_implementation must be one of {valid}, got {kda_impl!r}."
      raise ValueError(msg)
    self.kda_implementation = kda_impl

    self.hidden_size = config.hidden_size
    self.num_heads = config.linear_num_key_heads
    self.head_dim = config.linear_key_head_dim
    self.qkv_dim = self.num_heads * self.head_dim

    self.dtype = getattr(jnp, config.dtype)
    self.param_dtype = getattr(jnp, config.weight_dtype)

    def _lin(in_f, out_f, name, axes):
      km = {"sharding": sharding(name, axes)} if sharding else {}
      return nnx.Linear(
          in_features=in_f,
          out_features=out_f,
          use_bias=False,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          kernel_metadata=km,
          rngs=rngs,
      )

    def _cv(dim):
      return Glm5ShortConv1d(
          dim=dim,
          kernel_size=config.linear_conv_kernel_dim,
          param_dtype=self.param_dtype,
          sharding=sharding,
      )

    h, q = self.hidden_size, self.qkv_dim
    d, n = self.head_dim, self.num_heads

    self.q_proj = _lin(h, q, "q_proj", ("embed", "heads"))
    self.k_proj = _lin(h, q, "k_proj", ("embed", "heads"))
    self.v_proj = _lin(h, q, "v_proj", ("embed", "heads"))

    self.q_conv1d, self.k_conv1d, self.v_conv1d = _cv(q), _cv(q), _cv(q)

    self.f_a_proj = _lin(h, d, "f_a_proj", ("embed", None))
    self.f_b_proj = _lin(d, q, "f_b_proj", (None, "heads"))

    s_h = sharding("A_log", ("heads",)) if sharding else None
    s_dt = sharding("dt_bias", ("heads",)) if sharding else None
    self.A_log = nnx.Param(
        jnp.zeros((n,), dtype=self.param_dtype),
        sharding=s_h,
    )
    self.dt_bias = nnx.Param(
        jnp.zeros((q,), dtype=self.param_dtype),
        sharding=s_dt,
    )

    self.b_proj = _lin(h, n, "b_proj", ("embed", "heads"))
    self.g_a_proj = _lin(h, d, "g_a_proj", ("embed", None))
    self.g_b_proj = _lin(d, q, "g_b_proj", (None, "heads"))

    self.o_norm = Glm5RMSNormGated(
        hidden_size=self.head_dim,
        eps=config.rms_norm_eps,
        param_dtype=self.param_dtype,
        sharding=sharding,
    )
    self.o_proj = _lin(q, h, "o_proj", ("heads", "embed"))

  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> jax.Array:
    """Applies KDA attention with projections, short conv, and recurrence."""
    b, s, _ = hidden_states.shape

    mask = None
    if decoder_segment_ids is not None:
      mask = (decoder_segment_ids != 0)[..., None]
      hidden_states = jnp.where(mask, hidden_states, 0)

    # Q, K, V projections followed by causal depthwise conv1d + SiLU
    q = self.q_conv1d(self.q_proj(hidden_states))
    k = self.k_conv1d(self.k_proj(hidden_states))
    v = self.v_conv1d(self.v_proj(hidden_states))

    # Low-rank forget gate & beta
    f = self.f_b_proj(self.f_a_proj(hidden_states))
    acc_dtype = jnp.promote_types(hidden_states.dtype, jnp.float32)
    b_val = self.b_proj(hidden_states).astype(acc_dtype)
    beta = jax.nn.sigmoid(b_val).astype(hidden_states.dtype)

    if mask is not None:
      q, k, v = (jnp.where(mask, x, 0) for x in (q, k, v))
      beta = jnp.where(mask, beta, 0)

    # Reshape and transpose to head-first: [H, B, S, D]
    head_shape = (b, s, self.num_heads, self.head_dim)

    def _tr(x):
      return jnp.transpose(x.reshape(head_shape), (2, 0, 1, 3))

    q_h, k_h, v_h, f_h = _tr(q), _tr(k), _tr(v), _tr(f)
    q_h, k_h = l2norm(q_h, axis=-1, eps=1e-6), l2norm(k_h, axis=-1, eps=1e-6)
    beta_h = jnp.transpose(beta, (2, 0, 1))

    # Tokamax KDA
    impl = cast(tokamax_kda.Implementation, self.kda_implementation)
    attn_out, _ = tokamax_kda.kimi_delta_attention(
        query=q_h,
        key=k_h,
        value=v_h,
        gate=f_h,
        beta=beta_h,
        a_log=self.A_log[...],
        delta_time_bias=self.dt_bias[...],
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        use_qk_l2norm=False,
        implementation=impl,
    )

    # Transpose back: [H, B, S, D] -> [B, S, H, D]
    attn_out = jnp.transpose(attn_out, (1, 2, 0, 3))

    # Output gate and gated RMSNorm
    g_low = self.g_a_proj(hidden_states)
    g = self.g_b_proj(g_low).reshape(head_shape)
    normed = self.o_norm(attn_out, g)
    output = self.o_proj(normed.reshape(b, s, self.qkv_dim))

    if mask is not None:
      output = jnp.where(mask, output, 0)

    if self.sharding:
      axes = ("batch", "length", "embed")
      output = self.sharding.constrain(output, "kda_out", axes)

    return output
