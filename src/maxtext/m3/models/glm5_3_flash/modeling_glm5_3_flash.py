# Copyright 2026 Google LLC
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

"""Consolidated GLM-5.3-Flash autoregressive language model in m3 format."""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any, Literal, cast
import typing

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.typing import DTypeLike
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import glm5_config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Config
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5Sharding
import optax
from tokamax._src.ops.kda import api as tokamax_kda
from tokamax._src.ops.kda import pallas_mosaic_tpu as _tokamax_kda_mosaic  # pylint: disable=unused-import  # buildcleaner: keep

# ============================================================================
# Section 1: Linear Attention Primitives (KDA)
# ============================================================================


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
        out_sharding=spec,
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
    if rngs is None:
      raise ValueError("Glm5ShortConv1d requires rngs for weight initialization.")
    self.dim = dim
    self.kernel_size = kernel_size
    self.param_dtype = param_dtype
    self.sharding = sharding

    spec = sharding("conv1d", ("heads", None, None)) if sharding else None
    bound = 1.0 / math.sqrt(kernel_size)
    w_init = jax.random.uniform(
        rngs.params(),
        (dim, 1, kernel_size),
        dtype=jnp.float32,
        minval=-bound,
        maxval=bound,
    ).astype(param_dtype)
    self.weight = nnx.Param(w_init, out_sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Forward pass for causal depthwise 1D conv + SiLU.

    Args:
      x: Input tensor of shape [batch, seq_len, dim].

    Returns:
      Output tensor of shape [batch, seq_len, dim].
    """
    input_dtype = x.dtype
    b, s, _ = x.shape
    k = self.kernel_size
    x_fp32 = x.astype(jnp.float32)
    pad = jnp.zeros((b, k - 1, self.dim), dtype=jnp.float32)
    x_padded = jnp.concatenate([pad, x_fp32], axis=1)
    w_fp32 = self.weight[...].astype(jnp.float32)
    out = sum(x_padded[:, i : i + s, :] * w_fp32[:, 0, i] for i in range(k))
    return jax.nn.silu(out).astype(input_dtype)


class Glm5KdaAttention(nnx.Module):
  """Kimi Delta Attention (KDA) layer for GLM-5.3-Flash.

  Supports unpacked sequences only (`decoder_segment_ids` entries in `{0, 1}`
  where `0` is padding and `1` is the single active document, enforced at the
  config/entry boundary by `validate_no_packing`).

  `kda_implementation` selects the execution backend with no silent fallback:
  - `"xla"`: Tokamax token-by-token recurrence reference for unit tests.
  - `"xla_chunked"`: Chunked XLA backend usable on CPU and TPU v5.
  - `"mosaic"`: TPU v6+ Pallas hardware kernel.
  """

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
    if kda_impl in {"xla_chunked", "mosaic"} and config.linear_lower_bound is None:
      raise ValueError("linear_lower_bound cannot be None when kda_implementation is" f" {kda_impl!r}.")
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
          precision=jax.lax.Precision.HIGHEST,
          kernel_metadata=km,
          rngs=rngs,
      )

    def _cv(dim):
      return Glm5ShortConv1d(
          dim=dim,
          kernel_size=config.linear_conv_kernel_dim,
          param_dtype=jnp.float32,
          sharding=sharding,
          rngs=rngs,
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
    if config.linear_lower_bound is not None:
      a_log_init = jnp.zeros((n,), dtype=jnp.float32)
    else:
      a_init = jax.random.uniform(
          rngs.params(),
          (n,),
          dtype=jnp.float32,
          minval=1.0,
          maxval=16.0,
      )
      a_log_init = jnp.log(a_init)
    self.A_log = nnx.Param(
        a_log_init,
        out_sharding=s_h,
    )

    log_dt = jax.random.uniform(
        rngs.params(),
        (q,),
        dtype=jnp.float32,
        minval=math.log(1e-3),
        maxval=math.log(1e-1),
    )
    dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
    dt_bias_init = (dt + jnp.log(-jnp.expm1(-dt))).astype(jnp.float32)
    self.dt_bias = nnx.Param(dt_bias_init, out_sharding=s_dt)

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
    """Applies KDA attention with projections, short conv, and recurrence.

    Supports unpacked sequences only (`decoder_segment_ids` entries in `{0, 1}`
    where `0` is padding and `1` is the single active document).

    Args:
      hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
      decoder_segment_ids: Optional segment IDs in {0, 1} for right padding.

    Returns:
      Output tensor of shape [batch, seq_len, hidden_size].
    """
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

    def _run_kda(q_arg, k_arg, v_arg, f_arg, beta_arg, a_log_arg, dt_bias_arg):
      out, _ = tokamax_kda.kimi_delta_attention(
          query=q_arg,
          key=k_arg,
          value=v_arg,
          gate=f_arg,
          beta=beta_arg,
          a_log=a_log_arg,
          delta_time_bias=dt_bias_arg,
          use_gate_in_kernel=True,
          lower_bound=self.config.linear_lower_bound,
          use_qk_l2norm=False,
          implementation=impl,
      )
      return out

    if self.sharding is not None and not jax.sharding.get_abstract_mesh().empty:
      head_spec = self.sharding(
          "kda_head_act",
          ("heads", "batch", None, None),
          m3_sharding.TensorType.ACTIVATION,
      )
      beta_spec = self.sharding(
          "kda_beta_act",
          ("heads", "batch", None),
          m3_sharding.TensorType.ACTIVATION,
      )
      param_spec = self.sharding("kda_head_param", ("heads",))
      batch_axes = head_spec[1]
      if isinstance(batch_axes, str):
        batch_axes = (batch_axes,)
      mesh_axes = set(jax.sharding.get_abstract_mesh().axis_names)
      active_batch_axes = tuple(ax for ax in batch_axes or () if ax in mesh_axes)

      def _sharded_kda(q_arg, k_arg, v_arg, f_arg, beta_arg, a_log_arg, dt_bias_arg):
        if active_batch_axes:
          a_log_arg = jax.lax.pcast(a_log_arg, active_batch_axes, to="varying")
          dt_bias_arg = jax.lax.pcast(dt_bias_arg, active_batch_axes, to="varying")
        return _run_kda(q_arg, k_arg, v_arg, f_arg, beta_arg, a_log_arg, dt_bias_arg)

      attn_out = jax.shard_map(
          _sharded_kda,
          in_specs=(
              head_spec,
              head_spec,
              head_spec,
              head_spec,
              beta_spec,
              param_spec,
              param_spec,
          ),
          out_specs=head_spec,
          check_vma=False,
      )(q_h, k_h, v_h, f_h, beta_h, self.A_log[...], self.dt_bias[...])
    else:
      attn_out = _run_kda(q_h, k_h, v_h, f_h, beta_h, self.A_log[...], self.dt_bias[...])

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


# ============================================================================
# Section 2: Attention & Indexer (DSA / MLA)
# ============================================================================


class Glm5RMSNorm(nnx.Module):
  """RMSNorm matching Glm5NextTextRMSNorm (T5LayerNorm)."""

  def __init__(
      self,
      d: int,
      *,
      eps: float,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    normed = (x32 * jax.lax.rsqrt(v + self.eps)).astype(in_dt)
    return self.weight[...].astype(in_dt) * normed


class Glm5LayerNorm(nnx.Module):
  """LayerNorm with weight and bias matching nn.LayerNorm."""

  def __init__(
      self,
      d: int,
      *,
      eps: float,
      param_dtype: DTypeLike = jnp.float32,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.eps = eps
    spec = sharding("norm", ("norm",)) if sharding else None
    self.weight = nnx.Param(jnp.ones((d,), dtype=param_dtype), sharding=spec)
    self.bias = nnx.Param(jnp.zeros((d,), dtype=param_dtype), sharding=spec)

  def __call__(self, x: jax.Array) -> jax.Array:
    in_dt, x32 = x.dtype, x.astype(jnp.float32)
    m = jnp.mean(x32, axis=-1, keepdims=True)
    v = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - m) * jax.lax.rsqrt(v + self.eps)
    w = self.weight[...].astype(jnp.float32)
    b = self.bias[...].astype(jnp.float32)
    return (normed * w + b).astype(in_dt)


def _make_lin(dt: DTypeLike, p_dt: DTypeLike, sharding, rngs: nnx.Rngs):
  """Constructs a Linear layer factory closure with sharding metadata."""

  def _lin(in_f: int, out_f: int, name: str, axes: tuple[str | None, ...]):
    km = {"sharding": sharding(name, axes)} if sharding else {}
    return nnx.Linear(
        in_f,
        out_f,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        # Precision.HIGHEST is required on TPU so MXU matmuls match IEEE fp32
        # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=km,
        rngs=rngs,
    )

  return _lin


class Glm5Indexer(nnx.Module):
  """DeepSeek Sparse Attention (DSA) indexer with k-pool compression."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    h, d = config.hidden_size, config.index_head_dim
    n_heads, q_rank = config.index_n_heads, config.q_lora_rank
    self.head_dim = d
    self.index_topk = config.index_topk
    self.index_kpool = config.index_kpool
    self.index_kpool_always_select_tail = config.index_kpool_always_select_tail

    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)
    lin = _make_lin(dt, p_dt, sharding, rngs)

    self.wq_b = lin(q_rank, n_heads * d, "wq_b", ("q_lora", "heads"))
    self.wk = lin(h, d, "wk", ("embed", None))
    self.k_norm = Glm5LayerNorm(
        d,
        eps=1e-6,
        param_dtype=p_dt,
        sharding=sharding,
    )
    self.weights_proj = lin(h, n_heads, "weights_proj", ("embed", None))

    s_ape = (None, None)
    ape_spec = sharding("index_kpool_compress_ape", s_ape) if sharding else None
    ape_init = jnp.zeros((self.index_kpool, d), dtype=p_dt)
    self.index_kpool_compress_ape = nnx.Param(ape_init, sharding=ape_spec)

    s_gate = (None, "embed")
    g_name = "index_kpool_compress_gate"
    gate_spec = sharding(g_name, s_gate) if sharding else None
    gate_init = jnp.ones((d, h), dtype=p_dt)
    self.index_kpool_compress_gate = nnx.Param(gate_init, sharding=gate_spec)

  def __call__(
      self,
      hidden_states: jax.Array,
      q_resid: jax.Array,
      attention_mask: jax.Array | None = None,
  ) -> jax.Array:
    hidden_states = jax.lax.stop_gradient(hidden_states)
    q_resid = jax.lax.stop_gradient(q_resid)
    b, s = hidden_states.shape[:2]
    if attention_mask is None:
      attention_mask = jnp.ones((b, s), dtype=jnp.bool_)
    else:
      attention_mask = attention_mask.astype(jnp.bool_)

    q = self.wq_b(q_resid).reshape(b, s, -1, self.head_dim)
    k = self.k_norm(self.wk(hidden_states))
    gate_w = self.index_kpool_compress_gate[...].astype(hidden_states.dtype)
    # Precision.HIGHEST is required on TPU so MXU einsums match IEEE fp32
    # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
    gate = jnp.einsum(
        "bsd,hd->bsh",
        hidden_states,
        gate_w,
        precision=jax.lax.Precision.HIGHEST,
    )

    num_pools = (s + self.index_kpool - 1) // self.index_kpool
    any_valid = jnp.any(attention_mask, axis=-1)
    first_key = jnp.where(any_valid, jnp.argmax(attention_mask, axis=-1), s)
    tot_p = num_pools * self.index_kpool
    offsets = jnp.arange(tot_p).reshape(1, num_pools, self.index_kpool)
    raw_pool_indices = first_key[:, None, None] + offsets

    b_idx = jnp.arange(b)[:, None, None]
    safe_idx = jnp.clip(raw_pool_indices, 0, s - 1)
    grp_keys = k[b_idx, safe_idx]
    grp_gate = gate[b_idx, safe_idx]
    grp_valid = attention_mask[b_idx, safe_idx] & (raw_pool_indices < s)
    pool_valid = jnp.all(grp_valid, axis=-1)
    pool_indices = jnp.where(grp_valid, raw_pool_indices, -1)

    ape = self.index_kpool_compress_ape[...].astype(jnp.float32)
    gate_ape = grp_gate.astype(jnp.float32) + ape[None, None]
    logits = jnp.where(grp_valid[..., None], gate_ape, -jnp.inf)
    probs = jnp.nan_to_num(jax.nn.softmax(logits, axis=2), nan=0.0)
    pool_keys = jnp.sum(probs.astype(grp_keys.dtype) * grp_keys, axis=2)

    q_f32, p_f32 = q.astype(jnp.float32), pool_keys.astype(jnp.float32)
    # Precision.HIGHEST is required on TPU so MXU einsums match IEEE fp32
    # accumulation and avoid discrete top_k tie-breaking drift vs PyTorch.
    raw_scores = jnp.einsum("bshd,bpd->bshp", q_f32, p_f32, precision=jax.lax.Precision.HIGHEST)
    scores = jax.nn.relu(raw_scores * (self.head_dim**-0.5))
    scale_w = q.shape[2] ** -0.5
    w = self.weights_proj(hidden_states).astype(jnp.float32) * scale_w
    index_scores = jnp.einsum("bsh,bshp->bsp", w, scores, precision=jax.lax.Precision.HIGHEST)

    s_idx = jnp.arange(s)[None, :, None]
    p_end = jnp.clip(raw_pool_indices[..., -1], 0, s - 1)[:, None, :]
    valid_cand = (p_end <= s_idx) & pool_valid[:, None, :]
    min_score = jnp.finfo(jnp.float32).min
    index_scores = jnp.where(valid_cand, index_scores, min_score)

    sel_k = min(self.index_topk // self.index_kpool, num_pools)
    _, selected = jax.lax.top_k(index_scores, sel_k)
    sel_valid = valid_cand[b_idx, s_idx, selected]
    sel_indices = pool_indices[b_idx, selected]
    flat_len = sel_k * self.index_kpool
    v_rep = jnp.repeat(sel_valid[..., None], self.index_kpool, axis=-1)
    v_mask = v_rep.reshape(b, s, flat_len)
    topk_indices = jnp.where(v_mask, sel_indices.reshape(b, s, flat_len), -1)

    if self.index_kpool_always_select_tail:
      max_tail = self.index_kpool - 1
      if max_tail > 0:
        vis_cnt = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1)
        tail_cnt = vis_cnt % self.index_kpool
        t_offsets = jnp.arange(max_tail)
        t_start = first_key[:, None] + vis_cnt - tail_cnt
        t_indices = t_start[..., None] + t_offsets[None, None, :]
        t_cnt = tail_cnt[..., None]
        t_valid = (t_offsets[None, None, :] < t_cnt) & (t_indices < s)
        t_safe = jnp.clip(t_indices, 0, s - 1)
        t_vis = (t_safe <= s_idx) & attention_mask[b_idx, t_safe]
        t_masked = jnp.where(t_valid & t_vis, t_indices, -1)
        topk_indices = jnp.concatenate([topk_indices, t_masked], axis=-1)
      out_w = self.index_topk + max_tail
    else:
      out_w = self.index_topk

    diff_w = out_w - topk_indices.shape[-1]
    if diff_w > 0:
      pad = ((0, 0), (0, 0), (0, diff_w))
      topk_indices = jnp.pad(topk_indices, pad, constant_values=-1)
    res = jnp.where(attention_mask[..., None], topk_indices[..., :out_w], -1)
    return res.astype(jnp.int32)


class Glm5DsaMlaAttention(nnx.Module):
  """NoPE Multi-head Latent Attention with DSA Lightning Indexer.

  This dense [B, H, S, S] masked attention is the reference implementation
  for parity verification against HuggingFace Glm5NextTextAttention.
  TODO: Add Splash block-sparse Pallas kernel execution for sparse DSA
  attention.
  """

  def __init__(
      self,
      config: Glm5Config,
      layer_idx: int,
      *,
      indexer_type: str,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    if indexer_type != "full":
      raise ValueError(f"Shared indexer unsupported: {indexer_type!r}.")
    self.config, self.layer_idx = config, layer_idx
    self.indexer_type = indexer_type
    self.sharding = sharding
    if config.qk_rope_head_dim != 0 or config.rope_parameters is not None:
      raise ValueError(f"NoPE violation: {config.qk_rope_head_dim=}.")
    ilsf = config.indexer_loss_scaling_factor
    if ilsf != 0.0:
      # TODO: Support non-zero indexer_loss_scaling_factor for auxiliary indexer loss.
      raise ValueError(f"indexer_loss_scaling_factor != 0 is not yet supported: {ilsf}.")

    h, n_heads = config.hidden_size, config.num_attention_heads
    q_rank, kv_rank = config.q_lora_rank, config.kv_lora_rank
    qk_dim, v_dim = config.qk_nope_head_dim, config.v_head_dim
    self.scaling = qk_dim**-0.5
    self.num_heads = n_heads
    self.qk_nope_head_dim = qk_dim
    self.v_head_dim = v_dim

    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)
    lin = _make_lin(dt, p_dt, sharding, rngs)
    eps = config.rms_norm_eps

    def _norm(dim):
      return Glm5RMSNorm(dim, eps=eps, param_dtype=p_dt, sharding=sharding)

    self.q_a_proj = lin(h, q_rank, "q_a_proj", ("embed", None))
    self.q_a_layernorm = _norm(q_rank)
    self.q_b_proj = lin(q_rank, n_heads * qk_dim, "q_b_proj", ("q_lora", "heads"))

    self.kv_a_proj_with_mqa = lin(h, kv_rank, "kv_a_proj_with_mqa", ("embed", None))
    self.kv_a_layernorm = _norm(kv_rank)
    kv_tot = n_heads * (qk_dim + v_dim)
    self.kv_b_proj = lin(kv_rank, kv_tot, "kv_b_proj", ("kv_lora", "heads"))
    self.o_proj = lin(n_heads * v_dim, h, "o_proj", ("heads", "embed"))
    self.indexer = Glm5Indexer(config, sharding=sharding, rngs=rngs) if config.use_indexer else None

  def _build_causal_padding_mask(
      self,
      attn_mask: jax.Array | None,
      b: int,
      s: int,
      dtype: DTypeLike,
  ) -> jax.Array:
    """Builds dense additive attention mask [B, 1, S, S] combining causal and validity masks."""
    pos = jnp.arange(s)
    causal = pos[None, None, None, :] <= pos[None, None, :, None]
    if attn_mask is not None:
      vis = causal & attn_mask[:, None, None, :] & attn_mask[:, None, :, None]
    else:
      vis = jnp.broadcast_to(causal, (b, 1, s, s))
    min_val = jnp.finfo(dtype).min
    return jnp.where(vis, 0.0, min_val).astype(dtype)

  def build_attention_mask_from_topk(
      self,
      topk_indices: jax.Array,
      query_states: jax.Array,
      kv_len: int,
  ) -> jax.Array:
    """Converts top-k indices to additive attention bias mask [B, 1, S, kv_len]."""
    b, s = topk_indices.shape[:2]
    topk_valid = (topk_indices >= 0) & (topk_indices < kv_len)
    target_idx = jnp.where(topk_valid, topk_indices, kv_len)
    b_idx = jnp.arange(b)[:, None, None]
    s_idx = jnp.arange(s)[None, :, None]
    buf = jnp.zeros((b, s, kv_len + 1), dtype=jnp.bool_)
    mask = buf.at[b_idx, s_idx, target_idx].set(True)[:, None, :, :kv_len]
    min_val = jnp.finfo(query_states.dtype).min
    return jnp.where(mask, 0.0, min_val).astype(query_states.dtype)

  def _call_dense_mask(
      self,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      mask_or_topk: jax.Array,
      decoder_segment_ids: jax.Array | None,
      s: int | None = None,
  ) -> jax.Array:
    """Computes attention using dense bias mask [B, 1, S, S] with unified zero-padding."""
    if mask_or_topk.ndim == 4:
      mask = mask_or_topk
    else:
      if s is None:
        s = q.shape[2]
      mask = self.build_attention_mask_from_topk(mask_or_topk, q, s)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=jax.lax.Precision.HIGHEST) * self.scaling + mask
    scores_f32 = scores.astype(jnp.float32)
    weights = jax.nn.softmax(scores_f32, axis=-1).astype(q.dtype)
    out = jnp.matmul(weights, v, precision=jax.lax.Precision.HIGHEST)
    if decoder_segment_ids is not None:
      valid_q = (decoder_segment_ids != 0)[:, None, :, None]
      out = jnp.where(valid_q, out, 0.0)
    return out

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: Literal[False] = False,
  ) -> jax.Array:
    ...

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: Literal[True],
  ) -> tuple[jax.Array, jax.Array | None]:
    ...

  @typing.overload
  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    ...

  def __call__(
      self,
      hidden_states: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      *,
      _force_indexer: bool = False,
      return_topk_indices: bool = False,
  ) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    """Applies NoPE MLA attention with DSA top-k indexer masking.

    Supports unpacked sequences only (`decoder_segment_ids` entries in `{0, 1}`
    where `0` is padding and `1` is the single active document; packing is
    rejected by `Glm5Config(packing=False)` (`ValueError` raised in
    `Glm5Config.__post_init__` when `packing=True`); TODO: add sequence packing
    support).

    Args:
      hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
      decoder_segment_ids: Optional segment IDs in {0, 1} for right/left
        padding.
      _force_indexer: Internal flag to force running the indexer path for parity
        testing.
      return_topk_indices: If True, returns (output, topk_idx) tuple.

    Returns:
      Output tensor of shape [batch, seq_len, hidden_size] or tuple.
    """
    b, s = hidden_states.shape[:2]
    attn_mask = None
    if decoder_segment_ids is not None:
      attn_mask = decoder_segment_ids != 0

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_heads, qk_dim = self.num_heads, self.qk_nope_head_dim
    q = self.q_b_proj(q_resid).reshape(b, s, q_heads, qk_dim)
    q = q.transpose(0, 2, 1, 3)

    kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
    kv_dim = qk_dim + self.v_head_dim
    kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
    kv = kv.reshape(b, s, q_heads, kv_dim).transpose(0, 2, 1, 3)
    k, v = jnp.split(kv, [qk_dim], axis=-1)

    if not _force_indexer and (
        not self.config.use_indexer or (s <= self.config.index_topk and self.config.index_kpool_always_select_tail)
    ):
      mask = self._build_causal_padding_mask(attn_mask, b, s, q.dtype)
      topk_idx = (
          self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
          if (return_topk_indices and self.indexer is not None)
          else None
      )
    else:
      if self.indexer is None:
        raise ValueError("indexer is None but indexer path was requested.")
      topk_idx = self.indexer(hidden_states, q_resid, attention_mask=attn_mask)
      mask = self.build_attention_mask_from_topk(topk_idx, q, s)

    if self.config.dsa_attention_implementation == "dense_mask":
      out = self._call_dense_mask(q, k, v, mask, decoder_segment_ids)
    else:
      raise ValueError(
          "Unsupported Glm5Config.dsa_attention_implementation:" f" {self.config.dsa_attention_implementation!r}."
      )

    out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
    out = self.o_proj(out)
    if self.sharding:
      axes = ("batch", "length", "embed")
      out = self.sharding.constrain(out, "mla_out", axes)
    if return_topk_indices:
      return out, topk_idx
    return out


# ============================================================================
# Section 3: MoE & Clamped-SwiGLU MLP
# ============================================================================


class Glm5RouterBias(nnx.Variable):
  """Router bias state. Subclasses nnx.Variable to exclude from gradients."""


class Glm5MLP(nnx.Module):
  """Clamped SwiGLU dense MLP matching Glm5NextTextMLP."""

  def __init__(
      self,
      config: Glm5Config,
      intermediate_size: int,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    self.hidden_size = h = config.hidden_size
    self.intermediate_size = inter = intermediate_size
    self.swiglu_limit = config.mlp_clamp_limit

    dt, p_dt = getattr(jnp, config.dtype), getattr(jnp, config.weight_dtype)

    def _lin(in_f, out_f, name, axes):
      km = {"sharding": sharding(name, axes)} if sharding else {}
      return nnx.Linear(
          in_f,
          out_f,
          use_bias=False,
          dtype=dt,
          param_dtype=p_dt,
          precision=jax.lax.Precision.HIGHEST,
          kernel_metadata=km,
          rngs=rngs,
      )

    self.gate_proj = _lin(h, inter, "gate_proj", ("embed", "mlp"))
    self.up_proj = _lin(h, inter, "up_proj", ("embed", "mlp"))
    self.down_proj = _lin(inter, h, "down_proj", ("mlp", "embed"))

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = jnp.minimum(self.gate_proj(x), self.swiglu_limit)
    up = jnp.clip(self.up_proj(x), -self.swiglu_limit, self.swiglu_limit)
    return self.down_proj(jax.nn.silu(gate) * up)


class Glm5TopkRouter(nnx.Module):
  """Top-k sigmoid router with non-Param e_score_correction_bias."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    if (sf := config.scoring_func) != "sigmoid":
      raise ValueError(f"scoring_func must be 'sigmoid': {sf!r}.")

    self.hidden_dim = h = config.hidden_size
    self.num_experts = e = config.n_routed_experts
    self.top_k, self.num_group = config.num_experts_per_tok, config.n_group
    self.topk_group = config.topk_group
    self.routed_scaling_factor = config.routed_scaling_factor
    self.norm_topk_prob = config.norm_topk_prob

    r_p_dt = getattr(jnp, config.router_param_dtype)
    w_spec = sharding("router_weight", ("embed", None)) if sharding else None
    b_spec = sharding("router_bias", (None,)) if sharding else None
    w_init = nnx.initializers.normal(0.02)(rngs.params(), (h, e), r_p_dt)
    self.weight = nnx.Param(w_init, sharding=w_spec)
    self.e_score_correction_bias = Glm5RouterBias(
        jnp.zeros((e,), dtype=r_p_dt),
        sharding=b_spec,
    )

  def __call__(
      self,
      hidden_states: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    h_flat = hidden_states.reshape(-1, self.hidden_dim).astype(jnp.float32)
    router_logits = jnp.matmul(
        h_flat,
        self.weight[...].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = jax.nn.sigmoid(router_logits)
    bias_arr = self.e_score_correction_bias[...].astype(jnp.float32)
    scores_for_choice = jax.lax.stop_gradient(scores + bias_arr)

    exp_per_group = self.num_experts // self.num_group
    grouped = scores_for_choice.reshape(-1, self.num_group, exp_per_group)
    group_scores = jnp.sum(jax.lax.top_k(grouped, 2)[0], axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, self.topk_group)
    g_arange = jnp.arange(self.num_group)[None, :]
    group_mask = jnp.any(g_arange[:, :, None] == group_idx[:, None, :], axis=-1)
    score_mask = jnp.repeat(group_mask, exp_per_group, axis=-1)
    masked_scores = jnp.where(
        score_mask,
        scores_for_choice,
        jnp.finfo(jnp.float32).min,
    )

    _, topk_indices = jax.lax.top_k(masked_scores, self.top_k)
    topk_weights = jnp.take_along_axis(scores, topk_indices, axis=-1)
    if self.norm_topk_prob:
      denom = jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
      topk_weights = topk_weights / denom
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices


class Glm5Experts(nnx.Module):
  """Collection of expert weights stored as 3D/4D tensors.

  Replicated-EP (expert == 1) global-dispatch jax.lax.ragged_dot reference
  implementation (when data or fsdp > 1, GSPMD all-gathers the [B*S*K]
  token-expert indices for global argsort and ragged_dot; expert > 1 is rejected
  by Glm5Experts and Glm5MoE).
  """

  def __init__(
      self,
      config: Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided.")
    self.config = config
    e, h = config.n_routed_experts, config.hidden_size
    inter = config.moe_intermediate_size
    self.hidden_size = h
    self.intermediate_size = inter
    self.num_experts, self.top_k = e, config.num_experts_per_tok
    self.swiglu_limit = config.mlp_clamp_limit
    self.sharding = sharding
    self.ep_axis = sharding.ep_axis if isinstance(sharding, Glm5Sharding) else "expert"
    if isinstance(sharding, Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5Experts expert parallelism ({self.ep_axis}={sharding.ep_size} >"
          " 1) requires shard_map EP dispatch (follow-up F34)."
      )
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError(
          "Glm5Experts expert parallelism"
          f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires shard_map"
          " EP dispatch (follow-up F34)."
      )

    p_dt = getattr(jnp, config.weight_dtype)
    gu_axes = ("expert", "embed", None, "mlp")
    d_axes = ("expert", "mlp", "embed")
    gu_spec = sharding("gate_up_proj", gu_axes) if sharding else None
    d_spec = sharding("down_proj", d_axes) if sharding else None

    gu_init = nnx.initializers.normal(0.02)(rngs.params(), (e, h, 2, inter), p_dt)
    self.gate_up_proj = nnx.Param(gu_init, sharding=gu_spec)

    d_init = nnx.initializers.normal(0.02)(rngs.params(), (e, inter, h), p_dt)
    self.down_proj = nnx.Param(d_init, sharding=d_spec)

  def __call__(
      self,
      hidden_states: jax.Array,
      topk_indices: jax.Array,
      topk_weights: jax.Array,
  ) -> jax.Array:
    if isinstance(self.sharding, Glm5Sharding) and self.sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map EP"
          f" dispatch (follow-up F34); got ep_size={self.sharding.ep_size}."
      )
    mesh = None
    x_shd = jax.typeof(hidden_states).sharding
    param_shd = jax.typeof(self.gate_up_proj.value).sharding
    if isinstance(x_shd, jax.sharding.NamedSharding):
      mesh = x_shd.mesh
    elif isinstance(param_shd, jax.sharding.NamedSharding):
      mesh = param_shd.mesh
    elif not jax.sharding.get_abstract_mesh().empty:
      mesh = jax.sharding.get_abstract_mesh()

    if mesh is not None and not mesh.empty:
      if self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
        raise NotImplementedError(
            f"Glm5Experts.__call__ with {self.ep_axis} > 1 requires shard_map"
            " EP dispatch (follow-up F34); got mesh"
            f" {self.ep_axis}={mesh.shape[self.ep_axis]}."
        )

    flat_idx = topk_indices.reshape(-1)
    flat_w = topk_weights.reshape(-1, 1).astype(jnp.float32)
    perm = jnp.argsort(flat_idx)
    sorted_h = hidden_states[perm // self.top_k]
    inv_perm = jnp.argsort(perm)
    group_sizes = jnp.bincount(flat_idx, length=self.num_experts)

    dt = hidden_states.dtype
    gu = self.gate_up_proj[...].astype(dt)
    gate = jax.lax.ragged_dot(
        sorted_h,
        gu[:, :, 0, :],
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    up = jax.lax.ragged_dot(
        sorted_h,
        gu[:, :, 1, :],
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    act = jax.nn.silu(gate) * up
    d_w = self.down_proj[...].astype(dt)
    sorted_out = jax.lax.ragged_dot(
        act,
        d_w,
        group_sizes=group_sizes,
        preferred_element_type=dt,
        precision=jax.lax.Precision.HIGHEST,
    )
    n = hidden_states.shape[0]
    weighted = (sorted_out[inv_perm].astype(jnp.float32) * flat_w).reshape(n, self.top_k, -1)
    return jnp.sum(weighted, axis=1).astype(dt)


class Glm5MoE(nnx.Module):
  """Mixed expert module with routed experts and shared experts."""

  def __init__(
      self,
      config: Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    self.ep_axis = sharding.ep_axis if isinstance(sharding, Glm5Sharding) else "expert"
    if isinstance(sharding, Glm5Sharding) and sharding.ep_size > 1:
      raise NotImplementedError(
          f"Glm5MoE expert parallelism ({self.ep_axis}={sharding.ep_size} > 1)"
          " requires all-to-all dispatch (planned for G8); single-host bringup"
          f" requires {self.ep_axis}=1."
      )
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty and self.ep_axis in mesh.shape and mesh.shape[self.ep_axis] > 1:
      raise NotImplementedError(
          "Glm5MoE expert parallelism"
          f" ({self.ep_axis}={mesh.shape[self.ep_axis]} > 1) requires"
          " all-to-all dispatch (planned for G8); single-host bringup requires"
          f" {self.ep_axis}=1."
      )
    self.experts = Glm5Experts(config, rngs=rngs, sharding=sharding)
    self.gate = Glm5TopkRouter(config, rngs=rngs, sharding=sharding)
    self.shared_experts = Glm5MLP(
        config,
        config.moe_intermediate_size * config.n_shared_experts,
        rngs=rngs,
        sharding=sharding,
    )

  def __call__(self, hidden_states: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    orig_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, orig_shape[-1])
    router_logits, topk_weights, topk_indices = self.gate(h_flat)
    expert_out = self.experts(h_flat, topk_indices, topk_weights)
    expert_out = expert_out.reshape(orig_shape)
    shared_out = self.shared_experts(hidden_states)
    topk_reshaped = topk_indices.reshape(orig_shape[:-1] + (self.config.num_experts_per_tok,))
    return expert_out + shared_out, (router_logits, topk_reshaped)


def init_moe_lb_stats(
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Initializes [E]-sized accumulators (tokens_sum, prob_sum, total_rows) for MoE LB loss."""
  z = jnp.zeros((num_experts,), dtype=jnp.float32)
  return z, z, jnp.zeros((), dtype=jnp.float32)


def accumulate_moe_lb_stats(
    stats: tuple[jax.Array, jax.Array, jax.Array],
    layer_gate: jax.Array,
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Accumulates one MoE layer's (bincount, prob_sum, rows) into [E]-sized carry stats.

  Matches HuggingFace's load_balancing_loss_func (Switch Transformer eq. 4-6):
  routing_weights = softmax(layer_gate, axis=-1) and selected_experts =
  top_k(routing_weights) without e_score_correction_bias.
  """
  tokens_sum, prob_sum, total_rows = stats
  gate_2d = layer_gate.reshape(-1, layer_gate.shape[-1]).astype(jnp.float32)
  routing_weights = jax.nn.softmax(gate_2d, axis=-1)
  _, selected_experts = jax.lax.top_k(routing_weights, top_k)
  selected_flat = selected_experts.reshape(-1)

  if attention_mask is None:
    b_cnt = jnp.bincount(selected_flat, length=num_experts)
    probs = routing_weights
    rows = jnp.asarray(routing_weights.shape[0], dtype=jnp.float32)
  else:
    flat_mask = attention_mask.reshape(-1).astype(jnp.float32)
    mask_rep = jnp.repeat(flat_mask, top_k)
    b_cnt = jnp.bincount(selected_flat, weights=mask_rep, length=num_experts)
    probs = routing_weights * flat_mask[:, None]
    rows = jnp.sum(flat_mask)

  return (
      tokens_sum + b_cnt.astype(jnp.float32),
      prob_sum + jnp.sum(probs, axis=0),
      total_rows + rows,
  )


def finalize_moe_lb_loss(
    stats: tuple[jax.Array, jax.Array, jax.Array],
    num_experts: int,
) -> jax.Array:
  """Finalizes MoE load-balancing loss scalar from accumulated [E]-sized stats."""
  tokens_sum, prob_sum, total_rows = stats
  safe_rows = jnp.maximum(total_rows, 1.0)
  t_exp, r_prob = tokens_sum / safe_rows, prob_sum / safe_rows
  return jnp.sum(t_exp * r_prob) * num_experts


def compute_moe_load_balancing_loss(
    gate_logits: Sequence[jax.Array],
    num_experts: int,
    top_k: int,
    attention_mask: jax.Array | None = None,
) -> jax.Array:
  """Computes auxiliary load balancing loss matching HF load_balancing_loss_func."""
  if not gate_logits:
    raise ValueError("gate_logits must be non-empty.")

  stats = init_moe_lb_stats(num_experts)
  for layer_gate in gate_logits:
    stats = accumulate_moe_lb_stats(
        stats,
        layer_gate,
        num_experts,
        top_k,
        attention_mask=attention_mask,
    )
  return finalize_moe_lb_loss(stats, num_experts)


# ============================================================================
# Section 4: Hyper-Connections & Decoder Layer
# ============================================================================


class Glm5UnweightedRMSNorm(nnx.Module):
  """Unweighted RMSNorm: x * rsqrt(mean(x^2) + eps)."""

  def __init__(self, eps: float):
    self.eps = eps

  def __call__(self, x: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    v = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(v + self.eps)).astype(x.dtype)


class Glm5HyperConnection(nnx.Module):
  """Manifold-Constrained Hyper-Connections (mHC) with 20 Sinkhorn iterations.

  Architectural and numerical differences vs legacy src/maxtext/layers/mhc.py:
  1. m3 self-containment rule: m3/models/glm5_3_flash does not import legacy
     src/maxtext/layers/mhc.py or src/maxtext/configs/types.py.
  2. Unweighted FP32 RMSNorm (Glm5UnweightedRMSNorm, matching HF
     Glm5NextRMSNorm(weight=False)) vs. learnable scale parameter in legacy
     mhc_norm (mhc.py:132).
  3. Pure FP32 dynamic projection flat @ fn (matching HF
     torch.matmul(x_norm, self.phi)) vs. legacy
     jnp.asarray(x_norm, self.config.dtype) cast before projection (mhc.py:133).
  4. Unweighted stream mean jnp.mean(hidden_states_mhc, axis=2) in Glm5HyperHead
     vs. learned sigmoid gate in DeepSeek4HyperHead (mhc.py:215-225).
  """

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided to Glm5HyperConnection.")
    self.config, self.hc_mult = config, config.mhc_expansion_rate
    self.hc_sinkhorn_iters = config.mhc_sinkhorn_iters
    self.input_norm = Glm5UnweightedRMSNorm(eps=config.rms_norm_eps)

    p_dt = getattr(jnp, config.weight_dtype)
    mix = (2 + self.hc_mult) * self.hc_mult
    fn_init = nnx.initializers.normal(0.02)(rngs.params(), (self.hc_mult * config.hidden_size, mix), p_dt)
    self.fn = nnx.Param(
        fn_init,
        sharding=sharding("mhc_fn", ("embed", None)) if sharding else None,
    )
    self.base = nnx.Param(
        jnp.zeros((mix,), dtype=p_dt),
        sharding=sharding("mhc_base", (None,)) if sharding else None,
    )
    self.scale = nnx.Param(
        jnp.ones((3,), dtype=p_dt),
        sharding=sharding("mhc_scale", (None,)) if sharding else None,
    )

  def __call__(self, hidden_streams: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    hc, eps = self.hc_mult, self.config.mhc_eps
    b, s = hidden_streams.shape[:2]
    flat = self.input_norm(hidden_streams.reshape(b, s, -1).astype(jnp.float32))
    fn_w = self.fn[...].astype(jnp.float32)
    proj = jnp.matmul(flat, fn_w, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)
    pre_w, post_w, comb_w = jnp.split(proj, [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = jnp.split(self.base[...].astype(jnp.float32), [hc, 2 * hc], axis=-1)
    pre_s, post_s, comb_s = self.scale[...].astype(jnp.float32)

    pre = jax.nn.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * jax.nn.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.reshape(b, s, hc, hc) * comb_s + comb_b.reshape(hc, hc)
    comb = jax.nn.softmax(comb_logits, axis=-1) + eps
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
      for ax in (-1, -2):
        comb = comb / (jnp.sum(comb, axis=ax, keepdims=True) + eps)

    col = jnp.sum(pre[..., None] * hidden_streams.astype(jnp.float32), axis=2).astype(hidden_streams.dtype)
    return post, comb, col

  def apply_residual(
      self,
      residual: jax.Array,
      sublayer_out: jax.Array,
      post: jax.Array,
      comb: jax.Array,
  ) -> jax.Array:
    dt = residual.dtype
    branch1 = post.astype(dt)[..., None] * sublayer_out[:, :, None, :]
    return branch1 + jnp.matmul(
        jnp.swapaxes(comb.astype(dt), -1, -2),
        residual,
        precision=jax.lax.Precision.HIGHEST,
    )


class Glm5HyperHead(nnx.Module):
  """Final GLM-5.3-Flash HC-stream collapse via unweighted mean across streams."""

  def __call__(self, hidden_streams: jax.Array) -> jax.Array:
    return jnp.mean(hidden_streams, axis=2)


class Glm5DecoderLayer(nnx.Module):
  """GLM-5.3-Flash Transformer block with 4-stream mHC and heterogeneous attn/mlp."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      layer_idx: int,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if not 0 <= layer_idx < config.num_hidden_layers:
      raise ValueError(f"layer_idx {layer_idx} out of range [0, {config.num_hidden_layers}).")
    self.config, self.layer_idx = config, layer_idx
    if config.layer_types is None or config.indexer_types is None:
      raise ValueError("config.layer_types and config.indexer_types must not be None.")
    block_type = config.layer_types[layer_idx]
    self.block_type = block_type
    self.layer_type = block_type
    p_dt = getattr(jnp, config.weight_dtype)

    if block_type == "linear_attention":
      self.self_attn = Glm5KdaAttention(config, sharding=sharding, rngs=rngs)
    elif block_type == "full_attention":
      self.self_attn = Glm5DsaMlaAttention(
          config,
          layer_idx,
          indexer_type=config.indexer_types[layer_idx],
          sharding=sharding,
          rngs=rngs,
      )
    else:
      raise ValueError(f"Unknown block_type: {block_type}")

    self.is_moe = layer_idx >= config.first_k_dense_replace
    self.mlp = (
        Glm5MoE(config, sharding=sharding, rngs=rngs)
        if self.is_moe
        else Glm5MLP(config, config.intermediate_size, sharding=sharding, rngs=rngs)
    )

    def _norm() -> Glm5RMSNorm:
      return Glm5RMSNorm(
          config.hidden_size,
          eps=config.rms_norm_eps,
          param_dtype=p_dt,
          sharding=sharding,
      )

    self.input_layernorm, self.post_attention_layernorm = _norm(), _norm()
    self.attn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)
    self.ffn_hc = Glm5HyperConnection(config, sharding=sharding, rngs=rngs)

  def __call__(
      self,
      hidden_streams: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array | None]:
    p, c, col = self.attn_hc(hidden_streams)
    if isinstance(self.self_attn, Glm5DsaMlaAttention):
      a_out = self.self_attn(
          self.input_layernorm(col),
          decoder_segment_ids=decoder_segment_ids,
          return_topk_indices=self.config.enable_debug_stats,
      )
    else:
      a_out = self.self_attn(self.input_layernorm(col), decoder_segment_ids=decoder_segment_ids)
    indexer_topk_indices = None
    if isinstance(a_out, tuple):
      a_out, indexer_topk_indices = a_out
    hidden_streams = self.attn_hc.apply_residual(hidden_streams, a_out, p, c)

    p, c, col = self.ffn_hc(hidden_streams)
    normed_mlp_in = self.post_attention_layernorm(col)
    if isinstance(self.mlp, Glm5MoE):
      m_out, (router_logits, router_topk_indices) = self.mlp(normed_mlp_in)
    else:
      m_out = self.mlp(normed_mlp_in)
      router_logits = None
      router_topk_indices = None
    return (
        self.ffn_hc.apply_residual(hidden_streams, m_out, p, c),
        router_logits,
        router_topk_indices,
        indexer_topk_indices,
    )


# ============================================================================
# Section 5: Super-Block Scan & Remat
# ============================================================================


def layer_to_block_slot(
    layer_idx: int,
    first_k_dense_replace: int,
    num_hidden_layers: int,
) -> tuple[str, int, int | None]:
  """Maps flat layer index to (section, block_or_layer_idx, slot_idx)."""
  if num_hidden_layers <= 0:
    raise ValueError(f"num_hidden_layers must be positive: {num_hidden_layers}")
  if layer_idx < 0 or layer_idx >= num_hidden_layers:
    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_hidden_layers})")
  if first_k_dense_replace < 0 or first_k_dense_replace > num_hidden_layers:
    raise ValueError(f"first_k_dense_replace {first_k_dense_replace} invalid for" f" {num_hidden_layers} layers")
  if layer_idx < first_k_dense_replace:
    return ("prefix", layer_idx, None)
  num_super_blocks = (num_hidden_layers - first_k_dense_replace) // 4
  super_block_layers = 4 * num_super_blocks
  rel = layer_idx - first_k_dense_replace
  if rel < super_block_layers:
    return ("super_block", rel // 4, rel % 4)
  return ("suffix", rel - super_block_layers, None)


class Glm5SuperBlock(nnx.Module):
  """4-layer super-block: (MLA/MoE, KDA/MoE, KDA/MoE, KDA/MoE)."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    self.config = config
    fk = config.first_k_dense_replace
    # Slots 0..3 correspond to layers fk..fk+3 (1 MLA + 3 KDA, all with MoE).
    self.slot_0 = Glm5DecoderLayer(config, fk + 0, rngs=rngs, sharding=sharding)
    self.slot_1 = Glm5DecoderLayer(config, fk + 1, rngs=rngs, sharding=sharding)
    self.slot_2 = Glm5DecoderLayer(config, fk + 2, rngs=rngs, sharding=sharding)
    self.slot_3 = Glm5DecoderLayer(config, fk + 3, rngs=rngs, sharding=sharding)

  def __call__(
      self,
      hidden_streams: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      remat_policy: Any = None,
      cur_lb: tuple[jax.Array, jax.Array, jax.Array] | None = None,
      return_router_logits: bool = True,
  ) -> tuple[
      jax.Array,
      jax.Array | None,
      jax.Array | None,
      jax.Array | None,
      tuple[jax.Array, jax.Array, jax.Array] | None,
  ]:
    num_experts = self.config.n_routed_experts
    top_k = self.config.num_experts_per_tok
    attn_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None

    def _call_slot(
        slot: Glm5DecoderLayer,
        streams: jax.Array,
        lb: tuple[jax.Array, jax.Array, jax.Array] | None,
    ):
      streams, r, rt, idx = slot(streams, decoder_segment_ids=decoder_segment_ids)
      if r is None or rt is None:
        raise ValueError("Super-block slots must all be MoE with non-None router logits and" " topk.")
      if lb is not None:
        lb = accumulate_moe_lb_stats(
            lb,
            r,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attn_mask,
        )
      if return_router_logits:
        return streams, lb, r, rt, idx
      return streams, lb, idx

    slots = (self.slot_0, self.slot_1, self.slot_2, self.slot_3)
    router_logits = []
    router_topks = []
    indexer_topks = []
    for slot in slots:
      if remat_policy is not None:
        if not callable(remat_policy):
          raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
        exec_slot = jax.checkpoint(
            lambda s, lb, sl=slot: _call_slot(sl, s, lb),
            policy=remat_policy,
            prevent_cse=True,
        )
      else:

        def _exec_slot_direct(s: jax.Array, lb: Any, sl: Any = slot) -> Any:
          return _call_slot(sl, s, lb)

        exec_slot = _exec_slot_direct

      if return_router_logits:
        hidden_streams, cur_lb, r, rt, idx = exec_slot(hidden_streams, cur_lb)
        router_logits.append(r)
        router_topks.append(rt)
      else:
        hidden_streams, cur_lb, idx = exec_slot(hidden_streams, cur_lb)
      indexer_topks.append(idx)
    mla_idx = next(
        (idx for slot, idx in zip(slots, indexer_topks) if slot.layer_type == "full_attention"),
        None,
    )
    r_stack = jnp.stack(router_logits, axis=0) if return_router_logits else None
    rt_stack = jnp.stack(router_topks, axis=0) if return_router_logits else None
    return (
        hidden_streams,
        r_stack,
        rt_stack,
        mla_idx,
        cur_lb,
    )


def create_stacked_super_block(
    config: glm5_config.Glm5Config,
    num_super_blocks: int,
    *,
    rngs: nnx.Rngs,
    sharding: m3_sharding.Sharding | None = None,
    place_on_devices: bool,
) -> Glm5SuperBlock:
  """Constructs a stacked Glm5SuperBlock with leading dimension num_super_blocks."""
  if num_super_blocks <= 0:
    raise ValueError(f"num_super_blocks must be positive, got {num_super_blocks}")

  fk, lt, it = (
      config.first_k_dense_replace,
      config.layer_types,
      config.indexer_types,
  )
  if lt is None or it is None:
    raise ValueError(f"layer_types and indexer_types must not be None, got {lt=}, {it=}")
  if fk + 4 * num_super_blocks > config.num_hidden_layers:
    raise ValueError(
        f"Required {fk + 4 * num_super_blocks} layers for {num_super_blocks}"
        f" super blocks with first_k_dense_replace={fk} exceeds"
        f" num_hidden_layers={config.num_hidden_layers}."
    )
  for b in range(num_super_blocks):
    for s in range(4):
      idx = fk + 4 * b + s
      if lt[idx] != lt[fk + s] or it[idx] != it[fk + s]:
        raise ValueError(f"Non-periodic schedule at layer {idx}: {lt[idx]!r} / {it[idx]!r}.")

  def _init_single(r: nnx.Rngs) -> Glm5SuperBlock:
    return Glm5SuperBlock(config, rngs=r, sharding=sharding)

  forked_rngs = rngs.split(num_super_blocks)
  stacked_block = nnx.vmap(_init_single, in_axes=0, out_axes=0)(forked_rngs)
  graphdef, state = nnx.split(stacked_block)

  def _update_var(path, var):
    if isinstance(var, nnx.Variable):
      orig_sh = var.get_metadata("out_sharding", None)
      if orig_sh is None:
        if sharding is not None:
          raise ValueError(f"Expected sharding metadata on stacked variable at {path}, got" " None.")
        if place_on_devices:
          raise ValueError(
              "place_on_devices=True requires a non-empty NamedSharding mesh" f" on stacked parameters at {path}."
          )
        return var
      orig_spec = orig_sh.spec if isinstance(orig_sh, jax.sharding.NamedSharding) else orig_sh
      if not isinstance(orig_spec, jax.sharding.PartitionSpec):
        raise ValueError(f"Unsupported sharding metadata type at {path}: {type(orig_sh)}")
      new_spec = jax.sharding.PartitionSpec(None, *orig_spec)
      raw_val = var.get_raw_value()
      if place_on_devices:
        raw_sh = raw_val.sharding if isinstance(raw_val, jax.Array) else None
        mesh = (
            orig_sh.mesh
            if isinstance(orig_sh, jax.sharding.NamedSharding)
            else (raw_sh.mesh if isinstance(raw_sh, jax.sharding.NamedSharding) else None)
        )
        if mesh is None or mesh.empty:
          raise ValueError(
              "place_on_devices=True requires a non-empty NamedSharding mesh" f" on stacked parameters at {path}."
          )
        new_sh = jax.sharding.NamedSharding(mesh, new_spec)
        placed_val = jax.device_put(raw_val, new_sh)
        return var.replace(placed_val, out_sharding=new_spec)
      return var.replace(raw_val, out_sharding=new_spec)
    return var

  updated_state = jax.tree_util.tree_map_with_path(_update_var, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return nnx.merge(graphdef, updated_state)


def _slice_state(state: Any, i: int) -> Any:
  """Slices stacked super-block state at index i and strips the leading PartitionSpec axis."""

  def _slice_leaf(v: Any) -> Any:
    if isinstance(v, nnx.Variable):
      raw_val = v.get_raw_value()[i]
      orig_sh = (
          v.get_metadata("out_sharding", None)
          if hasattr(v, "get_metadata")
          else (v.out_sharding if hasattr(v, "out_sharding") else None)
      )
      if isinstance(orig_sh, jax.sharding.PartitionSpec) and len(orig_sh) > 0:
        return v.replace(raw_val, out_sharding=jax.sharding.PartitionSpec(*orig_sh[1:]))
      return v.replace(raw_val)
    return v[i]

  return jax.tree.map(_slice_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))


def scan_super_blocks(
    super_block: Glm5SuperBlock,
    hidden_streams: jax.Array,
    *,
    scan_layers: bool,
    decoder_segment_ids: jax.Array | None = None,
    remat_policy: Any = None,
    return_router_logits: bool,
    lb_stats: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> tuple[
    jax.Array,
    list[jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
    list[jax.Array],
    Any,
]:
  """Runs stacked Glm5SuperBlock via jax.lax.scan or unrolled loop."""
  if super_block.slot_0.attn_hc.base.ndim != 2:
    raise ValueError(
        "super_block is not stacked: scan_super_blocks requires a stacked"
        " Glm5SuperBlock with a leading num_super_blocks axis, got"
        f" slot_0.attn_hc.base.ndim={super_block.slot_0.attn_hc.base.ndim}"
    )

  cfg = super_block.config
  num_experts = cfg.n_routed_experts
  cur_lb_init = lb_stats if lb_stats is not None else init_moe_lb_stats(num_experts)

  graphdef, params, non_params = nnx.split(super_block, nnx.Param, ...)
  leaves = jax.tree.leaves(params)
  if not leaves:
    raise ValueError("super_block has no parameters.")
  num_super_blocks = leaves[0].shape[0]

  def step_fn(carry_in: Any, xs_i: tuple[Any, Any]) -> tuple[Any, Any]:
    p_i, np_i = xs_i
    carry, cur_lb = carry_in
    block = nnx.merge(graphdef, p_i, np_i)
    carry, block_router_logits, block_router_topk, block_idx, cur_lb = block(
        carry,
        decoder_segment_ids=decoder_segment_ids,
        remat_policy=remat_policy,
        cur_lb=cur_lb,
        return_router_logits=return_router_logits,
    )
    r_out = block_router_logits if return_router_logits else None
    rt_out = block_router_topk if return_router_logits else None
    carry_out = (carry, cur_lb)
    return carry_out, (r_out, rt_out, block_idx)

  if remat_policy is not None and not callable(remat_policy):
    raise ValueError(f"remat_policy must be None or callable: {remat_policy!r}.")
  step_exec = step_fn

  init_carry = (hidden_streams, cur_lb_init)
  if scan_layers:
    (final_streams, final_lb), (scanned_r, scanned_rt, scanned_idx) = jax.lax.scan(
        step_exec, init_carry, (params, non_params)
    )
    all_router_logits = (
        [scanned_r[b, s] for b in range(num_super_blocks) for s in range(4)]
        if return_router_logits and scanned_r is not None
        else []
    )
    all_router_topk = (
        [scanned_rt[b, s] for b in range(num_super_blocks) for s in range(4)]
        if return_router_logits and scanned_rt is not None
        else []
    )
    all_idx = [scanned_idx[b] for b in range(num_super_blocks)] if scanned_idx is not None else None
  else:
    carry_state = init_carry
    all_router_logits = []
    all_router_topk = []
    all_idx_list = []
    for i in range(num_super_blocks):
      p_i = _slice_state(params, i)
      np_i = _slice_state(non_params, i)
      carry_state, (r_step, rt_step, idx_step) = step_exec(carry_state, (p_i, np_i))
      if return_router_logits and r_step is not None:
        for s in range(4):
          all_router_logits.append(r_step[s])
          all_router_topk.append(rt_step[s])
      all_idx_list.append(idx_step)
    final_streams, final_lb = carry_state
    all_idx = all_idx_list if any(x is not None for x in all_idx_list) else None

  return final_streams, all_router_logits, final_lb, all_router_topk, all_idx


def apply_glm5_remat(model: Any, *, policy: Any) -> Any:
  """Applies remat policy to model without altering parameter paths."""
  if policy is not None and not callable(policy):
    raise ValueError(f"policy must be None or callable: {policy!r}.")
  model.remat_policy = policy
  return model


# ============================================================================
# Section 6: Multi-Token Prediction Block
# ============================================================================


class Glm5Mtp(nnx.Module):
  """Layer-45 Multi-Token Prediction (MTP) block for GLM-5.3-Flash.

  Matches SGLang Glm5NextForConditionalGenerationNextN and DeepSeek-V3
  NextN specifications, consuming post-final-norm backbone hidden
  states out.hidden_states (norm(hc_head(hidden_streams))).
  """

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      sharding: m3_sharding.Sharding | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sharding = sharding
    h = config.hidden_size
    eps = config.rms_norm_eps
    dt = getattr(jnp, config.dtype)
    p_dt = getattr(jnp, config.weight_dtype)

    self.enorm = Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.hnorm = Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    eh_meta = {"sharding": sharding("eh_proj", ("embed", None))} if sharding else {}
    self.eh_proj = nnx.Linear(
        2 * h,
        h,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=eh_meta,
        rngs=rngs,
    )
    self.input_layernorm = Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.self_attn = Glm5DsaMlaAttention(
        config,
        layer_idx=glm5_config.MTP_LAYER_INDEX,
        indexer_type=config.mtp_indexer_type,
        sharding=sharding,
        rngs=rngs,
    )
    self.post_attention_layernorm = Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)
    self.mlp = Glm5MoE(config, sharding=sharding, rngs=rngs)
    self.shared_head_norm = Glm5RMSNorm(h, eps=eps, param_dtype=p_dt, sharding=sharding)

  def __call__(
      self,
      hidden_states: jax.Array,
      next_token_ids: jax.Array,
      embed_tokens: nnx.Embed,
      lm_head: nnx.Linear | None,
      decoder_segment_ids: jax.Array | None,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes MTP logits and router logits for next-token target prediction.

    Args:
      hidden_states: Post-HyperHead mean collapse and post-final-norm hidden
        states out.hidden_states [B, S, H], matching Megatron and DeepSeek-V3
        NextN specifications.
      next_token_ids: Input tokens for the MTP step [B, S] (i.e. targets).
      embed_tokens: Shared token embedding module.
      lm_head: Shared linear projection to vocabulary logits, or None to return
        shared_head_norm(z) [B, S, H] for vocab tiling without materializing
        full logits.
      decoder_segment_ids: Optional sequence segmentation IDs [B, S].

    Returns:
      (mtp_out, router_logits, router_topk) tuple where mtp_out is mtp_logits
      of shape [B, S, V] if lm_head is provided, or normalized hidden states
      of shape [B, S, H] if lm_head is None. router_logits has shape
      [B * S, num_experts], and router_topk has shape [B, S, top_k].
    """
    emb = embed_tokens(next_token_ids)
    x = self.eh_proj(jnp.concatenate([self.enorm(emb), self.hnorm(hidden_states)], axis=-1))
    attn_out = self.self_attn(self.input_layernorm(x), decoder_segment_ids=decoder_segment_ids)
    if isinstance(attn_out, tuple):
      attn_out = attn_out[0]
    y = x + attn_out
    mlp_out, (router_logits, router_topk) = self.mlp(self.post_attention_layernorm(y))
    z = y + mlp_out
    normed_z = self.shared_head_norm(z)
    if lm_head is None:
      return normed_z, router_logits, router_topk
    head_dt = lm_head.kernel.dtype
    mtp_logits = lm_head(normed_z.astype(head_dt))
    return mtp_logits, router_logits, router_topk


# ============================================================================
# Section 7: Backbone & Top-Level Model
# ============================================================================


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Glm5Output:
  """Output of GLM-5.3-Flash forward pass with auxiliary data."""

  logits: jax.Array | None
  hidden_states: jax.Array
  aux: dict[str, Any]


class Glm5Embedding(nnx.Embed):
  """Token embedding module for GLM-5.3-Flash."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
  ):
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    super().__init__(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        dtype=getattr(jnp, config.dtype),
        param_dtype=getattr(jnp, config.weight_dtype),
        embedding_metadata={"sharding": sharding("embed", ("vocab", "embed"))} if sharding else {},
        rngs=rngs,
    )


class Glm5Model(nnx.Module):
  """GLM-5.3-Flash autoregressive language model with 4-stream mHC and hybrid layers."""

  def __init__(
      self,
      config: glm5_config.Glm5Config,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      sharding: m3_sharding.Sharding | None = None,
      place_on_devices: bool,
  ):
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    self.config = glm5_config.Glm5Config.from_maxtext_config(config)
    self.mesh = mesh
    if sharding is None and mesh is not None:
      sharding = glm5_config.Glm5Sharding.from_mesh(mesh)
    self.sharding = sharding
    self.remat_policy: Any = None

    cfg = self.config
    if cfg.first_k_dense_replace < 0 or (cfg.first_k_dense_replace == 0 and cfg.num_hidden_layers > 1):
      raise ValueError("first_k_dense_replace must be >= 1 unless num_hidden_layers == 1.")
    dt, p_dt = getattr(jnp, cfg.dtype), getattr(jnp, cfg.weight_dtype)
    self.embed_tokens = Glm5Embedding(cfg, rngs=rngs, sharding=sharding)

    num_prefix = min(cfg.num_hidden_layers, cfg.first_k_dense_replace)
    self.prefix_layers = nnx.List([Glm5DecoderLayer(cfg, i, rngs=rngs, sharding=sharding) for i in range(num_prefix)])
    rem = cfg.num_hidden_layers - num_prefix
    num_super_blocks, num_suffix = rem // 4, rem % 4
    self.super_block = (
        create_stacked_super_block(
            cfg,
            num_super_blocks,
            rngs=rngs,
            sharding=sharding,
            place_on_devices=place_on_devices,
        )
        if num_super_blocks > 0
        else None
    )
    suffix_start = num_prefix + 4 * num_super_blocks
    self.suffix_layers = nnx.List(
        [Glm5DecoderLayer(cfg, suffix_start + i, rngs=rngs, sharding=sharding) for i in range(num_suffix)]
    )

    self.hc_head = Glm5HyperHead()
    self.norm = Glm5RMSNorm(
        cfg.hidden_size,
        eps=cfg.rms_norm_eps,
        param_dtype=p_dt,
        sharding=sharding,
    )
    lm_meta = {"sharding": sharding("lm_head", ("embed", "vocab"))} if sharding else {}
    self.lm_head = nnx.Linear(
        cfg.hidden_size,
        cfg.vocab_size,
        use_bias=False,
        dtype=dt,
        param_dtype=p_dt,
        precision=jax.lax.Precision.HIGHEST,
        kernel_metadata=lm_meta,
        rngs=rngs,
    )
    self.mtp = Glm5Mtp(cfg, sharding=sharding, rngs=rngs) if cfg.num_nextn_predict_layers > 0 else None

  def forward_with_aux(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      compute_logits: bool = True,
      **kwargs: Any,
  ) -> Glm5Output:
    """Executes full forward pass returning logits, hidden states, and auxiliary loss."""
    del decoder_positions, kwargs
    x = self.embed_tokens(decoder_input_tokens)
    hidden_streams = jnp.repeat(x[:, :, None, :], self.config.mhc_expansion_rate, axis=2)
    if self.sharding is not None:
      hidden_streams = self.sharding.constrain(hidden_streams, "hidden_streams", ("batch", "length", None, "embed"))
    cfg = self.config

    # Synchronize enable_debug_stats to child layers and super_block so
    # model.config is single source of truth at call time (T40).
    def _sync_cfg(m):
      if m.config.enable_debug_stats != cfg.enable_debug_stats:
        m.config = dataclasses.replace(m.config, enable_debug_stats=cfg.enable_debug_stats)

    for l in self.prefix_layers:
      _sync_cfg(l)
    if self.super_block is not None:
      _sync_cfg(self.super_block)
      for s in range(4):
        _sync_cfg(getattr(self.super_block, f"slot_{s}"))
    for l in self.suffix_layers:
      _sync_cfg(l)

    attention_mask = (decoder_segment_ids != 0) if decoder_segment_ids is not None else None
    keep_router_logits = bool(cfg.enable_debug_stats)
    router_logits_list: list[jax.Array] = []
    router_topk_list: list[jax.Array] = []
    lb_stats = init_moe_lb_stats(cfg.n_routed_experts)

    layer_0_act: jax.Array | None = None
    dsa_topk_indices_list: list[jax.Array] = []

    def _run_layers(layers, streams, cur_lb):
      nonlocal layer_0_act
      for layer in layers:
        if self.remat_policy is not None:
          fn = jax.checkpoint(
              lambda s, lyr=layer: lyr(s, decoder_segment_ids=decoder_segment_ids),
              policy=self.remat_policy,
              prevent_cse=True,
          )
          streams, r_logits, r_topk, idx_topk = fn(streams)
        else:
          streams, r_logits, r_topk, idx_topk = layer(streams, decoder_segment_ids=decoder_segment_ids)
        if idx_topk is not None:
          dsa_topk_indices_list.append(idx_topk)
        if layer_0_act is None:
          layer_0_act = streams
        if r_logits is not None:
          cur_lb = accumulate_moe_lb_stats(
              cur_lb,
              r_logits,
              num_experts=cfg.n_routed_experts,
              top_k=cfg.num_experts_per_tok,
              attention_mask=attention_mask,
          )
          if keep_router_logits:
            router_logits_list.append(r_logits)
            router_topk_list.append(r_topk)
      return streams, cur_lb

    if self.prefix_layers:
      hidden_streams, lb_stats = _run_layers(self.prefix_layers, hidden_streams, lb_stats)
    if self.super_block is not None:
      hidden_streams, sb_r, lb_stats, sb_rt, sb_idx = scan_super_blocks(
          self.super_block,
          hidden_streams,
          decoder_segment_ids=decoder_segment_ids,
          scan_layers=cfg.scan_layers,
          remat_policy=self.remat_policy,
          return_router_logits=keep_router_logits,
          lb_stats=lb_stats,
      )
      if sb_idx is not None:
        for idx_item in sb_idx:
          if idx_item is not None:
            dsa_topk_indices_list.append(idx_item)
      if keep_router_logits:
        router_logits_list.extend(sb_r)
        router_topk_list.extend(sb_rt)
    if self.suffix_layers:
      hidden_streams, lb_stats = _run_layers(self.suffix_layers, hidden_streams, lb_stats)
    last_block_act = hidden_streams
    collapsed = self.hc_head(hidden_streams)
    normed = self.norm(collapsed)
    logits = self.lm_head(normed) if compute_logits else None
    num_moe_layers = max(0, cfg.num_hidden_layers - cfg.first_k_dense_replace)
    if keep_router_logits and len(router_logits_list) != num_moe_layers:
      raise ValueError(f"Expected {num_moe_layers} MoE router_logits entries, got" f" {len(router_logits_list)}.")
    if num_moe_layers == 0:
      moe_lb_loss = jnp.zeros((), dtype=jnp.float32)
    else:
      moe_lb_loss = finalize_moe_lb_loss(lb_stats, cfg.n_routed_experts)

    router_topk_indices_list: list[jax.Array] = router_topk_list if cfg.enable_debug_stats else []

    aux: dict[str, Any] = {
        "router_logits": (tuple(router_logits_list) if cfg.enable_debug_stats else ()),
        "moe_lb_loss": moe_lb_loss,
        "lb_stats": lb_stats,
    }
    if cfg.enable_debug_stats:
      if layer_0_act is None:
        raise RuntimeError("enable_debug_stats=True requires layer 0 activation, but" " layer_0_act was not captured.")
      l0_f32 = layer_0_act.astype(jnp.float32)
      last_f32 = last_block_act.astype(jnp.float32)
      aux["debug_stats"] = {
          "first_block_mean": jnp.mean(l0_f32),
          "first_block_var": jnp.var(l0_f32),
          "last_block_mean": jnp.mean(last_f32),
          "last_block_var": jnp.var(last_f32),
          "hidden_norm_mean": jnp.mean(jnp.linalg.norm(normed, axis=-1)),
          "logits_max": (jnp.max(logits) if logits is not None else jnp.zeros((), dtype=normed.dtype)),
          "logits_min": (jnp.min(logits) if logits is not None else jnp.zeros((), dtype=normed.dtype)),
          "router_topk_indices": tuple(router_topk_indices_list),
          "dsa_topk_indices": tuple(dsa_topk_indices_list),
      }
      aux["router_topk_indices"] = tuple(router_topk_indices_list)
      aux["dsa_topk_indices"] = tuple(dsa_topk_indices_list)
    return Glm5Output(logits=logits, hidden_states=normed, aux=aux)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      **kwargs: Any,
  ) -> jax.Array:
    out = self.forward_with_aux(decoder_input_tokens, decoder_positions, decoder_segment_ids, **kwargs)
    if out.logits is None:
      raise RuntimeError("out.logits is None in Glm5Model.__call__.")
    return out.logits

  def logits_from_hidden_states_for_vocab_tiling(
      self,
      hidden_states: jax.Array,
      deterministic: bool = True,
      **kwargs: Any,
  ) -> jax.Array:
    """Projects post-final-norm hidden states to output logits for vocab tiling.

    Args:
      hidden_states: Post-final-norm hidden states, matching
        Glm5Output.hidden_states (after collapsing the 4 mHC streams via hc_head
        and applying self.norm).
      deterministic: Unused flag for API parity.
      **kwargs: Ignored keyword arguments for API parity.

    Returns:
      Output logits projected via self.lm_head.
    """
    del deterministic, kwargs
    return self.lm_head(hidden_states)

  @classmethod
  def from_maxtext_config(
      cls,
      config: Any,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
      quant_mode_str: str | None = None,
      place_on_devices: bool,
      **kwargs: Any,
  ) -> "Glm5Model":
    """Constructs a Glm5Model from MaxText configuration."""
    if not isinstance(place_on_devices, bool):
      raise TypeError("place_on_devices must be a bool, got" f" {type(place_on_devices).__name__}")
    if rngs is None:
      raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
    if not isinstance(config, glm5_config.Glm5Config) and config.quantization not in (None, ""):
      raise ValueError("Quantization is unsupported for GLM-5.3-Flash:" f" {config.quantization!r}.")
    if quant_mode_str not in (None, "", "train", "serve", "convert"):
      raise ValueError("Runtime quantization is unsupported for GLM-5.3-Flash:" f" {quant_mode_str!r}.")
    if kwargs:
      raise TypeError("Unexpected keyword arguments in from_maxtext_config:" f" {sorted(kwargs)!r}.")
    return cls(config=config, mesh=mesh, rngs=rngs, place_on_devices=place_on_devices)


def create_glm5_3_flash_model(
    config: Any,
    mesh: Mesh | None = None,
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> Glm5Model:
  """Creates a Glm5Model instance from MaxText config and mesh."""
  if "place_on_devices" in kwargs:
    raise TypeError(
        "place_on_devices is fixed to False in create_glm5_3_flash_model;"
        " pass place_on_devices to Glm5Model or Glm5Model.from_maxtext_config"
        " directly instead."
    )
  if "remat_policy" in kwargs:
    raise TypeError("remat_policy must be configured via config.remat_policy, not as a" " keyword argument.")
  if not hasattr(config, "remat_policy"):
    raise AttributeError("config must define 'remat_policy'.")
  remat_policy = config.remat_policy
  if not isinstance(remat_policy, str):
    raise TypeError(f"remat_policy must be a str, got {type(remat_policy).__name__}.")
  if remat_policy == "none":
    policy = None
  elif remat_policy == "full":
    policy = jax.checkpoint_policies.nothing_saveable
  else:
    raise ValueError(
        f"Unsupported remat_policy for GLM-5.3-Flash: {remat_policy!r}." " Supported policies: ('none', 'full')."
    )
  model = Glm5Model.from_maxtext_config(config, mesh, rngs=rngs, place_on_devices=False, **kwargs)
  return apply_glm5_remat(model, policy=policy)


# ============================================================================
# Section 8: Loss Functions & Optimizer Masks
# ============================================================================


def compute_glm5_cross_entropy_loss(
    logits: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes masked token cross-entropy loss strictly in float32.

  Args:
    logits: Model logits of shape [batch_size, seq_len, vocab_size].
    decoder_target_tokens: Next-token shifted target token ids of shape
      [batch_size, seq_len] (`tokens[:, 1:]` aligned with `tokens[:, :-1]`).
    decoder_target_mask: Next-token shifted target loss mask of shape
      [batch_size, seq_len] (`tokens[:, 1:]` aligned with `tokens[:, :-1]`).

  Returns:
    (ce_loss, xent_sum, total_weights) tuple strictly in float32.
  """
  logits_f32 = logits.astype(jnp.float32)
  targets_arr = jnp.asarray(decoder_target_tokens)
  mask_f32 = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32)

  token_losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_f32, labels=targets_arr)
  masked_losses = token_losses * mask_f32
  xent_sum = jnp.sum(masked_losses).astype(jnp.float32)
  total_weights = jnp.sum(mask_f32).astype(jnp.float32)
  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def _compute_tiled_cross_entropy(
    model: Glm5Model,
    hidden_states: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
    num_vocab_tiling: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes cross-entropy loss tiled across batch-sequence tokens.

  Args:
    model: The Glm5Model instance exposing
      `logits_from_hidden_states_for_vocab_tiling`.
    hidden_states: Post-final-norm hidden states of shape [batch_size, seq_len,
      hidden_size].
    decoder_target_tokens: Target tokens of shape [batch_size, seq_len].
    decoder_target_mask: Target mask of shape [batch_size, seq_len].
    num_vocab_tiling: Number of tiles along the batch*seq_len token dimension.

  Returns:
    (ce_loss, xent_sum, total_weights) tuple strictly in float32.
  """
  if num_vocab_tiling <= 1:
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states)
    return compute_glm5_cross_entropy_loss(logits, decoder_target_tokens, decoder_target_mask)

  b, s, emb_dim = hidden_states.shape
  total_tokens = b * s
  if total_tokens % num_vocab_tiling != 0:
    raise ValueError(
        f"Batch tokens ({b} * {s} = {total_tokens}) must be divisible by" f" num_vocab_tiling={num_vocab_tiling}."
    )
  tile_size = total_tokens // num_vocab_tiling
  hidden_tiles = hidden_states.reshape((num_vocab_tiling, tile_size, emb_dim))
  targets_tiles = jnp.asarray(decoder_target_tokens).reshape((num_vocab_tiling, tile_size))
  mask_tiles = (jnp.asarray(decoder_target_mask) != 0).astype(jnp.float32).reshape((num_vocab_tiling, tile_size))

  total_weights = jnp.sum(mask_tiles).astype(jnp.float32)

  @jax.checkpoint
  def _tile_xent_sum(
      h_tile: jax.Array,
      t_tile: jax.Array,
      m_tile: jax.Array,
  ) -> jax.Array:
    chunk_logits = model.logits_from_hidden_states_for_vocab_tiling(h_tile).astype(jnp.float32)
    chunk_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=chunk_logits,
        labels=t_tile,
    )
    return jnp.sum(chunk_loss * m_tile).astype(jnp.float32)

  def _scan_step(
      acc: jax.Array,
      tile_inputs: tuple[jax.Array, jax.Array, jax.Array],
  ) -> tuple[jax.Array, None]:
    h_tile, t_tile, m_tile = tile_inputs
    tile_sum = _tile_xent_sum(h_tile, t_tile, m_tile)
    return acc + tile_sum, None

  xent_sum, _ = jax.lax.scan(
      _scan_step,
      jnp.zeros((), dtype=jnp.float32),
      (hidden_tiles, targets_tiles, mask_tiles),
  )

  ce_loss = (xent_sum / jnp.maximum(total_weights, 1.0)).astype(jnp.float32)
  return ce_loss, xent_sum, total_weights


def compute_glm5_loss(
    model: Glm5Model,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
    *,
    is_train: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes cross-entropy and load-balancing loss for GLM-5.3-Flash.

  Layer-45 MTP consumes post-final-norm backbone hidden states
  out.hidden_states (norm(hc_head(hidden_streams))), matching
  Megatron and DeepSeek-V3 NextN specifications.

  Contract:
    `decoder_target_tokens` and `decoder_target_mask` are next-token shifted
    targets (`tokens[:, 1:]` aligned with `tokens[:, :-1]`), matching
    MaxText's `loss_fn` convention.
    In CL7, `is_train` indicates training mode (selecting MTP inclusion added
    in CL8); `total_loss = ce_loss + moe_lb_loss_weight * moe_lb_loss` in both
    modes.

  Args:
    model: The Glm5Model instance being trained.
    tokens: Input token ids of shape [batch_size, seq_len].
    decoder_target_tokens: Next-token shifted targets [batch_size, seq_len].
    decoder_target_mask: Next-token shifted loss mask [batch_size, seq_len].
    decoder_segment_ids: Optional input segmentation [batch_size, seq_len].
    decoder_positions: Optional input positions [batch_size, seq_len].
    is_train: Required boolean flag gating training-specific loss components
      (including MTP).

  Returns:
    (total_loss, aux_metrics) tuple strictly in float32.
  """
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train).__name__}.")
  if tokens is None:
    raise ValueError("tokens must be provided as a non-None array.")
  if decoder_target_tokens is None:
    raise ValueError("decoder_target_tokens must be provided as a non-None array.")
  if decoder_target_mask is None:
    raise ValueError("decoder_target_mask must be provided as a non-None array.")

  tokens_arr = jnp.asarray(tokens)
  target_tokens_arr = jnp.asarray(decoder_target_tokens)
  target_mask_arr = jnp.asarray(decoder_target_mask)

  if target_tokens_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_tokens shape {target_tokens_arr.shape} != tokens shape" f" {tokens_arr.shape}.")
  if target_mask_arr.shape != tokens_arr.shape:
    raise ValueError(f"decoder_target_mask shape {target_mask_arr.shape} != tokens shape" f" {tokens_arr.shape}.")

  seg_ids = jnp.asarray(decoder_segment_ids) if decoder_segment_ids is not None else None
  positions = jnp.asarray(decoder_positions) if decoder_positions is not None else None

  num_vocab_tiling = model.config.num_vocab_tiling
  if num_vocab_tiling > 1:
    out = model.forward_with_aux(
        tokens_arr,
        decoder_positions=positions,
        decoder_segment_ids=seg_ids,
        compute_logits=False,
    )
    ce_loss, xent_sum, total_weights = _compute_tiled_cross_entropy(
        model=model,
        hidden_states=out.hidden_states,
        decoder_target_tokens=target_tokens_arr,
        decoder_target_mask=target_mask_arr,
        num_vocab_tiling=num_vocab_tiling,
    )
  else:
    out = model.forward_with_aux(
        tokens_arr,
        decoder_positions=positions,
        decoder_segment_ids=seg_ids,
        compute_logits=True,
    )
    if out.logits is None:
      raise RuntimeError("out.logits is None but compute_logits=True was requested.")
    ce_loss, xent_sum, total_weights = compute_glm5_cross_entropy_loss(out.logits, target_tokens_arr, target_mask_arr)

  moe_lb_loss = out.aux["moe_lb_loss"].astype(jnp.float32)
  mtp_loss = jnp.zeros((), dtype=jnp.float32)
  mtp_moe_lb_loss = jnp.zeros((), dtype=jnp.float32)

  if is_train and model.config.num_nextn_predict_layers > 0:
    if model.mtp is None:
      raise ValueError("model.mtp is None but model.config.num_nextn_predict_layers > 0")
    # MTP consumes post-final-norm backbone hidden states out.hidden_states
    # (norm(hc_head(hidden_streams))), matching Megatron/DeepSeek-V3 NextN spec.
    mtp_targets = jnp.roll(target_tokens_arr, -1, axis=-1)
    valid_mask = target_mask_arr != 0
    mtp_valid_mask = valid_mask & jnp.concatenate(
        [
            valid_mask[:, 1:],
            jnp.zeros((valid_mask.shape[0], 1), dtype=jnp.bool_),
        ],
        axis=-1,
    )
    if num_vocab_tiling > 1:
      mtp_normed, mtp_router_logits, _ = model.mtp(
          out.hidden_states,
          target_tokens_arr,
          embed_tokens=model.embed_tokens,
          lm_head=None,
          decoder_segment_ids=seg_ids,
      )
      mtp_loss, _, _ = _compute_tiled_cross_entropy(
          model=model,
          hidden_states=mtp_normed,
          decoder_target_tokens=mtp_targets,
          decoder_target_mask=mtp_valid_mask,
          num_vocab_tiling=num_vocab_tiling,
      )
    else:
      mtp_logits, mtp_router_logits, _ = model.mtp(
          out.hidden_states,
          target_tokens_arr,
          embed_tokens=model.embed_tokens,
          lm_head=model.lm_head,
          decoder_segment_ids=seg_ids,
      )
      mtp_loss, _, _ = compute_glm5_cross_entropy_loss(mtp_logits, mtp_targets, mtp_valid_mask)

    mtp_lb_stats = accumulate_moe_lb_stats(
        init_moe_lb_stats(model.config.n_routed_experts),
        mtp_router_logits,
        model.config.n_routed_experts,
        model.config.num_experts_per_tok,
        attention_mask=(seg_ids != 0) if seg_ids is not None else None,
    )
    mtp_moe_lb_loss = finalize_moe_lb_loss(mtp_lb_stats, model.config.n_routed_experts).astype(jnp.float32)

    # Pool MTP MoE router stats with backbone MoE router stats for equal balance pressure
    bb_stats = out.aux["lb_stats"]
    pooled_lb_stats = (
        bb_stats[0] + mtp_lb_stats[0],
        bb_stats[1] + mtp_lb_stats[1],
        bb_stats[2] + mtp_lb_stats[2],
    )
    moe_lb_loss = finalize_moe_lb_loss(pooled_lb_stats, model.config.n_routed_experts).astype(jnp.float32)

    total_loss = (
        ce_loss
        + jnp.float32(model.config.moe_lb_loss_weight) * moe_lb_loss
        + jnp.float32(model.config.mtp_loss_scaling_factor) * mtp_loss
    ).astype(jnp.float32)
  else:
    total_loss = (ce_loss + jnp.float32(model.config.moe_lb_loss_weight) * moe_lb_loss).astype(jnp.float32)

  aux_metrics = {
      "ce_loss": ce_loss,
      "moe_lb_loss": moe_lb_loss,
      "mtp_loss": mtp_loss,
      "mtp_moe_lb_loss": mtp_moe_lb_loss,
      "xent_sum": xent_sum,
      "total_weights": total_weights,
  }
  return total_loss, aux_metrics


def glm5_loss_fn(
    model: Glm5Model,
    *,
    inputs: jax.Array,
    targets: jax.Array,
    targets_segmentation: jax.Array,
    is_train: bool,
    inputs_segmentation: jax.Array | None = None,
    inputs_position: jax.Array | None = None,
    targets_position: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Computes masked token cross-entropy loss and folds MoE load-balancing loss.

  Contract:
    `targets` and `targets_segmentation` are next-token shifted targets
    (`inputs[:, 1:]` aligned with `inputs[:, :-1]`), matching MaxText's
    `loss_fn` convention.
    In CL7, `is_train` indicates training mode (selecting MTP inclusion added
    in CL8); `total_loss = ce_loss + moe_lb_loss_weight * moe_lb_loss` in both
    modes.

  Args:
    model: The Glm5Model instance being trained.
    inputs: Input token ids of shape [batch_size, seq_len].
    targets: Target token ids of shape [batch_size, seq_len].
    targets_segmentation: Required segmentation mask [batch_size, seq_len].
    is_train: Required boolean flag gating training-specific loss components.
    inputs_segmentation: Optional input segmentation [batch_size, seq_len].
    inputs_position: Optional input positions [batch_size, seq_len].
    targets_position: Optional target positions [batch_size, seq_len].

  Returns:
    (total_loss, aux_metrics) tuple.
  """
  if not isinstance(is_train, bool):
    raise TypeError(f"is_train must be a bool, got {type(is_train).__name__}.")
  del targets_position
  return compute_glm5_loss(
      model=model,
      tokens=inputs,
      decoder_target_tokens=targets,
      decoder_target_mask=targets_segmentation,
      decoder_segment_ids=inputs_segmentation,
      decoder_positions=inputs_position,
      is_train=is_train,
  )


def trainable_mask(model_or_params: Any) -> Any:
  """Returns boolean PyTree (False on indexer.* and e_score_correction_bias)."""
  tree = (
      nnx.state(model_or_params, (nnx.Param, Glm5RouterBias))
      if isinstance(model_or_params, nnx.Module)
      else model_or_params
  )

  def _is_trainable(path: tuple[Any, ...], leaf: Any) -> bool:
    del leaf
    path_str = jax.tree_util.keystr(path)
    if "indexer" in path_str:
      return False
    if "e_score_correction_bias" in path_str:
      return False
    return True

  return jax.tree_util.tree_map_with_path(
      _is_trainable,
      tree,
      is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.Array)),
  )
