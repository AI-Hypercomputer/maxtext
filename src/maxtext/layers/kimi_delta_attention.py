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

"""Kimi Delta Attention (KDA) linear attention, as used by GLM-5.3-Flash.

KDA is a gated delta rule with a *per-channel* forget gate: the recurrent state
S_t in R^{K x V} (per head) evolves as

  S_t = diag(exp(g_t)) S_{t-1}
  S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T
  o_t = S_t^T q_t

where g_t in R^K (one decay per key channel, unlike GatedDeltaNet's per-head scalar),
beta_t is a per-head scalar, and q, k are L2-normalized. The sequence is processed in
chunks (WY representation inside a chunk, recurrence across chunks).

Reference: `Glm5NextTextLinearAttention` / `chunk_kimi_delta_attention` in
transformers/models/glm5_next/modeling_glm5_next.py.
"""

from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Array, Config, DType, MODEL_MODE_TRAIN, get_weight_dtype
from maxtext.layers import quantizations
from maxtext.layers.initializers import nd_dense_init
from maxtext.layers.linears import DenseGeneral

_HIGHEST = jax.lax.Precision.HIGHEST


def l2norm(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
  """FLA-style L2 normalization: x / sqrt(sum(x^2) + eps)."""
  return x * jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def chunk_kda(
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    chunk_size: int = 64,
    initial_state: Optional[Array] = None,
    use_qk_l2norm: bool = True,
) -> tuple[Array, Array]:
  """Chunked Kimi Delta Attention.

  Args:
    query: [B, S, H, K].
    key: [B, S, H, K].
    value: [B, S, H, V].
    g: [B, S, H, K] per-channel log-decay (<= 0).
    beta: [B, S, H] write strength in (0, 1).
    chunk_size: chunk length C.
    initial_state: optional [B, H, K, V] recurrent state.
    use_qk_l2norm: L2-normalize q and k (in float32) before the recurrence.

  Returns:
    (output [B, S, H, V] in query.dtype, final state [B, H, K, V] in float32).
  """
  out_dtype = query.dtype
  q, k, v, g, beta = (x.astype(jnp.float32) for x in (query, key, value, g, beta))
  if use_qk_l2norm:
    q = l2norm(q)
    k = l2norm(k)

  b, s, h, dk = k.shape
  dv = v.shape[-1]
  q = q * (dk**-0.5)

  pad = (chunk_size - s % chunk_size) % chunk_size
  if pad:
    pad_fn = lambda x: jnp.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2))
    q, k, v, g, beta = (pad_fn(x) for x in (q, k, v, g, beta))
  n = q.shape[1] // chunk_size

  # [B, S, H, D] -> [B, H, N, C, D]
  to_chunks = lambda x: x.reshape(b, n, chunk_size, h, -1).transpose(0, 3, 1, 2, 4)
  q, k, v, g = (to_chunks(x) for x in (q, k, v, g))
  beta = beta.reshape(b, n, chunk_size, h).transpose(0, 3, 1, 2)  # [B, H, N, C]

  g = jnp.cumsum(g, axis=-2)  # per-channel cumulative log-decay within each chunk
  k_beta = k * beta[..., None]
  v_beta = v * beta[..., None]

  # Pairwise per-channel decay exp(g_i - g_j) for j <= i; masked *before* exp to avoid overflow.
  causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
  strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
  g_diff = g[..., :, None, :] - g[..., None, :, :]  # [B, H, N, C, C, K]
  decay = jnp.exp(jnp.where(causal[..., None], g_diff, -jnp.inf))

  # WY representation: T = (I + StrictLower(k_beta_i . k_j * decay_ij))^-1.
  a = jnp.einsum("bhnik,bhnjk,bhnijk->bhnij", k_beta, k, decay, precision=_HIGHEST)
  a = jnp.where(strict, a, 0.0)
  eye = jnp.eye(chunk_size, dtype=jnp.float32)
  t = jax.scipy.linalg.solve_triangular(eye + a, jnp.broadcast_to(eye, a.shape), lower=True, unit_diagonal=True)
  u = jnp.matmul(t, v_beta, precision=_HIGHEST)  # [B, H, N, C, V]
  w = jnp.matmul(t, k_beta * jnp.exp(g), precision=_HIGHEST)  # [B, H, N, C, K]

  # Intra-chunk q.k scores with decay (diagonal included).
  qk = jnp.einsum("bhnik,bhnjk,bhnijk->bhnij", q, k, decay, precision=_HIGHEST)
  qk = jnp.where(causal, qk, 0.0)

  state0 = jnp.zeros((b, h, dk, dv), jnp.float32) if initial_state is None else initial_state.astype(jnp.float32)

  def step(state, xs):
    q_c, k_c, g_c, u_c, w_c, qk_c = xs
    v_new = u_c - jnp.matmul(w_c, state, precision=_HIGHEST)
    o = jnp.matmul(q_c * jnp.exp(g_c), state, precision=_HIGHEST) + jnp.matmul(qk_c, v_new, precision=_HIGHEST)
    g_last = g_c[..., -1:, :]  # [B, H, 1, K]
    state = state * jnp.exp(g_last).swapaxes(-1, -2) + jnp.matmul(
        (k_c * jnp.exp(g_last - g_c)).swapaxes(-1, -2), v_new, precision=_HIGHEST
    )
    return state, o

  # Scan over chunks: move N to the front.
  xs = tuple(jnp.moveaxis(x, 2, 0) for x in (q, k, g, u, w, qk))
  final_state, o = jax.lax.scan(step, state0, xs)
  o = jnp.moveaxis(o, 0, 2)  # [B, H, N, C, V]
  o = o.transpose(0, 2, 3, 1, 4).reshape(b, n * chunk_size, h, dv)[:, :s]
  return o.astype(out_dtype), final_state


def recurrent_kda(query, key, value, g, beta, initial_state=None, use_qk_l2norm=True):
  """Token-by-token KDA recurrence (reference for tests and single-token decode)."""
  out_dtype = query.dtype
  q, k, v, g, beta = (x.astype(jnp.float32) for x in (query, key, value, g, beta))
  if use_qk_l2norm:
    q = l2norm(q)
    k = l2norm(k)
  b, _, h, dk = k.shape
  dv = v.shape[-1]
  q = q * (dk**-0.5)
  state0 = jnp.zeros((b, h, dk, dv), jnp.float32) if initial_state is None else initial_state.astype(jnp.float32)

  def step(state, xs):
    q_t, k_t, v_t, g_t, b_t = xs  # [B, H, K], [B, H, K], [B, H, V], [B, H, K], [B, H]
    state = state * jnp.exp(g_t)[..., None]
    kv_mem = jnp.einsum("bhkv,bhk->bhv", state, k_t, precision=_HIGHEST)
    delta = (v_t - kv_mem) * b_t[..., None]
    state = state + k_t[..., :, None] * delta[..., None, :]
    o_t = jnp.einsum("bhkv,bhk->bhv", state, q_t, precision=_HIGHEST)
    return state, o_t

  xs = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta))
  final_state, o = jax.lax.scan(step, state0, xs)
  return jnp.moveaxis(o, 0, 1).astype(out_dtype), final_state


def causal_depthwise_conv1d(x: Array, kernel: Array) -> Array:
  """Causal depthwise conv. x: [B, S, C], kernel: [W, C] (kernel[W-1] multiplies the current token)."""
  width = kernel.shape[0]
  seq_len = x.shape[1]
  x_pad = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
  out = jnp.zeros(x.shape, jnp.float32)
  for i in range(width):
    out = out + x_pad[:, i : i + seq_len, :].astype(jnp.float32) * kernel[i].astype(jnp.float32)
  return out


class KimiDeltaAttention(nnx.Module):
  """KDA linear-attention block (projections, short conv, gates, chunked delta rule, gated norm).

  Parameter layout (HF name -> MaxText name):
    q_proj/k_proj/v_proj      -> q_proj/k_proj/v_proj        kernel [E, H, K]
    q/k/v_conv1d [HK, 1, W]   -> q_conv1d/k_conv1d/v_conv1d  [W, H*K]
    f_a_proj, f_b_proj        -> f_a_proj [E, K], f_b_proj [K, H, K]
    dt_bias [HK], A_log [H]   -> dt_bias [H, K], A_log [H]
    b_proj                    -> b_proj [E, H]
    g_a_proj, g_b_proj        -> g_a_proj [E, K], g_b_proj [K, H, K]
    o_norm [K]                -> o_norm [K]
    o_proj                    -> o_proj [H, K, E]
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      dtype: DType,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.dtype = dtype
    self.num_heads = config.kda_num_heads
    self.head_dim = config.kda_head_dim
    self.conv_width = config.kda_conv_kernel_dim
    self.lower_bound = config.kda_gate_lower_bound
    self.chunk_size = config.kda_chunk_size
    self.norm_eps = config.normalization_layer_epsilon
    emb = config.emb_dim
    h, d = self.num_heads, self.head_dim
    kernel_init = nd_dense_init(1.0, "fan_in", "normal")

    def dense(name, in_shape, out_shape, axes, in_axis=-1):
      return DenseGeneral(
          in_features_shape=in_shape,
          out_features_shape=out_shape,
          axis=in_axis,
          kernel_init=kernel_init,
          kernel_axes=axes,
          dtype=dtype,
          weight_dtype=get_weight_dtype(config, name),
          weight_quant=quantizations.get_weight_quant_config(config, name),
          quant=quant,
          matmul_precision=config.matmul_precision,
          shard_mode=config.shard_mode,
          rngs=rngs,
      )

    self.q_proj = dense("kda_q_proj", emb, (h, d), ("embed", "q_heads", "kv"))
    self.k_proj = dense("kda_k_proj", emb, (h, d), ("embed", "kv_heads", "kv"))
    self.v_proj = dense("kda_v_proj", emb, (h, d), ("embed", "kv_heads", "kv"))
    self.f_a_proj = dense("kda_f_a_proj", emb, d, ("embed", "kv"))
    self.f_b_proj = dense("kda_f_b_proj", d, (h, d), ("kv", "kv_heads", None))
    self.b_proj = dense("kda_b_proj", emb, h, ("embed", "kv_heads"))
    self.g_a_proj = dense("kda_g_a_proj", emb, d, ("embed", "kv"))
    self.g_b_proj = dense("kda_g_b_proj", d, (h, d), ("kv", "kv_heads", None))
    self.o_proj = dense("kda_o_proj", (h, d), emb, ("heads", "kv", "embed"), in_axis=(-2, -1))

    conv_dtype = get_weight_dtype(config, "kda_conv1d")
    conv_init = nnx.initializers.normal(stddev=0.1)
    self.q_conv1d = nnx.Param(conv_init(rngs.params(), (self.conv_width, h * d), conv_dtype), sharding=(None, "mlp"))
    self.k_conv1d = nnx.Param(conv_init(rngs.params(), (self.conv_width, h * d), conv_dtype), sharding=(None, "mlp"))
    self.v_conv1d = nnx.Param(conv_init(rngs.params(), (self.conv_width, h * d), conv_dtype), sharding=(None, "mlp"))
    # Decay parameters are kept in float32 (HF `_keep_in_fp32_modules_strict`).
    self.A_log = nnx.Param(jnp.zeros((h,), jnp.float32), sharding=(None,))
    self.dt_bias = nnx.Param(jnp.zeros((h, d), jnp.float32), sharding=(None, None))
    self.o_norm = nnx.Param(jnp.ones((d,), get_weight_dtype(config, "kda_o_norm")), sharding=(None,))

  def forget_gate(self, x: Array) -> Array:
    """Per-channel log-decay g in [lower_bound, 0): [B, S, H, K] float32."""
    f = self.f_b_proj(self.f_a_proj(x)).astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
    decay_rate = jnp.exp(self.A_log[...].astype(jnp.float32))[:, None]  # [H, 1]
    if self.lower_bound is not None:
      return self.lower_bound * jax.nn.sigmoid(decay_rate * f)
    return -decay_rate * jax.nn.softplus(f)

  def gated_norm(self, x: Array, gate: Array) -> Array:
    """RMSNorm over head_dim in float32, scaled by weight and sigmoid(gate)."""
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + self.norm_eps)
    x32 = x32 * self.o_norm[...].astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
    return x32.astype(x.dtype)

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Optional[Array] = None,
      decoder_segment_ids: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ):
    """Returns (output [B, S, E], None) to match the attention-module calling convention."""
    del inputs_kv, kwargs
    if model_mode != MODEL_MODE_TRAIN:
      # TODO(glm5.3): add conv/recurrent state caching for prefill + autoregressive decode.
      raise NotImplementedError("KimiDeltaAttention currently supports only full-sequence (train) mode.")
    x = inputs_q
    if decoder_segment_ids is not None:
      # Zero padding tokens (HF `apply_mask_to_padding_states`).
      x = x * (decoder_segment_ids != 0)[..., None].astype(x.dtype)
    b, s, _ = x.shape
    h, d = self.num_heads, self.head_dim

    def conv(proj, kernel):
      y = proj(x).reshape(b, s, h * d)
      return jax.nn.silu(causal_depthwise_conv1d(y, kernel[...])).astype(self.dtype).reshape(b, s, h, d)

    q = conv(self.q_proj, self.q_conv1d)
    k = conv(self.k_proj, self.k_conv1d)
    v = conv(self.v_proj, self.v_conv1d)
    g = self.forget_gate(x)
    beta = jax.nn.sigmoid(self.b_proj(x).astype(jnp.float32))

    o, _ = chunk_kda(q, k, v, g, beta, chunk_size=self.chunk_size, use_qk_l2norm=True)
    gate = self.g_b_proj(self.g_a_proj(x))
    o = self.gated_norm(o.astype(self.dtype), gate)
    return self.o_proj(o), None
