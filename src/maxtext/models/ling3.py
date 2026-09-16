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

"""Ling-3.0 decoder layers (Kimi Delta Attention and MLA hybrid)."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config
from maxtext.layers import initializers as max_initializers
from maxtext.layers import linears
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import deepseek


class Ling3CausalConv1D(nnx.Module):
  """Depthwise Causal Conv1D for Ling3 KDA.

  Equivalent to PyTorch short conv with kernel_size W:
  x: [B, T, D] -> Conv1D(kernel_size=W, groups=D) -> [B, T, D]
  followed by SiLU activation.
  """

  def __init__(
      self,
      config: Config,
      dim: int,
      kernel_size: int = 4,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.dim = dim
    self.kernel_size = kernel_size
    self.mesh = mesh

    # Weights: Flax nnx.Conv uses (kernel_size, in_features, out_features) for 1D.
    # For depthwise, feature_group_count = dim, so in_features = 1, out_features = dim.
    # Weight shape: (kernel_size, 1, dim).
    self.conv = nnx.Conv(
        in_features=dim,
        out_features=dim,
        kernel_size=(kernel_size,),
        feature_group_count=dim,
        padding="VALID",
        use_bias=True,
        rngs=rngs,
    )

  def __call__(
      self,
      x: jnp.ndarray,
      cache: jnp.ndarray | None = None,
      output_final_state: bool = False,
  ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Applies causal conv1d.

    Args:
      x: [B, T, D] input tensor.
      cache: [B, D, W] previous conv state (or None).
      output_final_state: whether to return the final conv state.

    Returns:
      (out, final_state) where out is [B, T, D] after SiLU, and final_state is [B, D, W] or None.
    """
    W = self.kernel_size

    if cache is not None:
      # cache: [B, D, W], previous tokens are the last W-1 elements
      prev_x = cache[..., -(W - 1) :]  # [B, D, W-1]
      x_t = jnp.swapaxes(x, 1, 2)  # [B, D, T]
      x_full_t = jnp.concatenate([prev_x, x_t], axis=-1)  # [B, D, W-1+T]
    else:
      x_t = jnp.swapaxes(x, 1, 2)  # [B, D, T]
      pad_widths = ((0, 0), (0, 0), (W - 1, 0))
      x_full_t = jnp.pad(x_t, pad_widths)  # [B, D, W-1+T]

    final_state = None
    if output_final_state:
      final_state = x_full_t[..., -W:]  # [B, D, W]

    x_full = jnp.swapaxes(x_full_t, 1, 2)  # [B, W-1+T, D]
    conv_out = self.conv(x_full)  # [B, T, D] with 'VALID' padding implicit in slicing
    # Note: nnx.Conv defaults to 'VALID' padding: (W-1+T) - W + 1 = T outputs!
    out = jax.nn.silu(conv_out)
    return out, final_state


class Ling3RMSNormGated(nnx.Module):
  """RMSNorm followed by gate scaling: out = rms_norm(x) * sigmoid(g)."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.scale = nnx.Param(jnp.ones((num_features,), dtype=dtype))

  def __call__(self, x: jnp.ndarray, g: jnp.ndarray | None = None) -> jnp.ndarray:
    # x: [..., num_features]
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = (x * jax.lax.rsqrt(variance + self.epsilon)).astype(x.dtype)
    normed = normed * self.scale.value
    if g is not None:
      normed = normed * jax.nn.sigmoid(g.astype(jnp.float32)).astype(x.dtype)
    return normed


class Ling3KimiDeltaAttention(nnx.Module):
  """Ling-3.0 Kimi Delta Attention (KDA) layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant

    dim = config.emb_dim  # 2560
    # KDA head dimensions
    self.num_heads = getattr(config, "num_kda_heads", 32)
    self.head_k_dim = getattr(config, "kda_head_dim", 128)
    self.head_v_dim = getattr(config, "kda_head_dim", 128)
    self.hidden_dim = self.num_heads * self.head_k_dim  # 4096

    # Linear projections
    self.q_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.k_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.v_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

    # Causal Convolutions (kernel_size=4)
    self.q_conv1d = Ling3CausalConv1D(config=config, dim=self.hidden_dim, kernel_size=4, mesh=mesh, rngs=rngs)
    self.k_conv1d = Ling3CausalConv1D(config=config, dim=self.hidden_dim, kernel_size=4, mesh=mesh, rngs=rngs)
    self.v_conv1d = Ling3CausalConv1D(config=config, dim=self.hidden_dim, kernel_size=4, mesh=mesh, rngs=rngs)

    # Gating and beta projections
    self.f_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.b_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.num_heads,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.g_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

    # KDA recurrence parameters: A_log (num_heads,) and dt_bias (hidden_dim,)
    self.A_log = nnx.Param(jnp.zeros((self.num_heads,), dtype=jnp.float32))
    self.dt_bias = nnx.Param(jnp.zeros((self.hidden_dim,), dtype=jnp.float32))

    # Output gated norm & projection
    self.o_norm = Ling3RMSNormGated(
        num_features=self.head_v_dim, epsilon=config.normalization_layer_epsilon, dtype=config.dtype, rngs=rngs
    )
    self.o_proj = linears.DenseGeneral(
        in_features_shape=self.hidden_dim,
        out_features_shape=dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

  def _kda_recurrence(
      self,
      q: jnp.ndarray,
      k: jnp.ndarray,
      v: jnp.ndarray,
      decay: jnp.ndarray,
      beta: jnp.ndarray,
      initial_state: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies KDA recurrent delta rule via lax.scan over sequence length T.

    Args:
      q, k: [B, T, H, K]
      v: [B, T, H, V]
      decay: [B, T, H, K]
      beta: [B, T, H, 1]
      initial_state: [B, H, K, V] or None

    Returns:
      (out, final_state) where out is [B, T, H, V] and final_state is [B, H, K, V].
    """
    B, _, H, K = q.shape
    V = v.shape[-1]

    if initial_state is None:
      initial_state = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    # Transpose to [T, B, ...] for scan
    q_scan = jnp.swapaxes(q, 0, 1).astype(jnp.float32)
    k_scan = jnp.swapaxes(k, 0, 1).astype(jnp.float32)
    v_scan = jnp.swapaxes(v, 0, 1).astype(jnp.float32)
    decay_scan = jnp.swapaxes(decay, 0, 1).astype(jnp.float32)
    beta_scan = jnp.swapaxes(beta, 0, 1).astype(jnp.float32)

    def scan_fn(S, step_inputs):
      q_t, k_t, v_t, d_t, b_t = step_inputs
      # d_t: [B, H, K, 1]
      S = jnp.expand_dims(d_t, -1) * S
      # v_pred = sum_k (k_t * S) -> [B, H, V]
      v_pred = jnp.einsum("bhk,bhkv->bhv", k_t, S)
      delta_v = (v_t - v_pred) * b_t
      # S += k_t outer delta_v
      S = S + jnp.einsum("bhk,bhv->bhkv", k_t, delta_v)
      # o_t = q_t @ S
      o_t = jnp.einsum("bhk,bhkv->bhv", q_t, S)
      return S, o_t

    final_state, o_scan = jax.lax.scan(
        scan_fn,
        initial_state,
        (q_scan, k_scan, v_scan, decay_scan, beta_scan),
    )
    # Swap back [T, B, H, V] -> [B, T, H, V]
    out = jnp.swapaxes(o_scan, 0, 1).astype(v.dtype)
    return out, final_state

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      kv_cache: Any = None,
      **kwargs,
  ) -> tuple[jnp.ndarray, Any]:
    B, T, _ = hidden_states.shape

    # 1. Projections & 1D Convolutions
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    conv_cache_q, conv_cache_k, conv_cache_v = None, None, None
    rec_cache = None
    if kv_cache is not None and isinstance(kv_cache, dict):
      conv_cache_q = kv_cache.get("conv_q")
      conv_cache_k = kv_cache.get("conv_k")
      conv_cache_v = kv_cache.get("conv_v")
      rec_cache = kv_cache.get("rec_state")

    output_cache = kv_cache is not None
    q, conv_state_q = self.q_conv1d(q, cache=conv_cache_q, output_final_state=output_cache)
    k, conv_state_k = self.k_conv1d(k, cache=conv_cache_k, output_final_state=output_cache)
    v, conv_state_v = self.v_conv1d(v, cache=conv_cache_v, output_final_state=output_cache)

    # 2. Gating and Beta
    f_raw = self.f_proj(hidden_states)
    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))  # [B, T, H]
    beta = jnp.expand_dims(beta, -1)  # [B, T, H, 1]

    # 3. Reshape to multi-head: [B, T, H, D]
    q = jnp.reshape(q, (B, T, self.num_heads, self.head_k_dim))
    k = jnp.reshape(k, (B, T, self.num_heads, self.head_k_dim))
    v = jnp.reshape(v, (B, T, self.num_heads, self.head_v_dim))
    f_raw = jnp.reshape(f_raw, (B, T, self.num_heads, self.head_k_dim))

    # L2 normalize q and k
    q = q / (jnp.linalg.norm(q.astype(jnp.float32), ord=2, axis=-1, keepdims=True) + 1e-12)
    k = k / (jnp.linalg.norm(k.astype(jnp.float32), ord=2, axis=-1, keepdims=True) + 1e-12)
    scale = self.head_k_dim**-0.5
    q = (q * scale).astype(hidden_states.dtype)

    # KDA Decay gate
    # equation: g = -5.0 * sigmoid(exp(A_log) * (f_raw + dt_bias))
    A = jnp.exp(self.A_log.value).reshape((1, 1, self.num_heads, 1))
    dt_b = self.dt_bias.value.reshape((1, 1, self.num_heads, self.head_k_dim))
    g = -5.0 * jax.nn.sigmoid(A * (f_raw.astype(jnp.float32) + dt_b))
    decay = jnp.exp(g)

    # 4. KDA Recurrence
    o_core, final_rec_state = self._kda_recurrence(
        q=q,
        k=k,
        v=v,
        decay=decay,
        beta=beta,
        initial_state=rec_cache,
    )

    # 5. Output gate and RMSNormGated
    out_gate = self.g_proj(hidden_states)
    out_gate = jnp.reshape(out_gate, (B, T, self.num_heads, self.head_v_dim))
    o_normed = self.o_norm(o_core, out_gate)  # [B, T, H, V]

    # 6. Output projection
    o_flat = jnp.reshape(o_normed, (B, T, self.hidden_dim))
    out = self.o_proj(o_flat)

    new_kv_cache = None
    if output_cache:
      new_kv_cache = {
          "conv_q": conv_state_q,
          "conv_k": conv_state_k,
          "conv_v": conv_state_v,
          "rec_state": final_rec_state,
      }

    return out, new_kv_cache


class Ling3DenseMlp(nnx.Module):
  """Dense SwiGLU MLP for Ling-3.0 (layers 0 and 1)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant

    dim = config.emb_dim
    intermediate_dim = config.mlp_dim

    self.wi_0 = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=intermediate_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.wi_1 = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=intermediate_dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=intermediate_dim,
        out_features_shape=dim,
        use_bias=False,
        dtype=config.dtype,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    return self.wo(gate * up)


class Ling3DecoderLayer(nnx.Module):
  """Single decoder layer for Ling-3.0.

  Layers % 6 != 5: Kimi Delta Attention (KDA).
  Layers % 6 == 5: MLA Attention.
  Layers < 2: Dense SwiGLU MLP.
  Layers >= 2: MoE MLP.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant

    # Pre-attention norm
    self.input_layernorm = RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    # Attention block
    is_mla_layer = (self.layer_idx + 1) % config.inhomogeneous_layer_cycle_interval == 0
    if is_mla_layer:
      # DeepSeek MLA layer
      self.attention = deepseek.DeepSeekAttention(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=layer_idx,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.attention = Ling3KimiDeltaAttention(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=layer_idx,
          quant=quant,
          rngs=rngs,
      )

    # Post-attention norm
    self.post_attention_layernorm = RMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    # MLP block
    if self.layer_idx < config.first_num_dense_layers:
      self.mlp = Ling3DenseMlp(
          config=config,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = deepseek.DeepSeekMoEBlock(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          rngs=rngs,
      )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = True,
      model_mode: str = "prefill",
      previous_chunk: Any = None,
      slot: None | int = None,
      kv_cache: Any = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
      **kwargs,
  ) -> tuple[jnp.ndarray, Any]:
    # 1. Pre-norm and Attention
    normed_x = self.input_layernorm(inputs)
    is_mla_layer = (self.layer_idx + 1) % self.config.inhomogeneous_layer_cycle_interval == 0

    if is_mla_layer:
      attn_out, new_kv = self.attention(
          normed_x,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      attn_out, new_kv = self.attention(
          normed_x,
          kv_cache=kv_cache,
      )

    x = inputs + attn_out

    # 2. Post-norm and MLP
    normed_post = self.post_attention_layernorm(x)
    if self.layer_idx < self.config.first_num_dense_layers:
      mlp_out = self.mlp(normed_post)
    else:
      mlp_out = self.mlp(
          normed_post,
          deterministic=deterministic,
          forced_routed_experts=forced_routed_experts,
      )

    y = x + mlp_out
    return y, new_kv


class Ling3ScannableBlock(nnx.Module):
  """Scanned 6-layer cycle for Ling-3.0."""

  def __init__(self, config: Config, mesh: Mesh | None, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs

    for i in range(config.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()
      layer = Ling3DecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=i,
          quant=quant,
          rngs=layer_rngs,
      )
      setattr(self, f"layer_{i}", layer)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray = None,
      decoder_positions: None | jnp.ndarray = None,
      deterministic: bool = True,
      model_mode: str = "prefill",
      previous_chunk: Any = None,
      slot: None | int = None,
      forced_routed_experts: jnp.ndarray | None = None,
      **kwargs,
  ) -> tuple[Array, None]:
    x = carry
    for i in range(self.config.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      layer_forced_routed_experts = forced_routed_experts[i] if forced_routed_experts is not None else None
      x, _ = layer(
          x,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          forced_routed_experts=layer_forced_routed_experts,
      )
    return x, None
