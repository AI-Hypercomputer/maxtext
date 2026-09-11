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
from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config
from maxtext.layers import initializers as max_initializers
from maxtext.layers import linears
from maxtext.layers.attention_op import AttentionOp
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


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
        use_bias=False,
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
    self.num_heads = config.num_query_heads
    self.head_k_dim = config.head_dim
    self.head_v_dim = config.head_dim
    self.hidden_dim = self.num_heads * self.head_k_dim  # 4096

    # Linear projections
    self.q_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.k_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.v_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
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
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.b_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.num_heads,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.g_proj = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=self.hidden_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
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
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

  def __call__(self, hidden_states, kv_cache=None, decoder_segment_ids=None, decoder_positions=None, **kwargs):
    """Recurrent KDA with segment resets and padding-safe convolution/state caches."""
    b, t, _ = hidden_states.shape
    h, d = self.num_heads, self.head_k_dim
    valid = jnp.ones((b, t), bool) if decoder_segment_ids is None else decoder_segment_ids != 0
    reset = jnp.zeros((b, t), bool) if decoder_positions is None else decoder_positions == 0
    initial = {
        "conv_q": jnp.zeros((b, h * d, 4), hidden_states.dtype),
        "conv_k": jnp.zeros((b, h * d, 4), hidden_states.dtype),
        "conv_v": jnp.zeros((b, h * d, 4), hidden_states.dtype),
        "rec_state": jnp.zeros((b, h, d, d), jnp.float32),
    }
    if kv_cache:
      initial = kv_cache
    projections = [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)]
    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))
    raw_gate = self.f_proj(hidden_states).astype(jnp.float32).reshape(b, t, h, d)
    decay = jnp.exp(
        -5.0
        * jax.nn.sigmoid(
            jnp.exp(self.A_log.value).reshape(1, 1, h, 1) * (raw_gate + self.dt_bias.value.reshape(1, 1, h, d))
        )
    )
    kernels = [layer.conv.kernel.value[:, 0, :].T for layer in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]

    def step(state, data):
      q_raw, k_raw, v_raw, beta_t, decay_t, valid_t, reset_t = data
      old_state = state
      state = {key: jnp.where(reset_t.reshape((b,) + (1,) * (x.ndim - 1)), 0, x) for key, x in state.items()}
      convs = []
      new = {}
      for key, x, kernel in zip(("conv_q", "conv_k", "conv_v"), (q_raw, k_raw, v_raw), kernels):
        new[key] = jnp.concatenate((state[key][..., 1:], x[..., None]), axis=-1)
        convs.append(jax.nn.silu(jnp.sum(new[key] * kernel, axis=-1)).astype(hidden_states.dtype).reshape(b, h, d))
      q, k, v = [x.astype(jnp.float32) for x in convs]
      q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * d**-0.5
      k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
      recurrent = state["rec_state"] * decay_t[..., None]
      residual = (v - jnp.einsum("bhk,bhkv->bhv", k, recurrent, precision=jax.lax.Precision.HIGHEST)) * beta_t[..., None]
      recurrent += jnp.einsum("bhk,bhv->bhkv", k, residual, precision=jax.lax.Precision.HIGHEST)
      new["rec_state"] = recurrent
      out = jnp.einsum("bhk,bhkv->bhv", q, recurrent, precision=jax.lax.Precision.HIGHEST)
      new = {key: jnp.where(valid_t.reshape((b,) + (1,) * (x.ndim - 1)), x, old_state[key]) for key, x in new.items()}
      return new, jnp.where(valid_t[:, None, None], out, 0)

    data = tuple(jnp.swapaxes(x, 0, 1) for x in (*projections, beta, decay, valid, reset))
    final, output = jax.lax.scan(step, initial, data)
    output = output.swapaxes(0, 1).astype(hidden_states.dtype)
    gate = self.g_proj(hidden_states).reshape(b, t, h, d)
    out = self.o_proj(self.o_norm(output, gate).reshape(b, t, h * d))
    return out, final if kv_cache is not None else None


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
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.wi_1 = linears.DenseGeneral(
        in_features_shape=dim,
        out_features_shape=intermediate_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=intermediate_dim,
        out_features_shape=dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        rngs=rngs,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    return self.wo(gate * up)


class Ling3MLAAttention(nnx.Module):
  """Text-only MLA with interleaved RoPE and a head-wise sigmoid output gate."""

  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    self.config = config
    self.num_heads = config.num_query_heads
    self.nope_dim = config.qk_nope_head_dim
    self.rope_dim = config.qk_rope_head_dim
    self.value_dim = config.v_head_dim
    self.rank = config.kv_lora_rank
    self.q_proj = _projection(config, config.emb_dim, self.num_heads * (self.nope_dim + self.rope_dim), rngs)
    self.kv_a_proj_with_mqa = _projection(config, config.emb_dim, self.rank + self.rope_dim, rngs)
    self.kv_a_layernorm = RMSNorm(
        num_features=self.rank,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.kv_b_proj = _projection(config, self.rank, self.num_heads * (self.nope_dim + self.value_dim), rngs)
    self.g_proj = _projection(config, config.emb_dim, self.num_heads, rngs)
    self.dense = _projection(config, self.num_heads * self.value_dim, config.emb_dim, rngs)
    # Keep the Ling3 projections/cache layout stable for existing checkpoints,
    # and share the attention arithmetic with MaxText's standard MLA path.
    self.attention_op = AttentionOp(
        config=config,
        mesh=mesh,
        attention_kernel="dot_product",
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        num_query_heads=self.num_heads,
        num_kv_heads=self.num_heads,
        float32_qk_product=True,
        float32_logits=True,
        dtype=config.dtype,
        rngs=rngs,
    )

  def _rope(self, x, positions):
    frequencies = self.config.rope_max_timescale ** (-jnp.arange(0, self.rope_dim, 2, dtype=jnp.float32) / self.rope_dim)
    angles = positions[..., None, None] * frequencies
    even, odd = x[..., 0::2], x[..., 1::2]
    return (
        jnp.stack(
            (even * jnp.cos(angles) - odd * jnp.sin(angles), odd * jnp.cos(angles) + even * jnp.sin(angles)), axis=-1
        )
        .reshape(x.shape)
        .astype(x.dtype)
    )

  def __call__(
      self, hidden_states, decoder_positions=None, decoder_segment_ids=None, kv_cache=None, model_mode="train", **kwargs
  ):
    b, t, _ = hidden_states.shape
    if decoder_positions is None:
      decoder_positions = jnp.broadcast_to(jnp.arange(t), (b, t))
    q = self.q_proj(hidden_states).reshape(b, t, self.num_heads, self.nope_dim + self.rope_dim)
    q = jnp.concatenate((q[..., : self.nope_dim], self._rope(q[..., self.nope_dim :], decoder_positions)), axis=-1)
    kv_a = self.kv_a_proj_with_mqa(hidden_states)
    kv = self.kv_b_proj(self.kv_a_layernorm(kv_a[..., : self.rank]))
    kv = kv.reshape(b, t, self.num_heads, self.nope_dim + self.value_dim)
    k_rope = self._rope(kv_a[..., self.rank :, None].swapaxes(-1, -2), decoder_positions)
    k = jnp.concatenate(
        (kv[..., : self.nope_dim], jnp.broadcast_to(k_rope, (b, t, self.num_heads, self.rope_dim))), axis=-1
    )
    v = kv[..., self.nope_dim :]
    segments = jnp.ones((b, t), jnp.int32) if decoder_segment_ids is None else decoder_segment_ids
    key_positions = decoder_positions
    key_segments = segments
    if kv_cache and kv_cache.get("fixed", False):
      batch_indices = jnp.arange(b)[:, None]
      k = kv_cache["key"].at[batch_indices, decoder_positions].set(k)
      v = kv_cache["value"].at[batch_indices, decoder_positions].set(v)
      key_positions = kv_cache["positions"].at[batch_indices, decoder_positions].set(decoder_positions)
      key_segments = kv_cache["segments"].at[batch_indices, decoder_positions].set(segments)
    elif kv_cache:
      k = jnp.concatenate((kv_cache["key"], k), axis=1)
      v = jnp.concatenate((kv_cache["value"], v), axis=1)
      key_positions = jnp.concatenate((kv_cache["positions"], key_positions), axis=1)
      key_segments = jnp.concatenate((kv_cache["segments"], key_segments), axis=1)
    mask = (
        (key_positions[:, None, :] <= decoder_positions[:, :, None])
        & (segments[:, :, None] == key_segments[:, None, :])
        & (segments[:, :, None] != 0)
        & (key_segments[:, None, :] != 0)
    )
    scores = self.attention_op.qk_product(
        q.astype(jnp.float32),
        k.astype(jnp.float32),
        t,
        model_mode,
        partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST),
    )
    scores /= (self.nope_dim + self.rope_dim) ** 0.5
    # Ling3 caches carry absolute positions and segment IDs for both Q and KV.
    # Preserve their mask, including packed sequences and unused fixed slots.
    scores = jnp.where(mask[:, None, None], scores, -1e30)
    local_out, local_max, local_sum = self.attention_op.compute_local_attention(scores, v, t, model_mode, jnp.einsum)
    out = self.attention_op.normalize_attention([local_out], [local_max], [local_sum]).astype(v.dtype)
    out *= jax.nn.sigmoid(self.g_proj(hidden_states).astype(jnp.float32)).astype(out.dtype)[..., None]
    out = self.dense(out.reshape(b, t, -1))
    cache = None if kv_cache is None else dict(key=k, value=v, positions=key_positions, segments=key_segments)
    return out, cache


def _projection(config, input_dim, output_dim, rngs):
  return linears.DenseGeneral(
      in_features_shape=input_dim,
      out_features_shape=output_dim,
      use_bias=False,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      matmul_precision=config.matmul_precision,
      kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
      rngs=rngs,
  )


class Ling3MoE(nnx.Module):
  """Grouped sigmoid routing; biases select experts but do not change mixture weights."""

  def __init__(self, config, mesh, model_mode, layer_idx, quant=None, *, rngs):
    self.config = config
    self.layer_idx = layer_idx
    self.gate = _projection(config, config.emb_dim, config.num_experts, rngs)
    self.expert_bias = nnx.Param(jnp.zeros(config.num_experts, jnp.float32))
    init = max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    for name, shape in [
        ("wi_0", (config.num_experts, config.emb_dim, config.moe_mlp_dim)),
        ("wi_1", (config.num_experts, config.emb_dim, config.moe_mlp_dim)),
        ("wo", (config.num_experts, config.moe_mlp_dim, config.emb_dim)),
    ]:
      # Use the standard MoE logical axes so checkpoint restore and state
      # initialization partition the expert bank instead of replicating it.
      axes = ("exp", "mlp_moe", "embed_moe") if name == "wo" else ("exp", "embed_moe", "mlp_moe")
      setattr(
          self,
          name,
          nnx.Param(init(rngs.params(), shape, config.weight_dtype, in_axis=1, out_axis=2), out_sharding=axes),
      )
    self.shared_wi_0 = _projection(config, config.emb_dim, config.moe_mlp_dim, rngs)
    self.shared_wi_1 = _projection(config, config.emb_dim, config.moe_mlp_dim, rngs)
    self.shared_wo = _projection(config, config.moe_mlp_dim, config.emb_dim, rngs)

  def __call__(self, inputs, forced_routed_experts=None, **kwargs):
    cfg = self.config
    # Router math is fp32 even when the expert matmuls use bf16.
    logits = jnp.einsum(
        "...d,de->...e",
        inputs.astype(jnp.float32),
        self.gate.kernel.value.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = jax.nn.sigmoid(logits)
    corrected = scores + self.expert_bias.value
    grouped = corrected.reshape(*corrected.shape[:-1], cfg.n_routing_groups, -1)
    group_scores = jax.lax.top_k(grouped, 2)[0].sum(-1)
    groups = jax.lax.top_k(group_scores, cfg.topk_routing_group)[1]
    group_mask = jax.nn.one_hot(groups, cfg.n_routing_groups).sum(-2)
    mask = jnp.repeat(group_mask, cfg.num_experts // cfg.n_routing_groups, axis=-1)
    selected = jax.lax.top_k(jnp.where(mask, corrected, -jnp.inf), cfg.num_experts_per_tok)[1]
    if forced_routed_experts is not None:
      selected = forced_routed_experts
    weights = jnp.take_along_axis(scores, selected, axis=-1)
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-20) * cfg.routed_scaling_factor
    flat_x = inputs.reshape(-1, cfg.emb_dim)
    indices = selected.reshape(-1, cfg.num_experts_per_tok)
    weights = weights.reshape(-1, cfg.num_experts_per_tok)
    # Published late-layer SwiGLU limits; layers 0..34 are unclipped.
    limit = 4.0 if self.layer_idx >= 35 else None
    shared_limit = (7.0 if self.layer_idx >= 40 else 5.0) if self.layer_idx >= 35 else None

    def swiglu(gate, up, bound):
      gate = jax.nn.silu(gate)
      if bound is not None:
        gate = jnp.minimum(gate, bound)
        up = jnp.clip(up, -bound, bound)
      return gate * up

    def expert_step(i, total):
      expert = indices[:, i]
      gate = jnp.einsum(
          "td,tdf->tf", flat_x, self.wi_0.value[expert].astype(inputs.dtype), precision=cfg.matmul_precision
      )
      up = jnp.einsum("td,tdf->tf", flat_x, self.wi_1.value[expert].astype(inputs.dtype), precision=cfg.matmul_precision)
      out = jnp.einsum(
          "tf,tfd->td",
          swiglu(gate, up, limit),
          self.wo.value[expert].astype(inputs.dtype),
          precision=cfg.matmul_precision,
      )
      return total + out * weights[:, i, None].astype(out.dtype)

    routed = jax.lax.fori_loop(0, cfg.num_experts_per_tok, expert_step, jnp.zeros_like(flat_x)).reshape(inputs.shape)
    shared = self.shared_wo(swiglu(self.shared_wi_0(inputs), self.shared_wi_1(inputs), shared_limit))
    return routed + shared


class Ling3Cache(nnx.Module):
  """Per-leaf cache metadata lets MaxEngine insert prefills into decode slots."""

  def __init__(self, values):
    for name, value in values.items():
      setattr(self, name, nnx.Cache(value, sharding=("cache_batch",) + (None,) * (value.ndim - 1)))

  def read(self):
    return {name: variable[...] for name, variable in vars(self).items() if isinstance(variable, nnx.Cache)}

  def write(self, values):
    for name, value in values.items():
      getattr(self, name)[...] = value


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

    self.cache = nnx.data(None)
    if model_mode != "train":
      batch, _ = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      if (layer_idx + 1) % config.inhomogeneous_layer_cycle_interval:
        h, d = config.num_query_heads, config.head_dim
        self.cache = Ling3Cache(
            {
                "conv_q": jnp.zeros((batch, h * d, 4), config.dtype),
                "conv_k": jnp.zeros((batch, h * d, 4), config.dtype),
                "conv_v": jnp.zeros((batch, h * d, 4), config.dtype),
                "rec_state": jnp.zeros((batch, h, d, d), jnp.float32),
            }
        )
      else:
        self.cache = Ling3Cache(
            {
                "key": jnp.zeros(
                    (
                        batch,
                        config.max_target_length,
                        config.num_query_heads,
                        config.qk_nope_head_dim + config.qk_rope_head_dim,
                    ),
                    config.dtype,
                ),
                "value": jnp.zeros(
                    (batch, config.max_target_length, config.num_query_heads, config.v_head_dim), config.dtype
                ),
                "positions": jnp.zeros((batch, config.max_target_length), jnp.int32),
                "segments": jnp.zeros((batch, config.max_target_length), jnp.int32),
            }
        )

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
      # Ling MLA uses a head-wise output gate and interleaved text RoPE.
      self.attention = Ling3MLAAttention(
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
      self.mlp = Ling3MoE(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          layer_idx=layer_idx,
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
    # An external cache takes precedence (used by explicit-cache callers).
    internal_cache = kv_cache is None and self.cache is not None and model_mode != "train"
    if internal_cache:
      kv_cache = self.cache.read()
      if model_mode == "prefill":
        kv_cache = jax.tree.map(jnp.zeros_like, kv_cache)
      # Fixed MLA buffers are updated at absolute text positions.
      if "key" in kv_cache:
        kv_cache = dict(kv_cache, fixed=True)
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
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
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
    if internal_cache:
      self.cache.write(new_kv)
    return y, new_kv


class Ling3ScannableBlock(nnx.Module):
  """Scanned 6-layer cycle for Ling-3.0."""

  def __init__(self, config: Config, mesh: Mesh | None, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    if config.num_decoder_layers != config.inhomogeneous_layer_cycle_interval:
      raise ValueError(
          "Ling3 scanned mode currently supports exactly one six-layer cycle; use scan_layers=false otherwise."
      )

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
