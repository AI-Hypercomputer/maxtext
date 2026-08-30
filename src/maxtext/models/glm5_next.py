# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.3-Flash / GLM-5-Next model components."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, HyperConnectionType, ModelMode
from maxtext.layers import initializers, linears, mhc
from maxtext.layers.normalizations import RMSNorm


def causal_conv1d(
    inputs: Array,
    weight: Array,
    bias: Array | None = None,
) -> Array:
  """1D depthwise causal convolution for KDA attention.

  inputs: [B, S, C]
  weight: [K, C] (depthwise filter of kernel size K and channels C)
  """
  k, c = weight.shape
  pad_inputs = jnp.pad(inputs, ((0, 0), (k - 1, 0), (0, 0)))
  # Conv in JAX with dimension numbers (B, S, C), (K, 1, C)
  w = weight[:, None, :]  # [K, 1, C]
  out = jax.lax.conv_general_dilated(
      lhs=pad_inputs,
      rhs=w,
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NHC", "HIO", "NHC"),
      feature_group_count=c,
  )
  if bias is not None:
    out = out + bias
  return out


class Glm5NextAttention(nnx.Module):
  """GLM-5.3-Flash KDA (Knowledge-Driven Attention / Gated Delta Attention) Layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: ModelMode,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.emb_dim = config.emb_dim
    self.num_heads = config.num_query_heads
    self.head_dim = config.head_dim
    self.kda_conv_size = getattr(config, "kda_conv_size", 4)
    self.conv_dim = self.num_heads * self.head_dim

    # Projections
    self.q_proj = linears.Dense(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "q_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.k_proj = linears.Dense(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.v_proj = linears.Dense(
        self.emb_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.b_proj = linears.Dense(
        self.emb_dim,
        self.num_heads,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.f_a_proj = linears.Dense(
        self.emb_dim,
        self.num_heads,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.f_b_proj = linears.Dense(
        self.num_heads,
        self.num_heads,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_a_proj = linears.Dense(
        self.emb_dim,
        self.num_heads,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_b_proj = linears.Dense(
        self.num_heads,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.o_proj = linears.Dense(
        self.conv_dim,
        self.emb_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "embed"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Conv1D on concatenated Q, K, V
    conv_features = 3 * self.conv_dim
    self.conv1d = nnx.Conv(
        in_features=conv_features,
        out_features=conv_features,
        kernel_size=(self.kda_conv_size,),
        feature_group_count=conv_features,
        use_bias=False,
        padding="VALID",
        rngs=self.rngs,
    )

    self.A_log = nnx.Param(
        jnp.zeros((self.num_heads,), dtype=self.weight_dtype),
    )
    self.dt_bias = nnx.Param(
        jnp.zeros((self.num_heads,), dtype=self.weight_dtype),
    )

    self.o_norm = RMSNorm(
        dim=self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array | None = None,
      **kwargs,
  ) -> tuple[Array, None]:
    x = inputs_q
    b, s, _ = x.shape

    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # 1D causal convolution on concatenated (q, k, v)
    qkv = jnp.concatenate([q, k, v], axis=-1)
    pad_qkv = jnp.pad(qkv, ((0, 0), (self.kda_conv_size - 1, 0), (0, 0)))
    conv_out = jax.nn.silu(self.conv1d(pad_qkv))
    q, k, v = jnp.split(conv_out, 3, axis=-1)

    # Beta gating (dt)
    b_val = self.b_proj(x)
    dt = jax.nn.softplus(b_val + self.dt_bias[...])

    # Forget gate (g)
    f_a = self.f_a_proj(x)
    f_b = self.f_b_proj(f_a)
    a_val = -jnp.exp(self.A_log[...].astype(jnp.float32))
    g = jnp.exp(a_val * dt * jax.nn.sigmoid(f_b))

    # Output gate
    g_a = self.g_a_proj(x)
    g_b = self.g_b_proj(g_a)
    out_gate = jax.nn.sigmoid(g_b)

    # Reshape into [B, S, H, D]
    q = jnp.reshape(q, (b, s, self.num_heads, self.head_dim))
    k = jnp.reshape(k, (b, s, self.num_heads, self.head_dim))
    v = jnp.reshape(v, (b, s, self.num_heads, self.head_dim))

    # Gated Delta recurrence / causal scan
    q = q * (self.head_dim**-0.5)
    beta = dt[..., None]
    k_beta = k * beta

    def scan_step(state, inputs):
      g_t, q_t, k_t, v_t, k_beta_t = inputs
      g_t = g_t[..., None, None]
      state = state * g_t
      kv_mem = jnp.einsum("hd,he->hde", state, k_t)
      delta = v_t - kv_mem
      state = state + jnp.einsum("hd,he->hde", delta, k_beta_t)
      out_t = jnp.einsum("hde,hd->he", state, q_t)
      return state, out_t

    init_state = jnp.zeros((b, self.num_heads, self.head_dim, self.head_dim), dtype=q.dtype)

    def scan_over_seq(init_s, q_seq, k_seq, v_seq, k_b_seq, g_seq):
      _, seq_out = jax.lax.scan(
          scan_step,
          init_s,
          (g_seq, q_seq, k_seq, v_seq, k_b_seq),
      )
      return seq_out

    inputs_q_t = jnp.swapaxes(q, 0, 1)
    inputs_k_t = jnp.swapaxes(k, 0, 1)
    inputs_v_t = jnp.swapaxes(v, 0, 1)
    inputs_kb_t = jnp.swapaxes(k_beta, 0, 1)
    inputs_g_t = jnp.swapaxes(g, 0, 1)

    scan_vmap = jax.vmap(scan_over_seq, in_axes=(0, 1, 1, 1, 1, 1), out_axes=1)
    scan_out = scan_vmap(init_state, inputs_q_t, inputs_k_t, inputs_v_t, inputs_kb_t, inputs_g_t)
    scan_out = jnp.swapaxes(scan_out, 0, 1)

    norm_out = self.o_norm(scan_out)
    norm_flat = jnp.reshape(norm_out, (b, s, self.conv_dim))
    gated_out = norm_flat * out_gate
    output = self.o_proj(gated_out)
    return output, None


class Glm5NextDecoderLayer(nnx.Module):
  """GLM-5.3-Flash Decoder Layer wrapping Attention and MLP in mHC blocks."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: ModelMode,
      layer_idx: int,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.rngs = rngs

    self.input_layernorm = RMSNorm(
        dim=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        dim=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    self.attention = Glm5NextAttention(
        config=config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )
    self.mlp = linears.MlpBlock(
        config=config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

    self.attn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.ffn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      deterministic: bool = True,
      model_mode: ModelMode = "train",
  ) -> tuple[Array, None]:
    """Forward pass for GLM-5.3-Flash Decoder Layer."""
    x = inputs

    x, _ = self.attn_hc(
        norm_fn=self.input_layernorm,
        branch_fn=self.attention,
        x=x,
        mhc_type=HyperConnectionType.ATTENTION,
    )

    x, _ = self.ffn_hc(
        norm_fn=self.post_attention_layernorm,
        branch_fn=self.mlp,
        x=x,
        mhc_type=HyperConnectionType.MLP_DENSE,
    )

    return x, None
