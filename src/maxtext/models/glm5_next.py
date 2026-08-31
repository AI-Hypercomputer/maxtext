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

"""GLM-5.3-Flash model components based on the upstream GLM-5-Next architecture."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, HyperConnectionType, MODEL_MODE_TRAIN
from maxtext.layers import initializers, linears, mhc, moe, nnx_wrappers
from maxtext.layers.normalizations import RMSNorm


def l2norm(t: Array, eps: float = 1e-6) -> Array:
  return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps)


class Glm5NextAttention(nnx.Module):
  """GLM-5.3-Flash KDA (Knowledge-Driven Attention / Gated Delta Attention) Layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.emb_dim = config.emb_dim
    self.num_heads = config.linear_num_heads
    self.head_dim = config.linear_head_dim
    self.kda_conv_size = config.linear_conv_kernel_dim
    self.conv_dim = self.num_heads * self.head_dim

    # Projections
    self.q_proj = linears.DenseGeneral(
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
    self.k_proj = linears.DenseGeneral(
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
    self.v_proj = linears.DenseGeneral(
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
    self.b_proj = linears.DenseGeneral(
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
    self.f_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "head_dim"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.f_b_proj = linears.DenseGeneral(
        self.head_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("head_dim", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "head_dim"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.g_b_proj = linears.DenseGeneral(
        self.head_dim,
        self.conv_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("head_dim", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.o_proj = linears.DenseGeneral(
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
        jnp.zeros((self.conv_dim,), dtype=self.weight_dtype),
    )

    self.o_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("head_dim",),
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

    # Per-head update strength (beta).
    b_val = self.b_proj(x)
    beta = jax.nn.sigmoid(b_val)[..., None]

    # Per-channel forget gate.
    f_a = self.f_a_proj(x)
    f_b = self.f_b_proj(f_a)
    decay_rate = jnp.exp(self.A_log[...].astype(jnp.float32))[None, None, :, None]
    f_per_channel = jnp.reshape(f_b + self.dt_bias[...], (b, s, self.num_heads, self.head_dim))
    g_log = -5.0 * jax.nn.sigmoid(decay_rate * f_per_channel)
    g = jnp.exp(g_log)

    # Output gate
    g_a = self.g_a_proj(x)
    g_b = self.g_b_proj(g_a)
    gate = jnp.reshape(g_b, (b, s, self.num_heads, self.head_dim))

    # Reshape into [B, S, H, D]
    q = jnp.reshape(q, (b, s, self.num_heads, self.head_dim))
    k = jnp.reshape(k, (b, s, self.num_heads, self.head_dim))
    v = jnp.reshape(v, (b, s, self.num_heads, self.head_dim))

    # L2 Norm and scale
    q = l2norm(q) * (self.head_dim**-0.5)
    k = l2norm(k)

    def scan_step(state, inputs):
      g_t, q_t, k_t, v_t, beta_t = inputs
      g_t = g_t[..., None]
      state = state * g_t
      kv_mem = jnp.einsum("hde,hd->he", state, k_t)
      delta = (v_t - kv_mem) * beta_t
      state = state + jnp.einsum("hd,he->hde", k_t, delta)
      out_t = jnp.einsum("hde,hd->he", state, q_t)
      return state, out_t

    init_state = jnp.zeros((b, self.num_heads, self.head_dim, self.head_dim), dtype=q.dtype)

    def scan_over_seq(init_s, q_seq, k_seq, v_seq, b_seq, g_seq):
      _, seq_out = jax.lax.scan(
          scan_step,
          init_s,
          (g_seq, q_seq, k_seq, v_seq, b_seq),
      )
      return seq_out

    inputs_q_t = jnp.swapaxes(q, 0, 1)
    inputs_k_t = jnp.swapaxes(k, 0, 1)
    inputs_v_t = jnp.swapaxes(v, 0, 1)
    inputs_b_t = jnp.swapaxes(beta, 0, 1)
    inputs_g_t = jnp.swapaxes(g, 0, 1)

    scan_vmap = jax.vmap(scan_over_seq, in_axes=(0, 1, 1, 1, 1, 1), out_axes=1)
    scan_out = scan_vmap(init_state, inputs_q_t, inputs_k_t, inputs_v_t, inputs_b_t, inputs_g_t)
    scan_out = jnp.swapaxes(scan_out, 0, 1)

    norm_out = self.o_norm(scan_out) * jax.nn.sigmoid(gate)
    norm_flat = jnp.reshape(norm_out, (b, s, self.conv_dim))
    output = self.o_proj(norm_flat)
    return output, None


class Glm5NextSparseAttention(nnx.Module):
  """GLM-5.3-Flash MLA / DeepSeek Sparse Attention Layer for inhomogeneous layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
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
    self.q_lora_rank = getattr(config, "q_lora_rank", 1536)
    self.kv_lora_rank = getattr(config, "kv_lora_rank", 512)
    self.qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 256)
    self.v_head_dim = getattr(config, "v_head_dim", 256)
    self.head_dim = self.qk_nope_head_dim
    self.scaling = self.head_dim**-0.5

    # Query Projections
    self.q_a_proj = linears.DenseGeneral(
        self.emb_dim,
        self.q_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "q_lora"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.q_a_layernorm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.q_b_proj = linears.DenseGeneral(
        self.q_lora_rank,
        self.num_heads * self.qk_nope_head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("q_lora", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Key/Value Projections
    self.kv_a_proj_with_mqa = linears.DenseGeneral(
        self.emb_dim,
        self.kv_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "kv_lora"),
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.kv_a_layernorm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.kv_b_proj = linears.DenseGeneral(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("kv_lora", "heads"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Output projection
    self.o_proj = linears.DenseGeneral(
        self.num_heads * self.v_head_dim,
        self.emb_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=False,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("heads", "embed"),
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array | None = None,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      **kwargs,
  ) -> tuple[Array, None]:
    x = inputs_q
    b, s, _ = x.shape

    # Query: [B, S, Q_LORA] -> [B, S, Q_LORA] -> [B, S, H * D] -> [B, S, H, D]
    q_resid = self.q_a_layernorm(self.q_a_proj(x))
    q = self.q_b_proj(q_resid)
    q = jnp.reshape(q, (b, s, self.num_heads, self.qk_nope_head_dim))

    # Key / Value: [B, S, KV_LORA] -> [B, S, KV_LORA] -> [B, S, H * (D_K + D_V)]
    compressed_kv = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))
    kv_b = self.kv_b_proj(compressed_kv)
    kv_b = jnp.reshape(kv_b, (b, s, self.num_heads, self.qk_nope_head_dim + self.v_head_dim))
    k = kv_b[..., : self.qk_nope_head_dim]
    v = kv_b[..., self.qk_nope_head_dim :]

    # [B, H, S, D]
    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)

    # Scaled dot-product causal attention
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling

    causal_mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool_))
    mask_value = -1e9
    scores = jnp.where(causal_mask[None, None, :, :], scores, mask_value)

    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
    attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    # [B, S, H * D]
    attn_out = jnp.swapaxes(attn_out, 1, 2)
    attn_out = jnp.reshape(attn_out, (b, s, self.num_heads * self.v_head_dim))

    out = self.o_proj(attn_out)
    return out, None


class Glm5NextDenseMLP(nnx.Module):
  """GLM-5.3-Flash Dense MLP with SwiGLU clamping."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      in_features: int,
      intermediate_dim: int,
      rngs: nnx.Rngs,
      quant=None,
      model_mode: str = MODEL_MODE_TRAIN,
  ):
    self.config = config
    self.mesh = mesh
    self.in_features = in_features
    self.intermediate_dim = intermediate_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.swiglu_limit = getattr(config, "swiglu_limit", 10.0)

    self.wi_0 = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )
    self.wi_1 = linears.DenseGeneral(
        in_features_shape=in_features,
        out_features_shape=self.intermediate_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )
    self.wo = linears.DenseGeneral(
        in_features_shape=self.intermediate_dim,
        out_features_shape=in_features,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("mlp", "embed"),
        quant=quant,
        use_bias=False,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        mesh=self.mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: Array,
      **kwargs,
  ) -> Array:
    gate = self.wi_0(inputs)
    up = self.wi_1(inputs)
    gate = jnp.minimum(gate, self.swiglu_limit)
    up = jnp.clip(up, -self.swiglu_limit, self.swiglu_limit)
    act = jax.nn.silu(gate) * up
    return self.wo(act)


class Glm5NextDecoderLayer(nnx.Module):
  """GLM-5.3-Flash Decoder Layer wrapping Attention and MLP/MoE in mHC blocks."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      rngs: nnx.Rngs,
      quant=None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    self.rngs = rngs

    self.input_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    is_sparse_attn = (layer_idx + 1) % config.inhomogeneous_layer_cycle_interval == 0
    if is_sparse_attn:
      self.attention = Glm5NextSparseAttention(
          config=config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )
    else:
      self.attention = Glm5NextAttention(
          config=config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )

    self.is_moe = layer_idx >= getattr(config, "first_num_dense_layers", 3)
    if self.is_moe:
      self.mlp = moe.RoutedAndSharedMoE(
          config=self.config,
          mesh=self.mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed_moe", None),
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = Glm5NextDenseMLP(
          config=config,
          mesh=self.mesh,
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )

    self.attn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        dim=config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )
    self.ffn_hc = mhc.ManifoldConstrainedHyperConnections(
        config=config,
        dim=config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array | None = None,
      decoder_positions: Array | None = None,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> tuple[Array, None]:
    """Forward pass for GLM-5.3-Flash Decoder Layer."""
    x = inputs

    x, _ = self.attn_hc(
        norm_fn=self.input_layernorm,
        branch_fn=self.attention,
        x=x,
        mhc_type=HyperConnectionType.ATTENTION,
    )

    ffn_type = HyperConnectionType.MLP_MOE if self.is_moe else HyperConnectionType.MLP_DENSE
    x, _ = self.ffn_hc(
        norm_fn=self.post_attention_layernorm,
        branch_fn=self.mlp,
        x=x,
        mhc_type=ffn_type,
    )

    return x, None


Glm5NextDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Glm5NextDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
