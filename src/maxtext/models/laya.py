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

"""Laya (ModernBERT encoder + RL Decision Head) model definition for MaxText."""

from typing import Any, Optional

from flax import linen as nn
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import Array, Config, MODEL_MODE_TRAIN
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers.embeddings import Embed
from maxtext.layers.initializers import Initializer, nd_dense_init
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


class LayaLayerNorm(nnx.Module):
  """Standard LayerNorm (mean and variance normalization) with optional bias."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      use_bias: bool = False,
      reductions_in_fp32: bool = True,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.use_bias = use_bias
    self.reductions_in_fp32 = reductions_in_fp32
    self.parameter_memory_host_offload = parameter_memory_host_offload

    self.scale = nnx.Param(
        self.scale_init(rngs.params(), (num_features,), self.weight_dtype),
        sharding=self.kernel_axes,
    )
    if self.use_bias:
      self.bias = nnx.Param(
          initializers.default_bias_init(rngs.params(), (num_features,), self.weight_dtype),
          sharding=self.kernel_axes,
      )
    else:
      self.bias = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    """Applies standard layer normalization on the last dimension of the input."""
    orig_dtype = x.dtype
    x_f32 = jnp.asarray(x, jnp.float32) if self.reductions_in_fp32 else x
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
    normed = (x_f32 - mean) * lax.rsqrt(var + self.epsilon)

    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    scale_f32 = jnp.asarray(scale, jnp.float32) if self.reductions_in_fp32 else jnp.asarray(scale, self.dtype)
    output = normed * scale_f32

    if self.bias is not None:
      bias = self.bias[...]
      if self.parameter_memory_host_offload:
        bias = jax.device_put(bias, max_utils.device_space())
      bias_f32 = jnp.asarray(bias, jnp.float32) if self.reductions_in_fp32 else jnp.asarray(bias, self.dtype)
      output = output + bias_f32

    out_dtype = self.dtype if self.dtype is not None else orig_dtype
    output = jnp.asarray(output, out_dtype)
    if out_sharding is not None:
      output = lax.with_sharding_constraint(output, out_sharding)
    return output


def _rotate_half(x: Array) -> Array:
  """Rotates half the hidden dims of the input (HuggingFace ModernBert rotate_half)."""
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate([-x2, x1], axis=-1)


def apply_modernbert_rope(
    q: Array,
    k: Array,
    positions: Array,
    head_dim: int,
    rope_theta: float,
) -> tuple[Array, Array]:
  """Applies ModernBERT rotary position embeddings to query and key tensors.

  Args:
    q: Query tensor of shape [batch, seq_len, num_heads, head_dim].
    k: Key tensor of shape [batch, seq_len, num_heads, head_dim].
    positions: Position IDs of shape [batch, seq_len].
    head_dim: Per-head dimension (64 for ModernBERT-large).
    rope_theta: RoPE theta base (160000.0 for global/multilingual, 10000.0 for local).

  Returns:
    Tuple of (q_embed, k_embed) in the original dtype of q and k.
  """
  orig_dtype = q.dtype
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)))
  # positions: [B, S] -> freqs: [B, S, D/2]
  freqs = positions[:, :, None].astype(jnp.float32) * inv_freq[None, None, :]
  emb = jnp.concatenate([freqs, freqs], axis=-1)  # [B, S, D]
  cos = jnp.cos(emb)[:, :, None, :]  # [B, S, 1, D]
  sin = jnp.sin(emb)[:, :, None, :]  # [B, S, 1, D]

  q_f32 = q.astype(jnp.float32)
  k_f32 = k.astype(jnp.float32)
  q_embed = (q_f32 * cos) + (_rotate_half(q_f32) * sin)
  k_embed = (k_f32 * cos) + (_rotate_half(k_f32) * sin)
  return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class LayaAttention(nnx.Module):
  """ModernBERT bidirectional self-attention (global or sliding-window)."""

  def __init__(
      self,
      config: Config,
      layer_idx: int = 0,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.num_heads = config.num_query_heads
    self.head_dim = config.head_dim
    self.emb_dim = config.emb_dim
    self.is_global_attn = layer_idx % 3 == 0
    self.sliding_window = int(getattr(config, "sliding_window_size", 128)) // 2

    global_theta = float(getattr(config, "rope_max_timescale", 160000.0))
    local_theta = float(getattr(config, "local_rope_max_timescale", global_theta))
    if local_theta <= 0.0:
      local_theta = global_theta
    self.rope_theta = global_theta if self.is_global_attn else local_theta

    self.Wqkv = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=3 * self.emb_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "mlp"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )
    self.Wo = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.emb_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("mlp", "embed"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Optional[Array] = None,
      decoder_positions: Optional[Array] = None,
      deterministic: bool = False,
  ) -> Array:
    batch_size, seq_len, _ = inputs.shape
    if decoder_positions is None:
      decoder_positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))

    qkv = self.Wqkv(inputs)  # [B, S, 3 * emb_dim]
    qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]

    q, k = apply_modernbert_rope(q, k, decoder_positions, self.head_dim, self.rope_theta)

    precision = lax.Precision(self.config.matmul_precision)
    scale = self.head_dim**-0.5
    attn_weights = (
        jnp.einsum("bshd,bthd->bhst", q.astype(jnp.float32), k.astype(jnp.float32), precision=precision) * scale
    )

    # Build bidirectional mask: valid keys must have non-zero segment ID and match query segment ID
    if decoder_segment_ids is not None:
      valid_q = decoder_segment_ids[:, :, None] != 0
      valid_k = decoder_segment_ids[:, None, :] != 0
      same_seg = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      # If all queries are masked in a row, avoid NaN in softmax by allowing self-attention on diagonal
      mask = valid_k & (same_seg | ~valid_q)
    else:
      mask = jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool_)

    if not self.is_global_attn:
      q_pos = decoder_positions[:, :, None]
      k_pos = decoder_positions[:, None, :]
      window_mask = jnp.abs(q_pos - k_pos) <= self.sliding_window
      mask = mask & window_mask

    # Expand mask for heads: [B, 1, S, S]
    mask = mask[:, None, :, :]
    # Ensure at least one valid position per query row to prevent softmax NaN on fully padded positions
    row_has_valid = jnp.any(mask, axis=-1, keepdims=True)
    eye = jnp.eye(seq_len, dtype=jnp.bool_)[None, None, :, :]
    safe_mask = jnp.where(row_has_valid, mask, eye)

    mask_val = jnp.finfo(jnp.float32).min
    attn_weights = jnp.where(safe_mask, attn_weights, mask_val)
    attn_probs = jax.nn.softmax(attn_weights, axis=-1).astype(inputs.dtype)

    attn_out = jnp.einsum("bhst,bthd->bshd", attn_probs, v.astype(inputs.dtype), precision=precision)
    attn_out = jnp.reshape(attn_out, (batch_size, seq_len, self.emb_dim))
    return self.Wo(attn_out)


class LayaMLP(nnx.Module):
  """ModernBERT GeGLU MLP block: Wi projects to 2 * mlp_dim, splits into (input, gate),

  applies exact GELU(input) * gate, then Wo.
  """

  def __init__(
      self,
      config: Config,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.emb_dim = config.emb_dim
    self.mlp_dim = config.mlp_dim

    self.Wi = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=2 * self.mlp_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("embed", "mlp"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )
    self.Wo = DenseGeneral(
        in_features_shape=self.mlp_dim,
        out_features_shape=self.emb_dim,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
        kernel_axes=("mlp", "embed"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )

  def __call__(self, inputs: Array, deterministic: bool = False) -> Array:
    wi_out = self.Wi(inputs)
    inp, gate = jnp.split(wi_out, 2, axis=-1)
    hidden = jax.nn.gelu(inp.astype(jnp.float32), approximate=False).astype(inputs.dtype) * gate
    return self.Wo(hidden)


class LayaDecoderLayer(nnx.Module):
  """ModernBERT encoder layer adapted to the MaxText NNX decoder layer interface."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: Optional[Quant] = None,
      layer_idx: int = 0,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx

    # Layer 0 in ModernBERT has nn.Identity() for attn_norm (no parameters in checkpoint)
    if layer_idx > 0:
      self.attn_norm = LayaLayerNorm(
          num_features=config.emb_dim,
          epsilon=config.normalization_layer_epsilon,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          use_bias=False,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          rngs=rngs,
      )
    else:
      self.attn_norm = None

    self.attn = LayaAttention(config=config, layer_idx=layer_idx, rngs=rngs)

    self.mlp_norm = LayaLayerNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        use_bias=False,
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=rngs,
    )
    self.mlp = LayaMLP(config=config, rngs=rngs)

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Optional[Array] = None,
      decoder_positions: Optional[Array] = None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      page_state: Any = None,
      slot: Optional[int] = None,
      kv_cache: Optional[Any] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> tuple[Array, Optional[Any]]:
    normed_attn_in = self.attn_norm(inputs) if self.attn_norm is not None else inputs
    attn_out = self.attn(
        normed_attn_in,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        deterministic=deterministic,
    )
    hidden_states = inputs + attn_out
    mlp_out = self.mlp(self.mlp_norm(hidden_states), deterministic=deterministic)
    output = hidden_states + mlp_out
    if self.config.scan_layers:
      return output, None
    return output, kv_cache


LayaDecoderLayerToLinen = nnx_wrappers.to_linen_class(LayaDecoderLayer)


class LayaHeadTransformerLayer(nnx.Module):
  """PyTorch nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, activation=relu, norm_first=True)."""

  def __init__(
      self,
      config: Config,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.emb_dim = config.emb_dim
    self.num_heads = config.num_query_heads
    self.head_dim = config.head_dim
    self.ffn_dim = 4 * config.emb_dim  # 4096 for d=1024

    self.norm1 = LayaLayerNorm(
        num_features=self.emb_dim,
        epsilon=1e-5,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        use_bias=True,
        rngs=rngs,
    )
    self.in_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=3 * self.emb_dim,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.out_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.emb_dim,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.norm2 = LayaLayerNorm(
        num_features=self.emb_dim,
        epsilon=1e-5,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        use_bias=True,
        rngs=rngs,
    )
    self.linear1 = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.ffn_dim,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.linear2 = DenseGeneral(
        in_features_shape=self.ffn_dim,
        out_features_shape=self.emb_dim,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

  def __call__(self, x: Array, attention_mask: Optional[Array] = None) -> Array:
    batch_size, seq_len, _ = x.shape
    precision = lax.Precision(self.config.matmul_precision)
    normed = self.norm1(x)
    qkv = self.in_proj(normed)  # [B, S, 3 * D]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = jnp.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
    k = jnp.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
    v = jnp.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

    scale = self.head_dim**-0.5
    attn_weights = (
        jnp.einsum("bshd,bthd->bhst", q.astype(jnp.float32), k.astype(jnp.float32), precision=precision) * scale
    )
    if attention_mask is not None:
      valid_k = (attention_mask != 0)[:, None, None, :]  # [B, 1, 1, S]
      row_has_valid = jnp.any(valid_k, axis=-1, keepdims=True)
      eye = jnp.eye(seq_len, dtype=jnp.bool_)[None, None, :, :]
      safe_mask = jnp.where(row_has_valid, valid_k, eye)
      attn_weights = jnp.where(safe_mask, attn_weights, jnp.finfo(jnp.float32).min)
    attn_probs = jax.nn.softmax(attn_weights, axis=-1).astype(x.dtype)
    attn_out = jnp.einsum("bhst,bthd->bshd", attn_probs, v.astype(x.dtype), precision=precision)
    attn_out = jnp.reshape(attn_out, (batch_size, seq_len, self.emb_dim))
    x = x + self.out_proj(attn_out)

    ff_out = self.linear2(jax.nn.relu(self.linear1(self.norm2(x))))
    return x + ff_out


class LayaDecisionHead(nnx.Module):
  """Laya DecisionModel head on top of ModernBERT encoder outputs."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.emb_dim = config.emb_dim

    self.type_emb = Embed(
        num_embeddings=3,
        num_features=self.emb_dim,
        dtype=config.dtype,
        attend_dtype=jnp.float32,
        embedding_init=nn.initializers.normal(stddev=0.02),
        config=config,
        mesh=mesh,
        rngs=rngs,
    )
    self.head_layers_0 = LayaHeadTransformerLayer(config=config, rngs=rngs)
    self.head_layers_1 = LayaHeadTransformerLayer(config=config, rngs=rngs)

    self.scorer_0 = LayaLayerNorm(
        num_features=self.emb_dim,
        epsilon=1e-5,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        use_bias=True,
        rngs=rngs,
    )
    self.scorer_1 = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.emb_dim,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.scorer_3 = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=1,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, None),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    self.act_head_0 = DenseGeneral(
        in_features_shape=self.emb_dim + 4,
        out_features_shape=256,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, None),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )
    self.act_head_2 = DenseGeneral(
        in_features_shape=256,
        out_features_shape=2,
        use_bias=True,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, None),
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=rngs,
    )

    self.temperature = nnx.Param(
        jnp.ones((3,), dtype=config.weight_dtype),
        sharding=(None,),
    )

  def __call__(
      self,
      enc_hidden_states: Array,
      attention_mask: Array,
      qtype: Array,
      marker_pos: Array,
      marker_mask: Array,
      feat: Optional[Array] = None,
  ) -> tuple[Array, Array]:
    """Runs the Laya DecisionModel head.

    Args:
      enc_hidden_states: Final normalized encoder hidden states [B, S, D].
      attention_mask: Attention mask [B, S] (1 for valid tokens, 0 for padding).
      qtype: Question type indices [B] in {0, 1, 2}.
      marker_pos: Option marker token positions [B, K_max].
      marker_mask: Boolean mask for valid option markers [B, K_max].
      feat: Optional precomputed scalar option statistics [B, 4]. When None,
        computed from opt_logits and marker_mask identically to DecisionModel.forward.

    Returns:
      Tuple of (opt_logits [B, K_max], act_logits [B, 2]).
    """
    type_vec = self.type_emb(qtype.astype(jnp.int32))[:, None, :]  # [B, 1, D]
    h = enc_hidden_states + type_vec
    h = self.head_layers_0(h, attention_mask=attention_mask)
    h = self.head_layers_1(h, attention_mask=attention_mask)

    # Gather marker hidden states: marker_pos is [B, K_max] -> m is [B, K_max, D]
    idx = jnp.clip(marker_pos, 0, h.shape[1] - 1)[:, :, None]
    m = jnp.take_along_axis(h, idx, axis=1)

    s = self.scorer_0(m)
    s = self.scorer_1(s)
    s = jax.nn.gelu(s.astype(jnp.float32), approximate=False).astype(h.dtype)
    opt_logits = jnp.squeeze(self.scorer_3(s), axis=-1).astype(jnp.float32)  # [B, K_max]
    opt_logits = jnp.where(marker_mask.astype(jnp.bool_), opt_logits, -1e4)

    if feat is None:
      p = jax.nn.softmax(jax.lax.stop_gradient(opt_logits), axis=-1)
      k = jnp.maximum(jnp.sum(marker_mask.astype(jnp.float32), axis=-1), 2.0)
      ent = -jnp.sum(p * jnp.log(jnp.maximum(p, 1e-9)), axis=-1) / jnp.log(k)
      if p.shape[-1] >= 2:
        top2, _ = jax.lax.top_k(p, 2)
      else:
        top1, _ = jax.lax.top_k(p, 1)
        top2 = jnp.concatenate([top1, jnp.zeros_like(top1)], axis=-1)
      feat = jnp.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], axis=-1)

    cls = h[:, 0, :].astype(jnp.float32)  # [B, D]
    act_in = jnp.concatenate([cls, feat.astype(jnp.float32)], axis=-1).astype(h.dtype)  # [B, D + 4]
    act_h = self.act_head_0(act_in)
    act_h = jax.nn.gelu(act_h.astype(jnp.float32), approximate=False).astype(h.dtype)
    act_logits = self.act_head_2(act_h).astype(jnp.float32)  # [B, 2]
    return opt_logits, act_logits
