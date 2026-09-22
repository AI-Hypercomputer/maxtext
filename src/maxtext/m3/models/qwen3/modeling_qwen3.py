# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 dense model in m3 format.

This module provides a fully self-contained implementation of the Qwen3 model
family (dense variant) following the m3 modern MaxText architecture:
- Self-contained NNX components: Linear, RMSNorm, Embed, Attention, MLP, DecoderLayer.
- Completely free of legacy maxtext.layers dependencies (AttentionOp, DenseGeneral,
  Linen RMSNorm, Linen Embed, quantizations).
- Pure NNX module composition with clean parameter paths matching existing checkpoints.
- Uses m3 core RoPE.
- Integrates with MaxEngine for inference KV caching and generation.
- Designed with clear extension points for future MoE and hybrid variants.
"""

from typing import Any, Optional
from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import (
    Config,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.m3.core.kvcache import KVCache
from maxtext.m3.core.rope import apply_rope
from maxtext.utils import max_utils


def _parse_axis_indices(val: Any, default: tuple[int, ...]) -> tuple[int, ...]:
  """Parses axis permutation config values into an integer tuple."""
  if val is None:
    return default
  if isinstance(val, (tuple, list)):
    return tuple(int(x) for x in val)
  if isinstance(val, str):
    return tuple(int(x.strip()) for x in val.split(","))
  return default


class RMSNorm(nnx.Module):
  """Root Mean Square Layer Normalization (pure NNX).

  Computes x / sqrt(mean(x^2) + eps) * scale.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: DType = jnp.bfloat16,
      weight_dtype: DType = jnp.bfloat16,
      kernel_axes: tuple[str | None, ...] = ("norm",),
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes RMSNorm.

    Args:
      num_features: Dimensionality of features to normalize.
      epsilon: Small epsilon for numerical stability.
      dtype: Activation compute dtype.
      weight_dtype: Parameter storage dtype.
      kernel_axes: Logical mesh sharding axes for learned scale.
      rngs: NNX random number generators.
    """
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scale = nnx.Param(
        jnp.ones((num_features,), dtype=weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies root mean square layer normalization across the trailing dimension."""
    x_fp32 = jnp.asarray(x, jnp.float32)
    variance = jnp.mean(lax.square(x_fp32), axis=-1, keepdims=True)
    normed = x_fp32 * lax.rsqrt(variance + self.epsilon)
    scale = jnp.asarray(self.scale[...], jnp.float32)
    return jnp.asarray(normed * scale, self.dtype)


class Linear(nnx.Module):
  """Linear transformation with configurable in/out shapes and logical axis sharding."""

  def __init__(
      self,
      in_features: tuple[int, ...] | int,
      out_features: tuple[int, ...] | int,
      kernel_axes: tuple[str | None, ...],
      dtype: DType = jnp.bfloat16,
      weight_dtype: DType = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes Linear layer.

    Args:
      in_features: Input feature dimension(s) to contract.
      out_features: Output feature dimension(s) to produce.
      kernel_axes: Logical mesh sharding axes for weight tensor.
      dtype: Compute data type for matrix multiplication.
      weight_dtype: Storage data type for learned kernel weights.
      rngs: NNX random number generators.
    """
    self.in_shape = (in_features,) if isinstance(in_features, int) else tuple(in_features)
    self.out_shape = (out_features,) if isinstance(out_features, int) else tuple(out_features)
    self.kernel_shape = self.in_shape + self.out_shape
    self.kernel_axes = kernel_axes
    self.dtype = dtype

    fan_in = int(np.prod(self.in_shape))
    stddev = 1.0 / np.sqrt(fan_in)
    kernel_val = (
        jax.random.truncated_normal(rngs.params(), -2.0, 2.0, self.kernel_shape, dtype=jnp.float32) * stddev
    ).astype(weight_dtype)
    self.kernel = nnx.Param(kernel_val, sharding=kernel_axes)

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies linear transformation by contracting input with kernel weights."""
    in_contract = tuple(range(x.ndim - len(self.in_shape), x.ndim))
    kernel_contract = tuple(range(len(self.in_shape)))
    out = lax.dot_general(
        x.astype(self.dtype),
        self.kernel[...].astype(self.dtype),
        ((in_contract, kernel_contract), ((), ())),
    )
    return out


class Embed(nnx.Module):
  """Token embedding table with tied output projection (pure NNX)."""

  def __init__(
      self,
      num_embeddings: int,
      num_features: int,
      dtype: DType = jnp.bfloat16,
      weight_dtype: DType = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes Embed layer.

    Args:
      num_embeddings: Total vocabulary size.
      num_features: Embedding vector dimension.
      dtype: Output data type for looked up embeddings.
      weight_dtype: Storage data type for the embedding table.
      rngs: NNX random number generators.
    """
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    scale = 1.0 / np.sqrt(num_features)
    emb_val = (
        jax.random.normal(rngs.params(), (num_embeddings, num_features), dtype=jnp.float32) * scale
    ).astype(weight_dtype)
    self.embedding = nnx.Param(emb_val, sharding=("vocab", "embed_vocab"))

  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Looks up embedding vectors for input token IDs."""
    table = self.embedding[...].astype(self.dtype)
    return table[inputs]

  def attend(self, query: jax.Array, attend_dtype: DType = jnp.float32) -> jax.Array:
    """Projects hidden states against the transposed embedding table to produce logits."""
    table = self.embedding[...].astype(attend_dtype)
    return jnp.dot(query.astype(attend_dtype), table.T)


class Qwen3MLP(nnx.Module):
  """Qwen3 feed-forward network with SwiGLU activation.

  Submodule and parameter names match standard checkpoint paths:
  wi_0 (gate), wi_1 (up), wo (down).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes Qwen3MLP feed-forward network.

    Args:
      config: Model configuration specifying dimensions and dtypes.
      mesh: Device mesh used for parallel execution.
      rngs: NNX random number generators.
    """
    self.config = config
    self.mesh = mesh

    self.wi_0 = Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_axes=("embed", "mlp"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.wi_1 = Linear(
        in_features=config.emb_dim,
        out_features=config.mlp_dim,
        kernel_axes=("embed", "mlp"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.wo = Linear(
        in_features=config.mlp_dim,
        out_features=config.emb_dim,
        kernel_axes=("mlp", "embed"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Applies gated SwiGLU feed-forward projection: wo(silu(wi_0(x)) * wi_1(x))."""
    gate = jax.nn.silu(self.wi_0(x))
    up = self.wi_1(x)
    hidden = (gate * up).astype(self.config.dtype)
    return self.wo(hidden)


class Qwen3Attention(nnx.Module):
  """Qwen3 multi-head / grouped-query attention with QK-norm, RoPE, and self-contained attention.

  Submodule and parameter names match checkpoint paths:
  query, key, value, out, query_norm, key_norm.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes Qwen3 Attention module with Q, K, V projections, QK-Norm, and KV Cache.

    Args:
      config: Model configuration specifying dimensions, head counts, and RoPE settings.
      mesh: Device mesh used for parallel execution.
      model_mode: Operational mode ('train', 'prefill', or 'autoregressive').
      rngs: NNX random number generators.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode

    self.num_query_heads = config.num_query_heads
    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim

    # Q, K, V projections
    self.query = Linear(
        in_features=config.emb_dim,
        out_features=(self.num_query_heads, self.head_dim),
        kernel_axes=("embed", "heads", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.key = Linear(
        in_features=config.emb_dim,
        out_features=(self.num_kv_heads, self.head_dim),
        kernel_axes=("embed", "kv_heads", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.value = Linear(
        in_features=config.emb_dim,
        out_features=(self.num_kv_heads, self.head_dim),
        kernel_axes=("embed", "kv_heads", None),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.out = Linear(
        in_features=(self.num_query_heads, self.head_dim),
        out_features=config.emb_dim,
        kernel_axes=("heads", None, "embed"),
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    # QK-Norm
    self.query_norm = RMSNorm(
        num_features=self.head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.key_norm = RMSNorm(
        num_features=self.head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # KV Cache for decode / prefill (MaxEngine integration)
    prefill_order = _parse_axis_indices(config.prefill_cache_axis_order, (1, 2, 0, 3))
    ar_order = _parse_axis_indices(config.ar_cache_axis_order, (1, 2, 0, 3))
    batch_size, _ = max_utils.get_batch_seq_len_for_mode(config, model_mode)

    self.KVCache_0 = (
        KVCache(
            max_prefill_length=config.max_prefill_predict_length,
            max_target_length=config.max_target_length,
            batch=batch_size,
            key_seq_len=1,
            value_seq_len=1,
            key_heads=self.num_kv_heads,
            value_heads=self.num_kv_heads,
            key_head_size=self.head_dim,
            value_head_size=self.head_dim,
            dtype=config.dtype,
            prefill_cache_axis_order=prefill_order,
            ar_cache_axis_order=ar_order,
            use_chunked_prefill=config.use_chunked_prefill,
            model_mode=model_mode,
            rngs=rngs,
        )
        if model_mode != MODEL_MODE_TRAIN
        else None
    )

  def _compute_attention(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      model_mode: str,
      cached_values: Any = None,
      decoder_positions: Optional[jax.Array] = None,
      decoder_segment_ids: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Computes self-contained scaled dot-product attention with GQA and KV caching."""
    b, t, n_q, d = query.shape
    n_kv = key.shape[2]
    g = n_q // n_kv
    query_g = jnp.reshape(query, (b, t, n_kv, g, d))

    def _compute_local_block(q_g, k, v, mask=None):
      """Computes local attention output, max, and sum for scaled dot-product block."""
      weights = jnp.einsum("btkgd,bskd->bkgts", q_g, k)
      if mask is not None:
        weights = jnp.where(mask, weights, -1e10)
      if self.config.float32_logits:
        weights = weights.astype(jnp.float32)
      logits = jnp.reshape(weights, (b, n_q, t, -1))
      local_max = jnp.max(logits, axis=-1, keepdims=True)
      local_exp = jnp.exp(logits - local_max)
      local_sum = jnp.sum(local_exp, axis=-1, keepdims=True)

      local_exp = jnp.reshape(local_exp.astype(v.dtype), (b, n_kv, g, t, -1))
      local_out = jnp.einsum("bkgts,bskd->btkgd", local_exp, v)
      local_out = jnp.reshape(local_out, (b, t, n_q, d))

      local_max = jnp.transpose(local_max, (0, 2, 1, 3))
      local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))
      return local_out, local_max, local_sum

    if model_mode == MODEL_MODE_TRAIN or cached_values is None:
      s = key.shape[1]
      q_pos = jnp.arange(t) if decoder_positions is None else decoder_positions
      k_pos = jnp.arange(s) if decoder_positions is None else decoder_positions
      causal_mask = k_pos[..., None, :] <= q_pos[..., :, None]
      if decoder_segment_ids is not None:
        seg_mask = decoder_segment_ids[..., :, None] == decoder_segment_ids[..., None, :]
        mask = causal_mask & seg_mask
      else:
        mask = causal_mask
      mask = mask[:, None, None, :, :]
      out, _, lsum = _compute_local_block(query_g, key, value, mask)
      return (out / lsum).astype(self.config.dtype)

    prefill_cache, ar_cache = cached_values
    assert prefill_cache is not None
    k_prefill, v_prefill, seg_prefill = prefill_cache

    if model_mode == MODEL_MODE_PREFILL or ar_cache is None:
      q_pos = decoder_positions
      k_pos = decoder_positions
      causal_mask = k_pos[..., None, :] <= q_pos[..., :, None]
      if seg_prefill is not None:
        seg_mask = seg_prefill[..., :, None] == seg_prefill[..., None, :]
        mask = causal_mask & seg_mask
      else:
        mask = causal_mask
      mask = mask[:, None, None, :, :]
      out, _, lsum = _compute_local_block(query_g, k_prefill, v_prefill, mask)
      return (out / lsum).astype(self.config.dtype)

    elif model_mode == MODEL_MODE_AUTOREGRESSIVE or ar_cache is not None:
      # Autoregressive mode: combine prefill cache and ar cache via online softmax
      assert ar_cache is not None, 'ar_cache must not be None in autoregressive mode'
      k_ar, v_ar, seg_ar, lengths = ar_cache
    s_ar = k_ar.shape[1]

    if seg_prefill is not None:
      prefill_mask = (seg_prefill == DECODING_ACTIVE_SEQUENCE_INDICATOR)[:, None, None, None, :]
    else:
      prefill_mask = None
    out_p, max_p, sum_p = _compute_local_block(query_g, k_prefill, v_prefill, prefill_mask)

    ar_pos = jnp.arange(s_ar)[None, :]
    ar_mask = (ar_pos < lengths[:, None])[:, None, None, None, :]
    out_ar, max_ar, sum_ar = _compute_local_block(query_g, k_ar, v_ar, ar_mask)

    global_max = jnp.maximum(max_p, max_ar)
    factor_p = jnp.exp(max_p - global_max)
    factor_ar = jnp.exp(max_ar - global_max)
    global_sum = factor_p * sum_p + factor_ar * sum_ar
    combined = (factor_p / global_sum) * out_p + (factor_ar / global_sum) * out_ar
    return combined.astype(self.config.dtype)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      kv_cache: Any = None,
      **kwargs,
  ) -> tuple[jax.Array, Any]:
    """Executes Qwen3 self-attention forward pass with RoPE and KV caching."""
    query = self.query(inputs)
    key = self.key(inputs)
    value = self.value(inputs)

    # QK-Norm before RoPE
    if self.config.use_qk_norm:
      query = self.query_norm(query)
      key = self.key_norm(key)

    # RoPE
    query = apply_rope(query, decoder_positions, max_timescale=self.config.rope_max_timescale)
    key = apply_rope(key, decoder_positions, max_timescale=self.config.rope_max_timescale)

    # Scale query by 1/sqrt(head_dim)
    query_pre_attn_scalar = self.head_dim**-0.5
    query = query * query_pre_attn_scalar

    # Update KV cache
    cached_values = None
    if model_mode != MODEL_MODE_TRAIN and self.KVCache_0 is not None:
      cached_values = self.KVCache_0(
          key=key,
          value=value,
          decoder_segment_ids=decoder_segment_ids,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
      )

    attn_out = self._compute_attention(
        query=query,
        key=key,
        value=value,
        model_mode=model_mode,
        cached_values=cached_values,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    out = self.out(attn_out)
    return out, kv_cache


class Qwen3DecoderLayer(nnx.Module):
  """Qwen3 Transformer decoder layer (dense).

  Pre-norm architecture with self-attention and MLP residual blocks.
  Leaves clean extension points for MoE or hybrid variants.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes a Qwen3 decoder layer.

    Args:
      config: Model configuration specifying dimensions and layer parameters.
      mesh: Device mesh used for parallel execution.
      model_mode: Operational mode ('train', 'prefill', or 'autoregressive').
      rngs: NNX random number generators.
    """
    self.config = config
    self.mesh = mesh
    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.self_attention = Qwen3Attention(
        config=config,
        mesh=mesh,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.post_self_attention_layer_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.mlp = Qwen3MLP(
        config=config,
        mesh=mesh,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      decoder_positions: Optional[jax.Array] = None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      kv_cache: Any = None,
      **kwargs,
  ) -> tuple[jax.Array, Any]:
    """Applies pre-norm attention and pre-norm MLP blocks with residual connections."""
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    # Pre-attention norm & Self-attention
    normed_attn_in = self.pre_self_attention_layer_norm(inputs)
    attn_out, kv_cache = self.self_attention(
        normed_attn_in,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_cache=kv_cache,
    )
    x = (inputs + attn_out).astype(self.config.dtype)

    # Post-attention norm & MLP
    normed_mlp_in = self.post_self_attention_layer_norm(x)
    mlp_out = self.mlp(normed_mlp_in)
    output = (x + mlp_out).astype(self.config.dtype)

    return output, kv_cache


class Qwen3Decoder(nnx.Module):
  """Qwen3 Decoder stack composing decoder layers with flat attribute naming."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the Qwen3 decoder stack of N layers and final layer normalization.

    Args:
      config: Model configuration specifying layer count, dimensions, and dtypes.
      mesh: Device mesh used for parallel execution.
      model_mode: Operational mode ('train', 'prefill', or 'autoregressive').
      rngs: NNX random number generators.
    """
    self.config = config
    self.mesh = mesh
    self.num_layers = config.num_decoder_layers

    # Flat attribute names: layers_0, layers_1, ...
    for lyr in range(self.num_layers):
      layer = Qwen3DecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          rngs=rngs,
      )
      setattr(self, f"layers_{lyr}", layer)

    self.decoder_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

  def apply_output_head(
      self,
      shared_embedding: Any,
      y: jax.Array,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> jax.Array:
    """Applies final decoder norm and projects hidden states to logits via tied embedding."""
    y = self.decoder_norm(y)
    attend_dtype = jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype
    if hasattr(shared_embedding, "attend"):
      logits = shared_embedding.attend(y, attend_dtype=attend_dtype)
    else:
      emb = shared_embedding.embedding if hasattr(shared_embedding, "embedding") else shared_embedding
      table = emb[...]
      logits = jnp.dot(y.astype(attend_dtype), jnp.asarray(table, attend_dtype).T)

    if self.config.normalize_embedding_logits:
      logits = logits / jnp.sqrt(y.shape[-1])
    if self.config.final_logits_soft_cap:
      logits = logits / self.config.final_logits_soft_cap
      logits = jnp.tanh(logits) * self.config.final_logits_soft_cap
    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

  def __call__(
      self,
      shared_embedding: Any,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: Optional[jax.Array] = None,
      deterministic: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      kv_caches: Optional[list[Any]] = None,
      **kwargs,
  ) -> tuple[jax.Array, jax.Array, Optional[list[Any]]]:
    """Iterates through all decoder layers and projects to logits via tied embeddings."""
    y = shared_embedding(decoder_input_tokens)

    for lyr in range(self.num_layers):
      layer = getattr(self, f"layers_{lyr}")
      kv_cache = kv_caches[lyr] if kv_caches is not None else None
      y, kv_cache = layer(
          y,
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
      )
      if kv_caches is not None:
        kv_caches[lyr] = kv_cache

    logits = self.apply_output_head(
        shared_embedding, y, deterministic=deterministic, model_mode=model_mode
    )
    return logits, y, kv_caches


class Qwen3Model(nnx.Module):
  """Top-level Qwen3 autoregressive language model in m3 format."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
      **kwargs,
  ):
    """Initializes the top-level Qwen3 language model.

    Args:
      config: Model configuration specifying dimensions, layers, and vocabulary.
      mesh: Device mesh used for parallel execution.
      model_mode: Operational mode ('train', 'prefill', or 'autoregressive').
      rngs: NNX random number generators.
      **kwargs: Additional keyword arguments.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode

    self.token_embedder = Embed(
        num_embeddings=config.vocab_size,
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(
        config=config,
        mesh=mesh,
        model_mode=model_mode,
        rngs=rngs,
    )

  def init_cache(self, cache_size: int, batch_size: int, dtype=jnp.float32) -> bool:
    """Initializes cache placeholder for engine compatibility."""
    return True

  def logits_from_hidden_states_for_vocab_tiling(
      self, hidden_states: jax.Array, deterministic: bool, model_mode: str
  ) -> jax.Array:
    """Projects hidden states to logits for vocab-tiled computation."""
    return self.decoder.apply_output_head(
        shared_embedding=self.token_embedder,
        y=hidden_states,
        deterministic=deterministic,
        model_mode=model_mode,
    )

  def apply_prefill(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      **kwargs,
  ) -> tuple[jax.Array, list[Any]]:
    """Performs prompt prefill computation and returns output logits with KV caches."""
    kv_caches = [None] * self.decoder.num_layers
    logits, _, kv_caches = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=kwargs.get("decoder_segment_ids", None),
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_caches=kv_caches,
    )
    return logits, kv_caches

  def __call__(
      self,
      decoder_input_tokens: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray] = None,
      cache: Any = None,
      enable_dropout: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      true_length: Optional[int] = None,
      slot: Optional[int] = None,
      kv_caches: Optional[list[Any]] = None,
      **kwargs,
  ) -> jax.Array:
    """Executes full Qwen3 forward pass from input token IDs to output logits."""
    deterministic = kwargs.get('deterministic', not enable_dropout)
    logits, _, _ = self.decoder(
        shared_embedding=self.token_embedder,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_caches=kv_caches,
    )
    return logits


def create_qwen3_model(
    config: Config,
    mesh: Mesh,
    model_mode: str = MODEL_MODE_TRAIN,
    *,
    rngs: Optional[nnx.Rngs] = None,
    **kwargs,
) -> Qwen3Model:
  """Factory function creating a Qwen3Model instance."""
  if rngs is None:
    rngs = nnx.Rngs(0)
  return Qwen3Model(config, mesh, model_mode=model_mode, rngs=rngs, **kwargs)
