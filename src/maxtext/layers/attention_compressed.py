#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Compressed Attention Layer (DeepSeek-V4) - Custom Implementation."""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx

from maxtext.common.common_types import (
    Array,
    Config,
    DType,
    MODEL_MODE_TRAIN,
    AttentionType,
    DEFAULT_MASK_VALUE,
)

# Surgically import and reuse our custom Phase 1 primitives
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.linears import DenseGeneral, DeepSeekV4GroupedLinear
from maxtext.layers.normalizations import RMSNorm


def csa_overlap_pooling(
    hidden_states: Array,
    kv_proj: Any,
    gate_proj: Any,
    position_bias: Array,
    kv_norm: Any,
    compress_rate: int,
    head_dim: int,
) -> Array:
  """Stateless overlapping Ca/Cb pooling shared by the Indexer and CSA Compressor.

  Implements the overlapping Ca/Cb pooling logic. It splits the projected states
  into two halves (Ca and Cb), shifts the first half forward by one window, and
  concatenates them to form overlapping windows over which softmax gating is applied.
  """
  batch_size, seq_len, _ = hidden_states.shape

  # Project key/value and gate states
  kv = kv_proj(hidden_states)
  gate = gate_proj(hidden_states)

  usable = (seq_len // compress_rate) * compress_rate
  chunk_kv = kv[:, :usable]
  chunk_gate = gate[:, :usable]

  # Return zero tensor if there are no full windows available for pooling
  if chunk_kv.shape[1] == 0:
    return jnp.zeros((batch_size, 0, head_dim), dtype=hidden_states.dtype)

  n_windows = chunk_kv.shape[1] // compress_rate

  # Reshape flat sequence into discrete compression windows
  # -> [batch, n_windows, compress_rate, 2 * head_dim]
  chunk_kv = chunk_kv.reshape((batch_size, n_windows, compress_rate, 2 * head_dim))
  chunk_gate = chunk_gate.reshape((batch_size, n_windows, compress_rate, 2 * head_dim)) + position_bias

  # Overlap construction:
  # Ca (first head_dim) slice represents contribution to the next window.
  # Cb (last head_dim) slice represents contribution to the current window.
  new_kv = jnp.zeros((batch_size, n_windows, 2 * compress_rate, head_dim), dtype=chunk_kv.dtype)
  new_gate = jnp.full((batch_size, n_windows, 2 * compress_rate, head_dim), -jnp.inf, dtype=chunk_gate.dtype)

  # Fill current window Cb slice
  new_kv = new_kv.at[:, :, compress_rate:].set(chunk_kv[..., head_dim:])
  new_gate = new_gate.at[:, :, compress_rate:].set(chunk_gate[..., head_dim:])

  # Shift Ca slice forward from the previous window
  if n_windows > 1:
    new_kv = new_kv.at[:, 1:, :compress_rate].set(chunk_kv[:, :-1, :, :head_dim])
    new_gate = new_gate.at[:, 1:, :compress_rate].set(chunk_gate[:, :-1, :, :head_dim])

  # Compute gate-weighted softmax pooling
  gate_weights = jax.nn.softmax(new_gate, axis=2)
  compressed = kv_norm(jnp.sum(new_kv * gate_weights, axis=2))
  return compressed


class DeepseekV4Indexer(nnx.Module):
  """Stateless JAX/NNX Lightning Indexer (DeepSeek-V4 paper §2.3.1, eqs. 13-17)."""

  def __init__(
      self,
      config: Config,
      rotary_embedding: DeepSeekV4RotaryEmbedding,
      compress_rate: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_rate = compress_rate
    self.index_n_heads = config.indexer_n_heads
    self.index_head_dim = config.indexer_head_dim
    self.index_topk = config.indexer_topk
    self.softmax_scale = self.index_head_dim ** -0.5
    self.weights_scaling = self.index_n_heads ** -0.5
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    # Projections for the overlapping window compressor
    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    # Position bias for softmax pooling
    self.position_bias = nnx.Param(
        jnp.zeros((self.compress_rate, 2 * self.index_head_dim), dtype=self.weight_dtype)
    )

    self.kv_norm = RMSNorm(
        num_features=self.index_head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Low-rank query projection inside the Indexer
    self.q_proj = DenseGeneral(
        in_features_shape=config.q_lora_rank,
        out_features_shape=self.index_n_heads * self.index_head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    # Project hidden states to get head-importance weights
    self.weights_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.index_n_heads,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    # REUSE our custom Phase 1 RoPE!
    self.rotary_emb = rotary_embedding

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
  ) -> Array:
    batch_size, seq_len, _ = hidden_states.shape

    # 1. Overlap pooling & compression
    compressed = csa_overlap_pooling(
        hidden_states,
        self.kv_proj,
        self.gate_proj,
        self.position_bias.value,
        self.kv_norm,
        self.compress_rate,
        self.index_head_dim,
    )
    compressed_len = compressed.shape[1]

    # 2. Apply RoPE to compressed keys/values
    if compressed_len > 0:
      first_window_position = position_ids[:, 0:1]
      positions = jnp.arange(compressed_len) * self.compress_rate + first_window_position
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
    else:
      return jnp.zeros((batch_size, seq_len, min(self.index_topk, compressed_len)), dtype=jnp.int32)

    # Broadcast compressed representations across all indexer heads
    compressed_kv = jnp.expand_dims(compressed, axis=1)
    compressed_kv = jnp.broadcast_to(
        compressed_kv,
        (batch_size, self.index_n_heads, compressed_len, self.index_head_dim),
    )

    # 3. Project & apply RoPE to queries
    q = self.q_proj(q_latent).reshape((batch_size, seq_len, self.index_n_heads, self.index_head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))
    q = self.rotary_emb(q, position_ids, unsqueeze_dim=1)

    q = q.astype(jnp.float32)
    compressed_kv = compressed_kv.astype(jnp.float32)

    # 4. Compute dot-product scores: [Batch, Heads, SeqLen, n_windows]
    scores = jnp.einsum("bhsd,bhwd->bhsw", q, compressed_kv)
    scores = jax.nn.relu(scores) * self.softmax_scale

    # Compute head routing weights: [Batch, SeqLen, Heads]
    weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling

    # Combine scores across heads: [Batch, SeqLen, n_windows]
    index_scores = jnp.einsum("bhsw,bsh->bsw", scores, weights)

    k = min(self.index_topk, compressed_len)

    # 5. Causal window masking (prevent attending to future windows)
    causal_threshold = (position_ids + 1) // self.compress_rate
    entry_indices = jnp.arange(compressed_len)
    future_mask = entry_indices[None, None, :] >= jnp.expand_dims(causal_threshold, axis=-1)
    index_scores = jnp.where(future_mask, jnp.full_like(index_scores, -jnp.inf), index_scores)

    # Apply segment attention mask if present
    if attention_mask is not None:
      index_scores += attention_mask[:, :, :compressed_len]

    # Retrieve the Top-K highest-scoring block indices
    top_k_indices = jax.lax.top_k(index_scores, k)[1]

    # Invalidate future indices
    invalid = top_k_indices >= jnp.expand_dims(causal_threshold, axis=-1)
    top_k_indices = jnp.where(invalid, jnp.full_like(top_k_indices, -1), top_k_indices)

    return top_k_indices


class DeepseekV4CSACompressor(nnx.Module):
  """CSA Compressor (DeepSeek-V4 paper §2.3.1).

  Compresses every `compress_rate` source tokens using softmax-gated overlap pooling,
  and invokes the Lightning Indexer to return the Top-K active block indices and
  the corresponding sparse block bias mask.
  """

  def __init__(
      self,
      config: Config,
      rotary_embedding: DeepSeekV4RotaryEmbedding,
      compress_rate: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_rate = compress_rate
    self.head_dim = config.head_dim
    self.index_topk = config.indexer_topk
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    # Dense projections for Ca/Cb pooling
    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    self.position_bias = nnx.Param(
        jnp.zeros((self.compress_rate, 2 * self.head_dim), dtype=self.weight_dtype)
    )

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # The Indexer primitive
    self.indexer = DeepseekV4Indexer(config, rotary_embedding, compress_rate=compress_rate, rngs=rngs)

    # REUSE our custom Phase 1 RoPE!
    self.rotary_emb = rotary_embedding

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
  ) -> Tuple[Array, Array]:
    batch_size, seq_len, _ = hidden_states.shape

    # 1. Overlap pooling & compression
    compressed = csa_overlap_pooling(
        hidden_states,
        self.kv_proj,
        self.gate_proj,
        self.position_bias.value,
        self.kv_norm,
        self.compress_rate,
        self.head_dim,
    )
    compressed_len = compressed.shape[1]

    # 2. Apply RoPE to compressed states
    if compressed_len > 0:
      first_window_position = position_ids[:, 0:1]
      positions = jnp.arange(compressed_len) * self.compress_rate + first_window_position
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
    else:
      compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=hidden_states.dtype)

    compressed_kv = jnp.expand_dims(compressed, axis=1)  # [B, 1, n_windows, D]

    # 3. Invoke the indexer to get active block selections
    top_k_indices = self.indexer(hidden_states, q_latent, position_ids, attention_mask)

    # 4. No-Gather Sparse Block Bias Mask Generation
    # Construct a mask of shape [B, 1, seq_len, n_windows] containing 0.0 at
    # selected indices and -inf elsewhere.
    valid = top_k_indices >= 0
    safe_indices = jnp.where(valid, top_k_indices, jnp.full_like(top_k_indices, -1))

    # Broadcast indices for broadcasting matching
    # safe_indices: [B, S, K]
    w_indices = jnp.arange(compressed_len)
    # selected: [B, S, n_windows]
    selected = jnp.any(jnp.expand_dims(safe_indices, axis=-1) == w_indices, axis=2)
    selected = jnp.expand_dims(selected, axis=1)  # [B, 1, S, n_windows]

    block_bias = jnp.where(selected, 0.0, -jnp.inf)
    return compressed_kv, block_bias


class DeepseekV4HCACompressor(nnx.Module):
  """HCA Compressor (DeepSeek-V4 paper §2.3.2).

  Compresses every `compress_rate` source tokens using softmax-gated non-overlapping pooling,
  and generates a causal mask over all past heavily compressed blocks.
  """

  def __init__(
      self,
      config: Config,
      rotary_embedding: DeepSeekV4RotaryEmbedding,
      compress_rate: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.compress_rate = compress_rate
    self.head_dim = config.head_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    # Dense projections for closed window pooling
    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    self.position_bias = nnx.Param(
        jnp.zeros((self.compress_rate, self.head_dim), dtype=self.weight_dtype)
    )

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # REUSE our custom Phase 1 RoPE!
    self.rotary_emb = rotary_embedding

  def __call__(
      self,
      hidden_states: Array,
      position_ids: Array,
  ) -> Tuple[Array, Array]:
    batch_size, seq_len, _ = hidden_states.shape

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    usable = (seq_len // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]

    if chunk_kv.shape[1] == 0:
      compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=hidden_states.dtype)
    else:
      n_windows = chunk_kv.shape[1] // self.compress_rate

      # Reshape to non-overlapping windows: [B, n_windows, compress_rate, D]
      chunk_kv = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, self.head_dim))
      chunk_gate = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, self.head_dim)) + self.position_bias.value

      gate_weights = jax.nn.softmax(chunk_gate, axis=2)
      compressed = self.kv_norm(jnp.sum(chunk_kv * gate_weights, axis=2))

      first_window_position = position_ids[:, 0:1]
      positions = jnp.arange(n_windows) * self.compress_rate + first_window_position
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)

    compressed_kv = jnp.expand_dims(compressed, axis=1)  # [B, 1, n_windows, D]
    compressed_len = compressed_kv.shape[2]

    # Generate causal block mask: [B, 1, seq_len, n_windows]
    causal_threshold = (position_ids + 1) // self.compress_rate
    entry_indices = jnp.arange(compressed_len)
    future_mask = entry_indices[None, None, :] >= jnp.expand_dims(causal_threshold, axis=-1)

    block_bias = jnp.where(future_mask, -jnp.inf, 0.0)
    block_bias = jnp.expand_dims(block_bias, axis=1)  # [B, 1, seq_len, n_windows]

    return compressed_kv, block_bias


class DeepseekV4HyperHead(nnx.Module):
  """Model exit parallel residual stream collapse layer (DeepSeek-V4 paper §2.2)."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.hc_mult = config.hc_mult
    self.hidden_size = config.emb_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.input_norm = RMSNorm(
        num_features=self.hidden_size,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # Projects flattened streams to stream weights
    self.weights_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.hc_mult,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

  def __call__(self, hidden_states: Array) -> Array:
    # hidden_states shape: [Batch, SeqLen, hc_mult, hidden_size]
    batch_size, seq_len, hc_mult, hidden_size = hidden_states.shape

    # Average streams to compute collapse weight features
    # [B, S, hc_mult, D] -> [B, S, D]
    mean_stream = jnp.mean(hidden_states, axis=2)
    norm_mean = self.input_norm(mean_stream)

    # Project to collapse weights: [B, S, hc_mult]
    # Sigmoid-activated + small epsilon to guarantee non-zero weights
    collapse_weights = jax.nn.sigmoid(self.weights_proj(norm_mean)) + 1e-6

    # Normalize weights along stream dimension: [B, S, hc_mult]
    collapse_weights = collapse_weights / jnp.sum(collapse_weights, axis=-1, keepdims=True)

    # Collapse parallel streams via a weighted sum
    # [B, S, hc_mult, D] * [B, S, hc_mult, 1] -> sum along hc_mult -> [B, S, D]
    collapsed = jnp.sum(hidden_states * jnp.expand_dims(collapse_weights, axis=-1), axis=2)
    return collapsed


def unweighted_rms_norm(x: Array, epsilon: float = 1.0e-6) -> Array:
  """Stateless unweighted RMSNorm used by DeepSeek-V4 to stabilize Q-projections."""
  variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
  return (x * jax.lax.rsqrt(variance + epsilon)).astype(x.dtype)


class CompressedAttention(nnx.Module):
  """Unified Custom DeepSeek-V4 Attention block (Sliding, CSA, and HCA)."""

  def __init__(
      self,
      config: Config,
      compress_ratio: int,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: Tuple[int, int, int],
      inputs_kv_shape: Tuple[int, int, int],
      q_lora_rank: int,
      sliding_window_size: int,
      *,
      rngs: nnx.Rngs,
      **kwargs,
  ):
    self.config = config
    self.compress_ratio = compress_ratio

    # Map compress_ratio to layer_type
    if compress_ratio == 0:
      self.layer_type = "sliding_attention"
    elif compress_ratio == config.compress_ratios[1]: # Use compress_ratios list!
      self.layer_type = "compressed_sparse_attention"
    elif compress_ratio == config.compress_ratios[2]: # Use compress_ratios list!
      self.layer_type = "heavily_compressed_attention"
    else:
      # Direct fallback based on common defaults if ratios list doesn't match
      if compress_ratio == 4:
        self.layer_type = "compressed_sparse_attention"
      elif compress_ratio == 8 or compress_ratio == 128:
        self.layer_type = "heavily_compressed_attention"
      else:
        raise ValueError(f"Invalid compress_ratio: {compress_ratio}")

    self.hidden_size = config.emb_dim
    self.num_heads = num_query_heads
    self.head_dim = head_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    # Projection ranks
    self.q_lora_rank = q_lora_rank

    # 1. Rotary Embeddings
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=config.partial_rotary_factor,
        rope_theta=config.rope_max_timescale,
        dtype=self.dtype,
    )

    # 2. Query low-rank projections (Q-LoRA) - RENAME to wq_a and wq_b!
    self.wq_a = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.q_lora_rank,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.q_norm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=self.num_heads * self.head_dim,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    # 3. Key/Value projections (Single direct projection to head_dim, MQA layout!)
    # We specify out_features_shape as (num_kv_heads, head_dim) to build a 3D kernel
    # of shape [hidden_size, num_kv_heads, head_dim], matching weight copying shapes perfectly!
    self.wkv = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=(num_kv_heads, self.head_dim),
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.kv_norm = RMSNorm(
        num_features=self.head_dim, # Normalizes along head_dim!
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )

    # 4. Long-range historical compressors
    # Flax NNX Pro-Tip: Only define the compressor we actually instantiate!
    if self.layer_type == "compressed_sparse_attention":
      self.csa_compressor = DeepseekV4CSACompressor(config, self.rotary_emb, compress_rate=self.compress_ratio, rngs=rngs)
    elif self.layer_type == "heavily_compressed_attention":
      self.hca_compressor = DeepseekV4HCACompressor(config, self.rotary_emb, compress_rate=self.compress_ratio, rngs=rngs)

    # 5. Attention Sinks - RENAME to sinks!
    self.sinks = nnx.Param(jnp.zeros((1, self.num_heads, 1, 1), dtype=self.dtype))

    # 6. Grouped Output Projections
    self.o_groups = config.o_groups
    self.o_lora_rank = config.o_lora_rank

    # Instantiate using correct DeepSeekV4GroupedLinear parameter names!
    self.o_a_proj = DeepSeekV4GroupedLinear(
        in_features_per_group=(self.num_heads * self.head_dim) // self.o_groups,
        out_features=self.o_groups * self.o_lora_rank,
        n_groups=self.o_groups,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )
    self.o_b_proj = DenseGeneral(
        in_features_shape=self.o_groups * self.o_lora_rank,
        out_features_shape=self.hidden_size,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        rngs=rngs,
    )

    self.softmax_scale = self.head_dim ** -0.5

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      decoder_segment_ids: Array,
      inputs_positions: Array,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> Array:
    batch_size, seq_len, _ = inputs_q.shape
    hidden_states = inputs_q
    position_ids = inputs_positions
    positions_for_rope = inputs_positions

    # 1. Project Queries (Q-LoRA)
    q_latent = self.q_norm(self.wq_a(hidden_states))
    q = self.wq_b(q_latent).reshape((batch_size, seq_len, self.num_heads, self.head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, S, D]

    # Apply unweighted RMSNorm to query before RoPE!
    q = unweighted_rms_norm(q, epsilon=self.config.normalization_layer_epsilon)

    # Apply our custom Phase 1 RoPE to queries
    q = self.rotary_emb(q, positions_for_rope, unsqueeze_dim=1)

    # 2. Project Keys/Values (Single direct projection to head_dim, MQA layout!)
    # wkv returns shape [B, S, num_kv_heads, head_dim] (e.g. [B, S, 1, D])
    kv = self.wkv(inputs_kv)
    kv_normed = self.kv_norm(kv)
    
    # Transpose to [B, num_kv_heads, S, head_dim] (e.g. [B, 1, S, D]) to align head axis
    kv_normed = jnp.transpose(kv_normed, (0, 2, 1, 3))

    # Apply our custom Phase 1 RoPE to both key and value states (Triple RoPE layout!)
    k = self.rotary_emb(kv_normed, positions_for_rope, unsqueeze_dim=1)
    v = self.rotary_emb(kv_normed, positions_for_rope, unsqueeze_dim=1)

    # Broadcast Key/Value head axes from MQA layout [B, 1, S, D] to full attention shape [B, H, S, D]
    # This must be done BEFORE doing any sequence concatenations or attention logits dot-products!
    k = jnp.broadcast_to(k, (batch_size, self.num_heads, seq_len, self.head_dim))
    v = jnp.broadcast_to(v, (batch_size, self.num_heads, seq_len, self.head_dim))

    # 3. Build document packing segment masks
    segment_mask_sliding = None
    compressed_segment_mask = None
    if decoder_segment_ids is not None:
      # Segment mask for sliding window
      segment_mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      segment_mask_sliding = jnp.expand_dims(jnp.where(segment_mask, 0.0, -1e9), axis=1) # [B, 1, S, S]
      
      # Downsampled segment mask for the compressed dimension
      if self.compress_ratio > 0:
        compressed_segment_mask = jnp.where(segment_mask, 0.0, -1e9)[:, :, ::self.compress_ratio]

    # 4. Long-range historical compression (CSA/HCA)
    # Check compressor existence dynamically using hasattr
    if hasattr(self, "csa_compressor") and self.csa_compressor is not None:
      compressed_kv, block_bias = self.csa_compressor(hidden_states, q_latent, position_ids, compressed_segment_mask)
      k_compressed = jnp.broadcast_to(compressed_kv, (batch_size, self.num_heads, compressed_kv.shape[2], self.head_dim))
      v_compressed = jnp.broadcast_to(compressed_kv, (batch_size, self.num_heads, compressed_kv.shape[2], self.head_dim))
      k_combined = jnp.concatenate([k, k_compressed], axis=2)
      v_combined = jnp.concatenate([v, v_compressed], axis=2)
    elif hasattr(self, "hca_compressor") and self.hca_compressor is not None:
      compressed_kv, block_bias = self.hca_compressor(hidden_states, position_ids)
      k_compressed = jnp.broadcast_to(compressed_kv, (batch_size, self.num_heads, compressed_kv.shape[2], self.head_dim))
      v_compressed = jnp.broadcast_to(compressed_kv, (batch_size, self.num_heads, compressed_kv.shape[2], self.head_dim))
      k_combined = jnp.concatenate([k, k_compressed], axis=2)
      v_combined = jnp.concatenate([v, v_compressed], axis=2)
    else:
      k_combined = k
      v_combined = v
      block_bias = None

    # 5. Compute Attention Logits
    logits = jnp.matmul(q, jnp.transpose(k_combined, (0, 1, 3, 2))) * self.softmax_scale  # [B, H, S, S_combined]

    # 6. Apply Causal + Block Bias Masking
    # Standard causal mask for sliding window
    causal_mask = jnp.where(
        jnp.arange(seq_len)[:, None] >= jnp.arange(seq_len)[None, :],
        0.0,
        -jnp.inf
    )
    # Reshape and broadcast causal mask to [B, 1, S, S] to ensure batch dimension matches block_bias perfectly!
    causal_mask = jnp.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))

    # Combine sliding window mask and compressed block bias mask
    if block_bias is not None:
      combined_mask = jnp.concatenate([causal_mask, block_bias], axis=3)
      logits = logits + combined_mask
    else:
      logits = logits + causal_mask

    # Add external document packing segment mask if present
    if segment_mask_sliding is not None:
      if block_bias is not None and compressed_segment_mask is not None:
        # Extend segment mask to cover compressed dimension: [segment_mask_sliding, compressed_segment_mask]
        extended_segment_mask = jnp.concatenate([
            segment_mask_sliding,
            jnp.expand_dims(compressed_segment_mask[:, :, :block_bias.shape[-1]], axis=1)
        ], axis=3)
        logits = logits + extended_segment_mask
      else:
        logits = logits + segment_mask_sliding

    # 7. Append Attention Sinks
    # Reshape sinks dynamically to [1, num_heads, 1, 1] to prevent any shape overwrite issues
    sinks_reshaped = self.sinks.value.reshape((1, self.num_heads, 1, 1))
    sinks_broadcast = jnp.broadcast_to(sinks_reshaped, (batch_size, self.num_heads, seq_len, 1))
    logits_with_sinks = jnp.concatenate([logits, sinks_broadcast], axis=3)

    # 8. Softmax and drop sinks column
    weights = jax.nn.softmax(logits_with_sinks, axis=-1)[..., :-1]

    # 9. Compute Mixed Attention Output
    attn_output = jnp.matmul(weights, v_combined)

    # 10. Triple RoPE Inverse Output Rotation (Conjugate rotation to undo V rotation!)
    attn_output = self.rotary_emb(attn_output, positions_for_rope, unsqueeze_dim=1, reverse=True)

    # Transpose and reshape to group-wise format: [B, S, g, in_features_per_group]
    # to match PyTorch's logical sequence-first layout!
    attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [B, S, H, D]
    attn_output_grouped = attn_output.reshape((batch_size, seq_len, self.o_groups, -1))

    # 11. Grouped Output Projections
    grouped = self.o_a_proj(attn_output_grouped)
    # Flatten grouped outputs back to full projection shape
    grouped_flat = grouped.reshape((batch_size, seq_len, self.o_groups * self.o_lora_rank))
    output = self.o_b_proj(grouped_flat)

    return output, None
