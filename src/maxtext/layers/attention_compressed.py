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

"""Compressed Attention layers and long-range compressors."""

from typing import Any
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding, apply_rotary_pos_emb
from maxtext.layers.normalizations import DeepSeekV4RMSNorm, DeepSeekV4UnweightedRMSNorm
from maxtext.layers.linears import DeepSeekGroupedLinear
from maxtext.layers.attention_op import AttentionOp
from maxtext.common.common_types import MODEL_MODE_TRAIN, AttentionType


class HCACompressor(nnx.Module):
  """Heavily Compressed Attention (HCA) long-range compressor layer.

  This layer groups sequence features into non-overlapping windows of size 'compress_rate',
  applies learnable pooling gates combined with static positional bias, averages the features
  inside each window to emit a single compressed representation per window, and rotates the
  resulting compressed sequence using interleaved rotary embeddings.
  """

  def __init__(
      self,
      hidden_size: int,
      head_dim: int,
      config: Any,
      layer_idx: int,
      eps: float = 1e-6,
      weight_dtype: Any = jnp.float32,
      dtype: Any = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the Heavily Compressed Attention (HCA) long-range compressor.

    Args:
      hidden_size: The model's global hidden dimension size.
      head_dim: The projection size of each attention key-value channel.
      config: The DeepSeekV4 model configurations metadata.
      layer_idx: The sequential layer depth index of this compressor in the decoder stack.
      eps: The tiny additive variance limit for RMS normalization stability.
      weight_dtype: The parameter weights numerical data type.
      dtype: The mathematical execution numerical data type.
      rngs: The standard Flax NNX random number generator collection.
    """
    super().__init__()
    self.compress_rate = config.compress_ratios[layer_idx]
    self.head_dim = head_dim
    self.hidden_size = hidden_size
    self.eps = eps
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    rope_theta = config.compress_rope_theta

    # Linear projection of inputs to key/value representation
    self.kv_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Linear projection of inputs to gate logits
    self.gate_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Positional bias parameter added to gate logits inside each window
    self.position_bias = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(
            rngs.params(),
            (self.compress_rate, head_dim),
            weight_dtype,
        )
    )

    # RMS normalization applied to pooled window features
    self.kv_norm = DeepSeekV4RMSNorm(
        hidden_size=head_dim,
        eps=eps,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )

    # Interleaved rotary embeddings applied to the trailing slice
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
        rope_theta=rope_theta,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      q_residual: Any = None,
      position_ids: jnp.ndarray = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Applies Heavily Compressed Attention (HCA) compression to sequence keys and values.

    This method splits the sequence into non-overlapping windows of size 'compress_rate',
    aggregates feature representation vectors using Softmax-weighted gates, normalizes the
    resulting vectors using RMS norm, applies position-aware interleaved rotary embeddings,
    and expands the output dimension to match standard multi-head key-value layouts.

    Args:
      hidden_states: The input hidden representation sequence of shape [B, S, D_model].
      q_residual: Ignored optional placeholder matching polymorphic calling conventions.
      position_ids: Optional position indicators of shape [B, S].

    Returns:
      Compressed, position-encoded representation tensor of shape [B, 1, W, D_head],
      where W is the compressed sequence length equal to S // compress_rate.
    """
    # hidden_states: [B, S, D_model]
    # position_ids: [B, S]
    batch, seq_len, _ = hidden_states.shape

    # Project inputs to key/value and gate representations
    # kv: [B, S, D_head]
    # gate: [B, S, D_head]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # Compute sequence multiple bound corresponding to the window stride rate
    # usable: scalar integer
    usable = (seq_len // self.compress_rate) * self.compress_rate
    n_windows = usable // self.compress_rate

    # Slice sequences to match clean multiple dimensions
    # chunk_kv: [B, S_usable, D_head]
    # chunk_gate: [B, S_usable, D_head]
    chunk_kv = kv[:, :usable, :]
    chunk_gate = gate[:, :usable, :]

    # Reshape sliced inputs into non-overlapping windows of size 'compress_rate'
    # chunk_kv: [B, W, compress_rate, D_head]
    # chunk_gate: [B, W, compress_rate, D_head]
    chunk_kv = chunk_kv.reshape(batch, n_windows, self.compress_rate, self.head_dim)
    chunk_gate = chunk_gate.reshape(batch, n_windows, self.compress_rate, self.head_dim)

    # Add positional bias parameters to gate logits
    # chunk_gate: [B, W, compress_rate, D_head]
    position_bias = jnp.asarray(self.position_bias[...], self.dtype)
    chunk_gate = chunk_gate + position_bias[jnp.newaxis, jnp.newaxis, :, :]

    # Compute softmax aggregation probabilities in float32 for stability
    # gate_softmax: [B, W, compress_rate, D_head]
    gate_softmax = jax.nn.softmax(chunk_gate.astype(jnp.float32), axis=2).astype(self.dtype)

    # Aggregate key/value features using computed gate weights
    # pooled: [B, W, D_head]
    pooled = jnp.sum(chunk_kv * gate_softmax, axis=2)

    # Normalize aggregated window features
    # compressed: [B, W, D_head]
    compressed = self.kv_norm(pooled)

    # Determine absolute sequence indexes corresponding to each window start
    # positions: [B, W]
    positions = jnp.arange(n_windows, dtype=jnp.int32) * self.compress_rate
    positions = jnp.broadcast_to(positions[jnp.newaxis, :], (batch, n_windows))

    # Compute interleaved rotary embeddings sine and cosine values
    # cos: [B, W, D_rope/2]
    # sin: [B, W, D_rope/2]
    cos, sin = self.rotary_emb(compressed, positions)

    # Expand dimensions to allow broadcasting over head axis during rotary mapping
    # compressed_4d: [B, W, 1, D_head]
    compressed_4d = jnp.expand_dims(compressed, axis=2)

    # Apply interleaved RoPE rotation over the trailing slice
    # rotated_4d: [B, W, 1, D_head]
    rotated_4d = apply_rotary_pos_emb(compressed_4d, cos, sin, unsqueeze_dim=2)

    # Squeeze dummy head dimension to recover standard 3D shape layout
    # rotated: [B, W, D_head]
    rotated = jnp.squeeze(rotated_4d, axis=2)

    # Expand output format to match standard multi-head key/value dimensions
    # compressed_kv: [B, 1, W, D_head]
    compressed_kv = jnp.expand_dims(rotated, axis=1)

    # Evaluate caching dimensions boundary checks to prevent empty execution
    compressed_len = n_windows
    if seq_len == 1 or compressed_len == 0:
      return compressed_kv, None

    # Compute causal block bias mask over compressed sequence segments to prevent query leakage.
    # A query at sequence position `t` is restricted from attending to any compressed cache block
    # index `w` if `t <= w * compress_rate`. This represents future sequence information that is
    # mathematically unavailable at position `t`.
    #
    # entry_indices: [W] representing compressed block window positions
    entry_indices = jnp.arange(compressed_len, dtype=jnp.int32)
    # causal_threshold: [B, S] representing ready block count boundaries per sequence token
    causal_threshold = (position_ids + 1) // self.compress_rate
    # Construct sequence-level causal future mask via dimension broadcasting.
    # future_mask: [B, 1, S, W]
    future_mask = (
        entry_indices[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] >= causal_threshold[:, jnp.newaxis, :, jnp.newaxis]
    )
    # Initialize causal block bias containing -inf mask values for invalid future elements.
    # block_bias: [B, 1, S, W]
    block_bias = jnp.where(future_mask, -jnp.inf, 0.0)
    return compressed_kv, block_bias


class DeepSeekV4Indexer(nnx.Module):
  """Lightning Indexer (paper §2.3.1, eqs. 13–17).

  Used by Compressed Sparse Attention (CSA) to pick the top-k compressed KV
  blocks per query.
  """

  def __init__(
      self,
      hidden_size: int,
      q_lora_rank: int,
      config: Any,
      layer_idx: int,
      eps: float = 1e-6,
      weight_dtype: Any = jnp.float32,
      dtype: Any = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the Lightning Indexer.

    Args:
      hidden_size: The model's global hidden dimension size.
      q_lora_rank: The projection rank dimension of Q LoRA.
      config: The DeepSeekV4 model configurations metadata.
      layer_idx: The decoder stack layer index containing this indexer.
      eps: Tiny additive variance limit for RMS normalization stability.
      weight_dtype: The parameter weights numerical data type.
      dtype: The mathematical execution numerical data type.
      rngs: The Flax NNX random number generator collection.
    """
    super().__init__()
    self.compress_rate = config.compress_ratios[layer_idx]
    self.num_heads = config.index_n_heads
    self.head_dim = config.index_head_dim
    self.index_topk = config.index_topk
    self.softmax_scale = config.index_head_dim**-0.5
    self.weights_scaling = config.index_n_heads**-0.5
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    rope_theta = config.compress_rope_theta

    # Key projections for indexing-scale compression
    self.kv_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=2 * self.head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Gate projections for indexing-scale compression
    self.gate_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=2 * self.head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Positional bias parameters inside indexing windows
    self.position_bias = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(
            rngs.params(),
            (self.compress_rate, 2 * self.head_dim),
            weight_dtype,
        )
    )

    # RMS normalization for indexer key values
    self.kv_norm = DeepSeekV4RMSNorm(
        hidden_size=self.head_dim,
        eps=eps,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )

    # Query projection mapping Q LoRA rank to multi-head indexing features
    self.q_b_proj = nnx.Linear(
        in_features=q_lora_rank,
        out_features=self.num_heads * self.head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Dynamic score scaling projection
    self.weights_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=self.num_heads,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Interleaved rotary embedding aligning query/key pos representations
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=config.qk_rope_head_dim / self.head_dim,
        rope_theta=rope_theta,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      q_residual: jnp.ndarray,
      position_ids: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes top-k relevant compressed block indices per query position.

    This method compresses sequence keys and values into overlapping window
    segments, applies position-aware RoPE encoding, projects incoming query residuals
    into alignment spaces, computes similarity matrices across query positions and
    windows, dynamically scales/weights scores using projected head scaling arrays,
    and selects the top-k windows using JAX optimized top_k primitives.

    Args:
      hidden_states: The input sequence representations of shape [B, S, D_model].
      q_residual: The Q LoRA low-rank query projections of shape [B, S, D_rank].
      position_ids: The sequence absolute position identifiers of shape [B, S].

    Returns:
      Integer index array of shape [B, S, k] containing the gathered top-k
      compressed window indices for each query position, where k = index_topk.
    """
    # hidden_states: [B, S, D_model]
    # q_residual: [B, S, D_rank]
    # position_ids: [B, S]
    batch, seq_len, _ = hidden_states.shape

    # Project inputs to index keys and gates
    # kv: [B, S, 2 * D_idx]
    # gate: [B, S, 2 * D_idx]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # Calculate sequence bounds matching the stride rate
    # usable: scalar integer
    usable = (seq_len // self.compress_rate) * self.compress_rate
    n_windows = usable // self.compress_rate

    # Slice sequences to valid sequence bounds
    # chunk_kv: [B, S_usable, 2 * D_idx]
    # chunk_gate: [B, S_usable, 2 * D_idx]
    chunk_kv = kv[:, :usable, :]
    chunk_gate = gate[:, :usable, :]

    # Segment sliced elements into non-overlapping windows
    # chunk_kv: [B, W, compress_rate, 2 * D_idx]
    # chunk_gate: [B, W, compress_rate, 2 * D_idx]
    chunk_kv = chunk_kv.reshape(batch, n_windows, self.compress_rate, 2 * self.head_dim)
    chunk_gate = chunk_gate.reshape(batch, n_windows, self.compress_rate, 2 * self.head_dim)

    # Incorporate static positional bias parameters
    # chunk_gate: [B, W, compress_rate, 2 * D_idx]
    position_bias = jnp.asarray(self.position_bias[...], self.dtype)
    chunk_gate = chunk_gate + position_bias[jnp.newaxis, jnp.newaxis, :, :]

    # Overlap slicing setups: segment into Ca / Cb series
    # prev_kv: [B, W, compress_rate, D_idx] (Ca)
    # curr_kv: [B, W, compress_rate, D_idx] (Cb)
    # prev_gate: [B, W, compress_rate, D_idx] (Ca)
    # curr_gate: [B, W, compress_rate, D_idx] (Cb)
    prev_kv = chunk_kv[..., : self.head_dim]
    curr_kv = chunk_kv[..., self.head_dim :]
    prev_gate = chunk_gate[..., : self.head_dim]
    curr_gate = chunk_gate[..., self.head_dim :]

    # Set up combined padded layouts for boundary window overlap calculations
    # new_kv: [B, W, 2 * compress_rate, D_idx]
    # new_gate: [B, W, 2 * compress_rate, D_idx]
    new_kv = jnp.zeros((batch, n_windows, 2 * self.compress_rate, self.head_dim), dtype=self.dtype)
    new_gate = jnp.full((batch, n_windows, 2 * self.compress_rate, self.head_dim), -jnp.inf, dtype=self.dtype)

    # Map Cb representations into second half slots
    new_kv = new_kv.at[:, :, self.compress_rate :].set(curr_kv)
    new_gate = new_gate.at[:, :, self.compress_rate :].set(curr_gate)

    # Map Ca representations of preceding windows into first half slots
    if n_windows > 1:
      new_kv = new_kv.at[:, 1:, : self.compress_rate].set(prev_kv[:, :-1, :, :])
      new_gate = new_gate.at[:, 1:, : self.compress_rate].set(prev_gate[:, :-1, :, :])

    # Aggregate indexing features using gate softmax probabilities computed in float32
    # gate_softmax: [B, W, 2 * compress_rate, D_idx]
    gate_softmax = jax.nn.softmax(new_gate.astype(jnp.float32), axis=2).astype(self.dtype)
    # pooled: [B, W, D_idx]
    pooled = jnp.sum(new_kv * gate_softmax, axis=2)

    # Normalize index keys
    # compressed: [B, W, D_idx]
    compressed = self.kv_norm(pooled)

    # Extract absolute starting positions of index windows
    # positions: [B, W]
    positions = jnp.arange(n_windows, dtype=jnp.int32) * self.compress_rate
    positions = jnp.broadcast_to(positions[jnp.newaxis, :], (batch, n_windows))

    # Compute sinusoids and apply interleaved rotary embeddings
    # cos: [B, W, D_idx_rope/2]
    # sin: [B, W, D_idx_rope/2]
    cos, sin = self.rotary_emb(compressed, positions)
    # compressed_4d: [B, W, 1, D_idx]
    compressed_4d = jnp.expand_dims(compressed, axis=2)
    # rotated_4d: [B, W, 1, D_idx]
    rotated_4d = apply_rotary_pos_emb(compressed_4d, cos, sin, unsqueeze_dim=2)
    # compressed_kv: [B, W, D_idx]
    compressed_kv = jnp.squeeze(rotated_4d, axis=2)

    # Project and reshape queries to multiple head alignments
    # q: [B, S, H, D_idx]
    q = self.q_b_proj(q_residual)
    q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)

    # Compute rotary components matching current query positions
    # cos_q: [B, S, D_idx_rope/2]
    # sin_q: [B, S, D_idx_rope/2]
    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids)
    # Apply RoPE to query elements
    # q: [B, S, H, D_idx]
    q = apply_rotary_pos_emb(q, cos_q, sin_q, unsqueeze_dim=2)

    # Calculate attention alignment scores across windows
    # swaped_kv: [B, 1, D_idx, W]
    swaped_kv = jnp.swapaxes(compressed_kv, -1, -2)
    swaped_kv = jnp.expand_dims(swaped_kv, axis=1)
    # scores: [B, S, H, W]
    scores = jnp.matmul(q, swaped_kv)
    scores = jax.nn.relu(scores) * self.softmax_scale

    # Project and scale dynamic aggregation scoring weights
    # weights: [B, S, H]
    weights = self.weights_proj(hidden_states) * self.weights_scaling
    # Aggregate scoring profiles over heads axis
    # index_scores: [B, S, W]
    index_scores = jnp.sum(scores * jnp.expand_dims(weights, axis=-1), axis=2)

    # Extract top-k scoring compressed blocks per query sequence position
    # topk_indices: [B, S, k]
    compressed_len = compressed_kv.shape[1]
    topk_limit = min(self.index_topk, compressed_len)

    if compressed_len > 0:
      # Compute sequence-level causal ready block counts.
      # causal_threshold: [B, S]
      causal_threshold = (position_ids + 1) // self.compress_rate
      # entry_indices: [W]
      entry_indices = jnp.arange(compressed_len, dtype=jnp.int32)
      # Construct query-specific causal mask along compressed index dimension.
      # future_mask: [B, S, W]
      future_mask = entry_indices[jnp.newaxis, jnp.newaxis, :] >= causal_threshold[:, :, jnp.newaxis]
      # Zero-out future block scores by masking them with -inf prior to top-k calculations.
      # index_scores: [B, S, W]
      index_scores = jnp.where(future_mask, -jnp.inf, index_scores)
      # Select top-k indices per token position based on masked scores.
      # topk_indices: [B, S, k]
      _, topk_indices = jax.lax.top_k(index_scores, topk_limit)
      # Early tokens with too few ready blocks will still have invalid top-k selections pointing
      # to future blocks. Detect them and replace with a `-1` sentinel.
      # invalid: [B, S, k]
      invalid = topk_indices >= causal_threshold[:, :, jnp.newaxis]
      topk_indices = jnp.where(invalid, -1, topk_indices)
      return topk_indices

    # Fallback stateless default top-k select path
    _, topk_indices = jax.lax.top_k(index_scores, topk_limit)
    return topk_indices


class CSACompressor(nnx.Module):
  """Compressed Sparse Attention (CSA) compressor layer.

  This layer aggregates token representations into overlapping Ca/Cb window segments,
  normalizes/rotates them, and uses the DeepSeekV4Indexer to gather the top-k
  relevant compressed KV blocks per query.
  """

  def __init__(
      self,
      hidden_size: int,
      q_lora_rank: int,
      head_dim: int,
      config: Any,
      layer_idx: int,
      eps: float = 1e-6,
      weight_dtype: Any = jnp.float32,
      dtype: Any = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the Compressed Sparse Attention (CSA) compressor.

    Args:
      hidden_size: The model's global hidden dimension size.
      q_lora_rank: The projection rank dimension of Q LoRA.
      head_dim: The projection size of each attention key-value channel.
      config: The DeepSeekV4 model configurations metadata.
      layer_idx: The decoder stack layer index containing this compressor.
      eps: Tiny additive variance limit for RMS normalization stability.
      weight_dtype: The parameter weights numerical data type.
      dtype: The mathematical execution numerical data type.
      rngs: The Flax NNX random number generator collection.
    """
    super().__init__()
    self.compress_rate = config.compress_ratios[layer_idx]
    self.head_dim = head_dim
    self.hidden_size = hidden_size
    self.eps = eps
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    rope_theta = config.compress_rope_theta

    # Projections for outer compressed key/value formats
    self.kv_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=2 * head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Projections for outer gate logits
    self.gate_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=2 * head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Static positional biases added inside windows
    self.position_bias = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(
            rngs.params(),
            (config.compress_ratios[layer_idx], 2 * head_dim),
            weight_dtype,
        )
    )

    # RMS normalization applied to aggregated representations
    self.kv_norm = DeepSeekV4RMSNorm(
        hidden_size=head_dim,
        eps=eps,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )

    # Interleaved rotary embeddings for compressed sequences
    self.rotary_emb = DeepSeekV4RotaryEmbedding(
        head_dim=head_dim,
        partial_rotary_factor=config.qk_rope_head_dim / config.head_dim,
        rope_theta=rope_theta,
    )

    # Lightning Indexer component
    self.indexer = DeepSeekV4Indexer(
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        config=config,
        layer_idx=layer_idx,
        eps=eps,
        weight_dtype=weight_dtype,
        dtype=dtype,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      q_residual: jnp.ndarray,
      position_ids: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies Compressed Sparse Attention (CSA) compression and gathers top-k blocks.

    This method compresses sequence keys and values into overlapping window
    segments, applies position-aware RoPE encoding, runs the Lightning Indexer to
    extract the top-k scoring window indices for each query position, executes a
    high-performance TPU-efficient advanced gather, and shapes the output to match
    standard multi-head key-value layouts.

    Args:
      hidden_states: The input sequence representations of shape [B, S, D_model].
      q_residual: The Q LoRA low-rank query projections of shape [B, S, D_rank].
      position_ids: The sequence absolute position identifiers of shape [B, S].

    Returns:
      Position-encoded, gathered key-value representation tensor of shape
      [B, 1, S * k, D_head], where k = index_topk.
    """
    # hidden_states: [B, S, D_model]
    # q_residual: [B, S, D_rank]
    # position_ids: [B, S]
    batch, seq_len, _ = hidden_states.shape

    # Project input features to key/value and gate components
    # kv: [B, S, 2 * D]
    # gate: [B, S, 2 * D]
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # Determine valid sequence bounds
    # usable: scalar integer
    usable = (seq_len // self.compress_rate) * self.compress_rate
    n_windows = usable // self.compress_rate

    # Slice inputs to sequence bounds
    # chunk_kv: [B, S_usable, 2 * D]
    # chunk_gate: [B, S_usable, 2 * D]
    chunk_kv = kv[:, :usable, :]
    chunk_gate = gate[:, :usable, :]

    # Segment sliced elements into non-overlapping windows
    # chunk_kv: [B, W, compress_rate, 2 * D]
    # chunk_gate: [B, W, compress_rate, 2 * D]
    chunk_kv = chunk_kv.reshape(batch, n_windows, self.compress_rate, 2 * self.head_dim)
    chunk_gate = chunk_gate.reshape(batch, n_windows, self.compress_rate, 2 * self.head_dim)

    # Aggregate window gate logits with static positional biases
    # chunk_gate: [B, W, compress_rate, 2 * D]
    position_bias = jnp.asarray(self.position_bias[...], self.dtype)
    chunk_gate = chunk_gate + position_bias[jnp.newaxis, jnp.newaxis, :, :]

    # Overlap slicing: extract Ca / Cb configurations
    # prev_kv: [B, W, compress_rate, D] (Ca)
    # curr_kv: [B, W, compress_rate, D] (Cb)
    # prev_gate: [B, W, compress_rate, D] (Ca)
    # curr_gate: [B, W, compress_rate, D] (Cb)
    prev_kv = chunk_kv[..., : self.head_dim]
    curr_kv = chunk_kv[..., self.head_dim :]
    prev_gate = chunk_gate[..., : self.head_dim]
    curr_gate = chunk_gate[..., self.head_dim :]

    # Assemble padded window layouts for overlap combination
    # new_kv: [B, W, 2 * compress_rate, D]
    # new_gate: [B, W, 2 * compress_rate, D]
    new_kv = jnp.zeros((batch, n_windows, 2 * self.compress_rate, self.head_dim), dtype=self.dtype)
    new_gate = jnp.full((batch, n_windows, 2 * self.compress_rate, self.head_dim), -jnp.inf, dtype=self.dtype)

    # Set current window representations to second half slots
    new_kv = new_kv.at[:, :, self.compress_rate :].set(curr_kv)
    new_gate = new_gate.at[:, :, self.compress_rate :].set(curr_gate)

    # Set previous window representations to first half slots
    if n_windows > 1:
      new_kv = new_kv.at[:, 1:, : self.compress_rate].set(prev_kv[:, :-1, :, :])
      new_gate = new_gate.at[:, 1:, : self.compress_rate].set(prev_gate[:, :-1, :, :])

    # Aggregate features using window gate softmax probabilities computed in float32
    # gate_softmax: [B, W, 2 * compress_rate, D]
    gate_softmax = jax.nn.softmax(new_gate.astype(jnp.float32), axis=2).astype(self.dtype)
    # pooled: [B, W, D]
    pooled = jnp.sum(new_kv * gate_softmax, axis=2)

    # Normalize window features
    # compressed: [B, W, D]
    compressed = self.kv_norm(pooled)

    # Obtain starting positions of compressed windows
    # positions: [B, W]
    positions = jnp.arange(n_windows, dtype=jnp.int32) * self.compress_rate
    positions = jnp.broadcast_to(positions[jnp.newaxis, :], (batch, n_windows))

    # Apply interleaved rotary embeddings over aggregated outputs
    # cos: [B, W, D_rope/2]
    # sin: [B, W, D_rope/2]
    cos, sin = self.rotary_emb(compressed, positions)
    # compressed_4d: [B, W, 1, D]
    compressed_4d = jnp.expand_dims(compressed, axis=2)
    # rotated_4d: [B, W, 1, D]
    rotated_4d = apply_rotary_pos_emb(compressed_4d, cos, sin, unsqueeze_dim=2)
    # compressed_kv: [B, W, D]
    compressed_kv = jnp.squeeze(rotated_4d, axis=2)

    # Execute Lightning Indexer to obtain block indices per query
    # topk: [B, S, k]
    topk = self.indexer(hidden_states, q_residual, position_ids)

    # Clamp indices safely using jnp.clip to avoid JAX negative/out-of-bounds indexing exceptions
    # under indexer -1 sentinel conditions.
    # safe_indices: [B, S, k]
    safe_indices = jnp.clip(topk, a_min=0)
    # batch_idx: [B, 1, 1]
    batch_idx = jnp.arange(batch)[:, jnp.newaxis, jnp.newaxis]
    # Perform TPU-efficient JAX Advanced Indexing Gather.
    # gathered: [B, S, k, D]
    gathered = compressed_kv[batch_idx, safe_indices, :]

    # Reshape gathered elements to standardized multi-head formats
    # compressed_kv_out: [B, 1, S * k, D]
    compressed_kv_out = gathered.reshape(batch, 1, seq_len * topk.shape[-1], self.head_dim)

    # Vectorized block bias mask construction to filter out invalid sparse gathered entries.
    # valid: [B, S, k] indicating whether each top-k selection is valid (non-sentinel)
    valid = topk >= 0
    # allowed: [B, S, k] containing 0.0 for valid entries and -inf for invalid sentinels
    allowed = jnp.where(valid, 0.0, -jnp.inf)
    # Construct an equivalence diagonal mask matching query sequence indices.
    # eq_mask: [S, S, 1] representing identity query boundaries
    eq_mask = jnp.arange(seq_len)[:, jnp.newaxis, jnp.newaxis] == jnp.arange(seq_len)[jnp.newaxis, :, jnp.newaxis]
    # allowed_expanded: [B, S, 1, k]
    allowed_expanded = allowed[:, :, jnp.newaxis, :]
    # Distribute allowed masks diagonally using JAX vectorization to prevent cross-query leakage.
    # block_bias_5d: [B, S, S, k]
    block_bias_5d = jnp.where(eq_mask[jnp.newaxis, :, :, :], allowed_expanded, -jnp.inf)
    # Reshape and format to standard key-value sequence length formats
    # block_bias: [B, S, S * k]
    block_bias = block_bias_5d.reshape(batch, seq_len, seq_len * topk.shape[-1])
    # block_bias: [B, 1, S, S * k]
    block_bias = jnp.expand_dims(block_bias, axis=1)
    return compressed_kv_out, block_bias


class DeepSeekV4Attention(nnx.Module):
  """Main coordination attention block for DeepSeek-V4 compressed layer configurations.

  This module implements multi-head attention augmented with query-compression LoRA
  projections, unweighted key/value normalizations, optional heavily or sparsely
  compressed long-range context compressor integrations, learnable attention sinks,
  and parallelized grouped output mixing projections.
  """

  def __init__(
      self,
      hidden_size: int,
      q_lora_rank: int,
      head_dim: int,
      num_heads: int,
      config: Any,
      layer_idx: int,
      mesh: Mesh | None = None,
      eps: float = 1e-6,
      weight_dtype: Any = jnp.float32,
      dtype: Any = jnp.float32,
      attention_type: str = "compressed_sparse_attention",
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes the DeepSeekV4 Attention coordinator block.

    Args:
      hidden_size: The model's global hidden dimension size.
      q_lora_rank: The projection rank dimension of Q LoRA.
      head_dim: The projection size of each attention key-value channel.
      num_heads: The total number of query attention heads.
      config: The DeepSeekV4 model configurations metadata.
      layer_idx: The decoder stack layer index containing this attention module.
      eps: Tiny additive variance limit for RMS normalization stability.
      weight_dtype: The parameter weights numerical data type.
      dtype: The mathematical execution numerical data type.
      attention_type: The type of compressed attention being instantiated.
      rngs: The Flax NNX random number generator collection.
    """
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.attention_type = attention_type
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.sliding_window = config.sliding_window
    self.scaling = head_dim**-0.5
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    # Projections for query extraction and low-rank compression
    self.q_a_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=q_lora_rank,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    self.q_a_norm = DeepSeekV4RMSNorm(
        hidden_size=q_lora_rank,
        eps=eps,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )
    self.q_b_proj = nnx.Linear(
        in_features=q_lora_rank,
        out_features=num_heads * head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    self.q_b_norm = DeepSeekV4UnweightedRMSNorm(
        eps=eps,
        dtype=dtype,
    )

    # Unified projected shared MQA key/value block
    self.kv_proj = nnx.Linear(
        in_features=hidden_size,
        out_features=head_dim,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )
    self.kv_norm = DeepSeekV4RMSNorm(
        hidden_size=head_dim,
        eps=eps,
        dtype=dtype,
        weight_dtype=weight_dtype,
    )

    # Block-diagonal grouped linear layer for multi-head features mixing
    self.o_a_proj = DeepSeekGroupedLinear(
        in_features_per_group=num_heads * head_dim // config.o_groups,
        out_features=config.o_groups * config.o_lora_rank,
        n_groups=config.o_groups,
        weight_dtype=weight_dtype,
        dtype=dtype,
        rngs=rngs,
    )
    self.o_b_proj = nnx.Linear(
        in_features=config.o_groups * config.o_lora_rank,
        out_features=hidden_size,
        use_bias=False,
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

    # Attention Sink Parameter
    self.sinks = nnx.Param(jax.nn.initializers.zeros(rngs.params(), (num_heads,), weight_dtype))

    # Layer specific compressor allocation
    if self.attention_type == "heavily_compressed_attention":
      self.compressor = HCACompressor(
          hidden_size=hidden_size,
          head_dim=head_dim,
          config=config,
          layer_idx=layer_idx,
          eps=eps,
          weight_dtype=weight_dtype,
          dtype=dtype,
          rngs=rngs,
      )
    elif self.attention_type == "compressed_sparse_attention":
      self.compressor = CSACompressor(
          hidden_size=hidden_size,
          q_lora_rank=q_lora_rank,
          head_dim=head_dim,
          config=config,
          layer_idx=layer_idx,
          eps=eps,
          weight_dtype=weight_dtype,
          dtype=dtype,
          rngs=rngs,
      )
    else:
      self.compressor = None

    # Compute partial rotary factor dynamically from config to prevent dimension mismatches.
    # DeepSeek-V4 pairs consecutive channels to apply partial RoPE on qk_rope_head_dim channels,
    # requiring dynamic scaling: partial_rotary_factor = qk_rope_head_dim / head_dim.
    self.partial_rotary_factor = self.config.qk_rope_head_dim / self.config.head_dim

    self.rope_theta = (
        self.config.rope_max_timescale if self.attention_type == "sliding_attention" else self.config.compress_rope_theta
    )

    # Local rotary embedding block matching standard MaxText (Gemma/Llama2) paradigms.
    self.rotary_embedding = DeepSeekV4RotaryEmbedding(
        head_dim=self.head_dim,
        partial_rotary_factor=self.partial_rotary_factor,
        rope_theta=self.rope_theta,
    )

    # Scaling factor applied to query representations to match standard MaxText attention scaling.
    # MaxText's AttentionOp core expects queries to be pre-scaled by 1 / sqrt(head_dim).
    self.scaling = self.head_dim**-0.5

    self.attention_op = AttentionOp(
        config=self.config,
        mesh=mesh,
        attention_kernel=self.config.attention,
        max_target_length=self.config.max_target_length,
        num_query_heads=self.num_heads,
        num_kv_heads=1,
        dtype=self.dtype,
        compute_axis_order=(0, 1, 2, 3),
        attention_type=AttentionType.FULL,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray | None = None,
      position_ids: jnp.ndarray | None = None,
      attention_mask: jnp.ndarray | None = None,
      inputs_q: jnp.ndarray | None = None,
      inputs_kv: jnp.ndarray | None = None,
      **kwargs,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Executes the main coordination attention pass over sequence inputs.

    This method coordinates multi-head attention augmented with query-compression LoRA
    projections, unweighted key/value normalizations, long-range context compressor
    integrations, learnable attention sinks, and parallelized grouped output mixing.

    Args:
      hidden_states: Input sequence representations of shape [B, S, D_model].
      position_ids: Sequence absolute position identifiers of shape [B, S].
      attention_mask: Optional attention mask preventing invalid token attendance.
      inputs_q: Optional query input override for decoupled execution.
      inputs_kv: Optional key/value input override for decoupled execution.
      **kwargs: Additional runtime execution configurations (e.g., decoder_segment_ids).

    Returns:
      Tuple containing the projected output representations of shape [B, S, D_model]
      and an empty caching intermediate placeholder.
    """
    # Resolve input representations from standard hidden states or override inputs.
    # hidden_states: [B, S, D_model]
    if hidden_states is None:
      hidden_states = inputs_q
    batch, seq_len, _ = hidden_states.shape

    # Generate absolute position identifiers if not provided at runtime.
    # position_ids: [B, S]
    if position_ids is None:
      position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None], (batch, seq_len))

    # Resolve rotary position embedding sinusoids from runtime keyword arguments or compute local sinusoids.
    # Utilizing pre-computed sinusoids avoids redundant computation across decoder layers during forward passes.
    # cos: [B, S, D_rope/2]
    # sin: [B, S, D_rope/2]
    cos = kwargs.get("cos", None)
    sin = kwargs.get("sin", None)
    if cos is None or sin is None:
      cos, sin = self.rotary_embedding(hidden_states, position_ids)

    # Project input features to low-rank query residuals and apply RMS normalization.
    # # [B, S, D_model] -> [B, S, D_rank]
    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))

    # Project low-rank residuals to multi-head query dimensions and reshape.
    # # [B, S, D_rank] -> [B, S, H, D_head]
    q = self.q_b_proj(q_residual).reshape(batch, seq_len, self.num_heads, self.head_dim)

    # Apply scale-free unweighted RMS normalization across multi-head queries and scale by attention scaling factor.
    # Unweighted normalization stabilizes query variance without introducing learnable scaling parameters.
    # MaxText's AttentionOp core assumes pre-scaled query tensors, requiring explicit scaling here.
    # # [B, S, H, D_head] -> [B, S, H, D_head]
    q = self.q_b_norm(q) * self.scaling

    # Apply Rotary Position Embedding (RoPE) to query representations.
    # # [B, S, H, D_head] -> [B, S, H, D_head]
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)

    # Project input representations to shared key/value features and apply RMS normalization.
    # # [B, S, D_model] -> [B, S, 1, D_head]
    kv = self.kv_norm(self.kv_proj(hidden_states)).reshape(batch, seq_len, 1, self.head_dim)

    # Apply Rotary Position Embedding (RoPE) to shared key/value representations.
    # # [B, S, 1, D_head] -> [B, S, 1, D_head]
    kv = apply_rotary_pos_emb(kv, cos, sin, unsqueeze_dim=2)

    # Integrate long-range context compressor representations if configured.
    block_bias = None
    if self.compressor is not None:
      # Execute compressor pass to extract compressed key/value blocks and structural block bias masks.
      # compressed_kv: [B, 1, W, D_head] or [B, W, 1, D_head]
      # block_bias: [B, 1, S, W] or [B, S, W]
      compressed_kv, block_bias = self.compressor(hidden_states, q_residual, position_ids)

      # Standardize compressed key/value layout to match multi-head sequence formats.
      # # [B, 1, W, D_head] -> [B, W, 1, D_head]
      if compressed_kv.shape[1] == 1:
        compressed_kv = compressed_kv.transpose(0, 2, 1, 3)

      # Concatenate local sequence keys with compressed long-range cache blocks along sequence dimension.
      # # [B, S, 1, D_head] + [B, W, 1, D_head] -> [B, S + W, 1, D_head]
      kv = jnp.concatenate([kv, compressed_kv], axis=1)

    # Reconcile structural block bias masks with runtime attention masks.
    if attention_mask is not None:
      if block_bias is not None:
        # Concatenate block bias mask to attention mask along trailing sequence dimension.
        # # [B, 1, S, S] + [B, 1, S, W] -> [B, 1, S, S + W]
        attention_mask = jnp.concatenate([attention_mask, block_bias.astype(attention_mask.dtype)], axis=-1)
      elif kv.shape[1] > attention_mask.shape[-1]:
        # Pad attention mask with zero-value allowed elements to match extended key/value sequence length.
        # # [B, 1, S, S] -> [B, 1, S, S + W]
        pad_width = kv.shape[1] - attention_mask.shape[-1]
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, 0), (0, pad_width)), constant_values=0.0)
    # Ensure key/value sequence length is perfectly divisible by the Splash attention block size (sa_block_kv).
    # Hardware Matrix Multiply Units (MXUs) and XLA Pallas kernels enforce strict memory layout alignment grids.
    # When Splash Flash Attention is active, the runtime key/value sequence dimension must perfectly divide by sa_block_kv.
    # Because long-range cache compressors append dynamic auxiliary tokens (e.g. +32 tokens), the resulting combined length
    # may break hardware divisibility constraints (e.g. 4128 % 512 != 0).
    # This dynamic padding forces exact MXU grid alignment.
    # # [B, S + W, 1, D_head] -> [B, align(S + W, sa_block_kv), 1, D_head]
    if self.config.sa_block_kv > 0 and kv.shape[1] % self.config.sa_block_kv != 0:
      pad_len = self.config.sa_block_kv - (kv.shape[1] % self.config.sa_block_kv)
      kv = jnp.pad(kv, ((0, 0), (0, pad_len), (0, 0), (0, 0)), constant_values=0.0)
      if attention_mask is not None:
        # Pad 4D attention mask along trailing key/value sequence axis.
        # # [B, 1, Q, S + W] -> [B, 1, Q, align(S + W, sa_block_kv)]
        attention_mask = jnp.pad(attention_mask, ((0, 0), (0, 0), (0, 0), (0, pad_len)), constant_values=0.0)

    # Squeeze redundant head dimension from 4D attention masks to ensure compatibility with AttentionOp core.
    # # [B, 1, S, S + W] -> [B, S, S + W]
    unified_mask = (
        jnp.squeeze(attention_mask, axis=1) if attention_mask is not None and attention_mask.ndim == 4 else attention_mask
    )

    # Execute core attention operator pass over query and concatenated key/value sequences.
    # # q: [B, S, H, D_head], kv: [B, S + W, 1, D_head] -> [B, S, H, D_head]
    attn_output = self.attention_op(
        query=q,
        key=kv,
        value=kv,
        decoder_segment_ids=kwargs.get("decoder_segment_ids", None),
        inputs_positions=position_ids,
        model_mode=kwargs.get("model_mode", MODEL_MODE_TRAIN),
        indexer_mask=unified_mask,
        sinks=self.sinks,
    )

    # Apply conjugate RoPE rotation (-sin) to attention outputs to un-rotate representations.
    # Un-rotating aligns output feature spaces prior to multi-head mixing projections.
    # # [B, S, H, D_head] -> [B, S, H, D_head]
    attn_output = apply_rotary_pos_emb(attn_output, cos, -sin, unsqueeze_dim=2)

    # Reshape attention outputs into block-diagonal output groups.
    # # [B, S, H, D_head] -> [B, S, o_groups, H * D_head / o_groups]
    grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)

    # Apply block-diagonal grouped linear projections to mix intra-group features.
    # # [B, S, o_groups, H * D_head / o_groups] -> [B, S, o_groups, o_lora_rank]
    grouped = self.o_a_proj(grouped)

    # Flatten grouped representations into a unified feature vector per sequence position.
    # # [B, S, o_groups, o_lora_rank] -> [B, S, o_groups * o_lora_rank]
    grouped_flat = grouped.reshape(batch, seq_len, -1)

    # Project mixed representations back to global model hidden dimension.
    # # [B, S, o_groups * o_lora_rank] -> [B, S, D_model]
    output = self.o_b_proj(grouped_flat)

    return output, None
