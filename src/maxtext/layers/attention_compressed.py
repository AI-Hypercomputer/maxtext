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

"""Compressed Attention Layer (DeepSeek-V4)."""


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
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    AttentionType,
    DEFAULT_MASK_VALUE,
)

from maxtext.layers import nnx_wrappers
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned
from maxtext.layers.linears import DenseGeneral, DeepSeekV4GroupedLinear
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.inference.kvcache import KVQuant
from maxtext.inference import kvcache


# def csa_overlap_pooling(
#     hidden_states: Array,
#     kv_proj: Any,
#     gate_proj: Any,
#     position_bias: Array,
#     kv_norm: Any,
#     compress_rate: int,
#     head_dim: int,
# ) -> Array:
#   """Shared utility for Compressed Sparse Attention (CSA) overlap pooling.

#   Implements the overlapping Ca/Cb pooling logic shared by both the CSA Compressor
#   and the CSA Indexer. It splits the projected states into two halves (Ca and Cb),
#   shifts the first half forward by one window, and concatenates them to form
#   overlapping windows over which softmax gating is applied.

#   Args:
#     hidden_states: Input token embeddings. Shape: `[batch, seq_len, emb_dim]`.
#     kv_proj: Dense layer projecting to `2 * head_dim`.
#     gate_proj: Dense layer projecting to `2 * head_dim`.
#     position_bias: Bias tensor. Shape: `[compress_rate, 2 * head_dim]`.
#     kv_norm: RMSNorm instance.
#     compress_rate: Compression rate for CSA.
#     head_dim: Standard head dimension.

#   Returns:
#     compressed: The pooled overlapping states. Shape: `[batch, n_windows, head_dim]`.

#   Shape Transformations:
#     1. Projections: `[batch, seq_len, emb_dim]` -> `[batch, seq_len, 2 * head_dim]`
#     2. Reshape: -> `[batch, n_windows, compress_rate, 2 * head_dim]`
#     3. Split: -> 2x `[batch, n_windows, compress_rate, head_dim]`
#     4. Shift: Ca shifted forward by one window.
#     5. Concat (Ca + Cb): -> `[batch, n_windows, 2 * compress_rate, head_dim]`
#     6. Gating & Sum: -> `[batch, n_windows, head_dim]`
#   """
#   batch_size, seq_len, _ = hidden_states.shape

#   # [batch, seq_len, emb_dim] -> [batch, seq_len, 2 * head_dim]
#   kv = kv_proj(hidden_states)
#   # [batch, seq_len, emb_dim] -> [batch, seq_len, 2 * head_dim]
#   gate = gate_proj(hidden_states)

#   usable = (seq_len // compress_rate) * compress_rate
#   chunk_kv = kv[:, :usable]
#   chunk_gate = gate[:, :usable]

#   # Return zero tensor if there are no full windows available for pooling
#   if chunk_kv.shape[1] == 0:
#     return jnp.zeros((batch_size, 0, head_dim), dtype=hidden_states.dtype)

#   n_windows = chunk_kv.shape[1] // compress_rate

#   # Reshape flat sequence into discrete compression windows
#   # -> [batch, n_windows, compress_rate, 2 * head_dim]
#   chunk_kv = chunk_kv.reshape((batch_size, n_windows, compress_rate, 2 * head_dim))
#   chunk_gate = chunk_gate.reshape((batch_size, n_windows, compress_rate, 2 * head_dim)) + position_bias

#   # Split the projections into Ca and Cb components for overlapping
#   # 2x [batch, n_windows, compress_rate, head_dim]
#   a_kv, b_kv = jnp.split(chunk_kv, 2, axis=-1)
#   a_gate, b_gate = jnp.split(chunk_gate, 2, axis=-1)

#   # Shift Ca forward by one window to align with the next Cb
#   a_kv_shifted = jnp.concatenate(
#       [jnp.zeros((batch_size, 1, compress_rate, head_dim), dtype=a_kv.dtype), a_kv[:, :-1]], axis=1
#   )
#   a_gate_shifted = jnp.concatenate(
#       [jnp.full((batch_size, 1, compress_rate, head_dim), -jnp.inf, dtype=a_gate.dtype), a_gate[:, :-1]], axis=1
#   )

#   # Concatenate shifted Ca and unshifted Cb to form the final overlapping window
#   # -> [batch, n_windows, 2 * compress_rate, head_dim]
#   new_kv = jnp.concatenate([a_kv_shifted, b_kv], axis=2)
#   new_gate = jnp.concatenate([a_gate_shifted, b_gate], axis=2)

#   # Apply softmax gating and sum across the overlapping window dimension
#   gate_weights = jax.nn.softmax(new_gate, axis=2).astype(new_kv.dtype)
#   # -> [batch, n_windows, head_dim]
#   compressed = kv_norm(jnp.sum(new_kv * gate_weights, axis=2))

#   return compressed

def csa_overlap_pooling(chunk_kv, chunk_gate, kv_norm, head_dim, prior_kv, prior_gate):
    import jax.numpy as jnp
    
    B, W, C, D = chunk_kv.shape
    
    # 1. Safely handle cache priors 
    # MaxText tracing passes dummy cache vars that may have standard attention shapes.
    # If the size doesn't match our required (B * 1 * 1 * D), it's a dummy var. Fallback to zeros.
    expected_size = B * 1 * 1 * D
    
    if prior_kv is None or prior_kv.size != expected_size:
        prior_kv = jnp.zeros((B, 1, 1, D), dtype=chunk_kv.dtype)
        prior_gate = jnp.zeros((B, 1, 1, D), dtype=chunk_gate.dtype)
    else:
        # Guarantee exactly [B, 1, 1, D]
        prior_kv = prior_kv.reshape((B, 1, 1, D))
        prior_gate = prior_gate.reshape((B, 1, 1, D))

    # 2. Shift priors across Windows
    last_tokens_kv = chunk_kv[:, :-1, -1:]   
    window_priors_kv = jnp.concatenate([prior_kv, last_tokens_kv], axis=1) 
    
    last_tokens_gate = chunk_gate[:, :-1, -1:]
    window_priors_gate = jnp.concatenate([prior_gate, last_tokens_gate], axis=1) 
    
    # 3. Shift the sequence by 1 along the CompressRate dimension (axis=2)
    shifted_kv = jnp.concatenate([window_priors_kv, chunk_kv[:, :, :-1]], axis=2) 
    shifted_gate = jnp.concatenate([window_priors_gate, chunk_gate[:, :, :-1]], axis=2) 
    
    # 4. Pooling Arithmetic
    pool_a = jnp.sum(chunk_kv * chunk_gate, axis=2)      
    pool_b = jnp.sum(shifted_kv * shifted_gate, axis=2)  
    
    overlap = pool_a + pool_b                            
    
    # 5. Separate the Latent KV from RoPE
    compressed = overlap[..., :head_dim]                 # [B, W, 32]
    
    # 6. Apply the projection norm 
    if kv_norm is not None:
        compressed = kv_norm(compressed)
        
    # 7. Extract the next priors (Keep the full D=64 for the autoregressive cache!)
    next_prior_kv = chunk_kv[:, -1:, -1:] 
    next_prior_gate = chunk_gate[:, -1:, -1:]
    
    return compressed, next_prior_kv, next_prior_gate


class BaseDeepseekCompressor(nnx.Module):
  """Shared base class for DeepSeek-V4 long-range attention compressors.

  This module encapsulates the shared infrastructure for both the Heavily Compressed
  Attention (HCA) and Compressed Sparse Attention (CSA) paradigms introduced in DeepSeek-V4.

  Responsibilities:
    1. Initializes and holds the shared Linear projections (kv_proj, gate_proj) used to
       map embeddings into the compressed representation space.
    2. Owns the KV RMSNorm instance applied to the aggregated representations.
    3. Manages common hyperparameter properties (compress_rate, head_dim, dtype).
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      proj_multiplier: int,
      kernel_init: Any = nnx.initializers.normal(stddev=0.02),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.compress_rate = compress_ratio
    self.head_dim = config.head_dim
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.model_mode = model_mode
    self.rngs = rngs

    proj_dim = proj_multiplier * self.head_dim

    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=proj_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.position_bias = nnx.Param(jnp.zeros((self.compress_rate, proj_dim), dtype=self.weight_dtype))

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.rotary_emb = rotary_embedding


class DeepseekV4HCACompressor(BaseDeepseekCompressor):
  """Heavily Compressed Attention compressor.

  Compresses every `compress_rate_hca` source tokens into a single compressed KV entry
  using closed, non-overlapping windows. RoPE is applied to the final compressed token.

  Shape Transformations:
    1. Projections: `[batch, seq, emb_dim]` -> `[batch, seq, head_dim]`
    2. Chunking: -> `[batch, n_windows, compress_rate, head_dim]`
    3. Gating & Sum: -> `[batch, n_windows, head_dim]`
    4. RoPE: -> `[batch, n_windows, head_dim]`
    5. Output Expand: -> `[batch, n_windows, 1, head_dim]`
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      kernel_init: Any = nnx.initializers.normal(stddev=0.02),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the HCA Compressor.

    Args:
      config: The configuration object for the model containing architecture hyperparameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_hca) that determines the
        window size for heavily compressed attention.
      rotary_embedding: A rotary embedding instance used to inject positional information into the
        final compressed representations.
      kernel_init: The initializer used for the kernel weights.
      quant: Optional quantization scheme.
      model_mode: The operational mode (e.g., "train", "prefill").
      rngs: An optional Rngs instance for stochastic initializations or dropout.
    """
    super().__init__(config, compress_ratio, rotary_embedding, 1, kernel_init, quant, model_mode, rngs)

  def __call__(
      self,
      hidden_states: Array,
      q_normed: Array,
      position_ids: Array,
      model_mode: str,
      cache: Optional[Any] = None,
  ) -> Tuple[Array, Array]:
    """Forward pass for the HCA compressor.

    Args:
      hidden_states: Input token embeddings. Shape: `[batch, seq_len, emb_dim]`.
      q_normed: Latent query representation (unused in HCA).
      position_ids: Absolute token positions. Shape: `[batch, seq_len]`.

    Returns:
      compressed_kv: The pooled KV tensors. Shape: `[batch, n_windows, 1, head_dim]`.
      compressed_causal_mask: Causal mask preventing queries from seeing future blocks.
                              Shape: `[batch, 1, seq_len, n_windows]`.
    """
    batch_size, seq_len, _ = hidden_states.shape

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # --- AUTOREGRESSIVE DELEGATION ---
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:
      # Expand dims to match [B, S, H, D] format for the cache
      kv_exp = jnp.expand_dims(kv, 2)
      gate_exp = jnp.expand_dims(gate, 2)

      # Pad the batch dimension so [1, 1, 4, 64] -> [4, 1, 4, 64]
      target_batch = cache.batch
      if kv_exp.shape[0] != target_batch:
          if kv_exp.shape[0] < target_batch:
              pad_amt = target_batch - kv_exp.shape[0]
              kv_exp = jnp.pad(kv_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
              gate_exp = jnp.pad(gate_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
          else:
              kv_exp = kv_exp[:target_batch]
              gate_exp = gate_exp[:target_batch]
      
      cached_prefill, cached_ar = cache(
          key=kv_exp, value=kv_exp, gate=gate_exp, decoder_segment_ids=None, model_mode=model_mode
      )

      comp_dim = self.head_dim
      compressed_full = jnp.concatenate([cached_prefill[0], cached_ar[0]], axis=1)
      
      # Restore back to original batch_size and strip RoPE/dummy head
      compressed = compressed_full[:batch_size, :, 0, :comp_dim]
      
      compressed_kv = jnp.expand_dims(compressed, 2) # [B, N, 1, D]
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_kv.shape[1]), dtype=self.dtype)
      
      return compressed_kv, compressed_mask

    # --- PREFILL CHUNKING & PRIMING ---
    usable = (seq_len // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]
    first_window_position = position_ids[:, 0:1]

    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate
      chunk_kv = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, -1))
      chunk_gate = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, -1)) + self.position_bias.value

      gate_weights = jax.nn.softmax(chunk_gate, axis=2).astype(chunk_kv.dtype)
      compressed = self.kv_norm(jnp.sum(chunk_kv * gate_weights, axis=2))
      positions = jnp.arange(n_windows) * self.compress_rate + first_window_position
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
    else:
      compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=self.dtype)

    compressed_kv = jnp.expand_dims(compressed, axis=2)
    compressed_len = compressed_kv.shape[1]

    # --- PREFILL CACHE PRIMING ---
    if cache is not None:
      remainder = seq_len % self.compress_rate
      if remainder > 0:
        leftover_kv = kv[:, usable:]
        leftover_gate = gate[:, usable:]
        pad_len = self.compress_rate - remainder
        padded_kv = jnp.expand_dims(jnp.pad(leftover_kv, ((0, 0), (0, pad_len), (0, 0))), 2)
        padded_gate = jnp.expand_dims(jnp.pad(leftover_gate, ((0, 0), (0, pad_len), (0, 0))), 2)
        cache.leftover_buffer_kv.set_value(padded_kv)
        cache.leftover_buffer_gate.set_value(padded_gate)
        cache.accumulator_index.set_value(jnp.full((batch_size, 1), remainder, dtype=jnp.int32))

      if compressed_len > 0:
        cache_key_var = cache.cached_prefill_key
        update_blocks = compressed_kv
        operand_shape = cache_key_var.get_value().shape
        if update_blocks.shape[2] < operand_shape[2]:
            update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, operand_shape[2] - update_blocks.shape[2]), (0, 0)))
        if update_blocks.shape[3] < operand_shape[3]:
            update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, 0), (0, operand_shape[3] - update_blocks.shape[3])))
        
        update_blocks = update_blocks[:, :operand_shape[1], ...]
        cache_key_var.set_value(
            jax.lax.dynamic_update_slice_in_dim(cache_key_var.get_value(), update_blocks, 0, axis=1)
        )
        cache.entry_count.set_value(jnp.full((batch_size, 1), compressed_len, dtype=jnp.int32))

    if seq_len == 1 or compressed_len == 0:
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_len), dtype=self.dtype)
      return compressed_kv, compressed_mask

    entry_indices = jnp.arange(compressed_len)
    causal_threshold = (position_ids + 1) // self.compress_rate
    future_mask = entry_indices[None, None, None, :] >= jnp.expand_dims(causal_threshold, axis=(1, 3))
    compressed_causal_mask = jnp.where(future_mask, DEFAULT_MASK_VALUE, 0.0).astype(self.dtype)

    return compressed_kv, compressed_causal_mask


class DeepseekV4Indexer(nnx.Module):
  """Indexer module for Compressed Sparse Attention (DeepSeek-V4 paper §2.3.1).

  Evaluates query representations against compressed KV blocks to identify the top-k
  most relevant blocks to attend to.

  Shape Transformations:
    1. Pool KV: `[batch, seq, emb_dim]` -> `[batch, n_windows, index_head_dim]`
    2. Broadcast KV: -> `[batch, index_n_heads, n_windows, index_head_dim]`
    3. Project Q: `[batch, seq, emb_dim]` -> `[batch, index_n_heads, seq, index_head_dim]`
    4. Einsum Q*KV: -> `[batch, index_n_heads, seq, n_windows]`
    5. Project Weights: `[batch, seq, emb_dim]` -> `[batch, seq, index_n_heads]`
    6. Combine Scores: -> `[batch, seq, n_windows]`
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      kernel_init: Any = nnx.initializers.normal(stddev=0.02),
      quant: Optional[Quant] = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the Indexer for CSA.

    Args:
      config: Model configuration containing indexer parameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_csa).
      rotary_embedding: Rotary embedding instance for injecting position info into index representations.
      kernel_init: Weight initializer for the indexer projections.
      quant: Optional quantization scheme.
      rngs: Optional random state initialization.
    """
    self.config = config
    self.compress_rate = compress_ratio
    self.index_n_heads = config.indexer_n_heads
    self.index_head_dim = config.indexer_head_dim
    self.index_topk = config.indexer_topk
    self.softmax_scale = self.index_head_dim**-0.5
    self.weights_scaling = self.index_n_heads**-0.5
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.rngs = rngs

    self.q_proj = DenseGeneral(
        in_features_shape=config.q_lora_rank,
        out_features_shape=self.index_n_heads * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("q_lora", "indexer_q"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.kv_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    self.gate_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=2 * self.index_head_dim,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.position_bias = nnx.Param(jnp.zeros((self.compress_rate, 2 * self.index_head_dim), dtype=self.weight_dtype))

    self.kv_norm = RMSNorm(
        num_features=self.index_head_dim,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.weights_proj = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=self.index_n_heads,
        axis=-1,
        kernel_init=kernel_init,
        kernel_axes=("embed", "indexer_weights"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=quant,
        matmul_precision=config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )

    self.rotary_emb = rotary_embedding

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache: Optional[Any] = None,
  ) -> Array:
    batch_size, seq_len, _ = hidden_states.shape

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # --- AUTOREGRESSIVE DELEGATION ---
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:
      kv_exp = jnp.expand_dims(kv, 2)
      gate_exp = jnp.expand_dims(gate, 2)

      # --- STRICT BATCH ALIGNMENT ---
      target_batch = cache.batch
      if kv_exp.shape[0] != target_batch:
          if kv_exp.shape[0] < target_batch:
              pad_amt = target_batch - kv_exp.shape[0]
              kv_exp = jnp.pad(kv_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
              gate_exp = jnp.pad(gate_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
          else:
              kv_exp = kv_exp[:target_batch]
              gate_exp = gate_exp[:target_batch]
      # ------------------------------

      cached_prefill, cached_ar = cache(
          key=kv_exp, value=kv_exp, gate=gate_exp, decoder_segment_ids=None, model_mode=model_mode
      )

      comp_dim = self.index_head_dim
      compressed_full = jnp.concatenate([cached_prefill[0], cached_ar[0]], axis=1)
      
      # Restore back to original batch_size and strip RoPE/dummy head
      compressed = compressed_full[:batch_size, :, 0, :comp_dim]
      
      compressed_len = compressed.shape[1]
    
    # --- PREFILL CHUNKING & PRIMING ---
    else:
      usable = (seq_len // self.compress_rate) * self.compress_rate
      chunk_kv = kv[:, :usable]
      chunk_gate = gate[:, :usable]
      
      # Extract staggered overlap states if cache is available
      if cache is not None:
        # Convert from [batch, compress_rate, 1, head_dim] -> [batch, 1, compress_rate, head_dim]
        prior_kv = jnp.transpose(cache.overlap_kv.get_value(), (0, 2, 1, 3))
        prior_gate = jnp.transpose(cache.overlap_gate.get_value(), (0, 2, 1, 3))
      else:
        prior_kv, prior_gate = None, None

      if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        chunk_kv_reshaped = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, -1))
        chunk_gate_reshaped = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, -1)) + self.position_bias.value

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.index_head_dim, prior_kv, prior_gate
        )
        compressed_len = compressed.shape[1]
        
        positions = jnp.arange(compressed_len) * self.compress_rate + position_ids[:, 0:1]
        compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
      else:
        compressed = jnp.zeros((batch_size, 0, self.index_head_dim), dtype=self.dtype)
        compressed_len = 0
        next_prior_kv = prior_kv
        next_prior_gate = prior_gate

      # Prefill Cache Insertion
      if cache is not None:
        remainder = seq_len % self.compress_rate
        if remainder > 0:
          leftover_kv = kv[:, usable:]
          leftover_gate = gate[:, usable:]
          pad_len = self.compress_rate - remainder
          padded_kv = jnp.expand_dims(jnp.pad(leftover_kv, ((0, 0), (0, pad_len), (0, 0))), 2)
          padded_gate = jnp.expand_dims(jnp.pad(leftover_gate, ((0, 0), (0, pad_len), (0, 0))), 2)
          cache.leftover_buffer_kv.set_value(padded_kv)
          cache.leftover_buffer_gate.set_value(padded_gate)
          cache.accumulator_index.set_value(jnp.full((batch_size, 1), remainder, dtype=jnp.int32))
        
        if compressed_len > 0:
          cache_key_var = cache.cached_prefill_key
          update_blocks = jnp.expand_dims(compressed, 2)
          operand_shape = cache_key_var.get_value().shape
          if update_blocks.shape[2] < operand_shape[2]:
              update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, operand_shape[2] - update_blocks.shape[2]), (0, 0)))
          if update_blocks.shape[3] < operand_shape[3]:
              update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, 0), (0, operand_shape[3] - update_blocks.shape[3])))
          
          update_blocks = update_blocks[:, :operand_shape[1], ...]
          cache_key_var.set_value(
              jax.lax.dynamic_update_slice_in_dim(cache_key_var.get_value(), update_blocks, 0, axis=1)
          )
          cache.entry_count.set_value(jnp.full((batch_size, 1), compressed_len, dtype=jnp.int32))
          
          # Save the new trailing Ca slices to the overlap registers!
          # Convert from [batch, 1, compress_rate, head_dim] -> [batch, compress_rate, 1, head_dim]
          cache.overlap_kv.set_value(jnp.transpose(next_prior_kv, (0, 2, 1, 3)))
          cache.overlap_gate.set_value(jnp.transpose(next_prior_gate, (0, 2, 1, 3)))

    if compressed_len == 0:
      return jnp.zeros((batch_size, seq_len, min(self.index_topk, compressed_len)), dtype=jnp.int32)

    # --- TOP-K ROUTING MATH (Executes in both Prefill and AR) ---
    compressed_kv = jnp.expand_dims(compressed, axis=1)
    compressed_kv = jnp.broadcast_to(compressed_kv, (batch_size, self.index_n_heads, compressed_len, self.index_head_dim))

    q = self.q_proj(q_latent).reshape((batch_size, seq_len, self.index_n_heads, self.index_head_dim))
    q = jnp.transpose(q, (0, 2, 1, 3))
    q = self.rotary_emb(q, position_ids, unsqueeze_dim=1)

    q = q.astype(jnp.float32)
    compressed_kv = compressed_kv.astype(jnp.float32)

    scores = jnp.einsum("bhsd,bhwd->bhsw", q, compressed_kv)
    scores = jax.nn.relu(scores) * self.softmax_scale
    weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling
    index_scores = jnp.einsum("bhsw,bsh->bsw", scores, weights)

    k = min(self.index_topk, compressed_len)
    causal_threshold = (position_ids + 1) // self.compress_rate
    entry_indices = jnp.arange(compressed_len)
    future_mask = entry_indices[None, None, :] >= jnp.expand_dims(causal_threshold, axis=-1)

    index_scores = jnp.where(future_mask, jnp.full_like(index_scores, -jnp.inf), index_scores)

    if attention_mask is not None:
      index_scores += attention_mask[:, :, :compressed_len]

    top_k_indices = jax.lax.top_k(index_scores, k)[1]
    invalid = top_k_indices >= jnp.expand_dims(causal_threshold, axis=-1)
    return jnp.where(invalid, jnp.full_like(top_k_indices, -1), top_k_indices)


class DeepseekV4CSACompressor(BaseDeepseekCompressor):
  """Compressed Sparse Attention compressor (DeepSeek-V4 paper §2.3.1).

  Uses overlapping windows to compress local sequence contexts into sparse blocks,
  which are dynamically selected by the Indexer for long-range sparse attention.

  Shape Transformations:
    1. Pool KV (via overlap util): `[batch, seq, emb_dim]` -> `[batch, n_windows, head_dim]`
    2. RoPE: -> `[batch, n_windows, head_dim]`
    3. Expand Output: -> `[batch, n_windows, 1, head_dim]`
    4. Causal & Top-K Masking: -> `[batch, 1, seq, n_windows]`
  """

  def __init__(
      self,
      config: Any,
      compress_ratio: int,
      rotary_embedding: Any,
      indexer_rotary_embedding: Any = None,
      kernel_init: Any = nnx.initializers.normal(stddev=0.02),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the CSA Compressor.

    Args:
      config: The configuration object for the model containing architecture hyperparameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_csa) that determines the
        stride size for pooling representations in sparse attention.
      rotary_embedding: A rotary embedding instance used to inject positional information into the
        final compressed representations.
      kernel_init: The initializer used for the kernel weights.
      quant: Optional quantization scheme.
      model_mode: The operational mode (e.g., "train", "prefill").
      rngs: An optional Rngs instance for stochastic initializations or dropout.
    """
    super().__init__(config, compress_ratio, rotary_embedding, 2, kernel_init, quant, model_mode, rngs)

    self.indexer = DeepseekV4Indexer(
        config=config,
        compress_ratio=compress_ratio,
        rotary_embedding=indexer_rotary_embedding if indexer_rotary_embedding is not None else rotary_embedding,
        kernel_init=kernel_init,
        quant=quant,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache: Optional[Any] = None,
      indexer_cache: Optional[Any] = None,
  ) -> Tuple[Array, Array]:
    batch_size, seq_len, _ = hidden_states.shape

    # 1. ALWAYS Run Indexer (It fetches its own history inside AR)
    top_k_indices = self.indexer(
        hidden_states, q_latent, position_ids, attention_mask, model_mode, indexer_cache
    )

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    # --- AUTOREGRESSIVE DELEGATION ---
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:
      kv_exp = jnp.expand_dims(kv, 2)
      gate_exp = jnp.expand_dims(gate, 2)

      # --- STRICT BATCH ALIGNMENT ---
      target_batch = cache.batch
      if kv_exp.shape[0] != target_batch:
          if kv_exp.shape[0] < target_batch:
              pad_amt = target_batch - kv_exp.shape[0]
              kv_exp = jnp.pad(kv_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
              gate_exp = jnp.pad(gate_exp, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
          else:
              kv_exp = kv_exp[:target_batch]
              gate_exp = gate_exp[:target_batch]
      # ------------------------------

      cached_prefill, cached_ar = cache(
          key=kv_exp, value=kv_exp, gate=gate_exp, decoder_segment_ids=None, model_mode=model_mode
      )

      comp_dim = self.head_dim
      compressed_full = jnp.concatenate([cached_prefill[0], cached_ar[0]], axis=1)
      
      # Restore back to original batch_size and strip RoPE/dummy head
      compressed = compressed_full[:batch_size, :, 0, :comp_dim]
      
      compressed_kv = jnp.expand_dims(compressed, 2)
      compressed_len = compressed_kv.shape[1]
    
    # --- PREFILL CHUNKING & PRIMING ---
    else:
      usable = (seq_len // self.compress_rate) * self.compress_rate
      chunk_kv = kv[:, :usable]
      chunk_gate = gate[:, :usable]
      
      if cache is not None:
        # Convert from [batch, compress_rate, 1, head_dim] -> [batch, 1, compress_rate, head_dim]
        prior_kv = jnp.transpose(cache.overlap_kv.get_value(), (0, 2, 1, 3))
        prior_gate = jnp.transpose(cache.overlap_gate.get_value(), (0, 2, 1, 3))
      else:
        prior_kv, prior_gate = None, None

      if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // self.compress_rate
        chunk_kv_reshaped = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, -1))
        chunk_gate_reshaped = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, -1)) + self.position_bias.value

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.head_dim, prior_kv, prior_gate
        )
        compressed_len = compressed.shape[1]
        
        positions = jnp.arange(compressed_len) * self.compress_rate + position_ids[:, 0:1]
        compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
      else:
        compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=self.dtype)
        compressed_len = 0
        next_prior_kv = prior_kv
        next_prior_gate = prior_gate

      compressed_kv = jnp.expand_dims(compressed, 2)

      if cache is not None:
        remainder = seq_len % self.compress_rate
        if remainder > 0:
          leftover_kv = kv[:, usable:]
          leftover_gate = gate[:, usable:]
          pad_len = self.compress_rate - remainder
          padded_kv = jnp.expand_dims(jnp.pad(leftover_kv, ((0, 0), (0, pad_len), (0, 0))), 2)
          padded_gate = jnp.expand_dims(jnp.pad(leftover_gate, ((0, 0), (0, pad_len), (0, 0))), 2)
          cache.leftover_buffer_kv.set_value(padded_kv)
          cache.leftover_buffer_gate.set_value(padded_gate)
          cache.accumulator_index.set_value(jnp.full((batch_size, 1), remainder, dtype=jnp.int32))
        
        if compressed_len > 0:
          cache_key_var = cache.cached_prefill_key
          update_blocks = compressed_kv
          operand_shape = cache_key_var.get_value().shape
          if update_blocks.shape[2] < operand_shape[2]:
              update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, operand_shape[2] - update_blocks.shape[2]), (0, 0)))
          if update_blocks.shape[3] < operand_shape[3]:
              update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, 0), (0, operand_shape[3] - update_blocks.shape[3])))
          
          update_blocks = update_blocks[:, :operand_shape[1], ...]
          cache_key_var.set_value(
              jax.lax.dynamic_update_slice_in_dim(cache_key_var.get_value(), update_blocks, 0, axis=1)
          )
          cache.entry_count.set_value(jnp.full((batch_size, 1), compressed_len, dtype=jnp.int32))

          cache.overlap_kv.set_value(jnp.transpose(next_prior_kv, (0, 2, 1, 3)))
          cache.overlap_gate.set_value(jnp.transpose(next_prior_gate, (0, 2, 1, 3)))

    if compressed_len == 0:
      return compressed_kv, jnp.zeros((batch_size, 1, seq_len, 0), dtype=self.dtype)

    # 3. Apply Dynamic Masking Logic
    k = top_k_indices.shape[-1]
    if k > 0:
      valid = top_k_indices >= 0
      entry_indices = jnp.arange(compressed_len)[None, None, :]
      is_in_topk = jnp.expand_dims(top_k_indices, axis=-1) == entry_indices[None, ...]
      is_valid_and_in_topk = is_in_topk & jnp.expand_dims(valid, axis=-1)

      is_selected = jnp.any(is_valid_and_in_topk, axis=2)
      is_selected = jnp.expand_dims(is_selected, axis=1)

      compressed_mask = jnp.where(is_selected, 0.0, DEFAULT_MASK_VALUE).astype(self.dtype)
    else:
      compressed_mask = jnp.full((batch_size, 1, seq_len, compressed_len), DEFAULT_MASK_VALUE, dtype=self.dtype)

    return compressed_kv, compressed_mask


class CompressedAttention(Attention):
  """Compressed Attention layer (DeepSeek-V4).

  Wrapper around standard Attention that integrates HCA or CSA compressors based
  on the layer type. It compresses the inputs, concatenates the resulting sparse
  blocks to the standard KV sequence, and injects the compressor's block-masking
  matrix directly into the underlying attention operator.
  """

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: Tuple,
      inputs_kv_shape: Tuple,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      float32_qk_product: bool = False,
      float32_logits: bool = False,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.COMPRESSED,
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,
      # Compression Specific Parameters:
      q_lora_rank: int = 1536,
      compress_ratio: int = 0,
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
      **kwargs,
  ):
    """Inherits all standard Attention hyperparameters and selectively instantiates
    an underlying HCA or CSA compressor based on the provided `compress_ratio`.

    Highlights of DeepSeek-V4 attention integration:
    - Shared-KV: The layer supports decoupling Q and KV heads for heavy compression.
    - MQA: Multi-Query Attention used alongside heavy KV compression.
    - 3 Different Attention Modes: Sliding Window (prefix), HCA (128x), and CSA (4x).
    - Dual RoPE Theta: Uses 10000 for standard uncompressed tokens and 160000 for compressed.

    Args:
      (See maxtext.layers.attentions.Attention for standard attention arguments)
      q_lora_rank: The rank for the LoRA projection in the compressed query.
      compress_ratio: The compression ratio (0, 4, or 128) for the compressor.
    """
    self.q_lora_rank = q_lora_rank
    self.compress_ratio = compress_ratio

    # Determine the correct underlying attention type based on the compress_ratio
    if self.compress_ratio == 0:
      attention_type = AttentionType.LOCAL_SLIDING

    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
        float32_qk_product=float32_qk_product,
        float32_logits=float32_logits,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        use_ragged_attention=use_ragged_attention,
        ragged_block_size=ragged_block_size,
        use_qk_norm=use_qk_norm,
        query_pre_attn_scalar=query_pre_attn_scalar,
        use_bias_in_projections=use_bias_in_projections,
        name=name,
        rngs=rngs,
        **kwargs,
    )

    # DeepSeek-V4 uses a mathematical attention sink (a learnable scalar per-head added to the
    # attention logits prior to softmax, rather than a physical key/value token). We unconditionally
    # initialize it here, overriding the base Attention class which disables it by default.
    self.sinks = nnx.data(nnx.Param(jnp.zeros((self.num_query_heads,), dtype=self.weight_dtype), sharding=(None,)))

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the compressed projections and Unweighted RMSNorms."""
    # Query Projection Modules
    self.wq_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.q_lora_rank,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "q_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.q_norm = RMSNorm(
        num_features=self.q_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.q_up_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        with_scale=False,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Key-Value Projection Modules
    self.wkv = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    self.kv_norm = RMSNorm(
        num_features=self.head_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Override the base rotary embedding with the correct theta for this layer.
    # CSA / HCA layers use compressed_rope_max_timescale (160000).
    # Sliding window prefix layers use rope_max_timescale (10000).
    rope_theta = self.config.compressed_rope_max_timescale if self.compress_ratio > 0 else self.config.rope_max_timescale
    self.rotary_embedding = DeepSeekV4RotaryEmbedding(
        head_dim=self.config.head_dim,
        partial_rotary_factor=self.config.qk_rope_head_dim / self.config.head_dim,
        rope_theta=rope_theta,
        fprop_dtype=self.dtype,
    )

    if self.compress_ratio == 4:
      self.indexer_rotary_embedding = DeepSeekV4RotaryEmbedding(
          head_dim=self.config.indexer_head_dim,
          partial_rotary_factor=1.0, 
          rope_theta=rope_theta,
          fprop_dtype=self.dtype,
      )
    else:
      self.indexer_rotary_embedding = None

    if self.compress_ratio > 4:
      self.hca_compressor = DeepseekV4HCACompressor(
          config=self.config,
          compress_ratio=self.compress_ratio,
          rotary_embedding=self.rotary_embedding,
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )
    elif self.compress_ratio == 4:
      self.csa_compressor = DeepseekV4CSACompressor(
          config=self.config,
          compress_ratio=self.compress_ratio,
          rotary_embedding=self.rotary_embedding,
          indexer_rotary_embedding=self.indexer_rotary_embedding,
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
      )

    # Set softmax scaling. DeepSeek-V4 natively uses standard scaling.
    self.softmax_scale = self.head_dim**-0.5

    # Output Projections (Two-Step Grouped Linear)
    in_features_per_group = (self.num_query_heads * self.head_dim) // self.config.o_groups
    o_a_out_features = self.config.o_groups * self.config.o_lora_rank

    self.o_a_proj = DeepSeekV4GroupedLinear(
        in_features_per_group=in_features_per_group,
        out_features=o_a_out_features,
        n_groups=self.config.o_groups,
        kernel_init=self.kernel_init,
        kernel_axes=("o_groups", "q_heads", "o_lora_up_proj"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    self.o_b_proj = DenseGeneral(
        in_features_shape=o_a_out_features,
        out_features_shape=inputs_q_shape[-1],
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("o_lora_up_proj", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    if self.model_mode != MODEL_MODE_TRAIN and self.compress_ratio > 0:
      batch_size = inputs_q_shape[0]
      max_prefill_comp = max(1, self.max_prefill_predict_length // self.compress_ratio)
      max_target_comp = max(max_prefill_comp + 1, self.max_target_length // self.compress_ratio)

      comp_head_dim = self.head_dim if self.compress_ratio > 4 else 2 * self.head_dim

      self.compressor_cache = kvcache.KVCache(
          max_prefill_length=max_prefill_comp,
          max_target_length=max_target_comp,
          batch=batch_size,
          key_seq_len=max_target_comp,
          value_seq_len=max_target_comp,
          key_heads=1,
          value_heads=1,
          key_head_size=comp_head_dim,
          value_head_size=comp_head_dim,
          dtype=self.dtype,
          model_mode=self.model_mode,
          is_deepseek_v4=True,
          compress_rate=self.compress_ratio,
          rngs=self.rngs,
      )
    else:
      self.compressor_cache = None

    if self.model_mode != MODEL_MODE_TRAIN and self.compress_ratio == 4:
      self.indexer_cache = kvcache.KVCache(
          max_prefill_length=max_prefill_comp,
          max_target_length=max_target_comp,
          batch=batch_size,
          key_seq_len=max_prefill_comp,
          value_seq_len=max_prefill_comp,
          key_heads=1,
          value_heads=1,
          key_head_size=2 * self.config.indexer_head_dim,
          value_head_size=2 * self.config.indexer_head_dim,
          dtype=self.dtype,
          model_mode=self.model_mode,
          is_deepseek_v4=True,
          compress_rate=self.compress_ratio,
          is_indexer=True,
          rngs=self.rngs,
      )
    else:
      self.indexer_cache = None

  @property
  def out_head_dim(self) -> int:
    """Returns the head dimension used prior to the output projection."""
    return self.head_dim

  def compressed_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode) -> Array:
    """Query projection for Compressed Attention.

    Args:
      inputs_q: The query hidden states. Shape: `[batch, seq_len, emb_dim]`.
      inputs_positions: The token positions, used for Rotary Positional Embeddings (RoPE).
      model_mode: The execution mode (e.g., 'train', 'prefill', 'autoregressive').

    Returns:
      The projected and RoPE-applied query tensor.
      Shape: `[batch, seq_len, num_query_heads, head_dim]`.

    Shape Transformations:
      1. Project `inputs_q` [batch, seq_len, emb_dim] to latent space [batch, seq_len, q_lora_rank].
      2. Normalize latent space via RMSNorm.
      3. Up-project to full head dimension [batch, seq_len, num_query_heads, head_dim].
      4. Apply Unweighted RMSNorm over the `head_dim` axis.
      5. Apply Rotary Positional Embeddings over the entire vector.
      6. Scale by 1/sqrt(head_dim) for numerical stability during attention computation.
    """
    # [batch, seq_len, emb_dim] -> [batch, seq_len, q_lora_rank]
    q_latent = self.wq_a(inputs_q)
    q_normed = self.q_norm(q_latent)

    # [batch, seq_len, q_lora_rank] -> [batch, seq_len, num_query_heads, head_dim]
    q_up = self.wq_b(q_normed)

    q_up_normed = self.q_up_norm(q_up)

    # -> [batch, seq_len, num_query_heads, head_dim]
    try:
        q_out = self.rotary_embedding(q_up_normed, inputs_positions, unsqueeze_dim=-2)
    except TypeError:
        # If the embedding rejects the kwarg, reshape manually before passing it in, 
        # or rely on the embedding's internal broadcasting.
        q_out = self.rotary_embedding(q_up_normed, inputs_positions)

    # Scale queries by 1/sqrt(head_dim) prior to attention to prevent softmax saturation
    # -> [batch, seq_len, num_query_heads, head_dim]
    q_out = q_out * self.softmax_scale

    return q_out, q_normed

  def compressed_kv_projection(self, inputs_kv: Array, inputs_positions: Array, model_mode) -> Tuple[Array, Array]:
    """KV projection for Compressed Attention.

    Args:
      inputs_kv: The key/value hidden states. Shape: `[batch, seq_len, emb_dim]`.
      inputs_positions: The token positions, used for Rotary Positional Embeddings (RoPE).
      model_mode: The execution mode (e.g., 'train', 'prefill', 'autoregressive').

    Returns:
      A tuple of (key, value) tensors.
      Shapes: Both are `[batch, seq_len, num_kv_heads, head_dim]`.

    Shape Transformations:
      1. Project `inputs_kv` [batch, seq_len, emb_dim] to full head dimension [batch, seq_len, num_kv_heads, head_dim].
      2. Apply Unweighted RMSNorm over the `head_dim` axis.
      3. Apply Rotary Positional Embeddings over the entire vector.
      4. Note: Compressed caching will append additional slices downstream, but the base
         projections yield symmetrically shaped key and value vectors here.
    """
    # [batch, seq_len, emb_dim] -> [batch, seq_len, num_kv_heads, head_dim]
    kv_up = self.wkv(inputs_kv)

    kv_up_normed = self.kv_norm(kv_up)

    try:
        kv_out = self.rotary_embedding(kv_up_normed, inputs_positions, unsqueeze_dim=-2)
    except TypeError:
        kv_out = self.rotary_embedding(kv_up_normed, inputs_positions)

    return kv_out, kv_out

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      decoder_segment_ids: Array,
      inputs_positions: Array,
      deterministic: bool,
      model_mode: str = MODEL_MODE_TRAIN,
      **kwargs,
  ) -> Array:
    """Forward pass for Compressed Attention.

    Args:
      inputs_q: Query input. Shape: `[batch, q_length, embed_dim]`.
      inputs_kv: KV input. Shape: `[batch, kv_length, embed_dim]`.
      decoder_segment_pos: Segment IDs for masking.
      inputs_positions: Positions for rotary embeddings.
      deterministic: Disables dropout if set to True.
      model_mode: 'train', 'prefill', or 'autoregressive'.

    Returns:
      A tensor of shape `[batch, length, embed_dim]` containing the attended outputs.

    Shape Transformations:
      1. Projections: `[batch, len, emb_dim]` -> Q/K/V: `[batch, len, num_heads, head_dim]`.
      2. Dot Product: Attention over Q, K, V -> `[batch, q_length, num_query_heads, head_dim]`.
      3. Reverse RoPE: Applied in-place on the `head_dim` axis to undo V rotation.
      4. Group Reshape: `[batch, len, q_heads, head_dim]` -> `[batch, len, o_groups, in_features_per_group]`.
      5. Grouped Linear (o_a_proj): -> `[batch, q_length, o_groups, out_features_per_group]`.
      6. Flatten & Dense (o_b_proj): -> `[batch, q_length, emb_dim]`.
    """
    kv_cache = kwargs.get("kv_cache", None)
    attention_metadata = kwargs.get("attention_metadata", None)

    q, q_normed = self.compressed_query_projection(inputs_q, inputs_positions, model_mode)
    k, v = self.compressed_kv_projection(inputs_kv, inputs_positions, model_mode)

    current_kv_cache = kv_cache

    # 1. Update the Local (Sliding Window) KV Cache with the uncompressed tokens
    if model_mode != MODEL_MODE_TRAIN and self.KVCache_0 is not None:
      current_kv_cache = self.update_kv_caches(
          k, v, decoder_segment_ids, model_mode, kwargs.get("previous_chunk", None)
      )

    prefill_kv_cache = current_kv_cache[0] if current_kv_cache is not None else None
    ar_kv_cache = current_kv_cache[1] if current_kv_cache is not None else None

    # Generate compressed representations based on the configured layer type
    compressed_kv = None
    compressed_mask = None
    compressed_segment_mask = None
    
    if decoder_segment_ids is not None and self.compress_ratio > 0:
      segment_mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      segment_mask_additive = jnp.where(segment_mask, 0.0, DEFAULT_MASK_VALUE)
      compress_rate = self.compress_ratio
      compressed_segment_mask = segment_mask_additive[:, :, ::compress_rate]

    # Route to the appropriate compressor
    if self.compress_ratio > 4:
      compressed_kv, compressed_mask = self.hca_compressor(
        inputs_kv, q_normed, inputs_positions, model_mode, self.compressor_cache
      )
    elif self.compress_ratio == 4:
      compressed_kv, compressed_mask = self.csa_compressor(
          inputs_kv, q_normed, inputs_positions, compressed_segment_mask, model_mode, self.compressor_cache, self.indexer_cache
      )

    # Apply segment masking to the compressed blocks
    if compressed_segment_mask is not None and compressed_mask is not None:
      compressed_mask = compressed_mask + jnp.expand_dims(
          compressed_segment_mask[:, :, : compressed_mask.shape[-1]], axis=1
      )

    # Extend local KV tensors with the compressed blocks
    if compressed_kv is not None:
      k = jnp.concatenate([k, compressed_kv], axis=1)
      v = jnp.concatenate([v, compressed_kv], axis=1)

    if compressed_mask is not None:
      compressed_mask = jnp.expand_dims(compressed_mask, axis=2)

    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      q = q * self.query_pre_attn_scalar

    # Compute Attention (Now safely passing kv_cache so the kernel doesn't assert!)
    attn_out = self.attention_op(
        q,
        k,
        v,
        decoder_segment_ids,
        inputs_positions,
        model_mode,
        sinks=self.sinks.value if self.sinks is not None else None,
        compressed_mask=compressed_mask,
        cached_values=current_kv_cache,
    )

    # Reverse RoPE on Values
    try:
        attn_out = self.rotary_embedding(attn_out, inputs_positions, unsqueeze_dim=-2, reverse=True)
    except TypeError:
        if hasattr(self.rotary_embedding, 'reverse'):
             attn_out = self.rotary_embedding(attn_out, inputs_positions, reverse=True)

    # Project outputs through Grouped Linear layers
    b, s, h, d = attn_out.shape
    grouped_out = attn_out.reshape(b, s, self.config.o_groups, (h * d) // self.config.o_groups)
    grouped_out = self.o_a_proj(grouped_out)
    grouped_flat = grouped_out.reshape(b, s, -1)
    final_out = self.o_b_proj(grouped_flat)

    # Return the Tuple expected by the transformer block
    return final_out, current_kv_cache


def compressed_attention(
    *,
    config: Config,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_target_length: int,
    mesh: Mesh,
    attention_kernel: str,
    inputs_q_shape: Tuple,
    inputs_kv_shape: Tuple,
    dtype: DType = jnp.float32,
    weight_dtype: DType = jnp.float32,
    max_prefill_predict_length: int = -1,
    dropout_rate: float = 0.0,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
    float32_qk_product: bool = False,
    float32_logits: bool = False,
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.COMPRESSED,
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,
    q_lora_rank: int = 1536,
    name: str | None = None,
):
  """Wrapper to create the CompressedAttention linen module."""
  return nnx_wrappers.to_linen(
      CompressedAttention,
      config=config,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      max_target_length=max_target_length,
      mesh=mesh,
      attention_kernel=attention_kernel,
      inputs_q_shape=inputs_q_shape,
      inputs_kv_shape=inputs_kv_shape,
      dtype=dtype,
      weight_dtype=weight_dtype,
      max_prefill_predict_length=max_prefill_predict_length,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      float32_qk_product=float32_qk_product,
      float32_logits=float32_logits,
      quant=quant,
      kv_quant=kv_quant,
      attention_type=attention_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
      sliding_window_size=sliding_window_size,
      use_ragged_attention=use_ragged_attention,
      ragged_block_size=ragged_block_size,
      use_qk_norm=use_qk_norm,
      query_pre_attn_scalar=query_pre_attn_scalar,
      use_bias_in_projections=use_bias_in_projections,
      q_lora_rank=q_lora_rank,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )
