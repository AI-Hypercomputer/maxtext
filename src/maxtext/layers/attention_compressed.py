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


import enum
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from maxtext.utils import max_utils

from flax import nnx

from maxtext.common.common_types import (
    Array,
    Config,
    DType,
    MODEL_MODE_TRAIN,
    MODEL_MODE_AUTOREGRESSIVE,
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
from maxtext.utils.sharding import maybe_shard_with_logical
from maxtext.inference.kvcache import KVQuant
from maxtext.inference import kvcache


class CSAPoolingConfig(enum.IntEnum):
  """Configuration constants for Compressed Sparse Attention (CSA) overlap pooling.

  Attributes:
    OVERLAP_WINDOWS: Number of overlapping prior windows (W) carried forward from cache.
  """

  OVERLAP_WINDOWS = 1


def csa_overlap_pooling(
    chunk_kv: Array,
    chunk_gate: Array,
    kv_norm: Any,
    head_dim: int,
    prior_kv: Optional[Array] = None,
    prior_gate: Optional[Array] = None,
    is_same_doc: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
  """Computes overlapping window pooling for Compressed Sparse Attention (CSA).

  DeepSeek-V4 CSA uses a stride-4, window-8 pooling mechanism where each output block
  aggregates representations over a 2m (8-token) window formed by pairing the trailing
  m (4) tokens of the previous window (Ca) with the leading m (4) tokens of the current
  window (Cb).

  Pipeline:
    1. Split: `[batch, n_windows, compress_rate, 2 * head_dim]` -> 2x `[batch, n_windows, compress_rate, head_dim]`
    2. Shift: Ca shifted forward by one window (prepending cache prior if available).
    3. Concat (Ca + Cb): -> `[batch, n_windows, 2 * compress_rate, head_dim]`
    4. Gating & Sum: -> `[batch, n_windows, head_dim]`

  Args:
    chunk_kv: Input KV projection chunks. Shape: `[batch, n_windows, compress_rate, 2 * head_dim]`.
    chunk_gate: Input gate projection chunks. Shape: `[batch, n_windows, compress_rate, 2 * head_dim]`.
    kv_norm: RMSNorm instance applied to the aggregated representations.
    head_dim: Target head dimension.
    prior_kv: Previous window KV prior from cache (optional).
    prior_gate: Previous window gate prior from cache (optional).
    is_same_doc: Boolean tensor of shape [batch, n_windows] indicating if window i
      belongs to the same document as previous window i - 1. Without sequence packing,
      this is always True. With packing, this is False at document boundaries, causing
      the shifted prior window (Ca) to be masked out to prevent cross-document leakage.

  Returns:
    Tuple of (compressed, next_prior_kv, next_prior_gate):
      - compressed: The pooled overlapping states. Shape: `[batch, n_windows, head_dim]`.
      - next_prior_kv: Updated KV prior for the next window.
      - next_prior_gate: Updated gate prior for the next window.

  Shape Transformations:
    1. Split: `[batch, n_windows, compress_rate, 2 * head_dim]` -> 2x `[batch, n_windows, compress_rate, head_dim]`
    2. Shift: Ca shifted forward by one window (prepending cache prior if available).
    3. Concat (Ca + Cb): -> `[batch, n_windows, 2 * compress_rate, head_dim]`
    4. Gating & Sum: -> `[batch, n_windows, head_dim]`
  """
  # D2 is 2 * head_dim
  B, _, C, D2 = chunk_kv.shape
  # w is the number of overlapping windows carried forward across chunk boundaries (W=1)
  w = int(CSAPoolingConfig.OVERLAP_WINDOWS)
  expected_size = B * w * C * D2

  # 1. Split the projections into Ca and Cb components for overlapping
  # -> 2x [batch, n_windows, compress_rate, head_dim]
  a_kv, b_kv = jnp.split(chunk_kv, 2, axis=-1)
  a_gate, b_gate = jnp.split(chunk_gate, 2, axis=-1)

  # 2. Safely handle cache priors (using OVERLAP_WINDOWS for the prior window count w)
  if prior_kv is None or prior_kv.size != expected_size:
    prior_a_kv = jnp.zeros((B, w, C, head_dim), dtype=a_kv.dtype)
    # Note: Empty gate prior must be -inf so softmax(gate) = 0
    prior_a_gate = jnp.full((B, w, C, head_dim), -jnp.inf, dtype=a_gate.dtype)
  else:
    prior_kv = prior_kv.reshape((B, w, C, D2))
    prior_gate = prior_gate.reshape((B, w, C, D2))
    prior_a_kv, _ = jnp.split(prior_kv, 2, axis=-1)
    prior_a_gate, _ = jnp.split(prior_gate, 2, axis=-1)

    # KVCache initializes with zeros. If it's the very first step,
    # gate must be -inf for softmax to equal 0.
    is_empty = jnp.all(prior_a_kv == 0)
    prior_a_gate = jnp.where(is_empty, -jnp.inf, prior_a_gate)

  # 3. Shift Ca forward by one window
  # We prepend the prior window to the current chunks, and drop the last window of Ca
  a_kv_shifted = jnp.concatenate([prior_a_kv, a_kv[:, :-1]], axis=1)
  a_gate_shifted = jnp.concatenate([prior_a_gate, a_gate[:, :-1]], axis=1)

  if is_same_doc is not None:
    is_same_doc_exp = is_same_doc[:, :, None, None]
    a_kv_shifted = jnp.where(is_same_doc_exp, a_kv_shifted, 0.0)
    a_gate_shifted = jnp.where(is_same_doc_exp, a_gate_shifted, -jnp.inf)

  # 4. Concatenate shifted Ca and unshifted Cb to form the 2m overlapping window
  # -> [batch, n_windows, 2 * compress_rate, head_dim]
  new_kv = jnp.concatenate([a_kv_shifted, b_kv], axis=2)
  new_gate = jnp.concatenate([a_gate_shifted, b_gate], axis=2)

  # 5. Apply softmax gating and sum across the overlapping window dimension
  gate_weights = jax.nn.softmax(new_gate, axis=2).astype(new_kv.dtype)
  compressed = jnp.sum(new_kv * gate_weights, axis=2)

  # 6. Apply the projection norm
  if kv_norm is not None:
    compressed = kv_norm(compressed)

  # 7. Extract the next priors (Keep the full D2 so it fits the cache)
  # We grab the full last window of the current chunk to pass into the next chunk
  next_prior_kv = chunk_kv[:, -1:, :, :]
  next_prior_gate = chunk_gate[:, -1:, :, :]

  return compressed, next_prior_kv, next_prior_gate


def update_ar_cache_and_get_validity_mask(
    kv: Array,
    gate: Array,
    cache: Any,
    model_mode: str,
    compressor_fn: Any,
    comp_dim: int,
    batch_size: int,
    mask_ndims: Optional[int] = None,
) -> Tuple[Array, Optional[Array]]:
  """Helper for autoregressive decoding: updates KV cache and computes the AR validity mask.

  Delegates token-by-token compression and cache state updates to `cache(...)`, concatenates
  the cached prefill and autoregressive blocks, and optionally computes a binary validity mask
  to prevent queries from attending to statically allocated padding slots.

  Args:
    kv: Projected KV representations. Shape: `[batch, 1, emb_dim]`.
    gate: Projected gate representations. Shape: `[batch, 1, emb_dim]`.
    cache: KV cache instance for the compressor.
    model_mode: Execution mode (`MODEL_MODE_AUTOREGRESSIVE`).
    compressor_fn: Component-specific callback function that compresses a window of tokens.
    comp_dim: Target head dimension (`head_dim` or `index_head_dim`).
    batch_size: Original unpadded batch size.
    mask_ndims: Number of dimensions for the output validity mask (e.g. 4 for `[B, 1, 1, L]`,
      3 for `[B, 1, L]`, or None to skip mask calculation).

  Returns:
    Tuple of (compressed, is_valid_mask):
      - compressed: The unpadded concatenated compressed representations. Shape: `[batch, total_len, comp_dim]`.
      - is_valid_mask: Boolean array indicating valid (non-padded) cache positions, or None if `mask_ndims` is None.
  """
  kv_exp = jnp.expand_dims(kv, 2)
  gate_exp = jnp.expand_dims(gate, 2)

  cached_prefill, cached_ar = cache(
      key=kv_exp,
      value=kv_exp,
      gate=gate_exp,
      decoder_segment_ids=None,
      model_mode=model_mode,
      compressor_fn=compressor_fn,
  )

  compressed_full = jnp.concatenate([cached_prefill[0], cached_ar[0]], axis=1)
  compressed = compressed_full[:batch_size, :, 0, :comp_dim]

  if mask_ndims is None:
    return compressed, None

  # --- AUTOREGRESSIVE VALIDITY MASK ---
  # In autoregressive mode, the total compressed KV sequence is the concatenation of:
  #   1. The prefill cache region: indices [0, max_prefill_comp)
  #   2. The AR cache region:      indices [max_prefill_comp, max_prefill_comp + ar_max_len)
  # Because both regions are statically padded to maximum capacity, we construct a binary mask
  # so queries attend only to valid, completed compression blocks:
  #   - Prefill blocks are valid if index < prefill_blocks_count (total entry count - AR valid count).
  #   - AR blocks are valid if their offset (index - max_prefill_comp) < ar_valid.
  max_prefill_comp = cached_prefill[0].shape[1]
  ar_valid = jnp.expand_dims(cached_ar[3], axis=1)  # [B, 1]
  assert cache.entry_count is not None, "cache.entry_count cannot be None for compressed attention"
  prefill_blocks_count = cache.entry_count.get_value() - ar_valid  # [B, 1]

  total_len = compressed.shape[1]
  shape_prefix = (1,) * (mask_ndims - 1) + (total_len,)
  entry_indices = jnp.arange(total_len).reshape(shape_prefix)

  if mask_ndims == 4:
    # Expand to [B, 1, 1, 1] for [B, 1, 1, L] mask
    prefill_count_exp = jnp.expand_dims(prefill_blocks_count, axis=(1, 3))
    ar_valid_exp = jnp.expand_dims(ar_valid, axis=(1, 3))
  elif mask_ndims == 3:
    # Expand to [B, 1, 1] for [B, 1, L] mask
    prefill_count_exp = jnp.expand_dims(prefill_blocks_count, axis=1)
    ar_valid_exp = jnp.expand_dims(ar_valid, axis=1)
  else:
    raise ValueError(f"Unsupported mask_ndims={mask_ndims}; expected 3, 4, or None.")

  is_prefill = entry_indices < max_prefill_comp
  is_valid_prefill = is_prefill & (entry_indices < prefill_count_exp)

  is_ar = entry_indices >= max_prefill_comp
  is_valid_ar = is_ar & ((entry_indices - max_prefill_comp) < ar_valid_exp)

  is_valid_mask = is_valid_prefill | is_valid_ar

  return compressed, is_valid_mask


def compute_csa_prefill_chunk_pooling(
    kv: Array,
    gate: Array,
    seq_len: int,
    batch_size: int,
    compress_rate: int,
    position_ids: Array,
    position_bias: Array,
    kv_norm: Any,
    rotary_emb: Any,
    head_dim: int,
    dtype: Any,
    cache: Optional[Any] = None,
) -> Tuple[Array, int, Optional[Array], Optional[Array], int]:
  """Helper for CSA prefill chunking, overlap pooling, and RoPE embedding.

  Args:
    kv: Projected KV representations.
    gate: Projected gate representations.
    seq_len: Total sequence length.
    batch_size: Batch size.
    compress_rate: Compression rate.
    position_ids: Absolute token positions.
    position_bias: Position bias tensor.
    kv_norm: Normalization layer.
    rotary_emb: Rotary embedding layer.
    head_dim: Target head dimension (`head_dim` or `index_head_dim`).
    dtype: Computation dtype.
    cache: Optional KVCache instance.

  Returns:
    Tuple of (compressed, compressed_len, next_prior_kv, next_prior_gate, usable).
  """
  usable = (seq_len // compress_rate) * compress_rate
  chunk_kv = kv[:, :usable]
  chunk_gate = gate[:, :usable]

  if cache is not None:
    assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
    assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
    prior_kv = cache.overlap_kv.get_value()
    prior_gate = cache.overlap_gate.get_value()
  else:
    prior_kv, prior_gate = None, None

  if chunk_kv.shape[1] > 0:
    n_windows = chunk_kv.shape[1] // compress_rate
    chunk_kv_reshaped = chunk_kv.reshape((batch_size, n_windows, compress_rate, -1))
    chunk_gate_reshaped = chunk_gate.reshape((batch_size, n_windows, compress_rate, -1)) + position_bias

    block_positions = position_ids[:, :usable:compress_rate]
    prior_block_positions = jnp.concatenate([block_positions[:, 0:1] - compress_rate, block_positions[:, :-1]], axis=1)
    is_same_doc = block_positions == (prior_block_positions + compress_rate)

    compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
        chunk_kv_reshaped, chunk_gate_reshaped, kv_norm, head_dim, prior_kv, prior_gate, is_same_doc=is_same_doc
    )
    compressed_len = compressed.shape[1]
    positions = position_ids[:, :usable:compress_rate]
    compressed = rotary_emb(compressed, positions, unsqueeze_dim=None)
  else:
    compressed = jnp.zeros((batch_size, 0, head_dim), dtype=dtype)
    compressed_len = 0
    next_prior_kv = prior_kv
    next_prior_gate = prior_gate

  return compressed, compressed_len, next_prior_kv, next_prior_gate, usable


def prime_prefill_cache_state(
    kv: Array,
    gate: Array,
    compressed_kv: Array,
    seq_len: int,
    usable: int,
    compress_rate: int,
    cache: Any,
    next_prior_kv: Optional[Array] = None,
    next_prior_gate: Optional[Array] = None,
) -> None:
  """Shared utility to initialize and prime KVCache state during prefill.

  Handles:
    1. Leftover token buffering when `seq_len` is not divisible by `compress_rate`.
    2. Transposition, head-dim padding, sequence-len slicing, and batch-repeating of
       compressed blocks before inserting them into `cache.cached_prefill_key`.
    3. Updating `cache.entry_count` with the initial compressed length.
    4. Writing trailing Ca window priors to `cache.overlap_kv` / `cache.overlap_gate`
       for overlapping CSA compressors.

  Args:
    kv: Projected KV representations. Shape: `[batch, seq_len, head_dim]`.
    gate: Projected gate representations. Shape: `[batch, seq_len, head_dim]`.
    compressed_kv: Prefill compressed KV blocks. Shape: `[batch, compressed_len, 1, comp_dim]`.
    seq_len: Total uncompressed sequence length.
    usable: Number of tokens divisible by `compress_rate` that were compressed.
    compress_rate: Compression rate factor.
    cache: Target KVCache instance.
    next_prior_kv: Trailing window KV prior to write to overlap cache (optional).
    next_prior_gate: Trailing window gate prior to write to overlap cache (optional).
  """
  if compressed_kv.ndim == 3:
    compressed_kv = jnp.expand_dims(compressed_kv, 2)
  remainder = seq_len % compress_rate
  if remainder > 0:
    leftover_kv = kv[:, usable:]
    leftover_gate = gate[:, usable:]
    pad_len = compress_rate - remainder
    padded_kv = jnp.expand_dims(jnp.pad(leftover_kv, ((0, 0), (0, pad_len), (0, 0))), 2)
    padded_gate = jnp.expand_dims(jnp.pad(leftover_gate, ((0, 0), (0, pad_len), (0, 0))), 2)

    assert cache.leftover_buffer_kv is not None, "leftover_buffer_kv cannot be None"
    assert cache.leftover_buffer_gate is not None, "leftover_buffer_gate cannot be None"
    assert cache.accumulator_index is not None, "accumulator_index cannot be None"
    actual_batch = cache.leftover_buffer_kv.get_value().shape[0]
    # Note: We copy/repeat the same batch across the cache batch dimension so that
    # leftover buffers are initialized for all concurrent decode slots.
    if padded_kv.shape[0] != actual_batch:
      repeats = actual_batch // padded_kv.shape[0]
      padded_kv = jnp.repeat(padded_kv, repeats, axis=0)
      padded_gate = jnp.repeat(padded_gate, repeats, axis=0)

    cache.leftover_buffer_kv.set_value(padded_kv)
    cache.leftover_buffer_gate.set_value(padded_gate)
    cache.accumulator_index.set_value(jnp.full((actual_batch, 1), remainder, dtype=jnp.int32))

  compressed_len = compressed_kv.shape[1]
  if compressed_len > 0:
    cache_key_var = cache.cached_prefill_key
    assert cache_key_var is not None, "cached_prefill_key cannot be None"

    # Transpose from [B, L, H, D] -> [L, H, B, D] to match the cache's physical layout
    update_blocks = jnp.transpose(compressed_kv, cache.prefill_cache_axis_order)

    operand_shape = cache_key_var.get_value().shape
    # Pad the Head Dim (axis 3) if needed (e.g. 64 -> 128)
    if update_blocks.shape[3] < operand_shape[3]:
      pad_amt = operand_shape[3] - update_blocks.shape[3]
      update_blocks = jnp.pad(update_blocks, ((0, 0), (0, 0), (0, 0), (0, pad_amt)))

    update_blocks = update_blocks[:, : operand_shape[1], ...]

    operand = cache_key_var.get_value()
    batch_axis = cache.prefill_cache_axis_order.index(0)

    # Broadcast batch=1 prefill requests to fit the max_concurrent_decodes cache
    # Note: We repeat the same batch across the cache batch dimension (when prefill batch size
    # < max_concurrent_decodes) so that the prefill state is copied to all concurrent decode slots.
    if operand.shape[batch_axis] != update_blocks.shape[batch_axis]:
      repeats = operand.shape[batch_axis] // update_blocks.shape[batch_axis]
      update_blocks = jnp.repeat(update_blocks, repeats, axis=batch_axis)

    cache_key_var.set_value(jax.lax.dynamic_update_slice_in_dim(operand, update_blocks, 0, axis=0))
    # Ensure entry_count update matches the physical cache batch size
    assert cache.entry_count is not None, "entry_count cannot be None"
    actual_batch = cache.entry_count.get_value().shape[0]
    cache.entry_count.set_value(jnp.full((actual_batch, 1), compressed_len, dtype=jnp.int32))

    if next_prior_kv is not None and next_prior_gate is not None:
      overlap_kv_to_write = next_prior_kv
      overlap_gate_to_write = next_prior_gate

      # Note: We copy/repeat the same batch across the cache batch dimension so that
      # overlap registers are initialized for all concurrent decode slots.
      assert cache.overlap_kv is not None, "overlap_kv cannot be None"
      assert cache.overlap_gate is not None, "overlap_gate cannot be None"
      if overlap_kv_to_write.shape[0] != cache.overlap_kv.get_value().shape[0]:
        repeats = cache.overlap_kv.get_value().shape[0] // overlap_kv_to_write.shape[0]
        overlap_kv_to_write = jnp.repeat(overlap_kv_to_write, repeats, axis=0)
        overlap_gate_to_write = jnp.repeat(overlap_gate_to_write, repeats, axis=0)

      cache.overlap_kv.set_value(overlap_kv_to_write)
      cache.overlap_gate.set_value(overlap_gate_to_write)


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
    super().__init__(
        config,
        compress_ratio,
        rotary_embedding,
        1,
        kernel_init,
        quant,
        model_mode,
        rngs,
    )

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
      model_mode: The execution mode (e.g. train, prefill, or autoregressive).
      cache: Optional KV cache instance used during inference.

    Returns:
      compressed_kv: The pooled KV tensors. Shape: `[batch, n_windows, 1, head_dim]`.
      compressed_causal_mask: Causal mask preventing queries from seeing future blocks.
                              Shape: `[batch, 1, seq_len, n_windows]`.
    """
    batch_size, seq_len, _ = hidden_states.shape
    # Project hidden states to KV and Gate components
    # [batch, seq_len, emb_dim] -> [batch, seq_len, head_dim]
    kv = self.kv_proj(hidden_states)
    # [batch, seq_len, emb_dim] -> [batch, seq_len, head_dim]
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def hca_compressor_fn(buf_kv, buf_gate):
        gate_weights = jax.nn.softmax(buf_gate + self.position_bias.value[:, None, :], axis=1).astype(buf_kv.dtype)
        compressed = self.kv_norm(jnp.sum(buf_kv * gate_weights, axis=1))
        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)
        return jnp.expand_dims(compressed, 2), None, None

      compressed, is_valid = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=hca_compressor_fn,
          comp_dim=self.head_dim,
          batch_size=batch_size,
          mask_ndims=4,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)  # [B, N, 1, D]
      compressed_mask = jnp.where(is_valid, 0.0, DEFAULT_MASK_VALUE).astype(self.dtype)

      return compressed_kv, compressed_mask

    # --- PREFILL CHUNKING & PRIMING ---
    usable = (seq_len // self.compress_rate) * self.compress_rate
    chunk_kv = kv[:, :usable]
    chunk_gate = gate[:, :usable]

    # Process overlapping windows if there is enough sequence length
    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate

      # Reshape into blocks of size `compress_rate`
      # -> [batch, n_windows, compress_rate, head_dim]
      chunk_kv = chunk_kv.reshape((batch_size, n_windows, self.compress_rate, -1))
      chunk_gate = chunk_gate.reshape((batch_size, n_windows, self.compress_rate, -1)) + self.position_bias.value

      # Apply gating mechanism over each compression window
      # -> [batch, n_windows, head_dim]
      gate_weights = jax.nn.softmax(chunk_gate, axis=2).astype(chunk_kv.dtype)
      compressed = self.kv_norm(jnp.sum(chunk_kv * gate_weights, axis=2))

      # Calculate positions for the compressed blocks
      positions = position_ids[:, : usable : self.compress_rate]

      # Apply Rotary Positional Embeddings to the pooled representations
      # compressed is [batch, n_windows, head_dim]
      compressed = self.rotary_emb(compressed, positions, unsqueeze_dim=None)
    else:
      # Provide an empty tensor when the sequence is shorter than the compression rate
      compressed = jnp.zeros((batch_size, 0, self.head_dim), dtype=self.dtype)

    # Expand the feature dimension to match the standard KV projection shape
    # -> [batch, n_windows, 1, head_dim]
    compressed_kv = jnp.expand_dims(compressed, axis=2)
    compressed_len = compressed_kv.shape[1]

    # --- PREFILL CACHE PRIMING ---
    if cache is not None:
      prime_prefill_cache_state(
          kv=kv,
          gate=gate,
          compressed_kv=compressed_kv,
          seq_len=seq_len,
          usable=usable,
          compress_rate=self.compress_rate,
          cache=cache,
      )

    # Skip causal mask generation during decoding (seq_len == 1) or if no blocks were pooled
    if seq_len == 1 or compressed_len == 0:
      compressed_mask = jnp.zeros((batch_size, 1, seq_len, compressed_len), dtype=self.dtype)
      return compressed_kv, compressed_mask

    # Construct a causal mask preventing early queries from attending to future compressed blocks
    usable_len = compressed_len * self.compress_rate
    block_positions = position_ids[:, : usable_len : self.compress_rate]
    future_mask = (block_positions[:, None, None, :] + self.compress_rate) > (position_ids[:, None, :, None] + 1)
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
      mesh: Optional[Mesh] = None,
  ):
    """Initializes the Indexer for CSA.

    Args:
      config: Model configuration containing indexer parameters.
      compress_ratio: The compression ratio (e.g., config.compress_rate_csa).
      rotary_embedding: Rotary embedding instance for injecting position info into index representations.
      kernel_init: Weight initializer for the indexer projections.
      quant: Optional quantization scheme.
      rngs: Optional random state initialization.
      mesh: Device mesh for indexer activation sharding.
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
    self.mesh = mesh
    self.shard_indexer_acts = config.shard_indexer_acts and mesh is not None

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

  def _shard_acts(self, inputs: Array, logical_axes: Tuple) -> Array:
    return maybe_shard_with_logical(inputs, logical_axes, self.mesh, self.config.shard_mode, rules=None)

  def __call__(
      self,
      hidden_states: Array,
      q_latent: Array,
      position_ids: Array,
      attention_mask: Optional[Array] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      cache: Optional[Any] = None,
  ) -> Array:
    """Forward pass for the DeepSeek-V4 Indexer.

    Args:
      hidden_states: Input token embeddings.
      q_latent: Latent query representation for index score computation.
      position_ids: Absolute token positions.
      attention_mask: Optional attention mask.
      model_mode: Execution mode (train, prefill, or autoregressive).
      cache: Optional Indexer KV cache instance for inference.

    Returns:
      Top-K selected indices for each query position.
    """
    batch_size, seq_len, _ = hidden_states.shape
    future_mask = None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def indexer_compressor_fn(buf_kv, buf_gate):
        chunk_kv_reshaped = jnp.swapaxes(buf_kv, 1, 2)
        chunk_gate_reshaped = jnp.swapaxes(buf_gate, 1, 2) + self.position_bias.value

        assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
        assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
        prior_kv = cache.overlap_kv.get_value()
        prior_gate = cache.overlap_gate.get_value()

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.index_head_dim, prior_kv, prior_gate
        )

        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)

        return (
            jnp.expand_dims(compressed, 2),
            next_prior_kv,
            next_prior_gate,
        )

      compressed, is_valid = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=indexer_compressor_fn,
          comp_dim=self.index_head_dim,
          batch_size=batch_size,
          mask_ndims=3,
      )
      compressed_len = compressed.shape[1]
      future_mask = ~is_valid

    # --- PREFILL CHUNKING & PRIMING ---
    else:
      compressed, compressed_len, next_prior_kv, next_prior_gate, usable = compute_csa_prefill_chunk_pooling(
          kv=kv,
          gate=gate,
          seq_len=seq_len,
          batch_size=batch_size,
          compress_rate=self.compress_rate,
          position_ids=position_ids,
          position_bias=self.position_bias.value,
          kv_norm=self.kv_norm,
          rotary_emb=self.rotary_emb,
          head_dim=self.index_head_dim,
          dtype=self.dtype,
          cache=cache,
      )

      # Prefill Cache Insertion
      if cache is not None:
        prime_prefill_cache_state(
            kv=kv,
            gate=gate,
            compressed_kv=compressed,
            seq_len=seq_len,
            usable=usable,
            compress_rate=self.compress_rate,
            cache=cache,
            next_prior_kv=next_prior_kv,
            next_prior_gate=next_prior_gate,
        )

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

    shard_acts = self.shard_indexer_acts and model_mode != MODEL_MODE_AUTOREGRESSIVE

    if shard_acts:
      q = self._shard_acts(q, ("activation_batch", "activation_heads", "activation_length", None))
      compressed_kv = self._shard_acts(compressed_kv, ("activation_batch", "activation_heads", None, None))

    scores = jnp.einsum("bhsd,bhwd->bhsw", q, compressed_kv)
    scores = jax.nn.relu(scores) * self.softmax_scale
    if shard_acts:
      scores = self._shard_acts(scores, ("activation_batch", "activation_heads", "activation_length", None))
    weights = self.weights_proj(hidden_states).astype(jnp.float32) * self.weights_scaling
    index_scores = jnp.einsum("bhsw,bsh->bsw", scores, weights)
    if shard_acts:
      index_scores = self._shard_acts(index_scores, ("activation_batch", "activation_length", None))

    k = min(self.index_topk, compressed_len)

    # --- ONLY RUN MATHEMATICAL CAUSAL MASK IN PREFILL/TRAIN ---
    if future_mask is None:
      usable_len = compressed_len * self.compress_rate
      block_positions = position_ids[:, : usable_len : self.compress_rate]
      future_mask = (block_positions[:, None, :] + self.compress_rate) > (position_ids[:, :, None] + 1)

    # Apply the mask to the scores
    index_scores = jnp.where(future_mask, jnp.full_like(index_scores, -jnp.inf), index_scores)

    combined_invalid = future_mask
    if attention_mask is not None:
      att_m = attention_mask[:, :, :compressed_len]
      index_scores += att_m
      combined_invalid = combined_invalid | (att_m < -100.0)

    top_k_indices = jax.lax.top_k(index_scores, k)[1]
    invalid = jnp.take_along_axis(combined_invalid, top_k_indices, axis=-1)

    final_indices = jnp.where(invalid, jnp.full_like(top_k_indices, -1), top_k_indices)

    return final_indices


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
      kernel_init: Any = nnx.initializers.normal(stddev=0.02),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
      mesh: Optional[Mesh] = None,
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
      mesh: Device mesh for indexer activation sharding.
    """
    super().__init__(
        config,
        compress_ratio,
        rotary_embedding,
        2,
        kernel_init,
        quant,
        model_mode,
        rngs,
    )

    self.indexer = DeepseekV4Indexer(
        config=config,
        compress_ratio=compress_ratio,
        rotary_embedding=rotary_embedding,
        kernel_init=kernel_init,
        quant=quant,
        rngs=rngs,
        mesh=mesh,
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
    """Forward pass for the CSA compressor.

    Args:
      hidden_states: Input token embeddings.
      q_latent: Latent query representation for index score computation.
      position_ids: Absolute token positions.
      attention_mask: Optional attention mask.
      model_mode: Execution mode (train, prefill, or autoregressive).
      cache: Optional CSA compressor KV cache instance for inference.
      indexer_cache: Optional Indexer KV cache instance for inference.

    Returns:
      compressed_kv: The pooled KV tensors.
      compressed_mask: The sparse attention mask computed by the indexer.
    """
    batch_size, seq_len, _ = hidden_states.shape

    # 1. ALWAYS Run Indexer (It fetches its own history inside AR)
    top_k_indices = self.indexer(hidden_states, q_latent, position_ids, attention_mask, model_mode, indexer_cache)

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and cache is not None:

      def csa_compressor_fn(buf_kv, buf_gate):
        chunk_kv_reshaped = jnp.swapaxes(buf_kv, 1, 2)
        chunk_gate_reshaped = jnp.swapaxes(buf_gate, 1, 2) + self.position_bias.value

        assert cache.overlap_kv is not None, "cache.overlap_kv cannot be None"
        assert cache.overlap_gate is not None, "cache.overlap_gate cannot be None"
        prior_kv = cache.overlap_kv.get_value()
        prior_gate = cache.overlap_gate.get_value()

        compressed, next_prior_kv, next_prior_gate = csa_overlap_pooling(
            chunk_kv_reshaped, chunk_gate_reshaped, self.kv_norm, self.head_dim, prior_kv, prior_gate
        )

        block_pos = position_ids - (self.compress_rate - 1)
        compressed = self.rotary_emb(compressed, block_pos, unsqueeze_dim=None)

        return (
            jnp.expand_dims(compressed, 2),
            next_prior_kv,
            next_prior_gate,
        )

      compressed, _ = update_ar_cache_and_get_validity_mask(
          kv=kv,
          gate=gate,
          cache=cache,
          model_mode=model_mode,
          compressor_fn=csa_compressor_fn,
          comp_dim=self.head_dim,
          batch_size=batch_size,
          mask_ndims=None,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)
      compressed_len = compressed_kv.shape[1]

    # --- PREFILL CHUNKING & PRIMING ---
    else:
      compressed, compressed_len, next_prior_kv, next_prior_gate, usable = compute_csa_prefill_chunk_pooling(
          kv=kv,
          gate=gate,
          seq_len=seq_len,
          batch_size=batch_size,
          compress_rate=self.compress_rate,
          position_ids=position_ids,
          position_bias=self.position_bias.value,
          kv_norm=self.kv_norm,
          rotary_emb=self.rotary_emb,
          head_dim=self.head_dim,
          dtype=self.dtype,
          cache=cache,
      )
      compressed_kv = jnp.expand_dims(compressed, 2)

      if cache is not None:
        prime_prefill_cache_state(
            kv=kv,
            gate=gate,
            compressed_kv=compressed_kv,
            seq_len=seq_len,
            usable=usable,
            compress_rate=self.compress_rate,
            cache=cache,
            next_prior_kv=next_prior_kv,
            next_prior_gate=next_prior_gate,
        )

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
      compressed_mask = jnp.full(
          (batch_size, 1, seq_len, compressed_len),
          DEFAULT_MASK_VALUE,
          dtype=self.dtype,
      )

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
    self.sinks = nnx.data(
        nnx.Param(
            jnp.zeros((self.num_query_heads,), dtype=self.weight_dtype),
            sharding=(None,),
        )
    )

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
          kernel_init=self.kernel_init,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=self.rngs,
          mesh=self.mesh,
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
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(self.config, MODEL_MODE_AUTOREGRESSIVE)

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

  def _apply_rotary_embedding_v4(
      self, inputs: Array, inputs_positions: Array, unsqueeze_dim: int = -2, reverse: bool = False
  ) -> Array:
    """Applies rotary position embeddings, dispatching keyword arguments safely based on capability."""
    if isinstance(self.rotary_embedding, DeepSeekV4RotaryEmbedding):
      return self.rotary_embedding(inputs, inputs_positions, unsqueeze_dim=unsqueeze_dim, reverse=reverse)
    elif reverse and hasattr(self.rotary_embedding, "reverse"):
      return self.rotary_embedding(inputs, inputs_positions, reverse=True)
    else:
      return self.rotary_embedding(inputs, inputs_positions)

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
    q_out = self._apply_rotary_embedding_v4(q_up_normed, inputs_positions, unsqueeze_dim=-2)

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

    kv_out = self._apply_rotary_embedding_v4(kv_up_normed, inputs_positions, unsqueeze_dim=-2)

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

    q, q_normed = self.compressed_query_projection(inputs_q, inputs_positions, model_mode)
    q = checkpoint_name(q, "query_proj")
    kv, _ = self.compressed_kv_projection(inputs_kv, inputs_positions, model_mode)

    current_kv_cache = kv_cache

    # 1. Update the Local (Sliding Window) KV Cache with the uncompressed tokens
    if model_mode != MODEL_MODE_TRAIN and getattr(self, "KVCache_0", None) is not None:
      current_kv_cache = self.update_kv_caches(
          kv, kv, decoder_segment_ids, model_mode, kwargs.get("previous_chunk", None)
      )

    # Generate compressed representations based on the configured layer type
    compressed_kv = None
    compressed_mask = None
    compressed_segment_mask = None
    decoder_segment_ids_kv = decoder_segment_ids
    compressed_segment_ids = None

    if decoder_segment_ids is not None and self.compress_ratio > 0:
      compress_rate = self.compress_ratio
      num_blocks = inputs_kv.shape[1] // compress_rate
      usable = num_blocks * compress_rate
      if decoder_segment_ids.shape[1] < usable:
        pad_seg = usable - decoder_segment_ids.shape[1]
        last_seg = decoder_segment_ids[:, -1:]
        pad_block = jnp.repeat(last_seg, pad_seg, axis=1)
        padded_seg_ids = jnp.concatenate([decoder_segment_ids, pad_block], axis=1)
      else:
        padded_seg_ids = decoder_segment_ids[:, :usable]

      chunked_segment_ids = padded_seg_ids.reshape((decoder_segment_ids.shape[0], num_blocks, compress_rate))
      min_seg = jnp.min(chunked_segment_ids, axis=-1)
      max_seg = jnp.max(chunked_segment_ids, axis=-1)
      is_valid_window = min_seg == max_seg

      compressed_segment_ids = jnp.where(is_valid_window, min_seg, -1)
      decoder_segment_ids_kv = jnp.concatenate([decoder_segment_ids, compressed_segment_ids], axis=1)

      valid_comp_seg = (decoder_segment_ids[:, :, None] == compressed_segment_ids[:, None, :]) & (
          compressed_segment_ids[:, None, :] >= 0
      )
      compressed_segment_mask = jnp.where(valid_comp_seg, 0.0, DEFAULT_MASK_VALUE)

    # Route to the appropriate compressor depending on the layer's role in the architecture
    if self.compress_ratio > 4:
      compressed_kv, compressed_mask = self.hca_compressor(
          inputs_kv, q_normed, inputs_positions, model_mode, self.compressor_cache
      )
    elif self.compress_ratio == 4:
      compressed_kv, compressed_mask = self.csa_compressor(
          inputs_kv,
          q_normed,
          inputs_positions,
          compressed_segment_mask,
          model_mode,
          self.compressor_cache,
          self.indexer_cache,
      )

    # Apply segment masking to the compressed blocks
    if compressed_segment_mask is not None and compressed_mask is not None:
      # compressed_segment_mask is [batch, q_len, num_compressed_blocks]
      # compressed_mask is [batch, 1, q_len, num_compressed_blocks]
      compressed_mask = compressed_mask + jnp.expand_dims(
          compressed_segment_mask[:, :, : compressed_mask.shape[-1]], axis=1
      )

    kv = checkpoint_name(kv, "kv_proj")

    pad_kv_total = 0
    unpadded_kv = jnp.concatenate([kv, compressed_kv], axis=1) if compressed_kv is not None else kv

    # Pad total KV length to tile size multiple (config.sa_block_kv) for SPMD sequence divisibility and
    # Tokamax dynamic splash tile boundary alignment. Note: Tokamax kernel inside AttentionOp additionally
    # sets inner block size as min(block_kv, key_len) during kernel invocation.
    if self.attention_kernel == "flash":
      total_kv_len = kv.shape[1] + (compressed_kv.shape[1] if compressed_kv is not None else 0)
      block_size = self.config.sa_block_kv
      pad_kv_total = (block_size - (total_kv_len % block_size)) % block_size

      if pad_kv_total > 0:
        if compressed_kv is not None:
          # Prepend padding to the compressed blocks so they remain at the end of the sequence
          compressed_kv = jnp.pad(compressed_kv, ((0, 0), (pad_kv_total, 0), (0, 0), (0, 0)))

          if decoder_segment_ids is not None and compressed_segment_ids is not None:
            comp_seg_padded = jnp.pad(compressed_segment_ids, ((0, 0), (pad_kv_total, 0)), constant_values=-1)
            decoder_segment_ids_kv = jnp.concatenate([decoder_segment_ids, comp_seg_padded], axis=1)
        else:
          # Fallback: Pad at the end if no compressed blocks exist
          kv = jnp.pad(kv, ((0, 0), (0, pad_kv_total), (0, 0), (0, 0)))
          if decoder_segment_ids_kv is not None:
            decoder_segment_ids_kv = jnp.pad(decoder_segment_ids_kv, ((0, 0), (0, pad_kv_total)), constant_values=-1)

    # Prepare the mask shape for the underlying AttentionOp
    if compressed_mask is not None:
      compressed_mask = jnp.expand_dims(compressed_mask, axis=2)

    # Scale queries if a pre-attention scalar is defined
    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      q = q * self.query_pre_attn_scalar

    # Build indexer mask explicitly for tokamax splash kernel
    indexer_mask = None
    if self.attention_kernel == "flash" and compressed_mask is not None:
      indexer_mask = self.attention_op.generate_attention_mask(
          q,
          unpadded_kv,
          decoder_segment_ids,
          model_mode,
          compressed_mask=compressed_mask,
          pad_kv_total=pad_kv_total,
          decoder_segment_ids_kv=decoder_segment_ids_kv,
      )

      if indexer_mask is not None:
        # Extract single KV head and Query-per-KV head group axes [batch, 1, 1, Q, KV] -> [batch, Q, KV]
        indexer_mask = indexer_mask[:, 0, 0, :, :]

    # Compute Attention
    # -> [batch, q_length, num_query_heads, head_dim]
    attn_out = self.attention_op(
        q,
        kv,
        kv,
        decoder_segment_ids,
        inputs_positions,
        model_mode,
        sinks=self.sinks.value if self.sinks is not None else None,
        compressed_mask=compressed_mask,
        compressed_kv=compressed_kv,
        cached_values=current_kv_cache,
        indexer_mask=indexer_mask,
        decoder_segment_ids_kv=decoder_segment_ids_kv,
    )

    # Reverse RoPE on Values
    attn_out = self._apply_rotary_embedding_v4(attn_out, inputs_positions, unsqueeze_dim=-2, reverse=True)

    attn_out = checkpoint_name(attn_out, "attention_out")

    # Project outputs through Grouped Linear layers
    b, s, h, d = attn_out.shape
    # -> [batch, q_length, o_groups, in_features_per_group]
    grouped_out = attn_out.reshape(b, s, self.config.o_groups, (h * d) // self.config.o_groups)
    # -> [batch, q_length, o_groups, out_features_per_group]
    grouped_out = self.o_a_proj(grouped_out)
    # -> [batch, q_length, o_groups * out_features_per_group]
    grouped_flat = grouped_out.reshape(b, s, -1)
    # -> [batch, q_length, emb_dim]
    final_out = self.o_b_proj(grouped_flat)
    final_out = checkpoint_name(final_out, "out_proj")

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
