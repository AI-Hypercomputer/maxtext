#  Copyright 2023 Google LLC
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

"""Attentions Layers."""

import dataclasses
import enum
import functools
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from functools import partial
import math

import numpy as np

from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.gpu import attention as gpu_pallas_attention
from jax.experimental.pallas.ops.gpu import decode_attention as gpu_pallas_decode_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning

from MaxText import max_utils
from MaxText.common_types import DecoderBlockType, DEFAULT_MASK_VALUE, BATCH, BATCH_NO_EXP, HEAD, KV_LENGTH, PREFILL_LENGTH, D_KV, CACHE_BATCH_PREFILL, CACHE_SEQUENCE, AxisNames, CACHE_BATCH, CACHE_HEADS, CACHE_SCALE_BATCH, CACHE_KV, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV, AxisIdxes, LENGTH, LENGTH_NO_EXP, DType, Config, Array, Q_LENGTH, Q_LENGTH_NO_EXP, DECODE_LENGTH, DECODE_BATCH, PREFILL_KV_BATCH, KV_HEAD, KV_HEAD_DIM, KV_BATCH, KV_BATCH_NO_EXP, EMBED, MODEL_MODE_AUTOREGRESSIVE, DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, EP_AS_CONTEXT
from MaxText.inference import kvcache
from MaxText.inference import page_manager
from MaxText.inference import paged_attention
from MaxText.inference.kvcache import KVQuant, KVTensor
from MaxText.kernels.ragged_attention import ragged_gqa
from MaxText.kernels.ragged_attention import ragged_mha
from MaxText.layers import nnx_wrappers
from MaxText.layers.embeddings import (
    LLaMARotaryEmbedding,
    LlamaVisionRotaryEmbedding,
    RotaryEmbedding,
    YarnRotaryEmbedding,
)
from MaxText.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned, default_bias_init
from MaxText.layers.linears import DenseGeneral, canonicalize_tuple, normalize_axes
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant


# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


class AttentionType(enum.Enum):
  GLOBAL = "global"  # default, with causality
  LOCAL_SLIDING = "local_sliding"
  CHUNK = "chunk"
  MLA = "mla"
  FULL = "full"


# Used to pass in splash attention block sizes from config.
global_block_q = 0
global_block_kv = 0
global_block_kv_compute = 0
global_block_q_dkv = 0
global_block_kv_dkv = 0
global_block_kv_dkv_compute = 0
global_block_q_dq = 0
global_block_kv_dq = 0
global_use_fused_bwd_kernel = False
global_q_layout = ""
global_k_layout = ""
global_v_layout = ""

dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def validate_compute_axis_order(s: AxisIdxes) -> None:
  valid_compute_axis_order = ((0, 1, 2, 3), (0, 2, 1, 3))
  if s not in valid_compute_axis_order:  # currently supported compute_axis_order
    raise ValueError("Invalid compute_axis_order was passed. Valid options ", valid_compute_axis_order)


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


# TODO(agagik): change splash_attention_mask._ComputableMask to be non protected
class ChunkedCausalMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.

  This mask class inherits from splash_attention_mask._ComputableMask and is designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention kernel, avoiding the memory cost of
  materializing the full (sequence_length, sequence_length) boolean mask array, which can be prohibitive for long sequences.

  Attributes:
    chunk_size: The size of each attention chunk.
  """

  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError("chunk_size must be positive")
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      if q_ids.size == 0 or kv_ids.size == 0:
        return np.empty((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

      # Condition 1: Same chunk
      q_chunk = q_ids // self.chunk_size
      kv_chunk = kv_ids // self.chunk_size
      same_chunk = q_chunk == kv_chunk

      # Condition 2: Causal
      causal = q_ids >= kv_ids

      return same_chunk & causal

    # Initialize the parent ComputableMask with this function
    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  # Implement equality and hashing based on relevant attributes
  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    # Compare shape, chunk_size, and the underlying q_sequence array
    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.chunk_size,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )


def _generate_chunk_attention_mask(mask_shape: tuple[int, int], chunk_size: int, q_offset: int = 0) -> jax.Array:
  """Generates an explicit boolean mask for chunked causal attention.

  This function computes the full boolean mask array where True indicates
  attention is allowed based on chunked causal rules (tokens attend only
  within the same chunk, and causally within that chunk).

  Args:
    mask_shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """

  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_offset
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  # chunk mask calculation
  same_chunk = (row_ids // chunk_size) == (col_ids // chunk_size)
  chunk_mask = same_chunk & (row_ids >= col_ids)
  return chunk_mask


def _make_block_mask_indices(bidirectional_mask):
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _make_bidirectional_block_mask(bidirectional_mask):
  """Creates bidirectional block mask from bidirectional_mask, where True corresponds to image tokens.
  bidirectional_mask shape: [B, L]
  bidirectional_block_mask shape: [B, L, L]
  Examples:
  bidirectional_mask = [[0, 1, 1, 1, 0, 0]]
  bidirectional_block_mask = [[
      [False, False, False, False, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
  ]]
  """
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  bidirectional_block_mask = (kv_block_indices[:, None, :] == q_block_indices[..., None]) & (q_block_indices[..., None] > 0)
  return bidirectional_block_mask


def attention_op_as_linen(
    *,
    config: Config,
    mesh: Mesh,
    attention_kernel: str,
    max_target_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    float32_qk_product: bool = False,
    max_prefill_predict_length: int = -1,
    float32_logits: bool = False,
    flash_axis_names_q: AxisNames = (BATCH, HEAD, LENGTH_NO_EXP, D_KV),
    flash_axis_names_q_ep: AxisNames = (BATCH_NO_EXP, HEAD, LENGTH, D_KV),
    flash_axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV),
    flash_axis_names_kv_ep: AxisNames = (BATCH_NO_EXP, HEAD, KV_LENGTH, D_KV),
    flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH_NO_EXP),
    flash_axis_names_splash_kernel_ep: AxisNames = (HEAD, LENGTH),
    prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
    cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
    cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV),
    ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV),
    ragged_lengths_names: AxisNames = (CACHE_BATCH,),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    key_axis_order: AxisIdxes = (2, 0, 1, 3),
    reshape_q: bool = False,
    dropout_rate: float = 0.0,
    dtype: DType = jnp.float32,
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    chunk_attn_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
):
  """A factory function to create an AttentionOp as a Linen module.

  This function serves as a bridge to use the NNX-based `AttentionOp` within a
  Linen model.
  """
  return nnx_wrappers.to_linen(
      AttentionOp,
      config=config,
      mesh=mesh,
      attention_kernel=attention_kernel,
      max_target_length=max_target_length,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      float32_qk_product=float32_qk_product,
      max_prefill_predict_length=max_prefill_predict_length,
      float32_logits=float32_logits,
      flash_axis_names_q=flash_axis_names_q,
      flash_axis_names_q_ep=flash_axis_names_q_ep,
      flash_axis_names_kv=flash_axis_names_kv,
      flash_axis_names_kv_ep=flash_axis_names_kv_ep,
      flash_axis_names_splash_kernel=flash_axis_names_splash_kernel,
      flash_axis_names_splash_kernel_ep=flash_axis_names_splash_kernel_ep,
      prefill_cache_logical_axis_names=prefill_cache_logical_axis_names,
      cache_logical_axis_names=cache_logical_axis_names,
      cache_scale_logical_axis_names=cache_scale_logical_axis_names,
      ragged_qkv_axis_names=ragged_qkv_axis_names,
      ragged_lengths_names=ragged_lengths_names,
      compute_axis_order=compute_axis_order,
      key_axis_order=key_axis_order,
      reshape_q=reshape_q,
      dropout_rate=dropout_rate,
      dtype=dtype,
      quant=quant,
      kv_quant=kv_quant,
      attention_type=attention_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
      sliding_window_size=sliding_window_size,
      chunk_attn_window_size=chunk_attn_window_size,
      use_ragged_attention=use_ragged_attention,
      ragged_block_size=ragged_block_size,
      metadata_fn=variable_to_logically_partitioned,
  )


class AttentionOp(nnx.Module):
  """Attention operation"""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      attention_kernel: str,
      max_target_length: int,
      num_query_heads: int,
      num_kv_heads: int,
      float32_qk_product: bool = False,
      max_prefill_predict_length: int = -1,
      float32_logits: bool = False,
      flash_axis_names_q: AxisNames = (BATCH, HEAD, LENGTH_NO_EXP, D_KV),
      flash_axis_names_q_ep: AxisNames = (BATCH_NO_EXP, HEAD, LENGTH, D_KV),
      flash_axis_names_kv: AxisNames = (BATCH, HEAD, KV_LENGTH, D_KV),
      flash_axis_names_kv_ep: AxisNames = (BATCH_NO_EXP, HEAD, KV_LENGTH, D_KV),
      flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH_NO_EXP),
      flash_axis_names_splash_kernel_ep: AxisNames = (HEAD, LENGTH),
      prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
      cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
      cache_scale_logical_axis_names: AxisNames = (
          CACHE_SCALE_BATCH,
          CACHE_SCALE_SEQUENCE,
          CACHE_SCALE_HEADS,
          CACHE_SCALE_KV,
      ),
      ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV),
      ragged_lengths_names: AxisNames = (CACHE_BATCH,),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      key_axis_order: AxisIdxes = (2, 0, 1, 3),
      reshape_q: bool = False,
      dropout_rate: float = 0.0,
      dtype: DType = jnp.float32,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      chunk_attn_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      rngs: nnx.Rngs | None = None,
  ):
    """Initializes the AttentionOp module.

    Args:
      config: The configuration for the model.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use.
      max_target_length: The maximum target length.
      num_query_heads: The number of query heads.
      num_kv_heads: The number of key/value heads.
      float32_qk_product: Whether to compute qk_product in float32.
      max_prefill_predict_length: The maximum prefill predict length.
      float32_logits: Whether to compute logits in float32.
      flash_axis_names_kv: The logical axis names for the KV cache in flash attention.
      flash_axis_names_q: The logical axis names for the query in flash attention.
      flash_axis_names_splash_kernel: The logical axis names for the splash attention kernel.
      prefill_cache_logical_axis_names: The logical axis names for the prefill cache.
      cache_logical_axis_names: The logical axis names for the cache.
      cache_scale_logical_axis_names: The logical axis names for the cache scale.
      ragged_qkv_axis_names: The logical axis names for ragged QKV tensors.
      ragged_lengths_names: The logical axis names for ragged lengths.
      compute_axis_order: The order of axes for computation.
      key_axis_order: The order of axes for the key.
      ... and other configuration parameters.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    self.config = config
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.max_target_length = max_target_length
    self.num_query_heads = num_query_heads
    self.num_kv_heads = num_kv_heads
    self.float32_qk_product = float32_qk_product
    self.max_prefill_predict_length = max_prefill_predict_length
    self.float32_logits = float32_logits
    self.flash_axis_names_q = flash_axis_names_q
    self.flash_axis_names_q_ep = flash_axis_names_q_ep
    self.flash_axis_names_kv = flash_axis_names_kv
    self.flash_axis_names_kv_ep = flash_axis_names_kv_ep
    self.flash_axis_names_splash_kernel = flash_axis_names_splash_kernel
    self.flash_axis_names_splash_kernel_ep = flash_axis_names_splash_kernel_ep
    self.prefill_cache_logical_axis_names = prefill_cache_logical_axis_names
    self.cache_logical_axis_names = cache_logical_axis_names
    self.cache_scale_logical_axis_names = cache_scale_logical_axis_names
    self.ragged_qkv_axis_names = ragged_qkv_axis_names
    self.ragged_lengths_names = ragged_lengths_names
    self.compute_axis_order = compute_axis_order
    self.key_axis_order = key_axis_order
    self.reshape_q = reshape_q
    self.dropout_rate = dropout_rate
    self.dtype = dtype
    self.quant = quant
    self.kv_quant = kv_quant
    self.attention_type = attention_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.chunk_attn_window_size = chunk_attn_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size

    def maybe_create_nnx(einsum, *args):
      if isinstance(einsum, nn.Module):
        return nnx_wrappers.ToNNX(einsum, rngs=rngs).lazy_init(*args)
      return einsum

    # qk_product
    if self.kv_quant:
      # Dummy inputs for lazy initialization
      b = 1
      t_prefill = self.max_prefill_predict_length
      t_ar = 1  # Autoregressive mode has a query length of 1
      n = self.num_query_heads
      n_kv = self.num_kv_heads
      d = self.config.head_dim
      g = n // n_kv
      s_prefill = self.max_prefill_predict_length
      s_ar = self.max_target_length

      # Dummy query/key/value shapes as before...
      dummy_query_prefill = jnp.zeros((b, t_prefill, n_kv, g, d), dtype=self.dtype)
      dummy_key_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_query_ar = jnp.zeros((b, t_ar, n_kv, g, d), dtype=self.dtype)
      dummy_key_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      dummy_attn_weights_prefill = jnp.zeros((b, n_kv, g, t_prefill, s_prefill), dtype=jnp.float32)
      dummy_value_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_attn_weights_ar = jnp.zeros((b, n_kv, g, t_ar, s_ar), dtype=jnp.float32)
      dummy_value_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      # Prefill AqtEinsum instances
      self.AqtEinsum_0 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts", dummy_query_prefill, dummy_key_prefill
      )
      self.AqtEinsum_1 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd", dummy_attn_weights_prefill, dummy_value_prefill
      )
      # Autoregressive AqtEinsum instances
      self.AqtEinsum_2 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts", dummy_query_ar, dummy_key_ar
      )
      self.AqtEinsum_3 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd", dummy_attn_weights_ar, dummy_value_ar
      )
    else:
      self.AqtEinsum_0 = jnp.einsum
      self.AqtEinsum_1 = jnp.einsum
      self.AqtEinsum_2 = jnp.einsum
      self.AqtEinsum_3 = jnp.einsum

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, f"k (dim {key.ndim}), v (dim {value.ndim}) must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  def generate_attention_mask(
      self,
      query,
      key,
      decoder_segment_ids: Array | None,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
  ) -> Array | None:
    """Generates a combined attention mask for Transformer models.

    This function constructs an attention mask by potentially combining
    several types of masks based on the input parameters and model
    configuration. The generated mask dictates which query-key pairs are
    allowed to attend to each other.

    The masking logic can enforce:
    1.  **Sequence Separation:** Using `decoder_segment_ids`, attention is
      confined within distinct sequences in a batch. This is crucial when
      multiple unrelated sequences are packed together.
    2.  **Causality:** Preventing attention to future positions. This is
      standard for autoregressive decoding. For chunked prefill, as
      described in the SARATHI paper [2], causality is adjusted based
      on `previous_chunk` information.
    3.  **Specialized Attention Patterns:** Depending on `self.attention_type`,
      it can apply:
      * Local Sliding Window Attention: Restricts attention to a
          fixed-size window around each query position.
      * Chunk Attention: Divides sequences into chunks and applies
          masking at the chunk level.
    4.  **Bidirectional Attention for Sub-sequences:** If `bidirectional_mask`
      is provided (e.g., for image tokens in a multimodal model),
      those parts of the sequence can attend bidirectionally, and this
      mask is OR-ed with other generated masks.

    The overall approach and specific masking techniques are influenced by
    efficient attention mechanisms like those found in the Pallas MHA
    Flash Attention reference [1].

    Args:
      query: The query tensor, typically of shape
          `[batch_size, q_sequence_length, num_heads, head_dim]`.
          Used primarily for deriving sequence length.
      key: The key tensor, typically of shape
          `[batch_size, kv_sequence_length, num_heads, head_dim]`.
          Used primarily for deriving sequence length.
      decoder_segment_ids: Optional `Array` of shape `[batch_size, q_sequence_length]`.
          Identifies distinct sequences within the batch. Attention is
          restricted to elements within the same segment ID. In autoregressive
          mode, specific values (e.g., `common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR`)
          can mark the currently active sequence for decoding.
      model_mode: A string (e.g., `common_types.MODEL_MODE_AUTOREGRESSIVE`,
          `MODEL_MODE_PREFILL`) indicating the operational
          mode. This significantly influences mask generation, particularly
          how causality and segment separation are handled.
      previous_chunk: Optional. Information about previously processed
          key/value chunks, often a tensor representing the previous keys/values.
          Used to correctly offset causal masks in chunked attention or
          streaming scenarios. Its shape might be
          `[batch_size, prev_kv_sequence_length, ...]`.
      bidirectional_mask: Optional `Array` of shape `[batch_size, kv_sequence_length]`.
          If provided, this boolean mask indicates tokens (e.g., image tokens)
          that are allowed to attend bidirectionally. The resulting
          block-wise bidirectional mask is combined with other masks using a
          logical OR.

    Returns:
      An `Array` representing the attention mask, broadcastable to the shape
      `[batch_size, num_heads, q_sequence_length, kv_sequence_length]`.
      Positions with `0.0` allow attention, while positions with
      `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
      Returns `None` if no masking is determined to be necessary based on
      the inputs and configuration.

    References:
      [1] JAX Pallas MHA Flash Attention:
          https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
      [2] SARATHI: Efficient LLM Inference by Piggybacking Decodes with
          Chunked Prefills - ArXiv:2308.16369 (https://arxiv.org/abs/2308.16369)
    """
    mask = None
    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, None, :, :]

    _, q_seq_len, _, _ = query.shape
    _, kv_seq_len, _, _ = key.shape
    next_pos = 0
    if previous_chunk is not None:
      next_pos = previous_chunk.shape[1]
      if mask is not None:
        mask = mask[:, :, :, next_pos : next_pos + q_seq_len, :]
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
      # In autoregression, the query position is the last position in the KV sequence.
      next_pos = kv_seq_len - 1

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != MODEL_MODE_AUTOREGRESSIVE and self.attention_type != AttentionType.FULL:
      mask_shape = (q_seq_len, kv_seq_len)
      # row_ids indicates the position of query
      # col_ids indicates the position of kv
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      # Attention mask for chunked prefill is generated in the same way
      # as mentioned in SARATHI - https://arxiv.org/abs/2308.16369
      causal_mask = (col_ids <= row_ids + next_pos)[None, None, None, :, :]

    output_mask = None
    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask

    if self.attention_type == AttentionType.LOCAL_SLIDING and output_mask is not None:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")

      row_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (q_seq_len, 1), 0) + next_pos
      col_ids_sliding = jax.lax.broadcasted_iota(jnp.int32, (1, kv_seq_len), 1)
      sliding_mask = (col_ids_sliding > (row_ids_sliding - self.sliding_window_size)) & (col_ids_sliding <= row_ids_sliding)
      output_mask = sliding_mask * output_mask
    elif self.attention_type == AttentionType.CHUNK and output_mask is not None:
      mask_shape = (q_seq_len, kv_seq_len)
      chunk_mask = _generate_chunk_attention_mask(
          mask_shape=(q_seq_len, kv_seq_len), chunk_size=self.chunk_attn_window_size, q_offset=next_pos
      )
      output_mask = chunk_mask * output_mask

    if bidirectional_mask is not None:
      image_mask = _make_bidirectional_block_mask(bidirectional_mask)
      output_mask = output_mask | image_mask[:, None, None, ...]

    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None

  def apply_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      lengths: Array | None,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array = None,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply attention"""
    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    target_hardware = self.mesh.devices[(0,) * self.mesh.devices.ndim].platform

    if use_ragged_attention and model_mode == MODEL_MODE_AUTOREGRESSIVE:
      if lengths is None:
        lengths = jnp.sum(decoder_segment_ids, axis=-1)

      if target_hardware == "tpu":
        impl = self.tpu_ragged_attention
      elif target_hardware == "gpu":
        impl = self.gpu_ragged_attention
      else:
        raise NotImplementedError(target_hardware)
      return impl(query, key, value, lengths, self.ragged_block_size)

    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
        or (self.attention_kernel == "paged")
    ):
      return self.apply_attention_dot(
          query,
          key,
          value,
          decoder_segment_ids,
          model_mode,
          previous_chunk,
          bidirectional_mask=bidirectional_mask,
          sinks=sinks,
          qk_product_einsum=qk_product_einsum,
          wv_product_einsum=wv_product_einsum,
      )
    elif self.attention_kernel in ("flash", "autoselected"):
      if target_hardware == "tpu":
        if isinstance(key, KVTensor):
          key = key.dequant()
        if isinstance(value, KVTensor):
          value = value.dequant()

        if model_mode == MODEL_MODE_AUTOREGRESSIVE:
          raise ValueError(
              """Decode not supported with flash attention.
                              Use `dot_product` instead."""
          )
        return self.tpu_flash_attention(query, key, value, decoder_segment_ids, self.attn_logits_soft_cap, sinks), None, None
      else:
        if model_mode == MODEL_MODE_AUTOREGRESSIVE:
          # fallback to dot_product as pallas gpu flash attention doesn't support decode stage
          return self.apply_attention_dot(
              query,
              key,
              value,
              decoder_segment_ids,
              model_mode,
              bidirectional_mask=bidirectional_mask,
              qk_product_einsum=qk_product_einsum,
              wv_product_einsum=wv_product_einsum,
          )
        else:
          head_axis = -2
          num_query_heads = query.shape[head_axis]
          num_kv_heads = key.shape[head_axis]
          if num_query_heads != num_kv_heads:
            # Handle cases where the number of query heads is different from the number of key/value heads.
            if num_query_heads % num_kv_heads != 0:
              raise ValueError(
                  f"Number of query heads ({num_query_heads}) must be divisible by number of key/value heads ({num_kv_heads})."
              )
            # TODO Investigate if the KV copy can be eliminated. It's likely redundant.
            q_heads_per_kv_head = num_query_heads // num_kv_heads

            key = jnp.repeat(
                key, q_heads_per_kv_head, axis=head_axis
            )  # key shape [batch_size, kv_seq_len, num_kv_heads, head_dim]
            value = jnp.repeat(
                value, q_heads_per_kv_head, axis=head_axis
            )  # value shape [batch_size, kv_seq_len, num_kv_heads, head_dim]
          out = gpu_pallas_attention.mha(query, key, value, decoder_segment_ids, sm_scale=1.0, causal=True)
          return out, None, None
    elif self.attention_kernel == "cudnn_flash_te":
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()
      if model_mode == MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                           Use `dot_product` instead."""
        )
      return self.cudnn_flash_attention(query, key, value, decoder_segment_ids, model_mode), None, None
    elif self.attention_kernel == "cudnn_flash_jax":
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()
      return *self.cudnn_jax_flash_attention(query, key, value, decoder_segment_ids, model_mode), None
    else:
      raise ValueError(f"Unexpected attention kernel {self.attention_kernel=}.")

  def gpu_ragged_attention(self, q: Array, k: Array | KVTensor, v: Array | KVTensor, lengths: Array, block_size: int):
    """gpu ragged attention"""
    batch_size, q_length, q_heads, head_dim = q.shape

    # Reshape q to match gqa's expected shape
    q_for_gqa = q.squeeze(axis=1)

    # Define logical axis names - clearer and avoids repeated calls.
    b = nn.logical_to_mesh_axes(self.ragged_lengths_names)
    bsnd = nn.logical_to_mesh_axes(self.cache_logical_axis_names)
    bnd = nn.logical_to_mesh_axes((CACHE_BATCH, CACHE_HEADS, CACHE_KV))
    bn = nn.logical_to_mesh_axes((CACHE_BATCH, CACHE_HEADS))

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(bnd, bsnd, bsnd, b, None),
        out_specs=(bnd, bn, bn),
        check_rep=False,
    )
    def wrap_ragged_attention(q: Array, k: Array, v: Array, lengths: Array, block_size: int) -> Tuple[Array, Array, Array]:
      # Use the original gqa function to get the attention output
      """
      Wraps the GQA function with appropriate sharding.

      Args:
          q: Query tensor.
          k: Key tensor.
          v: Value tensor.
          lengths: Sequence lengths.
          block_size: Block size for attention.

      Returns:
          A tuple containing the output, max, and sum tensors.
      """
      # Use the original gqa function to get the attention output
      local_out, (local_sum, local_max) = gpu_pallas_decode_attention.gqa(
          q=q,
          k=k,
          v=v,
          kv_seq_len=lengths,
          block_k=block_size,
          sm_scale=1.0,
          return_residuals=True,
          normalize_output=False,
      )
      return local_out, local_max, local_sum

    local_out, local_max, local_sum = wrap_ragged_attention(q_for_gqa, k, v, lengths, block_size)

    # Reshape local_out, local_max and local_sum to match Maxtext requirements
    local_out = local_out.reshape(batch_size, q_length, q_heads, head_dim)
    local_max = local_max.reshape(batch_size, q_length, q_heads, 1)
    local_sum = local_sum.reshape(batch_size, q_length, q_heads, 1)
    return local_out, local_max, local_sum

  def tpu_ragged_attention(
      self, query: Array, key: Array | KVTensor, value: Array | KVTensor, lengths: Array, block_size: int
  ) -> tuple[Array, Array, Array]:
    """Ragged Attention."""
    if isinstance(query, KVTensor):
      raise TypeError("Ragged attention does not currently support quantized tensors.")
    b = nn.logical_to_mesh_axes(self.ragged_lengths_names)
    bsnd = nn.logical_to_mesh_axes(self.cache_logical_axis_names)

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            bsnd,
            bsnd,
            bsnd,
            b,
            None,
        ),
        out_specs=bsnd,
        check_rep=False,
    )
    def wrap_ragged_attention(query, key, value, lengths, block_size):
      if query.shape[-2] == key.shape[-2]:
        return ragged_mha(query, key, value, lengths, block_size=block_size)
      else:
        return ragged_gqa(query, key, value, lengths, block_size=block_size)

    return wrap_ragged_attention(query, key, value, lengths, block_size)

  def tpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attn_logits_soft_cap: float | None = None,
      sinks: Array = None,
  ) -> Array:
    """TPU Flash Attention."""

    cp_size = self.config.context_parallel_size
    load_balanced_context_parallel = self.config.context_parallel_load_balance

    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))
    segment_axis_names_q = None
    segment_axis_names_kv = None
    if decoder_segment_ids is not None:
      if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
        segment_axis_names_q = nn.logical_to_mesh_axes((BATCH_NO_EXP, Q_LENGTH))
        segment_axis_names_kv = nn.logical_to_mesh_axes((BATCH_NO_EXP, KV_LENGTH))
      else:
        segment_axis_names_q = nn.logical_to_mesh_axes((BATCH, Q_LENGTH_NO_EXP))
        segment_axis_names_kv = nn.logical_to_mesh_axes((BATCH, KV_LENGTH))

    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      axis_names_splash_kernel = nn.logical_to_mesh_axes(self.flash_axis_names_splash_kernel_ep)
      axis_names_q = nn.logical_to_mesh_axes(self.flash_axis_names_q_ep)
      axis_names_kv = nn.logical_to_mesh_axes(self.flash_axis_names_kv_ep)
    else:
      axis_names_splash_kernel = nn.logical_to_mesh_axes(self.flash_axis_names_splash_kernel)
      axis_names_q = nn.logical_to_mesh_axes(self.flash_axis_names_q)
      axis_names_kv = nn.logical_to_mesh_axes(self.flash_axis_names_kv)

    global global_block_q, global_block_kv, global_block_kv_compute, global_block_q_dkv, global_block_kv_dkv
    global global_block_kv_dkv_compute, global_block_q_dq, global_block_kv_dq, global_use_fused_bwd_kernel
    global global_q_layout, global_k_layout, global_v_layout
    global_block_q = self.config.sa_block_q
    global_block_kv = self.config.sa_block_kv
    global_block_kv_compute = self.config.sa_block_kv_compute
    global_block_q_dkv = self.config.sa_block_q_dkv
    global_block_kv_dkv = self.config.sa_block_kv_dkv
    global_block_kv_dkv_compute = self.config.sa_block_kv_dkv_compute
    global_block_q_dq = self.config.sa_block_q_dq
    global_block_kv_dq = self.config.sa_block_kv_dq
    global_use_fused_bwd_kernel = self.config.sa_use_fused_bwd_kernel
    global_q_layout = self.config.sa_q_layout
    global_k_layout = self.config.sa_k_layout
    global_v_layout = self.config.sa_v_layout

    devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        "Batch dimension should be shardable among the devices in data and fsdp"
        " axis"
        f" got {query.shape[0]=}/{devices_in_data_fsdp=}"
    )

    # create_splash_attention kernel
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(global_block_q, query.shape[2]),
        block_kv=min(global_block_kv, key.shape[2]),
        block_kv_compute=min(global_block_kv_compute, key.shape[2]),
        block_q_dkv=min(global_block_q_dkv, query.shape[2]),
        block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
        block_q_dq=None if global_use_fused_bwd_kernel else min(global_block_q_dq, query.shape[2]),
        block_kv_dq=None if global_use_fused_bwd_kernel else min(global_block_kv_dq, query.shape[2]),
        use_fused_bwd_kernel=global_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
    )

    mask_shape = (query.shape[2], key.shape[2])  # (q_seq_len, kv_seq_len)
    if self.attention_type == AttentionType.FULL:
      mask = splash_attention_mask.FullMask(mask_shape)
    else:
      mask = splash_attention_mask.CausalMask(shape=mask_shape)

    # Create LoadBalancedCausalMask if cp and load_balancing
    if cp_size > 1 and load_balanced_context_parallel:
      mask = LoadBalancedCausalMask(shape=mask_shape, cp_size=cp_size)

    # TODO: figure out local_sliding attention + load_balancing, default is global
    # Apply local masking if local sliding attention is enabled.
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
      mask &= splash_attention_mask.LocalMask(
          shape=(query.shape[2], key.shape[2]),
          window_size=(self.sliding_window_size, self.sliding_window_size),
          offset=0,
      )
      # Apply local masking if local sliding attention is enabled.
      if self.attention_type == AttentionType.LOCAL_SLIDING:
        if self.sliding_window_size is None:
          raise ValueError("Sliding_window_size must be set for Local Sliding attention type")
        mask &= splash_attention_mask.LocalMask(
            shape=(query.shape[2], key.shape[2]),
            window_size=(self.sliding_window_size, self.sliding_window_size),
            offset=0,
        )
      elif self.attention_type == AttentionType.CHUNK:
        if self.chunk_attn_window_size is None:
          raise ValueError("chunk_attn_window_size must be set for chunk attention type")

        mask &= ChunkedCausalMask(shape=(query.shape[2], key.shape[2]), chunk_size=self.chunk_attn_window_size)

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

    @partial(
        jax.jit,
        static_argnames=[
            "multi_head_mask",
            "shard_head_size",
        ],
    )
    def wrap_splash_kernel(multi_head_mask, shard_head_size=1):
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=shard_head_size,  # the size of the axis if sharding over heads
          q_seq_shards=cp_size,  # axis for sequence sharding
          block_sizes=block_sizes,
          attn_logits_soft_cap=attn_logits_soft_cap,
      )
      return splash_kernel

    logical_axis_rules_head = np.array(
        [self.mesh.shape[physical_axes] for physical_axes in dict(self.config.logical_axis_rules)[HEAD]]
    )
    shard_head_size = np.prod(logical_axis_rules_head)
    splash_kernel = wrap_splash_kernel(multi_head_mask, int(shard_head_size))
    named_sharding = jax.sharding.NamedSharding(self.mesh, axis_names_splash_kernel)
    segment_axis_names_splash_kernel = splash_kernel.manual_sharding_spec(named_sharding)

    # Now call the function wrap_flash_attention which does the actual computation.
    # The splash kernel is passed as a parameter to the function. Since we have the shard map
    # decorating the wrap_flash_attention function, the data will be correctly sharded
    # meaning q will be sharded over sequence aka context length but K and V will be duplicated
    # The shardings are specified in the in_specs and out_specs of the shard_map decorator:
    # 'segment_axis_names_q' maps to ['activation_q_length', ['context']] meaning that q is sharded over the context axis
    #  'segment_axis_names_kv' maps to ['activation_kv_length', []] meaning that K and V are not sharded
    # splash_kernel is sharded over (HEAD, LENGTH)
    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names_q,
            axis_names_kv,
            axis_names_kv,
            segment_axis_names_q,
            segment_axis_names_kv,
            segment_axis_names_splash_kernel,
            None,  # no sharding for cp_size
            None,  # no sharding for load_balanced_context_parallel
            None,  # no sharding for sinks
        ),
        out_specs=axis_names_q,
        check_rep=False,
    )
    def wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
    ):
      # If load_balanced_context_parallel is enabled, reorder the key and value tensors
      # to ensure that they are contiguous in memory.
      # This is necessary for the splash attention kernel to work correctly because it expects
      # the K and V to be contiguous. Note that K and V are not sharded over the sequence aka context axis
      # This was we get the unsharded unpermuted key and value tensors
      if cp_size > 1 and load_balanced_context_parallel:
        key = max_utils.reorder_sequence(tensor=key, cp_size=cp_size, seq_dim=2, to_contiguous=True)
        value = max_utils.reorder_sequence(tensor=value, cp_size=cp_size, seq_dim=2, to_contiguous=True)
        decoder_segment_ids_unpermuted = max_utils.reorder_sequence(
            tensor=decoder_segment_ids_kv, cp_size=cp_size, seq_dim=1, to_contiguous=True
        )

      if decoder_segment_ids_q is not None:
        if cp_size > 1 and load_balanced_context_parallel:
          decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(
              decoder_segment_ids_q, decoder_segment_ids_unpermuted
          )
        else:
          # if cp=1, decoder_segment_ids_q is the same as decoder_segment_ids_kv
          decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(decoder_segment_ids_q, decoder_segment_ids_kv)
      else:
        decoder_segment_ids_tuple = None
      print(f"query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}")
      print(f"sinks.shape: {sinks.shape}")
      attention_output = jax.vmap(splash_kernel, in_axes=(0, 0, 0, 0, None))(query, key, value, decoder_segment_ids_tuple, sinks)

      return attention_output

    x = wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids,
        decoder_segment_ids,
        splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
    )

    x = jnp.transpose(x, axes=(0, 2, 1, 3))

    return x

  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports GQA, SWA (only with causal masking)
    2. Head_dim = 256 is also supported from TE-1.12 stable release with CUDNN 12.6
    """
    # These imports are only meant to work in a GPU build.
    # pylint: disable=import-outside-toplevel
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    using_context_parallelism = self.mesh.shape["context"] > 1

    if self.attention_type == AttentionType.LOCAL_SLIDING and using_context_parallelism:
      raise AssertionError("Sliding window attention is not supported when context parallelism is enabled")

    sliding_window_size = None

    if self.attention_type == AttentionType.LOCAL_SLIDING or not self.config.enable_padding_causal_mask:
      sliding_window_size = [self.sliding_window_size, 0]

    if self.attention_type == AttentionType.LOCAL_SLIDING or using_context_parallelism:
      mask_type = "causal"  # SWA and Context Parallelism only work with causal masking
      attn_mask = None
    else:
      # generate attn_mask
      mask_type = "padding_causal"  # only padding_causal mask type can take a created mask
      attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)

    dpa_layer = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=self.num_query_heads,
        num_gqa_groups=self.num_kv_heads,
        attn_mask_type=mask_type,  # 'no_mask', 'padding', 'causal', or 'padding_causal'
        attn_bias_type="no_bias",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
        attention_dropout=self.dropout_rate,
        dropout_rng_name="aqt",
        dtype=self.dtype,
        float32_logits=self.float32_logits,
        qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        scale_factor=1.0,
        transpose_batch_sequence=False,
        window_size=sliding_window_size,
        context_parallel_causal_load_balanced=self.config.context_parallel_load_balance,
        context_parallel_axis="context",
    )
    return dpa_layer(query, key, value, mask=attn_mask)

  def cudnn_jax_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[Array, Array]:
    """CUDNN Flash Attention with JAX SDPA API."""
    # These imports are only meant to work in a GPU build.
    # pylint: disable=import-outside-toplevel
    from jax._src.cudnn.fused_attention_stablehlo import (
        dot_product_attention,
        MaskType,
    )

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    if model_mode == MODEL_MODE_AUTOREGRESSIVE:
      lengths = jnp.sum(decoder_segment_ids, axis=-1)

      output, lse = dot_product_attention(
          query,
          key,
          value,
          q_seqlen=lengths,
          kv_seqlen=lengths,
          mask_type=MaskType.PADDING,
          scale=1.0,
          dropout_rate=self.dropout_rate,
          qkv_layout="BTNH",
          return_residual=True,
      )
    else:
      output, lse = dot_product_attention(
          query,
          key,
          value,
          mask_type=MaskType.CAUSAL,
          scale=1.0 / math.sqrt(head_dim),
          dropout_rate=self.dropout_rate,
          qkv_layout="BTNH",
          return_residual=True,
      )
    output = checkpoint_name(output, "context")
    lse = checkpoint_name(lse, "context")
    return output, lse

  def compute_local_attention(
      self,
      attn_weights: Array,
      value: Array | KVTensor,
      q_seq_len: int,
      model_mode: str,
      wv_product_einsum: Callable[..., Array],
      sinks: Array = None,
  ) -> tuple[Array, Array, Array]:
    """Computes the attention of a local subset of the kv cache.
    Local attention results will need to be combined with any other local attentions and normalized
    Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py

    Args:
        attn_weights (Array): Product of query and key
        value (Array): Current value
        aqt_rng (PRNGKey | None): Optional rng

    Returns:
        (local_out, local_max,): where
          local_out is local unnormalized output
          local_max is the local max of exponentials
          local_sum is the sum of exponentials for this chunk, divided by exp(local_max).
    """
    if sinks is not None:
      b, n_kv, g, t, s = attn_weights.shape
      n_q = n_kv * g
      attn_weights_reshaped = jnp.reshape(attn_weights, (b, n_q, t, s))

      sinks_param = sinks.astype(attn_weights.dtype)  # (n_q,)
      sinks_logits = sinks_param[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]  # (1, n_q, 1, 1)
      sinks_logits = jnp.broadcast_to(sinks_logits, (b, n_q, t, 1))
      combined_logits = jnp.concatenate([attn_weights_reshaped, sinks_logits], axis=-1)

      local_max = jnp.max(combined_logits, axis=-1, keepdims=True)
      local_exps_combined = jnp.exp(combined_logits - local_max)
      local_sum = jnp.sum(local_exps_combined, axis=-1, keepdims=True)

      local_exps = local_exps_combined[..., :s]
      local_exps = jnp.reshape(local_exps, (b, n_kv, g, t, s))

      # Transpose for normalize_attention
      local_max = jnp.transpose(local_max, (0, 2, 1, 3))  # (b, t, n_q, 1)
      local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))  # (b, t, n_q, 1)
    else:
      local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
      local_exps = jnp.exp(attn_weights - local_max)
      local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

      local_sum = jnp.moveaxis(local_sum, -2, 1)
      local_max = jnp.moveaxis(local_max, -2, 1)

      local_max = jnp.reshape(
          local_max, (local_max.shape[0], local_max.shape[1], local_max.shape[2] * local_max.shape[3], 1)
      )
      local_sum = jnp.reshape(
          local_sum, (local_sum.shape[0], local_sum.shape[1], local_sum.shape[2] * local_sum.shape[3], 1)
      )
    local_out = self.wv_product(local_exps, value, model_mode, wv_product_einsum)
    if model_mode == MODEL_MODE_AUTOREGRESSIVE and self.is_partition_in_decode(q_seq_len):
      local_out = partitioning.with_sharding_constraint(local_out, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
    elif model_mode == MODEL_MODE_PREFILL:
      local_out = partitioning.with_sharding_constraint(local_out, (BATCH, KV_LENGTH, HEAD, D_KV))

    if self.reshape_q and q_seq_len == 1:
      local_max = local_max[:, 0:1, :, :]
      local_sum = local_sum[:, 0:1, :, :]
      local_out = local_out[:, 0:1, :, :]

    if model_mode == MODEL_MODE_AUTOREGRESSIVE and self.is_partition_in_decode(q_seq_len):
      local_max = partitioning.with_sharding_constraint(local_max, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      local_sum = partitioning.with_sharding_constraint(local_sum, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      local_out = partitioning.with_sharding_constraint(local_out, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))

    return local_out, local_max, local_sum

  def is_partition_in_decode(self, seq_len):
    return self.config.ici_context_autoregressive_parallelism > 0 and seq_len == 1

  def apply_attention_dot(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array = None,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply Attention."""
    validate_compute_axis_order(self.compute_axis_order)
    # Casting qk_product and softmaxt computation for float32 for model stability.
    if self.float32_qk_product:
      if isinstance(key, KVTensor):
        key = key.dequant()
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    # special sharding for decode
    q_seq_len = query.shape[1]
    prefill_qkv_sharding = (BATCH, PREFILL_LENGTH, HEAD, D_KV)
    decode_qkv_sharding = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV)
    if self.is_partition_in_decode(q_seq_len):
      query = partitioning.with_sharding_constraint(query, decode_qkv_sharding)
      # avoid sharding scale tensor when using kv cache quantization
      if self.kv_quant and isinstance(key, KVTensor) and isinstance(value, KVTensor):
        key.qvalue = partitioning.with_sharding_constraint(key.qvalue, decode_qkv_sharding)
        value.qvalue = partitioning.with_sharding_constraint(value.qvalue, decode_qkv_sharding)
      else:
        key = partitioning.with_sharding_constraint(key, decode_qkv_sharding)
        value = partitioning.with_sharding_constraint(value, decode_qkv_sharding)
    elif model_mode == MODEL_MODE_PREFILL:
      query = partitioning.with_sharding_constraint(query, prefill_qkv_sharding)
      # avoid sharding scale tensor when using kv cache quantization
      if self.kv_quant and isinstance(key, KVTensor) and isinstance(value, KVTensor):
        key.qvalue = partitioning.with_sharding_constraint(key.qvalue, prefill_qkv_sharding)
        value.qvalue = partitioning.with_sharding_constraint(value.qvalue, prefill_qkv_sharding)
      else:
        key = partitioning.with_sharding_constraint(key, prefill_qkv_sharding)
        value = partitioning.with_sharding_constraint(value, prefill_qkv_sharding)

    attn_weights = self.qk_product(query, key, q_seq_len, model_mode, qk_product_einsum)
    if self.is_partition_in_decode(q_seq_len):
      attn_weights = partitioning.with_sharding_constraint(attn_weights, (KV_LENGTH, HEAD, None, None, None))
    elif model_mode == MODEL_MODE_PREFILL:
      attn_weights = partitioning.with_sharding_constraint(attn_weights, (BATCH, HEAD, None, PREFILL_LENGTH, KV_LENGTH))

    if self.attn_logits_soft_cap:
      attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
      attn_weights = attn_weights * self.attn_logits_soft_cap

    # Casting softmaxt computation for float32 for model stability.
    if self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask)
    if self.is_partition_in_decode(q_seq_len):
      attn_mask = partitioning.with_sharding_constraint(attn_mask, (KV_LENGTH, HEAD, None, None, None))
    elif model_mode == MODEL_MODE_PREFILL:
      attn_mask = partitioning.with_sharding_constraint(attn_mask, (BATCH, HEAD, None, PREFILL_LENGTH, KV_LENGTH))
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    return self.compute_local_attention(attn_weights, value, q_seq_len, model_mode, wv_product_einsum, sinks)

  def qk_product(
      self, query: Array, key: Array | KVTensor, q_seq_len: int, model_mode: str, einsum: Callable[..., Array]
  ) -> Array:
    """Query-Key product.

    Args:
      query: Query projection, in shape of [b, t, n, d]
      key: Key projection in shape of [b, s, n_kv, d]

    Returns:
      results in shape [b, n_kv, n // n_kv, t, s].

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """
    b, t, n, d = query.shape
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    if model_mode == MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, 2, n_kv, n // n_kv, d))
      result = einsum("btkgd,bskd->bkgts", query, key)
    elif self.compute_axis_order == (0, 2, 1, 3):
      query = jnp.transpose(query, axes=self.compute_axis_order)
      key = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), key)
      query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, n_kv, n // n_kv, 2, d))
      result = einsum("bkgtd,bksd->bkgts", query, key)
    else:
      raise NotImplementedError(self.compute_axis_order)
    return result

  def wv_product(
      self, attn_weights: Array, value: Array | KVTensor, model_mode: str, einsum: Callable[..., Array]
  ) -> Array:
    """weighted value product.

    Args:
      attn_weights: Computed results of qk_einsum, in shape [b, n_kv, n // n_kv, t, s]
      value: Value projection, in shape of [b, s, n_kv, d]

    Returns:
      result in shape [b, t, n, d]

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """

    if self.kv_quant:
      # manually cast to bf16 to avoid the fp32 XLA ops for speedup
      if isinstance(value, KVTensor) and self.kv_quant.dtype == jnp.float8_e4m3fn:
        value.qvalue = value.qvalue.astype(jnp.bfloat16)
    if model_mode == MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      out = einsum("bkgts,bskd->btkgd", attn_weights, value)
      b, t, n_kv, g, d = out.shape
      result = jnp.reshape(out, (b, t, n_kv * g, d))
    elif self.compute_axis_order == (0, 2, 1, 3):
      value = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), value)
      out = einsum("bkgts,bksd->bkgtd", attn_weights, value)
      b, n_kv, g, t, d = out.shape
      result = jnp.reshape(out, (b, n_kv * g, t, d))
      result = self.reverse_transepose(result, self.compute_axis_order)
    return result

  def reverse_transepose(self, transposed_array, transpose_axis_order):
    return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)

  def normalize_cudnn_attention(self, local_outs, local_stats):
    """Normalize across two cuDNN attentions

    Args:
        local_outs (list): List of outputs entries for each cudnn attention
          in shape [b, t, n, d].
        local_stats (list): List of logsumexp entries for each cudnn attention
          in shape [b, n, t].

    Returns:
        Array: Combined attention that has been normalized in shape [b, t, n, d].
    """
    # reshape stat to have shape [b, n, t, 1]
    stat0 = local_stats[0].reshape((*local_stats[0].shape, 1))
    stat1 = local_stats[1].reshape((*local_stats[1].shape, 1))
    global_stat = jnp.log(jnp.exp(stat0) + jnp.exp(stat1))
    # # transpose stat to have shape [b, t, n, 1] for elemenwise multiplication
    attn_out = local_outs[0].astype(jnp.float32) * jnp.exp(stat0 - global_stat).transpose((0, 2, 1, 3)) + local_outs[
        1
    ].astype(jnp.float32) * jnp.exp(stat1 - global_stat).transpose((0, 2, 1, 3))
    return attn_out.astype(local_stats[0].dtype)

  def normalize_attention(self, local_outs, local_maxes, local_sums):
    """Normalize across multiple localized attentions

    Args:
        local_outs (list): List of unnormalized outputs entries for each local attention
        local_maxes (list): List of max exponentials entries for each local attention
        local_sums (list): List of exponential sum entries for each local attention

    Returns:
        Array: Combined attention that has been normalized
    """
    # Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
    global_max = functools.reduce(jnp.maximum, local_maxes)
    global_sum = sum(
        (jnp.exp(local_max - global_max) * local_sum for (local_sum, local_max) in zip(local_sums, local_maxes))
    )

    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
      local_normalizer = jnp.exp(local_max - global_max) / global_sum
      attn_out += local_normalizer * local_out
    return attn_out

  def __call__(
      self,
      query,
      key,
      value,
      decoder_segment_ids,
      model_mode,
      cached_values=None,
      previous_chunk=None,
      bidirectional_mask=None,
      sinks=None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):
    if cached_values is None:
      prefill_kv_cache, ar_kv_cache = None, None
    else:
      prefill_kv_cache, ar_kv_cache = cached_values[0], cached_values[1]
    if model_mode != MODEL_MODE_TRAIN:
      assert prefill_kv_cache
      key, value, decoder_segment_ids = prefill_kv_cache

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        lengths=None,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        qk_product_einsum=self.AqtEinsum_0,
        wv_product_einsum=self.AqtEinsum_1,
    )

    # Return the "prefill" cache if it actually the combined prefill+ar kv cache
    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    key, value, decoder_segment_ids, lengths = ar_kv_cache
    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum = self.apply_attention(
        query=query,
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        lengths=lengths,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        bidirectional_mask=bidirectional_mask,
        qk_product_einsum=self.AqtEinsum_2,
        wv_product_einsum=self.AqtEinsum_3,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      if prefill_exponentials_max is not None and prefill_exponentials_sum is None:
        prefill_stat = prefill_exponentials_max
        ar_stat = ar_exponentials_max
        stats = [prefill_stat, ar_stat]
        return self.normalize_cudnn_attention(unnormalized_outputs, stats)
      else:
        return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum


@dataclasses.dataclass(repr=False)
class L2Norm(nnx.Module):
  """
  Implementation of L2Norm in JAX.

  Args:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """

  eps: float = 1e-6
  rngs: nnx.Rngs = None  # Not used in L2Norm but passed in by nnx.bridge.to_linen

  def __call__(self, x):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


def l2_norm_as_linen(self, eps: float = 1e-6):
  """
  Initializes the L2Norm module and returns it as a Linen module.

  Args:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """
  return nnx_wrappers.to_linen(L2Norm, eps=eps, metadata_fn=variable_to_logically_partitioned)


def attention_as_linen(
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
    float32_qk_product: bool = False,  # computes logits in float32 for stability.
    float32_logits: bool = False,  # cast logits in float32 for stability.
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
    # Temperature tuning parameters used for Llama4
    temperature_tuning: bool = False,
    temperature_tuning_scale: float = 0.1,
    temperature_tuning_floor_scale: float = 8192.0,
    # Shard the query activation as the same as the key and value.
    # TODO: Find a better sharding axis name.
    # TODO: Further break down the Training and Inference axes for the q, k, v.
    prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
    ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
    out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
    ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
    prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
    decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
    prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
    decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    reshape_q: bool = False,
    is_nope_layer: bool = False,
    is_vision: bool = False,
    model_mode: str = MODEL_MODE_TRAIN,
    name: str | None = None,
):
  """A factory function to create an Attention as a Linen module.

  This function serves as a bridge to use the NNX-based `Attention` within a
  Linen model.
  """
  return nnx_wrappers.to_linen(
      Attention,
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
      temperature_tuning=temperature_tuning,
      temperature_tuning_scale=temperature_tuning_scale,
      temperature_tuning_floor_scale=temperature_tuning_floor_scale,
      prefill_query_axis_names=prefill_query_axis_names,
      prefill_key_axis_names=prefill_key_axis_names,
      prefill_value_axis_names=prefill_value_axis_names,
      query_axis_names=query_axis_names,
      key_axis_names=key_axis_names,
      value_axis_names=value_axis_names,
      ep_query_axis_names=ep_query_axis_names,
      ep_key_axis_names=ep_key_axis_names,
      ep_value_axis_names=ep_value_axis_names,
      input_axis_names=input_axis_names,
      ep_input_axis_names=ep_input_axis_names,
      out_axis_names=out_axis_names,
      ep_out_axis_names=ep_out_axis_names,
      prefill_input_axis_names=prefill_input_axis_names,
      decode_input_axis_names=decode_input_axis_names,
      prefill_out_axis_names=prefill_out_axis_names,
      decode_out_axis_names=decode_out_axis_names,
      prefill_cache_axis_order=prefill_cache_axis_order,
      ar_cache_axis_order=ar_cache_axis_order,
      compute_axis_order=compute_axis_order,
      reshape_q=reshape_q,
      is_nope_layer=is_nope_layer,
      is_vision=is_vision,
      model_mode=model_mode,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )


class Attention(nnx.Module):
  """Attention Module.

    This module implements multi-headed attention as described in the
    original Transformer paper. It projects the inputs into query, key, and
    value vectors, applies the attention mechanism, and projects the results to
    an output vector.

    Attributes:
      config: The model configuration.
      num_query_heads: Number of query attention heads.
      num_kv_heads: Number of key-value attention heads.
      head_dim: The dimension of each attention head.
      max_target_length: Maximum sequence length.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use (e.g., 'dot_product', 'flash').
      inputs_q_shape: Query inputs shape for initialization, required by NNX.
      inputs_kv_shape: Key/value inputs shape for initialization, required by NNX.
      dtype: The data type for computation.
      weight_dtype: The data type for weights.
      max_prefill_predict_length: Maximum length for prefill.
      dropout_rate: The dropout rate.
      kernel_init: Initializer for the kernel of the dense layers.
      float32_qk_product: If True, compute query-key product in float32.
      float32_logits: If True, cast logits to float32 before softmax.
      quant: Quantization configuration.
      kv_quant: KV cache quantization configuration.
      attention_type: The type of attention (e.g., 'global', 'local_sliding').
      attn_logits_soft_cap: Soft cap for attention logits.
      ... and other configuration parameters.
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
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = False,  # cast logits in float32 for stability.
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
      # Temperature tuning parameters used for Llama4
      temperature_tuning: bool = False,
      temperature_tuning_scale: float = 0.1,
      temperature_tuning_floor_scale: float = 8192.0,
      # Shard the query activation as the same as the key and value.
      # TODO: Find a better sharding axis name.
      # TODO: Further break down the Training and Inference axes for the q, k, v.
      prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
      ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
      out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
      ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
      prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
      decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      reshape_q: bool = False,
      is_nope_layer: bool = False,
      is_vision: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      base_kv_cache: bool = True,
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the Attention module.

    Attributes:
      config: The model configuration.
      num_query_heads: Number of query attention heads.
      num_kv_heads: Number of key-value attention heads.
      head_dim: The dimension of each attention head.
      max_target_length: Maximum sequence length.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use (e.g., 'dot_product', 'flash').
      inputs_q_shape: Query inputs shape for initialization, required by NNX.
      inputs_kv_shape: Key/value inputs shape for initialization, required by NNX.
      dtype: The data type for computation.
      weight_dtype: The data type for weights.
      max_prefill_predict_length: Maximum length for prefill.
      dropout_rate: The dropout rate.
      kernel_init: Initializer for the kernel of the dense layers.
      float32_qk_product: If True, compute query-key product in float32.
      float32_logits: If True, cast logits to float32 before softmax.
      quant: Quantization configuration.
      kv_quant: KV cache quantization configuration.
      attention_type: The type of attention (e.g., 'global', 'local_sliding').
      attn_logits_soft_cap: Soft cap for attention logits.
      sliding_window_size: The size of the sliding window for local attention.
      use_ragged_attention: Whether to use ragged attention for decoding.
      ragged_block_size: The block size for ragged attention.
      use_qk_norm: Whether to apply normalization to query and key.
      query_pre_attn_scalar: Scalar to apply to query before attention.
      use_bias_in_projections: Whether to use bias in Q, K, V, and output projections.
      temperature_tuning: Whether to use temperature tuning for attention.
      temperature_tuning_scale: The scale for temperature tuning.
      temperature_tuning_floor_scale: The floor scale for temperature tuning.
      ... other configuration parameters.
      is_nope_layer: Whether this is a "NoPE" (No Position-Embedding) layer.
      is_vision: Whether this is a vision attention layer.
      model_mode: The model's operational mode (e.g., 'train', 'prefill').
      base_kv_cache: Whether to use base (non-MLA) kv cache, if KVCache is used
      rngs: RNG state for initialization, passed by the nnx.to_linen wrapper.
    """

    self.config = config
    self.num_query_heads = num_query_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.max_target_length = max_target_length
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.max_prefill_predict_length = max_prefill_predict_length
    self.dropout_rate = dropout_rate
    self.kernel_init = kernel_init
    self.float32_qk_product = float32_qk_product
    self.float32_logits = float32_logits
    self.quant = quant
    self.kv_quant = kv_quant
    self.attention_type = attention_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size
    self.use_qk_norm = use_qk_norm
    self.query_pre_attn_scalar = query_pre_attn_scalar
    self.use_bias_in_projections = use_bias_in_projections
    self.temperature_tuning = temperature_tuning
    self.temperature_tuning_scale = temperature_tuning_scale
    self.temperature_tuning_floor_scale = temperature_tuning_floor_scale
    self.prefill_query_axis_names = prefill_query_axis_names
    self.prefill_key_axis_names = prefill_key_axis_names
    self.prefill_value_axis_names = prefill_value_axis_names
    self.query_axis_names = query_axis_names
    self.key_axis_names = key_axis_names
    self.value_axis_names = value_axis_names
    self.ep_query_axis_names = ep_query_axis_names
    self.ep_key_axis_names = ep_key_axis_names
    self.ep_value_axis_names = ep_value_axis_names
    self.input_axis_names = input_axis_names
    self.ep_input_axis_names = ep_input_axis_names
    self.out_axis_names = out_axis_names
    self.ep_out_axis_names = ep_out_axis_names
    self.prefill_input_axis_names = prefill_input_axis_names
    self.decode_input_axis_names = decode_input_axis_names
    self.prefill_out_axis_names = prefill_out_axis_names
    self.decode_out_axis_names = decode_out_axis_names
    self.prefill_cache_axis_order = prefill_cache_axis_order
    self.ar_cache_axis_order = ar_cache_axis_order
    self.compute_axis_order = compute_axis_order
    self.reshape_q = reshape_q
    self.is_nope_layer = is_nope_layer
    self.is_vision = is_vision
    self.model_mode = model_mode
    self.rngs = rngs

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.KVCache_0 = (
        self.init_kv_caches(inputs_kv_shape=inputs_kv_shape)
        if self.model_mode != MODEL_MODE_TRAIN and base_kv_cache
        else None
    )

    self.rotary_embedding = self.init_rotary_embedding()

    self.attention_op = AttentionOp(
        config=self.config,
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        kv_quant=self.kv_quant,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        compute_axis_order=self.compute_axis_order,
        reshape_q=self.reshape_q,
        attention_type=self.attention_type,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        chunk_attn_window_size=self.config.chunk_attn_window_size,
        use_ragged_attention=self.use_ragged_attention,
        ragged_block_size=self.ragged_block_size,
        rngs=self.rngs,
    )
    # When paged attention is enabled, paged attention op is used for all model modes except TRAIN,
    # which uses default attention op.
    if self.config.attention == "paged":
      self.paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=(self.config.max_target_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=(self.config.max_prefill_predict_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=self.head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          rngs=self.rngs,
      )

    if self.config.fused_qkv:
      self.qkv_proj = self.init_qkv_w(inputs_shape=inputs_q_shape)
    else:
      self.query = self.init_query_w(inputs_q_shape=inputs_q_shape)
      self.key = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)
      self.value = self.init_kv_w(inputs_kv_shape=inputs_kv_shape)

    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

    if self.config.attention_sink:
      self.sinks = nnx.Param(
          default_bias_init(self.rngs.params(), (self.config.num_query_heads,), self.weight_dtype),
          sharding=(None,),
      )
    else:
      self.sinks = None

    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
    if self.use_qk_norm and not is_llama4_decoder_block:
      self.query_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
      self.key_norm = RMSNorm(
          num_features=self.head_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.query_norm = None
      self.key_norm = None

  def init_query_w(self, inputs_q_shape: Tuple) -> nnx.Module:
    """Query projection initialization."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

    def query_init(*args):
      # pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    kernel_axes = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("embed", "q_heads", "kv")
    )
    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_q_shape),
        out_features_shape=(self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=query_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

    return self.query(inputs_q)

  def init_kv_w(self, inputs_kv_shape: Tuple) -> nnx.Module:
    """Initializes the key or value projection.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      A DenseGeneral module that performs the key or value projection.
    """
    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")

    kernel_axes = (
        (None, None, None)
        if self.config.ici_context_autoregressive_parallelism > 1
        else ("embed", "kv_heads", "kv_head_dim")
    )

    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_kv_shape),
        out_features_shape=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> nnx.Module:
    """Applies the key or value projection.

    Args:
      inputs_kv: The input tensor to project.
      proj_name: The name of the projection ("key" or "value").

    Returns:
      The projected key or value tensor.

    Raises:
      ValueError: If `proj_name` is not one of the supported values
        ("key", "value").

    """
    if proj_name == "key":
      return self.key(inputs_kv)
    elif proj_name == "value":
      return self.value(inputs_kv)
    else:
      raise ValueError(f"proj_name must be 'key' or 'value', but got {proj_name}")

  def init_qkv_w(self, inputs_shape: Tuple) -> nnx.Module:
    return DenseGeneral(
        in_features_shape=self.convert_dense_general_inputs_shape(inputs_shape),
        out_features_shape=(3, self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "qkv", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def qkv_projection(self, inputs: Array, proj_name: str):
    """Fused QKV projection"""

    qkv_proj = self.qkv_proj(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def init_out_w(self, output_dim: int) -> nnx.Module:
    """out projection"""
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed")
    )
    return DenseGeneral(
        in_features_shape=(self.num_query_heads, self.head_dim),
        out_features_shape=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,  # trade speed with memory
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.use_bias_in_projections,
        rngs=self.rngs,
    )

  def out_projection(self, out: Array) -> Array:
    """out projection"""

    return self.out(out)

  def convert_dense_general_inputs_shape(
      self,
      inputs_shape: tuple[int, ...] | None = None,
      axis: Union[Iterable[int], int] = -1,
  ) -> Union[Iterable[int], int]:
    axis = canonicalize_tuple(axis)
    return tuple(inputs_shape[ax] for ax in normalize_axes(axis, len(inputs_shape)))

  def init_rotary_embedding(self):
    """Initializes the rotary embeddings, handling different model types.

    Returns:
      The rotary embedding module that will be used in the model.
    """
    if self.config.attention_type == AttentionType.MLA.value:
      # For MLA attention RoPE is applied to only `self.qk_rope_head_dim` portion the heads.
      rope_embedding_dims = self.qk_rope_head_dim
    else:
      rope_embedding_dims = self.head_dim

    rope_type = self.config.rope_type.lower()
    rope_use_scale = self.config.rope_use_scale
    if self.is_vision:
      rotary_embedding = LlamaVisionRotaryEmbedding(
          image_size=self.config.image_size_for_vit,
          patch_size=self.config.patch_size_for_vit,
          hidden_size=self.config.hidden_size_for_vit,
          num_attention_heads=self.config.num_attention_heads_for_vit,
          rope_theta=self.config.rope_theta_for_vit,
          rngs=self.rngs,
      )
    elif self.config.model_name.startswith("llama3.1") or rope_type.startswith("llama3.1"):
      rotary_embedding = LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          use_scale=rope_use_scale,
          rngs=self.rngs,
      )
    elif rope_type.startswith("yarn"):
      rotary_embedding = YarnRotaryEmbedding(
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_theta=self.config.rope_max_timescale,
          rope_factor=self.config.rope_factor,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          rngs=self.rngs,
      )
    else:
      max_timescale = self.config.rope_max_timescale
      # For local attention use local_rope_max_timescale if it's is positive
      if self.attention_type == AttentionType.LOCAL_SLIDING and self.config.local_rope_max_timescale > 0:
        max_timescale = self.config.local_rope_max_timescale
      rotary_embedding = RotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          rngs=self.rngs,
      )
    return rotary_embedding

  def apply_rotary_embedding(self, inputs: Array, inputs_positions: Optional[Array | None] = None):
    """Applies rotary embeddings, handling different model types.

    Args:
      inputs: The input tensor to apply rotary embeddings to.
      inputs_positions: The positions of the inputs.
      name: A name for the embedding layer.

    Returns:
      The input tensor with rotary embeddings applied.
    """
    return self.rotary_embedding(inputs, inputs_positions)

  def init_kv_caches(self, inputs_kv_shape: Tuple):
    """Initializes KVCache.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      A KVCache module instance.

    """
    batch_size, _, _ = inputs_kv_shape
    # During initialization, seq_len of inputs_kv is max_target_length,
    # which is not always correct for some functions in KVCache.
    # However, KVCache internal cache shapes are based on max_prefill_length
    # and max_target_length, not the passed seq_len.
    # We can use a placeholder value. The correct fix might involve refactoring
    # KVCache.
    placeholder_seq_len = 1

    return kvcache.KVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=placeholder_seq_len,
        value_seq_len=placeholder_seq_len,
        key_heads=self.num_kv_heads,
        value_heads=self.num_kv_heads,
        key_head_size=self.head_dim,
        value_head_size=self.head_dim,
        dtype=self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        use_chunked_prefill=self.config.use_chunked_prefill,
        model_mode=self.model_mode,
        rngs=self.rngs,
    )

  def update_kv_caches(self, key, value, decoder_segment_ids, model_mode, previous_chunk):
    """Updates the KV caches for prefill and autoregressive modes.

    This method uses a kvcache module to update and retrieve the key-value
    caches based on the current operational mode.

    Args:
      key: The key tensor for the current attention computation.
      value: The value tensor for the current attention computation.
      decoder_segment_ids: Segment IDs for the decoder, used for masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      previous_chunk: Information about previously processed chunks, used for
        chunked prefill.

    Returns:
      A list containing two elements:
      - The prefill key-value cache, or None.
      - The autoregressive key-value cache, or None.
    """
    prefill_kv_cache, ar_kv_cache = self.KVCache_0(
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )
    return [prefill_kv_cache, ar_kv_cache]

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Any = None,
  ):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention, and project the results to an output vector.

    This method handles three modes:
    1.  **Training**: The KV cache is ignored.
    2.  **Prefill**: The KV cache is filled with the key-value pairs from the input sequence.
    3.  **Autoregressive Decoding**: The KV cache is used to provide context from previous steps.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: Input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: Key/values of shape `[batch, kv_length, kv_features]`.
      inputs_positions: Input positions for rotary embeddings.
      decoder_segment_ids: Segment IDs for masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      deterministic: If True, disables dropout.
      previous_chunk: Information about previously processed chunks for chunked prefill.
      slot: The batch slot index for paged attention.
      page_state: The current state of the paged attention manager.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = nn.with_logical_constraint(inputs_q, self.prefill_input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.prefill_input_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      inputs_q = nn.with_logical_constraint(inputs_q, self.ep_input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.ep_input_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)
    else:
      inputs_q = nn.with_logical_constraint(inputs_q, self.decode_input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.decode_input_axis_names)

    # apply projection.
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")

    is_llama4_decoder_block = self.config.decoder_block == DecoderBlockType.LLAMA4
    # NOTE: llama 4 does L2 normalization after RoPE
    if self.use_qk_norm and not is_llama4_decoder_block:
      query = self.query_norm(query)
      key = self.key_norm(key)

    # NOTE: is_nope_layer should be used in attention mask and also used in attention tuning
    use_rope = not self.is_nope_layer
    use_qk_norm = self.use_qk_norm and use_rope

    if use_rope:
      query = self.apply_rotary_embedding(query, inputs_positions=inputs_positions)
      key = self.apply_rotary_embedding(key, inputs_positions=inputs_positions)

    if use_qk_norm and is_llama4_decoder_block:
      l2_norm = L2Norm(eps=self.config.normalization_layer_epsilon)
      query = l2_norm(query)
      key = l2_norm(key)

    # apply query_pre_attn_scalar if it's present.
    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = query * self.query_pre_attn_scalar

    if self.temperature_tuning and not use_rope:
      attn_scales = (
          jnp.log(jnp.floor((inputs_positions.astype(self.dtype) + 1.0) / self.temperature_tuning_floor_scale) + 1.0)
          * self.temperature_tuning_scale
          + 1.0
      )
      query = (query * attn_scales[:, :, jnp.newaxis, jnp.newaxis]).astype(self.dtype)

    if model_mode == MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      query = nn.with_logical_constraint(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = nn.with_logical_constraint(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = nn.with_logical_constraint(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      query = nn.with_logical_constraint(query, self.ep_query_axis_names)
      key = nn.with_logical_constraint(key, self.ep_key_axis_names)
      value = nn.with_logical_constraint(value, self.ep_value_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    assert not self.config.quantize_kvcache or self.kv_quant

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      cached_values = [None, None]
      if model_mode != MODEL_MODE_TRAIN:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      out = self.attention_op(
          query, key, value, decoder_segment_ids, model_mode, cached_values, previous_chunk, bidirectional_mask, self.sinks
      )

    if model_mode == MODEL_MODE_PREFILL:
      out = nn.with_logical_constraint(out, self.prefill_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      out = nn.with_logical_constraint(out, self.ep_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      out = nn.with_logical_constraint(out, self.out_axis_names)
    else:
      out = nn.with_logical_constraint(out, self.decode_out_axis_names)
    out = self.out_projection(out)
    out = checkpoint_name(out, "out_proj")
    return out


def mla_as_linen(
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
    float32_qk_product: bool = False,  # computes logits in float32 for stability.
    float32_logits: bool = False,  # cast logits in float32 for stability.
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
    use_qk_norm: bool = False,
    query_pre_attn_scalar: float | None = None,
    use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
    # Temperature tuning parameters used for Llama4
    temperature_tuning: bool = False,
    temperature_tuning_scale: float = 0.1,
    temperature_tuning_floor_scale: float = 8192.0,
    # Shard the query activation as the same as the key and value.
    # TODO: Find a better sharding axis name.
    # TODO: Further break down the Training and Inference axes for the q, k, v.
    prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
    query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
    ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
    input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
    ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
    out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
    ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
    prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
    decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
    prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
    decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    reshape_q: bool = False,
    is_nope_layer: bool = False,
    is_vision: bool = False,
    model_mode: str = MODEL_MODE_TRAIN,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    max_position_embeddings: int = 4096 * 4,
    original_max_position_embeddings: int = 4096,
    mscale: float = 1.0,  # scaling factor for softmax
    rope_factor: float = 40.0,  # rotary embedding factor
    name: str | None = None,
):
  """A factory function to create an MLA as a Linen module.

  This function serves as a bridge to use the NNX-based `MLA` within a
  Linen model.
  """
  return nnx_wrappers.to_linen(
      MLA,
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
      temperature_tuning=temperature_tuning,
      temperature_tuning_scale=temperature_tuning_scale,
      temperature_tuning_floor_scale=temperature_tuning_floor_scale,
      prefill_query_axis_names=prefill_query_axis_names,
      prefill_key_axis_names=prefill_key_axis_names,
      prefill_value_axis_names=prefill_value_axis_names,
      query_axis_names=query_axis_names,
      key_axis_names=key_axis_names,
      value_axis_names=value_axis_names,
      ep_query_axis_names=ep_query_axis_names,
      ep_key_axis_names=ep_key_axis_names,
      ep_value_axis_names=ep_value_axis_names,
      input_axis_names=input_axis_names,
      ep_input_axis_names=ep_input_axis_names,
      out_axis_names=out_axis_names,
      ep_out_axis_names=ep_out_axis_names,
      prefill_input_axis_names=prefill_input_axis_names,
      decode_input_axis_names=decode_input_axis_names,
      prefill_out_axis_names=prefill_out_axis_names,
      decode_out_axis_names=decode_out_axis_names,
      prefill_cache_axis_order=prefill_cache_axis_order,
      ar_cache_axis_order=ar_cache_axis_order,
      compute_axis_order=compute_axis_order,
      reshape_q=reshape_q,
      is_nope_layer=is_nope_layer,
      is_vision=is_vision,
      model_mode=model_mode,
      q_lora_rank=q_lora_rank,
      kv_lora_rank=kv_lora_rank,
      qk_nope_head_dim=qk_nope_head_dim,
      qk_rope_head_dim=qk_rope_head_dim,
      v_head_dim=v_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      mscale=mscale,
      rope_factor=rope_factor,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
      abstract_init=False,
  )


class MLA(Attention):
  """Multi-Head Latent Attention (MLA) layer."""

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
      float32_qk_product: bool = False,  # computes logits in float32 for stability.
      float32_logits: bool = False,  # cast logits in float32 for stability.
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      use_qk_norm: bool = False,
      query_pre_attn_scalar: float | None = None,
      use_bias_in_projections: bool = False,  # Set to True will enable bias in q, k, v, o projections
      # Temperature tuning parameters used for Llama4
      temperature_tuning: bool = False,
      temperature_tuning_scale: float = 0.1,
      temperature_tuning_floor_scale: float = 8192.0,
      # Shard the query activation as the same as the key and value.
      # TODO: Find a better sharding axis name.
      # TODO: Further break down the Training and Inference axes for the q, k, v.
      prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_HEAD, KV_HEAD_DIM),
      query_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      key_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      value_axis_names: AxisNames = (KV_BATCH, LENGTH_NO_EXP, KV_HEAD, KV_HEAD_DIM),
      ep_query_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_key_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      ep_value_axis_names: AxisNames = (KV_BATCH_NO_EXP, LENGTH, KV_HEAD, KV_HEAD_DIM),
      input_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, EMBED),
      ep_input_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, EMBED),
      out_axis_names: AxisNames = (BATCH, LENGTH_NO_EXP, HEAD, D_KV),
      ep_out_axis_names: AxisNames = (BATCH_NO_EXP, LENGTH, HEAD, D_KV),
      prefill_input_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, EMBED),
      decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED),
      prefill_out_axis_names: AxisNames = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV),
      decode_out_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      reshape_q: bool = False,
      is_nope_layer: bool = False,
      is_vision: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      q_lora_rank: int = 0,
      kv_lora_rank: int = 512,
      qk_nope_head_dim: int = 128,
      qk_rope_head_dim: int = 64,
      v_head_dim: int = 128,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      mscale: float = 1.0,  # scaling factor for softmax
      rope_factor: float = 40.0,  # rotary embedding factor
      name: str | None = None,
      rngs: Optional[nnx.Rngs] = None,
  ):
    """Initializes the MLA module.

    Args:
      config: The model configuration.
      ... and other configuration parameters for MLA attention.
      rngs: The random number generators for initialization, passed by the nnx.to_linen wrapper.
    """
    base_kv_cache = config.attention != "paged" and config.mla_naive_kvcache

    # Setting these before call to super because a field is used in super
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.mscale = mscale
    self.rope_factor = rope_factor

    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

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
        temperature_tuning=temperature_tuning,
        temperature_tuning_scale=temperature_tuning_scale,
        temperature_tuning_floor_scale=temperature_tuning_floor_scale,
        prefill_query_axis_names=prefill_query_axis_names,
        prefill_key_axis_names=prefill_key_axis_names,
        prefill_value_axis_names=prefill_value_axis_names,
        query_axis_names=query_axis_names,
        key_axis_names=key_axis_names,
        value_axis_names=value_axis_names,
        ep_query_axis_names=ep_query_axis_names,
        ep_key_axis_names=ep_key_axis_names,
        ep_value_axis_names=ep_value_axis_names,
        input_axis_names=input_axis_names,
        ep_input_axis_names=ep_input_axis_names,
        out_axis_names=out_axis_names,
        ep_out_axis_names=ep_out_axis_names,
        prefill_input_axis_names=prefill_input_axis_names,
        decode_input_axis_names=decode_input_axis_names,
        prefill_out_axis_names=prefill_out_axis_names,
        decode_out_axis_names=decode_out_axis_names,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
        reshape_q=reshape_q,
        is_nope_layer=is_nope_layer,
        is_vision=is_vision,
        model_mode=model_mode,
        base_kv_cache=base_kv_cache,
        rngs=rngs,
    )

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.MlaKVCache_0 = self.init_mla_kv_caches(inputs_kv_shape) if model_mode != MODEL_MODE_TRAIN else None

    # Assert required configuration parameters for MLA attention.
    assert (
        self.config.attention_type == AttentionType.MLA.value
    ), f"MLA requires MLA attention type {AttentionType.MLA.value}"
    assert self.kv_lora_rank > 0, "KV LoRA rank must be > 0"
    assert self.qk_nope_head_dim > 0, "QK NoPe head dim must be > 0"
    assert self.qk_rope_head_dim > 0, "QK RoPE head dim must be > 0"
    assert self.v_head_dim > 0, "V head dim must be > 0"
    assert self.num_query_heads == self.num_kv_heads, "MLA requires equal number of query and kv heads"
    assert not self.config.fused_qkv, "Fused QKV is not supported for MLA"

    if self.q_lora_rank == 0:
      # Standard Q projection (without LoRA).
      self.query = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          rngs=self.rngs,
      )
    else:
      # LoRA path for Q.
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
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("q_lora", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
          rngs=self.rngs,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )
    self.kv_norm = RMSNorm(
        num_features=self.kv_lora_rank,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.wkv_b = DenseGeneral(
        in_features_shape=self.kv_lora_rank,
        out_features_shape=(
            self.num_query_heads,
            (self.qk_nope_head_dim + self.v_head_dim),
        ),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        rngs=self.rngs,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * jnp.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    # Setup paged attention op
    if self.config.attention == "paged":
      # Set head_dim to the max of qk_head_dim and v_head_dim. The current paged
      # attention kernel requires the head_dim to be the same for q, k, v.
      head_dim = max(self.qk_head_dim, self.v_head_dim)
      # Align head_dim to the pagedattn_head_dim_alignment if specified.
      if self.config.pagedattn_head_dim_alignment > 0:
        alignment = self.config.pagedattn_head_dim_alignment
        head_dim = (head_dim + alignment - 1) // alignment * alignment
      self.ds_paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=(self.config.max_target_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=(self.config.max_prefill_predict_length + self.config.pagedattn_tokens_per_page - 1)
          // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
          rngs=self.rngs,
      )

  def mla_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode) -> Array:
    """Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0."""
    # Set softmax scaling.
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * jnp.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    if self.q_lora_rank == 0:
      q = self.query(inputs_q)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q)  # [B, L, q_lora_rank]
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      q = self.wq_b(low_rank_q)  # [B, L, n_heads * qk_head_dim]

    # Split into non-positional and rotary parts.
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions=inputs_positions)
    # Query projection is scaled by self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale

    if model_mode == MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      query = nn.with_logical_constraint(query, self.ep_query_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
    return query

  def mla_get_key_value(self, low_rank_main, key_rope, model_mode):
    """get (key,value) pair from mla"""
    kv_out = self.wkv_b(low_rank_main)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))

    key = jnp.concatenate([key_nope, key_rope], axis=-1)

    if model_mode == MODEL_MODE_PREFILL:
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      key = nn.with_logical_constraint(key, self.ep_key_axis_names)
      value = nn.with_logical_constraint(value, self.ep_value_axis_names)
    else:
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)
    return key, value

  def init_mla_kv_caches(self, inputs_kv_shape: Tuple):
    """Initializes MlaKVCache.

    Args:
      inputs_kv_shape: Key/value inputs shape for initialization.

    Returns:
      An MlaKVCache module instance.

    Raises:
      ValueError: If the configuration is invalid.

    """
    batch_size, _, _ = inputs_kv_shape
    # During initialization, seq_len of inputs_kv is max_target_length,
    # which is not always correct for some functions in MlaKVCache.
    # However, MlaKVCache internal cache shapes are based on max_prefill_length
    # and max_target_length, not the passed seq_len.
    # We can use a placeholder value. The correct fix might involve refactoring
    # MlaKVCache.
    placeholder_seq_len = 1

    return kvcache.MlaKVCache(
        max_prefill_length=self.max_prefill_predict_length,
        max_target_length=self.max_target_length,
        batch=batch_size,
        key_seq_len=placeholder_seq_len,
        value_seq_len=placeholder_seq_len,
        key_head_size=self.kv_lora_rank,
        value_head_size=self.qk_rope_head_dim,
        dtype=self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        model_mode=self.model_mode,
        use_chunked_prefill=self.config.use_chunked_prefill,
        rngs=self.rngs,
    )

  def update_mla_kv_caches(self, low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk=None):
    """Updates the MLA (Multi-Head Latent Attention) KV caches.

    This method is specific to the MLA attention mechanism. It calls the
    `mla_kv_cache_as_linen` module to update and retrieve the caches, which
    store latent representations (`low_rank_main`) and RoPE-applied keys
    (`key_rope`). It then reconstructs the full key and value tensors from
    the cached components.

    Args:
      low_rank_main: The main latent component of the key.
      key_rope: The RoPE-applied component of the key.
      decoder_segment_ids: Segment IDs for decoder masking.
      model_mode: The operational mode ('train', 'prefill', 'autoregressive').
      previous_chunk: Information about previously processed chunks, for
        chunked prefill.

    Returns:
      A list containing two elements:
      - The prefill key-value cache, reconstructed from the MLA cache, or None.
      - The autoregressive key-value cache, reconstructed from the MLA cache, or None.
    """

    prefill_mla_cache, ar_mla_cache = self.MlaKVCache_0(
        key_latent=low_rank_main,
        key_rope=key_rope,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )

    if prefill_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids = prefill_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      prefill_kv_cache = key, value, decoder_segment_ids
    else:
      prefill_kv_cache = None

    if ar_mla_cache:
      low_rank_main, key_rope, decoder_segment_ids, lengths = ar_mla_cache
      key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
      ar_kv_cache = key, value, decoder_segment_ids, lengths
    else:
      ar_kv_cache = None
    return [prefill_kv_cache, ar_kv_cache]

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array, decoder_segment_ids, model_mode, previous_chunk):
    """MLA key/value projection with integrated rotary embedding."""
    low_rank = self.wkv_a(inputs)
    low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)
    low_rank_main = self.kv_norm(low_rank_main)

    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions=inputs_positions)

    key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
    cached_values = [None, None]
    if self.config.attention != "paged" and model_mode != MODEL_MODE_TRAIN:
      if self.config.mla_naive_kvcache:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      else:
        cached_values = self.update_mla_kv_caches(low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk)

    return key, value, cached_values

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Optional[Any] = None,
  ) -> Array:
    """Forward pass for MLA, reusing `AttentionOp` for the actual attention.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      model_mode: "train", "prefill", or "autoregressive".
      deterministic: Disables dropout if set to True.
      previous_chunk: Information about previously processed chunks for chunked prefill.
      slot: The batch slot index for paged attention.
      page_state: The current state of the paged attention manager.
      bidirectional_mask: A mask for bidirectional attention, used in multimodal models.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = nn.with_logical_constraint(inputs_q, self.prefill_input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.prefill_input_axis_names)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      inputs_q = nn.with_logical_constraint(inputs_q, self.ep_input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.ep_input_axis_names)
    else:
      inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
      inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    query = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.ds_paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      unnormalized_out = unnormalized_out[..., : self.v_head_dim]
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, cached_values)

    if model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      out = nn.with_logical_constraint(out, self.ep_out_axis_names)
    else:
      out = nn.with_logical_constraint(out, self.out_axis_names)

    out = self.out_projection(out)
    return out


# pylint: disable=protected-access
class LoadBalancedCausalMask(splash_attention_mask._ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.
  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int
  shape: tuple[int, int]
  cp_size: int

  def __init__(self, shape: tuple[int, int], offset: int = 0, shard_count: int = 1, cp_size: int = 4):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    arr = np.arange(shape[0])
    # we reorder the mask to be load balanced following the same approach as
    # used to reorder the input tokens
    out = max_utils.reorder_mask_load_balancing(arr[None, :, None, None], cp_size, 1)
    q_sequence = out[0, :, 0, 0]

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )
    self.q_sequence = q_sequence

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape and self.offset == other.offset and np.array_equal(self.q_sequence, other.q_sequence)

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.offset,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )
