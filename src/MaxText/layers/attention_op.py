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
#  pytype: disable=module-attr
"""Attentions Ops Layers."""
import dataclasses
import functools
from functools import partial
import math
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.gpu import attention as gpu_pallas_attention
from jax.experimental.pallas.ops.gpu import decode_attention as gpu_pallas_decode_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from MaxText.common_types import (
    Array,
    AttentionType,
    AxisIdxes,
    AxisNames,
    BATCH,
    BATCH_NO_EXP,
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    CACHE_KV,
    CACHE_SCALE_BATCH,
    CACHE_SCALE_HEADS,
    CACHE_SCALE_KV,
    CACHE_SCALE_SEQUENCE,
    CACHE_SEQUENCE,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DEFAULT_MASK_VALUE,
    DType,
    D_KV,
    EP_AS_CONTEXT,
    EP_AS_FSDP,
    HEAD,
    KV_LENGTH,
    LENGTH,
    LENGTH_NO_EXP,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_LENGTH,
    Q_LENGTH,
    Q_LENGTH_NO_EXP,
)
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import variable_to_logically_partitioned
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.sharding import logical_to_mesh_axes, maybe_shard_with_name
from maxtext.inference import page_manager
from maxtext.inference.kvcache import KVQuant, KVTensor
from maxtext.kernels.attention import jax_flash_attention
from maxtext.kernels.attention.ragged_attention import ragged_gqa
from maxtext.kernels.attention.ragged_attention import ragged_mha
from maxtext.utils import max_utils
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_kernel
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error

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


def validate_flash_attention_with_sinks_on_gpu(sinks: Array | None) -> None:
  """Helper function to check for sinks with flash attention on GPU."""
  if sinks is not None:
    raise ValueError("The flash attention with sinks is not supported on GPU yet.")


# TODO(agagik): change splash_attention_mask._ComputableMask to be non protected
class ChunkedCausalMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.

  This mask class inherits from splash_attention_mask._ComputableMask and is designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention kernel, avoiding the memory cost of
  materializing the full (sequence_length, sequence_length) boolean mask array, which can be prohibitive for long sequences.
  """

  #: The size of each attention chunk.
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
  bidirectional_block_mask = (kv_block_indices[:, None, :] == q_block_indices[..., None]) & (
      q_block_indices[..., None] > 0
  )
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
    self.rngs = rngs

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
          self.kv_quant.einsum_fn_with_rhs_qtensor(), "btkgd,bskd->bkgts", dummy_query_prefill, dummy_key_prefill
      )
      self.AqtEinsum_1 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_prefill,
          dummy_value_prefill,
      )
      # Autoregressive AqtEinsum instances
      self.AqtEinsum_2 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(), "btkgd,bskd->bkgts", dummy_query_ar, dummy_key_ar
      )
      self.AqtEinsum_3 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_ar,
          dummy_value_ar,
      )
    else:
      self.AqtEinsum_0 = jnp.einsum
      self.AqtEinsum_1 = jnp.einsum
      self.AqtEinsum_2 = jnp.einsum
      self.AqtEinsum_3 = jnp.einsum

  def _logical_to_mesh_axes(self, logical_name):
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=self.config.logical_axis_rules)

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, f"k (dim {key.ndim}), v (dim {value.ndim}) must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), f"{query.shape[:-3]=}, {key.shape[:-3]=}, {value.shape[:-3]=} batch dims must match."
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
      An `Array` representing the attention mask, with shape
       `[batch_size, 1, 1, q_sequence_length, kv_sequence_length]`.
      It is broadcastable to the shape
       `[batch_size, num_kv_heads, group_size=n_q // n_kv, q_sequence_length, kv_sequence_length]`.
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
      sliding_mask = (col_ids_sliding > (row_ids_sliding - self.sliding_window_size)) & (
          col_ids_sliding <= row_ids_sliding
      )
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

  def calculate_moba_gate_logic(self, q_item, k_item, q_pos_item):
    """Computes the block-level MoBA gating intermediates for one batch item.

    Args:
      q_item: Query tensor shaped `[q_len, n_q_heads, head_dim]`.
      k_item: Key tensor shaped `[kv_len, n_kv_heads, head_dim]`.
      q_pos_item: Absolute query positions shaped `[q_len]`, used to derive the
        chunk index for each query.
        For example, during prefill after 128 tokens
        have been processed `q_pos_item` is `jnp.arange(128, 128 + q_len)`,
        while in autoregressive decode with a single query token it is
        `jnp.array([kv_len - 1])`.

    Returns:
      `need_attend`, a boolean mask of shape `[n_kv_heads, g, q_len, num_block]`
      indicating which key blocks each query should attend to. The additional
      values in the returned tuple are debug intermediates used for logging and
      diagnostics when inspecting the gating behaviour.
    """
    q_len, n_q_heads, head_dim = q_item.shape
    kv_len, n_kv_heads, _ = k_item.shape
    g = n_q_heads // n_kv_heads

    q_item_f32 = q_item.astype(jnp.float32).reshape(q_len, n_kv_heads, g, head_dim)  # grouped-query attention (GQA)

    moba_chunk_size = self.config.moba_chunk_size
    moba_topk = self.config.moba_topk

    num_block = math.ceil(kv_len / moba_chunk_size)

    block_ids = jnp.arange(kv_len, dtype=jnp.int32) // moba_chunk_size  # chunk index for each key position
    # Sum key vectors per chunk so we can later average within each block.
    key_gate_weight_sum = jax.ops.segment_sum(
        k_item.astype(jnp.float32), block_ids, num_segments=num_block
    )  # [num_block, n_kv_heads, head_dim]
    # Count how many tokens end up in each chunk so we can take the mean.
    block_counts = jax.ops.segment_sum(
        jnp.ones((kv_len,), dtype=jnp.float32), block_ids, num_segments=num_block
    )  # [num_block]
    # Mean Pooling, Avoid division by zero for empty blocks.
    key_gate_weight = key_gate_weight_sum / jnp.maximum(
        block_counts[:, None, None], 1
    )  # [num_block, n_kv_heads, head_dim]

    # Take the dot product between each query and every key chunk to get a score.
    gate = jnp.einsum("skgd,Nkd->kgsN", q_item_f32, key_gate_weight)  # [n_kv_heads, g, q_len, num_block]
    gate_before_masking = gate

    q_block_idx = q_pos_item // moba_chunk_size  # chunk id for each query
    block_indices = jnp.arange(num_block)  # list every key chunk index

    q_block_idx_b = jnp.expand_dims(q_block_idx, axis=-1)  # [q_len, 1]
    block_indices_b = jnp.expand_dims(block_indices, axis=0)  # [1, num_block]

    # Block-causal masking: a query can't attend to future key blocks,
    # and must attend to its own key block.
    mask_future = q_block_idx_b > block_indices_b
    gate = jnp.where(mask_future, gate, -float("inf"))
    mask_diag = q_block_idx_b == block_indices_b
    gate = jnp.where(mask_diag, float("inf"), gate)
    gate_after_masking = gate

    k_for_topk = min(moba_topk, num_block)
    gate_top_k_val, gate_top_k_idx = jax.lax.top_k(gate, k=k_for_topk)  # [n_kv_heads, g, q_len, k_for_topk]
    gate_top_k_val_min = jnp.min(gate_top_k_val, axis=-1, keepdims=True)  # [n_kv_heads, g, q_len, 1]
    need_attend_threshold_mask = gate >= gate_top_k_val_min  # [n_kv_heads, g, q_len, num_block]

    # Tie-breaking: if multiple blocks have the same gate value as the k-th
    # block, we only select the ones that appear in the top-k indices.
    gate_idx_mask = jnp.sum(
        jax.nn.one_hot(gate_top_k_idx, num_block, dtype=jnp.bool_), axis=-2
    )  # [n_kv_heads, g, q_len, num_block]
    need_attend = jnp.logical_and(need_attend_threshold_mask, gate_idx_mask)  # [n_kv_heads, g, q_len, num_block]

    return (
        key_gate_weight,
        gate_before_masking,
        gate_after_masking,
        gate_top_k_val,
        gate_top_k_idx,
        gate_top_k_val_min,
        need_attend_threshold_mask,
        gate_idx_mask,
        need_attend,  # [n_kv_heads, g, q_len, num_block]
    )

  def generate_moba_mask_single_item(self, q_item, k_item, q_positions):
    """Generates the token-level MoBA additive mask for a single batch item."""
    q_len, _, _ = q_item.shape
    kv_len, _, _ = k_item.shape
    moba_chunk_size = self.config.moba_chunk_size

    # Run the gating logic to find which key blocks this query cares about.
    *_, need_attend = self.calculate_moba_gate_logic(q_item, k_item, q_positions)

    # Expand the block-level `need_attend` mask to a token-level mask.
    k_block_indices = jnp.arange(kv_len, dtype=jnp.int32) // moba_chunk_size
    token_level_need_attend = need_attend[..., k_block_indices]

    # Convert the boolean mask to float mask values.
    gate = jnp.where(token_level_need_attend, 0.0, -float("inf"))

    # Apply a final per-token causal mask to ensure causality within chunks.
    k_indices = jax.lax.broadcasted_iota(jnp.int32, (q_len, kv_len), 1)
    q_indices = q_positions[:, None]
    causal_mask = q_indices >= k_indices
    gate = jnp.where(causal_mask, gate, -float("inf"))

    # Return the additive mask for this batch item.
    return gate

  def _generate_moba_mask(self, query: Array, key: Array, q_positions: Array) -> Array:
    """Builds the token-level MoBA additive mask for the whole batch.

    Args:
      query: Query tensor shaped `[batch, q_len, n_q_heads, head_dim]`.
      key: Key tensor shaped `[batch, kv_len, n_kv_heads, head_dim]`.
      q_positions: Absolute query positions shaped `[q_len]`, shared across the
        batch, identifying the starting offset of each query token.
        For example, in prefill after 128 tokens we pass
        `jnp.arange(128, 128 + q_len)`, while in autoregressive decode with a
        single new token the vector is `[kv_len - 1]` for each batch element.

    Returns:
      Additive attention mask with shape
      `[batch, n_kv_heads, n_q_heads // n_kv_heads, q_len, kv_len]` containing
      `0.` for permitted positions and `-inf` for masked ones.
    """
    # vmap over the batch dimension of query and key. q_positions is constant across the batch.
    moba_mask = jax.vmap(self.generate_moba_mask_single_item, in_axes=(0, 0, None))(query, key, q_positions)
    return moba_mask

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
      sinks: Array | None = None,
      index_mask: Array | None = None,
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

    # 'vllm_rpa' uses the same dot-attention wrapper but routes to the vLLM
    # ragged paged attention kernel in `Attention.__call__`.
    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
        or (self.attention_kernel == "paged")
        or (self.attention_kernel == "vllm_rpa")
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
          index_mask=index_mask,
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
        return (
            self.tpu_flash_attention(
                query, key, value, decoder_segment_ids, self.attn_logits_soft_cap, sinks, index_mask
            ),
            None,
            None,
        )
      else:
        validate_flash_attention_with_sinks_on_gpu(sinks)
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
      validate_flash_attention_with_sinks_on_gpu(sinks)
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
      validate_flash_attention_with_sinks_on_gpu(sinks)
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
    b = self._logical_to_mesh_axes(self.ragged_lengths_names)
    bsnd = self._logical_to_mesh_axes(self.cache_logical_axis_names)
    bnd = self._logical_to_mesh_axes((CACHE_BATCH, CACHE_HEADS, CACHE_KV))
    bn = self._logical_to_mesh_axes((CACHE_BATCH, CACHE_HEADS))

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(bnd, bsnd, bsnd, b, None),
        out_specs=(bnd, bn, bn),
        check_vma=False,
    )
    def wrap_ragged_attention(
        q: Array, k: Array, v: Array, lengths: Array, block_size: int
    ) -> Tuple[Array, Array, Array]:
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
    b = self._logical_to_mesh_axes(self.ragged_lengths_names)
    bsnd = self._logical_to_mesh_axes(self.cache_logical_axis_names)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            bsnd,
            bsnd,
            bsnd,
            b,
            None,
        ),
        out_specs=bsnd,
        check_vma=False,
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
      sinks: Array | None = None,
      index_mask: Array | None = None,
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
    sink_axis_names = self._logical_to_mesh_axes((HEAD,))
    if decoder_segment_ids is not None:
      if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
        segment_axis_names_q = self._logical_to_mesh_axes((BATCH_NO_EXP, Q_LENGTH))
        segment_axis_names_kv = self._logical_to_mesh_axes((BATCH_NO_EXP, KV_LENGTH))
      else:
        segment_axis_names_q = self._logical_to_mesh_axes((BATCH, Q_LENGTH_NO_EXP))
        segment_axis_names_kv = self._logical_to_mesh_axes((BATCH, KV_LENGTH))

    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      axis_names_splash_kernel = self._logical_to_mesh_axes(self.flash_axis_names_splash_kernel_ep)
      axis_names_q = self._logical_to_mesh_axes(self.flash_axis_names_q_ep)
      axis_names_kv = self._logical_to_mesh_axes(self.flash_axis_names_kv_ep)
      index_mask_axis_names = self._logical_to_mesh_axes((BATCH_NO_EXP, Q_LENGTH, KV_LENGTH))
    else:
      axis_names_splash_kernel = self._logical_to_mesh_axes(self.flash_axis_names_splash_kernel)
      axis_names_q = self._logical_to_mesh_axes(self.flash_axis_names_q)
      axis_names_kv = self._logical_to_mesh_axes(self.flash_axis_names_kv)
      index_mask_axis_names = self._logical_to_mesh_axes((BATCH, Q_LENGTH, KV_LENGTH))

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

    # create_splash_attention config
    def create_sa_config(config, query, key, attn_logits_soft_cap):
      if config.use_tokamax_splash:
        sa_config = tokamax_splash_kernel.SplashConfig(
            block_q=min(global_block_q, query.shape[2]),
            block_kv=min(global_block_kv, key.shape[2]),
            block_kv_compute=min(global_block_kv_compute, key.shape[2]),
            block_q_dkv=min(global_block_q_dkv, query.shape[2]),
            block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
            block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
            use_fused_bwd_kernel=True,  # tokamax only supports fused bwd kernel
            q_layout=tokamax_splash_kernel.QKVLayout[global_q_layout],
            k_layout=tokamax_splash_kernel.QKVLayout[global_k_layout],
            v_layout=tokamax_splash_kernel.QKVLayout[global_v_layout],
            attn_logits_soft_cap=attn_logits_soft_cap,
            residual_checkpoint_name="context",
            fwd_cost_estimate=pl.CostEstimate(
                flops=config.cost_estimate_flops_fwd,
                transcendentals=0,
                bytes_accessed=0,
            )
            if config.cost_estimate_flops_fwd >= 0
            else None,
            bwd_cost_estimate=pl.CostEstimate(
                flops=config.cost_estimate_flops_bwd,
                transcendentals=0,
                bytes_accessed=0,
            )
            if config.cost_estimate_flops_bwd >= 0
            else None,
            dq_reduction_steps=config.dq_reduction_steps if config.dq_reduction_steps > 0 else None,
            use_experimental_scheduler=config.use_splash_scheduler,
        )
      else:
        sa_config = splash_attention_kernel.BlockSizes(
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
      return sa_config

    sa_config = create_sa_config(self.config, query, key, attn_logits_soft_cap)
    mask_shape = (query.shape[2], key.shape[2])  # (q_seq_len, kv_seq_len)
    mask_module = tokamax_splash_mask if self.config.use_tokamax_splash else splash_attention_mask
    if self.attention_type == AttentionType.FULL:
      mask = mask_module.FullMask(mask_shape)
    else:
      mask = mask_module.CausalMask(shape=mask_shape)

    # Create LoadBalancedCausalMask if cp and load_balancing
    if cp_size > 1 and load_balanced_context_parallel:
      mask = LoadBalancedCausalMask(shape=mask_shape, cp_size=cp_size)

    # TODO: figure out local_sliding attention + load_balancing, default is global
    # Apply local masking if local sliding attention is enabled.
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
      mask &= mask_module.LocalMask(
          shape=(query.shape[2], key.shape[2]),
          window_size=(self.sliding_window_size, self.sliding_window_size),
          offset=0,
      )
    elif self.attention_type == AttentionType.CHUNK:
      if self.chunk_attn_window_size is None:
        raise ValueError("chunk_attn_window_size must be set for chunk attention type")

      mask &= ChunkedCausalMask(shape=(query.shape[2], key.shape[2]), chunk_size=self.chunk_attn_window_size)

    max_logit_value = None
    if self.config.use_tokamax_splash:
      # Create mask
      single_head_mask = mask  # tokamax now just uses a single mask and assumes broadcast to all heads
      if self.config.use_max_logit_estimate > 0:
        sa_config = dataclasses.replace(sa_config, max_logit_const=self.config.use_max_logit_estimate)

      # Create the splash attention kernel object separately, jit it for performance
      @partial(
          jax.jit,
          static_argnames=[
              "single_head_mask",
              "shard_head_size",
          ],
      )
      def wrap_splash_kernel(single_head_mask, shard_head_size=1):
        splash_kernel = tokamax_splash_kernel.make_splash_mha(
            mask=single_head_mask,
            config=sa_config,
            q_seq_shards=cp_size,  # axis for sequence sharding,
        )
        return splash_kernel

      logical_axis_rules_head = np.array(
          [self.mesh.shape[physical_axes] for physical_axes in dict(self.config.logical_axis_rules)[HEAD]]
      )
      shard_head_size = np.prod(logical_axis_rules_head)
      splash_kernel = wrap_splash_kernel(single_head_mask, int(shard_head_size))
      if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
        segment_axis_names_splash_kernel = self._logical_to_mesh_axes((Q_LENGTH,))
      else:
        segment_axis_names_splash_kernel = self._logical_to_mesh_axes((Q_LENGTH_NO_EXP,))
    elif self.config.use_jax_splash and self.config.expert_shard_attention_option == EP_AS_FSDP:
      if self.config.use_max_logit_estimate > 0:
        sa_config = dataclasses.replace(sa_config, max_logit_const=self.config.use_max_logit_estimate)
      segment_axis_names_splash_kernel = nn.logical_to_mesh_axes((Q_LENGTH_NO_EXP,))
    else:
      # Create multi-head mask
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

      # Create the splash attention kernel object separately, jit it for performance
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
            block_sizes=sa_config,
            attn_logits_soft_cap=attn_logits_soft_cap,
            residual_checkpoint_name="context",
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
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names_q,
            axis_names_kv,
            axis_names_kv,
            segment_axis_names_q,
            segment_axis_names_kv,
            None,  # no sharding for config
            segment_axis_names_splash_kernel,
            None,  # no sharding for cp_size
            None,  # no sharding for load_balanced_context_parallel
            sink_axis_names,  # sharding align with query heads
            index_mask_axis_names,
        ),
        out_specs=axis_names_q,
        check_vma=False,
    )
    def wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        sa_config,
        splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
        index_mask,
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

      if self.config.use_tokamax_splash:
        if self.config.use_sparse_indexer and index_mask is not None:
          # Construct the splash kernel call with dynamic mask
          def dynamic_mask_splash_kernel(q, k, v, segment, sinks, index_mask):
            splash_kernel = tokamax_splash_kernel.make_dynamic_splash_mha(
                mask=index_mask,
                config=sa_config,
            )
            kernel = partial(splash_kernel, max_logit_value=max_logit_value)
            return kernel(q, k, v, segment, sinks=sinks)

          # Iterate over batch dimension for (query, key, value, segment, sinks, mask)
          attn_fn = jax.vmap(dynamic_mask_splash_kernel, (0, 0, 0, 0, None, 0))
          index_mask = jnp.isclose(index_mask, 0.0)
          attention_output = attn_fn(query, key, value, decoder_segment_ids_tuple, sinks, index_mask)
        else:
          kernel = partial(splash_kernel, max_logit_value=max_logit_value)
          attention_output = jax.vmap(lambda q, k, v, d, s: kernel(q, k, v, d, sinks=s), in_axes=(0, 0, 0, 0, None))(
              query, key, value, decoder_segment_ids_tuple, sinks
          )
      elif self.config.use_jax_splash:
        materialized_mask = jnp.asarray(mask[:, :])
        attention_output = jax_flash_attention.flash_attention_block_masked(
            query,
            key,
            value,
            decoder_segment_ids_tuple,
            block_kv=self.config.sa_block_kv,
            block_q=self.config.sa_block_q,
            mask=materialized_mask,
            mask_value=DEFAULT_MASK_VALUE,
        )
      else:
        attention_output = jax.vmap(splash_kernel, in_axes=(0, 0, 0, 0, None))(
            query, key, value, decoder_segment_ids_tuple, sinks
        )
      return attention_output

    def _maybe_shard_with_pspec(inputs, pspec: jax.sharding.PartitionSpec | None):
      # decoder_segment_ids can be None
      if pspec is None:
        return None
      sharding = NamedSharding(self.mesh, pspec)
      return maybe_shard_with_name(
          inputs,
          sharding,
          shard_mode=self.config.shard_mode,
          debug_sharding=self.config.debug_sharding,
          extra_stack_level=1,
      )

    query = _maybe_shard_with_pspec(query, axis_names_q)
    key = _maybe_shard_with_pspec(key, axis_names_kv)
    value = _maybe_shard_with_pspec(value, axis_names_kv)
    decoder_segment_ids_q = _maybe_shard_with_pspec(decoder_segment_ids, segment_axis_names_q)
    decoder_segment_ids_kv = _maybe_shard_with_pspec(decoder_segment_ids, segment_axis_names_kv)
    sinks = _maybe_shard_with_pspec(sinks, sink_axis_names)
    index_mask = _maybe_shard_with_pspec(index_mask, index_mask_axis_names)

    x = wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        sa_config,
        None if self.config.use_jax_splash else splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
        index_mask,
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
    1. Stable API, supports MHA, GQA, SWA, Packing and Context Parallelism
    2. Context Parallelism currently only supports causal masking and no packing
    """
    # These imports are only meant to work in a GPU build.
    # pylint: disable=import-outside-toplevel
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error
    from transformer_engine.jax.attention import SequenceDescriptor  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    using_context_parallelism = self.mesh.shape["context"] > 1

    # Initialize default attention configuration
    sliding_window_size = None
    mask_type = "padding_causal"
    qkv_layout = "BSHD_BSHD_BSHD"  # Non-packed format: 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
    max_segments_per_seq = 1  # max number of segments per sequence; for non-packed its 1

    # Handle local sliding window attention if configured
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      sliding_window_size = [self.sliding_window_size, 0]

    # Handle packing configurations
    if self.config.packing and self.config.dataset_type != "synthetic":
      qkv_layout = "THD_THD_THD"  # Packed format: 'T3HD', 'THD_T2HD' or 'THD_THD_THD'
      if decoder_segment_ids is None:
        decoder_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
      attn_mask = SequenceDescriptor.from_segment_ids_and_pos(segment_ids=decoder_segment_ids, segment_pos=None)
      # Create dummy SequenceDescriptor for lazy_init
      dummy_segment_ids = jnp.ones(shape=query.shape[:2], dtype=jnp.int32)
      dummy_attn_mask = SequenceDescriptor.from_segment_ids_and_pos(segment_ids=dummy_segment_ids, segment_pos=None)
      max_segments_per_seq = self.config.max_segments_per_seq
    elif using_context_parallelism:
      if self.attention_type == AttentionType.LOCAL_SLIDING:
        raise AssertionError("Sliding window attention is not supported for context parallelism")
      # Context parallelism without packing: only supports causal masking
      attn_mask = None
      dummy_attn_mask = None
      mask_type = "causal"
    else:
      # Default case: no packing, no context parallelism
      dummy_attn_mask = jnp.zeros((1, 1, 1, self.max_target_length, self.max_target_length), dtype=jnp.uint8)
      attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
      attn_mask = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), 0, 1).astype(jnp.uint8)

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
        qkv_layout=qkv_layout,
        scale_factor=1.0,
        transpose_batch_sequence=False,
        window_size=sliding_window_size,
        context_parallel_causal_load_balanced=self.config.context_parallel_load_balance,
        context_parallel_axis="context",
        context_parallel_strategy=self.config.context_parallel_strategy,
        max_segments_per_seq=max_segments_per_seq,
    )

    dpa_layer = nnx_wrappers.ToNNX(dpa_layer, rngs=self.rngs)
    dummy_query_prefill = jnp.zeros(
        (1, self.max_target_length, self.num_query_heads, self.config.head_dim), dtype=self.dtype
    )
    dummy_key_prefill = jnp.zeros((1, self.max_target_length, self.num_kv_heads, self.config.head_dim), dtype=self.dtype)
    dummy_value_prefill = jnp.zeros(
        (1, self.max_target_length, self.num_kv_heads, self.config.head_dim), dtype=self.dtype
    )

    dpa_layer.lazy_init(dummy_query_prefill, dummy_key_prefill, dummy_value_prefill, sequence_descriptor=dummy_attn_mask)
    return dpa_layer(query, key, value, sequence_descriptor=attn_mask)

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
      sinks: Array | None = None,
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
    b, n_kv, g, t, s = attn_weights.shape
    n_q = n_kv * g
    logits = jnp.reshape(attn_weights, (b, n_q, t, s))
    if sinks is not None:
      # broadcast sinks to match the attn weights dimension and combine
      sinks_param = sinks.astype(attn_weights.dtype)  # (n_q,)
      sinks_logits = sinks_param[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]  # (1, n_q, 1, 1)
      sinks_logits = jnp.broadcast_to(sinks_logits, (b, n_q, t, 1))
      logits = jnp.concatenate([logits, sinks_logits], axis=-1)

    # softmax
    local_max = jnp.max(logits, axis=-1, keepdims=True)
    local_exps_combined = jnp.exp(logits - local_max)
    local_sum = jnp.sum(local_exps_combined, axis=-1, keepdims=True)

    # reshape and transpose
    local_exps = local_exps_combined[..., :s]
    local_exps = jnp.reshape(local_exps, (b, n_kv, g, t, s))
    local_max = jnp.transpose(local_max, (0, 2, 1, 3))  # (b, t, n_q, 1)
    local_sum = jnp.transpose(local_sum, (0, 2, 1, 3))  # (b, t, n_q, 1)

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
      sinks: Array | None = None,
      index_mask: Array | None = None,
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

    attn_mask = self.generate_attention_mask(
        query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask
    )

    if self.config.moba:
      kv_seq_len = key.shape[1]
      # This logic for `next_pos` is duplicated from `generate_attention_mask`.
      # It determines the starting position of the query sequence.
      next_pos = 0
      if previous_chunk is not None:
        next_pos = previous_chunk.shape[1]
      elif model_mode == MODEL_MODE_AUTOREGRESSIVE and q_seq_len == 1:
        next_pos = kv_seq_len - 1
      q_positions = jnp.arange(next_pos, next_pos + q_seq_len)

      # The gate calculation in MoBA uses the unscaled query.
      # With scaled query, the gate values are scaled, but since the top-k selection
      # is scale-invariant, we can use the scaled query directly.
      moba_mask = self._generate_moba_mask(query, key, q_positions)
      attn_weights += moba_mask

    # Apply index mask, deepseek sparse attention
    # index mask contains 0.0 for kept tokens and large negative for masked tokens.
    if index_mask is not None:
      # index_mask: from [b, q_len, kv_len] to [b, 1, 1, q_len, kv_len]
      index_mask = index_mask[:, None, None, :, :]
      # attn_weights: [b, n_kv, n_q // n_kv, q_len, kv_len]
      attn_weights = apply_mask_to_logits(attn_weights, index_mask)

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
    precision_kwargs = {"precision": self.config.matmul_precision} if einsum is jnp.einsum else {}
    if model_mode == MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, 2, n_kv, n // n_kv, d))
      result = einsum("btkgd,bskd->bkgts", query, key, **precision_kwargs)
    elif self.compute_axis_order == (0, 2, 1, 3):
      query = jnp.transpose(query, axes=self.compute_axis_order)
      key = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), key)
      query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, n_kv, n // n_kv, 2, d))
      result = einsum("bkgtd,bksd->bkgts", query, key, **precision_kwargs)
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

    precision_kwargs = {"precision": self.config.matmul_precision} if einsum is jnp.einsum else {}
    if self.kv_quant:
      # manually cast to bf16 to avoid the fp32 XLA ops for speedup
      if isinstance(value, KVTensor) and self.kv_quant.dtype == jnp.float8_e4m3fn:
        value.qvalue = value.qvalue.astype(jnp.bfloat16)
    if model_mode == MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      out = einsum("bkgts,bskd->btkgd", attn_weights, value, **precision_kwargs)
      b, t, n_kv, g, d = out.shape
      result = jnp.reshape(out, (b, t, n_kv * g, d))
    elif self.compute_axis_order == (0, 2, 1, 3):
      value = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), value)
      out = einsum("bkgts,bksd->bkgtd", attn_weights, value, **precision_kwargs)
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
      index_mask: Optional[Array] = None,
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
        index_mask=index_mask,
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
