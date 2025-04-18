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

import enum
import functools
import math
import numpy as np
from typing import Any, Optional, Tuple

from MaxText import common_types
from flax import linen as nn
from flax.linen import partitioning
from MaxText.inference import kvcache
from MaxText.inference import page_manager
from MaxText.inference import paged_attention
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops.gpu import attention as gpu_pallas_attention
from jax.experimental.pallas.ops.gpu import decode_attention as gpu_pallas_decode_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
from MaxText.kernels.ragged_attention import ragged_gqa
from MaxText.kernels.ragged_attention import ragged_mha
from MaxText.layers import embeddings
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import quantizations


# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


class AttentionType(enum.Enum):
  GLOBAL = "global"
  LOCAL_SLIDING = "local_sliding"
  CHUNK = "chunk"
  MLA = "mla"


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey
DenseGeneral = linears.DenseGeneral
RMSNorm = linears.RMSNorm
RotaryEmbedding = embeddings.RotaryEmbedding
YarnRotaryEmbedding = embeddings.YarnRotaryEmbedding
NdInitializer = initializers.NdInitializer
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant
KVTensor = quantizations.KVTensor
AxisNames = common_types.AxisNames
AxisIdxes = common_types.AxisIdxes
BATCH = common_types.BATCH
PREFILL_KV_BATCH = common_types.PREFILL_KV_BATCH
KV_BATCH = common_types.KV_BATCH
DECODE_BATCH = common_types.DECODE_BATCH
DECODE_LENGTH = common_types.DECODE_LENGTH
LENGTH = common_types.LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
EMBED = common_types.EMBED
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM
CACHE_BATCH_PREFILL = common_types.CACHE_BATCH_PREFILL
CACHE_BATCH = common_types.CACHE_BATCH
CACHE_SEQUENCE = common_types.CACHE_SEQUENCE
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
CACHE_SCALE_BATCH = common_types.CACHE_SCALE_BATCH
CACHE_SCALE_SEQUENCE = common_types.CACHE_SCALE_SEQUENCE
CACHE_SCALE_HEADS = common_types.CACHE_SCALE_HEADS
CACHE_SCALE_KV = common_types.CACHE_SCALE_KV
DEFAULT_MASK_VALUE = common_types.DEFAULT_MASK_VALUE

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

nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

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
class ChunkedCausalMask(splash_attention_mask._ComputableMask):
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens attend to each other but not accross chunks.
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


def _generate_chunk_attention_mask(mask_shape: tuple[int, int], chunk_size: int) -> jax.Array:
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

  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  # chunk mask calculation
  same_chunk = (row_ids // chunk_size) == (col_ids // chunk_size)
  chunk_mask = same_chunk & (row_ids >= col_ids)
  return chunk_mask


class AttentionOp(nn.Module):
  config: Config
  mesh: Mesh
  attention_kernel: str
  max_target_length: int
  num_query_heads: int
  num_kv_heads: int
  float32_qk_product: bool = False
  max_prefill_predict_length: int = -1
  float32_logits: bool = False
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV)
  ragged_lengths_names: AxisNames = (CACHE_BATCH,)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  key_axis_order: AxisIdxes = (2, 0, 1, 3)

  reshape_q: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None
  attention_type: AttentionType = AttentionType.GLOBAL  # Default to global attention
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  chunk_attn_window_size: int | None = None
  use_ragged_attention: bool = False
  ragged_block_size: int = 256

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, f"k (dim {key.ndim}), v (dim {value.ndim}) must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
  # This mask models (1) separate sequences (decoder_segment_ids) and (2) causality
  def generate_attention_mask(
      self, query, key, decoder_segment_ids: Array | None, model_mode: str, previous_chunk: Any = None
  ) -> Array | None:
    mask = None
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
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

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != common_types.MODEL_MODE_AUTOREGRESSIVE:
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

      all_ones = jnp.ones_like(output_mask)
      sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(all_ones, self.sliding_window_size - 1)
      output_mask = sliding_mask * output_mask
    elif self.attention_type == AttentionType.CHUNK and output_mask is not None:
      mask_shape = (q_seq_len, kv_seq_len)
      chunk_mask = _generate_chunk_attention_mask(mask_shape=(q_seq_len, kv_seq_len), chunk_size=self.chunk_attn_window_size)
      output_mask = chunk_mask * output_mask

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
  ):
    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    if use_ragged_attention and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      if lengths is None:
        lengths = jnp.sum(decoder_segment_ids, axis=-1)

      if jax.devices()[0].platform == "tpu":
        impl = self.tpu_ragged_attention
      elif jax.devices()[0].platform == "gpu":
        impl = self.gpu_ragged_attention
      return impl(query, key, value, lengths, self.ragged_block_size)

    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
        or (self.attention_kernel == "paged")
    ):
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode, previous_chunk)
    elif self.attention_kernel == "flash" or self.attention_kernel == "autoselected":
      if jax.devices()[0].platform == "tpu":
        if isinstance(key, KVTensor):
          key = key.dequant()
        if isinstance(value, KVTensor):
          value = value.dequant()

        if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
          raise ValueError(
              """Decode not supported with flash attention.
                              Use `dot_product` instead."""
          )
        return self.tpu_flash_attention(query, key, value, decoder_segment_ids, self.attn_logits_soft_cap), None, None
      else:
        if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
          # fallback to dot_product as pallas gpu flash attention doesn't support decode stage
          return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
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
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                           Use `dot_product` instead."""
        )
      return self.cudnn_flash_attention(query, key, value, decoder_segment_ids, model_mode), None, None
    else:
      raise ValueError(f"Unexpected attention kernel {self.attention_kernel=}.")

  def gpu_ragged_attention(self, q: Array, k: Array | KVTensor, v: Array | KVTensor, lengths: Array, block_size: int):
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
    if isinstance(query, KVTensor) or isinstance(query, KVTensor):
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
  ) -> Array:
    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))

    if decoder_segment_ids is not None:
      decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
    axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    segment_axis_names = nn.logical_to_mesh_axes((BATCH, "activation_length_no_heads"))

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

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names,
            axis_names,
            axis_names,
            segment_axis_names,
        ),
        out_specs=axis_names,
        check_rep=False,
    )
    def wrap_flash_attention(query, key, value, decoder_segment_ids):
      if decoder_segment_ids is not None:
        assert (
            query.shape[2] == decoder_segment_ids.q.shape[1]
        ), "Sharding along sequence dimension not allowed in tpu kernel attention"
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

      mask = splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2]))

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
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=block_sizes,
          attn_logits_soft_cap=attn_logits_soft_cap,
      )

      return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

    devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        "Batch dimension should be shardable among the devices in data and fsdp"
        " axis"
        f" got {query.shape[0]=}/{devices_in_data_fsdp=}"
    )
    x = wrap_flash_attention(query, key, value, decoder_segment_ids)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x

  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports GQA, SWA (only with causal masking)
    2. Head_dim = 256 is also supported from TE-1.12 stable release with CUDNN 12.6
    """
    # These imports are only meant to work in a GPU build.
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

  def compute_local_attention(
      self, attn_weights: Array, value: Array | KVTensor, q_seq_len: int, model_mode: str
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
    local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
    local_exps = jnp.exp(attn_weights - local_max)
    local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

    local_sum = jnp.moveaxis(local_sum, -2, 1)
    local_max = jnp.moveaxis(local_max, -2, 1)

    local_max = jnp.reshape(local_max, (local_max.shape[0], local_max.shape[1], local_max.shape[2] * local_max.shape[3], 1))
    local_sum = jnp.reshape(local_sum, (local_sum.shape[0], local_sum.shape[1], local_sum.shape[2] * local_sum.shape[3], 1))

    local_out = self.wv_product(local_exps, value, model_mode)
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE and self.is_partition_in_decode(q_seq_len):
      local_out = partitioning.with_sharding_constraint(local_out, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      local_out = partitioning.with_sharding_constraint(local_out, (BATCH, KV_LENGTH, HEAD, D_KV))

    if self.reshape_q and q_seq_len == 1:
      local_max = local_max[:, 0:1, :, :]
      local_sum = local_sum[:, 0:1, :, :]
      local_out = local_out[:, 0:1, :, :]

    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE and self.is_partition_in_decode(q_seq_len):
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
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
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
    prefill_qkv_sharding = (BATCH, LENGTH, HEAD, D_KV)
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
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      query = partitioning.with_sharding_constraint(query, prefill_qkv_sharding)
      # avoid sharding scale tensor when using kv cache quantization
      if self.kv_quant and isinstance(key, KVTensor) and isinstance(value, KVTensor):
        key.qvalue = partitioning.with_sharding_constraint(key.qvalue, prefill_qkv_sharding)
        value.qvalue = partitioning.with_sharding_constraint(value.qvalue, prefill_qkv_sharding)
      else:
        key = partitioning.with_sharding_constraint(key, prefill_qkv_sharding)
        value = partitioning.with_sharding_constraint(value, prefill_qkv_sharding)

    attn_weights = self.qk_product(query, key, q_seq_len, model_mode)
    if self.is_partition_in_decode(q_seq_len):
      attn_weights = partitioning.with_sharding_constraint(attn_weights, (KV_LENGTH, HEAD, None, None, None))
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      attn_weights = partitioning.with_sharding_constraint(attn_weights, (BATCH, HEAD, None, LENGTH, KV_LENGTH))

    if self.attn_logits_soft_cap:
      attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
      attn_weights = attn_weights * self.attn_logits_soft_cap

    # Casting softmaxt computation for float32 for model stability.
    if self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode, previous_chunk)
    if self.is_partition_in_decode(q_seq_len):
      attn_mask = partitioning.with_sharding_constraint(attn_mask, (KV_LENGTH, HEAD, None, None, None))
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      attn_mask = partitioning.with_sharding_constraint(attn_mask, (BATCH, HEAD, None, LENGTH, KV_LENGTH))
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    return self.compute_local_attention(attn_weights, value, q_seq_len, model_mode)

  def qk_product(self, query: Array, key: Array | KVTensor, q_seq_len: int, model_mode: str) -> Array:
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
    einsum = jnp.einsum
    if self.kv_quant:
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor(key)
    b, t, n, d = query.shape
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
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
    return result

  def wv_product(self, attn_weights: Array, value: Array | KVTensor, model_mode: str) -> Array:
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

    einsum = jnp.einsum
    if self.kv_quant:
      # manually cast to bf16 to avoid the fp32 XLA ops for speedup
      if isinstance(value, KVTensor) and self.kv_quant.dtype == jnp.float8_e4m3fn:
        value.qvalue = value.qvalue.astype(jnp.bfloat16)
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(value)
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
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
        [jnp.exp(local_max - global_max) * local_sum for (local_sum, local_max) in zip(local_sums, local_maxes)]
    )

    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
      local_normalizer = jnp.exp(local_max - global_max) / global_sum
      attn_out += local_normalizer * local_out
    return attn_out

  @nn.compact
  def __call__(
      self,
      query,
      key,
      value,
      decoder_segment_ids,
      model_mode,
      cached_values=[None, None],
      previous_chunk=None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):

    prefill_kv_cache = cached_values[0]
    ar_kv_cache = cached_values[1]
    if model_mode != common_types.MODEL_MODE_TRAIN:
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
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum


class L2Norm(nn.Module):
  """
  Implementation of L2Norm in JAX.

  Attributes:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """

  eps: float = 1e-6

  @nn.compact
  def __call__(self, x):
    return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


class Attention(nn.Module):
  """Generic Attention.

  Attributes:
    num_query_heads: number of query attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    num_kv_heads: number of kv attention heads.
    head_dim: dimension of each head.
    mesh: Mesh, device mesh
    attention_kernel: str, guidance on if we should use an attention kernel
    dtype: the dtype of the computation.
    weight_dtype: the dtype of the weights.
    max_target_length: maximum target length
    max_prefill_predict_length: size of the maximum prefill
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
      numerical issues with bfloat16.
    float32_logits: bool, if True then cast logits to float32 before softmax to avoid
      numerical issues with bfloat16.
    quant: Quant, stores quantization parameters, defaults to None implying no quantization.
    kv_quant: KVQuant, stores KV cache quantization parameters, defaults to None
    is_nope_layer: bool, whether to skip RoPE on this Attention layer
  """

  config: Config
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  max_target_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  max_prefill_predict_length: int = -1
  dropout_rate: float = 0.0
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = False  # cast logits in float32 for stability.
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None

  attention_type: AttentionType = AttentionType.GLOBAL  # Default to global attention
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_ragged_attention: bool = False
  ragged_block_size: int = 256
  use_qk_norm: bool = False
  query_pre_attn_scalar: float | None = None

  # Shard the query activation as the same as the key and value.
  # TODO: Find a better sharding axis name.
  # TODO: Further break down the Training and Inference axes for the q, k, v.
  prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  input_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  decode_input_axis_names: AxisNames = (DECODE_BATCH, DECODE_LENGTH, EMBED)
  key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)
  decode_out_axis_names = (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV)

  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False

  is_nope_layer: bool = False

  def setup(self):
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
    )
    # When paged attention is enabled, paged attention op is used for all model modes except TRAIN,
    # which uses default attention op.
    if self.config.attention == "paged":
      self.paged_attention_op = paged_attention.PagedAttentionOp(
          mesh=self.mesh,
          num_pages=self.config.pagedattn_num_pages,
          tokens_per_page=self.config.pagedattn_tokens_per_page,
          max_pages_per_slot=self.config.max_target_length // self.config.pagedattn_tokens_per_page,
          max_pages_per_prefill=self.config.max_prefill_predict_length // self.config.pagedattn_tokens_per_page,
          pages_per_compute_block=self.config.pagedattn_pages_per_compute_block,
          num_kv_heads=self.num_kv_heads,
          kv_head_dim_size=self.head_dim,
          dtype=self.dtype,
          attn_logits_soft_cap=self.attn_logits_soft_cap,
      )

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

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
    query_proj = DenseGeneral(
        features=(self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=query_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="query",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(inputs_q)
    return query_proj

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length,
        num_kv_heads, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
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

    kv_proj = DenseGeneral(
        features=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(inputs_kv)
    return kv_proj

  def qkv_projection(self, inputs: Array, proj_name: str):
    """Fused QKV projection"""

    qkv_proj = DenseGeneral(
        features=(3, self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "qkv", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_kernel_axis = (
        (None, None, None) if self.config.ici_context_autoregressive_parallelism > 1 else ("heads", "kv", "embed")
    )
    out_proj = DenseGeneral(
        features=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=out_kernel_axis,  # trade speed with memory
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj

  def apply_rotary_embedding(self, inputs: Array, inputs_positions: Array, name: str):
    """Applies rotary embeddings, handling different model types.

    Args:
      inputs: The input tensor to apply rotary embeddings to.
      inputs_positions: The positions of the inputs.
      name: A name for the embedding layer.

    Returns:
      The input tensor with rotary embeddings applied.
    """
    if self.config.attention_type == AttentionType.MLA.value:
      # For MLA attention RoPE is applied to only `self.qk_rope_head_dim` portion the heads.
      rope_embedding_dims = self.qk_rope_head_dim
    else:
      rope_embedding_dims = self.head_dim

    rope_type = self.config.rope_type.lower()
    if self.config.model_name.startswith("llama3.1") or rope_type.startswith("llama3.1"):
      rotary_embedding = embeddings.LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          name=name,
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
          name=name,
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
          name=name,
      )
    inputs = rotary_embedding(inputs, inputs_positions)
    return inputs

  def update_kv_caches(self, key, value, decoder_segment_ids, model_mode, previous_chunk):
    """Updates the KV caches for prefill and autoregressive modes."""
    prefill_kv_cache, ar_kv_cache = kvcache.KVCache(
        self.max_prefill_predict_length,
        self.max_target_length,
        self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        use_chunked_prefill=self.config.use_chunked_prefill,
    )(
        key,
        value,
        decoder_segment_ids,
        model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
    )
    return [prefill_kv_cache, ar_kv_cache]

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are three modes: training, prefill and autoregression. During training, the KV cache
    is ignored. During prefill, the cache is filled. During autoregression the cache is used.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      model_mode: corresponding to train, prefill and decode.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    if model_mode == common_types.MODEL_MODE_PREFILL or model_mode == common_types.MODEL_MODE_TRAIN:
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
      is_llama4_decoder_block = self.config.decoder_block == "llama4"
      # NOTE: llama 4 does L2 normalization after RoPE
      if self.use_qk_norm and not is_llama4_decoder_block:
        query = RMSNorm(
            dtype=self.config.dtype,
            weight_dtype=self.config.weight_dtype,
            name="query_norm",
            epsilon=self.config.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )(query)

        key = RMSNorm(
            dtype=self.config.dtype,
            weight_dtype=self.config.weight_dtype,
            name="key_norm",
            epsilon=self.config.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )(key)

    # NOTE: is_nope_layer should be used in attention mask and also used in attention tuning
    use_rope = not self.is_nope_layer
    use_qk_norm = self.use_qk_norm and use_rope

    if use_rope:
      query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
      key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")

    if use_qk_norm and is_llama4_decoder_block:
      l2_norm = L2Norm(self.config.normalization_layer_epsilon)
      query = l2_norm(query)
      key = l2_norm(key)

    # apply query_pre_attn_scalar if it's present.
    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = query * self.query_pre_attn_scalar

    if model_mode == common_types.MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      query = nn.with_logical_constraint(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = nn.with_logical_constraint(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = nn.with_logical_constraint(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    assert not self.config.quantize_kvcache or self.kv_quant

    if self.config.attention == "paged" and model_mode != common_types.MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      cached_values = [None, None]
      if model_mode != common_types.MODEL_MODE_TRAIN:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, cached_values, previous_chunk)

    if model_mode == common_types.MODEL_MODE_PREFILL or model_mode == common_types.MODEL_MODE_TRAIN:
      out = nn.with_logical_constraint(out, self.out_axis_names)
    else:
      out = nn.with_logical_constraint(out, self.decode_out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, "out_proj")
    return out


class MLA(Attention):
  """Multi-Head Latent Attention (MLA) layer."""

  q_lora_rank: int = 0
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  max_position_embeddings: int = 4096 * 4
  original_max_position_embeddings: int = 4096
  mscale: float = 1.0  # scaling factor for softmax
  rope_factor: float = 40.0  # rotary embedding factor

  @property
  def qk_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def setup(self):
    """Initialize MLA-specific parameters."""
    super().setup()

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
      self.query_proj = DenseGeneral(
          features=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="query",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
    else:
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
          features=self.q_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_lora"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wq_a",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
      self.q_norm = RMSNorm(
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          name="q_norm",
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
      )
      self.wq_b = DenseGeneral(
          features=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("q_lora", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wq_b",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        features=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_a",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )
    self.kv_norm = RMSNorm(
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name="kv_norm",
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    self.wkv_b = DenseGeneral(
        features=(self.num_query_heads, (self.qk_nope_head_dim + self.v_head_dim)),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_b",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * jnp.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

  def mla_query_projection(self, inputs_q: Array, inputs_positions: Array, model_mode) -> Array:
    """Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0."""
    if self.q_lora_rank == 0:
      q = self.query_proj(inputs_q)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q)  # [B, L, q_lora_rank]
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      q = self.wq_b(low_rank_q)  # [B, L, n_heads * qk_head_dim]

    # Split into non-positional and rotary parts.
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions, name="query_rope")
    # Query projection is scaled by self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale
    if model_mode == common_types.MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
    return query

  def mla_get_key_value(self, low_rank_main, key_rope, model_mode):
    kv_out = self.wkv_b(low_rank_main)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))

    key = jnp.concatenate([key_nope, key_rope], axis=-1)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    else:
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)
    return key, value

  def update_mla_kv_caches(self, low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk=None):
    """Updates the MlaKvCache in prefill and autoregressive modes."""
    prefill_mla_cache, ar_mla_cache = kvcache.MlaKVCache(
        self.max_prefill_predict_length,
        self.max_target_length,
        self.dtype,
        kv_quant=self.kv_quant,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        use_chunked_prefill=self.config.use_chunked_prefill,
    )(
        low_rank_main,
        key_rope,
        decoder_segment_ids,
        model_mode,
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
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions, name="key_rope")

    key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
    cached_values = [None, None]
    if model_mode != common_types.MODEL_MODE_TRAIN:
      if self.config.mla_naive_kvcache:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      else:
        cached_values = self.update_mla_kv_caches(low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk)

    return key, value, cached_values

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
  ) -> Array:
    """Forward pass for MLA, reusing `AttentionOp` for the actual attention.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      model_mode: "train", "prefill", or "autoregressive".
      deterministic: Disables dropout if set to True.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    query = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, cached_values)
    out = nn.with_logical_constraint(out, self.out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)
    return out
