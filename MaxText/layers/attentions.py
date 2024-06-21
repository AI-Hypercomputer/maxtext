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

import functools
import math
from typing import Any, Optional

from flax import linen as nn
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
import jax.numpy as jnp

import common_types
from layers import embeddings
from layers import initializers
from layers import linears
from layers import quantizations


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey

DenseGeneral = linears.DenseGeneral
RotaryEmbedding = embeddings.RotaryEmbedding
NdInitializer = initializers.NdInitializer
Quant = quantizations.AqtQuantization

AxisNames = common_types.AxisNames
AxisIdxes = common_types.AxisIdxes
BATCH = common_types.BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM
CACHE_BATCH = common_types.CACHE_BATCH
CACHE_SEQUENCE = common_types.CACHE_SEQUENCE
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
CACHE_SCALE_BATCH = common_types.CACHE_SCALE_BATCH
CACHE_SCALE_SEQUENCE = common_types.CACHE_SCALE_SEQUENCE
CACHE_SCALE_HEADS = common_types.CACHE_SCALE_HEADS
CACHE_SCALE_KV = common_types.CACHE_SCALE_KV
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


def validate_compute_axis_order(s: AxisIdxes) -> None:
  valid_compute_axis_order = ((0,1,2,3), (0,2,1,3))
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


def _maybe_aqt_einsum(quant: Quant):
  """Maybe overwrite dot general with aqt_dot_general."""
  return jnp.einsum if quant is None else quant.einsum()


class AttentionOp(nn.Module):
  mesh: Mesh
  attention_kernel: str
  max_target_length: int
  num_query_heads: int
  num_kv_heads: int
  float32_qk_product: bool = False
  max_prefill_predict_length: int = -1
  float32_logits: bool = False
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV)
  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None
  quantize_kvcache: bool = False
  kv_quant_axis: str = "heads_and_dkv"

  def check_attention_inputs(self, query: Array, key: Array, value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, "k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
  # This mask models (1) separate sequences (decoder_segment_ids) and (2) causality
  def generate_attention_mask(self, query, key, decoder_segment_ids: Array | None, model_mode: str) -> Array | None:
    mask = None
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, None, :, :]

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != common_types.MODEL_MODE_AUTOREGRESSIVE:
      _, q_seq_len, _, _ = query.shape
      _, kv_seq_len, _, _ = key.shape
      mask_shape = (q_seq_len, kv_seq_len)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      causal_mask = (col_ids <= row_ids)[None, None, None, :, :]

    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask
    else:
      output_mask = None

    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None

  def apply_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None, model_mode: str):
    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    if (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
    ):
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
    elif self.attention_kernel == "flash" or self.attention_kernel == "autoselected":
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                            Use `dot_product` instead."""
        )
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids), None, None
    elif self.attention_kernel == "cudnn_flash_te":
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                           Use `dot_product` instead."""
        )
      return self.cudnn_flash_attention(query, key, value, decoder_segment_ids, model_mode), None, None
    else:
      raise ValueError(f"Unexpected attention kernel {self.attention_kernel=}.")

  def tpu_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None) -> Array:
    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))

    if decoder_segment_ids is not None:
      decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
    axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    segment_axis_names = nn.logical_to_mesh_axes((BATCH, "activation_length_no_heads"))

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
          block_q=min(512, query.shape[2]),
          block_kv_compute=min(512, key.shape[2]),
          block_kv=min(512, key.shape[2]),
          block_q_dkv=min(512, query.shape[2]),
          block_kv_dkv=min(512, key.shape[2]),
          block_kv_dkv_compute=min(512, query.shape[2]),
          block_q_dq=min(512, query.shape[2]),
          block_kv_dq=min(512, query.shape[2]),
      )

      masks = [splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2])) for i in range(query.shape[1])]
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=masks)
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes
      )

      return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

    devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        "Batch dimension should be shardable among the devices in data and fsdp" " axis"
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
    1. Stable API, supports GQA
    2. Supports head_dim till 128; head_dim=256 support will be added soon
    """
    # These imports are only meant to work in a GPU build.
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    # generate attn_mask
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)

    dpa_layer = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=self.num_query_heads,
        num_gqa_groups=self.num_kv_heads,
        attn_mask_type="causal",  # 'causal' or 'padding'
        attn_bias_type="NO_BIAS",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
        attention_dropout=self.dropout_rate,
        dropout_rng_name="aqt",
        dtype=self.dtype,
        float32_logits=self.float32_logits,
        qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        scale_factor=1.0 / math.sqrt(head_dim),
        transpose_batch_sequence=False,
    )
    return dpa_layer(query, key, value, mask=attn_mask)

  def compute_local_attention(self, attn_weights: Array, value: Array, q_seq_len: int, model_mode: str) -> tuple[Array, Array, Array]:
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

    if self.reshape_q and q_seq_len == 1:
      local_max = local_max[:,0:1,:,:]
      local_sum = local_sum[:,0:1,:,:]
      local_out = local_out[:,0:1,:,:]

    return local_out, local_max, local_sum

  def apply_attention_dot(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ):
    """Apply Attention."""
    validate_compute_axis_order(self.compute_axis_order)
    # Casting qk_product and softmaxt computation for float32 for model stability.
    if model_mode == common_types.MODEL_MODE_TRAIN and self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    q_seq_len = query.shape[1]
    attn_weights = self.qk_product(query, key, q_seq_len, model_mode)

    # Casting softmaxt computation for float32 for model stability.
    if model_mode == common_types.MODEL_MODE_TRAIN and self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    return self.compute_local_attention(attn_weights, value, q_seq_len, model_mode)

  def qk_product(self, query: Array, key: Array, q_seq_len: int, model_mode: str) -> Array:
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
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0,1,2,3):
      query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, 2, n_kv, n // n_kv, d))
      result = jnp.einsum("btkgd,bskd->bkgts", query, key)
    elif self.compute_axis_order == (0,2,1,3):
      query = jnp.transpose(query, axes=self.compute_axis_order)
      key = jnp.transpose(key, axes=self.compute_axis_order)
      query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, n_kv, n // n_kv, 2, d))
      result = jnp.einsum("bkgtd,bksd->bkgts", query, key)
    return result

  def wv_product(self, attn_weights: Array, value: Array, model_mode: str) -> Array:
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
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0,1,2,3):
      out = jnp.einsum("bkgts,bskd->btkgd", attn_weights, value)
      b, t, n_kv, g, d = out.shape
      result = jnp.reshape(out, (b, t, n_kv * g, d))
    elif self.compute_axis_order == (0,2,1,3):
      value = jnp.transpose(value, axes=self.compute_axis_order)
      out = jnp.einsum("bkgts,bksd->bkgtd", attn_weights, value)
      b, n_kv, g, t, d = out.shape
      result = jnp.reshape(out, (b, n_kv * g, t, d))
      result = self.reverse_transepose(result, self.compute_axis_order)
    return result

  def reverse_transepose(self, transposed_array, transpose_axis_order):
    return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)

  def transpose_tuple(self, items: tuple[Any, Any, Any, Any], axis_order: AxisIdxes) -> tuple[Any, Any, Any, Any]:
    return tuple([items[i] for i in axis_order])

  def _get_prefill_cache_vars(self, batch, heads, kv_head_size):

    dtype = jnp.int8 if self.quantize_kvcache else self.dtype
    cache_logical_shape = (batch, self.max_prefill_predict_length, heads, kv_head_size)

    cache_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.prefill_cache_axis_order)
    cache_shape = self.transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cached_key_var = self.variable(
        "cache",
        "cached_prefill_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_value_var = self.variable(
        "cache",
        "cached_prefill_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_segment_id_var = self.variable(
        "cache",
        "cache_prefill_segment_id",
        nn.with_logical_partitioning(jnp.zeros, (CACHE_BATCH, CACHE_SEQUENCE)),
        (cache_logical_shape[0], self.max_prefill_predict_length),
        jnp.int32,
    )

    if self.quantize_kvcache:

      if self.kv_quant_axis == "dkv":
        cache_scale_logical_shape = (batch, self.max_prefill_predict_length, heads, 1)
      elif self.kv_quant_axis == "heads_and_dkv":
        cache_scale_logical_shape = (batch, self.max_prefill_predict_length, 1, 1)

      cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.prefill_cache_axis_order)
      cache_scale_shape = self.transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_prefill_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_prefill_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var

  def _get_ar_cache_vars(self, batch, heads, kv_head_size):

    dtype = jnp.int8 if self.quantize_kvcache else self.dtype
    cache_length = self.max_target_length - self.max_prefill_predict_length
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)

    cache_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.ar_cache_axis_order)
    cache_shape = self.transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    # TODO(b/339703100): investigate the issue why with_logical_partitioning doesn't enforce sharding
    cached_key_var = self.variable(
        "cache",
        "cached_ar_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_key_var.value = nn.with_logical_constraint(
        cached_key_var.value,
        cache_axis_names,
    )

    cached_value_var = self.variable(
        "cache",
        "cached_ar_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_value_var.value = nn.with_logical_constraint(
        cached_value_var.value,
        cache_axis_names,
    )

    cached_segment_id_var = self.variable(
        "cache",
        "cache_ar_segment_id",
        nn.with_logical_partitioning(jnp.zeros, (CACHE_BATCH, CACHE_SEQUENCE)),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    if self.quantize_kvcache:

      if self.kv_quant_axis == "dkv":
        cache_scale_logical_shape = (batch, cache_length, heads, 1)
      elif self.kv_quant_axis == "heads_and_dkv":
        cache_scale_logical_shape = (batch, cache_length, 1, 1)

      cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      cache_scale_shape = self.transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_ar_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_ar_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    cache_index_var = self.variable(
      "cache", "cache_ar_index", nn.with_logical_partitioning(jnp.zeros, ()), (1,), jnp.int32)
    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var, cache_index_var

  def kv_cache_prefill(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
  ):
    """In prefill mode, we zero out the existing cache, run the computation and
    prepare the cache as necessary.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens

    Returns:
      key, value, decoder_segment_id.

    """
    batch, _, heads, kv_head_size = key.shape
    assert key.dtype == value.dtype, "Key and Value Dtypes should match."

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(batch, heads, kv_head_size)
    _ = self._get_ar_cache_vars(batch, heads, kv_head_size)  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    if self.quantize_kvcache:
      prefill_key_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.prefill_cache_axis_order)
      key_shaped_for_cache, key_scale_shaped_for_cache = quantizations.quantize_kv(
        key_shaped_for_cache, self.kv_quant_axis, prefill_key_axis_names)
      value_shaped_for_cache, value_scale_shaped_for_cache = quantizations.quantize_kv(
        value_shaped_for_cache, self.kv_quant_axis, prefill_key_axis_names)
      cached_prefill_key_vars[1].value = key_scale_shaped_for_cache
      cached_prefill_value_vars[1].value = value_scale_shaped_for_cache

    cached_prefill_key_vars[0].value = key_shaped_for_cache
    cached_prefill_value_vars[0].value = value_shaped_for_cache

    if decoder_segment_ids is not None:
      cached_prefill_segment_id_var.value = decoder_segment_ids

    return key, value, decoder_segment_ids

  def update_ar_key_value(
      self,
      one_token_key: Array,
      one_token_value: Array,
      cached_key_vars: tuple[nn.Variable, nn.Variable | None],
      cached_value_vars: tuple[nn.Variable, nn.Variable | None],
      one_hot_indices: Array,
  ) -> None:
    """Adds a single token's results to the ar kv cache

    Args:
        one_token_key (Array): Key of one token to add to the cache
        one_token_value (Array): Value of one token to add to the cache
        cached_ar_key (tuple[nn.Variable, nn.Variable|None],): Cached keys to add new token key to, possibly with scale
        cached_ar_value (tuple[nn.Variable, nn.Variable|None],: Cached values to add new token value to, possible with scale
        one_hot_indices (Array): Location of the new token within the cache

    Returns:
        tuple[Array, Array]: Updated caches for key and value with new token info added
    """

    cached_key_var, cached_key_scale_var = cached_key_vars
    cached_value_var, cached_value_scale_var = cached_value_vars

    # In order to update the key, value caches with the current key and
    # value, we reshape the one_token_key and one_token_value
    one_token_key_shaped_for_cache = jnp.transpose(one_token_key, self.ar_cache_axis_order)
    one_token_value_shaped_for_cache = jnp.transpose(one_token_value, self.ar_cache_axis_order)

    ar_cache_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.ar_cache_axis_order)
    if self.quantize_kvcache:
      one_token_key_shaped_for_cache, one_token_key_scale_shaped_for_cache = quantizations.quantize_kv(
        one_token_key_shaped_for_cache, self.kv_quant_axis, ar_cache_axis_names)
      one_token_value_shaped_for_cache, one_token_value_scale_shaped_for_cache = quantizations.quantize_kv(
        one_token_value_shaped_for_cache, self.kv_quant_axis, ar_cache_axis_names)

    one_hot_indices = one_hot_indices.astype(int)
    ar_cache_update_idx = jnp.squeeze(one_hot_indices)

    ar_cache_update_axis = ar_cache_axis_names.index(CACHE_SEQUENCE)
    cached_key_var.value = jax.lax.dynamic_update_index_in_dim(
      cached_key_var.value, one_token_key_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis)
    cached_key_var.value = nn.with_logical_constraint(cached_key_var.value, ar_cache_axis_names)
    cached_value_var.value = jax.lax.dynamic_update_index_in_dim(
      cached_value_var.value, one_token_value_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis)
    cached_value_var.value = nn.with_logical_constraint(cached_value_var.value, ar_cache_axis_names)

    if self.quantize_kvcache:
      ar_cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      ar_cache_scale_update_axis = ar_cache_scale_axis_names.index(CACHE_SCALE_SEQUENCE)
      cached_key_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_scale_var.value, one_token_key_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis)
      cached_value_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_scale_var.value, one_token_value_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis)

    return

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order):
    cache_var, cache_scale_var = cache_vars
    cached_value = cache_var.value
    if cache_scale_var is not None:
      cached_scale_value = cache_scale_var.value
      cached_value = quantizations.unquantize_kv(cached_value, cached_scale_value, target_dtype)

    cache_value_in_logical_shape = self.reverse_transepose(cached_value, cache_axis_order)
    return cache_value_in_logical_shape

  def kv_cache_autoregressive(
      self,
      key: Array,
      value: Array,
  ):
    """In autoregressive mode, we update the cache for this entry and
       then return the full cache.

    Args:
      key: in shape [b, 1, n, d].
      value: in shape [b, 1, n, d].
      decoder_segment_ids: [b, 1] -- marking segment ids for tokens

    Returns:
      tuple of (key, value, segment_id) for both prefill and ar cache,
    Raises:
      ValueError: when key/value shape is not [batch, 1, num_heads, heads_dim].
    """
    batch, sequence, heads, kv_head_size = key.shape
    if sequence != 1:
      raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")
    is_initialized = self.has_variable("cache", "cache_ar_index")
    if not is_initialized:
      raise ValueError("Error, we can't do autoregression if we haven't seeded the KV Cache.")

    cached_ar_key_vars, cached_ar_value_vars, cached_ar_segment_id_var, cache_ar_index_var = self._get_ar_cache_vars(batch, heads, kv_head_size)

    self.update_ar_key_value(key, value, cached_ar_key_vars, cached_ar_value_vars, cache_ar_index_var.value)
    active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    cached_ar_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
        cached_ar_segment_id_var.value, active_indicator, jnp.squeeze(cache_ar_index_var.value), 1
    )
    cache_ar_index_var.value = jnp.mod(cache_ar_index_var.value + 1, self.max_target_length - self.max_prefill_predict_length)

    # The below retrieves the existing prefill cache variables, not creating new ones
    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(batch, heads, kv_head_size)

    cached_prefill = (
        self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order),
        self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order),
        cached_prefill_segment_id_var.value,
    )

    cached_ar = (
        self.get_cached_values(cached_ar_key_vars, key.dtype, self.ar_cache_axis_order),
        self.get_cached_values(cached_ar_value_vars, value.dtype, self.ar_cache_axis_order),
        cached_ar_segment_id_var.value,
    )
    return cached_prefill, cached_ar

  def kv_cache(self, key: Array, value: Array, decoder_segment_ids: Array, model_mode: str) -> tuple:
    """KV cache takes the current state and updates the state accordingly.

    The key and value have dimension [b, s, n_kv, d],
    but we cache them with a reshape as defined in *_axis_order config as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n_kv, d].
      value: in shape [b, s, n_kv, d].
      model_mode: model mode controlling model

    Returns:
      two tuples of (k, v, decoder_segments) -- either can be Nones

    """
    if key.shape != value.shape:
      raise ValueError(f"Can't KV cache with mismatched shapes {key.shape=}, {value.shape=}")

    if model_mode == common_types.MODEL_MODE_TRAIN:
      return (key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")

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
  def __call__(self, query, key, value, decoder_segment_ids, model_mode):
    prefill_kv_cache, ar_kv_cache = self.kv_cache(key, value, decoder_segment_ids, model_mode)

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=prefill_kv_cache[0],
        value=prefill_kv_cache[1],
        decoder_segment_ids=prefill_kv_cache[2],
        model_mode=model_mode,
    )

    # Return the "prefill" cache if it actually the combined prefill+ar kv cache
    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum = self.apply_attention(
        query=query,
        key=ar_kv_cache[0],
        value=ar_kv_cache[1],
        decoder_segment_ids=ar_kv_cache[2],
        model_mode=model_mode,
    )

    unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
    exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
    exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
    return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)


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
    quantize_kvcache: bool, quantize the kv cache.
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
  quantize_kvcache: bool = False

  # Shard the query activation as the same as the key and value.
  # TODO: Find a better sharding axis name.
  query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False
  kv_quant_axis: str = "heads_and_dkv"

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

    def query_init(*args):
      # pylint: disable=no-value-for-parameter
      return self.kernel_init(*args) / depth_scaling

    query_proj = DenseGeneral(
        features=(self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=query_init,
        kernel_axes=("embed", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="query",
        quant=self.quant,
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

    kernel_axes = ("embed", "kv_heads", "kv_head_dim")

    kv_proj = DenseGeneral(
        features=(self.num_kv_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
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
    )(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_proj = DenseGeneral(
        features=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
    )(out)
    return out_proj

  def key_rotary(self, key: Array, inputs_positions: Array):
    """Apply Rotary Embedding to key."""
    key = RotaryEmbedding(min_timescale=self.config.rope_min_timescale, max_timescale = self.config.rope_max_timescale, 
                          embedding_dims=self.head_dim, name="key_rotary")(inputs=key, position=inputs_positions)
    return key

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
    # apply projection.
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")

    # apply ROPE
    query = RotaryEmbedding(min_timescale=self.config.rope_min_timescale, max_timescale = self.config.rope_max_timescale,
                             embedding_dims=self.head_dim, name="query_rotary")(inputs=query, position=inputs_positions)
    key = self.key_rotary(key, inputs_positions)

    # annotate with sharding constraint.
    query = nn.with_logical_constraint(query, self.query_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = nn.with_logical_constraint(key, self.key_axis_names)
    key = checkpoint_name(key, "key_proj")
    value = nn.with_logical_constraint(value, self.value_axis_names)
    value = checkpoint_name(value, "value_proj")

    attention_op = AttentionOp(
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        quantize_kvcache=self.quantize_kvcache,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        compute_axis_order=self.compute_axis_order,
        reshape_q=self.reshape_q,
        kv_quant_axis=self.kv_quant_axis,
    )

    out = attention_op(query, key, value, decoder_segment_ids, model_mode)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, "out_proj")
    return out
