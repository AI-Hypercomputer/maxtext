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

"""MLA Attention Layer."""

import math
from typing import Any, Optional, Tuple
import copy

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding
import jax.numpy as jnp

from flax import nnx

from MaxText.common_types import (
    Array,
    AxisIdxes,
    AxisNames,
    BATCH,
    BATCH_NO_EXP,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    D_KV,
    DType,
    EMBED,
    EP_AS_CONTEXT,
    HEAD,
    Q_LORA_UP_PROJ,
    KV_BATCH,
    KV_BATCH_NO_EXP,
    KV_HEAD,
    KV_HEAD_DIM,
    KV_LORA_UP_PROJ,
    LENGTH,
    LENGTH_NO_EXP,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_KV_BATCH,
    PREFILL_LENGTH,
    AttentionType,
    DEFAULT_MASK_VALUE,
)
from MaxText.inference import kvcache
from MaxText.inference import page_manager
from MaxText.inference import paged_attention
from MaxText.inference.kvcache import KVQuant
from MaxText.sharding import create_sharding
from MaxText.layers import nnx_wrappers
from MaxText.layers.attentions import Attention
from MaxText.layers.initializers import nd_dense_init, NdInitializer, variable_to_logically_partitioned
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant


class Indexer(nnx.Module):
  """
  Indexer for DeepSeek Sparse Attention (DSA).
  Introduced by DeepSeek V3.2: https://arxiv.org/pdf/2512.02556.
  """

  def __init__(
      self,
      config: Any,
      rotary_embedding,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      quant: Optional[Quant] = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.config = config
    self.rotary_embedding = rotary_embedding
    self.quant = quant
    self.kernel_init = kernel_init
    self.model_mode = model_mode
    self.rngs = rngs
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype

    self.n_heads = config.index_n_heads
    self.head_dim = config.index_head_dim
    self.index_topk = config.index_topk
    self.emb_dim = config.emb_dim
    self.rope_head_dim = config.qk_rope_head_dim
    self.q_lora_rank = config.q_lora_rank
    self.softmax_scale = self.head_dim**-0.5

    # Projection: Latent Query -> Indexer Query
    self.wq_b = DenseGeneral(
        in_features_shape=self.q_lora_rank,
        out_features_shape=(self.n_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        # TODO(shuningjin): double check kernel axes
        kernel_axes=("q_lora", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Projection: Input -> Shared Indexer Key
    self.wk = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Projection: Input -> Importance Weights for Heads (with high precision)
    self.weights_proj = DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.n_heads,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "q_heads"),
        # Enforce FP32 as requested for stability in importance scoring
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        quant=None,  # Typically we don't quantize the importance selector head
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Key Normalization with bias
    self.k_norm = nnx.LayerNorm(num_features=self.head_dim, use_bias=True, dtype=self.weight_dtype, rngs=rngs)

  def apply_partial_rope(
      self,
      inputs: Array,
      inputs_positions: Optional[Array | None] = None,
  ):
    """Applies partial RoPE to the indexer query or key

    The Indexer's RoPE implementation differs from MLA's in two key aspects:
    1. Split Order: Indexer splits the head dimension into [rope, nope], whereas MLA uses [nope, rope].
    2. Input Layout: Indexer uses concatenated layout (interleave=False), whereas MLA uses interleaved (interleave=True).

    Args:
      inputs: Input array of shape [batch, seqlen, index_n_heads, index_head_dim].
      positions: Position array of shape [batch, seqlen].

    Returns:
      Array with partial RoPE applied, with shape [batch, seqlen, index_n_heads, index_head_dim] 
    """
    # index_head_dim -> [rope_head_dim, index_head_dim - rope_head_dim]
    x_pe, x_nope = jnp.split(inputs, [self.rope_head_dim], axis=-1)
    # x_pe [B, S, H, rope_head_dim], positions [B, S]
    x_pe = self.rotary_embedding(x_pe, position=inputs_positions)
    x = jnp.concatenate([x_pe, x_nope], axis=-1)
    return x

  def generate_mask(self, topk_indices, s):
    """
    Creates a mask for top-k indices.

    Args:
        topk_indices: [b, t, k] int - The indices to keep.
        s: int - The total size to select from.

    Returns:
        mask: [b, t, s] - `0.0` at topk_indices, `DEFAULT_MASK_VALUE` (large negative) elsewhere.
    """
    # 1. Create a range [0, 1, ..., s-1]
    # 2. Broadcast compare against [b, t, k] to get [b, t, k, s]
    # 3. Use .any() to see if a s-index is present in any of the k slots
    is_topk = (jnp.arange(s) == topk_indices[..., None]).any(axis=-2)
    # 4. Use where to select between 0.0 and the mask value
    # cast values to dtype
    val_true = jnp.array(0.0, dtype=self.config.dtype)
    val_false = jnp.array(DEFAULT_MASK_VALUE, dtype=self.config.dtype)
    return jnp.where(is_topk, val_true, val_false)

  def __call__(
      self,
      inputs_q: Array,
      low_rank_q: Array,
      inputs_kv: Array,
      inputs_positions: Optional[Array | None] = None,
      attention_mask: Optional[Array | None] = None,
  ):
    """Computes the index score to determine the top-k relevant tokens.

    This uses a ReLU-based similarity for QK with MQA-style broadcasting (shared K).
    It uses weighted aggregation over heads to produce a single score per token pair.

    Steps:
      1. Q = RoPE(Wq @ q_lora)
      2. K = RoPE(Norm(Wk @ X))
      3. Logits = ReLU(Q @ K.T)                 # Pairwise similarity
      4. Head_Weights = (W_proj @ X) * scale    # Dynamic head importance
      5. Score = Sum(Logits * Head_Weights)     # Aggregate heads

    Args:
      inputs_q: Input of shape [b, t, embed_dim].
      low_rank_q: Low-rank latent query representations of shape [b, t, q_lora_rank].
      inputs_kv: Input of shape [b, s, embed_dim], same as inputs_q
      inputs_positions: Position indices of shape [b, s].
      attention_mask: Optional attention mask of shape [b, t, s].
        Positions with `0.0` allow attention, while positions with
        `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
        Returns `None` if no masking is determined to be necessary based on
        the inputs and configuration.

    Returns:
        - index_mask: A sparse mask [b, t, s] with 0.0 for top-k selected tokens
          and large negative values otherwise.
        - topk_indices: Indices of the top-k selected tokens [b, t, k].
        - index_score: The computed relevance scores [b, t, s].

    Notation:
      b: Batch size
      t: Query Sequence Length (Target), note t = s here
      s: Key/Value Sequence Length (Source)
      h: Number of Indexer Heads (index_n_heads)
      d: Indexer Head Dimension (index_head_dim)
    """
    # NOTE: If sequence length <= topk, indexer always selects all tokens.
    if self.config.max_target_length <= self.index_topk:
      return None, None, None

    bsz, seqlen = inputs_positions.shape  # s = t = seqlen

    # Query Processing: Project from Latent low_rank_q
    # TODO(shuningjin): why the shape is [b, t, h * d] rather than [b, t, h, d]?
    q = self.wq_b(low_rank_q)  # [b, t, q_lora_rank] -> [b, t, h * d]
    q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)  # [b, t, h, d]
    q = self.apply_partial_rope(q, inputs_positions=inputs_positions)

    # Key Processing: Project from Input
    k = self.wk(inputs_kv)  # [b, s, embed_dim] -> [b, s, d]
    k = self.k_norm(k)
    k = k[:, :, None, :]  # [b, s, d] -> [b, s, 1, d]
    k = self.apply_partial_rope(k, inputs_positions=inputs_positions)
    k = k.squeeze(2)  # [b, s, 1, d] -> [b, s, d]

    # Compute Index Scores
    # qk product: relu(q @ k.T), [b, t, s, h]
    # similar to MQA, each key is shared by h query head
    logits = jnp.einsum("bthd, bsd -> btsh", q, k, precision=self.config.matmul_precision)
    logits = jax.nn.relu(logits)
    # compute head weights: project from input, [b, t, embed_dim] -> [b, t, h]
    weights = self.weights_proj(inputs_q.astype(jnp.float32)) * (self.n_heads**-0.5) * self.softmax_scale
    # weighted sum over head: sum_h(logits * weights)
    index_score = jnp.einsum("btsh, bth -> bts", logits, weights, precision=self.config.matmul_precision) # [b, t, s]

    # Apply attention mask before TopK
    if attention_mask is not None:
      index_score += attention_mask

    # TopK selection based on index score
    _, topk_indices = jax.lax.top_k(index_score, k=self.index_topk) # topk_indices [b, t, k]

    # Create Sparse Index Mask: 0 and large negatives
    index_mask = self.generate_mask(topk_indices, seqlen) # [b, t, s]

    # Re-apply attention mask after TopK: in case number of unmasked tokens < TopK
    if attention_mask is not None:
      index_mask += attention_mask

    return index_mask, topk_indices, index_score


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
    attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
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
      attention_type: AttentionType = AttentionType.MLA,  # Default to MLA attention
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

    # Initialize Indexer
    self.use_sparse_indexer = config.use_sparse_indexer
    if self.use_sparse_indexer:
      # Need two versions of rope.
      # MLA applies yarn with interleave layout.
      # Indexer applies yarn with concatenate layout.
      indexer_rope = copy.copy(self.rotary_embedding)
      indexer_rope.interleave = False
      self.indexer = Indexer(
          config,
          rngs=rngs,
          rotary_embedding=indexer_rope,
          kernel_init=kernel_init,
          quant=quant,
          model_mode=model_mode,
      )

    # Module attribute names must match names previously passed to Linen for checkpointing
    self.MlaKVCache_0 = self.init_mla_kv_caches(inputs_kv_shape) if model_mode != MODEL_MODE_TRAIN else None

  def _init_projections(self, inputs_q_shape: Tuple, inputs_kv_shape: Tuple) -> None:
    """Initializes the MLA-specific projections."""
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
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )
    else:
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
          in_features_shape=self.config.emb_dim,
          out_features_shape=self.q_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_lora_up_proj"),
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
          out_features_shape=(self.num_query_heads, self.qk_head_dim),
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

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora_up_proj"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        shard_mode=self.config.shard_mode,
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
        shard_mode=self.config.shard_mode,
        rngs=self.rngs,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    self.out = self.init_out_w(output_dim=inputs_q_shape[-1])

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
    # specify query logical name
    if model_mode == MODEL_MODE_PREFILL:
      query_logical_name = self.prefill_query_axis_names
      wqa_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, Q_LORA_UP_PROJ)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      query_logical_name = self.ep_query_axis_names
      wqa_logical_name = (KV_BATCH_NO_EXP, LENGTH, Q_LORA_UP_PROJ)
    else:
      query_logical_name = self.query_axis_names
      wqa_logical_name = (KV_BATCH, LENGTH_NO_EXP, Q_LORA_UP_PROJ)
    query_sharding = create_sharding(self.mesh, query_logical_name)
    wqa_out_sharding = create_sharding(self.mesh, wqa_logical_name)
    # Set softmax scaling.
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.softmax_scale = self.qk_head_dim**-0.5
    if self.max_position_embeddings > self.original_max_position_embeddings:
      mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    # Low-rank latent vector for queries. This is also accessed by indexer.
    low_rank_q = None

    if self.q_lora_rank == 0:
      q = self.query(inputs_q, out_sharding=query_sharding)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q, out_sharding=wqa_out_sharding)  # [B, L, q_lora_rank]
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      low_rank_q = checkpoint_name(low_rank_q, "mla_q")
      q = self.wq_b(low_rank_q, out_sharding=query_sharding)  # [B, L, n_heads, qk_head_dim]

    # Partial RoPE: Split into non-positional and rotary parts.
    # last dimension: qk_nope_head_dim, qk_rope_head_dim
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_nope = self._maybe_shard_with_logical(q_nope, query_logical_name)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions=inputs_positions)
    q_pe = self._maybe_shard_with_logical(q_pe, query_logical_name)
    # Query projection is scaled by self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    query = jnp.concatenate([q_nope, q_pe], axis=-1) * self.softmax_scale
    query = self._maybe_shard_with_logical(query, query_logical_name)
    return query, low_rank_q

  def mla_get_key_value(self, low_rank_main, key_rope, model_mode):
    """get (key,value) pair from mla"""
    if model_mode == MODEL_MODE_PREFILL:
      key_logical_name = self.prefill_key_axis_names
      value_logical_name = self.prefill_value_axis_names
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      key_logical_name = self.ep_key_axis_names
      value_logical_name = self.ep_value_axis_names
    else:
      key_logical_name = self.key_axis_names
      value_logical_name = self.value_axis_names

    wkva_out_sharding = create_sharding(self.mesh, key_logical_name)
    kv_out = self.wkv_b(low_rank_main, out_sharding=wkva_out_sharding)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))
    key_nope = self._maybe_shard_with_logical(key_nope, key_logical_name)
    key_rope = self._maybe_shard_with_logical(key_rope, key_logical_name)

    key = jnp.concatenate([key_nope, key_rope], axis=-1)

    key = self._maybe_shard_with_logical(key, key_logical_name)
    value = self._maybe_shard_with_logical(value, value_logical_name)
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
    if model_mode == MODEL_MODE_PREFILL:
      wka_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, KV_LORA_UP_PROJ)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      wka_logical_name = (KV_BATCH_NO_EXP, LENGTH, KV_LORA_UP_PROJ)
    else:
      wka_logical_name = (KV_BATCH, LENGTH_NO_EXP, KV_LORA_UP_PROJ)
    wkva_out_sharding = create_sharding(self.mesh, wka_logical_name)
    low_rank = self.wkv_a(inputs, out_sharding=wkva_out_sharding)
    low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)
    low_rank_main = self.kv_norm(low_rank_main)
    low_rank_main = checkpoint_name(low_rank_main, "mla_kv")
    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions=inputs_positions)

    key, value = self.mla_get_key_value(low_rank_main, key_rope, model_mode)
    cached_values = [None, None]
    if self.config.attention != "paged" and model_mode != MODEL_MODE_TRAIN:
      if self.config.mla_naive_kvcache:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      else:
        cached_values = self.update_mla_kv_caches(
            low_rank_main, key_rope, decoder_segment_ids, model_mode, previous_chunk
        )

    return key, value, cached_values

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      bidirectional_mask: Optional[Any] = None,
      rope_kwargs: dict | None = None,
      kv_cache: Optional[Array] = None,
      attention_metadata: Optional[dict[str, Any]] = None,
  ) -> tuple[Array, Optional[Array]]:
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
      kv_cache: Optional key-value cache used when serving models with vLLM.
      attention_metadata: Optional attention-related metadata used when serving models with vLLM.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    if model_mode == MODEL_MODE_PREFILL:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.prefill_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.prefill_input_axis_names)
      out_logical_name = (PREFILL_KV_BATCH, PREFILL_LENGTH, HEAD, D_KV)
    elif model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.ep_input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.ep_input_axis_names)
      out_logical_name = (BATCH_NO_EXP, LENGTH, HEAD, D_KV)
    else:
      inputs_q = self._maybe_shard_with_logical(inputs_q, self.input_axis_names)
      inputs_kv = self._maybe_shard_with_logical(inputs_kv, self.input_axis_names)
      out_logical_name = (BATCH, LENGTH_NO_EXP, HEAD, D_KV)

    query, low_rank_q = self.mla_query_projection(inputs_q, inputs_positions, model_mode)
    key, value, cached_values = self.mla_kv_projection(
        inputs_kv, inputs_positions, decoder_segment_ids, model_mode, previous_chunk
    )
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    # Indexer Logic
    index_mask = None
    if self.use_sparse_indexer:
      if self.q_lora_rank == 0:
        raise NotImplementedError("Sparse indexer has not implemented for q_lora_rank = 0.")
      if model_mode != MODEL_MODE_TRAIN:
        raise NotImplementedError("Sparse indexer has not implemented for inference yet.")
      # generate mask: with 0 and large negative, [b, 1, 1, q_len, kv_len] -> [b, q_len, kv_len]
      attention_mask = self.attention_op.generate_attention_mask(
          query, key, decoder_segment_ids, model_mode, previous_chunk, bidirectional_mask
      ).squeeze(axis=(1, 2))
      # apply indexer, index_mask [b, q_len, kv_len]
      index_mask, _, _ = self.indexer(
          inputs_q=inputs_q,
          low_rank_q=low_rank_q,
          inputs_kv=inputs_kv,
          inputs_positions=inputs_positions,
          attention_mask=attention_mask,
      )
      if index_mask is not None:
        index_mask = index_mask[:, None, None, :, :]  # [b, 1, 1, q_len, kv_len]

    if self.config.attention == "paged" and model_mode != MODEL_MODE_TRAIN:
      unnormalized_out, _, exp_sum = self.ds_paged_attention_op(
          query, key, value, decoder_segment_ids, model_mode, previous_chunk, slot=slot, page_state=page_state
      )
      unnormalized_out = unnormalized_out[..., : self.v_head_dim]
      out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
      # Pass the index_mask to the Attention Op
      out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, cached_values, index_mask=index_mask)

    if model_mode == MODEL_MODE_TRAIN and self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      out = self._maybe_shard_with_logical(out, self.ep_out_axis_names)
    else:
      out = self._maybe_shard_with_logical(out, self.out_axis_names)

    out_sharding = create_sharding(self.mesh, out_logical_name)
    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache
