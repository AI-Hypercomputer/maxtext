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
from typing import Any, Optional, Tuple, Union

from flax import linen as nn
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import common_types
from kernels.ragged_attention import ragged_gqa
from kernels.ragged_attention import ragged_mha
import page_managers
from layers import embeddings
from layers import initializers
from layers import linears
from layers import quantizations
from page_managers import PageManager


# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


class AttentionType(enum.Enum):
  GLOBAL = "global"
  LOCAL_SLIDING = "local_sliding"


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
PRNGKey = common_types.PRNGKey

DenseGeneral = linears.DenseGeneral
RotaryEmbedding = embeddings.RotaryEmbedding
NdInitializer = initializers.NdInitializer
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant
KVTensor = quantizations.KVTensor

AxisNames = common_types.AxisNames
AxisIdxes = common_types.AxisIdxes
BATCH = common_types.BATCH
PREFILL_KV_BATCH = common_types.PREFILL_KV_BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
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


class PagedAttentionOp(nn.Module):
    """Paged Attention Operator for efficient attention computation with KV caching."""
    mesh: Mesh
    num_pages: int
    tokens_per_page: int
    max_pages_per_slot: int
    max_pages_per_prefill: int
    pages_per_compute_block: int
    num_kv_heads: int
    kv_head_dim_size: int
    config: Config
    dtype: DType = jnp.bfloat16

    @staticmethod
    def get_cache_axis_names():
        """Return sharding specifications for different cache components.
        
        Returns:
            Dict mapping cache component names to their sharding axis names.
        """
        return {
            "key_pages": ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size"),
            "value_pages": ("paged_kv_heads", "num_pages", "tokens_per_page", "paged_kv_head_dim_size"),
            "page_status": ("num_pages",),
            "page_map": ("cache_batch", "max_pages_per_slot"),
            "sequence_lengths": ("cache_batch",),
            "num_pages_used": ("cache_batch",),
            "current_page": ("cache_batch",),
            "current_page_position": ("cache_batch",)
        }

    def setup(self):
        """Initialize KV cache and page management state variables."""
        cache_axis_names = self.get_cache_axis_names()
        num_slots = int(self.config.per_device_batch_size * jax.device_count())

        # Initialize key pages
        self.key_pages = self.variable(
            "cache", "key_pages",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=self.dtype),
                cache_axis_names["key_pages"]
            ),
            (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size),
        )

        # Initialize value pages
        self.value_pages = self.variable(
            "cache", "value_pages",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=self.dtype),
                cache_axis_names["value_pages"]
            ),
            (self.num_kv_heads, self.num_pages, self.tokens_per_page, self.kv_head_dim_size),
        )

        # Page management state
        self.page_status = self.variable(
            "cache", "page_status",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                cache_axis_names["page_status"]
            ),
            (self.num_pages,),
        )

        self.page_map = self.variable(
            "cache", "page_map",
            nn.with_logical_partitioning(
                lambda shape: jnp.full(shape, -1, dtype=jnp.int32),
                cache_axis_names["page_map"]
            ),
            (num_slots, self.max_pages_per_slot)
        )

        self.sequence_lengths = self.variable(
            "cache", "sequence_lengths",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                cache_axis_names["sequence_lengths"]
            ),
            (num_slots,)
        )

        self.num_pages_used = self.variable(
            "cache", "num_pages_used",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                cache_axis_names["num_pages_used"]
            ),
            (num_slots,)
        )

        self.current_page = self.variable(
            "cache", "current_page",
            nn.with_logical_partitioning(
                lambda shape: jnp.full(shape, -1, dtype=jnp.int32),
                cache_axis_names["current_page"]
            ),
            (num_slots,)
        )

        self.current_page_position = self.variable(
            "cache", "current_page_position",
            nn.with_logical_partitioning(
                lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                cache_axis_names["current_page_position"]
            ),
            (num_slots,)
        )

    def find_next_free_page(self, page_status: Array) -> Array:
        """Finds index of next free page."""
        free_pages = jnp.where(page_status[1:] == 0, size=1, fill_value=-1)[0] + 1
        return free_pages[0]

    def release_slot_pages(self, slot: int) -> None:
        """Releases all pages associated with a slot."""
        # Find used pages
        used_pages = jnp.where(
            self.page_map.value[slot] > -1,
            self.page_map.value[slot],
            0
        )

        # Reset KV cache entries
        self.key_pages.value = self.key_pages.value.at[:, used_pages, :, :].set(0)
        self.value_pages.value = self.value_pages.value.at[:, used_pages, :, :].set(0)

        # Reset page status
        self.page_status.value = self.page_status.value.at[used_pages].set(0)

        # Reset slot state
        self.page_map.value = self.page_map.value.at[slot, :].set(-1)
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(0)
        self.num_pages_used.value = self.num_pages_used.value.at[slot].set(0)
        self.current_page.value = self.current_page.value.at[slot].set(-1)
        self.current_page_position.value = self.current_page_position.value.at[slot].set(0)

    def update_prefill_step_pages(
        self,
        key_pages_var: nn.Variable,
        value_pages_var: nn.Variable,
        key: Array,
        value: Array,
        *,
        slot: int,
        true_length: int,
    ) -> None:
        """Updates pages during prefill phase."""
        # Calculate pages needed
        num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page

        def _reserve_single_page(i, carry):
            page_status, page_map = carry
            next_free_page = self.find_next_free_page(page_status)
            page_status = page_status.at[next_free_page].set(1)
            page_map = page_map.at[slot, i].set(next_free_page)
            return page_status, page_map

        # Allocate required pages
        self.page_status.value, self.page_map.value = jax.lax.fori_loop(
            0, num_pages_needed, _reserve_single_page,
            (self.page_status.value, self.page_map.value)
        )

        def process_page(i, carry):
            """Process a single page of KV data."""
            key_pages, value_pages = carry
            
            start_token = i * self.tokens_per_page
            target_page = self.page_map.value[slot, i]
            
            # Extract page data
            k_data = jax.lax.dynamic_slice(
                key,
                (0, start_token, 0, 0),
                (1, self.tokens_per_page, self.num_kv_heads, self.kv_head_dim_size)
            )
            k_data = jnp.reshape(k_data, (self.num_kv_heads, self.tokens_per_page, self.kv_head_dim_size))
            
            v_data = jax.lax.dynamic_slice(
                value,
                (0, start_token, 0, 0),
                (1, self.tokens_per_page, self.num_kv_heads, self.kv_head_dim_size)
            )
            v_data = jnp.reshape(v_data, (self.num_kv_heads, self.tokens_per_page, self.kv_head_dim_size))
            
            # Update cache
            key_pages = key_pages.at[:, target_page, :, :].set(k_data)
            value_pages = value_pages.at[:, target_page, :, :].set(v_data)
            
            return key_pages, value_pages

        # Process all pages using fori_loop
        key_pages_var.value, value_pages_var.value = jax.lax.fori_loop(
            0, num_pages_needed, process_page,
            (key_pages_var.value, value_pages_var.value)
        )

        # Update state tracking - avoiding conditionals with traced values
        last_page = jnp.where(
            num_pages_needed > 0,
            self.page_map.value[slot, num_pages_needed - 1],
            -1
        )
        self.current_page.value = self.current_page.value.at[slot].set(last_page)
        
        last_page_position = jnp.where(
            true_length > 0,
            (true_length - 1) % self.tokens_per_page,
            0
        )
        self.current_page_position.value = self.current_page_position.value.at[slot].set(last_page_position)

    def update_decode_step_pages(
        self,
        key_pages_var: nn.Variable,
        value_pages_var: nn.Variable,
        key: Array,
        value: Array,
        *,
        slot: int,
    ) -> None:
        """Updates pages during decode steps with corrected reshaping."""
        curr_page = self.current_page.value[slot]
        curr_pos = self.current_page_position.value[slot]

        # Handle conditional logic with jnp.where
        need_new_page = jnp.logical_or(
            curr_page == -1,
            curr_pos >= self.tokens_per_page
        )
        
        # Find next free page if needed
        next_page = jnp.where(
            need_new_page,
            self.find_next_free_page(self.page_status.value),
            curr_page
        )
        
        # Update states using jnp.where
        self.page_status.value = jnp.where(
            need_new_page,
            self.page_status.value.at[next_page].set(1),
            self.page_status.value
        )
        
        self.page_map.value = jnp.where(
            need_new_page,
            self.page_map.value.at[slot, self.num_pages_used.value[slot]].set(next_page),
            self.page_map.value
        )
        
        self.num_pages_used.value = jnp.where(
            need_new_page,
            self.num_pages_used.value.at[slot].add(1),
            self.num_pages_used.value
        )
        
        self.current_page.value = self.current_page.value.at[slot].set(next_page)
        new_pos = jnp.where(need_new_page, 0, curr_pos)

        # CRITICAL FIX: Proper reshaping of key/value data
        # Extract single item for slot
        key_data = key[slot]  # Shape: [1, num_kv_heads, head_dim]
        value_data = value[slot]  # Shape: [1, num_kv_heads, head_dim]
        
        # Reshape to match cache layout
        key_data = jnp.reshape(key_data, (self.num_kv_heads, 1, self.kv_head_dim_size))
        value_data = jnp.reshape(value_data, (self.num_kv_heads, 1, self.kv_head_dim_size))

        # Update KV cache
        key_pages_var.value = key_pages_var.value.at[:, next_page, new_pos, :].set(
            key_data[:, 0, :]
        )
        value_pages_var.value = value_pages_var.value.at[:, next_page, new_pos, :].set(
            value_data[:, 0, :]
        )

        # Update position tracking
        self.current_page_position.value = self.current_page_position.value.at[slot].set(new_pos + 1)
        self.sequence_lengths.value = self.sequence_lengths.value.at[slot].add(1)


    @staticmethod
    @functools.partial(jax.jit, static_argnames=("model_mode", "pages_per_compute_block"))
    def paged_attention_impl(
        query: Array,
        key_pages: Array,
        value_pages: Array,
        sequence_lengths: Array,
        page_map: Array,
        model_mode: str,
        pages_per_compute_block: int,
    ) -> Array:
        """Core paged attention computation.
        
        Args:
            query: [batch, len, num_heads, head_dim]
            key_pages: [num_kv_heads, num_pages, tokens_per_page, head_dim]
            value_pages: [num_kv_heads, num_pages, tokens_per_page, head_dim]
            sequence_lengths: [batch]
            page_map: [batch, max_pages_per_slot]
            model_mode: Current model mode
            pages_per_compute_block: Number of pages per attention block
        """
        from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel

        if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
          # Reshape to the correct shape.  (b, 1, n, d) -> (b, n, d)
          query = jnp.squeeze(query, axis=1)
        
        return paged_attention_kernel.paged_attention(
            q=query,
            k_pages=key_pages,
            v_pages=value_pages,
            lengths=sequence_lengths,
            page_indices=page_map,
            pages_per_compute_block=pages_per_compute_block,
        )

    def paged_dot_product_attention(
        self,
        query: Array,
        key: Array,
        value: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute attention with causal masking for prefill phase.
        
        Args:
            query: [batch, seq_len, num_heads, head_dim]
            key: [batch, seq_len, num_kv_heads, head_dim]
            value: [batch, seq_len, num_kv_heads, head_dim]
            
        Returns:
            Tuple of (attention output, attention max scores, attention sum scores)
        """
        b, t, n, d = query.shape
        _, s, n_kv, _ = key.shape

        # Reshape query for GQA
        query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))

        # Compute attention weights
        attn_weights = jnp.einsum("btkgd,bskd->bkgts", query, key)

        # Apply causal mask
        mask_shape = (t, s)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = jax.lax.ge(row_ids, col_ids)
        causal_mask = jnp.reshape(causal_mask, (1, 1, 1, t, s))

        masked_weights = jnp.where(
            causal_mask,
            attn_weights,
            jnp.full_like(attn_weights, -1e10)
        )

        # Calculate attention
        local_max = jnp.max(masked_weights, axis=-1, keepdims=True)
        local_exps = jnp.exp(masked_weights - local_max)
        local_sums = jnp.sum(local_exps, axis=-1, keepdims=True)

        attn = jnp.einsum("bkgts,bskd->btkgd", local_exps, value)
        attn = jnp.reshape(attn, (b, t, n, d))

        # Reshape outputs for consistency with attention interface
        local_max = jnp.moveaxis(local_max, -2, 1)
        local_max = jnp.reshape(local_max, (b, t, n, 1))

        local_sums = jnp.moveaxis(local_sums, -2, 1)
        local_sums = jnp.reshape(local_sums, (b, t, n, 1))

        return attn, local_max, local_sums

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        decoder_segment_ids: Array,
        model_mode: str,
        *,
        slot: int,
        true_length: int,
    ) -> Union[Array, Tuple[Array, Array, Array]]:
        """Main entry point with proper shape handling."""
        
        if model_mode == common_types.MODEL_MODE_PREFILL:
            self.release_slot_pages(slot)
            
            num_pages_needed = (true_length + self.tokens_per_page - 1) // self.tokens_per_page
            self.sequence_lengths.value = self.sequence_lengths.value.at[slot].set(true_length)
            self.num_pages_used.value = self.num_pages_used.value.at[slot].set(num_pages_needed)

            self.update_prefill_step_pages(
                self.key_pages,
                self.value_pages,
                key,
                value,
                slot=slot,
                true_length=true_length
            )

            return self.paged_dot_product_attention(query, key, value)

        elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
            if key.shape[1] != 1:
                raise ValueError(f"Autoregressive mode expects sequence length 1, got {key.shape[1]}")

            self.update_decode_step_pages(
                self.key_pages,
                self.value_pages,
                key,
                value,
                slot=slot
            )

            return self.paged_attention_impl(
                query,
                self.key_pages.value,
                self.value_pages.value,
                self.sequence_lengths.value,
                self.page_map.value,
                model_mode,
                self.pages_per_compute_block
            )

        else:
            raise ValueError(f"Invalid model_mode: {model_mode}")


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
  prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV)
  ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV)
  ragged_lengths_names: AxisNames = (CACHE_BATCH,)
  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None
  attention_type: AttentionType = AttentionType.GLOBAL  # Default to global attention
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_ragged_attention: bool = False
  ragged_block_size: int = 256

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor, model_mode: str) -> None:
    """Check attention inputs."""
    # jax.debug.print("check_attention_inputs - model_mode: {}", model_mode)
    # jax.debug.print("check_attention_inputs - query.shape: {}", query.shape)
    # jax.debug.print("check_attention_inputs - key.shape: {}", key.shape)
    # jax.debug.print("check_attention_inputs - value.shape: {}", value.shape)

    # print(f"\ncheck_attention_inputs - model_mode: {model_mode}")
    # print(f"check_attention_inputs - query.shape: {query.shape}")
    # print(f"check_attention_inputs - key.shape: {key.shape}")
    # print(f"check_attention_inputs - value.shape: {value.shape}")

    assert key.ndim == value.ndim, "k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
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
  ):
    self.check_attention_inputs(query, key, value, model_mode)
    length = query.shape[-3]
    if use_ragged_attention and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      if lengths is None:
        lengths = jnp.sum(decoder_segment_ids, axis=-1)

      return self.ragged_attention(query, key, value, lengths, self.ragged_block_size)
    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
        or (self.attention_kernel == "paged")
    ):
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
    elif self.attention_kernel == "flash" or self.attention_kernel == "autoselected":
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

  def ragged_attention(
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
          raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
        mask &= splash_attention_mask.LocalMask(
            shape=(query.shape[2], query.shape[2]),
            window_size=(self.sliding_window_size, self.sliding_window_size),
            offset=0,
        )

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
    1. Stable API, supports GQA, SWA (only with causal masking)
    2. Head_dim = 256 is also supported from TE-1.12 stable release with CUDNN 12.6
    """
    # These imports are only meant to work in a GPU build.
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    sliding_window_size = self.sliding_window_size
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      sliding_window_size = [self.sliding_window_size, 0]
      mask_type = "causal"  # SWA only works with causal masking
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
        scale_factor=1.0 / math.sqrt(head_dim),
        transpose_batch_sequence=False,
        window_size=sliding_window_size,
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

    if self.reshape_q and q_seq_len == 1:
      local_max = local_max[:, 0:1, :, :]
      local_sum = local_sum[:, 0:1, :, :]
      local_out = local_out[:, 0:1, :, :]

    return local_out, local_max, local_sum

  def apply_attention_dot(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ):
    """Apply Attention."""
    validate_compute_axis_order(self.compute_axis_order)
    # Casting qk_product and softmaxt computation for float32 for model stability.
    if model_mode == common_types.MODEL_MODE_TRAIN and self.float32_qk_product:
      if isinstance(key, KVTensor):
        key = key.dequant()
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    q_seq_len = query.shape[1]
    attn_weights = self.qk_product(query, key, q_seq_len, model_mode)

    if self.attn_logits_soft_cap:
      attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
      attn_weights = attn_weights * self.attn_logits_soft_cap

    # Casting softmaxt computation for float32 for model stability.
    if model_mode == common_types.MODEL_MODE_TRAIN and self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
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

  def transpose_tuple(self, items: tuple[Any, Any, Any, Any], axis_order: AxisIdxes) -> tuple[Any, Any, Any, Any]:
    return tuple([items[i] for i in axis_order])

  def _get_cached_kv_dtype(self, dtype):
    return self.kv_quant.dtype if self.kv_quant else dtype

  def _get_cache_scale_logical_shape(self, batch, heads, cache_length):
    assert self.kv_quant
    if self.kv_quant.axis_cfg == "dkv":
      return (batch, cache_length, heads, 1)
    if self.kv_quant.axis_cfg == "heads_and_dkv":
      return (batch, cache_length, 1, 1)
    raise f"Invalid config for kv_quant_axis:{self.kv_quant.axis_cfg}"

  def _get_prefill_cache_vars(self, batch, heads, kv_head_size, model_mode):

    cache_length = self.max_prefill_predict_length
    dtype = self._get_cached_kv_dtype(self.dtype)
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names

    cache_axis_names = self.transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)
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
    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)

    cached_segment_id_var = self.variable(
        "cache",
        "cache_prefill_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
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

  def _get_ar_cache_vars(self, batch, heads, kv_head_size, model_mode):

    dtype = self._get_cached_kv_dtype(self.dtype)
    cache_length = self.max_target_length - self.max_prefill_predict_length
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names

    cache_axis_names = self.transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)
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

    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)
    cached_segment_id_var = self.variable(
        "cache",
        "cache_ar_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    cached_lengths_var = self.variable(
        "cache",
        "cached_ar_lengths",
        nn.with_logical_partitioning(jnp.zeros, (CACHE_BATCH,)),
        (cache_logical_shape[0],),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
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

    cache_index_var = self.variable("cache", "cache_ar_index", nn.with_logical_partitioning(jnp.zeros, ()), (1,), jnp.int32)
    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var, cache_index_var, cached_lengths_var

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

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, kv_head_size, common_types.MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(batch, heads, kv_head_size, common_types.MODEL_MODE_PREFILL)  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    if self.kv_quant:
      prefill_key_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.prefill_cache_axis_order)
      key_shaped_for_cache, key_scale_shaped_for_cache = self.kv_quant.quantize(key_shaped_for_cache, prefill_key_axis_names)
      value_shaped_for_cache, value_scale_shaped_for_cache = self.kv_quant.quantize(
          value_shaped_for_cache, prefill_key_axis_names
      )
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
      lengths: Array,
      use_ragged_attention: bool,
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
    if self.kv_quant:
      one_token_key_shaped_for_cache, one_token_key_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_key_shaped_for_cache, ar_cache_axis_names
      )
      one_token_value_shaped_for_cache, one_token_value_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_value_shaped_for_cache, ar_cache_axis_names
      )

    ar_cache_update_idx = jnp.squeeze(one_hot_indices)
    ar_cache_sequence_axis = ar_cache_update_axis = ar_cache_axis_names.index(CACHE_SEQUENCE)
    ar_cache_batch_axis = ar_cache_axis_names.index(CACHE_BATCH)

    if use_ragged_attention:
      cache_locations = [slice(None)] * 4
      new_token_locations = [slice(None)] * 4
      new_token_locations[ar_cache_sequence_axis] = 0

      def key_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_key_shaped_for_cache[tuple(new_token_locations)])

      def value_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_value_shaped_for_cache[tuple(new_token_locations)])

      cached_key_var.value = jax.lax.fori_loop(
          0, one_token_key_shaped_for_cache.shape[0], key_body, cached_key_var.value, unroll=8
      )
      cached_value_var.value = jax.lax.fori_loop(
          0, one_token_value_shaped_for_cache.shape[0], value_body, cached_value_var.value, unroll=8
      )

    else:
      one_hot_indices = one_hot_indices.astype(int)
      cached_key_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_var.value, one_token_key_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )
      cached_value_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_var.value, one_token_value_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )

    cached_key_var.value = nn.with_logical_constraint(cached_key_var.value, ar_cache_axis_names)
    cached_value_var.value = nn.with_logical_constraint(cached_value_var.value, ar_cache_axis_names)

    if self.kv_quant:
      ar_cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      ar_cache_scale_update_axis = ar_cache_scale_axis_names.index(CACHE_SCALE_SEQUENCE)
      cached_key_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_scale_var.value, one_token_key_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis
      )
      cached_value_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_scale_var.value,
          one_token_value_scale_shaped_for_cache,
          ar_cache_update_idx,
          ar_cache_scale_update_axis,
      )

    return

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order) -> jax.Array | KVTensor:
    cache_var, cache_scale_var = cache_vars
    cache_value = cache_var.value
    if cache_scale_var is not None:
      scale_value = cache_scale_var.value
      dtype = cache_value.dtype
      if dtype == jnp.int8:
        scale_value /= quantizations.MAX_INT8
      elif dtype == jnp.int4:
        scale_value /= quantizations.MAX_INT4

      cache_value = KVTensor(qvalue=cache_value, scale=[scale_value], scale_t=None, dequant_dtype=target_dtype, bias=[])
    cache_value_in_logical_shape = jax.tree.map(lambda x: self.reverse_transepose(x, cache_axis_order), cache_value)
    return cache_value_in_logical_shape

  def kv_cache_autoregressive(
      self,
      key: Array,
      value: Array,
      use_ragged_attention: bool = False,
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

    cached_ar_key_vars, cached_ar_value_vars, cached_ar_segment_id_var, cache_ar_index_var, cache_ar_lengths_var = (
        self._get_ar_cache_vars(batch, heads, kv_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE)
    )

    self.update_ar_key_value(
        key,
        value,
        cached_ar_key_vars,
        cached_ar_value_vars,
        cache_ar_index_var.value,
        cache_ar_lengths_var.value,
        use_ragged_attention,
    )
    active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    cached_ar_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
        cached_ar_segment_id_var.value, active_indicator, jnp.squeeze(cache_ar_index_var.value), 1
    )
    cache_ar_index_var.value = jnp.mod(
        cache_ar_index_var.value + 1, self.max_target_length - self.max_prefill_predict_length
    )
    cache_ar_lengths_var.value = cache_ar_lengths_var.value.at[:].add(1)

    # The below retrieves the existing prefill cache variables, not creating new ones
    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, kv_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE
    )

    cached_prefill = (
        self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order),
        self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order),
        cached_prefill_segment_id_var.value,
    )

    cached_ar = (
        self.get_cached_values(cached_ar_key_vars, key.dtype, self.ar_cache_axis_order),
        self.get_cached_values(cached_ar_value_vars, value.dtype, self.ar_cache_axis_order),
        cached_ar_segment_id_var.value,
        cache_ar_lengths_var.value,
    )
    return cached_prefill, cached_ar

  def kv_cache(
      self, key: Array, value: Array, decoder_segment_ids: Array, model_mode: str, use_ragged_attention: bool = False
  ) -> tuple:
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
        # jax.debug.print("AttentionOp.kv_cache - autoregressive - key shape: {}", key.shape)
        # jax.debug.print("AttentionOp.kv_cache - autoregressive - value shape: {}", value.shape)
        prefill_cache, ar_cache = self.kv_cache_autoregressive(key, value, use_ragged_attention)
        # jax.debug.print("AttentionOp.kv_cache - autoregressive - ar_cache[0] shape: {}", ar_cache[0].shape if ar_cache[0] is not None else None)
        # jax.debug.print("AttentionOp.kv_cache - autoregressive - ar_cache[1] shape: {}", ar_cache[1].shape if ar_cache[1] is not None else None)
        return prefill_cache, ar_cache
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
  def __call__(self, query, key, value, decoder_segment_ids, model_mode, page_state=None):
    prefill_kv_cache, ar_kv_cache = self.kv_cache(
        key, value, decoder_segment_ids, model_mode, use_ragged_attention=self.use_ragged_attention
    )

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=prefill_kv_cache[0],
        value=prefill_kv_cache[1],
        decoder_segment_ids=prefill_kv_cache[2],
        lengths=None,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
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
        lengths=ar_kv_cache[3],
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return ar_unnormalized_output / ar_exponentials_sum if ar_exponentials_sum is not None else ar_unnormalized_output


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

  # Shard the query activation as the same as the key and value.
  # TODO: Find a better sharding axis name.
  # TODO: Further break down the Training and Inference axes for the q, k, v.
  prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  input_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False

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
        kernel_axes=("embed", "q_heads", "kv"),
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
    out_proj = DenseGeneral(
        features=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj

  def apply_rotary_embedding(self, inputs: Array, inputs_positions: Array, name: str):
    if self.config.model_name.startswith("llama3.1"):
      rotary_embedding = embeddings.LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=self.head_dim,
          fprop_dtype=self.dtype,
          name=name,
      )
    else:
      rotary_embedding = embeddings.RotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=self.head_dim,
          fprop_dtype=self.dtype,
          name=name,
      )
    inputs = rotary_embedding(inputs, inputs_positions)
    return inputs

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
    page_state: Optional[Any] = None,
    slot: Optional[int] = None,
    true_length: Optional[int] = None,
  ):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
        inputs_q: input queries of shape `[batch, q_length, q_features]`.
        inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
        model_mode: corresponding to train, prefill and decode.
        deterministic: Disables dropout if set to True.
        page_state: Optional page manager state for paged attention.

    Returns:
        output of shape `[batch, length, q_features]`.
    """
    # jax.debug.print("Attention.__call__ - attention_kernel: {}, model_mode: {}", self.attention_kernel, model_mode)
    # jax.debug.print("Attention.__call__ - inputs_q shape: {}", inputs_q.shape)
    # jax.debug.print("Attention.__call__ - inputs_kv shape: {}", inputs_kv.shape)
    # jax.debug.print("Attention.__call__ - inputs_positions shape: {}", inputs_positions.shape)
    # jax.debug.print("Attention.__call__ - slot: {}", slot)

    print(f"Attention.__call__  print - attention_kernel: {self.attention_kernel}, model_mode: {model_mode}")
    print(f"Attention.__call__  print- inputs_q shape: {inputs_q.shape}")
    print(f"Attention.__call__  print- inputs_kv shape: {inputs_kv.shape}")
    print(f"Attention.__call__  print- inputs_positions shape: {inputs_positions.shape}")
    print(f"Attention.__call__  print- slot: {slot}")

    # 1. Initial validation
    if decoder_segment_ids is not None and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            f"During autoregressive decoding we assume tokens are in active sequence"
            f" which is always {common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR}."
        )

    # 2. Input constraints and partitioning
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    # 3. Query/Key/Value projections
    if self.config.fused_qkv:
        query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
        query = self.query_projection(inputs_q)
        key = self.kv_projection(inputs_kv, proj_name="key")
        value = self.kv_projection(inputs_kv, proj_name="value")

    # 4. Apply ROPE embeddings
    query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
    key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")

    # 5. Apply appropriate constraints based on model mode
    if model_mode == common_types.MODEL_MODE_PREFILL:
        query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
        key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
        value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    else:
        query = nn.with_logical_constraint(query, self.query_axis_names)
        key = nn.with_logical_constraint(key, self.key_axis_names)
        value = nn.with_logical_constraint(value, self.value_axis_names)

    # 6. Checkpoint naming for XLA optimization
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    assert not self.config.quantize_kvcache or self.kv_quant

    # Select and apply appropriate attention implementation
    if self.attention_kernel == "paged" and model_mode != common_types.MODEL_MODE_TRAIN:
        print(f"About to create PagedAttentionOp w/ {slot=}, {true_length=}, {model_mode=}, {self.attention_kernel=}")
        attention_op = PagedAttentionOp(
            mesh=self.mesh,
            num_pages=self.config.num_pages,
            tokens_per_page=self.config.tokens_per_page,
            max_pages_per_slot=self.config.max_target_length // self.config.tokens_per_page,
            max_pages_per_prefill=self.config.max_prefill_predict_length // self.config.tokens_per_page,
            pages_per_compute_block=self.config.pages_per_compute_block,
            num_kv_heads=self.num_kv_heads,
            kv_head_dim_size=self.head_dim,
            config=self.config,
            dtype=self.dtype,
        )
        print(f"About to call PagedAttentionOp w/ {slot=}, {true_length=}, {model_mode=}, {self.attention_kernel=}")
        return attention_op(
            query=query,  # Note: Using processed query/key/value
            key=key,
            value=value,
            decoder_segment_ids=decoder_segment_ids,
            model_mode=model_mode,
            slot=slot,
            true_length=true_length,
        )
    else:
        # Standard attention path
        attention_op = AttentionOp(
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
            prefill_cache_axis_order=self.prefill_cache_axis_order,
            ar_cache_axis_order=self.ar_cache_axis_order,
            compute_axis_order=self.compute_axis_order,
            reshape_q=self.reshape_q,
            attention_type=self.attention_type,
            attn_logits_soft_cap=self.attn_logits_soft_cap,
            sliding_window_size=self.sliding_window_size,
            use_ragged_attention=self.use_ragged_attention,
            ragged_block_size=self.ragged_block_size,
        )
        attention_output = attention_op(query, key, value, decoder_segment_ids, model_mode)

    # 8. Process attention output based on mode
    if self.attention_kernel == "paged" and model_mode != common_types.MODEL_MODE_TRAIN and true_length is not None and slot is not None:
        unnormalized_out, _, exp_sum = attention_output
        out = unnormalized_out / (exp_sum + 1e-9) if exp_sum is not None else unnormalized_out
    else:
        out = attention_output

    # 9. Apply output constraints and projection
    out = nn.with_logical_constraint(out, self.out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, "out_proj")

    return out