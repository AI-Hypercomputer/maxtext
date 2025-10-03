
# Copyright 2025 The MaxText Authors.
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

"""Attention mask utilities."""

from typing import Callable


def sliding_window_overlay(sliding_window: int) -> Callable:
  """This is an overlay depicting a sliding window pattern.

  Add it on top of a causal mask for a proper sliding
  window mask.

  Args:
    sliding_window: The size of the sliding window.

  Returns:
    A callable mask function.
  """

  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx > q_idx - sliding_window

  return inner_mask

from typing import Callable

# Re-used from Qwen3ForCausalLM.mask_utils.and_masks
from Qwen3ForCausalLM.mask_utils import and_masks
# Re-used from Qwen3ForCausalLM.attention_masks.sliding_window_overlay
from Qwen3ForCausalLM.attention_masks import sliding_window_overlay
# Re-used from Qwen3ForCausalLM.modeling_utils.causal_mask_function
from Qwen3ForCausalLM.modeling_utils import causal_mask_function


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
  """This return the mask_function function to create a sliding window mask."""
  return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array

from maxtext.layers import Config
# All the following imports are reused from the Qwen3 JAX implementation
# path: Qwen3ForCausalLM.cache.Cache
from Qwen3ForCausalLM.cache import Cache
# path: Qwen3ForCausalLM.modeling_utils._preprocess_mask_arguments
from Qwen3ForCausalLM.modeling_utils import _preprocess_mask_arguments
# path: Qwen3ForCausalLM.attention_masks.sliding_window_causal_mask_function
from Qwen3ForCausalLM.attention_masks import sliding_window_causal_mask_function
# path: Qwen3ForCausalLM.mask_utils.and_masks
from Qwen3ForCausalLM.mask_utils import and_masks
# path: Qwen3ForCausalLM.attention_utils.or_masks
from Qwen3ForCausalLM.attention_utils import or_masks
# path: Qwen3ForCausalLM.attention_utils.packed_sequence_mask_function
from Qwen3ForCausalLM.attention_utils import packed_sequence_mask_function


def create_sliding_window_causal_mask(
    config: Config,
    input_embeds: Array,
    attention_mask: Optional[Array],
    cache_position: Array,
    past_key_values: Optional[Cache],
    position_ids: Optional[Array] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Array]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`Config`):
            The model config.
        input_embeds (`Array`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`Array`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`Array`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`Array`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)

    # If we detected packing format
    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))

    # Allow slight deviations from sliding causal mask
    if or_mask_function is not None:
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
    if and_mask_function is not None:
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)

    # We now create the mask. This logic creates a float mask (0.0 for attend, -inf for mask).
    local_attention_mask = attention_mask
    if local_attention_mask is not None:
        # Pad if necessary
        padding_length = kv_length + kv_offset - local_attention_mask.shape[-1]
        if padding_length > 0:
            local_attention_mask = jnp.pad(local_attention_mask, ((0, 0), (0, padding_length)))

        def padding_mask_fn(batch_idx, head_idx, q_idx, kv_idx):
            # batch_idx and kv_idx are scalars here due to vmap
            return local_attention_mask[batch_idx, kv_idx]

        mask_factory_function = and_masks(mask_factory_function, padding_mask_fn)

    # Create a 4D boolean mask: (batch_size, 1, q_len, kv_len) using jax.vmap
    # vmap over kv_idx
    vmapped_fn_kv = jax.vmap(mask_factory_function, in_axes=(None, None, None, 0))
    # vmap over q_idx
    vmapped_fn_q_kv = jax.vmap(vmapped_fn_kv, in_axes=(None, None, 0, None))
    # vmap over batch_idx
    vmapped_fn_b_q_kv = jax.vmap(vmapped_fn_q_kv, in_axes=(0, None, None, None))

    batch_arange = jnp.arange(batch_size)
    kv_arange = jnp.arange(kv_length) + kv_offset

    # This produces (batch_size, q_len, kv_len)
    causal_mask = vmapped_fn_b_q_kv(batch_arange, 0, cache_position, kv_arange)

    # Add head dimension -> (batch_size, 1, q_len, kv_len)
    causal_mask = causal_mask[:, None, :, :]

    # Convert boolean mask to float mask (0.0 for attend, -inf for mask)
    min_dtype = jnp.finfo(dtype).min
    causal_mask = jnp.where(causal_mask, jnp.array(0.0, dtype=dtype), min_dtype)

    return causal_mask
