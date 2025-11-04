
from typing import Callable, List

import jax.numpy as jnp
from jax import Array


def and_masks(*mask_functions: List[Callable]) -> Callable:
  """Returns a mask function that is the intersection of provided mask functions."""
  if not all(callable(arg) for arg in mask_functions):
    raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

  def and_mask(batch_idx: Array, head_idx: Array, q_idx: Array, kv_idx: Array) -> Array:
    result = jnp.ones((), dtype=jnp.bool_)
    for mask in mask_functions:
      result = jnp.logical_and(result, mask(batch_idx, head_idx, q_idx, kv_idx))
    return result

  return and_mask
def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
  """This creates a basic lower-diagonal causal mask."""
  return kv_idx <= q_idx
from typing import Callable, List

import jax.numpy as jnp


def or_masks(*mask_functions: List[Callable]) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx: jnp.ndarray, head_idx: jnp.ndarray, q_idx: jnp.ndarray, kv_idx: jnp.ndarray) -> jnp.ndarray:
        result = jnp.zeros((), dtype=jnp.bool_)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return or_mask

from typing import Callable

import jax.numpy as jnp


def packed_sequence_mask_function(packed_sequence_mask: jnp.ndarray) -> Callable:
  """
  This return the mask_function function corresponding to a 2D packed sequence mask.
  """

  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

  return inner_mask

from typing import Optional, TypedDict

import jax


class FlashAttentionKwargs(TypedDict, total=False):
  """Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`jax.Array`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`jax.Array`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

  cumulative_seqlens_q: Optional[jax.Array]
  cumulative_seqlens_k: Optional[jax.Array]
  max_length_q: Optional[int]
  max_length_k: Optional[int]

# Copyright 2025 The T5X Authors.
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
    A callable function that computes the sliding window mask.
  """

  def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    return kv_idx > q_idx - sliding_window

  return inner_mask

from typing import Callable

# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.and_masks
from generated_code.Qwen3MoeForCausalLM.attention_utils import and_masks
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.causal_mask_function
from generated_code.Qwen3MoeForCausalLM.attention_utils import causal_mask_function
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.sliding_window_overlay
from generated_code.Qwen3MoeForCausalLM.attention_utils import sliding_window_overlay


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
  """
    This return the mask_function function to create a sliding window mask.
    """
  return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)

from typing import Callable, Optional

# In JAX, using global variables to cache functions is generally discouraged as it can
# lead to unexpected behavior with JIT compilation, especially if the function value
# changes. State, including function choices, should be handled explicitly within modules.
# This functionality is replaced by the dispatch logic in `src/MaxText/layers/attention_op.py`.
_flash_fn: Optional[Callable] = None
_flash_varlen_fn: Optional[Callable] = None
_pad_fn: Optional[Callable] = None
_unpad_fn: Optional[Callable] = None
_process_flash_kwargs_fn: Optional[Callable] = None

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.typing import Array


def _index_first_axis(tensor: Array, indices: Array) -> Array:
  """
  A local implementation of the PyTorch indexing operation `tensor[indices]`
  on the first axis, after flattening the first two dimensions of the tensor.
  This is functionally equivalent to FA2's `index_first_axis`.
  """
  # The input tensor is expected to be of shape (batch, seq_len, ...). We
  # flatten the first two dimensions to get (total_tokens, ...) before
  # indexing.
  reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
  return reshaped_tensor[indices]


def _unpad_input(
    hidden_states: Array,
    attention_mask: Array,
    unused_mask: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
  """
  unpad_input function for flash attention variants that do not have them
  within their pkg themselves, e.g. fa3.

  Args:
      hidden_states: (batch, seqlen, ...)
      attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means
        not valid.
      unused_mask: (batch, seqlen), bool / int, 1 means the element is
        allocated but unused.

  Returns:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens
        selected in attention_mask + unused_mask.
      indices: (total_nnz), the indices of masked tokens from the flattened
        input sequence.
      cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index
        into hidden_states.
      max_seqlen_in_batch: 0-dim array, max sequence length in the batch.
      seqused: (batch), returns the number of tokens selected in
        attention_mask + unused_mask.
  """
  if unused_mask is not None:
    all_masks = attention_mask + unused_mask
  else:
    all_masks = attention_mask
  seqlens_in_batch = all_masks.sum(axis=-1, dtype=jnp.int32)
  used_seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
  indices = jnp.nonzero(all_masks.flatten(), size=all_masks.size)[0]
  max_seqlen_in_batch = seqlens_in_batch.max()
  cu_seqlens = jnp.pad(
      jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),)
  )

  return (
      _index_first_axis(hidden_states, indices),
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
      used_seqlens_in_batch,
  )


def _pad_input(
    hidden_states: Array, indices: Array, batch: int, seqlen: int
) -> Array:
  """
  pad_input function for flash attention variants that do not have them
  within their pkg themselves, e.g. fa3.

  Args:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens in
        selected in attention_mask.
      indices: (total_nnz), the indices that represent the non-masked tokens
        of the original padded input sequence.
      batch: int, batch size for the padded sequence.
      seqlen: int, maximum sequence length for the padded sequence.

  Returns:
      hidden_states: (batch, seqlen, ...)
  """
  dim = hidden_states.shape[1:]
  output = jnp.zeros(
      (batch * seqlen, *dim), dtype=hidden_states.dtype
  ).at[indices].set(hidden_states)
  return output.reshape(batch, seqlen, *dim)


def _get_unpad_data(attention_mask: Array) -> Tuple[Array, Array, Array]:
  """
  Retrieves indexing data required to repad unpadded (ragged) tensors.

  Args:
      attention_mask (`jax.Array`):
          Boolean or int tensor of shape (batch_size, sequence_length), 1 means
          valid and 0 means not valid.

  Returns:
      indices (`jax.Array`):
          The indices of non-masked tokens from the flattened input sequence.
      cu_seqlens (`jax.Array`):
          The cumulative sequence lengths, used to index into ragged (unpadded)
          tensors. `cu_seqlens` shape is (batch_size + 1,).
      max_seqlen_in_batch (`jax.Array`):
          0-dim array of maximum sequence length in batch.
  """
  seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
  indices = jnp.nonzero(attention_mask.flatten(), size=attention_mask.size)[0]
  # NOTE: Similar to the `.item()` in prepare_fa2_from_position_ids, with
  # JAX JIT, this would cause a graph break. Returning a 0-d array is
  # JIT-compatible.
  max_seqlen_in_batch = seqlens_in_batch.max()
  cu_seqlens = jnp.pad(
      jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),)
  )
  return (
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
  )

from typing import Any, Callable

# Global variable for the lazily-loaded flash attention padding function.
_pad_fn: Callable[..., Any] | None = None

import inspect
import os
from functools import partial
from typing import Optional, TypedDict

from jax import Array


# function that processes kwargs, generalized to handle any supported kwarg within the function
_process_flash_kwargs_fn = None
# exceptions where hf API doesn't match the original flash attention API
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}


class FlashAttentionKwargs(TypedDict, total=False):
  """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cumulative_seqlens_q (`jax.Array`, *optional*)
            Gets cumulative sequence length for query state.
        cumulative_seqlens_k (`jax.Array`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

  cumulative_seqlens_q: Optional[Array]
  cumulative_seqlens_k: Optional[Array]
  max_length_q: Optional[int]
  max_length_k: Optional[int]


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[Array] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
  """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`jax.Array`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
  flash_kwargs = {
      "causal": is_causal and not (use_top_left_mask and query_length == 1),
      "softmax_scale": softmax_scale,
  }

  if supports_mapping["dropout_p"]:
    flash_kwargs["dropout_p"] = dropout

  if (
      supports_mapping["window_size"]
      and sliding_window is not None
      and key_length > sliding_window
  ):
    flash_kwargs["window_size"] = (sliding_window, sliding_window)

  if supports_mapping["deterministic"]:
    flash_kwargs["deterministic"] = (
        deterministic
        if deterministic is not None
        else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    )

  if supports_mapping["softcap"] and softcap is not None:
    flash_kwargs["softcap"] = softcap

  # Only within kernel implementation atm
  if supports_mapping["s_aux"] and s_aux is not None:
    flash_kwargs["s_aux"] = s_aux

  return flash_kwargs


def _lazy_define_process_function(flash_function):
  """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """
  global _process_flash_kwargs_fn, _hf_api_to_flash_mapping

  flash_parameters = inspect.signature(flash_function).parameters
  process_parameters = inspect.signature(
      _process_flash_attention_kwargs
  ).parameters

  supports_mapping = {}
  for param in process_parameters:
    fa_param = _hf_api_to_flash_mapping.get(param, param)
    supports_mapping[fa_param] = fa_param in flash_parameters

  return partial(
      _process_flash_attention_kwargs, supports_mapping=supports_mapping
  )

from typing import Callable, Optional

# `globals()` is not compatible with dynamo, hence we have do define them in global scope ourselves
_flash_fn: Optional[Callable] = None
_flash_varlen_fn: Optional[Callable] = None
_pad_fn: Optional[Callable] = None
_unpad_fn: Optional[Callable] = None

from jax import Array
import jax.numpy as jnp


def repeat_kv(hidden_states: Array, n_rep: int) -> Array:
  """
  This is the equivalent of jnp.repeat(x, repeats=n_rep, axis=1). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  if n_rep == 1:
    return hidden_states
  return jnp.repeat(hidden_states, repeats=n_rep, axis=1)

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.linen import nn
from jax import Array

# MaxText-matched dependency:
# generated_code.Qwen3MoeForCausalLM.attention_utils.repeat_kv
from generated_code.Qwen3MoeForCausalLM.attention_utils import repeat_kv


def eager_attention_forward(
    query: Array,
    key: Array,
    value: Array,
    num_key_value_groups: int,
    attention_mask: Optional[Array],
    scaling: float,
    dropout: float = 0.0,
    deterministic: bool = True,
    dropout_rng: Optional[jax.Array] = None,
    **kwargs,
) -> Tuple[Array, Array]:
    """Eager attention forward pass."""
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = jnp.matmul(query, jnp.swapaxes(key_states, -2, -1)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[..., : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(query.dtype)

    if dropout > 0.0 and not deterministic:
        if dropout_rng is None:
            raise ValueError("A 'dropout'-named RNG must be provided when applying dropout.")
        attn_weights = nn.Dropout(rate=dropout)(attn_weights, deterministic=False, rngs={"dropout": dropout_rng})

    attn_output = jnp.matmul(attn_weights, value_states)
    attn_output = jnp.swapaxes(attn_output, 1, 2)

    return attn_output, attn_weights

from typing import Optional, Tuple, Any

import jax.numpy as jnp
from jax import Array

from maxtext.utils import logging
from maxtext.common_types import MODEL_MODE_PREFILL


logger = logging.get_logger(__name__)


def flash_attention_forward(
    module: Any,
    query: Array,
    key: Array,
    value: Array,
    attention_mask: Optional[Array],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[Array, None]:
  """
  JAX implementation of flash_attention_forward.

  This function is an adapter to call a statically configured MaxText `AttentionOp`
  module. Unlike the PyTorch version, arguments like `scaling`, `sliding_window`,
  `softcap`, and the causal nature of the attention are expected to be configured
  during the initialization of `module.attention_op` and cannot be changed dynamically
  at runtime.

  Args:
    module: The Flax/NNX module instance calling this function. It is expected
      to have an `attention_op` attribute.
    query: The query tensor.
    key: The key tensor.
    value: The value tensor.
    attention_mask: An optional attention mask. This implementation assumes it's a
      2D padding mask of shape [batch, seq_len].
    dropout: Dropout rate. Not used in this JAX implementation as dropout is
      typically handled at a higher level in the transformer block.
    scaling: Softmax scaling factor. Not used in this JAX implementation as it's
      configured statically in the `AttentionOp` module.
    sliding_window: Sliding window size. Not used in this JAX implementation as
      it's configured statically in the `AttentionOp` module.
    softcap: Softcap value for attention logits. Not used in this JAX
      implementation as it's configured statically in the `AttentionOp` module.
    **kwargs: Additional keyword arguments.

  Returns:
    A tuple containing the attention output and None.
  """
  if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
    logger.warning_once(
        "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
        " Please set your attention to `eager` if you want any of these features."
    )

  if any(dim == 0 for dim in query.shape):
    raise ValueError(
        f"Tensor query has shape {query.shape} with a zero dimension.\n"
        "FlashAttention does not support inputs with dim=0.\n"
        "Please check your input shapes or use SDPA instead."
    )
  # MaxText's AttentionOp expects the format [batch, seq_len, num_heads, head_dim].
  # The PyTorch code transposes from [b, h, s, d] to [b, s, h, d].
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))

  # The complex dtype casting logic from PyTorch is omitted.
  # In JAX, dtypes are typically handled by the model's configuration (`config.dtype`)
  # and are expected to be consistent.

  # The `is_causal` flag is handled by the static configuration of the AttentionOp
  # module (e.g., `attention_type` in the config). It cannot be dynamically controlled here.
  # We pop it from kwargs to avoid it being passed down and causing an error.
  _ = kwargs.pop("is_causal", None)

  # The `attention_mask` is interpreted as segment IDs for packing/padding.
  # `AttentionOp` will combine this with its internally generated causal mask if configured to do so.
  decoder_segment_ids = None
  if attention_mask is not None:
    if attention_mask.ndim != 2:
      raise ValueError(
          "This simplified JAX adapter only supports 2D attention masks (padding masks)."
          f" Received mask with shape {attention_mask.shape}."
      )
    decoder_segment_ids = attention_mask.astype(jnp.int32)

  # Call the AttentionOp module, assuming a 'prefill' model_mode for a single forward pass.
  # The `module` is expected to have an `attention_op` attribute initialized in its `setup` method.
  # The `AttentionOp` module is a highly configurable module that encapsulates various attention mechanisms.
  # Used module: src.MaxText.layers.attention_op.AttentionOp
  attn_output = module.attention_op(
      query,
      key,
      value,
      decoder_segment_ids=decoder_segment_ids,
      model_mode=MODEL_MODE_PREFILL,
  )

  return attn_output, None

import os
from typing import Dict, Optional

from jax import Array


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[Array] = None,
    supports_mapping: Optional[Dict[str, bool]] = None,
    **kwargs,
):
  """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`jax.Array`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
  flash_kwargs = {
      "causal": is_causal and not (use_top_left_mask and query_length == 1),
      "softmax_scale": softmax_scale,
  }

  if supports_mapping["dropout_p"]:
    flash_kwargs["dropout_p"] = dropout

  if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
    flash_kwargs["window_size"] = (sliding_window, sliding_window)

  if supports_mapping["deterministic"]:
    flash_kwargs["deterministic"] = (
        deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
    )

  if supports_mapping["softcap"] and softcap is not None:
    flash_kwargs["softcap"] = softcap

  # Only within kernel implementation atm
  if supports_mapping["s_aux"] and s_aux is not None:
    flash_kwargs["s_aux"] = s_aux

  return flash_kwargs

# Copyright 2025 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Flash Attention utilities for JAX.
"""

import inspect
from functools import partial
from typing import Callable

from .attention_utils import _hf_api_to_flash_mapping, _process_flash_attention_kwargs


def _lazy_define_process_function(flash_function: Callable) -> partial:
  """
    Depending on the version and kernel some features are not supported. We opt to
    statically type which (optional) kwarg parameters are supported within
    `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """
  flash_parameters = inspect.signature(flash_function).parameters
  process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

  supports_mapping = {}
  for param in process_parameters:
    fa_param = _hf_api_to_flash_mapping.get(param, param)
    supports_mapping[fa_param] = fa_param in flash_parameters

  return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)

import jax.numpy as jnp
from jax import Array


def _pad_input(hidden_states: Array, indices: Array, batch: int, seqlen: int) -> Array:
  """pad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

  Arguments:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
      indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
      batch: int, batch size for the padded sequence.
      seqlen: int, maximum sequence length for the padded sequence.

  Return:
      hidden_states: (batch, seqlen, ...)
  """
  dim = hidden_states.shape[1:]
  output = jnp.zeros((batch * seqlen, *dim), dtype=hidden_states.dtype)
  output = output.at[indices].set(hidden_states)
  return output.reshape(batch, seqlen, *dim)

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
JAX flash attention module.
"""
from typing import Optional

import jax.numpy as jnp


def _is_packed_sequence(position_ids: Optional[jnp.ndarray], batch_size: int) -> bool:
  """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (jnp.diff(position_ids, axis=-1) >= 0).all()`,
           i.e. we have multiple increasing sequences
    """
  if position_ids is None:
    return False

  increasing_position_sequences = jnp.arange(position_ids.shape[1], dtype=position_ids.dtype) + jnp.min(position_ids)
  # The original PyTorch code `... .sum().bool()` checks if the sum is non-zero.
  # A non-zero sum indicates that the position_ids are not a single monotonically
  # increasing sequence, which implies a packed sequence.
  is_not_monotonic = jnp.sum(jnp.abs(increasing_position_sequences - position_ids)) != 0
  return (batch_size == 1) and is_not_monotonic


# Copyright 2024 Google LLC
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

"""A JAX implementation of the flash attention utility functions."""

from typing import Optional
from jax import Array
import jax.numpy as jnp

from MaxText import max_logging


def fa_peft_integration_check(
    q: Array,
    k: Array,
    v: Array,
    target_dtype: Optional[jnp.dtype] = None,
) -> tuple[Array, Array, Array]:
  """Checks for PEFT integration casting and corrects dtype if necessary.

  PEFT usually casts the layer norms in float32 for training stability reasons
  therefore the input hidden states gets silently casted in float32. Hence, we need
  cast them back in float16 / bfloat16 just to be sure everything works as expected.
  This might slowdown training & inference so it is recommended to not cast the LayerNorms!

  Args:
    q: The query tensor.
    k: The key tensor.
    v: The value tensor.
    target_dtype: The target dtype for flash-attn compatibility.

  Returns:
    A tuple of (q, k, v) with potentially corrected dtypes.
  """
  if target_dtype and q.dtype == jnp.float32:
    # warning_once is not a standard feature, using a regular warning.
    max_logging.warning(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
    q = q.astype(target_dtype)
    k = k.astype(target_dtype)
    v = v.astype(target_dtype)
  return q, k, v

from typing import Optional, Tuple
import jax
import jax.numpy as jnp


# This is a local helper function from the original file.
def _index_first_axis(tensor: jax.Array, indices: jax.Array) -> jax.Array:
  """A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
  after flattening the first two dimensions of the tensor. This is functionally equivalent to
  FA2's `index_first_axis` and replaces the need to import it.
  """
  # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
  # two dimensions to get (total_tokens, ...) before indexing.
  reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
  return reshaped_tensor[indices]


def _unpad_input(
    hidden_states: jax.Array,
    attention_mask: jax.Array,
    unused_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """unpad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

  Arguments:
      hidden_states: (batch, seqlen, ...)
      attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
      unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

  Return:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
      indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
      cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
      max_seqlen_in_batch: int
      seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
  """
  all_masks = (
      (attention_mask + unused_mask)
      if unused_mask is not None
      else attention_mask
  )
  seqlens_in_batch = all_masks.sum(axis=-1, dtype=jnp.int32)
  used_seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
  indices = jnp.where(all_masks.flatten())[0]
  max_seqlen_in_batch = seqlens_in_batch.max()
  cu_seqlens = jnp.pad(
      jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),)
  )

  return (
      _index_first_axis(hidden_states, indices),
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
      used_seqlens_in_batch,
  )

from typing import Optional, Callable, Tuple

import jax

# It's idiomatic in JAX to select implementations based on configuration and hardware backend
# rather than dynamic imports based on package availability. This function reflects that pattern.

# Try to import JAX's TPU flash attention kernels. These are hardware-specific.
# A real-world implementation might have similar checks for GPU kernels.
try:
  from jax.experimental.pallas.ops.tpu.flash_attention import (
      flash_attention as tpu_flash_attention,
      mha_varlen_reference as tpu_flash_attention_varlen,
  )

  _TPU_FLASH_ATTN_AVAILABLE = True
except ImportError:
  _TPU_FLASH_ATTN_AVAILABLE = False
  tpu_flash_attention = None
  tpu_flash_attention_varlen = None

# Re-use existing JAX modules for padding utilities.
# from generated_code.Qwen3MoeForCausalLM.attention_utils._get_unpad_data
from .attention_utils import _pad_input, _unpad_input


def _lazy_imports(
    implementation: Optional[str],
) -> Tuple[Callable, Callable, Callable, Callable]:
  """
    Selects the respective JAX flash attention implementations based on configuration.

    Return:
        flash_attn_func: The base flash attention function.
        flash_attn_varlen_func: The flash attention function supporting variable sequence lengths,
                                e.g. for padding-free training.
        pad_input: The function to pad inputs into one sequence and returning the respective kwargs.
        unpad_input: The function to unpad outputs based on the kwargs (from pad_input).
    """
  pad_input, unpad_input = _pad_input, _unpad_input

  flash_attn_func = None
  flash_attn_varlen_func = None

  backend = jax.default_backend()

  # In JAX, the specific flash attention implementation is tied to the hardware backend.
  # This logic checks for available kernels on the current backend.
  is_flash_attn_available_on_backend = False
  if backend == "tpu" and _TPU_FLASH_ATTN_AVAILABLE:
    is_flash_attn_available_on_backend = True
    # NOTE: The signatures of JAX Pallas kernels differ from the PyTorch flash-attn library.
    # The caller of these functions would need to be adapted to handle the JAX-specific signatures
    # and lack of features like dropout.
    flash_attn_func = tpu_flash_attention
    flash_attn_varlen_func = tpu_flash_attention_varlen
  # TODO: Add elif for 'gpu' backend if a standard JAX GPU flash attention kernel is available.

  # This logic simplifies the PyTorch version's fa2/fa3 checks into a single
  # "is flash available on this backend" check.
  if implementation in ["flash_attention_2", "flash_attention_3"] or implementation is None:
    if not is_flash_attn_available_on_backend:
      raise ValueError(
          f"Flash attention implementation '{implementation}' requested, but no compatible JAX kernel "
          f"is available for the current backend '{backend}'."
      )
  else:
    # This handles the "Kernels fallback" case from PyTorch.
    # Since we can't dynamically load from an object path in a JIT-compatible way,
    # we treat any other implementation string as an error.
    raise ValueError(
        f"Could not find the currently requested flash attention implementation at `{implementation}`."
        f"Make sure that you request a valid kernel, e.g., `flash_attention_2`."
    )

  return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input

from typing import Optional, Tuple, Callable

# Reused: generated_code.Qwen3MoeForCausalLM.attention_utils._lazy_imports
from .attention_utils import _lazy_imports
# Reused: generated_code.Qwen3MoeForCausalLM.attention_utils._lazy_define_process_function
from .attention_utils import _lazy_define_process_function


# `globals()` is not compatible with JAX, hence we have do define them in global scope ourselves
_flash_fn = None
_flash_varlen_fn = None
_pad_fn = None
_unpad_fn = None

# function that processes kwargs, generalized to handle any supported kwarg within the function
_process_flash_kwargs_fn = None


def lazy_import_flash_attention(
    implementation: Optional[str],
) -> Tuple[Tuple[Callable, Callable, Callable, Callable], Callable]:
  """
    Lazy loading flash attention and returning the respective functions + flags back

    NOTE: For fullgraph, this needs to be called before compile while no fullgraph can
          can work without preloading. See `_check_and_adjust_attn_implementation` in `modeling_utils`.
    """
  global _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn
  if any(k is None for k in [_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn]):
    _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn = _lazy_imports(
        implementation
    )

  global _process_flash_kwargs_fn
  if _process_flash_kwargs_fn is None:
    _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn)

  return (_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn), _process_flash_kwargs_fn

import jax

def flash_attn_supports_top_left_mask() -> bool:
  """Determines if the available flash attention implementation supports top-left masking.

  This is a legacy behavior for older versions of flash-attn. JAX flash attention
  implementations are modern and do not support/require this, so this function
  will return False if a JAX flash attention kernel is found.

  Returns:
      A boolean indicating support for top-left masking. In JAX, this is always False.
  """
  # In JAX, there is no Flash Attention 3 equivalent.
  is_fa3_available = False
  if is_fa3_available:
    return False

  # Check for JAX Pallas Flash Attention (analogous to FA2).
  is_fa2_available = False
  if jax.default_backend() == "tpu":
    try:
      from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
      is_fa2_available = True
    except ImportError:
      pass
  # TODO: Add similar check for GPU flash attention if/when available.

  if is_fa2_available:
    # The JAX Pallas kernel is a modern implementation, analogous to flash-attn >= 2.1.0.
    # Therefore, it does not support/require the top-left mask.
    is_modern_fa2 = True
    return not is_modern_fa2

  # The NPU part from the original PyTorch code is not applicable to JAX.
  # If no flash attention is available, the concept is not applicable.
  return False

# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.flash_attn_supports_top_left_mask
from ..modeling_flash_attention_utils import flash_attn_supports_top_left_mask


_use_top_left_mask = flash_attn_supports_top_left_mask()

# Copyright 2025 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""
This file contains utility functions for preparing and handling arguments for Flash Attention,
a memory-efficient attention mechanism. It includes functions for converting between padded
and unpadded tensor representations, which is crucial for variable-length sequence processing
in models like those using Flash Attention. The utilities are designed to be compatible with
JAX and its compilation requirements, providing JAX-native implementations for operations
that might be built-in to other frameworks' Flash Attention libraries.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


def prepare_fa_kwargs_from_position_ids(
    position_ids: jax.Array, is_packed_sequence: bool = True
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[int, int]]:
    """
    This function returns all the necessary kwargs to call `flash_attn_varlen_func`
    extracted from position_ids. The `position_ids` can be either packed sequence or
    the usual padded position ids, for example in inference time.

    Arguments:
        position_ids (`jax.Array`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        is_packed_sequence (`bool`, *optional*, defaults to `True`):
            Whether the input position ids are a packed sequence or not.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[jax.Array]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    # If the lengths are not equal, most probably we are in decoding stage with cache
    # In that case the position ids will not always start with `0` and we need a better way to infer
    # cumulative seq lengths.
    if not is_packed_sequence:
        last_position_ids = position_ids[:, -1]
        if position_ids.shape[-1] == 1:
            q_len = jnp.ones(position_ids.shape[0], dtype=jnp.int32)
        else:
            q_len = last_position_ids + 1

        cu_seq_lens_q = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), q_len.cumsum(0).astype(jnp.int32)], axis=0
        )
        cu_seq_lens_k = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), (last_position_ids + 1).cumsum(0).astype(jnp.int32)], axis=0
        )

        max_length_q = int(q_len.max())
        max_length_k = int(last_position_ids.max()) + 1
    else:
        position_ids = position_ids.flatten()
        indices_q = jnp.where(position_ids == 0)[0]

        cu_seq_lens_q = jnp.concatenate(
            (
                indices_q,
                jnp.array([position_ids.size], dtype=jnp.int32),
            ),
            axis=0,
        )
        cu_seq_lens_k = cu_seq_lens_q

        # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
        # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
        # for some models (e.g. qwen2-vl).
        max_length_q = jnp.diff(cu_seq_lens_q).max()
        # NOTE: In JAX, this will cause a compilation error if `max_length_q` is not a concrete value
        # during JIT compilation. This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
        # requires `max_length_q`, `max_length_k` to be passed as `int` and not `jax.Array`.
        max_length_q = int(max_length_q)
        max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)

from typing import Tuple
from jax import Array

# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.prepare_fa_kwargs_from_position_ids
from generated_code.Qwen3MoeForCausalLM.attention_utils import prepare_fa_kwargs_from_position_ids


def _prepare_from_posids(
    query: Array,
    key: Array,
    value: Array,
    position_ids: Array,
    query_length: int,
) -> Tuple[Array, Array, Array, Tuple[Array, Array], Tuple[int, int]]:
  """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cumulative lengths of each examples in the batch will be extracted from position_ids.
    NOTE: ideally cumulative lengths should be prepared at the data collator stage

    Args:
        query (`jax.Array`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`jax.Array`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`jax.Array`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`jax.Array`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Sequence length of the input queries.

    Return:
        query (`jax.Array`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`jax.Array`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`jax.Array`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        (cu_seqlens_q, cu_seqlens_k) (`tuple[Array]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
  kv_length = key.shape[1]
  is_packed_sequence = query_length == kv_length

  query = query.reshape(-1, query.shape[-2], query.shape[-1])
  key = key.reshape(-1, key.shape[-2], key.shape[-1])
  value = value.reshape(-1, value.shape[-2], value.shape[-1])

  (cu_seq_lens_q, cu_seq_lens_k), (
      max_length_q,
      max_length_k,
  ) = prepare_fa_kwargs_from_position_ids(
      position_ids, is_packed_sequence=is_packed_sequence
  )

  return (
      query,
      key,
      value,
      (cu_seq_lens_q, cu_seq_lens_k),
      (max_length_q, max_length_k),
  )

from typing import Tuple
import jax.numpy as jnp
from jax import Array


def _get_unpad_data(attention_mask: Array) -> Tuple[Array, Array, Array]:
  """Retrieves indexing data required to repad unpadded (ragged) tensors.

  Arguments:
      attention_mask (`jax.Array`):
          Boolean or int tensor of shape (batch_size, sequence_length), 1 means
          valid and 0 means not valid.

  Return:
      indices (`jax.Array`):
          The indices of non-masked tokens from the flattened input sequence.
      cu_seqlens (`jax.Array`):
          The cumulative sequence lengths, used to index into ragged (unpadded)
          tensors. `cu_seqlens` shape is (batch_size + 1,).
      max_seqlen_in_batch (`jax.Array`):
          Maximum sequence length in batch as a 0-d array.
  """
  seqlens_in_batch = attention_mask.sum(axis=-1, dtype=jnp.int32)
  # In JAX, jnp.nonzero returns a tuple of arrays, one for each dimension.
  # Since we flatten the mask, we get a tuple with one array.
  # We also need to provide the `size` argument for JIT compatibility.
  indices = jnp.nonzero(attention_mask.flatten(), size=attention_mask.size)[0]
  # NOTE: .item() is removed to keep the value as a JAX array for JIT
  # compatibility.
  max_seqlen_in_batch = seqlens_in_batch.max()
  cu_seqlens = jnp.pad(
      jnp.cumsum(seqlens_in_batch, axis=0, dtype=jnp.int32), ((1, 0),)
  )
  return (
      indices,
      cu_seqlens,
      max_seqlen_in_batch,
  )

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils._get_unpad_data
from generated_code.Qwen3MoeForCausalLM.attention_utils import _get_unpad_data
# Reused from generated_code.Qwen3MoeForCausalLM.tensor_utils._index_first_axis
from generated_code.Qwen3MoeForCausalLM.tensor_utils import _index_first_axis


def _upad_input(
    query_layer: jax.Array,
    key_layer: jax.Array,
    value_layer: jax.Array,
    attention_mask: jax.Array,
    query_length: int,
    unpad_input_func: Callable[
        [jax.Array, jax.Array],
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ],
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Tuple[jax.Array, jax.Array],
    Tuple[int, int],
]:
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.
    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`jax.Array`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`jax.Array`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`jax.Array`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`jax.Array`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        unpad_input_func:
            The function to use for unpadding the input tensors.

    Return:
        query_layer (`jax.Array`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`jax.Array`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`jax.Array`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`jax.Array`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[jax.Array]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    max_seqlen_in_batch_k = int(max_seqlen_in_batch_k)

    # With static caches, the k/v states may be larger than the mask -> we need to slice them to avoid generating garbage
    # It's a bit of an anti-pattern, but otherwise we silently compute wrong attentions scores
    seq_len = attention_mask.shape[-1]
    if key_layer.shape[1] > seq_len:
        key_layer, value_layer = (
            key_layer[:, :seq_len, :, :],
            value_layer[:, :seq_len, :, :],
        )

    batch_size, kv_seq_len, _, _ = key_layer.shape

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)
    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = jnp.arange(batch_size + 1, dtype=jnp.int32)
        indices_q = cu_seqlens_q[:-1]
        query_layer = jnp.squeeze(query_layer, axis=1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        (
            query_layer,
            indices_q,
            cu_seqlens_q,
            max_seqlen_in_batch_q,
            *_,
        ) = unpad_input_func(query_layer, attention_mask)
        max_seqlen_in_batch_q = int(max_seqlen_in_batch_q)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

from typing import Optional

import jax
import jax.numpy as jnp

# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.lazy_import_flash_attention
from generated_code.Qwen3MoeForCausalLM.attention_utils import lazy_import_flash_attention
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.fa_peft_integration_check
from generated_code.Qwen3MoeForCausalLM.attention_utils import fa_peft_integration_check
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils._is_packed_sequence
from generated_code.Qwen3MoeForCausalLM.attention_utils import _is_packed_sequence
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils._upad_input
from generated_code.Qwen3MoeForCausalLM.attention_utils import _upad_input
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils._prepare_from_posids
from generated_code.Qwen3MoeForCausalLM.attention_utils import _prepare_from_posids


def _flash_attention_forward(
    query_states: jax.Array,
    key_states: jax.Array,
    value_states: jax.Array,
    attention_mask: Optional[jax.Array],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[jax.Array] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[jax.Array] = None,
    cu_seq_lens_k: Optional[jax.Array] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[jnp.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
) -> jax.Array:
  """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`jax.Array`):
            Input query states to be passed to Flash Attention API
        key_states (`jax.Array`):
            Input key states to be passed to Flash Attention API
        value_states (`jax.Array`):
            Input value states to be passed to Flash Attention API
        attention_mask (`jax.Array`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
  (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_flash_kwargs_fn = lazy_import_flash_attention(implementation)

  # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
  query_states, key_states, value_states = fa_peft_integration_check(
      query_states, key_states, value_states, target_dtype
  )

  # Extract the flash attention kwargs that have been requested (and are supported by the implementation)
  flash_kwargs = process_flash_kwargs_fn(
      query_length=query_length,
      key_length=key_states.shape[1],
      is_causal=is_causal,
      dropout=dropout,
      softmax_scale=softmax_scale,
      sliding_window=sliding_window,
      use_top_left_mask=use_top_left_mask,
      softcap=softcap,
      deterministic=deterministic,
      **kwargs,
  )

  # We will use `flash_varlen_fn` to prevent cross-example attention and also allow padding free approach under two cases:
  # Case 1. If position ids is provided and the position ids indicate packed sequences, see `_is_packed_sequence`.
  # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
  # use `flash_varlen_fn` knowing we already have all necessary the kwargs.
  #
  # NOTE: it is user's responsibility to take care of flattening `position_ids` if that's needed by the model.
  # See #39121 for more information.
  is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.shape[0])
  is_fa_with_varlen_kwargs = all(
      kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
  )

  # Contains at least one padding token in the sequence
  if attention_mask is not None:
    q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
        query_states, key_states, value_states, attention_mask, query_length, unpad_fn
    )

    out_unpad = flash_varlen_fn(
        q,
        k,
        v,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        **flash_kwargs,
    )
    if isinstance(out_unpad, tuple):
      out_unpad = out_unpad[0]

    out = pad_fn(out_unpad, indices_q, query_states.shape[0], query_length)

  # Padding free, i.e. sequences flattened into one total sequence
  elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
    if cu_seq_lens_q is None or cu_seq_lens_k is None:
      q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
          query_states, key_states, value_states, position_ids, query_length=query_length
      )
    else:
      q = query_states.reshape(-1, query_states.shape[-2], query_states.shape[-1])
      k = key_states.reshape(-1, key_states.shape[-2], key_states.shape[-1])
      v = value_states.reshape(-1, value_states.shape[-2], value_states.shape[-1])

    out = flash_varlen_fn(
        q,
        k,
        v,
        cu_seqlens_q=cu_seq_lens_q,
        cu_seqlens_k=cu_seq_lens_k,
        max_seqlen_q=max_length_q,
        max_seqlen_k=max_length_k,
        **flash_kwargs,
    )
    if isinstance(out, tuple):
      out = out[0]

    out = out.reshape(query_states.shape[0], -1, out.shape[-2], out.shape[-1])

  # No padding
  else:
    out = flash_fn(query_states, key_states, value_states, **flash_kwargs)
    if isinstance(out, tuple):
      out = out[0]

  return out

from typing import Optional

import jax
import jax.numpy as jnp
from flax.linen import Module as nn_Module

from MaxText import max_logging
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils._flash_attention_forward
from MaxText.layers.attention_utils import _flash_attention_forward
# Reused from generated_code.Qwen3MoeForCausalLM.attention_utils.flash_attn_supports_top_left_mask
from MaxText.layers.attention_utils import flash_attn_supports_top_left_mask


_use_top_left_mask = flash_attn_supports_top_left_mask()


def flash_attention_forward(
    module: nn_Module,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: Optional[jax.Array],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[jax.Array, None]:
  """Flash attention forward pass."""
  if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
    max_logging.warning(
        "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
        " Please set your attention to `eager` if you want any of these features."
    )

  # This is before the transpose
  seq_len = query.shape[2]

  if any(dim == 0 for dim in query.shape):
    raise ValueError(
        "Tensor query has shape with a zero dimension.\n"
        "FlashAttention does not support inputs with dim=0.\n"
        "Please check your input shapes or use SDPA instead."
    )
  # FA2 uses non-transposed inputs
  query = jnp.transpose(query, (0, 2, 1, 3))
  key = jnp.transpose(key, (0, 2, 1, 3))
  value = jnp.transpose(value, (0, 2, 1, 3))

  # In PEFT, usually we cast the layer norms in float32 for training stability reasons
  # therefore the input hidden states gets silently casted in float32. Hence, we need
  # cast them back in the correct dtype just to be sure everything works as expected.
  # This might slowdown training & inference so it is recommended to not cast the LayerNorms
  # in fp32. (usually our RMSNorm modules handle it correctly)
  target_dtype = None
  if query.dtype == jnp.float32:
    # Handle the case where the model is quantized
    if hasattr(module.config, "_pre_quantization_dtype"):
      target_dtype = module.config._pre_quantization_dtype
    else:
      target_dtype = module.config.dtype

  # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
  is_causal = kwargs.pop("is_causal", None)
  if is_causal is None:
    is_causal = module.is_causal

  attn_output = _flash_attention_forward(
      query_states=query,
      key_states=key,
      value_states=value,
      attention_mask=attention_mask,
      position_ids=kwargs.get("position_ids"),
      is_causal=is_causal,
      dropout=dropout,
      scaling=scaling,
      sliding_window=sliding_window,
      softcap=softcap,
      implementation=module.config._attn_implementation,
      deterministic=kwargs.get("deterministic", False),
      target_dtype=target_dtype,
      module=module,
  )

  return attn_output, None

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

# from maxtext.common_types import Config
# from maxtext.layers.caches import Cache
# from maxtext.layers.attention import find_packed_sequence_indices
# Note: The above imports are placeholders for the actual MaxText paths.


def _preprocess_mask_arguments(
    config: "Config",
    input_embeds: Array,
    attention_mask: Optional[Array],
    cache_position: Array,
    past_key_values: Optional["Cache"],
    position_ids: Optional[Array],
    layer_idx: Optional[int],
) -> Tuple[bool, Optional[Array], Optional[Array], Optional[int], Optional[int]]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

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
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`Array` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`Array`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    # If the mask is already 4D, simply return as-is (it was already prepared, or it is custom)
    if attention_mask is not None and attention_mask.ndim == 4:
        return True, attention_mask, None, None, None

    # For custom attention kernels that do not require mask creation, we don't need a mask.
    # In MaxText, this is typically handled by checking config.attention_kernel.
    # This is a placeholder for such a check.
    # In the original PyTorch code, this checks against a registry of mask-creating attention functions.
    SUPPORTED_MASK_ATTENTION_KERNELS = {"dot_product", "flash", "autoselected"}  # Example set
    if config.attention_kernel not in SUPPORTED_MASK_ATTENTION_KERNELS:
        return True, None, None, None, None

    # Potentially switch dtype for efficiency
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.astype(jnp.bool_)

    # If using a cache, it can give all information about mask sizes based on seen tokens
    if past_key_values is not None:
        # Reused: generated_code.Qwen3MoeForCausalLM.cache_utils.Cache.get_mask_sizes
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    # Otherwise, the sizes are simply the input sizes
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    # We check the position_ids for potential packed sequence format (only if the 2D attention mask is explicitly None,
    # and we don't have past_key_values, i.e. generally a training setup)
    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        # The position ids are sometimes just unsqueezed, without being expanded
        if batch_size != position_ids.shape[0]:
            position_ids = jnp.broadcast_to(position_ids, (batch_size, position_ids.shape[-1]))
        # Reused: generated_code.Qwen3MoeForCausalLM.model_utils.find_packed_sequence_indices
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset

from typing import Callable, Optional, Union
import functools
import operator

import jax
import jax.numpy as jnp

from maxtext.common_types import Array, Config
# from transformers.models.qwen2_moe.modeling_flax_qwen2_moe import FlaxQwen2MoeCache as Cache # Placeholder for Cache type
# from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig as PretrainedConfig # Placeholder for PretrainedConfig type
# from transformers.utils.import_utils import is_torch_greater_or_equal # This is a placeholder, JAX code assumes modern functionality
# from transformers.modeling_flax_outputs import FlaxBaseModelOutput # Placeholder for BlockMask type

# Assuming BlockMask is an Array for JAX
BlockMask = Array
# Assuming Cache and PretrainedConfig are compatible with MaxText's Config or a similar structure
Cache = object # Using a generic object as a placeholder
PretrainedConfig = Config # Using MaxText Config

# This is a JAX equivalent of a PyTorch utility.
# In JAX, we assume modern functionality, so version checks are simplified.
_is_torch_greater_or_equal_than_2_6 = True
# This is a JAX equivalent of a PyTorch utility.
# In JAX, we don't have XPU-specific logic.
_is_torch_xpu_available = False


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        # Use functools.reduce for a more functional approach
        return functools.reduce(
            jnp.logical_and,
            (mask(batch_idx, head_idx, q_idx, kv_idx) for mask in mask_functions),
            jnp.ones((), dtype=jnp.bool_),
        )

    return and_mask


def or_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        return functools.reduce(
            jnp.logical_or,
            (mask(batch_idx, head_idx, q_idx, kv_idx) for mask in mask_functions),
            jnp.zeros((), dtype=jnp.bool_),
        )

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    """
    This creates a basic lower-diagonal causal mask.
    """
    return kv_idx <= q_idx


def padding_mask_function(padding_mask: Array) -> Callable:
    """
    This return the mask_function function corresponding to a 2D padding mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return padding_mask[batch_idx, kv_idx]

    return inner_mask


def packed_sequence_mask_function(packed_sequence_mask: Array) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]

    return inner_mask


def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs.
    """
    # Vmap over kv_idx
    vmapped_mask = jax.vmap(mask_function, in_axes=(None, None, None, 0), out_axes=-1)
    # Vmap over q_idx
    vmapped_mask = jax.vmap(vmapped_mask, in_axes=(None, None, 0, None), out_axes=-2)
    if bh_indices:
        # Vmap over head_idx
        vmapped_mask = jax.vmap(vmapped_mask, in_axes=(None, 0, None, None), out_axes=1)
        # Vmap over batch_idx
        vmapped_mask = jax.vmap(vmapped_mask, in_axes=(0, None, None, None), out_axes=0)
    return vmapped_mask


def prepare_padding_mask(
    attention_mask: Optional[Array], kv_length: int, kv_offset: int, _slice: bool = True
) -> Optional[Array]:
    """
    From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
    according to the `kv_offset` if `_slice` is `True`.
    """
    local_padding_mask = attention_mask
    if attention_mask is not None:
        padding_length = kv_length + kv_offset - attention_mask.shape[-1]
        if padding_length > 0:
            local_padding_mask = jnp.pad(attention_mask, ((0, 0), (0, padding_length)))
        if _slice:
            mask_indices = jnp.arange(kv_length) + kv_offset
            local_padding_mask = local_padding_mask[:, mask_indices]
    return local_padding_mask


def _ignore_causal_mask_sdpa(
    padding_mask: Optional[Array],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    In JAX, we are always in a 'traced' or 'compiled' context. The original PyTorch
    function avoids this optimization when tracing to prevent hard-coding `is_causal`.
    To be safe, we will always return False, ensuring a mask is generated.
    """
    return False


def sdpa_mask(
    batch_size: int,
    cache_position: Array,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[Array] = None,
    local_size: Optional[int] = None,
    allow_is_causal_skip: bool = True,
    **kwargs,
) -> Optional[Array]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)`.
    """
    q_length = cache_position.shape[0]
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)

    if allow_is_causal_skip and _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None

    kv_arange = jnp.arange(kv_length) + kv_offset

    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = jnp.arange(batch_size)
    head_arange = jnp.arange(1)

    causal_mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    return causal_mask


def eager_mask(
    batch_size: int,
    cache_position: Array,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: Optional[Array] = None,
    dtype: jnp.dtype = jnp.float32,
    **kwargs,
) -> Array:
    """
    Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)`.
    """
    _ = kwargs.pop("allow_is_causal_skip", None)
    mask = sdpa_mask(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        **kwargs,
    )
    min_dtype = jnp.finfo(dtype).min
    mask = jnp.where(mask, jnp.array(0.0, dtype=dtype), min_dtype)
    return mask


# A simple dictionary mapping attention implementation names to mask functions.
ALL_MASK_ATTENTION_FUNCTIONS = {
    "sdpa": sdpa_mask,
    "eager": eager_mask,
}


def find_packed_sequence_indices(position_ids: Array) -> Array:
    """
    Find the indices of the sequence to which each new query token belongs.
    """
    first_dummy_value = position_ids[:, :1] - 1
    position_diff = jnp.diff(position_ids, prepend=first_dummy_value, axis=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(axis=-1)
    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PretrainedConfig,
    input_embeds: Array,
    attention_mask: Optional[Union[Array, BlockMask]],
    cache_position: Array,
    past_key_values: Optional[Cache],
    position_ids: Optional[Array],
    layer_idx: Optional[int],
) -> tuple[bool, Optional[Union[Array, BlockMask]], Optional[Array], int, int]:
    """
    Perform some common pre-processing of the mask arguments.
    """
    if isinstance(attention_mask, jax.Array) and attention_mask.ndim == 4:
        return True, attention_mask, None, None, None

    if config.attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS:
        return True, None, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.astype(jnp.bool_)

    if past_key_values is not None:
        kv_length, kv_offset = past_key_values.get_mask_sizes(cache_position, layer_idx)
    else:
        kv_length, kv_offset = input_embeds.shape[1], 0

    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = input_embeds.shape[0]
        if batch_size != position_ids.shape[0]:
            position_ids = jnp.broadcast_to(position_ids, (batch_size, position_ids.shape[-1]))
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset


def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: Array,
    attention_mask: Optional[Array],
    cache_position: Array,
    past_key_values: Optional[Cache],
    position_ids: Optional[Array] = None,
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[Array, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used.
    """
    if hasattr(past_key_values, "is_sliding"):
        # JIT-compatible way to find the first index of False
        is_sliding_bool = jnp.array(past_key_values.is_sliding, dtype=jnp.bool_)
        indices = jnp.where(jnp.logical_not(is_sliding_bool), size=1, fill_value=-1)[0]
        layer_idx = indices[0]
        # Handle case where no False is found, though original code would error
        if layer_idx == -1:
            layer_idx = 0
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config.attn_implementation]

    # In JAX, we are always in a "compilable" context.
    # The original logic was to avoid skipping mask creation when compiling.
    allow_is_causal_skip = False if past_key_values is not None else True

    if packed_sequence_mask is not None and _is_torch_greater_or_equal_than_2_6:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
    )
    return causal_mask

from typing import Callable, Optional

from jax import Array
from maxtext.common_types import Config as PretrainedConfig
from maxtext.layers.kv_cache import KVCache as Cache

# The following imports are assumed to be available from a converted `attention_utils.py`
# based on the provided `FULL_FILE_CODE`.
from maxtext.layers.attention import _preprocess_mask_arguments
from maxtext.layers.attention import and_masks
from maxtext.layers.attention import or_masks
from maxtext.layers.attention import packed_sequence_mask_function
from maxtext.layers.attention import sliding_window_causal_mask_function
from maxtext.layers.attention import ALL_MASK_ATTENTION_FUNCTIONS


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
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
        config (`PretrainedConfig`):
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
  # NOTE: The following logic is not JIT-compatible if `is_sliding` is a JAX Array.
  # It is assumed to be a static attribute of the cache.
  layer_idx = 0
  if past_key_values is not None and hasattr(past_key_values, "is_sliding"):
    if True in past_key_values.is_sliding:
      layer_idx = past_key_values.is_sliding.index(True)

  (
      early_exit,
      attention_mask,
      packed_sequence_mask,
      kv_length,
      kv_offset,
  ) = _preprocess_mask_arguments(
      config,
      input_embeds,
      attention_mask,
      cache_position,
      past_key_values,
      position_ids,
      layer_idx,
  )
  if early_exit:
    return attention_mask

  sliding_window = getattr(config, "sliding_window", None)
  if sliding_window is None:
    raise ValueError(
        "Could not find a `sliding_window` argument in the config, or it is not"
        " set"
    )

  batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
  mask_factory_function = sliding_window_causal_mask_function(sliding_window)
  # Assuming config.attn_implementation is available and maps to MaxText attention kernels
  mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config.attn_implementation]

  # Do not allow skip if we are compiling (this is to match BC)
  # In JAX, we are always in a "compiling" context.
  allow_is_causal_skip = False if past_key_values is not None else True

  # If we detected packing format
  if packed_sequence_mask is not None:
    mask_factory_function = and_masks(
        mask_factory_function,
        packed_sequence_mask_function(packed_sequence_mask),
    )
    allow_is_causal_skip = False

  # Allow slight deviations from sliding causal mask
  if or_mask_function is not None:
    mask_factory_function = or_masks(mask_factory_function, or_mask_function)
    allow_is_causal_skip = False
  if and_mask_function is not None:
    mask_factory_function = and_masks(mask_factory_function, and_mask_function)
    allow_is_causal_skip = False

  # We now create the mask
  causal_mask = mask_interface(
      batch_size=batch_size,
      cache_position=cache_position,
      kv_length=kv_length,
      kv_offset=kv_offset,
      mask_function=mask_factory_function,
      attention_mask=attention_mask,
      allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa-like masks
      local_size=sliding_window,  # Additional kwarg for sdpa-like masks
      dtype=dtype,  # Additional kwarg for eager-like masks
      config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
  )
  return causal_mask
