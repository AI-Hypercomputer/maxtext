

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

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jax


@dataclass
class MoeCausalLMOutputWithPast:
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`jax.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`jax.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`jax.Array`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_logits (`tuple(jax.Array)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`Any`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a `cache_utils.Cache` instance. For more details, see the documentation for the specific cache implementation.

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(jax.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jax.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jax.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[jax.Array] = None
    aux_loss: Optional[jax.Array] = None
    logits: Optional[jax.Array] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple[jax.Array, ...]] = None
    attentions: Optional[Tuple[jax.Array, ...]] = None
    router_logits: Optional[Tuple[jax.Array, ...]] = None

from flax.struct import dataclass
from typing import Optional, Tuple

import jax

# Assuming 'Cache' is defined in a similar location in the JAX project.
# from .cache_utils import Cache
# Assuming 'ModelOutput' is defined in a similar location in the JAX project.
# from .utils import ModelOutput
from transformers.modeling_flax_outputs import FlaxModelOutput
from transformers.utils.import_utils import ENV_VARS_TRUE_VALUES


if ENV_VARS_TRUE_VALUES:
    from transformers.models.gemma.modeling_flax_gemma import GemmaCache as Cache


@dataclass
class MoeModelOutputWithPast(FlaxModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`jax.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(jax.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jax.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(jax.Array)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(jax.Array)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `jax.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """

    last_hidden_state: Optional[jax.Array] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[jax.Array, ...]] = None
    attentions: Optional[Tuple[jax.Array, ...]] = None
    router_logits: Optional[Tuple[jax.Array, ...]] = None

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from typing import Any

DType = Any


class Qwen3MoeRMSNorm(nn.Module):
  """
  Qwen3MoeRMSNorm is equivalent to T5LayerNorm. This implementation is functionally
  equivalent to src.MaxText.layers.normalizations.RMSNorm.
  """

  hidden_size: int
  eps: float = 1e-6
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
    """Applies RMS normalization to the input tensor."""
    weight = self.param(
        "weight",
        initializers.ones,
        (self.hidden_size,),
        self.weight_dtype,
    )
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.power(hidden_states, 2.0), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

    return (weight * hidden_states).astype(input_dtype)

# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from typing import Tuple

from flax import linen as nn
import jax.numpy as jnp

from ...common_types import Array
from .configuration_qwen3_moe import Qwen3MoeConfig
# from generated_code.Qwen3MoeForCausalLM.rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from .rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update


class Qwen3MoeRotaryEmbedding(nn.Module):
    config: Qwen3MoeConfig

    def setup(self):
        # BC: "rope_type" was originally "type"
        if hasattr(self.config, "rope_scaling") and isinstance(self.config.rope_scaling, dict):
            self.rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = self.config.max_position_embeddings
        self.original_max_seq_len = self.config.max_position_embeddings

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.inv_freq = inv_freq
        self.original_inv_freq = self.inv_freq

    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def __call__(self, x: Array, position_ids: Array) -> Tuple[Array, Array]:
        inv_freq_expanded = jnp.expand_dims(self.inv_freq, axis=(0, 2))
        inv_freq_expanded = jnp.broadcast_to(
            inv_freq_expanded, (position_ids.shape[0], inv_freq_expanded.shape[1], 1)
        ).astype(jnp.float32)
        position_ids_expanded = jnp.expand_dims(position_ids, axis=1).astype(jnp.float32)

        # Force float32
        freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

# Re-used from generated_code.Qwen3MoeForCausalLM.attention_utils.eager_attention_forward
from ..attention_utils import eager_attention_forward
# Re-used from generated_code.Qwen3MoeForCausalLM.rope_utils.apply_rotary_pos_emb
from ..rope_utils import apply_rotary_pos_emb
from .configuration_qwen3_moe import Qwen3MoeConfig
# Re-used from generated_code.Qwen3MoeForCausalLM.modeling.Qwen3MoeRMSNorm
from .modeling import Qwen3MoeRMSNorm
# Re-used from src.MaxText.layers.linears
from ....layers import linears


class Qwen3MoeAttention(nn.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  config: Qwen3MoeConfig
  layer_idx: int
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.head_dim = getattr(
        self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
    )
    self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = self.config.attention_dropout
    self.is_causal = True

    self.q_proj = linears.DenseGeneral(
        features=self.config.num_attention_heads * self.head_dim,
        use_bias=self.config.attention_bias,
        dtype=self.dtype,
        name="q_proj",
    )
    self.k_proj = linears.DenseGeneral(
        features=self.config.num_key_value_heads * self.head_dim,
        use_bias=self.config.attention_bias,
        dtype=self.dtype,
        name="k_proj",
    )
    self.v_proj = linears.DenseGeneral(
        features=self.config.num_key_value_heads * self.head_dim,
        use_bias=self.config.attention_bias,
        dtype=self.dtype,
        name="v_proj",
    )
    self.o_proj = linears.DenseGeneral(
        features=self.config.hidden_size,
        use_bias=self.config.attention_bias,
        dtype=self.dtype,
        name="o_proj",
    )
    self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=self.config.rms_norm_eps, name="q_norm")
    self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=self.config.rms_norm_eps, name="k_norm")
    self.sliding_window = getattr(self.config, "sliding_window", None)

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: Tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array],
      past_key_values: Optional[Tuple[jax.Array, jax.Array]] = None,
      deterministic: bool = True,
      use_cache: bool = False,
      output_attentions: bool = False,
      **kwargs,
  ) -> Tuple[jax.Array, Optional[Tuple[jax.Array, jax.Array]], Optional[jax.Array]]:
    batch_size, seq_length, _ = hidden_states.shape

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.reshape(
        batch_size, seq_length, self.config.num_attention_heads, self.head_dim
    )
    key_states = key_states.reshape(
        batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
    )
    value_states = value_states.reshape(
        batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
    )

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    query_states = jnp.transpose(query_states, (0, 2, 1, 3))
    key_states = jnp.transpose(key_states, (0, 2, 1, 3))
    value_states = jnp.transpose(value_states, (0, 2, 1, 3))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if use_cache and past_key_values is not None:
      # The cache update logic in PyTorch is stateful. Here we do a functional update.
      # The `cache_kwargs` are not needed for a simple concatenation-based cache.
      past_k, past_v = past_key_values
      key_states = jnp.concatenate([past_k, key_states], axis=2)
      value_states = jnp.concatenate([past_v, value_states], axis=2)

    # For simplicity, we only implement the 'eager' attention interface.
    # Other implementations like flash attention would require a similar dispatch logic.
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if deterministic else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = self.o_proj(attn_output)

    present_key_values = (key_states, value_states) if use_cache else None

    attn_weights_output = attn_weights if output_attentions else None

    return attn_output, present_key_values, attn_weights_output
