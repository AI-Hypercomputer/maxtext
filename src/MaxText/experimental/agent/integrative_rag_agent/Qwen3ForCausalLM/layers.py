
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional

# The functionality of this module is equivalent to src.MaxText.layers.normalizations.RMSNorm
class Qwen3MoeRMSNorm(nn.Module):
    """
    A Flax implementation of Qwen3MoeRMSNorm, which is equivalent to T5LayerNorm.
    """
    hidden_size: int
    eps: float = 1e-6
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        """Initializes the weight parameter."""
        self.weight = self.param(
            'weight',
            nn.initializers.ones,
            (self.hidden_size,),
            self.param_dtype,
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Applies RMS normalization to the input tensor."""
        input_dtype = hidden_states.dtype
        output_dtype = self.dtype if self.dtype is not None else input_dtype

        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jax.lax.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).astype(output_dtype)


# The functionality of preparing arguments for flash attention, which is
# initiated by the lazy loading pattern involving `_process_flash_kwargs_fn`,
# is handled by the `AttentionOp` class in MaxText. This class encapsulates
# different attention mechanisms and prepares the necessary arguments for the
# specific kernel being used (e.g., TPU Splash Attention, GPU Pallas MHA),
# based on the model configuration. Therefore, this variable and the associated
# dynamic argument processing are not needed in the JAX/MaxText implementation.
#
# Reused module: src.MaxText.layers.attention_op.AttentionOp

import math

from flax import linen as nn
import jax.numpy as jnp

from maxtext.common_types import Array


class AccurateGELUActivation(nn.Module):
  """
  Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
  https://github.com/hendrycks/GELUs

  Implemented along with MEGA (Moving Average Equipped Gated Attention)
  """

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies the Accurate GELU activation function."""
    precomputed_constant = math.sqrt(2 / math.pi)
    return 0.5 * inputs * (1 + jnp.tanh(precomputed_constant * (inputs + 0.044715 * jnp.power(inputs, 3))))

import math
from flax.linen import Module
import flax.linen as nn
import jax
import jax.numpy as jnp

from MaxText.common_types import Array


class GELUActivation(Module):
  """
  Original Implementation of the GELU activation function in Google BERT repo when initially created. For
  information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
  jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3)))) This is now written in C in nn.functional
  Also see the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
  """

  use_gelu_python: bool = False

  def _gelu_python(self, inputs: Array) -> Array:
    return inputs * 0.5 * (1.0 + jax.lax.erf(inputs / jnp.sqrt(2.0)))

  def __call__(self, inputs: Array) -> Array:
    if self.use_gelu_python:
      return self._gelu_python(inputs)
    else:
      return nn.gelu(inputs)

import flax.linen as nn

from maxtext.common_types import Array


class LinearActivation(nn.Module):
  """Applies the linear activation function, i.e. forwarding input directly to output."""

  def __call__(self, inputs: Array) -> Array:
    return inputs

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array


class MishActivation(nn.Module):
  """
  See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681). Also
  visit the official repository for the paper: https://github.com/digantamisra98/Mish
  """

  def _mish_jax(self, inputs: Array) -> Array:
    return inputs * jnp.tanh(jax.nn.softplus(inputs))

  def __call__(self, inputs: Array) -> Array:
    return nn.mish(inputs)

import math
from flax.linen import Module
import jax.numpy as jnp

from MaxText.common_types import Array


class NewGELUActivation(Module):
  """
  Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
  the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
  """

  def __call__(self, inputs: Array) -> Array:
    return 0.5 * inputs * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * jnp.power(inputs, 3.0))))

import flax.linen as nn

from MaxText.common_types import Array


class PytorchGELUTanh(nn.Module):
  """
  A fast C implementation of the tanh approximation of the GeLU activation function. See
  https://huggingface.co/papers/1606.08415.

  This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
  match due to rounding errors.
  """

  def __call__(self, inputs: Array) -> Array:
    """Applies the GELU activation function with tanh approximation."""
    return nn.gelu(inputs, approximate=True)

from flax import linen as nn
import jax.nn

from MaxText.common_types import Array


class QuickGELUActivation(nn.Module):
  """
  Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    return inputs * jax.nn.sigmoid(1.702 * inputs)

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


class ReLUSquaredActivation(nn.Module):
  """Applies the relu^2 activation introduced in https://huggingface.co/papers/2109.08668v2."""

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    relu_applied = nn.relu(inputs)
    squared = jnp.square(relu_applied)
    return squared

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

"""JAX activations functions."""

import flax.linen as nn
import jax.numpy as jnp
from MaxText.common_types import Array


class FastGELUActivation(nn.Module):
  """Applies GELU approximation that is slower than QuickGELU but more accurate.

  See: https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, input_tensor: Array) -> Array:
    return 0.5 * input_tensor * (
        1.0
        + jnp.tanh(
            input_tensor
            * 0.7978845608
            * (1.0 + 0.044715 * input_tensor * input_tensor)
        )
    )

from typing import Callable, Optional, Tuple

# Used: Qwen3ForCausalLM.modeling_utils._pad_input
# Used: Qwen3ForCausalLM.modeling_utils._unpad_input
from .modeling_utils import _pad_input, _unpad_input
# Used: Qwen3ForCausalLM.modeling_utils.is_flash_attn_2_available
# Used: Qwen3ForCausalLM.modeling_utils.is_flash_attn_3_available
# Used: Qwen3ForCausalLM.modeling_utils.is_torch_npu_available
from .utils import is_flash_attn_2_available, is_flash_attn_3_available, is_torch_npu_available


def _lazy_imports(implementation: Optional[str]) -> Tuple[Callable, Callable, Callable, Callable]:
  """
  Lazy loads the respective flash attention implementations.

  Return:
      flash_attn_func: The base flash attention function.
      flash_attn_varlen_func: The flash attention function supporting variable sequence lengths,
                              e.g. for padding-free training.
      pad_input: The function to pad inputs into one sequence and returning the respective kwargs.
      unpad_input: The function to unpad outputs based on the kwargs (from pad_input).
  """
  is_fa2 = is_flash_attn_2_available()
  is_fa3 = is_flash_attn_3_available()

  pad_input, unpad_input = _pad_input, _unpad_input

  if (implementation == "flash_attention_2" and is_fa2) or (implementation is None and is_fa2 and not is_fa3):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
  elif is_torch_npu_available():
    # Package `flash-attn` is unavailable on Ascend NPU, which will cause ImportError
    # Flash-Attention2 related apis for Ascend NPU must be imported from `.integrations.npu_flash_attention` module
    from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from .integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
  else:
    if implementation == "flash_attention_3" or (implementation is None and is_fa3):
      from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    # Kernels fallback
    else:
      flash_attn_func = getattr(implementation, "flash_attn_func", None)
      flash_attn_varlen_func = getattr(implementation, "flash_attn_varlen_func", None)
      if flash_attn_varlen_func is None or flash_attn_func is None:
        raise ValueError(
            f"Could not find the currently requested flash attention implementation at `{implementation}`."
            "Make sure that you request a valid kernel from the hub, e.g. `kernels-community/flash-attn`."
        )

  return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input

from __future__ import annotations

from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .configuration_qwen3_moe import Qwen3MoeConfig
# Re-used: Qwen3ForCausalLM.cache.Cache
from .cache import Cache
# Re-used: Qwen3ForCausalLM.layers.Qwen3MoeRMSNorm
from .layers import Qwen3MoeRMSNorm
# Re-used: Qwen3ForCausalLM.modeling_utils.apply_rotary_pos_emb
from .modeling_utils import apply_rotary_pos_emb
# Re-used: Qwen3ForCausalLM.modeling_utils.eager_attention_forward
from .modeling_utils import eager_attention_forward


class Qwen3MoeAttention(nn.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  config: Qwen3MoeConfig
  layer_idx: int
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None

  def setup(self):
    config = self.config
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    dense_kwargs = {
        "use_bias": config.attention_bias,
        "dtype": self.dtype,
        "param_dtype": self.param_dtype,
        "precision": self.precision,
        "kernel_init": nn.initializers.normal(self.config.initializer_range),
    }

    self.q_proj = nn.Dense(features=config.num_attention_heads * self.head_dim, **dense_kwargs)
    self.k_proj = nn.Dense(features=config.num_key_value_heads * self.head_dim, **dense_kwargs)
    self.v_proj = nn.Dense(features=config.num_key_value_heads * self.head_dim, **dense_kwargs)
    self.o_proj = nn.Dense(features=config.hidden_size, **dense_kwargs)

    rms_norm_kwargs = {
        "eps": config.rms_norm_eps,
        "dtype": self.dtype,
        "param_dtype": self.param_dtype,
    }
    self.q_norm = Qwen3MoeRMSNorm(self.head_dim, **rms_norm_kwargs)
    self.k_norm = Qwen3MoeRMSNorm(self.head_dim, **rms_norm_kwargs)
    self.sliding_window = getattr(config, "sliding_window", None)

  def __call__(
      self,
      hidden_states: jax.Array,
      position_embeddings: tuple[jax.Array, jax.Array],
      attention_mask: Optional[jax.Array],
      past_key_values: Optional[Cache] = None,
      cache_position: Optional[jax.Array] = None,
      deterministic: bool = True,
      **kwargs,
  ) -> tuple[jax.Array, Optional[jax.Array]]:
    input_shape = hidden_states.shape[:-1]
    batch_size, seq_len = input_shape
    hidden_shape_q = (batch_size, seq_len, self.config.num_attention_heads, self.head_dim)
    hidden_shape_kv = (batch_size, seq_len, self.config.num_key_value_heads, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).reshape(hidden_shape_q))
    key_states = self.k_norm(self.k_proj(hidden_states).reshape(hidden_shape_kv))
    value_states = self.v_proj(hidden_states).reshape(hidden_shape_kv)

    query_states = jnp.transpose(query_states, (0, 2, 1, 3))
    key_states = jnp.transpose(key_states, (0, 2, 1, 3))
    value_states = jnp.transpose(value_states, (0, 2, 1, 3))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
      # sin and cos are specific to RoPE models; cache_position needed for the static cache
      cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
      key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # In JAX, the choice of attention implementation is static.
    # We assume config._attn_implementation is 'eager' and use the corresponding JAX implementation.
    attention_interface: Callable = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if deterministic else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        deterministic=deterministic,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp

# Reused from Qwen3ForCausalLM.activations.ACT2FN
from Qwen3ForCausalLM.activations import ACT2FN
# Reused from Qwen3ForCausalLM.config.Qwen3MoeConfig
from Qwen3ForCausalLM.config import Qwen3MoeConfig

DType = Any


class Qwen3MoeMLP(nn.Module):
  """A Qwen3 MoE MLP block."""

  config: Qwen3MoeConfig
  intermediate_size: Optional[int] = None
  dtype: DType = jnp.float32
  param_dtype: DType = jnp.float32
  precision: Optional[str] = None

  def setup(self):
    if self.intermediate_size is not None:
      intermediate_size = self.intermediate_size
    else:
      intermediate_size = self.config.intermediate_size

    self.gate_proj = nn.Dense(
        intermediate_size,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name="gate_proj",
    )
    self.up_proj = nn.Dense(
        intermediate_size,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name="up_proj",
    )
    self.down_proj = nn.Dense(
        self.config.hidden_size,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name="down_proj",
    )
    self.act_fn = ACT2FN[self.config.hidden_act]

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the MLP block to the input tensor."""
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple

from MaxText.layers import linears
from MaxText.common_types import Config, Array, DType

class Qwen3MoeSparseMoeBlock(nn.Module):
  """
  A sparse MoE block for Qwen3, translated from PyTorch.

  This implementation uses a `fori_loop` over experts, which is a direct
  translation of the original PyTorch logic and is JIT-compatible.
  """

  config: Config
  dtype: DType = jnp.float32
  param_dtype: DType = jnp.float32
  precision: jax.lax.Precision = jax.lax.Precision.DEFAULT

  def setup(self):
    """Initializes the gate and expert layers."""
    self.num_experts = self.config.num_experts
    self.top_k = self.config.num_experts_per_tok
    self.norm_topk_prob = self.config.norm_topk_prob

    self.gate = nn.Dense(
        features=self.num_experts,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        name="gate",
    )
    self.experts = [
        # Re-used src.MaxText.layers.linears.mlp_block
        linears.mlp_block(
            config=self.config,
            in_features=self.config.hidden_size,
            intermediate_dim=self.config.moe_intermediate_size,
            activations=self.config.mlp_activations,
            intermediate_dropout_rate=0.0,
            name=f"expert_{i}",
            dtype=self.dtype,
        )
        for i in range(self.num_experts)
    ]

  def __call__(self, hidden_states: Array, deterministic: bool = False) -> Tuple[Array, Array]:
    """Applies the sparse MoE block to the input hidden states."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = jnp.reshape(hidden_states, (-1, hidden_dim))
    num_tokens = hidden_states_reshaped.shape[0]

    # router_logits: (num_tokens, n_experts)
    router_logits = self.gate(hidden_states_reshaped)

    routing_weights = jax.nn.softmax(router_logits, axis=1, dtype=jnp.float32)
    top_k_weights, top_k_indices = jax.lax.top_k(routing_weights, k=self.top_k)

    if self.norm_topk_prob:
      top_k_weights /= jnp.sum(top_k_weights, axis=-1, keepdims=True)

    top_k_weights = top_k_weights.astype(self.dtype)

    final_hidden_states = jnp.zeros_like(hidden_states_reshaped)

    # One hot encode the selected experts to create an expert mask
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    # Transpose to (num_experts, top_k, num_tokens) to match PyTorch loop structure
    expert_mask = jnp.transpose(expert_mask, (2, 1, 0))

    def expert_loop_body(expert_idx, loop_val):
      """Body of the fori_loop, processing one expert at a time."""
      current_final_hidden_states = loop_val

      # Find which tokens are routed to this expert
      # expert_mask[expert_idx] has shape (top_k, num_tokens)
      # top_k_ranks: k-th expert chosen for a token (0, 1, ...)
      # token_indices: index of the token
      # The size must be statically known for JIT, so we use a safe upper bound.
      capacity = num_tokens * self.top_k
      top_k_ranks, token_indices = jnp.where(expert_mask[expert_idx], size=capacity, fill_value=-1)

      valid_mask = token_indices != -1

      def process_expert(carry):
        """Process tokens for the current expert if it was selected."""
        h_states = carry

        valid_token_indices = token_indices[valid_mask]
        valid_top_k_ranks = top_k_ranks[valid_mask]

        # Index the correct hidden states
        current_state = h_states[valid_token_indices]

        # Apply the correct expert MLP using jax.lax.switch
        branches = [lambda x: expert(x, deterministic=deterministic) for expert in self.experts]
        expert_output = jax.lax.switch(expert_idx, branches, current_state)

        # Scale by routing weights
        weights_for_expert = top_k_weights[valid_token_indices, valid_top_k_ranks]
        expert_output *= weights_for_expert[:, None]

        # Add back to final hidden states
        h_states = h_states.at[valid_token_indices].add(expert_output.astype(self.dtype))
        return h_states

      def skip_expert(carry):
        """Do nothing if the expert was not selected."""
        return carry

      # Conditionally apply the expert logic
      updated_final_hidden_states = jax.lax.cond(
          jnp.any(valid_mask), process_expert, skip_expert, hidden_states_reshaped
      )
      # The scatter-add operation needs to be applied to the loop-carried state
      return jax.lax.cond(
          jnp.any(valid_mask),
          lambda: current_final_hidden_states
          + (updated_final_hidden_states - hidden_states_reshaped),
          lambda: current_final_hidden_states,
      )

    final_hidden_states = jax.lax.fori_loop(0, self.num_experts, expert_loop_body, final_hidden_states)

    final_hidden_states = jnp.reshape(final_hidden_states, (batch_size, sequence_length, hidden_dim))
    return final_hidden_states, router_logits

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from MaxText.common_types import Array, Config, DType, Mesh
from MaxText.layers import attentions, linears, moe, normalizations
from MaxText.layers.quantizations import Quant


class Qwen3MoeDecoderLayer(nn.Module):
  """A Qwen3 MoE decoder layer."""

  config: Config
  mesh: Mesh
  layer_idx: int
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      hidden_states: Array,
      decoder_segment_ids: Array,
      decoder_positions: Array,
      deterministic: bool,
      model_mode: str,
  ) -> Array | Tuple[Array, None]:
    """
    Args:
        hidden_states (`Array`): input to the layer of shape `(batch, seq_len, embed_dim)`
        decoder_segment_ids (`Array`): segment ids for the attention mask.
        decoder_positions (`Array`): positions for the rotary embedding.
        deterministic (`bool`):
            Whether or not to return the attentions tensors of all attention layers.
        model_mode (`str`):
            The mode of the model, e.g., "train", "prefill", "autoregressive".
    """
    # Self Attention
    residual = hidden_states

    hidden_states = normalizations.rms_norm(
        self.config.hidden_size,
        name="input_layernorm",
        epsilon=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
    )(hidden_states)

    # Reused from src.MaxText.layers.attentions.attention_as_linen
    attention_output, _ = attentions.attention_as_linen(
        self.config,
        num_query_heads=self.config.num_attention_heads,
        num_kv_heads=self.config.num_key_value_heads,
        head_dim=self.config.head_dim,
        mesh=self.mesh,
        attention_kernel=self.config.attention,
        name="self_attn",
        quant=self.quant,
    )(
        inputs_q=hidden_states,
        inputs_kv=hidden_states,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    hidden_states = residual + attention_output

    # Fully Connected
    residual = hidden_states
    hidden_states = normalizations.rms_norm(
        self.config.hidden_size,
        name="post_attention_layernorm",
        epsilon=self.config.rms_norm_eps,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
    )(hidden_states)

    is_moe_layer = (
        self.layer_idx not in self.config.mlp_only_layers
        and self.config.num_experts > 0
        and (self.layer_idx + 1) % self.config.decoder_sparse_step == 0
    )

    if is_moe_layer:
      # Reused from src.MaxText.layers.moe.get_routed_moe
      mlp_output, router_logits = moe.get_routed_moe(
          self.config,
          self.mesh,
          num_experts=self.config.num_experts,
          num_experts_per_tok=self.config.num_experts_per_tok,
          intermediate_dim=self.config.moe_intermediate_size,
          name="mlp",
          quant=self.quant,
      )(hidden_states, deterministic=deterministic)
      if router_logits is not None:
        self.sow("intermediates", "router_logits", router_logits)
    else:
      # Reused from src.MaxText.layers.linears.mlp_block
      mlp_output = linears.mlp_block(
          self.config,
          intermediate_dim=self.config.intermediate_size,
          name="mlp",
          quant=self.quant,
      )(hidden_states, deterministic=deterministic)

    hidden_states = residual + mlp_output

    if self.config.scan_layers:
      return hidden_states, None
    else:
      return hidden_states

from flax.linen import Module
import jax.numpy as jnp
from typing import Tuple

# Reused from Qwen3ForCausalLM.config.Qwen3MoeConfig
from Qwen3ForCausalLM.config import Qwen3MoeConfig
# Reused from Qwen3ForCausalLM.modeling_utils.dynamic_rope_update
from Qwen3ForCausalLM.modeling_utils import dynamic_rope_update
# Reused from Qwen3ForCausalLM.rope.ROPE_INIT_FUNCTIONS
from Qwen3ForCausalLM.rope import ROPE_INIT_FUNCTIONS
# Reused from maxtext.common_types.Array
from maxtext.common_types import Array
# Reused from maxtext.common_types.DType
from maxtext.common_types import DType


class Qwen3MoeRotaryEmbedding(Module):
  """Qwen3MoeRotaryEmbedding module."""

  config: Qwen3MoeConfig
  dtype: DType = jnp.float32

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
    self.inv_freq = self.variable("buffers", "inv_freq", lambda: inv_freq)
    self.original_inv_freq = inv_freq

  @dynamic_rope_update
  def __call__(self, x: Array, position_ids: Array) -> Tuple[Array, Array]:
    """Applies rotary position embedding to the input."""
    inv_freq = self.inv_freq.value

    inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 2))
    inv_freq_expanded = jnp.broadcast_to(inv_freq_expanded, (position_ids.shape[0], inv_freq.shape[0], 1))
    position_ids_expanded = jnp.expand_dims(position_ids, axis=1)

    # Force float32 for freqs computation
    freqs = (inv_freq_expanded.astype(jnp.float32) @ position_ids_expanded.astype(jnp.float32)).transpose(0, 2, 1)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    cos = jnp.cos(emb) * self.attention_scaling
    sin = jnp.sin(emb) * self.attention_scaling

    return cos.astype(self.dtype), sin.astype(self.dtype)
