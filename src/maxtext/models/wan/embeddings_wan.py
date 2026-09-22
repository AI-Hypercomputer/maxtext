# Copyright 2025 Google LLC
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

"""Wan diffusion embedding layers."""

import math
from typing import Any, Optional, Union

from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp

from maxtext.layers.normalizations import FP32LayerNorm
from maxtext.models.wan.attention_wan import NNXSimpleFeedForward
from maxtext.models.wan.wan_utils import get_activation
from maxtext.utils import max_logging
from maxtext.utils.max_utils import get_tpu_type, safe_getattr, TpuType


def get_sinusoidal_embeddings(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    freq_shift: float = 1,
    min_timescale: float = 1,
    max_timescale: float = 1.0e4,
    flip_sin_to_cos: bool = False,
    scale: float = 1.0,
) -> jnp.ndarray:
  """Returns the positional encoding (same as Tensor2Tensor).

  Args:
      timesteps: a 1-D or 2-D Tensor of indices.
      These may be fractional.
      embedding_dim: The number of output channels.
      min_timescale: The smallest time unit (should probably be 0.0).
      max_timescale: The largest time unit.
  Returns:
      a Tensor of timing signals [B, num_channels] or [B, N, num_channels]
  """
  assert timesteps.ndim <= 2, "Timesteps should be a 1d or 2d-array"
  assert embedding_dim % 2 == 0, f"Embedding dimension {embedding_dim} should be even"
  num_timescales = float(embedding_dim // 2)
  log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - freq_shift)
  inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  emb = jnp.expand_dims(timesteps, -1) * inv_timescales

  # scale embeddings
  scaled_time = scale * emb

  if flip_sin_to_cos:
    signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=-1)
  else:
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
  return signal


class NNXTimestepEmbedding(nnx.Module):
  r"""
  Time step Embedding Module. Learns embeddings for input time steps.

  Args:
      time_embed_dim (`int`, *optional*, defaults to `32`):
              Time step embedding dimension
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
              Parameters `dtype`
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_channels: int,
      time_embed_dim: int = 32,
      act_fn: str = "silu",
      out_dim: int = None,
      post_act_fn: Optional[str] = None,
      cond_proj_dim: int = None,
      sample_proj_bias=True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      sharding_specs: Optional[Any] = None,
  ):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_channels,
        out_features=time_embed_dim,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_1_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_1_bias),
    )

    if cond_proj_dim is not None:
      self.cond_proj = nnx.Linear(
          rngs=rngs,
      )
    else:
      self.cond_proj = None

    self.act = get_activation(act_fn)

    if out_dim is not None:
      time_embed_dim_out = out_dim
    else:
      time_embed_dim_out = time_embed_dim

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=time_embed_dim,
        out_features=time_embed_dim_out,
        use_bias=sample_proj_bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_2_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_2_bias),
    )

    if post_act_fn is None:
      self.post_act = None
    else:
      self.post_act = get_activation(post_act_fn)

  def __call__(self, sample, condition=None):
    if condition is not None:
      sample = sample + self.cond_proj(condition)
    sample = self.linear_1(sample)

    if self.act is not None:
      sample = self.act(sample)
    sample = self.linear_2(sample)

    if self.post_act is not None:
      sample = self.post_act(sample)
    return sample


class FlaxTimestepEmbedding(nn.Module):
  r"""
  Time step Embedding Module. Learns embeddings for input time steps.

  Args:
      time_embed_dim (`int`, *optional*, defaults to `32`):
              Time step embedding dimension
      dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
              Parameters `dtype`
  """

  time_embed_dim: int = 32
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, temb):
    temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, name="linear_1")(temb)
    temb = nn.silu(temb)
    temb = nn.Dense(self.time_embed_dim, dtype=self.dtype, param_dtype=self.weights_dtype, name="linear_2")(temb)
    return temb


class NNXFlaxTimesteps(nnx.Module):

  def __init__(
      self,
      dim: int = 32,
      flip_sin_to_cos: bool = False,
      freq_shift: float = 1.0,
      scale: int = 1,
  ):
    self.dim = dim
    self.flip_sin_to_cos = flip_sin_to_cos
    self.freq_shift = freq_shift
    self.scale = scale

  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(
        timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
    )


class FlaxTimesteps(nn.Module):
  r"""
  Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239

  Args:
      dim (`int`, *optional*, defaults to `32`):
              Time step embedding dimension
  """

  dim: int = 32
  flip_sin_to_cos: bool = False
  freq_shift: float = 1.0
  scale: int = 1

  @nn.compact
  def __call__(self, timesteps):
    return get_sinusoidal_embeddings(
        timesteps, embedding_dim=self.dim, flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
    )


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[jnp.array, int],
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=jnp.float32,
    use_real: bool = True,
):
  """
  Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
  """
  assert dim % 2 == 0

  if isinstance(pos, int):
    pos = jnp.arange(pos)

  theta = theta * ntk_factor
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor
  freqs = jnp.outer(pos, freqs)
  if use_real:
    # Flux
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    out = jnp.stack([freqs_cos, -freqs_sin, freqs_sin, freqs_cos], axis=-1)
  else:
    # Wan 2.1
    out = jnp.exp(1j * freqs)
  return out


class NNXWanImageEmbedding(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_features: int,
      out_features: int,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: jax.lax.Precision,
      pos_embed_seq_len=None,
      alignment: Optional[int] = None,
      flash_min_seq_length: int = 4096,
  ):
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=in_features, elementwise_affine=True, eps=1e-6)
    self.ff = NNXSimpleFeedForward(
        rngs=rngs,
        dim=in_features,
        dim_out=out_features,
        mult=1,
        activation_fn="gelu",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=out_features, elementwise_affine=True, eps=1e-6)
    if alignment is None:
      tpu_type = get_tpu_type()
      self.alignment = 256 if tpu_type in [TpuType.TPU_V6_LITE, TpuType.TPU_7X] else 128
    else:
      self.alignment = alignment
    self.flash_min_seq_length = flash_min_seq_length
    if pos_embed_seq_len is not None:
      self.pos_embed = nnx.Param(jnp.zeros((1, pos_embed_seq_len, in_features), dtype=dtype))
    else:
      self.pos_embed = nnx.data(None)

  def __call__(self, encoder_hidden_states_image: jax.Array) -> tuple[jax.Array, jax.Array]:
    hidden_states = encoder_hidden_states_image
    B, current_seq_len, D_in = hidden_states.shape

    if self.pos_embed is not None:
      pe_len = self.pos_embed.value.shape[1]
      add_len = min(current_seq_len, pe_len)
      # Apply pos_embed to the original sequence length
      hidden_states = hidden_states.at[:, :add_len, :].add(self.pos_embed.value[:, :add_len, :])
      if current_seq_len > pe_len:
        max_logging.log(f"[WARN] Input seq_len {current_seq_len} > pos_embed len {pe_len}")

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.ff(hidden_states)
    hidden_states = self.norm2(hidden_states)
    # hidden_states shape: (B, current_seq_len, out_features)
    B, current_seq_len, D_out = hidden_states.shape
    use_flash_attn = current_seq_len >= self.flash_min_seq_length

    if use_flash_attn:
      # --- Dynamic Padding to nearest multiple of self.alignment ---
      num_blocks = (current_seq_len + self.alignment - 1) // self.alignment
      target_seq_len = num_blocks * self.alignment
    else:
      target_seq_len = current_seq_len

    # Create attention mask: 1 for real tokens, 0 for padded tokens
    attention_mask = jnp.ones((B, current_seq_len), dtype=jnp.int32)

    if current_seq_len < target_seq_len:
      padding_size = target_seq_len - current_seq_len
      padding = jnp.zeros((B, padding_size, D_out), dtype=hidden_states.dtype)
      hidden_states = jnp.concatenate([hidden_states, padding], axis=1)

      # Extend mask with zeros for padded positions
      padding_mask = jnp.zeros((B, padding_size), dtype=jnp.int32)
      attention_mask = jnp.concatenate([attention_mask, padding_mask], axis=1)
    if not use_flash_attn:
      attention_mask = None
    return hidden_states, attention_mask


class NNXPixArtAlphaTextProjection(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      in_features: int,
      hidden_size: int,
      out_features: int = None,
      act_fn: str = "gelu_tanh",
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      sharding_specs: Optional[Any] = None,
  ):
    linear_1_kernel = safe_getattr(sharding_specs, "emb_linear_1_kernel", ("embed", "mlp"))
    linear_1_bias = safe_getattr(sharding_specs, "emb_linear_1_bias", ("mlp",))
    linear_2_kernel = safe_getattr(sharding_specs, "emb_linear_2_kernel", ("mlp", "embed"))
    linear_2_bias = safe_getattr(sharding_specs, "emb_linear_2_bias", ("embed",))
    if out_features is None:
      out_features = hidden_size

    self.linear_1 = nnx.Linear(
        rngs=rngs,
        in_features=in_features,
        out_features=hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_1_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_1_bias),
    )
    self.act_1 = get_activation(act_fn)

    self.linear_2 = nnx.Linear(
        rngs=rngs,
        in_features=hidden_size,
        out_features=out_features,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            linear_2_kernel,
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, linear_2_bias),
    )

  def __call__(self, caption):
    hidden_states = self.linear_1(caption)
    hidden_states = self.act_1(hidden_states)
    hidden_states = self.linear_2(hidden_states)
    return hidden_states


class PixArtAlphaTextProjection(nn.Module):
  """
  Projects caption embeddings. Also handles dropout for classifier-free guidance.
  Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
  """

  hidden_size: int
  out_features: int = None
  act_fn: str = "gelu_tanh"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None

  @nn.compact
  def __call__(self, caption):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_1",
    )(caption)
    if self.act_fn == "gelu_tanh":
      act_1 = nn.gelu
    elif self.act_fn == "silu":
      act_1 = nn.swish
    else:
      raise ValueError(f"Unknown activation function: {self.act_fn}")
    hidden_states = act_1(hidden_states)

    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        name="linear_2",
    )(hidden_states)
    return hidden_states
