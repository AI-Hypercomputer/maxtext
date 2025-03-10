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

"""Embedding Layers."""

import math
from typing import Any, Optional

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
from layers import initializers

Config = Any
Array = jnp.ndarray
DType = jnp.dtype

Initializer = initializers.Initializer
default_embed_init = initializers.default_embed_init
with_logical_partitioning = nn.with_logical_partitioning

_MAX_WAVELENGTH = 10_000


class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
  """

  # pylint: disable=attribute-defined-outside-init
  config: Config
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init

  def setup(self):
    self.embedding = self.param(
        "embedding",
        with_logical_partitioning(self.embedding_init, ("vocab", "embed")),
        (self.num_embeddings, self.features),
        self.config.weight_dtype,
    )

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
    output = nn.with_logical_constraint(
        output, ("activation_embed_and_logits_batch", "activation_length", "activation_embed")
    )
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, jnp.bfloat16).T)


class RotaryEmbedding(nn.Module):
  """Rotary Position Embedding.

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """

  min_timescale: int
  max_timescale: int
  embedding_dims: int = 0
  cast_as_fprop_dtype: bool = True
  fprop_dtype: DType = jnp.bfloat16

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    self.timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    self.timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Generates a jax.Array of sinusoids with different frequencies.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. Since rotary position embeddings are applied to query and
        keys after projection, it is assumed of shape [B, S, N, H].
      position: Optional position jax.Array which denotes the position of each
        token in the sequence. This only needs to be supplied when the sequence
        is packed. It is of shape [B, S].

    Returns:
      a jax.Array of shape [B, S, N, H] which includes the inputs together with
      the rotary position embedding incorporated in it.
    """
    assert position is not None
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape" "[batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
      )

    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale
    sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
    cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    x_out = jnp.concatenate((first_part, second_part), axis=-1)
    return x_out


class LLaMARotaryEmbedding(RotaryEmbedding):
  """LLaMA variant of ROPE."""

  # # LLaMA3.1 ROPE scaling, see the original pytorch implementation
  # https://github.com/meta-llama/llama-models/blob/301ca3a2b3b10e94ddcd1fdd2c57e52f812e1cac/models/llama3/reference_impl/model.py#L45C5-L45C18
  use_scale: bool = True

  def _apply_scaling_factor(self, freq):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * jnp.pi / freq

    def lower_wavelen(freq):
      return freq

    def bigger_or_equal_wavelen(freq):
      def bigger_wavelen(freq):
        return freq / scale_factor

      def equal_wavelen(freq):
        smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        return (1 - smooth) * freq / scale_factor + smooth * freq

      bigger_wavelen_cond = wavelen > low_freq_wavelen
      return jax.lax.cond(bigger_wavelen_cond, bigger_wavelen, equal_wavelen, freq)

    lower_wavelen_cond = wavelen < high_freq_wavelen
    return jax.lax.cond(lower_wavelen_cond, lower_wavelen, bigger_or_equal_wavelen, freq)

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    # Apply scaling factor if enabled
    if self.use_scale:
      timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(1.0 / timescale)

    # Expand timescale dimensions for broadcasting
    self.timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

  def __call__(self, inputs: jax.Array, position: Optional[jax.Array] = None) -> jax.Array:
    """Applies LLaMA variant of rotary position embedding.

    Args:
      inputs: The input sequence on which to apply the Rotary position
        embedding. It is assumed of shape [B, S, N, H].
      position: Optional position array [B, S]. Only needed when the sequence
        is packed.

    Returns:
      A jax.Array of shape [B, S, N, H] with rotary position embeddings applied.
    """
    # Ensure input is 4D
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [B, S, N, H].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError("The embedding dims of the rotary position embedding must match the hidden dimension of the inputs.")

    # Shift the inputs left and right as per LLaMA's specific behavior
    inputs_shifted_left = jnp.concatenate([inputs[..., 1:], inputs[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate([inputs[..., -1:], inputs[..., :-1]], axis=-1)
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(self.embedding_dims, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),
        inputs_shifted_right,
        inputs_shifted_left,
    )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.float32)[jnp.newaxis, :]

    # Calculate sinusoidal input
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / self.timescale

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    # Apply alternating sign
    sign = jnp.tile(jnp.array([-1, 1]), self.embedding_dims // 2)

    # Combine original inputs with sinusoidal information
    outputs = inputs * cos + inputs_shifted * sin * sign

    if self.cast_as_fprop_dtype:
      outputs = outputs.astype(self.fprop_dtype)

    return outputs


class YarnRotaryEmbedding(nn.Module):
  """Yarn rotary embedding.

  Based on https://arxiv.org/abs/2309.00071
  This implementation uses DeepSeek-v3 PyTorch as reference
  https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L294

  Attributes:
    embedding_dims: Dimension of the embedding to be generated.
    max_position_embeddings: The maximum sequence length that will be encountered.
    original_max_position_embeddings: The sequence length for which the base frequencies were defined.
    beta_fast: Lower bound parameter for correction.
    beta_slow: Upper bound parameter for correction.
    rope_theta: The base theta value for the frequency computation.
    rope_factor: Factor applied to adjust the frequencies.
  """

  embedding_dims: int
  max_position_embeddings: int = 4096 * 4
  original_max_position_embeddings: int = 4096
  beta_fast: float = 32
  beta_slow: float = 1
  rope_theta: float = 10000.0
  rope_factor: float = 40
  cast_as_fprop_dtype: bool = True
  fprop_dtype: DType = jnp.bfloat16

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

    half_dim = self.embedding_dims // 2
    # Compute base frequencies for each (even-indexed) dimension.
    # (Note: We use jnp.arange with float32 for precision.)
    freqs = 1.0 / (self.rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / self.embedding_dims))

    low, high = self._find_correction_range(
        self.beta_fast, self.beta_slow, self.embedding_dims, self.rope_theta, self.original_max_position_embeddings
    )
    smooth = 1 - self._linear_ramp_factor(low, high, half_dim)
    # The corrected frequency is a weighted mix of the scaled and base values.
    freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth

    # Precompute frequencies for all positions by taking the outer product.
    t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
    # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
    freqs = jnp.outer(t, freqs)
    # Compute the complex “cis” values: exp(i * theta).
    self.freqs_cis = jnp.exp(1j * freqs)  # shape [max_position_embeddings, half_dim]

  def _find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """Compute the correction dimension for a given number of rotations."""
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  def _find_correction_range(self, low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int):
    """Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_position_embeddings (int): Maximum sequence length.

    Returns:
        Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    """
    low = math.floor(self._find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(self._find_correction_dim(high_rot, dim, base, max_position_embeddings))
    low = max(low, 0)
    high = min(high, dim - 1)
    return low, high

  def _linear_ramp_factor(self, min_val: float, max_val: float, dim: int) -> Array:
    """Computes a linear ramp over the dimension.

    Returns a jax.Array of shape (dim,) with values between 0 and 1.
    """
    if min_val == max_val:
      max_val += 0.001  # Avoid division by zero.
    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
    return jnp.clip(linear_func, 0, 1)

  def __call__(self, inputs: Array, position: Optional[Array] = None) -> Array:
    """Applies the rotary positional embedding using the precomputed complex frequencies.

    Args:
      inputs: jax.Array of shape [B, S, N, H]. (H must equal self.embedding_dims.)
      position: jax.Array of shape [B, S] with integer positions (indexes into precomputed freqs).

    Returns:
      jax.Array of shape [B, S, N, H] with the rotary embedding applied.
    """
    if len(inputs.shape) != 4:
      raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, dims].")
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError("The embedding dims of the rotary position embedding must match the hidden dimension of the inputs.")

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]
    else:
      position = position.astype(jnp.int32)

    B, S, N, H = inputs.shape
    half_dim = H // 2

    # Convert the last dimension into a complex representation.
    # First reshape so that each pair of numbers represents the real and imaginary parts.
    inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
    inputs_complex = inputs_reshaped[..., 0] + 1j * inputs_reshaped[..., 1]  # shape: [B, S, N, half_dim]

    # Lookup the precomputed frequencies using the position indices.
    # self.freqs_cis has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
    # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
    freqs = jnp.take(self.freqs_cis, position, axis=0)  # shape: [B, S, half_dim]
    freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]

    # Apply the rotary transformation via complex multiplication.
    rotated = inputs_complex * freqs  # shape: [B, S, N, half_dim]

    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    rotated_real = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-2)  # shape: [B, S, N, 2, half_dim]
    # [sin1, sin2, sin3, ..., cos1, cos2, ...] at last dimension
    output = rotated_real.reshape(B, S, N, H)
    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)
    return output


class PositionalEmbedding(nn.Module):
  embedding_dims: int
  max_wavelength: int = _MAX_WAVELENGTH

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      input_embedding: jax.Array,
      position: jax.Array,
  ) -> jax.Array:
    num_timescales = self.embedding_dims // 2
    log_timescale_increment = jnp.log(float(self.max_wavelength)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
    )
    inv_timescales = jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
    position = position[:, :, jnp.newaxis]
    inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=-1)
    # signal = jnp.pad(signal, [[0, jnp.mod(self.embedding_dims, 2)]])
    position_embedding = signal.astype(jnp.float32)
    return input_embedding + position_embedding
