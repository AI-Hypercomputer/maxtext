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
        'embedding',
        with_logical_partitioning(self.embedding_init, ('vocab', 'embed')),
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
      raise ValueError('Input type must be an integer or unsigned integer.')

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
    output = nn.with_logical_constraint(
        output, ('activation_batch', 'activation_length', 'activation_embed')
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
  """RoPE

  Attributes:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
  """

  min_timescale: int = 1
  max_timescale: int = 10_000
  embedding_dims: int = 0
  cast_as_fprop_dtype: bool = True
  fprop_dtype: DType = jnp.bfloat16

  def setup(self) -> None:
    if self.embedding_dims % 2:
      raise ValueError(
          'Embedding dim for rotary position embedding must be a multiple of 2.'
      )

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: jax.Array,
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
      raise ValueError(
          'Input is assumed to be a rank 4 tensor of shape'
          '[batch, sequence, heads, dims].'
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          'The embedding dims of the rotary position embedding'
          'must match the hidden dimension of the inputs.'
      )
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = (
        self.min_timescale
        * (self.max_timescale / self.min_timescale) ** fraction
    )
    position = position[:, :, jnp.newaxis, jnp.newaxis]
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    if self.cast_as_fprop_dtype:
      first_part = first_part.astype(self.fprop_dtype)
      second_part = second_part.astype(self.fprop_dtype)
    x_out = jnp.concatenate((first_part, second_part), axis=-1)
    return x_out


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
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    position = position[:, :, jnp.newaxis]
    inv_timescales = inv_timescales[jnp.newaxis, jnp.newaxis, :]
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis = -1)
    # signal = jnp.pad(signal, [[0, jnp.mod(self.embedding_dims, 2)]])
    position_embedding = signal.astype(jnp.float32)
    return input_embedding + position_embedding