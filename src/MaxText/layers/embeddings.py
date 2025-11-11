# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding Layers."""

import dataclasses
import math

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from flax import linen as nn
from flax import nnx

from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import ShardMode, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, Array, Config, DType
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import Initializer, default_embed_init, variable_to_logically_partitioned

_MAX_WAVELENGTH = 10_000


def _maybe_move_embedding_to_device(embedding_table: Array, config: Config) -> Array:
  """Moves embedding table to device if parameter offloading is enabled."""
  if config.parameter_memory_host_offload:
    max_logging.log("embeddings.py: Moving embedding parameter to device")
    return jax.device_put(embedding_table, max_utils.device_space())
  return embedding_table


def embed_as_linen(
    *,
    num_embeddings: int,
    num_features: int,
    config: Config,
    mesh: Mesh,
    cast_input_dtype: None | DType = None,
    dtype: DType = jnp.float32,
    attend_dtype: None | DType = None,
    embedding_init: Initializer = default_embed_init,
    name: str | None = None,
):
  """Initializes the Embed NNX module and returns it as a Linen module.

  This function serves as a bridge to use the NNX-based `Embed` module within
  a Linen model. It wraps the `Embed` module using `nnx.bridge.to_linen`,
  making it compatible with the Linen API.

  Args:
    num_embeddings: The number of embeddings.
    num_features: The number of feature dimensions for each embedding.
    config: The model configuration.
    cast_input_dtype: The dtype to cast the input to, if any.
    dtype: The dtype of the embedding vectors.
    attend_dtype: The dtype for the `attend` method.
    embedding_init: The initializer for the embedding matrix.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `Embed` module.
  """
  return nnx_wrappers.to_linen(
      Embed,
      num_embeddings=num_embeddings,
      num_features=num_features,
      config=config,
      mesh=mesh,
      cast_input_dtype=cast_input_dtype,
      dtype=dtype,
      attend_dtype=attend_dtype,
      embedding_init=embedding_init,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class Embed(nnx.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors."""

  def __init__(
      self,
      num_embeddings: int,
      num_features: int,
      config: Config,
      mesh: Mesh,
      cast_input_dtype: None | DType = None,
      dtype: DType = jnp.float32,
      attend_dtype: None | DType = None,
      embedding_init: Initializer = default_embed_init,
      *,
      # Not used in Embed but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs,
  ):
    """Initializes the Embed module.

    Args:
      num_embeddings: The number of embeddings.
      num_features: The number of feature dimensions for each embedding.
      config: The model configuration.
      cast_input_dtype: The dtype to cast the input to, if any.
      dtype: The dtype of the embedding vectors.
      attend_dtype: The dtype for the `attend` method.
      embedding_init: The initializer for the embedding matrix.
      rngs: The random number generators for initialization.
    """
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.config = config
    self.mesh = mesh
    self.cast_input_dtype = cast_input_dtype
    self.dtype = dtype
    self.attend_dtype = attend_dtype

    self.embedding = nnx.Param(
        embedding_init(rngs.params(), (self.num_embeddings, self.num_features), self.config.weight_dtype),
        sharding=("vocab", "embed"),
    )

  def __call__(self, inputs: Array, model_mode: str = MODEL_MODE_TRAIN) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `num_features` dimension appended.
    """
    cfg = self.config
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")

    embedding = jnp.asarray(
        _maybe_move_embedding_to_device(self.embedding.value, self.config),
        self.dtype,
    )

    output_axis_names = (
        ("activation_embed_and_logits_batch", "prefill_activation_length", "activation_embed")
        if model_mode == MODEL_MODE_PREFILL
        else ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_embed")
    )
    out_pspec = nn.logical_to_mesh_axes(output_axis_names)

    out_sharding = NamedSharding(self.mesh, out_pspec) if self.config.shard_mode == ShardMode.EXPLICIT else None

    if cfg.use_iota_embed:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, embedding, out_sharding=out_sharding)
    else:
      output = embedding.at[inputs].get(out_sharding=out_sharding)

    return output

  def attend(self, query: Array, out_sharding: NamedSharding | None = None) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `num_features` of the
        embedding.
      out_sharding: NamedSharding object indicating how the output gets sharded

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    embedding = self.embedding.value
    attend_dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return attend_on_embedding(query, embedding, attend_dtype, self.config, out_sharding)


def attend_on_embedding(
    query: Array,
    embedding_table: Array,
    attend_dtype: DType,
    config: Config,
    out_sharding: NamedSharding | None = None,
) -> Array:
  """Attend over an embedding table using a query array.

  TODO: Remove this method when Embed bridge to Linen is no longer needed

  Args:
    query: An array with a last dimension equal to the feature depth of the embedding.
    embedding_table: The embedding table to attend over.
    attend_dtype: The data type for the attention computation.
    config: The model configuration, used to check for parameter offloading.
    out_sharding: NamedSharding object indicating the output sharding

  Returns:
    An array with a final dimension equal to `num_embeddings`, corresponding to the
    batched inner-product of the query vectors against each embedding.
  """
  # out_sharding must be None under auto shard_mode
  if config.shard_mode != ShardMode.EXPLICIT:
    out_sharding = None
  embedding_table = _maybe_move_embedding_to_device(embedding_table, config)
  return jnp.dot(
      query,
      jnp.asarray(embedding_table, jnp.bfloat16).T,
      preferred_element_type=attend_dtype,
      out_sharding=out_sharding,
  )


def rotary_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
):
  """Initializes the RotaryEmbedding module and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    name: Name of the Linen module.
  """
  return nnx_wrappers.to_linen(
      RotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class RotaryEmbedding(nnx.Module):
  """Rotary Position Embedding."""

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      # Not used in RotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rope_linear_scaling_factor: float = 1.0,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the RotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.embedding_dims = embedding_dims
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.rope_linear_scaling_factor = rope_linear_scaling_factor

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def timescale(self):
    """Returns the timescale for the rotary embedding."""
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
    if self.rope_linear_scaling_factor != 1.0:
      timescale = timescale * self.rope_linear_scaling_factor
    return timescale

  def __call__(
      self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      inputs: jax.Array,
      position: None | jax.Array = None,
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


def llama_rotary_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    use_scale: bool = True,
    name: str | None = None,
):
  """Initializes the LLaMARotaryEmbedding module and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index. Determines the periodicity of
      the added signal.
    max_timescale: End of the geometric index. Determines the frequency of the
      added signal.
    embedding_dims: Dimension of the embedding to be generated.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    use_scale: Whether to apply LLaMA3.1 scaling factor.
    name: Name of the Linen module.
  """
  return nnx_wrappers.to_linen(
      LLaMARotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      use_scale=use_scale,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


class LLaMARotaryEmbedding(RotaryEmbedding):
  """LLaMA variant of ROPE."""

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      use_scale: bool = True,
      # Not used in LLaMARotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the LLaMARotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
      cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
      fprop_dtype: The dtype of the output.
      use_scale: Whether to apply LLaMA3.1 scaling factor.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    super().__init__(
        min_timescale=min_timescale,
        max_timescale=max_timescale,
        embedding_dims=embedding_dims,
        cast_as_fprop_dtype=cast_as_fprop_dtype,
        fprop_dtype=fprop_dtype,
        rngs=rngs,
    )

    # LLaMA3.1 ROPE scaling, see the original pytorch implementation:
    # https://github.com/meta-llama/llama-models/blob/301ca3a2b3b10e94ddcd1fdd2c57e52f812e1cac/models/llama3/reference_impl/model.py#L45C5-L45C18
    self.use_scale = use_scale

  @property
  def timescale(self):
    half_embedding_dim = self.embedding_dims // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
    fraction = jnp.repeat(fraction, 2)
    timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction

    # Apply scaling factor if enabled
    if self.use_scale:
      timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(1.0 / timescale)

    # Expand timescale dimensions for broadcasting
    return timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

  def _apply_scaling_factor(self, freq):
    """apply scaling factor to rotary position embedding."""
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

  def __call__(self, inputs: jax.Array, position: None | jax.Array = None) -> jax.Array:
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
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

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


def yarn_rotary_embedding_as_linen(
    *,
    embedding_dims: int,
    max_position_embeddings: int = 4096 * 4,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32,
    beta_slow: float = 1,
    rope_theta: float = 10000.0,
    rope_factor: float = 40,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
    interleave: bool = True,
    truncate: bool = True,
    attention_scaling: bool = False,
):
  """Initializes the YarnRotaryEmbedding module and returns it as a Linen module.

  Args:
    embedding_dims: The dimension of the embeddings.
    max_position_embeddings: The maximum number of positions.
    original_max_position_embeddings: The original maximum number of positions.
    beta_fast: The fast beta parameter for YaRN.
    beta_slow: The slow beta parameter for YaRN.
    rope_theta: The base for the rotary frequencies.
    rope_factor: The scaling factor for RoPE.
    cast_as_fprop_dtype: Whether to cast the output to `fprop_dtype`.
    fprop_dtype: The forward pass dtype.
    name: The name of the module.
  """
  return nnx_wrappers.to_linen(
      YarnRotaryEmbedding,
      embedding_dims=embedding_dims,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_theta=rope_theta,
      rope_factor=rope_factor,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
      interleave=interleave,
      truncate=truncate,
      attention_scaling=attention_scaling,
  )


class YarnRotaryEmbedding(nnx.Module):
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
    cast_as_fprop_dtype: Whether to cast the output to `fprop_dtype`.
    fprop_dtype: The forward pass dtype.
    rope_interleave: Whether complex representation is interleaved or concatenated.
    rope_truncate: Whether or not to floor lower bound and ceil upper bound for correction range.
    rope_attention_scaling: Whether or not to scale the rotary embedding output.
    rngs: rng keys passed in by nnx.bridge.to_linen.
  """

  def __init__(
      self,
      embedding_dims: int,
      max_position_embeddings: int = 4096 * 4,
      original_max_position_embeddings: int = 4096,
      beta_fast: float = 32,
      beta_slow: float = 1,
      rope_theta: float = 10000.0,
      rope_factor: float = 40,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      interleave=True,
      truncate=True,
      attention_scaling=False,
      # Not used in YarnRotaryEmbedding but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the YarnRotaryEmbedding module."""
    self.embedding_dims = embedding_dims
    self.max_position_embeddings = max_position_embeddings
    self.original_max_position_embeddings = original_max_position_embeddings
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    self.rope_theta = rope_theta
    self.rope_factor = rope_factor
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.interleave = interleave
    self.truncate = truncate
    self.attention_scaling = attention_scaling

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def freqs_cis(self):
    """Frequencies for rotary embedding."""
    half_dim = self.embedding_dims // 2
    # Compute base frequencies for each (even-indexed) dimension.
    # (Note: We use jnp.arange with float32 for precision.)
    freqs = 1.0 / (self.rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / self.embedding_dims))

    low, high = self._find_correction_range(
        self.beta_fast,
        self.beta_slow,
        self.embedding_dims,
        self.rope_theta,
        self.original_max_position_embeddings,
        self.truncate,
    )
    smooth = 1 - self._linear_ramp_factor(low, high, half_dim)
    # The corrected frequency is a weighted mix of the scaled and base values.
    freqs = freqs / self.rope_factor * (1 - smooth) + freqs * smooth

    # Precompute frequencies for all positions by taking the outer product.
    t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
    # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
    freqs = jnp.outer(t, freqs)

    # Compute the complex “cis” values: exp(i * theta).
    return jnp.exp(1j * freqs)  # shape [max_position_embeddings, half_dim]

  def _find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    """Compute the correction dimension for a given number of rotations."""
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

  def _find_correction_range(
      self, low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int, truncate: bool
  ):
    """Computes the range of correction dimensions for rotary positional embeddings.

    Args:
        low_rot (float): Lower bound for the number of rotations.
        high_rot (float): Upper bound for the number of rotations.
        dim (int): Dimensionality of the embedding space.
        base (float): Base value for the exponential computation.
        max_position_embeddings (int): Maximum sequence length.
        truncate (bool): Whether to floor lower bound and ceil upper bound.

    Returns:
        tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
    """
    low = self._find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = self._find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
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

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
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
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Determine positions if not provided
    if position is None:
      seq_length = inputs.shape[1]
      position = jnp.arange(seq_length, dtype=jnp.int32)[jnp.newaxis, :]
    else:
      position = position.astype(jnp.int32)

    # Lookup the precomputed frequencies using the position indices.
    # self.freqs_cis has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
    # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
    freqs = jnp.take(self.freqs_cis, position, axis=0)  # shape: [B, S, half_dim]
    freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]

    if self.interleave:
      # Inputs with interleaved format [real1, img1, real2, img2, ...] at last dimension
      # Convert the last dimension into a complex representation.
      # First reshape so that each pair of numbers represents the real and imaginary parts.
      B, S, N, H = inputs.shape
      half_dim = H // 2
      inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
      first_half, second_half = inputs_reshaped[..., 0], inputs_reshaped[..., 1]
    else:
      # Inputs with concatenated format [real1, real2, ..., img1, img2, ...] at last dimension
      first_half, second_half = jnp.split(inputs, 2, axis=-1)

    inputs_complex = first_half + 1j * second_half  # shape: [B, S, N, half_dim]
    # Apply the rotary transformation via complex multiplication.
    rotated = inputs_complex * freqs  # shape: [B, S, N, half_dim]
    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    # [real1, real2, ..., img1, img2, ...]
    output = jnp.concatenate([jnp.real(rotated), jnp.imag(rotated)], axis=-1)

    if self.attention_scaling:
      attention_scaling = 1.0 if self.rope_factor <= 1 else (0.1 * math.log(self.rope_factor) + 1.0)
      output = output * attention_scaling

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)
    return output


def positional_embedding_as_linen(*, embedding_dims: int, max_wavelength: int = _MAX_WAVELENGTH):
  """Initializes the PositionalEmbedding module and returns it as a Linen module.

  Args:
    embedding_dims: The dimension of the embeddings.
    max_wavelength: The maximum wavelength for the sinusoidal positional embeddings.
  """
  return nnx_wrappers.to_linen(
      PositionalEmbedding,
      embedding_dims=embedding_dims,
      max_wavelength=max_wavelength,
      metadata_fn=variable_to_logically_partitioned,
  )


@dataclasses.dataclass(repr=False)
class PositionalEmbedding(nnx.Module):
  """A layer that adds sinusoidal positional embeddings to the input.

  Attributes:
    embedding_dims: The dimension of the embeddings.
    max_wavelength: The maximum wavelength for the sinusoidal positional embeddings.
    rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module.
  """

  embedding_dims: int
  max_wavelength: int = _MAX_WAVELENGTH

  rngs: nnx.Rngs = None  # Not used in PositionalEmbedding but passed in by nnx.bridge.to_linen

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


def llama_vision_rotary_embedding_as_linen(
    *,
    image_size: int,
    patch_size: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float = 10000.0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    name: str | None = None,
):
  """Initializes the LlamaVisionRotaryEmbedding module and returns it as a Linen module.

  Args:
    image_size: The size of the input image.
    patch_size: The size of the image patches.
    hidden_size: The size of the hidden dimension.
    num_attention_heads: The number of attention heads.
    rope_theta: The base theta value for the frequency computation.
    cast_as_fprop_dtype: Whether to cast the output to the fprop dtype.
    fprop_dtype: The dtype of the output.
    name: The name of the Linen module.
  """
  return nnx_wrappers.to_linen(
      LlamaVisionRotaryEmbedding,
      image_size=image_size,
      patch_size=patch_size,
      hidden_size=hidden_size,
      num_attention_heads=num_attention_heads,
      rope_theta=rope_theta,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )


@dataclasses.dataclass(repr=False)
class LlamaVisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Llama4 vision encoder.

  Based on Pytorch Reference
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
  This implementation follows the Llama4 vision encoder's rotary embedding approach,
  which uses 2D coordinates (x, y) to generate rotary position embeddings.

  Attributes:
    image_size: int size of the input image
    patch_size: int size of the image patches
    hidden_size: int size of the hidden dimension
    num_attention_heads: int number of attention heads
    rope_theta: float = 10000.0 base theta value for the frequency computation
    cast_as_fprop_dtype: bool = True whether to cast the output to the fprop dtype
    fprop_dtype: DType = jnp.bfloat16 the dtype of the output
    rngs: RNG state passed in by nnx.bridge.to_linen, not used in this module.
  Returns:
    jax.Array of shape [batch_size_times_tiles, num_patches_incl_cls, num_heads, head_dim]
    where vision rotary position embeddings are applied.
  """

  image_size: int
  patch_size: int
  hidden_size: int
  num_attention_heads: int
  rope_theta: float = 10000.0
  cast_as_fprop_dtype: bool = True
  fprop_dtype: DType = jnp.bfloat16
  # Not used in LlamaVisionRotaryEmbedding but passed in by nnx.bridge.to_linen.
  # TODO: Remove when bridge no longer needed
  rngs: nnx.Rngs = None

  @property
  def freqs_cis(self):
    """Frequencies for rotary embedding."""
    idx = self.image_size // self.patch_size
    img_idx = jnp.arange(idx**2, dtype=jnp.int32).reshape(idx**2, 1)
    img_idx = jnp.concatenate([img_idx, img_idx[:1]], axis=0)
    img_idx = img_idx.at[-1, -1].set(-2)  # ID_CLS_TOKEN

    # Get 2D coordinates
    frequencies_x = img_idx % idx  # x coordinates
    frequencies_y = img_idx // idx  # y coordinates

    # Compute frequency dimensions
    freq_dim = self.hidden_size // self.num_attention_heads // 2
    rope_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, freq_dim, 2)[: (freq_dim // 2)].astype(jnp.float32) / freq_dim))

    # Compute frequencies for x and y coordinates
    freqs_x = (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
    freqs_y = (frequencies_y + 1)[..., None] * rope_freq[None, None, :]

    # Interleave x and y frequencies
    freqs_x = jnp.repeat(freqs_x, 2, axis=-1)
    freqs_y = jnp.repeat(freqs_y, 2, axis=-1)

    # Combine frequencies
    freqs = jnp.concatenate([freqs_x, freqs_y], axis=-1).astype(jnp.float32)
    freqs = freqs[..., ::2]

    # Mask out invalid positions
    freqs = jnp.where(img_idx.reshape(-1, 1, 1) < 0, 0, freqs)
    # Convert to complex representation
    return jnp.exp(1j * freqs)

  def __call__(self, inputs: Array, position: None | Array = None) -> Array:
    """Applies rotary embeddings to the input tensor for Llama4 vision encoder.

    Args:
      inputs: Input tensor of shape [batch_size_times_tiles, num_patches_incl_cls, num_heads, head_dim]

    Returns:
      Tensor with rotary embeddings applied, maintaining the same shape as input.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          """Input is assumed to be a rank 4 tensor of shape [batch_size_times_tiles, num_patches_incl_cls, 
          num_heads, head_dim]."""
      )

    # Reshape inputs to complex representation
    B, S, N, H = inputs.shape
    half_dim = H // 2

    # Convert the last dimension into a complex representation.
    # First reshape so that each pair of numbers represents the real and imaginary parts.
    inputs_reshaped = inputs.reshape(B, S, N, half_dim, 2)
    inputs_complex = inputs_reshaped[..., 0] + 1j * inputs_reshaped[..., 1]

    # Reshape freqs_ci for broadcasting
    freqs_ci = self.freqs_cis[jnp.newaxis, :, :, :]

    # Apply rotary transformation
    rotated = inputs_complex * freqs_ci

    # Convert the complex result back to a real tensor.
    # Split the complex number into its real and imaginary parts.
    rotated_real = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
    output = rotated_real.reshape(B, S, N, H)

    if self.cast_as_fprop_dtype:
      output = output.astype(self.fprop_dtype)

    return output


class Qwen3OmniMoeThinkerTextRotaryEmbedding(nnx.Module):
  """Multi-dimensional Rotary Position Embedding (MRoPE) for Qwen3-Omni Thinker.

  This implements MRoPE which extends standard RoPE to handle 3D position IDs
  (temporal, height, width) for multimodal sequences containing text and vision tokens.

  For text-only sequences, it uses standard 2D position IDs.
  For sequences with vision tokens, it uses 3D position IDs where:
    - Dimension 0: Temporal position
    - Dimension 1: Height position (spatial)
    - Dimension 2: Width position (spatial)

  The implementation uses an interleaved pattern that reorganizes frequency
  components from chunked [TTT...HHH...WWW] to interleaved [THTHWHTHW...].
  """

  def __init__(
      self,
      min_timescale: int,
      max_timescale: int,
      embedding_dims: int = 0,
      cast_as_fprop_dtype: bool = True,
      fprop_dtype: DType = jnp.bfloat16,
      mrope_section: tuple[int, int, int] | None = None,
      rngs: nnx.Rngs = None,
  ):
    """Initializes the Qwen3OmniMoeThinkerTextRotaryEmbedding module.

    Args:
      min_timescale: Start of the geometric index (typically 1).
      max_timescale: End of the geometric index (rope_theta, e.g., 1000000).
      embedding_dims: Dimension of the embedding (head_dim).
      cast_as_fprop_dtype: Whether to cast output to fprop dtype.
      fprop_dtype: The dtype of the output.
      mrope_section: Tuple of (temporal_dim, height_dim, width_dim) for MRoPE.
                     Defaults to [24, 20, 20] if None.
      rngs: rng keys passed in by nnx.bridge.to_linen.
    """
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.embedding_dims = embedding_dims
    self.cast_as_fprop_dtype = cast_as_fprop_dtype
    self.fprop_dtype = fprop_dtype
    self.mrope_section = mrope_section if mrope_section is not None else (24, 20, 20)

    if self.embedding_dims % 2:
      raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

  @property
  def inv_freq(self):
    """Computes the inverse frequencies for RoPE."""
    # Standard RoPE inverse frequency computation
    # inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    fraction = jnp.arange(0, self.embedding_dims, 2, dtype=jnp.float32) / self.embedding_dims
    inv_freq = 1.0 / (self.max_timescale ** fraction)
    return inv_freq

  def _apply_interleaved_mrope(self, freqs: jax.Array) -> jax.Array:
    """Apply interleaved MRoPE pattern to 3D rotary embeddings.

    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...], preserving frequency continuity.

    Args:
      freqs: Shape (3, batch, seq_len, head_dim // 2)
        Dimension 0: temporal frequencies
        Dimension 1: height frequencies
        Dimension 2: width frequencies

    Returns:
      freqs_t: Shape (batch, seq_len, head_dim // 2) with interleaved pattern
    """
    # Start with temporal frequencies (dimension 0)
    freqs_t = freqs[0]  # (batch, seq_len, head_dim // 2)

    # Create interleaved pattern
    # For each spatial dimension (H, W), place frequencies at positions:
    # offset=1 for H, offset=2 for W, with stride=3
    for dim_idx, offset in enumerate([1, 2], start=1):  # H=1, W=2
      section_size = self.mrope_section[dim_idx] * 3  # Total positions for this dimension
      # Select positions with stride 3, starting at offset
      # Use slice syntax to match PyTorch behavior
      idx = slice(offset, section_size, 3)
      # Replace those positions with the corresponding spatial frequencies
      freqs_t = freqs_t.at[..., idx].set(freqs[dim_idx, ..., idx])

    return freqs_t

  def __call__(
      self,
      inputs: jax.Array,
      position: jax.Array,
  ) -> jax.Array:
    """Generates rotary position embeddings for multimodal sequences.

    Args:
      inputs: Input tensor of shape [batch, sequence, heads, head_dim].
      position: Position IDs with shape:
        - [batch, sequence] for text-only (2D)
        - [3, batch, sequence] for multimodal with vision (3D)
          where dim 0 = temporal, dim 1 = height, dim 2 = width

    Returns:
      Tensor of shape [batch, sequence, heads, head_dim] with RoPE applied.
    """
    if len(inputs.shape) != 4:
      raise ValueError(
          "Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, head_dim]."
      )
    if self.embedding_dims != inputs.shape[3]:
      raise ValueError(
          "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
      )

    # Handle both 2D (text-only) and 3D (multimodal) position IDs
    if position.ndim == 2:
      # Text-only: expand (batch, seq) -> (3, batch, seq) with same positions
      position = jnp.broadcast_to(position[jnp.newaxis, ...], (3,) + position.shape)
    elif position.ndim != 3 or position.shape[0] != 3:
      raise ValueError(
          f"Position IDs must be 2D (batch, seq) or 3D (3, batch, seq), got shape {position.shape}"
      )

    # Compute frequencies: (3, batch, seq, 1) @ (head_dim // 2, 1) -> (3, batch, seq, head_dim // 2)
    inv_freq_expanded = self.inv_freq[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (1, 1, 1, head_dim//2)
    position_expanded = position[..., jnp.newaxis]  # (3, batch, seq, 1)
    freqs = position_expanded * inv_freq_expanded  # (3, batch, seq, head_dim//2)

    # Apply interleaved MRoPE pattern for 3D positions
    freqs = self._apply_interleaved_mrope(freqs)  # (batch, seq, head_dim//2)

    # Compute sin and cos
    # Concatenate to get full head_dim: (batch, seq, head_dim//2) -> (batch, seq, head_dim)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # Duplicate for both halves
    cos_emb = jnp.cos(emb)  # (batch, seq, head_dim)
    sin_emb = jnp.sin(emb)  # (batch, seq, head_dim)

    # Expand for heads dimension: (batch, seq, head_dim) -> (batch, seq, 1, head_dim)
    cos_emb = cos_emb[:, :, jnp.newaxis, :]
    sin_emb = sin_emb[:, :, jnp.newaxis, :]

    # Apply rotary transformation using rotate_half
    # PyTorch: q_embed = (q * cos) + (rotate_half(q) * sin)
    # where rotate_half(x) splits x in half, negates second half, and concatenates as (-x2, x1)
    def rotate_half(x):
      """Rotates half the hidden dims of the input."""
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return jnp.concatenate([-x2, x1], axis=-1)

    x_out = (inputs * cos_emb) + (rotate_half(inputs) * sin_emb)

    if self.cast_as_fprop_dtype:
      x_out = x_out.astype(self.fprop_dtype)

    return x_out


def qwen3_omni_mrope_embedding_as_linen(
    *,
    min_timescale: int,
    max_timescale: int,
    embedding_dims: int = 0,
    cast_as_fprop_dtype: bool = True,
    fprop_dtype: DType = jnp.bfloat16,
    mrope_section: tuple[int, int, int] | None = None,
    name: str | None = None,
):
  """Initializes Qwen3OmniMoeThinkerTextRotaryEmbedding and returns it as a Linen module.

  Args:
    min_timescale: Start of the geometric index.
    max_timescale: End of the geometric index (rope_theta).
    embedding_dims: Dimension of the embedding (head_dim).
    cast_as_fprop_dtype: Whether to cast output to fprop dtype.
    fprop_dtype: The dtype of the output.
    mrope_section: Tuple of (temporal_dim, height_dim, width_dim) for MRoPE.
    name: Name of the Linen module.
  """
  return nnx_wrappers.to_linen(
      Qwen3OmniMoeThinkerTextRotaryEmbedding,
      min_timescale=min_timescale,
      max_timescale=max_timescale,
      embedding_dims=embedding_dims,
      cast_as_fprop_dtype=cast_as_fprop_dtype,
      fprop_dtype=fprop_dtype,
      mrope_section=mrope_section,
      metadata_fn=variable_to_logically_partitioned,
      name=name,
  )
