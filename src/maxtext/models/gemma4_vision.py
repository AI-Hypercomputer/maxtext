# Copyright 2023–2026 Google LLC
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


"""Vision transformer implementation for Gemma4."""

from typing import cast
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config, AttentionType
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers
from maxtext.layers import normalizations


def factorized_posemb(posemb: jax.Array, positions_xy: jax.Array, precision) -> jax.Array:
  """Computes factorized position embedding from (x, y) coordinates.

  Args:
    posemb: The factorized position embedding parameters.
    positions_xy: The (x, y) coordinates for each patch.
    precision: The precision for the einsum operation.

  Returns:
    The computed position embeddings.
  """
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
  nan = jnp.logical_and(nan, positions_xy[..., None] != -1)
  pos_oh = jnp.where(nan, jnp.nan, one_hot)
  pe_seq = jnp.einsum("...is,sid->i...d", pos_oh, posemb, precision=precision).astype(posemb.dtype)
  return jnp.sum(pe_seq, axis=0)


def patchify(images: jax.Array, patch_size: int) -> tuple[jax.Array, jax.Array]:
  """Patchifies images and returns patches and (x, y) coordinates.

  Args:
    images: The input images of shape [..., H, W, C].
    patch_size: The size of each square patch.

  Returns:
    A tuple containing:
      - patches: The extracted patches of shape [..., num_patches, patch_size * patch_size * C].
      - positions_xy: The (x, y) coordinates of the top-left corner of each patch,
        of shape [..., num_patches, 2].
  """
  # Using jax.lax.reshape and transpose instead of einshape for simplicity
  *b, h, w, c = images.shape

  p = patch_size
  q = patch_size

  # ... h w c -> ... (h//p) p (w//q) q c
  reshaped_images = jax.lax.reshape(images, tuple(b) + (h // p, p, w // q, q, c))
  # ... (h//p) p (w//q) q c -> ... (h//p) (w//q) p q c
  transposed_images = jnp.transpose(
      reshaped_images, axes=tuple(range(len(b))) + (len(b), len(b) + 2, len(b) + 1, len(b) + 3, len(b) + 4)
  )
  # ... (h//p) (w//q) p q c -> ... ((h//p)*(w//q)) (p*q*c)
  patches = jax.lax.reshape(transposed_images, tuple(b) + ((h // p) * (w // q), p * q * c))

  xy = jnp.meshgrid(jnp.arange(w // patch_size), jnp.arange(h // patch_size))
  positions_xy = jnp.stack(xy, axis=-1)
  # yxc -> (yx)c
  positions_xy = jnp.reshape(positions_xy, (-1, 2))

  return patches, jnp.broadcast_to(positions_xy, tuple(b) + positions_xy.shape)


class VisionEntry(nnx.Module):
  """The vision entry layer."""

  def __init__(
      self,
      d_model: int,
      patch_size: int,
      pos_emb_shape_yx: tuple[int, int],
      normalize_input_range: bool = False,
      *,
      rngs: nnx.Rngs,
      dtype,
      weight_dtype,
      matmul_precision,
  ):
    self.d_model = d_model
    self.patch_size = patch_size
    self.pos_emb_shape_yx = pos_emb_shape_yx
    self.normalize_input_range = normalize_input_range
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.matmul_precision = matmul_precision

    self.input_projection = linears.DenseGeneral(
        in_features_shape=self.patch_size * self.patch_size * 3,
        out_features_shape=self.d_model,
        use_bias=False,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
        axis=-1,
        rngs=rngs,
    )

    assert self.pos_emb_shape_yx[-1] == 2, f"{self.pos_emb_shape_yx=}"

    pos_emb_init = nnx.initializers.normal(stddev=0.02)
    self.pos_emb_param = nnx.Param(
        pos_emb_init(
            rngs.params(),
            (self.pos_emb_shape_yx[0], self.pos_emb_shape_yx[1], self.d_model),
            jnp.float32,
        )
    )

  def __call__(
      self,
      images_or_patches: jax.Array,
      positions_xy: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    """Processes input images or patches and applies projection and position embeddings."""
    if positions_xy is None:
      # If positions_xy is not provided, we assume the input are images
      # in the format [..., H, W, C] and need to be patchified.
      patches, positions_xy = patchify(images_or_patches, self.patch_size)
    else:
      # If positions_xy is provided, we assume the input are already patches.
      patches = images_or_patches
      assert patches.ndim == 3, f"Expected patches to have 3 dimensions, but got {patches.ndim}"
      assert positions_xy is not None, "positions_xy must be provided when images_or_patches are already patchified"
      # Ensure positions_xy has the correct batch dimension
      if positions_xy.shape[0] == patches.shape[0]:
        pass  # positions_xy already has batch dimension
      elif positions_xy.ndim == 2:
        positions_xy = jnp.broadcast_to(positions_xy, (patches.shape[0],) + positions_xy.shape)
      else:
        raise ValueError(f"Unexpected positions_xy shape: {positions_xy.shape}")

    if self.normalize_input_range:
      patches = 2 * (patches - 0.5)

    x = self.input_projection(patches)

    pos_embed = factorized_posemb(cast(jax.Array, self.pos_emb_param.value), positions_xy, self.matmul_precision).astype(
        x.dtype
    )

    return x + pos_embed, positions_xy


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    rotary_fraction: float | None = None,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies multidimensional RoPE. Based on Gemma 4 implementation.

  Args:
    inputs: The input array to apply RoPE to.
    positions: The positional information. Can be 1D or ND.
    base_frequency: The base frequency for the sinusoidal functions.
    rotary_fraction: The fraction of the hidden dimension to apply RoPE to. If None,
      applies to the full dimension.
    scale_factor: A scale factor applied to the sinusoidal arguments.

  Returns:
    The input array with multidimensional RoPE applied.
  """

  # Internal _apply_rope logic
  def _apply_rope(x_in: jax.Array, pos: jax.Array, base_freq: int, scale: float) -> jax.Array:
    # x_in: [B, L, N, H]
    # pos: [B, L] or similar
    dim = x_in.shape[-1]
    half_dim = dim // 2
    fraction = 2 * jnp.arange(0, half_dim) / dim
    timescale = base_freq**fraction

    # position shape logic
    reshaped_pos = pos[..., jnp.newaxis, jnp.newaxis]
    sinusoid_inp = reshaped_pos / timescale

    sin_half = jnp.sin(sinusoid_inp).astype(x_in.dtype)
    cos_half = jnp.cos(sinusoid_inp).astype(x_in.dtype)

    sin = jnp.concatenate([sin_half, sin_half], axis=-1)
    cos = jnp.concatenate([cos_half, cos_half], axis=-1)

    x1, x2 = jnp.split(x_in, 2, axis=-1)
    rotated_x = jnp.concatenate((-x2, x1), axis=-1)

    return (x_in * cos) + (rotated_x * sin)

  if positions.ndim + 2 == inputs.ndim:
    if rotary_fraction is None or rotary_fraction == 1.0:
      return _apply_rope(
          inputs,
          positions,
          base_frequency,
          scale_factor,
      )
    dim_to_rope = int(rotary_fraction * inputs.shape[-1])
    if dim_to_rope == inputs.shape[-1]:
      return _apply_rope(
          inputs,
          positions,
          base_frequency,
          scale_factor,
      )
    if dim_to_rope == 0:
      return inputs
    x1 = inputs[..., :dim_to_rope]
    x2 = inputs[..., dim_to_rope:]
    x1 = _apply_rope(
        x1,
        positions,
        base_frequency,
        scale_factor,
    )
    return jnp.concatenate([x1, x2], axis=-1)

  ndim = positions.shape[-1]
  num_input_channels = inputs.shape[-1]
  num_rotated_channels = num_input_channels
  if rotary_fraction is not None:
    num_rotated_channels = int(round(num_rotated_channels * rotary_fraction))
  num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

  assert num_rotated_channels_per_dim > 0, f"Requirement not satisfied: 2 * {ndim=} <= {num_input_channels=}."

  split_points = [(k + 1) * num_rotated_channels_per_dim for k in range(ndim)]
  if rotary_fraction is None:
    split_points = split_points[:-1]
  assert all(
      isinstance(sp, int) for sp in split_points
  ), f"Expected all split points to be integers, but got {split_points}"
  x_parts = jnp.split(inputs, split_points, axis=-1)
  y_parts = [
      _apply_rope(
          x_parts[k],
          positions[..., k],
          base_frequency,
          scale_factor,
      )
      for k in range(ndim)
  ]

  if rotary_fraction is not None:
    y_parts.append(x_parts[-1])

  return jnp.concatenate(y_parts, axis=-1)


def avg_pool_by_positions(
    x: jax.Array,
    *,
    positions_xy: jax.Array,
    length: int,
    precision,
) -> tuple[jax.Array, jax.Array]:
  """Performs 2D spatial pooling according to patch positions.

  Args:
    x: The input features of shape [B, L, D].
    positions_xy: The (x, y) coordinates of each patch of shape [B, L, 2].
    length: The desired output sequence length after pooling.
    precision: The precision for the einsum operation.

  Returns:
    A tuple containing:
      - output: The pooled features of shape [B, length, D].
      - mask: A boolean mask indicating valid pooled positions.
  """
  k = max(1, int((x.shape[1] // length) ** 0.5))
  assert k * k * length == x.shape[1], f"Cannot pool {x.shape=} to {length=}"

  max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
  kernel_idxs = jnp.floor_divide(positions_xy, k)
  flat_kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
  weights = jax.nn.one_hot(flat_kernel_idx, length) / k**2
  output = jnp.einsum("bLl,bLd->bld", weights, x, precision=precision)
  mask = jnp.logical_not((weights == 0).all(axis=1))
  return output, mask


class VisionExit(nnx.Module):
  """Vision exit layer with scaling and optional spatial pooling."""

  def __init__(self, d_model: int, output_length: int | tuple[int, ...] = 256, *, rngs: nnx.Rngs, precision):
    self.d_model = d_model
    self.output_length = output_length
    self.precision = precision

  def _maybe_downsample(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    """Downsamples the vision features if required by the output length."""
    cur_length = x.shape[1]

    POSITIONS_PAD_VALUE = -1

    if cur_length == length:
      if positions_xy is None:
        mask = jnp.ones(x.shape[:-1], dtype=jnp.bool_)
      else:
        mask = jnp.logical_not((positions_xy == POSITIONS_PAD_VALUE).all(axis=-1))
      return x, mask

    if positions_xy is not None:
      x_pooled, mask = avg_pool_by_positions(x, positions_xy=positions_xy, length=length, precision=self.precision)
      return x_pooled, mask

    cur_width = int(cur_length**0.5)
    if cur_width**2 != cur_length:
      raise ValueError(f"x.shape[1]={cur_length} must be a perfect square.")

    output_width = int(length**0.5)
    if output_width**2 != length:
      raise ValueError(f"{length=} must be a perfect square.")

    if cur_width % output_width != 0:
      raise ValueError(f"{cur_width=} must be divisible by {output_width=}.")

    x_2d = x.reshape(x.shape[0], cur_width, cur_width, x.shape[-1])

    window = cur_width // output_width
    window_shape = (window, window)
    x_2d = nnx.avg_pool(x_2d, window_shape=window_shape, strides=window_shape)

    x_pooled = x_2d.reshape(x.shape[0], length, x.shape[-1])
    mask = jnp.ones(x_pooled.shape[:-1], dtype=jnp.bool_)
    return x_pooled, mask

  def _single_call(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      length: int,
  ) -> tuple[jax.Array, jax.Array | None]:
    """Processes the features for a single target length."""
    x, mask = self._maybe_downsample(x, positions_xy=positions_xy, length=length)

    x = x * jnp.sqrt(self.d_model)

    return x, mask

  def __call__(
      self,
      x: jax.Array,
      *,
      positions_xy: jax.Array | None = None,
      output_length_overrides: tuple[int, ...] | None = None,
  ) -> tuple[tuple[jax.Array, jax.Array | None], ...]:
    """Applies vision exit processing, optionally downsampling to requested output lengths."""
    lens = (self.output_length,) if isinstance(self.output_length, int) else self.output_length
    if output_length_overrides is not None:
      lens = output_length_overrides

    return tuple(self._single_call(x, positions_xy=positions_xy, length=length) for length in lens)


class Gemma4VisionRotaryEmbedding(nnx.Module):
  """Rotary position embedding for Gemma 4 vision."""

  def __init__(
      self,
      base_frequency: int,
      rotary_fraction: float | None = None,
      scale_factor: float = 1.0,
  ):
    self.base_frequency = base_frequency
    self.rotary_fraction = rotary_fraction
    self.scale_factor = scale_factor

  def __call__(self, inputs: jax.Array, positions: jax.Array) -> jax.Array:
    """Applies rotary position embeddings to the inputs."""
    return apply_multidimensional_rope(
        inputs,
        positions,
        base_frequency=self.base_frequency,
        rotary_fraction=self.rotary_fraction,
        scale_factor=self.scale_factor,
    )


class Gemma4Attention(attentions.Attention):
  """Gemma 4 specific Attention module."""

  def init_rotary_embedding(self) -> Gemma4VisionRotaryEmbedding:
    """Initializes the rotary position embedding module for Gemma 4 vision."""
    return Gemma4VisionRotaryEmbedding(
        base_frequency=self.config.rope_theta_for_vit if hasattr(self.config, "rope_theta_for_vit") else 100,
        rotary_fraction=None,  # Or assume it from config if available
    )


class Gemma4EncoderBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    # Standard RMSNorm
    self.pre_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.post_attention_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Attention with Gemma 4 specifics
    # We need dummy shapes for initialization
    batch_size = config.per_device_batch_size

    # Handle both square (int) and rectangular (list/tuple) image sizes
    if isinstance(config.image_size_for_vit, (list, tuple)):
      img_h, img_w = config.image_size_for_vit
      seq_len = (img_h // config.patch_size_for_vit) * (img_w // config.patch_size_for_vit)
    else:
      seq_len = (config.image_size_for_vit // config.patch_size_for_vit) ** 2

    dummy_shape = (batch_size, seq_len, config.hidden_size_for_vit)

    self.attention = Gemma4Attention(
        config=config,
        num_query_heads=config.num_attention_heads_for_vit,
        num_kv_heads=config.num_attention_heads_for_vit,
        head_dim=config.hidden_size_for_vit // config.num_attention_heads_for_vit,
        max_target_length=seq_len,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=dummy_shape,
        inputs_kv_shape=dummy_shape,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        dropout_rate=config.dropout_rate,
        attention_type=AttentionType.FULL,
        use_qk_norm=True,
        use_v_norm=True,
        query_pre_attn_scalar=1.0,
        is_vision=True,
        rngs=self.rngs,
    )

    self.pre_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.post_ffw_norm = normalizations.RMSNorm(
        num_features=config.hidden_size_for_vit,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    self.mlp = linears.MlpBlock(
        config=config,
        mesh=mesh,
        in_features=config.hidden_size_for_vit,
        intermediate_dim=config.intermediate_size_for_vit,
        activations=("gelu", "linear"),
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        intermediate_dropout_rate=config.dropout_rate,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array, positions: jax.Array | None = None, deterministic: bool = False) -> jax.Array:
    """Applies the encoder block (MHSA + MLP) to the inputs."""
    x_normed = self.pre_attention_norm(x)
    # Pass positions to attention for RoPE
    x_attn, _ = self.attention(x_normed, x_normed, inputs_positions=positions, deterministic=deterministic)
    x_attn = self.post_attention_norm(x_attn)
    x_after_attn = x_attn + x

    x_ffw_normed = self.pre_ffw_norm(x_after_attn)
    x_ffw = self.mlp(x_ffw_normed, deterministic=deterministic)
    x_ffw = self.post_ffw_norm(x_ffw)
    x_after_ffw = x_ffw + x_after_attn
    return x_after_ffw


class Gemma4VisionEncoderLayer(nnx.Module):
  """Gemma 4 Vision Encoder Layer."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    # Input Projection (VisionEntry)
    self.vision_entry = VisionEntry(
        d_model=config.hidden_size_for_vit,
        patch_size=config.patch_size_for_vit,
        pos_emb_shape_yx=(config.num_position_embeddings_for_vit, 2),
        normalize_input_range=True,
        rngs=self.rngs,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
    )

    # Encoder Blocks
    for i in range(config.num_hidden_layers_for_vit):
      layer = Gemma4EncoderBlock(config, mesh, rngs=self.rngs)
      # Register submodules for NNX
      setattr(self, f"layer_{i}", layer)

    # Vision Exit
    self.vision_exit = VisionExit(
        d_model=config.hidden_size_for_vit,
        output_length=config.vision_output_length,
        rngs=self.rngs,
        precision=config.matmul_precision,
    )
    self.std_bias = nnx.Param(
        nnx.initializers.zeros(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype), sharding=(None,)
    )
    self.std_scale = nnx.Param(
        nnx.initializers.ones(self.rngs.params(), (config.hidden_size_for_vit,), config.weight_dtype), sharding=(None,)
    )

  def __call__(self, inputs: jax.Array, deterministic: bool = False) -> jax.Array:
    """Applies the vision encoder layer."""
    if inputs.ndim == 4:
      inputs = jnp.expand_dims(inputs, 1)
    b, n, h, w, c = inputs.shape
    inputs_flat = jnp.reshape(inputs, (b * n, h, w, c))

    x, positions_xy = self.vision_entry(inputs_flat)

    for i in range(self.config.num_hidden_layers_for_vit):
      layer = getattr(self, f"layer_{i}")
      x = layer(x, positions=positions_xy, deterministic=deterministic)

    vision_exit_results = self.vision_exit(x, positions_xy=positions_xy)

    # Return embeddings from VisionExit tuple
    # vision_exit_results is a tuple of (embeddings, mask) tuples, one for each output length.
    # We take the first result.
    (embeddings, _) = vision_exit_results[0]

    embeddings = (embeddings - self.std_bias.value.astype(embeddings.dtype)) * self.std_scale.value.astype(
        embeddings.dtype
    )

    # Unflatten batch and num_images
    final_x = jnp.reshape(embeddings, (b, n, embeddings.shape[1], embeddings.shape[2]))

    return final_x


class Gemma4VisionProjector(nnx.Module):
  """A layer that projects image embeddings to the embedding space of the text encoder."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs

    self.projection = linears.DenseGeneral(
        in_features_shape=config.hidden_size_for_vit,
        out_features_shape=config.emb_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        matmul_precision=config.matmul_precision,
        kernel_axes=("embed", "mlp"),
        rngs=self.rngs,
    )

    self.norm = normalizations.RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype_mm,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    """Projects vision embeddings into the text embedding space."""
    x_normed = self.norm(x)
    x_projected = self.projection(x_normed)
    return x_projected


def gemma4_vision_encoder_as_linen(config: Config, mesh: Mesh) -> nn.Module:
  """Wraps the Gemma 4 Vision Encoder as a Linen module."""
  return nnx_wrappers.to_linen(
      Gemma4VisionEncoderLayer,
      config=config,
      mesh=mesh,
      name="Gemma4VisionEncoderLayer",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
