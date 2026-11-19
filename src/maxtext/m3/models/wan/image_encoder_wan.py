# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan 2.1 CLIP vision image encoder (Flax NNX)."""

import dataclasses
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.m3.core.sharding import Sharding

_ACTIVATIONS = {
    "gelu": lambda x: jax.nn.gelu(x, approximate=False),
    "quick_gelu": lambda x: x * jax.nn.sigmoid(1.702 * x),
    "gelu_new": lambda x: jax.nn.gelu(x, approximate=True),
    "gelu_pytorch_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
}


@dataclasses.dataclass
class CLIPVisionModelOutput:
  """Output of `CLIPVisionModel`, matching HuggingFace `CLIPVisionModelOutput`."""

  last_hidden_state: jax.Array
  pooler_output: jax.Array
  image_embeds: jax.Array
  hidden_states: Optional[tuple[jax.Array, ...]] = None


class CLIPVisionEmbeddings(nnx.Module):
  """Patch + class token + learned positional embeddings for CLIP vision model."""

  def __init__(
      self,
      hidden_size: int,
      num_channels: int,
      image_size: int,
      patch_size: int,
      *,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: Optional[jax.lax.Precision],
      rngs: nnx.Rngs,
  ):
    self.embed_dim = hidden_size
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.dtype = dtype
    self.num_patches = (image_size // patch_size) ** 2
    self.num_positions = self.num_patches + 1

    self.class_embedding = nnx.Param(jnp.zeros((hidden_size,), dtype=weights_dtype))
    self.patch_embedding = nnx.Conv(
        in_features=num_channels,
        out_features=hidden_size,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="VALID",
        use_bias=False,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )
    self.position_embedding = nnx.Embed(
        num_embeddings=self.num_positions,
        features=hidden_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        rngs=rngs,
    )

  def __call__(self, pixel_values: jax.Array) -> jax.Array:
    batch_size = pixel_values.shape[0]
    # Convert NCHW (from CLIPImageProcessor) to NHWC for Flax Conv.
    if (
        pixel_values.ndim == 4
        and pixel_values.shape[1] == self.num_channels
        and pixel_values.shape[-1] != self.num_channels
    ):
      pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
    patch_embeds = self.patch_embedding(pixel_values.astype(self.dtype))
    patch_embeds = patch_embeds.reshape(batch_size, -1, self.embed_dim)
    class_embeds = jnp.broadcast_to(
        self.class_embedding[...].astype(patch_embeds.dtype)[None, None, :],
        (batch_size, 1, self.embed_dim),
    )
    embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
    position_ids = jnp.arange(self.num_positions, dtype=jnp.int32)[None, :]
    embeddings = embeddings + self.position_embedding(position_ids).astype(embeddings.dtype)
    return embeddings


class CLIPAttention(nnx.Module):
  """Multi-headed bidirectional self-attention for CLIP vision encoder."""

  def __init__(
      self,
      hidden_size: int,
      num_attention_heads: int,
      *,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: Optional[jax.lax.Precision],
      rngs: nnx.Rngs,
  ):
    self.embed_dim = hidden_size
    self.num_heads = num_attention_heads
    self.head_dim = hidden_size // num_attention_heads
    self.scale = self.head_dim**-0.5
    self.dtype = dtype
    self.precision = precision

    linear_kw = {
        "in_features": hidden_size,
        "out_features": hidden_size,
        "use_bias": True,
        "dtype": dtype,
        "param_dtype": weights_dtype,
        "precision": precision,
        "rngs": rngs,
    }
    self.q_proj = nnx.Linear(**linear_kw)
    self.k_proj = nnx.Linear(**linear_kw)
    self.v_proj = nnx.Linear(**linear_kw)
    self.out_proj = nnx.Linear(**linear_kw)

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    batch_size, seq_len, _ = hidden_states.shape
    queries = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    keys = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    values = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

    attn_weights = jnp.einsum("bqhd,bkhd->bhqk", queries, keys, precision=self.precision) * self.scale
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(values.dtype)
    attn_output = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, values, precision=self.precision)
    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
    return self.out_proj(attn_output)


class CLIPMLP(nnx.Module):
  """Two-layer MLP for CLIP vision encoder."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      hidden_act: str,
      *,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: Optional[jax.lax.Precision],
      rngs: nnx.Rngs,
  ):
    if hidden_act not in _ACTIVATIONS:
      raise ValueError(f"Unsupported CLIP activation {hidden_act!r}; supported: {sorted(_ACTIVATIONS)}")
    self.activation_fn = _ACTIVATIONS[hidden_act]
    self.fc1 = nnx.Linear(
        hidden_size,
        intermediate_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )
    self.fc2 = nnx.Linear(
        intermediate_size,
        hidden_size,
        use_bias=True,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    return self.fc2(hidden_states)


class CLIPEncoderLayer(nnx.Module):
  """Pre-norm Transformer encoder layer for CLIP vision model."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      num_attention_heads: int,
      hidden_act: str,
      layer_norm_eps: float,
      *,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: Optional[jax.lax.Precision],
      rngs: nnx.Rngs,
  ):
    self.layer_norm1 = nnx.LayerNorm(
        hidden_size,
        epsilon=layer_norm_eps,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.self_attn = CLIPAttention(
        hidden_size,
        num_attention_heads,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )
    self.layer_norm2 = nnx.LayerNorm(
        hidden_size,
        epsilon=layer_norm_eps,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.mlp = CLIPMLP(
        hidden_size,
        intermediate_size,
        hidden_act,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )

  def __call__(self, hidden_states: jax.Array) -> jax.Array:
    norm1_out = self.layer_norm1(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
    hidden_states = hidden_states + self.self_attn(norm1_out)
    norm2_out = self.layer_norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
    hidden_states = hidden_states + self.mlp(norm2_out)
    return hidden_states


class CLIPEncoder(nnx.Module):
  """Stack of `CLIPEncoderLayer` blocks."""

  def __init__(
      self,
      num_hidden_layers: int,
      hidden_size: int,
      intermediate_size: int,
      num_attention_heads: int,
      hidden_act: str,
      layer_norm_eps: float,
      *,
      dtype: jnp.dtype,
      weights_dtype: jnp.dtype,
      precision: Optional[jax.lax.Precision],
      rngs: nnx.Rngs,
  ):
    self.layers = nnx.data(
        [
            CLIPEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
                rngs=rngs,
            )
            for _ in range(num_hidden_layers)
        ]
    )

  def __call__(
      self, hidden_states: jax.Array, output_hidden_states: bool = False
  ) -> tuple[jax.Array, Optional[tuple[jax.Array, ...]]]:
    encoder_states = (hidden_states,) if output_hidden_states else None
    for layer in self.layers:
      hidden_states = layer(hidden_states)
      if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)
    return hidden_states, encoder_states


class CLIPVisionModel(nnx.Module):
  """Flax NNX port of `transformers.CLIPVisionModelWithProjection` (ViT-H/14 in Wan 2.1 I2V).

  The constructor arguments match `image_encoder/config.json` (`CLIPVisionConfig`), so
  `CLIPVisionModel(**config, rngs=...)` accepts the config stored by the checkpoint converter.
  `__call__(pixel_values, output_hidden_states=True)` returns `CLIPVisionModelOutput` where
  `hidden_states[-2]` is the penultimate layer's hidden state `[batch, 257, hidden_size]` used by Wan 2.1 I2V.
  """

  def __init__(
      self,
      hidden_size: int = 1280,
      intermediate_size: int = 5120,
      projection_dim: int = 1024,
      num_hidden_layers: int = 32,
      num_attention_heads: int = 16,
      num_channels: int = 3,
      image_size: int = 224,
      patch_size: int = 14,
      hidden_act: str = "gelu",
      layer_norm_eps: float = 1e-5,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.projection_dim = projection_dim
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.dtype = jnp.dtype(dtype)
    self.sharding = sharding
    weights_dtype = jnp.dtype(weights_dtype)

    self.embeddings = CLIPVisionEmbeddings(
        hidden_size=hidden_size,
        num_channels=num_channels,
        image_size=image_size,
        patch_size=patch_size,
        dtype=self.dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )
    self.pre_layrnorm = nnx.LayerNorm(
        hidden_size,
        epsilon=layer_norm_eps,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.encoder = CLIPEncoder(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        dtype=self.dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )
    self.post_layernorm = nnx.LayerNorm(
        hidden_size,
        epsilon=layer_norm_eps,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        rngs=rngs,
    )
    self.visual_projection = nnx.Linear(
        hidden_size,
        projection_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=weights_dtype,
        precision=precision,
        rngs=rngs,
    )

  def __call__(
      self,
      pixel_values: jax.Array,
      output_hidden_states: bool = False,
  ) -> CLIPVisionModelOutput:
    hidden_states = self.embeddings(pixel_values)
    hidden_states = self.pre_layrnorm(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
    last_hidden_state, all_hidden_states = self.encoder(hidden_states, output_hidden_states=output_hidden_states)
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output.astype(jnp.float32)).astype(last_hidden_state.dtype)
    image_embeds = self.visual_projection(pooled_output)
    return CLIPVisionModelOutput(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        image_embeds=image_embeds,
        hidden_states=all_hidden_states,
    )
