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

"""CLIP Text Encoder implementation in Flax Linen for Stable Diffusion."""

from typing import Any
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class QuickGELU(nn.Module):
  """QuickGELU activation function."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(1.702 * x)


class CLIPAttention(nn.Module):
  """CLIP Multi-Head Attention with causal mask support."""

  embed_dim: int = 768
  num_heads: int = 12
  dtype: Any = jnp.float32

  def setup(self):
    self.head_dim = self.embed_dim // self.num_heads
    self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, name="q_proj")
    self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, name="k_proj")
    self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, name="v_proj")
    self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, name="out_proj")

  def __call__(self, hidden_states: jnp.ndarray, causal_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    batch_size, seq_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
    k = self.k_proj(hidden_states).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
    v = self.v_proj(hidden_states).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))

    scale = 1.0 / np.sqrt(self.head_dim)
    attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
    if causal_mask is not None:
      attn_weights = attn_weights + causal_mask
    attn_probs = jax.nn.softmax(attn_weights, axis=-1)
    attn_output = jnp.matmul(attn_probs, v)
    attn_output = attn_output.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, self.embed_dim))
    return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
  """CLIP MLP block with QuickGELU activation."""

  intermediate_size: int = 3072
  hidden_size: int = 768
  dtype: Any = jnp.float32

  def setup(self):
    self.fc1 = nn.Dense(self.intermediate_size, dtype=self.dtype, name="fc1")
    self.fc2 = nn.Dense(self.hidden_size, dtype=self.dtype, name="fc2")
    self.act = QuickGELU()

  def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
    return self.fc2(self.act(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
  """CLIP Transformer Encoder Layer."""

  hidden_size: int = 768
  num_heads: int = 12
  intermediate_size: int = 3072
  dtype: Any = jnp.float32

  def setup(self):
    self.layer_norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="layer_norm1")
    self.self_attn = CLIPAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, dtype=self.dtype, name="self_attn")
    self.layer_norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="layer_norm2")
    self.mlp = CLIPMLP(intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, dtype=self.dtype, name="mlp")

  def __call__(self, hidden_states: jnp.ndarray, causal_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states, causal_mask=causal_mask)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class CLIPEncoder(nn.Module):
  """CLIP Transformer Encoder containing multiple layers."""

  num_layers: int = 12
  hidden_size: int = 768
  num_heads: int = 12
  intermediate_size: int = 3072
  dtype: Any = jnp.float32

  def setup(self):
    self.layers = [
        CLIPEncoderLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype,
            name=f"layers_{i}",
        )
        for i in range(self.num_layers)
    ]

  def __call__(self, hidden_states: jnp.ndarray, causal_mask: jnp.ndarray | None = None) -> jnp.ndarray:
    for layer in self.layers:
      hidden_states = layer(hidden_states, causal_mask=causal_mask)
    return hidden_states


class CLIPTextEmbeddings(nn.Module):
  """CLIP Token and Positional Embeddings."""

  vocab_size: int = 49408
  embed_dim: int = 768
  max_position_embeddings: int = 77
  dtype: Any = jnp.float32

  def setup(self):
    self.token_embedding = nn.Embed(self.vocab_size, self.embed_dim, dtype=self.dtype, name="token_embedding")
    self.position_embedding = nn.Embed(self.max_position_embeddings, self.embed_dim, dtype=self.dtype, name="position_embedding")

  def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    seq_len = input_ids.shape[-1]
    position_ids = jnp.arange(seq_len)[None, :]
    return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class FlaxCLIPTextModel(nn.Module):
  """Flax Linen CLIP Text Model."""

  vocab_size: int = 49408
  embed_dim: int = 768
  max_position_embeddings: int = 77
  num_layers: int = 12
  num_heads: int = 12
  intermediate_size: int = 3072
  dtype: Any = jnp.float32

  def setup(self):
    self.embeddings = CLIPTextEmbeddings(
        vocab_size=self.vocab_size,
        embed_dim=self.embed_dim,
        max_position_embeddings=self.max_position_embeddings,
        dtype=self.dtype,
        name="embeddings",
    )
    self.encoder = CLIPEncoder(
        num_layers=self.num_layers,
        hidden_size=self.embed_dim,
        num_heads=self.num_heads,
        intermediate_size=self.intermediate_size,
        dtype=self.dtype,
        name="encoder",
    )
    self.final_layer_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="final_layer_norm")

  def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    seq_len = input_ids.shape[-1]
    mask = jnp.where(jnp.tril(jnp.ones((seq_len, seq_len))), 0.0, -1e9)[None, None, :, :]

    hidden_states = self.embeddings(input_ids)
    hidden_states = self.encoder(hidden_states, causal_mask=mask)
    return self.final_layer_norm(hidden_states)
