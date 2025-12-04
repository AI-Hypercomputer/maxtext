
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array


class SelfAttention(nn.Module):
  """A simple self-attention module implemented in Flax.

  This module is a basic implementation of self-attention with query, key, and value
  projections, scaled dot-product attention, and an output projection. It is a
  JAX/Flax equivalent of a standard PyTorch self-attention layer.
  """

  embed_size: int
  heads: int

  @nn.compact
  def __call__(self, values: Array, keys: Array, query: Array, mask: Optional[Array] = None) -> Array:
    """Forward pass for the SelfAttention module.

    Args:
      values: Value vectors. Shape: (batch_size, value_len, embed_size).
      keys: Key vectors. Shape: (batch_size, key_len, embed_size).
      query: Query vectors. Shape: (batch_size, query_len, embed_size).
      mask: An optional mask to apply to the attention scores.
        Shape should be broadcastable to (batch_size, 1, 1, key_len).

    Returns:
      The output of the attention mechanism. Shape: (batch_size, query_len, embed_size).
    """
    head_dim = self.embed_size // self.heads

    assert head_dim * self.heads == self.embed_size, "embed_size needs to be divisible by heads"

    # Define layers
    values_layer = nn.Dense(features=self.embed_size, use_bias=False, name="values")
    keys_layer = nn.Dense(features=self.embed_size, use_bias=False, name="keys")
    queries_layer = nn.Dense(features=self.embed_size, use_bias=False, name="queries")
    fc_out = nn.Dense(features=self.embed_size, name="fc_out")

    # Get the batch size (N) and sequence lengths correctly
    n = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # Project the input embeddings
    values = values_layer(values)
    keys = keys_layer(keys)
    queries = queries_layer(query)

    # Split the embedding into self.heads pieces
    # Reshape: (N, Seq_Len, Heads, Head_Dim)
    values = values.reshape(n, value_len, self.heads, head_dim)
    keys = keys.reshape(n, key_len, self.heads, head_dim)
    queries = queries.reshape(n, query_len, self.heads, head_dim)

    # Transpose to bring Heads dimension forward for parallel multiplication
    # New shape: (N, Heads, Seq_Len, Head_Dim)
    values = jnp.transpose(values, axes=(0, 2, 1, 3))
    keys = jnp.transpose(keys, axes=(0, 2, 1, 3))
    queries = jnp.transpose(queries, axes=(0, 2, 1, 3))

    # Calculate Energy (Scaled Dot-Product Attention)
    # Shape: (N, Heads, Query_Len, Head_Dim) * (N, Heads, Head_Dim, Key_Len)
    # Result: (N, Heads, Query_Len, Key_Len)
    energy = jnp.matmul(queries, jnp.swapaxes(keys, -2, -1))

    # Scale the energy
    energy = energy / (head_dim**0.5)

    # Apply Mask (if provided)
    if mask is not None:
      # mask shape should broadcast to (N, 1, 1, Key_Len) or similar
      # We fill masked elements with a very low value so Softmax makes them 0
      energy = jnp.where(mask == 0, -1e20, energy)

    # Normalize energy to probabilities
    attention = jax.nn.softmax(energy, axis=3)

    # Apply attention to values
    # (N, Heads, Query_Len, Key_Len) * (N, Heads, Value_Len, Head_Dim)
    # Result: (N, Heads, Query_Len, Head_Dim)
    out = jnp.matmul(attention, values)

    # Reshape back to original dimensions
    # Swap Heads and Query_Len back: (N, Query_Len, Heads, Head_Dim)
    out = jnp.transpose(out, axes=(0, 2, 1, 3))

    # Flatten the last two dimensions: (N, Query_Len, Embed_Size)
    out = out.reshape(n, query_len, self.heads * head_dim)

    # Final linear layer
    out = fc_out(out)

    return out
