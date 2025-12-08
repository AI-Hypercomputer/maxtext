
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

# Re-used from src.MaxText.layers.linears
# from maxtext.layers import linears
from MaxText.layers import linears 


class SelfAttention(nn.Module):
  """A standard multi-head self-attention module.

  Attributes:
    embed_size: The embedding dimension of the input.
    heads: The number of attention heads.
  """

  embed_size: int
  heads: int

  def setup(self):
    """Initializes the dense layers for the attention mechanism."""
    self.head_dim = self.embed_size // self.heads

    assert (
        self.head_dim * self.heads == self.embed_size
    ), "embed_size needs to be divisible by heads"

    # Dense layers for value, key, and query projections
    self.values = linears.dense_general(
        in_features_shape=(self.embed_size,),
        out_features_shape=(self.embed_size,),
        use_bias=False,
        name="values_projection",
        matmul_precision="highest",
    )
    self.keys = linears.dense_general(
        in_features_shape=(self.embed_size,),
        out_features_shape=(self.embed_size,),
        use_bias=False,
        name="keys_projection",
        matmul_precision="highest",
    )
    self.queries = linears.dense_general(
        in_features_shape=(self.embed_size,),
        out_features_shape=(self.embed_size,),
        use_bias=False,
        name="queries_projection",
        matmul_precision="highest",
    )

    # Final output dense layer
    self.fc_out = linears.dense_general(
        in_features_shape=(self.embed_size,),
        out_features_shape=(self.embed_size,),
        use_bias=True,
        name="output_projection",
        matmul_precision="highest",
    )

  def __call__(
      self,
      values: jnp.ndarray,
      keys: jnp.ndarray,
      query: jnp.ndarray,
      mask: Optional[jnp.ndarray],
  ) -> jnp.ndarray:
    """Performs the forward pass of the self-attention mechanism.

    Args:
      values: The value tensor.
      keys: The key tensor.
      query: The query tensor.
      mask: An optional mask to apply to the attention scores.

    Returns:
      The output of the self-attention mechanism.
    """
    # Get the batch size (N) and sequence lengths
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # Project the input embeddings
    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(query)

    # Split the embedding into self.heads pieces
    # Reshape: (N, Seq_Len, Heads, Head_Dim)
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = queries.reshape(N, query_len, self.heads, self.head_dim)

    # Transpose to bring Heads dimension forward for parallel multiplication
    # New shape: (N, Heads, Seq_Len, Head_Dim)
    values = jnp.transpose(values, (0, 2, 1, 3))
    keys = jnp.transpose(keys, (0, 2, 1, 3))
    queries = jnp.transpose(queries, (0, 2, 1, 3))

    # Calculate Energy (Scaled Dot-Product Attention)
    # Shape: (N, Heads, Query_Len, Head_Dim) @ (N, Heads, Key_Len, Head_Dim).T
    # Result: (N, Heads, Query_Len, Key_Len)
    energy = jnp.matmul(queries, jnp.swapaxes(keys, -2, -1))

    # Scale the energy
    energy = energy / jnp.sqrt(self.head_dim)

    # Apply Mask (if provided)
    if mask is not None:
      # mask shape should broadcast to (N, 1, 1, Key_Len) or similar
      # We fill masked elements with a very low value so Softmax makes them 0
      energy = jnp.where(mask == 0, -1e20, energy)

    # Normalize energy to probabilities
    attention = jax.nn.softmax(energy, axis=-1)

    # Apply attention to values
    # (N, Heads, Query_Len, Key_Len) @ (N, Heads, Value_Len, Head_Dim)
    # Result: (N, Heads, Query_Len, Head_Dim)
    out = jnp.matmul(attention, values)

    # Reshape back to original dimensions
    # Swap Heads and Query_Len back: (N, Query_Len, Heads, Head_Dim)
    out = jnp.transpose(out, (0, 2, 1, 3))

    # Flatten the last two dimensions: (N, Query_Len, Embed_Size)
    out = out.reshape(N, query_len, self.embed_size)

    # Final linear layer
    out = self.fc_out(out)

    return out
