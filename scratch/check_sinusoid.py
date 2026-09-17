import jax
import jax.numpy as jnp
import numpy as np
from maxtext.layers.embeddings import PositionalEmbedding

# Try to match what TimestepEmbedder does
embedding_size = 256
pos_emb = PositionalEmbedding(embedding_dims=embedding_size)

t = jnp.array([980.0])
t_reshaped = t[:, jnp.newaxis]

temb = pos_emb(seq_len=1, position=t_reshaped)
temb = temb[:, 0, :]

# Flip sine and cosine embeddings to match HF flip_sin_to_cos=True
half_dim = temb.shape[-1] // 2
temb_flipped = jnp.concatenate([temb[:, half_dim:], temb[:, :half_dim]], axis=-1)

print("Temb shape:", temb_flipped.shape)
print("Temb first 10:", temb_flipped[0, :10])

# Let's also check without flip
print("Temb without flip first 10:", temb[0, :10])
