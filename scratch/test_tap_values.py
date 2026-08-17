import jax
import jax.numpy as jnp
from maxtext.models.qwen3 import jax_chunk_gated_delta_rule

batch, seq_len, num_k_heads, head_k_dim = 2, 256, 16, 128
num_v_heads, head_v_dim = 32, 128
chunk_size = 64

query = jnp.ones((batch, seq_len, num_k_heads, head_k_dim))
key = jnp.ones((batch, seq_len, num_k_heads, head_k_dim))
value = jnp.ones((batch, seq_len, num_v_heads, head_v_dim))
g = jnp.zeros((batch, seq_len, num_v_heads))
beta = jnp.ones((batch, seq_len, num_v_heads))

query = jnp.repeat(query, 2, axis=2)
key = jnp.repeat(key, 2, axis=2)

o, h, A = jax_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=chunk_size, compute_dtype=jnp.float32)

print("Sum of A:", float(jnp.sum(A)))
print("Max of A:", float(jnp.max(A)))
print("Min of A:", float(jnp.min(A)))
