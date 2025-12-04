
import jax.numpy as jnp
from flax import linen as nn

class GraphConvolution(nn.Module):
  """A graph convolutional layer."""
  out_features: int

  @nn.compact
  def __call__(self, node_features: jnp.ndarray, adj_matrix: jnp.ndarray) -> jnp.ndarray:
    """Forward pass for the GraphConvolution layer."""
    linear = nn.Dense(features=self.out_features, name="linear")

    adj_self_loop = adj_matrix + jnp.eye(adj_matrix.shape[0])

    d = adj_self_loop.sum(axis=1)
    d_inv_sqrt = jnp.power(d, -0.5)
    d_inv_sqrt = jnp.where(jnp.isinf(d_inv_sqrt), 0.0, d_inv_sqrt)
    d_mat_inv_sqrt = jnp.diag(d_inv_sqrt)

    normalized_adj = adj_self_loop @ d_mat_inv_sqrt.T @ d_mat_inv_sqrt

    support = linear(node_features)
    output = jnp.matmul(normalized_adj, support)
    return nn.relu(output)
