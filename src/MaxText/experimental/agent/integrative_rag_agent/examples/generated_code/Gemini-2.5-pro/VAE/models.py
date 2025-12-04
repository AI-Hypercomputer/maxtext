
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple

Array = jnp.ndarray

class VAE(nn.Module):
  """Variational Autoencoder."""
  input_dim: int
  hidden_dim: int
  latent_dim: int

  def setup(self):
    # Encoder
    self.fc1 = nn.Dense(features=self.hidden_dim, name='fc1')
    self.fc_mu = nn.Dense(features=self.latent_dim, name='fc_mu')
    self.fc_log_var = nn.Dense(features=self.latent_dim, name='fc_log_var')
    # Decoder
    self.fc3 = nn.Dense(features=self.hidden_dim, name='fc3')
    self.fc4 = nn.Dense(features=self.input_dim, name='fc4')

  def encode(self, x: Array) -> Tuple[Array, Array]:
    """Encodes the input into a latent space distribution."""
    h = nn.relu(self.fc1(x))
    return self.fc_mu(h), self.fc_log_var(h)

  def reparameterize(self, mu: Array, log_var: Array) -> Array:
    """Applies the reparameterization trick to sample from the latent distribution."""
    std = jnp.exp(0.5 * log_var)
    # The 'sampling' RNG stream must be provided when calling the module.
    eps = jax.random.normal(self.make_rng('sampling'), std.shape, dtype=std.dtype)
    return mu + eps * std

  def decode(self, z: Array) -> Array:
    """Decodes a latent vector back into the input space."""
    h = nn.relu(self.fc3(z))
    return nn.sigmoid(self.fc4(h))

  def __call__(self, x: Array) -> Tuple[Array, Array, Array]:
    """Full forward pass of the VAE."""
    mu, log_var = self.encode(jnp.reshape(x, (-1, self.input_dim)))
    z = self.reparameterize(mu, log_var)
    return self.decode(z), mu, log_var
