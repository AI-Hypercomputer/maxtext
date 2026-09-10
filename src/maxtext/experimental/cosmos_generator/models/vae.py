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

"""Vanilla VAE implementation in Flax NNX."""

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.layers.vae_encoder import VAEEncoder
from maxtext.layers.vae_decoder import VAEDecoder

class VAE(nnx.Module):
  """Vanilla Variational Autoencoder."""
  def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20, *, rngs: nnx.Rngs):
    self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim, rngs=rngs)
    self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim, rngs=rngs)

  def reparameterize(self, mu, logvar, rng_key):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng_key, mu.shape)
    return mu + eps * std

  def __call__(self, x, rng_key):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar, rng_key)
    x_hat = self.decoder(z)
    return x_hat, mu, logvar
