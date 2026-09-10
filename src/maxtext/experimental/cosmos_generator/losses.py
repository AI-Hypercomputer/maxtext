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

"""Loss functions and utilities for VAEs."""

import jax.numpy as jnp
import optax

def kl_divergence_standard_normal(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
  """Computes KL divergence between Q(z|x) and standard normal N(0, I).
  
  KL( Q(z|x) || N(0, I) ) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

  Args:
    mu: Mean of the approximate posterior.
    logvar: Log variance of the approximate posterior.

  Returns:
    KL divergence summed over the latent dimensions, averaged over batch.
  """
  # Sum over latent dimensions (axis=-1), then mean over batch
  return -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1))


def mse_reconstruction_loss(x_hat: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Computes MSE reconstruction loss.

  Args:
    x_hat: Reconstructed image.
    x: Original image.

  Returns:
    MSE loss summed over the image dimensions, averaged over batch.
  """
  # optax.l2_loss computes 0.5 * (x_hat - x)^2
  # Sum over pixel dimensions (axis=-1), then mean over batch
  return jnp.mean(jnp.sum(optax.l2_loss(x_hat, x), axis=-1))


def bce_reconstruction_loss_with_logits(logits: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
  """Computes Binary Cross Entropy reconstruction loss from logits.

  Useful for binary or normalized image data like MNIST.

  Args:
    logits: Unnormalized decoder outputs (logits).
    x: Target original image (values between 0 and 1).

  Returns:
    BCE loss summed over the image dimensions, averaged over batch.
  """
  return jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(logits, x), axis=-1))
