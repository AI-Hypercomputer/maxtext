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

"""VAE Decoder Layer."""

from flax import nnx
import jax.numpy as jnp
from maxtext.layers.linears import DenseGeneral

class VAEDecoder(nnx.Module):
  """VAE Decoder."""
  def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
    self.fc1 = DenseGeneral(latent_dim, hidden_dim, use_bias=True, rngs=rngs)
    self.fc2 = DenseGeneral(hidden_dim, output_dim, use_bias=True, rngs=rngs)

  def __call__(self, z):
    z = nnx.relu(self.fc1(z))
    x_hat = nnx.sigmoid(self.fc2(z))
    return x_hat
