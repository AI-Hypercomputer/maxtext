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

"""VAE Encoder Layer."""

from flax import nnx
import jax.numpy as jnp
from maxtext.layers.linears import DenseGeneral

class VAEEncoder(nnx.Module):
  """VAE Encoder."""
  def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, *, rngs: nnx.Rngs):
    self.fc1 = DenseGeneral(input_dim, hidden_dim, use_bias=True, rngs=rngs)
    self.fc_mu = DenseGeneral(hidden_dim, latent_dim, use_bias=True, rngs=rngs)
    self.fc_var = DenseGeneral(hidden_dim, latent_dim, use_bias=True, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.fc1(x))
    mu = self.fc_mu(x)
    logvar = self.fc_var(x)
    return mu, logvar
